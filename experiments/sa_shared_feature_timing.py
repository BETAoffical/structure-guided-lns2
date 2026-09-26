"""Same frozen raw actor, existing fast versus shared features, with Dual16+SA."""
from collections import Counter
import random
from types import SimpleNamespace

from experiments import sa_raw_fast_timing as old
from experiments import sa_raw_timed_runtime as reference
from experiments.sa_raw_selection_fast import FastPolicy, _bind
from experiments.sa_shared_features import SharedFeaturePolicy
from scripts import run_sa_onpolicy as run

ARMS = ('raw_reference', 'raw_fast', 'dual16_sa', 'official_sa', 'official')
LABELS = {'raw_reference': 'raw-1 + existing FastPolicy + SA',
          'raw_fast': 'raw-1 + SharedFeaturePolicy + SA', 'dual16_sa': 'Dual16 + SA',
          'official_sa': 'Official Adaptive + SA', 'official': 'Official Adaptive, standard PP acceptance'}


class OfficialPolicy(reference.Policy):
    def __init__(self, job, q, ctx):
        run.require(job['arm'] == 'official' and job['model'] is None, 'not standard official')
        super().__init__(dict(job, arm='official_sa'), q, ctx)
        self.job, self.sha = job, 'official'


class StandardRuntime:
    """Select the official transition validator without changing shared globals."""
    def __init__(self, native):
        self.native = native

    def __getattr__(self, name):
        return getattr(self.native, name)

    def validate_transition(self, before, after, metrics, selected, arm, temp, draw):
        run.require('experimental_acceptance' not in metrics, 'SA leaked into official baseline')
        return self.native.validate_transition(before, after, metrics, selected, 'standard', temp, draw)


def standard_environment(job):
    q, env, ctx = reference.prepare_environment(job)
    return StandardRuntime(q), env, ctx


def standard_transition(job, q, env, state, event, d, seconds):
    run.require(event['action'] == {'mode': 'official'}, 'official action altered')
    # These Python-only values are recorded for schema compatibility, never used by native PP.
    temp = q.temperature(d)
    uniform = run.stream_draw(job['plan'], job['phase'], job['pair_id'], job['replica'], d, 'accept')
    raw = q._plain(env.step_with_time_limit(event['action'], seconds))
    run.require('experimental_acceptance' not in raw['metrics'], 'official called experimental PP')
    return raw['observation'], raw['metrics'], temp, uniform


def standard_native(plan):
    return StandardRuntime(run.native_runtime(plan))


WORKERS = {'raw_reference': _bind(reference.timed_worker, Policy=FastPolicy),
           'raw_fast': _bind(reference.timed_worker, Policy=SharedFeaturePolicy),
           'dual16_sa': reference.timed_worker, 'official_sa': reference.timed_worker,
           'official': _bind(reference.timed_worker, Policy=OfficialPolicy,
                             prepare_environment=standard_environment, transition=standard_transition)}
PREFLIGHTS = {'raw_reference': _bind(reference.preflight_worker, Policy=FastPolicy),
              'raw_fast': _bind(reference.preflight_worker, Policy=SharedFeaturePolicy),
              'dual16_sa': reference.preflight_worker, 'official_sa': reference.preflight_worker}
_standard_audit = _bind(reference.audit_worker, Policy=OfficialPolicy,
                       run=SimpleNamespace(**(vars(run) | {'native_runtime': standard_native})))
pair_worker = old.pair_worker


def official_preflight(job):
    """Prove time-bounded standard steps equal direct official steps on each input."""
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    results = []
    for bounded in (False, True):
        q, env, _ = standard_environment(job)
        state = q._plain(env.reset(seed=job['solver_seed']))
        run.require(q.state_fingerprint(state) == job['expected_initial'], 'official initial differs')
        steps = []
        for _ in range(job['preflight_steps']):
            if state['feasible']:
                break
            action = {'mode': 'official'}
            raw = q._plain(env.step_with_time_limit(action, 20.) if bounded else env.step(action))
            after, metrics = raw['observation'], raw['metrics']
            run.require(metrics['pp_failure_reason'] != 'time_limit', 'unknown official prefix: time limit')
            q.validate_transition(state, after, metrics, metrics['neighborhood'], 'standard', 0., 0.)
            steps.append(dict(fingerprint=q.state_fingerprint(after),
                              **{k: metrics[k] for k in ('neighborhood', 'repair_order', 'pp_failure_reason',
                                                         'replan_success', 'applied_heuristic')}))
            state = after
        results.append(steps)
    run.require(results[0] == results[1], 'official time-limit wrapper changed paths or RNG')
    return dict(status='ok', job_id=job['job_id'], prefix_steps=len(results[0]),
                action_feature_path_equal=True, standard_direct_steps_equal=True,
                native_steps_executed=2*len(results[0]))


def timed_worker(job):
    run.require(job['comparison_arm'] in ARMS, 'unknown shared timing arm')
    return WORKERS[job['comparison_arm']](job)


def preflight_worker(job):
    run.require(job['comparison_arm'] in ARMS, 'unknown shared timing arm')
    return official_preflight(job) if job['comparison_arm'] == 'official' else PREFLIGHTS[job['comparison_arm']](job)


def audit_worker(job):
    return _standard_audit(job) if job['comparison_arm'] == 'official' else reference.audit_worker(job)


def summarize(rows, bootstrap, seed):
    rows = list(rows)
    validator = _bind(old._validate, ARMS=ARMS)
    summarizer = _bind(old.summarize, ARMS=ARMS, _validate=validator)
    result = summarizer(rows, bootstrap, seed)
    groups = validator(rows)
    maps = sorted({g['raw_fast']['map_id'] for g in groups})
    rng = random.Random(seed)
    draws = [Counter(rng.choices(maps, k=len(maps))) for _ in range(bootstrap)]
    sa_contrast = old._contrast([dict(g, raw_fast=g['official_sa']) for g in groups], 'official', draws)
    sa_contrast.update(target='official_sa', baseline='official', target_statistics_key='raw_fast')
    result.update(runtime_labels=LABELS, same_raw_model=True,
                  reference_is_existing_fast_policy=True, shared_features_only=True,
                  official_sa_vs_official=sa_contrast, repeat_is_not_independent_map=True)
    return result
