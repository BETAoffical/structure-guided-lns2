"""Frozen completion actor: change only complete-PP acceptance, not its input clock."""
from collections import Counter
import random
from types import SimpleNamespace

from experiments import sa_lean_timing as lean
from experiments import sa_raw_timed_runtime as reference
from experiments import sa_raw_timing_metrics as metrics
from experiments import sa_terminal_efficiency_timing as prior
from experiments.sa_raw_selection_fast import _bind
from experiments.sa_shared_features import SharedFeaturePolicy
from scripts import run_sa_onpolicy as run

ARMS = ('completion_sa', 'completion_greedy')
MODES = dict(completion_sa='annealed', completion_greedy='complete_greedy')
LABELS = dict(completion_sa='Frozen A + SA, complete PP',
              completion_greedy='Frozen A + non-worsening acceptance, complete PP')
MODEL_SHA = '81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062'
TIMING_MODE = lean.TIMING_MODE


def identity(job):
    arm = job['comparison_arm']
    run.require(arm in ARMS and job['acceptance_mode'] == MODES[arm], 'acceptance identity')
    run.require(job['arm'] == 'trained_actor' and job['model']['sha256'] == MODEL_SHA, 'frozen A identity')
    run.require(job.get('max_decisions') is None and job.get('execution_node_budget') is None,
                'unexpected work cap')


def transition(job, q, env, state, event, d, seconds):
    # The actor still receives the frozen virtual temperature schedule in both arms.
    # COMPLETE_GREEDY ignores temperature only in the native acceptance predicate.
    temp = q.temperature(d)
    uniform = run.stream_draw(job['plan'], job['phase'], job['pair_id'], job['replica'], d, 'accept')
    raw = q._plain(env.step_experimental_pp(event['action'], seconds, job['acceptance_mode'], temp, uniform))
    return raw['observation'], raw['metrics'], temp, uniform


class AcceptanceRuntime:
    def __init__(self, native, mode):
        run.require(mode in MODES.values(), 'unknown acceptance')
        self.native, self.mode = native, mode

    def __getattr__(self, name):
        return getattr(self.native, name)

    def validate_transition(self, before, after, metrics, selected, arm, temp, draw):
        run.require(arm == 'annealed', 'unexpected reference audit request')
        return self.native.validate_transition(before, after, metrics, selected, self.mode, temp, draw)


_timed = _bind(lean.lean_worker, Policy=SharedFeaturePolicy, transition=transition)


def timed_worker(job):
    identity(job)
    return _timed(job)


def preflight_worker(job):
    """One-step correctness only; compare the same attempted PP against the saved SA step."""
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    identity(job)
    folder = run.ROOT/job['source_folder']
    for name, digest in job['source_files'].items():
        run.require(run.sha256_file(folder/name) == digest, 'changed source trace')
    q, env, ctx = reference.prepare_environment(job)
    before = q._plain(env.reset(seed=job['solver_seed']))
    fp = q.state_fingerprint(before)
    run.require(before['initial_solution_complete'] and fp == job['expected_initial'], 'preflight initial')
    policy = SharedFeaturePolicy(job, q, ctx)
    policy.start(before)
    reader = run.trace_read(folder)
    original = next(reader, None)
    reader.close()
    if before['feasible']:
        run.require(original is None, 'source repaired an initially feasible task')
        return dict(status='ok', job_id=job['job_id'], native_repairs=0, initial_fingerprint=fp)
    run.require(original is not None, 'missing source first action')
    event = policy.choose(env, before, 0)
    run.require(q.state_fingerprint(q._plain(env.get_state())) == fp, 'proposal changed state')
    for key in ('action', 'pool', 'proposal_order', 'anchor_id', 'selected_id', 'selection_draw',
                'features', 'probabilities', 'candidate_ids', 'behavior_log_probability'):
        run.require(event[key] == original[key], 'frozen initial decision changed: '+key)
    after, m, temp, uniform = transition(job, q, env, before, event, 0, 30.)
    run.require(m['pp_failure_reason'] != 'time_limit', 'unknown preflight: PP deadline')
    q.validate_transition(before, after, m, event['action']['agents'], job['acceptance_mode'], temp, uniform)
    run.require(temp == original['temperature'] and uniform == original['uniform'], 'changed random stream')
    for key in ('neighborhood', 'repair_order', 'pp_attempted_agent_count', 'pp_inserted_agent_count',
                'pp_attempt_conflict_pair_count', 'pp_old_conflict_pair_count', 'acceptance_evaluated'):
        run.require(m[key] == original['metrics'][key], 'different PP attempt: '+key)
    saved = q.apply_state_delta(before, original['delta'])
    run.require(after['low_level'] == saved['low_level'], 'different PP search work')
    if job['acceptance_mode'] == 'annealed' or m['replan_success'] == original['metrics']['replan_success']:
        run.require(q.state_fingerprint(after) == q.state_fingerprint(saved), 'different accepted paths')
    return dict(status='ok', job_id=job['job_id'], native_repairs=1, initial_fingerprint=fp,
                same_initial_features_action_stream_and_pp_work=True, acceptance_mode=job['acceptance_mode'])


def audit_worker(job):
    identity(job)
    row = run.check_seal(run.read_json(run.ROOT/job['output']/'episodes'/job['job_id']/'result.json'))
    run.require(row['timing_mode'] == TIMING_MODE and row['trace_seconds'] == 0 and
                row['post_delivery_trace_seconds'] >= 0, 'scientific audit charged to TTF')
    native_loader = lambda plan: AcceptanceRuntime(run.native_runtime(plan), job['acceptance_mode'])
    audit = _bind(reference.audit_worker,
                  run=SimpleNamespace(**(vars(run) | {'native_runtime': native_loader})))
    return dict(audit(job), acceptance_mode=job['acceptance_mode'])


_validate = _bind(metrics._validate, ARMS=ARMS)


def summarize(rows, bootstrap=5000, seed=2026092902):
    rows = list(rows)
    run.require(bootstrap > 0, 'bootstrap count')
    run.require(all(r.get('timing_mode') == TIMING_MODE for r in rows), 'mixed clocks')
    groups = _validate(rows)
    run.require(len({r['policy_sha256'] for r in rows}) == 1, 'mixed models')
    maps = sorted({g[ARMS[0]]['map_id'] for g in groups})
    rng = random.Random(seed)
    draws = [Counter(rng.choices(maps, k=len(maps))) for _ in range(bootstrap)]

    def summary(gs, ds):
        # Preserve the established contrast math: completion is the SA target here.
        c = prior.contrast([dict(parent=g['completion_greedy'], completion=g['completion_sa']) for g in gs], ds)
        c.update(target='completion_sa', baseline='completion_greedy',
                 statistics_keys=dict(parent='completion_greedy', completion='completion_sa'))
        return dict(pairs=len(gs), arms={a: metrics._arm([g[a] for g in gs]) for a in ARMS}, contrast=c)

    return summary(groups, draws) | dict(
        by_map={m: summary([g for g in groups if g[ARMS[0]]['map_id'] == m], []) for m in maps},
        bootstrap=bootstrap, bootstrap_unit='map', runtime_labels=LABELS, timing_mode=TIMING_MODE,
        frozen_model=True, acceptance_only=True, no_training=True, automatic_promotion=False,
        actor_temperature_input_unchanged=True, complete_pp_both_arms=True,
        failure_time_imputation=False, reused_maps_development_only=True,
        scientific_audit_outside_ttf=True, modeled_not_robot_execution=True)
