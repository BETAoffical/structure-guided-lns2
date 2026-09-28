"""Four frozen policies on new maps; reuse the registered deferred-audit clock."""
from collections import Counter
import random

from experiments import sa_completion_independent_timing as old
from experiments import sa_shared_feature_timing as shared
from experiments.sa_raw_selection_fast import _bind

run, lean, reference, metrics, prior = old.run, old.lean, old.reference, old.metrics, old.prior
ARMS = ('official', 'official_sa', 'parent', 'completion')
LABELS = dict(old.LABELS, official=shared.LABELS['official'])
TIMING_MODE = old.TIMING_MODE
_validate = _bind(metrics._validate, ARMS=ARMS)
qualification_worker = old.qualification_worker


def identity(job):
    arm = job['comparison_arm']
    run.require(arm in ARMS, 'unknown expanded arm')
    if arm in ('official', 'official_sa'):
        run.require(job['arm'] == arm and job['model'] is None, 'official identity')
    else:
        run.require(job['arm'] == 'trained_actor' and job['model'] is not None, 'actor identity')


def preflight_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    identity(job)
    standard = job['comparison_arm'] == 'official'
    q, env, ctx = (shared.standard_environment if standard else reference.prepare_environment)(job)
    state = q._plain(env.reset(seed=job['solver_seed']))
    run.require(state['initial_solution_complete'], 'incomplete initial PP')
    fingerprint = q.state_fingerprint(state)
    run.require(fingerprint == job['expected_initial'], 'unpaired initial fingerprint')
    cls = shared.OfficialPolicy if standard else (
        reference.Policy if job['comparison_arm'] == 'official_sa' else old.SharedFeaturePolicy)
    policy = cls(job, q, ctx)
    policy.start(state)
    if not state['feasible']:
        event = policy.choose(env, state, 0)
        run.require(q.state_fingerprint(q._plain(env.get_state())) == fingerprint, 'proposal mutated state')
        if job['comparison_arm'] in ('official', 'official_sa'):
            run.require(event == dict(action=dict(mode='official')), 'official actor leakage')
        else:
            run.require(event['action']['mode'] == 'explicit_neighborhood' and event['pool'], 'invalid choice')
    return dict(status='ok', job_id=job['job_id'], initial_fingerprint=fingerprint,
                policy_sha256=policy.sha, selected=int(not state['feasible']), native_repairs=0)


def timed_worker(job):
    identity(job)
    if job['comparison_arm'] in ('official', 'official_sa'):
        return lean.WORKERS[job['comparison_arm']](job)
    return old._actor_worker(job)


def audit_worker(job):
    identity(job)
    return lean.audit_worker(job)


def summarize(rows, bootstrap=5000, seed=2026092806):
    rows = list(rows)
    for row in rows:
        run.require(row.get('timing_mode') == TIMING_MODE, 'mixed timing semantics')
        for key in ('selection_seconds', 'native_pp_seconds', 'bookkeeping_seconds', 'reset_seconds'):
            metrics._number(row[key], key)
    groups = _validate(rows)
    maps = sorted({g['parent']['map_id'] for g in groups})
    rng = random.Random(seed)
    draws = [Counter(rng.choices(maps, k=len(maps))) for _ in range(bootstrap)]

    def summary(gs, ds):
        contrasts = {}
        for target, baseline in (('completion', 'parent'), ('parent', 'official'),
                ('parent', 'official_sa'), ('completion', 'official'),
                ('completion', 'official_sa'), ('official_sa', 'official')):
            contrast = prior.contrast([dict(parent=g[baseline], completion=g[target]) for g in gs], ds)
            contrast.update(target=target, baseline=baseline,
                            statistics_keys=dict(parent=baseline, completion=target))
            contrasts[target+'_vs_'+baseline] = contrast
        common = [g for g in gs if all(g[a]['success_within_budget'] for a in ARMS)]
        curves = {a: {str(t): sum(g[a]['success_within_budget'] and g[a]['ttf_seconds'] <= t for g in gs)
                      for t in (10, 30, 60, 120)} for a in ARMS}
        return dict(pairs=len(gs), arms={a: metrics._arm([g[a] for g in gs]) for a in ARMS},
                    contrasts=contrasts, solved_by_seconds=curves,
                    four_arm_common_success=dict(count=len(common),
                        arms={a: {k: metrics._stats([g[a][k] for g in common])
                                  for k in ('ttf_seconds', 'soc', 'makespan')} for a in ARMS}))

    result = summary(groups, draws)
    return result | dict(by_map={m: summary([g for g in groups if g['parent']['map_id'] == m], []) for m in maps},
        bootstrap=bootstrap, bootstrap_unit='map', runtime_labels=LABELS,
        no_training=True, no_promotion=True, failure_time_imputation=False,
        new_same_family_maps=True, cross_layout_claim=False, timing_mode=TIMING_MODE,
        scientific_audit_outside_ttf=True, modeled_not_robot_execution=True)
