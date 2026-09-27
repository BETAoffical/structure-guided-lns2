"""New-map confirmation of two frozen actors and the same-SA official baseline."""
from collections import Counter
import random

from experiments import sa_lean_timing as lean
from experiments import sa_raw_timed_runtime as reference
from experiments import sa_raw_timing_metrics as metrics
from experiments import sa_terminal_efficiency_timing as prior
from experiments.sa_raw_selection_fast import _bind
from experiments.sa_shared_features import SharedFeaturePolicy
from scripts import run_sa_onpolicy as run

ARMS = ('parent', 'completion', 'official_sa')
LABELS = dict(prior.LABELS, official_sa='Official Adaptive + same SA')
TIMING_MODE = lean.TIMING_MODE
_actor_worker = _bind(lean.lean_worker, Policy=SharedFeaturePolicy)
_validate = _bind(metrics._validate, ARMS=ARMS)


def identity(job):
    arm = job['comparison_arm']
    run.require(arm in ARMS, 'unknown independent arm')
    if arm == 'official_sa':
        run.require(job['arm'] == arm and job['model'] is None, 'official selector identity')
    else:
        run.require(job['arm'] == 'trained_actor' and job['model'] is not None, 'actor identity')


def qualification_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    die_with_parent(job['parent_pid'])
    q, env, _ = reference.prepare_environment(job)
    state = q._plain(env.reset(seed=job['solver_seed']))
    run.require(state['initial_solution_complete'], 'incomplete initial PP; no replacement')
    validate_final(state)
    return dict(status='ok', job_id=job['job_id'], pair_id=job['pair_id'],
                map_id=job['case']['map_id'], initial_fingerprint=q.state_fingerprint(state),
                initial_conflicts=state['num_of_colliding_pairs'], feasible=state['feasible'],
                agents=len(state['agents']), generated=state['low_level']['generated'])


def preflight_worker(job):
    """Check new input admission, not rerun old engineering equivalence tests."""
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    identity(job)
    q, env, ctx = reference.prepare_environment(job)
    state = q._plain(env.reset(seed=job['solver_seed']))
    run.require(state['initial_solution_complete'], 'preflight PP incomplete')
    fp = q.state_fingerprint(state)
    run.require(fp == job['expected_initial'], 'unpaired initial fingerprint')
    policy = (reference.Policy if job['comparison_arm'] == 'official_sa' else SharedFeaturePolicy)(job, q, ctx)
    policy.start(state)
    count = 0
    if not state['feasible']:
        event = policy.choose(env, state, 0)
        run.require(q.state_fingerprint(q._plain(env.get_state())) == fp, 'proposal changed input state')
        if job['comparison_arm'] == 'official_sa':
            run.require(event == dict(action=dict(mode='official')), 'official generates actor features')
        else:
            run.require(event['action']['mode'] == 'explicit_neighborhood' and event['pool'], 'invalid actor choice')
        count = 1
    return dict(status='ok', job_id=job['job_id'], initial_fingerprint=fp,
                selected=count, policy_sha256=policy.sha, native_repairs=0)


def timed_worker(job):
    identity(job)
    return (lean.lean_worker if job['comparison_arm'] == 'official_sa' else _actor_worker)(job)


def audit_worker(job):
    identity(job)
    return lean.audit_worker(job)


def summarize(rows, bootstrap=5000, seed=2026092803):
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
        for target, baseline in (('completion', 'parent'), ('completion', 'official_sa'), ('parent', 'official_sa')):
            paired = [dict(parent=g[baseline], completion=g[target]) for g in gs]
            contrast = prior.contrast(paired, ds)
            contrast.update(target=target, baseline=baseline,
                            statistics_keys=dict(parent=baseline, completion=target))
            contrasts[target+'_vs_'+baseline] = contrast
        return dict(pairs=len(gs), arms={a: metrics._arm([g[a] for g in gs]) for a in ARMS},
                    contrasts=contrasts)

    result = summary(groups, draws)
    result.update(by_map={m: summary([g for g in groups if g['parent']['map_id'] == m], []) for m in maps},
                  bootstrap=bootstrap, bootstrap_unit='map', runtime_labels=LABELS,
                  no_training=True, no_promotion=True, failure_time_imputation=False,
                  new_same_family_maps=True, cross_layout_claim=False,
                  timing_mode=TIMING_MODE, scientific_audit_outside_ttf=True,
                  modeled_not_robot_execution=True)
    return result
