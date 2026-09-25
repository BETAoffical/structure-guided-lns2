"""Opt-in timer adapter and three-arm summaries; frozen timer is unchanged."""
from collections import Counter, defaultdict
import random

from experiments import sa_raw_timed_runtime as reference
from experiments import sa_raw_timing_metrics as metrics
from experiments.sa_raw_selection_fast import FastPolicy, _bind
from scripts import run_sa_onpolicy as run

ARMS = ('raw_reference', 'raw_fast', 'dual16_sa')
_fast_worker = _bind(reference.timed_worker, Policy=FastPolicy)
_fast_preflight = _bind(reference.preflight_worker, Policy=FastPolicy)
_validate = _bind(metrics._validate, ARMS=ARMS)


def timed_worker(job):
    run.require(job['comparison_arm'] in ARMS, 'unknown runtime arm')
    worker = _fast_worker if job['comparison_arm'] == 'raw_fast' else reference.timed_worker
    return worker(job)


def preflight_worker(job):
    run.require(job['comparison_arm'] in ARMS, 'unknown runtime arm')
    worker = _fast_preflight if job['comparison_arm'] == 'raw_fast' else reference.preflight_worker
    return worker(job)


def pair_worker(job):
    """Compare actual old/fast trajectories until either PP hits its deadline."""
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    q = run.native_runtime(job['plan'])
    folders = [run.ROOT/job['output']/'episodes'/j for j in job['runtime_jobs']]
    states = [run.read_json(f/'initial.json') for f in folders]
    run.require(q.state_fingerprint(states[0]) == q.state_fingerprint(states[1]), 'unpaired initial')
    count, deadline_branch = 0, False
    for old, fast in zip(*(run.trace_read(f) for f in folders)):
        for key in ('decision', 'before', 'action', 'pool', 'proposal_order', 'anchor_id',
                    'selected_id', 'selection_draw', 'features', 'probabilities',
                    'candidate_ids', 'behavior_log_probability', 'temperature', 'uniform'):
            run.require(old[key] == fast[key], 'runtime pair diverged: '+key)
        after = [q.apply_state_delta(s, e['delta']) for s, e in zip(states, (old, fast))]
        count += 1
        if any(e['metrics']['pp_failure_reason'] == 'time_limit' for e in (old, fast)):
            deadline_branch = True
            break
        run.require(q.state_fingerprint(after[0]) == q.state_fingerprint(after[1]), 'non-deadline path drift')
        states = after
    results = [run.check_seal(run.read_json(f/'result.json')) for f in folders]
    if not deadline_branch and all(r['feasible'] for r in results):
        run.require(results[0]['decisions'] == results[1]['decisions'] and
                    results[0]['final_fingerprint'] == results[1]['final_fingerprint'], 'feasible trajectory drift')
    return dict(status='ok', job_id=job['job_id'], common_decisions=count,
                deadline_branch=deadline_branch, action_equal=True,
                result_hashes=[run.sha256_file(f/'result.json') for f in folders])


def _contrast(groups, baseline, draws):
    def compare(eligible, value):
        old = [value(g[baseline]) for g in eligible]
        new = [value(g['raw_fast']) for g in eligible]
        diffs = [n-o for n, o in zip(new, old)]
        result = dict(count=len(diffs), baseline=metrics._stats(old), raw_fast=metrics._stats(new),
                      difference=metrics._stats(diffs), wins=sum(d < 0 for d in diffs),
                      ties=sum(d == 0 for d in diffs), losses=sum(d > 0 for d in diffs))
        totals = defaultdict(list)
        for g, d in zip(eligible, diffs):
            totals[g['raw_fast']['map_id']].append(d)
        estimates = []
        for draw in draws:
            sample = [v for m, c in draw.items() for _ in range(c) for v in totals[m]]
            if sample:
                estimates.append(sum(sample)/len(sample))
        ordered = sorted(estimates)
        result.update(ci95=[metrics._quantile(ordered, .025), metrics._quantile(ordered, .975)],
                      bootstrap_empty_draws=len(draws)-len(ordered))
        return result
    common = [g for g in groups if all(g[a]['success_within_budget'] for a in ('raw_fast', baseline))]
    delivered = [g for g in groups if all(g[a]['delivered_within_budget'] for a in ('raw_fast', baseline))]
    success = compare(groups, lambda r: int(r['success_within_budget']))
    success['wins'], success['losses'] = success['losses'], success['wins']
    return dict(
        success=success,
        common_pair_ids=[[g['raw_fast']['pair_id'], g['raw_fast']['replica']] for g in common],
        common_success={k: compare(common, lambda r, k=k: r[k]) for k in metrics.METRICS +
            ('selection_seconds', 'native_pp_seconds', 'bookkeeping_seconds')},
        common_delivered_count=len(delivered),
        modeled_completion={str(s): compare(delivered, lambda r: r['delivery_seconds']+s*r['makespan'])
                            for s in metrics.STEP_SECONDS})


def summarize(rows, bootstrap=5000, seed=2026092507):
    run.require(type(bootstrap) is int and bootstrap > 0, 'invalid bootstrap')
    rows = list(rows)
    for row in rows:
        for name in ('selection_seconds', 'native_pp_seconds', 'bookkeeping_seconds'):
            metrics._number(row[name], name)
    groups = _validate(rows)
    maps = sorted({g['raw_fast']['map_id'] for g in groups})
    rng = random.Random(seed)
    draws = [Counter(rng.choices(maps, k=len(maps))) for _ in range(bootstrap)]
    def summary(gs, ds):
        return dict(pairs=len(gs), arms={a: metrics._arm([g[a] for g in gs]) for a in ARMS},
                    contrasts={a: _contrast(gs, a, ds) for a in ARMS if a != 'raw_fast'})
    result = summary(groups, draws)
    result.update(by_map={m: summary([g for g in groups if g['raw_fast']['map_id'] == m], []) for m in maps},
                  bootstrap=bootstrap, bootstrap_unit='map', difference_direction='raw_fast_minus_baseline',
                  no_training=True, no_promotion=True, modeled_not_robot_execution=True,
                  reused_development_cases=True, not_independent_generalization=True)
    return result
