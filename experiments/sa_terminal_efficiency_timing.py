"""Frozen completion update versus parent using the existing lean timer."""
from collections import Counter, defaultdict
import random

from experiments import sa_lean_timing as lean
from experiments import sa_raw_timed_runtime as reference
from experiments import sa_raw_timing_metrics as metrics
from experiments.sa_raw_selection_fast import _bind
from experiments.sa_shared_features import SharedFeaturePolicy
from scripts import run_sa_onpolicy as run

ARMS = ('parent', 'completion')
LABELS = {'parent': 'raw-1 + shared features + SA',
          'completion': 'completion-only update A + shared features + SA'}
TIMING_MODE = lean.TIMING_MODE
_timed = _bind(lean.lean_worker, Policy=SharedFeaturePolicy)
_preflight = _bind(reference.preflight_worker, Policy=SharedFeaturePolicy)
_validate = _bind(metrics._validate, ARMS=ARMS)


def timed_worker(job):
    run.require(job['comparison_arm'] in ARMS, 'unknown terminal timing arm')
    return _timed(job)


def preflight_worker(job):
    run.require(job['comparison_arm'] in ARMS, 'unknown terminal timing arm')
    return _preflight(job)


def audit_worker(job):
    run.require(job['comparison_arm'] in ARMS, 'unknown terminal timing arm')
    return lean.audit_worker(job)


def contrast(groups, draws):
    def paired(eligible, value, *, higher_better=False):
        old = [value(g['parent']) for g in eligible]
        new = [value(g['completion']) for g in eligible]
        differences = [n-b for n, b in zip(new, old)]
        by_map = defaultdict(list)
        for g, d in zip(eligible, differences):
            by_map[g['completion']['map_id']].append(d)
        totals = {m: (sum(v), len(v)) for m, v in by_map.items()}
        estimates = []
        for draw in draws:
            count = sum(n*totals[m][1] for m, n in draw.items() if m in totals)
            if count:
                estimates.append(sum(n*totals[m][0] for m, n in draw.items() if m in totals)/count)
        estimates.sort()
        return dict(count=len(eligible), parent=metrics._stats(old), completion=metrics._stats(new),
                    difference=metrics._stats(differences),
                    wins=sum(d > 0 if higher_better else d < 0 for d in differences),
                    losses=sum(d < 0 if higher_better else d > 0 for d in differences),
                    ties=sum(d == 0 for d in differences),
                    ci95=[metrics._quantile(estimates, .025), metrics._quantile(estimates, .975)],
                    bootstrap_empty_draws=len(draws)-len(estimates))

    common = [g for g in groups if all(g[a]['success_within_budget'] for a in ARMS)]
    delivered = [g for g in groups if all(g[a]['delivered_within_budget'] for a in ARMS)]
    def keys(gs):
        return [[g['parent']['pair_id'], g['parent']['replica']] for g in gs]
    return dict(
        success=paired(groups, lambda r: int(r['success_within_budget']), higher_better=True),
        gain_pairs=keys([g for g in groups if g['completion']['success_within_budget'] and
                         not g['parent']['success_within_budget']]),
        loss_pairs=keys([g for g in groups if g['parent']['success_within_budget'] and
                         not g['completion']['success_within_budget']]),
        common_pair_ids=keys(common),
        common_success={k: paired(common, lambda r, k=k: r[k]) for k in metrics.METRICS +
                        ('selection_seconds', 'native_pp_seconds', 'bookkeeping_seconds', 'reset_seconds')},
        common_delivered_count=len(delivered),
        modeled_completion={str(s): paired(delivered, lambda r: r['delivery_seconds']+s*r['makespan'])
                            for s in metrics.STEP_SECONDS})


def summarize(rows, bootstrap=5000, seed=2026092702):
    rows = list(rows)
    run.require(type(bootstrap) is int and bootstrap > 0, 'invalid bootstrap')
    for row in rows:
        run.require(row.get('timing_mode') == TIMING_MODE, 'mixed timing semantics')
        for name in ('selection_seconds', 'native_pp_seconds', 'bookkeeping_seconds', 'reset_seconds'):
            metrics._number(row[name], name)
    groups = _validate(rows)
    maps = sorted({g['parent']['map_id'] for g in groups})
    rng = random.Random(seed)
    draws = [Counter(rng.choices(maps, k=len(maps))) for _ in range(bootstrap)]
    def summary(gs, ds):
        return dict(pairs=len(gs), arms={a: metrics._arm([g[a] for g in gs]) for a in ARMS},
                    contrast=contrast(gs, ds))
    result = summary(groups, draws)
    result.update(by_map={m: summary([g for g in groups if g['parent']['map_id'] == m], []) for m in maps},
                  bootstrap=bootstrap, bootstrap_unit='map', difference_direction='completion_minus_parent',
                  runtime_labels=LABELS, no_training=True, no_promotion=True, failure_time_imputation=False,
                  not_independent_generalization=True, timing_mode=TIMING_MODE,
                  engineering_comparison_reused_not_rerun=True, scientific_audit_outside_ttf=True,
                  modeled_not_robot_execution=True)
    return result
