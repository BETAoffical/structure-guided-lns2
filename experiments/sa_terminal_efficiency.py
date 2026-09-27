"""Complete-episode objectives, not one-step cost or remaining-round labels."""
from collections import defaultdict
import math
import random

from experiments.sa_onpolicy_contract import gradient_coefficients as shared_credit
from experiments.sa_uncapped_training_contract import terminal_return
from experiments.sa_paired_completion import require

BONUS = .1
OBJECTIVES = ('completion', 'completion_work')


def episode_return(row, *, objective, max_decisions, node_budget):
    require(objective in OBJECTIVES, 'unknown terminal objective')
    success = terminal_return(row, max_decisions=max_decisions, node_budget=node_budget)
    if success is None or objective == 'completion':
        return success
    # Only the total work up to actual feasibility earns a bounded bonus.
    return success * (1. + BONUS * (1. - min(row['generated'] / node_budget, 1.)))


def coefficients(rows, *, objective, policy_sha256, expected_groups, replicas, node_budget):
    def validate(row, **kwargs):
        return episode_return(row, objective=objective, **kwargs)
    return shared_credit(rows, policy_sha256=policy_sha256, expected_groups=expected_groups,
                         replicas=replicas, max_decisions=None, node_budget=node_budget,
                         terminal_validator=validate)


def contrast(rows, baseline, challenger, *, bootstrap=5000, seed=2026092701):
    groups = defaultdict(dict)
    for row in rows:
        arm = row['comparison_arm']
        if arm not in (baseline, challenger):
            continue
        key = (row['pair_id'], row['replica'])
        require(arm not in groups[key], 'duplicate paired result')
        require(row['status'] == 'ok', 'unknown outcome is not a failure label')
        groups[key][arm] = row
    require(groups, 'empty comparison')
    paired = []
    for key, group in sorted(groups.items()):
        require(set(group) == {baseline, challenger}, 'incomplete paired arms')
        a, b = group[baseline], group[challenger]
        require(all(a[k] == b[k] for k in ('initial_fingerprint', 'rng_stream_id', 'map_id')), 'unpaired state/stream')
        paired.append((a, b))

    def summarize(pairs):
        common = [(a, b) for a, b in pairs if a['success'] and b['success']]
        def metric(name):
            if not common:
                return dict(count=0, baseline=None, challenger=None, change_percent=None)
            a = math.fsum(a[name] for a, b in common) / len(common)
            b = math.fsum(b[name] for a, b in common) / len(common)
            return dict(count=len(common), baseline=a, challenger=b,
                        change_percent=100 * (b / a - 1) if a else None,
                        wins=sum(b[name] < a[name] for a, b in common),
                        losses=sum(b[name] > a[name] for a, b in common))
        return dict(pairs=len(pairs), baseline_success=sum(a['success'] for a, b in pairs),
                    challenger_success=sum(b['success'] for a, b in pairs),
                    gains=sum(b['success'] and not a['success'] for a, b in pairs),
                    losses=sum(a['success'] and not b['success'] for a, b in pairs),
                    common_success=len(common),
                    metrics={k:metric(k) for k in ('generated', 'decisions', 'soc', 'makespan')})

    maps = sorted({a['map_id'] for a, b in paired})
    by_map = {m:summarize([(a, b) for a, b in paired if a['map_id'] == m]) for m in maps}
    rng = random.Random(seed)
    success_diffs, work_diffs = [], []
    for _ in range(bootstrap):
        sample = [rng.choice(maps) for _ in maps]
        pp = [p for m in sample for p in paired if p[0]['map_id'] == m]
        success_diffs.append(math.fsum(b['success']-a['success'] for a, b in pp) / len(pp))
        common = [(a, b) for a, b in pp if a['success'] and b['success']]
        if common:
            work_diffs.append(math.fsum(b['generated']-a['generated'] for a, b in common) / len(common))
    def interval(values):
        values = sorted(values)
        return [values[int((len(values)-1)*q)] for q in (.025, .975)] if values else None
    return dict(baseline=baseline, challenger=challenger, **summarize(paired), by_map=by_map,
                success_difference_ci95=interval(success_diffs),
                common_generated_difference_ci95=interval(work_diffs),
                bootstrap_unit='map', bootstrap=bootstrap, no_ttf=True)
