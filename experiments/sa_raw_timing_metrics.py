"""Deterministic, paired raw timing summaries, with no training or promotion.

``summarize(rows, bootstrap=5000, seed=2026092603)`` returns a JSON-compatible
dict. Rows must contain all four ARMS for each observed (pair_id, replica).
There is no manifest argument, so wholly absent pairs cannot be detected.
Rows require search_end_seconds between TTF/reset (when present) and delivery,
and last_pp_failure_reason: None with zero decisions, a string otherwise.
Deadline stops must reach the search budget; PP-deadline stops require an
executed decision ending with time_limit and no feasible paths. Optional
reset_seconds is validated when present. Trace verification belongs to the
runtime, not this summary reader.

Success means first feasible env-return <= 120s; delivery success additionally
requires validated/persisted paths <= 120s. Feasible deadline overshoots remain
failures for both endpoints. Delivery times on the common-success cohort are
raw and may exceed 120s. Modeled completion uses only common *on-time delivered*
feasible pairs, and is not an observed execution time.

``arms``, ``contrasts``, ``four_arm_common_success``, and ``by_map`` carry explicit
cohort denominators. Contrasts are raw_updated minus the named baseline; lower
time/path/work values are wins. ``secondary_capped_ttf`` uses a 120s failure
penalty and is never the primary timing claim. Empty numeric summaries are
None, not zero. p50/p95 and percentile CI endpoints use linear interpolation.

Bootstrap intervals resample all maps uniformly with replacement, retaining
every pair/replica in each sampled map. Estimates remain pair-weighted. Draws
without eligible observations are excluded from the interval and counted.
Per-map reports are descriptive (no bootstrap intervals). Input order does not
affect results; inputs and the process-global random generator are unchanged.
"""

from collections import Counter, defaultdict
from collections.abc import Mapping
import math
import random

from experiments.sa_raw_confirmation import ARMS
from experiments.sa_paired_completion import require


BUDGET_SECONDS = 120.0
UPDATED = "raw_updated"
BASELINES = tuple(arm for arm in ARMS if arm != UPDATED)
TIME_METRICS = ("ttf_seconds", "delivery_seconds")
QUALITY_METRICS = ("decisions", "generated", "soc", "makespan", "wait_steps")
METRICS = TIME_METRICS + QUALITY_METRICS
STEP_SECONDS = (0.5, 1.0, 2.0)
_REQUIRED = frozenset((
    "job_id", "pair_id", "replica", "comparison_arm", "map_id",
    "initial_fingerprint", "rng_stream_id", "status", "budget_seconds",
    "feasible", "success_within_budget", "delivered_within_budget",
    "final_conflicts", "stop", "search_end_seconds", "last_pp_failure_reason",
)) | frozenset(METRICS)


def _number(value, name):
    require(type(value) in (int, float), f"{name} must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    require(finite and value >= 0, f"{name} must be finite and nonnegative")


def _validate(rows):
    paired, job_ids, pair_maps = defaultdict(dict), set(), {}
    for row in rows:
        require(isinstance(row, Mapping), "row must be a mapping")
        require(_REQUIRED <= row.keys(), f"missing fields: {sorted(_REQUIRED - row.keys())}")
        for field in ("job_id", "pair_id", "comparison_arm", "map_id",
                      "initial_fingerprint", "rng_stream_id"):
            require(isinstance(row[field], str) and bool(row[field]), f"invalid {field}")
        for field in ("replica", "final_conflicts") + QUALITY_METRICS:
            require(type(row[field]) is int and row[field] >= 0,
                    f"{field} must be a nonnegative integer")
        for field in ("feasible", "success_within_budget", "delivered_within_budget"):
            require(type(row[field]) is bool, f"{field} must be boolean")
        require(row["status"] == "ok", "incomplete/error row")
        require(row["comparison_arm"] in ARMS, "unknown comparison_arm")
        require(row["job_id"] not in job_ids, "duplicate job_id")
        job_ids.add(row["job_id"])
        _number(row["budget_seconds"], "budget_seconds")
        require(row["budget_seconds"] == BUDGET_SECONDS, "budget must be 120 seconds")
        _number(row["delivery_seconds"], "delivery_seconds")
        search_end = row["search_end_seconds"]
        _number(search_end, "search_end_seconds")
        require(search_end <= row["delivery_seconds"], "search_end_seconds exceeds delivery_seconds")
        if "reset_seconds" in row:
            _number(row["reset_seconds"], "reset_seconds")
            require(search_end >= row["reset_seconds"], "search_end_seconds precedes reset_seconds")
        ttf = row["ttf_seconds"]
        if ttf is not None:
            _number(ttf, "ttf_seconds")
            require(search_end >= ttf, "search_end_seconds precedes first feasible env-return")
        last_pp_failure = row["last_pp_failure_reason"]
        if row["decisions"] == 0:
            require(last_pp_failure is None, "last_pp_failure_reason must be None with zero decisions")
        else:
            require(isinstance(last_pp_failure, str),
                    "last_pp_failure_reason must be a string with executed decisions")
        feasible = row["feasible"]
        require(feasible == (row["final_conflicts"] == 0), "feasible/conflicts mismatch")
        require(feasible == (ttf is not None), "feasible/TTF mismatch")
        if feasible:
            require(row["delivery_seconds"] >= ttf, "delivery precedes first feasible env-return")
        success = feasible and ttf <= BUDGET_SECONDS
        delivered = feasible and row["delivery_seconds"] <= BUDGET_SECONDS
        require(row["success_within_budget"] == success, "success_within_budget mismatch")
        require(row["delivered_within_budget"] == delivered, "delivered_within_budget mismatch")
        stop = row["stop"]
        require(stop in ("feasible", "deadline", "pp_deadline"), "unknown stop")
        require(stop != "feasible" or feasible, "feasible stop without feasible paths")
        require(not success or stop == "feasible", "on-time success requires feasible stop")
        if stop == "deadline":
            require(search_end >= row["budget_seconds"], "deadline before budget")
        elif stop == "pp_deadline":
            require(row["decisions"] > 0 and last_pp_failure == "time_limit" and not feasible,
                    "pp_deadline requires an executed time_limit decision and no feasible paths")
        key = (row["pair_id"], row["replica"])
        arm = row["comparison_arm"]
        require(arm not in paired[key], "duplicate pair/replica/arm")
        require(pair_maps.setdefault(key[0], row["map_id"]) == row["map_id"], "map mismatch")
        paired[key][arm] = row
    require(paired, "empty timing rows")
    for values in paired.values():
        require(set(values) == set(ARMS), "missing arm")
        for field in ("initial_fingerprint", "rng_stream_id", "budget_seconds"):
            require(len({row[field] for row in values.values()}) == 1, f"unpaired {field}")
    return [paired[key] for key in sorted(paired)]


def _quantile(ordered, q):
    if not ordered:
        return None
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    fraction = position - lo
    return ordered[lo] * (1 - fraction) + ordered[hi] * fraction


def _stats(values):
    ordered = sorted(values)
    total = math.fsum(ordered) if ordered else None
    return dict(count=len(ordered), total=total,
                mean=total / len(ordered) if ordered else None,
                p50=_quantile(ordered, .5), p95=_quantile(ordered, .95),
                max=ordered[-1] if ordered else None)


def _pair_ids(groups):
    return [[g[UPDATED]["pair_id"], g[UPDATED]["replica"]] for g in groups]


def _interval(groups, differences, draws):
    if draws is None:
        return {}
    by_map = defaultdict(list)
    for group, difference in zip(groups, differences):
        by_map[group[UPDATED]["map_id"]].append(difference)
    totals = {m: (math.fsum(values), len(values)) for m, values in by_map.items()}
    estimates = []
    for draw in draws:
        denominator = sum(c * totals[m][1] for m, c in draw.items() if m in totals)
        if denominator:
            estimates.append(math.fsum(c * totals[m][0] for m, c in draw.items()
                                       if m in totals) / denominator)
    estimates.sort()
    return dict(ci95=[_quantile(estimates, .025), _quantile(estimates, .975)] if estimates else None,
                bootstrap_valid_draws=len(estimates),
                bootstrap_empty_draws=len(draws) - len(estimates))


def _paired_metric(groups, baseline, value, draws=None):
    old = [value(g[baseline]) for g in groups]
    new = [value(g[UPDATED]) for g in groups]
    differences = [n - b for n, b in zip(new, old)]
    result = dict(denominator=len(groups), baseline=_stats(old), raw_updated=_stats(new),
                  mean_difference=_stats(differences)["mean"],
                  wins=sum(d < 0 for d in differences), ties=sum(d == 0 for d in differences),
                  losses=sum(d > 0 for d in differences))
    result.update(_interval(groups, differences, draws))
    return result


def _success(groups, baseline, field, draws):
    gains = [g for g in groups if g[UPDATED][field] and not g[baseline][field]]
    losses = [g for g in groups if g[baseline][field] and not g[UPDATED][field]]
    differences = [int(g[UPDATED][field]) - int(g[baseline][field]) for g in groups]
    denominator = len(groups)
    new_count = sum(g[UPDATED][field] for g in groups)
    baseline_count = sum(g[baseline][field] for g in groups)
    result = dict(denominator=denominator, raw_updated_count=new_count,
                  baseline_count=baseline_count, raw_updated_rate=new_count / denominator,
                  baseline_rate=baseline_count / denominator,
                  delta=(len(gains) - len(losses)) / denominator,
                  wins=len(gains), ties=denominator - len(gains) - len(losses), losses=len(losses),
                  both_success=sum(g[UPDATED][field] and g[baseline][field] for g in groups),
                  both_failure=sum(not g[UPDATED][field] and not g[baseline][field] for g in groups),
                  gain_pairs=_pair_ids(gains), loss_pairs=_pair_ids(losses))
    result.update(_interval(groups, differences, draws))
    return result


def _metrics(groups, baseline, draws):
    return {field: _paired_metric(groups, baseline, lambda r: r[field],
                                  draws if field in TIME_METRICS else None)
            for field in METRICS}


def _cohort(groups, denominator, baseline, draws):
    return dict(count=len(groups), denominator=denominator, pair_ids=_pair_ids(groups),
                maps=len({g[UPDATED]["map_id"] for g in groups}),
                metrics=_metrics(groups, baseline, draws))


def _arm(rows):
    successful = [r for r in rows if r["success_within_budget"]]
    delivered = [r for r in rows if r["delivered_within_budget"]]
    failed = [r for r in rows if not r["success_within_budget"]]
    late = [r for r in failed if r["feasible"]]
    return dict(denominator=len(rows), feasible_count=sum(r["feasible"] for r in rows),
                success_count=len(successful), success_rate=len(successful) / len(rows),
                delivered_count=len(delivered), delivered_rate=len(delivered) / len(rows),
                successful={k: _stats([r[k] for r in successful]) for k in METRICS},
                delivered={k: _stats([r[k] for r in delivered]) for k in METRICS},
                unsuccessful_quality_diagnostic={k: _stats([r[k] for r in failed])
                                                 for k in QUALITY_METRICS + ("final_conflicts",)},
                late_feasible_times_diagnostic={k: _stats([r[k] for r in late]) for k in TIME_METRICS})


def _capped(row):
    return row["ttf_seconds"] if row["success_within_budget"] else BUDGET_SECONDS


def _completion(row, seconds_per_step):
    value = row["delivery_seconds"] + seconds_per_step * row["makespan"]
    _number(value, "modeled completion")
    return value


def _summary(groups, draws=None):
    denominator = len(groups)
    contrasts = {}
    for baseline in BASELINES:
        common = [g for g in groups if g[UPDATED]["success_within_budget"]
                  and g[baseline]["success_within_budget"]]
        delivered = [g for g in groups if g[UPDATED]["delivered_within_budget"]
                     and g[baseline]["delivered_within_budget"]]
        delivered_report = _cohort(delivered, denominator, baseline, draws)
        delivered_report["modeled_completion_seconds"] = {
            str(step): _paired_metric(delivered, baseline, lambda r: _completion(r, step), draws)
            for step in STEP_SECONDS
        }
        contrasts[baseline] = dict(
            success=_success(groups, baseline, "success_within_budget", draws),
            delivery_success=_success(groups, baseline, "delivered_within_budget", draws),
            common_success=_cohort(common, denominator, baseline, draws),
            common_delivered=delivered_report,
        )
    four = [g for g in groups if all(g[a]["success_within_budget"] for a in ARMS)]
    return dict(
        pairs=denominator, maps=len({g[UPDATED]["map_id"] for g in groups}),
        episodes=denominator * len(ARMS),
        arms={a: _arm([g[a] for g in groups]) for a in ARMS}, contrasts=contrasts,
        four_arm_common_success=dict(
            count=len(four), denominator=denominator, pair_ids=_pair_ids(four),
            maps=len({g[UPDATED]["map_id"] for g in four}),
            arms={a: {k: _stats([g[a][k] for g in four]) for k in METRICS} for a in ARMS},
            contrasts={b: _metrics(four, b, draws) for b in BASELINES}),
        secondary_capped_ttf=dict(
            label="secondary_120s_failure_penalty_not_raw_TTF_or_sole_claim",
            failure_penalty_seconds=BUDGET_SECONDS, denominator=denominator,
            arms={a: _stats([_capped(g[a]) for g in groups]) for a in ARMS},
            contrasts={b: _paired_metric(groups, b, _capped, draws) for b in BASELINES}),
    )


def summarize(rows, bootstrap=5000, seed=2026092603):
    """Validate complete observed pairs and return descriptive timing evidence.

    Raises ValueError for invalid/incomplete rows or bootstrap settings. Accepts
    an iterable of mappings. All success deltas are rate differences, not
    percentage points. Time differences and modeled completion are in seconds.
    """
    require(type(bootstrap) is int and bootstrap > 0, "bootstrap must be a positive integer")
    require(type(seed) is int, "seed must be an integer")
    groups = _validate(rows)
    by_map = defaultdict(list)
    for group in groups:
        by_map[group[UPDATED]["map_id"]].append(group)
    maps = sorted(by_map)
    rng = random.Random(seed)
    draws = [Counter(rng.choices(maps, k=len(maps))) for _ in range(bootstrap)]
    report = _summary(groups, draws)
    report.update(
        schema="lns2.sa.raw_timing_metrics.v1", budget_seconds=BUDGET_SECONDS,
        by_map={m: _summary(by_map[m]) for m in maps}, bootstrap=bootstrap, seed=seed,
        bootstrap_unit="map", bootstrap_weighting="pair_weighted_with_whole_map_resampling",
        bootstrap_empty_policy="exclude_and_count_no_eligible_pair_draws_never_impute_zero",
        difference_direction="raw_updated_minus_baseline", time_difference_units="seconds",
        metric_wins="raw_updated_lower_exact_ties", success_wins="raw_updated_only_success",
        quantile_method="linear_interpolation_at_(n-1)*q", no_training=True, no_promotion=True,
        interpretation="descriptive_paired_timing_not_a_promotion_decision",
        timing_semantics=dict(
            success="feasible_first_env_return_at_or_before_120s",
            delivery="validated_persisted_feasible_paths_at_or_before_120s",
            common_success_delivery="raw_delivery_seconds_including_delivery_overshoot",
            modeled_completion="delivery_seconds + seconds_per_step * makespan",
            modeled_cohort="both_arms_delivered_feasible_within_budget",
            modeled_caveat="modeled_not_observed_execution_completion_no_failure_imputation",
            unsuccessful_quality="diagnostic_only_excluded_from_success_quality"),
    )
    return report


__all__ = ["summarize"]
