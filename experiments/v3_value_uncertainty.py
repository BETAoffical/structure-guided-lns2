from __future__ import annotations

import collections
import csv
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.repair_collection import _write_json
from experiments.v3_value_pilot import _atomic_write_csv


V3_VALUE_UNCERTAINTY_SCHEMA = "lns2.v3.value_uncertainty_audit.v1"
DEFAULT_THRESHOLDS = (0.0, 0.05, 0.10)
PRIMARY_THRESHOLD = 0.05


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_merged_rollouts(source: str | Path) -> list[dict[str, Any]]:
    source_path = Path(source).resolve()
    path = (
        source_path / "merged_value_rollouts.csv"
        if source_path.is_dir()
        else source_path
    )
    rows = []
    with path.open(encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            rows.append(
                {
                    **raw,
                    "state_id": str(raw["state_id"]),
                    "arm_id": str(raw["arm_id"]),
                    "arm_aliases": tuple(
                        value
                        for value in str(raw.get("arm_aliases", "")).split(",")
                        if value
                    ),
                    "trial_index": int(raw["trial_index"]),
                    "agent_count": int(raw["agent_count"]),
                    "initial_conflicts": int(raw["initial_conflicts"]),
                    "final_conflicts": int(raw["final_conflicts"]),
                    "repair_iterations": int(raw["repair_iterations"]),
                    "feasible": _as_bool(raw["feasible"]),
                    "censored": _as_bool(raw["censored"]),
                    "observed_total_seconds": float(
                        raw["observed_total_seconds"]
                    ),
                    "normalized_conflict_auc_seconds": float(
                        raw["normalized_conflict_auc_seconds"]
                    ),
                }
            )
    if not rows:
        raise ValueError("uncertainty audit source contains no rollouts")
    keys = [
        (row["state_id"], row["arm_id"], row["trial_index"]) for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("uncertainty audit source contains duplicate keys")
    return rows


def _final_conflict_ratio(row: dict[str, Any]) -> float:
    return float(row["final_conflicts"]) / max(
        1.0, float(row["initial_conflicts"])
    )


def outcome_key(row: dict[str, Any]) -> tuple[float, ...]:
    if bool(row["feasible"]):
        return (
            0.0,
            float(row["observed_total_seconds"]),
            float(row["normalized_conflict_auc_seconds"]),
        )
    return (
        1.0,
        _final_conflict_ratio(row),
        float(row["normalized_conflict_auc_seconds"]),
        float(row["observed_total_seconds"]),
    )


def compare_outcomes(
    selected: dict[str, Any], baseline: dict[str, Any]
) -> int:
    if str(selected["arm_id"]) == str(baseline["arm_id"]):
        return 0
    selected_key = outcome_key(selected)
    baseline_key = outcome_key(baseline)
    return -1 if selected_key < baseline_key else 1 if selected_key > baseline_key else 0


def aggregate_arm(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise ValueError("cannot aggregate an empty arm")
    return {
        "state_id": str(values[0]["state_id"]),
        "arm_id": str(values[0]["arm_id"]),
        "trial_count": len(values),
        "feasible_rate": statistics.fmean(
            float(bool(row["feasible"])) for row in values
        ),
        "mean_final_conflict_ratio": statistics.fmean(
            _final_conflict_ratio(row) for row in values
        ),
        "mean_normalized_auc_seconds": statistics.fmean(
            float(row["normalized_conflict_auc_seconds"]) for row in values
        ),
        "mean_total_seconds": statistics.fmean(
            float(row["observed_total_seconds"]) for row in values
        ),
        "std_normalized_auc_seconds": (
            statistics.stdev(
                float(row["normalized_conflict_auc_seconds"]) for row in values
            )
            if len(values) >= 2
            else 0.0
        ),
        "std_total_seconds": (
            statistics.stdev(
                float(row["observed_total_seconds"]) for row in values
            )
            if len(values) >= 2
            else 0.0
        ),
    }


def aggregate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["feasible_rate"]),
        float(row["mean_final_conflict_ratio"]),
        float(row["mean_normalized_auc_seconds"]),
        float(row["mean_total_seconds"]),
        str(row["arm_id"]),
    )


def _relative_improvement(baseline: float, candidate: float) -> float:
    if candidate >= baseline:
        return 0.0
    return (baseline - candidate) / max(abs(baseline), 1e-12)


def select_expected_arm(
    aggregates: Iterable[dict[str, Any]],
    *,
    v2_arm_id: str,
    improvement_threshold: float,
) -> tuple[str, str]:
    values = list(aggregates)
    lookup = {str(row["arm_id"]): row for row in values}
    if v2_arm_id not in lookup:
        raise ValueError("v2 arm is missing from aggregate candidates")
    baseline = lookup[v2_arm_id]
    candidate = min(values, key=aggregate_key)
    if str(candidate["arm_id"]) == v2_arm_id:
        return v2_arm_id, "v2_is_aggregate_winner"
    feasible_delta = float(candidate["feasible_rate"]) - float(
        baseline["feasible_rate"]
    )
    if feasible_delta > 1e-12:
        return str(candidate["arm_id"]), "higher_feasible_rate"
    if feasible_delta < -1e-12:
        return v2_arm_id, "feasible_rate_guard"
    final_delta = float(baseline["mean_final_conflict_ratio"]) - float(
        candidate["mean_final_conflict_ratio"]
    )
    if final_delta > 1e-12:
        improvement = _relative_improvement(
            float(baseline["mean_final_conflict_ratio"]),
            float(candidate["mean_final_conflict_ratio"]),
        )
        return (
            (str(candidate["arm_id"]), "final_conflict_improvement")
            if improvement + 1e-12 >= float(improvement_threshold)
            else (v2_arm_id, "final_conflict_threshold_guard")
        )
    auc_improvement = _relative_improvement(
        float(baseline["mean_normalized_auc_seconds"]),
        float(candidate["mean_normalized_auc_seconds"]),
    )
    return (
        (str(candidate["arm_id"]), "auc_improvement")
        if auc_improvement + 1e-12 >= float(improvement_threshold)
        else (v2_arm_id, "auc_threshold_guard")
    )


def _alias_arm(
    rows: Iterable[dict[str, Any]], alias: str, *, state_id: str
) -> str:
    matches = {
        str(row["arm_id"]) for row in rows if alias in row["arm_aliases"]
    }
    if len(matches) != 1:
        raise ValueError(
            f"{state_id} must contain exactly one arm aliased as {alias}"
        )
    return next(iter(matches))


def _target_state_rows(
    rows: Iterable[dict[str, Any]], *, required_trials: int
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    result = {}
    for state_id, values in grouped.items():
        arm_trials: dict[str, set[int]] = collections.defaultdict(set)
        for row in values:
            arm_trials[str(row["arm_id"])].add(int(row["trial_index"]))
        expected = set(range(int(required_trials)))
        if arm_trials and all(trials == expected for trials in arm_trials.values()):
            result[state_id] = values
    if not result:
        raise ValueError("no states have complete required trial coverage")
    return result


def leave_one_seed_out(
    rows: Iterable[dict[str, Any]],
    *,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    required_trials: int = 4,
) -> list[dict[str, Any]]:
    states = _target_state_rows(rows, required_trials=int(required_trials))
    result = []
    for state_id, state_rows in sorted(states.items()):
        v2_arm = _alias_arm(state_rows, "v2_full", state_id=state_id)
        model_arm = _alias_arm(state_rows, "model_s3", state_id=state_id)
        arm_ids = sorted({str(row["arm_id"]) for row in state_rows})
        row_lookup = {
            (str(row["arm_id"]), int(row["trial_index"])): row
            for row in state_rows
        }
        for heldout in range(int(required_trials)):
            aggregates = [
                aggregate_arm(
                    row_lookup[(arm_id, trial)]
                    for trial in range(int(required_trials))
                    if trial != heldout
                )
                for arm_id in arm_ids
            ]
            for threshold in map(float, thresholds):
                selected_arm, reason = select_expected_arm(
                    aggregates,
                    v2_arm_id=v2_arm,
                    improvement_threshold=threshold,
                )
                selected = row_lookup[(selected_arm, heldout)]
                baseline = row_lookup[(v2_arm, heldout)]
                comparison = compare_outcomes(selected, baseline)
                result.append(
                    {
                        "policy_id": f"loo_expected_t{threshold:.2f}",
                        "improvement_threshold": threshold,
                        "state_id": state_id,
                        "map_id": str(selected["map_id"]),
                        "layout_mode": str(selected["layout_mode"]),
                        "agent_count": int(selected["agent_count"]),
                        "heldout_trial": heldout,
                        "v2_arm_id": v2_arm,
                        "model_s3_arm_id": model_arm,
                        "selected_arm_id": selected_arm,
                        "selection_reason": reason,
                        "deviated_from_v2": selected_arm != v2_arm,
                        "outcome": (
                            "win"
                            if comparison < 0
                            else "loss"
                            if comparison > 0
                            else "tie"
                        ),
                        "outcome_score": -comparison,
                        "selected_feasible": bool(selected["feasible"]),
                        "v2_feasible": bool(baseline["feasible"]),
                        "selected_total_seconds": float(
                            selected["observed_total_seconds"]
                        ),
                        "v2_total_seconds": float(
                            baseline["observed_total_seconds"]
                        ),
                        "selected_final_conflict_ratio": _final_conflict_ratio(
                            selected
                        ),
                        "v2_final_conflict_ratio": _final_conflict_ratio(
                            baseline
                        ),
                        "selected_normalized_auc_seconds": float(
                            selected["normalized_conflict_auc_seconds"]
                        ),
                        "v2_normalized_auc_seconds": float(
                            baseline["normalized_conflict_auc_seconds"]
                        ),
                    }
                )
            model = row_lookup[(model_arm, heldout)]
            baseline = row_lookup[(v2_arm, heldout)]
            comparison = compare_outcomes(model, baseline)
            result.append(
                {
                    "policy_id": "model_s3_reference",
                    "improvement_threshold": "",
                    "state_id": state_id,
                    "map_id": str(model["map_id"]),
                    "layout_mode": str(model["layout_mode"]),
                    "agent_count": int(model["agent_count"]),
                    "heldout_trial": heldout,
                    "v2_arm_id": v2_arm,
                    "model_s3_arm_id": model_arm,
                    "selected_arm_id": model_arm,
                    "selection_reason": "recorded_model_s3_arm",
                    "deviated_from_v2": model_arm != v2_arm,
                    "outcome": (
                        "win"
                        if comparison < 0
                        else "loss"
                        if comparison > 0
                        else "tie"
                    ),
                    "outcome_score": -comparison,
                    "selected_feasible": bool(model["feasible"]),
                    "v2_feasible": bool(baseline["feasible"]),
                    "selected_total_seconds": float(
                        model["observed_total_seconds"]
                    ),
                    "v2_total_seconds": float(
                        baseline["observed_total_seconds"]
                    ),
                    "selected_final_conflict_ratio": _final_conflict_ratio(
                        model
                    ),
                    "v2_final_conflict_ratio": _final_conflict_ratio(baseline),
                    "selected_normalized_auc_seconds": float(
                        model["normalized_conflict_auc_seconds"]
                    ),
                    "v2_normalized_auc_seconds": float(
                        baseline["normalized_conflict_auc_seconds"]
                    ),
                }
            )
    return result


def cluster_bootstrap_mean_interval(
    rows: Iterable[dict[str, Any]],
    *,
    value_key: str,
    cluster_key: str = "state_id",
    samples: int = 5000,
    seed: int = 20260725,
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_key])].append(float(row[value_key]))
    cluster_ids = sorted(grouped)
    if not cluster_ids:
        raise ValueError("cannot bootstrap empty rows")
    rng = random.Random(int(seed))
    estimates = []
    for _ in range(int(samples)):
        selected = [rng.choice(cluster_ids) for _ in cluster_ids]
        values = [
            value for cluster_id in selected for value in grouped[cluster_id]
        ]
        estimates.append(statistics.fmean(values))
    estimates.sort()
    lower_index = max(0, math.floor(0.025 * (len(estimates) - 1)))
    upper_index = min(
        len(estimates) - 1,
        math.ceil(0.975 * (len(estimates) - 1)),
    )
    return estimates[lower_index], estimates[upper_index]


def summarize_policies(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["policy_id"])].append(row)
    result = []
    for policy_id, values in sorted(grouped.items()):
        deviations = [row for row in values if bool(row["deviated_from_v2"])]
        wins = sum(row["outcome"] == "win" for row in values)
        losses = sum(row["outcome"] == "loss" for row in values)
        ties = len(values) - wins - losses
        lower, upper = cluster_bootstrap_mean_interval(
            values,
            value_key="outcome_score",
            cluster_key="map_id",
        )
        common_feasible = [
            row
            for row in values
            if bool(row["selected_feasible"]) and bool(row["v2_feasible"])
        ]
        result.append(
            {
                "policy_id": policy_id,
                "paired_count": len(values),
                "state_count": len({str(row["state_id"]) for row in values}),
                "deviation_count": len(deviations),
                "deviation_fraction": len(deviations) / len(values),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "win_rate_among_deviations": (
                    sum(row["outcome"] == "win" for row in deviations)
                    / len(deviations)
                    if deviations
                    else 0.0
                ),
                "mean_outcome_score": statistics.fmean(
                    float(row["outcome_score"]) for row in values
                ),
                "outcome_score_ci95_lower": lower,
                "outcome_score_ci95_upper": upper,
                "selected_feasible_rate": statistics.fmean(
                    float(bool(row["selected_feasible"])) for row in values
                ),
                "v2_feasible_rate": statistics.fmean(
                    float(bool(row["v2_feasible"])) for row in values
                ),
                "feasible_rate_delta": statistics.fmean(
                    float(bool(row["selected_feasible"]))
                    - float(bool(row["v2_feasible"]))
                    for row in values
                ),
                "common_feasible_count": len(common_feasible),
                "common_feasible_selected_mean_seconds": (
                    statistics.fmean(
                        float(row["selected_total_seconds"])
                        for row in common_feasible
                    )
                    if common_feasible
                    else 0.0
                ),
                "common_feasible_v2_mean_seconds": (
                    statistics.fmean(
                        float(row["v2_total_seconds"])
                        for row in common_feasible
                    )
                    if common_feasible
                    else 0.0
                ),
                "common_feasible_mean_time_delta_seconds": (
                    statistics.fmean(
                        float(row["selected_total_seconds"])
                        - float(row["v2_total_seconds"])
                        for row in common_feasible
                    )
                    if common_feasible
                    else 0.0
                ),
                "mean_final_conflict_ratio_delta": statistics.fmean(
                    float(row["selected_final_conflict_ratio"])
                    - float(row["v2_final_conflict_ratio"])
                    for row in values
                ),
                "mean_normalized_auc_delta_seconds": statistics.fmean(
                    float(row["selected_normalized_auc_seconds"])
                    - float(row["v2_normalized_auc_seconds"])
                    for row in values
                ),
            }
        )
    return result


def subgroup_summary(
    rows: Iterable[dict[str, Any]], *, group_key: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row[group_key])].append(row)
    result = []
    for group, values in sorted(grouped.items()):
        wins = sum(row["outcome"] == "win" for row in values)
        losses = sum(row["outcome"] == "loss" for row in values)
        ties = len(values) - wins - losses
        common_feasible = [
            row
            for row in values
            if bool(row["selected_feasible"]) and bool(row["v2_feasible"])
        ]
        result.append(
            {
                group_key: group,
                "state_count": len({str(row["state_id"]) for row in values}),
                "paired_count": len(values),
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "net_wins": wins - losses,
                "feasible_count_delta": sum(
                    bool(row["selected_feasible"]) for row in values
                )
                - sum(bool(row["v2_feasible"]) for row in values),
                "common_feasible_count": len(common_feasible),
                "common_feasible_mean_time_delta_seconds": (
                    statistics.fmean(
                        float(row["selected_total_seconds"])
                        - float(row["v2_total_seconds"])
                        for row in common_feasible
                    )
                    if common_feasible
                    else 0.0
                ),
            }
        )
    return result


def _arm_distribution_rows(
    rows: Iterable[dict[str, Any]], *, required_trials: int
) -> list[dict[str, Any]]:
    states = _target_state_rows(rows, required_trials=int(required_trials))
    result = []
    for state_id, state_rows in sorted(states.items()):
        grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in state_rows:
            grouped[str(row["arm_id"])].append(row)
        v2_arm = _alias_arm(state_rows, "v2_full", state_id=state_id)
        model_arm = _alias_arm(state_rows, "model_s3", state_id=state_id)
        for arm_id, values in sorted(grouped.items()):
            aggregate = aggregate_arm(values)
            result.append(
                {
                    **aggregate,
                    "is_v2_arm": arm_id == v2_arm,
                    "is_model_s3_arm": arm_id == model_arm,
                    "aliases": ",".join(values[0]["arm_aliases"]),
                    "censored_count": sum(
                        bool(row["censored"]) for row in values
                    ),
                }
            )
    return result


def _state_uncertainty_rows(
    predictions: Iterable[dict[str, Any]], *, primary_policy_id: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in predictions:
        if str(row["policy_id"]) == primary_policy_id:
            grouped[str(row["state_id"])].append(row)
    result = []
    for state_id, values in sorted(grouped.items()):
        counts = collections.Counter(
            str(row["selected_arm_id"]) for row in values
        )
        probabilities = [count / len(values) for count in counts.values()]
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        maximum_entropy = math.log(max(1, len(counts)))
        result.append(
            {
                "state_id": state_id,
                "map_id": str(values[0]["map_id"]),
                "layout_mode": str(values[0]["layout_mode"]),
                "agent_count": int(values[0]["agent_count"]),
                "fold_count": len(values),
                "selected_arm_count": len(counts),
                "selection_purity": max(counts.values()) / len(values),
                "normalized_selection_entropy": (
                    entropy / maximum_entropy if maximum_entropy else 0.0
                ),
                "selection_counts": ",".join(
                    f"{arm_id}:{counts[arm_id]}" for arm_id in sorted(counts)
                ),
                "wins": sum(row["outcome"] == "win" for row in values),
                "losses": sum(row["outcome"] == "loss" for row in values),
                "ties": sum(row["outcome"] == "tie" for row in values),
            }
        )
    return result


def _report_markdown(report: dict[str, Any]) -> str:
    primary = dict(report["primary_policy"])
    method = dict(report["method"])
    return "\n".join(
        [
            "# V3 value uncertainty audit",
            "",
            f"Decision: `{report['decision']}`",
            "",
            (
                f"- Complete four-seed states: {int(report['state_count'])}; "
                f"independent maps: {int(report['unique_map_count'])}."
            ),
            (
                f"- Primary policy: `{primary['policy_id']}`; deviations: "
                f"{int(primary['deviation_count'])}/{int(primary['paired_count'])}."
            ),
            (
                f"- Held-out wins/losses/ties versus v2: "
                f"{int(primary['wins'])}/{int(primary['losses'])}/"
                f"{int(primary['ties'])}."
            ),
            (
                "- Mean paired outcome and map-clustered 95% interval: "
                f"{float(primary['mean_outcome_score']):.3f} "
                f"[{float(primary['outcome_score_ci95_lower']):.3f}, "
                f"{float(primary['outcome_score_ci95_upper']):.3f}]."
            ),
            (
                "- Feasible-rate delta versus v2: "
                f"{float(primary['feasible_rate_delta']):+.3%}."
            ),
            (
                "- Maps with positive/nonnegative net wins: "
                f"{int(report['beneficial_map_count'])}/"
                f"{int(report['nonnegative_map_count'])} of "
                f"{int(report['unique_map_count'])}."
            ),
            "",
            "## Method",
            "",
            f"- Training-fold aggregate order: `{method['aggregate_order']}`.",
            f"- Held-out outcome order: `{method['heldout_outcome_order']}`.",
            f"- Primary guard: `{method['primary_guard']}`.",
            f"- Uncertainty interval: `{method['uncertainty_interval']}`.",
            "",
            "## Checks",
            "",
            *[
                f"- {name}: `{str(bool(value)).lower()}`"
                for name, value in sorted(report["checks"].items())
            ],
            "",
            "## Boundary",
            "",
            *[f"- {value}" for value in report["limitations"]],
            "",
        ]
    )


def audit_value_uncertainty(
    *,
    source: str | Path,
    output: str | Path,
    required_trials: int = 4,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    primary_threshold: float = PRIMARY_THRESHOLD,
) -> dict[str, Any]:
    source_path = Path(source).resolve()
    source_csv = (
        source_path / "merged_value_rollouts.csv"
        if source_path.is_dir()
        else source_path
    )
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("uncertainty audit output is non-empty")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = load_merged_rollouts(source_csv)
    predictions = leave_one_seed_out(
        rows,
        thresholds=tuple(map(float, thresholds)),
        required_trials=int(required_trials),
    )
    summaries = summarize_policies(predictions)
    primary_policy_id = f"loo_expected_t{float(primary_threshold):.2f}"
    summary_lookup = {str(row["policy_id"]): row for row in summaries}
    if primary_policy_id not in summary_lookup:
        raise ValueError("primary threshold is absent from threshold grid")
    primary = dict(summary_lookup[primary_policy_id])
    primary_predictions = [
        row
        for row in predictions
        if str(row["policy_id"]) == primary_policy_id
    ]
    map_rows = subgroup_summary(primary_predictions, group_key="map_id")
    agent_rows = subgroup_summary(
        primary_predictions, group_key="agent_count"
    )
    nonnegative_map_count = sum(
        int(row["net_wins"]) >= 0 for row in map_rows
    )
    beneficial_map_count = sum(int(row["net_wins"]) > 0 for row in map_rows)
    checks = {
        "at_least_eight_complete_states": int(primary["state_count"]) >= 8,
        "deviation_fraction_at_least_10pct": (
            float(primary["deviation_fraction"]) >= 0.10
        ),
        "win_rate_among_deviations_at_least_60pct": (
            float(primary["win_rate_among_deviations"]) >= 0.60
        ),
        "wins_exceed_losses": int(primary["wins"]) > int(primary["losses"]),
        "nonnegative_net_outcome_on_at_least_3_of_4_maps": (
            nonnegative_map_count >= math.ceil(0.75 * len(map_rows))
        ),
        "feasible_rate_not_lower_than_v2": (
            float(primary["feasible_rate_delta"]) >= -1e-12
        ),
        "map_clustered_ci_lower_not_below_minus_0_10": (
            float(primary["outcome_score_ci95_lower"]) >= -0.10
        ),
    }
    decision = (
        "distributional_value_signal_promising"
        if all(checks.values())
        else "distributional_value_signal_insufficient"
    )
    target_rows = _target_state_rows(rows, required_trials=int(required_trials))
    report = {
        "schema": V3_VALUE_UNCERTAINTY_SCHEMA,
        "decision": decision,
        "state_count": len(target_rows),
        "unique_map_count": len(
            {
                str(values[0]["map_id"])
                for values in target_rows.values()
            }
        ),
        "paired_fold_count": int(primary["paired_count"]),
        "required_trials": int(required_trials),
        "thresholds": list(map(float, thresholds)),
        "primary_threshold": float(primary_threshold),
        "primary_policy": primary,
        "policy_summaries": summaries,
        "map_summaries": map_rows,
        "agent_summaries": agent_rows,
        "beneficial_map_count": beneficial_map_count,
        "nonnegative_map_count": nonnegative_map_count,
        "checks": checks,
        "method": {
            "aggregate_order": (
                "maximize feasible_rate; minimize mean_final_conflict_ratio, "
                "mean_normalized_conflict_auc_seconds, mean_total_seconds"
            ),
            "heldout_outcome_order": (
                "feasible beats censored; among feasible minimize total_seconds; "
                "among censored minimize final_conflict_ratio; then minimize "
                "normalized_conflict_auc_seconds and total_seconds"
            ),
            "primary_guard": (
                "change v2 only for higher feasible_rate or at least 5% "
                "improvement in the first differing aggregate quality metric"
            ),
            "uncertainty_interval": (
                "5000-sample bootstrap resampling four map clusters"
            ),
        },
        "run_config": {
            "source": source_csv.name,
            "source_sha256": sha256_file(source_csv),
            "implementation_sha256": sha256_file(Path(__file__).resolve()),
        },
        "limitations": [
            "The audit uses only eight four-seed states from four maps; trials are paired outcomes, not independent maps.",
            "Three training seeds select an action for the held-out fourth seed, so this tests stochastic target stability without using state features.",
            "Continuation actions are official Adaptive and results are not complete v2 or v3 episodes.",
            "Retrospective Oracle-derived arms remain in the candidate set, making this an optimistic upper-bound diagnostic rather than deployment evidence.",
            "The 5% guard is fixed as the primary diagnostic; 0% and 10% are sensitivity checks and are not selected post hoc.",
        ],
    }
    arm_rows = _arm_distribution_rows(
        rows, required_trials=int(required_trials)
    )
    state_rows = _state_uncertainty_rows(
        predictions,
        primary_policy_id=primary_policy_id,
    )
    _atomic_write_csv(output_root / "loo_predictions.csv", predictions)
    _atomic_write_csv(output_root / "policy_summary.csv", summaries)
    _atomic_write_csv(output_root / "map_summary.csv", map_rows)
    _atomic_write_csv(output_root / "agent_summary.csv", agent_rows)
    _atomic_write_csv(output_root / "arm_distribution.csv", arm_rows)
    _atomic_write_csv(output_root / "state_uncertainty.csv", state_rows)
    _write_json(output_root / "value_uncertainty_report.json", report)
    (output_root / "value_uncertainty_report.md").write_text(
        _report_markdown(report),
        encoding="utf-8",
    )
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_VALUE_UNCERTAINTY_SCHEMA,
            "status": "complete",
            "error_count": 0,
            "decision": decision,
        },
    )
    return report
