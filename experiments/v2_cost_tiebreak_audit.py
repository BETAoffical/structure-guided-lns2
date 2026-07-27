from __future__ import annotations

import collections
import itertools
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.feature_schema_v3 import V3_FEATURE_NAMES
from experiments.repair_aware_training import _balanced_map_folds
from experiments.repair_collection import _read_jsonl, _write_json
from experiments.v3_training import (
    _fit,
    _index_predictions,
    _predict,
    _rows,
    _states,
    _with_runtime_predictions,
)


V2_COST_TIEBREAK_AUDIT_SCHEMA = "lns2.v2_cost_tiebreak_audit.v1"
TOP_K = 3
UNCERTAINTY_QUANTILE = 0.80
EFFECTIVE_TOLERANCE_GRID = (0.0, 0.02, 0.05)
NO_PROGRESS_TOLERANCE_GRID = (0.0, 0.02, 0.05)
QUALITY_RETENTION_GRID = (0.98, 1.0)
MINIMUM_TIME_IMPROVEMENT_GRID = (0.0, 0.05, 0.10)
MINIMUM_UTILITY_IMPROVEMENT_GRID = (0.0, 0.05, 0.10)


def _quantile(values: Iterable[float], fraction: float) -> float:
    ordered = sorted(map(float, values))
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = float(fraction) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _model_arms(state: dict[str, Any]) -> list[dict[str, Any]]:
    arms = [dict(arm) for arm in state["arms"] if str(arm["route"]) == "model"]
    if len(arms) < TOP_K:
        raise ValueError(f"state has fewer than {TOP_K} model candidates")
    return arms


def _v2_order(arms: list[dict[str, Any]]) -> list[int]:
    order = sorted(
        range(len(arms)),
        key=lambda index: (
            -float(arms[index]["v2_score"]),
            str(arms[index]["candidate_id"]),
        ),
    )
    bases = [index for index, arm in enumerate(arms) if bool(arm["base_selected"])]
    if len(bases) != 1:
        raise ValueError("state lacks exactly one v2 candidate")
    if order[0] != bases[0]:
        raise ValueError("saved v2 candidate is not first by saved v2 score")
    return order


def _uncertainty_calibration(states: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    all_arms = []
    for state in states:
        for arm in _model_arms(state):
            row = {
                "predicted_reduction": float(
                    arm["predicted"]["conflict_reduction"]
                ),
                "actual_reduction": float(arm["actual"]["conflict_reduction"]),
                "predicted_seconds": float(arm["predicted"]["repair_seconds"]),
                "actual_seconds": float(arm["actual"]["repair_seconds"]),
            }
            grouped[int(state["agent_count"])].append(row)
            all_arms.append(row)

    def margins(values: list[dict[str, Any]]) -> dict[str, float]:
        return {
            "reduction_overprediction": max(
                0.0,
                _quantile(
                    (
                        row["predicted_reduction"] - row["actual_reduction"]
                        for row in values
                    ),
                    UNCERTAINTY_QUANTILE,
                ),
            ),
            "seconds_underprediction": max(
                0.0,
                _quantile(
                    (
                        row["actual_seconds"] - row["predicted_seconds"]
                        for row in values
                    ),
                    UNCERTAINTY_QUANTILE,
                ),
            ),
            "candidate_count": len(values),
        }

    return {
        "quantile": UNCERTAINTY_QUANTILE,
        "global": margins(all_arms),
        "by_agent_count": {
            str(agent_count): margins(values)
            for agent_count, values in sorted(grouped.items())
        },
    }


def _margins(calibration: dict[str, Any], agent_count: int) -> dict[str, float]:
    raw = dict(calibration["by_agent_count"].get(str(agent_count), calibration["global"]))
    return {
        "reduction_overprediction": float(raw["reduction_overprediction"]),
        "seconds_underprediction": float(raw["seconds_underprediction"]),
    }


def _conservative_predictions(
    arm: dict[str, Any], margins: dict[str, float]
) -> dict[str, float]:
    predicted = dict(arm["predicted"])
    reduction = max(0.0, float(predicted["conflict_reduction"]))
    seconds = max(1e-9, float(predicted["repair_seconds"]))
    lower_reduction = max(
        0.0, reduction - float(margins["reduction_overprediction"])
    )
    upper_seconds = seconds + float(margins["seconds_underprediction"])
    return {
        "effective_progress_probability": float(
            predicted["effective_progress_probability"]
        ),
        "no_progress_probability": float(predicted["no_progress_probability"]),
        "conflict_reduction": reduction,
        "repair_seconds": seconds,
        "lower_conflict_reduction": lower_reduction,
        "upper_repair_seconds": upper_seconds,
        "lower_utility": lower_reduction / max(1e-9, upper_seconds),
    }


def select_cost_tiebreak(
    state: dict[str, Any],
    thresholds: dict[str, float],
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    arms = _model_arms(state)
    order = _v2_order(arms)
    base_index = order[0]
    top_indices = order[:TOP_K]
    margins = _margins(calibration, int(state["agent_count"]))
    conservative = {
        index: _conservative_predictions(arms[index], margins)
        for index in top_indices
    }
    base = conservative[base_index]
    eligible = []
    for index in top_indices:
        prediction = conservative[index]
        if (
            prediction["effective_progress_probability"]
            < base["effective_progress_probability"]
            - float(thresholds["effective_probability_tolerance"])
        ):
            continue
        if (
            prediction["no_progress_probability"]
            > base["no_progress_probability"]
            + float(thresholds["no_progress_probability_tolerance"])
        ):
            continue
        retention = float(thresholds["conflict_reduction_retention"])
        if prediction["conflict_reduction"] + 1e-12 < (
            retention * base["conflict_reduction"]
        ):
            continue
        if base["lower_conflict_reduction"] > 0.0 and (
            prediction["lower_conflict_reduction"] + 1e-12
            < retention * base["lower_conflict_reduction"]
        ):
            continue
        time_improvement = float(thresholds["minimum_time_improvement"])
        if prediction["upper_repair_seconds"] > (
            (1.0 - time_improvement) * base["upper_repair_seconds"] + 1e-12
        ):
            continue
        utility_improvement = float(thresholds["minimum_utility_improvement"])
        if prediction["lower_utility"] + 1e-12 < (
            (1.0 + utility_improvement) * base["lower_utility"]
        ):
            continue
        eligible.append(index)
    selected_index = max(
        eligible or [base_index],
        key=lambda index: (
            conservative[index]["lower_utility"],
            conservative[index]["lower_conflict_reduction"],
            -conservative[index]["upper_repair_seconds"],
            float(arms[index]["v2_score"]),
            str(arms[index]["candidate_id"]),
        ),
    )
    return arms[selected_index], {
        "base_candidate_id": str(arms[base_index]["candidate_id"]),
        "selected_candidate_id": str(arms[selected_index]["candidate_id"]),
        "override": selected_index != base_index,
        "selected_v2_rank": order.index(selected_index) + 1,
        "eligible_candidate_count": len(eligible),
        "base_prediction": base,
        "selected_prediction": conservative[selected_index],
    }


def _oracle_top3(state: dict[str, Any]) -> dict[str, Any]:
    arms = _model_arms(state)
    order = _v2_order(arms)
    base = arms[order[0]]
    candidates = []
    for index in order[:TOP_K]:
        arm = arms[index]
        actual = dict(arm["actual"])
        if float(actual["conflict_reduction"]) + 1e-12 < (
            0.98 * float(base["actual"]["conflict_reduction"])
        ):
            continue
        if float(actual["effective_rate"]) + 1e-12 < (
            float(base["actual"]["effective_rate"]) - 0.01
        ):
            continue
        if float(actual["no_progress_rate"]) > (
            float(base["actual"]["no_progress_rate"]) + 0.01 + 1e-12
        ):
            continue
        candidates.append(arm)
    return max(
        candidates or [base],
        key=lambda arm: (
            float(arm["actual"]["conflict_reduction"])
            / max(1e-9, float(arm["actual"]["repair_seconds"])),
            float(arm["actual"]["conflict_reduction"]),
            -float(arm["actual"]["repair_seconds"]),
            float(arm["v2_score"]),
            str(arm["candidate_id"]),
        ),
    )


def _metrics(selected: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [dict(arm["actual"]) for arm in selected]
    reduction_sum = math.fsum(float(row["conflict_reduction"]) for row in actual)
    seconds_sum = math.fsum(float(row["repair_seconds"]) for row in actual)
    return {
        "state_count": len(selected),
        "effective_rate": statistics.fmean(
            float(row["effective_rate"]) for row in actual
        ),
        "no_progress_rate": statistics.fmean(
            float(row["no_progress_rate"]) for row in actual
        ),
        "hard_failure_rate": statistics.fmean(
            float(row["hard_failure_rate"]) for row in actual
        ),
        "mean_conflict_reduction": statistics.fmean(
            float(row["conflict_reduction"]) for row in actual
        ),
        "mean_repair_seconds": statistics.fmean(
            float(row["repair_seconds"]) for row in actual
        ),
        "conflict_reduction_per_repair_second": reduction_sum
        / max(1e-9, seconds_sum),
        "size_fractions": {
            str(size): statistics.fmean(
                float(int(arm["actual_size"]) == size) for arm in selected
            )
            for size in (4, 8, 16)
        },
    }


def _comparison(selected: dict[str, Any], base: dict[str, Any]) -> dict[str, float]:
    return {
        "effective_rate_delta": float(selected["effective_rate"])
        - float(base["effective_rate"]),
        "no_progress_rate_delta": float(selected["no_progress_rate"])
        - float(base["no_progress_rate"]),
        "conflict_reduction_ratio": float(selected["mean_conflict_reduction"])
        / max(1e-9, float(base["mean_conflict_reduction"])),
        "repair_time_ratio": float(selected["mean_repair_seconds"])
        / max(1e-9, float(base["mean_repair_seconds"])),
        "efficiency_ratio": float(
            selected["conflict_reduction_per_repair_second"]
        )
        / max(1e-9, float(base["conflict_reduction_per_repair_second"])),
    }


def _cell_gate(
    states: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for index, state in enumerate(states):
        grouped[(str(state["layout_mode"]), int(state["agent_count"]))].append(index)
    rows = []
    for (layout, agents), indices in sorted(grouped.items()):
        chosen = _metrics([selected[index] for index in indices])
        base = _metrics(
            [
                next(
                    arm
                    for arm in _model_arms(states[index])
                    if bool(arm["base_selected"])
                )
                for index in indices
            ]
        )
        comparison = _comparison(chosen, base)
        rows.append(
            {
                "layout_mode": layout,
                "agent_count": agents,
                "state_count": len(indices),
                **comparison,
                "noninferior": (
                    comparison["conflict_reduction_ratio"] >= 0.95
                    and comparison["efficiency_ratio"] >= 0.95
                ),
            }
        )
    return {
        "cell_count": len(rows),
        "noninferior_cell_count": sum(int(row["noninferior"]) for row in rows),
        "rows": rows,
    }


def evaluate_policy(
    states: list[dict[str, Any]],
    thresholds: dict[str, float],
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = []
    diagnostics = []
    for state in states:
        arm, diagnostic = select_cost_tiebreak(state, thresholds, calibration)
        selected.append(arm)
        diagnostics.append(
            {
                "state_id": str(state["state_id"]),
                "map_id": str(state["map_id"]),
                "layout_mode": str(state["layout_mode"]),
                "agent_count": int(state["agent_count"]),
                "selected_actual_size": int(arm["actual_size"]),
                "selected_actual_conflict_reduction": float(
                    arm["actual"]["conflict_reduction"]
                ),
                "selected_actual_repair_seconds": float(
                    arm["actual"]["repair_seconds"]
                ),
                **diagnostic,
            }
        )
    base = [
        next(arm for arm in _model_arms(state) if bool(arm["base_selected"]))
        for state in states
    ]
    selected_metrics = _metrics(selected)
    base_metrics = _metrics(base)
    comparison = _comparison(selected_metrics, base_metrics)
    cells = _cell_gate(states, selected)
    override_fraction = statistics.fmean(
        float(row["override"]) for row in diagnostics
    )
    return {
        "selected": selected_metrics,
        "v2": base_metrics,
        "comparison": comparison,
        "override_fraction": override_fraction,
        "override_count": sum(int(row["override"]) for row in diagnostics),
        "cell_gate": cells,
    }, diagnostics


def _training_checks(report: dict[str, Any]) -> dict[str, bool]:
    comparison = dict(report["comparison"])
    cells = dict(report["cell_gate"])
    return {
        "overrides_at_least_5pct": float(report["override_fraction"]) >= 0.05,
        "conflict_reduction_retention_at_least_98pct": (
            float(comparison["conflict_reduction_ratio"]) >= 0.98
        ),
        "effective_rate_not_lower": float(comparison["effective_rate_delta"])
        >= -1e-12,
        "no_progress_rate_not_higher": float(
            comparison["no_progress_rate_delta"]
        )
        <= 1e-12,
        "repair_time_at_least_10pct_lower": float(comparison["repair_time_ratio"])
        <= 0.90 + 1e-12,
        "efficiency_at_least_10pct_higher": float(comparison["efficiency_ratio"])
        >= 1.10 - 1e-12,
        "at_least_five_of_six_cells_noninferior": (
            int(cells["cell_count"]) == 6
            and int(cells["noninferior_cell_count"]) >= 5
        ),
    }


def _diagnostic_checks(report: dict[str, Any]) -> dict[str, bool]:
    comparison = dict(report["comparison"])
    cells = dict(report["cell_gate"])
    return {
        "overrides_nonzero": int(report["override_count"]) > 0,
        "conflict_reduction_retention_at_least_98pct": (
            float(comparison["conflict_reduction_ratio"]) >= 0.98
        ),
        "effective_rate_not_lower_by_more_than_1pp": (
            float(comparison["effective_rate_delta"]) >= -0.01 - 1e-12
        ),
        "no_progress_rate_not_higher_by_more_than_1pp": (
            float(comparison["no_progress_rate_delta"]) <= 0.01 + 1e-12
        ),
        "repair_time_at_least_5pct_lower": float(comparison["repair_time_ratio"])
        <= 0.95 + 1e-12,
        "efficiency_not_lower": float(comparison["efficiency_ratio"])
        >= 1.0 - 1e-12,
        "at_least_five_of_six_cells_noninferior": (
            int(cells["cell_count"]) == 6
            and int(cells["noninferior_cell_count"]) >= 5
        ),
    }


def _threshold_grid() -> Iterable[dict[str, float]]:
    for effective, no_progress, retention, time_gain, utility_gain in itertools.product(
        EFFECTIVE_TOLERANCE_GRID,
        NO_PROGRESS_TOLERANCE_GRID,
        QUALITY_RETENTION_GRID,
        MINIMUM_TIME_IMPROVEMENT_GRID,
        MINIMUM_UTILITY_IMPROVEMENT_GRID,
    ):
        yield {
            "effective_probability_tolerance": effective,
            "no_progress_probability_tolerance": no_progress,
            "conflict_reduction_retention": retention,
            "minimum_time_improvement": time_gain,
            "minimum_utility_improvement": utility_gain,
        }


def calibrate_policy(
    states: list[dict[str, Any]], calibration: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for thresholds in _threshold_grid():
        metrics, _ = evaluate_policy(states, thresholds, calibration)
        checks = _training_checks(metrics)
        rows.append(
            {
                **thresholds,
                **dict(metrics["comparison"]),
                "override_fraction": float(metrics["override_fraction"]),
                "noninferior_cell_count": int(
                    metrics["cell_gate"]["noninferior_cell_count"]
                ),
                "passed": all(checks.values()),
            }
        )
    passing = [row for row in rows if bool(row["passed"])]
    selected = max(
        passing or rows,
        key=lambda row: (
            bool(row["passed"]),
            float(row["efficiency_ratio"]),
            -float(row["repair_time_ratio"]),
            float(row["conflict_reduction_ratio"]),
            -float(row["override_fraction"]),
        ),
    )
    thresholds = {
        name: float(selected[name])
        for name in (
            "effective_probability_tolerance",
            "no_progress_probability_tolerance",
            "conflict_reduction_retention",
            "minimum_time_improvement",
            "minimum_utility_improvement",
        )
    }
    return {
        "passed": bool(passing),
        "passing_configuration_count": len(passing),
        "configuration_count": len(rows),
        "selected_thresholds": thresholds,
        "selected_summary": selected,
    }, rows


def _render_report(report: dict[str, Any]) -> str:
    train = dict(report["training_oof"])
    diagnostic = dict(report["diagnostic"])
    oracle = dict(report["diagnostic_top3_oracle"])
    return "\n".join(
        [
            "# V2 cost-aware Top-3 tiebreak audit",
            "",
            f"Decision: `{report['decision']}`",
            "",
            "## Method",
            "",
            "- The frozen v2 score defines the candidate Top-3 and remains the quality anchor.",
            "- Existing one-step V3 heads predict progress, no-progress, conflict reduction, and true repair seconds.",
            "- A candidate may override v2 only after the 98% predicted quality guard and conservative residual margins.",
            "- Policy-train uses map-group OOF predictions; inspected policy-validation maps are diagnostic only.",
            "",
            "## Policy-train OOF",
            "",
            f"- Overrides: {int(train['override_count'])}/{int(train['selected']['state_count'])} ({float(train['override_fraction']):.3%})",
            f"- Conflict-reduction ratio vs v2: {float(train['comparison']['conflict_reduction_ratio']):.4f}",
            f"- Repair-time ratio vs v2: {float(train['comparison']['repair_time_ratio']):.4f}",
            f"- Efficiency ratio vs v2: {float(train['comparison']['efficiency_ratio']):.4f}",
            "",
            "## Diagnostic maps",
            "",
            f"- Overrides: {int(diagnostic['override_count'])}/{int(diagnostic['selected']['state_count'])} ({float(diagnostic['override_fraction']):.3%})",
            f"- Conflict-reduction ratio vs v2: {float(diagnostic['comparison']['conflict_reduction_ratio']):.4f}",
            f"- Repair-time ratio vs v2: {float(diagnostic['comparison']['repair_time_ratio']):.4f}",
            f"- Efficiency ratio vs v2: {float(diagnostic['comparison']['efficiency_ratio']):.4f}",
            "",
            "## Diagnostic Top-3 Oracle",
            "",
            f"- Conflict-reduction ratio vs v2: {float(oracle['comparison']['conflict_reduction_ratio']):.4f}",
            f"- Repair-time ratio vs v2: {float(oracle['comparison']['repair_time_ratio']):.4f}",
            f"- Efficiency ratio vs v2: {float(oracle['comparison']['efficiency_ratio']):.4f}",
            "",
            "This is an offline one-repair diagnostic. It does not establish complete-episode wall-clock improvement.",
            "",
        ]
    )


def _fold_summary(
    fold: dict[str, Any], training: list[dict[str, Any]], held: list[dict[str, Any]]
) -> dict[str, int]:
    """Summarize the map-group fold using the producer's canonical keys."""
    return {
        "fold": int(fold["fold"]),
        "training_map_count": len(fold["train_maps"]),
        "validation_map_count": len(fold["validation_maps"]),
        "training_trial_count": len(training),
        "validation_trial_count": len(held),
    }


def audit_v2_cost_tiebreak(*, source: Path, output: Path) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("v2 cost-tiebreak output is non-empty")
    feature_path = source / "collection" / "feature_index.jsonl"
    trial_path = source / "collection" / "trial_manifest.jsonl"
    feature_rows = _read_jsonl(feature_path)
    trial_rows = _read_jsonl(trial_path)
    train_rows = _rows(feature_rows, trial_rows, "policy_train", V3_FEATURE_NAMES)
    diagnostic_rows = _rows(
        feature_rows, trial_rows, "policy_validation", V3_FEATURE_NAMES
    )
    train_maps = {str(row["map_id"]) for row in train_rows}
    diagnostic_maps = {str(row["map_id"]) for row in diagnostic_rows}
    if train_maps & diagnostic_maps:
        raise ValueError("v2 cost-tiebreak train and diagnostic maps overlap")

    oof_index: dict[tuple[str, str], dict[str, list[float]]] = {}
    fold_rows = []
    for fold in _balanced_map_folds(train_rows):
        held_maps = set(fold["validation_maps"])
        training = [row for row in train_rows if row["map_id"] not in held_maps]
        held = [row for row in train_rows if row["map_id"] in held_maps]
        models = _fit(training)
        predicted = _with_runtime_predictions(_predict(models, held), 0.0)
        indexed = _index_predictions(held, predicted)
        if set(indexed) & set(oof_index):
            raise ValueError("v2 cost-tiebreak OOF predictions overlap")
        oof_index.update(indexed)
        fold_rows.append(_fold_summary(fold, training, held))
    train_states = _states(train_rows, oof_index)
    calibration = _uncertainty_calibration(train_states)
    policy_calibration, grid_rows = calibrate_policy(train_states, calibration)
    thresholds = dict(policy_calibration["selected_thresholds"])
    training_metrics, training_state_rows = evaluate_policy(
        train_states, thresholds, calibration
    )
    training_checks = _training_checks(training_metrics)

    final_models = _fit(train_rows)
    diagnostic_predictions = _with_runtime_predictions(
        _predict(final_models, diagnostic_rows), 0.0
    )
    diagnostic_states = _states(
        diagnostic_rows,
        _index_predictions(diagnostic_rows, diagnostic_predictions),
    )
    diagnostic_metrics, diagnostic_state_rows = evaluate_policy(
        diagnostic_states, thresholds, calibration
    )
    diagnostic_checks = _diagnostic_checks(diagnostic_metrics)
    oracle_selected = [_oracle_top3(state) for state in diagnostic_states]
    oracle_metrics = {
        "selected": _metrics(oracle_selected),
        "v2": _metrics(
            [
                next(
                    arm
                    for arm in _model_arms(state)
                    if bool(arm["base_selected"])
                )
                for state in diagnostic_states
            ]
        ),
    }
    oracle_metrics["comparison"] = _comparison(
        oracle_metrics["selected"], oracle_metrics["v2"]
    )
    if all(training_checks.values()) and all(diagnostic_checks.values()):
        decision = "four_seed_tiebreak_pilot_candidate"
    elif (
        float(oracle_metrics["comparison"]["conflict_reduction_ratio"]) >= 0.98
        and float(oracle_metrics["comparison"]["efficiency_ratio"]) >= 1.10
    ):
        decision = "prediction_insufficient_top3_headroom_exists"
    else:
        decision = "v2_cost_tiebreak_rejected_offline"
    report = {
        "schema": V2_COST_TIEBREAK_AUDIT_SCHEMA,
        "decision": decision,
        "diagnostic_only": True,
        "source": {
            "path": str(source),
            "feature_index_sha256": sha256_file(feature_path),
            "trial_manifest_sha256": sha256_file(trial_path),
        },
        "coverage": {
            "training_state_count": len(train_states),
            "diagnostic_state_count": len(diagnostic_states),
            "training_map_count": len(train_maps),
            "diagnostic_map_count": len(diagnostic_maps),
            "feature_count": len(V3_FEATURE_NAMES),
            "top_k": TOP_K,
        },
        "uncertainty_calibration": calibration,
        "policy_calibration": policy_calibration,
        "selected_thresholds": thresholds,
        "training_oof": {**training_metrics, "checks": training_checks},
        "diagnostic": {**diagnostic_metrics, "checks": diagnostic_checks},
        "diagnostic_top3_oracle": oracle_metrics,
        "folds": fold_rows,
        "limitations": [
            "The policy-validation maps were inspected in prior V3 work and are diagnostic, not a fresh locked validation set.",
            "Existing labels average two paired PP seeds and cover one repair only.",
            "The audit reuses V3 feature computation already available to v2; incremental native inference cost is not measured here.",
            "A passing offline result would only authorize a small four-seed pilot, not complete-episode promotion.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "threshold_grid.csv", grid_rows)
    atomic_write_csv(output / "training_state_selections.csv", training_state_rows)
    atomic_write_csv(output / "diagnostic_state_selections.csv", diagnostic_state_rows)
    atomic_write_csv(
        output / "training_cell_metrics.csv",
        list(training_metrics["cell_gate"]["rows"]),
    )
    atomic_write_csv(
        output / "diagnostic_cell_metrics.csv",
        list(diagnostic_metrics["cell_gate"]["rows"]),
    )
    _write_json(output / "v2_cost_tiebreak_audit_report.json", report)
    (output / "v2_cost_tiebreak_audit_report.md").write_text(
        _render_report(report), encoding="utf-8"
    )
    return report


__all__ = [
    "V2_COST_TIEBREAK_AUDIT_SCHEMA",
    "audit_v2_cost_tiebreak",
    "calibrate_policy",
    "evaluate_policy",
    "select_cost_tiebreak",
]
