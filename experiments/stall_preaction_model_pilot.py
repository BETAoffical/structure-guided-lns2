from __future__ import annotations

import collections
import csv
import math
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, read_json, sha256_file, write_json
from experiments.run_output_guard import prepare_run_output
from experiments.stall_preaction_features import FEATURE_NAMES
from experiments.stall_trigger_policy_audit import wilson_upper


STALL_PREACTION_MODEL_PILOT_SCHEMA = "lns2.stall_preaction_model_pilot.v1"
TRIGGER_FEATURE_NAMES = tuple(
    name
    for name in FEATURE_NAMES
    if name not in {"v2.rank_fraction", "v2.score_gap_to_rank1"}
)
RESCUE_FEATURE_NAMES = FEATURE_NAMES
TRIGGER_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
RESCUE_ESCAPE_PROBABILITY_FLOOR = 0.50


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in {"True", "true", "1"}:
        return True
    if value in {"False", "false", "0"}:
        return False
    raise ValueError(f"{field} is not a strict boolean")


def _values(row: dict[str, Any], names: Iterable[str]) -> list[float]:
    values = [float(row[f"feature:{name}"]) for name in names]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("stall pilot contains a non-finite feature")
    return values


def select_zero_false_threshold(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    thresholds: Iterable[float] = TRIGGER_THRESHOLD_GRID,
) -> dict[str, Any]:
    pairs = list(zip(map(int, labels), map(float, probabilities)))
    if not pairs or {label for label, _ in pairs} != {0, 1}:
        raise ValueError("threshold selection requires both trigger classes")
    rows = []
    positive_count = sum(label == 1 for label, _ in pairs)
    negative_count = len(pairs) - positive_count
    for threshold in thresholds:
        tp = sum(label == 1 and score >= threshold for label, score in pairs)
        fp = sum(label == 0 and score >= threshold for label, score in pairs)
        rows.append(
            {
                "threshold": float(threshold),
                "true_positive_count": tp,
                "false_positive_count": fp,
                "recall": tp / positive_count,
                "false_positive_rate": fp / negative_count,
            }
        )
    eligible = [row for row in rows if row["false_positive_count"] == 0]
    selected = (
        max(eligible, key=lambda row: (float(row["recall"]), -float(row["threshold"])))
        if eligible
        else max(rows, key=lambda row: (float(row["threshold"]), -float(row["false_positive_rate"])))
    )
    return {"selected": selected, "grid": rows}


def select_rescue_candidate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("rescue selection requires candidates")
    eligible = [
        row
        for row in materialized
        if float(row["predicted_stable_escape_probability"])
        >= RESCUE_ESCAPE_PROBABILITY_FLOOR
    ]
    if not eligible:
        maximum = max(
            float(row["predicted_stable_escape_probability"])
            for row in materialized
        )
        eligible = [
            row
            for row in materialized
            if float(row["predicted_stable_escape_probability"]) == maximum
        ]
    return min(
        eligible,
        key=lambda row: (
            -float(row["predicted_conflict_delta"]),
            -float(row["predicted_stable_escape_probability"]),
            float(row["predicted_total_decision_seconds"]),
            int(row["candidate_rank"]),
            str(row["candidate_id"]),
        ),
    )


def _trigger_pipeline() -> Any:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=2000,
            random_state=20260729,
            solver="liblinear",
        ),
    )


def _rescue_models() -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "escape": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=2000,
                random_state=20260729,
                solver="liblinear",
            ),
        ),
        "delta": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "log_seconds": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    }


def _fit_rescue(models: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    import numpy as np

    x = np.asarray([_values(row, RESCUE_FEATURE_NAMES) for row in rows], dtype=float)
    models["escape"].fit(
        x, np.asarray([int(row["stable_escape"]) for row in rows], dtype=int)
    )
    models["delta"].fit(
        x, np.asarray([float(row["mean_conflict_delta"]) for row in rows], dtype=float)
    )
    models["log_seconds"].fit(
        x,
        np.asarray(
            [math.log1p(float(row["mean_total_decision_seconds"])) for row in rows],
            dtype=float,
        ),
    )


def _predict_rescue(models: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import numpy as np

    x = np.asarray([_values(row, RESCUE_FEATURE_NAMES) for row in rows], dtype=float)
    probabilities = models["escape"].predict_proba(x)[:, 1]
    deltas = models["delta"].predict(x)
    log_seconds = models["log_seconds"].predict(x)
    return [
        {
            **row,
            "predicted_stable_escape_probability": float(probability),
            "predicted_conflict_delta": float(delta),
            "predicted_total_decision_seconds": max(
                0.0, math.expm1(float(log_second))
            ),
        }
        for row, probability, delta, log_second in zip(
            rows, probabilities, deltas, log_seconds
        )
    ]


def _rescue_summary(
    predictions: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in predictions:
        grouped[str(row["state_key"])].append(row)
    selections = []
    for state_key, rows in sorted(grouped.items()):
        ranked = sorted(rows, key=lambda row: int(row["candidate_rank"]))
        model = select_rescue_candidate(ranked)
        rank2 = next(row for row in ranked if int(row["candidate_rank"]) == 2)
        oracle = min(
            ranked,
            key=lambda row: (
                -int(bool(row["stable_escape"])),
                -float(row["mean_conflict_delta"]),
                float(row["mean_total_decision_seconds"]),
                int(row["candidate_rank"]),
            ),
        )
        selections.append(
            {
                "state_key": state_key,
                "map_id": str(model["map_id"]),
                "map_fold": int(model["map_fold"]),
                "model_rank": int(model["candidate_rank"]),
                "model_stable_escape": bool(model["stable_escape"]),
                "model_actual_conflict_delta": float(model["mean_conflict_delta"]),
                "model_actual_total_decision_seconds": float(
                    model["mean_total_decision_seconds"]
                ),
                "rank2_stable_escape": bool(rank2["stable_escape"]),
                "rank2_actual_conflict_delta": float(rank2["mean_conflict_delta"]),
                "rank2_actual_total_decision_seconds": float(
                    rank2["mean_total_decision_seconds"]
                ),
                "oracle_rank": int(oracle["candidate_rank"]),
                "oracle_actual_conflict_delta": float(oracle["mean_conflict_delta"]),
                "oracle_actual_total_decision_seconds": float(
                    oracle["mean_total_decision_seconds"]
                ),
            }
        )
    count = len(selections)
    mean = lambda key: math.fsum(float(row[key]) for row in selections) / count
    summary = {
        "state_count": count,
        "model_stable_escape_count": sum(
            bool(row["model_stable_escape"]) for row in selections
        ),
        "model_stable_escape_fraction": mean("model_stable_escape"),
        "rank2_stable_escape_count": sum(
            bool(row["rank2_stable_escape"]) for row in selections
        ),
        "rank2_stable_escape_fraction": mean("rank2_stable_escape"),
        "model_mean_actual_conflict_delta": mean("model_actual_conflict_delta"),
        "rank2_mean_actual_conflict_delta": mean("rank2_actual_conflict_delta"),
        "oracle_mean_actual_conflict_delta": mean("oracle_actual_conflict_delta"),
        "model_mean_actual_total_decision_seconds": mean(
            "model_actual_total_decision_seconds"
        ),
        "rank2_mean_actual_total_decision_seconds": mean(
            "rank2_actual_total_decision_seconds"
        ),
        "oracle_mean_actual_total_decision_seconds": mean(
            "oracle_actual_total_decision_seconds"
        ),
        "model_rank_counts": dict(
            sorted(collections.Counter(row["model_rank"] for row in selections).items())
        ),
    }
    return summary, selections


def run_stall_preaction_model_pilot(
    features: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    feature_root = Path(features).resolve()
    output_root = Path(output).resolve()
    feature_report_path = feature_root / "stall_preaction_feature_report.json"
    trigger_path = feature_root / "stall_trigger_features.csv"
    rescue_path = feature_root / "stall_rescue_features.csv"
    feature_report = dict(read_json(feature_report_path))
    if (
        feature_report.get("complete") is not True
        or tuple(feature_report.get("feature_names") or ()) != FEATURE_NAMES
        or feature_report.get("future_observational_class_used_as_feature") is not False
        or feature_report.get("map_id_used_as_feature") is not False
        or feature_report.get("measured_wall_time_used_as_feature") is not False
    ):
        raise ValueError("stall model pilot received an invalid feature artifact")
    identity = {
        "schema": STALL_PREACTION_MODEL_PILOT_SCHEMA,
        "feature_report_sha256": sha256_file(feature_report_path),
        "trigger_features_sha256": sha256_file(trigger_path),
        "rescue_features_sha256": sha256_file(rescue_path),
        "trigger_feature_names": list(TRIGGER_FEATURE_NAMES),
        "rescue_feature_names": list(RESCUE_FEATURE_NAMES),
        "trigger_threshold_grid": list(TRIGGER_THRESHOLD_GRID),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    prepare_run_output(output_root, resume=resume, identity=identity)

    states = []
    for raw in _read_csv(trigger_path):
        states.append(
            {
                **raw,
                "map_fold": int(raw["map_fold"]),
                "target": int(
                    _bool(
                        raw["target_rescuable_selector_failure"],
                        field="trigger target",
                    )
                ),
            }
        )
    candidates = []
    for raw in _read_csv(rescue_path):
        candidates.append(
            {
                **raw,
                "map_fold": int(raw["map_fold"]),
                "candidate_rank": int(raw["candidate_rank"]),
                "stable_escape": _bool(raw["stable_escape"], field="stable escape"),
                "stable_failure": _bool(raw["stable_failure"], field="stable failure"),
                "mean_conflict_delta": float(raw["mean_conflict_delta"]),
                "mean_total_decision_seconds": float(
                    raw["mean_total_decision_seconds"]
                ),
            }
        )
    training_states = [row for row in states if row["cohort_role"] == "enriched_training"]
    controls = [row for row in states if row["cohort_role"] == "outcome_blind_control"]
    if len(training_states) != 61 or len(controls) != 48:
        raise ValueError("stall pilot cohort roles differ from the registered labels")

    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    trigger_predictions = []
    rescue_predictions = []
    fold_rows = []
    for fold in range(4):
        train = [row for row in training_states if int(row["map_fold"]) != fold]
        validation = [row for row in training_states if int(row["map_fold"]) == fold]
        trigger = _trigger_pipeline()
        trigger.fit(
            np.asarray([_values(row, TRIGGER_FEATURE_NAMES) for row in train]),
            np.asarray([row["target"] for row in train], dtype=int),
        )
        probabilities = trigger.predict_proba(
            np.asarray([_values(row, TRIGGER_FEATURE_NAMES) for row in validation])
        )[:, 1]
        for row, probability in zip(validation, probabilities):
            trigger_predictions.append(
                {
                    "state_key": row["state_key"],
                    "map_id": row["map_id"],
                    "map_fold": fold,
                    "target": int(row["target"]),
                    "predicted_probability": float(probability),
                }
            )
        positive_keys = {
            str(row["state_key"]) for row in train if int(row["target"]) == 1
        }
        validation_positive_keys = {
            str(row["state_key"])
            for row in validation
            if int(row["target"]) == 1
        }
        rescue_train = [
            row
            for row in candidates
            if row["cohort_role"] == "enriched_training"
            and str(row["state_key"]) in positive_keys
            and int(row["candidate_rank"]) >= 2
        ]
        rescue_validation = [
            row
            for row in candidates
            if row["cohort_role"] == "enriched_training"
            and str(row["state_key"]) in validation_positive_keys
            and int(row["candidate_rank"]) >= 2
        ]
        models = _rescue_models()
        _fit_rescue(models, rescue_train)
        predicted_candidates = _predict_rescue(models, rescue_validation)
        rescue_predictions.extend(predicted_candidates)
        fold_summary, _fold_selections = _rescue_summary(predicted_candidates)
        fold_rows.append(
            {
                "map_fold": fold,
                "trigger_validation_state_count": len(validation),
                "trigger_validation_positive_count": sum(row["target"] for row in validation),
                "rescue_validation_state_count": fold_summary["state_count"],
                "rescue_model_stable_escape_fraction": fold_summary[
                    "model_stable_escape_fraction"
                ],
                "rescue_rank2_stable_escape_fraction": fold_summary[
                    "rank2_stable_escape_fraction"
                ],
            }
        )

    trigger_labels = [int(row["target"]) for row in trigger_predictions]
    trigger_scores = [float(row["predicted_probability"]) for row in trigger_predictions]
    threshold_audit = select_zero_false_threshold(trigger_labels, trigger_scores)
    selected_threshold = float(threshold_audit["selected"]["threshold"])
    trigger_summary = {
        "state_count": len(trigger_predictions),
        "positive_count": sum(trigger_labels),
        "negative_count": len(trigger_labels) - sum(trigger_labels),
        "roc_auc": float(roc_auc_score(trigger_labels, trigger_scores)),
        "average_precision": float(
            average_precision_score(trigger_labels, trigger_scores)
        ),
        "brier_score": float(brier_score_loss(trigger_labels, trigger_scores)),
        "selected_threshold": selected_threshold,
        "selected_threshold_metrics": threshold_audit["selected"],
        "threshold_grid": threshold_audit["grid"],
    }

    final_trigger = _trigger_pipeline()
    final_trigger.fit(
        np.asarray([_values(row, TRIGGER_FEATURE_NAMES) for row in training_states]),
        np.asarray([row["target"] for row in training_states], dtype=int),
    )
    control_probabilities = final_trigger.predict_proba(
        np.asarray([_values(row, TRIGGER_FEATURE_NAMES) for row in controls])
    )[:, 1]
    control_predictions = []
    for row, probability in zip(controls, control_probabilities):
        control_predictions.append(
            {
                "state_key": row["state_key"],
                "map_id": row["map_id"],
                "map_fold": int(row["map_fold"]),
                "target": int(row["target"]),
                "predicted_probability": float(probability),
                "would_trigger": float(probability) >= selected_threshold,
            }
        )
    control_negative_count = sum(row["target"] == 0 for row in control_predictions)
    control_false_positive_count = sum(
        row["target"] == 0 and row["would_trigger"] for row in control_predictions
    )
    control_positive_count = len(control_predictions) - control_negative_count
    control_true_positive_count = sum(
        row["target"] == 1 and row["would_trigger"] for row in control_predictions
    )
    control_summary = {
        "state_count": len(control_predictions),
        "positive_count": control_positive_count,
        "negative_count": control_negative_count,
        "true_positive_count": control_true_positive_count,
        "false_positive_count": control_false_positive_count,
        "observed_false_positive_rate": control_false_positive_count
        / control_negative_count,
        "false_positive_wilson_upper_95": wilson_upper(
            control_false_positive_count, control_negative_count
        ),
        "true_positive_recall": control_true_positive_count / control_positive_count,
        "required_negative_controls_for_one_percent_gate": 381,
        "additional_negative_controls_needed": max(0, 381 - control_negative_count),
    }
    rescue_summary, rescue_selections = _rescue_summary(rescue_predictions)

    training_positive_keys = {
        str(row["state_key"]) for row in training_states if int(row["target"]) == 1
    }
    control_positive_keys = {
        str(row["state_key"]) for row in controls if int(row["target"]) == 1
    }
    rescue_final_train = [
        row
        for row in candidates
        if row["cohort_role"] == "enriched_training"
        and str(row["state_key"]) in training_positive_keys
        and int(row["candidate_rank"]) >= 2
    ]
    rescue_control_rows = [
        row
        for row in candidates
        if row["cohort_role"] == "outcome_blind_control"
        and str(row["state_key"]) in control_positive_keys
        and int(row["candidate_rank"]) >= 2
    ]
    final_rescue_models = _rescue_models()
    _fit_rescue(final_rescue_models, rescue_final_train)
    rescue_control_predictions = _predict_rescue(
        final_rescue_models, rescue_control_rows
    )
    rescue_control_summary, rescue_control_selections = _rescue_summary(
        rescue_control_predictions
    )

    trigger_signal = bool(
        trigger_summary["roc_auc"] >= 0.65
        and float(trigger_summary["selected_threshold_metrics"]["recall"]) >= 0.20
    )
    rescue_signal = bool(
        rescue_summary["model_stable_escape_fraction"]
        >= rescue_summary["rank2_stable_escape_fraction"] + 0.05 - 1e-12
        and rescue_summary["model_mean_actual_conflict_delta"]
        >= 0.90 * rescue_summary["rank2_mean_actual_conflict_delta"] - 1e-12
    )
    rescue_control_signal = bool(
        rescue_control_summary["model_stable_escape_fraction"]
        >= rescue_control_summary["rank2_stable_escape_fraction"] + 0.05 - 1e-12
        and rescue_control_summary["model_mean_actual_conflict_delta"]
        >= 0.90 * rescue_control_summary["rank2_mean_actual_conflict_delta"]
        - 1e-12
    )
    report = {
        "schema": STALL_PREACTION_MODEL_PILOT_SCHEMA,
        "complete": True,
        "evidence_level": "map-group OOF learnability pilot",
        "feature_count": len(FEATURE_NAMES),
        "trigger_feature_count": len(TRIGGER_FEATURE_NAMES),
        "rescue_feature_count": len(RESCUE_FEATURE_NAMES),
        "trigger_model": "standardized L2 logistic regression",
        "rescue_model": (
            "standardized L2 logistic escape head plus ridge delta/time heads"
        ),
        "rescue_selection_rule": (
            "require predicted stable-escape probability >= 0.50, then maximize "
            "predicted conflict reduction, then probability, time, and v2 rank"
        ),
        "rescue_escape_probability_floor": RESCUE_ESCAPE_PROBABILITY_FLOOR,
        "strict_probability_lexicographic_rule_rejected": True,
        "hyperparameter_search_performed": False,
        "future_outcome_features_used": False,
        "training_state_count": len(training_states),
        "control_state_count": len(controls),
        "map_group_fold_count": 4,
        "trigger_oof": trigger_summary,
        "control_diagnostic": control_summary,
        "rescue_oof": rescue_summary,
        "rescue_control_diagnostic": rescue_control_summary,
        "folds": fold_rows,
        "trigger_signal_supported": trigger_signal,
        "rescue_signal_supported": rescue_signal,
        "rescue_control_signal_supported": rescue_control_signal,
        "pilot_signal_supported": (
            trigger_signal and rescue_signal and rescue_control_signal
        ),
        "post_failure_rescue_signal_supported": (
            rescue_signal and rescue_control_signal
        ),
        "shadow_promotion_eligible": False,
        "controller_actions_changed": False,
        "deployable_bundle_written": False,
        "deployment_promoted": False,
        "decision": (
            "collect_independent_negative_controls_then_shadow"
            if trigger_signal and rescue_signal and rescue_control_signal
            else (
                "retain_rescue_ranker_for_conservative_post_failure_shadow"
                if rescue_signal and rescue_control_signal
                else "stop_or_redesign_before_more_control_collection"
            )
        ),
    }
    atomic_write_csv(output_root / "trigger_oof_predictions.csv", trigger_predictions)
    atomic_write_csv(output_root / "trigger_control_predictions.csv", control_predictions)
    atomic_write_csv(output_root / "rescue_oof_predictions.csv", rescue_predictions)
    atomic_write_csv(output_root / "rescue_state_selections.csv", rescue_selections)
    atomic_write_csv(
        output_root / "rescue_control_predictions.csv", rescue_control_predictions
    )
    atomic_write_csv(
        output_root / "rescue_control_selections.csv", rescue_control_selections
    )
    atomic_write_csv(output_root / "map_fold_results.csv", fold_rows)
    write_json(output_root / "stall_preaction_model_pilot_report.json", report)
    return report


__all__ = [
    "RESCUE_FEATURE_NAMES",
    "STALL_PREACTION_MODEL_PILOT_SCHEMA",
    "TRIGGER_FEATURE_NAMES",
    "run_stall_preaction_model_pilot",
    "select_rescue_candidate",
    "select_zero_false_threshold",
]
