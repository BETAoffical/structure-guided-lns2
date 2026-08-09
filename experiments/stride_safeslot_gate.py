from __future__ import annotations

import collections
import hashlib
import json
import math
import statistics
from pathlib import Path, PurePosixPath
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_robustaction_label_collection import (
    state_artifact_tree_sha256,
)
from experiments.stride_slotpool import (
    BASE_FEATURE_NAMES,
    DERIVED_FEATURE_NAMES,
    STATE_FEATURE_NAMES,
    _slot_feature_values,
)
from lns2_selector.training.tree_utils import balanced_map_folds, histogram_trees


CONFIG_SCHEMA = "lns2.stride.safeslot_gate_training_config.v1"
TRAINING_ROW_SCHEMA = "lns2.stride.safeslot_gate_training_row.v1"
PREDICTION_SCHEMA = "lns2.stride.safeslot_gate_oof_prediction.v1"
POLICY_SCHEMA = "lns2.stride.safeslot_gate_oof_policy.v1"
REPORT_SCHEMA = "lns2.stride.safeslot_gate_training_report.v1"
MODEL_SCHEMA = "lns2.stride.safeslot_gate_hist_gbdt.v1"
IMPLEMENTATION_ID = "stride-safeslot-v1"
MODEL_ID = "stride-safeslot-gate-v1"
FEATURE_NAMES = (
    *(f"delta:{name}" for name in BASE_FEATURE_NAMES),
    *(f"challenger:{name}" for name in DERIVED_FEATURE_NAMES),
    *(f"shared:{name}" for name in STATE_FEATURE_NAMES),
)
PRODUCER_FILES = (
    "experiments/stride_safeslot_gate.py",
    "experiments/stride_slotpool.py",
    "lns2_selector/training/tree_utils.py",
)


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered SafeSlot gate input changed: {path}")
    return path


def validate_safeslot_gate_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_map_grouped_gate_training"
        or config.get("implementation_id") != IMPLEMENTATION_ID
        or config.get("model_id") != MODEL_ID
        or config.get("pre_registration_parent_commit")
        != "b4f15a1deb283c4dd6dd01b533da43514b585bcb"
    ):
        raise ValueError("SafeSlot gate identity changed")
    expected_inputs = {
        "residual_analysis_report",
        "safe_replace_labels",
        "residual_collection_report",
        "residual_action_aggregates",
        "readiness_comparisons",
        "slotpool_state_evaluation",
        "grid_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("SafeSlot gate input registry changed")
    if dict(config.get("cohort") or {}) != {
        "state_count": 98,
        "map_count": 16,
        "slotpool_candidate_count": 588,
        "exact_anchor_overlap_count": 1,
        "eligible_challenger_count": 587,
        "positive_challenger_count": 201,
        "negative_challenger_count": 386,
        "all_states_retained": True,
        "known_maze_long_tail_excluded": True,
        "fresh_map_labels_read": False,
    }:
        raise ValueError("SafeSlot gate cohort changed")
    features = dict(config.get("features") or {})
    if features != {
        "base_schema": "lns2.realized_features.v2",
        "base_dimension": len(BASE_FEATURE_NAMES),
        "challenger_minus_anchor_dimension": len(BASE_FEATURE_NAMES),
        "challenger_slot_dimension": len(DERIVED_FEATURE_NAMES),
        "shared_state_dimension": len(STATE_FEATURE_NAMES),
        "input_dimension": len(FEATURE_NAMES),
        "representation": "challenger_minus_exact_v2_anchor_plus_challenger_slot_features_plus_shared_state",
        "residual_teacher_fields_used_as_inputs": False,
        "outcome_fields_used_as_inputs": False,
        "recent_failed_neighborhood_overlap_used": False,
        "current_no_progress_streak_used": False,
        "history_feature_exclusion_reason": "no_matched_safeslot_runtime_history_exists_before_training",
    }:
        raise ValueError("SafeSlot gate feature contract changed")
    if len(BASE_FEATURE_NAMES) != 124 or len(DERIVED_FEATURE_NAMES) != 46:
        raise ValueError("SafeSlot gate upstream feature dimensions changed")
    labels = dict(config.get("labels") or {})
    if labels != {
        "schema": "lns2.stride.safeslot_safe_replace_label.v1",
        "target": "final_safe_replace_positive",
        "exact_anchor_overlap_action": "exclude_outcome_blind_before_training",
        "sample_weighting": "equal_total_weight_per_state_uniform_over_eligible_challengers",
        "runtime_or_ttf_used": False,
        "future_trajectory_used": False,
        "cost_to_go_used": False,
        "remaining_repair_rounds_used": False,
    }:
        raise ValueError("SafeSlot gate label contract changed")
    model = dict(config.get("model") or {})
    if (
        model.get("class") != "sklearn.ensemble.HistGradientBoostingClassifier"
        or dict(model.get("fixed_parameters") or {})
        != {
            "early_stopping": False,
            "learning_rate": 0.05,
            "max_iter": 100,
            "min_samples_leaf": 20,
            "random_state": 20260809,
        }
        or list(model.get("parameter_grid") or ())
        != [
            {"max_leaf_nodes": 7, "l2_regularization": 0.1},
            {"max_leaf_nodes": 7, "l2_regularization": 1.0},
            {"max_leaf_nodes": 15, "l2_regularization": 0.1},
            {"max_leaf_nodes": 15, "l2_regularization": 1.0},
        ]
        or int(model.get("outer_map_folds", 0)) != 4
        or int(model.get("inner_map_folds", 0)) != 3
        or model.get("hyperparameter_selection")
        != "maximum_feasible_replacement_coverage_then_precision_then_roc_auc_then_lowest_parameter_index"
    ):
        raise ValueError("SafeSlot gate model protocol changed")
    calibration = dict(config.get("threshold_calibration") or {})
    if calibration != {
        "threshold_grid": [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01],
        "selection": "lowest_feasible_threshold_per_parameter",
        "minimum_safe_replace_precision": 0.80,
        "minimum_replacement_state_fraction": 0.10,
        "minimum_selected_map_fraction": 0.50,
        "minimum_mean_current_step_advantage": 0.0,
        "minimum_first_fixed_half_advantage": 0.0,
        "minimum_second_fixed_half_advantage": 0.0,
        "maximum_mean_no_progress_rate_delta": 0.0,
        "maximum_mean_residual_risk_delta": 0.0,
        "maximum_map_mean_current_step_regret": 0.02,
        "maximum_map_mean_residual_risk_degradation": 0.02,
        "infeasible_action": "full_abstention_to_exact_v2_and_mark_calibration_infeasible",
    }:
        raise ValueError("SafeSlot gate threshold protocol changed")
    acceptance = dict(config.get("offline_acceptance") or {})
    if acceptance != {
        "minimum_candidate_roc_auc": 0.65,
        "minimum_weighted_accuracy_gain_over_majority": 0.03,
        "minimum_selected_safe_replace_precision": 0.80,
        "minimum_replacement_state_fraction": 0.10,
        "minimum_selected_map_count": 8,
        "minimum_mean_current_step_advantage": 0.0,
        "minimum_first_fixed_half_advantage": 0.0,
        "minimum_second_fixed_half_advantage": 0.0,
        "maximum_mean_no_progress_rate_delta": 0.0,
        "maximum_mean_residual_risk_delta": 0.0,
        "maximum_map_mean_current_step_regret": 0.02,
        "maximum_map_mean_residual_risk_degradation": 0.02,
        "minimum_feasible_outer_fold_fraction": 1.0,
    }:
        raise ValueError("SafeSlot gate offline acceptance changed")
    if dict(config.get("claim_boundary") or {}) != {
        "development_labels_only": True,
        "runtime_integration_allowed": False,
        "formal_ttf_claim": False,
        "fresh_map_label_confirmation_required": True,
        "runtime_success_noninferiority_required_later": True,
        "known_maze_regression_required_later": True,
    }:
        raise ValueError("SafeSlot gate claim boundary changed")
    if dict(config.get("outputs") or {}) != {
        "training": "build/stride-safeslot-gate-training-v1"
    }:
        raise ValueError("SafeSlot gate output changed")
    if project_root is not None:
        for specification in dict(config["inputs"]).values():
            _registered(project_root, dict(specification))
        for root_name, hash_name in (
            ("grid_state_artifact_root", "grid_state_artifact_tree_sha256"),
            ("base_state_artifact_root", "base_state_artifact_tree_sha256"),
        ):
            root = (project_root / str(config[root_name])).resolve()
            if state_artifact_tree_sha256(root) != str(config[hash_name]):
                raise ValueError(f"SafeSlot gate state tree changed: {root}")


def safeslot_feature_vector(
    challenger: dict[str, Any], anchor: dict[str, Any]
) -> list[float]:
    challenger_features = dict(challenger["features"])
    anchor_features = dict(anchor["features"])
    expected = set(BASE_FEATURE_NAMES)
    if set(challenger_features) != expected or set(anchor_features) != expected:
        raise ValueError("SafeSlot gate 124-dimensional feature schema changed")
    for name in STATE_FEATURE_NAMES:
        if not math.isclose(
            float(challenger_features[name]),
            float(anchor_features[name]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"SafeSlot shared state feature differs: {name}")
    values = [
        *(
            float(challenger_features[name]) - float(anchor_features[name])
            for name in BASE_FEATURE_NAMES
        ),
        *_slot_feature_values(challenger),
        *(float(anchor_features[name]) for name in STATE_FEATURE_NAMES),
    ]
    if len(values) != len(FEATURE_NAMES) or any(not math.isfinite(v) for v in values):
        raise ValueError("SafeSlot gate feature vector is invalid")
    return values


def _state_path(root: Path, row: dict[str, Any]) -> Path:
    filename = PurePosixPath(str(row["state_file"]).replace("\\", "/")).name
    path = root / filename
    if not path.is_file() or sha256_file(path) != str(row["state_file_sha256"]):
        raise ValueError(f"SafeSlot gate grid state changed: {path}")
    return path


def _base_state_path(root: Path, grid: dict[str, Any]) -> Path:
    filename = PurePosixPath(str(grid["preflight_state_file"]).replace("\\", "/")).name
    path = root / filename
    if not path.is_file() or sha256_file(path) != str(grid["preflight_state_sha256"]):
        raise ValueError(f"SafeSlot gate base state changed: {path}")
    return path


def _unique_index(
    rows: list[dict[str, Any]], *names: str
) -> dict[tuple[str, ...], dict[str, Any]]:
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row[name]) for name in names)
        if key in result:
            raise ValueError(f"duplicate SafeSlot gate row: {key}")
        result[key] = row
    return result


def build_safeslot_gate_rows(
    config: dict[str, Any], *, project_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    analysis = _read_json(inputs["residual_analysis_report"])
    collection = _read_json(inputs["residual_collection_report"])
    if (
        analysis.get("analysis_passed") is not True
        or analysis.get("model_training_design_allowed") is not True
        or collection.get("passed") is not True
        or collection.get("exact_prior_outcome_reproduction") is not True
    ):
        raise ValueError("SafeSlot gate training is not authorized")
    labels = _unique_index(
        _read_jsonl(inputs["safe_replace_labels"]),
        "state_id",
        "challenger_candidate_id",
    )
    comparisons = _unique_index(
        [
            row
            for row in _read_jsonl(inputs["readiness_comparisons"])
            if bool(row["slotpool_oof_retained"])
        ],
        "state_id",
        "challenger_candidate_id",
    )
    aggregates = _unique_index(
        _read_jsonl(inputs["residual_action_aggregates"]),
        "state_id",
        "candidate_id",
    )
    slot_rows = _unique_index(
        _read_jsonl(inputs["slotpool_state_evaluation"]), "state_id"
    )
    manifest = _read_jsonl(inputs["grid_manifest"])
    scope = dict(config["cohort"])
    if (
        len(labels) != int(scope["slotpool_candidate_count"])
        or len(comparisons) != int(scope["slotpool_candidate_count"])
        or len(manifest) != int(scope["state_count"])
        or len(slot_rows) != int(scope["state_count"])
        or len(aggregates) != 685
    ):
        raise ValueError("SafeSlot gate registered products changed")

    grid_root = (project_root / str(config["grid_state_artifact_root"])).resolve()
    base_root = (project_root / str(config["base_state_artifact_root"])).resolve()
    rows: list[dict[str, Any]] = []
    exact_overlaps: list[dict[str, str]] = []
    all_state_ids: set[str] = set()
    for manifest_row in sorted(manifest, key=lambda row: str(row["state_id"])):
        grid = _read_json(_state_path(grid_root, manifest_row))
        state_id = str(grid["state_id"])
        all_state_ids.add(state_id)
        decision = dict(grid["decision"])
        slot = slot_rows[(state_id,)]
        selected_ids = list(map(str, slot["selected_candidate_ids"]))
        if len(selected_ids) != 6 or len(set(selected_ids)) != 6:
            raise ValueError(f"SafeSlot gate SlotPool budget changed: {state_id}")
        grid_candidates = {
            str(candidate["candidate_id"]): candidate
            for candidate in list(grid["candidates"])
        }
        anchor_summary = dict(grid["v2_base_anchor"])
        anchor_id = str(anchor_summary["candidate_id"])
        base_state = _read_json(_base_state_path(base_root, grid))
        base_candidates = {
            str(candidate["candidate_id"]): candidate
            for candidate in list(base_state["candidates"])
        }
        anchor = base_candidates.get(anchor_id)
        if anchor is None or sorted(map(int, anchor["agents"])) != sorted(
            map(int, anchor_summary["agents"])
        ):
            raise ValueError(f"SafeSlot exact V2 anchor changed: {state_id}")
        for candidate_id in selected_ids:
            challenger = grid_candidates.get(candidate_id)
            key = (state_id, candidate_id)
            label = labels.get(key)
            comparison = comparisons.get(key)
            if challenger is None or label is None or comparison is None:
                raise ValueError(f"SafeSlot selected challenger changed: {key}")
            if str(label["anchor_candidate_id"]) != anchor_id or str(
                comparison["anchor_candidate_id"]
            ) != anchor_id:
                raise ValueError(f"SafeSlot exact anchor label changed: {key}")
            if (state_id, anchor_id) not in aggregates or key not in aggregates:
                raise ValueError(f"SafeSlot residual aggregate changed: {key}")
            exact_overlap = sorted(map(int, challenger["agents"])) == sorted(
                map(int, anchor["agents"])
            )
            if exact_overlap:
                exact_overlaps.append(
                    {
                        "state_id": state_id,
                        "candidate_id": candidate_id,
                        "anchor_candidate_id": anchor_id,
                    }
                )
                continue
            rows.append(
                {
                    "schema": TRAINING_ROW_SCHEMA,
                    "state_id": state_id,
                    "map_id": str(decision["map_id"]),
                    "layout_mode": str(decision["layout_mode"]),
                    "source_policy": str(decision["source_policy"]),
                    "anchor_candidate_id": anchor_id,
                    "challenger_candidate_id": candidate_id,
                    "feature_values": safeslot_feature_vector(challenger, anchor),
                    "label": bool(label["final_safe_replace_positive"]),
                    "mean_current_step_advantage": float(comparison["mean_advantage"]),
                    "first_fixed_half_advantage": float(
                        comparison["first_fixed_half_advantage"]
                    ),
                    "second_fixed_half_advantage": float(
                        comparison["second_fixed_half_advantage"]
                    ),
                    "no_progress_rate_delta": float(
                        comparison["challenger_no_progress_rate"]
                    )
                    - float(comparison["anchor_no_progress_rate"]),
                    "mean_residual_risk_delta": float(
                        label["mean_residual_risk_delta"]
                    ),
                    "first_fixed_half_residual_risk_delta": float(
                        label["first_fixed_half_residual_risk_delta"]
                    ),
                    "second_fixed_half_residual_risk_delta": float(
                        label["second_fixed_half_residual_risk_delta"]
                    ),
                }
            )
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    for state_rows in grouped.values():
        weight = 1.0 / len(state_rows)
        for row in state_rows:
            row["sample_weight"] = weight
    if (
        len(rows) != int(scope["eligible_challenger_count"])
        or sum(bool(row["label"]) for row in rows)
        != int(scope["positive_challenger_count"])
        or len(exact_overlaps) != int(scope["exact_anchor_overlap_count"])
        or len(grouped) != int(scope["state_count"])
        or len({str(row["map_id"]) for row in rows}) != int(scope["map_count"])
        or all_state_ids != set(grouped)
        or any(
            not math.isclose(
                sum(float(row["sample_weight"]) for row in state_rows),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for state_rows in grouped.values()
        )
    ):
        raise ValueError("SafeSlot gate row audit failed")
    return rows, {
        "state_count": len(grouped),
        "map_count": len({str(row["map_id"]) for row in rows}),
        "eligible_challenger_count": len(rows),
        "positive_challenger_count": sum(bool(row["label"]) for row in rows),
        "negative_challenger_count": sum(not bool(row["label"]) for row in rows),
        "exact_anchor_overlap_count": len(exact_overlaps),
        "exact_anchor_overlaps": exact_overlaps,
        "uniform_total_weight_per_state": True,
    }


def _parameters(config: dict[str, Any], parameter_index: int) -> dict[str, Any]:
    return {
        **dict(config["model"]["fixed_parameters"]),
        **dict(config["model"]["parameter_grid"][parameter_index]),
    }


def _fit_model(
    values: Any,
    labels: Any,
    weights: Any,
    indices: list[int],
    parameters: dict[str, Any],
) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    if not indices:
        raise ValueError("SafeSlot gate training split is empty")
    estimator = HistGradientBoostingClassifier(**parameters)
    estimator.fit(values[indices], labels[indices], sample_weight=weights[indices])
    return estimator


def _map_folds(
    rows: list[dict[str, Any]], maps: set[str], count: int
) -> list[dict[str, Any]]:
    layouts = {
        str(row["map_id"]): str(row["layout_mode"])
        for row in rows
        if str(row["map_id"]) in maps
    }
    if set(layouts) != maps:
        raise ValueError("SafeSlot gate map fold coverage changed")
    return balanced_map_folds(
        [
            {"map_id": map_id, "layout_mode": layout}
            for map_id, layout in sorted(layouts.items())
        ],
        count=count,
    )


def candidate_classification_metrics(
    rows: list[dict[str, Any]], probabilities: list[float]
) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(rows) != len(probabilities) or not rows:
        raise ValueError("SafeSlot candidate predictions are incomplete")
    labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.int8)
    weights = np.asarray([float(row["sample_weight"]) for row in rows])
    values = np.asarray(probabilities, dtype=np.float64)
    if len(set(map(int, labels))) != 2:
        raise ValueError("SafeSlot candidate evaluation requires both classes")
    accuracy = float(
        np.average((values >= 0.5) == labels.astype(bool), weights=weights)
    )
    positive_weight = float(weights[labels == 1].sum())
    negative_weight = float(weights[labels == 0].sum())
    majority = max(positive_weight, negative_weight) / float(weights.sum())
    return {
        "roc_auc": float(roc_auc_score(labels, values, sample_weight=weights)),
        "average_precision": float(
            average_precision_score(labels, values, sample_weight=weights)
        ),
        "weighted_accuracy": accuracy,
        "weighted_majority_accuracy": majority,
        "weighted_accuracy_gain_over_majority": accuracy - majority,
        "weighted_positive_prevalence": positive_weight / float(weights.sum()),
    }


def evaluate_gate_policy(
    rows: list[dict[str, Any]], probabilities: list[float], threshold: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(rows) != len(probabilities) or not rows:
        raise ValueError("SafeSlot gate policy predictions are incomplete")
    grouped: dict[str, list[tuple[dict[str, Any], float]]] = collections.defaultdict(list)
    for row, probability in zip(rows, probabilities):
        if not math.isfinite(float(probability)):
            raise ValueError("SafeSlot gate probability is non-finite")
        grouped[str(row["state_id"])].append((row, float(probability)))
    records: list[dict[str, Any]] = []
    for state_id, candidates in sorted(grouped.items()):
        best, probability = min(
            candidates,
            key=lambda item: (-float(item[1]), str(item[0]["challenger_candidate_id"])),
        )
        selected = probability >= float(threshold)
        records.append(
            {
                "schema": POLICY_SCHEMA,
                "state_id": state_id,
                "map_id": str(best["map_id"]),
                "layout_mode": str(best["layout_mode"]),
                "threshold": float(threshold),
                "anchor_candidate_id": str(best["anchor_candidate_id"]),
                "challenger_candidate_id": (
                    str(best["challenger_candidate_id"]) if selected else None
                ),
                "challenger_probability": probability,
                "selected_challenger": selected,
                "selected_safe_replace_positive": bool(best["label"]) if selected else False,
                "mean_current_step_advantage": (
                    float(best["mean_current_step_advantage"]) if selected else 0.0
                ),
                "first_fixed_half_advantage": (
                    float(best["first_fixed_half_advantage"]) if selected else 0.0
                ),
                "second_fixed_half_advantage": (
                    float(best["second_fixed_half_advantage"]) if selected else 0.0
                ),
                "no_progress_rate_delta": (
                    float(best["no_progress_rate_delta"]) if selected else 0.0
                ),
                "mean_residual_risk_delta": (
                    float(best["mean_residual_risk_delta"]) if selected else 0.0
                ),
            }
        )
    return records, summarize_gate_policy(records)


def summarize_gate_policy(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("SafeSlot gate policy has no states")
    selected = [row for row in records if bool(row["selected_challenger"])]
    map_advantages: dict[str, list[float]] = collections.defaultdict(list)
    map_residuals: dict[str, list[float]] = collections.defaultdict(list)
    for row in records:
        map_advantages[str(row["map_id"])].append(
            float(row["mean_current_step_advantage"])
        )
        map_residuals[str(row["map_id"])].append(
            float(row["mean_residual_risk_delta"])
        )
    map_mean_advantage = {
        name: statistics.fmean(values) for name, values in sorted(map_advantages.items())
    }
    map_mean_residual = {
        name: statistics.fmean(values) for name, values in sorted(map_residuals.items())
    }
    return {
        "state_count": len(records),
        "map_count": len(map_advantages),
        "selected_state_count": len(selected),
        "replacement_state_fraction": len(selected) / len(records),
        "selected_map_count": len({str(row["map_id"]) for row in selected}),
        "safe_replace_precision": (
            statistics.fmean(bool(row["selected_safe_replace_positive"]) for row in selected)
            if selected
            else 0.0
        ),
        "unsafe_replacement_fraction": (
            statistics.fmean(not bool(row["selected_safe_replace_positive"]) for row in selected)
            if selected
            else 0.0
        ),
        "mean_current_step_advantage": statistics.fmean(
            float(row["mean_current_step_advantage"]) for row in records
        ),
        "first_fixed_half_advantage": statistics.fmean(
            float(row["first_fixed_half_advantage"]) for row in records
        ),
        "second_fixed_half_advantage": statistics.fmean(
            float(row["second_fixed_half_advantage"]) for row in records
        ),
        "mean_no_progress_rate_delta": statistics.fmean(
            float(row["no_progress_rate_delta"]) for row in records
        ),
        "mean_residual_risk_delta": statistics.fmean(
            float(row["mean_residual_risk_delta"]) for row in records
        ),
        "maximum_map_mean_current_step_regret": max(
            (max(0.0, -value) for value in map_mean_advantage.values()), default=0.0
        ),
        "maximum_map_mean_residual_risk_degradation": max(
            (max(0.0, value) for value in map_mean_residual.values()), default=0.0
        ),
        "map_mean_current_step_advantage": map_mean_advantage,
        "map_mean_residual_risk_delta": map_mean_residual,
    }


def _calibration_checks(
    metrics: dict[str, Any], threshold: float, calibration: dict[str, Any]
) -> dict[str, bool]:
    return {
        "non_sentinel_threshold": float(threshold) <= 1.0,
        "safe_replace_precision": float(metrics["safe_replace_precision"])
        >= float(calibration["minimum_safe_replace_precision"]),
        "replacement_state_fraction": float(metrics["replacement_state_fraction"])
        >= float(calibration["minimum_replacement_state_fraction"]),
        "selected_map_fraction": float(metrics["selected_map_count"])
        / int(metrics["map_count"])
        >= float(calibration["minimum_selected_map_fraction"]),
        "mean_current_step_advantage": float(metrics["mean_current_step_advantage"])
        >= float(calibration["minimum_mean_current_step_advantage"]),
        "first_fixed_half_advantage": float(metrics["first_fixed_half_advantage"])
        >= float(calibration["minimum_first_fixed_half_advantage"]),
        "second_fixed_half_advantage": float(metrics["second_fixed_half_advantage"])
        >= float(calibration["minimum_second_fixed_half_advantage"]),
        "mean_no_progress_rate_delta": float(metrics["mean_no_progress_rate_delta"])
        <= float(calibration["maximum_mean_no_progress_rate_delta"]),
        "mean_residual_risk_delta": float(metrics["mean_residual_risk_delta"])
        <= float(calibration["maximum_mean_residual_risk_delta"]),
        "map_current_step_regret": float(
            metrics["maximum_map_mean_current_step_regret"]
        )
        <= float(calibration["maximum_map_mean_current_step_regret"]),
        "map_residual_risk_degradation": float(
            metrics["maximum_map_mean_residual_risk_degradation"]
        )
        <= float(calibration["maximum_map_mean_residual_risk_degradation"]),
    }


def _inner_select(
    *,
    config: dict[str, Any],
    rows: list[dict[str, Any]],
    values: Any,
    labels: Any,
    weights: Any,
    train_maps: set[str],
) -> tuple[int, float, bool, list[dict[str, Any]]]:
    folds = _map_folds(
        rows, train_maps, int(config["model"]["inner_map_folds"])
    )
    target_indices = [
        index for index, row in enumerate(rows) if str(row["map_id"]) in train_maps
    ]
    summaries: list[dict[str, Any]] = []
    for parameter_index in range(len(config["model"]["parameter_grid"])):
        predictions: dict[int, float] = {}
        for fold in folds:
            fit_maps = set(map(str, fold["train_maps"]))
            validation_maps = set(map(str, fold["validation_maps"]))
            if fit_maps & validation_maps or fit_maps | validation_maps != train_maps:
                raise RuntimeError("SafeSlot inner map leakage detected")
            fit_indices = [
                index
                for index, row in enumerate(rows)
                if str(row["map_id"]) in fit_maps
            ]
            validation_indices = [
                index
                for index, row in enumerate(rows)
                if str(row["map_id"]) in validation_maps
            ]
            estimator = _fit_model(
                values,
                labels,
                weights,
                fit_indices,
                _parameters(config, parameter_index),
            )
            probabilities = estimator.predict_proba(values[validation_indices])[:, 1]
            predictions.update(
                {
                    index: float(probability)
                    for index, probability in zip(validation_indices, probabilities)
                }
            )
        if set(predictions) != set(target_indices):
            raise RuntimeError("SafeSlot inner OOF prediction coverage changed")
        inner_rows = [rows[index] for index in target_indices]
        inner_probabilities = [predictions[index] for index in target_indices]
        classification = candidate_classification_metrics(
            inner_rows, inner_probabilities
        )
        threshold_summaries = []
        selected_threshold = 1.01
        selected_metrics: dict[str, Any] | None = None
        for threshold in map(float, config["threshold_calibration"]["threshold_grid"]):
            _records, policy = evaluate_gate_policy(
                inner_rows, inner_probabilities, threshold
            )
            checks = _calibration_checks(
                policy, threshold, dict(config["threshold_calibration"])
            )
            feasible = all(checks.values())
            threshold_summaries.append(
                {
                    "threshold": threshold,
                    "metrics": policy,
                    "checks": checks,
                    "feasible": feasible,
                }
            )
            if feasible and selected_metrics is None:
                selected_threshold = threshold
                selected_metrics = policy
        summaries.append(
            {
                "parameter_index": parameter_index,
                "parameters": _parameters(config, parameter_index),
                "candidate_metrics": classification,
                "selected_threshold": selected_threshold,
                "calibration_feasible": selected_metrics is not None,
                "selected_policy_metrics": selected_metrics,
                "threshold_summaries": threshold_summaries,
            }
        )
    feasible = [row for row in summaries if bool(row["calibration_feasible"])]
    if not feasible:
        return 0, 1.01, False, summaries
    selected = min(
        feasible,
        key=lambda row: (
            -float(row["selected_policy_metrics"]["replacement_state_fraction"]),
            -float(row["selected_policy_metrics"]["safe_replace_precision"]),
            -float(row["candidate_metrics"]["roc_auc"]),
            int(row["parameter_index"]),
        ),
    )
    return (
        int(selected["parameter_index"]),
        float(selected["selected_threshold"]),
        True,
        summaries,
    )


def export_safeslot_gate_model(
    *,
    estimator: Any,
    parameters: dict[str, Any],
    parameter_index: int,
    threshold: float,
) -> dict[str, Any]:
    if list(map(int, estimator.classes_)) != [0, 1]:
        raise ValueError("SafeSlot portable model requires binary classes")
    payload = {
        "schema": MODEL_SCHEMA,
        "implementation_id": IMPLEMENTATION_ID,
        "model_id": MODEL_ID,
        "feature_names": list(FEATURE_NAMES),
        "feature_dimension": len(FEATURE_NAMES),
        "selected_parameter_index": int(parameter_index),
        "parameters": dict(parameters),
        "safe_replace_threshold": float(threshold),
        "abstain_action": "exact_base_only_v2_anchor",
        "baseline": float(estimator._baseline_prediction[0, 0]),
        "trees": histogram_trees(estimator),
    }
    semantic = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    )
    payload["semantic_sha256"] = hashlib.sha256(
        semantic.encode("utf-8")
    ).hexdigest()
    return payload


def predict_safeslot_gate_model(payload: dict[str, Any], values: Any) -> Any:
    import numpy as np

    if (
        payload.get("schema") != MODEL_SCHEMA
        or payload.get("model_id") != MODEL_ID
        or tuple(payload.get("feature_names") or ()) != FEATURE_NAMES
    ):
        raise ValueError("SafeSlot portable model schema changed")
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise ValueError("SafeSlot portable prediction dimension changed")
    raw = np.full(matrix.shape[0], float(payload["baseline"]), dtype=np.float64)
    for tree in payload["trees"]:
        for row_index, row in enumerate(matrix):
            node_index = 0
            while not bool(tree[node_index]["is_leaf"]):
                node = tree[node_index]
                value = float(row[int(node["feature_idx"])])
                go_left = (
                    bool(node["missing_go_to_left"])
                    if math.isnan(value)
                    else value <= float(node["num_threshold"])
                )
                node_index = int(
                    node["left"] if go_left else node["right"]
                )
            raw[row_index] += float(tree[node_index]["value"])
    probabilities = np.empty_like(raw)
    positive = raw >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
    exponential = np.exp(raw[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def _offline_checks(
    *,
    candidate: dict[str, float],
    policy: dict[str, Any],
    feasible_outer_fold_fraction: float,
    config: dict[str, Any],
) -> dict[str, bool]:
    gates = dict(config["offline_acceptance"])
    return {
        "candidate_roc_auc": float(candidate["roc_auc"])
        >= float(gates["minimum_candidate_roc_auc"]),
        "weighted_accuracy_gain_over_majority": float(
            candidate["weighted_accuracy_gain_over_majority"]
        )
        >= float(gates["minimum_weighted_accuracy_gain_over_majority"]),
        "selected_safe_replace_precision": float(policy["safe_replace_precision"])
        >= float(gates["minimum_selected_safe_replace_precision"]),
        "replacement_state_fraction": float(policy["replacement_state_fraction"])
        >= float(gates["minimum_replacement_state_fraction"]),
        "selected_map_count": int(policy["selected_map_count"])
        >= int(gates["minimum_selected_map_count"]),
        "mean_current_step_advantage": float(policy["mean_current_step_advantage"])
        >= float(gates["minimum_mean_current_step_advantage"]),
        "first_fixed_half_advantage": float(policy["first_fixed_half_advantage"])
        >= float(gates["minimum_first_fixed_half_advantage"]),
        "second_fixed_half_advantage": float(policy["second_fixed_half_advantage"])
        >= float(gates["minimum_second_fixed_half_advantage"]),
        "mean_no_progress_rate_delta": float(policy["mean_no_progress_rate_delta"])
        <= float(gates["maximum_mean_no_progress_rate_delta"]),
        "mean_residual_risk_delta": float(policy["mean_residual_risk_delta"])
        <= float(gates["maximum_mean_residual_risk_delta"]),
        "map_current_step_regret": float(
            policy["maximum_map_mean_current_step_regret"]
        )
        <= float(gates["maximum_map_mean_current_step_regret"]),
        "map_residual_risk_degradation": float(
            policy["maximum_map_mean_residual_risk_degradation"]
        )
        <= float(gates["maximum_map_mean_residual_risk_degradation"]),
        "feasible_outer_fold_fraction": feasible_outer_fold_fraction
        >= float(gates["minimum_feasible_outer_fold_fraction"]),
    }


def train_safeslot_gate(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    import numpy as np

    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = _read_json(config_path)
    validate_safeslot_gate_config(config, project_root=project_root)
    rows, row_audit = build_safeslot_gate_rows(config, project_root=project_root)
    values = np.asarray([row["feature_values"] for row in rows], dtype=np.float32)
    labels = np.asarray([bool(row["label"]) for row in rows], dtype=np.int8)
    weights = np.asarray([float(row["sample_weight"]) for row in rows], dtype=np.float64)
    if values.shape != (len(rows), len(FEATURE_NAMES)):
        raise RuntimeError("SafeSlot gate training matrix changed")
    all_maps = {str(row["map_id"]) for row in rows}
    outer_folds = _map_folds(
        rows, all_maps, int(config["model"]["outer_map_folds"])
    )
    oof_probabilities: dict[int, float] = {}
    prediction_rows: list[dict[str, Any]] = []
    policy_records: list[dict[str, Any]] = []
    fold_diagnostics: list[dict[str, Any]] = []
    for fold in outer_folds:
        train_maps = set(map(str, fold["train_maps"]))
        validation_maps = set(map(str, fold["validation_maps"]))
        if train_maps & validation_maps or train_maps | validation_maps != all_maps:
            raise RuntimeError("SafeSlot outer map leakage detected")
        parameter_index, threshold, calibration_feasible, inner = _inner_select(
            config=config,
            rows=rows,
            values=values,
            labels=labels,
            weights=weights,
            train_maps=train_maps,
        )
        train_indices = [
            index
            for index, row in enumerate(rows)
            if str(row["map_id"]) in train_maps
        ]
        validation_indices = [
            index
            for index, row in enumerate(rows)
            if str(row["map_id"]) in validation_maps
        ]
        estimator = _fit_model(
            values,
            labels,
            weights,
            train_indices,
            _parameters(config, parameter_index),
        )
        probabilities = estimator.predict_proba(values[validation_indices])[:, 1]
        for index, probability in zip(validation_indices, probabilities):
            oof_probabilities[index] = float(probability)
            prediction_rows.append(
                {
                    "schema": PREDICTION_SCHEMA,
                    "outer_fold": int(fold["fold"]),
                    "state_id": str(rows[index]["state_id"]),
                    "map_id": str(rows[index]["map_id"]),
                    "anchor_candidate_id": str(rows[index]["anchor_candidate_id"]),
                    "challenger_candidate_id": str(
                        rows[index]["challenger_candidate_id"]
                    ),
                    "label": bool(rows[index]["label"]),
                    "probability": float(probability),
                    "sample_weight": float(rows[index]["sample_weight"]),
                }
            )
        fold_rows = [rows[index] for index in validation_indices]
        fold_probabilities = list(map(float, probabilities))
        fold_policy_rows, fold_policy = evaluate_gate_policy(
            fold_rows, fold_probabilities, threshold
        )
        for row in fold_policy_rows:
            row["outer_fold"] = int(fold["fold"])
            row["calibration_feasible"] = calibration_feasible
        policy_records.extend(fold_policy_rows)
        fold_diagnostics.append(
            {
                "outer_fold": int(fold["fold"]),
                "train_maps": sorted(train_maps),
                "validation_maps": sorted(validation_maps),
                "train_state_count": len(
                    {str(rows[index]["state_id"]) for index in train_indices}
                ),
                "validation_state_count": len(fold_policy_rows),
                "selected_parameter_index": parameter_index,
                "selected_parameters": _parameters(config, parameter_index),
                "selected_threshold": threshold,
                "calibration_feasible": calibration_feasible,
                "validation_candidate_metrics": candidate_classification_metrics(
                    fold_rows, fold_probabilities
                ),
                "validation_policy_metrics": fold_policy,
                "inner_parameter_summaries": inner,
            }
        )
    if set(oof_probabilities) != set(range(len(rows))):
        raise RuntimeError("SafeSlot outer OOF prediction coverage changed")
    ordered_probabilities = [oof_probabilities[index] for index in range(len(rows))]
    candidate_metrics = candidate_classification_metrics(rows, ordered_probabilities)
    policy_metrics = summarize_gate_policy(policy_records)
    feasible_fraction = statistics.fmean(
        bool(row["calibration_feasible"]) for row in fold_diagnostics
    )
    checks = _offline_checks(
        candidate=candidate_metrics,
        policy=policy_metrics,
        feasible_outer_fold_fraction=feasible_fraction,
        config=config,
    )
    passed = all(checks.values())

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "training_rows.jsonl"
    predictions_path = output / "oof_candidate_predictions.jsonl"
    policy_path = output / "oof_state_policy.jsonl"
    folds_path = output / "fold_diagnostics.jsonl"
    _write_jsonl(rows_path, rows)
    _write_jsonl(
        predictions_path,
        sorted(
            prediction_rows,
            key=lambda row: (
                str(row["state_id"]), str(row["challenger_candidate_id"])
            ),
        ),
    )
    _write_jsonl(
        policy_path,
        sorted(policy_records, key=lambda row: str(row["state_id"])),
    )
    _write_jsonl(folds_path, fold_diagnostics)

    final_parameter_index: int | None = None
    final_threshold: float | None = None
    final_calibration_feasible = False
    model_path = output / "safeslot_gate_model.json"
    portable_parity: float | None = None
    if passed:
        (
            final_parameter_index,
            final_threshold,
            final_calibration_feasible,
            final_summaries,
        ) = _inner_select(
            config=config,
            rows=rows,
            values=values,
            labels=labels,
            weights=weights,
            train_maps=all_maps,
        )
        if not final_calibration_feasible:
            raise RuntimeError("SafeSlot final calibration unexpectedly failed")
        final_estimator = _fit_model(
            values,
            labels,
            weights,
            list(range(len(rows))),
            _parameters(config, final_parameter_index),
        )
        portable = export_safeslot_gate_model(
            estimator=final_estimator,
            parameters=_parameters(config, final_parameter_index),
            parameter_index=final_parameter_index,
            threshold=final_threshold,
        )
        reference = final_estimator.predict_proba(values)[:, 1]
        reproduced = predict_safeslot_gate_model(portable, values)
        portable_parity = max(
            abs(float(left) - float(right))
            for left, right in zip(reference, reproduced)
        )
        if portable_parity > 1e-12:
            raise RuntimeError(
                f"SafeSlot portable model parity failed: {portable_parity}"
            )
        portable.update(
            {
                "sklearn_portable_parity_maximum_absolute_difference": portable_parity,
                "development_map_count": len(all_maps),
                "development_state_count": row_audit["state_count"],
                "development_candidate_count": len(rows),
                "fresh_map_labels_seen": False,
                "runtime_integration_allowed": False,
                "final_inner_parameter_summaries": final_summaries,
            }
        )
        _write_json(model_path, portable)

    artifacts = {
        "training_rows_sha256": sha256_file(rows_path),
        "oof_candidate_predictions_sha256": sha256_file(predictions_path),
        "oof_state_policy_sha256": sha256_file(policy_path),
        "fold_diagnostics_sha256": sha256_file(folds_path),
    }
    if model_path.is_file():
        artifacts["safeslot_gate_model_sha256"] = sha256_file(model_path)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": (
            "offline_gate_passed_pending_fresh_map_label_confirmation"
            if passed
            else "offline_gate_failed_runtime_hard_stop"
        ),
        "implementation_id": IMPLEMENTATION_ID,
        "model_id": MODEL_ID,
        "config_sha256": sha256_file(config_path),
        "producer": producer_identity(
            project_root=project_root,
            source_files=PRODUCER_FILES,
            native_required=False,
        ),
        "feature_dimension": len(FEATURE_NAMES),
        "row_audit": row_audit,
        "outer_fold_count": len(outer_folds),
        "feasible_outer_fold_fraction": feasible_fraction,
        "candidate_metrics": candidate_metrics,
        "policy_metrics": policy_metrics,
        "offline_checks": checks,
        "offline_passed": passed,
        "fold_diagnostics": fold_diagnostics,
        "final_parameter_index": final_parameter_index,
        "final_threshold": final_threshold,
        "final_calibration_feasible": final_calibration_feasible,
        "portable_parity_maximum_absolute_difference": portable_parity,
        "fresh_map_label_confirmation_allowed": passed,
        "runtime_integration_allowed": False,
        "formal_ttf_claim": False,
        "next_decision": (
            "preregister_fresh_map_safeslot_label_confirmation"
            if passed
            else "stop_runtime_and_reassess_gate_without_reusing_outer_labels_for_tuning"
        ),
        "artifacts": artifacts,
    }
    _write_json(output / "safeslot_gate_training_report.json", report)
    return report
