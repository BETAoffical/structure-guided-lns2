"""Two-head, 16-map, Stage1-only compact-flow router.

The gate head learns ``decisive vs ambiguous`` from every expanded Stage1
pair.  The direction head learns ``structural vs V2`` from decisive pairs
only.  Both heads are evaluated out of fold with fixed map folds, which also
keep every state's pairs together.  Only the adapter's registered 18-D
pre-action vector is used as model input.
"""

from __future__ import annotations

import collections
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
import experiments.stride_hierarchical_ch_compact_flow_stage1_router_v1 as base
from experiments.stride_hierarchical_ch_compact_flow_expanded_train_v1 import (
    EXPANDED_STAGE1_PAIR_SCHEMA,
    FIXED_MAP_FOLDS,
    REPORT_SCHEMA as EXPANDED_TRAIN_REPORT_SCHEMA,
    STAGE1_FEATURE_NAMES,
)


CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_two_head_config.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_two_head_report.v1"
)
PREDICTION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_two_head_prediction.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_expanded_stage1_two_head_v1"
DEFAULT_CONFIG = (
    "configs/stride_hierarchical_ch_compact_flow_expanded_stage1_two_head_v1.json"
)
DEFAULT_OUTPUT = (
    "build/stride-hierarchical-ch-compact-flow-expanded-stage1-two-head-v1"
)

ADAPTER_REPORT_SCHEMA = EXPANDED_TRAIN_REPORT_SCHEMA
EXPANDED_LABEL_SCHEMA = EXPANDED_STAGE1_PAIR_SCHEMA
FEATURE_NAMES = STAGE1_FEATURE_NAMES
FOLDS = {fold: tuple(maps) for fold, maps in FIXED_MAP_FOLDS.items()}
EXPANDED_MAPS = tuple(map_id for maps in FOLDS.values() for map_id in maps)
DECISIVE_LABELS = base.DECISIVE_LABELS
ALL_LABELS = base.ALL_LABELS
HEAD_MODEL_IDS = ("regularized_logistic", "low_capacity_hist_gradient_boosting")
GATE_THRESHOLDS = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.01,
)
DIRECTION_THRESHOLDS = (
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    1.01,
)

EXPECTED_HEAD_MODELS = [
    {
        "model_id": "regularized_logistic",
        "class": "sklearn.linear_model.LogisticRegression",
        "preprocessing": "StandardScaler",
        "parameters": {
            "C": 0.25,
            "class_weight": None,
            "max_iter": 1000,
            "random_state": 20260826,
            "solver": "liblinear",
        },
    },
    {
        "model_id": "low_capacity_hist_gradient_boosting",
        "class": "sklearn.ensemble.HistGradientBoostingClassifier",
        "preprocessing": "none",
        "parameters": {
            "early_stopping": False,
            "l2_regularization": 1.0,
            "learning_rate": 0.05,
            "max_iter": 60,
            "max_leaf_nodes": 5,
            "min_samples_leaf": 8,
            "random_state": 20260826,
        },
    },
]
EXPECTED_COMBINATIONS = [
    {
        "combination_id": "gate_logistic__direction_logistic",
        "gate_model_id": "regularized_logistic",
        "direction_model_id": "regularized_logistic",
    },
    {
        "combination_id": "gate_logistic__direction_hgb",
        "gate_model_id": "regularized_logistic",
        "direction_model_id": "low_capacity_hist_gradient_boosting",
    },
    {
        "combination_id": "gate_hgb__direction_logistic",
        "gate_model_id": "low_capacity_hist_gradient_boosting",
        "direction_model_id": "regularized_logistic",
    },
    {
        "combination_id": "gate_hgb__direction_hgb",
        "gate_model_id": "low_capacity_hist_gradient_boosting",
        "direction_model_id": "low_capacity_hist_gradient_boosting",
    },
]
EXPECTED_GATES = {
    "minimum_oof_balanced_accuracy": 0.55,
    "minimum_structural_win_override_recall": 0.35,
    "minimum_v2_win_protection_recall": 0.35,
    "minimum_decisive_structural_override_coverage": 0.25,
    "minimum_structural_override_precision": 0.6,
    "maximum_ambiguous_structural_override_rate": 0.25,
    "minimum_each_fold_balanced_accuracy": 0.45,
    "minimum_each_fold_structural_win_override_recall": 0.1,
    "maximum_fold_balanced_accuracy_spread": 0.35,
}
THRESHOLD_SELECTION_RULE = (
    "feasible_constraints_first",
    "maximum_final_action_balanced_accuracy",
    "maximum_structural_override_precision",
    "maximum_decisive_structural_override_coverage",
    "lower_gate_threshold",
    "lower_direction_threshold",
)
MODEL_SELECTION_RULE = (
    "all_hard_gates_first",
    "maximum_minimum_fold_balanced_accuracy",
    "maximum_oof_balanced_accuracy",
    "maximum_structural_override_precision",
    "maximum_decisive_structural_override_coverage",
    "config_order",
)
EXPECTED_CLAIM_BOUNDARY = {
    "stage1_only": True,
    "stage2_rows_loaded": False,
    "expanded_train_oof_only": True,
    "development_partition_present": False,
    "full_fit_model_export_allowed": False,
    "sequential_design_only": True,
    "passing_status": "TRAIN_OOF_CHALLENGER_ONLY",
    "failure_status": "NO_GO",
    "v2_replacement_allowed": False,
    "promotion_or_default_export_allowed": False,
    "runtime_or_ttf_claim_authorized": False,
    "final_claim_authorized": False,
    "external_map_disjoint_confirmation_required": True,
}


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_expanded_train_stage1_two_head_oof_challenger"
        or config.get("registration_status") != "REGISTERED"
    ):
        raise ValueError("expanded Stage1 two-head identity changed")
    if set(config.get("inputs") or {}) != {
        "expanded_train_report",
        "expanded_stage1_pair_labels",
    }:
        raise ValueError("two-head input registry changed")
    semantics = dict(config.get("input_semantics") or {})
    if semantics != {
        "required_adapter_report_schema": ADAPTER_REPORT_SCHEMA,
        "required_label_schema": EXPANDED_LABEL_SCHEMA,
        "required_research_split": "expanded_train",
        "expected_map_count": 16,
        "gate_target": "decisive_vs_ambiguous",
        "gate_fit_population": "all_stage1_pairs",
        "direction_target": "structural_win_vs_v2_win",
        "direction_fit_population": "decisive_stage1_pairs_only",
        "stage2_rows_loaded": False,
    }:
        raise ValueError("two-head input semantics changed")
    folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(config.get("expanded_map_folds") or {}).items()
    }
    if folds != FOLDS:
        raise ValueError("two-head fixed map folds changed")
    features = dict(config.get("feature_contract") or {})
    if (
        features.get("source") != "expanded_row.model_feature_vector"
        or tuple(features.get("feature_names") or ()) != FEATURE_NAMES
        or int(features.get("dimension", -1)) != 18
        or features.get("identifier_values_used_as_features") is not False
        or features.get("repair_outcome_or_runtime_used_as_features") is not False
        or features.get("map_task_seed_fields_used_for_grouping_only") is not True
    ):
        raise ValueError("two-head feature contract changed")
    forbidden = set(map(str, features.get("forbidden_input_tokens") or ()))
    if not {
        "map_id",
        "task_id",
        "solver_seed",
        "trial",
        "outcome",
        "result",
        "runtime",
        "pp_seconds",
        "no_progress",
        "rollback",
        "time_limit",
        "normalized_conflict_reduction",
    } <= forbidden:
        raise ValueError("two-head forbidden feature tokens weakened")
    if any(
        token in feature.lower()
        for feature in FEATURE_NAMES
        for token in forbidden
    ):
        raise ValueError("two-head registered feature violates leakage contract")
    if list(config.get("head_models") or ()) != EXPECTED_HEAD_MODELS:
        raise ValueError("two-head model parameters changed")
    if list(config.get("candidate_combinations") or ()) != EXPECTED_COMBINATIONS:
        raise ValueError("two-head candidate combinations changed")
    if dict(config.get("sample_weighting") or {}) != {
        "gate": {
            "unit": "state_occurrence_id",
            "base_weight": "1_over_all_pairs_in_state",
            "class_balance": False,
        },
        "direction": {
            "unit": "state_occurrence_id",
            "base_weight": "1_over_decisive_pairs_in_state",
            "balance_structural_vs_v2": True,
            "ambiguous_rows_in_fit": False,
        },
    }:
        raise ValueError("two-head sample weighting changed")
    thresholds = dict(config.get("threshold_calibration") or {})
    if (
        tuple(map(float, thresholds.get("gate_threshold_grid") or ()))
        != GATE_THRESHOLDS
        or tuple(map(float, thresholds.get("direction_confidence_grid") or ()))
        != DIRECTION_THRESHOLDS
        or float(thresholds.get("maximum_ambiguous_structural_override_rate", -1))
        != 0.25
        or float(
            thresholds.get("minimum_decisive_structural_override_coverage", -1)
        )
        != 0.25
        or tuple(thresholds.get("selection_rule") or ())
        != THRESHOLD_SELECTION_RULE
    ):
        raise ValueError("two-head threshold grid changed")
    if dict(config.get("hard_gates") or {}) != EXPECTED_GATES:
        raise ValueError("two-head hard gates changed")
    if tuple(config.get("model_selection_rule") or ()) != MODEL_SELECTION_RULE:
        raise ValueError("two-head model selection changed")
    if dict(config.get("execution") or {}) != {
        "workers": 16,
        "maximum_workers": 20,
        "parallel_head_jobs": 4,
        "workers_enter_run_fingerprint": True,
        "solver_invocation_allowed": False,
    }:
        raise ValueError("two-head execution contract changed")
    if dict(config.get("outputs") or {}) != {
        "root": DEFAULT_OUTPUT,
        "feature_manifest": "feature_manifest.json",
        "oof_predictions": "two_head_stage1_oof_predictions.jsonl",
        "evaluation_report": "two_head_stage1_evaluation_report.json",
    }:
        raise ValueError("two-head output contract changed")
    if dict(config.get("claim_boundary") or {}) != EXPECTED_CLAIM_BOUNDARY:
        raise ValueError("two-head claim boundary changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    return config_path, project_root, config


def _validate_adapter_report(
    report: Mapping[str, Any], *, pair_labels_path: Path
) -> None:
    if (
        report.get("schema") != ADAPTER_REPORT_SCHEMA
        or report.get("experiment_id")
        != "stride_hierarchical_ch_compact_flow_expanded_train_v1"
        or report.get("status") != "READY_EXPANDED_TRAIN"
        or report.get("complete") is not True
    ):
        raise ValueError("expanded-train adapter is not complete")
    folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(report.get("fixed_map_folds") or {}).items()
    }
    if folds != FOLDS or report.get("folds_frozen_before_label_read") is not True:
        raise ValueError("two-head adapter fold identity changed")
    derivation = dict(report.get("fold_derivation") or {})
    if derivation.get("label_or_outcome_inputs_used") is not False:
        raise ValueError("two-head folds used label/outcome inputs")
    if (
        report.get("all_output_rows_reclassified_as") != "expanded_train"
        or report.get("all_registered_maps_included") is not True
        or int(report.get("registered_map_count", -1)) != 16
        or report.get("expanded_training_use_authorized") is not True
        or report.get("training_authorized") is not True
        or report.get("model_fit_executed") is not False
        or report.get("model_exported") is not False
        or report.get("development_evaluation_authorized") is not False
        or report.get("final_claim_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("runtime_or_ttf_claim_authorized") is not False
        or report.get("map_disjoint_final_confirmation_required") is not True
    ):
        raise ValueError("two-head adapter claim boundary changed")
    pair_artifact = dict(
        dict(report.get("artifacts") or {}).get("expanded_stage1_pair_labels")
        or {}
    )
    if (
        pair_artifact.get("file") != pair_labels_path.name
        or pair_artifact.get("schema") != EXPANDED_LABEL_SCHEMA
        or pair_artifact.get("sha256") != sha256_file(pair_labels_path)
    ):
        raise ValueError("two-head pair artifact pin mismatch")
    feature_contract = dict(report.get("feature_contract") or {})
    if (
        int(feature_contract.get("feature_count", -1)) != len(FEATURE_NAMES)
        or tuple(feature_contract.get("stage1_feature_names") or ())
        != FEATURE_NAMES
        or feature_contract.get(
            "repair_outcome_or_runtime_used_as_model_feature"
        )
        is not False
    ):
        raise ValueError("two-head adapter feature contract changed")
    support = dict(dict(report.get("support") or {}).get("stage1_pair") or {})
    if (
        int(support.get("row_count", -1))
        != int(pair_artifact.get("row_count", -2))
        or int(support.get("active_map_count", -1)) != 16
        or support.get("support_ready") is not True
        or set(dict(support.get("label_counts") or {})) != set(ALL_LABELS)
    ):
        raise ValueError("two-head adapter support changed")


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric registered feature: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite registered feature: {field}")
    return result


def build_pair_example(row: Mapping[str, Any]) -> dict[str, Any]:
    """Read only the adapter's registered model feature vector."""
    state_id = str(row.get("state_occurrence_id", ""))
    pair_id = str(row.get("pair_id", ""))
    label = str(row.get("label", ""))
    if not state_id or not pair_id or label not in ALL_LABELS:
        raise ValueError("two-head pair identity/label changed")
    if (
        row.get("schema") != EXPANDED_LABEL_SCHEMA
        or row.get("research_split") != "expanded_train"
        or row.get("expanded_train") is not True
        or row.get("training_authorized") is not True
        or row.get("development_evaluation_authorized") is not False
        or row.get("final_claim_authorized") is not False
        or row.get("promotion_authorized") is not False
        or row.get("runtime_or_ttf_claim_authorized") is not False
        or row.get("map_disjoint_final_confirmation_required") is not True
        or row.get("runtime_or_pp_seconds_used_in_label") is not False
        or row.get("sequential_design_only") is not True
    ):
        raise ValueError(f"two-head pair trust contract changed: {state_id}")
    if (
        tuple(map(str, row.get("model_feature_names") or ())) != FEATURE_NAMES
        or row.get("model_feature_direction") != "structural_minus_v2"
        or row.get("repair_outcome_or_runtime_used_as_model_feature")
        is not False
        or row.get("exact_structural_v2_sets_differ") is not True
    ):
        raise ValueError(f"two-head registered feature contract changed: {state_id}")
    features = [
        _finite(value, field=f"model_feature_vector[{index}]")
        for index, value in enumerate(list(row.get("model_feature_vector") or ()))
    ]
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"two-head feature dimension changed: {state_id}")
    map_id = str(row.get("map_id", ""))
    expected_fold = next(
        (fold for fold, maps in FOLDS.items() if map_id in maps), None
    )
    if expected_fold is None or str(row.get("train_fold")) != expected_fold:
        raise ValueError(f"two-head map fold changed: {state_id}")
    structural_roles = tuple(
        sorted(map(str, row.get("structural_role_aliases") or ()))
    )
    if not structural_roles or not set(structural_roles) <= {
        "component16",
        "hotspot16",
    }:
        raise ValueError(f"two-head role stratum changed: {state_id}")
    structural_action_id = str(row.get("structural_action_id", ""))
    v2_action_id = str(row.get("v2_action_id", ""))
    structural_agents = tuple(map(int, row.get("exact_structural_agents") or ()))
    v2_agents = tuple(map(int, row.get("exact_v2_agents") or ()))
    if (
        not structural_action_id
        or not v2_action_id
        or structural_action_id == v2_action_id
        or not structural_agents
        or not v2_agents
        or structural_agents == v2_agents
    ):
        raise ValueError(f"two-head exact action identity changed: {state_id}")
    return {
        "pair_id": pair_id,
        "state_occurrence_id": state_id,
        "map_id": map_id,
        "research_split": "expanded_train",
        "train_fold": expected_fold,
        "label": label,
        "gate_target": int(label in DECISIVE_LABELS),
        "direction_target": (
            1
            if label == "structural_win"
            else 0
            if label == "v2_win"
            else None
        ),
        "depth_band": str(row.get("depth_band", "")),
        "role_stratum": "+".join(structural_roles),
        "hierarchical_stratum": str(row.get("hierarchical_stratum", "")),
        "structural_action_id": structural_action_id,
        "v2_action_id": v2_action_id,
        "exact_structural_agents": list(structural_agents),
        "exact_v2_agents": list(v2_agents),
        "features": features,
    }


class HeadClassSupportError(ValueError):
    """Raised when a nested training/calibration split lacks a head class."""


def _head_fit_rows(
    examples: Sequence[Mapping[str, Any]], head: str
) -> list[dict[str, Any]]:
    if head == "gate":
        selected = list(examples)
        target_field = "gate_target"
    elif head == "direction":
        selected = [
            row for row in examples if row.get("label") in DECISIVE_LABELS
        ]
        target_field = "direction_target"
    else:
        raise ValueError(f"unknown two-head head: {head}")
    rows = [{**row, "target": int(row[target_field])} for row in selected]
    if {int(row["target"]) for row in rows} != {0, 1}:
        raise HeadClassSupportError(f"two-head {head} fit lacks both classes")
    return rows


def _head_sample_weights(
    rows: Sequence[Mapping[str, Any]], head: str
) -> tuple[list[float], dict[str, Any]]:
    """Apply the frozen per-state weighting before head-specific balancing."""
    if not rows or {int(row["target"]) for row in rows} != {0, 1}:
        raise HeadClassSupportError(f"two-head {head} weights lack both classes")
    state_counts = collections.Counter(
        str(row["state_occurrence_id"]) for row in rows
    )
    base = [
        1.0 / state_counts[str(row["state_occurrence_id"])] for row in rows
    ]
    state_totals = collections.defaultdict(float)
    for row, weight in zip(rows, base):
        state_totals[str(row["state_occurrence_id"])] += weight
    if any(abs(total - 1.0) > 1e-10 for total in state_totals.values()):
        raise ValueError(f"two-head {head} base state weights do not sum to one")
    base_class_totals = {
        str(target): sum(
            weight
            for row, weight in zip(rows, base)
            if int(row["target"]) == target
        )
        for target in (0, 1)
    }
    if head == "gate":
        weights = base
        class_balance_applied = False
    elif head == "direction":
        if min(base_class_totals.values()) <= 0.0:
            raise HeadClassSupportError("direction class has zero base weight")
        balanced = [
            weight / base_class_totals[str(int(row["target"]))]
            for row, weight in zip(rows, base)
        ]
        scale = len(balanced) / sum(balanced)
        weights = [weight * scale for weight in balanced]
        class_balance_applied = True
    else:
        raise ValueError(f"unknown two-head head: {head}")
    effective_class_totals = {
        str(target): sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["target"]) == target
        )
        for target in (0, 1)
    }
    audit = {
        "head": head,
        "row_count": len(rows),
        "state_count": len(state_counts),
        "base_weight_rule": (
            "1_over_all_pairs_in_state"
            if head == "gate"
            else "1_over_decisive_pairs_in_state"
        ),
        "class_balance_applied": class_balance_applied,
        "base_total_weight": sum(base),
        "expected_base_total_weight": float(len(state_counts)),
        "base_state_total_min": min(state_totals.values()),
        "base_state_total_max": max(state_totals.values()),
        "base_class_weight_totals": base_class_totals,
        "effective_total_weight": sum(weights),
        "effective_class_weight_totals": effective_class_totals,
        "state_weight_conservation_passed": True,
    }
    if head == "gate" and any(
        abs(left - right) > 1e-10
        for left, right in zip(weights, base)
    ):
        raise ValueError("gate class balancing was applied")
    if head == "direction" and abs(
        effective_class_totals["0"] - effective_class_totals["1"]
    ) > 1e-9:
        raise ValueError("direction class weights are not balanced")
    return weights, audit


def _fit_head_estimator(
    model_spec: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    head: str,
) -> Any:
    import numpy as np

    values = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int8)
    weights, _ = _head_sample_weights(rows, head)
    sample_weights = np.asarray(weights, dtype=np.float64)
    model_id = str(model_spec["model_id"])
    parameters = dict(model_spec["parameters"])
    if model_id == "regularized_logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(**parameters)),
            ]
        )
        estimator.fit(values, labels, model__sample_weight=sample_weights)
        return estimator
    if model_id == "low_capacity_hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(**parameters)
        estimator.fit(values, labels, sample_weight=sample_weights)
        return estimator
    raise ValueError(f"unknown two-head model: {model_id}")


def cross_validate_head(
    examples: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    head: str,
    *,
    fold_names: Sequence[str] | None = None,
    fit: Callable[
        [Mapping[str, Any], Sequence[Mapping[str, Any]], str], Any
    ]
    | None = None,
    predict: Callable[[Any, Sequence[Mapping[str, Any]]], list[float]]
    = base._predict_probability,
) -> dict[str, Any]:
    active_folds = tuple(fold_names or FOLDS)
    if set(active_folds) != {
        str(row["train_fold"]) for row in examples
    }:
        raise ValueError("two-head inner fold population changed")
    fit_function = fit or _fit_head_estimator
    probabilities: dict[str, float] = {}
    leakage_audit = []
    for fold in active_folds:
        held_maps = FOLDS[fold]
        outer_fit = [row for row in examples if row.get("train_fold") != fold]
        held_rows = [row for row in examples if row.get("train_fold") == fold]
        fit_rows = _head_fit_rows(outer_fit, head)
        _, weight_audit = _head_sample_weights(fit_rows, head)
        fit_states = {str(row["state_occurrence_id"]) for row in fit_rows}
        held_states = {str(row["state_occurrence_id"]) for row in held_rows}
        fit_maps = {str(row["map_id"]) for row in fit_rows}
        held_map_set = {str(row["map_id"]) for row in held_rows}
        if fit_states & held_states or fit_maps & held_map_set:
            raise ValueError(f"two-head {head} fold leakage: {fold}")
        if held_map_set != set(held_maps):
            raise ValueError(f"two-head {head} held-map support changed: {fold}")
        estimator = fit_function(model_spec, fit_rows, head)
        predicted = predict(estimator, held_rows)
        if len(predicted) != len(held_rows):
            raise ValueError(f"two-head {head} prediction count changed")
        for row, value in zip(held_rows, predicted):
            probability = float(value)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(f"two-head {head} probability invalid")
            pair_id = str(row["pair_id"])
            if pair_id in probabilities:
                raise ValueError(f"two-head {head} duplicate OOF prediction")
            probabilities[pair_id] = probability
        leakage_audit.append(
            {
                "head": head,
                "model_id": str(model_spec["model_id"]),
                "fold": fold,
                "fit_map_count": len(fit_maps),
                "held_map_count": len(held_map_set),
                "fit_state_count": len(fit_states),
                "held_state_count": len(held_states),
                "state_overlap_count": 0,
                "map_overlap_count": 0,
                "leakage": False,
                "fit_weight_audit": weight_audit,
            }
        )
    expected_ids = {str(row["pair_id"]) for row in examples}
    if set(probabilities) != expected_ids:
        raise ValueError(f"two-head {head} OOF coverage changed")
    return {"probabilities": probabilities, "fold_leakage_audit": leakage_audit}


def _state_truth(rows: Sequence[Mapping[str, Any]]) -> str:
    labels = {str(row["label"]) for row in rows}
    if "structural_win" in labels:
        return "structural_win"
    if labels == {"v2_win"}:
        return "v2_win"
    return "ambiguous"


def _materialize_pair_action(
    example: Mapping[str, Any],
    *,
    gate_probability: float | None,
    direction_probability: float | None,
    gate_threshold: float,
    direction_threshold: float,
    combination: Mapping[str, Any] | None,
    inner_selection_status: str,
) -> dict[str, Any]:
    if gate_probability is not None and (
        not math.isfinite(gate_probability) or not 0.0 <= gate_probability <= 1.0
    ):
        raise ValueError("invalid gate probability")
    if direction_probability is not None and (
        not math.isfinite(direction_probability)
        or not 0.0 <= direction_probability <= 1.0
    ):
        raise ValueError("invalid direction probability")
    structural_override = bool(
        gate_probability is not None
        and direction_probability is not None
        and gate_probability + 1e-12 >= gate_threshold
        and direction_probability + 1e-12 >= direction_threshold
    )
    joint_score = (
        gate_probability * direction_probability
        if gate_probability is not None and direction_probability is not None
        else None
    )
    final_action_id = str(
        example[
            "structural_action_id" if structural_override else "v2_action_id"
        ]
    )
    final_agents = list(
        example[
            "exact_structural_agents"
            if structural_override
            else "exact_v2_agents"
        ]
    )
    if structural_override and final_action_id != str(example["structural_action_id"]):
        raise ValueError("structural override action mismatch")
    if not structural_override and final_action_id != str(example["v2_action_id"]):
        raise ValueError("exact V2 fallback action mismatch")
    return {
        **example,
        "combination_id": (
            str(combination["combination_id"]) if combination else None
        ),
        "gate_model_id": (
            str(combination["gate_model_id"]) if combination else None
        ),
        "direction_model_id": (
            str(combination["direction_model_id"]) if combination else None
        ),
        "gate_decisive_probability": gate_probability,
        "direction_structural_probability": direction_probability,
        "gate_threshold": float(gate_threshold),
        "direction_structural_threshold": float(direction_threshold),
        "joint_structural_override_score": joint_score,
        "structural_override": structural_override,
        "final_action_id": final_action_id,
        "final_action_role": (
            "structural_override" if structural_override else "exact_v2_fallback"
        ),
        "final_action_agents": final_agents,
        "inner_selection_status": inner_selection_status,
    }


def aggregate_state_actions(
    pair_predictions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in pair_predictions:
        grouped[str(row["state_occurrence_id"])].append(row)
    states: list[dict[str, Any]] = []
    for state_id, rows in sorted(grouped.items()):
        for field in ("map_id", "train_fold", "depth_band"):
            if len({str(row[field]) for row in rows}) != 1:
                raise ValueError(f"state {state_id} has inconsistent {field}")
        v2_action_ids = {str(row["v2_action_id"]) for row in rows}
        v2_agent_sets = {tuple(row["exact_v2_agents"]) for row in rows}
        if len(v2_action_ids) != 1 or len(v2_agent_sets) != 1:
            raise ValueError(f"state {state_id} exact V2 fallback changed across pairs")
        eligible = [row for row in rows if bool(row["structural_override"])]
        selected = (
            sorted(
                eligible,
                key=lambda row: (
                    -float(row["joint_structural_override_score"]),
                    str(row["pair_id"]),
                ),
            )[0]
            if eligible
            else None
        )
        if selected is None:
            final_action_id = next(iter(v2_action_ids))
            final_agents = list(next(iter(v2_agent_sets)))
            final_role = "exact_v2_fallback"
            selected_pair_id = None
        else:
            final_action_id = str(selected["structural_action_id"])
            final_agents = list(selected["exact_structural_agents"])
            final_role = "structural_override"
            selected_pair_id = str(selected["pair_id"])
        roles = sorted(
            {
                role
                for row in rows
                for role in str(row["role_stratum"]).split("+")
                if role
            }
        )
        states.append(
            {
                "state_occurrence_id": state_id,
                "map_id": str(rows[0]["map_id"]),
                "train_fold": str(rows[0]["train_fold"]),
                "depth_band": str(rows[0]["depth_band"]),
                "role_stratum": "+".join(roles),
                "truth_label": _state_truth(rows),
                "pair_count": len(rows),
                "structural_override": selected is not None,
                "selected_pair_id": selected_pair_id,
                "final_action_id": final_action_id,
                "final_action_role": final_role,
                "final_action_agents": final_agents,
                "outer_selection_status": str(rows[0]["inner_selection_status"]),
            }
        )
    return states


def state_action_metrics(
    states: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    structural = [row for row in states if row["truth_label"] == "structural_win"]
    v2 = [row for row in states if row["truth_label"] == "v2_win"]
    ambiguous = [row for row in states if row["truth_label"] == "ambiguous"]
    all_overrides = [row for row in states if bool(row["structural_override"])]
    decisive = structural + v2
    decisive_overrides = [row for row in decisive if bool(row["structural_override"])]
    correct_overrides = [row for row in structural if bool(row["structural_override"])]
    structural_recall = (
        len(correct_overrides) / len(structural) if structural else None
    )
    v2_protection = (
        sum(not bool(row["structural_override"]) for row in v2) / len(v2)
        if v2
        else None
    )
    balanced_accuracy = (
        (structural_recall + v2_protection) / 2.0
        if structural_recall is not None and v2_protection is not None
        else None
    )
    return {
        "state_count": len(states),
        "structural_win_state_count": len(structural),
        "v2_win_state_count": len(v2),
        "ambiguous_state_count": len(ambiguous),
        "structural_override_count": len(all_overrides),
        "correct_structural_override_count": len(correct_overrides),
        "structural_win_override_recall": structural_recall,
        "v2_win_protection_recall": v2_protection,
        "final_action_balanced_accuracy": balanced_accuracy,
        "ambiguous_structural_override_count": sum(
            bool(row["structural_override"]) for row in ambiguous
        ),
        "ambiguous_structural_override_rate": (
            sum(bool(row["structural_override"]) for row in ambiguous)
            / len(ambiguous)
            if ambiguous
            else 0.0
        ),
        "decisive_structural_override_coverage": (
            len(decisive_overrides) / len(decisive) if decisive else 0.0
        ),
        "structural_override_precision": (
            len(correct_overrides) / len(all_overrides)
            if all_overrides
            else None
        ),
        "structural_override_precision_on_decisive_only": (
            len(correct_overrides) / len(decisive_overrides)
            if decisive_overrides
            else None
        ),
        "default_v2_state_count": len(states) - len(all_overrides),
        "only_structural_is_an_override": True,
    }


def pair_state_weighted_metrics(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    counts = collections.Counter(str(row["state_occurrence_id"]) for row in rows)
    weighted = [(row, 1.0 / counts[str(row["state_occurrence_id"])]) for row in rows]

    def total(label: str, *, override: bool | None = None) -> float:
        return sum(
            weight
            for row, weight in weighted
            if str(row["label"]) == label
            and (override is None or bool(row["structural_override"]) is override)
        )

    s_total = total("structural_win")
    v_total = total("v2_win")
    a_total = total("ambiguous")
    s_override = total("structural_win", override=True)
    v_override = total("v2_win", override=True)
    a_override = total("ambiguous", override=True)
    all_override = s_override + v_override + a_override
    decisive_override = s_override + v_override
    s_recall = s_override / s_total if s_total else None
    v_protection = 1.0 - (v_override / v_total) if v_total else None
    return {
        "pair_row_count": len(rows),
        "state_count": len(counts),
        "state_normalized_total_weight": sum(weight for _, weight in weighted),
        "weighted_structural_win_support": s_total,
        "weighted_v2_win_support": v_total,
        "weighted_ambiguous_support": a_total,
        "structural_win_override_recall": s_recall,
        "v2_win_protection_recall": v_protection,
        "final_action_balanced_accuracy": (
            (s_recall + v_protection) / 2.0
            if s_recall is not None and v_protection is not None
            else None
        ),
        "ambiguous_structural_override_rate": (
            a_override / a_total if a_total else 0.0
        ),
        "decisive_structural_override_coverage": (
            decisive_override / (s_total + v_total)
            if s_total + v_total
            else 0.0
        ),
        "structural_override_precision": (
            s_override / all_override if all_override else None
        ),
        "structural_override_precision_on_decisive_only": (
            s_override / decisive_override if decisive_override else None
        ),
    }


def _group_state_metrics(
    states: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in states:
        groups[str(row[field])].append(row)
    return {
        name: state_action_metrics(rows) for name, rows in sorted(groups.items())
    }


def _group_pair_metrics(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        name: pair_state_weighted_metrics(group)
        for name, group in sorted(groups.items())
    }


def _direction_state_metrics(
    predictions: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    decisive = [row for row in predictions if row["label"] in DECISIVE_LABELS]
    if not decisive:
        raise HeadClassSupportError("direction threshold has no decisive rows")
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in decisive:
        groups[str(row["state_occurrence_id"])].append(row)
    states = []
    for state_id, rows in sorted(groups.items()):
        truth = _state_truth(rows)
        if truth not in DECISIVE_LABELS:
            raise ValueError(f"direction-only state truth is ambiguous: {state_id}")
        states.append(
            {
                "state_occurrence_id": state_id,
                "truth_label": truth,
                "structural_override": any(
                    float(row["direction_structural_probability"]) + 1e-12
                    >= threshold
                    for row in rows
                ),
            }
        )
    metrics = state_action_metrics(states)
    if (
        metrics["structural_win_state_count"] == 0
        or metrics["v2_win_state_count"] == 0
    ):
        raise HeadClassSupportError("direction threshold states lack S/V support")
    return metrics


def select_direction_threshold(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidates = []
    for threshold in DIRECTION_THRESHOLDS:
        metrics = _direction_state_metrics(predictions, threshold)
        candidates.append(
            {"direction_structural_threshold": threshold, "metrics": metrics}
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["metrics"]["final_action_balanced_accuracy"] or -1.0),
            float(row["metrics"]["structural_override_precision"] or -1.0),
            float(row["metrics"]["decisive_structural_override_coverage"]),
            -float(row["direction_structural_threshold"]),
        ),
    )
    return {
        "status": "selected_on_decisive_inner_oof_only",
        "selected_direction_structural_threshold": float(
            selected["direction_structural_threshold"]
        ),
        "selected_metrics": selected["metrics"],
        "grid": candidates,
        "ambiguous_rows_used": False,
        "selection_population": "inner_oof_decisive_states_only",
    }


def _finalize_predictions(
    predictions: Sequence[Mapping[str, Any]],
    gate_threshold: float,
    direction_threshold: float,
    combination: Mapping[str, Any],
    status: str,
) -> list[dict[str, Any]]:
    return [
        _materialize_pair_action(
            row,
            gate_probability=float(row["gate_decisive_probability"]),
            direction_probability=float(row["direction_structural_probability"]),
            gate_threshold=gate_threshold,
            direction_threshold=direction_threshold,
            combination=combination,
            inner_selection_status=status,
        )
        for row in predictions
    ]


def select_joint_thresholds(
    predictions: Sequence[Mapping[str, Any]],
    combination: Mapping[str, Any],
    calibration: Mapping[str, Any],
    gates: Mapping[str, Any],
    active_folds: Sequence[str],
) -> dict[str, Any]:
    candidates = []
    for gate_threshold in GATE_THRESHOLDS:
        for direction_threshold in DIRECTION_THRESHOLDS:
            pair_actions = _finalize_predictions(
                predictions,
                gate_threshold,
                direction_threshold,
                combination,
                "inner_oof_joint_threshold_candidate",
            )
            metrics = state_action_metrics(aggregate_state_actions(pair_actions))
            threshold_feasible = bool(
                metrics["ambiguous_structural_override_rate"]
                <= float(
                    calibration["maximum_ambiguous_structural_override_rate"]
                )
                + 1e-12
                and metrics["decisive_structural_override_coverage"]
                >= float(
                    calibration["minimum_decisive_structural_override_coverage"]
                )
                - 1e-12
            )
            by_fold = _group_state_metrics(
                aggregate_state_actions(pair_actions), "train_fold"
            )
            full_checks, stability = _hard_gate_checks(
                metrics, by_fold, gates, active_folds=active_folds
            )
            full_passed = all(full_checks.values())
            candidates.append(
                {
                    "gate_threshold": gate_threshold,
                    "direction_structural_threshold": direction_threshold,
                    "metrics": metrics,
                    "threshold_feasible": threshold_feasible,
                    "inner_state_level_by_fold": by_fold,
                    "inner_stability": stability,
                    "inner_full_gate_checks": full_checks,
                    "inner_full_gates_passed": full_passed,
                    "eligible_for_outer_prediction": (
                        threshold_feasible and full_passed
                    ),
                }
            )
    eligible = [row for row in candidates if row["eligible_for_outer_prediction"]]
    if not eligible:
        return {
            "status": "no_joint_threshold_passes_feasibility_and_full_inner_gates",
            "selected_gate_threshold": 1.01,
            "selected_direction_structural_threshold": 1.01,
            "selected_metrics": None,
            "grid": candidates,
            "joint_grid_size": len(candidates),
        }
    selected = max(
        eligible,
        key=lambda row: (
            float(row["metrics"]["final_action_balanced_accuracy"] or -1.0),
            float(row["metrics"]["structural_override_precision"] or -1.0),
            float(row["metrics"]["decisive_structural_override_coverage"]),
            -float(row["gate_threshold"]),
            -float(row["direction_structural_threshold"]),
        ),
    )
    return {
        "status": "joint_threshold_passes_feasibility_and_full_inner_gates",
        "selected_gate_threshold": float(selected["gate_threshold"]),
        "selected_direction_structural_threshold": float(
            selected["direction_structural_threshold"]
        ),
        "selected_metrics": selected["metrics"],
        "selected_inner_full_gate_checks": selected["inner_full_gate_checks"],
        "grid": candidates,
        "joint_grid_size": len(candidates),
    }


def _inner_core_checks(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, bool]:
    precision = metrics.get("structural_override_precision")
    return {
        "minimum_oof_balanced_accuracy": (
            metrics.get("final_action_balanced_accuracy") is not None
            and float(metrics["final_action_balanced_accuracy"])
            >= float(gates["minimum_oof_balanced_accuracy"])
        ),
        "minimum_structural_win_override_recall": (
            metrics.get("structural_win_override_recall") is not None
            and float(metrics["structural_win_override_recall"])
            >= float(gates["minimum_structural_win_override_recall"])
        ),
        "minimum_v2_win_protection_recall": (
            metrics.get("v2_win_protection_recall") is not None
            and float(metrics["v2_win_protection_recall"])
            >= float(gates["minimum_v2_win_protection_recall"])
        ),
        "minimum_decisive_structural_override_coverage": (
            float(metrics["decisive_structural_override_coverage"])
            >= float(gates["minimum_decisive_structural_override_coverage"])
        ),
        "minimum_structural_override_precision": (
            precision is not None
            and float(precision) >= float(gates["minimum_structural_override_precision"])
        ),
        "maximum_ambiguous_structural_override_rate": (
            float(metrics["ambiguous_structural_override_rate"])
            <= float(gates["maximum_ambiguous_structural_override_rate"])
        ),
    }


def select_inner_configuration(
    examples: Sequence[Mapping[str, Any]],
    head_results: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_reports = []
    feasible_indices = []
    for index, combination in enumerate(config["candidate_combinations"]):
        gate_probabilities = head_results[
            ("gate", str(combination["gate_model_id"]))
        ]["probabilities"]
        direction_probabilities = head_results[
            ("direction", str(combination["direction_model_id"]))
        ]["probabilities"]
        combined = [
            {
                **row,
                "gate_decisive_probability": float(
                    gate_probabilities[str(row["pair_id"])]
                ),
                "direction_structural_probability": float(
                    direction_probabilities[str(row["pair_id"])]
                ),
            }
            for row in examples
        ]
        joint = select_joint_thresholds(
            combined,
            combination,
            config["threshold_calibration"],
            config["hard_gates"],
            tuple(
                fold
                for fold in FOLDS
                if fold in {str(row["train_fold"]) for row in examples}
            ),
        )
        report = {
            "combination_id": str(combination["combination_id"]),
            "gate_model_id": str(combination["gate_model_id"]),
            "direction_model_id": str(combination["direction_model_id"]),
            "joint_threshold_selection": joint,
            "feasible": joint["status"]
            == "joint_threshold_passes_feasibility_and_full_inner_gates",
        }
        if report["feasible"]:
            pair_actions = _finalize_predictions(
                combined,
                float(joint["selected_gate_threshold"]),
                float(joint["selected_direction_structural_threshold"]),
                combination,
                "inner_oof_selected_candidate",
            )
            states = aggregate_state_actions(pair_actions)
            metrics = state_action_metrics(states)
            by_fold = _group_state_metrics(states, "train_fold")
            active_folds = tuple(
                fold for fold in FOLDS if fold in by_fold
            )
            checks, stability = _hard_gate_checks(
                metrics,
                by_fold,
                config["hard_gates"],
                active_folds=active_folds,
            )
            if not all(checks.values()):
                raise ValueError("eligible joint threshold failed full inner gates")
            report.update(
                {
                    "inner_oof_state_metrics": metrics,
                    "inner_oof_by_fold": by_fold,
                    "inner_full_gate_checks": checks,
                    "inner_full_gates_passed": all(checks.values()),
                    "inner_stability": stability,
                }
            )
            feasible_indices.append(index)
        else:
            report.update(
                {
                    "inner_oof_state_metrics": None,
                    "inner_oof_by_fold": {},
                    "inner_full_gate_checks": {},
                    "inner_full_gates_passed": False,
                }
            )
        candidate_reports.append(report)
    if not feasible_indices:
        return {
            "status": "no_feasible_inner_configuration_default_v2_outer_fold",
            "selected_combination": None,
            "candidate_combinations": candidate_reports,
        }

    def rank(index: int) -> tuple[Any, ...]:
        report = candidate_reports[index]
        metrics = report["inner_oof_state_metrics"]
        fold_values = [
            float(row["final_action_balanced_accuracy"])
            for row in report["inner_oof_by_fold"].values()
            if row.get("final_action_balanced_accuracy") is not None
        ]
        return (
            bool(report["inner_full_gates_passed"]),
            min(fold_values) if fold_values else -1.0,
            float(metrics.get("final_action_balanced_accuracy") or -1.0),
            float(metrics.get("structural_override_precision") or -1.0),
            float(metrics.get("decisive_structural_override_coverage") or 0.0),
            -index,
        )

    selected_index = max(feasible_indices, key=rank)
    selected_report = candidate_reports[selected_index]
    selected_spec = dict(config["candidate_combinations"][selected_index])
    selected_spec.update(
        {
            "gate_threshold": float(selected_report["joint_threshold_selection"]["selected_gate_threshold"]),
            "direction_structural_threshold": float(
                selected_report["joint_threshold_selection"][
                    "selected_direction_structural_threshold"
                ]
            ),
        }
    )
    return {
        "status": "feasible_inner_configuration_selected",
        "selected_combination": selected_spec,
        "candidate_combinations": candidate_reports,
    }


def _hard_gate_checks(
    overall: Mapping[str, Any],
    by_fold: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Any],
    *,
    active_folds: Sequence[str] | None = None,
) -> tuple[dict[str, bool], dict[str, Any]]:
    required_folds = tuple(active_folds or FOLDS)
    if set(by_fold) != set(required_folds):
        raise ValueError("state-level hard-gate fold support changed")
    core = _inner_core_checks(overall, gates)
    fold_balanced = [
        float(by_fold[fold]["final_action_balanced_accuracy"])
        for fold in required_folds
        if by_fold.get(fold, {}).get("final_action_balanced_accuracy") is not None
    ]
    fold_structural_recall = [
        float(by_fold[fold]["structural_win_override_recall"])
        for fold in required_folds
        if by_fold.get(fold, {}).get("structural_win_override_recall") is not None
    ]
    spread = (
        max(fold_balanced) - min(fold_balanced)
        if len(fold_balanced) == len(required_folds)
        else None
    )
    checks = {
        **core,
        "minimum_each_fold_balanced_accuracy": (
            len(fold_balanced) == len(required_folds)
            and min(fold_balanced) >= float(gates["minimum_each_fold_balanced_accuracy"])
        ),
        "minimum_each_fold_structural_win_override_recall": (
            len(fold_structural_recall) == len(required_folds)
            and min(fold_structural_recall)
            >= float(gates["minimum_each_fold_structural_win_override_recall"])
        ),
        "maximum_fold_balanced_accuracy_spread": (
            spread is not None
            and spread <= float(gates["maximum_fold_balanced_accuracy_spread"])
        ),
    }
    return checks, {
        "active_folds": list(required_folds),
        "fold_balanced_accuracy_spread": spread,
    }


def _diagnostic_spread(
    groups: Mapping[str, Mapping[str, Any]],
) -> float | None:
    values = [
        float(row["final_action_balanced_accuracy"])
        for row in groups.values()
        if row.get("final_action_balanced_accuracy") is not None
    ]
    return max(values) - min(values) if len(values) >= 2 else None


def _prediction_output(
    row: Mapping[str, Any], state: Mapping[str, Any], run_fingerprint: str
) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "pair_id": str(row["pair_id"]),
        "state_occurrence_id": str(row["state_occurrence_id"]),
        "map_id": str(row["map_id"]),
        "train_fold": str(row["train_fold"]),
        "research_split": "expanded_train",
        "label": str(row["label"]),
        "state_truth_label": str(state["truth_label"]),
        "depth_band": str(row["depth_band"]),
        "role_stratum": str(row["role_stratum"]),
        "combination_id": row["combination_id"],
        "gate_model_id": row["gate_model_id"],
        "direction_model_id": row["direction_model_id"],
        "gate_decisive_probability": row["gate_decisive_probability"],
        "direction_structural_probability": row[
            "direction_structural_probability"
        ],
        "gate_threshold": float(row["gate_threshold"]),
        "direction_structural_threshold": float(
            row["direction_structural_threshold"]
        ),
        "joint_structural_override_score": row[
            "joint_structural_override_score"
        ],
        "structural_override": bool(row["structural_override"]),
        "structural_action_id": str(row["structural_action_id"]),
        "v2_action_id": str(row["v2_action_id"]),
        "exact_structural_agents": list(row["exact_structural_agents"]),
        "exact_v2_agents": list(row["exact_v2_agents"]),
        "exact_structural_v2_sets_differ": True,
        "final_action_id": str(row["final_action_id"]),
        "final_action_role": str(row["final_action_role"]),
        "final_action_agents": list(row["final_action_agents"]),
        "state_selected_pair_id": state["selected_pair_id"],
        "state_final_action_id": str(state["final_action_id"]),
        "state_final_action_role": str(state["final_action_role"]),
        "inner_selection_status": str(row["inner_selection_status"]),
        "gate_reject_direction_reject_or_v2_direction_is_exact_v2": True,
        "ambiguous_fallback_is_safe_and_not_counted_as_selection": True,
        "expanded_train_nested_oof_only": True,
        "development_partition_present": False,
        "stage2_rows_loaded": False,
        "promotion_or_default_authorized": False,
        "final_claim_authorized": False,
    }


def run_expanded_stage1_two_head(
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path | None = None,
    *,
    workers: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path, project_root, config = load_config(config_path)
    execution = dict(config["execution"])
    worker_count = int(execution["workers"] if workers is None else workers)
    if worker_count <= 0 or worker_count > int(execution["maximum_workers"]):
        raise ValueError("workers must be in 1..20")
    inputs = {
        name: base._registered_input(project_root, specification, field=name)
        for name, specification in dict(config["inputs"]).items()
    }
    adapter_report = _read_json(inputs["expanded_train_report"])
    _validate_adapter_report(
        adapter_report,
        pair_labels_path=inputs["expanded_stage1_pair_labels"],
    )
    label_rows = _read_jsonl(inputs["expanded_stage1_pair_labels"])
    pair_ids = [str(row.get("pair_id", "")) for row in label_rows]
    if not pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("two-head pair IDs are empty or duplicated")
    examples = [build_pair_example(row) for row in label_rows]
    if {row["label"] for row in examples} != set(ALL_LABELS):
        raise ValueError("two-head labels lack S/V/A support")
    if {row["map_id"] for row in examples} != set(EXPANDED_MAPS):
        raise ValueError("two-head 16-map support changed")
    raw_state_groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in examples:
        raw_state_groups[str(row["state_occurrence_id"])].append(row)
    for state_id, rows in raw_state_groups.items():
        if len({row["train_fold"] for row in rows}) != 1 or len(
            {row["map_id"] for row in rows}
        ) != 1:
            raise ValueError(f"two-head state grouping changed: {state_id}")
        if len({row["v2_action_id"] for row in rows}) != 1:
            raise ValueError(f"two-head state V2 action changed: {state_id}")

    output_root = Path(output or config["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    else:
        output_root = output_root.resolve()
    if not dry_run and output_root.exists() and any(output_root.iterdir()):
        raise ValueError("two-head output already exists")

    model_by_id = {
        str(model["model_id"]): model for model in config["head_models"]
    }
    # Import sklearn once on the caller thread.  Concurrent first imports of
    # different sklearn submodules can observe a partially initialized package.
    import numpy  # noqa: F401
    from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.pipeline import Pipeline  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401

    outer_records = []
    pair_predictions: list[dict[str, Any]] = []
    for outer_fold, held_maps in FOLDS.items():
        outer_train = [row for row in examples if row["train_fold"] != outer_fold]
        outer_held = [row for row in examples if row["train_fold"] == outer_fold]
        inner_folds = tuple(fold for fold in FOLDS if fold != outer_fold)
        train_states = {row["state_occurrence_id"] for row in outer_train}
        held_states = {row["state_occurrence_id"] for row in outer_held}
        if train_states & held_states:
            raise ValueError(f"outer state leakage: {outer_fold}")
        if {row["map_id"] for row in outer_held} != set(held_maps):
            raise ValueError(f"outer held maps changed: {outer_fold}")
        jobs = [
            (head, model_id)
            for head in ("gate", "direction")
            for model_id in HEAD_MODEL_IDS
        ]
        try:
            with ThreadPoolExecutor(
                max_workers=min(
                    worker_count, int(execution["parallel_head_jobs"])
                )
            ) as pool:
                futures = {
                    (head, model_id): pool.submit(
                        cross_validate_head,
                        outer_train,
                        model_by_id[model_id],
                        head,
                        fold_names=inner_folds,
                    )
                    for head, model_id in jobs
                }
                head_results = {
                    key: future.result() for key, future in futures.items()
                }
            inner_selection = select_inner_configuration(
                outer_train, head_results, config
            )
        except HeadClassSupportError as error:
            head_results = {}
            inner_selection = {
                "status": "missing_inner_class_default_v2_outer_fold",
                "selected_combination": None,
                "candidate_combinations": [],
                "error": str(error),
            }

        selected = inner_selection["selected_combination"]
        outer_weight_audits: dict[str, Any] = {}
        if selected is None:
            held_predictions = [
                _materialize_pair_action(
                    row,
                    gate_probability=None,
                    direction_probability=None,
                    gate_threshold=1.01,
                    direction_threshold=1.01,
                    combination=None,
                    inner_selection_status=str(inner_selection["status"]),
                )
                for row in outer_held
            ]
            outer_status = "DEFAULT_V2_NO_FEASIBLE_INNER_CONFIGURATION"
        else:
            try:
                gate_fit_rows = _head_fit_rows(outer_train, "gate")
                direction_fit_rows = _head_fit_rows(outer_train, "direction")
                _, gate_weight_audit = _head_sample_weights(gate_fit_rows, "gate")
                _, direction_weight_audit = _head_sample_weights(
                    direction_fit_rows, "direction"
                )
                outer_weight_audits = {
                    "gate": gate_weight_audit,
                    "direction": direction_weight_audit,
                }
                gate_estimator = _fit_head_estimator(
                    model_by_id[str(selected["gate_model_id"])],
                    gate_fit_rows,
                    "gate",
                )
                direction_estimator = _fit_head_estimator(
                    model_by_id[str(selected["direction_model_id"])],
                    direction_fit_rows,
                    "direction",
                )
                gate_probabilities = base._predict_probability(
                    gate_estimator, outer_held
                )
                direction_probabilities = base._predict_probability(
                    direction_estimator, outer_held
                )
                held_predictions = [
                    _materialize_pair_action(
                        row,
                        gate_probability=float(gate_probability),
                        direction_probability=float(direction_probability),
                        gate_threshold=float(selected["gate_threshold"]),
                        direction_threshold=float(
                            selected["direction_structural_threshold"]
                        ),
                        combination=selected,
                        inner_selection_status=str(inner_selection["status"]),
                    )
                    for row, gate_probability, direction_probability in zip(
                        outer_held, gate_probabilities, direction_probabilities
                    )
                ]
                outer_status = "NESTED_INNER_SELECTED_AND_OUTER_HELD_PREDICTED"
            except HeadClassSupportError as error:
                held_predictions = [
                    _materialize_pair_action(
                        row,
                        gate_probability=None,
                        direction_probability=None,
                        gate_threshold=1.01,
                        direction_threshold=1.01,
                        combination=None,
                        inner_selection_status=(
                            "missing_outer_fit_class_default_v2_outer_fold"
                        ),
                    )
                    for row in outer_held
                ]
                outer_status = "DEFAULT_V2_MISSING_OUTER_FIT_CLASS"
                inner_selection = {
                    **inner_selection,
                    "outer_fit_error": str(error),
                }
        pair_predictions.extend(held_predictions)
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "held_maps": list(held_maps),
                "outer_train_map_count": len(
                    {row["map_id"] for row in outer_train}
                ),
                "outer_held_map_count": len(
                    {row["map_id"] for row in outer_held}
                ),
                "outer_train_state_count": len(train_states),
                "outer_held_state_count": len(held_states),
                "state_overlap_count": 0,
                "map_overlap_count": 0,
                "outer_status": outer_status,
                "inner_selection": inner_selection,
                "inner_head_oof_audits": {
                    f"{head}:{model_id}": result["fold_leakage_audit"]
                    for (head, model_id), result in sorted(head_results.items())
                },
                "outer_fit_weight_audits": outer_weight_audits,
            }
        )

    if len(pair_predictions) != len(examples) or {
        row["pair_id"] for row in pair_predictions
    } != set(pair_ids):
        raise ValueError("two-head nested outer OOF coverage changed")
    states = aggregate_state_actions(pair_predictions)
    state_by_id = {row["state_occurrence_id"]: row for row in states}
    overall = state_action_metrics(states)
    by_fold = _group_state_metrics(states, "train_fold")
    by_role = _group_state_metrics(states, "role_stratum")
    by_depth = _group_state_metrics(states, "depth_band")
    pair_secondary = pair_state_weighted_metrics(pair_predictions)
    pair_secondary_by_fold = _group_pair_metrics(pair_predictions, "train_fold")
    checks, stability = _hard_gate_checks(overall, by_fold, config["hard_gates"])
    nested_complete = all(
        row["outer_status"] == "NESTED_INNER_SELECTED_AND_OUTER_HELD_PREDICTED"
        for row in outer_records
    )
    passed = nested_complete and all(checks.values())
    status = "TRAIN_OOF_CHALLENGER_ONLY" if passed else "NO_GO"
    input_sha = {name: sha256_file(path) for name, path in inputs.items()}
    run_fingerprint = _fingerprint(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(config_path),
            "input_sha256": input_sha,
            "workers": worker_count,
            "fixed_map_folds": {fold: list(maps) for fold, maps in FOLDS.items()},
            "nested_map_cv": {"outer_folds": 4, "inner_folds_per_outer": 3},
            "head_models": config["head_models"],
            "candidate_combinations": config["candidate_combinations"],
            "gate_thresholds": list(GATE_THRESHOLDS),
            "direction_thresholds": list(DIRECTION_THRESHOLDS),
            "sample_weighting": config["sample_weighting"],
            "feature_names": list(FEATURE_NAMES),
        }
    )
    truth_counts = collections.Counter(row["truth_label"] for row in states)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": status,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "workers": worker_count,
        "parallel_head_jobs": int(execution["parallel_head_jobs"]),
        "workers_enter_run_fingerprint": True,
        "input_integrity": {
            "registered_sha256": input_sha,
            "expanded_stage1_pair_count": len(examples),
            "expanded_stage1_state_count": len(states),
            "expanded_map_count": len({row["map_id"] for row in examples}),
            "pair_label_counts": dict(
                sorted(collections.Counter(row["label"] for row in examples).items())
            ),
            "state_truth_counts": dict(sorted(truth_counts.items())),
            "development_row_count": 0,
            "stage2_row_count": 0,
        },
        "feature_contract": {
            "dimension": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "source": "expanded_row.model_feature_vector",
            "identifier_or_outcome_runtime_feature_count": 0,
            "map_task_seed_fields_used_for_grouping_only": True,
        },
        "fit_support": {
            "gate_pair_count": len(examples),
            "gate_state_count": len(raw_state_groups),
            "direction_decisive_pair_count": sum(
                row["label"] in DECISIVE_LABELS for row in examples
            ),
            "direction_decisive_state_count": len(
                {
                    row["state_occurrence_id"]
                    for row in examples
                    if row["label"] in DECISIVE_LABELS
                }
            ),
            "ambiguous_used_for_gate_fit": True,
            "ambiguous_used_for_direction_fit": False,
            "same_state_pairs_grouped_by_map_fold": True,
        },
        "sample_weighting": dict(config["sample_weighting"]),
        "nested_cv": {
            "outer_fold_count": 4,
            "inner_fold_count_per_outer": 3,
            "outer_folds_evaluation_only": True,
            "inner_oof_selects_combination_and_thresholds": True,
            "full_oof_reselection_performed": False,
            "outer_folds": outer_records,
            "all_outer_folds_have_feasible_inner_selection": nested_complete,
        },
        "primary_state_level_final_action_metrics": overall,
        "primary_state_level_by_fold": by_fold,
        "diagnostic_state_level_by_role": by_role,
        "diagnostic_state_level_by_depth_band": by_depth,
        "diagnostic_stability": {
            **stability,
            "role_balanced_accuracy_spread": _diagnostic_spread(by_role),
            "depth_balanced_accuracy_spread": _diagnostic_spread(by_depth),
        },
        "secondary_pair_state_weighted_metrics": pair_secondary,
        "secondary_pair_state_weighted_by_fold": pair_secondary_by_fold,
        "hard_gate_checks": checks,
        "nested_cv_prerequisite_checks": {
            "all_outer_folds_have_feasible_inner_selection": nested_complete
        },
        "hard_gates_passed": passed,
        "threshold_selection_rule": list(THRESHOLD_SELECTION_RULE),
        "model_selection_rule": list(MODEL_SELECTION_RULE),
        "final_action_semantics": {
            "structural_override_condition": (
                "p_gate_decisive>=tg AND p_direction_structural>=td"
            ),
            "all_other_paths": "exact_v2_fallback",
            "state_multi_pair_rule": (
                "any_eligible_pair_then_max_joint_score_tie_pair_id"
            ),
            "state_truth_rule": "any_S_then_S_else_all_V_then_V_else_A",
            "ambiguous_exact_v2_fallback_is_safe": True,
        },
        "model_fit_executed": True,
        "full_fit_model_executed": False,
        "model_exported": False,
        "development_rows_loaded": False,
        "stage2_rows_loaded": False,
        "promotion_or_default_exported": False,
        "v2_replaced": False,
        "runtime_or_ttf_claim_authorized": False,
        "final_claim_authorized": False,
        "next_decision": (
            "retain_train_oof_challenger_pending_external_map_disjoint_confirmation"
            if passed
            else "retain_exact_v2_and_stop_two_head_challenger_no_go"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
        "dry_run": dry_run,
    }
    if dry_run:
        return report

    output_root.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])
    feature_manifest = {
        "schema": (
            "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_two_head_feature_manifest.v1"
        ),
        "run_fingerprint": run_fingerprint,
        "dimension": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "source": "expanded_row.model_feature_vector",
        "forbidden_inputs": list(config["feature_contract"]["forbidden_input_tokens"]),
        "leakage_audit_passed": True,
        "nested_outer_evaluation_inner_selection": True,
        "development_partition_present": False,
        "stage2_rows_loaded": False,
    }
    feature_path = output_root / outputs["feature_manifest"]
    prediction_path = output_root / outputs["oof_predictions"]
    report_path = output_root / outputs["evaluation_report"]
    _write_json(feature_path, feature_manifest)
    output_rows = [
        _prediction_output(
            row,
            state_by_id[str(row["state_occurrence_id"])],
            run_fingerprint,
        )
        for row in sorted(pair_predictions, key=lambda item: str(item["pair_id"]))
    ]
    _write_jsonl(prediction_path, output_rows)
    report["artifacts"] = {
        "feature_manifest": {
            "file": feature_path.name,
            "sha256": sha256_file(feature_path),
        },
        "oof_predictions": {
            "file": prediction_path.name,
            "schema": PREDICTION_SCHEMA,
            "row_count": len(output_rows),
            "sha256": sha256_file(prediction_path),
        },
        "model": None,
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "ADAPTER_REPORT_SCHEMA",
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "EXPANDED_LABEL_SCHEMA",
    "EXPERIMENT_ID",
    "FEATURE_NAMES",
    "FOLDS",
    "GATE_THRESHOLDS",
    "DIRECTION_THRESHOLDS",
    "REPORT_SCHEMA",
    "aggregate_state_actions",
    "build_pair_example",
    "cross_validate_head",
    "pair_state_weighted_metrics",
    "run_expanded_stage1_two_head",
    "select_direction_threshold",
    "select_inner_configuration",
    "state_action_metrics",
    "validate_config",
]
