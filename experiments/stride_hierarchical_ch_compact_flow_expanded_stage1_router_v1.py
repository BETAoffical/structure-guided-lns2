"""Expanded-map Stage1-only abstaining router for compact-flow H1 labels.

The router consumes the explicit 16-map expanded-training adapter.  It fits
only decisive ``V2 vs structural`` labels, uses ambiguous rows only to select
an abstention threshold from out-of-fold predictions, and keeps every state
inside one of four fixed map folds.  It never reads Stage2 labels and cannot
make development, final, promotion, runtime, or TTF claims.
"""

from __future__ import annotations

import math
import pickle
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
from experiments.stride_hierarchical_ch_compact_flow_h1_collection_v1 import (
    H1_STATE_SCHEMA,
)
from experiments.stride_hierarchical_ch_compact_flow_expanded_train_v1 import (
    EXPANDED_STAGE1_PAIR_SCHEMA,
    FIXED_MAP_FOLDS,
    REPORT_SCHEMA as EXPANDED_TRAIN_REPORT_SCHEMA,
    STAGE1_FEATURE_NAMES,
)


CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_router_config.v1"
)
ADAPTER_REPORT_SCHEMA = EXPANDED_TRAIN_REPORT_SCHEMA
EXPANDED_LABEL_SCHEMA = EXPANDED_STAGE1_PAIR_SCHEMA
REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_router_report.v1"
)
PREDICTION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_router_prediction.v1"
)
MODEL_MANIFEST_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_challenger_manifest.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_expanded_stage1_router_v1"
DEFAULT_CONFIG = (
    "configs/stride_hierarchical_ch_compact_flow_expanded_stage1_router_v1.json"
)
DEFAULT_OUTPUT = (
    "build/stride-hierarchical-ch-compact-flow-expanded-stage1-router-v1"
)

DECISIVE_LABELS = base.DECISIVE_LABELS
ALL_LABELS = base.ALL_LABELS
FEATURE_NAMES = STAGE1_FEATURE_NAMES
MODEL_IDS = base.MODEL_IDS
THRESHOLD_GRID = base.THRESHOLD_GRID
SELECTION_RULE = base.SELECTION_RULE
MODEL_SELECTION_RULE = base.MODEL_SELECTION_RULE

# Result-blind construction: each fold contains two maps from the old source
# train partition and two from its diagnostic partition.  Capacity strata are
# U/S/M/L in folds 0-2 and U/U/S/M in fold 3 because the cohort has 5 U/3 L.
FOLDS = {
    "fold0": ("den404d", "den408d", "den207d", "den203d"),
    "fold1": ("lak101d", "den202d", "den998d", "den308d"),
    "fold2": ("lak108d", "den201d", "den009d", "den020d"),
    "fold3": ("lak110d", "ost102d", "hrt002d", "den101d"),
}
assert FOLDS == FIXED_MAP_FOLDS
EXPANDED_MAPS = tuple(map_id for maps in FOLDS.values() for map_id in maps)


EXPECTED_MODELS = [
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

EXPECTED_GATES = {
    "minimum_oof_balanced_accuracy": 0.55,
    "minimum_each_class_recall": 0.35,
    "minimum_oof_coverage": 0.25,
    "minimum_oof_selective_accuracy": 0.6,
    "maximum_ambiguous_selection_rate": 0.25,
    "minimum_each_fold_balanced_accuracy": 0.45,
    "minimum_each_fold_coverage": 0.1,
    "maximum_fold_balanced_accuracy_spread": 0.35,
    "maximum_supported_role_balanced_accuracy_spread": 0.35,
    "maximum_supported_depth_balanced_accuracy_spread": 0.35,
    "minimum_rows_for_stability_group": 3,
}

EXPECTED_CLAIM_BOUNDARY = {
    "stage1_only": True,
    "stage2_training_allowed": False,
    "expanded_train_only": True,
    "development_partition_present": False,
    "sequential_design_only": True,
    "passing_status": "EXPANDED_STAGE1_CHALLENGER_ONLY",
    "failure_status": "NO_GO",
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
        != "preregistered_expanded_train_stage1_only_challenger"
    ):
        raise ValueError("expanded Stage1 router identity changed")
    if set(config.get("inputs") or {}) != {
        "expanded_train_report",
        "expanded_stage1_pair_labels",
        "h1_collection_report",
    }:
        raise ValueError("expanded Stage1 input registry changed")
    semantics = dict(config.get("input_semantics") or {})
    if semantics != {
        "required_adapter_report_schema": ADAPTER_REPORT_SCHEMA,
        "required_label_schema": EXPANDED_LABEL_SCHEMA,
        "required_state_schema": H1_STATE_SCHEMA,
        "required_research_split": "expanded_train",
        "expected_map_count": 16,
        "fit_labels": list(DECISIVE_LABELS),
        "ambiguous_use": "abstention_threshold_calibration_only",
        "stage2_rows_loaded": False,
        "stage2_model_fit": False,
    }:
        raise ValueError("expanded Stage1 input semantics changed")
    folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(config.get("expanded_map_folds") or {}).items()
    }
    if folds != FOLDS:
        raise ValueError("expanded Stage1 fixed folds changed")
    folded = [map_id for maps in folds.values() for map_id in maps]
    if len(folded) != 16 or len(folded) != len(set(folded)):
        raise ValueError("expanded Stage1 folds overlap or omit maps")
    features = dict(config.get("feature_contract") or {})
    if (
        features.get("representation")
        != "structural_minus_exact_v2_anchor_with_pre_action_context"
        or tuple(features.get("feature_names") or ()) != FEATURE_NAMES
        or features.get("structural_alias_aggregation")
        != "arithmetic_mean_for_exact_set_aliases"
        or set(
            features.get("categorical_fields_are_evaluation_strata_not_model_inputs")
            or ()
        )
        != {"depth_band", "hierarchical_stratum", "structural_role_aliases"}
        or float(features.get("missing_value", math.nan)) != 0.0
        or features.get("identifier_values_used_as_features") is not False
        or features.get("repair_outcome_or_runtime_used_as_features") is not False
    ):
        raise ValueError("expanded Stage1 feature contract changed")
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
        raise ValueError("expanded Stage1 forbidden feature tokens weakened")
    for name in FEATURE_NAMES:
        if any(token in name.lower() for token in forbidden):
            raise ValueError(f"registered feature violates leakage contract: {name}")
    if list(config.get("models") or ()) != EXPECTED_MODELS:
        raise ValueError("expanded Stage1 model parameters changed")
    if dict(config.get("sample_weighting") or {}) != {
        "unit": "state_occurrence_id",
        "equal_total_weight_per_state": True,
        "balance_decisive_classes_after_state_weighting": True,
    }:
        raise ValueError("expanded Stage1 sample weighting changed")
    calibration = dict(config.get("abstention_calibration") or {})
    if (
        tuple(map(float, calibration.get("threshold_grid") or ()))
        != THRESHOLD_GRID
        or float(calibration.get("maximum_ambiguous_selection_rate", -1)) != 0.25
        or float(calibration.get("minimum_decisive_coverage", -1)) != 0.25
        or tuple(calibration.get("selection_rule") or ()) != SELECTION_RULE
    ):
        raise ValueError("expanded Stage1 abstention calibration changed")
    if dict(config.get("hard_gates") or {}) != EXPECTED_GATES:
        raise ValueError("expanded Stage1 hard gates changed")
    if tuple(config.get("model_selection_rule") or ()) != MODEL_SELECTION_RULE:
        raise ValueError("expanded Stage1 model selection changed")
    if dict(config.get("execution") or {}) != {
        "workers": 16,
        "maximum_workers": 20,
        "workers_enter_run_fingerprint": True,
        "solver_invocation_allowed": False,
    }:
        raise ValueError("expanded Stage1 execution contract changed")
    if dict(config.get("claim_boundary") or {}) != EXPECTED_CLAIM_BOUNDARY:
        raise ValueError("expanded Stage1 claim boundary changed")
    if dict(config.get("outputs") or {}) != {
        "root": DEFAULT_OUTPUT,
        "feature_manifest": "feature_manifest.json",
        "oof_predictions": "expanded_stage1_oof_predictions.jsonl",
        "evaluation_report": "expanded_stage1_router_evaluation_report.json",
        "challenger_model": "expanded_stage1_challenger.pkl",
        "challenger_manifest": "expanded_stage1_challenger_manifest.json",
    }:
        raise ValueError("expanded Stage1 outputs changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    return config_path, project_root, config


def _validate_adapter_report(
    report: Mapping[str, Any],
    *,
    pair_labels_path: Path,
    collection_report_path: Path,
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
    if folds != FOLDS:
        raise ValueError("expanded-train adapter fold identity changed")
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
        raise ValueError("expanded-train adapter claim boundary changed")
    artifacts = dict(report.get("artifacts") or {})
    pair_artifact = dict(artifacts.get("expanded_stage1_pair_labels") or {})
    h1_integrity = dict(
        dict(report.get("input_integrity") or {}).get("h1") or {}
    )
    collection_artifact = dict(h1_integrity.get("collection_report") or {})
    if (
        pair_artifact.get("file") != pair_labels_path.name
        or pair_artifact.get("sha256") != sha256_file(pair_labels_path)
        or collection_artifact.get("sha256") != sha256_file(collection_report_path)
    ):
        raise ValueError("expanded-train adapter artifact pin mismatch")
    support = dict(dict(report.get("support") or {}).get("stage1_pair") or {})
    if (
        int(support.get("active_map_count", -1)) != 16
        or int(support.get("row_count", -1)) != int(pair_artifact.get("row_count", -2))
        or support.get("support_ready") is not True
        or set(dict(support.get("label_counts") or {})) != set(ALL_LABELS)
    ):
        raise ValueError("expanded-train adapter support changed")


def _finite_number(value: Any, *, field: str) -> float:
    return base._finite_number(value, field=field)


def _arm_features(state: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    return base._arm_features(state, role)


def build_pair_example(
    label_row: Mapping[str, Any], state: Mapping[str, Any]
) -> dict[str, Any]:
    """Build one fixed 18-D, strictly pre-action expanded-train example."""
    state_id = str(label_row.get("state_occurrence_id", ""))
    if str(state.get("state_occurrence_id", "")) != state_id:
        raise ValueError(f"expanded Stage1 label/state mismatch: {state_id}")
    label = str(label_row.get("label", ""))
    if label not in ALL_LABELS:
        raise ValueError(f"unknown expanded Stage1 label: {label}")
    if (
        label_row.get("schema") != EXPANDED_LABEL_SCHEMA
        or label_row.get("research_split") != "expanded_train"
        or label_row.get("runtime_or_pp_seconds_used_in_label") is not False
        or label_row.get("sequential_design_only") is not True
        or label_row.get("training_authorized") is not True
        or label_row.get("final_claim_authorized") is not False
        or label_row.get("promotion_authorized") is not False
        or label_row.get("development_evaluation_authorized") is not False
        or label_row.get("runtime_or_ttf_claim_authorized") is not False
        or label_row.get("map_disjoint_final_confirmation_required") is not True
        or label_row.get("expanded_train") is not True
    ):
        raise ValueError(f"expanded Stage1 label trust contract changed: {state_id}")
    structural_id = str(label_row.get("structural_action_id", ""))
    v2_id = str(label_row.get("v2_action_id", ""))
    if not structural_id or not v2_id or structural_id == v2_id:
        raise ValueError(f"expanded Stage1 pair is not distinct: {state_id}")
    actions = {
        str(row.get("action_id", "")): dict(row)
        for row in list(state.get("unique_actions") or ())
    }
    if structural_id not in actions or v2_id not in actions:
        raise ValueError(f"expanded Stage1 action absent from state: {state_id}")
    structural_action = actions[structural_id]
    v2_action = actions[v2_id]
    structural_roles = tuple(
        sorted(map(str, label_row.get("structural_role_aliases") or ()))
    )
    if not structural_roles or not set(structural_roles) <= {
        "component16",
        "hotspot16",
    }:
        raise ValueError(f"expanded Stage1 structural aliases changed: {state_id}")
    role_map = {
        str(key): str(value)
        for key, value in dict(state.get("role_to_action_id") or {}).items()
    }
    if role_map.get("v2_anchor") != v2_id or any(
        role_map.get(role) != structural_id for role in structural_roles
    ):
        raise ValueError(f"expanded Stage1 role/action map changed: {state_id}")
    anchor = _arm_features(state, "v2_anchor")
    structural_arms = [_arm_features(state, role) for role in structural_roles]

    values: list[float] = []
    for name in FEATURE_NAMES:
        kind, raw = name.split(":", 1)
        if kind == "context":
            values.append(_finite_number(anchor.get(raw), field=name))
        elif kind == "delta_structural_minus_v2":
            structural_value = sum(
                _finite_number(features.get(raw), field=f"{name}/{role}")
                for role, features in zip(structural_roles, structural_arms)
            ) / len(structural_arms)
            values.append(
                structural_value
                - _finite_number(anchor.get(raw), field=f"{name}/v2")
            )
        elif kind == "relation":
            left = set(map(int, structural_action.get("agents") or ()))
            right = set(map(int, v2_action.get("agents") or ()))
            if not left or not right:
                raise ValueError(f"empty expanded Stage1 action set: {state_id}")
            intersection = len(left & right)
            union = len(left | right)
            relations = {
                "intersection_ratio": intersection / min(len(left), len(right)),
                "jaccard": intersection / union,
                "symmetric_difference_ratio": len(left ^ right) / union,
            }
            values.append(float(relations[raw]))
        else:  # pragma: no cover - fixed registration guard
            raise AssertionError(kind)
    if len(values) != len(FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise ValueError(f"expanded Stage1 feature vector invalid: {state_id}")
    registered_names = tuple(map(str, label_row.get("model_feature_names") or ()))
    registered_values = [
        _finite_number(value, field="model_feature_vector")
        for value in list(label_row.get("model_feature_vector") or ())
    ]
    if (
        registered_names != FEATURE_NAMES
        or len(registered_values) != len(values)
        or any(
            abs(left - right) > 1e-12
            for left, right in zip(registered_values, values)
        )
        or label_row.get("model_feature_direction") != "structural_minus_v2"
        or label_row.get("repair_outcome_or_runtime_used_as_model_feature")
        is not False
        or label_row.get("exact_structural_v2_sets_differ") is not True
    ):
        raise ValueError(f"expanded Stage1 registered feature contract changed: {state_id}")
    map_id = str(label_row.get("map_id", ""))
    expected_fold = next(
        (fold for fold, maps in FOLDS.items() if map_id in maps), None
    )
    if expected_fold is None or str(label_row.get("train_fold")) != expected_fold:
        raise ValueError(f"expanded Stage1 fold changed: {state_id}")
    return {
        "pair_id": str(label_row.get("pair_id", "")),
        "state_occurrence_id": state_id,
        "map_id": map_id,
        "research_split": "expanded_train",
        "train_fold": expected_fold,
        "label": label,
        "target": (
            1
            if label == "structural_win"
            else 0
            if label == "v2_win"
            else None
        ),
        "depth_band": str(label_row.get("depth_band", "")),
        "role_stratum": "+".join(structural_roles),
        "hierarchical_stratum": str(
            label_row.get("hierarchical_stratum", "")
        ),
        "features": values,
    }


def cross_validate_candidate(
    examples: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fit: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Any]
    = base._fit_estimator,
    predict: Callable[[Any, Sequence[Mapping[str, Any]]], list[float]]
    = base._predict_probability,
) -> dict[str, Any]:
    if not examples or any(
        row.get("research_split") != "expanded_train" for row in examples
    ):
        raise ValueError("expanded Stage1 OOF population changed")
    predictions: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for fold, held_maps in FOLDS.items():
        fit_rows = [
            row
            for row in examples
            if row.get("label") in DECISIVE_LABELS
            and row.get("train_fold") != fold
        ]
        held_rows = [row for row in examples if row.get("train_fold") == fold]
        fit_states = {str(row["state_occurrence_id"]) for row in fit_rows}
        held_states = {str(row["state_occurrence_id"]) for row in held_rows}
        fit_maps = {str(row["map_id"]) for row in fit_rows}
        held_map_set = {str(row["map_id"]) for row in held_rows}
        if fit_states & held_states or fit_maps & held_map_set:
            raise ValueError(f"expanded Stage1 fold leakage: {fold}")
        if held_map_set != set(held_maps):
            raise ValueError(f"expanded Stage1 held-map support changed: {fold}")
        estimator = fit(model_spec, fit_rows)
        probabilities = predict(estimator, held_rows)
        if len(probabilities) != len(held_rows):
            raise ValueError("expanded Stage1 prediction count mismatch")
        predictions.extend(
            {**row, "probability": float(probability), "model_id": model_spec["model_id"]}
            for row, probability in zip(held_rows, probabilities)
        )
        leakage_rows.append(
            {
                "fold": fold,
                "fit_map_count": len(fit_maps),
                "held_map_count": len(held_map_set),
                "fit_state_count": len(fit_states),
                "held_state_count": len(held_states),
                "state_overlap_count": 0,
                "map_overlap_count": 0,
                "leakage": False,
            }
        )
    if len(predictions) != len(examples) or {
        str(row["pair_id"]) for row in predictions
    } != {str(row["pair_id"]) for row in examples}:
        raise ValueError("expanded Stage1 OOF coverage changed")
    calibration = base.select_abstention_threshold(
        predictions, config["abstention_calibration"]
    )
    threshold = float(calibration["selected_threshold"])
    overall = base.classification_metrics(predictions, threshold)
    fold_metrics = base._group_metrics(predictions, threshold, "train_fold")
    role_metrics = base._group_metrics(predictions, threshold, "role_stratum")
    depth_metrics = base._group_metrics(predictions, threshold, "depth_band")
    checks, stability = base._candidate_gate_checks(
        overall,
        fold_metrics,
        role_metrics,
        depth_metrics,
        config["hard_gates"],
    )
    return {
        "model_id": str(model_spec["model_id"]),
        "threshold_calibration": calibration,
        "oof_metrics": overall,
        "by_fold": fold_metrics,
        "by_role": role_metrics,
        "by_depth_band": depth_metrics,
        "stability": stability,
        "hard_gate_checks": checks,
        "passed": all(checks.values()),
        "fold_leakage_audit": leakage_rows,
        "predictions": predictions,
    }


def _model_rank(result: Mapping[str, Any], index: int) -> tuple[Any, ...]:
    fold_values = [
        float(row["balanced_accuracy"])
        for row in dict(result["by_fold"]).values()
        if row.get("balanced_accuracy") is not None
    ]
    metrics = dict(result["oof_metrics"])
    return (
        bool(result["passed"]),
        min(fold_values) if len(fold_values) == len(FOLDS) else -1.0,
        float(metrics.get("balanced_accuracy") or -1.0),
        float(metrics.get("selective_accuracy") or -1.0),
        float(metrics.get("coverage") or 0.0),
        -index,
    )


def _prediction_output(
    row: Mapping[str, Any], threshold: float
) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "model_id": str(row["model_id"]),
        "pair_id": str(row["pair_id"]),
        "state_occurrence_id": str(row["state_occurrence_id"]),
        "research_split": "expanded_train",
        "map_id": str(row["map_id"]),
        "train_fold": str(row["train_fold"]),
        "label": str(row["label"]),
        "depth_band": str(row["depth_band"]),
        "role_stratum": str(row["role_stratum"]),
        "structural_win_probability": float(row["probability"]),
        "threshold": threshold,
        "decision": base._decision(float(row["probability"]), threshold),
        "ambiguous_used_for_fit": False,
        "expanded_train_oof_only": True,
        "development_partition_present": False,
        "sequential_design_only": True,
        "promotion_or_default_authorized": False,
        "final_claim_authorized": False,
    }


def _atomic_pickle(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def run_expanded_stage1_router(
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
        collection_report_path=inputs["h1_collection_report"],
    )
    states, state_integrity = base._load_states(
        inputs["h1_collection_report"], worker_count
    )
    label_rows = _read_jsonl(inputs["expanded_stage1_pair_labels"])
    pair_ids = [str(row.get("pair_id", "")) for row in label_rows]
    if not pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("expanded Stage1 pair IDs are empty or duplicated")
    examples = []
    for row in label_rows:
        state_id = str(row.get("state_occurrence_id", ""))
        if state_id not in states:
            raise ValueError(f"expanded Stage1 label has unknown state: {state_id}")
        examples.append(build_pair_example(row, states[state_id]))
    if {row["label"] for row in examples} != set(ALL_LABELS):
        raise ValueError("expanded Stage1 labels lack decisive/ambiguous support")
    if {row["map_id"] for row in examples} != set(EXPANDED_MAPS):
        raise ValueError("expanded Stage1 map support changed")
    if any(row["research_split"] != "expanded_train" for row in examples):
        raise ValueError("development/final rows entered expanded Stage1 router")

    models = list(config["models"])
    results = [
        cross_validate_candidate(examples, model, config) for model in models
    ]
    selected_index = max(
        range(len(results)),
        key=lambda index: _model_rank(results[index], index),
    )
    selected = results[selected_index]
    threshold = float(selected["threshold_calibration"]["selected_threshold"])
    fit_rows = [row for row in examples if row["label"] in DECISIVE_LABELS]
    final_estimator = base._fit_estimator(models[selected_index], fit_rows)
    passed = bool(selected["passed"])
    status = "EXPANDED_STAGE1_CHALLENGER_ONLY" if passed else "NO_GO"
    run_fingerprint = _fingerprint(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(config_path),
            "input_sha256": {
                name: sha256_file(path) for name, path in inputs.items()
            },
            "workers": worker_count,
            "fixed_map_folds": {fold: list(maps) for fold, maps in FOLDS.items()},
            "feature_names": list(FEATURE_NAMES),
        }
    )
    candidate_reports = [
        {key: value for key, value in result.items() if key != "predictions"}
        for result in results
    ]
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": status,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "workers": worker_count,
        "workers_enter_run_fingerprint": True,
        "input_integrity": {
            "registered_sha256": {
                name: sha256_file(path) for name, path in inputs.items()
            },
            **state_integrity,
            "expanded_stage1_pair_count": len(examples),
            "expanded_map_count": len({row["map_id"] for row in examples}),
            "development_row_count": 0,
            "stage2_row_count": 0,
        },
        "feature_contract": {
            "dimension": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "source": "H1 state_row pre-action arm features and exact action sets only",
            "trial_result_runtime_or_identifier_feature_count": 0,
            "depth_role_partition_are_evaluation_strata_not_inputs": True,
        },
        "fit_support": {
            "fit_labels": list(DECISIVE_LABELS),
            "decisive_row_count": len(fit_rows),
            "decisive_state_count": len(
                {row["state_occurrence_id"] for row in fit_rows}
            ),
            "ambiguous_row_count": sum(
                row["label"] == "ambiguous" for row in examples
            ),
            "ambiguous_fitted_as_hard_class": False,
            "same_state_pairs_grouped_by_fixed_map_fold": True,
            "fixed_map_fold_count": 4,
            "expanded_map_count": 16,
        },
        "candidate_models": candidate_reports,
        "model_selection_rule": list(MODEL_SELECTION_RULE),
        "selected_model_id": selected["model_id"],
        "selected_threshold": threshold,
        "hard_gates_passed": passed,
        "development_diagnostic": None,
        "development_rows_loaded": False,
        "stage2_rows_loaded": False,
        "stage2_model_fit": False,
        "model_fit_executed": True,
        "model_exported": passed and not dry_run,
        "promotion_or_default_exported": False,
        "final_claim_authorized": False,
        "next_decision": (
            "retain_expanded_stage1_challenger_pending_external_map_disjoint_confirmation"
            if passed
            else "stop_expanded_stage1_router_no_go_without_export"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
        "dry_run": dry_run,
    }
    if dry_run:
        report["model_exported"] = False
        return report

    output_root = Path(output or config["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    else:
        output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("expanded Stage1 router output already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])
    feature_manifest = {
        "schema": (
            "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_feature_manifest.v1"
        ),
        "run_fingerprint": run_fingerprint,
        "dimension": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "representation": config["feature_contract"]["representation"],
        "forbidden_inputs": list(
            config["feature_contract"]["forbidden_input_tokens"]
        ),
        "leakage_audit_passed": True,
        "development_partition_present": False,
    }
    _write_json(output_root / outputs["feature_manifest"], feature_manifest)
    oof_rows = []
    for result in results:
        candidate_threshold = float(
            result["threshold_calibration"]["selected_threshold"]
        )
        oof_rows.extend(
            _prediction_output(row, candidate_threshold)
            for row in result["predictions"]
        )
    _write_jsonl(output_root / outputs["oof_predictions"], oof_rows)
    if passed:
        model_path = output_root / outputs["challenger_model"]
        _atomic_pickle(
            model_path,
            {
                "experiment_id": EXPERIMENT_ID,
                "run_fingerprint": run_fingerprint,
                "model_id": selected["model_id"],
                "feature_names": FEATURE_NAMES,
                "threshold": threshold,
                "estimator": final_estimator,
                "status": "EXPANDED_STAGE1_CHALLENGER_ONLY",
            },
        )
        manifest = {
            "schema": MODEL_MANIFEST_SCHEMA,
            "status": "EXPANDED_STAGE1_CHALLENGER_ONLY",
            "run_fingerprint": run_fingerprint,
            "model_id": selected["model_id"],
            "threshold": threshold,
            "model_file": model_path.name,
            "model_sha256": sha256_file(model_path),
            "promotion_or_default": False,
            "final_claim_authorized": False,
            "external_map_disjoint_confirmation_required": True,
        }
        _write_json(output_root / outputs["challenger_manifest"], manifest)
        report["model_artifact"] = manifest
    else:
        report["model_artifact"] = None
    _write_json(output_root / outputs["evaluation_report"], report)
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
    "REPORT_SCHEMA",
    "build_pair_example",
    "cross_validate_candidate",
    "run_expanded_stage1_router",
    "validate_config",
]
