"""Stage1-only abstaining router for the compact-flow sequential cohort.

The router learns only ``V2 vs distinct structural action``.  It consumes the
frozen compact-flow Stage1 pair labels and pre-action H1 state/action-generation
features.  Ambiguous rows calibrate abstention but are never fitted as a third
class.  Four fixed map folds keep every state and all of its pairs together.

This identity cannot read Stage2 rows, export a default/promotion, or support a
runtime/TTF/final claim.  A passing artifact is only a sequential challenger.
"""

from __future__ import annotations

import collections
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_hierarchical_ch_compact_flow_h1_collection_v1 import (
    H1_MANIFEST_ROW_SCHEMA,
    H1_STATE_SCHEMA,
)
from experiments.stride_hierarchical_ch_compact_flow_labels_readiness_v1 import (
    DEVELOPMENT_MAPS,
    READINESS_SCHEMA,
    STAGE1_PAIR_SCHEMA,
    TRAIN_MAP_FOLDS,
    TRAIN_MAPS,
)


CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_stage1_router_config.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_stage1_router_report.v1"
)
PREDICTION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_stage1_router_prediction.v1"
)
MODEL_MANIFEST_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_stage1_challenger_manifest.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_stage1_router_v1"
DEFAULT_CONFIG = "configs/stride_hierarchical_ch_compact_flow_stage1_router_v1.json"
DEFAULT_OUTPUT = "build/stride-hierarchical-ch-compact-flow-stage1-router-v1"

DECISIVE_LABELS = ("structural_win", "v2_win")
ALL_LABELS = (*DECISIVE_LABELS, "ambiguous")
FEATURE_NAMES = (
    "context:state.agent_count",
    "context:state.colliding_pairs",
    "context:state.conflict_edge_density",
    "context:state.conflict_event_count",
    "context:state.conflicting_agent_ratio",
    "context:state.degree_max",
    "context:state.largest_component_ratio",
    "context:state.path_wait_ratio_mean",
    "delta:realized.incident_conflict_coverage",
    "delta:realized.internal_conflict_coverage",
    "delta:realized.boundary_conflict_edges",
    "delta:realized.conflict_degree_mean",
    "delta:realized.delay_mean",
    "delta:realized.path_overlap_mean",
    "delta:realized.path_wait_ratio_mean",
    "relation:intersection_ratio",
    "relation:jaccard",
    "relation:symmetric_difference_ratio",
)
FOLDS = {fold: tuple(maps) for fold, maps in TRAIN_MAP_FOLDS.items()}
MODEL_IDS = ("regularized_logistic", "low_capacity_hist_gradient_boosting")
THRESHOLD_GRID = (0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01)
SELECTION_RULE = (
    "feasible_constraints_first",
    "maximum_decisive_balanced_accuracy_with_abstentions_as_errors",
    "maximum_selective_accuracy",
    "maximum_decisive_coverage",
    "lower_threshold",
)
MODEL_SELECTION_RULE = (
    "all_hard_gates_first",
    "maximum_minimum_fold_balanced_accuracy",
    "maximum_oof_balanced_accuracy",
    "maximum_oof_selective_accuracy",
    "maximum_oof_coverage",
    "config_order",
)


def _contained(root: Path, value: str, *, field: str) -> Path:
    portable = PurePosixPath(str(value).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"{field} must be a contained project-relative path")
    result = (root / Path(*portable.parts)).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes project root") from error
    return result


def _require_sha(value: Any, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{field} is not a lowercase SHA256")
    return digest


def _registered_input(root: Path, specification: Mapping[str, Any], *, field: str) -> Path:
    if set(specification) != {"path", "sha256"}:
        raise ValueError(f"{field} input registration changed")
    path = _contained(root, str(specification["path"]), field=field)
    if not path.is_file():
        raise ValueError(f"{field} input is missing")
    expected = _require_sha(specification["sha256"], field=f"{field}.sha256")
    if sha256_file(path) != expected:
        raise ValueError(f"{field} SHA256 mismatch")
    return path


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_sequential_stage1_only_challenger"
    ):
        raise ValueError("compact-flow Stage1 router identity changed")
    if set(config.get("inputs") or {}) != {
        "labels_readiness_config",
        "readiness_report",
        "stage1_pair_labels",
        "h1_collection_report",
    }:
        raise ValueError("compact-flow Stage1 router input registry changed")
    semantics = dict(config.get("input_semantics") or {})
    if semantics != {
        "required_readiness_schema": READINESS_SCHEMA,
        "required_label_schema": STAGE1_PAIR_SCHEMA,
        "required_state_schema": H1_STATE_SCHEMA,
        "require_train_stage1_label_support_trainable": True,
        "require_train_stage2_label_support_trainable": False,
        "fit_labels": list(DECISIVE_LABELS),
        "ambiguous_use": "abstention_threshold_calibration_only",
        "stage2_rows_loaded": False,
        "stage2_model_fit": False,
    }:
        raise ValueError("compact-flow Stage1-only input semantics changed")
    folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(config.get("train_map_folds") or {}).items()
    }
    if folds != FOLDS:
        raise ValueError("compact-flow Stage1 fixed map folds changed")
    folded = [map_id for maps in folds.values() for map_id in maps]
    if len(folded) != len(set(folded)) or set(folded) != set(TRAIN_MAPS):
        raise ValueError("compact-flow Stage1 map folds overlap or omit maps")
    features = dict(config.get("feature_contract") or {})
    if (
        features.get("representation")
        != "structural_minus_exact_v2_anchor_with_pre_action_context"
        or tuple(features.get("feature_names") or ()) != FEATURE_NAMES
        or features.get("structural_alias_aggregation")
        != "arithmetic_mean_for_exact_set_aliases"
        or set(features.get("categorical_fields_are_evaluation_strata_not_model_inputs") or ())
        != {"depth_band", "canonical_partition", "structural_role_aliases"}
        or float(features.get("missing_value", math.nan)) != 0.0
        or features.get("identifier_values_used_as_features") is not False
        or features.get("repair_outcome_or_runtime_used_as_features") is not False
    ):
        raise ValueError("compact-flow Stage1 feature contract changed")
    forbidden = set(map(str, features.get("forbidden_input_tokens") or ()))
    if not {
        "map_id", "task_id", "solver_seed", "trial", "outcome", "result",
        "runtime", "pp_seconds", "no_progress", "rollback", "time_limit",
        "normalized_conflict_reduction",
    } <= forbidden:
        raise ValueError("compact-flow Stage1 forbidden feature tokens weakened")
    for name in FEATURE_NAMES:
        lowered = name.lower()
        if any(token in lowered for token in forbidden):
            raise ValueError(f"registered feature violates leakage contract: {name}")
    models = list(config.get("models") or ())
    if [row.get("model_id") for row in models] != list(MODEL_IDS):
        raise ValueError("compact-flow Stage1 candidate models changed")
    if models != [
        {
            "model_id": "regularized_logistic",
            "class": "sklearn.linear_model.LogisticRegression",
            "preprocessing": "StandardScaler",
            "parameters": {
                "C": 0.25, "class_weight": None, "max_iter": 1000,
                "random_state": 20260826, "solver": "liblinear",
            },
        },
        {
            "model_id": "low_capacity_hist_gradient_boosting",
            "class": "sklearn.ensemble.HistGradientBoostingClassifier",
            "preprocessing": "none",
            "parameters": {
                "early_stopping": False, "l2_regularization": 1.0,
                "learning_rate": 0.05, "max_iter": 60,
                "max_leaf_nodes": 5, "min_samples_leaf": 8,
                "random_state": 20260826,
            },
        },
    ]:
        raise ValueError("compact-flow Stage1 model parameters changed")
    if dict(config.get("sample_weighting") or {}) != {
        "unit": "state_occurrence_id",
        "equal_total_weight_per_state": True,
        "balance_decisive_classes_after_state_weighting": True,
    }:
        raise ValueError("compact-flow Stage1 sample weighting changed")
    calibration = dict(config.get("abstention_calibration") or {})
    if (
        tuple(map(float, calibration.get("threshold_grid") or ())) != THRESHOLD_GRID
        or float(calibration.get("maximum_ambiguous_selection_rate", -1)) != 0.25
        or float(calibration.get("minimum_decisive_coverage", -1)) != 0.25
        or tuple(calibration.get("selection_rule") or ()) != SELECTION_RULE
    ):
        raise ValueError("compact-flow Stage1 abstention calibration changed")
    expected_gates = {
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
    if dict(config.get("hard_gates") or {}) != expected_gates:
        raise ValueError("compact-flow Stage1 hard gates changed")
    if tuple(config.get("model_selection_rule") or ()) != MODEL_SELECTION_RULE:
        raise ValueError("compact-flow Stage1 model selection changed")
    if dict(config.get("execution") or {}) != {
        "workers": 16,
        "maximum_workers": 20,
        "workers_enter_run_fingerprint": True,
        "solver_invocation_allowed": False,
    }:
        raise ValueError("compact-flow Stage1 execution contract changed")
    if dict(config.get("claim_boundary") or {}) != {
        "stage1_only": True,
        "stage2_training_allowed": False,
        "sequential_design_only": True,
        "development_is_diagnostic_only": True,
        "passing_status": "SEQUENTIAL_CHALLENGER_ONLY",
        "failure_status": "NO_GO",
        "promotion_or_default_export_allowed": False,
        "runtime_or_ttf_claim_authorized": False,
        "final_map_disjoint_confirmation_required": True,
    }:
        raise ValueError("compact-flow Stage1 claim boundary changed")
    expected_outputs = {
        "root": DEFAULT_OUTPUT,
        "feature_manifest": "feature_manifest.json",
        "oof_predictions": "stage1_oof_predictions.jsonl",
        "development_predictions": "stage1_development_predictions.jsonl",
        "evaluation_report": "stage1_router_evaluation_report.json",
        "sequential_challenger_model": "sequential_challenger.pkl",
        "sequential_challenger_manifest": "sequential_challenger_manifest.json",
    }
    if dict(config.get("outputs") or {}) != expected_outputs:
        raise ValueError("compact-flow Stage1 outputs changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    return config_path, project_root, config


def _validate_readiness(readiness: Mapping[str, Any]) -> None:
    if readiness.get("schema") != READINESS_SCHEMA or readiness.get("complete") is not True:
        raise ValueError("compact-flow readiness report is not complete")
    train = dict(dict(readiness.get("splits") or {}).get("train") or {})
    stage1 = dict(train.get("stage1") or {})
    stage2 = dict(train.get("stage2") or {})
    if stage1.get("label_support_trainable") is not True:
        raise ValueError("train.stage1 label_support_trainable must be true")
    if stage2.get("label_support_trainable") is not False:
        raise ValueError("train.stage2 label_support_trainable must remain false")
    observed_folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(readiness.get("fixed_train_map_folds") or {}).items()
    }
    if observed_folds != FOLDS or readiness.get("development_has_training_folds") is not False:
        raise ValueError("readiness fold boundary changed")
    if (
        readiness.get("scientific_status") != "sequential_labels_only_readiness"
        or readiness.get("development_is_sequential_diagnostic_only") is not True
        or readiness.get("final_claim_authorized") is not False
        or readiness.get("runtime_or_ttf_claim_authorized") is not False
        or readiness.get("map_disjoint_confirmation_required") is not True
    ):
        raise ValueError("readiness claim boundary changed")


def _safe_state_path(collection_root: Path, relative: str) -> Path:
    portable = PurePosixPath(str(relative).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError("H1 state path escapes collection root")
    path = (collection_root / Path(*portable.parts)).resolve()
    try:
        path.relative_to(collection_root.resolve())
    except ValueError as error:
        raise ValueError("H1 state path escapes collection root") from error
    return path


def _load_state_payload(
    collection_root: Path, manifest_row: Mapping[str, Any]
) -> tuple[str, dict[str, Any]]:
    if manifest_row.get("schema") != H1_MANIFEST_ROW_SCHEMA:
        raise ValueError("H1 manifest row schema changed")
    state_id = str(manifest_row.get("state_occurrence_id", ""))
    path = _safe_state_path(collection_root, str(manifest_row.get("state_file", "")))
    if not path.is_file() or sha256_file(path) != _require_sha(
        manifest_row.get("state_sha256"), field="state_sha256"
    ):
        raise ValueError(f"H1 state payload SHA256 mismatch: {state_id}")
    payload = _read_json(path)
    if (
        payload.get("schema") != H1_STATE_SCHEMA
        or payload.get("complete") is not True
        or str(payload.get("state_occurrence_id", "")) != state_id
        or payload.get("runtime_used_in_label") is not False
        or payload.get("sequential_design_only") is not True
        or payload.get("final_claim_authorized") is not False
    ):
        raise ValueError(f"H1 state trust contract changed: {state_id}")
    state = dict(payload.get("state_row") or {})
    if (
        state.get("candidate_repair_actions_executed") is not False
        or state.get("outcome_filtering") is not False
        or list(state.get("target_outcome_fields_read") or ())
        or state.get("sequential_design_only") is not True
    ):
        raise ValueError(f"H1 state is not pre-action/result-blind: {state_id}")
    return state_id, state


def _load_states(collection_report_path: Path, workers: int) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    report = _read_json(collection_report_path)
    if (
        report.get("status") != "complete"
        or report.get("complete") is not True
        or report.get("sequential_design_only") is not True
        or report.get("final_claim_authorized") is not False
        or report.get("runtime_or_ttf_claim_authorized") is not False
    ):
        raise ValueError("H1 collection report trust contract changed")
    collection_root = collection_report_path.parent.resolve()
    artifact = dict(dict(report.get("artifacts") or {}).get("label_source_state_manifest") or {})
    if artifact.get("file") != "h1_state_manifest.jsonl" or artifact.get("state_schema") != H1_STATE_SCHEMA:
        raise ValueError("H1 state manifest identity changed")
    manifest_path = _safe_state_path(collection_root, str(artifact.get("file", "")))
    if not manifest_path.is_file() or sha256_file(manifest_path) != _require_sha(
        artifact.get("sha256"), field="state_manifest.sha256"
    ):
        raise ValueError("H1 state manifest SHA256 mismatch")
    rows = _read_jsonl(manifest_path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(lambda row: _load_state_payload(collection_root, row), rows))
    states = dict(loaded)
    if len(states) != len(loaded) or len(states) != int(report.get("observed_valid_state_count", -1)):
        raise ValueError("H1 state manifest count/identity mismatch")
    return states, {
        "collection_report_sha256": sha256_file(collection_report_path),
        "state_manifest_sha256": sha256_file(manifest_path),
        "state_count": len(states),
        "workers": workers,
    }


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric pre-action feature: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite pre-action feature: {field}")
    return result


def _arm_features(state: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    arm = dict(dict(state.get("arms") or {}).get(role) or {})
    features = dict(arm.get("features") or {})
    if not features:
        raise ValueError(f"missing pre-action features for role {role}")
    return features


def build_pair_example(label_row: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    """Build one fixed 18-D result-blind pair example."""
    state_id = str(label_row.get("state_occurrence_id", ""))
    if str(state.get("state_occurrence_id", "")) != state_id:
        raise ValueError(f"Stage1 label/state identity mismatch: {state_id}")
    label = str(label_row.get("label", ""))
    if label not in ALL_LABELS:
        raise ValueError(f"unknown Stage1 label: {label}")
    if (
        label_row.get("schema") != STAGE1_PAIR_SCHEMA
        or label_row.get("runtime_or_pp_seconds_used_in_label") is not False
        or label_row.get("sequential_design_only") is not True
        or label_row.get("training_authorized") is not False
    ):
        raise ValueError(f"Stage1 label trust contract changed: {state_id}")
    structural_id = str(label_row.get("structural_action_id", ""))
    v2_id = str(label_row.get("v2_action_id", ""))
    if not structural_id or not v2_id or structural_id == v2_id:
        raise ValueError(f"Stage1 pair is not a distinct structural/V2 action: {state_id}")
    actions = {
        str(row.get("action_id", "")): dict(row)
        for row in list(state.get("unique_actions") or ())
    }
    if structural_id not in actions or v2_id not in actions:
        raise ValueError(f"Stage1 pair action is absent from H1 state: {state_id}")
    structural_action = actions[structural_id]
    v2_action = actions[v2_id]
    structural_roles = tuple(sorted(map(str, label_row.get("structural_role_aliases") or ())))
    if not structural_roles or not set(structural_roles) <= {"component16", "hotspot16"}:
        raise ValueError(f"Stage1 structural role aliases changed: {state_id}")
    role_map = {str(k): str(v) for k, v in dict(state.get("role_to_action_id") or {}).items()}
    if role_map.get("v2_anchor") != v2_id or any(role_map.get(role) != structural_id for role in structural_roles):
        raise ValueError(f"Stage1 role/action map changed: {state_id}")
    anchor = _arm_features(state, "v2_anchor")
    structural_arms = [_arm_features(state, role) for role in structural_roles]

    values: list[float] = []
    for name in FEATURE_NAMES:
        kind, raw = name.split(":", 1)
        if kind == "context":
            values.append(_finite_number(anchor.get(raw), field=name))
        elif kind == "delta":
            structural_value = sum(
                _finite_number(features.get(raw), field=f"{name}/{role}")
                for role, features in zip(structural_roles, structural_arms)
            ) / len(structural_arms)
            values.append(structural_value - _finite_number(anchor.get(raw), field=f"{name}/v2"))
        elif kind == "relation":
            left = set(map(int, structural_action.get("agents") or ()))
            right = set(map(int, v2_action.get("agents") or ()))
            if not left or not right:
                raise ValueError(f"empty Stage1 action set: {state_id}")
            intersection = len(left & right)
            union = len(left | right)
            relations = {
                "intersection_ratio": intersection / min(len(left), len(right)),
                "jaccard": intersection / union,
                "symmetric_difference_ratio": len(left ^ right) / union,
            }
            values.append(float(relations[raw]))
        else:  # pragma: no cover - fixed schema guard
            raise AssertionError(kind)
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Stage1 feature vector invalid: {state_id}")
    split = str(label_row.get("research_split", ""))
    map_id = str(label_row.get("map_id", ""))
    fold = label_row.get("train_fold")
    if split == "train":
        expected_fold = next((name for name, maps in FOLDS.items() if map_id in maps), None)
        if expected_fold is None or str(fold) != expected_fold:
            raise ValueError(f"Stage1 train fold changed: {state_id}")
        fold = expected_fold
    elif split == "development":
        if map_id not in DEVELOPMENT_MAPS or fold is not None:
            raise ValueError(f"Stage1 development split/fold changed: {state_id}")
    else:
        raise ValueError(f"unknown Stage1 research split: {split}")
    return {
        "pair_id": str(label_row.get("pair_id", "")),
        "state_occurrence_id": state_id,
        "map_id": map_id,
        "research_split": split,
        "train_fold": fold,
        "label": label,
        "target": 1 if label == "structural_win" else 0 if label == "v2_win" else None,
        "depth_band": str(label_row.get("depth_band", "")),
        "role_stratum": "+".join(structural_roles),
        "hierarchical_stratum": str(label_row.get("hierarchical_stratum", "")),
        "features": values,
    }


def _state_class_balanced_weights(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    decisive = [row for row in rows if row.get("target") in {0, 1}]
    if len(decisive) != len(rows) or {int(row["target"]) for row in rows} != {0, 1}:
        raise ValueError("fit rows must contain both decisive classes and no ambiguous rows")
    by_state = collections.Counter(str(row["state_occurrence_id"]) for row in rows)
    weights = [1.0 / by_state[str(row["state_occurrence_id"])] for row in rows]
    totals = {
        target: sum(weight for weight, row in zip(weights, rows) if int(row["target"]) == target)
        for target in (0, 1)
    }
    if min(totals.values()) <= 0.0:
        raise ValueError("decisive class has zero state-balanced weight")
    weights = [weight / totals[int(row["target"])] for weight, row in zip(weights, rows)]
    scale = len(weights) / sum(weights)
    return [weight * scale for weight in weights]


def _fit_estimator(model_spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Any:
    import numpy as np

    values = np.asarray([row["features"] for row in rows], dtype=np.float64)
    labels = np.asarray([int(row["target"]) for row in rows], dtype=np.int8)
    weights = np.asarray(_state_class_balanced_weights(rows), dtype=np.float64)
    model_id = str(model_spec["model_id"])
    parameters = dict(model_spec["parameters"])
    if model_id == "regularized_logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(**parameters))]
        )
        estimator.fit(values, labels, model__sample_weight=weights)
        return estimator
    if model_id == "low_capacity_hist_gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        estimator = HistGradientBoostingClassifier(**parameters)
        estimator.fit(values, labels, sample_weight=weights)
        return estimator
    raise ValueError(f"unknown Stage1 model: {model_id}")


def _predict_probability(estimator: Any, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    if not rows:
        return []
    import numpy as np

    probabilities = estimator.predict_proba(
        np.asarray([row["features"] for row in rows], dtype=np.float64)
    )[:, 1]
    values = [float(value) for value in probabilities]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
        raise ValueError("Stage1 estimator returned invalid probabilities")
    return values


def _decision(probability: float, threshold: float) -> str:
    confidence = max(probability, 1.0 - probability)
    if confidence + 1e-12 < threshold:
        return "abstain_to_v2"
    return "structural" if probability >= 0.5 else "v2"


def classification_metrics(predictions: Sequence[Mapping[str, Any]], threshold: float) -> dict[str, Any]:
    decisive = [row for row in predictions if row.get("label") in DECISIVE_LABELS]
    ambiguous = [row for row in predictions if row.get("label") == "ambiguous"]
    selected = [row for row in decisive if _decision(float(row["probability"]), threshold) != "abstain_to_v2"]
    correct = [
        row for row in selected
        if (_decision(float(row["probability"]), threshold) == "structural")
        == (row.get("label") == "structural_win")
    ]
    recalls: dict[str, float | None] = {}
    for label, decision in (("structural_win", "structural"), ("v2_win", "v2")):
        truth = [row for row in decisive if row.get("label") == label]
        recalls[label] = (
            sum(_decision(float(row["probability"]), threshold) == decision for row in truth) / len(truth)
            if truth else None
        )
    present = [value for value in recalls.values() if value is not None]
    ambiguous_selected = sum(
        _decision(float(row["probability"]), threshold) != "abstain_to_v2"
        for row in ambiguous
    )
    return {
        "threshold": threshold,
        "decisive_row_count": len(decisive),
        "ambiguous_row_count": len(ambiguous),
        "selected_decisive_count": len(selected),
        "correct_selected_count": len(correct),
        "balanced_accuracy": sum(present) / len(present) if len(present) == 2 else None,
        "class_recall": recalls,
        "coverage": len(selected) / len(decisive) if decisive else 0.0,
        "selective_accuracy": len(correct) / len(selected) if selected else None,
        "ambiguous_selection_rate": ambiguous_selected / len(ambiguous) if ambiguous else 0.0,
    }


def select_abstention_threshold(
    predictions: Sequence[Mapping[str, Any]], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    maximum_ambiguous = float(calibration["maximum_ambiguous_selection_rate"])
    minimum_coverage = float(calibration["minimum_decisive_coverage"])
    candidates = [classification_metrics(predictions, threshold) for threshold in THRESHOLD_GRID]
    for row in candidates:
        row["feasible"] = (
            float(row["ambiguous_selection_rate"]) <= maximum_ambiguous + 1e-12
            and float(row["coverage"]) >= minimum_coverage - 1e-12
        )
    feasible = [row for row in candidates if row["feasible"]]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                float(row["balanced_accuracy"] or -1.0),
                float(row["selective_accuracy"] or -1.0),
                float(row["coverage"]),
                -float(row["threshold"]),
            ),
        )
        status = "feasible_train_oof_threshold"
    else:
        selected = next(row for row in candidates if float(row["threshold"]) == 1.01)
        status = "no_feasible_threshold_abstain_all"
    return {
        "status": status,
        "selected_threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "grid": candidates,
        "ambiguous_fitted_as_class": False,
        "calibration_population": "train_oof_only",
    }


def _group_metrics(
    predictions: Sequence[Mapping[str, Any]], threshold: float, field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in predictions:
        groups[str(row[field])].append(row)
    return {name: classification_metrics(rows, threshold) for name, rows in sorted(groups.items())}


def _spread(groups: Mapping[str, Mapping[str, Any]], minimum_rows: int) -> tuple[float | None, list[str]]:
    supported = {
        name: float(metrics["balanced_accuracy"])
        for name, metrics in groups.items()
        if int(metrics["decisive_row_count"]) >= minimum_rows
        and metrics.get("balanced_accuracy") is not None
    }
    if len(supported) < 2:
        return None, sorted(supported)
    return max(supported.values()) - min(supported.values()), sorted(supported)


def _candidate_gate_checks(
    overall: Mapping[str, Any],
    folds: Mapping[str, Mapping[str, Any]],
    roles: Mapping[str, Mapping[str, Any]],
    depths: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    minimum_rows = int(gates["minimum_rows_for_stability_group"])
    fold_values = [float(row["balanced_accuracy"]) for row in folds.values() if row.get("balanced_accuracy") is not None]
    fold_coverages = [float(row["coverage"]) for row in folds.values()]
    fold_spread = max(fold_values) - min(fold_values) if len(fold_values) == len(FOLDS) else None
    role_spread, supported_roles = _spread(roles, minimum_rows)
    depth_spread, supported_depths = _spread(depths, minimum_rows)
    recalls = dict(overall["class_recall"])
    checks = {
        "minimum_oof_balanced_accuracy": overall.get("balanced_accuracy") is not None and float(overall["balanced_accuracy"]) >= float(gates["minimum_oof_balanced_accuracy"]),
        "minimum_each_class_recall": all(recalls.get(label) is not None and float(recalls[label]) >= float(gates["minimum_each_class_recall"]) for label in DECISIVE_LABELS),
        "minimum_oof_coverage": float(overall["coverage"]) >= float(gates["minimum_oof_coverage"]),
        "minimum_oof_selective_accuracy": overall.get("selective_accuracy") is not None and float(overall["selective_accuracy"]) >= float(gates["minimum_oof_selective_accuracy"]),
        "maximum_ambiguous_selection_rate": float(overall["ambiguous_selection_rate"]) <= float(gates["maximum_ambiguous_selection_rate"]),
        "minimum_each_fold_balanced_accuracy": len(fold_values) == len(FOLDS) and min(fold_values) >= float(gates["minimum_each_fold_balanced_accuracy"]),
        "minimum_each_fold_coverage": len(fold_coverages) == len(FOLDS) and min(fold_coverages) >= float(gates["minimum_each_fold_coverage"]),
        "maximum_fold_balanced_accuracy_spread": fold_spread is not None and fold_spread <= float(gates["maximum_fold_balanced_accuracy_spread"]),
        "maximum_supported_role_balanced_accuracy_spread": role_spread is not None and role_spread <= float(gates["maximum_supported_role_balanced_accuracy_spread"]),
        "maximum_supported_depth_balanced_accuracy_spread": depth_spread is not None and depth_spread <= float(gates["maximum_supported_depth_balanced_accuracy_spread"]),
    }
    stability = {
        "fold_balanced_accuracy_spread": fold_spread,
        "role_balanced_accuracy_spread": role_spread,
        "depth_balanced_accuracy_spread": depth_spread,
        "supported_role_groups": supported_roles,
        "supported_depth_groups": supported_depths,
    }
    return checks, stability


def cross_validate_candidate(
    examples: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fit: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Any] = _fit_estimator,
    predict: Callable[[Any, Sequence[Mapping[str, Any]]], list[float]] = _predict_probability,
) -> dict[str, Any]:
    train_rows = [row for row in examples if row.get("research_split") == "train"]
    predictions: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for fold, held_maps in FOLDS.items():
        fit_rows = [row for row in train_rows if row.get("label") in DECISIVE_LABELS and row.get("train_fold") != fold]
        held_rows = [row for row in train_rows if row.get("train_fold") == fold]
        fit_states = {str(row["state_occurrence_id"]) for row in fit_rows}
        held_states = {str(row["state_occurrence_id"]) for row in held_rows}
        fit_maps = {str(row["map_id"]) for row in fit_rows}
        held_map_set = {str(row["map_id"]) for row in held_rows}
        leakage = bool(fit_states & held_states or fit_maps & held_map_set)
        if leakage or held_map_set != set(held_maps):
            raise ValueError(f"Stage1 group/map fold leakage: {fold}")
        estimator = fit(model_spec, fit_rows)
        probabilities = predict(estimator, held_rows)
        if len(probabilities) != len(held_rows):
            raise ValueError("Stage1 prediction count mismatch")
        for row, probability in zip(held_rows, probabilities):
            predictions.append({**row, "probability": float(probability), "model_id": model_spec["model_id"]})
        leakage_rows.append({
            "fold": fold,
            "fit_map_count": len(fit_maps),
            "held_map_count": len(held_map_set),
            "fit_state_count": len(fit_states),
            "held_state_count": len(held_states),
            "state_overlap_count": len(fit_states & held_states),
            "map_overlap_count": len(fit_maps & held_map_set),
            "leakage": False,
        })
    if len(predictions) != len(train_rows) or {str(row["pair_id"]) for row in predictions} != {str(row["pair_id"]) for row in train_rows}:
        raise ValueError("Stage1 OOF prediction coverage changed")
    calibration = select_abstention_threshold(predictions, config["abstention_calibration"])
    threshold = float(calibration["selected_threshold"])
    overall = classification_metrics(predictions, threshold)
    fold_metrics = _group_metrics(predictions, threshold, "train_fold")
    role_metrics = _group_metrics(predictions, threshold, "role_stratum")
    depth_metrics = _group_metrics(predictions, threshold, "depth_band")
    checks, stability = _candidate_gate_checks(
        overall, fold_metrics, role_metrics, depth_metrics, config["hard_gates"]
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


def _prediction_output(row: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "model_id": str(row["model_id"]),
        "pair_id": str(row["pair_id"]),
        "state_occurrence_id": str(row["state_occurrence_id"]),
        "research_split": str(row["research_split"]),
        "map_id": str(row["map_id"]),
        "train_fold": row.get("train_fold"),
        "label": str(row["label"]),
        "depth_band": str(row["depth_band"]),
        "role_stratum": str(row["role_stratum"]),
        "structural_win_probability": float(row["probability"]),
        "threshold": threshold,
        "decision": _decision(float(row["probability"]), threshold),
        "ambiguous_used_for_fit": False,
        "sequential_design_only": True,
        "final_claim_authorized": False,
    }


def _atomic_pickle(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def run_stage1_router(
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
        name: _registered_input(project_root, specification, field=name)
        for name, specification in dict(config["inputs"]).items()
    }
    readiness = _read_json(inputs["readiness_report"])
    _validate_readiness(readiness)
    readiness_inputs = dict(readiness.get("inputs") or {})
    if (
        dict(readiness_inputs.get("stage1_pair_labels") or {}).get("sha256")
        != sha256_file(inputs["stage1_pair_labels"])
    ):
        raise ValueError("readiness/stage1_pair_labels pin mismatch")
    states, state_integrity = _load_states(inputs["h1_collection_report"], worker_count)
    label_rows = _read_jsonl(inputs["stage1_pair_labels"])
    pair_ids = [str(row.get("pair_id", "")) for row in label_rows]
    if not pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Stage1 pair label IDs are empty or duplicated")
    examples = []
    for row in label_rows:
        state_id = str(row.get("state_occurrence_id", ""))
        if state_id not in states:
            raise ValueError(f"Stage1 label references unknown H1 state: {state_id}")
        examples.append(build_pair_example(row, states[state_id]))
    train_rows = [row for row in examples if row["research_split"] == "train"]
    development_rows = [row for row in examples if row["research_split"] == "development"]
    if {row["label"] for row in train_rows} != set(ALL_LABELS):
        raise ValueError("Stage1 train labels do not contain both decisive directions and ambiguous")
    models = list(config["models"])
    results = [cross_validate_candidate(examples, model, config) for model in models]
    selected_index = max(range(len(results)), key=lambda index: _model_rank(results[index], index))
    selected = results[selected_index]
    selected_model_spec = models[selected_index]
    threshold = float(selected["threshold_calibration"]["selected_threshold"])

    fit_rows = [row for row in train_rows if row["label"] in DECISIVE_LABELS]
    final_estimator = _fit_estimator(selected_model_spec, fit_rows)
    development_probabilities = _predict_probability(final_estimator, development_rows)
    development_predictions = [
        {**row, "probability": probability, "model_id": selected["model_id"]}
        for row, probability in zip(development_rows, development_probabilities)
    ]
    development_metrics = {
        "overall": classification_metrics(development_predictions, threshold),
        "by_role": _group_metrics(development_predictions, threshold, "role_stratum"),
        "by_depth_band": _group_metrics(development_predictions, threshold, "depth_band"),
        "used_for_model_or_threshold_selection": False,
        "scientific_role": "sequential_diagnostic_only",
    }
    passed = bool(selected["passed"])
    status = "SEQUENTIAL_CHALLENGER_ONLY" if passed else "NO_GO"
    run_fingerprint = _fingerprint({
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "workers": worker_count,
        "feature_names": list(FEATURE_NAMES),
    })
    candidate_reports = []
    for result in results:
        candidate_reports.append({key: value for key, value in result.items() if key != "predictions"})
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": status,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "workers": worker_count,
        "workers_enter_run_fingerprint": True,
        "input_integrity": {
            "registered_sha256": {name: sha256_file(path) for name, path in inputs.items()},
            **state_integrity,
            "stage1_pair_count": len(examples),
            "train_pair_count": len(train_rows),
            "development_pair_count": len(development_rows),
        },
        "readiness_preconditions": {
            "train_stage1_label_support_trainable": True,
            "train_stage2_label_support_trainable": False,
            "stage2_rows_loaded": False,
            "stage2_model_fit": False,
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
            "decisive_train_row_count": len(fit_rows),
            "decisive_train_state_count": len({row["state_occurrence_id"] for row in fit_rows}),
            "ambiguous_train_row_count": sum(row["label"] == "ambiguous" for row in train_rows),
            "ambiguous_fitted_as_hard_class": False,
            "same_state_pairs_grouped_by_fixed_map_fold": True,
        },
        "candidate_models": candidate_reports,
        "model_selection_rule": list(MODEL_SELECTION_RULE),
        "selected_model_id": selected["model_id"],
        "selected_threshold": threshold,
        "hard_gates_passed": passed,
        "development_diagnostic": development_metrics,
        "model_fit_executed": True,
        "model_exported": passed and not dry_run,
        "promotion_or_default_exported": False,
        "next_decision": (
            "retain_as_sequential_challenger_pending_new_map_disjoint_confirmation"
            if passed
            else "stop_stage1_router_no_go_without_export"
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
        raise ValueError("Stage1 router output already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])
    feature_manifest = {
        "schema": "lns2.stride.hierarchical_ch_compact_flow_stage1_feature_manifest.v1",
        "run_fingerprint": run_fingerprint,
        "dimension": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "representation": config["feature_contract"]["representation"],
        "forbidden_inputs": list(config["feature_contract"]["forbidden_input_tokens"]),
        "leakage_audit_passed": True,
    }
    _write_json(output_root / outputs["feature_manifest"], feature_manifest)
    oof_rows = []
    for result in results:
        candidate_threshold = float(result["threshold_calibration"]["selected_threshold"])
        oof_rows.extend(_prediction_output(row, candidate_threshold) for row in result["predictions"])
    _write_jsonl(output_root / outputs["oof_predictions"], oof_rows)
    _write_jsonl(
        output_root / outputs["development_predictions"],
        [_prediction_output(row, threshold) for row in development_predictions],
    )
    if passed:
        model_path = output_root / outputs["sequential_challenger_model"]
        _atomic_pickle(model_path, {
            "experiment_id": EXPERIMENT_ID,
            "run_fingerprint": run_fingerprint,
            "model_id": selected["model_id"],
            "feature_names": FEATURE_NAMES,
            "threshold": threshold,
            "estimator": final_estimator,
            "status": "SEQUENTIAL_CHALLENGER_ONLY",
        })
        manifest = {
            "schema": MODEL_MANIFEST_SCHEMA,
            "status": "SEQUENTIAL_CHALLENGER_ONLY",
            "run_fingerprint": run_fingerprint,
            "model_id": selected["model_id"],
            "threshold": threshold,
            "model_file": model_path.name,
            "model_sha256": sha256_file(model_path),
            "promotion_or_default": False,
            "final_map_disjoint_confirmation_required": True,
        }
        _write_json(output_root / outputs["sequential_challenger_manifest"], manifest)
        report["model_artifact"] = manifest
    else:
        report["model_artifact"] = None
    _write_json(output_root / outputs["evaluation_report"], report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "EXPERIMENT_ID",
    "FEATURE_NAMES",
    "REPORT_SCHEMA",
    "build_pair_example",
    "classification_metrics",
    "cross_validate_candidate",
    "run_stage1_router",
    "select_abstention_threshold",
    "validate_config",
]
