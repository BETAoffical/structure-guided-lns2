"""Conditional Stage2 router for the frozen 16-map expanded-train cohort.

This identity consumes only exact ``Component != Hotspot`` Stage2 rows.  Label
support is audited before sklearn is imported or any estimator is fitted.  A
support failure therefore produces a fail-closed ``NO_GO_LABEL_SUPPORT``
artifact rather than a small-sample model.  Passing support permits only
map-grouped train OOF evaluation of two low-capacity candidates; there is no
development, final, runtime, promotion, or default claim.
"""

from __future__ import annotations

import collections
import math
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_hierarchical_ch_compact_flow_labels_readiness_v1 import (
    DEVELOPMENT_MAPS,
    TRAIN_MAPS,
)
from experiments.stride_hierarchical_ch_compact_flow_stage1_router_v1 import (
    THRESHOLD_GRID,
    _fit_estimator as _stage1_fit_estimator,
    _predict_probability as _stage1_predict_probability,
)


CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage2_router_config.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage2_router_report.v1"
)
SUPPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage2_support_audit.v1"
)
PREDICTION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage2_prediction.v1"
)
MODEL_MANIFEST_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage2_challenger_manifest.v1"
)
INPUT_REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_train_report.v1"
)
LABEL_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage2_state_label.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_expanded_stage2_router_v1"
DEFAULT_CONFIG = (
    "configs/stride_hierarchical_ch_compact_flow_expanded_stage2_router_v1.json"
)
DEFAULT_OUTPUT = (
    "build/stride-hierarchical-ch-compact-flow-expanded-stage2-router-v1"
)

DECISIVE_LABELS = ("component_win", "hotspot_win")
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
    "delta_component_minus_hotspot:realized.incident_conflict_coverage",
    "delta_component_minus_hotspot:realized.internal_conflict_coverage",
    "delta_component_minus_hotspot:realized.boundary_conflict_edges",
    "delta_component_minus_hotspot:realized.conflict_degree_mean",
    "delta_component_minus_hotspot:realized.delay_mean",
    "delta_component_minus_hotspot:realized.path_overlap_mean",
    "delta_component_minus_hotspot:realized.path_wait_ratio_mean",
    "relation:intersection_ratio",
    "relation:jaccard",
    "relation:symmetric_difference_ratio",
)
EXPECTED_MAPS = frozenset((*TRAIN_MAPS, *DEVELOPMENT_MAPS))
FOLDS = {
    "fold0": ("den404d", "den408d", "den207d", "den203d"),
    "fold1": ("lak101d", "den202d", "den998d", "den308d"),
    "fold2": ("lak108d", "den201d", "den009d", "den020d"),
    "fold3": ("lak110d", "ost102d", "hrt002d", "den101d"),
}
MODEL_IDS = ("regularized_logistic", "low_capacity_hist_gradient_boosting")
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


def _normalized_folds(value: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(fold): tuple(map(str, maps))
        for fold, maps in value.items()
    }


def _validate_fold_partition(folds: Mapping[str, Sequence[str]]) -> None:
    if dict(folds) != FOLDS:
        raise ValueError("expanded Stage2 fold names/order changed")
    flattened = [map_id for maps in folds.values() for map_id in maps]
    if (
        any(len(maps) != 4 for maps in folds.values())
        or len(flattened) != 16
        or len(set(flattened)) != 16
        or set(flattened) != EXPECTED_MAPS
    ):
        raise ValueError("expanded Stage2 folds must partition the frozen 16 maps")


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_expanded_train_stage2_support_first"
    ):
        raise ValueError("expanded Stage2 router identity changed")
    if set(config.get("inputs") or {}) != {
        "expanded_train_report",
        "expanded_stage2_state_labels",
    }:
        raise ValueError("expanded Stage2 input registry changed")
    semantics = dict(config.get("input_semantics") or {})
    if semantics != {
        "required_integrity_schema": INPUT_REPORT_SCHEMA,
        "required_label_schema": LABEL_SCHEMA,
        "research_split": "expanded_train",
        "exact_component_hotspot_sets_differ": True,
        "fit_labels": list(DECISIVE_LABELS),
        "ambiguous_use": "abstention_threshold_calibration_only",
        "support_audit_precedes_any_model_fit": True,
    }:
        raise ValueError("expanded Stage2 input semantics changed")
    folds = _normalized_folds(dict(config.get("train_map_folds") or {}))
    _validate_fold_partition(folds)
    features = dict(config.get("feature_contract") or {})
    if (
        features.get("representation")
        != "component_minus_hotspot_with_pre_action_context_and_exact_set_relation"
        or tuple(features.get("feature_names") or ()) != tuple(FEATURE_NAMES)
        or features.get("identifier_values_used_as_features") is not False
        or features.get("repair_outcome_or_runtime_used_as_features") is not False
        or features.get("source")
        != "registered_expanded_label_vector_backed_by_sha_verified_h1_state"
    ):
        raise ValueError("expanded Stage2 feature contract changed")
    forbidden = set(map(str, features.get("forbidden_input_tokens") or ()))
    if not {
        "map_id", "task_id", "solver_seed", "trial", "outcome", "result",
        "runtime", "pp_seconds", "no_progress", "rollback", "time_limit",
        "normalized_conflict_reduction",
    } <= forbidden:
        raise ValueError("expanded Stage2 forbidden feature tokens weakened")
    for name in FEATURE_NAMES:
        if any(token in name.lower() for token in forbidden):
            raise ValueError(f"registered feature violates leakage contract: {name}")
    expected_support = {
        "minimum_examples_per_decisive_class": 2,
        "minimum_maps_per_decisive_class": 2,
        "minimum_depth_bands_per_decisive_class": 2,
        "minimum_examples_per_decisive_class_per_fold": 1,
        "minimum_maps_per_decisive_class_per_fold": 1,
        "each_fold_requires_both_decisive_classes": True,
    }
    if dict(config.get("support_gates") or {}) != expected_support:
        raise ValueError("expanded Stage2 support gates changed")
    models = list(config.get("models") or ())
    if [row.get("model_id") for row in models] != list(MODEL_IDS):
        raise ValueError("expanded Stage2 candidate models changed")
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
        raise ValueError("expanded Stage2 model parameters changed")
    if dict(config.get("sample_weighting") or {}) != {
        "unit": "state_occurrence_id",
        "equal_total_weight_per_state": True,
        "balance_decisive_classes_after_state_weighting": True,
    }:
        raise ValueError("expanded Stage2 sample weighting changed")
    calibration = dict(config.get("abstention_calibration") or {})
    if (
        tuple(map(float, calibration.get("threshold_grid") or ())) != THRESHOLD_GRID
        or float(calibration.get("maximum_ambiguous_selection_rate", -1)) != 0.25
        or float(calibration.get("minimum_decisive_coverage", -1)) != 0.25
        or tuple(calibration.get("selection_rule") or ()) != SELECTION_RULE
        or calibration.get("abstention_action") != "defer_to_v2_anchor"
    ):
        raise ValueError("expanded Stage2 abstention calibration changed")
    expected_gates = {
        "minimum_oof_balanced_accuracy": 0.55,
        "minimum_each_class_recall": 0.35,
        "minimum_oof_coverage": 0.25,
        "minimum_oof_selective_accuracy": 0.6,
        "maximum_ambiguous_selection_rate": 0.25,
        "minimum_each_fold_balanced_accuracy": 0.45,
        "minimum_each_fold_coverage": 0.1,
        "maximum_fold_balanced_accuracy_spread": 0.35,
        "maximum_supported_depth_balanced_accuracy_spread": 0.35,
        "minimum_rows_for_stability_group": 3,
    }
    if dict(config.get("hard_gates") or {}) != expected_gates:
        raise ValueError("expanded Stage2 hard gates changed")
    if tuple(config.get("model_selection_rule") or ()) != MODEL_SELECTION_RULE:
        raise ValueError("expanded Stage2 model selection changed")
    if dict(config.get("execution") or {}) != {
        "workers": 16,
        "maximum_workers": 20,
        "workers_enter_run_fingerprint": True,
        "solver_invocation_allowed": False,
    }:
        raise ValueError("expanded Stage2 execution contract changed")
    if dict(config.get("claim_boundary") or {}) != {
        "expanded_train_only": True,
        "development_rows_allowed": False,
        "final_rows_allowed": False,
        "support_failure_status": "NO_GO_LABEL_SUPPORT",
        "passing_status": "TRAIN_OOF_CHALLENGER_ONLY",
        "promotion_or_default_export_allowed": False,
        "runtime_or_ttf_claim_authorized": False,
        "development_claim_authorized": False,
        "final_claim_authorized": False,
        "fresh_map_disjoint_development_and_final_required": True,
    }:
        raise ValueError("expanded Stage2 claim boundary changed")
    expected_outputs = {
        "root": DEFAULT_OUTPUT,
        "support_audit": "stage2_support_audit.json",
        "feature_manifest": "feature_manifest.json",
        "oof_predictions": "stage2_oof_predictions.jsonl",
        "evaluation_report": "expanded_stage2_router_evaluation_report.json",
        "train_oof_challenger_model": "train_oof_challenger.pkl",
        "train_oof_challenger_manifest": "train_oof_challenger_manifest.json",
    }
    if dict(config.get("outputs") or {}) != expected_outputs:
        raise ValueError("expanded Stage2 outputs changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    return config_path, project_root, config


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric pre-action feature: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite pre-action feature: {field}")
    return result


def _exact_agents(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty exact agent list")
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value):
        raise ValueError(f"{field} contains an invalid agent id")
    agents = tuple(value)
    if agents != tuple(sorted(set(agents))):
        raise ValueError(f"{field} must be strictly increasing and unique")
    return agents


def _arm_features(state: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    features = dict(dict(dict(state.get("arms") or {}).get(role) or {}).get("features") or {})
    if not features:
        raise ValueError(f"missing pre-action features for role {role}")
    return features


def _recompute_feature_vector(
    state: Mapping[str, Any], component_agents: Sequence[int], hotspot_agents: Sequence[int]
) -> list[float]:
    component = _arm_features(state, "component16")
    hotspot = _arm_features(state, "hotspot16")
    left, right = set(component_agents), set(hotspot_agents)
    intersection, union = len(left & right), len(left | right)
    relations = {
        "intersection_ratio": intersection / min(len(left), len(right)),
        "jaccard": intersection / union,
        "symmetric_difference_ratio": len(left ^ right) / union,
    }
    values: list[float] = []
    for name in FEATURE_NAMES:
        kind, raw = name.split(":", 1)
        if kind == "context":
            values.append(_finite_number(component.get(raw), field=name))
        elif kind == "delta_component_minus_hotspot":
            values.append(
                _finite_number(component.get(raw), field=f"{name}/component")
                - _finite_number(hotspot.get(raw), field=f"{name}/hotspot")
            )
        elif kind == "relation":
            values.append(float(relations[raw]))
        else:  # pragma: no cover - fixed schema guard
            raise AssertionError(kind)
    return values


def build_stage2_example(
    row: Mapping[str, Any], project_root: Path, folds: Mapping[str, Sequence[str]]
) -> dict[str, Any]:
    state_id = str(row.get("state_occurrence_id", ""))
    label = str(row.get("label", ""))
    if (
        row.get("schema") != LABEL_SCHEMA
        or not state_id
        or label not in ALL_LABELS
        or row.get("research_split") != "expanded_train"
        or row.get("expanded_train") is not True
        or row.get("runtime_or_pp_seconds_used_in_label") is not False
        or row.get("sequential_design_only") is not True
        or row.get("training_authorized") is not True
        or row.get("exact_component_hotspot_sets_differ") is not True
    ):
        raise ValueError(f"expanded Stage2 label trust contract changed: {state_id}")
    component_id = str(row.get("component_action_id", ""))
    hotspot_id = str(row.get("hotspot_action_id", ""))
    component_agents = _exact_agents(
        row.get("exact_component_agents"), field="exact_component_agents"
    )
    hotspot_agents = _exact_agents(
        row.get("exact_hotspot_agents"), field="exact_hotspot_agents"
    )
    if not component_id or not hotspot_id or component_id == hotspot_id:
        raise ValueError(f"expanded Stage2 action IDs are not distinct: {state_id}")
    if component_agents == hotspot_agents:
        raise ValueError(f"expanded Stage2 exact C/H sets are equal: {state_id}")
    if tuple(row.get("model_feature_names") or ()) != tuple(FEATURE_NAMES):
        raise ValueError(f"expanded Stage2 feature names changed: {state_id}")
    registered_vector = [
        _finite_number(value, field=f"model_feature_vector[{index}]")
        for index, value in enumerate(row.get("model_feature_vector") or ())
    ]
    if len(registered_vector) != len(FEATURE_NAMES):
        raise ValueError(f"expanded Stage2 feature dimension changed: {state_id}")
    _require_sha(row.get("source_label_row_sha256"), field="source_label_row_sha256")
    source_state_file = str(row.get("source_h1_state_file", ""))
    if not source_state_file or PurePosixPath(source_state_file.replace("\\", "/")).is_absolute():
        raise ValueError(f"expanded Stage2 source H1 state file changed: {state_id}")
    _require_sha(
        row.get("source_h1_state_sha256"), field="source_h1_state_sha256"
    )
    if (
        row.get("repair_outcome_or_runtime_used_as_model_feature") is not False
        or row.get("development_evaluation_authorized") is not False
        or row.get("final_claim_authorized") is not False
        or row.get("runtime_or_ttf_claim_authorized") is not False
        or row.get("promotion_authorized") is not False
    ):
        raise ValueError(f"expanded Stage2 row claim boundary changed: {state_id}")
    map_id = str(row.get("map_id", ""))
    expected_fold = next((fold for fold, maps in folds.items() if map_id in maps), None)
    if expected_fold is None or str(row.get("train_fold", "")) != expected_fold:
        raise ValueError(f"expanded Stage2 frozen train fold changed: {state_id}")
    return {
        "state_occurrence_id": state_id,
        "map_id": map_id,
        "train_fold": expected_fold,
        "label": label,
        "target": 1 if label == "component_win" else 0 if label == "hotspot_win" else None,
        "depth_band": str(row.get("depth_band", "")),
        "map_family": str(row.get("map_family", "")),
        "hierarchical_stratum": str(row.get("hierarchical_stratum", "")),
        "features": registered_vector,
    }


def _class_support(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("label") == label]
    return {
        "row_count": len(selected),
        "state_count": len({str(row["state_occurrence_id"]) for row in selected}),
        "map_count": len({str(row["map_id"]) for row in selected}),
        "maps": sorted({str(row["map_id"]) for row in selected}),
        "depth_band_count": len({str(row["depth_band"]) for row in selected}),
        "depth_bands": sorted({str(row["depth_band"]) for row in selected}),
    }


def audit_stage2_support(
    rows: Sequence[Mapping[str, Any]],
    folds: Mapping[str, Sequence[str]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    _validate_fold_partition(folds)
    by_label = {label: _class_support(rows, label) for label in ALL_LABELS}
    by_fold: dict[str, Any] = {}
    for fold, maps in folds.items():
        held = [row for row in rows if row.get("train_fold") == fold]
        observed_maps = sorted({str(row["map_id"]) for row in held})
        class_support = {label: _class_support(held, label) for label in ALL_LABELS}
        by_fold[fold] = {
            "registered_maps": list(maps),
            "observed_maps": observed_maps,
            "row_count": len(held),
            "by_label": class_support,
            "both_decisive_classes_present": all(
                class_support[label]["row_count"]
                >= int(gates["minimum_examples_per_decisive_class_per_fold"])
                and class_support[label]["map_count"]
                >= int(gates["minimum_maps_per_decisive_class_per_fold"])
                for label in DECISIVE_LABELS
            ),
        }
    checks = {
        "no_unregistered_maps": {str(row["map_id"]) for row in rows} <= EXPECTED_MAPS,
        "unique_state_rows": len(rows)
        == len({str(row["state_occurrence_id"]) for row in rows}),
        "minimum_examples_per_decisive_class": all(
            by_label[label]["row_count"]
            >= int(gates["minimum_examples_per_decisive_class"])
            for label in DECISIVE_LABELS
        ),
        "minimum_maps_per_decisive_class": all(
            by_label[label]["map_count"] >= int(gates["minimum_maps_per_decisive_class"])
            for label in DECISIVE_LABELS
        ),
        "minimum_depth_bands_per_decisive_class": all(
            by_label[label]["depth_band_count"]
            >= int(gates["minimum_depth_bands_per_decisive_class"])
            for label in DECISIVE_LABELS
        ),
        "each_fold_requires_both_decisive_classes": all(
            row["both_decisive_classes_present"] for row in by_fold.values()
        ),
        "all_rows_match_registered_fold": all(
            set(row["observed_maps"]) <= set(row["registered_maps"])
            for row in by_fold.values()
        ),
    }
    return {
        "schema": SUPPORT_SCHEMA,
        "row_count": len(rows),
        "map_count": len({str(row["map_id"]) for row in rows}),
        "by_label": by_label,
        "by_fold": by_fold,
        "checks": checks,
        "passed": all(checks.values()),
        "model_fit_authorized": all(checks.values()),
        "failure_action": "NO_GO_LABEL_SUPPORT_WITHOUT_MODEL_FIT",
    }


def _decision(probability: float, threshold: float) -> str:
    if max(probability, 1.0 - probability) + 1e-12 < threshold:
        return "defer_to_v2_anchor"
    return "component" if probability >= 0.5 else "hotspot"


def classification_metrics(
    predictions: Sequence[Mapping[str, Any]], threshold: float
) -> dict[str, Any]:
    decisive = [row for row in predictions if row.get("label") in DECISIVE_LABELS]
    ambiguous = [row for row in predictions if row.get("label") == "ambiguous"]
    selected = [
        row for row in decisive
        if _decision(float(row["probability"]), threshold) != "defer_to_v2_anchor"
    ]
    correct = [
        row for row in selected
        if (_decision(float(row["probability"]), threshold) == "component")
        == (row.get("label") == "component_win")
    ]
    recalls: dict[str, float | None] = {}
    for label, decision in (("component_win", "component"), ("hotspot_win", "hotspot")):
        truth = [row for row in decisive if row.get("label") == label]
        recalls[label] = (
            sum(_decision(float(row["probability"]), threshold) == decision for row in truth)
            / len(truth)
            if truth else None
        )
    present = [value for value in recalls.values() if value is not None]
    ambiguous_selected = sum(
        _decision(float(row["probability"]), threshold) != "defer_to_v2_anchor"
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
    candidates = [classification_metrics(predictions, threshold) for threshold in THRESHOLD_GRID]
    for row in candidates:
        row["feasible"] = (
            float(row["ambiguous_selection_rate"])
            <= float(calibration["maximum_ambiguous_selection_rate"]) + 1e-12
            and float(row["coverage"])
            >= float(calibration["minimum_decisive_coverage"]) - 1e-12
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
        status = "no_feasible_threshold_defer_all"
    return {
        "status": status,
        "selected_threshold": float(selected["threshold"]),
        "selected_metrics": selected,
        "grid": candidates,
        "ambiguous_fitted_as_class": False,
        "calibration_population": "expanded_train_oof_only",
    }


def _group_metrics(
    predictions: Sequence[Mapping[str, Any]], threshold: float, field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in predictions:
        groups[str(row[field])].append(row)
    return {
        name: classification_metrics(group, threshold)
        for name, group in sorted(groups.items())
    }


def _spread(
    groups: Mapping[str, Mapping[str, Any]], minimum_rows: int
) -> tuple[float | None, list[str]]:
    supported = {
        name: float(metrics["balanced_accuracy"])
        for name, metrics in groups.items()
        if int(metrics["decisive_row_count"]) >= minimum_rows
        and metrics.get("balanced_accuracy") is not None
    }
    if len(supported) < 2:
        return None, sorted(supported)
    return max(supported.values()) - min(supported.values()), sorted(supported)


def _gate_checks(
    overall: Mapping[str, Any],
    folds: Mapping[str, Mapping[str, Any]],
    depths: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, Any]]:
    fold_values = [
        float(row["balanced_accuracy"])
        for row in folds.values()
        if row.get("balanced_accuracy") is not None
    ]
    fold_coverages = [float(row["coverage"]) for row in folds.values()]
    fold_spread = max(fold_values) - min(fold_values) if len(fold_values) == 4 else None
    depth_spread, supported_depths = _spread(
        depths, int(gates["minimum_rows_for_stability_group"])
    )
    recalls = dict(overall["class_recall"])
    checks = {
        "minimum_oof_balanced_accuracy": overall.get("balanced_accuracy") is not None
        and float(overall["balanced_accuracy"]) >= float(gates["minimum_oof_balanced_accuracy"]),
        "minimum_each_class_recall": all(
            recalls.get(label) is not None
            and float(recalls[label]) >= float(gates["minimum_each_class_recall"])
            for label in DECISIVE_LABELS
        ),
        "minimum_oof_coverage": float(overall["coverage"])
        >= float(gates["minimum_oof_coverage"]),
        "minimum_oof_selective_accuracy": overall.get("selective_accuracy") is not None
        and float(overall["selective_accuracy"])
        >= float(gates["minimum_oof_selective_accuracy"]),
        "maximum_ambiguous_selection_rate": float(overall["ambiguous_selection_rate"])
        <= float(gates["maximum_ambiguous_selection_rate"]),
        "minimum_each_fold_balanced_accuracy": len(fold_values) == 4
        and min(fold_values) >= float(gates["minimum_each_fold_balanced_accuracy"]),
        "minimum_each_fold_coverage": len(fold_coverages) == 4
        and min(fold_coverages) >= float(gates["minimum_each_fold_coverage"]),
        "maximum_fold_balanced_accuracy_spread": fold_spread is not None
        and fold_spread <= float(gates["maximum_fold_balanced_accuracy_spread"]),
        "maximum_supported_depth_balanced_accuracy_spread": depth_spread is not None
        and depth_spread
        <= float(gates["maximum_supported_depth_balanced_accuracy_spread"]),
    }
    return checks, {
        "fold_balanced_accuracy_spread": fold_spread,
        "depth_balanced_accuracy_spread": depth_spread,
        "supported_depth_groups": supported_depths,
    }


def _fit_estimator(model_spec: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> Any:
    return _stage1_fit_estimator(model_spec, rows)


def _predict_probability(estimator: Any, rows: Sequence[Mapping[str, Any]]) -> list[float]:
    return _stage1_predict_probability(estimator, rows)


def cross_validate_candidate(
    examples: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    fit: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Any] = _fit_estimator,
    predict: Callable[[Any, Sequence[Mapping[str, Any]]], list[float]] = _predict_probability,
) -> dict[str, Any]:
    folds = _normalized_folds(dict(config["train_map_folds"]))
    predictions: list[dict[str, Any]] = []
    leakage_rows: list[dict[str, Any]] = []
    for fold, held_maps in folds.items():
        fit_rows = [
            row for row in examples
            if row.get("label") in DECISIVE_LABELS and row.get("train_fold") != fold
        ]
        held_rows = [row for row in examples if row.get("train_fold") == fold]
        fit_states = {str(row["state_occurrence_id"]) for row in fit_rows}
        held_states = {str(row["state_occurrence_id"]) for row in held_rows}
        fit_maps = {str(row["map_id"]) for row in fit_rows}
        held_map_set = {str(row["map_id"]) for row in held_rows}
        if fit_states & held_states or fit_maps & held_map_set or held_map_set != set(held_maps):
            raise ValueError(f"expanded Stage2 group/map fold leakage: {fold}")
        estimator = fit(model_spec, fit_rows)
        probabilities = predict(estimator, held_rows)
        if len(probabilities) != len(held_rows):
            raise ValueError("expanded Stage2 prediction count mismatch")
        predictions.extend(
            {**row, "probability": float(probability), "model_id": model_spec["model_id"]}
            for row, probability in zip(held_rows, probabilities)
        )
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
    if len(predictions) != len(examples) or {
        str(row["state_occurrence_id"]) for row in predictions
    } != {str(row["state_occurrence_id"]) for row in examples}:
        raise ValueError("expanded Stage2 OOF prediction coverage changed")
    calibration = select_abstention_threshold(predictions, config["abstention_calibration"])
    threshold = float(calibration["selected_threshold"])
    overall = classification_metrics(predictions, threshold)
    fold_metrics = _group_metrics(predictions, threshold, "train_fold")
    depth_metrics = _group_metrics(predictions, threshold, "depth_band")
    checks, stability = _gate_checks(
        overall, fold_metrics, depth_metrics, config["hard_gates"]
    )
    return {
        "model_id": str(model_spec["model_id"]),
        "threshold_calibration": calibration,
        "oof_metrics": overall,
        "by_fold": fold_metrics,
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
        min(fold_values) if len(fold_values) == 4 else -1.0,
        float(metrics.get("balanced_accuracy") or -1.0),
        float(metrics.get("selective_accuracy") or -1.0),
        float(metrics.get("coverage") or 0.0),
        -index,
    )


def _prediction_output(row: Mapping[str, Any], threshold: float) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "model_id": str(row["model_id"]),
        "state_occurrence_id": str(row["state_occurrence_id"]),
        "research_split": "expanded_train",
        "map_id": str(row["map_id"]),
        "train_fold": str(row["train_fold"]),
        "label": str(row["label"]),
        "depth_band": str(row["depth_band"]),
        "component_win_probability": float(row["probability"]),
        "threshold": threshold,
        "decision": _decision(float(row["probability"]), threshold),
        "ambiguous_used_for_fit": False,
        "development_claim_authorized": False,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
    }


def evaluate_candidates_if_supported(
    examples: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    workers: int,
    fit: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Any] = _fit_estimator,
    predict: Callable[[Any, Sequence[Mapping[str, Any]]], list[float]] = _predict_probability,
) -> tuple[list[dict[str, Any]], bool]:
    """Return no candidates without touching ``fit`` when support is not passed."""
    if support.get("passed") is not True:
        return [], False
    with ThreadPoolExecutor(max_workers=min(workers, len(config["models"]))) as pool:
        futures = [
            pool.submit(
                cross_validate_candidate,
                examples,
                model,
                config,
                fit=fit,
                predict=predict,
            )
            for model in config["models"]
        ]
        return [future.result() for future in futures], True


def _atomic_pickle(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def _validate_integrity_report(
    report: Mapping[str, Any], labels_path: Path, folds: Mapping[str, Sequence[str]]
) -> None:
    if (
        report.get("schema") != INPUT_REPORT_SCHEMA
        or report.get("complete") is not True
        or report.get("status") != "READY_EXPANDED_TRAIN"
        or report.get("expanded_training_use_authorized") is not True
        or report.get("training_authorized") is not True
        or report.get("development_evaluation_authorized") is not False
        or report.get("final_claim_authorized") is not False
        or report.get("runtime_or_ttf_claim_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("model_fit_executed") is not False
        or report.get("all_registered_maps_included") is not True
        or int(report.get("registered_map_count", -1)) != 16
        or report.get("folds_frozen_before_label_read") is not True
    ):
        raise ValueError("expanded train integrity report trust contract changed")
    observed_folds = _normalized_folds(dict(report.get("fixed_map_folds") or {}))
    if observed_folds != dict(folds):
        raise ValueError("expanded train integrity report folds changed")
    artifacts = dict(report.get("artifacts") or {})
    label_artifact = dict(artifacts.get("expanded_stage2_state_labels") or {})
    if (
        label_artifact.get("file") != labels_path.name
        or label_artifact.get("schema") != LABEL_SCHEMA
        or int(label_artifact.get("row_count", -1)) != 96
    ):
        raise ValueError("expanded train report label artifact identity changed")
    registered_sha = label_artifact.get("sha256")
    if registered_sha is not None and _require_sha(
        registered_sha, field="expanded_stage2_state_labels.sha256"
    ) != sha256_file(labels_path):
        raise ValueError("expanded train report label SHA256 mismatch")
    integrity = dict(report.get("input_integrity") or {})
    if (
        integrity.get("every_h1_state_payload_sha256_verified") is not True
        or int(integrity.get("stage2_exact_component_hotspot_sets_differ_count", -1)) != 96
        or int(integrity.get("stage2_non_distinct_component_hotspot_set_count", -1)) != 0
        or integrity.get("map_or_row_selection_by_label") is not False
    ):
        raise ValueError("expanded train exact C/H integrity changed")
    feature_contract = dict(report.get("feature_contract") or {})
    if (
        int(feature_contract.get("feature_count", -1)) != len(FEATURE_NAMES)
        or tuple(feature_contract.get("stage2_feature_names") or ()) != FEATURE_NAMES
        or feature_contract.get("repair_outcome_or_runtime_used_as_model_feature") is not False
    ):
        raise ValueError("expanded train Stage2 feature contract changed")


def _validate_reported_support(
    report: Mapping[str, Any], support: Mapping[str, Any]
) -> None:
    reported = dict(dict(report.get("support") or {}).get("stage2_state") or {})
    if int(reported.get("row_count", -1)) != int(support["row_count"]):
        raise ValueError("expanded train reported Stage2 row count changed")
    expected_counts = {
        label: int(dict(support["by_label"])[label]["row_count"])
        for label in ALL_LABELS
    }
    if {str(key): int(value) for key, value in dict(reported.get("label_counts") or {}).items()} != expected_counts:
        raise ValueError("expanded train reported Stage2 label counts changed")
    reported_folds = dict(reported.get("by_fold") or {})
    for fold, row in dict(support["by_fold"]).items():
        observed = dict(reported_folds.get(fold) or {})
        expected = {
            label: int(dict(row["by_label"])[label]["row_count"])
            for label in ALL_LABELS
        }
        if (
            int(observed.get("row_count", -1)) != int(row["row_count"])
            or {str(key): int(value) for key, value in dict(observed.get("label_counts") or {}).items()}
            != expected
        ):
            raise ValueError(f"expanded train reported Stage2 fold support changed: {fold}")
    reported_gates = dict(reported.get("gates") or {})
    expected_gates = {
        "minimum_rows_per_direction": bool(
            dict(support["checks"])["minimum_examples_per_decisive_class"]
        ),
        "minimum_maps_per_direction": bool(
            dict(support["checks"])["minimum_maps_per_decisive_class"]
        ),
        "minimum_depth_bands_per_direction": bool(
            dict(support["checks"])["minimum_depth_bands_per_decisive_class"]
        ),
        "each_fold_has_both_directions": bool(
            dict(support["checks"])["each_fold_requires_both_decisive_classes"]
        ),
    }
    if reported_gates != expected_gates or bool(reported.get("support_ready")) != bool(support["passed"]):
        raise ValueError("expanded train reported Stage2 support decision changed")


def run_expanded_stage2_router(
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path | None = None,
    *,
    workers: int | None = None,
    dry_run: bool = False,
    fit: Callable[[Mapping[str, Any], Sequence[Mapping[str, Any]]], Any] = _fit_estimator,
    predict: Callable[[Any, Sequence[Mapping[str, Any]]], list[float]] = _predict_probability,
) -> dict[str, Any]:
    config_file, project_root, config = load_config(config_path)
    configured_workers = int(config["execution"]["workers"])
    workers = configured_workers if workers is None else int(workers)
    if workers < 1 or workers > int(config["execution"]["maximum_workers"]):
        raise ValueError("workers outside registered expanded Stage2 range")
    report_path = _registered_input(
        project_root,
        config["inputs"]["expanded_train_report"],
        field="expanded_train_report",
    )
    labels_path = _registered_input(
        project_root,
        config["inputs"]["expanded_stage2_state_labels"],
        field="expanded_stage2_state_labels",
    )
    folds = _normalized_folds(dict(config["train_map_folds"]))
    integrity_report = _read_json(report_path)
    _validate_integrity_report(integrity_report, labels_path, folds)
    label_rows = _read_jsonl(labels_path)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        examples = list(
            pool.map(lambda row: build_stage2_example(row, project_root, folds), label_rows)
        )
    support = audit_stage2_support(examples, folds, config["support_gates"])
    _validate_reported_support(integrity_report, support)
    run_fingerprint = _fingerprint({
        "schema": CONFIG_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_file),
        "integrity_report_sha256": sha256_file(report_path),
        "labels_sha256": sha256_file(labels_path),
        "workers": workers,
        "fixed_train_map_folds": {fold: list(maps) for fold, maps in folds.items()},
    })
    results, model_fit_executed = evaluate_candidates_if_supported(
        examples,
        support,
        config,
        workers=workers,
        fit=fit,
        predict=predict,
    )
    selected: dict[str, Any] | None = None
    if results:
        selected = max(
            results,
            key=lambda row: _model_rank(
                row, next(i for i, spec in enumerate(config["models"]) if spec["model_id"] == row["model_id"])
            ),
        )
    model_gate_passed = bool(selected and selected["passed"])
    status = (
        "NO_GO_LABEL_SUPPORT"
        if not support["passed"]
        else "TRAIN_OOF_CHALLENGER_ONLY"
        if model_gate_passed
        else "NO_GO_MODEL_GATES"
    )
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": status,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "workers": workers,
        "inputs": {
            "expanded_train_integrity_report_sha256": sha256_file(report_path),
            "expanded_stage2_state_labels_sha256": sha256_file(labels_path),
            "expanded_stage2_state_label_count": len(label_rows),
        },
        "fixed_train_map_folds": {fold: list(maps) for fold, maps in folds.items()},
        "exact_component_hotspot_sets_differ_for_every_row": True,
        "support_audit": support,
        "support_gate_passed_before_model_fit": bool(support["passed"]),
        "model_fit_executed": model_fit_executed,
        "candidate_models": [
            {key: value for key, value in result.items() if key != "predictions"}
            for result in results
        ],
        "selected_model_id": selected["model_id"] if selected else None,
        "selected_threshold": (
            float(selected["threshold_calibration"]["selected_threshold"])
            if selected else None
        ),
        "hard_gates_passed": model_gate_passed,
        "model_exported": False,
        "promotion_or_default_exported": False,
        "development_rows_loaded": False,
        "development_claim_authorized": False,
        "final_rows_loaded": False,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "claim_boundary": dict(config["claim_boundary"]),
        "next_decision": (
            "stop_without_fit_expand_cross_map_component_hotspot_support"
            if not support["passed"]
            else "retain_train_oof_challenger_pending_fresh_map_disjoint_development"
            if model_gate_passed
            else "stop_after_train_oof_model_gate_failure"
        ),
        "dry_run": dry_run,
    }
    if dry_run:
        return report
    output_root = Path(output or config["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    else:
        output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("expanded Stage2 router output already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])
    _write_json(output_root / outputs["support_audit"], support)
    _write_json(output_root / outputs["feature_manifest"], {
        "schema": "lns2.stride.hierarchical_ch_compact_flow_expanded_stage2_feature_manifest.v1",
        "run_fingerprint": run_fingerprint,
        "dimension": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "representation": config["feature_contract"]["representation"],
        "leakage_audit_passed": True,
    })
    prediction_rows: list[dict[str, Any]] = []
    for result in results:
        threshold = float(result["threshold_calibration"]["selected_threshold"])
        prediction_rows.extend(_prediction_output(row, threshold) for row in result["predictions"])
    _write_jsonl(output_root / outputs["oof_predictions"], prediction_rows)
    if model_gate_passed and selected is not None:
        decisive = [row for row in examples if row["label"] in DECISIVE_LABELS]
        final_estimator = fit(
            next(model for model in config["models"] if model["model_id"] == selected["model_id"]),
            decisive,
        )
        model_path = output_root / outputs["train_oof_challenger_model"]
        _atomic_pickle(model_path, {
            "experiment_id": EXPERIMENT_ID,
            "run_fingerprint": run_fingerprint,
            "model_id": selected["model_id"],
            "feature_names": tuple(FEATURE_NAMES),
            "threshold": float(selected["threshold_calibration"]["selected_threshold"]),
            "estimator": final_estimator,
            "status": "TRAIN_OOF_CHALLENGER_ONLY",
        })
        manifest = {
            "schema": MODEL_MANIFEST_SCHEMA,
            "status": "TRAIN_OOF_CHALLENGER_ONLY",
            "run_fingerprint": run_fingerprint,
            "model_id": selected["model_id"],
            "model_file": model_path.name,
            "model_sha256": sha256_file(model_path),
            "promotion_or_default": False,
            "fresh_map_disjoint_development_and_final_required": True,
        }
        _write_json(output_root / outputs["train_oof_challenger_manifest"], manifest)
        report["model_exported"] = True
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
    "INPUT_REPORT_SCHEMA",
    "LABEL_SCHEMA",
    "REPORT_SCHEMA",
    "SUPPORT_SCHEMA",
    "audit_stage2_support",
    "build_stage2_example",
    "classification_metrics",
    "cross_validate_candidate",
    "evaluate_candidates_if_supported",
    "run_expanded_stage2_router",
    "select_abstention_threshold",
    "validate_config",
]
