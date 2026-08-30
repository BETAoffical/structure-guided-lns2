"""Consensus-only Stage1 successor with selected-action nested evaluation.

The eligible population is the exact one-pair stratum ``C == H != V2``.
An Opportunity head learns ``structural_win vs ambiguous`` and a Direction
head learns ``structural_win vs v2_win``.  Both heads are regularized
logistic models; only the Opportunity feature view varies between the frozen
signed 18-D vector and the same vector with indices 8..14 made absolute.
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
import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_two_head_v1 as predecessor


CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_consensus_only_config.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_consensus_only_report.v1"
)
PREDICTION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_consensus_only_prediction.v1"
)
EXPERIMENT_ID = (
    "stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_only_v1"
)
DEFAULT_CONFIG = (
    "configs/stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_only_v1.json"
)
DEFAULT_OUTPUT = (
    "build/stride-hierarchical-ch-compact-flow-expanded-stage1-consensus-only-v1"
)

FEATURE_NAMES = predecessor.FEATURE_NAMES
FOLDS = predecessor.FOLDS
EXPANDED_MAPS = predecessor.EXPANDED_MAPS
ALL_LABELS = predecessor.ALL_LABELS
DECISIVE_LABELS = predecessor.DECISIVE_LABELS
GATE_THRESHOLDS = predecessor.GATE_THRESHOLDS
DIRECTION_THRESHOLDS = predecessor.DIRECTION_THRESHOLDS
DELTA_INDICES = tuple(range(8, 15))
EXPECTED_CONSENSUS_SUPPORT = {
    "row_count": 93,
    "state_count": 93,
    "active_map_count": 15,
    "inactive_registered_maps": ["den408d"],
    "label_counts": {"ambiguous": 69, "structural_win": 14, "v2_win": 10},
    "fold_row_counts": {"fold0": 17, "fold1": 25, "fold2": 24, "fold3": 27},
}
LOGISTIC_MODEL = {
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
}
OPPORTUNITY_VIEWS = (
    {
        "view_id": "signed18",
        "transform": "identity",
        "absolute_indices": [],
    },
    {
        "view_id": "abs_delta18",
        "transform": "absolute_values_only_at_indices_8_through_14",
        "absolute_indices": list(DELTA_INDICES),
    },
)
CANDIDATE_ARMS = (
    {
        "arm_id": "opportunity_signed18__direction_signed18",
        "opportunity_view_id": "signed18",
        "opportunity_model_id": "regularized_logistic",
        "direction_view_id": "signed18",
        "direction_model_id": "regularized_logistic",
    },
    {
        "arm_id": "opportunity_abs_delta18__direction_signed18",
        "opportunity_view_id": "abs_delta18",
        "opportunity_model_id": "regularized_logistic",
        "direction_view_id": "signed18",
        "direction_model_id": "regularized_logistic",
    },
)
INNER_GATES = {
    "maximum_v2_wrong_override_count": 0,
    "maximum_ambiguous_override_rate": 0.10,
    "minimum_selected_action_precision": 0.60,
    "minimum_full_pooled_decisive_balanced_accuracy": 0.55,
    "minimum_structural_win_override_recall": 0.25,
    "minimum_total_override_count": 5,
    "minimum_correct_structural_override_map_count": 3,
    "minimum_correct_structural_override_fold_count": 2,
}
FINAL_GATES = {
    "require_exact_v2_fallback_semantics_rate": 1.0,
    "maximum_v2_wrong_override_count": 0,
    "maximum_ambiguous_override_rate": 0.10,
    "minimum_selected_action_precision": 0.60,
    "minimum_full_pooled_decisive_balanced_accuracy": 0.55,
    "minimum_structural_win_override_recall": 0.25,
    "minimum_total_override_count": 8,
    "minimum_correct_structural_override_map_count": 4,
    "minimum_correct_structural_override_fold_count": 2,
}
THRESHOLD_SELECTION_RULE = (
    "all_inner_gates_first",
    "maximum_full_pooled_decisive_balanced_accuracy",
    "maximum_selected_action_precision",
    "maximum_structural_win_override_recall",
    "minimum_ambiguous_override_rate",
    "higher_opportunity_threshold",
    "higher_direction_threshold",
)
ARM_SELECTION_RULE = (
    "all_inner_gates_first",
    "maximum_full_pooled_decisive_balanced_accuracy",
    "maximum_selected_action_precision",
    "maximum_structural_win_override_recall",
    "minimum_ambiguous_override_rate",
    "higher_opportunity_threshold",
    "higher_direction_threshold",
    "config_order",
)
CLAIM_BOUNDARY = {
    "stage1_only": True,
    "consensus_only": True,
    "expanded_train_nested_oof_only": True,
    "development_partition_present": False,
    "stage2_rows_loaded": False,
    "full_fit_model_allowed": False,
    "model_export_allowed": False,
    "passing_status": "TRAIN_OOF_CONSENSUS_CHALLENGER_ONLY",
    "failure_status": "NO_GO",
    "v2_replacement_allowed": False,
    "promotion_or_default_export_allowed": False,
    "runtime_or_ttf_claim_authorized": False,
    "final_claim_authorized": False,
    "external_map_disjoint_confirmation_required": True,
}


class HeadClassSupportError(ValueError):
    """A nested fit split lacks one of the frozen head classes."""


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_consensus_only_selected_action_nested_oof"
        or config.get("registration_status") != "REGISTERED_NOT_RUN"
    ):
        raise ValueError("consensus-only identity changed")
    if set(config.get("inputs") or {}) != {
        "expanded_train_report",
        "expanded_stage1_pair_labels",
        "h1_collection_report",
    }:
        raise ValueError("consensus-only input registry changed")
    if dict(config.get("consensus_filter") or {}) != {
        "hierarchical_stratum": "consensus_structural",
        "structural_role_aliases_exact": ["component16", "hotspot16"],
        "canonical_partition": "consensus_structural",
        "unique_action_count": 2,
        "stage2_eligible": False,
        "component_equals_hotspot": True,
        "structural_differs_from_v2": True,
        "exact_component_hotspot_agents_equal": True,
        "exact_structural_v2_agents_differ": True,
        "exactly_one_pair_per_state": True,
    }:
        raise ValueError("consensus-only exact filter changed")
    folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(config.get("expanded_map_folds") or {}).items()
    }
    if folds != FOLDS:
        raise ValueError("consensus-only map folds changed")
    features = dict(config.get("feature_contract") or {})
    if (
        features.get("source") != "expanded_row.model_feature_vector"
        or tuple(features.get("signed_feature_names") or ()) != FEATURE_NAMES
        or int(features.get("dimension", -1)) != 18
        or tuple(features.get("abs_delta_indices") or ()) != DELTA_INDICES
        or features.get("direction_view") != "signed18"
        or features.get("identifier_values_used_as_features") is not False
        or features.get("repair_outcome_or_runtime_used_as_features") is not False
    ):
        raise ValueError("consensus-only feature contract changed")
    forbidden = set(map(str, features.get("forbidden_input_tokens") or ()))
    if not {
        "map_id",
        "task_id",
        "solver_seed",
        "trial",
        "outcome",
        "runtime",
        "pp_seconds",
        "rollback",
        "time_limit",
        "normalized_conflict_reduction",
    } <= forbidden:
        raise ValueError("consensus-only forbidden feature tokens weakened")
    if any(token in name.lower() for name in FEATURE_NAMES for token in forbidden):
        raise ValueError("consensus-only registered feature leaks")
    if dict(config.get("model") or {}) != LOGISTIC_MODEL:
        raise ValueError("consensus-only Logistic model changed")
    if tuple(config.get("opportunity_feature_views") or ()) != OPPORTUNITY_VIEWS:
        raise ValueError("consensus-only Opportunity views changed")
    if tuple(config.get("candidate_arms") or ()) != CANDIDATE_ARMS:
        raise ValueError("consensus-only candidate arms changed")
    if dict(config.get("head_contract") or {}) != {
        "opportunity": {
            "target": "structural_win_vs_ambiguous",
            "fit_labels": ["structural_win", "ambiguous"],
            "v2_rows_in_fit": False,
            "state_equal_before_class_balance": True,
            "class_balance": True,
        },
        "direction": {
            "target": "structural_win_vs_v2_win",
            "fit_labels": ["structural_win", "v2_win"],
            "ambiguous_rows_in_fit": False,
            "state_equal_before_class_balance": True,
            "class_balance": True,
        },
    }:
        raise ValueError("consensus-only head contract changed")
    thresholds = dict(config.get("threshold_calibration") or {})
    if (
        tuple(map(float, thresholds.get("opportunity_threshold_grid") or ()))
        != GATE_THRESHOLDS
        or tuple(map(float, thresholds.get("direction_threshold_grid") or ()))
        != DIRECTION_THRESHOLDS
        or tuple(thresholds.get("selection_rule") or ())
        != THRESHOLD_SELECTION_RULE
    ):
        raise ValueError("consensus-only threshold contract changed")
    if dict(config.get("inner_gates") or {}) != INNER_GATES:
        raise ValueError("consensus-only inner gates changed")
    if dict(config.get("final_gates") or {}) != FINAL_GATES:
        raise ValueError("consensus-only final gates changed")
    if tuple(config.get("arm_selection_rule") or ()) != ARM_SELECTION_RULE:
        raise ValueError("consensus-only arm selection changed")
    if dict(config.get("expected_support") or {}) != EXPECTED_CONSENSUS_SUPPORT:
        raise ValueError("consensus-only expected support changed")
    if dict(config.get("execution") or {}) != {
        "workers": 16,
        "maximum_workers": 20,
        "parallel_head_jobs": 3,
        "workers_enter_run_fingerprint": True,
        "solver_invocation_allowed": False,
    }:
        raise ValueError("consensus-only execution contract changed")
    if dict(config.get("outputs") or {}) != {
        "root": DEFAULT_OUTPUT,
        "feature_manifest": "feature_manifest.json",
        "oof_predictions": "consensus_only_oof_predictions.jsonl",
        "evaluation_report": "consensus_only_evaluation_report.json",
    }:
        raise ValueError("consensus-only outputs changed")
    if dict(config.get("claim_boundary") or {}) != CLAIM_BOUNDARY:
        raise ValueError("consensus-only claim boundary changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    return config_path, project_root, config


def _canonical_agent_tuple(values: Any, *, field: str) -> tuple[int, ...]:
    raw = list(values or ())
    if not raw or any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError(f"{field} must be a non-empty integer agent set")
    if len(raw) != len(set(raw)):
        raise ValueError(f"{field} contains duplicate agents")
    return tuple(sorted(raw))


def _consensus_state_contract(state: Mapping[str, Any]) -> dict[str, Any] | None:
    if (
        state.get("canonical_partition") != "consensus_structural"
        or state.get("hierarchical_stratum") != "consensus_structural"
        or int(state.get("unique_action_count", -1)) != 2
        or state.get("stage2_eligible") is not False
    ):
        return None
    role_map = {
        str(key): str(value)
        for key, value in dict(state.get("role_to_action_id") or {}).items()
    }
    component_id = role_map.get("component16", "")
    hotspot_id = role_map.get("hotspot16", "")
    v2_id = role_map.get("v2_anchor", "")
    if not component_id or component_id != hotspot_id or component_id == v2_id:
        raise ValueError("consensus H1 role IDs do not satisfy C=H!=V")
    arms = dict(state.get("arms") or {})
    component_agents = _canonical_agent_tuple(
        dict(arms.get("component16") or {}).get("agents"),
        field="component16.agents",
    )
    hotspot_agents = _canonical_agent_tuple(
        dict(arms.get("hotspot16") or {}).get("agents"),
        field="hotspot16.agents",
    )
    v2_agents = _canonical_agent_tuple(
        dict(arms.get("v2_anchor") or {}).get("agents"),
        field="v2_anchor.agents",
    )
    if (
        not component_agents
        or component_agents != hotspot_agents
        or component_agents == v2_agents
    ):
        raise ValueError("consensus H1 exact agent sets do not satisfy C=H!=V")
    action_rows = list(state.get("unique_actions") or ())
    action_ids = [str(row.get("action_id", "")) for row in action_rows]
    if (
        len(action_rows) != 2
        or any(not action_id for action_id in action_ids)
        or len(set(action_ids)) != 2
    ):
        raise ValueError("consensus H1 unique_actions must contain two unique non-empty IDs")
    actions = {
        action_id: dict(row) for action_id, row in zip(action_ids, action_rows)
    }
    if set(actions) != {component_id, v2_id}:
        raise ValueError("consensus H1 unique action identity changed")
    if _canonical_agent_tuple(
        actions[component_id].get("agents"), field="structural_action.agents"
    ) != component_agents:
        raise ValueError("consensus H1 structural action agents changed")
    if _canonical_agent_tuple(
        actions[v2_id].get("agents"), field="v2_action.agents"
    ) != v2_agents:
        raise ValueError("consensus H1 V2 action agents changed")
    if tuple(state.get("stage1_structural_action_ids") or ()) != (component_id,):
        raise ValueError("consensus H1 structural action list changed")
    return {
        "structural_action_id": component_id,
        "v2_action_id": v2_id,
        "exact_structural_agents": component_agents,
        "exact_v2_agents": v2_agents,
    }


def _load_manifest_index(collection_report_path: Path) -> dict[str, dict[str, Any]]:
    report = _read_json(collection_report_path)
    artifact = dict(
        dict(report.get("artifacts") or {}).get("label_source_state_manifest")
        or {}
    )
    manifest_path = collection_report_path.parent / str(artifact.get("file", ""))
    if (
        not manifest_path.is_file()
        or sha256_file(manifest_path) != str(artifact.get("sha256", ""))
    ):
        raise ValueError("consensus H1 state manifest pin mismatch")
    rows = _read_jsonl(manifest_path)
    result = {str(row.get("state_occurrence_id", "")): dict(row) for row in rows}
    if len(result) != len(rows):
        raise ValueError("consensus H1 manifest state identity changed")
    return result


def build_consensus_example(
    label_row: Mapping[str, Any],
    state: Mapping[str, Any],
    manifest_row: Mapping[str, Any],
) -> dict[str, Any]:
    contract = _consensus_state_contract(state)
    if contract is None:
        raise ValueError("non-consensus H1 state entered consensus builder")
    example = predecessor.build_pair_example(label_row)
    state_id = str(example["state_occurrence_id"])
    row_structural_agents = _canonical_agent_tuple(
        example["exact_structural_agents"], field="label.exact_structural_agents"
    )
    row_v2_agents = _canonical_agent_tuple(
        example["exact_v2_agents"], field="label.exact_v2_agents"
    )
    if row_structural_agents == row_v2_agents:
        raise ValueError(f"consensus label exact structural/V2 sets are equal: {state_id}")
    if (
        str(state.get("state_occurrence_id", "")) != state_id
        or str(manifest_row.get("state_occurrence_id", "")) != state_id
        or str(label_row.get("source_h1_state_file", ""))
        != str(manifest_row.get("state_file", ""))
        or str(label_row.get("source_h1_state_sha256", ""))
        != str(manifest_row.get("state_sha256", ""))
        or str(label_row.get("source_h1_manifest_sha256", "")) == ""
    ):
        raise ValueError(f"consensus H1 SHA/identity mismatch: {state_id}")
    if (
        label_row.get("hierarchical_stratum") != "consensus_structural"
        or tuple(sorted(map(str, label_row.get("structural_role_aliases") or ())))
        != ("component16", "hotspot16")
        or str(example["structural_action_id"])
        != contract["structural_action_id"]
        or str(example["v2_action_id"]) != contract["v2_action_id"]
        or row_structural_agents != contract["exact_structural_agents"]
        or row_v2_agents != contract["exact_v2_agents"]
    ):
        raise ValueError(f"consensus pair/H1 exact action mismatch: {state_id}")
    signed = [float(value) for value in example["features"]]
    absolute = [abs(value) if index in DELTA_INDICES else value for index, value in enumerate(signed)]
    label = str(example["label"])
    return {
        **example,
        "exact_structural_agents": list(row_structural_agents),
        "exact_v2_agents": list(row_v2_agents),
        "opportunity_target": (
            1 if label == "structural_win" else 0 if label == "ambiguous" else None
        ),
        "direction_target": (
            1 if label == "structural_win" else 0 if label == "v2_win" else None
        ),
        "opportunity_features_signed18": signed,
        "opportunity_features_abs_delta18": absolute,
        "direction_features_signed18": signed,
        "canonical_partition": "consensus_structural",
        "unique_action_count": 2,
        "stage2_eligible": False,
        "component_action_id": contract["structural_action_id"],
        "hotspot_action_id": contract["structural_action_id"],
        "source_h1_state_sha256": str(manifest_row["state_sha256"]),
    }


def _head_rows(
    examples: Sequence[Mapping[str, Any]], head: str, view_id: str
) -> list[dict[str, Any]]:
    if head == "opportunity":
        rows = [row for row in examples if row["label"] in {"structural_win", "ambiguous"}]
        target = "opportunity_target"
        feature_key = f"opportunity_features_{view_id}"
    elif head == "direction":
        if view_id != "signed18":
            raise ValueError("Direction feature view must remain signed18")
        rows = [row for row in examples if row["label"] in DECISIVE_LABELS]
        target = "direction_target"
        feature_key = "direction_features_signed18"
    else:
        raise ValueError(f"unknown consensus head: {head}")
    fit_rows = [
        {**row, "target": int(row[target]), "fit_features": list(row[feature_key])}
        for row in rows
    ]
    if {int(row["target"]) for row in fit_rows} != {0, 1}:
        raise HeadClassSupportError(f"consensus {head} fit lacks both classes")
    if head == "direction" and any(row["label"] == "ambiguous" for row in fit_rows):
        raise ValueError("ambiguous row entered Direction fit")
    if head == "opportunity" and any(row["label"] == "v2_win" for row in fit_rows):
        raise ValueError("V2 row entered Opportunity fit")
    return fit_rows


def _balanced_weights(
    rows: Sequence[Mapping[str, Any]], head: str
) -> tuple[list[float], dict[str, Any]]:
    state_counts = collections.Counter(str(row["state_occurrence_id"]) for row in rows)
    if set(state_counts.values()) != {1}:
        raise ValueError("consensus head population is not one pair per state")
    base_weights = [1.0 for _ in rows]
    base_class_totals = {
        str(target): sum(
            weight
            for row, weight in zip(rows, base_weights)
            if int(row["target"]) == target
        )
        for target in (0, 1)
    }
    if min(base_class_totals.values()) <= 0.0:
        raise HeadClassSupportError(f"consensus {head} class has zero weight")
    balanced = [
        weight / base_class_totals[str(int(row["target"]))]
        for row, weight in zip(rows, base_weights)
    ]
    scale = len(balanced) / sum(balanced)
    weights = [weight * scale for weight in balanced]
    effective = {
        str(target): sum(
            weight
            for row, weight in zip(rows, weights)
            if int(row["target"]) == target
        )
        for target in (0, 1)
    }
    if abs(effective["0"] - effective["1"]) > 1e-9:
        raise ValueError(f"consensus {head} class balance failed")
    return weights, {
        "head": head,
        "row_count": len(rows),
        "state_count": len(state_counts),
        "base_state_total_min": 1.0,
        "base_state_total_max": 1.0,
        "base_class_weight_totals": base_class_totals,
        "class_balance_applied": True,
        "effective_total_weight": sum(weights),
        "effective_class_weight_totals": effective,
        "state_equal_before_class_balance": True,
    }


def _fit_logistic(rows: Sequence[Mapping[str, Any]], head: str) -> Any:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    weights, _ = _balanced_weights(rows, head)
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", LogisticRegression(**LOGISTIC_MODEL["parameters"])),
        ]
    )
    estimator.fit(
        np.asarray([row["fit_features"] for row in rows], dtype=np.float64),
        np.asarray([int(row["target"]) for row in rows], dtype=np.int8),
        model__sample_weight=np.asarray(weights, dtype=np.float64),
    )
    return estimator


def _predict_probability(
    estimator: Any, rows: Sequence[Mapping[str, Any]], feature_key: str
) -> list[float]:
    if not rows:
        return []
    import numpy as np

    probabilities = estimator.predict_proba(
        np.asarray([row[feature_key] for row in rows], dtype=np.float64)
    )[:, 1]
    result = [float(value) for value in probabilities]
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result):
        raise ValueError("consensus Logistic returned invalid probabilities")
    return result


def cross_validate_head(
    examples: Sequence[Mapping[str, Any]],
    head: str,
    view_id: str,
    fold_names: Sequence[str],
    *,
    fit: Callable[[Sequence[Mapping[str, Any]], str], Any] = _fit_logistic,
    predict: Callable[[Any, Sequence[Mapping[str, Any]], str], list[float]] = _predict_probability,
) -> dict[str, Any]:
    active_folds = tuple(fold_names)
    if {str(row["train_fold"]) for row in examples} != set(active_folds):
        raise ValueError("consensus inner fold population changed")
    feature_key = (
        f"opportunity_features_{view_id}"
        if head == "opportunity"
        else "direction_features_signed18"
    )
    probabilities: dict[str, float] = {}
    audits = []
    for held_fold in active_folds:
        train = [row for row in examples if row["train_fold"] != held_fold]
        held = [row for row in examples if row["train_fold"] == held_fold]
        fit_rows = _head_rows(train, head, view_id)
        _, weight_audit = _balanced_weights(fit_rows, head)
        train_states = {row["state_occurrence_id"] for row in fit_rows}
        held_states = {row["state_occurrence_id"] for row in held}
        train_maps = {row["map_id"] for row in fit_rows}
        held_maps = {row["map_id"] for row in held}
        if train_states & held_states or train_maps & held_maps:
            raise ValueError(f"consensus inner leakage: {head}/{view_id}/{held_fold}")
        if not held_maps or not held_maps <= set(FOLDS[held_fold]):
            raise ValueError(f"consensus held-map support changed: {held_fold}")
        estimator = fit(fit_rows, head)
        values = predict(estimator, held, feature_key)
        for row, value in zip(held, values):
            pair_id = str(row["pair_id"])
            if pair_id in probabilities:
                raise ValueError("duplicate consensus inner OOF prediction")
            probabilities[pair_id] = float(value)
        audits.append(
            {
                "head": head,
                "view_id": view_id,
                "held_fold": held_fold,
                "train_map_count": len(train_maps),
                "held_map_count": len(held_maps),
                "train_state_count": len(train_states),
                "held_state_count": len(held_states),
                "state_overlap_count": 0,
                "map_overlap_count": 0,
                "fit_weight_audit": weight_audit,
            }
        )
    if set(probabilities) != {str(row["pair_id"]) for row in examples}:
        raise ValueError("consensus inner OOF coverage changed")
    return {"probabilities": probabilities, "fold_audits": audits}


def materialize_action(
    row: Mapping[str, Any],
    *,
    opportunity_probability: float | None,
    direction_probability: float | None,
    opportunity_threshold: float,
    direction_threshold: float,
    arm: Mapping[str, Any] | None,
    evaluation_active: bool,
    selection_status: str,
) -> dict[str, Any]:
    for name, value in (
        ("opportunity", opportunity_probability),
        ("direction", direction_probability),
    ):
        if value is not None and (
            not math.isfinite(value) or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"invalid consensus {name} probability")
    override = bool(
        evaluation_active
        and opportunity_probability is not None
        and direction_probability is not None
        and opportunity_probability + 1e-12 >= opportunity_threshold
        and direction_probability + 1e-12 >= direction_threshold
    )
    final_action_id = str(
        row["structural_action_id"] if override else row["v2_action_id"]
    )
    final_agents = list(
        row["exact_structural_agents"] if override else row["exact_v2_agents"]
    )
    if final_action_id != str(
        row["structural_action_id"] if override else row["v2_action_id"]
    ):
        raise ValueError("consensus selected action identity changed")
    return {
        **row,
        "arm_id": str(arm["arm_id"]) if arm else None,
        "opportunity_view_id": str(arm["opportunity_view_id"]) if arm else None,
        "opportunity_structural_probability": opportunity_probability,
        "direction_structural_probability": direction_probability,
        "opportunity_threshold": float(opportunity_threshold),
        "direction_threshold": float(direction_threshold),
        "joint_override_score": (
            opportunity_probability * direction_probability
            if opportunity_probability is not None and direction_probability is not None
            else None
        ),
        "structural_override": override,
        "final_action_id": final_action_id,
        "final_action_role": (
            "consensus_structural_override" if override else "exact_v2_fallback"
        ),
        "final_action_agents": final_agents,
        "evaluation_active": bool(evaluation_active),
        "selection_status": selection_status,
    }


def selected_action_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    structural = [row for row in rows if row["label"] == "structural_win"]
    v2 = [row for row in rows if row["label"] == "v2_win"]
    ambiguous = [row for row in rows if row["label"] == "ambiguous"]
    overrides = [row for row in rows if bool(row["structural_override"])]
    correct = [row for row in structural if bool(row["structural_override"])]
    v2_wrong = [row for row in v2 if bool(row["structural_override"])]
    ambiguous_wrong = [row for row in ambiguous if bool(row["structural_override"])]
    structural_recall = len(correct) / len(structural) if structural else None
    v2_protection = (
        (len(v2) - len(v2_wrong)) / len(v2) if v2 else None
    )
    balanced_accuracy = (
        (structural_recall + v2_protection) / 2.0
        if structural_recall is not None and v2_protection is not None
        else None
    )
    fallback_semantics = [
        (
            row["final_action_id"]
            == (
                row["structural_action_id"]
                if row["structural_override"]
                else row["v2_action_id"]
            )
            and row["final_action_agents"]
            == (
                row["exact_structural_agents"]
                if row["structural_override"]
                else row["exact_v2_agents"]
            )
            and row["final_action_role"]
            == (
                "consensus_structural_override"
                if row["structural_override"]
                else "exact_v2_fallback"
            )
        )
        for row in rows
    ]
    correct_maps = {str(row["map_id"]) for row in correct}
    correct_folds = {str(row["train_fold"]) for row in correct}
    override_folds = {str(row["train_fold"]) for row in overrides}
    return {
        "row_count": len(rows),
        "structural_win_count": len(structural),
        "v2_win_count": len(v2),
        "ambiguous_count": len(ambiguous),
        "evaluation_active_row_count": sum(bool(row["evaluation_active"]) for row in rows),
        "evaluation_inactive_exact_v2_row_count": sum(
            not bool(row["evaluation_active"]) for row in rows
        ),
        "total_override_count": len(overrides),
        "correct_structural_override_count": len(correct),
        "v2_wrong_override_count": len(v2_wrong),
        "ambiguous_wrong_override_count": len(ambiguous_wrong),
        "structural_win_override_recall": structural_recall,
        "v2_win_protection_recall": v2_protection,
        "full_pooled_decisive_balanced_accuracy": balanced_accuracy,
        "ambiguous_override_rate": (
            len(ambiguous_wrong) / len(ambiguous) if ambiguous else None
        ),
        "selected_action_precision": (
            len(correct) / len(overrides) if overrides else None
        ),
        "correct_structural_override_map_count": len(correct_maps),
        "correct_structural_override_maps": sorted(correct_maps),
        "correct_structural_override_fold_count": len(correct_folds),
        "correct_structural_override_folds": sorted(correct_folds),
        "all_override_fold_count": len(override_folds),
        "all_override_folds": sorted(override_folds),
        "exact_v2_fallback_semantics_rate": (
            sum(fallback_semantics) / len(fallback_semantics) if rows else None
        ),
        "fold_missing_class_is_not_imputed": True,
    }


def _group_metrics(
    rows: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        name: selected_action_metrics(group)
        for name, group in sorted(groups.items())
    }


def _inner_checks(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "maximum_v2_wrong_override_count": (
            int(metrics["v2_wrong_override_count"])
            <= int(gates["maximum_v2_wrong_override_count"])
        ),
        "maximum_ambiguous_override_rate": (
            metrics.get("ambiguous_override_rate") is not None
            and float(metrics["ambiguous_override_rate"])
            <= float(gates["maximum_ambiguous_override_rate"]) + 1e-12
        ),
        "minimum_selected_action_precision": (
            metrics.get("selected_action_precision") is not None
            and float(metrics["selected_action_precision"])
            >= float(gates["minimum_selected_action_precision"]) - 1e-12
        ),
        "minimum_full_pooled_decisive_balanced_accuracy": (
            metrics.get("full_pooled_decisive_balanced_accuracy") is not None
            and float(metrics["full_pooled_decisive_balanced_accuracy"])
            >= float(gates["minimum_full_pooled_decisive_balanced_accuracy"])
            - 1e-12
        ),
        "minimum_structural_win_override_recall": (
            metrics.get("structural_win_override_recall") is not None
            and float(metrics["structural_win_override_recall"])
            >= float(gates["minimum_structural_win_override_recall"]) - 1e-12
        ),
        "minimum_total_override_count": (
            int(metrics["total_override_count"])
            >= int(gates["minimum_total_override_count"])
        ),
        "minimum_correct_structural_override_map_count": (
            int(metrics["correct_structural_override_map_count"])
            >= int(gates["minimum_correct_structural_override_map_count"])
        ),
        "minimum_correct_structural_override_fold_count": (
            int(metrics["correct_structural_override_fold_count"])
            >= int(gates["minimum_correct_structural_override_fold_count"])
        ),
    }


def _final_checks(
    metrics: Mapping[str, Any], gates: Mapping[str, Any]
) -> dict[str, bool]:
    common = _inner_checks(
        metrics,
        {key: value for key, value in gates.items() if key != "require_exact_v2_fallback_semantics_rate"},
    )
    return {
        "require_exact_v2_fallback_semantics_rate": (
            metrics.get("exact_v2_fallback_semantics_rate") is not None
            and abs(
                float(metrics["exact_v2_fallback_semantics_rate"])
                - float(gates["require_exact_v2_fallback_semantics_rate"])
            )
            <= 1e-12
        ),
        **common,
    }


def _finalize_probability_rows(
    examples: Sequence[Mapping[str, Any]],
    opportunity_probabilities: Mapping[str, float],
    direction_probabilities: Mapping[str, float],
    arm: Mapping[str, Any],
    opportunity_threshold: float,
    direction_threshold: float,
    status: str,
) -> list[dict[str, Any]]:
    return [
        materialize_action(
            row,
            opportunity_probability=float(
                opportunity_probabilities[str(row["pair_id"])]
            ),
            direction_probability=float(
                direction_probabilities[str(row["pair_id"])]
            ),
            opportunity_threshold=opportunity_threshold,
            direction_threshold=direction_threshold,
            arm=arm,
            evaluation_active=True,
            selection_status=status,
        )
        for row in examples
    ]


def select_joint_thresholds(
    examples: Sequence[Mapping[str, Any]],
    opportunity_probabilities: Mapping[str, float],
    direction_probabilities: Mapping[str, float],
    arm: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    grid = []
    for opportunity_threshold in GATE_THRESHOLDS:
        for direction_threshold in DIRECTION_THRESHOLDS:
            rows = _finalize_probability_rows(
                examples,
                opportunity_probabilities,
                direction_probabilities,
                arm,
                opportunity_threshold,
                direction_threshold,
                "inner_joint_threshold_candidate",
            )
            metrics = selected_action_metrics(rows)
            checks = _inner_checks(metrics, gates)
            grid.append(
                {
                    "opportunity_threshold": opportunity_threshold,
                    "direction_threshold": direction_threshold,
                    "metrics": metrics,
                    "inner_gate_checks": checks,
                    "eligible": all(checks.values()),
                }
            )
    eligible = [row for row in grid if row["eligible"]]
    if not eligible:
        return {
            "status": "no_joint_threshold_passes_all_inner_gates",
            "selected_opportunity_threshold": 1.01,
            "selected_direction_threshold": 1.01,
            "selected_metrics": None,
            "joint_grid_size": len(grid),
            "grid": grid,
        }
    selected = max(
        eligible,
        key=lambda row: (
            float(row["metrics"]["full_pooled_decisive_balanced_accuracy"]),
            float(row["metrics"]["selected_action_precision"]),
            float(row["metrics"]["structural_win_override_recall"]),
            -float(row["metrics"]["ambiguous_override_rate"]),
            float(row["opportunity_threshold"]),
            float(row["direction_threshold"]),
        ),
    )
    return {
        "status": "joint_threshold_passes_all_inner_gates",
        "selected_opportunity_threshold": float(selected["opportunity_threshold"]),
        "selected_direction_threshold": float(selected["direction_threshold"]),
        "selected_metrics": selected["metrics"],
        "selected_inner_gate_checks": selected["inner_gate_checks"],
        "joint_grid_size": len(grid),
        "grid": grid,
    }


def select_inner_arm(
    examples: Sequence[Mapping[str, Any]],
    head_results: Mapping[tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reports = []
    eligible_indices = []
    direction = head_results[("direction", "signed18")]["probabilities"]
    for index, arm in enumerate(config["candidate_arms"]):
        opportunity = head_results[
            ("opportunity", str(arm["opportunity_view_id"]))
        ]["probabilities"]
        thresholds = select_joint_thresholds(
            examples, opportunity, direction, arm, config["inner_gates"]
        )
        report = {
            "arm_id": str(arm["arm_id"]),
            "opportunity_view_id": str(arm["opportunity_view_id"]),
            "joint_threshold_selection": thresholds,
            "eligible": thresholds["status"]
            == "joint_threshold_passes_all_inner_gates",
        }
        if report["eligible"]:
            eligible_indices.append(index)
        reports.append(report)
    if not eligible_indices:
        return {
            "status": "no_inner_arm_passes_all_gates_exact_v2_outer_fold",
            "selected_arm": None,
            "candidate_arms": reports,
        }

    def rank(index: int) -> tuple[Any, ...]:
        report = reports[index]
        selection = report["joint_threshold_selection"]
        metrics = selection["selected_metrics"]
        return (
            float(metrics["full_pooled_decisive_balanced_accuracy"]),
            float(metrics["selected_action_precision"]),
            float(metrics["structural_win_override_recall"]),
            -float(metrics["ambiguous_override_rate"]),
            float(selection["selected_opportunity_threshold"]),
            float(selection["selected_direction_threshold"]),
            -index,
        )

    selected_index = max(eligible_indices, key=rank)
    selected_report = reports[selected_index]
    selected_arm = dict(config["candidate_arms"][selected_index])
    selected_arm.update(
        {
            "opportunity_threshold": float(
                selected_report["joint_threshold_selection"][
                    "selected_opportunity_threshold"
                ]
            ),
            "direction_threshold": float(
                selected_report["joint_threshold_selection"][
                    "selected_direction_threshold"
                ]
            ),
        }
    )
    return {
        "status": "inner_arm_passes_all_gates",
        "selected_arm": selected_arm,
        "candidate_arms": reports,
    }


def _fit_outer_head(
    examples: Sequence[Mapping[str, Any]], head: str, view_id: str
) -> tuple[Any, dict[str, Any]]:
    rows = _head_rows(examples, head, view_id)
    _, audit = _balanced_weights(rows, head)
    return _fit_logistic(rows, head), audit


def _prediction_output(row: Mapping[str, Any], run_fingerprint: str) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "pair_id": str(row["pair_id"]),
        "state_occurrence_id": str(row["state_occurrence_id"]),
        "map_id": str(row["map_id"]),
        "train_fold": str(row["train_fold"]),
        "research_split": "expanded_train",
        "label": str(row["label"]),
        "canonical_partition": "consensus_structural",
        "component_action_id": str(row["component_action_id"]),
        "hotspot_action_id": str(row["hotspot_action_id"]),
        "structural_action_id": str(row["structural_action_id"]),
        "v2_action_id": str(row["v2_action_id"]),
        "exact_structural_agents": list(row["exact_structural_agents"]),
        "exact_v2_agents": list(row["exact_v2_agents"]),
        "exact_component_hotspot_agents_equal": True,
        "exact_structural_v2_sets_differ": True,
        "arm_id": row["arm_id"],
        "opportunity_view_id": row["opportunity_view_id"],
        "opportunity_structural_probability": row[
            "opportunity_structural_probability"
        ],
        "direction_structural_probability": row[
            "direction_structural_probability"
        ],
        "opportunity_threshold": float(row["opportunity_threshold"]),
        "direction_threshold": float(row["direction_threshold"]),
        "joint_override_score": row["joint_override_score"],
        "structural_override": bool(row["structural_override"]),
        "final_action_id": str(row["final_action_id"]),
        "final_action_role": str(row["final_action_role"]),
        "final_action_agents": list(row["final_action_agents"]),
        "evaluation_active": bool(row["evaluation_active"]),
        "selection_status": str(row["selection_status"]),
        "selected_action_equals_state_action": True,
        "inactive_outer_fold_is_exact_v2": not bool(row["evaluation_active"]),
        "inactive_outer_fold_is_not_a_breadth_win": True,
        "stage2_rows_loaded": False,
        "development_partition_present": False,
        "full_fit_model_exported": False,
        "promotion_or_default_authorized": False,
        "final_claim_authorized": False,
    }


def run_consensus_only(
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
    predecessor._validate_adapter_report(
        adapter_report,
        pair_labels_path=inputs["expanded_stage1_pair_labels"],
    )
    states, state_integrity = base._load_states(
        inputs["h1_collection_report"], worker_count
    )
    manifest = _load_manifest_index(inputs["h1_collection_report"])
    state_manifest_sha = str(state_integrity["state_manifest_sha256"])
    label_rows = _read_jsonl(inputs["expanded_stage1_pair_labels"])
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in label_rows:
        by_state[str(row.get("state_occurrence_id", ""))].append(row)
    consensus_state_ids = []
    for state_id, state in states.items():
        if _consensus_state_contract(state) is not None:
            consensus_state_ids.append(state_id)
    examples = []
    for state_id in sorted(consensus_state_ids):
        rows = by_state.get(state_id, [])
        if len(rows) != 1:
            raise ValueError(f"consensus state must have exactly one pair: {state_id}")
        row = rows[0]
        if str(row.get("source_h1_manifest_sha256", "")) != state_manifest_sha:
            raise ValueError(f"consensus label manifest SHA changed: {state_id}")
        examples.append(
            build_consensus_example(row, states[state_id], manifest[state_id])
        )
    label_counts = collections.Counter(row["label"] for row in examples)
    fold_counts = collections.Counter(row["train_fold"] for row in examples)
    active_maps = {str(row["map_id"]) for row in examples}
    observed_support = {
        "row_count": len(examples),
        "state_count": len({row["state_occurrence_id"] for row in examples}),
        "active_map_count": len(active_maps),
        "inactive_registered_maps": sorted(set(EXPANDED_MAPS) - active_maps),
        "label_counts": dict(sorted(label_counts.items())),
        "fold_row_counts": {fold: int(fold_counts[fold]) for fold in FOLDS},
    }
    if observed_support != EXPECTED_CONSENSUS_SUPPORT:
        raise ValueError("consensus-only support changed")
    if len({row["pair_id"] for row in examples}) != len(examples):
        raise ValueError("consensus pair IDs are duplicated")
    if any(row["research_split"] != "expanded_train" for row in examples):
        raise ValueError("development/final row entered consensus fit identity")
    inactive_pair_count = len(label_rows) - len(examples)
    inactive_state_count = len(states) - len(examples)

    output_root = Path(output or config["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    else:
        output_root = output_root.resolve()
    if not dry_run and output_root.exists() and any(output_root.iterdir()):
        raise ValueError("consensus-only output already exists")

    # Avoid concurrent first-import races while preserving the three frozen jobs.
    import numpy  # noqa: F401
    from sklearn.linear_model import LogisticRegression  # noqa: F401
    from sklearn.pipeline import Pipeline  # noqa: F401
    from sklearn.preprocessing import StandardScaler  # noqa: F401

    outer_records = []
    predictions: list[dict[str, Any]] = []
    all_inner_fit_s_counts = []
    all_inner_fit_v_counts = []
    for outer_fold, registered_held_maps in FOLDS.items():
        outer_train = [row for row in examples if row["train_fold"] != outer_fold]
        outer_held = [row for row in examples if row["train_fold"] == outer_fold]
        inner_folds = tuple(fold for fold in FOLDS if fold != outer_fold)
        train_states = {row["state_occurrence_id"] for row in outer_train}
        held_states = {row["state_occurrence_id"] for row in outer_held}
        train_maps = {row["map_id"] for row in outer_train}
        held_maps = {row["map_id"] for row in outer_held}
        if train_states & held_states or train_maps & held_maps:
            raise ValueError(f"consensus outer leakage: {outer_fold}")
        if not held_maps or not held_maps <= set(registered_held_maps):
            raise ValueError(f"consensus outer held maps changed: {outer_fold}")
        jobs = (
            ("opportunity", "signed18"),
            ("opportunity", "abs_delta18"),
            ("direction", "signed18"),
        )
        try:
            with ThreadPoolExecutor(
                max_workers=min(
                    worker_count, int(execution["parallel_head_jobs"])
                )
            ) as pool:
                futures = {
                    key: pool.submit(
                        cross_validate_head,
                        outer_train,
                        key[0],
                        key[1],
                        inner_folds,
                    )
                    for key in jobs
                }
                head_results = {
                    key: future.result() for key, future in futures.items()
                }
            for audit in head_results[("direction", "signed18")]["fold_audits"]:
                totals = audit["fit_weight_audit"]["base_class_weight_totals"]
                all_inner_fit_v_counts.append(int(totals["0"]))
                all_inner_fit_s_counts.append(int(totals["1"]))
            selection = select_inner_arm(outer_train, head_results, config)
        except HeadClassSupportError as error:
            head_results = {}
            selection = {
                "status": "missing_inner_fit_class_exact_v2_outer_fold",
                "selected_arm": None,
                "candidate_arms": [],
                "error": str(error),
            }
        selected = selection["selected_arm"]
        outer_fit_weight_audits: dict[str, Any] = {}
        if selected is None:
            held_predictions = [
                materialize_action(
                    row,
                    opportunity_probability=None,
                    direction_probability=None,
                    opportunity_threshold=1.01,
                    direction_threshold=1.01,
                    arm=None,
                    evaluation_active=False,
                    selection_status=str(selection["status"]),
                )
                for row in outer_held
            ]
            outer_status = "INACTIVE_EXACT_V2_FAIL_CLOSED"
        else:
            try:
                opportunity_estimator, opportunity_audit = _fit_outer_head(
                    outer_train,
                    "opportunity",
                    str(selected["opportunity_view_id"]),
                )
                direction_estimator, direction_audit = _fit_outer_head(
                    outer_train, "direction", "signed18"
                )
                outer_fit_weight_audits = {
                    "opportunity": opportunity_audit,
                    "direction": direction_audit,
                }
                opportunity_key = (
                    f"opportunity_features_{selected['opportunity_view_id']}"
                )
                opportunity_probabilities = _predict_probability(
                    opportunity_estimator, outer_held, opportunity_key
                )
                direction_probabilities = _predict_probability(
                    direction_estimator,
                    outer_held,
                    "direction_features_signed18",
                )
                held_predictions = [
                    materialize_action(
                        row,
                        opportunity_probability=float(opportunity_probability),
                        direction_probability=float(direction_probability),
                        opportunity_threshold=float(
                            selected["opportunity_threshold"]
                        ),
                        direction_threshold=float(selected["direction_threshold"]),
                        arm=selected,
                        evaluation_active=True,
                        selection_status=str(selection["status"]),
                    )
                    for row, opportunity_probability, direction_probability in zip(
                        outer_held,
                        opportunity_probabilities,
                        direction_probabilities,
                    )
                ]
                outer_status = "ACTIVE_NESTED_OUTER_PREDICTION"
            except HeadClassSupportError as error:
                held_predictions = [
                    materialize_action(
                        row,
                        opportunity_probability=None,
                        direction_probability=None,
                        opportunity_threshold=1.01,
                        direction_threshold=1.01,
                        arm=None,
                        evaluation_active=False,
                        selection_status="missing_outer_fit_class_exact_v2_outer_fold",
                    )
                    for row in outer_held
                ]
                selection = {**selection, "outer_fit_error": str(error)}
                outer_status = "INACTIVE_EXACT_V2_MISSING_OUTER_FIT_CLASS"
        predictions.extend(held_predictions)
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "registered_held_maps": list(registered_held_maps),
                "observed_held_maps": sorted(held_maps),
                "outer_train_state_count": len(train_states),
                "outer_held_state_count": len(held_states),
                "state_overlap_count": 0,
                "map_overlap_count": 0,
                "outer_status": outer_status,
                "inner_selection": selection,
                "inner_head_oof_audits": {
                    f"{head}:{view}": result["fold_audits"]
                    for (head, view), result in sorted(head_results.items())
                },
                "outer_fit_weight_audits": outer_fit_weight_audits,
                "outer_labels_never_used_for_selection": True,
            }
        )

    if len(predictions) != len(examples) or {
        row["pair_id"] for row in predictions
    } != {row["pair_id"] for row in examples}:
        raise ValueError("consensus nested outer OOF coverage changed")
    overall = selected_action_metrics(predictions)
    active_predictions = [row for row in predictions if row["evaluation_active"]]
    active_diagnostic = selected_action_metrics(active_predictions)
    by_fold = _group_metrics(predictions, "train_fold")
    by_map = _group_metrics(predictions, "map_id")
    checks = _final_checks(overall, config["final_gates"])
    passed = all(checks.values())
    status = "TRAIN_OOF_CONSENSUS_CHALLENGER_ONLY" if passed else "NO_GO"
    input_sha = {name: sha256_file(path) for name, path in inputs.items()}
    run_fingerprint = _fingerprint(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(config_path),
            "input_sha256": input_sha,
            "workers": worker_count,
            "fixed_map_folds": {fold: list(maps) for fold, maps in FOLDS.items()},
            "consensus_filter": config["consensus_filter"],
            "opportunity_feature_views": config["opportunity_feature_views"],
            "candidate_arms": config["candidate_arms"],
            "opportunity_thresholds": list(GATE_THRESHOLDS),
            "direction_thresholds": list(DIRECTION_THRESHOLDS),
            "inner_gates": config["inner_gates"],
            "final_gates": config["final_gates"],
            "selection_rules": {
                "threshold": list(THRESHOLD_SELECTION_RULE),
                "arm": list(ARM_SELECTION_RULE),
            },
        }
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": status,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "workers": worker_count,
        "workers_enter_run_fingerprint": True,
        "input_integrity": {
            "registered_sha256": input_sha,
            **state_integrity,
            "source_h1_manifest_sha256": state_manifest_sha,
            "consensus_support": observed_support,
            "inactive_nonconsensus_pair_count": inactive_pair_count,
            "inactive_nonconsensus_state_count": inactive_state_count,
            "development_row_count": 0,
            "stage2_row_count": 0,
        },
        "consensus_contract": {
            **dict(config["consensus_filter"]),
            "observed_support": observed_support,
            "all_consensus_states_have_one_pair": True,
            "all_component_hotspot_ids_and_agents_equal": True,
            "all_structural_v2_ids_and_agents_differ": True,
        },
        "feature_contract": {
            "dimension": 18,
            "signed_feature_names": list(FEATURE_NAMES),
            "opportunity_views": list(OPPORTUNITY_VIEWS),
            "direction_view": "signed18",
            "abs_delta_indices": list(DELTA_INDICES),
            "identifier_or_outcome_runtime_feature_count": 0,
        },
        "head_fit_contract": dict(config["head_contract"]),
        "fit_support": {
            "opportunity_all_rows": 83,
            "opportunity_structural_rows": 14,
            "opportunity_ambiguous_rows": 69,
            "direction_all_rows": 24,
            "direction_structural_rows": 14,
            "direction_v2_rows": 10,
            "minimum_inner_direction_fit_structural_count": (
                min(all_inner_fit_s_counts) if all_inner_fit_s_counts else None
            ),
            "minimum_inner_direction_fit_v2_count": (
                min(all_inner_fit_v_counts) if all_inner_fit_v_counts else None
            ),
            "ambiguous_rows_in_direction_fit": 0,
            "v2_rows_in_opportunity_fit": 0,
        },
        "nested_cv": {
            "outer_fold_count": 4,
            "inner_fold_count_per_outer": 3,
            "outer_labels_never_used_for_selection": True,
            "single_global_threshold_pair_per_outer_fold": True,
            "inactive_outer_fold_exact_v2_fail_closed": True,
            "inactive_outer_fold_not_a_breadth_win": True,
            "inactive_outer_fold_still_in_full_pooled_metrics": True,
            "outer_folds": outer_records,
        },
        "primary_full_pooled_selected_action_metrics": overall,
        "diagnostic_active_outer_subset_metrics": active_diagnostic,
        "diagnostic_by_fold": by_fold,
        "diagnostic_by_map": by_map,
        "fold_missing_class_semantics": {
            "v2_protection": "null_when_no_v2_win",
            "balanced_accuracy": "null_unless_both_S_and_V_present",
            "no_imputed_half_score": True,
            "no_per_fold_hard_gate": True,
        },
        "final_gate_checks": checks,
        "hard_gates_passed": passed,
        "threshold_selection_rule": list(THRESHOLD_SELECTION_RULE),
        "arm_selection_rule": list(ARM_SELECTION_RULE),
        "full_fit_model_executed": False,
        "model_exported": False,
        "development_rows_loaded": False,
        "stage2_rows_loaded": False,
        "promotion_or_default_exported": False,
        "v2_replaced": False,
        "runtime_or_ttf_claim_authorized": False,
        "final_claim_authorized": False,
        "next_decision": (
            "retain_consensus_train_oof_challenger_pending_map_disjoint_confirmation"
            if passed
            else "retain_exact_v2_for_consensus_stratum_no_go"
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
            "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_consensus_only_feature_manifest.v1"
        ),
        "run_fingerprint": run_fingerprint,
        "dimension": 18,
        "signed_feature_names": list(FEATURE_NAMES),
        "opportunity_feature_views": list(OPPORTUNITY_VIEWS),
        "direction_feature_view": "signed18",
        "leakage_audit_passed": True,
        "stage2_rows_loaded": False,
        "development_partition_present": False,
    }
    feature_path = output_root / outputs["feature_manifest"]
    prediction_path = output_root / outputs["oof_predictions"]
    report_path = output_root / outputs["evaluation_report"]
    _write_json(feature_path, feature_manifest)
    output_rows = [
        _prediction_output(row, run_fingerprint)
        for row in sorted(predictions, key=lambda item: str(item["pair_id"]))
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
    "ARM_SELECTION_RULE",
    "CANDIDATE_ARMS",
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "DELTA_INDICES",
    "DIRECTION_THRESHOLDS",
    "EXPERIMENT_ID",
    "FEATURE_NAMES",
    "FINAL_GATES",
    "FOLDS",
    "GATE_THRESHOLDS",
    "INNER_GATES",
    "OPPORTUNITY_VIEWS",
    "REPORT_SCHEMA",
    "build_consensus_example",
    "cross_validate_head",
    "materialize_action",
    "run_consensus_only",
    "select_inner_arm",
    "select_joint_thresholds",
    "selected_action_metrics",
    "validate_config",
]
