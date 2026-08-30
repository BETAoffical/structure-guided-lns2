"""Retrospective map-disjoint development screen for the fixed Stage1 guard.

The policy is fit once on the frozen 93-state exact-consensus cohort.  External
pre-action features and unlabeled predictions are frozen before the raw H1
state payloads are opened.  This is intentionally not a prospective, sealed,
promotion, deployment, or runtime experiment.
"""

from __future__ import annotations

import collections
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
import experiments.stride_fresh_matched_unique_action_hierarchical_overlay_v1 as overlay
import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_v1 as ambiguity
import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_only_v1 as frozen
import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_two_head_v1 as two_head
import experiments.stride_hierarchical_ch_compact_flow_expanded_train_v1 as expanded
import experiments.stride_hierarchical_ch_compact_flow_stage1_router_v1 as base
import experiments.stride_hierarchical_ch_labels_v1 as h1_labels


CONFIG_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_stage1_consensus_selective_guard_external_screen_config.v1"
)
POLICY_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_stage1_consensus_selective_guard_policy.v1"
)
PREDICTION_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_stage1_consensus_selective_guard_external_prediction.v1"
)
REPORT_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_stage1_consensus_selective_guard_external_screen_report.v1"
)
EXPERIMENT_ID = (
    "stride_hierarchical_ch_compact_flow_stage1_consensus_selective_guard_external_screen_v1"
)
SCIENTIFIC_STATUS = "retrospective_current_model_map_disjoint_development_screen"
DEFAULT_CONFIG = (
    "configs/stride_hierarchical_ch_compact_flow_stage1_consensus_selective_guard_external_screen_v1.json"
)
DEFAULT_OUTPUT = (
    "build/stride-hierarchical-ch-compact-flow-stage1-consensus-selective-guard-external-screen-v1"
)
FEATURE_NAMES = frozen.FEATURE_NAMES
MODEL = frozen.LOGISTIC_MODEL
OPPORTUNITY_THRESHOLD = 0.40
DIRECTION_THRESHOLD = 0.75
TRAINING_SUPPORT = {
    "row_count": 93,
    "label_counts": {"ambiguous": 69, "structural_win": 14, "v2_win": 10},
}
EXTERNAL_SUPPORT = {
    "state_count": 32,
    "states_per_fold": 8,
    "map_count": 7,
    "family_count": 4,
    "label_counts": {"ambiguous": 21, "structural_win": 8, "v2_win": 3},
}
EXTERNAL_FOLDS = {
    "fold0": (
        "den520d",
        "maze-128-128-10",
        "warehouse-10-20-10-2-1",
        "maze-128-128-1",
    ),
    "fold1": (
        "lak303d",
        "random-32-32-10",
        "room-32-32-4",
        "random-32-32-20",
    ),
    "fold2": (
        "maze-32-32-2",
        "random-64-64-10",
        "warehouse-20-40-10-2-1",
        "maze-32-32-4",
    ),
    "fold3": (
        "random-64-64-20",
        "room-64-64-8",
        "warehouse-20-40-10-2-2",
        "room-64-64-16",
    ),
}
TRAINING_REGISTERED_MAPS = tuple(
    sorted({map_id for maps in frozen.FOLDS.values() for map_id in maps})
)
GATES = {
    "minimum_total_override_count": 8,
    "minimum_override_map_count": 2,
    "minimum_selected_action_precision": 0.60,
    "maximum_v2_wrong_override_count": 0,
    "maximum_ambiguous_override_rate": 0.10,
    "minimum_structural_win_override_recall": 0.35,
    "minimum_full_pooled_decisive_balanced_accuracy": 0.55,
    "require_exact_v2_fallback_semantics_rate": 1.0,
}
CLAIM_BOUNDARY = {
    "identity": SCIENTIFIC_STATUS,
    "current_model_fixed_guard_only": True,
    "aggregate_label_support_already_seen": True,
    "prospective_claim_allowed": False,
    "sealed_final_or_confirmation_claim_allowed": False,
    "stage2_loaded": False,
    "model_export_allowed": False,
    "deployment_allowed": False,
    "runtime_or_ttf_claim_allowed": False,
    "promotion_allowed": False,
    "v2_replacement_allowed": False,
    "pass_status": "RETROSPECTIVE_GUARD_SUPPORT_ONLY",
    "failure_status": "RETROSPECTIVE_GUARD_NO_GO",
    "insufficient_support_status": "INSUFFICIENT_SUPPORT",
}


def _folds(value: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(value).items()
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or config.get("registration_status") != "REGISTERED_NOT_RUN"
    ):
        raise ValueError("external guard identity changed")
    required_inputs = {
        "training_controller_config",
        "expanded_train_report",
        "expanded_stage1_pair_labels",
        "training_h1_collection_report",
        "external_overlay_config",
        "external_selection_report",
        "external_selected_states",
        "external_h1_collection_report",
        "external_h1_state_manifest",
    }
    if set(config.get("inputs") or {}) != required_inputs:
        raise ValueError("external guard input registry changed")
    training = dict(config.get("training_contract") or {})
    if training != {
        "population": "current_93_exact_consensus_states",
        "expected_state_count": 93,
        "expected_label_counts": TRAINING_SUPPORT["label_counts"],
        "full_fit_once": True,
        "cross_validation_or_search": False,
        "external_rows_used_for_fit_threshold_or_selection": False,
    }:
        raise ValueError("external guard training contract changed")
    features = dict(config.get("feature_contract") or {})
    if features != {
        "view": "signed18",
        "dimension": 18,
        "feature_names": list(FEATURE_NAMES),
        "repair_outcome_or_runtime_used_as_feature": False,
        "identifier_used_as_feature": False,
    }:
        raise ValueError("external guard feature contract changed")
    if dict(config.get("heads") or {}) != {
        "opportunity": {
            "target": "decisive_structural_or_v2_vs_ambiguous",
            "fit_labels": ["structural_win", "v2_win", "ambiguous"],
            "class_balance": False,
            "threshold": OPPORTUNITY_THRESHOLD,
        },
        "direction": {
            "target": "structural_win_vs_v2_win",
            "fit_labels": ["structural_win", "v2_win"],
            "ambiguous_rows_in_fit": False,
            "class_balance": True,
            "threshold": DIRECTION_THRESHOLD,
        },
    }:
        raise ValueError("external guard heads changed")
    if dict(config.get("model") or {}) != MODEL:
        raise ValueError("external guard model changed")
    external = dict(config.get("external_contract") or {})
    if external != {
        "source_split": "fresh_matched_development",
        "filter": "exact_C_equals_H_not_V_consensus_only",
        "expected_state_count": 32,
        "expected_states_per_fold": 8,
        "expected_map_count": 7,
        "expected_family_count": 4,
        "expected_label_counts_already_seen_in_feasibility_audit": EXTERNAL_SUPPORT[
            "label_counts"
        ],
        "aggregate_label_support_was_known_before_registration": True,
        "prospective_or_sealed_final": False,
        "map_disjoint_from_training_required": True,
        "labels_loaded_only_after_policy_and_unlabeled_predictions_frozen": True,
    }:
        raise ValueError("external guard screen contract changed")
    folds = _folds(config.get("external_map_folds") or {})
    if folds != EXTERNAL_FOLDS:
        raise ValueError("external guard folds changed")
    flattened_maps = [map_id for maps in folds.values() for map_id in maps]
    if len(flattened_maps) != 16 or len(set(flattened_maps)) != 16:
        raise ValueError("external guard fold shape changed")
    if dict(config.get("gates") or {}) != GATES:
        raise ValueError("external guard gates changed")
    if dict(config.get("execution") or {}) != {
        "workers": 16,
        "maximum_workers": 20,
        "solver_invocation_allowed": False,
        "threshold_model_or_view_search_allowed": False,
    }:
        raise ValueError("external guard execution changed")
    if dict(config.get("outputs") or {}) != {
        "root": DEFAULT_OUTPUT,
        "policy_manifest": "policy_manifest.json",
        "predictions": "predictions.jsonl",
        "report": "report.json",
    }:
        raise ValueError("external guard outputs changed")
    if dict(config.get("claim_boundary") or {}) != CLAIM_BOUNDARY:
        raise ValueError("external guard claim boundary changed")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_config(config)
    return config_path, project_root, config


def _registered_inputs(
    project_root: Path, config: Mapping[str, Any]
) -> dict[str, Path]:
    return {
        name: base._registered_input(project_root, specification, field=name)
        for name, specification in dict(config["inputs"]).items()
    }


def _canonical_agents(value: Any, *, field: str) -> list[int]:
    raw = list(value or ())
    agents = [int(agent) for agent in raw]
    if not agents or len(set(agents)) != len(agents) or any(agent < 0 for agent in agents):
        raise ValueError(f"{field} has empty, negative, or duplicate agents")
    return sorted(agents)


def _load_training_examples(
    inputs: Mapping[str, Path], workers: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    controller = _read_json(inputs["training_controller_config"])
    ambiguity.validate_config(controller)
    expected_specs = {
        "expanded_train_report": controller["inputs"]["expanded_train_report"],
        "expanded_stage1_pair_labels": controller["inputs"][
            "expanded_stage1_pair_labels"
        ],
        "training_h1_collection_report": controller["inputs"][
            "h1_collection_report"
        ],
    }
    for name, source_spec in expected_specs.items():
        registered_name = (
            "h1_collection_report"
            if name == "training_h1_collection_report"
            else name
        )
        if sha256_file(inputs[name]) != str(source_spec["sha256"]):
            raise ValueError(f"training controller input changed: {registered_name}")
    adapter_report = _read_json(inputs["expanded_train_report"])
    two_head._validate_adapter_report(
        adapter_report,
        pair_labels_path=inputs["expanded_stage1_pair_labels"],
    )
    states, state_integrity = base._load_states(
        inputs["training_h1_collection_report"], workers
    )
    provenance = frozen._load_manifest_index(
        inputs["training_h1_collection_report"]
    )
    manifest_sha = str(state_integrity["state_manifest_sha256"])
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _read_jsonl(inputs["expanded_stage1_pair_labels"]):
        by_state[str(row.get("state_occurrence_id", ""))].append(row)
    examples = []
    for state_id, state in sorted(states.items()):
        if frozen._consensus_state_contract(state) is None:
            continue
        rows = by_state.get(state_id, [])
        if len(rows) != 1:
            raise ValueError(f"training consensus state must have one pair: {state_id}")
        if str(rows[0].get("source_h1_manifest_sha256", "")) != manifest_sha:
            raise ValueError(f"training H1 provenance changed: {state_id}")
        examples.append(
            ambiguity.build_control_example(rows[0], state, provenance[state_id])
        )
    labels = collections.Counter(str(row["label"]) for row in examples)
    if len(examples) != 93 or dict(sorted(labels.items())) != TRAINING_SUPPORT[
        "label_counts"
    ]:
        raise ValueError("training 93-state support changed")
    if len({row["state_occurrence_id"] for row in examples}) != 93:
        raise ValueError("training states are not unique")
    cohort_fingerprint = _fingerprint(
        [
            {
                "pair_id": row["pair_id"],
                "state_occurrence_id": row["state_occurrence_id"],
                "map_id": row["map_id"],
                "label": row["label"],
                "features": row["opportunity_features_signed18"],
            }
            for row in sorted(examples, key=lambda item: str(item["pair_id"]))
        ]
    )
    feature_fingerprint = _fingerprint(
        [
            {
                "pair_id": row["pair_id"],
                "signed18": row["opportunity_features_signed18"],
            }
            for row in sorted(examples, key=lambda item: str(item["pair_id"]))
        ]
    )
    return examples, {
        "cohort_fingerprint": cohort_fingerprint,
        "feature_matrix_fingerprint": feature_fingerprint,
        "map_ids": sorted({str(row["map_id"]) for row in examples}),
        "state_integrity": state_integrity,
    }


def build_external_example(
    row: Mapping[str, Any], map_to_fold: Mapping[str, str]
) -> dict[str, Any]:
    state_id = str(row.get("state_occurrence_id", ""))
    if (
        row.get("schema") != overlay.SELECTION_SCHEMA
        or str(row.get("research_split", "")) != "fresh_matched_development"
        or row.get("target_outcome_fields_read") is not False
        or str(row.get("hierarchical_stratum", "")) != "consensus_structural"
        or int(row.get("unique_action_count", -1)) != 2
    ):
        raise ValueError(f"external consensus pre-action contract changed: {state_id}")
    if any(
        key in row
        for key in (
            "label",
            "trials",
            "normalized_conflict_reduction",
            "outcome",
            "result",
        )
    ):
        raise ValueError(f"external outcome entered pre-action row: {state_id}")
    map_id = str(row.get("map_id", ""))
    if not state_id or map_id not in map_to_fold:
        raise ValueError(f"external state identity/fold changed: {state_id}")
    role_map = {
        str(role): str(action_id)
        for role, action_id in dict(row.get("role_to_action_id") or {}).items()
    }
    if set(role_map) != {"component16", "hotspot16", "v2_anchor"}:
        raise ValueError(f"external role map changed: {state_id}")
    structural_id = role_map["component16"]
    v2_id = role_map["v2_anchor"]
    if not structural_id or role_map["hotspot16"] != structural_id or structural_id == v2_id:
        raise ValueError(f"external exact C=H!=V contract changed: {state_id}")
    action_rows = list(row.get("unique_actions") or ())
    action_ids = [str(action.get("action_id", "")) for action in action_rows]
    if len(action_rows) != 2 or any(not action_id for action_id in action_ids) or len(set(action_ids)) != 2:
        raise ValueError(f"external unique action list changed: {state_id}")
    actions = {
        str(action["action_id"]): _canonical_agents(
            action.get("agents"), field=f"{state_id}/{action['action_id']}"
        )
        for action in action_rows
    }
    if set(actions) != {structural_id, v2_id}:
        raise ValueError(f"external action identities changed: {state_id}")
    arms = dict(row.get("arms") or {})
    if set(arms) != {"component16", "hotspot16", "v2_anchor"}:
        raise ValueError(f"external arms changed: {state_id}")
    structural_agents = actions[structural_id]
    v2_agents = actions[v2_id]
    if structural_agents == v2_agents:
        raise ValueError(f"external structural/V2 exact sets are equal: {state_id}")
    for role, expected_agents in (
        ("component16", structural_agents),
        ("hotspot16", structural_agents),
        ("v2_anchor", v2_agents),
    ):
        if _canonical_agents(arms[role].get("agents"), field=f"{state_id}/{role}") != expected_agents:
            raise ValueError(f"external arm/action agents differ: {state_id}/{role}")
    anchor = expanded._arm_features(row, "v2_anchor")
    structural_views = [
        expanded._arm_features(row, role)
        for role in ("component16", "hotspot16")
    ]
    for name in expanded.CONTEXT_FIELDS:
        anchor_value = expanded._finite(anchor.get(name), field=f"v2/{name}")
        if any(
            not math.isclose(
                expanded._finite(view.get(name), field=f"structural/{name}"),
                anchor_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for view in structural_views
        ):
            raise ValueError(f"external shared context differs: {state_id}/{name}")
    values = [
        expanded._finite(anchor.get(name), field=name)
        for name in expanded.CONTEXT_FIELDS
    ]
    for name in expanded.DELTA_FIELDS:
        structural_values = [
            expanded._finite(view.get(name), field=f"structural/{name}")
            for view in structural_views
        ]
        if not math.isclose(
            structural_values[0],
            structural_values[1],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"external C/H alias feature differs: {state_id}/{name}")
        structural_value = structural_values[0]
        values.append(
            structural_value - expanded._finite(anchor.get(name), field=f"v2/{name}")
        )
    values.extend(expanded._relation_features(structural_agents, v2_agents))
    if len(values) != 18 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"external signed18 vector changed: {state_id}")
    return {
        "state_occurrence_id": state_id,
        "pair_id": f"{state_id}::{structural_id}::{v2_id}",
        "map_id": map_id,
        "map_family": str(row.get("map_family", "")),
        "train_fold": str(map_to_fold[map_id]),
        "research_split": "fresh_matched_development",
        "structural_action_id": structural_id,
        "v2_action_id": v2_id,
        "component_action_id": structural_id,
        "hotspot_action_id": structural_id,
        "exact_structural_agents": structural_agents,
        "exact_v2_agents": v2_agents,
        "features": values,
        "model_feature_names": list(FEATURE_NAMES),
        "repair_outcome_or_runtime_used_as_model_feature": False,
    }


def _load_external_preaction(
    inputs: Mapping[str, Path], config: Mapping[str, Any], project_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    overlay_config = _read_json(inputs["external_overlay_config"])
    overlay.validate_config(overlay_config, project_root=project_root)
    if _folds(overlay_config["map_folds"]) != _folds(config["external_map_folds"]):
        raise ValueError("external map folds differ from overlay")
    selection = _read_json(inputs["external_selection_report"])
    if (
        selection.get("schema") != overlay.SELECTION_REPORT_SCHEMA
        or selection.get("experiment_id") != overlay.EXPERIMENT_ID
        or selection.get("status") != "ok"
        or selection.get("dry_run") is not False
        or int(selection.get("selected_state_count", -1)) != 96
        or selection.get("config_sha256") != sha256_file(inputs["external_overlay_config"])
        or selection.get("selected_states_sha256")
        != sha256_file(inputs["external_selected_states"])
        or selection.get("outcome_fields_read") is not False
    ):
        raise ValueError("external selection trust chain changed")
    map_to_fold = {
        map_id: fold
        for fold, maps in _folds(config["external_map_folds"]).items()
        for map_id in maps
    }
    selected = _read_jsonl(inputs["external_selected_states"])
    if len(selected) != 96 or len({row.get("state_occurrence_id") for row in selected}) != 96:
        raise ValueError("external selected-state support changed")
    examples = [
        build_external_example(row, map_to_fold)
        for row in selected
        if str(row.get("hierarchical_stratum", "")) == "consensus_structural"
    ]
    fold_counts = collections.Counter(row["train_fold"] for row in examples)
    if (
        len(examples) != EXTERNAL_SUPPORT["state_count"]
        or any(
            fold_counts[fold] != EXTERNAL_SUPPORT["states_per_fold"]
            for fold in EXTERNAL_FOLDS
        )
        or len({row["map_id"] for row in examples}) != EXTERNAL_SUPPORT["map_count"]
        or len({row["map_family"] for row in examples}) != EXTERNAL_SUPPORT["family_count"]
    ):
        raise ValueError("external consensus support changed")
    return examples, {
        "preaction_feature_fingerprint": _fingerprint(
            [
                {
                    "state_occurrence_id": row["state_occurrence_id"],
                    "pair_id": row["pair_id"],
                    "features": row["features"],
                }
                for row in sorted(examples, key=lambda item: item["state_occurrence_id"])
            ]
        ),
        "map_ids": sorted({row["map_id"] for row in examples}),
        "family_ids": sorted({row["map_family"] for row in examples}),
        "fold_counts": {fold: int(fold_counts[fold]) for fold in sorted(set(fold_counts))},
    }


def _estimator_parameters(estimator: Any) -> dict[str, Any]:
    scaler = estimator.named_steps["scale"]
    model = estimator.named_steps["model"]
    return {
        "feature_names": list(FEATURE_NAMES),
        "scaler_mean": [float(value) for value in scaler.mean_],
        "scaler_scale": [float(value) for value in scaler.scale_],
        "logistic_classes": [int(value) for value in model.classes_],
        "logistic_coefficients": [float(value) for value in model.coef_[0]],
        "logistic_intercept": float(model.intercept_[0]),
    }


def _policy_identity_from_manifest(manifest: Mapping[str, Any]) -> str:
    policy = dict(manifest.get("policy") or {})
    if not policy or any(
        key.startswith("external_") or key.startswith("evaluation_")
        for key in policy
    ) or {
        "experiment_id",
        "scientific_status",
        "screen_only",
        "model_exported",
        "deployment_authorized",
    } & set(policy):
        raise ValueError("policy identity contains external evaluation binding")
    return _fingerprint(policy)


def _unlabeled_predictions(
    examples: Sequence[Mapping[str, Any]],
    opportunity_estimator: Any,
    direction_estimator: Any,
    policy_fingerprint: str,
) -> list[dict[str, Any]]:
    opportunity = frozen._predict_probability(opportunity_estimator, examples, "features")
    direction = frozen._predict_probability(direction_estimator, examples, "features")
    rows = []
    for example, p_opportunity, p_direction in zip(examples, opportunity, direction):
        override = bool(
            p_opportunity + 1e-12 >= OPPORTUNITY_THRESHOLD
            and p_direction + 1e-12 >= DIRECTION_THRESHOLD
        )
        rows.append(
            {
                **dict(example),
                "policy_fingerprint": policy_fingerprint,
                "opportunity_decisive_probability": float(p_opportunity),
                "opportunity_decisive_probability_semantics": (
                    "probability_of_structural_or_v2_decisive_vs_ambiguous"
                ),
                "direction_structural_probability": float(p_direction),
                "opportunity_threshold": OPPORTUNITY_THRESHOLD,
                "direction_threshold": DIRECTION_THRESHOLD,
                "structural_override": override,
                "final_action_id": str(
                    example["structural_action_id"] if override else example["v2_action_id"]
                ),
                "final_action_role": (
                    "consensus_structural_override" if override else "exact_v2_fallback"
                ),
                "final_action_agents": list(
                    example["exact_structural_agents"]
                    if override
                    else example["exact_v2_agents"]
                ),
                "evaluation_active": True,
                "label_loaded": False,
            }
        )
    return rows


def _validate_raw_state_against_prediction(
    prediction: Mapping[str, Any], state: Mapping[str, Any]
) -> None:
    state_id = str(prediction["state_occurrence_id"])
    actions = dict(state.get("actions") or {})
    structural_id = str(prediction["structural_action_id"])
    v2_id = str(prediction["v2_action_id"])
    if set(actions) != {structural_id, v2_id}:
        raise ValueError(f"external raw H1 action IDs changed: {state_id}")
    raw_structural = _canonical_agents(
        actions[structural_id], field=f"{state_id}/raw_structural"
    )
    raw_v2 = _canonical_agents(actions[v2_id], field=f"{state_id}/raw_v2")
    if (
        raw_structural != list(prediction["exact_structural_agents"])
        or raw_v2 != list(prediction["exact_v2_agents"])
    ):
        raise ValueError(f"external raw H1 exact agents changed: {state_id}")
    rebuilt = build_external_example(
        dict(state.get("state_row") or {}),
        {str(prediction["map_id"]): str(prediction["train_fold"])},
    )
    if (
        rebuilt["state_occurrence_id"] != state_id
        or rebuilt["pair_id"] != prediction["pair_id"]
        or rebuilt["structural_action_id"] != structural_id
        or rebuilt["v2_action_id"] != v2_id
        or len(rebuilt["features"]) != len(prediction["features"])
        or any(
            not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(rebuilt["features"], prediction["features"])
        )
    ):
        raise ValueError(f"external raw H1 pre-action features changed: {state_id}")


def _load_external_labels_after_prediction(
    inputs: Mapping[str, Path],
    expected_predictions: Sequence[Mapping[str, Any]],
    overlay_config_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    collection = _read_json(inputs["external_h1_collection_report"])
    if (
        collection.get("schema") != overlay.H1_COLLECTION_SCHEMA
        or collection.get("experiment_id") != overlay.EXPERIMENT_ID
        or collection.get("status") != "complete"
        or collection.get("complete") is not True
        or collection.get("config_sha256") != overlay_config_sha256
        or collection.get("selection_identity")
        != sha256_file(inputs["external_selected_states"])
        or int(collection.get("observed_valid_state_count", -1)) != 96
        or int(collection.get("observed_unique_action_count", -1)) != 256
        or int(collection.get("observed_trial_count", -1)) != 4096
        or int(collection.get("invalid_state_count", -1)) != 0
        or collection.get("h8_or_training_or_ttf_executed") is not False
    ):
        raise ValueError("external H1 collection trust chain changed")
    manifest = _read_jsonl(inputs["external_h1_state_manifest"])
    if len(manifest) != 96:
        raise ValueError("external H1 manifest support changed")
    manifest_by_state = {
        str(row.get("state_occurrence_id", "")): row for row in manifest
    }
    if len(manifest_by_state) != 96:
        raise ValueError("external H1 manifest state IDs are duplicated")
    prediction_by_state = {
        str(row["state_occurrence_id"]): row for row in expected_predictions
    }
    if len(prediction_by_state) != 32:
        raise ValueError("external unlabeled prediction support changed")
    manifest_root = inputs["external_h1_state_manifest"].parent
    labels: dict[str, str] = {}
    state_hashes: dict[str, str] = {}
    for state_id in sorted(prediction_by_state):
        manifest_row = manifest_by_state.get(state_id)
        if manifest_row is None:
            raise ValueError(f"external H1 state is missing: {state_id}")
        state = h1_labels._validated_state(
            manifest_root,
            manifest_row,
            source_state_schema=h1_labels.SOURCE_STATE_SCHEMA,
        )
        if state["hierarchical_stratum"] != "consensus_structural":
            raise ValueError(f"external non-consensus H1 state loaded: {state_id}")
        prediction = prediction_by_state[state_id]
        _validate_raw_state_against_prediction(prediction, state)
        pair_rows, _state_row = h1_labels._stage1_rows(state)
        if len(pair_rows) != 1:
            raise ValueError(f"external consensus H1 pair count changed: {state_id}")
        pair = pair_rows[0]
        if (
            str(pair["pair_id"]) != str(prediction["pair_id"])
            or str(pair["structural_action_id"])
            != str(prediction["structural_action_id"])
            or str(pair["v2_action_id"]) != str(prediction["v2_action_id"])
            or pair.get("runtime_or_pp_seconds_used_in_label") is not False
        ):
            raise ValueError(f"external H1 label/action contract changed: {state_id}")
        labels[state_id] = str(pair["label"])
        state_hashes[state_id] = str(manifest_row["state_sha256"])
    counts = collections.Counter(labels.values())
    if dict(sorted(counts.items())) != EXTERNAL_SUPPORT["label_counts"]:
        raise ValueError("known retrospective external label support changed")
    return labels, {
        "raw_h1_state_files_loaded": 32,
        "three_unique_or_stage2_state_files_loaded": 0,
        "state_sha256": state_hashes,
        "label_counts": dict(sorted(counts.items())),
    }


def _merge_labels(
    predictions: Sequence[Mapping[str, Any]], labels: Mapping[str, str]
) -> list[dict[str, Any]]:
    if set(labels) != {str(row["state_occurrence_id"]) for row in predictions}:
        raise ValueError("external labels do not exactly cover predictions")
    return [
        {
            **dict(row),
            "label": str(labels[str(row["state_occurrence_id"])]),
            "label_loaded": True,
            "external_label_used_for_fit_threshold_or_selection": False,
        }
        for row in predictions
    ]


def _gate_checks(
    metrics: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> dict[str, bool]:
    override_maps = {
        str(row["map_id"]) for row in predictions if bool(row["structural_override"])
    }
    return {
        "minimum_total_override_count": int(metrics["total_override_count"])
        >= GATES["minimum_total_override_count"],
        "minimum_override_map_count": len(override_maps)
        >= GATES["minimum_override_map_count"],
        "minimum_selected_action_precision": (
            metrics.get("selected_action_precision") is not None
            and float(metrics["selected_action_precision"])
            >= GATES["minimum_selected_action_precision"] - 1e-12
        ),
        "maximum_v2_wrong_override_count": int(metrics["v2_wrong_override_count"])
        <= GATES["maximum_v2_wrong_override_count"],
        "maximum_ambiguous_override_rate": (
            metrics.get("ambiguous_override_rate") is not None
            and float(metrics["ambiguous_override_rate"])
            <= GATES["maximum_ambiguous_override_rate"] + 1e-12
        ),
        "minimum_structural_win_override_recall": (
            metrics.get("structural_win_override_recall") is not None
            and float(metrics["structural_win_override_recall"])
            >= GATES["minimum_structural_win_override_recall"] - 1e-12
        ),
        "minimum_full_pooled_decisive_balanced_accuracy": (
            metrics.get("full_pooled_decisive_balanced_accuracy") is not None
            and float(metrics["full_pooled_decisive_balanced_accuracy"])
            >= GATES["minimum_full_pooled_decisive_balanced_accuracy"] - 1e-12
        ),
        "require_exact_v2_fallback_semantics_rate": (
            metrics.get("exact_v2_fallback_semantics_rate") is not None
            and math.isclose(
                float(metrics["exact_v2_fallback_semantics_rate"]),
                GATES["require_exact_v2_fallback_semantics_rate"],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
    }


def _prediction_output(
    row: Mapping[str, Any], run_fingerprint: str, unlabeled_fingerprint: str
) -> dict[str, Any]:
    return {
        "schema": PREDICTION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_fingerprint": run_fingerprint,
        "policy_fingerprint": str(row["policy_fingerprint"]),
        "unlabeled_prediction_fingerprint": unlabeled_fingerprint,
        "state_occurrence_id": str(row["state_occurrence_id"]),
        "pair_id": str(row["pair_id"]),
        "map_id": str(row["map_id"]),
        "map_family": str(row["map_family"]),
        "external_fold": str(row["train_fold"]),
        "research_split": "fresh_matched_development",
        "label": str(row["label"]),
        "structural_action_id": str(row["structural_action_id"]),
        "v2_action_id": str(row["v2_action_id"]),
        "component_action_id": str(row["component_action_id"]),
        "hotspot_action_id": str(row["hotspot_action_id"]),
        "exact_structural_agents": list(row["exact_structural_agents"]),
        "exact_v2_agents": list(row["exact_v2_agents"]),
        "opportunity_decisive_probability": float(
            row["opportunity_decisive_probability"]
        ),
        "opportunity_decisive_probability_semantics": str(
            row["opportunity_decisive_probability_semantics"]
        ),
        "direction_structural_probability": float(
            row["direction_structural_probability"]
        ),
        "opportunity_threshold": OPPORTUNITY_THRESHOLD,
        "direction_threshold": DIRECTION_THRESHOLD,
        "structural_override": bool(row["structural_override"]),
        "final_action_id": str(row["final_action_id"]),
        "final_action_role": str(row["final_action_role"]),
        "final_action_agents": list(row["final_action_agents"]),
        "external_label_used_for_fit_threshold_or_selection": False,
        "stage2_loaded": False,
        "model_exported": False,
        "deployment_or_promotion_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "prospective_or_sealed_final": False,
    }


def run_external_screen(
    config_path: str | Path = DEFAULT_CONFIG,
    output: str | Path | None = None,
    *,
    workers: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path, project_root, config = load_config(config_path)
    worker_count = int(config["execution"]["workers"] if workers is None else workers)
    if worker_count <= 0 or worker_count > int(config["execution"]["maximum_workers"]):
        raise ValueError("workers must be in 1..20")
    inputs = _registered_inputs(project_root, config)
    input_sha = {name: sha256_file(path) for name, path in inputs.items()}

    # Pre-action-only stage.  No raw external H1 state payload is opened here.
    training, training_audit = _load_training_examples(inputs, worker_count)
    external, external_audit = _load_external_preaction(inputs, config, project_root)
    training_active_maps = set(training_audit["map_ids"])
    training_registered_maps = set(TRAINING_REGISTERED_MAPS)
    external_active_maps = set(external_audit["map_ids"])
    external_registered_maps = {
        map_id for maps in EXTERNAL_FOLDS.values() for map_id in maps
    }
    if (
        len(training_registered_maps) != 16
        or len(training_active_maps) != 15
        or not training_active_maps <= training_registered_maps
        or len(external_registered_maps) != 16
        or training_registered_maps & external_registered_maps
        or training_registered_maps & external_active_maps
    ):
        raise ValueError("external screen map-disjoint registry contract changed")
    opportunity_estimator, opportunity_weight_audit = ambiguity._fit_outer_head(
        training, "opportunity", "signed18"
    )
    direction_estimator, direction_weight_audit = frozen._fit_outer_head(
        training, "direction", "signed18"
    )
    policy_core = {
        "policy_semantics": (
            "signed18_fullfit93_opportunity_decisive_and_direction_structural_guard_v1"
        ),
        "training_cohort_fingerprint": training_audit["cohort_fingerprint"],
        "training_feature_matrix_fingerprint": training_audit[
            "feature_matrix_fingerprint"
        ],
        "feature_view": "signed18",
        "model": MODEL,
        "opportunity_threshold": OPPORTUNITY_THRESHOLD,
        "direction_threshold": DIRECTION_THRESHOLD,
        "override_rule": (
            "p_opportunity_decisive>=0.40_and_p_direction_structural>=0.75"
        ),
        "fallback_rule": "otherwise_exact_v2",
        "opportunity": _estimator_parameters(opportunity_estimator),
        "direction": _estimator_parameters(direction_estimator),
        "training_input_sha256": {
            name: input_sha[name]
            for name in (
                "training_controller_config",
                "expanded_train_report",
                "expanded_stage1_pair_labels",
                "training_h1_collection_report",
            )
        },
    }
    policy_fingerprint = _fingerprint(policy_core)
    policy_manifest = {
        "schema": POLICY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "policy_fingerprint": policy_fingerprint,
        "policy": policy_core,
        "evaluation_binding": {
            "external_selected_states_sha256": input_sha[
                "external_selected_states"
            ],
            "external_preaction_feature_fingerprint": external_audit[
                "preaction_feature_fingerprint"
            ],
            "binding_does_not_enter_policy_fingerprint": True,
        },
        "screen_only": True,
        "model_exported": False,
        "deployment_authorized": False,
    }
    if _policy_identity_from_manifest(policy_manifest) != policy_fingerprint:
        raise ValueError("policy manifest identity changed")
    unlabeled = _unlabeled_predictions(
        external,
        opportunity_estimator,
        direction_estimator,
        policy_fingerprint,
    )
    unlabeled_fingerprint = _fingerprint(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"label", "label_loaded"}
            }
            for row in sorted(unlabeled, key=lambda item: item["state_occurrence_id"])
        ]
    )

    # Outcome stage.  Only the 32 consensus raw H1 state files are opened,
    # after the policy and unlabeled predictions have immutable fingerprints.
    labels, label_audit = _load_external_labels_after_prediction(
        inputs,
        unlabeled,
        input_sha["external_overlay_config"],
    )
    predictions = _merge_labels(unlabeled, labels)
    metrics = frozen.selected_action_metrics(predictions)
    metrics["override_map_count"] = len(
        {row["map_id"] for row in predictions if row["structural_override"]}
    )
    metrics["override_maps"] = sorted(
        {row["map_id"] for row in predictions if row["structural_override"]}
    )
    checks = _gate_checks(metrics, predictions)
    insufficient = int(metrics["total_override_count"]) < GATES[
        "minimum_total_override_count"
    ]
    passed = not insufficient and all(checks.values())
    status = (
        "INSUFFICIENT_SUPPORT"
        if insufficient
        else "RETROSPECTIVE_GUARD_SUPPORT_ONLY"
        if passed
        else "RETROSPECTIVE_GUARD_NO_GO"
    )
    run_fingerprint = _fingerprint(
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(config_path),
            "input_sha256": input_sha,
            "policy_fingerprint": policy_fingerprint,
            "unlabeled_prediction_fingerprint": unlabeled_fingerprint,
            "external_label_counts": label_audit["label_counts"],
            "gates": GATES,
            "workers": worker_count,
        }
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": SCIENTIFIC_STATUS,
        "screen_status": status,
        "complete": True,
        "hard_gates_passed": passed,
        "insufficient_support": insufficient,
        "run_fingerprint": run_fingerprint,
        "policy_fingerprint": policy_fingerprint,
        "unlabeled_prediction_fingerprint": unlabeled_fingerprint,
        "input_integrity": {
            "registered_sha256": input_sha,
            "training": training_audit,
            "external_preaction": external_audit,
            "external_label": label_audit,
            "map_disjoint_training_external": True,
            "training_registered_map_count": len(training_registered_maps),
            "training_registered_maps": sorted(training_registered_maps),
            "training_active_map_count": len(training_active_maps),
            "training_active_maps": sorted(training_active_maps),
            "external_registered_map_count": len(external_registered_maps),
            "external_registered_maps": sorted(external_registered_maps),
            "external_active_map_count": len(external_active_maps),
            "external_active_maps": sorted(external_active_maps),
            "registered_map_overlap_count": 0,
            "active_external_vs_registered_training_overlap_count": 0,
        },
        "ordering_audit": {
            "external_preaction_features_built_before_external_h1_state_open": True,
            "full_fit_completed_before_external_h1_state_open": True,
            "policy_fingerprint_frozen_before_external_h1_state_open": True,
            "unlabeled_predictions_frozen_before_external_h1_state_open": True,
            "external_labels_used_for_fit_threshold_or_selection": False,
        },
        "training_contract": dict(config["training_contract"]),
        "fit_audit": {
            "full_fit_count": 1,
            "opportunity": opportunity_weight_audit,
            "direction": direction_weight_audit,
            "cross_validation_or_search_executed": False,
            "threshold_search_executed": False,
            "model_or_view_search_executed": False,
        },
        "external_contract": dict(config["external_contract"]),
        "known_aggregate_label_support": label_audit["label_counts"],
        "primary_selected_action_metrics": metrics,
        "diagnostic_by_fold": frozen._group_metrics(predictions, "train_fold"),
        "diagnostic_by_map": frozen._group_metrics(predictions, "map_id"),
        "gate_checks": checks,
        "claim_boundary": dict(config["claim_boundary"]),
        "full_fit_model_executed": True,
        "policy_manifest_is_screen_only_coefficients": True,
        "model_exported": False,
        "stage2_loaded": False,
        "development_labels_loaded_after_prediction_only": True,
        "deployment_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "promotion_authorized": False,
        "v2_replaced": False,
        "prospective_or_sealed_final_confirmation": False,
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return report
    output_root = Path(output or config["outputs"]["root"])
    if not output_root.is_absolute():
        output_root = (project_root / output_root).resolve()
    else:
        output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError("external guard output already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])
    policy_path = output_root / outputs["policy_manifest"]
    prediction_path = output_root / outputs["predictions"]
    report_path = output_root / outputs["report"]
    _write_json(policy_path, policy_manifest)
    output_rows = [
        _prediction_output(row, run_fingerprint, unlabeled_fingerprint)
        for row in sorted(predictions, key=lambda item: item["state_occurrence_id"])
    ]
    _write_jsonl(prediction_path, output_rows)
    report["artifacts"] = {
        "policy_manifest": {
            "file": policy_path.name,
            "schema": POLICY_SCHEMA,
            "sha256": sha256_file(policy_path),
            "screen_only": True,
            "deployable_model": False,
        },
        "predictions": {
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
    "CONFIG_SCHEMA",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "DIRECTION_THRESHOLD",
    "EXPERIMENT_ID",
    "OPPORTUNITY_THRESHOLD",
    "POLICY_SCHEMA",
    "PREDICTION_SCHEMA",
    "REPORT_SCHEMA",
    "SCIENTIFIC_STATUS",
    "build_external_example",
    "run_external_screen",
    "validate_config",
]
