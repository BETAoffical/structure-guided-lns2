"""Integrity-first 16-map expanded-train adapter for compact-flow H1 labels.

The adapter freezes four folds from the registered map metadata before it
opens any label artifact.  It then reclassifies every source train and
development label as expanded-train while preserving the source split and H1
state provenance.  No map, state, or row is selected from label outcomes.
"""

from __future__ import annotations

import collections
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from experiments._common import sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_hierarchical_ch_compact_flow_h1_collection_v1 import (
    H1_COLLECTION_SCHEMA,
    H1_MANIFEST_ROW_SCHEMA,
    H1_STATE_SCHEMA,
)
from experiments.stride_hierarchical_ch_compact_flow_labels_readiness_v1 import (
    DEVELOPMENT_MAPS,
    LABEL_DIAGNOSTICS_SCHEMA,
    READINESS_SCHEMA,
    TRAIN_MAP_FOLDS as SOURCE_TRAIN_MAP_FOLDS,
    TRAIN_MAPS,
)
from experiments.stride_hierarchical_ch_labels_v1 import (
    STAGE1_LABELS,
    STAGE1_PAIR_SCHEMA,
    STAGE1_STATE_SCHEMA,
    STAGE2_LABELS,
    STAGE2_STATE_SCHEMA,
)


CONFIG_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_expanded_train_config.v1"
REPORT_SCHEMA = "lns2.stride.hierarchical_ch_compact_flow_expanded_train_report.v1"
EXPANDED_STAGE1_PAIR_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_pair_label.v1"
)
EXPANDED_STAGE1_STATE_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage1_state_label.v1"
)
EXPANDED_STAGE2_STATE_SCHEMA = (
    "lns2.stride.hierarchical_ch_compact_flow_expanded_stage2_state_label.v1"
)
EXPERIMENT_ID = "stride_hierarchical_ch_compact_flow_expanded_train_v1"
DEFAULT_CONFIG = "configs/stride_hierarchical_ch_compact_flow_expanded_train_v1.json"
DEFAULT_OUTPUT = "build/stride-hierarchical-ch-compact-flow-expanded-train-v1"
DEFAULT_WORKERS = 16
MAXIMUM_WORKERS = 20

ALL_MAPS = tuple(sorted((*TRAIN_MAPS, *DEVELOPMENT_MAPS)))
FIXED_MAP_FOLDS = {
    "fold0": ("den404d", "den408d", "den207d", "den203d"),
    "fold1": ("lak101d", "den202d", "den998d", "den308d"),
    "fold2": ("lak108d", "den201d", "den009d", "den020d"),
    "fold3": ("lak110d", "ost102d", "hrt002d", "den101d"),
}
FOLD_POLICY_ID = "registered_source_split_capacity_balanced_fixed_v1"
EXPECTED_CAPACITY_PROFILES = {
    "fold0": ("ultra", "small", "medium", "large"),
    "fold1": ("ultra", "small", "medium", "large"),
    "fold2": ("ultra", "small", "medium", "large"),
    "fold3": ("ultra", "ultra", "small", "medium"),
}

CONTEXT_FIELDS = (
    "state.agent_count",
    "state.colliding_pairs",
    "state.conflict_edge_density",
    "state.conflict_event_count",
    "state.conflicting_agent_ratio",
    "state.degree_max",
    "state.largest_component_ratio",
    "state.path_wait_ratio_mean",
)
DELTA_FIELDS = (
    "realized.incident_conflict_coverage",
    "realized.internal_conflict_coverage",
    "realized.boundary_conflict_edges",
    "realized.conflict_degree_mean",
    "realized.delay_mean",
    "realized.path_overlap_mean",
    "realized.path_wait_ratio_mean",
)
RELATION_FIELDS = (
    "intersection_ratio",
    "jaccard",
    "symmetric_difference_ratio",
)
STAGE1_FEATURE_NAMES = tuple(
    [f"context:{name}" for name in CONTEXT_FIELDS]
    + [f"delta_structural_minus_v2:{name}" for name in DELTA_FIELDS]
    + [f"relation:{name}" for name in RELATION_FIELDS]
)
STAGE2_FEATURE_NAMES = tuple(
    [f"context:{name}" for name in CONTEXT_FIELDS]
    + [f"delta_component_minus_hotspot:{name}" for name in DELTA_FIELDS]
    + [f"relation:{name}" for name in RELATION_FIELDS]
)

CLAIM_BOUNDARY = {
    "all_rows_are_expanded_train": True,
    "source_research_split_preserved": True,
    "map_or_fold_selection_by_label_allowed": False,
    "development_evaluation_authorized": False,
    "sealed_final_access_authorized": False,
    "map_disjoint_final_confirmation_required": True,
    "expanded_training_use_authorized": True,
    "model_fit_executed": False,
    "model_exported": False,
    "promotion_authorized": False,
    "runtime_or_ttf_claim_authorized": False,
}


def _contained(root: Path, value: str, *, field: str) -> Path:
    portable = PurePosixPath(str(value).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"{field} must be a contained project-relative path")
    path = (root / Path(*portable.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes project root") from error
    return path


def _contained_child(root: Path, value: str, *, field: str) -> Path:
    portable = PurePosixPath(str(value).replace("\\", "/"))
    if portable.is_absolute() or ".." in portable.parts:
        raise ValueError(f"{field} must be a contained relative path")
    path = (root / Path(*portable.parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{field} escapes its artifact root") from error
    return path


def _require_sha(value: Any, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{field} is not a lowercase SHA256")
    return digest


def _capacity_stratum(free_cells: int) -> str:
    if free_cells <= 400:
        return "ultra"
    if free_cells <= 800:
        return "small"
    if free_cells <= 2000:
        return "medium"
    return "large"


def freeze_registered_folds(
    registration: Mapping[str, Any], fold_config: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze the registered metadata-only folds before any labels are read."""
    if (
        registration.get("schema")
        != "lns2.stride.hierarchical_ch_compact_flow_source_registration_report.v1"
        or registration.get("experiment_id")
        != "stride_hierarchical_ch_compact_flow_source_v1"
        or registration.get("passed") is not True
    ):
        raise ValueError("source registration trust contract changed")
    if (
        fold_config.get("policy_id") != FOLD_POLICY_ID
        or tuple(fold_config.get("inputs_allowed") or ())
        != (
            "map_id",
            "source_research_split",
            "capacity_stratum",
            "capacity_free_cells",
        )
        or fold_config.get("label_or_outcome_inputs_allowed") is not False
        or fold_config.get("folds_frozen_before_label_read") is not True
    ):
        raise ValueError("expanded-train fold derivation contract changed")

    registered = {
        str(map_id): dict(metadata)
        for map_id, metadata in dict(registration.get("map_registration") or {}).items()
    }
    if set(registered) != set(ALL_MAPS):
        raise ValueError("registered 16-map product changed")
    source_counts = collections.Counter()
    metadata_rows: list[dict[str, Any]] = []
    for map_id in ALL_MAPS:
        metadata = registered[map_id]
        split = str(metadata.get("split", ""))
        if split not in {"train", "development"}:
            raise ValueError(f"registered source split changed: {map_id}")
        source_counts[split] += 1
        free_cells = int(metadata.get("capacity_free_cells", -1))
        stratum = str(metadata.get("capacity_stratum", ""))
        if free_cells <= 0 or stratum != _capacity_stratum(free_cells):
            raise ValueError(f"registered capacity stratum changed: {map_id}")
        metadata_rows.append(
            {
                "map_id": map_id,
                "source_research_split": split,
                "capacity_stratum": stratum,
                "capacity_free_cells": free_cells,
            }
        )
    if source_counts != {"train": 8, "development": 8}:
        raise ValueError("registered source split balance changed")

    folds = {
        str(fold): tuple(map(str, maps))
        for fold, maps in dict(fold_config.get("fixed_map_folds") or {}).items()
    }
    if folds != FIXED_MAP_FOLDS:
        raise ValueError("fixed expanded-train folds changed")
    flattened = [map_id for maps in folds.values() for map_id in maps]
    if len(flattened) != 16 or len(set(flattened)) != 16 or set(flattened) != set(ALL_MAPS):
        raise ValueError("expanded-train folds do not cover each registered map once")
    expected_profiles = {
        str(fold): tuple(map(str, profile))
        for fold, profile in dict(
            fold_config.get("expected_capacity_profiles") or {}
        ).items()
    }
    if expected_profiles != EXPECTED_CAPACITY_PROFILES:
        raise ValueError("expanded-train capacity profiles changed")
    expected_split_counts = dict(
        fold_config.get("expected_source_split_counts_per_fold") or {}
    )
    if expected_split_counts != {"train": 2, "development": 2}:
        raise ValueError("expanded-train source split fold balance changed")
    by_id = {row["map_id"]: row for row in metadata_rows}
    for fold, maps in folds.items():
        strata = collections.Counter(by_id[map_id]["capacity_stratum"] for map_id in maps)
        if strata != collections.Counter(expected_profiles[fold]):
            raise ValueError(f"registered capacity profile changed: {fold}")
        splits = collections.Counter(by_id[map_id]["source_research_split"] for map_id in maps)
        if splits != expected_split_counts:
            raise ValueError(f"registered source split profile changed: {fold}")

    map_to_fold = {map_id: fold for fold, maps in folds.items() for map_id in maps}
    payload = {
        "policy_id": FOLD_POLICY_ID,
        "metadata_inputs": metadata_rows,
        "fixed_map_folds": {fold: list(maps) for fold, maps in folds.items()},
        "label_or_outcome_inputs_used": False,
    }
    return {
        **payload,
        "map_to_fold": map_to_fold,
        "folds_frozen_before_label_read": True,
        "fold_assignment_fingerprint": _fingerprint(payload),
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("expanded-train identity changed")
    if tuple(map(str, config.get("registered_maps") or ())) != ALL_MAPS:
        raise ValueError("expanded-train registered map list changed")
    inputs = dict(config.get("inputs") or {})
    expected_files = {
        "label_diagnostics": "label_diagnostics.json",
        "readiness_report": "readiness_report.json",
        "stage1_pair_labels": "stage1_pair_labels.jsonl",
        "stage1_state_labels": "stage1_state_labels.jsonl",
        "stage2_state_labels": "stage2_state_labels.jsonl",
        "h1_collection_report": "h1_collection_report.json",
        "h1_state_manifest": "h1_state_manifest.jsonl",
    }
    for key, expected in expected_files.items():
        spec = dict(inputs.get(key) or {})
        if spec.get("file") != expected:
            raise ValueError(f"expanded-train {key} file identity changed")
        _require_sha(spec.get("sha256"), field=f"inputs.{key}.sha256")
    registration = dict(inputs.get("registration_report") or {})
    if not str(registration.get("path", "")):
        raise ValueError("expanded-train registration report path missing")
    _require_sha(registration.get("sha256"), field="inputs.registration_report.sha256")
    if not str(inputs.get("source_labels_root", "")) or not str(
        inputs.get("h1_collection_root", "")
    ):
        raise ValueError("expanded-train input roots missing")
    if dict(config.get("claim_boundary") or {}) != CLAIM_BOUNDARY:
        raise ValueError("expanded-train claim boundary changed")
    execution = dict(config.get("execution") or {})
    if execution != {
        "workers": DEFAULT_WORKERS,
        "maximum_workers": MAXIMUM_WORKERS,
        "workers_enter_run_fingerprint": True,
        "solver_invocation_allowed": False,
    }:
        raise ValueError("expanded-train execution contract changed")


def _workers(config: Mapping[str, Any], override: int | None) -> int:
    execution = dict(config["execution"])
    value = int(execution["workers"] if override is None else override)
    if value <= 0 or value > int(execution["maximum_workers"]):
        raise ValueError("expanded-train workers out of range")
    return value


def _pinned_file(path: Path, spec: Mapping[str, Any], *, field: str) -> Path:
    if not path.is_file():
        raise ValueError(f"expanded-train {field} missing")
    if sha256_file(path) != _require_sha(spec.get("sha256"), field=f"{field}.sha256"):
        raise ValueError(f"expanded-train {field} SHA256 mismatch")
    return path


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"non-numeric pre-action feature: {field}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite pre-action feature: {field}")
    return result


def _arm_features(state: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    features = dict(dict(dict(state.get("arms") or {}).get(role) or {}).get("features") or {})
    if not features:
        raise ValueError(f"missing pre-action feature arm: {role}")
    return features


def _action_agents(state: Mapping[str, Any], action_id: str, *, field: str) -> list[int]:
    actions = {
        str(row.get("action_id", "")): dict(row)
        for row in list(state.get("unique_actions") or ())
    }
    if action_id not in actions:
        raise ValueError(f"{field} action missing from H1 state")
    raw = list(actions[action_id].get("agents") or ())
    if not raw or any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError(f"{field} action agents invalid")
    agents = sorted(raw)
    if len(agents) != len(set(agents)):
        raise ValueError(f"{field} action agents are not unique")
    return agents


def _relation_features(left: Sequence[int], right: Sequence[int]) -> list[float]:
    left_set, right_set = set(left), set(right)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    return [
        intersection / min(len(left_set), len(right_set)),
        intersection / union,
        len(left_set ^ right_set) / union,
    ]


def _source_row_contract(
    row: Mapping[str, Any], state: Mapping[str, Any], map_to_fold: Mapping[str, str]
) -> tuple[str, str]:
    state_id = str(row.get("state_occurrence_id", ""))
    map_id = str(row.get("map_id", ""))
    split = str(row.get("research_split", ""))
    if not state_id or map_id not in map_to_fold or split not in {"train", "development"}:
        raise ValueError(f"expanded-train source row identity changed: {state_id}")
    if (
        str(state.get("state_occurrence_id", "")) != state_id
        or str(state.get("map_id", "")) != map_id
        or str(state.get("split", "")) != split
    ):
        raise ValueError(f"expanded-train label/state metadata mismatch: {state_id}")
    for field in (
        "task_id",
        "solver_seed",
        "decision_index",
        "depth_band",
        "hierarchical_stratum",
    ):
        if row.get(field) != state.get(field):
            raise ValueError(f"expanded-train label/state {field} mismatch: {state_id}")
    source_fold = row.get("train_fold")
    expected_source_fold = next(
        (fold for fold, maps in SOURCE_TRAIN_MAP_FOLDS.items() if map_id in maps),
        None,
    )
    if (split == "train" and source_fold != expected_source_fold) or (
        split == "development" and source_fold is not None
    ):
        raise ValueError(f"expanded-train source fold changed: {state_id}")
    if (
        row.get("runtime_or_pp_seconds_used_in_label") is not False
        or row.get("sequential_design_only") is not True
        or row.get("training_authorized") is not False
    ):
        raise ValueError(f"expanded-train source label trust changed: {state_id}")
    return state_id, map_id


def _expanded_base(
    row: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    map_to_fold: Mapping[str, str],
    source_artifact_sha256: str,
    h1_manifest_sha256: str,
    schema: str,
) -> dict[str, Any]:
    source = dict(row)
    map_id = str(source["map_id"])
    result = dict(source)
    result.update(
        {
            "schema": schema,
            "source_schema": str(source.get("schema", "")),
            "source_research_split": str(source.get("research_split", "")),
            "source_train_fold": source.get("train_fold"),
            "research_split": "expanded_train",
            "train_fold": map_to_fold[map_id],
            "expanded_train": True,
            "expanded_train_experiment_id": EXPERIMENT_ID,
            "source_label_artifact_sha256": source_artifact_sha256,
            "source_label_row_sha256": _fingerprint(source),
            "source_h1_manifest_sha256": h1_manifest_sha256,
            "source_h1_state_file": str(provenance["state_file"]),
            "source_h1_state_sha256": str(provenance["state_sha256"]),
            "training_authorized": True,
            "development_evaluation_authorized": False,
            "final_claim_authorized": False,
            "runtime_or_ttf_claim_authorized": False,
            "map_disjoint_final_confirmation_required": True,
            "promotion_authorized": False,
        }
    )
    return result


def _expand_stage1_pair_row(
    row: Mapping[str, Any],
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    map_to_fold: Mapping[str, str],
    source_artifact_sha256: str,
    h1_manifest_sha256: str,
) -> dict[str, Any]:
    state_id, _map_id = _source_row_contract(row, state, map_to_fold)
    if row.get("schema") != STAGE1_PAIR_SCHEMA or str(row.get("label", "")) not in STAGE1_LABELS:
        raise ValueError(f"expanded Stage1 pair schema/label changed: {state_id}")
    structural_id = str(row.get("structural_action_id", ""))
    v2_id = str(row.get("v2_action_id", ""))
    if not structural_id or not v2_id or structural_id == v2_id:
        raise ValueError(f"expanded Stage1 exact actions do not differ: {state_id}")
    roles = tuple(sorted(map(str, row.get("structural_role_aliases") or ())))
    if not roles or not set(roles) <= {"component16", "hotspot16"}:
        raise ValueError(f"expanded Stage1 structural aliases changed: {state_id}")
    role_map = {str(k): str(v) for k, v in dict(state.get("role_to_action_id") or {}).items()}
    if role_map.get("v2_anchor") != v2_id or any(role_map.get(role) != structural_id for role in roles):
        raise ValueError(f"expanded Stage1 role/action mapping changed: {state_id}")
    structural_agents = _action_agents(state, structural_id, field="structural")
    v2_agents = _action_agents(state, v2_id, field="v2")
    if structural_agents == v2_agents:
        raise ValueError(f"expanded Stage1 exact structural/V2 sets do not differ: {state_id}")
    anchor = _arm_features(state, "v2_anchor")
    structural_features = [_arm_features(state, role) for role in roles]
    values = [_finite(anchor.get(name), field=name) for name in CONTEXT_FIELDS]
    for name in DELTA_FIELDS:
        structural_value = sum(_finite(features.get(name), field=f"{role}/{name}") for role, features in zip(roles, structural_features)) / len(structural_features)
        values.append(structural_value - _finite(anchor.get(name), field=f"v2/{name}"))
    values.extend(_relation_features(structural_agents, v2_agents))
    if len(values) != 18:
        raise ValueError(f"expanded Stage1 feature vector changed: {state_id}")
    result = _expanded_base(
        row,
        provenance,
        map_to_fold=map_to_fold,
        source_artifact_sha256=source_artifact_sha256,
        h1_manifest_sha256=h1_manifest_sha256,
        schema=EXPANDED_STAGE1_PAIR_SCHEMA,
    )
    result.update(
        {
            "exact_structural_agents": structural_agents,
            "exact_v2_agents": v2_agents,
            "exact_structural_v2_sets_differ": True,
            "model_feature_names": list(STAGE1_FEATURE_NAMES),
            "model_feature_vector": values,
            "model_feature_direction": "structural_minus_v2",
            "repair_outcome_or_runtime_used_as_model_feature": False,
        }
    )
    return result


def _expand_stage1_state_row(
    row: Mapping[str, Any],
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    map_to_fold: Mapping[str, str],
    source_artifact_sha256: str,
    h1_manifest_sha256: str,
) -> dict[str, Any]:
    state_id, _map_id = _source_row_contract(row, state, map_to_fold)
    if row.get("schema") != STAGE1_STATE_SCHEMA or str(row.get("label", "")) not in STAGE1_LABELS:
        raise ValueError(f"expanded Stage1 state schema/label changed: {state_id}")
    return _expanded_base(
        row,
        provenance,
        map_to_fold=map_to_fold,
        source_artifact_sha256=source_artifact_sha256,
        h1_manifest_sha256=h1_manifest_sha256,
        schema=EXPANDED_STAGE1_STATE_SCHEMA,
    )


def _expand_stage2_row(
    row: Mapping[str, Any],
    state: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    map_to_fold: Mapping[str, str],
    source_artifact_sha256: str,
    h1_manifest_sha256: str,
) -> dict[str, Any]:
    state_id, _map_id = _source_row_contract(row, state, map_to_fold)
    if row.get("schema") != STAGE2_STATE_SCHEMA or str(row.get("label", "")) not in STAGE2_LABELS:
        raise ValueError(f"expanded Stage2 schema/label changed: {state_id}")
    component_id = str(row.get("component_action_id", ""))
    hotspot_id = str(row.get("hotspot_action_id", ""))
    role_map = {str(k): str(v) for k, v in dict(state.get("role_to_action_id") or {}).items()}
    if (
        not component_id
        or not hotspot_id
        or component_id == hotspot_id
        or role_map.get("component16") != component_id
        or role_map.get("hotspot16") != hotspot_id
    ):
        raise ValueError(f"expanded Stage2 exact C/H action identity changed: {state_id}")
    component_agents = _action_agents(state, component_id, field="component")
    hotspot_agents = _action_agents(state, hotspot_id, field="hotspot")
    if component_agents == hotspot_agents:
        raise ValueError(f"expanded Stage2 exact C/H sets do not differ: {state_id}")
    component = _arm_features(state, "component16")
    hotspot = _arm_features(state, "hotspot16")
    values: list[float] = []
    for name in CONTEXT_FIELDS:
        component_value = _finite(component.get(name), field=f"component/{name}")
        hotspot_value = _finite(hotspot.get(name), field=f"hotspot/{name}")
        if not math.isclose(component_value, hotspot_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Stage2 shared state context differs: {state_id}/{name}")
        values.append(component_value)
    values.extend(
        _finite(component.get(name), field=f"component/{name}")
        - _finite(hotspot.get(name), field=f"hotspot/{name}")
        for name in DELTA_FIELDS
    )
    values.extend(_relation_features(component_agents, hotspot_agents))
    if len(values) != 18 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"expanded Stage2 feature vector changed: {state_id}")
    result = _expanded_base(
        row,
        provenance,
        map_to_fold=map_to_fold,
        source_artifact_sha256=source_artifact_sha256,
        h1_manifest_sha256=h1_manifest_sha256,
        schema=EXPANDED_STAGE2_STATE_SCHEMA,
    )
    result.update(
        {
            "exact_component_agents": component_agents,
            "exact_hotspot_agents": hotspot_agents,
            "exact_component_hotspot_sets_differ": True,
            "model_feature_names": list(STAGE2_FEATURE_NAMES),
            "model_feature_vector": values,
            "model_feature_direction": "component_minus_hotspot",
            "repair_outcome_or_runtime_used_as_model_feature": False,
        }
    )
    return result


def _load_h1_states(
    collection_root: Path,
    report_spec: Mapping[str, Any],
    manifest_spec: Mapping[str, Any],
    *,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    report_path = _pinned_file(
        collection_root / str(report_spec["file"]), report_spec, field="h1_collection_report"
    )
    report = _read_json(report_path)
    if (
        report.get("schema") != H1_COLLECTION_SCHEMA
        or report.get("status") != "complete"
        or report.get("complete") is not True
        or report.get("sequential_design_only") is not True
        or report.get("final_claim_authorized") is not False
        or report.get("runtime_or_ttf_claim_authorized") is not False
    ):
        raise ValueError("expanded-train H1 collection trust changed")
    registered_manifest = dict(
        dict(report.get("artifacts") or {}).get("label_source_state_manifest") or {}
    )
    if (
        registered_manifest.get("file") != manifest_spec.get("file")
        or registered_manifest.get("sha256") != manifest_spec.get("sha256")
    ):
        raise ValueError("expanded-train H1 manifest pin changed")
    manifest_path = _pinned_file(
        collection_root / str(manifest_spec["file"]), manifest_spec, field="h1_state_manifest"
    )
    manifest = _read_jsonl(manifest_path)
    if len(manifest) != int(manifest_spec.get("row_count", -1)):
        raise ValueError("expanded-train H1 manifest row count changed")

    def load(row: Mapping[str, Any]) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if row.get("schema") != H1_MANIFEST_ROW_SCHEMA:
            raise ValueError("expanded-train H1 manifest row schema changed")
        state_id = str(row.get("state_occurrence_id", ""))
        state_path = _contained_child(
            collection_root, str(row.get("state_file", "")), field="h1_state_file"
        )
        state_sha = _require_sha(row.get("state_sha256"), field="h1_state_sha256")
        if not state_path.is_file() or sha256_file(state_path) != state_sha:
            raise ValueError(f"expanded-train H1 state SHA256 mismatch: {state_id}")
        payload = _read_json(state_path)
        if (
            payload.get("schema") != H1_STATE_SCHEMA
            or payload.get("complete") is not True
            or str(payload.get("state_occurrence_id", "")) != state_id
            or payload.get("runtime_used_in_label") is not False
            or payload.get("sequential_design_only") is not True
            or payload.get("final_claim_authorized") is not False
        ):
            raise ValueError(f"expanded-train H1 state trust changed: {state_id}")
        state = dict(payload.get("state_row") or {})
        if (
            state.get("candidate_repair_actions_executed") is not False
            or state.get("outcome_filtering") is not False
            or state.get("target_outcome_fields_read") not in {False, None}
            or state.get("sequential_design_only") is not True
        ):
            raise ValueError(f"expanded-train H1 state is not pre-action: {state_id}")
        provenance = {
            "state_file": str(row["state_file"]),
            "state_sha256": state_sha,
        }
        return state_id, state, provenance

    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(load, manifest))
    states = {state_id: state for state_id, state, _provenance in loaded}
    provenance = {
        state_id: item for state_id, _state, item in loaded
    }
    if len(states) != len(loaded) or len(states) != int(report.get("observed_valid_state_count", -1)):
        raise ValueError("expanded-train H1 state identity/count changed")
    return states, provenance, {
        "collection_report": {
            "file": str(report_spec["file"]),
            "sha256": sha256_file(report_path),
        },
        "state_manifest": {
            "file": str(manifest_spec["file"]),
            "sha256": sha256_file(manifest_path),
            "row_count": len(manifest),
        },
        "every_state_payload_sha256_verified": True,
        "state_count": len(states),
    }


def _support(
    rows: Sequence[Mapping[str, Any]],
    *,
    labels: Sequence[str],
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    counts = collections.Counter(str(row.get("label", "")) for row in rows)
    unknown = set(counts) - set(labels)
    if unknown:
        raise ValueError(f"expanded-train unknown labels: {sorted(unknown)}")
    by_fold: dict[str, Any] = {}
    by_map: dict[str, Any] = {}
    for fold in FIXED_MAP_FOLDS:
        selected = [row for row in rows if row.get("train_fold") == fold]
        fold_counts = collections.Counter(str(row["label"]) for row in selected)
        by_fold[fold] = {
            "row_count": len(selected),
            "label_counts": {label: int(fold_counts[label]) for label in labels},
        }
    for map_id in ALL_MAPS:
        selected = [row for row in rows if row.get("map_id") == map_id]
        map_counts = collections.Counter(str(row["label"]) for row in selected)
        by_map[map_id] = {
            "row_count": len(selected),
            "label_counts": {label: int(map_counts[label]) for label in labels},
        }
    result: dict[str, Any] = {
        "row_count": len(rows),
        "active_map_count": len({str(row["map_id"]) for row in rows}),
        "label_counts": {label: int(counts[label]) for label in labels},
        "by_fold": by_fold,
        "by_map": by_map,
    }
    if thresholds is None:
        return result
    decisive = tuple(map(str, thresholds.get("decisive_labels") or ()))
    direction_support = {}
    for direction in decisive:
        selected = [row for row in rows if row.get("label") == direction]
        direction_support[direction] = {
            "row_count": len(selected),
            "maps": sorted({str(row["map_id"]) for row in selected}),
            "depth_bands": sorted({str(row["depth_band"]) for row in selected}),
        }
    gates = {
        "minimum_rows_per_direction": all(
            direction_support[label]["row_count"]
            >= int(thresholds["minimum_rows_per_direction"])
            for label in decisive
        ),
        "minimum_maps_per_direction": all(
            len(direction_support[label]["maps"])
            >= int(thresholds["minimum_maps_per_direction"])
            for label in decisive
        ),
        "minimum_depth_bands_per_direction": all(
            len(direction_support[label]["depth_bands"])
            >= int(thresholds["minimum_depth_bands_per_direction"])
            for label in decisive
        ),
        "each_fold_has_both_directions": all(
            all(by_fold[fold]["label_counts"][label] > 0 for label in decisive)
            for fold in FIXED_MAP_FOLDS
        ),
    }
    result.update(
        {
            "decisive_direction_support": direction_support,
            "gates": gates,
            "support_ready": all(gates.values()),
        }
    )
    return result


def build_expanded_train(
    config_path: str | Path,
    *,
    output: str | Path,
    workers: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    _validate_config(config)
    worker_count = _workers(config, workers)
    inputs = dict(config["inputs"])

    # Registration and fold freezing intentionally happen before any label read.
    registration_spec = dict(inputs["registration_report"])
    registration_path = _contained(
        project_root, str(registration_spec["path"]), field="registration_report"
    )
    _pinned_file(registration_path, registration_spec, field="registration_report")
    registration = _read_json(registration_path)
    frozen = freeze_registered_folds(registration, dict(config["fold_derivation"]))
    map_to_fold = dict(frozen["map_to_fold"])
    run_fingerprint = _fingerprint(
        {
            "namespace": "stride-hierarchical-ch-compact-flow-expanded-train-v1",
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(config_path),
            "workers": worker_count,
            "fold_assignment_fingerprint": frozen["fold_assignment_fingerprint"],
            "input_sha256": {
                key: dict(value).get("sha256")
                for key, value in inputs.items()
                if isinstance(value, dict) and "sha256" in value
            },
        }
    )

    labels_root = _contained(
        project_root, str(inputs["source_labels_root"]), field="source_labels_root"
    )
    diagnostic_spec = dict(inputs["label_diagnostics"])
    readiness_spec = dict(inputs["readiness_report"])
    diagnostic_path = _pinned_file(
        labels_root / str(diagnostic_spec["file"]), diagnostic_spec, field="label_diagnostics"
    )
    readiness_path = _pinned_file(
        labels_root / str(readiness_spec["file"]), readiness_spec, field="readiness_report"
    )
    diagnostic = _read_json(diagnostic_path)
    readiness = _read_json(readiness_path)
    if (
        diagnostic.get("schema") != LABEL_DIAGNOSTICS_SCHEMA
        or diagnostic.get("complete") is not True
        or diagnostic.get("status") != "COMPLETE_SEQUENTIAL_LABELS"
        or diagnostic.get("training_authorized") is not False
        or diagnostic.get("final_claim_authorized") is not False
        or diagnostic.get("runtime_or_ttf_claim_authorized") is not False
        or diagnostic.get("map_disjoint_confirmation_required") is not True
    ):
        raise ValueError("expanded-train source label diagnostics changed")
    if (
        readiness.get("schema") != READINESS_SCHEMA
        or readiness.get("complete") is not True
        or readiness.get("scientific_status") != "sequential_labels_only_readiness"
        or readiness.get("development_is_sequential_diagnostic_only") is not True
        or readiness.get("training_authorized") is not False
        or readiness.get("final_claim_authorized") is not False
        or readiness.get("runtime_or_ttf_claim_authorized") is not False
        or readiness.get("map_disjoint_confirmation_required") is not True
    ):
        raise ValueError("expanded-train source readiness changed")

    source_specs = {
        key: dict(inputs[key])
        for key in ("stage1_pair_labels", "stage1_state_labels", "stage2_state_labels")
    }
    diagnostic_artifacts = dict(diagnostic.get("artifacts") or {})
    readiness_inputs = dict(readiness.get("inputs") or {})
    source_paths: dict[str, Path] = {}
    source_rows: dict[str, list[dict[str, Any]]] = {}
    for key, spec in source_specs.items():
        if (
            dict(diagnostic_artifacts.get(key) or {}).get("sha256") != spec["sha256"]
            or dict(readiness_inputs.get(key) or {}).get("sha256") != spec["sha256"]
        ):
            raise ValueError(f"expanded-train source report/{key} pin mismatch")
        path = _pinned_file(labels_root / str(spec["file"]), spec, field=key)
        rows = _read_jsonl(path)
        if len(rows) != int(spec.get("row_count", -1)):
            raise ValueError(f"expanded-train source {key} row count changed")
        source_paths[key], source_rows[key] = path, rows
    expected_label_counts = dict(diagnostic.get("label_counts") or {})
    if expected_label_counts != {
        "stage1_pair_count": len(source_rows["stage1_pair_labels"]),
        "stage1_state_count": len(source_rows["stage1_state_labels"]),
        "stage2_state_count": len(source_rows["stage2_state_labels"]),
    }:
        raise ValueError("expanded-train source label diagnostics row counts changed")

    collection_root = _contained(
        project_root, str(inputs["h1_collection_root"]), field="h1_collection_root"
    )
    states, provenance, h1_integrity = _load_h1_states(
        collection_root,
        dict(inputs["h1_collection_report"]),
        dict(inputs["h1_state_manifest"]),
        workers=worker_count,
    )
    if dict(diagnostic.get("input_integrity") or {}).get("collection_report_sha256") != dict(
        inputs["h1_collection_report"]
    )["sha256"]:
        raise ValueError("expanded-train diagnostics/H1 report pin mismatch")
    if dict(diagnostic.get("input_integrity") or {}).get("manifest_sha256") != dict(
        inputs["h1_state_manifest"]
    )["sha256"]:
        raise ValueError("expanded-train diagnostics/H1 manifest pin mismatch")

    def expand_many(
        rows: Sequence[Mapping[str, Any]],
        function: Any,
        source_key: str,
    ) -> list[dict[str, Any]]:
        source_sha = str(source_specs[source_key]["sha256"])

        def one(row: Mapping[str, Any]) -> dict[str, Any]:
            state_id = str(row.get("state_occurrence_id", ""))
            if state_id not in states:
                raise ValueError(f"expanded-train label state absent from H1: {state_id}")
            return function(
                row,
                states[state_id],
                provenance[state_id],
                map_to_fold=map_to_fold,
                source_artifact_sha256=source_sha,
                h1_manifest_sha256=str(inputs["h1_state_manifest"]["sha256"]),
            )

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            return list(pool.map(one, rows))

    expanded_pairs = expand_many(
        source_rows["stage1_pair_labels"], _expand_stage1_pair_row, "stage1_pair_labels"
    )
    expanded_stage1_states = expand_many(
        source_rows["stage1_state_labels"], _expand_stage1_state_row, "stage1_state_labels"
    )
    expanded_stage2_states = expand_many(
        source_rows["stage2_state_labels"], _expand_stage2_row, "stage2_state_labels"
    )

    pair_ids = [str(row.get("pair_id", "")) for row in expanded_pairs]
    stage1_ids = [str(row.get("state_occurrence_id", "")) for row in expanded_stage1_states]
    stage2_ids = [str(row.get("state_occurrence_id", "")) for row in expanded_stage2_states]
    if len(set(pair_ids)) != len(pair_ids) or len(set(stage1_ids)) != len(stage1_ids) or len(set(stage2_ids)) != len(stage2_ids):
        raise ValueError("expanded-train output label identity duplicated")
    if set(stage1_ids) != set(states) or not set(stage2_ids) <= set(stage1_ids):
        raise ValueError("expanded-train state label coverage changed")
    if {str(row["map_id"]) for row in expanded_stage1_states} != set(ALL_MAPS):
        raise ValueError("expanded-train did not include all 16 registered maps")
    if any(row.get("exact_component_hotspot_sets_differ") is not True for row in expanded_stage2_states):
        raise ValueError("expanded-train Stage2 contains non-distinct exact C/H sets")

    thresholds = dict(config["support_thresholds"])
    support = {
        "stage1_pair": _support(
            expanded_pairs,
            labels=STAGE1_LABELS,
            thresholds=dict(thresholds["stage1_pair"]),
        ),
        "stage1_state": _support(expanded_stage1_states, labels=STAGE1_LABELS),
        "stage2_state": _support(
            expanded_stage2_states,
            labels=STAGE2_LABELS,
            thresholds=dict(thresholds["stage2_state"]),
        ),
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_EXPANDED_TRAIN",
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "workers": worker_count,
        "workers_in_fingerprint": True,
        "folds_frozen_before_label_read": True,
        "fold_assignment_fingerprint": frozen["fold_assignment_fingerprint"],
        "fold_derivation": {
            "policy_id": FOLD_POLICY_ID,
            "metadata_inputs": frozen["metadata_inputs"],
            "label_or_outcome_inputs_used": False,
        },
        "fixed_map_folds": {fold: list(maps) for fold, maps in FIXED_MAP_FOLDS.items()},
        "all_registered_maps_included": True,
        "registered_map_count": 16,
        "source_research_splits_preserved": True,
        "source_split_counts": {"train": 8, "development": 8},
        "all_output_rows_reclassified_as": "expanded_train",
        "input_integrity": {
            "registration_report": {
                "path": str(registration_spec["path"]),
                "sha256": sha256_file(registration_path),
            },
            "source_label_diagnostics": {
                "file": str(diagnostic_spec["file"]),
                "sha256": sha256_file(diagnostic_path),
            },
            "source_readiness_report": {
                "file": str(readiness_spec["file"]),
                "sha256": sha256_file(readiness_path),
            },
            "source_label_artifacts": {
                key: {
                    "file": str(source_specs[key]["file"]),
                    "sha256": sha256_file(source_paths[key]),
                    "row_count": len(source_rows[key]),
                }
                for key in source_specs
            },
            "h1": h1_integrity,
            "old_label_and_readiness_sha256_verified": True,
            "every_h1_state_payload_sha256_verified": True,
            "source_row_counts_preserved_exactly": True,
            "map_or_row_selection_by_label": False,
            "stage2_exact_component_hotspot_sets_differ_count": len(expanded_stage2_states),
            "stage2_non_distinct_component_hotspot_set_count": 0,
        },
        "feature_contract": {
            "stage1_feature_names": list(STAGE1_FEATURE_NAMES),
            "stage2_feature_names": list(STAGE2_FEATURE_NAMES),
            "feature_count": 18,
            "source": "SHA-verified H1 pre-action state arms and exact action sets",
            "repair_outcome_or_runtime_used_as_model_feature": False,
        },
        "support": support,
        "expanded_training_use_authorized": True,
        "training_authorized": True,
        "model_fit_executed": False,
        "model_exported": False,
        "development_evaluation_authorized": False,
        "final_claim_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "promotion_authorized": False,
        "map_disjoint_final_confirmation_required": True,
        "artifacts": {},
    }
    if dry_run:
        report["status"] = "DRY_RUN_VALIDATED"
        return report

    output_root = Path(output).resolve()
    if output_root == labels_root or output_root == collection_root:
        raise ValueError("expanded-train output must use a new artifact root")
    targets = {
        "expanded_stage1_pair_labels": output_root / "expanded_stage1_pair_labels.jsonl",
        "expanded_stage1_state_labels": output_root / "expanded_stage1_state_labels.jsonl",
        "expanded_stage2_state_labels": output_root / "expanded_stage2_state_labels.jsonl",
        "expanded_train_report": output_root / "expanded_train_report.json",
    }
    if any(path.exists() for path in targets.values()):
        raise ValueError("expanded-train output already exists")
    _write_jsonl(targets["expanded_stage1_pair_labels"], expanded_pairs)
    _write_jsonl(targets["expanded_stage1_state_labels"], expanded_stage1_states)
    _write_jsonl(targets["expanded_stage2_state_labels"], expanded_stage2_states)
    row_counts = {
        "expanded_stage1_pair_labels": len(expanded_pairs),
        "expanded_stage1_state_labels": len(expanded_stage1_states),
        "expanded_stage2_state_labels": len(expanded_stage2_states),
    }
    schemas = {
        "expanded_stage1_pair_labels": EXPANDED_STAGE1_PAIR_SCHEMA,
        "expanded_stage1_state_labels": EXPANDED_STAGE1_STATE_SCHEMA,
        "expanded_stage2_state_labels": EXPANDED_STAGE2_STATE_SCHEMA,
    }
    report["artifacts"] = {
        key: {
            "file": targets[key].name,
            "sha256": sha256_file(targets[key]),
            "row_count": row_counts[key],
            "schema": schemas[key],
        }
        for key in row_counts
    }
    _write_json(targets["expanded_train_report"], report)
    return report


__all__ = [
    "ALL_MAPS",
    "DEFAULT_CONFIG",
    "DEFAULT_OUTPUT",
    "EXPANDED_STAGE1_PAIR_SCHEMA",
    "EXPANDED_STAGE1_STATE_SCHEMA",
    "EXPANDED_STAGE2_STATE_SCHEMA",
    "EXPERIMENT_ID",
    "FIXED_MAP_FOLDS",
    "REPORT_SCHEMA",
    "STAGE1_FEATURE_NAMES",
    "STAGE2_FEATURE_NAMES",
    "_expand_stage2_row",
    "build_expanded_train",
    "freeze_registered_folds",
]
