from __future__ import annotations

import copy

import pytest

from experiments.stride_hierarchical_ch_compact_flow_expanded_train_v1 import (
    EXPANDED_STAGE2_STATE_SCHEMA,
    FIXED_MAP_FOLDS,
    STAGE2_FEATURE_NAMES,
    _expand_stage2_row,
    freeze_registered_folds,
)


SOURCE_SPLITS = {
    "den009d": "train",
    "den020d": "development",
    "den101d": "train",
    "den201d": "train",
    "den202d": "development",
    "den203d": "train",
    "den207d": "development",
    "den308d": "train",
    "den404d": "train",
    "den408d": "development",
    "den998d": "development",
    "hrt002d": "train",
    "lak101d": "train",
    "lak108d": "development",
    "lak110d": "development",
    "ost102d": "development",
}
STRATA = {
    "den009d": ("medium", 1003),
    "den020d": ("large", 3102),
    "den101d": ("medium", 1360),
    "den201d": ("small", 538),
    "den202d": ("small", 593),
    "den203d": ("large", 2629),
    "den207d": ("medium", 874),
    "den308d": ("large", 3155),
    "den404d": ("ultra", 358),
    "den408d": ("small", 548),
    "den998d": ("medium", 1853),
    "hrt002d": ("small", 754),
    "lak101d": ("ultra", 318),
    "lak108d": ("ultra", 286),
    "lak110d": ("ultra", 168),
    "ost102d": ("ultra", 249),
}


def _registration() -> dict:
    return {
        "schema": "lns2.stride.hierarchical_ch_compact_flow_source_registration_report.v1",
        "experiment_id": "stride_hierarchical_ch_compact_flow_source_v1",
        "passed": True,
        "map_registration": {
            map_id: {
                "split": SOURCE_SPLITS[map_id],
                "capacity_stratum": STRATA[map_id][0],
                "capacity_free_cells": STRATA[map_id][1],
            }
            for map_id in SOURCE_SPLITS
        },
    }


def _fold_config() -> dict:
    return {
        "policy_id": "registered_source_split_capacity_balanced_fixed_v1",
        "inputs_allowed": [
            "map_id",
            "source_research_split",
            "capacity_stratum",
            "capacity_free_cells",
        ],
        "label_or_outcome_inputs_allowed": False,
        "folds_frozen_before_label_read": True,
        "expected_capacity_profiles": {
            "fold0": ["ultra", "small", "medium", "large"],
            "fold1": ["ultra", "small", "medium", "large"],
            "fold2": ["ultra", "small", "medium", "large"],
            "fold3": ["ultra", "ultra", "small", "medium"],
        },
        "expected_source_split_counts_per_fold": {"train": 2, "development": 2},
        "fixed_map_folds": {key: list(value) for key, value in FIXED_MAP_FOLDS.items()},
    }


def test_folds_are_frozen_from_registered_metadata_before_labels() -> None:
    frozen = freeze_registered_folds(_registration(), _fold_config())
    assert frozen["fixed_map_folds"] == {
        key: list(value) for key, value in FIXED_MAP_FOLDS.items()
    }
    assert frozen["folds_frozen_before_label_read"] is True
    assert frozen["label_or_outcome_inputs_used"] is False
    assert len(frozen["fold_assignment_fingerprint"]) == 64
    assert set(frozen["map_to_fold"]) == set(SOURCE_SPLITS)

    tampered = _registration()
    tampered["map_registration"]["den404d"]["capacity_stratum"] = "large"
    with pytest.raises(ValueError, match="capacity stratum"):
        freeze_registered_folds(tampered, _fold_config())


def _features(offset: float) -> dict[str, float]:
    values = {
        "state.agent_count": 100.0,
        "state.colliding_pairs": 20.0,
        "state.conflict_edge_density": 0.2,
        "state.conflict_event_count": 30.0,
        "state.conflicting_agent_ratio": 0.4,
        "state.degree_max": 5.0,
        "state.largest_component_ratio": 0.5,
        "state.path_wait_ratio_mean": 0.1,
    }
    for index, name in enumerate(
        (
            "realized.incident_conflict_coverage",
            "realized.internal_conflict_coverage",
            "realized.boundary_conflict_edges",
            "realized.conflict_degree_mean",
            "realized.delay_mean",
            "realized.path_overlap_mean",
            "realized.path_wait_ratio_mean",
        )
    ):
        values[name] = offset + index
    return values


def _stage2_fixture() -> tuple[dict, dict, dict]:
    label = {
        "schema": "lns2.stride.hierarchical_ch_stage2_state_label.v1",
        "state_occurrence_id": "state-1",
        "map_id": "den404d",
        "research_split": "train",
        "train_fold": "fold0",
        "component_action_id": "component-action",
        "hotspot_action_id": "hotspot-action",
        "label": "component_win",
        "task_id": "task-1",
        "solver_seed": 41,
        "decision_index": 2,
        "depth_band": "d1_3",
        "hierarchical_stratum": "three_unique",
        "runtime_or_pp_seconds_used_in_label": False,
        "sequential_design_only": True,
        "training_authorized": False,
    }
    state = {
        "state_occurrence_id": "state-1",
        "map_id": "den404d",
        "split": "train",
        "task_id": "task-1",
        "solver_seed": 41,
        "decision_index": 2,
        "depth_band": "d1_3",
        "hierarchical_stratum": "three_unique",
        "role_to_action_id": {
            "component16": "component-action",
            "hotspot16": "hotspot-action",
        },
        "unique_actions": [
            {"action_id": "component-action", "agents": list(range(16))},
            {"action_id": "hotspot-action", "agents": list(range(8, 24))},
        ],
        "arms": {
            "component16": {"features": _features(10.0)},
            "hotspot16": {"features": _features(4.0)},
        },
    }
    provenance = {"state_file": "h1_states/state-1.json", "state_sha256": "a" * 64}
    return label, state, provenance


def test_stage2_expansion_embeds_exact_sets_and_fixed_pre_action_features() -> None:
    label, state, provenance = _stage2_fixture()
    expanded = _expand_stage2_row(
        label,
        state,
        provenance,
        map_to_fold={"den404d": "fold0"},
        source_artifact_sha256="b" * 64,
        h1_manifest_sha256="c" * 64,
    )
    assert expanded["schema"] == EXPANDED_STAGE2_STATE_SCHEMA
    assert expanded["source_research_split"] == "train"
    assert expanded["research_split"] == "expanded_train"
    assert expanded["train_fold"] == "fold0"
    assert expanded["training_authorized"] is True
    assert expanded["exact_component_hotspot_sets_differ"] is True
    assert expanded["exact_component_agents"] == list(range(16))
    assert expanded["exact_hotspot_agents"] == list(range(8, 24))
    assert expanded["model_feature_names"] == list(STAGE2_FEATURE_NAMES)
    assert len(expanded["model_feature_vector"]) == 18
    assert expanded["model_feature_vector"][8:15] == [6.0] * 7

    equal = copy.deepcopy(state)
    equal["unique_actions"][1]["agents"] = list(range(16))
    with pytest.raises(ValueError, match="exact C/H sets do not differ"):
        _expand_stage2_row(
            label,
            equal,
            provenance,
            map_to_fold={"den404d": "fold0"},
            source_artifact_sha256="b" * 64,
            h1_manifest_sha256="c" * 64,
        )
