from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

import experiments.stride_hierarchical_ch_compact_flow_stage1_consensus_selective_guard_external_screen_v1 as subject


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_stage1_consensus_selective_guard_external_screen_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _features(offset: float = 0.0) -> dict[str, float]:
    values = {}
    for index, name in enumerate(subject.expanded.CONTEXT_FIELDS):
        values[name] = float(index + 1)
    for index, name in enumerate(subject.expanded.DELTA_FIELDS):
        values[name] = float(index + 1) + offset
    return values


def _external_row() -> dict:
    structural_agents = [1, 2, 3]
    v2_agents = [2, 3, 4]
    return {
        "schema": subject.overlay.SELECTION_SCHEMA,
        "research_split": "fresh_matched_development",
        "target_outcome_fields_read": False,
        "hierarchical_stratum": "consensus_structural",
        "unique_action_count": 2,
        "state_occurrence_id": "external-state",
        "map_id": "random-32-32-20",
        "map_family": "random",
        "role_to_action_id": {
            "component16": "structural",
            "hotspot16": "structural",
            "v2_anchor": "v2",
        },
        "unique_actions": [
            {"action_id": "structural", "agents": structural_agents},
            {"action_id": "v2", "agents": v2_agents},
        ],
        "arms": {
            "component16": {
                "agents": structural_agents,
                "features": _features(2.0),
            },
            "hotspot16": {
                "agents": list(reversed(structural_agents)),
                "features": _features(2.0),
            },
            "v2_anchor": {"agents": v2_agents, "features": _features(0.0)},
        },
    }


class _FixedEstimator:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = probabilities

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        assert len(matrix) == len(self.probabilities)
        return np.asarray(
            [[1.0 - probability, probability] for probability in self.probabilities],
            dtype=np.float64,
        )


def test_registered_identity_freezes_maps_model_thresholds_and_claim_boundary() -> None:
    config = _config()
    subject.validate_config(config)
    assert config["heads"]["opportunity"]["threshold"] == 0.40
    assert config["heads"]["direction"]["threshold"] == 0.75
    assert config["external_map_folds"]["fold1"][-1] == "random-32-32-20"
    maps = [
        map_id
        for fold_maps in config["external_map_folds"].values()
        for map_id in fold_maps
    ]
    assert len(maps) == len(set(maps)) == 16
    assert config["claim_boundary"]["prospective_claim_allowed"] is False
    assert config["claim_boundary"]["model_export_allowed"] is False
    assert config["claim_boundary"]["stage2_loaded"] is False

    duplicated = copy.deepcopy(config)
    duplicated["external_map_folds"]["fold1"][-1] = "random-64-64-20"
    with pytest.raises(ValueError, match="folds changed"):
        subject.validate_config(duplicated)


def test_external_exact_consensus_builds_current_signed18_without_outcomes() -> None:
    config = _config()
    map_to_fold = {
        map_id: fold
        for fold, maps in config["external_map_folds"].items()
        for map_id in maps
    }
    example = subject.build_external_example(_external_row(), map_to_fold)
    assert example["train_fold"] == "fold1"
    assert example["component_action_id"] == example["hotspot_action_id"]
    assert example["structural_action_id"] != example["v2_action_id"]
    assert len(example["features"]) == 18
    assert all(np.isfinite(example["features"]))
    assert example["model_feature_names"] == list(subject.FEATURE_NAMES)
    assert example["repair_outcome_or_runtime_used_as_model_feature"] is False


def test_external_builder_fails_closed_on_nonconsensus_duplicates_and_outcomes() -> None:
    map_to_fold = {"random-32-32-20": "fold1"}
    nonconsensus = _external_row()
    nonconsensus["role_to_action_id"]["hotspot16"] = "other"
    with pytest.raises(ValueError, match="C=H!=V"):
        subject.build_external_example(nonconsensus, map_to_fold)

    duplicate = _external_row()
    duplicate["unique_actions"][0]["agents"] = [1, 1, 2]
    with pytest.raises(ValueError, match="duplicate agents"):
        subject.build_external_example(duplicate, map_to_fold)

    leaked = _external_row()
    leaked["label"] = "structural_win"
    with pytest.raises(ValueError, match="outcome entered"):
        subject.build_external_example(leaked, map_to_fold)

    alias_mismatch = _external_row()
    alias_mismatch["arms"]["hotspot16"]["features"][
        subject.expanded.DELTA_FIELDS[0]
    ] += 1e-6
    with pytest.raises(ValueError, match="alias feature differs"):
        subject.build_external_example(alias_mismatch, map_to_fold)


def test_fixed_policy_requires_both_thresholds_and_preserves_exact_v2_fallback() -> None:
    base = subject.build_external_example(
        _external_row(), {"random-32-32-20": "fold1"}
    )
    examples = [
        {**base, "state_occurrence_id": f"s{index}", "pair_id": f"p{index}"}
        for index in range(3)
    ]
    rows = subject._unlabeled_predictions(
        examples,
        _FixedEstimator([0.40, 0.39, 0.90]),
        _FixedEstimator([0.75, 0.99, 0.74]),
        "policy",
    )
    assert [row["structural_override"] for row in rows] == [True, False, False]
    for row in rows[1:]:
        assert row["final_action_id"] == row["v2_action_id"]
        assert row["final_action_agents"] == row["exact_v2_agents"]
        assert row["final_action_role"] == "exact_v2_fallback"
    assert all("label" not in row and row["label_loaded"] is False for row in rows)


def test_label_merge_and_gates_fail_closed_on_incomplete_support() -> None:
    prediction = {
        "state_occurrence_id": "s",
        "map_id": "m",
        "structural_override": False,
    }
    with pytest.raises(ValueError, match="exactly cover"):
        subject._merge_labels([prediction], {})

    metrics = {
        "total_override_count": 7,
        "selected_action_precision": 1.0,
        "v2_wrong_override_count": 0,
        "ambiguous_override_rate": 0.0,
        "structural_win_override_recall": 1.0,
        "full_pooled_decisive_balanced_accuracy": 1.0,
        "exact_v2_fallback_semantics_rate": 1.0,
    }
    rows = [
        {"map_id": "m1", "structural_override": True},
        {"map_id": "m2", "structural_override": True},
    ]
    checks = subject._gate_checks(metrics, rows)
    assert checks["minimum_total_override_count"] is False
    assert checks["minimum_override_map_count"] is True


def test_policy_identity_excludes_external_evaluation_binding() -> None:
    core = {
        "policy_semantics": "fixed_guard",
        "training_cohort_fingerprint": "train",
        "feature_view": "signed18",
    }
    first = {
        "experiment_id": "retrospective-screen",
        "policy": core,
        "evaluation_binding": {"external_preaction_feature_fingerprint": "a"},
    }
    second = {
        "experiment_id": "later-sealed-final",
        "policy": core,
        "evaluation_binding": {"external_preaction_feature_fingerprint": "b"},
    }
    assert subject._policy_identity_from_manifest(
        first
    ) == subject._policy_identity_from_manifest(second)

    contaminated = {"policy": {**core, "external_selected_states_sha256": "bad"}}
    with pytest.raises(ValueError, match="external evaluation binding"):
        subject._policy_identity_from_manifest(contaminated)

    contaminated_identity = {"policy": {**core, "experiment_id": "bad"}}
    with pytest.raises(ValueError, match="external evaluation binding"):
        subject._policy_identity_from_manifest(contaminated_identity)


def test_raw_h1_state_must_match_frozen_agents_and_signed18_features() -> None:
    source_row = _external_row()
    prediction = subject.build_external_example(
        source_row, {"random-32-32-20": "fold1"}
    )
    state = {
        "state_row": source_row,
        "actions": {
            "structural": tuple(prediction["exact_structural_agents"]),
            "v2": tuple(prediction["exact_v2_agents"]),
        },
    }
    subject._validate_raw_state_against_prediction(prediction, state)

    wrong_agents = copy.deepcopy(state)
    wrong_agents["actions"]["structural"] = (1, 2, 99)
    with pytest.raises(ValueError, match="exact agents changed"):
        subject._validate_raw_state_against_prediction(prediction, wrong_agents)

    wrong_features = copy.deepcopy(state)
    wrong_features["state_row"]["arms"]["component16"]["features"][
        subject.expanded.DELTA_FIELDS[0]
    ] += 1e-5
    wrong_features["state_row"]["arms"]["hotspot16"]["features"][
        subject.expanded.DELTA_FIELDS[0]
    ] += 1e-5
    with pytest.raises(ValueError, match="pre-action features changed"):
        subject._validate_raw_state_against_prediction(prediction, wrong_features)
