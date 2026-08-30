from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_two_head_v1 as subject


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_expanded_stage1_two_head_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _example(
    state: str,
    label: str,
    fold: str,
    suffix: str = "0",
) -> dict:
    map_id = subject.FOLDS[fold][0]
    return {
        "pair_id": f"{state}:pair:{suffix}",
        "state_occurrence_id": state,
        "map_id": map_id,
        "research_split": "expanded_train",
        "train_fold": fold,
        "label": label,
        "gate_target": int(label in subject.DECISIVE_LABELS),
        "direction_target": (
            1 if label == "structural_win" else 0 if label == "v2_win" else None
        ),
        "depth_band": "d1_3",
        "role_stratum": "component16",
        "hierarchical_stratum": "two_unique",
        "structural_action_id": f"{state}:structural:{suffix}",
        "v2_action_id": f"{state}:v2",
        "exact_structural_agents": [1, 2],
        "exact_v2_agents": [2, 3],
        "features": [0.0] * 18,
    }


def _probability_rows() -> list[dict]:
    rows = []
    for fold in ("fold0", "fold1", "fold2"):
        for label, gate, direction in (
            ("structural_win", 0.90, 0.90),
            ("v2_win", 0.90, 0.10),
            ("ambiguous", 0.10, 0.90),
        ):
            row = _example(f"{fold}:{label}", label, fold)
            rows.append(
                {
                    **row,
                    "gate_decisive_probability": gate,
                    "direction_structural_probability": direction,
                }
            )
    return rows


def test_registered_config_is_exact() -> None:
    config = _config()
    subject.validate_config(config)
    assert len(config["candidate_combinations"]) == 4
    assert config["threshold_calibration"]["gate_threshold_grid"][-2:] == [
        0.95,
        1.01,
    ]
    assert config["claim_boundary"]["full_fit_model_export_allowed"] is False


def test_build_pair_example_uses_registered_vector_and_preserves_exact_actions() -> None:
    row = {
        "schema": subject.EXPANDED_LABEL_SCHEMA,
        "research_split": "expanded_train",
        "expanded_train": True,
        "training_authorized": True,
        "development_evaluation_authorized": False,
        "final_claim_authorized": False,
        "promotion_authorized": False,
        "runtime_or_ttf_claim_authorized": False,
        "map_disjoint_final_confirmation_required": True,
        "runtime_or_pp_seconds_used_in_label": False,
        "sequential_design_only": True,
        "pair_id": "pair",
        "state_occurrence_id": "state",
        "label": "ambiguous",
        "model_feature_names": list(subject.FEATURE_NAMES),
        "model_feature_vector": list(range(18)),
        "model_feature_direction": "structural_minus_v2",
        "repair_outcome_or_runtime_used_as_model_feature": False,
        "exact_structural_v2_sets_differ": True,
        "map_id": "den404d",
        "train_fold": "fold0",
        "structural_role_aliases": ["component16"],
        "structural_action_id": "structural",
        "v2_action_id": "v2",
        "exact_structural_agents": [1, 2],
        "exact_v2_agents": [2, 3],
        "depth_band": "d1_3",
        "hierarchical_stratum": "two_unique",
        "runtime": 999.0,
        "outcome": "must_not_be_a_feature",
    }
    example = subject.build_pair_example(row)
    assert example["features"] == [float(value) for value in range(18)]
    assert example["structural_action_id"] == "structural"
    assert example["v2_action_id"] == "v2"
    assert "runtime" not in example and "outcome" not in example


def test_head_weights_keep_gate_natural_and_balance_direction() -> None:
    gate_rows = [
        {**_example("a", "ambiguous", "fold0", "0"), "target": 0},
        {**_example("a", "ambiguous", "fold0", "1"), "target": 0},
        {**_example("b", "structural_win", "fold1"), "target": 1},
        {**_example("c", "v2_win", "fold2"), "target": 1},
    ]
    gate_weights, gate_audit = subject._head_sample_weights(gate_rows, "gate")
    assert gate_weights == pytest.approx([0.5, 0.5, 1.0, 1.0])
    assert gate_audit["class_balance_applied"] is False
    assert gate_audit["effective_class_weight_totals"] == pytest.approx(
        {"0": 1.0, "1": 2.0}
    )

    direction_rows = [
        {**_example("s1", "structural_win", "fold0"), "target": 1},
        {**_example("s2", "structural_win", "fold1"), "target": 1},
        {**_example("v1", "v2_win", "fold2"), "target": 0},
    ]
    _, direction_audit = subject._head_sample_weights(
        direction_rows, "direction"
    )
    assert direction_audit["class_balance_applied"] is True
    assert direction_audit["effective_class_weight_totals"]["0"] == pytest.approx(
        direction_audit["effective_class_weight_totals"]["1"]
    )


def test_final_action_is_exact_v2_unless_both_heads_allow_structural() -> None:
    first = _example("state", "structural_win", "fold0", "a")
    second = _example("state", "ambiguous", "fold0", "b")
    combination = subject.EXPECTED_COMBINATIONS[0]
    override = subject._materialize_pair_action(
        first,
        gate_probability=0.9,
        direction_probability=0.8,
        gate_threshold=0.5,
        direction_threshold=0.6,
        combination=combination,
        inner_selection_status="test",
    )
    fallback = subject._materialize_pair_action(
        second,
        gate_probability=0.9,
        direction_probability=0.4,
        gate_threshold=0.5,
        direction_threshold=0.6,
        combination=combination,
        inner_selection_status="test",
    )
    assert override["final_action_id"] == first["structural_action_id"]
    assert fallback["final_action_id"] == second["v2_action_id"]
    assert fallback["final_action_role"] == "exact_v2_fallback"
    state = subject.aggregate_state_actions([fallback, override])[0]
    assert state["selected_pair_id"] == override["pair_id"]
    assert state["final_action_id"] == first["structural_action_id"]


def test_joint_grid_requires_all_active_fold_state_gates() -> None:
    config = _config()
    combination = config["candidate_combinations"][0]
    selected = subject.select_joint_thresholds(
        _probability_rows(),
        combination,
        config["threshold_calibration"],
        config["hard_gates"],
        ("fold0", "fold1", "fold2"),
    )
    assert selected["joint_grid_size"] == len(subject.GATE_THRESHOLDS) * len(
        subject.DIRECTION_THRESHOLDS
    )
    assert (
        selected["status"]
        == "joint_threshold_passes_feasibility_and_full_inner_gates"
    )
    assert all(selected["selected_inner_full_gate_checks"].values())

    fail_closed_rows = [
        {**row, "gate_decisive_probability": 0.0}
        for row in _probability_rows()
    ]
    failed = subject.select_joint_thresholds(
        fail_closed_rows,
        combination,
        config["threshold_calibration"],
        config["hard_gates"],
        ("fold0", "fold1", "fold2"),
    )
    assert failed["status"].startswith("no_joint_threshold")
    assert failed["selected_gate_threshold"] == 1.01
    assert failed["selected_direction_structural_threshold"] == 1.01
