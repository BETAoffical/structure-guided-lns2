from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_only_v1 as subject


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_only_v1.json"
)


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _state(state_id: str = "state") -> dict:
    structural_agents = [1, 2]
    v2_agents = [2, 3]
    return {
        "state_occurrence_id": state_id,
        "canonical_partition": "consensus_structural",
        "hierarchical_stratum": "consensus_structural",
        "unique_action_count": 2,
        "stage2_eligible": False,
        "role_to_action_id": {
            "component16": "structural",
            "hotspot16": "structural",
            "v2_anchor": "v2",
        },
        "stage1_structural_action_ids": ["structural"],
        "arms": {
            "component16": {"agents": structural_agents},
            "hotspot16": {"agents": structural_agents},
            "v2_anchor": {"agents": v2_agents},
        },
        "unique_actions": [
            {"action_id": "structural", "agents": structural_agents},
            {"action_id": "v2", "agents": v2_agents},
        ],
    }


def _label_row(label: str = "structural_win", state_id: str = "state") -> dict:
    return {
        "schema": subject.predecessor.EXPANDED_LABEL_SCHEMA,
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
        "pair_id": f"{state_id}:pair",
        "state_occurrence_id": state_id,
        "label": label,
        "model_feature_names": list(subject.FEATURE_NAMES),
        "model_feature_vector": [float(index - 10) for index in range(18)],
        "model_feature_direction": "structural_minus_v2",
        "repair_outcome_or_runtime_used_as_model_feature": False,
        "exact_structural_v2_sets_differ": True,
        "map_id": "den404d",
        "train_fold": "fold0",
        "structural_role_aliases": ["component16", "hotspot16"],
        "hierarchical_stratum": "consensus_structural",
        "structural_action_id": "structural",
        "v2_action_id": "v2",
        "exact_structural_agents": [1, 2],
        "exact_v2_agents": [2, 3],
        "depth_band": "d1_3",
        "source_h1_state_file": f"h1_states/{state_id}.json",
        "source_h1_state_sha256": "a" * 64,
        "source_h1_manifest_sha256": "b" * 64,
    }


def _example(
    state_id: str,
    label: str,
    fold: str,
    map_id: str,
) -> dict:
    return {
        "pair_id": f"{state_id}:pair",
        "state_occurrence_id": state_id,
        "map_id": map_id,
        "train_fold": fold,
        "research_split": "expanded_train",
        "label": label,
        "structural_action_id": f"{state_id}:structural",
        "v2_action_id": f"{state_id}:v2",
        "exact_structural_agents": [1, 2],
        "exact_v2_agents": [2, 3],
        "component_action_id": f"{state_id}:structural",
        "hotspot_action_id": f"{state_id}:structural",
        "opportunity_target": (
            1 if label == "structural_win" else 0 if label == "ambiguous" else None
        ),
        "direction_target": (
            1 if label == "structural_win" else 0 if label == "v2_win" else None
        ),
        "opportunity_features_signed18": [0.0] * 18,
        "opportunity_features_abs_delta18": [0.0] * 18,
        "direction_features_signed18": [0.0] * 18,
    }


def test_config_freezes_small_consensus_experiment_without_model_output() -> None:
    config = _config()
    subject.validate_config(config)
    assert len(config["candidate_arms"]) == 2
    assert config["expected_support"]["label_counts"] == {
        "ambiguous": 69,
        "structural_win": 14,
        "v2_win": 10,
    }
    assert "model" not in config["outputs"]
    assert config["claim_boundary"]["stage2_rows_loaded"] is False


def test_exact_consensus_builder_and_abs_delta_view() -> None:
    row = _label_row()
    manifest = {
        "state_occurrence_id": "state",
        "state_file": "h1_states/state.json",
        "state_sha256": "a" * 64,
    }
    reordered_consensus = _state()
    reordered_consensus["arms"]["hotspot16"]["agents"] = [2, 1]
    example = subject.build_consensus_example(row, reordered_consensus, manifest)
    assert example["component_action_id"] == example["hotspot_action_id"]
    assert example["component_action_id"] != example["v2_action_id"]
    signed = example["opportunity_features_signed18"]
    absolute = example["opportunity_features_abs_delta18"]
    for index in range(18):
        assert absolute[index] == (
            abs(signed[index]) if index in subject.DELTA_INDICES else signed[index]
        )

    broken = _state()
    broken["role_to_action_id"]["hotspot16"] = "other"
    with pytest.raises(ValueError, match="C=H"):
        subject._consensus_state_contract(broken)


def test_exact_agent_sets_reject_reordering_only_v2_and_duplicates() -> None:
    manifest = {
        "state_occurrence_id": "state",
        "state_file": "h1_states/state.json",
        "state_sha256": "a" * 64,
    }
    same_set_v2 = _label_row()
    same_set_v2["exact_v2_agents"] = [2, 1]
    with pytest.raises(ValueError, match="sets are equal"):
        subject.build_consensus_example(same_set_v2, _state(), manifest)

    duplicate = _state()
    duplicate["arms"]["component16"]["agents"] = [1, 1]
    with pytest.raises(ValueError, match="duplicate agents"):
        subject._consensus_state_contract(duplicate)

    duplicate_action_id = _state()
    duplicate_action_id["unique_actions"].append(
        {"action_id": "v2", "agents": [2, 3]}
    )
    with pytest.raises(ValueError, match="two unique non-empty IDs"):
        subject._consensus_state_contract(duplicate_action_id)


def test_head_populations_exclude_the_third_label_and_balance_classes() -> None:
    rows = [
        _example("s1", "structural_win", "fold0", "den404d"),
        _example("s2", "structural_win", "fold1", "den202d"),
        _example("a1", "ambiguous", "fold1", "den998d"),
        _example("a2", "ambiguous", "fold2", "den009d"),
        _example("a3", "ambiguous", "fold3", "den101d"),
        _example("v1", "v2_win", "fold3", "hrt002d"),
    ]
    opportunity = subject._head_rows(rows, "opportunity", "signed18")
    direction = subject._head_rows(rows, "direction", "signed18")
    assert {row["label"] for row in opportunity} == {
        "structural_win",
        "ambiguous",
    }
    assert {row["label"] for row in direction} == {
        "structural_win",
        "v2_win",
    }
    _, opportunity_audit = subject._balanced_weights(
        opportunity, "opportunity"
    )
    _, direction_audit = subject._balanced_weights(direction, "direction")
    assert opportunity_audit["effective_class_weight_totals"]["0"] == pytest.approx(
        opportunity_audit["effective_class_weight_totals"]["1"]
    )
    assert direction_audit["effective_class_weight_totals"]["0"] == pytest.approx(
        direction_audit["effective_class_weight_totals"]["1"]
    )


def test_inactive_exact_v2_still_enters_full_pooled_recall_and_fold_null() -> None:
    arm = subject.CANDIDATE_ARMS[0]
    active_s = subject.materialize_action(
        _example("s-active", "structural_win", "fold0", "den404d"),
        opportunity_probability=0.9,
        direction_probability=0.9,
        opportunity_threshold=0.5,
        direction_threshold=0.5,
        arm=arm,
        evaluation_active=True,
        selection_status="test",
    )
    inactive_s = subject.materialize_action(
        _example("s-inactive", "structural_win", "fold0", "den207d"),
        opportunity_probability=None,
        direction_probability=None,
        opportunity_threshold=1.01,
        direction_threshold=1.01,
        arm=None,
        evaluation_active=False,
        selection_status="fail_closed",
    )
    metrics = subject.selected_action_metrics([active_s, inactive_s])
    assert metrics["structural_win_override_recall"] == 0.5
    assert metrics["v2_win_protection_recall"] is None
    assert metrics["full_pooled_decisive_balanced_accuracy"] is None
    assert inactive_s["final_action_id"] == inactive_s["v2_action_id"]
    assert metrics["exact_v2_fallback_semantics_rate"] == 1.0

    wrong_role = {**inactive_s, "final_action_role": "wrong_role"}
    wrong_agents = {**inactive_s, "final_action_agents": [999]}
    wrong_id = {**inactive_s, "final_action_id": "wrong_id"}
    assert subject.selected_action_metrics([wrong_role])[
        "exact_v2_fallback_semantics_rate"
    ] == 0.0
    assert subject.selected_action_metrics([wrong_agents])[
        "exact_v2_fallback_semantics_rate"
    ] == 0.0
    assert subject.selected_action_metrics([wrong_id])[
        "exact_v2_fallback_semantics_rate"
    ] == 0.0


def test_joint_selection_uses_safe_high_threshold_tie_break() -> None:
    config = _config()
    examples = []
    structural_maps = ["den404d", "den207d", "den202d"]
    for index in range(6):
        fold = "fold0" if index < 3 else "fold1"
        examples.append(
            _example(
                f"s{index}",
                "structural_win",
                fold,
                structural_maps[index % len(structural_maps)],
            )
        )
    examples.extend(
        [
            _example("v0", "v2_win", "fold1", "den998d"),
            _example("v1", "v2_win", "fold2", "den009d"),
        ]
    )
    for index in range(10):
        examples.append(
            _example(
                f"a{index}",
                "ambiguous",
                "fold2",
                "den020d",
            )
        )
    opportunity = {
        row["pair_id"]: 0.90 if row["label"] == "structural_win" else 0.10
        for row in examples
    }
    direction = {
        row["pair_id"]: 0.90 if row["label"] == "structural_win" else 0.10
        for row in examples
    }
    selected = subject.select_joint_thresholds(
        examples,
        opportunity,
        direction,
        subject.CANDIDATE_ARMS[0],
        config["inner_gates"],
    )
    assert selected["status"] == "joint_threshold_passes_all_inner_gates"
    assert selected["joint_grid_size"] == len(subject.GATE_THRESHOLDS) * len(
        subject.DIRECTION_THRESHOLDS
    )
    assert selected["selected_opportunity_threshold"] == 0.90
    assert selected["selected_direction_threshold"] == 0.90
    assert all(selected["selected_inner_gate_checks"].values())


def test_inner_head_oof_keeps_maps_and_states_grouped() -> None:
    examples = []
    fold_maps = {
        "fold0": "den404d",
        "fold1": "den202d",
        "fold2": "den009d",
    }
    for fold, map_id in fold_maps.items():
        for label in ("structural_win", "v2_win", "ambiguous"):
            examples.append(_example(f"{fold}:{label}", label, fold, map_id))

    seen_fit_labels = []

    def fake_fit(rows, head):
        seen_fit_labels.append((head, {row["label"] for row in rows}))
        return head

    def fake_predict(estimator, rows, feature_key):
        return [0.6] * len(rows)

    direction = subject.cross_validate_head(
        examples,
        "direction",
        "signed18",
        ("fold0", "fold1", "fold2"),
        fit=fake_fit,
        predict=fake_predict,
    )
    opportunity = subject.cross_validate_head(
        examples,
        "opportunity",
        "abs_delta18",
        ("fold0", "fold1", "fold2"),
        fit=fake_fit,
        predict=fake_predict,
    )
    assert len(direction["probabilities"]) == len(examples)
    assert len(opportunity["probabilities"]) == len(examples)
    assert all(audit["state_overlap_count"] == 0 for audit in direction["fold_audits"])
    assert all(audit["map_overlap_count"] == 0 for audit in opportunity["fold_audits"])
    assert all(
        labels == {"structural_win", "v2_win"}
        for head, labels in seen_fit_labels
        if head == "direction"
    )
    assert all(
        labels == {"structural_win", "ambiguous"}
        for head, labels in seen_fit_labels
        if head == "opportunity"
    )
