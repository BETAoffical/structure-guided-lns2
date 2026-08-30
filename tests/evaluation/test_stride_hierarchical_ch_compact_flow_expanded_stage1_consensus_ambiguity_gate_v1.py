from __future__ import annotations

import json
from pathlib import Path

import pytest

import experiments.stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_v1 as subject


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_ambiguity_gate_v1.json"
)
PREDECESSOR_CONFIG_PATH = (
    ROOT
    / "configs"
    / "stride_hierarchical_ch_compact_flow_expanded_stage1_consensus_only_v1.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _example(index: int, label: str) -> dict:
    return {
        "pair_id": f"pair-{index}",
        "state_occurrence_id": f"state-{index}",
        "label": label,
        "opportunity_target": int(label in subject.DECISIVE_LABELS),
        "direction_target": (
            1 if label == "structural_win" else 0 if label == "v2_win" else None
        ),
        "opportunity_features_signed18": [0.0] * 18,
        "opportunity_features_abs_delta18": [0.0] * 18,
        "direction_features_signed18": [0.0] * 18,
    }


def test_registered_control_changes_only_the_frozen_opportunity_contract() -> None:
    config = _read(CONFIG_PATH)
    predecessor = _read(PREDECESSOR_CONFIG_PATH)
    subject.validate_config(config)
    subject._validate_frozen_equivalence(config, predecessor)

    assert config["predecessor_contract"]["allowed_method_differences"] == [
        "opportunity_target",
        "opportunity_fit_population",
        "opportunity_class_balance",
    ]
    assert config["head_contract"]["opportunity"] == {
        "target": "decisive_structural_or_v2_vs_ambiguous",
        "fit_labels": ["structural_win", "v2_win", "ambiguous"],
        "fit_population_count": 93,
        "state_equal_weighting": True,
        "class_balance": False,
    }
    assert (
        config["head_contract"]["direction"]
        == predecessor["head_contract"]["direction"]
    )
    assert config["claim_boundary"]["study_role"] == subject.STUDY_ROLE
    assert config["claim_boundary"]["passing_status"] == (
        "POSTHOC_MECHANISM_SUPPORT_ONLY"
    )
    assert config["claim_boundary"]["failure_status"] == "POSTHOC_CONTROL_NO_GO"
    assert config["claim_boundary"]["protocol_mismatch_status"] == "INVALID_CONTROL"
    assert config["claim_boundary"]["challenger_claim_allowed"] is False


def test_opportunity_fits_all_93_with_natural_state_equal_mass() -> None:
    examples = [
        *[_example(index, "ambiguous") for index in range(69)],
        *[_example(69 + index, "structural_win") for index in range(14)],
        *[_example(83 + index, "v2_win") for index in range(10)],
    ]
    rows = subject._head_rows(examples, "opportunity", "signed18")
    weights, audit = subject._head_weights(rows, "opportunity")

    assert len(rows) == 93
    assert [row["target"] for row in rows].count(0) == 69
    assert [row["target"] for row in rows].count(1) == 24
    assert weights == [1.0] * 93
    assert audit["base_class_weight_totals"] == {"0": 69.0, "1": 24.0}
    assert audit["effective_class_weight_totals"] == {"0": 69.0, "1": 24.0}
    assert audit["class_balance_applied"] is False


def test_direction_remains_decisive_only_and_class_balanced() -> None:
    examples = [
        *[_example(index, "ambiguous") for index in range(69)],
        *[_example(69 + index, "structural_win") for index in range(14)],
        *[_example(83 + index, "v2_win") for index in range(10)],
    ]
    rows = subject._head_rows(examples, "direction", "signed18")
    _, audit = subject._head_weights(rows, "direction")

    assert len(rows) == 24
    assert all(row["label"] != "ambiguous" for row in rows)
    assert audit["base_class_weight_totals"] == {"0": 10.0, "1": 14.0}
    assert audit["effective_class_weight_totals"]["0"] == pytest.approx(
        audit["effective_class_weight_totals"]["1"]
    )
    assert audit["class_balance_applied"] is True


def test_paired_audit_compares_direction_only_when_both_outer_predictions_exist() -> None:
    metrics = {
        "full_pooled_decisive_balanced_accuracy": 0.75,
        "selected_action_precision": 0.70,
        "structural_win_override_recall": 0.50,
        "v2_win_protection_recall": 1.0,
        "ambiguous_override_rate": 0.05,
        "total_override_count": 10,
        "correct_structural_override_count": 7,
        "v2_wrong_override_count": 0,
        "correct_structural_override_map_count": 4,
        "correct_structural_override_fold_count": 2,
    }
    predecessor_metrics = {**metrics, "selected_action_precision": 0.60}
    predictions = [
        {
            "pair_id": f"pair-{index}",
            "direction_structural_probability": 0.25,
        }
        for index in range(93)
    ]
    references = [
        {
            "pair_id": f"pair-{index}",
            "direction_structural_probability": 0.25 + 5e-13,
        }
        for index in range(93)
    ]
    audit = subject._paired_reference_audit(
        predictions,
        references,
        metrics,
        {"primary_full_pooled_selected_action_metrics": predecessor_metrics},
    )

    assert audit["paired_row_count"] == 93
    assert audit["direction_comparable_pair_count"] == 93
    assert audit["direction_inactive_or_incomparable_pair_count"] == 0
    assert audit["direction_complete_cohort_comparison_passed"] is True
    assert audit["direction_probability_reproduction_passed"] is True
    assert audit["paired_metric_delta_new_minus_predecessor"][
        "selected_action_precision"
    ] == pytest.approx(0.10)

    predictions[0]["direction_structural_probability"] = None
    incomplete = subject._paired_reference_audit(
        predictions,
        references,
        metrics,
        {"primary_full_pooled_selected_action_metrics": predecessor_metrics},
    )
    assert incomplete["direction_comparable_pair_count"] == 92
    assert incomplete["direction_probability_reproduction_passed"] is False

    predictions[0]["direction_structural_probability"] = 0.25
    references[0]["direction_structural_probability"] = 0.25000001
    mismatch = subject._paired_reference_audit(
        predictions,
        references,
        metrics,
        {"primary_full_pooled_selected_action_metrics": predecessor_metrics},
    )
    assert mismatch["direction_probability_reproduction_passed"] is False


def test_prediction_schema_uses_decisive_probability_and_keeps_inactive_v2() -> None:
    row = {
        **_example(1, "ambiguous"),
        "map_id": "den404d",
        "train_fold": "fold0",
        "component_action_id": "structural",
        "hotspot_action_id": "structural",
        "structural_action_id": "structural",
        "v2_action_id": "v2",
        "exact_structural_agents": [1, 2],
        "exact_v2_agents": [2, 3],
    }
    inactive = subject.frozen.materialize_action(
        row,
        opportunity_probability=None,
        direction_probability=0.375,
        opportunity_threshold=1.01,
        direction_threshold=1.01,
        arm=None,
        evaluation_active=False,
        selection_status="test_inactive_gate",
    )
    output = subject._prediction_output(inactive, "fingerprint")

    assert output["evaluation_active"] is False
    assert output["final_action_id"] == "v2"
    assert output["direction_structural_probability"] == pytest.approx(0.375)
    assert output["opportunity_decisive_probability"] is None
    assert output["opportunity_decisive_probability_semantics"] == (
        "probability_of_structural_or_v2_decisive_vs_ambiguous"
    )
    assert "opportunity_structural_probability" not in output


def test_control_reference_hashes_and_no_model_claim_are_frozen() -> None:
    config = _read(CONFIG_PATH)
    assert config["inputs"]["predecessor_config"]["sha256"] == (
        subject.PREDECESSOR_CONFIG_SHA256
    )
    assert config["inputs"]["predecessor_evaluation_report"]["sha256"] == (
        subject.PREDECESSOR_REPORT_SHA256
    )
    assert config["inputs"]["predecessor_oof_predictions"]["sha256"] == (
        subject.PREDECESSOR_OOF_SHA256
    )
    assert config["predecessor_contract"][
        "control_reference_used_for_fit_or_threshold_selection"
    ] is False
    assert "model" not in config["outputs"]
    assert config["claim_boundary"]["model_export_allowed"] is False
    assert config["claim_boundary"]["v2_replacement_allowed"] is False
