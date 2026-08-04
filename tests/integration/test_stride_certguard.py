from __future__ import annotations

import json
from pathlib import Path

from experiments.stride_certguard import (
    build_certguard_labels,
    validate_certguard_design,
    validate_certguard_label_config,
    validate_certguard_training_config,
)


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "configs" / "stride_certguard_design.json"
LABELS = ROOT / "configs" / "stride_certguard_labels.json"
TRAINING = ROOT / "configs" / "stride_certguard_training.json"


def test_certguard_is_a_distinct_abstaining_successor() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    validate_certguard_design(design)
    assert design["controller_id"] == "stride-certguard-v1"
    assert design["architecture"]["direction_model"] == (
        "frozen-maprank-direction-ranker"
    )
    assert design["architecture"]["uncertainty_role"] == "abstention_only"
    assert design["architecture"]["uncertainty_input_dimension"] == 248
    assert design["high_load_role"].startswith("post_hoc_development_only")
    assert design["fresh_map_role"] == "first_unseen_end_to_end_evidence"
    assert design["formal_speed_claim"] is False
    assert design["default_replacement_allowed"] is False


def test_certguard_label_protocol_keeps_all_robust_and_uncertain_pairs() -> None:
    config = json.loads(LABELS.read_text(encoding="utf-8"))
    validate_certguard_label_config(config)
    assert config["trial_indices"] == list(range(16))
    assert config["target"] == (
        "robust_current_step_winner_exists_for_unordered_candidate_pair"
    )
    assert config["maprank_reference"]["expected_possible_pair_count"] == 47471
    assert config["maprank_reference"]["expected_robust_pair_count"] == 21033
    assert config["maprank_reference"]["expected_uncertain_pair_count"] == 26438
    assert config["runtime_used_in_label"] is False
    assert config["future_trajectory_used_in_label"] is False


def test_certguard_training_is_train_map_only_and_preregistered() -> None:
    config = json.loads(TRAINING.read_text(encoding="utf-8"))
    validate_certguard_training_config(config)
    assert config["uncertainty_pair_representation"] == (
        "absolute_candidate_delta_plus_pair_mean"
    )
    assert config["frozen_direction"]["deployment_model_copied_without_retraining"]
    assert config["forbidden_calibration_splits"] == [
        "validation",
        "high_load",
        "test",
        "formal_ood",
    ]
    assert config["model_parameters"]["random_state"] == 20260805


def test_certguard_label_builder_rejects_unregistered_output(tmp_path: Path) -> None:
    try:
        build_certguard_labels(config_path=LABELS, output=tmp_path)
    except ValueError as error:
        assert "output differs from registration" in str(error)
    else:
        raise AssertionError("CertGuard accepted an unregistered label output")


def test_certguard_validator_rejects_post_hoc_high_load_calibration() -> None:
    config = json.loads(TRAINING.read_text(encoding="utf-8"))
    config["forbidden_calibration_splits"].remove("high_load")
    try:
        validate_certguard_training_config(config)
    except ValueError as error:
        assert "evidence boundary changed" in str(error)
    else:
        raise AssertionError("CertGuard accepted high-load threshold calibration")


def test_completed_certguard_training_respects_offline_gate_when_available() -> None:
    report_path = (
        ROOT
        / "build"
        / "stride-certguard-training-v1"
        / "certguard_training_report.json"
    )
    if not report_path.is_file():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "lns2.stride.certguard_training.v1"
    assert report["controller_id"] == "stride-certguard-v1"
    assert report["offline_passed"] is False
    assert report["runtime_bundle_exported"] is False
    assert report["shadow_eligible"] is False
    assert report["fresh_map_eligible"] is False
    assert report["high_load_used_for_training_or_calibration"] is False
    assert report["promotion_gates"]["uncertainty_roc_auc"] is True
    assert report["promotion_gates"]["uncertainty_accuracy_gain"] is True
    assert report["promotion_gates"]["directionally_safe_override_precision"] is False
    assert report["promotion_gates"]["normalized_regret_noninferior_to_maprank"] is False
    sweep = report["nested_train_oof"]["final_calibration"]["threshold_sweep"]
    assert [row["certainty_threshold"] for row in sweep] == [
        0.5,
        0.55,
        0.6,
        0.65,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        1.01,
    ]
    assert all(
        row["override_diagnostics"]["retained_override_count"] == 0
        for row in sweep
        if row["certainty_threshold"] >= 0.75
    )
