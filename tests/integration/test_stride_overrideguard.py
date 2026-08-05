from __future__ import annotations

import json
from pathlib import Path

from experiments.stride_overrideguard import (
    build_overrideguard_labels,
    validate_overrideguard_design,
    validate_overrideguard_label_config,
    validate_overrideguard_training_config,
)


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "configs" / "stride_overrideguard_design.json"
LABELS = ROOT / "configs" / "stride_overrideguard_labels.json"
TRAINING = ROOT / "configs" / "stride_overrideguard_training.json"


def test_overrideguard_is_action_conditioned_and_abstention_only() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    validate_overrideguard_design(design)
    assert design["controller_id"] == "stride-overrideguard-v1"
    assert design["architecture"]["pair_representation"] == (
        "oriented_candidate_minus_anchor_plus_pair_mean"
    )
    assert design["architecture"]["safety_role"] == "abstention_only"
    assert design["label_design"]["positive"] == (
        "candidate_is_registered_robust_winner_against_anchor"
    )
    assert design["label_design"]["negative"] == (
        "pair_is_uncertain_or_anchor_is_registered_robust_winner"
    )
    assert design["formal_speed_claim"] is False
    assert design["default_replacement_allowed"] is False


def test_overrideguard_uses_reported_power_state_without_extra_preflight() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    validate_overrideguard_design(design)
    policy = design["performance_environment_policy"]
    assert policy["power_state_source"] == "user_reported"
    assert policy["wall_clock_allowed_only_after_user_reports_comparable_performance_state"]
    assert policy["additional_performance_preflight_required"] is False
    assert policy["same_reported_power_state_all_comparators_rerun_required"] is True
    assert policy["pool_timing_across_unmatched_power_states"] is False
    assert policy["slow_charger_action"] == "allow_non_timing_work_only"


def test_overrideguard_label_build_is_v2_relative_and_current_step_only() -> None:
    config = json.loads(LABELS.read_text(encoding="utf-8"))
    validate_overrideguard_label_config(config)
    assert config["expected_state_count"] == 303
    assert config["expected_candidate_action_count"] == 5210
    assert config["candidate_actions"] == "all_non_anchor_candidates"
    assert config["negative_semantics"] == "uncertain_pair_or_robust_anchor_win"
    assert config["runtime_used_in_label"] is False
    assert config["future_trajectory_used_in_label"] is False


def test_overrideguard_training_is_nested_train_map_only() -> None:
    config = json.loads(TRAINING.read_text(encoding="utf-8"))
    validate_overrideguard_training_config(config)
    assert config["frozen_proposal"]["deployment_model_copied_without_retraining"]
    assert config["outer_map_folds"] == 4
    assert config["inner_map_folds"] == 3
    assert config["calibration_constraints"] == {
        "minimum_directionally_safe_override_precision": 0.60,
        "minimum_maprank_override_retention_fraction": 0.20,
        "top3_hit_rate_noninferiority_tolerance_vs_v2": 0.01,
        "exact_best_rate_noninferiority_tolerance_vs_v2": 0.01,
        "infeasible_fold_action": "v2_fallback_and_offline_failure",
    }
    assert config["forbidden_calibration_splits"] == [
        "validation",
        "high_load",
        "test",
        "formal_ood",
    ]


def test_overrideguard_builder_rejects_unregistered_output(tmp_path: Path) -> None:
    try:
        build_overrideguard_labels(config_path=LABELS, output=tmp_path)
    except ValueError as error:
        assert "output differs from registration" in str(error)
    else:
        raise AssertionError("OverrideGuard accepted an unregistered label output")


def test_overrideguard_validator_rejects_relaxed_safe_precision() -> None:
    config = json.loads(TRAINING.read_text(encoding="utf-8"))
    config["offline_gates"]["minimum_directionally_safe_override_precision"] = 0.0
    try:
        validate_overrideguard_training_config(config)
    except ValueError as error:
        assert "offline gates changed" in str(error)
    else:
        raise AssertionError("OverrideGuard accepted a relaxed safety gate")
