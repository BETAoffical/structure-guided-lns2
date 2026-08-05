from __future__ import annotations

import json
from pathlib import Path

from experiments.stride_anchorchoice import (
    validate_anchorchoice_design,
    validate_anchorchoice_training_config,
)


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "configs" / "stride_anchorchoice_design.json"
TRAINING = ROOT / "configs" / "stride_anchorchoice_training.json"


def test_anchorchoice_scans_all_actions_without_maprank_proposal_filter() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    validate_anchorchoice_design(design)
    architecture = design["architecture"]
    assert design["controller_id"] == "stride-anchorchoice-v1"
    assert architecture["candidate_actions"] == "all_non_anchor_candidates"
    assert architecture["maprank_role"] == "offline_comparator_only_not_proposal_filter"
    assert architecture["decision_rule"] == (
        "select_highest_predicted_robust_candidate_win_if_threshold_passes_else_v2_anchor"
    )
    assert design["default_replacement_allowed"] is False
    assert design["formal_speed_claim"] is False


def test_anchorchoice_training_is_nested_and_quality_gated() -> None:
    config = json.loads(TRAINING.read_text(encoding="utf-8"))
    validate_anchorchoice_training_config(config)
    assert config["outer_map_folds"] == 4
    assert config["inner_map_folds"] == 3
    assert config["calibration_constraints"] == {
        "minimum_directionally_safe_override_precision": 0.60,
        "minimum_state_override_fraction": 0.08,
        "top3_hit_rate_noninferiority_tolerance_vs_v2": 0.01,
        "exact_best_rate_noninferiority_tolerance_vs_v2": 0.01,
        "infeasible_fold_action": "v2_fallback_and_offline_failure",
    }
    assert config["offline_gates"][
        "minimum_relative_normalized_regret_improvement_over_frozen_v2"
    ] == 0.05
    assert config["selection_rule"]["maprank_used_as_proposal_filter"] is False


def test_anchorchoice_uses_user_reported_power_state_without_extra_preflight() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    policy = design["performance_environment_policy"]
    assert policy["power_state_source"] == "user_reported"
    assert policy["additional_performance_preflight_required"] is False
    assert policy["slow_charger_action"] == "allow_non_timing_work_only"


def test_anchorchoice_validator_rejects_relaxed_safe_precision() -> None:
    config = json.loads(TRAINING.read_text(encoding="utf-8"))
    config["offline_gates"]["minimum_directionally_safe_override_precision"] = 0.0
    try:
        validate_anchorchoice_training_config(config)
    except ValueError as error:
        assert "offline gates changed" in str(error)
    else:
        raise AssertionError("AnchorChoice accepted a relaxed safety gate")
