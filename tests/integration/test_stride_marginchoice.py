from __future__ import annotations

import json
from pathlib import Path

from experiments.stride_marginchoice import (
    build_marginchoice_labels,
    validate_marginchoice_design,
    validate_marginchoice_label_config,
    validate_marginchoice_training_config,
)


ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "configs" / "stride_marginchoice_design.json"
LABELS = ROOT / "configs" / "stride_marginchoice_labels.json"
TRAINING = ROOT / "configs" / "stride_marginchoice_training.json"


def test_marginchoice_uses_signed_seed_half_stable_current_step_target() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    validate_marginchoice_design(design)
    assert design["controller_id"] == "stride-marginchoice-v1"
    assert design["label_design"]["orientation"] == "candidate_minus_v2_anchor"
    assert design["label_design"]["target"] == (
        "minimum_of_first_and_second_eight_seed_mean_effects"
    )
    assert design["label_design"]["runtime_used_in_label"] is False
    assert design["label_design"]["future_trajectory_used_in_label"] is False


def test_marginchoice_scans_all_candidates_and_falls_back_to_v2() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    architecture = design["architecture"]
    assert architecture["candidate_actions"] == "all_non_anchor_candidates"
    assert architecture["maprank_role"] == "offline_comparator_only_not_proposal_filter"
    assert architecture["decision_rule"] == (
        "select_highest_predicted_conservative_margin_if_threshold_passes_else_v2_anchor"
    )


def test_marginchoice_training_is_nested_and_quality_gated() -> None:
    config = json.loads(TRAINING.read_text(encoding="utf-8"))
    validate_marginchoice_training_config(config)
    assert config["model_class"] == (
        "sklearn.ensemble.HistGradientBoostingRegressor"
    )
    assert config["outer_map_folds"] == 4
    assert config["inner_map_folds"] == 3
    assert config["calibration_constraints"][
        "minimum_directionally_safe_override_precision"
    ] == 0.60
    assert config["calibration_constraints"]["minimum_state_override_fraction"] == 0.08
    assert config["offline_gates"][
        "minimum_relative_normalized_regret_improvement_over_frozen_v2"
    ] == 0.05


def test_marginchoice_label_builder_rejects_unregistered_output(tmp_path: Path) -> None:
    config = json.loads(LABELS.read_text(encoding="utf-8"))
    validate_marginchoice_label_config(config)
    try:
        build_marginchoice_labels(config_path=LABELS, output=tmp_path)
    except ValueError as error:
        assert "output differs from registration" in str(error)
    else:
        raise AssertionError("MarginChoice accepted an unregistered label output")


def test_marginchoice_validator_rejects_relaxed_margin_correlation() -> None:
    config = json.loads(TRAINING.read_text(encoding="utf-8"))
    config["offline_gates"]["minimum_weighted_margin_correlation"] = 0.0
    try:
        validate_marginchoice_training_config(config)
    except ValueError as error:
        assert "offline gates changed" in str(error)
    else:
        raise AssertionError("MarginChoice accepted a relaxed regression gate")
