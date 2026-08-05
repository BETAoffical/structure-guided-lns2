from __future__ import annotations

from typing import Any

from experiments.stride_maprank import _valid_sha256


DESIGN_SCHEMA = "lns2.stride.anchorchoice_design.v1"
TRAINING_CONFIG_SCHEMA = "lns2.stride.anchorchoice_training_config.v1"
CONTROLLER_ID = "stride-anchorchoice-v1"
LABEL_SCHEMA = "lns2.stride.override_safety_label.v1"
FEATURE_SCHEMA = "lns2.realized_features.v2"
BASE_FEATURE_DIMENSION = 124
ACTION_FEATURE_DIMENSION = 248


def validate_anchorchoice_design(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != DESIGN_SCHEMA
        or config.get("scientific_status")
        != "post_overrideguard_rejection_preregistered_before_direct_action_oof"
        or config.get("design_id") != "stride-anchorchoice-design-v1"
        or config.get("controller_id") != CONTROLLER_ID
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or config.get("motivation")
        != "remove_the_low_quality_single_maprank_proposal_bottleneck_without_relaxing_safety_labels"
    ):
        raise ValueError("STRIDE-AnchorChoice design identity changed")
    if dict(config.get("architecture") or {}) != {
        "anchor": "frozen-v2-full",
        "candidate_pool": "frozen-maprank-base-plus-optional-topology-boundary",
        "candidate_actions": "all_non_anchor_candidates",
        "selection_model": "independent-action-conditioned-current-step-classifier",
        "decision_rule": "select_highest_predicted_robust_candidate_win_if_threshold_passes_else_v2_anchor",
        "maprank_role": "offline_comparator_only_not_proposal_filter",
        "pair_representation": "oriented_candidate_minus_anchor_plus_pair_mean",
        "base_feature_dimension": BASE_FEATURE_DIMENSION,
        "action_input_dimension": ACTION_FEATURE_DIMENSION,
        "deterministic_tie_break": "candidate_id_ascending",
    }:
        raise ValueError("STRIDE-AnchorChoice architecture changed")
    if dict(config.get("label_contract") or {}) != {
        "schema": LABEL_SCHEMA,
        "positive": "candidate_is_registered_robust_winner_against_anchor",
        "negative": "pair_is_uncertain_or_anchor_is_registered_robust_winner",
        "source": "frozen_stride_overrideguard_labels_v1",
        "runtime_used_in_label": False,
        "future_trajectory_used_in_label": False,
    }:
        raise ValueError("STRIDE-AnchorChoice label contract changed")
    if list(config.get("evaluation_order") or ()) != [
        "nested_train_map_oof",
        "legacy_validation_descriptive",
        "runtime_semantic_equivalence_only_if_offline_passes",
        "legacy_shadow_only_if_offline_passes",
        "paired_high_load_development_raw_ttf_only_after_power_state_restored",
        "paired_fresh_map_raw_ttf_only_after_development_pass",
    ]:
        raise ValueError("STRIDE-AnchorChoice evaluation order changed")
    if dict(config.get("performance_environment_policy") or {}) != {
        "power_state_source": "user_reported",
        "wall_clock_allowed_only_after_user_reports_comparable_performance_state": True,
        "additional_performance_preflight_required": False,
        "same_reported_power_state_all_comparators_rerun_required": True,
        "pool_timing_across_unmatched_power_states": False,
        "slow_charger_action": "allow_non_timing_work_only",
    }:
        raise ValueError("STRIDE-AnchorChoice power-state boundary changed")
    if dict(config.get("evidence_roles") or {}) != {
        "legacy_validation": "descriptive_only",
        "high_load": "post_hoc_development_only",
        "fresh_map": "first_unseen_end_to_end_evidence",
    }:
        raise ValueError("STRIDE-AnchorChoice evidence roles changed")
    if set(map(str, config.get("forbidden_inputs") or ())) != {
        "repair_runtime",
        "time_to_feasible",
        "future_repair_rounds",
        "cost_to_go",
        "receding_q",
        "controller_outcome",
        "high_load_labels_for_training_or_threshold_calibration",
        "legacy_validation_labels_for_threshold_calibration",
        "test_data",
        "formal_ood_data",
    }:
        raise ValueError("STRIDE-AnchorChoice forbidden inputs changed")


def validate_anchorchoice_training_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != TRAINING_CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_direct_action_oof_outcomes"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("label_schema") != LABEL_SCHEMA
        or config.get("feature_schema") != FEATURE_SCHEMA
        or int(config.get("base_feature_dimension", -1)) != BASE_FEATURE_DIMENSION
        or int(config.get("action_input_dimension", -1)) != ACTION_FEATURE_DIMENSION
        or config.get("pair_representation")
        != "oriented_candidate_minus_anchor_plus_pair_mean"
        or config.get("registered_output")
        != "build/stride-anchorchoice-training-v1"
        or config.get("design") != "configs/stride_anchorchoice_design.json"
        or config.get("action_labels")
        != "build/stride-overrideguard-labels-v1/action_safety_pairs.jsonl"
        or config.get("action_label_report")
        != "build/stride-overrideguard-labels-v1/overrideguard_label_report.json"
        or config.get("maprank_labels") != "build/stride-maprank-labels-v1"
        or config.get("maprank_selection")
        != "build/stride-maprank-selection-v1/state_selection.jsonl"
        or config.get("maprank_bundle")
        != "build/stride-maprank-training-v1/stride-maprank-v1"
        or config.get("maprank_training_report")
        != "build/stride-maprank-training-v1/maprank_training_report.json"
    ):
        raise ValueError("STRIDE-AnchorChoice training identity changed")
    if dict(config.get("selection_rule") or {}) != {
        "candidate_set": "all_non_anchor_candidates",
        "score": "predicted_probability_candidate_robustly_beats_v2_anchor",
        "choice": "highest_score_then_candidate_id_ascending",
        "fallback": "v2_anchor",
        "maprank_used_as_proposal_filter": False,
    }:
        raise ValueError("STRIDE-AnchorChoice selection rule changed")
    if (
        config.get("model_class")
        != "sklearn.ensemble.HistGradientBoostingClassifier"
        or dict(config.get("model_parameters") or {})
        != {
            "early_stopping": False,
            "l2_regularization": 0.1,
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "random_state": 20260807,
        }
        or int(config.get("outer_map_folds", -1)) != 4
        or int(config.get("inner_map_folds", -1)) != 3
        or list(map(float, config.get("selection_threshold_grid") or ()))
        != [
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            1.01,
        ]
        or config.get("calibration_objective")
        != "minimize_normalized_regret_subject_to_v2_noninferiority_safe_precision_and_state_coverage"
    ):
        raise ValueError("STRIDE-AnchorChoice model protocol changed")
    if dict(config.get("calibration_constraints") or {}) != {
        "minimum_directionally_safe_override_precision": 0.60,
        "minimum_state_override_fraction": 0.08,
        "top3_hit_rate_noninferiority_tolerance_vs_v2": 0.01,
        "exact_best_rate_noninferiority_tolerance_vs_v2": 0.01,
        "infeasible_fold_action": "v2_fallback_and_offline_failure",
    }:
        raise ValueError("STRIDE-AnchorChoice calibration constraints changed")
    if dict(config.get("offline_gates") or {}) != {
        "minimum_safety_roc_auc": 0.65,
        "minimum_relative_normalized_regret_improvement_over_frozen_v2": 0.05,
        "maximum_absolute_normalized_regret_degradation_vs_maprank": 0.005,
        "top3_hit_rate_noninferiority_tolerance_vs_v2": 0.01,
        "exact_best_rate_noninferiority_tolerance_vs_v2": 0.01,
        "maximum_unsafe_override_fraction": 0.40,
        "minimum_directionally_safe_override_precision": 0.60,
        "minimum_state_override_fraction": 0.08,
        "maximum_topology_group_normalized_regret_degradation_vs_v2": 0.03,
    }:
        raise ValueError("STRIDE-AnchorChoice offline gates changed")
    if (
        config.get("training_split") != "train"
        or config.get("legacy_validation_split") != "validation"
        or config.get("split_unit") != "map"
        or config.get("sample_weighting") != "equal_total_weight_per_state"
        or set(map(str, config.get("forbidden_calibration_splits") or ()))
        != {"validation", "high_load", "test", "formal_ood"}
        or config.get("legacy_validation_is_descriptive_only") is not True
        or config.get("high_load_is_development_only") is not True
        or bool(config.get("runtime_used_in_label"))
        or bool(config.get("future_trajectory_used_in_label"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("STRIDE-AnchorChoice evidence boundary changed")
    for field in (
        "design_sha256",
        "action_labels_sha256",
        "action_label_report_sha256",
        "maprank_label_summary_sha256",
        "maprank_candidate_aggregates_sha256",
        "maprank_direction_pairs_sha256",
        "maprank_selection_sha256",
        "maprank_manifest_sha256",
        "maprank_training_report_sha256",
        "frozen_v2_manifest_sha256",
    ):
        if not _valid_sha256(config.get(field)):
            raise ValueError(f"STRIDE-AnchorChoice hash is invalid: {field}")


__all__ = [
    "CONTROLLER_ID",
    "validate_anchorchoice_design",
    "validate_anchorchoice_training_config",
]
