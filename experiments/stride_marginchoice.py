from __future__ import annotations

import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_augcontrol import _load_candidates
from experiments.stride_certguard import LABEL_REPORT_SCHEMA as CERTAINTY_REPORT_SCHEMA
from experiments.stride_certguard import LABEL_SCHEMA as CERTAINTY_LABEL_SCHEMA
from experiments.stride_maprank import _valid_sha256
from experiments.stride_stage4 import _frozen_predictions


DESIGN_SCHEMA = "lns2.stride.marginchoice_design.v1"
LABEL_CONFIG_SCHEMA = "lns2.stride.marginchoice_label_config.v1"
LABEL_SCHEMA = "lns2.stride.conservative_margin_label.v1"
LABEL_REPORT_SCHEMA = "lns2.stride.marginchoice_label_build.v1"
TRAINING_CONFIG_SCHEMA = "lns2.stride.marginchoice_training_config.v1"
CONTROLLER_ID = "stride-marginchoice-v1"
LABEL_ARTIFACT_ID = "stride-marginchoice-labels-v1"
FEATURE_SCHEMA = "lns2.realized_features.v2"
BASE_FEATURE_DIMENSION = 124
ACTION_FEATURE_DIMENSION = 248


def validate_marginchoice_design(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != DESIGN_SCHEMA
        or config.get("scientific_status")
        != "post_anchorchoice_rejection_preregistered_before_margin_label_build"
        or config.get("design_id") != "stride-marginchoice-design-v1"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("label_artifact_id") != LABEL_ARTIFACT_ID
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or config.get("motivation")
        != "replace_rare_binary_action_target_with_signed_seed_half_stable_current_step_margin"
    ):
        raise ValueError("STRIDE-MarginChoice design identity changed")
    if dict(config.get("architecture") or {}) != {
        "anchor": "frozen-v2-full",
        "candidate_pool": "frozen-maprank-base-plus-optional-topology-boundary",
        "candidate_actions": "all_non_anchor_candidates",
        "selection_model": "independent_action_conditioned_conservative_margin_regressor",
        "decision_rule": "select_highest_predicted_conservative_margin_if_threshold_passes_else_v2_anchor",
        "maprank_role": "offline_comparator_only_not_proposal_filter",
        "pair_representation": "oriented_candidate_minus_anchor_plus_pair_mean",
        "base_feature_dimension": BASE_FEATURE_DIMENSION,
        "action_input_dimension": ACTION_FEATURE_DIMENSION,
        "deterministic_tie_break": "candidate_id_ascending",
    }:
        raise ValueError("STRIDE-MarginChoice architecture changed")
    if dict(config.get("label_design") or {}) != {
        "schema": LABEL_SCHEMA,
        "unit": "candidate_action_against_frozen_v2_anchor_within_state",
        "source_measure": "current_step_normalized_conflict_reduction",
        "orientation": "candidate_minus_v2_anchor",
        "target": "minimum_of_first_and_second_eight_seed_mean_effects",
        "purpose": "preserve_direction_magnitude_and_seed_half_stability",
        "candidate_actions": "all_non_anchor_candidates",
        "sample_weighting": "equal_total_weight_per_state",
        "runtime_used_in_label": False,
        "future_trajectory_used_in_label": False,
    }:
        raise ValueError("STRIDE-MarginChoice label semantics changed")
    if list(config.get("evaluation_order") or ()) != [
        "nested_train_map_oof",
        "legacy_validation_descriptive",
        "runtime_semantic_equivalence_only_if_offline_passes",
        "legacy_shadow_only_if_offline_passes",
        "paired_high_load_development_raw_ttf_only_after_power_state_restored",
        "paired_fresh_map_raw_ttf_only_after_development_pass",
    ]:
        raise ValueError("STRIDE-MarginChoice evaluation order changed")
    if dict(config.get("performance_environment_policy") or {}) != {
        "power_state_source": "user_reported",
        "wall_clock_allowed_only_after_user_reports_comparable_performance_state": True,
        "additional_performance_preflight_required": False,
        "same_reported_power_state_all_comparators_rerun_required": True,
        "pool_timing_across_unmatched_power_states": False,
        "slow_charger_action": "allow_non_timing_work_only",
    }:
        raise ValueError("STRIDE-MarginChoice power-state boundary changed")
    if dict(config.get("evidence_roles") or {}) != {
        "legacy_validation": "descriptive_only",
        "high_load": "post_hoc_development_only",
        "fresh_map": "first_unseen_end_to_end_evidence",
    }:
        raise ValueError("STRIDE-MarginChoice evidence roles changed")
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
        raise ValueError("STRIDE-MarginChoice forbidden inputs changed")


def validate_marginchoice_label_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != LABEL_CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_conservative_margin_label_build"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("label_artifact_id") != LABEL_ARTIFACT_ID
        or config.get("label_schema") != LABEL_SCHEMA
        or config.get("registered_output") != "build/stride-marginchoice-labels-v1"
        or config.get("target")
        != "minimum_oriented_first_and_second_seed_half_mean_effect"
        or config.get("candidate_actions") != "all_non_anchor_candidates"
        or config.get("sample_weighting") != "equal_total_weight_per_state"
        or config.get("feature_schema") != FEATURE_SCHEMA
        or int(config.get("base_feature_dimension", -1)) != BASE_FEATURE_DIMENSION
        or int(config.get("expected_state_count", -1)) != 303
        or int(config.get("expected_candidate_action_count", -1)) != 5210
        or bool(config.get("runtime_used_in_label"))
        or bool(config.get("future_trajectory_used_in_label"))
    ):
        raise ValueError("STRIDE-MarginChoice label config changed")
    for name in (
        "certainty_pairs",
        "certainty_report",
        "candidate_aggregates",
        "selection",
        "frozen_v2_manifest",
    ):
        row = dict(config.get("sources", {}).get(name) or {})
        if not row.get("path") or not _valid_sha256(row.get("sha256")):
            raise ValueError(f"STRIDE-MarginChoice source changed: {name}")


def validate_marginchoice_training_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != TRAINING_CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_conservative_margin_outcomes"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("label_schema") != LABEL_SCHEMA
        or config.get("feature_schema") != FEATURE_SCHEMA
        or int(config.get("base_feature_dimension", -1)) != BASE_FEATURE_DIMENSION
        or int(config.get("action_input_dimension", -1)) != ACTION_FEATURE_DIMENSION
        or config.get("pair_representation")
        != "oriented_candidate_minus_anchor_plus_pair_mean"
        or config.get("registered_output")
        != "build/stride-marginchoice-training-v1"
        or config.get("design") != "configs/stride_marginchoice_design.json"
        or config.get("label_config") != "configs/stride_marginchoice_labels.json"
        or config.get("margin_labels")
        != "build/stride-marginchoice-labels-v1/conservative_margin_actions.jsonl"
        or config.get("maprank_labels") != "build/stride-maprank-labels-v1"
        or config.get("maprank_selection")
        != "build/stride-maprank-selection-v1/state_selection.jsonl"
        or config.get("maprank_bundle")
        != "build/stride-maprank-training-v1/stride-maprank-v1"
        or config.get("maprank_training_report")
        != "build/stride-maprank-training-v1/maprank_training_report.json"
    ):
        raise ValueError("STRIDE-MarginChoice training identity changed")
    if dict(config.get("selection_rule") or {}) != {
        "candidate_set": "all_non_anchor_candidates",
        "score": "predicted_conservative_seed_half_margin",
        "choice": "highest_score_then_candidate_id_ascending",
        "fallback": "v2_anchor",
        "maprank_used_as_proposal_filter": False,
    }:
        raise ValueError("STRIDE-MarginChoice selection rule changed")
    if (
        config.get("model_class")
        != "sklearn.ensemble.HistGradientBoostingRegressor"
        or dict(config.get("model_parameters") or {})
        != {
            "early_stopping": False,
            "l2_regularization": 0.1,
            "learning_rate": 0.05,
            "loss": "squared_error",
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "random_state": 20260808,
        }
        or int(config.get("outer_map_folds", -1)) != 4
        or int(config.get("inner_map_folds", -1)) != 3
        or list(map(float, config.get("selection_threshold_grid") or ()))
        != [
            -0.05,
            -0.04,
            -0.03,
            -0.02,
            -0.01,
            0.0,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.075,
            0.10,
            0.15,
            0.20,
            1.01,
        ]
        or config.get("calibration_objective")
        != "minimize_normalized_regret_subject_to_v2_noninferiority_safe_precision_and_state_coverage"
    ):
        raise ValueError("STRIDE-MarginChoice model protocol changed")
    if dict(config.get("calibration_constraints") or {}) != {
        "minimum_directionally_safe_override_precision": 0.60,
        "minimum_state_override_fraction": 0.08,
        "top3_hit_rate_noninferiority_tolerance_vs_v2": 0.01,
        "exact_best_rate_noninferiority_tolerance_vs_v2": 0.01,
        "infeasible_fold_action": "v2_fallback_and_offline_failure",
    }:
        raise ValueError("STRIDE-MarginChoice calibration constraints changed")
    if dict(config.get("offline_gates") or {}) != {
        "minimum_weighted_margin_correlation": 0.15,
        "minimum_relative_normalized_regret_improvement_over_frozen_v2": 0.05,
        "maximum_absolute_normalized_regret_degradation_vs_maprank": 0.005,
        "top3_hit_rate_noninferiority_tolerance_vs_v2": 0.01,
        "exact_best_rate_noninferiority_tolerance_vs_v2": 0.01,
        "maximum_unsafe_override_fraction": 0.40,
        "minimum_directionally_safe_override_precision": 0.60,
        "minimum_state_override_fraction": 0.08,
        "maximum_topology_group_normalized_regret_degradation_vs_v2": 0.03,
    }:
        raise ValueError("STRIDE-MarginChoice offline gates changed")
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
        raise ValueError("STRIDE-MarginChoice evidence boundary changed")
    for field in (
        "design_sha256",
        "label_config_sha256",
        "maprank_label_summary_sha256",
        "maprank_candidate_aggregates_sha256",
        "maprank_direction_pairs_sha256",
        "maprank_selection_sha256",
        "maprank_manifest_sha256",
        "maprank_training_report_sha256",
        "frozen_v2_manifest_sha256",
    ):
        if not _valid_sha256(config.get(field)):
            raise ValueError(f"STRIDE-MarginChoice hash is invalid: {field}")


def _registered_path(project_root: Path, row: dict[str, Any]) -> Path:
    path = (project_root / str(row["path"])).resolve()
    if sha256_file(path) != str(row["sha256"]):
        raise ValueError(f"MarginChoice registered source differs: {path}")
    return path


def build_marginchoice_labels(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_marginchoice_label_config(config)
    output_path = Path(output).resolve()
    if output_path != (project_root / str(config["registered_output"])).resolve():
        raise ValueError("STRIDE-MarginChoice label output differs from registration")
    sources = dict(config["sources"])
    certainty_path = _registered_path(project_root, dict(sources["certainty_pairs"]))
    certainty_report_path = _registered_path(
        project_root, dict(sources["certainty_report"])
    )
    aggregate_path = _registered_path(
        project_root, dict(sources["candidate_aggregates"])
    )
    selection_path = _registered_path(project_root, dict(sources["selection"]))
    v2_manifest_path = _registered_path(
        project_root, dict(sources["frozen_v2_manifest"])
    )
    certainty_report = _read_json(certainty_report_path)
    if (
        certainty_report.get("schema") != CERTAINTY_REPORT_SCHEMA
        or bool(certainty_report.get("runtime_used_in_label"))
        or bool(certainty_report.get("future_trajectory_used_in_label"))
        or bool(certainty_report.get("high_load_data_read"))
    ):
        raise ValueError("MarginChoice certainty source is invalid")
    _, grouped = _load_candidates(
        aggregate_path, selection_path, topology_threshold=0.06
    )
    anchors = _frozen_predictions(grouped, v2_manifest_path.parent)
    certainty: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in _read_jsonl(certainty_path):
        if row.get("schema") != CERTAINTY_LABEL_SCHEMA:
            raise ValueError("MarginChoice certainty schema differs")
        key = (
            str(row["state_id"]),
            str(row["left_candidate_id"]),
            str(row["right_candidate_id"]),
        )
        if key[1] >= key[2] or key in certainty:
            raise ValueError("MarginChoice certainty orientation differs")
        certainty[key] = row

    labels: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    state_weights: Counter[str] = Counter()
    positive_targets = 0
    robust_wins = 0
    margins: list[float] = []
    for state_id, rows in sorted(grouped.items()):
        anchor = str(anchors[state_id])
        actions = [row for row in rows if str(row["candidate_id"]) != anchor]
        if len(actions) != len(rows) - 1:
            raise ValueError("MarginChoice V2 anchor is absent or duplicated")
        weight = 1.0 / len(actions)
        for action in actions:
            candidate = str(action["candidate_id"])
            left, right = sorted((candidate, anchor))
            source = certainty[(state_id, left, right)]
            orientation = 1.0 if candidate == left else -1.0
            mean_effect = orientation * float(source["mean_left_minus_right_effect"])
            first_effect = orientation * float(source["first_half_mean_effect"])
            second_effect = orientation * float(source["second_half_mean_effect"])
            conservative_margin = min(first_effect, second_effect)
            robust_candidate_win = int(
                int(source["label"]) == 1
                and str(source["robust_winner_candidate_id"]) == candidate
            )
            split = str(action["split"])
            labels.append(
                {
                    "schema": LABEL_SCHEMA,
                    "state_id": state_id,
                    "map_id": str(action["map_id"]),
                    "split": split,
                    "candidate_id": candidate,
                    "anchor_candidate_id": anchor,
                    "mean_effect": mean_effect,
                    "first_half_mean_effect": first_effect,
                    "second_half_mean_effect": second_effect,
                    "conservative_margin": conservative_margin,
                    "half_effect_gap": abs(first_effect - second_effect),
                    "robust_candidate_win": robust_candidate_win,
                    "sample_weight": weight,
                }
            )
            split_counts[split] += 1
            state_weights[state_id] += weight
            positive_targets += conservative_margin > 0.0
            robust_wins += robust_candidate_win
            margins.append(conservative_margin)
    if (
        len(grouped) != int(config["expected_state_count"])
        or len(labels) != int(config["expected_candidate_action_count"])
        or any(
            not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9)
            for value in state_weights.values()
        )
        or not any(value > 0.0 for value in margins)
        or not any(value < 0.0 for value in margins)
    ):
        raise ValueError("MarginChoice label coverage differs")
    output_path.mkdir(parents=True, exist_ok=True)
    label_path = output_path / "conservative_margin_actions.jsonl"
    _write_jsonl(label_path, labels)
    report = {
        "schema": LABEL_REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "label_artifact_id": LABEL_ARTIFACT_ID,
        "label_schema": LABEL_SCHEMA,
        "scientific_status": "train_and_legacy_validation_margin_labels_built",
        "state_count": len(grouped),
        "map_count": len({str(rows[0]["map_id"]) for rows in grouped.values()}),
        "candidate_action_count": len(labels),
        "candidate_action_count_by_split": dict(sorted(split_counts.items())),
        "positive_conservative_margin_count": positive_targets,
        "robust_candidate_win_count": robust_wins,
        "mean_conservative_margin": statistics.fmean(margins),
        "minimum_conservative_margin": min(margins),
        "maximum_conservative_margin": max(margins),
        "sample_weighting": config["sample_weighting"],
        "feature_schema": FEATURE_SCHEMA,
        "base_feature_dimension": BASE_FEATURE_DIMENSION,
        "pair_representation": "oriented_candidate_minus_anchor_plus_pair_mean",
        "action_input_dimension": ACTION_FEATURE_DIMENSION,
        "runtime_used_in_label": False,
        "future_trajectory_used_in_label": False,
        "high_load_data_read": False,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "config_sha256": sha256_file(config_path),
        "source_sha256": {
            name: sha256_file(_registered_path(project_root, dict(row)))
            for name, row in sources.items()
        },
        "artifacts": {
            "conservative_margin_actions": label_path.name,
            "conservative_margin_actions_sha256": sha256_file(label_path),
        },
    }
    _write_json(output_path / "marginchoice_label_report.json", report)
    return report


__all__ = [
    "CONTROLLER_ID",
    "LABEL_SCHEMA",
    "build_marginchoice_labels",
    "validate_marginchoice_design",
    "validate_marginchoice_label_config",
    "validate_marginchoice_training_config",
]
