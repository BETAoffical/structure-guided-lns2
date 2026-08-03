from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.stride_lns import (
    FROZEN_FEATURE_DIMENSION,
    FROZEN_FEATURE_SCHEMA_ID,
    REQUIRED_POST_STRUCTURE_FIELDS,
    STRIDE_TRIAL_SCHEMA,
    assign_structure_scores,
)
from experiments.stride_stage3 import _project_path
from experiments.v2_factorial_audit import _select_model, pair_vector
from lns2_selector.runtime.online_selection import score_online_candidates


CONFIG_SCHEMA = "lns2.stride.robuststep_design_config.v1"
REPORT_SCHEMA = "lns2.stride.robuststep_design_report.v1"
SEED_DEPTH_CONFIG_SCHEMA = "lns2.stride.robuststep_seed_depth_config.v1"
SEED_DEPTH_REPORT_SCHEMA = "lns2.stride.robuststep_seed_depth_report.v1"
SCORE_CONFIG_SCHEMA = "lns2.stride.robuststep_score_design_config.v1"
SCORE_REPORT_SCHEMA = "lns2.stride.robuststep_score_design_report.v1"
CONFIRMATION_CONFIG_SCHEMA = "lns2.stride.robuststep_confirmation_config.v1"
CONFIRMATION_REPORT_SCHEMA = "lns2.stride.robuststep_confirmation_report.v1"
V2_HEADROOM_CONFIG_SCHEMA = "lns2.stride.robuststep_v2_headroom_config.v1"
V2_HEADROOM_REPORT_SCHEMA = "lns2.stride.robuststep_v2_headroom_report.v1"
FEATURE_PROBE_CONFIG_SCHEMA = "lns2.stride.robuststep_feature_probe_config.v1"
FEATURE_PROBE_REPORT_SCHEMA = "lns2.stride.robuststep_feature_probe_report.v1"
STEPGATE_CONFIG_SCHEMA = "lns2.stride.robuststep_stepgate_config.v1"
STEPGATE_REPORT_SCHEMA = "lns2.stride.robuststep_stepgate_report.v1"
CONTROLLER_ID = "stride-robuststep-v1"
LABEL_SCHEMA = "lns2.stride.robust_step_label.v1"


def validate_robuststep_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected robust-step design config")
    if (
        config.get("scientific_status") != "consumed_design_only"
        or bool(config.get("formal_speed_claim"))
        or not bool(config.get("fresh_confirmation_required"))
    ):
        raise ValueError("robust-step design must remain consumed and non-formal")
    if config.get("controller_id") != CONTROLLER_ID:
        raise ValueError("unexpected robust-step controller id")
    if config.get("label_schema") != LABEL_SCHEMA:
        raise ValueError("unexpected robust-step label schema")
    indices = tuple(map(int, config.get("trial_indices") or ()))
    first = tuple(map(int, config.get("first_half_indices") or ()))
    second = tuple(map(int, config.get("second_half_indices") or ()))
    if indices != tuple(range(8)) or first != tuple(range(4)) or second != tuple(range(4, 8)):
        raise ValueError("robust-step requires registered 4+4 paired trial indices")
    if float(config.get("structure_weight", -1.0)) != 0.02:
        raise ValueError("robust-step structure weight differs")
    variants = list(config.get("variants") or [])
    ids = [str(row.get("id")) for row in variants]
    if len(variants) != 4 or len(set(ids)) != 4:
        raise ValueError("robust-step requires four unique design variants")
    for row in variants:
        win = float(row.get("minimum_paired_win_fraction", 0.0))
        tolerance = float(row.get("maximum_no_progress_disadvantage", -1.0))
        if not 0.5 < win <= 1.0 or not 0.0 <= tolerance <= 0.125:
            raise ValueError("invalid robust-step design variant")
    if bool(config.get("runtime_used_in_label")):
        raise ValueError("runtime cannot enter the robust-step label")


def validate_robuststep_seed_depth_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SEED_DEPTH_CONFIG_SCHEMA:
        raise ValueError("unexpected robust-step seed-depth config")
    if (
        config.get("scientific_status") != "consumed_seed_depth_diagnostic"
        or bool(config.get("formal_speed_claim"))
        or not bool(config.get("fresh_confirmation_required"))
    ):
        raise ValueError("seed-depth analysis must remain consumed and non-formal")
    if config.get("controller_id") != CONTROLLER_ID:
        raise ValueError("unexpected robust-step controller id")
    if config.get("label_schema") != LABEL_SCHEMA:
        raise ValueError("unexpected robust-step label schema")
    indices = tuple(map(int, config.get("trial_indices") or ()))
    first = tuple(map(int, config.get("first_half_indices") or ()))
    second = tuple(map(int, config.get("second_half_indices") or ()))
    if (
        indices != tuple(range(16))
        or first != tuple(range(8))
        or second != tuple(range(8, 16))
    ):
        raise ValueError("seed-depth diagnostic requires registered 8+8 indices")
    if float(config.get("structure_weight", -1.0)) != 0.02:
        raise ValueError("robust-step structure weight differs")
    variants = list(config.get("variants") or [])
    if len(variants) != 4 or len({str(row.get("id")) for row in variants}) != 4:
        raise ValueError("seed-depth diagnostic requires four unique variants")
    if not bool(config.get("require_all_cohorts_pass")):
        raise ValueError("seed-depth diagnostic must gate every registered cohort")
    cohorts = list(config.get("cohorts") or [])
    if len(cohorts) != 2 or len({str(row.get("id")) for row in cohorts}) != 2:
        raise ValueError("seed-depth diagnostic requires two disjoint cohorts")
    if bool(config.get("runtime_used_in_label")):
        raise ValueError("runtime cannot enter the robust-step label")


def validate_robuststep_score_config(config: dict[str, Any]) -> None:
    if config.get("schema") != SCORE_CONFIG_SCHEMA:
        raise ValueError("unexpected robust-step score config")
    if (
        config.get("scientific_status") != "consumed_score_design_only"
        or bool(config.get("formal_speed_claim"))
        or not bool(config.get("fresh_confirmation_required"))
    ):
        raise ValueError("score design must remain consumed and non-formal")
    if config.get("controller_id") != CONTROLLER_ID:
        raise ValueError("unexpected robust-step controller id")
    if config.get("score_schema") != "lns2.stride.robust_step_score.v1":
        raise ValueError("unexpected robust-step score schema")
    indices = tuple(map(int, config.get("trial_indices") or ()))
    first = tuple(map(int, config.get("first_half_indices") or ()))
    second = tuple(map(int, config.get("second_half_indices") or ()))
    if (
        indices != tuple(range(16))
        or first != tuple(range(8))
        or second != tuple(range(8, 16))
    ):
        raise ValueError("score design requires registered 8+8 indices")
    variants = list(config.get("variants") or [])
    ids = [str(row.get("id")) for row in variants]
    if len(variants) != 5 or len(set(ids)) != 5 or ids[0] != "mean":
        raise ValueError("score design requires mean plus four risk variants")
    for row in variants:
        if str(row.get("mode")) not in {"mean", "lower_quartile"}:
            raise ValueError("unsupported robust-step score mode")
        if float(row.get("deviation_weight", -1.0)) < 0.0:
            raise ValueError("score deviation weight must be nonnegative")
        if float(row.get("no_progress_penalty", -1.0)) < 0.0:
            raise ValueError("score no-progress penalty must be nonnegative")
    if config.get("baseline_variant_id") != "mean":
        raise ValueError("plain mean must remain the registered score baseline")
    if bool(config.get("runtime_used_in_score")):
        raise ValueError("runtime cannot enter the robust-step score")


def validate_robuststep_confirmation_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIRMATION_CONFIG_SCHEMA:
        raise ValueError("unexpected robust-step confirmation config")
    if (
        config.get("scientific_status") != "fresh_task_state_score_confirmation"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("historically_untouched_map_claim"))
    ):
        raise ValueError("robust-step confirmation has an invalid evidence boundary")
    if config.get("controller_id") != CONTROLLER_ID:
        raise ValueError("unexpected robust-step confirmation controller id")
    if config.get("score_schema") != "lns2.stride.robust_step_score.v1":
        raise ValueError("unexpected robust-step confirmation score schema")
    indices = tuple(map(int, config.get("trial_indices") or ()))
    first = tuple(map(int, config.get("first_half_indices") or ()))
    second = tuple(map(int, config.get("second_half_indices") or ()))
    if (
        indices != tuple(range(16))
        or first != tuple(range(8))
        or second != tuple(range(8, 16))
    ):
        raise ValueError("score confirmation requires registered 8+8 indices")
    if float(config.get("structure_weight", -1.0)) != 0.02:
        raise ValueError("score confirmation structure weight differs")
    baseline = dict(config.get("baseline_variant") or {})
    selected = dict(config.get("selected_variant") or {})
    if baseline != {
        "id": "mean",
        "mode": "mean",
        "deviation_weight": 0.0,
        "no_progress_penalty": 0.0,
    }:
        raise ValueError("score confirmation baseline was changed")
    if selected != {
        "id": "mean-np100",
        "mode": "mean",
        "deviation_weight": 0.0,
        "no_progress_penalty": 0.1,
    }:
        raise ValueError("score confirmation may not retune the selected score")
    if (
        bool(config.get("runtime_used_in_score"))
        or bool(config.get("future_repair_rounds_used_in_score"))
        or bool(config.get("cost_to_go_used_in_score"))
    ):
        raise ValueError("confirmation score contains a forbidden future/runtime field")
    if not bool(config.get("training_before_confirmation_pass_forbidden")):
        raise ValueError("training must remain forbidden before confirmation passes")
    if int(config.get("expected_feature_dimension", -1)) != FROZEN_FEATURE_DIMENSION:
        raise ValueError("confirmation feature dimension differs from the frozen schema")
    if config.get("expected_feature_schema_id") != FROZEN_FEATURE_SCHEMA_ID:
        raise ValueError("confirmation feature schema differs from the frozen schema")
    expected_maps = list(map(str, config.get("expected_maps") or ()))
    if len(expected_maps) != 6 or len(set(expected_maps)) != 6:
        raise ValueError("score confirmation requires six unique maps")
    grouped = [
        str(map_id)
        for map_ids in dict(config.get("map_groups") or {}).values()
        for map_id in map_ids
    ]
    if sorted(grouped) != sorted(expected_maps) or len(grouped) != len(set(grouped)):
        raise ValueError("score confirmation map groups must partition the six maps")
    contract = dict(config.get("selection_contract") or {})
    if (
        contract.get("source_policies") != ["official_adaptive", "v2-full"]
        or int(contract.get("maximum_source_decision_index", -1)) != 11
        or int(contract.get("maximum_states_per_episode", -1)) != 1
        or not bool(contract.get("result_blind"))
    ):
        raise ValueError("score confirmation selection contract was changed")


def validate_robuststep_v2_headroom_config(config: dict[str, Any]) -> None:
    if config.get("schema") != V2_HEADROOM_CONFIG_SCHEMA:
        raise ValueError("unexpected robust-step V2 headroom config")
    if (
        config.get("scientific_status") != "posthoc_diagnostic_only"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("training_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("diagnostic_result_may_promote_model"))
    ):
        raise ValueError("V2 headroom audit must remain diagnostic-only")
    if config.get("controller_id") != "v2-full":
        raise ValueError("V2 headroom audit requires frozen v2-full")
    if config.get("score_schema") != "lns2.stride.robust_step_score.v1":
        raise ValueError("unexpected V2 headroom score schema")
    indices = tuple(map(int, config.get("trial_indices") or ()))
    first = tuple(map(int, config.get("first_half_indices") or ()))
    second = tuple(map(int, config.get("second_half_indices") or ()))
    if (
        indices != tuple(range(16))
        or first != tuple(range(8))
        or second != tuple(range(8, 16))
    ):
        raise ValueError("V2 headroom audit requires registered 8+8 indices")
    if float(config.get("structure_weight", -1.0)) != 0.02:
        raise ValueError("V2 headroom structure weight differs")
    if dict(config.get("oracle_score") or {}) != {
        "id": "mean",
        "mode": "mean",
        "deviation_weight": 0.0,
        "no_progress_penalty": 0.0,
    }:
        raise ValueError("V2 headroom oracle must remain the plain immediate mean")
    if config.get("stable_state_definition") != (
        "exact_first_half_and_second_half_oracle_winner_agreement"
    ):
        raise ValueError("V2 headroom stable-state definition was changed")
    if (
        bool(config.get("runtime_used_in_oracle"))
        or bool(config.get("future_repair_rounds_used_in_oracle"))
        or bool(config.get("cost_to_go_used_in_oracle"))
    ):
        raise ValueError("V2 headroom oracle contains a forbidden field")
    if int(config.get("expected_feature_dimension", -1)) != FROZEN_FEATURE_DIMENSION:
        raise ValueError("V2 headroom feature dimension differs")
    if config.get("expected_feature_schema_id") != FROZEN_FEATURE_SCHEMA_ID:
        raise ValueError("V2 headroom feature schema differs")


def validate_robuststep_feature_probe_config(config: dict[str, Any]) -> None:
    if config.get("schema") != FEATURE_PROBE_CONFIG_SCHEMA:
        raise ValueError("unexpected robust-step feature-probe config")
    if (
        config.get("scientific_status")
        != "consumed_feature_sufficiency_diagnostic"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("diagnostic_result_may_promote_model"))
        or not bool(config.get("ephemeral_probe_training_allowed"))
    ):
        raise ValueError("feature probe must remain ephemeral and diagnostic-only")
    if (
        config.get("diagnostic_controller_id") != "stride-stepdiag-v1"
        or config.get("frozen_anchor_id") != "v2-full"
    ):
        raise ValueError("unexpected feature-probe controller identity")
    if config.get("score_schema") != "lns2.stride.robust_step_score.v1":
        raise ValueError("unexpected feature-probe score schema")
    indices = tuple(map(int, config.get("trial_indices") or ()))
    first = tuple(map(int, config.get("first_half_indices") or ()))
    second = tuple(map(int, config.get("second_half_indices") or ()))
    if (
        indices != tuple(range(16))
        or first != tuple(range(8))
        or second != tuple(range(8, 16))
    ):
        raise ValueError("feature probe requires registered 8+8 indices")
    if float(config.get("structure_weight", -1.0)) != 0.02:
        raise ValueError("feature-probe structure weight differs")
    if dict(config.get("oracle_score") or {}) != {
        "id": "mean",
        "mode": "mean",
        "deviation_weight": 0.0,
        "no_progress_penalty": 0.0,
    }:
        raise ValueError("feature-probe oracle must remain the plain immediate mean")
    if (
        bool(config.get("runtime_used_in_oracle"))
        or bool(config.get("future_repair_rounds_used_in_oracle"))
        or bool(config.get("cost_to_go_used_in_oracle"))
    ):
        raise ValueError("feature-probe oracle contains a forbidden field")
    if int(config.get("expected_feature_dimension", -1)) != FROZEN_FEATURE_DIMENSION:
        raise ValueError("feature-probe feature dimension differs")
    if config.get("expected_feature_schema_id") != FROZEN_FEATURE_SCHEMA_ID:
        raise ValueError("feature-probe feature schema differs")
    folds = dict(config.get("fold_protocol") or {})
    maps = tuple(map(str, folds.get("held_out_maps") or ()))
    if (
        folds.get("mode") != "leave_one_whole_map_out"
        or int(folds.get("fold_count", -1)) != 6
        or len(maps) != 6
        or len(set(maps)) != 6
        or not bool(folds.get("state_and_candidate_group_integrity"))
    ):
        raise ValueError("feature probe requires six whole-map held-out folds")
    pair_contract = dict(config.get("pair_contract") or {})
    if pair_contract != {
        "inclusion": "same_strict_direction_in_both_eight_seed_halves",
        "direction": "full_sixteen_seed_plain_mean",
        "weighting": "equal_total_weight_per_state",
    }:
        raise ValueError("feature-probe pair contract was changed")
    variants = list(config.get("variants") or ())
    if variants != [
        {"id": "stride-stepdiag-v1/exact-v2-86", "input_profile": "exact_v2_86"},
        {"id": "stride-stepdiag-v1/full-124-delta", "input_profile": "full_124_delta"},
        {"id": "stride-stepdiag-v1/full-124-context", "input_profile": "full_124_context"},
    ]:
        raise ValueError("feature-probe variants were changed")
    if dict(config.get("model_parameters") or {}) != {
        "early_stopping": False,
        "l2_regularization": 0.1,
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "random_state": 20260714,
    }:
        raise ValueError("feature-probe model parameters were changed")


def validate_robuststep_stepgate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != STEPGATE_CONFIG_SCHEMA:
        raise ValueError("unexpected robust-step step-gate config")
    if (
        config.get("scientific_status") != "consumed_nested_abstention_diagnostic"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("diagnostic_result_may_promote_model"))
        or not bool(config.get("ephemeral_probe_training_allowed"))
    ):
        raise ValueError("step gate must remain ephemeral and diagnostic-only")
    if (
        config.get("diagnostic_controller_id") != "stride-stepgate-v1"
        or config.get("frozen_anchor_id") != "v2-full"
        or config.get("challenger_id") != "stride-stepdiag-v1/exact-v2-86"
        or config.get("challenger_input_profile") != "exact_v2_86"
    ):
        raise ValueError("unexpected step-gate controller identity")
    outer = dict(config.get("outer_fold_protocol") or {})
    maps = tuple(map(str, outer.get("held_out_maps") or ()))
    if (
        outer.get("mode") != "leave_one_whole_map_out"
        or int(outer.get("fold_count", -1)) != 6
        or len(maps) != 6
        or len(set(maps)) != 6
        or not bool(outer.get("test_map_outcome_blind"))
    ):
        raise ValueError("step gate requires six outcome-blind outer map folds")
    calibration = dict(config.get("inner_calibration") or {})
    if (
        calibration.get("mode") != "leave_one_whole_training_map_out"
        or list(map(float, calibration.get("candidate_thresholds") or ()))
        != [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        or float(calibration.get("abstain_threshold", -1.0)) != 1.01
        or calibration.get("selection_rule") != "lowest_threshold_passing_all_inner_gates"
        or calibration.get("fallback") != "abstain_all_to_frozen_v2"
    ):
        raise ValueError("step-gate inner calibration was changed")
    if dict(config.get("inner_gates") or {}) != {
        "minimum_overall_regret_improvement": 0.02,
        "minimum_stable_regret_improvement": 0.02,
        "minimum_stable_top3_delta": 0.0,
        "maximum_worst_map_regret_degradation": 0.03,
        "minimum_switch_count": 2,
    }:
        raise ValueError("step-gate inner safety gates were changed")
    if dict(config.get("diagnostic_gates") or {}) != {
        "minimum_stable_state_count": 24,
        "minimum_overall_regret_improvement": 0.02,
        "minimum_stable_regret_improvement": 0.03,
        "minimum_stable_top3_delta": 0.0,
        "minimum_map_regret_win_count": 3,
        "maximum_worst_map_regret_degradation": 0.03,
        "minimum_switch_count": 4,
        "minimum_switch_precision": 0.60,
    }:
        raise ValueError("step-gate diagnostic gates were changed")
    if (
        bool(config.get("runtime_used_in_oracle"))
        or bool(config.get("future_repair_rounds_used_in_oracle"))
        or bool(config.get("cost_to_go_used_in_oracle"))
        or config.get("switch_timing") != "before_pp_repair"
        or config.get("failure_triggered_switching") is not False
    ):
        raise ValueError("step gate violates the current-step evidence boundary")


def robust_pair_winner(
    left_scores: list[float],
    right_scores: list[float],
    left_progress: list[bool],
    right_progress: list[bool],
    *,
    minimum_paired_win_fraction: float,
    maximum_no_progress_disadvantage: float,
) -> int:
    """Return 1 for left, -1 for right, or 0 for an uncertain pair."""

    size = len(left_scores)
    if (
        size == 0
        or len(right_scores) != size
        or len(left_progress) != size
        or len(right_progress) != size
    ):
        raise ValueError("robust pair inputs must be nonempty and paired")
    deltas = [left - right for left, right in zip(left_scores, right_scores)]
    left_wins = sum(delta > 1e-12 for delta in deltas)
    ties = sum(abs(delta) <= 1e-12 for delta in deltas)
    left_fraction = (left_wins + 0.5 * ties) / size
    mean_delta = statistics.fmean(deltas)
    direction = 0
    if left_fraction >= minimum_paired_win_fraction and mean_delta > 1e-12:
        direction = 1
    elif left_fraction <= 1.0 - minimum_paired_win_fraction and mean_delta < -1e-12:
        direction = -1
    if direction == 0:
        return 0
    left_no_progress = 1.0 - statistics.fmean(map(float, left_progress))
    right_no_progress = 1.0 - statistics.fmean(map(float, right_progress))
    if direction > 0:
        return (
            1
            if left_no_progress
            <= right_no_progress + maximum_no_progress_disadvantage + 1e-12
            else 0
        )
    return (
        -1
        if right_no_progress
        <= left_no_progress + maximum_no_progress_disadvantage + 1e-12
        else 0
    )


def _checked_input(project_root: Path, specification: dict[str, Any]) -> Path:
    path = _project_path(project_root, str(specification["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != str(specification["sha256"]).lower():
        raise ValueError(f"robust-step input hash differs: {path}")
    return path


def _load_seed_profiles(
    project_root: Path,
    config: dict[str, Any],
    *,
    state_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, list[Any]]]], dict[str, Any]]:
    expected_indices = set(map(int, config["trial_indices"]))
    state_metadata: dict[str, dict[str, Any]] = {}
    rows_by_state: defaultdict[str, defaultdict[int, dict[str, dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    seen: set[tuple[str, str, int]] = set()
    source_hashes: dict[str, str] = {}
    read_outcomes = 0
    for specification in config["trial_sources"]:
        path = _checked_input(project_root, dict(specification))
        source_hashes[str(specification["path"])] = sha256_file(path)
        for row in _read_jsonl(path):
            if row.get("schema") != STRIDE_TRIAL_SCHEMA:
                raise ValueError(f"unexpected repair-trial schema: {path}")
            state_id = str(row["state_id"])
            if state_ids is not None and state_id not in state_ids:
                continue
            trial_index = int(row["trial_index"])
            if trial_index not in expected_indices:
                continue
            candidate_id = str(row["candidate_id"])
            key = (state_id, candidate_id, trial_index)
            if key in seen:
                raise ValueError(f"duplicate robust-step outcome: {key}")
            seen.add(key)
            structure = row.get("post_structure")
            if not isinstance(structure, dict) or not REQUIRED_POST_STRUCTURE_FIELDS.issubset(structure):
                raise ValueError("robust-step requires complete post structure")
            metadata = {
                "map_id": str(row["map_id"]),
                "split": str(row["split"]),
                "source_policy": str(row["source_policy"]),
                "decision_stage": str(row["decision_stage"]),
                "agent_count": int(row["agent_count"]),
                "before_conflicts": int(row["before_conflicts"]),
            }
            previous = state_metadata.setdefault(state_id, metadata)
            if previous != metadata:
                raise ValueError(f"inconsistent robust-step state metadata: {state_id}")
            rows_by_state[state_id][trial_index][candidate_id] = {
                "candidate_id": candidate_id,
                "pp_seed": int(row["pp_seed"]),
                "conflicts_after": int(row["conflicts_after"]),
                "post_structure": {
                    name: float(structure[name])
                    for name in REQUIRED_POST_STRUCTURE_FIELDS
                },
            }
            read_outcomes += 1

    profiles: dict[str, dict[str, dict[str, list[Any]]]] = {}
    candidate_count = 0
    for state_id in sorted(rows_by_state):
        by_index = rows_by_state[state_id]
        if set(by_index) != expected_indices:
            raise ValueError(f"incomplete robust-step trial indices: {state_id}")
        candidate_sets = [set(by_index[index]) for index in sorted(by_index)]
        if any(candidate_set != candidate_sets[0] for candidate_set in candidate_sets[1:]):
            raise ValueError(f"candidate pool changed across seeds: {state_id}")
        candidates = sorted(candidate_sets[0])
        candidate_count += len(candidates)
        state_profiles = {
            candidate_id: {"scores": [], "progress": []}
            for candidate_id in candidates
        }
        before = int(state_metadata[state_id]["before_conflicts"])
        for trial_index in sorted(expected_indices):
            outcomes = by_index[trial_index]
            seeds = {int(row["pp_seed"]) for row in outcomes.values()}
            if len(seeds) != 1:
                raise ValueError(f"PP seeds are not paired within state/index: {state_id}/{trial_index}")
            scored = [
                {
                    "candidate_id": candidate_id,
                    "mean_post_structure": outcomes[candidate_id]["post_structure"],
                }
                for candidate_id in candidates
            ]
            assign_structure_scores(scored)
            structural = {
                str(row["candidate_id"]): float(row["structural_score"])
                for row in scored
            }
            for candidate_id in candidates:
                after = int(outcomes[candidate_id]["conflicts_after"])
                score = (before - after) / max(1, before) - float(
                    config["structure_weight"]
                ) * structural[candidate_id]
                state_profiles[candidate_id]["scores"].append(score)
                state_profiles[candidate_id]["progress"].append(after < before)
        profiles[state_id] = state_profiles
    integrity = {
        "state_count": len(profiles),
        "map_count": len({row["map_id"] for row in state_metadata.values()}),
        "candidate_count": candidate_count,
        "outcome_count": read_outcomes,
        "trial_source_sha256": source_hashes,
    }
    return state_metadata, profiles, integrity


def _variant_winners(
    candidates: dict[str, dict[str, list[Any]]],
    indices: list[int],
    variant: dict[str, Any],
) -> dict[tuple[str, str], int]:
    winners: dict[tuple[str, str], int] = {}
    for left_id, right_id in combinations(sorted(candidates), 2):
        left = candidates[left_id]
        right = candidates[right_id]
        direction = robust_pair_winner(
            [float(left["scores"][index]) for index in indices],
            [float(right["scores"][index]) for index in indices],
            [bool(left["progress"][index]) for index in indices],
            [bool(right["progress"][index]) for index in indices],
            minimum_paired_win_fraction=float(
                variant["minimum_paired_win_fraction"]
            ),
            maximum_no_progress_disadvantage=float(
                variant["maximum_no_progress_disadvantage"]
            ),
        )
        if direction:
            winners[(left_id, right_id)] = direction
    return winners


def _robust_good_set(
    candidate_ids: list[str], winners: dict[tuple[str, str], int]
) -> set[str]:
    losses = Counter({candidate_id: 0 for candidate_id in candidate_ids})
    for (left_id, right_id), direction in winners.items():
        losses[right_id if direction > 0 else left_id] += 1
    minimum = min(losses.values(), default=0)
    return {candidate_id for candidate_id, count in losses.items() if count == minimum}


def evaluate_robuststep_variant(
    profiles: dict[str, dict[str, dict[str, list[Any]]]],
    variant: dict[str, Any],
    first_indices: list[int],
    second_indices: list[int],
) -> dict[str, Any]:
    union_half_pairs = 0
    agreeing_half_pairs = 0
    full_pairs = 0
    all_pairs = 0
    five_pair_states = 0
    unique_winner_states = 0
    good_set_jaccards: list[float] = []
    full_good_set_sizes: list[int] = []
    state_rows = []
    full_indices = first_indices + second_indices
    for state_id, candidates in sorted(profiles.items()):
        candidate_ids = sorted(candidates)
        first = _variant_winners(candidates, first_indices, variant)
        second = _variant_winners(candidates, second_indices, variant)
        full = _variant_winners(candidates, full_indices, variant)
        pair_union = set(first) | set(second)
        pair_agreement = sum(
            pair in first and pair in second and first[pair] == second[pair]
            for pair in pair_union
        )
        first_good = _robust_good_set(candidate_ids, first)
        second_good = _robust_good_set(candidate_ids, second)
        full_good = _robust_good_set(candidate_ids, full)
        good_union = first_good | second_good
        jaccard = len(first_good & second_good) / len(good_union) if good_union else 1.0
        total = len(candidate_ids) * (len(candidate_ids) - 1) // 2
        union_half_pairs += len(pair_union)
        agreeing_half_pairs += pair_agreement
        full_pairs += len(full)
        all_pairs += total
        five_pair_states += len(full) >= 5
        unique_winner_states += len(full_good) == 1
        good_set_jaccards.append(jaccard)
        full_good_set_sizes.append(len(full_good))
        state_rows.append(
            {
                "state_id": state_id,
                "candidate_count": len(candidate_ids),
                "first_half_pair_count": len(first),
                "second_half_pair_count": len(second),
                "full_pair_count": len(full),
                "half_pair_agreement_count": pair_agreement,
                "half_pair_union_count": len(pair_union),
                "good_set_jaccard": jaccard,
                "full_good_set_size": len(full_good),
            }
        )
    state_count = len(profiles)
    return {
        "id": str(variant["id"]),
        "minimum_paired_win_fraction": float(
            variant["minimum_paired_win_fraction"]
        ),
        "maximum_no_progress_disadvantage": float(
            variant["maximum_no_progress_disadvantage"]
        ),
        "half_pair_union_count": union_half_pairs,
        "half_pair_agreement_count": agreeing_half_pairs,
        "half_pairwise_consistency": (
            agreeing_half_pairs / union_half_pairs if union_half_pairs else 0.0
        ),
        "full_pair_count": full_pairs,
        "all_unordered_pair_count": all_pairs,
        "full_pair_coverage": full_pairs / all_pairs if all_pairs else 0.0,
        "state_five_pair_coverage": five_pair_states / state_count if state_count else 0.0,
        "mean_good_set_jaccard": (
            statistics.fmean(good_set_jaccards) if good_set_jaccards else 0.0
        ),
        "mean_full_good_set_size": (
            statistics.fmean(full_good_set_sizes) if full_good_set_sizes else 0.0
        ),
        "unique_robust_winner_rate": (
            unique_winner_states / state_count if state_count else 0.0
        ),
        "action_uncertainty_rate": (
            1.0 - unique_winner_states / state_count if state_count else 1.0
        ),
        "states": state_rows,
    }


def run_robuststep_design(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robuststep_config(config)
    audit_path = _checked_input(project_root, dict(config["stage3_audit"]))
    audit = _read_json(audit_path)
    formal_path = _checked_input(project_root, dict(config["formal_ood_config"]))
    formal = _read_json(formal_path)
    formal_maps = {str(row["benchmark_id"]) for row in formal.get("cases", [])}
    metadata, profiles, integrity = _load_seed_profiles(project_root, config)
    observed_maps = {str(row["map_id"]) for row in metadata.values()}
    formal_overlap = sorted(observed_maps & formal_maps)
    integrity_gates = {
        "stage3_audit_passed": bool(audit.get("passed")),
        "state_count": integrity["state_count"] == int(config["expected_state_count"]),
        "map_count": integrity["map_count"] == int(config["expected_map_count"]),
        "candidate_count": integrity["candidate_count"] == int(config["expected_candidate_count"]),
        "outcome_count": integrity["outcome_count"] == int(config["expected_outcome_count"]),
        "formal_ood_overlap_zero": not formal_overlap,
    }
    thresholds = dict(config["design_gates"])
    variants = []
    for specification in config["variants"]:
        result = evaluate_robuststep_variant(
            profiles,
            dict(specification),
            list(map(int, config["first_half_indices"])),
            list(map(int, config["second_half_indices"])),
        )
        result["gates"] = {
            "half_pairwise_consistency": result["half_pairwise_consistency"]
            >= float(thresholds["minimum_half_pairwise_consistency"]),
            "mean_good_set_jaccard": result["mean_good_set_jaccard"]
            >= float(thresholds["minimum_mean_good_set_jaccard"]),
            "full_pair_coverage": result["full_pair_coverage"]
            >= float(thresholds["minimum_full_pair_coverage"]),
            "state_five_pair_coverage": result["state_five_pair_coverage"]
            >= float(thresholds["minimum_state_five_pair_coverage"]),
        }
        result["passed"] = all(result["gates"].values())
        variants.append(result)
    eligible = [row for row in variants if row["passed"]]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["full_pair_coverage"]),
            float(row["half_pairwise_consistency"]),
            float(row["mean_good_set_jaccard"]),
            str(row["id"]),
        ),
        default=None,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "consumed_design_only",
        "formal_speed_claim": False,
        "controller_id": CONTROLLER_ID,
        "label_schema": LABEL_SCHEMA,
        "runtime_used_in_label": False,
        "fresh_confirmation_required": True,
        "integrity": integrity,
        "integrity_gates": integrity_gates,
        "formal_ood_overlap": formal_overlap,
        "variants": variants,
        "selected_variant_id": str(selected["id"]) if selected else None,
        "design_passed": all(integrity_gates.values()) and selected is not None,
        "next_decision": (
            "freeze_selected_variant_and_collect_fresh_confirmation"
            if all(integrity_gates.values()) and selected is not None
            else "revise_robust_pair_contract_before_fresh_collection"
        ),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "stage3_audit_sha256": sha256_file(audit_path),
            "formal_ood_config_sha256": sha256_file(formal_path),
        },
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "robuststep_design_report.json", report)
    return report


def _apply_variant_gates(
    result: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, Any]:
    result["gates"] = {
        "half_pairwise_consistency": result["half_pairwise_consistency"]
        >= float(thresholds["minimum_half_pairwise_consistency"]),
        "mean_good_set_jaccard": result["mean_good_set_jaccard"]
        >= float(thresholds["minimum_mean_good_set_jaccard"]),
        "full_pair_coverage": result["full_pair_coverage"]
        >= float(thresholds["minimum_full_pair_coverage"]),
        "state_five_pair_coverage": result["state_five_pair_coverage"]
        >= float(thresholds["minimum_state_five_pair_coverage"]),
    }
    result["passed"] = all(result["gates"].values())
    return result


def run_robuststep_seed_depth(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    """Test whether 16 paired seeds stabilize the consumed robust-step contract."""

    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robuststep_seed_depth_config(config)
    formal_path = _checked_input(project_root, dict(config["formal_ood_config"]))
    formal = _read_json(formal_path)
    formal_maps = {str(row["benchmark_id"]) for row in formal.get("cases", [])}
    cohort_state_ids: dict[str, set[str]] = {}
    cohort_integrity: dict[str, Any] = {}
    assigned: set[str] = set()
    trial_sources = {
        str(row["path"]): dict(row) for row in config["trial_sources"]
    }
    for specification in config["cohorts"]:
        cohort_id = str(specification["id"])
        source_name = str(specification["membership_source"])
        if source_name not in trial_sources:
            raise ValueError("cohort membership source is not hash-registered")
        source_path = _checked_input(project_root, trial_sources[source_name])
        state_ids = {str(row["state_id"]) for row in _read_jsonl(source_path)}
        if assigned & state_ids:
            raise ValueError("seed-depth cohorts overlap")
        assigned.update(state_ids)
        cohort_state_ids[cohort_id] = state_ids
        cohort_integrity[cohort_id] = {
            "state_count": len(state_ids),
            "expected_state_count": int(specification["expected_state_count"]),
            "passed": len(state_ids) == int(specification["expected_state_count"]),
        }
    metadata, profiles, integrity = _load_seed_profiles(
        project_root, config, state_ids=assigned
    )
    missing = assigned - set(profiles)
    if missing:
        raise ValueError("seed-depth cohort has missing trial profiles")
    observed_maps = {str(row["map_id"]) for row in metadata.values()}
    formal_overlap = sorted(observed_maps & formal_maps)
    cohort_profiles = {
        cohort_id: {
            state_id: profiles[state_id] for state_id in sorted(state_ids)
        }
        for cohort_id, state_ids in cohort_state_ids.items()
    }
    integrity_gates = {
        "state_count": integrity["state_count"] == int(config["expected_state_count"]),
        "map_count": integrity["map_count"] == int(config["expected_map_count"]),
        "candidate_count": integrity["candidate_count"]
        == int(config["expected_candidate_count"]),
        "outcome_count": integrity["outcome_count"]
        == int(config["expected_outcome_count"]),
        "cohort_partition_complete": assigned == set(profiles),
        "cohort_integrity": all(row["passed"] for row in cohort_integrity.values()),
        "formal_ood_overlap_zero": not formal_overlap,
    }
    thresholds = dict(config["diagnostic_gates"])
    first = list(map(int, config["first_half_indices"]))
    second = list(map(int, config["second_half_indices"]))
    variants = []
    for specification in config["variants"]:
        result = _apply_variant_gates(
            evaluate_robuststep_variant(profiles, dict(specification), first, second),
            thresholds,
        )
        result["cohorts"] = {}
        for cohort_id, subset in cohort_profiles.items():
            result["cohorts"][cohort_id] = _apply_variant_gates(
                evaluate_robuststep_variant(
                    subset, dict(specification), first, second
                ),
                thresholds,
            )
        result["all_cohorts_passed"] = all(
            row["passed"] for row in result["cohorts"].values()
        )
        result["passed"] = bool(result["passed"] and result["all_cohorts_passed"])
        variants.append(result)
    eligible = [row for row in variants if row["passed"]]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["full_pair_coverage"]),
            float(row["half_pairwise_consistency"]),
            float(row["mean_good_set_jaccard"]),
            str(row["id"]),
        ),
        default=None,
    )
    passed = all(integrity_gates.values()) and selected is not None
    report = {
        "schema": SEED_DEPTH_REPORT_SCHEMA,
        "scientific_status": "consumed_seed_depth_diagnostic",
        "formal_speed_claim": False,
        "controller_id": CONTROLLER_ID,
        "label_schema": LABEL_SCHEMA,
        "runtime_used_in_label": False,
        "fresh_confirmation_required": True,
        "integrity": integrity,
        "cohort_integrity": cohort_integrity,
        "integrity_gates": integrity_gates,
        "formal_ood_overlap": formal_overlap,
        "variants": variants,
        "selected_variant_id": str(selected["id"]) if selected else None,
        "seed_depth_passed": passed,
        "next_decision": (
            "collect_fresh_sixteen_seed_confirmation_before_training"
            if passed
            else "revise_robust_pair_contract_before_new_collection"
        ),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "formal_ood_config_sha256": sha256_file(formal_path),
        },
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "robuststep_seed_depth_report.json", report)
    return report


def _lower_quartile(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("score aggregation requires outcomes")
    position = 0.25 * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _aggregate_candidate_score(
    profile: dict[str, list[Any]], indices: list[int], variant: dict[str, Any]
) -> float:
    values = [float(profile["scores"][index]) for index in indices]
    progress = [bool(profile["progress"][index]) for index in indices]
    if str(variant["mode"]) == "lower_quartile":
        return _lower_quartile(values)
    return (
        statistics.fmean(values)
        - float(variant["deviation_weight"]) * statistics.pstdev(values)
        - float(variant["no_progress_penalty"])
        * (1.0 - statistics.fmean(map(float, progress)))
    )


def evaluate_robuststep_score_variant(
    profiles: dict[str, dict[str, dict[str, list[Any]]]],
    variant: dict[str, Any],
    first_indices: list[int],
    second_indices: list[int],
) -> dict[str, Any]:
    pair_union = 0
    pair_agreement = 0
    top3_overlaps = []
    normalized_regrets = []
    winner_agreements = 0
    state_rows = []
    for state_id, candidates in sorted(profiles.items()):
        candidate_ids = sorted(candidates)
        halves = []
        for indices in (first_indices, second_indices):
            scores = {
                candidate_id: _aggregate_candidate_score(
                    candidates[candidate_id], indices, variant
                )
                for candidate_id in candidate_ids
            }
            ranking = sorted(candidate_ids, key=lambda value: (-scores[value], value))
            halves.append((scores, ranking))
        first_scores, first_rank = halves[0]
        second_scores, second_rank = halves[1]
        state_pairs = 0
        state_agreements = 0
        for left_id, right_id in combinations(candidate_ids, 2):
            first_delta = first_scores[left_id] - first_scores[right_id]
            second_delta = second_scores[left_id] - second_scores[right_id]
            if abs(first_delta) <= 1e-12 and abs(second_delta) <= 1e-12:
                continue
            state_pairs += 1
            state_agreements += int(first_delta * second_delta > 1e-24)
        top3 = len(set(first_rank[:3]) & set(second_rank[:3])) / 3.0
        state_regrets = []
        for selected, evaluation in (
            (first_rank[0], second_scores),
            (second_rank[0], first_scores),
        ):
            values = list(evaluation.values())
            span = max(values) - min(values)
            regret = max(values) - evaluation[selected]
            state_regrets.append(regret / span if span > 1e-12 else 0.0)
        pair_union += state_pairs
        pair_agreement += state_agreements
        top3_overlaps.append(top3)
        normalized_regrets.extend(state_regrets)
        winner_agreements += first_rank[0] == second_rank[0]
        state_rows.append(
            {
                "state_id": state_id,
                "candidate_count": len(candidate_ids),
                "pair_count": state_pairs,
                "pair_agreement_count": state_agreements,
                "pairwise_consistency": (
                    state_agreements / state_pairs if state_pairs else 1.0
                ),
                "top3_overlap": top3,
                "symmetric_normalized_regret": statistics.fmean(state_regrets),
                "winner_agreement": first_rank[0] == second_rank[0],
            }
        )
    state_count = len(profiles)
    return {
        "id": str(variant["id"]),
        "mode": str(variant["mode"]),
        "deviation_weight": float(variant["deviation_weight"]),
        "no_progress_penalty": float(variant["no_progress_penalty"]),
        "pair_count": pair_union,
        "pair_agreement_count": pair_agreement,
        "pairwise_consistency": pair_agreement / pair_union if pair_union else 0.0,
        "mean_top3_overlap": statistics.fmean(top3_overlaps) if top3_overlaps else 0.0,
        "mean_cross_half_normalized_regret": (
            statistics.fmean(normalized_regrets) if normalized_regrets else 0.0
        ),
        "winner_agreement_rate": winner_agreements / state_count if state_count else 0.0,
        "states": state_rows,
    }


def _score_gates(
    result: dict[str, Any], thresholds: dict[str, Any]
) -> dict[str, bool]:
    return {
        "pairwise_consistency": result["pairwise_consistency"]
        >= float(thresholds["minimum_pairwise_consistency"]),
        "mean_top3_overlap": result["mean_top3_overlap"]
        >= float(thresholds["minimum_mean_top3_overlap"]),
        "mean_cross_half_normalized_regret": result[
            "mean_cross_half_normalized_regret"
        ]
        <= float(thresholds["maximum_mean_cross_half_normalized_regret"]),
    }


def run_robuststep_score_design(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robuststep_score_config(config)
    formal_path = _checked_input(project_root, dict(config["formal_ood_config"]))
    formal = _read_json(formal_path)
    formal_maps = {str(row["benchmark_id"]) for row in formal.get("cases", [])}
    trial_sources = {
        str(row["path"]): dict(row) for row in config["trial_sources"]
    }
    cohort_state_ids: dict[str, set[str]] = {}
    assigned: set[str] = set()
    cohort_integrity = {}
    for specification in config["cohorts"]:
        cohort_id = str(specification["id"])
        source_name = str(specification["membership_source"])
        if source_name not in trial_sources:
            raise ValueError("score cohort membership source is not registered")
        source_path = _checked_input(project_root, trial_sources[source_name])
        state_ids = {str(row["state_id"]) for row in _read_jsonl(source_path)}
        if assigned & state_ids:
            raise ValueError("score design cohorts overlap")
        assigned.update(state_ids)
        cohort_state_ids[cohort_id] = state_ids
        expected = int(specification["expected_state_count"])
        cohort_integrity[cohort_id] = {
            "state_count": len(state_ids),
            "expected_state_count": expected,
            "passed": len(state_ids) == expected,
        }
    metadata, profiles, integrity = _load_seed_profiles(
        project_root, config, state_ids=assigned
    )
    cohort_profiles = {
        cohort_id: {
            state_id: profiles[state_id] for state_id in sorted(state_ids)
        }
        for cohort_id, state_ids in cohort_state_ids.items()
    }
    formal_overlap = sorted(
        {str(row["map_id"]) for row in metadata.values()} & formal_maps
    )
    integrity_gates = {
        "state_count": integrity["state_count"] == int(config["expected_state_count"]),
        "map_count": integrity["map_count"] == int(config["expected_map_count"]),
        "candidate_count": integrity["candidate_count"]
        == int(config["expected_candidate_count"]),
        "outcome_count": integrity["outcome_count"]
        == int(config["expected_outcome_count"]),
        "cohort_partition_complete": assigned == set(profiles),
        "cohort_integrity": all(row["passed"] for row in cohort_integrity.values()),
        "formal_ood_overlap_zero": not formal_overlap,
    }
    first = list(map(int, config["first_half_indices"]))
    second = list(map(int, config["second_half_indices"]))
    variants = []
    for specification in config["variants"]:
        result = evaluate_robuststep_score_variant(
            profiles, dict(specification), first, second
        )
        result["gates"] = _score_gates(result, dict(config["overall_gates"]))
        result["cohorts"] = {}
        for cohort_id, subset in cohort_profiles.items():
            cohort_result = evaluate_robuststep_score_variant(
                subset, dict(specification), first, second
            )
            cohort_result["gates"] = _score_gates(
                cohort_result, dict(config["cohort_gates"])
            )
            cohort_result["passed"] = all(cohort_result["gates"].values())
            result["cohorts"][cohort_id] = cohort_result
        result["stability_passed"] = all(result["gates"].values()) and all(
            row["passed"] for row in result["cohorts"].values()
        )
        variants.append(result)
    baseline = next(
        row for row in variants if row["id"] == str(config["baseline_variant_id"])
    )
    for result in variants:
        improvement = float(baseline["mean_cross_half_normalized_regret"]) - float(
            result["mean_cross_half_normalized_regret"]
        )
        result["regret_improvement_over_mean"] = improvement
        result["risk_aware"] = result["id"] != baseline["id"]
        result["improvement_gate"] = (
            result["risk_aware"]
            and improvement >= float(config["minimum_regret_improvement_over_mean"])
        )
        result["passed"] = bool(
            result["stability_passed"] and result["improvement_gate"]
        )
    eligible = [row for row in variants if row["passed"]]
    selected = min(
        eligible,
        key=lambda row: (
            float(row["mean_cross_half_normalized_regret"]),
            -float(row["mean_top3_overlap"]),
            -float(row["pairwise_consistency"]),
            str(row["id"]),
        ),
        default=None,
    )
    passed = all(integrity_gates.values()) and selected is not None
    report = {
        "schema": SCORE_REPORT_SCHEMA,
        "scientific_status": "consumed_score_design_only",
        "formal_speed_claim": False,
        "controller_id": CONTROLLER_ID,
        "score_schema": str(config["score_schema"]),
        "runtime_used_in_score": False,
        "fresh_confirmation_required": True,
        "integrity": integrity,
        "cohort_integrity": cohort_integrity,
        "integrity_gates": integrity_gates,
        "formal_ood_overlap": formal_overlap,
        "variants": variants,
        "selected_variant_id": str(selected["id"]) if selected else None,
        "score_design_passed": passed,
        "next_decision": (
            "freeze_risk_score_and_collect_fresh_confirmation"
            if passed
            else "retain_v2_actions_and_design_uncertainty_abstention"
        ),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "formal_ood_config_sha256": sha256_file(formal_path),
        },
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "robuststep_score_design_report.json", report)
    return report


def _confirmation_local_path(project_root: Path, value: str) -> Path:
    """Resolve project artifacts recorded by either the Windows or WSL host."""

    raw = Path(value)
    if raw.exists():
        return raw.resolve()
    normalized = str(value).replace("\\", "/")
    marker = f"/{project_root.name}/"
    if marker in normalized:
        relative = normalized.rsplit(marker, 1)[1]
        candidate = project_root / Path(relative)
        if candidate.exists():
            return candidate.resolve()
    return _project_path(project_root, value)


def _confirmation_path(project_root: Path, value: str) -> Path:
    path = _confirmation_local_path(project_root, value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _confirmation_feature_integrity(
    paths: list[Path], state_ids: set[str], trial_indices: set[int]
) -> dict[str, Any]:
    observed: dict[tuple[str, str], dict[str, float]] = {}
    schema_ids: set[str] = set()
    dimensions: Counter[int] = Counter()
    rows_read = 0
    for path in paths:
        for row in _read_jsonl(path):
            state_id = str(row.get("state_id", ""))
            if state_id not in state_ids or int(row.get("trial_index", -1)) not in trial_indices:
                continue
            features = row.get("features")
            if not isinstance(features, dict):
                raise ValueError("score confirmation requires candidate feature dictionaries")
            normalized = {str(name): float(value) for name, value in features.items()}
            key = (state_id, str(row["candidate_id"]))
            previous = observed.setdefault(key, normalized)
            if previous != normalized:
                raise ValueError(f"candidate features changed across PP seeds: {key}")
            schema_ids.add(str(row.get("feature_schema_id", "")))
            dimensions[len(normalized)] += 1
            rows_read += 1
    return {
        "candidate_feature_count": len(observed),
        "rows_read": rows_read,
        "feature_schema_ids": sorted(schema_ids),
        "feature_dimension_counts": {
            str(key): value for key, value in sorted(dimensions.items())
        },
    }


def run_robuststep_confirmation(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robuststep_confirmation_config(config)

    design_path = _checked_input(project_root, dict(config["score_design_report"]))
    design = _read_json(design_path)
    if (
        design.get("score_design_passed") is not True
        or design.get("selected_variant_id") != "mean-np100"
    ):
        raise ValueError("fresh confirmation is not backed by the frozen score design")
    formal_path = _checked_input(project_root, dict(config["formal_ood_config"]))
    dataset_config_path = _checked_input(project_root, dict(config["dataset_config"]))
    source_config_path = _checked_input(project_root, dict(config["source_config"]))
    formal = _read_json(formal_path)
    dataset_config = _read_json(dataset_config_path)
    source_config = _read_json(source_config_path)

    selection_path = _confirmation_path(project_root, str(config["selection_path"]))
    trial_paths = [
        _confirmation_path(project_root, str(value))
        for value in config["trial_sources"]
    ]
    selection = _read_jsonl(selection_path)
    state_ids = [str(row["state_id"]) for row in selection]
    selected_ids = set(state_ids)
    if len(state_ids) != len(selected_ids):
        raise ValueError("score confirmation selection repeats state IDs")

    loader_config = {
        "trial_indices": list(config["trial_indices"]),
        "structure_weight": float(config["structure_weight"]),
        "trial_sources": [
            {"path": str(path), "sha256": sha256_file(path)} for path in trial_paths
        ],
    }
    metadata, profiles, integrity = _load_seed_profiles(
        project_root, loader_config, state_ids=selected_ids
    )
    trial_indices = set(map(int, config["trial_indices"]))
    feature_integrity = _confirmation_feature_integrity(
        trial_paths, selected_ids, trial_indices
    )

    expected_maps = set(map(str, config["expected_maps"]))
    current_maps = {str(row["map_id"]) for row in metadata.values()}
    old_label_overlap = sorted(current_maps & set(map(str, config["old_label_map_ids"])))
    formal_maps = {str(row["benchmark_id"]) for row in formal.get("cases", [])}
    formal_overlap = sorted(current_maps & formal_maps)
    policy_counts = Counter(str(row["source_policy"]) for row in selection)
    episode_counts = Counter(
        (str(row["source_policy"]), str(row["episode_id"])) for row in selection
    )
    selection_forbidden_fields = sorted(
        {
            key
            for row in selection
            for key in (
                "actual_action",
                "after_fingerprint",
                "repair_seconds",
                "repair_state_changed",
            )
            if key in row
        }
    )
    source_roots = {
        _confirmation_local_path(project_root, str(row["source_root"]))
        for row in selection
    }
    dataset_root = _confirmation_local_path(
        project_root, str(config["dataset_root"])
    ).resolve()
    source_dataset_roots = set()
    for root in source_roots:
        run = _read_json(root / "run_config.json")
        source_dataset_roots.add(
            _confirmation_local_path(project_root, str(run["dataset"]))
        )
    dataset_manifest_path = dataset_root / str(source_config["split"]) / "manifest.jsonl"
    if not dataset_manifest_path.is_file():
        raise FileNotFoundError(dataset_manifest_path)
    dataset_rows = _read_jsonl(dataset_manifest_path)
    dataset_task_ids = {str(row["task_id"]) for row in dataset_rows}
    selected_task_ids = {str(row["task_id"]) for row in selection}
    registered_task_seeds = list(map(int, dataset_config.get("task_seeds") or ()))

    expected_policy_count = int(config["expected_states_per_source_policy"])
    expected_dimension = int(config["expected_feature_dimension"])
    expected_schema = str(config["expected_feature_schema_id"])
    integrity_gates = {
        "state_count": len(profiles) == int(config["expected_state_count"]),
        "selection_profile_identity": set(profiles) == selected_ids,
        "map_set": current_maps == expected_maps,
        "candidate_floor": integrity["candidate_count"]
        >= int(config["minimum_candidate_count"]),
        "outcome_cartesian_complete": integrity["outcome_count"]
        == integrity["candidate_count"] * len(trial_indices),
        "candidate_feature_complete": feature_integrity["candidate_feature_count"]
        == integrity["candidate_count"],
        "feature_schema": feature_integrity["feature_schema_ids"] == [expected_schema],
        "feature_dimension": feature_integrity["feature_dimension_counts"]
        == {str(expected_dimension): integrity["outcome_count"]},
        "source_policy_balance": policy_counts
        == Counter({"official_adaptive": expected_policy_count, "v2-full": expected_policy_count}),
        "one_state_per_source_episode": max(episode_counts.values(), default=0) <= 1,
        "source_decision_cap": all(int(row["decision_index"]) <= 11 for row in selection),
        "result_blind_selection": not selection_forbidden_fields,
        "fresh_task_seeds": registered_task_seeds
        == list(map(int, config["fresh_task_seeds"])),
        "source_solver_seeds": list(map(int, source_config["solver_seeds"]))
        == list(map(int, config["source_solver_seeds"])),
        "selection_tasks_registered": selected_task_ids <= dataset_task_ids,
        "source_dataset_identity": source_dataset_roots == {dataset_root},
        "old_label_map_overlap_zero": not old_label_overlap,
        "formal_ood_overlap_zero": not formal_overlap,
    }

    first = list(map(int, config["first_half_indices"]))
    second = list(map(int, config["second_half_indices"]))
    baseline = evaluate_robuststep_score_variant(
        profiles, dict(config["baseline_variant"]), first, second
    )
    selected = evaluate_robuststep_score_variant(
        profiles, dict(config["selected_variant"]), first, second
    )
    overall_thresholds = dict(config["overall_gates"])
    selected["gates"] = _score_gates(selected, overall_thresholds)
    regret_improvement = float(baseline["mean_cross_half_normalized_regret"]) - float(
        selected["mean_cross_half_normalized_regret"]
    )
    selected["regret_improvement_over_mean"] = regret_improvement
    selected["gates"]["regret_improvement_over_mean"] = regret_improvement >= float(
        overall_thresholds["minimum_regret_improvement_over_mean"]
    )
    selected["passed"] = all(selected["gates"].values())

    subgroup_thresholds = dict(config["subgroup_gates"])
    subgroup_specs: dict[str, set[str]] = {
        f"source_policy:{policy}": {
            state_id
            for state_id, row in metadata.items()
            if str(row["source_policy"]) == policy
        }
        for policy in ("official_adaptive", "v2-full")
    }
    subgroup_specs.update(
        {
            f"map_group:{group_id}": {
                state_id
                for state_id, row in metadata.items()
                if str(row["map_id"]) in set(map(str, map_ids))
            }
            for group_id, map_ids in dict(config["map_groups"]).items()
        }
    )
    subgroups = {}
    for subgroup_id, subgroup_state_ids in sorted(subgroup_specs.items()):
        subset = {state_id: profiles[state_id] for state_id in sorted(subgroup_state_ids)}
        result = evaluate_robuststep_score_variant(
            subset, dict(config["selected_variant"]), first, second
        )
        result["state_count"] = len(subset)
        result["gates"] = {
            "minimum_state_count": len(subset)
            >= int(subgroup_thresholds["minimum_state_count"]),
            **_score_gates(result, subgroup_thresholds),
        }
        result["passed"] = all(result["gates"].values())
        subgroups[subgroup_id] = result

    confirmation_passed = bool(
        all(integrity_gates.values())
        and selected["passed"]
        and all(row["passed"] for row in subgroups.values())
    )
    report = {
        "schema": CONFIRMATION_REPORT_SCHEMA,
        "scientific_status": "fresh_task_state_score_confirmation",
        "formal_speed_claim": False,
        "historically_untouched_map_claim": False,
        "controller_id": CONTROLLER_ID,
        "score_schema": str(config["score_schema"]),
        "runtime_used_in_score": False,
        "training_was_allowed_before_analysis": False,
        "integrity": integrity,
        "feature_integrity": feature_integrity,
        "integrity_gates": integrity_gates,
        "current_maps": sorted(current_maps),
        "old_label_map_overlap": old_label_overlap,
        "formal_ood_overlap": formal_overlap,
        "selection_forbidden_fields": selection_forbidden_fields,
        "baseline": baseline,
        "selected": selected,
        "subgroups": subgroups,
        "confirmation_passed": confirmation_passed,
        "next_decision": (
            "freeze_mean_np100_and_start_small_balanced_pilot"
            if confirmation_passed
            else "retain_v2_actions_and_do_not_train_robuststep"
        ),
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "score_design_report_sha256": sha256_file(design_path),
            "dataset_config_sha256": sha256_file(dataset_config_path),
            "source_config_sha256": sha256_file(source_config_path),
            "selection_sha256": sha256_file(selection_path),
            "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
            "trial_source_sha256": {
                str(path): sha256_file(path) for path in trial_paths
            },
        },
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "robuststep_confirmation_report.json", report)
    return report


def _confirmation_candidate_features(
    paths: list[Path], state_ids: set[str], trial_indices: set[int]
) -> dict[str, dict[str, dict[str, float]]]:
    result: defaultdict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for path in paths:
        for row in _read_jsonl(path):
            state_id = str(row.get("state_id", ""))
            if state_id not in state_ids or int(row.get("trial_index", -1)) not in trial_indices:
                continue
            candidate_id = str(row["candidate_id"])
            features = {str(name): float(value) for name, value in row["features"].items()}
            previous = result[state_id].setdefault(candidate_id, features)
            if previous != features:
                raise ValueError("V2 headroom candidate features changed across seeds")
    return {state_id: dict(rows) for state_id, rows in result.items()}


def _normalized_selection_regret(
    scores: dict[str, float], selected_id: str
) -> float:
    values = list(scores.values())
    span = max(values) - min(values)
    regret = max(values) - float(scores[selected_id])
    return regret / span if span > 1e-12 else 0.0


def _headroom_summary(
    records: list[dict[str, Any]], meaningful_regret: float
) -> dict[str, Any]:
    if not records:
        return {
            "state_count": 0,
            "v2_exact_best_rate": None,
            "v2_in_oracle_top3_rate": None,
            "oracle_in_v2_top3_rate": None,
            "mean_v2_normalized_regret": None,
            "meaningful_headroom_rate": None,
        }
    mean = lambda name: statistics.fmean(float(row[name]) for row in records)
    return {
        "state_count": len(records),
        "v2_exact_best_rate": mean("v2_exact_best"),
        "v2_in_oracle_top3_rate": mean("v2_in_oracle_top3"),
        "oracle_in_v2_top3_rate": mean("oracle_in_v2_top3"),
        "mean_v2_normalized_regret": mean("v2_normalized_regret"),
        "mean_v2_half_normalized_regret": mean("v2_half_normalized_regret"),
        "meaningful_headroom_rate": statistics.fmean(
            float(row["v2_normalized_regret"] >= meaningful_regret)
            for row in records
        ),
        "oracle_half_winner_agreement_rate": mean("oracle_half_winner_agreement"),
        "mean_oracle_half_top3_overlap": mean("oracle_half_top3_overlap"),
        "mean_v2_margin": mean("v2_margin"),
        "mean_selected_feature_outside_fraction": mean(
            "selected_feature_outside_fraction"
        ),
    }


def run_robuststep_v2_headroom(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robuststep_v2_headroom_config(config)
    confirmation_path = _checked_input(
        project_root, dict(config["confirmation_report"])
    )
    confirmation = _read_json(confirmation_path)
    if confirmation.get("confirmation_passed") is not False:
        raise ValueError("V2 headroom diagnostic requires the failed fresh confirmation")
    selection_path = _checked_input(project_root, dict(config["selection"]))
    bundle_manifest_path = _checked_input(
        project_root, dict(config["controller_bundle"])
    )
    trial_paths = [
        _checked_input(project_root, dict(specification))
        for specification in config["trial_sources"]
    ]
    selection = _read_jsonl(selection_path)
    state_ids = {str(row["state_id"]) for row in selection}
    selection_by_id = {str(row["state_id"]): row for row in selection}
    if len(state_ids) != len(selection):
        raise ValueError("V2 headroom selection repeats state IDs")
    metadata, profiles, integrity = _load_seed_profiles(
        project_root, config, state_ids=state_ids
    )
    trial_indices = set(map(int, config["trial_indices"]))
    features = _confirmation_candidate_features(
        trial_paths, state_ids, trial_indices
    )
    bundle = load_controller_bundle(bundle_manifest_path.parent)
    if str(bundle.manifest.get("default_controller")) != "v2-full":
        raise ValueError("V2 headroom bundle is not the frozen v2-full controller")
    model = bundle.main_models["realized_dynamic"]
    ranges = dict(bundle.main_ranges["realized_dynamic"])
    first = list(map(int, config["first_half_indices"]))
    second = list(map(int, config["second_half_indices"]))
    full = list(map(int, config["trial_indices"]))
    oracle = dict(config["oracle_score"])
    map_group_by_id = {
        str(map_id): str(group_id)
        for group_id, map_ids in dict(config["map_groups"]).items()
        for map_id in map_ids
    }
    records = []
    for state_id, candidates in sorted(profiles.items()):
        candidate_ids = sorted(candidates)
        if set(features.get(state_id, {})) != set(candidate_ids):
            raise ValueError(f"V2 headroom feature/candidate mismatch: {state_id}")
        score_sets = []
        rankings = []
        for indices in (first, second, full):
            scores = {
                candidate_id: _aggregate_candidate_score(
                    candidates[candidate_id], indices, oracle
                )
                for candidate_id in candidate_ids
            }
            ranking = sorted(candidate_ids, key=lambda value: (-scores[value], value))
            score_sets.append(scores)
            rankings.append(ranking)
        first_scores, second_scores, full_scores = score_sets
        first_rank, second_rank, full_rank = rankings
        candidate_rows = [
            {
                "candidate_id": candidate_id,
                "candidate_key": candidate_id,
                "features": {"realized_dynamic": features[state_id][candidate_id]},
            }
            for candidate_id in candidate_ids
        ]
        selected_index, v2_scores, margin = score_online_candidates(
            candidate_rows, model
        )
        selected_id = candidate_ids[selected_index]
        stable_v2_scores = [round(float(value), 12) for value in v2_scores]
        v2_order = sorted(
            range(len(candidate_ids)),
            key=lambda index: (-stable_v2_scores[index], candidate_ids[index]),
        )
        v2_rank = [candidate_ids[index] for index in v2_order]
        selected_features = features[state_id][selected_id]
        outside = sum(
            float(selected_features[name]) < float(bounds[0])
            or float(selected_features[name]) > float(bounds[1])
            for name, bounds in ranges.items()
        )
        outside_fraction = outside / len(ranges) if ranges else 0.0
        half_regrets = [
            _normalized_selection_regret(scores, selected_id)
            for scores in (first_scores, second_scores)
        ]
        source = selection_by_id[state_id]
        records.append(
            {
                "state_id": state_id,
                "map_id": str(source["map_id"]),
                "map_group": map_group_by_id[str(source["map_id"])],
                "source_policy": str(source["source_policy"]),
                "conflict_band": str(source["conflict_band"]),
                "candidate_count": len(candidate_ids),
                "v2_selected_candidate_id": selected_id,
                "full_oracle_candidate_id": full_rank[0],
                "v2_exact_best": full_scores[selected_id]
                >= full_scores[full_rank[0]] - 1e-12,
                "v2_in_oracle_top3": selected_id in set(full_rank[:3]),
                "oracle_in_v2_top3": full_rank[0] in set(v2_rank[:3]),
                "v2_normalized_regret": _normalized_selection_regret(
                    full_scores, selected_id
                ),
                "v2_half_normalized_regret": statistics.fmean(half_regrets),
                "oracle_half_winner_agreement": first_rank[0] == second_rank[0],
                "oracle_half_top3_overlap": len(
                    set(first_rank[:3]) & set(second_rank[:3])
                )
                / 3.0,
                "v2_margin": float(margin),
                "selected_feature_outside_fraction": outside_fraction,
            }
        )
    thresholds = dict(config["diagnostic_thresholds"])
    meaningful = float(thresholds["meaningful_normalized_regret"])
    stable = [row for row in records if row["oracle_half_winner_agreement"]]
    unstable = [row for row in records if not row["oracle_half_winner_agreement"]]
    overall = _headroom_summary(records, meaningful)
    stable_summary = _headroom_summary(stable, meaningful)
    unstable_summary = _headroom_summary(unstable, meaningful)
    stable_headroom = bool(
        len(stable) >= int(thresholds["minimum_stable_state_count"])
        and float(stable_summary["meaningful_headroom_rate"])
        >= float(thresholds["minimum_stable_headroom_rate"])
    )
    uncertainty_dominant = bool(
        overall["oracle_half_winner_agreement_rate"]
        <= float(thresholds["maximum_uncertainty_dominant_winner_agreement"])
        and not stable_headroom
    )
    diagnosis = (
        "frozen_v2_has_stable_model_or_representation_headroom"
        if stable_headroom
        else "pp_seed_uncertainty_dominant"
        if uncertainty_dominant
        else "mixed_or_limited_stable_headroom"
    )
    subgroups = {}
    for field in ("map_group", "source_policy", "conflict_band"):
        for value in sorted({str(row[field]) for row in records}):
            subset = [row for row in records if str(row[field]) == value]
            subgroups[f"{field}:{value}"] = _headroom_summary(subset, meaningful)
    integrity_gates = {
        "state_count": integrity["state_count"] == int(config["expected_state_count"]),
        "candidate_count": integrity["candidate_count"]
        == int(config["expected_candidate_count"]),
        "outcome_count": integrity["outcome_count"]
        == int(config["expected_outcome_count"]),
        "selection_identity": set(profiles) == state_ids,
        "feature_candidate_complete": sum(len(rows) for rows in features.values())
        == integrity["candidate_count"],
        "all_maps_grouped": all(
            str(row["map_id"]) in map_group_by_id for row in selection
        ),
    }
    report = {
        "schema": V2_HEADROOM_REPORT_SCHEMA,
        "scientific_status": "posthoc_diagnostic_only",
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "training_allowed": False,
        "controller_id": "v2-full",
        "oracle_score_id": "mean",
        "integrity": integrity,
        "integrity_gates": integrity_gates,
        "overall": overall,
        "stable_states": stable_summary,
        "unstable_states": unstable_summary,
        "subgroups": subgroups,
        "diagnosis": diagnosis,
        "diagnostic_complete": all(integrity_gates.values()),
        "next_decision": (
            "separate_feature_sufficiency_from_frozen_ranker_error"
            if stable_headroom
            else "retain_v2_and_stop_one_step_successor_training"
            if uncertainty_dominant
            else "retain_v2_and_require_new_diagnostic_design"
        ),
        "records": records,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "confirmation_report_sha256": sha256_file(confirmation_path),
            "selection_sha256": sha256_file(selection_path),
            "controller_manifest_sha256": sha256_file(bundle_manifest_path),
            "trial_source_sha256": {
                str(path): sha256_file(path) for path in trial_paths
            },
        },
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "robuststep_v2_headroom_report.json", report)
    return report


def _feature_probe_pairs(
    states: dict[str, list[dict[str, Any]]],
    state_ids: list[str],
    input_specs: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    import numpy as np

    values: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    stable_pair_count = 0
    possible_pair_count = 0
    states_with_pairs = 0
    for state_id in state_ids:
        state = states[state_id]
        state_examples: list[tuple[list[float], int]] = []
        for left, right in combinations(state, 2):
            possible_pair_count += 1
            first_delta = float(left["first_score"]) - float(right["first_score"])
            second_delta = float(left["second_score"]) - float(right["second_score"])
            if not (
                (first_delta > 1e-12 and second_delta > 1e-12)
                or (first_delta < -1e-12 and second_delta < -1e-12)
            ):
                continue
            stable_pair_count += 1
            winner, loser = (left, right) if first_delta > 0.0 else (right, left)
            state_examples.extend(
                (
                    (pair_vector(winner["features"], loser["features"], input_specs), 1),
                    (pair_vector(loser["features"], winner["features"], input_specs), 0),
                )
            )
        if not state_examples:
            continue
        states_with_pairs += 1
        weight = 1.0 / len(state_examples)
        for vector, label in state_examples:
            values.append(vector)
            labels.append(label)
            weights.append(weight)
    if set(labels) != {0, 1}:
        raise ValueError("feature-probe pair split requires both labels")
    normalized = np.asarray(weights, dtype=np.float64)
    normalized *= len(normalized) / float(np.sum(normalized))
    return {
        "values": np.asarray(values, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int8),
        "weights": normalized,
        "state_count": states_with_pairs,
        "directional_pair_count": len(labels),
        "stable_pair_count": stable_pair_count,
        "possible_pair_count": possible_pair_count,
    }


def _fit_feature_probe(pair_rows: dict[str, Any], parameters: dict[str, Any]) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    estimator = HistGradientBoostingClassifier(**parameters)
    estimator.fit(
        pair_rows["values"],
        pair_rows["labels"],
        sample_weight=pair_rows["weights"],
    )
    return estimator


def _feature_probe_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "state_count": 0,
            "exact_best_rate": None,
            "top3_hit_rate": None,
            "mean_normalized_regret": None,
        }
    mean = lambda name: statistics.fmean(float(row[name]) for row in records)
    return {
        "state_count": len(records),
        "exact_best_rate": mean("exact_best"),
        "top3_hit_rate": mean("top3_hit"),
        "mean_normalized_regret": mean("normalized_regret"),
        "mean_oracle_score": mean("selected_oracle_score"),
    }


def _feature_probe_record(
    model_id: str, state_id: str, selected_id: str, state: list[dict[str, Any]]
) -> dict[str, Any]:
    by_id = {str(row["candidate_id"]): row for row in state}
    selected = by_id[selected_id]
    ranking = sorted(
        state,
        key=lambda row: (-float(row["full_score"]), str(row["candidate_id"])),
    )
    scores = {str(row["candidate_id"]): float(row["full_score"]) for row in state}
    return {
        "model_id": model_id,
        "state_id": state_id,
        "map_id": str(selected["map_id"]),
        "map_group": str(selected["map_group"]),
        "source_policy": str(selected["source_policy"]),
        "conflict_band": str(selected["conflict_band"]),
        "oracle_half_winner_agreement": bool(
            selected["oracle_half_winner_agreement"]
        ),
        "selected_candidate_id": selected_id,
        "oracle_candidate_id": str(ranking[0]["candidate_id"]),
        "exact_best": scores[selected_id] >= scores[str(ranking[0]["candidate_id"])] - 1e-12,
        "top3_hit": selected_id in {str(row["candidate_id"]) for row in ranking[:3]},
        "normalized_regret": _normalized_selection_regret(scores, selected_id),
        "selected_oracle_score": scores[selected_id],
    }


def run_robuststep_feature_probe(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    import numpy as np

    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robuststep_feature_probe_config(config)
    headroom_path = _checked_input(project_root, dict(config["headroom_report"]))
    headroom = _read_json(headroom_path)
    if (
        headroom.get("diagnosis")
        != "frozen_v2_has_stable_model_or_representation_headroom"
        or headroom.get("diagnostic_complete") is not True
    ):
        raise ValueError("feature probe requires the completed stable-headroom diagnosis")
    selection_path = _checked_input(project_root, dict(config["selection"]))
    manifest_path = _checked_input(project_root, dict(config["controller_bundle"]))
    ranker_path = _checked_input(project_root, dict(config["frozen_ranker"]))
    trial_paths = [
        _checked_input(project_root, dict(specification))
        for specification in config["trial_sources"]
    ]
    selection = _read_jsonl(selection_path)
    selection_by_id = {str(row["state_id"]): row for row in selection}
    state_ids = set(selection_by_id)
    if len(state_ids) != len(selection):
        raise ValueError("feature-probe selection repeats state IDs")
    metadata, profiles, integrity = _load_seed_profiles(
        project_root, config, state_ids=state_ids
    )
    features = _confirmation_candidate_features(
        trial_paths, state_ids, set(map(int, config["trial_indices"]))
    )
    registered_names = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    if len(registered_names) != int(config["expected_feature_dimension"]):
        raise ValueError("feature-probe registered feature dimension differs")
    first = list(map(int, config["first_half_indices"]))
    second = list(map(int, config["second_half_indices"]))
    full = list(map(int, config["trial_indices"]))
    oracle = dict(config["oracle_score"])
    map_group_by_id = {
        str(map_id): str(group_id)
        for group_id, map_ids in dict(config["map_groups"]).items()
        for map_id in map_ids
    }
    states: dict[str, list[dict[str, Any]]] = {}
    for state_id, candidates in sorted(profiles.items()):
        source = selection_by_id[state_id]
        candidate_ids = sorted(candidates)
        if set(features.get(state_id, {})) != set(candidate_ids):
            raise ValueError(f"feature-probe feature/candidate mismatch: {state_id}")
        rows = []
        first_ranking = []
        second_ranking = []
        for candidate_id in candidate_ids:
            feature_row = features[state_id][candidate_id]
            if set(feature_row) != set(registered_names):
                raise ValueError(f"feature-probe schema mismatch: {state_id}/{candidate_id}")
            first_score = _aggregate_candidate_score(
                candidates[candidate_id], first, oracle
            )
            second_score = _aggregate_candidate_score(
                candidates[candidate_id], second, oracle
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_key": candidate_id,
                    "features": feature_row,
                    "first_score": first_score,
                    "second_score": second_score,
                    "full_score": _aggregate_candidate_score(
                        candidates[candidate_id], full, oracle
                    ),
                    "map_id": str(source["map_id"]),
                    "map_group": map_group_by_id[str(source["map_id"])],
                    "source_policy": str(source["source_policy"]),
                    "conflict_band": str(source["conflict_band"]),
                }
            )
            first_ranking.append((candidate_id, first_score))
            second_ranking.append((candidate_id, second_score))
        first_winner = min(first_ranking, key=lambda row: (-row[1], row[0]))[0]
        second_winner = min(second_ranking, key=lambda row: (-row[1], row[0]))[0]
        for row in rows:
            row["oracle_half_winner_agreement"] = first_winner == second_winner
        states[state_id] = rows

    ranker_payload = _read_json(ranker_path)
    exact_specs = tuple(
        (str(row["mode"]), str(row["name"]))
        for row in ranker_payload["input_features"]
    )
    delta_specs = tuple(("delta", name) for name in registered_names)
    context_specs = delta_specs + tuple(
        ("shared", name)
        for name in registered_names
        if name.startswith(("state.", "context."))
    )
    specifications = {
        "exact_v2_86": exact_specs,
        "full_124_delta": delta_specs,
        "full_124_context": context_specs,
    }
    if len(exact_specs) != 86:
        raise ValueError("feature-probe frozen input dimension differs")

    bundle = load_controller_bundle(manifest_path.parent)
    frozen_model = bundle.main_models["realized_dynamic"]
    frozen_predictions = {}
    for state_id, state in sorted(states.items()):
        online_rows = [
            {
                "candidate_id": row["candidate_id"],
                "candidate_key": row["candidate_key"],
                "features": {"realized_dynamic": row["features"]},
            }
            for row in state
        ]
        selected_index, _, _ = score_online_candidates(online_rows, frozen_model)
        frozen_predictions[state_id] = str(state[selected_index]["candidate_id"])

    held_out_maps = list(map(str, dict(config["fold_protocol"])["held_out_maps"]))
    parameters = dict(config["model_parameters"])
    predictions: dict[str, dict[str, str]] = {}
    fold_reports: dict[str, list[dict[str, Any]]] = {}
    oof_integrity = {}
    for variant in config["variants"]:
        model_id = str(variant["id"])
        input_specs = specifications[str(variant["input_profile"])]
        variant_predictions = {}
        variant_folds = []
        for fold_index, held_out_map in enumerate(held_out_maps):
            train_ids = sorted(
                state_id
                for state_id, state in states.items()
                if str(state[0]["map_id"]) != held_out_map
            )
            test_ids = sorted(
                state_id
                for state_id, state in states.items()
                if str(state[0]["map_id"]) == held_out_map
            )
            if not train_ids or not test_ids:
                raise ValueError(f"empty feature-probe map fold: {held_out_map}")
            train_pairs = _feature_probe_pairs(states, train_ids, input_specs)
            test_pairs = _feature_probe_pairs(states, test_ids, input_specs)
            estimator = _fit_feature_probe(train_pairs, parameters)
            probabilities = estimator.predict_proba(test_pairs["values"])[:, 1]
            correct = (probabilities >= 0.5) == (test_pairs["labels"] == 1)
            pairwise_accuracy = float(
                np.sum(test_pairs["weights"][correct])
                / np.sum(test_pairs["weights"])
            )
            for state_id in test_ids:
                chosen = _select_model(states[state_id], estimator, input_specs)
                variant_predictions[state_id] = str(chosen["candidate_id"])
            variant_folds.append(
                {
                    "fold_index": fold_index,
                    "held_out_map": held_out_map,
                    "training_state_count": len(train_ids),
                    "test_state_count": len(test_ids),
                    "training_directional_pair_count": train_pairs[
                        "directional_pair_count"
                    ],
                    "test_directional_pair_count": test_pairs[
                        "directional_pair_count"
                    ],
                    "test_pairwise_accuracy": pairwise_accuracy,
                }
            )
        predictions[model_id] = variant_predictions
        fold_reports[model_id] = variant_folds
        oof_integrity[model_id] = set(variant_predictions) == state_ids

    records_by_model = {
        "v2-full": [
            _feature_probe_record(
                "v2-full", state_id, frozen_predictions[state_id], states[state_id]
            )
            for state_id in sorted(state_ids)
        ]
    }
    for model_id, model_predictions in predictions.items():
        records_by_model[model_id] = [
            _feature_probe_record(
                model_id, state_id, model_predictions[state_id], states[state_id]
            )
            for state_id in sorted(state_ids)
        ]

    summaries = {}
    for model_id, records in records_by_model.items():
        summaries[model_id] = {
            "overall": _feature_probe_summary(records),
            "stable_states": _feature_probe_summary(
                [row for row in records if row["oracle_half_winner_agreement"]]
            ),
            "unstable_states": _feature_probe_summary(
                [row for row in records if not row["oracle_half_winner_agreement"]]
            ),
            "by_map": {
                map_id: _feature_probe_summary(
                    [row for row in records if row["map_id"] == map_id]
                )
                for map_id in held_out_maps
            },
        }

    thresholds = dict(config["diagnostic_thresholds"])
    frozen_summary = summaries["v2-full"]
    evaluations = {}
    passing_variants = []
    for variant in config["variants"]:
        model_id = str(variant["id"])
        summary = summaries[model_id]
        stable_regret_improvement = float(
            frozen_summary["stable_states"]["mean_normalized_regret"]
        ) - float(summary["stable_states"]["mean_normalized_regret"])
        overall_regret_improvement = float(
            frozen_summary["overall"]["mean_normalized_regret"]
        ) - float(summary["overall"]["mean_normalized_regret"])
        stable_top3_delta = float(summary["stable_states"]["top3_hit_rate"]) - float(
            frozen_summary["stable_states"]["top3_hit_rate"]
        )
        map_deltas = {
            map_id: float(summary["by_map"][map_id]["mean_normalized_regret"])
            - float(frozen_summary["by_map"][map_id]["mean_normalized_regret"])
            for map_id in held_out_maps
        }
        map_win_count = sum(delta < -1e-12 for delta in map_deltas.values())
        worst_map_degradation = max(map_deltas.values())
        gates = {
            "minimum_stable_state_count": int(
                summary["stable_states"]["state_count"]
            )
            >= int(thresholds["minimum_stable_state_count"]),
            "minimum_stable_regret_improvement": stable_regret_improvement
            >= float(thresholds["minimum_stable_regret_improvement"]),
            "minimum_overall_regret_improvement": overall_regret_improvement
            >= float(thresholds["minimum_overall_regret_improvement"]),
            "minimum_stable_top3_delta": stable_top3_delta
            >= float(thresholds["minimum_stable_top3_delta"]),
            "minimum_map_regret_win_count": map_win_count
            >= int(thresholds["minimum_map_regret_win_count"]),
            "maximum_worst_map_regret_degradation": worst_map_degradation
            <= float(thresholds["maximum_worst_map_regret_degradation"]),
        }
        passed = all(gates.values())
        if passed:
            passing_variants.append(model_id)
        evaluations[model_id] = {
            "stable_regret_improvement": stable_regret_improvement,
            "overall_regret_improvement": overall_regret_improvement,
            "stable_top3_delta": stable_top3_delta,
            "map_regret_deltas": map_deltas,
            "map_regret_win_count": map_win_count,
            "worst_map_regret_degradation": worst_map_degradation,
            "gates": gates,
            "passed": passed,
        }

    ordered = [str(row["id"]) for row in config["variants"]]
    first_passing = next((model_id for model_id in ordered if model_id in passing_variants), None)
    diagnosis = (
        "existing_v2_feature_subset_is_sufficient_old_training_objective_or_distribution_mismatch"
        if first_passing == "stride-stepdiag-v1/exact-v2-86"
        else "additional_candidate_features_are_needed"
        if first_passing == "stride-stepdiag-v1/full-124-delta"
        else "state_conditioning_is_needed"
        if first_passing == "stride-stepdiag-v1/full-124-context"
        else "feature_sufficiency_not_demonstrated_on_consumed_cohort"
    )
    headroom_frozen = {
        str(row["state_id"]): str(row["v2_selected_candidate_id"])
        for row in headroom["records"]
    }
    all_pair_rows = _feature_probe_pairs(states, sorted(state_ids), exact_specs)
    current_maps = {str(state[0]["map_id"]) for state in states.values()}
    integrity_gates = {
        "state_count": integrity["state_count"] == int(config["expected_state_count"]),
        "candidate_count": integrity["candidate_count"]
        == int(config["expected_candidate_count"]),
        "outcome_count": integrity["outcome_count"]
        == int(config["expected_outcome_count"]),
        "feature_candidate_complete": sum(len(rows) for rows in features.values())
        == integrity["candidate_count"],
        "whole_map_fold_partition": current_maps == set(held_out_maps),
        "frozen_selection_matches_headroom": frozen_predictions == headroom_frozen,
        "stable_pairs_cover_every_state": all_pair_rows["state_count"] == len(state_ids),
        "complete_oof_predictions": all(oof_integrity.values()),
    }
    report = {
        "schema": FEATURE_PROBE_REPORT_SCHEMA,
        "scientific_status": "consumed_feature_sufficiency_diagnostic",
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "runtime_model_exported": False,
        "formal_ood_claim": False,
        "diagnostic_controller_id": str(config["diagnostic_controller_id"]),
        "frozen_anchor_id": str(config["frozen_anchor_id"]),
        "integrity": integrity,
        "integrity_gates": integrity_gates,
        "pair_contract": {
            **dict(config["pair_contract"]),
            "stable_pair_count": all_pair_rows["stable_pair_count"],
            "possible_pair_count": all_pair_rows["possible_pair_count"],
            "stable_pair_fraction": all_pair_rows["stable_pair_count"]
            / all_pair_rows["possible_pair_count"],
        },
        "input_dimensions": {
            profile: len(specifications[profile]) for profile in specifications
        },
        "folds": fold_reports,
        "summaries": summaries,
        "evaluations": evaluations,
        "passing_variants": passing_variants,
        "diagnosis": diagnosis,
        "diagnostic_complete": all(integrity_gates.values()),
        "next_decision": (
            "register_new_fresh_data_before_any_successor_training"
            if passing_variants
            else "retain_v2_and_redesign_features_or_candidate_generation"
        ),
        "records": records_by_model,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "headroom_report_sha256": sha256_file(headroom_path),
            "selection_sha256": sha256_file(selection_path),
            "controller_manifest_sha256": sha256_file(manifest_path),
            "frozen_ranker_sha256": sha256_file(ranker_path),
            "trial_source_sha256": {
                str(path): sha256_file(path) for path in trial_paths
            },
        },
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "robuststep_feature_probe_report.json", report)
    return report


def _rebuild_stepgate_states(
    project_root: Path, feature_config: dict[str, Any]
) -> dict[str, Any]:
    selection_path = _checked_input(project_root, dict(feature_config["selection"]))
    manifest_path = _checked_input(
        project_root, dict(feature_config["controller_bundle"])
    )
    ranker_path = _checked_input(project_root, dict(feature_config["frozen_ranker"]))
    trial_paths = [
        _checked_input(project_root, dict(specification))
        for specification in feature_config["trial_sources"]
    ]
    selection = _read_jsonl(selection_path)
    selection_by_id = {str(row["state_id"]): row for row in selection}
    state_ids = set(selection_by_id)
    if len(state_ids) != len(selection):
        raise ValueError("step-gate selection repeats state IDs")
    _, profiles, integrity = _load_seed_profiles(
        project_root, feature_config, state_ids=state_ids
    )
    features = _confirmation_candidate_features(
        trial_paths,
        state_ids,
        set(map(int, feature_config["trial_indices"])),
    )
    registered_names = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    first = list(map(int, feature_config["first_half_indices"]))
    second = list(map(int, feature_config["second_half_indices"]))
    full = list(map(int, feature_config["trial_indices"]))
    oracle = dict(feature_config["oracle_score"])
    map_group_by_id = {
        str(map_id): str(group_id)
        for group_id, map_ids in dict(feature_config["map_groups"]).items()
        for map_id in map_ids
    }
    states: dict[str, list[dict[str, Any]]] = {}
    for state_id, candidates in sorted(profiles.items()):
        source = selection_by_id[state_id]
        candidate_ids = sorted(candidates)
        if set(features.get(state_id, {})) != set(candidate_ids):
            raise ValueError(f"step-gate feature/candidate mismatch: {state_id}")
        rows = []
        first_ranking = []
        second_ranking = []
        for candidate_id in candidate_ids:
            feature_row = features[state_id][candidate_id]
            if set(feature_row) != set(registered_names):
                raise ValueError(f"step-gate feature schema mismatch: {state_id}")
            first_score = _aggregate_candidate_score(
                candidates[candidate_id], first, oracle
            )
            second_score = _aggregate_candidate_score(
                candidates[candidate_id], second, oracle
            )
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_key": candidate_id,
                    "features": feature_row,
                    "first_score": first_score,
                    "second_score": second_score,
                    "full_score": _aggregate_candidate_score(
                        candidates[candidate_id], full, oracle
                    ),
                    "map_id": str(source["map_id"]),
                    "map_group": map_group_by_id[str(source["map_id"])],
                    "source_policy": str(source["source_policy"]),
                    "conflict_band": str(source["conflict_band"]),
                }
            )
            first_ranking.append((candidate_id, first_score))
            second_ranking.append((candidate_id, second_score))
        first_winner = min(first_ranking, key=lambda row: (-row[1], row[0]))[0]
        second_winner = min(second_ranking, key=lambda row: (-row[1], row[0]))[0]
        for row in rows:
            row["oracle_half_winner_agreement"] = first_winner == second_winner
        states[state_id] = rows

    ranker_payload = _read_json(ranker_path)
    input_specs = tuple(
        (str(row["mode"]), str(row["name"]))
        for row in ranker_payload["input_features"]
    )
    if len(input_specs) != 86:
        raise ValueError("step-gate exact V2 input dimension differs")
    bundle = load_controller_bundle(manifest_path.parent)
    frozen_model = bundle.main_models["realized_dynamic"]
    frozen_predictions = {}
    for state_id, state in sorted(states.items()):
        online_rows = [
            {
                "candidate_id": row["candidate_id"],
                "candidate_key": row["candidate_key"],
                "features": {"realized_dynamic": row["features"]},
            }
            for row in state
        ]
        selected_index, _, _ = score_online_candidates(online_rows, frozen_model)
        frozen_predictions[state_id] = str(state[selected_index]["candidate_id"])
    return {
        "states": states,
        "state_ids": state_ids,
        "input_specs": input_specs,
        "frozen_predictions": frozen_predictions,
        "integrity": integrity,
        "feature_candidate_count": sum(len(rows) for rows in features.values()),
        "input_paths": {
            "selection": selection_path,
            "controller_manifest": manifest_path,
            "frozen_ranker": ranker_path,
            "trial_sources": trial_paths,
        },
    }


def _stepgate_advantage_probability(
    estimator: Any,
    challenger: dict[str, Any],
    anchor: dict[str, Any],
    input_specs: tuple[tuple[str, str], ...],
) -> float:
    import numpy as np

    if challenger["candidate_id"] == anchor["candidate_id"]:
        return 0.5
    forward = estimator.predict_proba(
        np.asarray(
            [pair_vector(challenger["features"], anchor["features"], input_specs)],
            dtype=np.float32,
        )
    )[0, 1]
    reverse = estimator.predict_proba(
        np.asarray(
            [pair_vector(anchor["features"], challenger["features"], input_specs)],
            dtype=np.float32,
        )
    )[0, 1]
    return (float(forward) + (1.0 - float(reverse))) / 2.0


def _stepgate_fold_decisions(
    *,
    states: dict[str, list[dict[str, Any]]],
    train_ids: list[str],
    test_ids: list[str],
    input_specs: tuple[tuple[str, str], ...],
    parameters: dict[str, Any],
    frozen_predictions: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    pair_rows = _feature_probe_pairs(states, train_ids, input_specs)
    estimator = _fit_feature_probe(pair_rows, parameters)
    decisions = {}
    for state_id in test_ids:
        state = states[state_id]
        challenger = _select_model(state, estimator, input_specs)
        by_id = {str(row["candidate_id"]): row for row in state}
        anchor = by_id[frozen_predictions[state_id]]
        decisions[state_id] = {
            "challenger_candidate_id": str(challenger["candidate_id"]),
            "frozen_candidate_id": str(anchor["candidate_id"]),
            "challenger_advantage_probability": _stepgate_advantage_probability(
                estimator, challenger, anchor, input_specs
            ),
        }
    return decisions, {
        "training_state_count": len(train_ids),
        "test_state_count": len(test_ids),
        "training_directional_pair_count": pair_rows["directional_pair_count"],
    }


def _stepgate_predictions(
    decisions: dict[str, dict[str, Any]], threshold: float
) -> dict[str, str]:
    return {
        state_id: (
            str(row["challenger_candidate_id"])
            if str(row["challenger_candidate_id"])
            != str(row["frozen_candidate_id"])
            and float(row["challenger_advantage_probability"]) + 1e-12 >= threshold
            else str(row["frozen_candidate_id"])
        )
        for state_id, row in decisions.items()
    }


def _stepgate_model_summary(
    records: list[dict[str, Any]], maps: list[str]
) -> dict[str, Any]:
    return {
        "overall": _feature_probe_summary(records),
        "stable_states": _feature_probe_summary(
            [row for row in records if row["oracle_half_winner_agreement"]]
        ),
        "unstable_states": _feature_probe_summary(
            [row for row in records if not row["oracle_half_winner_agreement"]]
        ),
        "by_map": {
            map_id: _feature_probe_summary(
                [row for row in records if row["map_id"] == map_id]
            )
            for map_id in maps
        },
    }


def _stepgate_evaluation(
    *,
    predictions: dict[str, str],
    decisions: dict[str, dict[str, Any]],
    states: dict[str, list[dict[str, Any]]],
    frozen_predictions: dict[str, str],
    maps: list[str],
    gates_config: dict[str, Any],
    include_stable_count_gate: bool,
) -> dict[str, Any]:
    model_records = [
        _feature_probe_record(
            "stride-stepgate-v1", state_id, predictions[state_id], states[state_id]
        )
        for state_id in sorted(predictions)
    ]
    frozen_records = [
        _feature_probe_record(
            "v2-full", state_id, frozen_predictions[state_id], states[state_id]
        )
        for state_id in sorted(predictions)
    ]
    model = _stepgate_model_summary(model_records, maps)
    frozen = _stepgate_model_summary(frozen_records, maps)
    overall_improvement = float(frozen["overall"]["mean_normalized_regret"]) - float(
        model["overall"]["mean_normalized_regret"]
    )
    stable_improvement = float(
        frozen["stable_states"]["mean_normalized_regret"]
    ) - float(model["stable_states"]["mean_normalized_regret"])
    stable_top3_delta = float(model["stable_states"]["top3_hit_rate"]) - float(
        frozen["stable_states"]["top3_hit_rate"]
    )
    map_deltas = {
        map_id: float(model["by_map"][map_id]["mean_normalized_regret"])
        - float(frozen["by_map"][map_id]["mean_normalized_regret"])
        for map_id in maps
    }
    switched = [
        state_id
        for state_id, selected_id in predictions.items()
        if selected_id != frozen_predictions[state_id]
    ]
    model_by_state = {str(row["state_id"]): row for row in model_records}
    frozen_by_state = {str(row["state_id"]): row for row in frozen_records}
    improved_switches = sum(
        float(model_by_state[state_id]["normalized_regret"]) + 1e-12
        < float(frozen_by_state[state_id]["normalized_regret"])
        for state_id in switched
    )
    degraded_switches = sum(
        float(model_by_state[state_id]["normalized_regret"])
        > float(frozen_by_state[state_id]["normalized_regret"]) + 1e-12
        for state_id in switched
    )
    switch_precision = improved_switches / len(switched) if switched else 0.0
    gates = {
        "minimum_overall_regret_improvement": overall_improvement
        >= float(gates_config["minimum_overall_regret_improvement"]),
        "minimum_stable_regret_improvement": stable_improvement
        >= float(gates_config["minimum_stable_regret_improvement"]),
        "minimum_stable_top3_delta": stable_top3_delta
        >= float(gates_config["minimum_stable_top3_delta"]),
        "maximum_worst_map_regret_degradation": max(map_deltas.values())
        <= float(gates_config["maximum_worst_map_regret_degradation"]),
        "minimum_switch_count": len(switched)
        >= int(gates_config["minimum_switch_count"]),
    }
    if include_stable_count_gate:
        gates.update(
            {
                "minimum_stable_state_count": int(
                    model["stable_states"]["state_count"]
                )
                >= int(gates_config["minimum_stable_state_count"]),
                "minimum_map_regret_win_count": sum(
                    delta < -1e-12 for delta in map_deltas.values()
                )
                >= int(gates_config["minimum_map_regret_win_count"]),
                "minimum_switch_precision": switch_precision
                >= float(gates_config["minimum_switch_precision"]),
            }
        )
    return {
        "model": model,
        "frozen": frozen,
        "overall_regret_improvement": overall_improvement,
        "stable_regret_improvement": stable_improvement,
        "stable_top3_delta": stable_top3_delta,
        "map_regret_deltas": map_deltas,
        "map_regret_win_count": sum(delta < -1e-12 for delta in map_deltas.values()),
        "worst_map_regret_degradation": max(map_deltas.values()),
        "switch_count": len(switched),
        "switch_rate": len(switched) / len(predictions),
        "improved_switch_count": improved_switches,
        "degraded_switch_count": degraded_switches,
        "switch_precision": switch_precision,
        "gates": gates,
        "passed": all(gates.values()),
    }


def run_robuststep_stepgate(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_robuststep_stepgate_config(config)
    feature_config_path = _checked_input(
        project_root, dict(config["feature_probe_config"])
    )
    feature_report_path = _checked_input(
        project_root, dict(config["feature_probe_report"])
    )
    feature_config = _read_json(feature_config_path)
    validate_robuststep_feature_probe_config(feature_config)
    feature_report = _read_json(feature_report_path)
    if (
        feature_report.get("diagnosis")
        != "feature_sufficiency_not_demonstrated_on_consumed_cohort"
        or feature_report.get("diagnostic_complete") is not True
        or feature_report.get("default_replacement_allowed") is not False
    ):
        raise ValueError("step gate requires the completed failed feature probe")
    prepared = _rebuild_stepgate_states(project_root, feature_config)
    states = prepared["states"]
    state_ids = prepared["state_ids"]
    input_specs = prepared["input_specs"]
    frozen_predictions = prepared["frozen_predictions"]
    parameters = dict(feature_config["model_parameters"])
    maps = list(map(str, dict(config["outer_fold_protocol"])["held_out_maps"]))
    thresholds = list(map(float, dict(config["inner_calibration"])["candidate_thresholds"]))
    abstain_threshold = float(dict(config["inner_calibration"])["abstain_threshold"])
    inner_gates = dict(config["inner_gates"])

    outer_predictions = {}
    outer_challenger_predictions = {}
    outer_decisions = {}
    fold_reports = []
    for fold_index, outer_map in enumerate(maps):
        outer_train_maps = [map_id for map_id in maps if map_id != outer_map]
        inner_decisions = {}
        inner_fold_rows = []
        for inner_map in outer_train_maps:
            inner_train_ids = sorted(
                state_id
                for state_id, state in states.items()
                if str(state[0]["map_id"]) not in {outer_map, inner_map}
            )
            inner_test_ids = sorted(
                state_id
                for state_id, state in states.items()
                if str(state[0]["map_id"]) == inner_map
            )
            decisions, fit = _stepgate_fold_decisions(
                states=states,
                train_ids=inner_train_ids,
                test_ids=inner_test_ids,
                input_specs=input_specs,
                parameters=parameters,
                frozen_predictions=frozen_predictions,
            )
            inner_decisions.update(decisions)
            inner_fold_rows.append(
                {
                    "held_out_inner_map": inner_map,
                    "inner_training_maps": sorted(
                        set(outer_train_maps) - {inner_map}
                    ),
                    **fit,
                }
            )
        calibration_rows = []
        selected_threshold = abstain_threshold
        for threshold in thresholds:
            predictions = _stepgate_predictions(inner_decisions, threshold)
            evaluation = _stepgate_evaluation(
                predictions=predictions,
                decisions=inner_decisions,
                states=states,
                frozen_predictions=frozen_predictions,
                maps=outer_train_maps,
                gates_config=inner_gates,
                include_stable_count_gate=False,
            )
            calibration_rows.append(
                {
                    "threshold": threshold,
                    "overall_regret_improvement": evaluation[
                        "overall_regret_improvement"
                    ],
                    "stable_regret_improvement": evaluation[
                        "stable_regret_improvement"
                    ],
                    "stable_top3_delta": evaluation["stable_top3_delta"],
                    "worst_map_regret_degradation": evaluation[
                        "worst_map_regret_degradation"
                    ],
                    "switch_count": evaluation["switch_count"],
                    "gates": evaluation["gates"],
                    "passed": evaluation["passed"],
                }
            )
            if evaluation["passed"] and selected_threshold == abstain_threshold:
                selected_threshold = threshold

        outer_train_ids = sorted(
            state_id
            for state_id, state in states.items()
            if str(state[0]["map_id"]) != outer_map
        )
        outer_test_ids = sorted(
            state_id
            for state_id, state in states.items()
            if str(state[0]["map_id"]) == outer_map
        )
        decisions, fit = _stepgate_fold_decisions(
            states=states,
            train_ids=outer_train_ids,
            test_ids=outer_test_ids,
            input_specs=input_specs,
            parameters=parameters,
            frozen_predictions=frozen_predictions,
        )
        predictions = _stepgate_predictions(decisions, selected_threshold)
        outer_predictions.update(predictions)
        outer_decisions.update(decisions)
        outer_challenger_predictions.update(
            {
                state_id: str(row["challenger_candidate_id"])
                for state_id, row in decisions.items()
            }
        )
        fold_reports.append(
            {
                "fold_index": fold_index,
                "held_out_outer_map": outer_map,
                "outer_training_maps": outer_train_maps,
                "selected_threshold": selected_threshold,
                "calibration_fell_back_to_abstain": selected_threshold
                == abstain_threshold,
                "inner_folds": inner_fold_rows,
                "calibration": calibration_rows,
                "outer_fit": fit,
                "outer_switch_count": sum(
                    predictions[state_id] != frozen_predictions[state_id]
                    for state_id in outer_test_ids
                ),
            }
        )

    evaluation = _stepgate_evaluation(
        predictions=outer_predictions,
        decisions=outer_decisions,
        states=states,
        frozen_predictions=frozen_predictions,
        maps=maps,
        gates_config=dict(config["diagnostic_gates"]),
        include_stable_count_gate=True,
    )
    feature_frozen = {
        str(row["state_id"]): str(row["selected_candidate_id"])
        for row in feature_report["records"]["v2-full"]
    }
    feature_challenger = {
        str(row["state_id"]): str(row["selected_candidate_id"])
        for row in feature_report["records"][str(config["challenger_id"])]
    }
    integrity = prepared["integrity"]
    integrity_gates = {
        "state_count": integrity["state_count"]
        == int(feature_config["expected_state_count"]),
        "candidate_count": integrity["candidate_count"]
        == int(feature_config["expected_candidate_count"]),
        "outcome_count": integrity["outcome_count"]
        == int(feature_config["expected_outcome_count"]),
        "feature_candidate_complete": prepared["feature_candidate_count"]
        == integrity["candidate_count"],
        "outer_predictions_complete": set(outer_predictions) == state_ids,
        "outer_challenger_matches_feature_probe": outer_challenger_predictions
        == feature_challenger,
        "frozen_predictions_match_feature_probe": frozen_predictions == feature_frozen,
        "thresholds_are_inner_selected_or_abstain": all(
            float(row["selected_threshold"]) in {*thresholds, abstain_threshold}
            for row in fold_reports
        ),
        "outer_maps_partition_states": {
            str(state[0]["map_id"]) for state in states.values()
        }
        == set(maps),
    }
    diagnostic_passed = all(integrity_gates.values()) and bool(evaluation["passed"])
    report = {
        "schema": STEPGATE_REPORT_SCHEMA,
        "scientific_status": "consumed_nested_abstention_diagnostic",
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "runtime_model_exported": False,
        "formal_ood_claim": False,
        "switch_timing": "before_pp_repair",
        "failure_triggered_switching": False,
        "diagnostic_controller_id": str(config["diagnostic_controller_id"]),
        "frozen_anchor_id": str(config["frozen_anchor_id"]),
        "challenger_id": str(config["challenger_id"]),
        "integrity": integrity,
        "integrity_gates": integrity_gates,
        "folds": fold_reports,
        "evaluation": evaluation,
        "diagnostic_passed": diagnostic_passed,
        "diagnosis": (
            "nested_confidence_gate_controls_consumed_map_regression"
            if diagnostic_passed
            else "nested_confidence_gate_does_not_control_consumed_map_regression"
        ),
        "next_decision": (
            "register_fresh_map_confirmation_for_stepgate_without_promoting"
            if diagnostic_passed
            else "retain_v2_and_start_topology_interaction_feature_design"
        ),
        "decisions": outer_decisions,
        "predictions": outer_predictions,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "feature_probe_config_sha256": sha256_file(feature_config_path),
            "feature_probe_report_sha256": sha256_file(feature_report_path),
            "selection_sha256": sha256_file(
                prepared["input_paths"]["selection"]
            ),
            "controller_manifest_sha256": sha256_file(
                prepared["input_paths"]["controller_manifest"]
            ),
            "frozen_ranker_sha256": sha256_file(
                prepared["input_paths"]["frozen_ranker"]
            ),
            "trial_source_sha256": {
                str(path): sha256_file(path)
                for path in prepared["input_paths"]["trial_sources"]
            },
        },
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "robuststep_stepgate_report.json", report)
    return report


__all__ = [
    "CONTROLLER_ID",
    "LABEL_SCHEMA",
    "evaluate_robuststep_variant",
    "robust_pair_winner",
    "run_robuststep_design",
    "run_robuststep_confirmation",
    "run_robuststep_seed_depth",
    "run_robuststep_score_design",
    "run_robuststep_v2_headroom",
    "run_robuststep_feature_probe",
    "run_robuststep_stepgate",
    "validate_robuststep_config",
    "validate_robuststep_confirmation_config",
    "validate_robuststep_v2_headroom_config",
    "validate_robuststep_feature_probe_config",
    "validate_robuststep_stepgate_config",
    "validate_robuststep_seed_depth_config",
    "validate_robuststep_score_config",
    "evaluate_robuststep_score_variant",
]
