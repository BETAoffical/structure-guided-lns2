from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_robustaction_label_collection import state_artifact_tree_sha256


CONFIG_SCHEMA = "lns2.stride.safeslot_label_readiness_config.v1"
COMPARISON_SCHEMA = "lns2.stride.safeslot_pre_residual_comparison.v1"
STATE_SCHEMA = "lns2.stride.safeslot_state_readiness.v1"
REPORT_SCHEMA = "lns2.stride.safeslot_label_readiness_report.v1"
EXPERIMENT_ID = "stride-safeslot-label-readiness-v1"


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"registered SafeSlot readiness input is missing: {path}")
    observed = sha256_file(path)
    expected = str(specification["sha256"])
    if observed != expected:
        raise ValueError(
            f"registered SafeSlot readiness input changed: {path}: "
            f"expected {expected}, got {observed}"
        )
    return path


def validate_safeslot_label_readiness_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "retrospective_registered_pre_residual_teacher_readiness_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("implementation_id") != "stride-safeslot-v1"
        or config.get("pre_registration_parent_commit")
        != "464fef97126c29f2105d1c582cfd7195fcca3934"
    ):
        raise ValueError("SafeSlot label-readiness identity changed")

    expected_inputs = {
        "grid_report",
        "grid_manifest",
        "grid_label_report",
        "grid_candidate_aggregates",
        "grid_repair_trials",
        "base_label_report",
        "base_candidate_aggregates",
        "base_repair_trials",
        "slotpool_report",
        "slotpool_state_evaluation",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("SafeSlot label-readiness input registry changed")

    if dict(config.get("cohort") or {}) != {
        "state_count": 98,
        "map_count": 16,
        "grid_candidate_count": 1368,
        "grid_trial_count": 21888,
        "slotpool_selected_candidate_count": 588,
        "known_maze_long_tail_excluded": True,
        "all_states_retained": True,
    }:
        raise ValueError("SafeSlot label-readiness cohort changed")

    label = dict(config.get("paired_pre_residual_label") or {})
    if (
        tuple(map(int, label.get("trial_indices") or ())) != tuple(range(16))
        or tuple(map(int, label.get("first_fixed_half") or ())) != tuple(range(8))
        or tuple(map(int, label.get("second_fixed_half") or ()))
        != tuple(range(8, 16))
        or label.get("target")
        != "challenger_minus_base_only_v2_anchor_normalized_current_step_reduction"
        or float(label.get("minimum_mean_advantage", -1.0)) != 0.02
        or float(label.get("minimum_paired_win_fraction", -1.0)) != 0.75
        or label.get("fixed_half_rule")
        != "strictly_positive_mean_advantage_in_each_half"
        or label.get("no_progress_rule")
        != "challenger_rate_not_above_anchor_rate"
        or float(label.get("tie_epsilon", -1.0)) != 1e-12
        or label.get("pp_seed_pairing")
        != "same_state_and_trial_index_exact_seed"
        or label.get("residual_teacher_required_for_final_positive_label") is not True
    ):
        raise ValueError("SafeSlot pre-residual label contract changed")

    if dict(config.get("readiness_gates") or {}) != {
        "minimum_pre_residual_positive_state_count": 25,
        "minimum_pre_residual_positive_candidate_count": 50,
        "minimum_positive_map_count": 8,
        "minimum_slotpool_positive_state_retention": 0.85,
        "minimum_slotpool_retained_positive_candidate_count": 30,
    }:
        raise ValueError("SafeSlot label-readiness gates changed")

    boundary = dict(config.get("claim_boundary") or {})
    if set(boundary) != {
        "model_training_allowed",
        "runtime_integration_allowed",
        "formal_ttf_claim",
        "runtime_or_ttf_read",
        "future_trajectory_read",
        "cost_to_go_read",
        "remaining_repair_rounds_read",
        "known_maze_result_used",
        "post_state_collection_authorized_only_if_all_readiness_gates_pass",
    }:
        raise ValueError("SafeSlot label-readiness claim boundary changed")
    if boundary.get(
        "post_state_collection_authorized_only_if_all_readiness_gates_pass"
    ) is not True or any(
        boundary.get(name) is not False
        for name in (
            "model_training_allowed",
            "runtime_integration_allowed",
            "formal_ttf_claim",
            "runtime_or_ttf_read",
            "future_trajectory_read",
            "cost_to_go_read",
            "remaining_repair_rounds_read",
            "known_maze_result_used",
        )
    ):
        raise ValueError("SafeSlot label-readiness claim values changed")

    if dict(config.get("outputs") or {}) != {
        "comparison_rows": "safeslot_pre_residual_comparisons.jsonl",
        "state_rows": "safeslot_state_readiness.jsonl",
        "report": "safeslot_label_readiness_report.json",
    }:
        raise ValueError("SafeSlot label-readiness outputs changed")

    if project_root is not None:
        for specification in dict(config["inputs"]).values():
            _registered(project_root, dict(specification))
        state_root = (
            project_root / str(config["grid_state_artifact_root"])
        ).resolve()
        if state_artifact_tree_sha256(state_root) != str(
            config["grid_state_artifact_tree_sha256"]
        ):
            raise ValueError("SafeSlot grid state artifact tree changed")


def _trials_by_action(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    for key in result:
        result[key].sort(key=lambda row: int(row["trial_index"]))
    return dict(result)


def compare_paired_current_step_trials(
    challenger_trials: list[dict[str, Any]],
    anchor_trials: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    expected_indices = tuple(map(int, contract["trial_indices"]))
    challenger_indices = tuple(map(int, (row["trial_index"] for row in challenger_trials)))
    anchor_indices = tuple(map(int, (row["trial_index"] for row in anchor_trials)))
    if challenger_indices != expected_indices or anchor_indices != expected_indices:
        raise ValueError("SafeSlot paired trial indices changed")

    challenger_scores: list[float] = []
    anchor_scores: list[float] = []
    for challenger, anchor in zip(challenger_trials, anchor_trials):
        if (
            str(challenger["state_id"]) != str(anchor["state_id"])
            or int(challenger["pp_seed"]) != int(anchor["pp_seed"])
            or str(challenger["before_fingerprint"])
            != str(anchor["before_fingerprint"])
            or int(challenger["before_conflicts"]) != int(anchor["before_conflicts"])
        ):
            raise ValueError("SafeSlot paired trial state or PP seed differs")
        challenger_score = float(challenger["normalized_conflict_reduction"])
        anchor_score = float(anchor["normalized_conflict_reduction"])
        if not math.isfinite(challenger_score) or not math.isfinite(anchor_score):
            raise ValueError("SafeSlot paired trial score is non-finite")
        challenger_scores.append(challenger_score)
        anchor_scores.append(anchor_score)

    differences = [
        challenger - anchor
        for challenger, anchor in zip(challenger_scores, anchor_scores)
    ]
    first_indices = tuple(map(int, contract["first_fixed_half"]))
    second_indices = tuple(map(int, contract["second_fixed_half"]))
    mean_advantage = statistics.fmean(differences)
    first_half_advantage = statistics.fmean(differences[index] for index in first_indices)
    second_half_advantage = statistics.fmean(
        differences[index] for index in second_indices
    )
    epsilon = float(contract["tie_epsilon"])
    paired_win_fraction = sum(value > epsilon for value in differences) / len(differences)
    paired_nonloss_fraction = sum(value >= -epsilon for value in differences) / len(
        differences
    )
    challenger_no_progress_rate = sum(score <= epsilon for score in challenger_scores) / len(
        challenger_scores
    )
    anchor_no_progress_rate = sum(score <= epsilon for score in anchor_scores) / len(
        anchor_scores
    )
    pre_residual_positive = bool(
        mean_advantage >= float(contract["minimum_mean_advantage"])
        and paired_win_fraction >= float(contract["minimum_paired_win_fraction"])
        and first_half_advantage > epsilon
        and second_half_advantage > epsilon
        and challenger_no_progress_rate <= anchor_no_progress_rate + epsilon
    )
    return {
        "anchor_seed_mean": statistics.fmean(anchor_scores),
        "challenger_seed_mean": statistics.fmean(challenger_scores),
        "mean_advantage": mean_advantage,
        "first_fixed_half_advantage": first_half_advantage,
        "second_fixed_half_advantage": second_half_advantage,
        "paired_win_fraction": paired_win_fraction,
        "paired_nonloss_fraction": paired_nonloss_fraction,
        "anchor_no_progress_rate": anchor_no_progress_rate,
        "challenger_no_progress_rate": challenger_no_progress_rate,
        "pre_residual_positive": pre_residual_positive,
    }


def _grid_state_path(state_root: Path, row: dict[str, Any]) -> Path:
    filename = PurePosixPath(str(row["state_file"]).replace("\\", "/")).name
    path = state_root / filename
    if not path.is_file() or sha256_file(path) != str(row["state_file_sha256"]):
        raise ValueError(f"SafeSlot registered grid state changed: {path}")
    return path


def analyze_safeslot_label_readiness(
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = _read_json(config_path)
    validate_safeslot_label_readiness_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }

    grid_report = _read_json(inputs["grid_report"])
    grid_label_report = _read_json(inputs["grid_label_report"])
    base_label_report = _read_json(inputs["base_label_report"])
    slotpool_report = _read_json(inputs["slotpool_report"])
    if (
        grid_report.get("passed") is not True
        or grid_report.get("candidate_repair_trials_executed") is not False
        or grid_label_report.get("passed") is not True
        or base_label_report.get("integrity_passed") is not True
        or dict(slotpool_report.get("acceptance") or {}).get("passed") is not True
    ):
        raise ValueError("SafeSlot registered source report did not pass")

    cohort = dict(config["cohort"])
    grid_manifest = _read_jsonl(inputs["grid_manifest"])
    grid_aggregates = _read_jsonl(inputs["grid_candidate_aggregates"])
    grid_trials = _read_jsonl(inputs["grid_repair_trials"])
    base_aggregates = _read_jsonl(inputs["base_candidate_aggregates"])
    base_trials = _read_jsonl(inputs["base_repair_trials"])
    slotpool_states = _read_jsonl(inputs["slotpool_state_evaluation"])
    if (
        len(grid_manifest) != int(cohort["state_count"])
        or len(grid_aggregates) != int(cohort["grid_candidate_count"])
        or len(grid_trials) != int(cohort["grid_trial_count"])
        or len(slotpool_states) != int(cohort["state_count"])
        or sum(int(row["selected_candidate_count"]) for row in slotpool_states)
        != int(cohort["slotpool_selected_candidate_count"])
    ):
        raise ValueError("SafeSlot registered source product changed")

    grid_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in grid_aggregates:
        if row.get("candidate_kind") != "structpool-grid":
            raise ValueError("SafeSlot grid contains a non-structural candidate")
        grid_by_state[str(row["state_id"])].append(row)
    base_by_action = {
        (str(row["state_id"]), str(row["candidate_id"])): row
        for row in base_aggregates
    }
    grid_trials_by_action = _trials_by_action(grid_trials)
    base_trials_by_action = _trials_by_action(base_trials)
    slotpool_by_state = {str(row["state_id"]): row for row in slotpool_states}
    if len(slotpool_by_state) != int(cohort["state_count"]):
        raise ValueError("SafeSlot SlotPool state IDs are not unique")

    state_root = (
        project_root / str(config["grid_state_artifact_root"])
    ).resolve()
    label_contract = dict(config["paired_pre_residual_label"])
    comparisons: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    positive_maps: set[str] = set()
    positive_candidate_count = 0
    retained_positive_candidate_count = 0
    positive_state_count = 0
    retained_positive_state_count = 0
    map_summary: dict[str, Counter[str]] = defaultdict(Counter)

    for manifest_row in sorted(grid_manifest, key=lambda row: str(row["state_id"])):
        state_id = str(manifest_row["state_id"])
        if state_id in seen_states:
            raise ValueError(f"duplicate SafeSlot grid state: {state_id}")
        seen_states.add(state_id)
        state_payload = _read_json(_grid_state_path(state_root, manifest_row))
        anchor = dict(state_payload.get("v2_base_anchor") or {})
        anchor_id = str(anchor.get("candidate_id") or "")
        anchor_key = (state_id, anchor_id)
        anchor_aggregate = base_by_action.get(anchor_key)
        anchor_trials = base_trials_by_action.get(anchor_key)
        if (
            not anchor_id
            or anchor_aggregate is None
            or anchor_trials is None
            or anchor_aggregate.get("candidate_kind") != "base"
            or bool(anchor.get("candidate_repair_outcomes_read"))
        ):
            raise ValueError(f"SafeSlot exact base-only V2 anchor is unavailable: {state_id}")

        grid_rows = sorted(
            grid_by_state.get(state_id) or [], key=lambda row: str(row["candidate_id"])
        )
        expected_grid_ids = {
            str(candidate["candidate_id"])
            for candidate in list(state_payload.get("candidates") or ())
        }
        if {str(row["candidate_id"]) for row in grid_rows} != expected_grid_ids:
            raise ValueError(f"SafeSlot grid aggregate identity differs: {state_id}")
        slotpool_row = slotpool_by_state.get(state_id)
        if slotpool_row is None:
            raise ValueError(f"SafeSlot SlotPool row is missing: {state_id}")
        retained_ids = set(map(str, slotpool_row["selected_candidate_ids"]))
        if not retained_ids <= expected_grid_ids:
            raise ValueError(f"SafeSlot SlotPool selected an unknown candidate: {state_id}")

        state_positive = 0
        state_retained_positive = 0
        for candidate in grid_rows:
            candidate_id = str(candidate["candidate_id"])
            candidate_trials = grid_trials_by_action.get((state_id, candidate_id))
            if candidate_trials is None:
                raise ValueError(f"SafeSlot challenger trials are missing: {state_id}")
            metrics = compare_paired_current_step_trials(
                candidate_trials, anchor_trials, label_contract
            )
            retained = candidate_id in retained_ids
            positive = bool(metrics["pre_residual_positive"])
            if positive:
                positive_candidate_count += 1
                state_positive += 1
                positive_maps.add(str(candidate["map_id"]))
                if retained:
                    retained_positive_candidate_count += 1
                    state_retained_positive += 1
            comparisons.append(
                {
                    "schema": COMPARISON_SCHEMA,
                    "state_id": state_id,
                    "map_id": str(candidate["map_id"]),
                    "layout_mode": str(candidate["layout_mode"]),
                    "source_policy": str(candidate["source_policy"]),
                    "anchor_candidate_id": anchor_id,
                    "anchor_candidate_kind": "base",
                    "challenger_candidate_id": candidate_id,
                    "challenger_actual_size": int(candidate["actual_size"]),
                    "challenger_structural_groups": list(
                        candidate.get("structpool_family_groups") or ()
                    ),
                    "slotpool_oof_retained": retained,
                    **metrics,
                    "residual_teacher_available": False,
                    "final_safe_replace_label_available": False,
                }
            )

        has_positive = state_positive > 0
        retained_positive = state_retained_positive > 0
        if has_positive:
            positive_state_count += 1
            if retained_positive:
                retained_positive_state_count += 1
        map_id = str(manifest_row["map_id"])
        map_summary[map_id]["state_count"] += 1
        map_summary[map_id]["positive_state_count"] += int(has_positive)
        map_summary[map_id]["positive_candidate_count"] += state_positive
        map_summary[map_id]["retained_positive_candidate_count"] += (
            state_retained_positive
        )
        state_rows.append(
            {
                "schema": STATE_SCHEMA,
                "state_id": state_id,
                "map_id": map_id,
                "layout_mode": str(manifest_row["layout_mode"]),
                "source_policy": str(manifest_row["source_policy"]),
                "anchor_candidate_id": anchor_id,
                "grid_candidate_count": len(grid_rows),
                "slotpool_selected_candidate_count": len(retained_ids),
                "pre_residual_positive_candidate_count": state_positive,
                "slotpool_retained_positive_candidate_count": state_retained_positive,
                "has_pre_residual_positive": has_positive,
                "slotpool_retains_pre_residual_opportunity": retained_positive,
                "residual_teacher_available": False,
            }
        )

    if len(seen_states) != int(cohort["state_count"]):
        raise ValueError("SafeSlot state coverage changed")
    observed_maps = {str(row["map_id"]) for row in state_rows}
    if len(observed_maps) != int(cohort["map_count"]):
        raise ValueError("SafeSlot map coverage changed")

    retention = (
        retained_positive_state_count / positive_state_count
        if positive_state_count
        else 0.0
    )
    gates = dict(config["readiness_gates"])
    checks = {
        "pre_residual_positive_state_count": positive_state_count
        >= int(gates["minimum_pre_residual_positive_state_count"]),
        "pre_residual_positive_candidate_count": positive_candidate_count
        >= int(gates["minimum_pre_residual_positive_candidate_count"]),
        "positive_map_count": len(positive_maps)
        >= int(gates["minimum_positive_map_count"]),
        "slotpool_positive_state_retention": retention
        >= float(gates["minimum_slotpool_positive_state_retention"]),
        "slotpool_retained_positive_candidate_count": retained_positive_candidate_count
        >= int(gates["minimum_slotpool_retained_positive_candidate_count"]),
    }
    passed = all(checks.values())

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    outputs = dict(config["outputs"])
    comparisons_path = output / str(outputs["comparison_rows"])
    states_path = output / str(outputs["state_rows"])
    _write_jsonl(comparisons_path, comparisons)
    _write_jsonl(states_path, state_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": (
            "pre_residual_readiness_passed_post_state_collection_only"
            if passed
            else "pre_residual_readiness_failed_stop_before_post_state_collection"
        ),
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "integrity": {
            "registered_inputs": True,
            "grid_state_tree": True,
            "exact_base_only_v2_anchor": True,
            "all_states_and_maps": True,
            "exact_grid_candidate_and_trial_product": True,
            "exact_state_trial_pp_seed_pairing": True,
            "slotpool_oof_identity": True,
            "no_runtime_ttf_or_future_read": True,
        },
        "integrity_passed": True,
        "state_count": len(state_rows),
        "map_count": len(observed_maps),
        "candidate_count": len(comparisons),
        "trial_pair_count": len(comparisons) * 16,
        "pre_residual_opportunity": {
            "positive_state_count": positive_state_count,
            "positive_state_fraction": positive_state_count / len(state_rows),
            "positive_candidate_count": positive_candidate_count,
            "positive_candidate_fraction": positive_candidate_count
            / len(comparisons),
            "positive_map_count": len(positive_maps),
            "positive_maps": sorted(positive_maps),
        },
        "slotpool_retention": {
            "selected_candidate_count": sum(
                int(row["slotpool_selected_candidate_count"]) for row in state_rows
            ),
            "retained_positive_state_count": retained_positive_state_count,
            "positive_state_retention": retention,
            "retained_positive_candidate_count": retained_positive_candidate_count,
            "positive_candidate_retention": (
                retained_positive_candidate_count / positive_candidate_count
                if positive_candidate_count
                else 0.0
            ),
        },
        "by_map": {
            map_id: dict(counts) for map_id, counts in sorted(map_summary.items())
        },
        "readiness_checks": checks,
        "readiness_passed": passed,
        "residual_teacher_available": False,
        "final_safe_replace_labels_available": False,
        "post_state_collection_authorized": passed,
        "model_training_allowed": False,
        "runtime_integration_allowed": False,
        "formal_ttf_claim": False,
        "next_decision": (
            "preregister_paired_one_step_post_repair_residual_teacher_collection"
            if passed
            else "stop_and_reassess_label_or_candidate_pool_without_outcome_filtering"
        ),
        "artifacts": {
            "comparison_rows_sha256": sha256_file(comparisons_path),
            "state_rows_sha256": sha256_file(states_path),
        },
    }
    _write_json(output / str(outputs["report"]), report)
    return report
