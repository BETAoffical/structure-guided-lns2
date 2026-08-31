from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import closed_loop_producer_identity, registered_input, sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import _fingerprint, _read_json, _read_jsonl, _write_json
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.stride_maze_tail_full_episode import (
    CONTROLLERS,
    _controller_kwargs,
    _expected_keys,
    _pool_counts,
    _task_metadata,
    load_maze_tail_full_episode_config,
    maze_tail_full_episode_schedule,
)
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA
from lns2_selector.compatibility.retired_pool_profiles import (
    slotpool_runtime_augmentation,
)
from lns2_selector.evaluation.episode_statistics import mean as _mean
from lns2_selector.runtime.online_selection import (
    validate_structpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.maze_tail_state_collection_config.v2"
STATUS_SCHEMA = "lns2.stride.maze_tail_state_collection_status.v2"
REPORT_SCHEMA = "lns2.stride.maze_tail_state_collection_report.v2"
CHALLENGERS = CONTROLLERS[1:]
STATUS_FILENAME = "state_collection_status.json"
REPORT_FILENAME = "maze_tail_state_collection_report.json"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="Maze tail state collection")


def load_maze_tail_state_collection_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_fused_state_collection_revision_2_after_zero_episode_reset_protocol_rejection"
        or config.get("experiment_id") != "stride-maze-tail-state-collection-v2-r2"
        or config.get("pre_registration_parent_commit")
        != "f8da77aa2c0aaa2355ec0b102462246b05d71552"
        or config.get("pre_registration_revision_parent_commit")
        != "ebff095c5de296aca849f4a6c4e1634c53046ad9"
        or config.get("pre_registration_revision_reason")
        != "restore_qualification_compatible_native_unlimited_repair_while_preserving_outer_200_decision_and_300_second_fuse_before_any_episode"
        or config.get("pre_registration_revision_2_parent_commit")
        != "eb93fba90d40e95602830b85753c6c0fd970b9bc"
        or config.get("pre_registration_revision_2_reason")
        != "preserve_unlimited_native_time_in_reset_protocol_and_read_300_second_fuse_from_runtime_outer_loop_before_any_episode"
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("Maze tail state-collection identity changed")
    runtime_expected = {
        "config": "configs/stride_maze_tail_state_collection_runtime_v2.json",
        "stopping_rule": "historical",
        "maximum_repair_decisions": 200,
        "metric_iteration_budget": 200,
        "wall_time_budget_seconds": 300.0,
        "native_environment_time_limit_seconds": None,
        "native_environment_unlimited": True,
        "episode_process_timeout_seconds": 360.0,
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "deterministic_pp_replay": True,
        "workers": 1,
    }
    if dict(config.get("runtime") or {}) != runtime_expected:
        raise ValueError("Maze tail state-collection runtime changed")
    if dict(config.get("evidence_fuse") or {}) != {
        "maximum_repair_decisions": 200,
        "maximum_reset_inclusive_wall_seconds": 300.0,
        "normal_completion_before_fuse_is_success": True,
        "repair_limit_or_wall_timeout_is_valid_right_censoring": True,
        "external_process_timeout_is_execution_error": True,
        "right_censored_episode_is_not_solver_failure_claim": True,
    }:
        raise ValueError("Maze tail evidence fuse changed")
    if dict(config.get("fixed_horizon_tail_definition") or {}) != {
        "complete_pair_adverse_minimum_iteration_delta": 1,
        "complete_pair_severe_minimum_iteration_delta": 10,
        "complete_pair_severe_minimum_iteration_ratio": 2.0,
        "both_censored_adverse_minimum_normalized_auc_delta": 0.05,
        "both_censored_severe_minimum_normalized_auc_delta": 0.15,
        "both_censored_adverse_minimum_final_conflict_delta": 5,
        "challenger_censored_v2_complete_is_severe": True,
        "challenger_complete_v2_censored_is_beneficial": True,
    }:
        raise ValueError("Maze tail fixed-horizon definition changed")
    if dict(config.get("claim_boundary") or {}) != {
        "state_condition_collection_only": True,
        "run_to_feasibility_claim": False,
        "formal_speed_claim": False,
        "fresh_map_generalization_claim": False,
        "default_replacement_allowed": False,
        "training_allowed": False,
        "no_result_based_exclusions": True,
        "censored_ttf_is_not_imputed": True,
    }:
        raise ValueError("Maze tail state-collection claim boundary changed")
    v1_path = root / "configs" / "stride_maze_tail_full_episode_v1.json"
    _v1_path, _v1_root, v1 = load_maze_tail_full_episode_config(v1_path)
    if (
        config.get("cohort") != v1.get("cohort")
        or config.get("controller_bundle") != v1.get("controller_bundle")
        or config.get("full_structpool_augmentation")
        != v1.get("full_structpool_augmentation")
    ):
        raise ValueError("Maze tail state-collection cohort or treatment changed")
    if dict(config.get("comparison") or {}) != {
        "baseline": "frozen_v2_over_original_candidate_pool",
        "challengers": [
            "frozen_v2_over_original_plus_full_structpool",
            "frozen_v2_over_original_plus_frozen_slotpool",
        ],
        "execution_order": "strict_three_controller_rotation",
        "paired_solver_seed_required": True,
        "workers": 1,
        "primary_role": "fixed_horizon_state_condition_collection_not_run_to_feasibility",
    }:
        raise ValueError("Maze tail state-collection comparison changed")
    full = validate_structpool_augmentation(dict(config["full_structpool_augmentation"]))
    slot = slotpool_runtime_augmentation()
    if (
        full is None
        or full.get("pool_id") != "stride-structpool-v1"
        or slot.get("pool_id") != "stride-slotpool-v1"
    ):
        raise ValueError("Maze tail state-collection treatments changed")
    expected_inputs = {
        "dataset_summary",
        "dataset_manifest",
        "runtime_config",
        "controller_manifest",
        "slotpool_model",
        "qualification_manifest",
        "qualification_report",
        "qualification_run_config",
        "selected_cohort",
        "preflight_report",
        "v1_stop_report",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("Maze tail state-collection input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    runtime = _read_json(inputs["runtime_config"])
    if (
        runtime.get("max_decisions") != 200
        or runtime.get("metric_iteration_budget") != 200
        or runtime.get("wall_time_budget_seconds") != 300.0
        or runtime.get("episode_process_timeout_seconds") != 360.0
        or runtime.get("deterministic_pp_replay") is not True
        or dict(runtime.get("environment") or {}).get("time_limit") != 0.0
        or dict(runtime.get("environment") or {}).get("unlimited_time") is not True
        or dict(runtime.get("environment") or {}).get("max_repair_iterations") != 0
    ):
        raise ValueError("Maze tail fused runtime file changed")
    if str(slot["slotpool_model"]["sha256"]) != str(
        config["inputs"]["slotpool_model"]["sha256"]
    ):
        raise ValueError("Maze tail state-collection SlotPool model changed")
    selected = _read_jsonl(inputs["selected_cohort"])
    configured_tasks = {
        str(task)
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
    }
    if {str(row["task_id"]) for row in selected} != configured_tasks:
        raise ValueError("Maze tail state-collection selected tasks changed")
    return path, root, config


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_maze_tail_state_collection.py",
            "experiments/stride_maze_tail_full_episode.py",
            "lns2_selector/compatibility/retired_pool_profiles.py",
            "lns2_selector/evaluation/episode_statistics.py",
        ),
        native_required=native_required,
    )


def _fused_controller_kwargs(
    root: Path, config: dict[str, Any], controller: str
) -> dict[str, Any]:
    result = _controller_kwargs(root, config, controller)
    result["stopping_rule"] = "historical"
    return result


def _completed_counts(output: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for controller in CONTROLLERS:
        manifest = output / "controllers" / controller / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(manifest) if manifest.is_file() else []
        result[controller] = sum(row.get("status") in {"ok", "error", "timeout"} for row in rows)
    return result


def run_maze_tail_state_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_maze_tail_state_collection_config(config_path)
    schedule = maze_tail_full_episode_schedule(config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "paired_key_count": len(schedule) // 3,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
            "maximum_repair_decisions": 200,
            "wall_time_budget_seconds": 300.0,
        }
    output = Path(output).resolve()
    status_path = output / STATUS_FILENAME
    prepared = prepare_resumable_output(
        output,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=schedule,
        producer=_producer(root),
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="Maze tail fused state collection",
    )
    status_base = prepared.base_status
    if prepared.completed_report is not None:
        return prepared.completed_report
    dataset = (root / str(config["cohort"]["dataset"])).resolve()
    runtime_path = (root / str(config["runtime"]["config"])).resolve()
    expected = _expected_keys(config)
    job_keys = {(task, seed) for _group, task, seed in expected}
    qualification_source = (
        root / str(config["inputs"]["qualification_manifest"]["path"])
    ).resolve().parent
    for controller in CONTROLLERS:
        collection = output / "controllers" / controller
        run_closed_loop_collection(
            dataset,
            runtime_path,
            collection,
            phase="qualify",
            workers=1,
            resume=(prepared.resumed and collection.joinpath("run_config.json").is_file()),
            cohort_job_keys=job_keys,
            job_keys=job_keys,
            qualification_source=qualification_source,
            **_fused_controller_kwargs(root, config, controller),
        )
    completed = 0
    for item in schedule:
        controller = str(item["controller"])
        collection = output / "controllers" / controller
        manifest = collection / "realized_dynamic_manifest.jsonl"
        done = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in (_read_jsonl(manifest) if manifest.is_file() else [])
            if row.get("status") in {"ok", "error", "timeout"}
        }
        key = (str(item["task_id"]), int(item["solver_seed"]))
        if key not in done:
            run_closed_loop_collection(
                dataset,
                runtime_path,
                collection,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=job_keys,
                job_keys={key},
                **_fused_controller_kwargs(root, config, controller),
            )
        completed += 1
        _write_json(
            status_path,
            {
                **status_base,
                "completed_schedule_entries": completed,
                "controller_completed_episode_counts": _completed_counts(output),
                "current": item,
                "complete": False,
            },
        )
    report = analyze_maze_tail_state_collection(
        path, output, producer=status_base["producer_identity"]
    )
    _write_json(
        status_path,
        {
            **status_base,
            "completed_schedule_entries": completed,
            "controller_completed_episode_counts": _completed_counts(output),
            "complete": True,
            "report_sha256": sha256_file(output / REPORT_FILENAME),
        },
    )
    return report


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [dict(row.get("summary") or {}) for row in rows if row.get("status") == "ok"]
    successes = [row for row in summaries if bool(row.get("success"))]
    censored = [row for row in summaries if not bool(row.get("success"))]
    return {
        "episode_count": len(rows),
        "ok_count": len(summaries),
        "execution_error_count": len(rows) - len(summaries),
        "success_count": len(successes),
        "right_censored_count": len(censored),
        "repair_limit_count": sum(row.get("stop_reason") == "repair_limit" for row in censored),
        "wall_timeout_count": sum(row.get("stop_reason") == "wall_timeout" for row in censored),
        "mean_repair_iterations": _mean([float(row.get("repair_iterations", 0)) for row in summaries]),
        "mean_final_conflicts": _mean([float(row.get("final_conflicts", 0)) for row in summaries]),
        "mean_normalized_fixed_budget_conflict_auc": _mean(
            [float(row.get("normalized_fixed_budget_conflict_auc", 0.0)) for row in summaries]
        ),
        "mean_observed_wall_seconds": _mean(
            [float(row.get("ttf_observed_wall_seconds", 0.0)) for row in summaries]
        ),
        "mean_success_ttf": _mean(
            [float(row["wall_time_to_feasible"]) for row in successes]
        ),
        "invalid_action_count": sum(int(row.get("invalid_action_count", 0)) for row in summaries),
        "fingerprint_mismatch_count": sum(
            int(row.get("fingerprint_mismatch_count", 0)) for row in summaries
        ),
    }


def _classify_pair(
    definition: dict[str, Any], baseline: dict[str, Any], challenger: dict[str, Any]
) -> tuple[str, str]:
    left_success = bool(baseline.get("success"))
    right_success = bool(challenger.get("success"))
    if left_success and not right_success:
        return "severe", "challenger_censored_v2_complete"
    if not left_success and right_success:
        return "beneficial", "challenger_complete_v2_censored"
    if left_success and right_success:
        left_iterations = int(baseline.get("repair_iterations", 0))
        right_iterations = int(challenger.get("repair_iterations", 0))
        delta = right_iterations - left_iterations
        ratio = right_iterations / max(left_iterations, 1)
        if (
            delta >= int(definition["complete_pair_severe_minimum_iteration_delta"])
            and ratio >= float(definition["complete_pair_severe_minimum_iteration_ratio"])
        ):
            return "severe", "both_complete_repair_iteration_tail"
        if delta >= int(definition["complete_pair_adverse_minimum_iteration_delta"]):
            return "adverse", "both_complete_more_repair_iterations"
        return "beneficial_or_tied", "both_complete_noninferior_iterations"
    auc_delta = float(challenger["normalized_fixed_budget_conflict_auc"]) - float(
        baseline["normalized_fixed_budget_conflict_auc"]
    )
    conflict_delta = int(challenger.get("final_conflicts", 0)) - int(
        baseline.get("final_conflicts", 0)
    )
    if (
        auc_delta
        >= float(definition["both_censored_severe_minimum_normalized_auc_delta"])
        and conflict_delta
        >= int(definition["both_censored_adverse_minimum_final_conflict_delta"])
    ):
        return "severe", "both_censored_fixed_horizon_severe"
    if (
        auc_delta
        >= float(definition["both_censored_adverse_minimum_normalized_auc_delta"])
        or conflict_delta
        >= int(definition["both_censored_adverse_minimum_final_conflict_delta"])
    ):
        return "adverse", "both_censored_fixed_horizon_worse"
    return "inconclusive", "both_censored_no_registered_adverse_margin"


def analyze_maze_tail_state_collection(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_maze_tail_state_collection_config(config_path)
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename=STATUS_FILENAME,
        report_filename=REPORT_FILENAME,
        status_schema=STATUS_SCHEMA,
        report_schema=REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    producer = producer or _producer(root, native_required=False)
    expected = _expected_keys(config)
    metadata = _task_metadata(
        _read_jsonl(_registered(root, dict(config["inputs"]["selected_cohort"])))
    )
    qualification = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in _read_jsonl(
            _registered(root, dict(config["inputs"]["qualification_manifest"]))
        )
        if str(row["task_id"]) in metadata
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    counts = {controller: defaultdict(int) for controller in CONTROLLERS}
    errors: list[str] = []
    for controller in CONTROLLERS:
        manifest = output / "controllers" / controller / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(manifest)
        hashes[controller] = sha256_file(manifest)
        by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for row in rows:
            task = str(row["task_id"])
            meta = metadata.get(task)
            if meta is None:
                errors.append(f"{controller}: unexpected task {task}")
                continue
            key = (str(meta["map_id"]), task, int(row["solver_seed"]))
            if key not in expected:
                errors.append(f"{controller}: unexpected key {key}")
                continue
            if key in by_key:
                errors.append(f"{controller}: duplicate key {key}")
            by_key[key] = row
            for name, value in _pool_counts(row).items():
                counts[controller][name] += value
        if set(by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = by_key
    fingerprint_mismatches = conflict_mismatches = qualification_mismatches = 0
    bad_clock = bad_budget = bad_stop = process_failures = 0
    for key in sorted(expected):
        rows = [indexed[controller].get(key) for controller in CONTROLLERS]
        if any(row is None for row in rows):
            continue
        process_failures += sum(row.get("status") != "ok" for row in rows)
        if any(row.get("status") != "ok" for row in rows):
            continue
        summaries = [dict(row.get("summary") or {}) for row in rows]
        fingerprint_mismatches += len(
            {str(row.get("initial_fingerprint")) for row in summaries}
        ) != 1
        conflict_mismatches += len(
            {int(row.get("initial_conflicts", -1)) for row in summaries}
        ) != 1
        registered = qualification[(key[1], key[2])]
        qualification_mismatches += any(
            str(row.get("initial_fingerprint")) != str(registered["state_fingerprint"])
            or int(row.get("initial_conflicts", -1))
            != int(registered["initial_conflicts"])
            for row in summaries
        )
        bad_clock += sum(row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries)
        bad_budget += sum(
            int(row.get("repair_iterations", 0)) > 200
            or row.get("metric_iteration_budget") != 200
            or row.get("wall_time_budget_seconds") != 300.0
            or row.get("normalized_fixed_budget_conflict_auc") is None
            for row in summaries
        )
        bad_stop += sum(
            (not bool(row.get("success")))
            and row.get("stop_reason") not in {"repair_limit", "wall_timeout"}
            for row in summaries
        )
    controller_summaries = {
        controller: _summary(list(indexed[controller].values()))
        for controller in CONTROLLERS
    }
    definition = dict(config["fixed_horizon_tail_definition"])
    pair_rows: list[dict[str, Any]] = []
    if not errors and process_failures == 0:
        for challenger in CHALLENGERS:
            for key in sorted(expected):
                baseline = dict(indexed["v2-full"][key]["summary"])
                treatment = dict(indexed[challenger][key]["summary"])
                category, reason = _classify_pair(definition, baseline, treatment)
                meta = metadata[key[1]]
                pair_rows.append(
                    {
                        "challenger": challenger,
                        "map_id": key[0],
                        "task_id": key[1],
                        "solver_seed": key[2],
                        "task_variant_family": meta["task_variant_family"],
                        "conflict_band": meta["conflict_band"],
                        "agent_count": int(meta["agent_count"]),
                        "initial_conflicts": int(baseline["initial_conflicts"]),
                        "v2_success": bool(baseline["success"]),
                        "challenger_success": bool(treatment["success"]),
                        "v2_stop_reason": baseline["stop_reason"],
                        "challenger_stop_reason": treatment["stop_reason"],
                        "v2_repair_iterations": int(baseline["repair_iterations"]),
                        "challenger_repair_iterations": int(treatment["repair_iterations"]),
                        "v2_final_conflicts": int(baseline["final_conflicts"]),
                        "challenger_final_conflicts": int(treatment["final_conflicts"]),
                        "v2_normalized_fixed_auc": float(
                            baseline["normalized_fixed_budget_conflict_auc"]
                        ),
                        "challenger_normalized_fixed_auc": float(
                            treatment["normalized_fixed_budget_conflict_auc"]
                        ),
                        "v2_observed_wall_seconds": float(
                            baseline["ttf_observed_wall_seconds"]
                        ),
                        "challenger_observed_wall_seconds": float(
                            treatment["ttf_observed_wall_seconds"]
                        ),
                        "tail_category": category,
                        "classification_reason": reason,
                    }
                )
    adverse = [row for row in pair_rows if row["tail_category"] in {"adverse", "severe"}]
    severe = [row for row in pair_rows if row["tail_category"] == "severe"]
    adverse_per_challenger = {
        challenger: sum(row["challenger"] == challenger for row in adverse)
        for challenger in CHALLENGERS
    }
    thresholds = dict(config["tail_evidence_gates"])
    evidence_gates = {
        "minimum_adverse_or_severe_comparisons": len(adverse)
        >= int(thresholds["minimum_adverse_or_severe_comparisons"]),
        "minimum_severe_tail_comparisons": len(severe)
        >= int(thresholds["minimum_severe_tail_comparisons"]),
        "minimum_adverse_map_count": len({row["map_id"] for row in adverse})
        >= int(thresholds["minimum_adverse_map_count"]),
        "minimum_adverse_task_count": len({row["task_id"] for row in adverse})
        >= int(thresholds["minimum_adverse_task_count"]),
        "minimum_adverse_solver_seed_count": len({row["solver_seed"] for row in adverse})
        >= int(thresholds["minimum_adverse_solver_seed_count"]),
        "minimum_adverse_comparisons_per_challenger": all(
            count >= int(thresholds["minimum_adverse_comparisons_per_challenger"])
            for count in adverse_per_challenger.values()
        ),
    }
    integrity = {
        "complete_three_controller_coverage": not any("coverage" in error for error in errors),
        "zero_process_errors_or_timeouts": process_failures == 0,
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "registered_qualification_states_reused": qualification_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "fixed_horizon_metrics_complete": bad_budget == 0,
        "only_registered_censoring_reasons": bad_stop == 0,
        "zero_invalid_actions": all(
            value["invalid_action_count"] == 0 for value in controller_summaries.values()
        ),
        "zero_semantic_mismatches": all(
            value["fingerprint_mismatch_count"] == 0
            for value in controller_summaries.values()
        ),
        "full_structpool_activated": counts["v2-plus-structpool"]["gate_passed"] > 0,
        "full_structpool_selected": counts["v2-plus-structpool"]["structural_selected"] > 0,
        "slotpool_activated": counts["v2-plus-slotpool"]["gate_passed"] > 0,
        "slotpool_selected": counts["v2-plus-slotpool"]["structural_selected"] > 0,
        "slotpool_reduction_exercised": counts["v2-plus-slotpool"]["slotpool_reduced"] > 0,
    }
    integrity_passed = not errors and all(integrity.values())
    evidence_passed = integrity_passed and all(evidence_gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "fused_maze_tail_state_collection_complete",
        "producer_identity": producer,
        "state_condition_collection_only": True,
        "run_to_feasibility_claim": False,
        "formal_speed_claim": False,
        "training_allowed": False,
        "right_censoring_is_not_solver_failure_claim": True,
        "paired_key_count": len(expected),
        "episode_count_per_controller": len(expected),
        "comparison_count": len(pair_rows),
        "controller_summaries": controller_summaries,
        "tail_counts": {
            "adverse_or_severe": len(adverse),
            "severe": len(severe),
            "adverse_map_count": len({row["map_id"] for row in adverse}),
            "adverse_task_count": len({row["task_id"] for row in adverse}),
            "adverse_solver_seed_count": len({row["solver_seed"] for row in adverse}),
            "adverse_per_challenger": adverse_per_challenger,
        },
        "tail_comparisons": pair_rows,
        "runtime_counts": {controller: dict(value) for controller, value in counts.items()},
        "integrity_gates": integrity,
        "tail_evidence_gates": evidence_gates,
        "integrity_passed": integrity_passed,
        "tail_evidence_passed": evidence_passed,
        "next_step": (
            config["next_step_on_tail_evidence_pass"]
            if evidence_passed
            else config["next_step_on_tail_evidence_failure"]
        ),
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "controller_manifest_sha256": hashes,
        },
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


__all__ = [
    "REPORT_FILENAME",
    "STATUS_FILENAME",
    "analyze_maze_tail_state_collection",
    "load_maze_tail_state_collection_config",
    "run_maze_tail_state_collection",
]
