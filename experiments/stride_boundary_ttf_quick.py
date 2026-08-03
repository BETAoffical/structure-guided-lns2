from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_stage4r_quick import _controller_summary, _mean


CONFIG_SCHEMA = "lns2.stride.boundary_ttf_quick_config.v1"
REPORT_SCHEMA = "lns2.stride.boundary_ttf_quick_report.v1"
STATUS_SCHEMA = "lns2.stride.boundary_ttf_quick_status.v1"
CONTROLLERS = ("v2-full", "v2-boundary-explore-v1")
TTF_CLOCK_SCHEMA = "lns2.ttf.reset_inclusive_wall.v1"


def _registered_path(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"boundary TTF Quick input SHA differs: {specification['path']}")
    return path


def validate_boundary_ttf_quick_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected boundary TTF Quick config")
    experiment_id = str(config.get("experiment_id"))
    original = experiment_id == "stride-boundary-ttf-quick-v1"
    optimization = experiment_id == "stride-boundary-runtime-optimization-quick-v1"
    if not original and not optimization:
        raise ValueError("unsupported boundary TTF Quick identity")
    expected_status = (
        "exploratory_paired_runtime_quick_after_offline_shadow"
        if original
        else "outcome_informed_same_cohort_boundary_runtime_optimization_quick"
    )
    expected_controllers = (
        CONTROLLERS
        if original
        else ("v2-full", "stride-boundary-phase-guard-v2")
    )
    if (
        config.get("scientific_status") != expected_status
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("training_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("cohort_independent_of_shadow_outcomes"))
    ):
        raise ValueError("boundary TTF Quick must remain exploratory and non-promoting")
    if (
        config.get("primary_metric") != "mean_capped_wall_time_to_feasible"
        or config.get("success_constraint")
        != "challenger_success_count_gte_v2_full"
        or config.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA
        or tuple(map(str, config.get("controllers") or ())) != expected_controllers
        or config.get("executed_controller") != "v2-full"
        or config.get("controller_bundle")
        != "artifacts/initlns-closed-loop-controller-v2"
        or int(config.get("expected_state_count", -1)) != 18
        or int(config.get("expected_schedule_entry_count", -1)) != 36
        or float(config.get("wall_time_budget_seconds", 0.0)) != 60.0
        or float(config.get("environment_time_limit_seconds", 0.0)) != 60.0
        or float(config.get("episode_process_timeout_seconds", 0.0)) != 90.0
        or int(config.get("workers", 0)) != 1
        or config.get("execution_order") != "strict_alternating_pair_order"
        or not bool(config.get("deterministic_pp_replay_required"))
    ):
        raise ValueError("boundary TTF Quick runtime contract changed")
    jobs = [
        (str(value[0]), int(value[1]))
        for value in config.get("registered_job_keys") or []
    ]
    if len(jobs) != 18 or len(set(jobs)) != 18 or any(seed not in {1, 2} for _, seed in jobs):
        raise ValueError("boundary TTF Quick cohort changed")
    if dict(config.get("expected_state_count_by_group") or {}) != {
        "dao_low_articulation_control": 4,
        "dao_articulated": 8,
        "dao_ultra_bottleneck": 6,
    }:
        raise ValueError("boundary TTF Quick group registry changed")
    expected_augmentation: dict[str, Any] = {
        "enabled": True,
        "generator_id": "stride-topoboundary-v1",
        "neighborhood_size": 16,
        "core_budget": 4,
        "maximum_added_candidates": 2,
    }
    if optimization:
        expected_augmentation.update(
            {
                "runtime_id": "stride-boundary-phase-guard-v2",
                "static_grid_cache": True,
                "activation_gate": {
                    "gate_id": "stride-boundary-map-topology-v1",
                    "minimum_low_degree_cell_ratio": 0.06,
                },
                "phase_guard": {
                    "gate_id": "stride-boundary-phase-guard-v2",
                    "low_conflict_pair_threshold": 2,
                    "low_conflict_no_progress_streak": 2,
                    "maximum_no_progress_streak": 5,
                    "minimum_remaining_wall_seconds": 5.0,
                },
            }
        )
    if dict(config.get("topology_boundary_augmentation") or {}) != expected_augmentation:
        raise ValueError("boundary TTF Quick candidate augmentation changed")
    if dict(config.get("continuation_gates") or {}) != {
        "minimum_capped_ttf_relative_improvement": 0.05,
        "repair_iterations_noninferior": True,
        "maximum_relevant_group_capped_ttf_regression": 0.10,
        "maximum_control_group_capped_ttf_regression": 0.20,
        "minimum_boundary_selected_repair_count": 1,
    }:
        raise ValueError("boundary TTF Quick continuation gates changed")
    expected_inputs = {
        "runtime_config",
        "coverage_state_rows",
        "dataset_manifest",
        "qualification_manifest",
        "qualification_report",
        "shadow_report",
        "shadow_selections",
        "controller_manifest",
    }
    if optimization:
        expected_inputs.add("predecessor_report")
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("boundary TTF Quick input registry changed")


def boundary_ttf_quick_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    jobs = sorted(
        (str(value[0]), int(value[1])) for value in config["registered_job_keys"]
    )
    schedule = []
    for key_index, (task_id, solver_seed) in enumerate(jobs):
        controllers = tuple(map(str, config["controllers"]))
        order = controllers if key_index % 2 == 0 else tuple(reversed(controllers))
        for position, controller in enumerate(order):
            schedule.append(
                {
                    "ordinal": len(schedule),
                    "task_id": task_id,
                    "solver_seed": solver_seed,
                    "controller": controller,
                    "within_key_position": position,
                }
            )
    return schedule


def _registered_inputs(
    config_path: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Path]]:
    project_root = config_path.parents[1]
    return project_root, {
        name: _registered_path(project_root, dict(specification))
        for name, specification in config["inputs"].items()
    }


def _cohort_metadata(
    config: dict[str, Any], inputs: dict[str, Path]
) -> dict[tuple[str, int], dict[str, Any]]:
    expected = {
        (str(value[0]), int(value[1])) for value in config["registered_job_keys"]
    }
    state_rows = [
        row
        for row in _read_jsonl(inputs["coverage_state_rows"])
        if bool(row["articulation_relevant"]) or bool(row["low_degree_relevant"])
    ]
    metadata = {
        (str(row["task_id"]), int(row["solver_seed"])): row for row in state_rows
    }
    if set(metadata) != expected:
        raise ValueError("boundary TTF Quick coverage cohort differs")
    groups = Counter(str(row["layout_family"]) for row in metadata.values())
    if dict(groups) != dict(config["expected_state_count_by_group"]):
        raise ValueError("boundary TTF Quick coverage groups differ")
    dataset_tasks = {
        str(row["task_id"]) for row in _read_jsonl(inputs["dataset_manifest"])
    }
    if any(task_id not in dataset_tasks for task_id, _ in expected):
        raise ValueError("boundary TTF Quick task is absent from the dataset")
    qualification = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in _read_jsonl(inputs["qualification_manifest"])
    }
    for key, row in metadata.items():
        evidence = qualification.get(key)
        if (
            evidence is None
            or evidence.get("status") != "ok"
            or int(evidence.get("initial_conflicts", 0)) <= 0
            or str(evidence.get("state_fingerprint"))
            != str(row["state_fingerprint"])
        ):
            raise ValueError(f"boundary TTF Quick qualification differs: {key}")
    qualification_report = _read_json(inputs["qualification_report"])
    if (
        qualification_report.get("passed") is not True
        or int(qualification_report.get("incomplete_reset_count", -1)) != 0
        or int(qualification_report.get("inconsistent_initial_state_count", -1)) != 0
    ):
        raise ValueError("boundary TTF Quick source qualification has errors")
    shadow = _read_json(inputs["shadow_report"])
    if (
        shadow.get("passed_for_exploratory_quick") is not True
        or shadow.get("formal_confirmation_remains_failed") is not True
        or shadow.get("formal_speed_claim") is not False
    ):
        raise ValueError("boundary TTF Quick requires the passed non-promoting Shadow")
    shadow_keys = {
        (str(row["task_id"]), int(row["solver_seed"]))
        for row in _read_jsonl(inputs["shadow_selections"])
    }
    if shadow_keys != expected:
        raise ValueError("boundary TTF Quick Shadow cohort differs")
    runtime = _read_json(inputs["runtime_config"])
    if runtime.get("deterministic_pp_replay") is not True:
        raise ValueError("boundary TTF Quick requires paired PP replay")
    if "predecessor_report" in inputs:
        predecessor = _read_json(inputs["predecessor_report"])
        if (
            predecessor.get("warrants_larger_development_quick") is not False
            or predecessor.get("next_decision")
            != "retain_v2_and_stop_boundary_runtime_route"
            or predecessor.get("formal_speed_claim") is not False
        ):
            raise ValueError("boundary runtime optimization predecessor differs")
    return metadata


def _collection_kwargs(
    *,
    project_root: Path,
    config: dict[str, Any],
    controller: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "controller": "v2-full",
        "controller_bundle": str(
            (project_root / str(config["controller_bundle"])).resolve()
        ),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "audit",
        "stopping_rule": "wall-clock",
        "wall_time_budget_seconds": float(config["wall_time_budget_seconds"]),
        "episode_process_timeout_seconds": float(
            config["episode_process_timeout_seconds"]
        ),
        "environment_time_limit_seconds": float(
            config["environment_time_limit_seconds"]
        ),
    }
    if controller != "v2-full":
        kwargs["topology_boundary_augmentation"] = dict(
            config["topology_boundary_augmentation"]
        )
    return kwargs


def _extended_controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _controller_summary(rows)
    episodes = [
        dict(row["summary"])
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get("summary"), dict)
    ]
    totals = [dict(row.get("controller_totals") or {}) for row in episodes]
    selected_families = [
        dict(row.get("selected_family_counts") or {}) for row in episodes
    ]
    result.update(
        {
            "topology_boundary_generated_candidate_count": sum(
                int(row.get("topology_boundary_generated_count", 0)) for row in totals
            ),
            "topology_boundary_added_candidate_count": sum(
                int(row.get("topology_boundary_added_candidate_count", 0))
                for row in totals
            ),
            "topology_boundary_selected_repair_count": sum(
                int(count)
                for families in selected_families
                for family, count in families.items()
                if str(family).startswith("topology-boundary-")
            ),
            "mean_topology_boundary_analysis_seconds": _mean(
                [float(row.get("topology_boundary_analysis_seconds", 0.0)) for row in totals]
            ),
            "mean_topology_boundary_static_seconds": _mean(
                [float(row.get("topology_boundary_static_seconds", 0.0)) for row in totals]
            ),
            "mean_topology_boundary_dynamic_seconds": _mean(
                [float(row.get("topology_boundary_dynamic_seconds", 0.0)) for row in totals]
            ),
            "mean_topology_boundary_candidate_seconds": _mean(
                [float(row.get("topology_boundary_candidate_seconds", 0.0)) for row in totals]
            ),
            "mean_topology_boundary_gate_seconds": _mean(
                [float(row.get("topology_boundary_gate_seconds", 0.0)) for row in totals]
            ),
            "topology_boundary_gate_evaluated_count": sum(
                int(row.get("topology_boundary_gate_evaluated_count", 0))
                for row in totals
            ),
            "topology_boundary_gate_passed_count": sum(
                int(row.get("topology_boundary_gate_passed_count", 0)) for row in totals
            ),
            "topology_boundary_gate_reason_counts": dict(
                sorted(
                    Counter(
                        {
                            str(name).split("=", 1)[1]: sum(
                                int(row.get(name, 0)) for row in totals
                            )
                            for row in totals
                            for name in row
                            if str(name).startswith("topology_boundary_gate_reason=")
                        }
                    ).items()
                )
            ),
            "mean_initial_conflicts": _mean(
                [float(row.get("initial_conflicts", 0.0)) for row in episodes]
            ),
        }
    )
    return result


def _relative_improvement(baseline: float, challenger: float) -> float:
    return (baseline - challenger) / baseline if baseline else 0.0


def analyze_boundary_ttf_quick(
    config_path: str | Path, collection: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_boundary_ttf_quick_config(config)
    _, inputs = _registered_inputs(config_path, config)
    metadata = _cohort_metadata(config, inputs)
    expected = set(metadata)
    controllers = tuple(map(str, config["controllers"]))
    baseline_name, challenger_name = controllers
    collection = Path(collection).resolve()
    by_controller: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    errors = []
    for controller in controllers:
        path = collection / "controllers" / controller / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(path)
        indexed = {
            (str(row["task_id"]), int(row["solver_seed"])): row for row in rows
        }
        if set(indexed) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        episode_errors = sum(
            row.get("status") != "ok" or not isinstance(row.get("summary"), dict)
            for row in rows
        )
        if episode_errors:
            errors.append(f"{controller}: {episode_errors} episode execution error(s)")
        by_controller[controller] = indexed

    fingerprint_mismatches = 0
    conflict_mismatches = 0
    registered_state_mismatches = 0
    if not errors:
        for key in sorted(expected):
            summaries = [by_controller[name][key]["summary"] for name in controllers]
            fingerprint_mismatches += len(
                {str(summary["initial_fingerprint"]) for summary in summaries}
            ) != 1
            conflict_mismatches += len(
                {int(summary["initial_conflicts"]) for summary in summaries}
            ) != 1
            registered_state_mismatches += any(
                str(summary["initial_fingerprint"])
                != str(metadata[key]["state_fingerprint"])
                or int(summary["initial_conflicts"])
                != int(metadata[key]["initial_conflicts"])
                for summary in summaries
            )

    summaries = {
        controller: _extended_controller_summary(list(indexed.values()))
        for controller, indexed in by_controller.items()
    }
    subgroup_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for group in config["expected_state_count_by_group"]:
        keys = {key for key, row in metadata.items() if row["layout_family"] == group}
        subgroup_summaries[group] = {
            controller: _extended_controller_summary(
                [indexed[key] for key in sorted(keys) if key in indexed]
            )
            for controller, indexed in by_controller.items()
        }

    baseline = summaries[baseline_name]
    challenger = summaries[challenger_name]
    common = []
    if not errors:
        for key in sorted(expected):
            left = by_controller[baseline_name][key]["summary"]
            right = by_controller[challenger_name][key]["summary"]
            if bool(left["success"]) and bool(right["success"]):
                common.append(
                    (
                        float(left["wall_time_to_feasible"]),
                        float(right["wall_time_to_feasible"]),
                    )
                )
    base_capped = float(baseline["mean_capped_wall_time_to_feasible"])
    challenger_capped = float(challenger["mean_capped_wall_time_to_feasible"])
    capped_improvement = _relative_improvement(base_capped, challenger_capped)
    group_comparisons = {}
    for group, values in subgroup_summaries.items():
        base_value = float(values[baseline_name]["mean_capped_wall_time_to_feasible"])
        challenger_value = float(
            values[challenger_name]["mean_capped_wall_time_to_feasible"]
        )
        group_comparisons[group] = {
            "baseline_mean_capped_ttf": base_value,
            "challenger_mean_capped_ttf": challenger_value,
            "relative_improvement": _relative_improvement(base_value, challenger_value),
            "relative_regression": (challenger_value - base_value) / base_value
            if base_value
            else 0.0,
        }
    integrity_gates = {
        "complete_paired_coverage": not any("coverage" in error for error in errors),
        "zero_episode_errors": not any("execution error" in error for error in errors),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "registered_initial_states": registered_state_mismatches == 0,
        "zero_invalid_actions": all(
            int(summary.get("invalid_action_count", 0)) == 0
            for summary in summaries.values()
        ),
        "zero_semantic_mismatches": all(
            int(summary.get("fingerprint_mismatch_count", 0)) == 0
            for summary in summaries.values()
        ),
        "ttf_clock_registered": all(
            row.get("status") == "ok"
            and isinstance(row.get("summary"), dict)
            and row["summary"].get("ttf_clock_schema") == TTF_CLOCK_SCHEMA
            for indexed in by_controller.values()
            for row in indexed.values()
        ),
        "boundary_augmentation_activated": int(
            challenger["topology_boundary_added_candidate_count"]
        )
        > 0,
    }
    gates = dict(config["continuation_gates"])
    continuation_gates = {
        "success_noninferior": int(challenger["success_count"])
        >= int(baseline["success_count"]),
        "minimum_capped_ttf_relative_improvement": capped_improvement
        >= float(gates["minimum_capped_ttf_relative_improvement"]),
        "repair_iterations_noninferior": float(challenger["mean_repair_iterations"])
        <= float(baseline["mean_repair_iterations"]),
        "maximum_relevant_group_capped_ttf_regression": all(
            float(group_comparisons[group]["relative_regression"])
            <= float(gates["maximum_relevant_group_capped_ttf_regression"])
            for group in ("dao_articulated", "dao_ultra_bottleneck")
        ),
        "maximum_control_group_capped_ttf_regression": float(
            group_comparisons["dao_low_articulation_control"]["relative_regression"]
        )
        <= float(gates["maximum_control_group_capped_ttf_regression"]),
        "minimum_boundary_selected_repair_count": int(
            challenger["topology_boundary_selected_repair_count"]
        )
        >= int(gates["minimum_boundary_selected_repair_count"]),
    }
    passed = not errors and all(integrity_gates.values())
    warrants_larger = passed and all(continuation_gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": config["scientific_status"],
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "cohort_independent_of_shadow_outcomes": False,
        "primary_metric": config["primary_metric"],
        "success_constraint": config["success_constraint"],
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "subgroup_summaries": subgroup_summaries,
        "group_comparisons": group_comparisons,
        "comparison": {
            "success_noninferior": continuation_gates["success_noninferior"],
            "mean_capped_ttf_improvement_seconds": base_capped - challenger_capped,
            "mean_capped_ttf_relative_improvement": capped_improvement,
            "repair_iteration_relative_improvement": _relative_improvement(
                float(baseline["mean_repair_iterations"]),
                float(challenger["mean_repair_iterations"]),
            ),
            "common_success_count": len(common),
            "baseline_common_success_mean_ttf": _mean([left for left, _ in common]),
            "challenger_common_success_mean_ttf": _mean([right for _, right in common]),
        },
        "integrity_gates": integrity_gates,
        "continuation_gates": continuation_gates,
        "passed": passed,
        "warrants_larger_development_quick": warrants_larger,
        "next_decision": config[
            "next_decision_on_pass" if warrants_larger else "next_decision_on_failure"
        ],
        "fingerprint_mismatch_count": fingerprint_mismatches,
        "initial_conflict_mismatch_count": conflict_mismatches,
        "registered_initial_state_mismatch_count": registered_state_mismatches,
        "errors": errors,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "execution_schedule_sha256": sha256_file(
                collection / "execution_schedule.jsonl"
            ),
            "controller_manifests_sha256": {
                controller: sha256_file(
                    collection
                    / "controllers"
                    / controller
                    / "realized_dynamic_manifest.jsonl"
                )
                for controller in controllers
            },
        },
    }
    _write_json(collection / "boundary_ttf_quick_report.json", report)
    return report


def run_boundary_ttf_quick(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_boundary_ttf_quick_config(config)
    project_root, inputs = _registered_inputs(config_path, config)
    metadata = _cohort_metadata(config, inputs)
    cohort = set(metadata)
    controllers = tuple(map(str, config["controllers"]))
    schedule = boundary_ttf_quick_schedule(config)
    if len(schedule) != int(config["expected_schedule_entry_count"]):
        raise ValueError("boundary TTF Quick schedule size differs")
    output = Path(output).resolve()
    dataset = (project_root / str(config["dataset_root"])).resolve()
    runtime_config = inputs["runtime_config"]
    schedule_fingerprint = _fingerprint(schedule)
    registration = {
        "schema": STATUS_SCHEMA,
        "scientific_status": config["scientific_status"],
        "config_sha256": sha256_file(config_path),
        "schedule_fingerprint": schedule_fingerprint,
        "total_schedule_entries": len(schedule),
        "complete": False,
    }
    if dry_run:
        return {
            **registration,
            "dry_run": True,
            "controllers": {
                controller: run_closed_loop_collection(
                    dataset,
                    runtime_config,
                    output / "dry-run" / controller,
                    phase="realized_dynamic",
                    workers=1,
                    dry_run=True,
                    cohort_job_keys=cohort,
                    job_keys=cohort,
                    **_collection_kwargs(
                        project_root=project_root,
                        config=config,
                        controller=controller,
                    ),
                )
                for controller in controllers
            },
        }
    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "execution_schedule.jsonl"
    status_path = output / "quick_status.json"
    if schedule_path.is_file():
        if _read_jsonl(schedule_path) != schedule:
            raise ValueError("existing boundary TTF Quick schedule differs")
        if not resume:
            raise ValueError("boundary TTF Quick output exists; pass resume")
    else:
        _write_jsonl(schedule_path, schedule)
    _write_json(status_path, registration)

    qualification_root = output / "qualification"
    baseline_kwargs = _collection_kwargs(
        project_root=project_root, config=config, controller="v2-full"
    )
    run_closed_loop_collection(
        dataset,
        runtime_config,
        qualification_root,
        phase="qualify",
        workers=1,
        resume=qualification_root.joinpath("run_config.json").is_file(),
        cohort_job_keys=cohort,
        job_keys=cohort,
        **baseline_kwargs,
    )
    for controller in controllers:
        controller_root = output / "controllers" / controller
        run_closed_loop_collection(
            dataset,
            runtime_config,
            controller_root,
            phase="qualify",
            workers=1,
            resume=controller_root.joinpath("run_config.json").is_file(),
            cohort_job_keys=cohort,
            job_keys=cohort,
            qualification_source=qualification_root,
            **_collection_kwargs(
                project_root=project_root, config=config, controller=controller
            ),
        )

    completed = 0
    for item in schedule:
        controller = str(item["controller"])
        controller_root = output / "controllers" / controller
        manifest = controller_root / "realized_dynamic_manifest.jsonl"
        done = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in (_read_jsonl(manifest) if manifest.is_file() else [])
            if row.get("status") in {"ok", "error", "timeout"}
        }
        key = (str(item["task_id"]), int(item["solver_seed"]))
        if key not in done:
            run_closed_loop_collection(
                dataset,
                runtime_config,
                controller_root,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=cohort,
                job_keys={key},
                **_collection_kwargs(
                    project_root=project_root,
                    config=config,
                    controller=controller,
                ),
            )
        completed += 1
        _write_json(
            status_path,
            {
                **registration,
                "completed_schedule_entries": completed,
                "current": item,
            },
        )
    report = analyze_boundary_ttf_quick(config_path, output)
    _write_json(
        status_path,
        {
            **registration,
            "completed_schedule_entries": len(schedule),
            "complete": True,
            "report_sha256": sha256_file(output / "boundary_ttf_quick_report.json"),
        },
    )
    return report


__all__ = [
    "CONTROLLERS",
    "analyze_boundary_ttf_quick",
    "boundary_ttf_quick_schedule",
    "run_boundary_ttf_quick",
    "validate_boundary_ttf_quick_config",
]
