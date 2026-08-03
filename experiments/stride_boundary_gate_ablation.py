from __future__ import annotations

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
from experiments.stride_boundary_ttf_quick import (
    TTF_CLOCK_SCHEMA,
    _cohort_metadata as _boundary_cohort_metadata,
    _extended_controller_summary,
    _registered_path,
    _relative_improvement,
)


CONFIG_SCHEMA = "lns2.stride.boundary_gate_ablation_config.v1"
REPORT_SCHEMA = "lns2.stride.boundary_gate_ablation_report.v1"
STATUS_SCHEMA = "lns2.stride.boundary_gate_ablation_status.v1"
CONTROLLERS = (
    "v2-full",
    "stride-boundary-map-gate-v1",
    "stride-boundary-stall-guard-v1",
)
MAP_GATE = "stride-boundary-map-gate-v1"
STALL_GUARD = "stride-boundary-stall-guard-v1"


def _expected_augmentations() -> dict[str, dict[str, Any]]:
    base = {
        "enabled": True,
        "generator_id": "stride-topoboundary-v1",
        "neighborhood_size": 16,
        "core_budget": 4,
        "maximum_added_candidates": 2,
        "static_grid_cache": True,
        "activation_gate": {
            "gate_id": "stride-boundary-map-topology-v1",
            "minimum_low_degree_cell_ratio": 0.06,
        },
    }
    return {
        MAP_GATE: {**base, "runtime_id": MAP_GATE},
        STALL_GUARD: {
            **base,
            "runtime_id": STALL_GUARD,
            "phase_guard": {
                "gate_id": STALL_GUARD,
                "maximum_no_progress_streak": 5,
            },
        },
    }


def validate_boundary_gate_ablation_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected boundary gate ablation config")
    if (
        config.get("scientific_status")
        != "outcome_informed_same_cohort_run_to_completion_gate_ablation"
        or config.get("experiment_id") != "stride-boundary-gate-ablation-rtc-v1"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("training_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("cohort_independent_of_prior_outcomes"))
    ):
        raise ValueError("boundary gate ablation must remain non-promoting")
    if (
        config.get("primary_metric") != "mean_raw_wall_time_to_feasible"
        or config.get("completion_constraint")
        != "all_controllers_complete_all_registered_states"
        or config.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA
        or config.get("stopping_rule") != "run-to-completion"
        or config.get("scientific_time_limit_seconds") is not None
        or config.get("environment_time_limit_seconds") is not None
        or config.get("episode_process_timeout_seconds") is not None
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
        or config.get("executed_controller") != "v2-full"
        or config.get("controller_bundle")
        != "artifacts/initlns-closed-loop-controller-v2"
        or config.get("verification_profile") != "deployment"
        or int(config.get("expected_state_count", -1)) != 18
        or int(config.get("expected_schedule_entry_count", -1)) != 54
        or int(config.get("workers", 0)) != 1
        or config.get("execution_order") != "strict_rotating_triplet_order"
        or not bool(config.get("deterministic_pp_replay_required"))
    ):
        raise ValueError("boundary gate ablation runtime contract changed")
    jobs = [
        (str(value[0]), int(value[1]))
        for value in config.get("registered_job_keys") or []
    ]
    if len(jobs) != 18 or len(set(jobs)) != 18:
        raise ValueError("boundary gate ablation cohort changed")
    if dict(config.get("expected_state_count_by_group") or {}) != {
        "dao_low_articulation_control": 4,
        "dao_articulated": 8,
        "dao_ultra_bottleneck": 6,
    }:
        raise ValueError("boundary gate ablation groups changed")
    if dict(config.get("topology_boundary_augmentations") or {}) != (
        _expected_augmentations()
    ):
        raise ValueError("boundary gate ablation augmentations changed")
    if dict(config.get("selection_gates") or {}) != {
        "minimum_stall_vs_map_raw_ttf_improvement": 0.02,
        "stall_repair_iterations_noninferior": True,
        "maximum_group_raw_ttf_regression": 0.10,
        "minimum_boundary_selected_repair_count": 1,
    }:
        raise ValueError("boundary gate ablation selection gates changed")
    if set(config.get("inputs") or {}) != {
        "runtime_config",
        "coverage_state_rows",
        "dataset_manifest",
        "qualification_manifest",
        "qualification_report",
        "shadow_report",
        "shadow_selections",
        "controller_manifest",
        "predecessor_report",
    }:
        raise ValueError("boundary gate ablation inputs changed")


def boundary_gate_ablation_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    validate_boundary_gate_ablation_config(config)
    jobs = sorted(
        (str(value[0]), int(value[1])) for value in config["registered_job_keys"]
    )
    schedule: list[dict[str, Any]] = []
    for key_index, (task_id, solver_seed) in enumerate(jobs):
        offset = key_index % len(CONTROLLERS)
        order = CONTROLLERS[offset:] + CONTROLLERS[:offset]
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
    inputs = {
        name: _registered_path(project_root, dict(specification))
        for name, specification in config["inputs"].items()
    }
    predecessor = _read_json(inputs["predecessor_report"])
    if (
        predecessor.get("warrants_larger_development_quick") is not False
        or predecessor.get("next_decision")
        != "retain_v2_and_stop_boundary_runtime_optimization"
        or predecessor.get("formal_speed_claim") is not False
    ):
        raise ValueError("boundary gate ablation predecessor differs")
    return project_root, inputs


def _cohort_metadata(
    config: dict[str, Any], inputs: dict[str, Path]
) -> dict[tuple[str, int], dict[str, Any]]:
    return _boundary_cohort_metadata(
        config,
        {name: path for name, path in inputs.items() if name != "predecessor_report"},
    )


def _collection_kwargs(
    *, project_root: Path, config: dict[str, Any], controller: str
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "controller": "v2-full",
        "controller_bundle": str(
            (project_root / str(config["controller_bundle"])).resolve()
        ),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": str(config["verification_profile"]),
        "stopping_rule": "run-to-completion",
    }
    if controller != "v2-full":
        kwargs["topology_boundary_augmentation"] = dict(
            config["topology_boundary_augmentations"][controller]
        )
    return kwargs


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _raw_ttf(rows: list[dict[str, Any]]) -> float | None:
    values = [
        float(row["summary"]["wall_time_to_feasible"])
        for row in rows
        if row.get("status") == "ok"
        and isinstance(row.get("summary"), dict)
        and bool(row["summary"].get("success"))
    ]
    return _mean(values) if len(values) == len(rows) and values else None


def analyze_boundary_gate_ablation(
    config_path: str | Path, collection: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_boundary_gate_ablation_config(config)
    _, inputs = _registered_inputs(config_path, config)
    metadata = _cohort_metadata(config, inputs)
    expected = set(metadata)
    collection = Path(collection).resolve()
    by_controller: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    errors: list[str] = []
    for controller in CONTROLLERS:
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
            errors.append(f"{controller}: {episode_errors} episode error(s)")
        by_controller[controller] = indexed

    fingerprint_mismatches = 0
    conflict_mismatches = 0
    registered_state_mismatches = 0
    if not errors:
        for key in sorted(expected):
            rows = [by_controller[name][key]["summary"] for name in CONTROLLERS]
            fingerprint_mismatches += len(
                {str(row["initial_fingerprint"]) for row in rows}
            ) != 1
            conflict_mismatches += len(
                {int(row["initial_conflicts"]) for row in rows}
            ) != 1
            registered_state_mismatches += any(
                str(row["initial_fingerprint"])
                != str(metadata[key]["state_fingerprint"])
                or int(row["initial_conflicts"])
                != int(metadata[key]["initial_conflicts"])
                for row in rows
            )

    controller_rows = {
        name: list(indexed.values()) for name, indexed in by_controller.items()
    }
    summaries = {
        name: {
            **_extended_controller_summary(rows),
            "mean_raw_wall_time_to_feasible": _raw_ttf(rows),
        }
        for name, rows in controller_rows.items()
    }
    subgroup_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for group in config["expected_state_count_by_group"]:
        keys = {key for key, row in metadata.items() if row["layout_family"] == group}
        subgroup_summaries[group] = {}
        for controller in CONTROLLERS:
            rows = [by_controller[controller][key] for key in sorted(keys) if key in by_controller[controller]]
            subgroup_summaries[group][controller] = {
                **_extended_controller_summary(rows),
                "mean_raw_wall_time_to_feasible": _raw_ttf(rows),
            }

    all_completed = all(
        int(summary["success_count"]) == len(expected)
        and int(summary["execution_error_count"]) == 0
        and summary["mean_raw_wall_time_to_feasible"] is not None
        for summary in summaries.values()
    )
    comparisons: dict[str, dict[str, Any]] = {}
    if all_completed:
        baseline_ttf = float(summaries["v2-full"]["mean_raw_wall_time_to_feasible"])
        for controller in (MAP_GATE, STALL_GUARD):
            value = float(summaries[controller]["mean_raw_wall_time_to_feasible"])
            comparisons[f"{controller}_vs_v2_full"] = {
                "baseline_mean_raw_ttf": baseline_ttf,
                "challenger_mean_raw_ttf": value,
                "relative_improvement": _relative_improvement(baseline_ttf, value),
            }
        map_ttf = float(summaries[MAP_GATE]["mean_raw_wall_time_to_feasible"])
        stall_ttf = float(summaries[STALL_GUARD]["mean_raw_wall_time_to_feasible"])
        comparisons["stall_guard_vs_map_gate"] = {
            "map_gate_mean_raw_ttf": map_ttf,
            "stall_guard_mean_raw_ttf": stall_ttf,
            "relative_improvement": _relative_improvement(map_ttf, stall_ttf),
            "repair_iteration_relative_improvement": _relative_improvement(
                float(summaries[MAP_GATE]["mean_repair_iterations"]),
                float(summaries[STALL_GUARD]["mean_repair_iterations"]),
            ),
        }

    group_comparisons: dict[str, dict[str, Any]] = {}
    if all_completed:
        for group, values in subgroup_summaries.items():
            map_value = float(values[MAP_GATE]["mean_raw_wall_time_to_feasible"])
            stall_value = float(values[STALL_GUARD]["mean_raw_wall_time_to_feasible"])
            group_comparisons[group] = {
                "map_gate_mean_raw_ttf": map_value,
                "stall_guard_mean_raw_ttf": stall_value,
                "stall_relative_improvement": _relative_improvement(
                    map_value, stall_value
                ),
                "stall_relative_regression": (
                    (stall_value - map_value) / map_value if map_value else 0.0
                ),
            }

    integrity_gates = {
        "complete_paired_coverage": not any("coverage" in value for value in errors),
        "zero_episode_errors": not any("episode error" in value for value in errors),
        "all_controllers_completed_all_states": all_completed,
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
        "raw_ttf_clock_registered": all(
            row.get("status") == "ok"
            and isinstance(row.get("summary"), dict)
            and row["summary"].get("ttf_clock_schema") == TTF_CLOCK_SCHEMA
            and row["summary"].get("wall_time_budget_seconds") is None
            and row["summary"].get("capped_wall_time_to_feasible") is None
            for indexed in by_controller.values()
            for row in indexed.values()
        ),
    }
    gates = dict(config["selection_gates"])
    stall_comparison = comparisons.get("stall_guard_vs_map_gate", {})
    selection_gates = {
        "minimum_stall_vs_map_raw_ttf_improvement": all_completed
        and float(stall_comparison.get("relative_improvement", 0.0))
        >= float(gates["minimum_stall_vs_map_raw_ttf_improvement"]),
        "stall_repair_iterations_noninferior": all_completed
        and float(summaries[STALL_GUARD]["mean_repair_iterations"])
        <= float(summaries[MAP_GATE]["mean_repair_iterations"]),
        "maximum_group_raw_ttf_regression": all_completed
        and all(
            float(value["stall_relative_regression"])
            <= float(gates["maximum_group_raw_ttf_regression"])
            for value in group_comparisons.values()
        ),
        "minimum_boundary_selected_repair_count": all_completed
        and int(summaries[STALL_GUARD]["topology_boundary_selected_repair_count"])
        >= int(gates["minimum_boundary_selected_repair_count"]),
    }
    passed = not errors and all(integrity_gates.values())
    retain_stall = passed and all(selection_gates.values())
    selected_runtime = STALL_GUARD if retain_stall else MAP_GATE if passed else None
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": config["scientific_status"],
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "training_allowed": False,
        "cohort_independent_of_prior_outcomes": False,
        "primary_metric": config["primary_metric"],
        "completion_constraint": config["completion_constraint"],
        "stopping_rule": config["stopping_rule"],
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "subgroup_summaries": subgroup_summaries,
        "comparisons": comparisons,
        "group_comparisons": group_comparisons,
        "integrity_gates": integrity_gates,
        "selection_gates": selection_gates,
        "passed": passed,
        "retain_stall_guard": retain_stall,
        "selected_runtime": selected_runtime,
        "next_decision": config[
            "next_decision_on_stall_pass"
            if retain_stall
            else "next_decision_on_stall_failure"
        ]
        if passed
        else "repair_run_to_completion_integrity_before_selection",
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
                for controller in CONTROLLERS
            },
        },
    }
    _write_json(collection / "boundary_gate_ablation_report.json", report)
    return report


def run_boundary_gate_ablation(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_boundary_gate_ablation_config(config)
    project_root, inputs = _registered_inputs(config_path, config)
    metadata = _cohort_metadata(config, inputs)
    cohort = set(metadata)
    schedule = boundary_gate_ablation_schedule(config)
    output = Path(output).resolve()
    dataset = (project_root / str(config["dataset_root"])).resolve()
    runtime_config = inputs["runtime_config"]
    registration = {
        "schema": STATUS_SCHEMA,
        "scientific_status": config["scientific_status"],
        "config_sha256": sha256_file(config_path),
        "schedule_fingerprint": _fingerprint(schedule),
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
                for controller in CONTROLLERS
            },
        }

    output.mkdir(parents=True, exist_ok=True)
    schedule_path = output / "execution_schedule.jsonl"
    status_path = output / "ablation_status.json"
    if schedule_path.is_file():
        if _read_jsonl(schedule_path) != schedule:
            raise ValueError("existing boundary gate ablation schedule differs")
        if not resume:
            raise ValueError("boundary gate ablation output exists; pass resume")
    else:
        _write_jsonl(schedule_path, schedule)
    _write_json(status_path, registration)

    qualification_root = output / "qualification"
    run_closed_loop_collection(
        dataset,
        runtime_config,
        qualification_root,
        phase="qualify",
        workers=1,
        resume=qualification_root.joinpath("run_config.json").is_file(),
        cohort_job_keys=cohort,
        job_keys=cohort,
        **_collection_kwargs(
            project_root=project_root, config=config, controller="v2-full"
        ),
    )
    for controller in CONTROLLERS:
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
                    project_root=project_root, config=config, controller=controller
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
    report = analyze_boundary_gate_ablation(config_path, output)
    _write_json(
        status_path,
        {
            **registration,
            "completed_schedule_entries": len(schedule),
            "complete": True,
            "report_sha256": sha256_file(
                output / "boundary_gate_ablation_report.json"
            ),
        },
    )
    return report


__all__ = [
    "CONTROLLERS",
    "analyze_boundary_gate_ablation",
    "boundary_gate_ablation_schedule",
    "run_boundary_gate_ablation",
    "validate_boundary_gate_ablation_config",
]
