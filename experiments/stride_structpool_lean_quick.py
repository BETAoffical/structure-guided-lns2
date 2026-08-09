from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import (
    closed_loop_producer_identity,
    registered_input,
    sha256_file,
)
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.stride_maprank_raw_ttf import _paired_comparison
from experiments.stride_structpool_lns2_quick import (
    load_structpool_lns2_quick_config,
)
from experiments.stride_structpool_ttf_quick import (
    TTF_CLOCK_SCHEMA,
    _controller_kwargs as _v2_controller_kwargs,
    _quick_controller_summary,
    _structpool_trace_counts,
)
from lns2_selector.runtime.online_selection import validate_structpool_augmentation


CONFIG_SCHEMA = "lns2.stride.structpool_lean_quick_config.v1"
STATUS_SCHEMA = "lns2.stride.structpool_lean_quick_status.v1"
REPORT_SCHEMA = "lns2.stride.structpool_lean_quick_report.v1"
CONTROLLERS = (
    "official_adaptive",
    "v2-full",
    "v2-plus-structpool",
    "v2-plus-structpool-lean",
)
PHASES = {
    "official_adaptive": "official_adaptive",
    "v2-full": "realized_dynamic",
    "v2-plus-structpool": "realized_dynamic",
    "v2-plus-structpool-lean": "realized_dynamic",
}


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    return registered_input(root, specification, label="StructPool Lean Quick")


def load_structpool_lean_quick_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_development_four_controller_quick_before_timing"
        or config.get("experiment_id") != "stride-structpool-lean-quick-v1"
        or config.get("pre_registration_parent_commit")
        != "3b97422359e27ecd2c53cc9be76e9b4689596774"
        or tuple(config.get("controllers") or ()) != CONTROLLERS
    ):
        raise ValueError("StructPool Lean Quick identity changed")
    if dict(config.get("comparison") or {}) != {
        "lns2_baseline": "native_official_adaptive_neighborhood_generation",
        "v2_baseline": "frozen_v2_ranking_over_exact_base_pool",
        "full_structpool": "same_frozen_v2_over_base_plus_full_structpool",
        "lean_structpool": "same_frozen_v2_after_pure_bottleneck_filter",
        "execution_order": "strict_four_controller_rotation",
        "paired_solver_seed_required": True,
        "workers": 1,
    }:
        raise ValueError("StructPool Lean Quick comparison changed")
    if dict(config.get("runtime") or {}) != {
        "config": "configs/stride_stage4r_high_load_runtime.json",
        "stopping_rule": "run-to-completion",
        "scientific_time_limit_seconds": None,
        "environment_time_limit_seconds": None,
        "episode_process_timeout_seconds": None,
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "deterministic_pp_replay": True,
    }:
        raise ValueError("StructPool Lean Quick runtime changed")
    full = validate_structpool_augmentation(
        dict(config["structpool_augmentation"])
    )
    lean = validate_structpool_augmentation(
        dict(config["lean_structpool_augmentation"])
    )
    if full is None or "lean_filter" in full or lean is None or "lean_filter" not in lean:
        raise ValueError("StructPool Lean Quick treatment identity changed")

    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role") != "reused_high_load_development_quick_not_formal_ood"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2)
        or int(cohort.get("episode_count_per_controller", -1)) != 8
        or [str(row.get("id")) for row in groups] != ["maze300", "room500"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
    ):
        raise ValueError("StructPool Lean Quick cohort changed")
    if dict(config.get("performance_gates") or {}) != {
        "minimum_mean_raw_ttf_improvement_vs_full_structpool": 0.0,
        "minimum_paired_faster_fraction_vs_full_structpool": 0.5,
        "maximum_group_raw_ttf_regression_vs_full_structpool": 0.1,
        "maximum_mean_repair_iterations_delta_vs_full_structpool": 0.0,
        "minimum_success_count_delta_vs_full_structpool": 0,
        "minimum_selection_seconds_improvement_vs_full_structpool": 0.0,
    }:
        raise ValueError("StructPool Lean Quick performance gates changed")
    if dict(config.get("claim_boundary") or {}) != {
        "development_diagnostic_only": True,
        "formal_speed_claim": False,
        "fresh_map_claim": False,
        "default_replacement_allowed": False,
        "new_ranker_trained": False,
        "candidate_generator_changed": False,
        "result_must_report_all_six_pairwise_comparisons": True,
    }:
        raise ValueError("StructPool Lean Quick claim boundary changed")

    expected_inputs = {
        "source_quick_config",
        "speed2_report",
        "lean_audit_report",
        "runtime_config",
        "controller_manifest",
        "maze_dataset_summary",
        "room_dataset_summary",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("StructPool Lean Quick input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    _source_path, _source_root, source = load_structpool_lns2_quick_config(
        inputs["source_quick_config"]
    )
    if dict(source["structpool_augmentation"]) != dict(
        config["structpool_augmentation"]
    ):
        raise ValueError("StructPool Lean Quick changed full-pool semantics")
    if str(source["controller_bundle"]) != str(config["controller_bundle"]):
        raise ValueError("StructPool Lean Quick changed the frozen V2 bundle")
    source_groups = {
        str(row["id"]): (str(row["dataset"]), str(row["split"]), list(row["tasks"]))
        for row in source["cohort"]["groups"]
    }
    current_groups = {
        str(row["id"]): (str(row["dataset"]), str(row["split"]), list(row["tasks"]))
        for row in groups
    }
    if current_groups != source_groups:
        raise ValueError("StructPool Lean Quick changed the reused cohort")
    speed2 = _read_json(inputs["speed2_report"])
    lean_audit = _read_json(inputs["lean_audit_report"])
    if speed2.get("integrity_passed") is not True:
        raise ValueError("StructPool Lean Quick Speed2 source lacks integrity")
    if (
        lean_audit.get("integrity_passed") is not True
        or lean_audit.get("passed_for_runtime_quick") is not True
        or lean_audit.get("runtime_read") is not False
        or lean_audit.get("ttf_read") is not False
    ):
        raise ValueError("StructPool Lean Quick offline gate did not authorize timing")
    return path, root, config


def structpool_lean_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    ]
    schedule: list[dict[str, Any]] = []
    for index, (group_id, task_id, solver_seed) in enumerate(keys):
        offset = index % len(CONTROLLERS)
        order = CONTROLLERS[offset:] + CONTROLLERS[:offset]
        schedule.extend(
            {
                "group_id": group_id,
                "task_id": task_id,
                "solver_seed": solver_seed,
                "controller": controller,
                "within_key_position": position,
            }
            for position, controller in enumerate(order)
        )
    return schedule


def _controller_kwargs(
    root: Path, config: dict[str, Any], controller: str
) -> dict[str, Any]:
    if controller == "official_adaptive":
        result = _v2_controller_kwargs(root, config, "v2-full")
        result["controller"] = "official_adaptive"
        return result
    if controller == "v2-plus-structpool-lean":
        result = _v2_controller_kwargs(root, config, "v2-full")
        result["structpool_augmentation"] = dict(
            config["lean_structpool_augmentation"]
        )
        return result
    return _v2_controller_kwargs(root, config, controller)


def _runtime_counts(
    collection: Path, row: dict[str, Any]
) -> dict[str, int]:
    counts = _structpool_trace_counts(collection, row)
    totals = dict(dict(row.get("summary") or {}).get("controller_totals") or {})
    counts.update(
        {
            "generated_candidates": int(totals.get("structpool_generated_count", 0)),
            "retained_candidates": int(
                totals.get("structpool_retained_candidate_count", 0)
            ),
            "filtered_candidates": int(
                totals.get("structpool_filtered_candidate_count", 0)
            ),
            "lean_filter_enabled_decisions": int(
                totals.get("structpool_lean_filter_enabled_count", 0)
            ),
        }
    )
    return counts


def _group_dataset(root: Path, config: dict[str, Any], group: dict[str, Any]) -> Path:
    dataset = group.get("dataset", dict(config["cohort"]).get("dataset"))
    if dataset is None:
        raise ValueError(f"StructPool Lean group {group.get('id')} lacks a dataset")
    return (root / str(dataset)).resolve()


def run_structpool_lean_multigroup(
    path: Path,
    root: Path,
    config: dict[str, Any],
    output: str | Path,
    *,
    status_schema: str,
    status_filename: str,
    report_schema: str,
    report_filename: str,
    report_scientific_status: str,
    next_step_on_pass: str,
    next_step_on_failure: str,
    producer_source_files: tuple[str, ...] = (),
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    output = Path(output).resolve()
    schedule = structpool_lean_schedule(config)
    if dry_run:
        return {
            "schema": status_schema,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    status_path = output / status_filename
    prepared = prepare_resumable_output(
        output,
        status_filename=status_filename,
        status_schema=status_schema,
        config_path=path,
        schedule=schedule,
        producer=closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_structpool_lean_quick.py",
                "experiments/stride_structpool_lns2_quick.py",
                "experiments/stride_structpool_ttf_quick.py",
                "experiments/stride_maprank_raw_ttf.py",
                *producer_source_files,
            ),
        ),
        resume=resume,
        report_filename=report_filename,
        report_schema=report_schema,
        label="StructPool Lean",
    )
    status_base = prepared.base_status
    if prepared.completed_report is not None:
        return prepared.completed_report
    runtime = (root / str(config["runtime"]["config"])).resolve()
    registered_qualification = config.get("qualification_source")
    qualification_source = (
        (root / str(registered_qualification)).resolve()
        if registered_qualification is not None
        and bool(config.get("reuse_qualification_source", True))
        else None
    )
    groups = {str(row["id"]): dict(row) for row in config["cohort"]["groups"]}
    seeds = tuple(map(int, config["cohort"]["solver_seeds"]))
    all_keys = {
        (str(task), seed)
        for group in groups.values()
        for task in group["tasks"]
        for seed in seeds
    }
    shared_dataset = dict(config["cohort"]).get("dataset")
    if shared_dataset is not None:
        dataset = (root / str(shared_dataset)).resolve()
        qualification = output / "qualification"
        run_closed_loop_collection(
            dataset,
            runtime,
            qualification,
            phase="qualify",
            workers=1,
            resume=(
                prepared.resumed
                and qualification.joinpath("run_config.json").is_file()
            ),
            cohort_job_keys=all_keys,
            job_keys=all_keys,
            qualification_source=qualification_source,
            **_controller_kwargs(root, config, "v2-full"),
        )
        qualification_plans = [
            (group, controller, dataset, all_keys, qualification)
            for group in groups.values()
            for controller in CONTROLLERS
        ]
    else:
        qualification_plans = []
        for group in groups.values():
            dataset = _group_dataset(root, config, group)
            keys = {(str(task), seed) for task in group["tasks"] for seed in seeds}
            qualification = output / "groups" / str(group["id"]) / "qualification"
            run_closed_loop_collection(
                dataset,
                runtime,
                qualification,
                phase="qualify",
                workers=1,
                resume=(
                    prepared.resumed
                    and qualification.joinpath("run_config.json").is_file()
                ),
                cohort_job_keys=keys,
                job_keys=keys,
                qualification_source=qualification_source,
                **_controller_kwargs(root, config, "v2-full"),
            )
            qualification_plans.extend(
                (group, controller, dataset, keys, qualification)
                for controller in CONTROLLERS
            )
    for group, controller, dataset, keys, qualification in qualification_plans:
        collection = (
            output / "groups" / str(group["id"]) / "controllers" / controller
        )
        run_closed_loop_collection(
            dataset,
            runtime,
            collection,
            phase="qualify",
            workers=1,
            resume=(
                prepared.resumed
                and collection.joinpath("run_config.json").is_file()
            ),
            cohort_job_keys=keys,
            job_keys=keys,
            qualification_source=qualification,
            **_controller_kwargs(root, config, controller),
        )
    completed = 0
    for item in schedule:
        group = groups[str(item["group_id"])]
        dataset = _group_dataset(root, config, group)
        keys = (
            all_keys
            if shared_dataset is not None
            else {(str(task), seed) for task in group["tasks"] for seed in seeds}
        )
        controller = str(item["controller"])
        collection = (
            output / "groups" / str(item["group_id"]) / "controllers" / controller
        )
        manifest = collection / f"{PHASES[controller]}_manifest.jsonl"
        done = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in (_read_jsonl(manifest) if manifest.is_file() else [])
            if row.get("status") in {"ok", "error"}
        }
        key = (str(item["task_id"]), int(item["solver_seed"]))
        if key not in done:
            run_closed_loop_collection(
                dataset,
                runtime,
                collection,
                phase=PHASES[controller],
                workers=1,
                resume=True,
                cohort_job_keys=keys,
                job_keys={key},
                **_controller_kwargs(root, config, controller),
            )
        completed += 1
        _write_json(
            status_path,
            {
                **status_base,
                "completed_schedule_entries": completed,
                "current": item,
                "complete": False,
            },
        )
    report = analyze_structpool_lean_multigroup(
        path,
        config,
        output,
        status_filename=status_filename,
        report_schema=report_schema,
        report_filename=report_filename,
        report_scientific_status=report_scientific_status,
        next_step_on_pass=next_step_on_pass,
        next_step_on_failure=next_step_on_failure,
        producer=status_base["producer_identity"],
        producer_source_files=producer_source_files,
    )
    _write_json(
        status_path,
        {
            **status_base,
            "completed_schedule_entries": completed,
            "complete": True,
            "report_sha256": sha256_file(output / report_filename),
        },
    )
    return report


def run_structpool_lean_quick(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_structpool_lean_quick_config(config_path)
    return run_structpool_lean_multigroup(
        path,
        root,
        config,
        output,
        status_schema=STATUS_SCHEMA,
        status_filename="quick_status.json",
        report_schema=REPORT_SCHEMA,
        report_filename="structpool_lean_quick_report.json",
        report_scientific_status="development_four_controller_lean_quick",
        next_step_on_pass="retain_lean_runtime_and_preregister_fresh_map_confirmation",
        next_step_on_failure="retain_speed2_full_structpool_and_reject_lean_as_default",
        producer_source_files=("experiments/stride_structpool_lean_quick.py",),
        resume=resume,
        dry_run=dry_run,
    )


def analyze_structpool_lean_multigroup(
    path: Path,
    config: dict[str, Any],
    output: str | Path,
    *,
    status_filename: str,
    report_schema: str,
    report_filename: str,
    report_scientific_status: str,
    next_step_on_pass: str,
    next_step_on_failure: str,
    producer: dict[str, Any] | None = None,
    producer_source_files: tuple[str, ...] = (),
) -> dict[str, Any]:
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename=status_filename,
        report_filename=report_filename,
        report_schema=report_schema,
        config_path=path,
    )
    if completed is not None:
        return completed
    if producer is None:
        producer = closed_loop_producer_identity(
            project_root=path.parent.parent,
            source_files=(
                "experiments/stride_structpool_lean_quick.py",
                "experiments/stride_structpool_lns2_quick.py",
                "experiments/stride_structpool_ttf_quick.py",
                "experiments/stride_maprank_raw_ttf.py",
                *producer_source_files,
            ),
            native_required=False,
        )
    groups = [dict(row) for row in config["cohort"]["groups"]]
    seeds = tuple(map(int, config["cohort"]["solver_seeds"]))
    expected = {
        (str(group["id"]), str(task), seed)
        for group in groups
        for task in group["tasks"]
        for seed in seeds
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    hashes: dict[str, dict[str, str]] = defaultdict(dict)
    errors: list[str] = []
    pool_counts = {
        controller: {
            "selected": 0,
            "gate_passed": 0,
            "added_candidates": 0,
            "generated_candidates": 0,
            "retained_candidates": 0,
            "filtered_candidates": 0,
            "lean_filter_enabled_decisions": 0,
        }
        for controller in ("v2-plus-structpool", "v2-plus-structpool-lean")
    }
    for controller in CONTROLLERS:
        by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for group in groups:
            group_id = str(group["id"])
            collection = output / "groups" / group_id / "controllers" / controller
            manifest = collection / f"{PHASES[controller]}_manifest.jsonl"
            rows = _read_jsonl(manifest)
            hashes[controller][group_id] = sha256_file(manifest)
            for row in rows:
                key = (group_id, str(row["task_id"]), int(row["solver_seed"]))
                if key in by_key:
                    errors.append(f"{controller}: duplicate episode {key}")
                by_key[key] = row
                if controller in pool_counts and row.get("status") == "ok":
                    observed = _runtime_counts(collection, row)
                    for name, value in observed.items():
                        pool_counts[controller][name] += value
        if set(by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = by_key

    fingerprint_mismatches = conflict_mismatches = bad_clock = capped_values = 0
    for key in sorted(expected):
        rows = [indexed[controller].get(key) for controller in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries_for_key = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += (
            len({str(row.get("initial_fingerprint")) for row in summaries_for_key})
            != 1
        )
        conflict_mismatches += (
            len({int(row.get("initial_conflicts", -1)) for row in summaries_for_key})
            != 1
        )
        bad_clock += sum(
            row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA
            for row in summaries_for_key
        )
        capped_values += sum(
            row.get("capped_wall_time_to_feasible") is not None
            for row in summaries_for_key
        )
    summaries = {
        controller: _quick_controller_summary(list(indexed[controller].values()))
        for controller in CONTROLLERS
    }
    keys = sorted(expected)

    def comparisons(selected_keys: list[tuple[str, str, int]]) -> dict[str, Any]:
        official = indexed["official_adaptive"]
        v2 = indexed["v2-full"]
        full = indexed["v2-plus-structpool"]
        lean = indexed["v2-plus-structpool-lean"]
        return {
            "v2_vs_lns2": _paired_comparison(official, v2, selected_keys),
            "full_structpool_vs_lns2": _paired_comparison(
                official, full, selected_keys
            ),
            "lean_structpool_vs_lns2": _paired_comparison(
                official, lean, selected_keys
            ),
            "full_structpool_vs_v2": _paired_comparison(v2, full, selected_keys),
            "lean_structpool_vs_v2": _paired_comparison(v2, lean, selected_keys),
            "lean_vs_full_structpool": _paired_comparison(full, lean, selected_keys),
        }

    overall = comparisons(keys)
    per_group = {
        str(group["id"]): comparisons(
            [key for key in keys if key[0] == str(group["id"])]
        )
        for group in groups
    }
    integrity = {
        "complete_four_controller_coverage": not any(
            "coverage" in error for error in errors
        ),
        "zero_execution_errors": all(
            row.get("status") == "ok"
            for controller_rows in indexed.values()
            for row in controller_rows.values()
        ),
        "all_controllers_succeeded": all(
            summary["success_count"] == len(expected)
            for summary in summaries.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "no_capped_ttf_values": capped_values == 0,
        "zero_invalid_actions": all(
            summary["invalid_action_count"] == 0 for summary in summaries.values()
        ),
        "zero_semantic_mismatches": all(
            summary["fingerprint_mismatch_count"] == 0
            for summary in summaries.values()
        ),
        "full_structpool_activated_and_selected": (
            pool_counts["v2-plus-structpool"]["gate_passed"] > 0
            and pool_counts["v2-plus-structpool"]["selected"] > 0
        ),
        "lean_structpool_activated_and_selected": (
            pool_counts["v2-plus-structpool-lean"]["gate_passed"] > 0
            and pool_counts["v2-plus-structpool-lean"]["selected"] > 0
        ),
        "lean_filter_exercised": (
            pool_counts["v2-plus-structpool-lean"]["lean_filter_enabled_decisions"]
            > 0
            and pool_counts["v2-plus-structpool-lean"]["filtered_candidates"] > 0
        ),
        "full_pool_did_not_filter": (
            pool_counts["v2-plus-structpool"]["filtered_candidates"] == 0
            and pool_counts["v2-plus-structpool"][
                "lean_filter_enabled_decisions"
            ]
            == 0
        ),
    }
    comparison = overall["lean_vs_full_structpool"]
    gates = dict(config["performance_gates"])
    group_regression = max(
        (
            -float(row["lean_vs_full_structpool"].get(
                "mean_raw_ttf_relative_improvement", 0.0
            ))
            for row in per_group.values()
        ),
        default=0.0,
    )
    full_selection = float(
        summaries["v2-plus-structpool"]["mean_neighborhood_selection_seconds"]
    )
    lean_selection = float(
        summaries["v2-plus-structpool-lean"]["mean_neighborhood_selection_seconds"]
    )
    selection_improvement = (
        (full_selection - lean_selection) / full_selection
        if full_selection > 0.0
        else 0.0
    )
    performance = {
        "mean_raw_ttf_improvement_vs_full_structpool": float(
            comparison.get("mean_raw_ttf_relative_improvement", -1.0)
        )
        >= float(gates["minimum_mean_raw_ttf_improvement_vs_full_structpool"]),
        "paired_faster_fraction_vs_full_structpool": float(
            comparison.get("paired_faster_fraction", 0.0)
        )
        >= float(gates["minimum_paired_faster_fraction_vs_full_structpool"]),
        "maximum_group_raw_ttf_regression_vs_full_structpool": group_regression
        <= float(gates["maximum_group_raw_ttf_regression_vs_full_structpool"]),
        "mean_repair_iterations_noninferior_to_full_structpool": float(
            comparison.get("mean_repair_iterations_delta", 1.0)
        )
        <= float(gates["maximum_mean_repair_iterations_delta_vs_full_structpool"]),
        "success_count_noninferior_to_full_structpool": (
            int(summaries["v2-plus-structpool-lean"]["success_count"])
            - int(summaries["v2-plus-structpool"]["success_count"])
        )
        >= int(gates["minimum_success_count_delta_vs_full_structpool"]),
        "selection_seconds_improvement_vs_full_structpool": selection_improvement
        >= float(gates["minimum_selection_seconds_improvement_vs_full_structpool"]),
    }
    integrity_passed = not errors and all(integrity.values())
    performance_passed = integrity_passed and all(performance.values())
    report = {
        "schema": report_schema,
        "scientific_status": report_scientific_status,
        "formal_speed_claim": False,
        "producer_identity": producer,
        "fresh_map_claim": False,
        "default_replacement_allowed": False,
        "primary_metric": "mean_run_to_completion_raw_wall_ttf",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparisons": overall,
        "per_group": per_group,
        "structpool_runtime_counts": pool_counts,
        "selection_seconds_relative_improvement_vs_full_structpool": (
            selection_improvement
        ),
        "integrity_gates": integrity,
        "performance_gates": performance,
        "integrity_passed": integrity_passed,
        "performance_passed": performance_passed,
        "next_step": next_step_on_pass if performance_passed else next_step_on_failure,
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "controller_manifest_sha256": dict(hashes),
        },
    }
    _write_json(output / report_filename, report)
    return report


def analyze_structpool_lean_quick(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config = load_structpool_lean_quick_config(config_path)
    return analyze_structpool_lean_multigroup(
        path,
        config,
        output,
        status_filename="quick_status.json",
        report_schema=REPORT_SCHEMA,
        report_filename="structpool_lean_quick_report.json",
        report_scientific_status="development_four_controller_lean_quick",
        next_step_on_pass="retain_lean_runtime_and_preregister_fresh_map_confirmation",
        next_step_on_failure="retain_speed2_full_structpool_and_reject_lean_as_default",
        producer_source_files=("experiments/stride_structpool_lean_quick.py",),
    )


__all__ = [
    "CONTROLLERS",
    "analyze_structpool_lean_multigroup",
    "analyze_structpool_lean_quick",
    "load_structpool_lean_quick_config",
    "run_structpool_lean_multigroup",
    "run_structpool_lean_quick",
    "structpool_lean_schedule",
]
