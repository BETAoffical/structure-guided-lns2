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
from experiments.stride_augcontrol_evaluation import _dataset_tasks
from experiments.stride_maprank_raw_ttf import _paired_comparison
from experiments.stride_structpool_ttf_quick import (
    CONTROLLERS,
    TTF_CLOCK_SCHEMA,
    _controller_kwargs,
    _quick_controller_summary,
    _structpool_trace_counts,
    structpool_ttf_schedule,
)
from lns2_selector.runtime.online_selection import validate_structpool_augmentation


CONFIG_SCHEMA = "lns2.stride.structpool_ttf_fresh_config.v1"
STATUS_SCHEMA = "lns2.stride.structpool_ttf_fresh_status.v1"
REPORT_SCHEMA = "lns2.stride.structpool_ttf_fresh_report.v1"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    path = (root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered StructPool fresh input changed: {path}")
    return path


def load_structpool_ttf_fresh_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_fresh_map_raw_ttf_after_four_seed_development_pass_before_fresh_timing"
        or config.get("experiment_id") != "stride-structpool-ttf-fresh-v1"
        or config.get("pre_registration_parent_commit")
        != "77679eedb9b45f184895515efa9865de50b3f3b6"
        or tuple(config.get("controllers") or ()) != CONTROLLERS
    ):
        raise ValueError("StructPool fresh TTF identity changed")
    if dict(config.get("comparison") or {}) != {
        "baseline": "frozen_v2_ranking_over_exact_base_pool",
        "challenger": "same_frozen_v2_ranking_over_base_plus_structpool",
        "ranker_changed": False,
        "paired_solver_seed_required": True,
        "deterministic_pp_seed_contract": "candidate_bound_native_replay",
        "execution_order": "alternating_pair_order",
        "workers": 1,
    }:
        raise ValueError("StructPool fresh TTF comparison changed")
    if dict(config.get("runtime") or {}) != {
        "config": "configs/stride_structpool_fresh_runtime.json",
        "stopping_rule": "run-to-completion",
        "scientific_time_limit_seconds": None,
        "environment_time_limit_seconds": None,
        "episode_process_timeout_seconds": None,
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "deterministic_pp_replay": True,
    }:
        raise ValueError("StructPool fresh TTF runtime changed")
    validate_structpool_augmentation(dict(config["structpool_augmentation"]))
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role")
        != "structpool_unseen_cross_layout_raw_ttf_not_used_for_training_or_threshold_selection"
        or cohort.get("dataset") != "build/stride-augcontrol-ood-dataset-v1"
        or cohort.get("split") != "balanced_wall_clock"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2, 3)
        or int(cohort.get("episode_count_per_controller", -1)) != 36
        or [str(row.get("id")) for row in groups]
        != ["maze200", "room400", "random500", "warehouse500", "den300", "lak500"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
    ):
        raise ValueError("StructPool fresh TTF cohort changed")
    if dict(config.get("performance_gates") or {}) != {
        "minimum_mean_raw_ttf_improvement_vs_v2": 0.05,
        "maximum_group_raw_ttf_regression": 0.10,
        "minimum_paired_faster_fraction": 0.50,
        "repair_iterations_noninferior": True,
        "success_count_noninferior": True,
    }:
        raise ValueError("StructPool fresh TTF gates changed")
    expected_inputs = {
        "confirmation_config",
        "confirmation_report",
        "runtime_config",
        "controller_manifest",
        "dataset_config",
        "fetched_manifest",
        "dataset_summary",
        "dataset_manifest",
        "label_state_manifest",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("StructPool fresh TTF input registry changed")
    for specification in dict(config["inputs"]).values():
        _registered(root, dict(specification))
    confirmation = _read_json(
        root / str(config["inputs"]["confirmation_report"]["path"])
    )
    if (
        confirmation.get("integrity_passed") is not True
        or confirmation.get("performance_passed") is not True
        or confirmation.get("next_step") != "preregister_fresh_map_raw_ttf"
    ):
        raise ValueError("StructPool confirmation did not authorize fresh-map TTF")
    indexed = _dataset_tasks(
        (root / str(cohort["dataset"])).resolve(), str(cohort["split"])
    )
    expected_tasks = {
        str(task) for group in groups for task in list(group.get("tasks") or ())
    }
    if set(indexed) != expected_tasks:
        raise ValueError("StructPool fresh dataset task coverage changed")
    for group in groups:
        if {
            str(indexed[str(task)]["map_id"]) for task in group["tasks"]
        } != {str(group["map_id"])}:
            raise ValueError("StructPool fresh task-to-map identity changed")
    label_maps = {
        str(row["map_id"])
        for row in _read_jsonl(
            root / str(config["inputs"]["label_state_manifest"]["path"])
        )
    }
    fresh_maps = {str(group["map_id"]) for group in groups}
    if fresh_maps & label_maps:
        raise ValueError("StructPool fresh maps overlap the label cohort")
    confirmation_config = _read_json(
        root / str(config["inputs"]["confirmation_config"]["path"])
    )
    development_maps: set[str] = set()
    for group in confirmation_config["cohort"]["groups"]:
        source_rows = _dataset_tasks(
            (root / str(group["dataset"])).resolve(), str(group["split"])
        )
        development_maps.update(
            str(source_rows[str(task)]["map_id"]) for task in group["tasks"]
        )
    if fresh_maps & development_maps:
        raise ValueError("StructPool fresh maps overlap the development TTF cohort")
    return path, root, config


def structpool_fresh_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    return structpool_ttf_schedule(config)


def run_structpool_ttf_fresh(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_structpool_ttf_fresh_config(config_path)
    output = Path(output).resolve()
    schedule = structpool_fresh_schedule(config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    status_path = output / "fresh_status.json"
    if status_path.is_file() and not resume:
        raise ValueError("StructPool fresh TTF output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "execution_schedule.jsonl", schedule)
    base_status = {
        "schema": STATUS_SCHEMA,
        "config_sha256": sha256_file(path),
        "schedule_sha256": _fingerprint(schedule),
        "total_schedule_entries": len(schedule),
    }
    _write_json(
        status_path,
        {**base_status, "completed_schedule_entries": 0, "complete": False},
    )
    cohort = dict(config["cohort"])
    dataset = (root / str(cohort["dataset"])).resolve()
    runtime = (root / str(config["runtime"]["config"])).resolve()
    seeds = tuple(map(int, cohort["solver_seeds"]))
    all_keys = {
        (str(task), seed)
        for group in cohort["groups"]
        for task in group["tasks"]
        for seed in seeds
    }
    qualification = output / "qualification"
    run_closed_loop_collection(
        dataset,
        runtime,
        qualification,
        phase="qualify",
        workers=1,
        resume=qualification.joinpath("run_config.json").is_file(),
        cohort_job_keys=all_keys,
        job_keys=all_keys,
        **_controller_kwargs(root, config, "v2-full"),
    )
    for controller in CONTROLLERS:
        controller_root = output / "controllers" / controller
        run_closed_loop_collection(
            dataset,
            runtime,
            controller_root,
            phase="qualify",
            workers=1,
            resume=controller_root.joinpath("run_config.json").is_file(),
            cohort_job_keys=all_keys,
            job_keys=all_keys,
            qualification_source=qualification,
            **_controller_kwargs(root, config, controller),
        )
    completed = 0
    for item in schedule:
        controller = str(item["controller"])
        controller_root = output / "controllers" / controller
        manifest = controller_root / "realized_dynamic_manifest.jsonl"
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
                controller_root,
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=all_keys,
                job_keys={key},
                **_controller_kwargs(root, config, controller),
            )
        completed += 1
        _write_json(
            status_path,
            {
                **base_status,
                "completed_schedule_entries": completed,
                "current": item,
                "complete": False,
            },
        )
    report = analyze_structpool_ttf_fresh(path, output)
    _write_json(
        status_path,
        {
            **base_status,
            "completed_schedule_entries": completed,
            "complete": True,
            "report_sha256": sha256_file(
                output / "structpool_ttf_fresh_report.json"
            ),
        },
    )
    return report


def analyze_structpool_ttf_fresh(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config = load_structpool_ttf_fresh_config(config_path)
    output = Path(output).resolve()
    groups = [dict(row) for row in config["cohort"]["groups"]]
    seeds = tuple(map(int, config["cohort"]["solver_seeds"]))
    expected = {
        (str(group["id"]), str(task), seed)
        for group in groups
        for task in group["tasks"]
        for seed in seeds
    }
    group_by_task = {
        str(task): str(group["id"])
        for group in groups
        for task in group["tasks"]
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    manifest_hashes: dict[str, str] = {}
    errors: list[str] = []
    structpool_counts = {"selected": 0, "gate_passed": 0, "added_candidates": 0}
    for controller in CONTROLLERS:
        collection = output / "controllers" / controller
        manifest = collection / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(manifest)
        manifest_hashes[controller] = sha256_file(manifest)
        rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for row in rows:
            task = str(row["task_id"])
            key = (group_by_task.get(task, ""), task, int(row["solver_seed"]))
            if key in rows_by_key:
                errors.append(f"{controller}: duplicate episode {key}")
            rows_by_key[key] = row
            if controller == "v2-plus-structpool" and row.get("status") == "ok":
                observed = _structpool_trace_counts(collection, row)
                for name, value in observed.items():
                    structpool_counts[name] += value
        if set(rows_by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = rows_by_key
    fingerprint_mismatches = conflict_mismatches = bad_clock = capped_values = 0
    for key in sorted(expected):
        rows = [indexed[controller].get(key) for controller in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += (
            len({str(row.get("initial_fingerprint")) for row in summaries}) != 1
        )
        conflict_mismatches += (
            len({int(row.get("initial_conflicts", -1)) for row in summaries}) != 1
        )
        bad_clock += sum(
            row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries
        )
        capped_values += sum(
            row.get("capped_wall_time_to_feasible") is not None for row in summaries
        )
    summaries = {
        controller: _quick_controller_summary(list(indexed[controller].values()))
        for controller in CONTROLLERS
    }
    keys = sorted(expected)
    comparison = _paired_comparison(
        indexed["v2-full"], indexed["v2-plus-structpool"], keys
    )
    per_group = {
        str(group["id"]): _paired_comparison(
            indexed["v2-full"],
            indexed["v2-plus-structpool"],
            [key for key in keys if key[0] == str(group["id"])],
        )
        for group in groups
    }
    qualification_report = _read_json(output / "qualification" / "qualification_report.json")
    integrity = {
        "formal_qualification_passed": qualification_report.get("passed") is True,
        "complete_paired_coverage": not any("coverage" in error for error in errors),
        "zero_execution_errors": all(
            row.get("status") == "ok"
            for values in indexed.values()
            for row in values.values()
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
        "structpool_activated_and_selected": structpool_counts["gate_passed"] > 0
        and structpool_counts["selected"] > 0,
    }
    gates = dict(config["performance_gates"])
    maximum_regression = max(
        (
            -float(row.get("mean_raw_ttf_relative_improvement", 0.0))
            for row in per_group.values()
        ),
        default=0.0,
    )
    performance = {
        "mean_raw_ttf_improvement": float(
            comparison.get("mean_raw_ttf_relative_improvement", -1.0)
        )
        >= float(gates["minimum_mean_raw_ttf_improvement_vs_v2"]),
        "maximum_group_regression": maximum_regression
        <= float(gates["maximum_group_raw_ttf_regression"]),
        "paired_faster_fraction": float(
            comparison.get("paired_faster_fraction", 0.0)
        )
        >= float(gates["minimum_paired_faster_fraction"]),
        "repair_iterations_noninferior": float(
            comparison.get("mean_repair_iterations_delta", 1.0)
        )
        <= 0.0,
        "success_count_noninferior": summaries["v2-plus-structpool"][
            "success_count"
        ]
        >= summaries["v2-full"]["success_count"],
    }
    integrity_passed = not errors and all(integrity.values())
    performance_passed = integrity_passed and all(performance.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "fresh_map_cross_layout_raw_ttf_confirmation",
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "primary_metric": "mean_run_to_completion_raw_wall_ttf",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparison": comparison,
        "per_group": per_group,
        "structpool_runtime_counts": structpool_counts,
        "integrity_gates": integrity,
        "performance_gates": performance,
        "integrity_passed": integrity_passed,
        "performance_passed": performance_passed,
        "cross_layout_generalization_supported": performance_passed,
        "next_step": (
            "preregister_independent_replication_and_default_promotion_design"
            if performance_passed
            else "stop_structpool_ttf_promotion_and_diagnose_fresh_map_failures"
        ),
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "qualification_report_sha256": sha256_file(
                output / "qualification" / "qualification_report.json"
            ),
            "controller_manifest_sha256": manifest_hashes,
        },
    }
    _write_json(output / "structpool_ttf_fresh_report.json", report)
    return report


__all__ = [
    "analyze_structpool_ttf_fresh",
    "load_structpool_ttf_fresh_config",
    "run_structpool_ttf_fresh",
    "structpool_fresh_schedule",
]
