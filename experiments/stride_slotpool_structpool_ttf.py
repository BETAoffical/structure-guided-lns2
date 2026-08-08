from __future__ import annotations

from collections import defaultdict
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
from experiments.stride_guardpool_maze_regression import _controller_kwargs
from experiments.stride_maprank_raw_ttf import _paired_comparison
from experiments.stride_structpool_ttf_quick import (
    TTF_CLOCK_SCHEMA,
    _quick_controller_summary,
)
from lns2_selector.runtime.online_selection import (
    slotpool_runtime_augmentation,
    validate_structpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.slotpool_structpool_ttf_diagnostic_config.v1"
STATUS_SCHEMA = "lns2.stride.slotpool_structpool_ttf_diagnostic_status.v1"
REPORT_SCHEMA = "lns2.stride.slotpool_structpool_ttf_diagnostic_report.v1"
CONTROLLERS = ("v2-plus-structpool", "v2-plus-slotpool")
STATUS_FILENAME = "diagnostic_status.json"
REPORT_FILENAME = "slotpool_structpool_ttf_report.json"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    path = (root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered SlotPool TTF input changed: {path}")
    return path


def _expected_keys(config: dict[str, Any]) -> set[tuple[str, str, int]]:
    keys = {
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }
    for exclusion in config["cohort"]["excluded_keys"]:
        keys.discard(
            (
                str(exclusion["group_id"]),
                str(exclusion["task_id"]),
                int(exclusion["solver_seed"]),
            )
        )
    return keys


def load_slotpool_structpool_ttf_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_known_tail_excluded_diagnostic_before_timing"
        or config.get("experiment_id")
        != "stride-slotpool-structpool-ttf-diagnostic-v1"
        or config.get("pre_registration_parent_commit")
        != "d91780411c277613d309ec4b8b0380d921d499b5"
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("SlotPool versus StructPool TTF identity changed")
    if dict(config.get("comparison") or {}) != {
        "baseline": "frozen_v2_over_base_plus_full_structpool",
        "treatment": "frozen_v2_over_base_plus_frozen_slotpool_six_challengers",
        "execution_order": "strict_two_controller_alternation",
        "paired_solver_seed_required": True,
        "workers": 1,
    }:
        raise ValueError("SlotPool versus StructPool comparison changed")
    if dict(config.get("runtime") or {}) != {
        "config": "configs/stride_structpool_lean_confirmation_runtime.json",
        "stopping_rule": "run-to-completion",
        "scientific_time_limit_seconds": None,
        "environment_time_limit_seconds": None,
        "episode_process_timeout_seconds": None,
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "deterministic_pp_replay": True,
        "workers": 1,
    }:
        raise ValueError("SlotPool versus StructPool runtime changed")
    full = validate_structpool_augmentation(
        dict(config["full_structpool_augmentation"])
    )
    slot = slotpool_runtime_augmentation()
    if (
        full is None
        or full.get("pool_id") != "stride-structpool-v1"
        or slot.get("pool_id") != "stride-slotpool-v1"
    ):
        raise ValueError("SlotPool versus StructPool treatments changed")
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    exclusions = list(cohort.get("excluded_keys") or ())
    if (
        cohort.get("role")
        != "fixed_five_map_cohort_minus_one_previously_known_catastrophic_tail"
        or cohort.get("dataset")
        != "build/stride-structpool-revised-six-map-dataset-v1"
        or cohort.get("split") != "balanced_wall_clock"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2, 3)
        or int(cohort.get("paired_key_count", -1)) != 29
        or int(cohort.get("episode_count_per_controller", -1)) != 29
        or [str(row.get("id")) for row in groups]
        != ["den300", "maze100", "random500", "room400", "warehouse600"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
        or exclusions
        != [
            {
                "group_id": "maze100",
                "task_id": "maze-128-128-1__derived_opposite_exchange__task_seed_0233__agents_0100",
                "solver_seed": 3,
                "reason": "preexisting_known_long_tail_requested_exclusion",
            }
        ]
        or len(_expected_keys(config)) != 29
    ):
        raise ValueError("SlotPool versus StructPool cohort changed")
    if dict(config.get("diagnostic_gates") or {}) != {
        "minimum_success_count_delta_vs_full_structpool": 0,
        "minimum_mean_raw_ttf_improvement_vs_full_structpool": 0.0,
        "minimum_paired_faster_fraction_vs_full_structpool": 0.5,
        "maximum_group_raw_ttf_regression_vs_full_structpool": 0.1,
        "maximum_mean_repair_iterations_delta_vs_full_structpool": 0.0,
    }:
        raise ValueError("SlotPool versus StructPool diagnostic gates changed")
    if dict(config.get("claim_boundary") or {}) != {
        "known_outcome_based_exclusion": True,
        "formal_speed_claim": False,
        "fresh_map_claim": False,
        "default_replacement_allowed": False,
        "diagnostic_comparison_only": True,
        "no_additional_result_based_exclusions": True,
    }:
        raise ValueError("SlotPool versus StructPool claim boundary changed")
    expected_inputs = {
        "dataset_summary",
        "dataset_manifest",
        "runtime_config",
        "controller_manifest",
        "slotpool_model",
        "slotpool_fresh_report",
        "known_tail_report",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("SlotPool versus StructPool input registry changed")
    inputs = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    if str(slot["slotpool_model"]["sha256"]) != str(
        config["inputs"]["slotpool_model"]["sha256"]
    ):
        raise ValueError("SlotPool runtime model does not match registration")
    fresh = _read_json(inputs["slotpool_fresh_report"])
    tail = _read_json(inputs["known_tail_report"])
    if fresh.get("acceptance", {}).get("passed") is not True:
        raise ValueError("SlotPool fresh candidate-quality confirmation did not pass")
    if (
        tail.get("regression_passed") is not False
        or tail.get("scientific_status")
        != "known_regression_only_not_generalization_or_ttf_claim"
    ):
        raise ValueError("known-tail exclusion evidence changed")
    manifest_tasks = {
        str(row["task_id"]) for row in _read_jsonl(inputs["dataset_manifest"])
    }
    configured_tasks = {
        str(task) for group in groups for task in group["tasks"]
    }
    if configured_tasks - manifest_tasks:
        raise ValueError("SlotPool versus StructPool task is absent from dataset")
    return path, root, config


def slotpool_structpool_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, key in enumerate(sorted(_expected_keys(config))):
        group_id, task_id, solver_seed = key
        order = CONTROLLERS if index % 2 == 0 else tuple(reversed(CONTROLLERS))
        rows.extend(
            {
                "group_id": group_id,
                "task_id": task_id,
                "solver_seed": solver_seed,
                "controller": controller,
                "within_key_position": position,
            }
            for position, controller in enumerate(order)
        )
    return rows


def run_slotpool_structpool_ttf(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_slotpool_structpool_ttf_config(config_path)
    schedule = slotpool_structpool_schedule(config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "paired_key_count": len(schedule) // 2,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    output = Path(output).resolve()
    status_path = output / STATUS_FILENAME
    if status_path.is_file() and not resume:
        raise ValueError("SlotPool versus StructPool output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "execution_schedule.jsonl", schedule)
    status_base = {
        "schema": STATUS_SCHEMA,
        "config_sha256": sha256_file(path),
        "schedule_sha256": _fingerprint(schedule),
        "total_schedule_entries": len(schedule),
    }
    _write_json(
        status_path,
        {**status_base, "completed_schedule_entries": 0, "complete": False},
    )
    dataset = (root / str(config["cohort"]["dataset"])).resolve()
    runtime = (root / str(config["runtime"]["config"])).resolve()
    expected = _expected_keys(config)
    job_keys = {(task, seed) for _group, task, seed in expected}
    qualification = output / "qualification"
    run_closed_loop_collection(
        dataset,
        runtime,
        qualification,
        phase="qualify",
        workers=1,
        resume=qualification.joinpath("run_config.json").is_file(),
        cohort_job_keys=job_keys,
        job_keys=job_keys,
        **_controller_kwargs(root, config, "v2-full"),
    )
    for controller in CONTROLLERS:
        collection = output / "controllers" / controller
        run_closed_loop_collection(
            dataset,
            runtime,
            collection,
            phase="qualify",
            workers=1,
            resume=collection.joinpath("run_config.json").is_file(),
            cohort_job_keys=job_keys,
            job_keys=job_keys,
            qualification_source=qualification,
            **_controller_kwargs(root, config, controller),
        )
    completed = 0
    for item in schedule:
        controller = str(item["controller"])
        collection = output / "controllers" / controller
        manifest = collection / "realized_dynamic_manifest.jsonl"
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
                phase="realized_dynamic",
                workers=1,
                resume=True,
                cohort_job_keys=job_keys,
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
    report = analyze_slotpool_structpool_ttf(path, output)
    _write_json(
        status_path,
        {
            **status_base,
            "completed_schedule_entries": completed,
            "complete": True,
            "report_sha256": sha256_file(output / REPORT_FILENAME),
        },
    )
    return report


def _pool_counts(row: dict[str, Any]) -> dict[str, int]:
    totals = dict(dict(row.get("summary") or {}).get("controller_totals") or {})
    return {
        "gate_passed": int(totals.get("structpool_gate_passed_count", 0)),
        "structural_selected": int(
            totals.get("guardpool_selected_structural_count", 0)
        ),
        "generated": int(totals.get("structpool_generated_count", 0)),
        "slotpool_reduced": int(
            totals.get("slotpool_reduction_applied_count", 0)
        ),
        "slotpool_raw": int(
            totals.get("slotpool_raw_structural_candidate_count", 0)
        ),
        "slotpool_selected": int(
            totals.get("slotpool_selected_structural_candidate_count", 0)
        ),
    }


def analyze_slotpool_structpool_ttf(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config = load_slotpool_structpool_ttf_config(config_path)
    output = Path(output).resolve()
    expected = _expected_keys(config)
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    hashes: dict[str, str] = {}
    errors: list[str] = []
    counts = {controller: defaultdict(int) for controller in CONTROLLERS}
    for controller in CONTROLLERS:
        manifest = output / "controllers" / controller / "realized_dynamic_manifest.jsonl"
        rows = _read_jsonl(manifest)
        hashes[controller] = sha256_file(manifest)
        by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for row in rows:
            task = str(row["task_id"])
            seed = int(row["solver_seed"])
            matching = [key for key in expected if key[1:] == (task, seed)]
            if len(matching) != 1:
                errors.append(f"{controller}: unexpected episode {(task, seed)}")
                continue
            key = matching[0]
            if key in by_key:
                errors.append(f"{controller}: duplicate episode {key}")
            by_key[key] = row
            for name, value in _pool_counts(row).items():
                counts[controller][name] += value
        if set(by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = by_key

    fingerprint_mismatches = conflict_mismatches = bad_clock = capped = 0
    for key in sorted(expected):
        rows = [indexed[controller].get(key) for controller in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += len(
            {str(row.get("initial_fingerprint")) for row in summaries}
        ) != 1
        conflict_mismatches += len(
            {int(row.get("initial_conflicts", -1)) for row in summaries}
        ) != 1
        bad_clock += sum(
            row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries
        )
        capped += sum(
            row.get("capped_wall_time_to_feasible") is not None
            for row in summaries
        )
    summaries = {
        controller: _quick_controller_summary(list(indexed[controller].values()))
        for controller in CONTROLLERS
    }
    keys = sorted(expected)
    comparison = _paired_comparison(
        indexed["v2-plus-structpool"],
        indexed["v2-plus-slotpool"],
        keys,
    )
    per_group = {
        str(group["id"]): _paired_comparison(
            indexed["v2-plus-structpool"],
            indexed["v2-plus-slotpool"],
            [key for key in keys if key[0] == str(group["id"])],
        )
        for group in config["cohort"]["groups"]
    }
    integrity = {
        "complete_two_controller_coverage": not any(
            "coverage" in error for error in errors
        ),
        "exactly_one_preregistered_key_excluded": len(expected) == 29,
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
        "no_capped_ttf_values": capped == 0,
        "zero_invalid_actions": all(
            summary["invalid_action_count"] == 0 for summary in summaries.values()
        ),
        "zero_semantic_mismatches": all(
            summary["fingerprint_mismatch_count"] == 0
            for summary in summaries.values()
        ),
        "full_structpool_activated": counts["v2-plus-structpool"]["gate_passed"] > 0,
        "slotpool_activated": counts["v2-plus-slotpool"]["gate_passed"] > 0,
        "slotpool_reduction_exercised": counts["v2-plus-slotpool"]["slotpool_reduced"] > 0,
        "slotpool_selected_challengers": counts["v2-plus-slotpool"]["slotpool_selected"] > 0,
    }
    gates = dict(config["diagnostic_gates"])
    maximum_group_regression = max(
        (
            -float(row.get("mean_raw_ttf_relative_improvement", 0.0))
            for row in per_group.values()
        ),
        default=0.0,
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
        "maximum_group_raw_ttf_regression_vs_full_structpool": maximum_group_regression
        <= float(gates["maximum_group_raw_ttf_regression_vs_full_structpool"]),
        "mean_repair_iterations_noninferior_to_full_structpool": float(
            comparison.get("mean_repair_iterations_delta", 1.0)
        )
        <= float(gates["maximum_mean_repair_iterations_delta_vs_full_structpool"]),
        "success_count_noninferior_to_full_structpool": (
            int(summaries["v2-plus-slotpool"]["success_count"])
            - int(summaries["v2-plus-structpool"]["success_count"])
        )
        >= int(gates["minimum_success_count_delta_vs_full_structpool"]),
    }
    integrity_passed = not errors and all(integrity.values())
    performance_passed = integrity_passed and all(performance.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "known_tail_excluded_diagnostic_only",
        "formal_speed_claim": False,
        "fresh_map_claim": False,
        "default_replacement_allowed": False,
        "known_outcome_based_exclusion": True,
        "primary_metric": "mean_run_to_completion_raw_wall_ttf",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "paired_key_count": len(expected),
        "episode_count_per_controller": len(expected),
        "excluded_keys": list(config["cohort"]["excluded_keys"]),
        "controller_summaries": summaries,
        "slotpool_vs_full_structpool": comparison,
        "per_group": per_group,
        "runtime_counts": {
            controller: dict(value) for controller, value in counts.items()
        },
        "integrity_gates": integrity,
        "diagnostic_gates": performance,
        "integrity_passed": integrity_passed,
        "performance_passed": performance_passed,
        "winner_by_mean_raw_ttf": (
            "v2-plus-slotpool"
            if float(summaries["v2-plus-slotpool"]["mean_raw_wall_time_to_feasible"])
            < float(summaries["v2-plus-structpool"]["mean_raw_wall_time_to_feasible"])
            else "v2-plus-structpool"
        ),
        "next_step": "report_diagnostic_without_promotion_claim",
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
    "CONTROLLERS",
    "REPORT_FILENAME",
    "STATUS_FILENAME",
    "analyze_slotpool_structpool_ttf",
    "load_slotpool_structpool_ttf_config",
    "run_slotpool_structpool_ttf",
    "slotpool_structpool_schedule",
]
