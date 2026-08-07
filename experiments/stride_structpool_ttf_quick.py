from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.repair_collection import _fingerprint, _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_augcontrol_evaluation import _dataset_tasks, _metric
from experiments.stride_maprank_raw_ttf import _controller_summary, _paired_comparison
from experiments.stride_robuststep_preflight import _mean
from lns2_selector.runtime.online_selection import validate_structpool_augmentation


CONFIG_SCHEMA = "lns2.stride.structpool_ttf_quick_config.v1"
STATUS_SCHEMA = "lns2.stride.structpool_ttf_quick_status.v1"
REPORT_SCHEMA = "lns2.stride.structpool_ttf_quick_report.v1"
CONTROLLERS = ("v2-full", "v2-plus-structpool")
TTF_CLOCK_SCHEMA = "lns2.ttf.reset_inclusive_wall.v1"


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    path = (root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered StructPool TTF input changed: {path}")
    return path


def load_structpool_ttf_quick_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_development_quick_before_closed_loop_timing"
        or config.get("experiment_id") != "stride-structpool-ttf-quick-v1"
        or config.get("pre_registration_parent_commit")
        != "88413bfe8311b03268a09cb93cbb5a21f7cd1603"
        or tuple(config.get("controllers") or ()) != CONTROLLERS
    ):
        raise ValueError("StructPool TTF Quick identity changed")
    comparison = dict(config.get("comparison") or {})
    if comparison != {
        "baseline": "frozen_v2_ranking_over_exact_base_pool",
        "challenger": "same_frozen_v2_ranking_over_base_plus_structpool",
        "ranker_changed": False,
        "paired_solver_seed_required": True,
        "deterministic_pp_seed_contract": "candidate_bound_native_replay",
        "execution_order": "alternating_pair_order",
        "workers": 1,
    }:
        raise ValueError("StructPool TTF Quick comparison changed")
    runtime = dict(config.get("runtime") or {})
    if runtime != {
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
        raise ValueError("StructPool TTF Quick runtime changed")
    validate_structpool_augmentation(dict(config["structpool_augmentation"]))
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role") != "development_quick_not_formal_ood"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2)
        or int(cohort.get("episode_count_per_controller", -1)) != 8
        or [str(row.get("id")) for row in groups] != ["maze300", "room500"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
    ):
        raise ValueError("StructPool TTF Quick cohort changed")
    if dict(config.get("performance_gates") or {}) != {
        "minimum_mean_raw_ttf_improvement_vs_v2": 0.02,
        "maximum_group_raw_ttf_regression": 0.10,
        "minimum_paired_faster_fraction": 0.50,
        "repair_iterations_noninferior": True,
        "success_count_noninferior": True,
    }:
        raise ValueError("StructPool TTF Quick performance gates changed")
    if set(config.get("inputs") or {}) != {
        "structpool_design",
        "pool_effect_report",
        "source_evaluation_config",
        "runtime_config",
        "controller_manifest",
        "maze_dataset_summary",
        "room_dataset_summary",
    }:
        raise ValueError("StructPool TTF Quick input registry changed")
    for specification in dict(config["inputs"]).values():
        _registered(root, dict(specification))
    effect = _read_json(root / str(config["inputs"]["pool_effect_report"]["path"]))
    if effect.get("passed_for_paired_ttf_quick") is not True:
        raise ValueError("StructPool pool-effect gate did not authorize TTF Quick")
    source = _read_json(root / str(config["inputs"]["source_evaluation_config"]["path"]))
    registered = dict(source["high_load_development"])
    if tuple(map(int, registered["solver_seeds"][:2])) != (1, 2):
        raise ValueError("StructPool TTF Quick seed prefix changed")
    expected_groups = {
        str(row["id"]): (str(row["dataset"]), str(row["split"]), list(row["tasks"]))
        for row in registered["cohorts"]
    }
    observed_groups = {
        str(row["id"]): (str(row["dataset"]), str(row["split"]), list(row["tasks"]))
        for row in groups
    }
    if observed_groups != expected_groups:
        raise ValueError("StructPool TTF Quick development cohort changed")
    for group in groups:
        dataset = (root / str(group["dataset"])).resolve()
        indexed = _dataset_tasks(dataset, str(group["split"]))
        if set(map(str, group["tasks"])) - set(indexed):
            raise ValueError("StructPool TTF Quick task is absent from dataset")
    return path, root, config


def structpool_ttf_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    ]
    schedule = []
    for index, (group, task, seed) in enumerate(keys):
        order = CONTROLLERS if index % 2 == 0 else tuple(reversed(CONTROLLERS))
        schedule.extend(
            {
                "group_id": group,
                "task_id": task,
                "solver_seed": seed,
                "controller": controller,
                "pair_order": position,
            }
            for position, controller in enumerate(order)
        )
    return schedule


def _controller_kwargs(
    root: Path, config: dict[str, Any], controller: str
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "controller": "v2-full",
        "controller_bundle": str((root / str(config["controller_bundle"])).resolve()),
        "feature_backend": str(config["runtime"]["feature_backend"]),
        "controller_runtime": str(config["runtime"]["controller_runtime"]),
        "verification_profile": str(config["runtime"]["verification_profile"]),
        "stopping_rule": "run-to-completion",
    }
    if controller == "v2-plus-structpool":
        result["structpool_augmentation"] = dict(config["structpool_augmentation"])
    elif controller != "v2-full":
        raise ValueError(f"unexpected StructPool TTF controller: {controller}")
    return result


def run_structpool_ttf_quick(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_structpool_ttf_quick_config(config_path)
    output = Path(output).resolve()
    schedule = structpool_ttf_schedule(config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    status_path = output / "quick_status.json"
    if status_path.is_file() and not resume:
        raise ValueError("StructPool TTF Quick output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "execution_schedule.jsonl", schedule)
    base_status = {
        "schema": STATUS_SCHEMA,
        "config_sha256": sha256_file(path),
        "schedule_sha256": _fingerprint(schedule),
        "total_schedule_entries": len(schedule),
    }
    _write_json(status_path, {**base_status, "completed_schedule_entries": 0, "complete": False})
    runtime = (root / str(config["runtime"]["config"])).resolve()
    by_group = {str(row["id"]): dict(row) for row in config["cohort"]["groups"]}
    seeds = tuple(map(int, config["cohort"]["solver_seeds"]))
    for group in by_group.values():
        dataset = (root / str(group["dataset"])).resolve()
        keys = {(str(task), seed) for task in group["tasks"] for seed in seeds}
        qualification = output / "groups" / str(group["id"]) / "qualification"
        run_closed_loop_collection(
            dataset,
            runtime,
            qualification,
            phase="qualify",
            workers=1,
            resume=qualification.joinpath("run_config.json").is_file(),
            cohort_job_keys=keys,
            job_keys=keys,
            **_controller_kwargs(root, config, "v2-full"),
        )
        for controller in CONTROLLERS:
            controller_root = output / "groups" / str(group["id"]) / "controllers" / controller
            run_closed_loop_collection(
                dataset,
                runtime,
                controller_root,
                phase="qualify",
                workers=1,
                resume=controller_root.joinpath("run_config.json").is_file(),
                cohort_job_keys=keys,
                job_keys=keys,
                qualification_source=qualification,
                **_controller_kwargs(root, config, controller),
            )
    completed = 0
    for item in schedule:
        group = by_group[str(item["group_id"])]
        dataset = (root / str(group["dataset"])).resolve()
        keys = {(str(task), seed) for task in group["tasks"] for seed in seeds}
        controller = str(item["controller"])
        controller_root = output / "groups" / str(group["id"]) / "controllers" / controller
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
                cohort_job_keys=keys,
                job_keys={key},
                **_controller_kwargs(root, config, controller),
            )
        completed += 1
        _write_json(
            status_path,
            {**base_status, "completed_schedule_entries": completed, "current": item, "complete": False},
        )
    report = analyze_structpool_ttf_quick(path, output)
    _write_json(
        status_path,
        {
            **base_status,
            "completed_schedule_entries": completed,
            "complete": True,
            "report_sha256": sha256_file(output / "structpool_ttf_quick_report.json"),
        },
    )
    return report


def _structpool_trace_counts(collection: Path, row: dict[str, Any]) -> dict[str, int]:
    counts = {"selected": 0, "gate_passed": 0, "added_candidates": 0}
    for event in read_trace_events(collection / str(row["trace_file"])):
        if event.get("event") != "transition":
            continue
        controller = dict(event.get("controller") or {})
        selected = str(controller.get("selected_candidate_id") or "")
        pool = list(controller.get("candidate_pool") or ())
        selected_row = next(
            (candidate for candidate in pool if str(candidate.get("candidate_id")) == selected),
            None,
        )
        if selected_row and any(
            str(family).startswith("structpool-")
            for family in selected_row.get("selection_families", [])
        ):
            counts["selected"] += 1
    summary = dict(row.get("summary") or {})
    totals = dict(summary.get("controller_totals") or {})
    counts["gate_passed"] = int(totals.get("structpool_gate_passed_count", 0))
    counts["added_candidates"] = int(totals.get("structpool_added_candidate_count", 0))
    return counts


def _quick_controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _controller_summary(rows)
    totals = [
        dict(dict(row.get("summary") or {}).get("controller_totals") or {})
        for row in rows
        if row.get("status") == "ok"
    ]
    for name in (
        "candidate_generation_seconds",
        "structpool_analysis_seconds",
        "structpool_gate_seconds",
    ):
        result[f"mean_{name}"] = _mean([_metric(row, name) for row in totals])
    return result


def analyze_structpool_ttf_quick(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config = load_structpool_ttf_quick_config(config_path)
    output = Path(output).resolve()
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
    structpool_counts = {"selected": 0, "gate_passed": 0, "added_candidates": 0}
    for controller in CONTROLLERS:
        rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for group in groups:
            group_id = str(group["id"])
            collection = output / "groups" / group_id / "controllers" / controller
            manifest = collection / "realized_dynamic_manifest.jsonl"
            rows = _read_jsonl(manifest)
            hashes[controller][group_id] = sha256_file(manifest)
            for row in rows:
                key = (group_id, str(row["task_id"]), int(row["solver_seed"]))
                if key in rows_by_key:
                    errors.append(f"{controller}: duplicate episode {key}")
                rows_by_key[key] = row
                if controller == "v2-plus-structpool" and row.get("status") == "ok":
                    counts = _structpool_trace_counts(collection, row)
                    for name, value in counts.items():
                        structpool_counts[name] += value
        if set(rows_by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = rows_by_key
    fingerprint_mismatches = conflict_mismatches = bad_clock = capped_values = 0
    for key in sorted(expected):
        present = [indexed[name].get(key) for name in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in present):
            continue
        summaries = [dict(row["summary"]) for row in present]
        fingerprint_mismatches += len({str(row.get("initial_fingerprint")) for row in summaries}) != 1
        conflict_mismatches += len({int(row.get("initial_conflicts", -1)) for row in summaries}) != 1
        bad_clock += sum(row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries)
        capped_values += sum(row.get("capped_wall_time_to_feasible") is not None for row in summaries)
    summaries = {
        name: _quick_controller_summary(list(indexed[name].values()))
        for name in CONTROLLERS
    }
    all_keys = sorted(expected)
    total = _paired_comparison(indexed["v2-full"], indexed["v2-plus-structpool"], all_keys)
    per_group = {
        str(group["id"]): _paired_comparison(
            indexed["v2-full"],
            indexed["v2-plus-structpool"],
            [key for key in all_keys if key[0] == str(group["id"])],
        )
        for group in groups
    }
    integrity = {
        "complete_paired_coverage": not any("coverage" in error for error in errors),
        "zero_execution_errors": all(
            row.get("status") == "ok" for values in indexed.values() for row in values.values()
        ),
        "all_controllers_succeeded": all(
            summary["success_count"] == len(expected) for summary in summaries.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "no_capped_ttf_values": capped_values == 0,
        "zero_invalid_actions": all(summary["invalid_action_count"] == 0 for summary in summaries.values()),
        "zero_semantic_mismatches": all(
            summary["fingerprint_mismatch_count"] == 0 for summary in summaries.values()
        ),
        "structpool_activated": structpool_counts["gate_passed"] > 0,
        "structpool_selected": structpool_counts["selected"] > 0,
    }
    gates = dict(config["performance_gates"])
    maximum_regression = max(
        (-float(row.get("mean_raw_ttf_relative_improvement", 0.0)) for row in per_group.values()),
        default=0.0,
    )
    performance = {
        "mean_raw_ttf_improvement": float(total.get("mean_raw_ttf_relative_improvement", -1.0))
        >= float(gates["minimum_mean_raw_ttf_improvement_vs_v2"]),
        "maximum_group_regression": maximum_regression <= float(gates["maximum_group_raw_ttf_regression"]),
        "paired_faster_fraction": float(total.get("paired_faster_fraction", 0.0))
        >= float(gates["minimum_paired_faster_fraction"]),
        "repair_iterations_noninferior": float(total.get("mean_repair_iterations_delta", 1.0)) <= 0.0,
        "success_count_noninferior": summaries["v2-plus-structpool"]["success_count"]
        >= summaries["v2-full"]["success_count"],
    }
    integrity_passed = not errors and all(integrity.values())
    performance_passed = integrity_passed and all(performance.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "development_quick",
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "primary_metric": "mean_run_to_completion_raw_wall_ttf",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparison": total,
        "per_group": per_group,
        "structpool_runtime_counts": structpool_counts,
        "integrity_gates": integrity,
        "performance_gates": performance,
        "integrity_passed": integrity_passed,
        "performance_passed": performance_passed,
        "next_step": (
            "run_full_four_seed_development_confirmation"
            if performance_passed
            else "stop_ttf_expansion_and_diagnose_pool_activation_selection_or_overhead"
        ),
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "controller_manifest_sha256": dict(hashes),
        },
    }
    _write_json(output / "structpool_ttf_quick_report.json", report)
    return report


__all__ = [
    "analyze_structpool_ttf_quick",
    "load_structpool_ttf_quick_config",
    "run_structpool_ttf_quick",
    "structpool_ttf_schedule",
]
