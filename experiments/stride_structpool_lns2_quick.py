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
from experiments.stride_maprank_raw_ttf import _paired_comparison
from experiments.stride_structpool_ttf_quick import (
    TTF_CLOCK_SCHEMA,
    _controller_kwargs as _v2_controller_kwargs,
    _quick_controller_summary,
    _structpool_trace_counts,
    load_structpool_ttf_quick_config,
)
from lns2_selector.runtime.online_selection import validate_structpool_augmentation


CONFIG_SCHEMA = "lns2.stride.structpool_lns2_quick_config.v1"
STATUS_SCHEMA = "lns2.stride.structpool_lns2_quick_status.v1"
REPORT_SCHEMA = "lns2.stride.structpool_lns2_quick_report.v1"
CONTROLLERS = ("official_adaptive", "v2-full", "v2-plus-structpool")
PHASES = {
    "official_adaptive": "official_adaptive",
    "v2-full": "realized_dynamic",
    "v2-plus-structpool": "realized_dynamic",
}


def _registered(root: Path, specification: dict[str, Any]) -> Path:
    path = (root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered StructPool/LNS2 input changed: {path}")
    return path


def load_structpool_lns2_quick_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_development_three_controller_quick_before_timing"
        or config.get("experiment_id") != "stride-structpool-lns2-quick-v1"
        or config.get("pre_registration_parent_commit")
        != "967ec82073d2b55f780c7f39041f01a0996df50b"
        or tuple(config.get("controllers") or ()) != CONTROLLERS
    ):
        raise ValueError("StructPool/LNS2 Quick identity changed")
    if dict(config.get("comparison") or {}) != {
        "lns2_baseline": "native_official_adaptive_neighborhood_generation",
        "v2_baseline": "frozen_v2_ranking_over_exact_base_pool",
        "challenger": "same_frozen_v2_ranking_over_base_plus_optimized_structpool",
        "execution_order": "strict_three_controller_rotation",
        "paired_solver_seed_required": True,
        "workers": 1,
    }:
        raise ValueError("StructPool/LNS2 Quick comparison changed")
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
        raise ValueError("StructPool/LNS2 Quick runtime changed")
    validate_structpool_augmentation(dict(config["structpool_augmentation"]))
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role") != "unchanged_development_quick_not_formal_ood"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2)
        or int(cohort.get("episode_count_per_controller", -1)) != 8
        or [str(row.get("id")) for row in groups] != ["maze300", "room500"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
    ):
        raise ValueError("StructPool/LNS2 Quick cohort changed")
    expected_inputs = {
        "source_quick_config",
        "runtime_config",
        "controller_manifest",
        "maze_dataset_summary",
        "room_dataset_summary",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("StructPool/LNS2 Quick input registry changed")
    registered = {
        name: _registered(root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    _source_path, _source_root, source = load_structpool_ttf_quick_config(
        registered["source_quick_config"]
    )
    source_groups = {
        str(row["id"]): (str(row["dataset"]), str(row["split"]), list(row["tasks"]))
        for row in source["cohort"]["groups"]
    }
    current_groups = {
        str(row["id"]): (str(row["dataset"]), str(row["split"]), list(row["tasks"]))
        for row in groups
    }
    if current_groups != source_groups:
        raise ValueError("StructPool/LNS2 Quick changed the registered cohort")
    if dict(config["structpool_augmentation"]) != dict(
        source["structpool_augmentation"]
    ):
        raise ValueError("StructPool/LNS2 Quick changed StructPool semantics")
    if str(config["controller_bundle"]) != str(source["controller_bundle"]):
        raise ValueError("StructPool/LNS2 Quick changed the V2 bundle")
    return path, root, config


def structpool_lns2_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
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
    return _v2_controller_kwargs(root, config, controller)


def _manifest_path(collection: Path, controller: str) -> Path:
    return collection / f"{PHASES[controller]}_manifest.jsonl"


def run_structpool_lns2_quick(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_structpool_lns2_quick_config(config_path)
    output = Path(output).resolve()
    schedule = structpool_lns2_schedule(config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    status_path = output / "quick_status.json"
    if status_path.is_file() and not resume:
        raise ValueError("StructPool/LNS2 Quick output exists; pass --resume")
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
    runtime = (root / str(config["runtime"]["config"])).resolve()
    groups = {str(row["id"]): dict(row) for row in config["cohort"]["groups"]}
    seeds = tuple(map(int, config["cohort"]["solver_seeds"]))
    for group in groups.values():
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
            collection = (
                output
                / "groups"
                / str(group["id"])
                / "controllers"
                / controller
            )
            run_closed_loop_collection(
                dataset,
                runtime,
                collection,
                phase="qualify",
                workers=1,
                resume=collection.joinpath("run_config.json").is_file(),
                cohort_job_keys=keys,
                job_keys=keys,
                qualification_source=qualification,
                **_controller_kwargs(root, config, controller),
            )
    completed = 0
    for item in schedule:
        group = groups[str(item["group_id"])]
        dataset = (root / str(group["dataset"])).resolve()
        keys = {(str(task), seed) for task in group["tasks"] for seed in seeds}
        controller = str(item["controller"])
        collection = (
            output
            / "groups"
            / str(item["group_id"])
            / "controllers"
            / controller
        )
        manifest = _manifest_path(collection, controller)
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
    report = analyze_structpool_lns2_quick(path, output)
    _write_json(
        status_path,
        {
            **status_base,
            "completed_schedule_entries": completed,
            "complete": True,
            "report_sha256": sha256_file(
                output / "structpool_lns2_quick_report.json"
            ),
        },
    )
    return report


def analyze_structpool_lns2_quick(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, _root, config = load_structpool_lns2_quick_config(config_path)
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
        by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for group in groups:
            group_id = str(group["id"])
            collection = output / "groups" / group_id / "controllers" / controller
            manifest = _manifest_path(collection, controller)
            rows = _read_jsonl(manifest)
            hashes[controller][group_id] = sha256_file(manifest)
            for row in rows:
                key = (group_id, str(row["task_id"]), int(row["solver_seed"]))
                if key in by_key:
                    errors.append(f"{controller}: duplicate episode {key}")
                by_key[key] = row
                if controller == "v2-plus-structpool" and row.get("status") == "ok":
                    counts = _structpool_trace_counts(collection, row)
                    for name, value in counts.items():
                        structpool_counts[name] += value
        if set(by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = by_key
    fingerprint_mismatches = 0
    conflict_mismatches = 0
    bad_clock = 0
    capped_values = 0
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

    def comparisons(selected_keys: list[tuple[str, str, int]]) -> dict[str, Any]:
        return {
            "v2_vs_lns2": _paired_comparison(
                indexed["official_adaptive"], indexed["v2-full"], selected_keys
            ),
            "structpool_vs_lns2": _paired_comparison(
                indexed["official_adaptive"],
                indexed["v2-plus-structpool"],
                selected_keys,
            ),
            "structpool_vs_v2": _paired_comparison(
                indexed["v2-full"], indexed["v2-plus-structpool"], selected_keys
            ),
        }

    overall = comparisons(keys)
    per_group = {
        str(group["id"]): comparisons(
            [key for key in keys if key[0] == str(group["id"])]
        )
        for group in groups
    }
    integrity = {
        "complete_three_controller_coverage": not any(
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
        "structpool_activated": structpool_counts["gate_passed"] > 0,
        "structpool_selected": structpool_counts["selected"] > 0,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "development_three_controller_quick",
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "primary_metric": "mean_run_to_completion_raw_wall_ttf",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparisons": overall,
        "per_group": per_group,
        "structpool_runtime_counts": structpool_counts,
        "integrity_gates": integrity,
        "integrity_passed": not errors and all(integrity.values()),
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "controller_manifest_sha256": dict(hashes),
        },
    }
    _write_json(output / "structpool_lns2_quick_report.json", report)
    return report


__all__ = [
    "analyze_structpool_lns2_quick",
    "load_structpool_lns2_quick_config",
    "run_structpool_lns2_quick",
    "structpool_lns2_schedule",
]
