from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import (
    closed_loop_producer_identity,
    sha256_file,
)
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import _fingerprint, _read_json, _read_jsonl, _write_json
from experiments.stride_structpool_ttf_quick import (
    CONTROLLERS,
    TTF_CLOCK_SCHEMA,
    _controller_kwargs,
    _quick_controller_summary,
    _structpool_trace_counts,
    load_structpool_ttf_quick_config,
)
from lns2_selector.evaluation.episode_statistics import (
    dataset_tasks as _dataset_tasks,
    paired_raw_ttf_comparison as _paired_comparison,
)
from lns2_selector.runtime.online_selection import validate_structpool_augmentation


CONFIG_SCHEMA = "lns2.stride.structpool_ttf_confirmation_config.v1"
STATUS_SCHEMA = "lns2.stride.structpool_ttf_confirmation_status.v1"
REPORT_SCHEMA = "lns2.stride.structpool_ttf_confirmation_report.v1"


def load_structpool_ttf_confirmation_config(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_seed_extension_after_quick_pass_before_extension_timing"
        or config.get("experiment_id") != "stride-structpool-ttf-confirmation-v1"
        or config.get("pre_registration_parent_commit")
        != "60939747b5c80edc5217f5dc197552a7ccbd3acb"
        or tuple(config.get("controllers") or ()) != CONTROLLERS
        or config.get("quick_output") != "build/stride-structpool-ttf-quick-v1"
    ):
        raise ValueError("StructPool TTF confirmation identity changed")
    runtime = dict(config.get("runtime") or {})
    if runtime != {
        "config": "configs/stride_stage4r_high_load_runtime.json",
        "stopping_rule": "run-to-completion",
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "workers": 1,
        "scientific_time_limit_seconds": None,
        "environment_time_limit_seconds": None,
        "episode_process_timeout_seconds": None,
    }:
        raise ValueError("StructPool TTF confirmation runtime changed")
    validate_structpool_augmentation(dict(config["structpool_augmentation"]))
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role") != "unchanged_development_seed_extension_not_formal_ood"
        or tuple(map(int, cohort.get("quick_solver_seeds") or ())) != (1, 2)
        or tuple(map(int, cohort.get("extension_solver_seeds") or ())) != (3, 4)
        or tuple(map(int, cohort.get("pooled_solver_seeds") or ())) != (1, 2, 3, 4)
        or int(cohort.get("extension_episode_count_per_controller", -1)) != 8
        or int(cohort.get("pooled_episode_count_per_controller", -1)) != 16
        or [str(row.get("id")) for row in groups] != ["maze300", "room500"]
    ):
        raise ValueError("StructPool TTF confirmation cohort changed")
    gate_common = {
        "maximum_group_raw_ttf_regression": 0.10,
        "minimum_paired_faster_fraction": 0.50,
        "repair_iterations_noninferior": True,
        "success_count_noninferior": True,
    }
    if dict(config.get("extension_gates") or {}) != {
        **gate_common,
        "minimum_mean_raw_ttf_improvement_vs_v2": 0.0,
    } or dict(config.get("pooled_gates") or {}) != {
        **gate_common,
        "minimum_mean_raw_ttf_improvement_vs_v2": 0.02,
    }:
        raise ValueError("StructPool TTF confirmation gates changed")
    if set(config.get("inputs") or {}) != {"quick_config", "quick_report"}:
        raise ValueError("StructPool TTF confirmation input registry changed")
    registered = {}
    for name, specification in dict(config["inputs"]).items():
        input_path = (root / str(specification["path"])).resolve()
        if not input_path.is_file() or sha256_file(input_path) != str(
            specification["sha256"]
        ):
            raise ValueError(f"registered StructPool confirmation input changed: {name}")
        registered[name] = input_path
    _quick_path, _quick_root, quick = load_structpool_ttf_quick_config(
        registered["quick_config"]
    )
    quick_report = _read_json(registered["quick_report"])
    if (
        quick_report.get("integrity_passed") is not True
        or quick_report.get("performance_passed") is not True
    ):
        raise ValueError("StructPool Quick did not authorize confirmation")
    if dict(config["structpool_augmentation"]) != dict(
        quick["structpool_augmentation"]
    ) or str(config["controller_bundle"]) != str(quick["controller_bundle"]):
        raise ValueError("StructPool confirmation changed the method")
    quick_groups = {
        str(row["id"]): (str(row["dataset"]), str(row["split"]), list(row["tasks"]))
        for row in quick["cohort"]["groups"]
    }
    current_groups = {
        str(row["id"]): (str(row["dataset"]), str(row["split"]), list(row["tasks"]))
        for row in groups
    }
    if current_groups != quick_groups:
        raise ValueError("StructPool confirmation changed the Quick cohort")
    for group in groups:
        indexed = _dataset_tasks(
            (root / str(group["dataset"])).resolve(), str(group["split"])
        )
        if set(map(str, group["tasks"])) - set(indexed):
            raise ValueError("StructPool confirmation task is absent from dataset")
    return path, root, config


def structpool_confirmation_schedule(config: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["extension_solver_seeds"]
    ]
    rows = []
    for index, (group_id, task_id, solver_seed) in enumerate(keys):
        order = CONTROLLERS if index % 2 == 0 else tuple(reversed(CONTROLLERS))
        rows.extend(
            {
                "group_id": group_id,
                "task_id": task_id,
                "solver_seed": solver_seed,
                "controller": controller,
                "pair_order": pair_order,
            }
            for pair_order, controller in enumerate(order)
        )
    return rows


def _quick_compatible_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "controller_bundle": config["controller_bundle"],
        "runtime": config["runtime"],
        "structpool_augmentation": config["structpool_augmentation"],
    }


def run_structpool_ttf_confirmation(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_structpool_ttf_confirmation_config(config_path)
    output = Path(output).resolve()
    schedule = structpool_confirmation_schedule(config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "schedule_entry_count": len(schedule),
            "schedule_sha256": _fingerprint(schedule),
        }
    status_path = output / "confirmation_status.json"
    prepared = prepare_resumable_output(
        output,
        status_filename=status_path.name,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=schedule,
        producer=closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_structpool_ttf_confirmation.py",
                "experiments/stride_structpool_ttf_quick.py",
                "lns2_selector/evaluation/episode_statistics.py",
            ),
        ),
        resume=resume,
        report_filename="structpool_ttf_confirmation_report.json",
        report_schema=REPORT_SCHEMA,
        label="StructPool TTF confirmation",
    )
    base_status = prepared.base_status
    if prepared.completed_report is not None:
        return prepared.completed_report
    runtime = (root / str(config["runtime"]["config"])).resolve()
    groups = {str(row["id"]): dict(row) for row in config["cohort"]["groups"]}
    seeds = tuple(map(int, config["cohort"]["extension_solver_seeds"]))
    method = _quick_compatible_config(config)
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
            resume=(
                prepared.resumed
                and qualification.joinpath("run_config.json").is_file()
            ),
            cohort_job_keys=keys,
            job_keys=keys,
            **_controller_kwargs(root, method, "v2-full"),
        )
        for controller in CONTROLLERS:
            controller_root = output / "groups" / str(group["id"]) / "controllers" / controller
            run_closed_loop_collection(
                dataset,
                runtime,
                controller_root,
                phase="qualify",
                workers=1,
                resume=(
                    prepared.resumed
                    and controller_root.joinpath("run_config.json").is_file()
                ),
                cohort_job_keys=keys,
                job_keys=keys,
                qualification_source=qualification,
                **_controller_kwargs(root, method, controller),
            )
    completed = 0
    for item in schedule:
        group = groups[str(item["group_id"])]
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
                **_controller_kwargs(root, method, controller),
            )
        completed += 1
        _write_json(
            status_path,
            {**base_status, "completed_schedule_entries": completed, "current": item, "complete": False},
        )
    report = analyze_structpool_ttf_confirmation(
        path,
        output,
        producer=base_status["producer_identity"],
    )
    _write_json(
        status_path,
        {
            **base_status,
            "completed_schedule_entries": completed,
            "complete": True,
            "report_sha256": sha256_file(output / "structpool_ttf_confirmation_report.json"),
        },
    )
    return report


def _load_scope(
    root: Path,
    groups: list[dict[str, Any]],
    seeds: tuple[int, ...],
) -> tuple[
    dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    dict[str, dict[str, str]],
    dict[str, int],
    list[str],
]:
    expected = {
        (str(group["id"]), str(task), seed)
        for group in groups
        for task in group["tasks"]
        for seed in seeds
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    hashes: dict[str, dict[str, str]] = defaultdict(dict)
    counts = {"selected": 0, "gate_passed": 0, "added_candidates": 0}
    errors: list[str] = []
    for controller in CONTROLLERS:
        rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for group in groups:
            group_id = str(group["id"])
            collection = root / "groups" / group_id / "controllers" / controller
            manifest = collection / "realized_dynamic_manifest.jsonl"
            rows = _read_jsonl(manifest)
            hashes[controller][group_id] = sha256_file(manifest)
            for row in rows:
                key = (group_id, str(row["task_id"]), int(row["solver_seed"]))
                if key in rows_by_key:
                    errors.append(f"{controller}: duplicate episode {key}")
                rows_by_key[key] = row
                if controller == "v2-plus-structpool" and row.get("status") == "ok":
                    observed = _structpool_trace_counts(collection, row)
                    for name, value in observed.items():
                        counts[name] += value
        if set(rows_by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = rows_by_key
    return indexed, dict(hashes), counts, errors


def _scope_analysis(
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]],
    groups: list[dict[str, Any]],
    seeds: tuple[int, ...],
    gates: dict[str, Any],
) -> dict[str, Any]:
    expected = sorted(
        (str(group["id"]), str(task), seed)
        for group in groups
        for task in group["tasks"]
        for seed in seeds
    )
    comparison = _paired_comparison(indexed["v2-full"], indexed["v2-plus-structpool"], expected)
    per_group = {
        str(group["id"]): _paired_comparison(
            indexed["v2-full"],
            indexed["v2-plus-structpool"],
            [key for key in expected if key[0] == str(group["id"])],
        )
        for group in groups
    }
    summaries = {
        controller: _quick_controller_summary(list(indexed[controller].values()))
        for controller in CONTROLLERS
    }
    maximum_regression = max(
        (-float(row.get("mean_raw_ttf_relative_improvement", 0.0)) for row in per_group.values()),
        default=0.0,
    )
    performance = {
        "mean_raw_ttf_improvement": float(comparison.get("mean_raw_ttf_relative_improvement", -1.0))
        >= float(gates["minimum_mean_raw_ttf_improvement_vs_v2"]),
        "maximum_group_regression": maximum_regression <= float(gates["maximum_group_raw_ttf_regression"]),
        "paired_faster_fraction": float(comparison.get("paired_faster_fraction", 0.0))
        >= float(gates["minimum_paired_faster_fraction"]),
        "repair_iterations_noninferior": float(comparison.get("mean_repair_iterations_delta", 1.0)) <= 0.0,
        "success_count_noninferior": summaries["v2-plus-structpool"]["success_count"]
        >= summaries["v2-full"]["success_count"],
    }
    return {
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparison": comparison,
        "per_group": per_group,
        "performance_gates": performance,
        "performance_passed": all(performance.values()),
    }


def analyze_structpool_ttf_confirmation(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_structpool_ttf_confirmation_config(config_path)
    output = Path(output).resolve()
    completed = load_completed_report(
        output,
        status_filename="confirmation_status.json",
        report_filename="structpool_ttf_confirmation_report.json",
        status_schema=STATUS_SCHEMA,
        report_schema=REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    if producer is None:
        producer = closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_structpool_ttf_confirmation.py",
                "experiments/stride_structpool_ttf_quick.py",
                "lns2_selector/evaluation/episode_statistics.py",
            ),
            native_required=False,
        )
    quick_root = (root / str(config["quick_output"])).resolve()
    groups = [dict(row) for row in config["cohort"]["groups"]]
    quick_seeds = tuple(map(int, config["cohort"]["quick_solver_seeds"]))
    extension_seeds = tuple(map(int, config["cohort"]["extension_solver_seeds"]))
    pooled_seeds = tuple(map(int, config["cohort"]["pooled_solver_seeds"]))
    quick, quick_hashes, quick_counts, quick_errors = _load_scope(quick_root, groups, quick_seeds)
    extension, extension_hashes, extension_counts, extension_errors = _load_scope(
        output, groups, extension_seeds
    )
    quick_report = _read_json(root / str(config["inputs"]["quick_report"]["path"]))
    quick_hash_match = quick_hashes == dict(
        quick_report["inputs"]["controller_manifest_sha256"]
    )
    pooled = {
        controller: {**quick[controller], **extension[controller]}
        for controller in CONTROLLERS
    }
    expected = {
        (str(group["id"]), str(task), seed)
        for group in groups
        for task in group["tasks"]
        for seed in pooled_seeds
    }
    errors = [*quick_errors, *extension_errors]
    if any(set(pooled[controller]) != expected for controller in CONTROLLERS):
        errors.append("pooled four-seed coverage differs")
    fingerprint_mismatches = conflict_mismatches = bad_clock = capped_values = 0
    for key in sorted(expected):
        rows = [pooled[controller].get(key) for controller in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += len({str(row.get("initial_fingerprint")) for row in summaries}) != 1
        conflict_mismatches += len({int(row.get("initial_conflicts", -1)) for row in summaries}) != 1
        bad_clock += sum(row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries)
        capped_values += sum(row.get("capped_wall_time_to_feasible") is not None for row in summaries)
    extension_analysis = _scope_analysis(
        extension, groups, extension_seeds, dict(config["extension_gates"])
    )
    pooled_analysis = _scope_analysis(
        pooled, groups, pooled_seeds, dict(config["pooled_gates"])
    )
    total_counts = {
        name: int(quick_counts[name]) + int(extension_counts[name])
        for name in quick_counts
    }
    integrity = {
        "quick_report_and_manifests_match": quick_hash_match,
        "complete_extension_and_pooled_coverage": not errors,
        "zero_execution_errors": all(
            row.get("status") == "ok" for values in pooled.values() for row in values.values()
        ),
        "all_controllers_succeeded": all(
            pooled_analysis["controller_summaries"][controller]["success_count"] == len(expected)
            for controller in CONTROLLERS
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "no_capped_ttf_values": capped_values == 0,
        "zero_invalid_actions": all(
            pooled_analysis["controller_summaries"][controller]["invalid_action_count"] == 0
            for controller in CONTROLLERS
        ),
        "zero_semantic_mismatches": all(
            pooled_analysis["controller_summaries"][controller]["fingerprint_mismatch_count"] == 0
            for controller in CONTROLLERS
        ),
        "structpool_activated_and_selected": total_counts["gate_passed"] > 0
        and total_counts["selected"] > 0,
    }
    integrity_passed = not errors and all(integrity.values())
    performance_passed = (
        integrity_passed
        and extension_analysis["performance_passed"]
        and pooled_analysis["performance_passed"]
    )
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "four_seed_development_confirmation",
        "default_replacement_allowed": False,
        "producer_identity": producer,
        "formal_speed_claim": False,
        "primary_metric": "mean_run_to_completion_raw_wall_ttf",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "quick_scope": {
            "episode_count_per_controller": len(quick_seeds) * sum(len(row["tasks"]) for row in groups),
            "report_sha256": sha256_file(root / str(config["inputs"]["quick_report"]["path"])),
            "manifest_sha256": quick_hashes,
            "structpool_runtime_counts": quick_counts,
        },
        "extension_scope": {
            **extension_analysis,
            "manifest_sha256": extension_hashes,
            "structpool_runtime_counts": extension_counts,
        },
        "pooled_scope": {
            **pooled_analysis,
            "structpool_runtime_counts": total_counts,
        },
        "integrity_gates": integrity,
        "integrity_passed": integrity_passed,
        "performance_passed": performance_passed,
        "next_step": (
            "preregister_fresh_map_raw_ttf"
            if performance_passed
            else "stop_ttf_expansion_and_diagnose_seed_or_group_instability"
        ),
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
        },
    }
    _write_json(output / "structpool_ttf_confirmation_report.json", report)
    return report


__all__ = [
    "analyze_structpool_ttf_confirmation",
    "load_structpool_ttf_confirmation_config",
    "run_structpool_ttf_confirmation",
    "structpool_confirmation_schedule",
]
