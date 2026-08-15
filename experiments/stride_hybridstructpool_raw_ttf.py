from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from experiments._common import closed_loop_producer_identity, registered_input, sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.run_output_guard import load_completed_report, prepare_resumable_output
from experiments.stride_augcontrol_evaluation import _dataset_tasks, _metric
from experiments.stride_bounded_native_retry_continuation import _failed_job
from experiments.stride_maprank_raw_ttf import _controller_summary, _paired_comparison
from experiments.stride_robuststep_preflight import _mean
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA
from lns2_selector.runtime.hybridstructpool import (
    hybridstructpool_runtime_augmentation,
    validate_hybridstructpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.hybridstructpool_raw_ttf_quick_config.v1"
STATUS_SCHEMA = "lns2.stride.hybridstructpool_raw_ttf_quick_status.v1"
REPORT_SCHEMA = "lns2.stride.hybridstructpool_raw_ttf_quick_report.v1"
EXPERIMENT_ID = "stride-hybridstructpool-raw-ttf-quick-v1"
PRE_REGISTRATION_COMMITS = {
    "60258ba",
    "52096acb6120cc02c59a352bb7f72fdb9532fb6b",
}
CONTROLLERS = ("official_adaptive", "v2_full", "hybridstructpool_full")
STATUS_FILENAME = "collection_status.json"
REPORT_FILENAME = "raw_ttf_report.json"


def _registered(root: Path, specification: Mapping[str, Any]) -> Path:
    return registered_input(root, dict(specification), label="Hybrid raw TTF")


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_development_quick_before_raw_ttf"
        or config.get("experiment_id") != EXPERIMENT_ID
        or str(config.get("pre_registration_parent_commit"))
        not in PRE_REGISTRATION_COMMITS
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
    ):
        raise ValueError("Hybrid raw-TTF registration changed")
    comparison = dict(config.get("comparison") or {})
    if comparison != {
        "primary_baseline": "official_adaptive",
        "quality_anchor": "v2_full",
        "challenger": "hybridstructpool_full",
        "execution_order": "rotating_strict_three_controller_serial",
        "paired_solver_seed_required": True,
        "workers_for_timed_episodes": 1,
        "workers_for_qualification": 16,
    }:
        raise ValueError("Hybrid raw-TTF comparison changed")
    runtime = dict(config.get("runtime") or {})
    if runtime != {
        "config": "configs/stride_stage4r_high_load_runtime.json",
        "stopping_rule": "run-to-completion",
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "episode_process_timeout_seconds": 900.0,
    }:
        raise ValueError("Hybrid raw-TTF runtime changed")
    if validate_hybridstructpool_augmentation(
        dict(config["hybridstructpool_augmentation"])
    ) != hybridstructpool_runtime_augmentation():
        raise ValueError("Hybrid raw-TTF pool identity changed")
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role") != "development_quick_not_formal_or_result_blind"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (1, 2)
        or int(cohort.get("paired_key_count", -1)) != 8
        or int(cohort.get("episode_count_per_controller", -1)) != 8
        or [str(row.get("id")) for row in groups] != ["maze300", "room500"]
        or any(len(list(row.get("tasks") or ())) != 2 for row in groups)
    ):
        raise ValueError("Hybrid raw-TTF cohort changed")
    for specification in dict(config.get("inputs") or {}).values():
        _registered(root, specification)
    report = _read_json(root / str(config["inputs"]["hybrid_runtime_report"]["path"]))
    if report.get("all_gates_passed") is not True:
        raise ValueError("Hybrid engineering gate did not authorize raw-TTF Quick")
    for group in groups:
        tasks = _dataset_tasks((root / str(group["dataset"])).resolve(), str(group["split"]))
        if set(map(str, group["tasks"])) - set(tasks):
            raise ValueError("Hybrid raw-TTF task is absent from its dataset")
    return path, root, config


def schedule(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    keys = [
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    ]
    rows: list[dict[str, Any]] = []
    for index, (group_id, task_id, seed) in enumerate(keys):
        offset = index % len(CONTROLLERS)
        order = CONTROLLERS[offset:] + CONTROLLERS[:offset]
        rows.extend(
            {
                "group_id": group_id,
                "task_id": task_id,
                "solver_seed": seed,
                "controller": controller,
                "within_key_position": position,
            }
            for position, controller in enumerate(order)
        )
    return rows


def _controller_kwargs(root: Path, config: Mapping[str, Any], controller: str) -> dict[str, Any]:
    common = {
        "stopping_rule": "run-to-completion",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
    }
    if controller == "official_adaptive":
        return {
            **common,
            "controller": "official_adaptive",
            "feature_backend": "auto",
            "controller_runtime": "reference",
            "verification_profile": "audit",
        }
    result = {
        **common,
        "controller": "v2-full",
        "controller_bundle": str((root / str(config["controller_bundle"])).resolve()),
        "feature_backend": "native",
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
    }
    if controller == "hybridstructpool_full":
        result["hybridstructpool_augmentation"] = dict(
            config["hybridstructpool_augmentation"]
        )
    elif controller != "v2_full":
        raise ValueError(f"unknown Hybrid raw-TTF controller: {controller}")
    return result


def _controller_root(output: Path, item: Mapping[str, Any]) -> Path:
    return output / "groups" / str(item["group_id"]) / "controllers" / str(item["controller"])


def _manifest_path(output: Path, item: Mapping[str, Any]) -> Path:
    name = "official_adaptive_manifest.jsonl" if item["controller"] == "official_adaptive" else "realized_dynamic_manifest.jsonl"
    return _controller_root(output, item) / name


def _manifest(output: Path, item: Mapping[str, Any]) -> dict[str, Any] | None:
    path = _manifest_path(output, item)
    matches = [
        dict(row)
        for row in (_read_jsonl(path) if path.is_file() else [])
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("Hybrid raw-TTF manifest is ambiguous")
    return matches[0] if matches else None


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    _path, root, config = load_config(job["config_path"])
    item = dict(job["item"])
    group = next(
        dict(row) for row in config["cohort"]["groups"] if str(row["id"]) == str(item["group_id"])
    )
    dataset = (root / str(group["dataset"])).resolve()
    keys = {
        (str(task), int(seed))
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }
    key = (str(item["task_id"]), int(item["solver_seed"]))
    collection = _controller_root(Path(job["output_root"]).resolve(), item)
    kwargs = _controller_kwargs(root, config, str(item["controller"]))
    run_closed_loop_collection(
        dataset,
        root / str(config["runtime"]["config"]),
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_source=Path(job["qualification_source"]).resolve(),
        use_global_collection_lock=False,
        **kwargs,
    )
    run_closed_loop_collection(
        dataset,
        root / str(config["runtime"]["config"]),
        collection,
        phase=("official_adaptive" if item["controller"] == "official_adaptive" else "realized_dynamic"),
        workers=1,
        resume=True,
        cohort_job_keys=keys,
        job_keys={key},
        qualification_source=Path(job["qualification_source"]).resolve(),
        use_global_collection_lock=False,
        **kwargs,
    )
    manifest = _manifest(Path(job["output_root"]).resolve(), item)
    if manifest is None:
        raise RuntimeError("Hybrid raw-TTF episode completed without a manifest")
    status = str(manifest.get("status"))
    return {
        **item,
        "status": status if status in {"error", "timeout"} else "ok",
        "manifest_status": status,
        "error": manifest.get("error"),
    }


def _status(
    output: Path,
    items: list[dict[str, Any]],
    base_status: Mapping[str, Any],
    *,
    complete: bool = False,
) -> dict[str, Any]:
    present = [_manifest(output, item) for item in items]
    completed = [row for row in present if row is not None]
    by_controller = {
        controller: sum(
            _manifest(output, item) is not None
            for item in items
            if str(item["controller"]) == controller
        )
        for controller in CONTROLLERS
    }
    return {
        **dict(base_status),
        "completed_schedule_entries": len(completed),
        "completed_jobs": len(completed),
        "total_jobs": len(items),
        "completed_by_controller": by_controller,
        "error_jobs": sum(str(row.get("status")) == "error" for row in completed),
        "timeout_jobs": sum(str(row.get("status")) == "timeout" for row in completed),
        "complete": bool(complete and len(completed) == len(items)),
    }


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    items = schedule(config)
    if dry_run:
        return {
            "schema": STATUS_SCHEMA,
            "paired_key_count": len(items) // len(CONTROLLERS),
            "schedule_entry_count": len(items),
            "schedule_sha256": _fingerprint(items),
        }
    output = Path(output).resolve()
    prepared = prepare_resumable_output(
        output,
        status_filename=STATUS_FILENAME,
        status_schema=STATUS_SCHEMA,
        config_path=path,
        schedule=items,
        producer=closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_hybridstructpool_raw_ttf.py",
                "scripts/run_stride_hybridstructpool_raw_ttf.py",
                "experiments/closed_loop_confirmation.py",
                "lns2_selector/runtime/causalclosurepool.py",
                "lns2_selector/runtime/hybridstructpool.py",
                "lns2_selector/runtime/topology_candidates.py",
            ),
        ),
        resume=resume,
        report_filename=REPORT_FILENAME,
        report_schema=REPORT_SCHEMA,
        label="HybridStructPool raw TTF Quick",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    _write_jsonl(output / "execution_schedule.jsonl", items)
    by_group = {str(row["id"]): dict(row) for row in config["cohort"]["groups"]}
    for group_id, group in by_group.items():
        keys = {
            (str(task), int(seed))
            for task in group["tasks"]
            for seed in config["cohort"]["solver_seeds"]
        }
        qualification = output / "groups" / group_id / "qualification"
        run_closed_loop_collection(
            root / str(group["dataset"]),
            root / str(config["runtime"]["config"]),
            qualification,
            phase="qualify",
            workers=int(config["comparison"]["workers_for_qualification"]),
            resume=prepared.resumed and qualification.joinpath("run_config.json").is_file(),
            cohort_job_keys=keys,
            job_keys=keys,
            controller="v2-full",
            controller_bundle=str((root / str(config["controller_bundle"])).resolve()),
            feature_backend="native",
            controller_runtime="optimized",
            verification_profile="deployment",
            stopping_rule="run-to-completion",
            repair_seed_policy="episode_stream",
            deterministic_pp_replay=False,
        )
    pending = [item for item in items if _manifest(output, item) is None]
    jobs = [
        {
            "job_id": _fingerprint(item),
            "config_path": str(path),
            "output_root": str(output),
            "qualification_source": str(
                output / "groups" / str(item["group_id"]) / "qualification"
            ),
            "item": item,
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            1,
            phase="hybridstructpool-raw-ttf-quick",
            output_root=output,
            run_fingerprint=str(prepared.base_status["run_fingerprint"]),
            timeout_seconds=float(config["runtime"]["episode_process_timeout_seconds"]),
            failure_result=_failed_job,
            stop_on_failure=True,
        )
        terminal = [row for row in results if row.get("status") in {"error", "timeout"}]
        if terminal:
            status = _status(output, items, prepared.base_status)
            status["terminal_failure"] = terminal[0]
            _write_json(output / STATUS_FILENAME, status)
            return status
    status = _status(output, items, prepared.base_status)
    _write_json(output / STATUS_FILENAME, status)
    report = analyze(path, output, producer=prepared.base_status["producer_identity"])
    final_status = _status(output, items, prepared.base_status, complete=True)
    final_status["report_sha256"] = sha256_file(output / REPORT_FILENAME)
    _write_json(output / STATUS_FILENAME, final_status)
    return report


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = _controller_summary(rows)
    totals = [
        dict(dict(row.get("summary") or {}).get("controller_totals") or {})
        for row in rows
        if row.get("status") == "ok"
    ]
    for name in (
        "candidate_generation_seconds",
        "feature_seconds",
        "inference_seconds",
        "controller_seconds_before_repair",
        "pp_replan_seconds",
    ):
        result[f"mean_{name}"] = _mean([_metric(row, name) for row in totals])
    return result


def analyze(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
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
    if producer is None:
        producer = closed_loop_producer_identity(
            project_root=root,
            source_files=(
                "experiments/stride_hybridstructpool_raw_ttf.py",
                "scripts/run_stride_hybridstructpool_raw_ttf.py",
            ),
            native_required=False,
        )
    expected = {
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }
    indexed: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    errors: list[str] = []
    hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for controller in CONTROLLERS:
        rows_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for group in config["cohort"]["groups"]:
            item = {"group_id": group["id"], "controller": controller}
            manifest = _manifest_path(output, item)
            if not manifest.is_file():
                errors.append(f"{controller}/{group['id']}: missing manifest")
                continue
            hashes[controller][str(group["id"])] = sha256_file(manifest)
            for row in _read_jsonl(manifest):
                key = (str(group["id"]), str(row["task_id"]), int(row["solver_seed"]))
                if key in rows_by_key:
                    errors.append(f"{controller}: duplicate {key}")
                rows_by_key[key] = dict(row)
        if set(rows_by_key) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
        indexed[controller] = rows_by_key
    fingerprint_mismatches = conflict_mismatches = bad_clock = capped = 0
    for key in sorted(expected):
        rows = [indexed[name].get(key) for name in CONTROLLERS]
        if any(row is None or row.get("status") != "ok" for row in rows):
            continue
        summaries = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += len({str(row.get("initial_fingerprint")) for row in summaries}) != 1
        conflict_mismatches += len({int(row.get("initial_conflicts", -1)) for row in summaries}) != 1
        bad_clock += sum(row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries)
        capped += sum(row.get("capped_wall_time_to_feasible") is not None for row in summaries)
    summaries = {name: _summary(list(indexed[name].values())) for name in CONTROLLERS}
    keys = sorted(expected)
    comparisons = {
        "hybrid_vs_official": _paired_comparison(indexed["official_adaptive"], indexed["hybridstructpool_full"], keys),
        "hybrid_vs_v2": _paired_comparison(indexed["v2_full"], indexed["hybridstructpool_full"], keys),
        "v2_vs_official": _paired_comparison(indexed["official_adaptive"], indexed["v2_full"], keys),
    }
    per_group = {
        str(group["id"]): {
            "hybrid_vs_official": _paired_comparison(
                indexed["official_adaptive"], indexed["hybridstructpool_full"],
                [key for key in keys if key[0] == str(group["id"])],
            ),
            "hybrid_vs_v2": _paired_comparison(
                indexed["v2_full"], indexed["hybridstructpool_full"],
                [key for key in keys if key[0] == str(group["id"])],
            ),
        }
        for group in config["cohort"]["groups"]
    }
    integrity = {
        "complete_paired_coverage": not any("coverage" in error or "missing" in error for error in errors),
        "zero_execution_errors": all(
            row.get("status") == "ok" for values in indexed.values() for row in values.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "raw_ttf_clock_registered": bad_clock == 0,
        "no_capped_ttf_values": capped == 0,
        "zero_invalid_actions": all(summary["invalid_action_count"] == 0 for summary in summaries.values()),
        "zero_semantic_mismatches": all(summary["fingerprint_mismatch_count"] == 0 for summary in summaries.values()),
    }
    gates = dict(config["performance_gates"])
    ho = comparisons["hybrid_vs_official"]
    hv = comparisons["hybrid_vs_v2"]
    maximum_group_regression = max(
        (
            -float(comparison.get("mean_raw_ttf_relative_improvement", 0.0))
            for values in per_group.values()
            for comparison in values.values()
        ),
        default=0.0,
    )
    performance = {
        "success_noninferior_to_official": summaries["hybridstructpool_full"]["success_count"] >= summaries["official_adaptive"]["success_count"],
        "success_noninferior_to_v2": summaries["hybridstructpool_full"]["success_count"] >= summaries["v2_full"]["success_count"],
        "mean_raw_ttf_better_than_official": bool(ho.get("valid")) and float(ho["mean_raw_ttf_relative_improvement"]) >= float(gates["minimum_mean_raw_ttf_improvement_vs_official"]),
        "mean_raw_ttf_better_than_v2": bool(hv.get("valid")) and float(hv["mean_raw_ttf_relative_improvement"]) >= float(gates["minimum_mean_raw_ttf_improvement_vs_v2"]),
        "paired_faster_vs_official": bool(ho.get("valid")) and float(ho["paired_faster_fraction"]) >= float(gates["minimum_paired_faster_fraction_vs_official"]),
        "paired_faster_vs_v2": bool(hv.get("valid")) and float(hv["paired_faster_fraction"]) >= float(gates["minimum_paired_faster_fraction_vs_v2"]),
        "maximum_group_regression": maximum_group_regression <= float(gates["maximum_group_raw_ttf_regression"]),
        "repair_iterations_noninferior_to_v2": bool(hv.get("valid")) and float(hv["mean_repair_iterations_delta"]) <= 0.0,
    }
    integrity_passed = not errors and all(integrity.values())
    performance_passed = integrity_passed and all(performance.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "development_quick",
        "primary_metric": "mean_run_to_completion_reset_inclusive_raw_wall_ttf",
        "ttf_clock_schema": TTF_CLOCK_SCHEMA,
        "episode_count_per_controller": len(expected),
        "controller_summaries": summaries,
        "comparisons": comparisons,
        "per_group": per_group,
        "integrity_gates": integrity,
        "performance_gates": performance,
        "integrity_passed": integrity_passed,
        "performance_passed": performance_passed,
        "formal_speed_claim": False,
        "default_replacement_allowed": False,
        "next_step": (
            "preregister_larger_result_blind_paired_raw_ttf"
            if performance_passed
            else "stop_promotion_and_report_quality_overhead_tradeoff"
        ),
        "errors": errors,
        "producer_identity": producer,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": sha256_file(output / "execution_schedule.jsonl"),
            "controller_manifest_sha256": dict(hashes),
        },
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


__all__ = ["analyze", "load_config", "run", "schedule"]
