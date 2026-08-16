from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

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
from experiments.stride_augcontrol_evaluation import _dataset_tasks
from experiments.stride_bounded_native_retry_continuation import _failed_job
from experiments.stride_failure_informed_rescue_continuation import _decision_rows
from experiments.stride_hybridstructpool_routed_confirmation import (
    _bounded_paired_comparison,
    _bounded_summary,
    _quantile,
)
from experiments.stride_structpool_ttf_quick import TTF_CLOCK_SCHEMA
from experiments.stride_structshell_rollback_aware_platform_replay import (
    _exact_rollback,
    _exact_repair_platforms,
    _first_exact_repair_platform,
)
from lns2_selector.runtime.hybridstructpool_routed import (
    rollback_aware_routed_hybridstructpool_augmentation,
    validate_rollback_aware_routed_hybridstructpool_augmentation,
)


CONFIG_SCHEMA = "lns2.stride.structshell_rollback_aware_ttf_config.v1"
EXPERIMENT_ID = "stride-structshell-rollback-aware-ttf-v1"
CONTROLLERS = (
    "official_adaptive",
    "v2_only",
    "structshell_rollback_aware_v2",
)
SCREEN_STATUS_SCHEMA = "lns2.stride.structshell_rollback_aware_ttf_screen_status.v1"
SCREEN_REPORT_SCHEMA = "lns2.stride.structshell_rollback_aware_ttf_screen_report.v1"
FINAL_STATUS_SCHEMA = "lns2.stride.structshell_rollback_aware_ttf_final_status.v1"
FINAL_REPORT_SCHEMA = "lns2.stride.structshell_rollback_aware_ttf_final_report.v1"
STATUS_FILENAME = "collection_status.json"
SCREEN_REPORT_FILENAME = "screen_report.json"
FINAL_REPORT_FILENAME = "final_report.json"

_EXPECTED_COMPARISON = {
    "primary_baseline": "v2_only",
    "external_baseline": "official_adaptive",
    "challenger": "structshell_rollback_aware_v2",
    "execution_order": "rotating_strict_three_controller_serial",
    "paired_solver_seed_required": True,
    "workers_for_timed_episodes": 1,
    "workers_for_qualification": 16,
}
_EXPECTED_RUNTIME = {
    "stopping_rule": "wall-clock",
    "repair_seed_policy": "episode_stream",
    "deterministic_pp_replay": False,
    "wall_time_budget_seconds": 180.0,
    "environment_time_limit_seconds": 180.0,
    "episode_process_timeout_seconds": 240.0,
    "outer_job_timeout_seconds": 300.0,
    "native_pp_order_only": True,
    "maximum_pp_calls_per_decision": 1,
    "runtime_retry_or_rescue": False,
}
_EXPECTED_STAGES = {
    "screen": {
        "task_indices": [0],
        "solver_seeds": [16],
        "paired_key_count": 10,
        "episode_count": 30,
    },
    "extension": {
        "mode": "full_schedule_minus_screen",
        "additional_paired_key_count": 50,
        "additional_episode_count": 150,
        "final_paired_key_count": 60,
        "final_episode_count": 180,
    },
}
_EXPECTED_GATES = {
    "minimum_paired_faster_fraction": 0.5,
    "maximum_map_restricted_ttf_regression": 0.05,
    "paired_bootstrap_replicates": 10000,
    "paired_bootstrap_relative_improvement_lower_bound": 0.0,
    "timing_reconciliation_diagnostic_tolerance": 0.02,
}


def _registered(root: Path, specification: Mapping[str, Any], label: str) -> Path:
    return registered_input(root, dict(specification), label=label)


def _producer(root: Path, *, native_required: bool = True) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_structshell_rollback_aware_ttf.py",
            "scripts/run_stride_structshell_rollback_aware_ttf.py",
            "experiments/closed_loop_confirmation.py",
            "lns2_selector/runtime/hybridstructpool_routed.py",
            "lns2_selector/runtime/rollback_aware_selection.py",
        ),
        native_required=native_required,
    )


def _successful_manifest(row: Mapping[str, Any]) -> bool:
    return str(row.get("status")) in {"ok", "resumed"} and isinstance(
        row.get("summary"), Mapping
    )


def _metric_manifest(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    if _successful_manifest(result):
        result["status"] = "ok"
    return result


def load_config(path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_fresh_seed_fixed_task_paired_bounded_ttf_confirmation"
        or config.get("experiment_id") != EXPERIMENT_ID
        or str(config.get("pre_registration_parent_commit"))
        != "1dd724eb74cb82fb4d5a2f44fb8d4e493434171a"
        or tuple(map(str, config.get("controllers") or ())) != CONTROLLERS
        or dict(config.get("comparison") or {}) != _EXPECTED_COMPARISON
        or dict(config.get("runtime") or {}) != _EXPECTED_RUNTIME
        or dict(config.get("stages") or {}) != _EXPECTED_STAGES
        or dict(config.get("performance_gates") or {}) != _EXPECTED_GATES
    ):
        raise ValueError("rollback-aware TTF identity changed")
    augmentation = validate_rollback_aware_routed_hybridstructpool_augmentation(
        dict(config.get("challenger_augmentation") or {})
    )
    if augmentation != rollback_aware_routed_hybridstructpool_augmentation():
        raise ValueError("rollback-aware TTF challenger changed")
    cohort = dict(config.get("cohort") or {})
    groups = list(cohort.get("groups") or ())
    if (
        cohort.get("role") != "fresh_seed_fixed_task_paired_bounded_ttf_confirmation"
        or tuple(map(int, cohort.get("solver_seeds") or ())) != (16, 17, 18)
        or int(cohort.get("paired_key_count", -1)) != 60
        or int(cohort.get("episode_count_per_controller", -1)) != 60
        or cohort.get("result_based_filtering") is not False
        or cohort.get("known_platform_keys_included") is not False
        or len(groups) != 10
        or len({str(group["id"]) for group in groups}) != 10
        or any(len(list(group.get("tasks") or ())) != 2 for group in groups)
        or not {"maze", "room", "warehouse", "game", "dao"}
        <= {str(group["family"]) for group in groups}
    ):
        raise ValueError("rollback-aware TTF cohort changed")
    for label, specification in dict(config.get("inputs") or {}).items():
        _registered(root, specification, f"rollback-aware TTF {label}")
    evidence = _read_json(
        root / str(config["inputs"]["mechanism_replay_report"]["path"])
    )
    challenger_evidence = dict(
        dict(evidence.get("arm_summaries") or {}).get(
            "structshell_rollback_aware_v2"
        )
        or {}
    )
    mechanism = dict(evidence.get("mechanism_gates") or {})
    if (
        evidence.get("integrity_passed") is not True
        or int(challenger_evidence.get("success_count", -1)) != 2
        or mechanism.get("both_cases_really_escape_first_repair_platform") is not True
        or mechanism.get("pure_structshell_streak_bounded_to_three") is not True
    ):
        raise ValueError("registered escape evidence is insufficient for TTF")
    for group in groups:
        tasks = _dataset_tasks(
            (root / str(group["dataset"])).resolve(), str(group["split"])
        )
        if set(map(str, group["tasks"])) - set(tasks):
            raise ValueError(f"TTF task absent for map {group['id']}")
    if len(full_keys(config)) != 60 or len(screen_keys(config)) != 10:
        raise ValueError("rollback-aware TTF key schedule changed")
    if screen_keys(config) & extension_keys(config):
        raise ValueError("screen and extension keys overlap")
    if screen_keys(config) | extension_keys(config) != full_keys(config):
        raise ValueError("screen and extension do not partition the full schedule")
    return path, root, config


Key = tuple[str, str, int]


def full_keys(config: Mapping[str, Any]) -> set[Key]:
    return {
        (str(group["id"]), str(task), int(seed))
        for group in config["cohort"]["groups"]
        for task in group["tasks"]
        for seed in config["cohort"]["solver_seeds"]
    }


def screen_keys(config: Mapping[str, Any]) -> set[Key]:
    return {
        (str(group["id"]), str(group["tasks"][0]), 16)
        for group in config["cohort"]["groups"]
    }


def extension_keys(config: Mapping[str, Any]) -> set[Key]:
    return full_keys(config) - screen_keys(config)


def schedule(config: Mapping[str, Any], stage: str = "full") -> list[dict[str, Any]]:
    selected = (
        screen_keys(config)
        if stage == "screen"
        else extension_keys(config)
        if stage == "extension"
        else full_keys(config)
        if stage == "full"
        else None
    )
    if selected is None:
        raise ValueError(f"unknown TTF stage: {stage}")
    rows: list[dict[str, Any]] = []
    selected_key_index = 0
    for group in config["cohort"]["groups"]:
        for task in group["tasks"]:
            for seed in config["cohort"]["solver_seeds"]:
                key = (str(group["id"]), str(task), int(seed))
                if key in selected:
                    offset = selected_key_index % len(CONTROLLERS)
                    for position in range(len(CONTROLLERS)):
                        name = CONTROLLERS[(offset + position) % len(CONTROLLERS)]
                        rows.append(
                            {
                                "stage": stage,
                                "group_id": key[0],
                                "family": str(group["family"]),
                                "task_id": key[1],
                                "solver_seed": key[2],
                                "controller": name,
                                "within_key_position": position,
                            }
                        )
                    selected_key_index += 1
    return rows


def _group(config: Mapping[str, Any], group_id: str) -> dict[str, Any]:
    return next(
        dict(group)
        for group in config["cohort"]["groups"]
        if str(group["id"]) == str(group_id)
    )


def _controller_kwargs(root: Path, config: Mapping[str, Any], name: str) -> dict[str, Any]:
    runtime = dict(config["runtime"])
    common = {
        "stopping_rule": "wall-clock",
        "repair_seed_policy": "episode_stream",
        "deterministic_pp_replay": False,
        "wall_time_budget_seconds": float(runtime["wall_time_budget_seconds"]),
        "episode_process_timeout_seconds": float(
            runtime["episode_process_timeout_seconds"]
        ),
        "environment_time_limit_seconds": float(
            runtime["environment_time_limit_seconds"]
        ),
    }
    if name == "official_adaptive":
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
    if name == "structshell_rollback_aware_v2":
        result["hybridstructpool_augmentation"] = dict(
            config["challenger_augmentation"]
        )
    elif name != "v2_only":
        raise ValueError(f"unknown rollback-aware TTF controller: {name}")
    return result


def _runtime_config_path(
    root: Path, output: Path, config: Mapping[str, Any], group: Mapping[str, Any]
) -> Path:
    payload = _read_json((root / str(group["runtime_config"])).resolve())
    payload["solver_seeds"] = [16, 17, 18]
    destination = output / "runtime_configs" / f"{group['id']}__seeds_16_18.json"
    _write_json(destination, payload)
    return destination


def _stage_output(output: Path, stage: str) -> Path:
    return output / stage


def _controller_dir(output: Path, stage: str, item: Mapping[str, Any]) -> Path:
    return _stage_output(output, stage) / "maps" / str(item["group_id"]) / str(
        item["controller"]
    )


def _manifest_path(output: Path, stage: str, item: Mapping[str, Any]) -> Path:
    filename = (
        "official_adaptive_manifest.jsonl"
        if str(item["controller"]) == "official_adaptive"
        else "realized_dynamic_manifest.jsonl"
    )
    return _controller_dir(output, stage, item) / filename


def _manifest(
    output: Path, stage: str, item: Mapping[str, Any]
) -> dict[str, Any] | None:
    path = _manifest_path(output, stage, item)
    matches = [
        dict(row)
        for row in (_read_jsonl(path) if path.is_file() else [])
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matches) > 1:
        raise ValueError("rollback-aware TTF manifest is ambiguous")
    return matches[0] if matches else None


def _qualification_summary(output: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    details = []
    for group in config["cohort"]["groups"]:
        expected = 6
        path = output / "qualification" / str(group["id"]) / "qualification_report.json"
        if not path.is_file():
            details.append(
                {
                    "group_id": str(group["id"]),
                    "expected_count": expected,
                    "valid_count": 0,
                    "passed": False,
                    "decision": "missing_qualification_report",
                }
            )
            continue
        report = _read_json(path)
        valid = int(report.get("valid_count", -1))
        details.append(
            {
                "group_id": str(group["id"]),
                "expected_count": expected,
                "valid_count": valid,
                "passed": bool(
                    report.get("passed") is True
                    and valid == expected
                    and int(report.get("incomplete_reset_count", -1)) == 0
                ),
                "decision": str(report.get("decision") or ""),
            }
        )
    return {
        "completed_map_count": sum(row["valid_count"] == 6 for row in details),
        "passed_map_count": sum(row["passed"] for row in details),
        "total_map_count": 10,
        "all_maps_passed": all(row["passed"] for row in details),
        "details": details,
    }


def _ensure_qualification(
    root: Path, output: Path, config: Mapping[str, Any], *, resume: bool
) -> dict[str, Any]:
    for group in config["cohort"]["groups"]:
        keys = {
            (str(task), int(seed))
            for task in group["tasks"]
            for seed in config["cohort"]["solver_seeds"]
        }
        destination = output / "qualification" / str(group["id"])
        run_closed_loop_collection(
            root / str(group["dataset"]),
            _runtime_config_path(root, output, config, group),
            destination,
            phase="qualify",
            workers=16,
            # Qualification is shared by both timed stages.  The extension in
            # a first uninterrupted `run` must reuse the already completed
            # screen qualification even though the outer call is not a
            # command-line --resume invocation.
            resume=_qualification_resume(destination),
            cohort_job_keys=keys,
            job_keys=keys,
            **_controller_kwargs(root, config, "v2_only"),
        )
    return _qualification_summary(output, config)


def _qualification_resume(destination: Path) -> bool:
    """Reuse the one frozen 60-key qualification across both timed stages."""

    return destination.joinpath("run_config.json").is_file()


def _episode_job(job: dict[str, Any]) -> dict[str, Any]:
    _path, root, config = load_config(job["config_path"])
    output = Path(job["output_root"]).resolve()
    stage = str(job["stage"])
    item = dict(job["item"])
    group = _group(config, str(item["group_id"]))
    keys = {
        (key[1], key[2])
        for key in (screen_keys(config) if stage == "screen" else extension_keys(config))
        if key[0] == str(group["id"])
    }
    collection = _controller_dir(output, stage, item)
    kwargs = _controller_kwargs(root, config, str(item["controller"]))
    qualification = output / "qualification" / str(group["id"])
    runtime = _runtime_config_path(root, output, config, group)
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase="qualify",
        workers=1,
        resume=collection.joinpath("run_config.json").is_file(),
        cohort_job_keys=keys,
        job_keys=keys,
        qualification_source=qualification,
        use_global_collection_lock=False,
        **kwargs,
    )
    run_closed_loop_collection(
        root / str(group["dataset"]),
        runtime,
        collection,
        phase=(
            "official_adaptive"
            if item["controller"] == "official_adaptive"
            else "realized_dynamic"
        ),
        workers=1,
        resume=True,
        cohort_job_keys=keys,
        job_keys={(str(item["task_id"]), int(item["solver_seed"]))},
        qualification_source=qualification,
        use_global_collection_lock=False,
        **kwargs,
    )
    row = _manifest(output, stage, item)
    if row is None:
        raise RuntimeError("rollback-aware TTF episode completed without manifest")
    status = str(row.get("status"))
    if status not in {"ok", "resumed", "error", "timeout"}:
        raise RuntimeError(f"unknown rollback-aware TTF manifest status: {status}")
    return {
        **item,
        "status": status if status in {"error", "timeout"} else "ok",
    }


def _status(
    output: Path,
    stage: str,
    items: list[dict[str, Any]],
    base: Mapping[str, Any],
    *,
    complete: bool = False,
) -> dict[str, Any]:
    rows = [_manifest(output, stage, item) for item in items]
    present = [row for row in rows if row is not None]
    return {
        **dict(base),
        "stage": stage,
        "completed_jobs": len(present),
        "completed_schedule_entries": len(present),
        "total_jobs": len(items),
        "completed_by_controller": {
            name: sum(
                _manifest(output, stage, item) is not None
                for item in items
                if item["controller"] == name
            )
            for name in CONTROLLERS
        },
        "error_jobs": sum(row.get("status") == "error" for row in present),
        "timeout_jobs": sum(row.get("status") == "timeout" for row in present),
        "active_jobs": 0,
        "complete": bool(complete and len(present) == len(items)),
    }


def _collect_stage(
    config_path: str | Path,
    output: str | Path,
    stage: str,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    items = schedule(config, stage)
    if dry_run:
        return {
            "schema": SCREEN_STATUS_SCHEMA if stage == "screen" else FINAL_STATUS_SCHEMA,
            "stage": stage,
            "qualification_key_count": 60,
            "paired_key_count": len(items) // 3,
            "schedule_entry_count": len(items),
            "schedule_sha256": _fingerprint(items),
            "timed_worker_count": 1,
            "qualification_worker_limit": 16,
        }
    producer = _producer(root)
    if stage == "extension":
        screen_report = load_completed_report(
            _stage_output(output, "screen"),
            status_filename=STATUS_FILENAME,
            report_filename=SCREEN_REPORT_FILENAME,
            status_schema=SCREEN_STATUS_SCHEMA,
            report_schema=SCREEN_REPORT_SCHEMA,
            config_path=path,
        )
        if (
            screen_report is None
            or screen_report.get("screen_passed") is not True
            or screen_report.get("producer_identity") != producer
        ):
            return {
                "schema": FINAL_STATUS_SCHEMA,
                "stage": stage,
                "complete": False,
                "blocked": True,
                "terminal_failure": "screen_not_passed",
            }
    stage_output = _stage_output(output, stage)
    report_filename = SCREEN_REPORT_FILENAME if stage == "screen" else FINAL_REPORT_FILENAME
    report_schema = SCREEN_REPORT_SCHEMA if stage == "screen" else FINAL_REPORT_SCHEMA
    status_schema = SCREEN_STATUS_SCHEMA if stage == "screen" else FINAL_STATUS_SCHEMA
    prepared = prepare_resumable_output(
        stage_output,
        status_filename=STATUS_FILENAME,
        status_schema=status_schema,
        config_path=path,
        schedule=items,
        producer=producer,
        resume=resume,
        report_filename=report_filename,
        report_schema=report_schema,
        label=f"rollback-aware bounded TTF {stage}",
    )
    if prepared.completed_report is not None:
        return prepared.completed_report
    if prepared.resumed and prepared.status.get("terminal_failure") is not None:
        return dict(prepared.status)
    existing_terminal_manifests = [
        row
        for item in items
        if (row := _manifest(output, stage, item)) is not None
        and str(row.get("status")) in {"error", "timeout"}
    ]
    if existing_terminal_manifests:
        status = _status(output, stage, items, prepared.base_status)
        status["terminal_failure"] = existing_terminal_manifests[0]
        _write_json(stage_output / STATUS_FILENAME, status)
        return status
    _write_jsonl(stage_output / "execution_schedule.jsonl", items)
    qualification = _ensure_qualification(root, output, config, resume=resume)
    if not qualification["all_maps_passed"]:
        status = _status(output, stage, items, prepared.base_status)
        status["qualification"] = qualification
        status["terminal_failure"] = "qualification_failed"
        _write_json(stage_output / STATUS_FILENAME, status)
        return status
    pending = [item for item in items if _manifest(output, stage, item) is None]
    jobs = [
        {
            "job_id": _fingerprint(item),
            "config_path": str(path),
            "output_root": str(output),
            "stage": stage,
            "item": item,
        }
        for item in pending
    ]
    if jobs:
        results = _run_jobs(
            _episode_job,
            jobs,
            1,
            phase=f"rollback-aware-ttf-{stage}",
            output_root=stage_output,
            run_fingerprint=str(prepared.base_status["run_fingerprint"]),
            timeout_seconds=300.0,
            failure_result=_failed_job,
            stop_on_failure=True,
        )
        failures = [row for row in results if row.get("status") in {"error", "timeout"}]
        if failures:
            status = _status(output, stage, items, prepared.base_status)
            status["qualification"] = qualification
            status["terminal_failure"] = failures[0]
            _write_json(stage_output / STATUS_FILENAME, status)
            return status
    report = (
        analyze_screen(path, output, producer=producer)
        if stage == "screen"
        else analyze_final(path, output, producer=producer)
    )
    status = _status(output, stage, items, prepared.base_status, complete=True)
    status["qualification"] = qualification
    status["report_sha256"] = sha256_file(stage_output / report_filename)
    _write_json(stage_output / STATUS_FILENAME, status)
    return report


def run_screen(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _collect_stage(config_path, output, "screen", resume=resume, dry_run=dry_run)


def run_extension(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _collect_stage(config_path, output, "extension", resume=resume, dry_run=dry_run)


def run(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return {
            "screen": run_screen(config_path, output, dry_run=True),
            "extension": run_extension(config_path, output, dry_run=True),
        }
    screen = run_screen(config_path, output, resume=resume)
    if screen.get("screen_passed") is not True:
        return screen
    return run_extension(config_path, output, resume=resume)


def _read_index(
    config: Mapping[str, Any], output: Path, stages: Iterable[str]
) -> tuple[dict[str, dict[Key, dict[str, Any]]], list[str], dict[str, dict[str, str]]]:
    indexed = {name: {} for name in CONTROLLERS}
    errors: list[str] = []
    hashes: dict[str, dict[str, str]] = defaultdict(dict)
    for stage in stages:
        for controller in CONTROLLERS:
            for group in config["cohort"]["groups"]:
                item = {"group_id": group["id"], "controller": controller}
                path = _manifest_path(output, stage, item)
                if not path.is_file():
                    errors.append(f"{stage}/{controller}/{group['id']}: missing manifest")
                    continue
                hashes[controller][f"{stage}/{group['id']}"] = sha256_file(path)
                for raw in _read_jsonl(path):
                    row = _metric_manifest(raw)
                    key = (str(group["id"]), str(row["task_id"]), int(row["solver_seed"]))
                    if key in indexed[controller]:
                        errors.append(f"{controller}: duplicate {key}")
                    indexed[controller][key] = row
    return indexed, errors, hashes


def _metric_tail(rows: Iterable[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [
        float(dict(row["summary"])[field])
        for row in rows
        if _successful_manifest(row) and dict(row["summary"]).get(field) is not None
    ]
    return {
        "median": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "maximum": max(values) if values else None,
    }


def _extended_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    normalized = [_metric_manifest(row) for row in rows]
    result = _bounded_summary(normalized)
    result["restricted_ttf"] = _metric_tail(
        normalized, "capped_wall_time_to_feasible"
    )
    result["repair_decisions"] = _metric_tail(normalized, "repair_iterations")
    return result


def _timing_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    fields: dict[str, list[float]] = defaultdict(list)
    reconciliation: list[float] = []
    for row in rows:
        if not _successful_manifest(row):
            continue
        summary = dict(row["summary"])
        totals = dict(summary.get("controller_totals") or {})
        values = {
            "reset_seconds": float(summary.get("reset_wall_seconds", 0.0)),
            "candidate_generation_seconds": float(
                totals.get("candidate_generation_seconds", 0.0)
            ),
            "feature_seconds": float(totals.get("feature_seconds", 0.0)),
            "scoring_seconds": float(totals.get("inference_seconds", 0.0)),
            "selection_seconds": float(
                totals.get("neighborhood_selection_seconds", 0.0)
            ),
            "guard_seconds": float(totals.get("exact_rollback_guard_seconds", 0.0)),
            "pp_seconds": float(totals.get("pp_replan_seconds", 0.0)),
            "trace_seconds": float(summary.get("trace_write_seconds", 0.0)),
            "unaccounted_seconds": float(
                summary.get("timing_unaccounted_seconds", 0.0)
            ),
        }
        for name, value in values.items():
            fields[name].append(value)
        capped = float(summary["capped_wall_time_to_feasible"])
        accounted = (
            float(summary.get("reset_wall_seconds", 0.0))
            + float(totals.get("iteration_wall_seconds", 0.0))
            + float(summary.get("trace_write_seconds", 0.0))
            + float(summary.get("timing_unaccounted_seconds", 0.0))
        )
        reconciliation.append(abs(accounted - capped) / max(capped, 1e-9))
    return {
        "components": {
            name: {
                "mean": statistics.fmean(values) if values else None,
                "p95": _quantile(values, 0.95),
            }
            for name, values in fields.items()
        },
        "maximum_reconciliation_relative_error": max(reconciliation, default=math.inf),
    }


def _cluster_bootstrap(
    baseline: Mapping[Key, Mapping[str, Any]],
    challenger: Mapping[Key, Mapping[str, Any]],
    keys: list[Key],
    replicates: int,
) -> dict[str, Any]:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for key in keys:
        left, right = baseline[key], challenger[key]
        if not (_successful_manifest(left) and _successful_manifest(right)):
            return {"valid": False, "group_count": 0, "pair_count": 0}
        groups[key[0]].append(
            (
                float(dict(left["summary"])["capped_wall_time_to_feasible"]),
                float(dict(right["summary"])["capped_wall_time_to_feasible"]),
            )
        )
    names = sorted(groups)
    if not names:
        return {"valid": False, "group_count": 0, "pair_count": 0}

    def estimate(sampled: list[str]) -> float:
        pairs = [pair for name in sampled for pair in groups[name]]
        base = statistics.fmean(pair[0] for pair in pairs)
        return (base - statistics.fmean(pair[1] for pair in pairs)) / base

    rng = random.Random(20260816)
    estimates = [
        estimate([names[rng.randrange(len(names))] for _ in names])
        for _ in range(replicates)
    ]
    return {
        "valid": True,
        "group_count": len(names),
        "pair_count": sum(len(group) for group in groups.values()),
        "relative_improvement": estimate(names),
        "ci95_lower": _quantile(estimates, 0.025),
        "ci95_upper": _quantile(estimates, 0.975),
        "replicates": replicates,
        "seed": 20260816,
        "resampling_unit": "map_group_with_all_task_seed_pairs",
    }


def _platform_summary(
    config: Mapping[str, Any], output: Path, stages: Iterable[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for controller in CONTROLLERS:
        episodes = triggers = terminal = later = occupancy = 0
        longest = 0
        single_pp_per_decision = True
        escape_latencies: list[float] = []
        for stage in stages:
            for item in schedule(config, stage):
                if item["controller"] != controller:
                    continue
                manifest = _manifest(output, stage, item)
                if manifest is None or not _successful_manifest(manifest):
                    continue
                episodes += 1
                rows = _decision_rows(_controller_dir(output, stage, item), manifest)
                single_pp_per_decision = bool(
                    single_pp_per_decision
                    and len(rows)
                    == int(dict(manifest["summary"]).get("repair_iterations", -1))
                )
                first = _first_exact_repair_platform(rows)
                platforms = _exact_repair_platforms(rows)
                triggers += int(bool(first.get("formed")))
                latency = first.get("escape_latency_decisions")
                if latency is not None:
                    escape_latencies.append(float(latency))
                episode_longest = episode_streak = 0
                previous_signature: str | None = None
                for row in rows:
                    signature = str(row["before_platform_signature"])
                    if _exact_rollback(row):
                        episode_streak = (
                            episode_streak + 1
                            if signature == previous_signature
                            else 1
                        )
                        previous_signature = signature
                        episode_longest = max(episode_longest, episode_streak)
                    else:
                        episode_streak = 0
                        previous_signature = None
                longest = max(longest, episode_longest)
                occupancy += sum(
                    int(platform["exact_rollback_count"]) for platform in platforms
                )
                later += max(0, len(platforms) - 1)
                terminal += int(
                    bool(platforms)
                    and int(platforms[-1]["end_decision"]) == len(rows) - 1
                )
        result[controller] = {
            "episode_count": episodes,
            "platform_trigger_count": triggers,
            "mean_first_platform_escape_latency_decisions": (
                statistics.fmean(escape_latencies) if escape_latencies else None
            ),
            "longest_identical_repair_fingerprint_rollback_streak": longest,
            "cumulative_platform_occupancy_decisions": occupancy,
            "later_platform_count": later,
            "terminal_platform_count": terminal,
            "single_pp_per_decision": single_pp_per_decision,
            "diagnostic_only": True,
        }
    return result


def _analysis(
    config_path: str | Path,
    output: str | Path,
    *,
    final: bool,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path, root, config = load_config(config_path)
    output = Path(output).resolve()
    stages = ("screen", "extension") if final else ("screen",)
    completed = load_completed_report(
        _stage_output(output, "extension" if final else "screen"),
        status_filename=STATUS_FILENAME,
        report_filename=FINAL_REPORT_FILENAME if final else SCREEN_REPORT_FILENAME,
        status_schema=FINAL_STATUS_SCHEMA if final else SCREEN_STATUS_SCHEMA,
        report_schema=FINAL_REPORT_SCHEMA if final else SCREEN_REPORT_SCHEMA,
        config_path=path,
    )
    if completed is not None:
        return completed
    expected = full_keys(config) if final else screen_keys(config)
    indexed, errors, hashes = _read_index(config, output, stages)
    for controller in CONTROLLERS:
        if set(indexed[controller]) != expected:
            errors.append(f"{controller}: incomplete paired coverage")
    keys = sorted(expected)
    fingerprint_mismatches = conflict_mismatches = bad_clock = bad_capped = 0
    for key in keys:
        rows = [indexed[name].get(key) for name in CONTROLLERS]
        if any(row is None or not _successful_manifest(row) for row in rows):
            continue
        summaries = [dict(row["summary"]) for row in rows]
        fingerprint_mismatches += len(
            {row.get("initial_fingerprint") for row in summaries}
        ) != 1
        conflict_mismatches += len({row.get("initial_conflicts") for row in summaries}) != 1
        bad_clock += sum(row.get("ttf_clock_schema") != TTF_CLOCK_SCHEMA for row in summaries)
        bad_capped += sum(row.get("capped_wall_time_to_feasible") is None for row in summaries)
    summaries = {
        name: _extended_summary(indexed[name].values()) for name in CONTROLLERS
    }
    timing = {name: _timing_summary(indexed[name].values()) for name in CONTROLLERS}
    tolerance = float(
        config["performance_gates"]["timing_reconciliation_diagnostic_tolerance"]
    )
    for value in timing.values():
        value["within_registered_diagnostic_tolerance"] = bool(
            float(value["maximum_reconciliation_relative_error"]) <= tolerance
        )
    platform_diagnostics = _platform_summary(config, output, stages)
    comparisons = {
        "challenger_vs_v2": _bounded_paired_comparison(
            indexed["v2_only"], indexed["structshell_rollback_aware_v2"], keys
        ),
        "challenger_vs_official": _bounded_paired_comparison(
            indexed["official_adaptive"],
            indexed["structshell_rollback_aware_v2"],
            keys,
        ),
        "v2_vs_official": _bounded_paired_comparison(
            indexed["official_adaptive"], indexed["v2_only"], keys
        ),
    }
    per_map = {
        group_id: {
            "challenger_vs_v2": _bounded_paired_comparison(
                indexed["v2_only"],
                indexed["structshell_rollback_aware_v2"],
                [key for key in keys if key[0] == group_id],
            ),
            "challenger_vs_official": _bounded_paired_comparison(
                indexed["official_adaptive"],
                indexed["structshell_rollback_aware_v2"],
                [key for key in keys if key[0] == group_id],
            ),
        }
        for group_id in [str(group["id"]) for group in config["cohort"]["groups"]]
    }
    integrity = {
        "complete_paired_coverage": not any("coverage" in error or "missing" in error for error in errors),
        "zero_execution_errors_or_process_timeouts": all(
            _successful_manifest(row)
            for controller in indexed.values()
            for row in controller.values()
        ),
        "paired_initial_fingerprints": fingerprint_mismatches == 0,
        "paired_initial_conflicts": conflict_mismatches == 0,
        "registered_ttf_clock": bad_clock == 0,
        "bounded_ttf_complete": bad_capped == 0,
        "valid_bounded_stop_reasons": all(
            str(dict(row.get("summary") or {}).get("stop_reason"))
            in {"success", "wall_timeout", "controller_stalled", "native_terminal"}
            for controller in indexed.values()
            for row in controller.values()
            if _successful_manifest(row)
        ),
        "zero_invalid_actions": all(
            int(summaries[name].get("invalid_action_count", -1)) == 0
            for name in CONTROLLERS
        ),
        "zero_semantic_mismatches": all(
            int(summaries[name].get("fingerprint_mismatch_count", -1)) == 0
            for name in CONTROLLERS
        ),
        "challenger_one_pp_per_decision_without_retry_or_rescue": all(
            int(dict(row["summary"]).get("model_decision_count", -1))
            == int(dict(row["summary"]).get("repair_iterations", -2))
            and all(
                dict(row["summary"]).get(name) is None
                for name in (
                    "bounded_native_retry",
                    "failure_informed_rescue",
                    "signature_scoped_rescue",
                )
            )
            for row in indexed["structshell_rollback_aware_v2"].values()
            if _successful_manifest(row)
        )
        and bool(
            platform_diagnostics["structshell_rollback_aware_v2"][
                "single_pp_per_decision"
            ]
        ),
    }
    challenger = summaries["structshell_rollback_aware_v2"]
    v2 = summaries["v2_only"]
    official = summaries["official_adaptive"]
    vs_v2 = comparisons["challenger_vs_v2"]
    no_screen_loss = all(
        not (
            dict(indexed["v2_only"][key]["summary"]).get("success") is True
            and dict(indexed["structshell_rollback_aware_v2"][key]["summary"]).get("success") is not True
            and float(dict(indexed["structshell_rollback_aware_v2"][key]["summary"])["capped_wall_time_to_feasible"])
            > float(dict(indexed["v2_only"][key]["summary"])["capped_wall_time_to_feasible"])
        )
        for key in keys
    )
    if final:
        replicates = int(config["performance_gates"]["paired_bootstrap_replicates"])
        boot_v2 = _cluster_bootstrap(
            indexed["v2_only"], indexed["structshell_rollback_aware_v2"], keys, replicates
        )
        boot_official = _cluster_bootstrap(
            indexed["official_adaptive"],
            indexed["structshell_rollback_aware_v2"],
            keys,
            replicates,
        )
        max_regression = float(
            config["performance_gates"]["maximum_map_restricted_ttf_regression"]
        )

        def contrast_gates(baseline_name: str, bootstrap: Mapping[str, Any]) -> dict[str, bool]:
            base = summaries[baseline_name]
            comparison = comparisons[
                "challenger_vs_v2"
                if baseline_name == "v2_only"
                else "challenger_vs_official"
            ]
            map_field = (
                "challenger_vs_v2"
                if baseline_name == "v2_only"
                else "challenger_vs_official"
            )
            return {
                "success_noninferior": int(challenger["success_count"])
                >= int(base["success_count"]),
                "mean_restricted_ttf_lower": float(
                    comparison["mean_restricted_ttf_delta_seconds"]
                )
                < 0.0,
                "paired_faster_fraction_at_least_half": float(
                    comparison["paired_faster_fraction"]
                )
                >= 0.5,
                "map_cluster_bootstrap_lower_above_zero": bootstrap.get("ci95_lower")
                is not None
                and float(bootstrap["ci95_lower"]) > 0.0,
                "normalized_wall_auc_noninferior": float(
                    challenger["mean_normalized_wall_clock_conflict_auc"]
                )
                <= float(base["mean_normalized_wall_clock_conflict_auc"]),
                "right_censoring_noninferior": int(challenger["right_censored_count"])
                <= int(base["right_censored_count"]),
                "every_map_within_restricted_ttf_regression_limit": all(
                    -float(value[map_field]["mean_restricted_ttf_relative_improvement"])
                    <= max_regression
                    for value in per_map.values()
                ),
                "common_success_raw_ttf_noninferior": comparison.get(
                    "challenger_common_success_mean_raw_ttf"
                )
                is not None
                and comparison.get("baseline_common_success_mean_raw_ttf") is not None
                and float(comparison["challenger_common_success_mean_raw_ttf"])
                <= float(comparison["baseline_common_success_mean_raw_ttf"]),
            }

        v2_gates = contrast_gates("v2_only", boot_v2)
        v2_gates.update(
            {
                "p95_repair_decisions_noninferior": float(
                    challenger["repair_decisions"]["p95"]
                )
                <= float(v2["repair_decisions"]["p95"]),
                "maximum_repair_decisions_noninferior": float(
                    challenger["repair_decisions"]["maximum"]
                )
                <= float(v2["repair_decisions"]["maximum"]),
            }
        )
        official_gates = contrast_gates("official_adaptive", boot_official)
        performance = {
            "v2_contrast": v2_gates,
            "official_contrast": official_gates,
        }
        v2_passed = all(v2_gates.values())
        official_passed = all(official_gates.values())
        passed = not errors and all(integrity.values()) and v2_passed and official_passed
        bootstrap = {
            "challenger_vs_v2": boot_v2,
            "challenger_vs_official": boot_official,
        }
    else:
        performance = {
            "success_noninferior_to_v2": int(challenger["success_count"])
            >= int(v2["success_count"]),
            "mean_restricted_ttf_lower_than_v2": float(
                vs_v2["mean_restricted_ttf_delta_seconds"]
            )
            < 0.0,
            "paired_faster_fraction_at_least_half": float(
                vs_v2["paired_faster_fraction"]
            )
            >= 0.5,
            "normalized_wall_auc_noninferior_to_v2": float(
                challenger["mean_normalized_wall_clock_conflict_auc"]
            )
            <= float(v2["mean_normalized_wall_clock_conflict_auc"]),
            "right_censoring_noninferior_to_v2": int(challenger["right_censored_count"])
            <= int(v2["right_censored_count"]),
            "p95_restricted_ttf_noninferior_to_v2": float(
                challenger["restricted_ttf"]["p95"]
            )
            <= float(v2["restricted_ttf"]["p95"]),
            "p95_repair_decisions_noninferior_to_v2": float(
                challenger["repair_decisions"]["p95"]
            )
            <= float(v2["repair_decisions"]["p95"]),
            "no_key_combines_success_loss_and_worse_restricted_ttf": no_screen_loss,
        }
        passed = not errors and all(integrity.values()) and all(performance.values())
        v2_passed = official_passed = False
        bootstrap = None
    if producer is None:
        producer = _producer(root)
    report = {
        "schema": FINAL_REPORT_SCHEMA if final else SCREEN_REPORT_SCHEMA,
        "scientific_status": (
            "fresh_seed_fixed_task_paired_bounded_ttf_confirmation"
            if final
            else "preregistered_cost_screen_no_speed_claim"
        ),
        "stage": "final" if final else "screen",
        "map_count": 10,
        "paired_key_count": len(expected),
        "episode_count": sum(len(value) for value in indexed.values()),
        "controller_summaries": summaries,
        "comparisons": comparisons,
        "per_map": per_map,
        "timing_accountability": timing,
        "platform_diagnostics": platform_diagnostics,
        "paired_map_cluster_bootstrap": bootstrap,
        "integrity_gates": integrity,
        "performance_gates": performance,
        "integrity_passed": not errors and all(integrity.values()),
        "screen_passed": bool(passed) if not final else None,
        "v2_contrast_passed": bool(v2_passed) if final else None,
        "official_contrast_passed": bool(official_passed) if final else None,
        "confirmation_passed": bool(passed) if final else False,
        "bounded_ttf_claim": bool(passed) if final else False,
        "uncapped_raw_ttf_claim": False,
        "runtime_replacement_allowed": bool(passed) if final else False,
        "next_step": (
            "collect_registered_extension"
            if not final and passed
            else "stop_screen_keep_v2_default"
            if not final
            else "freeze_rollback_aware_runtime"
            if passed
            else "keep_v2_default"
        ),
        "errors": errors,
        "producer_identity": producer,
        "inputs": {
            "config_sha256": sha256_file(path),
            "schedule_sha256": {
                stage: sha256_file(
                    _stage_output(output, stage) / "execution_schedule.jsonl"
                )
                for stage in stages
            },
            "controller_manifest_sha256": hashes,
        },
    }
    destination = _stage_output(output, "extension" if final else "screen") / (
        FINAL_REPORT_FILENAME if final else SCREEN_REPORT_FILENAME
    )
    _write_json(destination, report)
    return report


def analyze_screen(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _analysis(config_path, output, final=False, producer=producer)


def analyze_final(
    config_path: str | Path,
    output: str | Path,
    *,
    producer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _analysis(config_path, output, final=True, producer=producer)


__all__ = [
    "CONTROLLERS",
    "analyze_final",
    "analyze_screen",
    "extension_keys",
    "full_keys",
    "load_config",
    "run",
    "run_extension",
    "run_screen",
    "schedule",
    "screen_keys",
]
