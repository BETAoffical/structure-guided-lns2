from __future__ import annotations

import argparse
import collections as collections_module
import concurrent.futures
import hashlib
import json
import logging
import math
import shutil
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments._common import sha256_file  # noqa: E402
from experiments.closed_loop_confirmation import (  # noqa: E402
    REPAIR_TIMING_SCHEMA,
    run_closed_loop_collection,
)
from experiments.closed_loop_trace_storage import TRACE_FORMAT_DELTA_GZIP_V2  # noqa: E402
from experiments.closed_loop_trace_storage import read_trace_events  # noqa: E402
from experiments.compact_controller_model import load_controller_bundle  # noqa: E402
from experiments.lns2_bottleneck import (  # noqa: E402
    generate_bottleneck_artifacts,
)
from experiments.parallel_runtime import (  # noqa: E402
    initialize_isolated_worker,
    isolated_lane_cpu_sets,
    parallel_runtime_metadata,
)
from experiments.repair_collection import (  # noqa: E402
    _load_dataset_rows,
    _read_json,
    _read_jsonl,
    _utc_now,
    _write_json,
    _write_jsonl,
)
from experiments.run_output_guard import prepare_run_output  # noqa: E402
from experiments.v3_controller import load_v3_controller_bundle  # noqa: E402


QUICK_TASKS = (
    "random-32-32-10__random_05__agents_0200",
    "maze-32-32-4__random_05__agents_0100",
    "room-64-64-16__random_05__agents_0400",
    "warehouse-10-20-10-2-2__random_05__agents_0500",
    "den312d__random_05__agents_0200",
    "maze-128-128-1__random_04__agents_0600",
    "room-64-64-16__random_04__agents_0600",
    "random-64-64-10__random_04__agents_0600",
)
REGISTERED_SOLVER_SEEDS = (1, 2, 3)
FORMAL_TASK_COUNT = 48
DUAL_COLLECTIONS = (
    ("official_adaptive", "v1-full", "official_adaptive"),
    ("v2-full", "v2-full", "realized_dynamic"),
)
STALL_SAFE_COLLECTION = ("v2-stall-safe", "v2-stall-safe", "realized_dynamic")
REPAIR_AWARE_COLLECTION = (
    "v2-repair-aware",
    "v2-repair-aware",
    "realized_dynamic",
)
V3_COLLECTION = ("v3-full", "v3-full", "realized_dynamic")
V3_H3_COLLECTION = ("v3-h3", "v3-h3", "realized_dynamic")
CONTROLLER_COLLECTIONS = {
    item[0]: item
    for item in (
        *DUAL_COLLECTIONS,
        STALL_SAFE_COLLECTION,
        REPAIR_AWARE_COLLECTION,
        V3_COLLECTION,
        V3_H3_COLLECTION,
    )
}
DUAL_DEFAULT_OUTPUTS = {
    "quick": "build/initlns-lns2-bottleneck-quick-v1",
    "formal": "build/initlns-lns2-bottleneck-formal-v1",
}


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("lns2.tradeoff")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def _status(path: Path, started_at: str, **values: Any) -> None:
    _write_json(
        path,
        {
            "schema": "lns2.lns2_tradeoff_status.v1",
            "started_at": started_at,
            "updated_at": _utc_now(),
            **values,
        },
    )


def _cohort_job_keys(
    dataset: Path,
    collection_config: Path,
    task_ids: list[str] | None,
    solver_seeds: tuple[int, ...] = REGISTERED_SOLVER_SEEDS,
) -> list[tuple[str, int]]:
    config = _read_json(collection_config)
    rows = _load_dataset_rows(dataset, [str(config["split"])])
    allowed = set(map(str, task_ids)) if task_ids is not None else None
    selected = [
        row for row in rows if allowed is None or str(row["task_id"]) in allowed
    ]
    if allowed is not None and {str(row["task_id"]) for row in selected} != allowed:
        missing = sorted(allowed - {str(row["task_id"]) for row in selected})
        raise ValueError(f"tradeoff task IDs are missing from the dataset: {missing}")
    keys = [
        (str(row["task_id"]), int(seed))
        for row in selected
        for seed in solver_seeds
    ]
    return sorted(
        keys,
        key=lambda value: hashlib.sha256(
            f"{value[0]}|{value[1]}".encode("utf-8")
        ).hexdigest(),
    )


def _run_interleaved_collections(
    *,
    roots: dict[str, Path],
    dataset: Path,
    collection_config: Path,
    controller_bundle: Path,
    stall_guard_config: Path | None,
    repair_aware_config: Path | None,
    repair_aware_bundle: Path | None,
    v3_bundle: Path | None,
    task_ids: list[str] | None,
    feature_backend: str,
    resume: bool,
    logger: logging.Logger,
    runner_progress: Path,
    schedule_path: Path,
    job_keys: set[tuple[str, int]] | None = None,
    wall_time_budget_seconds: float | None = None,
    episode_process_timeout_seconds: float | None = None,
    stopping_rule: str = "historical",
    collections: tuple[tuple[str, str, str], ...] = DUAL_COLLECTIONS,
    controller_runtime: str = "reference",
    verification_profile: str = "audit",
    paired_execution: str = "strict",
    parallel_lanes: int = 1,
) -> None:
    qualification_root: Path | None = None
    for name, controller, _policy in collections:
        root = roots[name]
        root.mkdir(parents=True, exist_ok=True)
        logger.info("%s: qualification", name)
        run_closed_loop_collection(
            dataset,
            collection_config,
            root,
            phase="qualify",
            workers=1,
            resume=resume or (root / "run_config.json").is_file(),
            task_ids=task_ids,
            trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
            controller=controller,
            feature_backend=feature_backend,
            controller_runtime=controller_runtime,
            verification_profile=verification_profile,
            controller_bundle=controller_bundle,
            stall_guard_config=(
                stall_guard_config if controller == "v2-stall-safe" else None
            ),
            repair_aware_config=(
                repair_aware_config if controller == "v2-repair-aware" else None
            ),
            repair_aware_bundle=(
                repair_aware_bundle if controller == "v2-repair-aware" else None
            ),
            v3_bundle=v3_bundle if controller in {"v3-full", "v3-h3"} else None,
            job_keys=job_keys,
            cohort_job_keys=job_keys,
            wall_time_budget_seconds=wall_time_budget_seconds,
            episode_process_timeout_seconds=episode_process_timeout_seconds,
            stopping_rule=stopping_rule,
            qualification_source=qualification_root,
        )
        if qualification_root is None:
            qualification_root = root

    keys = _cohort_job_keys(dataset, collection_config, task_ids)
    if job_keys is not None:
        normalized = {(str(task_id), int(seed)) for task_id, seed in job_keys}
        keys = [key for key in keys if key in normalized]
        if set(keys) != normalized:
            raise ValueError("paired collection job filter contains unknown task/seed pairs")
    schedule = []
    for task_id, seed in keys:
        digest = hashlib.sha256(f"{task_id}|{seed}".encode("utf-8")).hexdigest()
        rotation = int(digest[:8], 16) % len(collections)
        ordered = collections[rotation:] + collections[:rotation]
        schedule.append(
            {
                "task_id": task_id,
                "solver_seed": seed,
                "controller_order": [name for name, _controller, _policy in ordered],
            }
        )
    _write_json(
        schedule_path,
        {
            "schema": "lns2.controller_execution_schedule.v1",
            "method": "sha256-rotated-per-task-seed",
            "workers": (
                parallel_lanes if paired_execution == "isolated-parallel" else 1
            ),
            "paired_execution": paired_execution,
            "entries": schedule,
        },
    )
    if paired_execution == "isolated-parallel" and parallel_lanes > 1:
        _run_isolated_parallel_collections(
            roots=roots,
            dataset=dataset,
            collection_config=collection_config,
            controller_bundle=controller_bundle,
            stall_guard_config=stall_guard_config,
            repair_aware_config=repair_aware_config,
            repair_aware_bundle=repair_aware_bundle,
            v3_bundle=v3_bundle,
            task_ids=task_ids,
            feature_backend=feature_backend,
            runner_progress=runner_progress,
            schedule_path=schedule_path,
            schedule=schedule,
            job_keys=job_keys,
            wall_time_budget_seconds=wall_time_budget_seconds,
            episode_process_timeout_seconds=episode_process_timeout_seconds,
            stopping_rule=stopping_rule,
            collections=collections,
            controller_runtime=controller_runtime,
            verification_profile=verification_profile,
            parallel_lanes=parallel_lanes,
        )
        return
    collection_by_name = {item[0]: item for item in collections}
    completed = 0
    error_jobs = 0
    total = len(keys) * len(collections)
    for entry in schedule:
        task_id = str(entry["task_id"])
        seed = int(entry["solver_seed"])
        ordered = tuple(
            collection_by_name[str(name)] for name in entry["controller_order"]
        )
        for name, controller, policy in ordered:
            logger.info(
                "paired episode %s/%s: task=%s seed=%s controller=%s",
                completed + 1,
                total,
                task_id,
                seed,
                name,
            )
            run_closed_loop_collection(
                dataset,
                collection_config,
                roots[name],
                phase=policy,
                workers=1,
                resume=True,
                task_ids=task_ids,
                trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
                controller=controller,
                feature_backend=feature_backend,
                controller_runtime=controller_runtime,
                verification_profile=verification_profile,
                controller_bundle=controller_bundle,
                stall_guard_config=(
                    stall_guard_config if controller == "v2-stall-safe" else None
                ),
                repair_aware_config=(
                    repair_aware_config
                    if controller == "v2-repair-aware"
                    else None
                ),
                repair_aware_bundle=(
                    repair_aware_bundle
                    if controller == "v2-repair-aware"
                    else None
                ),
                v3_bundle=v3_bundle if controller in {"v3-full", "v3-h3"} else None,
                job_keys={(task_id, seed)},
                cohort_job_keys=job_keys,
                wall_time_budget_seconds=wall_time_budget_seconds,
                episode_process_timeout_seconds=episode_process_timeout_seconds,
                stopping_rule=stopping_rule,
            )
            completed += 1
            manifest_rows = _read_jsonl(
                roots[name] / f"{policy}_manifest.jsonl"
            )
            matching_rows = [
                row
                for row in manifest_rows
                if str(row.get("task_id")) == task_id
                and int(row.get("solver_seed", -1)) == seed
            ]
            if len(matching_rows) != 1:
                raise RuntimeError(
                    f"{name} did not write exactly one manifest row for "
                    f"{task_id}/seed={seed}"
                )
            episode_error = str(matching_rows[0].get("status")) not in {
                "ok",
                "resumed",
            }
            error_jobs += int(episode_error)
            _write_json(
                runner_progress,
                {
                    "schema": "lns2.lns2_tradeoff_progress.v2",
                    "phase": "paired-collections",
                    "status": "running",
                    "completed_jobs": completed,
                    "total_jobs": total,
                    "error_jobs": error_jobs,
                    "active_jobs": [],
                    "task_id": task_id,
                    "solver_seed": seed,
                    "controller": name,
                },
            )
            if episode_error:
                raise RuntimeError(
                    f"{name} episode failed for {task_id}/seed={seed}: "
                    f"{matching_rows[0].get('error')}"
                )
    expected = len(keys)
    for name, _controller, policy in collections:
        manifest = _read_jsonl(roots[name] / f"{policy}_manifest.jsonl")
        errors = [row for row in manifest if str(row.get("status")) not in {"ok", "resumed"}]
        if len(manifest) != expected or errors:
            raise RuntimeError(
                f"{name} collection is incomplete: episodes={len(manifest)}/{expected}, "
                f"errors={len(errors)}"
            )


def _csv_options(value: str, allowed: set[str], label: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(",") if item.strip())
    if not values or len(values) != len(set(values)) or any(item not in allowed for item in values):
        raise ValueError(f"invalid {label}: {value}")
    return values


def _resolve_parallel_lanes(paired_execution: str, value: str) -> int:
    if paired_execution == "strict":
        return 1
    if paired_execution != "isolated-parallel":
        raise ValueError(f"unsupported paired execution mode: {paired_execution}")
    if value == "auto":
        from experiments.parallel_runtime import physical_cpu_sets

        physical = len(physical_cpu_sets())
        if physical < 2:
            return 1
        return min(4, max(2, physical - 2))
    lanes = int(value)
    if lanes not in {2, 3, 4}:
        raise ValueError("--parallel-lanes must be auto, 2, 3, or 4")
    isolated_lane_cpu_sets(lanes)
    return lanes


def _require_native_timing_interface(*, require_optimized: bool = False) -> str:
    try:
        import lns2_env
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "the dual-track timing evaluation requires the rebuilt WSL lns2_env module"
        ) from error
    if str(getattr(lns2_env, "repair_timing_schema", "")) != REPAIR_TIMING_SCHEMA:
        raise RuntimeError(
            "lns2_env is stale and lacks repair timing schema v1; rebuild build/linux/project"
        )
    if not callable(
        getattr(
            getattr(lns2_env, "LNS2RepairEnv", object),
            "get_last_reset_timings",
            None,
        )
    ):
        raise RuntimeError("lns2_env is missing get_last_reset_timings; rebuild the module")
    if require_optimized:
        if not callable(
            getattr(
                getattr(lns2_env, "LNS2RepairEnv", object),
                "propose_batch_compact",
                None,
            )
        ):
            raise RuntimeError(
                "lns2_env is missing propose_batch_compact; rebuild the module"
            )
        if not callable(getattr(lns2_env, "batch_online_feature_vectors", None)):
            raise RuntimeError(
                "lns2_env is missing batch_online_feature_vectors; rebuild the module"
            )
    return str(getattr(lns2_env, "__file__", ""))


def _dual_preflight(
    *,
    dataset: Path,
    collection_config: Path,
    controller_bundle: Path,
    task_ids: list[str] | None,
    feature_backend: str,
    tracks: tuple[str, ...],
    wall_clock_seconds: float,
    controller_runtime: str,
    verification_profile: str,
    collections: tuple[tuple[str, str, str], ...] = DUAL_COLLECTIONS,
    cohort_job_keys: set[tuple[str, int]] | None = None,
    stall_guard_config: Path | None = None,
    repair_aware_config: Path | None = None,
    repair_aware_bundle: Path | None = None,
    v3_bundle: Path | None = None,
    allow_unpromoted_v3_diagnostic: bool = False,
) -> dict[str, Any]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"MovingAI OOD dataset is missing: {dataset}")
    if not collection_config.is_file():
        raise FileNotFoundError(f"collection config is missing: {collection_config}")
    native_module = _require_native_timing_interface(
        require_optimized=controller_runtime == "optimized"
    )
    bundle = load_controller_bundle(controller_bundle)
    promotion = dict(bundle.promotion_report)
    if not bool(promotion.get("exact_acceleration_passed")):
        raise ValueError("v2-full exact-equivalence audit has not passed")
    if not bool(promotion.get("feature_performance_passed")):
        raise ValueError("v2-full feature performance audit has not passed")
    if bundle.pruner_threshold is not None:
        raise ValueError("the active runtime does not support pruned controller bundles")
    v3_approval: dict[str, Any] | None = None
    if any(
        controller in {"v3-full", "v3-h3"}
        for _name, controller, _policy in collections
    ):
        if v3_bundle is None:
            raise ValueError("v3 evaluation requires --v3-bundle")
        loaded_v3 = load_v3_controller_bundle(v3_bundle)
        v3_approval = _v3_evaluation_approval(
            loaded_v3.report,
            allow_unpromoted_diagnostic=allow_unpromoted_v3_diagnostic,
        )
    expected_tasks = len(set(task_ids)) if task_ids is not None else FORMAL_TASK_COUNT
    estimates: dict[str, Any] = {}
    fingerprints: dict[str, Any] = {}
    for requested_track in tracks:
        stopping_rule = "historical" if requested_track == "historical" else "wall-clock"
        budget = 300.0 if requested_track == "historical" else wall_clock_seconds
        track_label = "historical" if requested_track == "historical" else f"wall-clock-{budget:g}"
        estimates[track_label] = {}
        fingerprints[track_label] = {}
        for name, controller, _policy in collections:
            dry_run = run_closed_loop_collection(
                dataset,
                collection_config,
                PROJECT_ROOT / "build" / ".lns2-bottleneck-preflight-unused",
                workers=1,
                dry_run=True,
                task_ids=task_ids,
                trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
                controller=controller,
                feature_backend=feature_backend,
                controller_runtime=controller_runtime,
                verification_profile=verification_profile,
                controller_bundle=controller_bundle,
                stall_guard_config=(
                    stall_guard_config if controller == "v2-stall-safe" else None
                ),
                repair_aware_config=(
                    repair_aware_config
                    if controller == "v2-repair-aware"
                    else None
                ),
                repair_aware_bundle=(
                    repair_aware_bundle
                    if controller == "v2-repair-aware"
                    else None
                ),
                v3_bundle=(
                    v3_bundle if controller in {"v3-full", "v3-h3"} else None
                ),
                cohort_job_keys=cohort_job_keys,
                wall_time_budget_seconds=budget,
                episode_process_timeout_seconds=budget + 60.0,
                stopping_rule=stopping_rule,
            )
            estimate = dict(dry_run["estimate"])
            if int(estimate["task_count"]) != expected_tasks:
                raise ValueError(f"{track_label}/{name} resolved to an unexpected task count")
            if cohort_job_keys is None and tuple(
                map(int, estimate["solver_seeds"])
            ) != REGISTERED_SOLVER_SEEDS:
                raise ValueError(f"{track_label}/{name} uses unexpected solver seeds")
            if cohort_job_keys is not None and int(estimate["reset_count"]) != len(
                cohort_job_keys
            ):
                raise ValueError(f"{track_label}/{name} resolved to an incomplete cohort")
            if stopping_rule == "historical":
                if int(estimate.get("maximum_decisions_per_episode") or -1) != 100 or int(estimate.get("environment_max_repair_iterations", -1)) != 100:
                    raise ValueError("historical track must keep the 100-repair limit")
            elif estimate.get("maximum_decisions_per_episode") is not None or int(estimate.get("environment_max_repair_iterations", -1)) != 0:
                raise ValueError("wall-clock track still has an active repair limit")
            estimates[track_label][name] = estimate
            fingerprints[track_label][name] = dry_run["run_fingerprint"]
    return {
        "passed": True,
        "expected_task_count": expected_tasks,
        "expected_episode_count_per_track": (
            len(cohort_job_keys)
            if cohort_job_keys is not None
            else expected_tasks * len(REGISTERED_SOLVER_SEEDS)
        )
        * len(collections),
        "tracks": list(tracks),
        "controller_runtime": controller_runtime,
        "verification_profile": verification_profile,
        "controllers": [item[0] for item in collections],
        "v3_evaluation_approval": v3_approval,
        "native_module": native_module,
        "estimates": estimates,
        "run_fingerprints": fingerprints,
    }


def _v3_evaluation_approval(
    report: dict[str, Any], *, allow_unpromoted_diagnostic: bool
) -> dict[str, Any]:
    pilot_passed = bool(report.get("pilot_passed"))
    if pilot_passed:
        return {
            "pilot_passed": True,
            "unpromoted_diagnostic": False,
            "decision": str(report.get("decision") or "v3_pilot_passed"),
        }
    if not allow_unpromoted_diagnostic:
        raise ValueError("v3 evaluation requires a v3_pilot_passed bundle")
    checks = dict(report.get("pilot_checks") or {})
    required_integrity = {
        "native_available": bool(report.get("native_available")),
        "native_audit_completed": bool(report.get("native_audit_completed")),
        "portable_parity": bool(checks.get("portable_parity")),
    }
    if not all(required_integrity.values()):
        failed = ", ".join(
            name for name, passed in required_integrity.items() if not passed
        )
        raise ValueError(f"unpromoted v3 diagnostic failed integrity checks: {failed}")
    return {
        "pilot_passed": False,
        "unpromoted_diagnostic": True,
        "decision": str(report.get("decision") or "v3_pilot_failed"),
        "integrity_checks": required_integrity,
        "failed_pilot_checks": sorted(
            name for name, passed in checks.items() if not bool(passed)
        ),
    }


def _unsolved_job_keys(
    roots: dict[str, Path],
    collections: tuple[tuple[str, str, str], ...] = DUAL_COLLECTIONS,
) -> set[tuple[str, int]]:
    result: set[tuple[str, int]] = set()
    for name, _controller, policy in collections:
        for row in _read_jsonl(roots[name] / f"{policy}_manifest.jsonl"):
            if str(row.get("status")) not in {"ok", "resumed"} or not bool(
                dict(row.get("summary") or {}).get("success")
            ):
                result.add((str(row["task_id"]), int(row["solver_seed"])))
    return result


def _require_dual_formal_audits(quick_status: Path, storage_audit: Path) -> None:
    if not quick_status.is_file():
        raise FileNotFoundError(f"formal run requires completed dual-track quick status: {quick_status}")
    quick = _read_json(quick_status)
    if str(quick.get("status")) != "complete" or not bool(
        dict(quick.get("bottleneck_validation") or {}).get("passed")
    ):
        raise ValueError("dual-track quick validation did not pass")
    if not storage_audit.is_file():
        raise FileNotFoundError(f"formal run requires compact storage audit: {storage_audit}")
    storage = _read_json(storage_audit)
    if not all(bool(storage.get(name)) for name in ("passed", "exact", "storage_target_passed")):
        raise ValueError("compact storage audit did not pass")


def _run_dual_track(arguments: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    try:
        parallel_lanes = _resolve_parallel_lanes(
            arguments.paired_execution, arguments.parallel_lanes
        )
        if arguments.parallelism_audit and arguments.paired_execution != "isolated-parallel":
            raise ValueError(
                "--parallelism-audit requires --paired-execution isolated-parallel"
            )
        if arguments.parallelism_audit_seconds <= 0.0:
            raise ValueError("--parallelism-audit-seconds must be positive")
        tracks = _csv_options(
            arguments.evaluation_tracks,
            {"historical", "wall-clock"},
            "evaluation tracks",
        )
        controllers = _csv_options(
            arguments.controllers,
            set(CONTROLLER_COLLECTIONS),
            "controllers",
        )
        if not {"official_adaptive", "v2-full"}.issubset(controllers):
            raise ValueError("fair paired evaluation requires official_adaptive and v2-full")
        collections = tuple(CONTROLLER_COLLECTIONS[name] for name in controllers)
        requested_task_ids = (
            [value.strip() for value in arguments.task_ids.split(",") if value.strip()]
            if arguments.task_ids
            else None
        )
        if requested_task_ids is not None and len(requested_task_ids) != len(
            set(requested_task_ids)
        ):
            raise ValueError("--task-ids contains duplicates")
        requested_solver_seeds = (
            tuple(
                int(value.strip())
                for value in arguments.solver_seeds.split(",")
                if value.strip()
            )
            if arguments.solver_seeds
            else REGISTERED_SOLVER_SEEDS
        )
        if (
            not requested_solver_seeds
            or len(requested_solver_seeds) != len(set(requested_solver_seeds))
            or not set(requested_solver_seeds) <= set(REGISTERED_SOLVER_SEEDS)
        ):
            raise ValueError("--solver-seeds must be a unique subset of 1,2,3")
        subset_requested = (
            requested_task_ids is not None
            or requested_solver_seeds != REGISTERED_SOLVER_SEEDS
        )
        if subset_requested and not arguments.diagnostic_subset:
            raise ValueError("task/seed overrides require --diagnostic-subset")
        if arguments.diagnostic_subset and not subset_requested:
            raise ValueError("--diagnostic-subset requires a task or seed override")
        if arguments.mode == "formal" and arguments.diagnostic_subset:
            raise ValueError("formal evaluation cannot use a diagnostic subset")
        if arguments.allow_unpromoted_v3_diagnostic:
            if not arguments.diagnostic_subset or arguments.mode != "quick":
                raise ValueError(
                    "--allow-unpromoted-v3-diagnostic is restricted to a quick diagnostic subset"
                )
            if not {"v3-full", "v3-h3"} & set(controllers):
                raise ValueError(
                    "--allow-unpromoted-v3-diagnostic requires a v3 controller"
                )
        if (
            (
                "v2-stall-safe" in controllers
                or "v2-repair-aware" in controllers
                or "v3-full" in controllers
                or "v3-h3" in controllers
                or arguments.diagnostic_subset
            )
            and not arguments.output
        ):
            raise ValueError(
                "experimental controllers and diagnostic runs require an explicit --output"
            )
        if arguments.wall_clock_seconds <= 0.0:
            raise ValueError("--wall-clock-seconds must be positive")
        if (
            arguments.controller_runtime == "optimized"
            and arguments.feature_backend == "python"
        ):
            raise ValueError(
                "optimized controller runtime requires --feature-backend auto or native"
            )
        if (
            not arguments.skip_wall_clock_sensitivity
            and arguments.wall_clock_sensitivity_seconds is not None
            and arguments.wall_clock_sensitivity_seconds <= arguments.wall_clock_seconds
        ):
            raise ValueError("wall-clock sensitivity budget must exceed the primary wall budget")
        if arguments.long_horizon_auto_extend_seconds is not None:
            if "wall-clock" not in tracks or arguments.wall_clock_seconds < 1800.0:
                raise ValueError(
                    "automatic long-horizon extension requires a wall-clock budget of at least 1800 seconds"
                )
            if (
                arguments.long_horizon_auto_extend_seconds
                <= arguments.wall_clock_seconds
            ):
                raise ValueError(
                    "long-horizon extension budget must exceed the primary wall budget"
                )
            if not arguments.skip_wall_clock_sensitivity:
                raise ValueError(
                    "automatic long-horizon extension requires --skip-wall-clock-sensitivity"
                )
    except ValueError as error:
        parser.error(str(error))

    output = _resolve(arguments.output or DUAL_DEFAULT_OUTPUTS[arguments.mode])
    dataset = _resolve(arguments.dataset)
    collection_config = _resolve(arguments.collection_config)
    controller_bundle = _resolve(arguments.controller_bundle)
    stall_guard_config = (
        _resolve(arguments.stall_guard_config)
        if "v2-stall-safe" in controllers
        else None
    )
    return _run_dual_track_after_validation(
        arguments,
        parser,
        parallel_lanes=parallel_lanes,
        tracks=tracks,
        controllers=controllers,
        collections=collections,
        requested_task_ids=requested_task_ids,
        requested_solver_seeds=requested_solver_seeds,
        output=output,
        dataset=dataset,
        collection_config=collection_config,
        controller_bundle=controller_bundle,
        stall_guard_config=stall_guard_config,
    )


def _paired_lane_worker(job: dict[str, Any]) -> dict[str, Any]:
    collections = {
        str(item[0]): tuple(item) for item in job["collections"]
    }
    completed = 0
    errors = 0
    for entry in job["entries"]:
        task_id = str(entry["task_id"])
        seed = int(entry["solver_seed"])
        for collection_name in entry["controller_order"]:
            name, controller, policy = collections[str(collection_name)]
            root = Path(job["lane_roots"][name])
            run_closed_loop_collection(
                job["dataset"],
                job["collection_config"],
                root,
                phase=policy,
                workers=1,
                resume=True,
                task_ids=job["task_ids"],
                trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
                controller=controller,
                feature_backend=job["feature_backend"],
                controller_runtime=job["controller_runtime"],
                verification_profile=job["verification_profile"],
                controller_bundle=job["controller_bundle"],
                stall_guard_config=(
                    job["stall_guard_config"]
                    if controller == "v2-stall-safe"
                    else None
                ),
                repair_aware_config=(
                    job["repair_aware_config"]
                    if controller == "v2-repair-aware"
                    else None
                ),
                repair_aware_bundle=(
                    job["repair_aware_bundle"]
                    if controller == "v2-repair-aware"
                    else None
                ),
                v3_bundle=(
                    job["v3_bundle"]
                    if controller in {"v3-full", "v3-h3"}
                    else None
                ),
                job_keys={(task_id, seed)},
                cohort_job_keys={tuple(value) for value in job["job_keys"]},
                wall_time_budget_seconds=job["wall_time_budget_seconds"],
                episode_process_timeout_seconds=job[
                    "episode_process_timeout_seconds"
                ],
                stopping_rule=job["stopping_rule"],
                use_global_collection_lock=False,
            )
            rows = _read_jsonl(root / f"{policy}_manifest.jsonl")
            matching = [
                row
                for row in rows
                if str(row.get("task_id")) == task_id
                and int(row.get("solver_seed", -1)) == seed
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"lane {job['lane_id']} did not write one result for "
                    f"{task_id}/seed={seed}/{name}"
                )
            completed += 1
            errors += int(str(matching[0].get("status")) not in {"ok", "resumed"})
    return {
        "lane_id": int(job["lane_id"]),
        "completed_jobs": completed,
        "error_jobs": errors,
        "lane_roots": dict(job["lane_roots"]),
    }


def _prepare_lane_root(canonical: Path, lane: Path) -> None:
    lane.mkdir(parents=True, exist_ok=True)
    canonical_config = _read_json(canonical / "run_config.json")
    lane_config_path = lane / "run_config.json"
    if lane_config_path.is_file():
        lane_config = _read_json(lane_config_path)
        if str(lane_config.get("run_fingerprint")) != str(
            canonical_config.get("run_fingerprint")
        ):
            raise ValueError(f"parallel lane contains a different run: {lane}")
    else:
        shutil.copy2(canonical / "run_config.json", lane_config_path)
    for name in ("qualification_manifest.jsonl", "qualification_report.json"):
        shutil.copy2(canonical / name, lane / name)


def _merge_lane_collection(
    canonical: Path,
    lane_roots: list[Path],
    policy: str,
) -> None:
    for lane in lane_roots:
        for directory in ("episodes", "state_blobs"):
            source = lane / directory
            if source.is_dir():
                shutil.copytree(source, canonical / directory, dirs_exist_ok=True)
    manifest_path = canonical / f"{policy}_manifest.jsonl"
    rows = _read_jsonl(manifest_path) if manifest_path.is_file() else []
    merged = {
        (str(row["task_id"]), int(row["solver_seed"])): row for row in rows
    }
    for lane in lane_roots:
        path = lane / f"{policy}_manifest.jsonl"
        if not path.is_file():
            continue
        for row in _read_jsonl(path):
            key = (str(row["task_id"]), int(row["solver_seed"]))
            previous = merged.get(key)
            if (
                previous is not None
                and str(previous.get("trace_sha256"))
                != str(row.get("trace_sha256"))
            ):
                raise RuntimeError(f"parallel lane result mismatch for {key}")
            merged[key] = row
    _write_jsonl(manifest_path, [merged[key] for key in sorted(merged)])


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _rank_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 1.0

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: (values[index], index))
        result = [0.0] * len(values)
        for rank, index in enumerate(order):
            result[index] = float(rank)
        return result

    left_rank = ranks(left)
    right_rank = ranks(right)
    left_mean = statistics.fmean(left_rank)
    right_mean = statistics.fmean(right_rank)
    numerator = math.fsum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank)
    )
    denominator = math.sqrt(
        math.fsum((value - left_mean) ** 2 for value in left_rank)
        * math.fsum((value - right_mean) ** 2 for value in right_rank)
    )
    return numerator / denominator if denominator > 0.0 else 1.0


def _parallel_audit_rows(
    roots: dict[str, Path], collections: tuple[tuple[str, str, str], ...]
) -> dict[tuple[str, int, str], list[dict[str, Any]]]:
    result = {}
    for name, _controller, policy in collections:
        for manifest in _read_jsonl(roots[name] / f"{policy}_manifest.jsonl"):
            trace = read_trace_events(roots[name] / str(manifest["trace_file"]))
            transitions = []
            for event in trace:
                if str(event.get("event")) != "transition":
                    continue
                controller = dict(event.get("controller") or {})
                timings = dict(event.get("timings") or {})
                transitions.append(
                    {
                        "before_fingerprint": str(event.get("before_fingerprint")),
                        "action": dict(event.get("action") or {}),
                        "selected_candidate_id": controller.get(
                            "selected_candidate_id"
                        ),
                        "pp_seconds": float(timings.get("pp_replan_seconds", 0.0)),
                        "iteration_seconds": float(
                            timings.get("iteration_wall_seconds", 0.0)
                        ),
                    }
                )
            result[
                (str(manifest["task_id"]), int(manifest["solver_seed"]), name)
            ] = transitions
    return result


def _compare_parallel_audit(
    strict_roots: dict[str, Path],
    parallel_roots: dict[str, Path],
    collections: tuple[tuple[str, str, str], ...],
    lanes: int,
) -> dict[str, Any]:
    strict = _parallel_audit_rows(strict_roots, collections)
    parallel = _parallel_audit_rows(parallel_roots, collections)
    if set(strict) != set(parallel):
        raise RuntimeError("parallelism audit has incomplete paired coverage")
    strict_pp: list[float] = []
    parallel_pp: list[float] = []
    strict_iteration: dict[str, list[float]] = collections_module.defaultdict(list)
    parallel_iteration: dict[str, list[float]] = collections_module.defaultdict(list)
    semantic_mismatches = 0
    paired_pp_left: list[float] = []
    paired_pp_right: list[float] = []
    for key in sorted(strict):
        strict_rows = strict[key]
        parallel_rows = parallel[key]
        common = min(len(strict_rows), len(parallel_rows))
        for left, right in zip(strict_rows[:common], parallel_rows[:common]):
            if (
                left["before_fingerprint"] != right["before_fingerprint"]
                or left["action"] != right["action"]
                or left["selected_candidate_id"] != right["selected_candidate_id"]
            ):
                semantic_mismatches += 1
            if left["pp_seconds"] > 0.0 and right["pp_seconds"] > 0.0:
                strict_pp.append(left["pp_seconds"])
                parallel_pp.append(right["pp_seconds"])
                paired_pp_left.append(left["pp_seconds"])
                paired_pp_right.append(right["pp_seconds"])
            strict_iteration[key[2]].append(left["iteration_seconds"])
            parallel_iteration[key[2]].append(right["iteration_seconds"])
    median_inflation = statistics.median(parallel_pp) / max(
        1e-9, statistics.median(strict_pp)
    )
    p95_inflation = _percentile(parallel_pp, 0.95) / max(
        1e-9, _percentile(strict_pp, 0.95)
    )
    correlation = _rank_correlation(paired_pp_left, paired_pp_right)
    reference = collections[0][0]
    pairwise_ratio_delta = 0.0
    strict_reference = statistics.median(strict_iteration[reference])
    parallel_reference = statistics.median(parallel_iteration[reference])
    for name, _controller, _policy in collections[1:]:
        strict_ratio = statistics.median(strict_iteration[name]) / max(
            1e-9, strict_reference
        )
        parallel_ratio = statistics.median(parallel_iteration[name]) / max(
            1e-9, parallel_reference
        )
        pairwise_ratio_delta = max(
            pairwise_ratio_delta,
            abs(parallel_ratio / max(1e-9, strict_ratio) - 1.0),
        )
    checks = {
        "semantic_mismatches_zero": semantic_mismatches == 0,
        "pp_median_inflation_at_most_3pct": median_inflation <= 1.03,
        "pp_p95_inflation_at_most_5pct": p95_inflation <= 1.05,
        "pp_rank_correlation_at_least_098": correlation >= 0.98,
        "controller_ratio_delta_at_most_5pct": pairwise_ratio_delta <= 0.05,
    }
    return {
        "schema": "lns2.paired_parallelism_audit.v1",
        "lanes": lanes,
        "paired_episode_count": len(strict),
        "paired_pp_count": len(paired_pp_left),
        "semantic_mismatch_count": semantic_mismatches,
        "pp_median_inflation": median_inflation,
        "pp_p95_inflation": p95_inflation,
        "pp_rank_correlation": correlation,
        "controller_pairwise_ratio_maximum_delta": pairwise_ratio_delta,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_parallelism_audit(
    *,
    output: Path,
    dataset: Path,
    collection_config: Path,
    controller_bundle: Path,
    stall_guard_config: Path | None,
    repair_aware_config: Path | None,
    repair_aware_bundle: Path | None,
    v3_bundle: Path | None,
    task_ids: list[str] | None,
    cohort_job_keys: set[tuple[str, int]],
    feature_backend: str,
    controller_runtime: str,
    verification_profile: str,
    collections: tuple[tuple[str, str, str], ...],
    requested_lanes: int,
    audit_seconds: float,
    resume: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    if audit_seconds <= 0.0:
        raise ValueError("parallelism audit seconds must be positive")
    preferred = [QUICK_TASKS[index] for index in (0, 2, 5, 6)]
    available_tasks = {task_id for task_id, _seed in cohort_job_keys}
    selected_tasks = [task for task in preferred if task in available_tasks]
    selected_tasks.extend(
        task
        for task in sorted(available_tasks)
        if task not in selected_tasks
    )
    selected_tasks = selected_tasks[:4]
    audit_keys = {
        min(
            (key for key in cohort_job_keys if key[0] == task),
            key=lambda value: value[1],
        )
        for task in selected_tasks
    }
    if len(audit_keys) < 2:
        raise ValueError("parallelism audit requires at least two task/seed cohorts")
    audit_task_ids = sorted({task for task, _seed in audit_keys})
    root = output / "parallelism-audit"
    strict_roots = {
        name: root / "strict" / "collections" / name
        for name, _controller, _policy in collections
    }
    logger.info(
        "Parallelism audit strict reference: cohorts=%s budget=%ss",
        len(audit_keys),
        audit_seconds,
    )
    _run_interleaved_collections(
        roots=strict_roots,
        dataset=dataset,
        collection_config=collection_config,
        controller_bundle=controller_bundle,
        stall_guard_config=stall_guard_config,
        repair_aware_config=repair_aware_config,
        repair_aware_bundle=repair_aware_bundle,
        v3_bundle=v3_bundle,
        task_ids=audit_task_ids,
        feature_backend=feature_backend,
        resume=resume or (root / "strict").is_dir(),
        logger=logger,
        runner_progress=root / "strict" / "progress.json",
        schedule_path=root / "strict" / "execution_schedule.json",
        job_keys=audit_keys,
        wall_time_budget_seconds=audit_seconds,
        episode_process_timeout_seconds=audit_seconds + 60.0,
        stopping_rule="wall-clock",
        collections=collections,
        controller_runtime=controller_runtime,
        verification_profile=verification_profile,
        paired_execution="strict",
        parallel_lanes=1,
    )
    attempts = []
    selected_lanes = 1
    for lanes in range(requested_lanes, 1, -1):
        parallel_roots = {
            name: root / f"lanes-{lanes}" / "collections" / name
            for name, _controller, _policy in collections
        }
        logger.info("Parallelism audit candidate: lanes=%s", lanes)
        _run_interleaved_collections(
            roots=parallel_roots,
            dataset=dataset,
            collection_config=collection_config,
            controller_bundle=controller_bundle,
            stall_guard_config=stall_guard_config,
            repair_aware_config=repair_aware_config,
            repair_aware_bundle=repair_aware_bundle,
            v3_bundle=v3_bundle,
            task_ids=audit_task_ids,
            feature_backend=feature_backend,
            resume=resume or (root / f"lanes-{lanes}").is_dir(),
            logger=logger,
            runner_progress=root / f"lanes-{lanes}" / "progress.json",
            schedule_path=root / f"lanes-{lanes}" / "execution_schedule.json",
            job_keys=audit_keys,
            wall_time_budget_seconds=audit_seconds,
            episode_process_timeout_seconds=audit_seconds + 60.0,
            stopping_rule="wall-clock",
            collections=collections,
            controller_runtime=controller_runtime,
            verification_profile=verification_profile,
            paired_execution="isolated-parallel",
            parallel_lanes=lanes,
        )
        comparison = _compare_parallel_audit(
            strict_roots, parallel_roots, collections, lanes
        )
        attempts.append(comparison)
        if bool(comparison["passed"]):
            selected_lanes = lanes
            break
    report = {
        "schema": "lns2.paired_parallelism_audit_selection.v1",
        "audit_seconds": audit_seconds,
        "audit_job_keys": [list(value) for value in sorted(audit_keys)],
        "requested_lanes": requested_lanes,
        "selected_lanes": selected_lanes,
        "attempts": attempts,
        "parallel_candidate_passed": selected_lanes > 1,
        "fallback_to_strict": selected_lanes == 1,
        "passed": True,
    }
    _write_json(root / "parallelism_audit_report.json", report)
    return report


def _run_isolated_parallel_collections(
    *,
    roots: dict[str, Path],
    dataset: Path,
    collection_config: Path,
    controller_bundle: Path,
    stall_guard_config: Path | None,
    repair_aware_config: Path | None,
    repair_aware_bundle: Path | None,
    v3_bundle: Path | None,
    task_ids: list[str] | None,
    feature_backend: str,
    runner_progress: Path,
    schedule_path: Path,
    schedule: list[dict[str, Any]],
    job_keys: set[tuple[str, int]] | None,
    wall_time_budget_seconds: float | None,
    episode_process_timeout_seconds: float | None,
    stopping_rule: str,
    collections: tuple[tuple[str, str, str], ...],
    controller_runtime: str,
    verification_profile: str,
    parallel_lanes: int,
) -> None:
    if parallel_lanes not in {2, 3, 4}:
        raise ValueError("isolated parallel timing requires 2, 3, or 4 lanes")
    runtime = parallel_runtime_metadata(parallel_lanes)
    effective_job_keys = job_keys or {
        (str(entry["task_id"]), int(entry["solver_seed"])) for entry in schedule
    }
    # Keep assignments for different audited lane counts isolated.  Otherwise
    # a resumed run that falls back from four lanes to two can accidentally
    # merge stale episodes from the old lane partition.
    lane_work = schedule_path.parent / "lane-work" / f"lanes-{parallel_lanes}"
    lane_entries = [[] for _ in range(parallel_lanes)]
    for index, entry in enumerate(schedule):
        lane_entries[index % parallel_lanes].append(entry)
    jobs = []
    for lane_id, entries in enumerate(lane_entries):
        if not entries:
            continue
        lane_roots = {
            name: lane_work / f"lane-{lane_id}" / name
            for name, _controller, _policy in collections
        }
        for name, lane_root in lane_roots.items():
            _prepare_lane_root(roots[name], lane_root)
        jobs.append(
            {
                "lane_id": lane_id,
                "entries": entries,
                "lane_roots": {name: str(path) for name, path in lane_roots.items()},
                "collections": [list(value) for value in collections],
                "dataset": str(dataset),
                "collection_config": str(collection_config),
                "controller_bundle": str(controller_bundle),
                "stall_guard_config": (
                    str(stall_guard_config) if stall_guard_config is not None else None
                ),
                "repair_aware_config": (
                    str(repair_aware_config) if repair_aware_config is not None else None
                ),
                "repair_aware_bundle": (
                    str(repair_aware_bundle) if repair_aware_bundle is not None else None
                ),
                "v3_bundle": str(v3_bundle) if v3_bundle is not None else None,
                "task_ids": task_ids,
                "feature_backend": feature_backend,
                "controller_runtime": controller_runtime,
                "verification_profile": verification_profile,
                "job_keys": [list(value) for value in sorted(effective_job_keys)],
                "wall_time_budget_seconds": wall_time_budget_seconds,
                "episode_process_timeout_seconds": episode_process_timeout_seconds,
                "stopping_rule": stopping_rule,
            }
        )
    total = len(schedule) * len(collections)
    completed = 0
    errors = 0
    cpu_sets = tuple(isolated_lane_cpu_sets(parallel_lanes))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=parallel_lanes,
        initializer=initialize_isolated_worker,
        initargs=(cpu_sets,),
    ) as pool:
        futures = [pool.submit(_paired_lane_worker, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += int(result["completed_jobs"])
            errors += int(result["error_jobs"])
            _write_json(
                runner_progress,
                {
                    "schema": "lns2.lns2_tradeoff_progress.v2",
                    "phase": "paired-isolated-parallel",
                    "status": "running",
                    "completed_jobs": completed,
                    "total_jobs": total,
                    "error_jobs": errors,
                    "parallel_runtime": runtime,
                },
            )
    for name, _controller, policy in collections:
        _merge_lane_collection(
            roots[name],
            [lane_work / f"lane-{lane_id}" / name for lane_id in range(parallel_lanes)],
            policy,
        )
    _write_json(
        runner_progress,
        {
            "schema": "lns2.lns2_tradeoff_progress.v2",
            "phase": "paired-isolated-parallel",
            "status": "complete" if errors == 0 else "error",
            "completed_jobs": completed,
            "total_jobs": total,
            "error_jobs": errors,
            "parallel_runtime": runtime,
        },
    )
    if completed != total or errors:
        raise RuntimeError(
            f"isolated parallel paired collection incomplete: {completed}/{total}, errors={errors}"
        )


def _run_dual_track_after_validation(
    arguments: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    parallel_lanes: int,
    tracks: tuple[str, ...],
    controllers: tuple[str, ...],
    collections: tuple[tuple[str, str, str], ...],
    requested_task_ids: list[str] | None,
    requested_solver_seeds: tuple[int, ...],
    output: Path,
    dataset: Path,
    collection_config: Path,
    controller_bundle: Path,
    stall_guard_config: Path | None,
) -> int:
    repair_aware_config = (
        _resolve(arguments.repair_aware_config)
        if "v2-repair-aware" in controllers
        else None
    )
    repair_aware_bundle = (
        _resolve(arguments.repair_aware_bundle)
        if "v2-repair-aware" in controllers
        else None
    )
    v3_bundle = (
        _resolve(arguments.v3_bundle)
        if {"v3-full", "v3-h3"} & set(controllers)
        else None
    )
    task_ids = (
        requested_task_ids
        if requested_task_ids is not None
        else list(QUICK_TASKS)
        if arguments.mode == "quick"
        else None
    )
    cohort_job_keys = (
        set(
            _cohort_job_keys(
                dataset,
                collection_config,
                task_ids,
                requested_solver_seeds,
            )
        )
        if task_ids is not None
        else None
    )
    sensitivity_seconds = (
        None
        if arguments.skip_wall_clock_sensitivity
        else arguments.wall_clock_sensitivity_seconds
    )
    try:
        preflight = _dual_preflight(
            dataset=dataset,
            collection_config=collection_config,
            controller_bundle=controller_bundle,
            task_ids=task_ids,
            feature_backend=arguments.feature_backend,
            tracks=tracks,
            wall_clock_seconds=float(arguments.wall_clock_seconds),
            controller_runtime=arguments.controller_runtime,
            verification_profile=arguments.verification_profile,
            collections=collections,
            cohort_job_keys=cohort_job_keys,
            stall_guard_config=stall_guard_config,
            repair_aware_config=repair_aware_config,
            repair_aware_bundle=repair_aware_bundle,
            v3_bundle=v3_bundle,
            allow_unpromoted_v3_diagnostic=bool(
                arguments.allow_unpromoted_v3_diagnostic
            ),
        )
        bundle = load_controller_bundle(controller_bundle)
        model_semantic_fingerprint = str(
            bundle.manifest["main_ranker_semantic_fingerprint"]
        )
        prepare_run_output(
            output,
            resume=arguments.resume,
            identity={
                "runner": "run_lns2_tradeoff_evaluation.dual_track",
                "schema_version": 6,
                "mode": arguments.mode,
                "dataset": str(dataset),
                "collection_config": str(collection_config),
                "controller_bundle": str(controller_bundle),
                "feature_backend": arguments.feature_backend,
                "controller_runtime": arguments.controller_runtime,
                "verification_profile": arguments.verification_profile,
                "workers": 1,
                "paired_execution": arguments.paired_execution,
                "parallel_lanes": parallel_lanes,
                "parallelism_audit": bool(arguments.parallelism_audit),
                "parallelism_audit_seconds": float(
                    arguments.parallelism_audit_seconds
                ),
                "tracks": list(tracks),
                "controllers": list(controllers),
                "wall_clock_seconds": float(arguments.wall_clock_seconds),
                "wall_clock_sensitivity_seconds": sensitivity_seconds,
                "long_horizon_auto_extend_seconds": arguments.long_horizon_auto_extend_seconds,
                "stall_guard_config": str(stall_guard_config)
                if stall_guard_config is not None
                else None,
                "repair_aware_config": str(repair_aware_config)
                if repair_aware_config is not None
                else None,
                "repair_aware_bundle": str(repair_aware_bundle)
                if repair_aware_bundle is not None
                else None,
                "v3_bundle": str(v3_bundle) if v3_bundle is not None else None,
                "allow_unpromoted_v3_diagnostic": bool(
                    arguments.allow_unpromoted_v3_diagnostic
                ),
                "diagnostic_subset": bool(arguments.diagnostic_subset),
                "solver_seeds": list(requested_solver_seeds),
                "task_ids": task_ids,
                "run_fingerprints": preflight["run_fingerprints"],
                "model_semantic_fingerprint": model_semantic_fingerprint,
                "runner_implementation_sha256": {
                    "runner": sha256_file(Path(__file__).resolve()),
                    "report": sha256_file(PROJECT_ROOT / "experiments" / "lns2_bottleneck.py"),
                    "collection": sha256_file(PROJECT_ROOT / "experiments" / "closed_loop_confirmation.py"),
                    "v3_controller": sha256_file(
                        PROJECT_ROOT / "experiments" / "v3_controller.py"
                    ),
                },
            },
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    logger = _logger(output / "run.log")
    status_path = output / "status.json"
    started_at = _utc_now()
    track_roots: dict[str, dict[str, Path]] = {}
    try:
        _status(status_path, started_at, status="running", phase="preflight", mode=arguments.mode, output=str(output))
        _write_json(output / "preflight_report.json", preflight)
        if (
            arguments.paired_execution == "isolated-parallel"
            and arguments.parallelism_audit
            and not arguments.skip_collections
        ):
            _status(
                status_path,
                started_at,
                status="running",
                phase="parallelism-audit",
                mode=arguments.mode,
                output=str(output),
            )
            audit_cohort = set(
                cohort_job_keys
                or _cohort_job_keys(
                    dataset,
                    collection_config,
                    task_ids,
                    requested_solver_seeds,
                )
            )
            parallelism_audit = _run_parallelism_audit(
                output=output,
                dataset=dataset,
                collection_config=collection_config,
                controller_bundle=controller_bundle,
                stall_guard_config=stall_guard_config,
                repair_aware_config=repair_aware_config,
                repair_aware_bundle=repair_aware_bundle,
                v3_bundle=v3_bundle,
                task_ids=task_ids,
                cohort_job_keys=audit_cohort,
                feature_backend=arguments.feature_backend,
                controller_runtime=arguments.controller_runtime,
                verification_profile=arguments.verification_profile,
                collections=collections,
                requested_lanes=parallel_lanes,
                audit_seconds=float(arguments.parallelism_audit_seconds),
                resume=arguments.resume,
                logger=logger,
            )
            parallel_lanes = int(parallelism_audit["selected_lanes"])
        if arguments.mode == "formal":
            quick_path = _resolve(arguments.quick_audit)
            if str(arguments.quick_audit) == "build/initlns-lns2-tradeoff-quick-native-v2/status.json":
                quick_path = _resolve(DUAL_DEFAULT_OUTPUTS["quick"] + "/status.json")
            _require_dual_formal_audits(quick_path, _resolve(arguments.storage_audit))

        for requested_track in tracks:
            stopping_rule = "historical" if requested_track == "historical" else "wall-clock"
            budget = 300.0 if requested_track == "historical" else float(arguments.wall_clock_seconds)
            label = "historical" if requested_track == "historical" else f"wall-clock-{budget:g}"
            roots = {
                name: output / "tracks" / label / "collections" / name
                for name, _controller, _policy in collections
            }
            track_roots[label] = roots
            if not arguments.skip_collections:
                _status(status_path, started_at, status="running", phase=label, mode=arguments.mode, output=str(output), progress_file=str(output / "collection_progress.json"))
                _run_interleaved_collections(
                    roots=roots,
                    dataset=dataset,
                    collection_config=collection_config,
                    controller_bundle=controller_bundle,
                    stall_guard_config=stall_guard_config,
                    repair_aware_config=repair_aware_config,
                    repair_aware_bundle=repair_aware_bundle,
                    v3_bundle=v3_bundle,
                    task_ids=task_ids,
                    feature_backend=arguments.feature_backend,
                    resume=arguments.resume,
                    logger=logger,
                    runner_progress=output / "collection_progress.json",
                    schedule_path=output / "tracks" / label / "execution_schedule.json",
                    job_keys=cohort_job_keys,
                    wall_time_budget_seconds=budget,
                    episode_process_timeout_seconds=budget + 60.0,
                    stopping_rule=stopping_rule,
                    collections=collections,
                    controller_runtime=arguments.controller_runtime,
                    verification_profile=arguments.verification_profile,
                    paired_execution=arguments.paired_execution,
                    parallel_lanes=parallel_lanes,
                )
            else:
                logger.warning("%s collections explicitly skipped; existing files will be used", label)

        if "wall-clock" in tracks and sensitivity_seconds is not None:
            primary_label = f"wall-clock-{float(arguments.wall_clock_seconds):g}"
            selected = _unsolved_job_keys(track_roots[primary_label], collections)
            sensitivity_budget = float(sensitivity_seconds)
            sensitivity_label = f"wall-clock-{sensitivity_budget:g}"
            sensitivity_roots = {
                name: output / "tracks" / sensitivity_label / "collections" / name
                for name, _controller, _policy in collections
            }
            track_roots[sensitivity_label] = sensitivity_roots
            logger.info("Wall-clock sensitivity: task-seeds=%s episodes=%s budget=%ss", len(selected), len(selected) * len(collections), sensitivity_budget)
            if selected and not arguments.skip_collections:
                _run_interleaved_collections(
                    roots=sensitivity_roots,
                    dataset=dataset,
                    collection_config=collection_config,
                    controller_bundle=controller_bundle,
                    stall_guard_config=stall_guard_config,
                    repair_aware_config=repair_aware_config,
                    repair_aware_bundle=repair_aware_bundle,
                    v3_bundle=v3_bundle,
                    task_ids=sorted({task for task, _seed in selected}),
                    feature_backend=arguments.feature_backend,
                    resume=arguments.resume,
                    logger=logger,
                    runner_progress=output / "collection_progress.json",
                    schedule_path=output / "tracks" / sensitivity_label / "execution_schedule.json",
                    job_keys=selected,
                    wall_time_budget_seconds=sensitivity_budget,
                    episode_process_timeout_seconds=sensitivity_budget + 60.0,
                    stopping_rule="wall-clock",
                    collections=collections,
                    controller_runtime=arguments.controller_runtime,
                    verification_profile=arguments.verification_profile,
                    paired_execution=arguments.paired_execution,
                    parallel_lanes=parallel_lanes,
                )
            elif not selected:
                track_roots.pop(sensitivity_label)

        if arguments.long_horizon_auto_extend_seconds is not None:
            interim = generate_bottleneck_artifacts(
                track_roots, output / "long-horizon-interim-report"
            )
            selected = {
                (str(task_id), int(seed))
                for task_id, seed in interim.get(
                    "long_horizon_extension_job_keys", []
                )
            }
            extension_budget = float(arguments.long_horizon_auto_extend_seconds)
            extension_label = f"wall-clock-{extension_budget:g}"
            extension_roots = {
                name: output / "tracks" / extension_label / "collections" / name
                for name, _controller, _policy in collections
            }
            if selected:
                logger.info(
                    "Long-horizon extension: task-seeds=%s episodes=%s budget=%ss",
                    len(selected),
                    len(selected) * len(collections),
                    extension_budget,
                )
                _run_interleaved_collections(
                    roots=extension_roots,
                    dataset=dataset,
                    collection_config=collection_config,
                    controller_bundle=controller_bundle,
                    stall_guard_config=stall_guard_config,
                    repair_aware_config=repair_aware_config,
                    repair_aware_bundle=repair_aware_bundle,
                    v3_bundle=v3_bundle,
                    task_ids=sorted({task for task, _seed in selected}),
                    feature_backend=arguments.feature_backend,
                    resume=arguments.resume,
                    logger=logger,
                    runner_progress=output / "collection_progress.json",
                    schedule_path=output
                    / "tracks"
                    / extension_label
                    / "execution_schedule.json",
                    job_keys=selected,
                    wall_time_budget_seconds=extension_budget,
                    episode_process_timeout_seconds=extension_budget + 60.0,
                    stopping_rule="wall-clock",
                    collections=collections,
                    controller_runtime=arguments.controller_runtime,
                    verification_profile=arguments.verification_profile,
                    paired_execution=arguments.paired_execution,
                    parallel_lanes=parallel_lanes,
                )
                track_roots[extension_label] = extension_roots
            else:
                logger.info(
                    "Long-horizon extension skipped: no unsolved task/seed improved by at least 1% in the final 600 seconds"
                )

        _status(status_path, started_at, status="running", phase="report", mode=arguments.mode, output=str(output))
        report = generate_bottleneck_artifacts(track_roots, output / "report")
        if not bool(dict(report.get("validation") or {}).get("passed")):
            raise RuntimeError("bottleneck timing validation failed")
        _write_json(
            output / "collection_progress.json",
            {
                "schema": "lns2.bottleneck_progress.v1",
                "phase": "complete",
                "status": "complete",
                "updated_at": _utc_now(),
                "episode_count": report["episode_count"],
                "iteration_count": report["iteration_count"],
                "error_jobs": 0,
            },
        )
        _status(
            status_path,
            started_at,
            status="complete",
            phase="complete",
            mode=arguments.mode,
            output=str(output),
            report=str(output / "report" / "v2_bottleneck_report.md"),
            stall_recovery_report=(
                str(output / "report" / "stall_recovery_report.md")
                if "v2-stall-safe" in controllers
                else None
            ),
            repair_aware_report=(
                str(output / "report" / "repair_aware_report.md")
                if "v2-repair-aware" in controllers
                else None
            ),
            v3_bundle=str(v3_bundle) if v3_bundle is not None else None,
            v3_report=(
                str(output / "report" / "v3_report.md")
                if {"v3-full", "v3-h3"} & set(controllers)
                else None
            ),
            evaluation_tracks=list(tracks),
            controllers=list(controllers),
            diagnostic_subset=bool(arguments.diagnostic_subset),
            controller_runtime=arguments.controller_runtime,
            verification_profile=arguments.verification_profile,
            model_semantic_fingerprint=model_semantic_fingerprint,
            bottleneck_validation=report["validation"],
            stall_promotion=report.get("stall_promotion"),
            repair_aware_promotion=report.get("repair_aware_promotion"),
            v3_promotion=report.get("v3_promotion"),
            targeted_stall_recovery=report.get("targeted_stall_recovery"),
            episode_count=report["episode_count"],
            iteration_count=report["iteration_count"],
        )
        logger.info("Dual-track bottleneck evaluation complete")
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        logger.error("Evaluation failed: %s", error)
        logger.error("%s", traceback.format_exc())
        _status(status_path, started_at, status="error", phase="failed", mode=arguments.mode, output=str(output), error=f"{type(error).__name__}: {error}")
        return 2


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the active LNS2/v2 paired quick or formal evaluation."
    )
    parser.add_argument("--mode", choices=("quick", "formal"), required=True)
    parser.add_argument(
        "--evaluation-tracks",
        default="historical,wall-clock",
        help="Comma-separated historical and/or wall-clock tracks.",
    )
    parser.add_argument(
        "--controllers",
        default="official_adaptive,v2-full",
        help=(
            "Active controllers: official_adaptive, v2-full, "
            "v2-stall-safe, v2-repair-aware, v3-full, v3-h3."
        ),
    )
    parser.add_argument("--wall-clock-seconds", type=float, default=300.0)
    parser.add_argument(
        "--wall-clock-sensitivity-seconds", type=float, default=600.0
    )
    parser.add_argument("--skip-wall-clock-sensitivity", action="store_true")
    parser.add_argument("--long-horizon-auto-extend-seconds", type=float)
    parser.add_argument("--diagnostic-subset", action="store_true")
    parser.add_argument(
        "--allow-unpromoted-v3-diagnostic",
        action="store_true",
        help=(
            "Allow a native-audited but unpromoted v3 bundle only for an explicit "
            "quick diagnostic subset; this never grants deployment promotion."
        ),
    )
    parser.add_argument("--task-ids")
    parser.add_argument("--solver-seeds")
    parser.add_argument(
        "--dataset", default="build/initlns-movingai-ood-dataset-v1"
    )
    parser.add_argument(
        "--collection-config", default="configs/movingai_ood_collection.json"
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--stall-guard-config", default="configs/v2_stall_guard_v1.json"
    )
    parser.add_argument(
        "--repair-aware-config", default="configs/v2_repair_aware_v1.json"
    )
    parser.add_argument(
        "--repair-aware-bundle",
        default="build/initlns-repair-aware-controller-v1",
    )
    parser.add_argument(
        "--v3-bundle",
        default="build/initlns-v3-pilot-v1/controller",
    )
    parser.add_argument(
        "--feature-backend",
        choices=("auto", "python", "native"),
        default="auto",
    )
    parser.add_argument(
        "--controller-runtime",
        choices=("reference", "optimized", "auto"),
        default="reference",
    )
    parser.add_argument(
        "--verification-profile",
        choices=("audit", "deployment"),
        default="audit",
    )
    parser.add_argument(
        "--paired-execution",
        choices=("strict", "isolated-parallel"),
        default="strict",
        help=(
            "Run each task/seed cohort serially, or run different cohorts on "
            "isolated physical cores while keeping controllers within a cohort serial."
        ),
    )
    parser.add_argument(
        "--parallel-lanes",
        default="auto",
        help="Isolated lane count: auto, 2, 3, or 4.",
    )
    parser.add_argument(
        "--parallelism-audit",
        action="store_true",
        help="Require the isolated timing audit before accepting parallel quick results.",
    )
    parser.add_argument(
        "--parallelism-audit-seconds",
        type=float,
        default=60.0,
        help="Wall-clock budget for each audit episode.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-collections", action="store_true")
    parser.add_argument(
        "--quick-audit",
        default="build/initlns-lns2-tradeoff-quick-native-v2/status.json",
    )
    parser.add_argument(
        "--storage-audit",
        default=(
            "build/initlns-movingai-ood-collection-v2-compact/"
            "equivalence_report.json"
        ),
    )
    arguments = parser.parse_args()
    if arguments.workers != 1:
        parser.error(
            "--workers remains 1 for each episode; use --paired-execution "
            "isolated-parallel and --parallel-lanes to parallelize cohorts"
        )
    return _run_dual_track(arguments, parser)


if __name__ == "__main__":
    raise SystemExit(main())
