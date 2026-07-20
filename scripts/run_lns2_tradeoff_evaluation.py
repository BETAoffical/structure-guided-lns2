from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import threading
import traceback
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.balanced_controller import load_balanced_controller  # noqa: E402
from experiments._common import sha256_file  # noqa: E402
from experiments.closed_loop_confirmation import (  # noqa: E402
    REPAIR_TIMING_SCHEMA,
    run_closed_loop_collection,
)
from experiments.closed_loop_trace_storage import TRACE_FORMAT_DELTA_GZIP_V2  # noqa: E402
from experiments.compact_controller_model import load_controller_bundle  # noqa: E402
from experiments.lns2_tradeoff import (  # noqa: E402
    generate_timeout_sensitivity_artifacts,
    generate_tradeoff_artifacts,
    timeout_job_keys,
)
from experiments.lns2_bottleneck import (  # noqa: E402
    generate_bottleneck_artifacts,
)
from experiments.repair_collection import (  # noqa: E402
    _load_dataset_rows,
    _read_json,
    _read_jsonl,
    _utc_now,
    _write_json,
)
from experiments.route_counterfactual import run_route_counterfactuals  # noqa: E402
from experiments.run_output_guard import prepare_run_output  # noqa: E402


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
DEFAULT_OUTPUTS = {
    "quick": "build/initlns-lns2-tradeoff-quick-native-v2",
    "formal": "build/initlns-lns2-tradeoff-formal-native-v2",
}
COLLECTIONS = (
    ("official_adaptive", "v1-full", "official_adaptive"),
    ("v1-full", "v1-full", "realized_dynamic"),
    ("v2-full", "v2-full", "realized_dynamic"),
    ("v2-balanced", "v2-balanced", "realized_dynamic"),
)
DUAL_COLLECTIONS = (
    ("official_adaptive", "v1-full", "official_adaptive"),
    ("v2-full", "v2-full", "realized_dynamic"),
)
STALL_SAFE_COLLECTION = ("v2-stall-safe", "v2-stall-safe", "realized_dynamic")
CONTROLLER_COLLECTIONS = {
    item[0]: item for item in (*DUAL_COLLECTIONS, STALL_SAFE_COLLECTION)
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


def _monitor_progress(
    path: Path,
    logger: logging.Logger,
    label: str,
    mirror_path: Path | None = None,
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def monitor() -> None:
        previous: tuple[Any, ...] | None = None
        while not stop.wait(10.0):
            if not path.is_file():
                continue
            try:
                row = _read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if mirror_path is not None:
                _write_json(
                    mirror_path,
                    {
                        **row,
                        "runner_label": label,
                        "source_progress_file": str(path),
                    },
                )
            current = (
                row.get("phase"),
                row.get("status"),
                row.get("completed_jobs"),
                row.get("total_jobs"),
                row.get("error_jobs"),
            )
            if current != previous:
                logger.info(
                    "%s progress: phase=%s status=%s completed=%s/%s errors=%s active=%s",
                    label,
                    row.get("phase"),
                    row.get("status"),
                    row.get("completed_jobs"),
                    row.get("total_jobs"),
                    row.get("error_jobs"),
                    ", ".join(map(str, row.get("active_jobs", []))) or "none",
                )
                previous = current

    thread = threading.Thread(target=monitor, name=f"{label}-progress", daemon=True)
    thread.start()
    return stop, thread


def _stop_monitor(monitor: tuple[threading.Event, threading.Thread]) -> None:
    monitor[0].set()
    monitor[1].join(timeout=2.0)


def _preflight(
    *,
    dataset: Path,
    collection_config: Path,
    controller_bundle: Path,
    balanced_config: Path,
    task_ids: list[str] | None,
    workers: int | None,
    feature_backend: str,
) -> dict[str, Any]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"MovingAI OOD dataset is missing: {dataset}")
    if not collection_config.is_file():
        raise FileNotFoundError(f"collection config is missing: {collection_config}")
    bundle = load_controller_bundle(controller_bundle)
    promotion = dict(bundle.promotion_report)
    if not bool(promotion.get("exact_acceleration_passed")):
        raise ValueError("v2-full exact-equivalence audit has not passed")
    if not bool(promotion.get("feature_performance_passed")):
        raise ValueError("v2-full feature performance audit has not passed")
    if bundle.pruner_threshold is not None:
        raise ValueError("four-way evaluation requires the proposal pruner to be disabled")
    balanced = load_balanced_controller(balanced_config)
    if balanced.pruner_threshold is not None:
        raise ValueError("balanced evaluation must not mix in the failed proposal pruner")
    if str(balanced.source.get("selection_unit")) != "complete_episode":
        raise ValueError(
            "balanced threshold must come from complete-episode calibration, not H4 states"
        )
    estimates = {}
    run_fingerprints = {}
    for name, controller, _policy in COLLECTIONS:
        dry_run = run_closed_loop_collection(
            dataset,
            collection_config,
            PROJECT_ROOT / "build" / ".lns2-tradeoff-preflight-unused",
            workers=workers,
            dry_run=True,
            task_ids=task_ids,
            trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
            controller=controller,
            feature_backend=feature_backend,
            controller_bundle=controller_bundle,
            balanced_config=balanced_config if controller == "v2-balanced" else None,
        )
        estimates[name] = dry_run["estimate"]
        run_fingerprints[name] = dry_run["run_fingerprint"]
    expected_tasks = len(QUICK_TASKS) if task_ids is not None else FORMAL_TASK_COUNT
    for name, estimate in estimates.items():
        if int(estimate["task_count"]) != expected_tasks:
            raise ValueError(f"{name} resolved to an unexpected task count")
        if tuple(map(int, estimate["solver_seeds"])) != REGISTERED_SOLVER_SEEDS:
            raise ValueError(f"{name} does not use the registered solver seeds")
        if (
            abs(float(estimate["wall_time_budget_seconds"]) - 300.0) > 1e-12
            or int(estimate["maximum_decisions_per_episode"]) != 100
        ):
            raise ValueError(
                f"{name} does not use the registered 100-repair / 300-second budget"
            )
    return {
        "passed": True,
        "expected_task_count": expected_tasks,
        "expected_episode_count_per_controller": expected_tasks
        * len(REGISTERED_SOLVER_SEEDS),
        "balanced_config": balanced.payload(),
        "controller_bundle": str(controller_bundle),
        "estimates": estimates,
        "run_fingerprints": run_fingerprints,
    }


def _require_formal_audits(
    quick_status: Path,
    storage_audit: Path,
    *,
    balanced_config_fingerprint: str,
    feature_backend: str,
    model_semantic_fingerprint: str,
) -> None:
    if not quick_status.is_file():
        raise FileNotFoundError(f"formal run requires completed quick status: {quick_status}")
    quick = _read_json(quick_status)
    if str(quick.get("status")) != "complete" or str(quick.get("mode")) != "quick":
        raise ValueError("quick tradeoff evaluation is not complete")
    if not bool(quick.get("counterfactual_coverage_complete")):
        raise ValueError("quick counterfactual coverage did not pass")
    if not bool(quick.get("v1_v2_semantic_equivalence_passed")):
        raise ValueError("quick v1/v2 common-prefix equivalence did not pass")
    if int(quick.get("paired_episode_count", -1)) != len(QUICK_TASKS) * len(
        REGISTERED_SOLVER_SEEDS
    ):
        raise ValueError("quick tradeoff evaluation has incomplete paired coverage")
    if int(quick.get("complete_episode_count", -1)) != len(QUICK_TASKS) * len(
        REGISTERED_SOLVER_SEEDS
    ) * len(COLLECTIONS):
        raise ValueError("quick tradeoff evaluation did not run all four controllers")
    if str(quick.get("balanced_config_fingerprint")) != balanced_config_fingerprint:
        raise ValueError("quick and formal balanced controller configurations differ")
    if str(quick.get("feature_backend")) != feature_backend:
        raise ValueError("quick and formal feature backends differ")
    if str(quick.get("model_semantic_fingerprint")) != model_semantic_fingerprint:
        raise ValueError("quick and formal frozen model semantics differ")
    if not storage_audit.is_file():
        raise FileNotFoundError(f"formal run requires compact storage audit: {storage_audit}")
    storage = _read_json(storage_audit)
    if not all(bool(storage.get(name)) for name in ("passed", "exact", "storage_target_passed")):
        raise ValueError("compact storage audit did not pass")


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
    balanced_config: Path,
    stall_guard_config: Path | None,
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
    collections: tuple[tuple[str, str, str], ...] = COLLECTIONS,
    controller_runtime: str = "reference",
    verification_profile: str = "audit",
) -> None:
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
            balanced_config=balanced_config if controller == "v2-balanced" else None,
            stall_guard_config=(
                stall_guard_config if controller == "v2-stall-safe" else None
            ),
            job_keys=job_keys,
            cohort_job_keys=job_keys,
            wall_time_budget_seconds=wall_time_budget_seconds,
            episode_process_timeout_seconds=episode_process_timeout_seconds,
            stopping_rule=stopping_rule,
        )

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
            "workers": 1,
            "entries": schedule,
        },
    )
    collection_by_name = {item[0]: item for item in collections}
    completed = 0
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
                balanced_config=(
                    balanced_config if controller == "v2-balanced" else None
                ),
                stall_guard_config=(
                    stall_guard_config if controller == "v2-stall-safe" else None
                ),
                job_keys={(task_id, seed)},
                cohort_job_keys=job_keys,
                wall_time_budget_seconds=wall_time_budget_seconds,
                episode_process_timeout_seconds=episode_process_timeout_seconds,
                stopping_rule=stopping_rule,
            )
            completed += 1
            _write_json(
                runner_progress,
                {
                    "schema": "lns2.lns2_tradeoff_progress.v2",
                    "phase": "paired-collections",
                    "status": "running",
                    "completed_jobs": completed,
                    "total_jobs": total,
                    "error_jobs": 0,
                    "active_jobs": [],
                    "task_id": task_id,
                    "solver_seed": seed,
                    "controller": name,
                },
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
        raise ValueError("bottleneck evaluation requires the proposal pruner to be disabled")
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
        "native_module": native_module,
        "estimates": estimates,
        "run_fingerprints": fingerprints,
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
        if (
            ("v2-stall-safe" in controllers or arguments.diagnostic_subset)
            and not arguments.output
        ):
            raise ValueError("stall-safe and diagnostic runs require an explicit --output")
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
                "schema_version": 4,
                "mode": arguments.mode,
                "dataset": str(dataset),
                "collection_config": str(collection_config),
                "controller_bundle": str(controller_bundle),
                "feature_backend": arguments.feature_backend,
                "controller_runtime": arguments.controller_runtime,
                "verification_profile": arguments.verification_profile,
                "workers": 1,
                "tracks": list(tracks),
                "controllers": list(controllers),
                "wall_clock_seconds": float(arguments.wall_clock_seconds),
                "wall_clock_sensitivity_seconds": sensitivity_seconds,
                "long_horizon_auto_extend_seconds": arguments.long_horizon_auto_extend_seconds,
                "stall_guard_config": str(stall_guard_config)
                if stall_guard_config is not None
                else None,
                "diagnostic_subset": bool(arguments.diagnostic_subset),
                "solver_seeds": list(requested_solver_seeds),
                "task_ids": task_ids,
                "run_fingerprints": preflight["run_fingerprints"],
                "model_semantic_fingerprint": model_semantic_fingerprint,
                "runner_implementation_sha256": {
                    "runner": sha256_file(Path(__file__).resolve()),
                    "report": sha256_file(PROJECT_ROOT / "experiments" / "lns2_bottleneck.py"),
                    "collection": sha256_file(PROJECT_ROOT / "experiments" / "closed_loop_confirmation.py"),
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
                    balanced_config=_resolve(arguments.balanced_config),
                    stall_guard_config=stall_guard_config,
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
                    balanced_config=_resolve(arguments.balanced_config),
                    stall_guard_config=stall_guard_config,
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
                    balanced_config=_resolve(arguments.balanced_config),
                    stall_guard_config=stall_guard_config,
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
            evaluation_tracks=list(tracks),
            controllers=list(controllers),
            diagnostic_subset=bool(arguments.diagnostic_subset),
            controller_runtime=arguments.controller_runtime,
            verification_profile=arguments.verification_profile,
            model_semantic_fingerprint=model_semantic_fingerprint,
            bottleneck_validation=report["validation"],
            stall_promotion=report.get("stall_promotion"),
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
        description=(
            "Run the default LNS2/v2 historical and equal-wall-clock paired "
            "evaluation, or the archived four-way evaluation explicitly."
        )
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
        help="The paired controllers; the default fair evaluation requires both.",
    )
    parser.add_argument("--wall-clock-seconds", type=float, default=300.0)
    parser.add_argument(
        "--wall-clock-sensitivity-seconds", type=float, default=600.0
    )
    parser.add_argument(
        "--skip-wall-clock-sensitivity",
        action="store_true",
        help="Do not run the automatic extended wall-clock cohort.",
    )
    parser.add_argument(
        "--long-horizon-auto-extend-seconds",
        type=float,
        help=(
            "After an >=1800-second wall-clock run, rerun all controllers to this "
            "budget only for unsolved task/seeds that improved >=1%% in the final 600 seconds."
        ),
    )
    parser.add_argument(
        "--diagnostic-subset",
        action="store_true",
        help="Mark an explicit task/seed subset as diagnostic and non-promotable.",
    )
    parser.add_argument("--task-ids", help="Comma-separated diagnostic task IDs.")
    parser.add_argument(
        "--solver-seeds", help="Comma-separated diagnostic solver seeds (subset of 1,2,3)."
    )
    parser.add_argument(
        "--legacy-four-way",
        action="store_true",
        help="Run the archived LNS2/v1/v2/balanced protocol instead.",
    )
    parser.add_argument(
        "--counterfactual-routes",
        choices=("skipped",),
        default="skipped",
        help="Run one model repair only for states routed to official LNS2.",
    )
    parser.add_argument("--dataset", default="build/initlns-movingai-ood-dataset-v1")
    parser.add_argument("--collection-config", default="configs/movingai_ood_collection.json")
    parser.add_argument(
        "--controller-bundle", default="artifacts/initlns-closed-loop-controller-v2"
    )
    parser.add_argument(
        "--stall-guard-config", default="configs/v2_stall_guard_v1.json"
    )
    parser.add_argument(
        "--balanced-config",
        default="build/initlns-lns2-speed-quality-calibration/balanced_controller.json",
    )
    parser.add_argument("--feature-backend", choices=("auto", "python", "native"), default="auto")
    parser.add_argument(
        "--controller-runtime",
        choices=("reference", "optimized", "auto"),
        default="reference",
        help="Select the exact reference or compact/dense v2 execution backend.",
    )
    parser.add_argument(
        "--verification-profile",
        choices=("audit", "deployment"),
        default="audit",
        help="Audit checks every decision and shadows reference; deployment samples full checks.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--counterfactual-workers", type=int)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-collections", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument(
        "--timeout-sensitivity-seconds",
        type=float,
        help=(
            "Legacy four-way protocol only: after the 300-second primary run, "
            "rerun all four controllers for task-seed pairs where any controller "
            "timed out. Use --wall-clock-sensitivity-seconds for the default "
            "two-controller evaluation."
        ),
    )
    parser.add_argument(
        "--quick-audit",
        default="build/initlns-lns2-tradeoff-quick-native-v2/status.json",
    )
    parser.add_argument(
        "--storage-audit",
        default="build/initlns-movingai-ood-collection-v2-compact/equivalence_report.json",
    )
    arguments = parser.parse_args()
    if arguments.workers != 1:
        parser.error("primary paired timing requires --workers 1")
    if not arguments.legacy_four_way:
        return _run_dual_track(arguments, parser)
    if (
        arguments.timeout_sensitivity_seconds is not None
        and arguments.timeout_sensitivity_seconds <= 300.0
    ):
        parser.error("--timeout-sensitivity-seconds must be greater than 300")

    output = _resolve(arguments.output or DEFAULT_OUTPUTS[arguments.mode])
    dataset = _resolve(arguments.dataset)
    collection_config = _resolve(arguments.collection_config)
    controller_bundle = _resolve(arguments.controller_bundle)
    balanced_config = _resolve(arguments.balanced_config)
    task_ids = list(QUICK_TASKS) if arguments.mode == "quick" else None
    roots = {
        name: output / "collections" / name for name, _controller, _policy in COLLECTIONS
    }
    counterfactual_root = output / "counterfactual"
    report_root = output / "report"
    try:
        frozen_balanced = load_balanced_controller(balanced_config).payload()
        frozen_bundle = load_controller_bundle(controller_bundle)
        model_semantic_fingerprint = str(
            frozen_bundle.manifest["main_ranker_semantic_fingerprint"]
        )
        preflight = _preflight(
            dataset=dataset,
            collection_config=collection_config,
            controller_bundle=controller_bundle,
            balanced_config=balanced_config,
            task_ids=task_ids,
            workers=arguments.workers,
            feature_backend=arguments.feature_backend,
        )
        prepare_run_output(
            output,
            resume=arguments.resume,
            identity={
                "runner": "run_lns2_tradeoff_evaluation",
                "schema_version": 2,
                "mode": arguments.mode,
                "dataset": str(dataset),
                "collection_config": str(collection_config),
                "controller_bundle": str(controller_bundle),
                "balanced_config": str(balanced_config),
                "feature_backend": arguments.feature_backend,
                "workers": arguments.workers,
                "counterfactual_workers": arguments.counterfactual_workers,
                "counterfactual_routes": arguments.counterfactual_routes,
                "task_ids": task_ids,
                "primary_budget_seconds": 300.0,
                "timeout_sensitivity_seconds": arguments.timeout_sensitivity_seconds,
                "collection_run_fingerprints": preflight["run_fingerprints"],
                "model_semantic_fingerprint": model_semantic_fingerprint,
                "balanced_config_fingerprint": frozen_balanced[
                    "configuration_fingerprint"
                ],
                "runner_implementation_sha256": {
                    "runner": sha256_file(Path(__file__).resolve()),
                    "report": sha256_file(
                        PROJECT_ROOT / "experiments" / "lns2_tradeoff.py"
                    ),
                    "counterfactual": sha256_file(
                        PROJECT_ROOT / "experiments" / "route_counterfactual.py"
                    ),
                    "output_guard": sha256_file(
                        PROJECT_ROOT / "experiments" / "run_output_guard.py"
                    ),
                },
            },
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    logger = _logger(output / "run.log")
    status_path = output / "status.json"
    started_at = _utc_now()
    try:
        _status(
            status_path,
            started_at,
            status="running",
            phase="preflight",
            mode=arguments.mode,
            output=str(output),
        )
        if arguments.mode == "formal":
            _require_formal_audits(
                _resolve(arguments.quick_audit),
                _resolve(arguments.storage_audit),
                balanced_config_fingerprint=str(
                    frozen_balanced["configuration_fingerprint"]
                ),
                feature_backend=arguments.feature_backend,
                model_semantic_fingerprint=model_semantic_fingerprint,
            )
        if not arguments.skip_preflight:
            _write_json(output / "preflight_report.json", preflight)
            logger.info("Preflight passed")
        else:
            logger.warning("Preflight explicitly skipped")

        if not arguments.skip_collections:
            _status(
                status_path,
                started_at,
                status="running",
                phase="four-way-paired-collections",
                mode=arguments.mode,
                output=str(output),
                progress_file=str(output / "collection_progress.json"),
            )
            _run_interleaved_collections(
                roots=roots,
                dataset=dataset,
                collection_config=collection_config,
                controller_bundle=controller_bundle,
                balanced_config=balanced_config,
                stall_guard_config=None,
                task_ids=task_ids,
                feature_backend=arguments.feature_backend,
                resume=arguments.resume,
                logger=logger,
                runner_progress=output / "collection_progress.json",
                schedule_path=output / "execution_schedule.json",
            )
        else:
            logger.warning("Collections explicitly skipped; existing files will be used")

        if not arguments.skip_counterfactual:
            _status(
                status_path,
                started_at,
                status="running",
                phase="counterfactual",
                mode=arguments.mode,
                output=str(output),
                progress_file=str(output / "collection_progress.json"),
            )
            logger.info(
                "Running one model repair only for states skipped by v2-balanced"
            )
            monitor = _monitor_progress(
                counterfactual_root / "collection_progress.json",
                logger,
                "counterfactual",
                output / "collection_progress.json",
            )
            try:
                counterfactual_summary = run_route_counterfactuals(
                    roots["v2-balanced"],
                    counterfactual_root,
                    workers=arguments.counterfactual_workers or arguments.workers or 1,
                    resume=arguments.resume or (counterfactual_root / "run_config.json").is_file(),
                )
            finally:
                _stop_monitor(monitor)
            if not bool(counterfactual_summary.get("passed")):
                raise RuntimeError("counterfactual coverage or replay validation failed")
            logger.info(
                "Skipped-state counterfactual complete: states=%s model-runs=%s",
                counterfactual_summary.get("counterfactual_state_count"),
                counterfactual_summary.get("model_counterfactual_count"),
            )
        else:
            logger.warning("Counterfactual explicitly skipped; existing files will be used")

        _status(
            status_path,
            started_at,
            status="running",
            phase="report",
            mode=arguments.mode,
            output=str(output),
        )
        report = generate_tradeoff_artifacts(
            roots,
            counterfactual_root,
            report_root,
            formal=arguments.mode == "formal",
        )
        sensitivity_report = None
        if arguments.timeout_sensitivity_seconds is not None:
            sensitivity_budget = float(arguments.timeout_sensitivity_seconds)
            selected_job_keys = timeout_job_keys(roots)
            sensitivity_label = f"{sensitivity_budget:g}".replace(".", "p")
            sensitivity_root = output / f"timeout-sensitivity-{sensitivity_label}"
            sensitivity_roots = {
                name: sensitivity_root / "collections" / name
                for name, _controller, _policy in COLLECTIONS
            }
            logger.info(
                "Timeout sensitivity cohort: task-seeds=%s episodes=%s budget=%ss",
                len(selected_job_keys),
                len(selected_job_keys) * len(COLLECTIONS),
                sensitivity_budget,
            )
            if selected_job_keys:
                _status(
                    status_path,
                    started_at,
                    status="running",
                    phase="timeout-sensitivity",
                    mode=arguments.mode,
                    output=str(output),
                    timeout_sensitivity_seconds=sensitivity_budget,
                    timeout_sensitivity_task_seed_count=len(selected_job_keys),
                    progress_file=str(output / "collection_progress.json"),
                )
                _run_interleaved_collections(
                    roots=sensitivity_roots,
                    dataset=dataset,
                    collection_config=collection_config,
                    controller_bundle=controller_bundle,
                    balanced_config=balanced_config,
                    stall_guard_config=None,
                    task_ids=sorted({task_id for task_id, _seed in selected_job_keys}),
                    feature_backend=arguments.feature_backend,
                    resume=arguments.resume,
                    logger=logger,
                    runner_progress=output / "collection_progress.json",
                    schedule_path=sensitivity_root / "execution_schedule.json",
                    job_keys=selected_job_keys,
                    wall_time_budget_seconds=sensitivity_budget,
                    episode_process_timeout_seconds=sensitivity_budget + 60.0,
                )
            sensitivity_report = generate_timeout_sensitivity_artifacts(
                roots,
                sensitivity_roots if selected_job_keys else None,
                sensitivity_root / "report",
                sensitivity_budget_seconds=sensitivity_budget,
            )
            if not bool(sensitivity_report.get("passed")):
                raise RuntimeError("timeout sensitivity validation failed")
        _write_json(
            output / "collection_progress.json",
            {
                "schema": "lns2.lns2_tradeoff_progress.v1",
                "phase": "complete",
                "status": "complete",
                "updated_at": _utc_now(),
                "paired_episode_count": report["paired_episode_count"],
                "complete_episode_count": report["complete_episode_count"],
                "counterfactual_state_count": report["counterfactual_summary"].get(
                    "counterfactual_state_count"
                ),
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
            report=str(report_root / "hybrid_necessity_report.md"),
            conclusion=report["promotion"]["conclusion"],
            eligible_to_replace_default=report["promotion"][
                "eligible_to_replace_default"
            ],
            counterfactual_coverage_complete=bool(
                report["counterfactual_summary"].get("passed")
            ),
            v1_v2_semantic_equivalence_passed=bool(
                report["semantic_equivalence"].get("passed")
            ),
            balanced_config_fingerprint=frozen_balanced[
                "configuration_fingerprint"
            ],
            feature_backend=arguments.feature_backend,
            model_semantic_fingerprint=model_semantic_fingerprint,
            paired_episode_count=report["paired_episode_count"],
            complete_episode_count=report["complete_episode_count"],
            timeout_sensitivity=(
                {
                    "report": str(
                        sensitivity_root / "report" / "timeout_sensitivity_report.md"
                    ),
                    **sensitivity_report,
                }
                if sensitivity_report is not None
                else None
            ),
        )
        logger.info("Evaluation complete: %s", report["promotion"]["conclusion"])
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        logger.error("Evaluation failed: %s", error)
        logger.error("%s", traceback.format_exc())
        _status(
            status_path,
            started_at,
            status="error",
            phase="failed",
            mode=arguments.mode,
            output=str(output),
            error=f"{type(error).__name__}: {error}",
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
