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
from experiments.closed_loop_confirmation import run_closed_loop_collection  # noqa: E402
from experiments.closed_loop_trace_storage import TRACE_FORMAT_DELTA_GZIP_V2  # noqa: E402
from experiments.compact_controller_model import load_controller_bundle  # noqa: E402
from experiments.lns2_tradeoff import generate_tradeoff_artifacts  # noqa: E402
from experiments.repair_collection import (  # noqa: E402
    _load_dataset_rows,
    _read_json,
    _read_jsonl,
    _utc_now,
    _write_json,
)
from experiments.route_counterfactual import run_route_counterfactuals  # noqa: E402


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
COLLECTIONS = (
    ("official_adaptive", "v1-full", "official_adaptive"),
    ("v1-full", "v1-full", "realized_dynamic"),
    ("v2-full", "v2-full", "realized_dynamic"),
    ("v2-balanced", "v2-balanced", "realized_dynamic"),
)


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
    for name, controller, _policy in COLLECTIONS:
        estimates[name] = run_closed_loop_collection(
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
        )["estimate"]
    expected_tasks = len(QUICK_TASKS) if task_ids is not None else FORMAL_TASK_COUNT
    for name, estimate in estimates.items():
        if int(estimate["task_count"]) != expected_tasks:
            raise ValueError(f"{name} resolved to an unexpected task count")
        if tuple(map(int, estimate["solver_seeds"])) != REGISTERED_SOLVER_SEEDS:
            raise ValueError(f"{name} does not use the registered solver seeds")
    return {
        "passed": True,
        "expected_task_count": expected_tasks,
        "expected_episode_count_per_controller": expected_tasks
        * len(REGISTERED_SOLVER_SEEDS),
        "balanced_config": balanced.payload(),
        "controller_bundle": str(controller_bundle),
        "estimates": estimates,
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
    dataset: Path, collection_config: Path, task_ids: list[str] | None
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
        for seed in REGISTERED_SOLVER_SEEDS
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
    task_ids: list[str] | None,
    feature_backend: str,
    resume: bool,
    logger: logging.Logger,
    runner_progress: Path,
    schedule_path: Path,
) -> None:
    for name, controller, _policy in COLLECTIONS:
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
            controller_bundle=controller_bundle,
            balanced_config=balanced_config if controller == "v2-balanced" else None,
        )

    keys = _cohort_job_keys(dataset, collection_config, task_ids)
    schedule = []
    for task_id, seed in keys:
        digest = hashlib.sha256(f"{task_id}|{seed}".encode("utf-8")).hexdigest()
        rotation = int(digest[:8], 16) % len(COLLECTIONS)
        ordered = COLLECTIONS[rotation:] + COLLECTIONS[:rotation]
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
    collection_by_name = {item[0]: item for item in COLLECTIONS}
    completed = 0
    total = len(keys) * len(COLLECTIONS)
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
                controller_bundle=controller_bundle,
                balanced_config=(
                    balanced_config if controller == "v2-balanced" else None
                ),
                job_keys={(task_id, seed)},
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
    for name, _controller, policy in COLLECTIONS:
        manifest = _read_jsonl(roots[name] / f"{policy}_manifest.jsonl")
        errors = [row for row in manifest if str(row.get("status")) not in {"ok", "resumed"}]
        if len(manifest) != expected or errors:
            raise RuntimeError(
                f"{name} collection is incomplete: episodes={len(manifest)}/{expected}, "
                f"errors={len(errors)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run four-way complete paired episodes for original LNS2, v1-full, "
            "v2-full, and v2-balanced."
        )
    )
    parser.add_argument("--mode", choices=("quick", "formal"), required=True)
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
        "--balanced-config",
        default="build/initlns-lns2-speed-quality-calibration/balanced_controller.json",
    )
    parser.add_argument("--feature-backend", choices=("auto", "python", "native"), default="auto")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--counterfactual-workers", type=int)
    parser.add_argument("--output")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-collections", action="store_true")
    parser.add_argument("--skip-counterfactual", action="store_true")
    parser.add_argument(
        "--quick-audit", default="build/initlns-lns2-tradeoff-quick/status.json"
    )
    parser.add_argument(
        "--storage-audit",
        default="build/initlns-movingai-ood-collection-v2-compact/equivalence_report.json",
    )
    arguments = parser.parse_args()
    if arguments.workers != 1:
        parser.error("primary paired timing requires --workers 1")

    output = _resolve(
        arguments.output or f"build/initlns-lns2-tradeoff-{arguments.mode}"
    )
    output.mkdir(parents=True, exist_ok=True)
    logger = _logger(output / "run.log")
    status_path = output / "status.json"
    started_at = _utc_now()
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
            preflight = _preflight(
                dataset=dataset,
                collection_config=collection_config,
                controller_bundle=controller_bundle,
                balanced_config=balanced_config,
                task_ids=task_ids,
                workers=arguments.workers,
                feature_backend=arguments.feature_backend,
            )
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
