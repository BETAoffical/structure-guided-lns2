from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.closed_loop_confirmation import (  # noqa: E402
    CONTROLLER_MODES,
    resolve_controller_mode,
    run_closed_loop_collection,
)
from experiments.closed_loop_trace_storage import (  # noqa: E402
    TRACE_FORMAT_DELTA_GZIP_V2,
    storage_fingerprint,
)
from experiments.final_model_evaluation import (  # noqa: E402
    POLICY_ORDER,
    generate_evaluation_artifacts,
)
from experiments.movingai_ood_confirmation import run_movingai_ood_analysis  # noqa: E402
from experiments.repair_collection import (  # noqa: E402
    _read_json,
    _read_jsonl,
    _utc_now,
    _write_json,
)
from scripts.verify_closed_loop_equivalence import (  # noqa: E402
    equivalence_comparison_fingerprint,
)


QUICK_TASKS = (
    "random-32-32-10__random_05__agents_0200",
    "maze-32-32-4__random_05__agents_0100",
    "room-64-64-16__random_05__agents_0400",
    "warehouse-10-20-10-2-2__random_05__agents_0500",
    "den312d__random_05__agents_0200",
)
REGISTERED_SOLVER_SEEDS = (1, 2, 3)
FORMAL_TASK_COUNT = 48


def _status(path: Path, started_at: str, **values: Any) -> None:
    _write_json(
        path,
        {
            "schema": "lns2.final_model_run_status.v1",
            "started_at": started_at,
            "updated_at": _utc_now(),
            **values,
        },
    )


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("lns2.final_model")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _run_command(command: list[str], logger: logging.Logger) -> None:
    logger.info("Running: %s", " ".join(command))
    environment = os.environ.copy()
    python_path = [str(PROJECT_ROOT), str(NATIVE_BUILD)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert process.stdout is not None
    for line in process.stdout:
        logger.info("%s", line.rstrip())
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"command failed with exit code {return_code}: {' '.join(command)}")


def _start_progress_monitor(
    path: Path, logger: logging.Logger, label: str
) -> tuple[threading.Event, threading.Thread, Callable[[], None]]:
    stop = threading.Event()
    previous: list[str | None] = [None]

    def emit() -> None:
        if not path.is_file():
            return
        try:
            progress = _read_json(path)
        except (OSError, ValueError):
            return
        if "completed_jobs" in progress:
            message = (
                f"{label}: phase={progress.get('phase')} "
                f"status={progress.get('status')} "
                f"completed={progress.get('completed_jobs')}/{progress.get('total_jobs')} "
                f"errors={progress.get('error_jobs', 0)}"
            )
        elif "validated_traces" in progress:
            message = (
                f"{label}: status={progress.get('status')} "
                f"validated={progress.get('validated_traces')}/{progress.get('total_traces')}"
            )
        else:
            message = f"{label}: status={progress.get('status')}"
        if message != previous[0]:
            previous[0] = message
            logger.info(message)

    def monitor() -> None:
        emit()
        while not stop.wait(1.0):
            emit()

    thread = threading.Thread(target=monitor, name=f"lns2-{label}-monitor", daemon=True)
    thread.start()
    return stop, thread, emit


def _stop_progress_monitor(
    monitor: tuple[threading.Event, threading.Thread, Callable[[], None]]
) -> None:
    stop, thread, emit = monitor
    stop.set()
    thread.join(timeout=2.0)
    emit()


def _preflight(
    dataset: Path,
    collection_config: Path,
    task_ids: list[str] | None,
    workers: int | None,
    controller: str,
    feature_backend: str,
    controller_bundle: Path,
    balanced_config: Path | None,
    feature_shadow_validation: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    if not dataset.is_dir():
        raise FileNotFoundError(f"MovingAI OOD dataset is missing: {dataset}")
    _run_command(
        ["cmake", "--build", str(NATIVE_BUILD), "--parallel", str(workers or 4)],
        logger,
    )
    _run_command(
        ["ctest", "--test-dir", str(NATIVE_BUILD), "--output-on-failure"],
        logger,
    )
    _run_command(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.test_closed_loop_trace_storage",
            "tests.test_closed_loop_confirmation",
            "tests.test_controller_v2",
            "tests.test_final_model_evaluation",
            "tests.test_lns2_tradeoff",
        ],
        logger,
    )
    dry_run = run_closed_loop_collection(
        dataset,
        collection_config,
        PROJECT_ROOT / "build" / ".final-model-preflight-unused",
        workers=workers,
        dry_run=True,
        task_ids=task_ids,
        trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
        controller=controller,
        feature_backend=feature_backend,
        controller_bundle=controller_bundle,
        balanced_config=balanced_config,
        feature_shadow_validation=feature_shadow_validation,
    )
    estimate = dict(dry_run["estimate"])
    expected_tasks = len(QUICK_TASKS) if task_ids is not None else FORMAL_TASK_COUNT
    expected_episodes = expected_tasks * len(REGISTERED_SOLVER_SEEDS) * len(POLICY_ORDER)
    if int(estimate["task_count"]) != expected_tasks:
        raise ValueError(
            f"{('quick' if task_ids is not None else 'formal')} task selection resolved "
            f"to {estimate['task_count']} tasks instead of {expected_tasks}"
        )
    if tuple(map(int, estimate["solver_seeds"])) != REGISTERED_SOLVER_SEEDS:
        raise ValueError("evaluation config does not use the registered three solver seeds")
    if tuple(map(str, estimate["policies"])) != POLICY_ORDER:
        raise ValueError("evaluation config does not use the registered five policy order")
    if int(estimate["policy_episode_count"]) != expected_episodes:
        raise ValueError(
            f"evaluation plan contains {estimate['policy_episode_count']} episodes "
            f"instead of {expected_episodes}"
        )
    return dry_run


def _require_controller_audit(
    controller_bundle: Path, controller: str
) -> dict[str, Any] | None:
    if controller == "v1-full":
        return None
    manifest_path = controller_bundle / "controller_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"feature-v2 controller bundle is missing: {manifest_path}")
    manifest = _read_json(manifest_path)
    report_row = dict(manifest.get("promotion_report", {}))
    report_path = controller_bundle / str(report_row.get("file", ""))
    if not report_path.is_file():
        raise FileNotFoundError(f"controller promotion report is missing: {report_path}")
    report = _read_json(report_path)
    if not bool(report.get("exact_acceleration_passed")):
        raise ValueError("feature-v2 exact-equivalence audit did not pass")
    if not bool(report.get("feature_performance_passed")):
        raise ValueError("feature-v2 performance benchmark did not pass")
    if controller == "v2-cascade" and not bool(
        report.get("pruner_offline_validation_passed")
    ):
        raise ValueError("v2-cascade did not pass locked offline validation")
    return report


def _require_quick_audit(
    status_path: Path, controller: str, feature_backend: str
) -> dict[str, Any]:
    if not status_path.is_file():
        raise FileNotFoundError(
            "formal mode requires a completed quick run status: " f"{status_path}"
        )
    status = _read_json(status_path)
    if str(status.get("status")) != "complete" or str(status.get("mode")) != "quick":
        raise ValueError("the registered quick run is not complete")
    quick_root = status_path.parent
    run_config = _read_json(quick_root / "run_config.json")
    if str(run_config.get("controller")) != controller:
        raise ValueError("quick and formal controller modes differ")
    if str(run_config.get("feature_backend")) != feature_backend:
        raise ValueError("quick and formal feature backends differ")
    if controller != "v1-full" and not bool(
        run_config.get("configuration", {}).get("feature_shadow_validation")
    ):
        raise ValueError("quick controller audit did not run paired v1/v2 shadow validation")
    if int(status.get("valid_trace_count", -1)) != 75:
        raise ValueError("quick audit does not contain the registered 75 episodes")
    if controller != "v1-full":
        learned_rows = _read_jsonl(quick_root / "realized_dynamic_manifest.jsonl")
        if len(learned_rows) != len(QUICK_TASKS) * len(REGISTERED_SOLVER_SEEDS):
            raise ValueError("quick shadow audit has an incomplete learned-policy manifest")
        for row in learned_rows:
            if str(row.get("status")) not in {"ok", "resumed"}:
                raise ValueError("quick shadow audit contains a failed learned episode")
            totals = dict(row.get("summary", {}).get("controller_totals") or {})
            if int(totals.get("shadow_validation_count", 0)) != int(
                totals.get("learned_decisions", 0)
            ):
                raise ValueError("quick run did not shadow-validate every learned decision")
            if float(totals.get("shadow_score_max_delta", 0.0)) > 1e-12:
                raise ValueError("quick v1/v2 shadow score tolerance was exceeded")
    return status


def _require_storage_audit(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            "formal mode requires the compact storage equivalence report: " f"{path}"
        )
    report = _read_json(path)
    if not all(
        bool(report.get(field))
        for field in ("passed", "exact", "storage_target_passed")
    ):
        raise ValueError("compact storage equivalence report did not pass")
    if str(report.get("comparison_fingerprint")) != equivalence_comparison_fingerprint():
        raise ValueError("storage audit was produced by a different comparison rule")
    policies = report.get("policies")
    if not isinstance(policies, dict) or set(policies) != set(POLICY_ORDER):
        raise ValueError("storage audit does not cover the registered five policies")
    if any(
        int(row.get("episode_count", -1)) != 144
        or int(row.get("matching_episode_count", -1)) != 144
        for row in policies.values()
        if isinstance(row, dict)
    ) or not all(isinstance(row, dict) for row in policies.values()):
        raise ValueError("storage audit does not contain 720 exactly matching episodes")
    candidate_root = path.parent
    candidate_run_path = candidate_root / "run_config.json"
    if not candidate_run_path.is_file():
        raise FileNotFoundError(
            f"audited compact collection is missing run_config.json: {candidate_root}"
        )
    candidate_run = _read_json(candidate_run_path)
    if str(candidate_run.get("trace_format")) != TRACE_FORMAT_DELTA_GZIP_V2:
        raise ValueError("storage audit candidate is not delta-gzip-v2")
    expected_storage = storage_fingerprint(TRACE_FORMAT_DELTA_GZIP_V2)
    if str(candidate_run.get("storage_fingerprint")) != expected_storage:
        raise ValueError("storage audit candidate uses a different storage fingerprint")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build, run, validate, and report the final frozen InitLNS model."
    )
    parser.add_argument("--mode", choices=("quick", "formal"), required=True)
    parser.add_argument(
        "--dataset", default="build/initlns-movingai-ood-dataset-v1"
    )
    parser.add_argument(
        "--collection-config", default="configs/movingai_ood_collection.json"
    )
    parser.add_argument(
        "--analysis-config", default="configs/movingai_ood_analysis.json"
    )
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--controller", choices=CONTROLLER_MODES)
    parser.add_argument(
        "--feature-backend", choices=("auto", "python", "native"), default="auto"
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--balanced-config",
        default="build/initlns-lns2-speed-quality-calibration/balanced_controller.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument(
        "--storage-audit",
        default="build/initlns-movingai-ood-collection-v2-compact/equivalence_report.json",
    )
    parser.add_argument(
        "--quick-audit",
        default="build/initlns-final-model-quick-v2/status.json",
    )
    arguments = parser.parse_args()

    output = Path(
        arguments.output
        or f"build/initlns-final-model-{arguments.mode}-v2"
    )
    if not output.is_absolute():
        output = PROJECT_ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    logger = _logger(output / "run.log")
    status_path = output / "status.json"
    started_at = _utc_now()
    task_ids = list(QUICK_TASKS) if arguments.mode == "quick" else None
    analysis_progress_path = (
        output / "report" / "registered" / "base" / "analysis_progress.json"
        if arguments.mode == "formal"
        else output / "report" / "analysis_progress.json"
    )
    dataset = (PROJECT_ROOT / arguments.dataset).resolve()
    collection_config = (PROJECT_ROOT / arguments.collection_config).resolve()
    analysis_config = (PROJECT_ROOT / arguments.analysis_config).resolve()
    controller_bundle = (PROJECT_ROOT / arguments.controller_bundle).resolve()
    balanced_config = (PROJECT_ROOT / arguments.balanced_config).resolve()
    controller = arguments.controller or "auto"
    feature_shadow_validation = False
    try:
        controller, controller_bundle, _ = resolve_controller_mode(
            PROJECT_ROOT, arguments.controller, controller_bundle
        )
        feature_shadow_validation = (
            arguments.mode == "quick" and controller != "v1-full"
        )
        _status(
            status_path,
            started_at,
            status="running",
            phase="preflight",
            mode=arguments.mode,
            controller=controller,
            feature_backend=arguments.feature_backend,
            feature_shadow_validation=feature_shadow_validation,
            output=str(output),
            progress_file=str(output / "collection_progress.json"),
            analysis_progress_file=str(analysis_progress_path),
        )
        if arguments.mode == "formal":
            audit_path = (PROJECT_ROOT / arguments.storage_audit).resolve()
            storage_audit = _require_storage_audit(audit_path)
            logger.info(
                "Storage audit passed: reduction %.2f%%",
                100.0 * float(storage_audit["storage"]["reduction_fraction"]),
            )
            quick_audit_path = (PROJECT_ROOT / arguments.quick_audit).resolve()
            _require_quick_audit(
                quick_audit_path, controller, arguments.feature_backend
            )
            logger.info("Quick controller audit passed: %s", quick_audit_path.parent)
        controller_audit = _require_controller_audit(controller_bundle, controller)
        if controller_audit is not None:
            logger.info("Controller exact-equivalence audit passed: %s", controller)
        if not arguments.skip_preflight:
            preflight = _preflight(
                dataset,
                collection_config,
                task_ids,
                arguments.workers,
                controller,
                arguments.feature_backend,
                controller_bundle,
                balanced_config if controller == "v2-balanced" else None,
                feature_shadow_validation,
                logger,
            )
            _write_json(output / "preflight_report.json", preflight)
            logger.info("Preflight passed; frozen model and dataset fingerprints verified")
        else:
            logger.warning("Preflight explicitly skipped")

        _status(
            status_path,
            started_at,
            status="running",
            phase="collection",
            mode=arguments.mode,
            controller=controller,
            feature_backend=arguments.feature_backend,
            feature_shadow_validation=feature_shadow_validation,
            output=str(output),
            progress_file=str(output / "collection_progress.json"),
            analysis_progress_file=str(analysis_progress_path),
        )
        resume = arguments.resume or (output / "run_config.json").is_file()
        collection_monitor = _start_progress_monitor(
            output / "collection_progress.json", logger, "collection"
        )
        try:
            collection_summary = run_closed_loop_collection(
                dataset,
                collection_config,
                output,
                phase="all",
                workers=arguments.workers,
                resume=resume,
                task_ids=task_ids,
                trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
                controller=controller,
                feature_backend=arguments.feature_backend,
                controller_bundle=controller_bundle,
                balanced_config=(
                    balanced_config if controller == "v2-balanced" else None
                ),
                feature_shadow_validation=feature_shadow_validation,
            )
        finally:
            _stop_progress_monitor(collection_monitor)
        _write_json(output / "runner_collection_summary.json", collection_summary)
        logger.info("Collection complete")

        _status(
            status_path,
            started_at,
            status="running",
            phase="analysis",
            mode=arguments.mode,
            controller=controller,
            feature_backend=arguments.feature_backend,
            feature_shadow_validation=feature_shadow_validation,
            output=str(output),
            progress_file=str(output / "collection_progress.json"),
            analysis_progress_file=str(analysis_progress_path),
        )
        official_report = None
        if arguments.mode == "formal":
            analysis_monitor = _start_progress_monitor(
                analysis_progress_path, logger, "analysis"
            )
            try:
                official_report = run_movingai_ood_analysis(
                    output, analysis_config, output / "report" / "registered"
                )
            finally:
                _stop_progress_monitor(analysis_monitor)
        report = generate_evaluation_artifacts(
            output,
            output / "report",
            formal=arguments.mode == "formal",
            validate_traces=arguments.mode == "quick",
            official_report=official_report,
        )
        conclusion = (
            official_report.get("acceptance", {}).get("decision")
            if official_report is not None
            else "quick_smoke_complete_not_for_formal_conclusion"
        )
        _status(
            status_path,
            started_at,
            status="complete",
            phase="complete",
            mode=arguments.mode,
            controller=controller,
            feature_backend=arguments.feature_backend,
            feature_shadow_validation=feature_shadow_validation,
            output=str(output),
            progress_file=str(output / "collection_progress.json"),
            analysis_progress_file=str(analysis_progress_path),
            report=str(output / "report" / "conclusion.md"),
            conclusion=conclusion,
            valid_trace_count=report["valid_trace_count"],
        )
        logger.info("Evaluation complete: %s", conclusion)
        print(json.dumps(report, indent=2, sort_keys=True))
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
            controller=controller,
            feature_backend=arguments.feature_backend,
            feature_shadow_validation=feature_shadow_validation,
            output=str(output),
            progress_file=str(output / "collection_progress.json"),
            analysis_progress_file=str(analysis_progress_path),
            error=f"{type(error).__name__}: {error}",
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
