from __future__ import annotations

import argparse
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

from experiments.lns2_speed_quality_calibration import (  # noqa: E402
    run_complete_episode_calibration,
)
from experiments.repair_collection import _utc_now, _write_json  # noqa: E402


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("lns2.speed_quality_calibration")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(path, encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _status(path: Path, started_at: str, **values: Any) -> None:
    _write_json(
        path,
        {
            "schema": "lns2.complete_episode_calibration_status.v2",
            "started_at": started_at,
            "updated_at": _utc_now(),
            **values,
        },
    )


def _progress_monitor(
    output: Path, destination: Path, logger: logging.Logger
) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()

    def monitor() -> None:
        previous: tuple[Any, ...] | None = None
        while not stop.wait(10.0):
            candidates = list(
                (output / "collections").glob(
                    "*/*/collection_progress.json"
                )
            )
            if not candidates:
                continue
            source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            try:
                row = json.loads(source.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(row, dict):
                continue
            label = source.parent.relative_to(output / "collections").as_posix()
            _write_json(
                destination,
                {
                    **row,
                    "calibration_collection": label,
                    "source_progress_file": str(source),
                    "selection_unit": "complete_episode",
                },
            )
            current = (
                label,
                row.get("phase"),
                row.get("completed_jobs"),
                row.get("total_jobs"),
                row.get("error_jobs"),
            )
            if current != previous:
                logger.info(
                    "Calibration %s: phase=%s episodes=%s/%s errors=%s",
                    label,
                    row.get("phase"),
                    row.get("completed_jobs"),
                    row.get("total_jobs"),
                    row.get("error_jobs"),
                )
                previous = current

    thread = threading.Thread(
        target=monitor, name="calibration-progress", daemon=True
    )
    thread.start()
    return stop, thread


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select the frozen v2-balanced conflict threshold from complete "
            "policy_train episodes and validate it once on policy_validation."
        )
    )
    parser.add_argument("--dataset", default="build/initlns-policy-visited-v1")
    parser.add_argument(
        "--collection-config",
        default="configs/policy_visited_natural_collection.json",
    )
    parser.add_argument(
        "--controller-bundle", default="artifacts/initlns-closed-loop-controller-v2"
    )
    parser.add_argument(
        "--output", default="build/initlns-lns2-speed-quality-calibration"
    )
    parser.add_argument(
        "--feature-backend", choices=("auto", "python", "native"), default="auto"
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--selection-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.workers != 1:
        parser.error("complete-episode timing calibration requires --workers 1")

    output = _resolve(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = _logger(output / "run.log")
    status_path = output / "status.json"
    progress_path = output / "collection_progress.json"
    started_at = _utc_now()
    try:
        _status(
            status_path,
            started_at,
            status="running",
            phase=("selection" if arguments.selection_only else "complete-episodes"),
            selection_unit="complete_episode",
            progress_file=str(progress_path),
            default_controller_changed=False,
        )
        _write_json(
            progress_path,
            {
                "schema": "lns2.complete_episode_calibration_progress.v2",
                "phase": (
                    "selection" if arguments.selection_only else "complete-episodes"
                ),
                "status": "running",
                "updated_at": _utc_now(),
                "registered_conflict_thresholds": [0, 1, 2, 4, 8, 16],
                "selection_unit": "complete_episode",
                "workers": 1,
                "error_jobs": 0,
            },
        )
        logger.info(
            "Calibrating thresholds from complete policy_train episodes; "
            "Horizon-4/state counterfactual selection is disabled"
        )
        monitor = _progress_monitor(output, progress_path, logger)
        try:
            report = run_complete_episode_calibration(
                _resolve(arguments.dataset),
                _resolve(arguments.collection_config),
                _resolve(arguments.controller_bundle),
                output,
                feature_backend=arguments.feature_backend,
                workers=arguments.workers,
                resume=arguments.resume,
                selection_only=arguments.selection_only,
            )
        finally:
            monitor[0].set()
            monitor[1].join(timeout=2.0)
        _write_json(
            progress_path,
            {
                "schema": "lns2.complete_episode_calibration_progress.v2",
                "phase": "complete",
                "status": "complete",
                "updated_at": _utc_now(),
                "selected_conflict_threshold": report["selected"][
                    "conflict_threshold"
                ],
                "selection_unit": "complete_episode",
                "error_jobs": 0,
            },
        )
        _status(
            status_path,
            started_at,
            status="complete",
            phase="complete",
            balanced_config=str(output / "balanced_controller.json"),
            selection_status=report["selection_status"],
            selected_conflict_threshold=report["selected"][
                "conflict_threshold"
            ],
            policy_validation_locked_passed=report["policy_validation"][
                "locked_validation_passed"
            ],
            selection_unit="complete_episode",
            default_controller_changed=False,
        )
        logger.info(
            "Calibration complete: threshold=%s status=%s; default remains v2-full",
            report["selected"]["conflict_threshold"],
            report["selection_status"],
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        logger.error("Calibration failed: %s", error)
        logger.error("%s", traceback.format_exc())
        _status(
            status_path,
            started_at,
            status="error",
            phase="failed",
            error=f"{type(error).__name__}: {error}",
            selection_unit="complete_episode",
            default_controller_changed=False,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
