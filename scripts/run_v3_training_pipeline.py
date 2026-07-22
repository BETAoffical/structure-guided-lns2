from __future__ import annotations

import argparse
import json
import logging
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments._common import resolve_cli_path, sha256_file  # noqa: E402
from experiments.repair_collection import _read_json, _utc_now, _write_json  # noqa: E402
from experiments.run_output_guard import prepare_run_output  # noqa: E402
from experiments.v3_pilot import collect_v3_pilot_data  # noqa: E402
from experiments.v3_training import (  # noqa: E402
    finalize_v3_native_audit,
    train_v3_controller,
)


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("lns2.v3-pilot")
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


def _status(path: Path, started_at: str, **values: object) -> None:
    _write_json(
        path,
        {
            "schema": "lns2.v3_pipeline_status.v1",
            "started_at": started_at,
            "updated_at": _utc_now(),
            **values,
        },
    )


def _markdown(report: dict[str, object]) -> str:
    training = dict(report["training"])
    gate = dict(training["diagnostic_gate"])
    v3 = dict(gate["v3"])
    v2 = dict(gate["v2"])
    adaptive = dict(gate["adaptive"])
    comparison = dict(gate["v3_vs_v2"])
    cells = dict(gate["cell_gate"])
    return "\n".join(
        [
            "# v3 cost-aware pilot report",
            "",
            f"Decision: `{training['decision']}`",
            "",
            "This is a pilot artifact. It is not deployment-promoted and does not start full, quick, formal, or MovingAI training.",
            "",
            "## Coverage",
            "",
            f"- Training states: {training['training_state_count']}",
            f"- Diagnostic states: {training['diagnostic_state_count']}",
            f"- Agent counts: {training['training_agent_counts']}",
            f"- Native predictor available: {training['native_available']}",
            f"- Portable maximum delta: {float(training['portable_maximum_delta']):.3g}",
            "",
            "## Diagnostic selection",
            "",
            "| Controller | Effective rate | No-progress rate | Mean conflict reduction | Reduction / total second |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| v3 | {float(v3['effective_rate']):.3%} | {float(v3['no_progress_rate']):.3%} | {float(v3['mean_conflict_reduction']):.4f} | {float(v3['conflict_reduction_per_total_second']):.4f} |",
            f"| v2 selected arm | {float(v2['effective_rate']):.3%} | {float(v2['no_progress_rate']):.3%} | {float(v2['mean_conflict_reduction']):.4f} | {float(v2['conflict_reduction_per_total_second']):.4f} |",
            f"| Adaptive reference | {float(adaptive['effective_rate']):.3%} | {float(adaptive['no_progress_rate']):.3%} | {float(adaptive['mean_conflict_reduction']):.4f} | {float(adaptive['conflict_reduction_per_total_second']):.4f} |",
            "",
            f"v3/v2 conflict-reduction ratio: {float(comparison['conflict_reduction_ratio']):.4f}.",
            f"v3/v2 efficiency ratio: {float(comparison['efficiency_ratio']):.4f}.",
            f"Non-inferior layout/agent cells: {cells['noninferior_cell_count']}/{cells['cell_count']}; worst ratio {float(cells['worst_efficiency_ratio']):.4f}.",
            "",
            "## Frozen outcome",
            "",
            json.dumps(training["pilot_checks"], ensure_ascii=False, indent=2, sort_keys=True),
            "",
        ]
    )


def run_pipeline(
    *,
    source: Path,
    output: Path,
    controller_bundle: Path,
    workers: int,
    resume: bool,
) -> dict[str, object]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    implementation = {
        name: sha256_file(PROJECT_ROOT / name)
        for name in (
            "scripts/run_v3_training_pipeline.py",
            "experiments/v3_pilot.py",
            "experiments/v3_training.py",
            "experiments/v3_controller.py",
            "experiments/feature_schema_v3.py",
            "experiments/high_load_rescue.py",
        )
    }
    source_inputs = {
        split: sha256_file(source / "sources" / split / "run_config.json")
        for split in ("policy_train", "policy_validation")
    }
    prepare_run_output(
        output,
        resume=resume,
        identity={
            "runner": "run_v3_training_pipeline.pilot",
            "schema_version": 1,
            "mode": "pilot",
            "source": str(source),
            "source_inputs": source_inputs,
            "controller_bundle": str(controller_bundle),
            "workers": int(workers),
            "implementation": implementation,
            "automatic_followup": False,
        },
    )
    complete_report = output / "v3_pilot_report.json"
    if resume and complete_report.is_file():
        previous = _read_json(complete_report)
        if bool(previous.get("complete")):
            return previous
    started_at = _utc_now()
    logger = _logger(output / "run.log")
    status_path = output / "status.json"
    try:
        logger.info("Selecting and replaying the registered 180-state v3 pilot")
        _status(
            status_path,
            started_at,
            status="running",
            phase="collection",
            mode="pilot",
            output=str(output),
            progress_file=str(output / "collection" / "status.json"),
        )
        collection = collect_v3_pilot_data(
            source=source,
            output=output / "collection",
            controller_bundle=controller_bundle,
            workers=workers,
            resume=resume or (output / "collection" / "run_config.json").is_file(),
        )
        if not bool(collection["complete"]):
            raise RuntimeError("v3 pilot collection did not complete")
        selection_overhead = float(
            dict(collection["selection"])["selection_overhead_seconds"]
        )
        logger.info(
            "Training v3 four-head bundle with frozen selection overhead %.6fs",
            selection_overhead,
        )
        _status(
            status_path,
            started_at,
            status="running",
            phase="training",
            mode="pilot",
            output=str(output),
            completed_states=int(collection["completed_state_count"]),
            total_states=int(collection["requested_state_count"]),
            error_states=int(collection["error_state_count"]),
        )
        training = train_v3_controller(
            feature_index=output / "collection" / "feature_index.jsonl",
            trial_manifest=output / "collection" / "trial_manifest.jsonl",
            controller_bundle=controller_bundle,
            output=output / "controller",
            selection_overhead_seconds=selection_overhead,
        )
        report: dict[str, object] = {
            "schema": "lns2.v3_pilot_pipeline.v1",
            "mode": "pilot",
            "complete": True,
            "collection": collection,
            "training": training,
            "decision": training["decision"],
            "deployment_promoted": False,
            "full_started": False,
            "quick_started": False,
            "formal_started": False,
        }
        _write_json(complete_report, report)
        (output / "v3_pilot_report.md").write_text(
            _markdown(report), encoding="utf-8", newline="\n"
        )
        _status(
            status_path,
            started_at,
            status="complete",
            phase="complete",
            mode="pilot",
            output=str(output),
            decision=str(training["decision"]),
            completed_states=int(collection["completed_state_count"]),
            total_states=int(collection["requested_state_count"]),
            error_states=0,
            report=str(output / "v3_pilot_report.md"),
            automatic_followup=False,
        )
        logger.info("v3 pilot complete: %s", training["decision"])
        return report
    except BaseException as error:
        logger.error("v3 pilot failed: %s", error)
        logger.error("%s", traceback.format_exc())
        _status(
            status_path,
            started_at,
            status="error",
            phase="failed",
            mode="pilot",
            output=str(output),
            error=f"{type(error).__name__}: {error}",
        )
        raise


def run_training_stage(
    *, output: Path, controller_bundle: Path
) -> dict[str, object]:
    collection_path = output / "collection" / "collection_report.json"
    if not collection_path.is_file():
        raise FileNotFoundError("v3 pilot collection report is missing")
    collection = _read_json(collection_path)
    if not bool(collection.get("complete")):
        raise ValueError("v3 pilot collection is incomplete")
    existing_status = _read_json(output / "status.json")
    started_at = str(existing_status.get("started_at") or _utc_now())
    logger = _logger(output / "run.log")
    try:
        _status(
            output / "status.json",
            started_at,
            status="running",
            phase="training",
            mode="pilot",
            output=str(output),
            completed_states=int(collection["completed_state_count"]),
            total_states=int(collection["requested_state_count"]),
            error_states=int(collection["error_state_count"]),
        )
        selection_overhead = float(
            dict(collection["selection"])["selection_overhead_seconds"]
        )
        training = train_v3_controller(
            feature_index=output / "collection" / "feature_index.jsonl",
            trial_manifest=output / "collection" / "trial_manifest.jsonl",
            controller_bundle=controller_bundle,
            output=output / "controller",
            selection_overhead_seconds=selection_overhead,
        )
        _write_json(output / "training_stage_report.json", training)
        _status(
            output / "status.json",
            started_at,
            status="running",
            phase="awaiting-native-audit",
            mode="pilot",
            output=str(output),
            completed_states=int(collection["completed_state_count"]),
            total_states=int(collection["requested_state_count"]),
            error_states=0,
        )
        logger.info("v3 Windows training stage complete; native audit pending")
        return training
    except BaseException as error:
        _status(
            output / "status.json",
            started_at,
            status="error",
            phase="training-failed",
            mode="pilot",
            output=str(output),
            error=f"{type(error).__name__}: {error}",
        )
        raise


def run_native_audit_stage(*, output: Path) -> dict[str, object]:
    collection = _read_json(output / "collection" / "collection_report.json")
    if not bool(collection.get("complete")):
        raise ValueError("v3 pilot collection is incomplete")
    existing_status = _read_json(output / "status.json")
    started_at = str(existing_status.get("started_at") or _utc_now())
    logger = _logger(output / "run.log")
    try:
        _status(
            output / "status.json",
            started_at,
            status="running",
            phase="native-audit",
            mode="pilot",
            output=str(output),
            completed_states=int(collection["completed_state_count"]),
            total_states=int(collection["requested_state_count"]),
            error_states=0,
        )
        training = finalize_v3_native_audit(
            feature_index=output / "collection" / "feature_index.jsonl",
            trial_manifest=output / "collection" / "trial_manifest.jsonl",
            controller_output=output / "controller",
        )
        report: dict[str, object] = {
            "schema": "lns2.v3_pilot_pipeline.v1",
            "mode": "pilot",
            "complete": True,
            "collection": collection,
            "training": training,
            "decision": training["decision"],
            "deployment_promoted": False,
            "full_started": False,
            "quick_started": False,
            "formal_started": False,
        }
        _write_json(output / "v3_pilot_report.json", report)
        (output / "v3_pilot_report.md").write_text(
            _markdown(report), encoding="utf-8", newline="\n"
        )
        _status(
            output / "status.json",
            started_at,
            status="complete",
            phase="complete",
            mode="pilot",
            output=str(output),
            decision=str(training["decision"]),
            completed_states=int(collection["completed_state_count"]),
            total_states=int(collection["requested_state_count"]),
            error_states=0,
            report=str(output / "v3_pilot_report.md"),
            automatic_followup=False,
        )
        logger.info("v3 native audit complete: %s", training["decision"])
        return report
    except BaseException as error:
        _status(
            output / "status.json",
            started_at,
            status="error",
            phase="native-audit-failed",
            mode="pilot",
            output=str(output),
            error=f"{type(error).__name__}: {error}",
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect, train, and diagnose the registered v3 cost-aware pilot. "
            "The command never starts full, quick, or formal work."
        )
    )
    parser.add_argument("--mode", choices=("pilot",), default="pilot")
    parser.add_argument(
        "--stage", choices=("all", "train", "native-audit"), default="all"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    try:
        output = resolve_cli_path(PROJECT_ROOT, arguments.output)
        controller_bundle = resolve_cli_path(
            PROJECT_ROOT, arguments.controller_bundle
        )
        if arguments.stage == "train":
            report = run_training_stage(
                output=output, controller_bundle=controller_bundle
            )
        elif arguments.stage == "native-audit":
            report = run_native_audit_stage(output=output)
        else:
            report = run_pipeline(
                source=resolve_cli_path(PROJECT_ROOT, arguments.source),
                output=output,
                controller_bundle=controller_bundle,
                workers=arguments.workers,
                resume=arguments.resume,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
