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
from experiments.compact_controller_model import load_controller_bundle  # noqa: E402
from experiments.parallel_runtime import parallel_runtime_metadata  # noqa: E402
from experiments.repair_collection import _read_json, _utc_now, _write_json  # noqa: E402
from experiments.run_output_guard import prepare_run_output  # noqa: E402
from experiments.v3_pilot import collect_v3_pilot_data  # noqa: E402
from experiments.v3_horizon import collect_v3_horizon_data  # noqa: E402
from experiments.v3_horizon_training import (  # noqa: E402
    finalize_v3_horizon_native_audit,
    train_v3_horizon_controller,
)
from experiments.v3_controller import load_v3_controller_bundle  # noqa: E402
from experiments.v3_training import (  # noqa: E402
    finalize_v3_native_audit,
    train_v3_controller,
)
from experiments.v3_s3_pipeline import (  # noqa: E402
    run_v3_s3_collection_stage,
    run_v3_s3_native_audit_stage,
    run_v3_s3_training_stage,
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


def run_horizon_pipeline(
    *,
    source: Path,
    output: Path,
    controller_bundle: Path,
    v3_bundle: Path,
    horizon: int,
    workers: int,
    resume: bool,
    reuse_horizon_collection: Path | None = None,
    stop_after_collection: bool = False,
) -> dict[str, object]:
    if horizon != 3:
        raise ValueError("the registered horizon pilot requires --horizon 3")
    implementation = {
        name: sha256_file(PROJECT_ROOT / name)
        for name in (
            "scripts/run_v3_training_pipeline.py",
            "experiments/v3_horizon.py",
            "experiments/v3_horizon_training.py",
            "experiments/v3_controller.py",
            "experiments/parallel_runtime.py",
            "experiments/feature_schema_v3.py",
        )
    }
    prepare_run_output(
        output,
        resume=resume,
        identity={
            "runner": "run_v3_training_pipeline.horizon-pilot",
            "schema_version": 1,
            "mode": "horizon-pilot",
            "source": str(source),
            "source_collection_sha256": sha256_file(
                source / "collection" / "collection_report.json"
            ),
            "controller_bundle": str(controller_bundle),
            "v3_bundle": str(v3_bundle),
            "horizon": horizon,
            "workers": workers,
            "implementation": implementation,
            "automatic_followup": False,
            "stop_after_collection": bool(stop_after_collection),
            "reuse_horizon_collection": (
                str(reuse_horizon_collection)
                if reuse_horizon_collection is not None
                else None
            ),
            "reuse_collection_report_sha256": (
                sha256_file(reuse_horizon_collection / "collection_report.json")
                if reuse_horizon_collection is not None
                else None
            ),
        },
    )
    report_path = output / "v3_horizon_pilot_report.json"
    if resume and report_path.is_file():
        previous = _read_json(report_path)
        if bool(previous.get("complete")):
            return previous
    started_at = _utc_now()
    logger = _logger(output / "run.log")
    status_path = output / "status.json"
    try:
        logger.info("Collecting paired Horizon-3 branches for 180 registered states")
        _status(
            status_path,
            started_at,
            status="running",
            phase="horizon-collection",
            mode="horizon-pilot",
            output=str(output),
            progress_file=str(output / "horizon_collection" / "status.json"),
        )
        collection = collect_v3_horizon_data(
            source=source,
            output=output / "horizon_collection",
            controller_bundle=controller_bundle,
            v3_bundle=v3_bundle,
            horizon=horizon,
            workers=workers,
            resume=resume
            or (output / "horizon_collection" / "run_config.json").is_file(),
            reuse_collection=reuse_horizon_collection,
        )
        if not bool(collection["complete"]):
            raise RuntimeError("v3-h3 collection did not complete")
        if stop_after_collection:
            report: dict[str, object] = {
                "schema": "lns2.v3_horizon_collection_stage.v1",
                "mode": "horizon-pilot",
                "complete": True,
                "collection": collection,
                "next_stage": "windows-training",
                "quick_started": False,
                "formal_started": False,
            }
            _write_json(output / "horizon_collection_stage_report.json", report)
            _status(
                status_path,
                started_at,
                status="running",
                phase="awaiting-horizon-windows-training",
                mode="horizon-pilot",
                output=str(output),
                completed_states=int(collection["completed_state_count"]),
                total_states=int(collection["requested_state_count"]),
                error_states=0,
            )
            logger.info("v3-H3 collection complete; Windows training pending")
            return report
        _status(
            status_path,
            started_at,
            status="running",
            phase="horizon-training",
            mode="horizon-pilot",
            output=str(output),
            completed_states=int(collection["completed_state_count"]),
            total_states=int(collection["requested_state_count"]),
            error_states=int(collection["error_state_count"]),
        )
        training = train_v3_horizon_controller(
            feature_index=source / "collection" / "feature_index.jsonl",
            one_step_manifest=source / "collection" / "trial_manifest.jsonl",
            horizon_manifest=output
            / "horizon_collection"
            / "horizon_manifest.jsonl",
            controller_bundle=controller_bundle,
            output=output / "controller",
        )
        report: dict[str, object] = {
            "schema": "lns2.v3_horizon_pipeline.v1",
            "mode": "horizon-pilot",
            "complete": True,
            "collection": collection,
            "training": training,
            "decision": training["decision"],
            "deployment_promoted": False,
            "quick_started": False,
            "strict_confirmation_started": False,
            "formal_started": False,
        }
        _write_json(report_path, report)
        gate = dict(training["diagnostic_gate"])
        markdown = "\n".join(
            [
                "# v3-H3 horizon pilot report",
                "",
                f"Decision: `{training['decision']}`",
                "",
                "This artifact predicts cumulative conflict progress and total time over three repairs. It is not deployment-promoted and starts no quick or formal run.",
                "",
                f"- States: {collection['completed_state_count']}/{collection['requested_state_count']}",
                f"- Collection errors: {collection['error_state_count']}",
                f"- H3/v2 efficiency ratio: {float(gate['efficiency_ratio']):.4f}",
                f"- H3/v2 conflict-reduction ratio: {float(gate['conflict_reduction_ratio']):.4f}",
                f"- Native available: {training['native_available']}",
                f"- Portable maximum delta: {float(training['portable_maximum_delta']):.3g}",
                "",
                "## Pilot checks",
                "",
                "```json",
                json.dumps(training["pilot_checks"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        (output / "v3_horizon_pilot_report.md").write_text(
            markdown, encoding="utf-8", newline="\n"
        )
        _status(
            status_path,
            started_at,
            status="complete",
            phase="complete",
            mode="horizon-pilot",
            output=str(output),
            decision=str(training["decision"]),
            completed_states=int(collection["completed_state_count"]),
            total_states=int(collection["requested_state_count"]),
            error_states=0,
            report=str(output / "v3_horizon_pilot_report.md"),
            automatic_followup=False,
        )
        logger.info("v3-H3 pilot complete: %s", training["decision"])
        return report
    except BaseException as error:
        logger.error("v3-H3 pilot failed: %s", error)
        logger.error("%s", traceback.format_exc())
        _status(
            status_path,
            started_at,
            status="error",
            phase="failed",
            mode="horizon-pilot",
            output=str(output),
            error=f"{type(error).__name__}: {error}",
        )
        raise


def run_horizon_training_stage(
    *, source: Path, output: Path, controller_bundle: Path
) -> dict[str, object]:
    collection_path = output / "horizon_collection" / "collection_report.json"
    if not collection_path.is_file():
        raise FileNotFoundError("v3-h3 collection report is missing")
    collection = _read_json(collection_path)
    if not bool(collection.get("complete")):
        raise ValueError("v3-h3 collection is incomplete")
    existing_status = _read_json(output / "status.json")
    started_at = str(existing_status.get("started_at") or _utc_now())
    _status(
        output / "status.json",
        started_at,
        status="running",
        phase="horizon-training",
        mode="horizon-pilot",
        output=str(output),
        completed_states=int(collection["completed_state_count"]),
        total_states=int(collection["requested_state_count"]),
        error_states=0,
    )
    training = train_v3_horizon_controller(
        feature_index=source / "collection" / "feature_index.jsonl",
        one_step_manifest=source / "collection" / "trial_manifest.jsonl",
        horizon_manifest=output
        / "horizon_collection"
        / "horizon_manifest.jsonl",
        controller_bundle=controller_bundle,
        output=output / "controller",
    )
    _write_json(output / "horizon_training_stage_report.json", training)
    _status(
        output / "status.json",
        started_at,
        status="running",
        phase="awaiting-horizon-native-audit",
        mode="horizon-pilot",
        output=str(output),
        completed_states=int(collection["completed_state_count"]),
        total_states=int(collection["requested_state_count"]),
        error_states=0,
    )
    return training


def run_horizon_native_audit_stage(
    *, source: Path, output: Path
) -> dict[str, object]:
    collection = _read_json(
        output / "horizon_collection" / "collection_report.json"
    )
    if not bool(collection.get("complete")):
        raise ValueError("v3-h3 collection is incomplete")
    existing_status = _read_json(output / "status.json")
    started_at = str(existing_status.get("started_at") or _utc_now())
    _status(
        output / "status.json",
        started_at,
        status="running",
        phase="horizon-native-audit",
        mode="horizon-pilot",
        output=str(output),
        completed_states=int(collection["completed_state_count"]),
        total_states=int(collection["requested_state_count"]),
        error_states=0,
    )
    training = finalize_v3_horizon_native_audit(
        feature_index=source / "collection" / "feature_index.jsonl",
        horizon_manifest=output
        / "horizon_collection"
        / "horizon_manifest.jsonl",
        controller_output=output / "controller",
    )
    report: dict[str, object] = {
        "schema": "lns2.v3_horizon_pipeline.v1",
        "mode": "horizon-pilot",
        "complete": True,
        "collection": collection,
        "training": training,
        "decision": training["decision"],
        "deployment_promoted": False,
        "quick_started": False,
        "strict_confirmation_started": False,
        "formal_started": False,
    }
    _write_json(output / "v3_horizon_pilot_report.json", report)
    gate = dict(training["diagnostic_gate"])
    markdown = "\n".join(
        [
            "# v3-H3 horizon pilot report",
            "",
            f"Decision: `{training['decision']}`",
            "",
            f"- States: {collection['completed_state_count']}/{collection['requested_state_count']}",
            f"- Collection errors: {collection['error_state_count']}",
            f"- Reused states: {collection.get('reused_state_count', 0)}",
            f"- H3/v2 efficiency ratio: {float(gate['efficiency_ratio']):.4f}",
            f"- H3/v2 conflict-reduction ratio: {float(gate['conflict_reduction_ratio']):.4f}",
            f"- Native available: {training['native_available']}",
            f"- Portable maximum delta: {float(training['portable_maximum_delta']):.3g}",
            "",
            "No quick, formal, or follow-up training was started.",
            "",
        ]
    )
    (output / "v3_horizon_pilot_report.md").write_text(
        markdown, encoding="utf-8", newline="\n"
    )
    _status(
        output / "status.json",
        started_at,
        status="complete",
        phase="complete",
        mode="horizon-pilot",
        output=str(output),
        decision=str(training["decision"]),
        completed_states=int(collection["completed_state_count"]),
        total_states=int(collection["requested_state_count"]),
        error_states=0,
        report=str(output / "v3_horizon_pilot_report.md"),
        automatic_followup=False,
    )
    return report


def horizon_pipeline_preflight(
    *,
    source: Path,
    controller_bundle: Path,
    v3_bundle: Path,
    horizon: int,
    workers: int,
) -> dict[str, object]:
    if horizon != 3:
        raise ValueError("the registered horizon pilot requires --horizon 3")
    collection = _read_json(source / "collection" / "collection_report.json")
    if not bool(collection.get("complete")) or int(
        collection.get("completed_state_count", 0)
    ) != 180:
        raise ValueError("horizon pilot requires a complete 180-state source")
    main = load_controller_bundle(controller_bundle)
    old_v3 = load_v3_controller_bundle(v3_bundle)
    if old_v3.is_horizon:
        raise ValueError("horizon pilot sampling requires the frozen one-step v3")
    if str(old_v3.manifest.get("main_ranker_semantic_fingerprint")) != str(
        main.manifest.get("main_ranker_semantic_fingerprint")
    ):
        raise ValueError("one-step v3 and v2 controller semantics do not match")
    maximum_model_candidates_per_state = 8
    branch_count = 180 * (maximum_model_candidates_per_state + 1) * 2
    return {
        "schema": "lns2.v3_horizon_preflight.v1",
        "passed": True,
        "source_state_count": 180,
        "horizon": horizon,
        "paired_trials": 2,
        "maximum_model_candidates_per_state": maximum_model_candidates_per_state,
        "maximum_branch_count": branch_count,
        "maximum_repair_execution_count": branch_count * horizon,
        "parallel_runtime": parallel_runtime_metadata(workers),
        "v2_semantic_fingerprint": main.manifest[
            "main_ranker_semantic_fingerprint"
        ],
        "old_v3_schema": old_v3.schema,
        "automatic_quick": False,
        "automatic_formal": False,
    }


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
    parser.add_argument(
        "--mode",
        choices=("pilot", "horizon-pilot", "sequence-pilot"),
        default="pilot",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "source", "collect", "train", "native-audit"),
        default="all",
    )
    parser.add_argument("--source")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument("--workers", default="4")
    parser.add_argument("--training-jobs", default="auto")
    parser.add_argument(
        "--dataset-config", default="configs/v3_s3_pilot_dataset.json"
    )
    parser.add_argument("--parallelism-audit", action="store_true")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--v3-bundle")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-horizon-collection")
    parser.add_argument(
        "--reuse-sequence-sources",
        help=(
            "Reuse a completed sequence-pilot source output in a new collection "
            "directory; the source report SHA256 is included in the run identity."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    try:
        output = resolve_cli_path(PROJECT_ROOT, arguments.output)
        controller_bundle = resolve_cli_path(
            PROJECT_ROOT, arguments.controller_bundle
        )
        if arguments.mode == "sequence-pilot":
            if arguments.source is not None:
                raise ValueError("sequence-pilot creates isolated sources; omit --source")
            if arguments.dry_run:
                raise ValueError("sequence-pilot does not support --dry-run")
            if arguments.stage == "all":
                raise ValueError(
                    "sequence-pilot crosses Windows/WSL boundaries; run "
                    "--stage collect, --stage train, and --stage native-audit "
                    "as separate commands"
                )
            dataset_config = resolve_cli_path(PROJECT_ROOT, arguments.dataset_config)
            reuse_sequence_sources = (
                resolve_cli_path(PROJECT_ROOT, arguments.reuse_sequence_sources)
                if arguments.reuse_sequence_sources
                else None
            )
            if arguments.stage in {"source", "collect", "all"}:
                report = run_v3_s3_collection_stage(
                    project_root=PROJECT_ROOT,
                    output=output,
                    controller_bundle=controller_bundle,
                    dataset_config=dataset_config,
                    workers=arguments.workers,
                    resume=arguments.resume,
                    parallelism_audit=arguments.parallelism_audit,
                    stop_after_sources=arguments.stage == "source",
                    reuse_source_output=reuse_sequence_sources,
                )
            if arguments.stage in {"train", "all"}:
                report = run_v3_s3_training_stage(
                    project_root=PROJECT_ROOT,
                    output=output,
                    training_jobs=arguments.training_jobs,
                    resume=arguments.resume,
                )
            if arguments.stage in {"native-audit", "all"}:
                report = run_v3_s3_native_audit_stage(output=output)
        elif arguments.mode == "horizon-pilot":
            if not arguments.source:
                raise ValueError("horizon-pilot requires --source")
            legacy_workers = int(arguments.workers)
            source = resolve_cli_path(PROJECT_ROOT, arguments.source)
            v3_bundle = (
                resolve_cli_path(PROJECT_ROOT, arguments.v3_bundle)
                if arguments.v3_bundle
                else source / "controller"
            )
            reuse_horizon_collection = (
                resolve_cli_path(PROJECT_ROOT, arguments.reuse_horizon_collection)
                if arguments.reuse_horizon_collection
                else None
            )
            if arguments.dry_run:
                if arguments.stage != "all":
                    raise ValueError("horizon dry-run requires --stage all")
                report = horizon_pipeline_preflight(
                    source=source,
                    controller_bundle=controller_bundle,
                    v3_bundle=v3_bundle,
                    horizon=arguments.horizon,
                    workers=legacy_workers,
                )
            elif arguments.stage == "train":
                report = run_horizon_training_stage(
                    source=source,
                    output=output,
                    controller_bundle=controller_bundle,
                )
            elif arguments.stage == "native-audit":
                report = run_horizon_native_audit_stage(
                    source=source,
                    output=output,
                )
            else:
                report = run_horizon_pipeline(
                    source=source,
                    output=output,
                    controller_bundle=controller_bundle,
                    v3_bundle=v3_bundle,
                    horizon=arguments.horizon,
                    workers=legacy_workers,
                    resume=arguments.resume,
                    reuse_horizon_collection=reuse_horizon_collection,
                    stop_after_collection=arguments.stage == "collect",
                )
        elif arguments.reuse_sequence_sources:
            raise ValueError("--reuse-sequence-sources requires --mode sequence-pilot")
        elif arguments.dry_run:
            raise ValueError("--dry-run is currently supported by horizon-pilot")
        elif arguments.stage == "train":
            report = run_training_stage(
                output=output, controller_bundle=controller_bundle
            )
        elif arguments.stage == "native-audit":
            report = run_native_audit_stage(output=output)
        else:
            if not arguments.source:
                raise ValueError("pilot requires --source")
            report = run_pipeline(
                source=resolve_cli_path(PROJECT_ROOT, arguments.source),
                output=output,
                controller_bundle=controller_bundle,
                workers=int(arguments.workers),
                resume=arguments.resume,
            )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
