from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import experiments.stride_structshell_rollback_aware_ttf as ttf
from experiments._common import (
    closed_loop_producer_identity,
    contained_file,
    sha256_file,
)
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
)
from experiments.repair_collection import (
    _read_json,
    _write_json,
    state_fingerprint,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.evaluation.trace_validation import validate_closed_loop_trace


RECOVERY_SCHEMA = "lns2.stride.rollback_aware_ttf_analysis_recovery.v1"
SIDECAR_FILENAME = "analysis_recovery.json"


def _initial_state(
    collection_root: Path, trace_path: Path, event: Mapping[str, Any]
) -> dict[str, Any]:
    if str(event.get("schema")) != EPISODE_SCHEMA_V2:
        state = event.get("state")
        if not isinstance(state, dict):
            raise ValueError("source trace is missing its initial state")
        return dict(state)
    state = read_state_blob(
        resolve_state_blob(trace_path, str(event["state_blob"]), collection_root)
    )
    extras = event.get("state_extras")
    if not isinstance(extras, dict):
        raise ValueError("source trace has invalid initial extras")
    state.update(extras)
    return state


def observed_decision_rows(
    collection_root: Path, manifest: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Reconstruct observed transitions without attempting counterfactual PP replay.

    Official Adaptive legitimately samples its native PP order internally and
    therefore has no deterministic replay seed.  Platform diagnostics only need
    the recorded before/after states, so this reader validates and applies the
    stored deltas without constructing a replay action or invoking the solver.
    """

    trace_path = contained_file(
        collection_root,
        manifest.get("trace_file"),
        field="trace_file",
    )
    expected_trace_sha = manifest.get("trace_sha256")
    if expected_trace_sha is not None and sha256_file(trace_path) != str(
        expected_trace_sha
    ):
        raise ValueError("source trace sha256 mismatch")
    events = read_trace_events(trace_path)
    if len(events) < 2:
        raise ValueError("source trace has no terminal event")
    state = _initial_state(collection_root, trace_path, events[0])
    rows: list[dict[str, Any]] = []
    for event in events[1:-1]:
        controller = event.get("controller")
        if not isinstance(controller, dict):
            raise ValueError("source transition is missing controller data")
        route = str(controller.get("route", ""))
        if route not in {"model", "official_adaptive"}:
            raise ValueError("source transition is missing a valid route")
        before_fingerprint = state_fingerprint(state)
        if before_fingerprint != str(event.get("before_fingerprint")):
            raise ValueError("source before fingerprint mismatch")
        before_repair = repair_structure_fingerprint(state)
        if str(event.get("schema")) == EPISODE_SCHEMA_V2:
            after = apply_state_delta(state, dict(event["state_delta"]))
            after.update(
                apply_extras_delta(state, dict(event["state_extras_delta"]))
            )
        else:
            raw_after = event.get("after")
            if not isinstance(raw_after, dict):
                raise ValueError("legacy source transition is missing after state")
            after = dict(raw_after)
        after_fingerprint = state_fingerprint(after)
        if after_fingerprint != str(event.get("after_fingerprint")):
            raise ValueError("source after fingerprint mismatch")
        metrics = event.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("source transition is missing metrics")
        before_conflicts = int(state["num_of_colliding_pairs"])
        after_conflicts = int(after["num_of_colliding_pairs"])
        if int(metrics.get("conflicts_before", -1)) != before_conflicts:
            raise ValueError("source before conflict count mismatch")
        if int(metrics.get("conflicts_after", -1)) != after_conflicts:
            raise ValueError("source after conflict count mismatch")
        rows.append(
            {
                "decision_index": int(event["decision_index"]),
                "before_platform_signature": before_repair,
                "after_platform_signature": repair_structure_fingerprint(after),
                "before_conflicts": before_conflicts,
                "after_conflicts": after_conflicts,
                "actual_action": dict(event.get("action") or {}),
                "actual_metrics": dict(metrics),
                "controller": dict(controller),
            }
        )
        state = after
    return rows


def _analysis_producer(root: Path) -> dict[str, Any]:
    return closed_loop_producer_identity(
        project_root=root,
        source_files=(
            "experiments/stride_structshell_rollback_aware_ttf_analysis_recovery.py",
            "scripts/recover_stride_structshell_rollback_aware_ttf_analysis.py",
            "experiments/stride_structshell_rollback_aware_ttf.py",
            "experiments/stride_structshell_rollback_aware_platform_replay.py",
            "experiments/closed_loop_trace_storage.py",
            "lns2_selector/runtime/fingerprints.py",
        ),
        native_required=True,
    )


def recover_screen_analysis(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config = ttf.load_config(config_path)
    output = Path(output).resolve()
    stage_output = output / "screen"
    status_path = stage_output / ttf.STATUS_FILENAME
    progress_path = stage_output / "collection_progress.json"
    sidecar_path = stage_output / SIDECAR_FILENAME
    report_path = stage_output / ttf.SCREEN_REPORT_FILENAME
    if sidecar_path.is_file():
        sidecar = _read_json(sidecar_path)
        report = _read_json(report_path)
        status = _read_json(status_path)
        if (
            str(sidecar.get("schema")) != RECOVERY_SCHEMA
            or sha256_file(report_path) != str(sidecar.get("report_sha256"))
            or status.get("complete") is not True
            or str(status.get("analysis_recovery_sha256"))
            != sha256_file(sidecar_path)
        ):
            raise ValueError("existing analysis recovery is incomplete or changed")
        return report
    source_status = _read_json(status_path)
    progress = _read_json(progress_path)
    schedule_rows = ttf.schedule(config, "screen")
    if str(source_status.get("schema")) != ttf.SCREEN_STATUS_SCHEMA:
        raise ValueError("source screen status schema changed")
    if str(source_status.get("config_sha256")) != sha256_file(path):
        raise ValueError("source screen config identity changed")
    schedule_path = stage_output / "execution_schedule.jsonl"
    recorded_schedule = ttf._read_jsonl(schedule_path)
    if recorded_schedule != schedule_rows:
        raise ValueError("source screen schedule content changed")
    if str(source_status.get("schedule_sha256")) != ttf._fingerprint(
        recorded_schedule
    ):
        raise ValueError("source screen schedule identity changed")
    if (
        int(progress.get("completed_jobs", -1)) != len(schedule_rows)
        or int(progress.get("total_jobs", -1)) != len(schedule_rows)
        or int(progress.get("error_jobs", -1)) != 0
        or int(progress.get("timeout_jobs", -1)) != 0
        or str(progress.get("status")) != "complete"
        or str(progress.get("run_fingerprint"))
        != str(source_status.get("run_fingerprint"))
        or int(source_status.get("total_schedule_entries", -1))
        != len(schedule_rows)
    ):
        raise ValueError("source screen collection is not complete and clean")
    collection_producer = source_status.get("producer_identity")
    if not isinstance(collection_producer, dict):
        raise ValueError("source screen collection has no producer identity")
    if ttf._producer(root) != collection_producer:
        raise ValueError("source screen collection producer changed")
    qualification = ttf._qualification_summary(output, config)
    if qualification.get("all_maps_passed") is not True:
        raise ValueError("source screen qualification is incomplete")

    trace_hashes: dict[str, str] = {}
    manifest_hashes: dict[str, str] = {}
    run_config_hashes: dict[str, str] = {}
    manifest_count = 0
    event_count = 0
    for item in schedule_rows:
        manifest = ttf._manifest(output, "screen", item)
        if manifest is None or not ttf._successful_manifest(manifest):
            raise ValueError("source screen manifest coverage changed")
        collection_root = ttf._controller_dir(output, "screen", item)
        manifest_path = ttf._manifest_path(output, "screen", item)
        run_config_path = collection_root / "run_config.json"
        run_config = _read_json(run_config_path)
        trace_path = contained_file(
            collection_root,
            manifest.get("trace_file"),
            field="trace_file",
        )
        actual = sha256_file(trace_path)
        if actual != str(manifest.get("trace_sha256")):
            raise ValueError("source screen trace identity changed")
        key = "/".join(
            (
                str(item["group_id"]),
                str(item["controller"]),
                str(item["task_id"]),
                str(item["solver_seed"]),
            )
        )
        validated = validate_closed_loop_trace(
            trace_path,
            str(run_config["run_fingerprint"]),
            expected_episode_id=str(manifest["episode_id"]),
            expected_policy=str(manifest["policy"]),
            expected_solver_seed=int(manifest["solver_seed"]),
            metric_iteration_budget=(
                int(manifest["metric_iteration_budget"])
                if manifest.get("metric_iteration_budget") is not None
                else None
            ),
            collection_root=collection_root,
        )
        if dict(validated["summary"]) != dict(manifest["summary"]):
            raise ValueError("source screen trace summary changed")
        if int(validated["event_count"]) != int(manifest["trace_event_count"]):
            raise ValueError("source screen trace event count changed")
        trace_hashes[key] = actual
        manifest_hashes[key] = sha256_file(manifest_path)
        run_config_hashes[key] = sha256_file(run_config_path)
        manifest_count += 1
        event_count += int(validated["event_count"])

    previous_reader = ttf._decision_rows
    ttf._decision_rows = observed_decision_rows
    try:
        report = ttf.analyze_screen(
            path,
            output,
            producer=dict(collection_producer),
        )
    finally:
        ttf._decision_rows = previous_reader
    if report.get("screen_passed") is not False:
        raise ValueError("analysis recovery is restricted to the failed frozen screen")

    analysis_producer = _analysis_producer(root)
    recovery = {
        "schema": RECOVERY_SCHEMA,
        "reason": "platform_diagnostic_requires_observed_delta_not_seeded_replay",
        "collection_run_fingerprint": source_status.get("run_fingerprint"),
        "collection_producer_identity": collection_producer,
        "analysis_producer_identity": analysis_producer,
        "source_status_sha256": sha256_file(status_path),
        "source_progress_sha256": sha256_file(progress_path),
        "source_schedule_content_sha256": str(source_status["schedule_sha256"]),
        "source_schedule_file_sha256": sha256_file(schedule_path),
        "manifest_count": manifest_count,
        "trace_event_count": event_count,
        "trace_count": len(trace_hashes),
        "manifest_sha256": manifest_hashes,
        "controller_run_config_sha256": run_config_hashes,
        "trace_sha256": trace_hashes,
        "collection_data_modified": False,
        "solver_reexecuted": False,
        "extension_authorized": False,
    }
    report["analysis_recovery"] = recovery
    _write_json(report_path, report)
    sidecar = {
        **recovery,
        "screen_passed": report.get("screen_passed"),
        "report_sha256": sha256_file(report_path),
    }
    _write_json(sidecar_path, sidecar)
    status = ttf._status(
        output,
        "screen",
        schedule_rows,
        source_status,
        complete=True,
    )
    status["qualification"] = qualification
    status["report_sha256"] = sha256_file(report_path)
    status["analysis_recovery_sha256"] = sha256_file(sidecar_path)
    status["analysis_recovery_schema"] = RECOVERY_SCHEMA
    _write_json(status_path, status)
    return report
