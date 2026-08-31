"""Strict resume validation for Warehouse checkpoint collections.

The checkpoint producers deliberately keep failed and invalid artifacts for
forensic inspection.  This module therefore only decides whether an existing
successful report is safe to reuse; it never repairs or rewrites artifacts.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments._common import read_json, read_jsonl, sha256_file
from experiments.warehouse_disruption_checkpoints import (
    compute_checkpoint_identity_sha256,
)


ReportBuilder = Callable[
    [list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]
]


def checkpoint_failure_record(
    job: Mapping[str, Any],
    status: str,
    error: str,
    *,
    incumbent_supply_error: str,
    include_count_fields: bool = True,
) -> dict[str, Any]:
    """Bind a failed worker result to every immutable schedule field."""

    item = _object(job.get("item"), label="checkpoint worker item")
    normalized = (
        "state_supply_unavailable"
        if status == "error" and error == incumbent_supply_error
        else status
    )
    counts = (
        {"state_count": 0, "outcome_count": 0}
        if include_count_fields
        else {}
    )
    return {
        "status": normalized,
        "error": error,
        **counts,
        **{key: value for key, value in item.items() if key != "row"},
    }


def _object(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return dict(value)


def _object_rows(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    return [_object(row, label=f"{label} row") for row in rows]


def _strict_index(value: Any, *, label: str) -> int:
    if type(value) is not int or int(value) < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return int(value)


def _validate_checkpoint_blob(checkpoint_root: Path, row: Mapping[str, Any]) -> None:
    registered_identity = str(row.get("checkpoint_identity_sha256") or "")
    try:
        computed_identity = compute_checkpoint_identity_sha256(row)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("checkpoint identity cannot be recomputed") from error
    if registered_identity != computed_identity:
        raise ValueError("checkpoint identity changed")

    relative = Path(str(row.get("state_blob") or ""))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint state blob path is not contained")
    blob = (checkpoint_root / relative).resolve()
    try:
        blob.relative_to(checkpoint_root.resolve())
    except ValueError as error:
        raise ValueError("checkpoint state blob path escapes its collection") from error
    if not blob.is_file() or sha256_file(blob) != str(row.get("state_blob_sha256") or ""):
        raise ValueError("checkpoint state blob is missing or changed")


def _validate_schedule_partition(
    *,
    schedule: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    checkpoint_root: Path,
) -> None:
    expected: dict[int, Mapping[str, Any]] = {}
    for item in schedule:
        index = _strict_index(item.get("key_index"), label="schedule key_index")
        if index in expected:
            raise ValueError("checkpoint schedule contains duplicate key_index values")
        expected[index] = item

    # Boundary/load-extension producers enumerate each distinct task before
    # materializing its disturbance replicas.  ``task_index`` is therefore a
    # schedule-only ordinal; bind it through the task_id stored in the success
    # manifest instead of silently leaving it unchecked.
    task_index_by_id: dict[str, int] = {}
    for item in (expected[index] for index in sorted(expected)):
        task_id = str(item.get("task_id") or "")
        if task_id and task_id not in task_index_by_id:
            task_index_by_id[task_id] = len(task_index_by_id)

    observed: set[int] = set()
    for row in rows:
        index = _strict_index(row.get("key_index"), label="manifest key_index")
        if index in observed or index not in expected:
            raise ValueError("checkpoint manifest contains duplicate or unknown schedule rows")
        item = expected[index]
        for field, expected_value in item.items():
            if field == "incumbent_solver_seed":
                incumbent = row.get("incumbent")
                actual_value = (
                    incumbent.get("solver_seed")
                    if isinstance(incumbent, Mapping)
                    else None
                )
            elif field == "selection_seed":
                actual_value = row.get("restore_seed")
            elif field == "task_index":
                actual_value = task_index_by_id.get(str(row.get("task_id") or ""))
            else:
                actual_value = row.get(field)
            if actual_value != expected_value:
                raise ValueError(f"checkpoint manifest {field} differs from its schedule")
        _validate_checkpoint_blob(checkpoint_root, row)
        observed.add(index)

    for failure in failures:
        index = _strict_index(failure.get("key_index"), label="worker failure key_index")
        if index in observed or index not in expected:
            raise ValueError("worker results contain duplicate or unknown schedule rows")
        item = expected[index]
        for field in item:
            if failure.get(field) != item.get(field):
                raise ValueError(f"worker failure {field} differs from its schedule")
        observed.add(index)

    if observed != set(expected):
        raise ValueError("checkpoint manifest and worker results do not cover the schedule")


def reusable_checkpoint_report(
    *,
    report_path: Path,
    plan_path: Path,
    manifest_path: Path,
    worker_path: Path,
    expected_schema: str,
    experiment_id: str,
    config_sha256: str,
    expected_plan: Mapping[str, Any],
    schedule: Sequence[Mapping[str, Any]],
    build_report: ReportBuilder,
) -> dict[str, Any] | None:
    """Return an authenticated successful report, or ``None`` to continue.

    A missing or explicitly/inferably incomplete report is resumable.  Once a
    report claims complete coverage, any failure or integrity mismatch is a
    hard error so the original evidence remains untouched.
    """

    if not report_path.is_file():
        return None
    report = _object(read_json(report_path), label="checkpoint report")
    if (
        report.get("schema") != expected_schema
        or report.get("experiment_id") != experiment_id
        or report.get("config_sha256") != config_sha256
    ):
        raise ValueError("existing checkpoint report identity changed")

    passed = report.get("passed")
    if type(passed) is not bool:
        raise ValueError("existing checkpoint report has invalid passed semantics")
    attempted = _strict_index(
        report.get("attempted_candidate_count"),
        label="report attempted_candidate_count",
    )
    if attempted > len(schedule):
        raise ValueError("checkpoint report exceeds its registered schedule")
    declared_complete = report.get("complete")
    if declared_complete is not None and type(declared_complete) is not bool:
        raise ValueError("existing checkpoint report has invalid complete semantics")
    inferred_complete = attempted == len(schedule)
    complete = inferred_complete if declared_complete is None else declared_complete
    if complete is False:
        if passed is not False:
            raise ValueError("incomplete checkpoint report cannot be passed")
        if inferred_complete:
            raise ValueError(
                "checkpoint report declares incomplete despite full attempted coverage"
            )
        return None
    if not inferred_complete:
        raise ValueError("checkpoint report claims completion without full schedule coverage")
    if passed is not True:
        raise ValueError("existing completed checkpoint report did not pass")

    missing = [
        path
        for path in (plan_path, manifest_path, worker_path)
        if not path.is_file()
    ]
    if missing:
        raise ValueError("completed checkpoint report is missing bound artifacts")
    plan = _object(read_json(plan_path), label="checkpoint plan")
    if plan != dict(expected_plan):
        raise ValueError("checkpoint plan or schedule changed")
    rows = _object_rows(manifest_path, label="checkpoint manifest")
    worker = _object(read_json(worker_path), label="checkpoint worker results")
    if set(worker) != {"completed_count", "failures"}:
        raise ValueError("checkpoint worker result schema changed")
    failures_value = worker.get("failures")
    if not isinstance(failures_value, list):
        raise ValueError("checkpoint worker failures must be a JSON array")
    failures = [
        _object(row, label="checkpoint worker failure") for row in failures_value
    ]
    completed_count = _strict_index(
        worker.get("completed_count"), label="worker completed_count"
    )
    if completed_count != len(rows):
        raise ValueError("checkpoint worker completed_count differs from manifest")

    _validate_schedule_partition(
        schedule=schedule,
        rows=rows,
        failures=failures,
        checkpoint_root=manifest_path.parent,
    )
    expected_report = build_report(rows, failures)
    comparable_report = dict(report)
    # ``complete`` is an optional forward-compatible declaration.  Coverage is
    # still derived independently from the schedule and artifact partition.
    comparable_report.pop("complete", None)
    comparable_expected = dict(expected_report)
    comparable_expected.pop("complete", None)
    if comparable_report != comparable_expected:
        raise ValueError("existing checkpoint report does not match its artifacts")
    return report


__all__ = ["checkpoint_failure_record", "reusable_checkpoint_report"]
