from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments._common import (
    json_fingerprint,
    read_json,
    read_jsonl,
    sha256_file,
    validate_producer_identity,
    write_json,
    write_jsonl,
)


RUNNER_CONFIG_SCHEMA = "lns2.runner_output_identity.v1"
RESUMABLE_RUN_IDENTITY_SCHEMA = "lns2.resumable_experiment_identity.v1"


@dataclass(frozen=True)
class PreparedOutput:
    """Validated state for a new or resumed experiment output directory."""

    base_status: dict[str, Any]
    status: dict[str, Any]
    resumed: bool
    completed_report: dict[str, Any] | None = None


def runner_identity_fingerprint(identity: dict[str, Any]) -> str:
    return json_fingerprint(identity)


def prepare_run_output(
    output: str | Path,
    *,
    resume: bool,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate a runner output before any log or status file is opened.

    A non-empty output is immutable unless the caller explicitly asks to resume
    and its runner identity is an exact match.  On a valid resume this function
    performs no writes.
    """

    root = Path(output).resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"runner output is not a directory: {root}")

    fingerprint = runner_identity_fingerprint(identity)
    config_path = root / "runner_config.json"
    has_content = root.is_dir() and next(root.iterdir(), None) is not None
    if has_content:
        if not resume:
            raise ValueError(
                f"runner output is non-empty; pass --resume to continue: {root}"
            )
        if not config_path.is_file():
            raise ValueError(
                "runner output predates output-identity protection or is incomplete; "
                f"choose a new --output directory: {root}"
            )
        try:
            existing = read_json(config_path)
        except (OSError, ValueError) as error:
            raise ValueError(f"runner identity cannot be read: {config_path}") from error
        if not isinstance(existing, dict):
            raise ValueError(f"runner identity is not an object: {config_path}")
        if str(existing.get("schema")) != RUNNER_CONFIG_SCHEMA:
            raise ValueError(f"runner identity schema mismatch: {config_path}")
        if str(existing.get("identity_fingerprint")) != fingerprint:
            raise ValueError(
                "runner output belongs to a different mode, configuration, or "
                f"implementation fingerprint: {root}"
            )
        if existing.get("identity") != identity:
            raise ValueError(f"runner identity payload mismatch: {config_path}")
        return existing

    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": RUNNER_CONFIG_SCHEMA,
        "schema_version": 1,
        "identity_fingerprint": fingerprint,
        "identity": identity,
    }
    write_json(config_path, payload)
    return payload


def load_completed_report(
    output: str | Path,
    *,
    status_filename: str,
    report_filename: str,
    status_schema: str | None = None,
    report_schema: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any] | None:
    """Return a hash-verified completed report without rewriting old evidence."""

    root = Path(output).resolve()
    status_path = root / status_filename
    if not status_path.is_file():
        return None
    status = read_json(status_path)
    if not isinstance(status, dict):
        raise ValueError(f"completed status is not an object: {status_path}")
    if status_schema is not None and status.get("schema") != status_schema:
        raise ValueError(f"completed status schema changed: {status_path}")
    if config_path is not None:
        expected_config = sha256_file(Path(config_path).resolve())
        if status.get("config_sha256") != expected_config:
            raise ValueError(f"completed status config changed: {status_path}")
    complete = status.get("complete")
    if type(complete) is not bool:
        raise ValueError(f"completed status flag is not boolean: {status_path}")
    if not complete:
        return None
    total = status.get("total_schedule_entries")
    completed = status.get("completed_schedule_entries")
    if (
        type(total) is not int
        or total < 0
        or type(completed) is not int
        or completed != total
    ):
        raise ValueError(f"completed status has incomplete schedule coverage: {status_path}")
    report_path = root / report_filename
    expected = status.get("report_sha256")
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(f"completed status lacks a report hash: {status_path}")
    if not report_path.is_file() or sha256_file(report_path) != expected:
        raise ValueError(f"completed report changed: {report_path}")
    report = read_json(report_path)
    if not isinstance(report, dict):
        raise ValueError(f"completed report is not an object: {report_path}")
    if report_schema is not None and report.get("schema") != report_schema:
        raise ValueError(f"completed report schema changed: {report_path}")
    expected_producer = status.get("producer_identity")
    if expected_producer is not None:
        if not isinstance(expected_producer, dict):
            raise ValueError(f"completed status producer identity is invalid: {status_path}")
        validate_producer_identity(
            expected_producer,
            native_required=bool(expected_producer.get("native_required")),
            optional_package_names=tuple(
                dict(expected_producer.get("packages") or {}).keys()
            ),
        )
        if report.get("producer_identity") != expected_producer:
            raise ValueError(f"completed report producer identity changed: {report_path}")
    return report


def prepare_resumable_output(
    output: str | Path,
    *,
    status_filename: str,
    status_schema: str,
    config_path: str | Path,
    schedule: list[dict[str, Any]],
    producer: dict[str, Any],
    resume: bool,
    schedule_filename: str = "execution_schedule.jsonl",
    report_filename: str | None = None,
    report_schema: str | None = None,
    label: str = "experiment",
) -> PreparedOutput:
    """Bind config, schedule, implementation and status before any reuse."""

    root = Path(output).resolve()
    config = Path(config_path).resolve()
    if not config.is_file():
        raise FileNotFoundError(f"{label} config is missing: {config}")
    validate_producer_identity(
        producer,
        native_required=bool(producer.get("native_required")),
        optional_package_names=tuple(producer.get("packages", {}).keys()),
    )
    schedule_fingerprint = json_fingerprint(schedule)
    identity = {
        "schema": RESUMABLE_RUN_IDENTITY_SCHEMA,
        "status_schema": status_schema,
        "config_sha256": sha256_file(config),
        "schedule_sha256": schedule_fingerprint,
        "producer_identity": producer,
        "report_schema": report_schema,
    }
    resumed = (root / "runner_config.json").is_file()
    runner = prepare_run_output(root, resume=resume, identity=identity)
    base_status = {
        "schema": status_schema,
        "config_sha256": identity["config_sha256"],
        "schedule_sha256": schedule_fingerprint,
        "total_schedule_entries": len(schedule),
        "producer_identity": producer,
        "run_fingerprint": runner["identity_fingerprint"],
    }
    status_path = root / status_filename
    schedule_path = root / schedule_filename
    if resumed:
        if not status_path.is_file():
            raise ValueError(f"{label} output has no trusted status; use a new output")
        status = read_json(status_path)
        if not isinstance(status, dict):
            raise ValueError(f"{label} status is not an object: {status_path}")
        for field, expected in base_status.items():
            if status.get(field) != expected:
                raise ValueError(f"{label} resume {field} mismatch")
        if not schedule_path.is_file():
            raise ValueError(f"{label} resume schedule is missing")
        if json_fingerprint(read_jsonl(schedule_path)) != schedule_fingerprint:
            raise ValueError(f"{label} resume schedule content mismatch")
        completed = status.get("completed_schedule_entries")
        if type(completed) is not int or not 0 <= completed <= len(schedule):
            raise ValueError(f"{label} resume completed count is invalid")
        if type(status.get("complete")) is not bool:
            raise ValueError(f"{label} resume complete flag is invalid")
        if status["complete"]:
            if report_filename is None:
                raise ValueError(f"{label} completed output has no report contract")
            completed_report = load_completed_report(
                root,
                status_filename=status_filename,
                report_filename=report_filename,
                status_schema=status_schema,
                report_schema=report_schema,
                config_path=config,
            )
        else:
            completed_report = None
        return PreparedOutput(base_status, status, True, completed_report)

    write_jsonl(schedule_path, schedule)
    status = {
        **base_status,
        "completed_schedule_entries": 0,
        "complete": False,
    }
    write_json(status_path, status)
    return PreparedOutput(base_status, status, False, None)


__all__ = [
    "PreparedOutput",
    "RESUMABLE_RUN_IDENTITY_SCHEMA",
    "RUNNER_CONFIG_SCHEMA",
    "load_completed_report",
    "prepare_resumable_output",
    "prepare_run_output",
    "runner_identity_fingerprint",
]
