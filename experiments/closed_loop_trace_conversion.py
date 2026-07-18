from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from experiments.closed_loop_confirmation import (
    CLOSED_LOOP_SCHEMA,
    validate_closed_loop_trace,
)
from experiments.closed_loop_trace_storage import (
    TRACE_FORMAT_DELTA_GZIP_V2,
    TRACE_FORMAT_FULL_V1,
    convert_v1_trace,
    partial_trace_path,
    storage_fingerprint,
    trace_file_metadata,
)
from experiments.repair_collection import (
    SCHEMA_VERSION,
    _read_json,
    _read_jsonl,
    _utc_now,
    _write_json,
    _write_jsonl,
)


CONVERSION_SCHEMA = "lns2.closed_loop_trace_conversion.v1"


def _safe_trace_path(root: Path, reference: str) -> Path:
    path = (root / reference).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"trace escapes collection root: {reference}") from error
    return path


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _copy_metadata(source: Path, output: Path) -> None:
    for name in (
        "qualification_manifest.jsonl",
        "qualification_report.json",
    ):
        source_path = source / name
        if source_path.is_file():
            destination = output / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)


def _converted_manifest_row(
    source_root: Path,
    output_root: Path,
    row: dict[str, Any],
    run_fingerprint: str,
    *,
    resume: bool,
    metric_iteration_budget: int,
) -> tuple[dict[str, Any], bool]:
    if str(row.get("status")) not in {"ok", "resumed"}:
        return {
            **row,
            "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
            "storage_fingerprint": storage_fingerprint(TRACE_FORMAT_DELTA_GZIP_V2),
        }, False
    source_reference = str(row["trace_file"])
    source_trace = _safe_trace_path(source_root, source_reference)
    destination = (
        output_root
        / "episodes"
        / str(row["split"])
        / str(row["policy"])
        / f"{row['episode_id']}.jsonl.gz"
    )
    resumed = False
    validated: dict[str, Any] | None = None
    if resume and destination.is_file():
        try:
            validated = validate_closed_loop_trace(
                destination,
                run_fingerprint,
                expected_episode_id=str(row["episode_id"]),
                expected_policy=str(row["policy"]),
                expected_solver_seed=int(row["solver_seed"]),
                metric_iteration_budget=metric_iteration_budget,
                collection_root=output_root,
            )
            if validated["summary"] != row.get("summary"):
                validated = None
        except ValueError:
            validated = None
        resumed = validated is not None
    if validated is None:
        partial = partial_trace_path(destination)
        metadata = convert_v1_trace(source_trace, partial, output_root)
        validated = validate_closed_loop_trace(
            partial,
            run_fingerprint,
            expected_episode_id=str(row["episode_id"]),
            expected_policy=str(row["policy"]),
            expected_solver_seed=int(row["solver_seed"]),
            metric_iteration_budget=metric_iteration_budget,
            collection_root=output_root,
        )
        if validated["summary"] != row.get("summary"):
            raise ValueError(f"converted summary mismatch: {row['episode_id']}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(partial, destination)
    else:
        metadata = {
            **trace_file_metadata(destination),
            "trace_event_count": int(validated["event_count"]),
            "initial_state_ref": validated.get("initial_state_ref"),
        }
    return (
        {
            **row,
            "trace_file": destination.relative_to(output_root).as_posix(),
            "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
            "storage_fingerprint": storage_fingerprint(TRACE_FORMAT_DELTA_GZIP_V2),
            **metadata,
            "status": "resumed" if resumed else str(row.get("status", "ok")),
            "source_trace_file": source_reference,
        },
        resumed,
    )


def convert_closed_loop_collection(
    source: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
    minimum_storage_reduction: float = 0.9,
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    if source_root == output_root:
        raise ValueError("source and output collections must differ")
    try:
        output_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError("output collection cannot be inside the source collection")
    if not 0.0 <= minimum_storage_reduction < 1.0:
        raise ValueError("minimum storage reduction must be in [0, 1)")

    source_run = _read_json(source_root / "run_config.json")
    if str(source_run.get("schema")) != CLOSED_LOOP_SCHEMA:
        raise ValueError("source is not a closed-loop collection")
    source_format = str(source_run.get("trace_format", TRACE_FORMAT_FULL_V1))
    if source_format != TRACE_FORMAT_FULL_V1:
        raise ValueError("converter currently requires a full-v1 source collection")
    run_fingerprint = str(source_run["run_fingerprint"])
    configuration = dict(source_run["configuration"])
    policies = tuple(map(str, configuration.get("policies", [])))
    if not policies:
        policies = tuple(
            sorted(
                path.name.removesuffix("_manifest.jsonl")
                for path in source_root.glob("*_manifest.jsonl")
                if path.name != "qualification_manifest.jsonl"
            )
        )
    storage_fp = storage_fingerprint(TRACE_FORMAT_DELTA_GZIP_V2)
    converted_run = {
        **source_run,
        "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
        "storage_fingerprint": storage_fp,
        "converted_from": {
            "collection": str(source_root),
            "trace_format": source_format,
            "run_fingerprint": run_fingerprint,
        },
    }
    output_run_path = output_root / "run_config.json"
    if output_run_path.is_file():
        existing = _read_json(output_run_path)
        if (
            str(existing.get("run_fingerprint")) != run_fingerprint
            or str(existing.get("storage_fingerprint")) != storage_fp
        ):
            raise ValueError("output contains a different conversion")
        if not resume:
            raise ValueError("output already exists; pass resume to continue")
    elif output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("output directory is non-empty and has no conversion metadata")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_run_path, converted_run)
    _copy_metadata(source_root, output_root)

    converted_count = 0
    resumed_count = 0
    error_count = 0
    metric_budget = int(configuration["metric_iteration_budget"])
    manifest_counts: dict[str, int] = {}
    source_manifests = {
        policy: _read_jsonl(source_root / f"{policy}_manifest.jsonl")
        for policy in policies
    }
    total_episode_count = sum(
        str(row.get("status")) in {"ok", "resumed"}
        for rows in source_manifests.values()
        for row in rows
    )
    completed_episode_count = 0
    progress_path = output_root / "conversion_progress.json"
    _write_json(
        progress_path,
        {
            "schema": CONVERSION_SCHEMA,
            "status": "running",
            "total_episodes": total_episode_count,
            "completed_episodes": 0,
            "current_policy": None,
            "updated_at": _utc_now(),
        },
    )
    for policy in policies:
        rows = source_manifests[policy]
        converted_rows = []
        for row in rows:
            try:
                converted, was_resumed = _converted_manifest_row(
                    source_root,
                    output_root,
                    row,
                    run_fingerprint,
                    resume=resume,
                    metric_iteration_budget=metric_budget,
                )
            except Exception as error:
                error_count += 1
                converted = {
                    **row,
                    "trace_file": None,
                    "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
                    "storage_fingerprint": storage_fp,
                    "status": "error",
                    "summary": None,
                    "error_kind": type(error).__name__,
                    "error": f"{type(error).__name__}: {error}",
                }
                was_resumed = False
            else:
                if str(row.get("status")) in {"ok", "resumed"}:
                    converted_count += 1
                    resumed_count += int(was_resumed)
                    completed_episode_count += 1
            converted_rows.append(converted)
            _write_json(
                progress_path,
                {
                    "schema": CONVERSION_SCHEMA,
                    "status": "running",
                    "total_episodes": total_episode_count,
                    "completed_episodes": completed_episode_count,
                    "current_policy": policy,
                    "current_episode_id": row.get("episode_id"),
                    "error_count": error_count,
                    "updated_at": _utc_now(),
                },
            )
        _write_jsonl(output_root / f"{policy}_manifest.jsonl", converted_rows)
        manifest_counts[policy] = len(converted_rows)

    source_summary_path = source_root / "collection_summary.json"
    if source_summary_path.is_file():
        source_summary = _read_json(source_summary_path)
        _write_json(
            output_root / "collection_summary.json",
            {
                **source_summary,
                "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
                "storage_fingerprint": storage_fp,
            },
        )

    source_trace_bytes = sum(
        _safe_trace_path(source_root, str(row["trace_file"])).stat().st_size
        for policy in policies
        for row in source_manifests[policy]
        if str(row.get("status")) in {"ok", "resumed"} and row.get("trace_file")
    )
    compact_trace_bytes = _tree_bytes(output_root / "episodes")
    state_blob_bytes = _tree_bytes(output_root / "state_blobs")
    compact_bytes = compact_trace_bytes + state_blob_bytes
    reduction = (
        1.0 - compact_bytes / source_trace_bytes if source_trace_bytes else 0.0
    )
    passed = error_count == 0 and reduction >= minimum_storage_reduction
    report = {
        "schema": CONVERSION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source": str(source_root),
        "output": str(output_root),
        "run_fingerprint": run_fingerprint,
        "source_trace_format": source_format,
        "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
        "storage_fingerprint": storage_fp,
        "policies": list(policies),
        "manifest_counts": manifest_counts,
        "converted_episode_count": converted_count,
        "resumed_episode_count": resumed_count,
        "error_count": error_count,
        "storage": {
            "source_trace_bytes": source_trace_bytes,
            "compact_trace_bytes": compact_trace_bytes,
            "state_blob_bytes": state_blob_bytes,
            "compact_total_bytes": compact_bytes,
            "reduction_fraction": reduction,
            "minimum_reduction_fraction": minimum_storage_reduction,
        },
        "passed": passed,
        "cleanup": {
            "deletion_authorized": False,
            "candidate": str(source_root / "episodes"),
            "candidate_bytes": source_trace_bytes,
            "require_equivalence_report": True,
        },
    }
    _write_json(output_root / "storage_conversion_report.json", report)
    _write_json(
        progress_path,
        {
            "schema": CONVERSION_SCHEMA,
            "status": "complete" if passed else "error",
            "total_episodes": total_episode_count,
            "completed_episodes": completed_episode_count,
            "current_policy": None,
            "error_count": error_count,
            "updated_at": _utc_now(),
        },
    )
    return report


__all__ = ["convert_closed_loop_collection"]
