"""Shared implementation details for experiment and audit modules.

This module intentionally contains only semantics-free helpers. Study-specific
Pareto definitions, bootstrap procedures, labels, and acceptance gates remain
in their owning experiment modules so historical results stay reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


def mean(values: Iterable[float | int | bool]) -> float:
    numbers = [float(value) for value in values]
    return statistics.fmean(numbers) if numbers else 0.0


def population_std(values: Iterable[float | int | bool]) -> float:
    numbers = [float(value) for value in values]
    return statistics.pstdev(numbers) if len(numbers) > 1 else 0.0


def ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def quantile(values: Iterable[float | int], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def relative_improvement(baseline: float, challenger: float) -> float:
    if baseline == 0.0:
        return 0.0 if challenger == 0.0 else -float("inf")
    return (baseline - challenger) / baseline


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_collection_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing collection file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_optional_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl_fsync(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def resolve_within(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"collection path escapes its root: {relative}") from error
    return path


def resolve_cli_path(project_root: Path, value: str | Path) -> Path:
    """Resolve a CLI path without requiring the target to remain inside the repo."""

    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def add_categorical_feature(
    features: dict[str, float], prefix: str, value: Any
) -> None:
    normalized = str(value or "unknown").strip().lower().replace(" ", "_")
    features[f"{prefix}={normalized}"] = 1.0


def feature_names(rows: Iterable[dict[str, Any]], profile: str) -> list[str]:
    return sorted(
        {
            name
            for row in rows
            for name in row["features"][profile]
        }
    )


def state_storage_id(state_id: str) -> str:
    payload = json.dumps(
        {"state_id": state_id},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"state-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]}"


def trial_job_id(state_id: str, candidate_id: str, trial_index: int) -> str:
    return (
        f"{state_storage_id(state_id)}__{candidate_id}"
        f"__trial_{trial_index:04d}"
    )


def episode_id(row: dict[str, Any], solver_seed: int, policy: str) -> str:
    return f"{row['task_id']}__seed_{solver_seed:04d}__{policy}"


def action_family(action: dict[str, Any]) -> str:
    return f"{action['heuristic']}:{int(action['neighborhood_size'])}"


def select_rows_by_task_id(
    rows: list[dict[str, Any]], task_ids: list[str] | None
) -> list[dict[str, Any]]:
    if task_ids is None:
        return rows
    requested = list(dict.fromkeys(map(str, task_ids)))
    indexed = {str(row["task_id"]): row for row in rows}
    missing = sorted(set(requested) - set(indexed))
    if missing:
        raise ValueError(f"unknown task ids: {missing}")
    return [indexed[task_id] for task_id in requested]


__all__ = [
    "action_family",
    "add_categorical_feature",
    "append_jsonl_fsync",
    "episode_id",
    "feature_names",
    "mean",
    "population_std",
    "quantile",
    "ratio",
    "read_collection_jsonl",
    "read_json",
    "read_jsonl",
    "read_optional_jsonl",
    "relative_improvement",
    "resolve_within",
    "sha256_file",
    "select_rows_by_task_id",
    "state_storage_id",
    "trial_job_id",
    "write_json",
    "write_jsonl",
]
