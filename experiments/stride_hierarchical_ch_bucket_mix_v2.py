from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Collection, Iterable, Sequence


ALGORITHM_SCHEMA = "lns2.stride.hierarchical_ch_bucket_mix.v2"
TASK_SCHEMA = "lns2.stride.hierarchical_ch_bucket_mix_task.v2"
VARIANT_IDS = ("bucket_mix_00", "bucket_mix_01")
_VARIANT_SALTS = {
    "bucket_mix_00": "stride-hierarchical-ch-source-v2/bucket-mix/variant-00",
    "bucket_mix_01": "stride-hierarchical-ch-source-v2/bucket-mix/variant-01",
}


@dataclass(frozen=True)
class ScenarioRow:
    source_line_number: int
    bucket: int
    map_name: str
    width: int
    height: int
    start: tuple[int, int]
    goal: tuple[int, int]
    distance: float
    distance_token: str

    @property
    def canonical_line(self) -> str:
        return "\t".join(
            (
                str(self.bucket),
                self.map_name,
                str(self.width),
                str(self.height),
                str(self.start[0]),
                str(self.start[1]),
                str(self.goal[0]),
                str(self.goal[1]),
                self.distance_token,
            )
        )


@dataclass(frozen=True)
class PrefixTask:
    task_id: str
    variant_id: str
    agent_count: int
    rows: tuple[ScenarioRow, ...]
    scenario_payload: bytes
    scenario_sha256: str
    selection_sha256: str


@dataclass(frozen=True)
class BucketMixVariant:
    variant_id: str
    unique_capacity: int
    accepted_rows: tuple[ScenarioRow, ...]
    tasks: tuple[PrefixTask, ...]

    def task_for_load(self, agent_count: int) -> PrefixTask:
        for task in self.tasks:
            if task.agent_count == agent_count:
                return task
        raise KeyError(agent_count)


@dataclass(frozen=True)
class BucketMixBuild:
    map_id: str
    source_scenario_sha256: str
    source_row_count: int
    bucket_count: int
    registered_loads: tuple[int, ...]
    variants: tuple[BucketMixVariant, ...]

    @property
    def capacity_by_variant(self) -> dict[str, int]:
        return {variant.variant_id: variant.unique_capacity for variant in self.variants}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _validate_hash(value: str, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} must be a lowercase-compatible SHA256 digest")
    return normalized


def _registered_loads(values: Sequence[int]) -> tuple[int, ...]:
    loads = tuple(int(value) for value in values)
    if not loads or any(value <= 0 for value in loads):
        raise ValueError("registered loads must be non-empty positive integers")
    if tuple(sorted(set(loads))) != loads:
        raise ValueError("registered loads must be unique and strictly increasing")
    return loads


def _authorize_map(
    map_id: str,
    *,
    allowed_map_ids: Collection[str],
    sealed_map_ids: Collection[str],
) -> None:
    allowed = set(allowed_map_ids)
    sealed = set(sealed_map_ids)
    if map_id in sealed:
        raise ValueError(f"sealed-final raw scenario access is not authorized: {map_id}")
    if map_id not in allowed:
        raise ValueError(f"map is not registered for train/development source use: {map_id}")
    if allowed & sealed:
        raise ValueError("allowed and sealed-final map registrations overlap")


def parse_official_bucket_scenario(
    payload: bytes,
    *,
    map_id: str,
) -> tuple[str, tuple[ScenarioRow, ...]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError("MovingAI scenario is not valid UTF-8") from error
    nonempty = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
    if not nonempty or not nonempty[0][1].strip().lower().startswith("version "):
        raise ValueError("MovingAI scenario version header is missing")
    header = nonempty[0][1].strip()
    rows: list[ScenarioRow] = []
    dimensions: set[tuple[int, int]] = set()
    for line_number, line in nonempty[1:]:
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f"invalid MovingAI scenario row at line {line_number}")
        try:
            bucket = int(fields[0])
            width, height = int(fields[2]), int(fields[3])
            start = (int(fields[4]), int(fields[5]))
            goal = (int(fields[6]), int(fields[7]))
            distance = float(fields[8])
        except ValueError as error:
            raise ValueError(f"invalid MovingAI scenario value at line {line_number}") from error
        if bucket < 0 or width <= 0 or height <= 0:
            raise ValueError(f"invalid bucket or dimensions at line {line_number}")
        if fields[1] != f"{map_id}.map":
            raise ValueError(f"scenario row names a different map at line {line_number}")
        if (
            min(*start, *goal) < 0
            or start[0] >= width
            or goal[0] >= width
            or start[1] >= height
            or goal[1] >= height
        ):
            raise ValueError(f"scenario coordinate is out of bounds at line {line_number}")
        if not math.isfinite(distance) or distance < 0.0:
            raise ValueError(f"invalid scenario distance at line {line_number}")
        dimensions.add((width, height))
        rows.append(
            ScenarioRow(
                source_line_number=line_number,
                bucket=bucket,
                map_name=fields[1],
                width=width,
                height=height,
                start=start,
                goal=goal,
                distance=distance,
                distance_token=fields[8],
            )
        )
    if not rows:
        raise ValueError("MovingAI scenario has no task rows")
    if len(dimensions) != 1:
        raise ValueError("MovingAI scenario rows disagree on map dimensions")
    return header, tuple(rows)


def _stable_row_key(row: ScenarioRow, *, map_id: str, variant_id: str) -> tuple[str, int]:
    preimage = "\0".join(
        (
            ALGORITHM_SCHEMA,
            _VARIANT_SALTS[variant_id],
            map_id,
            str(row.source_line_number),
            row.canonical_line,
        )
    ).encode("utf-8")
    return hashlib.sha256(preimage).hexdigest(), row.source_line_number


def _bucket_round_robin(
    rows: Iterable[ScenarioRow],
    *,
    map_id: str,
    variant_id: str,
) -> tuple[ScenarioRow, ...]:
    by_bucket: dict[int, list[ScenarioRow]] = defaultdict(list)
    for row in rows:
        by_bucket[row.bucket].append(row)
    ordered = {
        bucket: sorted(
            bucket_rows,
            key=lambda row: _stable_row_key(row, map_id=map_id, variant_id=variant_id),
        )
        for bucket, bucket_rows in by_bucket.items()
    }
    result: list[ScenarioRow] = []
    for offset in range(max(len(bucket_rows) for bucket_rows in ordered.values())):
        for bucket in sorted(ordered):
            if offset < len(ordered[bucket]):
                result.append(ordered[bucket][offset])
    return tuple(result)


def _greedy_unique(rows: Iterable[ScenarioRow]) -> tuple[ScenarioRow, ...]:
    starts: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    accepted: list[ScenarioRow] = []
    for row in rows:
        if row.start in starts or row.goal in goals:
            continue
        starts.add(row.start)
        goals.add(row.goal)
        accepted.append(row)
    return tuple(accepted)


def _scenario_payload(header: str, rows: Sequence[ScenarioRow]) -> bytes:
    return (header + "\n" + "".join(f"{row.canonical_line}\n" for row in rows)).encode(
        "utf-8"
    )


def _selection_sha256(
    *,
    map_id: str,
    variant_id: str,
    source_sha256: str,
    rows: Sequence[ScenarioRow],
) -> str:
    identity = {
        "schema": ALGORITHM_SCHEMA,
        "map_id": map_id,
        "variant_id": variant_id,
        "source_scenario_sha256": source_sha256,
        "ordered_source_line_numbers": [row.source_line_number for row in rows],
        "ordered_canonical_rows": [row.canonical_line for row in rows],
    }
    return _sha256_bytes(_json_bytes(identity))


def build_bucket_mix_variants_from_bytes(
    payload: bytes,
    *,
    expected_source_sha256: str,
    map_id: str,
    registered_loads: Sequence[int],
) -> BucketMixBuild:
    expected_hash = _validate_hash(expected_source_sha256, "expected source scenario hash")
    observed_hash = _sha256_bytes(payload)
    if observed_hash != expected_hash:
        raise ValueError(f"registered scenario full-file SHA256 mismatch: {map_id}")
    loads = _registered_loads(registered_loads)
    header, source_rows = parse_official_bucket_scenario(payload, map_id=map_id)
    variants: list[BucketMixVariant] = []
    for variant_id in VARIANT_IDS:
        round_robin = _bucket_round_robin(source_rows, map_id=map_id, variant_id=variant_id)
        accepted = _greedy_unique(round_robin)
        if len(accepted) < loads[-1]:
            raise ValueError(
                f"bucket-mix unique capacity is below registered load for "
                f"{map_id}/{variant_id}: {len(accepted)} < {loads[-1]}"
            )
        tasks: list[PrefixTask] = []
        for agent_count in loads:
            prefix = accepted[:agent_count]
            scenario_payload = _scenario_payload(header, prefix)
            tasks.append(
                PrefixTask(
                    task_id=f"{map_id}__{variant_id}__agents_{agent_count:04d}",
                    variant_id=variant_id,
                    agent_count=agent_count,
                    rows=prefix,
                    scenario_payload=scenario_payload,
                    scenario_sha256=_sha256_bytes(scenario_payload),
                    selection_sha256=_selection_sha256(
                        map_id=map_id,
                        variant_id=variant_id,
                        source_sha256=observed_hash,
                        rows=prefix,
                    ),
                )
            )
        variants.append(
            BucketMixVariant(
                variant_id=variant_id,
                unique_capacity=len(accepted),
                accepted_rows=accepted,
                tasks=tuple(tasks),
            )
        )
    return BucketMixBuild(
        map_id=map_id,
        source_scenario_sha256=observed_hash,
        source_row_count=len(source_rows),
        bucket_count=len({row.bucket for row in source_rows}),
        registered_loads=loads,
        variants=tuple(variants),
    )


def build_registered_bucket_mix_variants(
    scenario_path: str | Path,
    *,
    expected_source_sha256: str,
    map_id: str,
    registered_loads: Sequence[int],
    allowed_map_ids: Collection[str],
    sealed_map_ids: Collection[str],
) -> BucketMixBuild:
    # Authorization deliberately precedes every path operation and raw-file read.
    _authorize_map(map_id, allowed_map_ids=allowed_map_ids, sealed_map_ids=sealed_map_ids)
    path = Path(scenario_path)
    if path.name != f"{map_id}.map.scen":
        raise ValueError(f"registered scenario filename differs from map ID: {map_id}")
    payload = path.read_bytes()
    return build_bucket_mix_variants_from_bytes(
        payload,
        expected_source_sha256=expected_source_sha256,
        map_id=map_id,
        registered_loads=registered_loads,
    )


def _safe_relative_path(value: str, label: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value:
        raise ValueError(f"{label} must be a safe relative path")
    return path.as_posix()


def _write_exact(path: Path, payload: bytes, *, resume: bool) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing bucket-mix output differs: {path}")
        if not resume:
            raise FileExistsError(f"output already exists; use resume: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _task_payload(
    *,
    build: BucketMixBuild,
    variant: BucketMixVariant,
    task: PrefixTask,
    split: str,
) -> dict[str, Any]:
    distances = [row.distance for row in task.rows]
    return {
        "schema": TASK_SCHEMA,
        "algorithm_schema": ALGORITHM_SCHEMA,
        "task_semantics": "static MovingAI registered bucket-mixed scenario prefix",
        "split": split,
        "map_id": build.map_id,
        "task_id": task.task_id,
        "variant_id": variant.variant_id,
        "agent_count": task.agent_count,
        "source_scenario_sha256": build.source_scenario_sha256,
        "prefix_scenario_sha256": task.scenario_sha256,
        "selection_sha256": task.selection_sha256,
        "unique_capacity": variant.unique_capacity,
        "unique_start_count": len({row.start for row in task.rows}),
        "unique_goal_count": len({row.goal for row in task.rows}),
        "bucket_histogram": dict(sorted(Counter(row.bucket for row in task.rows).items())),
        "minimum_shortest_distance": min(distances),
        "maximum_shortest_distance": max(distances),
        "mean_shortest_distance": sum(distances) / len(distances),
        "ordered_source_line_numbers": [row.source_line_number for row in task.rows],
        "training_authorized": False,
        "outcome_filtering_used": False,
        "sealed_final_semantic_access": False,
    }


def materialize_registered_bucket_mix_map(
    scenario_path: str | Path,
    *,
    expected_source_sha256: str,
    map_id: str,
    registered_loads: Sequence[int],
    split: str,
    output_split_root: str | Path,
    allowed_map_ids: Collection[str],
    sealed_map_ids: Collection[str],
    map_file: str,
    map_sha256: str,
    layout_family: str,
    dry_run: bool = False,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Materialize two fixed scenario variants and exact registered-load prefixes.

    The caller owns map copying and whole-dataset manifests. Returned rows are ready
    to merge into such a manifest and deliberately contain no solver outcomes.
    """

    if split not in {"train", "development"}:
        raise ValueError(f"sealed-final or unregistered source split is not authorized: {split}")
    relative_map_file = _safe_relative_path(map_file, "map_file")
    pinned_map_hash = _validate_hash(map_sha256, "map hash")
    build = build_registered_bucket_mix_variants(
        scenario_path,
        expected_source_sha256=expected_source_sha256,
        map_id=map_id,
        registered_loads=registered_loads,
        allowed_map_ids=allowed_map_ids,
        sealed_map_ids=sealed_map_ids,
    )
    split_root = Path(output_split_root)
    rows: list[dict[str, Any]] = []
    for variant in build.variants:
        for task in variant.tasks:
            scenario_relative = f"scenarios/{task.task_id}.map.scen"
            task_relative = f"tasks/{task.task_id}.json"
            task_payload = _task_payload(build=build, variant=variant, task=task, split=split)
            task_bytes = _json_bytes(task_payload)
            if not dry_run:
                _write_exact(split_root / scenario_relative, task.scenario_payload, resume=resume)
                _write_exact(split_root / task_relative, task_bytes, resume=resume)
            rows.append(
                {
                    "split": split,
                    "source_group": "movingai_registered_bucket_mix_v2",
                    "map_id": map_id,
                    "task_id": task.task_id,
                    "agent_count": task.agent_count,
                    "map_file": relative_map_file,
                    "map_sha256": pinned_map_hash,
                    "scenario_file": scenario_relative,
                    "scenario_sha256": task.scenario_sha256,
                    "source_scenario_sha256": build.source_scenario_sha256,
                    "task_file": task_relative,
                    "task_sha256": _sha256_bytes(task_bytes),
                    "layout_mode": layout_family,
                    "layout_family": layout_family,
                    "layout_variant": map_id,
                    "scenario_type": variant.variant_id,
                    "task_variant": f"{variant.variant_id}_agents_{task.agent_count}",
                    "bucket_mix_variant": variant.variant_id,
                    "bucket_mix_algorithm_schema": ALGORITHM_SCHEMA,
                    "selection_sha256": task.selection_sha256,
                    "unique_capacity": variant.unique_capacity,
                    "unique_start_count": task.agent_count,
                    "unique_goal_count": task.agent_count,
                    "mean_shortest_distance": task_payload["mean_shortest_distance"],
                    "training_authorized": False,
                }
            )
    return rows


__all__ = [
    "ALGORITHM_SCHEMA",
    "TASK_SCHEMA",
    "VARIANT_IDS",
    "BucketMixBuild",
    "BucketMixVariant",
    "PrefixTask",
    "ScenarioRow",
    "build_bucket_mix_variants_from_bytes",
    "build_registered_bucket_mix_variants",
    "materialize_registered_bucket_mix_map",
    "parse_official_bucket_scenario",
]
