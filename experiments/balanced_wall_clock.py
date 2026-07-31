from __future__ import annotations

import collections
import csv
import hashlib
import heapq
import itertools
import json
import math
import os
import random
import shutil
import statistics
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    NATIVE_SEMANTICS_SCHEMA,
    PRODUCER_IDENTITY_SCHEMA,
    config_producer_fingerprint,
    sha256_file,
    validate_producer_identity,
)
from experiments.closed_loop_confirmation import (
    CONTROLLER_OPTIONAL_PACKAGES,
    CONTROLLER_REQUIRED_PACKAGES,
    WALL_CLOCK_SAFETY_MAX_DECISIONS,
    _controller_producer_identity,
    run_closed_loop_collection,
    validate_closed_loop_trace,
)
from experiments.closed_loop_trace_storage import read_state_blob, trace_file_metadata
from experiments.repair_collection import (
    _dataset_fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)
from experiments.state_analysis import summarize_initial_state_complexity


SPLIT = "balanced_wall_clock"
CONTROLLERS = ("official_adaptive", "v2-full", "mixed-full-v2")
CORRECTED_SCHEDULE_SCHEMA = "lns2.controller_execution_schedule.corrected_native_v1"
CORRECTED_REBIND_SCHEMA = "lns2.corrected_native_schedule_rebind.v1"
CORRECTED_SEED_SCHEDULE_SCHEMA = (
    "lns2.controller_execution_schedule.corrected_native_seed_reselected_v1"
)
CORRECTED_SEED_SELECTION_SCHEMA = (
    "lns2.corrected_native_seed_schedule_selection.v1"
)
CORRECTED_FULL_POOL_SCHEDULE_SCHEMA = (
    "lns2.controller_execution_schedule.corrected_native_full_pool.v3"
)
CORRECTED_FULL_POOL_REPORT_SCHEMA = (
    "lns2.corrected_native_full_pool_selection.v3"
)
LEGACY_CORRECTED_FULL_POOL_SCHEDULE_SCHEMAS = {
    "lns2.controller_execution_schedule.corrected_native_full_pool.v1",
    "lns2.controller_execution_schedule.corrected_native_full_pool.v2",
}
CORRECTED_SEED_SELECTION_SALT = "corrected-native-seed-reselection-v1"
CORRECTED_SCHEDULE_SCHEMAS = {
    CORRECTED_SCHEDULE_SCHEMA: CORRECTED_REBIND_SCHEMA,
    CORRECTED_SEED_SCHEDULE_SCHEMA: CORRECTED_SEED_SELECTION_SCHEMA,
    CORRECTED_FULL_POOL_SCHEDULE_SCHEMA: CORRECTED_FULL_POOL_REPORT_SCHEMA,
}
STRATA = (("low", 1, 10), ("medium", 11, 100), ("high", 101, 500))
INITIAL_PP_LOAD_STRATA = (
    ("low", 0, 100_000),
    ("medium", 100_001, 1_000_000),
    ("high", 1_000_001, None),
)
DEFAULT_DIFFICULTY_CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "balanced_wall_clock_difficulty_audit.json"
)


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _analysis_producer_identity() -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    files = {
        relative: sha256_file(project_root / relative)
        for relative in (
            "experiments/balanced_wall_clock.py",
            "experiments/closed_loop_confirmation.py",
            "experiments/closed_loop_trace_storage.py",
            "experiments/repair_collection.py",
        )
    }
    return {
        "schema": "lns2.balanced_wall_clock_analysis_producer.v1",
        "files": files,
        "fingerprint": _fingerprint(files),
    }


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _is_exact_number(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) == expected
    )


def _native_identity_from_producer(
    producer: Any,
) -> dict[str, Any] | None:
    if not isinstance(producer, dict):
        return None
    native = producer.get("native")
    if isinstance(native, dict):
        return native
    native = producer.get("native_module")
    return native if isinstance(native, dict) else None


def _validate_corrected_producer_identity(
    producer: Any, *, label: str, allow_legacy: bool = True
) -> dict[str, Any]:
    if not isinstance(producer, dict):
        raise ValueError(f"{label} producer identity is missing")
    if producer.get("schema") == PRODUCER_IDENTITY_SCHEMA:
        try:
            return validate_producer_identity(
                producer,
                native_required=True,
                package_names=CONTROLLER_REQUIRED_PACKAGES,
                optional_package_names=CONTROLLER_OPTIONAL_PACKAGES,
            )
        except ValueError as error:
            raise ValueError(
                f"{label} structured producer identity is invalid"
            ) from error
    if not allow_legacy:
        raise ValueError(
            f"{label} must use structured producer identity "
            f"{PRODUCER_IDENTITY_SCHEMA}"
        )
    native = _native_identity_from_producer(producer)
    if (
        native is None
        or native.get("native_semantics_schema") != NATIVE_SEMANTICS_SCHEMA
        or not _is_sha256(native.get("sha256"))
        or not isinstance(native.get("repair_timing_schema"), str)
        or not native["repair_timing_schema"]
    ):
        raise ValueError(f"{label} corrected-native identity is incomplete")
    if "source_sha256" in producer:
        sources = producer.get("source_sha256")
        if (
            not isinstance(sources, dict)
            or not sources
            or any(
                not isinstance(name, str)
                or not name
                or not _is_sha256(digest)
                for name, digest in sources.items()
            )
        ):
            raise ValueError(f"{label} producer source identity is incomplete")
    else:
        files = producer.get("files")
        implementation_sha = producer.get("sha256")
        if (
            not isinstance(files, dict)
            or not files
            or any(
                not isinstance(name, str)
                or not name
                or not _is_sha256(digest)
                for name, digest in files.items()
            )
            or not _is_sha256(implementation_sha)
            or implementation_sha
            != _fingerprint({"files": files, "native_module": native})
        ):
            raise ValueError(f"{label} implementation identity is incomplete")
    return producer


def _validated_structured_producer_config(
    config: dict[str, Any], *, label: str
) -> tuple[dict[str, Any], str]:
    if "controller_implementation" in config:
        raise ValueError(
            f"{label} uses legacy controller_implementation; "
            f"{PRODUCER_IDENTITY_SCHEMA} is required"
        )
    producer = _validate_corrected_producer_identity(
        config.get("producer_identity"),
        label=label,
        allow_legacy=False,
    )
    try:
        fingerprint = config_producer_fingerprint(
            config,
            label=label,
            native_required=True,
            package_names=CONTROLLER_REQUIRED_PACKAGES,
            optional_package_names=CONTROLLER_OPTIONAL_PACKAGES,
        )
    except ValueError as error:
        raise ValueError(f"{label} producer fingerprint is invalid") from error
    return producer, fingerprint


def _validate_corrected_full_pool_collection_preflight(
    config_path: str | Path,
    provenance: dict[str, Any],
) -> None:
    """Reject an invalid formal lane before any reset or repair job starts."""

    contract = provenance.get("stopping_contract")
    expected_contract = {
        "stopping_rule": "wall-clock-fixed-metric",
        "max_decisions": 0,
        "max_repair_iterations": 0,
        "metric_iteration_budget": 100,
        "wall_time_budget_seconds": 600.0,
        "environment_time_limit_seconds": 600.0,
        "episode_process_timeout_seconds": 660.0,
        "safety_max_decisions": WALL_CLOCK_SAFETY_MAX_DECISIONS,
        "safety_limit_is_not_metric_cap": True,
    }
    if contract != expected_contract:
        raise ValueError("corrected full-pool stopping contract is invalid")
    config = _read_json(Path(config_path).resolve())
    environment = config.get("environment")
    if (
        config.get("stopping_rule") != contract.get("stopping_rule")
        or type(config.get("max_decisions")) is not int
        or config["max_decisions"] != contract.get("max_decisions")
        or type(config.get("metric_iteration_budget")) is not int
        or config["metric_iteration_budget"]
        != contract.get("metric_iteration_budget")
        or not isinstance(environment, dict)
        or type(environment.get("max_repair_iterations")) is not int
        or environment["max_repair_iterations"]
        != contract.get("max_repair_iterations")
        or not _is_exact_number(
            config.get("wall_time_budget_seconds"),
            600.0,
        )
        or not _is_exact_number(
            environment.get("time_limit"),
            600.0,
        )
        or not _is_exact_number(
            config.get("episode_process_timeout_seconds"),
            660.0,
        )
        or contract.get("safety_max_decisions")
        != WALL_CLOCK_SAFETY_MAX_DECISIONS
        or contract.get("safety_limit_is_not_metric_cap") is not True
    ):
        raise ValueError(
            "corrected full-pool config differs from the frozen "
            "0/0/100/600/600/660 stopping contract"
        )

    scheduled_producer = _validate_corrected_producer_identity(
        provenance.get("qualification_producer_identity"),
        label="corrected full-pool schedule",
        allow_legacy=False,
    )
    scheduled_fingerprint = provenance.get(
        "qualification_producer_identity_fingerprint"
    )
    if (
        not _is_sha256(scheduled_fingerprint)
        or scheduled_fingerprint != _fingerprint(scheduled_producer)
    ):
        raise ValueError(
            "corrected full-pool schedule producer fingerprint is invalid"
        )
    current_producer = _controller_producer_identity(
        Path(__file__).resolve().parents[1]
    )
    validated_current = _validate_corrected_producer_identity(
        current_producer,
        label="current corrected full-pool producer",
        allow_legacy=False,
    )
    if (
        validated_current != scheduled_producer
        or _fingerprint(validated_current) != scheduled_fingerprint
    ):
        raise ValueError(
            "current package/source/native producer identity differs from "
            "the corrected full-pool qualification"
        )


def _corrected_schedule_report(
    schedule_root: Path,
    schedule_path: Path,
    schedule: dict[str, Any],
    *,
    dataset: str | Path | None = None,
) -> dict[str, Any] | None:
    expected_report_schema = CORRECTED_SCHEDULE_SCHEMAS.get(schedule.get("schema"))
    if expected_report_schema is None:
        return None
    if schedule.get("schema") == CORRECTED_FULL_POOL_SCHEDULE_SCHEMA:
        from experiments.corrected_native_full_pool import (
            validate_corrected_native_full_pool_schedule,
        )

        validated = validate_corrected_native_full_pool_schedule(
            schedule_root,
            dataset=dataset,
        )
        return dict(validated["report"])
    report_path = schedule_root / "cohort_report.json"
    if not report_path.is_file():
        raise ValueError("corrected-native schedule cohort report is missing")
    report = _read_json(report_path)
    if (
        report.get("schema") != expected_report_schema
        or report.get("passed") is not True
        or report.get("formal_collection_allowed") is not True
        or report.get("execution_schedule_sha256") != sha256_file(schedule_path)
        or report.get("provenance") != schedule.get("provenance")
    ):
        raise ValueError(
            "corrected-native cohort report does not bind its execution schedule"
        )
    gates = report.get("gates")
    if (
        not isinstance(gates, dict)
        or not gates
        or any(type(value) is not bool or value is not True for value in gates.values())
    ):
        raise ValueError("corrected-native cohort report has invalid formal gates")
    return report


def _validate_corrected_seed_schedule_semantics(
    schedule: dict[str, Any],
    qualification_rows: list[dict[str, Any]],
) -> None:
    if schedule.get("schema") != CORRECTED_SEED_SCHEDULE_SCHEMA:
        return
    entries = schedule.get("entries")
    provenance = schedule.get("provenance")
    if (
        not isinstance(entries, list)
        or len(entries) != 36
        or not isinstance(provenance, dict)
    ):
        raise ValueError("corrected seed schedule structure is invalid")
    selector = provenance.get("selector_identity")
    if (
        not isinstance(selector, dict)
        or selector.get("algorithm_schema")
        != "lns2.corrected_native_seed_assignment.v1"
        or selector.get("tie_break_salt") != CORRECTED_SEED_SELECTION_SALT
        or selector.get("implementation_sha256")
        != sha256_file(Path(__file__).resolve())
    ):
        raise ValueError("corrected seed schedule selector identity differs")
    selection = provenance.get("selection")
    constraints = (
        selection.get("constraints")
        if isinstance(selection, dict)
        else None
    )
    quotas = (
        constraints.get("registered_cell_source_quotas")
        if isinstance(constraints, dict)
        else None
    )
    if not isinstance(quotas, dict):
        raise ValueError("corrected seed schedule lacks registered cell quotas")
    qualification_index = {
        _episode_key(row): row for row in qualification_rows
    }
    if len(qualification_index) != len(qualification_rows):
        raise ValueError("corrected seed qualification contains duplicate keys")
    keys: set[tuple[str, int]] = set()
    maps: set[str] = set()
    cell_source_counts: collections.Counter[
        tuple[str, str, str]
    ] = collections.Counter()
    group_counts: collections.Counter[int] = collections.Counter()
    group_orders: dict[int, set[tuple[str, ...]]] = collections.defaultdict(set)
    for value in entries:
        if not isinstance(value, dict):
            raise ValueError("corrected seed schedule entry is not an object")
        key = _episode_key(value)
        if key in keys:
            raise ValueError("corrected seed schedule repeats a task/seed key")
        if any(existing[0] == key[0] for existing in keys):
            raise ValueError("corrected seed schedule repeats a task")
        keys.add(key)
        qualified = qualification_index.get(key)
        if qualified is None:
            raise ValueError("corrected seed schedule key lacks qualification")
        for field in (
            "map_id",
            "agent_count",
            "layout_mode",
            "initial_conflicts",
            "state_fingerprint",
        ):
            if value.get(field) != qualified.get(field):
                raise ValueError(
                    f"corrected seed schedule {field} differs from qualification"
                )
        complexity = qualified.get("initial_complexity")
        if (
            not isinstance(complexity, dict)
            or value.get("initial_low_level_generated")
            != complexity.get("initial_low_level_generated")
        ):
            raise ValueError(
                "corrected seed schedule PP load differs from qualification"
            )
        conflict_name = value.get("conflict_stratum")
        load_name = value.get("initial_pp_load_stratum")
        source = value.get("source_group")
        if (
            conflict_stratum(int(value["initial_conflicts"]))
            != conflict_name
            or initial_pp_load_stratum(
                int(value["initial_low_level_generated"])
            )
            != load_name
            or source not in {"generated", "movingai"}
        ):
            raise ValueError("corrected seed schedule strata are inconsistent")
        maps.add(str(value.get("map_id")))
        cell_source_counts[
            (str(conflict_name), str(load_name), str(source))
        ] += 1
        group = value.get("schedule_group")
        order = value.get("controller_order")
        if (
            type(group) is not int
            or group not in range(6)
            or not isinstance(order, list)
            or set(order) != set(CONTROLLERS)
            or len(order) != len(CONTROLLERS)
        ):
            raise ValueError("corrected seed schedule order is invalid")
        group_counts[group] += 1
        group_orders[group].add(tuple(map(str, order)))
    observed_quotas = {
        f"{conflict_name}__{load_name}": {
            source: cell_source_counts[
                (conflict_name, load_name, source)
            ]
            for source in ("generated", "movingai")
        }
        for conflict_name, _lower, _upper in STRATA
        for load_name, _load_lower, _load_upper in INITIAL_PP_LOAD_STRATA
    }
    if (
        observed_quotas != quotas
        or any(sum(counts.values()) != 4 for counts in observed_quotas.values())
        or len(maps) != 36
        or group_counts != collections.Counter({group: 6 for group in range(6)})
        or any(len(group_orders[group]) != 1 for group in range(6))
        or len({next(iter(group_orders[group])) for group in range(6)}) != 6
    ):
        raise ValueError("corrected seed schedule formal invariants differ")


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    partial.replace(path)


def _map_metrics(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers: dict[str, str] = {}
    marker = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "map":
            marker = index
            break
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            headers[parts[0].lower()] = parts[1]
    if marker is None or "height" not in headers or "width" not in headers:
        raise ValueError(f"invalid MovingAI map: {path}")
    rows, cols = int(headers["height"]), int(headers["width"])
    grid = lines[marker + 1 : marker + 1 + rows]
    if len(grid) != rows or any(len(row) != cols for row in grid):
        raise ValueError(f"MovingAI map dimensions differ from header: {path}")
    free = {
        row * cols + col
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value in {".", "G", "S"}
    }
    if not free:
        raise ValueError(f"MovingAI map has no free cells: {path}")

    def degree(cell: int) -> int:
        row, col = divmod(cell, cols)
        neighbors = []
        if row:
            neighbors.append((row - 1) * cols + col)
        if row + 1 < rows:
            neighbors.append((row + 1) * cols + col)
        if col:
            neighbors.append(row * cols + col - 1)
        if col + 1 < cols:
            neighbors.append(row * cols + col + 1)
        return sum(value in free for value in neighbors)

    degrees = [degree(cell) for cell in free]
    return {
        "rows": rows,
        "cols": cols,
        "free_cell_count": len(free),
        "obstacle_count": rows * cols - len(free),
        "obstacle_ratio": (rows * cols - len(free)) / (rows * cols),
        "average_free_degree": statistics.fmean(degrees),
        "minimum_free_degree": min(degrees),
        "maximum_free_degree": max(degrees),
        "dead_end_cell_count": sum(value <= 1 for value in degrees),
        "low_degree_cell_ratio": sum(value <= 2 for value in degrees) / len(degrees),
    }


def _movingai_passable_cells(path: Path) -> tuple[int, int, list[str], set[tuple[int, int]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers: dict[str, str] = {}
    marker = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "map":
            marker = index
            break
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            headers[parts[0].lower()] = parts[1]
    if marker is None or "height" not in headers or "width" not in headers:
        raise ValueError(f"invalid MovingAI map: {path}")
    rows, cols = int(headers["height"]), int(headers["width"])
    grid = lines[marker + 1 : marker + 1 + rows]
    if len(grid) != rows or any(len(row) != cols for row in grid):
        raise ValueError(f"MovingAI map dimensions differ from header: {path}")
    passable = {
        (row, col)
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value in {".", "G", "S"}
    }
    if not passable:
        raise ValueError(f"MovingAI map has no passable cells: {path}")
    return rows, cols, grid, passable


def _largest_four_connected_component(
    passable: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    remaining = set(passable)
    components: list[list[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = [start]
        queue: collections.deque[tuple[int, int]] = collections.deque([start])
        while queue:
            row, col = queue.popleft()
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        components.append(component)
    components.sort(key=lambda values: (-len(values), min(values)))
    return sorted(components[0])


def _derived_endpoint_seed(
    master_seed: int,
    map_id: str,
    task_seed: int,
    variant: str,
    agent_count: int,
) -> int:
    digest = hashlib.sha256(
        json.dumps(
            [master_seed, map_id, task_seed, variant, agent_count],
            separators=(",", ":"),
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _derived_endpoints(
    component: list[tuple[int, int]],
    agent_count: int,
    variant: str,
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    if agent_count <= 0 or agent_count > len(component):
        raise ValueError("derived task agent count exceeds component capacity")
    rng = random.Random(seed)
    if variant == "uniform_random":
        starts = rng.sample(component, agent_count)
        for _ in range(1_000):
            goals = rng.sample(component, agent_count)
            if all(start != goal for start, goal in zip(starts, goals)):
                return starts, goals
        raise ValueError("unable to derive a fixed-point-free uniform task")
    if variant == "opposite_exchange":
        sampled = rng.sample(component, agent_count)
        row_span = max(row for row, _col in sampled) - min(row for row, _col in sampled)
        col_span = max(col for _row, col in sampled) - min(col for _row, col in sampled)
        axis = 0 if row_span >= col_span else 1
        ordered = sorted(sampled, key=lambda cell: (cell[axis], cell[1 - axis]))
        goals = list(reversed(ordered))
        for offset in range(len(goals)):
            rotated = goals[offset:] + goals[:offset]
            if all(start != goal for start, goal in zip(ordered, rotated)):
                pairs = list(zip(ordered, rotated))
                rng.shuffle(pairs)
                return [start for start, _goal in pairs], [goal for _start, goal in pairs]
        raise ValueError("unable to derive a fixed-point-free exchange task")
    raise ValueError(f"unsupported map-derived task variant: {variant}")


def _four_neighbor_distances(
    passable: set[tuple[int, int]],
    starts: list[tuple[int, int]],
    goals: list[tuple[int, int]],
) -> list[int]:
    result = []
    for start, goal in zip(starts, goals):
        distances = {start: 0}
        queue = [
            (
                abs(start[0] - goal[0]) + abs(start[1] - goal[1]),
                0,
                start,
            )
        ]
        while queue:
            _estimate, distance, cell = heapq.heappop(queue)
            if distances.get(cell) != distance:
                continue
            if cell == goal:
                break
            row, col = cell
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                candidate = distance + 1
                if neighbor in passable and candidate < distances.get(
                    neighbor, candidate + 1
                ):
                    distances[neighbor] = candidate
                    heuristic = abs(neighbor[0] - goal[0]) + abs(
                        neighbor[1] - goal[1]
                    )
                    heapq.heappush(
                        queue, (candidate + heuristic, candidate, neighbor)
                    )
        if goal not in distances:
            raise ValueError(f"derived task has no four-neighbor path: {start} -> {goal}")
        result.append(distances[goal])
    return result


def _write_derived_scenario(
    path: Path,
    map_name: str,
    rows: int,
    cols: int,
    starts: list[tuple[int, int]],
    goals: list[tuple[int, int]],
    distances: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("version 1\n")
        for start, goal, distance in zip(starts, goals, distances):
            stream.write(
                "\t".join(
                    (
                        "0",
                        map_name,
                        str(cols),
                        str(rows),
                        str(start[1]),
                        str(start[0]),
                        str(goal[1]),
                        str(goal[0]),
                        str(distance),
                    )
                )
                + "\n"
            )
    partial.replace(path)


def prepare_movingai_map_derived_dataset(
    fetched: str | Path, source_config: str | Path, output: str | Path
) -> dict[str, Any]:
    """Create deterministic MAPF OD tasks on checksum-pinned MovingAI maps."""

    fetched_root = Path(fetched).resolve()
    config_path = Path(source_config).resolve()
    output_root = Path(output).resolve()
    config = _read_json(config_path)
    configuration_fingerprint = _fingerprint(config)
    summary_path = output_root / "dataset_summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("configuration_fingerprint") != configuration_fingerprint:
            raise ValueError("map-derived output belongs to a different configuration")
    elif output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("map-derived output is non-empty but has no summary")
    archive_spec = dict(config["map_archive"])
    archive_name = Path(str(archive_spec["url"])).name
    archive_path = fetched_root / "_archives" / archive_name
    if not archive_path.is_file():
        raise ValueError(f"MovingAI map archive is missing: {archive_path}")
    archive_sha = sha256_file(archive_path)
    if archive_sha != str(archive_spec["sha256"]):
        raise ValueError("MovingAI map archive SHA mismatch")

    task_seeds = [int(value) for value in config["task_seeds"]]
    variants = [str(value) for value in config["task_variants"]]
    if (
        not task_seeds
        or len(task_seeds) != len(set(task_seeds))
        or set(variants) != {"uniform_random", "opposite_exchange"}
    ):
        raise ValueError("map-derived task seeds or variants differ from registration")
    master_seed = int(config["master_seed"])
    split_root = output_root / SPLIT
    manifest: list[dict[str, Any]] = []
    map_ids: set[str] = set()
    with zipfile.ZipFile(archive_path) as bundle:
        for raw_case in config["benchmarks"]:
            case = dict(raw_case)
            map_id = str(case["id"])
            member = str(case["member"])
            if map_id in map_ids:
                raise ValueError(f"map-derived registration repeats map id: {map_id}")
            map_ids.add(map_id)
            map_path = split_root / "maps" / f"{map_id}.map"
            map_path.parent.mkdir(parents=True, exist_ok=True)
            partial = map_path.with_name(map_path.name + ".partial")
            try:
                with bundle.open(member) as source, partial.open("wb") as stream:
                    shutil.copyfileobj(source, stream)
            except KeyError as error:
                partial.unlink(missing_ok=True)
                raise ValueError(f"MovingAI map archive has no member: {member}") from error
            partial.replace(map_path)
            member_sha = sha256_file(map_path)
            if member_sha != str(case["member_sha256"]):
                raise ValueError(f"MovingAI map member SHA mismatch: {map_id}")

            rows, cols, _grid, passable = _movingai_passable_cells(map_path)
            component = _largest_four_connected_component(passable)
            metrics = _map_metrics(map_path)
            metadata_path = split_root / "maps" / f"{map_id}.json"
            _write_json(
                metadata_path,
                {
                    "schema_version": 1,
                    "benchmark_id": map_id,
                    "source": "MovingAI 2D grid benchmark",
                    "source_page": str(config["source"]),
                    "source_archive_url": str(archive_spec["url"]),
                    "source_archive_sha256": archive_sha,
                    "source_member": member,
                    "map_sha256": member_sha,
                    "largest_four_connected_component": len(component),
                    "topology_metrics": metrics,
                },
            )
            agent_counts = [int(value) for value in case["agent_counts"]]
            if (
                not agent_counts
                or len(agent_counts) != len(set(agent_counts))
                or any(value <= 0 or value > len(component) for value in agent_counts)
            ):
                raise ValueError(f"invalid map-derived agent counts: {map_id}")
            for task_seed in task_seeds:
                for variant in variants:
                    for agent_count in agent_counts:
                        endpoint_seed = _derived_endpoint_seed(
                            master_seed, map_id, task_seed, variant, agent_count
                        )
                        starts, goals = _derived_endpoints(
                            component, agent_count, variant, endpoint_seed
                        )
                        distances = _four_neighbor_distances(
                            passable, starts, goals
                        )
                        task_id = (
                            f"{map_id}__derived_{variant}__task_seed_{task_seed:04d}"
                            f"__agents_{agent_count:04d}"
                        )
                        scenario_path = split_root / "scenarios" / f"{task_id}.scen"
                        _write_derived_scenario(
                            scenario_path,
                            map_path.name,
                            rows,
                            cols,
                            starts,
                            goals,
                            distances,
                        )
                        task_path = split_root / "tasks" / f"{task_id}.json"
                        task_payload = {
                            "schema_version": 1,
                            "task_semantics_version": 1,
                            "task_semantics": (
                                "project-derived static MAPF OD on an official "
                                "MovingAI 2D benchmark map"
                            ),
                            "benchmark_id": map_id,
                            "source_map_sha256": member_sha,
                            "od_variant": variant,
                            "task_seed": task_seed,
                            "endpoint_seed": endpoint_seed,
                            "agent_count": agent_count,
                            "agent_density_largest_component": agent_count / len(component),
                            "unique_starts": len(set(starts)) == agent_count,
                            "unique_goals": len(set(goals)) == agent_count,
                            "fixed_point_count": sum(
                                start == goal for start, goal in zip(starts, goals)
                            ),
                            "distance_metric": "four_neighbor_unit",
                            "minimum_shortest_distance": min(distances),
                            "maximum_shortest_distance": max(distances),
                            "mean_shortest_distance": statistics.fmean(distances),
                            "scenario_sha256": sha256_file(scenario_path),
                        }
                        _write_json(task_path, task_payload)
                        manifest.append(
                            {
                                "split": SPLIT,
                                "source_group": "movingai",
                                "instance_origin": "movingai_map_project_derived_od",
                                "map_id": map_id,
                                "task_id": task_id,
                                "map_file": f"maps/{map_path.name}",
                                "scenario_file": f"scenarios/{scenario_path.name}",
                                "map_metadata_file": f"maps/{metadata_path.name}",
                                "task_file": f"tasks/{task_path.name}",
                                "layout_mode": str(case["layout_family"]),
                                "layout_variant": map_id,
                                "scenario_type": f"movingai_map_derived_{variant}",
                                "task_variant": (
                                    f"{variant}_seed_{task_seed}_agents_{agent_count}"
                                ),
                                "agent_count": agent_count,
                                "topology_metrics": metrics,
                                "dominant_flow_ratio": 0.0,
                                "hotspot_skew": 0.0,
                                "required_bottleneck_crossing_ratio": 0.0,
                                "mean_shortest_distance": task_payload[
                                    "mean_shortest_distance"
                                ],
                            }
                        )

    expected_maps = int(config["expected_map_count"])
    expected_instances = int(config["expected_instance_count"])
    if len(map_ids) != expected_maps or len(manifest) != expected_instances:
        raise ValueError("map-derived dataset dimensions differ from registration")
    manifest.sort(key=lambda row: str(row["task_id"]))
    _write_jsonl_atomic(split_root / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "dataset_revision": str(config["dataset_revision"]),
        "configuration_fingerprint": configuration_fingerprint,
        "source": (
            "MovingAI 2D benchmark maps with project-derived static MAPF OD tasks"
        ),
        "official_map_archive_sha256": archive_sha,
        "task_semantics": "derived_not_official_mapf_scenarios",
        "splits": {
            SPLIT: {
                "map_count": len(map_ids),
                "instance_count": len(manifest),
                "source_counts": {"movingai": len(manifest)},
            }
        },
    }
    _write_json(output_root / "dataset_summary.json", summary)
    return summary


def _scenario_metrics(path: Path, agent_count: int) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = lines[1:] if lines and lines[0].lower().startswith("version") else lines
    if len(rows) < agent_count:
        raise ValueError(f"scenario has fewer than {agent_count} agents: {path}")
    selected = [line.split() for line in rows[:agent_count]]
    if any(len(row) < 9 for row in selected):
        raise ValueError(f"invalid MovingAI scenario row: {path}")
    starts = [(int(row[4]), int(row[5])) for row in selected]
    goals = [(int(row[6]), int(row[7])) for row in selected]
    if len(starts) != len(set(starts)) or len(goals) != len(set(goals)):
        raise ValueError(f"scenario prefix repeats a start or goal: {path}")
    distances = [float(row[8]) for row in selected]
    return {
        "agent_count": agent_count,
        "mean_shortest_distance": statistics.fmean(distances),
        "minimum_shortest_distance": min(distances),
        "maximum_shortest_distance": max(distances),
    }


def prepare_movingai_dataset(
    fetched: str | Path, source_config: str | Path, output: str | Path
) -> dict[str, Any]:
    fetched_root = Path(fetched).resolve()
    config_path = Path(source_config).resolve()
    output_root = Path(output).resolve()
    config = _read_json(config_path)
    source_rows = _read_jsonl(fetched_root / "manifest.jsonl")
    source_index = {str(row["id"]): row for row in source_rows}
    case_index = {str(row["id"]): dict(row) for row in config["benchmarks"]}
    allow_fetched_superset = bool(config.get("allow_fetched_superset", False))
    registered_ids = set(case_index)
    fetched_ids = set(source_index)
    if (
        not registered_ids <= fetched_ids
        or not allow_fetched_superset
        and fetched_ids != registered_ids
    ):
        raise ValueError("fetched MovingAI manifest differs from registration")
    source_index = {map_id: source_index[map_id] for map_id in registered_ids}
    split_root = output_root / SPLIT
    manifest = []
    for map_id in sorted(source_index):
        source = source_index[map_id]
        case = case_index[map_id]
        source_map = fetched_root / str(source["map_file"])
        if sha256_file(source_map) != str(source["map_sha256"]):
            raise ValueError(f"MovingAI map SHA mismatch: {map_id}")
        map_path = split_root / "maps" / source_map.name
        map_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_map, map_path)
        metrics = _map_metrics(map_path)
        metadata_path = split_root / "maps" / f"{map_id}.json"
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "benchmark_id": map_id,
                "source": "MovingAI MAPF benchmark",
                "map_sha256": sha256_file(map_path),
                "topology_metrics": metrics,
            },
        )
        scenarios = {int(row["index"]): dict(row) for row in source["scenarios"]}
        for scenario_index in map(int, config["scenario_indices"]):
            scenario = scenarios[scenario_index]
            source_scenario = fetched_root / str(scenario["file"])
            if sha256_file(source_scenario) != str(scenario["sha256"]):
                raise ValueError(f"MovingAI scenario SHA mismatch: {map_id}/{scenario_index}")
            scenario_path = split_root / "scenarios" / source_scenario.name
            scenario_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_scenario, scenario_path)
            for agent_count in map(int, case["agent_counts"]):
                task_id = f"{map_id}__random_{scenario_index:02d}__agents_{agent_count:04d}"
                task_metrics = _scenario_metrics(scenario_path, agent_count)
                task_path = split_root / "tasks" / f"{task_id}.json"
                _write_json(
                    task_path,
                    {
                        "schema_version": 1,
                        "task_semantics": f"static MovingAI random-{scenario_index} prefix",
                        "benchmark_id": map_id,
                        "scenario_index": scenario_index,
                        "scenario_sha256": sha256_file(scenario_path),
                        **task_metrics,
                    },
                )
                manifest.append(
                    {
                        "split": SPLIT,
                        "source_group": "movingai",
                        "map_id": map_id,
                        "task_id": task_id,
                        "map_file": f"maps/{map_path.name}",
                        "scenario_file": f"scenarios/{scenario_path.name}",
                        "map_metadata_file": f"maps/{map_id}.json",
                        "task_file": f"tasks/{task_id}.json",
                        "layout_mode": str(case["layout_family"]),
                        "layout_variant": map_id,
                        "scenario_type": f"movingai_random_{scenario_index}",
                        "task_variant": f"random_{scenario_index}_agents_{agent_count}",
                        "agent_count": agent_count,
                        "topology_metrics": metrics,
                        "dominant_flow_ratio": 0.0,
                        "hotspot_skew": 0.0,
                        "required_bottleneck_crossing_ratio": 0.0,
                        "mean_shortest_distance": task_metrics["mean_shortest_distance"],
                    }
                )
    manifest.sort(key=lambda row: str(row["task_id"]))
    _write_jsonl_atomic(split_root / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "configuration_fingerprint": _fingerprint(
            {
                "source_config_sha256": sha256_file(config_path),
                "fetched_manifest_sha256": sha256_file(fetched_root / "manifest.jsonl"),
            }
        ),
        "source": "MovingAI MAPF benchmark random scenarios",
        "splits": {
            SPLIT: {
                "map_count": len(source_index),
                "instance_count": len(manifest),
            }
        },
    }
    _write_json(output_root / "dataset_summary.json", summary)
    return summary


def merge_datasets(
    generated: str | Path, movingai: str | Path, output: str | Path
) -> dict[str, Any]:
    output_root = Path(output).resolve()
    manifest = []
    for source_group, source_root_value in (
        ("generated", generated),
        ("movingai", movingai),
    ):
        source_root = Path(source_root_value).resolve()
        split_root = source_root / SPLIT
        for raw in _read_jsonl(split_root / "manifest.jsonl"):
            row = dict(raw)
            row["source_group"] = source_group
            for field in (
                "map_file",
                "scenario_file",
                "map_metadata_file",
                "task_file",
                "legacy_instance_file",
            ):
                if not row.get(field):
                    continue
                relative = Path(str(row[field]))
                source = split_root / relative
                if not source.is_file():
                    raise ValueError(f"dataset file is missing: {source}")
                destination = output_root / SPLIT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and sha256_file(destination) != sha256_file(source):
                    raise ValueError(f"dataset merge collision: {relative}")
                if not destination.is_file():
                    shutil.copy2(source, destination)
            manifest.append(row)
    if len(manifest) != 72 or len({str(row["task_id"]) for row in manifest}) != 72:
        raise ValueError("balanced wall-clock dataset must contain 72 unique tasks")
    manifest.sort(key=lambda row: str(row["task_id"]))
    _write_jsonl_atomic(output_root / SPLIT / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "configuration_fingerprint": _fingerprint(manifest),
        "source": "pinned MovingAI plus generated structured maps",
        "task_semantics": "static OD/scenario tasks",
        "splits": {
            SPLIT: {
                "map_count": len({str(row["map_id"]) for row in manifest}),
                "instance_count": len(manifest),
                "source_counts": dict(
                    sorted(collections.Counter(str(row["source_group"]) for row in manifest).items())
                ),
            }
        },
    }
    _write_json(output_root / "dataset_summary.json", summary)
    return summary


def build_replacement_dataset(
    *,
    original: str | Path,
    movingai_candidates: str | Path,
    generated_candidates: str | Path,
    additional_generated_candidates: str | Path,
    selection_config: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(selection_config).resolve()
    config = _read_json(config_path)
    source_roots = {
        "original": Path(original).resolve(),
        "movingai_candidates": Path(movingai_candidates).resolve(),
        "generated_candidates": Path(generated_candidates).resolve(),
        "additional_generated_candidates": Path(additional_generated_candidates).resolve(),
    }
    selected_by_source = {
        name: set(map(str, config[f"{name}_map_ids"])) for name in source_roots
    }
    selected_ids = set().union(*selected_by_source.values())
    if len(selected_ids) != int(config["expected_map_count"]):
        raise ValueError("replacement map registration is repeated or incomplete")
    removed = set(map(str, config["removed_map_ids"]))
    if selected_ids & removed:
        raise ValueError("a removed map is still selected for the replacement dataset")

    output_root = Path(output).resolve()
    if output_root in source_roots.values():
        raise ValueError("replacement output must differ from every source dataset")
    manifest: list[dict[str, Any]] = []
    observed_by_source: dict[str, set[str]] = {}
    for source_name, source_root in source_roots.items():
        split_root = source_root / SPLIT
        rows = _read_jsonl(split_root / "manifest.jsonl")
        available = {str(row["map_id"]) for row in rows}
        requested = selected_by_source[source_name]
        if not requested <= available:
            raise ValueError(
                f"replacement source {source_name} is missing maps: "
                f"{sorted(requested - available)}"
            )
        observed_by_source[source_name] = set()
        for raw in rows:
            if str(raw["map_id"]) not in requested:
                continue
            row = dict(raw)
            row["source_group"] = str(row.get("source_group", "generated"))
            for field in (
                "map_file",
                "scenario_file",
                "map_metadata_file",
                "task_file",
                "legacy_instance_file",
            ):
                if not row.get(field):
                    continue
                relative = Path(str(row[field]))
                source = split_root / relative
                if not source.is_file():
                    raise ValueError(f"replacement dataset file is missing: {source}")
                destination = output_root / SPLIT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and sha256_file(destination) != sha256_file(source):
                    raise ValueError(f"replacement dataset file collision: {relative}")
                if not destination.is_file():
                    shutil.copy2(source, destination)
            observed_by_source[source_name].add(str(row["map_id"]))
            manifest.append(row)

    map_ids = {str(row["map_id"]) for row in manifest}
    task_ids = {str(row["task_id"]) for row in manifest}
    expected_tasks = int(config["expected_instance_count"])
    if map_ids != selected_ids or len(manifest) != expected_tasks or len(task_ids) != expected_tasks:
        raise ValueError("replacement dataset dimensions differ from registration")
    source_counts = dict(
        sorted(collections.Counter(str(row["source_group"]) for row in manifest).items())
    )
    expected_source_counts = {
        str(name): int(value) for name, value in dict(config["source_counts"]).items()
    }
    if source_counts != expected_source_counts:
        raise ValueError("replacement dataset source counts differ from registration")
    manifest.sort(key=lambda row: str(row["task_id"]))
    _write_jsonl_atomic(output_root / SPLIT / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "dataset_revision": str(config["dataset_revision"]),
        "configuration_fingerprint": _fingerprint(config),
        "selection_config_sha256": sha256_file(config_path),
        "removed_map_ids": sorted(removed),
        "selected_map_ids": sorted(selected_ids),
        "source_map_counts": {
            name: len(values) for name, values in sorted(observed_by_source.items())
        },
        "splits": {
            SPLIT: {
                "map_count": len(map_ids),
                "instance_count": len(manifest),
                "source_counts": source_counts,
            }
        },
    }
    _write_json(output_root / "dataset_summary.json", summary)
    return summary


def materialize_compute_load_candidate_pool(
    registry: str | Path, output: str | Path
) -> dict[str, Any]:
    """Build the registered generated-only qualification pool."""

    registry_path = Path(registry).resolve()
    config = _read_json(registry_path)
    project_root = registry_path.parent.parent
    output_root = Path(output).resolve()
    split_output = output_root / SPLIT
    source_specs = [dict(value) for value in config["generated_sources"]]
    if not source_specs:
        raise ValueError("candidate pool must register at least one source")
    source_roots = [
        (project_root / str(value["dataset"])).resolve() for value in source_specs
    ]
    if output_root in source_roots:
        raise ValueError("candidate pool output must differ from every source dataset")

    registry_fingerprint = _fingerprint(config)
    summary_path = output_root / "dataset_summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("configuration_fingerprint") != registry_fingerprint:
            raise ValueError("candidate pool output belongs to a different registry")
    elif output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("candidate pool output is non-empty but has no summary")

    excluded_rows = [dict(value) for value in config.get("excluded_map_ids", [])]
    excluded = {str(value["map_id"]): value for value in excluded_rows}
    if len(excluded) != len(excluded_rows):
        raise ValueError("candidate pool repeats an excluded map id")

    manifest: list[dict[str, Any]] = []
    observed_excluded: set[str] = set()
    source_registration: list[dict[str, Any]] = []
    for spec, source_root in zip(source_specs, source_roots):
        source_config_path = (project_root / str(spec["config"])).resolve()
        source_summary_path = source_root / "dataset_summary.json"
        source_manifest_path = source_root / SPLIT / "manifest.jsonl"
        source_config = _read_json(source_config_path)
        source_summary = _read_json(source_summary_path)
        expected_fingerprint = _fingerprint(source_config)
        if source_summary.get("configuration_fingerprint") != expected_fingerprint:
            raise ValueError("generated source fingerprint differs from its config")
        rows = _read_jsonl(source_manifest_path)
        source_registration.append(
            {
                "config": str(spec["config"]),
                "config_sha256": sha256_file(source_config_path),
                "dataset": str(spec["dataset"]),
                "dataset_fingerprint": expected_fingerprint,
                "manifest_sha256": sha256_file(source_manifest_path),
            }
        )
        split_root = source_root / SPLIT
        for raw in rows:
            map_id = str(raw["map_id"])
            if map_id in excluded:
                map_relative = Path(str(raw["map_file"]))
                map_path = split_root / map_relative
                if sha256_file(map_path) != str(excluded[map_id]["map_sha256"]):
                    raise ValueError("excluded map SHA differs from its registration")
                observed_excluded.add(map_id)
                continue
            row = dict(raw)
            row["source_group"] = "generated"
            for field in (
                "map_file",
                "scenario_file",
                "map_metadata_file",
                "task_file",
                "legacy_instance_file",
            ):
                if not row.get(field):
                    continue
                relative = Path(str(row[field]))
                source = split_root / relative
                if not source.is_file():
                    raise ValueError(f"candidate pool source file is missing: {source}")
                destination = split_output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and sha256_file(destination) != sha256_file(source):
                    raise ValueError(f"candidate pool file collision: {relative}")
                if not destination.is_file():
                    shutil.copy2(source, destination)
            manifest.append(row)

    if observed_excluded != set(excluded):
        raise ValueError("candidate pool did not observe every excluded map")
    task_ids = [str(row["task_id"]) for row in manifest]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("candidate pool contains duplicate task ids")
    map_layouts: dict[str, str] = {}
    map_hashes: dict[str, str] = {}
    for row in manifest:
        map_id = str(row["map_id"])
        layout = str(row["layout_mode"])
        if map_id in map_layouts and map_layouts[map_id] != layout:
            raise ValueError("candidate pool map has inconsistent layouts")
        map_layouts[map_id] = layout
        map_path = split_output / str(row["map_file"])
        map_hashes[map_id] = sha256_file(map_path)
    if len(set(map_hashes.values())) != len(map_hashes):
        raise ValueError("candidate pool contains duplicate map grids")

    expected = dict(config["expected_usable"])
    layout_counts = dict(sorted(collections.Counter(map_layouts.values()).items()))
    expected_layouts = {
        str(name): int(value)
        for name, value in dict(expected["layout_map_counts"]).items()
    }
    if (
        len(map_layouts) != int(expected["map_count"])
        or len(manifest) != int(expected["instance_count"])
        or layout_counts != expected_layouts
    ):
        raise ValueError("candidate pool dimensions differ from registration")

    manifest.sort(key=lambda row: str(row["task_id"]))
    _write_jsonl_atomic(split_output / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "dataset_revision": "balanced-wall-clock-compute-load-pool-v3",
        "configuration_fingerprint": registry_fingerprint,
        "registry_sha256": sha256_file(registry_path),
        "source_registration": source_registration,
        "excluded_map_ids": sorted(excluded),
        "splits": {
            SPLIT: {
                "map_count": len(map_layouts),
                "instance_count": len(manifest),
                "source_counts": {"generated": len(manifest)},
                "layout_map_counts": layout_counts,
            }
        },
    }
    _write_json(summary_path, summary)
    return summary


def materialize_qualified_compute_load_pool(
    registry: str | Path,
    dataset_output: str | Path,
    qualification_output: str | Path,
) -> dict[str, Any]:
    """Merge checksum-pinned qualification pools without controller outcomes."""

    registry_path = Path(registry).resolve()
    config = _read_json(registry_path)
    project_root = registry_path.parent.parent
    dataset_root = Path(dataset_output).resolve()
    qualification_root = Path(qualification_output).resolve()
    if dataset_root == qualification_root:
        raise ValueError("qualified pool dataset and qualification outputs must differ")
    configuration_fingerprint = _fingerprint(config)
    for root, marker in (
        (dataset_root, "dataset_summary.json"),
        (qualification_root, "qualification_pool_info.json"),
    ):
        marker_path = root / marker
        if marker_path.is_file():
            existing = _read_json(marker_path)
            if existing.get("configuration_fingerprint") != configuration_fingerprint:
                raise ValueError("qualified pool output belongs to another registry")
        elif root.is_dir() and any(root.iterdir()):
            raise ValueError("qualified pool output is non-empty but unregistered")

    manifest: list[dict[str, Any]] = []
    qualification_rows: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    qualification_keys: set[tuple[str, int]] = set()
    map_hashes: dict[str, str] = {}
    source_registration = []
    for raw_source in config["sources"]:
        source = dict(raw_source)
        source_id = str(source["id"])
        source_group = str(source["source_group"])
        if source_group not in {"generated", "movingai"}:
            raise ValueError(f"unsupported qualified-pool source group: {source_group}")
        source_dataset = (project_root / str(source["dataset"])).resolve()
        source_qualification = (
            project_root / str(source["qualification"])
        ).resolve()
        source_manifest_path = source_dataset / SPLIT / "manifest.jsonl"
        source_qualification_path = (
            source_qualification / "qualification_manifest.jsonl"
        )
        if sha256_file(source_manifest_path) != str(source["dataset_manifest_sha256"]):
            raise ValueError(f"qualified-pool dataset SHA mismatch: {source_id}")
        if sha256_file(source_qualification_path) != str(
            source["qualification_manifest_sha256"]
        ):
            raise ValueError(f"qualified-pool qualification SHA mismatch: {source_id}")
        source_rows = _read_jsonl(source_manifest_path)
        source_tasks = {str(row["task_id"]): dict(row) for row in source_rows}
        if len(source_tasks) != len(source_rows):
            raise ValueError(f"qualified-pool source repeats task ids: {source_id}")
        if len(source_rows) != int(source["task_count"]):
            raise ValueError(f"qualified-pool source task count differs: {source_id}")

        source_split = source_dataset / SPLIT
        for task_id, raw_row in source_tasks.items():
            if task_id in task_ids:
                raise ValueError(f"qualified pool repeats task id: {task_id}")
            task_ids.add(task_id)
            row = dict(raw_row)
            row["source_group"] = source_group
            for field in (
                "map_file",
                "scenario_file",
                "map_metadata_file",
                "task_file",
                "legacy_instance_file",
            ):
                if not row.get(field):
                    continue
                relative = Path(str(row[field]))
                source_path = source_split / relative
                if not source_path.is_file():
                    raise ValueError(f"qualified-pool source file is missing: {source_path}")
                destination = dataset_root / SPLIT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source_sha = sha256_file(source_path)
                if destination.is_file() and sha256_file(destination) != source_sha:
                    raise ValueError(f"qualified-pool file collision: {relative}")
                if not destination.is_file():
                    shutil.copy2(source_path, destination)
            map_id = str(row["map_id"])
            map_sha = sha256_file(source_split / str(row["map_file"]))
            if map_id in map_hashes and map_hashes[map_id] != map_sha:
                raise ValueError(f"qualified pool map id has multiple grids: {map_id}")
            map_hashes[map_id] = map_sha
            manifest.append(row)

        source_qualified = _read_jsonl(source_qualification_path)
        if len(source_qualified) != int(source["qualification_count"]):
            raise ValueError(
                f"qualified-pool source qualification count differs: {source_id}"
            )
        for raw_result in source_qualified:
            result = dict(raw_result)
            task_id = str(result["task_id"])
            if task_id not in source_tasks:
                raise ValueError(
                    f"qualified-pool qualification has unknown task: {source_id}/{task_id}"
                )
            if str(result["map_id"]) != str(source_tasks[task_id]["map_id"]):
                raise ValueError("qualified-pool task and qualification maps differ")
            key = (task_id, int(result["solver_seed"]))
            if key in qualification_keys:
                raise ValueError(f"qualified pool repeats reset key: {key}")
            qualification_keys.add(key)
            result["source_group"] = source_group
            result["qualification_source_id"] = source_id
            qualification_rows.append(result)

        source_registration.append(
            {
                "id": source_id,
                "source_group": source_group,
                "dataset": str(source["dataset"]),
                "qualification": str(source["qualification"]),
                "dataset_manifest_sha256": str(source["dataset_manifest_sha256"]),
                "qualification_manifest_sha256": str(
                    source["qualification_manifest_sha256"]
                ),
                "task_count": len(source_rows),
                "qualification_count": len(source_qualified),
            }
        )

    if len(set(map_hashes.values())) != len(map_hashes):
        raise ValueError("qualified pool contains duplicate map grids")
    expected = dict(config["expected"])
    source_task_counts = dict(
        sorted(collections.Counter(row["source_group"] for row in manifest).items())
    )
    source_qualification_counts = dict(
        sorted(
            collections.Counter(row["source_group"] for row in qualification_rows).items()
        )
    )
    observed = {
        "map_count": len(map_hashes),
        "task_count": len(manifest),
        "qualification_count": len(qualification_rows),
        "source_task_counts": source_task_counts,
        "source_qualification_counts": source_qualification_counts,
    }
    normalized_expected = {
        "map_count": int(expected["map_count"]),
        "task_count": int(expected["task_count"]),
        "qualification_count": int(expected["qualification_count"]),
        "source_task_counts": {
            str(name): int(value)
            for name, value in dict(expected["source_task_counts"]).items()
        },
        "source_qualification_counts": {
            str(name): int(value)
            for name, value in dict(expected["source_qualification_counts"]).items()
        },
    }
    if observed != normalized_expected:
        raise ValueError("qualified pool dimensions differ from registration")

    manifest.sort(key=lambda row: str(row["task_id"]))
    qualification_rows.sort(
        key=lambda row: (str(row["task_id"]), int(row["solver_seed"]))
    )
    _write_jsonl_atomic(dataset_root / SPLIT / "manifest.jsonl", manifest)
    _write_jsonl_atomic(
        qualification_root / "qualification_manifest.jsonl", qualification_rows
    )
    summary = {
        "schema_version": 1,
        "dataset_revision": str(config["dataset_revision"]),
        "configuration_fingerprint": configuration_fingerprint,
        "selection_blind_to_controller_outcomes": True,
        "source_registration": source_registration,
        "splits": {SPLIT: observed},
    }
    _write_json(dataset_root / "dataset_summary.json", summary)
    qualification_info = {
        "schema_version": 1,
        "configuration_fingerprint": configuration_fingerprint,
        "selection_blind_to_controller_outcomes": True,
        "source_registration": source_registration,
        **observed,
        "dataset_manifest_sha256": sha256_file(
            dataset_root / SPLIT / "manifest.jsonl"
        ),
        "qualification_manifest_sha256": sha256_file(
            qualification_root / "qualification_manifest.jsonl"
        ),
    }
    _write_json(
        qualification_root / "qualification_pool_info.json", qualification_info
    )
    return {"dataset": summary, "qualification": qualification_info}


def conflict_stratum(conflicts: int) -> str | None:
    for name, lower, upper in STRATA:
        if lower <= conflicts <= upper:
            return name
    return None


def initial_pp_load_stratum(generated_nodes: int) -> str:
    if generated_nodes < 0:
        raise ValueError("initial PP generated-node count cannot be negative")
    for name, lower, upper in INITIAL_PP_LOAD_STRATA:
        if generated_nodes >= lower and (upper is None or generated_nodes <= upper):
            return name
    raise AssertionError("initial PP load strata do not cover the nonnegative integers")


def _load_difficulty_config(config: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(config).resolve()
    payload = _read_json(config_path)
    configured_conflicts = tuple(
        (name, int(bounds[0]), int(bounds[1]))
        for name, bounds in payload["conflict_strata"].items()
    )
    configured_load = tuple(
        (
            name,
            int(bounds[0]),
            int(bounds[1]) if bounds[1] is not None else None,
        )
        for name, bounds in payload["initial_pp_load_strata"].items()
    )
    if configured_conflicts != STRATA or configured_load != INITIAL_PP_LOAD_STRATA:
        raise ValueError("difficulty audit config differs from the implemented strata")
    return config_path, payload


def _agent_band(count: int) -> str:
    return "small" if count <= 200 else "medium" if count <= 400 else "large"


def select_balanced_cohort(
    dataset: str | Path, qualification: str | Path, output: str | Path
) -> dict[str, Any]:
    dataset_root = Path(dataset).resolve()
    qualification_root = Path(qualification).resolve()
    output_root = Path(output).resolve()
    rows = _read_jsonl(dataset_root / SPLIT / "manifest.jsonl")
    tasks = {str(row["task_id"]): row for row in rows}
    qualified = _read_jsonl(qualification_root / "qualification_manifest.jsonl")
    if len(qualified) != 216:
        raise ValueError(f"qualification must contain 216 results, found {len(qualified)}")
    candidates = []
    for result in qualified:
        if str(result.get("status")) != "ok" or not bool(result.get("initial_complete")):
            raise ValueError("qualification contains an invalid reset")
        task = tasks[str(result["task_id"])]
        conflicts = int(result["initial_conflicts"])
        candidates.append(
            {
                "task_id": str(result["task_id"]),
                "solver_seed": int(result["solver_seed"]),
                "map_id": str(result["map_id"]),
                "layout_mode": str(result["layout_mode"]),
                "source_group": str(task["source_group"]),
                "agent_count": int(result["agent_count"]),
                "agent_band": _agent_band(int(result["agent_count"])),
                "initial_conflicts": conflicts,
                "conflict_stratum": conflict_stratum(conflicts),
                "state_fingerprint": str(result["state_fingerprint"]),
            }
        )

    selected = []
    stratum_reports: dict[str, dict[str, Any]] = {}
    selection_passed = True
    for stratum, _lower, _upper in STRATA:
        eligible = [row for row in candidates if row["conflict_stratum"] == stratum]
        eligible.sort(
            key=lambda row: _fingerprint(
                ["balanced-wall-clock-cohort-v1", row["task_id"], row["solver_seed"]]
            )
        )
        chosen: list[dict[str, Any]] = []
        map_counts: collections.Counter[str] = collections.Counter()

        def take(predicate: Any) -> bool:
            for row in eligible:
                if row in chosen or map_counts[row["map_id"]] >= 2 or not predicate(row):
                    continue
                chosen.append(row)
                map_counts[row["map_id"]] += 1
                return True
            return False

        for source in ("generated", "movingai"):
            while sum(row["source_group"] == source for row in chosen) < 4:
                if not take(lambda row, source=source: row["source_group"] == source):
                    break
        while len({row["agent_band"] for row in chosen}) < 2:
            existing = {row["agent_band"] for row in chosen}
            if not take(lambda row, existing=existing: row["agent_band"] not in existing):
                break
        while len(chosen) < 12 and take(lambda _row: True):
            pass
        checks = {
            "count": len(chosen) == 12,
            "generated_minimum": sum(row["source_group"] == "generated" for row in chosen) >= 4,
            "movingai_minimum": sum(row["source_group"] == "movingai" for row in chosen) >= 4,
            "agent_band_count": len({row["agent_band"] for row in chosen}) >= 2,
            "map_cap": max(map_counts.values(), default=0) <= 2,
        }
        stratum_reports[stratum] = {
            "eligible": len(eligible),
            "eligible_by_source": dict(
                sorted(collections.Counter(row["source_group"] for row in eligible).items())
            ),
            "eligible_map_count": len({row["map_id"] for row in eligible}),
            "provisional_selected": len(chosen),
            "checks": checks,
            "passed": all(checks.values()),
        }
        selection_passed &= all(checks.values())
        selected.extend(chosen)

    if selection_passed:
        selected.sort(
            key=lambda row: (
                row["conflict_stratum"],
                row["map_id"],
                row["task_id"],
                row["solver_seed"],
            )
        )
        permutations = list(itertools.permutations(CONTROLLERS))
        schedule = []
        for index, row in enumerate(selected):
            schedule.append(
                {
                    **row,
                    "schedule_group": index % len(permutations),
                    "controller_order": list(permutations[index % len(permutations)]),
                }
            )
        _write_jsonl_atomic(output_root / "cohort.jsonl", selected)
        _write_json(
            output_root / "execution_schedule.json",
            {
                "schema": "lns2.controller_execution_schedule.v1",
                "selection_blind_to_controller_outcomes": True,
                "entries": schedule,
            },
        )
    report = {
        "schema": "lns2.balanced_wall_clock_cohort.v1",
        "qualification_count": len(qualified),
        "selected_count": len(selected),
        "selection_blind_to_controller_outcomes": True,
        "strata": stratum_reports,
        "excluded": {
            "zero_conflict": sum(row["initial_conflicts"] == 0 for row in candidates),
            "over_500": sum(row["initial_conflicts"] > 500 for row in candidates),
        },
        "passed": selection_passed and len(selected) == 36,
        "decision": "eligible_for_formal" if selection_passed else "data_gate_failed",
        "formal_collection_allowed": selection_passed,
    }
    _write_json(output_root / "cohort_report.json", report)
    return report


def select_compute_load_balanced_cohort(
    dataset: str | Path,
    qualification: str | Path,
    output: str | Path,
    config: str | Path = DEFAULT_DIFFICULTY_CONFIG,
) -> dict[str, Any]:
    """Freeze a 3x3 conflict/load cohort without consulting controller outcomes."""

    config_path, audit_config = _load_difficulty_config(config)
    selection = dict(audit_config["future_cohort_selection"])
    jobs_per_cell = int(selection["jobs_per_conflict_load_cell"])
    map_cap = int(selection["global_jobs_per_map_cap"])
    minimum_maps = int(selection["minimum_distinct_maps"])
    if int(selection.get("global_jobs_per_task_cap", 1)) != 1:
        raise ValueError("compute-load cohort requires one job per task")
    configured_movingai = selection.get("movingai_jobs_by_cell")
    if configured_movingai is None:
        jobs_per_source = int(selection["jobs_per_source_per_cell"])
        if jobs_per_cell != jobs_per_source * 2:
            raise ValueError("future cohort cell quota must equal two source quotas")
        movingai_jobs_by_cell = {
            f"{conflict}__{load}": jobs_per_source
            for conflict in ("low", "medium", "high")
            for load in ("low", "medium", "high")
        }
    else:
        movingai_jobs_by_cell = {
            str(name): int(value)
            for name, value in dict(configured_movingai).items()
        }
        expected_cells = {
            f"{conflict}__{load}"
            for conflict in ("low", "medium", "high")
            for load in ("low", "medium", "high")
        }
        if set(movingai_jobs_by_cell) != expected_cells or any(
            value < 0 or value > jobs_per_cell
            for value in movingai_jobs_by_cell.values()
        ):
            raise ValueError("movingai cell quotas do not cover the 3x3 cohort")

    dataset_root = Path(dataset).resolve()
    qualification_root = Path(qualification).resolve()
    output_root = Path(output).resolve()
    tasks = {
        str(row["task_id"]): row
        for row in _read_jsonl(dataset_root / SPLIT / "manifest.jsonl")
    }
    qualified = _read_jsonl(qualification_root / "qualification_manifest.jsonl")
    keys = [(str(row["task_id"]), int(row["solver_seed"])) for row in qualified]
    if len(keys) != len(set(keys)):
        raise ValueError("qualification contains duplicate task/solver-seed results")

    candidates = []
    for result in qualified:
        if str(result.get("status")) != "ok" or not bool(result.get("initial_complete")):
            raise ValueError("qualification contains an invalid reset")
        complexity = result.get("initial_complexity")
        if not isinstance(complexity, dict):
            raise ValueError(
                "qualification lacks initial_complexity; rerun qualification with the current collector"
            )
        task = tasks[str(result["task_id"])]
        conflicts = int(result["initial_conflicts"])
        if int(complexity.get("conflict_pair_count", -1)) != conflicts:
            raise ValueError("qualification complexity disagrees with initial conflict count")
        generated_nodes = int(complexity["initial_low_level_generated"])
        conflict_level = conflict_stratum(conflicts)
        candidates.append(
            {
                "task_id": str(result["task_id"]),
                "solver_seed": int(result["solver_seed"]),
                "map_id": str(result["map_id"]),
                "layout_mode": str(result["layout_mode"]),
                "source_group": str(task["source_group"]),
                "agent_count": int(result["agent_count"]),
                "agent_band": _agent_band(int(result["agent_count"])),
                "initial_conflicts": conflicts,
                "conflict_stratum": conflict_level,
                "initial_pp_load_stratum": initial_pp_load_stratum(generated_nodes),
                "initial_low_level_generated": generated_nodes,
                "initial_low_level_expanded": int(
                    complexity["initial_low_level_expanded"]
                ),
                "total_path_cost": int(complexity["total_path_cost"]),
                "conflict_event_count": int(complexity["conflict_event_count"]),
                "active_conflict_agent_ratio": float(
                    complexity["active_conflict_agent_ratio"]
                ),
                "largest_conflict_component_ratio": float(
                    complexity["largest_conflict_component_ratio"]
                ),
                "state_fingerprint": str(result["state_fingerprint"]),
            }
        )

    eligible = [row for row in candidates if row["conflict_stratum"] is not None]
    cell_specs = [
        (conflict, load)
        for conflict in ("low", "medium", "high")
        for load in ("low", "medium", "high")
    ]
    cell_specs.sort(
        key=lambda cell: (
            sum(
                row["conflict_stratum"] == cell[0]
                and row["initial_pp_load_stratum"] == cell[1]
                for row in eligible
            ),
            cell,
        )
    )
    selected: list[dict[str, Any]] = []
    selected_task_ids: set[str] = set()
    map_counts: collections.Counter[str] = collections.Counter()
    cell_reports: dict[str, dict[str, Any]] = {}
    for conflict_level, load_level in cell_specs:
        cell_name = f"{conflict_level}__{load_level}"
        pool = [
            row
            for row in eligible
            if row["conflict_stratum"] == conflict_level
            and row["initial_pp_load_stratum"] == load_level
        ]
        chosen: list[dict[str, Any]] = []
        chosen_maps: set[str] = set()
        source_quotas = {
            "movingai": movingai_jobs_by_cell[cell_name],
            "generated": jobs_per_cell - movingai_jobs_by_cell[cell_name],
        }
        for source in ("generated", "movingai"):
            source_pool = sorted(
                (row for row in pool if row["source_group"] == source),
                key=lambda row: (
                    map_counts[row["map_id"]],
                    _fingerprint(
                        [
                            "compute-load-balanced-cohort-v1-source",
                            conflict_level,
                            load_level,
                            source,
                            row["task_id"],
                            row["solver_seed"],
                        ]
                    ),
                ),
            )
            for row in source_pool:
                if (
                    sum(item["source_group"] == source for item in chosen)
                    >= source_quotas[source]
                ):
                    break
                if (
                    row["map_id"] in chosen_maps
                    or row["task_id"] in selected_task_ids
                    or map_counts[row["map_id"]] >= map_cap
                ):
                    continue
                chosen.append(row)
                chosen_maps.add(row["map_id"])
                selected_task_ids.add(row["task_id"])
                map_counts[row["map_id"]] += 1
        source_counts = collections.Counter(row["source_group"] for row in chosen)
        checks = {
            "cell_count": len(chosen) == jobs_per_cell,
            "generated_count": source_counts.get("generated", 0)
            == source_quotas["generated"],
            "movingai_count": source_counts.get("movingai", 0)
            == source_quotas["movingai"],
            "distinct_maps": len(chosen_maps) == jobs_per_cell,
        }
        cell_reports[cell_name] = {
            "eligible_count": len(pool),
            "eligible_by_source": dict(
                sorted(collections.Counter(row["source_group"] for row in pool).items())
            ),
            "source_quotas": source_quotas,
            "selected_count": len(chosen),
            "checks": checks,
            "passed": all(checks.values()),
        }
        selected.extend(chosen)

    selected_source_counts = collections.Counter(
        row["source_group"] for row in selected
    )
    expected_source_counts = {
        "movingai": sum(movingai_jobs_by_cell.values()),
        "generated": jobs_per_cell * len(cell_specs)
        - sum(movingai_jobs_by_cell.values()),
    }
    configured_source_counts = selection.get("exact_total_source_counts")
    if configured_source_counts is not None:
        registered_source_counts = {
            str(name): int(value)
            for name, value in dict(configured_source_counts).items()
        }
        if registered_source_counts != expected_source_counts:
            raise ValueError("cell quotas and total source quotas disagree")
    conflict_source_counts = {
        conflict: {
            source: sum(
                row["conflict_stratum"] == conflict
                and row["source_group"] == source
                for row in selected
            )
            for source in ("generated", "movingai")
        }
        for conflict in ("low", "medium", "high")
    }
    configured_conflict_counts = selection.get("source_counts_per_conflict_tier")
    conflict_balance_passed = True
    if configured_conflict_counts is not None:
        normalized_conflict_counts = {
            str(conflict): {
                str(source): int(value)
                for source, value in dict(counts).items()
            }
            for conflict, counts in dict(configured_conflict_counts).items()
        }
        conflict_balance_passed = conflict_source_counts == normalized_conflict_counts
    minimum_per_load = {
        str(source): int(value)
        for source, value in dict(
            selection.get("minimum_source_count_per_load_tier", {})
        ).items()
    }
    load_source_counts = {
        load: {
            source: sum(
                row["initial_pp_load_stratum"] == load
                and row["source_group"] == source
                for row in selected
            )
            for source in ("generated", "movingai")
        }
        for load in ("low", "medium", "high")
    }
    load_overlap_passed = all(
        counts.get(source, 0) >= minimum
        for counts in load_source_counts.values()
        for source, minimum in minimum_per_load.items()
    )
    overall_checks = {
        "all_nine_cells_pass": all(row["passed"] for row in cell_reports.values()),
        "selected_count": len(selected) == jobs_per_cell * len(cell_specs),
        "minimum_distinct_maps": len({row["map_id"] for row in selected}) >= minimum_maps,
        "global_map_cap": max(map_counts.values(), default=0) <= map_cap,
        "unique_tasks": len(selected_task_ids) == len(selected),
        "exact_source_balance": dict(selected_source_counts) == expected_source_counts,
        "source_balance_per_conflict_tier": conflict_balance_passed,
        "source_overlap_per_load_tier": load_overlap_passed,
    }
    passed = all(overall_checks.values())
    if passed:
        selected.sort(
            key=lambda row: (
                str(row["conflict_stratum"]),
                str(row["initial_pp_load_stratum"]),
                str(row["map_id"]),
                str(row["task_id"]),
                int(row["solver_seed"]),
            )
        )
        permutations = list(itertools.permutations(CONTROLLERS))
        schedule = [
            {
                **row,
                "schedule_group": index % len(permutations),
                "controller_order": list(permutations[index % len(permutations)]),
            }
            for index, row in enumerate(selected)
        ]
        _write_jsonl_atomic(output_root / "cohort.jsonl", selected)
        _write_json(
            output_root / "execution_schedule.json",
            {
                "schema": "lns2.controller_execution_schedule.compute_load_v1",
                "selection_blind_to_controller_outcomes": True,
                "entries": schedule,
            },
        )
    report = {
        "schema": "lns2.compute_load_balanced_wall_clock_cohort.v1",
        "configuration_sha256": sha256_file(config_path),
        "qualification_count": len(qualified),
        "eligible_nonzero_nonextreme_count": len(eligible),
        "selected_count": len(selected),
        "selection_blind_to_controller_outcomes": True,
        "cell_reports": dict(sorted(cell_reports.items())),
        "selected_source_counts": dict(sorted(selected_source_counts.items())),
        "source_counts_by_conflict_tier": conflict_source_counts,
        "source_counts_by_load_tier": load_source_counts,
        "overall_checks": overall_checks,
        "passed": passed,
        "decision": "eligible_for_formal" if passed else "compute_load_data_gate_failed",
        "formal_collection_allowed": passed,
    }
    _write_json(output_root / "cohort_report.json", report)
    return report


def verify_compute_load_cohort_registration(
    registration: str | Path,
) -> dict[str, Any]:
    registration_path = Path(registration).resolve()
    project_root = registration_path.parent.parent
    payload = _read_json(registration_path)
    registered_files = (
        ("pool_registry", "pool_registry_sha256"),
        ("selection_config", "selection_config_sha256"),
        ("merged_dataset_manifest", "merged_dataset_manifest_sha256"),
        ("merged_qualification_manifest", "merged_qualification_manifest_sha256"),
        ("cohort", "cohort_sha256"),
        ("execution_schedule", "execution_schedule_sha256"),
        ("cohort_report", "cohort_report_sha256"),
    )
    if "formal_dataset_manifest" in payload:
        registered_files += (
            ("formal_dataset_manifest", "formal_dataset_manifest_sha256"),
        )
    resolved: dict[str, Path] = {}
    for path_key, sha_key in registered_files:
        path = (project_root / str(payload[path_key])).resolve()
        if not path.is_file() or sha256_file(path) != str(payload[sha_key]):
            raise ValueError(f"compute-load cohort registration mismatch: {path_key}")
        resolved[path_key] = path

    cohort = _read_jsonl(resolved["cohort"])
    schedule = _read_json(resolved["execution_schedule"])
    entries = list(schedule["entries"])
    cohort_keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in cohort}
    schedule_keys = {
        (str(row["task_id"]), int(row["solver_seed"])) for row in entries
    }
    if len(cohort_keys) != len(cohort) or cohort_keys != schedule_keys:
        raise ValueError("registered cohort and execution schedule differ")
    orders = [tuple(map(str, row["controller_order"])) for row in entries]
    expected_orders = set(itertools.permutations(CONTROLLERS))
    if set(orders) != expected_orders or any(
        orders.count(order) != 6 for order in expected_orders
    ):
        raise ValueError("registered execution schedule is not order-balanced")

    cell_counts = collections.Counter(
        (row["conflict_stratum"], row["initial_pp_load_stratum"])
        for row in cohort
    )
    if len(cell_counts) != 9 or len(set(cell_counts.values())) != 1:
        raise ValueError("registered cohort does not balance all nine cells")
    observed_counts = {
        "jobs": len(cohort),
        "tasks": len({str(row["task_id"]) for row in cohort}),
        "maps": len({str(row["map_id"]) for row in cohort}),
        "jobs_per_conflict_load_cell": next(iter(cell_counts.values())),
        "source": dict(
            sorted(collections.Counter(row["source_group"] for row in cohort).items())
        ),
        "source_per_conflict_tier": {
            conflict: {
                source: sum(
                    row["conflict_stratum"] == conflict
                    and row["source_group"] == source
                    for row in cohort
                )
                for source in ("generated", "movingai")
            }
            for conflict in ("low", "medium", "high")
        },
        "source_per_initial_pp_load_tier": {
            load: {
                source: sum(
                    row["initial_pp_load_stratum"] == load
                    and row["source_group"] == source
                    for row in cohort
                )
                for source in ("generated", "movingai")
            }
            for load in ("low", "medium", "high")
        },
    }
    if observed_counts != dict(payload["counts"]):
        raise ValueError("registered cohort counts differ from its manifest")
    report = _read_json(resolved["cohort_report"])
    if not bool(report.get("formal_collection_allowed", False)):
        raise ValueError("registered cohort is not eligible for formal collection")
    return {
        "schema": str(payload["schema"]),
        "status": str(payload["status"]),
        "registration_sha256": sha256_file(registration_path),
        "cohort_sha256": str(payload["cohort_sha256"]),
        "execution_schedule_sha256": str(payload["execution_schedule_sha256"]),
        "counts": observed_counts,
        "passed": True,
    }


def materialize_registered_compute_load_cohort(
    registration: str | Path, output: str | Path
) -> dict[str, Any]:
    verification = verify_compute_load_cohort_registration(registration)
    registration_path = Path(registration).resolve()
    project_root = registration_path.parent.parent
    payload = _read_json(registration_path)
    source_manifest_path = (
        project_root / str(payload["merged_dataset_manifest"])
    ).resolve()
    cohort_path = (project_root / str(payload["cohort"])).resolve()
    source_split = source_manifest_path.parent
    output_root = Path(output).resolve()
    split_root = output_root / SPLIT
    configuration_fingerprint = _fingerprint(
        {
            "merged_dataset_manifest_sha256": str(
                payload["merged_dataset_manifest_sha256"]
            ),
            "cohort_sha256": str(payload["cohort_sha256"]),
        }
    )
    summary_path = output_root / "dataset_summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("configuration_fingerprint") != configuration_fingerprint:
            raise ValueError("registered cohort dataset belongs to another selection")
    elif output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("registered cohort dataset output is non-empty and unregistered")

    source_rows = {
        str(row["task_id"]): dict(row)
        for row in _read_jsonl(source_manifest_path)
    }
    cohort = _read_jsonl(cohort_path)
    manifest = []
    for selection in cohort:
        task_id = str(selection["task_id"])
        if task_id not in source_rows:
            raise ValueError(f"registered cohort task is absent from dataset: {task_id}")
        row = dict(source_rows[task_id])
        if (
            str(row["map_id"]) != str(selection["map_id"])
            or str(row["source_group"]) != str(selection["source_group"])
        ):
            raise ValueError("registered cohort metadata differs from source dataset")
        for field in (
            "map_file",
            "scenario_file",
            "map_metadata_file",
            "task_file",
            "legacy_instance_file",
        ):
            if not row.get(field):
                continue
            relative = Path(str(row[field]))
            source = source_split / relative
            if not source.is_file():
                raise ValueError(f"registered cohort source file is missing: {source}")
            destination = split_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source_sha = sha256_file(source)
            if destination.is_file() and sha256_file(destination) != source_sha:
                raise ValueError(f"registered cohort file collision: {relative}")
            if not destination.is_file():
                shutil.copy2(source, destination)
        manifest.append(row)

    expected = dict(payload["counts"])
    observed = {
        "map_count": len({str(row["map_id"]) for row in manifest}),
        "instance_count": len(manifest),
        "source_counts": dict(
            sorted(collections.Counter(row["source_group"] for row in manifest).items())
        ),
    }
    expected_dimensions = {
        "map_count": int(expected["maps"]),
        "instance_count": int(expected["tasks"]),
        "source_counts": {
            str(name): int(value) for name, value in dict(expected["source"]).items()
        },
    }
    if observed != expected_dimensions:
        raise ValueError("registered cohort dataset dimensions differ from registration")
    manifest.sort(key=lambda row: str(row["task_id"]))
    _write_jsonl_atomic(split_root / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "dataset_revision": "balanced-wall-clock-formal-cohort-v6",
        "configuration_fingerprint": configuration_fingerprint,
        "selection_blind_to_controller_outcomes": True,
        "cohort_sha256": str(payload["cohort_sha256"]),
        "registration_verification": verification,
        "splits": {SPLIT: observed},
    }
    _write_json(summary_path, summary)
    return summary


def qualify_corrected_native_schedule(
    *,
    dataset: str | Path,
    config: str | Path,
    source_schedule_root: str | Path,
    output: str | Path,
    original_bundle: str | Path,
    workers: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reset the frozen 36 task/seed keys under corrected native semantics."""

    if type(workers) is not int or workers <= 0:
        raise ValueError("corrected qualification workers must be positive")
    output_root = Path(output).resolve()
    if output_root.exists():
        raise ValueError(
            "corrected qualification output already exists; use a new directory"
        )
    schedule_path = (
        Path(source_schedule_root).resolve() / "execution_schedule.json"
    )
    schedule = _read_json(schedule_path)
    entries = schedule.get("entries")
    if not isinstance(entries, list) or len(entries) != 36:
        raise ValueError("corrected qualification requires the frozen 36-entry schedule")
    keys: set[tuple[str, int]] = set()
    for row in entries:
        if not isinstance(row, dict):
            raise ValueError("corrected qualification schedule row is not an object")
        key = _episode_key(row)
        if key in keys:
            raise ValueError("corrected qualification schedule has duplicate keys")
        keys.add(key)

    config_path = Path(config).resolve()
    payload = _read_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError("corrected qualification config is not an object")
    environment = payload.get("environment")
    if (
        payload.get("formal") is not True
        or WALL_CLOCK_SAFETY_MAX_DECISIONS != 100_000
        or not isinstance(environment, dict)
        or type(environment.get("max_repair_iterations")) is not int
        or environment["max_repair_iterations"] != 0
        or not _is_exact_number(environment.get("time_limit"), 600.0)
        or type(payload.get("max_decisions")) is not int
        or payload["max_decisions"] != 0
        or type(payload.get("metric_iteration_budget")) is not int
        or payload["metric_iteration_budget"] != 100
        or not _is_exact_number(
            payload.get("wall_time_budget_seconds"), 600.0
        )
        or not _is_exact_number(
            payload.get("episode_process_timeout_seconds"), 660.0
        )
    ):
        raise ValueError(
            "corrected qualification config must remove both 100-step "
            "execution limits while retaining metric_iteration_budget=100"
        )
    result = run_closed_loop_collection(
        dataset=dataset,
        config_path=config_path,
        output=output_root,
        phase="qualify",
        workers=workers,
        resume=False,
        dry_run=dry_run,
        task_ids=None,
        controller="v2-full",
        feature_backend="auto",
        controller_bundle=original_bundle,
        controller_runtime="optimized",
        verification_profile="deployment",
        job_keys=keys,
        cohort_job_keys=keys,
        stopping_rule="wall-clock-fixed-metric",
        use_global_collection_lock=False,
    )
    if not dry_run:
        qualification_report_path = output_root / "qualification_report.json"
        qualification_manifest_path = output_root / "qualification_manifest.jsonl"
        run_config_path = output_root / "run_config.json"
        if (
            not qualification_report_path.is_file()
            or not qualification_manifest_path.is_file()
            or not run_config_path.is_file()
        ):
            raise ValueError(
                "corrected qualification did not publish its complete artifact set"
            )
        qualification_report = _read_json(qualification_report_path)
        result_report = result.get("qualification")
        if (
            not isinstance(result_report, dict)
            or result_report != qualification_report
            or qualification_report.get("passed") is not True
            or type(qualification_report.get("valid_count")) is not int
            or qualification_report["valid_count"] != len(keys)
            or type(qualification_report.get("expected_reset_count")) is not int
            or qualification_report["expected_reset_count"] != len(keys)
            or qualification_report.get("errors") != []
            or qualification_report.get("incomplete_reset_count") != 0
            or qualification_report.get("inconsistent_initial_state_count") != 0
        ):
            raise ValueError(
                "corrected qualification report failed or is inconsistent"
            )
    return {
        "schema": "lns2.corrected_native_qualification_entry.v1",
        "schedule_sha256": sha256_file(schedule_path),
        "job_count": len(keys),
        "stopping_rule": "wall-clock-fixed-metric",
        "max_decisions": 0,
        "max_repair_iterations": 0,
        "metric_iteration_budget": 100,
        "output": str(output_root),
        "collector": result,
        "qualification_report_sha256": (
            sha256_file(output_root / "qualification_report.json")
            if not dry_run
            else None
        ),
    }


def qualify_corrected_native_seed_pool(
    *,
    dataset: str | Path,
    config: str | Path,
    source_schedule_root: str | Path,
    output: str | Path,
    original_bundle: str | Path,
    workers: int = 1,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reset every configured seed for each task in the frozen formal cohort."""

    if type(workers) is not int or workers <= 0:
        raise ValueError("corrected seed-pool qualification workers must be positive")
    output_root = Path(output).resolve()
    if output_root.exists():
        raise ValueError(
            "corrected seed-pool qualification output already exists; "
            "use a new directory"
        )
    source_schedule_path = (
        Path(source_schedule_root).resolve() / "execution_schedule.json"
    )
    source_schedule = _read_json(source_schedule_path)
    entries = source_schedule.get("entries")
    if not isinstance(entries, list) or len(entries) != 36:
        raise ValueError(
            "corrected seed-pool qualification requires the frozen "
            "36-entry schedule"
        )
    task_ids: list[str] = []
    for value in entries:
        if not isinstance(value, dict):
            raise ValueError(
                "corrected seed-pool source schedule row is not an object"
            )
        task_id, _solver_seed = _episode_key(value)
        task_ids.append(task_id)
    if len(set(task_ids)) != 36:
        raise ValueError(
            "corrected seed-pool source schedule must contain 36 distinct tasks"
        )

    config_path = Path(config).resolve()
    payload = _read_json(config_path)
    if not isinstance(payload, dict):
        raise ValueError("corrected seed-pool config is not an object")
    environment = payload.get("environment")
    raw_seeds = payload.get("solver_seeds")
    if (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) != 3
        or any(type(seed) is not int or seed < 0 for seed in raw_seeds)
        or len(set(raw_seeds)) != len(raw_seeds)
    ):
        raise ValueError(
            "corrected seed-pool qualification requires exactly three "
            "unique non-negative integer solver seeds"
        )
    solver_seeds = tuple(raw_seeds)
    if (
        payload.get("formal") is not True
        or WALL_CLOCK_SAFETY_MAX_DECISIONS != 100_000
        or not isinstance(environment, dict)
        or type(environment.get("max_repair_iterations")) is not int
        or environment["max_repair_iterations"] != 0
        or not _is_exact_number(environment.get("time_limit"), 600.0)
        or type(payload.get("max_decisions")) is not int
        or payload["max_decisions"] != 0
        or type(payload.get("metric_iteration_budget")) is not int
        or payload["metric_iteration_budget"] != 100
        or not _is_exact_number(
            payload.get("wall_time_budget_seconds"), 600.0
        )
        or not _is_exact_number(
            payload.get("episode_process_timeout_seconds"), 660.0
        )
    ):
        raise ValueError(
            "corrected seed-pool config must remove both 100-step "
            "execution limits while retaining metric_iteration_budget=100"
        )
    keys = {
        (task_id, solver_seed)
        for task_id in task_ids
        for solver_seed in solver_seeds
    }
    if len(keys) != 108:
        raise AssertionError("corrected seed-pool qualification must contain 108 jobs")
    result = run_closed_loop_collection(
        dataset=dataset,
        config_path=config_path,
        output=output_root,
        phase="qualify",
        workers=workers,
        resume=False,
        dry_run=dry_run,
        task_ids=None,
        controller="v2-full",
        feature_backend="auto",
        controller_bundle=original_bundle,
        controller_runtime="optimized",
        verification_profile="deployment",
        job_keys=keys,
        cohort_job_keys=keys,
        stopping_rule="wall-clock-fixed-metric",
        use_global_collection_lock=False,
    )
    if not dry_run:
        report_path = output_root / "qualification_report.json"
        manifest_path = output_root / "qualification_manifest.jsonl"
        run_config_path = output_root / "run_config.json"
        if (
            not report_path.is_file()
            or not manifest_path.is_file()
            or not run_config_path.is_file()
        ):
            raise ValueError(
                "corrected seed-pool qualification did not publish its "
                "complete artifact set"
            )
        report = _read_json(report_path)
        result_report = result.get("qualification")
        rows = _read_jsonl(manifest_path)
        observed_keys = {_episode_key(row) for row in rows}
        if (
            not isinstance(result_report, dict)
            or result_report != report
            or report.get("schema") != "lns2.closed_loop_confirmation.v1"
            or report.get("passed") is not True
            or type(report.get("valid_count")) is not int
            or report["valid_count"] != len(keys)
            or type(report.get("expected_reset_count")) is not int
            or report["expected_reset_count"] != len(keys)
            or report.get("errors") != []
            or report.get("incomplete_reset_count") != 0
            or report.get("inconsistent_initial_state_count") != 0
            or len(rows) != len(keys)
            or observed_keys != keys
        ):
            raise ValueError(
                "corrected seed-pool qualification report or coverage "
                "is inconsistent"
            )
    return {
        "schema": "lns2.corrected_native_seed_pool_qualification_entry.v1",
        "source_schedule_sha256": sha256_file(source_schedule_path),
        "task_count": len(task_ids),
        "solver_seeds": list(solver_seeds),
        "job_count": len(keys),
        "stopping_rule": "wall-clock-fixed-metric",
        "max_decisions": 0,
        "max_repair_iterations": 0,
        "metric_iteration_budget": 100,
        "output": str(output_root),
        "collector": result,
        "qualification_report_sha256": (
            sha256_file(output_root / "qualification_report.json")
            if not dry_run
            else None
        ),
    }


def _corrected_seed_candidate(
    qualified: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    key = _episode_key(qualified)
    if (
        qualified.get("schema") != "lns2.repair_collection.v2"
        or qualified.get("schema_version") != 2
        or qualified.get("status") != "ok"
        or qualified.get("error") is not None
    ):
        raise ValueError(f"corrected seed-pool reset failed for {key}")
    if qualified.get("initial_complete") is not True:
        raise ValueError(f"corrected seed-pool reset is incomplete for {key}")
    for field in ("map_id", "agent_count", "layout_mode"):
        if qualified.get(field) != source.get(field):
            raise ValueError(
                f"corrected seed-pool {field} differs from frozen task for {key}"
            )
    initial_feasible = qualified.get("initial_feasible")
    if type(initial_feasible) is not bool:
        raise ValueError(
            f"corrected seed-pool initial_feasible is invalid for {key}"
        )
    if qualified.get("repairable") is not (not initial_feasible):
        raise ValueError(
            f"corrected seed-pool repairable flag is inconsistent for {key}"
        )
    complexity = qualified.get("initial_complexity")
    if not isinstance(complexity, dict):
        raise ValueError(
            f"corrected seed-pool lacks initial complexity for {key}"
        )
    conflicts = qualified.get("initial_conflicts")
    if type(conflicts) is not int or conflicts < 0:
        raise ValueError(
            f"corrected seed-pool initial conflicts are invalid for {key}"
        )
    if initial_feasible is not (conflicts == 0):
        raise ValueError(
            f"corrected seed-pool feasibility conflicts with reset metrics for {key}"
        )
    if complexity.get("conflict_pair_count") != conflicts:
        raise ValueError(
            f"corrected seed-pool complexity conflicts mismatch for {key}"
        )
    generated = complexity.get("initial_low_level_generated")
    if type(generated) is not int or generated < 0:
        raise ValueError(
            f"corrected seed-pool initial PP load is invalid for {key}"
        )
    state_digest = qualified.get("state_fingerprint")
    if not _is_sha256(state_digest):
        raise ValueError(
            f"corrected seed-pool state fingerprint is invalid for {key}"
        )
    active_ratio = _strict_summary_number(
        complexity, "active_conflict_agent_ratio"
    )
    largest_ratio = _strict_summary_number(
        complexity, "largest_conflict_component_ratio"
    )
    if active_ratio > 1.0 or largest_ratio > 1.0:
        raise ValueError(
            f"corrected seed-pool complexity ratio exceeds one for {key}"
        )
    old_conflicts = source.get("initial_conflicts")
    old_generated = source.get("initial_low_level_generated")
    if (
        type(old_conflicts) is not int
        or old_conflicts < 0
        or type(old_generated) is not int
        or old_generated < 0
    ):
        raise ValueError(
            f"corrected seed source covariates are invalid for {key[0]}"
        )
    covariate_drift = (
        Fraction(abs(conflicts - old_conflicts), max(old_conflicts, 1))
        + Fraction(abs(generated - old_generated), max(old_generated, 1))
    )
    return {
        **source,
        "solver_seed": key[1],
        "initial_conflicts": conflicts,
        "state_fingerprint": state_digest,
        "conflict_stratum": conflict_stratum(conflicts),
        "initial_pp_load_stratum": initial_pp_load_stratum(generated),
        "active_conflict_agent_ratio": active_ratio,
        "conflict_event_count": _strict_summary_int(
            complexity, "conflict_event_count"
        ),
        "initial_low_level_expanded": _strict_summary_int(
            complexity, "initial_low_level_expanded"
        ),
        "initial_low_level_generated": generated,
        "largest_conflict_component_ratio": largest_ratio,
        "total_path_cost": _strict_summary_int(
            complexity, "total_path_cost"
        ),
        "_initial_feasible": initial_feasible,
        "_covariate_drift": covariate_drift,
        "_tie_break": _fingerprint(
            [
                CORRECTED_SEED_SELECTION_SALT,
                key[0],
                key[1],
                state_digest,
            ]
        ),
    }


def _best_seed_assignments_by_cell(
    tasks: list[dict[str, Any]],
    candidates_by_task: dict[str, list[dict[str, Any]]],
    original_seed_by_task: dict[str, int],
    solver_seed_rank: dict[int, int],
) -> dict[
    tuple[int, ...],
    tuple[
        int,
        Fraction,
        tuple[str, ...],
        dict[str, dict[str, Any]],
    ],
]:
    """Return the best deterministic assignment for every feasible cell-count state."""

    cells = [
        (conflict_name, load_name)
        for conflict_name, _lower, _upper in STRATA
        for load_name, _load_lower, _load_upper in INITIAL_PP_LOAD_STRATA
    ]
    cell_index = {cell: index for index, cell in enumerate(cells)}
    states: dict[
        tuple[int, ...],
        tuple[
            int,
            Fraction,
            tuple[str, ...],
            dict[str, dict[str, Any]],
        ],
    ] = {(0,) * len(cells): (0, Fraction(0), (), {})}
    for source in tasks:
        task_id = str(source["task_id"])
        choices_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
        for candidate in candidates_by_task.get(task_id, []):
            conflict_name = candidate.get("conflict_stratum")
            load_name = candidate.get("initial_pp_load_stratum")
            if (
                candidate.get("_initial_feasible") is True
                or conflict_name is None
                or (conflict_name, load_name) not in cell_index
            ):
                continue
            cell = (str(conflict_name), str(load_name))
            previous = choices_by_cell.get(cell)
            candidate_preference = (
                int(candidate["solver_seed"])
                != original_seed_by_task[task_id],
                candidate["_covariate_drift"],
                candidate["_tie_break"],
                solver_seed_rank[int(candidate["solver_seed"])],
            )
            previous_preference = (
                (
                    int(previous["solver_seed"])
                    != original_seed_by_task[task_id],
                    previous["_covariate_drift"],
                    previous["_tie_break"],
                    solver_seed_rank[int(previous["solver_seed"])],
                )
                if previous is not None
                else None
            )
            if previous_preference is None or candidate_preference < previous_preference:
                choices_by_cell[cell] = candidate
        if not choices_by_cell:
            return {}
        next_states: dict[
            tuple[int, ...],
            tuple[
                int,
                Fraction,
                tuple[str, ...],
                dict[str, dict[str, Any]],
            ],
        ] = {}
        for counts, (retained, drift, tie_break, assignment) in states.items():
            for cell, candidate in sorted(choices_by_cell.items()):
                position = cell_index[cell]
                if counts[position] >= 4:
                    continue
                conflict_offset = (position // 3) * 3
                if sum(counts[conflict_offset : conflict_offset + 3]) >= 6:
                    continue
                updated = list(counts)
                updated[position] += 1
                updated_counts = tuple(updated)
                seed = int(candidate["solver_seed"])
                updated_value = (
                    retained + int(seed == original_seed_by_task[task_id]),
                    drift + candidate["_covariate_drift"],
                    tie_break + (str(candidate["_tie_break"]),),
                    {**assignment, task_id: candidate},
                )
                existing = next_states.get(updated_counts)
                if (
                    existing is None
                    or updated_value[0] > existing[0]
                    or (
                        updated_value[0] == existing[0]
                        and updated_value[1] < existing[1]
                    )
                    or (
                        updated_value[0] == existing[0]
                        and updated_value[1] == existing[1]
                        and updated_value[2] < existing[2]
                    )
                ):
                    next_states[updated_counts] = updated_value
        states = next_states
        if not states:
            return {}
    return {
        counts: value
        for counts, value in states.items()
        if all(
            sum(counts[offset : offset + 3]) == 6
            for offset in range(0, len(cells), 3)
        )
    }


def _corrected_amendment_evidence_identity(
    root: str | Path,
    *,
    label: str,
    expected_count: int,
    dataset_fingerprint: str,
) -> dict[str, Any]:
    evidence_root = Path(root).resolve()
    manifest_path = evidence_root / "qualification_manifest.jsonl"
    report_path = evidence_root / "qualification_report.json"
    run_config_path = evidence_root / "run_config.json"
    if (
        not manifest_path.is_file()
        or not report_path.is_file()
        or not run_config_path.is_file()
    ):
        raise ValueError(f"{label} amendment evidence is incomplete")
    rows = _read_jsonl(manifest_path)
    report = _read_json(report_path)
    run_config = _read_json(run_config_path)
    if (
        len(rows) != expected_count
        or len({_episode_key(row) for row in rows}) != expected_count
        or any(
            row.get("schema") != "lns2.repair_collection.v2"
            or row.get("schema_version") != 2
            or row.get("status") != "ok"
            or row.get("error") is not None
            for row in rows
        )
        or report.get("schema") != "lns2.closed_loop_confirmation.v1"
        or report.get("passed") is not True
        or type(report.get("valid_count")) is not int
        or report["valid_count"] != expected_count
        or type(report.get("expected_reset_count")) is not int
        or report["expected_reset_count"] != expected_count
        or report.get("errors") != []
        or not isinstance(run_config, dict)
        or run_config.get("dataset_fingerprint") != dataset_fingerprint
        or not _is_sha256(run_config.get("run_fingerprint"))
    ):
        raise ValueError(f"{label} amendment evidence is inconsistent")
    producer = run_config.get("producer_identity")
    if not isinstance(producer, dict):
        producer = run_config.get("controller_implementation")
    validated_producer = _validate_corrected_producer_identity(
        producer, label=label
    )
    native = _native_identity_from_producer(validated_producer)
    assert native is not None
    return {
        "label": label,
        "qualification_manifest_sha256": sha256_file(manifest_path),
        "qualification_report_sha256": sha256_file(report_path),
        "run_config_sha256": sha256_file(run_config_path),
        "run_fingerprint": run_config["run_fingerprint"],
        "job_count": expected_count,
        "native": {
            "sha256": native["sha256"],
            "native_semantics_schema": native["native_semantics_schema"],
            "repair_timing_schema": native["repair_timing_schema"],
        },
    }


def select_corrected_native_seed_schedule(
    *,
    dataset: str | Path,
    source_schedule_root: str | Path,
    qualification: str | Path,
    observed_qualification: str | Path,
    seed_probe_qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Choose one corrected-native reset seed per frozen formal task.

    Selection is an exact deterministic assignment over reset-only conflict and
    initial-PP-load metrics.  No controller repair outcome is read.
    """

    source_root = Path(source_schedule_root).resolve()
    qualification_root = Path(qualification).resolve()
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    if output_root in {source_root, qualification_root, dataset_root}:
        raise ValueError(
            "corrected seed schedule output must be a new directory"
        )
    if output_root.exists():
        raise ValueError(
            "corrected seed schedule output already exists; use a new directory"
        )
    source_schedule_path = source_root / "execution_schedule.json"
    qualification_path = qualification_root / "qualification_manifest.jsonl"
    run_config_path = qualification_root / "run_config.json"
    qualification_report_path = qualification_root / "qualification_report.json"
    if not source_schedule_path.is_file():
        raise ValueError("corrected seed schedule source is missing")
    if (
        not qualification_path.is_file()
        or not run_config_path.is_file()
        or not qualification_report_path.is_file()
    ):
        raise ValueError(
            "corrected seed-pool qualification artifact set is missing"
        )

    source_schedule = _read_json(source_schedule_path)
    raw_entries = source_schedule.get("entries")
    if not isinstance(raw_entries, list) or len(raw_entries) != 36:
        raise ValueError(
            "corrected seed selection requires exactly 36 frozen episodes"
        )
    source_entries: list[dict[str, Any]] = []
    source_by_task: dict[str, dict[str, Any]] = {}
    original_seed_by_task: dict[str, int] = {}
    for raw_row in raw_entries:
        if not isinstance(raw_row, dict):
            raise ValueError("corrected seed source schedule row is not an object")
        row = dict(raw_row)
        task_id, solver_seed = _episode_key(row)
        if task_id in source_by_task:
            raise ValueError(
                "corrected seed source schedule contains duplicate tasks"
            )
        group = row.get("schedule_group")
        order = row.get("controller_order")
        if type(group) is not int or group not in range(6):
            raise ValueError(
                "corrected seed source schedule has an invalid order group"
            )
        if (
            not isinstance(order, list)
            or len(order) != len(CONTROLLERS)
            or set(order) != set(CONTROLLERS)
        ):
            raise ValueError(
                "corrected seed source schedule controller order is invalid"
            )
        for field in ("task_id", "map_id", "layout_mode", "source_group"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(
                    f"corrected seed source schedule {field} is invalid"
                )
        if row["source_group"] not in {"generated", "movingai"}:
            raise ValueError(
                "corrected seed source schedule source_group is invalid"
            )
        if type(row.get("agent_count")) is not int or row["agent_count"] <= 0:
            raise ValueError(
                "corrected seed source schedule agent_count is invalid"
            )
        source_entries.append(row)
        source_by_task[task_id] = row
        original_seed_by_task[task_id] = solver_seed
    group_counts = collections.Counter(
        int(row["schedule_group"]) for row in source_entries
    )
    group_orders = {
        group: {
            tuple(map(str, row["controller_order"]))
            for row in source_entries
            if row["schedule_group"] == group
        }
        for group in range(6)
    }
    if group_counts != collections.Counter({group: 6 for group in range(6)}) or any(
        len(orders) != 1 for orders in group_orders.values()
    ):
        raise ValueError(
            "corrected seed source schedule does not preserve balanced groups"
        )
    if len({next(iter(orders)) for orders in group_orders.values()}) != 6:
        raise ValueError(
            "corrected seed source schedule lacks all controller permutations"
        )
    if collections.Counter(
        str(row["source_group"]) for row in source_entries
    ) != {"generated": 18, "movingai": 18}:
        raise ValueError(
            "corrected seed source schedule lacks the frozen source balance"
        )
    if len({str(row["map_id"]) for row in source_entries}) != 36:
        raise ValueError(
            "corrected seed source schedule lacks 36 distinct maps"
        )
    dataset_manifest_path = dataset_root / SPLIT / "manifest.jsonl"
    if not dataset_manifest_path.is_file():
        raise ValueError("corrected seed formal dataset manifest is missing")
    dataset_rows = _read_jsonl(dataset_manifest_path)
    if len(dataset_rows) != 36:
        raise ValueError(
            "corrected seed formal dataset must contain exactly 36 tasks"
        )
    dataset_by_task: dict[str, dict[str, Any]] = {}
    for value in dataset_rows:
        if not isinstance(value, dict):
            raise ValueError(
                "corrected seed formal dataset row is not an object"
            )
        task_id = value.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError(
                "corrected seed formal dataset task_id is invalid"
            )
        if task_id in dataset_by_task:
            raise ValueError(
                "corrected seed formal dataset contains duplicate tasks"
            )
        dataset_by_task[task_id] = value
    if set(dataset_by_task) != set(source_by_task):
        raise ValueError(
            "corrected seed formal dataset tasks differ from source schedule"
        )
    for task_id, source in source_by_task.items():
        dataset_row = dataset_by_task[task_id]
        for field in (
            "map_id",
            "agent_count",
            "layout_mode",
            "source_group",
        ):
            if dataset_row.get(field) != source.get(field):
                raise ValueError(
                    "corrected seed formal dataset differs from source "
                    f"schedule for {task_id}/{field}"
                )

    run_config = _read_json(run_config_path)
    if not isinstance(run_config, dict):
        raise ValueError(
            "corrected seed-pool qualification run_config is not an object"
        )
    configuration = run_config.get("configuration")
    if (
        not isinstance(configuration, dict)
        or _fingerprint(configuration)
        != run_config.get("configuration_fingerprint")
    ):
        raise ValueError(
            "corrected seed-pool qualification configuration identity mismatch"
        )
    environment = configuration.get("environment")
    dataset_design = configuration.get("dataset_design")
    raw_solver_seeds = configuration.get("solver_seeds")
    if (
        not isinstance(raw_solver_seeds, list)
        or len(raw_solver_seeds) != 3
        or any(
            type(seed) is not int or seed < 0 for seed in raw_solver_seeds
        )
        or len(set(raw_solver_seeds)) != len(raw_solver_seeds)
    ):
        raise ValueError(
            "corrected seed-pool run requires exactly three solver seeds"
        )
    solver_seeds = tuple(raw_solver_seeds)
    if (
        configuration.get("stopping_rule") != "wall-clock-fixed-metric"
        or WALL_CLOCK_SAFETY_MAX_DECISIONS != 100_000
        or configuration.get("formal") is not True
        or type(configuration.get("max_decisions")) is not int
        or configuration["max_decisions"] != 0
        or type(configuration.get("metric_iteration_budget")) is not int
        or configuration["metric_iteration_budget"] != 100
        or not isinstance(environment, dict)
        or type(environment.get("max_repair_iterations")) is not int
        or environment["max_repair_iterations"] != 0
        or not _is_exact_number(environment.get("time_limit"), 600.0)
        or not _is_exact_number(
            configuration.get("wall_time_budget_seconds"), 600.0
        )
        or not _is_exact_number(
            configuration.get("episode_process_timeout_seconds"), 660.0
        )
    ):
        raise ValueError(
            "corrected seed-pool qualification does not use the uncapped "
            "fixed-metric contract"
        )
    if (
        not isinstance(dataset_design, dict)
        or dataset_design.get("mode") != "balanced_wall_clock"
        or dataset_design.get("dataset_revision")
        != "balanced-wall-clock-formal-cohort-v6"
        or type(dataset_design.get("map_count")) is not int
        or dataset_design["map_count"] != 36
        or type(dataset_design.get("instance_count")) is not int
        or dataset_design["instance_count"] != 36
        or dataset_design.get("source_counts")
        != {"generated": 18, "movingai": 18}
        or dataset_design.get("formal_dataset_manifest_sha256")
        != sha256_file(dataset_manifest_path)
    ):
        raise ValueError(
            "corrected seed-pool qualification does not bind the frozen "
            "formal dataset manifest"
        )
    registration_reference = dataset_design.get("source_task_registration")
    registration_sha = dataset_design.get(
        "source_task_registration_sha256"
    )
    if (
        not isinstance(registration_reference, str)
        or not registration_reference
        or not _is_sha256(registration_sha)
    ):
        raise ValueError(
            "corrected seed-pool config lacks its source registration identity"
        )
    registration_path = Path(registration_reference)
    if not registration_path.is_absolute():
        registration_path = (
            Path(__file__).resolve().parents[1] / registration_path
        )
    registration_path = registration_path.resolve()
    if (
        not registration_path.is_file()
        or sha256_file(registration_path) != registration_sha
    ):
        raise ValueError(
            "corrected seed-pool source registration SHA differs"
        )
    registration = _read_json(registration_path)
    if (
        registration.get("schema")
        != "lns2.compute_load_balanced_wall_clock_registration.v1"
        or registration.get("selection_blind_to_controller_outcomes")
        is not True
        or registration.get("execution_schedule_sha256")
        != sha256_file(source_schedule_path)
        or registration.get("formal_dataset_manifest_sha256")
        != sha256_file(dataset_manifest_path)
    ):
        raise ValueError(
            "corrected seed-pool source registration does not bind "
            "the schedule and formal dataset"
        )
    run_fingerprint = run_config.get("run_fingerprint")
    dataset_fingerprint = run_config.get("dataset_fingerprint")
    if not _is_sha256(run_fingerprint) or not _is_sha256(dataset_fingerprint):
        raise ValueError(
            "corrected seed-pool run or dataset identity is invalid"
        )
    configured_keys = configuration.get("cohort_job_keys_override")
    expected_keys = {
        (task_id, seed)
        for task_id in source_by_task
        for seed in solver_seeds
    }
    if not isinstance(configured_keys, list):
        raise ValueError(
            "corrected seed-pool qualification lacks cohort job keys"
        )
    normalized_configured_keys: set[tuple[str, int]] = set()
    for value in configured_keys:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or type(value[1]) is not int
        ):
            raise ValueError(
                "corrected seed-pool qualification has invalid cohort job keys"
            )
        normalized_configured_keys.add((value[0], value[1]))
    if (
        len(configured_keys) != len(normalized_configured_keys)
        or normalized_configured_keys != expected_keys
        or len(expected_keys) != 108
    ):
        raise ValueError(
            "corrected seed-pool qualification does not cover the "
            "36-task by 3-seed Cartesian product"
        )
    producer_identity_value = run_config.get("producer_identity")
    if isinstance(producer_identity_value, dict):
        producer_evidence = producer_identity_value
    else:
        implementation = run_config.get("controller_implementation")
        producer_evidence = (
            implementation if isinstance(implementation, dict) else None
        )
    _validate_corrected_producer_identity(
        producer_evidence, label="corrected seed-pool qualification"
    )
    amendment_evidence = {
        "observed_frozen_schedule_qualification": (
            _corrected_amendment_evidence_identity(
                observed_qualification,
                label="observed 36-job qualification",
                expected_count=36,
                dataset_fingerprint=str(dataset_fingerprint),
            )
        ),
        "seed_sensitivity_probe": _corrected_amendment_evidence_identity(
            seed_probe_qualification,
            label="two-job seed probe",
            expected_count=2,
            dataset_fingerprint=str(dataset_fingerprint),
        ),
    }
    pool_native = _native_identity_from_producer(producer_evidence)
    assert pool_native is not None
    expected_native = {
        "sha256": pool_native["sha256"],
        "native_semantics_schema": pool_native["native_semantics_schema"],
        "repair_timing_schema": pool_native["repair_timing_schema"],
    }
    if any(
        value["native"] != expected_native
        for value in amendment_evidence.values()
    ):
        raise ValueError(
            "corrected qualification amendment evidence uses a different "
            "native binary or semantics"
        )

    report = _read_json(qualification_report_path)
    report_gates = report.get("gates")
    report_initial_feasible_count = report.get("initial_feasible_count")
    report_nonzero_state_count = report.get("nonzero_state_count")
    if (
        report.get("schema") != "lns2.closed_loop_confirmation.v1"
        or report.get("passed") is not True
        or type(report.get("valid_count")) is not int
        or report["valid_count"] != len(expected_keys)
        or type(report.get("expected_reset_count")) is not int
        or report["expected_reset_count"] != len(expected_keys)
        or report.get("errors") != []
        or report.get("incomplete_reset_count") != 0
        or report.get("inconsistent_initial_state_count") != 0
        or type(report.get("formal")) is not bool
        or type(report_initial_feasible_count) is not int
        or report_initial_feasible_count < 0
        or type(report_nonzero_state_count) is not int
        or report_nonzero_state_count < 0
        or report_initial_feasible_count + report_nonzero_state_count
        != len(expected_keys)
        or not isinstance(report_gates, dict)
        or not report_gates
        or any(
            type(value) is not bool or value is not True
            for value in report_gates.values()
        )
    ):
        raise ValueError(
            "corrected seed-pool qualification report failed or is inconsistent"
        )
    report_solver_seeds = report.get("registered_solver_seeds")
    if (
        not isinstance(report_solver_seeds, list)
        or report_solver_seeds != list(solver_seeds)
    ):
        raise ValueError(
            "corrected seed-pool qualification report solver seeds differ"
        )

    qualification_rows = _read_jsonl(qualification_path)
    qualification_index: dict[tuple[str, int], dict[str, Any]] = {}
    candidates_by_task: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for raw_row in qualification_rows:
        if not isinstance(raw_row, dict):
            raise ValueError(
                "corrected seed-pool qualification row is not an object"
            )
        key = _episode_key(raw_row)
        if key in qualification_index:
            raise ValueError(
                "corrected seed-pool qualification contains duplicate keys"
            )
        qualification_index[key] = dict(raw_row)
        source = source_by_task.get(key[0])
        if source is None or key[1] not in solver_seeds:
            raise ValueError(
                "corrected seed-pool qualification contains an unknown key"
            )
        candidates_by_task[key[0]].append(
            _corrected_seed_candidate(dict(raw_row), source)
        )
    if set(qualification_index) != expected_keys:
        raise ValueError(
            "corrected seed-pool qualification coverage differs from "
            "the registered Cartesian product"
        )
    actual_repairable_keys = {
        key
        for key, row in qualification_index.items()
        if row.get("initial_complete") is True
        and row.get("initial_feasible") is False
        and type(row.get("initial_conflicts")) is int
        and row["initial_conflicts"] > 0
    }
    actual_initial_feasible_count = sum(
        row.get("initial_feasible") is True
        for row in qualification_index.values()
    )
    if (
        actual_initial_feasible_count != report_initial_feasible_count
        or len(actual_repairable_keys) != report_nonzero_state_count
    ):
        raise ValueError(
            "corrected seed-pool report aggregates differ from reset rows"
        )
    report_repairable_keys = report.get("repairable_episode_keys")
    if (
        not isinstance(report_repairable_keys, list)
        or any(
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or type(value[1]) is not int
            for value in report_repairable_keys
        )
        or {
            (value[0], value[1]) for value in report_repairable_keys
        }
        != actual_repairable_keys
        or len(report_repairable_keys) != len(actual_repairable_keys)
    ):
        raise ValueError(
            "corrected seed-pool report repairable coverage is inconsistent"
        )

    solver_seed_rank = {
        seed: index for index, seed in enumerate(solver_seeds)
    }
    source_tasks = {
        source: [
            row
            for row in source_entries
            if row["source_group"] == source
        ]
        for source in ("generated", "movingai")
    }
    selection_cells = [
        (conflict_name, load_name)
        for conflict_name, _lower, _upper in STRATA
        for load_name, _load_lower, _load_upper in INITIAL_PP_LOAD_STRATA
    ]
    registered_cell_source_counts = collections.Counter(
        (
            str(row["conflict_stratum"]),
            str(row["initial_pp_load_stratum"]),
            str(row["source_group"]),
        )
        for row in source_entries
    )
    if any(
        sum(
            registered_cell_source_counts[
                (conflict_name, load_name, source)
            ]
            for source in ("generated", "movingai")
        )
        != 4
        for conflict_name, load_name in selection_cells
    ):
        raise ValueError(
            "corrected seed source schedule lacks the registered "
            "four-per-cell design"
        )
    required_generated_counts = tuple(
        registered_cell_source_counts[
            (conflict_name, load_name, "generated")
        ]
        for conflict_name, load_name in selection_cells
    )
    generated_states = _best_seed_assignments_by_cell(
        source_tasks["generated"],
        candidates_by_task,
        original_seed_by_task,
        solver_seed_rank,
    )
    movingai_states = _best_seed_assignments_by_cell(
        source_tasks["movingai"],
        candidates_by_task,
        original_seed_by_task,
        solver_seed_rank,
    )
    best: tuple[
        int,
        Fraction,
        tuple[str, ...],
        dict[str, dict[str, Any]],
    ] | None = None
    for generated_counts, generated_value in generated_states.items():
        if generated_counts != required_generated_counts:
            continue
        required_movingai = tuple(4 - count for count in generated_counts)
        if any(count < 0 for count in required_movingai):
            continue
        movingai_value = movingai_states.get(required_movingai)
        if movingai_value is None:
            continue
        assignment = {
            **generated_value[3],
            **movingai_value[3],
        }
        load_overlap = all(
            any(
                row["source_group"] == source
                and row["initial_pp_load_stratum"] == load_name
                for row in assignment.values()
            )
            for source in ("generated", "movingai")
            for load_name, _lower, _upper in INITIAL_PP_LOAD_STRATA
        )
        if not load_overlap:
            continue
        retained = generated_value[0] + movingai_value[0]
        drift = generated_value[1] + movingai_value[1]
        tie_break = tuple(
            str(assignment[str(row["task_id"])]["_tie_break"])
            for row in source_entries
        )
        candidate_value = (retained, drift, tie_break, assignment)
        if (
            best is None
            or candidate_value[0] > best[0]
            or (
                candidate_value[0] == best[0]
                and candidate_value[1] < best[1]
            )
            or (
                candidate_value[0] == best[0]
                and candidate_value[1] == best[1]
                and candidate_value[2] < best[2]
            )
        ):
            best = candidate_value
    selector_identity = {
        "algorithm_schema": "lns2.corrected_native_seed_assignment.v1",
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "tie_break_salt": CORRECTED_SEED_SELECTION_SALT,
    }
    base_provenance = {
        "source_schedule_sha256": sha256_file(source_schedule_path),
        "source_registration_sha256": sha256_file(registration_path),
        "source_registration_execution_schedule_sha256": registration[
            "execution_schedule_sha256"
        ],
        "formal_dataset_manifest_sha256": sha256_file(
            dataset_manifest_path
        ),
        "qualification_manifest_sha256": sha256_file(qualification_path),
        "qualification_run_config_sha256": sha256_file(run_config_path),
        "qualification_report_sha256": sha256_file(
            qualification_report_path
        ),
        "qualification_run_fingerprint": run_fingerprint,
        "qualification_dataset_fingerprint": dataset_fingerprint,
        "qualification_configuration_fingerprint": str(
            run_config["configuration_fingerprint"]
        ),
        "qualification_producer_identity_fingerprint": _fingerprint(
            producer_evidence
        ),
        "native": expected_native,
        "native_semantics_schema": NATIVE_SEMANTICS_SCHEMA,
        "registered_solver_seeds": list(solver_seeds),
        "qualification_job_count": len(expected_keys),
        "selector_identity": selector_identity,
        "protocol_amendment": {
            "schema": "lns2.corrected_native_seed_protocol_amendment.v1",
            "reason": (
                "corrected-native resets changed the registered seed-specific "
                "difficulty cells"
            ),
            "pre_registered_before_observation": False,
            "controller_outcomes_used": False,
            "evidence": amendment_evidence,
        },
        "stopping_contract": {
            "stopping_rule": "wall-clock-fixed-metric",
            "max_decisions": 0,
            "max_repair_iterations": 0,
            "metric_iteration_budget": 100,
            "wall_time_budget_seconds": 600.0,
            "environment_time_limit_seconds": 600.0,
            "episode_process_timeout_seconds": 660.0,
            "safety_max_decisions": 100_000,
            "safety_limit_is_not_metric_cap": True,
        },
    }
    if best is None:
        candidate_task_counts = {
            f"{conflict_name}__{load_name}": len(
                {
                    str(row["task_id"])
                    for values in candidates_by_task.values()
                    for row in values
                    if row["_initial_feasible"] is False
                    and row["conflict_stratum"] == conflict_name
                    and row["initial_pp_load_stratum"] == load_name
                }
            )
            for conflict_name, load_name in selection_cells
        }
        failed_gates = {
            "qualification_cartesian_product_complete": True,
            "registered_cell_source_quotas_satisfiable": False,
            "one_seed_per_frozen_task": False,
            "formal_collection_allowed": False,
        }
        failed_report = {
            "schema": CORRECTED_SEED_SELECTION_SCHEMA,
            "selection_blind_to_controller_outcomes": True,
            "selection_uses_initial_reset_metrics_only": True,
            "qualification_count": len(expected_keys),
            "selected_count": 0,
            "candidate_distinct_task_counts_by_cell": (
                candidate_task_counts
            ),
            "registered_cell_source_quotas": {
                f"{conflict_name}__{load_name}": {
                    source: registered_cell_source_counts[
                        (conflict_name, load_name, source)
                    ]
                    for source in ("generated", "movingai")
                }
                for conflict_name, load_name in selection_cells
            },
            "gates": failed_gates,
            "passed": False,
            "formal_collection_allowed": False,
            "decision": (
                "corrected_native_seed_pool_insufficient_"
                "qualify_broader_task_pool"
            ),
            "execution_schedule_sha256": None,
            "provenance": base_provenance,
        }
        output_root.mkdir(parents=True)
        _write_json(
            output_root / "qualification_rebind_report.json",
            failed_report,
        )
        _write_json(output_root / "cohort_report.json", failed_report)
        return failed_report
    retained_count, total_covariate_drift, _tie_break, selected_by_task = best
    selected_entries: list[dict[str, Any]] = []
    for source in source_entries:
        task_id = str(source["task_id"])
        selected = dict(selected_by_task[task_id])
        selected.pop("_initial_feasible", None)
        selected.pop("_covariate_drift", None)
        selected.pop("_tie_break", None)
        selected_entries.append(selected)
    cell_counts = collections.Counter(
        (
            row["conflict_stratum"],
            row["initial_pp_load_stratum"],
        )
        for row in selected_entries
    )
    source_by_conflict = {
        conflict_name: collections.Counter(
            str(row["source_group"])
            for row in selected_entries
            if row["conflict_stratum"] == conflict_name
        )
        for conflict_name, _lower, _upper in STRATA
    }
    source_by_load = {
        load_name: collections.Counter(
            str(row["source_group"])
            for row in selected_entries
            if row["initial_pp_load_stratum"] == load_name
        )
        for load_name, _lower, _upper in INITIAL_PP_LOAD_STRATA
    }
    gates = {
        "one_seed_per_frozen_task": (
            len(selected_entries) == 36
            and len({str(row["task_id"]) for row in selected_entries}) == 36
        ),
        "all_resets_nonzero_nonextreme": all(
            row["conflict_stratum"] is not None for row in selected_entries
        ),
        "nine_cells_have_four_episodes": all(
            cell_counts[(conflict_name, load_name)] == 4
            for conflict_name, _lower, _upper in STRATA
            for load_name, _load_lower, _load_upper in INITIAL_PP_LOAD_STRATA
        ),
        "source_balance_per_conflict_tier": all(
            source_by_conflict[name] == {"generated": 6, "movingai": 6}
            for name, _lower, _upper in STRATA
        ),
        "registered_cell_source_quotas_preserved": all(
            sum(
                row["conflict_stratum"] == conflict_name
                and row["initial_pp_load_stratum"] == load_name
                and row["source_group"] == source
                for row in selected_entries
            )
            == registered_cell_source_counts[
                (conflict_name, load_name, source)
            ]
            for conflict_name, load_name in selection_cells
            for source in ("generated", "movingai")
        ),
        "source_overlap_per_load_tier": all(
            counts["generated"] > 0 and counts["movingai"] > 0
            for counts in source_by_load.values()
        ),
        "distinct_maps": (
            len({str(row["map_id"]) for row in selected_entries}) == 36
        ),
        "schedule_groups_preserved": all(
            row["schedule_group"]
            == source_by_task[str(row["task_id"])]["schedule_group"]
            for row in selected_entries
        ),
        "controller_orders_preserved": all(
            row["controller_order"]
            == source_by_task[str(row["task_id"])]["controller_order"]
            for row in selected_entries
        ),
    }
    if not all(gates.values()):
        raise AssertionError(
            "corrected seed assignment violated its exact selection constraints"
        )
    provenance = {
        **base_provenance,
        "selection": {
            "schema": "lns2.corrected_native_seed_assignment.v1",
            "controller_outcomes_used": False,
            "input_fields": [
                "source_schedule.solver_seed",
                "source_schedule.initial_conflicts",
                "source_schedule.initial_low_level_generated",
                "source_schedule.conflict_stratum",
                "source_schedule.initial_pp_load_stratum",
                "source_schedule.source_group",
                "initial_complete",
                "initial_feasible",
                "initial_conflicts",
                "initial_complexity.initial_low_level_generated",
                "state_fingerprint",
            ],
            "constraints": {
                "jobs_per_conflict_load_cell": 4,
                "registered_cell_source_quotas": {
                    f"{conflict_name}__{load_name}": {
                        source: registered_cell_source_counts[
                            (conflict_name, load_name, source)
                        ]
                        for source in ("generated", "movingai")
                    }
                    for conflict_name, load_name in selection_cells
                },
                "source_overlap_per_load_tier": True,
                "distinct_maps": 36,
                "one_seed_per_task": True,
            },
            "objective": [
                "maximize_original_solver_seed_retention",
                "minimize_total_relative_conflict_and_pp_load_drift",
                f"fixed_salt_sha256_tiebreak:{CORRECTED_SEED_SELECTION_SALT}",
            ],
        },
    }
    schedule = {
        "schema": CORRECTED_SEED_SCHEDULE_SCHEMA,
        "selection_blind_to_controller_outcomes": True,
        "selection_uses_initial_reset_metrics_only": True,
        "frozen_design": {
            "task_ids": True,
            "maps": True,
            "schedule_groups": True,
            "controller_orders": True,
            "solver_seed_reselected_from_registered_pool": True,
        },
        "provenance": provenance,
        "entries": selected_entries,
    }
    output_root.mkdir(parents=True)
    schedule_path = output_root / "execution_schedule.json"
    _write_json(schedule_path, schedule)
    changed = [
        {
            "task_id": str(row["task_id"]),
            "old_solver_seed": original_seed_by_task[str(row["task_id"])],
            "new_solver_seed": int(row["solver_seed"]),
        }
        for row in selected_entries
        if int(row["solver_seed"])
        != original_seed_by_task[str(row["task_id"])]
    ]
    report_value = {
        "schema": CORRECTED_SEED_SELECTION_SCHEMA,
        "selection_blind_to_controller_outcomes": True,
        "selection_uses_initial_reset_metrics_only": True,
        "qualification_count": len(expected_keys),
        "selected_count": len(selected_entries),
        "original_seed_retained_count": retained_count,
        "total_relative_covariate_drift": float(total_covariate_drift),
        "total_relative_covariate_drift_exact": {
            "numerator": total_covariate_drift.numerator,
            "denominator": total_covariate_drift.denominator,
        },
        "solver_seed_changed_count": len(changed),
        "solver_seed_changes": changed,
        "counts": {
            "cells": {
                f"{left}__{right}": cell_counts[(left, right)]
                for left, _lower, _upper in STRATA
                for right, _load_lower, _load_upper in INITIAL_PP_LOAD_STRATA
            },
            "source_per_conflict_tier": {
                name: dict(source_by_conflict[name])
                for name, _lower, _upper in STRATA
            },
            "source_per_load_tier": {
                name: dict(source_by_load[name])
                for name, _lower, _upper in INITIAL_PP_LOAD_STRATA
            },
        },
        "gates": gates,
        "passed": True,
        "formal_collection_allowed": True,
        "decision": "eligible_for_corrected_native_formal_collection",
        "execution_schedule_sha256": sha256_file(schedule_path),
        "provenance": provenance,
    }
    _write_json(output_root / "qualification_rebind_report.json", report_value)
    _write_json(output_root / "cohort_report.json", report_value)
    return report_value


def rebind_corrected_native_schedule(
    *,
    source_schedule_root: str | Path,
    qualification: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Bind the frozen task/order design to a fresh corrected-native reset.

    Task keys, seeds, order groups, and controller orders remain frozen.  Every
    reset-dependent field is replaced from the new qualification.  A changed
    difficulty cell is reported and can fail the formal balance gate; old
    schedule files are never modified or accepted as the output directory.
    """

    source_root = Path(source_schedule_root).resolve()
    qualification_root = Path(qualification).resolve()
    output_root = Path(output).resolve()
    if output_root == source_root or output_root == qualification_root:
        raise ValueError("corrected schedule output must be a new directory")
    if output_root.exists():
        raise ValueError(
            "corrected schedule output already exists; use a new directory"
        )

    source_schedule_path = source_root / "execution_schedule.json"
    qualification_path = qualification_root / "qualification_manifest.jsonl"
    run_config_path = qualification_root / "run_config.json"
    qualification_report_path = qualification_root / "qualification_report.json"
    if not source_schedule_path.is_file():
        raise ValueError("source execution schedule is missing")
    if (
        not qualification_path.is_file()
        or not run_config_path.is_file()
        or not qualification_report_path.is_file()
    ):
        raise ValueError(
            "fresh corrected-native qualification artifact set is missing"
        )
    source_schedule = _read_json(source_schedule_path)
    source_entries = source_schedule.get("entries")
    if not isinstance(source_entries, list) or len(source_entries) != 36:
        raise ValueError("frozen corrected-native rebind requires exactly 36 episodes")
    source_index: dict[tuple[str, int], dict[str, Any]] = {}
    for raw_row in source_entries:
        if not isinstance(raw_row, dict):
            raise ValueError("source schedule entry is not an object")
        row = dict(raw_row)
        key = _episode_key(row)
        if key in source_index:
            raise ValueError("source schedule contains duplicate episode keys")
        group = row.get("schedule_group")
        if type(group) is not int or group not in range(6):
            raise ValueError("source schedule contains an invalid order group")
        order = row.get("controller_order")
        if (
            not isinstance(order, list)
            or len(order) != len(CONTROLLERS)
            or set(order) != set(CONTROLLERS)
        ):
            raise ValueError("source schedule controller order is invalid")
        source_index[key] = row
    group_counts = collections.Counter(
        int(row["schedule_group"]) for row in source_index.values()
    )
    group_orders = {
        group: {
            tuple(map(str, row["controller_order"]))
            for row in source_index.values()
            if row["schedule_group"] == group
        }
        for group in range(6)
    }
    if group_counts != collections.Counter({group: 6 for group in range(6)}) or any(
        len(orders) != 1 for orders in group_orders.values()
    ):
        raise ValueError("source schedule does not preserve six balanced order groups")
    if len({next(iter(orders)) for orders in group_orders.values()}) != 6:
        raise ValueError("source schedule does not cover all controller permutations")

    run_config = _read_json(run_config_path)
    if not isinstance(run_config, dict):
        raise ValueError("corrected qualification run_config is not an object")
    configuration = run_config.get("configuration")
    if (
        not isinstance(configuration, dict)
        or _fingerprint(configuration)
        != run_config.get("configuration_fingerprint")
    ):
        raise ValueError("corrected qualification configuration identity mismatch")
    run_fingerprint = run_config.get("run_fingerprint")
    dataset_fingerprint = run_config.get("dataset_fingerprint")
    environment = configuration.get("environment")
    if not _is_sha256(run_fingerprint) or not _is_sha256(dataset_fingerprint):
        raise ValueError("corrected qualification run or dataset identity is invalid")
    if (
        configuration.get("stopping_rule") != "wall-clock-fixed-metric"
        or type(configuration.get("max_decisions")) is not int
        or configuration["max_decisions"] != 0
        or type(configuration.get("metric_iteration_budget")) is not int
        or configuration["metric_iteration_budget"] != 100
        or not isinstance(environment, dict)
        or type(environment.get("max_repair_iterations")) is not int
        or environment["max_repair_iterations"] != 0
    ):
        raise ValueError(
            "corrected qualification does not use the uncapped fixed-metric contract"
        )
    configured_keys = configuration.get("cohort_job_keys_override")
    if not isinstance(configured_keys, list):
        raise ValueError("corrected qualification lacks frozen cohort job keys")
    configured_key_set = set()
    for value in configured_keys:
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or type(value[1]) is not int
        ):
            raise ValueError("corrected qualification has invalid cohort job keys")
        configured_key_set.add((value[0], value[1]))
    if (
        len(configured_key_set) != len(configured_keys)
        or configured_key_set != set(source_index)
    ):
        raise ValueError(
            "corrected qualification job keys differ from the frozen schedule"
        )
    producer_identity_value = run_config.get("producer_identity")
    if isinstance(producer_identity_value, dict):
        producer_evidence = producer_identity_value
        native_identity = producer_identity_value.get("native")
    else:
        implementation = run_config.get("controller_implementation")
        producer_evidence = implementation if isinstance(implementation, dict) else None
        native_identity = (
            implementation.get("native_module")
            if isinstance(implementation, dict)
            else None
        )
    _validate_corrected_producer_identity(
        producer_evidence, label="qualification"
    )
    if native_identity is None:
        raise ValueError("qualification corrected-native identity is missing")

    qualification_report = _read_json(qualification_report_path)
    gates = qualification_report.get("gates")
    report_initial_feasible_count = qualification_report.get(
        "initial_feasible_count"
    )
    report_nonzero_state_count = qualification_report.get(
        "nonzero_state_count"
    )
    if (
        qualification_report.get("schema") != "lns2.closed_loop_confirmation.v1"
        or qualification_report.get("passed") is not True
        or type(qualification_report.get("valid_count")) is not int
        or qualification_report["valid_count"] != len(source_index)
        or type(qualification_report.get("expected_reset_count")) is not int
        or qualification_report["expected_reset_count"] != len(source_index)
        or qualification_report.get("errors") != []
        or qualification_report.get("incomplete_reset_count") != 0
        or qualification_report.get("inconsistent_initial_state_count") != 0
        or type(report_initial_feasible_count) is not int
        or report_initial_feasible_count < 0
        or report_initial_feasible_count > len(source_index)
        or type(report_nonzero_state_count) is not int
        or report_nonzero_state_count < 0
        or report_nonzero_state_count > len(source_index)
        or report_initial_feasible_count + report_nonzero_state_count
        != len(source_index)
        or not isinstance(gates, dict)
        or not gates
        or any(type(value) is not bool or value is not True for value in gates.values())
    ):
        raise ValueError(
            "corrected qualification report failed or is inconsistent"
        )
    report_keys = qualification_report.get("repairable_episode_keys")
    if (
        not isinstance(report_keys, list)
        or any(
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(value[0], str)
            or type(value[1]) is not int
            for value in report_keys
        )
        or not {(value[0], value[1]) for value in report_keys}
        <= set(source_index)
        or len(report_keys)
        != len({(value[0], value[1]) for value in report_keys})
        or len(report_keys) != report_nonzero_state_count
    ):
        raise ValueError(
            "corrected qualification report coverage differs from the frozen schedule"
        )

    qualification_rows = _read_jsonl(qualification_path)
    qualification_index: dict[tuple[str, int], dict[str, Any]] = {}
    for raw_row in qualification_rows:
        if not isinstance(raw_row, dict):
            raise ValueError("corrected qualification row is not an object")
        row = dict(raw_row)
        key = _episode_key(row)
        if key in qualification_index:
            raise ValueError("corrected qualification contains duplicate episode keys")
        qualification_index[key] = row
    if set(qualification_index) != set(source_index):
        raise ValueError(
            "corrected qualification coverage differs from the frozen schedule"
        )
    actual_repairable_keys: set[tuple[str, int]] = set()
    actual_initial_feasible_count = 0

    rebound_entries: list[dict[str, Any]] = []
    changed_cells: list[dict[str, Any]] = []
    for key, source in source_index.items():
        qualified = qualification_index[key]
        if (
            qualified.get("status") != "ok"
            or qualified.get("initial_complete") is not True
        ):
            raise ValueError(f"corrected qualification reset is incomplete for {key}")
        for field in ("map_id", "agent_count", "layout_mode"):
            if qualified.get(field) != source.get(field):
                raise ValueError(
                    f"corrected qualification {field} differs from frozen task for {key}"
                )
        complexity = qualified.get("initial_complexity")
        if not isinstance(complexity, dict):
            raise ValueError(
                f"corrected qualification lacks initial complexity for {key}"
            )
        conflicts = qualified.get("initial_conflicts")
        if type(conflicts) is not int or conflicts < 0:
            raise ValueError(
                f"corrected qualification initial conflicts are invalid for {key}"
            )
        initial_feasible = qualified.get("initial_feasible")
        if (
            type(initial_feasible) is not bool
            or initial_feasible is not (conflicts == 0)
        ):
            raise ValueError(
                f"corrected qualification feasibility is inconsistent for {key}"
            )
        if initial_feasible:
            actual_initial_feasible_count += 1
        else:
            actual_repairable_keys.add(key)
        generated = complexity.get("initial_low_level_generated")
        if type(generated) is not int or generated < 0:
            raise ValueError(
                f"corrected qualification initial PP load is invalid for {key}"
            )
        if complexity.get("conflict_pair_count") != conflicts:
            raise ValueError(
                f"corrected qualification complexity conflicts mismatch for {key}"
            )
        state_digest = qualified.get("state_fingerprint")
        if (
            not isinstance(state_digest, str)
            or len(state_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in state_digest
            )
        ):
            raise ValueError(
                f"corrected qualification state fingerprint is invalid for {key}"
            )
        active_ratio = _strict_summary_number(
            complexity, "active_conflict_agent_ratio"
        )
        largest_ratio = _strict_summary_number(
            complexity, "largest_conflict_component_ratio"
        )
        if active_ratio > 1.0 or largest_ratio > 1.0:
            raise ValueError(
                f"corrected qualification complexity ratio exceeds one for {key}"
            )
        conflict_events = _strict_summary_int(
            complexity, "conflict_event_count"
        )
        expanded = _strict_summary_int(
            complexity, "initial_low_level_expanded"
        )
        total_path_cost = _strict_summary_int(complexity, "total_path_cost")
        new_conflict_stratum = conflict_stratum(conflicts)
        new_load_stratum = initial_pp_load_stratum(generated)
        old_cell = (
            source.get("conflict_stratum"),
            source.get("initial_pp_load_stratum"),
        )
        new_cell = (new_conflict_stratum, new_load_stratum)
        if old_cell != new_cell:
            changed_cells.append(
                {
                    "task_id": key[0],
                    "solver_seed": key[1],
                    "old_cell": list(old_cell),
                    "new_cell": list(new_cell),
                }
            )
        rebound_entries.append(
            {
                **source,
                "initial_conflicts": conflicts,
                "state_fingerprint": state_digest,
                "conflict_stratum": new_conflict_stratum,
                "initial_pp_load_stratum": new_load_stratum,
                "active_conflict_agent_ratio": active_ratio,
                "conflict_event_count": conflict_events,
                "initial_low_level_expanded": expanded,
                "initial_low_level_generated": generated,
                "largest_conflict_component_ratio": largest_ratio,
                "total_path_cost": total_path_cost,
            }
        )
    if (
        actual_initial_feasible_count != report_initial_feasible_count
        or len(actual_repairable_keys) != report_nonzero_state_count
        or actual_repairable_keys
        != {(value[0], value[1]) for value in report_keys}
    ):
        raise ValueError(
            "corrected qualification report repairability differs from rows"
        )
    rebound_entries.sort(
        key=lambda row: (
            int(row["schedule_group"]),
            str(row["task_id"]),
            int(row["solver_seed"]),
        )
    )

    cell_counts = collections.Counter(
        (
            row["conflict_stratum"],
            row["initial_pp_load_stratum"],
        )
        for row in rebound_entries
    )
    conflict_counts = collections.Counter(
        row["conflict_stratum"] for row in rebound_entries
    )
    load_counts = collections.Counter(
        row["initial_pp_load_stratum"] for row in rebound_entries
    )
    source_counts = collections.Counter(
        str(row["source_group"]) for row in rebound_entries
    )
    source_by_conflict = {
        stratum: collections.Counter(
            str(row["source_group"])
            for row in rebound_entries
            if row["conflict_stratum"] == stratum
        )
        for stratum, _lower, _upper in STRATA
    }
    source_by_load = {
        stratum: collections.Counter(
            str(row["source_group"])
            for row in rebound_entries
            if row["initial_pp_load_stratum"] == stratum
        )
        for stratum, _lower, _upper in INITIAL_PP_LOAD_STRATA
    }
    gates = {
        "all_resets_nonzero_nonextreme": all(
            row["conflict_stratum"] is not None for row in rebound_entries
        ),
        "nine_cells_have_four_episodes": all(
            cell_counts[(conflict_name, load_name)] == 4
            for conflict_name, _lower, _upper in STRATA
            for load_name, _load_lower, _load_upper in INITIAL_PP_LOAD_STRATA
        ),
        "twelve_per_conflict_tier": all(
            conflict_counts[name] == 12 for name, _lower, _upper in STRATA
        ),
        "twelve_per_load_tier": all(
            load_counts[name] == 12
            for name, _lower, _upper in INITIAL_PP_LOAD_STRATA
        ),
        "source_balance": source_counts == {"generated": 18, "movingai": 18},
        "source_balance_per_conflict_tier": all(
            source_by_conflict[name] == {"generated": 6, "movingai": 6}
            for name, _lower, _upper in STRATA
        ),
        "source_overlap_per_load_tier": all(
            counts["generated"] > 0 and counts["movingai"] > 0
            for counts in source_by_load.values()
        ),
        "distinct_maps": len({str(row["map_id"]) for row in rebound_entries}) == 36,
    }
    formal_allowed = all(gates.values())
    provenance = {
        "source_schedule_sha256": sha256_file(source_schedule_path),
        "qualification_manifest_sha256": sha256_file(qualification_path),
        "qualification_run_config_sha256": sha256_file(run_config_path),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "qualification_run_fingerprint": str(run_config.get("run_fingerprint", "")),
        "qualification_dataset_fingerprint": dataset_fingerprint,
        "qualification_configuration_fingerprint": str(
            run_config["configuration_fingerprint"]
        ),
        "qualification_producer_identity_fingerprint": (
            _fingerprint(producer_evidence)
            if isinstance(producer_evidence, dict)
            else None
        ),
        "native_semantics_schema": NATIVE_SEMANTICS_SCHEMA,
        "stopping_contract": {
            "stopping_rule": "wall-clock-fixed-metric",
            "max_decisions": 0,
            "max_repair_iterations": 0,
            "metric_iteration_budget": 100,
            "wall_time_budget_seconds": configuration.get(
                "wall_time_budget_seconds"
            ),
            "environment_time_limit_seconds": environment.get("time_limit"),
        },
    }
    rebound_schedule = {
        "schema": "lns2.controller_execution_schedule.corrected_native_v1",
        "selection_blind_to_controller_outcomes": True,
        "frozen_design": {
            "task_keys": True,
            "solver_seeds": True,
            "schedule_groups": True,
            "controller_orders": True,
            "reset_dependent_metadata_rebound": True,
        },
        "provenance": provenance,
        "entries": rebound_entries,
    }
    output_root.mkdir(parents=True)
    schedule_path = output_root / "execution_schedule.json"
    _write_json(schedule_path, rebound_schedule)
    report = {
        "schema": "lns2.corrected_native_schedule_rebind.v1",
        "selection_blind_to_controller_outcomes": True,
        "selected_count": len(rebound_entries),
        "changed_cell_count": len(changed_cells),
        "changed_cells": changed_cells,
        "counts": {
            "cells": {
                f"{left}__{right}": count
                for (left, right), count in sorted(
                    cell_counts.items(), key=lambda item: str(item[0])
                )
            },
            "conflict_tiers": {
                (
                    "zero_or_over_500"
                    if name is None
                    else str(name)
                ): count
                for name, count in sorted(
                    conflict_counts.items(), key=lambda item: str(item[0])
                )
            },
            "load_tiers": dict(sorted(load_counts.items(), key=lambda item: str(item[0]))),
            "sources": dict(sorted(source_counts.items())),
        },
        "gates": gates,
        "passed": formal_allowed,
        "formal_collection_allowed": formal_allowed,
        "decision": (
            "eligible_for_corrected_native_formal_collection"
            if formal_allowed
            else "corrected_native_reset_changed_balance_reselect_required"
        ),
        "execution_schedule_sha256": sha256_file(schedule_path),
        "provenance": provenance,
    }
    _write_json(output_root / "qualification_rebind_report.json", report)
    _write_json(output_root / "cohort_report.json", report)
    return report


def prepare_corrected_native_formal_config(
    *,
    base_config: str | Path,
    schedule_root: str | Path,
    dataset: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Bind an uncapped corrected-native config to a materialized 36-task set."""

    from experiments.corrected_native_full_pool import (
        validate_corrected_native_full_pool_schedule,
    )

    base_path = Path(base_config).resolve()
    schedule_root_path = Path(schedule_root).resolve()
    dataset_root = Path(dataset).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise ValueError(
            "corrected-native formal config output must be a new file"
        )
    base = _read_json(base_path)
    try:
        serialized = json.dumps(base, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("corrected-native base config is not finite JSON") from error
    base = json.loads(serialized)
    environment = base.get("environment")
    if (
        type(base.get("schema_version")) is not int
        or base["schema_version"] != 1
        or base.get("formal") is not True
        or base.get("split") != SPLIT
        or base.get("solver_seeds") != [1, 2, 3]
        or base.get("policies") != ["official_adaptive", "realized_dynamic"]
        or not isinstance(environment, dict)
        or type(environment.get("max_repair_iterations")) is not int
        or environment["max_repair_iterations"] != 0
        or not _is_exact_number(environment.get("time_limit"), 600.0)
        or type(base.get("max_decisions")) is not int
        or base["max_decisions"] != 0
        or type(base.get("metric_iteration_budget")) is not int
        or base["metric_iteration_budget"] != 100
        or not _is_exact_number(
            base.get("wall_time_budget_seconds"), 600.0
        )
        or not _is_exact_number(
            base.get("episode_process_timeout_seconds"), 660.0
        )
        or base.get("stopping_rule", "wall-clock-fixed-metric")
        != "wall-clock-fixed-metric"
    ):
        raise ValueError(
            "corrected-native base config must use the uncapped "
            "0/0/100/600/660 contract"
        )
    validated = validate_corrected_native_full_pool_schedule(
        schedule_root_path,
        dataset=dataset_root,
    )
    provenance = validated["provenance"]
    expected_dataset_fingerprint = provenance.get(
        "formal_dataset_fingerprint"
    )
    if (
        not _is_sha256(expected_dataset_fingerprint)
        or validated.get("formal_dataset_fingerprint")
        != expected_dataset_fingerprint
        or _dataset_fingerprint(dataset_root)
        != expected_dataset_fingerprint
    ):
        raise ValueError(
            "derived schedule is not bound to the supplied formal dataset"
        )
    manifest_path = dataset_root / SPLIT / "manifest.jsonl"
    summary_path = dataset_root / "dataset_summary.json"
    materialization_path = schedule_root_path / "materialization_report.json"
    if (
        not manifest_path.is_file()
        or not summary_path.is_file()
        or not materialization_path.is_file()
    ):
        raise ValueError("derived corrected-native dataset artifacts are incomplete")
    manifest_rows = _read_jsonl(manifest_path)
    if any(not isinstance(row, dict) for row in manifest_rows):
        raise ValueError("derived corrected-native manifest row is not an object")
    task_ids = [str(row.get("task_id", "")) for row in manifest_rows]
    map_ids = [str(row.get("map_id", "")) for row in manifest_rows]
    sources = collections.Counter(
        str(row.get("source_group", "")) for row in manifest_rows
    )
    layouts = collections.Counter(
        str(row.get("layout_mode", "")) for row in manifest_rows
    )
    if (
        len(manifest_rows) != 36
        or not all(task_ids)
        or not all(map_ids)
        or len(set(task_ids)) != 36
        or len(set(map_ids)) != 36
        or sources != {"generated": 18, "movingai": 18}
        or not layouts
        or "" in layouts
    ):
        raise ValueError(
            "derived corrected-native dataset is not the registered "
            "36-task, 36-map, 18/18 design"
        )
    summary = _read_json(summary_path)
    split_summary = summary.get("splits", {}).get(SPLIT)
    if (
        not isinstance(split_summary, dict)
        or split_summary.get("map_count") != 36
        or split_summary.get("instance_count") != 36
        or dict(split_summary.get("source_counts", {}))
        != dict(sorted(sources.items()))
        or dict(split_summary.get("layout_counts", {}))
        != dict(sorted(layouts.items()))
    ):
        raise ValueError(
            "derived corrected-native dataset summary differs from its manifest"
        )
    historical_map_ids = list(
        dict(base.get("dataset_design", {})).get(
            "historical_map_ids", []
        )
    )
    base["formal"] = True
    base["experiment_revision"] = "corrected-native-selected-v3"
    base["stopping_rule"] = "wall-clock-fixed-metric"
    base["max_decisions"] = 0
    base["metric_iteration_budget"] = 100
    base["wall_time_budget_seconds"] = 600.0
    base["episode_process_timeout_seconds"] = 660.0
    base["environment"] = {
        **environment,
        "max_repair_iterations": 0,
        "time_limit": 600.0,
    }
    base["dataset_design"] = {
        "mode": SPLIT,
        "dataset_revision": str(
            summary.get(
                "dataset_revision",
                "balanced-wall-clock-corrected-native-selected-v3",
            )
        ),
        "formal_dataset_manifest": str(manifest_path),
        "formal_dataset_manifest_sha256": sha256_file(manifest_path),
        "formal_dataset_fingerprint": expected_dataset_fingerprint,
        "formal_dataset_summary_sha256": sha256_file(summary_path),
        "formal_materialization_report_sha256": sha256_file(
            materialization_path
        ),
        "formal_execution_schedule_sha256": validated[
            "execution_schedule_sha256"
        ],
        "formal_cohort_report_sha256": validated["cohort_report_sha256"],
        "map_count": 36,
        "instance_count": 36,
        "source_counts": {"generated": 18, "movingai": 18},
        "layout_counts": dict(sorted(layouts.items())),
        "historical_map_ids": historical_map_ids,
    }
    _write_json(output_path, base)
    return {
        "schema": "lns2.corrected_native_formal_config_preparation.v3",
        "passed": True,
        "config": str(output_path),
        "config_sha256": sha256_file(output_path),
        "base_config_sha256": sha256_file(base_path),
        "execution_schedule_sha256": validated[
            "execution_schedule_sha256"
        ],
        "cohort_report_sha256": validated["cohort_report_sha256"],
        "formal_dataset_fingerprint": expected_dataset_fingerprint,
        "max_decisions": 0,
        "max_repair_iterations": 0,
        "metric_iteration_budget": 100,
        "metric_iteration_budget_is_execution_cap": False,
        "wall_time_budget_seconds": 600.0,
        "episode_process_timeout_seconds": 660.0,
    }


def collect_scheduled(
    *,
    dataset: str | Path,
    config: str | Path,
    qualification: str | Path | None,
    schedule_root: str | Path,
    output: str | Path,
    original_bundle: str | Path,
    mixed_bundle: str | Path,
    resume: bool,
    dry_run: bool = False,
    registration: str | Path | None = None,
    stopping_rule: str = "wall-clock-fixed-metric",
) -> dict[str, Any]:
    if stopping_rule != "wall-clock-fixed-metric":
        raise ValueError(
            "formal balanced wall-clock collection requires "
            "wall-clock-fixed-metric: repair execution is unbounded while "
            "the frozen 100-step AUC remains diagnostic"
        )
    schedule_root_path = Path(schedule_root).resolve()
    cohort_report_path = schedule_root_path / "cohort_report.json"
    cohort_report = _read_json(cohort_report_path)
    if cohort_report.get("formal_collection_allowed") is not True:
        raise ValueError("balanced wall-clock cohort did not pass the preregistered data gate")
    schedule_path = schedule_root_path / "execution_schedule.json"
    schedule = _read_json(schedule_path)
    if schedule.get("schema") in LEGACY_CORRECTED_FULL_POOL_SCHEDULE_SCHEMAS:
        raise ValueError(
            "corrected-native full-pool v1/v2 artifacts are read-only; "
            "create a fresh v3 schedule for formal collection"
        )
    corrected_schedule = schedule.get("schema") in CORRECTED_SCHEDULE_SCHEMAS
    corrected_full_pool_schedule = (
        schedule.get("schema") == CORRECTED_FULL_POOL_SCHEDULE_SCHEMA
    )
    qualification_root = (
        Path(qualification).resolve() if qualification is not None else None
    )
    if corrected_schedule:
        if registration is not None:
            raise ValueError(
                "corrected-native schedule must use its rebind provenance, "
                "not the legacy cohort registration"
            )
        _corrected_schedule_report(
            schedule_root_path,
            schedule_path,
            schedule,
            dataset=dataset if corrected_full_pool_schedule else None,
        )
        provenance = schedule.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("native_semantics_schema")
            != NATIVE_SEMANTICS_SCHEMA
            or not provenance.get(
                "qualification_producer_identity_fingerprint"
            )
        ):
            raise ValueError(
                "corrected-native schedule lacks producer/native provenance"
            )
        if corrected_full_pool_schedule:
            if (
                not _is_sha256(
                    provenance.get("formal_dataset_fingerprint")
                )
                or _dataset_fingerprint(Path(dataset).resolve())
                != provenance["formal_dataset_fingerprint"]
            ):
                raise ValueError(
                    "full-pool schedule is not bound to the supplied derived dataset"
                )
            _validate_corrected_full_pool_collection_preflight(
                config,
                provenance,
            )
        else:
            if qualification_root is None:
                raise ValueError(
                    "legacy corrected-native schedule requires its "
                    "qualification artifact root"
                )
            qualification_manifest = (
                qualification_root / "qualification_manifest.jsonl"
            )
            qualification_run_config = qualification_root / "run_config.json"
            qualification_report = (
                qualification_root / "qualification_report.json"
            )
            if (
                not qualification_manifest.is_file()
                or not qualification_run_config.is_file()
                or not qualification_report.is_file()
                or sha256_file(qualification_manifest)
                != provenance.get("qualification_manifest_sha256")
                or sha256_file(qualification_run_config)
                != provenance.get("qualification_run_config_sha256")
                or sha256_file(qualification_report)
                != provenance.get("qualification_report_sha256")
            ):
                raise ValueError(
                    "corrected-native qualification differs from schedule provenance"
                )
            _validate_corrected_seed_schedule_semantics(
                schedule, _read_jsonl(qualification_manifest)
            )
    elif qualification_root is None:
        raise ValueError("balanced collection requires a qualification artifact root")
    registration_report = (
        verify_compute_load_cohort_registration(registration)
        if registration is not None
        else None
    )
    entries = list(schedule["entries"])
    output_root = Path(output).resolve()
    protected_roots = {
        Path(dataset).resolve(),
        Path(config).resolve(),
        schedule_root_path,
        Path(original_bundle).resolve(),
        Path(mixed_bundle).resolve(),
    }
    if qualification_root is not None:
        protected_roots.add(qualification_root)
    if output_root in protected_roots:
        raise ValueError(
            "formal collection output must differ from every input artifact"
        )
    progress = []
    bundles = {
        "official_adaptive": original_bundle,
        "v2-full": original_bundle,
        "mixed-full-v2": mixed_bundle,
    }
    phases = {
        "official_adaptive": "official_adaptive",
        "v2-full": "realized_dynamic",
        "mixed-full-v2": "realized_dynamic",
    }
    for group in range(6):
        group_rows = [row for row in entries if int(row["schedule_group"]) == group]
        if not group_rows:
            raise ValueError(f"execution schedule group is empty: {group}")
        order = tuple(map(str, group_rows[0]["controller_order"]))
        if any(tuple(map(str, row["controller_order"])) != order for row in group_rows):
            raise ValueError("execution schedule group has inconsistent controller order")
        keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in group_rows}
        task_ids = sorted({task_id for task_id, _seed in keys})
        for controller_id in order:
            lane = output_root / f"order_{group}" / controller_id
            common = dict(
                dataset=dataset,
                config_path=config,
                output=lane,
                workers=1,
                task_ids=task_ids,
                controller="v2-full",
                feature_backend="auto",
                controller_bundle=bundles[controller_id],
                controller_runtime="optimized",
                verification_profile="deployment",
                job_keys=keys,
                cohort_job_keys=keys,
                stopping_rule=stopping_rule,
                use_global_collection_lock=False,
            )
            if dry_run:
                result = run_closed_loop_collection(
                    **common, phase=phases[controller_id], dry_run=True
                )
            else:
                qualification_manifest_exists = (
                    lane / "qualification_manifest.jsonl"
                ).is_file()
                run_config_exists = (lane / "run_config.json").is_file()
                if qualification_manifest_exists and not run_config_exists:
                    raise ValueError(
                        "formal lane has a qualification manifest without its "
                        f"run identity; preserve the lane and use a new output: {lane}"
                    )
                if not qualification_manifest_exists:
                    reusable_qualification = (
                        qualification_root
                        if (
                            not corrected_full_pool_schedule
                            and qualification_root is not None
                            and (qualification_root / "run_config.json").is_file()
                        )
                        else None
                    )
                    run_closed_loop_collection(
                        **common,
                        phase="qualify",
                        qualification_source=reusable_qualification,
                        resume=run_config_exists,
                    )
                result = run_closed_loop_collection(
                    **common,
                    phase=phases[controller_id],
                    resume=(lane / "run_config.json").is_file() or resume,
                )
            progress.append(
                {
                    "group": group,
                    "controller": controller_id,
                    "job_count": len(keys),
                    "dry_run": dry_run,
                    "summary": result,
                }
            )
            _write_json(output_root / "collection_progress.json", {"entries": progress})
    return {
        "group_count": 6,
        "lane_count": len(progress),
        "registration": registration_report,
        "entries": progress,
    }


def recover_scheduled_partial_traces(
    collection: str | Path, schedule_root: str | Path
) -> dict[str, Any]:
    collection_root = Path(collection).resolve()
    schedule_root_path = Path(schedule_root).resolve()
    schedule_path = schedule_root_path / "execution_schedule.json"
    schedule = _read_json(schedule_path)
    _corrected_schedule_report(schedule_root_path, schedule_path, schedule)
    expected = {_episode_key(row) for row in schedule["entries"]}
    recovered: list[dict[str, Any]] = []
    remaining_errors: list[dict[str, Any]] = []
    for group in range(6):
        for controller in CONTROLLERS:
            phase = (
                "official_adaptive"
                if controller == "official_adaptive"
                else "realized_dynamic"
            )
            lane = collection_root / f"order_{group}" / controller
            manifest_path = lane / f"{phase}_manifest.jsonl"
            rows = _read_jsonl(manifest_path)
            changed = False
            run_config = _read_json(lane / "run_config.json")
            metric_budget = run_config["configuration"].get(
                "metric_iteration_budget"
            )
            for index, row in enumerate(rows):
                if str(row.get("status")) in {"ok", "resumed"}:
                    continue
                known_timing_rejection = (
                    str(row.get("error_kind")) == "ClosedLoopTraceError"
                    and str(row.get("error", "")).endswith(
                        "repair timing v2 episode delta is below native step"
                    )
                )
                partial_reference = row.get("partial_trace_file")
                if not known_timing_rejection or not partial_reference:
                    remaining_errors.append(
                        {
                            "controller": controller,
                            "task_id": str(row.get("task_id")),
                            "solver_seed": int(row.get("solver_seed", -1)),
                            "error": row.get("error"),
                        }
                    )
                    continue
                partial = (lane / str(partial_reference)).resolve()
                try:
                    partial.relative_to(lane)
                except ValueError as error:
                    raise ValueError("partial trace escapes its collection lane") from error
                if not partial.name.endswith(".partial"):
                    raise ValueError("registered partial trace lacks .partial suffix")
                final = partial.with_name(partial.name[: -len(".partial")])
                source = partial if partial.is_file() else final
                if not source.is_file():
                    raise ValueError(f"registered partial trace is missing: {partial}")
                validated = validate_closed_loop_trace(
                    source,
                    str(run_config["run_fingerprint"]),
                    expected_episode_id=str(row["episode_id"]),
                    expected_policy=str(row["policy"]),
                    expected_solver_seed=int(row["solver_seed"]),
                    metric_iteration_budget=(
                        int(metric_budget) if metric_budget is not None else None
                    ),
                    collection_root=lane,
                )
                if source == partial:
                    if final.exists():
                        raise ValueError(f"partial recovery target already exists: {final}")
                    os.replace(partial, final)
                metadata = trace_file_metadata(final)
                original_error = str(row.get("error"))
                rows[index] = {
                    **row,
                    "trace_file": final.relative_to(lane).as_posix(),
                    "trace_event_count": int(validated["event_count"]),
                    "initial_state_ref": validated.get("initial_state_ref"),
                    **metadata,
                    "status": "ok",
                    "summary": validated["summary"],
                    "partial_trace_file": None,
                    "error_kind": None,
                    "error": None,
                    "recovery": {
                        "reason": "corrected_repair_timing_v2_boundary_validation",
                        "original_error": original_error,
                        "solver_was_not_rerun": True,
                    },
                }
                recovered.append(
                    {
                        "controller": controller,
                        "task_id": str(row["task_id"]),
                        "solver_seed": int(row["solver_seed"]),
                        "trace_sha256": metadata["trace_sha256"],
                    }
                )
                changed = True
            if changed:
                _write_jsonl_atomic(manifest_path, rows)
    observed = {
        _episode_key(row)
        for group in range(6)
        for controller in CONTROLLERS
        for row in _read_jsonl(
            collection_root
            / f"order_{group}"
            / controller
            / (
                "official_adaptive_manifest.jsonl"
                if controller == "official_adaptive"
                else "realized_dynamic_manifest.jsonl"
            )
        )
    }
    if observed != expected:
        raise ValueError("partial recovery changed formal cohort coverage")
    report = {
        "schema": "lns2.balanced_wall_clock_partial_recovery.v1",
        "recovered_count": len(recovered),
        "recovered": recovered,
        "remaining_error_count": len(remaining_errors),
        "remaining_errors": remaining_errors,
        "solver_rerun_count": 0,
        "passed": bool(recovered) and not remaining_errors,
    }
    _write_json(collection_root / "partial_trace_recovery_report.json", report)
    return report


def _mean(values: Iterable[float]) -> float:
    rows = list(map(float, values))
    return statistics.fmean(rows) if rows else 0.0


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _relative_improvement(baseline: float, candidate: float) -> float:
    if abs(baseline) <= 1e-15:
        return 0.0 if abs(candidate) <= 1e-15 else -math.inf
    return 1.0 - candidate / baseline


def _episode_key(row: dict[str, Any]) -> tuple[str, int]:
    task_id = row.get("task_id")
    solver_seed = row.get("solver_seed")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("episode task_id must be a non-empty string")
    if type(solver_seed) is not int or solver_seed < 0:
        raise ValueError("episode solver_seed must be a non-negative integer")
    return task_id, solver_seed


def _strict_summary_bool(
    summary: dict[str, Any], field: str
) -> bool:
    value = summary.get(field)
    if type(value) is not bool:
        raise ValueError(f"episode summary {field} must be a boolean")
    return value


def _strict_summary_int(
    summary: dict[str, Any], field: str, *, minimum: int = 0
) -> int:
    value = summary.get(field)
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"episode summary {field} must be an integer >= {minimum}"
        )
    return value


def _strict_summary_number(
    summary: dict[str, Any],
    field: str,
    *,
    default: float | None = None,
    minimum: float = 0.0,
) -> float:
    value = summary.get(field, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"episode summary {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(
            f"episode summary {field} must be finite and >= {minimum}"
        )
    return result


def _scientific_summary(row: dict[str, Any]) -> dict[str, Any]:
    raw_summary = row.get("summary")
    if not isinstance(raw_summary, dict):
        raise ValueError("formal episode lacks a summary object")
    summary = dict(raw_summary)
    raw_totals = summary.get("controller_totals", {})
    if not isinstance(raw_totals, dict):
        raise ValueError("episode summary controller_totals must be an object")
    totals = dict(raw_totals)
    final_low_level = summary.get("final_low_level")
    if not isinstance(final_low_level, dict):
        raise ValueError("episode summary final_low_level must be an object")
    return {
        "success": _strict_summary_bool(summary, "success"),
        "capped_wall_ttf": _strict_summary_number(
            summary, "capped_wall_time_to_feasible"
        ),
        "fixed_auc": _strict_summary_number(
            summary, "fixed_budget_conflict_auc"
        ),
        "normalized_fixed_auc": _strict_summary_number(
            summary, "normalized_fixed_budget_conflict_auc"
        ),
        "repair_iterations": _strict_summary_int(summary, "repair_iterations"),
        "generated_nodes": _strict_summary_int(final_low_level, "generated"),
        "expanded_nodes": _strict_summary_int(final_low_level, "expanded"),
        "repair_wall_seconds": _strict_summary_number(
            summary, "repair_wall_seconds", default=0.0
        ),
        "environment_construct_seconds": _strict_summary_number(
            summary, "environment_construct_seconds", default=0.0
        ),
        "reset_wall_seconds": _strict_summary_number(
            summary, "reset_wall_seconds", default=0.0
        ),
        "episode_observed_wall_seconds": _strict_summary_number(
            summary, "episode_observed_wall_seconds", default=0.0
        ),
        "proposal_seconds": _strict_summary_number(
            totals, "proposal_seconds", default=0.0
        ),
        "feature_seconds": _strict_summary_number(
            totals, "feature_seconds", default=0.0
        ),
        "inference_seconds": _strict_summary_number(
            totals, "inference_seconds", default=0.0
        ),
        "fingerprint_seconds": _strict_summary_number(
            totals, "state_fingerprint_seconds", default=0.0
        ),
        "pp_replan_seconds": _strict_summary_number(
            totals, "pp_replan_seconds", default=0.0
        ),
        "controller_before_repair_seconds": _strict_summary_number(
            totals, "controller_seconds_before_repair", default=0.0
        ),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_scientific_summary(row) for row in rows]
    numeric_fields = (
        "capped_wall_ttf",
        "fixed_auc",
        "normalized_fixed_auc",
        "repair_iterations",
        "generated_nodes",
        "expanded_nodes",
        "repair_wall_seconds",
        "environment_construct_seconds",
        "reset_wall_seconds",
        "episode_observed_wall_seconds",
        "proposal_seconds",
        "feature_seconds",
        "inference_seconds",
        "fingerprint_seconds",
        "pp_replan_seconds",
        "controller_before_repair_seconds",
    )
    return {
        "episode_count": len(rows),
        "success_count": sum(value["success"] for value in values),
        **{f"mean_{field}": _mean(value[field] for value in values) for field in numeric_fields},
        "invalid_action_count": sum(
            _strict_summary_int(row["summary"], "invalid_action_count")
            for row in rows
        ),
        "fingerprint_mismatch_count": sum(
            _strict_summary_int(row["summary"], "fingerprint_mismatch_count")
            for row in rows
        ),
        "error_count": sum(str(row.get("status")) not in {"ok", "resumed"} for row in rows),
    }


def _paired_bootstrap_by_map(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    *,
    metric: str,
    samples: int = 5000,
    seed: int = 20270831,
) -> dict[str, Any]:
    by_map: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for key in sorted(baseline):
        left, right = baseline[key], candidate[key]
        if str(left["map_id"]) != str(right["map_id"]):
            raise ValueError("paired controller rows disagree on map id")
        by_map[str(left["map_id"])].append(
            (
                float(_scientific_summary(left)[metric]),
                float(_scientific_summary(right)[metric]),
            )
        )
    map_ids = sorted(by_map)
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [generator.choice(map_ids) for _ in map_ids]
        left_values = [value[0] for map_id in selected for value in by_map[map_id]]
        right_values = [value[1] for map_id in selected for value in by_map[map_id]]
        estimates.append(_relative_improvement(_mean(left_values), _mean(right_values)))
    left_mean = _mean(value[0] for values in by_map.values() for value in values)
    right_mean = _mean(value[1] for values in by_map.values() for value in values)
    return {
        "map_count": len(map_ids),
        "samples": samples,
        "seed": seed,
        "baseline_mean": left_mean,
        "candidate_mean": right_mean,
        "relative_improvement": _relative_improvement(left_mean, right_mean),
        "improvement_95_ci": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
    }


def _paired_group_comparison(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    schedule_index: dict[tuple[str, int], dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    for key in sorted(baseline):
        groups[str(schedule_index[key][field])].append((baseline[key], candidate[key]))
    result = {}
    for name, pairs in sorted(groups.items()):
        left_values = [_scientific_summary(left) for left, _right in pairs]
        right_values = [_scientific_summary(right) for _left, right in pairs]
        left_ttf = _mean(row["capped_wall_ttf"] for row in left_values)
        right_ttf = _mean(row["capped_wall_ttf"] for row in right_values)
        left_auc = _mean(row["fixed_auc"] for row in left_values)
        right_auc = _mean(row["fixed_auc"] for row in right_values)
        result[name] = {
            "episode_count": len(pairs),
            "baseline_success_count": sum(row["success"] for row in left_values),
            "candidate_success_count": sum(row["success"] for row in right_values),
            "ttf_improvement": _relative_improvement(left_ttf, right_ttf),
            "auc_improvement": _relative_improvement(left_auc, right_auc),
            "candidate_not_worse": (
                sum(row["success"] for row in right_values)
                >= sum(row["success"] for row in left_values)
                and right_ttf <= left_ttf
            ),
        }
    return result


def _successful_ttf(row: dict[str, Any]) -> float | None:
    summary = row.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("formal episode lacks a summary object")
    success = _strict_summary_bool(summary, "success")
    value = summary.get("wall_time_to_feasible")
    if not success:
        if value is not None:
            raise ValueError(
                "failed episode must use null wall_time_to_feasible"
            )
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("successful episode lacks numeric wall_time_to_feasible")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("successful episode has invalid wall_time_to_feasible")
    return result


def _success_ttf_summary(values: Iterable[float]) -> dict[str, Any]:
    rows = list(map(float, values))
    if not rows:
        return {
            "success_count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "success_count": len(rows),
        "mean_seconds": _mean(rows),
        "median_seconds": statistics.median(rows),
        "p95_seconds": _quantile(rows, 0.95),
        "min_seconds": min(rows),
        "max_seconds": max(rows),
    }


def _paired_success_ttf(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    schedule_index: dict[tuple[str, int], dict[str, Any]],
    keys: Iterable[tuple[str, int]],
    *,
    samples: int = 5000,
    seed: int = 20270831,
) -> dict[str, Any]:
    pairs = []
    for key in sorted(keys):
        baseline_ttf = _successful_ttf(baseline[key])
        candidate_ttf = _successful_ttf(candidate[key])
        if baseline_ttf is None or candidate_ttf is None:
            continue
        pairs.append(
            {
                "map_id": str(schedule_index[key]["map_id"]),
                "baseline": baseline_ttf,
                "candidate": candidate_ttf,
            }
        )
    baseline_values = [float(row["baseline"]) for row in pairs]
    candidate_values = [float(row["candidate"]) for row in pairs]
    if not pairs:
        return {
            "common_success_count": 0,
            "baseline": _success_ttf_summary([]),
            "candidate": _success_ttf_summary([]),
            "mean_improvement": None,
            "median_improvement": None,
            "mean_paired_delta_seconds": None,
            "candidate_faster_count": 0,
            "baseline_faster_count": 0,
            "tie_count": 0,
            "map_bootstrap": {
                "map_count": 0,
                "samples": samples,
                "seed": seed,
                "improvement_95_ci": [None, None],
            },
        }

    by_map: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for row in pairs:
        by_map[str(row["map_id"])].append(
            (float(row["baseline"]), float(row["candidate"]))
        )
    map_ids = sorted(by_map)
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [generator.choice(map_ids) for _ in map_ids]
        left = [pair[0] for map_id in selected for pair in by_map[map_id]]
        right = [pair[1] for map_id in selected for pair in by_map[map_id]]
        estimates.append(_relative_improvement(_mean(left), _mean(right)))

    baseline_mean = _mean(baseline_values)
    candidate_mean = _mean(candidate_values)
    baseline_median = statistics.median(baseline_values)
    candidate_median = statistics.median(candidate_values)
    return {
        "common_success_count": len(pairs),
        "baseline": _success_ttf_summary(baseline_values),
        "candidate": _success_ttf_summary(candidate_values),
        "mean_improvement": _relative_improvement(
            baseline_mean, candidate_mean
        ),
        "median_improvement": _relative_improvement(
            baseline_median, candidate_median
        ),
        "mean_paired_delta_seconds": _mean(
            left - right
            for left, right in zip(baseline_values, candidate_values)
        ),
        "candidate_faster_count": sum(
            right < left
            for left, right in zip(baseline_values, candidate_values)
        ),
        "baseline_faster_count": sum(
            left < right
            for left, right in zip(baseline_values, candidate_values)
        ),
        "tie_count": sum(
            left == right
            for left, right in zip(baseline_values, candidate_values)
        ),
        "map_bootstrap": {
            "map_count": len(map_ids),
            "samples": samples,
            "seed": seed,
            "improvement_95_ci": [
                _quantile(estimates, 0.025),
                _quantile(estimates, 0.975),
            ],
        },
    }


def _success_ttf_markdown(report: dict[str, Any]) -> str:
    def seconds(value: Any) -> str:
        return (
            "NA"
            if value is None
            else f"{float(value):.3f}"
        )

    def percentage(value: Any) -> str:
        return (
            "NA"
            if value is None
            else f"{float(value) * 100:.2f}%"
        )

    def interval_text(values: Any) -> str:
        if (
            not isinstance(values, list)
            or len(values) != 2
            or values[0] is None
            or values[1] is None
        ):
            return "NA"
        return (
            f"[{float(values[0]) * 100:.2f}%, "
            f"{float(values[1]) * 100:.2f}%]"
        )

    lines = [
        "# 去除失败惩罚后的 TTF 对比",
        "",
        "失败 episode 的 TTF 记为 `FAIL/NA`，不使用 600 秒 cap，也不把未到达可行解的"
        "实际停止时间当作 TTF。主要速度口径是同一实例上双方都成功的配对比较。",
        "",
        "## 各自成功样本",
        "",
        "| 控制器 | 成功 | 平均 TTF (s) | 中位数 | P95 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for controller in CONTROLLERS:
        row = report["success_only"][controller]
        lines.append(
            f"| {controller} | {row['success_count']}/{report['episode_count']} | "
            f"{seconds(row['mean_seconds'])} | "
            f"{seconds(row['median_seconds'])} | "
            f"{seconds(row['p95_seconds'])} |"
        )
    lines.extend(
        [
            "",
            "## 共同成功配对",
            "",
            "| 对比 | N | 基线均值 | 候选均值 | 均值改善 | 95% CI | 候选更快 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, pair in report["paired"]["overall"].items():
        interval = pair["map_bootstrap"]["improvement_95_ci"]
        lines.append(
            f"| {name} | {pair['common_success_count']} | "
            f"{seconds(pair['baseline']['mean_seconds'])} | "
            f"{seconds(pair['candidate']['mean_seconds'])} | "
            f"{percentage(pair['mean_improvement'])} | "
            f"{interval_text(interval)} | "
            f"{pair['candidate_faster_count']}/{pair['common_success_count']} |"
        )
    lines.extend(
        [
            "",
            "## 分层配对结果",
            "",
            "| 分层 | 组 | 对比 | N | 均值改善 | 95% CI |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for field in (
        "conflict_stratum",
        "initial_pp_load_stratum",
        "source_group",
        "conflict_load_cell",
    ):
        for group, comparisons in report["paired"][field].items():
            for name, pair in comparisons.items():
                interval = pair["map_bootstrap"]["improvement_95_ci"]
                improvement = pair["mean_improvement"]
                improvement_text = (
                    "NA" if improvement is None else f"{improvement * 100:.2f}%"
                )
                interval_text = (
                    "NA"
                    if interval[0] is None
                    else f"[{interval[0] * 100:.2f}%, {interval[1] * 100:.2f}%]"
                )
                lines.append(
                    f"| {field} | {group} | {name} | "
                    f"{pair['common_success_count']} | {improvement_text} | "
                    f"{interval_text} |"
                )
    lines.extend(
        [
            "",
            "## 逐地图",
            "",
            "| 地图 | 冲突×PP负载 | Adaptive | V2 | Mixed |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["instances"]:
        def display(value: float | None) -> str:
            return "FAIL" if value is None else f"{value:.3f}"

        lines.append(
            f"| {row['map_id']} | {row['conflict_load_cell']} | "
            f"{display(row['official_adaptive_ttf'])} | "
            f"{display(row['v2_full_ttf'])} | "
            f"{display(row['mixed_full_v2_ttf'])} |"
        )
    lines.extend(
        [
            "",
            "正改善表示候选控制器更快。分层样本很小时，bootstrap 区间仅作不确定性提示。",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_success_only_ttf(
    collection: str | Path, schedule_root: str | Path, output: str | Path
) -> dict[str, Any]:
    collection_root = Path(collection).resolve()
    schedule_root_path = Path(schedule_root).resolve()
    schedule_path = schedule_root_path / "execution_schedule.json"
    schedule = _read_json(schedule_path)
    _corrected_schedule_report(schedule_root_path, schedule_path, schedule)
    (
        schedule_index,
        by_controller,
        indexed,
        collection_identity,
    ) = _load_scheduled_controller_rows(collection_root, schedule)
    incomplete = [
        row
        for controller in CONTROLLERS
        for row in by_controller[controller]
        if str(row.get("status")) not in {"ok", "resumed"}
        or not isinstance(row.get("summary"), dict)
    ]
    if incomplete:
        raise ValueError("success-only TTF analysis requires complete formal episodes")

    keys = sorted(schedule_index)
    success_only = {
        controller: _success_ttf_summary(
            value
            for key in keys
            if (value := _successful_ttf(indexed[controller][key])) is not None
        )
        for controller in CONTROLLERS
    }
    pairs = {
        "v2_vs_adaptive": ("official_adaptive", "v2-full"),
        "mixed_vs_adaptive": ("official_adaptive", "mixed-full-v2"),
        "mixed_vs_v2": ("v2-full", "mixed-full-v2"),
    }

    def comparisons(group_keys: list[tuple[str, int]]) -> dict[str, Any]:
        return {
            name: _paired_success_ttf(
                indexed[baseline],
                indexed[candidate],
                schedule_index,
                group_keys,
            )
            for name, (baseline, candidate) in pairs.items()
        }

    paired: dict[str, Any] = {"overall": comparisons(keys)}
    for field in (
        "conflict_stratum",
        "initial_pp_load_stratum",
        "source_group",
    ):
        grouped: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
        for key in keys:
            grouped[str(schedule_index[key][field])].append(key)
        paired[field] = {
            name: comparisons(group_keys)
            for name, group_keys in sorted(grouped.items())
        }
    grouped_cells: dict[str, list[tuple[str, int]]] = collections.defaultdict(list)
    for key in keys:
        row = schedule_index[key]
        grouped_cells[
            f"{row['conflict_stratum']}__{row['initial_pp_load_stratum']}"
        ].append(key)
    paired["conflict_load_cell"] = {
        name: comparisons(group_keys)
        for name, group_keys in sorted(grouped_cells.items())
    }

    instances = []
    for key in keys:
        row = schedule_index[key]
        instances.append(
            {
                "task_id": key[0],
                "solver_seed": key[1],
                "map_id": str(row["map_id"]),
                "source_group": str(row["source_group"]),
                "conflict_stratum": str(row["conflict_stratum"]),
                "initial_pp_load_stratum": str(row["initial_pp_load_stratum"]),
                "conflict_load_cell": (
                    f"{row['conflict_stratum']}__{row['initial_pp_load_stratum']}"
                ),
                "initial_conflicts": int(row["initial_conflicts"]),
                "initial_low_level_generated": (
                    int(row["initial_low_level_generated"])
                    if row.get("initial_low_level_generated") is not None
                    else None
                ),
                "official_adaptive_ttf": _successful_ttf(
                    indexed["official_adaptive"][key]
                ),
                "v2_full_ttf": _successful_ttf(indexed["v2-full"][key]),
                "mixed_full_v2_ttf": _successful_ttf(
                    indexed["mixed-full-v2"][key]
                ),
            }
        )

    report = {
        "schema": "lns2.success_only_ttf_report.v2",
        "definition": {
            "failure_ttf": None,
            "failure_penalty_included": False,
            "primary_comparison": "paired_common_success",
            "ttf_field": "wall_time_to_feasible",
        },
        "input": {
            "schedule_sha256": sha256_file(schedule_path),
            "collection": collection_identity,
            "analysis_producer": _analysis_producer_identity(),
            "episode_count": len(keys),
            "controller_episode_count": len(keys) * len(CONTROLLERS),
        },
        "episode_count": len(keys),
        "success_only": success_only,
        "paired": paired,
        "instances": instances,
    }
    output_root = Path(output).resolve()
    _write_json(output_root / "success_only_ttf_report.json", report)
    _write_difficulty_csv(output_root / "success_only_ttf_instances.csv", instances)
    (output_root / "success_only_ttf_report_zh.md").write_text(
        _success_ttf_markdown(report), encoding="utf-8", newline="\n"
    )
    return report


def _load_scheduled_controller_rows(
    collection_root: Path, schedule: dict[str, Any]
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[tuple[str, int], dict[str, Any]]],
    dict[str, Any],
]:
    entries = schedule.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("execution schedule entries must be a non-empty list")
    corrected_schedule = schedule.get("schema") in CORRECTED_SCHEDULE_SCHEMAS
    corrected_full_pool_schedule = (
        schedule.get("schema") == CORRECTED_FULL_POOL_SCHEDULE_SCHEMA
    )
    provenance = schedule.get("provenance") if corrected_schedule else None
    corrected_dataset_fingerprint = (
        provenance.get("formal_dataset_fingerprint")
        if corrected_full_pool_schedule and isinstance(provenance, dict)
        else (
            provenance.get("qualification_dataset_fingerprint")
            if isinstance(provenance, dict)
            else None
        )
    )
    if corrected_schedule and (
        not isinstance(provenance, dict)
        or provenance.get("native_semantics_schema")
        != NATIVE_SEMANTICS_SCHEMA
        or not _is_sha256(
            provenance.get("qualification_producer_identity_fingerprint")
        )
        or not _is_sha256(corrected_dataset_fingerprint)
        or not isinstance(provenance.get("stopping_contract"), dict)
    ):
        raise ValueError("corrected-native schedule provenance is incomplete")
    if corrected_full_pool_schedule:
        assert isinstance(provenance, dict)
        if provenance.get("stopping_contract") != {
            "stopping_rule": "wall-clock-fixed-metric",
            "max_decisions": 0,
            "max_repair_iterations": 0,
            "metric_iteration_budget": 100,
            "wall_time_budget_seconds": 600.0,
            "environment_time_limit_seconds": 600.0,
            "episode_process_timeout_seconds": 660.0,
            "safety_max_decisions": WALL_CLOCK_SAFETY_MAX_DECISIONS,
            "safety_limit_is_not_metric_cap": True,
        }:
            raise ValueError(
                "corrected full-pool schedule stopping contract is invalid"
            )
    schedule_index: dict[tuple[str, int], dict[str, Any]] = {}
    group_orders: dict[int, tuple[str, ...]] = {}
    for raw_row in entries:
        if not isinstance(raw_row, dict):
            raise ValueError("execution schedule entry must be an object")
        row = dict(raw_row)
        key = _episode_key(row)
        if key in schedule_index:
            raise ValueError(f"execution schedule contains duplicate episode: {key}")
        group = row.get("schedule_group")
        if type(group) is not int or group not in range(6):
            raise ValueError("execution schedule group must be an integer from 0 to 5")
        raw_order = row.get("controller_order")
        if not isinstance(raw_order, list) or any(
            not isinstance(name, str) for name in raw_order
        ):
            raise ValueError("execution schedule controller_order must be a string list")
        order = tuple(raw_order)
        if len(order) != len(CONTROLLERS) or set(order) != set(CONTROLLERS):
            raise ValueError(
                "execution schedule controller_order must contain each controller once"
            )
        previous_order = group_orders.setdefault(group, order)
        if previous_order != order:
            raise ValueError(
                f"execution schedule group {group} has inconsistent controller order"
            )
        if corrected_schedule:
            conflicts = row.get("initial_conflicts")
            generated = row.get("initial_low_level_generated")
            state_digest = row.get("state_fingerprint")
            if (
                type(conflicts) is not int
                or conflicts < 0
                or type(generated) is not int
                or generated < 0
                or not _is_sha256(state_digest)
                or row.get("conflict_stratum") != conflict_stratum(conflicts)
                or row.get("initial_pp_load_stratum")
                != initial_pp_load_stratum(generated)
            ):
                raise ValueError(
                    f"corrected-native schedule reset metadata is invalid for {key}"
                )
        schedule_index[key] = row
    if set(group_orders) != set(range(6)):
        raise ValueError("execution schedule must cover all six order groups")
    if corrected_schedule:
        group_counts = collections.Counter(
            row["schedule_group"] for row in schedule_index.values()
        )
        if len(schedule_index) != 36 or group_counts != collections.Counter(
            {group: 6 for group in range(6)}
        ):
            raise ValueError(
                "corrected-native schedule must preserve the frozen 36-episode design"
            )

    expected = set(schedule_index)
    strict_run_configs = [
        (
            collection_root
            / f"order_{group}"
            / controller
            / "run_config.json"
        ).is_file()
        for group in range(6)
        for controller in CONTROLLERS
    ]
    if any(strict_run_configs) and not all(strict_run_configs):
        raise ValueError(
            "formal collection has only a partial set of lane run_config files"
        )
    strict_artifacts = all(strict_run_configs)
    if corrected_schedule and not strict_artifacts:
        raise ValueError(
            "corrected-native formal analysis requires every lane run_config"
        )
    collection_progress_sha256: str | None = None
    if strict_artifacts:
        progress_path = collection_root / "collection_progress.json"
        if not progress_path.is_file():
            raise ValueError("formal collection progress record is missing")
        progress = _read_json(progress_path)
        progress_entries = (
            progress.get("entries") if isinstance(progress, dict) else None
        )
        expected_order = [
            (group, controller)
            for group in range(6)
            for controller in group_orders[group]
        ]
        if not isinstance(progress_entries, list) or len(
            progress_entries
        ) != len(expected_order):
            raise ValueError("formal collection progress coverage is incomplete")
        observed_order = []
        for row in progress_entries:
            if not isinstance(row, dict):
                raise ValueError("formal collection progress entry is not an object")
            if (
                type(row.get("group")) is not int
                or not isinstance(row.get("controller"), str)
                or type(row.get("job_count")) is not int
                or row.get("dry_run") is not False
            ):
                raise ValueError("formal collection progress entry has invalid types")
            observed_order.append((row["group"], row["controller"]))
            expected_group_count = sum(
                scheduled["schedule_group"] == row["group"]
                for scheduled in schedule_index.values()
            )
            if row["job_count"] != expected_group_count:
                raise ValueError(
                    "formal collection progress job count differs from schedule"
                )
        if observed_order != expected_order:
            raise ValueError(
                "formal collection controller execution order differs from schedule"
            )
        collection_progress_sha256 = sha256_file(progress_path)
    rows = []
    lane_identities: list[dict[str, Any]] = []
    for group in range(6):
        group_expected = {
            key
            for key, row in schedule_index.items()
            if row["schedule_group"] == group
        }
        if not group_expected:
            raise ValueError(f"execution schedule group is empty: {group}")
        for controller in CONTROLLERS:
            phase = "official_adaptive" if controller == "official_adaptive" else "realized_dynamic"
            lane = collection_root / f"order_{group}" / controller
            manifest_path = lane / f"{phase}_manifest.jsonl"
            if not manifest_path.is_file():
                raise ValueError(f"formal lane manifest is missing: {manifest_path}")
            lane_rows = _read_jsonl(manifest_path)
            lane_index: dict[tuple[str, int], dict[str, Any]] = {}
            for raw_row in lane_rows:
                if not isinstance(raw_row, dict):
                    raise ValueError(
                        f"formal lane manifest row must be an object: {manifest_path}"
                    )
                row = dict(raw_row)
                key = _episode_key(row)
                if key in lane_index:
                    raise ValueError(
                        f"formal lane contains duplicate episode {key}: {manifest_path}"
                    )
                if key not in group_expected:
                    raise ValueError(
                        f"formal lane episode belongs to another schedule group: {key}"
                    )
                scheduled = schedule_index[key]
                schedule_fields = (
                    ("map_id", "agent_count", "layout_mode")
                    if corrected_schedule
                    else ("map_id", "agent_count")
                )
                for field in schedule_fields:
                    if row.get(field) != scheduled.get(field):
                        raise ValueError(
                            f"formal lane {field} differs from schedule for {key}"
                        )
                if row.get("policy") is not None and row.get("policy") != phase:
                    raise ValueError(
                        f"formal lane policy differs from controller for {key}"
                    )
                if strict_artifacts and row.get("policy") != phase:
                    raise ValueError(
                        f"formal lane lacks its expected policy for {key}"
                    )
                expected_episode_id = (
                    f"{key[0]}__seed_{key[1]:04d}__{phase}"
                )
                if row.get("episode_id") not in {None, expected_episode_id}:
                    raise ValueError(
                        f"formal lane episode_id differs from schedule for {key}"
                    )
                lane_index[key] = row
            if set(lane_index) != group_expected:
                missing = sorted(group_expected - set(lane_index))
                extra = sorted(set(lane_index) - group_expected)
                raise ValueError(
                    "formal lane coverage differs from its schedule group: "
                    f"{manifest_path}; missing={missing}; extra={extra}"
                )

            lane_identity: dict[str, Any] = {
                "schedule_group": group,
                "controller_id": controller,
                "policy": phase,
                "manifest_sha256": sha256_file(manifest_path),
                "episode_count": len(lane_rows),
            }
            if strict_artifacts:
                run_config_path = lane / "run_config.json"
                run_config = _read_json(run_config_path)
                if not isinstance(run_config, dict):
                    raise ValueError(
                        f"formal lane run_config is not an object: {run_config_path}"
                    )
                configuration = run_config.get("configuration")
                if not isinstance(configuration, dict):
                    raise ValueError(
                        f"formal lane run_config lacks configuration: {run_config_path}"
                    )
                if (
                    _fingerprint(configuration)
                    != run_config.get("configuration_fingerprint")
                ):
                    raise ValueError(
                        f"formal lane configuration fingerprint mismatch: {run_config_path}"
                    )
                run_fingerprint = run_config.get("run_fingerprint")
                if (
                    not isinstance(run_fingerprint, str)
                    or len(run_fingerprint) != 64
                    or any(character not in "0123456789abcdef" for character in run_fingerprint)
                ):
                    raise ValueError(
                        f"formal lane run fingerprint is invalid: {run_config_path}"
                    )
                configured_keys = configuration.get("cohort_job_keys_override")
                if not isinstance(configured_keys, list):
                    raise ValueError(
                        f"formal lane lacks frozen cohort job keys: {run_config_path}"
                    )
                if any(
                    not isinstance(value, list)
                    or len(value) != 2
                    or not isinstance(value[0], str)
                    or not value[0]
                    or type(value[1]) is not int
                    or value[1] < 0
                    for value in configured_keys
                ):
                    raise ValueError(
                        f"formal lane has invalid cohort job keys: {run_config_path}"
                    )
                configured_key_set = {
                    (value[0], value[1]) for value in configured_keys
                }
                if (
                    len(configured_key_set) != len(configured_keys)
                    or configured_key_set != group_expected
                ):
                    raise ValueError(
                        f"formal lane configured cohort differs from schedule group: "
                        f"{run_config_path}"
                    )
                if configuration.get("formal") is not True:
                    raise ValueError(
                        f"formal lane run_config is not marked formal: {run_config_path}"
                    )
                if run_config.get("verification_profile") != "deployment":
                    raise ValueError(
                        f"formal lane has unexpected verification profile: "
                        f"{run_config_path}"
                    )
                if run_config.get("storage_fingerprint") is None:
                    raise ValueError(
                        f"formal lane lacks storage identity: {run_config_path}"
                    )
                producer: dict[str, Any] | None = None
                lane_producer_fingerprint: str | None = None
                if corrected_schedule:
                    assert isinstance(provenance, dict)
                    if corrected_full_pool_schedule:
                        producer, lane_producer_fingerprint = (
                            _validated_structured_producer_config(
                                run_config,
                                label=f"formal lane {group}/{controller}",
                            )
                        )
                    else:
                        producer = run_config.get("producer_identity")
                        if not isinstance(producer, dict):
                            producer = run_config.get(
                                "controller_implementation"
                            )
                        producer = _validate_corrected_producer_identity(
                            producer,
                            label=f"formal lane {group}/{controller}",
                        )
                        lane_producer_fingerprint = _fingerprint(producer)
                    if (
                        lane_producer_fingerprint
                        != provenance[
                            "qualification_producer_identity_fingerprint"
                        ]
                        or (
                            corrected_full_pool_schedule
                            and producer
                            != provenance.get(
                                "qualification_producer_identity"
                            )
                        )
                        or run_config.get("dataset_fingerprint")
                        != corrected_dataset_fingerprint
                    ):
                        raise ValueError(
                            "formal lane producer or dataset identity differs "
                            f"from corrected schedule provenance: {run_config_path}"
                        )
                    contract = provenance["stopping_contract"]
                    environment = configuration.get("environment")
                    if (
                        configuration.get("stopping_rule")
                        != contract.get("stopping_rule")
                        or type(configuration.get("max_decisions")) is not int
                        or configuration["max_decisions"]
                        != contract.get("max_decisions")
                        or type(configuration.get("metric_iteration_budget"))
                        is not int
                        or configuration["metric_iteration_budget"]
                        != contract.get("metric_iteration_budget")
                        or not isinstance(environment, dict)
                        or type(environment.get("max_repair_iterations"))
                        is not int
                        or environment["max_repair_iterations"]
                        != contract.get("max_repair_iterations")
                        or configuration.get("wall_time_budget_seconds")
                        != contract.get("wall_time_budget_seconds")
                        or environment.get("time_limit")
                        != contract.get("environment_time_limit_seconds")
                        or (
                            "episode_process_timeout_seconds" in contract
                            and configuration.get(
                                "episode_process_timeout_seconds"
                            )
                            != contract.get(
                                "episode_process_timeout_seconds"
                            )
                        )
                    ):
                        raise ValueError(
                            "formal lane stopping contract differs from corrected "
                            f"qualification: {run_config_path}"
                        )
                metric_budget = configuration.get("metric_iteration_budget")
                for key, row in lane_index.items():
                    if str(row.get("status")) not in {"ok", "resumed"}:
                        continue
                    trace_reference = row.get("trace_file")
                    if not isinstance(trace_reference, str) or not trace_reference:
                        raise ValueError(
                            f"complete formal row lacks trace_file for {key}"
                        )
                    trace_path = (lane / trace_reference).resolve()
                    try:
                        trace_path.relative_to(lane.resolve())
                    except ValueError as error:
                        raise ValueError(
                            f"formal trace escapes its lane for {key}"
                        ) from error
                    if not trace_path.is_file():
                        raise ValueError(f"formal trace is missing for {key}")
                    if sha256_file(trace_path) != row.get("trace_sha256"):
                        raise ValueError(f"formal trace SHA256 mismatch for {key}")
                    validated = validate_closed_loop_trace(
                        trace_path,
                        run_fingerprint,
                        expected_episode_id=(
                            f"{key[0]}__seed_{key[1]:04d}__{phase}"
                        ),
                        expected_policy=phase,
                        expected_solver_seed=key[1],
                        metric_iteration_budget=(
                            int(metric_budget)
                            if metric_budget is not None
                            else None
                        ),
                        collection_root=lane,
                    )
                    if _fingerprint(validated["summary"]) != _fingerprint(
                        row.get("summary")
                    ):
                        raise ValueError(
                            f"formal trace summary differs from manifest for {key}"
                        )
                    if corrected_schedule:
                        summary = validated["summary"]
                        scheduled = schedule_index[key]
                        if (
                            not isinstance(summary, dict)
                            or summary.get("initial_fingerprint")
                            != scheduled.get("state_fingerprint")
                            or type(summary.get("initial_conflicts")) is not int
                            or summary["initial_conflicts"]
                            != scheduled.get("initial_conflicts")
                        ):
                            raise ValueError(
                                "formal trace initial state differs from corrected "
                                f"schedule for {key}"
                            )
                    if int(validated["event_count"]) != row.get(
                        "trace_event_count"
                    ):
                        raise ValueError(
                            f"formal trace event count differs from manifest for {key}"
                        )
                    if validated.get("initial_state_ref") != row.get(
                        "initial_state_ref"
                    ):
                        raise ValueError(
                            f"formal trace initial state differs from manifest for {key}"
                        )
                    metadata = trace_file_metadata(trace_path)
                    for field in (
                        "trace_sha256",
                        "trace_bytes",
                    ):
                        if metadata.get(field) != row.get(field):
                            raise ValueError(
                                f"formal trace {field} differs from manifest for {key}"
                            )
                    if row.get("trace_format") != run_config.get("trace_format"):
                        raise ValueError(
                            f"formal trace format differs from run_config for {key}"
                        )
                    if row.get("storage_fingerprint") != run_config.get(
                        "storage_fingerprint"
                    ):
                        raise ValueError(
                            f"formal row storage identity differs from run_config for {key}"
                        )

                lane_identity.update(
                    {
                        "run_config_sha256": sha256_file(run_config_path),
                        "run_fingerprint": run_fingerprint,
                        "configuration_fingerprint": run_config[
                            "configuration_fingerprint"
                        ],
                            "storage_fingerprint": run_config["storage_fingerprint"],
                            "producer_identity_fingerprint": (
                                lane_producer_fingerprint
                            ),
                        "controller_bundle_fingerprint": (
                            _fingerprint(run_config["controller_bundle"])
                            if isinstance(run_config.get("controller_bundle"), dict)
                            else None
                        ),
                    }
                )
            lane_identities.append(lane_identity)
            rows.extend(
                {
                    **row,
                    "controller_id": controller,
                    "_lane_root": str(lane),
                    "_schedule_group": group,
                }
                for row in lane_rows
            )
    by_controller: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_controller[str(row["controller_id"])].append(row)
    if any(
        len(by_controller[name]) != len(expected)
        or {_episode_key(row) for row in by_controller[name]} != expected
        for name in CONTROLLERS
    ):
        raise ValueError("formal controller coverage differs from the frozen cohort")
    indexed = {
        controller: {_episode_key(row): row for row in by_controller[controller]}
        for controller in CONTROLLERS
    }
    collection_identity = {
        "validation_mode": (
            "run_config_trace_semantic" if strict_artifacts else "synthetic_manifest_only"
        ),
        "collection_progress_sha256": collection_progress_sha256,
        "lanes": lane_identities,
    }
    collection_identity["fingerprint"] = _fingerprint(collection_identity)
    return schedule_index, by_controller, indexed, collection_identity


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = (cursor + end - 1) / 2.0
        for index in ordered[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_ranks, right_ranks = _average_ranks(left), _average_ranks(right)
    left_mean, right_mean = _mean(left_ranks), _mean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0


def _controller_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = _aggregate_rows(rows)
    summaries = [dict(row["summary"]) for row in rows]
    successes = [
        row
        for row in summaries
        if _strict_summary_bool(row, "success")
    ]
    return {
        **aggregate,
        "mean_actual_observed_wall_seconds": _mean(
            float(row["episode_observed_wall_seconds"]) for row in summaries
        ),
        "mean_initial_pp_seconds": _mean(
            float(dict(row.get("reset_timings", {})).get("initial_solution_seconds", 0.0))
            for row in summaries
        ),
        "mean_success_wall_ttf": _mean(
            float(row["wall_time_to_feasible"]) for row in successes
        ),
        "failure_cap_penalty_seconds": sum(
            max(
                0.0,
                float(row["capped_wall_time_to_feasible"])
                - float(row["episode_observed_wall_seconds"]),
            )
            for row in summaries
            if not _strict_summary_bool(row, "success")
        ),
    }


def _paired_controller_comparison(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    keys: list[tuple[str, int]],
) -> dict[str, Any]:
    left = [_scientific_summary(baseline[key]) for key in keys]
    right = [_scientific_summary(candidate[key]) for key in keys]

    def improvement(field: str) -> float:
        return _relative_improvement(
            _mean(row[field] for row in left), _mean(row[field] for row in right)
        )

    return {
        "episode_count": len(keys),
        "baseline_success_count": sum(row["success"] for row in left),
        "candidate_success_count": sum(row["success"] for row in right),
        "capped_wall_ttf_improvement": improvement("capped_wall_ttf"),
        "fixed_auc_improvement": improvement("fixed_auc"),
        "repair_wall_improvement": improvement("repair_wall_seconds"),
        "actual_observed_wall_improvement": _relative_improvement(
            _mean(
                float(baseline[key]["summary"]["episode_observed_wall_seconds"])
                for key in keys
            ),
            _mean(
                float(candidate[key]["summary"]["episode_observed_wall_seconds"])
                for key in keys
            ),
        ),
    }


def _grouped_difficulty_results(
    complexity_rows: list[dict[str, Any]],
    indexed: dict[str, dict[tuple[str, int], dict[str, Any]]],
    field: str,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in complexity_rows:
        grouped[str(row[field])].append(row)
    output = {}
    for name, rows in sorted(grouped.items()):
        keys = [(str(row["task_id"]), int(row["solver_seed"])) for row in rows]
        output[name] = {
            "episode_count": len(rows),
            "source_counts": dict(
                sorted(collections.Counter(str(row["source_group"]) for row in rows).items())
            ),
            "mean_initial_conflicts": _mean(float(row["initial_conflicts"]) for row in rows),
            "mean_initial_pp_generated": _mean(
                float(row["initial_low_level_generated"]) for row in rows
            ),
            "mean_total_path_cost": _mean(float(row["total_path_cost"]) for row in rows),
            "controllers": {
                controller: _controller_group_summary([indexed[controller][key] for key in keys])
                for controller in CONTROLLERS
            },
            "comparisons": {
                "v2_vs_adaptive": _paired_controller_comparison(
                    indexed["official_adaptive"], indexed["v2-full"], keys
                ),
                "mixed_vs_adaptive": _paired_controller_comparison(
                    indexed["official_adaptive"], indexed["mixed-full-v2"], keys
                ),
                "mixed_vs_v2": _paired_controller_comparison(
                    indexed["v2-full"], indexed["mixed-full-v2"], keys
                ),
            },
        }
    return output


def _write_difficulty_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    fieldnames = list(rows[0]) if rows else []
    with partial.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def _difficulty_markdown(report: dict[str, Any]) -> str:
    stratification = report["stratification"]
    overall = report["overall_results"]
    interpretation = (
        "冻结 cohort 已通过预注册的冲突层与初始 PP 负载平衡检查，"
        "可以在 3×3 难度网格内解释配对控制器结果。由于两个高负载单元缺少"
        "自然 MovingAI 样本，地图来源效应仍只作描述性分析。"
        if report["decision"] == "compute_load_balanced_confirmation"
        else "冻结实例内的配对比较仍然有效，但该 cohort 未通过计算负载平衡门槛。"
    )
    lines = [
        "# V2 / Mixed Full 分层墙钟确认",
        "",
        f"结论：`{report['decision']}`。{interpretation}",
        "",
        "## 实验完整性",
        "",
        "| 项目 | low | medium | high |",
        "| --- | ---: | ---: | ---: |",
        "| 唯一冲突 agent 对 | "
        + " | ".join(
            str(stratification["conflict_counts"].get(name, 0))
            for name in ("low", "medium", "high")
        )
        + " |",
        "| 初始 PP generated nodes | "
        + " | ".join(
            str(stratification["initial_pp_load_counts"].get(name, 0))
            for name in ("low", "medium", "high")
        )
        + " |",
        "",
        "冲突层为 1-10、11-100、101-500；初始 PP 负载层为不超过 100,000、"
        "100,001-1,000,000、超过 1,000,000 generated nodes。",
        "",
        "## 总体结果",
        "",
        "| 控制器 | 成功 | capped TTF (s) | 实际执行 (s) | 冲突 AUC | 修复轮数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for controller in CONTROLLERS:
        row = overall["controllers"][controller]
        lines.append(
            f"| {controller} | {row['success_count']}/{row['episode_count']} | "
            f"{row['mean_capped_wall_ttf']:.3f} | "
            f"{row['mean_actual_observed_wall_seconds']:.3f} | "
            f"{row['mean_fixed_auc']:.3f} | {row['mean_repair_iterations']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 3×3 难度单元",
            "",
            "| 冲突×PP负载 | N | Adaptive 成功 | V2 成功 | Mixed 成功 | "
            "V2→Mixed TTF | V2→Mixed AUC |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, cell in sorted(report["grouped_results"]["conflict_load_cell"].items()):
        controllers = cell["controllers"]
        comparison = cell["comparisons"]["mixed_vs_v2"]
        lines.append(
            f"| {name} | {cell['episode_count']} | "
            f"{controllers['official_adaptive']['success_count']} | "
            f"{controllers['v2-full']['success_count']} | "
            f"{controllers['mixed-full-v2']['success_count']} | "
            f"{comparison['capped_wall_ttf_improvement'] * 100:.1f}% | "
            f"{comparison['fixed_auc_improvement'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## 来源与 PP 负载",
            "",
            "| 来源 | low | medium | high |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for source in ("generated", "movingai"):
        counts = stratification["source_by_initial_pp_load"].get(source, {})
        lines.append(
            f"| {source} | {counts.get('low', 0)} | "
            f"{counts.get('medium', 0)} | {counts.get('high', 0)} |"
        )
    lines.extend(
        [
            "",
            "## 难度相关性",
            "",
            "下表为 Spearman 相关，仅用于解释初始难度，不用于选择 cohort 或模型。",
            "",
            "| 初始指标 | 初始 PP 时间 | 实际 episode 时间 | capped TTF |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, values in report["difficulty_correlations"].items():
        lines.append(
            f"| {name} | {values['initial_pp_seconds']:.3f} | "
            f"{values['actual_episode_seconds']:.3f} | "
            f"{values['capped_wall_ttf']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `num_of_colliding_pairs` 是唯一无序冲突 agent 对数，不是重复时空冲突事件数。",
            "- capped TTF 对未在 100 轮内可行的 episode 按 600 秒记账；实际执行时间单独报告。",
            "- MovingAI 部分使用官方地图和项目派生的静态 MAPF OD，不等同于官方 MAPF scenario。",
            "- Mixed Full 是否晋级仍由预注册的成功数、TTF、AUC、地图覆盖和 bootstrap 门槛决定。",
            "",
        ]
    )
    return "\n".join(lines)


def audit_balanced_cohort_difficulty(
    collection: str | Path,
    schedule_root: str | Path,
    output: str | Path,
    config: str | Path = DEFAULT_DIFFICULTY_CONFIG,
) -> dict[str, Any]:
    config_path, audit_config = _load_difficulty_config(config)
    collection_root = Path(collection).resolve()
    schedule_root_path = Path(schedule_root).resolve()
    schedule_path = schedule_root_path / "execution_schedule.json"
    schedule = _read_json(schedule_path)
    _corrected_schedule_report(schedule_root_path, schedule_path, schedule)
    (
        schedule_index,
        _by_controller,
        indexed,
        _collection_identity,
    ) = _load_scheduled_controller_rows(collection_root, schedule)
    complexity_rows = []
    integrity_errors = []
    for key in sorted(schedule_index):
        source = indexed["official_adaptive"][key]
        lane_root = Path(str(source["_lane_root"])).resolve()
        reference = str(source.get("initial_state_ref") or "")
        if not reference:
            raise ValueError(f"official episode is missing initial_state_ref: {key}")
        blob = (lane_root / reference).resolve()
        try:
            blob.relative_to(lane_root)
        except ValueError as error:
            raise ValueError(f"initial state reference escapes controller lane: {key}") from error
        state = read_state_blob(blob)
        complexity = summarize_initial_state_complexity(state)
        expected_fingerprint = str(source["summary"]["initial_fingerprint"])
        errors = []
        if state_fingerprint(state) != expected_fingerprint:
            errors.append("state_fingerprint")
        if int(complexity["conflict_pair_count"]) != int(
            schedule_index[key]["initial_conflicts"]
        ):
            errors.append("initial_conflicts")
        if conflict_stratum(int(complexity["conflict_pair_count"])) != str(
            schedule_index[key]["conflict_stratum"]
        ):
            errors.append("conflict_stratum")
        if int(complexity["agent_count"]) != int(schedule_index[key]["agent_count"]):
            errors.append("agent_count")
        if errors:
            integrity_errors.append(
                {"task_id": key[0], "solver_seed": key[1], "fields": errors}
            )
        complexity_rows.append(
            {
                "task_id": key[0],
                "solver_seed": key[1],
                "map_conflict_cell": (
                    f"{schedule_index[key]['map_id']}::{schedule_index[key]['conflict_stratum']}"
                ),
                **{
                    name: schedule_index[key][name]
                    for name in (
                        "map_id",
                        "source_group",
                        "layout_mode",
                        "agent_count",
                        "agent_band",
                        "initial_conflicts",
                        "conflict_stratum",
                    )
                },
                "initial_pp_load_stratum": initial_pp_load_stratum(
                    int(complexity["initial_low_level_generated"])
                ),
                **complexity,
            }
        )
        complexity_rows[-1]["conflict_load_cell"] = (
            f"{complexity_rows[-1]['conflict_stratum']}__"
            f"{complexity_rows[-1]['initial_pp_load_stratum']}"
        )
    if integrity_errors:
        raise ValueError(f"initial-state complexity audit failed: {integrity_errors[:3]}")

    conflict_counts = collections.Counter(
        str(row["conflict_stratum"]) for row in complexity_rows
    )
    load_counts = collections.Counter(
        str(row["initial_pp_load_stratum"]) for row in complexity_rows
    )
    source_by_load: dict[str, dict[str, int]] = {}
    for source in sorted({str(row["source_group"]) for row in complexity_rows}):
        source_by_load[source] = dict(
            sorted(
                collections.Counter(
                    str(row["initial_pp_load_stratum"])
                    for row in complexity_rows
                    if str(row["source_group"]) == source
                ).items()
            )
        )
    conflict_by_load: dict[str, dict[str, int]] = {}
    for conflict in ("low", "medium", "high"):
        conflict_by_load[conflict] = dict(
            sorted(
                collections.Counter(
                    str(row["initial_pp_load_stratum"])
                    for row in complexity_rows
                    if str(row["conflict_stratum"]) == conflict
                ).items()
            )
        )

    official = indexed["official_adaptive"]
    targets = {
        "initial_pp_seconds": [
            float(dict(official[(row["task_id"], row["solver_seed"])]["summary"].get("reset_timings", {})).get("initial_solution_seconds", 0.0))
            for row in complexity_rows
        ],
        "actual_episode_seconds": [
            float(official[(row["task_id"], row["solver_seed"])]["summary"]["episode_observed_wall_seconds"])
            for row in complexity_rows
        ],
        "capped_wall_ttf": [
            float(official[(row["task_id"], row["solver_seed"])]["summary"]["capped_wall_time_to_feasible"])
            for row in complexity_rows
        ],
    }
    predictor_names = (
        "conflict_pair_count",
        "conflict_event_count",
        "active_conflict_agent_ratio",
        "largest_conflict_component_ratio",
        "agent_count",
        "total_path_cost",
        "initial_low_level_generated",
        "initial_low_level_expanded",
    )
    correlations = {
        name: {
            target: _spearman(
                [float(row[name]) for row in complexity_rows], values
            )
            for target, values in targets.items()
        }
        for name in predictor_names
    }
    source_overlap = all(
        all(source_by_load.get(source, {}).get(load, 0) > 0 for source in ("generated", "movingai"))
        for load in ("low", "medium", "high")
    )
    load_sizes = [load_counts.get(name, 0) for name in ("low", "medium", "high")]
    load_balanced = max(load_sizes) - min(load_sizes) <= 1
    gates = {
        "conflict_tiers_have_12_each": all(conflict_counts.get(name, 0) == 12 for name in ("low", "medium", "high")),
        "initial_pp_load_tiers_balanced": load_balanced,
        "generated_and_movingai_overlap_in_every_load_tier": source_overlap,
        "initial_state_integrity": not integrity_errors,
    }
    gates_passed = all(gates.values())
    all_keys = [(str(row["task_id"]), int(row["solver_seed"])) for row in complexity_rows]
    overall_results = {
        "controllers": {
            controller: _controller_group_summary(
                [indexed[controller][key] for key in all_keys]
            )
            for controller in CONTROLLERS
        },
        "comparisons": {
            "v2_vs_adaptive": _paired_controller_comparison(
                indexed["official_adaptive"], indexed["v2-full"], all_keys
            ),
            "mixed_vs_adaptive": _paired_controller_comparison(
                indexed["official_adaptive"], indexed["mixed-full-v2"], all_keys
            ),
            "mixed_vs_v2": _paired_controller_comparison(
                indexed["v2-full"], indexed["mixed-full-v2"], all_keys
            ),
        },
    }

    csv_rows = []
    for row in complexity_rows:
        key = (str(row["task_id"]), int(row["solver_seed"]))
        flat = dict(row)
        for controller in CONTROLLERS:
            summary = dict(indexed[controller][key]["summary"])
            prefix = controller.replace("-", "_")
            flat.update(
                {
                    f"{prefix}_success": bool(summary["success"]),
                    f"{prefix}_capped_wall_ttf": float(summary["capped_wall_time_to_feasible"]),
                    f"{prefix}_actual_episode_seconds": float(summary["episode_observed_wall_seconds"]),
                    f"{prefix}_initial_pp_seconds": float(dict(summary.get("reset_timings", {})).get("initial_solution_seconds", 0.0)),
                    f"{prefix}_repair_wall_seconds": float(summary.get("repair_wall_seconds", 0.0)),
                    f"{prefix}_fixed_auc": float(summary["fixed_budget_conflict_auc"]),
                }
            )
        csv_rows.append(flat)

    report = {
        "schema": "lns2.balanced_wall_clock_difficulty_audit.v1",
        "evidence_level": (
            "preregistered_compute_load_balanced_end_to_end"
            if gates_passed
            and audit_config.get("evidence_level")
            == "qualification_only_precontroller_design"
            else "post_hoc_methodology_audit_of_frozen_end_to_end_results"
        ),
        "input": {
            "config_sha256": sha256_file(config_path),
            "schedule_sha256": sha256_file(schedule_path),
            "implementation_sha256": {
                relative: sha256_file(Path(__file__).resolve().parents[1] / relative)
                for relative in (
                    "experiments/balanced_wall_clock.py",
                    "experiments/closed_loop_trace_storage.py",
                    "experiments/state_analysis.py",
                )
            },
            "episode_count": len(complexity_rows),
            "controller_episode_count": len(complexity_rows) * len(CONTROLLERS),
        },
        "stratification": {
            "conflict_definition": "unique unordered colliding-agent pairs",
            "conflict_thresholds": [list(value) for value in STRATA],
            "initial_pp_load_definition": "initial PP low-level generated nodes",
            "initial_pp_load_thresholds": [list(value) for value in INITIAL_PP_LOAD_STRATA],
            "conflict_counts": dict(sorted(conflict_counts.items())),
            "initial_pp_load_counts": dict(sorted(load_counts.items())),
            "conflict_by_initial_pp_load": conflict_by_load,
            "source_by_initial_pp_load": source_by_load,
        },
        "difficulty_correlations": correlations,
        "overall_results": overall_results,
        "grouped_results": {
            "conflict_stratum": _grouped_difficulty_results(
                complexity_rows, indexed, "conflict_stratum"
            ),
            "initial_pp_load_stratum": _grouped_difficulty_results(
                complexity_rows, indexed, "initial_pp_load_stratum"
            ),
            "conflict_load_cell": _grouped_difficulty_results(
                complexity_rows, indexed, "conflict_load_cell"
            ),
            "source_group": _grouped_difficulty_results(
                complexity_rows, indexed, "source_group"
            ),
            "map_conflict_cell": _grouped_difficulty_results(
                complexity_rows, indexed, "map_conflict_cell"
            ),
        },
        "methodology_gates": gates,
        "decision": (
            "compute_load_balanced_confirmation"
            if gates_passed
            else "conflict_count_balanced_pilot_with_compute_load_confounding"
        ),
        "interpretation": (
            "The frozen cohort passed the preregistered conflict and initial-PP-load balance "
            "checks, so paired controller results can be interpreted across the 3x3 difficulty "
            "grid. Source effects remain descriptive because MovingAI coverage is asymmetric in "
            "the two naturally missing high-load cells."
            if gates_passed
            else "Paired comparisons remain valid within each frozen instance, but the cohort "
            "does not satisfy the computational-load balance gate."
        ),
    }
    output_root = Path(output).resolve()
    _write_json(output_root / "difficulty_audit.json", report)
    _write_difficulty_csv(output_root / "difficulty_episodes.csv", csv_rows)
    markdown = _difficulty_markdown(report)
    markdown_path = output_root / "difficulty_audit_zh.md"
    markdown_path.write_text(markdown, encoding="utf-8", newline="\n")
    return report


def analyze_scheduled(
    collection: str | Path, schedule_root: str | Path, output: str | Path
) -> dict[str, Any]:
    collection_root = Path(collection).resolve()
    schedule_root_path = Path(schedule_root).resolve()
    schedule_path = schedule_root_path / "execution_schedule.json"
    schedule = _read_json(schedule_path)
    _corrected_schedule_report(schedule_root_path, schedule_path, schedule)
    (
        schedule_index,
        by_controller,
        indexed,
        collection_identity,
    ) = _load_scheduled_controller_rows(collection_root, schedule)
    incomplete = [
        {
            "controller": controller,
            "task_id": str(row.get("task_id")),
            "solver_seed": int(row.get("solver_seed", -1)),
            "status": row.get("status"),
            "error": row.get("error"),
        }
        for controller in CONTROLLERS
        for row in by_controller[controller]
        if str(row.get("status")) not in {"ok", "resumed"}
        or not isinstance(row.get("summary"), dict)
    ]
    if incomplete:
        raise ValueError(
            f"formal controller collection contains {len(incomplete)} incomplete episode rows"
        )
    expected = set(schedule_index)
    integrity_errors = []
    for key in sorted(expected):
        summaries_at_key = [indexed[name][key]["summary"] for name in CONTROLLERS]
        fingerprints = {str(row["initial_fingerprint"]) for row in summaries_at_key}
        initial_conflicts = {int(row["initial_conflicts"]) for row in summaries_at_key}
        if len(fingerprints) != 1 or len(initial_conflicts) != 1:
            integrity_errors.append({"task_id": key[0], "solver_seed": key[1]})
    summaries = {controller: _aggregate_rows(by_controller[controller]) for controller in CONTROLLERS}
    original = summaries["v2-full"]
    mixed = summaries["mixed-full-v2"]
    adaptive = summaries["official_adaptive"]
    mixed_ttf_improvement = _relative_improvement(
        original["mean_capped_wall_ttf"], mixed["mean_capped_wall_ttf"]
    )
    mixed_auc_change = -_relative_improvement(
        original["mean_fixed_auc"], mixed["mean_fixed_auc"]
    )
    mixed_bootstrap = _paired_bootstrap_by_map(
        indexed["v2-full"], indexed["mixed-full-v2"], metric="capped_wall_ttf"
    )
    comparisons_by_stratum = _paired_group_comparison(
        indexed["v2-full"], indexed["mixed-full-v2"], schedule_index, "conflict_stratum"
    )
    has_compute_load = all(
        "initial_pp_load_stratum" in row for row in schedule_index.values()
    )
    if has_compute_load:
        comparisons_by_initial_pp_load = _paired_group_comparison(
            indexed["v2-full"],
            indexed["mixed-full-v2"],
            schedule_index,
            "initial_pp_load_stratum",
        )
        cell_schedule_index = {
            key: {
                **row,
                "conflict_load_cell": (
                    f"{row['conflict_stratum']}__{row['initial_pp_load_stratum']}"
                ),
            }
            for key, row in schedule_index.items()
        }
        comparisons_by_difficulty_cell = _paired_group_comparison(
            indexed["v2-full"],
            indexed["mixed-full-v2"],
            cell_schedule_index,
            "conflict_load_cell",
        )
    else:
        comparisons_by_initial_pp_load = {}
        comparisons_by_difficulty_cell = {}
    comparisons_by_map = _paired_group_comparison(
        indexed["v2-full"], indexed["mixed-full-v2"], schedule_index, "map_id"
    )
    comparisons_by_source = _paired_group_comparison(
        indexed["v2-full"], indexed["mixed-full-v2"], schedule_index, "source_group"
    )
    comparisons_by_agent_band = _paired_group_comparison(
        indexed["v2-full"], indexed["mixed-full-v2"], schedule_index, "agent_band"
    )
    maps_not_worse = sum(row["candidate_not_worse"] for row in comparisons_by_map.values())
    map_not_worse_requirement = math.ceil(2 * len(comparisons_by_map) / 3)
    strata_not_worse = sum(
        row["candidate_not_worse"] for row in comparisons_by_stratum.values()
    )
    gate = {
        "success_not_lower_than_v2": mixed["success_count"] >= original["success_count"],
        "ttf_improvement_at_least_5_percent": mixed_ttf_improvement >= 0.05,
        "auc_not_worse_than_2_percent": mixed_auc_change <= 0.02,
        "at_least_two_thirds_maps_not_worse": (
            maps_not_worse >= map_not_worse_requirement
        ),
        "at_least_2_of_3_strata_not_worse": strata_not_worse >= 2,
        "map_bootstrap_no_significant_degradation": mixed_bootstrap[
            "improvement_95_ci"
        ][1]
        >= 0.0,
        "identical_initial_states": not integrity_errors,
        "zero_experiment_errors": all(
            summaries[name][field] == 0
            for name in CONTROLLERS
            for field in ("invalid_action_count", "fingerprint_mismatch_count", "error_count")
        ),
    }
    report = {
        "schema": "lns2.balanced_wall_clock_report.v2",
        "evidence_level": "end_to_end_balanced_wall_clock",
        "input": {
            "schedule_sha256": sha256_file(
                schedule_path
            ),
            "collection": collection_identity,
            "analysis_producer": _analysis_producer_identity(),
        },
        "summaries": summaries,
        "comparisons": {
            "mixed_vs_v2_ttf_improvement": mixed_ttf_improvement,
            "mixed_vs_v2_auc_change": mixed_auc_change,
            "v2_vs_adaptive_ttf_improvement": _relative_improvement(
                adaptive["mean_capped_wall_ttf"], original["mean_capped_wall_ttf"]
            ),
            "mixed_vs_v2_map_bootstrap": mixed_bootstrap,
            "mixed_vs_v2_by_stratum": comparisons_by_stratum,
            "mixed_vs_v2_by_initial_pp_load": comparisons_by_initial_pp_load,
            "mixed_vs_v2_by_difficulty_cell": comparisons_by_difficulty_cell,
            "mixed_vs_v2_by_map": comparisons_by_map,
            "mixed_vs_v2_by_source": comparisons_by_source,
            "mixed_vs_v2_by_agent_band": comparisons_by_agent_band,
        },
        "initial_state_integrity": {
            "passed": not integrity_errors,
            "mismatch_count": len(integrity_errors),
            "mismatches": integrity_errors,
        },
        "promotion_gate": gate,
        "promotion_gate_counts": {
            "maps_not_worse": maps_not_worse,
            "map_not_worse_requirement": map_not_worse_requirement,
            "map_count": len(comparisons_by_map),
            "conflict_strata_not_worse": strata_not_worse,
        },
        "decision": "mixed_full_candidate" if all(gate.values()) else "keep_v2_full",
        "note": (
            "Wall TTF includes reset, online control, and repair. Runtime and generated-node "
            "components are diagnostics; promotion follows the preregistered paired gates."
        ),
    }
    output_root = Path(output).resolve()
    _write_json(output_root / "balanced_wall_clock_report.json", report)
    return report


__all__ = [
    "CONTROLLERS",
    "CORRECTED_FULL_POOL_REPORT_SCHEMA",
    "CORRECTED_FULL_POOL_SCHEDULE_SCHEMA",
    "INITIAL_PP_LOAD_STRATA",
    "STRATA",
    "analyze_scheduled",
    "analyze_success_only_ttf",
    "audit_balanced_cohort_difficulty",
    "build_replacement_dataset",
    "collect_scheduled",
    "conflict_stratum",
    "initial_pp_load_stratum",
    "materialize_compute_load_candidate_pool",
    "materialize_qualified_compute_load_pool",
    "materialize_registered_compute_load_cohort",
    "merge_datasets",
    "prepare_movingai_map_derived_dataset",
    "prepare_movingai_dataset",
    "prepare_corrected_native_formal_config",
    "qualify_corrected_native_schedule",
    "qualify_corrected_native_seed_pool",
    "rebind_corrected_native_schedule",
    "select_corrected_native_seed_schedule",
    "select_balanced_cohort",
    "select_compute_load_balanced_cohort",
    "recover_scheduled_partial_traces",
    "verify_compute_load_cohort_registration",
]
