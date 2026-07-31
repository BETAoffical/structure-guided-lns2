from __future__ import annotations

import collections
import csv
import hashlib
import heapq
import itertools
import json
import math
import random
import shutil
import statistics
import zipfile
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)
from experiments.state_analysis import summarize_initial_state_complexity


SPLIT = "balanced_wall_clock"
CONTROLLERS = ("official_adaptive", "v2-full", "mixed-full-v2")
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


def collect_scheduled(
    *,
    dataset: str | Path,
    config: str | Path,
    qualification: str | Path,
    schedule_root: str | Path,
    output: str | Path,
    original_bundle: str | Path,
    mixed_bundle: str | Path,
    resume: bool,
    dry_run: bool = False,
    registration: str | Path | None = None,
) -> dict[str, Any]:
    registration_report = (
        verify_compute_load_cohort_registration(registration)
        if registration is not None
        else None
    )
    schedule_root_path = Path(schedule_root).resolve()
    cohort_report = _read_json(schedule_root_path / "cohort_report.json")
    if not bool(cohort_report.get("formal_collection_allowed", False)):
        raise ValueError("balanced wall-clock cohort did not pass the preregistered data gate")
    schedule_path = schedule_root_path / "execution_schedule.json"
    schedule = _read_json(schedule_path)
    entries = list(schedule["entries"])
    output_root = Path(output).resolve()
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
                stopping_rule="historical",
                use_global_collection_lock=False,
            )
            if dry_run:
                result = run_closed_loop_collection(
                    **common, phase=phases[controller_id], dry_run=True
                )
            else:
                if not (lane / "qualification_manifest.jsonl").is_file():
                    reusable_qualification = (
                        qualification
                        if (Path(qualification).resolve() / "run_config.json").is_file()
                        else None
                    )
                    run_closed_loop_collection(
                        **common,
                        phase="qualify",
                        qualification_source=reusable_qualification,
                        resume=False,
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
    return str(row["task_id"]), int(row["solver_seed"])


def _scientific_summary(row: dict[str, Any]) -> dict[str, Any]:
    summary = dict(row["summary"])
    totals = dict(summary.get("controller_totals", {}))
    return {
        "success": bool(summary["success"]),
        "capped_wall_ttf": float(summary["capped_wall_time_to_feasible"]),
        "fixed_auc": float(summary["fixed_budget_conflict_auc"]),
        "normalized_fixed_auc": float(summary["normalized_fixed_budget_conflict_auc"]),
        "repair_iterations": int(summary["repair_iterations"]),
        "generated_nodes": int(summary["final_low_level"]["generated"]),
        "expanded_nodes": int(summary["final_low_level"]["expanded"]),
        "repair_wall_seconds": float(summary.get("repair_wall_seconds", 0.0)),
        "environment_construct_seconds": float(
            summary.get("environment_construct_seconds", 0.0)
        ),
        "reset_wall_seconds": float(summary.get("reset_wall_seconds", 0.0)),
        "episode_observed_wall_seconds": float(
            summary.get("episode_observed_wall_seconds", 0.0)
        ),
        "proposal_seconds": float(totals.get("proposal_seconds", 0.0)),
        "feature_seconds": float(totals.get("feature_seconds", 0.0)),
        "inference_seconds": float(totals.get("inference_seconds", 0.0)),
        "fingerprint_seconds": float(totals.get("state_fingerprint_seconds", 0.0)),
        "pp_replan_seconds": float(totals.get("pp_replan_seconds", 0.0)),
        "controller_before_repair_seconds": float(
            totals.get("controller_seconds_before_repair", 0.0)
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
            int(row["summary"].get("invalid_action_count", 0)) for row in rows
        ),
        "fingerprint_mismatch_count": sum(
            int(row["summary"].get("fingerprint_mismatch_count", 0)) for row in rows
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


def _load_scheduled_controller_rows(
    collection_root: Path, schedule: dict[str, Any]
) -> tuple[
    dict[tuple[str, int], dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, dict[tuple[str, int], dict[str, Any]]],
]:
    schedule_index = {_episode_key(row): dict(row) for row in schedule["entries"]}
    expected = set(schedule_index)
    rows = []
    for group in range(6):
        for controller in CONTROLLERS:
            phase = "official_adaptive" if controller == "official_adaptive" else "realized_dynamic"
            lane = collection_root / f"order_{group}" / controller
            path = lane / f"{phase}_manifest.jsonl"
            rows.extend(
                {**row, "controller_id": controller, "_lane_root": str(lane)}
                for row in _read_jsonl(path)
            )
    by_controller: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_controller[str(row["controller_id"])].append(row)
    if any(
        {_episode_key(row) for row in by_controller[name]} != expected
        for name in CONTROLLERS
    ):
        raise ValueError("formal controller coverage differs from the frozen cohort")
    indexed = {
        controller: {_episode_key(row): row for row in by_controller[controller]}
        for controller in CONTROLLERS
    }
    return schedule_index, by_controller, indexed


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
    successes = [row for row in summaries if bool(row["success"])]
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
            if not bool(row["success"])
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
    lines = [
        "# V2/Mixed Full 分层墙钟难度审计",
        "",
        f"结论：`{report['decision']}`。本报告是对冻结正式结果的事后方法审计，不重新训练或运行控制器。",
        "",
        "## 分层完整性",
        "",
        "| 分层 | low | medium | high |",
        "| --- | ---: | ---: | ---: |",
        "| 唯一冲突 agent 对 | "
        + " | ".join(str(report["stratification"]["conflict_counts"].get(name, 0)) for name in ("low", "medium", "high"))
        + " |",
        "| 初始 PP generated nodes | "
        + " | ".join(str(report["stratification"]["initial_pp_load_counts"].get(name, 0)) for name in ("low", "medium", "high"))
        + " |",
        "",
        "初始 PP 负载阈值固定为：low <= 100,000，medium 100,001-1,000,000，high > 1,000,000 generated nodes。",
        "",
        "## 来源与负载",
        "",
        "| 来源 | low | medium | high |",
        "| --- | ---: | ---: | ---: |",
    ]
    for source in ("generated", "movingai"):
        counts = report["stratification"]["source_by_initial_pp_load"].get(source, {})
        lines.append(
            f"| {source} | {counts.get('low', 0)} | {counts.get('medium', 0)} | {counts.get('high', 0)} |"
        )
    lines.extend(
        [
            "",
            "当前 cohort 的来源与计算负载完全混杂，因此只能支持同一实例内的控制器配对比较，不能把来源差异解释为等难度下的地图结构差异。",
            "",
            "## 难度相关性",
            "",
            "Spearman 相关性使用官方 Adaptive 的初始状态和时间，仅作解释性诊断。",
            "",
            "| 初始指标 | initial PP time | actual episode time | capped TTF |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, values in report["difficulty_correlations"].items():
        lines.append(
            f"| {name} | {values['initial_pp_seconds']:.3f} | {values['actual_episode_seconds']:.3f} | {values['capped_wall_ttf']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- `num_of_colliding_pairs` 是唯一冲突 agent 对数，不是重复时空冲突事件数。",
            "- capped TTF 对失败按 600 秒记账；actual episode time 是实际观察到的执行时间。",
            "- 旧结论保留为固定 cohort 上的配对结果，但不再称为计算负载均衡确认。",
            "- 下一次正式实验必须先通过来源与初始 PP 负载重叠门槛，再运行控制器。",
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
    config_path, _audit_config = _load_difficulty_config(config)
    collection_root = Path(collection).resolve()
    schedule_path = Path(schedule_root).resolve() / "execution_schedule.json"
    schedule = _read_json(schedule_path)
    schedule_index, _by_controller, indexed = _load_scheduled_controller_rows(
        collection_root, schedule
    )
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
        "evidence_level": "post_hoc_methodology_audit_of_frozen_end_to_end_results",
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
        "grouped_results": {
            "conflict_stratum": _grouped_difficulty_results(
                complexity_rows, indexed, "conflict_stratum"
            ),
            "initial_pp_load_stratum": _grouped_difficulty_results(
                complexity_rows, indexed, "initial_pp_load_stratum"
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
            if all(gates.values())
            else "conflict_count_balanced_pilot_with_compute_load_confounding"
        ),
        "interpretation": (
            "Paired controller comparisons on each frozen instance remain valid, but the pooled "
            "cohort is not a balanced computational-load benchmark and cannot isolate source or "
            "map-family effects at matched difficulty."
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
    schedule = _read_json(Path(schedule_root).resolve() / "execution_schedule.json")
    schedule_index, by_controller, indexed = _load_scheduled_controller_rows(
        collection_root, schedule
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
        "schema": "lns2.balanced_wall_clock_report.v1",
        "evidence_level": "end_to_end_balanced_wall_clock",
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
    "INITIAL_PP_LOAD_STRATA",
    "STRATA",
    "analyze_scheduled",
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
    "select_balanced_cohort",
    "select_compute_load_balanced_cohort",
    "verify_compute_load_cohort_registration",
]
