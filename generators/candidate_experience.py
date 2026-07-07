from __future__ import annotations

import collections
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from .repair_experience import (
    _cell_zone,
    _map_features,
    _task_features,
    _validate_conflict_events,
    _validate_path,
)


CANDIDATE_FEATURE_COUNT = 35


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path, rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _position(path: list[list[int]], timestep: int) -> tuple[int, int]:
    return tuple(path[min(timestep, len(path) - 1)])


def _pair_temporal_overlap(
    first: list[list[int]], second: list[list[int]]
) -> float:
    horizon = max(len(first), len(second))
    if horizon == 0:
        return 0.0
    overlaps = 0
    for timestep in range(horizon):
        first_now = _position(first, timestep)
        second_now = _position(second, timestep)
        if first_now == second_now:
            overlaps += 1
        elif timestep > 0:
            first_previous = _position(first, timestep - 1)
            second_previous = _position(second, timestep - 1)
            if (
                first_previous == second_now
                and first_now == second_previous
            ):
                overlaps += 1
    return overlaps / horizon


def candidate_raw_features(
    map_document: dict[str, Any],
    task_document: dict[str, Any],
    manifest_row: dict[str, Any],
    conflict_events: list[dict[str, Any]],
    paths: list[list[list[int]]],
    seed_conflict: list[int],
    candidate: dict[str, Any],
) -> dict[str, float]:
    map_features = _map_features(map_document, manifest_row)
    task_features = _task_features(task_document, manifest_row)
    metadata = task_document["metadata"]
    agent_count = int(metadata["agent_count"])
    rows = int(map_document["rows"])
    cols = int(map_document["cols"])
    free_cells = int(map_document["metadata"]["free_cell_count"])
    scale = max(1.0, float(rows + cols))

    graph: list[set[int]] = [set() for _ in range(agent_count)]
    conflict_cells: set[tuple[int, int]] = set()
    for event in conflict_events:
        first, second = (int(value) for value in event["agents"])
        graph[first].add(second)
        graph[second].add(first)
        conflict_cells.update(tuple(cell) for cell in event["cells"])
    degrees = [len(neighbors) for neighbors in graph]
    active_agents = sum(value > 0 for value in degrees)
    mean_path_length = (
        sum(max(0, len(path) - 1) for path in paths)
        / max(1, len(paths))
    )

    agents = [int(value) for value in candidate["agents"]]
    selected = set(agents)
    candidate_edges = sum(
        first in selected and second in selected
        for first, second in (
            tuple(int(value) for value in event["agents"])
            for event in conflict_events
        )
    )
    candidate_conflict_agents = sum(degrees[agent] > 0 for agent in agents)
    possible_edges = max(1, len(agents) * (len(agents) - 1) // 2)

    path_overlap_values = []
    blocker_count = 0
    prior_values = []
    structural_prior = map_document["metadata"][
        "structural_congestion_prior"
    ]
    for agent in agents:
        path = paths[agent]
        path_overlap_values.append(
            sum(tuple(cell) in conflict_cells for cell in path)
            / max(1, len(path))
        )
        start = tuple(task_document["starts"][agent])
        goal = tuple(task_document["goals"][agent])
        for other, other_path in enumerate(paths):
            if other == agent:
                continue
            blocker_count += sum(
                tuple(cell) in {start, goal} for cell in other_path
            )
        prior_values.extend(
            float(structural_prior[row][col]) for row, col in path
        )

    temporal_values = [
        _pair_temporal_overlap(paths[first], paths[second])
        for first, second in combinations(agents, 2)
    ]
    flow_counts = collections.Counter(
        str(metadata["flow_assignments"][agent]) for agent in agents
    )
    dominant_flow = sorted(
        metadata["realized_flow_counts"].items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[0][0]
    zone_sets = {
        name: {tuple(cell) for cell in cells}
        for name, cells in map_document["metadata"]["zones"].items()
    }
    cross_zone = sum(
        _cell_zone(task_document["starts"][agent], zone_sets)
        != _cell_zone(task_document["goals"][agent], zone_sets)
        for agent in agents
    )
    shortest_distances = [
        float(metadata["actual_shortest_distances"][agent])
        for agent in agents
    ]
    stretches = [
        max(0, len(paths[agent]) - 1)
        / max(
            1.0,
            float(metadata["actual_shortest_distances"][agent]),
        )
        for agent in agents
    ]
    event_count = len(conflict_events)
    seed_first, seed_second = (int(value) for value in seed_conflict)

    features = {
        "map.shelf_coverage": float(map_features["shelf_coverage"]),
        "map.free_cell_ratio": float(map_features["free_cell_ratio"]),
        "map.gate_ratio": float(map_features["gate_count"]) / scale,
        "map.dead_end_ratio": (
            float(map_features["horizontal_dead_end_count"])
            + float(map_features["vertical_dead_end_count"])
        )
        / scale,
        "map.articulation_ratio": (
            float(map_features["articulation_count"])
            / max(1.0, float(free_cells))
        ),
        "map.average_degree": float(
            map_features["average_free_degree"]
        ),
        "map.mean_prior": float(
            map_features["mean_structural_prior"]
        ),
        "map.maximum_prior": float(
            map_features["maximum_structural_prior"]
        ),
        "task.agent_count": float(agent_count),
        "task.density_free": float(
            task_features["agent_density_free_cells"]
        ),
        "task.density_service": float(
            task_features["agent_density_service_cells"]
        ),
        "task.dominant_flow_ratio": float(
            task_features["dominant_flow_ratio"]
        ),
        "task.hotspot_skew": float(task_features["hotspot_skew"]),
        "task.cluster_count": float(
            task_features["origin_cluster_count"]
            + task_features["goal_cluster_count"]
        ),
        "task.mean_shortest_distance": (
            float(task_features["mean_shortest_distance"]) / scale
        ),
        "state.conflict_density": event_count / max(1.0, agent_count),
        "state.vertex_ratio": (
            sum(event["type"] == "vertex" for event in conflict_events)
            / max(1, event_count)
        ),
        "state.active_agent_ratio": active_agents / max(1.0, agent_count),
        "state.mean_conflict_degree": (
            sum(degrees) / max(1.0, agent_count)
        ),
        "state.maximum_conflict_degree": (
            max(degrees, default=0) / max(1.0, agent_count - 1)
        ),
        "state.seed_degree": (
            degrees[seed_first] + degrees[seed_second]
        )
        / max(1.0, 2.0 * (agent_count - 1)),
        "state.mean_conflict_time": (
            sum(float(event["timestep"]) for event in conflict_events)
            / max(1, event_count)
            / max(1.0, mean_path_length)
        ),
        "candidate.conflict_agent_coverage": (
            candidate_conflict_agents / max(1.0, active_agents)
        ),
        "candidate.conflict_edge_coverage": (
            candidate_edges / max(1.0, event_count)
        ),
        "candidate.induced_edge_density": (
            candidate_edges / possible_edges
        ),
        "candidate.mean_conflict_degree": (
            sum(degrees[agent] for agent in agents)
            / max(1.0, len(agents) * max(1, agent_count - 1))
        ),
        "candidate.maximum_conflict_degree": (
            max((degrees[agent] for agent in agents), default=0)
            / max(1.0, agent_count - 1)
        ),
        "candidate.path_conflict_overlap": (
            sum(path_overlap_values) / max(1, len(path_overlap_values))
        ),
        "candidate.temporal_overlap": (
            sum(temporal_values) / max(1, len(temporal_values))
        ),
        "candidate.start_goal_blocker_ratio": (
            blocker_count
            / max(
                1.0,
                len(agents)
                * max(1, agent_count - 1)
                * max(1.0, mean_path_length),
            )
        ),
        "candidate.dominant_flow_ratio": (
            flow_counts[dominant_flow] / max(1.0, len(agents))
        ),
        "candidate.cross_zone_ratio": (
            cross_zone / max(1.0, len(agents))
        ),
        "candidate.mean_shortest_distance": (
            sum(shortest_distances)
            / max(1, len(shortest_distances))
            / scale
        ),
        "candidate.mean_path_stretch": (
            sum(stretches) / max(1, len(stretches))
        ),
        "candidate.mean_structural_prior": (
            sum(prior_values) / max(1, len(prior_values))
        ),
    }
    if len(features) != CANDIDATE_FEATURE_COUNT:
        raise AssertionError(
            f"expected {CANDIDATE_FEATURE_COUNT} features, "
            f"got {len(features)}"
        )
    if not all(math.isfinite(value) for value in features.values()):
        raise ValueError("candidate features contain a non-finite value")
    return features


def _validate_candidate(
    candidate: dict[str, Any],
    seed_conflict: list[int],
    agent_count: int,
    neighborhood_size: int,
) -> None:
    agents = [int(value) for value in candidate["agents"]]
    order = [int(value) for value in candidate["replan_order"]]
    if (
        len(agents) != neighborhood_size
        or len(set(agents)) != neighborhood_size
        or any(not 0 <= agent < agent_count for agent in agents)
    ):
        raise ValueError("candidate contains invalid Agent IDs")
    if not set(seed_conflict).issubset(agents):
        raise ValueError("candidate does not contain its seed pair")
    if set(order) != set(agents) or len(order) != len(agents):
        raise ValueError("candidate replan order is not a permutation")
    if not candidate.get("trial_performed", False):
        raise ValueError("candidate trial was not performed")


def build_candidate_experience(
    dataset: str | Path,
    collection: str | Path,
    output: str | Path,
    split: str,
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError(
            "candidate experience is limited to Train and Validation"
        )
    usage = "memory" if split == "train" else "evaluation"
    dataset_root = Path(dataset).resolve()
    collection_root = Path(collection).resolve()
    output_root = Path(output).resolve()
    dataset_rows = _read_jsonl(
        dataset_root / split / "manifest.jsonl"
    )
    collection_rows = _read_jsonl(
        collection_root / "collection_manifest.jsonl"
    )
    task_rows = {str(row["task_id"]): row for row in dataset_rows}
    cases: list[dict[str, Any]] = []
    state_count = 0
    valid_count = 0
    map_cache: dict[str, dict[str, Any]] = {}
    task_cache: dict[str, dict[str, Any]] = {}

    for run in collection_rows:
        if run["split"] != split:
            raise ValueError("collection crosses the requested split")
        if run.get("status") == "error" or run.get("result") is None:
            raise ValueError("collection contains an invalid run")
        task_id = str(run["task_id"])
        if task_id not in task_rows:
            raise ValueError(f"unknown task: {task_id}")
        manifest_row = task_rows[task_id]
        map_id = str(manifest_row["map_id"])
        map_cache.setdefault(
            map_id,
            _read_json(
                dataset_root / split / str(manifest_row["map_file"])
            ),
        )
        task_cache.setdefault(
            task_id,
            _read_json(
                dataset_root / split / str(manifest_row["task_file"])
            ),
        )
        map_document = map_cache[map_id]
        task_document = task_cache[task_id]
        agent_count = int(task_document["metadata"]["agent_count"])
        trace = _read_jsonl(Path(run["trace_file"]))
        if (
            not trace
            or trace[-1].get("event_type") != "summary"
            or any(row.get("schema_version") != 4 for row in trace)
            or trace[-1].get("candidate_mode") != "collect"
        ):
            raise ValueError("candidate collection requires Trace V4")
        run_id = (
            f"{task_id}__seed_{int(run['solver_seed']):04d}"
        )
        for iteration in (
            row for row in trace if row["event_type"] == "iteration"
        ):
            state_count += 1
            paths = iteration["paths_before"]
            events = iteration["conflict_events_before"]
            if len(paths) != agent_count:
                raise ValueError("Trace V4 omitted full current paths")
            _validate_conflict_events(
                events, map_document, agent_count
            )
            for agent, path in enumerate(paths):
                _validate_path(
                    path,
                    task_document["starts"][agent],
                    task_document["goals"][agent],
                    map_document["grid"],
                )
            candidates = iteration["candidate_trials"]
            expected_count = int(run.get("candidate_count", 8))
            if expected_count != 8 or len(candidates) != 8:
                raise ValueError("state does not contain eight candidates")
            normalized_sets: set[tuple[int, ...]] = set()
            state_id = (
                f"{run_id}__iteration_"
                f"{int(iteration['iteration']):04d}"
            )
            for candidate in candidates:
                _validate_candidate(
                    candidate,
                    iteration["seed_conflict"],
                    agent_count,
                    int(run["neighborhood_size"]),
                )
                key = tuple(sorted(int(value) for value in candidate["agents"]))
                if key in normalized_sets:
                    raise ValueError("state contains duplicate candidates")
                normalized_sets.add(key)
                candidate_valid = bool(candidate["candidate_valid"])
                if candidate_valid:
                    after_paths = {
                        int(item["agent"]): item["path"]
                        for item in candidate[
                            "neighborhood_paths_after"
                        ]
                    }
                    if set(after_paths) != set(candidate["agents"]):
                        raise ValueError(
                            "valid candidate omitted repaired paths"
                        )
                    for agent, path in after_paths.items():
                        _validate_path(
                            path,
                            task_document["starts"][agent],
                            task_document["goals"][agent],
                            map_document["grid"],
                        )
                    _validate_conflict_events(
                        candidate["conflict_events_after"],
                        map_document,
                        agent_count,
                    )
                    if len(candidate["conflict_events_after"]) != int(
                        candidate["conflicting_pairs_after"]
                    ):
                        raise ValueError(
                            "candidate conflict events do not match count"
                        )
                    valid_count += 1
                    conflict_reduction = (
                        int(iteration["conflicting_pairs_before"])
                        - int(candidate["conflicting_pairs_after"])
                    )
                    cost_improvement = (
                        int(iteration["sum_of_costs_before"])
                        - int(candidate["sum_of_costs_after"])
                    )
                else:
                    if candidate["neighborhood_paths_after"]:
                        raise ValueError(
                            "invalid candidate retained repaired paths"
                        )
                    conflict_reduction = None
                    cost_improvement = None
                cases.append(
                    {
                        "schema_version": 1,
                        "usage": usage,
                        "split": split,
                        "case_id": (
                            f"{state_id}__candidate_"
                            f"{int(candidate['candidate_index']):02d}"
                        ),
                        "state_id": state_id,
                        "run_id": run_id,
                        "map_id": map_id,
                        "task_id": task_id,
                        "solver_seed": int(run["solver_seed"]),
                        "iteration": int(iteration["iteration"]),
                        "candidate_index": int(
                            candidate["candidate_index"]
                        ),
                        "generator": str(candidate["generator"]),
                        "seed_conflict": iteration["seed_conflict"],
                        "agents": candidate["agents"],
                        "replan_order": candidate["replan_order"],
                        "features": candidate_raw_features(
                            map_document,
                            task_document,
                            manifest_row,
                            events,
                            paths,
                            iteration["seed_conflict"],
                            candidate,
                        ),
                        "outcome": {
                            "candidate_valid": candidate_valid,
                            "conflicting_pairs_before": int(
                                iteration[
                                    "conflicting_pairs_before"
                                ]
                            ),
                            "conflicting_pairs_after": int(
                                candidate[
                                    "conflicting_pairs_after"
                                ]
                            ),
                            "conflict_reduction": conflict_reduction,
                            "sum_of_costs_before": int(
                                iteration["sum_of_costs_before"]
                            ),
                            "sum_of_costs_after": int(
                                candidate["sum_of_costs_after"]
                            ),
                            "cost_improvement": cost_improvement,
                            "replan_runtime_ms": float(
                                candidate["replan_runtime_ms"]
                            ),
                            "total_runtime_ms": float(
                                candidate["total_runtime_ms"]
                            ),
                        },
                    }
                )

    cases.sort(key=lambda row: str(row["case_id"]))
    _write_jsonl(output_root / "candidate_cases.jsonl", cases)
    summary = {
        "schema_version": 1,
        "source_trace_schema_version": 4,
        "split": split,
        "usage": usage,
        "collection_run_count": len(collection_rows),
        "state_run_count": len({row["run_id"] for row in cases}),
        "state_count": state_count,
        "candidate_case_count": len(cases),
        "candidate_count_per_state": 8,
        "valid_candidate_count": valid_count,
        "map_count": len({row["map_id"] for row in cases}),
        "task_count": len({row["task_id"] for row in cases}),
        "feature_count": CANDIDATE_FEATURE_COUNT,
    }
    _write_json(output_root / "candidate_summary.json", summary)
    return summary
