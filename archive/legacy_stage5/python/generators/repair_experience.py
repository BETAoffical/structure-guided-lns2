from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sparse_heatmap(
    events: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    weights: collections.defaultdict[tuple[int, int], float] = (
        collections.defaultdict(float)
    )
    for event in events:
        cells = [tuple(cell) for cell in event["cells"]]
        if event["type"] == "vertex":
            if len(cells) != 1:
                raise ValueError("vertex conflict must contain one cell")
            weights[cells[0]] += 1.0
        elif event["type"] == "edge_swap":
            if len(cells) != 2:
                raise ValueError("edge conflict must contain two cells")
            for cell in cells:
                weights[cell] += 0.5
        else:
            raise ValueError(f"unknown conflict type: {event['type']}")
    sparse = [
        {"cell": [row, col], "weight": round(weight, 6)}
        for (row, col), weight in sorted(weights.items())
    ]
    return sparse, round(sum(weights.values()), 6)


def _merge_heatmap(
    target: collections.defaultdict[tuple[int, int], float],
    sparse: list[dict[str, Any]],
) -> None:
    for item in sparse:
        target[tuple(item["cell"])] += float(item["weight"])


def _serialize_heatmap(
    weights: collections.defaultdict[tuple[int, int], float],
) -> list[dict[str, Any]]:
    return [
        {"cell": [row, col], "weight": round(weight, 6)}
        for (row, col), weight in sorted(weights.items())
    ]


def _label_iteration(event: dict[str, Any]) -> tuple[str, bool]:
    if not event["candidate_valid"]:
        return "invalid", False
    if not event["accepted"]:
        return "rejected", False
    if (
        event["conflicting_pairs_after"]
        < event["conflicting_pairs_before"]
    ):
        return "conflict_reducing", True
    if event["sum_of_costs_after"] < event["sum_of_costs_before"]:
        return "cost_improving", True
    return "neutral", False


def _cell_zone(
    cell: list[int],
    zone_sets: dict[str, set[tuple[int, int]]],
) -> str:
    value = tuple(cell)
    for zone in (
        "left_storage",
        "center_storage",
        "right_storage",
        "station_approach",
    ):
        if value in zone_sets.get(zone, set()):
            return zone
    return "free"


def _validate_path(
    path: list[list[int]],
    start: list[int],
    goal: list[int],
    grid: list[str],
) -> None:
    if not path or path[0] != start or path[-1] != goal:
        raise ValueError("recorded path has the wrong start or goal")
    rows = len(grid)
    cols = len(grid[0])
    for index, cell in enumerate(path):
        row, col = cell
        if (
            row < 0
            or row >= rows
            or col < 0
            or col >= cols
            or grid[row][col] != "."
        ):
            raise ValueError("recorded path uses a blocked cell")
        if index > 0:
            previous = path[index - 1]
            if (
                abs(row - previous[0]) + abs(col - previous[1])
                > 1
            ):
                raise ValueError("recorded path contains an invalid move")


def _validate_conflict_events(
    events: list[dict[str, Any]],
    map_document: dict[str, Any],
    agent_count: int,
) -> None:
    rows = map_document["rows"]
    cols = map_document["cols"]
    grid = map_document["grid"]
    for event in events:
        first, second = event["agents"]
        if not 0 <= first < second < agent_count:
            raise ValueError("conflict event has invalid agents")
        if int(event["timestep"]) < 0:
            raise ValueError("conflict event has invalid timestep")
        for row, col in event["cells"]:
            if (
                row < 0
                or row >= rows
                or col < 0
                or col >= cols
                or grid[row][col] != "."
            ):
                raise ValueError("conflict event uses a blocked cell")


def _map_features(
    map_document: dict[str, Any],
    manifest_row: dict[str, Any],
) -> dict[str, Any]:
    metadata = map_document["metadata"]
    parameters = metadata["sampled_parameters"]
    topology = metadata["topology_metrics"]
    total_cells = map_document["rows"] * map_document["cols"]
    shelf_cells = sum(
        row.count("S") for row in metadata["obstacle_type_layer"]
    )
    prior_values = [
        value
        for row in metadata["structural_congestion_prior"]
        for value in row
    ]
    return {
        "layout_mode": manifest_row["layout_mode"],
        "layout_variant": manifest_row.get("layout_variant"),
        "rows": map_document["rows"],
        "cols": map_document["cols"],
        "shelf_coverage": round(shelf_cells / total_cells, 6),
        "free_cell_ratio": round(
            metadata["free_cell_count"] / total_cells, 6
        ),
        "horizontal_aisle_width": parameters[
            "horizontal_aisle_width"
        ],
        "vertical_aisle_width": parameters["vertical_aisle_width"],
        "outer_beltway_width": parameters["outer_beltway_width"],
        "divider_wall_count": manifest_row["divider_wall_count"],
        "gate_count": manifest_row["gate_count"],
        "horizontal_dead_end_count": manifest_row[
            "horizontal_dead_end_count"
        ],
        "vertical_dead_end_count": manifest_row[
            "vertical_dead_end_count"
        ],
        "articulation_count": topology["articulation_count"],
        "dead_end_cell_count": topology["dead_end_cell_count"],
        "average_free_degree": topology["average_free_degree"],
        "mean_structural_prior": round(
            sum(prior_values) / len(prior_values), 6
        ),
        "maximum_structural_prior": max(prior_values),
    }


def _task_features(
    task_document: dict[str, Any],
    manifest_row: dict[str, Any],
) -> dict[str, Any]:
    metadata = task_document["metadata"]
    return {
        "task_variant": manifest_row.get("task_variant"),
        "scenario_type": manifest_row["scenario_type"],
        "agent_count": metadata["agent_count"],
        "agent_density_free_cells": metadata[
            "agent_density_free_cells"
        ],
        "agent_density_service_cells": metadata[
            "agent_density_service_cells"
        ],
        "dominant_flow_ratio": metadata["dominant_flow_ratio"],
        "hotspot_skew": metadata["hotspot_skew"],
        "origin_cluster_count": metadata["origin_cluster_count"],
        "goal_cluster_count": metadata["goal_cluster_count"],
        "cluster_radius": metadata["cluster_radius"],
        "mean_shortest_distance": metadata["mean_shortest_distance"],
        "realized_flow_counts": metadata["realized_flow_counts"],
    }


def _agent_descriptors(
    iteration: dict[str, Any],
    map_document: dict[str, Any],
    task_document: dict[str, Any],
    zone_sets: dict[str, set[tuple[int, int]]],
) -> list[dict[str, Any]]:
    metadata = task_document["metadata"]
    before_paths = {
        int(item["agent"]): item["path"]
        for item in iteration["neighborhood_paths_before"]
    }
    after_paths = {
        int(item["agent"]): item["path"]
        for item in iteration["neighborhood_paths_after"]
    }
    result = []
    for agent in iteration["neighborhood"]:
        start = task_document["starts"][agent]
        goal = task_document["goals"][agent]
        _validate_path(
            before_paths[agent], start, goal, map_document["grid"]
        )
        if agent in after_paths:
            _validate_path(
                after_paths[agent], start, goal, map_document["grid"]
            )
        result.append(
            {
                "agent": agent,
                "start": start,
                "goal": goal,
                "start_zone": _cell_zone(start, zone_sets),
                "goal_zone": _cell_zone(goal, zone_sets),
                "flow_assignment": metadata["flow_assignments"][agent],
                "shortest_distance": metadata[
                    "actual_shortest_distances"
                ][agent],
                "path_before": before_paths[agent],
                "path_after": after_paths.get(agent),
            }
        )
    return result


def _build_experience(
    dataset: str | Path,
    collection: str | Path,
    output: str | Path,
    split: str,
    usage: str,
) -> dict[str, Any]:
    if usage not in {"memory", "evaluation"}:
        raise ValueError(f"unsupported experience usage: {usage}")
    dataset_root = Path(dataset).resolve()
    collection_root = Path(collection).resolve()
    output_root = Path(output).resolve()
    dataset_manifest_path = dataset_root / split / "manifest.jsonl"
    collection_manifest_path = (
        collection_root / "collection_manifest.jsonl"
    )
    if not dataset_manifest_path.is_file():
        raise ValueError(f"missing dataset manifest: {dataset_manifest_path}")
    if not collection_manifest_path.is_file():
        raise ValueError(
            f"missing collection manifest: {collection_manifest_path}"
        )

    dataset_rows = _read_jsonl(dataset_manifest_path)
    task_rows = {row["task_id"]: row for row in dataset_rows}
    collection_rows = _read_jsonl(collection_manifest_path)
    repair_cases: list[dict[str, Any]] = []
    run_cases: list[dict[str, Any]] = []
    label_counts: collections.Counter[str] = collections.Counter()
    map_cache: dict[str, dict[str, Any]] = {}
    task_cache: dict[str, dict[str, Any]] = {}
    total_conflict_events = 0
    total_heatmap_weight = 0.0

    for run in collection_rows:
        if run["split"] != split:
            raise ValueError(
                f"collection contains a run outside split {split}"
            )
        if run["status"] == "error" or run.get("result") is None:
            raise ValueError("collection contains an invalid solver run")
        task_id = str(run["task_id"])
        if task_id not in task_rows:
            raise ValueError(f"unknown task in collection: {task_id}")
        manifest_row = task_rows[task_id]
        map_id = str(manifest_row["map_id"])
        if map_id not in map_cache:
            map_cache[map_id] = _read_json(
                dataset_root
                / split
                / str(manifest_row["map_file"])
            )
        if task_id not in task_cache:
            task_cache[task_id] = _read_json(
                dataset_root
                / split
                / str(manifest_row["task_file"])
            )
        map_document = map_cache[map_id]
        task_document = task_cache[task_id]
        map_features = _map_features(map_document, manifest_row)
        task_features = _task_features(task_document, manifest_row)
        zone_sets = {
            name: {tuple(cell) for cell in cells}
            for name, cells in map_document["metadata"]["zones"].items()
        }

        trace_path_value = run.get("trace_file")
        if trace_path_value is None:
            raise ValueError(f"run has no trace: {task_id}")
        trace_events = _read_jsonl(Path(trace_path_value))
        if not trace_events or trace_events[-1]["event_type"] != "summary":
            raise ValueError(f"trace has no summary: {trace_path_value}")
        if any(event.get("schema_version") != 2 for event in trace_events):
            raise ValueError(f"trace is not schema version 2: {trace_path_value}")
        iterations = [
            event
            for event in trace_events
            if event["event_type"] == "iteration"
        ]
        if len(iterations) != run["result"]["iterations"]:
            raise ValueError(f"trace iteration count mismatch: {task_id}")

        run_id = f"{task_id}__seed_{int(run['solver_seed']):04d}"
        run_heatmap: collections.defaultdict[
            tuple[int, int], float
        ] = collections.defaultdict(float)
        run_labels: collections.Counter[str] = collections.Counter()
        agent_stats: collections.defaultdict[int, dict[str, int]] = (
            collections.defaultdict(
                lambda: {"selected": 0, "effective": 0}
            )
        )
        effective_neighborhoods: list[dict[str, Any]] = []
        failed_neighborhoods: list[dict[str, Any]] = []
        neutral_neighborhoods: list[dict[str, Any]] = []
        accepted_conflict_reduction = 0
        accepted_cost_improvement = 0

        for iteration in iterations:
            before_events = iteration["conflict_events_before"]
            after_events = iteration["conflict_events_after"]
            _validate_conflict_events(
                before_events, map_document, task_document["metadata"][
                    "agent_count"
                ]
            )
            _validate_conflict_events(
                after_events, map_document, task_document["metadata"][
                    "agent_count"
                ]
            )
            if len(before_events) != iteration[
                "conflicting_pairs_before"
            ]:
                raise ValueError("before conflict event count mismatch")
            if iteration["candidate_valid"] and len(after_events) != (
                iteration["conflicting_pairs_after"]
            ):
                raise ValueError("after conflict event count mismatch")
            if len(iteration["neighborhood_paths_before"]) != len(
                iteration["neighborhood"]
            ):
                raise ValueError("pre-repair neighborhood paths missing")
            if iteration["candidate_valid"] and len(
                iteration["neighborhood_paths_after"]
            ) != len(iteration["neighborhood"]):
                raise ValueError("candidate neighborhood paths missing")

            before_heatmap, before_weight = _sparse_heatmap(
                before_events
            )
            after_heatmap, after_weight = _sparse_heatmap(after_events)
            if abs(before_weight - len(before_events)) > 1e-9:
                raise ValueError("before heatmap weight is not conserved")
            if abs(after_weight - len(after_events)) > 1e-9:
                raise ValueError("after heatmap weight is not conserved")
            _merge_heatmap(run_heatmap, before_heatmap)
            total_conflict_events += len(before_events)
            total_heatmap_weight += before_weight

            label, effective = _label_iteration(iteration)
            label_counts[label] += 1
            run_labels[label] += 1
            neighborhood_record = {
                "iteration": iteration["iteration"],
                "agents": iteration["neighborhood"],
                "label": label,
            }
            if effective:
                effective_neighborhoods.append(neighborhood_record)
            elif label in {"rejected", "invalid"}:
                failed_neighborhoods.append(neighborhood_record)
            else:
                neutral_neighborhoods.append(neighborhood_record)
            for agent in iteration["neighborhood"]:
                agent_stats[agent]["selected"] += 1
                if effective:
                    agent_stats[agent]["effective"] += 1

            conflict_reduction = (
                iteration["conflicting_pairs_before"]
                - iteration["conflicting_pairs_after"]
                if iteration["candidate_valid"]
                else None
            )
            cost_improvement = (
                iteration["sum_of_costs_before"]
                - iteration["sum_of_costs_after"]
                if iteration["candidate_valid"]
                else None
            )
            if iteration["accepted"] and conflict_reduction is not None:
                accepted_conflict_reduction += conflict_reduction
                accepted_cost_improvement += int(cost_improvement)

            repair_cases.append(
                {
                    "schema_version": 1,
                    "usage": usage,
                    "case_id": (
                        f"{run_id}__iteration_"
                        f"{int(iteration['iteration']):04d}"
                    ),
                    "run_id": run_id,
                    "split": split,
                    "map_id": map_id,
                    "task_id": task_id,
                    "solver_seed": run["solver_seed"],
                    "iteration": iteration["iteration"],
                    "map_features": map_features,
                    "task_features": task_features,
                    "conflict_events_before": before_events,
                    "conflict_events_after": after_events,
                    "conflict_heatmap_before": before_heatmap,
                    "conflict_heatmap_after": after_heatmap,
                    "seed_conflict": iteration["seed_conflict"],
                    "neighborhood": iteration["neighborhood"],
                    "agents": _agent_descriptors(
                        iteration,
                        map_document,
                        task_document,
                        zone_sets,
                    ),
                    "outcome": {
                        "label": label,
                        "effective": effective,
                        "candidate_valid": iteration["candidate_valid"],
                        "accepted": iteration["accepted"],
                        "conflicting_pairs_before": iteration[
                            "conflicting_pairs_before"
                        ],
                        "conflicting_pairs_after": iteration[
                            "conflicting_pairs_after"
                        ],
                        "conflict_reduction": conflict_reduction,
                        "sum_of_costs_before": iteration[
                            "sum_of_costs_before"
                        ],
                        "sum_of_costs_after": iteration[
                            "sum_of_costs_after"
                        ],
                        "cost_improvement": cost_improvement,
                        "replan_runtime_ms": iteration[
                            "replan_runtime_ms"
                        ],
                    },
                }
            )

        summary_event = trace_events[-1]
        run_cases.append(
            {
                "schema_version": 1,
                "usage": usage,
                "run_id": run_id,
                "split": split,
                "map_id": map_id,
                "task_id": task_id,
                "solver_seed": run["solver_seed"],
                "map_features": map_features,
                "task_features": task_features,
                "result": {
                    key: value
                    for key, value in summary_event.items()
                    if key not in {"schema_version", "event_type"}
                },
                "repair_case_count": len(iterations),
                "label_counts": dict(sorted(run_labels.items())),
                "conflict_heatmap": _serialize_heatmap(run_heatmap),
                "neighborhoods": {
                    "effective": effective_neighborhoods,
                    "failed": failed_neighborhoods,
                    "neutral": neutral_neighborhoods,
                },
                "agent_statistics": [
                    {"agent": agent, **values}
                    for agent, values in sorted(agent_stats.items())
                ],
                "accepted_conflict_reduction": (
                    accepted_conflict_reduction
                ),
                "accepted_cost_improvement": accepted_cost_improvement,
            }
        )

    _write_jsonl(output_root / "repair_cases.jsonl", repair_cases)
    _write_jsonl(output_root / "run_cases.jsonl", run_cases)
    summary = {
        "schema_version": 1,
        "source_trace_schema_version": 2,
        "split": split,
        "usage": usage,
        "run_count": len(run_cases),
        "repair_case_count": len(repair_cases),
        "map_count": len({row["map_id"] for row in run_cases}),
        "task_count": len({row["task_id"] for row in run_cases}),
        "label_counts": dict(sorted(label_counts.items())),
        "effective_case_count": sum(
            count
            for label, count in label_counts.items()
            if label in {"conflict_reducing", "cost_improving"}
        ),
        "conflict_event_count": total_conflict_events,
        "conflict_heatmap_weight": round(total_heatmap_weight, 6),
    }
    (output_root / "experience_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def build_repair_experience(
    dataset: str | Path,
    collection: str | Path,
    output: str | Path,
    split: str = "train",
) -> dict[str, Any]:
    if split != "train":
        raise ValueError("repair experience may only be built from train")
    return _build_experience(
        dataset=dataset,
        collection=collection,
        output=output,
        split=split,
        usage="memory",
    )


def build_query_experience(
    dataset: str | Path,
    collection: str | Path,
    output: str | Path,
    split: str = "validation",
) -> dict[str, Any]:
    if split != "validation":
        raise ValueError(
            "query experience may only be built from validation"
        )
    return _build_experience(
        dataset=dataset,
        collection=collection,
        output=output,
        split=split,
        usage="evaluation",
    )
