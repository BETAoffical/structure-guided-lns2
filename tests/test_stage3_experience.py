from __future__ import annotations

import collections
import json
import tempfile
import unittest
from pathlib import Path

from generators.io import map_document, task_document
from generators.repair_experience import (
    _label_iteration,
    _sparse_heatmap,
    build_repair_experience,
)
from generators.task_flows import generate_tasks
from generators.warehouse import generate_warehouse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads(
    (PROJECT_ROOT / "configs" / "stage1_example.json").read_text(
        encoding="utf-8"
    )
)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _shortest_path(
    grid: list[str], start: tuple[int, int], goal: tuple[int, int]
) -> list[list[int]]:
    open_cells = collections.deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {
        start: None
    }
    while open_cells:
        row, col = open_cells.popleft()
        if (row, col) == goal:
            break
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            neighbor = (row + dr, col + dc)
            nr, nc = neighbor
            if (
                0 <= nr < len(grid)
                and 0 <= nc < len(grid[0])
                and grid[nr][nc] == "."
                and neighbor not in parent
            ):
                parent[neighbor] = (row, col)
                open_cells.append(neighbor)
    path = []
    current: tuple[int, int] | None = goal
    while current is not None:
        path.append([current[0], current[1]])
        current = parent[current]
    return list(reversed(path))


class RepairExperienceTests(unittest.TestCase):
    def test_labels_cover_success_and_failure_cases(self) -> None:
        base = {
            "candidate_valid": True,
            "accepted": True,
            "conflicting_pairs_before": 2,
            "conflicting_pairs_after": 1,
            "sum_of_costs_before": 10,
            "sum_of_costs_after": 12,
        }
        self.assertEqual(
            _label_iteration(base), ("conflict_reducing", True)
        )
        self.assertEqual(
            _label_iteration(
                {
                    **base,
                    "conflicting_pairs_after": 2,
                    "sum_of_costs_after": 9,
                }
            ),
            ("cost_improving", True),
        )
        self.assertEqual(
            _label_iteration(
                {
                    **base,
                    "conflicting_pairs_after": 2,
                    "sum_of_costs_after": 10,
                }
            ),
            ("neutral", False),
        )
        self.assertEqual(
            _label_iteration({**base, "accepted": False}),
            ("rejected", False),
        )
        self.assertEqual(
            _label_iteration({**base, "candidate_valid": False}),
            ("invalid", False),
        )

    def test_edge_heatmap_splits_weight(self) -> None:
        sparse, weight = _sparse_heatmap(
            [
                {
                    "type": "edge_swap",
                    "cells": [[1, 2], [1, 3]],
                }
            ]
        )
        self.assertEqual(weight, 1.0)
        self.assertEqual(
            sparse,
            [
                {"cell": [1, 2], "weight": 0.5},
                {"cell": [1, 3], "weight": 0.5},
            ],
        )

    def test_builds_train_repair_and_run_cases(self) -> None:
        map_data = generate_warehouse(
            CONFIG["map"], 901, "experience_map"
        )
        task_data = generate_tasks(
            map_data,
            {
                **CONFIG["task"],
                "agent_count": 2,
                "scenario_type": "uniform_random",
                "minimum_shortest_distance": 2,
                "hotspot_skew": 0.0,
            },
            902,
            "experience_task",
        )
        paths = [
            _shortest_path(map_data.grid, start, goal)
            for start, goal in zip(task_data.starts, task_data.goals)
        ]
        conflict_cell = paths[0][0]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            collection = root / "collection"
            output = root / "output"
            map_path = dataset / "train" / "maps" / "experience_map.json"
            task_path = (
                dataset
                / "train"
                / "instances"
                / "experience_task.json"
            )
            _write_json(map_path, map_document(map_data))
            _write_json(task_path, task_document(task_data))
            manifest_row = {
                "split": "train",
                "map_id": "experience_map",
                "task_id": "experience_task",
                "map_file": "maps/experience_map.json",
                "task_file": "instances/experience_task.json",
                "layout_mode": "regular_beltway",
                "layout_variant": None,
                "task_variant": "uniform_control",
                "scenario_type": "uniform_random",
                "divider_wall_count": 0,
                "gate_count": 0,
                "horizontal_dead_end_count": 0,
                "vertical_dead_end_count": 0,
            }
            _write_jsonl(
                dataset / "train" / "manifest.jsonl",
                [manifest_row],
            )
            iteration = {
                "schema_version": 2,
                "event_type": "iteration",
                "solver_seed": 1,
                "iteration": 1,
                "seed_conflict": [0, 1],
                "neighborhood": [0, 1],
                "conflicting_pairs_before": 1,
                "conflicting_pairs_after": 0,
                "sum_of_costs_before": 10,
                "sum_of_costs_after": 11,
                "candidate_valid": True,
                "accepted": True,
                "replan_runtime_ms": 1.25,
                "conflict_events_before": [
                    {
                        "agents": [0, 1],
                        "timestep": 0,
                        "type": "vertex",
                        "cells": [conflict_cell],
                    }
                ],
                "conflict_events_after": [],
                "neighborhood_paths_before": [
                    {"agent": 0, "path": paths[0]},
                    {"agent": 1, "path": paths[1]},
                ],
                "neighborhood_paths_after": [
                    {"agent": 0, "path": paths[0]},
                    {"agent": 1, "path": paths[1]},
                ],
            }
            summary_event = {
                "schema_version": 2,
                "event_type": "summary",
                "solver_seed": 1,
                "success": True,
                "initial_conflicting_pairs": 1,
                "final_conflicting_pairs": 0,
                "iterations": 1,
                "accepted_iterations": 1,
                "makespan": 10,
                "sum_of_costs": 11,
                "runtime_ms": 2.0,
            }
            trace_path = collection / "trace.jsonl"
            _write_jsonl(trace_path, [iteration, summary_event])
            collection_row = {
                **manifest_row,
                "solver_seed": 1,
                "trace_file": str(trace_path),
                "status": "solved",
                "result": {
                    "success": True,
                    "iterations": 1,
                },
            }
            _write_jsonl(
                collection / "collection_manifest.jsonl",
                [collection_row],
            )

            summary = build_repair_experience(
                dataset, collection, output
            )
            self.assertEqual(summary["run_count"], 1)
            self.assertEqual(summary["repair_case_count"], 1)
            self.assertEqual(summary["conflict_heatmap_weight"], 1.0)
            self.assertEqual(
                summary["label_counts"], {"conflict_reducing": 1}
            )
            repair = json.loads(
                (output / "repair_cases.jsonl").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                repair["outcome"]["label"], "conflict_reducing"
            )
            self.assertEqual(len(repair["agents"]), 2)
            run = json.loads(
                (output / "run_cases.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(run["repair_case_count"], 1)
            self.assertEqual(
                run["conflict_heatmap"],
                [{"cell": conflict_cell, "weight": 1.0}],
            )


if __name__ == "__main__":
    unittest.main()
