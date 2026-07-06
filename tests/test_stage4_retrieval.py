from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generators.repair_experience import (
    build_query_experience,
    build_repair_experience,
)
from generators.retrieval import (
    _rank_neighbors,
    build_retrieval_index,
    evaluate_retrieval,
    fit_feature_schema,
    vectorize,
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


def _map_features(index: int, layout: str) -> dict:
    return {
        "layout_mode": layout,
        "layout_variant": None,
        "rows": 10,
        "cols": 12,
        "shelf_coverage": 0.2 + index * 0.01,
        "free_cell_ratio": 0.7 - index * 0.01,
        "horizontal_aisle_width": 2,
        "vertical_aisle_width": 2,
        "outer_beltway_width": 2,
        "divider_wall_count": index % 2,
        "gate_count": (index % 2) * 2,
        "horizontal_dead_end_count": 0,
        "vertical_dead_end_count": 0,
        "articulation_count": index,
        "dead_end_cell_count": index % 2,
        "average_free_degree": 2.5 + index * 0.1,
        "mean_structural_prior": 0.1 + index * 0.01,
        "maximum_structural_prior": 1.0 + index,
    }


def _task_features(index: int, variant: str) -> dict:
    agent_count = 20 + index * 4
    return {
        "task_variant": variant,
        "scenario_type": "balanced_bidirectional",
        "agent_count": agent_count,
        "agent_density_free_cells": agent_count / 100,
        "agent_density_service_cells": agent_count / 50,
        "dominant_flow_ratio": 1.0,
        "hotspot_skew": 0.2 * index,
        "origin_cluster_count": index % 3,
        "goal_cluster_count": index % 3,
        "cluster_radius": 2 + index,
        "mean_shortest_distance": 10 + index,
        "realized_flow_counts": {
            "left_to_right": agent_count // 2,
            "right_to_left": agent_count - agent_count // 2,
        },
    }


def _run_case(
    task_index: int,
    seed: int,
    split: str,
    usage: str,
) -> dict:
    task_id = f"{split}_task_{task_index}"
    heatmap = (
        [{"cell": [task_index + 1, task_index + 2], "weight": 1.0}]
        if (task_index + seed) % 2
        else []
    )
    return {
        "schema_version": 1,
        "usage": usage,
        "run_id": f"{task_id}__seed_{seed:04d}",
        "split": split,
        "map_id": f"{split}_map_{task_index}",
        "task_id": task_id,
        "solver_seed": seed,
        "map_features": _map_features(
            task_index, "regular_beltway" if task_index % 2 else "compartmentalized"
        ),
        "task_features": _task_features(
            task_index, "balanced_base" if task_index % 2 else "balanced_dense"
        ),
        "result": {"success": True},
        "repair_case_count": 1,
        "label_counts": {},
        "conflict_heatmap": heatmap,
        "neighborhoods": {"effective": [], "failed": [], "neutral": []},
        "agent_statistics": [],
        "accepted_conflict_reduction": 0,
        "accepted_cost_improvement": 0,
    }


def _repair_case(
    task_index: int,
    seed: int,
    split: str,
    usage: str,
    effective: bool,
) -> dict:
    task_id = f"{split}_task_{task_index}"
    run_id = f"{task_id}__seed_{seed:04d}"
    flow = "left_to_right" if task_index % 2 else "right_to_left"
    reverse = "right_to_left" if task_index % 2 else "left_to_right"
    return {
        "schema_version": 1,
        "usage": usage,
        "case_id": f"{run_id}__iteration_0001",
        "run_id": run_id,
        "split": split,
        "map_id": f"{split}_map_{task_index}",
        "task_id": task_id,
        "solver_seed": seed,
        "iteration": 1,
        "map_features": _map_features(
            task_index, "regular_beltway" if task_index % 2 else "compartmentalized"
        ),
        "task_features": _task_features(
            task_index, "balanced_base" if task_index % 2 else "balanced_dense"
        ),
        "conflict_events_before": [
            {
                "agents": [0, 1],
                "timestep": 2,
                "type": "vertex" if task_index % 2 else "edge_swap",
                "cells": (
                    [[task_index + 1, task_index + 2]]
                    if task_index % 2
                    else [
                        [task_index + 1, task_index + 2],
                        [task_index + 1, task_index + 3],
                    ]
                ),
            }
        ],
        "conflict_events_after": [],
        "conflict_heatmap_before": [
            {"cell": [task_index + 1, task_index + 2], "weight": 1.0}
        ],
        "conflict_heatmap_after": [],
        "seed_conflict": [0, 1],
        "neighborhood": [0, 1],
        "agents": [
            {
                "agent": 0,
                "start": [0, 0],
                "goal": [0, 2],
                "start_zone": "left_storage",
                "goal_zone": "right_storage",
                "flow_assignment": flow,
                "shortest_distance": 2 + task_index,
                "path_before": [[0, 0], [0, 1], [0, 2]],
                "path_after": [[0, 0], [0, 1], [0, 2]],
            },
            {
                "agent": 1,
                "start": [1, 0],
                "goal": [1, 2],
                "start_zone": "right_storage",
                "goal_zone": "left_storage",
                "flow_assignment": reverse,
                "shortest_distance": 3 + task_index,
                "path_before": [[1, 0], [1, 1], [1, 2]],
                "path_after": [[1, 0], [1, 1], [1, 2]],
            },
        ],
        "outcome": {
            "label": "conflict_reducing" if effective else "rejected",
            "effective": effective,
        },
    }


class Stage4RetrievalTests(unittest.TestCase):
    def test_schema_drops_constants_and_handles_unknown_categories(self) -> None:
        rows = [
            {
                "numeric": {"map.constant": 1.0, "task.variable": 1.0},
                "categorical": {"conflict.kind": "vertex"},
            },
            {
                "numeric": {"map.constant": 1.0, "task.variable": 3.0},
                "categorical": {"conflict.kind": "edge_swap"},
            },
        ]
        schema = fit_feature_schema(rows)
        self.assertIn("map.constant", schema["dropped_zero_variance"])
        unknown = vectorize(
            {
                "numeric": {"map.constant": 1.0, "task.variable": 2.0},
                "categorical": {"conflict.kind": "new_kind"},
            },
            schema,
        )
        unknown_index = next(
            index
            for index, field in enumerate(schema["fields"])
            if field["name"] == "conflict.kind=__unknown__"
        )
        self.assertEqual(unknown[unknown_index], 1.0)

    def test_repair_neighbor_diversity(self) -> None:
        schema = {
            "fields": [
                {
                    "name": "task.value",
                    "kind": "numeric",
                    "group": "task",
                }
            ]
        }
        entries = [
            {
                "case_id": f"case_{index}",
                "run_id": run,
                "task_id": task,
                "vector": [float(index)],
            }
            for index, (run, task) in enumerate(
                [
                    ("run_a", "task_a"),
                    ("run_a", "task_a"),
                    ("run_b", "task_a"),
                    ("run_c", "task_a"),
                    ("run_d", "task_b"),
                ]
            )
        ]
        neighbors = _rank_neighbors(
            [0.0], entries, schema, {"task": 1.0}, 4, "repair"
        )
        self.assertEqual(len(neighbors), 3)
        self.assertEqual(
            len({entry["run_id"] for entry, _ in neighbors}), 3
        )
        self.assertLessEqual(
            sum(entry["task_id"] == "task_a" for entry, _ in neighbors), 2
        )

    def test_index_and_evaluation_are_isolated_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            queries = root / "queries"
            index = root / "index"
            first_output = root / "evaluation_a"
            second_output = root / "evaluation_b"
            train_runs = [
                _run_case(task, seed, "train", "memory")
                for task in range(4)
                for seed in (1, 2, 3)
            ]
            train_repairs = [
                _repair_case(
                    task, (task % 3) + 1, "train", "memory", task % 2 == 0
                )
                for task in range(4)
            ]
            _write_jsonl(memory / "run_cases.jsonl", train_runs)
            _write_jsonl(memory / "repair_cases.jsonl", train_repairs)
            _write_json(
                memory / "experience_summary.json",
                {"split": "train", "usage": "memory"},
            )
            validation_runs = [
                _run_case(4, seed, "validation", "evaluation")
                for seed in (1, 2, 3)
            ]
            validation_repairs = [
                _repair_case(
                    4, 1, "validation", "evaluation", True
                )
            ]
            _write_jsonl(queries / "run_cases.jsonl", validation_runs)
            _write_jsonl(
                queries / "repair_cases.jsonl", validation_repairs
            )
            _write_json(
                queries / "experience_summary.json",
                {"split": "validation", "usage": "evaluation"},
            )

            summary = build_retrieval_index(memory, index)
            self.assertEqual(summary["source_run_count"], 12)
            self.assertEqual(summary["run_prototype_count"], 4)
            self.assertEqual(summary["repair_case_count"], 4)
            repair_text = (index / "repair_index.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertNotIn('"agent":', repair_text)
            self.assertNotIn("path_after", repair_text)
            self.assertNotIn('"outcome":', repair_text)

            first = evaluate_retrieval(index, queries, first_output)
            second = evaluate_retrieval(index, queries, second_output)
            self.assertEqual(first, second)
            self.assertEqual(first["index_split"], "train")
            self.assertEqual(first["query_split"], "validation")
            self.assertFalse(first["test_data_read"])
            self.assertEqual(
                (first_output / "repair_guidance.jsonl").read_text(
                    encoding="utf-8"
                ),
                (second_output / "repair_guidance.jsonl").read_text(
                    encoding="utf-8"
                ),
            )

    def test_experience_entrypoints_reject_wrong_splits(self) -> None:
        with self.assertRaisesRegex(ValueError, "train"):
            build_repair_experience("missing", "missing", "missing", "test")
        with self.assertRaisesRegex(ValueError, "validation"):
            build_query_experience("missing", "missing", "missing", "test")


if __name__ == "__main__":
    unittest.main()
