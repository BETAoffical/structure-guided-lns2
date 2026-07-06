from __future__ import annotations

import unittest

from generators.guided_solver import RepairGuide
from generators.stage5 import _comparison, run_stage5_experiment


def _run(
    task: str,
    seed: int,
    strategy: str,
    success: bool,
    conflicts: int,
    cost: int,
) -> dict:
    return {
        "task_id": task,
        "solver_seed": seed,
        "layout_mode": "regular_beltway",
        "task_variant": "balanced_dense",
        "strategy": strategy,
        "result": {
            "success": success,
            "final_conflicting_pairs": conflicts,
            "iterations": 2,
            "accepted_iterations": 1,
            "sum_of_costs": cost,
            "makespan": 10,
            "runtime_ms": 20.0,
            "search_runtime_ms": 18.0,
            "guidance_runtime_ms": 2.0 if strategy == "guided_lns2" else 0.0,
            "guidance_requests": 2 if strategy == "guided_lns2" else 0,
            "guidance_used": 1 if strategy == "guided_lns2" else 0,
            "guidance_fallbacks": 1 if strategy == "guided_lns2" else 0,
        },
        "effective_guided_iteration_count": (
            1 if strategy == "guided_lns2" else 0
        ),
    }


class Stage5GuidanceTests(unittest.TestCase):
    def test_role_mapping_keeps_seed_pair_and_is_unique(self) -> None:
        guide = RepairGuide.__new__(RepairGuide)
        guide.role_weight = 0.5
        guide.conflict_weight = 0.35
        guide.baseline_weight = 0.15
        request = {
            "seed_conflict": [0, 1],
            "baseline_neighborhood": [0, 1, 2, 3, 4, 5],
            "conflict_events": [
                {"agents": [0, 1]},
                {"agents": [1, 2]},
                {"agents": [2, 3]},
            ],
        }
        agents = [
            {
                "agent": agent,
                "start_zone": (
                    "left_storage" if agent % 2 == 0 else "right_storage"
                ),
                "goal_zone": (
                    "right_storage" if agent % 2 == 0 else "left_storage"
                ),
                "flow_assignment": (
                    "left_to_right" if agent % 2 == 0 else "right_to_left"
                ),
                "shortest_distance": 10 + agent,
                "path_before": [[agent, 0], [agent, 1]],
            }
            for agent in range(8)
        ]
        query = {
            "agents": agents,
            "conflict_heatmap_before": [
                {"cell": [2, 1], "weight": 1.0}
            ],
        }
        template = {
            "additional_role_distribution": [
                {
                    "role": (
                        "left_storage->right_storage|left_to_right"
                    ),
                    "probability": 1.0,
                }
            ],
            "mean_shortest_distance": 12.0,
            "mean_path_stretch": 0.2,
            "mean_path_conflict_overlap": 0.1,
        }
        selected = guide._map_agents(request, query, template)
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(set(selected)), 6)
        self.assertIn(0, selected)
        self.assertIn(1, selected)

    def test_paired_comparison_prefers_success_then_conflicts(self) -> None:
        baseline = [
            _run("task_a", 1, "baseline_lns2", False, 2, 10),
            _run("task_b", 1, "baseline_lns2", True, 0, 20),
        ]
        guided = [
            _run("task_a", 1, "guided_lns2", True, 0, 15),
            _run("task_b", 1, "guided_lns2", True, 0, 18),
        ]
        summary = _comparison(baseline, guided)
        self.assertEqual(summary["paired_outcomes"]["guided_win"], 2)
        self.assertEqual(summary["guided"]["solved_count"], 2)
        self.assertEqual(summary["guided"]["guidance_used"], 2)
        self.assertEqual(
            summary["paired_statistics"]["success_mcnemar_exact_p"],
            1.0,
        )
        self.assertEqual(
            summary["paired_statistics"]["execution_order"][
                "baseline_first"
            ],
            2,
        )

    def test_experiment_split_guards(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation or test"):
            run_stage5_experiment(
                "missing",
                "missing",
                "missing",
                "missing",
                "missing",
                "train",
                [1],
            )
        with self.assertRaisesRegex(ValueError, "frozen"):
            run_stage5_experiment(
                "missing",
                "missing",
                "missing",
                "missing",
                "missing",
                "test",
                [1],
            )


if __name__ == "__main__":
    unittest.main()
