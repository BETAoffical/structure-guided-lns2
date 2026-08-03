from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.closed_loop_confirmation import _with_stopping_rule
from experiments.stride_augcontrol_evaluation import (
    CONTROLLERS,
    _controller_kwargs,
    _paired_comparison,
    _registered_tasks,
    _schedule,
    validate_augcontrol_evaluation_config,
)
from experiments.stride_repairability_collection import BOUNDARY_AUGMENTATION


class StrideAugcontrolEvaluationTest(unittest.TestCase):
    @staticmethod
    def _root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _config(self) -> dict:
        return json.loads(
            (
                self._root()
                / "configs"
                / "stride_augcontrol_evaluation.json"
            ).read_text(
                encoding="utf-8"
            )
        )

    def test_registration_freezes_raw_ttf_three_way_decomposition(self) -> None:
        config = self._config()
        validate_augcontrol_evaluation_config(config)
        self.assertEqual(tuple(config["controllers"]), CONTROLLERS)
        self.assertEqual(config["stopping_rule"], "run-to-completion")
        self.assertIsNone(config["scientific_time_limit_seconds"])
        self.assertEqual(
            config["topology_boundary_augmentation"], BOUNDARY_AUGMENTATION
        )
        self.assertEqual(
            config["formal_ood"]["cohorts"][0],
            {
                "id": "maze200",
                "map_id": "maze-128-128-10",
                "agent_count": 200,
            },
        )

    def test_registration_rejects_cap_or_candidate_pool_drift(self) -> None:
        config = self._config()
        config["scientific_time_limit_seconds"] = 60
        with self.assertRaisesRegex(ValueError, "identity changed"):
            validate_augcontrol_evaluation_config(config)
        config = self._config()
        config["topology_boundary_augmentation"]["maximum_added_candidates"] = 3
        with self.assertRaisesRegex(ValueError, "candidate pool"):
            validate_augcontrol_evaluation_config(config)

    def test_runtime_defaults_are_overridden_to_true_run_to_completion(self) -> None:
        config = self._config()
        bundles = {
            "v2-full": self._root() / "artifacts" / "initlns-closed-loop-controller-v2",
            "stride-augcontrol-v1": self._root() / "build" / "diagnostic-bundle",
        }
        for controller in CONTROLLERS:
            kwargs = _controller_kwargs(
                controller,
                bundles,
                config["topology_boundary_augmentation"],
            )
            self.assertEqual(kwargs["stopping_rule"], "run-to-completion")
        for runtime_name in (
            "stride_stage4r_high_load_runtime.json",
            "stride_augcontrol_ood_runtime.json",
        ):
            runtime = json.loads(
                (self._root() / "configs" / runtime_name).read_text(
                    encoding="utf-8"
                )
            )
            effective = _with_stopping_rule(runtime, "run-to-completion")
            self.assertEqual(effective["environment"]["time_limit"], 0.0)
            self.assertTrue(effective["environment"]["unlimited_time"])
            self.assertEqual(effective["environment"]["max_repair_iterations"], 0)
            self.assertIsNone(effective["wall_time_budget_seconds"])
            self.assertIsNone(effective["episode_process_timeout_seconds"])
            self.assertIsNone(effective["metric_iteration_budget"])

    def test_schedule_rotates_base_pool_and_ranker_controllers(self) -> None:
        cohorts = [
            {"id": "a", "tasks": ["a1", "a2"]},
            {"id": "b", "tasks": ["b1", "b2"]},
        ]
        schedule = _schedule(cohorts, (1, 2))
        self.assertEqual(len(schedule), 24)
        for offset in range(0, len(schedule), 3):
            rows = schedule[offset : offset + 3]
            self.assertEqual({row["controller"] for row in rows}, set(CONTROLLERS))
            self.assertEqual(len({row["within_key_position"] for row in rows}), 3)

    def test_raw_ttf_comparison_separates_ttf_and_repair_rounds(self) -> None:
        keys = [("cohort", "task", 1), ("cohort", "task", 2)]

        def row(ttf: float, rounds: int) -> dict:
            return {
                "status": "ok",
                "summary": {
                    "success": True,
                    "wall_time_to_feasible": ttf,
                    "repair_iterations": rounds,
                    "normalized_wall_clock_conflict_auc": 0.5,
                },
            }

        baseline = {keys[0]: row(10.0, 10), keys[1]: row(20.0, 20)}
        challenger = {keys[0]: row(8.0, 8), keys[1]: row(16.0, 16)}
        report = _paired_comparison(baseline, challenger, keys)
        self.assertTrue(report["valid"])
        self.assertAlmostEqual(report["mean_raw_ttf_relative_improvement"], 0.2)
        self.assertAlmostEqual(report["mean_repair_iterations_delta"], -3.0)
        self.assertEqual(report["faster_count"], 2)

    def test_formal_task_ids_use_two_scenarios_and_fixed_load(self) -> None:
        tasks = _registered_tasks(
            {"map_id": "room-64-64-16", "agent_count": 400}, True
        )
        self.assertEqual(
            tasks,
            [
                "room-64-64-16__random_04__agents_0400",
                "room-64-64-16__random_05__agents_0400",
            ],
        )

    def test_formal_dataset_and_runtime_register_identical_unseen_loads(self) -> None:
        dataset = json.loads(
            (
                self._root() / "configs" / "stride_augcontrol_ood_dataset.json"
            ).read_text(encoding="utf-8")
        )
        runtime = json.loads(
            (
                self._root() / "configs" / "stride_augcontrol_ood_runtime.json"
            ).read_text(encoding="utf-8")
        )
        evaluation = self._config()["formal_ood"]
        dataset_loads = {
            str(row["id"]): int(row["agent_counts"][0])
            for row in dataset["benchmarks"]
        }
        runtime_loads = {
            str(row["map_id"]): int(row["agent_counts"][0])
            for row in runtime["dataset_design"]["maps"]
        }
        evaluation_loads = {
            str(row["map_id"]): int(row["agent_count"])
            for row in evaluation["cohorts"]
        }
        self.assertEqual(dataset_loads, runtime_loads)
        self.assertEqual(runtime_loads, evaluation_loads)
        self.assertTrue(runtime["formal"])
        self.assertEqual(runtime["dataset_design"]["task_count"], 12)
        self.assertTrue(
            set(runtime_loads).isdisjoint(
                runtime["dataset_design"]["historical_map_ids"]
            )
        )


if __name__ == "__main__":
    unittest.main()
