from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from lns2_selector.evaluation.anytime_handoff import (
    HANDOFF_SCHEMA, continue_with_official_anytime, modeled_completion, paths_from_observation,
)


class HandoffContractTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.map = Path(folder.name) / "map.map"
        self.map.write_text("type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n", encoding="utf-8")
        self.map.with_suffix(".scen").write_text("version 1\n0\tmap.map\t4\t3\t0\t0\t3\t0\t3\n0\tmap.map\t4\t3\t0\t2\t3\t2\t3\n", encoding="utf-8")
        self.state = {"feasible": True, "num_of_colliding_pairs": 0, "sum_of_costs": 8,
                      "agents": [{"id": 0, "path": [0, 0, 1, 2, 3]}, {"id": 1, "path": [8, 8, 9, 10, 11]}]}
        self.native = SimpleNamespace(anytime_handoff_schema=HANDOFF_SCHEMA)
        self.native.optimize_feasible_paths = Mock(return_value={
            "schema": HANDOFF_SCHEMA, "initial_planner_called": False, "initial_soc": 8,
            "initial_paths": paths_from_observation(self.state), "paths": [[0, 1, 2, 3], [8, 9, 10, 11]],
            "soc": 6, "stage2_seed": 17,
        })

    def run_handoff(self, values=(102.0, 103.0, 105.0), **overrides):
        args = dict(module=self.native, map_path=self.map, scenario_path=self.map.with_suffix(".scen"),
                    observation=self.state, expected_native_sha256="frozen", planning_started=100.0,
                    total_budget_seconds=10.0, stage2_seed=17, clock=Mock(side_effect=values))
        args.update(overrides)
        with patch("lns2_selector.evaluation.anytime_handoff.native_identity", return_value={"sha256": "frozen"}):
            return continue_with_official_anytime(**args)

    def test_shared_deadline_and_exact_handoff(self):
        original = copy.deepcopy(self.state)
        result = self.run_handoff()
        self.assertEqual(result["stage2_remaining_budget_seconds"], 7.0)
        self.assertEqual(result["dispatch_wall_seconds"], 10.0)
        self.assertEqual(result["final_quality"]["soc_steps"], 6)
        self.assertEqual(self.state, original)
        self.assertEqual(self.native.optimize_feasible_paths.call_args.args[-1], 7.0)

    def test_no_remaining_budget_never_calls_native(self):
        result = self.run_handoff(values=(109.0, 111.0, 112.0))
        self.native.optimize_feasible_paths.assert_not_called()
        self.assertTrue(result["success_by_deadline"])
        self.assertEqual(result["dispatch_wall_seconds"], 12.0)

    def test_late_first_solution_is_not_success(self):
        result = self.run_handoff(values=(111.0, 112.0, 113.0))
        self.assertFalse(result["success_by_deadline"])

    def test_preserved_first_solution_timestamp_survives_finalization(self):
        result = self.run_handoff(values=(111.0, 112.0, 113.0), initial_feasible_elapsed_seconds=9.0)
        self.assertTrue(result["success_by_deadline"])
        self.native.optimize_feasible_paths.assert_not_called()

    def test_future_first_solution_timestamp_rejected(self):
        with self.assertRaisesRegex(ValueError, "timestamp"):
            self.run_handoff(initial_feasible_elapsed_seconds=3.0)

    def test_zero_budget_still_checks_task_endpoints(self):
        self.state["agents"][0]["path"] = [0, 0, 1, 2]
        self.state["sum_of_costs"] = 7
        with self.assertRaisesRegex(ValueError, "endpoints"):
            self.run_handoff(values=(111.0, 112.0, 113.0))
        self.native.optimize_feasible_paths.assert_not_called()

    def test_native_overshoot_is_charged(self):
        result = self.run_handoff(values=(102.0, 103.0, 111.5))
        self.assertEqual(result["budget_overshoot_seconds"], 1.5)
        self.assertEqual(result["dispatch_wall_seconds"], 11.5)

    def test_reject_wrong_identity(self):
        with self.assertRaisesRegex(ValueError, "SHA"):
            self.run_handoff(expected_native_sha256="wrong")
        self.native.optimize_feasible_paths.assert_not_called()

    def test_reject_silent_initial_replanning(self):
        self.native.optimize_feasible_paths.return_value["initial_planner_called"] = True
        with self.assertRaisesRegex(ValueError, "handoff"):
            self.run_handoff()

    def test_reject_changed_initial_paths(self):
        self.native.optimize_feasible_paths.return_value["initial_paths"] = [[0, 1, 2, 3], [8, 9, 10, 11]]
        with self.assertRaisesRegex(ValueError, "imported"):
            self.run_handoff()

    def test_reject_invalid_returned_path(self):
        self.native.optimize_feasible_paths.return_value["paths"][0] = [0, 3]
        with self.assertRaisesRegex(ValueError, "illegal move"):
            self.run_handoff()

    def test_reject_infeasible_input(self):
        self.state["feasible"] = False
        with self.assertRaises(ValueError):
            self.run_handoff()

    def test_agent_ordering(self):
        self.state["agents"].reverse()
        self.assertEqual(paths_from_observation(self.state)[0], [0, 0, 1, 2, 3])
        self.state["agents"][0]["id"] = 0
        with self.assertRaises(ValueError):
            paths_from_observation(self.state)

    def test_modeled_completion_units(self):
        self.assertEqual(modeled_completion(5, 100, 0.5), 55)
        for bad in (float("nan"), -1, True):
            with self.assertRaises(ValueError):
                modeled_completion(bad, 100, 0.5)


if __name__ == "__main__":
    unittest.main()
