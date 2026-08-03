from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_boundary_ttf_quick import (
    _extended_controller_summary,
    boundary_ttf_quick_schedule,
    validate_boundary_ttf_quick_config,
)


class StrideBoundaryTTFQuickTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (root / "configs" / "stride_boundary_ttf_quick.json").read_text(
                encoding="utf-8"
            )
        )

    def test_quick_is_paired_exploratory_and_non_promoting(self) -> None:
        config = self._config()
        validate_boundary_ttf_quick_config(config)
        self.assertFalse(config["formal_speed_claim"])
        self.assertFalse(config["default_replacement_allowed"])
        self.assertFalse(config["cohort_independent_of_shadow_outcomes"])
        self.assertEqual(config["primary_metric"], "mean_capped_wall_time_to_feasible")
        self.assertEqual(config["wall_time_budget_seconds"], 60.0)

    def test_schedule_is_complete_and_pair_order_alternates(self) -> None:
        schedule = boundary_ttf_quick_schedule(self._config())
        self.assertEqual(len(schedule), 36)
        by_key = {}
        for row in schedule:
            by_key.setdefault((row["task_id"], row["solver_seed"]), []).append(row)
        self.assertEqual(len(by_key), 18)
        self.assertTrue(all(len(rows) == 2 for rows in by_key.values()))
        first_controllers = [rows[0]["controller"] for rows in by_key.values()]
        self.assertEqual(first_controllers.count("v2-full"), 9)
        self.assertEqual(first_controllers.count("v2-boundary-explore-v1"), 9)

    def test_quick_rejects_gate_or_runtime_drift(self) -> None:
        config = self._config()
        config["wall_time_budget_seconds"] = 300.0
        with self.assertRaisesRegex(ValueError, "runtime contract changed"):
            validate_boundary_ttf_quick_config(config)
        config = self._config()
        config["continuation_gates"]["minimum_capped_ttf_relative_improvement"] = 0.0
        with self.assertRaisesRegex(ValueError, "continuation gates changed"):
            validate_boundary_ttf_quick_config(config)

    def test_summary_counts_boundary_generation_and_selection(self) -> None:
        rows = [
            {
                "status": "ok",
                "summary": {
                    "success": True,
                    "capped_wall_time_to_feasible": 2.0,
                    "wall_time_to_feasible": 2.0,
                    "repair_iterations": 3,
                    "initial_conflicts": 10,
                    "controller_totals": {
                        "topology_boundary_generated_count": 4,
                        "topology_boundary_added_candidate_count": 3,
                        "topology_boundary_analysis_seconds": 0.2,
                    },
                    "selected_family_counts": {
                        "topology-boundary-articulation:16": 2,
                        "target:4": 1,
                    },
                },
            }
        ]
        summary = _extended_controller_summary(rows)
        self.assertEqual(summary["topology_boundary_generated_candidate_count"], 4)
        self.assertEqual(summary["topology_boundary_added_candidate_count"], 3)
        self.assertEqual(summary["topology_boundary_selected_repair_count"], 2)


if __name__ == "__main__":
    unittest.main()
