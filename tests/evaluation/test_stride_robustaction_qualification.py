from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_qualification import (
    registered_runtime_matches,
    select_qualified_tasks,
    validate_robustaction_qualification_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_robustaction_structpool_qualification_design.json"


class RobustActionQualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_reset_only_design_is_valid(self) -> None:
        validate_robustaction_qualification_design(
            self.config, project_root=ROOT
        )

    def test_outcome_field_drift_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["forbidden_selection_inputs"].remove("candidate_runtime")
        with self.assertRaisesRegex(ValueError, "outcome boundary"):
            validate_robustaction_qualification_design(changed, project_root=ROOT)

    def test_selection_prefers_two_targets_and_distinct_variants(self) -> None:
        summaries = [
            {
                "map_id": "map-a",
                "task_id": "a-low",
                "task_variant": "uniform_random",
                "agent_count": 100,
                "mean_initial_conflicts": 24.0,
                "nonzero_solver_seed_fraction": 1.0,
            },
            {
                "map_id": "map-a",
                "task_id": "a-high-same",
                "task_variant": "uniform_random",
                "agent_count": 200,
                "mean_initial_conflicts": 99.0,
                "nonzero_solver_seed_fraction": 1.0,
            },
            {
                "map_id": "map-a",
                "task_id": "a-high-other",
                "task_variant": "opposite_exchange",
                "agent_count": 210,
                "mean_initial_conflicts": 92.0,
                "nonzero_solver_seed_fraction": 1.0,
            },
        ]
        selected, underloaded = select_qualified_tasks(summaries, self.config)
        self.assertFalse(underloaded)
        self.assertEqual(
            {row["task_id"] for row in selected},
            {"a-low", "a-high-other"},
        )

    def test_underloaded_map_is_reported_without_resampling(self) -> None:
        selected, underloaded = select_qualified_tasks(
            [
                {
                    "map_id": "map-a",
                    "task_id": "zero",
                    "task_variant": "uniform_random",
                    "agent_count": 1000,
                    "mean_initial_conflicts": 0.0,
                    "nonzero_solver_seed_fraction": 0.0,
                }
            ],
            self.config,
        )
        self.assertFalse(selected)
        self.assertEqual(underloaded, ["map-a"])

    def test_resolved_runtime_defaults_do_not_create_a_false_mismatch(self) -> None:
        registered = {
            "solver_seeds": [1, 2],
            "environment": {"time_limit": 600.0, "max_repair_iterations": 0},
            "metric_iteration_budget": 100,
        }
        observed = {
            "configuration": {
                **copy.deepcopy(registered),
                "metric_iteration_budget": None,
                "stopping_rule": "wall-clock",
                "controller": "v2-full",
                "feature_backend": "auto",
            }
        }
        self.assertTrue(registered_runtime_matches(observed, registered))
        observed["configuration"]["environment"]["time_limit"] = 599.0
        self.assertFalse(registered_runtime_matches(observed, registered))


if __name__ == "__main__":
    unittest.main()
