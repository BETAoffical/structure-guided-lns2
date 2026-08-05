from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_map_replacement import (
    REPLACED_MAPS,
    REPLACEMENT_MAPS,
    validate_map_replacement_qualification_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_map_replacement_qualification_design.json"
)
SOURCE_RUNTIME = (
    ROOT / "configs" / "stride_robustaction_structpool_source_runtime.json"
)


class RobustActionMapReplacementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_replacement_qualification_is_valid(self) -> None:
        validate_map_replacement_qualification_design(
            self.config, project_root=ROOT
        )
        self.assertEqual(len(REPLACED_MAPS), 4)
        self.assertEqual(len(REPLACEMENT_MAPS), 4)
        self.assertFalse(REPLACED_MAPS & REPLACEMENT_MAPS)

    def test_replaced_map_registry_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["replaced_map_ids"][0] = "keep_failed_map"
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_map_replacement_qualification_design(
                changed, project_root=ROOT
            )

    def test_repair_outcome_boundary_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["forbidden_selection_inputs"].remove("candidate_runtime")
        with self.assertRaisesRegex(ValueError, "outcome boundary"):
            validate_map_replacement_qualification_design(
                changed, project_root=ROOT
            )

    def test_registered_source_runtime_is_frozen(self) -> None:
        runtime = json.loads(SOURCE_RUNTIME.read_text(encoding="utf-8"))
        self.assertFalse(runtime["formal"])
        self.assertEqual(runtime["solver_seeds"], [1, 2])
        self.assertEqual(
            runtime["policies"], ["official_adaptive", "realized_dynamic"]
        )
        self.assertEqual(runtime["environment"]["time_limit"], 600.0)
        self.assertEqual(runtime["environment"]["max_repair_iterations"], 12)
        self.assertEqual(runtime["max_decisions"], 12)
        self.assertEqual(runtime["metric_iteration_budget"], 12)
        self.assertEqual(runtime["episode_process_timeout_seconds"], 660.0)
        self.assertTrue(runtime["deterministic_pp_replay"])
        self.assertEqual(runtime["dataset_design"]["map_count"], 20)
        self.assertEqual(runtime["dataset_design"]["instance_count"], 40)
        self.assertEqual(
            runtime["dataset_design"]["layout_counts"],
            {
                "dao_high_topology": 24,
                "dao_mid_topology": 12,
                "dao_low_topology_control": 4,
            },
        )
        self.assertEqual(runtime["qualification"]["minimum_active_maps"], 20)
        self.assertEqual(runtime["qualification"]["minimum_nonzero_states"], 40)


if __name__ == "__main__":
    unittest.main()
