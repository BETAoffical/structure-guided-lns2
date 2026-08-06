from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_da2_source_stability import (
    REPLACEMENTS,
    validate_da2_source_stability_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_da2_source_stability_design.json"
)


class RobustActionDA2SourceStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_stability_revision_is_valid(self) -> None:
        validate_da2_source_stability_design(self.config, project_root=ROOT)
        self.assertEqual(self.config["source_contract"]["workers"], 1)
        self.assertEqual(
            self.config["source_contract"]["expected_nonzero_qualification_rows"],
            31,
        )
        self.assertEqual(
            self.config["source_contract"]["expected_nonzero_by_solver_seed"],
            {"1": 16, "2": 15},
        )

    def test_replacements_keep_map_and_task_variant(self) -> None:
        configured = {
            row["removed_task_id"]: row["replacement_task_id"]
            for row in self.config["replacements"]
        }
        self.assertEqual(configured, REPLACEMENTS)
        for row in self.config["replacements"]:
            self.assertLess(row["replacement_agent_count"], row["removed_agent_count"])
            self.assertIn(row["task_variant"], row["replacement_task_id"])

    def test_source_outcomes_and_ttf_remain_forbidden(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["outcome_boundary"]["controller_outcomes_read"] = True
        with self.assertRaisesRegex(ValueError, "outcome boundary changed"):
            validate_da2_source_stability_design(changed, project_root=ROOT)

        changed = copy.deepcopy(self.config)
        changed["source_contract"]["failure_action"] = "keep_successes"
        with self.assertRaisesRegex(ValueError, "source contract changed"):
            validate_da2_source_stability_design(changed, project_root=ROOT)

    def test_capacity_keeps_complete_source_v4_unfiltered(self) -> None:
        frozen = self.config["source_v4_frozen_capacity"]
        self.assertTrue(frozen["must_remain_unfiltered"])
        self.assertEqual(frozen["maximum_obtainable_unique_state_ids"], 297)
        self.assertEqual(
            self.config["capacity_contract"]["minimum_combined_unique_state_ids"],
            320,
        )


if __name__ == "__main__":
    unittest.main()
