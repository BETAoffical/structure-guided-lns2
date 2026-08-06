from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_da2_recovery import (
    RETAINED_TASK_IDS,
    recovery_source_adapter,
    validate_da2_recovery_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_da2_recovery_design.json"
)


class RobustActionDA2RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_recovery_is_valid_and_outcome_informed(self) -> None:
        validate_da2_recovery_design(self.config, project_root=ROOT)
        self.assertTrue(self.config["outcome_boundary"]["outcome_informed"])
        self.assertFalse(
            self.config["outcome_boundary"]["candidate_repair_outcomes_read"]
        )
        self.assertEqual(
            tuple(self.config["retained"]["task_ids"]), RETAINED_TASK_IDS
        )

    def test_primary_and_fallback_adapters_are_dimensioned(self) -> None:
        validate_da2_recovery_design(self.config, project_root=ROOT)
        primary = recovery_source_adapter(self.config, mode="primary_repair")
        fallback = recovery_source_adapter(
            self.config, mode="fallback_replacement"
        )
        self.assertEqual(primary["expected_instance_count"], 8)
        self.assertEqual(primary["benchmarks"][0]["agent_counts"], [412, 686, 960, 1236])
        self.assertEqual(fallback["expected_instance_count"], 6)
        self.assertEqual(fallback["benchmarks"][0]["id"], "w_encounter1")

    def test_candidate_repair_outcomes_cannot_be_enabled(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["outcome_boundary"]["candidate_repair_outcomes_read"] = True
        with self.assertRaisesRegex(ValueError, "outcome boundary"):
            validate_da2_recovery_design(changed, project_root=ROOT)

    def test_retained_task_cohort_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["retained"]["task_ids"][0] = "posthoc_replacement"
        with self.assertRaisesRegex(ValueError, "retained cohort"):
            validate_da2_recovery_design(changed, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
