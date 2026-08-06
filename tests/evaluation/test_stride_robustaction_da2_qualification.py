from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_da2_qualification import (
    SUPPLEMENT_MAPS,
    validate_da2_supplement_qualification_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_da2_supplement_qualification_design.json"
)


class RobustActionDA2QualificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_qualification_is_valid(self) -> None:
        validate_da2_supplement_qualification_design(
            self.config, project_root=ROOT
        )
        self.assertEqual(len(SUPPLEMENT_MAPS), 8)
        self.assertEqual(self.config["expected_qualification_job_count"], 96)
        self.assertEqual(self.config["expected_selected_task_count"], 16)

    def test_repair_outcome_boundary_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["forbidden_selection_inputs"].remove("candidate_runtime")
        with self.assertRaisesRegex(ValueError, "outcome boundary"):
            validate_da2_supplement_qualification_design(
                changed, project_root=ROOT
            )

    def test_failure_action_cannot_filter_observed_tasks(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["next_decision_on_failure"] = "keep_only_successful_tasks"
        with self.assertRaisesRegex(ValueError, "decision rule"):
            validate_da2_supplement_qualification_design(
                changed, project_root=ROOT
            )


if __name__ == "__main__":
    unittest.main()
