from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_load_extension import (
    UNDERLOADED_MAPS,
    validate_load_extension_qualification_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_load_extension_qualification_design.json"
)


class RobustActionLoadExtensionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_extension_qualification_is_valid(self) -> None:
        validate_load_extension_qualification_design(
            self.config, project_root=ROOT
        )
        self.assertEqual(len(UNDERLOADED_MAPS), 12)

    def test_repair_outcome_boundary_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["forbidden_selection_inputs"].remove("candidate_runtime")
        with self.assertRaisesRegex(ValueError, "outcome boundary"):
            validate_load_extension_qualification_design(
                changed, project_root=ROOT
            )

    def test_failure_action_cannot_filter_observed_tasks(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["next_decision_on_failure"] = "keep_only_successful_tasks"
        with self.assertRaisesRegex(ValueError, "failure action"):
            validate_load_extension_qualification_design(
                changed, project_root=ROOT
            )


if __name__ == "__main__":
    unittest.main()
