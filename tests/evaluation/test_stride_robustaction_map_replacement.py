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


if __name__ == "__main__":
    unittest.main()
