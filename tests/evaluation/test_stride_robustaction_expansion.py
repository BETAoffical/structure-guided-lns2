from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_expansion import (
    topology_group,
    validate_robustaction_expansion_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_robustaction_expansion_design.json"


class RobustActionExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_design_is_valid(self) -> None:
        validate_robustaction_expansion_design(self.config)

    def test_topology_boundaries_are_deterministic(self) -> None:
        thresholds = [0.035, 0.06]
        self.assertEqual(topology_group(0.060, thresholds), "dao_high_topology")
        self.assertEqual(topology_group(0.035, thresholds), "dao_mid_topology")
        self.assertEqual(
            topology_group(0.034999999, thresholds),
            "dao_low_topology_control",
        )

    def test_forbidden_outcome_input_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["selection_boundary"]["allowed_inputs"].append(
            "candidate_repair_outcome"
        )
        with self.assertRaisesRegex(ValueError, "outcome boundary"):
            validate_robustaction_expansion_design(changed)

    def test_locked_map_overlap_is_rejected(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["formal_ood_map_ids"][0] = changed["benchmarks"][0]["id"]
        with self.assertRaisesRegex(ValueError, "locked evidence"):
            validate_robustaction_expansion_design(changed)


if __name__ == "__main__":
    unittest.main()
