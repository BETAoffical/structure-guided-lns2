from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_repairdependencypool import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    _causal_classification,
    _escape_classification,
)


class RepairDependencyPoolExperimentTest(unittest.TestCase):
    def test_registration_freezes_zero_solver_gate(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "configs" / "stride_repairdependencypool_v1_registration.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["schema"], CONFIG_SCHEMA)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(config["generator"]["maximum_added_agents_per_core"], 8)
        self.assertEqual(config["generator"]["maximum_neighborhood_size"], 32)
        self.assertEqual(config["generator"]["maximum_candidates_per_state"], 6)
        self.assertEqual(config["generator"]["fixed_preferred_sizes"], [])
        self.assertEqual(config["execution"]["workers"], 16)
        self.assertEqual(config["execution"]["new_solver_runs"], 0)
        self.assertFalse(config["claim_boundary"]["model_training_allowed"])
        self.assertFalse(config["claim_boundary"]["ttf_experiment_allowed"])

    def test_escape_and_causal_classes_remain_separate(self) -> None:
        witness = {
            "case_id": "case",
            "exit": {
                "outcome": "strict_reduction",
                "set_change": True,
                "same_agent_set": False,
            },
        }
        self.assertEqual(_escape_classification(witness), "set_change_escape")
        self.assertEqual(_causal_classification("order_defect"), "order")
        self.assertEqual(_causal_classification("set_and_order_joint"), "joint")
        witness["exit"] = {
            "outcome": "right_censored",
            "set_change": False,
            "same_agent_set": True,
        }
        self.assertEqual(_escape_classification(witness), "right_censored")


if __name__ == "__main__":
    unittest.main()
