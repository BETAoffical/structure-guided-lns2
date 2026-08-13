from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_causalclosurepool_structcoverage import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    closest_action,
    jaccard,
)


class CausalClosurePoolStructuralCoverageTest(unittest.TestCase):
    def test_registration_freezes_posthoc_claim_boundary(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (
                root
                / "configs"
                / "stride_causalclosurepool_structcoverage_v1.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["schema"], CONFIG_SCHEMA)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        self.assertTrue(
            config["claim_boundary"][
                "development_posthoc_coverage_diagnostic_only"
            ]
        )
        self.assertFalse(config["claim_boundary"]["model_training_allowed"])
        self.assertFalse(config["claim_boundary"]["ttf_experiment_allowed"])
        self.assertFalse(
            config["claim_boundary"]["long_tail_avoidance_claim_allowed"]
        )

    def test_jaccard_and_closest_action_use_realized_agent_sets(self) -> None:
        self.assertEqual(jaccard([1, 2], [1, 2]), 1.0)
        self.assertEqual(jaccard([1, 2], [2, 3]), 1.0 / 3.0)
        result = closest_action(
            {"agents": [1, 2, 3], "actual_size": 3},
            [
                {"candidate_id": "far", "agents": [8, 9], "actual_size": 2},
                {
                    "candidate_id": "near",
                    "agents": [1, 2, 3, 4],
                    "actual_size": 4,
                },
            ],
        )
        self.assertEqual(result["candidate_id"], "near")
        self.assertEqual(result["jaccard"], 0.75)
        self.assertFalse(result["exact_agent_set"])


if __name__ == "__main__":
    unittest.main()
