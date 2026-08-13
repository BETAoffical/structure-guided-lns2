from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_causaltopopool import DESIGN_SCHEMA, EXPERIMENT_ID


class CausalTopoPoolExperimentTest(unittest.TestCase):
    def test_design_freezes_hybrid_compactness_only(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "configs" / "stride_causaltopopool_v1_design.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["schema"], DESIGN_SCHEMA)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(config["generator"]["fixed_preferred_sizes"], [])
        self.assertEqual(
            config["generator"]["oversized_closure_policy"],
            "reject_exact_closure_without_truncation",
        )
        self.assertEqual(config["execution"]["workers"], 16)
        self.assertFalse(config["claim_boundary"]["native_pp_collection_allowed"])
        self.assertFalse(config["claim_boundary"]["model_training_allowed"])
        self.assertFalse(config["claim_boundary"]["ttf_experiment_allowed"])


if __name__ == "__main__":
    unittest.main()
