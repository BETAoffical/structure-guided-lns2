from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_topology_boundary_quality_confirmation import (
    QUALITY_GATES,
    _quality_gate_results,
    validate_topology_boundary_quality_confirmation_config,
)


class StrideTopologyBoundaryQualityConfirmationTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (
                root
                / "configs"
                / "stride_topology_boundary_quality_confirmation.json"
            ).read_text(encoding="utf-8")
        )

    def test_confirmation_collects_only_unseen_half_and_combines_eight(self) -> None:
        config = self._config()
        validate_topology_boundary_quality_confirmation_config(config)
        self.assertEqual(config["collection_trial_indices"], [4, 5, 6, 7])
        self.assertEqual(config["trial_indices"], list(range(8)))
        self.assertEqual(config["expected_collection_outcome_count"], 1388)
        self.assertEqual(config["expected_outcome_count"], 2776)
        self.assertTrue(
            config["freshness"]["new_seed_half_unseen_when_confirmation_registered"]
        )
        self.assertFalse(config["runtime_used_in_label"])
        self.assertFalse(config["future_repair_rounds_used"])
        self.assertFalse(config["cost_to_go_used"])
        self.assertFalse(config["training_allowed"])

    def test_confirmation_rejects_seed_or_gate_drift(self) -> None:
        config = self._config()
        config["collection_trial_indices"] = [0, 1, 2, 3]
        with self.assertRaisesRegex(ValueError, "cohort or identity changed"):
            validate_topology_boundary_quality_confirmation_config(config)
        config = self._config()
        config["independent_second_half_gates"][
            "minimum_augmented_pool_strict_win_rate"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "independent gates changed"):
            validate_topology_boundary_quality_confirmation_config(config)

    def test_unseen_half_must_pass_every_original_quality_gate(self) -> None:
        half = {
            "summary": {
                "augmented_pool_strict_win_rate": 0.50,
                "mean_normalized_augmented_pool_gain": 0.10,
                "boundary_top3_state_rate": 0.60,
                "mean_boundary_best_normalized_regret": 0.10,
            },
            "topology_groups": {
                "dao_ultra_bottleneck": {"strict_win_count": 1},
                "dao_articulated": {"strict_win_count": 1},
                "dao_low_articulation_control": {"strict_win_count": 0},
            },
        }
        self.assertTrue(all(_quality_gate_results(half, QUALITY_GATES).values()))
        half["summary"]["mean_boundary_best_normalized_regret"] = 0.16
        results = _quality_gate_results(half, QUALITY_GATES)
        self.assertFalse(results["maximum_mean_boundary_best_normalized_regret"])


if __name__ == "__main__":
    unittest.main()
