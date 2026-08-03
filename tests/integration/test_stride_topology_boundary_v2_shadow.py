from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_topology_boundary_v2_shadow import (
    _shadow_gate_results,
    _shadow_summary,
    validate_topology_boundary_v2_shadow_config,
)


class StrideTopologyBoundaryV2ShadowTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (root / "configs" / "stride_topology_boundary_v2_shadow.json").read_text(
                encoding="utf-8"
            )
        )

    def test_protocol_is_exploratory_and_outcome_blind(self) -> None:
        config = self._config()
        validate_topology_boundary_v2_shadow_config(config)
        self.assertEqual(config["frozen_controller_id"], "v2-full")
        self.assertFalse(config["formal_speed_claim"])
        self.assertFalse(config["training_allowed"])
        self.assertFalse(config["ttf_claim_allowed"])
        self.assertTrue(config["selection_protocol"]["outcome_blind_before_join"])
        self.assertEqual(config["expected_candidate_count"], 347)
        self.assertEqual(config["expected_outcome_count"], 2776)
        self.assertEqual(config["expected_generated_feature_dimension"], 124)
        self.assertEqual(config["expected_model_input_dimension"], 86)

    def test_protocol_rejects_gate_or_promotion_drift(self) -> None:
        config = self._config()
        config["ttf_claim_allowed"] = True
        with self.assertRaisesRegex(ValueError, "must remain exploratory"):
            validate_topology_boundary_v2_shadow_config(config)
        config = self._config()
        config["exploratory_quick_gates"]["minimum_mean_normalized_selected_gain"] = 0.0
        with self.assertRaisesRegex(ValueError, "exploratory gates changed"):
            validate_topology_boundary_v2_shadow_config(config)

    def test_summary_and_quick_gate_use_same_pool_quality(self) -> None:
        base = {
            "action_changed": True,
            "augmented_selected_boundary_only": True,
            "selected_quality_improved": True,
            "selected_quality_worsened": False,
            "normalized_selected_gain": 0.04,
            "first_half_normalized_selected_gain": 0.03,
            "unseen_half_normalized_selected_gain": 0.05,
            "baseline_normalized_regret": 0.20,
            "augmented_normalized_regret": 0.16,
            "baseline_exact_best": False,
            "augmented_exact_best": False,
            "baseline_in_quality_top3": False,
            "augmented_in_quality_top3": True,
            "baseline_v2_margin": 0.10,
            "augmented_v2_margin": 0.08,
            "augmented_selected_feature_outside_fraction": 0.0,
        }
        unchanged = {
            **base,
            "action_changed": False,
            "augmented_selected_boundary_only": False,
            "selected_quality_improved": False,
            "normalized_selected_gain": 0.0,
            "first_half_normalized_selected_gain": 0.0,
            "unseen_half_normalized_selected_gain": 0.0,
            "baseline_normalized_regret": 0.10,
            "augmented_normalized_regret": 0.10,
            "baseline_in_quality_top3": True,
        }
        summary = _shadow_summary([base, unchanged])
        self.assertEqual(summary["boundary_selected_state_count"], 1)
        self.assertEqual(summary["improved_state_count"], 1)
        self.assertAlmostEqual(summary["mean_normalized_selected_gain"], 0.02)
        self.assertTrue(
            all(
                _shadow_gate_results(
                    summary, self._config()["exploratory_quick_gates"]
                ).values()
            )
        )


if __name__ == "__main__":
    unittest.main()
