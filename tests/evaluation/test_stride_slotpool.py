from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_slotpool import (
    CANDIDATE_FEATURE_NAMES,
    DERIVED_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    stable_pair_table,
    summarize_slotpool_acceptance,
    validate_slotpool_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_slotpool_v1_registration.json"


class SlotPoolTests(unittest.TestCase):
    def test_registered_config_and_dimensions_are_valid(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_slotpool_config(config)
        self.assertEqual(len(DERIVED_FEATURE_NAMES), 46)
        self.assertEqual(len(CANDIDATE_FEATURE_NAMES), 170)
        self.assertEqual(len(PAIR_FEATURE_NAMES), 193)

    def test_registration_rejects_quality_or_claim_drift(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["offline_acceptance"]["global_best_retention_minimum"] = 0.8
        with self.assertRaisesRegex(ValueError, "acceptance gates"):
            validate_slotpool_config(changed)
        changed = copy.deepcopy(config)
        changed["claim_boundary"]["runtime_integration_before_acceptance"] = True
        with self.assertRaisesRegex(ValueError, "claim boundary"):
            validate_slotpool_config(changed)

    def test_stable_pairs_require_both_half_orders_and_uniform_state_weight(self) -> None:
        rows = [
            {
                "state_id": "state-a",
                "map_id": "map-a",
                "first_fixed_half_mean": 0.4,
                "second_fixed_half_mean": 0.3,
            },
            {
                "state_id": "state-a",
                "map_id": "map-a",
                "first_fixed_half_mean": 0.2,
                "second_fixed_half_mean": 0.1,
            },
            {
                "state_id": "state-a",
                "map_id": "map-a",
                "first_fixed_half_mean": 0.1,
                "second_fixed_half_mean": 0.4,
            },
            {
                "state_id": "state-b",
                "map_id": "map-b",
                "first_fixed_half_mean": 0.5,
                "second_fixed_half_mean": 0.5,
            },
            {
                "state_id": "state-b",
                "map_id": "map-b",
                "first_fixed_half_mean": 0.0,
                "second_fixed_half_mean": 0.0,
            },
        ]
        pairs = stable_pair_table(rows)
        self.assertGreater(len(pairs), 1)
        totals = {}
        for row in pairs:
            totals.setdefault(row["state_id"], 0.0)
            totals[row["state_id"]] += row["weight"]
        self.assertEqual(set(totals), {"state-a", "state-b"})
        for value in totals.values():
            self.assertAlmostEqual(value, 1.0)
        self.assertFalse(
            any(row["left"] == 0 and row["right"] == 2 for row in pairs)
        )

    def test_acceptance_requires_all_quality_and_budget_gates(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        rows = [
            {
                "map_id": f"map-{index % 2}",
                "layout_mode": f"layout-{index % 2}",
                "global_best_retained": index != 9,
                "first_fixed_half_best_retained": index != 9,
                "second_fixed_half_best_retained": index != 8,
                "normalized_regret": 0.0,
                "candidate_count": 20,
                "selected_candidate_count": 6,
            }
            for index in range(10)
        ]
        result = summarize_slotpool_acceptance(
            rows=rows, pairwise_accuracy=0.7, config=config
        )
        self.assertTrue(result["passed"])
        failed = copy.deepcopy(rows)
        failed[0]["selected_candidate_count"] = 7
        result = summarize_slotpool_acceptance(
            rows=failed, pairwise_accuracy=0.7, config=config
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["candidate_budget"])


if __name__ == "__main__":
    unittest.main()
