from __future__ import annotations

import unittest

from experiments.stride_tailswitch import (
    POLICIES,
    _contrast_gate,
    classify_adverse_pair,
    tailswitch_schedule,
)


class TailSwitchTest(unittest.TestCase):
    def test_schedule_is_strict_four_policy_rotation(self) -> None:
        schedule = tailswitch_schedule(
            [
                {
                    "state_id": "s0",
                    "task_id": "t0",
                    "solver_seed": 3,
                    "challenger": "v2-plus-structpool",
                },
                {
                    "state_id": "s1",
                    "task_id": "t1",
                    "solver_seed": 5,
                    "challenger": "v2-plus-slotpool",
                },
            ]
        )
        self.assertEqual([row["policy"] for row in schedule[:4]], list(POLICIES))
        self.assertEqual([row["state_position"] for row in schedule], [0] * 4 + [1] * 4)

    def test_adverse_pair_uses_completion_then_fixed_horizon(self) -> None:
        rule = {
            "normalized_auc_delta_at_least": 0.05,
            "or_final_conflict_delta_at_least": 5,
        }
        base = {"success": True, "normalized_fixed_auc": 0.2, "final_conflicts": 0}
        censored = {
            "success": False,
            "normalized_fixed_auc": 0.2,
            "final_conflicts": 0,
        }
        self.assertEqual(
            classify_adverse_pair(base, censored, rule)["classification"],
            "adverse",
        )
        degraded = {
            "success": True,
            "normalized_fixed_auc": 0.251,
            "final_conflicts": 0,
        }
        self.assertEqual(
            classify_adverse_pair(base, degraded, rule)["classification"],
            "adverse",
        )

    def test_contrast_gate_requires_both_challengers(self) -> None:
        gate = {
            "minimum_adverse_fraction": 2 / 3,
            "minimum_adverse_comparison_count": 2,
            "minimum_map_count": 2,
            "minimum_task_count": 2,
            "minimum_solver_seed_count": 2,
            "minimum_per_challenger_count": 1,
        }
        rows = [
            {
                "classification": "adverse",
                "challenger": "v2-plus-structpool",
                "map_id": "m0",
                "task_id": "t0",
                "solver_seed": 1,
                "normalized_auc_delta": 0.1,
                "final_conflict_delta": 5,
            },
            {
                "classification": "adverse",
                "challenger": "v2-plus-slotpool",
                "map_id": "m1",
                "task_id": "t1",
                "solver_seed": 2,
                "normalized_auc_delta": 0.1,
                "final_conflict_delta": 5,
            },
            {
                "classification": "neutral",
                "challenger": "v2-plus-slotpool",
                "map_id": "m1",
                "task_id": "t2",
                "solver_seed": 2,
                "normalized_auc_delta": 0.0,
                "final_conflict_delta": 0,
            },
        ]
        self.assertTrue(_contrast_gate(rows, gate)["passed"])


if __name__ == "__main__":
    unittest.main()
