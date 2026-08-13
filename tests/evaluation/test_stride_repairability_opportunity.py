from __future__ import annotations

import unittest

from experiments.stride_repairability_opportunity import (
    paired_repairability_diagnostic,
)


def _aggregate(
    candidate_id: str,
    *,
    seed_mean: float = 0.2,
    no_progress: float = 0.25,
    success_rate: float = 0.5,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_kind": "structural",
        "actual_size": 16,
        "seed_mean": seed_mean,
        "lower_half_mean": seed_mean,
        "first_fixed_half_mean": seed_mean,
        "second_fixed_half_mean": seed_mean,
        "no_progress_rate": no_progress,
        "replan_success_rate": success_rate,
        "selection_families": ["family"],
        "structpool_family_groups": ["group"],
    }


def _trials(successes: set[int]) -> list[dict]:
    return [
        {
            "trial_index": index,
            "pp_seed": 1000 + index,
            "replan_success": index in successes,
        }
        for index in range(16)
    ]


class RepairabilityOpportunityTests(unittest.TestCase):
    def test_quality_preserving_two_half_repairability_gain_qualifies(self) -> None:
        anchor = _aggregate("anchor", success_rate=0.25)
        candidate = _aggregate("candidate", seed_mean=0.21, success_rate=0.5)
        diagnostic = paired_repairability_diagnostic(
            candidate,
            anchor,
            _trials({0, 1, 2, 3, 8, 9, 10, 11}),
            _trials({0, 1, 8, 9}),
        )
        self.assertTrue(diagnostic["quality_noninferior"])
        self.assertTrue(diagnostic["repairability_dominates"])
        self.assertTrue(diagnostic["qualifies"])
        self.assertEqual(diagnostic["paired_win_count"], 4)
        self.assertEqual(diagnostic["paired_loss_count"], 0)

    def test_safe_but_lower_quality_candidate_does_not_qualify(self) -> None:
        anchor = _aggregate("anchor", success_rate=0.25)
        candidate = _aggregate("candidate", seed_mean=0.1, success_rate=0.75)
        diagnostic = paired_repairability_diagnostic(
            candidate,
            anchor,
            _trials(set(range(12))),
            _trials({0, 1, 8, 9}),
        )
        self.assertFalse(diagnostic["quality_noninferior"])
        self.assertTrue(diagnostic["repairability_dominates"])
        self.assertFalse(diagnostic["qualifies"])

    def test_one_half_only_gain_does_not_qualify(self) -> None:
        anchor = _aggregate("anchor", success_rate=0.5)
        candidate = _aggregate("candidate", success_rate=0.625)
        diagnostic = paired_repairability_diagnostic(
            candidate,
            anchor,
            _trials({0, 1, 2, 3, 4, 5, 6, 7, 14, 15}),
            _trials({0, 1, 2, 3, 8, 9, 10, 11}),
        )
        self.assertGreater(diagnostic["replan_success_rate_delta"], 0.0)
        self.assertFalse(
            diagnostic["repairability_checks"]["both_fixed_halves_nonnegative"]
        )
        self.assertFalse(diagnostic["qualifies"])

    def test_pairing_seed_mismatch_is_rejected(self) -> None:
        anchor_trials = _trials({0})
        candidate_trials = _trials({0, 1})
        candidate_trials[3]["pp_seed"] = -1
        with self.assertRaisesRegex(ValueError, "paired PP seed changed"):
            paired_repairability_diagnostic(
                _aggregate("candidate", success_rate=0.125),
                _aggregate("anchor", success_rate=0.0625),
                candidate_trials,
                anchor_trials,
            )


if __name__ == "__main__":
    unittest.main()
