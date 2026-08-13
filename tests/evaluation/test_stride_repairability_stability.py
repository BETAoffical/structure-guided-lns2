from __future__ import annotations

import unittest

from experiments.stride_repairability_stability import (
    _tie_aware_rank_correlation,
    checkpoint_stability,
)


def _trials(successes: set[int], score: float) -> list[dict]:
    return [
        {
            "trial_index": index,
            "pp_seed": 100 + index,
            "replan_success": index in successes,
            "normalized_conflict_reduction": score,
            "no_progress": score <= 0.0,
        }
        for index in range(16)
    ]


def _aggregate(candidate_id: str, score: float, size: int = 8) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_kind": "base",
        "actual_size": size,
        "seed_mean": score,
        "lower_half_mean": score,
        "first_fixed_half_mean": score,
        "second_fixed_half_mean": score,
        "no_progress_rate": 0.0,
    }


def _fixture(second_candidate_successes: set[int]) -> dict:
    state = "state"
    aggregates = {
        (state, "anchor"): _aggregate("anchor", 0.2),
        (state, "candidate"): _aggregate("candidate", 0.3),
    }
    trials = {
        (state, "anchor"): _trials({0, 1, 8, 9}, 0.2),
        (state, "candidate"): _trials(second_candidate_successes, 0.3),
    }
    return {
        "logical": {
            "logical_checkpoint_id": "logical",
            "case_id": "case",
            "state_fingerprint": state,
            "map_id": "map",
            "checkpoint_kind": "first_repeat_stall",
            "classification": "adverse",
            "challenger": "challenger",
            "candidate_ids": ["anchor", "candidate"],
        },
        "checkpoint": {
            "candidate_pool_diagnostic": {
                "original_pool_anchor_candidate_id": "anchor"
            }
        },
        "aggregates": aggregates,
        "trials": trials,
        "halves": (tuple(range(8)), tuple(range(8, 16))),
        "epsilon": 1e-12,
    }


class RepairabilityStabilityTests(unittest.TestCase):
    def test_two_constant_halves_are_decision_equivalent(self) -> None:
        self.assertEqual(
            _tie_aware_rank_correlation([1.0, 1.0], [0.5, 0.5]),
            (1.0, "both_halves_constant_decision_equivalent"),
        )

    def test_one_constant_half_is_not_rank_stable(self) -> None:
        self.assertEqual(
            _tie_aware_rank_correlation([1.0, 1.0], [0.5, 1.0]),
            (0.0, "one_half_constant"),
        )

    def test_consistent_candidate_has_full_overlap_and_zero_regret(self) -> None:
        row = checkpoint_stability(
            **_fixture({0, 1, 2, 3, 8, 9, 10, 11})
        )
        self.assertEqual(row["top3_overlap"], 1.0)
        self.assertEqual(row["mean_cross_half_regret"], 0.0)
        self.assertEqual(row["anchor_direction_agreement"], 1.0)

    def test_reversed_half_direction_is_detected(self) -> None:
        row = checkpoint_stability(**_fixture({0, 1, 2, 3, 4, 5, 6, 7}))
        self.assertEqual(row["anchor_direction_agreement"], 0.0)
        self.assertGreater(row["mean_cross_half_regret"], 0.0)


if __name__ == "__main__":
    unittest.main()
