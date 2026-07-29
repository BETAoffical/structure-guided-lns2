from __future__ import annotations

import unittest

from experiments.stall_rescue_order_audit import (
    recommend_rank_policy,
    simulate_rank_policy,
)


def _trial(rank: int, outcome: str, seconds: float, delta: float = 0.0) -> dict:
    return {
        "candidate_rank": rank,
        "repair_outcome": outcome,
        "repair_wall_seconds": seconds,
        "conflict_delta": delta,
    }


class StallRescueOrderAuditTests(unittest.TestCase):
    def test_stops_at_first_state_change_and_accumulates_repair_time(self) -> None:
        result = simulate_rank_policy(
            [
                _trial(2, "hard_failure", 1.0),
                _trial(3, "state_changed_no_reduction", 2.0),
                _trial(4, "conflict_reduced", 4.0, 8.0),
            ],
            (2, 3, 4),
        )
        self.assertTrue(result["escaped"])
        self.assertEqual(result["escape_rank"], 3)
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["repair_seconds"], 3.0)

    def test_reports_no_escape_after_all_no_progress_attempts(self) -> None:
        result = simulate_rank_policy(
            [
                _trial(2, "accepted_noop", 1.0),
                _trial(3, "hard_failure", 2.0),
            ],
            (2, 3),
        )
        self.assertFalse(result["escaped"])
        self.assertEqual(result["attempt_count"], 2)
        self.assertEqual(result["repair_seconds"], 3.0)

    def test_rejects_missing_candidate_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing candidate rank"):
            simulate_rank_policy([_trial(2, "hard_failure", 1.0)], (2, 3))

    def test_recommends_shortest_near_maximum_policy(self) -> None:
        summaries = [
            {
                "policy": "short",
                "rank_sequence": [2, 3],
                "stable_escape_state_fraction": 1.0,
                "escape_fraction": 0.985,
                "repair_seconds": {"mean": 1.0},
            },
            {
                "policy": "long",
                "rank_sequence": [2, 3, 4],
                "stable_escape_state_fraction": 1.0,
                "escape_fraction": 1.0,
                "repair_seconds": {"mean": 1.1},
            },
        ]
        self.assertEqual(recommend_rank_policy(summaries)["policy"], "short")


if __name__ == "__main__":
    unittest.main()
