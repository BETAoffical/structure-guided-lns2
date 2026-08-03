from __future__ import annotations

import unittest

from experiments.stride_robuststep import (
    evaluate_robuststep_variant,
    robust_pair_winner,
)


class StrideRobustStepTest(unittest.TestCase):
    def test_pair_requires_seed_agreement(self) -> None:
        winner = robust_pair_winner(
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [True, True, True, False],
            [True, True, True, False],
            minimum_paired_win_fraction=0.75,
            maximum_no_progress_disadvantage=0.0,
        )
        self.assertEqual(winner, 1)
        uncertain = robust_pair_winner(
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [True] * 4,
            [True] * 4,
            minimum_paired_win_fraction=0.75,
            maximum_no_progress_disadvantage=0.0,
        )
        self.assertEqual(uncertain, 0)

    def test_pair_rejects_worse_no_progress_risk(self) -> None:
        winner = robust_pair_winner(
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [True, True, False, False],
            [True, True, True, True],
            minimum_paired_win_fraction=0.75,
            maximum_no_progress_disadvantage=0.125,
        )
        self.assertEqual(winner, 0)

    def test_variant_reports_stable_good_set_and_uncertainty(self) -> None:
        profiles = {
            "state-a": {
                "a": {
                    "scores": [3.0] * 8,
                    "progress": [True] * 8,
                },
                "b": {
                    "scores": [2.0] * 8,
                    "progress": [True] * 8,
                },
                "c": {
                    "scores": [1.0] * 8,
                    "progress": [True] * 8,
                },
            }
        }
        report = evaluate_robuststep_variant(
            profiles,
            {
                "id": "test",
                "minimum_paired_win_fraction": 0.75,
                "maximum_no_progress_disadvantage": 0.0,
            },
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        )
        self.assertEqual(report["half_pairwise_consistency"], 1.0)
        self.assertEqual(report["mean_good_set_jaccard"], 1.0)
        self.assertEqual(report["unique_robust_winner_rate"], 1.0)
        self.assertEqual(report["full_pair_coverage"], 1.0)


if __name__ == "__main__":
    unittest.main()
