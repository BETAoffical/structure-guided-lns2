from __future__ import annotations

import unittest

from experiments.v3_bundle_equivalence import (
    _maximum_prediction_delta,
    _state_rows,
)


class V3BundleEquivalenceTests(unittest.TestCase):
    def test_prediction_delta_covers_every_output(self) -> None:
        left = {
            "effective_progress_probability": [0.1],
            "no_progress_probability": [0.2],
            "conflict_reduction": [3.0],
            "repair_seconds": [4.0],
            "utility": [5.0],
        }
        right = {name: list(values) for name, values in left.items()}
        right["utility"][0] += 0.25
        delta = _maximum_prediction_delta(left, right)
        self.assertEqual(delta["utility"], 0.25)
        self.assertEqual(delta["conflict_reduction"], 0.0)

    def test_state_rows_exclude_adaptive_and_sort_candidates(self) -> None:
        rows = [
            {"state_id": "s", "candidate_id": "b", "route": "model"},
            {
                "state_id": "s",
                "candidate_id": "official_adaptive",
                "route": "official_adaptive",
            },
            {"state_id": "s", "candidate_id": "a", "route": "model"},
        ]
        grouped = _state_rows(rows)
        self.assertEqual(
            [row["candidate_id"] for row in grouped["s"]], ["a", "b"]
        )

    def test_prediction_length_mismatch_is_rejected(self) -> None:
        left = {
            "effective_progress_probability": [0.1],
            "no_progress_probability": [0.2],
            "conflict_reduction": [3.0],
            "repair_seconds": [4.0],
            "utility": [5.0],
        }
        right = {name: list(values) for name, values in left.items()}
        right["repair_seconds"].append(6.0)
        with self.assertRaisesRegex(ValueError, "different lengths"):
            _maximum_prediction_delta(left, right)


if __name__ == "__main__":
    unittest.main()
