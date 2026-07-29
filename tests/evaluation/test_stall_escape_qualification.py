from __future__ import annotations

import unittest

from experiments.stall_escape_qualification import (
    classify_stall_sequences,
    validate_split_map_isolation,
)


def _decision(index: int, outcome: str, state: str = "state-a") -> dict:
    return {
        "decision_index": index,
        "before_repair_fingerprint": state,
        "no_progress": outcome in {"hard_failure", "accepted_noop"},
        "repair_outcome": outcome,
        "attempt_key": f"attempt-{index}",
        "actual_neighborhood_key": f"neighborhood-{index % 2}",
    }


class StallEscapeQualificationTests(unittest.TestCase):
    def test_changed_without_reduction_is_ordinary_progress(self) -> None:
        sequences, ordinary = classify_stall_sequences(
            [_decision(0, "state_changed_no_reduction")]
        )
        self.assertEqual(sequences, [])
        self.assertEqual(len(ordinary), 1)

    def test_short_streak_followed_by_change_is_natural_recovery(self) -> None:
        rows = [
            _decision(0, "hard_failure"),
            _decision(1, "accepted_noop"),
            _decision(2, "hard_failure"),
            _decision(3, "state_changed_no_reduction"),
        ]
        sequences, ordinary = classify_stall_sequences(rows)
        self.assertEqual(len(sequences), 1)
        self.assertEqual(sequences[0]["qualification_class"], "natural_recovery")
        self.assertEqual(sequences[0]["threshold"], 3)
        self.assertEqual(sequences[0]["recovery_delay_decisions"], 1)
        self.assertEqual(ordinary, [])

    def test_complete_future_window_without_change_confirms_stall(self) -> None:
        rows = [_decision(index, "hard_failure") for index in range(9)]
        sequences, _ordinary = classify_stall_sequences(rows)
        self.assertEqual(sequences[0]["threshold"], 6)
        self.assertEqual(
            sequences[0]["qualification_class"], "confirmed_long_stall"
        )
        self.assertEqual(sequences[0]["observed_after_trigger"], 3)

    def test_terminal_run_without_full_window_is_unresolved(self) -> None:
        rows = [_decision(index, "accepted_noop") for index in range(8)]
        sequences, _ordinary = classify_stall_sequences(rows)
        self.assertEqual(sequences[0]["threshold"], 6)
        self.assertEqual(sequences[0]["qualification_class"], "unresolved_stall")

    def test_same_attempt_repeated_does_not_qualify(self) -> None:
        rows = [_decision(index, "hard_failure") for index in range(9)]
        for row in rows:
            row["attempt_key"] = "same-attempt"
        sequences, _ordinary = classify_stall_sequences(rows)
        self.assertEqual(sequences, [])

    def test_train_and_diagnostic_maps_must_be_disjoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_map_isolation(
                {
                    "policy_train": {"map-a"},
                    "policy_validation": {"map-a"},
                }
            )


if __name__ == "__main__":
    unittest.main()
