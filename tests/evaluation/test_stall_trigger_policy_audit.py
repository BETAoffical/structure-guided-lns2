from __future__ import annotations

import unittest

from experiments.stall_trigger_policy_audit import (
    same_neighborhood_trigger_index,
    validate_source_threshold_coverage,
    wilson_upper,
)


class StallTriggerPolicyAuditTests(unittest.TestCase):
    def test_same_neighborhood_trigger_uses_distinct_attempts(self) -> None:
        history = [
            {"neighborhood_key": "a", "attempt_key": "a-1"},
            {"neighborhood_key": "b", "attempt_key": "b-1"},
            {"neighborhood_key": "a", "attempt_key": "a-1"},
            {"neighborhood_key": "a", "attempt_key": "a-2"},
        ]
        self.assertEqual(same_neighborhood_trigger_index(history, 2), 4)
        self.assertIsNone(same_neighborhood_trigger_index(history, 3))

    def test_wilson_upper_is_conservative_for_zero_failures(self) -> None:
        self.assertGreater(wilson_upper(0, 145), 0.01)
        self.assertLess(wilson_upper(0, 1000), 0.01)

    def test_rejects_invalid_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "integer >= 2"):
            same_neighborhood_trigger_index([], 1)

    def test_rejects_thresholds_outside_sequence_label_coverage(self) -> None:
        validate_source_threshold_coverage((3, 4, 5, 6), (3, 4, 6))
        with self.assertRaisesRegex(ValueError, "labelled sequence coverage"):
            validate_source_threshold_coverage((3, 9), (3, 4, 6))
        with self.assertRaisesRegex(ValueError, "labelled sequence coverage"):
            validate_source_threshold_coverage((2, 3), (3, 4, 6))


if __name__ == "__main__":
    unittest.main()
