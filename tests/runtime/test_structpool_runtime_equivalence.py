from __future__ import annotations

import copy
import unittest

from scripts.verify_structpool_runtime_equivalence import (
    _transition_signature,
    _without_runtime,
)


class StructPoolRuntimeEquivalenceTest(unittest.TestCase):
    def test_runtime_filter_keeps_candidate_and_gate_semantics(self) -> None:
        controller = {
            "controller_runtime": "optimized",
            "controller_seconds_before_repair": 0.5,
            "feature_seconds": 0.2,
            "candidate_pool": [{"candidate_id": "a", "agents": [1, 2], "score": 3.0}],
            "proposal": {
                "structpool_candidate_seconds": 0.1,
                "structpool_gate_passed": True,
                "structpool_added_candidate_count": 6,
            },
            "selected_candidate_id": "a",
        }
        self.assertEqual(
            _without_runtime(controller),
            {
                "candidate_pool": [
                    {"candidate_id": "a", "agents": [1, 2], "score": 3.0}
                ],
                "proposal": {
                    "structpool_gate_passed": True,
                    "structpool_added_candidate_count": 6,
                },
                "selected_candidate_id": "a",
            },
        )

    def test_transition_signature_ignores_only_runtime_changes(self) -> None:
        row = {
            "decision_index": 0,
            "before_fingerprint": "before",
            "after_fingerprint": "after",
            "action": {"mode": "explicit_neighborhood", "agents": [1, 2]},
            "low_level_delta": {"generated": 5},
            "terminated": False,
            "truncated": False,
            "metrics": {
                "action_valid": True,
                "conflicts_before": 3,
                "conflicts_after": 1,
                "neighborhood": [1, 2],
                "repair_order": [2, 1],
                "requested_pp_random_seed": 7,
                "applied_pp_random_seed": 7,
            },
            "controller": {
                "selected_candidate_id": "a",
                "controller_seconds_before_repair": 0.5,
                "proposal": {
                    "structpool_candidate_seconds": 0.1,
                    "structpool_gate_passed": True,
                },
            },
        }
        changed_timing = copy.deepcopy(row)
        changed_timing["controller"]["controller_seconds_before_repair"] = 1.5
        changed_timing["controller"]["proposal"]["structpool_candidate_seconds"] = 0.4
        self.assertEqual(
            _transition_signature(row), _transition_signature(changed_timing)
        )

        changed_action = copy.deepcopy(row)
        changed_action["action"]["agents"] = [1, 3]
        self.assertNotEqual(
            _transition_signature(row), _transition_signature(changed_action)
        )


if __name__ == "__main__":
    unittest.main()
