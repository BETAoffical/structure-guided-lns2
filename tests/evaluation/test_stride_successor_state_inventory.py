from __future__ import annotations

import unittest

from experiments.closed_loop_trace_storage import (
    encode_extras_delta,
    encode_state_delta,
)
from experiments.repair_collection import state_fingerprint
from experiments.stride_successor_state_inventory import successor_dynamics


def _state(marker: int, conflicts: list[list[int]]) -> dict:
    agents = [
        {
            "id": 0,
            "start": 0,
            "goal": 1,
            "path": [0, 1 + marker],
        },
        {
            "id": 1,
            "start": 1,
            "goal": 0,
            "path": [1, marker],
        },
    ]
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "rows": 1,
        "cols": 2,
        "obstacles": [],
        "agents": agents,
        "conflict_edges": conflicts,
        "num_of_colliding_pairs": len(conflicts),
        "sum_of_costs": 2 + marker,
        "low_level": {
            "runs": marker,
            "expanded": marker,
            "generated": marker,
            "reopened": 0,
        },
        "feasible": not conflicts,
        "done": not conflicts,
        "iteration": marker,
    }


def _transition(index: int, before: dict, after: dict) -> dict:
    return {
        "event": "transition",
        "decision_index": index,
        "before_fingerprint": state_fingerprint(before),
        "after_fingerprint": state_fingerprint(after),
        "state_delta": encode_state_delta(before, after),
        "state_extras_delta": encode_extras_delta(before, after),
    }


class SuccessorStateInventoryTests(unittest.TestCase):
    def test_counts_noops_after_forced_successor(self) -> None:
        initial = _state(0, [[0, 1]])
        successor = _state(1, [[0, 1]])
        exit_state = _state(2, [])
        rows = [
            _transition(0, initial, successor),
            _transition(1, successor, successor),
            _transition(2, successor, successor),
            _transition(3, successor, exit_state),
        ]
        result = successor_dynamics(initial, rows)
        self.assertEqual(result["exact_successor_noop_streak"], 2)
        self.assertEqual(result["decisions_to_first_state_change"], 3)
        self.assertEqual(result["decisions_to_first_strict_conflict_drop"], 3)
        self.assertFalse(result["strict_drop_right_censored"])

    def test_marks_unresolved_successor_as_right_censored(self) -> None:
        initial = _state(0, [[0, 1]])
        successor = _state(1, [[0, 1]])
        result = successor_dynamics(
            initial, [_transition(0, initial, successor)]
        )
        self.assertTrue(result["state_change_right_censored"])
        self.assertTrue(result["strict_drop_right_censored"])
        self.assertIsNone(result["decisions_to_first_state_change"])

    def test_terminal_successor_is_not_censored(self) -> None:
        initial = _state(0, [[0, 1]])
        successor = _state(1, [])
        result = successor_dynamics(
            initial, [_transition(0, initial, successor)]
        )
        self.assertTrue(result["successor_feasible"])
        self.assertFalse(result["state_change_right_censored"])
        self.assertFalse(result["strict_drop_right_censored"])


if __name__ == "__main__":
    unittest.main()
