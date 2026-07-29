from __future__ import annotations

import unittest

from experiments.stall_escape_continuation import (
    compare_continuation_prefix,
    resolve_unresolved_state,
)


def _decision(index: int, no_progress: bool, state: str = "state-a") -> dict:
    return {
        "decision_index": index,
        "before_repair_fingerprint": state,
        "no_progress": no_progress,
        "repair_outcome": "hard_failure" if no_progress else "conflict_reduced",
    }


class StallEscapeContinuationTests(unittest.TestCase):
    def test_new_diagnostic_fields_do_not_change_prefix_semantics(self) -> None:
        candidate = {
            "actual_size": 4,
            "agents": [0, 1, 2, 3],
            "candidate_id": "candidate",
            "feature_out_of_range_fraction": 0.0,
            "proposal_count_by_family": {"collision:4": 1},
            "proposal_seeds": [1],
            "retained": True,
            "score": 1.0,
            "seed_agents": [0],
            "selection_families": ["collision:4"],
        }
        controller = {
            "base_selected_candidate_id": "candidate",
            "base_selected_score": 1.0,
            "candidate_pool": [candidate],
            "inference_backend": "native-portable-tree",
            "route": "model",
            "selected_candidate_id": "candidate",
            "selected_score": 1.0,
        }
        metrics = {
            "requested_mode": "explicit_neighborhood",
            "requested_random_seed": 1,
            "requested_repair_order": [],
            "action_valid": True,
            "applied_heuristic": "Collision",
            "applied_pp_random_seed": None,
            "conflicts_before": 3,
            "conflicts_after": 3,
            "conflict_delta": 0,
            "iteration": 1,
            "neighborhood": [0, 1, 2, 3],
            "repair_order": [0, 1, 2, 3],
            "replan_success": False,
            "sum_of_costs_before": 10,
            "sum_of_costs_after": 10,
        }
        source_event = {
            "event": "transition",
            "decision_index": 0,
            "before_fingerprint": "before",
            "after_fingerprint": "after",
            "action": {"mode": "explicit_neighborhood", "agents": [0, 1, 2, 3]},
            "controller": controller,
            "metrics": metrics,
            "low_level_delta": {},
            "truncated": False,
        }
        continued_event = {
            **source_event,
            "controller": {
                **controller,
                "candidate_pool": [
                    {**candidate, "selection_rank_by_family": {"collision:4": 0}}
                ],
            },
            "metrics": {
                **metrics,
                "requested_pp_random_seed": -1,
                "applied_pp_random_seed": -1,
            },
        }
        row = {"after_repair_fingerprint": "repair-after"}
        report = compare_continuation_prefix(
            [row],
            [{"event": "initial"}, source_event, {"event": "finish"}],
            [row],
            [{"event": "initial"}, continued_event, {"event": "finish"}],
        )
        self.assertTrue(report["passed"])

    def test_original_threshold_is_not_moved_forward(self) -> None:
        decisions = [_decision(index, True) for index in range(6)]
        result = resolve_unresolved_state(
            decisions,
            {
                "anchor_decision_index": 0,
                "threshold": 3,
                "before_repair_fingerprint": "state-a",
            },
            future_observation_decisions=3,
        )
        self.assertEqual(result["resolution"], "confirmed_long_stall")

    def test_change_inside_future_window_is_natural_recovery(self) -> None:
        decisions = [
            _decision(0, True),
            _decision(1, True),
            _decision(2, True),
            _decision(3, True),
            _decision(4, False),
        ]
        result = resolve_unresolved_state(
            decisions,
            {
                "anchor_decision_index": 0,
                "threshold": 3,
                "before_repair_fingerprint": "state-a",
            },
            future_observation_decisions=3,
        )
        self.assertEqual(result["resolution"], "natural_recovery")
        self.assertEqual(result["recovery_delay_decisions"], 2)

    def test_short_terminal_extension_remains_unresolved(self) -> None:
        result = resolve_unresolved_state(
            [_decision(index, True) for index in range(5)],
            {
                "anchor_decision_index": 0,
                "threshold": 3,
                "before_repair_fingerprint": "state-a",
            },
            future_observation_decisions=3,
        )
        self.assertEqual(result["resolution"], "unresolved_stall")


if __name__ == "__main__":
    unittest.main()
