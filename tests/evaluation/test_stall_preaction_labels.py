from __future__ import annotations

import unittest

from experiments.stall_preaction_labels import (
    derive_preaction_state_labels,
    zero_false_negative_control_gate,
)


def _selected() -> dict:
    return {
        "state_key": "state",
        "map_id": "map",
        "map_fold": 1,
        "layout_mode": "layout",
        "agent_count": 400,
        "task_id": "task",
        "solver_seed": 2,
        "decision_index": 7,
        "before_repair_fingerprint": "a" * 64,
        "cohort_role": "enriched_training",
        "observational_class": "long_no_progress_recovery",
        "candidate_rank_limit": 3,
        "candidate_rows": [
            {
                "candidate_id": f"c{rank}",
                "rank": rank,
                "actual_size": 4,
                "score": float(4 - rank),
                "neighborhood_key": f"n{rank}",
            }
            for rank in range(1, 4)
        ],
    }


def _branch(rank: int, *, escape: bool, delta: float, seconds: float) -> dict:
    return {
        "branch_key": f"b{rank}",
        "neighborhood_key": f"n{rank}",
        "trial_count": 4,
        "escape_count": 4 if escape else 0,
        "escape_fraction": 1.0 if escape else 0.0,
        "stable_escape": escape,
        "stable_failure": not escape,
        "pp_order_sensitive": False,
        "mean_conflict_delta": delta,
        "mean_total_decision_seconds": seconds,
        "outcome_counts": {},
    }


class StallPreactionLabelTests(unittest.TestCase):
    def test_false_positive_gate_counts_only_confirmed_negative_controls(self) -> None:
        gate = zero_false_negative_control_gate(
            control_state_count=48,
            control_positive_count=12,
            minimum_negative_controls=381,
        )
        self.assertEqual(gate["control_negative_count"], 36)
        self.assertEqual(
            gate["additional_zero_false_negative_controls_needed"], 345
        )
        self.assertFalse(gate["negative_control_count_gate_passed"])

    def test_trigger_requires_failed_rank1_and_stable_rank2_8_escape(self) -> None:
        selected = _selected()
        probe = {
            "before_repair_fingerprint": "a" * 64,
            "task_id": "task",
            "solver_seed": 2,
            "decision_index": 7,
            "all_candidates": True,
            "trials_per_branch": 4,
            "branch_aliases": {"rank1": "b1", "rank_2": "b2", "rank_3": "b3"},
        }
        audit = {
            "before_repair_fingerprint": "a" * 64,
            "classification": "selector_failure",
            "branches": [
                _branch(1, escape=False, delta=0.0, seconds=2.0),
                _branch(2, escape=True, delta=3.0, seconds=1.5),
                _branch(3, escape=True, delta=3.0, seconds=1.0),
            ],
        }
        state, candidates = derive_preaction_state_labels(selected, probe, audit)
        self.assertTrue(state["target_rescuable_selector_failure"])
        self.assertEqual(state["recommended_rescue_rank"], 3)
        self.assertEqual(
            [row["candidate_rank"] for row in candidates if row["recommended_rescue"]],
            [3],
        )

    def test_normal_rank1_is_not_a_trigger_even_with_good_alternatives(self) -> None:
        selected = _selected()
        probe = {
            "before_repair_fingerprint": "a" * 64,
            "task_id": "task",
            "solver_seed": 2,
            "decision_index": 7,
            "all_candidates": True,
            "trials_per_branch": 4,
            "branch_aliases": {"rank1": "b1", "rank_2": "b2", "rank_3": "b3"},
        }
        audit = {
            "before_repair_fingerprint": "a" * 64,
            "classification": "no_confirmed_v2_failure",
            "branches": [
                _branch(1, escape=True, delta=2.0, seconds=1.0),
                _branch(2, escape=True, delta=3.0, seconds=1.0),
                _branch(3, escape=False, delta=0.0, seconds=0.5),
            ],
        }
        state, candidates = derive_preaction_state_labels(selected, probe, audit)
        self.assertFalse(state["target_rescuable_selector_failure"])
        self.assertFalse(any(row["recommended_rescue"] for row in candidates))


if __name__ == "__main__":
    unittest.main()
