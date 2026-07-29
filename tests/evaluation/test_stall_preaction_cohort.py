from __future__ import annotations

import unittest

from experiments.stall_preaction_cohort import (
    assign_map_folds,
    classify_preaction_decisions,
    outcome_blind_map_sample,
)


def _decision(
    index: int,
    state: str,
    *,
    outcome: str,
    no_progress: bool,
) -> dict:
    return {
        "decision_index": index,
        "before_repair_fingerprint": state,
        "repair_outcome": outcome,
        "no_progress": no_progress,
    }


class StallPreactionCohortTests(unittest.TestCase):
    def test_classifies_progress_and_recovery_runs_without_duplicate_states(self) -> None:
        decisions = [
            _decision(0, "a", outcome="conflict_reduced", no_progress=False),
            _decision(1, "b", outcome="hard_failure", no_progress=True),
            _decision(2, "b", outcome="conflict_reduced", no_progress=False),
            _decision(3, "c", outcome="accepted_noop", no_progress=True),
            _decision(4, "c", outcome="hard_failure", no_progress=True),
            _decision(5, "c", outcome="state_changed_no_reduction", no_progress=False),
        ]
        rows = classify_preaction_decisions(
            decisions, long_stall_threshold=3, future_observation_decisions=2
        )
        self.assertEqual(
            [row["observational_class"] for row in rows],
            [
                "immediate_progress",
                "single_no_progress_recovery",
                "immediate_progress",
                "multi_no_progress_recovery",
                "state_changed_no_reduction",
            ],
        )
        self.assertEqual(rows[1]["no_progress_run_length"], 1)
        self.assertEqual(rows[3]["no_progress_run_length"], 2)

    def test_distinguishes_long_recovery_confirmed_and_unresolved(self) -> None:
        long_recovery = [
            _decision(i, "a", outcome="hard_failure", no_progress=True)
            for i in range(3)
        ] + [_decision(3, "a", outcome="conflict_reduced", no_progress=False)]
        confirmed = [
            _decision(i, "b", outcome="accepted_noop", no_progress=True)
            for i in range(5)
        ]
        unresolved = [
            _decision(i, "c", outcome="hard_failure", no_progress=True)
            for i in range(4)
        ]
        self.assertEqual(
            classify_preaction_decisions(
                long_recovery, long_stall_threshold=3, future_observation_decisions=2
            )[0]["observational_class"],
            "long_no_progress_recovery",
        )
        self.assertEqual(
            classify_preaction_decisions(
                confirmed, long_stall_threshold=3, future_observation_decisions=2
            )[0]["observational_class"],
            "confirmed_long_stall",
        )
        self.assertEqual(
            classify_preaction_decisions(
                unresolved, long_stall_threshold=3, future_observation_decisions=2
            )[0]["observational_class"],
            "unresolved_no_progress",
        )

    def test_control_selection_is_outcome_blind(self) -> None:
        rows = [
            {
                "state_key": f"state-{index}",
                "map_id": f"map-{index % 3}",
                "observational_class": "class-a",
            }
            for index in range(18)
        ]
        first = outcome_blind_map_sample(rows, 9, master_seed=7)
        changed = [
            {**row, "observational_class": "class-b" if index % 2 else "class-c"}
            for index, row in enumerate(rows)
        ]
        second = outcome_blind_map_sample(changed, 9, master_seed=7)
        self.assertEqual(
            [row["state_key"] for row in first],
            [row["state_key"] for row in second],
        )

    def test_map_folds_are_map_isolated_and_deterministic(self) -> None:
        maps = [f"map-{index}" for index in range(8)]
        first = assign_map_folds(maps + maps, master_seed=11, fold_count=4)
        second = assign_map_folds(reversed(maps), master_seed=11, fold_count=4)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(maps))
        self.assertEqual(set(first.values()), {0, 1, 2, 3})


if __name__ == "__main__":
    unittest.main()
