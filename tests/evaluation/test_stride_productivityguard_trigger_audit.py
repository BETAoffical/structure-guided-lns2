from __future__ import annotations

import unittest

from experiments.stride_productivityguard_trigger_audit import (
    evaluate_variant,
    productivity_trigger_events,
)


def _row(
    decision: int,
    candidate: str,
    before: int,
    after: int,
    *,
    kind: str = "structural",
    unresolved: float = 0.9,
) -> dict[str, object]:
    return {
        "decision_index": decision,
        "selected_candidate_id": candidate,
        "selected_kind": kind,
        "conflicts_before": before,
        "conflicts_after": after,
        "unresolved_edge_fraction": unresolved,
    }


class ProductivityGuardTriggerAuditTests(unittest.TestCase):
    def test_trigger_uses_only_two_completed_prior_repairs(self) -> None:
        events = productivity_trigger_events(
            [
                _row(0, "a", 10, 10),
                _row(1, "a", 10, 10),
                _row(2, "a", 10, 0),
            ],
            {
                "id": "zero-progress",
                "maximum_prior_cumulative_conflict_progress": 0.0,
            },
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["decision_index"], 2)
        self.assertEqual(events[0]["prior_cumulative_conflict_progress"], 0.0)

    def test_trigger_rejects_productive_history_and_low_persistence(self) -> None:
        rows = [
            _row(0, "a", 10, 8, unresolved=0.9),
            _row(1, "a", 8, 7, unresolved=0.9),
            _row(2, "a", 7, 7),
        ]
        self.assertEqual(
            productivity_trigger_events(
                rows,
                {
                    "id": "low-progress",
                    "maximum_prior_cumulative_conflict_progress": 0.1,
                },
            ),
            [],
        )
        self.assertEqual(
            productivity_trigger_events(
                [
                    _row(0, "a", 10, 10, unresolved=0.6),
                    _row(1, "a", 10, 10, unresolved=0.7),
                    _row(2, "a", 10, 10),
                ],
                {
                    "id": "persistent",
                    "maximum_prior_cumulative_conflict_progress": 0.1,
                    "minimum_prior_mean_unresolved_edge_fraction": 0.8,
                },
            ),
            [],
        )

    def test_evaluate_variant_applies_registered_rates(self) -> None:
        pair_rows = []
        for classification, count in (("adverse", 2), ("beneficial", 1), ("neutral", 1)):
            for index in range(count):
                stalled = classification == "adverse"
                pair_rows.append(
                    {
                        "state_id": f"{classification}-{index}",
                        "contrast": "continuation_after_v2",
                        "map_id": f"map-{index}",
                        "task_id": "task",
                        "solver_seed": index,
                        "challenger": f"challenger-{index}",
                        "classification": classification,
                        "normalized_auc_delta": 0.0,
                        "final_conflict_delta": 0,
                        "transitions": (
                            [
                                _row(0, "a", 10, 10),
                                _row(1, "a", 10, 10),
                                _row(2, "a", 10, 10),
                            ]
                            if stalled
                            else [
                                _row(0, "a", 10, 8),
                                _row(1, "a", 8, 6),
                                _row(2, "a", 6, 6),
                            ]
                        ),
                    }
                )
        result = evaluate_variant(
            pair_rows,
            {
                "id": "zero-progress",
                "maximum_prior_cumulative_conflict_progress": 0.0,
            },
            {
                "minimum_adverse_recall": 1.0,
                "maximum_beneficial_false_trigger_rate": 0.0,
                "maximum_neutral_false_trigger_rate": 0.0,
                "minimum_triggered_adverse_pair_count": 2,
                "minimum_triggered_adverse_map_count": 2,
                "minimum_triggered_adverse_challenger_count": 2,
            },
            early_window=20,
        )
        self.assertTrue(result["all_registered_gates_passed"])


if __name__ == "__main__":
    unittest.main()
