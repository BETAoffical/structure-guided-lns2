from __future__ import annotations

import unittest

from experiments.stride_loopguard_trigger_audit import (
    evaluate_variant,
    trigger_decisions,
)


def _row(
    decision: int,
    candidate: str,
    *,
    kind: str = "structural",
    agent_jaccard: float = 1.0,
    conflict_jaccard: float = 1.0,
) -> dict[str, object]:
    return {
        "decision_index": decision,
        "selected_candidate_id": candidate,
        "selected_kind": kind,
        "adjacent_agent_jaccard": agent_jaccard,
        "pre_action_adjacent_conflict_jaccard": conflict_jaccard,
    }


class LoopGuardTriggerAuditTests(unittest.TestCase):
    def test_exact_triplet_uses_forced_first_action_history(self) -> None:
        rows = [_row(0, "a"), _row(1, "a"), _row(2, "a"), _row(3, "b")]
        decisions = trigger_decisions(
            rows,
            {
                "id": "exact-triplet",
                "consecutive_structural_actions": 3,
                "same_candidate_id_across_run": True,
            },
        )
        self.assertEqual(decisions, [2])

    def test_overlap_variant_requires_structural_run_and_conflict_persistence(self) -> None:
        variant = {
            "id": "overlap",
            "consecutive_structural_actions": 3,
            "minimum_adjacent_agent_jaccard": 0.9,
            "minimum_adjacent_pre_action_conflict_jaccard": 0.8,
        }
        self.assertEqual(
            trigger_decisions(
                [_row(0, "a"), _row(1, "b", agent_jaccard=0.95), _row(2, "c", agent_jaccard=0.95)],
                variant,
            ),
            [2],
        )
        self.assertEqual(
            trigger_decisions(
                [_row(0, "a"), _row(1, "b", conflict_jaccard=0.7), _row(2, "c")],
                variant,
            ),
            [],
        )
        self.assertEqual(
            trigger_decisions(
                [_row(0, "a"), _row(1, "b", kind="base"), _row(2, "c")],
                variant,
            ),
            [],
        )

    def test_evaluate_variant_applies_registered_rates(self) -> None:
        pair_rows = []
        for classification, count in (("adverse", 2), ("beneficial", 1), ("neutral", 1)):
            for index in range(count):
                trigger = classification == "adverse"
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
                            [_row(0, "a"), _row(1, "a"), _row(2, "a")]
                            if trigger
                            else [_row(0, "a"), _row(1, "b"), _row(2, "c")]
                        ),
                    }
                )
        result = evaluate_variant(
            pair_rows,
            {
                "id": "exact-triplet",
                "consecutive_structural_actions": 3,
                "same_candidate_id_across_run": True,
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
        self.assertEqual(
            result["by_classification"]["adverse"]["early_trigger_rate"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
