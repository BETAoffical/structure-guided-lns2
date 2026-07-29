from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.stall_trigger_counterfactual import (
    compare_trial_branches,
    load_trigger_plan,
    ordered_trial_branches,
    paired_continuation_seed,
)


def _row(branch: str, rank: int, conflicts: int, seconds: float) -> dict:
    return {
        "task_id": "task",
        "solver_seed": 1,
        "trigger_decision_index": 9,
        "trial_index": 0,
        "trigger_resolution": "confirmed_stall",
        "branch": branch,
        "first_candidate_rank": rank,
        "final_conflicts": conflicts,
        "total_decision_seconds": seconds,
        "selection_seconds": 0.1,
        "pp_replan_seconds": seconds - 0.1,
        "feasible": conflicts == 0,
    }


class StallTriggerCounterfactualTests(unittest.TestCase):
    def test_branch_order_is_deterministic_and_reversed_on_paired_trial(self) -> None:
        branches = [{"branch": name} for name in ("baseline", "rank2", "rank3")]
        first = ordered_trial_branches(
            branches, state_anchor_fingerprint="fingerprint", trial_index=0
        )
        second = ordered_trial_branches(
            branches, state_anchor_fingerprint="fingerprint", trial_index=1
        )
        repeated = ordered_trial_branches(
            branches, state_anchor_fingerprint="fingerprint", trial_index=0
        )
        self.assertEqual(first, repeated)
        self.assertEqual(second, list(reversed(first)))
        self.assertEqual(
            {row["branch"] for row in first}, {row["branch"] for row in branches}
        )

    def test_seed_is_paired_by_trial_and_step(self) -> None:
        first = paired_continuation_seed("fingerprint", 2, 1)
        self.assertEqual(first, paired_continuation_seed("fingerprint", 2, 1))
        self.assertNotEqual(first, paired_continuation_seed("fingerprint", 2, 2))
        self.assertNotEqual(first, paired_continuation_seed("fingerprint", 3, 1))

    def test_comparison_keeps_fixed_first_suggestion_separate_from_oracle(self) -> None:
        comparisons = compare_trial_branches(
            [
                _row("v2_rank1", 1, 8, 3.0),
                _row("rescue_rank2", 2, 7, 4.0),
                _row("rescue_rank3", 3, 5, 2.0),
            ]
        )
        by_policy = {row["policy"]: row for row in comparisons}
        self.assertEqual(by_policy["first_suggestion"]["rescue_rank"], 2)
        self.assertFalse(by_policy["first_suggestion"]["rescue_dominates"])
        self.assertEqual(by_policy["oracle_rescue"]["rescue_rank"], 3)
        self.assertTrue(by_policy["oracle_rescue"]["rescue_dominates"])

    def test_load_trigger_plan_requires_json_lists_and_unique_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "triggers.csv"
            fields = (
                "task_id",
                "solver_seed",
                "trigger_decision_index",
                "resolution",
                "state_anchor_fingerprint",
                "suggested_rescue_candidate_ids",
                "suggested_rescue_ranks",
            )
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "task_id": "task",
                        "solver_seed": 1,
                        "trigger_decision_index": 9,
                        "resolution": "premature_trigger",
                        "state_anchor_fingerprint": "fingerprint",
                        "suggested_rescue_candidate_ids": json.dumps(["b", "c"]),
                        "suggested_rescue_ranks": json.dumps([2, 3]),
                    }
                )
            rows = load_trigger_plan(path)
            self.assertEqual(rows[0]["suggested_rescue_ranks"], [2, 3])
            with path.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writerow(
                    {
                        "task_id": "task",
                        "solver_seed": 1,
                        "trigger_decision_index": 9,
                        "resolution": "premature_trigger",
                        "state_anchor_fingerprint": "fingerprint",
                        "suggested_rescue_candidate_ids": json.dumps(["b"]),
                        "suggested_rescue_ranks": json.dumps([2]),
                    }
                )
            with self.assertRaisesRegex(ValueError, "repeats"):
                load_trigger_plan(path)


if __name__ == "__main__":
    unittest.main()
