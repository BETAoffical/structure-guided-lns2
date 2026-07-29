from __future__ import annotations

import unittest

from experiments.stall_oracle_batch import (
    ORACLE_CLASSIFICATIONS,
    normalize_oracle_plan,
    oracle_job_id,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "source": "/source",
        "task_id": "task",
        "solver_seed": 0,
        "decision_index": 7,
        "before_repair_fingerprint": "a" * 64,
        "trials_per_branch": 4,
        "all_candidates": True,
    }
    row.update(overrides)
    return row


class StallOracleBatchTests(unittest.TestCase):
    def test_accepts_every_classification_emitted_by_strict_oracle(self) -> None:
        self.assertEqual(
            ORACLE_CLASSIFICATIONS,
            {
                "candidate_pool_failure",
                "inconclusive",
                "no_confirmed_v2_failure",
                "selector_failure",
            },
        )

    def test_normalizes_strict_full_pool_plan(self) -> None:
        rows = normalize_oracle_plan([_row()])
        self.assertEqual(rows[0]["decision_index"], 7)
        self.assertTrue(rows[0]["all_candidates"])
        self.assertEqual(oracle_job_id(2, rows[0]), "oracle-002-d0007-aaaaaaaaaa")

    def test_rejects_duplicate_states(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_oracle_plan([_row(), _row()])

    def test_rejects_weak_or_partial_trials(self) -> None:
        for changed in (
            {"trials_per_branch": 3},
            {"all_candidates": False},
            {"before_repair_fingerprint": "short"},
        ):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "invalid"):
                    normalize_oracle_plan([_row(**changed)])


if __name__ == "__main__":
    unittest.main()
