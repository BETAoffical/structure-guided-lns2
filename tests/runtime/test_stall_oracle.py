from __future__ import annotations

import json
import unittest

from experiments.stall_oracle import classify_stall_oracle_trials
from experiments.stalled_state_probe import (
    STALLED_STATE_PROBE_SCHEMA,
    STALLED_STATE_PROBE_VERSION,
    TRIAL_STATE_RESTORE,
    paired_probe_seed,
)


def _trial(
    branch: str,
    rank: int,
    trial: int,
    *,
    escaped: bool,
    agents: list[int],
) -> dict:
    seed = paired_probe_seed("full-before", trial)
    after_repair = f"after-{branch}-{trial}" if escaped else "repair-before"
    return {
        "schema": STALLED_STATE_PROBE_SCHEMA,
        "schema_version": STALLED_STATE_PROBE_VERSION,
        "complete": True,
        "run_fingerprint": "run",
        "branch_key": branch,
        "branch_aliases": "rank1" if rank == 1 else f"rank_{rank}",
        "branch_mode": "explicit_neighborhood",
        "candidate_id": branch,
        "candidate_rank": rank,
        "candidate_score": float(3 - rank),
        "candidate_size": len(agents),
        "candidate_agents": json.dumps(agents),
        "trial_index": trial,
        "random_seed": seed,
        "before_fingerprint": "full-before",
        "before_repair_fingerprint": "repair-before",
        "after_fingerprint": f"full-after-{branch}-{trial}" if escaped else "full-before",
        "after_repair_fingerprint": after_repair,
        "replay_fingerprint_match": True,
        "trial_state_restore": TRIAL_STATE_RESTORE,
        "source_before_fingerprint": "full-before",
        "restored_before_fingerprint": "restored-before",
        "full_state_fingerprint_match": "False",
        "repair_state_fingerprint_match": "True",
        "replan_success": "True",
        "repair_outcome": "conflict_reduced" if escaped else "accepted_noop",
        "terminated": "False",
        "truncated": "False",
        "conflicts_before": 10,
        "conflicts_after": 9 if escaped else 10,
        "conflict_delta": 1 if escaped else 0,
        "requested_pp_random_seed": seed,
        "applied_pp_random_seed": seed,
        "repair_order": json.dumps(agents if trial % 2 else list(reversed(agents))),
        "total_decision_seconds": 1.0,
    }


class StallOracleTests(unittest.TestCase):
    def test_official_reference_may_generate_different_neighborhoods(self) -> None:
        rows = [
            *[
                _trial("winner", 1, trial, escaped=False, agents=[1, 2, 3, 4])
                for trial in range(4)
            ],
        ]
        for trial in range(4):
            row = _trial(
                "official_adaptive",
                2,
                trial,
                escaped=trial == 0,
                agents=[10 + trial, 20 + trial],
            )
            row.update(
                {
                    "branch_aliases": "official_adaptive",
                    "branch_mode": "official",
                    "candidate_id": None,
                    "candidate_rank": None,
                    "candidate_score": None,
                }
            )
            rows.append(row)
        report = classify_stall_oracle_trials(rows)
        official = next(
            row for row in report["branches"] if row["branch_mode"] == "official"
        )
        self.assertEqual(official["unique_actual_neighborhood_count"], 4)
        self.assertIsNone(official["candidate_size"])
        self.assertEqual(report["classification"], "candidate_pool_failure")

    def test_selector_failure_requires_stable_same_pool_alternative(self) -> None:
        rows = [
            *[
                _trial("winner", 1, trial, escaped=False, agents=[1, 2, 3, 4])
                for trial in range(4)
            ],
            *[
                _trial("backup", 2, trial, escaped=True, agents=[5, 6, 7, 8])
                for trial in range(4)
            ],
        ]
        report = classify_stall_oracle_trials(rows)
        self.assertEqual(report["classification"], "selector_failure")
        self.assertTrue(report["selector_failure"])
        self.assertFalse(report["candidate_pool_failure"])

    def test_full_pool_failure_is_not_automatically_repairer_failure(self) -> None:
        rows = [
            *[
                _trial("winner", 1, trial, escaped=False, agents=[1, 2, 3, 4])
                for trial in range(4)
            ],
            *[
                _trial("backup", 2, trial, escaped=False, agents=[5, 6, 7, 8])
                for trial in range(4)
            ],
        ]
        report = classify_stall_oracle_trials(rows)
        self.assertEqual(report["classification"], "candidate_pool_failure")
        self.assertTrue(report["candidate_pool_failure"])
        self.assertFalse(report["repairer_failure"])
        self.assertFalse(report["repairer_failure_evaluated"])
        self.assertEqual(
            report["repairer_failure_status"], "not_evaluated_by_pp_only_probe"
        )

    def test_mixed_pp_results_are_order_sensitive_not_stable_escape(self) -> None:
        rows = [
            *[
                _trial("winner", 1, trial, escaped=False, agents=[1, 2, 3, 4])
                for trial in range(4)
            ],
            *[
                _trial(
                    "backup",
                    2,
                    trial,
                    escaped=trial < 2,
                    agents=[5, 6, 7, 8],
                )
                for trial in range(4)
            ],
        ]
        report = classify_stall_oracle_trials(rows)
        self.assertEqual(report["classification"], "inconclusive")
        self.assertEqual(report["pp_order_sensitive_neighborhood_count"], 1)

    def test_duplicate_trial_does_not_count_as_distinct_pp_evidence(self) -> None:
        rows = [
            *[
                _trial("winner", 1, trial, escaped=False, agents=[1, 2, 3, 4])
                for trial in range(4)
            ],
            *[
                _trial("backup", 2, trial, escaped=True, agents=[5, 6, 7, 8])
                for trial in range(4)
            ],
        ]
        rows[3] = dict(rows[0])
        with self.assertRaisesRegex(ValueError, "repeats a trial index"):
            classify_stall_oracle_trials(rows)

    def test_branches_must_cover_the_same_paired_trials(self) -> None:
        rows = [
            *[
                _trial("winner", 1, trial, escaped=False, agents=[1, 2, 3, 4])
                for trial in range(4)
            ],
            *[
                _trial("backup", 2, trial, escaped=True, agents=[5, 6, 7, 8])
                for trial in range(1, 5)
            ],
        ]
        with self.assertRaisesRegex(ValueError, "same paired trials"):
            classify_stall_oracle_trials(rows)

    def test_fractional_trial_index_is_rejected(self) -> None:
        rows = [
            *[
                _trial("winner", 1, trial, escaped=False, agents=[1, 2, 3, 4])
                for trial in range(4)
            ],
            *[
                _trial("backup", 2, trial, escaped=True, agents=[5, 6, 7, 8])
                for trial in range(4)
            ],
        ]
        rows[0]["trial_index"] = 0.5
        with self.assertRaisesRegex(ValueError, "trial_index must be an integer"):
            classify_stall_oracle_trials(rows)


if __name__ == "__main__":
    unittest.main()
