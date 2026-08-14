from __future__ import annotations

import unittest

from experiments.neighborhood_candidates import candidate_id
from experiments.stride_hybridstructpool_audit import _ratio, _state_job


def _candidate(agents: list[int], kind: str) -> dict:
    normalized = sorted(set(agents))
    return {
        "candidate_id": candidate_id(normalized),
        "agents": normalized,
        "actual_size": len(normalized),
        "candidate_kind": kind,
        "selection_families": [kind],
        "structpool_family_groups": [],
    }


class HybridStructPoolAuditTest(unittest.TestCase):
    def test_state_manifest_preserves_all_sources_without_outcomes(self) -> None:
        base = _candidate([1, 2], "base")
        structural = _candidate([3, 4], "structural")
        causal = _candidate([5, 6], "causalclosure")
        row = _state_job(
            {
                "state_fingerprint": "state-a",
                "map_id": "map-a",
                "task_id": "task-a",
                "solver_seed": 7,
                "base": [base],
                "structural": [structural],
                "causal": [causal],
                "legacy_robust_action_ids": [structural["candidate_id"]],
                "causalclosure_only_best_candidate_id": causal["candidate_id"],
            }
        )

        self.assertEqual(row["union_candidate_count"], 3)
        self.assertEqual(row["base_candidate_recall_count"], 1)
        self.assertEqual(row["structural_candidate_recall_count"], 1)
        self.assertEqual(row["causal_candidate_recall_count"], 1)
        self.assertEqual(row["legacy_robust_action_recall_count"], 1)
        self.assertTrue(row["causalclosure_only_best_preserved"])
        self.assertFalse(row["candidate_outcomes_used_to_construct_pool"])
        self.assertFalse(row["runtime_budget_applied"])
        self.assertNotIn("seed_mean", row)

    def test_cross_source_duplicate_retains_both_provenances(self) -> None:
        base = _candidate([1, 2], "base")
        structural = _candidate([1, 2], "structural")
        row = _state_job(
            {
                "state_fingerprint": "state-a",
                "map_id": "map-a",
                "task_id": "task-a",
                "solver_seed": 7,
                "base": [base],
                "structural": [structural],
                "causal": [],
                "legacy_robust_action_ids": [structural["candidate_id"]],
                "causalclosure_only_best_candidate_id": None,
            }
        )

        self.assertEqual(row["union_candidate_count"], 1)
        self.assertEqual(row["exact_duplicate_count"], 1)
        self.assertEqual(
            row["provenance_by_candidate_id"][base["candidate_id"]],
            ["structshell_equal_four_size", "v2_base"],
        )

    def test_empty_denominator_is_vacuous_recall(self) -> None:
        self.assertEqual(_ratio(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
