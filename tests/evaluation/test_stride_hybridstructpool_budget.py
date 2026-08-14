from __future__ import annotations

import unittest

from experiments.neighborhood_candidates import candidate_id
from experiments.stride_hybridstructpool_budget import _fraction, _state_job


def _candidate(agents: list[int], kind: str, group: str | None = None) -> dict:
    normalized = sorted(set(agents))
    return {
        "candidate_id": candidate_id(normalized),
        "agents": normalized,
        "actual_size": len(normalized),
        "candidate_kind": kind,
        "features": {
            "realized.incident_event_coverage": 0.5,
            "realized.internal_conflict_coverage": 0.25,
            "realized.component_coverage_max": 0.5,
        },
        "structpool_family_groups": [group] if group else [],
    }


class HybridStructPoolBudgetTest(unittest.TestCase):
    def test_state_job_preserves_v2_outside_budget_and_never_exceeds_cap(self) -> None:
        groups = (
            "topology_boundary",
            "conflict_component",
            "spatiotemporal_hotspot",
            "bottleneck_crossing",
            "path_overlap",
        )
        structural = [
            _candidate([10 + index, 30 + index], "structural", group)
            for index, group in enumerate(groups)
        ]
        causal = [_candidate([50, 51], "causalclosure")]
        row = _state_job(
            {
                "state_fingerprint": "state-a",
                "map_id": "map-a",
                "task_id": "task-a",
                "solver_seed": 1,
                "base": [_candidate([1, 2], "base")],
                "structural": structural,
                "causal": causal,
                "legacy_robust_action_ids": [structural[0]["candidate_id"]],
                "causalclosure_only_best_candidate_id": causal[0]["candidate_id"],
            }
        )

        for budget in (6, 8, 12):
            result = row["budgets"][str(budget)]
            self.assertLessEqual(result["selected_candidate_count"], budget)
            self.assertEqual(result["base_candidate_recall_count"], 1)
            self.assertTrue(result["budget_respected"])
            self.assertFalse(result["candidate_outcomes_used"])
            self.assertFalse(result["fixed_family_size_preference_used"])
        self.assertTrue(row["budgets"]["6"]["causalclosure_only_best_preserved"])

    def test_fraction_handles_empty_registered_subgroup(self) -> None:
        self.assertEqual(_fraction(0, 0), 1.0)
        self.assertEqual(_fraction(1, 2), 0.5)


if __name__ == "__main__":
    unittest.main()
