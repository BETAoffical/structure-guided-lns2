from __future__ import annotations

import collections
import unittest

from experiments.state_analysis import ConflictEvent, StateAnalysis
from lns2_selector.runtime.repairdependencypool import (
    REPAIRDEPENDENCYPOOL_ID,
    generate_repairdependency_candidates,
    select_repairdependency_candidate,
)


def _state() -> tuple[dict, StateAnalysis]:
    agents = [
        {"id": 0, "path": [0, 1, 2, 3], "conflict_degree": 1},
        {"id": 1, "path": [4, 1, 5, 6], "conflict_degree": 1},
        {"id": 2, "path": [3, 2, 1, 7], "conflict_degree": 0},
        {"id": 3, "path": [8, 9, 10, 11], "conflict_degree": 0},
    ]
    events = [ConflictEvent(1, "vertex", 0, 1, (1,))]
    visit_heat: collections.Counter[int] = collections.Counter()
    agent_heat: collections.Counter[int] = collections.Counter()
    for row in agents:
        visit_heat.update(row["path"])
        agent_heat.update(set(row["path"]))
    state = {
        "agents": agents,
        "conflict_edges": [[0, 1]],
        "num_of_colliding_pairs": 1,
        "rows": 4,
        "cols": 4,
        "obstacles": [0] * 16,
    }
    analysis = StateAnalysis(
        rows=4,
        cols=4,
        free_cells=set(range(16)),
        degrees={cell: (2 if cell in {1, 2} else 3) for cell in range(16)},
        articulation={1, 2},
        obstacle_rate_2={},
        obstacle_rate_4={},
        visit_heat=visit_heat,
        agent_heat=agent_heat,
        events=events,
        pair_set={(0, 1)},
        component_id={0: 0, 1: 0},
        component_members={0: {0, 1}},
    )
    return state, analysis


class RepairDependencyPoolTest(unittest.TestCase):
    def test_generation_is_pre_action_evidenced_deterministic_and_bounded(self) -> None:
        state, analysis = _state()
        first = generate_repairdependency_candidates(state, analysis)
        second = generate_repairdependency_candidates(state, analysis)
        self.assertEqual(first, second)
        self.assertTrue(first.candidates)
        self.assertLessEqual(len(first.candidates), 6)
        for candidate in first.candidates:
            self.assertEqual(
                candidate["repairdependencypool_id"], REPAIRDEPENDENCYPOOL_ID
            )
            self.assertLessEqual(len(candidate["added_blockers"]), 8)
            self.assertLessEqual(candidate["actual_size"], 32)
            self.assertIsNone(candidate["rejection_reason"])
            self.assertEqual(
                set(map(str, candidate["added_blockers"])),
                set(candidate["blocker_evidence"]),
            )
            self.assertFalse(candidate["truncated_to_near_global"])

    def test_oversized_variant_is_rejected_without_truncation(self) -> None:
        state, analysis = _state()
        result = generate_repairdependency_candidates(
            state, analysis, maximum_neighborhood_size=2
        )
        self.assertTrue(
            any(
                row.get("rejection_reason")
                == "variant_oversized_without_truncation"
                for row in result.attempts
            )
        )
        self.assertTrue(all(row["actual_size"] <= 2 for row in result.candidates))

    def test_selection_requires_unique_pareto_candidate(self) -> None:
        anchor = {"candidate_id": "v2", "agents": [0, 1]}
        candidate = {
            "candidate_id": "new",
            "agents": [0, 1, 2],
            "repairdependencypool_id": REPAIRDEPENDENCYPOOL_ID,
            "dependency_coverage_count": 1,
            "evidence_density": 2.0,
            "actual_size": 3,
            "frozen_v2_quality_score": 0.6,
        }
        selected, reason = select_repairdependency_candidate(anchor, [candidate])
        self.assertEqual(selected["candidate_id"], "new")
        self.assertEqual(reason["selection_source"], "repairdependency_unique_pareto")
        other = {
            **candidate,
            "candidate_id": "other",
            "agents": [0, 1, 3],
        }
        selected, reason = select_repairdependency_candidate(
            anchor, [candidate, other]
        )
        self.assertEqual(selected["candidate_id"], "v2")
        self.assertEqual(reason["rejection_reason"], "dependency_pareto_not_unique")


if __name__ == "__main__":
    unittest.main()
