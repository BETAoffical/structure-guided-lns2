from __future__ import annotations

import collections
import unittest

from experiments.state_analysis import ConflictEvent, StateAnalysis
from lns2_selector.runtime.frontierdependencypool import (
    FRONTIERDEPENDENCYPOOL_ID,
    generate_frontierdependency_candidates,
    select_frontierdependency_candidate,
)
from lns2_selector.runtime.temporal_state import temporal_history_context


def _state() -> tuple[dict, StateAnalysis]:
    agents = [
        {"id": agent, "path": [agent, agent + 1, agent + 2], "conflict_degree": 2}
        for agent in range(6)
    ]
    events = [
        ConflictEvent(1, "vertex", 0, 1, (1,)),
        ConflictEvent(2, "vertex", 1, 2, (2,)),
        ConflictEvent(1, "edge", 0, 3, (3, 4)),
        ConflictEvent(2, "edge", 0, 3, (3, 4)),
        ConflictEvent(3, "vertex", 0, 4, (4,)),
        ConflictEvent(3, "vertex", 1, 5, (5,)),
    ]
    visit_heat: collections.Counter[int] = collections.Counter()
    agent_heat: collections.Counter[int] = collections.Counter()
    for row in agents:
        visit_heat.update(row["path"])
        agent_heat.update(set(row["path"]))
    state = {
        "agents": agents,
        "conflict_edges": [[0, 1], [1, 2], [0, 3], [0, 4], [1, 5]],
        "num_of_colliding_pairs": 5,
        "rows": 3,
        "cols": 4,
        "obstacles": [0] * 12,
    }
    analysis = StateAnalysis(
        rows=3,
        cols=4,
        free_cells=set(range(12)),
        degrees={cell: 3 for cell in range(12)},
        articulation=set(),
        obstacle_rate_2={},
        obstacle_rate_4={},
        visit_heat=visit_heat,
        agent_heat=agent_heat,
        events=events,
        pair_set={(0, 1), (1, 2), (0, 3), (0, 4), (1, 5)},
        component_id={agent: 0 for agent in range(6)},
        component_members={0: set(range(6))},
    )
    return state, analysis


class FrontierDependencyPoolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.state, self.analysis = _state()
        self.base = {"candidate_id": "base", "agents": [0, 1, 2]}
        self.history = temporal_history_context(
            recent_neighborhoods=[[2, 3]],
            persistent_conflict_edges=[[0, 3], [1, 5]],
            agent_repair_counts={2: 3, 0: 1, 1: 1},
            recent_pp_history=[(False, True, False)],
        )

    def test_candidates_are_pre_action_bounded_and_deterministic(self) -> None:
        first = generate_frontierdependency_candidates(
            self.state,
            self.analysis,
            base_candidate=self.base,
            history=self.history,
            maximum_neighborhood_size=5,
        )
        second = generate_frontierdependency_candidates(
            self.state,
            self.analysis,
            base_candidate=self.base,
            history=self.history,
            maximum_neighborhood_size=5,
        )
        self.assertEqual(first, second)
        self.assertTrue(first.candidates)
        self.assertLessEqual(len(first.candidates), 6)
        self.assertEqual(
            {row["frontierdependency_variant"] for row in first.candidates},
            {"compact-augment", "same-size-exchange"},
        )
        for row in first.candidates:
            self.assertEqual(
                row["frontierdependencypool_id"], FRONTIERDEPENDENCYPOOL_ID
            )
            self.assertLessEqual(len(row["added_blockers"]), 8)
            self.assertLessEqual(row["actual_size"], 5)
            self.assertEqual(
                set(map(str, row["added_blockers"])), set(row["blocker_evidence"])
            )
            self.assertFalse(row["future_outcome_used"])
            self.assertFalse(row["pp_probe_used"])
            self.assertGreater(row["converted_boundary_edge_count"], 0)
            if row["frontierdependency_variant"] == "same-size-exchange":
                self.assertEqual(row["actual_size"], len(self.base["agents"]))
                self.assertTrue(row["same_size_as_core"])

    def test_tight_cap_rejects_augment_but_keeps_exchange(self) -> None:
        result = generate_frontierdependency_candidates(
            self.state,
            self.analysis,
            base_candidate=self.base,
            history=self.history,
            maximum_neighborhood_size=3,
        )
        self.assertTrue(
            any(
                row.get("rejection_reason") == "maximum_neighborhood_size"
                for row in result.attempts
            )
        )
        self.assertTrue(result.candidates)
        self.assertTrue(
            all(
                row["frontierdependency_variant"] == "same-size-exchange"
                for row in result.candidates
            )
        )

    def test_selector_is_transparent_and_variant_specific(self) -> None:
        result = generate_frontierdependency_candidates(
            self.state,
            self.analysis,
            base_candidate=self.base,
            history=self.history,
            maximum_neighborhood_size=5,
        )
        for variant in ("compact-augment", "same-size-exchange"):
            selected, diagnostic = select_frontierdependency_candidate(
                self.base, result.candidates, variant=variant
            )
            self.assertEqual(selected["frontierdependency_variant"], variant)
            self.assertIsNone(diagnostic["rejection_reason"])
        selected, diagnostic = select_frontierdependency_candidate(
            self.base, [], variant="compact-augment"
        )
        self.assertEqual(selected["candidate_id"], "base")
        self.assertEqual(diagnostic["rejection_reason"], "no_eligible_candidate")


if __name__ == "__main__":
    unittest.main()
