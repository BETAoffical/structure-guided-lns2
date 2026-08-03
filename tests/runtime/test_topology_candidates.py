from __future__ import annotations

import collections
import unittest

from experiments.state_analysis import ConflictEvent, StateAnalysis
from lns2_selector.runtime.topology_candidates import (
    generate_topology_anchor_candidates,
    generate_topology_boundary_candidates,
    merge_topology_anchor_candidates,
)


class TopologyCandidatesTest(unittest.TestCase):
    def test_pair_set_cover_is_deterministic_and_closes_relevant_pairs(self) -> None:
        state = {
            "agents": [
                {"id": index, "path": [index], "conflict_degree": 1}
                for index in range(6)
            ],
            "conflict_edges": [[0, 1], [1, 2], [3, 4]],
        }
        analysis = StateAnalysis(
            rows=2,
            cols=3,
            free_cells=set(range(6)),
            degrees={0: 3, 1: 2, 2: 3, 3: 3, 4: 3, 5: 3},
            articulation={1},
            obstacle_rate_2={},
            obstacle_rate_4={},
            visit_heat=collections.Counter(),
            agent_heat=collections.Counter(),
            events=[
                ConflictEvent(1, "vertex", 0, 1, (1,)),
                ConflictEvent(2, "vertex", 1, 2, (1,)),
                ConflictEvent(3, "vertex", 3, 4, (3,)),
            ],
            pair_set={(0, 1), (1, 2), (3, 4)},
            component_id={},
            component_members={},
        )
        first = generate_topology_anchor_candidates(state, analysis, [4])
        second = generate_topology_anchor_candidates(state, analysis, [4])
        self.assertEqual(first, second)
        articulation = next(
            row
            for row in first
            if "topology-anchor-articulation:4" in row["selection_families"]
        )
        self.assertTrue({0, 1, 2}.issubset(set(articulation["agents"])))
        self.assertEqual(articulation["actual_size"], 4)

    def test_merge_deduplicates_equal_agent_sets(self) -> None:
        base = [
            {
                "candidate_id": "same",
                "agents": [0, 1],
                "actual_size": 2,
                "selection_families": ["target:4"],
                "selection_rank_by_family": {"target:4": 0},
                "proposal_count_by_family": {"target:4": 1},
                "proposal_seeds": [7],
                "seed_agents": [0],
            }
        ]
        anchors = [
            {
                "candidate_id": "same",
                "agents": [0, 1],
                "actual_size": 2,
                "selection_families": ["topology-anchor-articulation:4"],
                "selection_rank_by_family": {"topology-anchor-articulation:4": 0},
                "proposal_count_by_family": {"topology-anchor-articulation:4": 1},
                "proposal_seeds": [],
                "seed_agents": [],
            }
        ]
        merged = merge_topology_anchor_candidates(base, anchors)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]["selection_families"]), 2)

    def test_boundary_candidate_is_deterministic_and_discourages_pair_closure(self) -> None:
        state = {
            "agents": [
                {"id": index, "path": [index], "conflict_degree": 2}
                for index in range(8)
            ],
            "conflict_edges": [[0, 1], [1, 2], [2, 3], [4, 5], [6, 7]],
        }
        events = [
            ConflictEvent(index, "vertex", left, right, (1,))
            for index, (left, right) in enumerate(state["conflict_edges"])
        ]
        analysis = StateAnalysis(
            rows=2,
            cols=4,
            free_cells=set(range(8)),
            degrees={index: (2 if index == 1 else 3) for index in range(8)},
            articulation={1},
            obstacle_rate_2={},
            obstacle_rate_4={},
            visit_heat=collections.Counter(),
            agent_heat=collections.Counter(),
            events=events,
            pair_set={tuple(edge) for edge in state["conflict_edges"]},
            component_id={0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 2, 7: 2},
            component_members={0: {0, 1, 2, 3}, 1: {4, 5}, 2: {6, 7}},
        )
        first = generate_topology_boundary_candidates(
            state, analysis, neighborhood_size=4, core_budget=2
        )
        second = generate_topology_boundary_candidates(
            state, analysis, neighborhood_size=4, core_budget=2
        )
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertTrue(all(row["actual_size"] == 4 for row in first))
        self.assertTrue(
            all(row["proposal_audit"]["global_event_boundary_ratio"] >= 0.5 for row in first)
        )


if __name__ == "__main__":
    unittest.main()
