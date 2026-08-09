from __future__ import annotations

import unittest

from experiments.state_analysis import reconstruct_conflicts
from experiments.stride_closurepool_longtail import (
    _outcome_category,
    action_diagnostics,
    trajectory_diagnostics,
)


def _agent(identifier: int, path: list[int]) -> dict:
    return {
        "id": identifier,
        "start": path[0],
        "goal": path[-1],
        "path": path,
        "path_cost": len(path) - 1,
        "shortest_path_cost": len(path) - 1,
        "delay": 0,
        "conflict_degree": 0,
    }


def _state(paths: list[list[int]]) -> dict:
    agents = [_agent(index, path) for index, path in enumerate(paths)]
    events = reconstruct_conflicts(agents)
    pairs = sorted({(event.left, event.right) for event in events})
    degrees = {identifier: 0 for identifier in range(len(agents))}
    for left, right in pairs:
        degrees[left] += 1
        degrees[right] += 1
    for agent in agents:
        agent["conflict_degree"] = degrees[int(agent["id"])]
    return {
        "rows": 4,
        "cols": 4,
        "obstacles": [0] * 16,
        "agents": agents,
        "conflict_edges": [list(pair) for pair in pairs],
        "num_of_colliding_pairs": len(pairs),
    }


class ClosurePoolLongtailTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paths_both = [
            [0, 1, 2],
            [2, 1, 0],
            [12, 13, 14],
            [14, 13, 12],
        ]
        self.paths_external_only = [
            [0, 1, 2],
            [2, 6, 5, 4, 0],
            [12, 13, 14],
            [14, 13, 12],
        ]
        self.paths_none = [
            [0, 1, 2],
            [2, 6, 5, 4, 0],
            [12, 13, 14],
            [14, 10, 9, 8, 12],
        ]

    def test_action_diagnostics_detects_complete_component_repair(self) -> None:
        diagnostics = action_diagnostics(
            _state(self.paths_both),
            _state(self.paths_external_only),
            {0, 1},
        )
        self.assertEqual(diagnostics["before_conflict_pair_count"], 2)
        self.assertEqual(diagnostics["after_conflict_pair_count"], 1)
        self.assertEqual(diagnostics["internal_conflict_pair_count"], 1)
        self.assertEqual(diagnostics["boundary_conflict_pair_count"], 0)
        self.assertEqual(diagnostics["external_conflict_pair_count"], 1)
        self.assertEqual(diagnostics["fully_covered_component_count"], 1)
        self.assertEqual(diagnostics["partially_covered_component_count"], 0)
        self.assertEqual(diagnostics["conflict_pair_retention_ratio"], 0.5)
        self.assertEqual(diagnostics["new_conflict_pair_fraction"], 0.0)

    def test_trajectory_diagnostics_finds_persistent_pair_and_plateau(self) -> None:
        states = [
            _state(self.paths_both),
            _state(self.paths_external_only),
            _state(self.paths_external_only),
            _state(self.paths_external_only),
            _state(self.paths_none),
        ]
        metrics, pairs, agents = trajectory_diagnostics(
            states,
            {0, 1},
            start_index=0,
            persistent_pair_minimum=3,
        )
        self.assertEqual(metrics["maximum_single_conflict_plateau"], 3)
        self.assertEqual(metrics["persistent_conflict_pair_count"], 1)
        self.assertEqual(metrics["maximum_pair_consecutive_state_count"], 4)
        external = next(
            row for row in pairs if (row["left_agent"], row["right_agent"]) == (2, 3)
        )
        self.assertEqual(external["state_presence_count"], 4)
        self.assertIsNotNone(external["event_summary"])
        self.assertTrue(any(row["agent_id"] == 2 for row in agents))

    def test_outcome_categories_are_descriptive_and_fixed(self) -> None:
        definition = {
            "adverse_iteration_delta_minimum": 1,
            "severe_iteration_delta_minimum": 10,
            "severe_iteration_ratio_minimum": 2.0,
        }
        self.assertEqual(_outcome_category(30, 10, definition), "severe_tail")
        self.assertEqual(_outcome_category(12, 10, definition), "adverse")
        self.assertEqual(_outcome_category(8, 10, definition), "beneficial")
        self.assertEqual(_outcome_category(10, 10, definition), "tied")


if __name__ == "__main__":
    unittest.main()
