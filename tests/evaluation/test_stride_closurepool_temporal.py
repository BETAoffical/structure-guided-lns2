from __future__ import annotations

import unittest

from experiments.state_analysis import analyze_static_grid
from experiments.stride_closurepool_temporal import (
    build_corridor_layout,
    segment_visits,
    temporal_action_diagnostics,
    temporal_dependency_edges,
)


def _state(paths: list[list[int]]) -> dict:
    return {
        "rows": 1,
        "cols": 5,
        "obstacles": [0, 0, 0, 0, 0],
        "agents": [
            {
                "id": identifier,
                "start": path[0],
                "goal": path[-1],
                "path": path,
            }
            for identifier, path in enumerate(paths)
        ],
        "conflict_edges": [],
        "num_of_colliding_pairs": 0,
    }


class ClosurePoolTemporalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.before = _state(
            [
                [0, 1, 2, 3],
                [3, 2, 1, 0],
                [4, 4, 4, 4],
            ]
        )

    def test_corridor_layout_uses_map_degree_only(self) -> None:
        layout = build_corridor_layout(analyze_static_grid(self.before))
        self.assertEqual(layout["low_degree_cell_count"], 5)
        self.assertEqual(len(layout["segments"]), 1)
        self.assertEqual(layout["segments"][0]["cell_count"], 5)

    def test_temporal_dependencies_detect_reverse_shared_edge(self) -> None:
        layout = build_corridor_layout(analyze_static_grid(self.before))
        visits = segment_visits(self.before, layout)
        edges = temporal_dependency_edges(visits)
        self.assertEqual(set(edges), {(0, 1), (0, 2), (1, 2)})
        self.assertGreater(edges[(0, 1)]["opposing_overlap_count"], 0)
        self.assertEqual(edges[(0, 2)]["opposing_overlap_count"], 0)

    def test_action_metrics_separate_internal_and_boundary_queue(self) -> None:
        layout = build_corridor_layout(analyze_static_grid(self.before))
        metrics = temporal_action_diagnostics(
            self.before,
            self.before,
            {0, 1},
            layout=layout,
        )
        self.assertEqual(metrics["internal_dependency_edge_count"], 1)
        self.assertEqual(metrics["boundary_dependency_edge_count"], 2)
        self.assertEqual(metrics["boundary_dependency_outside_agent_count"], 1)
        self.assertEqual(metrics["internal_opposing_dependency_edge_count"], 1)
        self.assertEqual(metrics["new_dependency_edge_count"], 0)


if __name__ == "__main__":
    unittest.main()
