from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from experiments.graph_feature_gbdt_audit import (
    GRAPH_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    _articulation_and_bridges,
    _core_numbers,
    _validate_config,
    build_conflict_graph,
    extract_graph_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def waiting_conflict_state() -> dict:
    return {
        "rows": 1,
        "cols": 4,
        "obstacles": [0, 0, 0, 0],
        "conflict_edges": [[10, 20]],
        "agents": [
            {"id": 10, "start": 0, "goal": 1, "path": [0, 1]},
            {"id": 20, "start": 2, "goal": 1, "path": [2, 1]},
            {"id": 30, "start": 3, "goal": 3, "path": [3, 3, 3, 3]},
        ],
    }


def swap_conflict_state() -> dict:
    return {
        "rows": 1,
        "cols": 3,
        "obstacles": [0, 0, 0],
        "conflict_edges": [[10, 20]],
        "agents": [
            {"id": 10, "start": 0, "goal": 1, "path": [0, 1]},
            {"id": 20, "start": 1, "goal": 0, "path": [1, 0]},
            {"id": 30, "start": 2, "goal": 2, "path": [2, 2, 2]},
        ],
    }


class GraphFeatureGbdtAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (PROJECT_ROOT / "configs/graph_feature_gbdt_audit.json").read_text(
                encoding="utf-8"
            )
        )

    def test_registered_model_and_feature_groups_are_frozen(self) -> None:
        _validate_config(self.config)
        self.assertEqual(len(GRAPH_FEATURE_NAMES), 51)
        self.assertEqual(len(STRUCTURAL_FEATURE_NAMES), 23)
        self.assertEqual(len(TEMPORAL_FEATURE_NAMES), 28)
        changed = json.loads(json.dumps(self.config))
        changed["model_parameters"]["max_iter"] = 101
        with self.assertRaisesRegex(ValueError, "parameters"):
            _validate_config(changed)

    def test_non_contiguous_ids_and_terminal_waiting_events(self) -> None:
        graph = build_conflict_graph(waiting_conflict_state())
        self.assertEqual(graph.agent_ids, (10, 20, 30))
        self.assertEqual(graph.edges, frozenset({(10, 20)}))
        self.assertEqual([event.time for event in graph.events], [1, 2, 3])
        self.assertEqual(graph.bridges, frozenset({(10, 20)}))
        features = extract_graph_features(graph, [10])
        self.assertEqual(features["temporal.boundary.event_mass_ratio"], 1.0)
        self.assertTrue(
            math.isclose(
                features["temporal.boundary.pair_repeat_excess_ratio"], 2 / 3
            )
        )
        self.assertTrue(
            math.isclose(features["temporal.boundary.first_time_ratio"], 1 / 3)
        )
        self.assertEqual(features["temporal.boundary.last_time_ratio"], 1.0)
        self.assertEqual(features["temporal.boundary.vertex_event_ratio"], 1.0)

    def test_articulation_bridges_and_core_on_disconnected_graph(self) -> None:
        nodes = [10, 20, 30, 40, 50]
        adjacency = {
            10: {20},
            20: {10, 30},
            30: {20},
            40: {50},
            50: {40},
        }
        articulation, bridges = _articulation_and_bridges(nodes, adjacency)
        self.assertEqual(articulation, {20})
        self.assertEqual(bridges, {(10, 20), (20, 30), (40, 50)})
        self.assertEqual(_core_numbers(nodes, adjacency), {10: 1, 20: 1, 30: 1, 40: 1, 50: 1})

    def test_candidate_order_is_irrelevant_and_unknown_agent_is_rejected(self) -> None:
        graph = build_conflict_graph(waiting_conflict_state())
        self.assertEqual(
            extract_graph_features(graph, [10, 20]),
            extract_graph_features(graph, [20, 10]),
        )
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            extract_graph_features(graph, [999])

    def test_edge_swap_is_preserved_in_temporal_composition(self) -> None:
        graph = build_conflict_graph(swap_conflict_state())
        self.assertEqual(len(graph.events), 1)
        self.assertEqual(graph.events[0].kind, "edge")
        features = extract_graph_features(graph, [10])
        self.assertEqual(features["temporal.boundary.event_mass_ratio"], 1.0)
        self.assertEqual(features["temporal.boundary.vertex_event_ratio"], 0.0)

    def test_graph_feature_names_exclude_outcome_and_static_context(self) -> None:
        forbidden = (
            "outcome",
            "after",
            "runtime",
            "generated",
            "layout",
            "task_variant",
            "agent_density",
            "context.",
        )
        self.assertFalse(
            any(fragment in name for name in GRAPH_FEATURE_NAMES for fragment in forbidden)
        )
        features = extract_graph_features(build_conflict_graph(waiting_conflict_state()), [10])
        self.assertEqual(set(features), set(GRAPH_FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(value) for value in features.values()))


if __name__ == "__main__":
    unittest.main()
