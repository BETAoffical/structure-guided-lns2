from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.local_representation_audit import (
    FORBIDDEN_FEATURE_FRAGMENTS,
    _map_folds,
    analyze_state,
    articulation_cells,
    build_local_indexes,
    realized_neighborhood_features,
    reconstruct_conflicts,
    relabel,
)


def _agent(agent_id: int, path: list[int]) -> dict:
    return {
        "id": agent_id,
        "start": path[0],
        "goal": path[-1],
        "path": path,
        "path_cost": len(path) - 1,
        "shortest_path_cost": max(1, len(set(path)) - 1),
        "delay": max(0, len(path) - len(set(path))),
        "conflict_degree": 0,
    }


def _outcome(state_id: str, trial: int, conflicts: int, generated: int) -> dict:
    horizons = []
    for horizon in (1, 4):
        horizons.append(
            {
                "horizon": horizon,
                "available": True,
                "solved": conflicts == 0,
                "conflicts_after": conflicts,
                "conflict_auc": float(conflicts * horizon),
                "branch_runtime": 0.01 * horizon,
                "low_level_delta": {"generated": generated},
            }
        )
    return {
        "state_id": state_id,
        "action_valid": True,
        "candidate_action": {
            "mode": "seed",
            "seed_agent": 0,
            "heuristic": "target",
            "neighborhood_size": 1,
        },
        "trial_index": trial,
        "trial_seed": trial + 10,
        "horizon_outcomes": horizons,
        "steps": [
            {"step": 0},
            {"step": 1, "metrics": {"neighborhood": [0]}},
        ],
    }


def _write_collection(root: Path, split: str = "train") -> None:
    state_id = "episode__decision_0000"
    inner = {
        "rows": 1,
        "cols": 3,
        "obstacles": [0, 0, 0],
        "agents": [_agent(0, [0, 1]), _agent(1, [2, 2])],
        "conflict_edges": [],
        "num_of_colliding_pairs": 0,
        "iteration": 0,
        "sum_of_costs": 2,
        "low_level": {"generated": 2, "runs": 2},
        "context": {
            "split": split,
            "map_id": "map-a",
            "task_id": "task-a",
            "agent_count": 2,
            "layout_mode": "regular_beltway",
            "layout_variant": "full",
            "scenario_type": "balanced_bidirectional",
            "task_variant": "balanced_80",
            "topology_metrics": {},
        },
    }
    state = {
        "state_id": state_id,
        "episode_id": "episode",
        "decision_index": 0,
        "state": inner,
    }
    episode = root / "counterfactual" / "train" / "episode"
    episode.mkdir(parents=True)
    (episode / "states.jsonl").write_text(json.dumps(state) + "\n", encoding="utf-8")
    (episode / "outcomes.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (_outcome(state_id, 0, 3, 10), _outcome(state_id, 1, 1, 30))
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "ok",
        "states_file": "counterfactual/train/episode/states.jsonl",
        "outcomes_file": "counterfactual/train/episode/outcomes.jsonl",
    }
    (root / "counterfactual_manifest.jsonl").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )


def _candidate(state: str, map_id: str, size: int, generated: int) -> dict:
    return {
        "state_id": state,
        "map_id": map_id,
        "task_id": state,
        "candidate_key": f"0:target:{size}",
        "candidate_action": {
            "seed_agent": 0,
            "heuristic": "target",
            "neighborhood_size": size,
        },
        "outcomes": {
            "1": {
                "solved": False,
                "solved_rate": 0.0,
                "conflicts_after": 2,
                "conflict_auc": 3.0,
                "generated": generated,
                "branch_runtime": 0.1,
            }
        },
    }


class LocalRepresentationAuditTests(unittest.TestCase):
    def test_reconstructs_vertex_edge_and_goal_wait_conflicts(self) -> None:
        events = reconstruct_conflicts(
            [_agent(0, [0, 1]), _agent(1, [1, 0]), _agent(2, [2, 2, 1])]
        )
        signatures = {(event.kind, event.time, event.left, event.right) for event in events}
        self.assertIn(("edge", 1, 0, 1), signatures)
        self.assertIn(("vertex", 2, 0, 2), signatures)

    def test_articulation_and_path_heat_features(self) -> None:
        self.assertEqual(articulation_cells(1, 3, [0, 0, 0]), {1})
        state = {
            "rows": 1,
            "cols": 3,
            "obstacles": [0, 0, 0],
            "agents": [_agent(0, [0, 1, 1]), _agent(1, [2, 1])],
            "conflict_edges": [[0, 1]],
        }
        analysis = analyze_state(state)
        self.assertEqual(analysis.visit_heat[1], 3)
        self.assertEqual(analysis.agent_heat[1], 2)
        features = realized_neighborhood_features(state, analysis, 0, 2, [0, 1])
        self.assertEqual(features["realized.actual_size"], 2.0)
        self.assertEqual(features["realized.internal_conflict_edges"], 1.0)
        self.assertGreater(features["realized.path_articulation_ratio"], 0.0)

    def test_trial_aggregation_and_feature_leakage_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_collection(root)
            action, realized, integrity = build_local_indexes(
                root, expected_outcomes=None
            )
        self.assertTrue(integrity["passed"])
        self.assertEqual((len(action), len(realized)), (1, 2))
        self.assertEqual(action[0]["trial_count"], 2)
        self.assertEqual(action[0]["outcomes"]["1"]["conflicts_after"], 2.0)
        for row in realized:
            for features in row["features"].values():
                self.assertFalse(
                    any(
                        fragment in name.lower()
                        for name in features
                        for fragment in FORBIDDEN_FEATURE_FRAGMENTS
                    )
                )

    def test_effectiveness_and_compute_pareto_expose_size_bias(self) -> None:
        rows = [
            _candidate("state-a", "map-a", 4, 10),
            _candidate("state-a", "map-a", 8, 20),
        ]
        effectiveness = relabel(rows, 1, "effectiveness")
        compute = relabel(rows, 1, "compute_aware")
        self.assertEqual(sum(row["pareto"] for row in effectiveness), 2)
        self.assertEqual(sum(row["pareto"] for row in compute), 1)
        self.assertTrue(compute[0]["pareto"])

    def test_map_folds_have_no_group_leakage_and_are_deterministic(self) -> None:
        rows = [
            _candidate(f"state-{index}", f"map-{index}", 4, 10)
            for index in range(3)
        ]
        first = _map_folds(rows)
        second = _map_folds(rows)
        self.assertEqual(first, second)
        for fold in first:
            self.assertFalse(set(fold["train_maps"]) & set(fold["validation_maps"]))

    def test_forbidden_ood_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_collection(root, split="test_ood_layout")
            with self.assertRaisesRegex(ValueError, "forbidden Test/OOD"):
                build_local_indexes(root, expected_outcomes=None)

    def test_index_build_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_collection(root)
            first = build_local_indexes(root, expected_outcomes=None)[:2]
            second = build_local_indexes(root, expected_outcomes=None)[:2]
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
