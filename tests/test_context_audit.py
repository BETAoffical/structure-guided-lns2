from __future__ import annotations

import unittest

from experiments.context_audit import (
    _pair_vector,
    _validate_split_isolation,
    candidate_features,
)


def _state() -> dict:
    return {
        "state": {
            "rows": 2,
            "cols": 3,
            "obstacles": [0, 0, 1, 0, 0, 0],
            "iteration": 2,
            "num_of_colliding_pairs": 1,
            "sum_of_costs": 9,
            "low_level": {"generated": 30, "runs": 4},
            "conflict_edges": [[0, 1]],
            "agents": [
                {
                    "id": 0,
                    "conflict_degree": 1,
                    "delay": 2,
                    "path_cost": 5,
                    "shortest_path_cost": 3,
                },
                {
                    "id": 1,
                    "conflict_degree": 1,
                    "delay": 1,
                    "path_cost": 4,
                    "shortest_path_cost": 4,
                },
            ],
            "context": {
                "agent_count": 2,
                "layout_mode": "room",
                "layout_variant": "four",
                "scenario_type": "balanced",
                "task_variant": "balanced_2",
                "mean_shortest_distance": 3.5,
                "topology_metrics": {"average_free_degree": 2.5},
            },
        }
    }


def _action(seed: int, heuristic: str, size: int) -> dict:
    return {
        "candidate_action": {
            "seed_agent": seed,
            "heuristic": heuristic,
            "neighborhood_size": size,
        }
    }


class ContextAuditTests(unittest.TestCase):
    def test_feature_profiles_preserve_ablation_boundary(self) -> None:
        features = candidate_features(
            _state(),
            _action(0, "target", 4),
            "middle",
            {
                "flow_type": "scenario_controlled",
                "opposing_flow_ratio": 0.5,
                "realized_flow_counts": {"left_to_right": 1},
            },
        )
        self.assertFalse(any(name.startswith("state.") for name in features["action_seed"]))
        self.assertFalse(any(name.startswith("context.") for name in features["dynamic"]))
        self.assertEqual(features["full_context"]["context.agent_count"], 2.0)
        self.assertEqual(
            features["full_context"][
                "context.realized_flow_counts.left_to_right_ratio"
            ],
            0.5,
        )

    def test_pair_vector_keeps_shared_context(self) -> None:
        left = {"features": candidate_features(_state(), _action(0, "target", 4), "early")}
        right = {"features": candidate_features(_state(), _action(1, "collision", 8), "early")}
        names = sorted(set(left["features"]["full_context"]) | set(right["features"]["full_context"]))
        vector = _pair_vector(left, right, "full_context", names)
        shared_count = sum(name.startswith(("state.", "context.")) for name in names)
        self.assertGreater(shared_count, 0)
        self.assertTrue(any(value != 0 for value in vector[-shared_count:]))

    def test_train_validation_map_overlap_is_rejected(self) -> None:
        rows = [
            {"split": "train", "map_id": "same", "task_id": "train-task"},
            {"split": "validation", "map_id": "same", "task_id": "validation-task"},
        ]
        with self.assertRaisesRegex(ValueError, "isolation failed"):
            _validate_split_isolation(rows)

    def test_test_or_ood_labels_are_rejected(self) -> None:
        rows = [{"split": "test_ood_layout", "map_id": "m", "task_id": "t"}]
        with self.assertRaisesRegex(ValueError, "forbidden Test/OOD"):
            _validate_split_isolation(rows)


if __name__ == "__main__":
    unittest.main()
