from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from pathlib import Path

from research.studies.context.context_audit import PairwiseModel
from research.studies.representation.local_representation_audit import (
    FORBIDDEN_FEATURE_FRAGMENTS,
    _dominance_pairs,
    _dominates,
    _map_folds,
    _permuted_fold_records,
    analyze_state,
    analyze_static_grid,
    articulation_cells,
    build_local_indexes,
    feature_diagnostics,
    realized_neighborhood_features,
    reconstruct_conflicts,
    relabel,
    scientific_result_payload,
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


def _outcome(
    state_id: str,
    trial: int,
    conflicts: int,
    generated: int,
    neighborhood: list[int] | None = None,
) -> dict:
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
            {"step": 1, "metrics": {"neighborhood": neighborhood or [0]}},
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
            for row in (
                _outcome(state_id, 0, 3, 10, [0]),
                _outcome(state_id, 1, 1, 30, [1]),
            )
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


class _DifferenceEstimator:
    def predict_proba(self, matrix):
        import numpy as np

        probability = 1.0 / (1.0 + np.exp(-matrix[:, 0]))
        return np.column_stack((1.0 - probability, probability))


def _permutation_candidate(
    state_id: str, task_id: str, action_value: float, context_value: float
) -> dict:
    return {
        "state_id": state_id,
        "map_id": f"map-{task_id}",
        "task_id": task_id,
        "candidate_key": f"candidate-{action_value}",
        "candidate_action": {
            "seed_agent": 0,
            "heuristic": "target",
            "neighborhood_size": 4 if action_value < 0 else 8,
        },
        "features": {
            "local_pre_context": {
                "action.value": action_value,
                "context.value": context_value,
            }
        },
        "outcomes": {
            "1": {
                "solved": False,
                "solved_rate": 0.0,
                "conflicts_after": 1.0 if action_value > 0 else 2.0,
                "conflict_auc": 1.0 if action_value > 0 else 2.0,
                "generated": 10.0,
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

    def test_cached_static_grid_matches_uncached_analysis(self) -> None:
        state = {
            "rows": 2,
            "cols": 3,
            "obstacles": [0, 0, 1, 0, 0, 0],
            "agents": [_agent(0, [0, 1, 1]), _agent(1, [4, 1])],
            "conflict_edges": [[0, 1]],
        }
        static_grid = analyze_static_grid(state)
        self.assertEqual(
            analyze_state(state), analyze_state(state, static_grid=static_grid)
        )
        changed = dict(state)
        changed["obstacles"] = [0, 0, 0, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "cached static grid"):
            analyze_state(changed, static_grid=static_grid)

    def test_iterative_articulation_handles_open_grids_and_long_corridor(self) -> None:
        for side in (32, 64):
            with self.subTest(side=side):
                self.assertEqual(
                    articulation_cells(side, side, [0] * (side * side)), set()
                )
        length = 1500
        self.assertEqual(
            articulation_cells(1, length, [0] * length), set(range(1, length - 1))
        )

    def test_non_contiguous_agent_ids_are_supported(self) -> None:
        state = {
            "rows": 1,
            "cols": 2,
            "obstacles": [0, 0],
            "agents": [_agent(2, [0, 1]), _agent(7, [1, 0])],
            "conflict_edges": [[2, 7]],
        }
        analysis = analyze_state(state)
        self.assertEqual(analysis.pair_set, {(2, 7)})
        self.assertEqual(analysis.component_members, {0: {2, 7}})

    def test_realized_neighborhood_members_are_validated(self) -> None:
        state = {
            "rows": 1,
            "cols": 2,
            "obstacles": [0, 0],
            "agents": [_agent(2, [0, 1]), _agent(7, [1, 0])],
            "conflict_edges": [[2, 7]],
        }
        analysis = analyze_state(state)
        with self.assertRaisesRegex(ValueError, "duplicate agent"):
            realized_neighborhood_features(state, analysis, 2, 2, [2, 2])
        with self.assertRaisesRegex(ValueError, "unknown agent"):
            realized_neighborhood_features(state, analysis, 2, 2, [2, 99])

    def test_state_validation_rejects_malformed_solver_data(self) -> None:
        valid = {
            "rows": 1,
            "cols": 3,
            "obstacles": [0, 0, 0],
            "agents": [_agent(0, [0, 1]), _agent(1, [2, 2])],
            "conflict_edges": [],
        }
        cases = []
        duplicate = dict(valid)
        duplicate["agents"] = [_agent(0, [0, 1]), _agent(0, [2, 2])]
        cases.append((duplicate, "duplicate agent"))
        out_of_range = dict(valid)
        out_of_range["agents"] = [_agent(0, [0, 3])]
        cases.append((out_of_range, "out-of-range"))
        obstacle = dict(valid)
        obstacle["obstacles"] = [0, 1, 0]
        cases.append((obstacle, "enters an obstacle"))
        non_adjacent = dict(valid)
        non_adjacent["agents"] = [_agent(0, [0, 2])]
        cases.append((non_adjacent, "non-adjacent"))
        unknown_edge = dict(valid)
        unknown_edge["conflict_edges"] = [[0, 99]]
        cases.append((unknown_edge, "unknown agent"))
        invalid_grid = dict(valid)
        invalid_grid["obstacles"] = [0, 0]
        cases.append((invalid_grid, "dimensions"))
        for state, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    analyze_state(state)

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
        self.assertNotEqual(
            realized[0]["realized_neighborhood_sha256"],
            realized[1]["realized_neighborhood_sha256"],
        )
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

    def test_dominance_pair_cache_matches_naive_enumeration(self) -> None:
        rows = relabel(
            [
                _candidate("state-a", "map-a", 4, 10),
                _candidate("state-a", "map-a", 8, 20),
                _candidate("state-a", "map-a", 16, 30),
            ],
            1,
            "compute_aware",
        )
        cached = _dominance_pairs(rows, "compute_aware")
        naive = []
        for left, right in itertools.combinations(rows, 2):
            if _dominates(left["outcome"], right["outcome"], "compute_aware"):
                naive.append((left, right, 1))
            elif _dominates(right["outcome"], left["outcome"], "compute_aware"):
                naive.append((left, right, 0))
        signature = lambda values: [
            (left["candidate_key"], right["candidate_key"], label)
            for left, right, label in values
        ]
        self.assertEqual(signature(cached), signature(naive))

    def test_permutation_cache_matches_naive_evaluation(self) -> None:
        rows = []
        for state_id, task_id, context in (
            ("state-a", "task-a", -1.0),
            ("state-b", "task-b", 1.0),
        ):
            rows.extend(
                [
                    _permutation_candidate(state_id, task_id, -1.0, context),
                    _permutation_candidate(state_id, task_id, 1.0, context),
                ]
            )
        labeled = relabel(rows, 1, "effectiveness")
        model = PairwiseModel(
            "local_pre_context",
            ["action.value", "context.value"],
            _DifferenceEstimator(),
        )
        cached = _permuted_fold_records(labeled, model, 1, 8, use_cache=True)
        naive = _permuted_fold_records(labeled, model, 1, 8, use_cache=False)
        self.assertEqual(cached, naive)

    def test_feature_diagnostics_and_timings_do_not_change_scientific_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_collection(root)
            action, realized, _ = build_local_indexes(root, expected_outcomes=None)
        diagnostics = feature_diagnostics(action, realized)
        self.assertTrue(diagnostics["compact_index_v2_deferred"])
        self.assertGreater(
            diagnostics["index_storage"]["realized_index"]["redundant_feature_values"],
            0,
        )
        base = {
            "schema_version": 1,
            "model_seed": 1,
            "index_sha256": "same",
            "integrity": {"passed": True},
            "folds": [],
            "pre_registration": {},
            "analyses": {},
            "context_permutation": {},
            "auxiliary_metric_regressors": {},
            "acceptance": {"passed": False},
            "timings_seconds": {"total": 1.0},
        }
        changed = dict(base)
        changed["timings_seconds"] = {"total": 999.0}
        changed["feature_diagnostics"] = diagnostics
        self.assertEqual(
            scientific_result_payload(base), scientific_result_payload(changed)
        )

    def test_forbidden_ood_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_collection(root, split="test_ood_layout")
            with self.assertRaisesRegex(ValueError, "forbidden Test/OOD"):
                build_local_indexes(root, expected_outcomes=None)

    def test_unknown_candidate_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_collection(root)
            outcomes_path = (
                root / "counterfactual" / "train" / "episode" / "outcomes.jsonl"
            )
            outcomes = [
                json.loads(line)
                for line in outcomes_path.read_text(encoding="utf-8").splitlines()
            ]
            for outcome in outcomes:
                outcome["candidate_action"]["seed_agent"] = 99
            outcomes_path.write_text(
                "\n".join(json.dumps(outcome) for outcome in outcomes) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "candidate seed.*unknown agent"):
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
