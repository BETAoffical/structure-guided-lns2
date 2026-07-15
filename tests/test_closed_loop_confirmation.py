from __future__ import annotations

import tempfile
import unittest
import json
import pickle
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.closed_loop_confirmation import (
    _closed_loop_episode_worker,
    closed_loop_dataset_design,
    closed_loop_qualification_report,
    feature_range_diagnostic,
    fixed_budget_conflict_auc,
    generate_online_candidates,
    online_candidate_rows,
    proposal_random_seed,
    repair_random_seed,
    load_frozen_policy_bundle,
    PortablePairwiseModel,
    score_online_candidates,
)
from experiments.closed_loop_confirmation_analysis import (
    closed_loop_acceptance,
    compare_policies,
)
from experiments.realized_neighborhood_ranking_audit import _feature_profiles
from experiments.local_representation_audit import analyze_state
from experiments.repair_collection import state_fingerprint


def _agent(identifier: int, path: list[int], conflicts: int = 0) -> dict:
    return {
        "id": identifier,
        "start": path[0],
        "goal": path[-1],
        "path": path,
        "path_cost": len(path) - 1,
        "shortest_path_cost": max(1, len(path) - 1),
        "conflict_degree": conflicts,
        "delay": identifier,
    }


def make_state(conflicts: int = 1) -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": conflicts == 0,
        "done": conflicts == 0,
        "iteration": 0,
        "runtime": 0.01,
        "rows": 2,
        "cols": 4,
        "sum_of_costs": 8,
        "num_of_colliding_pairs": conflicts,
        "low_level": {"expanded": 4, "generated": 8, "reopened": 0, "runs": 4},
        "obstacles": [0] * 8,
        "conflict_edges": [[0, 1]] if conflicts else [],
        "agents": [
            _agent(0, [0, 1, 2], conflicts),
            _agent(1, [1, 0, 4], conflicts),
            _agent(2, [5, 6, 7]),
            _agent(3, [7, 7, 7]),
        ],
        "context": {
            "split": "closed_loop",
            "map_id": "map-a",
            "task_id": "task-a",
            "layout_mode": "regular_beltway",
            "layout_variant": "fixture",
            "scenario_type": "balanced_bidirectional",
            "task_variant": "balanced_80",
            "agent_count": 4,
            "mean_shortest_distance": 2.0,
            "dominant_flow_ratio": 1.0,
            "topology_metrics": {
                "articulation_count": 0,
                "average_free_degree": 2.0,
                "dead_end_cell_count": 0,
                "route_redundancy_proxy": 1.0,
            },
        },
    }


def make_candidate(identifier: str, agents: list[int], family: str) -> dict:
    return {
        "candidate_id": identifier,
        "agents": agents,
        "actual_size": len(agents),
        "selection_families": [family],
        "proposal_count_by_family": {family: 2},
        "proposal_seeds": [10, 11],
        "seed_agents": [0],
    }


def make_dataset_rows() -> list[dict]:
    rows = []
    variants = ("balanced_80", "balanced_100", "bottleneck_80", "bottleneck_100")
    for layout_number, layout in enumerate(
        ("regular_beltway", "compartmentalized", "dead_end_aisles")
    ):
        for map_number in range(2):
            map_id = f"closed_loop_{layout}_{map_number:04d}"
            for variant in variants:
                rows.append(
                    {
                        "split": "closed_loop",
                        "map_id": map_id,
                        "task_id": f"{map_id}__{variant}",
                        "layout_mode": layout,
                        "task_variant": variant,
                        "agent_count": 80 if variant.endswith("80") else 100,
                        "map_seed": 1000 + layout_number * 10 + map_number,
                        "task_seed": 2000 + len(rows),
                    }
                )
    return rows


class FakeEstimator:
    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        probability = np.where(values[:, 0] >= 0.0, 0.75, 0.25)
        return np.column_stack((1.0 - probability, probability))


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeProposalEnvironment:
    def __init__(self, state: dict) -> None:
        self.state = state
        self.calls = 0

    def propose(self, action: dict) -> dict:
        self.calls += 1
        agents = [0, 1] if action["heuristic"] != "random" else [0, 2]
        return {
            "action_valid": True,
            "generated": True,
            "neighborhood": agents,
        }

    def get_state(self) -> dict:
        return self.state


class ZeroConflictEnvironment:
    def reset(self, seed: int) -> dict:
        return make_state(0)

    def step(self, action: dict) -> dict:
        raise AssertionError("zero-conflict episode must not execute a repair")


class ClosedLoopConfirmationTests(unittest.TestCase):
    def test_registered_dataset_design_and_qualification_keep_zero_conflicts(self) -> None:
        rows = make_dataset_rows()
        design = closed_loop_dataset_design(rows, "closed_loop")
        self.assertTrue(design["passed"])
        qualification = []
        for index, row in enumerate(rows):
            conflicts = 0 if index < 3 else 5
            qualification.append(
                {
                    **row,
                    "initial_conflicts": conflicts,
                    "state_fingerprint": f"state-{index}",
                    "status": "ok",
                    "error": None,
                }
            )
        report = closed_loop_qualification_report(
            rows,
            qualification,
            {
                "qualification": {
                    "minimum_nonzero_states": 18,
                    "minimum_nonzero_states_per_layout": 4,
                    "minimum_active_maps": 5,
                },
                "severity_thresholds": {"low_max": 0.001, "medium_max": 0.01},
            },
            design,
            {"passed": True},
            formal=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["initial_feasible_count"], 3)
        self.assertEqual(report["nonzero_state_count"], 21)

    def test_online_features_equal_the_audited_offline_extractor(self) -> None:
        state = make_state()
        candidate = make_candidate("candidate-a", [0, 1], "target:4")
        online = online_candidate_rows(state, [candidate])[0]["features"]
        expected = _feature_profiles(state, analyze_state(state), candidate)
        self.assertEqual(online, expected)
        for profile in online.values():
            self.assertFalse(any("conflicts_after" in name for name in profile))

    def test_online_proposals_are_deterministic_and_do_not_change_state(self) -> None:
        state = make_state()
        environment = FakeProposalEnvironment(state)
        config = {
            "max_seed_agents": 1,
            "heuristics": ["target", "collision", "random"],
            "neighborhood_sizes": [4],
            "trials": 2,
            "candidates_per_family": 1,
        }
        first, metrics = generate_online_candidates(
            environment,
            state,
            task_id="task-a",
            solver_seed=0,
            decision_index=0,
            proposal_config=config,
        )
        second, _ = generate_online_candidates(
            environment,
            state,
            task_id="task-a",
            solver_seed=0,
            decision_index=0,
            proposal_config=config,
        )
        self.assertEqual(first, second)
        self.assertEqual(metrics["proposal_count"], 6)
        self.assertEqual(state_fingerprint(environment.state), state_fingerprint(state))

    def test_proposal_and_repair_seeds_are_deterministic_and_disjoint(self) -> None:
        proposal = proposal_random_seed("task", 0, "state", 1, 2, "target", 8, 3)
        self.assertEqual(
            proposal,
            proposal_random_seed("task", 0, "state", 1, 2, "target", 8, 3),
        )
        repair = repair_random_seed("task", 0, "state", 1, "candidate", [proposal])
        self.assertNotEqual(repair, proposal)
        self.assertEqual(
            repair,
            repair_random_seed("task", 0, "state", 1, "candidate", [proposal]),
        )

    def test_pairwise_scoring_is_deterministic_and_hash_breaks_ties(self) -> None:
        rows = [
            {
                "candidate_key": "b",
                "features": {"realized_dynamic": {"x": 1.0}},
            },
            {
                "candidate_key": "a",
                "features": {"realized_dynamic": {"x": 0.0}},
            },
        ]
        model = SimpleNamespace(
            profile="realized_dynamic", feature_names=["x"], estimator=FakeEstimator()
        )
        selected, scores, margin = score_online_candidates(rows, model)
        self.assertEqual(selected, 0)
        self.assertEqual(scores, [0.75, 0.25])
        self.assertAlmostEqual(margin, 0.5)
        tied_model = SimpleNamespace(
            profile="realized_dynamic",
            feature_names=["missing"],
            estimator=FakeEstimator(),
        )
        selected, _, _ = score_online_candidates(rows, tied_model)
        self.assertEqual(selected, 1)

    def test_portable_tree_inference_matches_a_binary_split(self) -> None:
        model = PortablePairwiseModel(
            profile="realized_dynamic",
            feature_names=["x"],
            baseline=0.0,
            trees=[
                [
                    {
                        "value": 0.0,
                        "feature_idx": 0,
                        "num_threshold": 0.0,
                        "missing_go_to_left": True,
                        "left": 1,
                        "right": 2,
                        "is_leaf": False,
                    },
                    {"value": -1.0, "is_leaf": True},
                    {"value": 1.0, "is_leaf": True},
                ]
            ],
        )
        probabilities = model.predict_positive([[-1.0], [1.0]])
        self.assertLess(probabilities[0], 0.5)
        self.assertGreater(probabilities[1], 0.5)

    def test_feature_range_diagnostic_treats_missing_one_hot_as_zero(self) -> None:
        row = {"features": {"realized_dynamic": {"a": 2.0}}}
        diagnostic = feature_range_diagnostic(
            row, "realized_dynamic", {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        )
        self.assertEqual(diagnostic["outside_features"], ["a"])
        self.assertEqual(diagnostic["outside_fraction"], 0.5)

    def test_frozen_loader_prefers_registered_portable_index_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index_path = root / "ranking_index.jsonl"
            index_path.write_text(
                json.dumps(
                    {
                        "features": {
                            "proposal_dynamic": {"x": 1.0},
                            "realized_dynamic": {"x": 2.0},
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            model_rows = []
            hashes = {}
            for profile in ("proposal_dynamic", "realized_dynamic"):
                model = SimpleNamespace(
                    profile=profile, feature_names=["x"], estimator=FakeEstimator()
                )
                path = root / "models" / f"{profile}.pkl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("wb") as stream:
                    pickle.dump(model, stream)
                hashes[profile] = _digest(path)
                model_rows.append(
                    {
                        "profile": profile,
                        "model_file": path.relative_to(root).as_posix(),
                        "model_sha256": hashes[profile],
                    }
                )
            (root / "freeze_manifest.json").write_text(
                json.dumps(
                    {
                        "confirmation_labels_seen": False,
                        "development_index": "C:\\\\nonportable\\\\index.jsonl",
                        "models": model_rows,
                    }
                ),
                encoding="utf-8",
            )
            bundle = load_frozen_policy_bundle(
                root,
                {
                    "development_index": str(index_path),
                    "development_index_sha256": _digest(index_path),
                    "model_sha256": hashes,
                },
            )
        self.assertEqual(set(bundle.models), {"proposal_dynamic", "realized_dynamic"})

    def test_fixed_budget_auc_penalizes_failure(self) -> None:
        self.assertEqual(fixed_budget_conflict_auc([4, 2, 0], 4, success=True), 4.0)
        self.assertEqual(fixed_budget_conflict_auc([4, 2], 4, success=False), 9.0)
        with self.assertRaises(ValueError):
            fixed_budget_conflict_auc([], 4, success=False)

    def test_zero_conflict_episode_skips_the_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            job = {
                "row": {
                    "split": "closed_loop",
                    "map_id": "map-a",
                    "task_id": "task-a",
                    "layout_mode": "regular_beltway",
                    "task_variant": "balanced_80",
                    "agent_count": 4,
                },
                "policy": "official_adaptive",
                "solver_seed": 0,
                "output_root": directory,
                "run_fingerprint": "run",
                "resume": False,
                "dataset_root": directory,
                "environment": {},
                "max_decisions": 100,
                "metric_iteration_budget": 100,
                "wall_time_budget_seconds": 300.0,
                "proposal": {},
            }
            with patch(
                "experiments.closed_loop_confirmation._make_environment",
                return_value=ZeroConflictEnvironment(),
            ):
                result = _closed_loop_episode_worker(job)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["summary"]["success"])
        self.assertEqual(result["summary"]["repair_iterations"], 0)

    def test_comparison_uses_failure_penalties_and_map_pairing(self) -> None:
        def row(policy: str, map_id: str, task: str, auc: float, seconds: float) -> dict:
            return {
                "policy": policy,
                "map_id": map_id,
                "task_id": task,
                "solver_seed": 0,
                "status": "ok",
                "summary": {
                    "repairable": True,
                    "fixed_budget_conflict_auc": auc,
                    "capped_wall_time_to_feasible": seconds,
                },
            }

        adaptive = [row("official_adaptive", "map-a", "a", 100, 10)]
        realized = [row("realized_dynamic", "map-a", "a", 80, 8)]
        report = compare_policies(adaptive, realized, 20)
        self.assertEqual(report["paired_repairable_count"], 1)
        self.assertAlmostEqual(
            report["metrics"]["fixed_budget_conflict_auc"]["relative_improvement"],
            0.2,
        )

    def test_acceptance_requires_success_and_one_complete_metric_gate(self) -> None:
        common = {
            "error_count": 0,
            "success_count": 24,
            "invalid_action_count": 0,
            "fingerprint_mismatch_count": 0,
        }
        comparison = {
            "metrics": {
                "fixed_budget_conflict_auc": {
                    "relative_improvement": 0.1,
                    "maps_no_worse": 5,
                    "bootstrap": {"improvement_95_ci": [-0.1, 0.2]},
                },
                "capped_wall_time_to_feasible": {
                    "relative_improvement": -0.2,
                    "maps_no_worse": 1,
                    "bootstrap": {"improvement_95_ci": [-0.3, -0.1]},
                },
            }
        }
        report = closed_loop_acceptance(
            {"passed": True},
            {
                "official_adaptive": dict(common),
                "proposal_dynamic": dict(common),
                "realized_dynamic": dict(common),
            },
            comparison,
            {"passed": True},
            {"minimum_metric_improvement": 0.05, "minimum_maps_no_worse": 4},
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["qualifying_metrics"], ["fixed_budget_conflict_auc"])


if __name__ == "__main__":
    unittest.main()
