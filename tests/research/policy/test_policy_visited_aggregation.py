from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.studies.policy.policy_visited_aggregation import (
    _aggregate_trial_results,
    _evaluation_trial_worker,
    _proposal_worker,
    build_policy_visited_index,
    candidate_core,
    policy_visited_dataset_design,
    policy_visited_qualification_report,
    select_policy_states,
)
from research.studies.policy.policy_visited_aggregation_analysis import (
    closed_loop_v2_acceptance,
    export_aggregated_portable_bundle,
    offline_acceptance,
    train_equal_state_pairwise_model,
)
from experiments.closed_loop_confirmation import generate_online_candidates
from experiments.repair_collection import state_fingerprint


def _agent(identifier: int, path: list[int], degree: int = 0) -> dict:
    return {
        "id": identifier,
        "start": path[0],
        "goal": path[-1],
        "path": path,
        "path_cost": len(path) - 1,
        "shortest_path_cost": max(1, len(path) - 1),
        "conflict_degree": degree,
        "delay": identifier,
    }


def make_state(iteration: int = 1, conflicts: int = 1) -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": conflicts == 0,
        "done": conflicts == 0,
        "iteration": iteration,
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
            "split": "policy_train",
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
    split_counts = {"policy_train": 4, "policy_validation": 2}
    for split, count in split_counts.items():
        for layout_number, layout in enumerate(
            ("regular_beltway", "compartmentalized", "dead_end_aisles")
        ):
            for map_number in range(count):
                map_id = f"{split}_{layout}_{map_number:04d}"
                map_seed = 10000 + len(rows)
                for variant in variants:
                    rows.append(
                        {
                            "split": split,
                            "map_id": map_id,
                            "task_id": f"{map_id}__{variant}",
                            "layout_mode": layout,
                            "task_variant": variant,
                            "agent_count": 80 if variant.endswith("80") else 100,
                            "map_seed": map_seed,
                            "task_seed": 20000 + len(rows),
                        }
                    )
    return rows


def make_config() -> dict:
    return {
        "splits": ["policy_train", "policy_validation"],
        "solver_seeds": [1, 2, 3],
        "dataset_design": {
            "tasks_per_map": 4,
            "task_variants": [
                "balanced_80",
                "balanced_100",
                "bottleneck_80",
                "bottleneck_100",
            ],
            "layout_counts": {
                "policy_train": {
                    "regular_beltway": 4,
                    "compartmentalized": 4,
                    "dead_end_aisles": 4,
                },
                "policy_validation": {
                    "regular_beltway": 2,
                    "compartmentalized": 2,
                    "dead_end_aisles": 2,
                },
            },
        },
        "qualification": {
            "minimum_nonzero_by_split": {
                "policy_train": 108,
                "policy_validation": 54,
            },
            "minimum_nonzero_per_layout": {
                "policy_train": 36,
                "policy_validation": 18,
            },
            "minimum_active_maps": {
                "policy_train": 11,
                "policy_validation": 5,
            },
        },
    }


class PrefixEvaluationEnvironment:
    def __init__(self) -> None:
        self.initial = make_state(iteration=0)
        self.selected = make_state(iteration=1)

    def reset(self, seed: int) -> dict:
        return self.initial

    def step(self, action: dict) -> dict:
        if action.get("mode") == "prefix":
            return {"observation": self.selected, "metrics": {}}
        after = {
            **self.selected,
            "feasible": True,
            "done": True,
            "iteration": 2,
            "num_of_colliding_pairs": 0,
            "conflict_edges": [],
            "low_level": {
                "expanded": 6,
                "generated": 12,
                "reopened": 0,
                "runs": 6,
            },
        }
        return {
            "observation": after,
            "metrics": {
                "action_valid": True,
                "neighborhood": list(action["agents"]),
                "step_runtime": 0.01,
            },
        }


class PrefixProposalEnvironment(PrefixEvaluationEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.current = self.initial

    def reset(self, seed: int) -> dict:
        self.current = self.initial
        return self.current

    def step(self, action: dict) -> dict:
        if action.get("mode") != "prefix":
            return super().step(action)
        self.current = self.selected
        return {"observation": self.current, "metrics": {}}

    def propose(self, action: dict) -> dict:
        agents = [0, 2] if action["heuristic"] == "random" else [0, 1]
        return {"action_valid": True, "generated": True, "neighborhood": agents}

    def propose_batch(self, actions: list[dict]) -> list[dict]:
        return [self.propose(action) for action in actions]

    def get_state(self) -> dict:
        return self.current


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def make_collection(root: Path) -> None:
    state = make_state()
    candidates = [
        make_candidate("candidate-a", [0, 1], "target:4"),
        make_candidate("candidate-b", [0, 2], "random:4"),
    ]
    write_json(root / "run_config.json", {"run_fingerprint": "run"})
    write_jsonl(
        root / "candidates.jsonl",
        [
            {
                "state_id": "state-a",
                "episode_id": "episode-a",
                "decision_index": 1,
                "decision_count": 3,
                "decision_fraction": 0.5,
                "stage": "middle",
                "state_fingerprint": state_fingerprint(state),
                "state": state,
                "split": "policy_train",
                "map_id": "map-a",
                "task_id": "task-a",
                "layout_mode": "regular_beltway",
                "task_variant": "balanced_80",
                "agent_count": 4,
                "candidate_count": 2,
                "candidates": candidates,
            }
        ],
    )
    outcomes = []
    for candidate_number, candidate in enumerate(candidates):
        for trial in range(4):
            conflicts = candidate_number
            outcomes.append(
                {
                    "state_id": "state-a",
                    "candidate_id": candidate["candidate_id"],
                    "evaluation_trial_index": trial,
                    "evaluation_seed": 100 + candidate_number * 10 + trial,
                    "evaluation_seed_disjoint": True,
                    "agents": candidate["agents"],
                    "actual_neighborhood": candidate["agents"],
                    "action_valid": True,
                    "solved": conflicts == 0,
                    "conflicts_before": 1,
                    "conflicts_after": conflicts,
                    "conflict_auc": (1 + conflicts) / 2,
                    "generated": 10 + candidate_number,
                    "runtime": 0.01,
                }
            )
    outcomes_path = root / "explicit" / "outcomes.jsonl"
    errors_path = root / "explicit" / "errors.jsonl"
    write_jsonl(outcomes_path, outcomes)
    write_jsonl(errors_path, [])
    write_jsonl(
        root / "collection_manifest.jsonl",
        [
            {
                "state_id": "state-a",
                "status": "ok",
                "outcomes_file": outcomes_path.relative_to(root).as_posix(),
                "errors_file": errors_path.relative_to(root).as_posix(),
            }
        ],
    )


def synthetic_index_rows() -> list[dict]:
    rows = []
    for state_number in range(4):
        for candidate in range(2):
            wins = candidate == state_number % 2
            conflicts = 0 if wins else 2
            rows.append(
                {
                    "state_id": f"state-{state_number}",
                    "candidate_id": f"candidate-{candidate}",
                    "candidate_key": f"candidate-{candidate}",
                    "map_id": f"map-{state_number // 2}",
                    "task_id": f"task-{state_number}",
                    "actual_size": 4 + 4 * candidate,
                    "selection_families": ["target:4" if candidate == 0 else "random:8"],
                    "features": {
                        "realized_dynamic": {
                            "state.conflicts": float(state_number + 1),
                            "realized.choice": float(candidate),
                            "realized.winner": float(wins),
                        },
                        "proposal_dynamic": {
                            "state.conflicts": float(state_number + 1),
                            "proposal.choice": float(candidate),
                        },
                    },
                    "outcome": {
                        "solved_rate": float(wins),
                        "conflicts_after": float(conflicts),
                        "conflict_auc": float(conflicts),
                        "generated": 10.0 + candidate,
                        "runtime": 0.01,
                    },
                    "labels": {
                        "effectiveness_pareto": wins,
                        "compute_aware_pareto": wins,
                        "runtime_sensitive_pareto": wins,
                    },
                }
            )
    return rows


class PolicyVisitedAggregationTests(unittest.TestCase):
    def test_registered_design_and_qualification_group_solver_seeds(self) -> None:
        rows = make_dataset_rows()
        config = make_config()
        design = policy_visited_dataset_design(rows, config)
        self.assertTrue(design["passed"])
        qualification = []
        for source in rows:
            for seed in (1, 2, 3):
                qualification.append(
                    {
                        **source,
                        "solver_seed": seed,
                        "initial_conflicts": 3,
                        "state_fingerprint": f"{source['task_id']}-{seed}",
                        "status": "ok",
                        "error": None,
                    }
                )
        report = policy_visited_qualification_report(
            rows,
            qualification,
            config,
            design,
            {"passed": True},
            formal=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["valid_count"], 216)
        self.assertEqual(report["nonzero_by_split"]["policy_train"], 144)
        self.assertEqual(report["nonzero_by_split"]["policy_validation"], 72)
        for row in qualification:
            row["state_fingerprint"] = row["task_id"]
        duplicate = policy_visited_qualification_report(
            rows,
            qualification,
            config,
            design,
            {"passed": True},
            formal=True,
        )
        self.assertFalse(duplicate["passed"])
        self.assertEqual(len(duplicate["duplicate_solver_seed_trajectories"]), 3)

    def test_early_middle_late_selection_is_unique_and_deterministic(self) -> None:
        self.assertEqual(select_policy_states(0), [])
        self.assertEqual(select_policy_states(2), [0, 1])
        self.assertEqual(select_policy_states(7), [0, 3, 6])
        self.assertEqual(select_policy_states(8), [0, 3, 7])

    def test_candidate_core_ignores_scores_but_preserves_provenance(self) -> None:
        candidate = make_candidate("candidate", [1, 0], "target:4")
        candidate["score"] = 100.0
        candidate["feature_out_of_range_fraction"] = 0.5
        normalized = candidate_core(candidate)
        self.assertEqual(normalized["agents"], [0, 1])
        self.assertNotIn("score", normalized)
        self.assertEqual(normalized["proposal_count_by_family"], {"target:4": 2})

    def test_trial_replays_prefix_and_keeps_four_trial_identity(self) -> None:
        selected = make_state(iteration=1)
        candidate = make_candidate("candidate", [0, 1], "collision:4")
        state_row = {
            "state_id": "state",
            "state_fingerprint": state_fingerprint(selected),
            "solver_seed": 1,
            "prefix_actions": [{"mode": "prefix"}],
            "split": "policy_train",
            "map_id": "map",
            "task_id": "task",
        }
        base = {
            "state_id": "state",
            "candidate_id": "candidate",
            "state_row": state_row,
            "candidate": candidate,
            "row": {
                "split": "policy_train",
                "map_id": "map",
                "task_id": "task",
                "agent_count": 4,
            },
            "dataset_root": ".",
            "environment": {},
            "run_fingerprint": "run",
        }
        outcomes = []
        with patch(
            "research.studies.policy.policy_visited_aggregation._make_environment",
            side_effect=lambda *_args, **_kwargs: PrefixEvaluationEnvironment(),
        ):
            for trial in range(4):
                result = _evaluation_trial_worker(
                    {
                        **base,
                        "job_id": f"trial-{trial}",
                        "evaluation_trial_index": trial,
                    }
                )
                self.assertEqual(result["status"], "ok")
                outcomes.append(result)
        seeds = {row["outcome"]["evaluation_seed"] for row in outcomes}
        self.assertEqual(len(seeds), 4)
        self.assertTrue(all(row["outcome"]["actual_neighborhood"] == [0, 1] for row in outcomes))

    def test_proposal_replay_requires_exact_source_candidate_pool(self) -> None:
        proposal = {
            "max_seed_agents": 2,
            "heuristics": ["target", "collision", "random"],
            "neighborhood_sizes": [4],
            "trials": 2,
            "candidates_per_family": 1,
        }
        environment = PrefixProposalEnvironment()
        environment.reset(1)
        environment.step({"mode": "prefix"})
        candidates, _ = generate_online_candidates(
            environment,
            environment.get_state(),
            task_id="task",
            solver_seed=1,
            decision_index=1,
            proposal_config=proposal,
        )
        state_row = {
            "state_id": "state",
            "episode_id": "episode",
            "decision_index": 1,
            "decision_count": 2,
            "decision_fraction": 1.0,
            "stage": "late",
            "state_fingerprint": state_fingerprint(make_state(iteration=1)),
            "prefix_actions": [{"mode": "prefix"}],
            "source_candidates": sorted(
                [candidate_core(value) for value in candidates],
                key=lambda value: value["candidate_id"],
            ),
            "source_selected_candidate_id": candidates[0]["candidate_id"],
            "state": make_state(iteration=1),
            "split": "policy_train",
            "map_id": "map",
            "task_id": "task",
            "layout_mode": "regular_beltway",
            "task_variant": "balanced_80",
            "agent_count": 4,
            "solver_seed": 1,
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "research.studies.policy.policy_visited_aggregation._make_environment",
            side_effect=lambda *_args, **_kwargs: PrefixProposalEnvironment(),
        ):
            job = {
                "state_row": state_row,
                "row": {
                    "split": "policy_train",
                    "map_id": "map",
                    "task_id": "task",
                    "agent_count": 4,
                },
                "dataset_root": ".",
                "environment": {},
                "proposal": proposal,
                "output_root": directory,
                "run_fingerprint": "run",
                "resume": False,
            }
            result = _proposal_worker(job)
            self.assertEqual(result["status"], "ok")
            self.assertTrue(result["candidate_pool_match"])
            changed = json.loads(json.dumps(state_row))
            changed["source_candidates"][0]["agents"] = [2, 3]
            failed = _proposal_worker({**job, "state_row": changed})
            self.assertEqual(failed["status"], "error")
            self.assertIn("candidate pool", failed["error"])

    def test_index_aggregates_four_trials_and_excludes_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_collection(root)
            rows, integrity = build_policy_visited_index(root)
        self.assertTrue(integrity["passed"])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["trial_count"], 4)
        self.assertEqual(set(rows[0]["features"]), {"proposal_dynamic", "realized_dynamic"})
        self.assertFalse(
            any(
                name.startswith("context.")
                for profile in rows[0]["features"].values()
                for name in profile
            )
        )
        self.assertTrue(rows[0]["labels"]["effectiveness_pareto"])

    def test_equal_state_pairwise_training_assigns_unit_state_weight(self) -> None:
        model, diagnostic = train_equal_state_pairwise_model(
            synthetic_index_rows(),
            "realized_dynamic",
            {
                "learning_rate": 0.05,
                "max_iter": 10,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 2,
                "l2_regularization": 0.1,
                "random_state": 20260714,
            },
        )
        self.assertEqual(model.profile, "realized_dynamic")
        self.assertTrue(diagnostic["equal_state_weight"])
        self.assertAlmostEqual(diagnostic["state_weight_min"], 1.0)
        self.assertAlmostEqual(diagnostic["state_weight_max"], 1.0)

    def test_portable_export_matches_sklearn_selection_and_scores(self) -> None:
        rows = synthetic_index_rows()
        parameters = {
            "learning_rate": 0.05,
            "max_iter": 10,
            "max_leaf_nodes": 7,
            "min_samples_leaf": 2,
            "l2_regularization": 0.1,
            "random_state": 20260714,
        }
        models = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_paths = {}
            for profile in ("proposal_dynamic", "realized_dynamic"):
                model, _ = train_equal_state_pairwise_model(rows, profile, parameters)
                path = root / "models" / f"{profile}.pkl"
                path.parent.mkdir(parents=True, exist_ok=True)
                import pickle

                with path.open("wb") as stream:
                    pickle.dump(model, stream)
                models[profile] = model
                model_paths[profile] = path
            index_path = root / "index.jsonl"
            write_jsonl(index_path, rows)
            manifest, equivalence = export_aggregated_portable_bundle(
                models,
                rows,
                model_paths,
                index_path,
                root / "portable",
                {"model_parameters": parameters},
            )
        self.assertTrue(equivalence["passed"])
        self.assertEqual(len(manifest["models"]), 2)

    def test_registered_offline_and_closed_loop_gates(self) -> None:
        offline = offline_acceptance(
            {"pareto_top1_hit_rate": 0.4, "mean_conflict_regret": 0.5},
            {
                "pareto_top1_hit_rate": 0.44,
                "mean_conflict_regret": 0.46,
                "maximum_size_share": 0.7,
            },
            {
                "top1_delta_95_ci": [-0.01, 0.08],
                "conflict_regret_improvement_95_ci": [-0.02, 0.12],
            },
            {"multiple_sizes_supported": True},
            {
                "minimum_top1_improvement": 0.03,
                "minimum_conflict_regret_improvement": 0.05,
                "maximum_top1_degradation": 0.02,
                "maximum_conflict_regret_degradation": 0.05,
                "maximum_single_size_share": 0.8,
            },
        )
        self.assertTrue(offline["passed"])
        summary = {
            "success_count": 72,
            "error_count": 0,
            "invalid_action_count": 0,
            "fingerprint_mismatch_count": 0,
        }
        closed = closed_loop_v2_acceptance(
            summary,
            summary,
            summary,
            {"metrics": {"fixed_budget_conflict_auc": {"relative_improvement": 0.1}}},
            {
                "metrics": {
                    "fixed_budget_conflict_auc": {
                        "relative_improvement": -0.02,
                        "maps_no_worse": 4,
                    }
                }
            },
            {"passed": True},
            {
                "minimum_closed_loop_auc_improvement_over_adaptive": 0.05,
                "maximum_closed_loop_auc_degradation_from_v1": 0.05,
                "minimum_validation_maps_no_worse_than_v1": 4,
            },
        )
        self.assertTrue(closed["passed"])


if __name__ == "__main__":
    unittest.main()
