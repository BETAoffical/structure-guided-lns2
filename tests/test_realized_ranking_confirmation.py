from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.realized_ranking_confirmation import (
    _dataset_design,
    _evaluation_worker,
    _proposal_seed,
    _proposal_worker,
    qualification_report,
)
from experiments.realized_ranking_confirmation_analysis import _acceptance
from experiments.repair_collection import state_fingerprint


def make_state() -> dict:
    agents = []
    for agent_id in range(4):
        agents.append(
            {
                "id": agent_id,
                "start": agent_id,
                "goal": agent_id + 4,
                "path_cost": 4,
                "shortest_path_cost": 3,
                "delay": agent_id,
                "conflict_degree": 1 if agent_id < 2 else 0,
                "path": [agent_id, agent_id + 1, agent_id + 4],
            }
        )
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": False,
        "done": False,
        "iteration": 0,
        "rows": 3,
        "cols": 3,
        "sum_of_costs": 16,
        "num_of_colliding_pairs": 1,
        "runtime": 0.1,
        "low_level": {"expanded": 4, "generated": 8, "reopened": 0, "runs": 4},
        "obstacles": [0] * 9,
        "conflict_edges": [[0, 1]],
        "agents": agents,
        "context": {
            "split": "confirmation",
            "map_id": "confirmation_regular_beltway_0000",
            "task_id": "task",
            "layout_mode": "regular_beltway",
            "task_variant": "balanced_80",
            "agent_count": 4,
        },
    }


class FakeProposalEnvironment:
    def __init__(self, mutate: bool = False) -> None:
        self.state = make_state()
        self.mutate = mutate

    def reset(self, seed: int) -> dict:
        return self.state

    def get_state(self) -> dict:
        return self.state

    def propose(self, action: dict) -> dict:
        if self.mutate:
            self.state = {**self.state, "iteration": 1}
        other = 1 if int(action["seed_agent"]) == 0 else 0
        return {
            "action_valid": True,
            "generated": True,
            "neighborhood": [int(action["seed_agent"]), other],
        }


class FakeEvaluationEnvironment:
    def __init__(self) -> None:
        self.state = make_state()

    def reset(self, seed: int) -> dict:
        return self.state

    def step(self, action: dict) -> dict:
        after = {**self.state, "feasible": True, "done": True, "num_of_colliding_pairs": 0}
        after["low_level"] = {"expanded": 6, "generated": 12, "reopened": 0, "runs": 6}
        return {
            "observation": after,
            "metrics": {
                "action_valid": True,
                "neighborhood": list(action["agents"]),
                "step_runtime": 0.01,
            },
        }


def make_rows() -> list[dict]:
    rows = []
    layouts = ("regular_beltway", "compartmentalized", "dead_end_aisles")
    variants = ("balanced_80", "balanced_100", "bottleneck_80", "bottleneck_100")
    for layout_index, layout in enumerate(layouts):
        for map_index in range(4):
            map_id = f"confirmation_{layout}_{map_index:04d}"
            for task_index, variant in enumerate(variants):
                rows.append(
                    {
                        "split": "confirmation",
                        "map_id": map_id,
                        "task_id": f"{map_id}__task_{task_index:04d}",
                        "layout_mode": layout,
                        "task_variant": variant,
                        "agent_count": 80 if variant.endswith("80") else 100,
                        "map_seed": 1000 + 10 * layout_index + map_index,
                        "task_seed": 2000 + len(rows),
                    }
                )
    return rows


class ConfirmationTests(unittest.TestCase):
    def test_registered_dataset_design_and_qualification(self) -> None:
        rows = make_rows()
        design = _dataset_design(rows, "confirmation")
        self.assertTrue(design["passed"])
        qualification = [
            {
                **row,
                "solver_seed": 0,
                "initial_conflicts": 3,
                "repairable": True,
                "status": "ok",
                "error": None,
            }
            for row in rows
        ]
        config = {
            "qualification": {
                "minimum_initial_conflicts": 1,
                "maximum_initial_conflicts": 200,
                "maximum_agent_count": 100,
                "minimum_repairable_tasks": 36,
                "minimum_repairable_tasks_per_layout": 12,
            }
        }
        report = qualification_report(
            rows,
            qualification,
            config,
            design,
            {"passed": True},
            formal=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["paired_map_count"], 12)

        rows[-1]["task_variant"] = "balanced_80"
        self.assertFalse(_dataset_design(rows, "confirmation")["passed"])

    def test_proposal_seed_is_deterministic_and_namespaced(self) -> None:
        first = _proposal_seed("state", 1, "target", 8, 0)
        self.assertEqual(first, _proposal_seed("state", 1, "target", 8, 0))
        self.assertNotEqual(first, _proposal_seed("state", 1, "target", 8, 1))

    def test_proposal_worker_is_outcome_blind_and_state_preserving(self) -> None:
        row = {
            "split": "confirmation",
            "map_id": "map",
            "task_id": "task",
            "layout_mode": "regular_beltway",
            "task_variant": "balanced_80",
            "agent_count": 4,
        }
        with tempfile.TemporaryDirectory() as directory:
            job = {
                "row": row,
                "solver_seed": 0,
                "dataset_root": directory,
                "environment": {},
                "proposal": {
                    "max_seed_agents": 2,
                    "heuristics": ["target"],
                    "neighborhood_sizes": [2],
                    "trials": 2,
                    "candidates_per_family": 2,
                },
                "output_root": directory,
                "run_fingerprint": "run",
                "resume": False,
            }
            with patch(
                "experiments.realized_ranking_confirmation._make_environment",
                return_value=FakeProposalEnvironment(),
            ):
                result = _proposal_worker(job)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["proposal_count"], 4)
            self.assertGreaterEqual(result["candidate_count"], 1)

            with patch(
                "experiments.realized_ranking_confirmation._make_environment",
                return_value=FakeProposalEnvironment(mutate=True),
            ):
                failed = _proposal_worker({**job, "run_fingerprint": "mutated"})
            self.assertEqual(failed["status"], "error")
            self.assertIn("changed the source repair state", failed["error"])

    def test_explicit_evaluation_uses_disjoint_seed_and_fixed_agents(self) -> None:
        state = make_state()
        state_row = {
            "state_id": "state",
            "episode_id": "episode",
            "state_fingerprint": state_fingerprint(state),
            "solver_seed": 0,
            "candidates": [
                {
                    "candidate_id": "candidate",
                    "agents": [0, 1],
                    "selection_families": ["target:2"],
                    "proposal_seeds": [10, 11],
                }
            ],
        }
        row = {
            "split": "confirmation",
            "map_id": "map",
            "task_id": "task",
        }
        with tempfile.TemporaryDirectory() as directory:
            job = {
                "row": row,
                "state_row": state_row,
                "dataset_root": directory,
                "environment": {},
                "evaluation_trials": 2,
                "output_root": directory,
                "run_fingerprint": "run",
                "resume": False,
            }
            with patch(
                "experiments.realized_ranking_confirmation._make_environment",
                side_effect=lambda *args, **kwargs: FakeEvaluationEnvironment(),
            ):
                result = _evaluation_worker(job)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["outcome_count"], 2)

    def test_acceptance_requires_simple_baselines_and_no_size_collapse(self) -> None:
        summaries = {
            "uniform_random": {"pareto_top1_hit_rate": 0.2, "mean_conflict_regret": 0.8},
            "internal_conflict_coverage": {
                "pareto_top1_hit_rate": 0.3,
                "mean_conflict_regret": 0.6,
            },
            "realized_dynamic": {
                "pareto_top1_hit_rate": 0.5,
                "mean_conflict_regret": 0.3,
                "maximum_size_share": 0.7,
            },
        }
        comparison = {
            "pareto_top1_gain": 0.1,
            "relative_conflict_regret_reduction": 0.2,
            "maps_no_worse": 10,
        }
        bootstrap = {
            "hit_gain_95_ci": [-0.1, 0.2],
            "conflict_improvement_95_ci": [-0.1, 0.2],
        }
        thresholds = {
            "minimum_top1_gain": 0.05,
            "minimum_conflict_regret_reduction": 0.05,
            "minimum_maps_no_worse": 8,
            "maximum_size_share": 0.8,
        }
        report = _acceptance(
            summaries,
            comparison,
            bootstrap,
            {"multiple_sizes_supported": True},
            thresholds,
        )
        self.assertTrue(report["passed"])
        summaries["realized_dynamic"]["maximum_size_share"] = 0.81
        self.assertFalse(
            _acceptance(
                summaries,
                comparison,
                bootstrap,
                {"multiple_sizes_supported": True},
                thresholds,
            )["passed"]
        )


if __name__ == "__main__":
    unittest.main()
