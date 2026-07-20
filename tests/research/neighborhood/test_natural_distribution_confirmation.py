from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.studies.neighborhood.natural_distribution_confirmation import (
    _aggregate_trial_results,
    _evaluation_trial_worker,
    _normalize_trial_result,
    _state_storage_id,
    conflict_density,
    conflict_severity,
    natural_qualification_report,
)
from research.studies.neighborhood.natural_distribution_confirmation_analysis import natural_acceptance
from experiments.repair_collection import _failed_job_result, state_fingerprint


def make_state(conflicts: int = 1) -> dict:
    agents = [
        {
            "id": identifier,
            "start": identifier,
            "goal": identifier + 4,
            "path_cost": 4,
            "shortest_path_cost": 3,
            "delay": identifier,
            "conflict_degree": int(identifier < 2),
            "path": [identifier, identifier + 1, identifier + 4],
        }
        for identifier in range(4)
    ]
    edges = [[0, 1]] if conflicts else []
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": conflicts == 0,
        "done": conflicts == 0,
        "iteration": 0,
        "rows": 3,
        "cols": 3,
        "sum_of_costs": 16,
        "num_of_colliding_pairs": conflicts,
        "runtime": 0.1,
        "low_level": {"expanded": 4, "generated": 8, "reopened": 0, "runs": 4},
        "obstacles": [0] * 9,
        "conflict_edges": edges,
        "agents": agents,
        "context": {"split": "confirmation_v2"},
    }


class FakeEvaluationEnvironment:
    def __init__(self, conflicts: int = 1) -> None:
        self.state = make_state(conflicts)

    def reset(self, seed: int) -> dict:
        return self.state

    def step(self, action: dict) -> dict:
        after = {
            **self.state,
            "feasible": True,
            "done": True,
            "num_of_colliding_pairs": 0,
            "conflict_edges": [],
            "low_level": {"expanded": 6, "generated": 12, "reopened": 0, "runs": 6},
        }
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
    variants = ("balanced_80", "balanced_100", "bottleneck_80", "bottleneck_100")
    for layout_index, layout in enumerate(
        ("regular_beltway", "compartmentalized", "dead_end_aisles")
    ):
        for map_index in range(4):
            map_id = f"confirmation_v2_{layout}_{map_index:04d}"
            for variant in variants:
                rows.append(
                    {
                        "split": "confirmation_v2",
                        "map_id": map_id,
                        "task_id": f"{map_id}__{variant}",
                        "layout_mode": layout,
                        "task_variant": variant,
                        "agent_count": 80 if variant.endswith("80") else 100,
                        "map_seed": 1000 + layout_index * 10 + map_index,
                        "task_seed": 2000 + len(rows),
                    }
                )
    return rows


class NaturalDistributionConfirmationTests(unittest.TestCase):
    def test_qualification_retains_zero_and_high_conflict_tasks(self) -> None:
        rows = make_rows()
        qualification = []
        for index, row in enumerate(rows):
            conflicts = 0 if index in {0, 16, 32} else (619 if index == 17 else 3)
            qualification.append(
                {
                    **row,
                    "initial_conflicts": conflicts,
                    "repairable": conflicts > 0,
                    "initial_feasible": conflicts == 0,
                    "status": "ok",
                    "error": None,
                }
            )
        report = natural_qualification_report(
            rows,
            qualification,
            {
                "qualification": {
                    "minimum_nonzero_states": 30,
                    "minimum_nonzero_states_per_layout": 8,
                    "minimum_active_maps": 10,
                },
                "severity_thresholds": {"low_max": 0.001, "medium_max": 0.01},
            },
            {"passed": True},
            {"passed": True},
            formal=True,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["valid_count"], 48)
        self.assertEqual(report["initial_feasible_count"], 3)
        self.assertEqual(report["nonzero_state_count"], 45)
        self.assertIn(rows[17]["task_id"], report["repairable_task_ids"])
        recorded = {
            row["task_id"]: row
            for row in report["natural_distribution"]["tasks"]
        }
        self.assertEqual(recorded[rows[17]["task_id"]]["initial_conflicts"], 619)

    def test_conflict_density_and_fixed_severity_boundaries(self) -> None:
        thresholds = {"low_max": 0.001, "medium_max": 0.01}
        self.assertAlmostEqual(conflict_density(4950, 100), 1.0)
        self.assertEqual(conflict_severity(0.001, thresholds), "low")
        self.assertEqual(conflict_severity(0.0011, thresholds), "medium")
        self.assertEqual(conflict_severity(0.01, thresholds), "medium")
        self.assertEqual(conflict_severity(0.0101, thresholds), "high")
        self.assertEqual(_state_storage_id("state"), _state_storage_id("state"))
        self.assertNotEqual(_state_storage_id("state"), _state_storage_id("other"))
        self.assertLess(len(_state_storage_id("state")), 32)

    def test_trial_worker_is_isolated_and_uses_explicit_agents(self) -> None:
        state = make_state()
        state_row = {
            "state_id": "state",
            "state_fingerprint": state_fingerprint(state),
            "solver_seed": 0,
        }
        candidate = {
            "candidate_id": "candidate",
            "agents": [0, 1],
            "selection_families": ["collision:4"],
            "proposal_seeds": [10, 11],
        }
        job = {
            "job_id": "trial",
            "state_id": "state",
            "candidate_id": "candidate",
            "evaluation_trial_index": 0,
            "row": {
                "split": "confirmation_v2",
                "map_id": "map",
                "task_id": "task",
                "agent_count": 4,
            },
            "state_row": state_row,
            "candidate": candidate,
            "solver_seed": 0,
            "dataset_root": ".",
            "environment": {},
            "run_fingerprint": "run",
        }
        with patch(
            "research.studies.neighborhood.natural_distribution_confirmation._make_environment",
            return_value=FakeEvaluationEnvironment(),
        ):
            first = _evaluation_trial_worker(job)
            second = _evaluation_trial_worker(job)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["outcome"], second["outcome"])
        self.assertEqual(first["outcome"]["actual_neighborhood"], [0, 1])
        self.assertTrue(first["outcome"]["evaluation_seed_disjoint"])

    def test_timeout_keeps_trial_identity_and_aggregation_reports_it(self) -> None:
        state = make_state()
        state_row = {
            "state_id": "state",
            "episode_id": "episode",
            "state_fingerprint": state_fingerprint(state),
            "solver_seed": 0,
            "split": "confirmation_v2",
            "map_id": "map",
            "task_id": "task",
            "candidates": [
                {
                    "candidate_id": "candidate",
                    "agents": [0, 1],
                    "selection_families": ["target:4"],
                    "proposal_seeds": [10],
                }
            ],
        }
        base_job = {
            "job_id": "state__candidate__trial_0001",
            "state_id": "state",
            "candidate_id": "candidate",
            "evaluation_trial_index": 1,
            "row": {
                "split": "confirmation_v2",
                "map_id": "map",
                "task_id": "task",
                "agent_count": 4,
            },
            "solver_seed": 0,
        }
        timeout = _failed_job_result(base_job, "timeout", "too slow")
        self.assertEqual(timeout["job_id"], base_job["job_id"])
        self.assertEqual(timeout["candidate_id"], "candidate")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            timeout = _normalize_trial_result(timeout, base_job, root, "run")
            successful = {
                "status": "ok",
                "complete": True,
                "state_id": "state",
                "candidate_id": "candidate",
                "evaluation_trial_index": 0,
                "outcome": {"state_id": "state", "candidate_id": "candidate"},
            }
            manifests = _aggregate_trial_results(
                root, state_rows=[state_row],
                trial_rows=[successful, timeout],
                expected_trials=2,
                run_fingerprint="run",
            )
            self.assertEqual(manifests[0]["status"], "error")
            self.assertEqual(manifests[0]["error_count"], 1)
            errors = [
                json.loads(line)
                for line in (root / manifests[0]["errors_file"]).read_text().splitlines()
            ]
            self.assertEqual(errors[0]["status"], "timeout")

    def test_acceptance_uses_fraction_of_active_maps(self) -> None:
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
        report = natural_acceptance(
            summaries,
            {
                "pareto_top1_gain": 0.1,
                "relative_conflict_regret_reduction": 0.2,
                "maps_no_worse": 7,
                "map_count": 10,
            },
            {
                "hit_gain_95_ci": [-0.1, 0.2],
                "conflict_improvement_95_ci": [-0.1, 0.2],
            },
            {"multiple_sizes_supported": True},
            {
                "minimum_top1_gain": 0.05,
                "minimum_conflict_regret_reduction": 0.05,
                "minimum_map_fraction_no_worse": 2 / 3,
                "maximum_size_share": 0.8,
            },
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["minimum_maps_no_worse"], 7)


if __name__ == "__main__":
    unittest.main()
