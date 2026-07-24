from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.parallel_runtime import (
    candidate_lane_counts,
    isolated_lane_cpu_sets,
    physical_cpu_sets,
    select_parallel_lane_count,
)
from experiments.feature_schema_v3 import V3_FEATURE_NAMES
from experiments.repair_collection import _write_json
from experiments.v3_horizon import (
    _coverage,
    _horizon_environment_configuration,
    _horizon_trial,
    _reuse_completed_states,
    select_horizon_candidates,
)
from experiments.v3_horizon_training import _h3_rows


class _OldV3:
    thresholds = {
        "effective_probability_tolerance": 0.1,
        "no_progress_probability_tolerance": 0.1,
        "conflict_reduction_retention": 0.9,
    }

    def predict(self, rows):
        count = len(rows)
        utility = [float(index) for index in range(count)]
        return {
            "effective_progress_probability": [0.9] * count,
            "no_progress_probability": [0.1] * count,
            "conflict_reduction": [4.0] * count,
            "repair_seconds": [1.0] * count,
            "utility": utility,
        }


class HorizonCandidateSamplingTests(unittest.TestCase):
    def test_horizon_extends_late_source_iteration_budget(self) -> None:
        configuration = _horizon_environment_configuration(
            {"max_repair_iterations": 30}, prefix_length=29, horizon=3
        )
        self.assertEqual(configuration["max_repair_iterations"], 32)

    def test_completed_states_can_seed_a_corrected_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            relative = Path("states") / "policy_train" / "state.json"
            _write_json(source / "run_config.json", {"run_fingerprint": "run"})
            _write_json(
                source / relative,
                {"run_fingerprint": "run", "complete": True},
            )
            jobs = [{"state_file": str(output / relative)}]
            reused = _reuse_completed_states(source, output, jobs, "run")
            self.assertEqual(reused, 1)
            self.assertTrue((output / relative).is_file())

    def test_sampling_upgrade_reuses_only_states_containing_v2_winner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            common = {"schema": "collection", "horizon": 3}
            _write_json(
                source / "run_config.json",
                {
                    **common,
                    "candidate_sampling": "top2-v2-per-size-plus-v3-winner-and-adaptive",
                    "run_fingerprint": "old",
                },
            )
            _write_json(
                output / "run_config.json",
                {
                    **common,
                    "candidate_sampling": "top2-v2-per-size-plus-v2-winner-v3-winner-and-adaptive",
                    "run_fingerprint": "new",
                },
            )
            jobs = []
            for state_id, selected in (("kept", ["base"]), ("rerun", ["other"])):
                relative = Path("states") / f"{state_id}.json"
                _write_json(
                    source / relative,
                    {
                        "run_fingerprint": "old",
                        "complete": True,
                        "state_id": state_id,
                        "selected_candidate_ids": selected,
                    },
                )
                jobs.append({"state_file": str(output / relative)})
            reused = _reuse_completed_states(
                source,
                output,
                jobs,
                "new",
                {"kept": "base", "rerun": "base"},
            )
            self.assertEqual(reused, 1)
            self.assertEqual(
                json.loads((output / "states" / "kept.json").read_text())[
                    "run_fingerprint"
                ],
                "new",
            )
            self.assertFalse((output / "states" / "rerun.json").exists())

    def test_sampling_keeps_top_two_per_size_and_old_v3_winner(self) -> None:
        candidates = [
            {"candidate_id": f"c{size}-{index}", "actual_size": size}
            for size in (4, 8, 16)
            for index in range(3)
        ]
        scores = [3.0, 2.0, 1.0] * 3
        selected = select_horizon_candidates(
            candidates, [{} for _ in candidates], scores, _OldV3()
        )
        selected_ids = {candidates[index]["candidate_id"] for index in selected}
        self.assertTrue({"c4-0", "c4-1", "c8-0", "c8-1", "c16-0", "c16-1"} <= selected_ids)
        self.assertIn("c16-2", selected_ids)

    def test_sampling_always_keeps_global_v2_winner_with_fallback_size(self) -> None:
        candidates = [
            {"candidate_id": "size4", "actual_size": 4},
            {"candidate_id": "fallback15", "actual_size": 15},
            {"candidate_id": "size16", "actual_size": 16},
        ]
        selected = select_horizon_candidates(
            candidates, [{} for _ in candidates], [1.0, 10.0, 2.0], _OldV3()
        )
        self.assertIn(1, selected)

    def test_coverage_rejects_missing_adaptive(self) -> None:
        rows = [
            {
                "state_id": "state",
                "candidate_id": f"c{index}",
                "trial_index": trial,
                "horizon": 3,
                "executed_steps": 3,
            }
            for index in range(3)
            for trial in range(2)
        ]
        report = _coverage([{"state_id": "state"}], rows, 2)
        self.assertFalse(report["passed"])
        self.assertTrue(any("Adaptive" in value for value in report["errors"]))

    def test_horizon_trial_stops_when_second_repair_becomes_feasible(self) -> None:
        initial = {
            "num_of_colliding_pairs": 10,
            "feasible": False,
            "done": False,
        }
        observations = [
            {
                "observation": {
                    "num_of_colliding_pairs": 8,
                    "feasible": False,
                    "done": False,
                },
                "metrics": {"replan_success": True, "pp_replan_seconds": 0.3},
            },
            {
                "observation": {
                    "num_of_colliding_pairs": 0,
                    "feasible": True,
                    "done": True,
                },
                "metrics": {"replan_success": True, "pp_replan_seconds": 0.4},
            },
        ]

        class Environment:
            def step(self, _action):
                return observations.pop(0)

        decision = {
            "split": "policy_train",
            "state_id": "state",
            "prefix_actions": [],
            "before_fingerprint": "full-10",
            "before_repair_fingerprint": "repair-10",
            "task_id": "task",
            "solver_seed": 1,
            "decision_index": 2,
        }
        candidate = {
            "candidate_id": "candidate",
            "route": "model",
            "agents": [1, 2, 3, 4],
            "actual_size": 4,
        }

        def repair_fingerprint(state):
            return f"repair-{state['num_of_colliding_pairs']}"

        with (
            patch(
                "experiments.v3_horizon.replay_prefix",
                return_value=(Environment(), initial),
            ),
            patch(
                "experiments.v3_horizon.state_fingerprint",
                return_value="full-10",
            ),
            patch(
                "experiments.v3_horizon.repair_structure_fingerprint",
                side_effect=repair_fingerprint,
            ),
            patch(
                "experiments.v3_horizon._low_level_delta", return_value={}
            ),
            patch(
                "experiments.v3_horizon._v2_action",
                return_value=(
                    {"mode": "explicit_neighborhood", "agents": [5, 6]},
                    {
                        "controller_seconds": 0.2,
                        "candidate_id": "continuation",
                        "actual_size": 8,
                    },
                ),
            ),
        ):
            row = _horizon_trial(
                {},
                decision,
                candidate,
                trial_index=0,
                horizon=3,
                first_selection_seconds=0.1,
                proposal={},
                main_model=object(),
            )
        self.assertEqual(row["executed_steps"], 2)
        self.assertEqual(row["conflict_trajectory"], [10, 8, 0])
        self.assertEqual(row["h3"]["conflict_reduction"], 10)
        self.assertAlmostEqual(row["h3"]["pp_replan_seconds"], 0.7)
        self.assertAlmostEqual(row["h3"]["controller_seconds"], 0.3)
        self.assertTrue(row["h3"]["feasible"])
        self.assertEqual(row["route"], "model")
        self.assertEqual(row["actual_size"], 4)
        self.assertEqual(row["steps"][0]["action"]["pp_random_seed"], row["steps"][0]["action"]["random_seed"])

    def test_adaptive_reference_does_not_require_candidate_features(self) -> None:
        feature = {
            "split": "policy_train",
            "state_id": "state",
            "candidate_id": "model",
            "map_id": "map",
            "layout_mode": "layout",
            "agent_count": 400,
            "route": "model",
            "actual_size": 8,
            "base_selected": True,
            "main_score": 1.0,
            "features": {
                "realized_dynamic": {name: 0.0 for name in V3_FEATURE_NAMES}
            },
        }

        def horizon(candidate_id: str, route: str, size: int) -> dict:
            return {
                "split": "policy_train",
                "state_id": "state",
                "candidate_id": candidate_id,
                "route": route,
                "actual_size": size,
                "h1": {
                    "effective_progress": True,
                    "no_progress": False,
                    "conflict_reduction": 2,
                    "pp_replan_seconds": 0.2,
                },
                "h3": {
                    "conflict_reduction": 4,
                    "total_seconds": 0.5,
                    "no_progress": False,
                },
            }

        rows = _h3_rows(
            [feature],
            [
                horizon("model", "model", 8),
                horizon("official_adaptive", "official_adaptive", 0),
            ],
            "policy_train",
        )
        adaptive = next(row for row in rows if row["route"] == "official_adaptive")
        self.assertEqual(adaptive["actual_size"], 0)
        self.assertFalse(adaptive["base_selected"])
        self.assertEqual(adaptive["map_id"], "map")


class IsolatedParallelRuntimeTests(unittest.TestCase):
    def test_lane_sets_use_distinct_physical_cores(self) -> None:
        available = physical_cpu_sets()
        lanes = isolated_lane_cpu_sets(min(2, len(available)))
        self.assertEqual(len(lanes), min(2, len(available)))
        if len(lanes) == 2:
            self.assertFalse(set(lanes[0]) & set(lanes[1]))

    def test_registered_auto_lanes_never_exceed_available_cores(self) -> None:
        values = candidate_lane_counts()
        self.assertEqual(values[0], 1)
        self.assertLessEqual(values[-1], len(physical_cpu_sets()))

    def test_parallel_audit_chooses_highest_safe_lane_count(self) -> None:
        report = select_parallel_lane_count(
            [
                {
                    "lanes": 4,
                    "strict_pp_seconds": [1.0, 2.0],
                    "parallel_pp_seconds": [1.01, 2.02],
                    "cost_rank_correlation": 1.0,
                    "peak_memory_bytes": 1,
                },
                {
                    "lanes": 8,
                    "strict_pp_seconds": [1.0, 2.0],
                    "parallel_pp_seconds": [1.2, 2.4],
                    "cost_rank_correlation": 1.0,
                    "peak_memory_bytes": 1,
                },
            ]
        )
        self.assertEqual(report["selected_lanes"], 4)

    def test_parallel_audit_rejects_controller_timing_inflation(self) -> None:
        report = select_parallel_lane_count(
            [
                {
                    "lanes": 1,
                    "strict_pp_seconds": [1.0, 2.0],
                    "parallel_pp_seconds": [1.0, 2.0],
                    "strict_controller_seconds": [0.1, 0.2],
                    "parallel_controller_seconds": [0.1, 0.2],
                    "cost_rank_correlation": 1.0,
                    "peak_memory_bytes": 1,
                },
                {
                    "lanes": 4,
                    "strict_pp_seconds": [1.0, 2.0],
                    "parallel_pp_seconds": [1.0, 2.0],
                    "strict_controller_seconds": [0.1, 0.2],
                    "parallel_controller_seconds": [0.12, 0.24],
                    "cost_rank_correlation": 1.0,
                    "peak_memory_bytes": 1,
                },
            ]
        )
        self.assertEqual(report["selected_lanes"], 1)
        lane4 = next(row for row in report["attempts"] if row["lanes"] == 4)
        self.assertFalse(lane4["checks"]["controller_time_ratio"])


if __name__ == "__main__":
    unittest.main()
