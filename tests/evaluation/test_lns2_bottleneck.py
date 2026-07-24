from __future__ import annotations

import csv
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.lns2_bottleneck import (
    controller_pairwise_rows,
    controller_pairwise_summary,
    _episode_row,
    _iteration_row,
    _sensitivity_rows,
    _stall_promotion_gate,
    _v3_promotion_gate,
    generate_bottleneck_artifacts,
    long_horizon_diagnostics,
    paired_decomposition,
    stall_prefix_equivalence,
    stall_guard_attempt_limit_violations,
    targeted_stall_recovery_diagnostic,
)
from scripts.run_lns2_tradeoff_evaluation import (
    _merge_lane_collection,
    _rank_correlation,
    _require_native_timing_interface,
    _resolve_parallel_lanes,
    _run_dual_track,
    _run_dual_track_after_validation,
    _run_isolated_parallel_collections,
    _unsolved_job_keys,
    _v3_evaluation_approval,
)


def _event(controller_seconds: float, pp_seconds: float) -> dict:
    native_neighborhood = 0.01
    selection = controller_seconds + native_neighborhood
    return {
        "event": "transition",
        "decision_index": 0,
        "within_wall_budget": True,
        "elapsed_wall_seconds": 1.0,
        "native_timing_schema": "lns2.repair_timing.v1",
        "metrics": {
            "conflicts_before": 10,
            "conflicts_after": 6,
            "neighborhood": [0, 1, 2, 3],
            "replan_success": True,
        },
        "low_level_delta": {
            "expanded": 100,
            "generated": 200,
            "reopened": 2,
            "runs": 4,
        },
        "timings": {
            "controller_before_repair_seconds": controller_seconds,
            "candidate_generation_seconds": controller_seconds * 0.4,
            "state_check_seconds": 0.0,
            "state_analysis_seconds": controller_seconds * 0.2,
            "proposal_feature_seconds": 0.0,
            "realized_feature_seconds": controller_seconds * 0.2,
            "ranking_inference_seconds": controller_seconds * 0.1,
            "selection_residual_seconds": controller_seconds * 0.1,
            "native_neighborhood_generation_seconds": native_neighborhood,
            "neighborhood_selection_seconds": selection,
            "pp_replan_seconds": pp_seconds,
            "repair_bookkeeping_seconds": 0.02,
            "state_export_seconds": 0.03,
            "environment_step_residual_seconds": 0.01,
            "environment_step_wall_seconds": native_neighborhood
            + pp_seconds
            + 0.02
            + 0.03
            + 0.01,
            "pre_step_orchestration_seconds": 0.001,
            "post_step_orchestration_seconds": 0.002,
            "state_fingerprint_seconds": 0.001,
            "iteration_wall_seconds": controller_seconds
            + native_neighborhood
            + pp_seconds
            + 0.063,
        },
    }


def _source(controller: str) -> dict:
    return {
        "episode_id": f"episode-{controller}",
        "task_id": "task-a",
        "map_id": "map-a",
        "layout_mode": "maze",
        "agent_count": 100,
        "solver_seed": 1,
        "status": "ok",
        "episode_finalization_timings": {
            "finish_event_orchestration_seconds": 0.002,
            "finish_trace_write_seconds": 0.005,
            "trace_close_seconds": 0.001,
            "trace_validation_seconds": 0.008,
            "atomic_rename_seconds": 0.001,
            "trace_metadata_seconds": 0.003,
            "post_algorithm_finalize_seconds": 0.02,
            "episode_process_wall_seconds": 1.12,
        },
        "summary": {
            "initial_fingerprint": "same-state",
            "repairable": True,
            "success": True,
            "stop_reason": "success",
            "stopping_rule": "wall-clock",
            "wall_time_budget_seconds": 300.0,
            "initial_conflicts": 10,
            "final_conflicts": 6,
            "budget_final_conflicts": 6,
            "repair_iterations": 1,
            "repair_iterations_within_budget": 1,
            "wall_time_to_feasible": 1.0,
            "capped_wall_time_to_feasible": 1.0,
            "wall_clock_conflict_auc": 8.0,
            "normalized_wall_clock_conflict_auc": 0.8,
            "episode_observed_wall_seconds": 1.1,
            "environment_construct_seconds": 0.01,
            "reset_wall_seconds": 0.02,
            "initial_fingerprint_seconds": 0.0002,
            "final_fingerprint_seconds": 0.0002,
            "reset_timings": {"initial_solution_seconds": 0.01},
            "trace_write_seconds": 0.004,
            "timing_unaccounted_seconds": (
                0.593 if controller == "official_adaptive" else 0.293
            ),
        },
    }


def _write_schedule(root: Path, keys: list[tuple[str, int]]) -> None:
    (root / "execution_schedule.json").write_text(
        json.dumps(
            {
                "schema": "lns2.controller_execution_schedule.v1",
                "method": "test",
                "workers": 1,
                "entries": [
                    {
                        "task_id": task_id,
                        "solver_seed": solver_seed,
                        "controller_order": ["official_adaptive", "v2-full"],
                    }
                    for task_id, solver_seed in keys
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_run_config(
    root: Path,
    *,
    controller: str,
    keys: list[tuple[str, int]],
    explicit_cohort: bool = False,
) -> None:
    task_ids = list(dict.fromkeys(task_id for task_id, _seed in keys))
    solver_seeds = list(dict.fromkeys(seed for _task_id, seed in keys))
    (root / "run_config.json").write_text(
        json.dumps(
            {
                "run_fingerprint": f"run-{controller}",
                "dataset_fingerprint": "dataset",
                "configuration": {
                    "stopping_rule": "wall-clock",
                    "task_ids_override": task_ids,
                    "solver_seeds": solver_seeds,
                    "cohort_job_keys_override": (
                        [[task_id, seed] for task_id, seed in keys]
                        if explicit_cohort
                        else None
                    ),
                    "environment": {
                        "replan_algorithm": "PP",
                        "use_sipp": True,
                    },
                },
                "controller_implementation": {
                    "native_module": {"sha256": "native-module"}
                },
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(
    root: Path,
    *,
    controller: str,
    keys: list[tuple[str, int]],
) -> None:
    manifest_name = (
        "official_adaptive_manifest.jsonl"
        if controller == "official_adaptive"
        else "realized_dynamic_manifest.jsonl"
    )
    sources = []
    for task_id, solver_seed in keys:
        source = _source(controller)
        source.update(
            {
                "episode_id": f"episode-{controller}-{task_id}-{solver_seed}",
                "task_id": task_id,
                "solver_seed": solver_seed,
                "trace_file": "episode.jsonl",
            }
        )
        source["summary"]["transition_trace_write_seconds"] = [0.004]
        sources.append(source)
    (root / manifest_name).write_text(
        "".join(json.dumps(source) + "\n" for source in sources),
        encoding="utf-8",
    )
    if sources:
        (root / "episode.jsonl").write_text(
            "\n".join(
                json.dumps(row)
                for row in (
                    {"event": "initial"},
                    _event(
                        0.0 if controller == "official_adaptive" else 0.2,
                        0.4 if controller == "official_adaptive" else 0.5,
                    ),
                    {"event": "finish"},
                )
            )
            + "\n",
            encoding="utf-8",
        )


class Lns2BottleneckTests(unittest.TestCase):
    def test_parallel_helpers_do_not_split_dual_track_entrypoint(self) -> None:
        entrypoint = inspect.getsource(_run_dual_track)
        continuation = inspect.getsource(_run_dual_track_after_validation)
        parallel = inspect.getsource(_run_isolated_parallel_collections)
        self.assertIn("return _run_dual_track_after_validation", entrypoint)
        self.assertIn("repair_aware_config", continuation)
        self.assertNotIn("repair_aware_config =", parallel)

    def test_parallel_lane_resolution_keeps_strict_single_worker(self) -> None:
        self.assertEqual(_resolve_parallel_lanes("strict", "auto"), 1)
        with patch(
            "experiments.parallel_runtime.physical_cpu_sets", return_value=[(0,)]
        ):
            self.assertEqual(
                _resolve_parallel_lanes("isolated-parallel", "auto"), 1
            )

    def test_parallel_rank_correlation_detects_reversal(self) -> None:
        self.assertAlmostEqual(_rank_correlation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)
        self.assertAlmostEqual(_rank_correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]), -1.0)

    def test_parallel_lane_merge_preserves_unique_episode_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical"
            lane0 = root / "lane0"
            lane1 = root / "lane1"
            for lane, seed in ((lane0, 1), (lane1, 2)):
                (lane / "episodes").mkdir(parents=True)
                (lane / "episodes" / f"episode-{seed}.json").write_text(
                    str(seed), encoding="utf-8"
                )
                (lane / "realized_dynamic_manifest.jsonl").write_text(
                    json.dumps(
                        {
                            "task_id": "task",
                            "solver_seed": seed,
                            "trace_sha256": f"sha-{seed}",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            canonical.mkdir()
            _merge_lane_collection(
                canonical, [lane0, lane1], "realized_dynamic"
            )
            rows = [
                json.loads(value)
                for value in (canonical / "realized_dynamic_manifest.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["solver_seed"] for row in rows], [1, 2])
            self.assertTrue((canonical / "episodes" / "episode-1.json").is_file())
            self.assertTrue((canonical / "episodes" / "episode-2.json").is_file())

    def test_v3_promotion_enforces_quality_speed_and_fallback_gates(self) -> None:
        wall = {
            "official_adaptive": {
                "success_count": 6,
                "mean_normalized_wall_clock_conflict_auc": 0.50,
                "mean_iteration_selection_seconds": 0.01,
                "total_repair_iterations": 60,
                "no_improvement_repair_count": 20,
                "mean_longest_failed_replan_streak": 4.0,
            },
            "v2-full": {
                "success_count": 6,
                "mean_normalized_wall_clock_conflict_auc": 0.45,
                "mean_iteration_selection_seconds": 0.10,
                "total_repair_iterations": 60,
                "no_improvement_repair_count": 18,
                "mean_longest_failed_replan_streak": 3.0,
            },
            "v3-full": {
                "success_count": 6,
                "mean_normalized_wall_clock_conflict_auc": 0.44,
                "mean_iteration_selection_seconds": 0.104,
                "total_repair_iterations": 60,
                "v3_no_progress_count": 10,
                "v3_adaptive_fallback_decision_count": 2,
                "v3_cache_hit_count": 10,
                "v3_rescued_state_count": 3,
                "mean_v3_longest_unchanged_streak": 2.0,
            },
        }
        summaries = [
            {
                "track": "wall-clock-300",
                "controller": controller,
                "group_type": "all",
                **values,
            }
            for controller, values in wall.items()
        ] + [
            {
                "track": "historical",
                "controller": "v2-full",
                "group_type": "all",
                "mean_normalized_fixed_budget_conflict_auc": 0.40,
            },
            {
                "track": "historical",
                "controller": "v3-full",
                "group_type": "all",
                "mean_normalized_fixed_budget_conflict_auc": 0.405,
            },
        ]
        pairwise = []
        for map_index in range(6):
            for reference, reference_auc, reference_ttf in (
                ("v2-full", 0.45, 10.5),
                ("official_adaptive", 0.50, 10.0),
            ):
                pairwise.append(
                    {
                        "track": "wall-clock-300",
                        "candidate": "v3-full",
                        "reference": reference,
                        "map_id": f"map-{map_index}",
                        "common_success": True,
                        "reference_normalized_wall_clock_conflict_auc": reference_auc,
                        "candidate_normalized_wall_clock_conflict_auc": 0.44,
                        "reference_restricted_time_to_feasible": reference_ttf,
                        "candidate_restricted_time_to_feasible": 10.4,
                    }
                )
        gate = _v3_promotion_gate(
            summaries,
            pairwise,
            primary_track="wall-clock-300",
            validation_passed=True,
        )
        self.assertTrue(gate["passed"], gate)
        self.assertLessEqual(
            gate["auc_map_bootstrap"]["v2-full"]["one_sided_95_upper"],
            0.02,
        )
        self.assertLessEqual(gate["adaptive_fallback_fraction"], 0.05)

        h3_summaries = [
            {
                **row,
                "controller": (
                    "v3-h3" if row["controller"] == "v3-full" else row["controller"]
                ),
            }
            for row in summaries
        ]
        h3_pairwise = [
            {
                **row,
                "candidate": (
                    "v3-h3" if row["candidate"] == "v3-full" else row["candidate"]
                ),
            }
            for row in pairwise
        ]
        h3_gate = _v3_promotion_gate(
            h3_summaries,
            h3_pairwise,
            primary_track="wall-clock-300",
            validation_passed=True,
            candidate_controller="v3-h3",
        )
        self.assertTrue(h3_gate["passed"], h3_gate)
        self.assertEqual(h3_gate["candidate_controller"], "v3-h3")

    def test_stall_guard_attempt_limit_and_target_recovery(self) -> None:
        guarded = [
            {
                "track": "wall-clock-600",
                "controller": "v2-stall-safe",
                "task_id": "maze-128-128-1__random_04__agents_0600",
                "solver_seed": 2,
                "decision_index": index,
                "route": "model",
                "stall_guard_state_anchor_fingerprint": "same-state",
                "stall_guard_active_size_cap": 16,
                "elapsed_wall_seconds": 100.0 + index,
                "replan_success": False,
                "conflict_delta": 0 if index < 2 else 3,
            }
            for index in range(3)
        ]
        violations = stall_guard_attempt_limit_violations(guarded)
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["attempt_count"], 3)

        full = [
            {
                **guarded[0],
                "controller": "v2-full",
                "decision_index": index,
                "elapsed_wall_seconds": 90.0 + index,
                "conflict_delta": 0,
            }
            for index in range(3)
        ]
        episodes = [
            {
                "track": "wall-clock-600",
                "controller": "v2-full",
                "task_id": "maze-128-128-1__random_04__agents_0600",
                "solver_seed": 2,
                "budget_final_conflicts": 8821,
            },
            {
                "track": "wall-clock-600",
                "controller": "v2-stall-safe",
                "task_id": "maze-128-128-1__random_04__agents_0600",
                "solver_seed": 2,
                "budget_final_conflicts": 8818,
            },
        ]
        diagnostic = targeted_stall_recovery_diagnostic(
            episodes,
            full + guarded,
            primary_track="wall-clock-600",
        )
        self.assertTrue(diagnostic["passed"], diagnostic)

    def test_stall_prefix_equivalence_stops_at_first_override(self) -> None:
        rows = []
        for controller in ("v2-full", "v2-stall-safe"):
            for decision in range(3):
                rows.append(
                    {
                        "track": "wall-clock-600",
                        "controller": controller,
                        "task_id": "maze600",
                        "solver_seed": 2,
                        "decision_index": decision,
                        "before_fingerprint": f"state-{decision}",
                        "candidate_score_fingerprint": f"scores-{decision}",
                        "candidate_ranking_fingerprint": f"ranking-{decision}",
                        "selected_candidate_id": f"candidate-{decision}",
                        "actual_neighborhood_fingerprint": f"action-{decision}",
                        "route": "model",
                        "stall_guard_base_selection_preserved": (
                            decision < 2 if controller == "v2-stall-safe" else None
                        ),
                    }
                )
        result = stall_prefix_equivalence(rows)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["comparison_count"], 2)
        self.assertEqual(result["trigger_count"], 1)

        rows[-5]["candidate_score_fingerprint"] = "different"
        mismatch = stall_prefix_equivalence(rows)
        self.assertFalse(mismatch["passed"])
        self.assertEqual(mismatch["mismatch_count"], 1)

    def test_stall_promotion_requires_repeated_failure_reduction(self) -> None:
        summaries = []
        wall_values = {
            "official_adaptive": (2, 0.9, 0.2, 2.0, 0.0, 2.0),
            "v2-full": (2, 0.8, 0.3, 5.0, 1.0, 0.0),
            "v2-stall-safe": (3, 0.7, 0.1, 1.0, 1.01, 0.01),
        }
        for controller, (
            successes,
            wall_auc,
            failure_fraction,
            longest_streak,
            selection_seconds,
            guard_seconds,
        ) in wall_values.items():
            summaries.append(
                {
                    "track": "wall-clock-300",
                    "controller": controller,
                    "group_type": "all",
                    "success_count": successes,
                    "mean_normalized_wall_clock_conflict_auc": wall_auc,
                    "failed_replan_fraction": failure_fraction,
                    "mean_longest_failed_replan_streak": longest_streak,
                    "mean_total_neighborhood_selection_seconds": selection_seconds,
                    "mean_iteration_selection_seconds": selection_seconds,
                    "mean_total_stall_guard_seconds": guard_seconds,
                }
            )
        for controller, fixed_auc in {
            "official_adaptive": 1.0,
            "v2-full": 0.8,
            "v2-stall-safe": 0.81,
        }.items():
            summaries.append(
                {
                    "track": "historical",
                    "controller": controller,
                    "group_type": "all",
                    "mean_normalized_fixed_budget_conflict_auc": fixed_auc,
                }
            )
        pairwise = [
            {
                "track": "wall-clock-300",
                "pair": "v2-stall-safe_vs_v2-full",
                "common_success": True,
                "delta_restricted_time_to_feasible_candidate_minus_reference": 0.4,
                "reference_restricted_time_to_feasible": 10.0,
            }
        ]
        passed = _stall_promotion_gate(
            summaries,
            pairwise,
            primary_track="wall-clock-300",
            validation_passed=True,
        )
        self.assertTrue(passed["passed"], passed)

        summaries[2]["failed_replan_fraction"] = 0.3
        failed = _stall_promotion_gate(
            summaries,
            pairwise,
            primary_track="wall-clock-300",
            validation_passed=True,
        )
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["gates"]["pp_failure_fraction_reduced"])

    def test_long_horizon_marks_progress_for_extension_and_failure_plateau(self) -> None:
        episode = {
            "track": "wall-clock-1800",
            "controller": "v2-stall-safe",
            "task_id": "maze",
            "solver_seed": 1,
            "status": "ok",
            "stopping_rule": "wall-clock",
            "wall_time_budget_seconds": 1800.0,
            "initial_conflicts": 100,
            "success": False,
        }
        iterations = [
            {
                "track": "wall-clock-1800",
                "controller": "v2-stall-safe",
                "task_id": "maze",
                "solver_seed": 1,
                "elapsed_wall_seconds": 1200.0,
                "conflicts_after": 90,
                "replan_success": True,
            },
            {
                "track": "wall-clock-1800",
                "controller": "v2-stall-safe",
                "task_id": "maze",
                "solver_seed": 1,
                "elapsed_wall_seconds": 1700.0,
                "conflicts_after": 80,
                "replan_success": False,
            },
        ]
        checkpoints, diagnostics, extension = long_horizon_diagnostics(
            [episode], iterations
        )
        self.assertEqual([row["checkpoint_seconds"] for row in checkpoints], [300.0, 600.0, 1200.0, 1800.0])
        self.assertTrue(diagnostics[0]["extension_to_3600_recommended"])
        self.assertFalse(diagnostics[0]["plateau"])
        self.assertEqual(extension, [["maze", 1]])

    def test_three_controller_pairwise_summary_covers_all_pairs(self) -> None:
        episodes = []
        for controller, auc, success in (
            ("official_adaptive", 0.30, False),
            ("v2-full", 0.25, True),
            ("v2-stall-safe", 0.20, True),
        ):
            episodes.append(
                {
                    "track": "wall-clock-300",
                    "controller": controller,
                    "task_id": "task",
                    "solver_seed": 1,
                    "status": "ok",
                    "initial_fingerprint": "same",
                    "success": success,
                    "normalized_wall_clock_conflict_auc": auc,
                    "restricted_time_to_feasible": 300.0,
                    "budget_final_conflicts": 5,
                    "budget_final_sum_of_costs": 10,
                    "repair_iterations": 2,
                    "neighborhood_selection_seconds": 1.0,
                    "pp_replan_seconds": 2.0,
                    "iteration_wall_seconds": 3.0,
                    "failed_replan_count": 1,
                }
            )
        rows = controller_pairwise_rows(
            episodes, ("official_adaptive", "v2-full", "v2-stall-safe")
        )
        summaries = controller_pairwise_summary(rows)
        self.assertEqual(len(rows), 3)
        self.assertEqual(len(summaries), 3)
        self.assertEqual(
            {row["pair"] for row in summaries},
            {
                "v2-full_vs_official_adaptive",
                "v2-stall-safe_vs_official_adaptive",
                "v2-stall-safe_vs_v2-full",
            },
        )

    def test_dual_preflight_rejects_a_stale_native_module(self) -> None:
        stale = types.ModuleType("lns2_env")
        stale.repair_timing_schema = "old"
        with patch.dict(sys.modules, {"lns2_env": stale}):
            with self.assertRaisesRegex(RuntimeError, "stale"):
                _require_native_timing_interface()

        current = types.ModuleType("lns2_env")
        current.repair_timing_schema = "lns2.repair_timing.v1"
        current.__file__ = "/tmp/lns2_env.so"

        class Environment:
            def get_last_reset_timings(self) -> dict:
                return {}

        current.LNS2RepairEnv = Environment
        with patch.dict(sys.modules, {"lns2_env": current}):
            self.assertEqual(
                _require_native_timing_interface(),
                "/tmp/lns2_env.so",
            )
            with self.assertRaisesRegex(RuntimeError, "propose_batch_compact"):
                _require_native_timing_interface(require_optimized=True)

        Environment.propose_batch_compact = lambda self, actions: []
        current.batch_online_feature_vectors = lambda *args, **kwargs: {}
        with patch.dict(sys.modules, {"lns2_env": current}):
            self.assertEqual(
                _require_native_timing_interface(require_optimized=True),
                "/tmp/lns2_env.so",
            )

    def test_unpromoted_v3_requires_explicit_diagnostic_and_native_integrity(self) -> None:
        report = {
            "decision": "v3_pilot_failed",
            "pilot_passed": False,
            "native_available": True,
            "native_audit_completed": True,
            "pilot_checks": {
                "portable_parity": True,
                "worst_cell": False,
            },
        }
        with self.assertRaisesRegex(ValueError, "v3_pilot_passed"):
            _v3_evaluation_approval(
                report, allow_unpromoted_diagnostic=False
            )
        approval = _v3_evaluation_approval(
            report, allow_unpromoted_diagnostic=True
        )
        self.assertTrue(approval["unpromoted_diagnostic"])
        self.assertEqual(approval["failed_pilot_checks"], ["worst_cell"])

    def test_sensitivity_selects_a_pair_when_either_controller_is_unsolved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = {
                "official_adaptive": root / "official",
                "v2-full": root / "v2",
            }
            for value in roots.values():
                value.mkdir()
            rows = {
                "official_adaptive": [
                    {"task_id": "task-a", "solver_seed": 1, "status": "ok", "summary": {"success": True}},
                    {"task_id": "task-b", "solver_seed": 1, "status": "ok", "summary": {"success": False}},
                    {"task_id": "task-c", "solver_seed": 1, "status": "ok", "summary": {"success": True}},
                ],
                "v2-full": [
                    {"task_id": "task-a", "solver_seed": 1, "status": "ok", "summary": {"success": False}},
                    {"task_id": "task-b", "solver_seed": 1, "status": "ok", "summary": {"success": True}},
                    {"task_id": "task-c", "solver_seed": 1, "status": "ok", "summary": {"success": True}},
                ],
            }
            (roots["official_adaptive"] / "official_adaptive_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows["official_adaptive"]),
                encoding="utf-8",
            )
            (roots["v2-full"] / "realized_dynamic_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows["v2-full"]),
                encoding="utf-8",
            )
            selected = _unsolved_job_keys(roots)
        self.assertEqual(selected, {("task-a", 1), ("task-b", 1)})

    def test_sensitivity_summary_counts_only_new_successes(self) -> None:
        common = {
            "task_id": "task-a",
            "solver_seed": 1,
            "initial_fingerprint": "same",
            "lns2_budget_final_conflicts": 2,
            "v2_budget_final_conflicts": 1,
        }
        rows = [
            {
                **common,
                "track": "wall-clock-300",
                "lns2_success": False,
                "v2_success": True,
            },
            {
                **common,
                "track": "wall-clock-600",
                "lns2_success": True,
                "v2_success": True,
                "lns2_budget_final_conflicts": 0,
                "v2_budget_final_conflicts": 0,
            },
        ]
        summary = {
            row["controller"]: row
            for row in _sensitivity_rows(rows, "wall-clock-300")
        }
        self.assertEqual(summary["official_adaptive"]["new_success_count"], 1)
        self.assertEqual(summary["v2-full"]["new_success_count"], 0)
        self.assertEqual(summary["v2-full"]["lost_success_count"], 0)
        self.assertEqual(
            summary["official_adaptive"]["initial_fingerprint_mismatch_count"],
            0,
        )

    def test_iteration_and_episode_rows_keep_selection_and_pp_separate(self) -> None:
        event = _event(0.2, 0.5)
        manifest = _source("v2-full")
        iteration = _iteration_row(
            track="wall-clock-300",
            controller="v2-full",
            manifest=manifest,
            event=event,
            trace_write_seconds=0.004,
        )
        self.assertAlmostEqual(iteration["neighborhood_selection_seconds"], 0.21)
        self.assertAlmostEqual(iteration["pp_replan_seconds"], 0.5)
        self.assertTrue(iteration["timing_instrumented"])
        episode = _episode_row(
            track="wall-clock-300",
            controller="v2-full",
            source=manifest,
            iterations=[iteration],
        )
        self.assertTrue(episode["timing_instrumentation_complete"])
        self.assertAlmostEqual(episode["neighborhood_selection_seconds"], 0.21)
        self.assertAlmostEqual(episode["pp_replan_seconds"], 0.5)
        self.assertEqual(episode["repair_low_level_expanded"], 100)
        self.assertAlmostEqual(episode["episode_state_fingerprint_seconds"], 0.0014)
        self.assertAlmostEqual(
            episode["process_timing_closure_error_seconds"], 0.0
        )

    def test_episode_row_reports_fixed_and_normalized_fixed_auc_separately(self) -> None:
        source = _source("v2-full")
        source["summary"].update(
            {
                "fixed_budget_conflict_auc": 250.0,
                "normalized_fixed_budget_conflict_auc": 0.25,
                "metric_iteration_budget": 100,
            }
        )
        row = _episode_row(
            track="historical",
            controller="v2-full",
            source=source,
            iterations=[],
        )
        self.assertEqual(row["fixed_budget_conflict_auc"], 250.0)
        self.assertEqual(row["normalized_fixed_budget_conflict_auc"], 0.25)
        self.assertEqual(row["metric_iteration_budget"], 100)

    def test_paired_decomposition_uses_matching_task_and_seed(self) -> None:
        lns2_manifest = _source("official_adaptive")
        v2_manifest = _source("v2-full")
        lns2_iteration = _iteration_row(
            track="wall-clock-300",
            controller="official_adaptive",
            manifest=lns2_manifest,
            event=_event(0.0, 0.4),
            trace_write_seconds=0.004,
        )
        v2_iteration = _iteration_row(
            track="wall-clock-300",
            controller="v2-full",
            manifest=v2_manifest,
            event=_event(0.2, 0.5),
            trace_write_seconds=0.004,
        )
        episodes = [
            _episode_row(
                track="wall-clock-300",
                controller="official_adaptive",
                source=lns2_manifest,
                iterations=[lns2_iteration],
            ),
            _episode_row(
                track="wall-clock-300",
                controller="v2-full",
                source=v2_manifest,
                iterations=[v2_iteration],
            ),
        ]
        rows = paired_decomposition(episodes)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["initial_fingerprint_match"])
        self.assertAlmostEqual(
            rows[0]["delta_neighborhood_selection_seconds_v2_minus_lns2"],
            0.2,
        )
        self.assertAlmostEqual(
            rows[0]["delta_pp_replan_seconds_v2_minus_lns2"],
            0.1,
        )

    def test_empty_manifests_fail_registered_cohort_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [("task-a", 1)]
            _write_schedule(root, expected)
            roots = {
                controller: root / controller for controller in ("official_adaptive", "v2-full")
            }
            for controller, collection in roots.items():
                collection.mkdir()
                _write_run_config(
                    collection,
                    controller=controller,
                    keys=expected,
                )
                _write_manifest(collection, controller=controller, keys=[])

            report = generate_bottleneck_artifacts(
                {"wall-clock-300": roots}, root / "report"
            )

        validation = report["validation"]
        coverage = validation["track_coverage"]["wall-clock-300"]
        self.assertFalse(validation["passed"])
        self.assertFalse(validation["coverage_passed"])
        self.assertEqual(validation["empty_manifest_controller_count"], 2)
        self.assertEqual(validation["missing_episode_key_count"], 2)
        self.assertEqual(coverage["expected_key_count"], 1)
        self.assertTrue(coverage["controller_key_sets_match"])

    def test_sensitivity_schedule_catches_symmetric_missing_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [("task-a", 1), ("task-b", 1)]
            observed = [("task-a", 1)]
            _write_schedule(root, expected)
            roots = {
                controller: root / controller for controller in ("official_adaptive", "v2-full")
            }
            for controller, collection in roots.items():
                collection.mkdir()
                _write_run_config(
                    collection,
                    controller=controller,
                    keys=expected,
                    explicit_cohort=True,
                )
                _write_manifest(
                    collection,
                    controller=controller,
                    keys=observed,
                )

            report = generate_bottleneck_artifacts(
                {"wall-clock-600": roots}, root / "report"
            )

        validation = report["validation"]
        coverage = validation["track_coverage"]["wall-clock-600"]
        self.assertFalse(validation["passed"])
        self.assertEqual(coverage["expected_source"], "execution_schedule")
        self.assertTrue(coverage["configured_cohorts_match_schedule"])
        self.assertTrue(coverage["controller_key_sets_match"])
        self.assertEqual(validation["missing_episode_key_count"], 2)
        for controller in ("official_adaptive", "v2-full"):
            self.assertEqual(
                coverage["controllers"][controller]["missing_keys"],
                [["task-b", 1]],
            )

    def test_unexpected_duplicate_and_controller_mismatch_fail_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = [("task-a", 1)]
            _write_schedule(root, expected)
            roots = {
                controller: root / controller for controller in ("official_adaptive", "v2-full")
            }
            for controller, collection in roots.items():
                collection.mkdir()
                _write_run_config(
                    collection,
                    controller=controller,
                    keys=expected,
                )
            _write_manifest(
                roots["official_adaptive"],
                controller="official_adaptive",
                keys=[("task-a", 1), ("task-a", 1), ("task-b", 1)],
            )
            _write_manifest(
                roots["v2-full"],
                controller="v2-full",
                keys=expected,
            )

            report = generate_bottleneck_artifacts(
                {"wall-clock-300": roots}, root / "report"
            )

        validation = report["validation"]
        coverage = validation["track_coverage"]["wall-clock-300"]
        self.assertFalse(validation["passed"])
        self.assertFalse(coverage["controller_key_sets_match"])
        self.assertEqual(validation["unexpected_episode_key_count"], 1)
        self.assertEqual(validation["duplicate_episode_key_count"], 1)

    def test_artifact_generation_writes_all_bottleneck_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_schedule(root, [("task-a", 1)])
            roots = {}
            for controller, controller_seconds, pp_seconds in (
                ("official_adaptive", 0.0, 0.4),
                ("v2-full", 0.2, 0.5),
            ):
                collection = root / controller
                collection.mkdir()
                roots[controller] = collection
                (collection / "run_config.json").write_text(
                    json.dumps(
                        {
                            "run_fingerprint": f"run-{controller}",
                            "dataset_fingerprint": "dataset",
                            "configuration": {
                                "stopping_rule": "wall-clock",
                                "task_ids_override": ["task-a"],
                                "solver_seeds": [1],
                                "cohort_job_keys_override": None,
                                "environment": {
                                    "replan_algorithm": "PP",
                                    "use_sipp": True,
                                },
                            },
                            "controller_implementation": {
                                "native_module": {"sha256": "native-module"}
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                source = _source(controller)
                source["trace_file"] = "episode.jsonl"
                source["summary"]["transition_trace_write_seconds"] = [0.004]
                manifest_name = (
                    "official_adaptive_manifest.jsonl"
                    if controller == "official_adaptive"
                    else "realized_dynamic_manifest.jsonl"
                )
                (collection / manifest_name).write_text(
                    json.dumps(source) + "\n", encoding="utf-8"
                )
                (collection / "episode.jsonl").write_text(
                    "\n".join(
                        json.dumps(row)
                        for row in (
                            {"event": "initial"},
                            _event(controller_seconds, pp_seconds),
                            {"event": "finish"},
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
            report = generate_bottleneck_artifacts(
                {"wall-clock-300": roots}, root / "report"
            )
            self.assertTrue(report["validation"]["passed"])
            for name in (
                "iteration_timings.csv",
                "episode_timing_breakdown.csv",
                "paired_bottleneck_decomposition.csv",
                "controller_pairwise_episodes.csv",
                "controller_pairwise_summary.csv",
                "stall_guard_usage.csv",
                "stall_prefix_mismatches.csv",
                "stall_prefix_equivalence.json",
                "stall_guard_attempt_limit_violations.csv",
                "targeted_stall_recovery.json",
                "long_horizon_checkpoints.csv",
                "long_horizon_diagnostics.csv",
                "timing_summary.csv",
                "neighborhood_pp_summary.csv",
                "wall_clock_sensitivity.csv",
                "timing_breakdown.svg",
                "loop_count_and_time.svg",
                "neighborhood_size_vs_pp.svg",
                "conflicts_over_wall_time.svg",
                "v2_bottleneck_report.md",
                "stall_recovery_report.md",
            ):
                self.assertTrue((root / "report" / name).is_file(), name)
            with (root / "report" / "timing_summary.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                group_types = {row["group_type"] for row in csv.DictReader(stream)}
            self.assertTrue(
                {"all", "map_id", "layout_family", "agent_count", "common_success"}.issubset(
                    group_types
                )
            )


if __name__ == "__main__":
    unittest.main()
