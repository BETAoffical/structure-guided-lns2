from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research.engineering.balanced.balanced_controller import (
    BalancedControllerConfig,
    validate_balanced_payload,
)
from experiments.closed_loop_trace_storage import open_trace_text
from research.engineering.balanced.lns2_speed_quality_calibration import (
    _complete_episode_metrics,
    _layout_balanced_folds,
    _passes,
    _split_config,
)
from research.engineering.legacy_tradeoff.lns2_tradeoff import (
    CONTROLLER_ORDER,
    _controller_speed_rows,
    _frontier_rows,
    _promotion_report,
    _route_usage_rows,
    _semantic_equivalence_report,
    generate_tradeoff_artifacts,
)
from research.engineering.balanced.route_counterfactual import (
    ROUTE_COUNTERFACTUAL_SCHEMA,
    ROUTE_COUNTERFACTUAL_VERSION,
    _pareto_relation,
    _valid_checkpoint,
)


def _episode(
    controller: str,
    *,
    task_id: str,
    auc: float,
    wall: float,
    controller_seconds: float,
    soc: int = 100,
) -> dict:
    balanced = controller == "v2-balanced"
    return {
        "controller": controller,
        "episode_id": f"{task_id}__seed_1__{controller}",
        "task_id": task_id,
        "map_id": f"map-{int(task_id.rsplit('-', 1)[-1]) // 4:02d}",
        "layout_family": "maze",
        "agent_count": 100,
        "solver_seed": 1,
        "status": "ok",
        "trace_file": "episode.jsonl.gz",
        "repairable": True,
        "success": True,
        "external_timeout": False,
        "initial_fingerprint": f"initial-{task_id}",
        "initial_conflicts": 10,
        "initial_conflict_density": 0.01,
        "initial_conflict_severity": "low",
        "final_conflicts": 0,
        "repair_iterations": 2,
        "fixed_budget_conflict_auc": auc,
        "normalized_fixed_budget_conflict_auc": auc / 10.0,
        "capped_wall_time_seconds": wall,
        "repair_wall_seconds": wall * 0.8,
        "controller_seconds": controller_seconds,
        "feature_seconds": controller_seconds * 0.7,
        "inference_seconds": controller_seconds * 0.01,
        "final_sum_of_costs": soc,
        "low_level_expanded": 100,
        "low_level_generated": 200,
        "low_level_reopened": 2,
        "invalid_action_count": 0,
        "fingerprint_mismatch_count": 0,
        "learned_decision_count": 2 if controller in {"v1-full", "v2-full"} else 0,
        "shadow_validation_count": 0,
        "shadow_score_max_delta": 0.0,
        "model_decision_count": 1 if balanced else 0,
        "official_decision_count": 1 if balanced else 0,
        "model_route_fraction": 0.5 if balanced else 0.0,
        "route_switch_count": 1 if balanced else 0,
        "candidate_count_before": 18 if balanced else 0,
        "candidate_count_after": 18 if balanced else 0,
        "model_controller_seconds": 0.2 if balanced else 0.0,
        "model_repair_seconds": 0.8 if balanced else 0.0,
        "model_total_decision_seconds": 1.0 if balanced else 0.0,
        "official_controller_seconds": 0.0,
        "official_repair_seconds": 0.8 if balanced else 0.0,
        "official_total_decision_seconds": 0.8 if balanced else 0.0,
    }


def _calibration_row(
    controller: str,
    *,
    task_id: str,
    auc: float,
    wall: float,
    success: bool = True,
) -> dict:
    return {
        "controller": controller,
        "task_id": task_id,
        "map_id": "map-a",
        "layout_mode": "regular_beltway",
        "solver_seed": 1,
        "initial_fingerprint": f"state-{task_id}",
        "repairable": True,
        "success": success,
        "fixed_budget_conflict_auc": auc,
        "capped_wall_time_seconds": wall,
        "final_sum_of_costs": 100,
        "model_decision_count": 1 if controller == "v2-balanced" else 0,
        "official_decision_count": 1 if controller == "v2-balanced" else 0,
    }


class Lns2TradeoffTests(unittest.TestCase):
    @staticmethod
    def _semantic_transition(
        decision_index: int,
        *,
        before: str,
        after: str,
        truncated: bool = False,
        elapsed: float = 1.0,
    ) -> dict:
        return {
            "event": "transition",
            "decision_index": decision_index,
            "before_fingerprint": before,
            "after_fingerprint": after,
            "action": {
                "mode": "explicit_neighborhood",
                "agents": [0, 1],
                "random_seed": 17 + decision_index,
            },
            "low_level_delta": {
                "expanded": 10 + decision_index,
                "generated": 20 + decision_index,
                "reopened": 0,
            },
            "controller": {
                "selected_candidate_id": "candidate-a",
                "candidate_pool": [
                    {
                        "candidate_id": "candidate-a",
                        "retained": True,
                        "score": 1.0,
                    },
                    {
                        "candidate_id": "candidate-b",
                        "retained": True,
                        "score": 0.0,
                    },
                ],
            },
            "terminated": False,
            "truncated": truncated,
            "elapsed_wall_seconds": elapsed,
        }

    @staticmethod
    def _run_semantic_fixture(
        root: Path,
        left: list[dict],
        right: list[dict],
        *,
        left_timeout: bool,
        right_timeout: bool,
    ) -> tuple[dict, list[dict], list[dict]]:
        roots = {
            "v1-full": root / "v1-full",
            "v2-full": root / "v2-full",
        }
        episodes = []
        for controller, transitions, external_timeout in (
            ("v1-full", left, left_timeout),
            ("v2-full", right, right_timeout),
        ):
            collection = roots[controller]
            collection.mkdir(parents=True)
            (collection / "run_config.json").write_text(
                json.dumps(
                    {"configuration": {"wall_time_budget_seconds": 300.0}}
                ),
                encoding="utf-8",
            )
            final_fingerprint = (
                str(transitions[-1]["after_fingerprint"])
                if transitions
                else "initial"
            )
            trace = collection / "episode.jsonl.gz"
            with gzip.open(trace, "wt", encoding="utf-8") as stream:
                for event in (
                    {"event": "initial", "state_fingerprint": "initial"},
                    *transitions,
                    {"event": "finish", "final_fingerprint": final_fingerprint},
                ):
                    stream.write(json.dumps(event) + "\n")
            episodes.append(
                {
                    "controller": controller,
                    "task_id": "task-a",
                    "solver_seed": 1,
                    "status": "ok",
                    "trace_file": trace.name,
                    "external_timeout": external_timeout,
                    "truncated": external_timeout,
                }
            )
        return _semantic_equivalence_report(
            roots, episodes, {("task-a", 1)}
        )

    def test_budget_boundary_excludes_only_incomplete_post_repair_state(self) -> None:
        left = [
            self._semantic_transition(
                0,
                before="initial",
                after="partial-after",
                truncated=True,
                elapsed=300.1,
            )
        ]
        right = [
            self._semantic_transition(0, before="initial", after="complete-after"),
            self._semantic_transition(
                1,
                before="complete-after",
                after="right-partial-after",
                truncated=True,
                elapsed=300.2,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report, mismatches, boundaries = self._run_semantic_fixture(
                Path(directory),
                left,
                right,
                left_timeout=True,
                right_timeout=True,
            )
        self.assertTrue(report["passed"])
        self.assertEqual(report["common_decision_count"], 1)
        self.assertEqual(report["post_repair_comparison_count"], 0)
        self.assertEqual(report["budget_boundary_exclusion_count"], 1)
        self.assertEqual(report["allowed_budget_length_difference_count"], 1)
        self.assertEqual(report["unexplained_length_difference_count"], 0)
        self.assertEqual(mismatches, [])
        self.assertEqual(len(boundaries), 1)

    def test_completed_repair_at_loop_budget_boundary_allows_length_difference(
        self,
    ) -> None:
        left = [
            self._semantic_transition(
                0,
                before="initial",
                after="shared-after",
                elapsed=299.996,
            )
        ]
        right = [
            self._semantic_transition(
                0,
                before="initial",
                after="shared-after",
                elapsed=200.0,
            ),
            self._semantic_transition(
                1,
                before="shared-after",
                after="right-after",
                elapsed=300.2,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report, mismatches, boundaries = self._run_semantic_fixture(
                Path(directory),
                left,
                right,
                left_timeout=True,
                right_timeout=True,
            )
        self.assertTrue(report["passed"])
        self.assertEqual(report["common_decision_count"], 1)
        self.assertEqual(report["post_repair_comparison_count"], 1)
        self.assertEqual(report["budget_boundary_exclusion_count"], 0)
        self.assertEqual(report["allowed_budget_length_difference_count"], 1)
        self.assertEqual(
            report["completed_repair_budget_length_difference_count"], 1
        )
        self.assertEqual(
            report["truncated_repair_budget_length_difference_count"], 0
        )
        self.assertEqual(report["unexplained_length_difference_count"], 0)
        self.assertEqual(mismatches, [])
        self.assertEqual(boundaries, [])

    def test_completed_repair_well_before_budget_does_not_hide_length_difference(
        self,
    ) -> None:
        left = [
            self._semantic_transition(
                0,
                before="initial",
                after="shared-after",
                elapsed=298.0,
            )
        ]
        right = [
            self._semantic_transition(
                0,
                before="initial",
                after="shared-after",
                elapsed=200.0,
            ),
            self._semantic_transition(
                1,
                before="shared-after",
                after="right-after",
                elapsed=300.2,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report, mismatches, _ = self._run_semantic_fixture(
                Path(directory),
                left,
                right,
                left_timeout=True,
                right_timeout=True,
            )
        self.assertFalse(report["passed"])
        self.assertIn(
            "unexplained_length_difference",
            {row["mismatch_kind"] for row in mismatches},
        )

    def test_budget_boundary_still_compares_scores_ranking_and_action(self) -> None:
        def fixture() -> tuple[list[dict], list[dict]]:
            left = [
                self._semantic_transition(
                    0,
                    before="initial",
                    after="partial-after",
                    truncated=True,
                    elapsed=300.1,
                )
            ]
            right = [
                self._semantic_transition(0, before="initial", after="complete-after"),
                self._semantic_transition(
                    1,
                    before="complete-after",
                    after="right-partial-after",
                    truncated=True,
                    elapsed=300.2,
                ),
            ]
            return left, right

        for mismatch_kind in ("score", "ranking", "action"):
            with self.subTest(mismatch_kind=mismatch_kind):
                left, right = fixture()
                if mismatch_kind == "score":
                    right[0]["controller"]["candidate_pool"][0]["score"] = 0.9
                elif mismatch_kind == "ranking":
                    right[0]["controller"]["candidate_pool"][0]["score"] = 0.0
                    right[0]["controller"]["candidate_pool"][1]["score"] = 1.0
                else:
                    right[0]["action"]["agents"] = [0, 2]
                with tempfile.TemporaryDirectory() as directory:
                    report, _, _ = self._run_semantic_fixture(
                        Path(directory),
                        left,
                        right,
                        left_timeout=True,
                        right_timeout=True,
                    )
                self.assertFalse(report["passed"])

    def test_completed_repair_still_compares_after_state_and_search(self) -> None:
        left = [self._semantic_transition(0, before="initial", after="left-after")]
        right = [self._semantic_transition(0, before="initial", after="right-after")]
        right[0]["low_level_delta"]["expanded"] += 1
        with tempfile.TemporaryDirectory() as directory:
            report, mismatches, _ = self._run_semantic_fixture(
                Path(directory),
                left,
                right,
                left_timeout=False,
                right_timeout=False,
            )
        self.assertFalse(report["passed"])
        self.assertEqual(report["post_repair_comparison_count"], 1)
        self.assertEqual(
            {row["mismatch_kind"] for row in mismatches},
            {"after_fingerprint", "low_level_search", "final_fingerprint"},
        )

    def test_nonterminal_truncation_and_unexplained_length_difference_fail(self) -> None:
        cases = []
        nonterminal_left = [
            self._semantic_transition(
                0,
                before="initial",
                after="after-0",
                truncated=True,
                elapsed=10.0,
            ),
            self._semantic_transition(1, before="after-0", after="after-1"),
        ]
        nonterminal_right = [dict(row) for row in nonterminal_left]
        cases.append((nonterminal_left, nonterminal_right, "nonterminal_truncation"))
        short = [self._semantic_transition(0, before="initial", after="after-0")]
        long = [
            self._semantic_transition(0, before="initial", after="after-0"),
            self._semantic_transition(1, before="after-0", after="after-1"),
        ]
        cases.append((short, long, "unexplained_length_difference"))
        for left, right, expected_kind in cases:
            with self.subTest(expected_kind=expected_kind):
                with tempfile.TemporaryDirectory() as directory:
                    report, mismatches, _ = self._run_semantic_fixture(
                        Path(directory),
                        left,
                        right,
                        left_timeout=False,
                        right_timeout=False,
                    )
                self.assertFalse(report["passed"])
                self.assertIn(
                    expected_kind,
                    {row["mismatch_kind"] for row in mismatches},
                )

    def test_balanced_config_routes_and_detects_tampering(self) -> None:
        config = BalancedControllerConfig(
            conflict_threshold=4,
            pruner_threshold=None,
            source={"study_role": "fixture"},
        )
        self.assertEqual(config.route(4), "official_adaptive")
        self.assertEqual(config.route(5), "model")
        self.assertEqual(validate_balanced_payload(config.payload()), config)
        tampered = config.payload()
        tampered["conflict_threshold"] = 8
        with self.assertRaises(ValueError):
            validate_balanced_payload(tampered)

    def test_complete_episode_calibration_metrics_use_full_runs(self) -> None:
        official = [
            _calibration_row(
                "official_adaptive", task_id="task-a", auc=100.0, wall=10.0
            )
        ]
        full = [
            _calibration_row("v2-full", task_id="task-a", auc=80.0, wall=15.0)
        ]
        balanced = [
            _calibration_row(
                "v2-balanced", task_id="task-a", auc=90.0, wall=9.0
            )
        ]
        metrics = _complete_episode_metrics(official, full, balanced)
        self.assertEqual(metrics["episode_count"], 1)
        self.assertAlmostEqual(metrics["auc_gain_retention"], 0.5)
        self.assertAlmostEqual(metrics["speedup_over_full"], 0.4)
        self.assertEqual(metrics["model_decision_count"], 1)
        self.assertEqual(metrics["official_decision_count"], 1)
        self.assertTrue(_passes(metrics))

    def test_training_folds_have_one_map_per_layout(self) -> None:
        rows = []
        for layout in (
            "regular_beltway",
            "compartmentalized",
            "dead_end_aisles",
        ):
            for index in range(4):
                rows.append(
                    {"map_id": f"{layout}-{index}", "layout_mode": layout}
                )
        folds = _layout_balanced_folds(rows)
        self.assertEqual(len(folds), 4)
        self.assertTrue(all(len(value) == 3 for value in folds.values()))

    def test_split_config_converts_registered_two_split_config(self) -> None:
        source = {
            "dataset_design": {
                "tasks_per_map": 4,
                "task_variants": ["a", "b", "c", "d"],
                "layout_counts": {
                    "policy_train": {"regular_beltway": 4},
                    "policy_validation": {"regular_beltway": 2},
                },
            },
            "qualification": {
                "minimum_nonzero_by_split": {
                    "policy_train": 10,
                    "policy_validation": 5,
                },
                "minimum_nonzero_per_layout": {
                    "policy_train": 2,
                    "policy_validation": 1,
                },
                "minimum_active_maps": {
                    "policy_train": 4,
                    "policy_validation": 2,
                },
            },
            "solver_seeds": [1, 2, 3],
            "environment": {},
            "proposal": {},
            "max_decisions": 100,
            "metric_iteration_budget": 100,
            "wall_time_budget_seconds": 300,
            "episode_process_timeout_seconds": 360,
            "frozen_models": "frozen",
            "model_registration": {},
        }
        config = _split_config(source, "policy_train")
        self.assertEqual(config["split"], "policy_train")
        self.assertEqual(config["dataset_design"]["map_count"], 4)
        self.assertEqual(
            config["policies"], ["official_adaptive", "realized_dynamic"]
        )

    def test_route_usage_accounting_and_episode_types(self) -> None:
        episodes = [
            _episode(
                "v2-balanced",
                task_id="task-000",
                auc=90.0,
                wall=9.0,
                controller_seconds=0.2,
            ),
            _episode(
                "v2-balanced",
                task_id="task-001",
                auc=90.0,
                wall=9.0,
                controller_seconds=0.2,
            ),
        ]
        episodes[1]["model_decision_count"] = 0
        episodes[1]["official_decision_count"] = 2
        episodes[1]["model_route_fraction"] = 0.0
        decisions = [
            {
                "episode_id": "a",
                "decision_index": 0,
                "actual_route": "model",
                "decision_phase": "early",
            },
            {
                "episode_id": "a",
                "decision_index": 1,
                "actual_route": "official_adaptive",
                "decision_phase": "late",
            },
        ]
        rows = _route_usage_rows(episodes, decisions)
        overall = next(
            row
            for row in rows
            if row["record_type"] == "aggregate"
            and row["stratum_kind"] == "all"
        )
        self.assertEqual(overall["total_decision_count"], 4)
        self.assertEqual(overall["model_decision_count"], 1)
        kinds = {
            row["stratum_value"]
            for row in rows
            if row["record_type"] == "episode"
        }
        self.assertEqual(kinds, {"mixed", "all_lns2"})

    def test_promotion_uses_four_complete_controller_runs(self) -> None:
        episodes = []
        for task_index in range(24):
            task_id = f"task-{task_index:03d}"
            episodes.extend(
                [
                    _episode(
                        "official_adaptive",
                        task_id=task_id,
                        auc=100.0,
                        wall=10.0,
                        controller_seconds=0.0,
                    ),
                    _episode(
                        "v1-full",
                        task_id=task_id,
                        auc=80.0,
                        wall=15.0,
                        controller_seconds=1.0,
                    ),
                    _episode(
                        "v2-full",
                        task_id=task_id,
                        auc=80.0,
                        wall=12.0,
                        controller_seconds=0.5,
                    ),
                    _episode(
                        "v2-balanced",
                        task_id=task_id,
                        auc=90.0,
                        wall=9.0,
                        controller_seconds=0.2,
                    ),
                ]
            )
        frontier, keys = _frontier_rows(episodes)
        speed = _controller_speed_rows(episodes, keys)
        report = _promotion_report(
            episodes,
            frontier,
            speed,
            keys,
            [
                {
                    "state_count": 10,
                    "equal_map_model_remaining_conflict_improvement": 0.01,
                    "equal_map_model_conflict_delta_advantage": 0.0,
                    "equal_map_model_success_delta": 0.0,
                    "equal_map_model_time_increase": 0.2,
                }
            ],
            {
                "passed": True,
                "error_count": 0,
                "missing_model_result_count": 0,
                "replay_fingerprint_mismatch_count": 0,
            },
            {"passed": True},
            formal=False,
        )
        self.assertEqual(report["conclusion"], "hybrid_supported")
        self.assertTrue(report["v2_full_promotion"]["passed"])
        self.assertTrue(report["v2_balanced_promotion"]["passed"])
        self.assertFalse(report["eligible_to_replace_default"])

    def test_counterfactual_pareto_relation_is_one_repair_only(self) -> None:
        lns2 = {
            "success": False,
            "conflicts_after": 10,
            "sum_of_costs_delta": 2,
            "total_decision_seconds": 2.0,
        }
        model = {
            "success": True,
            "conflicts_after": 5,
            "sum_of_costs_delta": 1,
            "total_decision_seconds": 1.0,
        }
        self.assertEqual(_pareto_relation(lns2, model), "model_dominates")

    def test_checkpoint_requires_official_source_and_one_model_use(self) -> None:
        row = {
            "schema": ROUTE_COUNTERFACTUAL_SCHEMA,
            "schema_version": ROUTE_COUNTERFACTUAL_VERSION,
            "run_fingerprint": "run-a",
            "episode_id": "episode-a",
            "decision_index": 0,
            "before_fingerprint": "state-a",
            "actual_route": "official_adaptive",
            "baseline_source": "balanced-main-trace",
            "actual_lns2": {"outcome": {}},
            "counterfactual_model": {"model_use_count": 1},
            "replay_fingerprint_match": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json.gz"
            with open_trace_text(path, "w") as stream:
                stream.write(json.dumps(row) + "\n")
            self.assertIsNotNone(
                _valid_checkpoint(
                    path,
                    run_fingerprint="run-a",
                    episode_id="episode-a",
                    decision_index=0,
                    before_fingerprint="state-a",
                )
            )
            row["actual_route"] = "model"
            with open_trace_text(path, "w") as stream:
                stream.write(json.dumps(row) + "\n")
            self.assertIsNone(
                _valid_checkpoint(
                    path,
                    run_fingerprint="run-a",
                    episode_id="episode-a",
                    decision_index=0,
                    before_fingerprint="state-a",
                )
            )

    def test_artifact_generation_writes_four_way_tables(self) -> None:
        def controller_event(route: str) -> dict:
            if route == "official_adaptive":
                return {
                    "route": route,
                    "controller_seconds_before_repair": 0.0,
                    "total_decision_seconds": 1.0,
                }
            pool = [
                {
                    "candidate_id": "candidate-a",
                    "retained": True,
                    "score": 1.0,
                },
                {
                    "candidate_id": "candidate-b",
                    "retained": True,
                    "score": 0.0,
                },
            ]
            return {
                "route": route,
                "candidate_pool": pool,
                "selected_candidate_id": "candidate-a",
                "controller_seconds_before_repair": 0.2,
                "total_decision_seconds": 1.0,
            }

        def events(route: str) -> list[dict]:
            action = (
                {"mode": "official"}
                if route == "official_adaptive"
                else {
                    "mode": "explicit_neighborhood",
                    "agents": [0, 1],
                    "random_seed": 7,
                }
            )
            return [
                {
                    "event": "initial",
                    "state_fingerprint": "initial-task-a",
                },
                {
                    "event": "transition",
                    "decision_index": 0,
                    "before_fingerprint": "initial-task-a",
                    "after_fingerprint": "after-task-a",
                    "action": action,
                    "low_level_delta": {
                        "expanded": 10,
                        "generated": 20,
                        "reopened": 1,
                    },
                    "controller": controller_event(route),
                },
                {"event": "finish", "final_fingerprint": "after-task-a"},
            ]

        def summary(controller: str, auc: float, wall: float) -> dict:
            balanced = controller == "v2-balanced"
            full = controller in {"v1-full", "v2-full"}
            return {
                "repairable": True,
                "success": True,
                "external_timeout": False,
                "initial_fingerprint": "initial-task-a",
                "initial_conflicts": 10,
                "final_conflicts": 0,
                "repair_iterations": 1,
                "fixed_budget_conflict_auc": auc,
                "capped_wall_time_to_feasible": wall,
                "repair_wall_seconds": wall * 0.8,
                "final_sum_of_costs": 100,
                "final_low_level": {
                    "expanded": 10,
                    "generated": 20,
                    "reopened": 1,
                },
                "invalid_action_count": 0,
                "fingerprint_mismatch_count": 0,
                "model_decision_count": 0,
                "official_decision_count": 1 if balanced else 0,
                "model_route_fraction": 0.0,
                "route_switch_count": 0,
                "controller_totals": {
                    "learned_decisions": 1 if full else 0,
                    "controller_seconds_before_repair": 0.2 if full else 0.0,
                    "official_decision_count": 1 if balanced else 0,
                    "official_repair_seconds": 0.8 if balanced else 0.0,
                    "official_total_decision_seconds": 0.8 if balanced else 0.0,
                },
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collections = {}
            parameters = {
                "official_adaptive": ("v1-full", 100.0, 10.0, "official_adaptive"),
                "v1-full": ("v1-full", 80.0, 15.0, "model"),
                "v2-full": ("v2-full", 80.0, 12.0, "model"),
                "v2-balanced": ("v2-balanced", 90.0, 9.0, "official_adaptive"),
            }
            for controller in CONTROLLER_ORDER:
                mode, auc, wall, route = parameters[controller]
                collection = root / controller
                collection.mkdir()
                collections[controller] = collection
                run = {
                    "controller": mode,
                    "dataset_fingerprint": "dataset-a",
                    "run_fingerprint": f"run-{controller}",
                    "trace_format": "delta-gzip-v2",
                    "feature_backend": "python",
                    "controller_bundle": (
                        {"main_ranker_semantic_fingerprint": "model-a"}
                        if controller in {"v2-full", "v2-balanced"}
                        else None
                    ),
                    "balanced_config": (
                        {
                            "configuration_fingerprint": "balanced-a",
                            "source": {"selection_unit": "complete_episode"},
                        }
                        if controller == "v2-balanced"
                        else None
                    ),
                    "configuration": {
                        "severity_thresholds": {
                            "low_max": 0.001,
                            "medium_max": 0.01,
                        },
                        "feature_shadow_validation": False,
                    },
                }
                (collection / "run_config.json").write_text(
                    json.dumps(run), encoding="utf-8"
                )
                trace = collection / "episode.jsonl.gz"
                with gzip.open(trace, "wt", encoding="utf-8") as stream:
                    for event in events(route):
                        stream.write(json.dumps(event) + "\n")
                policy = (
                    "official_adaptive"
                    if controller == "official_adaptive"
                    else "realized_dynamic"
                )
                manifest = {
                    "episode_id": f"episode-{controller}",
                    "policy": policy,
                    "layout_mode": "maze",
                    "map_id": "map-a",
                    "task_id": "task-a",
                    "agent_count": 100,
                    "solver_seed": 1,
                    "status": "ok",
                    "summary": summary(controller, auc, wall),
                    "trace_file": trace.name,
                    "trace_bytes": trace.stat().st_size,
                    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                }
                (collection / f"{policy}_manifest.jsonl").write_text(
                    json.dumps(manifest) + "\n", encoding="utf-8"
                )

            counterfactual = root / "counterfactual"
            (counterfactual / "episodes").mkdir(parents=True)
            (counterfactual / "run_config.json").write_text(
                json.dumps(
                    {
                        "source_run_fingerprint": "run-v2-balanced",
                        "scope": "official-routes-model-once",
                    }
                ),
                encoding="utf-8",
            )
            trace = counterfactual / "episodes" / "episode.jsonl.gz"
            lns2_outcome = {
                "conflicts_before": 10,
                "conflicts_after": 6,
                "conflict_delta": 4,
                "success": False,
                "sum_of_costs_delta": 0,
                "low_level_delta": {
                    "expanded": 10,
                    "generated": 20,
                    "reopened": 1,
                },
                "controller_seconds": 0.0,
                "repair_seconds": 1.0,
                "total_decision_seconds": 1.0,
            }
            model_outcome = {
                **lns2_outcome,
                "conflicts_after": 5,
                "conflict_delta": 5,
                "controller_seconds": 0.2,
                "repair_seconds": 1.0,
                "total_decision_seconds": 1.2,
            }
            route_row = {
                "schema": ROUTE_COUNTERFACTUAL_SCHEMA,
                "schema_version": ROUTE_COUNTERFACTUAL_VERSION,
                "episode_id": "episode-v2-balanced",
                "task_id": "task-a",
                "map_id": "map-a",
                "layout_mode": "maze",
                "agent_count": 100,
                "solver_seed": 1,
                "decision_index": 0,
                "actual_route": "official_adaptive",
                "before_fingerprint": "initial-task-a",
                "before_conflicts": 10,
                "baseline_source": "balanced-main-trace",
                "replay_fingerprint_match": True,
                "actual_lns2": {
                    "action": {"mode": "official"},
                    "metrics": {"neighborhood": [0, 1]},
                    "outcome": lns2_outcome,
                },
                "counterfactual_model": {
                    "model_use_count": 1,
                    "action": {
                        "mode": "explicit_neighborhood",
                        "agents": [0, 1],
                    },
                    "controller": {
                        "selected_candidate_id": "candidate-a",
                        "candidate_count_before": 18,
                        "candidate_count_after": 18,
                    },
                    "outcome": model_outcome,
                },
                "pareto_relation": "quality_time_tradeoff",
            }
            with gzip.open(trace, "wt", encoding="utf-8") as stream:
                stream.write(json.dumps(route_row) + "\n")
            (counterfactual / "counterfactual_manifest.jsonl").write_text(
                json.dumps(
                    {
                        "status": "ok",
                        "trace_file": "episodes/episode.jsonl.gz",
                        "trace_bytes": trace.stat().st_size,
                        "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (counterfactual / "counterfactual_summary.json").write_text(
                json.dumps(
                    {
                        "passed": True,
                        "error_count": 0,
                        "counterfactual_state_count": 1,
                        "model_counterfactual_count": 1,
                        "extra_lns2_execution_count": 0,
                        "source_model_route_count": 0,
                        "missing_model_result_count": 0,
                        "replay_fingerprint_mismatch_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            report_root = root / "report"
            report = generate_tradeoff_artifacts(
                collections, counterfactual, report_root, formal=False
            )
            self.assertEqual(report["paired_episode_count"], 1)
            self.assertEqual(report["complete_episode_count"], 4)
            self.assertTrue(report["semantic_equivalence"]["passed"])
            for filename in (
                "paired_episodes.csv",
                "controller_speed_comparison.csv",
                "route_usage.csv",
                "skipped_model_once.csv",
                "quality_speed_frontier.csv",
                "v1_v2_semantic_equivalence.json",
                "v1_v2_budget_boundary_exclusions.csv",
                "hybrid_necessity_report.md",
                "route_usage.svg",
                "counterfactual_pareto.svg",
                "controller_time_comparison.svg",
                "quality_speed_frontier.svg",
            ):
                self.assertTrue((report_root / filename).is_file(), filename)
            self.assertFalse(report["promotion"]["eligible_to_replace_default"])


if __name__ == "__main__":
    unittest.main()
