from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from generators.candidate_retrieval import (
    build_candidate_index,
    evaluate_candidate_retrieval,
)
from generators.candidate_guided_solver import CandidateGuide, RankerCandidateGuide
from generators.candidate_diagnostics import build_candidate_diagnostics
from generators.candidate_experience import _aggregate_order_outcomes
from generators.candidate_ranker import (
    evaluate_candidate_ranker,
    train_candidate_ranker,
)
from generators.experience import _result_from_trace
from generators.rollout_experience import _aggregate_rollouts


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _cases(split: str, state_count: int) -> list[dict]:
    rows = []
    for state in range(state_count):
        for candidate in range(8):
            valid = candidate != 7
            conflict_reduction = (
                2 if candidate == (state % 7) else 0
            )
            features = {
                "map.shelf_coverage": 0.3 + 0.01 * state,
                "map.free_cell_ratio": 0.8 - 0.01 * state,
                "task.agent_count": 36.0 + state,
                "state.conflict_density": 0.1 + 0.01 * state,
                "state.vertex_ratio": 1.0,
                "candidate.conflict_edge_coverage": candidate / 7.0,
                "candidate.mean_conflict_degree": 0.2 * candidate,
                "candidate.mean_path_stretch": (
                    1.0 + 0.05 * candidate
                ),
            }
            rows.append(
                {
                    "schema_version": 1,
                    "usage": (
                        "memory" if split == "train" else "evaluation"
                    ),
                    "split": split,
                    "case_id": (
                        f"{split}_state_{state:02d}"
                        f"__candidate_{candidate:02d}"
                    ),
                    "state_id": f"{split}_state_{state:02d}",
                    "run_id": f"{split}_run_{state:02d}",
                    "map_id": f"{split}_map_{state // 2:02d}",
                    "task_id": f"{split}_task_{state:02d}",
                    "solver_seed": 1,
                    "candidate_index": candidate,
                    "generator": f"generator_{candidate}",
                    "features": features,
                    "outcome": {
                        "candidate_valid": valid,
                        "conflict_reduction": (
                            conflict_reduction if valid else None
                        ),
                        "cost_improvement": (
                            candidate if valid else None
                        ),
                        "replan_runtime_ms": 10.0 + candidate,
                        "total_runtime_ms": 11.0 + candidate,
                    },
                }
            )
    return rows


class CandidateRetrievalTests(unittest.TestCase):
    def test_trace_resume_rejects_candidate_profile_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            _write_jsonl(
                trace,
                [
                    {
                        "schema_version": 6,
                        "event_type": "summary",
                        "success": True,
                        "initial_conflicting_pairs": 1,
                        "final_conflicting_pairs": 0,
                        "iterations": 1,
                        "accepted_iterations": 1,
                        "makespan": 4,
                        "sum_of_costs": 8,
                        "runtime_ms": 1.0,
                        "search_runtime_ms": 1.0,
                        "guidance_runtime_ms": 0.0,
                        "counterfactual_runtime_ms": 0.0,
                        "guidance_requests": 0,
                        "guidance_used": 0,
                        "guidance_fallbacks": 0,
                        "candidate_generator_profile": "full8",
                        "candidate_replan_order_seeds": [0, 1, 2],
                        "candidate_rollout_horizons": [10, 25, 50],
                    }
                ],
            )

            self.assertIsNotNone(
                _result_from_trace(
                    trace,
                    expected_candidate_generator_profile="full8",
                    expected_replan_order_seeds=[0, 1, 2],
                    expected_rollout_horizons=[10, 25, 50],
                )
            )
            self.assertIsNone(
                _result_from_trace(
                    trace,
                    expected_candidate_generator_profile="core5",
                    expected_replan_order_seeds=[0, 1, 2],
                    expected_rollout_horizons=[10, 25, 50],
                )
            )

    def test_train_index_and_validation_tuning_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            queries = root / "queries"
            index = root / "index"
            evaluation = root / "evaluation"
            train_cases = _cases("train", 12)
            validation_cases = _cases("validation", 4)
            _write_jsonl(memory / "candidate_cases.jsonl", train_cases)
            _write_json(
                memory / "candidate_summary.json",
                {"split": "train", "usage": "memory"},
            )
            _write_jsonl(
                queries / "candidate_cases.jsonl", validation_cases
            )
            _write_json(
                queries / "candidate_summary.json",
                {"split": "validation", "usage": "evaluation"},
            )

            index_summary = build_candidate_index(memory, index)
            summary = evaluate_candidate_retrieval(
                index, queries, evaluation
            )
            self.assertEqual(index_summary["case_count"], 96)
            self.assertEqual(summary["index_split"], "train")
            self.assertEqual(summary["query_split"], "validation")
            self.assertFalse(summary["test_data_read"])
            config = json.loads(
                (evaluation / "selected_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(config["selected_on_split"], "validation")
            self.assertFalse(config["test_data_read"])
            self.assertEqual(config["feature_profile"], "full")
            normalizer_text = (
                index / "normalizer.json"
            ).read_text(encoding="utf-8")
            for forbidden in (
                "agent_id",
                "generator_name",
                "absolute_coordinates",
                "post_repair_paths",
            ):
                self.assertNotIn(forbidden, normalizer_text)

            repeat = root / "repeat"
            repeated = evaluate_candidate_retrieval(
                index, queries, repeat
            )
            self.assertEqual(
                summary["selected_parameters"],
                repeated["selected_parameters"],
            )
            self.assertEqual(
                (evaluation / "candidate_guidance.jsonl").read_text(
                    encoding="utf-8"
                ),
                (repeat / "candidate_guidance.jsonl").read_text(
                    encoding="utf-8"
                ),
            )

    def test_wrong_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_jsonl(
                root / "memory" / "candidate_cases.jsonl",
                _cases("validation", 1),
            )
            _write_json(
                root / "memory" / "candidate_summary.json",
                {"split": "validation", "usage": "evaluation"},
            )
            with self.assertRaisesRegex(ValueError, "Train"):
                build_candidate_index(
                    root / "memory", root / "index"
                )

    def test_feature_profile_filters_and_drops_zero_variance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            index = root / "index"
            _write_jsonl(
                memory / "candidate_cases.jsonl", _cases("train", 4)
            )
            _write_json(
                memory / "candidate_summary.json",
                {"split": "train", "usage": "memory"},
            )

            summary = build_candidate_index(
                memory, index, feature_profile="dedup20"
            )
            normalizer = json.loads(
                (index / "normalizer.json").read_text(
                    encoding="utf-8"
                )
            )
            names = [entry["name"] for entry in normalizer["features"]]
            self.assertEqual(summary["feature_profile"], "dedup20")
            self.assertIn("map.shelf_coverage", names)
            self.assertIn("candidate.mean_path_stretch", names)
            self.assertNotIn("map.free_cell_ratio", names)
            self.assertNotIn("candidate.mean_conflict_degree", names)
            self.assertIn(
                "state.vertex_ratio",
                normalizer["zero_variance_features"],
            )

    def test_unknown_feature_profile_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            _write_jsonl(
                memory / "candidate_cases.jsonl", _cases("train", 2)
            )
            _write_json(
                memory / "candidate_summary.json",
                {"split": "train", "usage": "memory"},
            )
            with self.assertRaisesRegex(ValueError, "unknown"):
                build_candidate_index(
                    memory,
                    root / "index",
                    feature_profile="tiny",
                )

    def test_candidate_guide_rejects_profile_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            split_root = dataset / "test"
            split_root.mkdir(parents=True)
            _write_jsonl(
                split_root / "manifest.jsonl",
                [
                    {
                        "task_id": "test_task_000",
                        "map_file": "map.json",
                        "task_file": "task.json",
                    }
                ],
            )
            _write_json(split_root / "map.json", {"rows": 1, "cols": 1})
            _write_json(split_root / "task.json", {"metadata": {}})
            index = root / "index"
            _write_json(
                index / "normalizer.json",
                {
                    "fit_split": "train",
                    "feature_profile": "dedup20",
                    "features": [],
                },
            )
            _write_jsonl(
                index / "candidate_index.jsonl",
                [{"task_id": "train_task_000"}],
            )
            config = root / "selected_config.json"
            _write_json(
                config,
                {
                    "selected_on_split": "validation",
                    "test_data_read": False,
                    "feature_profile": "core12",
                    "k": 3,
                    "group_weights": {},
                    "minimum_margin": 0.0,
                    "ood_distance_threshold": 1.0,
                    "minimum_valid_probability": 0.5,
                },
            )

            with self.assertRaisesRegex(ValueError, "profiles"):
                CandidateGuide(
                    dataset,
                    "test",
                    "test_task_000",
                    index,
                    config,
                )

    def test_order_trial_outcomes_are_aggregated(self) -> None:
        aggregate = _aggregate_order_outcomes(
            [
                {
                    "candidate_valid": True,
                    "conflicting_pairs_before": 5,
                    "conflicting_pairs_after": 3,
                    "conflict_reduction": 2,
                    "sum_of_costs_before": 100,
                    "sum_of_costs_after": 98,
                    "cost_improvement": 2,
                    "replan_runtime_ms": 10.0,
                    "total_runtime_ms": 11.0,
                },
                {
                    "candidate_valid": True,
                    "conflicting_pairs_before": 5,
                    "conflicting_pairs_after": 5,
                    "conflict_reduction": 0,
                    "sum_of_costs_before": 100,
                    "sum_of_costs_after": 99,
                    "cost_improvement": 1,
                    "replan_runtime_ms": 12.0,
                    "total_runtime_ms": 13.0,
                },
                {
                    "candidate_valid": False,
                    "conflicting_pairs_before": 5,
                    "conflicting_pairs_after": -1,
                    "conflict_reduction": None,
                    "sum_of_costs_before": 100,
                    "sum_of_costs_after": -1,
                    "cost_improvement": None,
                    "replan_runtime_ms": 8.0,
                    "total_runtime_ms": 9.0,
                },
            ]
        )
        self.assertTrue(aggregate["candidate_valid"])
        self.assertAlmostEqual(aggregate["valid_probability"], 2 / 3)
        self.assertAlmostEqual(aggregate["conflict_reduction"], 2 / 3)
        self.assertAlmostEqual(
            aggregate["mean_valid_conflict_reduction"], 1.0
        )
        self.assertAlmostEqual(aggregate["cost_improvement"], 1.0)
        self.assertEqual(aggregate["order_trial_count"], 3)

    def test_candidate_diagnostics_reports_oracle_and_order_noise(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            output = root / "diagnostics"
            cases = _cases("train", 3)
            order_cases = []
            for case in cases[:8]:
                for seed in (0, 1):
                    order_cases.append(
                        {
                            **case,
                            "case_id": (
                                f"{case['case_id']}__order_{seed:04d}"
                            ),
                            "order_seed": seed,
                            "outcome": {
                                **case["outcome"],
                                "conflict_reduction": (
                                    (case["outcome"][
                                        "conflict_reduction"
                                    ] or 0)
                                    + seed
                                ),
                            },
                        }
                    )
            _write_jsonl(memory / "candidate_cases.jsonl", cases)
            _write_jsonl(
                memory / "candidate_order_cases.jsonl", order_cases
            )
            _write_json(
                memory / "candidate_summary.json",
                {"split": "train", "usage": "memory"},
            )

            summary = build_candidate_diagnostics(memory, output)
            self.assertEqual(summary["oracle"]["state_count"], 3)
            self.assertGreater(
                summary["oracle"][
                    "states_with_better_alternative_ratio"
                ],
                0.0,
            )
            self.assertGreater(
                summary["order_noise"][
                    "candidate_with_multiple_orders"
                ],
                0,
            )
            self.assertTrue(
                (output / "candidate_diagnostics.md").is_file()
            )

    def test_candidate_ranker_trains_and_tunes_without_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            queries = root / "queries"
            ranker = root / "ranker"
            evaluation = root / "evaluation"
            _write_jsonl(memory / "candidate_cases.jsonl", _cases("train", 8))
            _write_json(
                memory / "candidate_summary.json",
                {"split": "train", "usage": "memory"},
            )
            _write_jsonl(
                queries / "candidate_cases.jsonl",
                _cases("validation", 4),
            )
            _write_json(
                queries / "candidate_summary.json",
                {"split": "validation", "usage": "evaluation"},
            )

            train_summary = train_candidate_ranker(
                memory,
                ranker,
                feature_profile="dedup20",
                models=["pairwise_linear"],
            )
            eval_summary = evaluate_candidate_ranker(
                ranker,
                queries,
                evaluation,
            )
            self.assertEqual(train_summary["fit_split"], "train")
            self.assertFalse(train_summary["test_data_read"])
            self.assertEqual(eval_summary["ranker_split"], "train")
            self.assertEqual(eval_summary["query_split"], "validation")
            self.assertFalse(eval_summary["test_data_read"])
            config = json.loads(
                (evaluation / "selected_config.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(config["guide_type"], "ranker")
            self.assertEqual(config["feature_profile"], "dedup20")
            self.assertEqual(config["model_type"], "pairwise_linear")

            repeat = root / "repeat"
            repeated = evaluate_candidate_ranker(ranker, queries, repeat)
            self.assertEqual(
                eval_summary["selected_parameters"],
                repeated["selected_parameters"],
            )
            self.assertEqual(
                (evaluation / "ranker_guidance.jsonl").read_text(
                    encoding="utf-8"
                ),
                (repeat / "ranker_guidance.jsonl").read_text(
                    encoding="utf-8"
                ),
            )

    def test_candidate_ranker_rejects_non_train_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            memory = root / "memory"
            _write_jsonl(
                memory / "candidate_cases.jsonl", _cases("validation", 1)
            )
            _write_json(
                memory / "candidate_summary.json",
                {"split": "validation", "usage": "evaluation"},
            )
            with self.assertRaisesRegex(ValueError, "Train"):
                train_candidate_ranker(memory, root / "ranker")

    def test_ranker_guide_rejects_profile_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            split_root = dataset / "test"
            split_root.mkdir(parents=True)
            _write_jsonl(
                split_root / "manifest.jsonl",
                [
                    {
                        "task_id": "test_task_000",
                        "map_file": "map.json",
                        "task_file": "task.json",
                    }
                ],
            )
            _write_json(split_root / "map.json", {"rows": 1, "cols": 1})
            _write_json(split_root / "task.json", {"metadata": {}})
            ranker = root / "ranker"
            _write_json(
                ranker / "ranker_summary.json",
                {
                    "fit_split": "train",
                    "feature_profile": "dedup20",
                    "models": [{"model_type": "pairwise_linear"}],
                },
            )
            _write_json(
                ranker / "normalizer.json",
                {
                    "fit_split": "train",
                    "feature_profile": "dedup20",
                    "features": [],
                },
            )
            _write_json(
                ranker / "pairwise_linear.json",
                {
                    "model_type": "pairwise_linear",
                    "weights": [],
                    "bias": 0.0,
                },
            )
            config = root / "selected_config.json"
            _write_json(
                config,
                {
                    "selected_on_split": "validation",
                    "test_data_read": False,
                    "guide_type": "ranker",
                    "feature_profile": "core12",
                    "model_type": "pairwise_linear",
                    "minimum_margin": 0.0,
                },
            )

            with self.assertRaisesRegex(ValueError, "profiles"):
                RankerCandidateGuide(
                    dataset,
                    "test",
                    "test_task_000",
                    ranker,
                    config,
                )

    def test_rollout_outcomes_aggregate_closed_loop_labels(self) -> None:
        aggregated = _aggregate_rollouts(
            [
                {
                    "candidate_valid": True,
                    "conflicting_pairs_before": 5,
                    "sum_of_costs_before": 100,
                    "one_step_conflict_reduction": 1,
                    "one_step_cost_improvement": -2,
                    "rollouts": [
                        {
                            "horizon": 10,
                            "solved": False,
                            "iterations": 10,
                            "accepted_iterations": 3,
                            "conflicting_pairs_after": 2,
                            "sum_of_costs_after": 110,
                            "runtime_ms": 20.0,
                        }
                    ],
                },
                {
                    "candidate_valid": True,
                    "conflicting_pairs_before": 5,
                    "sum_of_costs_before": 100,
                    "one_step_conflict_reduction": 0,
                    "one_step_cost_improvement": 1,
                    "rollouts": [
                        {
                            "horizon": 10,
                            "solved": True,
                            "iterations": 4,
                            "accepted_iterations": 2,
                            "conflicting_pairs_after": 0,
                            "sum_of_costs_after": 99,
                            "runtime_ms": 10.0,
                        }
                    ],
                },
            ]
        )
        outcome = aggregated[10]
        self.assertAlmostEqual(outcome["conflict_reduction"], 4.0)
        self.assertAlmostEqual(outcome["cost_improvement"], -4.5)
        self.assertAlmostEqual(outcome["solved_probability"], 0.5)
        self.assertAlmostEqual(
            outcome["one_step_conflict_reduction"], 0.5
        )


if __name__ == "__main__":
    unittest.main()
