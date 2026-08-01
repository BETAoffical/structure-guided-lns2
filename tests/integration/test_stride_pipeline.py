from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.stride_lns import (
    STRIDE_STAGE1_CONFIG_SCHEMA,
    STRIDE_TRIAL_SCHEMA,
    aggregate_stride_candidate,
    assign_structure_scores,
    build_stride_labels,
    post_structure_metrics,
    run_stage1_audit,
    stride_dominates,
)
from experiments.stride_collection import (
    STRIDE_COLLECTION_SCHEMA,
    STRIDE_SELECTION_SCHEMA,
    _state_artifact_valid,
    load_stride_selection,
    stride_pp_seed,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class StrideStage1AuditTest(unittest.TestCase):
    def _bundle(self, root: Path, controller_id: str) -> None:
        payload = {
            "feature_schema_id": "lns2.realized_features.v2",
            "feature_schema_sha256": "feature-sha",
            "feature_dimensions": {"realized_dynamic": 124},
        }
        if controller_id == "v2-full":
            payload["default_controller"] = controller_id
        else:
            payload["controller_id"] = controller_id
        _write_json(root / "controller_manifest.json", payload)

    def test_audit_freezes_bundles_and_rejects_aggregated_old_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root / "v2", "v2-full")
            self._bundle(root / "mixed", "mixed-full-v2")
            _write_jsonl(
                root / "old.jsonl",
                [
                    {
                        "state_id": "state-1",
                        "map_id": "map-1",
                        "split": "train",
                        "agent_count": 80,
                        "trial_count": 4,
                        "features": {"realized_dynamic": {str(i): 0 for i in range(124)}},
                        "outcome": {"conflicts_after": 2.0},
                    }
                ],
            )
            config = {
                "schema": STRIDE_STAGE1_CONFIG_SCHEMA,
                "research_line": "stride-lns",
                "frozen_bundles": [
                    {"controller_id": "v2-full", "path": "v2"},
                    {"controller_id": "mixed-full-v2", "path": "mixed"},
                ],
                "historical_sources": [
                    {
                        "name": "old",
                        "role": "historical",
                        "kind": "ranking_index",
                        "path": "old.jsonl",
                    }
                ],
            }
            _write_json(root / "config.json", config)

            report = run_stage1_audit(
                config_path=root / "config.json",
                output=root / "output",
                project_root=root,
            )

            self.assertTrue(report["stage1_passed"])
            self.assertTrue(report["requires_fresh_stride_collection"])
            source = report["historical_sources"][0]
            self.assertFalse(source["reusable_for_stride_labels"])
            self.assertEqual(source["feature_dimensions"], {"124": 1})

    def test_sequence_trials_require_four_pp_seeds_and_post_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root / "v2", "v2-full")
            self._bundle(root / "mixed", "mixed-full-v2")
            rows = []
            for trial_index, seed in enumerate((11, 12, 13, 14)):
                rows.append(
                    {
                        "state_id": "state-1",
                        "map_id": "map-1",
                        "split": "dev",
                        "trial_index": trial_index,
                        "steps": [
                            {
                                "executed": True,
                                "candidate_id": "candidate-1",
                                "action": {"pp_random_seed": seed},
                                "post_structure": {
                                    "post_largest_component_ratio": 0.1,
                                    "post_conflict_edge_density": 0.2,
                                    "post_event_density": 0.3,
                                    "post_degree_concentration": 0.4,
                                },
                            }
                        ],
                    }
                )
            _write_jsonl(root / "trials.jsonl", rows)
            config = {
                "schema": STRIDE_STAGE1_CONFIG_SCHEMA,
                "research_line": "stride-lns",
                "frozen_bundles": [
                    {"controller_id": "v2-full", "path": "v2"},
                    {"controller_id": "mixed-full-v2", "path": "mixed"},
                ],
                "historical_sources": [
                    {
                        "name": "trials",
                        "role": "historical",
                        "kind": "sequence_trials",
                        "path": "trials.jsonl",
                    }
                ],
            }
            _write_json(root / "config.json", config)

            report = run_stage1_audit(
                config_path=root / "config.json",
                output=root / "output",
                project_root=root,
            )

            source = report["historical_sources"][0]
            self.assertTrue(source["has_individual_paired_trials"])
            self.assertTrue(source["has_post_state_structure"])
            self.assertTrue(source["reusable_for_stride_labels"])
            self.assertFalse(report["requires_fresh_stride_collection"])

    def test_non_historical_map_overlap_fails_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root / "v2", "v2-full")
            self._bundle(root / "mixed", "mixed-full-v2")
            row = {
                "state_id": "state-1",
                "map_id": "leaked-map",
                "split": "x",
                "agent_count": 80,
                "trial_count": 4,
                "features": {"realized_dynamic": {}},
            }
            _write_jsonl(root / "dev.jsonl", [row])
            _write_jsonl(root / "formal.jsonl", [row])
            config = {
                "schema": STRIDE_STAGE1_CONFIG_SCHEMA,
                "research_line": "stride-lns",
                "frozen_bundles": [
                    {"controller_id": "v2-full", "path": "v2"},
                    {"controller_id": "mixed-full-v2", "path": "mixed"},
                ],
                "historical_sources": [
                    {"name": "dev", "role": "dev", "kind": "ranking_index", "path": "dev.jsonl"},
                    {"name": "formal", "role": "formal", "kind": "ranking_index", "path": "formal.jsonl"},
                ],
            }
            _write_json(root / "config.json", config)

            report = run_stage1_audit(
                config_path=root / "config.json",
                output=root / "output",
                project_root=root,
            )

            self.assertFalse(report["stage1_passed"])
            self.assertEqual(report["map_leakage"][0]["maps"], ["leaked-map"])


class StrideQualityLabelTest(unittest.TestCase):
    def test_single_conflict_pair_is_normalized_by_all_agents(self) -> None:
        agents = []
        for agent_id in range(10):
            cell = 0 if agent_id in {0, 1} else agent_id
            agents.append({"id": agent_id, "path": [cell]})
        state = {
            "rows": 1,
            "cols": 10,
            "obstacles": [0] * 10,
            "agents": agents,
            "conflict_edges": [[0, 1]],
            "num_of_colliding_pairs": 1,
            "low_level": {},
        }

        metrics = post_structure_metrics(state)

        self.assertAlmostEqual(metrics["post_largest_component_ratio"], 0.2)
        self.assertAlmostEqual(metrics["post_conflict_edge_density"], 0.1)
        self.assertAlmostEqual(metrics["post_event_density"], 0.1)
        self.assertAlmostEqual(metrics["post_degree_concentration"], 0.1)

    def test_four_seed_aggregate_uses_two_worst_reductions(self) -> None:
        structures = {
            "post_largest_component_ratio": 0.2,
            "post_conflict_edge_density": 0.1,
            "post_event_density": 0.3,
            "post_degree_concentration": 0.1,
        }
        outcomes = [
            {
                "pp_seed": seed,
                "feasible": feasible,
                "conflicts_after": after,
                "post_structure": structures,
            }
            for seed, feasible, after in (
                (1, True, 5),
                (2, True, 7),
                (3, False, 10),
                (4, True, 6),
            )
        ]

        aggregate = aggregate_stride_candidate(before_conflicts=10, outcomes=outcomes)

        self.assertEqual(aggregate["feasible_rate"], 0.75)
        self.assertEqual(aggregate["progress_rate"], 0.75)
        self.assertEqual(aggregate["mean_conflicts_after"], 7.0)
        self.assertEqual(aggregate["robust_reduction"], 1.5)

    def test_structure_score_uses_midranks_and_lower_is_better(self) -> None:
        candidates = [
            {
                "mean_post_structure": {
                    name: value
                    for name in (
                        "post_largest_component_ratio",
                        "post_conflict_edge_density",
                        "post_event_density",
                        "post_degree_concentration",
                    )
                }
            }
            for value in (0.1, 0.1, 0.3)
        ]

        assign_structure_scores(candidates)

        self.assertAlmostEqual(candidates[0]["structural_score"], 0.25)
        self.assertAlmostEqual(candidates[1]["structural_score"], 0.25)
        self.assertEqual(candidates[2]["structural_score"], 1.0)

    def test_dominance_enforces_quality_floor_and_strict_structure_gain(self) -> None:
        right = {
            "feasible_rate": 0.75,
            "progress_rate": 0.75,
            "robust_reduction": 5.0,
            "structural_score": 0.6,
        }
        left = {
            "feasible_rate": 0.75,
            "progress_rate": 0.75,
            "robust_reduction": 4.0,
            "structural_score": 0.55,
        }

        self.assertTrue(stride_dominates(left, right, before_conflicts=20))
        left["structural_score"] = 0.551
        self.assertFalse(stride_dominates(left, right, before_conflicts=20))
        left["structural_score"] = 0.4
        left["robust_reduction"] = 3.99
        self.assertFalse(stride_dominates(left, right, before_conflicts=20))

    def test_label_builder_emits_reverse_pairs_with_equal_state_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            metric_names = (
                "post_largest_component_ratio",
                "post_conflict_edge_density",
                "post_event_density",
                "post_degree_concentration",
            )
            for candidate_index in range(4):
                for pp_seed in (101, 102, 103, 104):
                    rows.append(
                        {
                            "schema": STRIDE_TRIAL_SCHEMA,
                            "state_id": "state-1",
                            "candidate_id": f"candidate-{candidate_index}",
                            "map_id": "map-1",
                            "split": "pilot_train",
                            "source_policy": "v2-full",
                            "decision_stage": "early",
                            "before_conflicts": 10,
                            "agent_count": 80,
                            "pp_seed": pp_seed,
                            "feasible": True,
                            "conflicts_after": candidate_index,
                            "features": {f"feature-{i}": float(i) for i in range(124)},
                            "post_structure": {
                                name: 0.1 * candidate_index for name in metric_names
                            },
                            "repair_seconds": 999.0,
                        }
                    )
            _write_jsonl(root / "trials.jsonl", rows)

            summary = build_stride_labels(
                trials_path=root / "trials.jsonl", output=root / "labels"
            )
            pairs = [
                json.loads(line)
                for line in (root / "labels" / "dominance_pairs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertTrue(summary["coverage_gate_passed"])
            self.assertEqual(summary["dominance_pair_count"], 6)
            self.assertEqual(len(pairs), 12)
            self.assertEqual({row["label"] for row in pairs}, {0, 1})
            self.assertAlmostEqual(sum(row["sample_weight"] for row in pairs), 1.0)
            self.assertTrue(all("repair_seconds" not in row for row in pairs))


class StrideCollectionContractTest(unittest.TestCase):
    def test_pp_seeds_are_state_paired_and_trial_distinct(self) -> None:
        first = [stride_pp_seed("repair-state", index) for index in range(4)]
        second = [stride_pp_seed("repair-state", index) for index in range(4)]
        other = [stride_pp_seed("other-state", index) for index in range(4)]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 4)
        self.assertNotEqual(first, other)

    def test_selection_rejects_duplicate_states(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.jsonl"
            row = {
                "schema": STRIDE_SELECTION_SCHEMA,
                "state_id": "state-1",
                "map_id": "map-1",
                "task_id": "task-1",
                "split": "train",
                "source_policy": "v2-full",
                "decision_stage": "early",
                "source_root": "source",
                "before_fingerprint": "fingerprint",
                "solver_seed": 1,
                "decision_index": 2,
                "agent_count": 80,
                "prefix_actions": [],
            }
            _write_jsonl(path, [row, row])
            with self.assertRaisesRegex(ValueError, "duplicate selected state"):
                load_stride_selection(path)

    def test_state_artifact_requires_every_candidate_seed_pair(self) -> None:
        payload = {
            "schema": STRIDE_COLLECTION_SCHEMA,
            "run_fingerprint": "run",
            "state_id": "state",
            "complete": True,
            "candidates": [{"candidate_id": "a"}, {"candidate_id": "b"}],
            "trials": [
                {"candidate_id": candidate, "trial_index": trial}
                for candidate in ("a", "b")
                for trial in range(4)
            ],
        }
        self.assertTrue(
            _state_artifact_valid(payload, run_fingerprint="run", state_id="state")
        )
        payload["trials"].pop()
        self.assertFalse(
            _state_artifact_valid(payload, run_fingerprint="run", state_id="state")
        )

    def test_state_artifact_rejects_duplicate_trial_rows(self) -> None:
        trials = [
            {"candidate_id": "candidate", "trial_index": trial}
            for trial in range(4)
        ]
        trials[-1] = dict(trials[0])
        payload = {
            "schema": STRIDE_COLLECTION_SCHEMA,
            "run_fingerprint": "run",
            "state_id": "state",
            "complete": True,
            "candidates": [{"candidate_id": "candidate"}],
            "trials": trials,
        }
        self.assertFalse(
            _state_artifact_valid(payload, run_fingerprint="run", state_id="state")
        )


if __name__ == "__main__":
    unittest.main()
