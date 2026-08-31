from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments._common import sha256_file
from experiments.stride_lns import (
    FROZEN_FEATURE_SCHEMA_ID,
    REQUIRED_POST_STRUCTURE_FIELDS,
    STRIDE_TRIAL_SCHEMA,
)
from experiments.stride_repairability import (
    _seed_half_action_stability,
    build_repairability_labels,
    validate_repairability_label_config,
)
from experiments.stride_repairability_audit import AUDIT_SCHEMA
from lns2_selector.compatibility.schemas import (
    STRIDE_MAPBASE_AUDIT_SCHEMA as MAPBASE_AUDIT_SCHEMA,
)
from experiments.stride_repairability_collection import (
    STATE_SCHEMA,
    _artifact_valid,
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT, state_before_decision
from experiments.stride_repairability_selection import (
    prepare_repairability_selection,
    validate_repairability_data_design,
)


class StrideRepairabilityTest(unittest.TestCase):
    def _config_path(self) -> Path:
        return (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "stride_repairability_label_design.json"
        )

    def _data_config_path(self) -> Path:
        return (
            Path(__file__).resolve().parents[2]
            / "configs"
            / "stride_repairability_data_design.json"
        )

    def _rows(self, *, unpaired: bool = False, runtime_scale: float = 1.0) -> list[dict]:
        names = PROFILE_FEATURE_NAMES["realized_dynamic"]
        rows = []
        for trial_index in range(16):
            after_by_candidate = {
                "good": 6,
                "medium": 8,
                "unstable": 3 if trial_index % 2 == 0 else 10,
            }
            for position, (candidate_id, conflicts_after) in enumerate(
                after_by_candidate.items()
            ):
                rows.append(
                    {
                        "schema": STRIDE_TRIAL_SCHEMA,
                        "state_id": "state-a",
                        "candidate_id": candidate_id,
                        "candidate_kind": "base",
                        "actual_size": 4,
                        "selection_families": [
                            ("target:4", "collision:4", "random:4")[position],
                            *(
                                ["topology-boundary-core"]
                                if candidate_id == "medium"
                                else []
                            ),
                        ],
                        "agents": [4 * position + value for value in range(4)],
                        "trial_index": trial_index,
                        "pp_seed": 1000 + trial_index + int(unpaired and position == 2),
                        "map_id": "map-a",
                        "split": "train",
                        "source_policy": "v2-full",
                        "decision_stage": "middle",
                        "before_conflicts": 10,
                        "agent_count": 100,
                        "feasible": False,
                        "conflicts_after": conflicts_after,
                        "step_runtime": runtime_scale * (position + 1),
                        "features": {name: float(position) for name in names},
                        "post_structure": {
                            name: 0.1 for name in REQUIRED_POST_STRUCTURE_FIELDS
                        },
                    }
                )
        return rows

    @staticmethod
    def _write_rows(path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    @staticmethod
    def _write_audit(
        path: Path,
        trials: Path,
        *,
        passed: bool = True,
        schema: str = AUDIT_SCHEMA,
    ) -> Path:
        path.write_text(
            json.dumps(
                {
                    "schema": schema,
                    "passed": passed,
                    "run_fingerprint": "test-run",
                    "state_count": 1,
                    "sha256": {"repair_trials": sha256_file(trials)},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path

    def test_config_freezes_identity_seed_product_and_forbidden_inputs(self) -> None:
        config = json.loads(self._config_path().read_text(encoding="utf-8"))
        validate_repairability_label_config(config)
        config["robust_pair_rule"]["minimum_paired_win_fraction"] = 0.5
        with self.assertRaisesRegex(ValueError, "robust-pair rule changed"):
            validate_repairability_label_config(config)

    def test_data_design_and_result_blind_selection_are_map_disjoint(self) -> None:
        config = json.loads(self._data_config_path().read_text(encoding="utf-8"))
        validate_repairability_data_design(config)
        maps = [
            map_id
            for split_maps in config["map_split"].values()
            for map_id in split_maps
        ]
        rows = []
        for index in range(240):
            map_id = maps[index % len(maps)]
            map_round = index // len(maps)
            rows.append(
                {
                    "state_id": f"state-{index:03d}",
                    "map_id": map_id,
                    "task_id": f"task-{index:03d}",
                    "split": "source",
                    "source_policy": ("official_adaptive", "v2-full")[
                        map_round % 2
                    ],
                    "decision_stage": ("early", "middle", "late")[index % 3],
                    "source_root": "/source",
                    "episode_id": f"episode-{index:03d}",
                    "before_fingerprint": f"fingerprint-{index:03d}",
                    "before_conflicts": 1 + index % 501,
                    "solver_seed": index % 3,
                    "decision_index": index % 12,
                    "agent_count": 100 + index % 500,
                    "prefix_actions": [],
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            self._write_rows(source, rows)
            report = prepare_repairability_selection(
                config_path=self._data_config_path(),
                selection_path=source,
                output=root / "selection",
            )
            self.assertTrue(report["passed"])
            self.assertEqual(report["state_count"], 240)
            self.assertEqual(report["replacements"], {})
            selected = [
                json.loads(line)
                for line in (root / "selection" / "state_selection.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            train = {
                row["map_id"] for row in selected if row["research_split"] == "train"
            }
            validation = {
                row["map_id"]
                for row in selected
                if row["research_split"] == "validation"
            }
            self.assertFalse(train & validation)
            replaced_rows = [
                {
                    **row,
                    "map_id": (
                        "den206d" if row["map_id"] == "arena2" else row["map_id"]
                    ),
                }
                for row in rows
            ]
            replacement_source = root / "replacement-source.jsonl"
            self._write_rows(replacement_source, replaced_rows)
            replacement = prepare_repairability_selection(
                config_path=self._data_config_path(),
                selection_path=replacement_source,
                output=root / "replacement-selection",
            )
            self.assertEqual(replacement["replacements"], {"arena2": "den206d"})

    def test_builder_keeps_only_cross_seed_robust_pair_and_equal_state_weight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "trials.jsonl"
            self._write_rows(trials, self._rows())
            audit = self._write_audit(root / "audit.json", trials)
            summary = build_repairability_labels(
                config_path=self._config_path(),
                trial_paths=[trials],
                audit_report_paths=[audit],
                output=root / "labels",
            )
            pairs = [
                json.loads(line)
                for line in (root / "labels" / "dominance_pairs.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(summary["possible_pair_count"], 3)
            self.assertEqual(summary["robust_pair_count"], 1)
            self.assertEqual(summary["uncertain_pair_count"], 2)
            self.assertEqual(summary["conflict_only_robust_pair_count"], 1)
            stability = summary["seed_half_action_stability"]
            self.assertEqual(stability["state_count"], 1)
            self.assertEqual(stability["exact_winner_agreement_rate"], 1.0)
            self.assertEqual(stability["mean_top3_overlap"], 1.0)
            stability_rows = [
                json.loads(line)
                for line in (
                    root / "labels" / "seed_half_action_stability.jsonl"
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(stability_rows[0]["first_half_winner"], "good")
            self.assertEqual(stability_rows[0]["second_half_winner"], "good")
            self.assertEqual({row["label"] for row in pairs}, {0, 1})
            self.assertEqual(sum(row["sample_weight"] for row in pairs), 1.0)
            positive = next(row for row in pairs if row["label"] == 1)
            self.assertEqual(positive["left_candidate_id"], "good")
            self.assertEqual(positive["right_candidate_id"], "medium")
            self.assertEqual(positive["paired_win_fraction"], 1.0)
            candidates = [
                json.loads(line)
                for line in (root / "labels" / "candidate_aggregates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            good = next(row for row in candidates if row["candidate_id"] == "good")
            self.assertEqual(good["candidate_kind"], "base")
            self.assertEqual(good["agents"], [0, 1, 2, 3])

    def test_builder_accepts_passed_mapbase_audit_with_identical_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "mapbase-trials.jsonl"
            self._write_rows(trials, self._rows())
            audit = self._write_audit(
                root / "mapbase-audit.json",
                trials,
                schema=MAPBASE_AUDIT_SCHEMA,
            )
            summary = build_repairability_labels(
                config_path=self._config_path(),
                trial_paths=[trials],
                audit_report_paths=[audit],
                output=root / "labels",
            )
            self.assertEqual(summary["audit_sources"][0]["schema"], MAPBASE_AUDIT_SCHEMA)
            self.assertEqual(summary["state_count"], 1)

    def test_runtime_fields_do_not_change_candidates_or_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            self._write_rows(first, self._rows(runtime_scale=1.0))
            self._write_rows(second, self._rows(runtime_scale=999.0))
            first_audit = self._write_audit(root / "first-audit.json", first)
            second_audit = self._write_audit(root / "second-audit.json", second)
            build_repairability_labels(
                config_path=self._config_path(),
                trial_paths=[first],
                audit_report_paths=[first_audit],
                output=root / "first-labels",
            )
            build_repairability_labels(
                config_path=self._config_path(),
                trial_paths=[second],
                audit_report_paths=[second_audit],
                output=root / "second-labels",
            )
            for name in (
                "candidate_aggregates.jsonl",
                "dominance_pairs.jsonl",
                "conflict_only_dominance_pairs.jsonl",
                "seed_half_action_stability.jsonl",
            ):
                self.assertEqual(
                    (root / "first-labels" / name).read_bytes(),
                    (root / "second-labels" / name).read_bytes(),
                )

    def test_builder_rejects_unpaired_pp_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "trials.jsonl"
            self._write_rows(trials, self._rows(unpaired=True))
            audit = self._write_audit(root / "audit.json", trials)
            with self.assertRaisesRegex(ValueError, "PP seeds are not paired"):
                build_repairability_labels(
                    config_path=self._config_path(),
                    trial_paths=[trials],
                    audit_report_paths=[audit],
                    output=root / "labels",
                )

    def test_builder_rejects_map_level_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self._rows()
            validation = []
            for row in rows:
                validation.append(
                    {**row, "state_id": "state-b", "split": "validation"}
                )
            trials = root / "trials.jsonl"
            self._write_rows(trials, [*rows, *validation])
            audit = self._write_audit(root / "audit.json", trials)
            with self.assertRaisesRegex(ValueError, "map split leakage"):
                build_repairability_labels(
                    config_path=self._config_path(),
                    trial_paths=[trials],
                    audit_report_paths=[audit],
                    output=root / "labels",
                )

    def test_builder_requires_passed_matching_collection_audit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "trials.jsonl"
            self._write_rows(trials, self._rows())
            failed = self._write_audit(root / "failed-audit.json", trials, passed=False)
            with self.assertRaisesRegex(ValueError, "audit did not pass"):
                build_repairability_labels(
                    config_path=self._config_path(),
                    trial_paths=[trials],
                    audit_report_paths=[failed],
                    output=root / "failed-labels",
                )
            passed = self._write_audit(root / "passed-audit.json", trials)
            trials.write_text(trials.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from its audit"):
                build_repairability_labels(
                    config_path=self._config_path(),
                    trial_paths=[trials],
                    audit_report_paths=[passed],
                    output=root / "mismatched-labels",
                )

    def test_seed_half_winner_tie_is_reported_as_uncertain(self) -> None:
        result = _seed_half_action_stability(
            "state-tied",
            {"candidate-a": [0.5] * 16, "candidate-b": [0.5] * 16},
            tie_epsilon=1e-12,
        )
        self.assertFalse(result["exact_winner_agreement"])
        self.assertEqual(result["comparable_pair_count"], 0)
        self.assertEqual(result["full_score_top1_margin"], 0.0)

    def test_collection_seed_is_paired_by_state_and_distinct_by_trial(self) -> None:
        first = [
            repairability_pp_seed("state-fingerprint", index) for index in range(16)
        ]
        second = [
            repairability_pp_seed("state-fingerprint", index) for index in range(16)
        ]
        self.assertEqual(first, second)
        self.assertEqual(len(set(first)), 16)
        self.assertNotEqual(
            first,
            [repairability_pp_seed("other-state", index) for index in range(16)],
        )
        self.assertEqual(
            repairability_restore_seed("state-fingerprint"),
            repairability_restore_seed("state-fingerprint"),
        )
        self.assertNotEqual(
            repairability_restore_seed("state-fingerprint"),
            repairability_restore_seed("other-state"),
        )

    def test_target_state_reconstruction_does_not_read_target_outcome(self) -> None:
        initial = {
            "initialized": True,
            "initial_solution_complete": True,
            "feasible": False,
            "done": False,
            "rows": 1,
            "cols": 2,
            "obstacles": [],
            "sum_of_costs": 2,
            "num_of_colliding_pairs": 1,
            "conflict_edges": [[0, 1]],
            "agents": [
                {"id": 0, "path": [0, 1], "conflict_degree": 1},
                {"id": 1, "path": [1, 0], "conflict_degree": 1},
            ],
            "iteration": 7,
            "low_level": {"expanded": 1, "generated": 2, "reopened": 0, "runs": 1},
        }
        from experiments.repair_collection import state_fingerprint

        target = {
            "decision_index": 7,
            "before_fingerprint": state_fingerprint(initial),
            "state_delta": "must-not-be-read",
            "state_extras_delta": "must-not-be-read",
            "metrics": "must-not-be-read",
            "after": "must-not-be-read",
        }
        self.assertEqual(
            state_before_decision(
                initial,
                [target],
                decision_index=7,
                expected_fingerprint=state_fingerprint(initial),
            ),
            initial,
        )

    def test_collection_resume_artifact_requires_complete_candidate_product(self) -> None:
        decision = {
            "state_id": "state-a",
            "before_fingerprint": "before-state",
            "before_conflicts": 5,
            "layout_mode": "family",
            "map_id": "map",
            "task_id": "task",
            "research_split": "research",
            "split": "source",
            "source_policy": "v2-full",
            "decision_stage": "middle",
            "solver_seed": 1,
            "agent_count": 10,
        }
        candidates = [
            {
                "candidate_id": candidate_id,
                "agents": [position],
                "candidate_kind": "base",
                "actual_size": 1,
                "selection_families": ["target:4"],
            }
            for position, candidate_id in enumerate(("a", "b"))
        ]
        features = {
            name: 0.0 for name in PROFILE_FEATURE_NAMES["realized_dynamic"]
        }
        trials = [
            {
                "schema": STRIDE_TRIAL_SCHEMA,
                "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
                "state_id": "state-a",
                "candidate_id": candidate["candidate_id"],
                "candidate_kind": "base",
                "actual_size": 1,
                "selection_families": ["target:4"],
                "agents": candidate["agents"],
                "layout_family": "family",
                "map_id": "map",
                "task_id": "task",
                "split": "research",
                "source_split": "source",
                "source_policy": "v2-full",
                "decision_stage": "middle",
                "solver_seed": 1,
                "agent_count": 10,
                "before_conflicts": 5,
                "before_fingerprint": "before-state",
                "before_repair_fingerprint": "before-repair",
                "trial_index": index,
                "pp_seed": repairability_pp_seed("before-repair", index),
                "features": dict(features),
                "feasible": False,
                "replan_success": True,
                "repair_outcome": "conflict_reduced",
                "conflicts_after": 4,
                "after_fingerprint": f"after-{candidate['candidate_id']}-{index}",
                "after_repair_fingerprint": (
                    f"after-repair-{candidate['candidate_id']}-{index}"
                ),
                "post_structure": {
                    name: 0.1 for name in REQUIRED_POST_STRUCTURE_FIELDS
                },
                "native_step_seconds": 0.1,
                "pp_replan_seconds": 0.05,
            }
            for candidate in candidates
            for index in range(16)
        ]
        payload = {
            "schema": STATE_SCHEMA,
            "run_fingerprint": "run-a",
            "state_id": "state-a",
            "complete": True,
            "decision": decision,
            "before_fingerprint": "before-state",
            "before_repair_fingerprint": "before-repair",
            "before_conflicts": 5,
            "state_restore": {
                "contract": TARGET_STATE_RESTORE_CONTRACT,
                "restore_seed": repairability_restore_seed("before-repair"),
                "repair_structure_fingerprint": "before-repair",
            },
            "base_candidate_count": 2,
            "boundary_candidate_count": 0,
            "candidates": candidates,
            "trials": trials,
        }
        self.assertTrue(
            _artifact_valid(
                payload,
                run_fingerprint="run-a",
                state_id="state-a",
                trial_indices=tuple(range(16)),
                decision=decision,
            )
        )
        payload["trials"] = trials[:-1]
        self.assertFalse(
            _artifact_valid(
                payload,
                run_fingerprint="run-a",
                state_id="state-a",
                trial_indices=tuple(range(16)),
                decision=decision,
            )
        )


if __name__ == "__main__":
    unittest.main()
