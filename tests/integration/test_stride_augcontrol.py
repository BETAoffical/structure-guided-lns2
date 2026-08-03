from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.stride_augcontrol import (
    CONFLICT_CONTROLLER_ID,
    _input_specifications,
    _load_pair_table,
    _oracle_pool_opportunity,
    _pair_label_subgroup_coverage,
    _prediction_records,
    _validate_label_audit_provenance,
    validate_augcontrol_training_config,
)
from experiments._common import sha256_file
from experiments.stride_repairability_audit import AUDIT_SCHEMA
from experiments.stride_repairability import LABEL_SCHEMA


class StrideAugcontrolTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (root / "configs" / "stride_augcontrol_training.json").read_text(
                encoding="utf-8"
            )
        )

    def test_training_registration_freezes_name_capacity_and_147_inputs(self) -> None:
        config = self._config()
        validate_augcontrol_training_config(config)
        names, specifications = _input_specifications()
        self.assertEqual(len(names), 124)
        self.assertEqual(len(specifications), 147)
        self.assertEqual(config["controller_id"], "stride-augcontrol-v1")
        self.assertEqual(
            config["conflict_ablation_id"], CONFLICT_CONTROLLER_ID
        )
        self.assertFalse(config["hyperparameter_tuning"])
        self.assertFalse(config["default_replacement_allowed"])
        self.assertEqual(
            config["label_coverage_gates"],
            {
                "minimum_train_pair_state_fraction": 0.50,
                "minimum_validation_pair_state_fraction": 0.50,
                "minimum_train_pair_maps": 16,
                "minimum_validation_pair_maps": 6,
                "minimum_pair_states_per_map": 3,
                "required_subgroup_fields": [
                    "source_policy",
                    "decision_stage",
                    "topology_group",
                    "agent_band",
                ],
                "minimum_subgroup_pair_state_fraction": 0.30,
                "minimum_pair_states_per_subgroup": 5,
            },
        )

    def test_training_registration_rejects_runtime_and_capacity_drift(self) -> None:
        config = self._config()
        config["formal_ood_data_allowed"] = True
        with self.assertRaisesRegex(ValueError, "training protocol changed"):
            validate_augcontrol_training_config(config)
        config = self._config()
        config["model_parameters"]["max_iter"] = 200
        with self.assertRaisesRegex(ValueError, "model capacity changed"):
            validate_augcontrol_training_config(config)
        config = self._config()
        config["label_coverage_gates"]["minimum_validation_pair_maps"] = 5
        with self.assertRaisesRegex(ValueError, "label coverage gates changed"):
            validate_augcontrol_training_config(config)

    def test_training_rechecks_label_collection_audit_and_raw_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trials = root / "repair_trials.jsonl"
            trials.write_text('{"trial": 1}\n', encoding="utf-8")
            audit = root / "repairability_audit_report.json"
            audit_payload = {
                "schema": AUDIT_SCHEMA,
                "passed": True,
                "run_fingerprint": "run-a",
                "state_count": 240,
                "sha256": {"repair_trials": sha256_file(trials)},
            }
            audit.write_text(json.dumps(audit_payload), encoding="utf-8")
            summary = {
                "state_count": 240,
                "trial_sources": [
                    {"path": str(trials), "sha256": sha256_file(trials)}
                ],
                "audit_sources": [
                    {
                        "path": str(audit),
                        "sha256": sha256_file(audit),
                        "run_fingerprint": "run-a",
                        "state_count": 240,
                    }
                ],
            }
            verified = _validate_label_audit_provenance(summary)
            self.assertEqual(verified[0]["state_count"], 240)
            trials.write_text('{"trial": 2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw trial source differs"):
                _validate_label_audit_provenance(summary)

    def test_pair_table_reports_labeled_state_and_map_coverage(self) -> None:
        rows = []
        candidate_index = {}
        for state_index, map_id in enumerate(("map-a", "map-b")):
            state_id = f"state-{state_index}"
            left = f"left-{state_index}"
            right = f"right-{state_index}"
            candidate_index[(state_id, left)] = 2 * state_index
            candidate_index[(state_id, right)] = 2 * state_index + 1
            rows.extend(
                (
                    {
                        "schema": LABEL_SCHEMA,
                        "split": "train",
                        "state_id": state_id,
                        "map_id": map_id,
                        "left_candidate_id": left,
                        "right_candidate_id": right,
                        "label": 1,
                        "sample_weight": 0.5,
                    },
                    {
                        "schema": LABEL_SCHEMA,
                        "split": "train",
                        "state_id": state_id,
                        "map_id": map_id,
                        "left_candidate_id": right,
                        "right_candidate_id": left,
                        "label": 0,
                        "sample_weight": 0.5,
                    },
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            table = _load_pair_table(
                path,
                candidate_index,
                split="train",
                expected_schema=LABEL_SCHEMA,
            )
        self.assertEqual(table["state_count"], 2)
        self.assertEqual(table["state_ids"], ["state-0", "state-1"])
        self.assertEqual(table["map_state_counts"], {"map-a": 1, "map-b": 1})

    def test_subgroup_coverage_rejects_concentrated_pair_labels(self) -> None:
        common = {
            "split": "validation",
            "decision_stage": "early",
            "topology_group": "control",
            "agent_band": "low_mid",
        }
        grouped = {
            **{
                f"official-{index}": [
                    {**common, "source_policy": "official_adaptive"}
                ]
                for index in range(10)
            },
            **{
                f"v2-{index}": [{**common, "source_policy": "v2-full"}]
                for index in range(10)
            },
        }
        rows = _pair_label_subgroup_coverage(
            grouped,
            {f"official-{index}" for index in range(10)},
            split="validation",
            fields=("source_policy",),
            minimum_fraction=0.30,
            minimum_count=5,
        )
        by_value = {row["value"]: row for row in rows}
        self.assertTrue(by_value["official_adaptive"]["passed"])
        self.assertFalse(by_value["v2-full"]["passed"])

    @staticmethod
    def _candidates() -> dict[str, list[dict]]:
        common = {
            "map_id": "map-a",
            "split": "validation",
            "source_policy": "v2-full",
            "decision_stage": "middle",
            "agent_band": "high",
            "topology_group": "boundary_relevant",
            "mean_conflicts_after": 4.0,
            "mean_conflict_reduction_ratio": 0.6,
            "progress_rate": 1.0,
            "feasible_rate": 0.0,
        }
        return {
            "state-a": [
                {
                    **common,
                    "candidate_id": "base-good",
                    "candidate_kind": "base",
                    "repairability_score": 0.6,
                },
                {
                    **common,
                    "candidate_id": "boundary-best",
                    "candidate_kind": "boundary_only",
                    "repairability_score": 0.8,
                },
                {
                    **common,
                    "candidate_id": "base-bad",
                    "candidate_kind": "base",
                    "repairability_score": 0.2,
                },
            ]
        }

    def test_prediction_regret_and_pool_oracle_use_current_step_score(self) -> None:
        grouped = self._candidates()
        records = _prediction_records(
            "model-a",
            {"state-a": "base-good"},
            grouped,
            evaluation_pool="augmented",
        )
        self.assertAlmostEqual(records[0]["repairability_regret"], 0.2)
        self.assertAlmostEqual(
            records[0]["normalized_repairability_regret"], 1.0 / 3.0
        )
        opportunity = _oracle_pool_opportunity(grouped)
        self.assertEqual(opportunity["boundary_unique_best_rate"], 1.0)
        self.assertAlmostEqual(
            opportunity["mean_best_score_improvement_over_base_pool"], 0.2
        )


if __name__ == "__main__":
    unittest.main()
