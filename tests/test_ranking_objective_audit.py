from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from experiments.ranking_objective_audit import (
    DualOutcomeModel,
    PortableRawRegressor,
    candidate_reliability_report,
    export_objective_model,
    leave_one_train_map_out,
    objective_acceptance,
    score_dual_outcome_candidates,
    train_dual_outcome_model,
    train_impact_pairwise_model,
)
from experiments.ranking_objective_confirmation import (
    generate_gated_confirmation_dataset,
    validate_confirmation_dataset_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def candidate(
    state: str,
    map_id: str,
    key: str,
    conflicts: float,
    solved: float = 0.0,
    feature: float = 0.0,
    before: float = 4.0,
) -> dict:
    pareto = conflicts == 1.0 or solved > 0.0
    return {
        "state_id": state,
        "map_id": map_id,
        "task_id": f"task-{state}",
        "candidate_id": key,
        "candidate_key": key,
        "actual_size": 4 if key.endswith("a") else 8,
        "selection_families": ["target:4"],
        "features": {
            "realized_dynamic": {
                "x": feature,
                "state.colliding_pairs": before,
            }
        },
        "outcome": {
            "solved_rate": solved,
            "conflicts_after": conflicts,
            "conflict_auc": conflicts + 0.5,
            "generated": 10.0 + conflicts,
            "runtime": 0.01,
        },
        "labels": {
            "effectiveness_pareto": pareto,
            "compute_aware_pareto": pareto,
            "runtime_sensitive_pareto": pareto,
        },
    }


def small_parameters() -> dict:
    return {
        "learning_rate": 0.05,
        "max_iter": 5,
        "max_leaf_nodes": 3,
        "min_samples_leaf": 2,
        "l2_regularization": 0.1,
        "random_state": 20260714,
    }


class FixedEstimator:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.values[: len(matrix)], dtype=float)


class RankingObjectiveAuditTests(unittest.TestCase):
    def test_impact_pairwise_and_dual_training_keep_equal_state_weight(self) -> None:
        rows = [
            candidate("s1", "m1", "a", 1.0, feature=0.0),
            candidate("s1", "m1", "b", 3.0, feature=1.0),
            candidate("s1", "m1", "c", 4.0, feature=2.0),
            candidate("s2", "m2", "a", 1.0, feature=0.5),
            candidate("s2", "m2", "b", 2.0, feature=1.5),
        ]
        impact, impact_diagnostic = train_impact_pairwise_model(
            rows, small_parameters()
        )
        dual, dual_diagnostic = train_dual_outcome_model(
            rows, small_parameters(), 0.25
        )
        self.assertTrue(impact_diagnostic["equal_state_weight"])
        self.assertTrue(dual_diagnostic["equal_state_weight"])
        self.assertEqual(impact.profile, "realized_dynamic")
        self.assertFalse(any("outcome" in name for name in dual.feature_names))

    def test_dual_outcome_quantizes_solved_rate_and_hash_breaks_ties(self) -> None:
        rows = [
            candidate("s", "m", "b", 2.0, feature=1.0),
            candidate("s", "m", "a", 2.0, feature=0.0),
        ]
        model = DualOutcomeModel(
            "realized_dynamic",
            ["x"],
            FixedEstimator([0.501, 0.499]),
            FixedEstimator([0.5, 0.5]),
            0.25,
        )
        selected, values = score_dual_outcome_candidates(rows, model)
        self.assertEqual(values["quantized_solved_rate"], [0.5, 0.5])
        self.assertEqual(selected, 1)

    def test_portable_raw_regressor_uses_identity_output(self) -> None:
        model = PortableRawRegressor(
            ["x"],
            2.0,
            [
                [
                    {
                        "value": 0.0,
                        "feature_idx": 0,
                        "num_threshold": 0.0,
                        "missing_go_to_left": True,
                        "left": 1,
                        "right": 2,
                        "is_leaf": False,
                    },
                    {"value": -0.5, "is_leaf": True},
                    {"value": 1.5, "is_leaf": True},
                ]
            ],
        )
        self.assertEqual(model.predict([[-1.0], [1.0]]), [1.5, 3.5])

    def test_leave_one_map_out_keeps_anchors_only_on_training_side(self) -> None:
        train = []
        for map_number in range(12):
            map_id = f"map-{map_number:02d}"
            train.extend(
                [
                    candidate(f"state-{map_number}", map_id, "a", 1.0),
                    candidate(f"state-{map_number}", map_id, "b", 2.0),
                ]
            )
        anchors = [
            candidate("anchor", "historical-map", "a", 1.0),
            candidate("anchor", "historical-map", "b", 2.0),
        ]
        folds = leave_one_train_map_out(train, anchors)
        self.assertEqual(len(folds), 12)
        for fold in folds:
            self.assertEqual(fold["anchor_state_count"], 1)
            self.assertEqual(
                {row["state_id"] for row in fold["held_rows"]},
                {f"state-{int(fold['validation_map'].split('-')[1])}"},
            )
            self.assertIn("anchor", {row["state_id"] for row in fold["fit_rows"]})

    def test_candidate_reliability_requires_four_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / "collection"
            offline = root / "offline"
            collection.mkdir()
            offline.mkdir()
            manifest = []
            for candidate_id, outcomes in {
                "a": [1, 1, 1, 1],
                "b": [2, 3, 2, 3],
            }.items():
                for trial, conflicts in enumerate(outcomes):
                    manifest.append(
                        {
                            "complete": True,
                            "error": None,
                            "state_id": "state",
                            "candidate_id": candidate_id,
                            "outcome": {
                                "evaluation_trial_index": trial,
                                "conflicts_before": 4,
                                "conflicts_after": conflicts,
                            },
                        }
                    )
            (collection / "evaluation_trial_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
            )
            prediction = {
                "state_id": "state",
                "map_id": "map",
                "candidate_id": "a",
                "conflict_regret": 0.0,
            }
            for name in ("v1_realized_dynamic", "v2_realized_dynamic"):
                (offline / f"offline_predictions__{name}.jsonl").write_text(
                    json.dumps(prediction) + "\n", encoding="utf-8"
                )
            report = candidate_reliability_report(collection, offline, 4)
            self.assertEqual(report["trial_count"], 8)
            self.assertEqual(report["oracle_conflict_reduction"]["positive_state_fraction"], 1.0)
            manifest.pop()
            (collection / "evaluation_trial_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "trial coverage"):
                candidate_reliability_report(collection, offline, 4)

    def test_gate_does_not_round_a_near_five_percent_improvement_up(self) -> None:
        baseline = {
            "pareto_top1_hit_rate": 0.35,
            "mean_conflict_regret": 1.0,
        }
        challenger = {
            "pareto_top1_hit_rate": 0.35,
            "mean_conflict_regret": 0.950007,
            "maximum_size_share": 0.5,
        }
        result = objective_acceptance(
            baseline,
            challenger,
            {
                "top1_delta_95_ci": [0.0, 0.0],
                "conflict_regret_improvement_95_ci": [0.0, 0.1],
            },
            12,
            12,
            {
                "minimum_top1_improvement": 0.03,
                "minimum_conflict_regret_improvement": 0.05,
                "maximum_top1_degradation": 0.02,
                "maximum_conflict_regret_degradation": 0.05,
                "maximum_single_size_share": 0.8,
                "minimum_train_maps_no_worse": 8,
                "minimum_validation_maps_no_worse": 4,
            },
        )
        self.assertFalse(result["passed"])
        self.assertLess(result["conflict_regret_improvement"], 0.05)

    def test_dual_outcome_portable_export_matches_sklearn(self) -> None:
        rows = []
        for state_number in range(4):
            rows.extend(
                [
                    candidate(
                        f"s{state_number}",
                        f"m{state_number}",
                        "a",
                        1.0,
                        solved=0.25,
                        feature=float(state_number),
                    ),
                    candidate(
                        f"s{state_number}",
                        f"m{state_number}",
                        "b",
                        3.0,
                        feature=float(state_number + 1),
                    ),
                ]
            )
        model, _ = train_dual_outcome_model(rows, small_parameters(), 0.25)
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_objective_model(
                "dual_outcome", model, rows, Path(directory), 0.25
            )
        self.assertTrue(manifest["equivalence"]["passed"])
        self.assertEqual(manifest["selector_type"], "dual_outcome")

    def test_confirmation_config_registers_twelve_isolated_maps(self) -> None:
        config = json.loads(
            (
                PROJECT_ROOT
                / "configs"
                / "ranking_objective_confirmation_dataset.json"
            ).read_text(encoding="utf-8")
        )
        validate_confirmation_dataset_config(config)
        self.assertEqual(config["master_seed"], 20270421)
        self.assertNotIn(
            config["master_seed"],
            {20270317, 20261219, 20261117, 20260714},
        )
        counts = config["splits"]["policy_confirmation"]["layout_counts"]
        self.assertEqual(sum(counts.values()), 12)
        self.assertEqual(sum(counts.values()) * config["tasks_per_map"] * 3, 144)

    def test_failed_development_gate_forbids_confirmation_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "audit_report.json"
            report.write_text(
                json.dumps(
                    {
                        "decision": "stop_objective_alignment",
                        "confirmation_generation_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch(
                "experiments.ranking_objective_confirmation.generate_dataset"
            ) as generate:
                with self.assertRaisesRegex(ValueError, "generation is forbidden"):
                    generate_gated_confirmation_dataset(
                        report,
                        PROJECT_ROOT
                        / "configs"
                        / "ranking_objective_confirmation_dataset.json",
                    )
            generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
