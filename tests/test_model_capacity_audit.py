from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments.model_capacity_audit import (
    CAPACITY_ORDER,
    _diagnosis,
    _validate_config,
    export_capacity_model,
)
from experiments.policy_visited_aggregation_analysis import (
    train_equal_state_pairwise_model,
)
from experiments.ranking_objective_confirmation import (
    generate_gated_confirmation_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def row(state: str, candidate: str, conflicts: float, feature: float) -> dict:
    pareto = conflicts == 1.0
    return {
        "state_id": state,
        "map_id": f"map-{state}",
        "task_id": f"task-{state}",
        "candidate_id": candidate,
        "candidate_key": candidate,
        "actual_size": 4 if candidate == "a" else 8,
        "selection_families": ["target:4"],
        "features": {
            "realized_dynamic": {
                "x": feature,
                "state.colliding_pairs": 4.0,
            }
        },
        "outcome": {
            "solved_rate": 0.0,
            "conflicts_after": conflicts,
            "conflict_auc": conflicts + 0.5,
            "generated": conflicts + 10.0,
            "runtime": 0.01,
        },
        "labels": {
            "effectiveness_pareto": pareto,
            "compute_aware_pareto": pareto,
            "runtime_sensitive_pareto": pareto,
        },
    }


class ModelCapacityAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (PROJECT_ROOT / "configs" / "model_capacity_audit.json").read_text(
                encoding="utf-8"
            )
        )

    def test_registered_capacity_curve_is_fixed(self) -> None:
        _validate_config(self.config)
        self.assertEqual(tuple(self.config["capacity_order"]), CAPACITY_ORDER)
        self.assertEqual(
            [self.config["capacities"][name]["max_iter"] for name in CAPACITY_ORDER],
            [50, 100, 300, 500],
        )
        changed = json.loads(json.dumps(self.config))
        changed["capacities"]["very_large"]["max_iter"] = 501
        with self.assertRaisesRegex(ValueError, "very_large"):
            _validate_config(changed)

    def test_diagnosis_distinguishes_overfit_and_representation_limit(self) -> None:
        cross_validation = {"passed": False}
        baseline = {"pareto_top1_hit_rate": 0.5, "mean_conflict_regret": 0.3}
        improved = {"pareto_top1_hit_rate": 0.54, "mean_conflict_regret": 0.28}
        unchanged = {"pareto_top1_hit_rate": 0.51, "mean_conflict_regret": 0.3}
        in_sample = {
            "summaries": {
                "current": baseline,
                "large": improved,
                "very_large": unchanged,
            }
        }
        self.assertEqual(
            _diagnosis(cross_validation, in_sample, self.config), "overfit"
        )
        in_sample["summaries"]["large"] = unchanged
        self.assertEqual(
            _diagnosis(cross_validation, in_sample, self.config),
            "representation_limited",
        )
        self.assertEqual(
            _diagnosis({"passed": True}, in_sample, self.config),
            "capacity_limited",
        )

    def test_capacity_pairwise_portable_matches_sklearn(self) -> None:
        rows = []
        for state_number in range(4):
            rows.extend(
                [
                    row(f"s{state_number}", "a", 1.0, float(state_number)),
                    row(f"s{state_number}", "b", 3.0, float(state_number + 1)),
                ]
            )
        parameters = dict(self.config["capacities"]["small"])
        parameters["max_iter"] = 5
        parameters["min_samples_leaf"] = 2
        model, diagnostic = train_equal_state_pairwise_model(
            rows, "realized_dynamic", parameters
        )
        self.assertTrue(diagnostic["equal_state_weight"])
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_capacity_model(
                "small", model, rows, Path(directory)
            )
        self.assertTrue(manifest["equivalence"]["passed"])
        self.assertEqual(manifest["capacity"], "small")

    def test_capacity_validation_can_unlock_registered_confirmation_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "capacity_report.json"
            report.write_text(
                json.dumps(
                    {
                        "decision": "eligible_for_independent_confirmation",
                        "confirmation_generation_allowed": True,
                        "development_validation": {"acceptance": {"passed": True}},
                    }
                ),
                encoding="utf-8",
            )
            expected = {"generated": True}
            with mock.patch(
                "experiments.ranking_objective_confirmation.generate_dataset",
                return_value=expected,
            ) as generate:
                actual = generate_gated_confirmation_dataset(
                    report,
                    PROJECT_ROOT
                    / "configs"
                    / "ranking_objective_confirmation_dataset.json",
                )
            self.assertEqual(actual, expected)
            generate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
