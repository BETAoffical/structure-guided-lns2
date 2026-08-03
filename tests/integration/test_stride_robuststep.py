from __future__ import annotations

import unittest

from experiments.stride_robuststep_preflight import analyze_preflight_rows

from experiments.stride_robuststep import (
    evaluate_robuststep_variant,
    robust_pair_winner,
    validate_robuststep_seed_depth_config,
)


class StrideRobustStepTest(unittest.TestCase):
    def test_pair_requires_seed_agreement(self) -> None:
        winner = robust_pair_winner(
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [True, True, True, False],
            [True, True, True, False],
            minimum_paired_win_fraction=0.75,
            maximum_no_progress_disadvantage=0.0,
        )
        self.assertEqual(winner, 1)
        uncertain = robust_pair_winner(
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
            [True] * 4,
            [True] * 4,
            minimum_paired_win_fraction=0.75,
            maximum_no_progress_disadvantage=0.0,
        )
        self.assertEqual(uncertain, 0)

    def test_pair_rejects_worse_no_progress_risk(self) -> None:
        winner = robust_pair_winner(
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [True, True, False, False],
            [True, True, True, True],
            minimum_paired_win_fraction=0.75,
            maximum_no_progress_disadvantage=0.125,
        )
        self.assertEqual(winner, 0)

    def test_variant_reports_stable_good_set_and_uncertainty(self) -> None:
        profiles = {
            "state-a": {
                "a": {
                    "scores": [3.0] * 8,
                    "progress": [True] * 8,
                },
                "b": {
                    "scores": [2.0] * 8,
                    "progress": [True] * 8,
                },
                "c": {
                    "scores": [1.0] * 8,
                    "progress": [True] * 8,
                },
            }
        }
        report = evaluate_robuststep_variant(
            profiles,
            {
                "id": "test",
                "minimum_paired_win_fraction": 0.75,
                "maximum_no_progress_disadvantage": 0.0,
            },
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        )
        self.assertEqual(report["half_pairwise_consistency"], 1.0)
        self.assertEqual(report["mean_good_set_jaccard"], 1.0)
        self.assertEqual(report["unique_robust_winner_rate"], 1.0)
        self.assertEqual(report["full_pair_coverage"], 1.0)

    def test_seed_depth_contract_requires_independent_eight_seed_halves(self) -> None:
        config = {
            "schema": "lns2.stride.robuststep_seed_depth_config.v1",
            "scientific_status": "consumed_seed_depth_diagnostic",
            "formal_speed_claim": False,
            "fresh_confirmation_required": True,
            "controller_id": "stride-robuststep-v1",
            "label_schema": "lns2.stride.robust_step_label.v1",
            "trial_indices": list(range(16)),
            "first_half_indices": list(range(8)),
            "second_half_indices": list(range(8, 16)),
            "structure_weight": 0.02,
            "variants": [
                {"id": f"v{index}"} for index in range(4)
            ],
            "cohorts": [
                {"id": "design"},
                {"id": "confirmation"},
            ],
            "require_all_cohorts_pass": True,
            "runtime_used_in_label": False,
        }
        validate_robuststep_seed_depth_config(config)
        config["second_half_indices"] = list(range(7, 15))
        with self.assertRaisesRegex(ValueError, "8\\+8"):
            validate_robuststep_seed_depth_config(config)

    def test_preflight_selection_is_outcome_blind_and_targeted(self) -> None:
        source = {
            "role": "stride_robuststep_outcome_blind_load_preflight",
            "solver_seeds": [1, 2],
            "expected_map_count": 1,
            "expected_task_count": 2,
            "consumed_development_exceptions": [],
            "benchmarks": [{"id": "map-a", "layout_family": "maze"}],
            "selection_rule": {
                "minimum_mean_initial_conflicts": 10.0,
                "target_mean_initial_conflicts": [50.0, 200.0],
                "maximum_tasks_per_map": 2,
            },
        }
        dataset = [
            {
                "task_id": f"task-{index}", "map_id": "map-a",
                "layout_mode": "maze", "scenario_type": f"movingai_random_{index}",
                "agent_count": 100 * index,
            }
            for index in (1, 2)
        ]
        qualification = []
        for index, conflicts in ((1, 40), (2, 180)):
            for seed in (1, 2):
                qualification.append(
                    {
                        "task_id": f"task-{index}", "map_id": "map-a",
                        "solver_seed": seed, "agent_count": 100 * index,
                        "status": "ok", "initial_complete": True,
                        "state_fingerprint": f"state-{index}-{seed}",
                        "initial_conflicts": conflicts,
                        "initial_complexity": {
                            "conflict_pair_density": 0.01,
                            "mean_path_cost": 12.0,
                            "initial_low_level_expanded": 100.0,
                        },
                    }
                )
        report = analyze_preflight_rows(
            source, dataset, qualification, {"passed": True}, {"cases": []}
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            [row["task_id"] for row in report["recommended_tasks"]],
            ["task-1", "task-2"],
        )
        qualification[0]["controller_action"] = "forbidden"
        report = analyze_preflight_rows(
            source, dataset, qualification, {"passed": True}, {"cases": []}
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["forbidden_outcome_fields_found"], ["controller_action"])


if __name__ == "__main__":
    unittest.main()
