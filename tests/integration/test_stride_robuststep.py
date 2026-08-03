from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

from experiments.stride_robuststep_preflight import (
    analyze_preflight_rows,
    prepare_congestion_preflight_dataset,
)

from experiments.stride_robuststep import (
    _confirmation_local_path,
    evaluate_robuststep_variant,
    evaluate_robuststep_score_variant,
    robust_pair_winner,
    validate_robuststep_confirmation_config,
    validate_robuststep_feature_probe_config,
    validate_robuststep_seed_depth_config,
    validate_robuststep_stepgate_config,
    validate_robuststep_v2_headroom_config,
)


class StrideRobustStepTest(unittest.TestCase):
    def test_registered_topology_preflight_has_generator_compatible_dimensions(self) -> None:
        root = Path(__file__).resolve().parents[2]
        source = json.loads(
            (root / "configs" / "stride_robuststep_topology_preflight_source.json")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(source["expected_map_count"], 6)
        self.assertEqual(source["expected_task_count"], 48)
        self.assertEqual(
            source["expected_instance_count"], source["expected_task_count"]
        )
        self.assertEqual(source["solver_seeds"], [1, 2])
        self.assertTrue(
            {"controller_repair_outcome", "candidate_repair_outcome"}
            <= set(source["selection_inputs_forbidden"])
        )

    def test_confirmation_resolves_wsl_project_path_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "build" / "artifact.json"
            artifact.parent.mkdir()
            artifact.write_text("{}\n", encoding="utf-8")
            recorded = f"/mnt/c/unrelated/{root.name}/build/artifact.json"
            self.assertEqual(
                _confirmation_local_path(root, recorded), artifact.resolve()
            )

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

    def test_topology_preflight_requires_qualified_noncontrol_groups(self) -> None:
        groups = ("ultra", "articulated", "control")
        source = {
            "role": "stride_robuststep_topology_balanced_preflight",
            "solver_seeds": [1],
            "expected_map_count": 3,
            "expected_task_count": 3,
            "consumed_development_exceptions": [],
            "benchmarks": [
                {"id": f"map-{group}", "layout_family": group}
                for group in groups
            ],
            "selection_rule": {
                "minimum_mean_initial_conflicts": 10.0,
                "target_mean_initial_conflicts": [25.0],
                "maximum_tasks_per_map": 1,
            },
            "topology_group_gates": {
                "required_groups": list(groups),
                "minimum_qualified_map_count": 2,
                "minimum_qualified_maps_by_group": {
                    "ultra": 1, "articulated": 1, "control": 0,
                },
                "underloaded_allowed_only_in_groups": ["control"],
            },
        }
        dataset = [
            {
                "task_id": f"task-{group}",
                "map_id": f"map-{group}",
                "layout_mode": group,
                "task_variant": "uniform_random_seed_1_agents_20",
                "agent_count": 20,
            }
            for group in groups
        ]
        qualification = [
            {
                "task_id": f"task-{group}",
                "map_id": f"map-{group}",
                "solver_seed": 1,
                "agent_count": 20,
                "status": "ok",
                "initial_complete": True,
                "state_fingerprint": f"state-{group}",
                "initial_conflicts": 0 if group == "control" else 20,
                "initial_complexity": {
                    "conflict_pair_density": 0.01,
                    "mean_path_cost": 12.0,
                    "initial_low_level_expanded": 100.0,
                },
            }
            for group in groups
        ]
        report = analyze_preflight_rows(
            source, dataset, qualification, {"passed": True}, {"cases": []}
        )
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["next_decision"], "register_proposal_only_candidate_coverage"
        )
        self.assertEqual(
            report["topology_group_reports"]["control"]["qualified_map_count"], 0
        )
        qualification[0]["initial_conflicts"] = 0
        report = analyze_preflight_rows(
            source, dataset, qualification, {"passed": True}, {"cases": []}
        )
        self.assertFalse(report["passed"])

    def test_congestion_preflight_is_pinned_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fetched = root / "fetched"
            (fetched / "maps").mkdir(parents=True)
            map_path = fetched / "maps" / "map-a.map"
            map_path.write_text(
                "type octile\nheight 6\nwidth 6\nmap\n" + "......\n" * 6,
                encoding="utf-8",
            )
            map_sha = hashlib.sha256(map_path.read_bytes()).hexdigest()
            manifest = fetched / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {"id": "map-a", "map_file": "maps/map-a.map", "map_sha256": map_sha},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            predecessor = root / "predecessor.json"
            predecessor.write_text("{}\n", encoding="utf-8")
            config = root / "config.json"
            config.write_text(
                json.dumps(
                    {
                        "role": "stride_robuststep_outcome_blind_congestion_preflight",
                        "dataset_revision": "test-congestion-v1",
                        "predecessor_report": {
                            "path": str(predecessor),
                            "sha256": hashlib.sha256(predecessor.read_bytes()).hexdigest(),
                        },
                        "fetched_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                        "master_seed": 123,
                        "task_seeds": [7],
                        "task_variants": ["opposite_exchange"],
                        "expected_map_count": 1,
                        "expected_task_count": 1,
                        "benchmarks": [
                            {"id": "map-a", "layout_family": "test", "map_sha256": map_sha, "agent_counts": [8]}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            first = prepare_congestion_preflight_dataset(
                fetched=fetched, source_config=config, output=root / "first"
            )
            second = prepare_congestion_preflight_dataset(
                fetched=fetched, source_config=config, output=root / "second"
            )
            self.assertEqual(first, second)
            first_scenario = next((root / "first" / "balanced_wall_clock" / "scenarios").glob("*.scen"))
            second_scenario = next((root / "second" / "balanced_wall_clock" / "scenarios").glob("*.scen"))
            self.assertEqual(first_scenario.read_bytes(), second_scenario.read_bytes())

    def test_distributional_score_reports_stable_top_set_and_zero_regret(self) -> None:
        profiles = {
            "state-a": {
                candidate: {
                    "scores": [score] * 16,
                    "progress": [score > 0.0] * 16,
                }
                for candidate, score in (("a", 4.0), ("b", 3.0), ("c", 2.0), ("d", 1.0))
            }
        }
        report = evaluate_robuststep_score_variant(
            profiles,
            {
                "id": "mean-sd025",
                "mode": "mean",
                "deviation_weight": 0.25,
                "no_progress_penalty": 0.0,
            },
            list(range(8)),
            list(range(8, 16)),
        )
        self.assertEqual(report["pairwise_consistency"], 1.0)
        self.assertEqual(report["mean_top3_overlap"], 1.0)
        self.assertEqual(report["mean_cross_half_normalized_regret"], 0.0)
        self.assertEqual(report["winner_agreement_rate"], 1.0)

    def test_confirmation_contract_freezes_selected_score_and_evidence_boundary(self) -> None:
        config = {
            "schema": "lns2.stride.robuststep_confirmation_config.v1",
            "scientific_status": "fresh_task_state_score_confirmation",
            "formal_speed_claim": False,
            "historically_untouched_map_claim": False,
            "controller_id": "stride-robuststep-v1",
            "score_schema": "lns2.stride.robust_step_score.v1",
            "trial_indices": list(range(16)),
            "first_half_indices": list(range(8)),
            "second_half_indices": list(range(8, 16)),
            "structure_weight": 0.02,
            "baseline_variant": {
                "id": "mean", "mode": "mean", "deviation_weight": 0.0,
                "no_progress_penalty": 0.0,
            },
            "selected_variant": {
                "id": "mean-np100", "mode": "mean", "deviation_weight": 0.0,
                "no_progress_penalty": 0.1,
            },
            "runtime_used_in_score": False,
            "future_repair_rounds_used_in_score": False,
            "cost_to_go_used_in_score": False,
            "training_before_confirmation_pass_forbidden": True,
            "expected_feature_dimension": 124,
            "expected_feature_schema_id": "lns2.realized_features.v2",
            "expected_maps": [f"map-{index}" for index in range(6)],
            "map_groups": {
                "compact": ["map-0", "map-1", "map-2"],
                "ultra": ["map-3", "map-4", "map-5"],
            },
            "selection_contract": {
                "source_policies": ["official_adaptive", "v2-full"],
                "maximum_source_decision_index": 11,
                "maximum_states_per_episode": 1,
                "result_blind": True,
            },
        }
        validate_robuststep_confirmation_config(config)
        config["selected_variant"]["no_progress_penalty"] = 0.2
        with self.assertRaisesRegex(ValueError, "retune"):
            validate_robuststep_confirmation_config(config)

    def test_v2_headroom_contract_is_diagnostic_and_uses_plain_mean(self) -> None:
        config = {
            "schema": "lns2.stride.robuststep_v2_headroom_config.v1",
            "scientific_status": "posthoc_diagnostic_only",
            "formal_speed_claim": False,
            "default_replacement_allowed": False,
            "training_allowed": False,
            "formal_ood_allowed": False,
            "diagnostic_result_may_promote_model": False,
            "controller_id": "v2-full",
            "score_schema": "lns2.stride.robust_step_score.v1",
            "trial_indices": list(range(16)),
            "first_half_indices": list(range(8)),
            "second_half_indices": list(range(8, 16)),
            "structure_weight": 0.02,
            "oracle_score": {
                "id": "mean", "mode": "mean", "deviation_weight": 0.0,
                "no_progress_penalty": 0.0,
            },
            "stable_state_definition": (
                "exact_first_half_and_second_half_oracle_winner_agreement"
            ),
            "runtime_used_in_oracle": False,
            "future_repair_rounds_used_in_oracle": False,
            "cost_to_go_used_in_oracle": False,
            "expected_feature_dimension": 124,
            "expected_feature_schema_id": "lns2.realized_features.v2",
        }
        validate_robuststep_v2_headroom_config(config)
        config["training_allowed"] = True
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            validate_robuststep_v2_headroom_config(config)

    def test_feature_probe_contract_is_grouped_ephemeral_and_fixed(self) -> None:
        config = {
            "schema": "lns2.stride.robuststep_feature_probe_config.v1",
            "scientific_status": "consumed_feature_sufficiency_diagnostic",
            "formal_speed_claim": False,
            "default_replacement_allowed": False,
            "runtime_export_allowed": False,
            "formal_ood_allowed": False,
            "diagnostic_result_may_promote_model": False,
            "ephemeral_probe_training_allowed": True,
            "diagnostic_controller_id": "stride-stepdiag-v1",
            "frozen_anchor_id": "v2-full",
            "score_schema": "lns2.stride.robust_step_score.v1",
            "trial_indices": list(range(16)),
            "first_half_indices": list(range(8)),
            "second_half_indices": list(range(8, 16)),
            "structure_weight": 0.02,
            "oracle_score": {
                "id": "mean", "mode": "mean", "deviation_weight": 0.0,
                "no_progress_penalty": 0.0,
            },
            "runtime_used_in_oracle": False,
            "future_repair_rounds_used_in_oracle": False,
            "cost_to_go_used_in_oracle": False,
            "expected_feature_dimension": 124,
            "expected_feature_schema_id": "lns2.realized_features.v2",
            "fold_protocol": {
                "mode": "leave_one_whole_map_out",
                "fold_count": 6,
                "held_out_maps": [f"map-{index}" for index in range(6)],
                "state_and_candidate_group_integrity": True,
            },
            "pair_contract": {
                "inclusion": "same_strict_direction_in_both_eight_seed_halves",
                "direction": "full_sixteen_seed_plain_mean",
                "weighting": "equal_total_weight_per_state",
            },
            "variants": [
                {"id": "stride-stepdiag-v1/exact-v2-86", "input_profile": "exact_v2_86"},
                {"id": "stride-stepdiag-v1/full-124-delta", "input_profile": "full_124_delta"},
                {"id": "stride-stepdiag-v1/full-124-context", "input_profile": "full_124_context"},
            ],
            "model_parameters": {
                "early_stopping": False,
                "l2_regularization": 0.1,
                "learning_rate": 0.05,
                "max_iter": 100,
                "max_leaf_nodes": 15,
                "min_samples_leaf": 20,
                "random_state": 20260714,
            },
        }
        validate_robuststep_feature_probe_config(config)
        config["runtime_export_allowed"] = True
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            validate_robuststep_feature_probe_config(config)

    def test_stepgate_contract_uses_nested_map_calibration_and_abstention(self) -> None:
        config = {
            "schema": "lns2.stride.robuststep_stepgate_config.v1",
            "scientific_status": "consumed_nested_abstention_diagnostic",
            "formal_speed_claim": False,
            "default_replacement_allowed": False,
            "runtime_export_allowed": False,
            "formal_ood_allowed": False,
            "diagnostic_result_may_promote_model": False,
            "ephemeral_probe_training_allowed": True,
            "diagnostic_controller_id": "stride-stepgate-v1",
            "frozen_anchor_id": "v2-full",
            "challenger_id": "stride-stepdiag-v1/exact-v2-86",
            "challenger_input_profile": "exact_v2_86",
            "outer_fold_protocol": {
                "mode": "leave_one_whole_map_out",
                "fold_count": 6,
                "held_out_maps": [f"map-{index}" for index in range(6)],
                "test_map_outcome_blind": True,
            },
            "inner_calibration": {
                "mode": "leave_one_whole_training_map_out",
                "candidate_thresholds": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
                "abstain_threshold": 1.01,
                "selection_rule": "lowest_threshold_passing_all_inner_gates",
                "fallback": "abstain_all_to_frozen_v2",
            },
            "inner_gates": {
                "minimum_overall_regret_improvement": 0.02,
                "minimum_stable_regret_improvement": 0.02,
                "minimum_stable_top3_delta": 0.0,
                "maximum_worst_map_regret_degradation": 0.03,
                "minimum_switch_count": 2,
            },
            "diagnostic_gates": {
                "minimum_stable_state_count": 24,
                "minimum_overall_regret_improvement": 0.02,
                "minimum_stable_regret_improvement": 0.03,
                "minimum_stable_top3_delta": 0.0,
                "minimum_map_regret_win_count": 3,
                "maximum_worst_map_regret_degradation": 0.03,
                "minimum_switch_count": 4,
                "minimum_switch_precision": 0.60,
            },
            "switch_timing": "before_pp_repair",
            "failure_triggered_switching": False,
            "runtime_used_in_oracle": False,
            "future_repair_rounds_used_in_oracle": False,
            "cost_to_go_used_in_oracle": False,
        }
        validate_robuststep_stepgate_config(config)
        config["failure_triggered_switching"] = True
        with self.assertRaisesRegex(ValueError, "current-step"):
            validate_robuststep_stepgate_config(config)


if __name__ == "__main__":
    unittest.main()
