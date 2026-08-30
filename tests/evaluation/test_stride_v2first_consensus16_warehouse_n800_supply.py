from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments._common import read_json, sha256_file
from experiments.stride_v2first_consensus16_warehouse_n800_supply import (
    CANDIDATE_SEEDS,
    CONTROLLERS,
    MAXIMUM_SCREEN_TRACE_DECISIONS,
    PREFIX_V2_OUTCOMES,
    TASK_IDS,
    V2_EXPERIMENT_ID,
    V2_STATUS_SCHEMA,
    V3_EXPERIMENT_ID,
    V3_STATUS_SCHEMA,
    _screen_qualification_root,
    _screen_preaction_certificate,
    formal_schedule,
    load_config,
    plan,
    run,
    run_formal,
    run_screen,
    screen_schedule,
    select_screen_keys,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / "configs" / "stride_v2first_consensus16_warehouse_n800_supply_v1.json"
)
CONFIG_SHA256 = "4c4ed554d53f00e4dfc5e63a4f843aedd426d0cb3e69564ef775f03e5e321ed4"
V2_CONFIG = (
    ROOT / "configs" / "stride_v2first_consensus16_warehouse_n800_supply_v2.json"
)
V2_CONFIG_SHA256 = "484f1944135106782c8355712fc7393c5825827bb25ccf3b6a635be48b16ca3b"
V3_CONFIG = (
    ROOT / "configs" / "stride_v2first_consensus16_warehouse_n800_supply_v3.json"
)
V3_CONFIG_SHA256 = "4d93f791ed4f401ce6e2c8167dccdbd0763fb1076c29ec92041fc93918f78705"


def _screen_decision(index: int, *, offer: bool = False) -> dict:
    fingerprint = "platform-a"
    generation = None
    phase = "v2_only"
    if offer:
        phase = "consensus_rescue_due"
        agents = [3, 5, 8]
        generation = {
            "attempted": True,
            "available": True,
            "offered": True,
            "challenger_present": True,
            "structural_selected": False,
            "consumed": True,
            "decision_index": index,
            "before_repair_fingerprint": fingerprint,
            "component_available": True,
            "hotspot_available": True,
            "component_candidate_id": "consensus16",
            "hotspot_candidate_id": "consensus16",
            "component_agents": agents,
            "hotspot_agents": agents,
            "exact_agent_consensus": True,
            "consensus_candidate_id": "consensus16",
            "candidate_id": "consensus16",
            "fallback": None,
            "generated_candidate_count": 1,
        }
    return {
        "decision_index": index,
        "before_platform_signature": fingerprint,
        # These outcome fields must not participate in screen selection.
        "success": index % 2 == 0,
        "capped_wall_time_to_feasible": 999.0 - index,
        "controller": {
            "v2_first_consensus_rescue": {
                "selection": {
                    "decision_index": index,
                    "before_repair_fingerprint": fingerprint,
                    "repair_fingerprint": fingerprint,
                    "selection_phase": phase,
                    "offered": offer,
                    "rescue_due": offer,
                    "structural_profile": "component_hotspot_consensus",
                    "nominal_size": 16,
                    "minimum_consecutive_v2_exact_rollbacks": 3,
                    "consensus_required": True,
                    "consecutive_v2_exact_rollbacks": 3 if offer else 0,
                },
                "generation": generation,
            }
        },
    }


class WarehouseN800SupplyRunnerTest(unittest.TestCase):
    def test_registered_config_and_task_hashes_are_live(self) -> None:
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)
        path, root, config = load_config(CONFIG)
        self.assertEqual(path, CONFIG.resolve())
        self.assertEqual(tuple(config["cohort"]["candidate_solver_seeds"]), CANDIDATE_SEEDS)
        self.assertEqual(
            tuple(row["id"] for row in config["cohort"]["tasks"]), TASK_IDS
        )
        self.assertEqual(len(config["cohort"]["tasks"]), 4)
        for task in config["cohort"]["tasks"]:
            for field in ("task_file", "scenario_file"):
                registered = task[field]
                self.assertEqual(
                    sha256_file(root / registered["path"]), registered["sha256"]
                )
        for registered in config["inputs"].values():
            self.assertEqual(
                sha256_file(root / registered["path"]), registered["sha256"]
            )

    def test_plan_freezes_sixteen_screen_keys_and_worker_boundary(self) -> None:
        _path, _root, config = load_config(CONFIG)
        schedule = screen_schedule(config)
        self.assertEqual(len(schedule), 16)
        self.assertEqual(
            {(row["task_id"], row["solver_seed"]) for row in schedule},
            {(task, seed) for task in TASK_IDS for seed in CANDIDATE_SEEDS},
        )
        self.assertEqual({row["controller"] for row in schedule}, {"consensus16_rescue"})
        result = plan(CONFIG)
        self.assertEqual(result["screen"]["candidate_key_count"], 16)
        self.assertEqual(result["screen"]["workers"], 16)
        self.assertFalse(result["screen"]["included_in_ttf"])
        self.assertEqual(result["formal"]["timed_workers"], 1)
        self.assertEqual(result["formal"]["fresh_qualification_workers"], 16)
        self.assertTrue(result["formal"]["strict_serial"])
        self.assertEqual(result["formal"]["episode_count"], 8)
        self.assertEqual(result["formal"]["maximum_registered_ttf_seconds"], 960.0)
        self.assertFalse(result["solver_or_controller_invoked"])

    def test_v2_overlay_reuses_core_with_known_qualification_and_new_identity(self) -> None:
        self.assertEqual(sha256_file(V2_CONFIG), V2_CONFIG_SHA256)
        _path, root, config = load_config(V2_CONFIG)
        self.assertEqual(config["experiment_id"], V2_EXPERIMENT_ID)
        self.assertEqual(
            config["scientific_status"],
            "post_v1_invalid_preregistered_before_v2_controller",
        )
        self.assertEqual(config["_profile"]["profile"], "v2")
        self.assertEqual(
            tuple(config["cohort"]["candidate_solver_seeds"]), (20, 21, 22)
        )

        schedule = screen_schedule(config)
        self.assertEqual(len(schedule), 12)
        self.assertEqual(
            {(row["task_id"], row["solver_seed"]) for row in schedule},
            {(task, seed) for task in TASK_IDS for seed in (20, 21, 22)},
        )

        historical = config["historical_reset_qualification"]
        self.assertEqual(
            historical["role"],
            "registered_reset_validity_evidence_only_not_screen_or_ttf_reuse",
        )
        self.assertEqual(historical["required_exact_task_seed_rows"], 12)
        self.assertEqual(
            historical["registered_n800_controller_or_ttf_manifest_match_count"], 0
        )
        self.assertFalse(historical["outcome_fields_read_for_seed_selection"])
        for field in ("qualification_manifest", "qualification_report"):
            registered = historical[field]
            self.assertEqual(
                sha256_file(root / registered["path"]), registered["sha256"]
            )
        boundary = config["claim_boundary"]
        self.assertTrue(boundary["ttf_unseen_at_selection"])
        self.assertFalse(boundary["historical_reset_state_reused"])
        qualification_policy = config["screen_qualification_policy"]
        self.assertTrue(qualification_policy["fresh_reset"])
        self.assertTrue(
            qualification_policy[
                "historical_rows_used_only_to_freeze_candidate_seed_universe"
            ]
        )
        self.assertFalse(qualification_policy["historical_state_or_cache_reused"])
        self.assertEqual(
            (
                config["screen"]["wall_time_safety_fuse_seconds"],
                config["screen"]["environment_time_safety_fuse_seconds"],
                config["screen"]["episode_process_timeout_seconds"],
            ),
            (60.0, 60.0, 90.0),
        )

        invalid = config["v1_invalid_evidence"]
        for field in ("config", "status"):
            registered = invalid[field]
            self.assertEqual(
                sha256_file(root / registered["path"]), registered["sha256"]
            )
        invalid_status = read_json(root / invalid["status"]["path"])
        self.assertEqual(invalid_status["decision_status"], "INVALID")
        self.assertEqual(invalid_status["completed_schedule_entries"], 0)
        self.assertFalse(invalid_status["complete"])

        v1_plan = plan(CONFIG)
        v2_plan = plan(V2_CONFIG)
        self.assertEqual(v2_plan["schema"], V2_STATUS_SCHEMA)
        self.assertEqual(v2_plan["experiment_id"], V2_EXPERIMENT_ID)
        self.assertEqual(v2_plan["candidate_solver_seeds"], [20, 21, 22])
        self.assertEqual(v2_plan["screen"]["candidate_key_count"], 12)
        self.assertFalse(v2_plan["screen_ttf_or_outcome_ranking"])
        self.assertTrue(v2_plan["historically_reset_qualified"])
        self.assertTrue(v2_plan["screen_fresh_reset"])
        self.assertTrue(v2_plan["ttf_unseen_at_selection"])
        self.assertTrue(v2_plan["required_new_output"])
        self.assertNotEqual(v2_plan["schema"], v1_plan["schema"])
        self.assertNotEqual(v2_plan["experiment_id"], v1_plan["experiment_id"])
        self.assertNotEqual(
            v2_plan["screen"]["schedule_sha256"],
            v1_plan["screen"]["schedule_sha256"],
        )
        self.assertEqual(
            config["output_identity"],
            {"required_new_output": True, "previous_v1_output_resumable": False},
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused-v2-output"
            with patch(
                "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                "run_closed_loop_collection",
                side_effect=AssertionError("solver must not run"),
            ):
                dry = run_screen(V2_CONFIG, output, dry_run=True)
            self.assertEqual(dry["schema"], V2_STATUS_SCHEMA)
            self.assertEqual(dry["experiment_id"], V2_EXPERIMENT_ID)
            self.assertFalse(dry["solver_or_controller_invoked"])
            self.assertFalse(output.exists())

    def test_v2_run_screen_routes_through_fresh_output_qualification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = (Path(directory) / "new-v2-output").resolve()
            qualification = _screen_qualification_root(output)
            prepared = SimpleNamespace(
                completed_report=None,
                status={},
                resumed=False,
                base_status={"total_schedule_entries": 12},
            )
            with (
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "_producer",
                    return_value={"identity": "test"},
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "prepare_resumable_output",
                    return_value=prepared,
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "_reset_qualification",
                    return_value=qualification,
                ) as reset,
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "_run_screen_jobs",
                ) as run_jobs,
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "analyze_screen",
                    return_value={"decision_status": "PASS_STATE_SUPPLY"},
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "sha256_file",
                    return_value="report-sha",
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "write_json",
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "run_closed_loop_collection",
                    side_effect=AssertionError("solver must not run"),
                ),
            ):
                result = run_screen(V2_CONFIG, output)
            self.assertEqual(result["decision_status"], "PASS_STATE_SUPPLY")
            self.assertEqual(
                qualification,
                output / "screen" / "reset_qualification",
            )
            reset.assert_called_once()
            self.assertEqual(reset.call_args.kwargs["phase"], "screen")
            self.assertFalse(reset.call_args.kwargs["resume"])
            self.assertEqual(run_jobs.call_args.args[-1], qualification)
            self.assertFalse(output.exists())

    def test_v3_budget_corrected_overlay_identity_and_live_evidence(self) -> None:
        self.assertEqual(sha256_file(V3_CONFIG), V3_CONFIG_SHA256)
        _path, root, config = load_config(V3_CONFIG)
        self.assertEqual(config["experiment_id"], V3_EXPERIMENT_ID)
        self.assertEqual(config["_profile"]["profile"], "v3")
        self.assertEqual(
            config["scientific_status"],
            "post_v2_invalid_budget_corrected_preregistered_before_v3_controller",
        )
        self.assertEqual(len(screen_schedule(config)), 12)
        self.assertEqual(
            (
                config["screen"]["wall_time_safety_fuse_seconds"],
                config["screen"]["environment_time_safety_fuse_seconds"],
                config["screen"]["episode_process_timeout_seconds"],
            ),
            (200.0, 200.0, 300.0),
        )
        self.assertEqual(
            (
                config["formal"]["wall_time_budget_seconds"],
                config["formal"]["environment_time_limit_seconds"],
                config["formal"]["episode_process_timeout_seconds"],
            ),
            (200.0, 200.0, 300.0),
        )
        self.assertTrue(config["screen_qualification_policy"]["fresh_reset"])
        self.assertTrue(
            config["screen_qualification_policy"]["reset_failure_keys_retained"]
        )
        self.assertFalse(
            config["screen_qualification_policy"]
            ["post_reset_task_or_seed_filtering"]
        )
        self.assertFalse(
            config["screen_qualification_policy"]
            ["diagnostic_state_or_cache_reused"]
        )
        for evidence_name in ("v2_invalid_evidence", "budget_diagnostic_evidence"):
            evidence = config[evidence_name]
            fields = (
                ("config", "status")
                if evidence_name == "v2_invalid_evidence"
                else ("run_config", "qualification_manifest", "qualification_report")
            )
            for field in fields:
                registered = evidence[field]
                self.assertEqual(
                    sha256_file(root / registered["path"]), registered["sha256"]
                )
        invalid_status = read_json(
            root / config["v2_invalid_evidence"]["status"]["path"]
        )
        self.assertEqual(invalid_status["decision_status"], "INVALID")
        self.assertEqual(invalid_status["completed_schedule_entries"], 0)
        diagnostic_report = read_json(
            root
            / config["budget_diagnostic_evidence"]["qualification_report"]["path"]
        )
        self.assertTrue(diagnostic_report["passed"])
        self.assertEqual(diagnostic_report["valid_count"], 4)
        self.assertEqual(diagnostic_report["incomplete_reset_count"], 0)

        result = plan(V3_CONFIG)
        self.assertEqual(result["schema"], V3_STATUS_SCHEMA)
        self.assertEqual(result["experiment_id"], V3_EXPERIMENT_ID)
        self.assertEqual(result["screen"]["candidate_key_count"], 12)
        self.assertEqual(
            (
                result["screen"]["wall_time_safety_fuse_seconds"],
                result["screen"]["environment_time_safety_fuse_seconds"],
                result["screen"]["episode_process_timeout_seconds"],
            ),
            (200.0, 200.0, 300.0),
        )
        self.assertEqual(result["formal"]["maximum_registered_ttf_seconds"], 1600.0)
        self.assertTrue(result["budget_corrected_profile"])
        self.assertTrue(result["screen_fresh_reset"])
        self.assertFalse(result["diagnostic_state_or_cache_reused"])
        self.assertFalse(result["post_reset_task_or_seed_filtering"])

    def test_v3_run_screen_fresh_qualifies_all_twelve_keys(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = (Path(directory) / "new-v3-output").resolve()
            qualification = _screen_qualification_root(output)
            prepared = SimpleNamespace(
                completed_report=None,
                status={},
                resumed=False,
                base_status={"total_schedule_entries": 12},
            )
            with (
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "_producer",
                    return_value={"identity": "test"},
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "prepare_resumable_output",
                    return_value=prepared,
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "_reset_qualification",
                    return_value=qualification,
                ) as reset,
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "_run_screen_jobs",
                ) as run_jobs,
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "analyze_screen",
                    return_value={"decision_status": "PASS_STATE_SUPPLY"},
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "sha256_file",
                    return_value="report-sha",
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "write_json",
                ),
                patch(
                    "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                    "run_closed_loop_collection",
                    side_effect=AssertionError("solver must not run"),
                ),
            ):
                result = run_screen(V3_CONFIG, output)
            self.assertEqual(result["decision_status"], "PASS_STATE_SUPPLY")
            reset.assert_called_once()
            self.assertEqual(reset.call_args.kwargs["phase"], "screen")
            self.assertFalse(reset.call_args.kwargs["resume"])
            self.assertEqual(len(reset.call_args.kwargs["keys"]), 12)
            self.assertEqual(run_jobs.call_args.args[-1], qualification)
            self.assertFalse(output.exists())

    def test_latin_priority_selects_one_key_per_task_without_backfill(self) -> None:
        candidates = [
            {
                "task_id": task,
                "solver_seed": seed,
                "eligible": True,
            }
            for task in TASK_IDS
            for seed in CANDIDATE_SEEDS
        ]
        selected, missing = select_screen_keys(candidates)
        self.assertEqual(missing, [])
        self.assertEqual(
            [(row["task_id"], row["solver_seed"]) for row in selected],
            list(zip(TASK_IDS, CANDIDATE_SEEDS)),
        )

        candidates = [
            row
            for row in candidates
            if not (
                row["task_id"] == TASK_IDS[1]
                and row["solver_seed"] == CANDIDATE_SEEDS[1]
            )
        ]
        selected, missing = select_screen_keys(candidates)
        self.assertEqual(missing, [])
        self.assertEqual(
            next(row["solver_seed"] for row in selected if row["task_id"] == TASK_IDS[1]),
            CANDIDATE_SEEDS[2],
        )

        candidates = [row for row in candidates if row["task_id"] != TASK_IDS[3]]
        selected, missing = select_screen_keys(candidates)
        self.assertEqual(missing, [TASK_IDS[3]])
        self.assertEqual({row["task_id"] for row in selected}, set(TASK_IDS[:3]))

    def test_formal_schedule_is_same_selected_key_and_at_most_eight_episodes(self) -> None:
        _path, _root, config = load_config(CONFIG)
        selected = [
            {"task_id": task, "solver_seed": seed, "eligible": True}
            for task, seed in zip(TASK_IDS, CANDIDATE_SEEDS)
        ]
        schedule = formal_schedule(config, selected)
        self.assertLessEqual(len(schedule), 8)
        self.assertEqual(len(schedule), 2 * len(TASK_IDS))
        for task, seed in zip(TASK_IDS, CANDIDATE_SEEDS):
            pair = [row for row in schedule if row["task_id"] == task]
            self.assertEqual(len(pair), 2)
            self.assertEqual({row["solver_seed"] for row in pair}, {seed})
            self.assertEqual({row["controller"] for row in pair}, set(CONTROLLERS))

    def test_first_offer_can_occur_early_or_at_the_k64_boundary(self) -> None:
        self.assertEqual(PREFIX_V2_OUTCOMES, 64)
        self.assertEqual(MAXIMUM_SCREEN_TRACE_DECISIONS, 65)

        early = [_screen_decision(index) for index in range(12)]
        early[7] = _screen_decision(7, offer=True)
        for row in early[8:]:
            row["controller"]["v2_first_consensus_rescue"]["selection"][
                "selection_phase"
            ] = "v2_only_after_offer"
        early_certificate = _screen_preaction_certificate(early)
        self.assertTrue(early_certificate["eligible"])
        self.assertEqual(early_certificate["trigger_outcome_count"], 7)
        self.assertEqual(early_certificate["offer_decision_index"], 7)

        outcome_mutation = copy.deepcopy(early)
        for row in outcome_mutation[7:]:
            row["success"] = not bool(row["success"])
            row["capped_wall_time_to_feasible"] = -123456.0
            row["final_conflicts"] = 987654
            row["pp_replan_seconds"] = 999999.0
            row["actual_action"] = {"forbidden_for_selection": True}
        self.assertEqual(
            _screen_preaction_certificate(outcome_mutation), early_certificate
        )

        invalid_prefix = copy.deepcopy(early)
        invalid_prefix[2]["controller"]["v2_first_consensus_rescue"]["selection"][
            "selection_phase"
        ] = "v2_only_after_offer"
        with self.assertRaisesRegex(ValueError, "before its first rescue offer"):
            _screen_preaction_certificate(invalid_prefix)

        decisions = [_screen_decision(index) for index in range(65)]
        decisions[64] = _screen_decision(64, offer=True)
        certificate = _screen_preaction_certificate(decisions)
        self.assertTrue(certificate["eligible"])
        self.assertEqual(certificate["trigger_outcome_count"], 64)
        self.assertEqual(certificate["trigger_completion_decision_index"], 63)
        self.assertEqual(certificate["offer_decision_index"], 64)
        self.assertTrue(certificate["selection_fields_only"])

        with self.assertRaisesRegex(ValueError, "K64 plus"):
            _screen_preaction_certificate(decisions + [_screen_decision(65)])

    def test_generation_count_is_the_unique_family_candidate_id_count(self) -> None:
        exact = [_screen_decision(index) for index in range(4)]
        exact[3] = _screen_decision(3, offer=True)
        self.assertEqual(
            exact[3]["controller"]["v2_first_consensus_rescue"]["generation"]
            ["generated_candidate_count"],
            1,
        )
        self.assertTrue(_screen_preaction_certificate(exact)["eligible"])

        distinct = copy.deepcopy(exact)
        generation = distinct[3]["controller"]["v2_first_consensus_rescue"][
            "generation"
        ]
        generation["hotspot_candidate_id"] = "hotspot16"
        generation["hotspot_agents"] = [3, 5, 9]
        generation["exact_agent_consensus"] = False
        generation["consensus_candidate_id"] = None
        generation["candidate_id"] = None
        generation["available"] = False
        generation["challenger_present"] = False
        generation["fallback"] = "fresh_v2_after_no_consensus"
        generation["generated_candidate_count"] = 2
        certificate = _screen_preaction_certificate(distinct)
        self.assertTrue(certificate["triggered_within_k"])
        self.assertFalse(certificate["eligible"])

    def test_dry_runs_do_not_call_solver_or_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            with patch(
                "experiments.stride_v2first_consensus16_warehouse_n800_supply."
                "run_closed_loop_collection",
                side_effect=AssertionError("solver must not run"),
            ):
                screen = run_screen(CONFIG, output, dry_run=True)
                formal = run_formal(CONFIG, output, dry_run=True)
                combined = run(CONFIG, output, dry_run=True)
            self.assertFalse(screen["solver_or_controller_invoked"])
            self.assertFalse(formal["solver_or_controller_invoked"])
            self.assertFalse(combined["solver_or_controller_invoked"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
