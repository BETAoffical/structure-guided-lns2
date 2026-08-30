from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.stride_v2first_family_rescue_g1 import (
    CONTROLLERS,
    EXPERIMENT_ID,
    RESCUE_ARMS,
    SOLVER_SEED,
    _load_trusted_batch_a,
    _pre_intervention_parity,
    _producer,
    _rescue_event,
    _v2_first_trace_audit,
    controller_kwargs,
    evaluate_batch_a_arm,
    evaluate_map_gate,
    load_config,
    plan,
    run_batch_b,
    schedule,
)
from experiments._common import sha256_file, write_json


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_v2first_family_rescue_g1_v1.json"


def _trace_row(index: int, *, agent: int = 7) -> dict:
    return {
        "decision_index": index,
        "before_platform_signature": f"before-{index}",
        "after_platform_signature": f"after-{index}",
        "before_conflicts": 20 - index,
        "after_conflicts": 19 - index,
        "actual_action": {
            "mode": "explicit_neighborhood",
            "agents": [agent],
            "random_seed": 100 + index,
        },
        "actual_metrics": {
            "requested_pp_random_seed": -1,
            "applied_pp_random_seed": -1,
            "repair_order": [agent],
        },
        "controller": {"selected_candidate_id": f"candidate-{index}"},
    }


def _auditable_pair(*, unavailable: bool = False) -> tuple[list[dict], list[dict]]:
    baseline = [_trace_row(index) for index in range(6)]
    for index, row in enumerate(baseline):
        row["controller"].update(
            {
                "repair_seed_draw_index": index,
                "candidate_pool": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "agents": [7],
                        "actual_size": 1,
                        "selection_families": ["collision:16"],
                        "retained": True,
                        "score": 1.0,
                    }
                ],
            }
        )
        row["actual_metrics"]["requested_random_seed"] = 100 + index
        if index <= 3:
            row["before_platform_signature"] = "platform-a"
            row["after_platform_signature"] = "platform-a"
            row["before_conflicts"] = row["after_conflicts"] = 10
            row["actual_metrics"].update(
                {
                    "pp_failure_reason": "conflict_bound_exceeded",
                    "replan_success": False,
                    "pp_rolled_back": True,
                }
            )
    challenger = copy.deepcopy(baseline)
    for index, row in enumerate(challenger):
        if index < 3:
            selection_phase = "v2_only"
            offered = False
            generation = execution = None
            observation = {
                "decision_index": index,
                "decision_mode": "v2_only",
                "before_repair_fingerprint": "platform-a",
                "after_repair_fingerprint": "platform-a",
                "v2_exact_conflict_bound_rollback": True,
                "consecutive_v2_exact_rollbacks": index + 1,
                "rescue_scheduled_for_next_decision": index == 2,
            }
            row["controller"]["proposal"] = {
                "v2_first_rescue_mode": "v2_only"
            }
        elif index == 3:
            selection_phase = "single_family_rescue_due"
            offered = True
            generation = {
                "attempted": True,
                "offered": True,
                "consumed": True,
                "available": not unavailable,
                "challenger_present": not unavailable,
                "structural_selected": False,
                "candidate_id": None if unavailable else "component16",
                "decision_index": 3,
                "before_repair_fingerprint": "platform-a",
            }
            if unavailable:
                execution = None
                observation = {
                    "decision_index": 3,
                    "decision_mode": "v2_only",
                    "before_repair_fingerprint": "platform-a",
                    "after_repair_fingerprint": "platform-a",
                    "v2_exact_conflict_bound_rollback": True,
                }
                row["controller"]["proposal"] = {
                    "v2_first_rescue_mode": "fresh_v2_after_unavailable"
                }
            else:
                execution = {
                    "offered": True,
                    "consumed": True,
                    "challenger_present": True,
                    "structural_selected": True,
                    "candidate_id": "component16",
                }
                observation = {
                    "decision_index": 3,
                    "decision_mode": "single_family_rescue",
                    "before_repair_fingerprint": "platform-a",
                    "after_repair_fingerprint": "platform-b",
                    "exact_conflict_bound_rollback": False,
                }
                candidate = {
                    "candidate_id": "component16",
                    "agents": [0, 1, 2, 3],
                    "actual_size": 4,
                    "selection_families": [
                        "structpool-conflict-component:16"
                    ],
                    "structpool_family_groups": ["conflict_component"],
                    "proposal_count_by_family": {
                        "structpool-conflict-component:16": 1
                    },
                    "selection_rank_by_family": {
                        "structpool-conflict-component:16": 0
                    },
                    "hybridstructpool_provenance": [
                        "structshell_equal_four_size",
                        "v2_first_single_family_rescue",
                    ],
                    "retained": True,
                    "score": 0.0,
                }
                row["controller"].update(
                    {
                        "selected_candidate_id": "component16",
                        "candidate_pool": [candidate],
                        "inference_seconds": 0.0,
                    }
                )
                row["actual_action"]["agents"] = [0, 1, 2, 3]
                row["actual_metrics"].update(
                    {
                        "pp_failure_reason": "none",
                        "replan_success": True,
                        "pp_rolled_back": False,
                        "repair_order": [3, 2, 1, 0],
                    }
                )
                row["after_platform_signature"] = "platform-b"
                row["after_conflicts"] = 9
        else:
            selection_phase = "v2_only_after_offer"
            offered = False
            generation = execution = None
            observation = {
                "decision_index": index,
                "decision_mode": "v2_only",
                "before_repair_fingerprint": row["before_platform_signature"],
                "after_repair_fingerprint": row["after_platform_signature"],
                "v2_exact_conflict_bound_rollback": False,
            }
            row["controller"]["proposal"] = {
                "v2_first_rescue_mode": "v2_only_after_offer"
            }
        selection = {
            "decision_index": index,
            "before_repair_fingerprint": row["before_platform_signature"],
            "selection_phase": selection_phase,
            "offered": offered,
            "structural_profile": "conflict_component",
            "nominal_size": 16,
            "consecutive_v2_exact_rollbacks": min(index, 3),
        }
        row["controller"]["v2_first_single_family_rescue"] = {
            "selection": selection,
            "generation": generation,
            "execution": execution,
            "observation": observation,
        }
    return baseline, challenger


class V2FirstFamilyRescueG1Test(unittest.TestCase):
    def test_plan_freezes_fresh_seed_stages_and_worker_boundary(self) -> None:
        _path, _root, config = load_config(CONFIG)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        batch_a = schedule(config, "batch_a")
        batch_b = schedule(config, "batch_b")
        self.assertEqual(len(batch_a), 9)
        self.assertEqual(len(batch_b), 3)
        self.assertEqual({row["solver_seed"] for row in batch_a + batch_b}, {26})
        self.assertEqual(
            [row["controller"] for row in batch_a if row["within_key_position"] == 0],
            ["v2_only", "component16_rescue", "hotspot16_rescue"],
        )
        self.assertEqual(
            [row["controller"] for row in batch_b], list(CONTROLLERS)
        )
        result = plan(CONFIG)
        self.assertEqual(result["solver_seed"], SOLVER_SEED)
        self.assertEqual(result["qualification"]["workers"], 16)
        self.assertFalse(result["qualification"]["included_in_ttf"])
        self.assertEqual(result["timed_episodes"]["workers"], 1)
        self.assertTrue(result["timed_episodes"]["strict_serial"])
        self.assertTrue(result["timed_episodes"]["included_in_ttf"])
        self.assertEqual(result["maximum_all_stage_ttf_seconds"], 2400.0)
        self.assertFalse(result["solver_or_controller_invoked"])

    def test_controller_wiring_is_v2_or_one_direct_family_only(self) -> None:
        _path, root, config = load_config(CONFIG)
        v2 = controller_kwargs(root, config, "v2_only")
        component = controller_kwargs(root, config, "component16_rescue")
        hotspot = controller_kwargs(root, config, "hotspot16_rescue")
        self.assertNotIn("hybridstructpool_augmentation", v2)
        for kwargs, profile, family in (
            (component, "conflict_component", "conflict_component"),
            (hotspot, "hotspot", "spatiotemporal_hotspot"),
        ):
            augmentation = kwargs["hybridstructpool_augmentation"]
            self.assertEqual(augmentation["structural_profile"], profile)
            self.assertEqual(
                augmentation["runtime_structural_family_sizes"], {family: [16]}
            )
            self.assertEqual(augmentation["maximum_added_candidates"], 1)
            self.assertEqual(augmentation["maximum_total_candidates"], 1)
            self.assertEqual(
                augmentation["v2_first_rescue"]["selection"],
                "direct_unique_structural_candidate",
            )
            self.assertEqual(
                augmentation["v2_first_rescue"][
                    "maximum_rescue_offers_per_episode"
                ],
                1,
            )
            self.assertTrue(
                augmentation["v2_first_rescue"]["permanent_v2_after_offer"]
            )

    def test_map_and_batch_gate_are_independent_per_arm(self) -> None:
        gates = load_config(CONFIG)[2]["performance_gates"]["batch_a"]
        rows = []
        for index in range(3):
            map_gate = evaluate_map_gate(
                baseline_success=True,
                baseline_ttf=100.0,
                challenger_success=True,
                challenger_ttf=95.0,
            )
            rows.append(
                {
                    "baseline_success": True,
                    "challenger_success": True,
                    "baseline_ttf": 100.0,
                    "challenger_ttf": 95.0,
                    "map_gate": map_gate,
                    "rescue": {
                        "executed": index < 2,
                        "unavailable": False,
                        "repeat_exact_rollback": False if index < 2 else None,
                    },
                    "baseline_counterfactual": (
                        {"repeat_exact_rollback": True} if index < 2 else None
                    ),
                }
            )
        passed = evaluate_batch_a_arm(rows, gates)
        self.assertTrue(passed["passed"])
        self.assertEqual(passed["rescue_triggered_map_count"], 2)
        self.assertEqual(
            passed["post_trigger_repeat_rollback_rate_reduction"], 1.0
        )

        one_trigger = copy.deepcopy(rows)
        one_trigger[1]["rescue"] = {
            "executed": False,
            "unavailable": True,
            "repeat_exact_rollback": None,
        }
        one_trigger[1]["baseline_counterfactual"] = None
        failed = evaluate_batch_a_arm(one_trigger, gates)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["checks"]["minimum_triggered_maps"])
        self.assertEqual(set(RESCUE_ARMS), {
            "component16_rescue", "hotspot16_rescue"
        })

    def test_pre_intervention_parity_includes_candidate_action_and_pp_order(self) -> None:
        baseline = [_trace_row(index) for index in range(4)]
        challenger = copy.deepcopy(baseline)
        self.assertTrue(
            _pre_intervention_parity(baseline, challenger, 3)["passed"]
        )
        challenger[1]["actual_metrics"]["repair_order"] = [99]
        mismatch = _pre_intervention_parity(baseline, challenger, 3)
        self.assertFalse(mismatch["passed"])
        self.assertEqual(mismatch["first_mismatch_decision"], 1)

    def test_trace_audit_distinguishes_executed_and_unavailable_consumed_offer(self) -> None:
        executed = _trace_row(3)
        executed["before_platform_signature"] = "same"
        executed["after_platform_signature"] = "same"
        executed["before_conflicts"] = executed["after_conflicts"] = 10
        executed["actual_metrics"].update(
            {
                "pp_failure_reason": "conflict_bound_exceeded",
                "replan_success": False,
                "pp_rolled_back": True,
            }
        )
        executed["controller"]["v2_first_single_family_rescue"] = {
            "selection": {
                "selection_phase": "single_family_rescue_due",
                "offered": True,
                "consumed": False,
            },
            "generation": {
                "attempted": True,
                "offered": True,
                "challenger_present": True,
                "consumed": True,
            },
            "execution": {
                "structural_selected": True,
                "challenger_present": True,
                "consumed": True,
            },
            "observation": {"exact_conflict_bound_rollback": True},
        }
        with patch(
            "experiments.stride_v2first_family_rescue_g1._decision_rows",
            return_value=[executed],
        ):
            event = _rescue_event(Path("unused"), {})
        self.assertTrue(event["offered"])
        self.assertTrue(event["offer_consumed"])
        self.assertTrue(event["executed"])
        self.assertFalse(event["unavailable"])
        self.assertTrue(event["repeat_exact_rollback"])

        unavailable = copy.deepcopy(executed)
        unavailable["controller"]["v2_first_single_family_rescue"] = {
            "selection": {
                "selection_phase": "single_family_rescue_due",
                "offered": True,
                "consumed": False,
            },
            "generation": {
                "attempted": True,
                "offered": True,
                "challenger_present": False,
                "available": False,
                "consumed": True,
            },
            "execution": None,
            "observation": {"decision_mode": "v2_only"},
        }
        with patch(
            "experiments.stride_v2first_family_rescue_g1._decision_rows",
            return_value=[unavailable],
        ):
            event = _rescue_event(Path("unused"), {})
        self.assertTrue(event["offer_consumed"])
        self.assertFalse(event["executed"])
        self.assertTrue(event["unavailable"])
        self.assertIsNone(event["repeat_exact_rollback"])

    def test_linear_trace_audit_locks_full_rescue_state_machine(self) -> None:
        baseline, executed = _auditable_pair()
        result = _v2_first_trace_audit(
            executed, baseline, "conflict_component"
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["trigger_exact_rollback_count"], 3)
        self.assertTrue(result["executed"])
        self.assertTrue(result["baseline_counterfactual"]["seed_parity"])

        _baseline, unavailable = _auditable_pair(unavailable=True)
        result = _v2_first_trace_audit(
            unavailable, baseline, "conflict_component"
        )
        self.assertTrue(result["unavailable"])
        self.assertFalse(result["executed"])

        for label, mutation, message in (
            (
                "missing transition field",
                lambda rows: rows[1]["controller"][
                    "v2_first_single_family_rescue"
                ].__setitem__("observation", None),
                "incomplete",
            ),
            (
                "post offer reopened",
                lambda rows: rows[4]["controller"][
                    "v2_first_single_family_rescue"
                ]["selection"].__setitem__("selection_phase", "v2_only"),
                "permanent V2-only",
            ),
            (
                "wrong family",
                lambda rows: rows[3]["controller"]["candidate_pool"][0].__setitem__(
                    "selection_families", ["structpool-spatiotemporal-hotspot:16"]
                ),
                "family/size/provenance",
            ),
            (
                "unpaired seed",
                lambda rows: rows[3]["actual_action"].__setitem__(
                    "random_seed", 999
                ),
                "random or PP seed",
            ),
        ):
            with self.subTest(label=label):
                changed = copy.deepcopy(executed)
                mutation(changed)
                with self.assertRaisesRegex(ValueError, message):
                    _v2_first_trace_audit(
                        changed, baseline, "conflict_component"
                    )

        changed_unavailable = copy.deepcopy(unavailable)
        changed_unavailable[3]["controller"]["candidate_pool"][0][
            "candidate_id"
        ] = "changed-v2"
        with self.assertRaisesRegex(ValueError, "fresh V2 parity"):
            _v2_first_trace_audit(
                changed_unavailable, baseline, "conflict_component"
            )

    def test_batch_b_is_blocked_without_a_passing_batch_a_report(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "experiments.stride_v2first_family_rescue_g1._producer",
                return_value={"native_required": False},
            ),
        ):
            result = run_batch_b(CONFIG, directory)
        self.assertTrue(result["blocked"])
        self.assertEqual(result["terminal_failure"], "batch_a_trust_failed")

    def test_batch_a_trust_chain_rejects_report_hash_tamper(self) -> None:
        producer = _producer(ROOT, native_required=False)
        config_hash = sha256_file(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            stage = output / "batch_a"
            report_path = stage / "batch_a_report.json"
            status_path = stage / "collection_status.json"
            report = {
                "schema": (
                    "lns2.stride.v2first_family_rescue_g1_batch_a_report.v1"
                ),
                "batch_a_passed": True,
                "completed_group_count": 3,
                "cancelled_group_ids": [],
                "inputs": {"config_sha256": config_hash},
                "producer_identity": producer,
            }
            write_json(report_path, report)
            status = {
                "schema": "lns2.stride.v2first_family_rescue_g1_status.v1",
                "config_sha256": config_hash,
                "total_schedule_entries": 9,
                "completed_schedule_entries": 9,
                "complete": True,
                "producer_identity": producer,
                "report_sha256": sha256_file(report_path),
                "run_fingerprint": "trusted-batch-a",
            }
            write_json(status_path, status)
            loaded, loaded_status = _load_trusted_batch_a(
                CONFIG, ROOT, output, producer=producer
            )
            self.assertTrue(loaded["batch_a_passed"])
            self.assertEqual(loaded_status["run_fingerprint"], "trusted-batch-a")

            report["batch_a_passed"] = False
            write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "completed report changed"):
                _load_trusted_batch_a(
                    CONFIG, ROOT, output, producer=producer
                )

    def test_config_tamper_is_rejected(self) -> None:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["runtime"]["timed_episodes"]["workers"] = 16
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "identity changed"):
                load_config(changed)


if __name__ == "__main__":
    unittest.main()
