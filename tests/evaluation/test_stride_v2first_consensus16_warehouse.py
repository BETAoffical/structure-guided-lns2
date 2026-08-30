from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.stride_v2first_consensus16_warehouse import (
    CONTROLLERS,
    EXPERIMENT_ID,
    TASK_IDS,
    _consensus_trace_audit,
    controller_kwargs,
    evaluate_combined,
    evaluate_stage_a,
    load_config,
    plan,
    run_stage_a,
    run_stage_b,
    schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_v2first_consensus16_warehouse_v1.json"


def _trace_row(index: int) -> dict:
    return {
        "decision_index": index,
        "before_platform_signature": f"before-{index}",
        "after_platform_signature": f"after-{index}",
        "before_conflicts": 20 - index,
        "after_conflicts": 19 - index,
        "actual_action": {
            "mode": "explicit_neighborhood",
            "agents": [7],
            "random_seed": 100 + index,
        },
        "actual_metrics": {
            "requested_random_seed": 100 + index,
            "requested_pp_random_seed": -1,
            "applied_pp_random_seed": -1,
            "repair_order": [7],
        },
        "controller": {
            "selected_candidate_id": f"candidate-{index}",
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
        },
    }


def _auditable_pair(*, exact_consensus: bool) -> tuple[list[dict], list[dict]]:
    baseline = [_trace_row(index) for index in range(6)]
    for index, row in enumerate(baseline):
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
            phase = "v2_only"
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
            row["controller"]["proposal"] = {"v2_first_rescue_mode": "v2_only"}
        elif index == 3:
            phase = "consensus_rescue_due"
            component = [0, 1, 2, 3]
            hotspot = component if exact_consensus else [0, 1, 2, 4]
            generation = {
                "attempted": True,
                "available": exact_consensus,
                "offered": True,
                "challenger_present": exact_consensus,
                "structural_selected": False,
                "consumed": True,
                "candidate_id": "consensus" if exact_consensus else None,
                "fallback": None if exact_consensus else "fresh_v2_after_no_consensus",
                "decision_index": 3,
                "before_repair_fingerprint": "platform-a",
                "component_candidate_id": (
                    "consensus" if exact_consensus else "component"
                ),
                "hotspot_candidate_id": (
                    "consensus" if exact_consensus else "hotspot"
                ),
                "component_agents": component,
                "hotspot_agents": hotspot,
                "component_available": True,
                "hotspot_available": True,
                "exact_agent_consensus": exact_consensus,
                "consensus_candidate_id": "consensus" if exact_consensus else None,
                "generated_candidate_count": 2,
            }
            if exact_consensus:
                execution = {
                    "executed": True,
                    "candidate_id": "consensus",
                    "offered": True,
                    "challenger_present": True,
                    "structural_selected": True,
                    "consumed": True,
                }
                observation = {
                    "decision_index": 3,
                    "decision_mode": "consensus_rescue",
                    "before_repair_fingerprint": "platform-a",
                    "after_repair_fingerprint": "platform-b",
                }
                row["controller"].update(
                    {
                        "selected_candidate_id": "consensus",
                        "candidate_pool": [
                            {
                                "candidate_id": "consensus",
                                "agents": component,
                                "actual_size": 4,
                                "selection_families": [
                                    "structpool-conflict-component:16",
                                    "structpool-spatiotemporal-hotspot:16",
                                ],
                                "structpool_family_groups": [
                                    "conflict_component",
                                    "spatiotemporal_hotspot",
                                ],
                                "hybridstructpool_provenance": [
                                    "structshell_equal_four_size",
                                    "v2_first_consensus_rescue",
                                ],
                                "retained": True,
                                "score": 0.0,
                            }
                        ],
                        "inference_seconds": 0.0,
                        "proposal": {"v2_first_rescue_mode": "consensus_rescue"},
                    }
                )
                row["actual_action"]["agents"] = component
                row["actual_metrics"].update(
                    {
                        "pp_failure_reason": "none",
                        "replan_success": True,
                        "pp_rolled_back": False,
                        "repair_order": list(reversed(component)),
                    }
                )
                row["after_platform_signature"] = "platform-b"
                row["after_conflicts"] = 9
            else:
                execution = None
                observation = {
                    "decision_index": 3,
                    "decision_mode": "v2_only",
                    "before_repair_fingerprint": "platform-a",
                    "after_repair_fingerprint": "platform-a",
                    "v2_exact_conflict_bound_rollback": True,
                }
                row["controller"]["proposal"] = {
                    "v2_first_rescue_mode": "fresh_v2_after_no_consensus"
                }
        else:
            phase = "v2_only_after_offer"
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
        row["controller"]["v2_first_consensus_rescue"] = {
            "selection": {
                "decision_index": index,
                "before_repair_fingerprint": row["before_platform_signature"],
                "selection_phase": phase,
                "offered": index == 3,
                "structural_profile": "component_hotspot_consensus",
                "nominal_size": 16,
                "consecutive_v2_exact_rollbacks": min(index, 3),
            },
            "generation": generation,
            "execution": execution,
            "observation": observation,
        }
    return baseline, challenger


def _result_row(
    *,
    task_id: str,
    variant: str,
    seed: int,
    executed: bool,
    baseline_ttf: float = 10.0,
    challenger_ttf: float = 9.0,
) -> dict:
    return {
        "task_id": task_id,
        "task_variant": variant,
        "solver_seed": seed,
        "baseline_success": True,
        "challenger_success": True,
        "baseline_ttf": baseline_ttf,
        "challenger_ttf": challenger_ttf,
        "key_gate": {
            "restricted_ttf_regression": (challenger_ttf - baseline_ttf)
            / baseline_ttf
        },
        "rescue": {
            "executed": executed,
            "repeat_exact_rollback": False if executed else None,
        },
        "baseline_counterfactual": (
            {"repeat_exact_rollback": True} if executed else None
        ),
    }


class WarehouseConsensus16RunnerTest(unittest.TestCase):
    def test_plan_freezes_two_stages_and_worker_boundary(self) -> None:
        _path, _root, config = load_config(CONFIG)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        stage_a = schedule(config, "stage_a")
        stage_b = schedule(config, "stage_b")
        self.assertEqual(len(stage_a), 8)
        self.assertEqual(len(stage_b), 8)
        self.assertEqual({row["solver_seed"] for row in stage_a}, {27})
        self.assertEqual({row["solver_seed"] for row in stage_b}, {28})
        self.assertEqual(
            [row["controller"] for row in stage_a[:4]],
            ["v2_only", "consensus16_rescue", "consensus16_rescue", "v2_only"],
        )
        for task_id in TASK_IDS:
            order_a = [
                row["controller"] for row in stage_a if row["task_id"] == task_id
            ]
            order_b = [
                row["controller"] for row in stage_b if row["task_id"] == task_id
            ]
            self.assertEqual(order_b, list(reversed(order_a)))
        result = plan(CONFIG)
        self.assertEqual(result["controllers"], list(CONTROLLERS))
        self.assertEqual(result["task_ids"], list(TASK_IDS))
        self.assertEqual(result["qualification"]["workers"], 16)
        self.assertEqual(result["timed_episodes"]["workers"], 1)
        self.assertTrue(result["timed_episodes"]["strict_serial"])
        self.assertEqual(result["maximum_all_stage_ttf_seconds"], 960.0)
        self.assertFalse(result["solver_or_controller_invoked"])

    def test_controller_wiring_is_v2_or_exact_consensus_only(self) -> None:
        _path, root, config = load_config(CONFIG)
        v2 = controller_kwargs(root, config, "v2_only")
        consensus = controller_kwargs(root, config, "consensus16_rescue")
        self.assertNotIn("hybridstructpool_augmentation", v2)
        augmentation = consensus["hybridstructpool_augmentation"]
        self.assertEqual(
            augmentation["source_mode"], "v2_first_consensus16_rescue"
        )
        self.assertEqual(augmentation["maximum_generated_candidates"], 2)
        self.assertEqual(augmentation["maximum_total_candidates"], 1)
        self.assertEqual(
            augmentation["v2_first_rescue"]["agreement"],
            "sorted_agent_set_exact_equality",
        )
        self.assertEqual(
            augmentation["v2_first_rescue"]["maximum_pp_calls_per_decision"], 1
        )

    def test_consensus_trace_audit_accepts_direct_and_fresh_v2_fallback(self) -> None:
        baseline, direct = _auditable_pair(exact_consensus=True)
        result = _consensus_trace_audit(direct, baseline)
        self.assertTrue(result["passed"])
        self.assertTrue(result["executed"])
        self.assertFalse(result["no_consensus"])
        baseline, fallback = _auditable_pair(exact_consensus=False)
        result = _consensus_trace_audit(fallback, baseline)
        self.assertTrue(result["passed"])
        self.assertFalse(result["executed"])
        self.assertTrue(result["no_consensus"])

    def test_stage_a_gate_marks_low_coverage_inconclusive(self) -> None:
        rows = [
            _result_row(
                task_id=task,
                variant="opposite_exchange" if "__oe__" in task else "uniform_random",
                seed=27,
                executed=index in {0, 2},
            )
            for index, task in enumerate(TASK_IDS)
        ]
        self.assertTrue(evaluate_stage_a(rows)["passed"])
        rows[2]["rescue"] = {"executed": False, "repeat_exact_rollback": None}
        rows[2]["baseline_counterfactual"] = None
        result = evaluate_stage_a(rows)
        self.assertFalse(result["passed"])
        self.assertEqual(result["decision_status"], "INCONCLUSIVE")

    def test_combined_gate_requires_both_seeds_variants_and_five_faster(self) -> None:
        rows = []
        for seed in (27, 28):
            for index, task in enumerate(TASK_IDS):
                rows.append(
                    _result_row(
                        task_id=task,
                        variant=(
                            "opposite_exchange" if "__oe__" in task else "uniform_random"
                        ),
                        seed=seed,
                        executed=(index in {0, 2}),
                    )
                )
        result = evaluate_combined(rows)
        self.assertTrue(result["passed"])
        self.assertEqual(result["paired_faster_key_count"], 8)
        self.assertEqual(result["executed_consensus_by_seed"], {"27": 2, "28": 2})

    def test_dry_runs_do_not_call_solver_or_create_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            with patch(
                "experiments.stride_v2first_consensus16_warehouse."
                "run_closed_loop_collection",
                side_effect=AssertionError("solver must not run"),
            ):
                stage_a = run_stage_a(CONFIG, output, dry_run=True)
                stage_b = run_stage_b(CONFIG, output, dry_run=True)
            self.assertFalse(stage_a["solver_or_controller_invoked"])
            self.assertFalse(stage_b["solver_or_controller_invoked"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
