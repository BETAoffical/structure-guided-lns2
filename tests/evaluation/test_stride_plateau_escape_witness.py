from __future__ import annotations

import unittest

from experiments.stride_plateau_escape_witness import (
    _repairability_hazard_diagnostic,
    _trigger_pool_escape_opportunity,
    build_plateau_witness,
    identify_trigger_plateau,
)


def _transition(
    decision: int,
    before: int,
    after: int,
    candidate: str,
    agents: list[int],
    order: list[int],
    *,
    structural: bool = True,
    changed: bool = False,
) -> dict:
    before_edges = {(0, 1)} if before else set()
    after_edges = {(0, 1)} if after else set()
    return {
        "decision_index": decision,
        "candidate_id": candidate,
        "selected_agents": agents,
        "selected_structural": structural,
        "repair_order": order,
        "pp_random_seed": decision + 100,
        "replan_success": changed,
        "conflicts_before": before,
        "conflicts_after": after,
        "before_edges": before_edges,
        "after_edges": after_edges,
        "changed_agent_count": int(changed),
        "exact_repair_noop": not changed and before_edges == after_edges,
    }


def _case() -> dict:
    return {
        "case_id": "case",
        "state_id": "state",
        "map_id": "map",
        "task_id": "task",
        "solver_seed": 1,
        "challenger": "v2-plus-structpool",
        "treatment_policy": "struct-then-struct",
        "classification": "adverse",
        "first_structural_decision": 0,
        "first_repeat_stall_decision": 3,
        "trace_sha256": "trace",
    }


def _causal(root_cause: str = "set_defect") -> dict:
    return {
        "root_cause": root_cause,
        "state_fingerprint": "fingerprint",
        "external_blocker_union": [2, 3],
        "failed_agent_union": [1],
        "baseline_replan_success_rate": 0.25,
        "baseline_mean_external_blocker_count": 2.0,
    }


def _finish(success: bool = True) -> dict:
    return {
        "success": success,
        "summary": {
            "stop_reason": "success" if success else "repair_limit",
            "repair_iterations": 10,
            "final_conflicts": 0 if success else 1,
        },
    }


class PlateauEscapeWitnessTests(unittest.TestCase):
    def test_identify_trigger_plateau_expands_both_directions(self) -> None:
        rows = [
            _transition(0, 3, 1, "entry", [0, 1], [0, 1]),
            _transition(1, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(2, 1, 1, "repeat", [0, 1], [1, 0]),
            _transition(3, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(4, 1, 1, "other", [0, 2], [0, 2]),
            _transition(5, 1, 0, "escape", [0, 2], [2, 0]),
        ]
        self.assertEqual(identify_trigger_plateau(rows, 3), (1, 4))

    def test_set_change_escape_records_added_blocker(self) -> None:
        rows = [
            _transition(0, 3, 1, "entry", [0, 1], [0, 1]),
            _transition(1, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(2, 1, 1, "repeat", [0, 1], [1, 0]),
            _transition(3, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(4, 1, 0, "escape", [0, 1, 2], [2, 0, 1]),
        ]
        witness = build_plateau_witness(
            rows,
            case=_case(),
            causal=_causal(),
            finish=_finish(),
            minimum_pre_trigger_flat_length=2,
        )
        self.assertEqual(witness["plateau"]["action_count"], 3)
        self.assertEqual(witness["exit"]["outcome"], "strict_reduction")
        self.assertTrue(witness["exit"]["set_change"])
        self.assertEqual(witness["exit"]["added_agents"], [2])
        self.assertEqual(witness["exit"]["agent_set"], [0, 1, 2])
        self.assertEqual(witness["exit"]["added_external_blockers"], [2])
        self.assertEqual(witness["plateau"]["unique_pp_seed_count"], 3)

    def test_same_candidate_same_set_can_escape_with_new_order(self) -> None:
        rows = [
            _transition(0, 3, 1, "entry", [0, 1], [0, 1]),
            _transition(1, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(2, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(3, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(4, 1, 0, "repeat", [0, 1], [1, 0]),
        ]
        witness = build_plateau_witness(
            rows,
            case=_case(),
            causal=_causal("order_defect"),
            finish=_finish(),
            minimum_pre_trigger_flat_length=2,
        )
        self.assertTrue(witness["exit"]["same_candidate"])
        self.assertTrue(witness["exit"]["same_agent_set"])
        self.assertTrue(witness["exit"]["order_changed_from_last_attempt"])

    def test_posthoc_diagnostics_separate_hazard_and_pool_availability(self) -> None:
        rows = [
            _transition(0, 3, 1, "entry", [0, 1], [0, 1]),
            _transition(1, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(2, 1, 1, "repeat", [0, 1], [1, 0]),
            _transition(3, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(4, 1, 0, "escape", [0, 1, 2], [2, 0, 1]),
        ]
        slow = build_plateau_witness(
            rows,
            case=_case(),
            causal={**_causal(), "baseline_replan_success_rate": 0.0},
            finish=_finish(),
            minimum_pre_trigger_flat_length=2,
        )
        fast = build_plateau_witness(
            rows[:3] + [_transition(3, 1, 0, "repeat", [0, 1], [1, 0])],
            case=_case(),
            causal={**_causal(), "baseline_replan_success_rate": 1.0},
            finish=_finish(),
            minimum_pre_trigger_flat_length=2,
        )
        fast["case_id"] = "fast"
        diagnostic = _repairability_hazard_diagnostic([slow, fast])
        self.assertEqual(
            diagnostic["correlation"]["all"]["spearman_plateau_length"]
            ["baseline_replan_success_rate"],
            -1.0,
        )
        opportunity = _trigger_pool_escape_opportunity(
            [slow],
            {
                ("case", "first_repeat_stall"): {
                    "candidate_pool": [
                        {
                            "candidate_id": "escape",
                            "agents": [0, 1, 2],
                        }
                    ]
                }
            },
        )
        self.assertEqual(opportunity["exact_exit_agent_set_present_count"], 1)
        self.assertEqual(opportunity["exact_exit_candidate_id_present_count"], 1)

    def test_trace_end_is_right_censored(self) -> None:
        rows = [
            _transition(0, 3, 1, "entry", [0, 1], [0, 1]),
            _transition(1, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(2, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(3, 1, 1, "repeat", [0, 1], [0, 1]),
        ]
        witness = build_plateau_witness(
            rows,
            case=_case(),
            causal=_causal(),
            finish=_finish(False),
            minimum_pre_trigger_flat_length=2,
        )
        self.assertEqual(witness["exit"]["outcome"], "right_censored")
        self.assertFalse(witness["episode"]["success"])

    def test_trigger_action_itself_can_be_escape_witness(self) -> None:
        rows = [
            _transition(0, 3, 1, "entry", [0, 1], [0, 1]),
            _transition(1, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(2, 1, 1, "repeat", [0, 1], [0, 1]),
            _transition(3, 1, 0, "repeat", [0, 1], [1, 0]),
        ]
        witness = build_plateau_witness(
            rows,
            case=_case(),
            causal=_causal("order_defect"),
            finish=_finish(),
            minimum_pre_trigger_flat_length=2,
        )
        self.assertEqual(witness["plateau"]["action_count"], 2)
        self.assertEqual(witness["exit"]["decision_index"], 3)
        self.assertTrue(witness["exit"]["same_candidate"])
        self.assertTrue(witness["exit"]["order_changed_from_last_attempt"])


if __name__ == "__main__":
    unittest.main()
