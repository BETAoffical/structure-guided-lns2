from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.repair_collection import _read_json
from experiments.stride_safeslot_residual_teacher import (
    RISK_COMPONENTS,
    _action_aggregate,
    compare_residual_teacher,
    residual_risk_components,
    residual_structure_metrics,
    validate_safeslot_residual_teacher_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_safeslot_residual_teacher_v1.json"


def _state() -> dict[str, object]:
    return {
        "rows": 1,
        "cols": 3,
        "obstacles": [0, 0, 0],
        "agents": [
            {"id": 0, "start": 0, "goal": 1, "path": [0, 1]},
            {"id": 1, "start": 2, "goal": 1, "path": [2, 1]},
        ],
        "conflict_edges": [[0, 1]],
        "num_of_colliding_pairs": 1,
    }


def _trial(index: int, risk: float) -> dict[str, object]:
    return {
        "trial_index": index,
        "residual_risk": risk,
        "residual_risk_components": {name: risk for name in RISK_COMPONENTS},
    }


def _aggregate(
    mean: float, first: float, second: float, component: float
) -> dict[str, object]:
    return {
        "mean_residual_risk": mean,
        "first_fixed_half_mean_residual_risk": first,
        "second_fixed_half_mean_residual_risk": second,
        "component_mean_residual_risk": {
            name: component for name in RISK_COMPONENTS
        },
    }


class SafeSlotResidualTeacherTests(unittest.TestCase):
    def test_registered_config_freezes_teacher_only_boundary(self) -> None:
        config = _read_json(CONFIG)
        validate_safeslot_residual_teacher_config(config)
        self.assertFalse(config["claim_boundary"]["runtime_integration_allowed"])
        self.assertFalse(config["claim_boundary"]["formal_ttf_claim"])
        self.assertEqual(config["fixed_action_scope"]["total_trial_count"], 10960)

    def test_residual_structure_metrics_reconstruct_current_conflicts(self) -> None:
        metrics = residual_structure_metrics(_state())
        self.assertEqual(metrics["conflict_pair_count"], 1)
        self.assertEqual(metrics["conflict_event_count"], 1)
        self.assertEqual(metrics["active_conflict_agent_count"], 2)
        self.assertEqual(metrics["largest_conflict_component_size"], 2)
        self.assertEqual(metrics["bottleneck_conflict_event_count"], 1)
        self.assertEqual(metrics["repeated_conflict_event_excess"], 0)

    def test_residual_risk_uses_safe_denominator_for_zero_before(self) -> None:
        before = {
            "conflict_event_count": 4,
            "largest_conflict_component_size": 2,
            "bottleneck_conflict_event_count": 0,
            "repeated_conflict_event_excess": 0,
        }
        after = {
            "conflict_event_count": 2,
            "largest_conflict_component_size": 1,
            "bottleneck_conflict_event_count": 1,
            "repeated_conflict_event_excess": 2,
        }
        components = residual_risk_components(before, after)
        self.assertEqual(components["conflict_event_ratio_to_before"], 0.5)
        self.assertEqual(components["largest_component_ratio_to_before"], 0.5)
        self.assertEqual(components["bottleneck_event_ratio_to_before"], 1.0)
        self.assertEqual(components["repeated_event_excess_ratio_to_before"], 2.0)

    def test_action_aggregate_requires_exact_trial_matrix(self) -> None:
        action = {
            "candidate_id": "candidate-1",
            "candidate_role": "slotpool_structural_challenger",
            "agents": [0, 1],
        }
        aggregate = _action_aggregate(
            state_id="state-1",
            action=action,
            trials=[_trial(index, index / 10.0) for index in range(16)],
        )
        self.assertEqual(aggregate["trial_count"], 16)
        self.assertAlmostEqual(aggregate["mean_residual_risk"], 0.75)
        invalid = [_trial(index, 0.1) for index in range(16)]
        invalid[-1] = copy.deepcopy(invalid[-2])
        with self.assertRaisesRegex(ValueError, "trial matrix changed"):
            _action_aggregate(state_id="state-1", action=action, trials=invalid)

    def test_residual_teacher_requires_both_halves_and_component_safety(self) -> None:
        contract = _read_json(CONFIG)["final_label_contract"]
        anchor = _aggregate(1.0, 1.0, 1.0, 1.0)
        safe = compare_residual_teacher(
            _aggregate(0.9, 0.9, 0.9, 1.01), anchor, contract
        )
        self.assertTrue(safe["residual_teacher_pass"])
        unstable_half = compare_residual_teacher(
            _aggregate(0.9, 0.8, 1.01, 0.9), anchor, contract
        )
        self.assertFalse(unstable_half["residual_teacher_pass"])
        component_regression = compare_residual_teacher(
            _aggregate(0.9, 0.9, 0.9, 1.03), anchor, contract
        )
        self.assertFalse(component_regression["residual_teacher_pass"])


if __name__ == "__main__":
    unittest.main()
