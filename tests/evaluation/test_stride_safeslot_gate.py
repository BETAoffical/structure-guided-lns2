from __future__ import annotations

import unittest
from pathlib import Path

from experiments.repair_collection import _read_json
from experiments.stride_safeslot_gate import (
    FEATURE_NAMES,
    evaluate_gate_policy,
    safeslot_feature_vector,
    validate_safeslot_gate_config,
)
from experiments.stride_slotpool import BASE_FEATURE_NAMES, STATE_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_safeslot_gate_v1.json"


def _candidate(*, state_value: float, other_value: float) -> dict[str, object]:
    family = "structpool-boundary-articulation:24"
    return {
        "candidate_id": "challenger",
        "features": {
            name: state_value if name in STATE_FEATURE_NAMES else other_value
            for name in BASE_FEATURE_NAMES
        },
        "selection_families": [family],
        "structpool_support_count_by_family": {family: 8},
        "structpool_support_ratio_by_family": {family: 0.25},
        "v2_anchor_jaccard": 0.1,
    }


def _policy_row(
    state_id: str, candidate_id: str, *, label: bool, advantage: float
) -> dict[str, object]:
    return {
        "state_id": state_id,
        "map_id": f"map-{state_id}",
        "layout_mode": "topology",
        "anchor_candidate_id": f"anchor-{state_id}",
        "challenger_candidate_id": candidate_id,
        "label": label,
        "mean_current_step_advantage": advantage,
        "first_fixed_half_advantage": advantage,
        "second_fixed_half_advantage": advantage,
        "no_progress_rate_delta": -0.1 if label else 0.1,
        "mean_residual_risk_delta": -0.1 if label else 0.1,
    }


class SafeSlotGateTests(unittest.TestCase):
    def test_registered_config_freezes_map_grouped_offline_boundary(self) -> None:
        config = _read_json(CONFIG)
        validate_safeslot_gate_config(config)
        self.assertFalse(config["claim_boundary"]["runtime_integration_allowed"])
        self.assertTrue(
            config["claim_boundary"]["fresh_map_label_confirmation_required"]
        )
        self.assertEqual(config["features"]["input_dimension"], 193)

    def test_feature_vector_is_anchor_relative_and_pre_action_only(self) -> None:
        challenger = _candidate(state_value=10.0, other_value=2.0)
        anchor = _candidate(state_value=10.0, other_value=1.0)
        values = safeslot_feature_vector(challenger, anchor)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertEqual(len(values), 193)
        state_indices = [BASE_FEATURE_NAMES.index(name) for name in STATE_FEATURE_NAMES]
        self.assertTrue(all(values[index] == 0.0 for index in state_indices))
        self.assertTrue(all(value == 10.0 for value in values[-23:]))

    def test_policy_selects_one_challenger_or_exact_v2_abstention(self) -> None:
        rows = [
            _policy_row("one", "c1", label=True, advantage=0.2),
            _policy_row("one", "c2", label=False, advantage=-0.1),
            _policy_row("two", "c3", label=False, advantage=-0.1),
            _policy_row("two", "c4", label=True, advantage=0.2),
        ]
        records, metrics = evaluate_gate_policy(
            rows, [0.9, 0.2, 0.7, 0.6], threshold=0.8
        )
        self.assertEqual(len(records), 2)
        self.assertEqual(metrics["selected_state_count"], 1)
        self.assertEqual(metrics["replacement_state_fraction"], 0.5)
        self.assertEqual(metrics["safe_replace_precision"], 1.0)
        abstained = next(row for row in records if row["state_id"] == "two")
        self.assertFalse(abstained["selected_challenger"])
        self.assertIsNone(abstained["challenger_candidate_id"])


if __name__ == "__main__":
    unittest.main()
