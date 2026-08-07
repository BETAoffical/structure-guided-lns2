from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_opportunity import (
    fixed_half_uncertainty,
    robust_action_comparison,
    validate_robustaction_opportunity_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_opportunity_audit.json"
)


def _candidate(candidate_id: str, kind: str = "base") -> dict[str, object]:
    return {"candidate_id": candidate_id, "candidate_kind": kind}


class RobustActionOpportunityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_config_and_inputs_are_valid(self) -> None:
        validate_robustaction_opportunity_config(
            self.config, project_root=ROOT
        )
        self.assertEqual(
            self.config["opportunity_gates"][
                "minimum_states_with_nonanchor_robust_win_fraction"
            ],
            0.25,
        )

    def test_robust_action_requires_pairing_effect_and_both_halves(self) -> None:
        challenger_scores = [0.1] * 12 + [0.0] * 4
        anchor_scores = [0.0] * 16
        result = robust_action_comparison(
            challenger=_candidate("challenger", "structpool_only"),
            anchor=_candidate("anchor"),
            challenger_scores=challenger_scores,
            anchor_scores=anchor_scores,
            minimum_paired_win_fraction=0.75,
            minimum_absolute_mean_effect=0.02,
            epsilon=1e-12,
        )
        self.assertEqual(result["paired_win_count"], 12)
        self.assertEqual(result["paired_win_fraction"], 0.75)
        self.assertTrue(result["robust_positive"])

        one_half_only = robust_action_comparison(
            challenger=_candidate("challenger", "structpool_only"),
            anchor=_candidate("anchor"),
            challenger_scores=[0.1] * 8 + [0.0] * 8,
            anchor_scores=anchor_scores,
            minimum_paired_win_fraction=0.5,
            minimum_absolute_mean_effect=0.02,
            epsilon=1e-12,
        )
        self.assertFalse(one_half_only["robust_positive"])

    def test_fixed_half_uncertainty_is_candidate_order_independent(self) -> None:
        candidates = [
            {
                "candidate_id": "a",
                "first_fixed_half_mean": 0.3,
                "second_fixed_half_mean": 0.1,
            },
            {
                "candidate_id": "b",
                "first_fixed_half_mean": 0.2,
                "second_fixed_half_mean": 0.3,
            },
            {
                "candidate_id": "c",
                "first_fixed_half_mean": 0.1,
                "second_fixed_half_mean": 0.2,
            },
        ]
        first = fixed_half_uncertainty(candidates, epsilon=1e-12)
        second = fixed_half_uncertainty(list(reversed(candidates)), epsilon=1e-12)
        self.assertEqual(first, second)
        self.assertFalse(first["fixed_half_exact_winner_agreement"])
        self.assertEqual(first["fixed_half_top3_overlap"], 1.0)
        self.assertAlmostEqual(
            first["fixed_half_pairwise_direction_agreement"], 1.0 / 3.0
        )

    def test_preregistered_gates_and_claim_boundary_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["opportunity_gates"][
            "minimum_robust_positive_action_fraction"
        ] = 0.01
        with self.assertRaisesRegex(ValueError, "opportunity gates"):
            validate_robustaction_opportunity_config(changed)

        changed = copy.deepcopy(self.config)
        changed["claim_boundary"]["ttf_read"] = True
        with self.assertRaisesRegex(ValueError, "claim boundary"):
            validate_robustaction_opportunity_config(changed)


if __name__ == "__main__":
    unittest.main()
