from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_pool_effect import (
    build_pool_effect_record,
    summarize_pool_effect,
    validate_robustaction_pool_effect_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_robustaction_structpool_pool_effect.json"


def _selection(*, active: bool = True) -> dict[str, object]:
    return {
        "state_id": "state-0",
        "map_id": "map-0",
        "task_id": "task-0",
        "source_policy": "v2-full",
        "layout_mode": "dao_high_topology",
        "before_conflicts": 20,
        "candidate_count": 2,
        "base_candidate_count": 1,
        "structpool_candidate_count": int(active),
        "structpool_active": active,
        "baseline_candidate_id": "base",
        "augmented_candidate_id": "struct" if active else "base",
        "augmented_candidate_kind": "structpool" if active else "base",
        "action_changed": active,
        "baseline_v2_score": 0.0,
        "augmented_v2_score": 1.0 if active else 0.0,
        "baseline_v2_margin": 0.0,
        "augmented_v2_margin": 1.0 if active else 0.0,
        "candidate_repair_outcomes_read": False,
    }


class RobustActionPoolEffectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_config_and_inputs_are_valid(self) -> None:
        validate_robustaction_pool_effect_config(
            self.config, project_root=ROOT
        )
        self.assertEqual(
            self.config["cohort"]["active_structpool_state_count"], 98
        )

    def test_paired_record_keeps_selection_and_outcomes_separate(self) -> None:
        candidates = [
            {"candidate_id": "base", "candidate_kind": "base"},
            {"candidate_id": "struct", "candidate_kind": "structpool"},
        ]
        record = build_pool_effect_record(
            selection=_selection(),
            candidates=candidates,
            scores_by_candidate={
                "base": [0.0] * 16,
                "struct": [0.1] * 12 + [0.0] * 4,
            },
            epsilon=1e-12,
            robust_win_fraction=0.75,
            robust_mean_effect=0.02,
        )
        self.assertEqual(record["paired_win_count"], 12)
        self.assertAlmostEqual(record["raw_selected_gain"], 0.075)
        self.assertTrue(record["robust_improved"])
        self.assertFalse(record["robust_worsened"])
        self.assertFalse(record["candidate_repair_outcomes_read"])

    def test_summary_separates_improvement_and_regression(self) -> None:
        first = {
            **_selection(),
            "raw_selected_gain": 0.1,
            "first_half_raw_selected_gain": 0.1,
            "second_half_raw_selected_gain": 0.1,
            "normalized_selected_gain": 0.5,
            "first_half_normalized_selected_gain": 0.5,
            "second_half_normalized_selected_gain": 0.5,
            "paired_win_count": 16,
            "paired_loss_count": 0,
            "selected_quality_improved": True,
            "selected_quality_worsened": False,
            "selected_quality_tied": False,
            "robust_improved": True,
            "robust_worsened": False,
            "baseline_normalized_regret": 0.5,
            "augmented_normalized_regret": 0.0,
            "baseline_exact_best": False,
            "augmented_exact_best": True,
            "baseline_in_quality_top3": True,
            "augmented_in_quality_top3": True,
        }
        second = {
            **first,
            "raw_selected_gain": -0.1,
            "first_half_raw_selected_gain": -0.1,
            "second_half_raw_selected_gain": -0.1,
            "normalized_selected_gain": -0.5,
            "first_half_normalized_selected_gain": -0.5,
            "second_half_normalized_selected_gain": -0.5,
            "paired_win_count": 0,
            "paired_loss_count": 16,
            "selected_quality_improved": False,
            "selected_quality_worsened": True,
            "robust_improved": False,
            "robust_worsened": True,
            "baseline_normalized_regret": 0.0,
            "augmented_normalized_regret": 0.5,
            "baseline_exact_best": True,
            "augmented_exact_best": False,
        }
        summary = summarize_pool_effect([first, second])
        self.assertEqual(summary["improved_state_count"], 1)
        self.assertEqual(summary["worsened_state_count"], 1)
        self.assertEqual(summary["mean_raw_selected_gain"], 0.0)

    def test_gates_and_no_training_boundary_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["paired_ttf_quick_gates"][
            "maximum_worsened_state_rate"
        ] = 0.5
        with self.assertRaisesRegex(ValueError, "TTF gates"):
            validate_robustaction_pool_effect_config(changed)

        changed = copy.deepcopy(self.config)
        changed["claim_boundary"]["training_allowed"] = True
        with self.assertRaisesRegex(ValueError, "claim boundary"):
            validate_robustaction_pool_effect_config(changed)


if __name__ == "__main__":
    unittest.main()
