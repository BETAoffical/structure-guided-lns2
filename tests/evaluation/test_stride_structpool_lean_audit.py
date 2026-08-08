from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_structpool_lean_audit import (
    build_lean_candidate_pool,
    build_lean_effect_record,
    is_removed_by_lean_filter,
    summarize_lean_effect,
    validate_structpool_lean_audit_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_lean_audit.json"


def _selection(*, changed: bool = True) -> dict[str, object]:
    return {
        "state_id": "state-0",
        "map_id": "map-0",
        "task_id": "task-0",
        "source_policy": "v2-full",
        "layout_mode": "dao_high_topology",
        "before_conflicts": 20,
        "full_candidate_count": 3,
        "lean_candidate_count": 2,
        "full_structpool_candidate_count": 2,
        "lean_structpool_candidate_count": 1,
        "removed_candidate_count": 1,
        "removed_candidate_ids": ["pure-bottleneck"],
        "structpool_active": True,
        "full_candidate_id": "base",
        "lean_candidate_id": "mixed" if changed else "base",
        "full_candidate_kind": "base",
        "lean_candidate_kind": "structpool" if changed else "base",
        "action_changed": changed,
        "full_v2_score": 0.0,
        "lean_v2_score": 1.0 if changed else 0.0,
        "full_v2_margin": 0.0,
        "lean_v2_margin": 1.0 if changed else 0.0,
        "removed_candidate_was_full_selection": False,
        "candidate_repair_outcomes_read": False,
    }


class StructPoolLeanAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_config_and_inputs_are_valid(self) -> None:
        validate_structpool_lean_audit_config(self.config, project_root=ROOT)
        self.assertEqual(
            self.config["cohort"]["expected_removed_candidate_count"], 49
        )

    def test_filter_removes_only_pure_bottleneck_candidate(self) -> None:
        base = {
            "candidate_id": "base",
            "candidate_kind": "base",
            "structpool_family_groups": [],
        }
        pure = {
            "candidate_id": "pure",
            "candidate_kind": "structpool",
            "structpool_family_groups": ["bottleneck_crossing"],
        }
        mixed = {
            "candidate_id": "mixed",
            "candidate_kind": "structpool",
            "structpool_family_groups": ["bottleneck_crossing", "path_overlap"],
        }
        self.assertFalse(is_removed_by_lean_filter(base))
        self.assertTrue(is_removed_by_lean_filter(pure))
        self.assertFalse(is_removed_by_lean_filter(mixed))
        self.assertEqual(build_lean_candidate_pool([base, pure, mixed]), [base, mixed])

    def test_paired_effect_uses_all_sixteen_fixed_seed_outcomes(self) -> None:
        candidates = [
            {"candidate_id": "base", "candidate_kind": "base"},
            {"candidate_id": "mixed", "candidate_kind": "structpool"},
            {"candidate_id": "pure-bottleneck", "candidate_kind": "structpool"},
        ]
        record = build_lean_effect_record(
            selection=_selection(),
            candidates=candidates,
            scores_by_candidate={
                "base": [0.0] * 16,
                "mixed": [0.1] * 16,
                "pure-bottleneck": [-0.1] * 16,
            },
            epsilon=1e-12,
            robust_win_fraction=0.75,
            robust_mean_effect=0.02,
        )
        self.assertEqual(record["paired_win_count"], 16)
        self.assertAlmostEqual(record["raw_selected_gain"], 0.1)
        self.assertTrue(record["robust_improved"])
        self.assertFalse(record["robust_worsened"])
        self.assertTrue(record["outcomes_joined_after_selection"])

    def test_summary_reports_regret_and_top3_deltas(self) -> None:
        common = {
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
            "full_normalized_regret": 0.5,
            "lean_normalized_regret": 0.0,
            "full_exact_best": False,
            "lean_exact_best": True,
            "full_in_quality_top3": False,
            "lean_in_quality_top3": True,
        }
        summary = summarize_lean_effect([common])
        self.assertEqual(summary["action_changed_count"], 1)
        self.assertEqual(summary["selected_quality_worsened_count"], 0)
        self.assertAlmostEqual(summary["mean_normalized_regret_delta"], -0.5)
        self.assertAlmostEqual(summary["quality_top3_rate_delta"], 1.0)

    def test_filter_and_claim_boundary_cannot_drift(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["lean_filter"]["replacement_candidate_added"] = True
        with self.assertRaisesRegex(ValueError, "filter changed"):
            validate_structpool_lean_audit_config(changed)

        changed = copy.deepcopy(self.config)
        changed["claim_boundary"]["training_allowed"] = True
        with self.assertRaisesRegex(ValueError, "claim boundary"):
            validate_structpool_lean_audit_config(changed)


if __name__ == "__main__":
    unittest.main()
