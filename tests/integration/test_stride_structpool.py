from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_structpool import (
    high_stress_gate,
    load_structpool_design,
    validate_structpool_design,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_design.json"


class StrideStructPoolDesignTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_structpool_design(CONFIG)

    def test_registered_design_and_evidence_checksums_are_valid(self) -> None:
        validate_structpool_design(self.config, project_root=ROOT)
        self.assertFalse(self.config["research_scope"]["formal_speed_claim"])
        self.assertFalse(
            self.config["proposal_only_stage"]["repair_or_controller_step_allowed"]
        )

    def test_candidate_pool_preserves_v2_and_incumbent_boundaries(self) -> None:
        candidate = self.config["candidate_space"]
        self.assertEqual(candidate["base_pool"]["generator"], "frozen_v2")
        self.assertEqual(
            candidate["incumbent_additions"]["generator"],
            "stride-topoboundary-v1",
        )
        self.assertEqual(candidate["maximum_added_candidates"], 6)
        self.assertEqual(len(candidate["novel_family_groups"]), 5)

    def test_high_stress_gate_is_current_state_only(self) -> None:
        eligible = {
            "agent_count": 120,
            "conflict_pair_count": 20,
            "active_conflict_agent_count": 24,
            "largest_conflict_component_size": 12,
        }
        self.assertTrue(high_stress_gate(eligible, self.config))
        self.assertFalse(
            high_stress_gate(
                {**eligible, "agent_count": 80, "conflict_pair_count": 15},
                self.config,
            )
        )
        with self.assertRaisesRegex(ValueError, "forbidden outcome"):
            high_stress_gate(
                {**eligible, "candidate_conflicts_after": 0}, self.config
            )

    def test_runtime_evidence_remains_locked_by_power_report(self) -> None:
        runtime = self.config["runtime_stage"]
        self.assertTrue(
            runtime["allowed_only_after_user_reports_comparable_performance_restored"]
        )
        self.assertEqual(
            runtime["primary_metric"], "mean_run_to_completion_raw_wall_ttf"
        )

    def test_validator_rejects_post_hoc_gate_drift(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["headroom_pilot"][
            "minimum_mean_best_expected_gain_over_incumbent_pool"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "headroom pilot changed"):
            validate_structpool_design(mutated)

    def test_json_contains_no_unregistered_nan_values(self) -> None:
        payload = json.dumps(self.config, allow_nan=False, sort_keys=True)
        self.assertIn("stride-structpool-v1", payload)


if __name__ == "__main__":
    unittest.main()
