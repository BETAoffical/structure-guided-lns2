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
from experiments.stride_structpool_coverage import (
    analyze_structpool_coverage_rows,
    select_structpool_states,
    validate_structpool_coverage_config,
)
from experiments.stride_structpool_headroom import (
    analyze_structpool_headroom,
    select_headroom_states,
    validate_structpool_headroom_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_design.json"
COVERAGE_CONFIG = ROOT / "configs" / "stride_structpool_coverage.json"
HEADROOM_CONFIG = ROOT / "configs" / "stride_structpool_headroom_pilot.json"


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


class StrideStructPoolCoverageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(COVERAGE_CONFIG.read_text(encoding="utf-8"))
        validate_structpool_coverage_config(cls.config)

    def test_selection_is_outcome_blind_deterministic_and_map_balanced(self) -> None:
        rows = []
        for map_index in range(12):
            for policy in ("official_adaptive", "v2-full"):
                for index in range(3):
                    rows.append(
                        {
                            "state_id": f"state-{map_index:02d}-{policy}-{index}",
                            "map_id": f"map-{map_index:02d}",
                            "source_policy": policy,
                            "research_split": "train",
                            "agent_count": 120,
                            "before_conflicts": 20,
                            "candidate_outcome_that_must_not_be_read": index,
                        }
                    )
        first = select_structpool_states(rows, self.config)
        second = select_structpool_states(list(reversed(rows)), self.config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 48)
        self.assertEqual(len({row["map_id"] for row in first}), 12)

    def test_proposal_only_report_passes_only_complete_contract(self) -> None:
        rows = []
        families = [
            "bottleneck_crossing",
            "conflict_component",
            "topology_boundary",
            "spatiotemporal_hotspot",
            "path_overlap",
        ]
        for index in range(48):
            rows.append(
                {
                    "state_id": f"state-{index}",
                    "map_id": f"map-{index % 12}",
                    "source_policy": (
                        "official_adaptive" if index % 2 == 0 else "v2-full"
                    ),
                    "added_candidate_count": 6,
                    "added_family_groups": families,
                    "added_sizes": [8, 16, 24, 32],
                    "high_stress_gate_passed": True,
                    "deterministic": True,
                    "base_preserved": True,
                    "incumbent_boundary_preserved": True,
                    "candidate_cap_preserved": True,
                    "native_explicit_action_legal": True,
                    "state_fingerprint_preserved": True,
                    "maximum_novel_jaccard_similarity": 0.75,
                }
            )
        report = analyze_structpool_coverage_rows(self.config, rows)
        self.assertTrue(report["passed"])
        self.assertFalse(report["candidate_repair_trials_executed"])
        self.assertFalse(report["controller_actions_executed"])
        self.assertFalse(report["selection_outcome_fields_read"])

        rows[0]["base_preserved"] = False
        failed = analyze_structpool_coverage_rows(self.config, rows)
        self.assertFalse(failed["passed"])
        self.assertFalse(failed["gates"]["exact_base_preservation"])

    def test_coverage_validator_rejects_repair_permission(self) -> None:
        mutated = copy.deepcopy(self.config)
        mutated["candidate_repair_trials_allowed"] = True
        with self.assertRaisesRegex(ValueError, "proposal-only"):
            validate_structpool_coverage_config(mutated)


class StrideStructPoolHeadroomTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(HEADROOM_CONFIG.read_text(encoding="utf-8"))
        validate_structpool_headroom_config(cls.config)

    def test_pilot_cohort_is_fixed_without_repair_outcomes(self) -> None:
        rows = []
        for map_index in range(13):
            for policy in ("official_adaptive", "v2-full"):
                rows.append(
                    {
                        "state_id": f"state-{map_index:02d}-{policy}",
                        "map_id": f"map-{map_index:02d}",
                        "source_policy": policy,
                        "repair_outcome_that_must_not_be_read": map_index,
                    }
                )
        first = select_headroom_states(rows, self.config)
        second = select_headroom_states(list(reversed(rows)), self.config)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 16)
        self.assertEqual(len({row["map_id"] for row in first}), 13)
        self.assertEqual(
            {row["source_policy"] for row in first},
            {"official_adaptive", "v2-full"},
        )

    def test_headroom_gates_use_expected_one_step_gain(self) -> None:
        passing = [
            {"best_expected_gain_over_incumbent": 0.02 if index < 4 else 0.01}
            for index in range(16)
        ]
        report = analyze_structpool_headroom(self.config, passing)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(report["opportunity_fraction"], 0.25)
        self.assertGreaterEqual(report["mean_gain"], 0.01)

        failing = [
            {"best_expected_gain_over_incumbent": 0.0} for _ in range(16)
        ]
        failed = analyze_structpool_headroom(self.config, failing)
        self.assertFalse(failed["passed"])

    def test_headroom_validator_rejects_timing_or_future_labels(self) -> None:
        timing = copy.deepcopy(self.config)
        timing["timing_fields_allowed"] = True
        with self.assertRaisesRegex(ValueError, "evidence boundary"):
            validate_structpool_headroom_config(timing)
        future = copy.deepcopy(self.config)
        future["label"]["future_state_used"] = True
        with self.assertRaisesRegex(ValueError, "label changed"):
            validate_structpool_headroom_config(future)


if __name__ == "__main__":
    unittest.main()
