from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_lns import (
    FROZEN_FEATURE_DIMENSION,
    FROZEN_FEATURE_SCHEMA_ID,
)
from experiments.stride_robustaction_preflight import (
    analyze_robustaction_preflight_rows,
    validate_robustaction_label_preflight_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT
    / "configs"
    / "stride_robustaction_structpool_label_preflight.json"
)


def _rows() -> list[dict[str, object]]:
    families = [
        "bottleneck_crossing",
        "conflict_component",
        "path_overlap",
        "spatiotemporal_hotspot",
        "topology_boundary",
    ]
    rows: list[dict[str, object]] = []
    for index in range(320):
        policy = "official_adaptive" if index < 160 else "v2-full"
        active = index < 51 or 160 <= index < 207
        map_index = index % 28
        if map_index < 12:
            layout = "dao_high_topology"
        elif map_index < 22:
            layout = "dao_mid_topology"
        else:
            layout = "dao_low_topology_control"
        rows.append(
            {
                "state_id": f"state-{index:03d}",
                "map_id": f"map-{map_index:02d}",
                "task_id": f"task-{index % 56:02d}",
                "source_policy": policy,
                "layout_mode": layout,
                "before_conflicts": 20 if active else 4,
                "high_stress_gate_passed": active,
                "selection_proxy_eligible": active,
                "topology_analysis_executed": active,
                "base_candidate_count": 18,
                "added_candidate_count": 6 if active else 0,
                "total_candidate_count": 24 if active else 18,
                "added_family_groups": families if active else [],
                "added_sizes": [8, 16, 24, 32] if active else [],
                "deterministic": True,
                "base_preserved": True,
                "incumbent_boundary_preserved": True,
                "candidate_cap_preserved": True,
                "native_explicit_action_legal": True,
                "maximum_novel_jaccard_similarity": 0.75,
                "state_fingerprint_preserved": True,
                "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
                "feature_dimension": FROZEN_FEATURE_DIMENSION,
                "candidate_signature": f"candidate-{index}",
                "feature_signature": f"feature-{index}",
            }
        )
    return rows


class RobustActionLabelPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_registered_config_and_inputs_are_valid(self) -> None:
        validate_robustaction_label_preflight_config(
            self.config, project_root=ROOT
        )
        self.assertEqual(
            self.config["feature_contract"]["dimension"],
            FROZEN_FEATURE_DIMENSION,
        )
        self.assertFalse(
            self.config["outcome_boundary"]["candidate_repair_step_allowed"]
        )

    def test_analyzer_passes_only_the_complete_registered_contract(self) -> None:
        report = analyze_robustaction_preflight_rows(
            self.config, _rows(), error_count=0
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["active_state_count"], 98)
        self.assertEqual(report["states_with_added_candidates"], 98)
        self.assertEqual(
            report["active_state_count_by_policy"],
            {"official_adaptive": 51, "v2-full": 47},
        )
        self.assertTrue(all(report["gates"].values()))

    def test_analyzer_rejects_missing_policy_coverage_and_features(self) -> None:
        rows = _rows()
        for row in rows:
            if row["source_policy"] == "v2-full" and row["high_stress_gate_passed"]:
                row["added_candidate_count"] = 0
                row["total_candidate_count"] = 18
                row["added_family_groups"] = []
                row["added_sizes"] = []
        rows[0]["feature_dimension"] = FROZEN_FEATURE_DIMENSION - 1
        report = analyze_robustaction_preflight_rows(
            self.config, rows, error_count=0
        )
        self.assertFalse(report["passed"])
        self.assertFalse(
            report["gates"]["minimum_states_with_added_candidates_per_policy"]
        )
        self.assertFalse(report["gates"]["complete_124_feature_rows"])

    def test_outcome_boundary_cannot_be_enabled(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["outcome_boundary"]["ttf_read"] = True
        with self.assertRaisesRegex(ValueError, "outcome boundary changed"):
            validate_robustaction_label_preflight_config(changed)


if __name__ == "__main__":
    unittest.main()
