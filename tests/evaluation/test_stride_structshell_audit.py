from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_structshell_audit import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    _rule_candidates,
    parse_structpool_family,
    select_structural_knee,
    select_support_nearest,
)


def _row(size: int, coverage: float, support: int = 17) -> dict:
    return {
        "candidate_id": f"candidate-{size}",
        "nominal_size": size,
        "support_count": support,
        "features": {
            "realized.incident_event_coverage": coverage,
            "realized.internal_conflict_coverage": coverage,
            "realized.component_coverage_max": coverage,
            "realized.incident_conflict_coverage": coverage,
            "realized.boundary_conflict_edges": coverage * 10.0,
            "state.colliding_pairs": 10.0,
        },
    }


class StructShellAuditTest(unittest.TestCase):
    def test_registration_freezes_fallback_and_claim_boundary(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (
                root
                / "configs"
                / "stride_structshell_audit_v1_registration.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["schema"], CONFIG_SCHEMA)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(
            config["fallback"]["when_any_primary_gate_fails"],
            "retain_equal_four_size_grid_8_16_24_32",
        )
        self.assertFalse(config["fallback"]["fixed_family_preferred_sizes_allowed"])
        self.assertFalse(config["claim_boundary"]["model_training_allowed"])
        self.assertFalse(config["claim_boundary"]["ttf_experiment_allowed"])
        self.assertFalse(
            config["claim_boundary"]["long_tail_avoidance_claim_allowed"]
        )
        self.assertEqual(
            config["cohorts"]["maze_difficult_states"][
                "legacy_best_opportunity_state_count"
            ],
            41,
        )
        self.assertEqual(
            config["cohorts"]["maze_difficult_states"][
                "robust_action_opportunity_state_count"
            ],
            47,
        )
        self.assertFalse(
            config["execution_amendment"]["rules_or_thresholds_changed"]
        )

    def test_family_parser_preserves_boundary_variants(self) -> None:
        self.assertEqual(
            parse_structpool_family("structpool-boundary-low_degree:16"),
            ("topology_boundary_low_degree", 16),
        )
        with self.assertRaises(ValueError):
            parse_structpool_family("structpool-path-overlap:12")

    def test_structural_knee_uses_only_structure_and_ties_to_smaller_size(self) -> None:
        rows = [
            _row(8, 0.0),
            _row(16, 0.8),
            _row(24, 0.9),
            _row(32, 1.0),
        ]
        selected = select_structural_knee(rows, "path_overlap")
        self.assertEqual(selected["nominal_size"], 16)
        flat = [_row(size, 0.5) for size in (8, 16, 24, 32)]
        self.assertEqual(
            select_structural_knee(flat, "path_overlap")["nominal_size"], 8
        )

    def test_support_nearest_uses_equal_four_size_alternatives(self) -> None:
        rows = [_row(size, 0.5, support=20) for size in (8, 16, 24, 32)]
        self.assertEqual(
            select_support_nearest(rows, "path_overlap")["nominal_size"], 16
        )

    def test_missing_historical_support_is_not_imputed(self) -> None:
        rows = [_row(size, 0.5) for size in (8, 16, 24, 32)]
        for row in rows:
            row.pop("support_count")
        selected, available = _rule_candidates(
            {"path_overlap": rows},
            {
                "bottleneck_crossing": 8,
                "conflict_component": 24,
                "topology_boundary_articulation": 16,
                "topology_boundary_low_degree": 16,
                "spatiotemporal_hotspot": 16,
                "path_overlap": 32,
            },
        )
        self.assertFalse(available["support_nearest"])
        self.assertEqual(selected["support_nearest"], set())
        self.assertTrue(available["structural_knee"])


if __name__ == "__main__":
    unittest.main()
