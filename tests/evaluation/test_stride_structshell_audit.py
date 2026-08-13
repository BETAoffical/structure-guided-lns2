from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_structshell_audit import (
    CONFIG_SCHEMA,
    EXPERIMENT_ID,
    _rule_candidates,
    _tail_rows,
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

    def test_pretail_identity_uses_first_structural_checkpoint(self) -> None:
        case_id = "case-1"
        candidate_id = "candidate-16"
        checkpoints = [
            {
                "case_id": case_id,
                "checkpoint_kind": "first_structural_selection",
                "state_fingerprint": "structural-state",
                "task_id": "task-1",
                "solver_seed": 3,
            },
            {
                "case_id": case_id,
                "checkpoint_kind": "first_repeat_stall",
                "state_fingerprint": "stall-state",
                "task_id": "task-1",
                "solver_seed": 3,
            },
        ]
        schedule = [
            {
                "case_id": case_id,
                "trial_index": 0,
                "arm": "coverage_diverse",
                "candidate_id": candidate_id,
                "task_id": "task-1",
                "solver_seed": 3,
            }
        ]
        comparisons = [
            {
                "case_id": case_id,
                "trial_index": 0,
                "arm": "coverage_diverse",
                "classification": "beneficial",
                "reason": "lower_auc",
                "identical_action": False,
            }
        ]
        common = {
            "map_id": "maze",
            "task_id": "task-1",
            "solver_seed": 3,
            "base_candidate_ids": [],
            "rule_selected_candidate_ids": {
                rule: [candidate_id]
                for rule in (
                    "structural_knee",
                    "support_nearest",
                    "fixed_preferred",
                    "equal_four_size_grid",
                )
            },
            "rule_available": {
                rule: True
                for rule in (
                    "structural_knee",
                    "support_nearest",
                    "fixed_preferred",
                    "equal_four_size_grid",
                )
            },
        }
        maze_states = [
            {
                **common,
                "state_fingerprint": "structural-state",
                "all_candidate_ids": [candidate_id],
            },
            {
                **common,
                "state_fingerprint": "stall-state",
                "all_candidate_ids": ["other-candidate"],
            },
        ]
        rows, errors = _tail_rows(
            comparisons, schedule, checkpoints, maze_states
        )
        self.assertEqual(errors, [])
        self.assertEqual(rows[0]["state_fingerprint"], "structural-state")

    def test_pretail_identity_rejects_candidate_outside_state_pool(self) -> None:
        checkpoints = [
            {
                "case_id": "case-1",
                "checkpoint_kind": "first_structural_selection",
                "state_fingerprint": "state-1",
                "task_id": "task-1",
                "solver_seed": 3,
            }
        ]
        schedule = [
            {
                "case_id": "case-1",
                "trial_index": 0,
                "arm": "coverage_diverse",
                "candidate_id": "missing-candidate",
                "task_id": "task-1",
                "solver_seed": 3,
            }
        ]
        comparisons = [
            {
                "case_id": "case-1",
                "trial_index": 0,
                "arm": "coverage_diverse",
                "classification": "beneficial",
                "reason": "lower_auc",
                "identical_action": False,
            }
        ]
        maze_states = [
            {
                "state_fingerprint": "state-1",
                "map_id": "maze",
                "task_id": "task-1",
                "solver_seed": 3,
                "all_candidate_ids": ["present-candidate"],
                "base_candidate_ids": [],
                "rule_selected_candidate_ids": {},
                "rule_available": {},
            }
        ]
        rows, errors = _tail_rows(
            comparisons, schedule, checkpoints, maze_states
        )
        self.assertEqual(rows, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("pretail_candidate_left_state_pool", errors[0])


if __name__ == "__main__":
    unittest.main()
