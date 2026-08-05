from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_topology_anchor_quality import _pilot_artifact_valid
from experiments.stride_topology_boundary_quality import (
    STATE_SCHEMA,
    TRIAL_SCHEMA,
    validate_topology_boundary_quality_config,
)
from experiments.stride_stability import stride_extended_pp_seed


class StrideTopologyBoundaryQualityTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (
                root / "configs" / "stride_topology_boundary_quality_pilot.json"
            ).read_text(encoding="utf-8")
        )

    def test_registered_pilot_freezes_fresh_current_step_boundary(self) -> None:
        config = self._config()
        validate_topology_boundary_quality_config(config)
        self.assertEqual(config["expected_state_count"], 18)
        self.assertEqual(config["expected_candidate_count"], 347)
        self.assertEqual(config["expected_boundary_only_candidate_count"], 23)
        self.assertEqual(config["expected_outcome_count"], 1388)
        self.assertFalse(
            config["freshness"][
                "prior_quality_outcomes_used_for_state_or_candidate_selection"
            ]
        )
        self.assertFalse(config["runtime_used_in_label"])
        self.assertFalse(config["future_repair_rounds_used"])
        self.assertFalse(config["cost_to_go_used"])
        self.assertFalse(config["training_allowed"])

    def test_registration_rejects_runtime_or_gate_drift(self) -> None:
        config = self._config()
        config["runtime_used_in_label"] = True
        with self.assertRaisesRegex(ValueError, "immediate and non-promoting"):
            validate_topology_boundary_quality_config(config)
        config = self._config()
        config["pilot_gates"]["minimum_augmented_pool_strict_win_rate"] = 0.0
        with self.assertRaisesRegex(ValueError, "gates changed"):
            validate_topology_boundary_quality_config(config)

    def test_boundary_state_artifact_uses_boundary_schema(self) -> None:
        candidate_ids = ["base", "boundary"]
        state_row = {
            "state_id": "state",
            "task_id": "task",
            "map_id": "map",
            "layout_family": "family",
            "solver_seed": 1,
            "state_fingerprint": "before-state",
            "initial_conflicts": 5,
        }
        candidates = [
            {
                "candidate_id": candidate_id,
                "candidate_kind": (
                    "base" if candidate_id == "base" else "boundary_only"
                ),
                "agents": [index],
                "actual_size": 1,
                "selection_families": [
                    "target:4" if candidate_id == "base" else "topology-boundary:16"
                ],
            }
            for index, candidate_id in enumerate(candidate_ids)
        ]
        by_id = {row["candidate_id"]: row for row in candidates}
        payload = {
            "schema": STATE_SCHEMA,
            "identity": "identity",
            "state_id": "state",
            "complete": True,
            "state": state_row,
            "candidate_ids": candidate_ids,
            "trials": [
                {
                    "schema": TRIAL_SCHEMA,
                    "state_id": "state",
                    "task_id": "task",
                    "map_id": "map",
                    "layout_family": "family",
                    "solver_seed": 1,
                    "candidate_id": candidate_id,
                    "candidate_kind": by_id[candidate_id]["candidate_kind"],
                    "selection_families": by_id[candidate_id][
                        "selection_families"
                    ],
                    "actual_size": 1,
                    "before_conflicts": 5,
                    "before_fingerprint": "before-state",
                    "before_repair_fingerprint": "before-repair",
                    "trial_index": trial_index,
                    "pp_seed": stride_extended_pp_seed(
                        "before-repair", trial_index
                    ),
                    "replan_success": True,
                    "feasible": False,
                    "conflicts_after": 4,
                    "after_fingerprint": f"after-{candidate_id}-{trial_index}",
                    "after_repair_fingerprint": (
                        f"after-repair-{candidate_id}-{trial_index}"
                    ),
                    "repair_outcome": "conflict_reduced",
                    "post_structure": {
                        "post_largest_component_ratio": 0.1,
                        "post_conflict_edge_density": 0.1,
                        "post_event_density": 0.1,
                        "post_degree_concentration": 0.1,
                    },
                    "native_step_seconds": 0.1,
                    "pp_replan_seconds": 0.05,
                }
                for candidate_id in candidate_ids
                for trial_index in (0, 1, 2, 3)
            ],
        }
        self.assertTrue(
            _pilot_artifact_valid(
                payload,
                identity="identity",
                state_id="state",
                candidate_ids=candidate_ids,
                trial_indices=(0, 1, 2, 3),
                state_schema=STATE_SCHEMA,
                trial_schema=TRIAL_SCHEMA,
                state_row=state_row,
                candidates=candidates,
            )
        )


if __name__ == "__main__":
    unittest.main()
