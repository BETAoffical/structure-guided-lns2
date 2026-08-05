from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_topology_anchor_quality import (
    STATE_SCHEMA,
    TRIAL_SCHEMA,
    _pilot_artifact_valid,
    _quality_ranking,
    validate_topology_anchor_quality_config,
)
from experiments.stride_stability import stride_extended_pp_seed


POST_STRUCTURE = {
    "post_largest_component_ratio": 0.1,
    "post_conflict_edge_density": 0.1,
    "post_event_density": 0.1,
    "post_degree_concentration": 0.1,
}


class StrideTopologyAnchorQualityTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (root / "configs" / "stride_topology_anchor_quality_pilot.json").read_text(
                encoding="utf-8"
            )
        )

    def test_registered_pilot_uses_only_current_step_quality(self) -> None:
        config = self._config()
        validate_topology_anchor_quality_config(config)
        self.assertFalse(config["runtime_used_in_label"])
        self.assertFalse(config["future_repair_rounds_used"])
        self.assertFalse(config["cost_to_go_used"])
        self.assertFalse(config["training_allowed"])
        config["runtime_used_in_label"] = True
        with self.assertRaisesRegex(ValueError, "immediate and non-promoting"):
            validate_topology_anchor_quality_config(config)

    def test_mean_np100_penalizes_repeated_no_progress(self) -> None:
        scores = {
            "steady": {
                0: 0.10,
                1: 0.10,
                2: 0.10,
                3: 0.10,
                "progress_0": True,
                "progress_1": True,
                "progress_2": True,
                "progress_3": True,
            },
            "stalled": {
                0: 0.14,
                1: 0.14,
                2: 0.14,
                3: 0.14,
                "progress_0": False,
                "progress_1": False,
                "progress_2": False,
                "progress_3": False,
            },
        }
        aggregated, ranking = _quality_ranking(scores, [0, 1, 2, 3], 0.10)
        self.assertEqual(ranking[0], "steady")
        self.assertAlmostEqual(aggregated["stalled"], 0.04)

    def test_state_artifact_requires_complete_candidate_seed_product(self) -> None:
        candidate_ids = ["a", "b"]
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
                "candidate_kind": "base",
                "agents": [index],
                "actual_size": 1,
                "selection_families": ["target:4"],
            }
            for index, candidate_id in enumerate(candidate_ids)
        ]
        candidates_by_id = {row["candidate_id"]: row for row in candidates}
        trials = [
            {
                "schema": TRIAL_SCHEMA,
                "state_id": "state",
                "task_id": "task",
                "map_id": "map",
                "layout_family": "family",
                "solver_seed": 1,
                "candidate_id": candidate_id,
                "candidate_kind": "base",
                "selection_families": ["target:4"],
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
                "post_structure": dict(POST_STRUCTURE),
                "native_step_seconds": 0.1,
                "pp_replan_seconds": 0.05,
            }
            for candidate_id in candidate_ids
            for trial_index in (0, 1, 2, 3)
        ]
        payload = {
            "schema": STATE_SCHEMA,
            "identity": "identity",
            "state_id": "state",
            "complete": True,
            "state": state_row,
            "candidate_ids": candidate_ids,
            "trials": trials,
        }
        self.assertTrue(
            _pilot_artifact_valid(
                payload,
                identity="identity",
                state_id="state",
                candidate_ids=candidate_ids,
                trial_indices=(0, 1, 2, 3),
                state_row=state_row,
                candidates=[candidates_by_id[value] for value in candidate_ids],
            )
        )
        payload["trials"].pop()
        self.assertFalse(
            _pilot_artifact_valid(
                payload,
                identity="identity",
                state_id="state",
                candidate_ids=candidate_ids,
                trial_indices=(0, 1, 2, 3),
                state_row=state_row,
                candidates=[candidates_by_id[value] for value in candidate_ids],
            )
        )


if __name__ == "__main__":
    unittest.main()
