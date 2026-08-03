from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_topology_anchor_quality import (
    STATE_SCHEMA,
    _pilot_artifact_valid,
    _quality_ranking,
    validate_topology_anchor_quality_config,
)


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
        trials = [
            {"candidate_id": candidate_id, "trial_index": trial_index}
            for candidate_id in candidate_ids
            for trial_index in (0, 1, 2, 3)
        ]
        payload = {
            "schema": STATE_SCHEMA,
            "identity": "identity",
            "state_id": "state",
            "complete": True,
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
            )
        )


if __name__ == "__main__":
    unittest.main()
