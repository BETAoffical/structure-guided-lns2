from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import experiments.stride_safeslot_pair_motif_sequential_ablation as ablation
from experiments._common import sha256_file
from experiments.repair_collection import _read_json


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/stride_safeslot_pair_motif_sequential_ablation_v1.json"
CONFIG_SHA256 = "d4579080afe046d7d7b2c3ccac7fe204be47f4784a726c4acf69cafc43d215cc"


def _state() -> dict:
    return {
        "rows": 2,
        "cols": 3,
        "obstacles": [0] * 6,
        "agents": [
            {"id": 0, "path": [0, 1, 2], "start": 0, "goal": 2},
            {"id": 1, "path": [2, 1, 0], "start": 2, "goal": 0},
            {"id": 2, "path": [3, 4, 5], "start": 3, "goal": 5},
            {"id": 3, "path": [5, 4, 3], "start": 5, "goal": 3},
        ],
        "conflict_edges": [[0, 1], [2, 3]],
        "num_of_colliding_pairs": 2,
    }


class PairMotifSequentialAblationTests(unittest.TestCase):
    def test_frozen_schema_and_config(self) -> None:
        config = _read_json(CONFIG)
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)
        ablation.validate_config(config, project_root=ROOT)
        self.assertEqual(len(ablation.STATIC_FEATURE_NAMES), 193)
        self.assertEqual(len(ablation.MOTIF_FEATURE_NAMES), 18)
        self.assertEqual(len(ablation.FEATURE_NAMES), 211)
        self.assertEqual(
            tuple(config["features"]["motif_feature_names"]),
            ablation.MOTIF_FEATURE_NAMES,
        )
        self.assertEqual(config["model"]["parameter_index"], 0)
        self.assertEqual(config["model"]["workers"], 16)
        self.assertNotIn("threshold_grid", json.dumps(config, sort_keys=True))

    def test_same_candidate_has_zero_deltas_and_unit_nonempty_jaccards(self) -> None:
        values = ablation.pair_motif_feature_vector(_state(), {0, 1}, {0, 1})
        self.assertEqual(values[:16], [0.0] * 16)
        self.assertEqual(values[16:], [1.0, 1.0])

    def test_swap_is_antisymmetric_for_deltas_and_symmetric_for_jaccards(self) -> None:
        state = _state()
        forward = ablation.pair_motif_feature_vector(state, {1, 2}, {0, 2})
        reverse = ablation.pair_motif_feature_vector(state, {0, 2}, {1, 2})
        for left, right in zip(forward[:16], reverse[:16]):
            self.assertAlmostEqual(left, -right, places=12)
        self.assertEqual(forward[16:], reverse[16:])
        self.assertTrue(all(math.isfinite(value) for value in forward))

    def test_feature_extractor_ignores_poisoned_forbidden_fields(self) -> None:
        state = _state()
        expected = ablation.pair_motif_feature_vector(state, {1, 2}, {0, 2})
        poisoned = copy.deepcopy(state)
        poisoned.update(
            {
                "history": [{"label": True, "ttf": -999}],
                "target_action": {"agents": [999]},
                "target_metrics": {"success": True},
                "target_state_delta": {"poison": True},
                "target_outcome": "poison",
                "after_state": {"poison": True},
                "future_transition": {"poison": True},
                "pp_seed": 999,
                "pp_time": -1.0,
                "runtime": -1.0,
                "ttf": -1.0,
            }
        )
        self.assertEqual(
            ablation.pair_motif_feature_vector(poisoned, {1, 2}, {0, 2}),
            expected,
        )

    def test_state_cache_reuses_single_analysis_across_candidates(self) -> None:
        original = ablation.analyze_state
        with mock.patch.object(ablation, "analyze_state", wraps=original) as wrapped:
            cache = ablation._pair_motif_cache(_state())
            first = ablation._pair_motif_feature_vector_from_cache(
                cache, {1, 2}, {0, 2}
            )
            second = ablation._pair_motif_feature_vector_from_cache(
                cache, {1, 3}, {0, 3}
            )
        self.assertEqual(wrapped.call_count, 1)
        self.assertEqual(len(first), 18)
        self.assertEqual(len(second), 18)
        self.assertTrue(cache.occupancy)
        self.assertTrue(cache.transitions)
        self.assertEqual(len(cache.same_time_conflict_pairs), 2)

    def test_plan_and_run_dry_run_are_solver_free_and_write_nothing(self) -> None:
        planned = ablation.plan(config_path=CONFIG)
        self.assertEqual(planned["cohort"]["candidate_count"], 587)
        self.assertEqual(planned["configured_workers"], 16)
        self.assertFalse(planned["new_ablation_threshold_policy_or_export"])
        self.assertTrue(
            planned["legacy_identity_reproduction_replays_registered_static_calibration"]
        )
        self.assertTrue(planned["legacy_calibration_not_interpreted_or_exported"])
        self.assertFalse(planned["solver_or_controller_invoked"])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "must-not-exist"
            result = ablation.run_ablation(
                config_path=CONFIG, output=output, dry_run=True
            )
            self.assertTrue(result["dry_run"])
            self.assertFalse(result["solver_or_controller_invoked"])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
