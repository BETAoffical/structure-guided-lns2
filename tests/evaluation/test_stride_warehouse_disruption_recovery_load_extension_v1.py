from __future__ import annotations

import unittest
from pathlib import Path

from experiments.stride_warehouse_disruption_recovery_load_extension_v1 import (
    BANDS, CONTROLLERS, _comparison, _schedule, dry_run, load_config,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_warehouse_disruption_recovery_load_extension_v1.json"


class WarehouseLoadExtensionTests(unittest.TestCase):
    def test_registered_product_and_isolation(self) -> None:
        _path, _root, config = load_config(CONFIG)
        self.assertEqual(tuple(config["dataset"]["load_bands"].values()), BANDS)
        self.assertEqual(config["checkpoint_generation"]["expected_candidate_count"], 36)
        self.assertEqual(config["checkpoint_generation"]["minimum_qualified_per_load_band"], 10)
        self.assertEqual(len(set(config["excluded_map_sha256"])), 12)
        text = CONFIG.read_text(encoding="utf-8")
        self.assertNotIn("warehouse-10-20-10-2-1", text)
        self.assertNotIn("v2first", text.lower())

    def test_dry_run_is_solver_free(self) -> None:
        result = dry_run(CONFIG)
        self.assertEqual(result["map_count"], 6)
        self.assertEqual(result["task_count"], 18)
        self.assertEqual(result["checkpoint_candidate_count"], 36)
        self.assertEqual(result["maximum_timed_episode_count"], 72)
        self.assertFalse(result["native_or_controller_invoked"])

    def test_rotating_strict_serial_schedule(self) -> None:
        rows = []
        for key in range(36):
            rows.append({"key_index": key, "checkpoint_id": f"cp-{key}", "checkpoint_identity_sha256": f"{key:064x}",
                "map_id": f"map-{key//6}", "task_id": f"task-{key//2}", "task_variant": "balanced_od_d10",
                "load_band": BANDS[(key//2)%3], "disturbance_replica": key%2, "screen_solver_seed": 2026091400+key})
        schedule = _schedule(rows)
        self.assertEqual(len(schedule), 72)
        for key in range(36):
            pair = [r for r in schedule if r["key_index"] == key]
            self.assertEqual({r["controller"] for r in pair}, set(CONTROLLERS))
            self.assertEqual([r["controller"] for r in pair], list(CONTROLLERS) if key % 2 == 0 else list(reversed(CONTROLLERS)))

    def test_band_gate_and_map_cluster_summary(self) -> None:
        rows = []
        for key in range(12):
            for controller, ttf in (("official_adaptive", 10.0), ("dual16", 7.0)):
                rows.append({"checkpoint_id": f"cp-{key}", "controller": controller, "map_id": f"map-{key//2}",
                    "summary": {"capped_wall_time_to_feasible": ttf, "success": True, "external_timeout": False}})
        summaries = {"official_adaptive": {"successes_per_observed_hour": 360.0, "success_rate": 1.0},
                     "dual16": {"successes_per_observed_hour": 500.0, "success_rate": 1.0}}
        gate = {"minimum_paired_win_rate": .70, "minimum_mean_capped_ttf_improvement": .15,
                "minimum_successes_per_hour_improvement": .20, "maximum_success_rate_loss": 0.0}
        result = _comparison(rows, summaries, gate)
        self.assertTrue(result["screen_gate_passed"])
        self.assertEqual(result["paired_win_count"], 12)
        self.assertEqual(result["map_cluster_count"], 6)
        self.assertAlmostEqual(result["mean_capped_ttf_improvement"], .30)

    def test_additional_censor_rejects(self) -> None:
        rows = []
        for controller, success in (("official_adaptive", True), ("dual16", False)):
            rows.append({"checkpoint_id": "cp", "controller": controller, "map_id": "map",
                "summary": {"capped_wall_time_to_feasible": 10.0 if success else 60.0, "success": success, "external_timeout": not success}})
        summaries = {"official_adaptive": {"successes_per_observed_hour": 360.0, "success_rate": 1.0},
                     "dual16": {"successes_per_observed_hour": 0.0, "success_rate": 0.0}}
        gate = {"minimum_paired_win_rate": .70, "minimum_mean_capped_ttf_improvement": .15,
                "minimum_successes_per_hour_improvement": .20, "maximum_success_rate_loss": 0.0}
        result = _comparison(rows, summaries, gate)
        self.assertFalse(result["screen_gate_passed"])
        self.assertEqual(result["additional_censor_count"], 1)

    def test_zero_official_throughput_with_dual16_success_is_infinite_gain(self) -> None:
        rows = []
        for key in range(4):
            rows.extend((
                {"checkpoint_id": f"cp-{key}", "controller": "official_adaptive", "map_id": f"map-{key}",
                 "summary": {"capped_wall_time_to_feasible": 60.0, "success": False, "external_timeout": False}},
                {"checkpoint_id": f"cp-{key}", "controller": "dual16", "map_id": f"map-{key}",
                 "summary": {"capped_wall_time_to_feasible": 10.0, "success": True, "external_timeout": False}},
            ))
        summaries = {"official_adaptive": {"successes_per_observed_hour": 0.0, "success_rate": 0.0},
                     "dual16": {"successes_per_observed_hour": 360.0, "success_rate": 1.0}}
        gate = {"minimum_paired_win_rate": .70, "minimum_mean_capped_ttf_improvement": .15,
                "minimum_successes_per_hour_improvement": .20, "maximum_success_rate_loss": 0.0}
        result = _comparison(rows, summaries, gate)
        self.assertEqual(result["throughput_improvement"], float("inf"))
        self.assertTrue(result["screen_gate_passed"])


if __name__ == "__main__":
    unittest.main()
