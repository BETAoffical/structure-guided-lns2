from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_boundary_gate_ablation import (
    CONTROLLERS,
    boundary_gate_ablation_schedule,
    validate_boundary_gate_ablation_config,
)


class StrideBoundaryGateAblationTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (
                root
                / "configs"
                / "stride_boundary_gate_ablation_run_to_completion.json"
            ).read_text(encoding="utf-8")
        )

    def test_contract_is_uncapped_and_non_promoting(self) -> None:
        config = self._config()
        validate_boundary_gate_ablation_config(config)
        self.assertEqual(config["stopping_rule"], "run-to-completion")
        self.assertIsNone(config["scientific_time_limit_seconds"])
        self.assertIsNone(config["environment_time_limit_seconds"])
        self.assertIsNone(config["episode_process_timeout_seconds"])
        self.assertEqual(config["verification_profile"], "deployment")
        self.assertFalse(config["formal_speed_claim"])
        self.assertFalse(config["default_replacement_allowed"])
        self.assertFalse(config["training_allowed"])

    def test_schedule_is_complete_paired_and_rotating(self) -> None:
        schedule = boundary_gate_ablation_schedule(self._config())
        self.assertEqual(len(schedule), 54)
        by_key: dict[tuple[str, int], list[dict]] = {}
        for row in schedule:
            by_key.setdefault((row["task_id"], row["solver_seed"]), []).append(row)
        self.assertEqual(len(by_key), 18)
        self.assertTrue(
            all({row["controller"] for row in rows} == set(CONTROLLERS) for rows in by_key.values())
        )
        first = [rows[0]["controller"] for rows in by_key.values()]
        self.assertTrue(all(first.count(controller) == 6 for controller in CONTROLLERS))

    def test_contract_rejects_time_and_gate_drift(self) -> None:
        config = self._config()
        config["scientific_time_limit_seconds"] = 60.0
        with self.assertRaisesRegex(ValueError, "runtime contract changed"):
            validate_boundary_gate_ablation_config(config)
        config = self._config()
        config["topology_boundary_augmentations"][
            "stride-boundary-stall-guard-v1"
        ]["phase_guard"]["maximum_no_progress_streak"] = 6
        with self.assertRaisesRegex(ValueError, "augmentations changed"):
            validate_boundary_gate_ablation_config(config)


if __name__ == "__main__":
    unittest.main()
