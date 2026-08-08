from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.stride_guardpool_maze_regression import (
    CONTROLLERS,
    _controller_kwargs,
    guardpool_maze_schedule,
    load_guardpool_maze_regression_config,
    run_guardpool_maze_regression,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_guardpool_maze_regression_v1.json"


class GuardPoolMazeRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.path, cls.root, cls.config = load_guardpool_maze_regression_config(
            CONFIG
        )

    def test_schedule_is_the_exact_four_controller_known_key(self) -> None:
        schedule = guardpool_maze_schedule(self.config)
        self.assertEqual(tuple(row["controller"] for row in schedule), CONTROLLERS)
        self.assertEqual(len(schedule), 4)
        self.assertEqual({row["solver_seed"] for row in schedule}, {3})
        self.assertEqual(
            {row["task_id"] for row in schedule},
            {
                "maze-128-128-1__derived_opposite_exchange__task_seed_0233__agents_0100"
            },
        )

    def test_slotpool_and_guardpool_differ_only_by_the_frozen_guard(self) -> None:
        slot = _controller_kwargs(self.root, self.config, "v2-plus-slotpool")
        guard = _controller_kwargs(self.root, self.config, "stride-guardpool-v1")
        slot_config = dict(slot["structpool_augmentation"])
        guard_config = dict(guard["structpool_augmentation"])
        self.assertNotIn("stall_guard", slot_config)
        self.assertEqual(guard_config.pop("stall_guard")["no_progress_limit"], 8)
        slot_config["runtime_id"] = guard_config["runtime_id"]
        self.assertEqual(slot_config, guard_config)

    def test_dry_run_does_not_execute_the_solver(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = run_guardpool_maze_regression(
                CONFIG, Path(directory) / "output", dry_run=True
            )
        self.assertEqual(result["schedule_entry_count"], 4)

    def test_registration_rejects_regression_gate_drift(self) -> None:
        changed = json.loads(CONFIG.read_text(encoding="utf-8"))
        changed["gates"]["guardpool_maximum_repair_iterations"] = 31
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "gates changed"):
                load_guardpool_maze_regression_config(path)


if __name__ == "__main__":
    unittest.main()
