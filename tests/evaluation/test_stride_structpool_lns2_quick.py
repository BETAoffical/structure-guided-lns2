from __future__ import annotations

import unittest
from pathlib import Path

from experiments.stride_structpool_lns2_quick import (
    CONTROLLERS,
    load_structpool_lns2_quick_config,
    structpool_lns2_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_lns2_quick.json"


class StructPoolLns2QuickTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _path, _root, cls.config = load_structpool_lns2_quick_config(CONFIG)

    def test_schedule_is_complete_paired_and_strictly_rotated(self) -> None:
        schedule = structpool_lns2_schedule(self.config)
        self.assertEqual(len(schedule), 24)
        groups = [schedule[index : index + 3] for index in range(0, 24, 3)]
        self.assertEqual(len(groups), 8)
        expected_orders = [
            CONTROLLERS[index % len(CONTROLLERS) :]
            + CONTROLLERS[: index % len(CONTROLLERS)]
            for index in range(8)
        ]
        for rows, expected in zip(groups, expected_orders):
            keys = {
                (row["group_id"], row["task_id"], row["solver_seed"])
                for row in rows
            }
            self.assertEqual(len(keys), 1)
            self.assertEqual(tuple(row["controller"] for row in rows), expected)

    def test_official_lns2_and_claim_boundary_are_explicit(self) -> None:
        self.assertEqual(
            self.config["comparison"]["lns2_baseline"],
            "native_official_adaptive_neighborhood_generation",
        )
        self.assertEqual(self.config["runtime"]["stopping_rule"], "run-to-completion")
        self.assertFalse(self.config["claim_boundary"]["formal_speed_claim"])
        self.assertFalse(self.config["claim_boundary"]["default_replacement_allowed"])


if __name__ == "__main__":
    unittest.main()
