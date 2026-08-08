from __future__ import annotations

import unittest
from pathlib import Path

from experiments.stride_safeslot_v2_baseline import (
    load_safeslot_v2_baseline_config,
    safeslot_v2_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_safeslot_v2_baseline_v1.json"


class SafeSlotV2BaselineTests(unittest.TestCase):
    def test_registered_same_runtime_schedule(self) -> None:
        _path, _root, config, expected = load_safeslot_v2_baseline_config(CONFIG)
        schedule = safeslot_v2_schedule(expected)
        self.assertEqual(len(schedule), 29)
        self.assertEqual(
            len({(row["task_id"], row["solver_seed"]) for row in schedule}), 29
        )
        self.assertEqual({row["controller"] for row in schedule}, {"v2-full"})
        self.assertFalse(config["claim_boundary"]["wall_clock_comparison_allowed"])
        self.assertFalse(config["claim_boundary"]["model_training_allowed"])


if __name__ == "__main__":
    unittest.main()
