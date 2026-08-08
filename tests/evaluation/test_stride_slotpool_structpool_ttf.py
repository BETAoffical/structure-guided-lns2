from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.stride_slotpool_structpool_ttf import (
    CONTROLLERS,
    load_slotpool_structpool_ttf_config,
    slotpool_structpool_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_slotpool_structpool_ttf_diagnostic_v1.json"


class SlotPoolStructPoolTtfTests(unittest.TestCase):
    def test_registered_config_and_schedule(self) -> None:
        _path, _root, config = load_slotpool_structpool_ttf_config(CONFIG)
        schedule = slotpool_structpool_schedule(config)
        self.assertEqual(len(schedule), 58)
        keys = {
            (row["group_id"], row["task_id"], row["solver_seed"])
            for row in schedule
        }
        self.assertEqual(len(keys), 29)
        excluded = (
            "maze100",
            "maze-128-128-1__derived_opposite_exchange__task_seed_0233__agents_0100",
            3,
        )
        self.assertNotIn(excluded, keys)
        for key in keys:
            rows = [
                row
                for row in schedule
                if (row["group_id"], row["task_id"], row["solver_seed"]) == key
            ]
            self.assertEqual({row["controller"] for row in rows}, set(CONTROLLERS))
            self.assertEqual({row["within_key_position"] for row in rows}, {0, 1})

    def test_only_registered_exclusion_is_removed(self) -> None:
        _path, _root, config = load_slotpool_structpool_ttf_config(CONFIG)
        changed = copy.deepcopy(config)
        changed["cohort"]["excluded_keys"].append(
            {
                "group_id": "den300",
                "task_id": "den312d__random_04__agents_0300",
                "solver_seed": 1,
                "reason": "unregistered",
            }
        )
        from experiments.stride_slotpool_structpool_ttf import _expected_keys

        self.assertEqual(len(_expected_keys(changed)), 28)


if __name__ == "__main__":
    unittest.main()
