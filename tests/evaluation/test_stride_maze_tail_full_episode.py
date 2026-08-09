from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.stride_maze_tail_full_episode import (
    CONTROLLERS,
    load_maze_tail_full_episode_config,
    maze_tail_full_episode_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_maze_tail_full_episode_v1.json"


class MazeTailFullEpisodeTests(unittest.TestCase):
    def test_registered_config_and_three_way_rotation(self) -> None:
        _path, _root, config = load_maze_tail_full_episode_config(CONFIG)
        schedule = maze_tail_full_episode_schedule(config)
        self.assertEqual(len(schedule), 99)
        keys = {
            (row["group_id"], row["task_id"], row["solver_seed"])
            for row in schedule
        }
        self.assertEqual(len(keys), 33)
        orders = []
        for key in sorted(keys):
            rows = [
                row
                for row in schedule
                if (row["group_id"], row["task_id"], row["solver_seed"]) == key
            ]
            rows.sort(key=lambda row: row["within_key_position"])
            self.assertEqual({row["controller"] for row in rows}, set(CONTROLLERS))
            self.assertEqual({row["within_key_position"] for row in rows}, {0, 1, 2})
            orders.append(tuple(row["controller"] for row in rows))
        self.assertEqual(
            set(orders),
            {
                CONTROLLERS,
                CONTROLLERS[1:] + CONTROLLERS[:1],
                CONTROLLERS[2:] + CONTROLLERS[:2],
            },
        )

    def test_result_based_exclusion_is_not_permitted(self) -> None:
        _path, _root, config = load_maze_tail_full_episode_config(CONFIG)
        changed = copy.deepcopy(config)
        changed["cohort"]["result_based_exclusions"].append(
            {"task_id": changed["cohort"]["groups"][0]["tasks"][0]}
        )
        self.assertTrue(changed["cohort"]["result_based_exclusions"])
        self.assertEqual(len(maze_tail_full_episode_schedule(config)), 99)

    def test_tail_gate_and_claim_boundary_are_frozen(self) -> None:
        _path, _root, config = load_maze_tail_full_episode_config(CONFIG)
        self.assertFalse(config["claim_boundary"]["training_allowed"])
        self.assertFalse(config["claim_boundary"]["formal_speed_claim"])
        self.assertEqual(
            config["tail_definition"]["severe_minimum_repair_iteration_delta"],
            10,
        )
        self.assertEqual(
            config["tail_evidence_gates"]["minimum_severe_tail_comparisons"],
            4,
        )


if __name__ == "__main__":
    unittest.main()
