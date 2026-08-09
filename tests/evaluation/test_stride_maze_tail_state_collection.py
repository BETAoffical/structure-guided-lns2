from __future__ import annotations

import unittest
from pathlib import Path

from experiments.stride_maze_tail_full_episode import maze_tail_full_episode_schedule
from experiments.stride_maze_tail_state_collection import (
    _classify_pair,
    load_maze_tail_state_collection_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_maze_tail_state_collection_v2.json"


class MazeTailStateCollectionTests(unittest.TestCase):
    def test_registered_fuse_and_schedule(self) -> None:
        _path, _root, config = load_maze_tail_state_collection_config(CONFIG)
        self.assertEqual(len(maze_tail_full_episode_schedule(config)), 99)
        self.assertEqual(config["runtime"]["maximum_repair_decisions"], 200)
        self.assertEqual(config["runtime"]["wall_time_budget_seconds"], 300.0)
        self.assertFalse(config["claim_boundary"]["training_allowed"])
        self.assertFalse(config["claim_boundary"]["run_to_feasibility_claim"])

    def test_challenger_only_censoring_is_severe(self) -> None:
        _path, _root, config = load_maze_tail_state_collection_config(CONFIG)
        category, reason = _classify_pair(
            config["fixed_horizon_tail_definition"],
            {"success": True, "repair_iterations": 20},
            {"success": False, "repair_iterations": 200},
        )
        self.assertEqual(category, "severe")
        self.assertEqual(reason, "challenger_censored_v2_complete")

    def test_both_censored_use_fixed_horizon_metrics(self) -> None:
        _path, _root, config = load_maze_tail_state_collection_config(CONFIG)
        category, reason = _classify_pair(
            config["fixed_horizon_tail_definition"],
            {
                "success": False,
                "normalized_fixed_budget_conflict_auc": 0.30,
                "final_conflicts": 10,
            },
            {
                "success": False,
                "normalized_fixed_budget_conflict_auc": 0.50,
                "final_conflicts": 20,
            },
        )
        self.assertEqual(category, "severe")
        self.assertEqual(reason, "both_censored_fixed_horizon_severe")

    def test_censored_ttf_is_not_imputed(self) -> None:
        _path, _root, config = load_maze_tail_state_collection_config(CONFIG)
        self.assertTrue(config["claim_boundary"]["censored_ttf_is_not_imputed"])
        self.assertTrue(
            config["evidence_fuse"]["repair_limit_or_wall_timeout_is_valid_right_censoring"]
        )


if __name__ == "__main__":
    unittest.main()
