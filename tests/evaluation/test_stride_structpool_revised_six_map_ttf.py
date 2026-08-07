from __future__ import annotations

import unittest
from pathlib import Path

from experiments.stride_structpool_revised_six_map_ttf import (
    load_revised_six_map_ttf_config,
    revised_six_map_ttf_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_revised_six_map_ttf.json"


class StructPoolRevisedSixMapTtfTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _path, _root, cls.config = load_revised_six_map_ttf_config(CONFIG)

    def test_schedule_is_complete_paired_and_strictly_alternating(self) -> None:
        rows = revised_six_map_ttf_schedule(self.config)
        self.assertEqual(len(rows), 72)
        self.assertEqual({row["solver_seed"] for row in rows}, {1, 2, 3})
        self.assertEqual(len({row["group_id"] for row in rows}), 6)
        for index in range(0, len(rows), 2):
            pair = rows[index : index + 2]
            self.assertEqual(
                len(
                    {
                        (row["group_id"], row["task_id"], row["solver_seed"])
                        for row in pair
                    }
                ),
                1,
            )
            self.assertEqual(
                {row["controller"] for row in pair}, set(self.config["controllers"])
            )
            expected_first = "v2-full" if index // 2 % 2 == 0 else "v2-plus-structpool"
            self.assertEqual(pair[0]["controller"], expected_first)

    def test_only_candidate_pool_changes(self) -> None:
        comparison = self.config["comparison"]
        self.assertFalse(comparison["ranker_changed"])
        self.assertTrue(comparison["candidate_pool_only_difference"])
        self.assertEqual(
            comparison["deterministic_pp_seed_contract"],
            "candidate_bound_native_replay",
        )
        self.assertEqual(self.config["runtime"]["stopping_rule"], "run-to-completion")
        self.assertIsNone(self.config["runtime"]["scientific_time_limit_seconds"])

    def test_claim_is_qualification_conditioned(self) -> None:
        boundary = self.config["claim_boundary"]
        self.assertTrue(boundary["qualification_conditioned"])
        self.assertFalse(boundary["fresh_ood_or_generalization_claim"])
        self.assertFalse(boundary["formal_speed_claim"])
        self.assertFalse(boundary["default_replacement_allowed"])


if __name__ == "__main__":
    unittest.main()
