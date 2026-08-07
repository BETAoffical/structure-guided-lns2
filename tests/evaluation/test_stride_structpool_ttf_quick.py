from __future__ import annotations

import unittest
from pathlib import Path

from experiments.stride_structpool_ttf_quick import (
    load_structpool_ttf_quick_config,
    structpool_ttf_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_ttf_quick.json"


class StructPoolTtfQuickTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _path, _root, cls.config = load_structpool_ttf_quick_config(CONFIG)

    def test_schedule_is_complete_paired_and_alternating(self) -> None:
        schedule = structpool_ttf_schedule(self.config)
        self.assertEqual(len(schedule), 16)
        pairs = [schedule[index : index + 2] for index in range(0, len(schedule), 2)]
        self.assertEqual(len(pairs), 8)
        for index, pair in enumerate(pairs):
            self.assertEqual(
                {(row["group_id"], row["task_id"], row["solver_seed"]) for row in pair},
                {(pair[0]["group_id"], pair[0]["task_id"], pair[0]["solver_seed"])},
            )
            expected = (
                ["v2-full", "v2-plus-structpool"]
                if index % 2 == 0
                else ["v2-plus-structpool", "v2-full"]
            )
            self.assertEqual([row["controller"] for row in pair], expected)

    def test_ranker_and_raw_ttf_contract_are_frozen(self) -> None:
        self.assertFalse(self.config["comparison"]["ranker_changed"])
        self.assertEqual(self.config["runtime"]["stopping_rule"], "run-to-completion")
        self.assertIsNone(self.config["runtime"]["scientific_time_limit_seconds"])
        self.assertIsNone(self.config["runtime"]["environment_time_limit_seconds"])
        self.assertFalse(self.config["claim_boundary"]["formal_speed_claim"])

    def test_performance_gate_is_preregistered(self) -> None:
        self.assertEqual(
            self.config["performance_gates"][
                "minimum_mean_raw_ttf_improvement_vs_v2"
            ],
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
