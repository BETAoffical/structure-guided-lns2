from __future__ import annotations

import unittest
from pathlib import Path

from experiments.stride_structpool_ttf_confirmation import (
    load_structpool_ttf_confirmation_config,
    structpool_confirmation_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_ttf_confirmation.json"


class StructPoolTtfConfirmationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _path, _root, cls.config = load_structpool_ttf_confirmation_config(CONFIG)

    def test_extension_schedule_is_complete_and_alternating(self) -> None:
        rows = structpool_confirmation_schedule(self.config)
        self.assertEqual(len(rows), 16)
        self.assertEqual({row["solver_seed"] for row in rows}, {3, 4})
        for index in range(0, len(rows), 2):
            pair = rows[index : index + 2]
            self.assertEqual(
                {(row["group_id"], row["task_id"], row["solver_seed"]) for row in pair},
                {(pair[0]["group_id"], pair[0]["task_id"], pair[0]["solver_seed"])},
            )
            self.assertEqual(set(row["controller"] for row in pair), set(self.config["controllers"]))

    def test_confirmation_keeps_method_and_claim_boundary(self) -> None:
        self.assertEqual(self.config["cohort"]["pooled_solver_seeds"], [1, 2, 3, 4])
        self.assertFalse(self.config["claim_boundary"]["new_ranker_trained"])
        self.assertFalse(self.config["claim_boundary"]["formal_speed_claim"])
        self.assertEqual(
            self.config["pooled_gates"]["minimum_mean_raw_ttf_improvement_vs_v2"],
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
