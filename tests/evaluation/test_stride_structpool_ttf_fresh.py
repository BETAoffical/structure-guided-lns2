from __future__ import annotations

import unittest
from pathlib import Path

from experiments.stride_structpool_ttf_fresh import (
    load_structpool_ttf_fresh_config,
    structpool_fresh_schedule,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_ttf_fresh.json"


class StructPoolTtfFreshTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _path, _root, cls.config = load_structpool_ttf_fresh_config(CONFIG)

    def test_schedule_is_complete_paired_and_alternating(self) -> None:
        rows = structpool_fresh_schedule(self.config)
        self.assertEqual(len(rows), 72)
        self.assertEqual({row["solver_seed"] for row in rows}, {1, 2, 3})
        self.assertEqual(len({row["group_id"] for row in rows}), 6)
        for index in range(0, len(rows), 2):
            pair = rows[index : index + 2]
            self.assertEqual(
                {
                    (row["group_id"], row["task_id"], row["solver_seed"])
                    for row in pair
                },
                {(pair[0]["group_id"], pair[0]["task_id"], pair[0]["solver_seed"])},
            )
            self.assertEqual(
                {row["controller"] for row in pair},
                set(self.config["controllers"]),
            )

    def test_cross_layout_gate_and_claim_boundary_are_frozen(self) -> None:
        self.assertEqual(
            self.config["performance_gates"][
                "minimum_mean_raw_ttf_improvement_vs_v2"
            ],
            0.05,
        )
        self.assertEqual(
            self.config["performance_gates"][
                "maximum_group_raw_ttf_regression"
            ],
            0.10,
        )
        boundary = self.config["claim_boundary"]
        self.assertFalse(boundary["new_ranker_trained"])
        self.assertFalse(boundary["formal_speed_claim"])
        self.assertFalse(boundary["default_replacement_allowed"])
        self.assertTrue(
            boundary[
                "fresh_maps_not_used_by_structpool_label_or_development_ttf_cohorts"
            ]
        )


if __name__ == "__main__":
    unittest.main()
