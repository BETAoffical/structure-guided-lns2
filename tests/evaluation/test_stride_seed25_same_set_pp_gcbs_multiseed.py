from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.stride_seed25_same_set_pp_gcbs_multiseed import (
    EXPERIMENT_ID,
    _decision,
    build_plan,
    load_registration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_seed25_same_set_pp_gcbs_multiseed_v1.json"


class SameSetPpGcbsMultiseedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan(CONFIG)
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_plan_is_four_states_sixteen_pairs_and_thirty_two_attempts(self) -> None:
        self.assertEqual(self.plan["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(self.plan["state_count"], 4)
        self.assertEqual(self.plan["pair_count"], 16)
        self.assertEqual(self.plan["attempt_count"], 32)
        self.assertEqual(self.plan["maximum_native_action_budget_seconds"], 160.0)
        self.assertEqual(
            [row["repair_seed"] for row in self.plan["pairs"]],
            [
                923011459, 1819703210, 384107260, 2013200100,
                401095199, 585308340, 521859152, 2051380339,
                2080532616, 2136352329, 649325684, 1580664547,
                1197558189, 559692226, 1393063763, 1888071520,
            ],
        )
        for index, pair in enumerate(self.plan["pairs"]):
            expected = ["PP", "GCBS"] if index % 2 == 0 else ["GCBS", "PP"]
            self.assertEqual(pair["algorithm_order"], expected)

    def test_registration_requires_frozen_qualification(self) -> None:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["source"]["screen_registration"] = str(
            PROJECT_ROOT / payload["source"]["screen_registration"]
        )
        payload["qualification"]["report_path"] = str(
            PROJECT_ROOT / payload["qualification"]["report_path"]
        )
        payload["qualification"]["report_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / CONFIG.name
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_registration(path)

    def test_scientific_gate_accepts_only_broad_gcbs_improvement(self) -> None:
        rows = []
        maps = [
            "maze-32-32-4-n300",
            "random-32-32-20-high-load",
            "room-64-64-16",
            "warehouse-w1020a-opposite-exchange",
        ]
        for map_id in maps:
            for trial in range(4):
                for algorithm in ("PP", "GCBS"):
                    strict = algorithm == "GCBS" or (map_id == maps[3] and trial < 3)
                    rows.append(
                        {
                            "map_id": map_id,
                            "repair_algorithm": algorithm,
                            "strict_drop": strict,
                            "rollback_exact": not strict,
                            "failure_reason": "no_improvement" if not strict else "none",
                            "conflict_reduction": 2 if strict else 0,
                            "normalized_conflict_reduction": 0.02 if strict else 0.0,
                            "native_replan_seconds": 0.1,
                            "after_repair_structure_fingerprint": f"{algorithm}-{trial}",
                        }
                    )
        decision = _decision(rows, self.config)
        self.assertTrue(decision["passed"])
        for row in rows:
            if row["map_id"] == maps[1] and row["repair_algorithm"] == "GCBS":
                row["strict_drop"] = False
                row["rollback_exact"] = True
                row["conflict_reduction"] = 0
                row["normalized_conflict_reduction"] = 0.0
        failed = _decision(rows, self.config)
        self.assertFalse(failed["passed"])
        self.assertIn(
            "gcbs_hard_state_drop_floor_failed:random-32-32-20-high-load",
            failed["failures"],
        )


if __name__ == "__main__":
    unittest.main()
