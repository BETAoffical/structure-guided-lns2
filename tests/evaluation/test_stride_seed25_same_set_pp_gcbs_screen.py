from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.stride_seed25_same_set_pp_gcbs_screen import (
    ALGORITHMS,
    EXPERIMENT_ID,
    build_plan,
    load_registration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_seed25_same_set_pp_gcbs_screen_v1.json"


class SameSetPpGcbsScreenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = build_plan(CONFIG)

    def test_plan_is_exactly_four_states_and_eight_one_step_branches(self) -> None:
        self.assertEqual(self.plan["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(self.plan["state_count"], 4)
        self.assertEqual(self.plan["branch_count"], 8)
        self.assertEqual(tuple(self.plan["repair_algorithms"]), ALGORITHMS)
        self.assertTrue(self.plan["one_step_only"])
        self.assertEqual(
            [row["selected_action"]["role"] for row in self.plan["states"]],
            ["component16", "observed_dual_winner", "component16", "component16"],
        )
        self.assertEqual(
            [row["decision_index"] for row in self.plan["states"]],
            [31, 48, 30, 0],
        )
        for row in self.plan["states"]:
            self.assertEqual(row["selected_action"]["agents"], row["dual_action_agents"])
            self.assertIsNone(row["candidates"])

    def test_claim_and_qualification_boundaries_are_fail_closed(self) -> None:
        self.assertFalse(self.plan["claim_boundary"]["ttf_claim_allowed"])
        self.assertFalse(
            self.plan["claim_boundary"]["repairer_runtime_promotion_allowed"]
        )
        self.assertFalse(self.plan["qualification_boundary"]["pbs_included"])
        self.assertFalse(
            self.plan["qualification_boundary"][
                "gcbs_cooperative_action_deadline_validated"
            ]
        )
        self.assertTrue(
            self.plan["qualification_boundary"]["gcbs_external_process_fuse_required"]
        )

    def test_registration_rejects_algorithm_expansion(self) -> None:
        payload = json.loads(CONFIG.read_text(encoding="utf-8"))
        payload["execution"]["repair_algorithms"].append("PBS")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stride_seed25_same_set_pp_gcbs_screen_v1.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_registration(path)


if __name__ == "__main__":
    unittest.main()
