from __future__ import annotations

import unittest
from pathlib import Path

from experiments.repair_collection import _read_json
from experiments.stride_structpool_fresh_load_recovery import (
    summarize_recovery_loads,
    validate_structpool_fresh_load_recovery,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_fresh_load_recovery.json"


class StructPoolFreshLoadRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _read_json(CONFIG)
        validate_structpool_fresh_load_recovery(cls.config, project_root=ROOT)

    def test_outcome_boundary_and_ladders_are_frozen(self) -> None:
        self.assertTrue(self.config["outcome_boundary"]["outcome_informed"])
        self.assertFalse(
            self.config["outcome_boundary"]["candidate_repair_outcomes_read"]
        )
        self.assertFalse(self.config["outcome_boundary"]["ttf_outcomes_read"])
        self.assertEqual(
            [row["agent_counts"] for row in self.config["candidate_ladders"]],
            [[400, 600, 800], [600, 800, 1000], [600, 800, 1000]],
        )

    def test_lowest_complete_gate_eligible_load_is_selected(self) -> None:
        manifests = []
        results = []
        for ladder in self.config["candidate_ladders"]:
            for load_index, agents in enumerate(ladder["agent_counts"]):
                for scenario in (4, 5):
                    task_id = (
                        f"{ladder['map_id']}__random_{scenario:02d}__agents_{agents:04d}"
                    )
                    manifests.append(
                        {
                            "task_id": task_id,
                            "map_id": ladder["map_id"],
                            "agent_count": agents,
                        }
                    )
                    for seed in (1, 2, 3):
                        conflicts = 20 if load_index > 0 or (scenario, seed) in {
                            (4, 1),
                            (4, 2),
                            (5, 1),
                        } else 0
                        results.append(
                            {
                                "task_id": task_id,
                                "map_id": ladder["map_id"],
                                "agent_count": agents,
                                "solver_seed": seed,
                                "status": "ok",
                                "initial_complete": True,
                                "initial_conflicts": conflicts,
                                "state_fingerprint": f"{task_id}-{seed}",
                            }
                        )
        summaries, selected, errors, forbidden = summarize_recovery_loads(
            manifests, results, self.config
        )
        self.assertEqual(len(summaries), 9)
        self.assertFalse(errors)
        self.assertFalse(forbidden)
        self.assertEqual(
            [row["agent_count"] for row in selected], [400, 600, 600]
        )
        self.assertTrue(all(row["gate_eligible_reset_count"] == 3 for row in selected))


if __name__ == "__main__":
    unittest.main()
