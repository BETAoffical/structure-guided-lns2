from __future__ import annotations

import unittest
from pathlib import Path

from experiments.repair_collection import _read_json
from experiments.stride_structpool_fresh_congestion_recovery import (
    summarize_congestion_loads,
    validate_structpool_fresh_congestion_recovery,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_fresh_congestion_recovery.json"


class StructPoolFreshCongestionRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _read_json(CONFIG)
        validate_structpool_fresh_congestion_recovery(cls.config, project_root=ROOT)

    def test_derived_task_and_outcome_boundaries_are_frozen(self) -> None:
        self.assertEqual(
            self.config["task_semantics"],
            "derived_opposite_exchange_not_official_movingai_scenario",
        )
        self.assertFalse(
            self.config["outcome_boundary"]["candidate_repair_outcomes_read"]
        )
        self.assertFalse(self.config["outcome_boundary"]["controller_outcomes_read"])
        self.assertFalse(self.config["outcome_boundary"]["ttf_outcomes_read"])
        self.assertEqual(
            [row["agent_counts"] for row in self.config["candidate_ladders"]],
            [[100, 200, 300], [200, 400, 600], [200, 400, 600]],
        )

    def test_lowest_complete_task_seed_covered_load_is_selected(self) -> None:
        manifests = []
        results = []
        for ladder in self.config["candidate_ladders"]:
            for load_index, agents in enumerate(ladder["agent_counts"]):
                for task_seed in (233, 277):
                    task_id = (
                        f"{ladder['map_id']}__opposite_exchange_"
                        f"{task_seed}__agents_{agents:04d}"
                    )
                    manifests.append(
                        {
                            "task_id": task_id,
                            "map_id": ladder["map_id"],
                            "agent_count": agents,
                            "task_seed": task_seed,
                        }
                    )
                    for solver_seed in (1, 2, 3):
                        conflicts = (
                            20
                            if load_index > 0
                            or (task_seed, solver_seed)
                            in {(233, 1), (233, 2), (277, 1)}
                            else 0
                        )
                        results.append(
                            {
                                "task_id": task_id,
                                "map_id": ladder["map_id"],
                                "agent_count": agents,
                                "solver_seed": solver_seed,
                                "status": "ok",
                                "initial_complete": True,
                                "initial_conflicts": conflicts,
                                "state_fingerprint": f"{task_id}-{solver_seed}",
                            }
                        )
        summaries, selected, errors, forbidden = summarize_congestion_loads(
            manifests, results, self.config
        )
        self.assertEqual(len(summaries), 9)
        self.assertFalse(errors)
        self.assertFalse(forbidden)
        self.assertEqual([row["agent_count"] for row in selected], [100, 200, 200])
        self.assertTrue(all(row["gate_eligible_reset_count"] == 3 for row in selected))
        self.assertTrue(
            all(row["gate_eligible_task_seeds"] == [233, 277] for row in selected)
        )


if __name__ == "__main__":
    unittest.main()
