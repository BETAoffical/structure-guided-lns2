from __future__ import annotations

import unittest
from pathlib import Path

from experiments.repair_collection import _read_json
from experiments.stride_structpool_fresh_congestion_recovery import (
    summarize_congestion_loads,
)
from experiments.stride_structpool_warehouse_game_map_swap import (
    validate_structpool_warehouse_game_map_swap,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_warehouse_game_map_swap.json"


class StructPoolWarehouseGameMapSwapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _read_json(CONFIG)
        validate_structpool_warehouse_game_map_swap(cls.config, project_root=ROOT)

    def test_only_maps_change(self) -> None:
        self.assertEqual(
            self.config["unchanged_inputs"]["task_generator"],
            "opposite_exchange",
        )
        self.assertEqual(
            self.config["unchanged_inputs"]["agent_counts"], [200, 400, 600]
        )
        self.assertEqual(
            {row["replacement_map"] for row in self.config["map_swaps"]},
            {"warehouse-10-20-10-2-1", "lt_hangedman"},
        )
        self.assertFalse(self.config["outcome_boundary"]["ttf_outcomes_read"])

    def test_lowest_complete_load_per_map_is_selected(self) -> None:
        manifests = []
        results = []
        for map_id in ("warehouse-10-20-10-2-1", "lt_hangedman"):
            for load_index, agents in enumerate((200, 400, 600)):
                for task_seed in (233, 277):
                    task_id = f"{map_id}-{task_seed}-{agents}"
                    manifests.append(
                        {
                            "task_id": task_id,
                            "map_id": map_id,
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
                                "map_id": map_id,
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
        self.assertEqual(len(summaries), 6)
        self.assertFalse(errors)
        self.assertFalse(forbidden)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["agent_count"] == 200 for row in selected))
        self.assertTrue(
            all(row["gate_eligible_task_seeds"] == [233, 277] for row in selected)
        )


if __name__ == "__main__":
    unittest.main()
