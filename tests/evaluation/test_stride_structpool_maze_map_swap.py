from __future__ import annotations

import unittest
from pathlib import Path

from experiments.repair_collection import _read_json
from experiments.stride_structpool_fresh_congestion_recovery import (
    summarize_congestion_loads,
)
from experiments.stride_structpool_maze_map_swap import (
    validate_structpool_maze_map_swap,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_maze_map_swap.json"


class StructPoolMazeMapSwapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = _read_json(CONFIG)
        validate_structpool_maze_map_swap(cls.config, project_root=ROOT)

    def test_only_map_layout_changes(self) -> None:
        self.assertEqual(
            self.config["map_swap"],
            {
                "replaced_map": "maze-128-128-10",
                "replacement_map": "maze-128-128-1",
                "only_intended_changed_variable": "map_layout",
            },
        )
        self.assertEqual(
            self.config["unchanged_inputs"]["task_generator"],
            "opposite_exchange",
        )
        self.assertEqual(
            self.config["unchanged_inputs"]["agent_counts"], [100, 200, 300]
        )
        self.assertFalse(self.config["outcome_boundary"]["ttf_outcomes_read"])

    def test_lowest_complete_load_is_selected(self) -> None:
        manifests = []
        results = []
        for load_index, agents in enumerate((100, 200, 300)):
            for task_seed in (233, 277):
                task_id = f"maze-1-{task_seed}-{agents}"
                manifests.append(
                    {
                        "task_id": task_id,
                        "map_id": "maze-128-128-1",
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
                            "map_id": "maze-128-128-1",
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
        self.assertEqual(len(summaries), 3)
        self.assertFalse(errors)
        self.assertFalse(forbidden)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["agent_count"], 100)
        self.assertEqual(selected[0]["gate_eligible_task_seeds"], [233, 277])


if __name__ == "__main__":
    unittest.main()
