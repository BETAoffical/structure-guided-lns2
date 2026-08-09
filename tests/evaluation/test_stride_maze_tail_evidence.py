from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.stride_maze_tail_evidence import (
    load_maze_tail_evidence_config,
    select_maze_tail_tasks,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_maze_tail_evidence_preflight_v1.json"


class MazeTailEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _path, _root, cls.config, _inputs, cls.source = (
            load_maze_tail_evidence_config(CONFIG)
        )

    def _synthetic_rows(self) -> tuple[list[dict], list[dict]]:
        manifest = []
        qualification = []
        for benchmark in self.source["benchmarks"]:
            map_id = str(benchmark["id"])
            counts = list(map(int, benchmark["agent_counts"]))
            for task_seed in self.source["task_seeds"]:
                for variant in self.source["task_variants"]:
                    for load_index, agent_count in enumerate(counts):
                        task_id = (
                            f"{map_id}__derived_{variant}"
                            f"__task_seed_{task_seed:04d}__agents_{agent_count:04d}"
                        )
                        manifest.append(
                            {
                                "split": "balanced_wall_clock",
                                "map_id": map_id,
                                "task_id": task_id,
                                "task_variant_family": variant,
                                "scenario_type": f"movingai_map_derived_{variant}",
                                "task_seed": task_seed,
                                "agent_count": agent_count,
                            }
                        )
                        conflict = 0
                        if task_seed == 811 and load_index == 0:
                            conflict = 50
                        elif task_seed == 811 and load_index == 2:
                            conflict = 190
                        elif task_seed == 853 and load_index == 1:
                            conflict = 52
                        elif task_seed == 853 and load_index == 3:
                            conflict = 200
                        for solver_seed in self.config["solver_seeds"]:
                            qualification.append(
                                {
                                    "map_id": map_id,
                                    "task_id": task_id,
                                    "agent_count": agent_count,
                                    "solver_seed": solver_seed,
                                    "status": "ok",
                                    "initial_complete": True,
                                    "initial_state_consistent": True,
                                    "initial_conflicts": conflict,
                                    "initial_complexity": {},
                                    "state_fingerprint": (
                                        f"fp-{map_id}-{task_seed}-{variant}"
                                        f"-{agent_count}-{solver_seed}"
                                    ),
                                }
                            )
        return manifest, qualification

    def test_registered_config_dimensions(self) -> None:
        self.assertEqual(self.config["expected_task_count"], 64)
        self.assertEqual(self.config["expected_reset_count"], 192)
        self.assertEqual(self.source["task_seeds"], [811, 853])
        self.assertEqual(self.source["solver_seeds"], [17, 29, 43])

    def test_selection_is_reset_only_and_seed_diverse(self) -> None:
        manifest, qualification = self._synthetic_rows()
        summaries, selected, errors, forbidden = select_maze_tail_tasks(
            manifest, qualification, self.config
        )
        self.assertEqual(len(summaries), 64)
        self.assertEqual(len(selected), 16)
        self.assertFalse(errors)
        self.assertFalse(forbidden)
        self.assertEqual({row["map_id"] for row in selected}, {
            "maze-32-32-4",
            "maze-128-128-1",
            "maze-128-128-2",
            "maze-128-128-10",
        })
        self.assertEqual({row["task_seed"] for row in selected}, {811, 853})
        self.assertEqual(
            {row["conflict_band"] for row in selected}, {"moderate", "high"}
        )

    def test_forbidden_outcome_field_is_detected(self) -> None:
        manifest, qualification = self._synthetic_rows()
        changed = copy.deepcopy(qualification)
        changed[0]["repair_iterations"] = 7
        _summaries, _selected, _errors, forbidden = select_maze_tail_tasks(
            manifest, changed, self.config
        )
        self.assertEqual(forbidden, {"repair_iterations"})


if __name__ == "__main__":
    unittest.main()
