from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

from experiments._common import read_json
from lns2_selector.evaluation.path_quality_pressure import case_plan


ROOT = Path(__file__).resolve().parents[2]


class PressureTaskReplicationTests(unittest.TestCase):
    def test_only_generation_seed_and_output_change(self):
        original = read_json(ROOT / "configs/path_quality_pressure_pilot_v2.json")
        replica = read_json(ROOT / "configs/path_quality_pressure_tasks_replica2.json")
        changed = {key for key in original.keys() | replica.keys()
                   if original.get(key) != replica.get(key)}
        self.assertEqual(changed, {"master_seed", "output"})
        self.assertFalse(replica["allow_solver_execution"])
        self.assertEqual(replica["solver_seeds_for_later_reset"], [61, 62])

    def test_replicated_factorial_has_disjoint_deterministic_task_seeds(self):
        sources = [{"map_id": f"very_high_confirm_v2_station_centric_{i:04d}",
                    "task_seed": i} for i in range(8)]
        configs = [read_json(ROOT / relative) for relative in (
            "configs/path_quality_pressure_pilot_v2.json",
            "configs/path_quality_pressure_tasks_replica2.json")]
        batches = [case_plan(config, sources) for config in configs]
        self.assertEqual([len(rows) for rows in batches], [48, 48])
        self.assertTrue({r["task_seed"] for r in batches[0]}.isdisjoint(
            {r["task_seed"] for r in batches[1]}))
        self.assertTrue({r["task_id"] for r in batches[0]}.isdisjoint(
            {r["task_id"] for r in batches[1]}))
        cells = Counter((r["map_id"], r["density"], r["mode"]["name"])
                        for rows in batches for r in rows)
        self.assertEqual(len(cells), 48)
        self.assertEqual(set(cells.values()), {2})
        self.assertEqual(batches[1], case_plan(configs[1], list(reversed(sources))))


if __name__ == "__main__":
    unittest.main()
