from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.repair_collection import _read_json, _read_jsonl
from scripts.materialize_stride_structpool_revised_six_map import materialize


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_structpool_revised_six_map_materialization.json"


class StructPoolRevisedSixMapTest(unittest.TestCase):
    def test_registration_reuses_exact_tasks_without_generation(self) -> None:
        config = _read_json(CONFIG)
        tasks = [
            str(task_id)
            for source in config["sources"]
            for task_id in source["task_ids"]
        ]
        self.assertEqual(len(tasks), 12)
        self.assertEqual(len(set(tasks)), 12)
        self.assertEqual(len(config["expected_maps"]), 6)
        self.assertFalse(config["task_generation_changed"])
        self.assertFalse(config["ttf_outcomes_read"])

    def test_materialization_has_two_tasks_per_registered_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            report = materialize(CONFIG, output)
            rows = _read_jsonl(output / "balanced_wall_clock" / "manifest.jsonl")
            counts: dict[str, int] = {}
            for row in rows:
                map_id = str(row["map_id"])
                counts[map_id] = counts.get(map_id, 0) + 1
            self.assertEqual(report["map_count"], 6)
            self.assertEqual(report["task_count"], 12)
            self.assertEqual(set(counts.values()), {2})
            self.assertTrue((output / "artifact_registry.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
