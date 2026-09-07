from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments._common import read_json, sha256_file, write_json, write_jsonl
from generators.models import TaskData
from lns2_selector.evaluation.path_quality_pressure import case_plan, prepare_pressure, verify_outputs


class PressurePilotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = {
            "schema": "lns2.path_quality_pressure_pilot.v1",
            "role": "reused_map_design_pilot_not_confirmation",
            "allow_solver_execution": False, "master_seed": 2026090801,
            "expected_maps": 1, "densities": [0.15, 0.2, 0.25],
            "modes": [{"name": "balanced", "required_bottleneck_crossing_ratio": 0.0},
                      {"name": "eligible", "required_bottleneck_crossing_ratio": 0.6}],
            "solver_seeds_for_later_reset": [61, 62], "task": {}, "output": "build/pilot",
        }
        self.sources = [{"map_id": "sample", "task_seed": 123,
                         "map_file": "maps/sample.map", "map_metadata_file": "maps/sample.json"}]

    def test_full_factorial_independent_seeds(self):
        rows = case_plan(self.config, self.sources)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len({r["task_seed"] for r in rows}), 6)
        self.assertNotIn(123, {r["task_seed"] for r in rows})
        self.assertEqual(rows, case_plan(self.config, self.sources))

    def test_source_order_does_not_change_cases(self):
        config = {**self.config, "expected_maps": 2}
        sources = self.sources + [{**self.sources[0], "map_id": "other", "task_seed": 456}]
        self.assertEqual(case_plan(config, sources), case_plan(config, sources[::-1]))

    def test_invalid_design_rejected(self):
        for field, value in (("allow_solver_execution", True), ("densities", [0.2, 0.2]),
                             ("densities", [1.0]), ("expected_maps", 2)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                case_plan({**self.config, field: value}, self.sources)

    def fixture(self):
        for relative in ("lns2_selector/evaluation/path_quality_pressure.py",
                         "lns2_selector/evaluation/path_quality_preflight.py", "experiments/_common.py"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n", encoding="utf-8")
        maps = self.root / "source/maps"
        maps.mkdir(parents=True)
        (maps / "sample.map").write_text("type octile\nheight 2\nwidth 3\nmap\n...\n...\n", encoding="utf-8")
        write_json(maps / "sample.json", {"map_id": "sample", "seed": 1, "grid": ["...", "..."], "metadata": {}})
        manifest = self.root / "source/manifest.jsonl"
        write_jsonl(manifest, self.sources)
        config = copy.deepcopy(self.config)
        config["densities"] = [0.2]
        config["modes"] = config["modes"][:1]
        config["source_manifest"] = {"manifest": "source/manifest.jsonl", "sha256": sha256_file(manifest)}
        path = self.root / "config.json"
        write_json(path, config)
        return path

    @staticmethod
    def generated(map_data, config, seed, task_id):
        return TaskData(task_id, map_data.map_id, seed, [(0, 0)], [(0, 2)],
                        {"required_bottlenecks": [None], "actual_shortest_distances": [2]})

    def test_static_only_resume_and_verify(self):
        config = self.fixture()
        with patch("lns2_selector.evaluation.path_quality_pressure.generate_tasks", side_effect=self.generated) as generate:
            first = prepare_pressure(self.root, config)
            second = prepare_pressure(self.root, config)
            self.assertEqual(generate.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first, prepare_pressure(self.root, config, verify=True))
        self.assertEqual(first["solver_calls"], 0)
        self.assertFalse(first["timing_authorized"])
        self.assertFalse(first["joint_feasibility_proven"])

    def test_failed_generation_retained_not_resampled(self):
        config = self.fixture()
        with patch("lns2_selector.evaluation.path_quality_pressure.generate_tasks", side_effect=ValueError("capacity")) as generate:
            first = prepare_pressure(self.root, config)
            second = prepare_pressure(self.root, config)
            self.assertEqual(generate.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "blocked_generation_errors")
        self.assertEqual(first["task_count"], 1)

    def test_changed_input_rejected(self):
        config = self.fixture()
        with patch("lns2_selector.evaluation.path_quality_pressure.generate_tasks", side_effect=self.generated):
            prepare_pressure(self.root, config)
        data = read_json(config)
        data["master_seed"] += 1
        write_json(config, data)
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            prepare_pressure(self.root, config)

    def test_tampered_output_rejected(self):
        config = self.fixture()
        with patch("lns2_selector.evaluation.path_quality_pressure.generate_tasks", side_effect=self.generated):
            report = prepare_pressure(self.root, config)
        task = self.root / report["cases"][0]["files"]["task_file"]
        task.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            verify_outputs(self.root, report)


if __name__ == "__main__":
    unittest.main()
