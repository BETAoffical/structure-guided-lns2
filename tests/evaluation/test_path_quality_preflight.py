from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from experiments._common import read_json, sha256_file, write_json
from lns2_selector.evaluation.path_quality_preflight import (
    audit_paths, audit_task, checked_input, contained, prepare, publish, read_grid,
)


class StaticPreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.map = self.root / "sample.map"
        self.map.write_text("type octile\nheight 2\nwidth 3\nmap\n...\n...\n", encoding="utf-8")
        self.scenario = self.root / "sample.scen"
        self.scenario.write_text("version 1\n0 sample.map 3 2 0 0 2 0 2\n", encoding="utf-8")
        self.task = self.root / "task.json"
        write_json(self.task, {"starts": [[0, 0]], "goals": [[0, 2]]})

    def test_valid_task(self):
        result = audit_task(self.map, self.scenario, self.task, 1)
        self.assertEqual(result["sum_shortest_distances"], 2)
        self.assertFalse(result["native_reset_performed"])
        self.assertFalse(result["joint_feasibility_proven"])

    def test_movingai_metadata_sidecar(self):
        write_json(self.task, {"agent_count": 1, "benchmark_id": "sample", "scenario_sha256": sha256_file(self.scenario)})
        self.assertEqual(audit_task(self.map, self.scenario, self.task, 1)["agent_count"], 1)
        data = read_json(self.task)
        data["scenario_sha256"] = "0" * 64
        write_json(self.task, data)
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            audit_task(self.map, self.scenario, self.task, 1)

    def test_scenario_prefix_mismatch(self):
        write_json(self.task, {"starts": [[1, 0]], "goals": [[0, 2]]})
        with self.assertRaisesRegex(ValueError, "prefix"):
            audit_task(self.map, self.scenario, self.task, 1)

    def test_scenario_distance_is_not_four_neighbor_ground_truth(self):
        self.scenario.write_text("version 1\n0 sample.map 3 2 0 0 2 0 1.5\n", encoding="utf-8")
        result = audit_task(self.map, self.scenario, self.task, 1)
        self.assertEqual(result["sum_shortest_distances"], 2)
        self.assertEqual(result["scenario_distance_difference_count"], 1)

    def test_unreachable(self):
        self.map.write_text("type octile\nheight 2\nwidth 3\nmap\n.@.\n.@.\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unreachable"):
            audit_task(self.map, self.scenario, self.task, 1)

    def test_bad_grid(self):
        self.map.write_text("type octile\nheight 2\nwidth 3\nmap\n..\n...\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dimensions"):
            read_grid(self.map)

    def test_duplicate_endpoints(self):
        write_json(self.task, {"starts": [[0, 0], [0, 0]], "goals": [[0, 1], [0, 2]]})
        self.scenario.write_text("version 1\n0 sample.map 3 2 0 0 1 0 1\n0 sample.map 3 2 0 0 2 0 2\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            audit_task(self.map, self.scenario, self.task, 2)

    def test_path_escape_and_absolute(self):
        for path in ("../outside", "C:/outside", "/outside"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                contained(self.root, path)

    def test_input_hash_change(self):
        spec = {"manifest": "task.json", "sha256": sha256_file(self.task)}
        checked_input(self.root, spec)
        write_json(self.task, {})
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            checked_input(self.root, spec)

    def test_cannot_authorize_solver(self):
        write_json(self.task, {"schema": "lns2.path_quality_preflight.v1", "allow_solver_execution": True})
        with self.assertRaisesRegex(ValueError, "authorize"):
            prepare(self.root, self.task)

    def test_publish_verify_and_refuse_overwrite(self):
        report = {"status": "not_started", "cases": [], "blocking_items": [], "fingerprint": "x"}
        publish(self.root, report, "build/prep")
        before = sha256_file(self.root / "build/prep/preflight_report.json")
        publish(self.root, report, "build/prep", verify=True)
        self.assertEqual(before, sha256_file(self.root / "build/prep/preflight_report.json"))
        with self.assertRaises(ValueError):
            publish(self.root, {**report, "fingerprint": "changed"}, "build/prep")
        with self.assertRaises(ValueError):
            publish(self.root, report, "outside-build")


class PathQualityTests(unittest.TestCase):
    def test_soc_makespan_waits_and_padding(self):
        paths = [[0, 0, 1, 2, 2], [3, 4, 5]]
        original = copy.deepcopy(paths)
        result = audit_paths(["...", "..."], [[0, 0], [1, 0]], [[0, 2], [1, 2]], paths)
        self.assertTrue(result["feasible"])
        self.assertEqual((result["soc_steps"], result["makespan_steps"], result["wait_steps"]), (5, 3, 1))
        self.assertEqual(paths, original)

    def test_edge_swap(self):
        result = audit_paths([".."], [[0, 0], [0, 1]], [[0, 1], [0, 0]], [[0, 1], [1, 0]])
        self.assertEqual(result["colliding_pairs"], 1)

    def test_stay_at_goal_conflict(self):
        result = audit_paths(["..."], [[0, 0], [0, 2]], [[0, 1], [0, 0]], [[0, 1], [2, 2, 1, 0]])
        self.assertFalse(result["feasible"])

    def test_illegal_row_wrap(self):
        with self.assertRaisesRegex(ValueError, "illegal move"):
            audit_paths(["..", ".."], [[0, 1]], [[1, 0]], [[1, 2]])

    def test_empty_path(self):
        with self.assertRaises(ValueError):
            audit_paths([".."], [[0, 0]], [[0, 1]], [[]])

    def test_zero_length_task(self):
        result = audit_paths(["."], [[0, 0]], [[0, 0]], [[0]])
        self.assertEqual(result["makespan_steps"], 0)
        self.assertTrue(result["feasible"])


if __name__ == "__main__":
    unittest.main()
