from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import STATE_FINGERPRINT_KEYS, state_fingerprint
from lns2_selector.evaluation.path_quality_analysis import METRICS, inspect_episode, publish_analysis, summarize
from lns2_selector.evaluation.path_quality_cohort import admission_key, collect
from lns2_selector.evaluation.path_quality_execution import PathJournal
from scripts.run_path_quality_evaluation import main


class CohortContractTests(unittest.TestCase):
    def test_collect_gate_precedes_io(self):
        with patch("lns2_selector.evaluation.path_quality_cohort.verified_registration") as lookup:
            with self.assertRaises(PermissionError):
                collect(Path("missing"), Path("missing"))
            lookup.assert_not_called()

    def test_cli_collect_is_not_implicitly_authorized(self):
        with self.assertRaises(PermissionError):
            main(["collect"])

    def test_admission_shared_across_methods_but_not_budgets(self):
        item = {"task_id": "t", "solver_seed": 51, "budget_seconds": 60, "controller": "v2-full"}
        self.assertEqual(admission_key(item), admission_key({**item, "controller": "dual16"}))
        self.assertNotEqual(admission_key(item), admission_key({**item, "budget_seconds": 120}))

    def test_zero_baseline_bootstrap_and_paired_failure(self):
        rows = []
        for controller in ("official_adaptive", "v2-full", "dual16"):
            for seed in (1, 2):
                success = not (controller == "dual16" and seed == 2)
                rows.append({"task_id": "t", "map_id": "m", "family": "warehouse", "solver_seed": seed,
                    "controller": controller, "protocol": "first_feasible", "budget_seconds": 120,
                    "status": "completed" if success else "external_timeout", "success": success,
                    "found_within_budget": success, "delivered_within_budget": success,
                    **{m: (0.0 if controller == "official_adaptive" and m == "waits" else 1.0) if success else None for m in METRICS}})
        report = summarize(rows, samples=20)
        pair = next(c for c in report["comparisons"] if c["group"].endswith("/all") and c["challenger"] == "dual16")
        self.assertEqual(pair["scheduled_pairs"], 2)
        self.assertEqual(pair["common_success_pairs"], 1)
        self.assertIsNone(pair["metrics"]["waits"]["map_bootstrap"]["improvement_95_ci"])
        self.assertEqual(pair["metrics"]["waits"]["map_bootstrap"]["paired_delta_95_ci"], [1.0, 1.0])
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            publish_analysis(output, rows, {"statistics": report})
            digests = {p.name:sha256_file(p) for p in output.iterdir()}
            publish_analysis(output, rows, {"statistics": report})
            self.assertEqual(digests, {p.name:sha256_file(p) for p in output.iterdir()})

    def test_missing_paired_row_rejected(self):
        row = {"task_id": "t", "map_id": "m", "family": "warehouse", "solver_seed": 1,
               "controller": "v2-full", "protocol": "first_feasible", "budget_seconds": 120,
               "status": "pending", "success": False, "found_within_budget": False, "delivered_within_budget": False}
        with self.assertRaisesRegex(ValueError, "paired schedule"):
            summarize([row], samples=10)


class ArtifactInspectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.map, self.scen = self.root / "tiny.map", self.root / "tiny.scen"
        self.map.write_text("type octile\nheight 1\nwidth 2\nmap\n..\n", encoding="utf-8")
        self.scen.write_text("version 1\n0\ttiny.map\t2\t1\t0\t0\t1\t0\t1\n", encoding="utf-8")
        self.case = {"files": {"map_file": "tiny.map", "scenario_file": "tiny.scen"}, "static_audit": {"agent_count": 1}}
        self.item = {"controller": "official_adaptive", "task_id": "t", "map_id": "m", "family": "warehouse",
                     "protocol": "first_feasible", "budget_seconds": 1, "solver_seed": 51, "stage2_seed": 5}
        state = {key: None for key in STATE_FINGERPRINT_KEYS}
        state.update(agents=[{"id": 0, "path": [0, 1]}], feasible=True, num_of_colliding_pairs=0, sum_of_costs=1)
        self.state, self.fp = state, state_fingerprint(state)
        self.journal = PathJournal(self.root, "binding", self.map, self.scen, self.fp)

    def populate(self, ready=0.2):
        j = self.journal
        write_json(self.root / "binding.json", {"binding": "binding", "item": self.item})
        j("initial", self.state, 1, 1.1, self.fp)
        j("terminal", self.state, 1, 1.1, self.fp)
        first_at = read_json(self.root / "first_feasible.json")["payload"]["available_elapsed_seconds"]
        j.save("first_phase_result", {"status": "ok", "summary": {"repair_iterations": 0}})
        j.save("final_paths", {"paths": [[0, 1]], "final_quality": j.quality(self.state)})
        j.save("result", {"status": "completed", "paths_saved_elapsed_seconds": ready,
            "dispatch_wall_seconds": ready, "first_feasible_elapsed_seconds": first_at,
            "delivered_within_budget": ready <= 1, "success_by_deadline": True,
            "cold_start_to_paths_seconds": ready+0.1, "startup_before_reset_seconds": 0.1,
            "environment_construct_seconds": 0.01})
        j.save("supervisor", {"status": "completed", "success_by_deadline": True, "first_feasible_within_budget": True})

    def inspect(self):
        return inspect_episode(self.root, self.case, self.item, self.root, "binding", {"state_fingerprint": self.fp})

    def test_success_is_recomputed_from_paths(self):
        self.populate()
        result = self.inspect()
        self.assertTrue(result["success"])
        self.assertEqual(result["makespan"], 1)
        self.assertEqual(result["completion_1.0s"], 1.2)

    def test_late_delivery_distinct_from_found_within_budget(self):
        self.populate(ready=1.2)
        result = self.inspect()
        self.assertTrue(result["found_within_budget"])
        self.assertTrue(result["flow_completed"])
        self.assertFalse(result["delivered_within_budget"])

    def test_changed_initial_anchor_rejected_before_save(self):
        with self.assertRaisesRegex(ValueError, "reset admission"):
            self.journal("initial", self.state, 1, 1.1, "wrong")
        self.assertFalse((self.root / "initial.json").exists())

    def test_tampered_paths_rejected(self):
        self.populate()
        payload = read_json(self.root / "final_paths.json")
        payload["payload"]["paths"] = [[0]]
        write_json(self.root / "final_paths.json", payload)
        with self.assertRaisesRegex(ValueError, "SHA"):
            self.inspect()

    def test_missing_artifact_rejected(self):
        self.populate()
        (self.root / "first_feasible.json").unlink()
        with self.assertRaises(ValueError):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
