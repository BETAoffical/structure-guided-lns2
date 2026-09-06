from __future__ import annotations

import copy
import tempfile
import time
import unittest
from pathlib import Path

from experiments._common import read_json, write_json
from lns2_selector.evaluation.path_quality_execution import (
    CONTROLLERS, PathJournal, controller_job, execution_schedule, read_artifact,
    spec_fingerprint, supervise_episode,
)


def _fake_child(spec):
    output = Path(spec["output"])
    journal = PathJournal(output, spec["binding"], output / "tiny.map", output / "tiny.scen")
    state = {"agents": [{"id": 0, "path": [0, 1]}], "feasible": True,
             "num_of_colliding_pairs": 0, "sum_of_costs": 1}
    journal("initial", state, 1.0, 1.001, "state")
    if spec.get("hang"):
        time.sleep(30)
    if spec.get("crash"):
        raise RuntimeError("intentional child failure")
    journal.save("result", {"status": "completed", "success_by_deadline": True})


def _cases(count=14):
    return [{"task_id": f"t{i:02}", "map_id": f"m{i:02}", "family": "warehouse",
             "status": "static_ready_runtime_unverified", "solver_seeds": [51, 52],
             "files": {"map_file": "tiny.map", "scenario_file": "tiny.scen"},
             "static_audit": {"agent_count": 1}} for i in range(count)]


PROTOCOL = {"first_feasible_budget_seconds": 120, "total_planning_budgets_seconds": [60, 120]}


class SchedulingTests(unittest.TestCase):
    def test_protocols_and_no_authorization(self):
        cases = _cases()
        cases.append({**cases[0], "task_id": "quarantine", "status": "quarantined"})
        rows = execution_schedule(cases, PROTOCOL)
        self.assertEqual(len(rows), 252)
        self.assertEqual(len({r["job_id"] for r in rows}), 252)
        self.assertTrue(all(not r["execution_authorized"] and r["repair_iteration_cap"] is None for r in rows))
        self.assertNotIn("quarantine", {r["task_id"] for r in rows})
        self.assertEqual(rows, execution_schedule(list(reversed(cases)), PROTOCOL))
        for i in range(0, len(rows), 3):
            self.assertEqual({r["controller"] for r in rows[i:i+3]}, set(CONTROLLERS))
            self.assertEqual(len({r["stage2_seed"] for r in rows[i:i+3]}), 1)
        self.assertEqual(len({tuple(r["controller"] for r in rows[i:i+3]) for i in range(0, 18, 3)}), 6)

    def test_duplicate_and_bad_budget_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            execution_schedule(_cases(1) * 2, PROTOCOL)
        with self.assertRaises(ValueError):
            execution_schedule(_cases(1), {**PROTOCOL, "first_feasible_budget_seconds": float("nan")})

    def test_controller_mapping_only_dual16_augments(self):
        template = {"proposal": {"heuristics": ["target", "collision", "random"]},
                    "environment": {"replan_algorithm": "PP", "use_sipp": True},
                    "frozen_models": "frozen", "model_registration": {}}
        before = copy.deepcopy(template)
        case = _cases(1)[0]
        jobs = {item["controller"]: controller_job(Path("."), case, item, template, Path("out"), "fp")
                for item in execution_schedule([case], PROTOCOL)[:3]}
        self.assertEqual(template, before)
        self.assertEqual(jobs["official_adaptive"]["policy"], "official_adaptive")
        self.assertEqual(jobs["v2-full"]["controller"], jobs["dual16"]["controller"])
        self.assertNotIn("hybridstructpool", jobs["v2-full"]["proposal"])
        self.assertIn("hybridstructpool", jobs["dual16"]["proposal"])
        self.assertEqual(jobs["v2-full"]["proposal_state_verification"], "sampled")
        self.assertEqual(jobs["dual16"]["proposal_state_verification"], "sampled")
        self.assertTrue(all(j["max_decisions"] == 0 and j["safety_max_decisions"] is None for j in jobs.values()))


class JournalAndProcessTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.map = self.root / "tiny.map"
        self.scen = self.root / "tiny.scen"
        self.map.write_text("type octile\nheight 1\nwidth 2\nmap\n..\n", encoding="utf-8")
        self.scen.write_text("version 1\n0\ttiny.map\t2\t1\t0\t0\t1\t0\t1\n", encoding="utf-8")
        self.state = {"agents": [{"id": 0, "path": [0, 1]}], "feasible": True,
                      "num_of_colliding_pairs": 0, "sum_of_costs": 1}

    def spec(self, **changes):
        spec = {"output": str(self.root), "native_sha256": "test-only",
                "item": {"budget_seconds": 0.1}, "process_timeout_seconds": 2.0, **changes}
        spec["binding"] = spec_fingerprint(spec)
        return spec

    def test_journal_preserves_first_and_checks_sha(self):
        journal = PathJournal(self.root, "binding", self.map, self.scen)
        before = copy.deepcopy(self.state)
        journal("initial", self.state, 1, 1.1, "initial")
        journal("terminal", self.state, 1, 1.2, "initial")
        first = read_artifact(self.root / "first_feasible.json", "binding")
        self.assertAlmostEqual(first["available_elapsed_seconds"], 0.1)
        self.assertEqual(self.state, before)
        payload = read_json(self.root / "first_feasible.json")
        payload["payload"]["observation"]["agents"][0]["path"] = [0]
        write_json(self.root / "first_feasible.json", payload)
        with self.assertRaises(ValueError):
            read_artifact(self.root / "first_feasible.json", "binding")

    def test_bad_observation_rejected(self):
        self.state["feasible"] = False
        with self.assertRaises(ValueError):
            PathJournal(self.root, "x", self.map, self.scen)("initial", self.state, 1, 2, "f")

    def test_execution_gate(self):
        with self.assertRaises(PermissionError):
            supervise_episode(self.spec(), child_entry=_fake_child)
        self.assertFalse((self.root / "binding.json").exists())

    def test_success_resume_and_changed_spec(self):
        spec = self.spec()
        result = supervise_episode(spec, authorized=True, child_entry=_fake_child)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result, supervise_episode(spec, authorized=True, resume=True, child_entry=_fake_child))
        bad = {**spec, "native_sha256": "changed"}
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            supervise_episode(bad, authorized=True, resume=True, child_entry=_fake_child)
        self.assertFalse((self.root / "run.lock").exists())

    def test_timeout_keeps_first_but_not_normal_success(self):
        result = supervise_episode(self.spec(hang=True), authorized=True, child_entry=_fake_child)
        self.assertEqual(result["status"], "external_timeout")
        self.assertTrue(result["first_feasible_preserved"])
        self.assertTrue(result["first_feasible_within_budget"])
        self.assertFalse(result["success_by_deadline"])
        self.assertFalse((self.root / "run.lock").exists())

    def test_child_crash_is_not_no_solution(self):
        result = supervise_episode(self.spec(crash=True), authorized=True, child_entry=_fake_child)
        self.assertEqual(result["status"], "worker_exit_error")
        self.assertTrue(result["error"])

    def test_existing_lock_rejected(self):
        (self.root / "run.lock").write_text("existing", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "locked"):
            supervise_episode(self.spec(), authorized=True, child_entry=_fake_child)


if __name__ == "__main__":
    unittest.main()
