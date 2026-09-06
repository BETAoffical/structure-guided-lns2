"""Tiny functional fixtures, not the prepared warehouse/Room/Maze benchmark."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

try:
    import lns2_env
except ImportError:
    lns2_env = None

from lns2_selector.evaluation.anytime_handoff import continue_with_official_anytime
from lns2_selector.solver.native import native_identity


@unittest.skipUnless(lns2_env is not None and hasattr(lns2_env, "optimize_feasible_paths"), "new native handoff module required")
class AnytimeNativeTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.map = Path(folder.name) / "tiny.map"
        self.scen = Path(folder.name) / "tiny.scen"
        self.map.write_text("type octile\nheight 3\nwidth 4\nmap\n....\n....\n....\n", encoding="utf-8")
        self.scen.write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t0\t2\t3\t2\t3\n", encoding="utf-8")
        self.paths = [[0, 0, 1, 2, 3], [8, 8, 9, 10, 11]]

    def optimize(self, paths=None, budget=10, seed=17, strategy="Adaptive"):
        return lns2_env.optimize_feasible_paths(str(self.map), str(self.scen), 2,
            self.paths if paths is None else paths, seed, budget, max_iterations=2, strategy=strategy, neighborhood_size=8)

    def test_zero_budget_preserves_exact_paths(self):
        result = self.optimize(budget=0)
        self.assertEqual(result["paths"], self.paths)
        self.assertEqual(result["initial_paths"], self.paths)
        self.assertEqual(result["iterations"], 0)
        self.assertFalse(result["initial_planner_called"])
        self.assertEqual(result["soc"], 8)

    def test_official_loop_reduces_cost(self):
        before = copy.deepcopy(self.paths)
        result = self.optimize()
        self.assertEqual(result["initial_paths"], before)
        self.assertEqual(self.paths, before)
        self.assertEqual(result["soc"], 6)
        self.assertEqual(result["iterations"], 2)
        self.assertEqual(result["cost_history"], sorted(result["cost_history"], reverse=True))

    def test_independent_seed_repeats(self):
        one = self.optimize(seed=5)
        self.optimize(seed=123)
        two = self.optimize(seed=5)
        self.assertEqual(one["paths"], two["paths"])
        self.assertEqual(one["cost_history"], two["cost_history"])

    def test_reject_malformed_paths_and_parameters(self):
        for paths in ([[], self.paths[1]], [[0, 3], self.paths[1]], [[0, 1, 2, 3, 3], self.paths[1]]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                self.optimize(paths)
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.optimize(budget=value)

    def test_reject_colliding_paths(self):
        self.scen.write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "edge conflict"):
            self.optimize([[0, 1, 2, 3], [3, 2, 1, 0]])
        with self.assertRaisesRegex(ValueError, "vertex"):
            self.optimize([[0, 1, 2, 3], [3, 3, 2, 1, 0]])

    def test_live_initlns_rng_cannot_silently_continue(self):
        self.scen.write_text("version 1\n0\ttiny.map\t4\t3\t0\t0\t3\t0\t3\n0\ttiny.map\t4\t3\t3\t0\t0\t0\t3\n", encoding="utf-8")
        env = lns2_env.LNS2RepairEnv(str(self.map), str(self.scen), 2)
        env.reset_paths([[0, 1, 2, 3], [3, 2, 1, 0]], seed=1)
        self.optimize([[0, 1, 2, 3], [3, 7, 6, 5, 4, 0]])
        with self.assertRaisesRegex(ValueError, "random stream"):
            env.step({"mode": "official"})

    def test_real_initlns_observation_to_anytime(self):
        import time
        started = time.perf_counter()
        env = lns2_env.LNS2RepairEnv(str(self.map), str(self.scen), 2)
        state = env.reset(7)
        self.assertTrue(state["feasible"])
        result = continue_with_official_anytime(lns2_env, map_path=self.map, scenario_path=self.scen,
            observation=state, expected_native_sha256=native_identity(lns2_env)["sha256"],
            planning_started=started, total_budget_seconds=10, stage2_seed=11, max_iterations=2)
        self.assertEqual(result["initial_paths"], [a["path"] for a in state["agents"]])
        self.assertLessEqual(result["final_quality"]["soc_steps"], result["initial_quality"]["soc_steps"])

    def test_three_controller_workers_save_same_tiny_initial_paths(self):
        from experiments._common import read_json
        from experiments.closed_loop_confirmation import _closed_loop_episode_worker
        from lns2_selector.evaluation.path_quality_execution import PathJournal, controller_job, read_artifact
        root = Path(__file__).resolve().parents[2]
        template = read_json(root / "configs/stride_warehouse_fixed16_development_runtime_v2.json")
        # Only a two-agent, conflict-free fixture. No cohort is admitted here.
        case = {"status": "static_ready_runtime_unverified", "task_id": "tiny", "map_id": "tiny",
                "family": "warehouse", "static_audit": {"agent_count": 2},
                "files": {"map_file": str(self.map), "scenario_file": str(self.scen)}}
        initial_hashes = []
        for controller in ("official_adaptive", "v2-full", "dual16"):
            with self.subTest(controller=controller):
                output = self.map.parent / controller
                item = {"task_id": "tiny", "controller": controller, "budget_seconds": 2.0, "solver_seed": 7}
                job = controller_job(root, case, item, template, output, "tiny-test")
                job["max_decisions"] = 2
                journal = PathJournal(output, "tiny-test", self.map, self.scen)
                result = _closed_loop_episode_worker(job, path_observer=journal)
                self.assertEqual(result["status"], "ok", result.get("error"))
                self.assertEqual(result["summary"]["repair_iterations"], 0)
                initial = read_artifact(output / "initial.json", "tiny-test")
                first = read_artifact(output / "first_feasible.json", "tiny-test")
                self.assertEqual(initial["observation"], first["observation"])
                initial_hashes.append(initial["state_fingerprint"])
        self.assertEqual(len(set(initial_hashes)), 1)


if __name__ == "__main__":
    unittest.main()
