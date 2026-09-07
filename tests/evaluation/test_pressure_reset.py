import copy
from contextlib import ExitStack, nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments._common import read_json, write_json
from lns2_selector.evaluation.pressure_reset import pressure_metrics, run_resets


class ResetMetricsTests(unittest.TestCase):
    def fixture(self):
        task = {"starts": [[0, 0], [0, 2]], "goals": [[0, 2], [0, 0]],
                "metadata": {"required_bottlenecks": [[0, 1], None]}}
        state = {"agents": [{"id": 0, "path": [0, 1, 2]}, {"id": 1, "path": [2, 1, 0]}],
                 "iteration": 0, "initial_solution_complete": True,
                 "conflict_edges": [[0, 1]], "num_of_colliding_pairs": 1}
        return state, task, ["..."]

    def test_conflicts_components_and_actual_bottleneck_visit(self):
        result = pressure_metrics(*self.fixture())
        self.assertEqual(result["conflict_component_sizes"], [2])
        self.assertEqual(result["conflict_density"], 1)
        self.assertEqual(result["designated_bottleneck_visit_fraction"], 1)

    def test_invalid_reset_rejected(self):
        for field, value in (("initial_solution_complete", False), ("iteration", 1),
                             ("num_of_colliding_pairs", 2), ("conflict_edges", [[0, 9]])):
            state, task, grid = self.fixture()
            state[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                pressure_metrics(state, task, grid)

    def test_empty_path_and_agent_identity(self):
        state, task, grid = self.fixture()
        broken = copy.deepcopy(state)
        broken["agents"][0]["path"] = []
        with self.assertRaises(ValueError):
            pressure_metrics(broken, task, grid)
        state["agents"][0]["id"] = 1
        with self.assertRaises(ValueError):
            pressure_metrics(state, task, grid)

    def test_zero_conflict_retained(self):
        state = {"agents": [{"id": 0, "path": [0, 1, 2]}], "iteration": 0,
                 "initial_solution_complete": True, "conflict_edges": [], "num_of_colliding_pairs": 0}
        task = {"starts": [[0, 0]], "goals": [[0, 2]], "metadata": {"required_bottlenecks": [None]}}
        result = pressure_metrics(state, task, ["..."])
        self.assertTrue(result["feasible"])
        self.assertEqual(result["largest_conflict_component"], 0)
        self.assertIsNone(result["designated_bottleneck_visit_fraction"])


class ResetOrchestrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.output = self.root / "build/resets"
        self.output.mkdir(parents=True)
        for relative in ("experiments/repair_collection.py", "lns2_selector/solver/native.py"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture", encoding="utf-8")
        self.config = {"schema": "lns2.pressure_reset_only.v1", "workers": 1,
                       "repairs_allowed": False, "formal_timing_allowed": False,
                       "preparation_config": "prepare.json", "native_file": "build/fake.so",
                       "native_sha256": "native", "output": "build/resets", "reset_process_timeout_seconds": 0.01}
        write_json(self.root / "config.json", self.config)
        write_json(self.root / "prepare.json", {"solver_seeds_for_later_reset": [61], "output": "build/prepared"})
        write_json(self.root / "build/prepared/preparation_report.json", {"fixture": True})
        self.prepared = {"generation_errors": 0, "status": "static_ready_reset_pending",
                         "fingerprint": "prepared", "cases": [{"case": {"task_id": "task"}}]}
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in (("prepare_pressure", self.prepared),
                            ("native_identity", {"sha256": "native", "path": str(self.root / "build/fake.so")}),
                            ("state_fingerprint", "state")):
            stack.enter_context(patch(f"lns2_selector.evaluation.pressure_reset.{name}", return_value=value))
        stack.enter_context(patch("lns2_selector.evaluation.pressure_reset._CollectionRunLock", return_value=nullcontext()))
        self.context = stack.enter_context(patch("lns2_selector.evaluation.pressure_reset.multiprocessing.get_context")).return_value

    def fake_process(self, timeout=False):
        def factory(target, args):
            job = args[0]
            class Process:
                exitcode = 0
                alive = timeout
                def start(self):
                    if not timeout:
                        write_json(Path(job["output"]), {"status": "ok", "binding": job["binding"],
                                   "key": job["key"], "repairs_executed": 0, "observation": {},
                                   "state_fingerprint": "state", "quality": {"feasible": False, "colliding_pairs": 1}})
                def join(self, *args):
                    pass
                def is_alive(self):
                    return self.alive
                def terminate(self):
                    self.alive = False
                def kill(self):
                    self.alive = False
                def close(self):
                    pass
            return Process()
        self.context.Process.side_effect = factory

    def test_resume_reuses_completed_reset(self):
        self.fake_process()
        first = run_resets(self.root, self.root / "config.json")
        second = run_resets(self.root, self.root / "config.json")
        self.assertEqual(first, second)
        self.assertEqual(self.context.Process.call_count, 1)

    def test_safe_stop_does_not_launch_reset(self):
        (self.output / "STOP_AFTER_RESET").touch()
        with self.assertRaises(InterruptedError):
            run_resets(self.root, self.root / "config.json")
        self.context.Process.assert_not_called()
        self.assertEqual(read_json(self.output / "run_status.json")["status"], "paused")

    def test_timeout_is_retained_not_retried(self):
        self.fake_process(timeout=True)
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, "timeout"):
                run_resets(self.root, self.root / "config.json")
        self.assertEqual(self.context.Process.call_count, 1)
        self.assertEqual(read_json(self.output / "run_status.json")["status"], "failed")


if __name__ == "__main__":
    unittest.main()
