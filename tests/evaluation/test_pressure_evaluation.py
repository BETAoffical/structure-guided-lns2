from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments._common import read_json
from lns2_selector.evaluation.path_quality_execution import CONTROLLERS, execution_schedule
from lns2_selector.evaluation import pressure_evaluation as module


ROOT = Path(__file__).resolve().parents[2]


class PressureEvaluationTests(unittest.TestCase):
    def test_no_implicit_timing(self):
        with patch.object(module, "evaluation_path") as prepare:
            with self.assertRaises(PermissionError):
                module.collect_pressure(ROOT, ROOT / "unused.json")
            prepare.assert_not_called()

    def test_all_tasks_all_seeds_three_protocols(self):
        cases = [{"task_id": f"map{m}_task{t}", "map_id": f"map{m}", "family": "warehouse",
                  "solver_seeds": [61, 62], "status": "static_ready_runtime_unverified"}
                 for m in range(8) for t in range(6)]
        protocol = {"first_feasible_budget_seconds": 120, "total_planning_budgets_seconds": [60, 120]}
        schedule = execution_schedule(cases, protocol)
        self.assertEqual(len(schedule), 864)
        self.assertEqual(len({r["job_id"] for r in schedule}), 864)
        self.assertEqual(len({(r["task_id"], r["solver_seed"], r["budget_seconds"]) for r in schedule}), 192)
        self.assertEqual(schedule, execution_schedule(cases[::-1], protocol))
        counts = Counter((r["protocol"], r["budget_seconds"], r["controller"]) for r in schedule)
        self.assertEqual(set(counts.values()), {96})
        orders = Counter(tuple(r["controller"] for r in schedule[i:i + 3]) for i in range(0, len(schedule), 3))
        self.assertEqual(len(orders), 6)
        self.assertEqual(set(orders.values()), {48})
        self.assertTrue(all(r["repair_iteration_cap"] is None and not r["execution_authorized"] for r in schedule))

    def test_runtime_protocol_remains_frozen(self):
        design = read_json(ROOT / "configs/path_quality_pressure_evaluation_v1.json")
        original = read_json(ROOT / "configs/path_quality_preflight_v1.json")
        self.assertEqual(design["controllers"], list(CONTROLLERS))
        self.assertEqual(original["protocol"]["total_planning_budgets_seconds"], [60, 120])
        self.assertTrue(design["timing_requires_separate_authorization"])

    def test_write_and_copy_reject_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            module.write_once(root / "source.json", {"value": 1})
            module.copy_once(root / "source.json", root / "copy.json")
            module.copy_once(root / "source.json", root / "copy.json")
            with self.assertRaisesRegex(ValueError, "changed"):
                module.write_once(root / "source.json", {"value": 2})
            module.write_json(root / "copy.json", {"value": 3})
            with self.assertRaisesRegex(ValueError, "differs"):
                module.copy_once(root / "source.json", root / "copy.json")

    def run_fake_collection(self, root, pause=False):
        output = root / "build/run"
        output.mkdir(parents=True)
        if pause:
            (output / "pause.request").touch()
        item = {"task_id": "task", "job_id": "job", "controller": "official_adaptive"}
        report = {"cases": [{"task_id": "task"}], "execution_schedule": [item]}
        config = {"branch": "codex/test", "remote": "origin"}
        registry = {"fingerprint": "reg"}
        module.write_json(output / "admission.json", {})
        def git(root, *args):
            if args == ("status", "--porcelain"):
                return ""
            if args == ("branch", "--show-current"):
                return "codex/test"
            return "sha refs/heads/codex/test" if args[0] == "ls-remote" else "sha"
        with patch.object(module, "evaluation_path", return_value=root / "config.json"), patch.object(
            module, "_CollectionRunLock", return_value=nullcontext()), patch.multiple(module.cohort,
            verified_registration=unittest.mock.Mock(return_value=(config, output, registry, report)),
            load_admission=unittest.mock.Mock(return_value={"key": {}}),
            admission_key=unittest.mock.Mock(return_value="key"),
            git_read=unittest.mock.Mock(side_effect=git),
            runtime_environment=unittest.mock.Mock(return_value={"load_average": [0], "logical_cpus": 20}),
            bound_spec=unittest.mock.Mock(return_value={"output": str(output / "episode"), "binding": "b"}),
            inspect_episode=unittest.mock.Mock(return_value={"status": "completed"})), patch.object(
            module.cohort, "supervise_episode", return_value={"error": None}) as supervise:
            result = module.collect_pressure(root, root / "design.json", authorize=True, quiet_machine=True, on_ac=True)
            return result, supervise.call_count

    def test_pause_never_starts_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            result, count = self.run_fake_collection(Path(folder), pause=True)
            self.assertEqual(result["status"], "paused")
            self.assertEqual(count, 0)

    def test_completion_uses_schedule_size_not_252(self):
        with tempfile.TemporaryDirectory() as folder:
            result, count = self.run_fake_collection(Path(folder))
            self.assertEqual(result["completed"], 1)
            self.assertEqual(count, 1)


if __name__ == "__main__":
    unittest.main()
