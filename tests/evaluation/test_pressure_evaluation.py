from collections import Counter
from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments._common import json_fingerprint, read_json, sha256_file, write_json, write_jsonl
from lns2_selector.evaluation.path_quality_execution import CONTROLLERS, execution_schedule
from lns2_selector.evaluation import pressure_evaluation as module


ROOT = Path(__file__).resolve().parents[2]


class PressureEvaluationTests(unittest.TestCase):
    @staticmethod
    def full_schedule():
        cases = [{"task_id": f"map{m}_task{t}", "map_id": f"map{m}", "family": "warehouse",
                  "solver_seeds": [61, 62], "status": "static_ready_runtime_unverified"}
                 for m in range(8) for t in range(6)]
        return execution_schedule(cases, {"first_feasible_budget_seconds": 120,
                                         "total_planning_budgets_seconds": [60, 120]})

    def test_recovery_scope_preserves_original_items_and_order(self):
        schedule = self.full_schedule()
        self.assertEqual(module.scoped_schedule({"expected_episodes": 864}, schedule), schedule)
        selected = module.scoped_schedule({"protocol_scope": "fixed_budget_only", "expected_episodes": 576}, schedule)
        self.assertEqual(selected, schedule[288:])
        self.assertEqual(selected[0]["schedule_index"], 288)
        self.assertTrue(all(r["protocol"] == "fixed_budget" for r in selected))
        self.assertEqual(set(Counter(tuple(r["controller"] for r in selected[i:i+3])
                                     for i in range(0, len(selected), 3)).values()), {32})

    def test_unknown_or_wrong_count_scope_rejected(self):
        for design in ({"protocol_scope": "successful_only", "expected_episodes": 864},
                       {"protocol_scope": "fixed_budget_only", "expected_episodes": 864}):
            with self.assertRaises(ValueError):
                module.scoped_schedule(design, self.full_schedule())
        with self.assertRaisesRegex(ValueError, "evidence"):
            module.verify_recovery_scope(ROOT, {"protocol_scope": "fixed_budget_only"}, [])

    def recovery_fixture(self, root):
        schedule = self.full_schedule()
        source = root / "build/old"
        registry = {"schema": "fixture"}
        registry["fingerprint"] = json_fingerprint(registry)
        write_json(source / "registration.json", registry)
        write_json(root / "build/audit.json", {"status": "completed", "solver_calls": 0,
                   "formal_timing_resumed": False, "batches": [{"batch": "old", "registration": registry["fingerprint"]}]})
        rows = [{"batch": "old", "item": row, "timing_authorized": False,
                 "action": "retain_complete_old_protocol" if i < 288 else "pending_new_version"}
                for i, row in enumerate(schedule)]
        write_jsonl(root / "build/scope.jsonl", rows)
        recovery = {"source_output": "build/old"}
        for key, path in (("registration", "build/old/registration.json"), ("audit", "build/audit.json"),
                          ("recommended_scope", "build/scope.jsonl")):
            recovery[key] = {"manifest": path, "sha256": sha256_file(root / path)}
        return {"output": "build/new", "recovery": recovery}, schedule[288:]

    def test_recovery_binding_and_changed_schedule(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            design, schedule = self.recovery_fixture(root)
            self.assertEqual(len(module.verify_recovery_scope(root, design, schedule)), 3)
            for changed in (schedule[::-1], schedule[:-1], [dict(schedule[0], solver_seed=99), *schedule[1:]]):
                with self.assertRaisesRegex(ValueError, "audited recovery"):
                    module.verify_recovery_scope(root, design, changed)

    def test_recovery_rejects_overlap_and_changed_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            design, schedule = self.recovery_fixture(root)
            for output in ("build/old", "build/old/new", "build"):
                with self.assertRaisesRegex(ValueError, "overlap"):
                    module.verify_recovery_scope(root, {**design, "output": output}, schedule)
            write_json(root / "build/audit.json", {"status": "changed"})
            with self.assertRaises(ValueError):
                module.verify_recovery_scope(root, design, schedule)

    def test_recovery_rejects_authorized_scope(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            design, schedule = self.recovery_fixture(root)
            rows = module.read_jsonl(root / "build/scope.jsonl")
            rows[0]["timing_authorized"] = True
            write_jsonl(root / "build/scope.jsonl", rows)
            design["recovery"]["recommended_scope"]["sha256"] = sha256_file(root / "build/scope.jsonl")
            with self.assertRaisesRegex(ValueError, "implicit timing"):
                module.verify_recovery_scope(root, design, schedule)

    def test_new_design_counts_and_legacy_compatibility(self):
        for file, count in (("path_quality_pressure_evaluation_v1.json", 864),
                            ("path_quality_pressure_deadline_recovery_v1.json", 576),
                            ("path_quality_pressure_deadline_recovery_replica2.json", 864)):
            design, output = module.load_design(ROOT, ROOT / "configs" / file)
            self.assertEqual(design["expected_episodes"], count)
            self.assertTrue(design["timing_requires_separate_authorization"])
            self.assertEqual(design["controllers"], list(CONTROLLERS))

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

    def run_fake_collection(self, root, pause=False, resume=False, existing=False):
        output = root / "build/run"
        output.mkdir(parents=True, exist_ok=True)
        if pause:
            (output / "pause.request").touch()
        item = {"task_id": "task", "job_id": "job", "controller": "official_adaptive"}
        report = {"cases": [{"task_id": "task"}], "execution_schedule": [item]}
        if existing:
            report["execution_schedule"].append({**item, "job_id": "job2"})
            write_jsonl(output / "collection_manifest.jsonl", [{"job_id": "job", "status": "completed",
                "summary": {"error": None}, "environment": {"recorded_at_original_run": True}}])
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
            read_artifact=unittest.mock.Mock(return_value={"error": None}),
            inspect_episode=unittest.mock.Mock(return_value={"status": "completed"})), patch.object(
            module.cohort, "supervise_episode", return_value={"error": None}) as supervise:
            result = module.collect_pressure(root, root / "design.json", authorize=True, quiet_machine=True, on_ac=True, resume=resume)
            return result, supervise.call_count

    def test_resume_pause_preserves_completed_count_and_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, count = self.run_fake_collection(root, pause=True, resume=True, existing=True)
            self.assertEqual(result["completed"], 1)
            self.assertEqual(count, 0)
            saved = module.read_jsonl(root / "build/run/collection_manifest.jsonl")
            self.assertEqual(saved[0]["environment"], {"recorded_at_original_run": True})

    def test_resume_skips_completed_solver_and_retains_original_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result, count = self.run_fake_collection(root, resume=True, existing=True)
            self.assertEqual((result["completed"], count), (2, 1))
            saved = module.read_jsonl(root / "build/run/collection_manifest.jsonl")
            self.assertEqual(saved[0]["environment"], {"recorded_at_original_run": True})

    def test_existing_manifest_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "request resume"):
                self.run_fake_collection(Path(folder), existing=True)

    def test_unmanifested_episode_requires_audit_not_automatic_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "build/run/episode/binding.json", {"binding": "saved"})
            with self.assertRaisesRegex(ValueError, "interruption audit"):
                self.run_fake_collection(root, resume=True)

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
