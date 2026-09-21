import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments._common import read_json, sha256_file, write_json
from scripts import audit_sa_restart_accounting as a


def row(seed=61, ttf=20, task="task", map_id="map", controller="dual16_sa"):
    return dict(task_id=task, map_id=map_id, controller=controller, solver_seed=seed,
                success=ttf is not None, found_within_budget=ttf is not None,
                status="completed" if ttf is not None else "no_feasible_solution",
                ttf_seconds=ttf, protocol="first_feasible", budget_seconds=120,
                timed_workers=1, repair_iteration_cap=None,
                initial_fingerprint=f"{task}-{seed}", job_id=f"{task}-{seed}-{controller}")


def config():
    return dict(budget_seconds=120, cutoff_seconds=60, seed_orders=[[61, 62], [62, 61]],
                expected_tasks=2, expected_maps=2, bootstrap_replicates=100, bootstrap_seed=9,
                sources=[dict(controllers=["dual16_sa"])], output="build/accounting",
                evidence_role="retrospective_conditional_log_accounting_not_runtime_confirmation")


def source_fixture(root):
    cfg = config()
    cfg["native_sha256"] = "frozen-native"
    base = root / "source"
    write_json(root / "map.json", dict(cells=[0]))
    inputs = {"map.json": sha256_file(root / "map.json")}
    rows = [row(), row(seed=62), row(task="other", map_id="map2", ttf=None),
            row(seed=62, task="other", map_id="map2")]
    cases = [dict(task_id=task, map_id=map_id, files=dict(map_file="map.json"),
                  static_audit=dict(agent_count=1), pressure_design={})
             for task, map_id in (("task", "map"), ("other", "map2"))]
    jobs = {}
    for item in rows:
        bind = "binding-" + item["job_id"]
        files = {
            "binding.json": dict(binding=bind, item=item, native_sha256="frozen-native"),
            "result.json": dict(binding=bind, payload=dict(status=item["status"],
                                success_by_deadline=item["success"], first_feasible_elapsed_seconds=item["ttf_seconds"])),
            "initial.json": dict(binding=bind, payload=dict(state_fingerprint=item["initial_fingerprint"])),
            "supervisor.json": dict(binding=bind, payload=dict(error=False, status=item["status"]))}
        pins = {}
        for name, value in files.items():
            path = base / "episodes" / item["job_id"] / name
            write_json(path, value)
            pins[name] = sha256_file(path)
        jobs[item["job_id"]] = dict(status=item["status"], success=item["success"], files=pins)
    write_json(base / "registration.json", dict(fingerprint="source", native_sha256="frozen-native",
               controllers=["dual16_sa"], cases=cases, inputs=inputs, schedule=rows))
    write_json(base / "manifest.json", dict(binding="source", jobs=jobs))
    # A copied comparison arm must not silently become another source episode.
    write_json(base / "analysis/report.json", dict(episodes=rows + [row(controller="other")]))
    cfg["sources"] = [dict(root="source", controllers=["dual16_sa"],
                           pins={name: sha256_file(base / name) for name in
                                 ("registration.json", "manifest.json", "analysis/report.json")})]
    return cfg, rows


class RestartAccountingTest(unittest.TestCase):
    def test_source_verification_excludes_copied_comparison_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, expected = source_fixture(root)
            loaded, receipts = a.load_sources(cfg, root)
            self.assertEqual(loaded, expected)
            self.assertEqual(receipts[0]["summary_files_verified"], 16)

    def test_changed_raw_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, rows = source_fixture(root)
            write_json(root / "source/episodes" / rows[0]["job_id"] / "result.json", dict(wrong=True))
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                a.load_sources(cfg, root)

    def test_consistent_hash_does_not_hide_raw_ttf_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, rows = source_fixture(root)
            path = root / "source/episodes" / rows[0]["job_id"] / "result.json"
            value = read_json(path)
            value["payload"]["first_feasible_elapsed_seconds"] = 99
            write_json(path, value)
            manifest_path = root / "source/manifest.json"
            manifest = read_json(manifest_path)
            manifest["jobs"][rows[0]["job_id"]]["files"]["result.json"] = sha256_file(path)
            write_json(manifest_path, manifest)
            cfg["sources"][0]["pins"]["manifest.json"] = sha256_file(manifest_path)
            with self.assertRaisesRegex(ValueError, "raw TTF differs"):
                a.load_sources(cfg, root)

    def test_duplicate_registered_controller_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _ = source_fixture(root)
            cfg["sources"] *= 2
            with self.assertRaisesRegex(ValueError, "duplicate controller"):
                a.load_sources(cfg, root)

    def test_wrong_native_and_changed_task_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg, _ = source_fixture(root)
            with self.assertRaisesRegex(ValueError, "native identity"):
                a.load_sources(dict(cfg, native_sha256="other"), root)
            write_json(root / "map.json", dict(cells=[1]))
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                a.load_sources(cfg, root)

    def test_no_oracle_seed_choice_when_first_is_early(self):
        result = a.account(row(ttf=50), row(seed=62, ttf=1), 60, 120)
        self.assertEqual(result["restart_ttf"], 50)
        self.assertFalse(result["switched"])

    def test_restart_charges_full_first_segment_and_second_reset(self):
        result = a.account(row(ttf=None), row(seed=62, ttf=17), 60, 120)
        self.assertEqual(result["restart_ttf"], 77)
        self.assertTrue(result["gained"])
        self.assertEqual(result["baseline_capped"], 120)
        self.assertIsNone(result["baseline_ttf"])

    def test_late_success_can_be_lost(self):
        result = a.account(row(ttf=61), row(seed=62, ttf=90), 60, 120)
        self.assertTrue(result["lost"])
        self.assertTrue(result["sacrificed_late_first_success"])
        self.assertIsNone(result["restart_ttf"])
        self.assertEqual(result["restart_capped"], 120)

    def test_boundary_is_inclusive(self):
        self.assertFalse(a.account(row(ttf=60), row(seed=62, ttf=None), 60, 120)["switched"])
        self.assertEqual(a.account(row(ttf=None), row(seed=62, ttf=60), 60, 120)["restart_ttf"], 120)

    def test_reject_unknown_negative_nan_or_inconsistent_results(self):
        for changes in (dict(status="hard_timeout"), dict(ttf_seconds=-1),
                        dict(ttf_seconds=float("nan")), dict(found_within_budget=False),
                        dict(ttf_seconds=121), dict(timed_workers=20),
                        dict(repair_iteration_cap=100), dict(protocol="anytime")):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                a.validate_row(dict(row(), **changes), 120)
        with self.assertRaises(ValueError):
            a.validate_row(dict(row(ttf=None), ttf_seconds=0), 120)

    def test_restart_preserves_task_and_controller_not_initialization(self):
        for second in (row(seed=62, task="wrong"), row(seed=62, controller="official_sa"), row()):
            with self.assertRaises(ValueError):
                a.account(row(), second, 60, 120)
        a.account(row(), row(seed=62), 60, 120)

    def test_both_orders_reported_without_treating_them_as_distinct_tasks(self):
        rows = [row(ttf=None), row(seed=62, ttf=10),
                row(task="other", map_id="map2", ttf=20),
                row(seed=62, task="other", map_id="map2", ttf=80)]
        cases, summaries = a.analyze(rows, config())
        summary = summaries["dual16_sa"]
        self.assertEqual((summary["distinct_tasks"], summary["ordered_cases"]), (2, 4))
        self.assertEqual(set(summary["by_order"]), {"61-62", "62-61"})
        self.assertEqual(summary["bootstrap"]["maps"], 2)
        self.assertEqual((cases, summaries), a.analyze(list(reversed(rows)), config()))

    def test_missing_duplicate_extra_seed_and_cutoff_tuning_rejected(self):
        rows = [row(), row(seed=62), row(task="other", map_id="map2"),
                row(seed=62, task="other", map_id="map2")]
        for values in (rows[:-1], rows + rows[:1], rows + [row(seed=63)]):
            with self.assertRaises(ValueError):
                a.analyze(values, config())
        with self.assertRaises(ValueError):
            a.analyze(rows, dict(config(), cutoff_seconds=40))

    def test_bootstrap_groups_dependent_orders_with_map(self):
        values = [a.account(row(ttf=None), row(seed=62, ttf=20), 60, 120)]
        doubled = values * 2
        self.assertEqual(a.map_bootstrap(values, 100, 1), a.map_bootstrap(doubled, 100, 1))
        self.assertEqual(a.map_bootstrap(values, 100, 1)["success_gain_pp"], [100, 100])

    def test_pin_and_containment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "input.json", dict(value=1))
            sha = sha256_file(root / "input.json")
            a.checked(root, "input.json", sha)
            write_json(root / "input.json", dict(value=2))
            with self.assertRaises(ValueError):
                a.checked(root, "input.json", sha)
            with self.assertRaises(ValueError):
                a.contained(root, "../outside")

    def test_run_deterministic_and_refuses_changed_result(self):
        rows = [row(), row(seed=62), row(task="other", map_id="map2"),
                row(seed=62, task="other", map_id="map2")]
        with tempfile.TemporaryDirectory() as tmp, patch.object(a, "load_sources", return_value=(rows, [])):
            root = Path(tmp)
            path = root / "config.json"
            write_json(path, config())
            first = a.run(path, root)
            self.assertEqual(first, a.run(path, root))
            altered = copy.deepcopy(first)
            altered["episode_count"] = 999
            write_json(root / "build/accounting/report.json", altered)
            with self.assertRaises(ValueError):
                a.run(path, root)
            self.assertEqual(read_json(root / "build/accounting/report.json"), altered)


if __name__ == "__main__":
    unittest.main()
