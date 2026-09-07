import copy
import unittest
from pathlib import Path

from experiments._common import read_json
from lns2_selector.evaluation.path_quality_execution import execution_schedule
from scripts.audit_path_quality_pressure_batches import check_rosters


class PressureBatchAdmissionTests(unittest.TestCase):
    def test_frozen_import_exceptions_are_narrow_and_documented(self):
        root = Path(__file__).resolve().parents[2]
        config = read_json(root / "configs/repository_hygiene.json")
        exceptions = [r for r in config["allowed_unused_imports"]
                      if r["path"] == "lns2_selector/evaluation/path_quality_cohort.py"]
        self.assertEqual({r["name"] for r in exceptions}, {"time", "read_artifact"})
        self.assertEqual(len(exceptions), 2)
        self.assertTrue(all(r.get("reason") for r in exceptions))

    def test_replica_changes_inputs_not_controller_protocol(self):
        root = Path(__file__).resolve().parents[2]
        original = read_json(root / "configs/path_quality_pressure_evaluation_v1.json")
        replica = read_json(root / "configs/path_quality_pressure_evaluation_replica2.json")
        self.assertEqual({key for key in original.keys() | replica.keys()
                          if original.get(key) != replica.get(key)},
                         {"preparation_config", "preparation_report", "reset_report", "output"})
        old_reset = read_json(root / "configs/path_quality_pressure_reset_v2.json")
        new_reset = read_json(root / "configs/path_quality_pressure_reset_replica2.json")
        self.assertEqual({key for key in old_reset.keys() | new_reset.keys()
                          if old_reset.get(key) != new_reset.get(key)},
                         {"preparation_config", "output"})

    def reports(self):
        reports = []
        for batch in range(2):
            cases = []
            for m in range(8):
                for t in range(6):
                    cases.append({"task_id": f"batch{batch}_map{m}_task{t}", "map_id": f"map{m}",
                                  "family": "warehouse", "solver_seeds": [61, 62],
                                  "status": "static_ready_runtime_unverified",
                                  "pressure_design": {"density": (0.15, 0.20, 0.25)[t // 2],
                                      "mode": {"name": ("balanced", "bottleneck_eligible")[t % 2]},
                                      "task_seed": batch * 1000 + m * 6 + t}})
            schedule = execution_schedule(cases, {"first_feasible_budget_seconds": 120,
                                                 "total_planning_budgets_seconds": [60, 120]})
            reports.append({"cases": cases, "execution_schedule": schedule})
        return reports

    def test_complete_replicas_remain_eight_maps(self):
        self.assertEqual(check_rosters(self.reports()), {"independent_maps": 8, "tasks": 96, "episodes": 1728})

    def test_duplicate_task_seed_or_changed_cell_rejected(self):
        original = self.reports()
        for field in ("task_id", "task_seed", "density"):
            rows = copy.deepcopy(original)
            first, second = rows[0]["cases"][0], rows[1]["cases"][0]
            if field == "task_id":
                second[field] = first[field]
            elif field == "task_seed":
                second["pressure_design"][field] = first["pressure_design"][field]
            else:
                second["pressure_design"][field] = 0.3
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_rosters(rows)

    def test_missing_duplicate_or_authorized_schedule_rejected(self):
        for change in ("missing", "duplicate", "authorized"):
            rows = self.reports()
            schedule = rows[1]["execution_schedule"]
            if change == "missing":
                schedule.pop()
            elif change == "duplicate":
                schedule[0] = schedule[1]
            else:
                schedule[0]["execution_authorized"] = True
            with self.subTest(change=change), self.assertRaises(ValueError):
                check_rosters(rows)


if __name__ == "__main__":
    unittest.main()
