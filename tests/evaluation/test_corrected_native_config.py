from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CorrectedNativeConfigurationTests(unittest.TestCase):
    def test_formal_rerun_has_no_repair_iteration_cap(self) -> None:
        path = (
            PROJECT_ROOT
            / "configs"
            / "balanced_wall_clock_corrected_native_v1.json"
        )
        configuration = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(
            configuration["experiment_revision"],
            "balanced-wall-clock-corrected-native-v1",
        )
        self.assertEqual(configuration["environment"]["max_repair_iterations"], 0)
        self.assertEqual(configuration["max_decisions"], 0)
        self.assertEqual(configuration["metric_iteration_budget"], 100)
        self.assertEqual(
            configuration["policies"],
            ["official_adaptive", "realized_dynamic"],
        )
        self.assertEqual(configuration["wall_time_budget_seconds"], 600.0)
        design = configuration["dataset_design"]
        self.assertIn("source_task_registration", design)
        self.assertNotIn("cohort_registration", design)

    def test_historical_formal_configuration_remains_immutable(self) -> None:
        path = (
            PROJECT_ROOT
            / "configs"
            / "balanced_wall_clock_collection_v3.json"
        )
        configuration = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(configuration["environment"]["max_repair_iterations"], 100)
        self.assertEqual(configuration["max_decisions"], 100)


if __name__ == "__main__":
    unittest.main()
