from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_repairability_source import (
    validate_repairability_source_cohort_config,
)


class StrideRepairabilitySourceTest(unittest.TestCase):
    def _root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _config(self) -> dict:
        return json.loads(
            (
                self._root()
                / "configs"
                / "stride_repairability_source_cohort.json"
            ).read_text(encoding="utf-8")
        )

    def test_source_cohort_freezes_effective_split_and_replacements(self) -> None:
        config = self._config()
        validate_repairability_source_cohort_config(config)
        split = config["effective_map_split"]
        self.assertEqual(len(split["train"]), 16)
        self.assertEqual(len(split["validation"]), 6)
        self.assertFalse(set(split["train"]) & set(split["validation"]))
        self.assertEqual(
            config["replacements"],
            {
                "arena2": "den206d",
                "den001d": "den011d",
                "den005d": "ht_mansion_n",
            },
        )
        self.assertEqual(config["expected_task_count"], 44)

    def test_source_cohort_rejects_outcome_and_identity_drift(self) -> None:
        config = self._config()
        config["selection_inputs_forbidden"].remove("controller_ttf")
        with self.assertRaisesRegex(ValueError, "outcome boundary changed"):
            validate_repairability_source_cohort_config(config)

        config = self._config()
        config["controller_id"] = "v2-full"
        with self.assertRaisesRegex(ValueError, "contract changed"):
            validate_repairability_source_cohort_config(config)

    def test_runtime_source_contract_is_uncapped_by_selector_logic(self) -> None:
        runtime = json.loads(
            (
                self._root()
                / "configs"
                / "stride_repairability_source_runtime.json"
            ).read_text(encoding="utf-8")
        )
        self.assertFalse(runtime["formal"])
        self.assertEqual(runtime["split"], "balanced_wall_clock")
        self.assertEqual(runtime["solver_seeds"], [1, 2])
        self.assertEqual(
            runtime["policies"], ["official_adaptive", "realized_dynamic"]
        )
        self.assertEqual(runtime["max_decisions"], 12)
        self.assertEqual(runtime["metric_iteration_budget"], 12)
        self.assertNotIn("remaining_time_guard", runtime)
        self.assertNotIn("selector_time_limit", runtime)
        self.assertTrue(runtime["deterministic_pp_replay"])
        self.assertEqual(runtime["dataset_design"]["map_count"], 22)
        self.assertEqual(runtime["dataset_design"]["instance_count"], 44)


if __name__ == "__main__":
    unittest.main()
