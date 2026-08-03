from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_repairability_preflight import (
    _variant,
    validate_repairability_load_confirmation_source,
    validate_repairability_reserve_extension_source,
)


class StrideRepairabilityPreflightTest(unittest.TestCase):
    def _config(self) -> dict:
        root = Path(__file__).resolve().parents[2]
        return json.loads(
            (
                root
                / "configs"
                / "stride_repairability_load_confirmation_source.json"
            ).read_text(encoding="utf-8")
        )

    def test_confirmation_freezes_split_preserving_replacements(self) -> None:
        config = self._config()
        validate_repairability_load_confirmation_source(config)
        replacements = {
            row["primary_map_id"]: row["id"]
            for row in config["benchmarks"]
            if row["primary_map_id"] != row["id"]
        }
        self.assertEqual(
            replacements, {"arena2": "den206d", "den001d": "den011d"}
        )
        self.assertEqual(config["expected_qualification_job_count"], 96)
        self.assertEqual(config["minimum_conflicting_jobs_per_effective_map"], 8)

    def test_confirmation_rejects_outcome_or_seed_drift(self) -> None:
        config = self._config()
        config["selection_inputs_forbidden"].remove("controller_ttf")
        with self.assertRaisesRegex(ValueError, "outcome boundary changed"):
            validate_repairability_load_confirmation_source(config)
        config = self._config()
        config["solver_seeds"] = [1, 3]
        with self.assertRaisesRegex(ValueError, "identity changed"):
            validate_repairability_load_confirmation_source(config)

    def test_variant_is_derived_only_from_registered_task_identity(self) -> None:
        self.assertEqual(
            _variant({"task_id": "m__derived_uniform_random__task_seed_1"}),
            "uniform_random",
        )
        self.assertEqual(
            _variant({"task_id": "m__derived_opposite_exchange__task_seed_1"}),
            "opposite_exchange",
        )
        with self.assertRaisesRegex(ValueError, "unregistered"):
            _variant({"task_id": "m__unknown"})

    def test_den005_reserve_extension_is_fixed_before_candidate_outcomes(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (
                root
                / "configs"
                / "stride_repairability_den005_reserve_source.json"
            ).read_text(encoding="utf-8")
        )
        validate_repairability_reserve_extension_source(config)
        self.assertEqual(config["benchmarks"][0]["id"], "ht_mansion_n")
        self.assertEqual(config["benchmarks"][0]["agent_counts"], [900])
        self.assertIn("candidate_repair_outcome", config["selection_inputs_forbidden"])


if __name__ == "__main__":
    unittest.main()
