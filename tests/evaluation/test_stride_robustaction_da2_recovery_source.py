from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_robustaction_da2_recovery_source import (
    summarize_episode_capacity,
    validate_da2_recovery_source_design,
)


class RobustActionDA2RecoverySourceTests(unittest.TestCase):
    def _root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def _config(self) -> dict:
        return json.loads(
            (
                self._root()
                / "configs"
                / "stride_robustaction_structpool_da2_recovery_source_design.json"
            ).read_text(encoding="utf-8")
        )

    def test_registered_source_design_is_valid(self) -> None:
        config = self._config()
        validate_da2_recovery_source_design(config, project_root=self._root())
        self.assertEqual(config["source_contract"]["expected_total_episode_rows"], 64)
        self.assertEqual(config["source_contract"]["max_decisions"], 12)
        self.assertTrue(config["source_v4_frozen_capacity"]["must_remain_unfiltered"])

    def test_runtime_freezes_historical_twelve_step_source(self) -> None:
        runtime = json.loads(
            (
                self._root()
                / "configs"
                / "stride_robustaction_structpool_da2_recovery_source_runtime.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(runtime["solver_seeds"], [1, 2])
        self.assertEqual(
            runtime["policies"], ["official_adaptive", "realized_dynamic"]
        )
        self.assertEqual(runtime["max_decisions"], 12)
        self.assertEqual(runtime["metric_iteration_budget"], 12)
        self.assertEqual(runtime["environment"]["max_repair_iterations"], 12)
        self.assertTrue(runtime["deterministic_pp_replay"])
        self.assertEqual(runtime["dataset_design"]["map_count"], 8)
        self.assertEqual(runtime["dataset_design"]["instance_count"], 16)
        self.assertEqual(runtime["qualification"]["minimum_nonzero_states"], 32)
        self.assertEqual(
            runtime["qualification"]["minimum_nonzero_states_per_agent_band"],
            {"high": 32},
        )

    def test_capacity_counts_all_episodes_without_success_filtering(self) -> None:
        report = summarize_episode_capacity(
            {
                "official_adaptive": {"a0": 0, "a1": 1, "a2": 4},
                "realized_dynamic": {"v0": 0, "v1": 2, "v2": 3},
            },
            maximum_per_episode=2,
        )
        self.assertEqual(report["episode_count"], 6)
        self.assertEqual(report["raw_positive_pre_action_state_count"], 10)
        self.assertEqual(report["maximum_obtainable_unique_state_ids"], 7)
        self.assertEqual(report["maximum_obtainable_unique_episode_ids"], 4)
        self.assertEqual(
            report["policies"]["official_adaptive"]["episode_state_count_bands"],
            {"zero": 1, "one": 1, "two_plus": 1},
        )

    def test_capacity_rejects_duplicate_episode_ids_across_policies(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate source episode ids"):
            summarize_episode_capacity(
                {
                    "official_adaptive": {"shared": 1},
                    "realized_dynamic": {"shared": 2},
                },
                maximum_per_episode=2,
            )

    def test_outcome_and_failure_boundary_cannot_drift(self) -> None:
        config = copy.deepcopy(self._config())
        config["outcome_boundary"]["ttf_read_for_capacity"] = True
        with self.assertRaisesRegex(ValueError, "outcome boundary changed"):
            validate_da2_recovery_source_design(config, project_root=self._root())

        config = copy.deepcopy(self._config())
        config["source_contract"]["failure_action"] = "keep_successes"
        with self.assertRaisesRegex(ValueError, "source contract changed"):
            validate_da2_recovery_source_design(config, project_root=self._root())


if __name__ == "__main__":
    unittest.main()
