from __future__ import annotations

import json
import unittest
from pathlib import Path

from experiments.stride_causaltopopool_opportunity import (
    EXECUTION_SCHEMA,
    EXPERIMENT_ID,
    TRIAL_SCHEMA,
)


class CausalTopoPoolOpportunityTest(unittest.TestCase):
    def test_execution_freezes_strong_ranker_free_gate(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (
                root
                / "configs"
                / "stride_causaltopopool_v1_opportunity_execution.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(config["schema"], EXECUTION_SCHEMA)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(config["cohort"]["topology_candidate_count"], 366)
        self.assertEqual(config["execution"]["workers"], 16)
        self.assertEqual(config["execution"]["per_job_timeout_seconds"], 300)
        self.assertEqual(config["execution"]["full_trial_indices"], list(range(16)))
        self.assertEqual(
            config["readiness_gates"][
                "minimum_hybrid_robust_best_base_opportunity_fraction"
            ],
            0.4,
        )
        self.assertFalse(config["claim_boundary"]["model_training_allowed"])
        self.assertFalse(config["claim_boundary"]["ttf_experiment_allowed"])

    def test_generic_collector_accepts_registered_trial_schema(self) -> None:
        from experiments.stride_causalclosurepool_opportunity import _valid_trial

        job = {
            "trial_schema": TRIAL_SCHEMA,
            "state_fingerprint": "state",
            "candidate": {"candidate_id": "candidate"},
            "trial_index": 0,
            "pp_seed": 7,
        }
        row = {
            "schema": TRIAL_SCHEMA,
            "run_fingerprint": "run",
            "state_fingerprint": "state",
            "candidate_id": "candidate",
            "trial_index": 0,
            "pp_seed": 7,
            "before_conflicts": 4,
            "conflicts_after": 3,
            "normalized_conflict_reduction": 0.25,
            "native_action_validated": True,
            "repair_order_controlled": False,
            "runtime_fields_stored": False,
            "future_trajectory_stored": False,
        }
        self.assertTrue(_valid_trial(row, job=job, run_fingerprint="run"))

    def test_aggregate_metadata_is_joined_from_frozen_candidate(self) -> None:
        source = Path(__file__).resolve().parents[2] / "experiments" / (
            "stride_causaltopopool_opportunity.py"
        )
        text = source.read_text(encoding="utf-8")
        self.assertIn("topology_metadata = {", text)
        self.assertIn('metadata["causaltopo_closure_levels"]', text)


if __name__ == "__main__":
    unittest.main()
