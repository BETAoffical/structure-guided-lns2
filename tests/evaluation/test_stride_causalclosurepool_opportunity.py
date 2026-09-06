from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.stride_causalclosurepool_opportunity import (
    EXECUTION_SCHEMA,
    EXPERIMENT_ID,
    TRIAL_SCHEMA,
    _opportunity_summary,
    _valid_trial,
)


class CausalClosurePoolOpportunityTest(unittest.TestCase):
    def test_summary_reports_pool_opportunity_without_selector_metrics(self) -> None:
        rows = [
            {
                "new_stably_dominates_base_best": True,
                "stable_frontier_addition_count": 1,
                "new_best_seed_mean_advantage": 0.1,
                "new_candidate_count": 3,
                "mean_new_candidate_size": 9.0,
            },
            {
                "new_stably_dominates_base_best": False,
                "stable_frontier_addition_count": 0,
                "new_best_seed_mean_advantage": -0.02,
                "new_candidate_count": 2,
                "mean_new_candidate_size": 6.0,
            },
        ]
        summary = _opportunity_summary(rows)
        self.assertEqual(summary["robust_best_base_opportunity_fraction"], 0.5)
        self.assertEqual(summary["stable_frontier_addition_fraction"], 0.5)
        self.assertEqual(summary["mean_new_candidate_count"], 2.5)
        self.assertEqual(summary["mean_new_candidate_size"], 7.5)
        self.assertNotIn("ttf", summary)
        self.assertNotIn("ranker", summary)
        self.assertEqual(set(_opportunity_summary([]).values()), {0})

    def test_execution_freezes_ranker_free_bounded_native_audit(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "configs" / "stride_causalclosurepool_v2_opportunity_execution.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["schema"], EXECUTION_SCHEMA)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(config["execution"]["workers"], 16)
        self.assertEqual(config["execution"]["per_job_timeout_seconds"], 300)
        self.assertEqual(config["cohort"]["causal_candidate_count"], 930)
        self.assertFalse(config["claim_boundary"]["current_ranker_used"])
        self.assertFalse(config["claim_boundary"]["ttf_experiment_allowed"])

    @patch("experiments.stride_causalclosurepool_opportunity._forbidden_hits", return_value=set())
    def test_trial_validation_requires_exact_paired_identity(self, _forbidden: object) -> None:
        job = {
            "candidate": {"candidate_id": "candidate-a"},
            "state_fingerprint": "state-a",
            "trial_index": 3,
            "pp_seed": 17,
        }
        row = {
            "schema": TRIAL_SCHEMA,
            "run_fingerprint": "run-a",
            "state_fingerprint": "state-a",
            "candidate_id": "candidate-a",
            "trial_index": 3,
            "pp_seed": 17,
            "before_conflicts": 5,
            "conflicts_after": 3,
            "normalized_conflict_reduction": 0.4,
            "native_action_validated": True,
            "repair_order_controlled": False,
            "runtime_fields_stored": False,
            "future_trajectory_stored": False,
        }
        self.assertTrue(_valid_trial(row, job=job, run_fingerprint="run-a"))
        row["pp_seed"] = 18
        self.assertFalse(_valid_trial(row, job=job, run_fingerprint="run-a"))


if __name__ == "__main__":
    unittest.main()
