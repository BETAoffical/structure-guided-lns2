from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.stride_repairclosurepool import (
    DESIGN_SCHEMA,
    EXPERIMENT_ID,
    _summary,
    _valid_trial,
)


class RepairClosurePoolExperimentTest(unittest.TestCase):
    def test_design_freezes_no_ranker_or_repair_order(self) -> None:
        root = Path(__file__).resolve().parents[2]
        config = json.loads(
            (root / "configs" / "stride_repairclosurepool_v1_design.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(config["schema"], DESIGN_SCHEMA)
        self.assertEqual(config["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(config["generator"]["fixed_preferred_sizes"], [])
        self.assertFalse(config["claim_boundary"]["current_ranker_used"])
        self.assertFalse(config["claim_boundary"]["repair_order_intervention_allowed"])
        self.assertFalse(config["claim_boundary"]["ttf_experiment_allowed"])
        self.assertEqual(config["execution"]["workers"], 16)
        self.assertEqual(config["execution"]["per_job_timeout_seconds"], 300)

    @patch("experiments.stride_repairclosurepool._forbidden_hits", return_value=set())
    def test_trial_validation_requires_exact_native_identity(self, _forbidden: object) -> None:
        candidate = {"candidate_id": "candidate-a"}
        job = {
            "candidate": candidate,
            "state_fingerprint": "state-a",
            "trial_index": 3,
            "pp_seed": 17,
        }
        row = {
            "schema": "lns2.stride.repairclosurepool_trial.v1",
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
        summary = _summary(rows)
        self.assertEqual(summary["robust_best_base_opportunity_fraction"], 0.5)
        self.assertEqual(summary["stable_frontier_addition_fraction"], 0.5)
        self.assertNotIn("ttf", summary)
        self.assertNotIn("ranker", summary)


if __name__ == "__main__":
    unittest.main()
