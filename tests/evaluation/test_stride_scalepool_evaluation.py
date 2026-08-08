from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from experiments.stride_scalepool_evaluation import (
    _validate_external_label_audit,
    summarize_scalepool_acceptance,
    validate_scalepool_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "stride_scalepool_v1_registration.json"


class ScalePoolEvaluationTests(unittest.TestCase):
    def test_registered_config_is_valid(self) -> None:
        validate_scalepool_config(json.loads(CONFIG.read_text(encoding="utf-8")))

    def test_config_rejects_gate_or_runtime_drift(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["offline_acceptance"]["global_best_retention_minimum"] = 0.8
        with self.assertRaisesRegex(ValueError, "acceptance gates"):
            validate_scalepool_config(changed)
        changed = copy.deepcopy(config)
        changed["claim_boundary"]["runtime_integration_before_acceptance"] = True
        with self.assertRaisesRegex(ValueError, "claim boundary"):
            validate_scalepool_config(changed)

    def test_acceptance_requires_every_preregistered_gate(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        rows = [
            {
                "map_id": f"map-{index % 2}",
                "layout_mode": f"layout-{index % 2}",
                "global_best_retained": index != 9,
                "first_fixed_half_best_retained": index != 9,
                "second_fixed_half_best_retained": index != 8,
                "normalized_regret": 0.0 if index != 9 else 0.02,
            }
            for index in range(10)
        ]
        result = summarize_scalepool_acceptance(
            rows=rows,
            config=config,
            full_raw_candidate_count=240,
            scalepool_raw_candidate_count=80,
        )
        self.assertTrue(result["passed"])
        failed = copy.deepcopy(rows)
        failed[0]["normalized_regret"] = 0.6
        result = summarize_scalepool_acceptance(
            rows=failed,
            config=config,
            full_raw_candidate_count=240,
            scalepool_raw_candidate_count=80,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["map_group_regret"])

    def test_external_label_audit_is_bound_to_source_hashes(self) -> None:
        artifacts = {
            "repair_trials_sha256": "trials",
            "candidate_aggregates_sha256": "aggregates",
            "state_manifest_sha256": "manifest",
            "state_artifact_tree_sha256": "tree",
        }
        label_report = {
            "run_fingerprint": "run",
            "completed_state_count": 98,
            "candidate_count": 1368,
            "trial_count": 21888,
            "artifacts": dict(artifacts),
        }
        audit_report = {
            "schema": "lns2.stride.structpool_size_label_audit.v1",
            "passed": True,
            "source_artifacts_modified": False,
            "matrix_validation": {"passed": True},
            "source_run_fingerprint": "run",
            "state_count": 98,
            "candidate_count": 1368,
            "trial_count": 21888,
            "source_artifacts": dict(artifacts),
        }
        _validate_external_label_audit(
            label_report=label_report,
            audit_report=audit_report,
            observed_artifacts=artifacts,
        )
        broken = copy.deepcopy(audit_report)
        broken["source_artifacts"]["repair_trials_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "source artifact hashes"):
            _validate_external_label_audit(
                label_report=label_report,
                audit_report=broken,
                observed_artifacts=artifacts,
            )


if __name__ == "__main__":
    unittest.main()
