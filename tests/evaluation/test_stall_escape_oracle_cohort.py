from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments._common import sha256_file
from experiments.stall_escape_oracle_cohort import (
    build_stall_escape_oracle_cohort,
)
from experiments.stall_oracle import STALL_ORACLE_SCHEMA
from experiments.stall_risk_audit import STALL_RISK_AUDIT_SCHEMA


class StallEscapeOracleCohortTests(unittest.TestCase):
    def _pair(self, root: Path, name: str, *, auc: float) -> tuple[Path, Path]:
        oracle_path = root / f"{name}-oracle.json"
        candidate_id = f"{name}-candidate"
        oracle = {
            "schema": STALL_ORACLE_SCHEMA,
            "before_fingerprint": f"{name}-full",
            "before_repair_fingerprint": f"{name}-repair",
            "classification": "selector_failure",
            "selected_v2_branch": {
                "candidate_rank": 1,
                "candidate_size": 16,
                "escape_fraction": 0.0,
                "stable_failure": True,
            },
            "stable_alternatives": [
                {"candidate_rank": 2, "candidate_size": 8}
            ],
            "branches": [
                {
                    "branch_mode": "explicit_neighborhood",
                    "candidate_id": candidate_id,
                    "trial_count": 4,
                }
            ],
        }
        oracle_path.write_text(json.dumps(oracle), encoding="utf-8")
        risk_root = root / f"{name}-risk"
        risk_root.mkdir()
        risk = {
            "schema": STALL_RISK_AUDIT_SCHEMA,
            "v3_manifest_sha256": "v3",
            "before_fingerprint": f"{name}-full",
            "before_repair_fingerprint": f"{name}-repair",
            "task_id": name,
            "solver_seed": 0,
            "decision_index": 3,
            "candidate_count": 1,
            "trial_count": 4,
            "candidates": [{"candidate_id": candidate_id}],
            "local_signal_supported": False,
            "trial_no_progress_auc": auc,
            "candidate_no_progress_spearman": 0.1,
            "selected_v2": {"predicted_risk_rank": 1},
            "best_stable_alternative": {"predicted_risk_rank": 1},
            "portable_native_maximum_delta": 0.0,
        }
        risk_path = risk_root / "stall_risk_audit_report.json"
        risk_path.write_text(json.dumps(risk), encoding="utf-8")
        (risk_root / "runner_config.json").write_text(
            json.dumps(
                {
                    "identity": {
                        "oracle_report_sha256": sha256_file(oracle_path),
                        "v3_manifest_sha256": "v3",
                    }
                }
            ),
            encoding="utf-8",
        )
        return oracle_path, risk_root

    def test_combines_bound_exact_states_without_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = self._pair(root, "first", auc=0.4)
            second = self._pair(root, "second", auc=0.6)
            report = build_stall_escape_oracle_cohort(
                [first[0], second[0]],
                [first[1], second[1]],
                root / "output",
            )
            self.assertEqual(report["state_count"], 2)
            self.assertEqual(report["selector_failure_count"], 2)
            self.assertEqual(report["paired_trial_count"], 8)
            self.assertAlmostEqual(report["mean_v3_trial_auc"], 0.5)
            self.assertEqual(
                report["decision"],
                "stall_specific_shadow_labels_required_frozen_v3_rejected",
            )
            self.assertFalse(report["deployment_promoted"])
            self.assertFalse(report["training_started"])

    def test_rejects_oracle_changed_after_risk_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oracle, risk = self._pair(root, "state", auc=0.5)
            oracle.write_text(oracle.read_text(encoding="utf-8") + "\n")
            with self.assertRaisesRegex(ValueError, "does not bind its Oracle"):
                build_stall_escape_oracle_cohort(
                    [oracle], [risk], root / "output"
                )


if __name__ == "__main__":
    unittest.main()
