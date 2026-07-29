from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from experiments._common import sha256_file
from experiments.stall_oracle import STALL_ORACLE_SCHEMA
from experiments.stall_risk_audit import (
    audit_frozen_v3_stall_risk,
    evaluate_stall_risk_predictions,
)
from experiments.stalled_state_probe import STALLED_STATE_PROBE_SCHEMA


def _branch(
    candidate_id: str,
    rank: int,
    *,
    escapes: int,
    trials: int = 8,
) -> dict:
    return {
        "branch_mode": "explicit_neighborhood",
        "candidate_id": candidate_id,
        "candidate_rank": rank,
        "candidate_size": 16 if rank == 1 else 8,
        "trial_count": trials,
        "escape_count": escapes,
        "escape_fraction": escapes / trials,
        "stable_escape": escapes == trials,
        "stable_failure": escapes <= trials // 4,
        "pp_order_sensitive": 0 < escapes < trials,
    }


def _prediction(candidate_id: str, risk: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "v2_score": 1.0,
        "predicted_no_progress_probability": risk,
        "predicted_effective_progress_probability": 1.0 - risk,
        "predicted_conflict_reduction": 1.0,
        "predicted_repair_seconds": 1.0,
        "predicted_utility": 1.0,
    }


class StallRiskAuditTests(unittest.TestCase):
    def test_audit_uses_manifest_compatibility_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            probe = root / "probe"
            source = root / "source"
            controller = root / "controller"
            output = root / "output"
            probe.mkdir()
            source.mkdir()
            controller.mkdir()
            runner_path = probe / "runner_config.json"
            runner_path.write_text("{}\n", encoding="utf-8")
            probe_report_path = probe / "stalled_state_probe_report.json"
            probe_report_path.write_text(
                json.dumps(
                    {
                        "schema": STALLED_STATE_PROBE_SCHEMA,
                        "candidate_pool_fingerprint": "pool",
                        "before_repair_fingerprint": "repair",
                        "source_collection": str(source),
                        "task_id": "task",
                        "solver_seed": 7,
                    }
                ),
                encoding="utf-8",
            )
            oracle_path = root / "oracle.json"
            oracle_path.write_text(
                json.dumps(
                    {
                        "schema": STALL_ORACLE_SCHEMA,
                        "source_probe_report_sha256": sha256_file(
                            probe_report_path
                        ),
                        "source_probe_runner_sha256": sha256_file(runner_path),
                    }
                ),
                encoding="utf-8",
            )
            (controller / "v3_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (source / "run_config.json").write_text(
                json.dumps(
                    {
                        "configuration": {},
                        "run_fingerprint": "run",
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "task_id": "task",
                "solver_seed": 7,
                "status": "ok",
                "episode_id": "episode",
            }
            with (
                mock.patch(
                    "experiments.stall_risk_audit.load_v3_controller_bundle",
                    return_value=object(),
                ),
                mock.patch(
                    "experiments.stall_risk_audit.prepare_run_output",
                    return_value={"identity_fingerprint": "audit"},
                ),
                mock.patch(
                    "experiments.stall_risk_audit._read_jsonl",
                    return_value=[manifest],
                ),
                mock.patch(
                    "experiments.stall_risk_audit.validate_manifest_trace"
                ) as validate,
                mock.patch(
                    "experiments.stall_risk_audit.decision_rows",
                    side_effect=RuntimeError("stop after validation"),
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "stop after validation"
                ):
                    audit_frozen_v3_stall_risk(
                        probe, oracle_path, controller, output
                    )
            validate.assert_called_once_with(
                source,
                manifest,
                run_fingerprint="run",
                expected_policy="realized_dynamic",
            )

    def test_local_signal_requires_low_risk_stable_alternative(self) -> None:
        oracle = {
            "selector_failure": True,
            "branches": [
                _branch("winner", 1, escapes=0),
                _branch("backup", 2, escapes=8),
                _branch("mixed", 3, escapes=4),
            ],
        }
        report = evaluate_stall_risk_predictions(
            [
                _prediction("winner", 0.9),
                _prediction("backup", 0.1),
                _prediction("mixed", 0.5),
            ],
            oracle,
        )
        self.assertTrue(report["local_signal_supported"])
        self.assertEqual(report["best_stable_alternative"]["candidate_id"], "backup")
        self.assertEqual(report["selected_v2"]["predicted_risk_rank"], 3)
        self.assertGreater(report["trial_no_progress_auc"], 0.9)
        self.assertFalse(report["deployment_promoted"])
        self.assertFalse(report["training_started"])

    def test_inverted_risk_is_not_supported(self) -> None:
        oracle = {
            "selector_failure": True,
            "branches": [
                _branch("winner", 1, escapes=0),
                _branch("backup", 2, escapes=8),
            ],
        }
        report = evaluate_stall_risk_predictions(
            [_prediction("winner", 0.1), _prediction("backup", 0.9)],
            oracle,
        )
        self.assertFalse(report["local_signal_supported"])
        self.assertEqual(
            report["decision"], "frozen_v3_stall_risk_signal_not_supported"
        )

    def test_prediction_must_cover_the_exact_oracle_pool(self) -> None:
        oracle = {
            "selector_failure": True,
            "branches": [
                _branch("winner", 1, escapes=0),
                _branch("backup", 2, escapes=8),
            ],
        }
        with self.assertRaisesRegex(ValueError, "cover the Oracle pool"):
            evaluate_stall_risk_predictions([_prediction("winner", 0.9)], oracle)

    def test_probability_must_be_finite_and_bounded(self) -> None:
        oracle = {
            "selector_failure": True,
            "branches": [_branch("winner", 1, escapes=0)],
        }
        with self.assertRaisesRegex(ValueError, "finite probability"):
            evaluate_stall_risk_predictions(
                [_prediction("winner", float("nan"))], oracle
            )


if __name__ == "__main__":
    unittest.main()
