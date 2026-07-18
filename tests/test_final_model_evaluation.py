from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.final_model_evaluation import (
    BASELINES,
    POLICY_ORDER,
    _aggregate_rows,
    _bar_chart,
    _controller_performance_evidence,
    _paired_rows,
    _render_markdown,
)
from experiments.closed_loop_trace_storage import (
    TRACE_FORMAT_DELTA_GZIP_V2,
    storage_fingerprint,
)
from scripts.run_final_model_evaluation import _require_quick_audit, _require_storage_audit
from scripts.verify_closed_loop_equivalence import equivalence_comparison_fingerprint


def _episode(policy: str, auc: float, success: bool = True) -> dict:
    return {
        "episode_id": f"task-a__seed-1__{policy}",
        "policy": policy,
        "layout_family": "maze",
        "map_id": "map-a",
        "task_id": "task-a",
        "agent_count": 100,
        "solver_seed": 1,
        "status": "ok",
        "success": success,
        "repairable": True,
        "initial_conflicts": 10,
        "final_conflicts": 0 if success else 2,
        "fixed_budget_conflict_auc": auc,
        "raw_conflict_auc": auc,
        "repair_iterations": 2,
        "capped_wall_time_seconds": auc / 10.0,
        "wall_time_to_feasible_seconds": auc / 10.0 if success else None,
        "final_sum_of_costs": 100,
        "trace_format": "delta-gzip-v2",
        "trace_bytes": 1000,
    }


class FinalModelEvaluationTests(unittest.TestCase):
    def test_aggregates_and_pairing_cover_all_registered_policies(self) -> None:
        rows = [
            _episode(policy, 80.0 if policy == "realized_dynamic" else 100.0)
            for policy in POLICY_ORDER
        ]
        aggregates = _aggregate_rows(rows)
        overall = [row for row in aggregates if row["layout_family"] == "all"]
        self.assertEqual({row["policy"] for row in overall}, set(POLICY_ORDER))
        paired = _paired_rows(rows)
        self.assertEqual([row["baseline"] for row in paired], list(BASELINES))
        self.assertTrue(all(row["auc_relative_improvement"] == 0.2 for row in paired))

    def test_controller_performance_evidence_is_sha_verified_and_exported(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        evidence = _controller_performance_evidence(
            {
                "controller": "v2-full",
                "configuration": {
                    "controller_bundle": str(
                        project_root
                        / "artifacts"
                        / "initlns-closed-loop-controller-v2"
                    )
                },
            },
            project_root,
        )
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertTrue(evidence["performance_gate_passed"])
        self.assertGreater(float(evidence["feature_speedup"]), 1.0)
        rows = [
            _episode(policy, 80.0 if policy == "realized_dynamic" else 100.0)
            for policy in POLICY_ORDER
        ]
        aggregate = _aggregate_rows(rows, evidence)
        primary = next(
            row
            for row in aggregate
            if row["policy"] == "realized_dynamic"
            and row["layout_family"] == "all"
        )
        self.assertEqual(
            primary["fixed_suite_feature_speedup"], evidence["feature_speedup"]
        )

    def test_svg_chart_supports_negative_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chart.svg"
            _bar_chart(
                path,
                "Signed values",
                ["positive", "negative"],
                [0.2, -0.1],
                percent=True,
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("<svg", text)
            self.assertIn('role="img"', text)
            self.assertIn("<title", text)
            self.assertIn("<desc", text)
            self.assertNotIn('height="-', text)
            self.assertIn("-10.0%", text)

    def test_quick_markdown_marks_nonformal_and_handles_no_repairs(self) -> None:
        aggregates = []
        for policy in POLICY_ORDER:
            aggregates.append(
                {
                    "policy": policy,
                    "layout_family": "all",
                    "episode_count": 1,
                    "success_count": 1,
                    "success_rate": 1.0,
                    "mean_fixed_budget_conflict_auc": None,
                    "mean_capped_wall_time_seconds": None,
                }
            )
        markdown = _render_markdown(
            {
                "trace_format": "delta-gzip-v2",
                "valid_trace_count": 5,
                "expected_trace_count": 5,
                "aggregates": aggregates,
                "paired_comparisons": [],
            },
            formal=False,
            official_report=None,
        )
        self.assertIn("非正式试跑，不可作为正式结论", markdown)
        self.assertIn("| n/a | n/a |", markdown)

    def test_formal_storage_audit_requires_current_five_policy_v2_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            compact = Path(directory) / "compact"
            compact.mkdir()
            run_config = {
                "trace_format": TRACE_FORMAT_DELTA_GZIP_V2,
                "storage_fingerprint": storage_fingerprint(
                    TRACE_FORMAT_DELTA_GZIP_V2
                ),
            }
            (compact / "run_config.json").write_text(
                json.dumps(run_config), encoding="utf-8"
            )
            audit = {
                "passed": True,
                "exact": True,
                "storage_target_passed": True,
                "comparison_fingerprint": equivalence_comparison_fingerprint(),
                "policies": {
                    policy: {
                        "episode_count": 144,
                        "matching_episode_count": 144,
                    }
                    for policy in POLICY_ORDER
                },
            }
            audit_path = compact / "equivalence_report.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            self.assertTrue(_require_storage_audit(audit_path)["passed"])

            run_config["storage_fingerprint"] = "wrong"
            (compact / "run_config.json").write_text(
                json.dumps(run_config), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                _require_storage_audit(audit_path)

    def test_formal_gate_requires_every_quick_decision_to_be_shadow_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "mode": "quick",
                        "valid_trace_count": 75,
                    }
                ),
                encoding="utf-8",
            )
            (root / "run_config.json").write_text(
                json.dumps(
                    {
                        "controller": "v2-full",
                        "feature_backend": "auto",
                        "configuration": {"feature_shadow_validation": True},
                    }
                ),
                encoding="utf-8",
            )
            rows = [
                {
                    "status": "ok",
                    "summary": {
                        "controller_totals": {
                            "learned_decisions": 2,
                            "shadow_validation_count": 2,
                            "shadow_score_max_delta": 0.0,
                        }
                    },
                }
                for _ in range(15)
            ]
            manifest = root / "realized_dynamic_manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            self.assertEqual(
                _require_quick_audit(root / "status.json", "v2-full", "auto")[
                    "status"
                ],
                "complete",
            )
            rows[0]["summary"]["controller_totals"]["shadow_validation_count"] = 1
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _require_quick_audit(root / "status.json", "v2-full", "auto")


if __name__ == "__main__":
    unittest.main()
