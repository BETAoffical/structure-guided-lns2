from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from experiments.result_consolidation import (
    EvidenceVerificationError,
    json_path,
    run_result_consolidation,
    validate_config,
    verify_build_sources,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "result_consolidation.json"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_config(root: Path) -> dict:
    context_path = root / "context.json"
    movingai_path = root / "movingai.json"
    context_path.write_text(
        json.dumps({"acceptance": {"passed": False}, "gain": 0.04}),
        encoding="utf-8",
    )
    movingai_path.write_text(
        json.dumps(
            {
                "acceptance": {
                    "passed": False,
                    "decision": "stop_cross_layout_claim_and_consolidate_results",
                },
                "auc": 0.041,
            }
        ),
        encoding="utf-8",
    )
    parity = "a" * 64
    (root / "CMakeLists.txt").write_text(parity, encoding="utf-8")
    common = {
        "title_zh": "测试",
        "category": "test",
        "data_boundary": "test",
        "document": "docs/test.md",
        "registration": {"status": "same_commit", "result_commit": None},
    }
    return {
        "schema": "lns2.result_consolidation.v1",
        "schema_version": 1,
        "snapshot_date": "2026-07-17",
        "active_claim_zh": "test",
        "report_figure_prefix": "figures",
        "research_boundary": {
            "static_context_increment_confirmed": False,
            "same_family_generalization_confirmed": True,
            "movingai_cross_layout_confirmed": False,
            "rl_trained": False,
        },
        "solver_baseline": {
            "upstream": "test",
            "upstream_commit": "test",
            "parity_source": "CMakeLists.txt",
            "expected_path_sha256": parity,
        },
        "related_work": [],
        "experiments": [
            {
                **common,
                "id": "context_primary_audit",
                "status": "unsupported",
                "decision": "fail_context_gate",
                "claim": "not confirmed",
                "source": {"path": "context.json", "sha256": _digest(context_path)},
                "checks": [
                    {"json_path": "acceptance.passed", "expected": False}
                ],
                "metrics": [
                    {
                        "id": "gain",
                        "label_zh": "gain",
                        "json_path": "gain",
                        "value": 0.04,
                        "unit": "ratio",
                        "role": "gate",
                    }
                ],
            },
            {
                **common,
                "id": "movingai_ood_closed_loop",
                "status": "external_support",
                "decision": "stop_cross_layout_claim_and_consolidate_results",
                "claim": "near threshold",
                "source": {"path": "movingai.json", "sha256": _digest(movingai_path)},
                "checks": [
                    {"json_path": "acceptance.passed", "expected": False},
                    {
                        "json_path": "acceptance.decision",
                        "expected": "stop_cross_layout_claim_and_consolidate_results",
                    },
                ],
                "metrics": [
                    {
                        "id": "auc_improvement",
                        "label_zh": "AUC",
                        "json_path": "auc",
                        "value": 0.041,
                        "unit": "ratio",
                        "role": "gate",
                    }
                ],
            },
        ],
    }


class ResultConsolidationTests(unittest.TestCase):
    def test_json_path_supports_objects_and_lists(self) -> None:
        value = {"a": {"b": [{"c": 3}]}}
        self.assertEqual(json_path(value, "a.b.0.c"), 3)
        with self.assertRaises(KeyError):
            json_path(value, "a.missing")

    def test_canonical_config_preserves_claim_boundaries(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        validate_config(config)
        by_id = {row["id"]: row for row in config["experiments"]}
        self.assertEqual(len(by_id), 24)
        self.assertEqual(by_id["movingai_ood_closed_loop"]["status"], "external_support")
        self.assertEqual(by_id["context_primary_audit"]["status"], "unsupported")
        self.assertFalse(config["research_boundary"]["rl_trained"])
        for identifier in (
            "natural_distribution_confirmation",
            "closed_loop_confirmation",
            "closed_loop_multiseed",
            "movingai_ood_closed_loop",
        ):
            checks = {
                row["json_path"]: row["expected"]
                for row in by_id[identifier]["checks"]
            }
            self.assertTrue(checks["qualification.seed_isolation.passed"])
            self.assertFalse(checks["frozen_models.confirmation_labels_seen"])

    def test_claim_inflation_is_rejected(self) -> None:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        changed = copy.deepcopy(config)
        changed["research_boundary"]["movingai_cross_layout_confirmed"] = True
        with self.assertRaisesRegex(EvidenceVerificationError, "cannot confirm MovingAI"):
            validate_config(changed)
        changed = copy.deepcopy(config)
        for experiment in changed["experiments"]:
            if experiment["id"] == "movingai_ood_closed_loop":
                experiment["metrics"] = [
                    {
                        **metric,
                        "value": 0.05,
                    }
                    if metric["id"] == "auc_improvement"
                    else metric
                    for metric in experiment["metrics"]
                ]
        with self.assertRaisesRegex(EvidenceVerificationError, "below the 5% gate"):
            validate_config(changed)

    def test_strict_verification_rejects_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _minimal_config(root)
            validate_config(config)
            report = verify_build_sources(config, root)
            self.assertEqual(report["status"], "passed")
            (root / "movingai.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceVerificationError, "SHA256 mismatch"):
                verify_build_sources(config, root)

    def test_relative_source_paths_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = _minimal_config(Path(temporary))
            config["experiments"][0]["source"]["path"] = str(
                Path(temporary).resolve() / "context.json"
            )
            with self.assertRaisesRegex(EvidenceVerificationError, "repository-relative"):
                validate_config(config)

    def test_snapshot_generation_is_deterministic_without_build_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            first_report = root / "first.md"
            second_report = root / "second.md"
            one = run_result_consolidation(
                CONFIG_PATH,
                first,
                first_report,
                repository_root=PROJECT_ROOT,
                verify_build=False,
            )
            two = run_result_consolidation(
                CONFIG_PATH,
                second,
                second_report,
                repository_root=PROJECT_ROOT,
                verify_build=False,
            )
            self.assertEqual(one["experiment_count"], 24)
            self.assertEqual(two["verification"], "not_requested")
            for relative in (
                "evidence_manifest.json",
                "metrics.csv",
                "verification.json",
                "figures/offline_evidence.svg",
                "figures/closed_loop_evidence.svg",
                "figures/movingai_ood.svg",
                "figures/audit_outcomes.svg",
            ):
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes())
            self.assertEqual(first_report.read_bytes(), second_report.read_bytes())
            manifest_text = (first / "evidence_manifest.json").read_text(encoding="utf-8")
            self.assertNotIn(str(PROJECT_ROOT), manifest_text)
            report_text = first_report.read_text(encoding="utf-8")
            self.assertIn(
                "https://github.com/BETAoffical/structure-guided-lns2/blob/"
                "pre-minimal-runtime-2026-07-20/research/docs/context/CONTEXT_AUDIT.md",
                report_text,
            )
            for heading in (
                "## 1. 问题定义",
                "## 5. 方法",
                "## 8. 迁移边界",
                "## 10. 负面结果与停止规则",
                "## 附录 A：冻结证据登记",
            ):
                self.assertIn(heading, report_text)
            for svg in (first / "figures").glob("*.svg"):
                ET.parse(svg)


if __name__ == "__main__":
    unittest.main()
