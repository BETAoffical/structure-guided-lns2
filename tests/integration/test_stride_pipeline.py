from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.stride_lns import (
    STRIDE_STAGE1_CONFIG_SCHEMA,
    run_stage1_audit,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class StrideStage1AuditTest(unittest.TestCase):
    def _bundle(self, root: Path, controller_id: str) -> None:
        payload = {
            "feature_schema_id": "lns2.realized_features.v2",
            "feature_schema_sha256": "feature-sha",
            "feature_dimensions": {"realized_dynamic": 124},
        }
        if controller_id == "v2-full":
            payload["default_controller"] = controller_id
        else:
            payload["controller_id"] = controller_id
        _write_json(root / "controller_manifest.json", payload)

    def test_audit_freezes_bundles_and_rejects_aggregated_old_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root / "v2", "v2-full")
            self._bundle(root / "mixed", "mixed-full-v2")
            _write_jsonl(
                root / "old.jsonl",
                [
                    {
                        "state_id": "state-1",
                        "map_id": "map-1",
                        "split": "train",
                        "agent_count": 80,
                        "trial_count": 4,
                        "features": {"realized_dynamic": {str(i): 0 for i in range(124)}},
                        "outcome": {"conflicts_after": 2.0},
                    }
                ],
            )
            config = {
                "schema": STRIDE_STAGE1_CONFIG_SCHEMA,
                "research_line": "stride-lns",
                "frozen_bundles": [
                    {"controller_id": "v2-full", "path": "v2"},
                    {"controller_id": "mixed-full-v2", "path": "mixed"},
                ],
                "historical_sources": [
                    {
                        "name": "old",
                        "role": "historical",
                        "kind": "ranking_index",
                        "path": "old.jsonl",
                    }
                ],
            }
            _write_json(root / "config.json", config)

            report = run_stage1_audit(
                config_path=root / "config.json",
                output=root / "output",
                project_root=root,
            )

            self.assertTrue(report["stage1_passed"])
            self.assertTrue(report["requires_fresh_stride_collection"])
            source = report["historical_sources"][0]
            self.assertFalse(source["reusable_for_stride_labels"])
            self.assertEqual(source["feature_dimensions"], {"124": 1})

    def test_sequence_trials_require_four_pp_seeds_and_post_structure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root / "v2", "v2-full")
            self._bundle(root / "mixed", "mixed-full-v2")
            rows = []
            for trial_index, seed in enumerate((11, 12, 13, 14)):
                rows.append(
                    {
                        "state_id": "state-1",
                        "map_id": "map-1",
                        "split": "dev",
                        "trial_index": trial_index,
                        "steps": [
                            {
                                "executed": True,
                                "candidate_id": "candidate-1",
                                "action": {"pp_random_seed": seed},
                                "post_structure": {
                                    "post_largest_component_ratio": 0.1,
                                    "post_conflict_edge_density": 0.2,
                                    "post_event_density": 0.3,
                                    "post_degree_concentration": 0.4,
                                },
                            }
                        ],
                    }
                )
            _write_jsonl(root / "trials.jsonl", rows)
            config = {
                "schema": STRIDE_STAGE1_CONFIG_SCHEMA,
                "research_line": "stride-lns",
                "frozen_bundles": [
                    {"controller_id": "v2-full", "path": "v2"},
                    {"controller_id": "mixed-full-v2", "path": "mixed"},
                ],
                "historical_sources": [
                    {
                        "name": "trials",
                        "role": "historical",
                        "kind": "sequence_trials",
                        "path": "trials.jsonl",
                    }
                ],
            }
            _write_json(root / "config.json", config)

            report = run_stage1_audit(
                config_path=root / "config.json",
                output=root / "output",
                project_root=root,
            )

            source = report["historical_sources"][0]
            self.assertTrue(source["has_individual_paired_trials"])
            self.assertTrue(source["has_post_state_structure"])
            self.assertTrue(source["reusable_for_stride_labels"])
            self.assertFalse(report["requires_fresh_stride_collection"])

    def test_non_historical_map_overlap_fails_stage1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._bundle(root / "v2", "v2-full")
            self._bundle(root / "mixed", "mixed-full-v2")
            row = {
                "state_id": "state-1",
                "map_id": "leaked-map",
                "split": "x",
                "agent_count": 80,
                "trial_count": 4,
                "features": {"realized_dynamic": {}},
            }
            _write_jsonl(root / "dev.jsonl", [row])
            _write_jsonl(root / "formal.jsonl", [row])
            config = {
                "schema": STRIDE_STAGE1_CONFIG_SCHEMA,
                "research_line": "stride-lns",
                "frozen_bundles": [
                    {"controller_id": "v2-full", "path": "v2"},
                    {"controller_id": "mixed-full-v2", "path": "mixed"},
                ],
                "historical_sources": [
                    {"name": "dev", "role": "dev", "kind": "ranking_index", "path": "dev.jsonl"},
                    {"name": "formal", "role": "formal", "kind": "ranking_index", "path": "formal.jsonl"},
                ],
            }
            _write_json(root / "config.json", config)

            report = run_stage1_audit(
                config_path=root / "config.json",
                output=root / "output",
                project_root=root,
            )

            self.assertFalse(report["stage1_passed"])
            self.assertEqual(report["map_leakage"][0]["maps"], ["leaked-map"])


if __name__ == "__main__":
    unittest.main()
