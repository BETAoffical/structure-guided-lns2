from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import _write_json, _write_jsonl
from experiments.stride_lns import FROZEN_FEATURE_SCHEMA_ID, STRIDE_TRIAL_SCHEMA
from experiments.stride_repairability_audit import audit_repairability_collection
from experiments.stride_repairability_collection import (
    COLLECTION_SCHEMA,
    STATE_SCHEMA,
    repairability_pp_seed,
)
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT


class StrideRepairabilityAuditTest(unittest.TestCase):
    def test_complete_collection_audit_checks_paired_product_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            collection = root / "collection"
            output = root / "audit"
            state_id = "state-a"
            repair_fingerprint = "repair-a"
            features = {
                name: 0.0
                for name in PROFILE_FEATURE_NAMES["realized_dynamic"]
            }
            candidates = [
                {
                    "candidate_id": "base-a",
                    "candidate_kind": "base",
                    "actual_size": 4,
                    "agents": [0, 1, 2, 3],
                    "selection_families": ["target:4", "collision:4", "random:4"],
                },
                {
                    "candidate_id": "boundary-a",
                    "candidate_kind": "boundary_only",
                    "actual_size": 16,
                    "agents": list(range(16)),
                    "selection_families": ["topology-boundary-core"],
                },
            ]
            trials = []
            for candidate in candidates:
                for index in range(16):
                    trials.append(
                        {
                            "schema": STRIDE_TRIAL_SCHEMA,
                            "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
                            "state_id": state_id,
                            "candidate_id": candidate["candidate_id"],
                            "trial_index": index,
                            "pp_seed": repairability_pp_seed(
                                repair_fingerprint, index
                            ),
                            "features": features,
                        }
                    )
            selection = {
                "state_id": state_id,
                "map_id": "map-a",
                "research_split": "train",
                "before_fingerprint": "full-a",
                "before_conflicts": 2,
            }
            run_fingerprint = "run-a"
            _write_json(
                collection / "run_config.json",
                {
                    "schema": COLLECTION_SCHEMA,
                    "run_fingerprint": run_fingerprint,
                    "selected_state_ids": [state_id],
                    "pp_trial_indices": list(range(16)),
                    "target_state_restore_contract": TARGET_STATE_RESTORE_CONTRACT,
                },
            )
            _write_jsonl(collection / "state_selection.jsonl", [selection])
            complete = {
                "status": "complete",
                "complete": True,
                "error_state_count": 0,
            }
            _write_json(collection / "collection_status.json", complete)
            _write_json(collection / "collection_report.json", complete)
            _write_jsonl(collection / "repair_trials.jsonl", trials)
            _write_json(
                collection / "states" / "state-a.json",
                {
                    "schema": STATE_SCHEMA,
                    "run_fingerprint": run_fingerprint,
                    "complete": True,
                    "state_id": state_id,
                    "decision": selection,
                    "before_fingerprint": "full-a",
                    "before_repair_fingerprint": repair_fingerprint,
                    "before_conflicts": 2,
                    "state_restore": {
                        "contract": TARGET_STATE_RESTORE_CONTRACT,
                        "source_full_fingerprint": "full-a",
                        "repair_structure_fingerprint": repair_fingerprint,
                    },
                    "base_candidate_count": 1,
                    "boundary_candidate_count": 1,
                    "candidates": candidates,
                    "trials": trials,
                },
            )
            report = audit_repairability_collection(
                collection=collection,
                output=output,
                expected_state_count=1,
            )
            self.assertTrue(report["passed"], report["errors"])
            self.assertEqual(report["trial_count"], 32)
            self.assertEqual(report["candidate_count"], 2)
            self.assertEqual(report["map_count_by_split"], {"train": 1, "validation": 0})
            self.assertEqual(len(report["sha256"]["repair_trials"]), 64)


if __name__ == "__main__":
    unittest.main()
