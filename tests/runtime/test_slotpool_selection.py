from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lns2_selector.runtime.slotpool_selection import (
    load_slotpool_model,
    reduce_slotpool_candidates,
)


ROOT = Path(__file__).resolve().parents[2]
MODEL = (
    ROOT
    / "build"
    / "stride-slotpool-v1-offline-evaluation"
    / "slotpool_pairwise_model.json"
)
MODEL_SHA256 = "e0a42c7a21cb341a6c5ce13ad6d89e6f5279c04378535fb9c44ce76a1dfe2319"
FRESH_COLLECTION = (
    ROOT / "build" / "stride-slotpool-fresh-confirmation-collection-v1"
)
FRESH_ANALYSIS = (
    ROOT / "build" / "stride-slotpool-fresh-confirmation-analysis-v1"
)


class SlotPoolSelectionTests(unittest.TestCase):
    def test_frozen_model_rejects_file_and_semantic_drift(self) -> None:
        if not MODEL.is_file():
            self.skipTest("local frozen SlotPool artifact is not retained")
        payload = load_slotpool_model(MODEL, expected_sha256=MODEL_SHA256)
        self.assertEqual(payload["implementation_id"], "stride-slotpool-v1")
        with self.assertRaisesRegex(ValueError, "SHA-256 changed"):
            load_slotpool_model(MODEL, expected_sha256="0" * 64)
        changed = json.loads(MODEL.read_text(encoding="utf-8"))
        changed["baseline"] = float(changed["baseline"]) + 1.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "semantic hash changed"):
                load_slotpool_model(path)

    def test_runtime_reproduces_all_fresh_confirmation_selections(self) -> None:
        states = FRESH_COLLECTION / "states"
        expected_path = FRESH_ANALYSIS / "state_evaluation.jsonl"
        if not MODEL.is_file() or not states.is_dir() or not expected_path.is_file():
            self.skipTest("local fresh SlotPool confirmation artifacts are not retained")
        model = load_slotpool_model(MODEL, expected_sha256=MODEL_SHA256)
        expected = {
            str(row["state_id"]): row
            for row in (
                json.loads(line)
                for line in expected_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        observed_state_ids = set()
        for path in sorted(states.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            aggregates = list(payload["candidate_aggregates"])
            anchor = dict(payload["v2_base_anchor"])
            state_id = str(aggregates[0]["state_id"])
            observed_state_ids.add(state_id)
            candidates = [
                {
                    "candidate_id": str(anchor["candidate_id"]),
                    "agents": list(anchor["agents"]),
                },
                *aggregates,
            ]
            rows = [
                {
                    "features": {
                        "realized_dynamic": dict(aggregates[0]["features"])
                    }
                },
                *[
                    {"features": {"realized_dynamic": dict(row["features"])}}
                    for row in aggregates
                ],
            ]
            result = reduce_slotpool_candidates(
                candidates=candidates,
                candidate_rows=rows,
                model_payload=model,
                v2_anchor_index=0,
                maximum_challengers=6,
            )
            self.assertEqual(
                result["selected_candidate_ids"],
                expected[state_id]["selected_candidate_ids"],
            )
            self.assertEqual(len(result["selected_structural_indices"]), 6)
        self.assertEqual(observed_state_ids, set(expected))
        self.assertEqual(len(observed_state_ids), 24)


if __name__ == "__main__":
    unittest.main()
