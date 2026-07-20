from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.verify_closed_loop_equivalence import compare_collections


def _write_collection(root: Path, *, conflicts_after: int = 0) -> None:
    for name in ("official_adaptive", "proposal_dynamic", "realized_dynamic"):
        episode_id = f"task__seed_0000__{name}"
        trace = Path("episodes") / name / f"{episode_id}.jsonl"
        trace_path = root / trace
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        events = [
            {"event": "initial"},
            {
                "event": "transition",
                "decision_index": 0,
                "before_fingerprint": "before",
                "after_fingerprint": "after",
                "action": {"mode": "official"},
                "low_level_delta": {"generated": 1},
                "terminated": True,
                "truncated": False,
                "metrics": {
                    "action_valid": True,
                    "conflicts_before": 1,
                    "conflicts_after": conflicts_after,
                    "conflict_delta": 1 - conflicts_after,
                    "iteration": 1,
                    "neighborhood": [0, 1],
                    "replan_success": True,
                    "requested_mode": "official",
                    "requested_random_seed": -1,
                    "sum_of_costs_before": 10,
                    "sum_of_costs_after": 10,
                },
                "controller": {"controller_seconds_before_repair": 0.1},
            },
            {"event": "finish"},
        ]
        trace_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in events),
            encoding="utf-8",
        )
        summary = {
            "success": conflicts_after == 0,
            "repairable": True,
            "truncated": False,
            "external_timeout": False,
            "initial_fingerprint": "before",
            "initial_conflicts": 1,
            "final_conflicts": conflicts_after,
            "conflict_trajectory": [1, conflicts_after],
            "conflict_auc": float(conflicts_after),
            "fixed_budget_conflict_auc": float(conflicts_after),
            "repair_iterations": 1,
            "final_low_level": {"generated": 1},
            "final_sum_of_costs": 10,
            "invalid_action_count": 0,
            "fingerprint_mismatch_count": 0,
            "wall_time_to_feasible": 0.2,
        }
        manifest = {
            "episode_id": episode_id,
            "trace_file": trace.as_posix(),
            "summary": summary,
        }
        (root / f"{name}_manifest.jsonl").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )


class ClosedLoopEquivalenceTests(unittest.TestCase):
    def test_timing_is_excluded_but_scientific_changes_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            candidate = root / "candidate"
            _write_collection(reference)
            _write_collection(candidate)
            report = compare_collections(reference, candidate)
            self.assertTrue(report["exact"])

            _write_collection(candidate, conflicts_after=1)
            report = compare_collections(reference, candidate)
            self.assertFalse(report["exact"])
            self.assertEqual(report["mismatch_count"], 3)


if __name__ == "__main__":
    unittest.main()
