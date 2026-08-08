from __future__ import annotations

import copy
import unittest
from pathlib import Path

from experiments.stride_safeslot_first_divergence import (
    first_divergence,
    load_safeslot_first_divergence_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "stride_safeslot_first_divergence_v1.json"


def _decision(
    index: int,
    before: int,
    after: int,
    candidate: str,
    seed: int,
    *,
    structural: bool = False,
    anchor: str | None = None,
) -> dict[str, object]:
    return {
        "decision_index": index,
        "before_conflicts": before,
        "after_conflicts": after,
        "conflict_reduction": before - after,
        "before_fingerprint": f"before-{index}",
        "after_fingerprint": f"after-{index}-{candidate}",
        "pp_random_seed": seed,
        "selected_candidate_id": candidate,
        "selected_score": 1.0,
        "selected_size": 8,
        "selected_agents": [0, 1],
        "selected_families": ["collision:8"],
        "selected_structural": structural,
        "selected_structural_groups": ["bottleneck_crossing"] if structural else [],
        "base_candidate_ids": [anchor or candidate],
        "mixed_pool_top_base_candidate_id": anchor or candidate,
        "v2_anchor_candidate_id": anchor or candidate,
    }


def _trace(decisions: list[dict[str, object]]) -> dict[str, object]:
    trajectory = [int(decisions[0]["before_conflicts"])]
    trajectory.extend(int(row["after_conflicts"]) for row in decisions)
    return {
        "initial_fingerprint": "initial",
        "initial_conflicts": trajectory[0],
        "repair_iterations": len(decisions),
        "conflict_trajectory": trajectory,
        "decisions": decisions,
    }


class SafeSlotFirstDivergenceTests(unittest.TestCase):
    def test_registered_config_keeps_exact_29_key_scope(self) -> None:
        _path, _root, config, expected = load_safeslot_first_divergence_config(
            CONFIG
        )
        self.assertEqual(len(expected), 29)
        self.assertTrue(config["claim_boundary"]["retrospective_existing_outcomes"])
        self.assertFalse(config["claim_boundary"]["model_training_allowed"])
        self.assertFalse(
            config["claim_boundary"]["cross_run_v2_timing_comparison_allowed"]
        )

    def test_first_divergence_detects_immediate_win_terminal_loss(self) -> None:
        shared = _decision(0, 10, 7, "base-0", 11)
        v2 = _trace(
            [
                shared,
                _decision(1, 7, 5, "base-1", 12),
                _decision(2, 5, 0, "base-2", 13),
            ]
        )
        challenger_shared = copy.deepcopy(shared)
        challenger = _trace(
            [
                challenger_shared,
                _decision(
                    1,
                    7,
                    4,
                    "struct-1",
                    21,
                    structural=True,
                    anchor="base-1",
                ),
                _decision(2, 4, 4, "base-3", 22),
                _decision(3, 4, 3, "base-4", 23),
            ]
        )
        result = first_divergence(v2, challenger, short_window=2)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result["decision_index"], 1)
        self.assertEqual(result["immediate_conflict_reduction_advantage"], 1)
        self.assertEqual(result["terminal_repair_iterations_delta"], 1)
        self.assertTrue(result["immediate_better_but_terminal_worse"])
        self.assertTrue(result["mixed_pool_top_base_matches_v2"])

    def test_identical_trace_has_no_divergence(self) -> None:
        trace = _trace([_decision(0, 2, 0, "base-0", 11)])
        self.assertIsNone(
            first_divergence(trace, copy.deepcopy(trace), short_window=8)
        )

    def test_anchor_mismatch_is_rejected(self) -> None:
        v2 = _trace([_decision(0, 3, 0, "base-0", 11)])
        challenger = _trace(
            [
                _decision(
                    0,
                    3,
                    0,
                    "struct-0",
                    12,
                    structural=True,
                    anchor="wrong-base",
                )
            ]
        )
        challenger["decisions"][0]["base_candidate_ids"].append("base-0")
        with self.assertRaisesRegex(ValueError, "anchor differs"):
            first_divergence(v2, challenger, short_window=8)


if __name__ == "__main__":
    unittest.main()
