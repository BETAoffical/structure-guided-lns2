from __future__ import annotations

import unittest

from experiments.stride_tailswitch_sequence_forensics import (
    _pair_rows,
    original_pool_anchor,
    transition_metrics,
)


class TailSwitchSequenceForensicsTests(unittest.TestCase):
    def test_original_pool_anchor_excludes_structural_candidates(self) -> None:
        anchor = original_pool_anchor(
            [
                {
                    "candidate_id": "struct",
                    "score": 100.0,
                    "selection_families": ["structpool-hotspot:16"],
                },
                {
                    "candidate_id": "base-z",
                    "score": 2.0,
                    "selection_families": ["collision:16"],
                },
                {
                    "candidate_id": "base-a",
                    "score": 2.0,
                    "selection_families": ["target:16"],
                },
            ]
        )

        self.assertIsNotNone(anchor)
        self.assertEqual(anchor["candidate_id"], "base-a")

    def test_transition_metrics_measure_reuse_and_conflict_targeting(self) -> None:
        result = transition_metrics(
            {(0, 1), (1, 2), (3, 4)},
            {(1, 2), (3, 4), (4, 5)},
            {0, 1, 3},
            [{0, 2}, {1, 3}],
            selected_candidate_id="candidate-a",
            previous_candidate_ids=["candidate-z", "candidate-a"],
        )

        self.assertAlmostEqual(result["adjacent_agent_jaccard"], 2.0 / 3.0)
        self.assertEqual(result["recent_three_reuse_fraction"], 1.0)
        self.assertTrue(result["exact_candidate_repeat"])
        self.assertEqual(result["conflict_touch_coverage"], 1.0)
        self.assertAlmostEqual(result["conflict_full_coverage"], 1.0 / 3.0)
        self.assertAlmostEqual(result["unresolved_edge_fraction"], 2.0 / 3.0)
        self.assertAlmostEqual(result["new_edge_fraction"], 1.0 / 3.0)

    def test_transition_metrics_handle_empty_conflict_graph(self) -> None:
        result = transition_metrics(
            set(),
            set(),
            {2, 3},
            [],
            selected_candidate_id="candidate-a",
            previous_candidate_ids=[],
        )

        self.assertEqual(result["conflict_touch_coverage"], 0.0)
        self.assertEqual(result["unresolved_edge_fraction"], 0.0)
        self.assertEqual(result["largest_component_persistence"], 0.0)

    def test_pair_rows_use_tailswitch_struct_continuation_names(self) -> None:
        fields = {
            "structural_selection_count": 0,
            "structural_selection_rate": 0.0,
            "exact_candidate_repeat_rate": 0.0,
            "maximum_structural_run_length": 0,
            "mean_recent_three_reuse_fraction": 0.0,
            "mean_unresolved_edge_fraction": 0.0,
            "mean_conflict_touch_coverage": 0.0,
            "mean_normalized_conflict_progress": 0.0,
        }
        summaries = {
            ("state-a", policy): dict(fields)
            for policy in (
                "v2-then-v2",
                "struct-then-v2",
                "v2-then-struct",
                "struct-then-struct",
            )
        }
        state = {
            "state_id": "state-a",
            "map_id": "map-a",
            "task_id": "task-a",
            "solver_seed": 1,
            "challenger": "v2-plus-structpool",
            "contrasts": {
                name: {
                    "classification": "neutral",
                    "normalized_auc_delta": 0.0,
                    "final_conflict_delta": 0,
                }
                for name in (
                    "struct_continuation_after_v2",
                    "struct_continuation_after_struct",
                )
            },
        }

        rows = _pair_rows([state], summaries)

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            {row["contrast"] for row in rows},
            {"continuation_after_v2", "continuation_after_struct"},
        )


if __name__ == "__main__":
    unittest.main()
