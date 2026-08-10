from __future__ import annotations

import unittest

from experiments.stride_tailswitch_sequence_forensics import transition_metrics


class TailSwitchSequenceForensicsTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
