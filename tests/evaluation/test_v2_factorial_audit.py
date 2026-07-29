from __future__ import annotations

import unittest

from experiments.v2_factorial_audit import (
    _cell_gate,
    compare_metrics,
    effectiveness_dominates,
    equal_state_weights,
    pair_vector,
    quality_checks,
)


class V2FactorialAuditTests(unittest.TestCase):
    def test_effectiveness_dominance_excludes_runtime(self) -> None:
        winner = {"solved_rate": 0.5, "conflicts_after": 4.0, "repair_seconds": 9.0}
        loser = {"solved_rate": 0.0, "conflicts_after": 4.0, "repair_seconds": 1.0}
        self.assertTrue(effectiveness_dominates(winner, loser))
        self.assertFalse(effectiveness_dominates(loser, winner))

    def test_tradeoff_is_not_labeled_as_dominance(self) -> None:
        solved = {"solved_rate": 1.0, "conflicts_after": 3.0}
        lower_conflicts = {"solved_rate": 0.0, "conflicts_after": 2.0}
        self.assertFalse(effectiveness_dominates(solved, lower_conflicts))
        self.assertFalse(effectiveness_dominates(lower_conflicts, solved))

    def test_pair_vector_separates_delta_and_shared_inputs(self) -> None:
        self.assertEqual(
            pair_vector(
                {"candidate": 4.0, "state": 10.0},
                {"candidate": 1.0, "state": 10.0},
                (("delta", "candidate"), ("shared", "state")),
            ),
            [3.0, 10.0],
        )

    def test_equal_state_weights_give_each_state_equal_total_mass(self) -> None:
        rows = [
            {"state_id": "a"},
            {"state_id": "a"},
            {"state_id": "b"},
        ]
        weights = equal_state_weights(rows)
        self.assertAlmostEqual(weights[0] + weights[1], weights[2])
        self.assertAlmostEqual(sum(weights), 3.0)

    def test_quality_gate_does_not_promote_efficiency_alone(self) -> None:
        selected = {
            "effective_rate": 0.90,
            "no_progress_rate": 0.10,
            "mean_conflict_reduction": 9.0,
            "mean_repair_seconds": 0.1,
            "conflict_reduction_per_repair_second": 90.0,
        }
        frozen = {
            "effective_rate": 0.92,
            "no_progress_rate": 0.08,
            "mean_conflict_reduction": 10.0,
            "mean_repair_seconds": 1.0,
            "conflict_reduction_per_repair_second": 10.0,
        }
        comparison = compare_metrics(selected, frozen)
        self.assertGreater(comparison["efficiency_ratio"], 1.0)
        self.assertFalse(all(quality_checks(comparison).values()))

    def test_cell_gate_requires_five_of_six_cells(self) -> None:
        self.assertFalse(
            _cell_gate(
                {"cell_count": 6, "quality_noninferior_cell_count": 4}
            )
        )
        self.assertTrue(
            _cell_gate(
                {"cell_count": 6, "quality_noninferior_cell_count": 5}
            )
        )


if __name__ == "__main__":
    unittest.main()
