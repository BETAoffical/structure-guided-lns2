from __future__ import annotations

import unittest

from experiments.stride_repairdependency_predictability import (
    PREDICTORS,
    PRIMARY_PREDICTOR,
    _pareto_frontier,
    _scientific_row,
)


class RepairDependencyPredictabilityTests(unittest.TestCase):
    def test_predictor_order_and_primary_are_frozen(self) -> None:
        self.assertEqual(
            PREDICTORS,
            (
                "direct_conflict_boundary",
                "spatial_low_degree_frontier",
                "temporal_corridor_frontier",
                "repair_dependency_frontier",
                "full_temporal_boundary_reference",
            ),
        )
        self.assertEqual(PRIMARY_PREDICTOR, "repair_dependency_frontier")

    def test_pareto_frontier_is_unweighted_and_deterministic(self) -> None:
        rows = {
            1: {"a": 1, "b": 1},
            2: {"a": 2, "b": 1},
            3: {"a": 1, "b": 3},
            4: {"a": 0, "b": 0},
        }
        self.assertEqual(_pareto_frontier(rows, ("a", "b")), {2, 3})

    def test_worker_identity_is_not_scientific_data(self) -> None:
        row = {"state_fingerprint": "state", "worker_pid": 123}
        self.assertEqual(_scientific_row(row), {"state_fingerprint": "state"})


if __name__ == "__main__":
    unittest.main()
