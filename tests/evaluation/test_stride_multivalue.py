from __future__ import annotations

import unittest

from experiments.stride_multivalue import (
    WorkerPreflightMeasurement,
    anchor_relative_targets,
    choose_dynamic_worker_count,
    multihorizon_targets,
)


def _state(edges: list[tuple[int, int]], *, feasible: bool = False) -> dict:
    return {
        "conflict_edges": [list(edge) for edge in edges],
        "num_of_colliding_pairs": len(edges),
        "feasible": feasible,
    }


def _transition(agents: list[int]) -> dict:
    return {"metrics": {"neighborhood": agents}}


class MultiValueTargetsTest(unittest.TestCase):
    def test_completed_rollout_produces_all_distributional_targets(self) -> None:
        states = [
            _state([(0, 1), (2, 3)]),
            _state([(0, 1), (1, 2)]),
            _state([], feasible=True),
        ]
        transitions = [_transition([0, 1]), _transition([0, 1])]
        result = multihorizon_targets(
            states,
            transitions,
            stop_reason="success",
            horizons=(1, 2, 4),
        )
        first = result["horizons"]["1"]
        self.assertTrue(first["horizon_observed"])
        self.assertFalse(first["feasible_event_observed"])
        self.assertAlmostEqual(first["original_conflict_edge_residual_rate"], 0.5)
        self.assertAlmostEqual(first["new_conflict_edge_rate"], 0.5)
        self.assertEqual(first["neighborhood_history"]["exact_repeat_count"], 0)
        fourth = result["horizons"]["4"]
        self.assertTrue(fourth["horizon_observed"])
        self.assertTrue(fourth["feasible_event_observed"])
        self.assertEqual(fourth["terminal_conflict_ratio"], 0.0)
        self.assertEqual(
            fourth["neighborhood_history"]["maximum_exact_repeat_streak"], 1
        )

    def test_right_censoring_masks_unobserved_full_horizon_metrics(self) -> None:
        states = [_state([(0, 1), (2, 3)]), _state([(0, 1)])]
        result = multihorizon_targets(
            states,
            [_transition([0, 1])],
            stop_reason="wall_timeout",
            horizons=(1, 8),
        )
        observed = result["horizons"]["1"]
        censored = result["horizons"]["8"]
        self.assertTrue(observed["horizon_observed"])
        self.assertFalse(observed["right_censored"])
        self.assertFalse(censored["horizon_observed"])
        self.assertTrue(censored["right_censored"])
        self.assertIsNone(censored["normalized_conflict_auc"])
        self.assertIsNone(censored["terminal_conflict_ratio"])
        self.assertIsNotNone(censored["observed_normalized_conflict_auc"])
        self.assertEqual(censored["observed_terminal_conflict_ratio"], 0.5)

    def test_anchor_delta_keeps_censoring_mask(self) -> None:
        anchor = multihorizon_targets(
            [_state([(0, 1)]), _state([], feasible=True)],
            [_transition([0, 1])],
            stop_reason="success",
            horizons=(1, 8),
        )
        candidate = multihorizon_targets(
            [_state([(0, 1)]), _state([(0, 1)])],
            [_transition([0, 1])],
            stop_reason="repair_limit",
            horizons=(1, 8),
        )
        delta = anchor_relative_targets(candidate, anchor)
        self.assertEqual(delta["horizons"]["1"]["feasible_event_delta"], -1)
        self.assertIsNotNone(
            delta["horizons"]["1"]["normalized_conflict_auc_delta"]
        )
        self.assertFalse(delta["horizons"]["8"]["paired_horizon_observed"])
        self.assertIsNone(
            delta["horizons"]["8"]["normalized_conflict_auc_delta"]
        )

    def test_dynamic_workers_choose_throughput_under_cpu_and_memory_limits(self) -> None:
        gib = 1024**3
        measurements = [
            WorkerPreflightMeasurement(12, 3.0, 8 * gib),
            WorkerPreflightMeasurement(14, 3.5, 10 * gib),
            WorkerPreflightMeasurement(16, 3.8, 19 * gib),
        ]
        self.assertEqual(
            choose_dynamic_worker_count(
                measurements,
                logical_cpu_count=20,
                available_memory_bytes=24 * gib,
            ),
            14,
        )
        with self.assertRaisesRegex(ValueError, "eligible"):
            choose_dynamic_worker_count(
                [WorkerPreflightMeasurement(12, 3.0, 8 * gib, error_count=1)],
                logical_cpu_count=20,
                available_memory_bytes=24 * gib,
            )


if __name__ == "__main__":
    unittest.main()
