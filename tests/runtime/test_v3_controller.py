from __future__ import annotations

import unittest

from experiments.repair_aware import PortableScalarModel
from experiments.v3_controller import (
    V3_H3_BUNDLE_SCHEMA,
    V3ControllerBundle,
    V3ControllerState,
    v3_candidate_order,
    v3_h3_candidate_order,
)


def bundle() -> V3ControllerBundle:
    return V3ControllerBundle(
        models={},
        thresholds={
            "effective_probability_tolerance": 0.10,
            "no_progress_probability_tolerance": 0.10,
            "conflict_reduction_retention": 0.90,
        },
        selection_overhead_seconds=0.1,
        manifest={},
        report={},
    )


def candidates() -> list[dict[str, object]]:
    return [
        {"candidate_id": "c0", "agents": [0, 1], "actual_size": 4},
        {"candidate_id": "c0-alias", "agents": [1, 0], "actual_size": 4},
        {"candidate_id": "c1", "agents": [2, 3], "actual_size": 8},
        {"candidate_id": "c2", "agents": [4, 5], "actual_size": 16},
        {"candidate_id": "c3", "agents": [6, 7], "actual_size": 8},
    ]


PREDICTIONS = {
    "effective_progress_probability": [0.90] * 5,
    "no_progress_probability": [0.10] * 5,
    "conflict_reduction": [3.0] * 5,
    "repair_seconds": [1.0] * 5,
    "utility": [5.0, 4.9, 4.0, 3.0, 2.0],
}
V2_SCORES = [1.0, 5.0, 4.0, 3.0, 2.0]


class V3CandidateOrderingTests(unittest.TestCase):
    def test_utility_selects_first_repair_instead_of_v2_score(self) -> None:
        order = v3_candidate_order(
            candidates(), PREDICTIONS, V2_SCORES, bundle().thresholds
        )
        self.assertEqual(order[0], 0)
        self.assertNotEqual(order[0], V2_SCORES.index(max(V2_SCORES)))

    def test_safety_shortlists_before_utility(self) -> None:
        predictions = {name: list(values) for name, values in PREDICTIONS.items()}
        predictions["effective_progress_probability"][0] = 0.1
        order = v3_candidate_order(
            candidates(), predictions, V2_SCORES, bundle().thresholds
        )
        self.assertNotIn(0, order)

    def test_horizon_utility_prefers_two_fast_size8_repairs(self) -> None:
        values = [
            {"candidate_id": "size8", "actual_size": 8},
            {"candidate_id": "size16", "actual_size": 16},
        ]
        predictions = {
            "effective_progress_probability": [0.9, 0.9],
            "no_progress_probability": [0.1, 0.1],
            "h3_no_progress_probability": [0.1, 0.1],
            "utility": [10.0, 5.0],
        }
        order = v3_h3_candidate_order(
            values,
            predictions,
            [1.0, 2.0],
            {
                "h1_effective_probability_tolerance": 0.1,
                "h1_no_progress_probability_tolerance": 0.1,
                "minimum_h3_utility_improvement": 0.0,
            },
        )
        self.assertEqual(order[0], 0)

    def test_horizon_utility_prefers_one_size16_when_repeated_size8_is_slower(self) -> None:
        values = [
            {"candidate_id": "size8", "actual_size": 8},
            {"candidate_id": "size16", "actual_size": 16},
        ]
        predictions = {
            "effective_progress_probability": [0.9, 0.9],
            "no_progress_probability": [0.1, 0.1],
            "h3_no_progress_probability": [0.1, 0.1],
            "utility": [4.0, 6.0],
        }
        order = v3_h3_candidate_order(
            values,
            predictions,
            [2.0, 1.0],
            {
                "h1_effective_probability_tolerance": 0.1,
                "h1_no_progress_probability_tolerance": 0.1,
                "minimum_h3_utility_improvement": 0.0,
            },
        )
        self.assertEqual(order[0], 1)

    def test_horizon_margin_keeps_v2_winner_for_small_predicted_gain(self) -> None:
        values = [{"candidate_id": "new"}, {"candidate_id": "v2"}]
        predictions = {
            "effective_progress_probability": [0.9, 0.9],
            "no_progress_probability": [0.1, 0.1],
            "h3_no_progress_probability": [0.1, 0.1],
            "utility": [10.0, 9.5],
        }
        order = v3_h3_candidate_order(
            values,
            predictions,
            [1.0, 2.0],
            {
                "h1_effective_probability_tolerance": 0.1,
                "h1_no_progress_probability_tolerance": 0.1,
                "minimum_h3_utility_improvement": 0.10,
            },
        )
        self.assertEqual(order[0], 1)

    def test_horizon_equal_utility_keeps_v2_winner(self) -> None:
        values = [{"candidate_id": "h3"}, {"candidate_id": "v2"}]
        predictions = {
            "effective_progress_probability": [0.9, 0.9],
            "no_progress_probability": [0.1, 0.1],
            "h3_no_progress_probability": [0.1, 0.1],
            "utility": [5.0, 5.0],
        }
        order = v3_h3_candidate_order(
            values,
            predictions,
            [1.0, 2.0],
            {
                "h1_effective_probability_tolerance": 0.1,
                "h1_no_progress_probability_tolerance": 0.1,
                "minimum_h3_utility_improvement": 0.0,
            },
        )
        self.assertEqual(order[0], 1)

    def test_horizon_v2_score_tie_uses_formal_lexicographic_rule(self) -> None:
        values = [
            {"candidate_id": "z", "candidate_key": "z"},
            {"candidate_id": "a", "candidate_key": "a"},
        ]
        predictions = {
            "effective_progress_probability": [0.9, 0.9],
            "no_progress_probability": [0.1, 0.1],
            "h3_no_progress_probability": [0.1, 0.1],
            "utility": [5.0, 5.0],
        }
        order = v3_h3_candidate_order(
            values,
            predictions,
            [2.0, 2.0],
            {
                "h1_effective_probability_tolerance": 0.1,
                "h1_no_progress_probability_tolerance": 0.1,
                "minimum_h3_utility_improvement": 0.0,
            },
        )
        self.assertEqual(order[0], 1)


class V3RuntimeProjectionTests(unittest.TestCase):
    def test_bundle_reports_union_of_runtime_model_features(self) -> None:
        models = {
            "a": PortableScalarModel(
                "a",
                "realized_dynamic",
                ["x", "y"],
                0.0,
                [],
                "identity",
                "a",
                declared_feature_count=4,
            ),
            "b": PortableScalarModel(
                "b",
                "realized_dynamic",
                ["y", "z"],
                0.0,
                [],
                "identity",
                "b",
                declared_feature_count=4,
            ),
        }
        runtime_bundle = V3ControllerBundle(
            models=models,
            thresholds={},
            selection_overhead_seconds=0.0,
            manifest={"feature_names": ["w", "x", "y", "z"]},
            report={},
        )
        self.assertEqual(runtime_bundle.required_feature_names, ("x", "y", "z"))
        self.assertEqual(
            runtime_bundle.runtime_projection,
            {
                "declared_feature_count": 4,
                "runtime_feature_count": 3,
                "removed_runtime_feature_count": 1,
                "model_runtime_feature_counts": {"a": 2, "b": 2},
            },
        )

    def test_horizon_bundle_schema_is_independent_from_v3(self) -> None:
        runtime_bundle = V3ControllerBundle(
            models={},
            thresholds={},
            selection_overhead_seconds=0.0,
            manifest={"schema": V3_H3_BUNDLE_SCHEMA},
            report={},
        )
        self.assertTrue(runtime_bundle.is_horizon)


class V3ControllerStateTests(unittest.TestCase):
    def observe_no_progress(
        self, state: V3ControllerState, *, hard_failure: bool = False
    ) -> None:
        state.observe(
            before_fingerprint="state",
            after_fingerprint="state",
            replan_success=not hard_failure,
            conflicts_before=10,
            conflicts_after=10,
            feasible=False,
        )

    def test_blacklist_uses_actual_agent_neighborhood(self) -> None:
        state = V3ControllerState(bundle())
        selected, _diagnostic = state.select(
            candidates(), V2_SCORES, PREDICTIONS, before_fingerprint="state"
        )
        self.assertEqual(selected, 0)
        self.observe_no_progress(state)
        selected, diagnostic = state.select(
            candidates(), V2_SCORES, PREDICTIONS, before_fingerprint="state"
        )
        self.assertEqual(selected, 2)
        self.assertEqual(diagnostic["excluded_candidate_count"], 2)

    def test_three_distinct_failures_then_adaptive_persists(self) -> None:
        state = V3ControllerState(bundle(), maximum_distinct_failures=3)
        selected_ids = []
        for index in range(3):
            selected, diagnostic = state.select(
                candidates(), V2_SCORES, PREDICTIONS, before_fingerprint="state"
            )
            self.assertIsNotNone(selected)
            selected_ids.append(candidates()[int(selected)]["candidate_id"])
            self.observe_no_progress(state, hard_failure=index == 1)
        self.assertEqual(len(set(selected_ids)), 3)

        selected, diagnostic = state.select(
            candidates(), V2_SCORES, None, before_fingerprint="state"
        )
        self.assertIsNone(selected)
        self.assertEqual(diagnostic["route"], "official_adaptive")
        self.observe_no_progress(state)
        selected, _diagnostic = state.select(
            candidates(), V2_SCORES, None, before_fingerprint="state"
        )
        self.assertIsNone(selected)

    def test_state_change_clears_cache_blacklist_and_fallback(self) -> None:
        state = V3ControllerState(bundle(), maximum_distinct_failures=1)
        state.select(candidates(), V2_SCORES, PREDICTIONS, before_fingerprint="state")
        self.observe_no_progress(state)
        selected, _diagnostic = state.select(
            candidates(), V2_SCORES, None, before_fingerprint="state"
        )
        self.assertIsNone(selected)
        state.observe(
            before_fingerprint="state",
            after_fingerprint="new-state",
            replan_success=True,
            conflicts_before=10,
            conflicts_after=9,
            feasible=False,
        )
        selected, diagnostic = state.select(
            candidates(), V2_SCORES, PREDICTIONS, before_fingerprint="new-state"
        )
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["failed_candidate_count"], 0)
        self.assertFalse(diagnostic["adaptive_fallback_active"])


if __name__ == "__main__":
    unittest.main()
