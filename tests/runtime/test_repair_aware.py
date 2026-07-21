from __future__ import annotations

import unittest

from experiments.repair_aware import (
    PortableScalarModel,
    RepairAwareBundle,
    RepairAwareState,
    classify_repair_outcome,
    guarded_tiebreak_candidate,
    load_repair_aware_config,
)


CONFIG = {
    "schema": "lns2.repair_aware_controller.v2",
    "mode": "rescue-only",
    "max_model_rescues": None,
    "same_candidate_attempt_limit": 2,
    "lazy_neighborhood_sizes": [12],
    "terminal_fallback": "official_adaptive",
    "fallback_until_state_change": True,
    "reset_on_state_fingerprint_change": True,
}


def bundle(*, guarded: bool = False) -> RepairAwareBundle:
    return RepairAwareBundle(
        models={},
        thresholds={
            "rescue_probability_tolerance": 0.05,
            "tie_score_gap_fraction": 0.10,
            "tie_progress_margin": 0.05,
            "tie_minimum_reduction_ratio": 0.98,
            "tie_maximum_generated_ratio": 0.90,
        },
        guarded_tiebreak_eligible=guarded,
        manifest={},
        report={},
    )


def candidates() -> list[dict[str, object]]:
    return [
        {"candidate_id": f"c{index}", "actual_size": size}
        for index, size in enumerate((16, 8, 4, 8, 4))
    ]


PREDICTIONS = {
    "progress_probability": [0.40, 0.90, 0.80, 0.70, 0.60],
    "conflict_reduction": [4.0, 3.0, 2.5, 2.0, 1.0],
    "generated": [100.0, 70.0, 50.0, 40.0, 30.0],
}
SCORES = [10.0, 9.9, 9.8, 9.7, 9.6]


class RepairOutcomeTests(unittest.TestCase):
    def test_outcomes_distinguish_failure_noop_and_progress(self) -> None:
        self.assertEqual(
            classify_repair_outcome(
                before_fingerprint="a",
                after_fingerprint="a",
                replan_success=False,
                conflicts_before=5,
                conflicts_after=5,
            ),
            "hard_failure",
        )
        self.assertEqual(
            classify_repair_outcome(
                before_fingerprint="a",
                after_fingerprint="a",
                replan_success=True,
                conflicts_before=5,
                conflicts_after=5,
            ),
            "accepted_noop",
        )
        self.assertEqual(
            classify_repair_outcome(
                before_fingerprint="a",
                after_fingerprint="b",
                replan_success=True,
                conflicts_before=5,
                conflicts_after=3,
            ),
            "conflict_reduced",
        )

    def test_failed_replan_must_not_change_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "failed PP changed"):
            classify_repair_outcome(
                before_fingerprint="a",
                after_fingerprint="b",
                replan_success=False,
                conflicts_before=5,
                conflicts_after=5,
            )


class RepairAwareStateTests(unittest.TestCase):
    def state(self) -> RepairAwareState:
        return RepairAwareState(load_repair_aware_config(CONFIG), bundle())

    def observe_noop(self, state: RepairAwareState, *, hard: bool = False) -> None:
        state.observe(
            before_fingerprint="state",
            after_fingerprint="state",
            replan_success=not hard,
            conflicts_before=10,
            conflicts_after=10,
            feasible=False,
        )

    def test_first_decision_preserves_v2_then_uses_distinct_rescues(self) -> None:
        state = self.state()
        self.assertFalse(state.predictions_required("state"))
        selected, diagnostic = state.select(
            candidates(), SCORES, 0, None, before_fingerprint="state"
        )
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["selection_kind"], "base")
        self.assertIsNone(diagnostic["shadow_selected_candidate_id"])
        self.assertEqual(diagnostic["predictions"], {})
        self.observe_noop(state)
        self.assertTrue(state.predictions_required("state"))

        selected, diagnostic = state.select(
            candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="state"
        )
        self.assertEqual(selected, 1)
        self.assertEqual(diagnostic["selection_kind"], "rescue")
        self.observe_noop(state, hard=True)

        selected, _ = state.select(
            candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="state"
        )
        self.assertEqual(selected, 1)
        self.observe_noop(state)
        selected, _ = state.select(
            candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="state"
        )
        self.assertNotIn(selected, {0, 1})

    def test_three_rescues_then_adaptive_persists_until_state_change(self) -> None:
        state = self.state()
        state.select(candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="state")
        self.observe_noop(state)
        for _ in range(3):
            selected, diagnostic = state.select(
                candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="state"
            )
            self.assertIsNotNone(selected)
            self.assertEqual(diagnostic["selection_kind"], "rescue")
            self.observe_noop(state)
        selected, diagnostic = state.select(
            candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="state"
        )
        self.assertIsNone(selected)
        self.assertEqual(diagnostic["selection_kind"], "official_fallback")
        self.observe_noop(state)
        self.assertFalse(state.consume_refresh())
        selected, diagnostic = state.select(
            candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="state"
        )
        self.assertIsNone(selected)
        self.assertEqual(diagnostic["selection_kind"], "official_fallback")
        state.observe(
            before_fingerprint="state",
            after_fingerprint="new-state",
            replan_success=True,
            conflicts_before=10,
            conflicts_after=9,
            feasible=False,
        )
        selected, diagnostic = state.select(
            candidates(), SCORES, 0, None, before_fingerprint="new-state"
        )
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["selection_kind"], "base")

    def test_state_change_resets_failure_memory(self) -> None:
        state = self.state()
        state.select(candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="state")
        result = state.observe(
            before_fingerprint="state",
            after_fingerprint="new-state",
            replan_success=True,
            conflicts_before=10,
            conflicts_after=8,
            feasible=False,
        )
        self.assertEqual(result["repair_outcome"], "conflict_reduced")
        selected, diagnostic = state.select(
            candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="new-state"
        )
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["selection_kind"], "base")


class GuardedTieBreakTests(unittest.TestCase):
    def test_guard_can_select_a_cheaper_near_tie(self) -> None:
        predictions = {
            "progress_probability": [0.50, 0.70, 0.20],
            "conflict_reduction": [4.0, 4.0, 9.0],
            "generated": [100.0, 80.0, 10.0],
        }
        selected = guarded_tiebreak_candidate(
            candidates()[:3],
            predictions,
            [2.0, 1.95, 0.0],
            0,
            bundle(guarded=True).thresholds,
        )
        self.assertEqual(selected, 1)

    def test_unpromoted_guarded_mode_preserves_base(self) -> None:
        config = {**CONFIG, "mode": "guarded-tiebreak"}
        state = RepairAwareState(load_repair_aware_config(config), bundle(guarded=False))
        self.assertFalse(state.predictions_required("state"))
        selected, diagnostic = state.select(
            candidates(), SCORES, 0, None, before_fingerprint="state"
        )
        self.assertEqual(selected, 0)
        self.assertFalse(diagnostic["guarded_tiebreak_enabled"])

    def test_shadow_mode_requires_predictions_without_changing_base(self) -> None:
        config = {**CONFIG, "mode": "shadow"}
        state = RepairAwareState(load_repair_aware_config(config), bundle())
        self.assertTrue(state.predictions_required("state"))
        with self.assertRaisesRegex(ValueError, "predictions are required"):
            state.select(candidates(), SCORES, 0, None, before_fingerprint="state")
        selected, diagnostic = state.select(
            candidates(), SCORES, 0, PREDICTIONS, before_fingerprint="state"
        )
        self.assertEqual(selected, 0)
        self.assertIsNotNone(diagnostic["shadow_selected_candidate_id"])


class PortableScalarModelTests(unittest.TestCase):
    def test_python_scalar_tree_supports_raw_and_sigmoid(self) -> None:
        tree = [[
            {
                "value": 0.0,
                "feature_idx": 0,
                "num_threshold": 0.0,
                "missing_go_to_left": False,
                "left": 1,
                "right": 2,
                "is_leaf": False,
            },
            {"value": -1.0, "feature_idx": 0, "num_threshold": 0.0, "missing_go_to_left": False, "left": 0, "right": 0, "is_leaf": True},
            {"value": 1.0, "feature_idx": 0, "num_threshold": 0.0, "missing_go_to_left": False, "left": 0, "right": 0, "is_leaf": True},
        ]]
        rows = [
            {
                "feature_profile": "realized_dynamic",
                "feature_names": ("x",),
                "feature_values": (value,),
            }
            for value in (-1.0, 1.0)
        ]
        raw = PortableScalarModel("raw", "realized_dynamic", ["x"], 0.0, tree, "identity", "test")
        probability = PortableScalarModel("p", "realized_dynamic", ["x"], 0.0, tree, "sigmoid", "test")
        self.assertEqual(raw.predict(rows), [-1.0, 1.0])
        self.assertLess(probability.predict(rows)[0], 0.5)
        self.assertGreater(probability.predict(rows)[1], 0.5)


if __name__ == "__main__":
    unittest.main()
