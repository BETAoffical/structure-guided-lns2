from __future__ import annotations

import unittest

from experiments.stall_shadow import (
    StallShadowState,
    load_stall_shadow_config,
    pp_attempt_key,
)


def _config(**overrides):
    raw = {
        "schema": "lns2.stall_shadow.v2",
        "schema_version": 2,
        "mode": "shadow",
        "unchanged_attempt_thresholds": [3, 4, 6],
        "minimum_distinct_pp_attempts": 2,
        "future_observation_decisions": 3,
        "maximum_false_trigger_rate": 0.01,
        "post_state_change_cooldown_decisions": 2,
        "deployment_enabled": False,
    }
    raw.update(overrides)
    return load_stall_shadow_config(raw)


CANDIDATES = [
    {"candidate_id": "winner", "agents": [1, 2, 3, 4], "actual_size": 4},
    {"candidate_id": "backup", "agents": [5, 6, 7, 8], "actual_size": 4},
]
SCORES = [2.0, 1.0]


class StallShadowTests(unittest.TestCase):
    def _step(
        self,
        state: StallShadowState,
        *,
        decision: int,
        before: str,
        after: str,
        replan_success: bool,
        before_conflicts: int = 10,
        after_conflicts: int = 10,
        seed: int = 1,
    ):
        selected, before_diagnostic = state.before_selection(
            CANDIDATES,
            SCORES,
            0,
            before_fingerprint=before,
            decision_index=decision,
        )
        self.assertEqual(selected, 0)
        self.assertTrue(before_diagnostic["base_selection_preserved"])
        observed = state.observe(
            before_fingerprint=before,
            after_fingerprint=after,
            replan_success=replan_success,
            conflicts_before=before_conflicts,
            conflicts_after=after_conflicts,
            feasible=after_conflicts == 0,
            candidate_id="winner",
            actual_agents=[1, 2, 3, 4],
            step_random_seed=seed,
            requested_pp_seed=seed,
            applied_pp_seed=seed,
            repair_order=[1, 2, 3, 4] if seed % 2 else [4, 3, 2, 1],
        )
        return before_diagnostic, observed

    def test_single_failure_never_triggers(self) -> None:
        state = StallShadowState(_config())
        diagnostic, observed = self._step(
            state,
            decision=0,
            before="same",
            after="same",
            replan_success=False,
        )
        self.assertEqual(diagnostic["triggered_thresholds"], [])
        self.assertEqual(observed["repair_outcome"], "hard_failure")
        selected, diagnostic = state.before_selection(
            CANDIDATES,
            SCORES,
            0,
            before_fingerprint="same",
            decision_index=1,
        )
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["triggered_thresholds"], [])

    def test_state_changed_without_reduction_resets_and_never_triggers(self) -> None:
        state = StallShadowState(_config())
        _diagnostic, observed = self._step(
            state,
            decision=0,
            before="state-a",
            after="state-b",
            replan_success=True,
        )
        self.assertEqual(observed["repair_outcome"], "state_changed_no_reduction")
        self.assertFalse(observed["no_progress"])
        _selected, diagnostic = state.before_selection(
            CANDIDATES,
            SCORES,
            0,
            before_fingerprint="state-b",
            decision_index=1,
        )
        self.assertEqual(diagnostic["cooldown_remaining_before"], 2)
        self.assertEqual(diagnostic["triggered_thresholds"], [])
        self.assertEqual(
            state.summary()["state_changed_no_reduction_reset_count"], 1
        )

    def test_three_distinct_same_state_noops_trigger_shadow_only(self) -> None:
        state = StallShadowState(_config())
        for decision, seed in enumerate((11, 12, 13)):
            diagnostic, observed = self._step(
                state,
                decision=decision,
                before="same",
                after="same",
                replan_success=True,
                seed=seed,
            )
            self.assertEqual(diagnostic["triggered_thresholds"], [])
            self.assertEqual(observed["repair_outcome"], "accepted_noop")
        selected, diagnostic = state.before_selection(
            CANDIDATES,
            SCORES,
            0,
            before_fingerprint="same",
            decision_index=3,
        )
        self.assertEqual(selected, 0)
        self.assertEqual(diagnostic["triggered_thresholds"], [3])
        self.assertTrue(diagnostic["base_selection_preserved"])
        observed = state.observe(
            before_fingerprint="same",
            after_fingerprint="changed",
            replan_success=True,
            conflicts_before=10,
            conflicts_after=9,
            feasible=False,
            candidate_id="winner",
            actual_agents=[1, 2, 3, 4],
            step_random_seed=14,
            requested_pp_seed=14,
            applied_pp_seed=14,
            repair_order=[2, 1, 4, 3],
        )
        self.assertEqual(
            observed["resolved_triggers"][0]["resolution"],
            "premature_trigger",
        )
        summary = state.summary()
        self.assertEqual(summary["action_override_count"], 0)
        self.assertEqual(
            summary["thresholds"]["3"]["premature_trigger_count"], 1
        )
        self.assertFalse(summary["thresholds"]["3"]["false_trigger_gate_passed"])
        self.assertEqual(
            summary["thresholds"]["3"]["gate_status"],
            "external_audit_required",
        )

    def test_action_seed_is_part_of_pp_attempt_identity(self) -> None:
        common = {
            "state_fingerprint": "same",
            "agents": [1, 2, 3, 4],
            "requested_pp_seed": 7,
            "applied_pp_seed": 7,
            "repair_order": [1, 2, 3, 4],
        }
        self.assertNotEqual(
            pp_attempt_key(step_random_seed=11, **common),
            pp_attempt_key(step_random_seed=12, **common),
        )

    def test_negative_native_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested PP seed"):
            pp_attempt_key(
                state_fingerprint="same",
                agents=[1, 2],
                step_random_seed=1,
                requested_pp_seed=-1,
                applied_pp_seed=None,
                repair_order=[],
            )

    def test_aborted_selection_rolls_back_trigger_and_decision(self) -> None:
        state = StallShadowState(_config())
        for decision, seed in enumerate((11, 12, 13)):
            self._step(
                state,
                decision=decision,
                before="same",
                after="same",
                replan_success=False,
                seed=seed,
            )
        _selected, diagnostic = state.before_selection(
            CANDIDATES,
            SCORES,
            0,
            before_fingerprint="same",
            decision_index=3,
        )
        self.assertEqual(diagnostic["triggered_thresholds"], [3])
        state.abort_selection()
        summary = state.summary()
        self.assertEqual(summary["decision_count"], 3)
        self.assertEqual(summary["thresholds"]["3"]["trigger_count"], 0)
        self.assertEqual(summary["thresholds"]["3"]["unresolved_trigger_count"], 0)

    def test_deployment_cannot_be_enabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "diagnostic-only"):
            _config(deployment_enabled=True)

    def test_config_does_not_coerce_integer_strings(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be an integer"):
            _config(minimum_distinct_pp_attempts="2")


if __name__ == "__main__":
    unittest.main()
