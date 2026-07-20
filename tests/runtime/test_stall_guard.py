from __future__ import annotations

import unittest

from experiments.stall_guard import StallGuardState, load_stall_guard_config


CONFIG = {
    "schema": "lns2.stall_guard.v1",
    "schema_version": 1,
    "unchanged_state_attempts_per_level": 2,
    "size_caps": [16, 8, 4],
    "terminal_fallback": "official_adaptive",
    "reset_on_state_fingerprint_change": True,
}


def candidates() -> list[dict]:
    return [
        {"candidate_id": "large", "actual_size": 16},
        {"candidate_id": "medium", "actual_size": 8},
        {"candidate_id": "small", "actual_size": 4},
    ]


class StallGuardTests(unittest.TestCase):
    def guard(self) -> StallGuardState:
        return StallGuardState(load_stall_guard_config(CONFIG))

    def fail_once(self, guard: StallGuardState, before: str = "state") -> dict:
        selected, _ = guard.select(candidates(), [3.0, 2.0, 1.0], before_fingerprint=before)
        self.assertIsNotNone(selected)
        return guard.observe(
            after_fingerprint="attempt-counter",
            replan_success=False,
            paths_changed=False,
            conflict_graph_changed=False,
            sum_of_costs_changed=False,
            actual_neighborhood_size=16,
        )

    def test_first_attempt_preserves_v2_winner(self) -> None:
        guard = self.guard()
        selected, row = guard.select(
            candidates(), [3.0, 2.0, 1.0], before_fingerprint="state"
        )
        self.assertEqual(selected, 0)
        self.assertTrue(row["base_selection_preserved"])
        self.assertEqual(row["active_size_cap"], 16)

    def test_two_failures_back_off_and_blacklist_candidate(self) -> None:
        guard = self.guard()
        self.fail_once(guard)
        second = self.fail_once(guard)
        self.assertTrue(second["backoff_triggered"])
        self.assertTrue(second["candidate_blacklisted_after"])
        self.assertEqual(second["next_active_size_cap"], 8)
        selected, row = guard.select(
            candidates(), [3.0, 2.0, 1.0], before_fingerprint="counter-changed"
        )
        self.assertEqual(selected, 1)
        self.assertEqual(row["effective_selected_candidate_id"], "medium")

    def test_guard_progresses_16_to_8_to_4_to_official(self) -> None:
        guard = self.guard()
        for _ in range(6):
            self.fail_once(guard)
        self.assertEqual(guard.route_before_selection("ignored"), "official_adaptive")
        selected, row = guard.select(
            candidates(), [3.0, 2.0, 1.0], before_fingerprint="ignored"
        )
        self.assertIsNone(selected)
        self.assertEqual(row["fallback_reason"], "terminal_stagnation")

    def test_success_resets_memory_and_records_rescue(self) -> None:
        guard = self.guard()
        self.fail_once(guard)
        self.fail_once(guard)
        guard.select(candidates(), [3.0, 2.0, 1.0], before_fingerprint="state")
        observed = guard.observe(
            after_fingerprint="new-state",
            replan_success=True,
            paths_changed=True,
            conflict_graph_changed=True,
            sum_of_costs_changed=True,
            actual_neighborhood_size=8,
        )
        self.assertTrue(observed["repair_state_changed"])
        self.assertEqual(guard.active_size_cap, 16)
        self.assertFalse(guard.blacklisted_candidates)
        self.assertEqual(guard.summary()["rescued_state_count"], 1)

    def test_success_without_structural_change_is_still_stagnant(self) -> None:
        guard = self.guard()
        guard.select(candidates(), [3.0, 2.0, 1.0], before_fingerprint="state")
        observed = guard.observe(
            after_fingerprint="attempt-counter",
            replan_success=True,
            paths_changed=False,
            conflict_graph_changed=False,
            sum_of_costs_changed=False,
            actual_neighborhood_size=16,
        )
        self.assertTrue(observed["stagnant_attempt"])
        self.assertFalse(observed["repair_state_changed"])

    def test_invalid_config_is_rejected(self) -> None:
        invalid = dict(CONFIG)
        invalid["size_caps"] = [8, 16, 4]
        with self.assertRaises(ValueError):
            load_stall_guard_config(invalid)


if __name__ == "__main__":
    unittest.main()
