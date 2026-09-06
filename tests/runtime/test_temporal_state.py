from __future__ import annotations

import unittest

from lns2_selector.runtime.temporal_state import (
    TEMPORAL_STATE_SCHEMA,
    history_before_decision,
    temporal_history_context,
    temporal_state_identity,
)


def _trace_state(edges: list[tuple[int, int]], revision: int) -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "rows": 1,
        "cols": 7,
        "sum_of_costs": 6 + revision,
        "obstacles": [0] * 7,
        "agents": [
            {"id": agent, "path": [agent, agent + 1]}
            for agent in range(6)
        ],
        "conflict_edges": [list(edge) for edge in edges],
        "num_of_colliding_pairs": len(edges),
        "feasible": not edges,
        "version": revision,
    }


def _trace_transition(agents: list[int], candidate: str) -> dict:
    return {
        "action": {"agents": agents},
        "metrics": {"neighborhood": agents, "replan_success": True},
        "controller": {"selected_candidate_id": candidate},
    }


class TemporalStateIdentityTest(unittest.TestCase):
    def test_history_uses_only_strict_prefix_and_keeps_repetition(self) -> None:
        states = [
            _trace_state([(0, 1)], 0),
            _trace_state([(0, 1), (2, 3)], 1),
            _trace_state([(0, 1)], 2),
            _trace_state([(4, 5)], 3),
        ]
        transitions = [
            _trace_transition([0, 1], "a"),
            _trace_transition([0, 1], "a"),
            _trace_transition([4, 5], "future"),
        ]
        history = history_before_decision(states, transitions, decision_index=2)
        self.assertEqual(history.recent_candidate_ids, ("a", "a"))
        self.assertEqual(history.recent_neighborhood_exact_repeat, (False, True))
        self.assertEqual(history.recent_neighborhood_max_jaccard, (0.0, 1.0))
        self.assertEqual(history.persistent_conflict_edges, ((0, 1),))
        self.assertEqual(history.disappeared_conflict_edges, ((2, 3),))
        self.assertNotIn("future", history.recent_candidate_ids)
        window = history_before_decision(
            states, transitions, decision_index=2, history_limit=1
        )
        self.assertEqual(window.recent_candidate_ids, ("a",))
        self.assertEqual(window.recent_neighborhood_exact_repeat, (True,))
        self.assertEqual(window.agent_repair_counts, ((0, 2), (1, 2)))
        self.assertEqual(window.persistent_conflict_edges, history.persistent_conflict_edges)

    def test_history_at_initial_decision_has_no_future_actions(self) -> None:
        history = history_before_decision(
            [_trace_state([(0, 1)], 0)],
            [{"invalid_future_transition": True}],
            decision_index=0,
        )
        self.assertEqual(history.recent_neighborhoods, ())
        self.assertEqual(history.agent_repair_counts, ())
        self.assertEqual(history.new_conflict_edges, ((0, 1),))
        self.assertEqual(history.conflict_signature_streak, 1)

    def _history(self):
        return temporal_history_context(
            recent_neighborhoods=[[3, 2, 2], [5, 4]],
            recent_neighborhood_exact_repeat=[True, False],
            recent_neighborhood_max_jaccard=[1.0, 0.25],
            persistent_conflict_edges=[[9, 2], [2, 9]],
            new_conflict_edges=[[3, 4]],
            disappeared_conflict_edges=[[5, 6]],
            reappeared_conflict_edges=[[7, 8]],
            conflict_signature_streak=4,
            agent_repair_counts={3: 2, 2: 5},
            recent_candidate_ids=["candidate-a", "candidate-b"],
            recent_pp_history=[(True, False, True), (False, True, False)],
        )

    def test_identity_keeps_same_fingerprint_at_distinct_decisions(self) -> None:
        history = self._history()
        first = temporal_state_identity(
            episode_id="episode-1",
            decision_index=2,
            state_fingerprint="a" * 64,
            history=history,
        )
        second = temporal_state_identity(
            episode_id="episode-1",
            decision_index=3,
            state_fingerprint="a" * 64,
            history=history,
        )
        self.assertEqual(first["schema"], TEMPORAL_STATE_SCHEMA)
        self.assertEqual(first["state_fingerprint"], second["state_fingerprint"])
        self.assertNotEqual(first["state_occurrence_id"], second["state_occurrence_id"])

    def test_identity_and_history_hash_are_deterministic(self) -> None:
        first_history = self._history()
        second_history = temporal_history_context(
            recent_neighborhoods=[[2, 3], [4, 5]],
            recent_neighborhood_exact_repeat=[True, False],
            recent_neighborhood_max_jaccard=[1, 0.25],
            persistent_conflict_edges=[[2, 9]],
            new_conflict_edges=[[4, 3]],
            disappeared_conflict_edges=[[6, 5]],
            reappeared_conflict_edges=[[8, 7]],
            conflict_signature_streak=4,
            agent_repair_counts=[(2, 5), (3, 2)],
            recent_candidate_ids=["candidate-a", "candidate-b"],
            recent_pp_history=[(True, False, True), (False, True, False)],
        )
        self.assertEqual(first_history, second_history)
        self.assertEqual(first_history.sha256, second_history.sha256)
        identity = temporal_state_identity(
            episode_id="episode-1",
            decision_index=2,
            state_fingerprint="ABCDEF" * 10 + "ABCD",
            history=first_history,
        )
        repeated = temporal_state_identity(
            episode_id="episode-1",
            decision_index=2,
            state_fingerprint="abcdef" * 10 + "abcd",
            history=second_history,
        )
        self.assertEqual(identity, repeated)

    def test_rejects_invalid_history_and_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid conflict edge"):
            temporal_history_context(persistent_conflict_edges=[[1, 1]])
        with self.assertRaisesRegex(ValueError, "must align"):
            temporal_history_context(
                recent_neighborhoods=[[1, 2]],
                recent_neighborhood_exact_repeat=[True, False],
            )
        with self.assertRaisesRegex(ValueError, "Jaccard"):
            temporal_history_context(
                recent_neighborhoods=[[1, 2]],
                recent_neighborhood_max_jaccard=[1.1],
            )
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            temporal_state_identity(
                episode_id="episode-1",
                decision_index=0,
                state_fingerprint="z" * 64,
                history=temporal_history_context(),
            )


if __name__ == "__main__":
    unittest.main()
