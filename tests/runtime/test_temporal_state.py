from __future__ import annotations

import unittest

from lns2_selector.runtime.temporal_state import (
    TEMPORAL_STATE_SCHEMA,
    temporal_history_context,
    temporal_state_identity,
)


class TemporalStateIdentityTest(unittest.TestCase):
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
