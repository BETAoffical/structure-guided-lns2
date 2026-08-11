from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


TEMPORAL_STATE_SCHEMA = "lns2.stride.temporal_state_identity.v1"


def _normalized_agent_set(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted(set(map(int, values))))


def _normalized_edge(value: Iterable[int]) -> tuple[int, int]:
    edge = tuple(map(int, value))
    if len(edge) != 2 or edge[0] == edge[1]:
        raise ValueError("temporal history contains an invalid conflict edge")
    return tuple(sorted(edge))


@dataclass(frozen=True)
class TemporalHistoryContext:
    """Outcome-blind history attached to one concrete decision occurrence."""

    recent_neighborhoods: tuple[tuple[int, ...], ...] = ()
    recent_neighborhood_exact_repeat: tuple[bool, ...] = ()
    recent_neighborhood_max_jaccard: tuple[float, ...] = ()
    persistent_conflict_edges: tuple[tuple[int, int], ...] = ()
    new_conflict_edges: tuple[tuple[int, int], ...] = ()
    disappeared_conflict_edges: tuple[tuple[int, int], ...] = ()
    reappeared_conflict_edges: tuple[tuple[int, int], ...] = ()
    conflict_signature_streak: int = 0
    agent_repair_counts: tuple[tuple[int, int], ...] = ()
    recent_candidate_ids: tuple[str, ...] = ()
    recent_pp_history: tuple[tuple[bool, bool, bool], ...] = ()

    def __post_init__(self) -> None:
        if self.conflict_signature_streak < 0:
            raise ValueError("conflict signature streak must be non-negative")
        if self.recent_neighborhood_exact_repeat and len(
            self.recent_neighborhood_exact_repeat
        ) != len(self.recent_neighborhoods):
            raise ValueError("exact-repeat history must align with neighborhoods")
        if self.recent_neighborhood_max_jaccard and len(
            self.recent_neighborhood_max_jaccard
        ) != len(self.recent_neighborhoods):
            raise ValueError("Jaccard history must align with neighborhoods")
        if any(
            not 0.0 <= value <= 1.0
            for value in self.recent_neighborhood_max_jaccard
        ):
            raise ValueError("neighborhood Jaccard values must be in [0, 1]")
        if any(count < 0 for _agent, count in self.agent_repair_counts):
            raise ValueError("agent repair counts must be non-negative")
        agents = [agent for agent, _count in self.agent_repair_counts]
        if len(agents) != len(set(agents)):
            raise ValueError("agent repair counts contain duplicates")

    def payload(self) -> dict[str, Any]:
        return {
            "recent_neighborhoods": [list(row) for row in self.recent_neighborhoods],
            "recent_neighborhood_exact_repeat": list(
                self.recent_neighborhood_exact_repeat
            ),
            "recent_neighborhood_max_jaccard": list(
                self.recent_neighborhood_max_jaccard
            ),
            "persistent_conflict_edges": [
                list(edge) for edge in self.persistent_conflict_edges
            ],
            "new_conflict_edges": [list(edge) for edge in self.new_conflict_edges],
            "disappeared_conflict_edges": [
                list(edge) for edge in self.disappeared_conflict_edges
            ],
            "reappeared_conflict_edges": [
                list(edge) for edge in self.reappeared_conflict_edges
            ],
            "conflict_signature_streak": int(self.conflict_signature_streak),
            "agent_repair_counts": [
                [int(agent), int(count)] for agent, count in self.agent_repair_counts
            ],
            "recent_candidate_ids": list(self.recent_candidate_ids),
            "recent_pp_history": [
                {
                    "success": success,
                    "noop": noop,
                    "state_changed": state_changed,
                }
                for success, noop, state_changed in self.recent_pp_history
            ],
        }

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def temporal_history_context(
    *,
    recent_neighborhoods: Iterable[Iterable[int]] = (),
    recent_neighborhood_exact_repeat: Iterable[bool] = (),
    recent_neighborhood_max_jaccard: Iterable[float] = (),
    persistent_conflict_edges: Iterable[Iterable[int]] = (),
    new_conflict_edges: Iterable[Iterable[int]] = (),
    disappeared_conflict_edges: Iterable[Iterable[int]] = (),
    reappeared_conflict_edges: Iterable[Iterable[int]] = (),
    conflict_signature_streak: int = 0,
    agent_repair_counts: Mapping[int, int] | Iterable[tuple[int, int]] = (),
    recent_candidate_ids: Iterable[str] = (),
    recent_pp_history: Iterable[tuple[bool, bool, bool]] = (),
) -> TemporalHistoryContext:
    counts = dict(agent_repair_counts)
    return TemporalHistoryContext(
        recent_neighborhoods=tuple(
            _normalized_agent_set(row) for row in recent_neighborhoods
        ),
        recent_neighborhood_exact_repeat=tuple(
            map(bool, recent_neighborhood_exact_repeat)
        ),
        recent_neighborhood_max_jaccard=tuple(
            map(float, recent_neighborhood_max_jaccard)
        ),
        persistent_conflict_edges=tuple(
            sorted({_normalized_edge(edge) for edge in persistent_conflict_edges})
        ),
        new_conflict_edges=tuple(
            sorted({_normalized_edge(edge) for edge in new_conflict_edges})
        ),
        disappeared_conflict_edges=tuple(
            sorted({_normalized_edge(edge) for edge in disappeared_conflict_edges})
        ),
        reappeared_conflict_edges=tuple(
            sorted({_normalized_edge(edge) for edge in reappeared_conflict_edges})
        ),
        conflict_signature_streak=int(conflict_signature_streak),
        agent_repair_counts=tuple(
            sorted((int(agent), int(count)) for agent, count in counts.items())
        ),
        recent_candidate_ids=tuple(map(str, recent_candidate_ids)),
        recent_pp_history=tuple(
            (bool(success), bool(noop), bool(state_changed))
            for success, noop, state_changed in recent_pp_history
        ),
    )


def temporal_state_identity(
    *,
    episode_id: str,
    decision_index: int,
    state_fingerprint: str,
    history: TemporalHistoryContext,
) -> dict[str, Any]:
    if not episode_id:
        raise ValueError("temporal state identity requires an episode id")
    if int(decision_index) < 0:
        raise ValueError("temporal state identity requires a non-negative decision")
    fingerprint = str(state_fingerprint).lower()
    if len(fingerprint) != 64 or any(
        character not in string.hexdigits.lower() for character in fingerprint
    ):
        raise ValueError("temporal state identity requires a SHA-256 state fingerprint")
    payload = {
        "schema": TEMPORAL_STATE_SCHEMA,
        "episode_id": str(episode_id),
        "decision_index": int(decision_index),
        "state_fingerprint": fingerprint,
        "history_context_sha256": history.sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return {
        **payload,
        "state_occurrence_id": hashlib.sha256(encoded).hexdigest(),
        "history_context": history.payload(),
    }


__all__ = [
    "TEMPORAL_STATE_SCHEMA",
    "TemporalHistoryContext",
    "temporal_history_context",
    "temporal_state_identity",
]
