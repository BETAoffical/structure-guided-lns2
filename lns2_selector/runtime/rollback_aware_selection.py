from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


ROLLBACK_AWARE_SELECTION_ID = "stride-exact-rollback-candidate-guard-v1"


def is_exact_conflict_bound_rollback(
    metrics: Mapping[str, Any],
    *,
    before_repair_fingerprint: str,
    after_repair_fingerprint: str,
) -> bool:
    """Return whether native PP atomically restored the repair structure."""

    return bool(
        metrics.get("pp_failure_reason") == "conflict_bound_exceeded"
        and metrics.get("replan_success") is False
        and metrics.get("pp_rolled_back") is True
        and before_repair_fingerprint == after_repair_fingerprint
    )


def ranked_candidate_indices(
    candidate_rows: Sequence[Mapping[str, Any]], scores: Sequence[float]
) -> list[int]:
    if len(candidate_rows) != len(scores) or not candidate_rows:
        raise ValueError("candidate rows and scores must be non-empty and aligned")
    return sorted(
        range(len(candidate_rows)),
        key=lambda index: (
            -round(float(scores[index]), 12),
            str(candidate_rows[index]["candidate_key"]),
        ),
    )


@dataclass
class ExactRollbackCandidateGuard:
    """Suppress a repeatedly rolled-back Hybrid challenger without another PP.

    Bans are scoped to one repair-structure fingerprint and are cleared as soon
    as the paths or conflicts change.  The frozen V2 anchor is never removed;
    it is the deterministic fallback if no unbanned challenger remains.
    """

    exact_rollback_limit: int = 3
    repair_fingerprint: str | None = None
    last_candidate_id: str | None = None
    consecutive_exact_rollbacks: int = 0
    banned_candidate_ids: set[str] = field(default_factory=set)
    cache_reuse_allowed: bool = False

    def __post_init__(self) -> None:
        if self.exact_rollback_limit < 1:
            raise ValueError("exact rollback limit must be positive")

    def prepare(self, repair_fingerprint: str) -> None:
        fingerprint = str(repair_fingerprint)
        if self.repair_fingerprint == fingerprint:
            return
        self.repair_fingerprint = fingerprint
        self.last_candidate_id = None
        self.consecutive_exact_rollbacks = 0
        self.banned_candidate_ids.clear()
        self.cache_reuse_allowed = False

    def select(
        self,
        *,
        repair_fingerprint: str,
        candidates: Sequence[Mapping[str, Any]],
        candidate_rows: Sequence[Mapping[str, Any]],
        scores: Sequence[float],
        v2_anchor_candidate_id: str,
        bannable_candidate_ids: Iterable[str],
    ) -> tuple[int, dict[str, Any]]:
        self.prepare(repair_fingerprint)
        if len(candidates) != len(candidate_rows):
            raise ValueError("candidate records and rows must be aligned")
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        anchor_id = str(v2_anchor_candidate_id)
        if anchor_id not in candidate_ids:
            raise ValueError("V2 anchor is missing from the candidate pool")
        bannable = set(map(str, bannable_candidate_ids))
        if anchor_id in bannable:
            raise ValueError("V2 anchor cannot be a bannable challenger")
        order = ranked_candidate_indices(candidate_rows, scores)
        base_index = order[0]
        unbanned_challengers = {
            candidate_id
            for candidate_id in bannable
            if candidate_id in candidate_ids
            and candidate_id not in self.banned_candidate_ids
        }
        fallback_used = bool(self.banned_candidate_ids and not unbanned_challengers)
        if fallback_used:
            selected_index = candidate_ids.index(anchor_id)
            selection_phase = "v2_anchor_fallback"
        elif self.banned_candidate_ids:
            # Once a structural challenger has crossed the rollback limit,
            # finish the bounded structural scan before falling back to V2.
            # Otherwise an ordinary V2 candidate between two challengers in
            # the frozen ranking could become an unbannable cached attractor.
            selected_index = next(
                index
                for index in order
                if candidate_ids[index] in unbanned_challengers
            )
            selection_phase = "remaining_structshell_challenger"
        else:
            selected_index = base_index
            selection_phase = "frozen_base_ranking"
        selected_id = candidate_ids[selected_index]
        return selected_index, {
            "guard_id": ROLLBACK_AWARE_SELECTION_ID,
            "repair_fingerprint": str(repair_fingerprint),
            "base_selected_candidate_id": candidate_ids[base_index],
            "selected_candidate_id": selected_id,
            "selection_overridden": selected_index != base_index,
            "v2_anchor_candidate_id": anchor_id,
            "v2_anchor_fallback_used": fallback_used,
            "selection_phase": selection_phase,
            "selected_candidate_is_bannable": selected_id in bannable,
            "bannable_candidate_count": len(bannable),
            "banned_candidate_ids": sorted(self.banned_candidate_ids),
            "consecutive_exact_rollbacks": self.consecutive_exact_rollbacks,
            "cache_reuse_allowed": self.cache_reuse_allowed,
        }

    def observe(
        self,
        *,
        before_repair_fingerprint: str,
        after_repair_fingerprint: str,
        selected_candidate_id: str,
        metrics: Mapping[str, Any],
        bannable_candidate_ids: Iterable[str],
    ) -> dict[str, Any]:
        before = str(before_repair_fingerprint)
        after = str(after_repair_fingerprint)
        selected_id = str(selected_candidate_id)
        self.prepare(before)
        exact_rollback = is_exact_conflict_bound_rollback(
            metrics,
            before_repair_fingerprint=before,
            after_repair_fingerprint=after,
        )
        bannable = set(map(str, bannable_candidate_ids))
        newly_banned = False
        fallback_cycle_reset = False
        if exact_rollback:
            if selected_id in bannable:
                if self.last_candidate_id == selected_id:
                    self.consecutive_exact_rollbacks += 1
                else:
                    self.last_candidate_id = selected_id
                    self.consecutive_exact_rollbacks = 1
                self.cache_reuse_allowed = True
                if (
                    self.consecutive_exact_rollbacks >= self.exact_rollback_limit
                    and selected_id not in self.banned_candidate_ids
                ):
                    self.banned_candidate_ids.add(selected_id)
                    newly_banned = True
            else:
                # Never freeze a V2 candidate or anchor.  If the bounded
                # structural scan exhausted itself and the anchor also rolled
                # back, reopen the scan on the next decision instead of
                # manufacturing an unbounded anchor loop.
                fallback_cycle_reset = bool(self.banned_candidate_ids)
                self.last_candidate_id = None
                self.consecutive_exact_rollbacks = 0
                self.banned_candidate_ids.clear()
                self.cache_reuse_allowed = False
        else:
            if before != after:
                self.prepare(after)
            else:
                self.last_candidate_id = None
                self.consecutive_exact_rollbacks = 0
                self.cache_reuse_allowed = False
        return {
            "guard_id": ROLLBACK_AWARE_SELECTION_ID,
            "exact_conflict_bound_rollback": exact_rollback,
            "selected_candidate_id": selected_id,
            "consecutive_exact_rollbacks": self.consecutive_exact_rollbacks,
            "newly_banned": newly_banned,
            "fallback_cycle_reset": fallback_cycle_reset,
            "banned_candidate_ids": sorted(self.banned_candidate_ids),
            "cache_reuse_allowed": self.cache_reuse_allowed,
        }


__all__ = [
    "ExactRollbackCandidateGuard",
    "ROLLBACK_AWARE_SELECTION_ID",
    "is_exact_conflict_bound_rollback",
    "ranked_candidate_indices",
]
