from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from lns2_selector.runtime.rollback_aware_selection import (
    is_exact_conflict_bound_rollback,
    ranked_candidate_indices,
)


OVERALL_ROLLBACK_SELECTION_ID = "stride-exact-rollback-state-guard-v1"


@dataclass
class _RepairFingerprintState:
    pure_structshell_exact_rollbacks: int = 0
    structshell_suppressed: bool = False
    cache_reuse_allowed: bool = False
    last_selected_candidate_id: str | None = None


@dataclass
class ExactRollbackStateGuard:
    """Bound StructShell work across a complete repair-state fingerprint.

    Unlike the candidate-level guard, the budget is shared by every pure
    StructShell candidate selected at the same repair fingerprint.  Exhausting
    the budget latches StructShell off for that fingerprint.  History is kept
    per fingerprint, so visiting another repair state and later returning does
    not reopen the structural scan.

    ``select`` deliberately returns ``None`` after suppression.  That is a
    request for the caller to run its ordinary, fresh V2-only proposal and
    selection path; choosing a cached anchor from the combined pool would not
    be an equivalent fallback.
    """

    exact_rollback_limit: int = 3
    active_repair_fingerprint: str | None = None
    _history: dict[str, _RepairFingerprintState] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.exact_rollback_limit < 1:
            raise ValueError("exact rollback limit must be positive")

    def _state(
        self,
        repair_fingerprint: str,
        *,
        activate: bool = True,
    ) -> _RepairFingerprintState:
        fingerprint = str(repair_fingerprint)
        if activate:
            self.active_repair_fingerprint = fingerprint
        return self._history.setdefault(fingerprint, _RepairFingerprintState())

    @staticmethod
    def _pure_structshell_ids(
        *,
        pure_structshell_candidate_ids: Iterable[str] | None,
        bannable_candidate_ids: Iterable[str] | None,
    ) -> set[str]:
        pure = (
            None
            if pure_structshell_candidate_ids is None
            else set(map(str, pure_structshell_candidate_ids))
        )
        compatible = (
            None
            if bannable_candidate_ids is None
            else set(map(str, bannable_candidate_ids))
        )
        if pure is not None and compatible is not None and pure != compatible:
            raise ValueError(
                "pure StructShell and bannable candidate IDs must agree"
            )
        return pure if pure is not None else (compatible or set())

    @property
    def cache_reuse_allowed(self) -> bool:
        """Whether the active combined pool is safe to reuse once more."""

        if self.active_repair_fingerprint is None:
            return False
        return self._state(
            self.active_repair_fingerprint,
            activate=False,
        ).cache_reuse_allowed

    def structshell_suppressed(self, repair_fingerprint: str) -> bool:
        """Return whether StructShell is latched off at this repair state."""

        return self._state(
            repair_fingerprint,
            activate=False,
        ).structshell_suppressed

    def requires_fresh_v2_fallback(self, repair_fingerprint: str) -> bool:
        """Return whether StructShell is latched off at this repair state."""

        return self.structshell_suppressed(repair_fingerprint)

    def snapshot(self, repair_fingerprint: str) -> dict[str, Any]:
        """Return a serialisable diagnostic without exposing mutable state."""

        fingerprint = str(repair_fingerprint)
        state = self._state(fingerprint, activate=False)
        return {
            "guard_id": OVERALL_ROLLBACK_SELECTION_ID,
            "repair_fingerprint": fingerprint,
            "pure_structshell_exact_rollbacks": (
                state.pure_structshell_exact_rollbacks
            ),
            "state_exact_rollbacks": state.pure_structshell_exact_rollbacks,
            "exact_rollback_limit": self.exact_rollback_limit,
            "structshell_suppressed": state.structshell_suppressed,
            "fresh_v2_fallback_required": state.structshell_suppressed,
            "cache_reuse_allowed": state.cache_reuse_allowed,
            "last_selected_candidate_id": state.last_selected_candidate_id,
            "tracked_repair_fingerprint_count": len(self._history),
        }

    def select(
        self,
        *,
        repair_fingerprint: str,
        candidates: Sequence[Mapping[str, Any]],
        candidate_rows: Sequence[Mapping[str, Any]],
        scores: Sequence[float],
        pure_structshell_candidate_ids: Iterable[str] | None = None,
        bannable_candidate_ids: Iterable[str] | None = None,
        v2_anchor_candidate_id: str | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        """Use the frozen ranking, or request a fresh V2-only fallback.

        A ``None`` candidate index is intentional and must not be replaced by
        a cached V2 anchor.  The caller should generate and score a new V2-only
        pool through the same path used by the frozen V2 controller.
        """

        if len(candidates) != len(candidate_rows):
            raise ValueError("candidate records and rows must be aligned")
        candidate_ids = [
            str(candidate["candidate_id"]) for candidate in candidates
        ]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        pure_structshell = self._pure_structshell_ids(
            pure_structshell_candidate_ids=pure_structshell_candidate_ids,
            bannable_candidate_ids=bannable_candidate_ids,
        )
        anchor_id = (
            None
            if v2_anchor_candidate_id is None
            else str(v2_anchor_candidate_id)
        )
        if anchor_id is not None and anchor_id not in candidate_ids:
            raise ValueError("V2 anchor is missing from the candidate pool")
        if anchor_id is not None and anchor_id in pure_structshell:
            raise ValueError("V2 anchor cannot be a pure StructShell candidate")
        state = self._state(repair_fingerprint)
        order = ranked_candidate_indices(candidate_rows, scores)
        base_index = order[0]
        base_candidate_id = candidate_ids[base_index]

        if state.structshell_suppressed:
            selected_index: int | None = None
            selected_candidate_id: str | None = None
            selection_phase = "fresh_v2_fallback_required"
        else:
            selected_index = base_index
            selected_candidate_id = base_candidate_id
            selection_phase = "frozen_base_ranking"

        diagnostic = self.snapshot(repair_fingerprint)
        diagnostic.update(
            {
                "base_selected_candidate_id": base_candidate_id,
                "selected_candidate_id": selected_candidate_id,
                "selected_candidate_is_pure_structshell": (
                    selected_candidate_id in pure_structshell
                    if selected_candidate_id is not None
                    else False
                ),
                "pure_structshell_candidate_count": len(
                    pure_structshell.intersection(candidate_ids)
                ),
                "v2_anchor_candidate_id": anchor_id,
                "selection_phase": selection_phase,
                "selection_overridden": selected_index != base_index,
                "newly_suppressed": False,
            }
        )
        return selected_index, diagnostic

    def observe(
        self,
        *,
        before_repair_fingerprint: str,
        after_repair_fingerprint: str,
        selected_candidate_id: str,
        metrics: Mapping[str, Any],
        pure_structshell_candidate_ids: Iterable[str] | None = None,
        bannable_candidate_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Record one PP result without ever reopening exhausted state.

        Only exact ``conflict_bound_exceeded`` native rollbacks from a pure
        StructShell candidate consume the budget.  Time limits, successful or
        otherwise non-exact same-state outcomes, and V2 outcomes preserve the
        accumulated count and the suppression latch.
        """

        before = str(before_repair_fingerprint)
        after = str(after_repair_fingerprint)
        selected_id = str(selected_candidate_id)
        state = self._state(before)
        pure_structshell = self._pure_structshell_ids(
            pure_structshell_candidate_ids=pure_structshell_candidate_ids,
            bannable_candidate_ids=bannable_candidate_ids,
        )
        exact_rollback = is_exact_conflict_bound_rollback(
            metrics,
            before_repair_fingerprint=before,
            after_repair_fingerprint=after,
        )
        selected_is_pure_structshell = selected_id in pure_structshell
        counted = False
        newly_suppressed = False

        if (
            exact_rollback
            and selected_is_pure_structshell
            and not state.structshell_suppressed
        ):
            state.pure_structshell_exact_rollbacks += 1
            state.last_selected_candidate_id = selected_id
            counted = True
            if (
                state.pure_structshell_exact_rollbacks
                >= self.exact_rollback_limit
            ):
                state.pure_structshell_exact_rollbacks = self.exact_rollback_limit
                state.structshell_suppressed = True
                state.cache_reuse_allowed = False
                newly_suppressed = True
            else:
                state.cache_reuse_allowed = True
        else:
            # A non-exact result may make a cached combined pool stale.  It
            # must not, however, erase the cumulative budget or suppression.
            state.cache_reuse_allowed = False

        # Move the active pointer without deleting either fingerprint's
        # history.  This is what makes leave-and-return suppression persistent.
        self._state(after)
        diagnostic = self.snapshot(before)
        diagnostic.update(
            {
                "after_repair_fingerprint": after,
                "selected_candidate_id": selected_id,
                "selected_candidate_is_pure_structshell": (
                    selected_is_pure_structshell
                ),
                "exact_conflict_bound_rollback": exact_rollback,
                "rollback_counted": counted,
                "newly_suppressed": newly_suppressed,
                # This policy never bans an individual candidate.  Keep the
                # legacy key for trace consumers, but do not double-count a
                # state-level suppression as a candidate-level ban.
                "newly_banned": False,
            }
        )
        return diagnostic


__all__ = [
    "ExactRollbackStateGuard",
    "OVERALL_ROLLBACK_SELECTION_ID",
]
