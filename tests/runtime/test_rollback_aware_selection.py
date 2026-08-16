from __future__ import annotations

from lns2_selector.runtime.rollback_aware_selection import (
    ExactRollbackCandidateGuard,
    is_exact_conflict_bound_rollback,
)


def _rollback_metrics() -> dict:
    return {
        "pp_failure_reason": "conflict_bound_exceeded",
        "replan_success": False,
        "pp_rolled_back": True,
    }


def test_exact_rollback_requires_native_failure_and_identical_repair_state() -> None:
    assert is_exact_conflict_bound_rollback(
        _rollback_metrics(),
        before_repair_fingerprint="same",
        after_repair_fingerprint="same",
    )
    changed = _rollback_metrics()
    changed["pp_rolled_back"] = False
    assert not is_exact_conflict_bound_rollback(
        changed,
        before_repair_fingerprint="same",
        after_repair_fingerprint="same",
    )
    assert not is_exact_conflict_bound_rollback(
        _rollback_metrics(),
        before_repair_fingerprint="before",
        after_repair_fingerprint="after",
    )


def test_three_exact_rollbacks_ban_only_the_structural_challenger() -> None:
    guard = ExactRollbackCandidateGuard(exact_rollback_limit=3)
    candidates = [
        {"candidate_id": "struct"},
        {"candidate_id": "v2-other"},
        {"candidate_id": "v2-anchor"},
    ]
    rows = [
        {"candidate_key": "struct"},
        {"candidate_key": "v2-other"},
        {"candidate_key": "v2-anchor"},
    ]
    scores = [3.0, 2.0, 1.0]
    for expected_streak in (1, 2, 3):
        selected, _ = guard.select(
            repair_fingerprint="repair-a",
            candidates=candidates,
            candidate_rows=rows,
            scores=scores,
            v2_anchor_candidate_id="v2-anchor",
            bannable_candidate_ids={"struct"},
        )
        assert selected == 0
        observation = guard.observe(
            before_repair_fingerprint="repair-a",
            after_repair_fingerprint="repair-a",
            selected_candidate_id="struct",
            metrics=_rollback_metrics(),
            bannable_candidate_ids={"struct"},
        )
        assert observation["consecutive_exact_rollbacks"] == expected_streak
    assert observation["newly_banned"]
    assert observation["cache_reuse_allowed"]

    selected, diagnostic = guard.select(
        repair_fingerprint="repair-a",
        candidates=candidates,
        candidate_rows=rows,
        scores=scores,
        v2_anchor_candidate_id="v2-anchor",
        bannable_candidate_ids={"struct"},
    )
    assert selected == 2
    assert diagnostic["selection_overridden"]
    assert diagnostic["v2_anchor_fallback_used"]
    assert diagnostic["banned_candidate_ids"] == ["struct"]


def test_v2_anchor_is_retained_and_state_change_clears_guard_state() -> None:
    guard = ExactRollbackCandidateGuard(exact_rollback_limit=1)
    candidates = [{"candidate_id": "struct"}, {"candidate_id": "v2-anchor"}]
    rows = [{"candidate_key": "struct"}, {"candidate_key": "v2-anchor"}]
    guard.observe(
        before_repair_fingerprint="repair-a",
        after_repair_fingerprint="repair-a",
        selected_candidate_id="struct",
        metrics=_rollback_metrics(),
        bannable_candidate_ids={"struct"},
    )
    selected, diagnostic = guard.select(
        repair_fingerprint="repair-a",
        candidates=candidates,
        candidate_rows=rows,
        scores=[2.0, 1.0],
        v2_anchor_candidate_id="v2-anchor",
        bannable_candidate_ids={"struct"},
    )
    assert selected == 1
    assert diagnostic["selected_candidate_id"] == "v2-anchor"

    selected, diagnostic = guard.select(
        repair_fingerprint="repair-b",
        candidates=candidates,
        candidate_rows=rows,
        scores=[2.0, 1.0],
        v2_anchor_candidate_id="v2-anchor",
        bannable_candidate_ids={"struct"},
    )
    assert selected == 0
    assert diagnostic["banned_candidate_ids"] == []
    assert not diagnostic["cache_reuse_allowed"]


def test_non_exact_noop_does_not_authorize_repair_state_cache_reuse() -> None:
    guard = ExactRollbackCandidateGuard(exact_rollback_limit=3)
    guard.observe(
        before_repair_fingerprint="repair-a",
        after_repair_fingerprint="repair-a",
        selected_candidate_id="struct",
        metrics={
            "pp_failure_reason": "none",
            "replan_success": True,
            "pp_rolled_back": False,
        },
        bannable_candidate_ids={"struct"},
    )
    assert not guard.cache_reuse_allowed
    assert guard.consecutive_exact_rollbacks == 0


def test_v2_rollback_never_authorizes_cache_or_candidate_ban() -> None:
    guard = ExactRollbackCandidateGuard(exact_rollback_limit=1)
    candidates = [{"candidate_id": "v2-winner"}, {"candidate_id": "v2-anchor"}]
    rows = [{"candidate_key": "v2-winner"}, {"candidate_key": "v2-anchor"}]
    selected, diagnostic = guard.select(
        repair_fingerprint="repair-a",
        candidates=candidates,
        candidate_rows=rows,
        scores=[2.0, 1.0],
        v2_anchor_candidate_id="v2-anchor",
        bannable_candidate_ids=set(),
    )
    assert selected == 0
    assert diagnostic["selection_phase"] == "frozen_base_ranking"
    observation = guard.observe(
        before_repair_fingerprint="repair-a",
        after_repair_fingerprint="repair-a",
        selected_candidate_id="v2-winner",
        metrics=_rollback_metrics(),
        bannable_candidate_ids=set(),
    )
    assert observation["exact_conflict_bound_rollback"]
    assert not observation["cache_reuse_allowed"]
    assert observation["banned_candidate_ids"] == []


def test_structural_scan_does_not_freeze_an_intervening_v2_candidate() -> None:
    guard = ExactRollbackCandidateGuard(exact_rollback_limit=1)
    candidates = [
        {"candidate_id": "struct-a"},
        {"candidate_id": "v2-other"},
        {"candidate_id": "struct-b"},
        {"candidate_id": "v2-anchor"},
    ]
    rows = [{"candidate_key": value["candidate_id"]} for value in candidates]
    guard.observe(
        before_repair_fingerprint="repair-a",
        after_repair_fingerprint="repair-a",
        selected_candidate_id="struct-a",
        metrics=_rollback_metrics(),
        bannable_candidate_ids={"struct-a", "struct-b"},
    )
    selected, diagnostic = guard.select(
        repair_fingerprint="repair-a",
        candidates=candidates,
        candidate_rows=rows,
        scores=[4.0, 3.0, 2.0, 1.0],
        v2_anchor_candidate_id="v2-anchor",
        bannable_candidate_ids={"struct-a", "struct-b"},
    )
    assert selected == 2
    assert diagnostic["selection_phase"] == "remaining_structshell_challenger"


def test_failed_v2_anchor_reopens_structural_scan_without_cache_reuse() -> None:
    guard = ExactRollbackCandidateGuard(exact_rollback_limit=1)
    candidates = [{"candidate_id": "struct"}, {"candidate_id": "v2-anchor"}]
    rows = [{"candidate_key": "struct"}, {"candidate_key": "v2-anchor"}]
    guard.observe(
        before_repair_fingerprint="repair-a",
        after_repair_fingerprint="repair-a",
        selected_candidate_id="struct",
        metrics=_rollback_metrics(),
        bannable_candidate_ids={"struct"},
    )
    selected, _diagnostic = guard.select(
        repair_fingerprint="repair-a",
        candidates=candidates,
        candidate_rows=rows,
        scores=[2.0, 1.0],
        v2_anchor_candidate_id="v2-anchor",
        bannable_candidate_ids={"struct"},
    )
    assert selected == 1
    observation = guard.observe(
        before_repair_fingerprint="repair-a",
        after_repair_fingerprint="repair-a",
        selected_candidate_id="v2-anchor",
        metrics=_rollback_metrics(),
        bannable_candidate_ids={"struct"},
    )
    assert observation["fallback_cycle_reset"]
    assert observation["banned_candidate_ids"] == []
    assert not observation["cache_reuse_allowed"]
    selected, diagnostic = guard.select(
        repair_fingerprint="repair-a",
        candidates=candidates,
        candidate_rows=rows,
        scores=[2.0, 1.0],
        v2_anchor_candidate_id="v2-anchor",
        bannable_candidate_ids={"struct"},
    )
    assert selected == 0
    assert diagnostic["selection_phase"] == "frozen_base_ranking"
