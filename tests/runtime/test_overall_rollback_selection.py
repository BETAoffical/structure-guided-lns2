from __future__ import annotations

from lns2_selector.runtime.overall_rollback_selection import (
    OVERALL_ROLLBACK_SELECTION_ID,
    ExactRollbackStateGuard,
)


def _rollback_metrics() -> dict:
    return {
        "pp_failure_reason": "conflict_bound_exceeded",
        "replan_success": False,
        "pp_rolled_back": True,
    }


def _pool() -> tuple[list[dict], list[dict], list[float]]:
    candidates = [
        {"candidate_id": "struct-a"},
        {"candidate_id": "struct-b"},
        {"candidate_id": "v2"},
    ]
    rows = [{"candidate_key": row["candidate_id"]} for row in candidates]
    return candidates, rows, [3.0, 2.0, 1.0]


def _observe(
    guard: ExactRollbackStateGuard,
    candidate_id: str,
    *,
    before: str = "repair-a",
    after: str = "repair-a",
    metrics: dict | None = None,
) -> dict:
    return guard.observe(
        before_repair_fingerprint=before,
        after_repair_fingerprint=after,
        selected_candidate_id=candidate_id,
        metrics=_rollback_metrics() if metrics is None else metrics,
        pure_structshell_candidate_ids={"struct-a", "struct-b", "struct-c"},
    )


def test_state_budget_accumulates_across_structshell_candidate_ids() -> None:
    guard = ExactRollbackStateGuard(exact_rollback_limit=3)

    first = _observe(guard, "struct-a")
    second = _observe(guard, "struct-b")
    third = _observe(guard, "struct-c")

    assert first["pure_structshell_exact_rollbacks"] == 1
    assert second["pure_structshell_exact_rollbacks"] == 2
    assert third["pure_structshell_exact_rollbacks"] == 3
    assert third["newly_suppressed"]
    assert not third["newly_banned"]
    assert third["structshell_suppressed"]
    assert not third["cache_reuse_allowed"]
    assert third["guard_id"] == OVERALL_ROLLBACK_SELECTION_ID


def test_select_uses_frozen_ranking_then_signals_fresh_v2_fallback() -> None:
    guard = ExactRollbackStateGuard(exact_rollback_limit=1)
    candidates, rows, scores = _pool()

    selected, diagnostic = guard.select(
        repair_fingerprint="repair-a",
        candidates=candidates,
        candidate_rows=rows,
        scores=scores,
        pure_structshell_candidate_ids={"struct-a", "struct-b"},
    )
    assert selected == 0
    assert diagnostic["selection_phase"] == "frozen_base_ranking"
    assert not diagnostic["selection_overridden"]

    _observe(guard, "struct-a")
    selected, diagnostic = guard.select(
        repair_fingerprint="repair-a",
        candidates=candidates,
        candidate_rows=rows,
        scores=scores,
        pure_structshell_candidate_ids={"struct-a", "struct-b"},
    )
    assert selected is None
    assert diagnostic["fresh_v2_fallback_required"]
    assert diagnostic["selection_phase"] == "fresh_v2_fallback_required"
    assert diagnostic["selected_candidate_id"] is None


def test_v2_exact_rollback_never_reopens_suppressed_structshell() -> None:
    guard = ExactRollbackStateGuard(exact_rollback_limit=1)
    _observe(guard, "struct-a")

    observation = _observe(guard, "v2")

    assert observation["exact_conflict_bound_rollback"]
    assert not observation["rollback_counted"]
    assert observation["pure_structshell_exact_rollbacks"] == 1
    assert observation["structshell_suppressed"]
    assert guard.requires_fresh_v2_fallback("repair-a")


def test_leave_and_return_preserves_latch_while_new_fingerprint_is_independent(
) -> None:
    guard = ExactRollbackStateGuard(exact_rollback_limit=1)
    _observe(guard, "struct-a")

    _observe(
        guard,
        "v2",
        before="repair-a",
        after="repair-b",
        metrics={
            "pp_failure_reason": "none",
            "replan_success": True,
            "pp_rolled_back": False,
        },
    )
    assert not guard.requires_fresh_v2_fallback("repair-b")
    _observe(
        guard,
        "v2",
        before="repair-b",
        after="repair-a",
        metrics={
            "pp_failure_reason": "none",
            "replan_success": True,
            "pp_rolled_back": False,
        },
    )
    assert guard.requires_fresh_v2_fallback("repair-a")
    assert not guard.requires_fresh_v2_fallback("repair-b")


def test_nonexact_same_state_preserves_accumulated_budget() -> None:
    guard = ExactRollbackStateGuard(exact_rollback_limit=3)
    _observe(guard, "struct-a")
    _observe(guard, "struct-b")

    observation = _observe(
        guard,
        "struct-a",
        metrics={
            "pp_failure_reason": "none",
            "replan_success": True,
            "pp_rolled_back": False,
        },
    )
    assert observation["pure_structshell_exact_rollbacks"] == 2
    assert not observation["rollback_counted"]
    assert not observation["cache_reuse_allowed"]

    final = _observe(guard, "struct-b")
    assert final["pure_structshell_exact_rollbacks"] == 3
    assert final["structshell_suppressed"]


def test_time_limit_does_not_consume_or_reset_budget() -> None:
    guard = ExactRollbackStateGuard(exact_rollback_limit=3)
    _observe(guard, "struct-a")

    observation = _observe(
        guard,
        "struct-b",
        metrics={
            "pp_failure_reason": "time_limit",
            "replan_success": False,
            "pp_rolled_back": True,
        },
    )

    assert not observation["exact_conflict_bound_rollback"]
    assert not observation["rollback_counted"]
    assert observation["pure_structshell_exact_rollbacks"] == 1
    assert not observation["structshell_suppressed"]


def test_fingerprint_histories_accumulate_independently() -> None:
    guard = ExactRollbackStateGuard(exact_rollback_limit=2)
    _observe(guard, "struct-a", before="repair-a", after="repair-a")
    _observe(guard, "struct-a", before="repair-b", after="repair-b")
    _observe(guard, "struct-b", before="repair-a", after="repair-a")

    assert guard.snapshot("repair-a")["pure_structshell_exact_rollbacks"] == 2
    assert guard.snapshot("repair-a")["structshell_suppressed"]
    assert guard.snapshot("repair-b")["pure_structshell_exact_rollbacks"] == 1
    assert not guard.snapshot("repair-b")["structshell_suppressed"]
    assert guard.snapshot("repair-a")["tracked_repair_fingerprint_count"] == 2


def test_old_guard_argument_names_and_active_cache_property_are_supported() -> None:
    guard = ExactRollbackStateGuard(exact_rollback_limit=3)
    candidates, rows, scores = _pool()
    selected, _ = guard.select(
        repair_fingerprint="repair-a",
        candidates=candidates,
        candidate_rows=rows,
        scores=scores,
        v2_anchor_candidate_id="v2",
        bannable_candidate_ids={"struct-a", "struct-b"},
    )
    assert selected == 0
    observation = guard.observe(
        before_repair_fingerprint="repair-a",
        after_repair_fingerprint="repair-a",
        selected_candidate_id="struct-a",
        metrics=_rollback_metrics(),
        bannable_candidate_ids={"struct-a", "struct-b"},
    )
    assert observation["state_exact_rollbacks"] == 1
    assert guard.cache_reuse_allowed
    assert not guard.structshell_suppressed("repair-a")
