from __future__ import annotations

import pytest

from lns2_selector.runtime.bounded_native_retry import (
    BoundedNativeRetryTracker,
    attempt_snapshot,
    merged_retry_metrics,
)


def _state(*, changed: bool = False) -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": False,
        "rows": 2,
        "cols": 2,
        "sum_of_costs": 4,
        "num_of_colliding_pairs": 1,
        "obstacles": [],
        "conflict_edges": [[0, 1]],
        "agents": [
            {"id": 0, "path": [0, 1 if not changed else 2]},
            {"id": 1, "path": [1, 0]},
        ],
    }


def _metrics(seed: int, *, reason: str = "conflict_bound_exceeded") -> dict:
    return {
        "requested_random_seed": seed,
        "requested_pp_random_seed": seed,
        "applied_pp_random_seed": seed,
        "requested_collect_pp_diagnostics": True,
        "neighborhood": [0, 1],
        "repair_order": [1, 0],
        "replan_success": reason == "none",
        "pp_failure_reason": reason,
        "pp_attempted_agent_count": 2,
        "pp_inserted_agent_count": 1,
        "pp_failed_agent": 0 if reason != "none" else -1,
        "pp_failed_order_index": 1 if reason != "none" else -1,
        "pp_rolled_back": reason != "none",
        "conflicts_after": 1,
        "pp_agent_diagnostics": [],
    }


def _tracker(*, enabled: bool = True) -> BoundedNativeRetryTracker:
    return BoundedNativeRetryTracker.from_spec(
        _state(),
        {
            "enabled": enabled,
            "minimum_consecutive_rollbacks": 3,
            "maximum_interventions": 3,
            "initial_repeat_count": 2,
            "seed_namespace": "test",
            "episode_key": "episode",
            "trial_index": 0,
            "first_retry_seed": 22,
        },
    )


def test_first_registered_exact_rollback_triggers_and_changed_retry_resolves() -> None:
    tracker = _tracker()
    record = tracker.observe_first_attempt(
        before=_state(), after=_state(), metrics=_metrics(11), decision_index=0
    )
    assert record["triggered"] is True
    assert record["retry_seed"] == 22
    record = tracker.observe_retry(
        record,
        before=_state(),
        after=_state(changed=True),
        metrics=_metrics(22, reason="none"),
    )
    tracker.finalize_decision(record)
    assert record["resolved_by_retry"] is True
    assert tracker.summary()["persistent_platform_seen"] is False


def test_same_platform_signature_is_never_retried_twice() -> None:
    tracker = _tracker()
    first = tracker.observe_first_attempt(
        before=_state(), after=_state(), metrics=_metrics(11), decision_index=0
    )
    first = tracker.observe_retry(
        first, before=_state(), after=_state(), metrics=_metrics(22)
    )
    tracker.finalize_decision(first)
    second = tracker.observe_first_attempt(
        before=_state(), after=_state(), metrics=_metrics(33), decision_index=1
    )
    tracker.finalize_decision(second)
    assert second["trigger_eligible"] is False
    assert second["triggered"] is False
    assert tracker.summary()["intervention_count"] == 1
    assert tracker.summary()["persistent_platform_seen"] is True


def test_time_limit_and_baseline_never_retry() -> None:
    treatment = _tracker()
    timed = treatment.observe_first_attempt(
        before=_state(),
        after=_state(),
        metrics=_metrics(11, reason="time_limit"),
        decision_index=0,
    )
    assert timed["triggered"] is False
    baseline = _tracker(enabled=False)
    eligible = baseline.observe_first_attempt(
        before=_state(), after=_state(), metrics=_metrics(11), decision_index=0
    )
    assert eligible["trigger_eligible"] is True
    assert eligible["triggered"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("enabled", "false"),
        ("minimum_consecutive_rollbacks", "3"),
        ("trial_index", True),
        ("first_retry_seed", "22"),
        ("seed_namespace", 7),
        ("episode_key", False),
    ),
)
def test_retry_spec_rejects_type_coercion(field: str, value: object) -> None:
    specification = {
        "enabled": True,
        "minimum_consecutive_rollbacks": 3,
        "maximum_interventions": 3,
        "initial_repeat_count": 2,
        "seed_namespace": "test",
        "episode_key": "episode",
        "trial_index": 0,
        "first_retry_seed": 22,
    }
    specification[field] = value
    with pytest.raises(ValueError):
        BoundedNativeRetryTracker.from_spec(_state(), specification)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("requested_collect_pp_diagnostics", "false"),
        ("replan_success", 0),
        ("pp_attempted_agent_count", True),
        ("neighborhood", [0, "1"]),
        ("pp_agent_diagnostics", [False]),
    ),
)
def test_attempt_snapshot_rejects_type_coercion(field: str, value: object) -> None:
    metrics = _metrics(11)
    metrics[field] = value
    with pytest.raises(ValueError):
        attempt_snapshot(metrics)


def test_cancelled_retry_releases_signature_and_intervention_slot() -> None:
    tracker = _tracker()
    record = tracker.observe_first_attempt(
        before=_state(), after=_state(), metrics=_metrics(11), decision_index=0
    )
    record = tracker.cancel_retry(record, reason="episode_wall_budget_exhausted")
    tracker.finalize_decision(record)
    assert record["triggered"] is False
    assert record["retry_skipped_reason"] == "episode_wall_budget_exhausted"
    assert tracker.summary()["intervention_count"] == 0
    assert tracker.summary()["used_platform_signature_count"] == 0
    assert tracker.summary()["persistent_platform_seen"] is True


def test_merged_retry_metrics_charge_both_native_calls() -> None:
    first = _metrics(11)
    retry = _metrics(22, reason="none")
    first.update(
        {
            "native_step_seconds": 1.25,
            "binding_solver_call_seconds": 1.0,
            "binding_total_seconds": 1.5,
        }
    )
    retry.update(
        {
            "native_step_seconds": 2.5,
            "binding_solver_call_seconds": 2.0,
            "binding_total_seconds": 2.5,
        }
    )
    merged = merged_retry_metrics(first, retry, {"triggered": True})
    assert merged["native_step_seconds"] == 3.75
    assert merged["binding_solver_call_seconds"] == 3.0
    assert merged["binding_total_seconds"] == 4.0
    assert merged["bounded_native_retry"] == {"triggered": True}
