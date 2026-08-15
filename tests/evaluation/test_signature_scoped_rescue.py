from __future__ import annotations

import pytest

from lns2_selector.runtime.signature_scoped_rescue import (
    SignatureScopedRescueTracker,
)


def _state(version: int = 0) -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": False,
        "rows": 2,
        "cols": 3,
        "sum_of_costs": 6,
        "num_of_colliding_pairs": 1,
        "obstacles": [0, 0, 0, 0, 0, 0],
        "conflict_edges": [[0, 1]],
        "agents": [
            {"id": 0, "path": [0, 1 if version == 0 else 3]},
            {"id": 1, "path": [1, 0]},
            {"id": 2, "path": [2, 5]},
            {"id": 3, "path": [3, 4]},
        ],
    }


def _metrics(
    seed: int,
    *,
    neighborhood: list[int] | None = None,
    reason: str = "conflict_bound_exceeded",
) -> dict:
    agents = list(neighborhood or [0, 1])
    return {
        "requested_random_seed": seed,
        "requested_pp_random_seed": seed,
        "applied_pp_random_seed": seed,
        "requested_collect_pp_diagnostics": True,
        "neighborhood": agents,
        "repair_order": list(reversed(agents)),
        "replan_success": reason == "none",
        "pp_failure_reason": reason,
        "pp_attempted_agent_count": len(agents),
        "pp_inserted_agent_count": max(0, len(agents) - 1),
        "pp_failed_agent": agents[0] if reason != "none" else -1,
        "pp_failed_order_index": len(agents) - 1 if reason != "none" else -1,
        "pp_rolled_back": reason != "none",
        "conflicts_after": 1,
        "pp_agent_diagnostics": [
            {
                "agent_id": agent,
                "order_index": index,
                "external_blocker_agents": [2, 3] if index == 0 else [],
            }
            for index, agent in enumerate(reversed(agents))
        ],
    }


def _tracker() -> SignatureScopedRescueTracker:
    return SignatureScopedRescueTracker.from_spec(
        _state(),
        {
            "maximum_added_blockers": 8,
            "minimum_consecutive_rollbacks": 3,
            "maximum_interventions": 3,
            "initial_repeat_count": 2,
            "seed_namespace": "test",
            "episode_key": "episode",
            "trial_index": 0,
            "enable_semantic_compaction_audit": True,
        },
    )


def test_rescue_is_deferred_and_same_signature_cannot_repeat() -> None:
    tracker = _tracker()
    trigger = tracker.observe_decision(
        decision_index=0,
        before=_state(),
        after=_state(),
        metrics=_metrics(11),
    )
    assert trigger["scheduled"] is True
    assert trigger["selected_blockers"] == [2, 3]
    assert tracker.action_for_decision(0, _state()) is None
    action = tracker.action_for_decision(1, _state())
    assert action is not None
    failed = tracker.observe_decision(
        decision_index=1,
        before=_state(),
        after=_state(),
        metrics=_metrics(
            int(action["pp_random_seed"]), neighborhood=list(action["agents"])
        ),
    )
    assert failed["intervention_executed"] is True
    assert failed["resolved_by_rescue"] is False
    assert failed["scheduled"] is False
    assert tracker.action_for_decision(2, _state()) is None
    assert tracker.summary()["intervention_count"] == 1


def test_changed_state_must_form_new_platform_before_second_rescue() -> None:
    tracker = _tracker()
    tracker.observe_decision(
        decision_index=0,
        before=_state(),
        after=_state(),
        metrics=_metrics(11),
    )
    first = tracker.action_for_decision(1, _state())
    assert first is not None
    resolved = tracker.observe_decision(
        decision_index=1,
        before=_state(),
        after=_state(1),
        metrics=_metrics(
            int(first["pp_random_seed"]),
            neighborhood=list(first["agents"]),
            reason="none",
        ),
    )
    assert resolved["resolved_by_rescue"] is True
    for decision in (2, 3):
        row = tracker.observe_decision(
            decision_index=decision,
            before=_state(1),
            after=_state(1),
            metrics=_metrics(20 + decision),
        )
        assert row["scheduled"] is False
    third = tracker.observe_decision(
        decision_index=4,
        before=_state(1),
        after=_state(1),
        metrics=_metrics(24),
    )
    assert third["scheduled"] is True
    assert tracker.action_for_decision(5, _state(1)) is not None
    assert tracker.summary()["new_signature_intervention_count"] == 1


def test_time_limit_does_not_form_platform() -> None:
    tracker = _tracker()
    row = tracker.observe_decision(
        decision_index=0,
        before=_state(),
        after=_state(),
        metrics=_metrics(11, reason="time_limit"),
    )
    assert row["scheduled"] is False
    assert tracker.summary()["intervention_count"] == 0


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("maximum_added_blockers", 7),
        ("minimum_consecutive_rollbacks", 2),
        ("maximum_interventions", 4),
        ("initial_repeat_count", 1),
        ("enable_semantic_compaction_audit", False),
    ),
)
def test_frozen_limits_cannot_change(field: str, value: object) -> None:
    spec = {
        "maximum_added_blockers": 8,
        "minimum_consecutive_rollbacks": 3,
        "maximum_interventions": 3,
        "initial_repeat_count": 2,
        "seed_namespace": "test",
        "episode_key": "episode",
        "trial_index": 0,
        "enable_semantic_compaction_audit": True,
    }
    spec[field] = value
    with pytest.raises(ValueError):
        SignatureScopedRescueTracker.from_spec(_state(), spec)
