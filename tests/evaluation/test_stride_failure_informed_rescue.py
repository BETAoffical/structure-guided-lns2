from __future__ import annotations

from lns2_selector.runtime.failure_informed_rescue import (
    BLOCKER_AUGMENTED_MODE,
    CONTROL_MODE,
    SAME_SET_MODE,
    FailureInformedRescueTracker,
)
from experiments.stride_failure_informed_rescue_continuation import (
    ARMS,
    _episode_override,
    continuation_schedule,
    prepare_cases,
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
            {"id": 2, "path": [2, 3]},
            {"id": 3, "path": [3, 2]},
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
        "pp_agent_diagnostics": [
            {
                "agent_id": 1,
                "order_index": 0,
                "external_blocker_agents": [2, 3],
            },
            {
                "agent_id": 0,
                "order_index": 1,
                "external_blocker_agents": [3],
            },
        ],
    }


def _tracker(mode: str) -> FailureInformedRescueTracker:
    return FailureInformedRescueTracker.from_spec(
        {
            "mode": mode,
            "maximum_added_blockers": 8,
            "seed_namespace": "test",
            "episode_key": "episode",
            "trial_index": 0,
            "initial_repeat_count": 2,
        }
    )


def test_blocker_rescue_is_deferred_to_next_decision() -> None:
    tracker = _tracker(BLOCKER_AUGMENTED_MODE)
    trigger = tracker.observe_decision(
        decision_index=0,
        before=_state(),
        after=_state(),
        metrics=_metrics(11),
    )
    assert trigger is not None and trigger["triggered"] is True
    assert trigger["selected_blockers"] == [2, 3]
    assert tracker.action_for_decision(0, _state()) is None
    action = tracker.action_for_decision(1, _state())
    assert action is not None
    assert action["agents"] == [0, 1, 2, 3]
    rescue = tracker.observe_decision(
        decision_index=1,
        before=_state(),
        after=_state(changed=True),
        metrics={
            **_metrics(int(action["pp_random_seed"]), reason="none"),
            "neighborhood": [0, 2, 1, 3],
        },
    )
    assert rescue is not None and rescue["resolved_by_rescue"] is True
    assert tracker.summary()["rescue_executed"] is True


def test_same_set_rescue_uses_no_blockers_and_control_never_overrides() -> None:
    same = _tracker(SAME_SET_MODE)
    trigger = same.observe_decision(
        decision_index=0,
        before=_state(),
        after=_state(),
        metrics=_metrics(11),
    )
    assert trigger is not None and trigger["observed_external_blockers"] == [2, 3]
    assert trigger["selected_blockers"] == []
    assert same.action_for_decision(1, _state())["agents"] == [0, 1]

    control = _tracker(CONTROL_MODE)
    record = control.observe_decision(
        decision_index=0,
        before=_state(),
        after=_state(),
        metrics=_metrics(11),
    )
    assert record is not None and record["trigger_eligible"] is True
    assert record["triggered"] is False
    assert control.action_for_decision(1, _state()) is None


def test_time_limit_never_schedules_rescue() -> None:
    tracker = _tracker(BLOCKER_AUGMENTED_MODE)
    record = tracker.observe_decision(
        decision_index=0,
        before=_state(),
        after=_state(),
        metrics=_metrics(11, reason="time_limit"),
    )
    assert record is not None and record["triggered"] is False
    assert tracker.action_for_decision(1, _state()) is None


def test_registered_schedule_and_override_keep_one_native_call_per_decision() -> None:
    _loaded, cases = prepare_cases(
        "configs/stride_failure_informed_rescue_continuation_v1_registration.json"
    )
    assert len(cases) == 45
    schedule = continuation_schedule(cases, (0, 1, 2, 3))
    assert len(schedule) == 45 * 4 * len(ARMS)
    first_triplet = schedule[: len(ARMS)]
    assert {row["arm"] for row in first_triplet} == set(ARMS)
    assert len({row["first_pp_seed"] for row in first_triplet}) == 1
    override = _episode_override(cases[0], trial_index=0, arm=BLOCKER_AUGMENTED_MODE)
    assert "repair_order" not in override["forced_first_action"]
    assert "bounded_native_retry" not in override
    assert override["failure_informed_rescue"]["maximum_added_blockers"] == 8
