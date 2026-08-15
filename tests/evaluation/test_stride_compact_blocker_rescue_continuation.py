from __future__ import annotations

from experiments.stride_compact_blocker_rescue_continuation import (
    ARMS,
    _episode_override,
    compact_blocker_schedule,
    prepare_factorial_cases,
)
from lns2_selector.runtime.failure_informed_rescue import (
    BLOCKER_AUGMENTED_MODE,
    COMPACT_BLOCKER_AUGMENTED_MODE,
    COMPACT_SAME_SET_MODE,
    CONTROL_MODE,
    SAME_SET_MODE,
    FailureInformedRescueTracker,
)


CONFIG = "configs/stride_compact_blocker_rescue_continuation_v1_registration.json"


def _state(*, changed: bool = False) -> dict:
    return {
        "initialized": True,
        "initial_solution_complete": True,
        "feasible": False,
        "rows": 2,
        "cols": 4,
        "sum_of_costs": 8,
        "num_of_colliding_pairs": 1,
        "obstacles": [0] * 8,
        "conflict_edges": [[0, 1]],
        "agents": [
            {"id": 0, "start": 0, "goal": 2, "path": [0, 1, 2]},
            {
                "id": 1,
                "start": 2,
                "goal": 0,
                "path": [2, 1, 0 if not changed else 3],
            },
            {"id": 2, "start": 4, "goal": 6, "path": [4, 5, 6]},
            {"id": 3, "start": 7, "goal": 7, "path": [7, 7, 7]},
        ],
    }


def _metrics(seed: int, agents: list[int], *, success: bool = False) -> dict:
    diagnostics = []
    for index, agent in enumerate(reversed(agents)):
        diagnostics.append(
            {
                "agent_id": agent,
                "order_index": index,
                "external_blocker_agents": [3] if index == 0 else [],
            }
        )
    return {
        "requested_random_seed": seed,
        "requested_pp_random_seed": seed,
        "applied_pp_random_seed": seed,
        "requested_collect_pp_diagnostics": True,
        "neighborhood": agents,
        "repair_order": list(reversed(agents)),
        "replan_success": success,
        "pp_failure_reason": "none" if success else "conflict_bound_exceeded",
        "pp_attempted_agent_count": len(agents),
        "pp_inserted_agent_count": len(agents) if success else len(agents) - 1,
        "pp_failed_agent": -1 if success else agents[0],
        "pp_failed_order_index": -1 if success else len(agents) - 1,
        "pp_rolled_back": not success,
        "conflicts_after": 0 if success else 1,
        "pp_agent_diagnostics": diagnostics,
    }


def _tracker(mode: str) -> FailureInformedRescueTracker:
    return FailureInformedRescueTracker.from_spec(
        {
            "mode": mode,
            "maximum_added_blockers": 8,
            "seed_namespace": "compact-blocker-test",
            "episode_key": "episode",
            "trial_index": 0,
            "initial_repeat_count": 2,
            "enable_semantic_compaction_audit": True,
        }
    )


def test_factorial_membership_changes_one_rescue_call_only() -> None:
    actions = {}
    for mode in (
        SAME_SET_MODE,
        COMPACT_SAME_SET_MODE,
        BLOCKER_AUGMENTED_MODE,
        COMPACT_BLOCKER_AUGMENTED_MODE,
    ):
        tracker = _tracker(mode)
        record = tracker.observe_decision(
            decision_index=0,
            before=_state(),
            after=_state(),
            metrics=_metrics(11, [0, 1, 2]),
        )
        assert record is not None and record["trigger_eligible"] is True
        assert record["compact_plan"]["removed_agents"] == [2]
        action = tracker.action_for_decision(1, _state())
        assert action is not None
        actions[mode] = action

    assert actions[SAME_SET_MODE]["agents"] == [0, 1, 2]
    assert actions[COMPACT_SAME_SET_MODE]["agents"] == [0, 1]
    assert actions[BLOCKER_AUGMENTED_MODE]["agents"] == [0, 1, 2, 3]
    assert actions[COMPACT_BLOCKER_AUGMENTED_MODE]["agents"] == [0, 1, 3]
    assert len({actions[arm]["pp_random_seed"] for arm in actions}) == 1
    for action in actions.values():
        assert "repair_order" not in action


def test_compact_blocker_rescue_records_membership_and_executes_once() -> None:
    tracker = _tracker(COMPACT_BLOCKER_AUGMENTED_MODE)
    tracker.observe_decision(
        decision_index=0,
        before=_state(),
        after=_state(),
        metrics=_metrics(11, [0, 1, 2]),
    )
    action = tracker.action_for_decision(1, _state())
    assert action is not None
    record = tracker.observe_decision(
        decision_index=1,
        before=_state(),
        after=_state(changed=True),
        metrics=_metrics(int(action["pp_random_seed"]), [0, 1, 3], success=True),
    )
    assert record is not None and record["resolved_by_rescue"] is True
    assert record["selected_blockers"] == [3]
    assert record["removed_agents"] == [2]
    assert tracker.summary()["rescue_executed"] is True


def test_control_audits_compaction_but_never_rescues() -> None:
    tracker = _tracker(CONTROL_MODE)
    record = tracker.observe_decision(
        decision_index=0,
        before=_state(),
        after=_state(),
        metrics=_metrics(11, [0, 1, 2]),
    )
    assert record is not None and record["compact_plan"]["eligible"] is True
    assert record["triggered"] is False
    assert tracker.action_for_decision(1, _state()) is None


def test_registration_freezes_five_paired_arms_and_one_call() -> None:
    _loaded, cases = prepare_factorial_cases(CONFIG)
    assert len(cases) == 45
    schedule = compact_blocker_schedule(cases, (0, 1, 2, 3))
    assert len(schedule) == 45 * 4 * 5
    assert ARMS == (
        CONTROL_MODE,
        SAME_SET_MODE,
        COMPACT_SAME_SET_MODE,
        BLOCKER_AUGMENTED_MODE,
        COMPACT_BLOCKER_AUGMENTED_MODE,
    )
    first = schedule[: len(ARMS)]
    assert {row["arm"] for row in first} == set(ARMS)
    assert len({row["first_pp_seed"] for row in first}) == 1
    override = _episode_override(
        cases[0], trial_index=0, arm=COMPACT_BLOCKER_AUGMENTED_MODE
    )
    assert override["failure_informed_rescue"][
        "enable_semantic_compaction_audit"
    ] is True
    assert "repair_order" not in override["forced_first_action"]
    assert "bounded_native_retry" not in override
