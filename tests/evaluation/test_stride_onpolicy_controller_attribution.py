from __future__ import annotations

from experiments.stride_onpolicy_controller_attribution import (
    ADAPTIVE_ARM,
    ARMS,
    SLOT_ARM,
    STRUCT_ARM,
    TARGET_ARM,
    _episode_override,
    _platform_diagnostics,
    continuation_schedule,
    prepare_tasks,
    run_collection,
)


CONFIG = "configs/stride_onpolicy_controller_attribution_v1_registration.json"
R2_CONFIG = "configs/stride_onpolicy_controller_attribution_v1_r2_registration.json"


def _row(index: int, before: str, after: str, *, rollback: bool) -> dict:
    return {
        "decision_index": index,
        "before_platform_signature": before,
        "after_platform_signature": after,
        "actual_metrics": {
            "pp_failure_reason": "conflict_bound_exceeded" if rollback else "none",
            "replan_success": not rollback,
            "pp_rolled_back": rollback,
        },
    }


def test_registration_collapses_to_unique_initial_task_keys() -> None:
    loaded, tasks = prepare_tasks(CONFIG)
    assert loaded[2]["execution"]["worker_count"] == 16
    assert len(tasks) == 19
    assert len({(row["task_id"], row["solver_seed"]) for row in tasks}) == 19
    assert len(continuation_schedule(tasks, range(4))) == 304
    assert ARMS == (SLOT_ARM, STRUCT_ARM, ADAPTIVE_ARM, TARGET_ARM)


def test_initial_override_changes_trial_seed_without_restoring_a_state() -> None:
    _loaded, tasks = prepare_tasks(CONFIG)
    trial0 = _episode_override(tasks[0], trial_index=0)
    trial1 = _episode_override(tasks[0], trial_index=1)
    assert "initial_restore" not in trial0
    assert "forced_first_action" not in trial0
    assert "bounded_native_retry" not in trial0
    assert trial0["pp_replay_seed_salt"] != trial1["pp_replay_seed_salt"]


def test_r2_replays_the_registered_decision_zero_path() -> None:
    loaded, tasks = prepare_tasks(R2_CONFIG)
    assert loaded[2]["execution"]["initial_path_replay"] is True
    assert len(tasks) == 19
    task = tasks[0]
    assert task["initial_source_case_fingerprint"]
    assert task["initial_state_fingerprint"]

    override = _episode_override(
        task,
        trial_index=0,
        source_cases=list(loaded[4][-1]),
    )
    restore = override["initial_restore"]
    assert restore["decision_index"] == 0
    assert restore["expected_fingerprint"] == task["initial_state_fingerprint"]
    assert restore["expected_conflicts"] > 0
    assert "forced_first_action" not in override
    assert "bounded_native_retry" not in override


def test_r2_dry_run_is_a_full_restart_with_sixteen_workers() -> None:
    result = run_collection(
        R2_CONFIG,
        "build/test-stride-onpolicy-controller-attribution-r2-dry-run",
        phase="initial",
        dry_run=True,
    )
    assert result["task_count"] == 19
    assert result["schedule_entry_count"] == 304
    assert result["worker_count"] == 16


def test_platform_diagnostics_start_with_zero_historical_streak() -> None:
    two_rollbacks = _platform_diagnostics(
        [_row(index, "a", "a", rollback=True) for index in range(2)]
    )
    assert two_rollbacks["entered_platform"] is False

    entered = _platform_diagnostics(
        [_row(index, "a", "a", rollback=True) for index in range(3)]
    )
    assert entered["entered_platform"] is True
    assert entered["first_platform_entry_decision"] == 3

    escaped_then_reentered = _platform_diagnostics(
        [
            *[_row(index, "a", "a", rollback=True) for index in range(3)],
            _row(3, "a", "b", rollback=False),
            *[_row(index, "b", "b", rollback=True) for index in range(4, 7)],
        ]
    )
    assert escaped_then_reentered["escaped_first_platform"] is True
    assert escaped_then_reentered["post_escape_platform_reentry"] is True


def test_dry_run_uses_four_arms_and_sixteen_workers() -> None:
    result = run_collection(
        CONFIG,
        "build/test-stride-onpolicy-controller-attribution-dry-run",
        phase="initial",
        dry_run=True,
        limit_tasks=1,
    )
    assert result["task_count"] == 1
    assert result["schedule_entry_count"] == 16
    assert result["worker_count"] == 16
