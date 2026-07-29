from __future__ import annotations

from experiments.stall_confirmation_rule_audit import evaluate_no_progress_run


def _failure(index: int) -> dict:
    return {
        "attempt_key": f"attempt-{index}",
        "actual_lns2": {"outcome": {"repair_seconds": 1.0}},
    }


def test_recovery_inside_window_is_premature() -> None:
    result = evaluate_no_progress_run(
        [_failure(index) for index in range(5)],
        {"no_progress": False},
        threshold=3,
        minimum_distinct_attempts=2,
        future_horizon=3,
    )
    assert result is not None
    assert result["recovery_delay_decisions"] == 3
    assert result["resolution"] == "premature_v2_self_recovery"


def test_long_unchanged_window_is_confirmed() -> None:
    result = evaluate_no_progress_run(
        [_failure(index) for index in range(8)],
        None,
        threshold=3,
        minimum_distinct_attempts=2,
        future_horizon=3,
    )
    assert result is not None
    assert result["resolution"] == "confirmed_no_change_window"


def test_terminal_short_window_is_unresolved() -> None:
    result = evaluate_no_progress_run(
        [_failure(index) for index in range(4)],
        None,
        threshold=3,
        minimum_distinct_attempts=2,
        future_horizon=3,
    )
    assert result is not None
    assert result["resolution"] == "right_censored_unresolved"


def test_repeated_identical_attempts_do_not_confirm_stall() -> None:
    repeated = _failure(1)
    result = evaluate_no_progress_run(
        [dict(repeated) for _ in range(6)],
        None,
        threshold=4,
        minimum_distinct_attempts=2,
        future_horizon=3,
    )
    assert result is None


def test_recovery_after_window_is_confirmed_not_premature() -> None:
    result = evaluate_no_progress_run(
        [_failure(index) for index in range(7)],
        {"no_progress": False},
        threshold=3,
        minimum_distinct_attempts=2,
        future_horizon=3,
    )
    assert result is not None
    assert result["recovery_delay_decisions"] == 5
    assert result["resolution"] == "confirmed_no_change_window"
