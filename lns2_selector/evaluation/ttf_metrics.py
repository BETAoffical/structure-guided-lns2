"""Semantic checks for reset-inclusive, deadline-capped TTF summaries."""

from __future__ import annotations

import math
from typing import Any


_REQUIRED_FIELDS = frozenset(
    {
        "success",
        "external_timeout",
        "truncated",
        "stop_reason",
        "final_conflicts",
        "repair_iterations",
        "transition_elapsed_seconds",
        "initial_state_elapsed_seconds",
        "ttf_clock_schema",
        "wall_time_budget_seconds",
        "wall_time_to_feasible",
        "capped_wall_time_to_feasible",
        "ttf_observed_wall_seconds",
        "episode_observed_wall_seconds",
        "reset_wall_seconds",
        "repair_wall_seconds",
    }
)
_STOP_REASONS = frozenset(
    {"success", "controller_stalled", "repair_limit", "wall_timeout", "native_terminal", "truncated"}
)
_TIME_SERIES = frozenset(
    {"transition_elapsed_seconds", "transition_trace_write_seconds"}
)


def _time(value: Any, name: str, *, positive: bool = False) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"TTF summary {name} must be numeric")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"TTF summary {name} must be finite and {qualifier}")
    return value


def _at_most(value: int | float, limit: int | float, name: str) -> None:
    # Independent perf_counter differences and their sums can round slightly.
    if value > limit and not math.isclose(value, limit, rel_tol=1e-9, abs_tol=1e-8):
        raise ValueError(f"TTF summary {name} exceeds its enclosing clock")


def validate_ttf_summary(
    summary: Any, *, wall_time_budget_seconds: float | None = None
) -> dict:
    """Validate a timed episode summary without coercing or replacing values.

    The optional budget is the caller's registered deadline. A failed episode
    may have zero final conflicts when native repair becomes feasible after
    that deadline; it is still a timeout with no successful TTF observation.
    This validates summary metrics, not the underlying trace or its identity.
    """

    if not isinstance(summary, dict):
        raise ValueError("TTF summary must be an object")
    missing = _REQUIRED_FIELDS - summary.keys()
    if missing:
        raise ValueError(f"TTF summary is missing fields: {sorted(missing)}")
    for name in ("success", "external_timeout", "truncated"):
        if type(summary[name]) is not bool:
            raise ValueError(f"TTF summary {name} must be boolean")
    for name, value in summary.items():
        if isinstance(name, str) and (
            name.endswith(("_count", "_iterations"))
            or name in {"initial_conflicts", "final_conflicts", "budget_final_conflicts", "repair_iterations_within_budget"}
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"TTF summary {name} must be a nonnegative integer")
    final_conflicts = summary["final_conflicts"]
    if summary["ttf_clock_schema"] != "lns2.ttf.reset_inclusive_wall.v1":
        raise ValueError("TTF summary ttf_clock_schema is unsupported")
    reason = summary["stop_reason"]
    if not isinstance(reason, str) or reason not in _STOP_REASONS:
        raise ValueError("TTF summary stop_reason is invalid")

    budget = _time(summary["wall_time_budget_seconds"], "wall_time_budget_seconds", positive=True)
    if wall_time_budget_seconds is not None:
        expected = _time(wall_time_budget_seconds, "registered wall_time_budget_seconds", positive=True)
        if budget != expected:
            raise ValueError("TTF summary wall_time_budget_seconds differs from registration")
    capped = _time(summary["capped_wall_time_to_feasible"], "capped_wall_time_to_feasible")
    if capped > budget:
        raise ValueError("TTF summary capped_wall_time_to_feasible exceeds budget")
    for name, value in summary.items():
        if isinstance(name, str) and name.endswith("_seconds") and name not in _TIME_SERIES:
            _time(value, name)
    for name in ("reset_timings", "controller_totals"):
        if name not in summary:
            continue
        if not isinstance(summary[name], dict):
            raise ValueError(f"TTF summary {name} must be an object")
        for field, value in summary[name].items():
            if isinstance(field, str) and field.endswith("_seconds"):
                _time(value, f"{name}.{field}")
    for name in _TIME_SERIES:
        if name not in summary:
            continue
        values = summary[name]
        if not isinstance(values, list):
            raise ValueError(f"TTF summary {name} must be a list")
        previous = summary["initial_state_elapsed_seconds"] if name == "transition_elapsed_seconds" else 0
        for value in values:
            value = _time(value, name)
            if name == "transition_elapsed_seconds":
                if value < previous:
                    raise ValueError("TTF summary transition times must follow reset and be ordered")
                _at_most(value, summary["ttf_observed_wall_seconds"], name)
                previous = value

    observed = summary["ttf_observed_wall_seconds"]
    episode_observed = summary["episode_observed_wall_seconds"]
    reset = summary["reset_wall_seconds"]
    repair = summary["repair_wall_seconds"]
    _at_most(observed, episode_observed, "TTF observation")
    _at_most(reset + repair, observed, "reset plus repair time")
    initial = summary["initial_state_elapsed_seconds"]
    _at_most(reset, initial, "reset time")
    _at_most(initial, observed, "initial state time")
    transitions = summary["transition_elapsed_seconds"]
    if summary["repair_iterations"] != len(transitions):
        raise ValueError("TTF summary repair_iterations differs from transition count")
    algorithm_elapsed = transitions[-1] if transitions else initial
    if "environment_construct_seconds" in summary:
        _at_most(summary["environment_construct_seconds"], episode_observed, "environment construction")

    success = summary["success"]
    external_timeout = summary["external_timeout"]
    if summary["truncated"] is success or success != (reason == "success"):
        raise ValueError("TTF summary success, truncated and stop_reason disagree")
    if success:
        ttf = _time(summary["wall_time_to_feasible"], "wall_time_to_feasible")
        if final_conflicts != 0 or external_timeout or ttf > budget:
            raise ValueError("TTF summary successful result has conflicts or exceeds its deadline")
        if ttf != algorithm_elapsed:
            raise ValueError("TTF summary successful TTF differs from final algorithm elapsed time")
        expected_capped = min(ttf, budget)
        _at_most(ttf, observed, "successful TTF")
        _at_most(reset + repair, ttf, "successful reset plus repair time")
        _at_most(initial, ttf, "initial state time")
    else:
        if summary["wall_time_to_feasible"] is not None:
            raise ValueError("TTF summary failed result must not contain a successful TTF")
        expected_capped = budget
        if algorithm_elapsed >= budget and not external_timeout:
            raise ValueError("TTF summary failed deadline completion requires external_timeout")
        if final_conflicts == 0 and (not external_timeout or algorithm_elapsed <= budget):
            raise ValueError("TTF summary failed zero-conflict result must finish after its deadline")
    if capped != expected_capped:
        raise ValueError("TTF summary capped_wall_time_to_feasible is inconsistent")
    if external_timeout:
        if reason not in {"wall_timeout", "controller_stalled", "repair_limit"}:
            raise ValueError("TTF summary external_timeout and stop_reason disagree")
        _at_most(budget, observed, "timeout deadline")
    elif reason == "wall_timeout":
        raise ValueError("TTF summary wall_timeout requires external_timeout")
    if "native_time_to_feasible" in summary:
        native_ttf = summary["native_time_to_feasible"]
        if success:
            _time(native_ttf, "native_time_to_feasible")
        elif native_ttf is not None:
            raise ValueError("TTF summary failed result must not contain a native TTF")
    return summary


__all__ = ["validate_ttf_summary"]
