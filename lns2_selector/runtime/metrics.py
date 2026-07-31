from __future__ import annotations

import math


def wall_clock_conflict_auc(
    trajectory: list[int],
    transition_elapsed_seconds: list[float],
    budget_seconds: float,
) -> float:
    """Integrate all observed conflicts through one shared wall-clock deadline."""

    if (
        not trajectory
        or len(transition_elapsed_seconds) != len(trajectory) - 1
        or not math.isfinite(float(budget_seconds))
        or float(budget_seconds) <= 0.0
    ):
        raise ValueError("invalid wall-clock conflict trajectory")
    budget = float(budget_seconds)
    previous_time = 0.0
    current_conflicts = int(trajectory[0])
    area = 0.0
    for elapsed, after_conflicts in zip(
        transition_elapsed_seconds, trajectory[1:]
    ):
        event_time = float(elapsed)
        if not math.isfinite(event_time) or event_time < previous_time:
            raise ValueError("wall-clock transition times must be finite and ordered")
        clipped = min(event_time, budget)
        area += current_conflicts * max(0.0, clipped - previous_time)
        if event_time > budget:
            return area
        previous_time = event_time
        current_conflicts = int(after_conflicts)
    area += current_conflicts * max(0.0, budget - previous_time)
    return area
