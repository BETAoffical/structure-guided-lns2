from __future__ import annotations


def fixed_budget_conflict_auc(
    trajectory: list[int], budget: int, *, success: bool
) -> float:
    """Read the historical fixed-repair window without limiting execution."""

    if budget <= 0 or not trajectory:
        raise ValueError("invalid fixed-budget conflict trajectory")
    values = list(map(int, trajectory[: budget + 1]))
    pad = 0 if success else values[-1]
    values.extend([pad] * (budget + 1 - len(values)))
    return sum(
        (values[index] + values[index + 1]) / 2.0
        for index in range(budget)
    )
