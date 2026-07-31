from __future__ import annotations


REPAIR_OUTCOMES = (
    "hard_failure",
    "accepted_noop",
    "state_changed_no_reduction",
    "conflict_reduced",
    "feasible",
)


def classify_repair_outcome(
    *,
    before_fingerprint: str,
    after_fingerprint: str,
    replan_success: bool,
    conflicts_before: int,
    conflicts_after: int,
    feasible: bool = False,
) -> str:
    """Classify one native repair without embedding a controller policy."""

    before_conflicts = int(conflicts_before)
    after_conflicts = int(conflicts_after)
    changed = str(before_fingerprint) != str(after_fingerprint)
    if not replan_success and changed:
        raise ValueError("failed PP changed the solver state instead of rolling back")
    if not changed and before_conflicts != after_conflicts:
        raise ValueError("unchanged state fingerprint has different conflict counts")
    if feasible or after_conflicts == 0:
        if not changed and before_conflicts > 0:
            raise ValueError("an unchanged conflicting state cannot become feasible")
        return "feasible"
    if not replan_success:
        return "hard_failure"
    if not changed:
        return "accepted_noop"
    if after_conflicts < before_conflicts:
        return "conflict_reduced"
    return "state_changed_no_reduction"
