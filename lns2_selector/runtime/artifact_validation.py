from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


POST_STRUCTURE_FIELDS = frozenset(
    {
        "post_largest_component_ratio",
        "post_conflict_edge_density",
        "post_event_density",
        "post_degree_concentration",
    }
)


def finite_number(value: Any, *, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and (minimum is None or numeric >= minimum)


def strict_integer(value: Any, *, minimum: int | None = None) -> bool:
    return type(value) is int and (minimum is None or value >= minimum)


def candidate_records(
    rows: Any, *, require_agents: bool = True
) -> dict[str, dict[str, Any]] | None:
    """Validate JSON candidate identities and return their unique ID mapping."""

    if not isinstance(rows, list) or not rows:
        return None
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            return None
        raw_id = row.get("candidate_id")
        if not isinstance(raw_id, str) or not raw_id or raw_id in result:
            return None
        if require_agents:
            agents = row.get("agents")
            if (
                not isinstance(agents, list)
                or not agents
                or any(not strict_integer(agent, minimum=0) for agent in agents)
                or len(agents) != len(set(agents))
            ):
                return None
            if "actual_size" in row and (
                not strict_integer(row["actual_size"], minimum=1)
                or int(row["actual_size"]) != len(agents)
            ):
                return None
        result[raw_id] = row
    return result


def trial_product_matches(
    rows: Any,
    *,
    candidate_ids: Sequence[str],
    trial_indices: Sequence[int],
    candidate_field: str = "candidate_id",
) -> bool:
    if not isinstance(rows, list):
        return False
    normalized_candidates = tuple(map(str, candidate_ids))
    normalized_indices = tuple(trial_indices)
    if (
        not normalized_candidates
        or len(set(normalized_candidates)) != len(normalized_candidates)
        or not normalized_indices
        or any(not strict_integer(value, minimum=0) for value in normalized_indices)
        or len(set(normalized_indices)) != len(normalized_indices)
    ):
        return False
    expected = {
        (candidate_id, trial_index)
        for candidate_id in normalized_candidates
        for trial_index in normalized_indices
    }
    observed: list[tuple[str, int]] = []
    for row in rows:
        if not isinstance(row, dict):
            return False
        candidate = row.get(candidate_field)
        trial_index = row.get("trial_index")
        if not isinstance(candidate, str) or not strict_integer(
            trial_index, minimum=0
        ):
            return False
        observed.append((candidate, trial_index))
    return len(observed) == len(expected) and set(observed) == expected


def _post_structure_valid(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == POST_STRUCTURE_FIELDS
        and all(finite_number(metric, minimum=0.0) for metric in value.values())
    )


def repair_trial_semantics_valid(
    row: Any,
    *,
    schema: str | None,
    state_id: str | None,
    candidate_id: str | None,
    trial_index: int,
    pp_seed: int,
    before_conflicts: int,
    before_repair_fingerprint: str,
    before_fingerprint: str | None = None,
    feature_schema_id: str | None = None,
    required_feature_names: Sequence[str] | set[str] | frozenset[str] | None = None,
    expected_metadata: Mapping[str, Any] | None = None,
    conflicts_before_field: str = "before_conflicts",
    require_after_fingerprint: bool = True,
) -> bool:
    """Validate one stored native repair result without replaying the solver."""

    if not isinstance(row, dict):
        return False
    if schema is not None and row.get("schema") != schema:
        return False
    if state_id is not None and row.get("state_id") != state_id:
        return False
    if candidate_id is not None and row.get("candidate_id") != candidate_id:
        return False
    if row.get("trial_index") != trial_index or row.get("pp_seed") != pp_seed:
        return False
    if (
        not strict_integer(before_conflicts, minimum=1)
        or row.get(conflicts_before_field) != before_conflicts
        or row.get("before_repair_fingerprint") != before_repair_fingerprint
    ):
        return False
    if before_fingerprint is not None and row.get("before_fingerprint") != before_fingerprint:
        return False
    if feature_schema_id is not None and row.get("feature_schema_id") != feature_schema_id:
        return False
    if expected_metadata is not None and any(
        row.get(name) != value for name, value in expected_metadata.items()
    ):
        return False
    conflicts_after = row.get("conflicts_after")
    feasible = row.get("feasible")
    replan_success = row.get("replan_success")
    after_repair = row.get("after_repair_fingerprint")
    if (
        not strict_integer(conflicts_after, minimum=0)
        or type(feasible) is not bool
        or type(replan_success) is not bool
        or not isinstance(after_repair, str)
        or not after_repair
        or not _post_structure_valid(row.get("post_structure"))
        or not finite_number(row.get("native_step_seconds"), minimum=0.0)
        or not finite_number(row.get("pp_replan_seconds"), minimum=0.0)
    ):
        return False
    if require_after_fingerprint and (
        not isinstance(row.get("after_fingerprint"), str)
        or not row["after_fingerprint"]
    ):
        return False
    if required_feature_names is not None:
        features = row.get("features")
        if (
            not isinstance(features, dict)
            or set(features) != set(required_feature_names)
            or any(not finite_number(value) for value in features.values())
        ):
            return False
    try:
        expected_outcome = classify_repair_outcome(
            before_fingerprint=before_repair_fingerprint,
            after_fingerprint=after_repair,
            replan_success=replan_success,
            conflicts_before=before_conflicts,
            conflicts_after=conflicts_after,
            feasible=feasible,
        )
    except (TypeError, ValueError):
        return False
    return row.get("repair_outcome") == expected_outcome


__all__ = [
    "POST_STRUCTURE_FIELDS",
    "candidate_records",
    "finite_number",
    "repair_trial_semantics_valid",
    "strict_integer",
    "trial_product_matches",
]
