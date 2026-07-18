from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable


FEATURE_SCHEMA_ID = "lns2.realized_features.v2"
FEATURE_SCHEMA_VERSION = 2

PROPOSAL_FAMILIES = tuple(
    f"{heuristic}:{size}"
    for heuristic in ("collision", "random", "target")
    for size in (4, 8, 16)
)
ACTUAL_SIZES = tuple(range(1, 17))
AGGREGATES = ("mean", "std", "min", "max", "sum")

STATE_FEATURE_NAMES_V1 = (
    "state.agent_count",
    "state.iteration",
    "state.colliding_pairs",
    "state.conflict_edge_density",
    "state.conflict_event_count",
    "state.vertex_event_ratio",
    "state.conflicting_agent_ratio",
    "state.component_count",
    "state.largest_component",
    "state.largest_component_ratio",
    "state.degree_mean",
    "state.degree_std",
    "state.degree_max",
    "state.delay_mean",
    "state.delay_std",
    "state.delay_max",
    "state.path_cost_mean",
    "state.path_cost_std",
    "state.path_stretch_mean",
    "state.path_wait_ratio_mean",
    "state.conflict_time_mean",
    "state.conflict_time_std",
    "state.sum_of_costs_per_agent",
    "state.low_level_generated_per_agent",
    "state.low_level_runs_per_agent",
)

PROPOSAL_FEATURE_NAMES_V1 = tuple(
    sorted(
        {
            "proposal.actual_size",
            "proposal.actual_size_ratio_agents",
            "proposal.total_count",
            "proposal.unique_proposal_seed_count",
            "proposal.seed_agent_count",
            "proposal.selection_family_count",
            "proposal.support_family_count",
            *(f"proposal.actual_size={size}" for size in ACTUAL_SIZES),
            *(f"proposal.selection_family={family}" for family in PROPOSAL_FAMILIES),
            *(f"proposal.family_count={family}" for family in PROPOSAL_FAMILIES),
            *(f"proposal.family_ratio={family}" for family in PROPOSAL_FAMILIES),
            *(
                f"proposal.{group}_{aggregate}"
                for group in (
                    "seed_conflict_degree",
                    "seed_delay",
                    "seed_path_cost",
                    "seed_component_size",
                )
                for aggregate in AGGREGATES
            ),
        }
    )
)

REALIZED_FEATURE_NAMES_V1 = tuple(
    sorted(
        {
            "realized.actual_size",
            "realized.actual_size_ratio_agents",
            "realized.conflicting_agent_ratio",
            "realized.conflicting_agent_coverage",
            "realized.component_count",
            "realized.component_coverage_mean",
            "realized.component_coverage_max",
            "realized.internal_conflict_edges",
            "realized.boundary_conflict_edges",
            "realized.incident_conflict_coverage",
            "realized.internal_conflict_coverage",
            "realized.internal_event_coverage",
            "realized.incident_event_coverage",
            "realized.path_overlap_mean",
            "realized.path_overlap_max",
            "realized.path_union_cell_ratio",
            "realized.path_bbox_area_ratio",
            "realized.path_degree_mean",
            "realized.path_low_degree_ratio",
            "realized.path_articulation_ratio",
            "realized.path_visit_heat_mean",
            "realized.path_obstacle_rate_r2",
            "realized.path_obstacle_rate_r4",
            "realized.path_wait_ratio_mean",
            *(
                f"realized.{group}_{aggregate}"
                for group in ("delay", "conflict_degree", "path_cost", "path_stretch")
                for aggregate in AGGREGATES
            ),
        }
    )
)

# The removed value is ``scale * canonical`` for every non-constant alias.
# These identities are tied to the registered two-representative proposal
# generator and are checked before a v2 model/controller is exported.
REDUNDANT_FEATURE_ALIASES: dict[str, tuple[str, float]] = {
    "realized.actual_size": ("proposal.actual_size", 1.0),
    "realized.actual_size_ratio_agents": (
        "proposal.actual_size_ratio_agents",
        1.0,
    ),
    "proposal.family_ratio=random:16": (
        "proposal.family_count=random:16",
        1.0,
    ),
    "proposal.selection_family=random:16": (
        "proposal.family_count=random:16",
        1.0,
    ),
    "proposal.family_ratio=random:8": (
        "proposal.family_count=random:8",
        1.0,
    ),
    "proposal.selection_family=random:8": (
        "proposal.family_count=random:8",
        1.0,
    ),
    "proposal.selection_family=collision:16": (
        "proposal.family_ratio=collision:16",
        1.0,
    ),
    "proposal.selection_family=collision:8": (
        "proposal.family_ratio=collision:8",
        1.0,
    ),
    "proposal.selection_family=random:4": (
        "proposal.family_ratio=random:4",
        1.0,
    ),
    "proposal.seed_component_size_max": (
        "proposal.seed_component_size_mean",
        1.0,
    ),
    "proposal.seed_component_size_min": (
        "proposal.seed_component_size_mean",
        1.0,
    ),
    "proposal.unique_proposal_seed_count": ("proposal.total_count", 1.0),
    "state.sum_of_costs_per_agent": ("state.path_cost_mean", 1.0),
    "state.degree_mean": ("state.conflict_edge_density", 2.0),
}
CONSTANT_REDUNDANT_FEATURES = {
    "proposal.seed_component_size_std": 0.0,
}
REMOVED_FEATURE_NAMES = frozenset(
    (*REDUNDANT_FEATURE_ALIASES, *CONSTANT_REDUNDANT_FEATURES)
)

STATE_FEATURE_NAMES = tuple(
    sorted(set(STATE_FEATURE_NAMES_V1) - REMOVED_FEATURE_NAMES)
)
PROPOSAL_FEATURE_NAMES = tuple(
    sorted(set(PROPOSAL_FEATURE_NAMES_V1) - REMOVED_FEATURE_NAMES)
)
REALIZED_FEATURE_NAMES = tuple(
    sorted(set(REALIZED_FEATURE_NAMES_V1) - REMOVED_FEATURE_NAMES)
)
PROFILE_FEATURE_NAMES = {
    "proposal_dynamic": tuple(sorted((*STATE_FEATURE_NAMES, *PROPOSAL_FEATURE_NAMES))),
    "realized_dynamic": tuple(
        sorted((*STATE_FEATURE_NAMES, *PROPOSAL_FEATURE_NAMES, *REALIZED_FEATURE_NAMES))
    ),
}


def _schema_payload() -> dict[str, Any]:
    return {
        "schema": FEATURE_SCHEMA_ID,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "actual_sizes": list(ACTUAL_SIZES),
        "proposal_families": list(PROPOSAL_FAMILIES),
        "profiles": {
            profile: list(names) for profile, names in sorted(PROFILE_FEATURE_NAMES.items())
        },
        "aliases": {
            name: {"canonical": canonical, "scale": scale}
            for name, (canonical, scale) in sorted(REDUNDANT_FEATURE_ALIASES.items())
        },
        "constants": dict(sorted(CONSTANT_REDUNDANT_FEATURES.items())),
    }


FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        _schema_payload(),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def feature_schema_manifest() -> dict[str, Any]:
    return {**_schema_payload(), "sha256": FEATURE_SCHEMA_SHA256}


def canonical_feature(name: str) -> tuple[str, float]:
    return REDUNDANT_FEATURE_ALIASES.get(str(name), (str(name), 1.0))


def canonicalize_features(
    features: dict[str, Any], profile: str
) -> dict[str, float]:
    try:
        names = PROFILE_FEATURE_NAMES[profile]
    except KeyError as error:
        raise ValueError(f"unsupported feature-v2 profile: {profile}") from error
    values = {name: float(features.get(name, 0.0)) for name in names}
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError(f"non-finite feature value in {profile}")
    return values


def unsupported_actual_size(features: dict[str, Any]) -> bool:
    try:
        raw = float(features.get("proposal.actual_size", 0.0))
    except (TypeError, ValueError):
        return True
    return not math.isfinite(raw) or not raw.is_integer() or int(raw) not in ACTUAL_SIZES


def redundancy_violations(
    features: dict[str, Any], *, absolute_tolerance: float = 1e-12
) -> list[dict[str, Any]]:
    violations = []
    for removed, (canonical, scale) in REDUNDANT_FEATURE_ALIASES.items():
        actual = float(features.get(removed, 0.0))
        expected = scale * float(features.get(canonical, 0.0))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=absolute_tolerance):
            violations.append(
                {
                    "feature": removed,
                    "canonical": canonical,
                    "actual": actual,
                    "expected": expected,
                }
            )
    for removed, expected in CONSTANT_REDUNDANT_FEATURES.items():
        actual = float(features.get(removed, 0.0))
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=absolute_tolerance):
            violations.append(
                {
                    "feature": removed,
                    "canonical": None,
                    "actual": actual,
                    "expected": expected,
                }
            )
    return violations


def validate_redundancies(
    rows: Iterable[dict[str, Any]], profile: str = "realized_dynamic"
) -> dict[str, Any]:
    row_count = 0
    violation_count = 0
    violations = []
    for row in rows:
        row_count += 1
        features = dict(row.get("features", {}).get(profile, {}))
        for violation in redundancy_violations(features):
            violation_count += 1
            if len(violations) < 100:
                violations.append(
                    {
                        "state_id": row.get("state_id"),
                        "candidate_id": row.get("candidate_id"),
                        **violation,
                    }
                )
    return {
        "schema": "lns2.feature_redundancy_audit.v2",
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "row_count": row_count,
        "removed_feature_count": len(REMOVED_FEATURE_NAMES),
        "violation_count": violation_count,
        "violations": violations,
        "violations_truncated": violation_count > len(violations),
        "passed": violation_count == 0,
    }


if len(PROFILE_FEATURE_NAMES["proposal_dynamic"]) != 82:
    raise RuntimeError("feature-v2 proposal_dynamic registry must contain 82 features")
if len(PROFILE_FEATURE_NAMES["realized_dynamic"]) != 124:
    raise RuntimeError("feature-v2 realized_dynamic registry must contain 124 features")
if len(REMOVED_FEATURE_NAMES) != 15:
    raise RuntimeError("feature-v2 must remove exactly 15 redundant features")


__all__ = [
    "ACTUAL_SIZES",
    "CONSTANT_REDUNDANT_FEATURES",
    "FEATURE_SCHEMA_ID",
    "FEATURE_SCHEMA_SHA256",
    "FEATURE_SCHEMA_VERSION",
    "PROFILE_FEATURE_NAMES",
    "PROPOSAL_FAMILIES",
    "REDUNDANT_FEATURE_ALIASES",
    "REMOVED_FEATURE_NAMES",
    "canonical_feature",
    "canonicalize_features",
    "feature_schema_manifest",
    "redundancy_violations",
    "unsupported_actual_size",
    "validate_redundancies",
]
