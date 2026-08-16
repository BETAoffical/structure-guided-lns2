from __future__ import annotations

import json
import time
from typing import Any, Iterable

from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.hybridstructpool import (
    HybridStructPoolResult,
    merge_hybridstructpool_candidates,
)
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_subset,
)


STRUCTSHELL_SINGLE_FAMILY_POOL_ID = "stride-structshell-single-family-v1"
STRUCTSHELL_SINGLE_FAMILY_RUNTIME_ID = (
    "stride-structshell-single-family-runtime-v1"
)
STRUCTSHELL_SINGLE_FAMILY_PROFILES = (
    "bottleneck",
    "conflict_component",
    "hotspot",
    "path_overlap",
)
STRUCTSHELL_SINGLE_FAMILY_NOMINAL_SIZES = (8, 16, 24, 32)

_PROFILE_VARIANTS = {
    "bottleneck": "bottleneck_crossing",
    "conflict_component": "conflict_component",
    "hotspot": "spatiotemporal_hotspot",
    "path_overlap": "path_overlap",
}
_ABLATION_ACTIVATION_GATE = {
    "gate_id": "single_family_ablation_all_states",
}


def structshell_single_family_augmentation(
    profile: str, nominal_size: int = 16
) -> dict[str, Any]:
    """Return one separately versioned StructShell family/size contract."""

    normalized_profile = str(profile)
    if normalized_profile not in STRUCTSHELL_SINGLE_FAMILY_PROFILES:
        raise ValueError(
            f"unsupported single-family StructShell profile: {normalized_profile}"
        )
    if type(nominal_size) is not int:
        raise ValueError("single-family StructShell nominal size must be an integer")
    normalized_size = nominal_size
    if normalized_size not in STRUCTSHELL_SINGLE_FAMILY_NOMINAL_SIZES:
        raise ValueError(
            "single-family StructShell nominal size must be one of 8/16/24/32"
        )
    variant = _PROFILE_VARIANTS[normalized_profile]
    result = {
        "enabled": True,
        "pool_id": STRUCTSHELL_SINGLE_FAMILY_POOL_ID,
        "runtime_id": STRUCTSHELL_SINGLE_FAMILY_RUNTIME_ID,
        "full_union_required": False,
        "full_union_audit_preserved": True,
        "runtime_filter_id": "single_family_single_size_v1",
        "source_mode": "structshell_single_family",
        "structural_profile": normalized_profile,
        "nominal_size": normalized_size,
        "runtime_structural_family_sizes": {variant: [normalized_size]},
        "maximum_added_candidates": 1,
        "maximum_total_candidates": 64,
        "static_grid_cache": True,
        "activation_gate": _ABLATION_ACTIVATION_GATE,
    }
    return json.loads(json.dumps(result))


def validate_structshell_single_family_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    profile = str(result.get("structural_profile") or "")
    nominal_size = result.get("nominal_size")
    if type(nominal_size) is not int:
        raise ValueError(
            "unsupported single-family StructShell runtime augmentation"
        )
    if result != structshell_single_family_augmentation(profile, nominal_size):
        raise ValueError(
            "unsupported single-family StructShell runtime augmentation"
        )
    return result


def structshell_single_family_ablation_gate(
    state: dict[str, Any], value: dict[str, Any]
) -> dict[str, Any]:
    """Enable the ablation on every valid, non-terminal repair state."""

    config = validate_structshell_single_family_augmentation(value)
    assert config is not None
    started = time.perf_counter()
    agent_rows = list(state.get("agents", []))
    agent_ids = [int(agent["id"]) for agent in agent_rows]
    if not agent_ids:
        raise ValueError("single-family StructShell gate requires at least one agent")
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("single-family StructShell gate received duplicate agent ids")
    agent_id_set = set(agent_ids)
    raw_edges = list(state.get("conflict_edges", []))
    if any(
        not isinstance(edge, (list, tuple)) or len(edge) != 2
        for edge in raw_edges
    ):
        raise ValueError("single-family StructShell gate received an invalid edge")
    edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in raw_edges
    }
    if any(
        left == right or left not in agent_id_set or right not in agent_id_set
        for left, right in edges
    ):
        raise ValueError("single-family StructShell gate received an invalid edge")
    raw_conflict_pairs = state.get("num_of_colliding_pairs")
    if type(raw_conflict_pairs) is not int or raw_conflict_pairs < 0:
        raise ValueError(
            "single-family StructShell gate requires a nonnegative conflict count"
        )
    conflict_pairs = raw_conflict_pairs
    if conflict_pairs != len(edges):
        raise ValueError(
            "single-family StructShell gate conflict count differs from edges"
        )
    active = {agent for edge in edges for agent in edge}
    adjacency = {agent: set() for agent in active}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    largest = 0
    remaining = set(active)
    while remaining:
        root = min(remaining)
        component = {root}
        frontier = [root]
        remaining.remove(root)
        while frontier:
            current = frontier.pop()
            for neighbor in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
        largest = max(largest, len(component))

    gate = dict(config["activation_gate"])
    passed = conflict_pairs > 0 and not bool(
        state.get("done", False) or state.get("feasible", False)
    )
    return {
        "passed": passed,
        "reason": str(gate["gate_id"]) if passed else "terminal_state",
        "seconds": time.perf_counter() - started,
        "gate_id": str(gate["gate_id"]),
        "agent_count": len(agent_id_set),
        "conflict_pair_count": conflict_pairs,
        "active_conflict_agent_count": len(active),
        "largest_conflict_component_size": largest,
    }


def generate_structshell_single_family_runtime_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_candidates: Iterable[dict[str, Any]],
    v2_anchors: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> HybridStructPoolResult:
    """Merge the complete V2 pool with at most one StructShell challenger."""

    specification = validate_structshell_single_family_augmentation(config)
    assert specification is not None
    base = list(v2_candidates)
    anchors = list(v2_anchors)
    if not base or not anchors:
        raise ValueError(
            "single-family StructShell requires the full V2 pool and a V2 anchor"
        )
    family_sizes = {
        str(family): tuple(map(int, sizes))
        for family, sizes in dict(
            specification["runtime_structural_family_sizes"]
        ).items()
    }
    started = time.perf_counter()
    structural = generate_structpool_candidate_subset(
        state,
        analysis,
        family_sizes=family_sizes,
    )
    structural_seconds = time.perf_counter() - started
    if len(structural) > int(specification["maximum_added_candidates"]):
        raise RuntimeError(
            "single-family StructShell generated more than one candidate"
        )
    result = merge_hybridstructpool_candidates(base, structural, [])
    if len(result.challengers) > int(specification["maximum_added_candidates"]):
        raise RuntimeError(
            "single-family StructShell retained more than one challenger"
        )
    result.structural_generation_seconds = structural_seconds
    result.causal_generation_seconds = 0.0
    return result


__all__ = [
    "STRUCTSHELL_SINGLE_FAMILY_NOMINAL_SIZES",
    "STRUCTSHELL_SINGLE_FAMILY_POOL_ID",
    "STRUCTSHELL_SINGLE_FAMILY_PROFILES",
    "STRUCTSHELL_SINGLE_FAMILY_RUNTIME_ID",
    "generate_structshell_single_family_runtime_candidates",
    "structshell_single_family_augmentation",
    "structshell_single_family_ablation_gate",
    "validate_structshell_single_family_augmentation",
]
