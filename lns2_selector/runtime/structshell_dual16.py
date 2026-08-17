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


STRUCTSHELL_DUAL16_POOL_ID = "stride-structshell-dual16-v1"
STRUCTSHELL_DUAL16_RUNTIME_ID = "stride-structshell-dual16-runtime-v1"
STRUCTSHELL_DUAL16_PLATEAU_RUNTIME_ID = (
    "stride-structshell-dual16-plateau-guard-runtime-v1"
)

_DUAL16_FAMILY_SIZES = {
    "conflict_component": [16],
    "spatiotemporal_hotspot": [16],
}
_DUAL16_ACTIVATION_GATE = {
    "gate_id": "dual16_ablation_all_states",
}


def structshell_dual16_augmentation() -> dict[str, Any]:
    """Return the immutable Component16 + Hotspot16 runtime contract."""

    result = {
        "enabled": True,
        "pool_id": STRUCTSHELL_DUAL16_POOL_ID,
        "runtime_id": STRUCTSHELL_DUAL16_RUNTIME_ID,
        "full_union_required": False,
        "full_union_audit_preserved": True,
        "runtime_filter_id": "dual_family_fixed16_v1",
        "source_mode": "structshell_dual16",
        "nominal_size": 16,
        "runtime_structural_family_sizes": _DUAL16_FAMILY_SIZES,
        "maximum_added_candidates": 2,
        "maximum_total_candidates": 65,
        "static_grid_cache": True,
        "activation_gate": _DUAL16_ACTIVATION_GATE,
    }
    return json.loads(json.dumps(result))


def structshell_dual16_plateau_augmentation() -> dict[str, Any]:
    """Return Dual16 with a late no-progress fallback to fresh full V2."""

    result = structshell_dual16_augmentation()
    result.update(
        {
            "runtime_id": STRUCTSHELL_DUAL16_PLATEAU_RUNTIME_ID,
            "runtime_filter_id": "dual_family_fixed16_plateau_guard_v1",
            "stall_guard": {
                "guard_id": "stride-dual16-plateau-guard-v1",
                "no_progress_limit": 8,
                "counter": "consecutive_non_decreasing_conflict_decisions",
                "fallback": "fresh_v2_full",
                "release_condition": "strict_conflict_decrease",
                "wall_time_condition": None,
                "maximum_pp_calls_per_decision": 1,
                "retry_rollback_or_rescue": False,
            },
        }
    )
    return json.loads(json.dumps(result))


def validate_structshell_dual16_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    if result not in (
        structshell_dual16_augmentation(),
        structshell_dual16_plateau_augmentation(),
    ):
        raise ValueError("unsupported Dual16 StructShell runtime augmentation")
    return result


def structshell_dual16_ablation_gate(
    state: dict[str, Any], value: dict[str, Any]
) -> dict[str, Any]:
    """Enable Dual16 on every valid, non-terminal conflicted repair state."""

    config = validate_structshell_dual16_augmentation(value)
    assert config is not None
    started = time.perf_counter()
    agent_rows = list(state.get("agents", []))
    agent_ids = [int(agent["id"]) for agent in agent_rows]
    if not agent_ids:
        raise ValueError("Dual16 StructShell gate requires at least one agent")
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("Dual16 StructShell gate received duplicate agent ids")
    agent_id_set = set(agent_ids)
    raw_edges = list(state.get("conflict_edges", []))
    if any(
        not isinstance(edge, (list, tuple)) or len(edge) != 2
        for edge in raw_edges
    ):
        raise ValueError("Dual16 StructShell gate received an invalid edge")
    edges = {
        tuple(sorted((int(edge[0]), int(edge[1])))) for edge in raw_edges
    }
    if any(
        left == right or left not in agent_id_set or right not in agent_id_set
        for left, right in edges
    ):
        raise ValueError("Dual16 StructShell gate received an invalid edge")
    raw_conflict_pairs = state.get("num_of_colliding_pairs")
    if type(raw_conflict_pairs) is not int or raw_conflict_pairs < 0:
        raise ValueError(
            "Dual16 StructShell gate requires a nonnegative conflict count"
        )
    conflict_pairs = raw_conflict_pairs
    if conflict_pairs != len(edges):
        raise ValueError(
            "Dual16 StructShell gate conflict count differs from edges"
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

    gate_id = str(config["activation_gate"]["gate_id"])
    passed = conflict_pairs > 0 and not bool(
        state.get("done", False) or state.get("feasible", False)
    )
    return {
        "passed": passed,
        "reason": gate_id if passed else "terminal_state",
        "seconds": time.perf_counter() - started,
        "gate_id": gate_id,
        "agent_count": len(agent_id_set),
        "conflict_pair_count": conflict_pairs,
        "active_conflict_agent_count": len(active),
        "largest_conflict_component_size": largest,
    }


def generate_structshell_dual16_runtime_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_candidates: Iterable[dict[str, Any]],
    v2_anchors: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> HybridStructPoolResult:
    """Merge the complete V2 pool with at most two fixed-size challengers."""

    specification = validate_structshell_dual16_augmentation(config)
    assert specification is not None
    base = list(v2_candidates)
    anchors = list(v2_anchors)
    if not base or not anchors:
        raise ValueError(
            "Dual16 StructShell requires the full V2 pool and a V2 anchor"
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
    maximum_added = int(specification["maximum_added_candidates"])
    if len(structural) > maximum_added:
        raise RuntimeError("Dual16 StructShell generated more than two candidates")
    result = merge_hybridstructpool_candidates(base, structural, [])
    if len(result.challengers) > maximum_added:
        raise RuntimeError("Dual16 StructShell retained more than two challengers")
    result.structural_generation_seconds = structural_seconds
    result.causal_generation_seconds = 0.0
    return result


__all__ = [
    "STRUCTSHELL_DUAL16_PLATEAU_RUNTIME_ID",
    "STRUCTSHELL_DUAL16_POOL_ID",
    "STRUCTSHELL_DUAL16_RUNTIME_ID",
    "generate_structshell_dual16_runtime_candidates",
    "structshell_dual16_ablation_gate",
    "structshell_dual16_augmentation",
    "structshell_dual16_plateau_augmentation",
    "validate_structshell_dual16_augmentation",
]
