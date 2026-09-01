from __future__ import annotations

import json
import time
from typing import Any, Iterable

from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.causalclosurepool import generate_causalclosure_candidates
from lns2_selector.runtime.hybridstructpool import (
    HYBRIDSTRUCTPOOL_ID,
    RUNTIME_STRUCTURAL_FAMILY_SIZES,
    HybridStructPoolResult,
    merge_hybridstructpool_candidates,
    validate_hybridstructpool_augmentation,
)
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_subset,
)
from lns2_selector.runtime.structshell_component16 import (
    STRUCTSHELL_COMPONENT16_POOL_ID,
    validate_structshell_component16_augmentation,
)
from lns2_selector.runtime.structshell_dual16 import (
    STRUCTSHELL_DUAL16_POOL_ID,
    validate_structshell_dual16_augmentation,
)


ROUTED_HYBRIDSTRUCTPOOL_ID = "stride-hybridstructpool-routed-v1"
ROUTED_SOURCE_MODES = (
    "structshell_only",
    "causal_only",
    "routed_structshell",
)
_COMMON = {
    "enabled": True,
    "pool_id": ROUTED_HYBRIDSTRUCTPOOL_ID,
    "runtime_id": "stride-hybridstructpool-routed-runtime-v1",
    "full_union_required": False,
    "full_union_audit_preserved": True,
    "runtime_filter_id": "source_route_v1",
    "structural_sizes": [8, 16, 24, 32],
    "runtime_structural_family_sizes": {
        family: list(sizes)
        for family, sizes in RUNTIME_STRUCTURAL_FAMILY_SIZES.items()
    },
    "maximum_causal_candidates": 12,
    "maximum_causal_neighborhood_size": 64,
    "causal_temporal_window": 2,
    "maximum_causal_jaccard_similarity": 0.9,
    "maximum_total_candidates": 64,
    "static_grid_cache": True,
}
_LEGACY_GATE = {
    "gate_id": "stride-highstress-state-v1",
    "minimum_conflict_pair_count": 16,
    "any_of": {
        "minimum_agent_count": 96,
        "minimum_active_conflict_agent_count": 32,
        "minimum_largest_conflict_component_size": 16,
    },
}
_STRICT_GATE = {
    "gate_id": "stride-highstress-conflict-structure-v1",
    "minimum_conflict_pair_count": 16,
    "any_of": {
        "minimum_active_conflict_agent_count": 32,
        "minimum_largest_conflict_component_size": 16,
    },
}


def routed_hybridstructpool_augmentation(source_mode: str) -> dict[str, Any]:
    mode = str(source_mode)
    if mode not in ROUTED_SOURCE_MODES:
        raise ValueError(f"unsupported routed HybridStructPool source mode: {mode}")
    result = {
        **_COMMON,
        "source_mode": mode,
        "activation_gate": (
            _STRICT_GATE if mode == "routed_structshell" else _LEGACY_GATE
        ),
    }
    return json.loads(json.dumps(result))


def validate_routed_hybridstructpool_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    mode = str(result.get("source_mode") or "")
    if mode not in ROUTED_SOURCE_MODES or result != routed_hybridstructpool_augmentation(mode):
        raise ValueError("unsupported routed HybridStructPool runtime augmentation")
    return result


def validate_any_hybridstructpool_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    pool_id = str(value.get("pool_id") or "")
    if pool_id == HYBRIDSTRUCTPOOL_ID:
        return validate_hybridstructpool_augmentation(value)
    if pool_id == ROUTED_HYBRIDSTRUCTPOOL_ID:
        return validate_routed_hybridstructpool_augmentation(value)
    if pool_id == STRUCTSHELL_COMPONENT16_POOL_ID:
        return validate_structshell_component16_augmentation(value)
    if pool_id == STRUCTSHELL_DUAL16_POOL_ID:
        return validate_structshell_dual16_augmentation(value)
    raise ValueError("unsupported HybridStructPool runtime augmentation")


def routed_hybridstructpool_high_stress_gate(
    state: dict[str, Any], value: dict[str, Any]
) -> dict[str, Any]:
    config = validate_routed_hybridstructpool_augmentation(value)
    assert config is not None
    started = time.perf_counter()
    agent_ids = {int(agent["id"]) for agent in state.get("agents", [])}
    if not agent_ids:
        raise ValueError("HybridStructPool gate requires at least one agent")
    edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in state.get("conflict_edges", [])
    }
    if any(
        left == right or left not in agent_ids or right not in agent_ids
        for left, right in edges
    ):
        raise ValueError("HybridStructPool gate received an invalid conflict edge")
    conflict_pairs = int(state.get("num_of_colliding_pairs", -1))
    if conflict_pairs != len(edges):
        raise ValueError("HybridStructPool gate conflict count differs from edges")
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
    any_of = dict(gate["any_of"])
    conflict_ok = conflict_pairs >= int(gate["minimum_conflict_pair_count"])
    stress_ok = bool(
        len(active) >= int(any_of["minimum_active_conflict_agent_count"])
        or largest >= int(any_of["minimum_largest_conflict_component_size"])
        or (
            "minimum_agent_count" in any_of
            and len(agent_ids) >= int(any_of["minimum_agent_count"])
        )
    )
    passed = bool(conflict_ok and stress_ok)
    return {
        "passed": passed,
        "reason": (
            "high_stress_state"
            if passed
            else "conflict_pair_count_below_threshold"
            if not conflict_ok
            else "stress_indicator_below_threshold"
        ),
        "seconds": time.perf_counter() - started,
        "gate_id": str(gate["gate_id"]),
        "agent_count": len(agent_ids),
        "conflict_pair_count": conflict_pairs,
        "active_conflict_agent_count": len(active),
        "largest_conflict_component_size": largest,
    }


def generate_routed_hybridstructpool_runtime_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_candidates: Iterable[dict[str, Any]],
    v2_anchors: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> HybridStructPoolResult:
    specification = validate_routed_hybridstructpool_augmentation(config)
    assert specification is not None
    mode = str(specification["source_mode"])
    base = list(v2_candidates)
    anchors = list(v2_anchors)
    if not base or not anchors:
        raise ValueError("HybridStructPool requires the full V2 pool and a V2 anchor")

    structural: list[dict[str, Any]] = []
    causal_rows: list[dict[str, Any]] = []
    causal_attempts: list[dict[str, Any]] = []
    structural_seconds = 0.0
    causal_seconds = 0.0
    if mode in {"structshell_only", "routed_structshell"}:
        started = time.perf_counter()
        structural = generate_structpool_candidate_subset(
            state,
            analysis,
            family_sizes=RUNTIME_STRUCTURAL_FAMILY_SIZES,
        )
        structural_seconds = time.perf_counter() - started
    if mode == "causal_only":
        started = time.perf_counter()
        causal = generate_causalclosure_candidates(
            state,
            analysis,
            v2_anchors=anchors,
            maximum_candidates=int(specification["maximum_causal_candidates"]),
            maximum_neighborhood_size=int(
                specification["maximum_causal_neighborhood_size"]
            ),
            temporal_window=int(specification["causal_temporal_window"]),
            maximum_jaccard_similarity=float(
                specification["maximum_causal_jaccard_similarity"]
            ),
        )
        causal_seconds = time.perf_counter() - started
        causal_rows = list(causal.candidates)
        causal_attempts = list(causal.attempts)
    result = merge_hybridstructpool_candidates(base, structural, causal_rows)
    result.causal_attempts = causal_attempts
    result.structural_generation_seconds = structural_seconds
    result.causal_generation_seconds = causal_seconds
    return result


__all__ = [
    "ROUTED_HYBRIDSTRUCTPOOL_ID",
    "ROUTED_SOURCE_MODES",
    "generate_routed_hybridstructpool_runtime_candidates",
    "routed_hybridstructpool_augmentation",
    "routed_hybridstructpool_high_stress_gate",
    "validate_any_hybridstructpool_augmentation",
    "validate_routed_hybridstructpool_augmentation",
]
