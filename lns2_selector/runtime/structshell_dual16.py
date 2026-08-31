from __future__ import annotations

import copy
import json
import time
from typing import Any, Iterable

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.hybridstructpool import (
    HybridStructPoolResult,
)
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_subset,
)


STRUCTSHELL_DUAL16_POOL_ID = "stride-structshell-dual16-v1"
STRUCTSHELL_DUAL16_RUNTIME_ID = "stride-structshell-dual16-runtime-v3"

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


def validate_structshell_dual16_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    if result != structshell_dual16_augmentation():
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


def _normalized_dual16_candidate(
    row: dict[str, Any],
) -> tuple[tuple[int, ...], str]:
    agents = tuple(sorted(set(map(int, row.get("agents") or ()))))
    if not agents:
        raise ValueError("HybridStructPool candidates must contain agents")
    identity = str(row.get("candidate_id") or "")
    if identity != candidate_id(agents):
        raise ValueError("HybridStructPool candidate identity does not match agents")
    return agents, identity


def _merge_structshell_dual16_candidates(
    base_candidates: Iterable[dict[str, Any]],
    structural_candidates: Iterable[dict[str, Any]],
) -> HybridStructPoolResult:
    """Merge the fixed Dual16 union without deep-copying the complete V2 pool.

    Runtime consumers add provenance only at the candidate dictionary's top
    level.  A shallow copy therefore forms the copy-on-write boundary for V2
    rows while preserving every frozen nested value.  The at-most-two newly
    generated structural rows retain the generic merge's full copy isolation.

    Online V2 candidates are already ordered by candidate id, so the common
    path is a linear merge.  The fallback sort preserves the generic merge
    semantics for callers that provide an unordered iterable.
    """

    base = list(base_candidates)
    structural = list(structural_candidates)
    provenance: dict[tuple[int, ...], set[str]] = {}
    candidate_id_by_agents: dict[tuple[int, ...], str] = {}
    base_entries: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []
    structural_entries: list[tuple[str, tuple[int, ...], dict[str, Any]]] = []

    seen_base: set[tuple[int, ...]] = set()
    base_is_sorted = True
    previous_identity: str | None = None
    for row in base:
        agents, identity = _normalized_dual16_candidate(row)
        if agents in seen_base:
            raise ValueError(
                "HybridStructPool source contains duplicate sets: v2_base"
            )
        seen_base.add(agents)
        provenance[agents] = {"v2_base"}
        candidate_id_by_agents[agents] = identity
        if previous_identity is not None and previous_identity > identity:
            base_is_sorted = False
        previous_identity = identity
        # Downstream annotation writes only new top-level keys.  Copying the
        # outer mapping is sufficient to keep the frozen V2 input untouched.
        base_entries.append((identity, agents, dict(row)))

    seen_structural: set[tuple[int, ...]] = set()
    duplicate_count = 0
    for row in structural:
        agents, identity = _normalized_dual16_candidate(row)
        if agents in seen_structural:
            raise ValueError(
                "HybridStructPool source contains duplicate sets: "
                "structshell_equal_four_size"
            )
        seen_structural.add(agents)
        provenance.setdefault(agents, set()).add(
            "structshell_equal_four_size"
        )
        if agents in seen_base:
            duplicate_count += 1
            continue
        candidate_id_by_agents[agents] = identity
        structural_entries.append((identity, agents, copy.deepcopy(row)))

    structural_entries.sort(key=lambda entry: entry[0])
    if base_is_sorted:
        merged_entries: list[
            tuple[str, tuple[int, ...], dict[str, Any]]
        ] = []
        base_index = 0
        structural_index = 0
        while (
            base_index < len(base_entries)
            and structural_index < len(structural_entries)
        ):
            base_entry = base_entries[base_index]
            structural_entry = structural_entries[structural_index]
            if base_entry[0] <= structural_entry[0]:
                merged_entries.append(base_entry)
                base_index += 1
            else:
                merged_entries.append(structural_entry)
                structural_index += 1
        merged_entries.extend(base_entries[base_index:])
        merged_entries.extend(structural_entries[structural_index:])
    else:
        merged_entries = sorted(
            [*base_entries, *structural_entries], key=lambda entry: entry[0]
        )

    ordered = [entry[2] for entry in merged_entries]
    provenance_by_id = {
        candidate_id_by_agents[agents]: tuple(sorted(values))
        for agents, values in sorted(
            provenance.items(),
            key=lambda item: candidate_id_by_agents[item[0]],
        )
    }
    challengers = [
        copy.deepcopy(row)
        for _identity, agents, row in merged_entries
        if "v2_base" not in provenance[agents]
    ]
    return HybridStructPoolResult(
        candidates=ordered,
        challengers=challengers,
        provenance_by_candidate_id=provenance_by_id,
        base_candidate_count=len(base),
        structural_candidate_count=len(structural),
        causal_candidate_count=0,
        exact_duplicate_count=duplicate_count,
        causal_attempts=[],
    )


def generate_structshell_dual16_runtime_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_candidates: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> HybridStructPoolResult:
    """Merge the complete V2 pool with at most two fixed-size challengers."""

    specification = validate_structshell_dual16_augmentation(config)
    assert specification is not None
    base = list(v2_candidates)
    if not base:
        raise ValueError("Dual16 StructShell requires the full V2 pool")
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
    result = _merge_structshell_dual16_candidates(base, structural)
    if len(result.challengers) > maximum_added:
        raise RuntimeError("Dual16 StructShell retained more than two challengers")
    result.structural_generation_seconds = structural_seconds
    result.causal_generation_seconds = 0.0
    return result


__all__ = [
    "STRUCTSHELL_DUAL16_POOL_ID",
    "STRUCTSHELL_DUAL16_RUNTIME_ID",
    "generate_structshell_dual16_runtime_candidates",
    "structshell_dual16_ablation_gate",
    "structshell_dual16_augmentation",
    "validate_structshell_dual16_augmentation",
]
