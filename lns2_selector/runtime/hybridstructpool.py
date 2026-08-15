from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from typing import Any, Iterable

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.causalclosurepool import generate_causalclosure_candidates
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_grid,
    generate_structpool_candidate_subset,
)


HYBRIDSTRUCTPOOL_ID = "stride-hybridstructpool-v1"
STRUCTURAL_SIZES = (8, 16, 24, 32)
RUNTIME_STRUCTURAL_FAMILY_SIZES = {
    "bottleneck_crossing": STRUCTURAL_SIZES,
    "conflict_component": STRUCTURAL_SIZES,
    "path_overlap": STRUCTURAL_SIZES,
    "spatiotemporal_hotspot": STRUCTURAL_SIZES,
    "topology_boundary_articulation": STRUCTURAL_SIZES,
    "topology_boundary_low_degree": STRUCTURAL_SIZES,
}
HYBRID_GROUP_ORDER = (
    "causalclosure_v2",
    "topology_boundary",
    "conflict_component",
    "spatiotemporal_hotspot",
    "bottleneck_crossing",
    "path_overlap",
)
_HYBRIDSTRUCTPOOL_RUNTIME_CONFIG = {
    "enabled": True,
    "pool_id": HYBRIDSTRUCTPOOL_ID,
    "runtime_id": "stride-hybridstructpool-full-engineered-runtime-v5",
    "full_union_required": True,
    "full_union_audit_preserved": True,
    "runtime_filter_id": "none",
    "structural_sizes": list(STRUCTURAL_SIZES),
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
    "activation_gate": {
        "gate_id": "stride-highstress-state-v1",
        "minimum_conflict_pair_count": 16,
        "any_of": {
            "minimum_agent_count": 96,
            "minimum_active_conflict_agent_count": 32,
            "minimum_largest_conflict_component_size": 16,
        },
    },
}


def hybridstructpool_runtime_augmentation() -> dict[str, Any]:
    """Return the immutable full-union engineered runtime contract."""

    return json.loads(json.dumps(_HYBRIDSTRUCTPOOL_RUNTIME_CONFIG))


def validate_hybridstructpool_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    if result != _HYBRIDSTRUCTPOOL_RUNTIME_CONFIG:
        raise ValueError("unsupported HybridStructPool runtime augmentation")
    return result


def hybridstructpool_high_stress_gate(
    state: dict[str, Any], value: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the registered outcome-blind activation gate."""

    config = validate_hybridstructpool_augmentation(value)
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
        len(agent_ids) >= int(any_of["minimum_agent_count"])
        or len(active) >= int(any_of["minimum_active_conflict_agent_count"])
        or largest >= int(any_of["minimum_largest_conflict_component_size"])
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
        "agent_count": len(agent_ids),
        "conflict_pair_count": conflict_pairs,
        "active_conflict_agent_count": len(active),
        "largest_conflict_component_size": largest,
    }


@dataclass
class HybridStructPoolResult:
    """Outcome-blind union of V2, the complete StructShell, and CausalClosure."""

    candidates: list[dict[str, Any]]
    challengers: list[dict[str, Any]]
    provenance_by_candidate_id: dict[str, tuple[str, ...]]
    base_candidate_count: int
    structural_candidate_count: int
    causal_candidate_count: int
    exact_duplicate_count: int
    causal_attempts: list[dict[str, Any]]


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set, right_set = set(map(int, left)), set(map(int, right))
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _audit_vector(row: dict[str, Any]) -> tuple[float, float, float]:
    audit = dict(row.get("proposal_audit") or {})
    features = dict(row.get("features") or {})
    return (
        float(
            audit.get(
                "global_event_incident_coverage",
                features.get("realized.incident_event_coverage", 0.0),
            )
        ),
        float(
            audit.get(
                "global_pair_internal_coverage",
                features.get("realized.internal_conflict_coverage", 0.0),
            )
        ),
        float(
            audit.get(
                "conflict_component_reach",
                features.get("realized.component_coverage_max", 0.0),
            )
        ),
    )


def _semantic_groups(
    row: dict[str, Any], provenance: Iterable[str]
) -> tuple[str, ...]:
    groups = set(map(str, row.get("structpool_family_groups") or ()))
    sources = set(map(str, provenance))
    if "causalclosure_v2" in sources:
        groups.add("causalclosure_v2")
    return tuple(group for group in HYBRID_GROUP_ORDER if group in groups)


def _pareto_rank(rows: list[dict[str, Any]]) -> dict[str, int]:
    remaining = {str(row["candidate_id"]): row for row in rows}
    ranks: dict[str, int] = {}
    rank = 0
    while remaining:
        front: list[str] = []
        for identity, row in remaining.items():
            vector = _audit_vector(row)
            size = int(row["actual_size"])
            dominated = False
            for other_id, other in remaining.items():
                if other_id == identity:
                    continue
                other_vector = _audit_vector(other)
                other_size = int(other["actual_size"])
                weak = all(a >= b for a, b in zip(other_vector, vector)) and other_size <= size
                strict = any(a > b for a, b in zip(other_vector, vector)) or other_size < size
                if weak and strict:
                    dominated = True
                    break
            if not dominated:
                front.append(identity)
        if not front:
            raise RuntimeError("HybridStructPool Pareto ranking produced no front")
        for identity in sorted(front):
            ranks[identity] = rank
            remaining.pop(identity)
        rank += 1
    return ranks


def reduce_hybridstructpool_challengers(
    result: HybridStructPoolResult,
    *,
    maximum_challengers: int,
) -> list[dict[str, Any]]:
    """Apply outcome-blind semantic round-robin and set diversity.

    V2 candidates remain outside this budget.  Every available semantic group
    receives one turn before any group receives a second turn.  Within a group,
    current-state coverage, smaller size, and distance from already selected
    sets are Pareto/diversity criteria; no PP result is read.
    """

    if maximum_challengers not in {6, 8, 12}:
        raise ValueError("HybridStructPool audit supports budgets 6, 8, and 12")
    challengers = [copy.deepcopy(row) for row in result.challengers]
    if not challengers:
        return []
    ranks = _pareto_rank(challengers)
    by_group: dict[str, list[dict[str, Any]]] = {group: [] for group in HYBRID_GROUP_ORDER}
    for row in challengers:
        identity = str(row["candidate_id"])
        for group in _semantic_groups(row, result.provenance_by_candidate_id[identity]):
            by_group[group].append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    while len(selected) < maximum_challengers:
        added = False
        for group in HYBRID_GROUP_ORDER:
            choices = [
                row
                for row in by_group[group]
                if str(row["candidate_id"]) not in selected_ids
            ]
            if not choices:
                continue
            chosen = min(
                choices,
                key=lambda row: (
                    int(ranks[str(row["candidate_id"])]),
                    -min(
                        (
                            1.0 - _jaccard(row["agents"], previous["agents"])
                            for previous in selected
                        ),
                        default=1.0,
                    ),
                    -sum(_audit_vector(row)) / max(1, int(row["actual_size"])),
                    int(row["actual_size"]),
                    str(row["candidate_id"]),
                ),
            )
            selected.append(copy.deepcopy(chosen))
            selected_ids.add(str(chosen["candidate_id"]))
            added = True
            if len(selected) >= maximum_challengers:
                break
        if not added:
            break
    return selected


def _normalized_candidate(row: dict[str, Any]) -> tuple[tuple[int, ...], str]:
    agents = tuple(sorted(set(map(int, row.get("agents") or ()))))
    if not agents:
        raise ValueError("HybridStructPool candidates must contain agents")
    identity = str(row.get("candidate_id") or "")
    expected = candidate_id(agents)
    if identity != expected:
        raise ValueError("HybridStructPool candidate identity does not match agents")
    return agents, identity


def merge_hybridstructpool_candidates(
    base_candidates: Iterable[dict[str, Any]],
    structural_candidates: Iterable[dict[str, Any]],
    causal_candidates: Iterable[dict[str, Any]],
) -> HybridStructPoolResult:
    """Merge exact agent sets while retaining source provenance outside rows.

    Input rows are copied without adding outcome or selector fields.  V2 rows
    win exact-set ties so the frozen base representation remains authoritative.
    """

    sources = (
        ("v2_base", [dict(row) for row in base_candidates]),
        ("structshell_equal_four_size", [dict(row) for row in structural_candidates]),
        ("causalclosure_v2", [dict(row) for row in causal_candidates]),
    )
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    provenance: dict[tuple[int, ...], set[str]] = {}
    source_counts: dict[str, int] = {}
    duplicate_count = 0
    for source, rows in sources:
        source_counts[source] = len(rows)
        seen_in_source: set[tuple[int, ...]] = set()
        for row in rows:
            agents, _identity = _normalized_candidate(row)
            if agents in seen_in_source:
                raise ValueError(f"HybridStructPool source contains duplicate sets: {source}")
            seen_in_source.add(agents)
            provenance.setdefault(agents, set()).add(source)
            if agents in merged:
                duplicate_count += 1
                continue
            merged[agents] = copy.deepcopy(row)

    ordered = sorted(merged.values(), key=lambda row: str(row["candidate_id"]))
    provenance_by_id = {
        str(merged[agents]["candidate_id"]): tuple(sorted(values))
        for agents, values in sorted(
            provenance.items(), key=lambda item: str(merged[item[0]]["candidate_id"])
        )
    }
    challengers = [
        copy.deepcopy(row)
        for row in ordered
        if "v2_base" not in provenance_by_id[str(row["candidate_id"])]
    ]
    return HybridStructPoolResult(
        # Every merged row was already deep-copied from its source above.
        # Returning the private ordered list avoids a second full-pool copy;
        # callers may still annotate it without mutating any input candidate.
        candidates=ordered,
        challengers=challengers,
        provenance_by_candidate_id=provenance_by_id,
        base_candidate_count=source_counts["v2_base"],
        structural_candidate_count=source_counts["structshell_equal_four_size"],
        causal_candidate_count=source_counts["causalclosure_v2"],
        exact_duplicate_count=duplicate_count,
        causal_attempts=[],
    )


def generate_hybridstructpool_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_candidates: Iterable[dict[str, Any]],
    v2_anchors: Iterable[dict[str, Any]],
    structural_sizes: Iterable[int] = STRUCTURAL_SIZES,
    maximum_causal_candidates: int = 12,
    maximum_causal_neighborhood_size: int = 64,
    causal_temporal_window: int = 2,
    maximum_causal_jaccard: float = 0.9,
) -> HybridStructPoolResult:
    """Generate the full candidate contract without a runtime budget reducer.

    The function intentionally exposes every exact-deduplicated four-size
    structural action.  It does not rank, score with outcomes, or compress the
    union to a runtime budget.
    """

    sizes = tuple(sorted(set(map(int, structural_sizes))))
    if sizes != STRUCTURAL_SIZES:
        raise ValueError("HybridStructPool requires symmetric sizes 8, 16, 24, and 32")
    base = [dict(row) for row in v2_candidates]
    anchors = [dict(row) for row in v2_anchors]
    if not base or not anchors:
        raise ValueError("HybridStructPool requires the full V2 pool and a V2 anchor")
    structural = generate_structpool_candidate_grid(
        state, analysis, neighborhood_sizes=sizes
    )
    causal = generate_causalclosure_candidates(
        state,
        analysis,
        v2_anchors=anchors,
        maximum_candidates=maximum_causal_candidates,
        maximum_neighborhood_size=maximum_causal_neighborhood_size,
        temporal_window=causal_temporal_window,
        maximum_jaccard_similarity=maximum_causal_jaccard,
    )
    result = merge_hybridstructpool_candidates(base, structural, causal.candidates)
    result.causal_attempts = copy.deepcopy(causal.attempts)
    return result


def generate_hybridstructpool_runtime_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_candidates: Iterable[dict[str, Any]],
    v2_anchors: Iterable[dict[str, Any]],
    structural_family_sizes: dict[str, Iterable[int]],
    maximum_causal_candidates: int = 12,
    maximum_causal_neighborhood_size: int = 64,
    causal_temporal_window: int = 2,
    maximum_causal_jaccard: float = 0.9,
) -> HybridStructPoolResult:
    """Generate the full runtime union through the subset-capable API.

    Runtime acceleration must not alter candidate membership.  The registered
    mapping therefore contains every structural family/size cell; the
    subset-capable implementation is retained only so a future, separately
    validated selector can use it without changing the complete audit API.
    """

    normalized = {
        str(family): tuple(sorted(set(map(int, sizes))))
        for family, sizes in structural_family_sizes.items()
    }
    if normalized != RUNTIME_STRUCTURAL_FAMILY_SIZES:
        raise ValueError("unsupported HybridStructPool runtime structural mask")
    base = [dict(row) for row in v2_candidates]
    anchors = [dict(row) for row in v2_anchors]
    if not base or not anchors:
        raise ValueError("HybridStructPool requires the full V2 pool and a V2 anchor")
    structural = generate_structpool_candidate_subset(
        state,
        analysis,
        family_sizes=normalized,
    )
    causal = generate_causalclosure_candidates(
        state,
        analysis,
        v2_anchors=anchors,
        maximum_candidates=maximum_causal_candidates,
        maximum_neighborhood_size=maximum_causal_neighborhood_size,
        temporal_window=causal_temporal_window,
        maximum_jaccard_similarity=maximum_causal_jaccard,
    )
    result = merge_hybridstructpool_candidates(base, structural, causal.candidates)
    result.causal_attempts = copy.deepcopy(causal.attempts)
    return result


__all__ = [
    "HYBRIDSTRUCTPOOL_ID",
    "STRUCTURAL_SIZES",
    "RUNTIME_STRUCTURAL_FAMILY_SIZES",
    "HybridStructPoolResult",
    "generate_hybridstructpool_candidates",
    "generate_hybridstructpool_runtime_candidates",
    "hybridstructpool_high_stress_gate",
    "hybridstructpool_runtime_augmentation",
    "merge_hybridstructpool_candidates",
    "reduce_hybridstructpool_challengers",
    "validate_hybridstructpool_augmentation",
]
