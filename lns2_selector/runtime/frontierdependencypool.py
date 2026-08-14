from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.temporal_state import TemporalHistoryContext
from lns2_selector.runtime.topology_candidates import _jaccard, topology_candidate_audit


FRONTIERDEPENDENCYPOOL_ID = "stride-frontierdependencypool-v1"
_VARIANTS = ("compact-augment", "same-size-exchange")


@dataclass
class FrontierDependencyPoolResult:
    candidates: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    raw_motif_count: int
    raw_variant_count: int
    exact_duplicate_count: int
    jaccard_rejection_count: int


def _normalized_agents(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(sorted(set(map(int, values))))
    if not result:
        raise ValueError("FrontierDependencyPool requires a non-empty base candidate")
    return result


def _event_indexes(
    analysis: StateAnalysis,
) -> tuple[
    collections.Counter[tuple[int, int]],
    dict[int, set[int]],
    collections.Counter[int],
]:
    pair_events: collections.Counter[tuple[int, int]] = collections.Counter()
    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    event_weight: collections.Counter[int] = collections.Counter()
    for event in analysis.events:
        left, right = sorted((int(event.left), int(event.right)))
        pair_events[(left, right)] += 1
        adjacency[left].add(right)
        adjacency[right].add(left)
        event_weight[left] += 1
        event_weight[right] += 1
    return pair_events, dict(adjacency), event_weight


def _pair_counts(
    pairs: Iterable[tuple[int, int]], agents: set[int]
) -> tuple[int, int, int]:
    internal = 0
    incident = 0
    boundary = 0
    for left, right in pairs:
        left_selected = left in agents
        right_selected = right in agents
        internal += int(left_selected and right_selected)
        incident += int(left_selected or right_selected)
        boundary += int(left_selected != right_selected)
    return internal, incident, boundary


def _blocker_evidence(
    blocker: int,
    *,
    core: set[int],
    pivot: int,
    pair_events: Mapping[tuple[int, int], int],
    adjacency: Mapping[int, set[int]],
    persistent: set[tuple[int, int]],
    recent: set[int],
) -> dict[str, Any]:
    boundary_edges = sorted(
        tuple(sorted((blocker, member)))
        for member in adjacency.get(blocker, set()) & core
    )
    persistent_edges = sorted(set(boundary_edges) & persistent)
    return {
        "pivot_agent": int(pivot),
        "current_boundary_edges": [list(edge) for edge in boundary_edges],
        "persistent_boundary_edges": [list(edge) for edge in persistent_edges],
        "current_boundary_event_count": sum(
            int(pair_events[edge]) for edge in boundary_edges
        ),
        "current_core_neighbor_count": len(boundary_edges),
        "current_conflict_degree": len(adjacency.get(blocker, set())),
        "in_most_recent_neighborhood": blocker in recent,
    }


def _blocker_priority(
    agent: int,
    evidence: Mapping[int, Mapping[str, Any]],
) -> tuple[int, int, int, int, int, int]:
    row = evidence[agent]
    return (
        -len(row["persistent_boundary_edges"]),
        -int(row["in_most_recent_neighborhood"]),
        -int(row["current_boundary_event_count"]),
        -int(row["current_core_neighbor_count"]),
        -int(row["current_conflict_degree"]),
        int(agent),
    )


def _removal_priority(
    agent: int,
    *,
    core: set[int],
    adjacency: Mapping[int, set[int]],
    event_weight: Mapping[int, int],
    repair_counts: Mapping[int, int],
) -> tuple[int, int, int, int, int]:
    return (
        len(adjacency.get(agent, set()) & core),
        int(event_weight.get(agent, 0)),
        -int(repair_counts.get(agent, 0)),
        len(adjacency.get(agent, set())),
        int(agent),
    )


def _candidate_metrics(
    *,
    core: set[int],
    agents: set[int],
    added: set[int],
    evidence: Mapping[int, Mapping[str, Any]],
    pair_events: Mapping[tuple[int, int], int],
    persistent: set[tuple[int, int]],
) -> dict[str, Any]:
    pairs = set(pair_events)
    before_internal, before_incident, before_boundary = _pair_counts(pairs, core)
    after_internal, after_incident, after_boundary = _pair_counts(pairs, agents)
    converted = sorted(
        edge
        for edge in pairs
        if (edge[0] in core) != (edge[1] in core)
        and edge[0] in agents
        and edge[1] in agents
    )
    persistent_converted = sorted(set(converted) & persistent)
    event_support = sum(
        int(evidence[agent]["current_boundary_event_count"]) for agent in added
    )
    return {
        "converted_boundary_edges": [list(edge) for edge in converted],
        "persistent_converted_boundary_edges": [
            list(edge) for edge in persistent_converted
        ],
        "converted_boundary_edge_count": len(converted),
        "persistent_converted_boundary_edge_count": len(persistent_converted),
        "current_pair_internal_delta": after_internal - before_internal,
        "current_pair_incident_delta": after_incident - before_incident,
        "current_pair_boundary_delta": after_boundary - before_boundary,
        "boundary_event_support": event_support,
        "evidence_density": event_support / max(1, len(added)),
    }


def _variant_row(
    *,
    variant: str,
    pivot: int,
    core: set[int],
    added: set[int],
    removed: set[int],
    evidence: Mapping[int, Mapping[str, Any]],
    analysis: StateAnalysis,
    pair_events: Mapping[tuple[int, int], int],
    persistent: set[tuple[int, int]],
) -> dict[str, Any]:
    agents = (core - removed) | added
    metrics = _candidate_metrics(
        core=core,
        agents=agents,
        added=added,
        evidence=evidence,
        pair_events=pair_events,
        persistent=persistent,
    )
    family = f"frontierdependency:{variant}"
    ordered = sorted(agents)
    return {
        "candidate_id": candidate_id(ordered),
        "agents": ordered,
        "actual_size": len(ordered),
        "selection_source": family,
        "selection_families": [family],
        "selection_rank_by_family": {family: 0},
        "proposal_count_by_family": {family: 1},
        "proposal_seeds": [],
        "seed_agents": [int(pivot)],
        "proposal_audit": topology_candidate_audit(analysis, ordered),
        "candidate_kind": "frontierdependency",
        "frontierdependencypool_id": FRONTIERDEPENDENCYPOOL_ID,
        "frontierdependency_variant": variant,
        "core_agents": sorted(core),
        "pivot_agents": [int(pivot)],
        "added_blockers": sorted(added),
        "removed_agents": sorted(removed),
        "blocker_evidence": {
            str(agent): dict(evidence[agent]) for agent in sorted(added)
        },
        "same_size_as_core": len(agents) == len(core),
        "maximum_added_agents": 8,
        "future_outcome_used": False,
        "pp_probe_used": False,
        **metrics,
    }


def _candidate_order(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -int(row["persistent_converted_boundary_edge_count"]),
        -int(row["converted_boundary_edge_count"]),
        int(row["current_pair_boundary_delta"]),
        -int(row["current_pair_internal_delta"]),
        -float(row["evidence_density"]),
        len(row["added_blockers"]),
        str(row["candidate_id"]),
    )


def generate_frontierdependency_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    base_candidate: Mapping[str, Any],
    history: TemporalHistoryContext | None = None,
    maximum_candidates: int = 6,
    maximum_added_agents: int = 8,
    maximum_neighborhood_size: int = 40,
    maximum_jaccard_similarity: float = 0.9,
) -> FrontierDependencyPoolResult:
    if maximum_candidates <= 0 or maximum_added_agents <= 0:
        raise ValueError("FrontierDependencyPool limits must be positive")
    if maximum_neighborhood_size <= 1:
        raise ValueError("FrontierDependencyPool neighborhood cap is invalid")
    if not 0.0 <= maximum_jaccard_similarity < 1.0:
        raise ValueError("FrontierDependencyPool Jaccard threshold must be in [0, 1)")
    core_tuple = _normalized_agents(base_candidate["agents"])
    core = set(core_tuple)
    known_agents = {int(row["id"]) for row in state["agents"]}
    if not core <= known_agents:
        raise ValueError("FrontierDependencyPool base contains an unknown agent")
    if len(core) > maximum_neighborhood_size:
        raise ValueError("FrontierDependencyPool base exceeds its neighborhood cap")
    if not analysis.events:
        return FrontierDependencyPoolResult([], [], 0, 0, 0, 0)
    context = history or TemporalHistoryContext()
    persistent = set(map(tuple, context.persistent_conflict_edges))
    recent = set(context.recent_neighborhoods[-1]) if context.recent_neighborhoods else set()
    repair_counts = dict(context.agent_repair_counts)
    pair_events, adjacency, event_weight = _event_indexes(analysis)
    attempts: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    motif_count = 0
    for pivot in sorted(core):
        blockers = sorted(adjacency.get(pivot, set()) - core)
        if not blockers:
            continue
        motif_count += 1
        evidence = {
            blocker: _blocker_evidence(
                blocker,
                core=core,
                pivot=pivot,
                pair_events=pair_events,
                adjacency=adjacency,
                persistent=persistent,
                recent=recent,
            )
            for blocker in blockers
        }
        ranked = sorted(blockers, key=lambda agent: _blocker_priority(agent, evidence))[
            :maximum_added_agents
        ]
        if not ranked:
            continue
        augment_added = set(ranked)
        augment_size = len(core) + len(augment_added)
        if augment_size <= maximum_neighborhood_size:
            row = _variant_row(
                variant="compact-augment",
                pivot=pivot,
                core=core,
                added=augment_added,
                removed=set(),
                evidence=evidence,
                analysis=analysis,
                pair_events=pair_events,
                persistent=persistent,
            )
            raw.append(row)
            attempts.append(
                {
                    "pivot_agent": pivot,
                    "variant": "compact-augment",
                    "decision": "generated",
                    "candidate_id": row["candidate_id"],
                    "rejection_reason": None,
                }
            )
        else:
            attempts.append(
                {
                    "pivot_agent": pivot,
                    "variant": "compact-augment",
                    "decision": "rejected",
                    "candidate_id": None,
                    "rejection_reason": "maximum_neighborhood_size",
                    "proposed_size": augment_size,
                }
            )

        exchange_added: set[int] = set()
        exchange_removed: set[int] = set()
        for count in range(len(ranked), 0, -1):
            proposed = set(ranked[:count])
            protected = {pivot}
            for blocker in proposed:
                protected.update(adjacency.get(blocker, set()) & core)
            removable = sorted(
                core - protected,
                key=lambda agent: _removal_priority(
                    agent,
                    core=core,
                    adjacency=adjacency,
                    event_weight=event_weight,
                    repair_counts=repair_counts,
                ),
            )
            if len(removable) >= count:
                exchange_added = proposed
                exchange_removed = set(removable[:count])
                break
        if exchange_added:
            row = _variant_row(
                variant="same-size-exchange",
                pivot=pivot,
                core=core,
                added=exchange_added,
                removed=exchange_removed,
                evidence=evidence,
                analysis=analysis,
                pair_events=pair_events,
                persistent=persistent,
            )
            raw.append(row)
            attempts.append(
                {
                    "pivot_agent": pivot,
                    "variant": "same-size-exchange",
                    "decision": "generated",
                    "candidate_id": row["candidate_id"],
                    "rejection_reason": None,
                }
            )
        else:
            attempts.append(
                {
                    "pivot_agent": pivot,
                    "variant": "same-size-exchange",
                    "decision": "rejected",
                    "candidate_id": None,
                    "rejection_reason": "no_unprotected_core_agent",
                }
            )

    merged: dict[str, dict[str, Any]] = {}
    duplicates = 0
    for row in sorted(raw, key=_candidate_order):
        identity = str(row["candidate_id"])
        prior = merged.get(identity)
        if prior is None:
            merged[identity] = row
            continue
        duplicates += 1
        prior["pivot_agents"] = sorted(
            set(map(int, prior["pivot_agents"])) | set(map(int, row["pivot_agents"]))
        )
    by_variant = {
        variant: sorted(
            [
                row
                for row in merged.values()
                if row["frontierdependency_variant"] == variant
            ],
            key=_candidate_order,
        )
        for variant in _VARIANTS
    }
    selected: list[dict[str, Any]] = []
    rejected_jaccard = 0
    cursor = {variant: 0 for variant in _VARIANTS}
    while len(selected) < maximum_candidates:
        progressed = False
        for variant in _VARIANTS:
            rows = by_variant[variant]
            while cursor[variant] < len(rows):
                row = rows[cursor[variant]]
                cursor[variant] += 1
                maximum_similarity = max(
                    (_jaccard(row["agents"], prior["agents"]) for prior in selected),
                    default=0.0,
                )
                if maximum_similarity > maximum_jaccard_similarity:
                    rejected_jaccard += 1
                    attempts.append(
                        {
                            "pivot_agent": row["pivot_agents"][0],
                            "variant": variant,
                            "decision": "rejected",
                            "candidate_id": row["candidate_id"],
                            "rejection_reason": "candidate_jaccard",
                            "maximum_selected_jaccard": maximum_similarity,
                        }
                    )
                    continue
                selected.append(row)
                progressed = True
                break
            if len(selected) == maximum_candidates:
                break
        if not progressed:
            break
    selected.sort(
        key=lambda row: (
            _VARIANTS.index(str(row["frontierdependency_variant"])),
            _candidate_order(row),
        )
    )
    return FrontierDependencyPoolResult(
        candidates=selected,
        attempts=attempts,
        raw_motif_count=motif_count,
        raw_variant_count=len(raw),
        exact_duplicate_count=duplicates,
        jaccard_rejection_count=rejected_jaccard,
    )


def select_frontierdependency_candidate(
    base_candidate: Mapping[str, Any],
    candidates: Iterable[Mapping[str, Any]],
    *,
    variant: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if variant not in _VARIANTS:
        raise ValueError(f"unsupported FrontierDependencyPool variant: {variant}")
    eligible = [
        dict(row)
        for row in candidates
        if row.get("frontierdependencypool_id") == FRONTIERDEPENDENCYPOOL_ID
        and row.get("frontierdependency_variant") == variant
    ]
    if not eligible:
        return dict(base_candidate), {
            "selection_source": "frontierdependency_fallback",
            "rejection_reason": "no_eligible_candidate",
            "requested_variant": variant,
        }
    selected = min(eligible, key=_candidate_order)
    return selected, {
        "selection_source": f"frontierdependency_deterministic_{variant}",
        "rejection_reason": None,
        "requested_variant": variant,
        "candidate_id": str(selected["candidate_id"]),
        "ranking_key": list(_candidate_order(selected)[:-1]),
    }


__all__ = [
    "FRONTIERDEPENDENCYPOOL_ID",
    "FrontierDependencyPoolResult",
    "generate_frontierdependency_candidates",
    "select_frontierdependency_candidate",
]
