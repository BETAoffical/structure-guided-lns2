from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Iterable

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import ConflictEvent, StateAnalysis
from lns2_selector.runtime.topology_candidates import _jaccard, topology_candidate_audit


CAUSALCLOSUREPOOL_ID = "stride-causalclosurepool-v2"


@dataclass(frozen=True)
class CausalContactEvidence:
    conflict_events: int = 0
    exact_reservations: int = 0
    nearby_reservations: int = 0
    reverse_edges: int = 0
    bottleneck_contacts: int = 0

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.conflict_events),
            int(self.exact_reservations),
            int(self.nearby_reservations),
            int(self.reverse_edges),
            int(self.bottleneck_contacts),
        )


@dataclass(frozen=True)
class _CausalContext:
    paths: dict[int, tuple[int, ...]]
    horizon: int
    occupancy: dict[tuple[int, int], frozenset[int]]
    transitions: dict[tuple[int, int, int], frozenset[int]]
    conflict_adjacency: dict[int, frozenset[int]]
    events_by_agent: dict[int, tuple[ConflictEvent, ...]]


@dataclass(frozen=True)
class _CausalDraft:
    agents: tuple[int, ...]
    core_id: str
    core_agents: tuple[int, ...]
    family: str
    closure_depth: int
    causal_time_count: int
    support_edge_count: int
    evidence_by_agent: tuple[tuple[int, tuple[int, int, int, int, int]], ...]


@dataclass
class CausalClosurePoolResult:
    candidates: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    raw_candidate_count: int
    pareto_front_count: int
    oversized_family_count: int
    family_attempt_count: int
    oversized_core_count: int
    core_attempt_count: int


def _causal_context(
    state: dict[str, Any], *, relevant_times: frozenset[int], horizon: int
) -> _CausalContext:
    rows = {int(row["id"]): row for row in state["agents"]}
    if not rows:
        raise ValueError("CausalClosurePool requires agents")
    if not relevant_times or min(relevant_times) < 0:
        raise ValueError("CausalClosurePool requires non-negative relevant times")
    raw_paths = {
        agent: tuple(map(int, row["path"])) for agent, row in rows.items()
    }
    if any(not path for path in raw_paths.values()):
        raise ValueError("CausalClosurePool requires non-empty paths")
    if horizon <= max(relevant_times):
        raise ValueError("CausalClosurePool relevant times exceed the path horizon")
    ordered_times = tuple(sorted(relevant_times))
    # PP semantics keep an agent at its terminal cell after its explicit path
    # ends.  Materialize that extension once so the thousands of causal-contact
    # lookups below can use direct tuple indexing instead of repeatedly taking
    # ``min(time, len(path) - 1)``.  The padded paths are internal only and do
    # not change candidate identity or any recorded path.
    paths = {
        agent: path + (path[-1],) * (horizon - len(path))
        for agent, path in raw_paths.items()
    }
    occupancy: dict[tuple[int, int], set[int]] = collections.defaultdict(set)
    transitions: dict[tuple[int, int, int], set[int]] = collections.defaultdict(set)
    for agent, path in paths.items():
        for time in ordered_times:
            cell = path[time]
            occupancy[(time, cell)].add(agent)
            previous = path[time - 1] if time else cell
            if time and previous != cell:
                transitions[(time, previous, cell)].add(agent)
    adjacency: dict[int, set[int]] = {agent: set() for agent in paths}
    for edge in state.get("conflict_edges", []):
        left, right = sorted(map(int, edge))
        if left not in paths or right not in paths:
            raise ValueError("CausalClosurePool conflict edge references unknown agent")
        adjacency[left].add(right)
        adjacency[right].add(left)
    events_by_agent: dict[int, list[ConflictEvent]] = collections.defaultdict(list)
    return _CausalContext(
        paths=paths,
        horizon=horizon,
        occupancy={key: frozenset(value) for key, value in occupancy.items()},
        transitions={key: frozenset(value) for key, value in transitions.items()},
        conflict_adjacency={key: frozenset(value) for key, value in adjacency.items()},
        events_by_agent=events_by_agent,
    )


def _with_events(
    context: _CausalContext, analysis: StateAnalysis
) -> _CausalContext:
    events_by_agent: dict[int, list[ConflictEvent]] = collections.defaultdict(list)
    for event in analysis.events:
        events_by_agent[int(event.left)].append(event)
        events_by_agent[int(event.right)].append(event)
    return _CausalContext(
        paths=context.paths,
        horizon=context.horizon,
        occupancy=context.occupancy,
        transitions=context.transitions,
        conflict_adjacency=context.conflict_adjacency,
        events_by_agent={
            agent: tuple(sorted(events, key=lambda item: (item.time, item.kind, item.left, item.right)))
            for agent, events in events_by_agent.items()
        },
    )


def _localized_conflict_closure(
    context: _CausalContext,
    core: set[int],
    seed_time: int,
    event_window: int,
) -> tuple[set[int], frozenset[int], dict[int, CausalContactEvidence]]:
    eligible_times = set(
        range(
            max(0, int(seed_time) - int(event_window)),
            min(context.horizon, int(seed_time) + int(event_window) + 1),
        )
    )
    selected = set(core)
    stack = list(core)
    evidence: dict[int, CausalContactEvidence] = {}
    while stack:
        agent = stack.pop()
        for event in context.events_by_agent.get(agent, ()):
            if int(event.time) not in eligible_times:
                continue
            other = int(event.right) if int(event.left) == agent else int(event.left)
            if other in selected:
                continue
            selected.add(other)
            stack.append(other)
            evidence[other] = _merge_evidence(
                evidence.get(other), conflict_events=1
            )
    return selected, frozenset(eligible_times), evidence


def _merge_evidence(
    previous: CausalContactEvidence | None,
    *,
    conflict_events: int = 0,
    exact_reservations: int = 0,
    nearby_reservations: int = 0,
    reverse_edges: int = 0,
    bottleneck_contacts: int = 0,
) -> CausalContactEvidence:
    previous = previous or CausalContactEvidence()
    return CausalContactEvidence(
        conflict_events=previous.conflict_events + conflict_events,
        exact_reservations=previous.exact_reservations + exact_reservations,
        nearby_reservations=previous.nearby_reservations + nearby_reservations,
        reverse_edges=previous.reverse_edges + reverse_edges,
        bottleneck_contacts=previous.bottleneck_contacts + bottleneck_contacts,
    )


def _temporal_neighbors(
    context: _CausalContext,
    analysis: StateAnalysis,
    agent: int,
    causal_times: frozenset[int],
    *,
    temporal_radius: int,
    bottleneck_only: bool,
) -> dict[int, CausalContactEvidence]:
    # Accumulate primitive counters in place and instantiate the immutable
    # evidence records only once per neighbor.  The former implementation
    # rebuilt a dataclass on every reservation contact, which dominates this
    # hot loop on long Maze paths.
    counters: dict[int, list[int]] = {}
    path = context.paths[agent]
    for source_time in causal_times:
        cell = path[source_time]
        bottleneck = cell in analysis.articulation or int(analysis.degrees.get(cell, 4)) <= 2
        if bottleneck_only and not bottleneck:
            continue
        for delta in range(-temporal_radius, temporal_radius + 1):
            other_time = source_time + delta
            if other_time not in causal_times:
                continue
            for other in context.occupancy.get((other_time, cell), ()):
                if other == agent:
                    continue
                values = counters.setdefault(other, [0, 0, 0, 0, 0])
                values[1] += int(delta == 0)
                values[2] += int(delta != 0)
                values[4] += int(bottleneck)
            if source_time == 0 or other_time == 0:
                continue
            previous = path[source_time - 1]
            if previous == cell:
                continue
            for other in context.transitions.get((other_time, cell, previous), ()):
                if other == agent:
                    continue
                values = counters.setdefault(other, [0, 0, 0, 0, 0])
                values[1] += int(delta == 0)
                values[2] += int(delta != 0)
                values[3] += 1
                values[4] += int(bottleneck)
    return {
        other: CausalContactEvidence(*values)
        for other, values in counters.items()
    }


def _closure_draft(
    context: _CausalContext,
    analysis: StateAnalysis,
    *,
    core_id: str,
    core_agents: tuple[int, ...],
    seed_time: int,
    family: str,
    temporal_radius: int | None,
    bottleneck_only: bool,
    event_window: int,
    maximum_neighborhood_size: int,
    localized: tuple[
        set[int], frozenset[int], dict[int, CausalContactEvidence]
    ]
    | None = None,
    temporal_neighbor_cache: dict[
        tuple[int, frozenset[int], int, bool],
        dict[int, CausalContactEvidence],
    ]
    | None = None,
) -> tuple[_CausalDraft | None, dict[str, Any] | None]:
    core = set(core_agents)
    if localized is None:
        localized = _localized_conflict_closure(
            context, core, seed_time, event_window
        )
    selected = set(localized[0])
    causal_times = localized[1]
    evidence = dict(localized[2])
    if len(selected) > maximum_neighborhood_size:
        return None, {
            "core_id": core_id,
            "family": family,
            "seed_time": seed_time,
            "decision": "rejected",
            "rejection_reason": "causal_closure_oversized",
            "proposed_size": len(selected),
        }
    for agent in selected - core:
        direct = sum(
            int(event.time) in causal_times
            and (
                (event.left in selected and event.right == agent)
                or (event.right in selected and event.left == agent)
            )
            for event in context.events_by_agent.get(agent, ())
        )
        evidence[agent] = _merge_evidence(
            evidence.get(agent), conflict_events=direct
        )
    depth = 0
    if temporal_radius is not None and causal_times:
        frontier = set(selected)
        while frontier:
            additions: dict[int, CausalContactEvidence] = {}
            for agent in sorted(frontier):
                cache_key = (
                    int(agent),
                    causal_times,
                    int(temporal_radius),
                    bool(bottleneck_only),
                )
                supports = (
                    temporal_neighbor_cache.get(cache_key)
                    if temporal_neighbor_cache is not None
                    else None
                )
                if supports is None:
                    supports = _temporal_neighbors(
                        context,
                        analysis,
                        agent,
                        causal_times,
                        temporal_radius=temporal_radius,
                        bottleneck_only=bottleneck_only,
                    )
                    if temporal_neighbor_cache is not None:
                        temporal_neighbor_cache[cache_key] = supports
                for other, support in supports.items():
                    if other in selected:
                        continue
                    additions[other] = _merge_evidence(
                        additions.get(other),
                        exact_reservations=support.exact_reservations,
                        nearby_reservations=support.nearby_reservations,
                        reverse_edges=support.reverse_edges,
                        bottleneck_contacts=support.bottleneck_contacts,
                    )
            if not additions:
                break
            proposed = selected | set(additions)
            if len(proposed) > maximum_neighborhood_size:
                return None, {
                    "core_id": core_id,
                    "family": family,
                    "seed_time": seed_time,
                    "decision": "rejected",
                    "rejection_reason": "causal_closure_oversized",
                    "proposed_size": len(proposed),
                    "closure_depth": depth + 1,
                }
            depth += 1
            selected = proposed
            frontier = set(additions)
            for other, support in additions.items():
                evidence[other] = _merge_evidence(
                    evidence.get(other),
                    exact_reservations=support.exact_reservations,
                    nearby_reservations=support.nearby_reservations,
                    reverse_edges=support.reverse_edges,
                    bottleneck_contacts=support.bottleneck_contacts,
                )
    support_edge_count = sum(sum(value.as_tuple()) for value in evidence.values())
    return _CausalDraft(
        agents=tuple(sorted(selected)),
        core_id=core_id,
        core_agents=tuple(sorted(core)),
        family=family,
        closure_depth=depth,
        causal_time_count=len(causal_times),
        support_edge_count=support_edge_count,
        evidence_by_agent=tuple(
            sorted((agent, value.as_tuple()) for agent, value in evidence.items())
        ),
    ), None


def _materialize(
    analysis: StateAnalysis,
    drafts: list[_CausalDraft],
    agent_count: int,
) -> dict[str, Any]:
    representative = min(
        drafts,
        key=lambda draft: (
            draft.closure_depth,
            draft.family,
            draft.core_id,
        ),
    )
    audit = topology_candidate_audit(analysis, representative.agents)
    evidence: dict[int, list[int]] = {}
    for draft in drafts:
        for agent, values in draft.evidence_by_agent:
            current = evidence.setdefault(agent, [0, 0, 0, 0, 0])
            for index, value in enumerate(values):
                current[index] = max(current[index], int(value))
    families = sorted({draft.family for draft in drafts})
    provenance = sorted({f"{draft.core_id}:{draft.family}" for draft in drafts})
    added = max(0, len(representative.agents) - len(representative.core_agents))
    support = max(draft.support_edge_count for draft in drafts)
    return {
        "candidate_id": candidate_id(representative.agents),
        "agents": list(representative.agents),
        "actual_size": len(representative.agents),
        "selection_families": [f"causalclosure:{item}" for item in provenance],
        "selection_rank_by_family": {
            f"causalclosure:{item}": 0 for item in provenance
        },
        "proposal_count_by_family": {
            f"causalclosure:{item}": 1 for item in provenance
        },
        "proposal_seeds": [],
        "seed_agents": list(representative.core_agents),
        "structpool_family_groups": ["localized_causal_repair_closure"],
        "candidate_kind": "causalclosure",
        "causalclosurepool_id": CAUSALCLOSUREPOOL_ID,
        "causalclosure_core_ids": sorted({draft.core_id for draft in drafts}),
        "causalclosure_core_agents": list(representative.core_agents),
        "causalclosure_families": families,
        "causalclosure_depth": max(draft.closure_depth for draft in drafts),
        "causalclosure_causal_time_count": max(
            draft.causal_time_count for draft in drafts
        ),
        "causalclosure_support_edge_count": support,
        "causalclosure_added_agent_count": added,
        "causalclosure_support_per_added_agent": support / max(1, added),
        "causalclosure_evidence_by_agent": {
            str(agent): {
                "conflict_events": values[0],
                "exact_reservations": values[1],
                "nearby_reservations": values[2],
                "reverse_edges": values[3],
                "bottleneck_contacts": values[4],
            }
            for agent, values in sorted(evidence.items())
        },
        "proposal_audit": audit,
        "neighborhood_fraction": len(representative.agents) / max(1, agent_count),
    }


def _candidate_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_audit = left["proposal_audit"]
    right_audit = right["proposal_audit"]
    comparisons = (
        int(left["actual_size"]) <= int(right["actual_size"]),
        float(left["causalclosure_support_per_added_agent"])
        >= float(right["causalclosure_support_per_added_agent"]),
        float(left_audit["global_event_incident_coverage"])
        >= float(right_audit["global_event_incident_coverage"]),
        float(left_audit["global_pair_internal_coverage"])
        >= float(right_audit["global_pair_internal_coverage"]),
    )
    strict = (
        int(left["actual_size"]) < int(right["actual_size"])
        or float(left["causalclosure_support_per_added_agent"])
        > float(right["causalclosure_support_per_added_agent"])
        or float(left_audit["global_event_incident_coverage"])
        > float(right_audit["global_event_incident_coverage"])
        or float(left_audit["global_pair_internal_coverage"])
        > float(right_audit["global_pair_internal_coverage"])
    )
    return all(comparisons) and strict


def _dominance_vector(row: dict[str, Any]) -> tuple[int, float, float, float]:
    """Return minimization coordinates for the frozen causal Pareto order."""

    audit = row["proposal_audit"]
    return (
        int(row["actual_size"]),
        -float(row["causalclosure_support_per_added_agent"]),
        -float(audit["global_event_incident_coverage"]),
        -float(audit["global_pair_internal_coverage"]),
    )


def _front_ranks(rows: list[dict[str, Any]]) -> dict[str, int]:
    identities = [str(row["candidate_id"]) for row in rows]
    # Extract the four immutable comparison coordinates once.  The former
    # pair loop repeatedly traversed nested dictionaries and called
    # ``_candidate_dominates`` up to twice for every pair.  Tuple coordinates
    # preserve the exact weak/strict Pareto relation while keeping the O(n^2)
    # contract deterministic and substantially reducing Python overhead.
    vectors = [_dominance_vector(row) for row in rows]
    dominates: list[list[int]] = [[] for _ in rows]
    dominated_count = [0 for _ in rows]
    for left_index, left in enumerate(vectors):
        for right_index in range(left_index + 1, len(rows)):
            right = vectors[right_index]
            different = left != right
            if different and (
                left[0] <= right[0]
                and left[1] <= right[1]
                and left[2] <= right[2]
                and left[3] <= right[3]
            ):
                dominates[left_index].append(right_index)
                dominated_count[right_index] += 1
            elif different and (
                right[0] <= left[0]
                and right[1] <= left[1]
                and right[2] <= left[2]
                and right[3] <= left[3]
            ):
                dominates[right_index].append(left_index)
                dominated_count[left_index] += 1

    front = [index for index, count in enumerate(dominated_count) if count == 0]
    ranks: dict[str, int] = {}
    rank = 0
    assigned = 0
    while front:
        next_front: list[int] = []
        for index in front:
            ranks[identities[index]] = rank
            assigned += 1
            for dominated in dominates[index]:
                dominated_count[dominated] -= 1
                if dominated_count[dominated] == 0:
                    next_front.append(dominated)
        front = next_front
        rank += 1
    if assigned != len(rows):
        raise RuntimeError("CausalClosurePool candidate dominance produced no front")
    return ranks


def generate_causalclosure_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_anchors: Iterable[dict[str, Any]],
    maximum_candidates: int = 12,
    maximum_neighborhood_size: int = 64,
    temporal_window: int = 2,
    maximum_jaccard_similarity: float = 0.9,
) -> CausalClosurePoolResult:
    if maximum_candidates <= 0 or maximum_neighborhood_size <= 0:
        raise ValueError("CausalClosurePool limits must be positive")
    if temporal_window < 0:
        raise ValueError("CausalClosurePool temporal window must be non-negative")
    if not 0.0 <= maximum_jaccard_similarity < 1.0:
        raise ValueError("CausalClosurePool Jaccard threshold must be in [0, 1)")
    if not analysis.events:
        return CausalClosurePoolResult([], [], 0, 0, 0, 0, 0, 0)
    event_times = {int(event.time) for event in analysis.events}
    horizon = max(len(row["path"]) for row in state["agents"])
    # A localized conflict closure first spans ``temporal_window`` around an
    # event.  The near-temporal families can then inspect another
    # ``temporal_window`` around every time in that closure.  Index the full
    # composed window so the sparse context remains semantically identical to
    # the former all-horizon context.
    indexed_radius = 2 * temporal_window
    relevant_times = frozenset(
        time
        for seed_time in event_times
        for time in range(
            max(0, seed_time - indexed_radius),
            min(horizon, seed_time + indexed_radius + 1),
        )
    )
    context = _with_events(
        _causal_context(
            state,
            relevant_times=relevant_times,
            horizon=horizon,
        ),
        analysis,
    )
    agent_count = len(state["agents"])
    base_sets: dict[tuple[int, ...], str] = {}
    cores: list[tuple[str, tuple[int, ...], int]] = []
    for anchor in v2_anchors:
        agents = tuple(sorted(set(map(int, anchor.get("agents") or ()))))
        if not agents:
            raise ValueError("CausalClosurePool V2 anchor is empty")
        base_sets[agents] = str(anchor["candidate_id"])
        event_times = sorted(
            {
                int(event.time)
                for agent in agents
                for event in context.events_by_agent.get(agent, ())
            }
        )
        for seed_time in event_times:
            cores.append(
                (
                    f"v2_{str(anchor['candidate_id'])[-8:]}_time_{seed_time}",
                    agents,
                    seed_time,
                )
            )
    events_by_time: dict[int, list[ConflictEvent]] = collections.defaultdict(list)
    for event in analysis.events:
        events_by_time[int(event.time)].append(event)
    for seed_time, events in sorted(events_by_time.items()):
        adjacency: dict[int, set[int]] = collections.defaultdict(set)
        for event in events:
            adjacency[int(event.left)].add(int(event.right))
            adjacency[int(event.right)].add(int(event.left))
        unvisited = set(adjacency)
        slice_index = 0
        while unvisited:
            start = min(unvisited)
            members = {start}
            stack = [start]
            unvisited.remove(start)
            while stack:
                agent = stack.pop()
                for other in adjacency[agent]:
                    if other in unvisited:
                        unvisited.remove(other)
                        members.add(other)
                        stack.append(other)
            cores.append(
                (
                    f"event_time_{seed_time}_component_{slice_index}",
                    tuple(sorted(members)),
                    seed_time,
                )
            )
            slice_index += 1

    family_specs = (
        ("direct_conflict", None, False),
        ("exact_temporal", 0, False),
        ("bottleneck_exact", 0, True),
        ("near_temporal", temporal_window, False),
        ("bottleneck_near", temporal_window, True),
    )
    attempts: list[dict[str, Any]] = []
    drafts: list[_CausalDraft] = []
    temporal_neighbor_cache: dict[
        tuple[int, frozenset[int], int, bool],
        dict[int, CausalContactEvidence],
    ] = {}
    oversized = 0
    oversized_cores = 0
    feasible_core_count = 0
    for core_id, core_agents, seed_time in cores:
        if len(core_agents) > maximum_neighborhood_size:
            oversized_cores += 1
            attempts.append(
                {
                    "core_id": core_id,
                    "seed_time": seed_time,
                    "decision": "rejected",
                    "rejection_reason": "core_oversized",
                    "proposed_size": len(core_agents),
                }
            )
            continue
        feasible_core_count += 1
        localized = _localized_conflict_closure(
            context,
            set(core_agents),
            seed_time,
            temporal_window,
        )
        for family, radius, bottleneck_only in family_specs:
            draft, rejection = _closure_draft(
                context,
                analysis,
                core_id=core_id,
                core_agents=core_agents,
                seed_time=seed_time,
                family=family,
                temporal_radius=radius,
                bottleneck_only=bottleneck_only,
                event_window=temporal_window,
                maximum_neighborhood_size=maximum_neighborhood_size,
                localized=localized,
                temporal_neighbor_cache=temporal_neighbor_cache,
            )
            if rejection is not None:
                attempts.append(rejection)
                oversized += 1
            elif draft is not None:
                drafts.append(draft)
    grouped: dict[tuple[int, ...], list[_CausalDraft]] = collections.defaultdict(list)
    for draft in drafts:
        grouped[draft.agents].append(draft)
    rows: list[dict[str, Any]] = []
    for agents, variants in sorted(grouped.items()):
        duplicate = base_sets.get(agents)
        if duplicate is not None:
            attempts.append(
                {
                    "candidate_id": candidate_id(agents),
                    "decision": "rejected",
                    "rejection_reason": "duplicate_v2_anchor",
                    "duplicate_candidate_id": duplicate,
                }
            )
            continue
        rows.append(_materialize(analysis, variants, agent_count))
    if not rows:
        return CausalClosurePoolResult(
            [],
            attempts,
            0,
            0,
            oversized,
            feasible_core_count * len(family_specs),
            oversized_cores,
            len(cores),
        )
    ranks = _front_ranks(rows)
    for row in rows:
        row["causalclosure_pareto_rank"] = ranks[str(row["candidate_id"])]

    def order_key(row: dict[str, Any]) -> tuple[Any, ...]:
        audit = row["proposal_audit"]
        return (
            int(row["causalclosure_pareto_rank"]),
            int(row["actual_size"]),
            -float(audit["global_event_incident_coverage"]),
            -float(audit["global_pair_internal_coverage"]),
            -float(row["causalclosure_support_per_added_agent"]),
            str(row["candidate_id"]),
        )

    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=order_key):
        if len(selected) >= maximum_candidates:
            break
        similarity = max(
            (_jaccard(row["agents"], other["agents"]) for other in selected),
            default=0.0,
        )
        if similarity > maximum_jaccard_similarity:
            attempts.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "decision": "rejected",
                    "rejection_reason": "candidate_jaccard",
                    "maximum_selected_jaccard": similarity,
                }
            )
            continue
        selected.append(row)
        attempts.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "decision": "selected",
                "rejection_reason": None,
                "maximum_selected_jaccard": similarity,
            }
        )
    return CausalClosurePoolResult(
        candidates=selected,
        attempts=attempts,
        raw_candidate_count=len(rows),
        pareto_front_count=sum(rank == 0 for rank in ranks.values()),
        oversized_family_count=oversized,
        family_attempt_count=feasible_core_count * len(family_specs),
        oversized_core_count=oversized_cores,
        core_attempt_count=len(cores),
    )


__all__ = [
    "CAUSALCLOSUREPOOL_ID",
    "CausalClosurePoolResult",
    "CausalContactEvidence",
    "generate_causalclosure_candidates",
]
