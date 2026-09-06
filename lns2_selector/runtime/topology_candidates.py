from __future__ import annotations

import collections
import copy
import heapq
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import ConflictEvent, StateAnalysis


@dataclass(frozen=True)
class _NeighborhoodContext:
    agent_rows: dict[int, dict[str, Any]]
    agent_ids: frozenset[int]
    adjacency: dict[int, frozenset[int]]


@dataclass
class StructuralCandidateContext:
    """State-level inputs shared by every structural candidate family.

    The context deliberately contains only current-state, outcome-free data.
    Mutable caches are local to one decision and never cross repair states.
    """

    state: dict[str, Any]
    analysis: StateAnalysis
    neighborhood: _NeighborhoodContext
    path_sets: dict[int, frozenset[int]]
    event_weights: collections.Counter[int]
    partner_counts: dict[int, collections.Counter[int]]
    component_internal_events: dict[int, int]
    relevant_events_by_kind: dict[str, list[ConflictEvent]]
    bottleneck_events: list[ConflictEvent]
    component_seed_data: tuple[set[int], collections.Counter[int]] | None
    hotspot_seed_data: tuple[set[int], collections.Counter[int]] | None
    overlap_seed_data: tuple[list[int], dict[int, float]] | None
    audit_index: TopologyCandidateAuditIndex
    audit_cache: dict[tuple[int, ...], dict[str, float]]
    finalized_cache: dict[tuple[int, ...], dict[str, Any]]

    @property
    def agent_rows(self) -> dict[int, dict[str, Any]]:
        return self.neighborhood.agent_rows

    @property
    def conflict_adjacency(self) -> dict[int, frozenset[int]]:
        return self.neighborhood.adjacency


@dataclass(frozen=True)
class StructuralCandidateDraft:
    """A structural action before audit and score materialization."""

    agents: tuple[int, ...]
    family_group: str
    family: str
    nominal_size: int


@dataclass(frozen=True)
class _MergedStructuralCandidate:
    agents: tuple[int, ...]
    selection_families: tuple[str, ...]
    family_groups: tuple[str, ...]
    incumbent_boundary: bool




@dataclass(frozen=True)
class TopologyCandidateAuditIndex:
    """Conflict incidence reused by every candidate audited in one state."""

    event_neighbors: dict[int, dict[int, int]]
    event_incident_counts: dict[int, int]
    pair_neighbors: dict[int, frozenset[int]]
    event_count: int
    pair_count: int


def _neighborhood_context(state: dict[str, Any]) -> _NeighborhoodContext:
    agent_rows = {int(agent["id"]): agent for agent in state["agents"]}
    adjacency: dict[int, set[int]] = {
        agent: set() for agent in agent_rows
    }
    for edge in state.get("conflict_edges", []):
        left, right = map(int, edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    return _NeighborhoodContext(
        agent_rows=agent_rows,
        agent_ids=frozenset(agent_rows),
        adjacency={agent: frozenset(peers) for agent, peers in adjacency.items()},
    )


def _relevant_events(
    analysis: StateAnalysis, kind: str
) -> list[ConflictEvent]:
    if kind == "articulation":
        return [
            event
            for event in analysis.events
            if any(cell in analysis.articulation for cell in event.cells)
        ]
    if kind == "low_degree":
        low_degree = {
            cell
            for cell in analysis.free_cells
            if int(analysis.degrees.get(cell, 0)) <= 2
        }
        return [
            event
            for event in analysis.events
            if any(cell in low_degree for cell in event.cells)
        ]
    raise ValueError(f"unsupported topology anchor kind: {kind}")


def _fill_neighborhood(
    selected: set[int],
    state: dict[str, Any],
    event_weight: collections.Counter[int],
    size: int,
    *,
    context: _NeighborhoodContext | None = None,
) -> list[int]:
    context = context or _neighborhood_context(state)
    agent_rows = context.agent_rows
    if not selected <= context.agent_ids:
        raise ValueError("topology anchor selected an unknown agent")
    limit = min(size, len(agent_rows))
    if len(selected) >= limit:
        return sorted(selected)

    remaining = set(context.agent_ids) - selected
    selected_neighbor_count = {
        agent: len(context.adjacency[agent] & selected) for agent in remaining
    }

    def priority(agent: int) -> tuple[int, int, int, int]:
        return (
            -selected_neighbor_count[agent],
            -int(agent_rows[agent].get("conflict_degree", 0)),
            -int(event_weight[agent]),
            agent,
        )

    heap = [priority(agent) for agent in remaining]
    heapq.heapify(heap)
    while len(selected) < limit:
        while True:
            entry = heapq.heappop(heap)
            chosen = int(entry[-1])
            if chosen not in remaining:
                continue
            current = priority(chosen)
            if entry != current:
                heapq.heappush(heap, current)
                continue
            break
        remaining.remove(chosen)
        selected.add(chosen)
        for neighbor in context.adjacency[chosen]:
            if neighbor not in remaining:
                continue
            selected_neighbor_count[neighbor] += 1
            heapq.heappush(heap, priority(neighbor))
    return sorted(selected)


def _anchor_neighborhood(
    state: dict[str, Any],
    events: list[ConflictEvent],
    size: int,
    *,
    context: _NeighborhoodContext | None = None,
) -> list[int]:
    if size <= 0 or not events:
        raise ValueError("topology anchor requires events and a positive size")
    uncovered = len(events)
    selected: set[int] = set()
    event_weight: collections.Counter[int] = collections.Counter()
    pair_event_count: collections.Counter[tuple[int, int]] = collections.Counter()
    partner_event_count: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    for event in events:
        event_weight[event.left] += 1
        event_weight[event.right] += 1
        pair_event_count[(event.left, event.right)] += 1
        partner_event_count[event.left][event.right] += 1
        partner_event_count[event.right][event.left] += 1
    uncovered_weight = collections.Counter(event_weight)
    pairs = sorted({(event.left, event.right) for event in events})
    while uncovered and len(selected) < size:
        best_score: tuple[float, int, int, int, int, int] | None = None
        best_addition: set[int] | None = None
        best_covered = 0
        for left, right in pairs:
            addition = {left, right} - selected
            if not addition or len(selected) + len(addition) > size:
                continue
            internal = (
                int(pair_event_count[(left, right)])
                if len(addition) == 2
                else 0
            )
            covered = sum(uncovered_weight[agent] for agent in addition) - internal
            score = (
                covered / len(addition),
                covered,
                internal,
                sum(event_weight[agent] for agent in addition),
                -left,
                -right,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_addition = addition
                best_covered = covered
        if best_addition is None:
            break
        selected.update(best_addition)
        uncovered -= best_covered
        for agent in best_addition:
            uncovered_weight[agent] = 0
        for agent in best_addition:
            for other, count in partner_event_count[agent].items():
                if other not in selected:
                    uncovered_weight[other] -= count
    return _fill_neighborhood(
        selected, state, event_weight, size, context=context
    )


def generate_topology_anchor_candidates(
    state: dict[str, Any], analysis: StateAnalysis, neighborhood_sizes: Iterable[int]
) -> list[dict[str, Any]]:
    sizes = sorted(set(map(int, neighborhood_sizes)))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("topology anchor sizes must be positive")
    by_agents: dict[tuple[int, ...], dict[str, Any]] = {}
    for kind in ("articulation", "low_degree"):
        events = _relevant_events(analysis, kind)
        if not events:
            continue
        for size in sizes:
            agents = tuple(_anchor_neighborhood(state, events, size))
            family = f"topology-anchor-{kind}:{size}"
            row = by_agents.setdefault(
                agents,
                {
                    "candidate_id": candidate_id(agents),
                    "agents": list(agents),
                    "actual_size": len(agents),
                    "selection_families": [],
                    "selection_rank_by_family": {},
                    "proposal_count_by_family": {},
                    "proposal_seeds": [],
                    "seed_agents": [],
                },
            )
            row["selection_families"].append(family)
            row["selection_rank_by_family"][family] = 0
            row["proposal_count_by_family"][family] = 1
    for row in by_agents.values():
        row["selection_families"].sort()
        row["selection_rank_by_family"] = dict(
            sorted(row["selection_rank_by_family"].items())
        )
        row["proposal_count_by_family"] = dict(
            sorted(row["proposal_count_by_family"].items())
        )
    return sorted(by_agents.values(), key=lambda row: str(row["candidate_id"]))


def topology_candidate_audit_index(
    analysis: StateAnalysis,
) -> TopologyCandidateAuditIndex:
    event_neighbors: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    event_incident_counts: collections.Counter[int] = collections.Counter()
    for event in analysis.events:
        left = int(event.left)
        right = int(event.right)
        event_neighbors[left][right] += 1
        event_neighbors[right][left] += 1
        event_incident_counts[left] += 1
        event_incident_counts[right] += 1
    pair_neighbors: dict[int, set[int]] = collections.defaultdict(set)
    for left, right in analysis.pair_set:
        left = int(left)
        right = int(right)
        pair_neighbors[left].add(right)
        pair_neighbors[right].add(left)
    return TopologyCandidateAuditIndex(
        event_neighbors={
            agent: dict(counts) for agent, counts in event_neighbors.items()
        },
        event_incident_counts=dict(event_incident_counts),
        pair_neighbors={
            agent: frozenset(neighbors)
            for agent, neighbors in pair_neighbors.items()
        },
        event_count=len(analysis.events),
        pair_count=len(analysis.pair_set),
    )


def topology_candidate_audit(
    analysis: StateAnalysis,
    selected_agents: Iterable[int],
    *,
    audit_index: TopologyCandidateAuditIndex | None = None,
) -> dict[str, float]:
    """Return outcome-free conflict coverage diagnostics for one neighborhood."""

    selected = set(map(int, selected_agents))
    if not selected:
        raise ValueError("topology candidate audit requires a non-empty neighborhood")
    known = set(analysis.component_id)
    audit_index = audit_index or topology_candidate_audit_index(analysis)
    event_internal = sum(
        count
        for left in selected
        for right, count in audit_index.event_neighbors.get(left, {}).items()
        if left < right and right in selected
    )
    event_degree_sum = sum(
        audit_index.event_incident_counts.get(agent, 0) for agent in selected
    )
    event_boundary = event_degree_sum - 2 * event_internal
    event_incident = event_degree_sum - event_internal
    pair_internal = sum(
        1
        for left in selected
        for right in audit_index.pair_neighbors.get(left, ())
        if left < right and right in selected
    )
    pair_degree_sum = sum(
        len(audit_index.pair_neighbors.get(agent, ())) for agent in selected
    )
    pair_boundary = pair_degree_sum - 2 * pair_internal
    pair_incident = pair_degree_sum - pair_internal
    reached_components = {
        int(analysis.component_id[agent]) for agent in selected if agent in known
    }
    component_count = len(analysis.component_members)
    return {
        "global_event_incident_coverage": event_incident / audit_index.event_count if audit_index.event_count else 0.0,
        "global_event_internal_coverage": event_internal / audit_index.event_count if audit_index.event_count else 0.0,
        "global_event_boundary_ratio": event_boundary / event_incident if event_incident else 0.0,
        "global_pair_incident_coverage": pair_incident / audit_index.pair_count if audit_index.pair_count else 0.0,
        "global_pair_internal_coverage": pair_internal / audit_index.pair_count if audit_index.pair_count else 0.0,
        "global_pair_boundary_ratio": pair_boundary / pair_incident if pair_incident else 0.0,
        "conflict_component_reach": (
            len(reached_components) / component_count if component_count else 0.0
        ),
    }


def _boundary_neighborhood(
    state: dict[str, Any],
    analysis: StateAnalysis,
    events: list[ConflictEvent],
    *,
    size: int,
    core_budget: int,
    context: _NeighborhoodContext | None = None,
) -> list[int]:
    """Select incident endpoints while discouraging closure of covered conflicts."""

    context = context or _neighborhood_context(state)
    agent_rows = context.agent_rows
    if size <= 0 or core_budget <= 0 or not events:
        raise ValueError("topology boundary candidate requires events and positive budgets")
    selected: set[int] = set()
    selected_components: set[int] = set()

    def incident_index(
        scored_events: list[ConflictEvent],
    ) -> tuple[
        dict[int, int], dict[int, collections.Counter[int]]
    ]:
        totals: collections.Counter[int] = collections.Counter()
        partners: dict[int, collections.Counter[int]] = collections.defaultdict(
            collections.Counter
        )
        for event in scored_events:
            left = int(event.left)
            right = int(event.right)
            totals[left] += 1
            totals[right] += 1
            partners[left][right] += 1
            partners[right][left] += 1
        return dict(totals), dict(partners)

    relevant_totals, relevant_partners = incident_index(events)
    if events is analysis.events:
        all_totals = relevant_totals
        all_partners = relevant_partners
    else:
        all_totals, all_partners = incident_index(analysis.events)
    relevant_selected_events: collections.Counter[int] = collections.Counter()
    all_selected_events = (
        relevant_selected_events
        if events is analysis.events
        else collections.Counter()
    )

    def score(
        agent: int,
        totals: dict[int, int],
        selected_events: collections.Counter[int],
    ) -> tuple[int, int, int, int, int, int]:
        newly_internal = int(selected_events[agent])
        newly_incident = int(totals.get(agent, 0)) - newly_internal
        component = analysis.component_id.get(agent)
        component_novel = int(
            component is not None and component not in selected_components
        )
        return (
            3 * newly_incident - newly_internal,
            newly_incident,
            -newly_internal,
            component_novel,
            int(agent_rows[agent].get("conflict_degree", 0)),
            -agent,
        )

    def priority(
        agent: int,
        totals: dict[int, int],
        selected_events: collections.Counter[int],
    ) -> tuple[int, int, int, int, int, int]:
        return tuple(-value for value in score(agent, totals, selected_events))

    def candidate_heap(
        available: set[int],
        totals: dict[int, int],
        selected_events: collections.Counter[int],
    ) -> list[tuple[int, int, int, int, int, int]]:
        heap = [priority(agent, totals, selected_events) for agent in available]
        heapq.heapify(heap)
        return heap

    def choose(
        heap: list[tuple[int, int, int, int, int, int]],
        available: set[int],
        totals: dict[int, int],
        selected_events: collections.Counter[int],
    ) -> tuple[int, int]:
        # Every selection can only lower the remaining agents' scores.  A stale
        # heap entry is therefore optimistically small and will be refreshed
        # before it can win, so no full-domain reprioritization is required.
        while True:
            entry = heapq.heappop(heap)
            chosen = int(entry[-1])
            if chosen not in available:
                continue
            current = priority(chosen, totals, selected_events)
            if entry != current:
                heapq.heappush(heap, current)
                continue
            break
        return chosen, int(totals.get(chosen, 0)) - int(selected_events[chosen])

    def add_selected(agent: int) -> None:
        selected.add(agent)
        component = analysis.component_id.get(agent)
        if component is not None:
            selected_components.add(component)
        for other, count in relevant_partners.get(agent, {}).items():
            relevant_selected_events[other] += count
        if all_selected_events is not relevant_selected_events:
            for other, count in all_partners.get(agent, {}).items():
                all_selected_events[other] += count

    relevant_agents = {
        agent for event in events for agent in (int(event.left), int(event.right))
    }
    relevant_available = relevant_agents - selected
    relevant_heap = candidate_heap(
        relevant_available, relevant_totals, relevant_selected_events
    )
    while len(selected) < min(core_budget, size) and relevant_available:
        candidate, newly_incident = choose(
            relevant_heap,
            relevant_available,
            relevant_totals,
            relevant_selected_events,
        )
        if newly_incident == 0:
            break
        relevant_available.remove(candidate)
        add_selected(candidate)

    active_agents = {
        agent for event in analysis.events for agent in (int(event.left), int(event.right))
    }
    limit = min(size, len(agent_rows))
    active_available = active_agents - selected
    active_heap = candidate_heap(active_available, all_totals, all_selected_events)
    while len(selected) < limit and active_available:
        candidate, _ = choose(
            active_heap, active_available, all_totals, all_selected_events
        )
        active_available.remove(candidate)
        add_selected(candidate)
    inactive_available = set(agent_rows) - selected
    inactive_heap = candidate_heap(inactive_available, all_totals, all_selected_events)
    while len(selected) < limit:
        candidate, _ = choose(
            inactive_heap, inactive_available, all_totals, all_selected_events
        )
        inactive_available.remove(candidate)
        add_selected(candidate)
    return sorted(selected)


def generate_topology_boundary_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    neighborhood_size: int,
    core_budget: int,
    context: _NeighborhoodContext | None = None,
    relevant_events_by_kind: dict[str, list[ConflictEvent]] | None = None,
) -> list[dict[str, Any]]:
    """Generate at most one boundary-oriented size-16 candidate per topology kind."""

    by_agents: dict[tuple[int, ...], dict[str, Any]] = {}
    for kind in ("articulation", "low_degree"):
        events = (
            relevant_events_by_kind[kind]
            if relevant_events_by_kind is not None
            else _relevant_events(analysis, kind)
        )
        if not events:
            continue
        agents = tuple(
            _boundary_neighborhood(
                state,
                analysis,
                events,
                size=neighborhood_size,
                core_budget=core_budget,
                context=context,
            )
        )
        family = f"topology-boundary-{kind}:{neighborhood_size}"
        row = by_agents.setdefault(
            agents,
            {
                "candidate_id": candidate_id(agents),
                "agents": list(agents),
                "actual_size": len(agents),
                "selection_families": [],
                "selection_rank_by_family": {},
                "proposal_count_by_family": {},
                "proposal_seeds": [],
                "seed_agents": [],
                "proposal_audit": topology_candidate_audit(analysis, agents),
            },
        )
        row["selection_families"].append(family)
        row["selection_rank_by_family"][family] = 0
        row["proposal_count_by_family"][family] = 1
    for row in by_agents.values():
        row["selection_families"].sort()
        row["selection_rank_by_family"] = dict(sorted(row["selection_rank_by_family"].items()))
        row["proposal_count_by_family"] = dict(sorted(row["proposal_count_by_family"].items()))
    return sorted(by_agents.values(), key=lambda row: str(row["candidate_id"]))


_STRUCTPOOL_FAMILY_ORDER = (
    "bottleneck_crossing",
    "conflict_component",
    "topology_boundary",
    "spatiotemporal_hotspot",
    "path_overlap",
)

_STRUCTPOOL_PREFERRED_SIZE = {
    "bottleneck_crossing": 8,
    "conflict_component": 24,
    "topology_boundary": 16,
    "spatiotemporal_hotspot": 16,
    "path_overlap": 32,
}

_STRUCTPOOL_DRAFT_VARIANT_ORDER = (
    "bottleneck_crossing",
    "conflict_component",
    "spatiotemporal_hotspot",
    "path_overlap",
    "topology_boundary_articulation",
    "topology_boundary_low_degree",
)

def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    left_set = set(map(int, left))
    right_set = set(map(int, right))
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 1.0


def _event_weights(events: Iterable[ConflictEvent]) -> collections.Counter[int]:
    weights: collections.Counter[int] = collections.Counter()
    for event in events:
        weights[int(event.left)] += 1
        weights[int(event.right)] += 1
    return weights


def _ranked_seed_neighborhood(
    state: dict[str, Any], analysis: StateAnalysis, seeds: Iterable[int], *, size: int,
    priority: dict[int, float] | collections.Counter[int] | None = None,
    context: _NeighborhoodContext | None = None,
) -> list[int]:
    """Build a deterministic candidate from a ranked structural seed set."""

    context = context or _neighborhood_context(state)
    agent_rows = context.agent_rows
    if size <= 0 or not agent_rows:
        raise ValueError("structural neighborhood requires agents and a positive size")
    priority = priority or {}
    unique_seeds = {int(agent) for agent in seeds if int(agent) in agent_rows}
    ranked = sorted(
        unique_seeds,
        key=lambda agent: (
            -float(priority.get(agent, 0.0)),
            -int(agent_rows[agent].get("conflict_degree", 0)),
            agent,
        ),
    )
    selected = set(ranked[: min(size, len(ranked))])
    return _fill_neighborhood(
        selected,
        state,
        collections.Counter(
            {agent: int(round(1000.0 * float(priority.get(agent, 0.0)))) for agent in agent_rows}
        ),
        size,
        context=context,
    )


def _conflict_component_neighborhood(
    state: dict[str, Any], analysis: StateAnalysis, *, size: int,
    context: _NeighborhoodContext | None = None,
    seed_data: tuple[set[int], collections.Counter[int]] | None = None,
) -> list[int] | None:
    if seed_data is None:
        seed_data = _conflict_component_seed_data(analysis)
    if seed_data is None:
        return None
    members, event_weight = seed_data

    return _ranked_seed_neighborhood(
        state,
        analysis,
        members,
        size=size,
        priority=event_weight,
        context=context,
    )


def _conflict_component_seed_data(
    analysis: StateAnalysis,
    *,
    event_weight: collections.Counter[int] | None = None,
    component_internal_events: dict[int, int] | None = None,
) -> tuple[set[int], collections.Counter[int]] | None:
    if not analysis.component_members:
        return None
    event_weight = event_weight or _event_weights(analysis.events)

    def component_score(item: tuple[int, set[int]]) -> tuple[int, int, int]:
        component, members = item
        internal_events = (
            int(component_internal_events.get(int(component), 0))
            if component_internal_events is not None
            else sum(
                event.left in members and event.right in members
                for event in analysis.events
            )
        )
        return internal_events, len(members), -int(component)

    _component, members = max(
        analysis.component_members.items(), key=component_score
    )
    return members, event_weight


def _hotspot_neighborhood(
    state: dict[str, Any], analysis: StateAnalysis, *, size: int,
    context: _NeighborhoodContext | None = None,
    seed_data: tuple[set[int], collections.Counter[int]] | None = None,
) -> list[int] | None:
    if seed_data is None:
        seed_data = _hotspot_seed_data(analysis)
    if seed_data is None:
        return None
    seeds, priority = seed_data
    return _ranked_seed_neighborhood(
        state,
        analysis,
        seeds,
        size=size,
        priority=priority,
        context=context,
    )


def _hotspot_seed_data(
    analysis: StateAnalysis,
) -> tuple[set[int], collections.Counter[int]] | None:
    if not analysis.events:
        return None
    buckets: dict[tuple[int, int], list[ConflictEvent]] = collections.defaultdict(list)
    for event in analysis.events:
        for cell in event.cells:
            buckets[(int(event.time) // 4, int(cell))].append(event)
    if not buckets:
        return None
    key, events = min(
        buckets.items(),
        key=lambda item: (-len(item[1]), item[0][0], item[0][1]),
    )
    del key
    priority = _event_weights(events)
    seeds = {agent for event in events for agent in (event.left, event.right)}
    return seeds, priority


def _path_overlap_neighborhood(
    state: dict[str, Any], analysis: StateAnalysis, *, size: int,
    context: _NeighborhoodContext | None = None,
    seed_data: tuple[list[int], dict[int, float]] | None = None,
) -> list[int] | None:
    if seed_data is None:
        seed_data = _path_overlap_seed_data(state, analysis)
    if seed_data is None:
        return None
    seeds, priority = seed_data
    return _ranked_seed_neighborhood(
        state,
        analysis,
        seeds,
        size=size,
        priority=priority,
        context=context,
    )


def _path_overlap_seed_data(
    state: dict[str, Any], analysis: StateAnalysis, *,
    path_sets: dict[int, frozenset[int]] | None = None,
) -> tuple[list[int], dict[int, float]] | None:
    overlap_priority = {
        int(cell): float(agent_visits - 1)
        + 0.01 * float(analysis.visit_heat[int(cell)])
        for cell, agent_visits in analysis.agent_heat.items()
        if int(agent_visits) >= 2
    }
    if not overlap_priority:
        return None
    overlapping_cells = set(overlap_priority)
    priority: dict[int, float] = {}
    for agent in state["agents"]:
        agent_id = int(agent["id"])
        path_cells = (
            path_sets[agent_id]
            if path_sets is not None
            else set(map(int, agent.get("path", [])))
        ) & overlapping_cells
        priority[agent_id] = sum(overlap_priority[cell] for cell in path_cells)
    seeds = [agent for agent, score in priority.items() if score > 0.0]
    if not seeds:
        return None
    # Explicit native repair actions must touch the current conflict graph.
    # Path overlap is broader than collision incidence, so reserve one active
    # anchor before ranking the remaining overlap-heavy agents.
    event_weight = _event_weights(analysis.events)
    active_agents = set(event_weight)
    if not active_agents:
        return None
    anchor = min(
        active_agents,
        key=lambda agent: (
            -float(priority.get(agent, 0.0)),
            -int(event_weight[agent]),
            agent,
        ),
    )
    priority[anchor] = max(priority.values(), default=0.0) + 1.0
    seeds.append(anchor)
    return seeds, priority


def _structural_candidate_context(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    include_path_overlap: bool = True,
    requested_variants: Iterable[str] | None = None,
) -> StructuralCandidateContext:
    variants = (
        frozenset(_STRUCTPOOL_DRAFT_VARIANT_ORDER)
        if requested_variants is None
        else frozenset(map(str, requested_variants))
    )
    needs_bottleneck = "bottleneck_crossing" in variants
    needs_component = "conflict_component" in variants
    needs_hotspot = "spatiotemporal_hotspot" in variants
    needs_path_overlap = include_path_overlap and "path_overlap" in variants
    materialize_path_sets = requested_variants is None or needs_path_overlap
    needs_boundary = needs_bottleneck or any(
        variant.startswith("topology_boundary_") for variant in variants
    )
    neighborhood = _neighborhood_context(state)
    if materialize_path_sets:
        path_sets = {
            int(agent["id"]): frozenset(map(int, agent.get("path", [])))
            for agent in state["agents"]
        }
        if any(not path for path in path_sets.values()):
            raise ValueError("StructPool requires non-empty agent paths")
    else:
        if any(not agent.get("path", []) for agent in state["agents"]):
            raise ValueError("StructPool requires non-empty agent paths")
        path_sets = {}
    event_weights = _event_weights(analysis.events)
    partner_counts: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    component_internal_events: collections.Counter[int] = collections.Counter()
    for event in analysis.events:
        left = int(event.left)
        right = int(event.right)
        if requested_variants is None:
            partner_counts[left][right] += 1
            partner_counts[right][left] += 1
        if needs_component:
            component = analysis.component_id.get(left)
            if component is not None and analysis.component_id.get(right) == component:
                component_internal_events[int(component)] += 1
    relevant_events_by_kind = {
        kind: _relevant_events(analysis, kind) if needs_boundary else []
        for kind in ("articulation", "low_degree")
    }
    bottleneck_events = (
        sorted(
            {
                event
                for kind in ("articulation", "low_degree")
                for event in relevant_events_by_kind[kind]
            },
            key=lambda event: (
                event.time,
                event.kind,
                event.left,
                event.right,
                event.cells,
            ),
        )
        if needs_bottleneck
        else []
    )
    return StructuralCandidateContext(
        state=state,
        analysis=analysis,
        neighborhood=neighborhood,
        path_sets=path_sets,
        event_weights=event_weights,
        partner_counts=dict(partner_counts),
        component_internal_events=dict(component_internal_events),
        relevant_events_by_kind=relevant_events_by_kind,
        bottleneck_events=bottleneck_events,
        component_seed_data=(
            _conflict_component_seed_data(
                analysis,
                event_weight=event_weights,
                component_internal_events=dict(component_internal_events),
            )
            if needs_component
            else None
        ),
        hotspot_seed_data=(
            _hotspot_seed_data(analysis) if needs_hotspot else None
        ),
        overlap_seed_data=(
            _path_overlap_seed_data(state, analysis, path_sets=path_sets)
            if needs_path_overlap
            else None
        ),
        audit_index=topology_candidate_audit_index(analysis),
        audit_cache={},
        finalized_cache={},
    )


def _draft(
    agents: Iterable[int], *, family_group: str, family: str, nominal_size: int
) -> StructuralCandidateDraft:
    ordered = tuple(sorted(set(map(int, agents))))
    if not ordered:
        raise ValueError("StructPool candidate draft must be non-empty")
    return StructuralCandidateDraft(
        agents=ordered,
        family_group=family_group,
        family=family,
        nominal_size=int(nominal_size),
    )


def _generate_structpool_variant_draft(
    context: StructuralCandidateContext,
    *,
    variant: str,
    size: int,
) -> StructuralCandidateDraft | None:
    state = context.state
    analysis = context.analysis
    neighborhood = context.neighborhood
    if variant == "bottleneck_crossing":
        if not context.bottleneck_events:
            return None
        return _draft(
            _anchor_neighborhood(
                state,
                context.bottleneck_events,
                size,
                context=neighborhood,
            ),
            family_group="bottleneck_crossing",
            family=f"structpool-bottleneck-crossing:{size}",
            nominal_size=size,
        )
    if variant == "conflict_component":
        if context.component_seed_data is None:
            return None
        agents = _conflict_component_neighborhood(
            state,
            analysis,
            size=size,
            context=neighborhood,
            seed_data=context.component_seed_data,
        )
        return (
            _draft(
                agents,
                family_group="conflict_component",
                family=f"structpool-conflict-component:{size}",
                nominal_size=size,
            )
            if agents
            else None
        )
    if variant == "spatiotemporal_hotspot":
        if context.hotspot_seed_data is None:
            return None
        agents = _hotspot_neighborhood(
            state,
            analysis,
            size=size,
            context=neighborhood,
            seed_data=context.hotspot_seed_data,
        )
        return (
            _draft(
                agents,
                family_group="spatiotemporal_hotspot",
                family=f"structpool-spatiotemporal-hotspot:{size}",
                nominal_size=size,
            )
            if agents
            else None
        )
    if variant == "path_overlap":
        if context.overlap_seed_data is None:
            return None
        agents = _path_overlap_neighborhood(
            state,
            analysis,
            size=size,
            context=neighborhood,
            seed_data=context.overlap_seed_data,
        )
        return (
            _draft(
                agents,
                family_group="path_overlap",
                family=f"structpool-path-overlap:{size}",
                nominal_size=size,
            )
            if agents
            else None
        )
    if variant.startswith("topology_boundary_"):
        kind = variant.removeprefix("topology_boundary_")
        if kind not in {"articulation", "low_degree"}:
            raise ValueError(f"unsupported StructPool boundary variant: {variant}")
        events = context.relevant_events_by_kind[kind]
        if not events:
            return None
        return _draft(
            _boundary_neighborhood(
                state,
                analysis,
                events,
                size=size,
                core_budget=min(4, size),
                context=neighborhood,
            ),
            family_group="topology_boundary",
            family=f"structpool-boundary-{kind}:{size}",
            nominal_size=size,
        )
    raise ValueError(f"unsupported StructPool family variant: {variant}")


def generate_structpool_candidate_drafts(
    context: StructuralCandidateContext,
    *,
    neighborhood_sizes: Iterable[int] = (8, 16, 24, 32),
) -> list[StructuralCandidateDraft]:
    """Generate agent sets and provenance without computing audits or scores."""

    sizes = sorted(set(map(int, neighborhood_sizes)))
    if sizes != [8, 16, 24, 32]:
        raise ValueError("StructPool v1 requires sizes 8, 16, 24, and 32")
    drafts: list[StructuralCandidateDraft] = []
    for size in sizes:
        for variant in _STRUCTPOOL_DRAFT_VARIANT_ORDER:
            draft = _generate_structpool_variant_draft(
                context, variant=variant, size=size
            )
            if draft is not None:
                drafts.append(draft)
    return drafts


def _structpool_score(
    audit: dict[str, float], *, family_group: str,
) -> float:
    family_signal = {
        "bottleneck_crossing": audit["global_pair_internal_coverage"],
        "conflict_component": audit["conflict_component_reach"],
        "topology_boundary": audit["global_pair_boundary_ratio"],
        "spatiotemporal_hotspot": audit["global_event_incident_coverage"],
        "path_overlap": audit["global_pair_incident_coverage"],
    }[family_group]
    return (
        2.0 * audit["global_event_incident_coverage"]
        + 0.5 * audit["global_event_boundary_ratio"]
        + 0.25 * audit["conflict_component_reach"]
        + 0.25 * family_signal
    )


def _merge_structpool_drafts(
    drafts: Iterable[StructuralCandidateDraft],
) -> list[_MergedStructuralCandidate]:
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    for draft in drafts:
        current = merged.setdefault(
            draft.agents,
            {
                "selection_families": set(),
                "family_groups": set(),
                "incumbent_boundary": False,
            },
        )
        current["selection_families"].add(draft.family)
        current["family_groups"].add(draft.family_group)
        current["incumbent_boundary"] = bool(
            current["incumbent_boundary"]
            or (
                draft.family_group == "topology_boundary"
                and draft.nominal_size == 16
            )
        )
    return sorted(
        (
            _MergedStructuralCandidate(
                agents=agents,
                selection_families=tuple(sorted(value["selection_families"])),
                family_groups=tuple(sorted(value["family_groups"])),
                incumbent_boundary=bool(value["incumbent_boundary"]),
            )
            for agents, value in merged.items()
        ),
        key=lambda candidate: candidate_id(candidate.agents),
    )


def _finalize_structpool_candidate(
    context: StructuralCandidateContext,
    candidate: _MergedStructuralCandidate,
) -> dict[str, Any]:
    cached = context.finalized_cache.get(candidate.agents)
    if cached is not None:
        return cached
    audit = context.audit_cache.get(candidate.agents)
    if audit is None:
        audit = topology_candidate_audit(
            context.analysis,
            candidate.agents,
            audit_index=context.audit_index,
        )
        context.audit_cache[candidate.agents] = audit
    families = list(candidate.selection_families)
    family_groups = list(candidate.family_groups)
    row = {
        "candidate_id": candidate_id(candidate.agents),
        "agents": list(candidate.agents),
        "actual_size": len(candidate.agents),
        "selection_families": families,
        "selection_rank_by_family": {family: 0 for family in families},
        "proposal_count_by_family": {family: 1 for family in families},
        "proposal_seeds": [],
        "seed_agents": [],
        "proposal_audit": audit,
        "structpool_family_groups": family_groups,
        "structpool_score": max(
            _structpool_score(audit, family_group=family_group)
            for family_group in family_groups
        ),
    }
    context.finalized_cache[candidate.agents] = row
    return row


def finalize_structpool_candidates(
    context: StructuralCandidateContext,
    candidates: Iterable[_MergedStructuralCandidate],
) -> list[dict[str, Any]]:
    """Materialize complete rows only for candidates retained by reduction."""

    return [
        copy.deepcopy(_finalize_structpool_candidate(context, candidate))
        for candidate in candidates
    ]


def _structpool_support_by_family(
    context: StructuralCandidateContext,
    sizes: Iterable[int],
) -> dict[str, set[int]]:
    support: dict[str, set[int]] = {}
    bottleneck = {
        int(agent)
        for event in context.bottleneck_events
        for agent in (event.left, event.right)
    }
    component = (
        set(map(int, context.component_seed_data[0]))
        if context.component_seed_data is not None
        else set()
    )
    hotspot = (
        set(map(int, context.hotspot_seed_data[0]))
        if context.hotspot_seed_data is not None
        else set()
    )
    overlap = (
        set(map(int, context.overlap_seed_data[0])) & set(context.event_weights)
        if context.overlap_seed_data is not None
        else set()
    )
    boundary = {
        kind: {
            int(agent)
            for event in context.relevant_events_by_kind[kind]
            for agent in (event.left, event.right)
        }
        for kind in ("articulation", "low_degree")
    }
    for size in sizes:
        support[f"structpool-bottleneck-crossing:{size}"] = set(bottleneck)
        support[f"structpool-conflict-component:{size}"] = set(component)
        support[f"structpool-spatiotemporal-hotspot:{size}"] = set(hotspot)
        support[f"structpool-path-overlap:{size}"] = set(overlap)
        for kind in ("articulation", "low_degree"):
            support[f"structpool-boundary-{kind}:{size}"] = set(boundary[kind])
    return support


def generate_structpool_candidate_grid(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    neighborhood_sizes: Iterable[int] = (8, 16, 24, 32),
) -> list[dict[str, Any]]:
    """Materialize the uncapped four-size StructPool ablation grid.

    Exact agent-set duplicates are repaired only once while every generating
    family/size remains in provenance.  This API is outcome blind and is not a
    runtime candidate selector.
    """

    sizes = sorted(set(map(int, neighborhood_sizes)))
    if sizes != [8, 16, 24, 32]:
        raise ValueError("StructPool size grid requires sizes 8, 16, 24, and 32")
    if not analysis.events:
        return []
    context = _structural_candidate_context(state, analysis)
    drafts = generate_structpool_candidate_drafts(
        context, neighborhood_sizes=sizes
    )
    merged = _merge_structpool_drafts(drafts)
    rows = finalize_structpool_candidates(context, merged)
    support = _structpool_support_by_family(context, sizes)
    agent_count = len(state["agents"])
    for row in rows:
        families = list(map(str, row["selection_families"]))
        row["structpool_support_count_by_family"] = {
            family: len(support[family]) for family in families
        }
        row["structpool_support_ratio_by_family"] = {
            family: len(support[family]) / max(1, agent_count)
            for family in families
        }
        row["structpool_nominal_size_by_family"] = {
            family: int(family.rsplit(":", 1)[1]) for family in families
        }
        row["structpool_grid_pure_family"] = (
            len(row["structpool_family_groups"]) == 1
        )
        row["structpool_grid_duplicate_provenance_count"] = len(families)
    return rows


def generate_structpool_candidate_subset(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    family_sizes: Mapping[str, Iterable[int]],
) -> list[dict[str, Any]]:
    """Materialize a registered subset of the complete StructPool grid.

    This is a runtime engineering API.  It never changes the complete
    four-size audit contract exposed by :func:`generate_structpool_candidate_grid`.
    Every retained draft is bit-for-bit the same agent set as its full-grid
    counterpart; omitted family/size cells are not constructed or finalized.
    """

    unknown = set(map(str, family_sizes)) - set(_STRUCTPOOL_DRAFT_VARIANT_ORDER)
    if unknown:
        raise ValueError(f"unknown StructPool runtime variants: {sorted(unknown)}")
    normalized = {
        str(variant): tuple(sorted(set(map(int, sizes))))
        for variant, sizes in family_sizes.items()
    }
    if not normalized or any(
        not sizes or any(size not in {8, 16, 24, 32} for size in sizes)
        for sizes in normalized.values()
    ):
        raise ValueError("StructPool runtime family sizes must use 8/16/24/32")
    if not analysis.events:
        return []

    context = _structural_candidate_context(
        state,
        analysis,
        include_path_overlap="path_overlap" in normalized,
        requested_variants=(
            normalized
            if normalized
            == {
                "conflict_component": (16,),
                "spatiotemporal_hotspot": (16,),
            }
            else None
        ),
    )
    drafts: list[StructuralCandidateDraft] = []
    for variant in _STRUCTPOOL_DRAFT_VARIANT_ORDER:
        for size in normalized.get(variant, ()):
            draft = _generate_structpool_variant_draft(
                context,
                variant=variant,
                size=size,
            )
            if draft is not None:
                drafts.append(draft)
    merged = _merge_structpool_drafts(drafts)
    rows = finalize_structpool_candidates(context, merged)
    support = _structpool_support_by_family(
        context,
        sorted({size for sizes in normalized.values() for size in sizes}),
    )
    agent_count = len(state["agents"])
    for row in rows:
        families = list(map(str, row["selection_families"]))
        row["structpool_support_count_by_family"] = {
            family: len(support[family]) for family in families
        }
        row["structpool_support_ratio_by_family"] = {
            family: len(support[family]) / max(1, agent_count)
            for family in families
        }
        row["structpool_nominal_size_by_family"] = {
            family: int(family.rsplit(":", 1)[1]) for family in families
        }
        row["structpool_grid_pure_family"] = (
            len(row["structpool_family_groups"]) == 1
        )
        row["structpool_grid_duplicate_provenance_count"] = len(families)
        row["structpool_runtime_subset"] = True
    return rows




def reduce_structpool_candidates(
    context: StructuralCandidateContext,
    drafts: Iterable[StructuralCandidateDraft],
    *,
    maximum_added_candidates: int = 6,
    maximum_jaccard_similarity: float = 0.8,
) -> list[_MergedStructuralCandidate]:
    """Apply exact StructPool v1 merge, ordering, cap and diversity rules."""

    if maximum_added_candidates != 6:
        raise ValueError("StructPool v1 requires a six-candidate addition cap")
    if not 0.0 <= maximum_jaccard_similarity < 1.0:
        raise ValueError("StructPool Jaccard threshold must be in [0, 1)")
    merged = _merge_structpool_drafts(drafts)
    incumbent_keys = {
        candidate.agents for candidate in merged if candidate.incumbent_boundary
    }
    by_key = {candidate.agents: candidate for candidate in merged}
    selected = [by_key[key] for key in sorted(incumbent_keys)]
    selected_keys = {candidate.agents for candidate in selected}

    def novel_is_diverse(candidate: _MergedStructuralCandidate) -> bool:
        if candidate.agents in selected_keys:
            return False
        novel_selected = [
            previous
            for previous in selected
            if previous.agents not in incumbent_keys
        ]
        return all(
            _jaccard(candidate.agents, previous.agents)
            <= maximum_jaccard_similarity
            for previous in novel_selected
        )

    groups_with_incumbent = {"topology_boundary"} if incumbent_keys else set()
    for group in _STRUCTPOOL_FAMILY_ORDER:
        if len(selected) >= maximum_added_candidates:
            break
        if group in groups_with_incumbent:
            continue
        choices = [
            candidate
            for candidate in merged
            if group in candidate.family_groups and novel_is_diverse(candidate)
        ]
        if not choices:
            continue
        minimum_distance = min(
            abs(len(candidate.agents) - _STRUCTPOOL_PREFERRED_SIZE[group])
            for candidate in choices
        )
        nearest = [
            candidate
            for candidate in choices
            if abs(len(candidate.agents) - _STRUCTPOOL_PREFERRED_SIZE[group])
            == minimum_distance
        ]
        chosen = min(
            nearest,
            key=lambda candidate: (
                -float(
                    _finalize_structpool_candidate(context, candidate)[
                        "structpool_score"
                    ]
                ),
                len(candidate.agents),
                candidate_id(candidate.agents),
            ),
        )
        selected.append(chosen)
        selected_keys.add(chosen.agents)

    if len(selected) < maximum_added_candidates:
        remaining = sorted(
            merged,
            key=lambda candidate: (
                -float(
                    _finalize_structpool_candidate(context, candidate)[
                        "structpool_score"
                    ]
                ),
                len(candidate.agents),
                candidate_id(candidate.agents),
            ),
        )
        for candidate in remaining:
            if len(selected) >= maximum_added_candidates:
                break
            if novel_is_diverse(candidate):
                selected.append(candidate)
                selected_keys.add(candidate.agents)
    return selected


def generate_structpool_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    neighborhood_sizes: Iterable[int] = (8, 16, 24, 32),
    maximum_added_candidates: int = 6,
    maximum_jaccard_similarity: float = 0.8,
) -> list[dict[str, Any]]:
    """Generate the outcome-free, capped STRIDE StructPool v1 additions.

    The two incumbent size-16 boundary actions are retained first.  Novel
    families then receive one deterministic round-robin opportunity before a
    structural-score fill.  Diversity is applied only among novel additions so
    it cannot remove an incumbent action.
    """

    sizes = sorted(set(map(int, neighborhood_sizes)))
    if sizes != [8, 16, 24, 32]:
        raise ValueError("StructPool v1 requires sizes 8, 16, 24, and 32")
    if maximum_added_candidates != 6:
        raise ValueError("StructPool v1 requires a six-candidate addition cap")
    if not 0.0 <= maximum_jaccard_similarity < 1.0:
        raise ValueError("StructPool Jaccard threshold must be in [0, 1)")
    if not analysis.events:
        return []

    context = _structural_candidate_context(state, analysis)
    drafts = generate_structpool_candidate_drafts(
        context, neighborhood_sizes=sizes
    )
    reduced = reduce_structpool_candidates(
        context,
        drafts,
        maximum_added_candidates=maximum_added_candidates,
        maximum_jaccard_similarity=maximum_jaccard_similarity,
    )
    return finalize_structpool_candidates(context, reduced)


def merge_structpool_candidates(
    base_candidates: list[dict[str, Any]],
    structpool_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append unique StructPool actions without changing any frozen base row."""

    merged = copy.deepcopy(base_candidates)
    known = {tuple(map(int, row["agents"])) for row in merged}
    if len(known) != len(merged):
        raise ValueError("base candidate pool contains duplicate agent sets")
    for candidate in structpool_candidates:
        key = tuple(map(int, candidate["agents"]))
        if key in known:
            continue
        merged.append(copy.deepcopy(candidate))
        known.add(key)
    return merged


def merge_topology_anchor_candidates(
    base_candidates: list[dict[str, Any]], anchor_candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged = {
        tuple(map(int, candidate["agents"])): {
            **candidate,
            "agents": list(map(int, candidate["agents"])),
            "selection_families": list(map(str, candidate["selection_families"])),
            "selection_rank_by_family": dict(candidate["selection_rank_by_family"]),
            "proposal_count_by_family": dict(candidate["proposal_count_by_family"]),
            "proposal_seeds": list(map(int, candidate.get("proposal_seeds", []))),
            "seed_agents": list(map(int, candidate.get("seed_agents", []))),
        }
        for candidate in base_candidates
    }
    for candidate in anchor_candidates:
        key = tuple(map(int, candidate["agents"]))
        existing = merged.get(key)
        if existing is None:
            merged[key] = candidate
            continue
        for family in candidate["selection_families"]:
            if family not in existing["selection_families"]:
                existing["selection_families"].append(family)
            existing["selection_rank_by_family"][family] = 0
            existing["proposal_count_by_family"][family] = 1
        existing["selection_families"].sort()
        existing["selection_rank_by_family"] = dict(
            sorted(existing["selection_rank_by_family"].items())
        )
        existing["proposal_count_by_family"] = dict(
            sorted(existing["proposal_count_by_family"].items())
        )
    return sorted(merged.values(), key=lambda row: str(row["candidate_id"]))


__all__ = [
    "StructuralCandidateContext",
    "StructuralCandidateDraft",
    "TopologyCandidateAuditIndex",
    "finalize_structpool_candidates",
    "generate_structpool_candidate_drafts",
    "generate_structpool_candidate_grid",
    "generate_structpool_candidate_subset",
    "generate_topology_anchor_candidates",
    "generate_topology_boundary_candidates",
    "generate_structpool_candidates",
    "merge_structpool_candidates",
    "merge_topology_anchor_candidates",
    "reduce_structpool_candidates",
    "topology_candidate_audit",
    "topology_candidate_audit_index",
]
