from __future__ import annotations

import collections
from typing import Any, Iterable

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import ConflictEvent, StateAnalysis


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
    selected: set[int], state: dict[str, Any], event_weight: collections.Counter[int], size: int
) -> list[int]:
    agent_rows = {int(agent["id"]): agent for agent in state["agents"]}
    if not selected <= set(agent_rows):
        raise ValueError("topology anchor selected an unknown agent")
    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    for edge in state.get("conflict_edges", []):
        left, right = map(int, edge)
        adjacency[left].add(right)
        adjacency[right].add(left)
    while len(selected) < min(size, len(agent_rows)):
        remaining = set(agent_rows) - selected
        chosen = min(
            remaining,
            key=lambda agent: (
                -len(adjacency[agent] & selected),
                -int(agent_rows[agent].get("conflict_degree", 0)),
                -int(event_weight[agent]),
                agent,
            ),
        )
        selected.add(chosen)
    return sorted(selected)


def _anchor_neighborhood(
    state: dict[str, Any], events: list[ConflictEvent], size: int
) -> list[int]:
    if size <= 0 or not events:
        raise ValueError("topology anchor requires events and a positive size")
    uncovered = set(range(len(events)))
    selected: set[int] = set()
    event_weight: collections.Counter[int] = collections.Counter()
    for event in events:
        event_weight[event.left] += 1
        event_weight[event.right] += 1
    pairs = sorted({(event.left, event.right) for event in events})
    while uncovered and len(selected) < size:
        options = []
        for left, right in pairs:
            addition = {left, right} - selected
            if not addition or len(selected) + len(addition) > size:
                continue
            covered = {
                index
                for index in uncovered
                if events[index].left in addition or events[index].right in addition
            }
            internal = {
                index
                for index in uncovered
                if events[index].left in (selected | addition)
                and events[index].right in (selected | addition)
            }
            options.append(
                (
                    len(covered) / len(addition),
                    len(covered),
                    len(internal),
                    sum(event_weight[agent] for agent in addition),
                    -left,
                    -right,
                    addition,
                )
            )
        if not options:
            break
        *_, addition = max(options, key=lambda value: value[:-1])
        selected.update(addition)
        uncovered = {
            index
            for index in uncovered
            if events[index].left not in selected and events[index].right not in selected
        }
    return _fill_neighborhood(selected, state, event_weight, size)


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


def topology_candidate_audit(
    analysis: StateAnalysis, selected_agents: Iterable[int]
) -> dict[str, float]:
    """Return outcome-free conflict coverage diagnostics for one neighborhood."""

    selected = set(map(int, selected_agents))
    if not selected:
        raise ValueError("topology candidate audit requires a non-empty neighborhood")
    known = set(analysis.component_id)
    event_count = len(analysis.events)
    pair_count = len(analysis.pair_set)

    def counts(items: Iterable[Any]) -> tuple[int, int, int]:
        internal = 0
        incident = 0
        boundary = 0
        for item in items:
            left = int(item.left if hasattr(item, "left") else item[0])
            right = int(item.right if hasattr(item, "right") else item[1])
            left_selected = left in selected
            right_selected = right in selected
            internal += int(left_selected and right_selected)
            incident += int(left_selected or right_selected)
            boundary += int(left_selected != right_selected)
        return internal, incident, boundary

    event_internal, event_incident, event_boundary = counts(analysis.events)
    pair_internal, pair_incident, pair_boundary = counts(sorted(analysis.pair_set))
    reached_components = {
        int(analysis.component_id[agent]) for agent in selected if agent in known
    }
    component_count = len(analysis.component_members)
    return {
        "global_event_incident_coverage": event_incident / event_count if event_count else 0.0,
        "global_event_internal_coverage": event_internal / event_count if event_count else 0.0,
        "global_event_boundary_ratio": event_boundary / event_incident if event_incident else 0.0,
        "global_pair_incident_coverage": pair_incident / pair_count if pair_count else 0.0,
        "global_pair_internal_coverage": pair_internal / pair_count if pair_count else 0.0,
        "global_pair_boundary_ratio": pair_boundary / pair_incident if pair_incident else 0.0,
        "conflict_component_reach": (
            len(reached_components) / component_count if component_count else 0.0
        ),
    }


def _boundary_neighborhood(
    state: dict[str, Any], analysis: StateAnalysis, events: list[ConflictEvent],
    *, size: int, core_budget: int,
) -> list[int]:
    """Select incident endpoints while discouraging closure of covered conflicts."""

    agent_rows = {int(agent["id"]): agent for agent in state["agents"]}
    if size <= 0 or core_budget <= 0 or not events:
        raise ValueError("topology boundary candidate requires events and positive budgets")
    selected: set[int] = set()
    selected_components: set[int] = set()

    def incident_index(
        scored_events: list[ConflictEvent],
    ) -> dict[int, tuple[ConflictEvent, ...]]:
        incident: dict[int, list[ConflictEvent]] = collections.defaultdict(list)
        for event in scored_events:
            incident[int(event.left)].append(event)
            incident[int(event.right)].append(event)
        return {agent: tuple(rows) for agent, rows in incident.items()}

    relevant_incident = incident_index(events)
    all_incident = (
        relevant_incident
        if events is analysis.events
        else incident_index(analysis.events)
    )

    def choose(
        available: set[int],
        incident: dict[int, tuple[ConflictEvent, ...]],
    ) -> int:
        def score(agent: int) -> tuple[int, int, int, int, int, int]:
            agent_events = incident.get(agent, ())
            newly_incident = sum(
                event.left not in selected
                and event.right not in selected
                for event in agent_events
            )
            newly_internal = sum(
                (event.left in selected) != (event.right in selected)
                for event in agent_events
            )
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

        return max(available, key=score)

    relevant_agents = {
        agent for event in events for agent in (int(event.left), int(event.right))
    }
    while len(selected) < min(core_budget, size) and relevant_agents - selected:
        candidate = choose(relevant_agents - selected, relevant_incident)
        before_coverage = sum(
            event.left in selected or event.right in selected for event in events
        )
        selected.add(candidate)
        component = analysis.component_id.get(candidate)
        if component is not None:
            selected_components.add(component)
        after_coverage = sum(
            event.left in selected or event.right in selected for event in events
        )
        if after_coverage == before_coverage:
            selected.remove(candidate)
            selected_components = {
                int(analysis.component_id[agent])
                for agent in selected
                if agent in analysis.component_id
            }
            break

    active_agents = {
        agent for event in analysis.events for agent in (int(event.left), int(event.right))
    }
    limit = min(size, len(agent_rows))
    while len(selected) < limit:
        remaining_active = active_agents - selected
        available = remaining_active if remaining_active else set(agent_rows) - selected
        candidate = choose(available, all_incident)
        selected.add(candidate)
        component = analysis.component_id.get(candidate)
        if component is not None:
            selected_components.add(component)
    return sorted(selected)


def generate_topology_boundary_candidates(
    state: dict[str, Any], analysis: StateAnalysis, *, neighborhood_size: int,
    core_budget: int,
) -> list[dict[str, Any]]:
    """Generate at most one boundary-oriented size-16 candidate per topology kind."""

    by_agents: dict[tuple[int, ...], dict[str, Any]] = {}
    for kind in ("articulation", "low_degree"):
        events = _relevant_events(analysis, kind)
        if not events:
            continue
        agents = tuple(
            _boundary_neighborhood(
                state, analysis, events, size=neighborhood_size, core_budget=core_budget
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
    "generate_topology_anchor_candidates",
    "generate_topology_boundary_candidates",
    "merge_topology_anchor_candidates",
    "topology_candidate_audit",
]
