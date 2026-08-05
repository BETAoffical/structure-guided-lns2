from __future__ import annotations

import collections
import copy
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
) -> list[int]:
    """Build a deterministic candidate from a ranked structural seed set."""

    agent_rows = {int(agent["id"]): agent for agent in state["agents"]}
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
    )


def _conflict_component_neighborhood(
    state: dict[str, Any], analysis: StateAnalysis, *, size: int,
) -> list[int] | None:
    if not analysis.component_members:
        return None
    event_weight = _event_weights(analysis.events)

    def component_score(item: tuple[int, set[int]]) -> tuple[int, int, int]:
        component, members = item
        internal_events = sum(
            event.left in members and event.right in members for event in analysis.events
        )
        return internal_events, len(members), -int(component)

    _component, members = max(
        analysis.component_members.items(), key=component_score
    )
    return _ranked_seed_neighborhood(
        state, analysis, members, size=size, priority=event_weight
    )


def _hotspot_neighborhood(
    state: dict[str, Any], analysis: StateAnalysis, *, size: int,
) -> list[int] | None:
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
    return _ranked_seed_neighborhood(
        state, analysis, seeds, size=size, priority=priority
    )


def _path_overlap_neighborhood(
    state: dict[str, Any], analysis: StateAnalysis, *, size: int,
) -> list[int] | None:
    overlapping_cells = {
        int(cell)
        for cell, agent_visits in analysis.agent_heat.items()
        if int(agent_visits) >= 2
    }
    if not overlapping_cells:
        return None
    priority: dict[int, float] = {}
    for agent in state["agents"]:
        agent_id = int(agent["id"])
        path_cells = set(map(int, agent.get("path", []))) & overlapping_cells
        priority[agent_id] = sum(
            float(analysis.agent_heat[cell] - 1)
            + 0.01 * float(analysis.visit_heat[cell])
            for cell in path_cells
        )
    seeds = [agent for agent, score in priority.items() if score > 0.0]
    if not seeds:
        return None
    return _ranked_seed_neighborhood(
        state, analysis, seeds, size=size, priority=priority
    )


def _structpool_score(
    analysis: StateAnalysis, agents: Iterable[int], *, family_group: str,
) -> float:
    audit = topology_candidate_audit(analysis, agents)
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


def _structpool_row(
    analysis: StateAnalysis, agents: Iterable[int], *, family_group: str,
    family: str,
) -> dict[str, Any]:
    ordered = tuple(sorted(set(map(int, agents))))
    if not ordered:
        raise ValueError("StructPool candidate must be non-empty")
    return {
        "candidate_id": candidate_id(ordered),
        "agents": list(ordered),
        "actual_size": len(ordered),
        "selection_families": [family],
        "selection_rank_by_family": {family: 0},
        "proposal_count_by_family": {family: 1},
        "proposal_seeds": [],
        "seed_agents": [],
        "proposal_audit": topology_candidate_audit(analysis, ordered),
        "structpool_family_groups": [family_group],
        "structpool_score": _structpool_score(
            analysis, ordered, family_group=family_group
        ),
    }


def _merge_structpool_raw_rows(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    for source in rows:
        key = tuple(map(int, source["agents"]))
        existing = merged.get(key)
        if existing is None:
            merged[key] = copy.deepcopy(source)
            continue
        for family in source["selection_families"]:
            if family not in existing["selection_families"]:
                existing["selection_families"].append(family)
            existing["selection_rank_by_family"][family] = 0
            existing["proposal_count_by_family"][family] = 1
        for group in source["structpool_family_groups"]:
            if group not in existing["structpool_family_groups"]:
                existing["structpool_family_groups"].append(group)
        existing["structpool_score"] = max(
            float(existing["structpool_score"]), float(source["structpool_score"])
        )
    for row in merged.values():
        row["selection_families"].sort()
        row["selection_rank_by_family"] = dict(
            sorted(row["selection_rank_by_family"].items())
        )
        row["proposal_count_by_family"] = dict(
            sorted(row["proposal_count_by_family"].items())
        )
        row["structpool_family_groups"].sort()
    return sorted(merged.values(), key=lambda row: str(row["candidate_id"]))


def _candidate_order(row: dict[str, Any], family_group: str) -> tuple[Any, ...]:
    return (
        abs(int(row["actual_size"]) - _STRUCTPOOL_PREFERRED_SIZE[family_group]),
        -float(row["structpool_score"]),
        int(row["actual_size"]),
        str(row["candidate_id"]),
    )


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

    raw: list[dict[str, Any]] = []
    bottleneck_events = sorted(
        {
            event
            for kind in ("articulation", "low_degree")
            for event in _relevant_events(analysis, kind)
        },
        key=lambda event: (event.time, event.kind, event.left, event.right, event.cells),
    )
    for size in sizes:
        if bottleneck_events:
            agents = _anchor_neighborhood(state, bottleneck_events, size)
            raw.append(
                _structpool_row(
                    analysis,
                    agents,
                    family_group="bottleneck_crossing",
                    family=f"structpool-bottleneck-crossing:{size}",
                )
            )

        component = _conflict_component_neighborhood(
            state, analysis, size=size
        )
        if component:
            raw.append(
                _structpool_row(
                    analysis,
                    component,
                    family_group="conflict_component",
                    family=f"structpool-conflict-component:{size}",
                )
            )

        hotspot = _hotspot_neighborhood(state, analysis, size=size)
        if hotspot:
            raw.append(
                _structpool_row(
                    analysis,
                    hotspot,
                    family_group="spatiotemporal_hotspot",
                    family=f"structpool-spatiotemporal-hotspot:{size}",
                )
            )

        overlap = _path_overlap_neighborhood(state, analysis, size=size)
        if overlap:
            raw.append(
                _structpool_row(
                    analysis,
                    overlap,
                    family_group="path_overlap",
                    family=f"structpool-path-overlap:{size}",
                )
            )

        for boundary in generate_topology_boundary_candidates(
            state,
            analysis,
            neighborhood_size=size,
            core_budget=min(4, size),
        ):
            for family in boundary["selection_families"]:
                raw.append(
                    _structpool_row(
                        analysis,
                        boundary["agents"],
                        family_group="topology_boundary",
                        family=family.replace("topology-boundary", "structpool-boundary"),
                    )
                )

    raw_rows = _merge_structpool_raw_rows(raw)
    incumbent = generate_topology_boundary_candidates(
        state, analysis, neighborhood_size=16, core_budget=4
    )
    incumbent_keys = {tuple(map(int, row["agents"])) for row in incumbent}
    by_key = {tuple(map(int, row["agents"])): row for row in raw_rows}
    selected = [copy.deepcopy(by_key[key]) for key in sorted(incumbent_keys) if key in by_key]
    selected_keys = {tuple(map(int, row["agents"])) for row in selected}

    def novel_is_diverse(row: dict[str, Any]) -> bool:
        key = tuple(map(int, row["agents"]))
        if key in selected_keys:
            return False
        novel_selected = [
            previous
            for previous in selected
            if tuple(map(int, previous["agents"])) not in incumbent_keys
        ]
        return all(
            _jaccard(row["agents"], previous["agents"])
            <= maximum_jaccard_similarity
            for previous in novel_selected
        )

    groups_with_incumbent = {"topology_boundary"} if incumbent_keys else set()
    for group in _STRUCTPOOL_FAMILY_ORDER:
        if len(selected) >= maximum_added_candidates:
            break
        if group in groups_with_incumbent:
            continue
        choices = sorted(
            (
                row
                for row in raw_rows
                if group in row["structpool_family_groups"] and novel_is_diverse(row)
            ),
            key=lambda row: _candidate_order(row, group),
        )
        if choices:
            chosen = copy.deepcopy(choices[0])
            selected.append(chosen)
            selected_keys.add(tuple(map(int, chosen["agents"])))

    remaining = sorted(
        raw_rows,
        key=lambda row: (
            -float(row["structpool_score"]),
            int(row["actual_size"]),
            str(row["candidate_id"]),
        ),
    )
    for row in remaining:
        if len(selected) >= maximum_added_candidates:
            break
        if novel_is_diverse(row):
            chosen = copy.deepcopy(row)
            selected.append(chosen)
            selected_keys.add(tuple(map(int, chosen["agents"])))
    return selected


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
    "generate_topology_anchor_candidates",
    "generate_topology_boundary_candidates",
    "generate_structpool_candidates",
    "merge_structpool_candidates",
    "merge_topology_anchor_candidates",
    "topology_candidate_audit",
]
