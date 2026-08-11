from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Iterable

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.temporal_state import TemporalHistoryContext
from lns2_selector.runtime.topology_candidates import (
    _SCALEPOOL_VARIANT_ORDER,
    _generate_structpool_variant_draft,
    _jaccard,
    _structpool_family_name,
    _structpool_score,
    _structural_candidate_context,
    topology_candidate_audit,
)


PARETOPOOL_ID = "stride-paretopool-v1"
PARETOPOOL_BUDGETS = (6, 8, 12)


@dataclass(frozen=True)
class _ParetoDraft:
    agents: tuple[int, ...]
    selection_families: tuple[str, ...]
    family_groups: tuple[str, ...]
    sources: tuple[str, ...]
    support_count_by_family: tuple[tuple[str, int], ...]
    natural_breakpoints_by_family: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass
class ParetoPoolGenerationResult:
    candidates: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    raw_candidate_count: int
    pareto_front_count: int
    history_candidate_count: int


def _variant_support(context: Any, variant: str) -> set[int]:
    if variant == "bottleneck_crossing":
        return {
            int(agent)
            for event in context.bottleneck_events
            for agent in (event.left, event.right)
        }
    if variant == "conflict_component":
        return (
            set(map(int, context.component_seed_data[0]))
            if context.component_seed_data is not None
            else set()
        )
    if variant == "spatiotemporal_hotspot":
        return (
            set(map(int, context.hotspot_seed_data[0]))
            if context.hotspot_seed_data is not None
            else set()
        )
    if variant == "path_overlap":
        return (
            set(map(int, context.overlap_seed_data[0])) & set(context.event_weights)
            if context.overlap_seed_data is not None
            else set()
        )
    if variant.startswith("topology_boundary_"):
        kind = variant.removeprefix("topology_boundary_")
        return {
            int(agent)
            for event in context.relevant_events_by_kind[kind]
            for agent in (event.left, event.right)
        }
    raise ValueError(f"unsupported ParetoPool family: {variant}")


def natural_structural_breakpoints(
    context: Any,
    variant: str,
    *,
    maximum_neighborhood_size: int,
) -> tuple[int, ...]:
    """Return state-derived closure sizes without a preferred-size grid."""

    support = _variant_support(context, variant)
    if not support:
        return ()
    agent_ids = set(context.agent_rows)
    limit = min(int(maximum_neighborhood_size), len(agent_ids))
    if limit <= 0:
        raise ValueError("ParetoPool maximum neighborhood size must be positive")
    support &= agent_ids
    conflict_closure = set(support)
    for agent in support:
        conflict_closure.update(context.conflict_adjacency.get(agent, ()))
    component_closure = set(support)
    touched_components = {
        context.analysis.component_id[agent]
        for agent in support
        if agent in context.analysis.component_id
    }
    for component in touched_components:
        component_closure.update(context.analysis.component_members.get(component, ()))
    support_cells = {
        cell for agent in support for cell in context.path_sets.get(agent, ())
    }
    path_closure = set(support)
    path_closure.update(
        agent
        for agent, cells in context.path_sets.items()
        if agent in context.event_weights and cells & support_cells
    )
    sizes = {
        min(limit, len(group))
        for group in (support, conflict_closure, component_closure, path_closure)
        if group
    }
    return tuple(sorted(size for size in sizes if size >= 2))


def _current_drafts(
    context: Any, *, maximum_neighborhood_size: int
) -> list[_ParetoDraft]:
    drafts: list[_ParetoDraft] = []
    for variant in _SCALEPOOL_VARIANT_ORDER:
        support = _variant_support(context, variant)
        breakpoints = natural_structural_breakpoints(
            context,
            variant,
            maximum_neighborhood_size=maximum_neighborhood_size,
        )
        for size in breakpoints:
            draft = _generate_structpool_variant_draft(
                context, variant=variant, size=size
            )
            if draft is None:
                continue
            family = _structpool_family_name(variant, size)
            if draft.family != family:
                raise RuntimeError("ParetoPool family identity drifted")
            drafts.append(
                _ParetoDraft(
                    agents=draft.agents,
                    selection_families=(family,),
                    family_groups=(draft.family_group,),
                    sources=("current_state",),
                    support_count_by_family=((family, len(support)),),
                    natural_breakpoints_by_family=((family, breakpoints),),
                )
            )
    return drafts


def _history_drafts(
    context: Any,
    history: TemporalHistoryContext | None,
    *,
    maximum_neighborhood_size: int,
) -> list[_ParetoDraft]:
    if history is None or not history.persistent_conflict_edges:
        return []
    agent_ids = set(context.agent_rows)
    persistent = [
        edge
        for edge in history.persistent_conflict_edges
        if edge[0] in agent_ids and edge[1] in agent_ids
    ]
    if not persistent:
        return []
    recent = [set(row) for row in history.recent_neighborhoods]
    pair_reuse = {
        edge: sum(edge[0] in row and edge[1] in row for row in recent)
        for edge in persistent
    }
    ordered_edges = sorted(
        persistent,
        key=lambda edge: (
            pair_reuse[edge],
            -int(context.event_weights.get(edge[0], 0))
            - int(context.event_weights.get(edge[1], 0)),
            edge,
        ),
    )
    limit = min(int(maximum_neighborhood_size), len(agent_ids))
    if limit < 2:
        return []
    endpoints: set[int] = set()
    for edge in ordered_edges:
        addition = set(edge) - endpoints
        if len(endpoints) + len(addition) <= limit:
            endpoints.update(edge)
    if len(endpoints) < 2:
        return []
    repair_counts = dict(history.agent_repair_counts)
    bridge_score: collections.Counter[int] = collections.Counter()
    for left, right in ordered_edges:
        left_path = context.path_sets[left]
        right_path = context.path_sets[right]
        for agent, path in context.path_sets.items():
            if agent in endpoints:
                continue
            bridge_score[agent] += min(
                len(path & left_path), len(path & right_path)
            )
    connectors = [
        agent
        for agent, score in sorted(
            bridge_score.items(),
            key=lambda item: (
                -item[1],
                -int(context.event_weights.get(item[0], 0)),
                item[0],
            ),
        )
        if score > 0
    ]
    connector_budget = max(0, limit - len(endpoints))
    bridge_agents = tuple(sorted(endpoints | set(connectors[:connector_budget])))
    recent_agent_count: collections.Counter[int] = collections.Counter(
        agent for row in recent for agent in row
    )
    active = set(context.event_weights)
    undercovered_order = sorted(
        active,
        key=lambda agent: (
            int(recent_agent_count[agent]),
            int(repair_counts.get(agent, 0)),
            -int(context.event_weights.get(agent, 0)),
            agent,
        ),
    )
    undercovered_agents = tuple(
        sorted(
            endpoints
            | set(undercovered_order[: max(0, limit - len(endpoints))])
        )
    )
    raw = [
        ("paretopool-history-persistent-bridge", bridge_agents),
        ("paretopool-history-undercovered", undercovered_agents),
    ]
    result: list[_ParetoDraft] = []
    for family, agents in raw:
        if not agents:
            continue
        result.append(
            _ParetoDraft(
                agents=agents,
                selection_families=(family,),
                family_groups=(family.removeprefix("paretopool-"),),
                sources=("history",),
                support_count_by_family=((family, len(endpoints)),),
                natural_breakpoints_by_family=((family, (len(agents),)),),
            )
        )
    return result


def _merge_drafts(drafts: Iterable[_ParetoDraft]) -> list[_ParetoDraft]:
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    for draft in drafts:
        row = merged.setdefault(
            draft.agents,
            {
                "families": set(),
                "groups": set(),
                "sources": set(),
                "support": {},
                "breakpoints": {},
            },
        )
        row["families"].update(draft.selection_families)
        row["groups"].update(draft.family_groups)
        row["sources"].update(draft.sources)
        row["support"].update(dict(draft.support_count_by_family))
        row["breakpoints"].update(dict(draft.natural_breakpoints_by_family))
    return [
        _ParetoDraft(
            agents=agents,
            selection_families=tuple(sorted(row["families"])),
            family_groups=tuple(sorted(row["groups"])),
            sources=tuple(sorted(row["sources"])),
            support_count_by_family=tuple(sorted(row["support"].items())),
            natural_breakpoints_by_family=tuple(sorted(row["breakpoints"].items())),
        )
        for agents, row in sorted(merged.items())
    ]


def _materialize(context: Any, draft: _ParetoDraft, anchor: tuple[int, ...]) -> dict[str, Any]:
    audit = topology_candidate_audit(context.analysis, draft.agents)
    known_groups = [
        group
        for group in draft.family_groups
        if group
        in {
            "bottleneck_crossing",
            "conflict_component",
            "topology_boundary",
            "spatiotemporal_hotspot",
            "path_overlap",
        }
    ]
    structural_score = max(
        (_structpool_score(audit, family_group=group) for group in known_groups),
        default=(
            2.0 * float(audit["global_event_incident_coverage"])
            + float(audit["global_pair_internal_coverage"])
            + 0.25 * float(audit["conflict_component_reach"])
        ),
    )
    families = list(draft.selection_families)
    support = dict(draft.support_count_by_family)
    return {
        "candidate_id": candidate_id(draft.agents),
        "agents": list(draft.agents),
        "actual_size": len(draft.agents),
        "selection_families": families,
        "selection_rank_by_family": {family: 0 for family in families},
        "proposal_count_by_family": {family: 1 for family in families},
        "proposal_seeds": [],
        "seed_agents": [],
        "proposal_audit": audit,
        "structpool_family_groups": list(draft.family_groups),
        "structpool_score": float(structural_score),
        "structpool_support_count_by_family": support,
        "structpool_support_ratio_by_family": {
            family: count / max(1, len(context.agent_rows))
            for family, count in support.items()
        },
        "structpool_nominal_size_by_family": {
            family: len(draft.agents) for family in families
        },
        "paretopool_id": PARETOPOOL_ID,
        "paretopool_sources": list(draft.sources),
        "paretopool_history_conditioned": "history" in draft.sources,
        "paretopool_natural_breakpoints_by_family": {
            family: list(values)
            for family, values in draft.natural_breakpoints_by_family
        },
        "v2_anchor_jaccard": _jaccard(draft.agents, anchor),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    first = left["proposal_audit"]
    second = right["proposal_audit"]
    comparisons = (
        int(left["actual_size"]) <= int(right["actual_size"]),
        float(first["global_event_incident_coverage"])
        >= float(second["global_event_incident_coverage"]),
        float(first["global_pair_internal_coverage"])
        >= float(second["global_pair_internal_coverage"]),
        float(first["conflict_component_reach"])
        >= float(second["conflict_component_reach"]),
    )
    strict = (
        int(left["actual_size"]) < int(right["actual_size"])
        or float(first["global_event_incident_coverage"])
        > float(second["global_event_incident_coverage"])
        or float(first["global_pair_internal_coverage"])
        > float(second["global_pair_internal_coverage"])
        or float(first["conflict_component_reach"])
        > float(second["conflict_component_reach"])
    )
    return all(comparisons) and strict


def _front_ranks(rows: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, list[str]]]:
    remaining = {str(row["candidate_id"]): row for row in rows}
    ranks: dict[str, int] = {}
    dominated_by: dict[str, list[str]] = {}
    rank = 0
    while remaining:
        front = []
        for candidate_id_value, candidate in remaining.items():
            dominators = [
                other_id
                for other_id, other in remaining.items()
                if other_id != candidate_id_value and _dominates(other, candidate)
            ]
            if not dominators:
                front.append(candidate_id_value)
            else:
                dominated_by.setdefault(candidate_id_value, sorted(dominators))
        if not front:
            raise RuntimeError("ParetoPool dominance relation produced no front")
        for candidate_id_value in sorted(front):
            ranks[candidate_id_value] = rank
            remaining.pop(candidate_id_value)
        rank += 1
    return ranks, dominated_by


def generate_paretopool_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_anchor_agents: Iterable[int],
    history: TemporalHistoryContext | None = None,
    maximum_candidates: int = 6,
    maximum_neighborhood_size: int = 64,
    maximum_jaccard_similarity: float = 0.8,
    maximum_anchor_jaccard_similarity: float = 0.9,
) -> ParetoPoolGenerationResult:
    if maximum_candidates not in PARETOPOOL_BUDGETS:
        raise ValueError("ParetoPool budget must be 6, 8, or 12")
    if maximum_neighborhood_size <= 0:
        raise ValueError("ParetoPool maximum neighborhood size must be positive")
    if not 0.0 <= maximum_jaccard_similarity < 1.0:
        raise ValueError("ParetoPool candidate Jaccard threshold must be in [0, 1)")
    if not 0.0 <= maximum_anchor_jaccard_similarity < 1.0:
        raise ValueError("ParetoPool anchor Jaccard threshold must be in [0, 1)")
    anchor = tuple(sorted(set(map(int, v2_anchor_agents))))
    if not anchor:
        raise ValueError("ParetoPool requires a non-empty V2 anchor")
    if not analysis.events:
        return ParetoPoolGenerationResult([], [], 0, 0, 0)
    context = _structural_candidate_context(state, analysis)
    current = _current_drafts(
        context, maximum_neighborhood_size=maximum_neighborhood_size
    )
    historical = _history_drafts(
        context,
        history,
        maximum_neighborhood_size=maximum_neighborhood_size,
    )[:2]
    merged = _merge_drafts([*current, *historical])
    rows = [_materialize(context, draft, anchor) for draft in merged]
    ranks, dominated_by = _front_ranks(rows)
    for row in rows:
        identity = str(row["candidate_id"])
        row["paretopool_front_rank"] = int(ranks[identity])
        row["paretopool_dominated_by"] = list(dominated_by.get(identity, ()))

    def order_key(row: dict[str, Any]) -> tuple[Any, ...]:
        audit = row["proposal_audit"]
        return (
            int(row["paretopool_front_rank"]),
            -float(audit["global_event_incident_coverage"]),
            -float(audit["global_pair_internal_coverage"]),
            -float(audit["conflict_component_reach"]),
            int(row["actual_size"]),
            str(row["candidate_id"]),
        )

    attempts: list[dict[str, Any]] = []
    eligible = []
    for row in sorted(rows, key=order_key):
        if float(row["v2_anchor_jaccard"]) > maximum_anchor_jaccard_similarity:
            attempts.append(
                {
                    "candidate_id": row["candidate_id"],
                    "decision": "rejected",
                    "rejection_reason": "v2_anchor_jaccard",
                    "pareto_front_rank": row["paretopool_front_rank"],
                }
            )
        else:
            eligible.append(row)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(row: dict[str, Any]) -> bool:
        identity = str(row["candidate_id"])
        if identity in selected_ids:
            return False
        maximum_similarity = max(
            (_jaccard(row["agents"], previous["agents"]) for previous in selected),
            default=0.0,
        )
        if maximum_similarity > maximum_jaccard_similarity:
            attempts.append(
                {
                    "candidate_id": identity,
                    "decision": "rejected",
                    "rejection_reason": "candidate_jaccard",
                    "maximum_selected_jaccard": maximum_similarity,
                    "pareto_front_rank": row["paretopool_front_rank"],
                }
            )
            selected_ids.add(identity)
            return False
        selected.append(row)
        selected_ids.add(identity)
        attempts.append(
            {
                "candidate_id": identity,
                "decision": "selected",
                "rejection_reason": None,
                "maximum_selected_jaccard": maximum_similarity,
                "pareto_front_rank": row["paretopool_front_rank"],
            }
        )
        return True

    group_order = (
        "bottleneck_crossing",
        "conflict_component",
        "topology_boundary",
        "spatiotemporal_hotspot",
        "path_overlap",
        "history-persistent-bridge",
        "history-undercovered",
    )
    for group in group_order:
        if len(selected) >= maximum_candidates:
            break
        choice = next(
            (
                row
                for row in eligible
                if group in set(row["structpool_family_groups"])
                and str(row["candidate_id"]) not in selected_ids
            ),
            None,
        )
        if choice is not None:
            add(choice)
    for row in eligible:
        if len(selected) >= maximum_candidates:
            break
        add(row)
    for row in eligible:
        identity = str(row["candidate_id"])
        if identity not in selected_ids:
            attempts.append(
                {
                    "candidate_id": identity,
                    "decision": "rejected",
                    "rejection_reason": "maximum_candidates",
                    "pareto_front_rank": row["paretopool_front_rank"],
                }
            )
    selected.sort(key=order_key)
    return ParetoPoolGenerationResult(
        candidates=selected,
        attempts=attempts,
        raw_candidate_count=len(rows),
        pareto_front_count=max(ranks.values(), default=-1) + 1,
        history_candidate_count=sum(
            bool(row["paretopool_history_conditioned"]) for row in rows
        ),
    )


__all__ = [
    "PARETOPOOL_BUDGETS",
    "PARETOPOOL_ID",
    "ParetoPoolGenerationResult",
    "generate_paretopool_candidates",
    "natural_structural_breakpoints",
]
