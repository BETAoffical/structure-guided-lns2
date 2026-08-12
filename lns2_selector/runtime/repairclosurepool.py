from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Iterable

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.topology_candidates import _jaccard, topology_candidate_audit


REPAIRCLOSUREPOOL_ID = "stride-repairclosurepool-v1"


@dataclass(frozen=True)
class DependencyEvidence:
    conflict_events: int
    temporal_reservations: int
    bottleneck_cells: int
    shared_path_cells: int

    @property
    def active_dimensions(self) -> int:
        return sum(value > 0 for value in self.as_tuple())

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            int(self.conflict_events),
            int(self.temporal_reservations),
            int(self.bottleneck_cells),
            int(self.shared_path_cells),
        )


@dataclass(frozen=True)
class _ClosureDraft:
    agents: tuple[int, ...]
    core_id: str
    core_agents: tuple[int, ...]
    closure_layer: int
    dependent_agent_count: int
    included_dependent_agent_count: int
    evidence_by_agent: tuple[tuple[int, tuple[int, int, int, int]], ...]


@dataclass(frozen=True)
class _DependencyContext:
    paths: dict[int, tuple[int, ...]]
    path_cells: dict[int, frozenset[int]]
    horizon: int


@dataclass
class RepairClosurePoolResult:
    candidates: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    raw_candidate_count: int
    pareto_front_count: int


def _dominates_evidence(left: DependencyEvidence, right: DependencyEvidence) -> bool:
    first = left.as_tuple()
    second = right.as_tuple()
    return all(a >= b for a, b in zip(first, second)) and any(
        a > b for a, b in zip(first, second)
    )


def _evidence_fronts(
    evidence: dict[int, DependencyEvidence],
) -> list[tuple[int, ...]]:
    agents_by_value: dict[DependencyEvidence, list[int]] = collections.defaultdict(list)
    for agent, value in evidence.items():
        agents_by_value[value].append(agent)
    ranks: dict[DependencyEvidence, int] = {}
    for value in sorted(
        agents_by_value,
        key=lambda item: (-sum(item.as_tuple()), tuple(-part for part in item.as_tuple())),
    ):
        dominator_ranks = [
            rank for other, rank in ranks.items() if _dominates_evidence(other, value)
        ]
        ranks[value] = 1 + max(dominator_ranks) if dominator_ranks else 0
    fronts: dict[int, list[int]] = collections.defaultdict(list)
    for value, agents in agents_by_value.items():
        fronts[ranks[value]].extend(agents)
    return [tuple(sorted(fronts[rank])) for rank in sorted(fronts)]


def _positions(path: Iterable[int], horizon: int) -> tuple[int, ...]:
    values = list(map(int, path))
    if not values:
        raise ValueError("RepairClosurePool requires non-empty paths")
    return tuple(values[min(index, len(values) - 1)] for index in range(horizon))


def _dependency_context(state: dict[str, Any]) -> _DependencyContext:
    rows = {int(row["id"]): row for row in state["agents"]}
    horizon = max(len(row["path"]) for row in rows.values())
    paths = {agent: _positions(row["path"], horizon) for agent, row in rows.items()}
    return _DependencyContext(
        paths=paths,
        path_cells={agent: frozenset(path) for agent, path in paths.items()},
        horizon=horizon,
    )


def _dependency_evidence(
    context: _DependencyContext,
    analysis: StateAnalysis,
    core: set[int],
    *,
    temporal_window: int,
) -> dict[int, DependencyEvidence]:
    if not core or not core <= set(context.paths):
        raise ValueError("RepairClosurePool core contains unknown agents")
    core_path_cells = {cell for agent in core for cell in context.paths[agent]}
    bottlenecks = {
        cell
        for cell in core_path_cells
        if cell in analysis.articulation or int(analysis.degrees.get(cell, 4)) <= 2
    }
    core_times: dict[int, set[int]] = collections.defaultdict(set)
    for agent in core:
        for time, cell in enumerate(context.paths[agent]):
            core_times[cell].add(time)
    conflict_counts: collections.Counter[int] = collections.Counter()
    for event in analysis.events:
        if event.left in core and event.right not in core:
            conflict_counts[event.right] += 1
        elif event.right in core and event.left not in core:
            conflict_counts[event.left] += 1
    temporal_reservations = {
        (cell, time + delta)
        for cell, times in core_times.items()
        for time in times
        for delta in range(-temporal_window, temporal_window + 1)
        if 0 <= time + delta < context.horizon
    }
    result: dict[int, DependencyEvidence] = {}
    for agent, path in context.paths.items():
        if agent in core:
            continue
        temporal = sum(
            (cell, time) in temporal_reservations
            for time, cell in enumerate(path)
        )
        path_cells = context.path_cells[agent]
        shared = len(path_cells & core_path_cells)
        bottleneck = len(path_cells & bottlenecks)
        evidence = DependencyEvidence(
            conflict_events=int(conflict_counts[agent]),
            temporal_reservations=temporal,
            bottleneck_cells=bottleneck,
            shared_path_cells=shared,
        )
        if evidence.active_dimensions:
            result[agent] = evidence
    return result


def _closure_drafts(
    context: _DependencyContext,
    analysis: StateAnalysis,
    *,
    core_id: str,
    core_agents: Iterable[int],
    temporal_window: int,
    maximum_neighborhood_size: int,
) -> list[_ClosureDraft]:
    core = set(map(int, core_agents))
    active = {agent for edge in analysis.pair_set for agent in edge}
    if not core or not core & active or len(core) > maximum_neighborhood_size:
        return []
    evidence = _dependency_evidence(
        context, analysis, core, temporal_window=temporal_window
    )
    ordered_evidence = tuple(
        sorted((agent, value.as_tuple()) for agent, value in evidence.items())
    )
    drafts = [
        _ClosureDraft(
            agents=tuple(sorted(core)),
            core_id=core_id,
            core_agents=tuple(sorted(core)),
            closure_layer=0,
            dependent_agent_count=len(evidence),
            included_dependent_agent_count=0,
            evidence_by_agent=ordered_evidence,
        )
    ]
    selected = set(core)
    included = 0
    for layer, front in enumerate(_evidence_fronts(evidence), start=1):
        if len(selected) + len(front) > maximum_neighborhood_size:
            break
        selected.update(front)
        included += len(front)
        drafts.append(
            _ClosureDraft(
                agents=tuple(sorted(selected)),
                core_id=core_id,
                core_agents=tuple(sorted(core)),
                closure_layer=layer,
                dependent_agent_count=len(evidence),
                included_dependent_agent_count=included,
                evidence_by_agent=ordered_evidence,
            )
        )
    return drafts


def _materialize(
    analysis: StateAnalysis, draft: _ClosureDraft, agent_count: int
) -> dict[str, Any]:
    audit = topology_candidate_audit(analysis, draft.agents)
    uncovered = draft.dependent_agent_count - draft.included_dependent_agent_count
    return {
        "candidate_id": candidate_id(draft.agents),
        "agents": list(draft.agents),
        "actual_size": len(draft.agents),
        "selection_families": [f"repairclosure:{draft.core_id}:layer_{draft.closure_layer}"],
        "selection_rank_by_family": {
            f"repairclosure:{draft.core_id}": int(draft.closure_layer)
        },
        "proposal_count_by_family": {f"repairclosure:{draft.core_id}": 1},
        "proposal_seeds": [],
        "seed_agents": list(draft.core_agents),
        "structpool_family_groups": ["repair_dependency_closure"],
        "candidate_kind": "repairclosure",
        "repairclosurepool_id": REPAIRCLOSUREPOOL_ID,
        "repairclosure_core_id": draft.core_id,
        "repairclosure_core_agents": list(draft.core_agents),
        "repairclosure_layer": int(draft.closure_layer),
        "repairclosure_dependent_agent_count": int(draft.dependent_agent_count),
        "repairclosure_included_dependent_agent_count": int(
            draft.included_dependent_agent_count
        ),
        "repairclosure_uncovered_dependency_ratio": (
            uncovered / draft.dependent_agent_count
            if draft.dependent_agent_count
            else 0.0
        ),
        "repairclosure_evidence_by_agent": {
            str(agent): {
                "conflict_events": values[0],
                "temporal_reservations": values[1],
                "bottleneck_cells": values[2],
                "shared_path_cells": values[3],
            }
            for agent, values in draft.evidence_by_agent
        },
        "proposal_audit": audit,
        "neighborhood_fraction": len(draft.agents) / max(1, agent_count),
    }


def _candidate_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_audit = left["proposal_audit"]
    right_audit = right["proposal_audit"]
    comparisons = (
        int(left["actual_size"]) <= int(right["actual_size"]),
        float(left["repairclosure_uncovered_dependency_ratio"])
        <= float(right["repairclosure_uncovered_dependency_ratio"]),
        float(left_audit["global_event_incident_coverage"])
        >= float(right_audit["global_event_incident_coverage"]),
        float(left_audit["global_pair_internal_coverage"])
        >= float(right_audit["global_pair_internal_coverage"]),
    )
    strict = (
        int(left["actual_size"]) < int(right["actual_size"])
        or float(left["repairclosure_uncovered_dependency_ratio"])
        < float(right["repairclosure_uncovered_dependency_ratio"])
        or float(left_audit["global_event_incident_coverage"])
        > float(right_audit["global_event_incident_coverage"])
        or float(left_audit["global_pair_internal_coverage"])
        > float(right_audit["global_pair_internal_coverage"])
    )
    return all(comparisons) and strict


def _front_ranks(rows: list[dict[str, Any]]) -> dict[str, int]:
    remaining = {str(row["candidate_id"]): row for row in rows}
    ranks: dict[str, int] = {}
    rank = 0
    while remaining:
        front = [
            identity
            for identity, row in remaining.items()
            if not any(
                other_id != identity and _candidate_dominates(other, row)
                for other_id, other in remaining.items()
            )
        ]
        if not front:
            raise RuntimeError("RepairClosurePool candidate dominance produced no front")
        for identity in sorted(front):
            ranks[identity] = rank
            remaining.pop(identity)
        rank += 1
    return ranks


def generate_repairclosure_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_anchors: Iterable[dict[str, Any]],
    maximum_candidates: int = 12,
    maximum_neighborhood_size: int = 64,
    temporal_window: int = 2,
    maximum_jaccard_similarity: float = 0.9,
) -> RepairClosurePoolResult:
    if maximum_candidates <= 0 or maximum_neighborhood_size <= 0:
        raise ValueError("RepairClosurePool limits must be positive")
    if temporal_window < 0:
        raise ValueError("RepairClosurePool temporal window must be non-negative")
    if not 0.0 <= maximum_jaccard_similarity < 1.0:
        raise ValueError("RepairClosurePool Jaccard threshold must be in [0, 1)")
    if not analysis.events:
        return RepairClosurePoolResult([], [], 0, 0)
    agent_count = len(state["agents"])
    context = _dependency_context(state)
    base_sets: dict[tuple[int, ...], str] = {}
    cores: list[tuple[str, tuple[int, ...]]] = []
    for anchor in v2_anchors:
        agents = tuple(sorted(set(map(int, anchor.get("agents") or ()))))
        if not agents:
            raise ValueError("RepairClosurePool V2 anchor is empty")
        base_sets[agents] = str(anchor["candidate_id"])
        cores.append((f"v2_{str(anchor['candidate_id'])[-8:]}", agents))
    for component, members in sorted(analysis.component_members.items()):
        cores.append((f"component_{int(component)}", tuple(sorted(members))))

    drafts: list[_ClosureDraft] = []
    for core_id, agents in cores:
        drafts.extend(
            _closure_drafts(
                context,
                analysis,
                core_id=core_id,
                core_agents=agents,
                temporal_window=temporal_window,
                maximum_neighborhood_size=maximum_neighborhood_size,
            )
        )
    merged: dict[tuple[int, ...], _ClosureDraft] = {}
    for draft in drafts:
        prior = merged.get(draft.agents)
        if prior is None or (
            draft.closure_layer,
            draft.core_id,
        ) < (
            prior.closure_layer,
            prior.core_id,
        ):
            merged[draft.agents] = draft
    attempts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for agents, draft in sorted(merged.items()):
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
        rows.append(_materialize(analysis, draft, agent_count))
    if not rows:
        return RepairClosurePoolResult([], attempts, len(merged), 0)
    ranks = _front_ranks(rows)
    for row in rows:
        row["repairclosure_pareto_rank"] = ranks[str(row["candidate_id"])]

    def order_key(row: dict[str, Any]) -> tuple[Any, ...]:
        audit = row["proposal_audit"]
        return (
            int(row["repairclosure_pareto_rank"]),
            float(row["repairclosure_uncovered_dependency_ratio"]),
            -float(audit["global_event_incident_coverage"]),
            -float(audit["global_pair_internal_coverage"]),
            int(row["actual_size"]),
            str(row["candidate_id"]),
        )

    selected: list[dict[str, Any]] = []
    considered: set[str] = set()

    def add(row: dict[str, Any]) -> bool:
        identity = str(row["candidate_id"])
        if identity in considered:
            return False
        considered.add(identity)
        similarity = max(
            (_jaccard(row["agents"], other["agents"]) for other in selected),
            default=0.0,
        )
        if similarity > maximum_jaccard_similarity:
            attempts.append(
                {
                    "candidate_id": identity,
                    "decision": "rejected",
                    "rejection_reason": "candidate_jaccard",
                    "maximum_selected_jaccard": similarity,
                }
            )
            return False
        selected.append(row)
        attempts.append(
            {
                "candidate_id": identity,
                "decision": "selected",
                "rejection_reason": None,
                "maximum_selected_jaccard": similarity,
            }
        )
        return True

    ordered = sorted(rows, key=order_key)
    for row in ordered:
        if len(selected) >= maximum_candidates:
            break
        add(row)
    return RepairClosurePoolResult(
        candidates=selected,
        attempts=attempts,
        raw_candidate_count=len(rows),
        pareto_front_count=sum(rank == 0 for rank in ranks.values()),
    )


__all__ = [
    "DependencyEvidence",
    "REPAIRCLOSUREPOOL_ID",
    "RepairClosurePoolResult",
    "generate_repairclosure_candidates",
]
