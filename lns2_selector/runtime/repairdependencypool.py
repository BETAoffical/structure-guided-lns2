from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.causaltopopool import (
    _family_group,
    _natural_closures,
    _structural_candidate_context,
    _variant_support,
)
from lns2_selector.runtime.topology_candidates import _jaccard, topology_candidate_audit


REPAIRDEPENDENCYPOOL_ID = "stride-repairdependencypool-v1"
_CORE_VARIANTS = (
    "topology_boundary_articulation",
    "topology_boundary_low_degree",
    "spatiotemporal_hotspot",
    "path_overlap",
)
_FAMILY_ROTATION = ("topology_boundary", "spatiotemporal_hotspot", "path_overlap")


@dataclass
class RepairDependencyPoolResult:
    candidates: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    core_count: int
    raw_variant_count: int
    exact_existing_duplicate_count: int
    exact_new_duplicate_count: int
    jaccard_rejection_count: int
    oversized_rejection_count: int


def _positions(path: Iterable[int], horizon: int) -> tuple[int, ...]:
    values = tuple(map(int, path))
    if not values:
        raise ValueError("RepairDependencyPool requires non-empty paths")
    return tuple(values[min(index, len(values) - 1)] for index in range(horizon))


def _core_drafts(state: dict[str, Any], analysis: StateAnalysis) -> list[dict[str, Any]]:
    context = _structural_candidate_context(state, analysis)
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    for variant in _CORE_VARIANTS:
        support = _variant_support(context, variant)
        for level, agents in _natural_closures(context, variant):
            ordered = tuple(sorted(map(int, agents)))
            if len(ordered) < 2:
                continue
            row = merged.setdefault(
                ordered,
                {
                    "agents": ordered,
                    "families": set(),
                    "variants": set(),
                    "closure_levels": set(),
                    "support_agents": set(),
                },
            )
            row["families"].add(_family_group(variant))
            row["variants"].add(variant)
            row["closure_levels"].add(level)
            row["support_agents"].update(map(int, support))
    return [
        {
            "core_id": candidate_id(agents),
            "agents": list(agents),
            "families": sorted(row["families"]),
            "variants": sorted(row["variants"]),
            "closure_levels": sorted(row["closure_levels"]),
            "support_agents": sorted(row["support_agents"]),
        }
        for agents, row in sorted(merged.items())
    ]


def _direct_boundary_evidence(
    core: set[int], analysis: StateAnalysis
) -> dict[int, dict[str, Any]]:
    evidence: dict[int, dict[str, Any]] = {}
    for event in analysis.events:
        left, right = int(event.left), int(event.right)
        if (left in core) == (right in core):
            continue
        blocker = right if left in core else left
        row = evidence.setdefault(
            blocker,
            {
                "current_conflict_edges": set(),
                "conflict_events": [],
                "corridor_cells": set(),
                "temporal_overlap_count": 0,
                "reverse_queue_count": 0,
            },
        )
        row["current_conflict_edges"].add(tuple(sorted((left, right))))
        row["conflict_events"].append(
            {
                "time": int(event.time),
                "kind": str(event.kind),
                "agents": [left, right],
                "cells": list(map(int, event.cells)),
            }
        )
    return evidence


def _temporal_corridor_evidence(
    state: dict[str, Any], analysis: StateAnalysis, core: set[int]
) -> dict[int, dict[str, Any]]:
    agents = {int(row["id"]): row for row in state["agents"]}
    horizon = max(len(row["path"]) for row in agents.values())
    paths = {
        agent: _positions(row["path"], horizon) for agent, row in agents.items()
    }
    corridor = {
        int(cell)
        for cell in analysis.free_cells
        if int(cell) in analysis.articulation or int(analysis.degrees.get(cell, 4)) <= 2
    }
    evidence: dict[int, dict[str, Any]] = {}
    for blocker in sorted(set(paths) - core):
        overlap_count = 0
        reverse_count = 0
        cells: set[int] = set()
        for member in sorted(core):
            member_path = paths[member]
            blocker_path = paths[blocker]
            for time in range(horizon):
                member_cell = member_path[time]
                if member_cell not in corridor:
                    continue
                for delta in (-1, 0, 1):
                    other_time = time + delta
                    if 0 <= other_time < horizon and blocker_path[other_time] == member_cell:
                        overlap_count += 1
                        cells.add(member_cell)
                if time == 0:
                    continue
                member_previous = member_path[time - 1]
                if member_previous == member_cell:
                    continue
                for delta in (-1, 0, 1):
                    other_time = time + delta
                    if other_time <= 0 or other_time >= horizon:
                        continue
                    if (
                        blocker_path[other_time - 1] == member_cell
                        and blocker_path[other_time] == member_previous
                    ):
                        reverse_count += 1
                        cells.update((member_previous, member_cell))
        # A temporal-corridor dependency needs both queue overlap and opposing flow.
        if overlap_count and reverse_count:
            evidence[blocker] = {
                "current_conflict_edges": set(),
                "conflict_events": [],
                "corridor_cells": cells,
                "temporal_overlap_count": overlap_count,
                "reverse_queue_count": reverse_count,
            }
    return evidence


def _merge_evidence(
    direct: dict[int, dict[str, Any]], temporal: dict[int, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for agent in sorted(set(direct) | set(temporal)):
        left = direct.get(agent, {})
        right = temporal.get(agent, {})
        merged[agent] = {
            "current_conflict_edges": set(left.get("current_conflict_edges", ()))
            | set(right.get("current_conflict_edges", ())),
            "conflict_events": list(left.get("conflict_events", ()))
            + list(right.get("conflict_events", ())),
            "corridor_cells": set(left.get("corridor_cells", ()))
            | set(right.get("corridor_cells", ())),
            "temporal_overlap_count": int(left.get("temporal_overlap_count", 0))
            + int(right.get("temporal_overlap_count", 0)),
            "reverse_queue_count": int(left.get("reverse_queue_count", 0))
            + int(right.get("reverse_queue_count", 0)),
        }
    return merged


def _evidence_weight(row: dict[str, Any]) -> int:
    return (
        4 * len(row["current_conflict_edges"])
        + 2 * int(row["reverse_queue_count"])
        + int(row["temporal_overlap_count"])
    )


def _serialize_evidence(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_conflict_edges": [
            list(edge) for edge in sorted(row["current_conflict_edges"])
        ],
        "conflict_events": sorted(
            row["conflict_events"],
            key=lambda item: (
                int(item["time"]),
                str(item["kind"]),
                tuple(item["agents"]),
                tuple(item["cells"]),
            ),
        ),
        "corridor_cells": sorted(map(int, row["corridor_cells"])),
        "temporal_overlap_count": int(row["temporal_overlap_count"]),
        "reverse_queue_count": int(row["reverse_queue_count"]),
        "evidence_weight": _evidence_weight(row),
    }


def _materialize_variant(
    core: dict[str, Any],
    evidence: dict[int, dict[str, Any]],
    analysis: StateAnalysis,
    *,
    variant: str,
    maximum_added_agents: int,
    maximum_neighborhood_size: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    ordered_blockers = sorted(
        evidence,
        key=lambda agent: (-_evidence_weight(evidence[agent]), int(agent)),
    )
    added = ordered_blockers[:maximum_added_agents]
    attempt = {
        "core_id": str(core["core_id"]),
        "core_agents": list(core["agents"]),
        "core_families": list(core["families"]),
        "variant": variant,
        "visible_supported_blocker_count": len(ordered_blockers),
        "proposed_added_blockers": list(added),
        "proposed_size": len(core["agents"]) + len(added),
    }
    if not added:
        return None, {
            **attempt,
            "decision": "rejected",
            "rejection_reason": "no_supported_external_dependency",
        }
    if len(core["agents"]) + len(added) > maximum_neighborhood_size:
        return None, {
            **attempt,
            "decision": "rejected",
            "rejection_reason": "variant_oversized_without_truncation",
        }
    agents = sorted(set(map(int, core["agents"])) | set(added))
    audit = topology_candidate_audit(analysis, agents)
    total_weight = sum(_evidence_weight(evidence[agent]) for agent in added)
    family = min(core["families"])
    row = {
        "candidate_id": candidate_id(agents),
        "agents": agents,
        "actual_size": len(agents),
        "core_agents": list(core["agents"]),
        "added_blockers": list(added),
        "blocker_evidence": {
            str(agent): _serialize_evidence(evidence[agent]) for agent in added
        },
        "rejection_reason": None,
        "selection_source": f"repairdependency:{family}:{variant}",
        "selection_families": [f"repairdependency:{family}:{variant}"],
        "selection_rank_by_family": {f"repairdependency:{family}:{variant}": 0},
        "proposal_count_by_family": {f"repairdependency:{family}:{variant}": 1},
        "proposal_seeds": [],
        "seed_agents": list(core["support_agents"]),
        "proposal_audit": audit,
        "candidate_kind": "repairdependency",
        "repairdependencypool_id": REPAIRDEPENDENCYPOOL_ID,
        "repairdependency_core_id": str(core["core_id"]),
        "repairdependency_core_families": list(core["families"]),
        "repairdependency_core_variants": list(core["variants"]),
        "repairdependency_core_closure_levels": list(core["closure_levels"]),
        "repairdependency_variant": variant,
        "dependency_coverage_count": len(added),
        "dependency_coverage_ratio": len(added) / len(ordered_blockers),
        "evidence_weight": total_weight,
        "evidence_density": total_weight / len(added),
        "maximum_added_agents": maximum_added_agents,
        "truncated_to_near_global": False,
    }
    return row, {**attempt, "decision": "generated", "rejection_reason": None}


def generate_repairdependency_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    existing_candidates: Iterable[dict[str, Any]] = (),
    maximum_candidates: int = 6,
    maximum_added_agents: int = 8,
    maximum_neighborhood_size: int = 32,
    maximum_jaccard_similarity: float = 0.9,
) -> RepairDependencyPoolResult:
    if maximum_candidates <= 0 or maximum_added_agents <= 0:
        raise ValueError("RepairDependencyPool candidate limits must be positive")
    if maximum_neighborhood_size <= 1:
        raise ValueError("RepairDependencyPool neighborhood limit must exceed one")
    if not 0.0 <= maximum_jaccard_similarity < 1.0:
        raise ValueError("RepairDependencyPool Jaccard threshold must be in [0, 1)")
    if not analysis.events:
        return RepairDependencyPoolResult([], [], 0, 0, 0, 0, 0, 0)

    existing_ids = {
        str(row.get("candidate_id") or candidate_id(row["agents"]))
        for row in existing_candidates
    }
    attempts: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    cores = _core_drafts(state, analysis)
    oversized = 0
    for core in cores:
        core_agents = set(map(int, core["agents"]))
        direct = _direct_boundary_evidence(core_agents, analysis)
        temporal = _temporal_corridor_evidence(state, analysis, core_agents)
        for variant, evidence in (
            ("direct-boundary", direct),
            ("temporal-corridor", _merge_evidence(direct, temporal)),
        ):
            row, attempt = _materialize_variant(
                core,
                evidence,
                analysis,
                variant=variant,
                maximum_added_agents=maximum_added_agents,
                maximum_neighborhood_size=maximum_neighborhood_size,
            )
            attempts.append(attempt)
            oversized += int(
                attempt.get("rejection_reason")
                == "variant_oversized_without_truncation"
            )
            if row is not None:
                raw.append(row)

    # Exact set deduplication keeps the strongest visible evidence and merges provenance.
    merged: dict[str, dict[str, Any]] = {}
    exact_new_duplicates = 0
    exact_existing = 0
    for row in sorted(
        raw,
        key=lambda item: (
            -int(item["dependency_coverage_count"]),
            -float(item["evidence_density"]),
            int(item["actual_size"]),
            str(item["candidate_id"]),
            str(item["selection_source"]),
        ),
    ):
        identity = str(row["candidate_id"])
        if identity in existing_ids:
            exact_existing += 1
            attempts.append(
                {
                    "candidate_id": identity,
                    "decision": "rejected",
                    "rejection_reason": "exact_existing_candidate",
                }
            )
            continue
        if identity in merged:
            exact_new_duplicates += 1
            target = merged[identity]
            target["selection_families"] = sorted(
                set(target["selection_families"]) | set(row["selection_families"])
            )
            attempts.append(
                {
                    "candidate_id": identity,
                    "decision": "rejected",
                    "rejection_reason": "exact_new_candidate_duplicate",
                }
            )
            continue
        merged[identity] = row

    ordered = sorted(
        merged.values(),
        key=lambda row: (
            -int(row["dependency_coverage_count"]),
            -float(row["evidence_density"]),
            int(row["actual_size"]),
            str(row["candidate_id"]),
        ),
    )
    queues: dict[str, list[dict[str, Any]]] = {family: [] for family in _FAMILY_ROTATION}
    for row in ordered:
        groups = list(row["repairdependency_core_families"])
        family = next((item for item in _FAMILY_ROTATION if item in groups), None)
        if family is not None:
            queues[family].append(row)

    selected: list[dict[str, Any]] = []
    jaccard_rejections = 0
    exhausted = False
    while len(selected) < maximum_candidates and not exhausted:
        exhausted = True
        for family in _FAMILY_ROTATION:
            queue = queues[family]
            while queue and len(selected) < maximum_candidates:
                exhausted = False
                row = queue.pop(0)
                similarity = max(
                    (_jaccard(row["agents"], prior["agents"]) for prior in selected),
                    default=0.0,
                )
                if similarity > maximum_jaccard_similarity:
                    jaccard_rejections += 1
                    attempts.append(
                        {
                            "candidate_id": row["candidate_id"],
                            "decision": "rejected",
                            "rejection_reason": "candidate_jaccard",
                            "maximum_selected_jaccard": similarity,
                        }
                    )
                    continue
                row["maximum_selected_jaccard"] = similarity
                selected.append(row)
                attempts.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "decision": "selected",
                        "rejection_reason": None,
                        "maximum_selected_jaccard": similarity,
                    }
                )
                break
    selected_ids = {str(row["candidate_id"]) for row in selected}
    for row in ordered:
        if str(row["candidate_id"]) not in selected_ids and not any(
            attempt.get("candidate_id") == row["candidate_id"]
            and attempt.get("rejection_reason") == "candidate_jaccard"
            for attempt in attempts
        ):
            attempts.append(
                {
                    "candidate_id": row["candidate_id"],
                    "decision": "rejected",
                    "rejection_reason": "maximum_candidates",
                }
            )
    return RepairDependencyPoolResult(
        candidates=selected,
        attempts=attempts,
        core_count=len(cores),
        raw_variant_count=len(raw),
        exact_existing_duplicate_count=exact_existing,
        exact_new_duplicate_count=exact_new_duplicates,
        jaccard_rejection_count=jaccard_rejections,
        oversized_rejection_count=oversized,
    )


def select_repairdependency_candidate(
    v2_anchor: dict[str, Any], candidates: Iterable[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply the frozen transparent rule; ambiguity falls back to the V2 anchor."""

    eligible = [
        dict(row)
        for row in candidates
        if row.get("repairdependencypool_id") == REPAIRDEPENDENCYPOOL_ID
        and int(row.get("dependency_coverage_count", 0)) > 0
        and row.get("frozen_v2_quality_score") is not None
    ]
    if not eligible:
        return dict(v2_anchor), {
            "selection_source": "v2_anchor_fallback",
            "rejection_reason": "no_quality_scored_dependency_candidate",
        }

    def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
        comparisons = (
            int(left["dependency_coverage_count"])
            >= int(right["dependency_coverage_count"]),
            float(left["evidence_density"]) >= float(right["evidence_density"]),
            int(left["actual_size"]) <= int(right["actual_size"]),
            float(left["frozen_v2_quality_score"])
            >= float(right["frozen_v2_quality_score"]),
        )
        strict = (
            int(left["dependency_coverage_count"])
            > int(right["dependency_coverage_count"])
            or float(left["evidence_density"]) > float(right["evidence_density"])
            or int(left["actual_size"]) < int(right["actual_size"])
            or float(left["frozen_v2_quality_score"])
            > float(right["frozen_v2_quality_score"])
        )
        return all(comparisons) and strict

    front = [
        row
        for row in eligible
        if not any(
            other["candidate_id"] != row["candidate_id"] and dominates(other, row)
            for other in eligible
        )
    ]
    if len(front) != 1:
        return dict(v2_anchor), {
            "selection_source": "v2_anchor_fallback",
            "rejection_reason": "dependency_pareto_not_unique",
            "pareto_candidate_ids": sorted(str(row["candidate_id"]) for row in front),
        }
    selected = front[0]
    return selected, {
        "selection_source": "repairdependency_unique_pareto",
        "rejection_reason": None,
        "candidate_id": str(selected["candidate_id"]),
    }


__all__ = [
    "REPAIRDEPENDENCYPOOL_ID",
    "RepairDependencyPoolResult",
    "generate_repairdependency_candidates",
    "select_repairdependency_candidate",
]
