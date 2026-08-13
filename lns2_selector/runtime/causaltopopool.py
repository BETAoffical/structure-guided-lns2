from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.topology_candidates import (
    _SCALEPOOL_VARIANT_ORDER,
    _jaccard,
    _structpool_family_name,
    _structpool_score,
    _structural_candidate_context,
    topology_candidate_audit,
)


CAUSALTOPOPOOL_ID = "stride-causaltopopool-v1"


@dataclass(frozen=True)
class _NaturalDraft:
    agents: tuple[int, ...]
    variants: tuple[str, ...]
    family_groups: tuple[str, ...]
    closure_levels: tuple[str, ...]
    support_count_by_variant: tuple[tuple[str, int], ...]


@dataclass
class CausalTopoPoolResult:
    candidates: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    closure_attempt_count: int
    oversized_closure_count: int
    undersized_closure_count: int
    exact_existing_duplicate_count: int
    raw_candidate_count: int
    pareto_front_count: int


def _family_group(variant: str) -> str:
    if variant.startswith("topology_boundary_"):
        return "topology_boundary"
    return variant


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
    raise ValueError(f"unsupported CausalTopoPool family: {variant}")


def _natural_closures(context: Any, variant: str) -> tuple[tuple[str, set[int]], ...]:
    """Build exact state-derived closures; no target size is filled or truncated."""

    agent_ids = set(map(int, context.agent_rows))
    support = _variant_support(context, variant) & agent_ids
    if not support:
        return ()

    conflict_neighbors = set(support)
    for agent in support:
        conflict_neighbors.update(context.conflict_adjacency.get(agent, ()))

    touched_components = {
        context.analysis.component_id[agent]
        for agent in support
        if agent in context.analysis.component_id
    }
    component_closure = set(support)
    for component in touched_components:
        component_closure.update(context.analysis.component_members.get(component, ()))

    support_cells = {
        cell for agent in support for cell in context.path_sets.get(agent, ())
    }
    active_path_contacts = set(support)
    active_path_contacts.update(
        agent
        for agent, cells in context.path_sets.items()
        if agent in context.event_weights and cells & support_cells
    )

    return (
        ("support", support),
        ("conflict_neighbor_closure", conflict_neighbors),
        ("touched_component_closure", component_closure),
        ("active_path_contact_closure", active_path_contacts),
    )


def _merge_drafts(rows: list[dict[str, Any]]) -> list[_NaturalDraft]:
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in rows:
        agents = tuple(row["agents"])
        target = merged.setdefault(
            agents,
            {
                "variants": set(),
                "groups": set(),
                "levels": set(),
                "support": {},
            },
        )
        target["variants"].add(str(row["variant"]))
        target["groups"].add(str(row["family_group"]))
        target["levels"].add(str(row["closure_level"]))
        target["support"][str(row["variant"])] = int(row["support_count"])
    return [
        _NaturalDraft(
            agents=agents,
            variants=tuple(sorted(row["variants"])),
            family_groups=tuple(sorted(row["groups"])),
            closure_levels=tuple(sorted(row["levels"])),
            support_count_by_variant=tuple(sorted(row["support"].items())),
        )
        for agents, row in sorted(merged.items())
    ]


def _materialize(context: Any, draft: _NaturalDraft) -> dict[str, Any]:
    audit = topology_candidate_audit(context.analysis, draft.agents)
    structural_score = max(
        (
            _structpool_score(audit, family_group=group)
            for group in draft.family_groups
            if group
            in {
                "bottleneck_crossing",
                "conflict_component",
                "topology_boundary",
                "spatiotemporal_hotspot",
                "path_overlap",
            }
        ),
        default=0.0,
    )
    families = tuple(
        sorted(
            {
                _structpool_family_name(variant, len(draft.agents))
                for variant in draft.variants
            }
        )
    )
    support = dict(draft.support_count_by_variant)
    return {
        "candidate_id": candidate_id(draft.agents),
        "agents": list(draft.agents),
        "actual_size": len(draft.agents),
        "selection_families": [f"causaltopo:{family}" for family in families],
        "selection_rank_by_family": {
            f"causaltopo:{family}": 0 for family in families
        },
        "proposal_count_by_family": {
            f"causaltopo:{family}": 1 for family in families
        },
        "proposal_seeds": [],
        "seed_agents": sorted(
            {
                agent
                for variant in draft.variants
                for agent in _variant_support(context, variant)
            }
        ),
        "proposal_audit": audit,
        "structpool_family_groups": list(draft.family_groups),
        "structpool_score": float(structural_score),
        "candidate_kind": "causaltopology",
        "causaltopopool_id": CAUSALTOPOPOOL_ID,
        "causaltopo_variants": list(draft.variants),
        "causaltopo_closure_levels": list(draft.closure_levels),
        "causaltopo_support_count_by_variant": support,
        "causaltopo_added_agent_count_by_variant": {
            variant: len(draft.agents) - count
            for variant, count in support.items()
        },
        "causaltopo_exact_natural_closure": True,
        "causaltopo_truncated": False,
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_audit = left["proposal_audit"]
    right_audit = right["proposal_audit"]
    comparisons = (
        int(left["actual_size"]) <= int(right["actual_size"]),
        float(left_audit["global_event_incident_coverage"])
        >= float(right_audit["global_event_incident_coverage"]),
        float(left_audit["global_pair_internal_coverage"])
        >= float(right_audit["global_pair_internal_coverage"]),
        float(left_audit["conflict_component_reach"])
        >= float(right_audit["conflict_component_reach"]),
    )
    strict = (
        int(left["actual_size"]) < int(right["actual_size"])
        or float(left_audit["global_event_incident_coverage"])
        > float(right_audit["global_event_incident_coverage"])
        or float(left_audit["global_pair_internal_coverage"])
        > float(right_audit["global_pair_internal_coverage"])
        or float(left_audit["conflict_component_reach"])
        > float(right_audit["conflict_component_reach"])
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
                other_id != identity and _dominates(other, row)
                for other_id, other in remaining.items()
            )
        ]
        if not front:
            raise RuntimeError("CausalTopoPool dominance relation produced no front")
        for identity in sorted(front):
            ranks[identity] = rank
            remaining.pop(identity)
        rank += 1
    return ranks


def generate_causaltopopool_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    existing_candidates: Iterable[dict[str, Any]] = (),
    maximum_candidates: int = 12,
    maximum_neighborhood_size: int = 64,
    maximum_jaccard_similarity: float = 0.9,
) -> CausalTopoPoolResult:
    if maximum_candidates <= 0 or maximum_neighborhood_size <= 0:
        raise ValueError("CausalTopoPool limits must be positive")
    if not 0.0 <= maximum_jaccard_similarity < 1.0:
        raise ValueError("CausalTopoPool Jaccard threshold must be in [0, 1)")
    if not analysis.events:
        return CausalTopoPoolResult([], [], 0, 0, 0, 0, 0, 0)

    context = _structural_candidate_context(state, analysis)
    attempts: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    oversized = 0
    undersized = 0
    closure_attempts = 0
    for variant in _SCALEPOOL_VARIANT_ORDER:
        support_count = len(_variant_support(context, variant))
        for level, agents in _natural_closures(context, variant):
            closure_attempts += 1
            proposed = tuple(sorted(map(int, agents)))
            attempt = {
                "variant": variant,
                "family_group": _family_group(variant),
                "closure_level": level,
                "proposed_size": len(proposed),
            }
            if len(proposed) < 2:
                undersized += 1
                attempts.append(
                    {**attempt, "decision": "rejected", "rejection_reason": "undersized"}
                )
                continue
            if len(proposed) > maximum_neighborhood_size:
                oversized += 1
                attempts.append(
                    {
                        **attempt,
                        "decision": "rejected",
                        "rejection_reason": "oversized_exact_closure",
                    }
                )
                continue
            raw.append(
                {
                    **attempt,
                    "agents": proposed,
                    "support_count": support_count,
                }
            )

    drafts = _merge_drafts(raw)
    existing_ids = {
        str(candidate.get("candidate_id") or candidate_id(candidate["agents"]))
        for candidate in existing_candidates
    }
    rows: list[dict[str, Any]] = []
    exact_existing = 0
    for draft in drafts:
        row = _materialize(context, draft)
        if str(row["candidate_id"]) in existing_ids:
            exact_existing += 1
            attempts.append(
                {
                    "candidate_id": row["candidate_id"],
                    "proposed_size": row["actual_size"],
                    "decision": "rejected",
                    "rejection_reason": "exact_existing_candidate",
                }
            )
            continue
        rows.append(row)

    ranks = _front_ranks(rows) if rows else {}
    for row in rows:
        row["causaltopo_pareto_rank"] = int(ranks[str(row["candidate_id"])])

    def order_key(row: dict[str, Any]) -> tuple[Any, ...]:
        audit = row["proposal_audit"]
        return (
            int(row["causaltopo_pareto_rank"]),
            -float(audit["global_event_incident_coverage"]),
            -float(audit["global_pair_internal_coverage"]),
            -float(audit["conflict_component_reach"]),
            int(row["actual_size"]),
            str(row["candidate_id"]),
        )

    ordered = sorted(rows, key=order_key)
    selected: list[dict[str, Any]] = []
    considered: set[str] = set()

    def add(row: dict[str, Any]) -> bool:
        identity = str(row["candidate_id"])
        if identity in considered:
            return False
        considered.add(identity)
        similarity = max(
            (_jaccard(row["agents"], previous["agents"]) for previous in selected),
            default=0.0,
        )
        if similarity > maximum_jaccard_similarity:
            attempts.append(
                {
                    "candidate_id": identity,
                    "proposed_size": row["actual_size"],
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
                "proposed_size": row["actual_size"],
                "decision": "selected",
                "rejection_reason": None,
                "maximum_selected_jaccard": similarity,
            }
        )
        return True

    for group in (
        "bottleneck_crossing",
        "conflict_component",
        "topology_boundary",
        "spatiotemporal_hotspot",
        "path_overlap",
    ):
        if len(selected) >= maximum_candidates:
            break
        choice = next(
            (
                row
                for row in ordered
                if group in row["structpool_family_groups"]
                and str(row["candidate_id"]) not in considered
            ),
            None,
        )
        if choice is not None:
            add(choice)
    for row in ordered:
        if len(selected) >= maximum_candidates:
            break
        add(row)
    for row in ordered:
        if str(row["candidate_id"]) not in considered:
            attempts.append(
                {
                    "candidate_id": row["candidate_id"],
                    "proposed_size": row["actual_size"],
                    "decision": "rejected",
                    "rejection_reason": "maximum_candidates",
                }
            )
    selected.sort(key=order_key)
    return CausalTopoPoolResult(
        candidates=selected,
        attempts=attempts,
        closure_attempt_count=closure_attempts,
        oversized_closure_count=oversized,
        undersized_closure_count=undersized,
        exact_existing_duplicate_count=exact_existing,
        raw_candidate_count=len(rows),
        pareto_front_count=max(ranks.values(), default=-1) + 1,
    )


__all__ = [
    "CAUSALTOPOPOOL_ID",
    "CausalTopoPoolResult",
    "generate_causaltopopool_candidates",
]
