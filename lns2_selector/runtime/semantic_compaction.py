from __future__ import annotations

from typing import Any

from experiments.state_analysis import analyze_state
from lns2_selector.runtime.repairdependencypool import (
    _serialize_evidence,
    _temporal_corridor_evidence,
)


COMPACTION_RULE_ID = "current-conflict-plus-temporal-corridor-v1"


def semantic_compact_plan(
    state: dict[str, Any], base_agents: list[int]
) -> dict[str, Any]:
    """Remove only members without current-conflict or corridor evidence.

    The rule is deterministic, uses only the state available before the rescue
    action, and requests no target neighborhood size.
    """

    base_order = list(map(int, base_agents))
    if len(base_order) != len(set(base_order)):
        raise ValueError("semantic compaction base contains duplicate agents")
    base = set(base_order)
    current_core = {
        int(agent)
        for edge in state.get("conflict_edges", ())
        for agent in edge
        if int(agent) in base
    }
    if len(current_core) < 2:
        return {
            "rule_id": COMPACTION_RULE_ID,
            "eligible": False,
            "rejection_reason": "fewer_than_two_current_conflict_agents",
            "base_agents": sorted(base),
            "compact_agents": sorted(base),
            "current_conflict_core": sorted(current_core),
            "temporal_corridor_support": [],
            "temporal_corridor_evidence": {},
            "removed_agents": [],
            "base_size": len(base),
            "actual_size": len(base),
            "fixed_target_size": None,
        }
    analysis = analyze_state(state)
    temporal = _temporal_corridor_evidence(state, analysis, current_core)
    retained_temporal = sorted(base & set(temporal))
    compact = current_core | set(retained_temporal)
    removed = sorted(base - compact)
    eligible = bool(removed)
    return {
        "rule_id": COMPACTION_RULE_ID,
        "eligible": eligible,
        "rejection_reason": None if eligible else "no_unsupported_agent_to_remove",
        "base_agents": sorted(base),
        "compact_agents": sorted(compact) if eligible else sorted(base),
        "current_conflict_core": sorted(current_core),
        "temporal_corridor_support": retained_temporal,
        "temporal_corridor_evidence": {
            str(agent): _serialize_evidence(temporal[agent])
            for agent in retained_temporal
        },
        "removed_agents": removed,
        "base_size": len(base),
        "actual_size": len(compact) if eligible else len(base),
        "fixed_target_size": None,
    }


__all__ = ["COMPACTION_RULE_ID", "semantic_compact_plan"]
