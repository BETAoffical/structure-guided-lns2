from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_subset,
)
from lns2_selector.runtime.v2_first_single_family_rescue import (
    V2FirstSingleFamilyRescueTracker,
)


V2_FIRST_CONSENSUS16_RESCUE_POOL_ID = (
    "stride-v2-first-consensus16-rescue-v1"
)
V2_FIRST_CONSENSUS16_RESCUE_RUNTIME_ID = (
    "stride-v2-first-consensus16-rescue-runtime-v1"
)
V2_FIRST_CONSENSUS16_RESCUE_TRACKER_ID = (
    "stride-v2-first-consensus16-rescue-tracker-v1"
)

_COMPONENT_VARIANT = "conflict_component"
_HOTSPOT_VARIANT = "spatiotemporal_hotspot"
_COMPONENT_FAMILY = "structpool-conflict-component:16"
_HOTSPOT_FAMILY = "structpool-spatiotemporal-hotspot:16"


def v2_first_consensus_rescue_augmentation(
    nominal_size: int = 16,
    minimum_consecutive_v2_exact_rollbacks: int = 3,
) -> dict[str, Any]:
    """Return the frozen one-shot Component16/Hotspot16 consensus contract."""

    if type(nominal_size) is not int or nominal_size != 16:
        raise ValueError("V2-first consensus rescue nominal size must be exactly 16")
    if (
        type(minimum_consecutive_v2_exact_rollbacks) is not int
        or minimum_consecutive_v2_exact_rollbacks != 3
    ):
        raise ValueError("V2-first consensus rescue requires exactly three rollbacks")

    result = {
        "enabled": True,
        "pool_id": V2_FIRST_CONSENSUS16_RESCUE_POOL_ID,
        "runtime_id": V2_FIRST_CONSENSUS16_RESCUE_RUNTIME_ID,
        "full_union_required": False,
        "full_union_audit_preserved": True,
        "runtime_filter_id": "v2_first_exact_agent_consensus_rescue_v1",
        "source_mode": "v2_first_consensus16_rescue",
        "structural_profile": "component_hotspot_consensus",
        "nominal_size": nominal_size,
        "runtime_structural_family_sizes": {
            _COMPONENT_VARIANT: [nominal_size],
            _HOTSPOT_VARIANT: [nominal_size],
        },
        "maximum_generated_candidates": 2,
        "maximum_added_candidates": 1,
        "maximum_total_candidates": 1,
        "static_grid_cache": True,
        "activation_gate": {
            "gate_id": "v2_first_three_exact_rollbacks_v1",
        },
        "v2_first_rescue": {
            "tracker_id": V2_FIRST_CONSENSUS16_RESCUE_TRACKER_ID,
            "minimum_consecutive_v2_exact_rollbacks": (
                minimum_consecutive_v2_exact_rollbacks
            ),
            "maximum_rescue_offers_per_episode": 1,
            "maximum_executed_rescues_per_episode": 1,
            "maximum_generation_attempts_per_repair_fingerprint": 1,
            "selection": "direct_only_on_exact_nonempty_agent_consensus",
            "agreement": "sorted_agent_set_exact_equality",
            "fallback_if_no_consensus": "fresh_v2_only_same_decision",
            "permanent_v2_after_offer": True,
            "maximum_pp_calls_per_decision": 1,
            "same_decision_retry": False,
            "repairer": "PP",
        },
    }
    return json.loads(json.dumps(result))


def validate_v2_first_consensus_rescue_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    if result != v2_first_consensus_rescue_augmentation():
        raise ValueError("unsupported V2-first consensus16 rescue augmentation")
    return result


@dataclass
class V2FirstConsensusRescueTracker:
    """Reuse the frozen V2-first trigger while recording consensus semantics."""

    minimum_consecutive_rollbacks: int = 3
    _delegate: V2FirstSingleFamilyRescueTracker = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.minimum_consecutive_rollbacks != 3:
            raise ValueError("V2-first consensus rescue requires exactly three rollbacks")
        self._delegate = V2FirstSingleFamilyRescueTracker(
            structural_profile="conflict_component",
            minimum_consecutive_rollbacks=self.minimum_consecutive_rollbacks,
        )

    @classmethod
    def from_spec(
        cls, specification: Mapping[str, Any]
    ) -> "V2FirstConsensusRescueTracker":
        config = validate_v2_first_consensus_rescue_augmentation(
            dict(specification)
        )
        assert config is not None
        tracker = dict(config["v2_first_rescue"])
        return cls(
            minimum_consecutive_rollbacks=int(
                tracker["minimum_consecutive_v2_exact_rollbacks"]
            )
        )

    def selection(self, repair_fingerprint: str) -> dict[str, Any]:
        result = dict(self._delegate.selection(repair_fingerprint))
        phase = str(result["selection_phase"])
        if phase == "single_family_rescue_due":
            phase = "consensus_rescue_due"
        result.update(
            {
                "tracker_id": V2_FIRST_CONSENSUS16_RESCUE_TRACKER_ID,
                "selection_phase": phase,
                "structural_profile": "component_hotspot_consensus",
                "consensus_required": True,
            }
        )
        return result

    def record_generation(
        self,
        *,
        repair_fingerprint: str,
        candidate_id: str | None,
        consensus_audit: Mapping[str, Any],
    ) -> dict[str, Any]:
        audit = dict(consensus_audit)
        exact_consensus = bool(audit.get("exact_agent_consensus"))
        if exact_consensus != (candidate_id is not None):
            raise ValueError("consensus audit and direct candidate disagree")
        result = dict(
            self._delegate.record_generation(
                repair_fingerprint=repair_fingerprint,
                candidate_id=candidate_id,
            )
        )
        result.update(audit)
        result["fallback"] = (
            None
            if candidate_id is not None
            else "fresh_v2_after_no_consensus"
        )
        return result

    def mark_executed(
        self,
        *,
        repair_fingerprint: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        return self._delegate.mark_executed(
            repair_fingerprint=repair_fingerprint,
            candidate_id=candidate_id,
        )

    def observe_v2(
        self,
        *,
        before_repair_fingerprint: str,
        after_repair_fingerprint: str,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._delegate.observe_v2(
            before_repair_fingerprint=before_repair_fingerprint,
            after_repair_fingerprint=after_repair_fingerprint,
            metrics=metrics,
        )

    def observe_rescue(
        self,
        *,
        before_repair_fingerprint: str,
        after_repair_fingerprint: str,
        conflicts_before: int,
        conflicts_after: int,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        result = dict(
            self._delegate.observe_rescue(
                before_repair_fingerprint=before_repair_fingerprint,
                after_repair_fingerprint=after_repair_fingerprint,
                conflicts_before=conflicts_before,
                conflicts_after=conflicts_after,
                metrics=metrics,
            )
        )
        result["decision_mode"] = "consensus_rescue"
        return result


def _candidate_for_family(
    candidates: list[dict[str, Any]], family: str
) -> dict[str, Any] | None:
    matches = [
        candidate
        for candidate in candidates
        if family in set(map(str, candidate.get("selection_families", [])))
    ]
    if len(matches) > 1:
        raise RuntimeError(f"consensus rescue generated multiple {family} candidates")
    return None if not matches else matches[0]


def _normalized_agents(candidate: dict[str, Any] | None) -> tuple[int, ...]:
    if candidate is None:
        return ()
    agents = tuple(sorted(map(int, candidate.get("agents", []))))
    if len(agents) != len(set(agents)):
        raise RuntimeError("consensus rescue candidate contains duplicate agents")
    return agents


def generate_v2_first_consensus_rescue_candidate(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, float, dict[str, Any]]:
    """Generate both fixed16 families and retain only exact agent consensus."""

    specification = validate_v2_first_consensus_rescue_augmentation(config)
    assert specification is not None
    family_sizes = {
        str(family): tuple(map(int, sizes))
        for family, sizes in dict(
            specification["runtime_structural_family_sizes"]
        ).items()
    }
    started = time.perf_counter()
    candidates = generate_structpool_candidate_subset(
        state,
        analysis,
        family_sizes=family_sizes,
    )
    generation_seconds = time.perf_counter() - started
    if len(candidates) > 2:
        raise RuntimeError("consensus rescue generated more than two candidates")

    component = _candidate_for_family(candidates, _COMPONENT_FAMILY)
    hotspot = _candidate_for_family(candidates, _HOTSPOT_FAMILY)
    component_agents = _normalized_agents(component)
    hotspot_agents = _normalized_agents(hotspot)
    exact_consensus = bool(
        component_agents
        and hotspot_agents
        and component_agents == hotspot_agents
    )
    if exact_consensus and (
        str(component["candidate_id"]) != str(hotspot["candidate_id"])
    ):
        raise RuntimeError("exact consensus candidates have different IDs")

    consensus_candidate = None
    if exact_consensus:
        consensus_candidate = dict(component)
        consensus_candidate["agents"] = list(component_agents)
        consensus_candidate["actual_size"] = len(component_agents)
        consensus_candidate["hybridstructpool_provenance"] = [
            "structshell_equal_four_size",
            "v2_first_consensus_rescue",
        ]

    audit = {
        "component_candidate_id": (
            None if component is None else str(component["candidate_id"])
        ),
        "hotspot_candidate_id": (
            None if hotspot is None else str(hotspot["candidate_id"])
        ),
        "component_agents": list(component_agents),
        "hotspot_agents": list(hotspot_agents),
        "component_available": component is not None,
        "hotspot_available": hotspot is not None,
        "exact_agent_consensus": exact_consensus,
        "consensus_candidate_id": (
            None
            if consensus_candidate is None
            else str(consensus_candidate["candidate_id"])
        ),
        "generated_candidate_count": len(candidates),
    }
    return consensus_candidate, generation_seconds, audit


__all__ = [
    "V2_FIRST_CONSENSUS16_RESCUE_POOL_ID",
    "V2_FIRST_CONSENSUS16_RESCUE_RUNTIME_ID",
    "V2_FIRST_CONSENSUS16_RESCUE_TRACKER_ID",
    "V2FirstConsensusRescueTracker",
    "generate_v2_first_consensus_rescue_candidate",
    "v2_first_consensus_rescue_augmentation",
    "validate_v2_first_consensus_rescue_augmentation",
]
