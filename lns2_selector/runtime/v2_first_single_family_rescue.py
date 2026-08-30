from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.rollback_aware_selection import (
    is_exact_conflict_bound_rollback,
)
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_subset,
)


V2_FIRST_SINGLE_FAMILY_RESCUE_POOL_ID = (
    "stride-v2-first-single-family-rescue-v1"
)
V2_FIRST_SINGLE_FAMILY_RESCUE_RUNTIME_ID = (
    "stride-v2-first-single-family-rescue-runtime-v1"
)
V2_FIRST_SINGLE_FAMILY_RESCUE_TRACKER_ID = (
    "stride-v2-first-single-family-rescue-tracker-v1"
)
V2_FIRST_RESCUE_PROFILES = ("conflict_component", "hotspot")

_PROFILE_VARIANTS = {
    "conflict_component": "conflict_component",
    "hotspot": "spatiotemporal_hotspot",
}


def v2_first_single_family_rescue_augmentation(
    profile: str,
    nominal_size: int = 16,
) -> dict[str, Any]:
    """Return the frozen V2-first, one-shot structural rescue contract."""

    normalized_profile = str(profile)
    if normalized_profile not in V2_FIRST_RESCUE_PROFILES:
        raise ValueError(
            f"unsupported V2-first rescue profile: {normalized_profile}"
        )
    if type(nominal_size) is not int or nominal_size != 16:
        raise ValueError("V2-first rescue nominal size must be exactly 16")
    variant = _PROFILE_VARIANTS[normalized_profile]
    result = {
        "enabled": True,
        "pool_id": V2_FIRST_SINGLE_FAMILY_RESCUE_POOL_ID,
        "runtime_id": V2_FIRST_SINGLE_FAMILY_RESCUE_RUNTIME_ID,
        "full_union_required": False,
        "full_union_audit_preserved": True,
        "runtime_filter_id": "v2_first_exact_rollback_rescue_v1",
        "source_mode": "v2_first_single_family_rescue",
        "structural_profile": normalized_profile,
        "nominal_size": nominal_size,
        "runtime_structural_family_sizes": {variant: [nominal_size]},
        "maximum_added_candidates": 1,
        "maximum_total_candidates": 1,
        "static_grid_cache": True,
        "activation_gate": {
            "gate_id": "v2_first_three_exact_rollbacks_v1",
        },
        "v2_first_rescue": {
            "tracker_id": V2_FIRST_SINGLE_FAMILY_RESCUE_TRACKER_ID,
            "minimum_consecutive_v2_exact_rollbacks": 3,
            "maximum_rescue_offers_per_episode": 1,
            "maximum_executed_rescues_per_episode": 1,
            "maximum_generation_attempts_per_repair_fingerprint": 1,
            "selection": "direct_unique_structural_candidate",
            "fallback_if_unavailable": "fresh_v2_only",
            "permanent_v2_after_offer": True,
            "maximum_pp_calls_per_decision": 1,
            "same_decision_retry": False,
            "repairer": "PP",
        },
    }
    return json.loads(json.dumps(result))


def validate_v2_first_single_family_rescue_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    profile = str(result.get("structural_profile") or "")
    nominal_size = result.get("nominal_size")
    if type(nominal_size) is not int:
        raise ValueError("unsupported V2-first rescue augmentation")
    if result != v2_first_single_family_rescue_augmentation(
        profile, nominal_size
    ):
        raise ValueError("unsupported V2-first rescue augmentation")
    return result


@dataclass
class V2FirstSingleFamilyRescueTracker:
    """Schedule one family-isolated rescue after three exact V2 rollbacks.

    The trigger is deliberately scoped to one repair fingerprint.  Offering
    the rescue consumes the episode-global opportunity whether or not the
    family has a candidate; every later decision is therefore V2-only.  This
    prevents unavailable families or failed repairs from reopening the branch.
    """

    structural_profile: str
    minimum_consecutive_rollbacks: int = 3
    consecutive_fingerprint: str | None = None
    consecutive_v2_exact_rollbacks: int = 0
    rescue_used: bool = False
    rescue_executed_fingerprint: str | None = None
    _attempted_fingerprints: set[str] = field(
        default_factory=set,
        init=False,
        repr=False,
    )
    _available_candidate_by_fingerprint: dict[str, str] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.structural_profile not in V2_FIRST_RESCUE_PROFILES:
            raise ValueError("unsupported V2-first rescue profile")
        if self.minimum_consecutive_rollbacks != 3:
            raise ValueError("V2-first rescue requires exactly three rollbacks")

    @classmethod
    def from_spec(
        cls, specification: Mapping[str, Any]
    ) -> "V2FirstSingleFamilyRescueTracker":
        config = validate_v2_first_single_family_rescue_augmentation(
            dict(specification)
        )
        assert config is not None
        tracker = dict(config["v2_first_rescue"])
        return cls(
            structural_profile=str(config["structural_profile"]),
            minimum_consecutive_rollbacks=int(
                tracker["minimum_consecutive_v2_exact_rollbacks"]
            ),
        )

    def selection(self, repair_fingerprint: str) -> dict[str, Any]:
        fingerprint = str(repair_fingerprint)
        due = bool(
            not self.rescue_used
            and fingerprint not in self._attempted_fingerprints
            and self.consecutive_fingerprint == fingerprint
            and self.consecutive_v2_exact_rollbacks
            >= self.minimum_consecutive_rollbacks
        )
        if self.rescue_used:
            phase = "v2_only_after_offer"
        elif fingerprint in self._attempted_fingerprints:
            phase = "fresh_v2_after_unavailable"
        elif due:
            phase = "single_family_rescue_due"
        else:
            phase = "v2_only"
        return {
            "tracker_id": V2_FIRST_SINGLE_FAMILY_RESCUE_TRACKER_ID,
            "repair_fingerprint": fingerprint,
            "selection_phase": phase,
            "rescue_due": due,
            "offered": due,
            "structural_profile": self.structural_profile,
            "nominal_size": 16,
            "consecutive_v2_exact_rollbacks": (
                self.consecutive_v2_exact_rollbacks
                if self.consecutive_fingerprint == fingerprint
                else 0
            ),
            "minimum_consecutive_v2_exact_rollbacks": (
                self.minimum_consecutive_rollbacks
            ),
            "generation_already_attempted_at_fingerprint": (
                fingerprint in self._attempted_fingerprints
            ),
            "rescue_used": self.rescue_used,
            "consumed": self.rescue_used,
            "rescue_executed_fingerprint": self.rescue_executed_fingerprint,
        }

    def record_generation(
        self,
        *,
        repair_fingerprint: str,
        candidate_id: str | None,
    ) -> dict[str, Any]:
        fingerprint = str(repair_fingerprint)
        if not self.selection(fingerprint)["rescue_due"]:
            raise ValueError("V2-first rescue generation was not due")
        if fingerprint in self._attempted_fingerprints:
            raise ValueError("V2-first rescue generation already attempted")
        self._attempted_fingerprints.add(fingerprint)
        normalized_candidate = (
            None if candidate_id is None else str(candidate_id)
        )
        if normalized_candidate is not None:
            if not normalized_candidate:
                raise ValueError("V2-first rescue candidate ID must be non-empty")
            self._available_candidate_by_fingerprint[
                fingerprint
            ] = normalized_candidate
        self.rescue_used = True
        return {
            "attempted": True,
            "available": normalized_candidate is not None,
            "offered": True,
            "challenger_present": normalized_candidate is not None,
            "structural_selected": False,
            "consumed": True,
            "repair_fingerprint": fingerprint,
            "candidate_id": normalized_candidate,
            "fallback": (
                None
                if normalized_candidate is not None
                else "fresh_v2_only"
            ),
            "episode_rescue_consumed": True,
        }

    def mark_executed(
        self,
        *,
        repair_fingerprint: str,
        candidate_id: str,
    ) -> dict[str, Any]:
        fingerprint = str(repair_fingerprint)
        normalized_candidate = str(candidate_id)
        if not self.rescue_used:
            raise ValueError("V2-first rescue execution was not offered")
        if self.rescue_executed_fingerprint is not None:
            raise ValueError("V2-first rescue was already executed")
        if self._available_candidate_by_fingerprint.get(fingerprint) != (
            normalized_candidate
        ):
            raise ValueError("V2-first rescue execution was not generated")
        self.rescue_used = True
        self.rescue_executed_fingerprint = fingerprint
        return {
            "executed": True,
            "repair_fingerprint": fingerprint,
            "candidate_id": normalized_candidate,
            "offered": True,
            "challenger_present": True,
            "structural_selected": True,
            "consumed": True,
            "episode_rescue_consumed": True,
        }

    def observe_v2(
        self,
        *,
        before_repair_fingerprint: str,
        after_repair_fingerprint: str,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        before = str(before_repair_fingerprint)
        after = str(after_repair_fingerprint)
        exact_rollback = is_exact_conflict_bound_rollback(
            metrics,
            before_repair_fingerprint=before,
            after_repair_fingerprint=after,
        )
        if not self.rescue_used and exact_rollback:
            if self.consecutive_fingerprint == before:
                self.consecutive_v2_exact_rollbacks += 1
            else:
                self.consecutive_fingerprint = before
                self.consecutive_v2_exact_rollbacks = 1
        elif not self.rescue_used:
            self.consecutive_fingerprint = after
            self.consecutive_v2_exact_rollbacks = 0
        scheduled = bool(self.selection(after)["rescue_due"])
        return {
            "decision_mode": "v2_only",
            "before_repair_fingerprint": before,
            "after_repair_fingerprint": after,
            "v2_exact_conflict_bound_rollback": exact_rollback,
            "consecutive_v2_exact_rollbacks": (
                self.consecutive_v2_exact_rollbacks
                if self.consecutive_fingerprint == after
                else 0
            ),
            "rescue_scheduled_for_next_decision": scheduled,
            "rescue_used": self.rescue_used,
        }

    def observe_rescue(
        self,
        *,
        before_repair_fingerprint: str,
        after_repair_fingerprint: str,
        conflicts_before: int,
        conflicts_after: int,
        metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        before = str(before_repair_fingerprint)
        after = str(after_repair_fingerprint)
        exact_rollback = is_exact_conflict_bound_rollback(
            metrics,
            before_repair_fingerprint=before,
            after_repair_fingerprint=after,
        )
        strict_drop = int(conflicts_after) < int(conflicts_before)
        return {
            "decision_mode": "single_family_rescue",
            "before_repair_fingerprint": before,
            "after_repair_fingerprint": after,
            "exact_conflict_bound_rollback": exact_rollback,
            "strict_conflict_drop": strict_drop,
            "conflict_reduction": max(
                0, int(conflicts_before) - int(conflicts_after)
            ),
            "rescue_used": self.rescue_used,
            "future_selection_phase": "v2_only_after_offer",
        }


def generate_v2_first_single_family_rescue_candidate(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, float]:
    """Generate zero or one isolated structural candidate without a V2 pool."""

    specification = validate_v2_first_single_family_rescue_augmentation(config)
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
    if len(candidates) > 1:
        raise RuntimeError("V2-first rescue generated more than one candidate")
    if not candidates:
        return None, generation_seconds
    candidate = dict(candidates[0])
    candidate["hybridstructpool_provenance"] = [
        "structshell_equal_four_size",
        "v2_first_single_family_rescue",
    ]
    return candidate, generation_seconds


__all__ = [
    "V2_FIRST_RESCUE_PROFILES",
    "V2_FIRST_SINGLE_FAMILY_RESCUE_POOL_ID",
    "V2_FIRST_SINGLE_FAMILY_RESCUE_RUNTIME_ID",
    "V2_FIRST_SINGLE_FAMILY_RESCUE_TRACKER_ID",
    "V2FirstSingleFamilyRescueTracker",
    "generate_v2_first_single_family_rescue_candidate",
    "v2_first_single_family_rescue_augmentation",
    "validate_v2_first_single_family_rescue_augmentation",
]
