from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


V2_ANCHOR_ROLE = "v2_anchor"
COMPONENT_ROLE = "component16"
HOTSPOT_ROLE = "hotspot16"
ROLE_ORDER = (V2_ANCHOR_ROLE, COMPONENT_ROLE, HOTSPOT_ROLE)

ALL_SHARED = "all_shared"
CONSENSUS_STRUCTURAL = "consensus_structural"
ANCHOR_COMPONENT_SHARED = "anchor_component_shared"
ANCHOR_HOTSPOT_SHARED = "anchor_hotspot_shared"
THREE_UNIQUE = "three_unique"
PARTITIONS = (
    ALL_SHARED,
    CONSENSUS_STRUCTURAL,
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
    THREE_UNIQUE,
)


@dataclass(frozen=True)
class CanonicalAction:
    """One exact agent set with every role that proposed that set."""

    agents: tuple[int, ...]
    role_aliases: tuple[str, ...]
    candidate_ids_by_role: tuple[tuple[str, str], ...]

    @property
    def actual_size(self) -> int:
        return len(self.agents)

    @property
    def candidate_id(self) -> str:
        ids = {candidate_id for _role, candidate_id in self.candidate_ids_by_role}
        if len(ids) != 1:
            raise RuntimeError("canonical action aliases have inconsistent candidate IDs")
        return next(iter(ids))


@dataclass(frozen=True)
class CanonicalRoleActions:
    """Canonical V2/Component/Hotspot actions for one pre-action state."""

    partition: str
    actions: tuple[CanonicalAction, ...]
    role_to_action_index: tuple[tuple[str, int], ...]

    @property
    def unique_action_count(self) -> int:
        return len(self.actions)

    def action_index(self, role: str) -> int:
        for observed_role, index in self.role_to_action_index:
            if observed_role == role:
                return index
        raise KeyError(role)

    def action_for_role(self, role: str) -> CanonicalAction:
        return self.actions[self.action_index(role)]

    def agents_for_role(self, role: str) -> tuple[int, ...]:
        return self.action_for_role(role).agents


def _canonical_agents(
    agents: Any,
    *,
    label: str,
    agent_count: int | None = None,
) -> tuple[int, ...]:
    if not isinstance(agents, Sequence) or isinstance(agents, (str, bytes)):
        raise TypeError(f"{label} agents must be a sequence")
    values = tuple(agents)
    if not values:
        raise ValueError(f"{label} agents must be non-empty")
    if any(type(agent) is not int for agent in values):
        raise TypeError(f"{label} agents must contain only plain integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} agents must be unique")
    if agent_count is not None:
        if type(agent_count) is not int or agent_count <= 0:
            raise ValueError("agent_count must be a positive plain integer")
        if any(agent < 0 or agent >= agent_count for agent in values):
            raise ValueError(f"{label} agents fall outside agent_count")
    elif any(agent < 0 for agent in values):
        raise ValueError(f"{label} agents must be non-negative")
    return tuple(sorted(values))


def canonical_agent_tuple(
    candidate: Mapping[str, Any], *, agent_count: int | None = None
) -> tuple[int, ...]:
    """Validate a candidate and return its exact sorted agent tuple.

    Sorting makes ordering irrelevant; it deliberately does not use ``set`` so
    duplicate agent IDs remain an integrity error instead of being hidden.
    """

    if not isinstance(candidate, Mapping):
        raise TypeError("candidate must be a mapping")
    agents = _canonical_agents(
        candidate.get("agents"), label="candidate", agent_count=agent_count
    )
    if "actual_size" in candidate and (
        type(candidate["actual_size"]) is not int
        or int(candidate["actual_size"]) != len(agents)
    ):
        raise ValueError("candidate actual_size does not match its exact agent set")
    return agents


def classify_vch_partition(
    role_to_agents: Mapping[str, Sequence[int]],
    *,
    agent_count: int | None = None,
) -> str:
    """Classify the five possible equality partitions of V, C, and H."""

    if not isinstance(role_to_agents, Mapping):
        raise TypeError("role_to_agents must be a mapping")
    observed_roles = set(role_to_agents)
    expected_roles = set(ROLE_ORDER)
    if observed_roles != expected_roles:
        missing = sorted(expected_roles - observed_roles)
        extra = sorted(observed_roles - expected_roles)
        raise ValueError(f"V/C/H roles changed: missing={missing}, extra={extra}")
    normalized = {
        role: _canonical_agents(
            role_to_agents[role], label=role, agent_count=agent_count
        )
        for role in ROLE_ORDER
    }
    anchor = normalized[V2_ANCHOR_ROLE]
    component = normalized[COMPONENT_ROLE]
    hotspot = normalized[HOTSPOT_ROLE]
    if anchor == component == hotspot:
        return ALL_SHARED
    if component == hotspot:
        return CONSENSUS_STRUCTURAL
    if anchor == component:
        return ANCHOR_COMPONENT_SHARED
    if anchor == hotspot:
        return ANCHOR_HOTSPOT_SHARED
    return THREE_UNIQUE


def canonicalize_role_actions(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    agent_count: int | None = None,
    require_shared_candidate_id: bool = True,
) -> CanonicalRoleActions:
    """Merge V/C/H roles that name the same exact agent set.

    Role order, not mapping insertion order, determines the stable action and
    alias order. Candidate IDs are required because they are part of the
    proposal identity. By default, aliases for one exact set must also carry
    one deterministic candidate ID.
    """

    if not isinstance(arms, Mapping):
        raise TypeError("arms must be a mapping")
    observed_roles = set(arms)
    expected_roles = set(ROLE_ORDER)
    if observed_roles != expected_roles:
        missing = sorted(expected_roles - observed_roles)
        extra = sorted(observed_roles - expected_roles)
        raise ValueError(f"V/C/H arms changed: missing={missing}, extra={extra}")

    agents_by_role: dict[str, tuple[int, ...]] = {}
    candidate_ids: dict[str, str] = {}
    for role in ROLE_ORDER:
        candidate = arms[role]
        if not isinstance(candidate, Mapping):
            raise TypeError(f"{role} candidate must be a mapping")
        agents_by_role[role] = canonical_agent_tuple(
            candidate, agent_count=agent_count
        )
        declared_role = candidate.get("role")
        if declared_role is not None and declared_role != role:
            raise ValueError(f"{role} candidate declares role {declared_role!r}")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{role} candidate_id must be a non-empty string")
        candidate_ids[role] = candidate_id

    grouped_roles: dict[tuple[int, ...], list[str]] = {}
    for role in ROLE_ORDER:
        grouped_roles.setdefault(agents_by_role[role], []).append(role)

    actions: list[CanonicalAction] = []
    role_to_index: list[tuple[str, int]] = []
    for agents, aliases in grouped_roles.items():
        ids = {candidate_ids[role] for role in aliases}
        if require_shared_candidate_id and len(ids) != 1:
            raise ValueError(
                "roles sharing one exact agent set have different candidate IDs"
            )
        action_index = len(actions)
        actions.append(
            CanonicalAction(
                agents=agents,
                role_aliases=tuple(aliases),
                candidate_ids_by_role=tuple(
                    (role, candidate_ids[role]) for role in aliases
                ),
            )
        )
        role_to_index.extend((role, action_index) for role in aliases)
    role_to_index.sort(key=lambda item: ROLE_ORDER.index(item[0]))

    partition = classify_vch_partition(agents_by_role, agent_count=agent_count)
    expected_action_counts = {
        ALL_SHARED: 1,
        CONSENSUS_STRUCTURAL: 2,
        ANCHOR_COMPONENT_SHARED: 2,
        ANCHOR_HOTSPOT_SHARED: 2,
        THREE_UNIQUE: 3,
    }
    if len(actions) != expected_action_counts[partition]:
        raise RuntimeError("canonical V/C/H partition and action count disagree")
    return CanonicalRoleActions(
        partition=partition,
        actions=tuple(actions),
        role_to_action_index=tuple(role_to_index),
    )


def pairwise_set_features(
    left_agents: Sequence[int],
    right_agents: Sequence[int],
    *,
    prefix: str,
    agent_count: int | None = None,
) -> dict[str, float]:
    """Return deterministic, outcome-free relationship features for two sets."""

    if not isinstance(prefix, str) or not prefix:
        raise ValueError("pairwise set feature prefix must be non-empty")
    left = _canonical_agents(
        left_agents, label=f"{prefix}.left", agent_count=agent_count
    )
    right = _canonical_agents(
        right_agents, label=f"{prefix}.right", agent_count=agent_count
    )
    left_set = set(left)
    right_set = set(right)
    intersection = left_set & right_set
    union = left_set | right_set
    left_only = left_set - right_set
    right_only = right_set - left_set
    symmetric_difference = left_set ^ right_set
    return {
        f"{prefix}.exact_equal": float(left == right),
        f"{prefix}.left_size": float(len(left)),
        f"{prefix}.right_size": float(len(right)),
        f"{prefix}.size_delta": float(len(left) - len(right)),
        f"{prefix}.absolute_size_delta": float(abs(len(left) - len(right))),
        f"{prefix}.intersection_size": float(len(intersection)),
        f"{prefix}.union_size": float(len(union)),
        f"{prefix}.left_only_size": float(len(left_only)),
        f"{prefix}.right_only_size": float(len(right_only)),
        f"{prefix}.symmetric_difference_size": float(len(symmetric_difference)),
        f"{prefix}.jaccard": len(intersection) / len(union),
        f"{prefix}.overlap_left": len(intersection) / len(left),
        f"{prefix}.overlap_right": len(intersection) / len(right),
        f"{prefix}.left_subset_right": float(left_set <= right_set),
        f"{prefix}.right_subset_left": float(right_set <= left_set),
    }


def vch_relational_features(
    canonical: CanonicalRoleActions,
) -> dict[str, float]:
    """Build exact-set features shared by the two hierarchical model stages."""

    if not isinstance(canonical, CanonicalRoleActions):
        raise TypeError("canonical must be CanonicalRoleActions")
    if canonical.partition not in PARTITIONS:
        raise ValueError("unknown canonical V/C/H partition")
    anchor = canonical.agents_for_role(V2_ANCHOR_ROLE)
    component = canonical.agents_for_role(COMPONENT_ROLE)
    hotspot = canonical.agents_for_role(HOTSPOT_ROLE)
    result = {
        "set.unique_action_count": float(canonical.unique_action_count),
        **{
            f"set.partition={partition}": float(canonical.partition == partition)
            for partition in PARTITIONS
        },
    }
    result.update(
        pairwise_set_features(anchor, component, prefix="set.v2_component")
    )
    result.update(pairwise_set_features(anchor, hotspot, prefix="set.v2_hotspot"))
    result.update(
        pairwise_set_features(component, hotspot, prefix="set.component_hotspot")
    )
    return result


__all__ = [
    "ALL_SHARED",
    "ANCHOR_COMPONENT_SHARED",
    "ANCHOR_HOTSPOT_SHARED",
    "COMPONENT_ROLE",
    "CONSENSUS_STRUCTURAL",
    "CanonicalAction",
    "CanonicalRoleActions",
    "HOTSPOT_ROLE",
    "PARTITIONS",
    "ROLE_ORDER",
    "THREE_UNIQUE",
    "V2_ANCHOR_ROLE",
    "canonical_agent_tuple",
    "canonicalize_role_actions",
    "classify_vch_partition",
    "pairwise_set_features",
    "vch_relational_features",
]
