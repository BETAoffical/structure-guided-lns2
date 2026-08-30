from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from lns2_selector.runtime.unique_action_hierarchy import (
    ALL_SHARED,
    ANCHOR_COMPONENT_SHARED,
    ANCHOR_HOTSPOT_SHARED,
    COMPONENT_ROLE,
    CONSENSUS_STRUCTURAL,
    HOTSPOT_ROLE,
    ROLE_ORDER,
    THREE_UNIQUE,
    V2_ANCHOR_ROLE,
    canonical_agent_tuple,
    canonicalize_role_actions,
    classify_vch_partition,
    pairwise_set_features,
    vch_relational_features,
)


def _candidate(role: str, agents: list[int], candidate_id: str | None = None) -> dict:
    normalized = tuple(sorted(agents))
    return {
        "role": role,
        "candidate_id": candidate_id or "candidate-" + "-".join(map(str, normalized)),
        "agents": list(agents),
        "actual_size": len(agents),
    }


def _arms(
    anchor: list[int], component: list[int], hotspot: list[int]
) -> dict[str, dict]:
    return {
        V2_ANCHOR_ROLE: _candidate(V2_ANCHOR_ROLE, anchor),
        COMPONENT_ROLE: _candidate(COMPONENT_ROLE, component),
        HOTSPOT_ROLE: _candidate(HOTSPOT_ROLE, hotspot),
    }


@pytest.mark.parametrize(
    ("anchor", "component", "hotspot", "partition", "action_count", "aliases"),
    (
        (
            [0, 1],
            [1, 0],
            [0, 1],
            ALL_SHARED,
            1,
            ((V2_ANCHOR_ROLE, COMPONENT_ROLE, HOTSPOT_ROLE),),
        ),
        (
            [0, 1],
            [2, 3],
            [3, 2],
            CONSENSUS_STRUCTURAL,
            2,
            ((V2_ANCHOR_ROLE,), (COMPONENT_ROLE, HOTSPOT_ROLE)),
        ),
        (
            [0, 1],
            [1, 0],
            [2, 3],
            ANCHOR_COMPONENT_SHARED,
            2,
            ((V2_ANCHOR_ROLE, COMPONENT_ROLE), (HOTSPOT_ROLE,)),
        ),
        (
            [0, 1],
            [2, 3],
            [1, 0],
            ANCHOR_HOTSPOT_SHARED,
            2,
            ((V2_ANCHOR_ROLE, HOTSPOT_ROLE), (COMPONENT_ROLE,)),
        ),
        (
            [0, 1],
            [2, 3],
            [4, 5],
            THREE_UNIQUE,
            3,
            ((V2_ANCHOR_ROLE,), (COMPONENT_ROLE,), (HOTSPOT_ROLE,)),
        ),
    ),
)
def test_all_vch_equality_partitions_are_canonicalized_once(
    anchor: list[int],
    component: list[int],
    hotspot: list[int],
    partition: str,
    action_count: int,
    aliases: tuple[tuple[str, ...], ...],
) -> None:
    arms = _arms(anchor, component, hotspot)
    # Equal exact sets must carry the same deterministic candidate ID.
    for left_role, right_role in (
        (V2_ANCHOR_ROLE, COMPONENT_ROLE),
        (V2_ANCHOR_ROLE, HOTSPOT_ROLE),
        (COMPONENT_ROLE, HOTSPOT_ROLE),
    ):
        if sorted(arms[left_role]["agents"]) == sorted(arms[right_role]["agents"]):
            arms[right_role]["candidate_id"] = arms[left_role]["candidate_id"]

    canonical = canonicalize_role_actions(arms, agent_count=6)

    assert canonical.partition == partition
    assert canonical.unique_action_count == action_count
    assert tuple(action.role_aliases for action in canonical.actions) == aliases
    assert classify_vch_partition(
        {role: arms[role]["agents"] for role in ROLE_ORDER}, agent_count=6
    ) == partition
    for role in ROLE_ORDER:
        assert canonical.agents_for_role(role) == tuple(sorted(arms[role]["agents"]))


def test_canonicalization_is_stable_under_arm_and_agent_order() -> None:
    arms = _arms([2, 0, 1], [5, 4, 3], [3, 5, 4])
    arms[HOTSPOT_ROLE]["candidate_id"] = arms[COMPONENT_ROLE]["candidate_id"]
    reversed_arms = {
        role: {**arms[role], "agents": list(reversed(arms[role]["agents"]))}
        for role in reversed(ROLE_ORDER)
    }

    left = canonicalize_role_actions(arms, agent_count=6)
    right = canonicalize_role_actions(reversed_arms, agent_count=6)

    assert left == right
    assert left.actions[0].agents == (0, 1, 2)
    assert left.actions[1].agents == (3, 4, 5)
    assert left.role_to_action_index == (
        (V2_ANCHOR_ROLE, 0),
        (COMPONENT_ROLE, 1),
        (HOTSPOT_ROLE, 1),
    )
    with pytest.raises(FrozenInstanceError):
        left.partition = THREE_UNIQUE  # type: ignore[misc]


@pytest.mark.parametrize(
    ("candidate", "error"),
    (
        ({"agents": []}, ValueError),
        ({"agents": [0, 0]}, ValueError),
        ({"agents": [0, "1"]}, TypeError),
        ({"agents": [0, True]}, TypeError),
        ({"agents": [-1, 0]}, ValueError),
        ({"agents": [0, 1], "actual_size": 3}, ValueError),
    ),
)
def test_canonical_agent_tuple_rejects_illegal_sets(
    candidate: dict, error: type[Exception]
) -> None:
    with pytest.raises(error):
        canonical_agent_tuple(candidate, agent_count=2)


def test_canonicalization_rejects_role_and_identity_drift() -> None:
    missing = _arms([0], [1], [2])
    missing.pop(HOTSPOT_ROLE)
    with pytest.raises(ValueError, match="roles changed|arms changed"):
        canonicalize_role_actions(missing)

    wrong_role = _arms([0], [1], [2])
    wrong_role[COMPONENT_ROLE]["role"] = HOTSPOT_ROLE
    with pytest.raises(ValueError, match="declares role"):
        canonicalize_role_actions(wrong_role)

    missing_id = _arms([0], [1], [2])
    missing_id[COMPONENT_ROLE]["candidate_id"] = ""
    with pytest.raises(ValueError, match="candidate_id"):
        canonicalize_role_actions(missing_id)

    alias_id_drift = _arms([0], [1, 2], [2, 1])
    alias_id_drift[HOTSPOT_ROLE]["candidate_id"] = "different-hotspot-id"
    with pytest.raises(ValueError, match="different candidate IDs"):
        canonicalize_role_actions(alias_id_drift)

    out_of_range = _arms([0], [1], [3])
    with pytest.raises(ValueError, match="outside agent_count"):
        canonicalize_role_actions(out_of_range, agent_count=3)


def test_pairwise_set_features_are_exact_and_directional() -> None:
    features = pairwise_set_features(
        [3, 1, 2], [5, 4, 3, 2], prefix="set.test"
    )

    assert features == {
        "set.test.exact_equal": 0.0,
        "set.test.left_size": 3.0,
        "set.test.right_size": 4.0,
        "set.test.size_delta": -1.0,
        "set.test.absolute_size_delta": 1.0,
        "set.test.intersection_size": 2.0,
        "set.test.union_size": 5.0,
        "set.test.left_only_size": 1.0,
        "set.test.right_only_size": 2.0,
        "set.test.symmetric_difference_size": 3.0,
        "set.test.jaccard": pytest.approx(0.4),
        "set.test.overlap_left": pytest.approx(2.0 / 3.0),
        "set.test.overlap_right": 0.5,
        "set.test.left_subset_right": 0.0,
        "set.test.right_subset_left": 0.0,
    }


def test_vch_relational_features_encode_partition_and_exact_aliases() -> None:
    arms = _arms([0, 1], [2, 3], [3, 2])
    arms[HOTSPOT_ROLE]["candidate_id"] = arms[COMPONENT_ROLE]["candidate_id"]
    canonical = canonicalize_role_actions(arms)

    features = vch_relational_features(canonical)

    assert features["set.unique_action_count"] == 2.0
    assert features[f"set.partition={CONSENSUS_STRUCTURAL}"] == 1.0
    assert sum(
        features[key] for key in features if key.startswith("set.partition=")
    ) == 1.0
    assert features["set.component_hotspot.exact_equal"] == 1.0
    assert features["set.component_hotspot.jaccard"] == 1.0
    assert features["set.v2_component.exact_equal"] == 0.0
    assert features["set.v2_component.jaccard"] == 0.0
