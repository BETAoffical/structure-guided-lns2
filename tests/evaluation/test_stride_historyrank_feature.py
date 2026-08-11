from __future__ import annotations

import math

from experiments.stride_historyrank_feature import (
    choose_by_directed_feature,
    directed_feature_values,
)


def _quality(*, agents: list[int], families: list[str], value: float) -> dict:
    features = {f"feature_{index:03d}": 0.0 for index in range(124)}
    features.update(
        {
            "realized.internal_conflict_coverage": value,
            "realized.incident_conflict_coverage": value,
            "realized.component_coverage_mean": value,
            "realized.boundary_conflict_edges": value,
            "realized.path_overlap_mean": value,
        }
    )
    while len(features) > 124:
        features.pop(next(iter(features)))
    return {
        "feature_count": 124,
        "features": features,
        "selection_families": families,
    }


def test_directed_features_reward_agent_novelty() -> None:
    repeated_meta = {"agents": [1, 2, 3, 4], "score": 4.0}
    candidate_meta = {"agents": [3, 4, 5, 6], "score": 3.0}
    repeated = _quality(agents=[1, 2, 3, 4], families=["a"], value=1.0)
    candidate = _quality(agents=[3, 4, 5, 6], families=["b"], value=2.0)
    ranges = {name: 1.0 for name in repeated["features"]}
    values = directed_feature_values(
        candidate_meta=candidate_meta,
        repeated_meta=repeated_meta,
        candidate_quality=candidate,
        repeated_quality=repeated,
        ranges=ranges,
    )
    assert math.isclose(values["agent_novelty"], 2.0 / 3.0)
    assert values["family_novelty"] == 1.0
    assert values["removed_agent_ratio"] == 0.5
    assert values["added_agent_ratio"] == 0.5
    assert values["v2_score_margin"] == -1.0


def test_choose_by_directed_feature_uses_v2_score_then_id_ties() -> None:
    rows = [
        {
            "candidate_id": "z",
            "score": 2.0,
            "directed_features": {"agent_novelty": 1.0},
        },
        {
            "candidate_id": "b",
            "score": 3.0,
            "directed_features": {"agent_novelty": 1.0},
        },
        {
            "candidate_id": "a",
            "score": 3.0,
            "directed_features": {"agent_novelty": 1.0},
        },
    ]
    chosen = choose_by_directed_feature(rows, feature_name="agent_novelty")
    assert chosen["candidate_id"] == "a"
