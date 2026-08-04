from __future__ import annotations

from experiments.stride_maprank_failure_analysis import _first_override


def _transition(
    before: str,
    after: str,
    selected: str,
    *,
    selected_other: str,
    conflict_delta: int,
    generated: int,
) -> dict:
    return {
        "before_fingerprint": before,
        "after_fingerprint": after,
        "controller": {
            "selected_candidate_id": selected,
            "candidate_pool": [
                {
                    "candidate_id": selected,
                    "actual_size": 4,
                    "selection_families": ["collision:4"],
                    "feature_out_of_range_fraction": 0.1,
                    "score": 2.0,
                },
                {
                    "candidate_id": selected_other,
                    "actual_size": 8,
                    "selection_families": ["target:8"],
                    "feature_out_of_range_fraction": 0.2,
                    "score": 1.0,
                },
            ],
        },
        "metrics": {
            "conflicts_before": 3,
            "conflicts_after": 3 - conflict_delta,
            "conflict_delta": conflict_delta,
            "replan_success": True,
            "pp_replan_seconds": 0.5,
            "sum_of_costs_before": 100,
            "sum_of_costs_after": 102,
        },
        "low_level_delta": {
            "generated": generated,
            "expanded": generated // 2,
            "reopened": 1,
        },
    }


def test_first_override_preserves_common_prefix_and_cross_scores() -> None:
    common_left = _transition(
        "state-0", "state-1", "candidate-a", selected_other="candidate-b",
        conflict_delta=1, generated=20,
    )
    common_right = _transition(
        "state-0", "state-1", "candidate-a", selected_other="candidate-b",
        conflict_delta=1, generated=20,
    )
    baseline = _transition(
        "state-1", "state-left", "candidate-a", selected_other="candidate-b",
        conflict_delta=1, generated=30,
    )
    challenger = _transition(
        "state-1", "state-right", "candidate-b", selected_other="candidate-a",
        conflict_delta=2, generated=10,
    )
    challenger["controller"]["candidate_pool"][0]["score"] = 3.0
    challenger["controller"]["candidate_pool"][1]["score"] = 4.0
    prefix, override, errors = _first_override(
        [common_left, baseline], [common_right, challenger]
    )
    assert errors == []
    assert prefix == 1
    assert override is not None
    assert override["baseline_candidate"]["candidate_id"] == "candidate-a"
    assert override["challenger_candidate"]["candidate_id"] == "candidate-b"
    assert override["immediate_effect"]["conflict_delta_difference"] == 1
    assert override["immediate_effect"]["low_level_generated_difference"] == -20


def test_first_override_rejects_state_divergence_before_action() -> None:
    left = _transition(
        "left", "after-left", "candidate-a", selected_other="candidate-b",
        conflict_delta=1, generated=10,
    )
    right = _transition(
        "right", "after-right", "candidate-b", selected_other="candidate-a",
        conflict_delta=1, generated=10,
    )
    prefix, override, errors = _first_override([left], [right])
    assert prefix == 0
    assert override is None
    assert errors == ["state diverged before action at decision 0"]
