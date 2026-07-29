from __future__ import annotations

from experiments.stall_preaction_model_pilot import (
    select_rescue_candidate,
    select_zero_false_threshold,
)


def test_trigger_threshold_prefers_highest_zero_false_recall() -> None:
    report = select_zero_false_threshold(
        [1, 1, 0, 0], [0.92, 0.72, 0.69, 0.10], thresholds=(0.5, 0.7, 0.9)
    )
    assert report["selected"]["threshold"] == 0.7
    assert report["selected"]["recall"] == 1.0
    assert report["selected"]["false_positive_count"] == 0


def test_rescue_ranking_uses_time_only_after_escape_and_delta() -> None:
    rows = [
        {
            "candidate_id": "slow-big-delta",
            "candidate_rank": 3,
            "predicted_stable_escape_probability": 0.8,
            "predicted_conflict_delta": 4.0,
            "predicted_total_decision_seconds": 5.0,
        },
        {
            "candidate_id": "fast-small-delta",
            "candidate_rank": 2,
            "predicted_stable_escape_probability": 0.9,
            "predicted_conflict_delta": 3.0,
            "predicted_total_decision_seconds": 1.0,
        },
        {
            "candidate_id": "lower-escape",
            "candidate_rank": 4,
            "predicted_stable_escape_probability": 0.4,
            "predicted_conflict_delta": 10.0,
            "predicted_total_decision_seconds": 0.1,
        },
    ]
    assert select_rescue_candidate(rows)["candidate_id"] == "slow-big-delta"


def test_rescue_falls_back_to_highest_escape_probability_below_floor() -> None:
    rows = [
        {
            "candidate_id": "higher-probability",
            "candidate_rank": 2,
            "predicted_stable_escape_probability": 0.4,
            "predicted_conflict_delta": 1.0,
            "predicted_total_decision_seconds": 2.0,
        },
        {
            "candidate_id": "higher-delta",
            "candidate_rank": 3,
            "predicted_stable_escape_probability": 0.3,
            "predicted_conflict_delta": 10.0,
            "predicted_total_decision_seconds": 1.0,
        },
    ]
    assert select_rescue_candidate(rows)["candidate_id"] == "higher-probability"
