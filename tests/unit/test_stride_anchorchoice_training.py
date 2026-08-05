from __future__ import annotations

from experiments.stride_anchorchoice_training import _anchorchoice_predictions


def test_anchorchoice_uses_highest_probability_then_candidate_id() -> None:
    scores = {
        "choose": [
            {"candidate_id": "c", "safety_probability": 0.7},
            {"candidate_id": "b", "safety_probability": 0.7},
        ],
        "fallback": [
            {"candidate_id": "y", "safety_probability": 0.49},
            {"candidate_id": "z", "safety_probability": 0.2},
        ],
    }
    anchor = {"choose": "a", "fallback": "x"}
    assert _anchorchoice_predictions(
        candidate_scores=scores,
        anchor_predictions=anchor,
        selection_threshold=0.5,
    ) == {"choose": "b", "fallback": "x"}


def test_anchorchoice_never_returns_outside_candidate_or_anchor_set() -> None:
    scores = {"state": [{"candidate_id": "candidate", "safety_probability": 0.8}]}
    anchor = {"state": "anchor"}
    selected = _anchorchoice_predictions(
        candidate_scores=scores,
        anchor_predictions=anchor,
        selection_threshold=0.5,
    )
    assert selected["state"] in {"anchor", "candidate"}
