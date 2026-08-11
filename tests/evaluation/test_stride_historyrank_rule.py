from __future__ import annotations

from experiments.stride_historyrank_rule import choose_fallback


def test_choose_fallback_excludes_only_exact_repeat_and_uses_v2_score() -> None:
    candidates = [
        {"candidate_id": "repeat", "score": 10.0},
        {"candidate_id": "z", "score": 8.0},
        {"candidate_id": "a", "score": 8.0},
        {"candidate_id": "low", "score": 1.0},
    ]
    selected = choose_fallback(candidates, repeated_candidate_id="repeat")
    assert selected["candidate_id"] == "a"


def test_choose_fallback_rejects_empty_remaining_pool() -> None:
    try:
        choose_fallback(
            [{"candidate_id": "repeat", "score": 10.0}],
            repeated_candidate_id="repeat",
        )
    except ValueError as exc:
        assert "no fallback" in str(exc)
    else:
        raise AssertionError("expected empty fallback pool to fail")
