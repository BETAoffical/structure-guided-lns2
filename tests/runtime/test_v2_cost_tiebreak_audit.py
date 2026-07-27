from __future__ import annotations

from experiments import v2_cost_tiebreak_audit as module


def _arm(
    candidate: str,
    score: float,
    *,
    base: bool = False,
    reduction: float = 10.0,
    seconds: float = 1.0,
    progress: float = 0.9,
    no_progress: float = 0.1,
) -> dict:
    return {
        "candidate_id": candidate,
        "route": "model",
        "actual_size": 4,
        "base_selected": base,
        "v2_score": score,
        "predicted": {
            "effective_progress_probability": progress,
            "no_progress_probability": no_progress,
            "conflict_reduction": reduction,
            "repair_seconds": seconds,
        },
        "actual": {
            "effective_rate": progress,
            "no_progress_rate": no_progress,
            "hard_failure_rate": no_progress,
            "accepted_noop_rate": 0.0,
            "conflict_reduction": reduction,
            "repair_seconds": seconds,
        },
    }


def _state(arms: list[dict]) -> dict:
    return {
        "state_id": "state",
        "map_id": "map",
        "layout_mode": "layout",
        "agent_count": 400,
        "arms": arms,
    }


def _calibration() -> dict:
    return {
        "global": {
            "reduction_overprediction": 0.0,
            "seconds_underprediction": 0.0,
        },
        "by_agent_count": {
            "400": {
                "reduction_overprediction": 0.0,
                "seconds_underprediction": 0.0,
            }
        },
    }


def _thresholds() -> dict:
    return {
        "effective_probability_tolerance": 0.0,
        "no_progress_probability_tolerance": 0.0,
        "conflict_reduction_retention": 0.98,
        "minimum_time_improvement": 0.10,
        "minimum_utility_improvement": 0.10,
    }


def test_selects_cheaper_top3_candidate_with_quality_guard() -> None:
    state = _state(
        [
            _arm("base", 3.0, base=True, seconds=1.0),
            _arm("fast", 2.0, reduction=9.9, seconds=0.5),
            _arm("third", 1.0, reduction=9.8, seconds=0.9),
        ]
    )
    selected, diagnostic = module.select_cost_tiebreak(
        state, _thresholds(), _calibration()
    )
    assert selected["candidate_id"] == "fast"
    assert diagnostic["override"] is True
    assert diagnostic["selected_v2_rank"] == 2


def test_quality_guard_keeps_v2_candidate() -> None:
    state = _state(
        [
            _arm("base", 3.0, base=True, seconds=1.0),
            _arm("too-weak", 2.0, reduction=9.0, seconds=0.1),
            _arm("third", 1.0, reduction=8.0, seconds=0.1),
        ]
    )
    selected, diagnostic = module.select_cost_tiebreak(
        state, _thresholds(), _calibration()
    )
    assert selected["candidate_id"] == "base"
    assert diagnostic["override"] is False


def test_candidate_outside_top3_cannot_override() -> None:
    state = _state(
        [
            _arm("base", 4.0, base=True, seconds=1.0),
            _arm("second", 3.0, seconds=1.0),
            _arm("third", 2.0, seconds=1.0),
            _arm("outside", 1.0, seconds=0.01),
        ]
    )
    selected, _ = module.select_cost_tiebreak(
        state, _thresholds(), _calibration()
    )
    assert selected["candidate_id"] == "base"


def test_uncertainty_margin_can_block_apparent_speed_gain() -> None:
    state = _state(
        [
            _arm("base", 3.0, base=True, seconds=1.0),
            _arm("fast", 2.0, reduction=9.9, seconds=0.5),
            _arm("third", 1.0, reduction=9.8, seconds=0.9),
        ]
    )
    calibration = _calibration()
    calibration["by_agent_count"]["400"]["reduction_overprediction"] = 9.0
    selected, _ = module.select_cost_tiebreak(
        state, _thresholds(), calibration
    )
    assert selected["candidate_id"] == "base"


def test_evaluation_uses_cumulative_reduction_and_time() -> None:
    states = [
        _state(
            [
                _arm("base", 3.0, base=True, reduction=10.0, seconds=1.0),
                _arm("fast", 2.0, reduction=10.0, seconds=0.5),
                _arm("third", 1.0, reduction=9.0, seconds=0.5),
            ]
        )
    ]
    report, rows = module.evaluate_policy(
        states, _thresholds(), _calibration()
    )
    assert report["override_count"] == 1
    assert report["comparison"]["conflict_reduction_ratio"] == 1.0
    assert report["comparison"]["repair_time_ratio"] == 0.5
    assert report["comparison"]["efficiency_ratio"] == 2.0
    assert rows[0]["selected_candidate_id"] == "fast"


def test_fold_summary_uses_balanced_fold_train_key() -> None:
    summary = module._fold_summary(
        {
            "fold": 2,
            "train_maps": ["map-a", "map-b"],
            "validation_maps": ["map-c"],
        },
        [{}, {}, {}],
        [{}],
    )
    assert summary == {
        "fold": 2,
        "training_map_count": 2,
        "validation_map_count": 1,
        "training_trial_count": 3,
        "validation_trial_count": 1,
    }
