from __future__ import annotations

import pytest

from experiments.v2_cost_top3_runtime import (
    load_v2_cost_top3_config,
    select_v2_cost_top3,
)


def _config() -> dict:
    return {
        "schema": "lns2.v2_cost_top3_config.v1",
        "top_k": 3,
        "thresholds": {
            "conflict_reduction_retention": 0.98,
            "effective_probability_tolerance": 0.05,
            "minimum_time_improvement": 0.10,
            "minimum_utility_improvement": 0.0,
            "no_progress_probability_tolerance": 0.05,
        },
        "uncertainty_calibration": {
            "quantile": 0.8,
            "global": {
                "reduction_overprediction": 0.0,
                "seconds_underprediction": 0.0,
            },
            "by_agent_count": {},
        },
        "source": {},
    }


def _candidates() -> list[dict]:
    return [
        {"candidate_id": f"c{index}", "candidate_key": f"c{index}"}
        for index in range(4)
    ]


def _predictions() -> dict[str, list[float]]:
    return {
        "effective_progress_probability": [0.9, 0.92, 0.2, 0.95],
        "no_progress_probability": [0.1, 0.08, 0.8, 0.05],
        "conflict_reduction": [10.0, 9.9, 20.0, 12.0],
        "repair_seconds": [2.0, 1.0, 0.2, 0.1],
    }


def test_selects_cheaper_candidate_only_within_frozen_v2_top3() -> None:
    selected, diagnostic = select_v2_cost_top3(
        _candidates(),
        [4.0, 3.0, 2.0, 1.0],
        _predictions(),
        load_v2_cost_top3_config(_config()),
        agent_count=400,
    )
    assert selected == 1
    assert diagnostic["override"] is True
    assert diagnostic["selected_v2_rank"] == 2
    assert diagnostic["top3_candidate_ids"] == ["c0", "c1", "c2"]


def test_keeps_v2_when_no_candidate_clears_guard() -> None:
    predictions = _predictions()
    predictions["conflict_reduction"][1] = 5.0
    selected, diagnostic = select_v2_cost_top3(
        _candidates(),
        [4.0, 3.0, 2.0, 1.0],
        predictions,
        load_v2_cost_top3_config(_config()),
        agent_count=100,
    )
    assert selected == 0
    assert diagnostic["override"] is False
    assert diagnostic["agent_count_calibration"] == "global"


def test_rejects_incomplete_predictions() -> None:
    predictions = _predictions()
    predictions.pop("repair_seconds")
    with pytest.raises(ValueError, match="incomplete"):
        select_v2_cost_top3(
            _candidates(),
            [4.0, 3.0, 2.0, 1.0],
            predictions,
            load_v2_cost_top3_config(_config()),
            agent_count=400,
        )


def test_ignores_bundle_derived_utility_output() -> None:
    predictions = _predictions()
    predictions["utility"] = [1.0] * 4
    selected, _diagnostic = select_v2_cost_top3(
        _candidates(),
        [4.0, 3.0, 2.0, 1.0],
        predictions,
        load_v2_cost_top3_config(_config()),
        agent_count=400,
    )
    assert selected == 1


def test_config_requires_exact_frozen_threshold_set() -> None:
    payload = _config()
    payload["thresholds"].pop("minimum_time_improvement")
    with pytest.raises(ValueError, match="incomplete"):
        load_v2_cost_top3_config(payload)
