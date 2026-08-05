from __future__ import annotations

import math

from experiments.stride_marginchoice_training import _margin_regression_metrics


def test_margin_regression_metrics_reward_correct_signed_order() -> None:
    metrics = _margin_regression_metrics(
        [-1.0, 0.0, 1.0],
        [-0.5, 0.0, 0.5],
        [1.0, 1.0, 1.0],
    )
    assert math.isclose(metrics["weighted_margin_correlation"], 1.0)
    assert metrics["weighted_mean_absolute_error"] < (
        metrics["weighted_constant_mean_absolute_error"]
    )


def test_margin_regression_metrics_detect_reversed_order() -> None:
    metrics = _margin_regression_metrics(
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
    )
    assert math.isclose(metrics["weighted_margin_correlation"], -1.0)
