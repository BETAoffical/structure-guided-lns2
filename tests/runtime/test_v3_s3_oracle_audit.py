from __future__ import annotations

import unittest

from experiments.v3_s3_oracle_audit import (
    _metric_order_key,
    _safe_ratio,
    select_first_action_oracle,
    select_oracle_row,
)


def _trial(
    *,
    reduction: float,
    seconds: float,
    outcome: str = "conflict_reduced",
) -> dict:
    return {
        "steps": [
            {
                "step": 1,
                "executed": True,
                "template_valid": True,
                "repair_outcome": outcome,
                "conflict_reduction": reduction,
                "conflicts_before": 10,
                "conflicts_after": 10 - int(reduction),
                "selection_seconds": 0.1,
                "total_seconds": seconds,
            }
        ]
    }


def _row(
    sequence_id: str,
    *,
    first_template: str,
    reduction: float,
    seconds: float,
) -> dict:
    family, size, representative = first_template.split(":")
    template = {
        "family": family,
        "requested_size": int(size.removeprefix("size")),
        "representative": int(representative.removeprefix("rep")),
        "template_key": first_template,
    }
    return {
        "sequence_id": sequence_id,
        "templates": [template, template, template],
        "actual": {
            "runtime_prefix_net_conflict_reduction": reduction,
            "runtime_prefix_total_seconds": seconds,
            "runtime_prefix_selection_seconds": 0.1,
            "runtime_prefix_no_progress_rate": float(reduction <= 0.0),
            "runtime_prefix_feasible_rate": 0.0,
            "trials": [
                _trial(reduction=reduction, seconds=seconds),
                _trial(reduction=reduction, seconds=seconds),
            ],
        },
    }


class V3S3OracleSelectionTests(unittest.TestCase):
    def test_efficiency_oracle_uses_observed_reduction_per_second(self) -> None:
        slow_large = _row(
            "large",
            first_template="collision:size16:rep0",
            reduction=10.0,
            seconds=5.0,
        )
        fast_small = _row(
            "small",
            first_template="collision:size4:rep0",
            reduction=6.0,
            seconds=2.0,
        )
        self.assertIs(
            select_oracle_row(
                [slow_large, fast_small], objective="efficiency"
            ),
            fast_small,
        )

    def test_quality_time_oracle_rejects_fast_low_reduction(self) -> None:
        high_quality = _row(
            "quality",
            first_template="target:size16:rep0",
            reduction=10.0,
            seconds=5.0,
        )
        fast_low_quality = _row(
            "fast",
            first_template="target:size4:rep0",
            reduction=8.0,
            seconds=1.0,
        )
        self.assertIs(
            select_oracle_row(
                [high_quality, fast_low_quality],
                objective="quality_time",
                quality_retention=0.98,
            ),
            high_quality,
        )

    def test_reduction_oracle_breaks_tie_by_time(self) -> None:
        slower = _row(
            "slower",
            first_template="random:size8:rep0",
            reduction=8.0,
            seconds=3.0,
        )
        faster = _row(
            "faster",
            first_template="random:size8:rep1",
            reduction=8.0,
            seconds=2.0,
        )
        self.assertIs(
            select_oracle_row([slower, faster], objective="reduction"),
            faster,
        )

    def test_first_action_oracle_groups_shared_first_templates(self) -> None:
        rows = [
            _row(
                "a",
                first_template="collision:size4:rep0",
                reduction=2.0,
                seconds=1.0,
            ),
            _row(
                "b",
                first_template="collision:size4:rep0",
                reduction=2.0,
                seconds=1.0,
            ),
            _row(
                "c",
                first_template="collision:size8:rep0",
                reduction=3.0,
                seconds=1.0,
            ),
        ]
        template, metrics = select_first_action_oracle(rows)
        self.assertEqual(template, "collision:size8:rep0")
        self.assertEqual(metrics["conflict_reduction"], 3.0)

    def test_ratio_and_order_are_deterministic_at_zero_time(self) -> None:
        self.assertGreater(_safe_ratio(1.0, 0.0), 1e9)
        left = _metric_order_key(
            "a",
            {
                "conflict_reduction": 1.0,
                "total_seconds": 1.0,
                "no_progress": 0.0,
                "feasible": 0.0,
            },
            objective="efficiency",
        )
        right = _metric_order_key(
            "b",
            {
                "conflict_reduction": 1.0,
                "total_seconds": 1.0,
                "no_progress": 0.0,
                "feasible": 0.0,
            },
            objective="efficiency",
        )
        self.assertLess(left, right)


if __name__ == "__main__":
    unittest.main()
