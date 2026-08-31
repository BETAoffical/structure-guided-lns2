from __future__ import annotations

import json
from pathlib import Path

import pytest

from lns2_selector.evaluation.episode_statistics import (
    capped_controller_summary,
    dataset_tasks,
    finite_metric,
    mean,
    metric,
    paired_raw_ttf_comparison,
    raw_ttf_controller_summary,
)


def _episode(
    ttf: float,
    *,
    success: bool = True,
    iterations: int = 2,
    auc: float = 0.5,
) -> dict:
    return {
        "status": "ok",
        "summary": {
            "success": success,
            "wall_time_to_feasible": ttf,
            "capped_wall_time_to_feasible": ttf,
            "repair_iterations": iterations,
            "normalized_fixed_budget_conflict_auc": auc,
            "normalized_wall_clock_conflict_auc": auc,
            "repair_wall_seconds": ttf,
            "controller_totals": {
                "pp_replan_seconds": 0.4,
                "controller_seconds_before_repair": 0.2,
                "neighborhood_selection_seconds": 0.1,
                "proposal_seconds": 0.05,
                "feature_seconds": 0.04,
                "inference_seconds": 0.03,
                "state_export_seconds": 0.02,
                "pruner_fallback_count": 1,
            },
            "invalid_action_count": 0,
            "fingerprint_mismatch_count": 0,
        },
    }


def test_dataset_tasks_preserves_manifest_index_and_json_safety(tmp_path: Path) -> None:
    manifest = tmp_path / "split" / "manifest.jsonl"
    manifest.parent.mkdir()
    manifest.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"task_id": "a", "value": 1},
                {"task_id": "b", "value": 2},
            )
        ),
        encoding="utf-8",
    )
    assert dataset_tasks(tmp_path, "split") == {
        "a": {"task_id": "a", "value": 1},
        "b": {"task_id": "b", "value": 2},
    }

    manifest.write_text('{"task_id":"a","value":NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        dataset_tasks(tmp_path, "split")


def test_numeric_helpers_keep_historical_zero_and_finite_semantics() -> None:
    assert mean([]) == 0.0
    assert mean([1, 2, 3]) == 2.0
    assert metric({"value": 2}, "value") == 2.0
    assert metric({}, "value") == 0.0
    assert finite_metric({"value": float("nan")}, "value") == 0.0


def test_controller_summaries_preserve_capped_and_raw_contracts() -> None:
    rows = [_episode(2.0), _episode(4.0), {"status": "timeout"}]
    capped = capped_controller_summary(rows)
    assert capped["episode_count"] == 3
    assert capped["completed_episode_count"] == 2
    assert capped["execution_error_kind_counts"] == {"missing_summary": 1}
    assert capped["mean_capped_wall_time_to_feasible"] == 3.0
    assert capped["fallback_count"] == 2

    raw = raw_ttf_controller_summary(rows)
    assert raw["episode_count"] == 3
    assert raw["execution_error_count"] == 1
    assert raw["mean_raw_wall_time_to_feasible"] == 3.0
    assert raw["median_raw_wall_time_to_feasible"] == 3.0


def test_paired_raw_ttf_comparison_requires_complete_successful_pairs() -> None:
    keys = [("task-a", 1), ("task-b", 1)]
    baseline = {keys[0]: _episode(4.0), keys[1]: _episode(2.0)}
    challenger = {keys[0]: _episode(2.0), keys[1]: _episode(3.0)}
    result = paired_raw_ttf_comparison(baseline, challenger, keys)
    assert result["valid"] is True
    assert result["paired_episode_count"] == 2
    assert result["baseline_mean_raw_ttf"] == 3.0
    assert result["challenger_mean_raw_ttf"] == 2.5
    assert result["faster_count"] == 1
    assert result["slower_count"] == 1

    challenger[keys[1]] = _episode(3.0, success=False)
    assert paired_raw_ttf_comparison(baseline, challenger, keys) == {
        "valid": False,
        "paired_episode_count": 1,
    }
