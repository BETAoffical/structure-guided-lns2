"""Shared, controller-agnostic episode evaluation helpers."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Hashable, Iterable


EpisodeKey = tuple[Hashable, ...]


def mean(values: Iterable[float | int]) -> float:
    """Return the arithmetic mean, using zero for an empty collection."""

    numbers = list(map(float, values))
    return statistics.fmean(numbers) if numbers else 0.0


def metric(source: dict[str, Any], name: str) -> float:
    """Read a numeric metric while preserving the historical zero default."""

    value = source.get(name)
    return float(value) if isinstance(value, (int, float)) else 0.0


def finite_metric(source: dict[str, Any], name: str) -> float:
    """Read a finite numeric metric, using zero for missing/non-finite values."""

    value = metric(source, name)
    return value if math.isfinite(value) else 0.0


def dataset_tasks(dataset: Path, split: str) -> dict[str, dict[str, Any]]:
    """Index a dataset manifest by task id."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    manifest = dataset / split / "manifest.jsonl"
    with manifest.open("r", encoding="utf-8") as stream:
        rows = [
            json.loads(line, parse_constant=reject_constant)
            for line in stream
            if line.strip()
        ]
    return {str(row["task_id"]): row for row in rows}


def capped_controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize reset-inclusive, capped wall-clock episodes."""

    completed_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and isinstance(row.get("summary"), dict)
    ]
    error_rows = [
        row
        for row in rows
        if row.get("status") != "ok" or not isinstance(row.get("summary"), dict)
    ]
    episodes = [dict(row["summary"]) for row in completed_rows]
    totals = [dict(summary.get("controller_totals") or {}) for summary in episodes]
    successes = [summary for summary in episodes if bool(summary.get("success"))]
    error_kinds = Counter(
        str(row.get("error_kind") or "missing_summary") for row in error_rows
    )
    return {
        "episode_count": len(rows),
        "completed_episode_count": len(episodes),
        "execution_error_count": len(error_rows),
        "execution_error_kind_counts": dict(sorted(error_kinds.items())),
        "success_count": len(successes),
        "failure_count": len(rows) - len(successes),
        "mean_capped_wall_time_to_feasible": mean(
            finite_metric(summary, "capped_wall_time_to_feasible")
            for summary in episodes
        ),
        "mean_success_wall_time_to_feasible": mean(
            finite_metric(summary, "wall_time_to_feasible")
            for summary in successes
        ),
        "mean_repair_iterations": mean(
            int(summary.get("repair_iterations", 0)) for summary in episodes
        ),
        "mean_normalized_fixed_budget_conflict_auc": mean(
            finite_metric(summary, "normalized_fixed_budget_conflict_auc")
            for summary in episodes
        ),
        "mean_normalized_wall_clock_conflict_auc": mean(
            finite_metric(summary, "normalized_wall_clock_conflict_auc")
            for summary in episodes
        ),
        "mean_pp_replan_seconds": mean(
            finite_metric(total, "pp_replan_seconds") for total in totals
        ),
        "mean_repair_wall_seconds": mean(
            finite_metric(summary, "repair_wall_seconds") for summary in episodes
        ),
        "mean_controller_seconds": mean(
            finite_metric(total, "controller_seconds_before_repair")
            for total in totals
        ),
        "mean_proposal_seconds": mean(
            finite_metric(total, "proposal_seconds") for total in totals
        ),
        "mean_feature_seconds": mean(
            finite_metric(total, "feature_seconds") for total in totals
        ),
        "mean_inference_seconds": mean(
            finite_metric(total, "inference_seconds") for total in totals
        ),
        "mean_state_export_seconds": mean(
            finite_metric(total, "state_export_seconds") for total in totals
        ),
        "invalid_action_count": sum(
            int(summary.get("invalid_action_count", 0)) for summary in episodes
        ),
        "fingerprint_mismatch_count": sum(
            int(summary.get("fingerprint_mismatch_count", 0))
            for summary in episodes
        ),
        "fallback_count": sum(
            int(total.get("pruner_fallback_count", 0)) for total in totals
        ),
    }


def raw_ttf_controller_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize successful run-to-completion time-to-feasible episodes."""

    summaries = [
        dict(row.get("summary") or {})
        for row in rows
        if row.get("status") == "ok"
    ]
    successes = [row for row in summaries if bool(row.get("success"))]
    totals = [dict(row.get("controller_totals") or {}) for row in summaries]
    ttf = [metric(row, "wall_time_to_feasible") for row in successes]
    return {
        "episode_count": len(rows),
        "execution_error_count": len(rows) - len(summaries),
        "success_count": len(successes),
        "mean_raw_wall_time_to_feasible": mean(ttf),
        "median_raw_wall_time_to_feasible": statistics.median(ttf) if ttf else 0.0,
        "mean_repair_iterations": mean(
            float(row.get("repair_iterations", 0)) for row in summaries
        ),
        "mean_normalized_wall_clock_conflict_auc": mean(
            metric(row, "normalized_wall_clock_conflict_auc") for row in summaries
        ),
        "mean_pp_replan_seconds": mean(
            metric(row, "pp_replan_seconds") for row in totals
        ),
        "mean_controller_seconds_before_repair": mean(
            metric(row, "controller_seconds_before_repair") for row in totals
        ),
        "mean_neighborhood_selection_seconds": mean(
            metric(row, "neighborhood_selection_seconds") for row in totals
        ),
        "invalid_action_count": sum(
            int(row.get("invalid_action_count", 0)) for row in summaries
        ),
        "fingerprint_mismatch_count": sum(
            int(row.get("fingerprint_mismatch_count", 0)) for row in summaries
        ),
    }


def paired_raw_ttf_comparison(
    baseline: dict[EpisodeKey, dict[str, Any]],
    challenger: dict[EpisodeKey, dict[str, Any]],
    keys: Iterable[EpisodeKey],
) -> dict[str, Any]:
    """Compare paired successful episodes on raw wall-clock TTF."""

    pairs = []
    for key in keys:
        left = baseline.get(key)
        right = challenger.get(key)
        if (
            left is None
            or right is None
            or left.get("status") != "ok"
            or right.get("status") != "ok"
        ):
            return {"valid": False, "paired_episode_count": len(pairs)}
        left_summary = dict(left.get("summary") or {})
        right_summary = dict(right.get("summary") or {})
        if not bool(left_summary.get("success")) or not bool(
            right_summary.get("success")
        ):
            return {"valid": False, "paired_episode_count": len(pairs)}
        pairs.append((key, left_summary, right_summary))

    left_ttf = [metric(left, "wall_time_to_feasible") for _, left, _ in pairs]
    right_ttf = [metric(right, "wall_time_to_feasible") for _, _, right in pairs]
    baseline_mean = mean(left_ttf)
    challenger_mean = mean(right_ttf)
    deltas = [right - left for left, right in zip(left_ttf, right_ttf)]
    return {
        "valid": True,
        "paired_episode_count": len(pairs),
        "baseline_mean_raw_ttf": baseline_mean,
        "challenger_mean_raw_ttf": challenger_mean,
        "mean_raw_ttf_delta_seconds": challenger_mean - baseline_mean,
        "mean_raw_ttf_relative_improvement": (
            (baseline_mean - challenger_mean) / baseline_mean
            if baseline_mean
            else 0.0
        ),
        "faster_count": sum(value < -1e-9 for value in deltas),
        "slower_count": sum(value > 1e-9 for value in deltas),
        "tied_count": sum(abs(value) <= 1e-9 for value in deltas),
        "paired_faster_fraction": (
            sum(value < -1e-9 for value in deltas) / len(deltas)
            if deltas
            else 0.0
        ),
        "mean_repair_iterations_delta": mean(
            float(right.get("repair_iterations", 0))
            - float(left.get("repair_iterations", 0))
            for _, left, right in pairs
        ),
        "mean_normalized_wall_auc_delta": mean(
            metric(right, "normalized_wall_clock_conflict_auc")
            - metric(left, "normalized_wall_clock_conflict_auc")
            for _, left, right in pairs
        ),
    }


__all__ = [
    "capped_controller_summary",
    "dataset_tasks",
    "finite_metric",
    "mean",
    "metric",
    "paired_raw_ttf_comparison",
    "raw_ttf_controller_summary",
]
