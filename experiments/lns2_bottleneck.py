from __future__ import annotations

import csv
import html
import json
import math
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, sha256_file
from experiments.closed_loop_confirmation import (
    ClosedLoopTraceError,
    validate_closed_loop_trace,
)
from experiments.closed_loop_trace_storage import (
    TRACE_FORMAT_DELTA_GZIP_V2,
    TRACE_FORMAT_FULL_V1,
    open_trace_text,
    read_trace_events,
    storage_fingerprint,
)
from experiments.tradeoff_evaluation import _manifest_path as _controller_manifest_path
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)


REPORT_SCHEMA = "lns2.selector_bottleneck.v3"
CONTROLLERS = (
    "official_adaptive",
    "v2-full",
    "mixed-full-v2",
    "v3-s3",
)
LABELS = {
    "official_adaptive": "Original LNS2 Adaptive",
    "v2-full": "Optimized model (v2)",
    "mixed-full-v2": "Mixed-load model (v2)",
    "v3-s3": "Sequence-aware model (v3-S3)",
}
CONTROLLER_COLORS = {
    "official_adaptive": "#4c78a8",
    "v2-full": "#f58518",
    "mixed-full-v2": "#54a24b",
    "v3-s3": "#e45756",
}
TIMING_FIELDS = (
    "native_step_seconds",
    "episode_runtime_delta_seconds",
    "neighborhood_selection_seconds",
    "candidate_generation_seconds",
    "state_check_seconds",
    "state_check_fingerprint_seconds",
    "state_analysis_seconds",
    "proposal_feature_seconds",
    "realized_feature_seconds",
    "ranking_inference_seconds",
    "v3_s3_seconds",
    "selection_residual_seconds",
    "pp_replan_seconds",
    "repair_bookkeeping_seconds",
    "native_residual_seconds",
    "state_export_seconds",
    "environment_step_residual_seconds",
    "pre_step_orchestration_seconds",
    "post_step_orchestration_seconds",
    "state_fingerprint_seconds",
    "iteration_wall_seconds",
    "trace_write_seconds",
)
SUPPORTED_NATIVE_TIMING_SCHEMAS = {
    "lns2.repair_timing.v1",
    "lns2.repair_timing.v2",
}
DECOMPOSITION_FIELDS = (
    "environment_construct_seconds",
    "reset_wall_seconds",
    "neighborhood_selection_seconds",
    "pp_replan_seconds",
    "repair_bookkeeping_seconds",
    "state_export_seconds",
    "environment_step_residual_seconds",
    "orchestration_seconds",
    "trace_write_seconds",
    "finalization_non_trace_seconds",
    "timing_unaccounted_seconds",
)


def _mean(values: Iterable[float | int | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def _median(values: Iterable[float | int | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.median(numbers) if numbers else None


def _p95(values: Iterable[float | int | None]) -> float | None:
    numbers = sorted(float(value) for value in values if value is not None)
    if not numbers:
        return None
    return numbers[min(len(numbers) - 1, math.ceil(0.95 * len(numbers)) - 1)]


def _correlation(
    left: Iterable[float | int | None], right: Iterable[float | int | None]
) -> float | None:
    pairs = [
        (float(x), float(y))
        for x, y in zip(left, right)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None
    xs, ys = zip(*pairs)
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in pairs)
    x_scale = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_scale == 0.0 or y_scale == 0.0:
        return None
    return numerator / (x_scale * y_scale)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for name in row:
            if name not in seen:
                seen.add(name)
                fields.append(name)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _job_key(task_id: Any, solver_seed: Any) -> tuple[str, int] | None:
    if task_id is None or not str(task_id):
        return None
    try:
        seed = int(solver_seed)
    except (TypeError, ValueError):
        return None
    return str(task_id), seed


def _serialized_keys(keys: Iterable[tuple[str, int]]) -> list[list[Any]]:
    return [[task_id, seed] for task_id, seed in sorted(set(keys))]


def _configured_cohort(run: dict[str, Any]) -> dict[str, Any]:
    configuration = dict(run.get("configuration") or {})
    raw_cohort = configuration.get("cohort_job_keys_override")
    raw_keys: list[Any]
    structural_invalid = 0
    if raw_cohort is not None:
        source = "cohort_job_keys_override"
        if isinstance(raw_cohort, list):
            raw_keys = list(raw_cohort)
        else:
            raw_keys = []
            structural_invalid = 1
    else:
        task_ids = configuration.get("task_ids_override")
        solver_seeds = configuration.get("solver_seeds")
        if task_ids is None or solver_seeds is None:
            return {
                "source": "execution_schedule",
                "available": False,
                "keys": [],
                "invalid_entry_count": 0,
                "duplicate_key_count": 0,
            }
        source = "task_ids_override_x_solver_seeds"
        if not isinstance(task_ids, list) or not isinstance(solver_seeds, list):
            raw_keys = []
            structural_invalid = 1
        else:
            raw_keys = [
                [task_id, solver_seed]
                for task_id in task_ids
                for solver_seed in solver_seeds
            ]

    keys: list[tuple[str, int]] = []
    invalid = structural_invalid
    for value in raw_keys:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            invalid += 1
            continue
        key = _job_key(value[0], value[1])
        if key is None:
            invalid += 1
            continue
        keys.append(key)
    counts = Counter(keys)
    return {
        "source": source,
        "available": True,
        "keys": _serialized_keys(keys),
        "invalid_entry_count": invalid,
        "duplicate_key_count": sum(count - 1 for count in counts.values()),
    }


def _track_schedule_paths(roots: dict[str, Path]) -> list[Path]:
    parents = {Path(value).resolve().parent for value in roots.values()}
    if len(parents) != 1:
        return []
    parent = next(iter(parents))
    return [
        path
        for path in (parent / "execution_schedule.json", parent.parent / "execution_schedule.json")
        if path.is_file()
    ]


def _track_coverage(
    *,
    track: str,
    roots: dict[str, Path],
    episodes: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    controllers = tuple(map(str, roots))
    schedule_paths = _track_schedule_paths(roots)
    schedule_present = len(schedule_paths) == 1
    schedule_schema_valid = False
    schedule_entries: list[Any] = []
    if schedule_present:
        try:
            schedule = _read_json(schedule_paths[0])
            schedule_schema_valid = (
                str(schedule.get("schema")) == "lns2.controller_execution_schedule.v1"
            )
            raw_entries = schedule.get("entries")
            if isinstance(raw_entries, list):
                schedule_entries = raw_entries
        except (OSError, TypeError, ValueError):
            schedule_entries = []

    schedule_keys: list[tuple[str, int]] = []
    invalid_schedule_entries = 0
    invalid_controller_orders = 0
    for entry in schedule_entries:
        if not isinstance(entry, dict):
            invalid_schedule_entries += 1
            continue
        key = _job_key(entry.get("task_id"), entry.get("solver_seed"))
        if key is None:
            invalid_schedule_entries += 1
            continue
        schedule_keys.append(key)
        controller_order = entry.get("controller_order")
        if (
            not isinstance(controller_order, list)
            or len(controller_order) != len(controllers)
            or set(map(str, controller_order)) != set(controllers)
        ):
            invalid_controller_orders += 1
    schedule_counts = Counter(schedule_keys)
    schedule_duplicate_keys = {
        key: count for key, count in schedule_counts.items() if count > 1
    }
    schedule_key_set = set(schedule_keys)

    configured_sets: dict[str, set[tuple[str, int]]] = {}
    configuration_rows: dict[str, Any] = {}
    for controller in controllers:
        configured = dict(dict(metadata.get(controller) or {}).get("configured_cohort") or {})
        keys = {
            key
            for value in configured.get("keys", [])
            if isinstance(value, (list, tuple))
            and len(value) == 2
            and (key := _job_key(value[0], value[1])) is not None
        }
        if bool(configured.get("available")):
            configured_sets[controller] = keys
        configuration_rows[controller] = {
            "source": configured.get("source", "execution_schedule"),
            "available": bool(configured.get("available")),
            "key_count": len(keys),
            "invalid_entry_count": int(configured.get("invalid_entry_count", 0)),
            "duplicate_key_count": int(configured.get("duplicate_key_count", 0)),
            "missing_from_schedule": _serialized_keys(schedule_key_set - keys)
            if bool(configured.get("available"))
            else [],
            "unexpected_vs_schedule": _serialized_keys(keys - schedule_key_set)
            if bool(configured.get("available"))
            else [],
        }

    configured_union = set().union(*configured_sets.values()) if configured_sets else set()
    expected_keys = schedule_key_set or configured_union
    expected_source = "execution_schedule" if schedule_key_set else "run_config_cohort"
    configured_key_sets_match = len({frozenset(keys) for keys in configured_sets.values()}) <= 1
    configured_cohorts_match_schedule = all(
        keys == schedule_key_set for keys in configured_sets.values()
    ) if schedule_present else False
    configuration_valid = all(
        row["invalid_entry_count"] == 0 and row["duplicate_key_count"] == 0
        for row in configuration_rows.values()
    )

    controller_rows: dict[str, Any] = {}
    observed_sets: dict[str, set[tuple[str, int]]] = {}
    for controller in controllers:
        rows = [
            row
            for row in episodes
            if str(row.get("track")) == track
            and str(row.get("controller")) == controller
        ]
        observed: list[tuple[str, int]] = []
        invalid_manifest_entries = 0
        for row in rows:
            key = _job_key(row.get("task_id"), row.get("solver_seed"))
            if key is None:
                invalid_manifest_entries += 1
            else:
                observed.append(key)
        counts = Counter(observed)
        observed_set = set(observed)
        observed_sets[controller] = observed_set
        duplicate_keys = {
            key: count for key, count in counts.items() if count > 1
        }
        controller_rows[controller] = {
            "manifest_row_count": len(rows),
            "unique_key_count": len(observed_set),
            "invalid_entry_count": invalid_manifest_entries,
            "missing_keys": _serialized_keys(expected_keys - observed_set),
            "unexpected_keys": _serialized_keys(observed_set - expected_keys),
            "duplicate_keys": [
                [task_id, seed, count]
                for (task_id, seed), count in sorted(duplicate_keys.items())
            ],
            "duplicate_key_count": sum(count - 1 for count in duplicate_keys.values()),
        }

    controller_key_sets_match = len(
        {frozenset(keys) for keys in observed_sets.values()}
    ) == 1
    schedule_valid = (
        schedule_present
        and schedule_schema_valid
        and bool(schedule_entries)
        and not invalid_schedule_entries
        and not invalid_controller_orders
        and not schedule_duplicate_keys
    )
    manifests_valid = all(
        row["manifest_row_count"] > 0
        and row["invalid_entry_count"] == 0
        and not row["missing_keys"]
        and not row["unexpected_keys"]
        and row["duplicate_key_count"] == 0
        for row in controller_rows.values()
    )
    passed = (
        schedule_valid
        and bool(expected_keys)
        and configuration_valid
        and configured_key_sets_match
        and configured_cohorts_match_schedule
        and controller_key_sets_match
        and manifests_valid
    )
    return {
        "passed": passed,
        "schedule_path": str(schedule_paths[0]) if schedule_present else None,
        "schedule_present": schedule_present,
        "schedule_schema_valid": schedule_schema_valid,
        "schedule_entry_count": len(schedule_entries),
        "schedule_invalid_entry_count": invalid_schedule_entries,
        "schedule_invalid_controller_order_count": invalid_controller_orders,
        "schedule_duplicate_key_count": sum(
            count - 1 for count in schedule_duplicate_keys.values()
        ),
        "expected_source": expected_source,
        "expected_key_count": len(expected_keys),
        "expected_keys": _serialized_keys(expected_keys),
        "configured_key_sets_match": configured_key_sets_match,
        "configured_cohorts_match_schedule": configured_cohorts_match_schedule,
        "controller_key_sets_match": controller_key_sets_match,
        "configured_cohorts": configuration_rows,
        "controllers": controller_rows,
    }


def _iteration_row(
    *,
    track: str,
    controller: str,
    manifest: dict[str, Any],
    event: dict[str, Any],
    trace_write_seconds: float,
) -> dict[str, Any]:
    metrics = dict(event.get("metrics") or {})
    timings = dict(event.get("timings") or {})
    low = dict(event.get("low_level_delta") or {})
    controller_data = dict(event.get("controller") or {})
    v3_s3 = dict(controller_data.get("v3_s3") or {})
    neighborhood = list(metrics.get("neighborhood") or [])
    conflicts_before = int(metrics.get("conflicts_before", 0))
    conflicts_after = int(metrics.get("conflicts_after", conflicts_before))
    scored_candidates = [
        {
            "candidate_id": str(candidate.get("candidate_id")),
            "score": candidate.get("score"),
        }
        for candidate in list(controller_data.get("candidate_pool") or [])
        if candidate.get("score") is not None
    ]
    ranking = [
        row["candidate_id"]
        for row in sorted(
            scored_candidates,
            key=lambda value: (
                -round(float(value["score"]), 12),
                str(value["candidate_id"]),
            ),
        )
    ]
    row: dict[str, Any] = {
        "track": track,
        "controller": controller,
        "task_id": manifest.get("task_id"),
        "map_id": manifest.get("map_id"),
        "layout_family": manifest.get("layout_mode"),
        "agent_count": int(manifest.get("agent_count", 0)),
        "solver_seed": int(manifest.get("solver_seed", -1)),
        "decision_index": int(event.get("decision_index", -1)),
        "before_fingerprint": event.get("before_fingerprint"),
        "within_wall_budget": bool(event.get("within_wall_budget", True)),
        "elapsed_wall_seconds": _number(event.get("elapsed_wall_seconds")),
        "conflicts_before": conflicts_before,
        "conflicts_after": conflicts_after,
        "conflict_delta": conflicts_before - conflicts_after,
        "neighborhood_size": len(neighborhood),
        "replan_success": bool(metrics.get("replan_success")),
        "route": controller_data.get("route"),
        "selected_candidate_id": controller_data.get("selected_candidate_id"),
        "base_selected_candidate_id": controller_data.get(
            "base_selected_candidate_id"
        ),
        "candidate_score_fingerprint": (
            _fingerprint(scored_candidates) if scored_candidates else None
        ),
        "candidate_ranking_fingerprint": (
            _fingerprint(ranking) if ranking else None
        ),
        "actual_neighborhood_fingerprint": _fingerprint(sorted(map(int, neighborhood))),
        "repair_outcome": v3_s3.get("repair_outcome"),
        "v3_s3_selection_kind": v3_s3.get("selection_kind"),
        "v3_s3_template_id": v3_s3.get("template_id"),
        "v3_s3_continuation_expected": v3_s3.get(
            "continuation_expected"
        ),
        "v3_s3_cache_hit": bool(
            controller_data.get("v3_s3_cache_hit", False)
        ),
        "low_level_expanded": int(low.get("expanded", 0)),
        "low_level_generated": int(low.get("generated", 0)),
        "low_level_reopened": int(low.get("reopened", 0)),
        "low_level_runs": int(low.get("runs", 0)),
        "trace_write_seconds": trace_write_seconds,
        "timing_instrumented": bool(event.get("timings"))
        and event.get("native_timing_schema") in SUPPORTED_NATIVE_TIMING_SCHEMAS
        and "pp_replan_seconds" in timings
        and "neighborhood_selection_seconds" in timings,
    }
    for name in TIMING_FIELDS:
        if name == "trace_write_seconds":
            continue
        row[name] = _number(timings.get(name))
    row["iteration_with_trace_seconds"] = (
        row["iteration_wall_seconds"] + trace_write_seconds
    )
    row["conflict_reduction_per_second"] = (
        row["conflict_delta"] / row["iteration_with_trace_seconds"]
        if row["iteration_with_trace_seconds"] > 0.0
        else None
    )
    return row


def _episode_row(
    *,
    track: str,
    controller: str,
    source: dict[str, Any],
    iterations: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = dict(source.get("summary") or {})
    reset = dict(summary.get("reset_timings") or {})
    finalization = dict(source.get("episode_finalization_timings") or {})
    budget_low_level = dict(summary.get("budget_final_low_level") or {})
    v3_s3 = dict(summary.get("v3_s3") or {})
    repairable = bool(summary.get("repairable"))
    initial_conflicts = int(summary.get("initial_conflicts", 0))
    fixed_auc = summary.get("fixed_budget_conflict_auc")
    metric_iteration_budget = summary.get("metric_iteration_budget")
    if metric_iteration_budget is None and fixed_auc is not None:
        # Backward-compatible interpretation of registered historical traces.
        metric_iteration_budget = 100
    normalized_fixed_auc = summary.get("normalized_fixed_budget_conflict_auc")
    if (
        normalized_fixed_auc is None
        and fixed_auc is not None
        and initial_conflicts > 0
        and metric_iteration_budget is not None
    ):
        normalized_fixed_auc = float(fixed_auc) / (
            initial_conflicts * int(metric_iteration_budget)
        )
    reduced = max(0, initial_conflicts - int(summary.get("budget_final_conflicts", summary.get("final_conflicts", 0))))
    longest_failed_replan_streak = 0
    failed_replan_streak = 0
    for item in iterations:
        if not bool(item["replan_success"]):
            failed_replan_streak += 1
            longest_failed_replan_streak = max(
                longest_failed_replan_streak, failed_replan_streak
            )
        else:
            failed_replan_streak = 0
    failed_replan_count = sum(not bool(item["replan_success"]) for item in iterations)
    row: dict[str, Any] = {
        "track": track,
        "controller": controller,
        "task_id": source.get("task_id"),
        "map_id": source.get("map_id"),
        "layout_family": source.get("layout_mode"),
        "agent_count": int(source.get("agent_count", 0)),
        "solver_seed": int(source.get("solver_seed", -1)),
        "status": source.get("status"),
        "initial_fingerprint": summary.get("initial_fingerprint"),
        "repairable": repairable,
        "success": bool(summary.get("success")),
        "stop_reason": summary.get("stop_reason"),
        "stopping_rule": summary.get("stopping_rule"),
        "wall_time_budget_seconds": _number(summary.get("wall_time_budget_seconds")),
        "initial_conflicts": initial_conflicts,
        "budget_final_conflicts": int(summary.get("budget_final_conflicts", summary.get("final_conflicts", 0))),
        "budget_final_sum_of_costs": int(
            summary.get("budget_final_sum_of_costs", summary.get("final_sum_of_costs", 0))
        ),
        "budget_low_level_expanded": int(budget_low_level.get("expanded", 0)),
        "budget_low_level_generated": int(budget_low_level.get("generated", 0)),
        "budget_low_level_reopened": int(budget_low_level.get("reopened", 0)),
        "budget_low_level_runs": int(budget_low_level.get("runs", 0)),
        "repair_iterations": int(summary.get("repair_iterations", len(iterations))),
        "repair_iterations_within_budget": int(summary.get("repair_iterations_within_budget", len(iterations))),
        "wall_time_to_feasible": summary.get("wall_time_to_feasible"),
        "restricted_time_to_feasible": summary.get("capped_wall_time_to_feasible"),
        "wall_clock_conflict_auc": summary.get("wall_clock_conflict_auc"),
        "normalized_wall_clock_conflict_auc": summary.get("normalized_wall_clock_conflict_auc"),
        "fixed_budget_conflict_auc": fixed_auc,
        "normalized_fixed_budget_conflict_auc": normalized_fixed_auc,
        "metric_iteration_budget": metric_iteration_budget,
        "episode_observed_wall_seconds": _number(summary.get("episode_observed_wall_seconds")),
        "episode_process_wall_seconds": _number(
            finalization.get(
                "episode_process_wall_seconds",
                summary.get("episode_observed_wall_seconds"),
            )
        ),
        "post_algorithm_finalize_seconds": _number(
            finalization.get("post_algorithm_finalize_seconds")
        ),
        "finish_event_orchestration_seconds": _number(
            finalization.get("finish_event_orchestration_seconds")
        ),
        "finish_trace_write_seconds": _number(
            finalization.get("finish_trace_write_seconds")
        ),
        "trace_validation_seconds": _number(
            finalization.get("trace_validation_seconds")
        ),
        "atomic_rename_seconds": _number(
            finalization.get("atomic_rename_seconds")
        ),
        "trace_metadata_seconds": _number(
            finalization.get("trace_metadata_seconds")
        ),
        "finalization_timing_instrumented": bool(finalization),
        "budget_overshoot_seconds": _number(summary.get("budget_overshoot_seconds")),
        "environment_construct_seconds": _number(summary.get("environment_construct_seconds")),
        "reset_wall_seconds": _number(summary.get("reset_wall_seconds")),
        "initial_solution_seconds": _number(reset.get("initial_solution_seconds")),
        "reset_state_export_seconds": _number(reset.get("state_snapshot_seconds")) + _number(reset.get("state_to_python_seconds")),
        "initial_fingerprint_seconds": _number(
            summary.get("initial_fingerprint_seconds")
        ),
        "final_fingerprint_seconds": _number(summary.get("final_fingerprint_seconds")),
        "timing_unaccounted_seconds": _number(summary.get("timing_unaccounted_seconds")),
        "instrumented_iteration_count": sum(
            bool(item.get("timing_instrumented")) for item in iterations
        ),
        "timing_instrumentation_complete": len(iterations) == int(summary.get("repair_iterations", len(iterations))) and all(bool(item.get("timing_instrumented")) for item in iterations),
        "successful_replan_count": sum(bool(item["replan_success"]) for item in iterations),
        "failed_replan_count": failed_replan_count,
        "failed_replan_fraction": (
            failed_replan_count / len(iterations) if iterations else 0.0
        ),
        "longest_failed_replan_streak": longest_failed_replan_streak,
        "conflict_reducing_repair_count": sum(int(item["conflict_delta"]) > 0 for item in iterations),
        "no_improvement_repair_count": sum(int(item["conflict_delta"]) <= 0 for item in iterations),
        "conflicts_reduced_at_budget": reduced,
        "repair_low_level_expanded": sum(int(item["low_level_expanded"]) for item in iterations),
        "repair_low_level_generated": sum(int(item["low_level_generated"]) for item in iterations),
        "repair_low_level_reopened": sum(int(item["low_level_reopened"]) for item in iterations),
        "model_decision_count": int(summary.get("model_decision_count", 0)),
        "official_decision_count": int(summary.get("official_decision_count", 0)),
        "model_route_fraction": _number(summary.get("model_route_fraction")),
        "v3_s3_history_length": int(v3_s3.get("history_length", 0)),
        "v3_s3_template_unavailable_count": int(
            v3_s3.get("template_unavailable_count", 0)
        ),
    }
    for name in TIMING_FIELDS:
        row[name] = (
            _number(summary.get("trace_write_seconds"))
            + _number(finalization.get("finish_trace_write_seconds"))
            if name == "trace_write_seconds"
            else sum(float(item.get(name, 0.0)) for item in iterations)
        )
        row[f"mean_{name}"] = _mean(item.get(name) for item in iterations)
        row[f"median_{name}"] = _median(item.get(name) for item in iterations)
        row[f"p95_{name}"] = _p95(item.get(name) for item in iterations)
    row["orchestration_seconds"] = (
        row["pre_step_orchestration_seconds"]
        + row["post_step_orchestration_seconds"]
    )
    row["finalization_non_trace_seconds"] = max(
        0.0,
        row["post_algorithm_finalize_seconds"]
        - row["finish_trace_write_seconds"],
    )
    row["episode_state_fingerprint_seconds"] = (
        row["initial_fingerprint_seconds"]
        + row["state_fingerprint_seconds"]
        + row["final_fingerprint_seconds"]
    )
    row["other_runtime_seconds"] = (
        row["environment_construct_seconds"]
        + row["reset_wall_seconds"]
        + row["repair_bookkeeping_seconds"]
        + row["environment_step_residual_seconds"]
        + row["orchestration_seconds"]
        + row["finalization_non_trace_seconds"]
        + row["timing_unaccounted_seconds"]
    )
    accounted_process_seconds = sum(
        _number(row.get(name)) for name in DECOMPOSITION_FIELDS
    )
    row["process_timing_closure_error_seconds"] = abs(
        row["episode_process_wall_seconds"] - accounted_process_seconds
    )
    row["selection_seconds_per_conflict_reduced"] = (
        row["neighborhood_selection_seconds"] / reduced if reduced else None
    )
    row["pp_seconds_per_conflict_reduced"] = (
        row["pp_replan_seconds"] / reduced if reduced else None
    )
    row["total_seconds_per_conflict_reduced"] = (
        row["iteration_wall_seconds"] / reduced if reduced else None
    )
    return row


def validate_manifest_trace(
    root: Path,
    source: dict[str, Any],
    *,
    run_fingerprint: str,
    expected_policy: str,
) -> tuple[Path, list[dict[str, Any]], Path | None]:
    """Validate the immutable trace metadata and episode identity in a manifest row."""

    if not run_fingerprint:
        raise ValueError("collection run fingerprint is missing")
    trace_path = contained_file(
        root, source.get("trace_file"), field="trace_file"
    )
    expected_sha256 = str(source.get("trace_sha256") or "")
    if not expected_sha256:
        raise ValueError(f"trace SHA256 is missing: {source.get('episode_id')}")
    actual_sha256 = sha256_file(trace_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"trace SHA256 mismatch: {source.get('episode_id')}")
    try:
        expected_bytes = int(source["trace_bytes"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"trace byte count is missing or invalid: {source.get('episode_id')}"
        ) from error
    if trace_path.stat().st_size != expected_bytes:
        raise ValueError(f"trace byte count mismatch: {source.get('episode_id')}")

    episode_id = str(source.get("episode_id") or "")
    policy = str(source.get("policy") or "")
    if not episode_id:
        raise ValueError("manifest episode id is missing")
    if policy != expected_policy:
        raise ValueError(f"manifest policy mismatch: {episode_id}")
    try:
        solver_seed = int(source["solver_seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"manifest solver seed is invalid: {episode_id}") from error

    preflight_events = read_trace_events(trace_path)
    if len(preflight_events) < 2:
        raise ValueError(f"trace must contain initial and finish events: {episode_id}")
    preflight_initial = preflight_events[0]
    manifest_state_ref = source.get("initial_state_ref")
    event_state_ref = preflight_initial.get("state_blob")
    blob_path: Path | None = None
    if manifest_state_ref is not None or event_state_ref is not None:
        manifest_reference = (
            manifest_state_ref if isinstance(manifest_state_ref, str) else ""
        )
        event_reference = event_state_ref if isinstance(event_state_ref, str) else ""
        if (
            not manifest_reference
            or not event_reference
            or manifest_reference != event_reference
        ):
            raise ValueError(
                f"manifest and trace state blob references differ: {episode_id}"
            )
        blob_path = contained_file(
            root, event_reference, field="initial_state_ref"
        )
    manifest_summary = source.get("summary")
    metric_iteration_budget = (
        manifest_summary.get("metric_iteration_budget")
        if isinstance(manifest_summary, dict)
        else None
    )
    validation_events: list[dict[str, Any]] = []
    legacy_timing_view = False
    for event in preflight_events:
        validation_event = dict(event)
        if (
            event.get("event") == "transition"
            and event.get("native_timing_schema") == "lns2.repair_timing.v1"
            and isinstance(event.get("timings"), dict)
            and isinstance(event.get("metrics"), dict)
        ):
            timings = dict(event["timings"])
            metrics = dict(event["metrics"])
            raw_step_runtime = metrics.get("step_runtime")
            legacy_negative_step_runtime = bool(
                isinstance(raw_step_runtime, (int, float))
                and not isinstance(raw_step_runtime, bool)
                and math.isfinite(float(raw_step_runtime))
                and float(raw_step_runtime) < 0.0
            )
            derived = {
                "native_step_seconds": metrics.get("native_step_seconds"),
                "episode_runtime_delta_seconds": metrics.get(
                    "episode_runtime_delta_seconds",
                    None if legacy_negative_step_runtime else raw_step_runtime,
                ),
            }
            for name, value in derived.items():
                if name not in timings and value is not None:
                    timings[name] = value
                    legacy_timing_view = True
            validation_event["timings"] = timings
            # Three authenticated pilot traces predate the native timing v2
            # contract and contain a negative legacy ``step_runtime`` even
            # though their independently recorded native timing partition is
            # complete and non-negative.  Validate a derived compatibility
            # view against that native partition while returning the original
            # immutable events to the caller.  Current v2 traces and any v1
            # trace without valid native evidence remain strict failures.
            native_step = timings.get("native_step_seconds")
            if (
                legacy_negative_step_runtime
                and isinstance(native_step, (int, float))
                and not isinstance(native_step, bool)
                and math.isfinite(float(native_step))
                and float(native_step) >= 0.0
            ):
                metrics["step_runtime"] = float(native_step)
                if (
                    isinstance(timings.get("episode_runtime_delta_seconds"), (int, float))
                    and float(timings["episode_runtime_delta_seconds"]) < 0.0
                ):
                    timings.pop("episode_runtime_delta_seconds")
                    validation_event["timings"] = timings
                validation_event["metrics"] = metrics
                legacy_timing_view = True
        validation_events.append(validation_event)

    def validate_trace(validation_path: Path) -> dict[str, Any]:
        return validate_closed_loop_trace(
            validation_path,
            run_fingerprint,
            expected_episode_id=episode_id,
            expected_policy=expected_policy,
            expected_solver_seed=solver_seed,
            metric_iteration_budget=(
                int(metric_iteration_budget)
                if metric_iteration_budget is not None
                else None
            ),
            collection_root=root,
        )

    try:
        if legacy_timing_view:
            with tempfile.TemporaryDirectory(
                prefix="lns2-trace-validation-"
            ) as directory:
                # The authenticated source may be gzip-compressed, but this
                # derived validation view need not be; schema markers determine
                # full-v1 versus compact semantics.
                validation_path = Path(directory) / "trace.jsonl"
                with open_trace_text(validation_path, "w") as stream:
                    for event in validation_events:
                        stream.write(
                            json.dumps(
                                event,
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                validated = validate_trace(validation_path)
        else:
            validated = validate_trace(trace_path)
    except ClosedLoopTraceError as error:
        raise ValueError(f"trace validation failed: {episode_id}: {error}") from error
    events = preflight_events
    initial = events[0]
    finish = events[-1]
    validated_trace_format = str(validated["trace_format"])
    expected_storage_fingerprint = storage_fingerprint(validated_trace_format)
    manifest_trace_format = source.get("trace_format")
    manifest_storage_fingerprint = source.get("storage_fingerprint")
    if validated_trace_format == TRACE_FORMAT_DELTA_GZIP_V2:
        if str(manifest_trace_format or "") != TRACE_FORMAT_DELTA_GZIP_V2:
            raise ValueError(f"compact trace format mismatch: {episode_id}")
        if str(manifest_storage_fingerprint or "") != expected_storage_fingerprint:
            raise ValueError(f"compact trace storage fingerprint mismatch: {episode_id}")
        if blob_path is None or str(validated.get("initial_state_ref") or "") != str(
            manifest_state_ref or ""
        ):
            raise ValueError(
                f"manifest and trace state blob references differ: {episode_id}"
            )
    elif validated_trace_format == TRACE_FORMAT_FULL_V1:
        if manifest_trace_format not in {None, "", TRACE_FORMAT_FULL_V1}:
            raise ValueError(f"full trace format mismatch: {episode_id}")
        if manifest_storage_fingerprint not in {
            None,
            "",
            expected_storage_fingerprint,
        }:
            raise ValueError(f"full trace storage fingerprint mismatch: {episode_id}")
        if any(
            "trace_format" in event
            and str(event.get("trace_format") or "") != TRACE_FORMAT_FULL_V1
            for event in events
        ):
            raise ValueError(f"full trace event format mismatch: {episode_id}")
        if any(
            "storage_fingerprint" in event
            and str(event.get("storage_fingerprint") or "")
            != expected_storage_fingerprint
            for event in events
        ):
            raise ValueError(
                f"full trace event storage fingerprint mismatch: {episode_id}"
            )
    else:
        raise ValueError(f"unsupported validated trace format: {validated_trace_format}")
    if any(str(event.get("run_fingerprint") or "") != run_fingerprint for event in events):
        raise ValueError(f"trace run fingerprint mismatch: {episode_id}")
    if any(str(event.get("episode_id") or "") != episode_id for event in events):
        raise ValueError(f"trace episode id mismatch: {episode_id}")
    if any(
        "policy" in event and str(event.get("policy") or "") != expected_policy
        for event in events
    ):
        raise ValueError(f"trace policy mismatch: {episode_id}")
    if str(initial.get("policy") or "") != expected_policy or str(
        finish.get("policy") or ""
    ) != expected_policy:
        raise ValueError(f"trace policy mismatch: {episode_id}")
    try:
        trace_seed = int(initial.get("solver_seed"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"trace solver seed is invalid: {episode_id}") from error
    if trace_seed != solver_seed:
        raise ValueError(f"trace solver seed mismatch: {episode_id}")
    for event in events:
        if "solver_seed" not in event:
            continue
        try:
            event_seed = int(event["solver_seed"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"trace solver seed is invalid: {episode_id}") from error
        if event_seed != solver_seed:
            raise ValueError(f"trace solver seed mismatch: {episode_id}")
    task_id = str(source.get("task_id") or "")
    if not task_id:
        raise ValueError(f"manifest task id is missing: {episode_id}")
    if any(
        "task_id" in event and str(event.get("task_id") or "") != task_id
        for event in events
    ):
        raise ValueError(f"trace task id mismatch: {episode_id}")
    try:
        expected_event_count = int(source["trace_event_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"trace event count is missing or invalid: {episode_id}") from error
    if expected_event_count != len(events):
        raise ValueError(f"trace event count mismatch: {episode_id}")
    finish_summary = finish.get("summary")
    if not isinstance(manifest_summary, dict) or not isinstance(finish_summary, dict):
        raise ValueError(f"trace summary is missing or invalid: {episode_id}")
    if manifest_summary != finish_summary:
        raise ValueError(f"trace summary mismatch: {episode_id}")
    return trace_path, events, blob_path


def load_track(
    track: str, roots: dict[str, Path]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for controller in roots:
        root = Path(roots[controller]).resolve()
        run = _read_json(root / "run_config.json")
        configuration = dict(run.get("configuration") or {})
        environment = dict(configuration.get("environment") or {})
        implementation = dict(run.get("controller_implementation") or {})
        native_module = dict(implementation.get("native_module") or {})
        run_fingerprint = str(run.get("run_fingerprint") or "")
        if not run_fingerprint:
            raise ValueError(f"collection run fingerprint is missing: {root}")
        metadata[controller] = {
            "root": str(root),
            "run_fingerprint": run_fingerprint,
            "dataset_fingerprint": run.get("dataset_fingerprint"),
            "stopping_rule": configuration.get("stopping_rule"),
            "replan_algorithm": environment.get("replan_algorithm"),
            "use_sipp": environment.get("use_sipp"),
            "native_module_sha256": native_module.get("sha256"),
            "configured_cohort": _configured_cohort(run),
        }
        for source in _read_jsonl(_controller_manifest_path(root, controller)):
            if str(source.get("status")) not in {"ok", "resumed"}:
                episodes.append(
                    {
                        "track": track,
                        "controller": controller,
                        "task_id": source.get("task_id"),
                        "solver_seed": source.get("solver_seed"),
                        "status": source.get("status"),
                    }
                )
                continue
            expected_policy = (
                "official_adaptive"
                if controller == "official_adaptive"
                else "realized_dynamic"
            )
            _trace_path, events, _blob_path = validate_manifest_trace(
                root,
                source,
                run_fingerprint=run_fingerprint,
                expected_policy=expected_policy,
            )
            transitions = [event for event in events if event.get("event") == "transition"]
            summary = dict(source.get("summary") or {})
            trace_times = list(summary.get("transition_trace_write_seconds") or [])
            if trace_times and len(trace_times) != len(transitions):
                raise ValueError(f"trace-write timing length mismatch: {source.get('episode_id')}")
            episode_iterations = [
                _iteration_row(
                    track=track,
                    controller=controller,
                    manifest=source,
                    event=event,
                    trace_write_seconds=(
                        _number(trace_times[index]) if trace_times else 0.0
                    ),
                )
                for index, event in enumerate(transitions)
            ]
            iterations.extend(episode_iterations)
            episodes.append(
                _episode_row(
                    track=track,
                    controller=controller,
                    source=source,
                    iterations=episode_iterations,
                )
            )
    return episodes, iterations, metadata


def paired_decomposition(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (str(row["track"]), str(row["controller"]), str(row.get("task_id")), int(row.get("solver_seed", -1))): row
        for row in episodes
        if row.get("status") in {"ok", "resumed"}
    }
    keys = sorted({(key[0], key[2], key[3]) for key in indexed})
    result: list[dict[str, Any]] = []
    for track, task_id, seed in keys:
        left = indexed.get((track, "official_adaptive", task_id, seed))
        right = indexed.get((track, "v2-full", task_id, seed))
        if left is None or right is None:
            continue
        row: dict[str, Any] = {
            "track": track,
            "task_id": task_id,
            "map_id": left.get("map_id"),
            "layout_family": left.get("layout_family"),
            "agent_count": left.get("agent_count"),
            "solver_seed": seed,
            "initial_fingerprint": left.get("initial_fingerprint"),
            "initial_fingerprint_match": left.get("initial_fingerprint") == right.get("initial_fingerprint"),
            "lns2_success": left.get("success"),
            "v2_success": right.get("success"),
            "common_success": bool(left.get("success") and right.get("success")),
            "lns2_repair_iterations": left.get("repair_iterations_within_budget"),
            "v2_repair_iterations": right.get("repair_iterations_within_budget"),
            "repair_iteration_delta_v2_minus_lns2": int(right.get("repair_iterations_within_budget", 0)) - int(left.get("repair_iterations_within_budget", 0)),
            "lns2_attempted_repair_iterations": left.get("repair_iterations"),
            "v2_attempted_repair_iterations": right.get("repair_iterations"),
            "attempted_repair_iteration_delta_v2_minus_lns2": int(right.get("repair_iterations", 0)) - int(left.get("repair_iterations", 0)),
            "lns2_time_to_feasible": left.get("wall_time_to_feasible"),
            "v2_time_to_feasible": right.get("wall_time_to_feasible"),
            "lns2_restricted_time_to_feasible": left.get("restricted_time_to_feasible"),
            "v2_restricted_time_to_feasible": right.get("restricted_time_to_feasible"),
            "lns2_normalized_wall_clock_conflict_auc": left.get("normalized_wall_clock_conflict_auc"),
            "v2_normalized_wall_clock_conflict_auc": right.get("normalized_wall_clock_conflict_auc"),
            "normalized_wall_clock_conflict_auc_delta_v2_minus_lns2": _number(right.get("normalized_wall_clock_conflict_auc")) - _number(left.get("normalized_wall_clock_conflict_auc")),
            "lns2_fixed_budget_conflict_auc": left.get("fixed_budget_conflict_auc"),
            "v2_fixed_budget_conflict_auc": right.get("fixed_budget_conflict_auc"),
            "fixed_budget_conflict_auc_delta_v2_minus_lns2": (
                _number(right.get("fixed_budget_conflict_auc"))
                - _number(left.get("fixed_budget_conflict_auc"))
            ),
            "lns2_normalized_fixed_budget_conflict_auc": left.get(
                "normalized_fixed_budget_conflict_auc"
            ),
            "v2_normalized_fixed_budget_conflict_auc": right.get(
                "normalized_fixed_budget_conflict_auc"
            ),
            "normalized_fixed_budget_conflict_auc_delta_v2_minus_lns2": (
                _number(right.get("normalized_fixed_budget_conflict_auc"))
                - _number(left.get("normalized_fixed_budget_conflict_auc"))
            ),
            "lns2_budget_final_conflicts": left.get("budget_final_conflicts"),
            "v2_budget_final_conflicts": right.get("budget_final_conflicts"),
            "budget_final_conflict_delta_v2_minus_lns2": int(right.get("budget_final_conflicts", 0)) - int(left.get("budget_final_conflicts", 0)),
            "lns2_budget_final_sum_of_costs": left.get("budget_final_sum_of_costs"),
            "v2_budget_final_sum_of_costs": right.get("budget_final_sum_of_costs"),
            "budget_final_soc_delta_v2_minus_lns2": int(right.get("budget_final_sum_of_costs", 0)) - int(left.get("budget_final_sum_of_costs", 0)),
        }
        for name in DECOMPOSITION_FIELDS:
            left_value = _number(left.get(name))
            right_value = _number(right.get(name))
            row[f"lns2_{name}"] = left_value
            row[f"v2_{name}"] = right_value
            row[f"delta_{name}_v2_minus_lns2"] = right_value - left_value
        lns2_loops = int(left.get("repair_iterations", 0))
        v2_loops = int(right.get("repair_iterations", 0))
        loops_saved = lns2_loops - v2_loops
        lns2_loop_seconds = _number(left.get("mean_iteration_wall_seconds"))
        lns2_repair_work_per_loop = sum(
            _number(left.get(f"mean_{name}"))
            for name in (
                "pp_replan_seconds",
                "repair_bookkeeping_seconds",
                "state_export_seconds",
                "environment_step_residual_seconds",
            )
        )
        row.update(
            {
                "repair_loops_saved_by_v2": loops_saved,
                "estimated_loop_wall_seconds_saved_by_v2": loops_saved
                * lns2_loop_seconds,
                "estimated_repair_work_seconds_saved_by_v2": loops_saved
                * lns2_repair_work_per_loop,
                "additional_v2_selection_seconds": _number(
                    right.get("neighborhood_selection_seconds")
                )
                - _number(left.get("neighborhood_selection_seconds")),
                "v2_pp_total_delta_seconds": _number(right.get("pp_replan_seconds"))
                - _number(left.get("pp_replan_seconds")),
                "v2_iteration_wall_delta_seconds": _number(
                    right.get("iteration_wall_seconds")
                )
                - _number(left.get("iteration_wall_seconds")),
            }
        )
        result.append(row)
    return result


PAIRWISE_METRICS = (
    "normalized_wall_clock_conflict_auc",
    "normalized_fixed_budget_conflict_auc",
    "restricted_time_to_feasible",
    "budget_final_conflicts",
    "budget_final_sum_of_costs",
    "repair_iterations",
    "neighborhood_selection_seconds",
    "pp_replan_seconds",
    "iteration_wall_seconds",
    "failed_replan_count",
)


def _summary_rows(
    episodes: list[dict[str, Any]], iterations: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    controller_names = {str(row["controller"]) for row in episodes}
    successful_controllers: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in episodes:
        if row.get("status") in {"ok", "resumed"} and row.get("success"):
            successful_controllers[
                (str(row["track"]), str(row.get("task_id")), int(row.get("solver_seed", -1)))
            ].add(str(row["controller"]))
    common_success_keys = {
        key for key, controllers in successful_controllers.items()
        if controller_names.issubset(controllers)
    }
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        if row.get("status") not in {"ok", "resumed"}:
            continue
        track = str(row["track"])
        controller = str(row["controller"])
        groups[(track, controller, "all", "all")].append(row)
        groups[(track, controller, "map_id", str(row.get("map_id")))].append(row)
        groups[(track, controller, "layout_family", str(row.get("layout_family")))].append(row)
        groups[(track, controller, "agent_count", str(row.get("agent_count")))].append(row)
        episode_key = (track, str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if episode_key in common_success_keys:
            groups[(track, controller, "common_success", "all")].append(row)
    iteration_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in iterations:
        track = str(row["track"])
        controller = str(row["controller"])
        keys = [
            (track, controller, "all", "all"),
            (track, controller, "map_id", str(row.get("map_id"))),
            (track, controller, "layout_family", str(row.get("layout_family"))),
            (track, controller, "agent_count", str(row.get("agent_count"))),
        ]
        episode_key = (track, str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if episode_key in common_success_keys:
            keys.append((track, controller, "common_success", "all"))
        for key in keys:
            iteration_groups[key].append(row)
    result: list[dict[str, Any]] = []
    for (track, controller, group_type, group_value), rows in sorted(groups.items()):
        repairable = [row for row in rows if row.get("repairable")]
        loop_rows = iteration_groups[(track, controller, group_type, group_value)]
        result.append(
            {
                "track": track,
                "controller": controller,
                "group_type": group_type,
                "group_value": group_value,
                "episode_count": len(rows),
                "repairable_episode_count": len(repairable),
                "success_count": sum(bool(row.get("success")) for row in rows),
                "total_repair_iterations": sum(int(row.get("repair_iterations", 0)) for row in repairable),
                "mean_repair_iterations": _mean(row.get("repair_iterations") for row in repairable),
                "median_repair_iterations": _median(row.get("repair_iterations") for row in repairable),
                "p95_repair_iterations": _p95(row.get("repair_iterations") for row in repairable),
                "mean_repairs_completed_within_budget": _mean(row.get("repair_iterations_within_budget") for row in repairable),
                "mean_restricted_time_to_feasible": _mean(row.get("restricted_time_to_feasible") for row in repairable),
                "mean_wall_time_to_feasible_on_success": _mean(row.get("wall_time_to_feasible") for row in repairable if row.get("success")),
                "mean_normalized_wall_clock_conflict_auc": _mean(row.get("normalized_wall_clock_conflict_auc") for row in repairable),
                "mean_fixed_budget_conflict_auc": _mean(
                    row.get("fixed_budget_conflict_auc") for row in repairable
                ),
                "mean_normalized_fixed_budget_conflict_auc": _mean(
                    row.get("normalized_fixed_budget_conflict_auc")
                    for row in repairable
                ),
                "mean_budget_final_sum_of_costs": _mean(
                    row.get("budget_final_sum_of_costs") for row in repairable
                ),
                **{
                    f"mean_total_{name}": _mean(row.get(name) for row in repairable)
                    for name in dict.fromkeys(DECOMPOSITION_FIELDS + TIMING_FIELDS)
                },
                "successful_replan_count": sum(int(row.get("successful_replan_count", 0)) for row in repairable),
                "failed_replan_count": sum(int(row.get("failed_replan_count", 0)) for row in repairable),
                "failed_replan_fraction": (
                    sum(int(row.get("failed_replan_count", 0)) for row in repairable)
                    / sum(int(row.get("repair_iterations", 0)) for row in repairable)
                    if sum(int(row.get("repair_iterations", 0)) for row in repairable)
                    else 0.0
                ),
                "mean_longest_failed_replan_streak": _mean(
                    row.get("longest_failed_replan_streak") for row in repairable
                ),
                "mean_v3_s3_history_length": _mean(
                    row.get("v3_s3_history_length") for row in repairable
                ),
                "v3_s3_template_unavailable_count": sum(
                    int(row.get("v3_s3_template_unavailable_count", 0))
                    for row in repairable
                ),
                "conflict_reducing_repair_count": sum(int(row.get("conflict_reducing_repair_count", 0)) for row in repairable),
                "no_improvement_repair_count": sum(int(row.get("no_improvement_repair_count", 0)) for row in repairable),
                "mean_iteration_selection_seconds": _mean(row.get("neighborhood_selection_seconds") for row in loop_rows),
                "median_iteration_selection_seconds": _median(row.get("neighborhood_selection_seconds") for row in loop_rows),
                "p95_iteration_selection_seconds": _p95(row.get("neighborhood_selection_seconds") for row in loop_rows),
                "mean_iteration_pp_seconds": _mean(row.get("pp_replan_seconds") for row in loop_rows),
                "median_iteration_pp_seconds": _median(row.get("pp_replan_seconds") for row in loop_rows),
                "p95_iteration_pp_seconds": _p95(row.get("pp_replan_seconds") for row in loop_rows),
                "mean_iteration_wall_seconds": _mean(row.get("iteration_wall_seconds") for row in loop_rows),
                "median_iteration_wall_seconds": _median(row.get("iteration_wall_seconds") for row in loop_rows),
                "p95_iteration_wall_seconds": _p95(row.get("iteration_wall_seconds") for row in loop_rows),
                "mean_conflict_delta_per_repair": _mean(row.get("conflict_delta") for row in loop_rows),
                "median_conflict_delta_per_repair": _median(row.get("conflict_delta") for row in loop_rows),
                "p95_conflict_delta_per_repair": _p95(row.get("conflict_delta") for row in loop_rows),
                "mean_neighborhood_size": _mean(row.get("neighborhood_size") for row in loop_rows),
                "mean_low_level_expanded_per_repair": _mean(row.get("low_level_expanded") for row in loop_rows),
                "median_low_level_expanded_per_repair": _median(row.get("low_level_expanded") for row in loop_rows),
                "p95_low_level_expanded_per_repair": _p95(row.get("low_level_expanded") for row in loop_rows),
                "mean_low_level_generated_per_repair": _mean(row.get("low_level_generated") for row in loop_rows),
                "median_low_level_generated_per_repair": _median(row.get("low_level_generated") for row in loop_rows),
                "p95_low_level_generated_per_repair": _p95(row.get("low_level_generated") for row in loop_rows),
                "mean_low_level_reopened_per_repair": _mean(row.get("low_level_reopened") for row in loop_rows),
                "median_low_level_reopened_per_repair": _median(row.get("low_level_reopened") for row in loop_rows),
                "p95_low_level_reopened_per_repair": _p95(row.get("low_level_reopened") for row in loop_rows),
                "mean_selection_seconds_per_conflict_reduced": _mean(row.get("selection_seconds_per_conflict_reduced") for row in repairable),
                "mean_pp_seconds_per_conflict_reduced": _mean(row.get("pp_seconds_per_conflict_reduced") for row in repairable),
                "mean_total_seconds_per_conflict_reduced": _mean(row.get("total_seconds_per_conflict_reduced") for row in repairable),
                "mean_episode_state_fingerprint_seconds": _mean(
                    row.get("episode_state_fingerprint_seconds") for row in repairable
                ),
                "mean_total_other_runtime_seconds": _mean(
                    row.get("other_runtime_seconds") for row in repairable
                ),
                "neighborhood_size_pp_time_correlation": _correlation(
                    (row.get("neighborhood_size") for row in loop_rows),
                    (row.get("pp_replan_seconds") for row in loop_rows),
                ),
            }
        )
    return result


def _neighborhood_pp_rows(iterations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in iterations:
        groups[
            (str(row["track"]), str(row["controller"]), int(row["neighborhood_size"]))
        ].append(row)
    return [
        {
            "track": track,
            "controller": controller,
            "neighborhood_size": size,
            "repair_count": len(rows),
            "mean_pp_replan_seconds": _mean(row.get("pp_replan_seconds") for row in rows),
            "median_pp_replan_seconds": _median(row.get("pp_replan_seconds") for row in rows),
            "p95_pp_replan_seconds": _p95(row.get("pp_replan_seconds") for row in rows),
            "mean_low_level_expanded": _mean(row.get("low_level_expanded") for row in rows),
            "mean_conflict_delta": _mean(row.get("conflict_delta") for row in rows),
        }
        for (track, controller, size), rows in sorted(groups.items())
    ]


def _conflicts_at_time(
    episode: dict[str, Any], iterations: list[dict[str, Any]], seconds: float
) -> int:
    conflicts = int(episode.get("initial_conflicts", 0))
    for row in sorted(iterations, key=lambda value: float(value["elapsed_wall_seconds"])):
        if float(row["elapsed_wall_seconds"]) > seconds:
            break
        conflicts = int(row["conflicts_after"])
    return conflicts


def long_horizon_diagnostics(
    episodes: list[dict[str, Any]], iterations: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[list[Any]]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in iterations:
        grouped[
            (
                str(row["track"]),
                str(row["controller"]),
                str(row["task_id"]),
                int(row["solver_seed"]),
            )
        ].append(row)
    checkpoints: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    extension_keys: set[tuple[str, int]] = set()
    for episode in episodes:
        if (
            episode.get("status") not in {"ok", "resumed"}
            or episode.get("stopping_rule") != "wall-clock"
        ):
            continue
        budget = float(episode.get("wall_time_budget_seconds", 0.0))
        if budget < 600.0:
            continue
        key = (
            str(episode["track"]),
            str(episode["controller"]),
            str(episode["task_id"]),
            int(episode["solver_seed"]),
        )
        rows = sorted(grouped.get(key, []), key=lambda value: float(value["elapsed_wall_seconds"]))
        for checkpoint in (300.0, 600.0, 1200.0, 1800.0, 3600.0):
            if checkpoint > budget + 1e-9:
                continue
            observed = [row for row in rows if float(row["elapsed_wall_seconds"]) <= checkpoint]
            checkpoints.append(
                {
                    "track": episode["track"],
                    "controller": episode["controller"],
                    "task_id": episode["task_id"],
                    "map_id": episode.get("map_id"),
                    "agent_count": episode.get("agent_count"),
                    "solver_seed": episode["solver_seed"],
                    "checkpoint_seconds": checkpoint,
                    "conflicts": _conflicts_at_time(episode, rows, checkpoint),
                    "repair_iteration_count": len(observed),
                    "successful_replan_count": sum(
                        bool(row["replan_success"]) for row in observed
                    ),
                    "failed_replan_count": sum(
                        not bool(row["replan_success"]) for row in observed
                    ),
                    "solved_by_checkpoint": bool(
                        episode.get("success")
                        and float(episode.get("wall_time_to_feasible") or math.inf)
                        <= checkpoint
                    ),
                }
            )
        window_start = max(0.0, budget - 600.0)
        start_conflicts = _conflicts_at_time(episode, rows, window_start)
        end_conflicts = _conflicts_at_time(episode, rows, budget)
        window_rows = [
            row
            for row in rows
            if window_start < float(row["elapsed_wall_seconds"]) <= budget
        ]
        improvement_fraction = (
            (start_conflicts - end_conflicts) / start_conflicts
            if start_conflicts > 0
            else 0.0
        )
        failure_fraction = (
            sum(not bool(row["replan_success"]) for row in window_rows)
            / len(window_rows)
            if window_rows
            else 1.0
        )
        plateau = bool(
            not episode.get("success")
            and improvement_fraction < 0.01
            and failure_fraction > 0.95
        )
        extend = bool(
            budget >= 1800.0
            and budget < 3600.0
            and not episode.get("success")
            and improvement_fraction >= 0.01
        )
        if extend:
            extension_keys.add((str(episode["task_id"]), int(episode["solver_seed"])))
        diagnostics.append(
            {
                "track": episode["track"],
                "controller": episode["controller"],
                "task_id": episode["task_id"],
                "map_id": episode.get("map_id"),
                "agent_count": episode.get("agent_count"),
                "solver_seed": episode["solver_seed"],
                "budget_seconds": budget,
                "success": bool(episode.get("success")),
                "last_600_start_conflicts": start_conflicts,
                "last_600_final_conflicts": end_conflicts,
                "last_600_conflict_improvement_fraction": improvement_fraction,
                "last_600_repair_count": len(window_rows),
                "last_600_pp_failure_fraction": failure_fraction,
                "plateau": plateau,
                "extension_to_3600_recommended": extend,
            }
        )
    return checkpoints, diagnostics, [list(value) for value in sorted(extension_keys)]


def _sensitivity_rows(
    paired: list[dict[str, Any]], primary_track: str
) -> list[dict[str, Any]]:
    primary = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in paired
        if row["track"] == primary_track
    }
    result: list[dict[str, Any]] = []
    sensitivity_tracks = sorted(
        {
            str(row["track"])
            for row in paired
            if str(row["track"]).startswith("wall-clock-")
            and str(row["track"]) != primary_track
        }
    )
    for track in sensitivity_tracks:
        rows = [row for row in paired if row["track"] == track]
        for controller, prefix in (
            ("official_adaptive", "lns2"),
            ("v2-full", "v2"),
        ):
            comparisons = [
                (primary.get((str(row["task_id"]), int(row["solver_seed"]))), row)
                for row in rows
            ]
            comparisons = [
                (base, extended)
                for base, extended in comparisons
                if base is not None
            ]
            result.append(
                {
                    "sensitivity_track": track,
                    "controller": controller,
                    "selected_pair_count": len(comparisons),
                    "primary_success_count": sum(
                        bool(base[f"{prefix}_success"])
                        for base, _extended in comparisons
                    ),
                    "sensitivity_success_count": sum(
                        bool(extended[f"{prefix}_success"])
                        for _base, extended in comparisons
                    ),
                    "new_success_count": sum(
                        not bool(base[f"{prefix}_success"])
                        and bool(extended[f"{prefix}_success"])
                        for base, extended in comparisons
                    ),
                    "lost_success_count": sum(
                        bool(base[f"{prefix}_success"])
                        and not bool(extended[f"{prefix}_success"])
                        for base, extended in comparisons
                    ),
                    "initial_fingerprint_mismatch_count": sum(
                        str(base.get("initial_fingerprint"))
                        != str(extended.get("initial_fingerprint"))
                        for base, extended in comparisons
                    ),
                    "mean_budget_final_conflict_change": _mean(
                        int(extended[f"{prefix}_budget_final_conflicts"])
                        - int(base[f"{prefix}_budget_final_conflicts"])
                        for base, extended in comparisons
                    ),
                }
            )
    return result


def _svg_frame(title: str, body: str, width: int = 900, height: int = 480) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="30" y="32" font-family="sans-serif" font-size="20">{html.escape(title)}</text>'
        f'{body}</svg>\n'
    )


def _stacked_timing_svg(path: Path, rows: list[dict[str, Any]], track: str) -> None:
    selected = {
        str(row["controller"]): row
        for row in rows
        if row["track"] == track and row["group_type"] == "all"
    }
    components = (
        ("selection", "mean_total_neighborhood_selection_seconds", "#4c78a8"),
        ("PP", "mean_total_pp_replan_seconds", "#f58518"),
        ("state export", "mean_total_state_export_seconds", "#54a24b"),
        ("trace", "mean_total_trace_write_seconds", "#72b7b2"),
        ("other", "mean_total_other_runtime_seconds", "#b279a2"),
    )
    maximum = max(
        (sum(_number(row.get(field)) for _label, field, _color in components) for row in selected.values()),
        default=1.0,
    ) or 1.0
    body = []
    controllers = sorted(selected, key=lambda value: (value not in CONTROLLERS, value))
    for index, controller in enumerate(controllers):
        row = selected.get(controller, {})
        y = 95 + index * 115
        x = 180.0
        body.append(f'<text x="20" y="{y + 24}" font-family="sans-serif" font-size="14">{html.escape(LABELS.get(controller, controller))}</text>')
        for label, field, color in components:
            value = _number(row.get(field))
            width = 650.0 * value / maximum
            body.append(f'<rect x="{x:.2f}" y="{y}" width="{width:.2f}" height="34" fill="{color}"/>')
            x += width
        body.append(f'<text x="{x + 8:.2f}" y="{y + 23}" font-family="sans-serif" font-size="12">{sum(_number(row.get(field)) for _l, field, _c in components):.3f}s</text>')
    for index, (label, _field, color) in enumerate(components):
        x = 45 + index * 140
        body.append(f'<rect x="{x}" y="380" width="16" height="16" fill="{color}"/><text x="{x + 22}" y="393" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    path.write_text(_svg_frame(f"Timing breakdown: {track}", "".join(body)), encoding="utf-8")


def _loop_svg(path: Path, rows: list[dict[str, Any]], track: str) -> None:
    selected = [row for row in rows if row["track"] == track and row["group_type"] == "all"]
    body = []
    for index, row in enumerate(selected):
        x = 180 + index * 300
        iterations = _number(row.get("mean_repair_iterations"))
        loop_time = _number(row.get("mean_iteration_wall_seconds"))
        body.append(f'<text x="{x}" y="90" font-family="sans-serif" font-size="14">{html.escape(LABELS.get(str(row["controller"]), str(row["controller"])))}</text>')
        body.append(f'<rect x="{x}" y="{330 - min(230, iterations * 4):.2f}" width="70" height="{min(230, iterations * 4):.2f}" fill="#4c78a8"/><text x="{x}" y="355" font-family="sans-serif" font-size="12">loops {iterations:.2f}</text>')
        body.append(f'<rect x="{x + 100}" y="{330 - min(230, loop_time * 90):.2f}" width="70" height="{min(230, loop_time * 90):.2f}" fill="#f58518"/><text x="{x + 90}" y="375" font-family="sans-serif" font-size="12">loop {loop_time:.3f}s</text>')
    path.write_text(_svg_frame(f"Loop count and per-loop time: {track}", "".join(body)), encoding="utf-8")


def _scatter_svg(path: Path, iterations: list[dict[str, Any]], track: str) -> None:
    rows = [row for row in iterations if row["track"] == track]
    max_size = max((int(row["neighborhood_size"]) for row in rows), default=1)
    max_pp = max((_number(row["pp_replan_seconds"]) for row in rows), default=1.0) or 1.0
    colors = CONTROLLER_COLORS
    body = ['<line x1="70" y1="420" x2="860" y2="420" stroke="black"/><line x1="70" y1="60" x2="70" y2="420" stroke="black"/>']
    for row in rows:
        x = 70 + 780 * int(row["neighborhood_size"]) / max_size
        y = 420 - 350 * _number(row["pp_replan_seconds"]) / max_pp
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{colors.get(str(row["controller"]), "#999999")}" fill-opacity="0.55"/>')
    body.append(f'<text x="360" y="455" font-family="sans-serif" font-size="13">neighborhood size (max {max_size})</text><text x="8" y="55" font-family="sans-serif" font-size="12">PP seconds (max {max_pp:.3f})</text>')
    path.write_text(_svg_frame(f"Neighborhood size vs PP time: {track}", "".join(body)), encoding="utf-8")


def _conflict_curve_svg(path: Path, episodes: list[dict[str, Any]], iterations: list[dict[str, Any]], track: str) -> None:
    episode_index = {
        (str(row["controller"]), str(row["task_id"]), int(row["solver_seed"])): row
        for row in episodes
        if row["track"] == track and row.get("repairable")
    }
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in iterations:
        if row["track"] == track:
            grouped[(str(row["controller"]), str(row["task_id"]), int(row["solver_seed"]))].append(row)
    samples = [index / 20.0 for index in range(21)]
    colors = CONTROLLER_COLORS
    body = ['<line x1="70" y1="420" x2="860" y2="420" stroke="black"/><line x1="70" y1="60" x2="70" y2="420" stroke="black"/>']
    controllers = sorted(
        {str(row["controller"]) for row in episodes if row["track"] == track},
        key=lambda value: (value not in CONTROLLERS, value),
    )
    for controller in controllers:
        curves = []
        for key, episode in episode_index.items():
            if key[0] != controller or int(episode.get("initial_conflicts", 0)) <= 0:
                continue
            budget = _number(episode.get("wall_time_budget_seconds"), 1.0) or 1.0
            points = sorted(grouped.get(key, []), key=lambda row: int(row["decision_index"]))
            values = []
            for sample in samples:
                deadline = sample * budget
                conflict = int(episode["initial_conflicts"])
                for point in points:
                    if _number(point["elapsed_wall_seconds"]) > deadline:
                        break
                    conflict = int(point["conflicts_after"])
                values.append(conflict / int(episode["initial_conflicts"]))
            curves.append(values)
        means = [statistics.fmean(values[index] for values in curves) for index in range(len(samples))] if curves else [0.0] * len(samples)
        points_text = " ".join(f"{70 + 780 * sample:.2f},{420 - 350 * min(1.2, value) / 1.2:.2f}" for sample, value in zip(samples, means))
        body.append(f'<polyline points="{points_text}" fill="none" stroke="{colors.get(controller, "#999999")}" stroke-width="3"/>')
    body.append('<text x="380" y="455" font-family="sans-serif" font-size="13">fraction of wall-clock budget</text><text x="8" y="55" font-family="sans-serif" font-size="12">normalized conflicts</text>')
    path.write_text(_svg_frame(f"Conflict trajectory over wall time: {track}", "".join(body)), encoding="utf-8")


def _fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _report_markdown(
    *,
    primary_track: str,
    summaries: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    validation: dict[str, Any],
    sensitivity: list[dict[str, Any]],
) -> str:
    all_rows = {
        str(row["controller"]): row
        for row in summaries
        if row["track"] == primary_track and row["group_type"] == "all"
    }
    report_controllers = sorted(
        all_rows, key=lambda value: (value not in CONTROLLERS, value)
    )
    lns2 = all_rows.get("official_adaptive", {})
    v2 = all_rows.get("v2-full", {})
    common = [row for row in paired if row["track"] == primary_track and row["common_success"]]
    component_deltas = {
        name: _mean(row.get(f"delta_{name}_v2_minus_lns2") for row in common)
        for name in DECOMPOSITION_FIELDS
    }
    positive = [(name, value) for name, value in component_deltas.items() if value is not None and value > 0.0]
    bottleneck = max(positive, key=lambda item: item[1]) if positive else None
    loops_saved = _mean(row.get("repair_loops_saved_by_v2") for row in common)
    loop_seconds_saved = _mean(
        row.get("estimated_loop_wall_seconds_saved_by_v2") for row in common
    )
    repair_work_saved = _mean(
        row.get("estimated_repair_work_seconds_saved_by_v2") for row in common
    )
    selection_overhead = _mean(
        row.get("additional_v2_selection_seconds") for row in common
    )
    observed_iteration_delta = _mean(
        row.get("v2_iteration_wall_delta_seconds") for row in common
    )
    selection_stage_fields = (
        "candidate_generation_seconds",
        "state_check_seconds",
        "state_analysis_seconds",
        "proposal_feature_seconds",
        "realized_feature_seconds",
        "ranking_inference_seconds",
        "selection_residual_seconds",
    )
    v2_selection_stages = [
        (name, _number(v2.get(f"mean_total_{name}")))
        for name in selection_stage_fields
    ]
    dominant_selection_stage = (
        max(v2_selection_stages, key=lambda item: item[1])
        if v2_selection_stages
        else None
    )
    if selection_overhead is None or repair_work_saved is None:
        compensation_statement = "insufficient common-success timing data"
    elif repair_work_saved >= selection_overhead:
        compensation_statement = (
            "estimated saved repair work covers the additional selection cost"
        )
    else:
        compensation_statement = (
            "estimated saved repair work does not cover the additional selection cost"
        )
    selection_delta = (
        _number(v2.get("mean_iteration_selection_seconds"))
        - _number(lns2.get("mean_iteration_selection_seconds"))
    )
    pp_delta = (
        _number(v2.get("mean_iteration_pp_seconds"))
        - _number(lns2.get("mean_iteration_pp_seconds"))
    )
    lines = [
        "# v2 speed bottleneck report",
        "",
        f"Primary diagnostic track: `{primary_track}`.",
        f"Validation passed: `{bool(validation.get('passed'))}`.",
        "",
        "## Direct answers",
        "",
        f"1. Mean neighborhood-selection time per repair: LNS2 {_fmt(lns2.get('mean_iteration_selection_seconds'), 6)}s, v2 {_fmt(v2.get('mean_iteration_selection_seconds'), 6)}s; v2-LNS2 {_fmt(selection_delta, 6)}s.",
        f"2. Mean attempted repair loops per repairable episode: LNS2 {_fmt(lns2.get('mean_repair_iterations'), 2)}, v2 {_fmt(v2.get('mean_repair_iterations'), 2)}. On common successes, v2 saves {_fmt(loops_saved, 2)} loops, an estimated {_fmt(loop_seconds_saved, 6)}s of loop wall time ({_fmt(repair_work_saved, 6)}s excluding selection).",
        f"3. Mean PP time per repair: LNS2 {_fmt(lns2.get('mean_iteration_pp_seconds'), 6)}s, v2 {_fmt(v2.get('mean_iteration_pp_seconds'), 6)}s; v2-LNS2 {_fmt(pp_delta, 6)}s. This identifies whether v2 neighborhoods make PP more expensive.",
        f"4. On {len(common)} common-success pairs, v2 adds {_fmt(selection_overhead, 6)}s of selection while its observed total iteration-time delta is {_fmt(observed_iteration_delta, 6)}s. Result: {compensation_statement}.",
        (
            f"5. Largest positive v2 wall-time contribution on common successes: `{bottleneck[0]}` ({bottleneck[1]:.6f}s per episode)."
            if bottleneck is not None
            else "5. No positive bottleneck contribution could be identified from the available common-success episodes."
        ),
        (
            f"6. Inside v2 selection, the largest measured stage is `{dominant_selection_stage[0]}` ({dominant_selection_stage[1]:.6f}s per episode)."
            if dominant_selection_stage is not None
            else "6. No v2 selection-stage timing was available."
        ),
        f"7. Mean state-fingerprint cost per repairable episode: LNS2 {_fmt(lns2.get('mean_episode_state_fingerprint_seconds'), 6)}s, v2 {_fmt(v2.get('mean_episode_state_fingerprint_seconds'), 6)}s; trace writing is shown separately in the table.",
        "",
        "## Controller totals",
        "",
        "| controller | successes | fixed-step AUC | normalized fixed-step AUC | normalized wall AUC | final SOC | loops mean/median/P95 | selection/episode | PP/episode | state export/episode | trace/episode | restricted TTF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for controller in report_controllers:
        row = all_rows.get(controller, {})
        lines.append(
            f"| {LABELS.get(controller, controller)} | {int(row.get('success_count', 0))}/{int(row.get('episode_count', 0))} | "
            f"{_fmt(row.get('mean_fixed_budget_conflict_auc'), 6)} | "
            f"{_fmt(row.get('mean_normalized_fixed_budget_conflict_auc'), 6)} | "
            f"{_fmt(row.get('mean_normalized_wall_clock_conflict_auc'), 6)} | "
            f"{_fmt(row.get('mean_budget_final_sum_of_costs'), 2)} | "
            f"{_fmt(row.get('mean_repair_iterations'), 2)}/{_fmt(row.get('median_repair_iterations'), 2)}/{_fmt(row.get('p95_repair_iterations'), 2)} | "
            f"{_fmt(row.get('mean_total_neighborhood_selection_seconds'))}s | "
            f"{_fmt(row.get('mean_total_pp_replan_seconds'))}s | "
            f"{_fmt(row.get('mean_total_state_export_seconds'))}s | "
            f"{_fmt(row.get('mean_total_trace_write_seconds'))}s | "
            f"{_fmt(row.get('mean_restricted_time_to_feasible'))}s |"
        )
    lines.extend(
        [
            "",
            "## AUC by evaluation track",
            "",
            "Fixed-step AUC measures decision quality per repair; wall AUC measures conflict progress per real second. Lower is better for both.",
            "",
            "| track | controller | fixed-step AUC | normalized fixed-step AUC | normalized wall AUC |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in summaries:
        if row.get("group_type") != "all":
            continue
        controller = str(row["controller"])
        lines.append(
            f"| {row['track']} | {LABELS.get(controller, controller)} | "
            f"{_fmt(row.get('mean_fixed_budget_conflict_auc'), 6)} | "
            f"{_fmt(row.get('mean_normalized_fixed_budget_conflict_auc'), 6)} | "
            f"{_fmt(row.get('mean_normalized_wall_clock_conflict_auc'), 6)} |"
        )
    lines.extend(["", "## Paired common-success decomposition", ""])
    for name in DECOMPOSITION_FIELDS:
        lines.append(f"- `{name}`: mean v2-LNS2 delta {_fmt(component_deltas.get(name), 6)}s")
    lines.extend(
        [
            "",
            "A positive delta means v2 spent more time in that observed stage. This is an additive runtime decomposition, not a causal counterfactual, because LNS2 and v2 may visit different states and choose different neighborhoods.",
            "",
        ]
    )
    if sensitivity:
        lines.extend(
            [
                "## Wall-clock sensitivity",
                "",
                "Only task/seed pairs unsolved by either controller on the primary wall-clock track are included; both controllers are rerun without a repair-count cap.",
                "",
                "| track | controller | selected pairs | primary successes | extended successes | new successes | lost successes | mean final-conflict change |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sensitivity:
            lines.append(
                f"| {row['sensitivity_track']} | {LABELS.get(str(row['controller']), str(row['controller']))} | "
                f"{row['selected_pair_count']} | {row['primary_success_count']} | "
                f"{row['sensitivity_success_count']} | {row['new_success_count']} | "
                f"{row['lost_success_count']} | "
                f"{_fmt(row.get('mean_budget_final_conflict_change'), 4)} |"
            )
        lines.append("")
    return "\n".join(lines)


def generate_bottleneck_artifacts(
    track_roots: dict[str, dict[str, Path]], output: str | Path
) -> dict[str, Any]:
    """Generate the retained paired wall-clock evaluation artifacts."""

    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    controllers_by_track = {
        str(track): tuple(map(str, roots)) for track, roots in track_roots.items()
    }
    if not controllers_by_track:
        raise ValueError("bottleneck report requires at least one track")
    if len({values for values in controllers_by_track.values()}) != 1:
        raise ValueError("all bottleneck tracks must contain the same controllers")
    report_controllers = next(iter(controllers_by_track.values()))
    unknown_controllers = set(report_controllers) - set(CONTROLLERS)
    if unknown_controllers:
        raise ValueError(
            f"unsupported active controllers: {sorted(unknown_controllers)}"
        )
    for track, roots in track_roots.items():
        track_episodes, track_iterations, track_metadata = load_track(track, roots)
        episodes.extend(track_episodes)
        iterations.extend(track_iterations)
        metadata[track] = track_metadata

    track_coverage = {
        track: _track_coverage(
            track=track,
            roots=roots,
            episodes=episodes,
            metadata=dict(metadata.get(track) or {}),
        )
        for track, roots in track_roots.items()
    }
    coverage_failure_tracks = [
        track for track, coverage in track_coverage.items() if not coverage["passed"]
    ]
    paired = paired_decomposition(episodes)
    long_checkpoints, long_diagnostics, extension_keys = long_horizon_diagnostics(
        episodes, iterations
    )
    summaries = _summary_rows(episodes, iterations)
    primary_track = next(
        (
            track
            for track in track_roots
            if str(track).startswith("wall-clock-")
        ),
        next(iter(track_roots)),
    )
    sensitivity = _sensitivity_rows(paired, primary_track)

    valid_episodes = [
        row for row in episodes if row.get("status") in {"ok", "resumed"}
    ]
    errors = [
        row for row in episodes if row.get("status") not in {"ok", "resumed"}
    ]
    wall_rows = [
        row for row in valid_episodes if row.get("stopping_rule") == "wall-clock"
    ]
    paired_coverage: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in valid_episodes:
        paired_coverage[
            (
                str(row["track"]),
                str(row.get("task_id")),
                int(row.get("solver_seed", -1)),
            )
        ].add(str(row["controller"]))
    unpaired_episode_keys = [
        key
        for key, controllers in paired_coverage.items()
        if set(controllers) != set(controllers_by_track[key[0]])
    ]
    dataset_fingerprint_mismatch_tracks = [
        track
        for track, values in metadata.items()
        if len(
            {
                str(dict(value).get("dataset_fingerprint"))
                for value in dict(values).values()
            }
        )
        != 1
    ]
    pp_configuration_mismatch_tracks = [
        track
        for track, values in metadata.items()
        if {
            str(dict(value).get("replan_algorithm"))
            for value in dict(values).values()
        }
        != {"PP"}
        or {
            bool(dict(value).get("use_sipp"))
            for value in dict(values).values()
        }
        != {True}
        or any(
            not dict(value).get("native_module_sha256")
            for value in dict(values).values()
        )
        or len(
            {
                str(dict(value).get("native_module_sha256"))
                for value in dict(values).values()
            }
        )
        != 1
    ]
    paired_fingerprint_mismatches = [
        row for row in paired if not row["initial_fingerprint_match"]
    ]
    repair_limit_violations = [
        row for row in wall_rows if row.get("stop_reason") == "repair_limit"
    ]
    instrumentation_missing = [
        row
        for row in valid_episodes
        if row.get("repair_iterations", 0) > 0
        and not row.get("timing_instrumentation_complete")
    ]
    finalization_missing = [
        row
        for row in valid_episodes
        if not row.get("finalization_timing_instrumented")
    ]
    timing_closure_failures = [
        row
        for row in valid_episodes
        if _number(row.get("process_timing_closure_error_seconds"))
        > max(0.02, 0.02 * _number(row.get("episode_process_wall_seconds")))
    ]
    sensitivity_integrity_failures = [
        row
        for row in sensitivity
        if int(row.get("initial_fingerprint_mismatch_count", 0)) > 0
    ]
    validation = {
        "passed": not errors
        and not unpaired_episode_keys
        and not dataset_fingerprint_mismatch_tracks
        and not pp_configuration_mismatch_tracks
        and not paired_fingerprint_mismatches
        and not repair_limit_violations
        and not instrumentation_missing
        and not finalization_missing
        and not timing_closure_failures
        and not sensitivity_integrity_failures
        and not coverage_failure_tracks,
        "coverage_passed": not coverage_failure_tracks,
        "coverage_failure_track_count": len(coverage_failure_tracks),
        "track_coverage": track_coverage,
        "expected_task_seed_count": sum(
            int(coverage["expected_key_count"])
            for coverage in track_coverage.values()
        ),
        "missing_episode_key_count": sum(
            len(controller["missing_keys"])
            for coverage in track_coverage.values()
            for controller in dict(coverage["controllers"]).values()
        ),
        "unexpected_episode_key_count": sum(
            len(controller["unexpected_keys"])
            for coverage in track_coverage.values()
            for controller in dict(coverage["controllers"]).values()
        ),
        "duplicate_episode_key_count": sum(
            int(controller["duplicate_key_count"])
            for coverage in track_coverage.values()
            for controller in dict(coverage["controllers"]).values()
        ),
        "empty_manifest_controller_count": sum(
            int(controller["manifest_row_count"] == 0)
            for coverage in track_coverage.values()
            for controller in dict(coverage["controllers"]).values()
        ),
        "sensitivity_integrity_passed": not sensitivity_integrity_failures,
        "episode_count": len(valid_episodes),
        "iteration_count": len(iterations),
        "error_episode_count": len(errors),
        "unpaired_episode_key_count": len(unpaired_episode_keys),
        "dataset_fingerprint_mismatch_track_count": len(
            dataset_fingerprint_mismatch_tracks
        ),
        "pp_configuration_mismatch_track_count": len(
            pp_configuration_mismatch_tracks
        ),
        "paired_fingerprint_mismatch_count": len(paired_fingerprint_mismatches),
        "wall_clock_repair_limit_violation_count": len(repair_limit_violations),
        "instrumentation_missing_episode_count": len(instrumentation_missing),
        "finalization_timing_missing_episode_count": len(finalization_missing),
        "timing_closure_failure_episode_count": len(timing_closure_failures),
        "sensitivity_integrity_failure_count": len(
            sensitivity_integrity_failures
        ),
        "sensitivity_lost_success_count": sum(
            int(row.get("lost_success_count", 0)) for row in sensitivity
        ),
        "maximum_process_timing_closure_error_seconds": max(
            (
                _number(row.get("process_timing_closure_error_seconds"))
                for row in valid_episodes
            ),
            default=0.0,
        ),
        "wall_clock_max_repair_iterations": max(
            (int(row.get("repair_iterations", 0)) for row in wall_rows), default=0
        ),
        "wall_clock_exceeded_100_repairs": any(
            int(row.get("repair_iterations", 0)) > 100 for row in wall_rows
        ),
    }

    _write_csv(output_root / "iteration_timings.csv", iterations)
    _write_csv(output_root / "episode_timing_breakdown.csv", episodes)
    _write_csv(output_root / "paired_bottleneck_decomposition.csv", paired)
    _write_csv(output_root / "long_horizon_checkpoints.csv", long_checkpoints)
    _write_csv(output_root / "long_horizon_diagnostics.csv", long_diagnostics)
    _write_csv(output_root / "timing_summary.csv", summaries)
    _write_csv(
        output_root / "neighborhood_pp_summary.csv",
        _neighborhood_pp_rows(iterations),
    )
    _write_csv(output_root / "wall_clock_sensitivity.csv", sensitivity)
    _stacked_timing_svg(output_root / "timing_breakdown.svg", summaries, primary_track)
    _loop_svg(output_root / "loop_count_and_time.svg", summaries, primary_track)
    _scatter_svg(output_root / "neighborhood_size_vs_pp.svg", iterations, primary_track)
    _conflict_curve_svg(
        output_root / "conflicts_over_wall_time.svg",
        episodes,
        iterations,
        primary_track,
    )
    report = {
        "schema": REPORT_SCHEMA,
        "primary_track": primary_track,
        "tracks": list(track_roots),
        "controllers": list(report_controllers),
        "metadata": metadata,
        "validation": validation,
        "episode_count": len(valid_episodes),
        "iteration_count": len(iterations),
        "paired_episode_count": len(paired),
        "wall_clock_sensitivity": sensitivity,
        "long_horizon_extension_job_keys": extension_keys,
        "long_horizon_plateau_count": sum(
            bool(row["plateau"]) for row in long_diagnostics
        ),
    }
    _write_json(output_root / "bottleneck_report.json", report)
    (output_root / "v2_bottleneck_report.md").write_text(
        _report_markdown(
            primary_track=primary_track,
            summaries=summaries,
            paired=paired,
            validation=validation,
            sensitivity=sensitivity,
        ),
        encoding="utf-8",
    )
    return report


__all__ = [
    "CONTROLLERS",
    "DECOMPOSITION_FIELDS",
    "REPORT_SCHEMA",
    "SUPPORTED_NATIVE_TIMING_SCHEMAS",
    "TIMING_FIELDS",
    "generate_bottleneck_artifacts",
    "long_horizon_diagnostics",
    "validate_manifest_trace",
    "load_track",
    "paired_decomposition",
]
