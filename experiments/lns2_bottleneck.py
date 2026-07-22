from __future__ import annotations

import csv
import html
import itertools
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.closed_loop_trace_storage import read_trace_events
from experiments.tradeoff_evaluation import _manifest_path as _controller_manifest_path
from experiments.repair_collection import _fingerprint, _read_json, _read_jsonl, _write_json


REPORT_SCHEMA = "lns2.v2_bottleneck_diagnostic.v2"
CONTROLLERS = ("official_adaptive", "v2-full")
LABELS = {
    "official_adaptive": "Original LNS2 Adaptive",
    "v2-full": "Optimized model (v2)",
    "v2-stall-safe": "Optimized model (v2 stall-safe)",
    "v2-repair-aware": "Optimized model (v2 repair-aware)",
    "v3-full": "Cost-aware model (v3)",
}
CONTROLLER_COLORS = {
    "official_adaptive": "#4c78a8",
    "v2-full": "#f58518",
    "v2-stall-safe": "#54a24b",
    "v2-repair-aware": "#b279a2",
    "v3-full": "#e45756",
}
TIMING_FIELDS = (
    "neighborhood_selection_seconds",
    "candidate_generation_seconds",
    "state_check_seconds",
    "state_check_fingerprint_seconds",
    "state_analysis_seconds",
    "proposal_feature_seconds",
    "realized_feature_seconds",
    "ranking_inference_seconds",
    "stall_guard_seconds",
    "repair_aware_seconds",
    "v3_seconds",
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
    proposal = dict(controller_data.get("proposal") or {})
    guard = dict(controller_data.get("stall_guard") or {})
    repair_aware = dict(controller_data.get("repair_aware") or {})
    v3 = dict(controller_data.get("v3") or {})
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
        "stall_guard_active_size_cap": guard.get("active_size_cap"),
        "stall_guard_state_anchor_fingerprint": guard.get(
            "state_anchor_fingerprint"
        ),
        "stall_guard_base_candidate_id": guard.get("base_selected_candidate_id"),
        "stall_guard_effective_candidate_id": guard.get(
            "effective_selected_candidate_id"
        ),
        "stall_guard_base_selection_preserved": guard.get(
            "base_selection_preserved"
        ),
        "stall_guard_stagnant_attempt": guard.get("stagnant_attempt"),
        "stall_guard_backoff_triggered": guard.get("backoff_triggered"),
        "stall_guard_fallback_reason": guard.get("fallback_reason"),
        "repair_outcome": repair_aware.get(
            "repair_outcome", v3.get("repair_outcome")
        ),
        "repair_aware_selection_kind": repair_aware.get("selection_kind"),
        "repair_aware_state_anchor_fingerprint": repair_aware.get(
            "state_anchor_fingerprint"
        ),
        "repair_aware_base_candidate_id": repair_aware.get(
            "base_selected_candidate_id"
        ),
        "repair_aware_effective_candidate_id": repair_aware.get(
            "effective_selected_candidate_id"
        ),
        "repair_aware_shadow_candidate_id": repair_aware.get(
            "shadow_selected_candidate_id"
        ),
        "repair_aware_base_selection_preserved": repair_aware.get(
            "base_selection_preserved"
        ),
        "repair_aware_cache_hit": bool(
            controller_data.get("repair_aware_cache_hit", False)
        ),
        "repair_aware_no_progress": repair_aware.get("no_progress"),
        "repair_aware_failed_candidate_count": repair_aware.get(
            "failed_candidate_count_after"
        ),
        "repair_aware_rescue_attempts": repair_aware.get(
            "rescue_attempts_after"
        ),
        "repair_aware_adaptive_fallback_active": repair_aware.get(
            "adaptive_fallback_active"
        ),
        "repair_aware_lazy_candidate_count": int(
            proposal.get("lazy_candidate_count", 0)
        ),
        "repair_aware_lazy_generation_seconds": float(
            proposal.get("lazy_generation_seconds", 0.0)
        ),
        "v3_selection_kind": v3.get("selection_kind"),
        "v3_state_anchor_fingerprint": v3.get("state_anchor_fingerprint"),
        "v3_effective_candidate_id": v3.get("effective_selected_candidate_id"),
        "v3_cache_hit": bool(controller_data.get("v3_cache_hit", False)),
        "v3_no_progress": v3.get("no_progress"),
        "v3_failed_candidate_count": v3.get("failed_candidate_count_after"),
        "v3_blacklisted_neighborhood_count": v3.get(
            "blacklisted_neighborhood_count_after"
        ),
        "v3_adaptive_fallback_active": v3.get("adaptive_fallback_active"),
        "low_level_expanded": int(low.get("expanded", 0)),
        "low_level_generated": int(low.get("generated", 0)),
        "low_level_reopened": int(low.get("reopened", 0)),
        "low_level_runs": int(low.get("runs", 0)),
        "trace_write_seconds": trace_write_seconds,
        "timing_instrumented": bool(event.get("timings"))
        and event.get("native_timing_schema") == "lns2.repair_timing.v1"
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
    stall_guard = dict(summary.get("stall_guard") or {})
    repair_aware = dict(summary.get("repair_aware") or {})
    v3 = dict(summary.get("v3") or {})
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
        "stall_guard_size_backoff_count": int(
            stall_guard.get("size_backoff_count", 0)
        ),
        "stall_guard_fallback_activation_count": int(
            stall_guard.get("fallback_activation_count", 0)
        ),
        "stall_guard_official_fallback_decision_count": int(
            stall_guard.get("official_fallback_decision_count", 0)
        ),
        "stall_guard_blacklist_addition_count": int(
            stall_guard.get("blacklist_addition_count", 0)
        ),
        "stall_guard_base_selection_preserved_count": int(
            stall_guard.get("base_selection_preserved_count", 0)
        ),
        "stall_guard_model_override_count": int(
            stall_guard.get("model_override_count", 0)
        ),
        "stall_guard_stagnant_attempt_count": int(
            stall_guard.get("stagnant_attempt_count", 0)
        ),
        "stall_guard_longest_unchanged_state_streak": int(
            stall_guard.get("longest_unchanged_state_streak", 0)
        ),
        "stall_guard_rescued_state_count": int(
            stall_guard.get("rescued_state_count", 0)
        ),
        "repair_aware_no_progress_count": int(
            repair_aware.get("no_progress_count", 0)
        ),
        "repair_aware_hard_failure_count": int(
            repair_aware.get("hard_failure_count", 0)
        ),
        "repair_aware_accepted_noop_count": int(
            repair_aware.get("accepted_noop_count", 0)
        ),
        "repair_aware_rescue_selection_count": int(
            repair_aware.get("rescue_selection_count", 0)
        ),
        "repair_aware_fallback_count": int(
            repair_aware.get("fallback_count", 0)
        ),
        "repair_aware_cache_hit_count": int(
            repair_aware.get("cache_hit_count", 0)
        ),
        "repair_aware_cache_refresh_count": int(
            repair_aware.get("cache_refresh_count", 0)
        ),
        "repair_aware_shadow_difference_count": int(
            repair_aware.get("shadow_difference_count", 0)
        ),
        "repair_aware_tiebreak_override_count": int(
            repair_aware.get("tiebreak_override_count", 0)
        ),
        "repair_aware_rescued_state_count": int(
            repair_aware.get("rescued_state_count", 0)
        ),
        "repair_aware_longest_unchanged_streak": int(
            repair_aware.get("longest_unchanged_streak", 0)
        ),
        "v3_no_progress_count": int(v3.get("no_progress_count", 0)),
        "v3_hard_failure_count": int(v3.get("hard_failure_count", 0)),
        "v3_accepted_noop_count": int(v3.get("accepted_noop_count", 0)),
        "v3_adaptive_fallback_decision_count": int(
            v3.get("adaptive_fallback_decision_count", 0)
        ),
        "v3_adaptive_fallback_fraction": _number(
            v3.get("adaptive_fallback_fraction")
        ),
        "v3_blacklist_addition_count": int(
            v3.get("blacklist_addition_count", 0)
        ),
        "v3_cache_hit_count": int(v3.get("cache_hit_count", 0)),
        "v3_rescued_state_count": int(v3.get("rescued_state_count", 0)),
        "v3_longest_unchanged_streak": int(
            v3.get("longest_unchanged_streak", 0)
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
        metadata[controller] = {
            "root": str(root),
            "run_fingerprint": run.get("run_fingerprint"),
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
            trace_path = root / str(source["trace_file"])
            events = read_trace_events(trace_path)
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


def controller_pairwise_rows(
    episodes: list[dict[str, Any]], controllers: Iterable[str]
) -> list[dict[str, Any]]:
    names = tuple(map(str, controllers))
    indexed = {
        (
            str(row["track"]),
            str(row["controller"]),
            str(row.get("task_id")),
            int(row.get("solver_seed", -1)),
        ): row
        for row in episodes
        if row.get("status") in {"ok", "resumed"}
    }
    keys = sorted({(key[0], key[2], key[3]) for key in indexed})
    result: list[dict[str, Any]] = []
    for reference, candidate in itertools.combinations(names, 2):
        for track, task_id, seed in keys:
            left = indexed.get((track, reference, task_id, seed))
            right = indexed.get((track, candidate, task_id, seed))
            if left is None or right is None:
                continue
            row: dict[str, Any] = {
                "track": track,
                "pair": f"{candidate}_vs_{reference}",
                "reference": reference,
                "candidate": candidate,
                "task_id": task_id,
                "map_id": left.get("map_id"),
                "layout_family": left.get("layout_family"),
                "agent_count": left.get("agent_count"),
                "solver_seed": seed,
                "initial_fingerprint_match": left.get("initial_fingerprint")
                == right.get("initial_fingerprint"),
                "reference_success": bool(left.get("success")),
                "candidate_success": bool(right.get("success")),
                "common_success": bool(left.get("success") and right.get("success")),
            }
            for metric in PAIRWISE_METRICS:
                left_value = left.get(metric)
                right_value = right.get(metric)
                row[f"reference_{metric}"] = left_value
                row[f"candidate_{metric}"] = right_value
                row[f"delta_{metric}_candidate_minus_reference"] = (
                    float(right_value) - float(left_value)
                    if left_value is not None and right_value is not None
                    else None
                )
            result.append(row)
    return result


def controller_pairwise_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["track"]), str(row["pair"]))].append(row)
    result: list[dict[str, Any]] = []
    for (track, pair), values in sorted(grouped.items()):
        first = values[0]
        summary: dict[str, Any] = {
            "track": track,
            "pair": pair,
            "reference": first["reference"],
            "candidate": first["candidate"],
            "paired_episode_count": len(values),
            "initial_fingerprint_mismatch_count": sum(
                not bool(row["initial_fingerprint_match"]) for row in values
            ),
            "reference_success_count": sum(
                bool(row["reference_success"]) for row in values
            ),
            "candidate_success_count": sum(
                bool(row["candidate_success"]) for row in values
            ),
            "common_success_count": sum(bool(row["common_success"]) for row in values),
        }
        for metric in PAIRWISE_METRICS:
            field = f"delta_{metric}_candidate_minus_reference"
            summary[f"mean_{field}"] = _mean(row.get(field) for row in values)
            summary[f"median_{field}"] = _median(row.get(field) for row in values)
        result.append(summary)
    return result


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
                "repair_aware_no_progress_count": sum(
                    int(row.get("repair_aware_no_progress_count", 0))
                    for row in repairable
                ),
                "repair_aware_rescue_selection_count": sum(
                    int(row.get("repair_aware_rescue_selection_count", 0))
                    for row in repairable
                ),
                "repair_aware_fallback_count": sum(
                    int(row.get("repair_aware_fallback_count", 0))
                    for row in repairable
                ),
                "repair_aware_cache_hit_count": sum(
                    int(row.get("repair_aware_cache_hit_count", 0))
                    for row in repairable
                ),
                "repair_aware_rescued_state_count": sum(
                    int(row.get("repair_aware_rescued_state_count", 0))
                    for row in repairable
                ),
                "mean_repair_aware_longest_unchanged_streak": _mean(
                    row.get("repair_aware_longest_unchanged_streak")
                    for row in repairable
                ),
                "v3_no_progress_count": sum(
                    int(row.get("v3_no_progress_count", 0)) for row in repairable
                ),
                "v3_adaptive_fallback_decision_count": sum(
                    int(row.get("v3_adaptive_fallback_decision_count", 0))
                    for row in repairable
                ),
                "v3_cache_hit_count": sum(
                    int(row.get("v3_cache_hit_count", 0)) for row in repairable
                ),
                "v3_rescued_state_count": sum(
                    int(row.get("v3_rescued_state_count", 0)) for row in repairable
                ),
                "mean_v3_adaptive_fallback_fraction": _mean(
                    row.get("v3_adaptive_fallback_fraction") for row in repairable
                ),
                "mean_v3_longest_unchanged_streak": _mean(
                    row.get("v3_longest_unchanged_streak") for row in repairable
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


def stall_prefix_equivalence(iterations: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare stall-safe with v2-full until the guard first changes an action."""

    safe_rows = [
        row for row in iterations if str(row.get("controller")) == "v2-stall-safe"
    ]
    if not safe_rows or not any(row.get("candidate_score_fingerprint") for row in safe_rows):
        return {
            "applicable": False,
            "passed": False,
            "reason": "stall_safe_candidate_diagnostics_absent",
            "comparison_count": 0,
            "mismatch_count": 0,
            "trigger_count": 0,
        }
    grouped: dict[tuple[str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in iterations:
        controller = str(row.get("controller"))
        if controller not in {"v2-full", "v2-stall-safe"}:
            continue
        grouped[
            (
                str(row.get("track")),
                str(row.get("task_id")),
                int(row.get("solver_seed", -1)),
            )
        ][controller].append(row)
    mismatch_rows: list[dict[str, Any]] = []
    comparisons = 0
    triggers = 0
    for key, controllers in sorted(grouped.items()):
        full_by_index = {
            int(row["decision_index"]): row
            for row in controllers.get("v2-full", [])
        }
        for safe in sorted(
            controllers.get("v2-stall-safe", []),
            key=lambda value: int(value["decision_index"]),
        ):
            triggered = bool(
                safe.get("route") == "official_adaptive"
                or safe.get("stall_guard_base_selection_preserved") is False
            )
            if triggered:
                triggers += 1
                break
            decision_index = int(safe["decision_index"])
            full = full_by_index.get(decision_index)
            comparisons += 1
            fields = (
                "before_fingerprint",
                "candidate_score_fingerprint",
                "candidate_ranking_fingerprint",
                "selected_candidate_id",
                "actual_neighborhood_fingerprint",
            )
            differences = [
                field
                for field in fields
                if full is None or safe.get(field) != full.get(field)
            ]
            if differences:
                mismatch_rows.append(
                    {
                        "track": key[0],
                        "task_id": key[1],
                        "solver_seed": key[2],
                        "decision_index": decision_index,
                        "different_fields": differences,
                    }
                )
                break
    return {
        "applicable": True,
        "passed": not mismatch_rows,
        "comparison_count": comparisons,
        "mismatch_count": len(mismatch_rows),
        "trigger_count": triggers,
        "mismatches": mismatch_rows,
    }


def stall_guard_attempt_limit_violations(
    iterations: list[dict[str, Any]], *, maximum_attempts: int = 2
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in iterations:
        if (
            str(row.get("controller")) != "v2-stall-safe"
            or str(row.get("route")) != "model"
            or not row.get("stall_guard_state_anchor_fingerprint")
            or row.get("stall_guard_active_size_cap") is None
        ):
            continue
        grouped[
            (
                str(row.get("track")),
                str(row.get("task_id")),
                int(row.get("solver_seed", -1)),
                str(row["stall_guard_state_anchor_fingerprint"]),
                int(row["stall_guard_active_size_cap"]),
            )
        ].append(row)
    return [
        {
            "track": key[0],
            "task_id": key[1],
            "solver_seed": key[2],
            "state_anchor_fingerprint": key[3],
            "size_cap": key[4],
            "attempt_count": len(rows),
        }
        for key, rows in sorted(grouped.items())
        if len(rows) > maximum_attempts
    ]


def targeted_stall_recovery_diagnostic(
    episodes: list[dict[str, Any]],
    iterations: list[dict[str, Any]],
    *,
    primary_track: str,
    task_id: str = "maze-128-128-1__random_04__agents_0600",
    solver_seed: int = 2,
    historical_final_conflicts: int = 8821,
) -> dict[str, Any]:
    selected_episodes = {
        str(row.get("controller")): row
        for row in episodes
        if str(row.get("track")) == primary_track
        and str(row.get("task_id")) == task_id
        and int(row.get("solver_seed", -1)) == solver_seed
        and str(row.get("controller")) in {"v2-full", "v2-stall-safe"}
    }
    if set(selected_episodes) != {"v2-full", "v2-stall-safe"}:
        return {"applicable": False, "passed": False, "reason": "target_pair_absent"}
    selected_iterations = {
        controller: sorted(
            (
                row
                for row in iterations
                if str(row.get("track")) == primary_track
                and str(row.get("task_id")) == task_id
                and int(row.get("solver_seed", -1)) == solver_seed
                and str(row.get("controller")) == controller
            ),
            key=lambda value: int(value["decision_index"]),
        )
        for controller in ("v2-full", "v2-stall-safe")
    }
    terminal_stall = []
    for row in reversed(selected_iterations["v2-full"]):
        if bool(row.get("replan_success")) or int(row.get("conflict_delta", 0)) > 0:
            break
        terminal_stall.append(row)
    terminal_stall.reverse()
    stall_start_seconds = (
        float(terminal_stall[0]["elapsed_wall_seconds"]) if terminal_stall else None
    )
    safe_progress_after_stall = bool(
        stall_start_seconds is not None
        and any(
            float(row.get("elapsed_wall_seconds", 0.0)) >= stall_start_seconds
            and int(row.get("conflict_delta", 0)) > 0
            for row in selected_iterations["v2-stall-safe"]
        )
    )
    safe_final = int(selected_episodes["v2-stall-safe"]["budget_final_conflicts"])
    return {
        "applicable": True,
        "passed": bool(
            terminal_stall
            and safe_progress_after_stall
            and safe_final < historical_final_conflicts
        ),
        "task_id": task_id,
        "solver_seed": solver_seed,
        "historical_final_conflicts": historical_final_conflicts,
        "v2_full_final_conflicts": int(
            selected_episodes["v2-full"]["budget_final_conflicts"]
        ),
        "v2_stall_safe_final_conflicts": safe_final,
        "v2_full_terminal_stall_start_seconds": stall_start_seconds,
        "v2_full_terminal_stall_iteration_count": len(terminal_stall),
        "stall_safe_progress_after_stall": safe_progress_after_stall,
    }


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


def _stall_promotion_gate(
    summaries: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    *,
    primary_track: str,
    validation_passed: bool,
) -> dict[str, Any]:
    controllers = {str(row["controller"]) for row in summaries}
    if "v2-stall-safe" not in controllers:
        return {"applicable": False, "passed": False, "reason": "stall_safe_absent"}
    all_rows = {
        (str(row["track"]), str(row["controller"])): row
        for row in summaries
        if row.get("group_type") == "all"
    }
    wall_safe = all_rows.get((primary_track, "v2-stall-safe"), {})
    wall_v2 = all_rows.get((primary_track, "v2-full"), {})
    wall_lns2 = all_rows.get((primary_track, "official_adaptive"), {})
    historical_tracks = sorted(
        {str(row["track"]) for row in summaries if str(row["track"]) == "historical"}
    )
    fixed_available = bool(historical_tracks)
    fixed_safe = all_rows.get(("historical", "v2-stall-safe"), {})
    fixed_v2 = all_rows.get(("historical", "v2-full"), {})
    fixed_lns2 = all_rows.get(("historical", "official_adaptive"), {})

    def no_higher(left: Any, right: Any) -> bool:
        return left is not None and right is not None and float(left) <= float(right)

    success_gate = int(wall_safe.get("success_count", -1)) >= max(
        int(wall_v2.get("success_count", 0)), int(wall_lns2.get("success_count", 0))
    )
    wall_v2_gate = no_higher(
        wall_safe.get("mean_normalized_wall_clock_conflict_auc"),
        wall_v2.get("mean_normalized_wall_clock_conflict_auc"),
    )
    wall_lns2_gate = no_higher(
        wall_safe.get("mean_normalized_wall_clock_conflict_auc"),
        wall_lns2.get("mean_normalized_wall_clock_conflict_auc"),
    )
    safe_fixed = fixed_safe.get("mean_normalized_fixed_budget_conflict_auc")
    v2_fixed = fixed_v2.get("mean_normalized_fixed_budget_conflict_auc")
    lns2_fixed = fixed_lns2.get("mean_normalized_fixed_budget_conflict_auc")
    fixed_degradation_gate = bool(
        fixed_available
        and safe_fixed is not None
        and v2_fixed is not None
        and float(safe_fixed) <= float(v2_fixed) * 1.02
    )
    fixed_retention_gate = bool(
        fixed_available
        and safe_fixed is not None
        and v2_fixed is not None
        and lns2_fixed is not None
        and (
            float(lns2_fixed) - float(safe_fixed)
            >= 0.9 * (float(lns2_fixed) - float(v2_fixed))
        )
    )
    selection_safe = wall_safe.get("mean_iteration_selection_seconds")
    selection_v2 = wall_v2.get("mean_iteration_selection_seconds")
    selection_gate = bool(
        selection_safe is not None
        and selection_v2 is not None
        and float(selection_safe) <= float(selection_v2) * 1.02
    )
    guard_seconds = wall_safe.get("mean_total_stall_guard_seconds")
    total_selection_seconds = wall_safe.get(
        "mean_total_neighborhood_selection_seconds"
    )
    guard_overhead_gate = bool(
        guard_seconds is not None
        and total_selection_seconds is not None
        and (
            float(guard_seconds) == 0.0
            or float(total_selection_seconds) > 0.0
            and float(guard_seconds) / float(total_selection_seconds) <= 0.02
        )
    )
    repeated_failure_gate = bool(
        wall_safe.get("mean_longest_failed_replan_streak") is not None
        and wall_v2.get("mean_longest_failed_replan_streak") is not None
        and float(wall_safe["mean_longest_failed_replan_streak"])
        < float(wall_v2["mean_longest_failed_replan_streak"])
    )
    pp_failure_gate = bool(
        wall_safe.get("failed_replan_fraction") is not None
        and wall_v2.get("failed_replan_fraction") is not None
        and float(wall_safe["failed_replan_fraction"])
        < float(wall_v2["failed_replan_fraction"])
    )
    common_ttf_deltas = [
        row.get("delta_restricted_time_to_feasible_candidate_minus_reference")
        for row in pairwise
        if row["track"] == primary_track
        and row["pair"] == "v2-stall-safe_vs_v2-full"
        and row["common_success"]
    ]
    common_ttf_delta = _mean(common_ttf_deltas)
    common_v2_ttf = _mean(
        row.get("reference_restricted_time_to_feasible")
        for row in pairwise
        if row["track"] == primary_track
        and row["pair"] == "v2-stall-safe_vs_v2-full"
        and row["common_success"]
    )
    ttf_gate = bool(
        common_ttf_delta is not None
        and common_v2_ttf is not None
        and float(common_ttf_delta) <= 0.05 * float(common_v2_ttf)
    )
    gates = {
        "validation": bool(validation_passed),
        "success_not_lower": success_gate,
        "wall_auc_not_worse_than_v2_full": wall_v2_gate,
        "wall_auc_not_worse_than_lns2": wall_lns2_gate,
        "fixed_auc_degradation_at_most_2pct": fixed_degradation_gate,
        "fixed_auc_benefit_retention_at_least_90pct": fixed_retention_gate,
        "selection_overhead_at_most_2pct": selection_gate,
        "guard_logic_fraction_at_most_2pct": guard_overhead_gate,
        "longest_failed_replan_streak_reduced": repeated_failure_gate,
        "pp_failure_fraction_reduced": pp_failure_gate,
        "common_success_ttf_degradation_at_most_5pct": ttf_gate,
    }
    return {
        "applicable": True,
        "passed": fixed_available and all(gates.values()),
        "fixed_track_available": fixed_available,
        "gates": gates,
        "common_success_mean_ttf_delta_seconds": common_ttf_delta,
    }


def _stall_recovery_markdown(
    promotion: dict[str, Any],
    pairwise_summary: list[dict[str, Any]],
    guard_usage: list[dict[str, Any]],
    prefix_equivalence: dict[str, Any],
    targeted_diagnostic: dict[str, Any],
) -> str:
    lines = [
        "# v2 stall-recovery report",
        "",
        f"Promotion gate applicable: `{promotion.get('applicable')}`; passed: `{promotion.get('passed')}`.",
        "",
        (
            "Pre-trigger v2 equivalence: "
            f"applicable=`{prefix_equivalence.get('applicable')}`, "
            f"passed=`{prefix_equivalence.get('passed')}`, "
            f"comparisons={prefix_equivalence.get('comparison_count', 0)}, "
            f"mismatches={prefix_equivalence.get('mismatch_count', 0)}, "
            f"guard triggers={prefix_equivalence.get('trigger_count', 0)}."
        ),
        "",
        (
            "Target maze600/seed=2 diagnostic: "
            f"applicable=`{targeted_diagnostic.get('applicable')}`, "
            f"passed=`{targeted_diagnostic.get('passed')}`, "
            f"v2-full final={targeted_diagnostic.get('v2_full_final_conflicts')}, "
            f"stall-safe final={targeted_diagnostic.get('v2_stall_safe_final_conflicts')}, "
            f"post-stall progress=`{targeted_diagnostic.get('stall_safe_progress_after_stall')}`."
        ),
        "",
        "## Pairwise controller results",
        "",
        "| track | pair | episodes | successes reference/candidate | normalized wall AUC delta | normalized fixed AUC delta | TTF delta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in pairwise_summary:
        lines.append(
            f"| {row['track']} | {row['pair']} | {row['paired_episode_count']} | "
            f"{row['reference_success_count']}/{row['candidate_success_count']} | "
            f"{_fmt(row.get('mean_delta_normalized_wall_clock_conflict_auc_candidate_minus_reference'), 6)} | "
            f"{_fmt(row.get('mean_delta_normalized_fixed_budget_conflict_auc_candidate_minus_reference'), 6)} | "
            f"{_fmt(row.get('mean_delta_restricted_time_to_feasible_candidate_minus_reference'), 6)} |"
        )
    lines.extend(
        [
            "",
            "## Stall guard usage",
            "",
            "| track | task | seed | backoffs | fallback decisions | longest stagnant run | rescued states |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in guard_usage:
        lines.append(
            f"| {row['track']} | {row['task_id']} | {row['solver_seed']} | "
            f"{row['stall_guard_size_backoff_count']} | "
            f"{row['stall_guard_official_fallback_decision_count']} | "
            f"{row['stall_guard_longest_unchanged_state_streak']} | "
            f"{row['stall_guard_rescued_state_count']} |"
        )
    if promotion.get("gates"):
        lines.extend(["", "## Promotion gates", ""])
        for name, passed in dict(promotion["gates"]).items():
            lines.append(f"- `{name}`: `{bool(passed)}`")
    lines.append("")
    return "\n".join(lines)


def _repair_aware_promotion_gate(
    summaries: list[dict[str, Any]],
    *,
    primary_track: str,
    validation_passed: bool,
) -> dict[str, Any]:
    controllers = {str(row["controller"]) for row in summaries}
    if "v2-repair-aware" not in controllers:
        return {
            "applicable": False,
            "passed": False,
            "reason": "repair_aware_absent",
        }
    all_rows = {
        (str(row["track"]), str(row["controller"])): row
        for row in summaries
        if row.get("group_type") == "all"
    }
    aware = all_rows.get((primary_track, "v2-repair-aware"), {})
    v2 = all_rows.get((primary_track, "v2-full"), {})
    lns2 = all_rows.get((primary_track, "official_adaptive"), {})
    fixed_aware = all_rows.get(("historical", "v2-repair-aware"), {})
    fixed_v2 = all_rows.get(("historical", "v2-full"), {})
    fixed_available = bool(fixed_aware and fixed_v2)

    def no_higher(left: Any, right: Any) -> bool:
        return left is not None and right is not None and float(left) <= float(right)

    aware_loops = max(1, int(aware.get("total_repair_iterations", 0)))
    v2_loops = max(1, int(v2.get("total_repair_iterations", 0)))
    aware_no_progress = int(aware.get("no_improvement_repair_count", 0)) / aware_loops
    v2_no_progress = int(v2.get("no_improvement_repair_count", 0)) / v2_loops
    selection_aware = aware.get("mean_iteration_selection_seconds")
    selection_v2 = v2.get("mean_iteration_selection_seconds")
    gates = {
        "validation": bool(validation_passed),
        "success_not_lower_than_v2_and_lns2": int(
            aware.get("success_count", -1)
        )
        >= max(int(v2.get("success_count", 0)), int(lns2.get("success_count", 0))),
        "wall_auc_not_worse_than_v2": no_higher(
            aware.get("mean_normalized_wall_clock_conflict_auc"),
            v2.get("mean_normalized_wall_clock_conflict_auc"),
        ),
        "wall_auc_not_worse_than_lns2": no_higher(
            aware.get("mean_normalized_wall_clock_conflict_auc"),
            lns2.get("mean_normalized_wall_clock_conflict_auc"),
        ),
        "fixed_auc_degradation_at_most_2pct": bool(
            fixed_available
            and fixed_aware.get("mean_normalized_fixed_budget_conflict_auc")
            is not None
            and fixed_v2.get("mean_normalized_fixed_budget_conflict_auc") is not None
            and float(fixed_aware["mean_normalized_fixed_budget_conflict_auc"])
            <= 1.02 * float(fixed_v2["mean_normalized_fixed_budget_conflict_auc"])
        ),
        "selection_overhead_at_most_5pct": bool(
            selection_aware is not None
            and selection_v2 is not None
            and float(selection_aware) <= 1.05 * float(selection_v2)
        ),
        "no_progress_fraction_reduced": aware_no_progress < v2_no_progress,
        "unchanged_streak_reduced": bool(
            aware.get("mean_repair_aware_longest_unchanged_streak") is not None
            and v2.get("mean_longest_failed_replan_streak") is not None
            and float(aware["mean_repair_aware_longest_unchanged_streak"])
            < float(v2["mean_longest_failed_replan_streak"])
        ),
    }
    return {
        "applicable": True,
        "passed": fixed_available and all(gates.values()),
        "fixed_track_available": fixed_available,
        "gates": gates,
        "repair_aware_no_progress_fraction": aware_no_progress,
        "v2_no_improvement_fraction": v2_no_progress,
        "cache_hit_count": int(aware.get("repair_aware_cache_hit_count", 0)),
        "rescue_selection_count": int(
            aware.get("repair_aware_rescue_selection_count", 0)
        ),
        "rescued_state_count": int(
            aware.get("repair_aware_rescued_state_count", 0)
        ),
    }


def _repair_aware_markdown(
    promotion: dict[str, Any], pairwise_summary: list[dict[str, Any]]
) -> str:
    lines = [
        "# v2 repair-aware report",
        "",
        f"Promotion applicable: `{promotion.get('applicable')}`; passed: `{promotion.get('passed')}`.",
        "",
        f"Cache hits: `{promotion.get('cache_hit_count', 0)}`; rescue selections: `{promotion.get('rescue_selection_count', 0)}`; rescued states: `{promotion.get('rescued_state_count', 0)}`.",
        "",
        "## Promotion gates",
        "",
    ]
    for name, passed in dict(promotion.get("gates") or {}).items():
        lines.append(f"- `{name}`: `{bool(passed)}`")
    lines.extend(["", "## Paired summaries", ""])
    for row in pairwise_summary:
        if "v2-repair-aware" not in str(row.get("pair")):
            continue
        lines.append(
            f"- `{row['track']}` `{row['pair']}`: successes "
            f"{row['candidate_success_count']} vs {row['reference_success_count']}; "
            "mean normalized wall-AUC delta "
            f"{_fmt(row.get('mean_delta_normalized_wall_clock_conflict_auc_candidate_minus_reference'), 6)}."
        )
    lines.append("")
    return "\n".join(lines)


def _paired_metric_values(
    rows: list[dict[str, Any]],
    *,
    candidate: str,
    reference: str,
    metric: str,
    common_success_only: bool = False,
) -> list[tuple[str, float, float]]:
    result: list[tuple[str, float, float]] = []
    for row in rows:
        if common_success_only and not bool(row.get("common_success")):
            continue
        if row.get("candidate") == candidate and row.get("reference") == reference:
            reference_value = row.get(f"reference_{metric}")
            candidate_value = row.get(f"candidate_{metric}")
        elif row.get("candidate") == reference and row.get("reference") == candidate:
            reference_value = row.get(f"candidate_{metric}")
            candidate_value = row.get(f"reference_{metric}")
        else:
            continue
        if reference_value is None or candidate_value is None:
            continue
        result.append(
            (str(row.get("map_id")), float(reference_value), float(candidate_value))
        )
    return result


def _map_bootstrap_relative_degradation(
    values: list[tuple[str, float, float]],
    *,
    samples: int = 5000,
    seed: int = 20260722,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for map_id, reference, candidate in values:
        grouped[map_id].append((reference, candidate))
    map_ids = sorted(grouped)
    if not map_ids:
        return {
            "map_count": 0,
            "pair_count": 0,
            "observed_relative_degradation": None,
            "one_sided_95_upper": None,
        }

    def degradation(selected_maps: list[str]) -> float:
        selected = [value for map_id in selected_maps for value in grouped[map_id]]
        reference_mean = statistics.fmean(value[0] for value in selected)
        candidate_mean = statistics.fmean(value[1] for value in selected)
        return (candidate_mean - reference_mean) / max(1e-12, reference_mean)

    rng = random.Random(seed)
    bootstrap = sorted(
        degradation([rng.choice(map_ids) for _map_id in map_ids])
        for _ in range(samples)
    )
    upper_index = min(len(bootstrap) - 1, math.ceil(0.95 * len(bootstrap)) - 1)
    return {
        "map_count": len(map_ids),
        "pair_count": len(values),
        "samples": samples,
        "observed_relative_degradation": degradation(map_ids),
        "one_sided_95_upper": bootstrap[upper_index],
    }


def _v3_promotion_gate(
    summaries: list[dict[str, Any]],
    pairwise: list[dict[str, Any]],
    *,
    primary_track: str,
    validation_passed: bool,
) -> dict[str, Any]:
    controllers = {str(row["controller"]) for row in summaries}
    if "v3-full" not in controllers:
        return {"applicable": False, "passed": False, "reason": "v3_absent"}
    all_rows = {
        (str(row["track"]), str(row["controller"])): row
        for row in summaries
        if row.get("group_type") == "all"
    }
    v3 = all_rows.get((primary_track, "v3-full"), {})
    v2 = all_rows.get((primary_track, "v2-full"), {})
    lns2 = all_rows.get((primary_track, "official_adaptive"), {})
    fixed_v3 = all_rows.get(("historical", "v3-full"), {})
    fixed_v2 = all_rows.get(("historical", "v2-full"), {})
    fixed_available = bool(fixed_v3 and fixed_v2)

    primary_pairs = [row for row in pairwise if row.get("track") == primary_track]
    auc_bootstrap = {
        baseline: _map_bootstrap_relative_degradation(
            _paired_metric_values(
                primary_pairs,
                candidate="v3-full",
                reference=baseline,
                metric="normalized_wall_clock_conflict_auc",
            ),
            seed=20260722 + offset,
        )
        for offset, baseline in enumerate(("v2-full", "official_adaptive"))
    }
    ttf = {}
    for baseline in ("v2-full", "official_adaptive"):
        values = _paired_metric_values(
            primary_pairs,
            candidate="v3-full",
            reference=baseline,
            metric="restricted_time_to_feasible",
            common_success_only=True,
        )
        ttf[baseline] = {
            "pair_count": len(values),
            "reference_mean": _mean(value[1] for value in values),
            "v3_mean": _mean(value[2] for value in values),
        }

    def no_higher(left: Any, right: Any, factor: float = 1.0) -> bool:
        return (
            left is not None
            and right is not None
            and float(left) <= factor * float(right) + 1e-12
        )

    v3_loops = max(1, int(v3.get("total_repair_iterations", 0)))
    v2_loops = max(1, int(v2.get("total_repair_iterations", 0)))
    v3_no_progress = int(v3.get("v3_no_progress_count", 0)) / v3_loops
    v2_no_improvement = int(v2.get("no_improvement_repair_count", 0)) / v2_loops
    fallback_fraction = int(
        v3.get("v3_adaptive_fallback_decision_count", 0)
    ) / v3_loops
    v2_ttf = ttf["v2-full"]
    lns2_ttf = ttf["official_adaptive"]
    checks = {
        "validation": bool(validation_passed),
        "success_not_lower_than_v2_and_lns2": int(v3.get("success_count", -1))
        >= max(int(v2.get("success_count", 0)), int(lns2.get("success_count", 0))),
        "mean_wall_auc_not_worse_than_v2": no_higher(
            v3.get("mean_normalized_wall_clock_conflict_auc"),
            v2.get("mean_normalized_wall_clock_conflict_auc"),
        ),
        "mean_wall_auc_not_worse_than_lns2": no_higher(
            v3.get("mean_normalized_wall_clock_conflict_auc"),
            lns2.get("mean_normalized_wall_clock_conflict_auc"),
        ),
        "paired_auc_upper_degradation_vs_v2_at_most_2pct": bool(
            auc_bootstrap["v2-full"]["one_sided_95_upper"] is not None
            and float(auc_bootstrap["v2-full"]["one_sided_95_upper"]) <= 0.02
        ),
        "paired_auc_upper_degradation_vs_lns2_at_most_2pct": bool(
            auc_bootstrap["official_adaptive"]["one_sided_95_upper"] is not None
            and float(
                auc_bootstrap["official_adaptive"]["one_sided_95_upper"]
            )
            <= 0.02
        ),
        "common_success_ttf_not_slower_than_v2": no_higher(
            v2_ttf["v3_mean"], v2_ttf["reference_mean"]
        ),
        "common_success_ttf_not_over_5pct_slower_than_lns2": no_higher(
            lns2_ttf["v3_mean"], lns2_ttf["reference_mean"], 1.05
        ),
        "fixed_auc_degradation_at_most_2pct": bool(
            fixed_available
            and no_higher(
                fixed_v3.get("mean_normalized_fixed_budget_conflict_auc"),
                fixed_v2.get("mean_normalized_fixed_budget_conflict_auc"),
                1.02,
            )
        ),
        "pp_no_progress_rate_reduced": v3_no_progress < v2_no_improvement,
        "longest_repeated_failure_reduced": bool(
            v3.get("mean_v3_longest_unchanged_streak") is not None
            and v2.get("mean_longest_failed_replan_streak") is not None
            and float(v3["mean_v3_longest_unchanged_streak"])
            < float(v2["mean_longest_failed_replan_streak"])
        ),
        "controller_time_increase_at_most_5pct": no_higher(
            v3.get("mean_iteration_selection_seconds"),
            v2.get("mean_iteration_selection_seconds"),
            1.05,
        ),
        "adaptive_fallback_fraction_at_most_5pct": fallback_fraction <= 0.05,
    }
    return {
        "applicable": True,
        "passed": fixed_available and all(checks.values()),
        "fixed_track_available": fixed_available,
        "gates": checks,
        "auc_map_bootstrap": auc_bootstrap,
        "common_success_time_to_feasible": ttf,
        "v3_no_progress_fraction": v3_no_progress,
        "v2_no_improvement_fraction": v2_no_improvement,
        "adaptive_fallback_fraction": fallback_fraction,
        "cache_hit_count": int(v3.get("v3_cache_hit_count", 0)),
        "rescued_state_count": int(v3.get("v3_rescued_state_count", 0)),
    }


def _v3_markdown(
    promotion: dict[str, Any], pairwise_summary: list[dict[str, Any]]
) -> str:
    lines = [
        "# v3 cost-aware controller report",
        "",
        f"Promotion applicable: `{promotion.get('applicable')}`; passed: `{promotion.get('passed')}`.",
        "",
        "## Promotion gates",
        "",
    ]
    for name, passed in dict(promotion.get("gates") or {}).items():
        lines.append(f"- `{name}`: `{bool(passed)}`")
    lines.extend(["", "## Paired summaries", ""])
    for row in pairwise_summary:
        if "v3-full" not in str(row.get("pair")):
            continue
        lines.append(
            f"- `{row['track']}` `{row['pair']}`: successes "
            f"{row['candidate_success_count']} vs {row['reference_success_count']}; "
            "mean normalized wall-AUC delta "
            f"{_fmt(row.get('mean_delta_normalized_wall_clock_conflict_auc_candidate_minus_reference'), 6)}."
        )
    lines.append("")
    return "\n".join(lines)


def generate_bottleneck_artifacts(
    track_roots: dict[str, dict[str, Path]], output: str | Path
) -> dict[str, Any]:
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    episodes: list[dict[str, Any]] = []
    iterations: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    controllers_by_track = {
        str(track): tuple(map(str, roots)) for track, roots in track_roots.items()
    }
    if len({values for values in controllers_by_track.values()}) != 1:
        raise ValueError("all bottleneck tracks must contain the same controllers")
    report_controllers = next(iter(controllers_by_track.values()))
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
    pairwise = controller_pairwise_rows(episodes, report_controllers)
    pairwise_summary = controller_pairwise_summary(pairwise)
    guard_usage = [
        row for row in episodes if str(row.get("controller")) == "v2-stall-safe"
    ]
    repair_aware_usage = [
        row
        for row in episodes
        if str(row.get("controller")) == "v2-repair-aware"
    ]
    v3_usage = [
        row for row in episodes if str(row.get("controller")) == "v3-full"
    ]
    long_checkpoints, long_diagnostics, extension_keys = long_horizon_diagnostics(
        episodes, iterations
    )
    stall_prefix = stall_prefix_equivalence(iterations)
    guard_attempt_violations = stall_guard_attempt_limit_violations(iterations)
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
    targeted_stall = targeted_stall_recovery_diagnostic(
        episodes, iterations, primary_track=primary_track
    )

    valid_episodes = [row for row in episodes if row.get("status") in {"ok", "resumed"}]
    errors = [row for row in episodes if row.get("status") not in {"ok", "resumed"}]
    wall_rows = [row for row in valid_episodes if row.get("stopping_rule") == "wall-clock"]
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
    paired_fingerprint_mismatches = [row for row in paired if not row["initial_fingerprint_match"]]
    repair_limit_violations = [row for row in wall_rows if row.get("stop_reason") == "repair_limit"]
    instrumentation_missing = [
        row
        for row in valid_episodes
        if row.get("repair_iterations", 0) > 0 and not row.get("timing_instrumentation_complete")
    ]
    finalization_missing = [
        row for row in valid_episodes if not row.get("finalization_timing_instrumented")
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
        and not coverage_failure_tracks
        and not guard_attempt_violations
        and (
            not stall_prefix["applicable"] or bool(stall_prefix["passed"])
        ),
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
        "stall_prefix_equivalence": stall_prefix,
        "stall_guard_attempt_limit_violation_count": len(
            guard_attempt_violations
        ),
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
    stall_promotion = _stall_promotion_gate(
        summaries,
        pairwise,
        primary_track=primary_track,
        validation_passed=bool(validation["passed"]),
    )
    repair_aware_promotion = _repair_aware_promotion_gate(
        summaries,
        primary_track=primary_track,
        validation_passed=bool(validation["passed"]),
    )
    v3_promotion = _v3_promotion_gate(
        summaries,
        pairwise,
        primary_track=primary_track,
        validation_passed=bool(validation["passed"]),
    )

    _write_csv(output_root / "iteration_timings.csv", iterations)
    _write_csv(output_root / "episode_timing_breakdown.csv", episodes)
    _write_csv(output_root / "paired_bottleneck_decomposition.csv", paired)
    _write_csv(output_root / "controller_pairwise_episodes.csv", pairwise)
    _write_csv(output_root / "controller_pairwise_summary.csv", pairwise_summary)
    _write_csv(output_root / "stall_guard_usage.csv", guard_usage)
    _write_csv(output_root / "repair_aware_usage.csv", repair_aware_usage)
    _write_csv(output_root / "v3_usage.csv", v3_usage)
    _write_json(
        output_root / "repair_aware_promotion.json", repair_aware_promotion
    )
    (output_root / "repair_aware_report.md").write_text(
        _repair_aware_markdown(repair_aware_promotion, pairwise_summary),
        encoding="utf-8",
    )
    _write_json(output_root / "v3_promotion.json", v3_promotion)
    (output_root / "v3_report.md").write_text(
        _v3_markdown(v3_promotion, pairwise_summary), encoding="utf-8"
    )
    _write_csv(
        output_root / "stall_prefix_mismatches.csv",
        list(stall_prefix.get("mismatches") or []),
    )
    _write_json(output_root / "stall_prefix_equivalence.json", stall_prefix)
    _write_csv(
        output_root / "stall_guard_attempt_limit_violations.csv",
        guard_attempt_violations,
    )
    _write_json(output_root / "targeted_stall_recovery.json", targeted_stall)
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
    _conflict_curve_svg(output_root / "conflicts_over_wall_time.svg", episodes, iterations, primary_track)
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
        "repair_aware_promotion": repair_aware_promotion,
        "v3_promotion": v3_promotion,
        "stall_promotion": stall_promotion,
        "stall_prefix_equivalence": stall_prefix,
        "targeted_stall_recovery": targeted_stall,
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
    (output_root / "stall_recovery_report.md").write_text(
        _stall_recovery_markdown(
            stall_promotion,
            pairwise_summary,
            guard_usage,
            stall_prefix,
            targeted_stall,
        ),
        encoding="utf-8",
    )
    return report


__all__ = [
    "CONTROLLERS",
    "DECOMPOSITION_FIELDS",
    "REPORT_SCHEMA",
    "TIMING_FIELDS",
    "generate_bottleneck_artifacts",
    "controller_pairwise_rows",
    "controller_pairwise_summary",
    "long_horizon_diagnostics",
    "stall_guard_attempt_limit_violations",
    "stall_prefix_equivalence",
    "targeted_stall_recovery_diagnostic",
    "load_track",
    "paired_decomposition",
]
