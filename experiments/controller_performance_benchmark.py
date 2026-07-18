from __future__ import annotations

import math
import statistics
import time
from pathlib import Path
from typing import Any

from experiments.closed_loop_confirmation import online_candidate_rows
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
)
from experiments.feature_schema_v2 import canonicalize_features
from experiments.online_feature_engine import OnlineFeatureEngine, static_grid_for_state
from experiments.repair_collection import _read_jsonl, _write_json


BENCHMARK_SCHEMA = "lns2.controller_feature_benchmark.v2"
REGISTERED_CASES = (
    ("random", "random", None),
    # Keep the five-family representatives distinct from the explicit
    # high-agent stress cases below.
    ("maze", "maze", 100),
    ("room", "room", None),
    ("warehouse", "warehouse", None),
    ("game", "game", None),
    ("maze400", "maze", 400),
    ("maze600", "maze", 600),
    ("room600", "room", 600),
    ("random600", "random", 600),
)


def _initial_state(
    root: Path, trace_path: Path, event: dict[str, Any]
) -> dict[str, Any]:
    if str(event.get("schema")) != EPISODE_SCHEMA_V2:
        return dict(event["state"])
    state = read_state_blob(
        resolve_state_blob(trace_path, str(event["state_blob"]), root)
    )
    state.update(dict(event["state_extras"]))
    return state


def _trace_samples(
    root: Path, manifest: dict[str, Any], maximum_decisions: int
) -> list[dict[str, Any]]:
    trace_path = root / str(manifest["trace_file"])
    events = read_trace_events(trace_path)
    state = _initial_state(root, trace_path, events[0])
    samples = []
    changed_agents = None
    for event in events[1:-1]:
        controller = dict(event.get("controller") or {})
        candidates = list(controller.get("candidate_pool") or [])
        if candidates and len(samples) < maximum_decisions:
            samples.append(
                {
                    "state": state,
                    "candidates": candidates,
                    "changed_agents": changed_agents,
                    "recorded_feature_seconds": float(
                        controller.get("feature_seconds", 0.0)
                    ),
                    "recorded_controller_seconds": float(
                        controller.get("controller_seconds_before_repair", 0.0)
                    ),
                    "repair_wall_seconds": float(
                        event.get("repair_wall_seconds", 0.0)
                    ),
                }
            )
        if str(event.get("schema")) == EPISODE_SCHEMA_V2:
            after = apply_state_delta(state, event.get("state_delta"))
            after.update(apply_extras_delta(state, event.get("state_extras_delta")))
        else:
            after = dict(event["after"])
        changed_agents = list(map(int, event.get("metrics", {}).get("neighborhood", [])))
        state = after
        if len(samples) >= maximum_decisions:
            break
    return samples


def _select_manifests(root: Path) -> dict[str, dict[str, Any]]:
    rows = [
        row
        for row in _read_jsonl(root / "realized_dynamic_manifest.jsonl")
        if str(row.get("status")) in {"ok", "resumed"}
    ]
    selected = {}
    for name, family, agent_count in REGISTERED_CASES:
        matches = [
            row
            for row in rows
            if str(row.get("layout_mode")) == family
            and (agent_count is None or int(row.get("agent_count", -1)) == agent_count)
            and int(row.get("summary", {}).get("repair_iterations", 0)) > 0
        ]
        if not matches:
            raise ValueError(f"benchmark collection has no eligible {name} episode")
        matches.sort(key=lambda row: (str(row["task_id"]), int(row["solver_seed"])))
        selected[name] = matches[0]
    return selected


def _percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _portable_location(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        return f"external/{path.name}"


def _reference_sequence(samples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    static_grid = static_grid_for_state(samples[0]["state"])
    return [
        online_candidate_rows(
            sample["state"], sample["candidates"], static_grid=static_grid
        )
        for sample in samples
    ]


def _optimized_sequence(
    samples: list[dict[str, Any]], *, shadow_validation: bool, backend: str
) -> list[list[dict[str, Any]]]:
    engine = OnlineFeatureEngine(
        samples[0]["state"], backend=backend, shadow_validation=shadow_validation
    )
    result = []
    for index, sample in enumerate(samples):
        if index:
            engine.prepare(
                sample["state"], changed_agents=sample["changed_agents"] or []
            )
        rows, _ = engine.realized_rows(
            sample["candidates"], state_hash=f"benchmark-{index}"
        )
        result.append(rows)
    return result


def _assert_equivalent(
    reference: list[list[dict[str, Any]]],
    optimized: list[list[dict[str, Any]]],
    tolerance: float,
) -> float:
    maximum_delta = 0.0
    if len(reference) != len(optimized):
        raise ValueError("benchmark sequence lengths differ")
    for old_rows, new_rows in zip(reference, optimized):
        if len(old_rows) != len(new_rows):
            raise ValueError("benchmark candidate counts differ")
        for old, new in zip(old_rows, new_rows):
            expected = canonicalize_features(
                old["features"]["realized_dynamic"], "realized_dynamic"
            )
            actual = new["features"]["realized_dynamic"]
            for name in expected:
                maximum_delta = max(maximum_delta, abs(expected[name] - actual[name]))
    if maximum_delta > tolerance:
        raise ValueError(
            f"optimized feature extractor differs from reference by {maximum_delta}"
        )
    return maximum_delta


def run_controller_feature_benchmark(
    collection: str | Path,
    output: str | Path,
    *,
    repeats: int = 7,
    maximum_decisions: int = 3,
    tolerance: float = 1e-12,
    feature_backend: str = "auto",
) -> dict[str, Any]:
    if repeats < 3:
        raise ValueError("performance benchmark requires at least three repeats")
    root = Path(collection).resolve()
    selected = _select_manifests(root)
    cases = []
    all_reference_runs = []
    all_optimized_runs = []
    for name, manifest in selected.items():
        samples = _trace_samples(root, manifest, maximum_decisions)
        if not samples:
            raise ValueError(f"benchmark episode {name} has no learned decisions")
        reference = _reference_sequence(samples)
        optimized = _optimized_sequence(
            samples, shadow_validation=True, backend=feature_backend
        )
        maximum_delta = _assert_equivalent(reference, optimized, tolerance)
        # One untimed warm-up per implementation precedes repeated measurements.
        _reference_sequence(samples)
        _optimized_sequence(
            samples, shadow_validation=False, backend=feature_backend
        )
        reference_times = []
        optimized_times = []
        for _ in range(repeats):
            started = time.perf_counter()
            _reference_sequence(samples)
            reference_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            _optimized_sequence(
                samples, shadow_validation=False, backend=feature_backend
            )
            optimized_times.append(time.perf_counter() - started)
        all_reference_runs.append(reference_times)
        all_optimized_runs.append(optimized_times)
        reference_median = statistics.median(reference_times)
        optimized_median = statistics.median(optimized_times)
        feature_ratio = optimized_median / reference_median
        recorded_feature = math.fsum(
            sample["recorded_feature_seconds"] for sample in samples
        )
        recorded_controller = math.fsum(
            sample["recorded_controller_seconds"] for sample in samples
        )
        recorded_repair = math.fsum(sample["repair_wall_seconds"] for sample in samples)
        estimated_controller = max(
            0.0,
            recorded_controller - recorded_feature + recorded_feature * feature_ratio,
        )
        recorded_end_to_end = recorded_controller + recorded_repair
        estimated_end_to_end = estimated_controller + recorded_repair
        cases.append(
            {
                "case": name,
                "layout_family": manifest["layout_mode"],
                "agent_count": int(manifest["agent_count"]),
                "task_id": manifest["task_id"],
                "decision_count": len(samples),
                "mean_candidate_count": statistics.fmean(
                    len(sample["candidates"]) for sample in samples
                ),
                "maximum_feature_delta": maximum_delta,
                "reference_median_seconds": reference_median,
                "reference_p95_seconds": _percentile95(reference_times),
                "optimized_median_seconds": optimized_median,
                "optimized_p95_seconds": _percentile95(optimized_times),
                "feature_time_reduction": 1.0 - optimized_median / reference_median,
                "speedup": reference_median / optimized_median,
                "recorded_controller_seconds": recorded_controller,
                "estimated_optimized_controller_seconds": estimated_controller,
                "estimated_controller_time_reduction": (
                    1.0 - estimated_controller / recorded_controller
                    if recorded_controller
                    else 0.0
                ),
                "estimated_controller_speedup": (
                    recorded_controller / estimated_controller
                    if estimated_controller
                    else None
                ),
                "recorded_controller_plus_repair_seconds": recorded_end_to_end,
                "estimated_optimized_controller_plus_repair_seconds": estimated_end_to_end,
                "estimated_end_to_end_time_reduction": (
                    1.0 - estimated_end_to_end / recorded_end_to_end
                    if recorded_end_to_end
                    else 0.0
                ),
                "estimated_end_to_end_speedup": (
                    recorded_end_to_end / estimated_end_to_end
                    if estimated_end_to_end
                    else None
                ),
            }
        )
    reference_totals = [
        math.fsum(run[index] for run in all_reference_runs)
        for index in range(repeats)
    ]
    optimized_totals = [
        math.fsum(run[index] for run in all_optimized_runs)
        for index in range(repeats)
    ]
    overall_reference = statistics.median(reference_totals)
    overall_optimized = statistics.median(optimized_totals)
    overall_reduction = 1.0 - overall_optimized / overall_reference
    maze600 = next(row for row in cases if row["case"] == "maze600")
    recorded_controller_total = math.fsum(
        float(row["recorded_controller_seconds"]) for row in cases
    )
    estimated_controller_total = math.fsum(
        float(row["estimated_optimized_controller_seconds"]) for row in cases
    )
    recorded_end_to_end_total = math.fsum(
        float(row["recorded_controller_plus_repair_seconds"]) for row in cases
    )
    estimated_end_to_end_total = math.fsum(
        float(row["estimated_optimized_controller_plus_repair_seconds"])
        for row in cases
    )
    controller_reduction = (
        1.0 - estimated_controller_total / recorded_controller_total
        if recorded_controller_total
        else 0.0
    )
    end_to_end_reduction = (
        1.0 - estimated_end_to_end_total / recorded_end_to_end_total
        if recorded_end_to_end_total
        else 0.0
    )
    performance_gate_passed = (
        overall_reduction >= 0.35
        and float(maze600["feature_time_reduction"]) >= 0.50
        and controller_reduction >= 0.25
        and end_to_end_reduction >= 0.08
    )
    report = {
        "schema": BENCHMARK_SCHEMA,
        "schema_version": 2,
        "collection": _portable_location(root),
        "repeats": repeats,
        "maximum_decisions_per_case": maximum_decisions,
        "floating_tolerance": tolerance,
        "feature_backend": feature_backend,
        "cases": cases,
        "overall": {
            "reference_median_seconds": overall_reference,
            "reference_p95_seconds": _percentile95(reference_totals),
            "optimized_median_seconds": overall_optimized,
            "optimized_p95_seconds": _percentile95(optimized_totals),
            "feature_time_reduction": overall_reduction,
            "speedup": overall_reference / overall_optimized,
            "estimated_controller_time_reduction": controller_reduction,
            "estimated_controller_speedup": (
                recorded_controller_total / estimated_controller_total
                if estimated_controller_total
                else None
            ),
            "estimated_end_to_end_time_reduction": end_to_end_reduction,
            "estimated_end_to_end_speedup": (
                recorded_end_to_end_total / estimated_end_to_end_total
                if estimated_end_to_end_total
                else None
            ),
            "end_to_end_measurement": (
                "estimate_from_recorded_v1_controller_and_repair timings; "
                "quick/formal paired wall time remains a promotion gate"
            ),
        },
        "performance_gate": {
            "overall_required_reduction": 0.35,
            "maze600_required_reduction": 0.50,
            "controller_required_reduction": 0.25,
            "estimated_end_to_end_required_reduction": 0.08,
            "passed": performance_gate_passed,
        },
        "native_backend_required": (
            not performance_gate_passed and feature_backend != "native"
        ),
    }
    _write_json(Path(output).resolve(), report)
    return report


__all__ = [
    "BENCHMARK_SCHEMA",
    "REGISTERED_CASES",
    "run_controller_feature_benchmark",
]
