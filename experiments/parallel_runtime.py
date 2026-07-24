from __future__ import annotations

import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable


_THREADPOOL_LIMIT: Any | None = None
# The registered fairness audit deliberately stops at four isolated lanes.
# Larger values may improve throughput but were not part of the frozen pilot
# protocol and therefore cannot be selected automatically.
AUTO_LANE_CANDIDATES = (1, 2, 3, 4)


def physical_cpu_sets() -> list[tuple[int, ...]]:
    """Return available logical CPUs grouped by physical core."""

    available = (
        set(os.sched_getaffinity(0))
        if callable(getattr(os, "sched_getaffinity", None))
        else set(range(os.cpu_count() or 1))
    )
    grouped: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(available):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            package = int((root / "physical_package_id").read_text().strip())
            core = int((root / "core_id").read_text().strip())
        except (OSError, ValueError):
            package, core = 0, cpu
        grouped.setdefault((package, core), []).append(cpu)
    return [tuple(values) for _key, values in sorted(grouped.items())]


def isolated_lane_cpu_sets(requested: int) -> list[tuple[int, ...]]:
    if requested <= 0:
        raise ValueError("parallel lane count must be positive")
    cores = physical_cpu_sets()
    if len(cores) < requested:
        raise ValueError(
            f"requested {requested} isolated lanes but only {len(cores)} cores are available"
        )
    return cores[:requested]


def initialize_isolated_worker(cpu_sets: tuple[tuple[int, ...], ...]) -> None:
    """Limit a process to one physical core and one native thread."""

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    identity = getattr(__import__("multiprocessing").current_process(), "_identity", ())
    lane = (int(identity[-1]) - 1) % len(cpu_sets) if identity else 0
    assigned = set(cpu_sets[lane])
    setter = getattr(os, "sched_setaffinity", None)
    if callable(setter):
        setter(0, assigned)
    global _THREADPOOL_LIMIT
    try:
        from threadpoolctl import threadpool_limits

        _THREADPOOL_LIMIT = threadpool_limits(limits=1)
    except ImportError:
        _THREADPOOL_LIMIT = None


def parallel_runtime_metadata(lanes: int) -> dict[str, Any]:
    cpu_sets = isolated_lane_cpu_sets(lanes)
    return {
        "schema": "lns2.isolated_parallel_runtime.v1",
        "lane_count": lanes,
        "physical_core_count": len(physical_cpu_sets()),
        "lane_cpu_sets": [list(values) for values in cpu_sets],
        "native_thread_limit": 1,
    }


def candidate_lane_counts(maximum: int | None = None) -> tuple[int, ...]:
    """Return registered lane counts supported by the current WSL topology."""

    available = len(physical_cpu_sets())
    limit = available if maximum is None else min(available, int(maximum))
    values = tuple(value for value in AUTO_LANE_CANDIDATES if value <= limit)
    if not values:
        return (1,)
    return tuple(sorted(set(values)))


def select_parallel_lane_count(
    measurements: Iterable[dict[str, Any]],
    *,
    memory_limit_bytes: int = 20 * 1024**3,
) -> dict[str, Any]:
    """Select the fastest registered lane count that preserves timing labels.

    Each measurement contains paired strict and parallel PP samples plus the
    observed peak resident memory.  The helper is intentionally independent of
    the collector so it can be unit-tested and reused by S3 data collection.
    """

    rows = []
    for raw in measurements:
        row = dict(raw)
        lanes = int(row["lanes"])
        strict = list(map(float, row.get("strict_pp_seconds", ())))
        parallel = list(map(float, row.get("parallel_pp_seconds", ())))
        if not strict or len(strict) != len(parallel):
            raise ValueError("parallel audit requires paired non-empty PP samples")
        if any(value < 0.0 for value in (*strict, *parallel)):
            raise ValueError("parallel audit PP samples must be nonnegative")
        strict_median = statistics.median(strict)
        parallel_median = statistics.median(parallel)
        ordered_strict = sorted(strict)
        ordered_parallel = sorted(parallel)
        p95_index = max(0, min(len(strict) - 1, math.ceil(0.95 * len(strict)) - 1))
        strict_p95 = ordered_strict[p95_index]
        parallel_p95 = ordered_parallel[p95_index]
        median_inflation = parallel_median / max(1e-12, strict_median) - 1.0
        p95_inflation = parallel_p95 / max(1e-12, strict_p95) - 1.0
        rank_correlation = float(row.get("cost_rank_correlation", 0.0))
        peak_memory = int(row.get("peak_memory_bytes", 0))
        strict_controller = list(map(float, row.get("strict_controller_seconds", ())))
        parallel_controller = list(map(float, row.get("parallel_controller_seconds", ())))
        if bool(strict_controller) != bool(parallel_controller) or (
            strict_controller and len(strict_controller) != len(parallel_controller)
        ):
            raise ValueError("parallel audit controller samples must be paired")
        controller_time_ratio = (
            statistics.median(parallel_controller)
            / max(1e-12, statistics.median(strict_controller))
            if strict_controller
            else 1.0
        )
        checks = {
            "median_pp_inflation": median_inflation <= 0.03 + 1e-12,
            "p95_pp_inflation": p95_inflation <= 0.05 + 1e-12,
            "cost_rank_correlation": rank_correlation + 1e-12 >= 0.98,
            "controller_time_ratio": abs(controller_time_ratio - 1.0) <= 0.05 + 1e-12,
            "memory": peak_memory <= int(memory_limit_bytes),
        }
        rows.append(
            {
                **row,
                "lanes": lanes,
                "median_pp_inflation": median_inflation,
                "p95_pp_inflation": p95_inflation,
                "controller_time_ratio": controller_time_ratio,
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    if not rows:
        raise ValueError("parallel audit has no measurements")
    passing = [row for row in rows if bool(row["passed"])]
    selected = max((int(row["lanes"]) for row in passing), default=1)
    return {
        "schema": "lns2.training_parallelism_audit.v1",
        "selected_lanes": selected,
        "attempts": sorted(rows, key=lambda row: int(row["lanes"])),
        "passed": bool(passing),
    }


__all__ = [
    "initialize_isolated_worker",
    "isolated_lane_cpu_sets",
    "candidate_lane_counts",
    "parallel_runtime_metadata",
    "physical_cpu_sets",
    "select_parallel_lane_count",
]
