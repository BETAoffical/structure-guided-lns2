"""Explicit stage-two adapter; not a collector and never invoked by preflight."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

from lns2_selector.evaluation.path_quality_preflight import audit_paths, read_grid, scenario_prefix
from lns2_selector.solver.native import native_identity

HANDOFF_SCHEMA = "lns2.anytime_handoff.v1"


def paths_from_observation(state: dict) -> list[list[int]]:
    if state.get("feasible") is not True or type(state.get("num_of_colliding_pairs")) is not int or state["num_of_colliding_pairs"] != 0:
        raise ValueError("handoff requires an explicitly feasible observation")
    agents = sorted(state["agents"], key=lambda row: row["id"])
    if not agents or any(type(a["id"]) is not int for a in agents) or [a["id"] for a in agents] != list(range(len(agents))):
        raise ValueError("observation agent IDs must match the scenario ordering")
    paths = [list(agent["path"]) for agent in agents]
    if any(not path or any(type(cell) is not int for cell in path) for path in paths):
        raise ValueError("invalid observation paths")
    return paths


def modeled_completion(dispatch_seconds: float, makespan_steps: int, step_seconds: float) -> float:
    if type(makespan_steps) is not int or makespan_steps < 0:
        raise ValueError("makespan must be nonnegative integer steps")
    for value in (dispatch_seconds, step_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid execution model duration")
    if step_seconds == 0:
        raise ValueError("step_seconds must be positive")
    return dispatch_seconds + makespan_steps * step_seconds


def continue_with_official_anytime(
    module: Any, *, map_path: Path, scenario_path: Path, observation: dict,
    expected_native_sha256: str, planning_started: float, total_budget_seconds: float,
    stage2_seed: int, max_iterations: int = 0,
    initial_feasible_elapsed_seconds: float | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    """Charge validation/import/export against one caller-owned monotonic budget.

    max_iterations is for bounded functional tests only; zero means no iteration
    cap. Production admission remains outside this adapter and requires consent.
    """
    for value in (planning_started, total_budget_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("invalid planning clock/budget")
    if type(stage2_seed) is not int or stage2_seed < 0 or type(max_iterations) is not int or max_iterations < 0:
        raise ValueError("invalid stage-two seed/iteration limit")
    entered = clock()
    if entered < planning_started:
        raise ValueError("planning start is after the current monotonic clock")
    if getattr(module, "anytime_handoff_schema", None) != HANDOFF_SCHEMA:
        raise ValueError("native module lacks the required anytime handoff schema")
    identity = native_identity(module)
    if not expected_native_sha256 or identity["sha256"] != expected_native_sha256:
        raise ValueError("loaded native binary SHA mismatch")
    paths = paths_from_observation(observation)
    grid = read_grid(map_path)
    rows = scenario_prefix(map_path, scenario_path, grid, len(paths))
    starts = [(int(row[5]), int(row[4])) for row in rows]
    goals = [(int(row[7]), int(row[6])) for row in rows]
    initial = audit_paths(grid, starts, goals, paths)
    if not initial["feasible"] or any(len(p) > 1 and p[-2] == p[-1] for p in paths):
        raise ValueError("initial paths must be feasible and without terminal padding")
    if type(observation.get("sum_of_costs")) is not int or observation["sum_of_costs"] != initial["soc_steps"]:
        raise ValueError("observation SOC differs from actual input paths")
    before_native = clock()
    if before_native < entered:
        raise ValueError("monotonic clock moved backwards")
    available = entered - planning_started if initial_feasible_elapsed_seconds is None else initial_feasible_elapsed_seconds
    if type(available) not in (int, float) or not math.isfinite(available) or not 0 <= available <= entered - planning_started:
        raise ValueError("invalid first feasible availability timestamp")
    initial_within_budget = available <= total_budget_seconds
    remaining = max(0.0, planning_started + total_budget_seconds - before_native)
    native_result = None
    returned = paths
    if remaining > 0:
        native_result = module.optimize_feasible_paths(
            str(map_path), str(scenario_path), len(paths), paths, stage2_seed,
            remaining, max_iterations=max_iterations, strategy="Adaptive", neighborhood_size=8,
        )
        if native_result.get("schema") != HANDOFF_SCHEMA or native_result.get("initial_planner_called") is not False:
            raise ValueError("native did not attest direct path handoff")
        if native_result.get("initial_paths") != paths or native_result.get("initial_soc") != initial["soc_steps"]:
            raise ValueError("native changed the imported solution")
        if native_result.get("stage2_seed") != stage2_seed:
            raise ValueError("native stage-two seed mismatch")
        returned = native_result["paths"]
    final = audit_paths(grid, starts, goals, returned)
    if not final["feasible"] or final["soc_steps"] > initial["soc_steps"]:
        raise ValueError("official anytime returned infeasible or higher-cost paths")
    if native_result is not None and native_result.get("soc") != final["soc_steps"]:
        raise ValueError("native final SOC differs from paths")
    completed = clock()
    if completed < before_native:
        raise ValueError("monotonic clock moved backwards")
    elapsed = completed - planning_started
    dispatch = max(total_budget_seconds, elapsed)
    return {
        "schema": "lns2.two_stage_handoff.v1", "native_sha256": identity["sha256"],
        "stage2_seed": stage2_seed, "initial_paths": paths, "paths": returned,
        "initial_quality": initial, "final_quality": final,
        "initial_feasible_available_within_budget": initial_within_budget,
        "success_by_deadline": initial_within_budget,
        "stage2_called": native_result is not None,
        "stage2_remaining_budget_seconds": remaining,
        "total_planning_budget_seconds": total_budget_seconds,
        "actual_planning_wall_seconds": elapsed,
        "budget_overshoot_seconds": max(0.0, elapsed - total_budget_seconds),
        "dispatch_wall_seconds": dispatch,
        "execution_model": "fixed_budget_plan_then_execute",
        "native_result": native_result,
    }
