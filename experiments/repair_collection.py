from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA_VERSION = 1
EPISODE_SCHEMA = "lns2.repair_episode.v1"
COUNTERFACTUAL_SCHEMA = "lns2.counterfactual.v1"
POLICY_DESTROY_STRATEGIES = {
    "official_adaptive": "Adaptive",
    "fixed_target": "Target",
    "fixed_collision": "Collision",
    "fixed_random": "Random",
}


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _plain(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_write_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def state_fingerprint(state: dict[str, Any]) -> str:
    """Hash deterministic solver state while excluding wall-clock and context."""

    keys = (
        "initialized",
        "initial_solution_complete",
        "feasible",
        "done",
        "iteration",
        "rows",
        "cols",
        "sum_of_costs",
        "num_of_colliding_pairs",
        "low_level",
        "obstacles",
        "conflict_edges",
        "agents",
    )
    return _fingerprint({key: _plain(state[key]) for key in keys})


def select_seed_agents(state: dict[str, Any], maximum: int) -> list[int]:
    if maximum <= 0:
        raise ValueError("maximum seed count must be positive")
    conflicting = [
        agent for agent in state["agents"] if int(agent["conflict_degree"]) > 0
    ]
    selected: list[int] = []

    def add(values: Iterable[dict[str, Any]]) -> None:
        for value in values:
            agent_id = int(value["id"])
            if agent_id not in selected and len(selected) < maximum:
                selected.append(agent_id)

    add(
        sorted(
            conflicting,
            key=lambda item: (
                -int(item["conflict_degree"]),
                -int(item["delay"]),
                int(item["id"]),
            ),
        )[:2]
    )
    add(
        sorted(
            conflicting,
            key=lambda item: (
                -int(item["delay"]),
                -int(item["conflict_degree"]),
                int(item["id"]),
            ),
        )[:2]
    )
    remaining = [
        int(item["id"])
        for item in conflicting
        if int(item["id"]) not in selected
    ]
    rng = random.Random(int(state_fingerprint(state)[:16], 16))
    rng.shuffle(remaining)
    for agent_id in remaining:
        if len(selected) >= maximum:
            break
        selected.append(agent_id)
    return selected


def candidate_actions(
    state: dict[str, Any],
    maximum_seeds: int,
    heuristics: list[str],
    neighborhood_sizes: list[int],
) -> list[dict[str, Any]]:
    supported = {"target", "collision", "random"}
    if not heuristics or any(value not in supported for value in heuristics):
        raise ValueError("counterfactual heuristics must be target, collision, or random")
    if not neighborhood_sizes or any(value <= 0 for value in neighborhood_sizes):
        raise ValueError("counterfactual neighborhood sizes must be positive")
    return [
        {
            "mode": "seed",
            "heuristic": heuristic,
            "seed_agent": seed_agent,
            "neighborhood_size": size,
        }
        for seed_agent in select_seed_agents(state, maximum_seeds)
        for heuristic in heuristics
        for size in neighborhood_sizes
    ]


def _load_environment_module() -> Any:
    try:
        import lns2_env
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "lns2_env is unavailable; build the native module and set PYTHONPATH"
        ) from error
    return lns2_env


def _context(row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "split",
        "map_id",
        "task_id",
        "layout_mode",
        "layout_variant",
        "scenario_type",
        "task_variant",
        "agent_count",
        "topology_metrics",
        "dominant_flow_ratio",
        "hotspot_skew",
        "required_bottleneck_crossing_ratio",
        "mean_shortest_distance",
    )
    return {key: _plain(row.get(key)) for key in keys}


def _make_environment(
    dataset_root: str,
    row: dict[str, Any],
    environment_config: dict[str, Any],
    destroy_strategy: str,
) -> Any:
    module = _load_environment_module()
    split_root = Path(dataset_root) / str(row["split"])
    return module.LNS2RepairEnv(
        str(split_root / str(row["map_file"])),
        str(split_root / str(row["scenario_file"])),
        agent_count=int(row["agent_count"]),
        time_limit=float(environment_config["time_limit"]),
        neighborhood_size=int(environment_config["neighborhood_size"]),
        destroy_strategy=destroy_strategy,
        replan_algorithm=str(environment_config["replan_algorithm"]),
        use_sipp=bool(environment_config["use_sipp"]),
        max_repair_iterations=int(environment_config["max_repair_iterations"]),
        screen=0,
        context=_context(row),
    )


def _low_level_delta(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, int]:
    return {
        key: int(after["low_level"][key]) - int(before["low_level"][key])
        for key in ("expanded", "generated", "reopened", "runs")
    }


def _conflict_auc(values: list[int]) -> float:
    return sum(
        (float(values[index]) + float(values[index + 1])) / 2.0
        for index in range(len(values) - 1)
    )


def _episode_id(row: dict[str, Any], solver_seed: int, policy: str) -> str:
    return f"{row['task_id']}__seed_{solver_seed:04d}__{policy}"


def _qualification_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    solver_seed = int(job["solver_seed"])
    try:
        environment = _make_environment(
            job["dataset_root"], row, job["environment"], "Adaptive"
        )
        state = _plain(environment.reset(seed=solver_seed))
        return {
            "schema_version": SCHEMA_VERSION,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "initial_conflicts": int(state["num_of_colliding_pairs"]),
            "repairable": not bool(state["done"]),
            "initial_feasible": bool(state["feasible"]),
            "initial_complete": bool(state["initial_solution_complete"]),
            "state_fingerprint": state_fingerprint(state),
            "status": "ok",
            "error": None,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def _valid_episode_trace(path: Path, run_fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        rows = _read_jsonl(path)
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not rows
        or rows[-1].get("event") != "finish"
        or rows[-1].get("run_fingerprint") != run_fingerprint
    ):
        return None
    return rows[-1].get("summary")


def _baseline_worker(job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    policy = str(job["policy"])
    solver_seed = int(job["solver_seed"])
    episode_id = _episode_id(row, solver_seed, policy)
    output_root = Path(job["output_root"])
    trace_path = output_root / "episodes" / str(row["split"]) / policy / f"{episode_id}.jsonl"
    relative_trace = trace_path.relative_to(output_root).as_posix()
    if job["resume"]:
        summary = _valid_episode_trace(trace_path, job["run_fingerprint"])
        if summary is not None:
            return {
                "schema_version": SCHEMA_VERSION,
                "episode_id": episode_id,
                "split": row["split"],
                "map_id": row["map_id"],
                "task_id": row["task_id"],
                "layout_mode": row["layout_mode"],
                "task_variant": row.get("task_variant"),
                "agent_count": int(row["agent_count"]),
                "solver_seed": solver_seed,
                "policy": policy,
                "trace_file": relative_trace,
                "status": "resumed",
                "summary": summary,
                "error": None,
            }
    try:
        environment = _make_environment(
            job["dataset_root"],
            row,
            job["environment"],
            POLICY_DESTROY_STRATEGIES[policy],
        )
        state = _plain(environment.reset(seed=solver_seed))
        conflicts = [int(state["num_of_colliding_pairs"])]
        events: list[dict[str, Any]] = [
            {
                "schema": EPISODE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": job["run_fingerprint"],
                "event": "initial",
                "episode_id": episode_id,
                "policy": policy,
                "solver_seed": solver_seed,
                "state_fingerprint": state_fingerprint(state),
                "state": state,
            }
        ]
        step_runtime = 0.0
        while not bool(state["done"]):
            before = state
            action = {"mode": "official"}
            result = _plain(environment.step(action))
            state = result["observation"]
            metrics = result["metrics"]
            step_runtime += float(metrics["step_runtime"])
            conflicts.append(int(state["num_of_colliding_pairs"]))
            events.append(
                {
                    "schema": EPISODE_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "run_fingerprint": job["run_fingerprint"],
                    "event": "transition",
                    "episode_id": episode_id,
                    "action": action,
                    "before_fingerprint": state_fingerprint(before),
                    "after_fingerprint": state_fingerprint(state),
                    "metrics": metrics,
                    "low_level_delta": _low_level_delta(before, state),
                    "terminated": bool(result["terminated"]),
                    "truncated": bool(result["truncated"]),
                    "after": state,
                }
            )
        summary = {
            "initial_conflicts": conflicts[0],
            "final_conflicts": conflicts[-1],
            "repairable": conflicts[0] > 0,
            "success": bool(state["feasible"]),
            "truncated": bool(state["done"] and not state["feasible"]),
            "repair_iterations": len(conflicts) - 1,
            "conflict_trajectory": conflicts,
            "conflict_auc": _conflict_auc(conflicts),
            "initial_runtime": float(events[0]["state"]["runtime"]),
            "repair_step_runtime": step_runtime,
            "time_to_feasible": float(state["runtime"]) if state["feasible"] else None,
            "final_sum_of_costs": int(state["sum_of_costs"]),
        }
        events.append(
            {
                "schema": EPISODE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": job["run_fingerprint"],
                "event": "finish",
                "episode_id": episode_id,
                "success": bool(state["feasible"]),
                "final_fingerprint": state_fingerprint(state),
                "summary": summary,
            }
        )
        _write_jsonl(trace_path, events)
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "layout_mode": row["layout_mode"],
            "task_variant": row.get("task_variant"),
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_file": relative_trace,
            "status": "ok",
            "summary": summary,
            "error": None,
        }
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": episode_id,
            "split": row["split"],
            "map_id": row["map_id"],
            "task_id": row["task_id"],
            "agent_count": int(row["agent_count"]),
            "solver_seed": solver_seed,
            "policy": policy,
            "trace_file": None,
            "status": "error",
            "summary": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _decision_states(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events or events[0].get("event") != "initial":
        raise ValueError("baseline trace does not start with an initial event")
    decisions: list[dict[str, Any]] = []
    prefix: list[dict[str, Any]] = []
    initial = events[0]["state"]
    if not initial["done"]:
        decisions.append(
            {"decision_index": 0, "state": initial, "prefix_actions": []}
        )
    index = 1
    for event in events:
        if event.get("event") != "transition":
            continue
        prefix.append(_plain(event["action"]))
        state = event["after"]
        if not state["done"]:
            decisions.append(
                {
                    "decision_index": index,
                    "state": state,
                    "prefix_actions": list(prefix),
                }
            )
        index += 1
    return decisions


def _select_evenly(values: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if maximum <= 0:
        raise ValueError("maximum state count must be positive")
    if len(values) <= maximum:
        return values
    if maximum == 1:
        return [values[0]]
    indices = {
        round(index * (len(values) - 1) / (maximum - 1))
        for index in range(maximum)
    }
    return [values[index] for index in sorted(indices)]


def _trial_seed(
    episode_id: str,
    state_id: str,
    action: dict[str, Any],
    trial_index: int,
) -> int:
    value = _fingerprint(
        {
            "episode_id": episode_id,
            "state_id": state_id,
            "action": action,
            "trial_index": trial_index,
        }
    )
    return int(value[:16], 16) % (2**31)


def _horizon_outcomes(
    initial: dict[str, Any],
    points: list[dict[str, Any]],
    horizons: list[int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for horizon in horizons:
        if len(points) - 1 >= horizon:
            available = True
            point = points[horizon]
            selected = points[: horizon + 1]
        elif points[-1]["state"]["feasible"]:
            available = True
            point = points[-1]
            selected = list(points)
            while len(selected) < horizon + 1:
                selected.append(
                    {
                        **point,
                        "step": len(selected),
                        "step_runtime": 0.0,
                    }
                )
        else:
            available = False
            point = points[-1]
            selected = points
        conflicts = [int(item["state"]["num_of_colliding_pairs"]) for item in selected]
        solved_step = next(
            (
                int(item["step"])
                for item in selected
                if bool(item["state"]["feasible"])
            ),
            None,
        )
        results.append(
            {
                "horizon": horizon,
                "available": available,
                "executed_steps": min(horizon, len(points) - 1),
                "solved": bool(point["state"]["feasible"]),
                "solved_step": solved_step,
                "conflicts_after": int(point["state"]["num_of_colliding_pairs"]),
                "conflict_reduction": int(initial["num_of_colliding_pairs"])
                - int(point["state"]["num_of_colliding_pairs"]),
                "conflict_auc": _conflict_auc(conflicts) if available else None,
                "sum_of_costs_after": int(point["state"]["sum_of_costs"]),
                "cost_improvement": int(initial["sum_of_costs"])
                - int(point["state"]["sum_of_costs"]),
                "low_level_delta": _low_level_delta(initial, point["state"]),
                "branch_runtime": sum(
                    float(item.get("step_runtime", 0.0)) for item in selected[1:]
                ),
                "time_to_feasible": (
                    sum(
                        float(item.get("step_runtime", 0.0))
                        for item in selected[1 : solved_step + 1]
                    )
                    if solved_step is not None
                    else None
                ),
            }
        )
    return results


def _counterfactual_worker(job: dict[str, Any]) -> dict[str, Any]:
    manifest = job["manifest"]
    episode_id = str(manifest["episode_id"])
    output_root = Path(job["output_root"])
    episode_root = output_root / "counterfactual" / str(manifest["split"]) / episode_id
    metadata_path = episode_root / "metadata.json"
    relative_metadata = metadata_path.relative_to(output_root).as_posix()
    if job["resume"] and metadata_path.is_file():
        metadata = _read_json(metadata_path)
        if (
            metadata.get("run_fingerprint") == job["run_fingerprint"]
            and metadata.get("complete") is True
        ):
            metadata = dict(metadata)
            metadata["status"] = "resumed"
            return metadata
    try:
        trace_path = output_root / str(manifest["trace_file"])
        events = _read_jsonl(trace_path)
        decisions = _select_evenly(
            _decision_states(events), int(job["counterfactual"]["max_states_per_episode"])
        )
        states: list[dict[str, Any]] = []
        outcomes: list[dict[str, Any]] = []
        errors: list[dict[str, Any]] = []
        row = job["row"]
        solver_seed = int(manifest["solver_seed"])
        horizons = sorted(int(value) for value in job["counterfactual"]["horizons"])
        maximum_horizon = max(horizons)
        for decision in decisions:
            state = decision["state"]
            fingerprint = state_fingerprint(state)
            state_id = f"{episode_id}__decision_{int(decision['decision_index']):04d}"
            actions = candidate_actions(
                state,
                int(job["counterfactual"]["max_seed_agents"]),
                list(job["counterfactual"]["heuristics"]),
                [int(value) for value in job["counterfactual"]["neighborhood_sizes"]],
            )
            states.append(
                {
                    "schema": COUNTERFACTUAL_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "run_fingerprint": job["run_fingerprint"],
                    "episode_id": episode_id,
                    "state_id": state_id,
                    "decision_index": int(decision["decision_index"]),
                    "state_fingerprint": fingerprint,
                    "prefix_actions": decision["prefix_actions"],
                    "candidate_count": len(actions),
                    "state": state,
                }
            )
            for candidate_index, candidate in enumerate(actions):
                for trial_index in range(int(job["counterfactual"]["trials"])):
                    branch_seed = _trial_seed(
                        episode_id, state_id, candidate, trial_index
                    )
                    try:
                        environment = _make_environment(
                            job["dataset_root"], row, job["environment"], "Adaptive"
                        )
                        replayed = _plain(environment.reset(seed=solver_seed))
                        for prefix_action in decision["prefix_actions"]:
                            if replayed["done"]:
                                raise RuntimeError("replay terminated before the decision state")
                            replayed = _plain(environment.step(prefix_action))["observation"]
                        replayed_fingerprint = state_fingerprint(replayed)
                        if replayed_fingerprint != fingerprint:
                            raise RuntimeError(
                                "replay fingerprint mismatch: "
                                f"expected {fingerprint}, got {replayed_fingerprint}"
                            )
                        action = dict(candidate)
                        action["random_seed"] = branch_seed
                        points = [
                            {
                                "step": 0,
                                "state": replayed,
                                "action": None,
                                "metrics": None,
                                "step_runtime": 0.0,
                            }
                        ]
                        current = replayed
                        for step in range(1, maximum_horizon + 1):
                            if current["done"]:
                                break
                            requested = action if step == 1 else {"mode": "official"}
                            result = _plain(environment.step(requested))
                            current = result["observation"]
                            points.append(
                                {
                                    "step": step,
                                    "state": current,
                                    "action": requested,
                                    "metrics": result["metrics"],
                                    "step_runtime": float(
                                        result["metrics"]["step_runtime"]
                                    ),
                                    "terminated": bool(result["terminated"]),
                                    "truncated": bool(result["truncated"]),
                                }
                            )
                        outcomes.append(
                            {
                                "schema": COUNTERFACTUAL_SCHEMA,
                                "schema_version": SCHEMA_VERSION,
                                "run_fingerprint": job["run_fingerprint"],
                                "episode_id": episode_id,
                                "state_id": state_id,
                                "state_fingerprint": fingerprint,
                                "candidate_index": candidate_index,
                                "candidate_action": action,
                                "trial_index": trial_index,
                                "trial_seed": branch_seed,
                                "action_valid": bool(
                                    points[1]["metrics"]["action_valid"]
                                ),
                                "conflict_trajectory": [
                                    int(point["state"]["num_of_colliding_pairs"])
                                    for point in points
                                ],
                                "steps": [
                                    {
                                        key: value
                                        for key, value in point.items()
                                        if key != "state"
                                    }
                                    | {
                                        "state_fingerprint": state_fingerprint(
                                            point["state"]
                                        ),
                                        "conflicts": int(
                                            point["state"]["num_of_colliding_pairs"]
                                        ),
                                        "sum_of_costs": int(
                                            point["state"]["sum_of_costs"]
                                        ),
                                    }
                                    for point in points
                                ],
                                "horizon_outcomes": _horizon_outcomes(
                                    replayed, points, horizons
                                ),
                            }
                        )
                    except Exception as error:
                        errors.append(
                            {
                                "schema": COUNTERFACTUAL_SCHEMA,
                                "schema_version": SCHEMA_VERSION,
                                "episode_id": episode_id,
                                "state_id": state_id,
                                "candidate_index": candidate_index,
                                "candidate_action": candidate,
                                "trial_index": trial_index,
                                "trial_seed": branch_seed,
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
        states_path = episode_root / "states.jsonl"
        outcomes_path = episode_root / "outcomes.jsonl"
        errors_path = episode_root / "errors.jsonl"
        _write_jsonl(states_path, states)
        _write_jsonl(outcomes_path, outcomes)
        _write_jsonl(errors_path, errors)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": job["run_fingerprint"],
            "episode_id": episode_id,
            "split": manifest["split"],
            "state_count": len(states),
            "outcome_count": len(outcomes),
            "error_count": len(errors),
            "states_file": states_path.relative_to(output_root).as_posix(),
            "outcomes_file": outcomes_path.relative_to(output_root).as_posix(),
            "errors_file": errors_path.relative_to(output_root).as_posix(),
            "metadata_file": relative_metadata,
            "complete": True,
            "status": "ok" if not errors else "error",
        }
        _write_json(metadata_path, metadata)
        return metadata
    except Exception as error:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": job["run_fingerprint"],
            "episode_id": episode_id,
            "split": manifest["split"],
            "state_count": 0,
            "outcome_count": 0,
            "error_count": 1,
            "metadata_file": relative_metadata,
            "complete": False,
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }


def _run_jobs(
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    jobs: list[dict[str, Any]],
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if workers == 1 or len(jobs) <= 1:
        return [worker(job) for job in jobs]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=context
    ) as executor:
        return list(executor.map(worker, jobs))


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported repair collection config schema")
    seeds = [int(value) for value in config.get("solver_seeds", [])]
    if not seeds or len(seeds) != len(set(seeds)) or any(value < 0 for value in seeds):
        raise ValueError("solver_seeds must be unique non-negative integers")
    policies = list(config.get("policies", []))
    if (
        not policies
        or len(policies) != len(set(policies))
        or any(value not in POLICY_DESTROY_STRATEGIES for value in policies)
    ):
        raise ValueError("collection config contains an unknown policy")
    environment = dict(config.get("environment", {}))
    required_environment = {
        "time_limit",
        "max_repair_iterations",
        "neighborhood_size",
        "replan_algorithm",
        "use_sipp",
    }
    if not required_environment.issubset(environment):
        raise ValueError("collection config omits environment settings")
    if (
        float(environment["time_limit"]) <= 0
        or int(environment["max_repair_iterations"]) <= 0
        or int(environment["neighborhood_size"]) <= 0
    ):
        raise ValueError("collection environment limits must be positive")
    counterfactual = dict(config.get("counterfactual", {}))
    if counterfactual.get("source_policy") != "official_adaptive":
        raise ValueError("the first counterfactual source must be official_adaptive")
    horizons = [int(value) for value in counterfactual.get("horizons", [])]
    if (
        not horizons
        or len(horizons) != len(set(horizons))
        or any(value <= 0 for value in horizons)
    ):
        raise ValueError("counterfactual horizons must be unique and positive")
    if (
        int(counterfactual.get("max_states_per_episode", 0)) <= 0
        or int(counterfactual.get("trials", 0)) <= 0
    ):
        raise ValueError("counterfactual state and trial counts must be positive")
    minimum_conflicts = int(counterfactual.get("minimum_initial_conflicts", 1))
    maximum_conflicts = counterfactual.get("maximum_initial_conflicts")
    if minimum_conflicts <= 0 or (
        maximum_conflicts is not None
        and int(maximum_conflicts) < minimum_conflicts
    ):
        raise ValueError("counterfactual initial-conflict bounds are invalid")
    if not isinstance(counterfactual.get("require_source_success", False), bool):
        raise ValueError("counterfactual require_source_success must be boolean")
    maximum_agent_count = counterfactual.get("maximum_agent_count")
    if maximum_agent_count is not None and int(maximum_agent_count) <= 0:
        raise ValueError("counterfactual maximum_agent_count must be positive")
    candidate_actions(
        {
            "initialized": True,
            "initial_solution_complete": True,
            "feasible": False,
            "done": False,
            "iteration": 0,
            "rows": 1,
            "cols": 2,
            "sum_of_costs": 1,
            "num_of_colliding_pairs": 1,
            "low_level": {"expanded": 0, "generated": 0, "reopened": 0, "runs": 0},
            "obstacles": [0, 0],
            "conflict_edges": [[0, 1]],
            "agents": [
                {"id": 0, "delay": 0, "conflict_degree": 1, "path": [0]},
                {"id": 1, "delay": 0, "conflict_degree": 1, "path": [1]},
            ],
        },
        int(counterfactual["max_seed_agents"]),
        list(counterfactual["heuristics"]),
        [int(value) for value in counterfactual["neighborhood_sizes"]],
    )


def _counterfactual_source_eligible(
    row: dict[str, Any], counterfactual: dict[str, Any]
) -> bool:
    return _counterfactual_source_reason(row, counterfactual) == "eligible"


def _counterfactual_source_reason(
    row: dict[str, Any], counterfactual: dict[str, Any]
) -> str:
    summary = row.get("summary", {})
    if not bool(summary.get("repairable")):
        return "not_repairable"
    initial_conflicts = int(summary.get("initial_conflicts", 0))
    if initial_conflicts < int(counterfactual.get("minimum_initial_conflicts", 1)):
        return "below_minimum_initial_conflicts"
    maximum_conflicts = counterfactual.get("maximum_initial_conflicts")
    if maximum_conflicts is not None and initial_conflicts > int(maximum_conflicts):
        return "above_maximum_initial_conflicts"
    if bool(counterfactual.get("require_source_success", False)) and not bool(
        summary.get("success")
    ):
        return "source_policy_unsolved"
    maximum_agent_count = counterfactual.get("maximum_agent_count")
    if maximum_agent_count is not None and int(row.get("agent_count", 0)) > int(
        maximum_agent_count
    ):
        return "above_maximum_agent_count"
    return "eligible"


def recover_counterfactual_manifest(output: str | Path) -> dict[str, Any]:
    output_root = Path(output).resolve()
    run_config = _read_json(output_root / "run_config.json")
    run_fingerprint = str(run_config["run_fingerprint"])
    rows = []
    invalid = []
    pattern = "counterfactual/*/*/metadata.json"
    for metadata_path in sorted(output_root.glob(pattern)):
        metadata = _read_json(metadata_path)
        reason = None
        if not bool(metadata.get("complete")):
            reason = "incomplete_metadata"
        elif str(metadata.get("run_fingerprint")) != run_fingerprint:
            reason = "run_fingerprint_mismatch"
        else:
            for key, count_key in (
                ("states_file", "state_count"),
                ("outcomes_file", "outcome_count"),
                ("errors_file", "error_count"),
            ):
                path = output_root / str(metadata.get(key, ""))
                if not path.is_file():
                    reason = f"missing_{key}"
                    break
                if len(_read_jsonl(path)) != int(metadata.get(count_key, -1)):
                    reason = f"count_mismatch_{key}"
                    break
        if reason is None:
            rows.append(metadata)
        else:
            invalid.append(
                {
                    "metadata_file": metadata_path.relative_to(output_root).as_posix(),
                    "reason": reason,
                }
            )
    rows.sort(key=lambda row: str(row["episode_id"]))
    _write_jsonl(output_root / "counterfactual_manifest.jsonl", rows)
    summary = _update_summary(output_root)
    expected = []
    source_path = output_root / "counterfactual_source_manifest.jsonl"
    if source_path.is_file():
        expected = [
            str(row["episode_id"])
            for row in _read_jsonl(source_path)
            if bool(row.get("eligible"))
        ]
    recovered_ids = {str(row["episode_id"]) for row in rows}
    return {
        "recovered_count": len(rows),
        "expected_eligible_count": len(expected),
        "missing_episode_ids": sorted(set(expected) - recovered_ids),
        "invalid_metadata": invalid,
        "summary": summary,
    }


def _dataset_fingerprint(dataset_root: Path) -> str:
    summary_path = dataset_root / "dataset_summary.json"
    if not summary_path.is_file():
        raise ValueError(f"missing dataset summary: {summary_path}")
    resolved_root = dataset_root.resolve()
    paths = {summary_path.resolve()}
    for manifest_path in sorted(dataset_root.glob("*/manifest.jsonl")):
        paths.add(manifest_path.resolve())
        for row in _read_jsonl(manifest_path):
            for key in (
                "map_file",
                "scenario_file",
                "map_metadata_file",
                "task_file",
            ):
                path = (manifest_path.parent / str(row[key])).resolve()
                try:
                    path.relative_to(resolved_root)
                except ValueError as error:
                    raise ValueError(
                        f"dataset manifest path escapes its root: {path}"
                    ) from error
                if not path.is_file():
                    raise ValueError(f"dataset input is missing: {path}")
                paths.add(path)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(resolved_root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_dataset_rows(dataset_root: Path, splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        manifest_path = dataset_root / split / "manifest.jsonl"
        if not manifest_path.is_file():
            raise ValueError(f"missing dataset split manifest: {manifest_path}")
        for row in _read_jsonl(manifest_path):
            if str(row.get("split")) != split:
                raise ValueError(f"manifest row crosses split boundary: {manifest_path}")
            rows.append(row)
    return rows


def _effective_config(
    config: dict[str, Any],
    max_states: int | None,
    max_seed_agents: int | None,
    neighborhood_sizes: list[int] | None,
    trials: int | None,
    horizons: list[int] | None,
) -> dict[str, Any]:
    value = json.loads(json.dumps(config))
    counterfactual = value["counterfactual"]
    if max_states is not None:
        counterfactual["max_states_per_episode"] = max_states
    if max_seed_agents is not None:
        counterfactual["max_seed_agents"] = max_seed_agents
    if neighborhood_sizes is not None:
        counterfactual["neighborhood_sizes"] = neighborhood_sizes
    if trials is not None:
        counterfactual["trials"] = trials
    if horizons is not None:
        counterfactual["horizons"] = horizons
    _validate_config(value)
    return value


def _prepare_run(
    dataset_root: Path,
    output_root: Path,
    config: dict[str, Any],
    splits: list[str],
    resume: bool,
) -> tuple[str, dict[str, Any]]:
    dataset_hash = _dataset_fingerprint(dataset_root)
    configuration_hash = _fingerprint(config)
    run_fingerprint = _fingerprint(
        {
            "dataset_fingerprint": dataset_hash,
            "configuration_fingerprint": configuration_hash,
            "splits": splits,
        }
    )
    run_config = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_hash,
        "configuration": config,
        "configuration_fingerprint": configuration_hash,
        "splits": splits,
        "run_fingerprint": run_fingerprint,
    }
    path = output_root / "run_config.json"
    if path.is_file():
        existing = _read_json(path)
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ValueError("output contains a different dataset or collection config")
        if not resume:
            raise ValueError("output already exists; pass --resume to continue it")
    else:
        _write_json(path, run_config)
    return run_fingerprint, run_config


def _qualification_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "run_count": len(rows),
        "error_count": sum(row["status"] == "error" for row in rows),
        "repairable_count": sum(bool(row.get("repairable")) for row in rows),
        "by_split": {},
    }
    for split in sorted({str(row["split"]) for row in rows}):
        selected = [row for row in rows if row["split"] == split]
        valid = [row for row in selected if row["status"] == "ok"]
        repairable = sum(bool(row["repairable"]) for row in valid)
        result["by_split"][split] = {
            "run_count": len(selected),
            "valid_count": len(valid),
            "repairable_count": repairable,
            "repairable_rate": repairable / len(valid) if valid else 0.0,
            "mean_initial_conflicts": (
                sum(int(row["initial_conflicts"]) for row in valid) / len(valid)
                if valid
                else 0.0
            ),
        }
    return result


def _update_summary(output_root: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    qualification_path = output_root / "qualification_manifest.jsonl"
    if qualification_path.is_file():
        summary["qualification"] = _qualification_summary(
            _read_jsonl(qualification_path)
        )
    baseline_path = output_root / "collection_manifest.jsonl"
    if baseline_path.is_file():
        rows = _read_jsonl(baseline_path)
        summary["baseline"] = {
            "episode_count": len(rows),
            "error_count": sum(row["status"] == "error" for row in rows),
            "success_count": sum(
                bool(row.get("summary", {}).get("success"))
                for row in rows
                if row.get("summary")
            ),
            "repairable_count": sum(
                bool(row.get("summary", {}).get("repairable"))
                for row in rows
                if row.get("summary")
            ),
        }
    counterfactual_path = output_root / "counterfactual_manifest.jsonl"
    if counterfactual_path.is_file():
        rows = _read_jsonl(counterfactual_path)
        summary["counterfactual"] = {
            "episode_count": len(rows),
            "state_count": sum(int(row.get("state_count", 0)) for row in rows),
            "outcome_count": sum(int(row.get("outcome_count", 0)) for row in rows),
            "error_count": sum(int(row.get("error_count", 0)) for row in rows),
        }
    source_path = output_root / "counterfactual_source_manifest.jsonl"
    if source_path.is_file():
        rows = _read_jsonl(source_path)
        summary["counterfactual_sources"] = {
            "episode_count": len(rows),
            "eligible_count": sum(bool(row.get("eligible")) for row in rows),
            "by_reason": dict(
                sorted(collections.Counter(str(row["reason"]) for row in rows).items())
            ),
        }
    _write_json(output_root / "summary.json", summary)
    return summary


def run_collection(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    phase: str = "all",
    splits: list[str] | None = None,
    workers: int | None = None,
    resume: bool = False,
    max_episodes: int | None = None,
    max_states: int | None = None,
    max_seed_agents: int | None = None,
    neighborhood_sizes: list[int] | None = None,
    trials: int | None = None,
    horizons: list[int] | None = None,
) -> dict[str, Any]:
    if phase not in {"qualify", "baseline", "counterfactual", "all"}:
        raise ValueError("phase must be qualify, baseline, counterfactual, or all")
    dataset_root = Path(dataset).resolve()
    output_root = Path(output).resolve()
    config = _effective_config(
        _read_json(Path(config_path)),
        max_states,
        max_seed_agents,
        neighborhood_sizes,
        trials,
        horizons,
    )
    dataset_summary = _read_json(dataset_root / "dataset_summary.json")
    available_splits = list(dataset_summary["splits"])
    requested_splits = splits or available_splits
    if not requested_splits or any(value not in available_splits for value in requested_splits):
        raise ValueError("requested split is not present in the dataset")
    worker_count = int(workers if workers is not None else config.get("workers", 4))
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    if max_states is not None and max_states <= 0:
        raise ValueError("max_states must be positive")
    run_fingerprint, _ = _prepare_run(
        dataset_root, output_root, config, requested_splits, resume
    )
    rows = _load_dataset_rows(dataset_root, requested_splits)
    environment = dict(config["environment"])
    solver_seeds = [int(value) for value in config["solver_seeds"]]

    if phase in {"qualify", "all"}:
        qualification_path = output_root / "qualification_manifest.jsonl"
        existing_qualification = (
            _read_jsonl(qualification_path)
            if resume and qualification_path.is_file()
            else []
        )
        existing_index = {
            (str(row["task_id"]), int(row["solver_seed"])): row
            for row in existing_qualification
            if row.get("status") == "ok"
        }
        jobs = [
            {
                "dataset_root": str(dataset_root),
                "row": row,
                "solver_seed": seed,
                "environment": environment,
            }
            for row in rows
            for seed in solver_seeds
            if (str(row["task_id"]), seed) not in existing_index
        ]
        qualification = list(existing_index.values())
        qualification.extend(_run_jobs(_qualification_worker, jobs, worker_count))
        qualification.sort(
            key=lambda row: (str(row["split"]), str(row["task_id"]), int(row["solver_seed"]))
        )
        _write_jsonl(qualification_path, qualification)

    qualification_path = output_root / "qualification_manifest.jsonl"
    qualification = _read_jsonl(qualification_path) if qualification_path.is_file() else []
    qualification_index = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in qualification
        if row["status"] == "ok"
    }
    pairs = [(row, seed) for row in rows for seed in solver_seeds]
    if max_episodes is not None:
        pairs.sort(
            key=lambda item: (
                not bool(
                    qualification_index.get(
                        (str(item[0]["task_id"]), int(item[1])), {}
                    ).get("repairable")
                ),
                str(item[0]["split"]),
                str(item[0]["task_id"]),
                int(item[1]),
            )
        )
        pairs = pairs[:max_episodes]

    if phase in {"baseline", "all"}:
        jobs = [
            {
                "dataset_root": str(dataset_root),
                "output_root": str(output_root),
                "row": row,
                "solver_seed": seed,
                "policy": policy,
                "environment": environment,
                "run_fingerprint": run_fingerprint,
                "resume": resume,
            }
            for row, seed in pairs
            for policy in config["policies"]
        ]
        baseline = _run_jobs(_baseline_worker, jobs, worker_count)
        baseline.sort(key=lambda row: str(row["episode_id"]))
        _write_jsonl(output_root / "collection_manifest.jsonl", baseline)

    if phase in {"counterfactual", "all"}:
        manifest_path = output_root / "collection_manifest.jsonl"
        if not manifest_path.is_file():
            raise ValueError("counterfactual phase requires a baseline collection manifest")
        baseline = _read_jsonl(manifest_path)
        row_index = {str(row["task_id"]): row for row in rows}
        eligible_splits = set(config["counterfactual"]["eligible_splits"])
        source_policy = str(config["counterfactual"]["source_policy"])
        considered = [
            row
            for row in baseline
            if row["policy"] == source_policy
            and row["split"] in eligible_splits
            and row["status"] != "error"
        ]
        source_selection = []
        for row in considered:
            reason = _counterfactual_source_reason(
                row, config["counterfactual"]
            )
            source_selection.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": str(row["episode_id"]),
                    "split": str(row["split"]),
                    "map_id": str(row["map_id"]),
                    "task_id": str(row["task_id"]),
                    "solver_seed": int(row["solver_seed"]),
                    "initial_conflicts": int(
                        row.get("summary", {}).get("initial_conflicts", 0)
                    ),
                    "source_success": bool(
                        row.get("summary", {}).get("success")
                    ),
                    "eligible": reason == "eligible",
                    "reason": reason,
                }
            )
        _write_jsonl(
            output_root / "counterfactual_source_manifest.jsonl",
            sorted(source_selection, key=lambda row: str(row["episode_id"])),
        )
        selected = [
            row
            for row in considered
            if _counterfactual_source_eligible(row, config["counterfactual"])
        ]
        jobs = [
            {
                "dataset_root": str(dataset_root),
                "output_root": str(output_root),
                "row": row_index[str(manifest["task_id"])],
                "manifest": manifest,
                "environment": environment,
                "counterfactual": config["counterfactual"],
                "run_fingerprint": run_fingerprint,
                "resume": resume,
            }
            for manifest in selected
        ]
        counterfactual = _run_jobs(_counterfactual_worker, jobs, worker_count)
        counterfactual.sort(key=lambda row: str(row["episode_id"]))
        _write_jsonl(
            output_root / "counterfactual_manifest.jsonl", counterfactual
        )

    return _update_summary(output_root)
