from __future__ import annotations

import collections
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import append_jsonl_fsync as _append_jsonl
from experiments.closed_loop_confirmation import (
    fixed_budget_conflict_auc,
    generate_online_candidates,
    load_frozen_policy_bundle,
    online_candidate_rows,
    repair_random_seed,
    score_online_candidates,
)
from research.studies.representation.local_representation_audit import analyze_static_grid
from research.studies.policy.policy_visited_aggregation import candidate_core
from experiments.repair_collection import (
    SCHEMA_VERSION,
    CollectionLockError,
    _CollectionRunLock,
    _dataset_fingerprint,
    _fingerprint,
    _policy_train_dataset_lookup as _dataset_lookup,
    _low_level_delta,
    _make_environment,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from research.studies.sequential.sequential_credit_audit import (
    _half_candidates,
    _jaccard,
    _mean,
    _prepare_state_worker,
    _quantile,
    _resolve,
    _sha256,
    _spearman_effectiveness,
    _state_storage_id,
    best_ids,
    pareto_ids,
)


PROBE_SCHEMA = "lns2.repair_order_probe.v1"
TRIAL_SCHEMA = "lns2.repair_order_trial.v1"
REPORT_SCHEMA = "lns2.repair_order_report.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    "experiments/_common.py",
    "research/studies/sequential/repair_order_probe.py",
    "research/studies/sequential/sequential_credit_audit.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/repair_collection.py",
    "src/python_bindings.cpp",
    "src/jsonl_observer.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported repair-order probe config")
    selection = dict(config.get("selection", {}))
    if (
        int(selection.get("map_count", 0)) != 12
        or int(selection.get("states_per_map", 0)) != 2
        or int(selection.get("total_states", 0)) != 24
        or int(selection.get("candidates_per_state", 0)) != 6
        or list(map(int, selection.get("required_sizes", []))) != [4, 8, 16]
    ):
        raise ValueError("formal repair-order selection must be 12 maps x 2 states x 6 candidates")
    conditions = dict(config.get("order_conditions", {}))
    if (
        int(conditions.get("random_crn_trials", 0)) != 8
        or int(conditions.get("deterministic_repeats", 0)) != 2
        or list(map(str, conditions.get("deterministic_policies", [])))
        != [
            "agent_id_ascending",
            "conflict_degree_descending",
            "delay_descending",
            "path_length_descending",
        ]
    ):
        raise ValueError("repair-order conditions differ from the preregistration")
    if int(config.get("horizon", 0)) != 4:
        raise ValueError("repair-order probe requires Horizon 4")
    if int(config.get("workers", 0)) != 4:
        raise ValueError("formal repair-order probe uses four workers")
    if float(config.get("trial_process_timeout_seconds", 0.0)) != 180.0:
        raise ValueError("formal repair-order trial timeout must be 180 seconds")
    if int(config.get("bootstrap_samples", 0)) != 5000:
        raise ValueError("formal repair-order analysis requires 5,000 bootstrap samples")
    environment = dict(config.get("environment", {}))
    if str(environment.get("replan_algorithm")) != "PP":
        raise ValueError("repair_order is only defined for PP repair")


def _registered_inputs(config: dict[str, Any]) -> dict[str, Any]:
    source = dict(config["source"])
    root = _resolve(source["root"])
    paths = {
        "selected_states": root / str(source["selected_states"]),
        "index": root / str(source["index"]),
        "report": root / str(source["report"]),
        "source_run_config": root / str(source["run_config"]),
    }
    expected = {
        "selected_states": str(source["selected_states_sha256"]),
        "index": str(source["index_sha256"]),
        "report": str(source["report_sha256"]),
        "source_run_config": str(source["run_config_sha256"]),
    }
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"registered repair-order input is missing: {path}")
        hashes[name] = _sha256(path)
        if hashes[name] != expected[name]:
            raise ValueError(f"registered repair-order input SHA256 mismatch: {name}")
    dataset_root = _resolve(config["dataset"])
    dataset_hash = _dataset_fingerprint(dataset_root)
    if dataset_hash != str(config["dataset_fingerprint"]):
        raise ValueError("repair-order dataset fingerprint mismatch")
    implementation = {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
        if (PROJECT_ROOT / relative).is_file()
    }
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_hash,
        "implementation": implementation,
    }


def _variance(values: Iterable[float | int]) -> float:
    numbers = list(map(float, values))
    return statistics.pvariance(numbers) if numbers else 0.0


def diagnose_existing(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        config = _read_json(_resolve(config))
    _validate_config(config)
    registered = _registered_inputs(config)
    index = _read_jsonl(Path(registered["paths"]["index"]))
    metrics: dict[str, Any] = {
        "state_count": len(index),
        "candidate_count": sum(len(row["candidates"]) for row in index),
        "trial_count": sum(
            len(candidate["trial_outcomes"])
            for row in index
            for candidate in row["candidates"]
        ),
    }
    for horizon in ("h1", "h4"):
        spearman, pareto, best = [], [], []
        for state in index:
            first = _half_candidates(state, (0, 1), horizon)
            second = _half_candidates(state, (2, 3), horizon)
            spearman.append(_spearman_effectiveness(first, second, horizon=horizon))
            pareto.append(
                _jaccard(pareto_ids(first, horizon), pareto_ids(second, horizon))
            )
            best.append(_jaccard(best_ids(first, horizon), best_ids(second, horizon)))
        metrics[f"{horizon}_split_spearman"] = _mean(spearman)
        metrics[f"{horizon}_pareto_jaccard"] = _mean(pareto)
        metrics[f"{horizon}_best_jaccard"] = _mean(best)
    candidate_rows = [candidate for row in index for candidate in row["candidates"]]
    c1_values = [
        [outcome["h1"]["final_conflicts"] for outcome in candidate["trial_outcomes"]]
        for candidate in candidate_rows
    ]
    h4_values = [
        [outcome["h4"]["conflict_auc"] for outcome in candidate["trial_outcomes"]]
        for candidate in candidate_rows
    ]
    metrics["variable_c1_conflict_fraction"] = _mean(
        _variance(values) > 0.0 for values in c1_values
    )
    metrics["variable_h4_auc_fraction"] = _mean(
        _variance(values) > 0.0 for values in h4_values
    )
    same_c1 = [
        h4
        for c1, h4 in zip(c1_values, h4_values)
        if _variance(c1) == 0.0
    ]
    metrics["same_c1_but_variable_h4_fraction"] = _mean(
        _variance(values) > 0.0 for values in same_c1
    )
    expected = dict(config["expected_diagnosis"])
    tolerance = float(expected.pop("absolute_tolerance"))
    checks = {}
    for name, value in expected.items():
        actual = metrics.get(name)
        if isinstance(value, int):
            checks[name] = int(actual) == value
        else:
            checks[name] = math.isclose(
                float(actual), float(value), rel_tol=0.0, abs_tol=tolerance
            )
    return {
        "schema": PROBE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "metrics": metrics,
        "registered_inputs": registered,
    }


def select_probe_states(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    selection = dict(config["selection"])
    required_split = str(config["source"]["required_split"])
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if str(row.get("split")) == required_split:
            grouped[str(row["map_id"])].append(_plain(row))
    if len(grouped) != int(selection["map_count"]):
        raise ValueError("repair-order source does not contain exactly 12 Train maps")
    fields = list(map(str, selection["state_balance_fields"]))
    counts = {field: collections.Counter() for field in fields}
    selected: list[dict[str, Any]] = []
    map_order = sorted(
        grouped,
        key=lambda value: _fingerprint(
            {"namespace": "repair-order-map-order-v1", "map_id": value}
        ),
    )
    for round_index in range(int(selection["states_per_map"])):
        for map_id in map_order:
            used = {str(row["state_id"]) for row in selected if str(row["map_id"]) == map_id}
            candidates = [row for row in grouped[map_id] if str(row["state_id"]) not in used]
            if not candidates:
                raise ValueError(f"map lacks repair-order source states: {map_id}")

            def key(row: dict[str, Any]) -> tuple[Any, ...]:
                field_counts = [counts[field][str(row[field])] for field in fields]
                diversity = sum(
                    str(row[field]) != str(previous[field])
                    for previous in selected
                    if str(previous["map_id"]) == map_id
                    for field in fields
                )
                return (
                    sum(field_counts),
                    max(field_counts, default=0),
                    -diversity,
                    _fingerprint(
                        {
                            "namespace": "repair-order-state-selection-v1",
                            "round": round_index,
                            "state_id": row["state_id"],
                        }
                    ),
                )

            chosen = min(candidates, key=key)
            selected.append(chosen)
            for field in fields:
                counts[field][str(chosen[field])] += 1
    selected.sort(key=lambda row: (str(row["map_id"]), str(row["state_id"])))
    if len(selected) != int(selection["total_states"]):
        raise ValueError("repair-order selected state count differs from registration")
    return selected


def select_probe_candidates(state: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = list(state["candidates"])
    by_id = {str(row["candidate_id"]): row for row in candidates}
    source_id = str(state["source_selected_candidate_id"])
    if source_id not in by_id:
        raise ValueError("frozen v1 source selection is absent from candidate pool")
    chosen = [source_id]

    def candidate_hash(row: dict[str, Any]) -> str:
        return _fingerprint(
            {
                "namespace": "repair-order-candidate-selection-v1",
                "state_id": state["state_id"],
                "candidate_id": row["candidate_id"],
            }
        )

    for size in map(int, config["selection"]["required_sizes"]):
        options = [
            row
            for row in candidates
            if int(row["actual_size"]) == size and str(row["candidate_id"]) not in chosen
        ]
        if not options and not any(int(by_id[value]["actual_size"]) == size for value in chosen):
            raise ValueError(f"candidate pool lacks required size {size}")
        if options and not any(int(by_id[value]["actual_size"]) == size for value in chosen):
            chosen.append(str(min(options, key=candidate_hash)["candidate_id"]))
    for row in sorted(candidates, key=candidate_hash):
        candidate_id = str(row["candidate_id"])
        if candidate_id not in chosen:
            chosen.append(candidate_id)
        if len(chosen) == int(config["selection"]["candidates_per_state"]):
            break
    if len(chosen) != int(config["selection"]["candidates_per_state"]):
        raise ValueError("candidate pool cannot supply six unique neighborhoods")
    return [candidate_core(by_id[value]) for value in chosen]


def repair_order_for_policy(
    state: dict[str, Any], agents: Iterable[int], policy: str
) -> list[int]:
    selected = set(map(int, agents))
    by_id = {int(row["id"]): row for row in state["agents"]}
    if not selected or not selected.issubset(by_id):
        raise ValueError("repair-order agents are missing from the state")
    if policy == "agent_id_ascending":
        key = lambda agent: (agent,)
    elif policy == "conflict_degree_descending":
        key = lambda agent: (-int(by_id[agent]["conflict_degree"]), agent)
    elif policy == "delay_descending":
        key = lambda agent: (-int(by_id[agent]["delay"]), agent)
    elif policy == "path_length_descending":
        key = lambda agent: (-len(by_id[agent]["path"]), agent)
    else:
        raise ValueError(f"unknown repair-order policy: {policy}")
    return sorted(selected, key=key)


def solution_fingerprint(state: dict[str, Any]) -> str:
    agents = [
        {
            "id": int(row["id"]),
            "path": list(map(int, row["path"])),
            "path_cost": int(row["path_cost"]),
            "delay": int(row["delay"]),
            "conflict_degree": int(row["conflict_degree"]),
        }
        for row in sorted(state["agents"], key=lambda value: int(value["id"]))
    ]
    return _fingerprint(
        {
            "sum_of_costs": int(state["sum_of_costs"]),
            "num_of_colliding_pairs": int(state["num_of_colliding_pairs"]),
            "conflict_edges": sorted(
                [sorted(map(int, edge)) for edge in state["conflict_edges"]]
            ),
            "agents": agents,
        }
    )


def _state_crn_seed(state: dict[str, Any], trial_index: int) -> int:
    forbidden = {
        int(seed)
        for candidate in state["candidates"]
        for seed in candidate["proposal_seeds"]
    }
    value = int(
        _fingerprint(
            {
                "namespace": "repair-order-common-random-number-v1",
                "state_id": state["state_id"],
                "trial_index": trial_index,
            }
        )[:16],
        16,
    ) % (2**31)
    while value in forbidden:
        value = (value + 1) % (2**31)
    return value


def _deterministic_seed(state: dict[str, Any], policy: str) -> int:
    return int(
        _fingerprint(
            {
                "namespace": "repair-order-deterministic-condition-v1",
                "state_id": state["state_id"],
                "policy": policy,
            }
        )[:16],
        16,
    ) % (2**31)


def order_conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    values = [
        {
            "condition_id": f"random_crn_{index:02d}",
            "kind": "random_crn",
            "trial_index": index,
            "policy": None,
            "repeat": None,
        }
        for index in range(int(config["order_conditions"]["random_crn_trials"]))
    ]
    for policy in map(str, config["order_conditions"]["deterministic_policies"]):
        for repeat in range(int(config["order_conditions"]["deterministic_repeats"])):
            values.append(
                {
                    "condition_id": f"{policy}__repeat_{repeat}",
                    "kind": "deterministic",
                    "trial_index": None,
                    "policy": policy,
                    "repeat": repeat,
                }
            )
    return values


def _replay_prefix(environment: Any, source: dict[str, Any]) -> dict[str, Any]:
    state = _plain(environment.reset(seed=int(source["solver_seed"])))
    for action in source["prefix_actions"]:
        if bool(state["done"]):
            raise RuntimeError("repair-order prefix terminated before selected state")
        state = _plain(environment.step(action))["observation"]
    if state_fingerprint(state) != str(source["state_fingerprint"]):
        raise RuntimeError("repair-order prefix replay fingerprint mismatch")
    return state


def execute_order_trial(
    environment: Any,
    source: dict[str, Any],
    candidate: dict[str, Any],
    condition: dict[str, Any],
    bundle: Any,
    proposal_config: dict[str, Any],
) -> dict[str, Any]:
    state = _replay_prefix(environment, source)
    initial_state_fingerprint = state_fingerprint(state)
    initial_solution_fingerprint = solution_fingerprint(state)
    static_grid = analyze_static_grid(state)
    agents = sorted(map(int, candidate["agents"]))
    if str(condition["kind"]) == "random_crn":
        requested_order: list[int] = []
        initial_seed = _state_crn_seed(source, int(condition["trial_index"]))
    else:
        requested_order = repair_order_for_policy(state, agents, str(condition["policy"]))
        initial_seed = _deterministic_seed(source, str(condition["policy"]))
    trajectory = [int(state["num_of_colliding_pairs"])]
    steps = []
    low_level_totals: collections.Counter[str] = collections.Counter()
    h1 = None
    for rollout_step in range(4):
        if bool(state["done"]):
            break
        before = state
        before_hash = state_fingerprint(before)
        decision_index = int(source["decision_index"]) + rollout_step
        if rollout_step == 0:
            selected = candidate
            action = {
                "mode": "explicit_neighborhood",
                "agents": agents,
                "random_seed": initial_seed,
            }
            if requested_order:
                action["repair_order"] = requested_order
            controller = {
                "policy": "repair_order_probe_initial",
                "condition": condition,
                "selected_candidate_id": candidate["candidate_id"],
            }
        else:
            candidates, proposal_metrics = generate_online_candidates(
                environment,
                before,
                task_id=str(source["task_id"]),
                solver_seed=int(source["solver_seed"]),
                decision_index=decision_index,
                proposal_config=proposal_config,
            )
            candidate_rows = online_candidate_rows(before, candidates, static_grid=static_grid)
            selected_index, scores, margin = score_online_candidates(
                candidate_rows, bundle.models["realized_dynamic"]
            )
            selected = candidates[selected_index]
            seed = repair_random_seed(
                str(source["task_id"]),
                int(source["solver_seed"]),
                before_hash,
                decision_index,
                str(selected["candidate_id"]),
                selected["proposal_seeds"],
            )
            action = {
                "mode": "explicit_neighborhood",
                "agents": selected["agents"],
                "random_seed": seed,
            }
            controller = {
                "policy": "frozen_v1_realized_dynamic",
                "candidate_ids": [str(value["candidate_id"]) for value in candidates],
                "selected_candidate_id": str(selected["candidate_id"]),
                "selected_score": float(scores[selected_index]),
                "score_margin": float(margin),
                "proposal": proposal_metrics,
            }
        started = time.perf_counter()
        result = _plain(environment.step(action))
        repair_wall = time.perf_counter() - started
        after = result["observation"]
        metrics = dict(result["metrics"])
        actual_agents = sorted(map(int, metrics.get("neighborhood", [])))
        actual_order = list(map(int, metrics.get("repair_order", [])))
        if not bool(metrics.get("action_valid")) or actual_agents != sorted(map(int, action["agents"])):
            raise RuntimeError("repair-order explicit action was rejected or changed")
        order_applied = bool(actual_order)
        if order_applied and sorted(actual_order) != actual_agents:
            raise RuntimeError("actual PP repair order is not a neighborhood permutation")
        if order_applied and action.get("repair_order") and actual_order != list(action["repair_order"]):
            raise RuntimeError("actual PP repair order differs from the requested order")
        low_level = _low_level_delta(before, after)
        low_level_totals.update(low_level)
        trajectory.append(int(after["num_of_colliding_pairs"]))
        step = {
            "rollout_step": rollout_step + 1,
            "decision_index": decision_index,
            "before_fingerprint": before_hash,
            "after_fingerprint": state_fingerprint(after),
            "after_solution_fingerprint": solution_fingerprint(after),
            "selected_candidate_id": str(selected["candidate_id"]),
            "selected_agents": sorted(map(int, action["agents"])),
            "requested_repair_order": list(map(int, action.get("repair_order", []))),
            "actual_repair_order": actual_order,
            "order_applied": order_applied,
            "random_seed": int(action["random_seed"]),
            "conflicts_before": trajectory[-2],
            "conflicts_after": trajectory[-1],
            "feasible_after": bool(after["feasible"]),
            "low_level_delta": low_level,
            "repair_wall_seconds": repair_wall,
            "controller": controller,
        }
        steps.append(step)
        state = after
        if rollout_step == 0:
            h1 = {
                "feasible": bool(after["feasible"]),
                "final_conflicts": trajectory[-1],
                "conflict_auc": (trajectory[0] + trajectory[-1]) / 2.0,
                "solution_fingerprint": step["after_solution_fingerprint"],
            }
    if h1 is None:
        raise RuntimeError("repair-order trial executed no initial repair")
    success = bool(state["feasible"])
    padded = list(trajectory)
    padded.extend(([0] if success else [padded[-1]]) * (5 - len(padded)))
    return {
        "condition": condition,
        "initial_state_fingerprint": initial_state_fingerprint,
        "initial_solution_fingerprint": initial_solution_fingerprint,
        "initial_random_seed": initial_seed,
        "initial_requested_repair_order": requested_order,
        "initial_actual_repair_order": steps[0]["actual_repair_order"],
        "raw_conflict_trajectory": trajectory,
        "padded_conflict_trajectory": padded,
        "h1": h1,
        "h4": {
            "feasible": success,
            "final_conflicts": padded[4],
            "conflict_auc": fixed_budget_conflict_auc(trajectory, 4, success=success),
            "solution_fingerprint": solution_fingerprint(state),
        },
        "low_level": dict(low_level_totals),
        "steps": steps,
    }


def _trial_worker(job: dict[str, Any]) -> dict[str, Any]:
    source, candidate, condition = job["state_row"], job["candidate"], job["condition"]
    common = {
        "schema": TRIAL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": job["run_fingerprint"],
        "job_id": job["job_id"],
        "state_id": str(source["state_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "condition_id": str(condition["condition_id"]),
        "map_id": str(source["map_id"]),
        "task_id": str(source["task_id"]),
        "solver_seed": int(source["solver_seed"]),
    }
    try:
        environment = _make_environment(
            job["dataset_root"], job["row"], job["environment"], "Adaptive"
        )
        bundle = load_frozen_policy_bundle(job["frozen_models"], job["model_registration"])
        outcome = execute_order_trial(
            environment, source, candidate, condition, bundle, job["proposal"]
        )
        return {
            **common,
            "status": "ok",
            "complete": True,
            "outcome_count": 1,
            "outcome": outcome,
            "error": None,
        }
    except Exception as error:
        return {
            **common,
            "status": "error",
            "complete": False,
            "outcome_count": 0,
            "outcome": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _trial_path(output_root: Path, job: dict[str, Any]) -> Path:
    return (
        output_root
        / "trials"
        / _state_storage_id(str(job["state_id"]))
        / str(job["candidate_id"])
        / f"{job['condition_id']}.json"
    )


def _prepare_states(
    output_root: Path,
    selected: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]],
    config: dict[str, Any],
    run_fingerprint: str,
    resume: bool,
) -> list[dict[str, Any]]:
    jobs, resumed = [], []
    for source in selected:
        path = output_root / "prepared_states" / f"{_state_storage_id(str(source['state_id']))}.json"
        if resume and path.is_file():
            row = _read_json(path)
            if row.get("run_fingerprint") == run_fingerprint and bool(row.get("complete")):
                resumed.append({**row, "status": "resumed"})
                continue
        jobs.append(
            {
                "job_id": f"prepare__{_state_storage_id(str(source['state_id']))}",
                "state_row": source,
                "row": lookup[str(source["task_id"])],
                "dataset_root": str(_resolve(config["dataset"])),
                "environment": config["environment"],
                "proposal": config["proposal"],
                "run_fingerprint": run_fingerprint,
            }
        )
    by_job = {str(job["job_id"]): job for job in jobs}
    completed = []

    def record(result: dict[str, Any]) -> None:
        job = by_job[str(result["job_id"])]
        path = output_root / "prepared_states" / f"{_state_storage_id(str(job['state_row']['state_id']))}.json"
        _write_json(path, result)
        completed.append(result)

    if jobs:
        _run_jobs(
            _prepare_state_worker,
            jobs,
            int(config["workers"]),
            phase="repair-order-prepare",
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(config["prepare_process_timeout_seconds"]),
            on_result=record,
        )
    rows = sorted(resumed + completed, key=lambda value: str(value["state_id"]))
    _write_jsonl(output_root / "state_preparation_manifest.jsonl", rows)
    return rows


def _trial_jobs(
    selected: list[dict[str, Any]],
    lookup: dict[str, dict[str, Any]],
    config: dict[str, Any],
    run_fingerprint: str,
) -> list[dict[str, Any]]:
    jobs = []
    for source in selected:
        for candidate in source["probe_candidates"]:
            for condition in order_conditions(config):
                condition_id = str(condition["condition_id"])
                job_id = (
                    f"{_state_storage_id(str(source['state_id']))}__"
                    f"{candidate['candidate_id']}__{condition_id}"
                )
                jobs.append(
                    {
                        "job_id": job_id,
                        "state_id": str(source["state_id"]),
                        "candidate_id": str(candidate["candidate_id"]),
                        "condition_id": condition_id,
                        "condition": condition,
                        "state_row": source,
                        "candidate": candidate,
                        "row": lookup[str(source["task_id"])],
                        "solver_seed": int(source["solver_seed"]),
                        "dataset_root": str(_resolve(config["dataset"])),
                        "environment": config["environment"],
                        "proposal": config["proposal"],
                        "frozen_models": str(_resolve(config["frozen_models"])),
                        "model_registration": config["model_registration"],
                        "run_fingerprint": run_fingerprint,
                    }
                )
    return jobs


def _collect_trials(
    output_root: Path,
    jobs: list[dict[str, Any]],
    config: dict[str, Any],
    run_fingerprint: str,
    resume: bool,
) -> list[dict[str, Any]]:
    pending, resumed = [], []
    for job in jobs:
        path = _trial_path(output_root, job)
        if resume and path.is_file():
            row = _read_json(path)
            if row.get("run_fingerprint") == run_fingerprint and bool(row.get("complete")):
                resumed.append({**row, "status": "resumed"})
                continue
        pending.append(job)
    by_job = {str(job["job_id"]): job for job in pending}
    completed = []

    def record(result: dict[str, Any]) -> None:
        job = by_job[str(result["job_id"])]
        normalized = {
            **result,
            "schema": TRIAL_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
        }
        _write_json(_trial_path(output_root, job), normalized)
        completed.append(normalized)
        _append_jsonl(
            output_root / "progress.jsonl",
            {
                "job_id": normalized["job_id"],
                "state_id": normalized["state_id"],
                "candidate_id": normalized["candidate_id"],
                "condition_id": normalized["condition_id"],
                "status": normalized["status"],
            },
        )

    if pending:
        _run_jobs(
            _trial_worker,
            pending,
            int(config["workers"]),
            phase="repair-order-trial",
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(config["trial_process_timeout_seconds"]),
            on_result=record,
        )
    rows = sorted(
        resumed + completed,
        key=lambda row: (str(row["state_id"]), str(row["candidate_id"]), str(row["condition_id"])),
    )
    _write_jsonl(output_root / "trial_manifest.jsonl", rows)
    return rows


def _aggregate_outcomes(rows: list[dict[str, Any]], horizon: str) -> dict[str, float]:
    values = [row["outcome"][horizon] for row in rows]
    return {
        "feasible_rate": _mean(value["feasible"] for value in values),
        "final_conflicts": _mean(value["final_conflicts"] for value in values),
        "conflict_auc": _mean(value["conflict_auc"] for value in values),
    }


def _map_bootstrap(values: dict[str, list[float]], samples: int, seed: int) -> dict[str, Any]:
    maps = sorted(values)
    means = {key: _mean(values[key]) for key in maps}
    rng = random.Random(seed)
    estimates = [
        _mean(means[rng.choice(maps)] for _ in maps)
        for _ in range(samples)
    ]
    return {
        "unit": "map_id",
        "samples": samples,
        "mean": _mean(means.values()),
        "ci95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
        "by_map": dict(sorted(means.items())),
    }


def analyze_trials(
    selected: list[dict[str, Any]], rows: list[dict[str, Any]], config: dict[str, Any], *, formal: bool
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    policies = list(map(str, config["order_conditions"]["deterministic_policies"]))
    random_trials = int(config["order_conditions"]["random_crn_trials"])
    deterministic_repeats = int(config["order_conditions"]["deterministic_repeats"])
    integrity_errors = []
    crn_spearman, crn_pareto, crn_best = [], [], []
    solution_divergence, conflict_divergence, opportunities = [], [], []
    opportunities_by_map: dict[str, list[float]] = collections.defaultdict(list)
    near_oracle: dict[str, list[bool]] = {policy: [] for policy in policies}
    condition_index: list[dict[str, Any]] = []
    crn_seed_checks: dict[tuple[str, int], set[int]] = collections.defaultdict(set)
    for state in selected:
        half_rows = {"first": [], "second": []}
        for candidate in state["probe_candidates"]:
            key = (str(state["state_id"]), str(candidate["candidate_id"]))
            values = grouped.get(key, [])
            random_rows = sorted(
                [row for row in values if row["outcome"]["condition"]["kind"] == "random_crn"],
                key=lambda row: int(row["outcome"]["condition"]["trial_index"]),
            )
            deterministic = {
                policy: sorted(
                    [row for row in values if row["outcome"]["condition"].get("policy") == policy],
                    key=lambda row: int(row["outcome"]["condition"]["repeat"]),
                )
                for policy in policies
            }
            if len(random_rows) != random_trials or any(
                len(deterministic[policy]) != deterministic_repeats for policy in policies
            ):
                integrity_errors.append({"state_id": key[0], "candidate_id": key[1], "error": "condition coverage"})
                continue
            for row in random_rows:
                trial = int(row["outcome"]["condition"]["trial_index"])
                crn_seed_checks[(key[0], trial)].add(int(row["outcome"]["initial_random_seed"]))
            for policy, duplicates in deterministic.items():
                first, second = duplicates
                comparable = (
                    first["outcome"]["initial_actual_repair_order"],
                    first["outcome"]["h1"]["solution_fingerprint"],
                    first["outcome"]["h4"]["solution_fingerprint"],
                    first["outcome"]["raw_conflict_trajectory"],
                )
                repeated = (
                    second["outcome"]["initial_actual_repair_order"],
                    second["outcome"]["h1"]["solution_fingerprint"],
                    second["outcome"]["h4"]["solution_fingerprint"],
                    second["outcome"]["raw_conflict_trajectory"],
                )
                if comparable != repeated:
                    integrity_errors.append(
                        {"state_id": key[0], "candidate_id": key[1], "policy": policy, "error": "deterministic duplicate mismatch"}
                    )
            first_half = random_rows[: random_trials // 2]
            second_half = random_rows[random_trials // 2 :]
            half_rows["first"].append(
                {"candidate_id": key[1], "h4": _aggregate_outcomes(first_half, "h4")}
            )
            half_rows["second"].append(
                {"candidate_id": key[1], "h4": _aggregate_outcomes(second_half, "h4")}
            )
            policy_metrics = {
                policy: _aggregate_outcomes(deterministic[policy], "h4") for policy in policies
            }
            solution_divergence.append(
                len({deterministic[policy][0]["outcome"]["h1"]["solution_fingerprint"] for policy in policies}) > 1
            )
            conflict_divergence.append(
                len({deterministic[policy][0]["outcome"]["h1"]["final_conflicts"] for policy in policies}) > 1
            )
            random_auc = _mean(row["outcome"]["h4"]["conflict_auc"] for row in random_rows)
            oracle_auc = min(value["conflict_auc"] for value in policy_metrics.values())
            improvement = (random_auc - oracle_auc) / random_auc if random_auc > 0.0 else 0.0
            opportunities.append(improvement)
            opportunities_by_map[str(state["map_id"])].append(improvement)
            initial_conflicts = int(random_rows[0]["outcome"]["raw_conflict_trajectory"][0])
            scale = max(1.0, 4.0 * initial_conflicts)
            for policy in policies:
                regret = (float(policy_metrics[policy]["conflict_auc"]) - oracle_auc) / scale
                near_oracle[policy].append(
                    regret <= float(config["thresholds"]["fixed_rule_normalized_regret_tolerance"])
                )
            condition_index.append(
                {
                    "state_id": key[0],
                    "candidate_id": key[1],
                    "map_id": state["map_id"],
                    "random_h4": _aggregate_outcomes(random_rows, "h4"),
                    "deterministic_h4": policy_metrics,
                    "oracle_auc": oracle_auc,
                    "oracle_improvement": improvement,
                }
            )
        if half_rows["first"] and len(half_rows["first"]) == len(state["probe_candidates"]):
            crn_spearman.append(
                _spearman_effectiveness(half_rows["first"], half_rows["second"], horizon="h4")
            )
            crn_pareto.append(
                _jaccard(
                    pareto_ids(half_rows["first"], "h4"),
                    pareto_ids(half_rows["second"], "h4"),
                )
            )
            crn_best.append(
                _jaccard(
                    best_ids(half_rows["first"], "h4"),
                    best_ids(half_rows["second"], "h4"),
                )
            )
    bad_crn = [key for key, seeds in crn_seed_checks.items() if len(seeds) != 1]
    if bad_crn:
        integrity_errors.append({"error": "CRN seed differs across candidates", "count": len(bad_crn)})
    bootstrap = _map_bootstrap(
        opportunities_by_map, int(config["bootstrap_samples"]), int(config["bootstrap_seed"])
    )
    fixed_shares = {policy: _mean(values) for policy, values in near_oracle.items()}
    best_fixed_policy = max(fixed_shares, key=fixed_shares.get)
    metrics = {
        "state_count": len(selected),
        "candidate_count": sum(len(row["probe_candidates"]) for row in selected),
        "trial_count": len(rows),
        "integrity_error_count": len(integrity_errors),
        "crn_stability": {
            "mean_spearman": _mean(crn_spearman),
            "mean_pareto_jaccard": _mean(crn_pareto),
            "mean_best_jaccard": _mean(crn_best),
        },
        "order_effect": {
            "solution_divergence_fraction": _mean(solution_divergence),
            "c1_conflict_divergence_fraction": _mean(conflict_divergence),
            "mean_oracle_auc_improvement": _mean(opportunities),
            "positive_opportunity_fraction": _mean(value > 0.0 for value in opportunities),
            "bootstrap": bootstrap,
        },
        "fixed_rules": {
            "near_oracle_shares": fixed_shares,
            "best_policy": best_fixed_policy,
            "best_share": fixed_shares[best_fixed_policy],
        },
    }
    thresholds = dict(config["thresholds"])
    integrity = (
        not integrity_errors
        and (not formal or len(selected) == 24)
        and all(str(row.get("status")) in {"ok", "resumed"} and bool(row.get("complete")) for row in rows)
    )
    crn_stable = (
        metrics["crn_stability"]["mean_spearman"] >= float(thresholds["minimum_crn_spearman"])
        and metrics["crn_stability"]["mean_pareto_jaccard"] >= float(thresholds["minimum_crn_pareto_jaccard"])
        and metrics["crn_stability"]["mean_best_jaccard"] >= float(thresholds["minimum_crn_best_jaccard"])
    )
    order_material = (
        metrics["order_effect"]["solution_divergence_fraction"]
        >= float(thresholds["minimum_solution_divergence_fraction"])
        and metrics["order_effect"]["c1_conflict_divergence_fraction"]
        >= float(thresholds["minimum_c1_conflict_divergence_fraction"])
        and metrics["order_effect"]["mean_oracle_auc_improvement"]
        >= float(thresholds["minimum_oracle_auc_improvement"])
        and metrics["order_effect"]["positive_opportunity_fraction"]
        >= float(thresholds["minimum_positive_opportunity_fraction"])
        and bootstrap["ci95"][0] >= float(thresholds["bootstrap_lower_bound"])
    )
    fixed_dominates = (
        metrics["fixed_rules"]["best_share"] >= float(thresholds["fixed_rule_dominance_share"])
    )
    if not integrity:
        decision = "stop_integrity_failure"
    elif order_material and fixed_dominates:
        decision = "adopt_fixed_repair_order"
    elif order_material:
        decision = "advance_to_contextual_repair_order"
    elif crn_stable:
        decision = "advance_to_crn_expected_neighborhood_value"
    else:
        decision = "stop_neighborhood_ranking_and_rl"
    report = {
        "schema": REPORT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": formal,
        "decision": decision,
        "gates": {
            "integrity": integrity,
            "crn_stability": crn_stable,
            "repair_order_material": order_material,
            "fixed_rule_dominance": fixed_dominates,
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "errors": integrity_errors,
        "condition_index": condition_index,
    }
    return report


def _run_metadata(
    config: dict[str, Any], registered: dict[str, Any], selected: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    selected_signature = [
        {
            "state_id": row["state_id"],
            "candidate_ids": [value["candidate_id"] for value in row["probe_candidates"]],
        }
        for row in selected
    ]
    payload = {
        "configuration_fingerprint": _fingerprint(config),
        "registered_sha256": registered["sha256"],
        "dataset_fingerprint": registered["dataset_fingerprint"],
        "implementation": registered["implementation"],
        "selected": selected_signature,
    }
    run_fingerprint = _fingerprint(payload)
    conditions = len(order_conditions(config))
    return run_fingerprint, {
        "schema": PROBE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        **payload,
        "dataset": registered["dataset"],
        "state_count": len(selected),
        "candidate_count": sum(len(row["probe_candidates"]) for row in selected),
        "condition_count": conditions,
        "trial_count": sum(len(row["probe_candidates"]) for row in selected) * conditions,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    lines = [
        "# InitLNS PP Repair Order Probe",
        "",
        f"- Formal: `{report['formal']}`",
        f"- Decision: `{report['decision']}`",
        f"- States/candidates/trials: {metrics['state_count']} / {metrics['candidate_count']} / {metrics['trial_count']}",
        "",
        "## CRN Stability",
        "",
        f"- Spearman: {metrics['crn_stability']['mean_spearman']:.6f}",
        f"- Pareto Jaccard: {metrics['crn_stability']['mean_pareto_jaccard']:.6f}",
        f"- Best-set Jaccard: {metrics['crn_stability']['mean_best_jaccard']:.6f}",
        "",
        "## Repair Order",
        "",
        f"- C1 solution divergence: {metrics['order_effect']['solution_divergence_fraction']:.6f}",
        f"- C1 conflict divergence: {metrics['order_effect']['c1_conflict_divergence_fraction']:.6f}",
        f"- H4 oracle AUC improvement: {metrics['order_effect']['mean_oracle_auc_improvement']:.6f}",
        f"- Positive opportunity: {metrics['order_effect']['positive_opportunity_fraction']:.6f}",
        f"- Best fixed rule/share: {metrics['fixed_rules']['best_policy']} / {metrics['fixed_rules']['best_share']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_repair_order_probe(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    resume: bool = False,
    smoke_states: int | None = None,
) -> dict[str, Any]:
    if phase not in {"diagnose", "dry-run", "collect", "analyze", "all"}:
        raise ValueError(f"unsupported repair-order phase: {phase}")
    config = _read_json(_resolve(config_path))
    _validate_config(config)
    output_root = _resolve(output)
    output_root.mkdir(parents=True, exist_ok=True)
    diagnosis = diagnose_existing(config)
    if not diagnosis["passed"]:
        raise RuntimeError("existing sequential-credit diagnosis did not reproduce")
    _write_json(output_root / "existing_diagnosis.json", diagnosis)
    if phase == "diagnose":
        return diagnosis
    source_rows = _read_jsonl(Path(diagnosis["registered_inputs"]["paths"]["selected_states"]))
    selected = select_probe_states(source_rows, config)
    for row in selected:
        row["probe_candidates"] = select_probe_candidates(row, config)
    full_selected = selected
    if smoke_states is not None:
        if smoke_states <= 0:
            raise ValueError("smoke state count must be positive")
        selected = selected[:smoke_states]
    run_fingerprint, run_config = _run_metadata(
        config, diagnosis["registered_inputs"], selected
    )
    condition_count = len(order_conditions(config))
    dry_run = {
        "schema": PROBE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": smoke_states is None,
        "state_count": len(selected),
        "candidate_count": sum(len(row["probe_candidates"]) for row in selected),
        "condition_count": condition_count,
        "trial_count": sum(len(row["probe_candidates"]) for row in selected) * condition_count,
        "maximum_repairs": sum(len(row["probe_candidates"]) for row in selected)
        * condition_count
        * int(config["horizon"]),
        "states_by_map": dict(collections.Counter(str(row["map_id"]) for row in selected)),
    }
    _write_json(output_root / "dry_run.json", dry_run)
    _write_jsonl(output_root / "selected_states.jsonl", selected)
    if phase == "dry-run":
        return dry_run
    run_config_path = output_root / "run_config.json"
    if run_config_path.is_file():
        existing = _read_json(run_config_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("repair-order output fingerprint differs; refusing incompatible resume")
    _write_json(run_config_path, run_config)
    formal = smoke_states is None
    if phase in {"collect", "all"}:
        lookup = _dataset_lookup(Path(diagnosis["registered_inputs"]["dataset"]))
        status = {
            "schema": PROBE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "status": "running",
            "formal": formal,
        }
        _write_json(output_root / "run_status.json", status)
        try:
            with _CollectionRunLock(output_root, run_fingerprint, "repair-order-probe"):
                prepared = _prepare_states(
                    output_root, selected, lookup, config, run_fingerprint, resume
                )
                if len(prepared) != len(selected) or any(not bool(row.get("complete")) for row in prepared):
                    raise RuntimeError("repair-order state preparation failed")
                jobs = _trial_jobs(selected, lookup, config, run_fingerprint)
                trials = _collect_trials(
                    output_root, jobs, config, run_fingerprint, resume
                )
                if len(trials) != len(jobs) or any(not bool(row.get("complete")) for row in trials):
                    raise RuntimeError("repair-order trial collection is incomplete")
            status.update(
                {
                    "status": "complete",
                    "prepared_state_count": len(prepared),
                    "trial_count": len(trials),
                }
            )
            _write_json(output_root / "run_status.json", status)
            _write_json(output_root / "COLLECTION_COMPLETE.json", status)
        except BaseException as error:
            status.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            _write_json(output_root / "run_status.json", status)
            raise
        if phase == "collect":
            return status
    trial_path = output_root / "trial_manifest.jsonl"
    if not trial_path.is_file():
        raise ValueError("repair-order analysis requires trial_manifest.jsonl")
    report = analyze_trials(selected, _read_jsonl(trial_path), config, formal=formal)
    report["run_fingerprint"] = run_fingerprint
    report["existing_diagnosis"] = diagnosis["metrics"]
    condition_index = report.pop("condition_index")
    _write_jsonl(output_root / "condition_index.jsonl", condition_index)
    _write_json(output_root / "repair_order_report.json", report)
    _write_markdown(output_root / "repair_order_report.md", report)
    _write_json(
        output_root / "ANALYSIS_COMPLETE.json",
        {
            "schema": PROBE_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "status": "complete",
            "decision": report["decision"],
        },
    )
    return report


__all__ = [
    "CollectionLockError",
    "analyze_trials",
    "diagnose_existing",
    "execute_order_trial",
    "order_conditions",
    "repair_order_for_policy",
    "run_repair_order_probe",
    "select_probe_candidates",
    "select_probe_states",
    "solution_fingerprint",
]
