from __future__ import annotations

import concurrent.futures
import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import summarize_initial_state_complexity
from experiments.stride_lns import (
    FROZEN_FEATURE_DIMENSION,
    FROZEN_FEATURE_SCHEMA_ID,
    STRIDE_TRIAL_SCHEMA,
    post_structure_metrics,
)
from experiments.trace_replay import replay_prefix
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import generate_online_candidates
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


STRIDE_SELECTION_SCHEMA = "lns2.stride.state_selection.v1"
STRIDE_COLLECTION_SCHEMA = "lns2.stride.repair_collection.v1"
STRIDE_COLLECTION_PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_lns.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/online_selection.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)
FULL_POOL_PROPOSAL = {
    "heuristics": ["target", "collision", "random"],
    "neighborhood_sizes": [4, 8, 16],
    "candidates_per_family": 2,
}
PP_TRIAL_INDICES = (0, 1, 2, 3)


def stride_pp_seed(state_repair_fingerprint: str, trial_index: int) -> int:
    if trial_index not in PP_TRIAL_INDICES:
        raise ValueError("STRIDE PP trial index must be 0, 1, 2, or 3")
    return int(
        _fingerprint(
            {
                "namespace": "stride-lns-paired-pp-v1",
                "repair_state": str(state_repair_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def _selection_row_errors(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_strings = (
        "state_id",
        "map_id",
        "task_id",
        "split",
        "source_policy",
        "decision_stage",
        "source_root",
        "before_fingerprint",
    )
    for name in required_strings:
        if not isinstance(row.get(name), str) or not str(row[name]):
            errors.append(f"{name} must be a non-empty string")
    for name in ("solver_seed", "decision_index", "agent_count"):
        value = row.get(name)
        if type(value) is not int or int(value) < 0:
            errors.append(f"{name} must be a nonnegative integer")
    if int(row.get("agent_count", 0)) <= 0:
        errors.append("agent_count must be positive")
    prefix = row.get("prefix_actions")
    if not isinstance(prefix, list) or any(not isinstance(item, dict) for item in prefix):
        errors.append("prefix_actions must be a list of action objects")
    return errors


def load_stride_selection(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError("STRIDE state selection is empty")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("schema") not in {None, STRIDE_SELECTION_SCHEMA}:
            raise ValueError(f"selection row {index} has an unsupported schema")
        errors = _selection_row_errors(row)
        if errors:
            raise ValueError(f"selection row {index}: {'; '.join(errors)}")
        state_id = str(row["state_id"])
        if state_id in seen:
            raise ValueError(f"duplicate selected state: {state_id}")
        seen.add(state_id)
    return rows


def _replay_job(decision: dict[str, Any]) -> dict[str, Any]:
    source_root = Path(str(decision["source_root"])).resolve()
    run = _read_json(source_root / "run_config.json")
    dataset_root = Path(str(run["dataset"])).resolve()
    matches = [
        row
        for row in _load_dataset_rows(dataset_root, [str(decision["split"])])
        if str(row["task_id"]) == str(decision["task_id"])
    ]
    if len(matches) != 1:
        raise ValueError(
            f"selected task must resolve exactly once: {decision['task_id']}"
        )
    configuration = dict(run["configuration"])
    environment = dict(configuration["environment"])
    environment["max_repair_iterations"] = max(
        int(environment.get("max_repair_iterations", 0)),
        len(decision["prefix_actions"]) + 1,
    )
    return {
        "dataset_root": str(dataset_root),
        "row": matches[0],
        "environment": environment,
        "proposal": dict(configuration["proposal"]),
        "solver_seed": int(decision["solver_seed"]),
        "replay_destroy_strategy": "Adaptive",
    }


def _paired_action(agents: list[int], seed: int) -> dict[str, Any]:
    if not agents:
        raise ValueError("STRIDE explicit neighborhood cannot be empty")
    return {
        "mode": "explicit_neighborhood",
        "agents": list(map(int, agents)),
        "random_seed": int(seed),
        "pp_random_seed": int(seed),
    }


def _validate_native_repair(
    result: dict[str, Any], *, expected_agents: list[int], expected_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = dict(result["observation"])
    metrics = dict(result["metrics"])
    if metrics.get("step_applied") is not True:
        raise RuntimeError("native deadline ended before STRIDE repair was applied")
    if not isinstance(metrics.get("replan_success"), bool):
        raise RuntimeError("native repair omitted strict replan_success")
    neighborhood = metrics.get("neighborhood")
    if not isinstance(neighborhood, list) or sorted(map(int, neighborhood)) != sorted(
        map(int, expected_agents)
    ):
        raise RuntimeError("native neighborhood differs from STRIDE candidate")
    requested = metrics.get("requested_pp_random_seed")
    if type(requested) is not int or int(requested) != int(expected_seed):
        raise RuntimeError("native requested PP seed differs from STRIDE seed")
    repair_order = metrics.get("repair_order")
    if not isinstance(repair_order, list):
        raise RuntimeError("native repair_order is missing")
    applied = metrics.get("applied_pp_random_seed")
    expected_applied = int(expected_seed) if repair_order else -1
    if type(applied) is not int or int(applied) != expected_applied:
        raise RuntimeError("native applied PP seed differs from STRIDE seed")
    terminated = result.get("terminated")
    truncated = result.get("truncated")
    if type(terminated) is not bool or type(truncated) is not bool:
        raise RuntimeError("native repair omitted strict terminal flags")
    if bool(state.get("done")) != bool(terminated or truncated):
        raise RuntimeError("native done flag disagrees with terminal flags")
    if bool(state.get("feasible")) != bool(terminated):
        raise RuntimeError("native feasible flag disagrees with termination")
    return state, metrics


def _state_artifact_valid(
    payload: dict[str, Any], *, run_fingerprint: str, state_id: str
) -> bool:
    if (
        payload.get("schema") != STRIDE_COLLECTION_SCHEMA
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("state_id") != state_id
        or payload.get("complete") is not True
    ):
        return False
    candidates = payload.get("candidates")
    trials = payload.get("trials")
    if not isinstance(candidates, list) or not candidates or not isinstance(trials, list):
        return False
    expected = {
        (str(candidate["candidate_id"]), trial_index)
        for candidate in candidates
        for trial_index in PP_TRIAL_INDICES
    }
    observed = {
        (str(row.get("candidate_id")), int(row.get("trial_index", -1)))
        for row in trials
        if isinstance(row, dict)
    }
    return observed == expected and len(trials) == len(expected)


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _state_artifact_valid(
            existing, run_fingerprint=run_fingerprint, state_id=str(decision["state_id"])
        ):
            return {
                "state_id": str(decision["state_id"]),
                "state_file": str(output_path),
                "status": "resumed",
                "candidate_count": len(existing["candidates"]),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"completed STRIDE state artifact is invalid: {output_path}")

    replay = _replay_job(decision)
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    initial_fingerprint = state_fingerprint(state)
    if initial_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError(
            f"STRIDE replay mismatch for {decision['state_id']}: "
            f"expected {decision['before_fingerprint']}, got {initial_fingerprint}"
        )
    initial_repair_fingerprint = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    if before_conflicts <= 0 or bool(state.get("done")):
        raise ValueError("STRIDE selected state must be active and conflicting")
    state_summary = summarize_initial_state_complexity(state)
    proposal = {**dict(replay["proposal"]), **FULL_POOL_PROPOSAL}
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=proposal,
        state_hash=initial_fingerprint,
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    feature_engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={
            "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
        },
        dense_output=False,
    )
    feature_rows, feature_metrics = feature_engine.realized_rows(
        candidates, state_hash=initial_fingerprint
    )
    features_by_candidate = {
        str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
        for row in feature_rows
    }
    if len(candidates) != len(features_by_candidate):
        raise RuntimeError("STRIDE candidate and feature counts differ")
    if any(len(values) != FROZEN_FEATURE_DIMENSION for values in features_by_candidate.values()):
        raise RuntimeError("STRIDE realized feature dimension differs from frozen V2")

    trials: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        agents = list(map(int, candidate["agents"]))
        for trial_index in PP_TRIAL_INDICES:
            branch_environment, branch_state = replay_prefix(
                replay, decision["prefix_actions"]
            )
            branch_fingerprint = state_fingerprint(branch_state)
            if branch_fingerprint != initial_fingerprint:
                raise RuntimeError("STRIDE paired branch replay fingerprint changed")
            seed = stride_pp_seed(initial_repair_fingerprint, trial_index)
            action = _paired_action(agents, seed)
            result = _plain(branch_environment.step(action))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair_fingerprint = repair_structure_fingerprint(after)
            repair_outcome = classify_repair_outcome(
                before_fingerprint=initial_repair_fingerprint,
                after_fingerprint=after_repair_fingerprint,
                replan_success=bool(metrics["replan_success"]),
                conflicts_before=before_conflicts,
                conflicts_after=conflicts_after,
                feasible=bool(after.get("feasible")),
            )
            trials.append(
                {
                    "schema": STRIDE_TRIAL_SCHEMA,
                    "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
                    "state_id": str(decision["state_id"]),
                    "candidate_id": candidate_id,
                    "map_id": str(decision["map_id"]),
                    "split": str(decision["split"]),
                    "source_policy": str(decision["source_policy"]),
                    "decision_stage": str(decision["decision_stage"]),
                    "agent_count": int(decision["agent_count"]),
                    "before_conflicts": before_conflicts,
                    "before_fingerprint": initial_fingerprint,
                    "before_repair_fingerprint": initial_repair_fingerprint,
                    "features": features_by_candidate[candidate_id],
                    "trial_index": trial_index,
                    "pp_seed": seed,
                    "feasible": bool(after.get("feasible")),
                    "replan_success": bool(metrics["replan_success"]),
                    "repair_outcome": repair_outcome,
                    "conflicts_after": conflicts_after,
                    "after_fingerprint": state_fingerprint(after),
                    "after_repair_fingerprint": after_repair_fingerprint,
                    "post_structure": post_structure_metrics(after),
                    "native_step_seconds": float(metrics["native_step_seconds"]),
                    "pp_replan_seconds": float(metrics.get("pp_replan_seconds", 0.0)),
                }
            )

    payload = {
        "schema": STRIDE_COLLECTION_SCHEMA,
        "schema_version": 1,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": str(decision["state_id"]),
        "decision": decision,
        "before_fingerprint": initial_fingerprint,
        "before_repair_fingerprint": initial_repair_fingerprint,
        "before_conflicts": before_conflicts,
        "state_summary": state_summary,
        "candidate_generation": generation,
        "proposal": proposal,
        "feature_metrics": feature_metrics,
        "candidates": candidates,
        "trials": trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": str(decision["state_id"]),
        "state_file": str(output_path),
        "status": "ok",
        "candidate_count": len(candidates),
        "trial_count": len(trials),
    }


def collect_stride_repairs(
    *,
    selection_path: Path,
    output: Path,
    workers: int,
    resume: bool,
    max_states: int | None = None,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("STRIDE workers must be positive")
    selected = load_stride_selection(selection_path)
    if max_states is not None:
        if max_states <= 0:
            raise ValueError("STRIDE max_states must be positive")
        selected = selected[:max_states]
    project_root = Path(__file__).resolve().parents[1]
    producer = producer_identity(
        project_root=project_root,
        source_files=STRIDE_COLLECTION_PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    source_run_configs = {
        str(Path(str(row["source_root"])).resolve()): sha256_file(
            Path(str(row["source_root"])).resolve() / "run_config.json"
        )
        for row in selected
    }
    identity = {
        "schema": STRIDE_COLLECTION_SCHEMA,
        "schema_version": 1,
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "selected_state_ids": [str(row["state_id"]) for row in selected],
        "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
        "feature_dimension": FROZEN_FEATURE_DIMENSION,
        "proposal": FULL_POOL_PROPOSAL,
        "source_run_config_sha256": dict(sorted(source_run_configs.items())),
        "pp_trial_indices": list(PP_TRIAL_INDICES),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    output = output.resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if existing.get("run_fingerprint") != run_fingerprint:
            raise ValueError("STRIDE output belongs to another collection identity")
        if not resume:
            raise ValueError("STRIDE output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_jsonl(
        output / "state_selection.jsonl",
        [{**row, "schema": STRIDE_SELECTION_SCHEMA} for row in selected],
    )

    jobs = []
    for row in selected:
        key = _fingerprint(
            {"state_id": row["state_id"], "before": row["before_fingerprint"]}
        )[:20]
        jobs.append(
            {
                "decision": row,
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_collect_state, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "state_id": str(job["decision"]["state_id"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            _write_json(
                output / "collection_status.json",
                {
                    "schema": STRIDE_COLLECTION_SCHEMA,
                    "run_fingerprint": run_fingerprint,
                    "requested_state_count": len(jobs),
                    "completed_state_count": len(results),
                    "error_state_count": len(errors),
                    "errors": errors,
                    "status": "running",
                },
            )

    state_files = [Path(row["state_file"]) for row in results]
    all_trials: list[dict[str, Any]] = []
    candidate_counts: Counter[int] = Counter()
    for state_file in sorted(state_files):
        payload = _read_json(state_file)
        if not _state_artifact_valid(
            payload,
            run_fingerprint=run_fingerprint,
            state_id=str(payload.get("state_id", "")),
        ):
            errors.append(
                {"state_id": str(payload.get("state_id", "")), "error": "invalid state artifact"}
            )
            continue
        candidate_counts[len(payload["candidates"])] += 1
        all_trials.extend(payload["trials"])
    complete = len(results) == len(jobs) and not errors
    if complete:
        _write_jsonl(output / "repair_trials.jsonl", all_trials)
    report = {
        "schema": STRIDE_COLLECTION_SCHEMA,
        "schema_version": 1,
        "run_fingerprint": run_fingerprint,
        "selection_sha256": identity["selection_sha256"],
        "requested_state_count": len(jobs),
        "completed_state_count": len(results),
        "new_state_count": sum(row["status"] == "ok" for row in results),
        "resumed_state_count": sum(row["status"] == "resumed" for row in results),
        "error_state_count": len(errors),
        "errors": errors,
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "trial_count": len(all_trials),
        "expected_trial_count": sum(key * value * 4 for key, value in candidate_counts.items()),
        "complete": complete,
    }
    _write_json(output / "collection_report.json", report)
    _write_json(
        output / "collection_status.json",
        {**report, "status": "complete" if complete else "error"},
    )
    return report
