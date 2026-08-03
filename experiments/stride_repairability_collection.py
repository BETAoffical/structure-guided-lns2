from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import summarize_initial_state_complexity
from experiments.stride_collection import (
    FULL_POOL_PROPOSAL,
    _paired_action,
    _replay_job,
    _validate_native_repair,
    load_stride_selection,
)
from experiments.stride_lns import (
    FROZEN_FEATURE_DIMENSION,
    FROZEN_FEATURE_SCHEMA_ID,
    STRIDE_TRIAL_SCHEMA,
    post_structure_metrics,
)
from experiments.stride_repairability import validate_repairability_label_config
from experiments.trace_replay import replay_prefix
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import generate_online_candidates
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


COLLECTION_SCHEMA = "lns2.stride.repairability_collection.v1"
STATE_SCHEMA = "lns2.stride.repairability_state.v1"
BOUNDARY_AUGMENTATION = {
    "enabled": True,
    "generator_id": "stride-topoboundary-v1",
    "neighborhood_size": 16,
    "core_budget": 4,
    "maximum_added_candidates": 2,
    "runtime_id": "stride-boundary-static-cache-v1",
    "static_grid_cache": True,
}
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_lns.py",
    "experiments/stride_repairability.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/online_selection.py",
    "lns2_selector/runtime/topology_candidates.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def repairability_pp_seed(state_repair_fingerprint: str, trial_index: int) -> int:
    if trial_index not in range(16):
        raise ValueError("repairability PP trial index must be in 0..15")
    return int(
        _fingerprint(
            {
                "namespace": "stride-repairability-paired-pp-v1",
                "repair_state": str(state_repair_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def _candidate_signature(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(candidate["candidate_id"]),
        tuple(sorted(map(int, candidate["agents"]))),
    )


def _artifact_valid(
    payload: dict[str, Any],
    *,
    run_fingerprint: str,
    state_id: str,
    trial_indices: tuple[int, ...],
) -> bool:
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("state_id") != state_id
        or payload.get("complete") is not True
    ):
        return False
    candidates = payload.get("candidates")
    trials = payload.get("trials")
    if not isinstance(candidates, list) or not candidates or not isinstance(trials, list):
        return False
    if any(
        not isinstance(candidate, dict) or candidate.get("candidate_id") is None
        for candidate in candidates
    ):
        return False
    candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        return False
    required_names = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    allowed_indices = set(trial_indices)
    for row in trials:
        if not isinstance(row, dict):
            return False
        try:
            trial_index = int(row.get("trial_index", -1))
        except (TypeError, ValueError):
            return False
        if (
            row.get("schema") != STRIDE_TRIAL_SCHEMA
            or row.get("feature_schema_id") != FROZEN_FEATURE_SCHEMA_ID
            or str(row.get("candidate_id")) not in candidate_ids
            or trial_index not in allowed_indices
            or type(row.get("pp_seed")) is not int
            or not isinstance(row.get("features"), dict)
            or set(row["features"]) != required_names
        ):
            return False
    expected = {
        (candidate_id, trial_index)
        for candidate_id in candidate_ids
        for trial_index in trial_indices
    }
    observed = {
        (str(row.get("candidate_id")), int(row.get("trial_index", -1)))
        for row in trials
        if isinstance(row, dict)
    }
    if observed != expected or len(trials) != len(expected):
        return False
    seeds_by_trial = {
        trial_index: {
            int(row["pp_seed"])
            for row in trials
            if int(row["trial_index"]) == trial_index
        }
        for trial_index in trial_indices
    }
    return (
        all(len(seeds) == 1 for seeds in seeds_by_trial.values())
        and len({next(iter(seeds)) for seeds in seeds_by_trial.values()})
        == len(trial_indices)
    )


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    trial_indices = tuple(map(int, job["trial_indices"]))
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _artifact_valid(
            existing,
            run_fingerprint=run_fingerprint,
            state_id=str(decision["state_id"]),
            trial_indices=trial_indices,
        ):
            return {
                "state_id": str(decision["state_id"]),
                "state_file": str(output_path),
                "status": "resumed",
                "state_count": 1,
                "outcome_count": len(existing["trials"]),
                "error_count": 0,
                "candidate_count": len(existing["candidates"]),
                "boundary_candidate_count": int(existing["boundary_candidate_count"]),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"completed repairability artifact is invalid: {output_path}")

    replay = _replay_job(decision)
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    initial_fingerprint = state_fingerprint(state)
    if initial_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError(
            f"repairability replay mismatch for {decision['state_id']}"
        )
    initial_repair_fingerprint = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    if before_conflicts <= 0 or bool(state.get("done")):
        raise ValueError("repairability state must be active and conflicting")

    base_proposal = {**dict(replay["proposal"]), **FULL_POOL_PROPOSAL}
    base_proposal.pop("topology_boundary", None)
    base_candidates, base_generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=base_proposal,
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
    augmented_proposal = {
        **base_proposal,
        "topology_boundary": dict(BOUNDARY_AUGMENTATION),
    }
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=augmented_proposal,
        state_hash=initial_fingerprint,
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
        topology_static_grid=feature_engine.static_grid,
        topology_no_progress_streak=0,
        topology_remaining_wall_seconds=None,
    )
    if state_fingerprint(state) != initial_fingerprint:
        raise RuntimeError("repairability candidate generation changed the state")
    base_by_id = {
        str(candidate["candidate_id"]): _candidate_signature(candidate)
        for candidate in base_candidates
    }
    augmented_by_id = {
        str(candidate["candidate_id"]): _candidate_signature(candidate)
        for candidate in candidates
    }
    if not set(base_by_id) <= set(augmented_by_id) or any(
        base_by_id[candidate_id] != augmented_by_id[candidate_id]
        for candidate_id in base_by_id
    ):
        raise RuntimeError("repairability augmentation changed the frozen base pool")
    boundary_ids = set(augmented_by_id) - set(base_by_id)
    if len(boundary_ids) > 2 or len(candidates) > 20:
        raise RuntimeError("repairability augmented pool exceeds its cap")

    feature_rows, feature_metrics = feature_engine.realized_rows(
        candidates, state_hash=initial_fingerprint
    )
    features_by_candidate = {
        str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
        for row in feature_rows
    }
    if set(features_by_candidate) != set(augmented_by_id) or any(
        len(values) != FROZEN_FEATURE_DIMENSION
        for values in features_by_candidate.values()
    ):
        raise RuntimeError("repairability candidate features are incomplete")

    trials: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        agents = sorted(map(int, candidate["agents"]))
        candidate_kind = "boundary_only" if candidate_id in boundary_ids else "base"
        candidate_record = {
            **candidate,
            "candidate_kind": candidate_kind,
            "actual_size": len(agents),
            "agents": agents,
            "selection_families": sorted(
                map(str, candidate.get("selection_families") or ())
            ),
        }
        candidate_records.append(candidate_record)
        for trial_index in trial_indices:
            branch_environment, branch_state = replay_prefix(
                replay, decision["prefix_actions"]
            )
            if state_fingerprint(branch_state) != initial_fingerprint:
                raise RuntimeError("repairability paired branch replay changed")
            seed = repairability_pp_seed(initial_repair_fingerprint, trial_index)
            result = _plain(branch_environment.step(_paired_action(agents, seed)))
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
                    "candidate_kind": candidate_kind,
                    "actual_size": len(agents),
                    "selection_families": candidate_record["selection_families"],
                    "agents": agents,
                    "layout_family": str(decision.get("layout_mode", "unknown")),
                    "map_id": str(decision["map_id"]),
                    "task_id": str(decision["task_id"]),
                    "split": str(decision["research_split"]),
                    "source_split": str(decision["split"]),
                    "source_policy": str(decision["source_policy"]),
                    "decision_stage": str(decision["decision_stage"]),
                    "solver_seed": int(decision["solver_seed"]),
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
                    "pp_replan_seconds": float(
                        metrics.get("pp_replan_seconds", 0.0)
                    ),
                }
            )

    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": str(decision["state_id"]),
        "decision": decision,
        "before_fingerprint": initial_fingerprint,
        "before_repair_fingerprint": initial_repair_fingerprint,
        "before_conflicts": before_conflicts,
        "state_summary": summarize_initial_state_complexity(state),
        "base_candidate_generation": base_generation,
        "augmented_candidate_generation": generation,
        "base_candidate_count": len(base_candidates),
        "boundary_candidate_count": len(boundary_ids),
        "feature_metrics": feature_metrics,
        "candidates": candidate_records,
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
        "state_count": 1,
        "outcome_count": len(trials),
        "error_count": 0,
        "candidate_count": len(candidates),
        "boundary_candidate_count": len(boundary_ids),
        "trial_count": len(trials),
    }


def collect_repairability_trials(
    *,
    config_path: str | Path,
    selection_path: str | Path,
    output: str | Path,
    workers: int = 4,
    resume: bool = False,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("repairability workers must be positive")
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_repairability_label_config(config)
    if dict(config["candidate_pool"]) != {
        "base_families": ["target", "collision", "random"],
        "base_sizes": [4, 8, 16],
        "augmentation": "stride-topoboundary-v1",
        "boundary_size": 16,
        "boundary_core_budget": 4,
        "maximum_boundary_candidates": 2,
        "maximum_total_candidates": 20,
    }:
        raise ValueError("repairability collection candidate pool changed")
    selection_path = Path(selection_path).resolve()
    selected = load_stride_selection(selection_path)
    map_splits: dict[str, set[str]] = {}
    for row in selected:
        research_split = str(row.get("research_split", ""))
        if research_split not in {"train", "validation"}:
            raise ValueError(
                "repairability selection requires an explicit research_split"
            )
        map_splits.setdefault(str(row["map_id"]), set()).add(research_split)
    leaking_maps = sorted(
        map_id for map_id, splits in map_splits.items() if len(splits) != 1
    )
    if leaking_maps:
        raise ValueError(f"repairability selection leaks maps: {leaking_maps}")
    trial_indices = tuple(map(int, config["trial_indices"]))
    project_root = Path(__file__).resolve().parents[1]
    producer = producer_identity(
        project_root=project_root,
        source_files=PRODUCER_FILES,
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
        "schema": COLLECTION_SCHEMA,
        "selection_path": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "selected_state_ids": [str(row["state_id"]) for row in selected],
        "label_config_sha256": sha256_file(config_path),
        "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
        "feature_dimension": FROZEN_FEATURE_DIMENSION,
        "base_proposal": FULL_POOL_PROPOSAL,
        "boundary_augmentation": BOUNDARY_AUGMENTATION,
        "pp_trial_indices": list(trial_indices),
        "source_run_config_sha256": dict(sorted(source_run_configs.items())),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    output = Path(output).resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("repairability output belongs to another collection")
        if not resume:
            raise ValueError("repairability output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_jsonl(output / "state_selection.jsonl", selected)
    jobs = []
    for row in selected:
        key = _fingerprint(
            {"state_id": row["state_id"], "before": row["before_fingerprint"]}
        )[:20]
        jobs.append(
            {
                "job_id": str(row["state_id"]),
                "row": row,
                "state_id": str(row["state_id"]),
                "solver_seed": int(row["solver_seed"]),
                "decision": row,
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "trial_indices": list(trial_indices),
                "resume": bool(resume),
            }
        )
    observed: list[dict[str, Any]] = []

    def update_status(result: dict[str, Any]) -> None:
        observed.append(result)
        failures = [
            row for row in observed if row.get("status") in {"error", "timeout"}
        ]
        _write_json(
            output / "collection_status.json",
            {
                "schema": COLLECTION_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "error_state_count": len(failures),
                "status": "running",
                "errors": [
                    {
                        "state_id": str(row.get("state_id", row.get("job_id"))),
                        "error": str(row.get("error")),
                    }
                    for row in failures
                ],
            },
        )

    _write_json(
        output / "collection_status.json",
        {
            "schema": COLLECTION_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "requested_state_count": len(jobs),
            "completed_state_count": 0,
            "error_state_count": 0,
            "status": "running",
            "errors": [],
        },
    )
    try:
        observed = _run_jobs(
            _collect_state,
            jobs,
            workers,
            phase="stride-repairability",
            output_root=output,
            run_fingerprint=run_fingerprint,
            on_result=update_status,
        )
    except BaseException as error:
        current = _read_json(output / "collection_status.json")
        _write_json(
            output / "collection_status.json",
            {
                **current,
                "status": (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
                ),
                "runner_error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    results = [row for row in observed if row.get("status") in {"ok", "resumed"}]
    errors = [
        {
            "state_id": str(row.get("state_id", row.get("job_id"))),
            "error": str(row.get("error")),
        }
        for row in observed
        if row.get("status") in {"error", "timeout"}
    ]
    all_trials: list[dict[str, Any]] = []
    candidate_counts: Counter[int] = Counter()
    boundary_counts: Counter[int] = Counter()
    for result in sorted(results, key=lambda row: str(row["state_id"])):
        payload = _read_json(Path(str(result["state_file"])))
        if not _artifact_valid(
            payload,
            run_fingerprint=run_fingerprint,
            state_id=str(result["state_id"]),
            trial_indices=trial_indices,
        ):
            errors.append(
                {"state_id": str(result["state_id"]), "error": "invalid state artifact"}
            )
            continue
        candidate_counts[len(payload["candidates"])] += 1
        boundary_counts[int(payload["boundary_candidate_count"])] += 1
        all_trials.extend(payload["trials"])
    complete = len(results) == len(jobs) and not errors
    if complete:
        _write_jsonl(output / "repair_trials.jsonl", all_trials)
    report = {
        "schema": COLLECTION_SCHEMA,
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
        "boundary_candidate_count_distribution": {
            str(key): value for key, value in sorted(boundary_counts.items())
        },
        "trial_count": len(all_trials),
        "expected_trial_count": sum(
            key * value * len(trial_indices)
            for key, value in candidate_counts.items()
        ),
        "complete": complete,
    }
    _write_json(output / "collection_report.json", report)
    _write_json(
        output / "collection_status.json",
        {**report, "status": "complete" if complete else "error"},
    )
    return report


__all__ = ["collect_repairability_trials", "repairability_pp_seed"]
