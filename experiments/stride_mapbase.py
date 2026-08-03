from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.feature_schema_v2 import (
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    PROFILE_FEATURE_NAMES,
)
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import summarize_initial_state_complexity
from experiments.stride_collection import (
    FULL_POOL_PROPOSAL,
    _paired_action,
    _validate_native_repair,
)
from experiments.stride_lns import (
    FROZEN_FEATURE_DIMENSION,
    FROZEN_FEATURE_SCHEMA_ID,
    STRIDE_TRIAL_SCHEMA,
    post_structure_metrics,
)
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import restore_repair_state, replay_prefix
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import generate_online_candidates
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


CONFIG_SCHEMA = "lns2.stride.mapbase_collection_config.v1"
COLLECTION_SCHEMA = "lns2.stride.mapbase_collection.v1"
STATE_SCHEMA = "lns2.stride.mapbase_state.v1"
AUDIT_SCHEMA = "lns2.stride.mapbase_collection_audit.v1"
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/stride_collection.py",
    "experiments/stride_lns.py",
    "experiments/stride_mapbase.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/online_selection.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _registered_path(project_root: Path, artifact: dict[str, Any]) -> Path:
    path = (project_root / str(artifact["path"])).resolve()
    if sha256_file(path) != str(artifact["sha256"]):
        raise ValueError(f"registered MapBase input SHA differs: {artifact['path']}")
    return path


def validate_mapbase_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected STRIDE-MapBase config")
    if (
        config.get("scientific_status")
        != "preregistered_map_diverse_base_candidate_repair_collection"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("topology_boundary_candidates_allowed"))
        or bool(config.get("controller_actions_allowed"))
        or bool(config.get("controller_outcomes_allowed"))
    ):
        raise ValueError("STRIDE-MapBase must remain base-candidate training data")
    if (
        config.get("collection_id") != "stride-mapbase-v1"
        or config.get("predecessor_id") != "stride-guardrank-relevance-v1"
        or int(config.get("expected_map_count", -1)) != 8
        or int(config.get("expected_task_count", -1)) != 32
        or int(config.get("expected_active_state_count", -1)) != 63
        or tuple(map(int, config.get("solver_seeds") or ())) != (1, 2)
        or tuple(map(int, config.get("trial_indices") or ())) != tuple(range(16))
        or config.get("feature_schema_id") != FEATURE_SCHEMA_ID
        or config.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
        or int(config.get("feature_dimension", -1)) != FROZEN_FEATURE_DIMENSION
    ):
        raise ValueError("STRIDE-MapBase identity or feature schema changed")
    if set(config.get("inputs") or {}) != {
        "relevance_report",
        "recommended_tasks",
        "relevance_states",
        "dataset_manifest",
        "qualification_manifest",
        "qualification_report",
        "runtime_config",
    }:
        raise ValueError("STRIDE-MapBase input registry changed")
    if dict(config.get("candidate_pool") or {}) != {
        "families": ["target", "collision", "random"],
        "requested_sizes": [4, 8, 16],
        "proposal_backend": "optimized",
        "minimum_candidates_per_state": 12,
        "maximum_candidates_per_state": 18,
        "topology_boundary": "forbidden",
    }:
        raise ValueError("STRIDE-MapBase candidate pool changed")
    if dict(config.get("selection") or {}) != {
        "research_split": "train",
        "maximum_tasks_per_map": 4,
        "exclude_initially_feasible_states": True,
        "selection_inputs": [
            "map_id",
            "layout_family",
            "agent_count",
            "initial_conflicts",
            "input_topology_relevance",
        ],
        "forbidden_selection_inputs": [
            "candidate_repair_outcome",
            "candidate_conflicts_after",
            "candidate_runtime",
            "controller_ttf",
        ],
    }:
        raise ValueError("STRIDE-MapBase selection contract changed")
    if (
        int(config.get("workers", -1)) != 8
        or float(config.get("state_timeout_seconds", -1.0)) != 3600.0
    ):
        raise ValueError("STRIDE-MapBase execution limits changed")


def build_mapbase_selection(
    config: dict[str, Any],
    *,
    recommended_tasks: list[dict[str, Any]],
    relevance_states: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    task_ids = {str(row["task_id"]) for row in recommended_tasks}
    if len(task_ids) != int(config["expected_task_count"]):
        raise ValueError("STRIDE-MapBase recommended task count changed")
    dataset_by_task = {str(row["task_id"]): row for row in dataset_rows}
    relevance_by_state = {
        str(row["state_id"]): row for row in relevance_states
    }
    selected = []
    for qualification in qualification_rows:
        task_id = str(qualification["task_id"])
        if task_id not in task_ids:
            continue
        solver_seed = int(qualification["solver_seed"])
        state_id = f"{task_id}::solver_seed_{solver_seed}"
        relevance = relevance_by_state.get(state_id)
        dataset = dataset_by_task.get(task_id)
        if relevance is None or dataset is None or qualification.get("status") != "ok":
            raise ValueError(f"STRIDE-MapBase state input is incomplete: {state_id}")
        before_conflicts = int(qualification["initial_conflicts"])
        if before_conflicts <= 0:
            continue
        selected.append(
            {
                "state_id": state_id,
                "task_id": task_id,
                "solver_seed": solver_seed,
                "map_id": str(relevance["map_id"]),
                "layout_family": str(relevance["layout_family"]),
                "agent_count": int(dataset["agent_count"]),
                "before_conflicts": before_conflicts,
                "before_fingerprint": str(qualification["state_fingerprint"]),
                "input_topology_relevance": bool(relevance["relevant"]),
                "research_split": str(config["selection"]["research_split"]),
                "source_split": str(dataset["split"]),
            }
        )
    selected.sort(key=lambda row: str(row["state_id"]))
    if len(selected) != int(config["expected_active_state_count"]):
        raise ValueError("STRIDE-MapBase active state count changed")
    if len({str(row["map_id"]) for row in selected}) != int(
        config["expected_map_count"]
    ):
        raise ValueError("STRIDE-MapBase map coverage changed")
    return selected


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
    candidate_ids = [str(row.get("candidate_id")) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        return False
    if any(
        row.get("candidate_kind") != "base"
        or any(
            str(family).startswith("topology-")
            for family in row.get("selection_families") or ()
        )
        for row in candidates
    ):
        return False
    expected = {
        (candidate_id, trial_index)
        for candidate_id in candidate_ids
        for trial_index in trial_indices
    }
    required_names = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    observed = set()
    seeds_by_trial: dict[int, set[int]] = {
        trial_index: set() for trial_index in trial_indices
    }
    for row in trials:
        if not isinstance(row, dict):
            return False
        trial_index = int(row.get("trial_index", -1))
        candidate_id = str(row.get("candidate_id"))
        if (
            row.get("schema") != STRIDE_TRIAL_SCHEMA
            or row.get("feature_schema_id") != FROZEN_FEATURE_SCHEMA_ID
            or row.get("candidate_kind") != "base"
            or (candidate_id, trial_index) not in expected
            or set(row.get("features") or {}) != required_names
            or type(row.get("pp_seed")) is not int
        ):
            return False
        observed.add((candidate_id, trial_index))
        seeds_by_trial[trial_index].add(int(row["pp_seed"]))
    return observed == expected and all(
        len(seeds) == 1 for seeds in seeds_by_trial.values()
    )


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    selection = dict(job["selection"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    trial_indices = tuple(map(int, job["trial_indices"]))
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _artifact_valid(
            existing,
            run_fingerprint=run_fingerprint,
            state_id=str(selection["state_id"]),
            trial_indices=trial_indices,
        ):
            return {
                "state_id": str(selection["state_id"]),
                "state_file": str(output_path),
                "status": "resumed",
                "candidate_count": len(existing["candidates"]),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"completed MapBase artifact is invalid: {output_path}")
    replay = {
        "dataset_root": str(job["dataset_root"]),
        "row": dict(job["dataset_row"]),
        "environment": dict(job["environment"]),
        "solver_seed": int(selection["solver_seed"]),
        "replay_destroy_strategy": "Adaptive",
    }
    environment, state = replay_prefix(replay, [])
    initial_fingerprint = state_fingerprint(state)
    if (
        initial_fingerprint != str(selection["before_fingerprint"])
        or int(state["num_of_colliding_pairs"])
        != int(selection["before_conflicts"])
    ):
        raise RuntimeError(f"MapBase initial state changed: {selection['state_id']}")
    initial_repair_fingerprint = repair_structure_fingerprint(state)
    restore_seed = repairability_restore_seed(initial_repair_fingerprint)
    _, restored_state = restore_repair_state(replay, state, seed=restore_seed)
    if repair_structure_fingerprint(restored_state) != initial_repair_fingerprint:
        raise RuntimeError("MapBase restore contract changed")
    proposal = {**dict(job["proposal"]), **FULL_POOL_PROPOSAL}
    proposal.pop("topology_boundary", None)
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(selection["task_id"]),
        solver_seed=int(selection["solver_seed"]),
        decision_index=0,
        proposal_config=proposal,
        state_hash=initial_fingerprint,
        verify_full_state=False,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    minimum = int(job["minimum_candidates"])
    maximum = int(job["maximum_candidates"])
    if not minimum <= len(candidates) <= maximum:
        raise RuntimeError("MapBase base candidate count is outside registration")
    if any(
        str(family).startswith("topology-")
        for candidate in candidates
        for family in candidate.get("selection_families") or ()
    ):
        raise RuntimeError("MapBase generated a forbidden topology candidate")
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
    if any(
        len(values) != FROZEN_FEATURE_DIMENSION
        for values in features_by_candidate.values()
    ):
        raise RuntimeError("MapBase realized features are incomplete")
    candidate_records = []
    trials = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        agents = sorted(map(int, candidate["agents"]))
        families = sorted(map(str, candidate.get("selection_families") or ()))
        candidate_records.append(
            {
                **candidate,
                "candidate_kind": "base",
                "actual_size": len(agents),
                "agents": agents,
                "selection_families": families,
            }
        )
        for trial_index in trial_indices:
            branch_environment, branch_state = restore_repair_state(
                replay, state, seed=restore_seed
            )
            if (
                repair_structure_fingerprint(branch_state)
                != initial_repair_fingerprint
            ):
                raise RuntimeError("MapBase paired branch restore changed")
            seed = repairability_pp_seed(initial_repair_fingerprint, trial_index)
            result = _plain(branch_environment.step(_paired_action(agents, seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair_fingerprint = repair_structure_fingerprint(after)
            outcome = classify_repair_outcome(
                before_fingerprint=initial_repair_fingerprint,
                after_fingerprint=after_repair_fingerprint,
                replan_success=bool(metrics["replan_success"]),
                conflicts_before=int(selection["before_conflicts"]),
                conflicts_after=conflicts_after,
                feasible=bool(after.get("feasible")),
            )
            trials.append(
                {
                    "schema": STRIDE_TRIAL_SCHEMA,
                    "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
                    "state_id": str(selection["state_id"]),
                    "candidate_id": candidate_id,
                    "candidate_kind": "base",
                    "actual_size": len(agents),
                    "selection_families": families,
                    "agents": agents,
                    "layout_family": str(selection["layout_family"]),
                    "map_id": str(selection["map_id"]),
                    "task_id": str(selection["task_id"]),
                    "split": "train",
                    "source_split": str(selection["source_split"]),
                    "source_policy": "initial_state_map_expansion",
                    "decision_stage": "initial",
                    "solver_seed": int(selection["solver_seed"]),
                    "agent_count": int(selection["agent_count"]),
                    "before_conflicts": int(selection["before_conflicts"]),
                    "before_fingerprint": initial_fingerprint,
                    "before_repair_fingerprint": initial_repair_fingerprint,
                    "input_topology_relevance": bool(
                        selection["input_topology_relevance"]
                    ),
                    "features": features_by_candidate[candidate_id],
                    "trial_index": trial_index,
                    "pp_seed": seed,
                    "feasible": bool(after.get("feasible")),
                    "replan_success": bool(metrics["replan_success"]),
                    "repair_outcome": outcome,
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
        "state_id": str(selection["state_id"]),
        "selection": selection,
        "before_fingerprint": initial_fingerprint,
        "before_repair_fingerprint": initial_repair_fingerprint,
        "before_conflicts": int(selection["before_conflicts"]),
        "state_restore": {
            "restore_seed": restore_seed,
            "repair_structure_fingerprint": initial_repair_fingerprint,
        },
        "state_summary": summarize_initial_state_complexity(state),
        "candidate_generation": generation,
        "feature_metrics": feature_metrics,
        "candidates": candidate_records,
        "trials": trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": str(selection["state_id"]),
        "state_file": str(output_path),
        "status": "ok",
        "candidate_count": len(candidate_records),
        "trial_count": len(trials),
    }


def _load_inputs(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Path], list[dict[str, Any]], dict[str, Any]]:
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_mapbase_config(config)
    inputs = {
        name: _registered_path(project_root, dict(artifact))
        for name, artifact in config["inputs"].items()
    }
    relevance_report = _read_json(inputs["relevance_report"])
    if (
        relevance_report.get("passed") is not True
        or relevance_report.get("next_decision")
        != "keep_boundary_optional_and_prioritize_map_diverse_base_candidates"
    ):
        raise ValueError("STRIDE-MapBase requires the registered relevance decision")
    qualification_report = _read_json(inputs["qualification_report"])
    if qualification_report.get("errors") or qualification_report.get("valid_count") != 96:
        raise ValueError("STRIDE-MapBase qualification input is invalid")
    dataset_rows = _read_jsonl(inputs["dataset_manifest"])
    selection = build_mapbase_selection(
        config,
        recommended_tasks=_read_jsonl(inputs["recommended_tasks"]),
        relevance_states=_read_jsonl(inputs["relevance_states"]),
        dataset_rows=dataset_rows,
        qualification_rows=_read_jsonl(inputs["qualification_manifest"]),
    )
    return config, inputs, selection, {
        "dataset_rows": dataset_rows,
        "runtime": _read_json(inputs["runtime_config"]),
    }


def collect_mapbase_trials(
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, inputs, selection, loaded = _load_inputs(config_path)
    worker_count = int(config["workers"] if workers is None else workers)
    if worker_count <= 0:
        raise ValueError("STRIDE-MapBase workers must be positive")
    project_root = config_path.parents[1]
    producer = producer_identity(
        project_root=project_root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    identity = {
        "schema": COLLECTION_SCHEMA,
        "collection_id": str(config["collection_id"]),
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "selected_state_ids": [str(row["state_id"]) for row in selection],
        "feature_schema_id": FROZEN_FEATURE_SCHEMA_ID,
        "feature_dimension": FROZEN_FEATURE_DIMENSION,
        "trial_indices": list(map(int, config["trial_indices"])),
        "candidate_pool": dict(config["candidate_pool"]),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    output_root = Path(output).resolve()
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("STRIDE-MapBase output belongs to another run")
        if not resume:
            raise ValueError("STRIDE-MapBase output exists; pass --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_jsonl(output_root / "state_selection.jsonl", selection)
    dataset_by_task = {
        str(row["task_id"]): row for row in loaded["dataset_rows"]
    }
    environment = dict(loaded["runtime"]["environment"])
    environment["max_repair_iterations"] = max(
        1, int(environment.get("max_repair_iterations", 0))
    )
    jobs = []
    for row in selection:
        key = _fingerprint(
            {"state_id": row["state_id"], "before": row["before_fingerprint"]}
        )[:20]
        jobs.append(
            {
                "job_id": str(row["state_id"]),
                "state_id": str(row["state_id"]),
                "selection": row,
                "dataset_root": str(
                    (project_root / str(config["dataset_root"])).resolve()
                ),
                "dataset_row": dataset_by_task[str(row["task_id"])],
                "environment": environment,
                "proposal": dict(loaded["runtime"]["proposal"]),
                "minimum_candidates": int(
                    config["candidate_pool"]["minimum_candidates_per_state"]
                ),
                "maximum_candidates": int(
                    config["candidate_pool"]["maximum_candidates_per_state"]
                ),
                "trial_indices": list(map(int, config["trial_indices"])),
                "output_path": str(output_root / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
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
            output_root / "collection_status.json",
            {
                "schema": COLLECTION_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "error_state_count": len(failures),
                "status": "running",
                "errors": failures,
            },
        )

    _write_json(
        output_root / "collection_status.json",
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
    observed = _run_jobs(
        _collect_state,
        jobs,
        worker_count,
        phase="stride-mapbase",
        output_root=output_root,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["state_timeout_seconds"]),
        on_result=update_status,
    )
    results = [row for row in observed if row.get("status") in {"ok", "resumed"}]
    errors = [
        {
            "state_id": str(row.get("state_id", row.get("job_id"))),
            "error": str(row.get("error")),
        }
        for row in observed
        if row.get("status") in {"error", "timeout"}
    ]
    all_trials = []
    candidate_counts: Counter[int] = Counter()
    trial_indices = tuple(map(int, config["trial_indices"]))
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
        all_trials.extend(payload["trials"])
    complete = len(results) == len(jobs) and not errors
    if complete:
        _write_jsonl(output_root / "repair_trials.jsonl", all_trials)
    report = {
        "schema": COLLECTION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(jobs),
        "completed_state_count": len(results),
        "new_state_count": sum(row["status"] == "ok" for row in results),
        "resumed_state_count": sum(row["status"] == "resumed" for row in results),
        "error_state_count": len(errors),
        "errors": errors,
        "candidate_count_distribution": {
            str(key): value for key, value in sorted(candidate_counts.items())
        },
        "boundary_candidate_count": 0,
        "trial_count": len(all_trials),
        "expected_trial_count": sum(
            key * value * len(trial_indices)
            for key, value in candidate_counts.items()
        ),
        "complete": complete,
    }
    _write_json(output_root / "collection_report.json", report)
    _write_json(
        output_root / "collection_status.json",
        {**report, "status": "complete" if complete else "error"},
    )
    return report


def audit_mapbase_collection(
    config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, _inputs, selection, _loaded = _load_inputs(config_path)
    collection_root = Path(collection).resolve()
    run = _read_json(collection_root / "run_config.json")
    collection_report = _read_json(collection_root / "collection_report.json")
    state_files = sorted((collection_root / "states").glob("*.json"))
    consolidated_path = collection_root / "repair_trials.jsonl"
    consolidated = _read_jsonl(consolidated_path) if consolidated_path.is_file() else []
    state_rows = []
    all_trials = []
    errors = []
    trial_indices = tuple(map(int, config["trial_indices"]))
    for state_file in state_files:
        payload = _read_json(state_file)
        state_id = str(payload.get("state_id", ""))
        if not _artifact_valid(
            payload,
            run_fingerprint=str(run["run_fingerprint"]),
            state_id=state_id,
            trial_indices=trial_indices,
        ):
            errors.append(f"invalid state artifact: {state_file.name}")
            continue
        expected_seeds = {
            trial_index: repairability_pp_seed(
                str(payload["before_repair_fingerprint"]), trial_index
            )
            for trial_index in trial_indices
        }
        if any(
            int(row["pp_seed"]) != expected_seeds[int(row["trial_index"])]
            for row in payload["trials"]
        ):
            errors.append(f"paired PP seed mismatch: {state_id}")
        state_rows.append(payload)
        all_trials.extend(payload["trials"])
    expected_ids = {str(row["state_id"]) for row in selection}
    observed_ids = {str(row["state_id"]) for row in state_rows}
    gates = {
        "collection_complete": collection_report.get("complete") is True,
        "zero_collection_errors": int(collection_report.get("error_state_count", -1)) == 0,
        "complete_state_coverage": observed_ids == expected_ids,
        "expected_state_artifact_count": len(state_files)
        == int(config["expected_active_state_count"]),
        "base_candidates_only": all(
            candidate.get("candidate_kind") == "base"
            for payload in state_rows
            for candidate in payload["candidates"]
        ),
        "no_topology_families": all(
            not str(family).startswith("topology-")
            for payload in state_rows
            for candidate in payload["candidates"]
            for family in candidate.get("selection_families") or ()
        ),
        "candidate_caps": all(
            int(config["candidate_pool"]["minimum_candidates_per_state"])
            <= len(payload["candidates"])
            <= int(config["candidate_pool"]["maximum_candidates_per_state"])
            for payload in state_rows
        ),
        "consolidated_trials_match": _fingerprint(all_trials)
        == _fingerprint(consolidated),
        "trial_count_matches": len(consolidated)
        == int(collection_report.get("trial_count", -1)),
        "feature_schema_exact": all(
            row.get("feature_schema_id") == FROZEN_FEATURE_SCHEMA_ID
            and set(row.get("features") or {})
            == set(PROFILE_FEATURE_NAMES["realized_dynamic"])
            for row in consolidated
        ),
        "all_train_split": all(row.get("split") == "train" for row in consolidated),
    }
    report = {
        "schema": AUDIT_SCHEMA,
        "collection_id": str(config["collection_id"]),
        "state_count": len(state_rows),
        "trial_count": len(consolidated),
        "errors": errors,
        "gates": gates,
        "passed": not errors and all(gates.values()),
        "sha256": {
            "config": sha256_file(config_path),
            "run_config": sha256_file(collection_root / "run_config.json"),
            "state_selection": sha256_file(collection_root / "state_selection.jsonl"),
            "collection_report": sha256_file(collection_root / "collection_report.json"),
            "repair_trials": (
                sha256_file(consolidated_path) if consolidated_path.is_file() else None
            ),
        },
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "mapbase_audit_report.json", report)
    return report


__all__ = [
    "audit_mapbase_collection",
    "build_mapbase_selection",
    "collect_mapbase_trials",
    "validate_mapbase_config",
]
