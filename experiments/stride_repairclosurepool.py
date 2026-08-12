from __future__ import annotations

import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
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
from experiments.state_analysis import analyze_state, analyze_static_grid
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_marginalpool_action_replay import (
    _replay_job,
    aggregate_candidate,
    build_frozen_cohort,
    stable_dominates,
)
from experiments.stride_marginalpool_root_diagnostic import _feature_payload, _structural
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_robustaction_label_collection import _forbidden_hits
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT, restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome
from lns2_selector.runtime.repairclosurepool import (
    REPAIRCLOSUREPOOL_ID,
    generate_repairclosure_candidates,
)


DESIGN_SCHEMA = "lns2.stride.repairclosurepool_design.v1"
EXECUTION_SCHEMA = "lns2.stride.repairclosurepool_execution.v1"
COHORT_SCHEMA = "lns2.stride.repairclosurepool_cohort.v1"
TRIAL_SCHEMA = "lns2.stride.repairclosurepool_trial.v1"
STATUS_SCHEMA = "lns2.stride.repairclosurepool_collection_status.v1"
REPORT_SCHEMA = "lns2.stride.repairclosurepool_report.v1"
EXPERIMENT_ID = "stride-repairclosurepool-v1"
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/closed_loop_trace_storage.py",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_marginalpool_action_replay.py",
    "experiments/stride_marginalpool_root_diagnostic.py",
    "experiments/stride_repairability_collection.py",
    "experiments/stride_repairclosurepool.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/repair_outcomes.py",
    "lns2_selector/runtime/repairclosurepool.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _load_design(path: str | Path) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != DESIGN_SCHEMA
        or config.get("scientific_status")
        != "preregistered_outcome_blind_candidate_generator_and_pool_opportunity_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("RepairClosurePool design identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "marginalpool_registration",
        "root_checkpoints",
        "base_candidate_aggregates",
        "transactionalrepair_discovery",
    }:
        raise ValueError("RepairClosurePool input registry changed")
    generator = dict(config.get("generator") or {})
    if (
        generator.get("id") != REPAIRCLOSUREPOOL_ID
        or int(generator.get("temporal_window", -1)) != 2
        or int(generator.get("maximum_candidates", -1)) != 12
        or int(generator.get("maximum_neighborhood_size", -1)) != 64
        or not math.isclose(
            float(generator.get("maximum_candidate_jaccard", math.nan)),
            0.9,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or list(generator.get("fixed_preferred_sizes") or ()) != []
        or generator.get("must_touch_current_conflict") is not True
        or generator.get("future_outcomes_used") is not False
        or generator.get("pp_failure_diagnostics_used_online") is not False
        or generator.get("repair_order_controlled") is not False
    ):
        raise ValueError("RepairClosurePool generator contract changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "candidate_pool_opportunity_audit_only": True,
        "current_ranker_used": False,
        "model_training_allowed": False,
        "repair_order_intervention_allowed": False,
        "runtime_integration_allowed": False,
        "ttf_experiment_allowed": False,
        "long_tail_avoidance_claim_allowed": False,
        "default_controller_replacement_allowed": False,
    }:
        raise ValueError("RepairClosurePool claim boundary changed")
    return path, root, config, inputs


def _anchors_by_state(checkpoints: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for checkpoint in checkpoints:
        base = [
            dict(candidate)
            for candidate in checkpoint["candidate_pool"]
            if not _structural(candidate)
        ]
        if not base or any(candidate.get("score") is None for candidate in base):
            raise ValueError("RepairClosurePool checkpoint has no scored V2 base pool")
        anchor = max(
            base,
            key=lambda candidate: (
                float(candidate["score"]),
                str(candidate["candidate_id"]),
            ),
        )
        state_key = str(checkpoint["state_fingerprint"])
        result[state_key][str(anchor["candidate_id"])] = {
            "candidate_id": str(anchor["candidate_id"]),
            "agents": sorted(map(int, anchor["agents"])),
        }
    return {
        state_key: [rows[key] for key in sorted(rows)]
        for state_key, rows in sorted(result.items())
    }


def _materialize_state(job: dict[str, Any]) -> dict[str, Any]:
    source = dict(job["source"])
    state_key = str(source["state_fingerprint"])
    state_blob = Path(str(source["state_blob"]))
    if sha256_file(state_blob) != str(source["state_blob_sha256"]):
        raise ValueError("RepairClosurePool source state blob changed")
    state = read_state_blob(state_blob)
    state["context"] = dict(source["state_context"])
    if state_fingerprint(state) != state_key:
        raise ValueError("RepairClosurePool source state fingerprint changed")
    analysis = analyze_state(state, static_grid=analyze_static_grid(state))
    base_candidates = [
        dict(candidate)
        for candidate in source["candidates"]
        if str(candidate["candidate_kind"]) == "base"
    ]
    if not base_candidates:
        raise ValueError("RepairClosurePool state has no frozen V2 base candidates")
    registered_base_ids = set(map(str, job["registered_base_candidate_ids"]))
    missing_base = [
        str(candidate["candidate_id"])
        for candidate in base_candidates
        if str(candidate["candidate_id"]) not in registered_base_ids
    ]
    if missing_base:
        raise ValueError(f"RepairClosurePool base labels are missing: {missing_base}")
    state_anchors = list(job["v2_anchors"])
    if not state_anchors:
        raise ValueError("RepairClosurePool state has no frozen V2 anchor")
    generator = dict(job["generator"])
    generated = generate_repairclosure_candidates(
        state,
        analysis,
        v2_anchors=state_anchors,
        maximum_candidates=int(generator["maximum_candidates"]),
        maximum_neighborhood_size=int(generator["maximum_neighborhood_size"]),
        temporal_window=int(generator["temporal_window"]),
        maximum_jaccard_similarity=float(generator["maximum_candidate_jaccard"]),
    )
    known_ids = {str(candidate["candidate_id"]) for candidate in source["candidates"]}
    novel = []
    active = {agent for edge in analysis.pair_set for agent in edge}
    for candidate in generated.candidates:
        if str(candidate["candidate_id"]) in known_ids:
            continue
        if not set(map(int, candidate["agents"])) & active:
            raise RuntimeError("RepairClosurePool produced a no-conflict action")
        novel.append({**candidate, **_feature_payload(state, candidate, analysis)})
    if not novel:
        raise ValueError("RepairClosurePool state has no novel candidate")
    row = {
        "schema": COHORT_SCHEMA,
        "state_fingerprint": state_key,
        "state_blob": str(source["state_blob"]),
        "state_blob_sha256": str(source["state_blob_sha256"]),
        "state_context": dict(source["state_context"]),
        "map_id": str(source["map_id"]),
        "task_id": str(source["task_id"]),
        "solver_seed": int(source["solver_seed"]),
        "split": str(source["split"]),
        "source_run_config": str(source["source_run_config"]),
        "source_run_config_sha256": str(source["source_run_config_sha256"]),
        "logical_checkpoint_ids": list(source["logical_checkpoint_ids"]),
        "v2_anchors": state_anchors,
        "base_candidate_ids": sorted(
            str(candidate["candidate_id"]) for candidate in base_candidates
        ),
        "repairclosure_candidates": novel,
        "generator": {
            "raw_candidate_count": generated.raw_candidate_count,
            "pareto_front_count": generated.pareto_front_count,
            "selected_candidate_count": len(generated.candidates),
            "novel_candidate_count": len(novel),
            "attempts": generated.attempts,
        },
        "candidate_outcomes_used": False,
        "future_trajectory_used": False,
        "repair_order_controlled": False,
    }
    return {"status": "ok", "job_id": state_key, "error_count": 0, "row": row}


def materialize_repairclosure_cohort(
    design_path: str | Path, output: str | Path
) -> dict[str, Any]:
    design_path, _root, config, inputs = _load_design(design_path)
    metadata, states, logical = build_frozen_cohort(inputs["marginalpool_registration"])
    checkpoints = _read_jsonl(inputs["root_checkpoints"])
    anchors = _anchors_by_state(checkpoints)
    aggregates = {
        (str(row["state_fingerprint"]), str(row["candidate_id"])): row
        for row in _read_jsonl(inputs["base_candidate_aggregates"])
    }
    generator = dict(config["generator"])
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    aggregate_ids: dict[str, list[str]] = defaultdict(list)
    for state_key, candidate_id in aggregates:
        aggregate_ids[state_key].append(candidate_id)
    jobs = [
        {
            "job_id": str(source["state_fingerprint"]),
            "source": source,
            "v2_anchors": anchors.get(str(source["state_fingerprint"]), []),
            "registered_base_candidate_ids": aggregate_ids[
                str(source["state_fingerprint"])
            ],
            "generator": generator,
        }
        for source in states
    ]
    results = _run_jobs(
        _materialize_state,
        jobs,
        int(config["execution"]["workers"]),
        phase="repairclosurepool-materialize",
        output_root=output,
        run_fingerprint=sha256_file(design_path),
        timeout_seconds=float(config["execution"]["per_job_timeout_seconds"]),
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") != "ok"]
    if failures or len(results) != len(jobs):
        raise RuntimeError(f"RepairClosurePool materialization failed: {failures}")
    rows = [dict(result["row"]) for result in results]
    rows.sort(key=lambda row: str(row["state_fingerprint"]))
    manifest = output / "repairclosure_cohort.jsonl"
    _write_jsonl(manifest, rows)
    candidate_count = sum(len(row["repairclosure_candidates"]) for row in rows)
    report = {
        "schema": "lns2.stride.repairclosurepool_materialization_report.v1",
        "experiment_id": EXPERIMENT_ID,
        "design_sha256": sha256_file(design_path),
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "source_cohort_fingerprint": metadata["cohort_fingerprint"],
        "state_count": len(rows),
        "logical_checkpoint_count": len(logical),
        "base_candidate_count": sum(len(row["base_candidate_ids"]) for row in rows),
        "novel_candidate_count": candidate_count,
        "minimum_novel_candidates_per_state": min(
            len(row["repairclosure_candidates"]) for row in rows
        ),
        "maximum_novel_candidates_per_state": max(
            len(row["repairclosure_candidates"]) for row in rows
        ),
        "mean_novel_candidates_per_state": candidate_count / len(rows),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "candidate_outcomes_used": False,
        "future_trajectory_used": False,
        "repair_order_controlled": False,
        "integrity_passed": len(rows) == 78 and len(logical) == 90 and candidate_count > 0,
    }
    _write_json(output / "materialization_report.json", report)
    return report


def _load_execution(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != EXECUTION_SCHEMA
        or config.get("scientific_status")
        != "preregistered_repairclosurepool_paired_native_pp_opportunity_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("RepairClosurePool execution identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "design",
        "cohort",
        "materialization_report",
        "base_candidate_aggregates",
    }:
        raise ValueError("RepairClosurePool execution registry changed")
    cohort = dict(config.get("cohort") or {})
    execution = dict(config.get("execution") or {})
    gates = dict(config.get("readiness_gates") or {})
    boundary = dict(config.get("claim_boundary") or {})
    if (
        int(cohort.get("state_count", -1)) != 78
        or int(cohort.get("novel_candidate_count", 0)) <= 0
        or int(execution.get("workers", -1)) != 16
        or execution.get("job_granularity") != "state_x_candidate_x_trial"
        or int(execution.get("per_job_timeout_seconds", -1)) != 300
        or execution.get("stop_on_first_error_or_timeout") is not True
        or int(execution.get("preflight_state_count", -1)) != 2
        or list(execution.get("preflight_trial_indices") or ()) != [0, 1]
        or list(execution.get("full_trial_indices") or ()) != list(range(16))
    ):
        raise ValueError("RepairClosurePool execution contract changed")
    if gates != {
        "minimum_robust_best_base_opportunity_fraction": 0.10,
        "minimum_stable_frontier_addition_fraction": 0.10,
        "zero_errors_and_timeouts": True,
        "base_pool_preserved": True,
    }:
        raise ValueError("RepairClosurePool readiness gates changed")
    if boundary != {
        "candidate_pool_opportunity_audit_only": True,
        "current_ranker_used": False,
        "model_training_allowed": False,
        "repair_order_intervention_allowed": False,
        "runtime_integration_allowed": False,
        "ttf_experiment_allowed": False,
        "long_tail_avoidance_claim_allowed": False,
        "default_controller_replacement_allowed": False,
    }:
        raise ValueError("RepairClosurePool execution claim boundary changed")
    return path, root, config, inputs


def _trial_file(output: Path, state_key: str, candidate_id: str, trial: int) -> Path:
    safe = candidate_id.replace("/", "_")
    return output / "trials" / state_key / f"{safe}__trial_{trial:02d}.json"


def _valid_trial(
    row: dict[str, Any], *, job: dict[str, Any], run_fingerprint: str
) -> bool:
    candidate = dict(job["candidate"])
    before = int(row.get("before_conflicts", -1))
    after = row.get("conflicts_after")
    return bool(
        row.get("schema") == TRIAL_SCHEMA
        and row.get("run_fingerprint") == run_fingerprint
        and row.get("state_fingerprint") == job["state_fingerprint"]
        and row.get("candidate_id") == candidate["candidate_id"]
        and int(row.get("trial_index", -1)) == int(job["trial_index"])
        and int(row.get("pp_seed", -1)) == int(job["pp_seed"])
        and before > 0
        and type(after) is int
        and after >= 0
        and math.isclose(
            float(row.get("normalized_conflict_reduction", math.nan)),
            (before - after) / before,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and row.get("native_action_validated") is True
        and row.get("repair_order_controlled") is False
        and row.get("runtime_fields_stored") is False
        and row.get("future_trajectory_stored") is False
        and not _forbidden_hits(row)
    )


def _collect_trial(job: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if output_path.is_file():
        existing = _read_json(output_path)
        if bool(job["resume"]) and _valid_trial(
            existing, job=job, run_fingerprint=run_fingerprint
        ):
            return {"status": "resumed", "job_id": str(job["job_id"]), "error_count": 0}
        raise ValueError(f"invalid or unrequested RepairClosurePool trial: {output_path}")
    state = read_state_blob(Path(str(job["state_blob"])))
    state["context"] = dict(job["state_context"])
    if state_fingerprint(state) != str(job["state_fingerprint"]):
        raise RuntimeError("RepairClosurePool trial state fingerprint changed")
    if sha256_file(Path(str(job["state_blob"]))) != str(job["state_blob_sha256"]):
        raise RuntimeError("RepairClosurePool trial state blob changed")
    replay_record = {
        key: job[key]
        for key in (
            "source_run_config",
            "source_run_config_sha256",
            "split",
            "task_id",
            "solver_seed",
        )
    }
    replay = _replay_job(replay_record)
    before_repair = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    restore_seed = repairability_restore_seed(before_repair)
    environment, restored = restore_repair_state(replay, state, seed=restore_seed)
    if (
        repair_structure_fingerprint(restored) != before_repair
        or int(restored["num_of_colliding_pairs"]) != before_conflicts
    ):
        raise RuntimeError("RepairClosurePool native restore changed")
    candidate = dict(job["candidate"])
    agents = list(map(int, candidate["agents"]))
    pp_seed = int(job["pp_seed"])
    result = _plain(environment.step(_paired_action(agents, pp_seed)))
    after, metrics = _validate_native_repair(
        result, expected_agents=agents, expected_seed=pp_seed
    )
    conflicts_after = int(after["num_of_colliding_pairs"])
    after_repair = repair_structure_fingerprint(after)
    payload = {
        "schema": TRIAL_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "state_fingerprint": str(job["state_fingerprint"]),
        "candidate_id": str(candidate["candidate_id"]),
        "trial_index": int(job["trial_index"]),
        "pp_seed": pp_seed,
        "before_conflicts": before_conflicts,
        "before_repair_fingerprint": before_repair,
        "conflicts_after": conflicts_after,
        "normalized_conflict_reduction": (
            before_conflicts - conflicts_after
        ) / max(1, before_conflicts),
        "no_progress": conflicts_after >= before_conflicts,
        "replan_success": bool(metrics["replan_success"]),
        "feasible": bool(after.get("feasible")),
        "repair_outcome": classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics["replan_success"]),
            conflicts_before=before_conflicts,
            conflicts_after=conflicts_after,
            feasible=bool(after.get("feasible")),
        ),
        "after_repair_fingerprint": after_repair,
        "native_action_validated": True,
        "repair_order_controlled": False,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
    }
    if not _valid_trial(payload, job=job, run_fingerprint=run_fingerprint):
        raise RuntimeError("RepairClosurePool trial artifact is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)
    return {"status": "ok", "job_id": str(job["job_id"]), "error_count": 0}


def collect_repairclosure_trials(
    *,
    execution_path: str | Path,
    output: str | Path,
    mode: str,
    workers: int | None = None,
    resume: bool = False,
    preflight_output: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in {"preflight", "full"}:
        raise ValueError("RepairClosurePool mode must be preflight or full")
    execution_path, root, config, inputs = _load_execution(execution_path)
    cohort = _read_jsonl(inputs["cohort"])
    expected_states = int(config["cohort"]["state_count"])
    expected_candidates = int(config["cohort"]["novel_candidate_count"])
    if len(cohort) != expected_states or sum(
        len(row["repairclosure_candidates"]) for row in cohort
    ) != expected_candidates:
        raise ValueError("RepairClosurePool materialized cohort changed")
    execution = dict(config["execution"])
    worker_count = int(workers or execution["workers"])
    if worker_count != 16:
        raise ValueError("RepairClosurePool collection requires 16 workers")
    selected = cohort[: int(execution["preflight_state_count"])] if mode == "preflight" else cohort
    trials = tuple(
        map(
            int,
            execution[
                "preflight_trial_indices" if mode == "preflight" else "full_trial_indices"
            ],
        )
    )
    if mode == "full":
        if preflight_output is None:
            raise ValueError("RepairClosurePool full collection requires preflight")
        preflight = _read_json(Path(preflight_output).resolve() / "collection_report.json")
        if (
            preflight.get("schema") != REPORT_SCHEMA
            or preflight.get("mode") != "preflight"
            or preflight.get("integrity_passed") is not True
            or preflight.get("execution_sha256") != sha256_file(execution_path)
        ):
            raise ValueError("RepairClosurePool preflight contract changed")
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    identity = {
        "schema": "lns2.stride.repairclosurepool_run.v1",
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "execution_sha256": sha256_file(execution_path),
        "cohort_sha256": sha256_file(inputs["cohort"]),
        "trial_indices": list(trials),
        "state_fingerprints": [str(row["state_fingerprint"]) for row in selected],
        "producer": producer,
        "target_state_restore_contract": TARGET_STATE_RESTORE_CONTRACT,
    }
    run_fingerprint = _fingerprint(identity)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("RepairClosurePool output belongs to another run")
        if not resume:
            raise ValueError("RepairClosurePool output exists; pass --resume")
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    jobs: list[dict[str, Any]] = []
    for state_row in selected:
        state_key = str(state_row["state_fingerprint"])
        state_blob = Path(str(state_row["state_blob"]))
        if sha256_file(state_blob) != str(state_row["state_blob_sha256"]):
            raise ValueError("RepairClosurePool source state blob changed")
        source_state = read_state_blob(state_blob)
        source_state["context"] = dict(state_row["state_context"])
        if state_fingerprint(source_state) != state_key:
            raise ValueError("RepairClosurePool source state fingerprint changed")
        before_repair = repair_structure_fingerprint(source_state)
        paired_seeds = {
            trial_index: repairability_pp_seed(before_repair, trial_index)
            for trial_index in trials
        }
        for candidate in state_row["repairclosure_candidates"]:
            candidate_id = str(candidate["candidate_id"])
            for trial_index in trials:
                jobs.append(
                    {
                        **{
                            key: state_row[key]
                            for key in (
                                "state_fingerprint",
                                "state_blob",
                                "state_blob_sha256",
                                "state_context",
                                "source_run_config",
                                "source_run_config_sha256",
                                "split",
                                "task_id",
                                "solver_seed",
                            )
                        },
                        "candidate": candidate,
                        "trial_index": trial_index,
                        "pp_seed": paired_seeds[trial_index],
                        "job_id": f"{state_key}:{candidate_id}:{trial_index:02d}",
                        "output_path": str(
                            _trial_file(output, state_key, candidate_id, trial_index)
                        ),
                        "run_fingerprint": run_fingerprint,
                        "resume": bool(resume),
                        "registered_before_repair_fingerprint": before_repair,
                    }
                )
    status_path = output / "collection_status.json"
    progress_path = output / "collection_progress.json"
    observed: list[dict[str, Any]] = []

    def update(result: dict[str, Any]) -> None:
        observed.append(result)
        errors = [row for row in observed if row.get("status") in {"error", "timeout"}]
        _write_json(
            progress_path,
            {
                "schema": STATUS_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "completed_jobs": len(observed) - len(errors),
                "total_jobs": len(jobs),
                "pending_jobs": len(jobs) - len(observed),
                "error_jobs": sum(row.get("status") == "error" for row in errors),
                "timeout_jobs": sum(row.get("status") == "timeout" for row in errors),
                "active_jobs": [],
            },
        )

    _write_json(
        status_path,
        {
            "schema": STATUS_SCHEMA,
            "run_fingerprint": run_fingerprint,
            "status": "running",
            "total_jobs": len(jobs),
            "completed_jobs": 0,
            "error_jobs": 0,
            "timeout_jobs": 0,
        },
    )
    results = _run_jobs(
        _collect_trial,
        jobs,
        worker_count,
        phase=f"repairclosurepool-{mode}",
        output_root=output,
        run_fingerprint=run_fingerprint,
        on_result=update,
        timeout_seconds=float(execution["per_job_timeout_seconds"]),
        stop_on_failure=True,
    )
    failures = [row for row in results if row.get("status") in {"error", "timeout"}]
    status = {
        "schema": STATUS_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "status": "failed" if failures else "complete",
        "total_jobs": len(jobs),
        "completed_jobs": len(results) - len(failures),
        "error_jobs": sum(row.get("status") == "error" for row in failures),
        "timeout_jobs": sum(row.get("status") == "timeout" for row in failures),
        "failures": failures,
        "workers": worker_count,
    }
    _write_json(status_path, status)
    if not failures and len(results) == len(jobs):
        report = analyze_repairclosure_trials(
            execution_path=execution_path,
            collection=output,
            output=output,
            mode=mode,
        )
        return report
    return status


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state_count": len(rows),
        "robust_best_base_opportunity_fraction": statistics.fmean(
            float(row["new_stably_dominates_base_best"]) for row in rows
        ) if rows else 0.0,
        "stable_frontier_addition_fraction": statistics.fmean(
            float(row["stable_frontier_addition_count"] > 0) for row in rows
        ) if rows else 0.0,
        "mean_new_best_seed_mean_advantage": statistics.fmean(
            float(row["new_best_seed_mean_advantage"]) for row in rows
        ) if rows else 0.0,
        "mean_new_candidate_count": statistics.fmean(
            int(row["new_candidate_count"]) for row in rows
        ) if rows else 0.0,
        "mean_new_candidate_size": statistics.fmean(
            float(row["mean_new_candidate_size"]) for row in rows
        ) if rows else 0.0,
    }


def analyze_repairclosure_trials(
    *,
    execution_path: str | Path,
    collection: str | Path,
    output: str | Path,
    mode: str,
) -> dict[str, Any]:
    execution_path, _root, config, inputs = _load_execution(execution_path)
    collection = Path(collection).resolve()
    run = _read_json(collection / "run_config.json")
    status = _read_json(collection / "collection_status.json")
    cohort = _read_jsonl(inputs["cohort"])
    execution = dict(config["execution"])
    selected = cohort[: int(execution["preflight_state_count"])] if mode == "preflight" else cohort
    trial_indices = tuple(
        map(
            int,
            execution[
                "preflight_trial_indices" if mode == "preflight" else "full_trial_indices"
            ],
        )
    )
    expected_jobs = sum(len(row["repairclosure_candidates"]) for row in selected) * len(trial_indices)
    trial_rows: list[dict[str, Any]] = []
    for state_row in selected:
        state_key = str(state_row["state_fingerprint"])
        for candidate in state_row["repairclosure_candidates"]:
            for trial_index in trial_indices:
                path = _trial_file(
                    collection, state_key, str(candidate["candidate_id"]), trial_index
                )
                if not path.is_file():
                    raise ValueError(f"RepairClosurePool trial is missing: {path}")
                trial_rows.append(_read_json(path))
    integrity = {
        "collection_complete": status.get("status") == "complete",
        "expected_job_count": len(trial_rows) == expected_jobs,
        "zero_errors": int(status.get("error_jobs", -1)) == 0,
        "zero_timeouts": int(status.get("timeout_jobs", -1)) == 0,
        "native_actions_valid": all(row.get("native_action_validated") is True for row in trial_rows),
        "repair_order_uncontrolled": all(row.get("repair_order_controlled") is False for row in trial_rows),
        "no_runtime_or_future_fields": all(not _forbidden_hits(row) for row in trial_rows),
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "execution_sha256": sha256_file(execution_path),
        "run_fingerprint": run["run_fingerprint"],
        "state_count": len(selected),
        "job_count": len(trial_rows),
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "current_ranker_used": False,
        "oracle_is_pool_diagnostic_only": True,
        "model_training_allowed": False,
        "ttf_experiment_allowed": False,
    }
    if mode == "full" and report["integrity_passed"]:
        base_aggregates = {
            (str(row["state_fingerprint"]), str(row["candidate_id"])): dict(row)
            for row in _read_jsonl(inputs["base_candidate_aggregates"])
        }
        trials_by_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in trial_rows:
            trials_by_action[(str(row["state_fingerprint"]), str(row["candidate_id"]))].append(row)
        state_results = []
        for state_row in selected:
            state_key = str(state_row["state_fingerprint"])
            base = [base_aggregates[(state_key, candidate_id)] for candidate_id in state_row["base_candidate_ids"]]
            new = [
                aggregate_candidate(
                    candidate,
                    trials_by_action[(state_key, str(candidate["candidate_id"]))],
                )
                for candidate in state_row["repairclosure_candidates"]
            ]
            base_best = max(base, key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"])))
            new_best = max(new, key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"])))
            frontier = [
                candidate
                for candidate in new
                if any(stable_dominates(candidate, old) for old in base)
                and not any(stable_dominates(old, candidate) for old in base)
            ]
            state_results.append(
                {
                    "state_fingerprint": state_key,
                    "map_id": str(state_row["map_id"]),
                    "base_candidate_count": len(base),
                    "new_candidate_count": len(new),
                    "base_best_candidate_id": str(base_best["candidate_id"]),
                    "new_best_candidate_id": str(new_best["candidate_id"]),
                    "new_stably_dominates_base_best": stable_dominates(new_best, base_best),
                    "stable_frontier_addition_count": len(frontier),
                    "new_best_seed_mean_advantage": float(new_best["seed_mean"]) - float(base_best["seed_mean"]),
                    "new_best_first_half_advantage": float(new_best["first_fixed_half_mean"]) - float(base_best["first_fixed_half_mean"]),
                    "new_best_second_half_advantage": float(new_best["second_fixed_half_mean"]) - float(base_best["second_fixed_half_mean"]),
                    "mean_new_candidate_size": statistics.fmean(int(row["actual_size"]) for row in new),
                    "maximum_new_candidate_size": max(int(row["actual_size"]) for row in new),
                }
            )
        by_map = {
            map_id: _summary([row for row in state_results if row["map_id"] == map_id])
            for map_id in sorted({str(row["map_id"]) for row in state_results})
        }
        overall = _summary(state_results)
        gates = dict(config["readiness_gates"])
        pool_opportunity_passed = (
            overall["robust_best_base_opportunity_fraction"]
            >= float(gates["minimum_robust_best_base_opportunity_fraction"])
            and overall["stable_frontier_addition_fraction"]
            >= float(gates["minimum_stable_frontier_addition_fraction"])
        )
        report.update(
            {
                "overall": overall,
                "by_map": by_map,
                "states": state_results,
                "readiness_gates": gates,
                "pool_opportunity_passed": pool_opportunity_passed,
                "next_step": "result-blind bounded forced-continuation confirmation"
                if pool_opportunity_passed
                else "revise outcome-blind dependency closure; do not train a ranker",
            }
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "collection_report.json", report)
    return report


__all__ = [
    "analyze_repairclosure_trials",
    "collect_repairclosure_trials",
    "materialize_repairclosure_cohort",
]
