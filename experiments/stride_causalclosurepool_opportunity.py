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
    state_fingerprint,
)
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_marginalpool_action_replay import (
    _replay_job,
    aggregate_candidate,
    stable_dominates,
)
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_repairclosurepool import _summary as _opportunity_summary
from experiments.stride_robustaction_label_collection import _forbidden_hits
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT, restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


EXECUTION_SCHEMA = "lns2.stride.causalclosurepool_opportunity_execution.v1"
TRIAL_SCHEMA = "lns2.stride.causalclosurepool_opportunity_trial.v1"
STATUS_SCHEMA = "lns2.stride.causalclosurepool_opportunity_status.v1"
REPORT_SCHEMA = "lns2.stride.causalclosurepool_opportunity_report.v1"
EXPERIMENT_ID = "stride-causalclosurepool-v2-opportunity-v1"
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/closed_loop_trace_storage.py",
    "experiments/repair_collection.py",
    "experiments/stride_causalclosurepool_opportunity.py",
    "experiments/stride_collection.py",
    "experiments/stride_marginalpool_action_replay.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "lns2_selector/runtime/repair_outcomes.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _load_execution(
    path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(path).resolve()
    root = path.parents[1]
    config = _read_json(path)
    if (
        config.get("schema") != EXECUTION_SCHEMA
        or config.get("scientific_status")
        != "preregistered_ranker_free_paired_native_pp_opportunity_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("CausalClosurePool opportunity identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "generator_design",
        "cohort",
        "compactness_report",
        "base_candidate_aggregates",
    }:
        raise ValueError("CausalClosurePool opportunity inputs changed")
    compactness = _read_json(inputs["compactness_report"])
    if (
        compactness.get("compactness_readiness_passed") is not True
        or int(compactness.get("state_count", -1)) != 78
        or int(compactness.get("candidate_count", -1)) != 930
        or compactness.get("native_pp_run") is not False
    ):
        raise ValueError("CausalClosurePool compactness prerequisite changed")
    cohort = dict(config.get("cohort") or {})
    execution = dict(config.get("execution") or {})
    if (
        int(cohort.get("state_count", -1)) != 78
        or int(cohort.get("causal_candidate_count", -1)) != 930
        or int(cohort.get("base_candidate_count", -1)) != 1367
        or int(execution.get("workers", -1)) != 16
        or execution.get("job_granularity") != "state_x_candidate_x_trial"
        or int(execution.get("per_job_timeout_seconds", -1)) != 300
        or execution.get("stop_on_first_error_or_timeout") is not True
        or int(execution.get("preflight_state_count", -1)) != 2
        or list(execution.get("preflight_trial_indices") or ()) != [0, 1]
        or list(execution.get("full_trial_indices") or ()) != list(range(16))
    ):
        raise ValueError("CausalClosurePool opportunity execution changed")
    if dict(config.get("readiness_gates") or {}) != {
        "minimum_robust_best_base_opportunity_fraction": 0.10,
        "minimum_stable_frontier_addition_fraction": 0.10,
        "zero_errors_and_timeouts": True,
        "base_pool_preserved": True,
    }:
        raise ValueError("CausalClosurePool opportunity gates changed")
    if dict(config.get("claim_boundary") or {}) != {
        "candidate_pool_opportunity_audit_only": True,
        "current_ranker_used": False,
        "model_training_allowed": False,
        "repair_order_intervention_allowed": False,
        "runtime_integration_allowed": False,
        "ttf_experiment_allowed": False,
        "long_tail_avoidance_claim_allowed": False,
        "default_controller_replacement_allowed": False,
    }:
        raise ValueError("CausalClosurePool opportunity boundary changed")
    return path, root, config, inputs


def _trial_file(output: Path, state_key: str, action_id: str, trial: int) -> Path:
    safe = action_id.replace("/", "_")
    return output / "trials" / state_key / f"{safe}__trial_{trial:02d}.json"


def _valid_trial(
    row: dict[str, Any], *, job: dict[str, Any], run_fingerprint: str
) -> bool:
    before = int(row.get("before_conflicts", -1))
    after = row.get("conflicts_after")
    candidate = dict(job["candidate"])
    expected_schema = str(job.get("trial_schema", TRIAL_SCHEMA))
    return bool(
        row.get("schema") == expected_schema
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
        raise ValueError(f"invalid or unrequested CausalClosurePool trial: {output_path}")
    state_path = Path(str(job["state_blob"]))
    if sha256_file(state_path) != str(job["state_blob_sha256"]):
        raise RuntimeError("CausalClosurePool trial state blob changed")
    state = read_state_blob(state_path)
    state["context"] = dict(job["state_context"])
    if state_fingerprint(state) != str(job["state_fingerprint"]):
        raise RuntimeError("CausalClosurePool trial state fingerprint changed")
    replay = _replay_job(
        {
            key: job[key]
            for key in (
                "source_run_config",
                "source_run_config_sha256",
                "split",
                "task_id",
                "solver_seed",
            )
        }
    )
    before_repair = repair_structure_fingerprint(state)
    if before_repair != str(job["registered_before_repair_fingerprint"]):
        raise RuntimeError("CausalClosurePool registered repair fingerprint changed")
    before_conflicts = int(state["num_of_colliding_pairs"])
    environment, restored = restore_repair_state(
        replay, state, seed=repairability_restore_seed(before_repair)
    )
    if (
        repair_structure_fingerprint(restored) != before_repair
        or int(restored["num_of_colliding_pairs"]) != before_conflicts
    ):
        raise RuntimeError("CausalClosurePool native restore changed")
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
        "schema": str(job.get("trial_schema", TRIAL_SCHEMA)),
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
        raise RuntimeError("CausalClosurePool trial artifact is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)
    return {"status": "ok", "job_id": str(job["job_id"]), "error_count": 0}


def _load_cohort(
    config: dict[str, Any], inputs: dict[str, Path]
) -> list[dict[str, Any]]:
    cohort = _read_jsonl(inputs["cohort"])
    if len(cohort) != int(config["cohort"]["state_count"]):
        raise ValueError("CausalClosurePool state count changed")
    candidates = sum(len(row["causalclosure_candidates"]) for row in cohort)
    if candidates != int(config["cohort"]["causal_candidate_count"]):
        raise ValueError("CausalClosurePool candidate count changed")
    base_count = sum(len(row["base_candidate_ids"]) for row in cohort)
    if base_count != int(config["cohort"]["base_candidate_count"]):
        raise ValueError("CausalClosurePool base pool changed")
    return cohort


def collect_causalclosure_trials(
    *,
    execution_path: str | Path,
    output: str | Path,
    mode: str,
    resume: bool = False,
    preflight_output: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in {"preflight", "full"}:
        raise ValueError("CausalClosurePool mode must be preflight or full")
    execution_path, root, config, inputs = _load_execution(execution_path)
    cohort = _load_cohort(config, inputs)
    execution = dict(config["execution"])
    selected = (
        cohort[: int(execution["preflight_state_count"])]
        if mode == "preflight"
        else cohort
    )
    trial_indices = tuple(
        map(
            int,
            execution[
                "preflight_trial_indices" if mode == "preflight" else "full_trial_indices"
            ],
        )
    )
    if mode == "full":
        if preflight_output is None:
            raise ValueError("CausalClosurePool full collection requires preflight")
        preflight = _read_json(Path(preflight_output).resolve() / "collection_report.json")
        if (
            preflight.get("schema") != REPORT_SCHEMA
            or preflight.get("mode") != "preflight"
            or preflight.get("integrity_passed") is not True
            or preflight.get("execution_sha256") != sha256_file(execution_path)
        ):
            raise ValueError("CausalClosurePool preflight changed")
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    identity = {
        "schema": "lns2.stride.causalclosurepool_opportunity_run.v1",
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "execution_sha256": sha256_file(execution_path),
        "cohort_sha256": sha256_file(inputs["cohort"]),
        "trial_indices": list(trial_indices),
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
            raise ValueError("CausalClosurePool output belongs to another run")
        if not resume:
            raise ValueError("CausalClosurePool output exists; pass --resume")
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    jobs: list[dict[str, Any]] = []
    for state_row in selected:
        state_path = Path(str(state_row["state_blob"]))
        state = read_state_blob(state_path)
        state["context"] = dict(state_row["state_context"])
        state_key = str(state_row["state_fingerprint"])
        if state_fingerprint(state) != state_key:
            raise ValueError("CausalClosurePool source state changed")
        before_repair = repair_structure_fingerprint(state)
        paired_seeds = {
            trial: repairability_pp_seed(before_repair, trial)
            for trial in trial_indices
        }
        for candidate in state_row["causalclosure_candidates"]:
            action_id = str(candidate["candidate_id"])
            for trial in trial_indices:
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
                        "trial_index": trial,
                        "pp_seed": paired_seeds[trial],
                        "job_id": f"{state_key}:{action_id}:{trial:02d}",
                        "output_path": str(_trial_file(output, state_key, action_id, trial)),
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
        failures = [row for row in observed if row.get("status") in {"error", "timeout"}]
        _write_json(
            progress_path,
            {
                "schema": STATUS_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "completed_jobs": len(observed) - len(failures),
                "total_jobs": len(jobs),
                "pending_jobs": len(jobs) - len(observed),
                "error_jobs": sum(row.get("status") == "error" for row in failures),
                "timeout_jobs": sum(row.get("status") == "timeout" for row in failures),
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
            "workers": 16,
        },
    )
    results = _run_jobs(
        _collect_trial,
        jobs,
        16,
        phase=f"causalclosurepool-opportunity-{mode}",
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
        "workers": 16,
    }
    _write_json(status_path, status)
    if not failures and len(results) == len(jobs):
        return analyze_causalclosure_trials(
            execution_path=execution_path,
            collection=output,
            output=output,
            mode=mode,
        )
    return status


def analyze_causalclosure_trials(
    *,
    execution_path: str | Path,
    collection: str | Path,
    output: str | Path,
    mode: str,
) -> dict[str, Any]:
    execution_path, _root, config, inputs = _load_execution(execution_path)
    cohort = _load_cohort(config, inputs)
    execution = dict(config["execution"])
    selected = (
        cohort[: int(execution["preflight_state_count"])]
        if mode == "preflight"
        else cohort
    )
    trials = tuple(
        map(
            int,
            execution[
                "preflight_trial_indices" if mode == "preflight" else "full_trial_indices"
            ],
        )
    )
    collection = Path(collection).resolve()
    run = _read_json(collection / "run_config.json")
    status = _read_json(collection / "collection_status.json")
    rows: list[dict[str, Any]] = []
    for state_row in selected:
        state_key = str(state_row["state_fingerprint"])
        for candidate in state_row["causalclosure_candidates"]:
            for trial in trials:
                path = _trial_file(collection, state_key, str(candidate["candidate_id"]), trial)
                if not path.is_file():
                    raise ValueError(f"CausalClosurePool trial is missing: {path}")
                rows.append(_read_json(path))
    expected = sum(len(row["causalclosure_candidates"]) for row in selected) * len(trials)
    integrity = {
        "collection_complete": status.get("status") == "complete",
        "expected_job_count": len(rows) == expected,
        "zero_errors": int(status.get("error_jobs", -1)) == 0,
        "zero_timeouts": int(status.get("timeout_jobs", -1)) == 0,
        "native_actions_valid": all(row.get("native_action_validated") is True for row in rows),
        "repair_order_uncontrolled": all(row.get("repair_order_controlled") is False for row in rows),
        "no_runtime_or_future_fields": all(not _forbidden_hits(row) for row in rows),
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "execution_sha256": sha256_file(execution_path),
        "run_fingerprint": run["run_fingerprint"],
        "state_count": len(selected),
        "candidate_count": sum(len(row["causalclosure_candidates"]) for row in selected),
        "job_count": len(rows),
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "current_ranker_used": False,
        "oracle_is_pool_diagnostic_only": True,
        "model_training_allowed": False,
        "ttf_experiment_allowed": False,
    }
    if mode == "full" and report["integrity_passed"]:
        base = {
            (str(row["state_fingerprint"]), str(row["candidate_id"])): dict(row)
            for row in _read_jsonl(inputs["base_candidate_aggregates"])
        }
        by_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_action[(str(row["state_fingerprint"]), str(row["candidate_id"]))].append(row)
        state_results = []
        for state_row in selected:
            state_key = str(state_row["state_fingerprint"])
            base_rows = [base[(state_key, action)] for action in state_row["base_candidate_ids"]]
            new_rows = [
                aggregate_candidate(
                    candidate,
                    by_action[(state_key, str(candidate["candidate_id"]))],
                )
                for candidate in state_row["causalclosure_candidates"]
            ]
            base_best = max(base_rows, key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"])))
            new_best = max(new_rows, key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"])))
            frontier = [
                candidate
                for candidate in new_rows
                if any(stable_dominates(candidate, old) for old in base_rows)
                and not any(stable_dominates(old, candidate) for old in base_rows)
            ]
            state_results.append(
                {
                    "state_fingerprint": state_key,
                    "map_id": str(state_row["map_id"]),
                    "base_candidate_count": len(base_rows),
                    "new_candidate_count": len(new_rows),
                    "base_best_candidate_id": str(base_best["candidate_id"]),
                    "new_best_candidate_id": str(new_best["candidate_id"]),
                    "new_stably_dominates_base_best": stable_dominates(new_best, base_best),
                    "stable_frontier_addition_count": len(frontier),
                    "new_best_seed_mean_advantage": float(new_best["seed_mean"]) - float(base_best["seed_mean"]),
                    "new_best_first_half_advantage": float(new_best["first_fixed_half_mean"]) - float(base_best["first_fixed_half_mean"]),
                    "new_best_second_half_advantage": float(new_best["second_fixed_half_mean"]) - float(base_best["second_fixed_half_mean"]),
                    "mean_new_candidate_size": statistics.fmean(int(row["actual_size"]) for row in new_rows),
                    "maximum_new_candidate_size": max(int(row["actual_size"]) for row in new_rows),
                }
            )
        overall = _opportunity_summary(state_results)
        by_map = {
            map_id: _opportunity_summary(
                [row for row in state_results if row["map_id"] == map_id]
            )
            for map_id in sorted({row["map_id"] for row in state_results})
        }
        gates = dict(config["readiness_gates"])
        passed = (
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
                "pool_opportunity_passed": passed,
                "next_step": (
                    "separately preregister result-blind forced-continuation confirmation"
                    if passed
                    else "stop CausalClosurePool; do not train or select successful states"
                ),
            }
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "collection_report.json", report)
    return report


__all__ = [
    "EXECUTION_SCHEMA",
    "EXPERIMENT_ID",
    "REPORT_SCHEMA",
    "TRIAL_SCHEMA",
    "_valid_trial",
    "analyze_causalclosure_trials",
    "collect_causalclosure_trials",
]
