from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_state_blob
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    state_fingerprint,
)
from experiments.stride_causalclosurepool_opportunity import (
    _collect_trial,
    _trial_file,
    _valid_trial,
)
from experiments.stride_marginalpool_action_replay import (
    aggregate_candidate,
    stable_dominates,
)
from experiments.stride_repairability_collection import repairability_pp_seed
from experiments.stride_robustaction_label_collection import _forbidden_hits
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


EXECUTION_SCHEMA = "lns2.stride.causaltopopool_opportunity_execution.v1"
TRIAL_SCHEMA = "lns2.stride.causaltopopool_opportunity_trial.v1"
STATUS_SCHEMA = "lns2.stride.causaltopopool_opportunity_status.v1"
REPORT_SCHEMA = "lns2.stride.causaltopopool_opportunity_report.v1"
EXPERIMENT_ID = "stride-causaltopopool-opportunity-v1"
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/closed_loop_trace_storage.py",
    "experiments/repair_collection.py",
    "experiments/stride_causalclosurepool_opportunity.py",
    "experiments/stride_causaltopopool_opportunity.py",
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
        != "preregistered_ranker_free_paired_native_pp_topology_opportunity_audit"
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("CausalTopoPool opportunity identity changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config.get("inputs") or {}).items()
    }
    if set(inputs) != {
        "generator_design",
        "cohort",
        "compactness_report",
        "base_candidate_aggregates",
        "causalclosure_opportunity_report",
    }:
        raise ValueError("CausalTopoPool opportunity inputs changed")
    compactness = _read_json(inputs["compactness_report"])
    if (
        compactness.get("compactness_readiness_passed") is not True
        or int(compactness.get("state_count", -1)) != 78
        or int(compactness.get("topology_candidate_count", -1)) != 366
        or int(compactness.get("truncated_closure_count", -1)) != 0
        or compactness.get("native_pp_run") is not False
    ):
        raise ValueError("CausalTopoPool compactness prerequisite changed")
    causal_report = _read_json(inputs["causalclosure_opportunity_report"])
    if (
        causal_report.get("integrity_passed") is not True
        or int(causal_report.get("state_count", -1)) != 78
        or int(causal_report.get("candidate_count", -1)) != 930
    ):
        raise ValueError("CausalClosure opportunity prerequisite changed")
    cohort = dict(config.get("cohort") or {})
    execution = dict(config.get("execution") or {})
    if (
        int(cohort.get("state_count", -1)) != 78
        or int(cohort.get("topology_candidate_count", -1)) != 366
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
        raise ValueError("CausalTopoPool opportunity execution changed")
    if dict(config.get("readiness_gates") or {}) != {
        "minimum_topology_robust_best_base_opportunity_fraction": 0.3,
        "minimum_hybrid_robust_best_base_opportunity_fraction": 0.4,
        "minimum_topology_stable_frontier_addition_fraction": 0.3,
        "minimum_per_map_hybrid_opportunity_fraction": 0.25,
        "zero_errors_and_timeouts": True,
        "base_and_causal_pools_preserved": True,
    }:
        raise ValueError("CausalTopoPool opportunity gates changed")
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
        raise ValueError("CausalTopoPool opportunity boundary changed")
    return path, root, config, inputs


def _load_cohort(config: dict[str, Any], inputs: dict[str, Path]) -> list[dict[str, Any]]:
    rows = _read_jsonl(inputs["cohort"])
    if len(rows) != int(config["cohort"]["state_count"]):
        raise ValueError("CausalTopoPool state count changed")
    if sum(len(row["causaltopology_candidates"]) for row in rows) != int(
        config["cohort"]["topology_candidate_count"]
    ):
        raise ValueError("CausalTopoPool topology candidate count changed")
    if sum(len(row["causalclosure_candidates"]) for row in rows) != int(
        config["cohort"]["causal_candidate_count"]
    ):
        raise ValueError("CausalTopoPool preserved causal pool changed")
    if sum(len(row["base_candidate_ids"]) for row in rows) != int(
        config["cohort"]["base_candidate_count"]
    ):
        raise ValueError("CausalTopoPool preserved base pool changed")
    return rows


def collect_causaltopology_trials(
    *,
    execution_path: str | Path,
    output: str | Path,
    mode: str,
    resume: bool = False,
    preflight_output: str | Path | None = None,
) -> dict[str, Any]:
    if mode not in {"preflight", "full"}:
        raise ValueError("CausalTopoPool mode must be preflight or full")
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
            raise ValueError("CausalTopoPool full collection requires preflight")
        preflight = _read_json(Path(preflight_output).resolve() / "collection_report.json")
        if (
            preflight.get("schema") != REPORT_SCHEMA
            or preflight.get("mode") != "preflight"
            or preflight.get("integrity_passed") is not True
            or preflight.get("execution_sha256") != sha256_file(execution_path)
        ):
            raise ValueError("CausalTopoPool preflight changed")

    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    identity = {
        "schema": "lns2.stride.causaltopopool_opportunity_run.v1",
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
            raise ValueError("CausalTopoPool output belongs to another run")
        if not resume:
            raise ValueError("CausalTopoPool output exists; pass --resume")
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})

    jobs: list[dict[str, Any]] = []
    for state_row in selected:
        state_path = Path(str(state_row["state_blob"]))
        state = read_state_blob(state_path)
        state["context"] = dict(state_row["state_context"])
        state_key = str(state_row["state_fingerprint"])
        if state_fingerprint(state) != state_key:
            raise ValueError("CausalTopoPool source state changed")
        before_repair = repair_structure_fingerprint(state)
        paired_seeds = {
            trial: repairability_pp_seed(before_repair, trial)
            for trial in trial_indices
        }
        for candidate in state_row["causaltopology_candidates"]:
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
                        "trial_schema": TRIAL_SCHEMA,
                        "trial_index": trial,
                        "pp_seed": paired_seeds[trial],
                        "job_id": f"{state_key}:{action_id}:{trial:02d}",
                        "output_path": str(
                            _trial_file(output, state_key, action_id, trial)
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
        phase=f"causaltopopool-opportunity-{mode}",
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
        return analyze_causaltopology_trials(
            execution_path=execution_path,
            collection=output,
            output=output,
            mode=mode,
        )
    return status


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state_count": len(rows),
        "topology_robust_best_base_opportunity_fraction": statistics.fmean(
            float(row["topology_stably_dominates_base_best"]) for row in rows
        )
        if rows
        else 0.0,
        "topology_stable_frontier_addition_fraction": statistics.fmean(
            float(row["topology_stable_frontier_addition_count"] > 0) for row in rows
        )
        if rows
        else 0.0,
        "hybrid_robust_best_base_opportunity_fraction": statistics.fmean(
            float(row["hybrid_has_robust_opportunity"]) for row in rows
        )
        if rows
        else 0.0,
        "additional_topology_opportunity_over_causal_fraction": statistics.fmean(
            float(row["topology_adds_opportunity_over_causal"]) for row in rows
        )
        if rows
        else 0.0,
        "mean_topology_best_seed_mean_advantage": statistics.fmean(
            float(row["topology_best_seed_mean_advantage"]) for row in rows
        )
        if rows
        else 0.0,
        "mean_topology_candidate_count": statistics.fmean(
            int(row["topology_candidate_count"]) for row in rows
        )
        if rows
        else 0.0,
        "mean_topology_candidate_size": statistics.fmean(
            float(row["mean_topology_candidate_size"]) for row in rows
        )
        if rows
        else 0.0,
    }


def _size_band(size: int) -> str:
    if size <= 8:
        return "02_08"
    if size <= 16:
        return "09_16"
    if size <= 32:
        return "17_32"
    if size <= 48:
        return "33_48"
    return "49_64"


def analyze_causaltopology_trials(
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
    trial_indices = tuple(
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
    trial_rows: list[dict[str, Any]] = []
    validation_errors: list[str] = []
    for state_row in selected:
        state_key = str(state_row["state_fingerprint"])
        before_repair = None
        for candidate in state_row["causaltopology_candidates"]:
            for trial in trial_indices:
                path = _trial_file(
                    collection, state_key, str(candidate["candidate_id"]), trial
                )
                if not path.is_file():
                    raise ValueError(f"CausalTopoPool trial is missing: {path}")
                row = _read_json(path)
                if before_repair is None:
                    before_repair = str(row.get("before_repair_fingerprint"))
                job = {
                    "candidate": candidate,
                    "trial_schema": TRIAL_SCHEMA,
                    "state_fingerprint": state_key,
                    "trial_index": trial,
                    "pp_seed": repairability_pp_seed(str(before_repair), trial),
                }
                if not _valid_trial(row, job=job, run_fingerprint=run["run_fingerprint"]):
                    validation_errors.append(f"{state_key}:{candidate['candidate_id']}:{trial}")
                trial_rows.append(row)
    expected = sum(len(row["causaltopology_candidates"]) for row in selected) * len(
        trial_indices
    )
    integrity = {
        "collection_complete": status.get("status") == "complete",
        "expected_job_count": len(trial_rows) == expected,
        "zero_errors": int(status.get("error_jobs", -1)) == 0,
        "zero_timeouts": int(status.get("timeout_jobs", -1)) == 0,
        "registered_trials_valid": not validation_errors,
        "native_actions_valid": all(
            row.get("native_action_validated") is True for row in trial_rows
        ),
        "repair_order_uncontrolled": all(
            row.get("repair_order_controlled") is False for row in trial_rows
        ),
        "no_runtime_or_future_fields": all(
            not _forbidden_hits(row) for row in trial_rows
        ),
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "mode": mode,
        "execution_sha256": sha256_file(execution_path),
        "run_fingerprint": run["run_fingerprint"],
        "state_count": len(selected),
        "candidate_count": sum(
            len(row["causaltopology_candidates"]) for row in selected
        ),
        "job_count": len(trial_rows),
        "integrity": integrity,
        "integrity_errors": validation_errors,
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
        causal_report = _read_json(inputs["causalclosure_opportunity_report"])
        causal_by_state = {
            str(row["state_fingerprint"]): dict(row)
            for row in causal_report["states"]
        }
        by_action: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in trial_rows:
            by_action[(str(row["state_fingerprint"]), str(row["candidate_id"]))].append(
                row
            )
        state_results: list[dict[str, Any]] = []
        candidate_results: list[dict[str, Any]] = []
        for state_row in selected:
            state_key = str(state_row["state_fingerprint"])
            base_rows = [
                base[(state_key, str(identity))]
                for identity in state_row["base_candidate_ids"]
            ]
            topology_rows = [
                aggregate_candidate(
                    candidate,
                    by_action[(state_key, str(candidate["candidate_id"]))],
                )
                for candidate in state_row["causaltopology_candidates"]
            ]
            topology_metadata = {
                str(candidate["candidate_id"]): candidate
                for candidate in state_row["causaltopology_candidates"]
            }
            base_best = max(
                base_rows,
                key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"])),
            )
            topology_best = max(
                topology_rows,
                key=lambda row: (float(row["seed_mean"]), str(row["candidate_id"])),
            )
            frontier = [
                candidate
                for candidate in topology_rows
                if any(stable_dominates(candidate, old) for old in base_rows)
                and not any(stable_dominates(old, candidate) for old in base_rows)
            ]
            causal_state = causal_by_state[state_key]
            if str(causal_state["base_best_candidate_id"]) != str(
                base_best["candidate_id"]
            ):
                raise ValueError("CausalTopoPool base-best identity changed")
            topology_opportunity = stable_dominates(topology_best, base_best)
            causal_opportunity = bool(causal_state["new_stably_dominates_base_best"])
            for candidate in topology_rows:
                metadata = topology_metadata[str(candidate["candidate_id"])]
                candidate_results.append(
                    {
                        "state_fingerprint": state_key,
                        "map_id": str(state_row["map_id"]),
                        "candidate_id": str(candidate["candidate_id"]),
                        "actual_size": int(candidate["actual_size"]),
                        "size_band": _size_band(int(candidate["actual_size"])),
                        "family_groups": list(metadata["structpool_family_groups"]),
                        "closure_levels": list(metadata["causaltopo_closure_levels"]),
                        "stably_dominates_base_best": stable_dominates(
                            candidate, base_best
                        ),
                        "seed_mean_advantage": float(candidate["seed_mean"])
                        - float(base_best["seed_mean"]),
                        "no_progress_rate_advantage": float(base_best["no_progress_rate"])
                        - float(candidate["no_progress_rate"]),
                    }
                )
            state_results.append(
                {
                    "state_fingerprint": state_key,
                    "map_id": str(state_row["map_id"]),
                    "base_candidate_count": len(base_rows),
                    "causal_candidate_count": len(state_row["causalclosure_candidates"]),
                    "topology_candidate_count": len(topology_rows),
                    "base_best_candidate_id": str(base_best["candidate_id"]),
                    "topology_best_candidate_id": str(topology_best["candidate_id"]),
                    "topology_stably_dominates_base_best": topology_opportunity,
                    "causal_stably_dominates_base_best": causal_opportunity,
                    "hybrid_has_robust_opportunity": topology_opportunity
                    or causal_opportunity,
                    "topology_adds_opportunity_over_causal": topology_opportunity
                    and not causal_opportunity,
                    "topology_stable_frontier_addition_count": len(frontier),
                    "topology_best_seed_mean_advantage": float(
                        topology_best["seed_mean"]
                    )
                    - float(base_best["seed_mean"]),
                    "topology_best_first_half_advantage": float(
                        topology_best["first_fixed_half_mean"]
                    )
                    - float(base_best["first_fixed_half_mean"]),
                    "topology_best_second_half_advantage": float(
                        topology_best["second_fixed_half_mean"]
                    )
                    - float(base_best["second_fixed_half_mean"]),
                    "mean_topology_candidate_size": statistics.fmean(
                        int(row["actual_size"]) for row in topology_rows
                    ),
                    "maximum_topology_candidate_size": max(
                        int(row["actual_size"]) for row in topology_rows
                    ),
                }
            )
        overall = _summary(state_results)
        by_map = {
            map_id: _summary(
                [row for row in state_results if row["map_id"] == map_id]
            )
            for map_id in sorted({row["map_id"] for row in state_results})
        }
        by_size_band = {}
        for band in ("02_08", "09_16", "17_32", "33_48", "49_64"):
            rows = [row for row in candidate_results if row["size_band"] == band]
            by_size_band[band] = {
                "candidate_count": len(rows),
                "robust_candidate_count": sum(
                    bool(row["stably_dominates_base_best"]) for row in rows
                ),
                "state_opportunity_count": len(
                    {
                        row["state_fingerprint"]
                        for row in rows
                        if row["stably_dominates_base_best"]
                    }
                ),
                "mean_seed_mean_advantage": statistics.fmean(
                    float(row["seed_mean_advantage"]) for row in rows
                )
                if rows
                else None,
            }
        gates_config = dict(config["readiness_gates"])
        gates = {
            "topology_robust_best_base_opportunity_fraction": overall[
                "topology_robust_best_base_opportunity_fraction"
            ]
            >= float(
                gates_config[
                    "minimum_topology_robust_best_base_opportunity_fraction"
                ]
            ),
            "hybrid_robust_best_base_opportunity_fraction": overall[
                "hybrid_robust_best_base_opportunity_fraction"
            ]
            >= float(
                gates_config["minimum_hybrid_robust_best_base_opportunity_fraction"]
            ),
            "topology_stable_frontier_addition_fraction": overall[
                "topology_stable_frontier_addition_fraction"
            ]
            >= float(
                gates_config[
                    "minimum_topology_stable_frontier_addition_fraction"
                ]
            ),
            "per_map_hybrid_opportunity_fraction": min(
                row["hybrid_robust_best_base_opportunity_fraction"]
                for row in by_map.values()
            )
            >= float(gates_config["minimum_per_map_hybrid_opportunity_fraction"]),
            "zero_errors_and_timeouts": report["integrity_passed"],
            "base_and_causal_pools_preserved": True,
        }
        passed = all(gates.values())
        report.update(
            {
                "overall": overall,
                "by_map": by_map,
                "by_size_band": by_size_band,
                "states": state_results,
                "robust_candidate_count": sum(
                    bool(row["stably_dominates_base_best"])
                    for row in candidate_results
                ),
                "readiness_thresholds": gates_config,
                "readiness_gates": gates,
                "pool_opportunity_passed": passed,
                "next_step": (
                    "preregister independent result-blind forced-continuation confirmation"
                    if passed
                    else "revise natural topology relations; do not train or select successful states"
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
    "analyze_causaltopology_trials",
    "collect_causaltopology_trials",
]
