from __future__ import annotations

import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import mean, producer_identity, registered_input, sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _run_jobs,
    _write_json,
)
from experiments.state_analysis import analyze_state
from experiments.stride_nativeorder_transactionalrepair import (
    _load_restored_source,
    _timed_attempt,
    _trial_files,
    build_nativeorder_cohort,
    retry_pp_seed,
)
from experiments.stride_repairability_collection import repairability_pp_seed
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repairdependencypool import (
    _serialize_evidence,
    _temporal_corridor_evidence,
)


CONFIG_SCHEMA = "lns2.stride.compact_nativeorder_repair_registration.v1"
RUN_SCHEMA = "lns2.stride.compact_nativeorder_repair_run.v1"
TRIAL_SCHEMA = "lns2.stride.compact_nativeorder_repair_trial.v1"
POLICY_SCHEMA = "lns2.stride.compact_nativeorder_repair_policy.v1"
STATUS_SCHEMA = "lns2.stride.compact_nativeorder_repair_status.v1"
REPORT_SCHEMA = "lns2.stride.compact_nativeorder_repair_report.v1"
EXPERIMENT_ID = "stride-compact-nativeorder-repair-v1"

FULL_SINGLE = "full_single_attempt"
FULL_RETRY = "full_same_set_native_retry"
COMPACT_SINGLE = "compact_single_attempt"
COMPACT_RETRY = "compact_same_set_native_retry"
POLICIES = (FULL_SINGLE, FULL_RETRY, COMPACT_SINGLE, COMPACT_RETRY)
RETRY_POLICIES = (FULL_RETRY, COMPACT_RETRY)
COMPACT_POLICIES = (COMPACT_SINGLE, COMPACT_RETRY)
PHASES = ("initial", "extension")

PRODUCER_FILES = (
    "experiments/stride_compact_nativeorder_repair.py",
    "scripts/run_stride_compact_nativeorder_repair.py",
    "experiments/stride_nativeorder_transactionalrepair.py",
    "experiments/stride_transactionalrepair.py",
    "experiments/stride_repairability_causal_audit.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "experiments/state_analysis.py",
    "lns2_selector/runtime/repairdependencypool.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], Path, Path]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("Compact NativeOrder registration changed")
    inputs = {
        name: registered_input(root, row, label=f"compact NativeOrder {name}")
        for name, row in dict(config["inputs"]).items()
    }
    if set(inputs) != {"parent_registration", "parent_report"}:
        raise ValueError("Compact NativeOrder input registry changed")
    parent_report = _read_json(inputs["parent_report"])
    if (
        parent_report.get("integrity_passed") is not True
        or parent_report.get("mechanism_selection") != "same_set_native_retry"
        or parent_report.get("phase") != "extended"
    ):
        raise ValueError("Compact NativeOrder parent result changed")
    cohort = dict(config["cohort"])
    if (
        int(cohort["required_state_count"]) != 45
        or int(cohort["required_map_count"]) != 3
        or list(map(int, cohort["initial_trial_indices"])) != list(range(8))
        or list(map(int, cohort["extension_trial_indices"])) != list(range(8, 16))
    ):
        raise ValueError("Compact NativeOrder cohort changed")
    policies = dict(config["policies"])
    if tuple(policies["ids"]) != POLICIES or int(policies["maximum_attempts"]) != 2:
        raise ValueError("Compact NativeOrder policies changed")
    execution = dict(config["execution"])
    if (
        int(execution["worker_count"]) != 16
        or float(execution["transaction_pp_wall_budget_seconds"]) != 270.0
        or float(execution["per_policy_job_timeout_seconds"]) != 300.0
    ):
        raise ValueError("Compact NativeOrder execution changed")
    return config_path, root, config, inputs["parent_registration"], inputs["parent_report"]


def build_cohort(
    config_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path, _root, _config, parent_registration, parent_report = load_registration(
        config_path
    )
    parent_metadata, cohort = build_nativeorder_cohort(parent_registration)
    return {
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "parent_registration_sha256": sha256_file(parent_registration),
        "parent_report_sha256": sha256_file(parent_report),
        "parent_cohort_fingerprint": parent_metadata["cohort_fingerprint"],
        "cohort_fingerprint": _fingerprint(
            [
                {
                    "state_fingerprint": row["state_fingerprint"],
                    "state_blob_sha256": row["state_blob_sha256"],
                    "selected_candidate": row["selected_candidate"],
                }
                for row in cohort
            ]
        ),
    }, cohort


def semantic_compact_plan(
    state: dict[str, Any], base_agents: list[int]
) -> dict[str, Any]:
    base = set(map(int, base_agents))
    current_core = {
        int(agent)
        for edge in state.get("conflict_edges", ())
        for agent in edge
        if int(agent) in base
    }
    if len(current_core) < 2:
        return {
            "eligible": False,
            "rejection_reason": "fewer_than_two_current_conflict_agents",
            "base_agents": sorted(base),
            "compact_agents": sorted(base),
            "current_conflict_core": sorted(current_core),
            "temporal_corridor_support": [],
            "temporal_corridor_evidence": {},
            "removed_agents": [],
            "base_size": len(base),
            "actual_size": len(base),
        }
    analysis = analyze_state(state)
    temporal = _temporal_corridor_evidence(state, analysis, current_core)
    retained_temporal = sorted(base & set(temporal))
    compact = current_core | set(retained_temporal)
    removed = sorted(base - compact)
    eligible = bool(removed)
    return {
        "eligible": eligible,
        "rejection_reason": None if eligible else "no_unsupported_agent_to_remove",
        "base_agents": sorted(base),
        "compact_agents": sorted(compact) if eligible else sorted(base),
        "current_conflict_core": sorted(current_core),
        "temporal_corridor_support": retained_temporal,
        "temporal_corridor_evidence": {
            str(agent): _serialize_evidence(temporal[agent])
            for agent in retained_temporal
        },
        "removed_agents": removed,
        "base_size": len(base),
        "actual_size": len(compact) if eligible else len(base),
    }


def _run_policy(
    *,
    replay: dict[str, Any],
    state: dict[str, Any],
    before_repair: str,
    before_conflicts: int,
    restore_seed: int,
    agents: list[int],
    first_seed: int,
    retry_seed: int,
    policy_id: str,
    transaction_budget_seconds: float,
    compact_plan: dict[str, Any],
) -> dict[str, Any]:
    environment, restored = restore_repair_state(replay, state, seed=restore_seed)
    if repair_structure_fingerprint(restored) != before_repair:
        raise RuntimeError("Compact NativeOrder branch restore changed")
    attempts: list[dict[str, Any]] = []
    started = time.perf_counter()

    def remaining() -> float:
        return max(0.0, float(transaction_budget_seconds) - (time.perf_counter() - started))

    first, final_state = _timed_attempt(
        environment=environment,
        agents=agents,
        repair_order=None,
        pp_seed=first_seed,
        before_repair=before_repair,
        before_conflicts=before_conflicts,
        attempt_index=0,
        reason=("compact_native_order" if policy_id in COMPACT_POLICIES else "full_native_order"),
        pp_time_limit_seconds=remaining(),
    )
    attempts.append(first)
    termination = "committed_first_success" if first["replan_success"] else "single_attempt_policy"
    if not first["replan_success"]:
        failure = str(first["failure_reason"])
        if failure == "time_limit":
            termination = "stopped_time_limit"
        elif failure != "conflict_bound_exceeded":
            termination = f"stopped_{failure}"
        elif policy_id in RETRY_POLICIES:
            second, final_state = _timed_attempt(
                environment=environment,
                agents=agents,
                repair_order=None,
                pp_seed=retry_seed,
                before_repair=before_repair,
                before_conflicts=before_conflicts,
                attempt_index=1,
                reason=(
                    "compact_fresh_native_retry"
                    if policy_id == COMPACT_RETRY
                    else "full_fresh_native_retry"
                ),
                pp_time_limit_seconds=remaining(),
            )
            attempts.append(second)
            termination = (
                "committed_retry_success"
                if second["replan_success"]
                else f"retry_failed_{second['failure_reason']}"
            )
    final_repair = repair_structure_fingerprint(final_state)
    conflicts_after = int(final_state["num_of_colliding_pairs"])
    return {
        "schema": POLICY_SCHEMA,
        "policy_id": policy_id,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "termination": termination,
        "replan_success": bool(attempts[-1]["replan_success"]),
        "strict_conflict_reduction": conflicts_after < before_conflicts,
        "normalized_conflict_reduction": (
            before_conflicts - conflicts_after
        ) / max(1, before_conflicts),
        "conflicts_after": conflicts_after,
        "after_repair_fingerprint": final_repair,
        "returned_unchanged": final_repair == before_repair,
        "total_attempt_wall_seconds": sum(
            float(row["attempt_wall_seconds"]) for row in attempts
        ),
        "total_transaction_wall_seconds": time.perf_counter() - started,
        "transaction_pp_wall_budget_seconds": float(transaction_budget_seconds),
        "actual_agents": list(map(int, agents)),
        "actual_size": len(agents),
        "compact_eligible": bool(compact_plan["eligible"]),
        "removed_agent_count": len(compact_plan["removed_agents"]),
        "future_trajectory_stored": False,
    }


def _collect_job(job: dict[str, Any]) -> dict[str, Any]:
    state_record = dict(job["state_record"])
    state_key = str(state_record["state_fingerprint"])
    trial_index = int(job["trial_index"])
    policy_id = str(job["policy_id"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if output_path.is_file():
        existing = _read_json(output_path)
        if (
            existing.get("schema") == TRIAL_SCHEMA
            and existing.get("run_fingerprint") == run_fingerprint
            and existing.get("state_fingerprint") == state_key
            and int(existing.get("trial_index", -1)) == trial_index
            and existing.get("policy", {}).get("policy_id") == policy_id
        ):
            return {
                "job_id": str(job["job_id"]),
                "status": "resumed",
                "outcome_count": 1,
                "error_count": 0,
                "worker_pid": os.getpid(),
            }
        raise ValueError("invalid completed compact NativeOrder trial")
    state, replay, before_repair, before_conflicts, restore_seed = _load_restored_source(
        state_record
    )
    base_agents = sorted(map(int, state_record["selected_candidate"]["agents"]))
    compact_plan = semantic_compact_plan(state, base_agents)
    agents = (
        list(compact_plan["compact_agents"])
        if policy_id in COMPACT_POLICIES
        else base_agents
    )
    first_seed = repairability_pp_seed(before_repair, trial_index)
    second_seed = retry_pp_seed(before_repair, trial_index)
    policy = _run_policy(
        replay=replay,
        state=state,
        before_repair=before_repair,
        before_conflicts=before_conflicts,
        restore_seed=restore_seed,
        agents=agents,
        first_seed=first_seed,
        retry_seed=second_seed,
        policy_id=policy_id,
        transaction_budget_seconds=float(job["transaction_budget_seconds"]),
        compact_plan=compact_plan,
    )
    _write_json(
        output_path,
        {
            "schema": TRIAL_SCHEMA,
            "complete": True,
            "phase": str(job["phase"]),
            "run_fingerprint": run_fingerprint,
            "state_fingerprint": state_key,
            "state_blob_sha256": str(state_record["state_blob_sha256"]),
            "map_id": str(state_record["map_id"]),
            "task_id": str(state_record["task_id"]),
            "solver_seed": int(state_record["solver_seed"]),
            "trial_index": trial_index,
            "first_pp_seed": first_seed,
            "retry_pp_seed": second_seed,
            "before_conflicts": before_conflicts,
            "before_repair_fingerprint": before_repair,
            "agent_count": len(state["agents"]),
            "base_agents": base_agents,
            "compact_plan": compact_plan,
            "policy": policy,
            "future_trajectory_stored": False,
        },
    )
    return {
        "job_id": str(job["job_id"]),
        "status": "ok",
        "outcome_count": 1,
        "error_count": 0,
        "worker_pid": os.getpid(),
    }


def _failed_job(job: dict[str, Any], status: str, message: str) -> dict[str, Any]:
    return {
        "job_id": str(job["job_id"]),
        "status": status,
        "outcome_count": 0,
        "error_count": 1,
        "error": message,
    }


def collect_phase(
    *, config_path: str | Path, output: str | Path, phase: str, resume: bool = False
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown compact NativeOrder phase: {phase}")
    metadata, cohort = build_cohort(config_path)
    _path, root, config, _parent, _report = load_registration(config_path)
    trial_indices = list(map(int, config["cohort"][f"{phase}_trial_indices"]))
    output = Path(output).resolve()
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=(),
    )
    run_identity = {
        "schema": RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        **metadata,
        "state_fingerprints": [str(row["state_fingerprint"]) for row in cohort],
        "policies": list(POLICIES),
        "trial_indices": trial_indices,
        "worker_count": int(config["execution"]["worker_count"]),
        "transaction_pp_wall_budget_seconds": float(
            config["execution"]["transaction_pp_wall_budget_seconds"]
        ),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(run_identity)
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("compact NativeOrder output belongs to another run")
        if not resume:
            raise ValueError("compact NativeOrder output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**run_identity, "run_fingerprint": run_fingerprint})
    jobs = []
    for state_record in cohort:
        state_key = str(state_record["state_fingerprint"])
        for trial_index in trial_indices:
            for policy_id in POLICIES:
                jobs.append(
                    {
                        "job_id": f"{state_key}:{trial_index:02d}:{policy_id}",
                        "phase": phase,
                        "state_record": state_record,
                        "trial_index": trial_index,
                        "policy_id": policy_id,
                        "output_path": str(
                            output
                            / "trials"
                            / state_key
                            / f"trial_{trial_index:02d}__{policy_id}.json"
                        ),
                        "run_fingerprint": run_fingerprint,
                        "transaction_budget_seconds": float(
                            config["execution"]["transaction_pp_wall_budget_seconds"]
                        ),
                    }
                )
    results = _run_jobs(
        _collect_job,
        jobs,
        int(config["execution"]["worker_count"]),
        phase=f"compact-nativeorder-{phase}",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_policy_job_timeout_seconds"]),
        failure_result=_failed_job,
        stop_on_failure=True,
    )
    files = _trial_files(output)
    errors = [row for row in results if row["status"] in {"error", "timeout"}]
    pids = {
        int(row["worker_pid"])
        for row in results
        if row.get("worker_pid") is not None
    }
    status = {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "status": (
            "complete"
            if len(files) == len(jobs) and not errors
            else "failed" if errors else "incomplete"
        ),
        "run_fingerprint": run_fingerprint,
        "completed_jobs": len(files),
        "required_jobs": len(jobs),
        "error_jobs": sum(row["status"] == "error" for row in errors),
        "timeout_jobs": sum(row["status"] == "timeout" for row in errors),
        "active_jobs": [],
        "worker_limit": int(config["execution"]["worker_count"]),
        "worker_process_count": len(pids),
        "errors": errors,
    }
    _write_json(output / "collection_status.json", status)
    return status


def _summary(rows: list[dict[str, Any]], policy_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row["policy_id"] == policy_id]
    if not selected:
        return {"row_count": 0}
    return {
        "row_count": len(selected),
        "replan_success_rate": mean(float(row["replan_success"]) for row in selected),
        "strict_conflict_reduction_rate": mean(
            float(row["strict_conflict_reduction"]) for row in selected
        ),
        "returned_unchanged_rate": mean(
            float(row["returned_unchanged"]) for row in selected
        ),
        "mean_normalized_conflict_reduction": mean(
            float(row["normalized_conflict_reduction"]) for row in selected
        ),
        "mean_attempt_count": mean(float(row["attempt_count"]) for row in selected),
        "mean_actual_size": mean(float(row["actual_size"]) for row in selected),
        "mean_removed_agent_count": mean(
            float(row["removed_agent_count"]) for row in selected
        ),
        "mean_total_attempt_wall_seconds": mean(
            float(row["total_attempt_wall_seconds"]) for row in selected
        ),
    }


def _paired_bootstrap(
    rows: list[dict[str, Any]], left: str, right: str, key: str, replicates: int
) -> dict[str, float]:
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["policy_id"] in {left, right}:
            values[str(row["state_fingerprint"])][str(row["policy_id"])].append(
                float(row[key])
            )
    differences = [
        mean(group[left]) - mean(group[right])
        for group in values.values()
        if group[left] and group[right]
    ]
    if not differences:
        raise ValueError("compact NativeOrder bootstrap has no paired states")
    rng = random.Random(20260814)
    samples = sorted(
        mean(rng.choice(differences) for _ in differences)
        for _ in range(int(replicates))
    )
    return {
        "difference": mean(differences),
        "lower_95": samples[int(0.025 * (len(samples) - 1))],
        "upper_95": samples[int(0.975 * (len(samples) - 1))],
        "state_cluster_count": len(differences),
        "replicates": int(replicates),
    }


def analyze(
    config_path: str | Path,
    initial_output: str | Path,
    extension_output: str | Path | None = None,
) -> dict[str, Any]:
    metadata, cohort = build_cohort(config_path)
    _path, _root, config, _parent, _parent_report = load_registration(config_path)
    outputs = [Path(initial_output).resolve()]
    phase = "initial"
    if extension_output is not None:
        outputs.append(Path(extension_output).resolve())
        phase = "extended"
    statuses = [_read_json(output / "collection_status.json") for output in outputs]
    if any(row.get("status") != "complete" for row in statuses):
        raise ValueError("compact NativeOrder collection is incomplete")
    artifacts = [
        _read_json(path) for output in outputs for path in _trial_files(output)
    ]
    trial_indices = list(map(int, config["cohort"]["initial_trial_indices"]))
    if extension_output is not None:
        trial_indices += list(map(int, config["cohort"]["extension_trial_indices"]))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    rows = []
    for artifact in artifacts:
        key = (str(artifact["state_fingerprint"]), int(artifact["trial_index"]))
        grouped[key].append(artifact)
        rows.append(
            {
                **dict(artifact["policy"]),
                "state_fingerprint": key[0],
                "trial_index": key[1],
                "map_id": str(artifact["map_id"]),
                "base_size": len(artifact["base_agents"]),
            }
        )
    expected_pairs = len(cohort) * len(trial_indices)
    compact_plans = {
        str(row["state_fingerprint"]): dict(row["compact_plan"])
        for row in artifacts
    }
    eligible_states = sorted(
        key for key, plan in compact_plans.items() if bool(plan["eligible"])
    )
    eligible_set = set(eligible_states)
    eligible_rows = [row for row in rows if row["state_fingerprint"] in eligible_set]
    integrity = {
        "state_trial_count": len(grouped) == expected_pairs,
        "policy_artifact_count": len(artifacts) == expected_pairs * len(POLICIES),
        "policy_coverage": all(
            {row["policy"]["policy_id"] for row in group} == set(POLICIES)
            for group in grouped.values()
        ),
        "state_count": len({key[0] for key in grouped}) == 45,
        "trial_coverage": {key[1] for key in grouped} == set(trial_indices),
        "same_first_seed": all(
            len({int(row["first_pp_seed"]) for row in group}) == 1
            for group in grouped.values()
        ),
        "full_first_attempt_parity": all(
            len(
                {
                    row["policy"]["attempts"][0]["attempt_signature"]
                    for row in group
                    if row["policy"]["policy_id"] in {FULL_SINGLE, FULL_RETRY}
                }
            )
            == 1
            for group in grouped.values()
        ),
        "compact_first_attempt_parity": all(
            len(
                {
                    row["policy"]["attempts"][0]["attempt_signature"]
                    for row in group
                    if row["policy"]["policy_id"] in {COMPACT_SINGLE, COMPACT_RETRY}
                }
            )
            == 1
            for group in grouped.values()
        ),
        "compact_is_subset": all(
            set(row["compact_plan"]["compact_agents"]).issubset(row["base_agents"])
            for row in artifacts
        ),
        "no_fixed_size_target": all(
            row["compact_plan"].get("target_size") is None for row in artifacts
        ),
        "native_order_only": all(
            attempt["explicit_repair_order_requested"] is False
            for row in artifacts
            for attempt in row["policy"]["attempts"]
        ),
        "attempt_cap": all(row["policy"]["attempt_count"] <= 2 for row in artifacts),
        "no_time_limit_retry": all(
            len(row["policy"]["attempts"]) == 1
            for row in artifacts
            if row["policy"]["attempts"][0]["failure_reason"] == "time_limit"
        ),
        "no_future_trajectory": all(
            row.get("future_trajectory_stored") is False
            and row["policy"].get("future_trajectory_stored") is False
            for row in artifacts
        ),
        "zero_errors_and_timeouts": all(
            int(status["error_jobs"]) == 0 and int(status["timeout_jobs"]) == 0
            for status in statuses
        ),
    }
    summaries = {policy: _summary(rows, policy) for policy in POLICIES}
    eligible_summaries = {
        policy: _summary(eligible_rows, policy) for policy in POLICIES
    }
    eligible_by_map = {
        policy: {
            map_id: _summary(
                [row for row in eligible_rows if row["map_id"] == map_id], policy
            )
            for map_id in sorted({row["map_id"] for row in eligible_rows})
        }
        for policy in POLICIES
    }
    compact = eligible_summaries[COMPACT_RETRY]
    full = eligible_summaries[FULL_RETRY]
    maximum_map_worsening = float(
        config["initial_gate"]["maximum_per_map_returned_unchanged_increase"]
    )
    initial_gate = {
        "minimum_eligible_state_count": len(eligible_states)
        >= int(config["initial_gate"]["minimum_eligible_state_count"]),
        "compact_mean_size_lower": float(compact["mean_actual_size"])
        < float(full["mean_actual_size"]),
        "compact_returned_unchanged_lower": float(compact["returned_unchanged_rate"])
        < float(full["returned_unchanged_rate"]),
        "compact_success_not_lower": float(compact["replan_success_rate"])
        >= float(full["replan_success_rate"]),
        "compact_strict_reduction_not_lower": float(
            compact["strict_conflict_reduction_rate"]
        )
        >= float(full["strict_conflict_reduction_rate"]),
        "compact_wall_lower": float(compact["mean_total_attempt_wall_seconds"])
        < float(full["mean_total_attempt_wall_seconds"]),
        "per_map_not_worse_by_more_than_five_points": all(
            float(eligible_by_map[COMPACT_RETRY][map_id]["returned_unchanged_rate"])
            - float(eligible_by_map[FULL_RETRY][map_id]["returned_unchanged_rate"])
            <= maximum_map_worsening
            for map_id in eligible_by_map[COMPACT_RETRY]
        ),
    }
    extension_qualified = all(initial_gate.values())
    bootstraps: dict[str, Any] = {}
    final_gate: dict[str, bool] = {}
    if phase == "extended":
        replicates = int(config["final_gate"]["bootstrap_replicates"])
        bootstraps = {
            "returned_unchanged_compact_minus_full": _paired_bootstrap(
                eligible_rows,
                COMPACT_RETRY,
                FULL_RETRY,
                "returned_unchanged",
                replicates,
            ),
            "wall_compact_minus_full": _paired_bootstrap(
                eligible_rows,
                COMPACT_RETRY,
                FULL_RETRY,
                "total_attempt_wall_seconds",
                replicates,
            ),
        }
        final_gate = {
            "platform_continuation_authorized": all(integrity.values())
            and float(
                bootstraps["returned_unchanged_compact_minus_full"]["upper_95"]
            )
            < 0.0
            and float(bootstraps["wall_compact_minus_full"]["upper_95"]) < 0.0
            and float(compact["replan_success_rate"])
            >= float(full["replan_success_rate"])
            and float(compact["strict_conflict_reduction_rate"])
            >= float(full["strict_conflict_reduction_rate"])
            and initial_gate["per_map_not_worse_by_more_than_five_points"],
        }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        **metadata,
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "state_count": 45,
        "state_trial_count": len(grouped),
        "policy_artifact_count": len(artifacts),
        "eligible_state_count": len(eligible_states),
        "eligible_state_fingerprints": eligible_states,
        "policy_summaries": summaries,
        "eligible_policy_summaries": eligible_summaries,
        "eligible_by_map": eligible_by_map,
        "initial_gate": initial_gate,
        "extension_qualified": extension_qualified,
        "paired_state_cluster_bootstrap": bootstraps,
        "final_gate": final_gate,
        "next_step": (
            "uniform_extension"
            if phase == "initial" and extension_qualified
            else "preregister_bounded_platform_continuation"
            if final_gate.get("platform_continuation_authorized") is True
            else "stop_compact_branch"
        ),
        "claim_boundary": dict(config["claim_boundary"]),
    }
    destination = outputs[-1] / f"compact_nativeorder_{phase}_report.json"
    _write_json(destination, report)
    return report
