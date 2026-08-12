from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from experiments._common import mean, producer_identity, registered_input, sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
)
from experiments.stride_collection import _validate_native_repair
from experiments.stride_repairability_causal_audit import (
    _diagnostic_action,
    _load_restored_source,
    _validate_pp_diagnostic,
    build_causal_cohort,
    conflict_priority_order,
    external_blocker_order,
)
from experiments.stride_repairability_collection import repairability_pp_seed
from experiments.stride_repairdependency_predictability import (
    _contained as _registered_collection_path,
)
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


CONFIG_SCHEMA = "lns2.stride.transactionalrepair_registration.v1"
RUN_SCHEMA = "lns2.stride.transactionalrepair_run.v1"
TRIAL_SCHEMA = "lns2.stride.transactionalrepair_trial.v1"
POLICY_SCHEMA = "lns2.stride.transactionalrepair_policy_result.v1"
ATTEMPT_SCHEMA = "lns2.stride.transactionalrepair_attempt.v1"
STATUS_SCHEMA = "lns2.stride.transactionalrepair_collection_status.v1"
REPORT_SCHEMA = "lns2.stride.transactionalrepair_report.v1"
EXECUTION_SCHEMA = "lns2.stride.transactionalrepair_execution.v1"
EXPERIMENT_ID = "stride-transactionalrepair-v1"
POLICIES = (
    "selected_single_attempt",
    "transactional_set_retry",
    "transactional_set_order_retry",
)
PRIMARY_POLICY = "transactional_set_order_retry"
BASELINE_POLICY = "selected_single_attempt"
SET_POLICY = "transactional_set_retry"
PRODUCER_FILES = (
    "experiments/stride_transactionalrepair.py",
    "scripts/run_stride_transactionalrepair.py",
    "experiments/stride_repairability_causal_audit.py",
    "experiments/stride_repairdependency_predictability.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], Path]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("TransactionalRepair registration changed")
    inputs = {
        name: registered_input(root, row, label=f"transactional repair {name}")
        for name, row in dict(config["inputs"]).items()
    }
    collection = _registered_collection_path(
        root,
        str(config["causal_collection"]["path"]),
        label="causal collection",
    )
    if tuple(config["policies"]["ids"]) != POLICIES:
        raise ValueError("TransactionalRepair policies changed")
    if str(config["policies"]["primary"]) != PRIMARY_POLICY:
        raise ValueError("TransactionalRepair primary policy changed")
    if int(config["execution"]["worker_count"]) != 16:
        raise ValueError("TransactionalRepair worker count changed")
    if int(config["policies"]["maximum_added_external_blockers"]) != 8:
        raise ValueError("TransactionalRepair blocker cap changed")
    if int(config["policies"]["maximum_attempts"][PRIMARY_POLICY]) != 3:
        raise ValueError("TransactionalRepair attempt cap changed")
    return config_path, root, config, inputs, collection


def build_transactional_cohort(
    config_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path, _root, config, inputs, collection = load_registration(config_path)
    causal_report = _read_json(inputs["causal_report"])
    predictability_report = _read_json(inputs["predictability_report"])
    if (
        causal_report.get("integrity_passed") is not True
        or predictability_report.get("integrity_passed") is not True
        or predictability_report.get("predictability_passed") is not False
    ):
        raise ValueError("TransactionalRepair source evidence changed")
    metadata, cohort = build_causal_cohort(inputs["causal_registration"])
    manifest = _read_jsonl(inputs["causal_state_manifest"])
    manifest_by_state = {str(row["state_fingerprint"]): row for row in manifest}
    cohort_by_state = {str(row["state_fingerprint"]): row for row in cohort}
    required = dict(config["cohort"])
    if (
        len(manifest_by_state) != int(required["required_state_count"])
        or len(cohort_by_state) != int(required["required_state_count"])
        or set(manifest_by_state) != set(cohort_by_state)
        or len({str(row["map_id"]) for row in cohort})
        != int(required["required_map_count"])
    ):
        raise ValueError("TransactionalRepair cohort changed")
    for state_key, row in manifest_by_state.items():
        path = collection / "states" / str(row["file_name"])
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError("TransactionalRepair causal state artifact changed")
    effects = _read_jsonl(inputs["causal_state_effects"])
    root_by_state = {
        str(row["state_fingerprint"]): str(row["root_cause"]) for row in effects
    }
    if set(root_by_state) != set(cohort_by_state):
        raise ValueError("TransactionalRepair root classification changed")
    frozen = [
        {**row, "discovery_root_cause": root_by_state[str(row["state_fingerprint"])]}
        for row in cohort
    ]
    return {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "source_cohort_fingerprint": str(metadata["cohort_fingerprint"]),
        "cohort_fingerprint": _fingerprint(
            [
                {
                    "state_fingerprint": row["state_fingerprint"],
                    "state_blob_sha256": row["state_blob_sha256"],
                    "selected_candidate": row["selected_candidate"],
                    "discovery_root_cause": row["discovery_root_cause"],
                }
                for row in frozen
            ]
        ),
    }, frozen


def retry_plan(
    *,
    policy_id: str,
    state: dict[str, Any],
    current_agents: list[int],
    applied_order: list[int],
    diagnostic_agents: list[dict[str, Any]],
    added_agents: list[int],
    priority_used: bool,
    maximum_added_agents: int,
) -> tuple[list[int], list[int], list[int], bool, str] | None:
    if policy_id == BASELINE_POLICY:
        return None
    remaining = max(0, int(maximum_added_agents) - len(added_agents))
    blockers = external_blocker_order(
        diagnostic_agents, current_agents, maximum=remaining
    )
    blockers = [agent for agent in blockers if agent not in set(current_agents)]
    if blockers:
        next_agents = list(current_agents) + blockers
        return (
            next_agents,
            list(applied_order) + blockers,
            list(added_agents) + blockers,
            priority_used,
            "append_observed_external_blockers",
        )
    if policy_id == SET_POLICY:
        return None
    priority = conflict_priority_order(state, current_agents)
    if not priority_used and priority != list(applied_order):
        return (
            list(current_agents),
            priority,
            list(added_agents),
            True,
            "conflict_priority_reorder",
        )
    return None


def _attempt_signature(row: dict[str, Any]) -> str:
    return _fingerprint(
        {
            key: row[key]
            for key in (
                "agents",
                "repair_order",
                "replan_success",
                "failure_reason",
                "attempted_agent_count",
                "inserted_agent_count",
                "failed_agent",
                "failed_order_index",
                "rolled_back",
                "external_blockers",
                "conflicts_after",
                "after_repair_fingerprint",
            )
        }
    )


def _run_attempt(
    *,
    environment: Any,
    agents: list[int],
    repair_order: list[int] | None,
    pp_seed: int,
    before_repair: str,
    before_conflicts: int,
    attempt_index: int,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _plain(
        environment.step(
            _diagnostic_action(agents, pp_seed, repair_order=repair_order)
        )
    )
    after, metrics = _validate_native_repair(
        result, expected_agents=agents, expected_seed=pp_seed
    )
    diagnostic = _validate_pp_diagnostic(metrics)
    after_repair = repair_structure_fingerprint(after)
    conflicts_after = int(after["num_of_colliding_pairs"])
    blockers = external_blocker_order(
        list(diagnostic["agents"]), agents, maximum=len(after["agents"])
    )
    row = {
        "schema": ATTEMPT_SCHEMA,
        "attempt_index": int(attempt_index),
        "attempt_reason": reason,
        "agents": list(map(int, agents)),
        "repair_order": list(map(int, metrics["repair_order"])),
        "replan_success": bool(metrics["replan_success"]),
        "failure_reason": str(diagnostic["failure_reason"]),
        "attempted_agent_count": int(diagnostic["attempted_agent_count"]),
        "inserted_agent_count": int(diagnostic["inserted_agent_count"]),
        "failed_agent": int(diagnostic["failed_agent"]),
        "failed_order_index": int(diagnostic["failed_order_index"]),
        "rolled_back": bool(diagnostic["rolled_back"]),
        "external_blockers": blockers,
        "conflicts_after": conflicts_after,
        "after_repair_fingerprint": after_repair,
        "repair_outcome": classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics["replan_success"]),
            conflicts_before=before_conflicts,
            conflicts_after=conflicts_after,
            feasible=bool(after.get("feasible")),
        ),
    }
    row["attempt_signature"] = _attempt_signature(row)
    if not row["replan_success"] and (
        not row["rolled_back"]
        or after_repair != before_repair
        or conflicts_after != before_conflicts
    ):
        raise RuntimeError("TransactionalRepair failed attempt did not roll back")
    return row, after


def _run_policy(
    *,
    replay: dict[str, Any],
    state: dict[str, Any],
    before_repair: str,
    before_conflicts: int,
    restore_seed: int,
    base_agents: list[int],
    pp_seed: int,
    policy_id: str,
    maximum_added_agents: int,
    maximum_attempts: int,
) -> dict[str, Any]:
    environment, restored = restore_repair_state(replay, state, seed=restore_seed)
    if repair_structure_fingerprint(restored) != before_repair:
        raise RuntimeError("TransactionalRepair branch restore changed")
    current_agents = list(base_agents)
    repair_order: list[int] | None = None
    added_agents: list[int] = []
    priority_used = False
    reason = "selected_native_order"
    attempts: list[dict[str, Any]] = []
    final_state = restored
    termination = "attempt_cap_reached"
    for attempt_index in range(int(maximum_attempts)):
        attempt, final_state = _run_attempt(
            environment=environment,
            agents=current_agents,
            repair_order=repair_order,
            pp_seed=pp_seed,
            before_repair=before_repair,
            before_conflicts=before_conflicts,
            attempt_index=attempt_index,
            reason=reason,
        )
        attempts.append(attempt)
        if bool(attempt["replan_success"]):
            termination = "committed_success"
            break
        if str(attempt["failure_reason"]) != "conflict_bound_exceeded":
            termination = f"stopped_{attempt['failure_reason']}"
            break
        plan = retry_plan(
            policy_id=policy_id,
            state=state,
            current_agents=current_agents,
            applied_order=list(attempt["repair_order"]),
            diagnostic_agents=[
                {
                    "order_index": index,
                    "external_blocker_agents": [agent],
                }
                for index, agent in enumerate(attempt["external_blockers"])
            ],
            added_agents=added_agents,
            priority_used=priority_used,
            maximum_added_agents=maximum_added_agents,
        )
        if plan is None:
            termination = "no_distinct_bounded_retry"
            break
        current_agents, repair_order, added_agents, priority_used, reason = plan
    final_repair = repair_structure_fingerprint(final_state)
    conflicts_after = int(final_state["num_of_colliding_pairs"])
    return {
        "schema": POLICY_SCHEMA,
        "policy_id": policy_id,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "termination": termination,
        "added_agents": added_agents,
        "added_agent_count": len(added_agents),
        "final_agent_count": len(attempts[-1]["agents"]),
        "replan_success": bool(attempts[-1]["replan_success"]),
        "strict_conflict_reduction": conflicts_after < before_conflicts,
        "normalized_conflict_reduction": (
            before_conflicts - conflicts_after
        )
        / max(1, before_conflicts),
        "conflicts_after": conflicts_after,
        "after_repair_fingerprint": final_repair,
        "returned_unchanged": final_repair == before_repair,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
    }


def _collect_job(job: dict[str, Any]) -> dict[str, Any]:
    state_record = dict(job["state_record"])
    state_key = str(state_record["state_fingerprint"])
    trial_index = int(job["trial_index"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if output_path.is_file():
        existing = _read_json(output_path)
        if (
            existing.get("schema") == TRIAL_SCHEMA
            and existing.get("run_fingerprint") == run_fingerprint
            and existing.get("state_fingerprint") == state_key
            and int(existing.get("trial_index", -1)) == trial_index
        ):
            return {
                "job_id": str(job["job_id"]),
                "status": "resumed",
                "state_count": 0,
                "outcome_count": len(existing["policies"]),
                "error_count": 0,
                "worker_pid": os.getpid(),
            }
        raise ValueError("invalid completed TransactionalRepair trial")
    state, replay, before_repair, before_conflicts, restore_seed = (
        _load_restored_source(state_record)
    )
    base_agents = list(map(int, state_record["selected_candidate"]["agents"]))
    pp_seed = repairability_pp_seed(before_repair, trial_index)
    policies = []
    for policy_id in POLICIES:
        policies.append(
            _run_policy(
                replay=replay,
                state=state,
                before_repair=before_repair,
                before_conflicts=before_conflicts,
                restore_seed=restore_seed,
                base_agents=base_agents,
                pp_seed=pp_seed,
                policy_id=policy_id,
                maximum_added_agents=int(job["maximum_added_agents"]),
                maximum_attempts=int(job["maximum_attempts"][policy_id]),
            )
        )
    initial_signatures = {
        str(row["attempts"][0]["attempt_signature"]) for row in policies
    }
    if len(initial_signatures) != 1:
        raise RuntimeError("TransactionalRepair paired initial attempt changed")
    payload = {
        "schema": TRIAL_SCHEMA,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "state_fingerprint": state_key,
        "state_blob_sha256": str(state_record["state_blob_sha256"]),
        "map_id": str(state_record["map_id"]),
        "task_id": str(state_record["task_id"]),
        "solver_seed": int(state_record["solver_seed"]),
        "discovery_root_cause": str(state_record["discovery_root_cause"]),
        "trial_index": trial_index,
        "pp_seed": pp_seed,
        "before_conflicts": before_conflicts,
        "before_repair_fingerprint": before_repair,
        "base_agents": base_agents,
        "agent_count": len(state["agents"]),
        "initial_attempt_parity": True,
        "policies": policies,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)
    return {
        "job_id": str(job["job_id"]),
        "status": "ok",
        "state_count": 0,
        "outcome_count": len(policies),
        "error_count": 0,
        "worker_pid": os.getpid(),
    }


def _failed_job(job: dict[str, Any], status: str, message: str) -> dict[str, Any]:
    return {
        "job_id": str(job["job_id"]),
        "status": status,
        "state_count": 0,
        "outcome_count": 0,
        "error_count": 1,
        "error": message,
    }


def _trial_files(output: Path) -> list[Path]:
    return sorted((output / "trials").glob("*/*.json"))


def collect_transactional_audit(
    *, config_path: str | Path, output: str | Path, resume: bool = False
) -> dict[str, Any]:
    metadata, cohort = build_transactional_cohort(config_path)
    _config_path, root, config, _inputs, _collection = load_registration(config_path)
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
        **metadata,
        "policies": list(POLICIES),
        "trial_indices": list(map(int, config["cohort"]["trial_indices"])),
        "worker_count": int(config["execution"]["worker_count"]),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(run_identity)
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("TransactionalRepair output belongs to another run")
        if not resume:
            raise ValueError("TransactionalRepair output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**run_identity, "run_fingerprint": run_fingerprint})
    maximum_attempts = {
        str(key): int(value)
        for key, value in dict(config["policies"]["maximum_attempts"]).items()
    }
    jobs = []
    for state_record in cohort:
        state_key = str(state_record["state_fingerprint"])
        for trial_index in map(int, config["cohort"]["trial_indices"]):
            jobs.append(
                {
                    "job_id": f"{state_key}:{trial_index:02d}",
                    "state_record": state_record,
                    "trial_index": trial_index,
                    "output_path": str(
                        output / "trials" / state_key / f"trial_{trial_index:02d}.json"
                    ),
                    "run_fingerprint": run_fingerprint,
                    "maximum_added_agents": int(
                        config["policies"]["maximum_added_external_blockers"]
                    ),
                    "maximum_attempts": maximum_attempts,
                }
            )
    results = _run_jobs(
        _collect_job,
        jobs,
        int(config["execution"]["worker_count"]),
        phase="transactionalrepair-state-trial",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_state_trial_timeout_seconds"]),
        failure_result=_failed_job,
    )
    files = _trial_files(output)
    expected = len(cohort) * len(config["cohort"]["trial_indices"])
    errors = [row for row in results if row["status"] in {"error", "timeout"}]
    execution_pids = {
        int(row["worker_pid"]) for row in results if row.get("worker_pid") is not None
    }
    final_pids = {
        int(row["worker_pid"])
        for row in results[-32:]
        if row.get("worker_pid") is not None
    }
    status = {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if len(files) == expected and not errors else "incomplete",
        "run_fingerprint": run_fingerprint,
        "completed_state_trial_count": len(files),
        "required_state_trial_count": expected,
        "completed_policy_row_count": len(files) * len(POLICIES),
        "required_policy_row_count": expected * len(POLICIES),
        "error_job_count": sum(row["status"] == "error" for row in errors),
        "timeout_job_count": sum(row["status"] == "timeout" for row in errors),
        "worker_process_count": len(execution_pids),
        "final_32_completed_job_worker_process_count": len(final_pids),
        "worker_limit": int(config["execution"]["worker_count"]),
        "tail_parallelism_observed": len(final_pids) > 1,
        "errors": errors,
    }
    _write_json(output / "collection_status.json", status)
    return status


def _summary(rows: list[dict[str, Any]], policy_id: str) -> dict[str, Any]:
    selected = [row for row in rows if row["policy_id"] == policy_id]
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
        "mean_retry_count": mean(float(row["retry_count"]) for row in selected),
        "mean_added_agent_count": mean(
            float(row["added_agent_count"]) for row in selected
        ),
        "mean_added_agent_ratio": mean(
            float(row["added_agent_count"]) / max(1, int(row["base_agent_count"]))
            for row in selected
        ),
        "maximum_total_neighborhood_fraction": max(
            (
                float(row["final_agent_count"]) / max(1, int(row["agent_count"]))
                for row in selected
            ),
            default=0.0,
        ),
    }


def _group_summary(
    rows: list[dict[str, Any]], policy_id: str, key: str
) -> dict[str, dict[str, Any]]:
    return {
        value: _summary([row for row in rows if str(row[key]) == value], policy_id)
        for value in sorted({str(row[key]) for row in rows})
    }


def analyze_transactional_audit(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    metadata, cohort = build_transactional_cohort(config_path)
    _config_path, root, config, _inputs, _collection = load_registration(config_path)
    output = Path(output).resolve()
    status = _read_json(output / "collection_status.json")
    if status.get("status") != "complete":
        raise ValueError("TransactionalRepair collection is incomplete")
    files = _trial_files(output)
    trials = [_read_json(path) for path in files]
    expected_trials = len(cohort) * len(config["cohort"]["trial_indices"])
    rows: list[dict[str, Any]] = []
    for trial in trials:
        seed_half = "first" if int(trial["trial_index"]) < 8 else "second"
        for policy in trial["policies"]:
            rows.append(
                {
                    **policy,
                    "state_fingerprint": str(trial["state_fingerprint"]),
                    "map_id": str(trial["map_id"]),
                    "trial_index": int(trial["trial_index"]),
                    "seed_half": seed_half,
                    "discovery_root_cause": str(trial["discovery_root_cause"]),
                    "before_conflicts": int(trial["before_conflicts"]),
                    "base_agent_count": len(trial["base_agents"]),
                    "agent_count": int(trial["agent_count"]),
                }
            )
    integrity = {
        "state_trial_count": len(trials) == expected_trials,
        "policy_row_count": len(rows) == expected_trials * len(POLICIES),
        "state_count": len({str(row["state_fingerprint"]) for row in rows})
        == int(config["cohort"]["required_state_count"]),
        "map_count": len({str(row["map_id"]) for row in rows})
        == int(config["cohort"]["required_map_count"]),
        "initial_attempt_parity": all(
            bool(trial["initial_attempt_parity"]) for trial in trials
        ),
        "rollback_integrity": all(
            not attempt["rolled_back"]
            or (
                attempt["after_repair_fingerprint"]
                == trial["before_repair_fingerprint"]
                and int(attempt["conflicts_after"]) == int(trial["before_conflicts"])
            )
            for trial in trials
            for policy in trial["policies"]
            for attempt in policy["attempts"]
        ),
        "no_runtime_or_future_fields": all(
            row["runtime_fields_stored"] is False
            and row["future_trajectory_stored"] is False
            for row in rows
        ),
        "attempt_caps": all(
            int(row["attempt_count"])
            <= int(config["policies"]["maximum_attempts"][row["policy_id"]])
            for row in rows
        ),
        "no_worse_conflict_commit": all(
            int(row["conflicts_after"]) <= int(row["before_conflicts"])
            for row in rows
        ),
    }
    summaries = {policy: _summary(rows, policy) for policy in POLICIES}
    by_map = {
        policy: _group_summary(rows, policy, "map_id") for policy in POLICIES
    }
    by_half = {
        policy: _group_summary(rows, policy, "seed_half") for policy in POLICIES
    }
    by_root = {
        policy: _group_summary(rows, policy, "discovery_root_cause")
        for policy in POLICIES
    }
    primary = summaries[PRIMARY_POLICY]
    baseline = summaries[BASELINE_POLICY]
    set_only = summaries[SET_POLICY]
    gates_config = dict(config["readiness_gates"])
    unchanged_advantage = (
        float(baseline["returned_unchanged_rate"])
        - float(primary["returned_unchanged_rate"])
    )
    success_advantage = (
        float(primary["replan_success_rate"])
        - float(baseline["replan_success_rate"])
    )
    gates = {
        "integrity": all(integrity.values()),
        "zero_errors_and_timeouts": int(status["error_job_count"]) == 0
        and int(status["timeout_job_count"]) == 0,
        "overall_unchanged_reduction": unchanged_advantage
        >= float(gates_config["minimum_returned_unchanged_rate_reduction"]),
        "overall_success_improvement": success_advantage
        >= float(gates_config["minimum_replan_success_rate_improvement"]),
        "strict_reduction_not_worse": float(
            primary["strict_conflict_reduction_rate"]
        )
        >= float(baseline["strict_conflict_reduction_rate"]),
        "per_map_unchanged_improvement": all(
            float(by_map[BASELINE_POLICY][key]["returned_unchanged_rate"])
            > float(by_map[PRIMARY_POLICY][key]["returned_unchanged_rate"])
            for key in by_map[PRIMARY_POLICY]
        ),
        "seed_half_unchanged_reduction": all(
            float(by_half[BASELINE_POLICY][key]["returned_unchanged_rate"])
            - float(by_half[PRIMARY_POLICY][key]["returned_unchanged_rate"])
            >= float(gates_config["minimum_seed_half_unchanged_rate_reduction"])
            for key in by_half[PRIMARY_POLICY]
        ),
        "joint_not_worse_than_set_only": float(primary["returned_unchanged_rate"])
        <= float(set_only["returned_unchanged_rate"]),
        "bounded_mean_attempts": float(primary["mean_attempt_count"])
        <= float(gates_config["maximum_mean_attempt_count"]),
        "bounded_mean_expansion": float(primary["mean_added_agent_ratio"])
        <= float(gates_config["maximum_mean_added_agent_ratio"]),
        "not_near_global": float(primary["maximum_total_neighborhood_fraction"])
        < float(gates_config["maximum_total_neighborhood_fraction"]),
        "sixteen_worker_execution": int(status["worker_limit"]) == 16
        and int(status["worker_process_count"]) >= 16,
        "tail_parallelism": bool(status["tail_parallelism_observed"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "state_count": int(config["cohort"]["required_state_count"]),
        "state_trial_count": len(trials),
        "policy_row_count": len(rows),
        "policy_summaries": summaries,
        "by_map": by_map,
        "by_seed_half": by_half,
        "by_discovery_root_cause": by_root,
        "readiness_gates": gates,
        "mechanism_readiness_passed": all(gates.values()),
        "execution": {
            "schema": EXECUTION_SCHEMA,
            "worker_limit": int(status["worker_limit"]),
            "worker_process_count": int(status["worker_process_count"]),
            "final_32_completed_job_worker_process_count": int(
                status["final_32_completed_job_worker_process_count"]
            ),
            "task_granularity": "state_x_trial_index_with_three_paired_policies",
            "per_job_timeout_seconds": float(
                config["execution"]["per_state_trial_timeout_seconds"]
            ),
            "tail_parallelism_observed": bool(status["tail_parallelism_observed"]),
        },
        "config_sha256": metadata["config_sha256"],
        "input_sha256": metadata["input_sha256"],
        "claim_boundary": dict(config["claim_boundary"]),
        "next_step": (
            "preregister an independent result-blind TransactionalRepair cohort"
            if all(gates.values())
            else "stop TransactionalRepair; do not tune on the discovery cohort"
        ),
        "producer": producer_identity(
            project_root=root,
            source_files=PRODUCER_FILES,
            native_required=True,
            package_names=(),
        ),
    }
    _write_json(output / "transactionalrepair_report.json", report)
    return report
