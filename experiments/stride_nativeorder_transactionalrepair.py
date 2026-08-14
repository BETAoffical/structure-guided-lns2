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
    _read_jsonl,
    _run_jobs,
    _write_json,
)
from experiments.stride_repairability_causal_audit import build_causal_cohort
from experiments.stride_repairability_collection import repairability_pp_seed
from experiments.stride_repairdependency_predictability import (
    _contained as _registered_collection_path,
)
from experiments.stride_transactionalrepair import (
    _load_restored_source,
    _run_attempt,
)
from experiments.trace_replay import restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.nativeorder_transactionalrepair_registration.v1"
RUN_SCHEMA = "lns2.stride.nativeorder_transactionalrepair_run.v1"
TRIAL_SCHEMA = "lns2.stride.nativeorder_transactionalrepair_trial.v1"
POLICY_SCHEMA = "lns2.stride.nativeorder_transactionalrepair_policy_result.v1"
STATUS_SCHEMA = "lns2.stride.nativeorder_transactionalrepair_status.v1"
REPORT_SCHEMA = "lns2.stride.nativeorder_transactionalrepair_report.v1"
EXPERIMENT_ID = "stride-nativeorder-transactionalrepair-v1"

BASELINE_POLICY = "selected_single_attempt"
SAME_SET_POLICY = "same_set_native_retry"
AUGMENTED_POLICY = "blocker_augmented_native_retry"
UPPER_BOUND_POLICY = "preserved_prefix_blocker_tail_upper_bound"
POLICIES = (
    BASELINE_POLICY,
    SAME_SET_POLICY,
    AUGMENTED_POLICY,
    UPPER_BOUND_POLICY,
)
DEPLOYABLE_POLICIES = (SAME_SET_POLICY, AUGMENTED_POLICY)
PHASES = ("qualification", "initial", "extension")

PRODUCER_FILES = (
    "experiments/stride_nativeorder_transactionalrepair.py",
    "scripts/run_stride_nativeorder_transactionalrepair.py",
    "experiments/stride_transactionalrepair.py",
    "experiments/stride_repairability_causal_audit.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def retry_pp_seed(state_repair_fingerprint: str, trial_index: int) -> int:
    seed = int(
        _fingerprint(
            {
                "namespace": "stride-nativeorder-transactionalrepair-retry-v1",
                "repair_state": str(state_repair_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)
    first = repairability_pp_seed(state_repair_fingerprint, trial_index)
    return (seed + 1) % (2**31) if seed == first else seed


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
        raise ValueError("NativeOrder TransactionalRepair registration changed")
    inputs = {
        name: registered_input(
            root, row, label=f"native-order transactional repair {name}"
        )
        for name, row in dict(config["inputs"]).items()
    }
    collection = _registered_collection_path(
        root,
        str(config["causal_collection"]["path"]),
        label="native-order transactional repair causal collection",
    )
    policy = dict(config["policies"])
    if tuple(policy["ids"]) != POLICIES:
        raise ValueError("NativeOrder TransactionalRepair policies changed")
    if tuple(policy["deployable_ids"]) != DEPLOYABLE_POLICIES:
        raise ValueError("NativeOrder deployable policy set changed")
    if int(policy["maximum_attempts"]) != 2:
        raise ValueError("NativeOrder attempt cap changed")
    if int(policy["maximum_added_external_blockers"]) != 8:
        raise ValueError("NativeOrder blocker cap changed")
    execution = dict(config["execution"])
    if int(execution["worker_count"]) != 16:
        raise ValueError("NativeOrder worker count changed")
    if int(execution["per_policy_job_timeout_seconds"]) != 300:
        raise ValueError("NativeOrder job timeout changed")
    return config_path, root, config, inputs, collection


def build_nativeorder_cohort(
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
        raise ValueError("NativeOrder source evidence changed")
    source_metadata, cohort = build_causal_cohort(inputs["causal_registration"])
    manifest = _read_jsonl(inputs["causal_state_manifest"])
    manifest_by_state = {str(row["state_fingerprint"]): row for row in manifest}
    cohort_by_state = {str(row["state_fingerprint"]): row for row in cohort}
    required = dict(config["cohort"])
    if (
        len(cohort_by_state) != int(required["required_state_count"])
        or set(cohort_by_state) != set(manifest_by_state)
        or len({str(row["map_id"]) for row in cohort})
        != int(required["required_map_count"])
    ):
        raise ValueError("NativeOrder cohort changed")
    for state_key, row in manifest_by_state.items():
        path = collection / "states" / str(row["file_name"])
        if sha256_file(path) != str(row["sha256"]):
            raise ValueError("NativeOrder causal state artifact changed")
    effects = _read_jsonl(inputs["causal_state_effects"])
    root_by_state = {
        str(row["state_fingerprint"]): str(row["root_cause"]) for row in effects
    }
    if set(root_by_state) != set(cohort_by_state):
        raise ValueError("NativeOrder root classifications changed")
    frozen = [
        {**row, "discovery_root_cause": root_by_state[str(row["state_fingerprint"])]}
        for row in cohort
    ]
    frozen.sort(key=lambda row: str(row["state_fingerprint"]))
    return {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "source_cohort_fingerprint": str(source_metadata["cohort_fingerprint"]),
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
    policy_id: str,
    base_agents: list[int],
    first_repair_order: list[int],
    observed_blockers: list[int],
    maximum_added_agents: int,
) -> tuple[list[int], list[int] | None, list[int], str] | None:
    if policy_id == BASELINE_POLICY:
        return None
    base = list(map(int, base_agents))
    base_set = set(base)
    blockers = [
        int(agent)
        for agent in observed_blockers
        if int(agent) not in base_set
    ][: int(maximum_added_agents)]
    if policy_id == SAME_SET_POLICY:
        return base, None, [], "fresh_native_same_set"
    augmented = base + blockers
    if policy_id == AUGMENTED_POLICY:
        return augmented, None, blockers, "fresh_native_blocker_augmented"
    if policy_id == UPPER_BOUND_POLICY:
        return (
            augmented,
            list(map(int, first_repair_order)) + blockers,
            blockers,
            "preserved_prefix_blocker_tail",
        )
    raise ValueError(f"unknown NativeOrder policy: {policy_id}")


def _timed_attempt(**kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    row, state = _run_attempt(**kwargs)
    row["attempt_wall_seconds"] = time.perf_counter() - started
    row["requested_pp_seed"] = int(kwargs["pp_seed"])
    row["explicit_repair_order_requested"] = kwargs["repair_order"] is not None
    return row, state


def _run_policy(
    *,
    replay: dict[str, Any],
    state: dict[str, Any],
    before_repair: str,
    before_conflicts: int,
    restore_seed: int,
    base_agents: list[int],
    first_seed: int,
    retry_seed: int,
    policy_id: str,
    maximum_added_agents: int,
) -> dict[str, Any]:
    environment, restored = restore_repair_state(replay, state, seed=restore_seed)
    if repair_structure_fingerprint(restored) != before_repair:
        raise RuntimeError("NativeOrder branch restore changed")
    attempts: list[dict[str, Any]] = []
    first, final_state = _timed_attempt(
        environment=environment,
        agents=base_agents,
        repair_order=None,
        pp_seed=first_seed,
        before_repair=before_repair,
        before_conflicts=before_conflicts,
        attempt_index=0,
        reason="selected_native_order",
    )
    attempts.append(first)
    added_agents: list[int] = []
    termination = "single_attempt_policy"
    if bool(first["replan_success"]):
        termination = "committed_first_success"
    elif str(first["failure_reason"]) == "time_limit":
        termination = "stopped_time_limit"
    elif str(first["failure_reason"]) != "conflict_bound_exceeded":
        termination = f"stopped_{first['failure_reason']}"
    else:
        plan = retry_plan(
            policy_id,
            base_agents,
            list(first["repair_order"]),
            list(first["external_blockers"]),
            maximum_added_agents,
        )
        if plan is not None:
            retry_agents, retry_order, added_agents, reason = plan
            second, final_state = _timed_attempt(
                environment=environment,
                agents=retry_agents,
                repair_order=retry_order,
                pp_seed=retry_seed,
                before_repair=before_repair,
                before_conflicts=before_conflicts,
                attempt_index=1,
                reason=reason,
            )
            attempts.append(second)
            termination = (
                "committed_retry_success"
                if bool(second["replan_success"])
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
        "added_agents": added_agents,
        "added_agent_count": len(added_agents),
        "final_agent_count": len(attempts[-1]["agents"]),
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
        raise ValueError("invalid completed NativeOrder trial")
    state, replay, before_repair, before_conflicts, restore_seed = (
        _load_restored_source(state_record)
    )
    base_agents = list(map(int, state_record["selected_candidate"]["agents"]))
    first_seed = repairability_pp_seed(before_repair, trial_index)
    second_seed = retry_pp_seed(before_repair, trial_index)
    policy = _run_policy(
        replay=replay,
        state=state,
        before_repair=before_repair,
        before_conflicts=before_conflicts,
        restore_seed=restore_seed,
        base_agents=base_agents,
        first_seed=first_seed,
        retry_seed=second_seed,
        policy_id=policy_id,
        maximum_added_agents=int(job["maximum_added_agents"]),
    )
    payload = {
        "schema": TRIAL_SCHEMA,
        "complete": True,
        "phase": str(job["phase"]),
        "run_fingerprint": run_fingerprint,
        "state_fingerprint": state_key,
        "state_blob_sha256": str(state_record["state_blob_sha256"]),
        "map_id": str(state_record["map_id"]),
        "task_id": str(state_record["task_id"]),
        "solver_seed": int(state_record["solver_seed"]),
        "discovery_root_cause": str(state_record["discovery_root_cause"]),
        "trial_index": trial_index,
        "first_pp_seed": first_seed,
        "retry_pp_seed": second_seed,
        "before_conflicts": before_conflicts,
        "before_repair_fingerprint": before_repair,
        "base_agents": base_agents,
        "agent_count": len(state["agents"]),
        "policy": policy,
        "future_trajectory_stored": False,
    }
    _write_json(output_path, payload)
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


def _trial_files(output: Path) -> list[Path]:
    return sorted((output / "trials").glob("*/*.json"))


def collect_phase(
    *,
    config_path: str | Path,
    output: str | Path,
    phase: str,
    resume: bool = False,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown NativeOrder phase: {phase}")
    metadata, cohort = build_nativeorder_cohort(config_path)
    _path, root, config, _inputs, _collection = load_registration(config_path)
    if phase == "qualification":
        qualification = dict(config["qualification"])
        cohort = [
            row
            for row in cohort
            if str(row["state_fingerprint"])
            == str(qualification["state_fingerprint"])
        ]
        trial_indices = [int(qualification["trial_index"])]
        if len(cohort) != 1:
            raise ValueError("NativeOrder qualification state changed")
    else:
        trial_indices = list(
            map(int, config["cohort"][f"{phase}_trial_indices"])
        )
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
        "producer": producer,
    }
    run_fingerprint = _fingerprint(run_identity)
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("NativeOrder output belongs to another run")
        if not resume:
            raise ValueError("NativeOrder output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**run_identity, "run_fingerprint": run_fingerprint})
    jobs: list[dict[str, Any]] = []
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
                        "maximum_added_agents": int(
                            config["policies"]["maximum_added_external_blockers"]
                        ),
                    }
                )
    results = _run_jobs(
        _collect_job,
        jobs,
        int(config["execution"]["worker_count"]),
        phase=f"nativeorder-transactionalrepair-{phase}",
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
    expected = len(jobs)
    status = {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "status": "complete" if len(files) == expected and not errors else "failed" if errors else "incomplete",
        "run_fingerprint": run_fingerprint,
        "completed_jobs": len(files),
        "required_jobs": expected,
        "error_jobs": sum(row["status"] == "error" for row in errors),
        "timeout_jobs": sum(row["status"] == "timeout" for row in errors),
        "active_jobs": [],
        "worker_limit": int(config["execution"]["worker_count"]),
        "worker_process_count": len(pids),
        "errors": errors,
    }
    _write_json(output / "collection_status.json", status)
    return status


def _load_artifacts(outputs: list[Path]) -> list[dict[str, Any]]:
    return [_read_json(path) for output in outputs for path in _trial_files(output)]


def analyze_qualification(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    _metadata, _cohort = build_nativeorder_cohort(config_path)
    _path, _root, config, _inputs, _collection = load_registration(config_path)
    output = Path(output).resolve()
    artifacts = _load_artifacts([output])
    by_policy = {str(row["policy"]["policy_id"]): row for row in artifacts}
    if set(by_policy) != set(POLICIES):
        raise ValueError("NativeOrder qualification is incomplete")
    first = [row["policy"]["attempts"][0] for row in artifacts]
    baseline_first = by_policy[BASELINE_POLICY]["policy"]["attempts"][0]
    retry_rows = {
        key: by_policy[key]["policy"]["attempts"][1]
        for key in POLICIES[1:]
        if len(by_policy[key]["policy"]["attempts"]) == 2
    }
    base_agents = list(map(int, by_policy[BASELINE_POLICY]["base_agents"]))
    blockers = list(map(int, baseline_first["external_blockers"]))[: int(
        config["policies"]["maximum_added_external_blockers"]
    )]
    gates = {
        "four_policy_artifacts": len(artifacts) == 4,
        "first_attempt_parity": len({str(row["attempt_signature"]) for row in first}) == 1,
        "first_attempt_conflict_bound_failure": baseline_first["failure_reason"] == "conflict_bound_exceeded",
        "first_attempt_exact_rollback": bool(baseline_first["rolled_back"]),
        "observed_external_blocker": bool(blockers),
        "three_retry_rows": set(retry_rows) == set(POLICIES[1:]),
        "fresh_retry_seed": all(
            int(row["requested_pp_seed"])
            != int(by_policy[key]["first_pp_seed"])
            for key, row in retry_rows.items()
        ),
        "paired_retry_seed": len(
            {int(row["requested_pp_seed"]) for row in retry_rows.values()}
        ) == 1,
        "same_set_native": list(retry_rows[SAME_SET_POLICY]["agents"]) == base_agents
        and retry_rows[SAME_SET_POLICY]["explicit_repair_order_requested"] is False,
        "augmented_set_native": list(retry_rows[AUGMENTED_POLICY]["agents"])
        == base_agents + blockers
        and retry_rows[AUGMENTED_POLICY]["explicit_repair_order_requested"] is False,
        "upper_bound_explicit_order": list(retry_rows[UPPER_BOUND_POLICY]["agents"])
        == base_agents + blockers
        and list(retry_rows[UPPER_BOUND_POLICY]["repair_order"])
        == list(baseline_first["repair_order"]) + blockers
        and retry_rows[UPPER_BOUND_POLICY]["explicit_repair_order_requested"] is True,
    }
    report = {
        "schema": "lns2.stride.nativeorder_transactionalrepair_qualification.v1",
        "experiment_id": EXPERIMENT_ID,
        "state_fingerprint": str(artifacts[0]["state_fingerprint"]),
        "trial_index": int(artifacts[0]["trial_index"]),
        "excluded_from_scientific_analysis": True,
        "observed_blockers": blockers,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _write_json(output / "qualification_report.json", report)
    return report


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
        "mean_total_attempt_wall_seconds": mean(
            float(row["total_attempt_wall_seconds"]) for row in selected
        ),
    }


def _group_summary(
    rows: list[dict[str, Any]], policy_id: str, key: str
) -> dict[str, dict[str, Any]]:
    return {
        value: _summary([row for row in rows if str(row[key]) == value], policy_id)
        for value in sorted({str(row[key]) for row in rows})
    }


def _paired_state_bootstrap(
    rows: list[dict[str, Any]], left: str, right: str, replicates: int
) -> dict[str, float]:
    values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["policy_id"] in {left, right}:
            values[str(row["state_fingerprint"])][str(row["policy_id"])].append(
                float(row["returned_unchanged"])
            )
    state_differences = [
        mean(group[left]) - mean(group[right])
        for group in values.values()
        if group[left] and group[right]
    ]
    if not state_differences:
        raise ValueError("NativeOrder bootstrap has no paired states")
    rng = random.Random(20260814)
    samples = sorted(
        mean(rng.choice(state_differences) for _ in state_differences)
        for _ in range(int(replicates))
    )
    lower_index = int(0.025 * (len(samples) - 1))
    upper_index = int(0.975 * (len(samples) - 1))
    return {
        "risk_difference": mean(state_differences),
        "lower_95": samples[lower_index],
        "upper_95": samples[upper_index],
        "state_cluster_count": len(state_differences),
        "replicates": int(replicates),
    }


def analyze(
    config_path: str | Path,
    initial_output: str | Path,
    extension_output: str | Path | None = None,
) -> dict[str, Any]:
    metadata, cohort = build_nativeorder_cohort(config_path)
    _path, root, config, _inputs, _collection = load_registration(config_path)
    outputs = [Path(initial_output).resolve()]
    phase = "initial"
    if extension_output is not None:
        outputs.append(Path(extension_output).resolve())
        phase = "extended"
    statuses = [_read_json(output / "collection_status.json") for output in outputs]
    if any(status.get("status") != "complete" for status in statuses):
        raise ValueError("NativeOrder collection is incomplete")
    artifacts = _load_artifacts(outputs)
    trial_indices = list(map(int, config["cohort"]["initial_trial_indices"]))
    if extension_output is not None:
        trial_indices += list(map(int, config["cohort"]["extension_trial_indices"]))
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        key = (str(artifact["state_fingerprint"]), int(artifact["trial_index"]))
        grouped[key].append(artifact)
        policy = dict(artifact["policy"])
        rows.append(
            {
                **policy,
                "state_fingerprint": key[0],
                "trial_index": key[1],
                "map_id": str(artifact["map_id"]),
                "discovery_root_cause": str(artifact["discovery_root_cause"]),
                "before_conflicts": int(artifact["before_conflicts"]),
                "base_agent_count": len(artifact["base_agents"]),
                "agent_count": int(artifact["agent_count"]),
            }
        )
    expected_pairs = len(cohort) * len(trial_indices)
    first_attempt_parity = all(
        len(
            {
                str(artifact["policy"]["attempts"][0]["attempt_signature"])
                for artifact in group
            }
        )
        == 1
        for group in grouped.values()
    )
    integrity = {
        "state_trial_count": len(grouped) == expected_pairs,
        "policy_artifact_count": len(artifacts) == expected_pairs * len(POLICIES),
        "policy_coverage": all(
            {str(row["policy"]["policy_id"]) for row in group} == set(POLICIES)
            for group in grouped.values()
        ),
        "state_count": len({row[0] for row in grouped})
        == int(config["cohort"]["required_state_count"]),
        "trial_coverage": {row[1] for row in grouped} == set(trial_indices),
        "first_attempt_parity": first_attempt_parity,
        "exact_failed_rollback": all(
            bool(attempt["replan_success"])
            or (
                bool(attempt["rolled_back"])
                and attempt["after_repair_fingerprint"]
                == artifact["before_repair_fingerprint"]
                and int(attempt["conflicts_after"])
                == int(artifact["before_conflicts"])
            )
            for artifact in artifacts
            for attempt in artifact["policy"]["attempts"]
        ),
        "attempt_cap": all(int(row["attempt_count"]) <= 2 for row in rows),
        "native_order_deployable_retries": all(
            attempt["explicit_repair_order_requested"] is False
            for artifact in artifacts
            if artifact["policy"]["policy_id"] in DEPLOYABLE_POLICIES
            for attempt in artifact["policy"]["attempts"]
        ),
        "fresh_retry_seed": all(
            len(artifact["policy"]["attempts"]) == 1
            or int(artifact["policy"]["attempts"][1]["requested_pp_seed"])
            == int(artifact["retry_pp_seed"])
            != int(artifact["first_pp_seed"])
            for artifact in artifacts
        ),
        "no_time_limit_retry": all(
            len(artifact["policy"]["attempts"]) == 1
            for artifact in artifacts
            if artifact["policy"]["attempts"][0]["failure_reason"] == "time_limit"
        ),
        "no_future_trajectory": all(
            artifact.get("future_trajectory_stored") is False
            and artifact["policy"].get("future_trajectory_stored") is False
            for artifact in artifacts
        ),
    }
    summaries = {policy: _summary(rows, policy) for policy in POLICIES}
    by_map = {policy: _group_summary(rows, policy, "map_id") for policy in POLICIES}
    baseline = summaries[BASELINE_POLICY]
    initial_gates: dict[str, dict[str, bool]] = {}
    for policy in DEPLOYABLE_POLICIES:
        current = summaries[policy]
        initial_gates[policy] = {
            "lower_returned_unchanged_rate": float(current["returned_unchanged_rate"])
            < float(baseline["returned_unchanged_rate"]),
            "success_not_lower": float(current["replan_success_rate"])
            >= float(baseline["replan_success_rate"]),
            "strict_reduction_not_lower": float(
                current["strict_conflict_reduction_rate"]
            )
            >= float(baseline["strict_conflict_reduction_rate"]),
            "per_map_not_worse_by_more_than_five_points": all(
                float(by_map[policy][map_id]["returned_unchanged_rate"])
                - float(by_map[BASELINE_POLICY][map_id]["returned_unchanged_rate"])
                <= float(
                    config["initial_extension_gate"][
                        "maximum_per_map_returned_unchanged_rate_increase"
                    ]
                )
                for map_id in by_map[policy]
            ),
        }
    extension_qualified = any(all(gates.values()) for gates in initial_gates.values())
    bootstraps: dict[str, dict[str, float]] = {}
    final_gates: dict[str, dict[str, bool]] = {}
    mechanism_selection = "initial_phase_only"
    if phase == "extended":
        replicates = int(
            config["final_readiness_gates"][
                "paired_state_cluster_bootstrap_replicates"
            ]
        )
        for policy in POLICIES[1:]:
            bootstraps[policy] = _paired_state_bootstrap(
                rows, policy, BASELINE_POLICY, replicates
            )
        for policy in DEPLOYABLE_POLICIES:
            current = summaries[policy]
            final_gates[policy] = {
                "paired_risk_upper_below_zero": float(
                    bootstraps[policy]["upper_95"]
                )
                < 0.0,
                "success_not_lower": float(current["replan_success_rate"])
                >= float(baseline["replan_success_rate"]),
                "strict_reduction_not_lower": float(
                    current["strict_conflict_reduction_rate"]
                )
                >= float(baseline["strict_conflict_reduction_rate"]),
                "per_map_not_worse_by_more_than_five_points": all(
                    float(by_map[policy][map_id]["returned_unchanged_rate"])
                    - float(by_map[BASELINE_POLICY][map_id]["returned_unchanged_rate"])
                    <= float(
                        config["final_readiness_gates"][
                            "maximum_per_map_returned_unchanged_rate_increase"
                        ]
                    )
                    for map_id in by_map[policy]
                ),
                "bounded_attempts": float(current["mean_attempt_count"])
                <= float(
                    config["final_readiness_gates"]["maximum_mean_attempt_count"]
                ),
                "bounded_expansion": float(current["mean_added_agent_ratio"])
                <= float(
                    config["final_readiness_gates"]["maximum_mean_added_agent_ratio"]
                ),
                "not_near_global": float(
                    current["maximum_total_neighborhood_fraction"]
                )
                < float(
                    config["final_readiness_gates"][
                        "maximum_total_neighborhood_fraction"
                    ]
                ),
            }
        same_pass = all(final_gates[SAME_SET_POLICY].values())
        augmented_pass = all(final_gates[AUGMENTED_POLICY].values())
        upper_pass = float(bootstraps[UPPER_BOUND_POLICY]["upper_95"]) < 0.0
        augmented_vs_same = _paired_state_bootstrap(
            rows, AUGMENTED_POLICY, SAME_SET_POLICY, replicates
        )
        bootstraps["augmented_vs_same_set"] = augmented_vs_same
        if augmented_pass and float(augmented_vs_same["upper_95"]) < 0.0:
            mechanism_selection = AUGMENTED_POLICY
        elif same_pass:
            mechanism_selection = SAME_SET_POLICY
        elif augmented_pass:
            mechanism_selection = "native_retry_helps_but_membership_increment_unresolved"
        elif upper_pass:
            mechanism_selection = "only_controlled_order_upper_bound_passed"
        else:
            mechanism_selection = "no_bounded_retry_arm_passed"
    zero_failures = all(
        int(status["error_jobs"]) == 0 and int(status["timeout_jobs"]) == 0
        for status in statuses
    )
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "phase": phase,
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "state_count": len({row[0] for row in grouped}),
        "state_trial_count": len(grouped),
        "policy_artifact_count": len(artifacts),
        "policy_summaries": summaries,
        "by_map": by_map,
        "initial_extension_gates": initial_gates,
        "extension_qualified": extension_qualified,
        "paired_state_cluster_bootstrap": bootstraps,
        "final_readiness_gates": final_gates,
        "mechanism_selection": mechanism_selection,
        "zero_errors_and_timeouts": zero_failures,
        "sixteen_worker_configuration": all(
            int(status["worker_limit"]) == 16 for status in statuses
        ),
        "config_sha256": metadata["config_sha256"],
        "input_sha256": metadata["input_sha256"],
        "claim_boundary": dict(config["claim_boundary"]),
        "next_step": (
            "collect the uniform trial 8-15 extension"
            if phase == "initial" and extension_qualified
            else "stop without extension"
            if phase == "initial"
            else "preregister an independent result-blind confirmation"
            if mechanism_selection in DEPLOYABLE_POLICIES
            else "stop or redesign the repairer without tuning this cohort"
        ),
        "producer": producer_identity(
            project_root=root,
            source_files=PRODUCER_FILES,
            native_required=True,
            package_names=(),
        ),
    }
    report_name = (
        "nativeorder_transactionalrepair_initial_report.json"
        if phase == "initial"
        else "nativeorder_transactionalrepair_extended_report.json"
    )
    _write_json(outputs[-1] / report_name, report)
    return report
