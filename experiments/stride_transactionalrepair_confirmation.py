from __future__ import annotations

import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, registered_input, sha256_file
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
)
from experiments.stride_collection import _replay_job, load_stride_selection
from experiments.stride_repairability_collection import (
    _source_target_state,
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_repairdependency_predictability import (
    _contained as _registered_collection_path,
)
from experiments.stride_transactionalrepair import (
    BASELINE_POLICY,
    POLICIES,
    PRIMARY_POLICY,
    SET_POLICY,
    _failed_job,
    _group_summary,
    _run_policy,
    _summary,
    _trial_files,
)
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.transactionalrepair_confirmation_registration.v1"
COHORT_ROW_SCHEMA = "lns2.stride.transactionalrepair_confirmation_cohort_row.v1"
COHORT_REPORT_SCHEMA = "lns2.stride.transactionalrepair_confirmation_cohort.v1"
RUN_SCHEMA = "lns2.stride.transactionalrepair_confirmation_run.v1"
TRIAL_SCHEMA = "lns2.stride.transactionalrepair_confirmation_trial.v1"
STATUS_SCHEMA = "lns2.stride.transactionalrepair_confirmation_status.v1"
REPORT_SCHEMA = "lns2.stride.transactionalrepair_confirmation_report.v1"
EXECUTION_SCHEMA = "lns2.stride.transactionalrepair_confirmation_execution.v1"
EXPERIMENT_ID = "stride-transactionalrepair-confirmation-v1"
FORBIDDEN_SELECTION_FIELDS = {
    "actual_action",
    "actual_lns2",
    "after_fingerprint",
    "candidate_repair_outcome",
    "controller_action",
    "controller_ttf",
    "future_trajectory",
    "repair_outcome",
    "repair_runtime",
    "repair_seconds",
    "repair_state_changed",
    "replay_action",
}
PRODUCER_FILES = (
    "experiments/repair_collection.py",
    "experiments/stride_transactionalrepair.py",
    "experiments/stride_transactionalrepair_confirmation.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "scripts/run_stride_transactionalrepair_confirmation.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _selected_neighborhood(
    decision: dict[str, Any], trace_path: Path
) -> tuple[list[int], str]:
    events = read_trace_events(trace_path)
    matches = [
        event
        for event in events[1:-1]
        if int(event.get("decision_index", -1)) == int(decision["decision_index"])
    ]
    if len(matches) != 1:
        raise ValueError("confirmation decision must resolve to one source transition")
    event = matches[0]
    if str(event.get("before_fingerprint")) != str(decision["before_fingerprint"]):
        raise ValueError("confirmation source before fingerprint changed")
    metrics = event.get("metrics")
    action = event.get("action")
    if not isinstance(metrics, dict) or not isinstance(action, dict):
        raise ValueError("confirmation source transition is incomplete")
    neighborhood = metrics.get("neighborhood")
    if not isinstance(neighborhood, list):
        raise ValueError("confirmation source transition lacks a neighborhood")
    agents = sorted(map(int, neighborhood))
    if not agents or len(agents) != len(set(agents)):
        raise ValueError("confirmation source neighborhood is invalid")
    return agents, str(action.get("mode", "unknown"))


def _prepare_confirmation_state(decision: dict[str, Any]) -> dict[str, Any]:
    forbidden = sorted(FORBIDDEN_SELECTION_FIELDS & set(decision))
    if forbidden:
        raise ValueError(f"result-blind confirmation row has forbidden fields: {forbidden}")
    state, manifest, trace_path = _source_target_state(decision)
    agents, source_action_mode = _selected_neighborhood(decision, trace_path)
    valid_agents = {int(row["id"]) for row in state.get("agents", [])}
    if not set(agents) <= valid_agents:
        raise ValueError("confirmation source neighborhood has unknown agents")
    before_repair = repair_structure_fingerprint(state)
    trace_sha256 = sha256_file(trace_path)
    if trace_sha256 != str(manifest.get("trace_sha256")):
        raise ValueError("confirmation source trace hash changed")
    candidate_id = "source-selected-" + _fingerprint(
        {
            "episode_id": decision["episode_id"],
            "decision_index": int(decision["decision_index"]),
            "before_fingerprint": decision["before_fingerprint"],
            "agents": agents,
        }
    )[:24]
    return {
        **decision,
        "schema": COHORT_ROW_SCHEMA,
        "state_fingerprint": str(decision["before_fingerprint"]),
        "before_repair_fingerprint": before_repair,
        "restore_seed": repairability_restore_seed(before_repair),
        "source_trace_file": str(manifest["trace_file"]),
        "source_trace_sha256": trace_sha256,
        "selected_candidate": {
            "candidate_id": candidate_id,
            "agents": agents,
            "actual_size": len(agents),
            "selection_families": ["source_selected_neighborhood"],
            "source_action_mode": source_action_mode,
        },
        "candidate_outcomes_read": False,
        "future_trajectory_read": False,
    }


def prepare_confirmation_cohort(
    *,
    selection_path: str | Path,
    selection_report_path: str | Path,
    source_selection_report_path: str | Path,
    discovery_report_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    selection_path = Path(selection_path).resolve()
    selection_report_path = Path(selection_report_path).resolve()
    source_selection_report_path = Path(source_selection_report_path).resolve()
    discovery_report_path = Path(discovery_report_path).resolve()
    selection_report = _read_json(selection_report_path)
    source_report = _read_json(source_selection_report_path)
    discovery_report = _read_json(discovery_report_path)
    if (
        selection_report.get("passed") is not True
        or selection_report.get("result_blind") is not True
        or selection_report.get("candidate_outcomes_read") is not False
        or source_report.get("passed") is not True
        or source_report.get("result_blind") is not True
        or source_report.get("candidate_outcomes_read") is not False
        or source_report.get("controller_outcomes_read") is not False
        or discovery_report.get("mechanism_readiness_passed") is not True
    ):
        raise ValueError("confirmation source evidence is not eligible")
    selected = load_stride_selection(selection_path)
    prepared = [_prepare_confirmation_state(row) for row in selected]
    prepared.sort(key=lambda row: (str(row["source_policy"]), str(row["state_id"])))
    state_ids = [str(row["state_id"]) for row in prepared]
    maps = {str(row["map_id"]) for row in prepared}
    discovery_maps = set(discovery_report["by_map"][BASELINE_POLICY])
    policy_counts = Counter(str(row["source_policy"]) for row in prepared)
    gates = {
        "state_count": len(prepared) == 240,
        "unique_state_count": len(state_ids) == len(set(state_ids)) == 240,
        "map_count": len(maps) == 22,
        "source_policy_balance": policy_counts
        == {"official_adaptive": 120, "v2-full": 120},
        "selection_exact": set(state_ids)
        == {str(row["state_id"]) for row in selected},
        "discovery_map_disjoint": not maps & discovery_maps,
        "candidate_outcomes_unread": all(
            row["candidate_outcomes_read"] is False for row in prepared
        ),
        "future_trajectory_unread": all(
            row["future_trajectory_read"] is False for row in prepared
        ),
        "selected_neighborhoods_valid": all(
            row["selected_candidate"]["agents"]
            and len(row["selected_candidate"]["agents"])
            == len(set(row["selected_candidate"]["agents"]))
            for row in prepared
        ),
    }
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohort_path = output / "confirmation_cohort.jsonl"
    _write_jsonl(cohort_path, prepared)
    report = {
        "schema": COHORT_REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "result_blind": True,
        "candidate_outcomes_read": False,
        "controller_outcomes_read": False,
        "future_trajectory_read": False,
        "selected_neighborhood_membership_source": (
            "recorded pre-repair neighborhood of the already frozen source decision"
        ),
        "state_count": len(prepared),
        "map_count": len(maps),
        "source_policy_counts": dict(sorted(policy_counts.items())),
        "research_split_counts": dict(
            sorted(Counter(str(row["research_split"]) for row in prepared).items())
        ),
        "conflict_band_counts": dict(
            sorted(Counter(str(row["conflict_band"]) for row in prepared).items())
        ),
        "decision_stage_counts": dict(
            sorted(Counter(str(row["decision_stage"]) for row in prepared).items())
        ),
        "discovery_map_ids": sorted(discovery_maps),
        "confirmation_map_ids": sorted(maps),
        "gates": gates,
        "passed": all(gates.values()),
        "input_sha256": {
            "selection": sha256_file(selection_path),
            "selection_report": sha256_file(selection_report_path),
            "source_selection_report": sha256_file(source_selection_report_path),
            "discovery_report": sha256_file(discovery_report_path),
        },
        "cohort_sha256": sha256_file(cohort_path),
    }
    _write_json(output / "confirmation_cohort_report.json", report)
    return report


def load_confirmation_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], Path, list[dict[str, Any]]]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("TransactionalRepair confirmation registration changed")
    inputs = {
        name: registered_input(root, row, label=f"confirmation {name}")
        for name, row in dict(config["inputs"]).items()
    }
    discovery_registration = _read_json(inputs["discovery_registration"])
    discovery_report = _read_json(inputs["discovery_report"])
    cohort_report = _read_json(inputs["cohort_report"])
    if (
        tuple(config["policies"]["ids"]) != POLICIES
        or str(config["policies"]["primary"]) != PRIMARY_POLICY
        or dict(config["policies"]) != dict(discovery_registration["policies"])
        or dict(config["readiness_gates"])
        != dict(discovery_registration["readiness_gates"])
    ):
        raise ValueError("confirmation changed the discovery policy or gates")
    if (
        discovery_report.get("mechanism_readiness_passed") is not True
        or cohort_report.get("passed") is not True
        or cohort_report.get("result_blind") is not True
        or cohort_report.get("candidate_outcomes_read") is not False
    ):
        raise ValueError("confirmation prerequisite changed")
    source_collection = _registered_collection_path(
        root, str(config["source_collection"]["path"]), label="source collection"
    )
    cohort = _read_jsonl(inputs["cohort"])
    required = dict(config["cohort"])
    state_ids = [str(row["state_id"]) for row in cohort]
    maps = {str(row["map_id"]) for row in cohort}
    policies = Counter(str(row["source_policy"]) for row in cohort)
    if (
        len(cohort) != int(required["required_state_count"])
        or len(state_ids) != len(set(state_ids))
        or len(maps) != int(required["required_map_count"])
        or policies != dict(required["required_source_policy_counts"])
        or sha256_file(inputs["cohort"]) != str(cohort_report["cohort_sha256"])
        or any(row.get("schema") != COHORT_ROW_SCHEMA for row in cohort)
        or any(Path(str(row["source_root"])).resolve() != source_collection for row in cohort)
        or any(row.get("candidate_outcomes_read") is not False for row in cohort)
        or maps & set(discovery_report["by_map"][BASELINE_POLICY])
    ):
        raise ValueError("confirmation cohort changed")
    return config_path, root, config, inputs, source_collection, cohort


def _collect_confirmation_job(job: dict[str, Any]) -> dict[str, Any]:
    record = dict(job["state_record"])
    state_id = str(record["state_id"])
    trial_index = int(job["trial_index"])
    policy_id = str(job["policy_id"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if output_path.is_file():
        existing = _read_json(output_path)
        if (
            existing.get("schema") == TRIAL_SCHEMA
            and existing.get("run_fingerprint") == run_fingerprint
            and existing.get("state_id") == state_id
            and int(existing.get("trial_index", -1)) == trial_index
            and existing.get("policy", {}).get("policy_id") == policy_id
        ):
            return {
                "job_id": str(job["job_id"]),
                "status": "resumed",
                "state_count": 0,
                "outcome_count": 1,
                "error_count": 0,
                "worker_pid": os.getpid(),
            }
        raise ValueError("invalid completed confirmation trial")
    state, manifest, trace_path = _source_target_state(record)
    before_repair = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    if (
        before_repair != str(record["before_repair_fingerprint"])
        or before_conflicts != int(record["before_conflicts"])
        or sha256_file(trace_path) != str(record["source_trace_sha256"])
        or str(manifest["trace_file"]) != str(record["source_trace_file"])
    ):
        raise ValueError("confirmation source state changed")
    replay = _replay_job(record)
    restore_seed = int(record["restore_seed"])
    base_agents = list(map(int, record["selected_candidate"]["agents"]))
    pp_seed = repairability_pp_seed(before_repair, trial_index)
    policy = _run_policy(
        replay=replay,
        state=state,
        before_repair=before_repair,
        before_conflicts=before_conflicts,
        restore_seed=restore_seed,
        base_agents=base_agents,
        pp_seed=pp_seed,
        policy_id=policy_id,
        maximum_added_agents=int(job["maximum_added_agents"]),
        maximum_attempts=int(job["maximum_attempts"]),
    )
    payload = {
        "schema": TRIAL_SCHEMA,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "state_id": state_id,
        "state_fingerprint": str(record["state_fingerprint"]),
        "before_repair_fingerprint": before_repair,
        "source_trace_sha256": str(record["source_trace_sha256"]),
        "map_id": str(record["map_id"]),
        "task_id": str(record["task_id"]),
        "source_policy": str(record["source_policy"]),
        "research_split": str(record["research_split"]),
        "conflict_band": str(record["conflict_band"]),
        "decision_stage": str(record["decision_stage"]),
        "topology_group": str(record["topology_group"]),
        "trial_index": trial_index,
        "pp_seed": pp_seed,
        "before_conflicts": before_conflicts,
        "base_candidate_id": str(record["selected_candidate"]["candidate_id"]),
        "base_agents": base_agents,
        "agent_count": len(state["agents"]),
        "policy": policy,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)
    return {
        "job_id": str(job["job_id"]),
        "status": "ok",
        "state_count": 0,
        "outcome_count": 1,
        "error_count": 0,
        "worker_pid": os.getpid(),
    }


def collect_confirmation(
    *, config_path: str | Path, output: str | Path, resume: bool = False
) -> dict[str, Any]:
    config_path, root, config, inputs, _source, cohort = (
        load_confirmation_registration(config_path)
    )
    output = Path(output).resolve()
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=(),
    )
    identity = {
        "schema": RUN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in sorted(inputs.items())},
        "cohort_fingerprint": _fingerprint(cohort),
        "policies": list(POLICIES),
        "trial_indices": list(map(int, config["cohort"]["trial_indices"])),
        "worker_count": int(config["execution"]["worker_count"]),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("confirmation output belongs to another run")
        if not resume:
            raise ValueError("confirmation output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    maximum_attempts = {
        str(key): int(value)
        for key, value in dict(config["policies"]["maximum_attempts"]).items()
    }
    jobs = []
    for record in cohort:
        state_key = _fingerprint(
            {"state_id": record["state_id"], "state": record["state_fingerprint"]}
        )[:24]
        for trial_index in map(int, config["cohort"]["trial_indices"]):
            for policy_id in POLICIES:
                jobs.append(
                    {
                        "job_id": f"{record['state_id']}:{trial_index:02d}:{policy_id}",
                        "state_record": record,
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
                        "maximum_attempts": maximum_attempts[policy_id],
                    }
                )
    results = _run_jobs(
        _collect_confirmation_job,
        jobs,
        int(config["execution"]["worker_count"]),
        phase="transactionalrepair-confirmation-policy",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(
            config["execution"]["per_state_trial_policy_timeout_seconds"]
        ),
        failure_result=_failed_job,
        stop_on_failure=True,
    )
    files = _trial_files(output)
    expected = len(cohort) * len(config["cohort"]["trial_indices"]) * len(POLICIES)
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
        "status": "complete" if len(files) == expected and not errors else "failed" if errors else "incomplete",
        "run_fingerprint": run_fingerprint,
        "completed_policy_job_count": len(files),
        "required_policy_job_count": expected,
        "completed_state_trial_count": len(files) // len(POLICIES),
        "required_state_trial_count": expected // len(POLICIES),
        "error_job_count": sum(row["status"] == "error" for row in errors),
        "timeout_job_count": sum(row["status"] == "timeout" for row in errors),
        "worker_process_count": len(execution_pids),
        "final_32_completed_job_worker_process_count": len(final_pids),
        "worker_limit": int(config["execution"]["worker_count"]),
        "tail_parallelism_observed": len(final_pids) > 1,
        "stopped_on_first_failure": bool(errors),
        "errors": errors,
    }
    _write_json(output / "collection_status.json", status)
    return status


def analyze_confirmation(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, root, config, inputs, _source, cohort = (
        load_confirmation_registration(config_path)
    )
    output = Path(output).resolve()
    status = _read_json(output / "collection_status.json")
    run = _read_json(output / "run_config.json")
    if status.get("status") != "complete":
        raise ValueError("TransactionalRepair confirmation collection is incomplete")
    artifacts = [_read_json(path) for path in _trial_files(output)]
    cohort_by_id = {str(row["state_id"]): row for row in cohort}
    expected_trials = len(cohort) * len(config["cohort"]["trial_indices"])
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        grouped[(str(artifact["state_id"]), int(artifact["trial_index"]))].append(artifact)
        policy = dict(artifact["policy"])
        rows.append(
            {
                **policy,
                "state_id": str(artifact["state_id"]),
                "map_id": str(artifact["map_id"]),
                "source_policy": str(artifact["source_policy"]),
                "research_split": str(artifact["research_split"]),
                "conflict_band": str(artifact["conflict_band"]),
                "decision_stage": str(artifact["decision_stage"]),
                "topology_group": str(artifact["topology_group"]),
                "trial_index": int(artifact["trial_index"]),
                "seed_half": "first" if int(artifact["trial_index"]) < 8 else "second",
                "before_conflicts": int(artifact["before_conflicts"]),
                "base_agent_count": len(artifact["base_agents"]),
                "agent_count": int(artifact["agent_count"]),
            }
        )
    expected_keys = {
        (str(record["state_id"]), trial_index)
        for record in cohort
        for trial_index in map(int, config["cohort"]["trial_indices"])
    }
    integrity = {
        "state_trial_count": len(grouped) == expected_trials,
        "state_trial_identity": set(grouped) == expected_keys,
        "policy_artifact_count": len(artifacts) == expected_trials * len(POLICIES),
        "policy_coverage": all(
            {str(artifact["policy"]["policy_id"]) for artifact in group}
            == set(POLICIES)
            for group in grouped.values()
        ),
        "state_count": len({str(row["state_id"]) for row in rows})
        == int(config["cohort"]["required_state_count"]),
        "map_count": len({str(row["map_id"]) for row in rows})
        == int(config["cohort"]["required_map_count"]),
        "run_fingerprint": all(
            artifact.get("run_fingerprint") == run["run_fingerprint"]
            for artifact in artifacts
        ),
        "cohort_candidate_identity": all(
            artifact["base_candidate_id"]
            == cohort_by_id[str(artifact["state_id"])]["selected_candidate"]["candidate_id"]
            and artifact["base_agents"]
            == cohort_by_id[str(artifact["state_id"])]["selected_candidate"]["agents"]
            for artifact in artifacts
        ),
        "source_trace_identity": all(
            artifact["source_trace_sha256"]
            == cohort_by_id[str(artifact["state_id"])]["source_trace_sha256"]
            for artifact in artifacts
        ),
        "initial_attempt_parity": all(
            len(
                {
                    str(artifact["policy"]["attempts"][0]["attempt_signature"])
                    for artifact in group
                }
            )
            == 1
            for group in grouped.values()
        ),
        "rollback_integrity": all(
            not attempt["rolled_back"]
            or (
                attempt["after_repair_fingerprint"]
                == artifact["before_repair_fingerprint"]
                and int(attempt["conflicts_after"])
                == int(artifact["before_conflicts"])
            )
            for artifact in artifacts
            for attempt in artifact["policy"]["attempts"]
        ),
        "no_runtime_or_future_fields": all(
            artifact["runtime_fields_stored"] is False
            and artifact["future_trajectory_stored"] is False
            for artifact in artifacts
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
    by_map = {policy: _group_summary(rows, policy, "map_id") for policy in POLICIES}
    by_half = {policy: _group_summary(rows, policy, "seed_half") for policy in POLICIES}
    by_source = {
        policy: _group_summary(rows, policy, "source_policy") for policy in POLICIES
    }
    by_split = {
        policy: _group_summary(rows, policy, "research_split") for policy in POLICIES
    }
    by_conflict = {
        policy: _group_summary(rows, policy, "conflict_band") for policy in POLICIES
    }
    primary = summaries[PRIMARY_POLICY]
    baseline = summaries[BASELINE_POLICY]
    set_only = summaries[SET_POLICY]
    gates_config = dict(config["readiness_gates"])
    gates = {
        "integrity": all(integrity.values()),
        "result_blind_cohort": _read_json(inputs["cohort_report"])["passed"] is True,
        "zero_errors_and_timeouts": int(status["error_job_count"]) == 0
        and int(status["timeout_job_count"]) == 0,
        "overall_unchanged_reduction": float(baseline["returned_unchanged_rate"])
        - float(primary["returned_unchanged_rate"])
        >= float(gates_config["minimum_returned_unchanged_rate_reduction"]),
        "overall_success_improvement": float(primary["replan_success_rate"])
        - float(baseline["replan_success_rate"])
        >= float(gates_config["minimum_replan_success_rate_improvement"]),
        "strict_reduction_not_worse": float(primary["strict_conflict_reduction_rate"])
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
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity": integrity,
        "integrity_passed": all(integrity.values()),
        "confirmation_passed": passed,
        "state_count": len(cohort),
        "state_trial_count": len(grouped),
        "policy_row_count": len(rows),
        "policy_summaries": summaries,
        "by_map": by_map,
        "by_seed_half": by_half,
        "by_source_policy": by_source,
        "by_research_split": by_split,
        "by_conflict_band": by_conflict,
        "confirmation_gates": gates,
        "execution": {
            "schema": EXECUTION_SCHEMA,
            "worker_limit": int(status["worker_limit"]),
            "worker_process_count": int(status["worker_process_count"]),
            "final_32_completed_job_worker_process_count": int(
                status["final_32_completed_job_worker_process_count"]
            ),
            "task_granularity": "state_x_trial_index_x_policy",
            "per_job_timeout_seconds": float(
                config["execution"]["per_state_trial_policy_timeout_seconds"]
            ),
            "tail_parallelism_observed": bool(status["tail_parallelism_observed"]),
        },
        "config_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in sorted(inputs.items())},
        "claim_boundary": dict(config["claim_boundary"]),
        "next_step": (
            "preregister runtime semantic integration validation; TTF remains forbidden"
            if passed
            else "stop TransactionalRepair without tuning or cohort filtering"
        ),
        "producer": producer_identity(
            project_root=root,
            source_files=PRODUCER_FILES,
            native_required=True,
            package_names=(),
        ),
    }
    _write_json(output / "transactionalrepair_confirmation_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "COHORT_REPORT_SCHEMA",
    "COHORT_ROW_SCHEMA",
    "EXPERIMENT_ID",
    "analyze_confirmation",
    "collect_confirmation",
    "load_confirmation_registration",
    "prepare_confirmation_cohort",
]
