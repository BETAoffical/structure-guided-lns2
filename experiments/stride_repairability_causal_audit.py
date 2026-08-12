from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

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
from experiments.stride_collection import _validate_native_repair
from experiments.stride_marginalpool_action_replay import (
    _replay_job,
    build_frozen_cohort,
)
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT, restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


CONFIG_SCHEMA = "lns2.stride.repairability_causal_audit_registration.v1"
RUN_SCHEMA = "lns2.stride.repairability_causal_audit_run.v1"
STATE_SCHEMA = "lns2.stride.repairability_causal_audit_state.v1"
PARTIAL_SCHEMA = "lns2.stride.repairability_causal_audit_partial_state.v1"
TRIAL_SCHEMA = "lns2.stride.repairability_causal_trial.v1"
REPORT_SCHEMA = "lns2.stride.repairability_causal_audit_report.v1"
PREFLIGHT_SCHEMA = "lns2.stride.repairability_worker_preflight.v1"
EXPERIMENT_ID = "stride-repairability-causal-audit-v1"
ARMS = (
    "selected_native_order",
    "selected_reverse_order",
    "selected_conflict_priority",
    "blocker_augmented_tail",
    "blocker_augmented_head",
)
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/stride_repairability_causal_audit.py",
    "scripts/run_stride_repairability_causal_audit.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _mean(values: Iterable[float | int]) -> float:
    rows = [float(value) for value in values]
    return float(statistics.fmean(rows)) if rows else 0.0


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    config_path = Path(config_path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
    ):
        raise ValueError("Repairability causal-audit registration changed")
    inputs = {
        name: registered_input(root, row, label=f"causal audit {name}")
        for name, row in dict(config["inputs"]).items()
    }
    return config_path, root, config, inputs


def build_causal_cohort(
    config_path: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path, _root, config, inputs = load_registration(config_path)
    analysis = _read_json(inputs["action_replay_analysis"])
    root_report = _read_json(inputs["root_diagnostic_report"])
    if (
        analysis.get("integrity_passed") is not True
        or root_report.get("integrity_passed") is not True
    ):
        raise ValueError("Repairability causal-audit source evidence did not pass")
    source_metadata, states, logical_rows = build_frozen_cohort(
        inputs["marginalpool_registration"]
    )
    checkpoint_kind = str(config["cohort"]["checkpoint_kind"])
    selected_logical = [
        row for row in logical_rows if str(row["checkpoint_kind"]) == checkpoint_kind
    ]
    logical_manifest = _read_jsonl(inputs["logical_checkpoint_manifest"])
    if selected_logical != [
        row for row in logical_manifest if str(row["checkpoint_kind"]) == checkpoint_kind
    ]:
        raise ValueError("Repairability logical checkpoint manifest changed")
    logical_by_state = {
        str(row["state_fingerprint"]): row for row in selected_logical
    }
    if len(logical_by_state) != len(selected_logical):
        raise ValueError("Repairability first-repeat states are no longer unique")
    state_by_key = {str(row["state_fingerprint"]): row for row in states}
    cohort: list[dict[str, Any]] = []
    for state_key in sorted(logical_by_state):
        logical = logical_by_state[state_key]
        state = dict(state_by_key[state_key])
        candidate_by_id = {
            str(row["candidate_id"]): dict(row) for row in state["candidates"]
        }
        selected_id = str(logical["selected_candidate_id"])
        if selected_id not in candidate_by_id:
            raise ValueError("Repairability selected candidate disappeared")
        cohort.append(
            {
                **state,
                "logical_checkpoint": logical,
                "selected_candidate": candidate_by_id[selected_id],
            }
        )
    required = dict(config["cohort"])
    if (
        len(selected_logical) != int(required["required_logical_checkpoint_count"])
        or len(cohort) != int(required["required_unique_state_count"])
        or len({str(row["map_id"]) for row in cohort})
        != int(required["required_map_count"])
    ):
        raise ValueError("Repairability causal-audit frozen cohort changed")
    metadata = {
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
                    "logical_checkpoint": row["logical_checkpoint"],
                    "selected_candidate": row["selected_candidate"],
                }
                for row in cohort
            ]
        ),
    }
    return metadata, cohort


def conflict_priority_order(state: dict[str, Any], agents: list[int]) -> list[int]:
    degrees = {
        int(row["id"]): int(row["conflict_degree"]) for row in state["agents"]
    }
    return sorted(map(int, agents), key=lambda agent: (-degrees[agent], agent))


def external_blocker_order(
    diagnostic_rows: list[dict[str, Any]], base_agents: list[int], *, maximum: int
) -> list[int]:
    base = set(map(int, base_agents))
    first_seen: dict[int, int] = {}
    for row in diagnostic_rows:
        order_index = int(row["order_index"])
        for blocker in map(int, row["external_blocker_agents"]):
            if blocker not in base:
                first_seen[blocker] = min(first_seen.get(blocker, order_index), order_index)
    return [
        agent
        for agent, _index in sorted(first_seen.items(), key=lambda item: (item[1], item[0]))
    ][: int(maximum)]


def _diagnostic_action(
    agents: list[int], pp_seed: int, *, repair_order: list[int] | None = None
) -> dict[str, Any]:
    action: dict[str, Any] = {
        "mode": "explicit_neighborhood",
        "agents": list(map(int, agents)),
        "random_seed": int(pp_seed),
        "pp_random_seed": int(pp_seed),
        "collect_pp_diagnostics": True,
    }
    if repair_order is not None:
        action["repair_order"] = list(map(int, repair_order))
    return action


def _validate_pp_diagnostic(metrics: dict[str, Any]) -> dict[str, Any]:
    if metrics.get("requested_collect_pp_diagnostics") is not True:
        raise RuntimeError("native PP diagnostic was not requested")
    reason = str(metrics.get("pp_failure_reason"))
    rows = metrics.get("pp_agent_diagnostics")
    if reason not in {"none", "conflict_bound_exceeded", "time_limit"}:
        raise RuntimeError(f"native PP diagnostic has invalid reason: {reason}")
    if (
        not isinstance(rows, list)
        or len(rows) != int(metrics.get("pp_attempted_agent_count", -1))
    ):
        raise RuntimeError("native PP diagnostic attempted-agent count changed")
    repair_order = list(map(int, metrics["repair_order"]))
    neighborhood = set(map(int, metrics["neighborhood"]))
    clean_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        agent = int(row["agent_id"])
        if int(row["order_index"]) != index or repair_order[index] != agent:
            raise RuntimeError("native PP diagnostic order changed")
        internal = list(map(int, row["internal_blocker_agents"]))
        external = list(map(int, row["external_blocker_agents"]))
        if not set(internal) <= neighborhood or set(external) & neighborhood:
            raise RuntimeError("native PP blocker partition changed")
        clean_rows.append(
            {
                "agent_id": agent,
                "order_index": index,
                "path_cost_before": int(row["path_cost_before"]),
                "path_cost_after": int(row["path_cost_after"]),
                "path_changed": bool(row["path_changed"]),
                "low_level_collision_count": int(row["low_level_collision_count"]),
                "cumulative_conflict_pair_count": int(
                    row["cumulative_conflict_pair_count"]
                ),
                "inserted_into_path_table": bool(row["inserted_into_path_table"]),
                "new_conflict_pairs": [
                    list(map(int, pair)) for pair in row["new_conflict_pairs"]
                ],
                "internal_blocker_agents": internal,
                "external_blocker_agents": external,
            }
        )
    return {
        "schema": "lns2.pp_repair_diagnostic.v1",
        "failure_reason": reason,
        "attempted_agent_count": int(metrics["pp_attempted_agent_count"]),
        "inserted_agent_count": int(metrics["pp_inserted_agent_count"]),
        "failed_agent": int(metrics["pp_failed_agent"]),
        "failed_order_index": int(metrics["pp_failed_order_index"]),
        "old_conflict_pair_count": int(metrics["pp_old_conflict_pair_count"]),
        "attempt_conflict_pair_count": int(
            metrics["pp_attempt_conflict_pair_count"]
        ),
        "rolled_back": bool(metrics["pp_rolled_back"]),
        "agents": clean_rows,
    }


def _load_restored_source(
    state_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str, int, int]:
    blob_path = Path(str(state_record["state_blob"]))
    if sha256_file(blob_path) != str(state_record["state_blob_sha256"]):
        raise RuntimeError("Repairability causal state blob changed")
    state = read_state_blob(blob_path)
    state["context"] = dict(state_record["state_context"])
    state_key = str(state_record["state_fingerprint"])
    if state_fingerprint(state) != state_key:
        raise RuntimeError("Repairability causal state fingerprint changed")
    before_repair = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    restore_seed = repairability_restore_seed(before_repair)
    return state, _replay_job(state_record), before_repair, before_conflicts, restore_seed


def _run_arm(
    *,
    replay: dict[str, Any],
    state: dict[str, Any],
    before_repair: str,
    before_conflicts: int,
    restore_seed: int,
    state_key: str,
    trial_index: int,
    pp_seed: int,
    arm: str,
    agents: list[int],
    repair_order: list[int] | None,
) -> dict[str, Any]:
    environment, restored = restore_repair_state(replay, state, seed=restore_seed)
    if repair_structure_fingerprint(restored) != before_repair:
        raise RuntimeError("Repairability causal branch restore changed")
    result = _plain(
        environment.step(
            _diagnostic_action(agents, pp_seed, repair_order=repair_order)
        )
    )
    after, metrics = _validate_native_repair(
        result, expected_agents=agents, expected_seed=pp_seed
    )
    diagnostic = _validate_pp_diagnostic(metrics)
    conflicts_after = int(after["num_of_colliding_pairs"])
    after_repair = repair_structure_fingerprint(after)
    return {
        "schema": TRIAL_SCHEMA,
        "state_fingerprint": state_key,
        "trial_index": int(trial_index),
        "pp_seed": int(pp_seed),
        "arm": arm,
        "applicable": True,
        "agents": list(map(int, agents)),
        "repair_order": list(map(int, metrics["repair_order"])),
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
        "pp_diagnostic": diagnostic,
    }


def _not_applicable_row(
    *, state_key: str, trial_index: int, pp_seed: int, arm: str, reason: str
) -> dict[str, Any]:
    return {
        "schema": TRIAL_SCHEMA,
        "state_fingerprint": state_key,
        "trial_index": int(trial_index),
        "pp_seed": int(pp_seed),
        "arm": arm,
        "applicable": False,
        "not_applicable_reason": reason,
    }


def _write_partial(
    path: Path,
    *,
    run_fingerprint: str,
    state_record: dict[str, Any],
    before_repair: str,
    before_conflicts: int,
    rows: list[dict[str, Any]],
) -> None:
    _write_json(
        path,
        {
            "schema": PARTIAL_SCHEMA,
            "complete": False,
            "run_fingerprint": run_fingerprint,
            "state_fingerprint": str(state_record["state_fingerprint"]),
            "state_blob_sha256": str(state_record["state_blob_sha256"]),
            "before_repair_fingerprint": before_repair,
            "before_conflicts": before_conflicts,
            "trials": rows,
            "runtime_fields_stored": False,
            "future_trajectory_stored": False,
        },
    )


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    state_record = dict(job["state_record"])
    state_key = str(state_record["state_fingerprint"])
    output_path = Path(str(job["output_path"]))
    partial_path = output_path.with_name(output_path.name + ".partial")
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if (
            existing.get("schema") == STATE_SCHEMA
            and existing.get("complete") is True
            and existing.get("run_fingerprint") == run_fingerprint
            and existing.get("state_fingerprint") == state_key
        ):
            return {
                "state_fingerprint": state_key,
                "status": "resumed",
                "state_count": 1,
                "trial_count": len(existing["trials"]),
                "executed_trial_count": sum(
                    bool(row["applicable"]) for row in existing["trials"]
                ),
                "error_count": 0,
            }
        raise ValueError("invalid completed Repairability causal state")

    state, replay, before_repair, before_conflicts, restore_seed = (
        _load_restored_source(state_record)
    )
    selected = dict(state_record["selected_candidate"])
    base_agents = list(map(int, selected["agents"]))
    conflict_order = conflict_priority_order(state, base_agents)
    rows: list[dict[str, Any]] = []
    if bool(job["resume"]) and partial_path.is_file():
        partial = _read_json(partial_path)
        if (
            partial.get("schema") != PARTIAL_SCHEMA
            or partial.get("run_fingerprint") != run_fingerprint
            or partial.get("state_fingerprint") != state_key
            or partial.get("before_repair_fingerprint") != before_repair
        ):
            raise ValueError("invalid partial Repairability causal state")
        rows = list(partial["trials"])
    row_by_key = {
        (int(row["trial_index"]), str(row["arm"])): row for row in rows
    }
    maximum_blockers = int(job["maximum_added_external_blockers"])
    trial_indices = tuple(map(int, job["trial_indices"]))
    for trial_index in trial_indices:
        pp_seed = repairability_pp_seed(before_repair, trial_index)
        baseline_key = (trial_index, "selected_native_order")
        baseline = row_by_key.get(baseline_key)
        if baseline is None:
            baseline = _run_arm(
                replay=replay,
                state=state,
                before_repair=before_repair,
                before_conflicts=before_conflicts,
                restore_seed=restore_seed,
                state_key=state_key,
                trial_index=trial_index,
                pp_seed=pp_seed,
                arm="selected_native_order",
                agents=base_agents,
                repair_order=None,
            )
            rows.append(baseline)
            row_by_key[baseline_key] = baseline
            _write_partial(
                partial_path,
                run_fingerprint=run_fingerprint,
                state_record=state_record,
                before_repair=before_repair,
                before_conflicts=before_conflicts,
                rows=rows,
            )
        base_order = list(map(int, baseline["repair_order"]))
        variants: list[tuple[str, list[int], list[int]]] = [
            ("selected_reverse_order", base_agents, list(reversed(base_order))),
            ("selected_conflict_priority", base_agents, conflict_order),
        ]
        blockers = external_blocker_order(
            list(baseline["pp_diagnostic"]["agents"]),
            base_agents,
            maximum=maximum_blockers,
        )
        if blockers:
            augmented = base_agents + blockers
            variants.extend(
                [
                    ("blocker_augmented_tail", augmented, base_order + blockers),
                    ("blocker_augmented_head", augmented, blockers + base_order),
                ]
            )
        else:
            for arm in ("blocker_augmented_tail", "blocker_augmented_head"):
                key = (trial_index, arm)
                if key not in row_by_key:
                    row = _not_applicable_row(
                        state_key=state_key,
                        trial_index=trial_index,
                        pp_seed=pp_seed,
                        arm=arm,
                        reason="base_attempt_exposed_no_external_blocker",
                    )
                    rows.append(row)
                    row_by_key[key] = row
                    _write_partial(
                        partial_path,
                        run_fingerprint=run_fingerprint,
                        state_record=state_record,
                        before_repair=before_repair,
                        before_conflicts=before_conflicts,
                        rows=rows,
                    )
        for arm, agents, order in variants:
            key = (trial_index, arm)
            if key in row_by_key:
                continue
            row = _run_arm(
                replay=replay,
                state=state,
                before_repair=before_repair,
                before_conflicts=before_conflicts,
                restore_seed=restore_seed,
                state_key=state_key,
                trial_index=trial_index,
                pp_seed=pp_seed,
                arm=arm,
                agents=agents,
                repair_order=order,
            )
            rows.append(row)
            row_by_key[key] = row
            _write_partial(
                partial_path,
                run_fingerprint=run_fingerprint,
                state_record=state_record,
                before_repair=before_repair,
                before_conflicts=before_conflicts,
                rows=rows,
            )
    rows.sort(key=lambda row: (int(row["trial_index"]), ARMS.index(str(row["arm"]))))
    expected_keys = {(trial, arm) for trial in trial_indices for arm in ARMS}
    if set(row_by_key) != expected_keys:
        raise RuntimeError("Repairability causal state trial grid is incomplete")
    payload = {
        "schema": STATE_SCHEMA,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "state_fingerprint": state_key,
        "state_blob_sha256": str(state_record["state_blob_sha256"]),
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "logical_checkpoint": dict(state_record["logical_checkpoint"]),
        "selected_candidate": selected,
        "state_restore": {
            "contract": TARGET_STATE_RESTORE_CONTRACT,
            "restore_seed": restore_seed,
        },
        "trials": rows,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, payload)
    partial_path.unlink(missing_ok=True)
    return {
        "state_fingerprint": state_key,
        "status": "ok",
        "state_count": 1,
        "trial_count": len(rows),
        "executed_trial_count": sum(bool(row["applicable"]) for row in rows),
        "error_count": 0,
    }


def _probe_state(job: dict[str, Any]) -> dict[str, Any]:
    state_record = dict(job["state_record"])
    state, replay, before_repair, before_conflicts, restore_seed = (
        _load_restored_source(state_record)
    )
    selected = dict(state_record["selected_candidate"])
    agents = list(map(int, selected["agents"]))
    trial_index = int(job["trial_index"])
    pp_seed = repairability_pp_seed(before_repair, trial_index)
    row = _run_arm(
        replay=replay,
        state=state,
        before_repair=before_repair,
        before_conflicts=before_conflicts,
        restore_seed=restore_seed,
        state_key=str(state_record["state_fingerprint"]),
        trial_index=trial_index,
        pp_seed=pp_seed,
        arm="selected_native_order",
        agents=agents,
        repair_order=None,
    )
    return {
        "state_fingerprint": str(state_record["state_fingerprint"]),
        "status": "ok",
        "state_count": 1,
        "trial_count": 1,
        "executed_trial_count": 1,
        "error_count": 0,
        "integrity_fingerprint": _fingerprint(row),
    }


def _failed_result(job: dict[str, Any], status: str, message: str) -> dict[str, Any]:
    return {
        "state_fingerprint": str(job["state_record"]["state_fingerprint"]),
        "status": status,
        "state_count": 0,
        "trial_count": 0,
        "executed_trial_count": 0,
        "error_count": 1,
        "error": message,
    }


def _memory_sampler(stop: threading.Event, samples: list[float]) -> None:
    try:
        import psutil
    except ImportError:
        return
    while not stop.wait(0.05):
        samples.append(float(psutil.virtual_memory().percent) / 100.0)


def run_worker_preflight(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    metadata, cohort = build_causal_cohort(config_path)
    _config_path, root, config, _inputs = load_registration(config_path)
    selected = cohort[:16]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("psutil",),
    )
    rows: list[dict[str, Any]] = []
    for workers in map(int, config["worker_preflight"]["worker_counts"]):
        jobs = [
            {
                "job_id": str(row["state_fingerprint"]),
                "state_record": row,
                "trial_index": int(config["worker_preflight"]["trial_indices"][0]),
            }
            for row in selected
        ]
        samples: list[float] = []
        stop = threading.Event()
        sampler = threading.Thread(
            target=_memory_sampler, args=(stop, samples), daemon=True
        )
        sampler.start()
        started = time.perf_counter()
        results = _run_jobs(
            _probe_state,
            jobs,
            workers,
            phase=f"repairability-worker-preflight-{workers}",
            output_root=output / f"workers-{workers}",
            run_fingerprint=metadata["cohort_fingerprint"],
            timeout_seconds=float(
                config["execution"]["per_state_attempt_timeout_seconds"]
            ),
            failure_result=_failed_result,
        )
        elapsed = time.perf_counter() - started
        stop.set()
        sampler.join(timeout=1.0)
        completed = sum(row["status"] == "ok" for row in results)
        peak_memory = max(samples) if samples else 0.0
        safe = (
            completed == len(jobs)
            and not any(int(row["error_count"]) for row in results)
            and peak_memory
            <= float(config["worker_preflight"]["maximum_memory_fraction"])
        )
        rows.append(
            {
                "workers": workers,
                "job_count": len(jobs),
                "completed_job_count": completed,
                "elapsed_seconds": elapsed,
                "throughput_jobs_per_second": completed / max(elapsed, 1e-9),
                "peak_system_memory_fraction": peak_memory,
                "safe": safe,
                "result_fingerprint": _fingerprint(
                    sorted(
                        (row["state_fingerprint"], row.get("integrity_fingerprint"))
                        for row in results
                    )
                ),
                "errors": [row for row in results if row["status"] != "ok"],
            }
        )
    fingerprints = {row["result_fingerprint"] for row in rows if row["safe"]}
    if len(fingerprints) != 1:
        raise RuntimeError("worker preflight changed scientific outputs")
    safe_rows = [row for row in rows if row["safe"]]
    if not safe_rows:
        raise RuntimeError("no safe worker count passed preflight")
    selected_workers = int(safe_rows[0]["workers"])
    prior = safe_rows[0]
    for row in safe_rows[1:]:
        if float(row["throughput_jobs_per_second"]) > float(
            prior["throughput_jobs_per_second"]
        ):
            selected_workers = int(row["workers"])
        prior = row
    report = {
        "schema": PREFLIGHT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "complete": True,
        "integrity_passed": True,
        "config_sha256": metadata["config_sha256"],
        "cohort_fingerprint": metadata["cohort_fingerprint"],
        "selected_state_fingerprints": [
            str(row["state_fingerprint"]) for row in selected
        ],
        "results": rows,
        "selected_worker_count": selected_workers,
        "outcomes_used_for_protocol_changes": False,
        "producer": producer,
    }
    _write_json(output / "worker_preflight_report.json", report)
    return report


def collect_causal_audit(
    *,
    config_path: str | Path,
    preflight: str | Path,
    output: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    metadata, cohort = build_causal_cohort(config_path)
    _config_path, root, config, _inputs = load_registration(config_path)
    preflight_path = Path(preflight).resolve()
    report = _read_json(preflight_path)
    if (
        report.get("schema") != PREFLIGHT_SCHEMA
        or report.get("complete") is not True
        or report.get("integrity_passed") is not True
        or report.get("config_sha256") != metadata["config_sha256"]
        or report.get("cohort_fingerprint") != metadata["cohort_fingerprint"]
    ):
        raise ValueError("Repairability worker preflight changed")
    workers = int(report["selected_worker_count"])
    if workers > int(config["execution"]["maximum_workers"]):
        raise ValueError("Repairability worker preflight exceeds registered maximum")
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
        "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_file(preflight_path),
        "selected_worker_count": workers,
        "trial_indices": list(map(int, config["interventions"]["trial_indices"])),
        "arms": list(ARMS),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(run_identity)
    output = Path(output).resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("Repairability causal output belongs to another run")
        if not resume:
            raise ValueError("Repairability causal output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**run_identity, "run_fingerprint": run_fingerprint})
    jobs = [
        {
            "job_id": str(row["state_fingerprint"]),
            "state_record": row,
            "output_path": str(
                output / "states" / f"{row['state_fingerprint']}.json"
            ),
            "run_fingerprint": run_fingerprint,
            "trial_indices": list(map(int, config["interventions"]["trial_indices"])),
            "maximum_added_external_blockers": int(
                config["interventions"]["maximum_added_external_blockers"]
            ),
            "resume": bool(resume),
        }
        for row in cohort
    ]
    status_path = output / "collection_status.json"
    observed: dict[str, dict[str, Any]] = {}
    pending = jobs
    attempts: list[dict[str, Any]] = []
    prior_progress: dict[str, int] = {}
    no_progress: dict[str, int] = defaultdict(int)
    maximum_attempts = int(config["execution"]["maximum_state_attempts"])

    def write_status(status: str, attempt: int) -> None:
        successful = [row for row in observed.values() if row["status"] in {"ok", "resumed"}]
        failures = [row for row in observed.values() if row["status"] not in {"ok", "resumed"}]
        _write_json(
            status_path,
            {
                "schema": RUN_SCHEMA,
                "run_fingerprint": run_fingerprint,
                "status": status,
                "requested_state_count": len(jobs),
                "completed_state_count": len(successful),
                "completed_trial_row_count": sum(
                    int(row["trial_count"]) for row in successful
                ),
                "completed_executed_trial_count": sum(
                    int(row["executed_trial_count"]) for row in successful
                ),
                "error_state_count": sum(row["status"] == "error" for row in failures),
                "timeout_state_count": sum(row["status"] == "timeout" for row in failures),
                "active_jobs": [],
                "current_attempt": attempt,
                "maximum_state_attempts": maximum_attempts,
                "attempt_history": attempts,
                "selected_worker_count": workers,
                "errors": failures,
            },
        )

    write_status("running", 0)
    for attempt in range(1, maximum_attempts + 1):
        attempt_results: dict[str, dict[str, Any]] = {}

        def observe(row: dict[str, Any]) -> None:
            key = str(row["state_fingerprint"])
            attempt_results[key] = row
            observed[key] = row
            write_status("running", attempt)

        _run_jobs(
            _collect_state,
            pending,
            workers,
            phase=f"repairability-causal-attempt-{attempt}",
            output_root=output,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(
                config["execution"]["per_state_attempt_timeout_seconds"]
            ),
            on_result=observe,
            failure_result=_failed_result,
        )
        retry: list[dict[str, Any]] = []
        for job in pending:
            key = str(job["state_record"]["state_fingerprint"])
            result = attempt_results[key]
            if result["status"] in {"ok", "resumed"}:
                continue
            partial = Path(str(job["output_path"])).with_name(
                Path(str(job["output_path"])).name + ".partial"
            )
            progress = len(_read_json(partial).get("trials") or ()) if partial.is_file() else 0
            no_progress[key] = no_progress[key] + 1 if progress <= prior_progress.get(key, 0) else 0
            prior_progress[key] = progress
            attempts.append(
                {
                    "attempt": attempt,
                    "state_fingerprint": key,
                    "status": result["status"],
                    "checkpoint_trial_row_count": progress,
                    "consecutive_no_progress_attempts": no_progress[key],
                }
            )
            if no_progress[key] < int(
                config["execution"]["stop_after_consecutive_no_progress_attempts"]
            ):
                job["resume"] = True
                retry.append(job)
        pending = retry
        if not pending:
            break
    completed = sum(row["status"] in {"ok", "resumed"} for row in observed.values())
    final_status = "complete" if completed == len(jobs) else "failed"
    write_status(final_status, min(maximum_attempts, max(1, len(attempts) + 1)))
    return _read_json(status_path)


def stable_effect(
    reference: list[dict[str, Any]], treatment: list[dict[str, Any]], *, minimum: float
) -> dict[str, Any]:
    ref = {int(row["trial_index"]): row for row in reference if row["applicable"]}
    treat = {int(row["trial_index"]): row for row in treatment if row["applicable"]}
    indices = sorted(set(ref) & set(treat))
    deltas = [
        float(treat[index]["normalized_conflict_reduction"])
        - float(ref[index]["normalized_conflict_reduction"])
        for index in indices
    ]
    first = [value for index, value in zip(indices, deltas) if index < 8]
    second = [value for index, value in zip(indices, deltas) if index >= 8]
    no_progress_delta = _mean(
        int(treat[index]["no_progress"]) - int(ref[index]["no_progress"])
        for index in indices
    )
    passed = (
        indices == list(range(16))
        and _mean(deltas) >= float(minimum) - 1e-15
        and no_progress_delta <= 1e-15
        and _mean(first) > 1e-15
        and _mean(second) > 1e-15
    )
    return {
        "paired_trial_count": len(indices),
        "mean_normalized_reduction_delta": _mean(deltas),
        "first_half_mean_delta": _mean(first),
        "second_half_mean_delta": _mean(second),
        "no_progress_rate_delta": no_progress_delta,
        "stable_effect_passed": passed,
    }


def analyze_causal_audit(
    *, config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    metadata, cohort = build_causal_cohort(config_path)
    _config_path, _root, config, _inputs = load_registration(config_path)
    collection = Path(collection).resolve()
    status = _read_json(collection / "collection_status.json")
    run = _read_json(collection / "run_config.json")
    if (
        status.get("status") != "complete"
        or int(status.get("completed_state_count", -1)) != len(cohort)
        or run.get("cohort_fingerprint") != metadata["cohort_fingerprint"]
    ):
        raise ValueError("Repairability causal collection is incomplete")
    minimum = float(
        config["stable_effect"][
            "minimum_mean_normalized_conflict_reduction_advantage"
        ]
    )
    state_effects: list[dict[str, Any]] = []
    all_trials: list[dict[str, Any]] = []
    for state_record in cohort:
        state_key = str(state_record["state_fingerprint"])
        state_path = collection / "states" / f"{state_key}.json"
        payload = _read_json(state_path)
        if (
            payload.get("schema") != STATE_SCHEMA
            or payload.get("complete") is not True
            or payload.get("run_fingerprint") != run["run_fingerprint"]
            or payload.get("runtime_fields_stored") is not False
            or payload.get("future_trajectory_stored") is not False
            or sha256_file(Path(str(state_record["state_blob"])))
            != str(payload["state_blob_sha256"])
        ):
            raise ValueError(f"invalid Repairability causal state: {state_key}")
        trials = list(payload["trials"])
        if len(trials) != 16 * len(ARMS):
            raise ValueError("Repairability causal trial grid changed")
        all_trials.extend(trials)
        by_arm = {
            arm: [row for row in trials if str(row["arm"]) == arm] for arm in ARMS
        }
        reverse = stable_effect(
            by_arm["selected_native_order"],
            by_arm["selected_reverse_order"],
            minimum=minimum,
        )
        priority = stable_effect(
            by_arm["selected_native_order"],
            by_arm["selected_conflict_priority"],
            minimum=minimum,
        )
        set_effect = stable_effect(
            by_arm["selected_native_order"],
            by_arm["blocker_augmented_tail"],
            minimum=minimum,
        )
        joint_vs_tail = stable_effect(
            by_arm["blocker_augmented_tail"],
            by_arm["blocker_augmented_head"],
            minimum=minimum,
        )
        joint_vs_base = stable_effect(
            by_arm["selected_native_order"],
            by_arm["blocker_augmented_head"],
            minimum=minimum,
        )
        order_passed = bool(
            reverse["stable_effect_passed"] or priority["stable_effect_passed"]
        )
        set_passed = bool(set_effect["stable_effect_passed"])
        joint_passed = bool(
            joint_vs_base["stable_effect_passed"]
            and joint_vs_tail["stable_effect_passed"]
        )
        if joint_passed and not set_passed:
            root = "set_and_order_joint"
        elif set_passed:
            root = "set_defect"
        elif order_passed:
            root = "order_defect"
        else:
            root = "residual_pp_instability"
        baseline = [row for row in by_arm["selected_native_order"] if row["applicable"]]
        state_effects.append(
            {
                "schema": "lns2.stride.repairability_causal_state_effect.v1",
                "state_fingerprint": state_key,
                "map_id": str(state_record["map_id"]),
                "task_id": str(state_record["task_id"]),
                "solver_seed": int(state_record["solver_seed"]),
                "root_cause": root,
                "baseline_replan_success_rate": _mean(
                    int(row["replan_success"]) for row in baseline
                ),
                "baseline_failure_reasons": dict(
                    Counter(
                        str(row["pp_diagnostic"]["failure_reason"])
                        for row in baseline
                    )
                ),
                "baseline_mean_failed_order_index": _mean(
                    int(row["pp_diagnostic"]["failed_order_index"])
                    for row in baseline
                    if int(row["pp_diagnostic"]["failed_order_index"]) >= 0
                ),
                "baseline_mean_external_blocker_count": _mean(
                    len(
                        {
                            blocker
                            for agent in row["pp_diagnostic"]["agents"]
                            for blocker in agent["external_blocker_agents"]
                        }
                    )
                    for row in baseline
                ),
                "effects": {
                    "reverse_order": reverse,
                    "conflict_priority_order": priority,
                    "blocker_augmented_tail": set_effect,
                    "blocker_head_vs_tail": joint_vs_tail,
                    "blocker_head_vs_base": joint_vs_base,
                },
            }
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "state_effects.jsonl", state_effects)
    root_counts = Counter(str(row["root_cause"]) for row in state_effects)
    by_map: dict[str, Counter[str]] = defaultdict(Counter)
    for row in state_effects:
        by_map[str(row["map_id"])][str(row["root_cause"])] += 1
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": True,
        "state_count": len(state_effects),
        "trial_row_count": len(all_trials),
        "executed_trial_count": sum(bool(row["applicable"]) for row in all_trials),
        "not_applicable_trial_count": sum(
            not bool(row["applicable"]) for row in all_trials
        ),
        "root_cause_counts": dict(root_counts),
        "root_cause_fraction": {
            key: value / len(state_effects) for key, value in sorted(root_counts.items())
        },
        "by_map": {
            map_id: dict(counts) for map_id, counts in sorted(by_map.items())
        },
        "mean_baseline_replan_success_rate": _mean(
            row["baseline_replan_success_rate"] for row in state_effects
        ),
        "mean_baseline_failed_order_index": _mean(
            row["baseline_mean_failed_order_index"] for row in state_effects
        ),
        "mean_baseline_external_blocker_count": _mean(
            row["baseline_mean_external_blocker_count"] for row in state_effects
        ),
        "state_effects_sha256": sha256_file(output / "state_effects.jsonl"),
        "claim_boundary": dict(config["claim_boundary"]),
        "next_step": (
            "preregister an outcome-blind set-and-order repairability policy using only "
            "the dominant causal mechanism; no training or TTF is authorized by this audit"
        ),
    }
    _write_json(output / "repairability_causal_report.json", report)
    return report
