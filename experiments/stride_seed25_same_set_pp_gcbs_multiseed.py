from __future__ import annotations

import hashlib
import multiprocessing
import statistics
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _plain, _read_json, _write_json
from experiments.stride_collection import (
    _paired_action,
    _validate_native_repair,
    stride_pp_seed,
)
from experiments.stride_repairability_collection import repairability_restore_seed
from experiments.stride_seed25_same_set_pp_gcbs_screen import (
    ALGORITHMS,
    _one_manifest,
    _replay_job,
    build_plan as build_source_plan,
)
from experiments.trace_replay import restore_repair_state, target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.seed25_same_set_pp_gcbs_multiseed_registration.v1"
PLAN_SCHEMA = "lns2.stride.seed25_same_set_pp_gcbs_multiseed_plan.v1"
BRANCH_SCHEMA = "lns2.stride.seed25_same_set_pp_gcbs_multiseed_branch.v1"
REPORT_SCHEMA = "lns2.stride.seed25_same_set_pp_gcbs_multiseed_report.v1"
QUALIFICATION_SCHEMA = "lns2.stride.gcbs_cooperative_deadline_qualification.v1"
EXPERIMENT_ID = "stride-seed25-same-set-pp-gcbs-multiseed-v1"
MAX_ACTION_OVERSHOOT_SECONDS = 0.25


def _resolve(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def _trial_seed(namespace: str, state: dict[str, Any], trial_index: int) -> int:
    if namespace == "stride-lns-paired-pp-v1":
        return stride_pp_seed(str(state["repair_structure_fingerprint"]), trial_index)
    material = (
        f"{namespace}|{state['map_id']}|{state['decision_index']}|{trial_index}"
    ).encode("utf-8")
    return 1 + (int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % 2_147_483_646)


def load_registration(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    project_root = path.parent.parent
    config = _read_json(path)
    source = dict(config.get("source") or {})
    execution = dict(config.get("execution") or {})
    qualification = dict(config.get("qualification") or {})
    claims = dict(config.get("claim_boundary") or {})
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or int(execution.get("trial_count_per_state_algorithm", -1)) != 4
        or tuple(execution.get("repair_algorithms") or ()) != ALGORITHMS
        or execution.get("repair_seed_namespace") != "stride-lns-paired-pp-v1"
        or float(execution.get("action_time_limit_seconds", -1.0)) != 5.0
        or float(execution.get("maximum_action_deadline_overshoot_seconds", -1.0))
        != MAX_ACTION_OVERSHOOT_SECONDS
        or float(execution.get("pair_process_fuse_seconds", -1.0)) != 25.0
        or execution.get("strict_serial_pairs") is not True
        or any(
            execution.get(name) is not False
            for name in ("candidate_generation", "controller_selection", "continuation", "ttf")
        )
        or qualification.get("gcbs_cooperative_action_deadline_validated") is not True
        or qualification.get("gcbs_failure_rollback_validated") is not True
        or qualification.get("gcbs_root_failure_safe") is not True
        or qualification.get("pbs_included") is not False
        or claims.get("mechanism_diagnostic_only") is not True
        or any(
            claims.get(name) is not False
            for name in (
                "repairer_runtime_promotion_allowed",
                "ttf_claim_allowed",
                "selector_replacement_allowed",
                "generalization_claim_allowed",
            )
        )
    ):
        raise ValueError("multiseed same-set PP/GCBS registration contract changed")

    source_registration = _resolve(project_root, source["screen_registration"])
    if sha256_file(source_registration) != str(source["screen_registration_sha256"]):
        raise ValueError("source one-trial registration SHA-256 changed")
    qualification_report = _resolve(project_root, qualification["report_path"])
    if sha256_file(qualification_report) != str(qualification["report_sha256"]):
        raise ValueError("GCBS qualification report SHA-256 changed")
    qualified = _read_json(qualification_report)
    if (
        qualified.get("schema") != QUALIFICATION_SCHEMA
        or qualified.get("status") != "passed"
        or qualified.get("gcbs_cooperative_action_deadline_validated") is not True
        or qualified.get("gcbs_failure_rollback_validated") is not True
        or qualified.get("gcbs_root_failure_safe") is not True
        or dict(qualified.get("producer_sha256") or {})
        != dict(qualification["producer_sha256"])
    ):
        raise ValueError("GCBS qualification evidence is incomplete")
    for relative, expected in sorted(dict(qualification["producer_sha256"]).items()):
        if sha256_file(_resolve(project_root, relative)) != str(expected):
            raise ValueError(f"qualified GCBS producer changed: {relative}")
    return path, project_root, config


def build_plan(config_path: str | Path) -> dict[str, Any]:
    path, project_root, config = load_registration(config_path)
    source_registration = _resolve(project_root, config["source"]["screen_registration"])
    source_plan = build_source_plan(source_registration)
    trial_count = int(config["execution"]["trial_count_per_state_algorithm"])
    namespace = str(config["execution"]["repair_seed_namespace"])
    states: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for state in source_plan["states"]:
        copied = dict(state)
        states.append(copied)
        for trial_index in range(trial_count):
            seed = _trial_seed(namespace, copied, trial_index)
            algorithms = list(ALGORITHMS if (len(pairs) % 2 == 0) else reversed(ALGORITHMS))
            pairs.append(
                {
                    "pair_index": len(pairs),
                    "map_id": str(copied["map_id"]),
                    "trial_index": trial_index,
                    "repair_seed": seed,
                    "algorithm_order": algorithms,
                }
            )
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "source_screen_registration_sha256": sha256_file(source_registration),
        "state_count": len(states),
        "trial_count_per_state_algorithm": trial_count,
        "pair_count": len(pairs),
        "attempt_count": len(pairs) * len(ALGORITHMS),
        "maximum_native_action_budget_seconds": len(pairs)
        * len(ALGORITHMS)
        * float(config["execution"]["action_time_limit_seconds"]),
        "pair_process_fuse_seconds": float(config["execution"]["pair_process_fuse_seconds"]),
        "states": states,
        "pairs": pairs,
        "qualification": dict(config["qualification"]),
        "decision_gate": dict(config["decision_gate"]),
        "claim_boundary": dict(config["claim_boundary"]),
        "solver_or_controller_invoked": False,
    }


def _qualification_environment(
    state_row: dict[str, Any], *, max_repair_iterations: int
) -> tuple[Any, dict[str, Any], dict[str, Any], str]:
    collection = Path(str(state_row["source_collection"]))
    manifest = _one_manifest(collection, solver_seed=int(state_row["solver_seed"]))
    source_state, _trace = target_state_from_trace(
        collection,
        manifest,
        decision_index=int(state_row["decision_index"]),
        expected_fingerprint=str(state_row["state_fingerprint"]),
    )
    before_repair = repair_structure_fingerprint(source_state)
    if before_repair != str(state_row["repair_structure_fingerprint"]):
        raise RuntimeError("qualification source repair structure changed")
    job = _replay_job(state_row, "GCBS", 30.0)
    job["environment"] = dict(
        job["environment"], max_repair_iterations=int(max_repair_iterations)
    )
    environment, before = restore_repair_state(
        job, source_state, seed=repairability_restore_seed(before_repair)
    )
    if repair_structure_fingerprint(before) != before_repair:
        raise RuntimeError("qualification restore changed repair structure")
    return environment, before, source_state, before_repair


def run_qualification(
    source_config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    source_path = Path(source_config_path).resolve()
    source_plan = build_source_plan(source_path)
    by_map = {str(row["map_id"]): row for row in source_plan["states"]}
    timeout_state = by_map["maze-32-32-4-n300"]
    positive_state = by_map["warehouse-w1020a-opposite-exchange"]

    timeout_seed = _trial_seed("gcbs-deadline-qualification-v1", timeout_state, 0)
    probe_seed = _trial_seed("gcbs-deadline-qualification-v1", timeout_state, 1)
    timeout_agents = list(map(int, timeout_state["selected_action"]["agents"]))
    timed_environment, timed_before, _source, before_repair = _qualification_environment(
        timeout_state, max_repair_iterations=3
    )
    timeout_result = _plain(
        timed_environment.step_with_time_limit(
            _paired_action(timeout_agents, timeout_seed), 0.0
        )
    )
    timeout_after, timeout_metrics = _validate_native_repair(
        timeout_result, expected_agents=timeout_agents, expected_seed=timeout_seed
    )
    timeout_after_repair = repair_structure_fingerprint(timeout_after)
    timeout_atomic = (
        timeout_after_repair == before_repair
        and int(timeout_after["num_of_colliding_pairs"])
        == int(timed_before["num_of_colliding_pairs"])
        and not bool(timeout_metrics["replan_success"])
        and bool(timeout_metrics.get("pp_rolled_back"))
        and str(timeout_metrics.get("pp_failure_reason")) == "time_limit"
    )
    if not timeout_atomic:
        raise RuntimeError("zero-budget GCBS did not report an exact TIME_LIMIT rollback")
    if float(timeout_metrics.get("native_replan_seconds", 0.0)) > MAX_ACTION_OVERSHOOT_SECONDS:
        raise RuntimeError("zero-budget GCBS exceeded the deadline tolerance")

    fresh_environment, fresh_before, _source, fresh_before_repair = (
        _qualification_environment(timeout_state, max_repair_iterations=3)
    )
    if fresh_before_repair != before_repair:
        raise RuntimeError("paired qualification restores differ")
    probe_action = _paired_action(timeout_agents, probe_seed)
    timed_probe = _plain(timed_environment.step_with_time_limit(probe_action, 5.0))
    fresh_probe = _plain(fresh_environment.step_with_time_limit(probe_action, 5.0))
    timed_probe_after, timed_probe_metrics = _validate_native_repair(
        timed_probe, expected_agents=timeout_agents, expected_seed=probe_seed
    )
    fresh_probe_after, fresh_probe_metrics = _validate_native_repair(
        fresh_probe, expected_agents=timeout_agents, expected_seed=probe_seed
    )
    next_step_equivalent = (
        repair_structure_fingerprint(timed_probe_after)
        == repair_structure_fingerprint(fresh_probe_after)
        and int(timed_probe_after["num_of_colliding_pairs"])
        == int(fresh_probe_after["num_of_colliding_pairs"])
        and bool(timed_probe_metrics["replan_success"])
        == bool(fresh_probe_metrics["replan_success"])
        and str(timed_probe_metrics.get("pp_failure_reason"))
        == str(fresh_probe_metrics.get("pp_failure_reason"))
    )
    if not next_step_equivalent:
        raise RuntimeError("GCBS rollback changed the following same-seed repair")
    probe_seconds = [
        float(metrics.get("native_replan_seconds", 0.0))
        for metrics in (timed_probe_metrics, fresh_probe_metrics)
    ]
    if any(value > 5.0 + MAX_ACTION_OVERSHOOT_SECONDS for value in probe_seconds):
        raise RuntimeError(
            f"post-timeout GCBS probe exceeded the deadline tolerance: {probe_seconds}"
        )

    positive_seed = _trial_seed("gcbs-deadline-qualification-v1", positive_state, 0)
    positive_agents = list(map(int, positive_state["selected_action"]["agents"]))
    positive_environment, positive_before, _source, _positive_repair = (
        _qualification_environment(positive_state, max_repair_iterations=1)
    )
    positive_result = _plain(
        positive_environment.step_with_time_limit(
            _paired_action(positive_agents, positive_seed), 5.0
        )
    )
    positive_after, positive_metrics = _validate_native_repair(
        positive_result, expected_agents=positive_agents, expected_seed=positive_seed
    )
    positive_drop = int(positive_before["num_of_colliding_pairs"]) - int(
        positive_after["num_of_colliding_pairs"]
    )
    if not bool(positive_metrics["replan_success"]) or positive_drop <= 0:
        raise RuntimeError("qualified GCBS positive control did not strictly improve")
    if (
        float(positive_metrics.get("native_replan_seconds", 0.0))
        > 5.0 + MAX_ACTION_OVERSHOOT_SECONDS
    ):
        raise RuntimeError("qualified GCBS positive control exceeded the deadline tolerance")

    project_root = source_path.parent.parent
    producer_paths = [
        "third_party/mapf_lns2/inc/CBS/GCBS.h",
        "third_party/mapf_lns2/src/CBS/GCBS.cpp",
        "third_party/mapf_lns2/inc/InitLNS.h",
        "third_party/mapf_lns2/src/InitLNS.cpp",
        "third_party/mapf_lns2/inc/RepairPolicy.h",
        "src/python_bindings.cpp",
        "build/linux/project/lns2_env.cpython-310-x86_64-linux-gnu.so",
    ]
    producer_sha256 = {
        relative: sha256_file(_resolve(project_root, relative))
        for relative in producer_paths
    }
    report = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "passed",
        "source_screen_registration": str(source_path),
        "source_screen_registration_sha256": sha256_file(source_path),
        "producer_sha256": producer_sha256,
        "zero_budget_timeout": {
            "map_id": str(timeout_state["map_id"]),
            "repair_seed": timeout_seed,
            "failure_reason": str(timeout_metrics.get("pp_failure_reason")),
            "rolled_back_metric": bool(timeout_metrics.get("pp_rolled_back")),
            "exact_repair_rollback": timeout_atomic,
            "native_replan_seconds": float(timeout_metrics.get("native_replan_seconds", 0.0)),
        },
        "post_timeout_same_seed_probe": {
            "repair_seed": probe_seed,
            "repair_structure_and_outcome_equal_to_fresh_restore": next_step_equivalent,
        },
        "positive_control": {
            "map_id": str(positive_state["map_id"]),
            "repair_seed": positive_seed,
            "conflict_reduction": positive_drop,
            "native_replan_seconds": float(positive_metrics.get("native_replan_seconds", 0.0)),
        },
        "gcbs_cooperative_action_deadline_validated": True,
        "gcbs_failure_rollback_validated": True,
        "gcbs_root_failure_safe": True,
        "solver_or_controller_invoked": True,
        "ttf_computed": False,
    }
    destination = Path(output).resolve()
    if destination.exists():
        raise ValueError("GCBS qualification output already exists")
    _write_json(destination, report)
    return report


def _run_attempt(
    state_row: dict[str, Any], algorithm: str, trial_index: int, repair_seed: int,
    time_limit: float,
) -> dict[str, Any]:
    candidate = dict(state_row["selected_action"])
    collection = Path(str(state_row["source_collection"]))
    manifest = _one_manifest(collection, solver_seed=int(state_row["solver_seed"]))
    source_state, _trace = target_state_from_trace(
        collection,
        manifest,
        decision_index=int(state_row["decision_index"]),
        expected_fingerprint=str(state_row["state_fingerprint"]),
    )
    before_repair = repair_structure_fingerprint(source_state)
    if before_repair != str(state_row["repair_structure_fingerprint"]):
        raise RuntimeError("source repair structure changed before multiseed screen")
    job = _replay_job(state_row, algorithm, time_limit)
    environment, before = restore_repair_state(
        job, source_state, seed=repairability_restore_seed(before_repair)
    )
    if repair_structure_fingerprint(before) != before_repair:
        raise RuntimeError("restored repair structure changed")

    agents = list(map(int, candidate["agents"]))
    action = _paired_action(agents, repair_seed)
    result = _plain(environment.step_with_time_limit(action, time_limit))
    after, metrics = _validate_native_repair(
        result, expected_agents=agents, expected_seed=repair_seed
    )
    if int(metrics.get("requested_random_seed", -1)) != int(repair_seed):
        raise RuntimeError("native requested action seed differs from paired seed")
    if metrics.get("action_valid") is not True or metrics.get("generated") is not True:
        raise RuntimeError("registered explicit neighborhood was not applied")
    before_conflicts = int(before["num_of_colliding_pairs"])
    after_conflicts = int(after["num_of_colliding_pairs"])
    after_repair = repair_structure_fingerprint(after)
    if before_conflicts != int(state_row["conflicts"]):
        raise RuntimeError("restored conflict count changed")
    if after_conflicts > before_conflicts:
        raise RuntimeError("repairer committed a conflict increase")
    rollback_exact = after_repair == before_repair and after_conflicts == before_conflicts
    if not bool(metrics["replan_success"]) and not rollback_exact:
        raise RuntimeError("failed repairer action was not an exact repair-structure rollback")
    failure_reason = str(metrics.get("pp_failure_reason", ""))
    if failure_reason == "time_limit" and not rollback_exact:
        raise RuntimeError("timed repairer failure was not atomic")
    native_replan_seconds = float(metrics.get("native_replan_seconds", 0.0))
    if native_replan_seconds > time_limit + MAX_ACTION_OVERSHOOT_SECONDS:
        raise RuntimeError("repairer exceeded the cooperative action deadline tolerance")
    return {
        "schema": BRANCH_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "ok",
        "map_id": str(state_row["map_id"]),
        "task_id": str(state_row["task_id"]),
        "source_decision_index": int(state_row["decision_index"]),
        "source_state_fingerprint": str(state_row["state_fingerprint"]),
        "source_repair_structure_fingerprint": before_repair,
        "candidate_role": str(candidate["role"]),
        "candidate_id": str(candidate["candidate_id"]),
        "agents": agents,
        "trial_index": int(trial_index),
        "repair_seed": int(repair_seed),
        "repair_algorithm": algorithm,
        "requested_random_seed": int(metrics["requested_random_seed"]),
        "requested_pp_random_seed": int(metrics["requested_pp_random_seed"]),
        "applied_pp_random_seed": int(metrics["applied_pp_random_seed"]),
        "configured_seconds": float(time_limit),
        "replan_success": bool(metrics["replan_success"]),
        "rollback_exact": rollback_exact,
        "rolled_back_metric": bool(metrics.get("pp_rolled_back", False)),
        "failure_reason": failure_reason,
        "conflicts_before": before_conflicts,
        "conflicts_after": after_conflicts,
        "conflict_reduction": before_conflicts - after_conflicts,
        "normalized_conflict_reduction": (before_conflicts - after_conflicts)
        / max(1, before_conflicts),
        "strict_drop": after_conflicts < before_conflicts,
        "native_replan_seconds": native_replan_seconds,
        "native_step_seconds": float(metrics.get("native_step_seconds", 0.0)),
        "after_repair_structure_fingerprint": after_repair,
    }


def _pair_worker(
    sender: Any, state_row: dict[str, Any], pair: dict[str, Any], time_limit: float
) -> None:
    try:
        rows = [
            _run_attempt(
                state_row,
                algorithm,
                int(pair["trial_index"]),
                int(pair["repair_seed"]),
                time_limit,
            )
            for algorithm in pair["algorithm_order"]
        ]
        sender.send({"ok": True, "rows": rows})
    except BaseException as error:
        sender.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
    finally:
        sender.close()


def _isolated_pair(
    state_row: dict[str, Any], pair: dict[str, Any], *, time_limit: float,
    fuse_seconds: float,
) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_pair_worker, args=(sender, state_row, pair, time_limit)
    )
    process.start()
    sender.close()
    process.join(fuse_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        receiver.close()
        raise RuntimeError(f"pair process timeout: {pair['map_id']} trial {pair['trial_index']}")
    payload = receiver.recv() if receiver.poll(1.0) else None
    receiver.close()
    if process.exitcode != 0 or not isinstance(payload, dict):
        raise RuntimeError(f"pair process failed with exit code {process.exitcode}")
    if payload.get("ok") is not True:
        raise RuntimeError(str(payload.get("error")))
    rows = [dict(row) for row in payload["rows"]]
    if len(rows) != len(ALGORITHMS):
        raise RuntimeError("pair worker did not return both algorithms")
    return rows


def _algorithm_summary(rows: list[dict[str, Any]], algorithm: str) -> dict[str, Any]:
    selected = [row for row in rows if row["repair_algorithm"] == algorithm]
    return {
        "repair_algorithm": algorithm,
        "attempts": len(selected),
        "strict_drop_count": sum(bool(row["strict_drop"]) for row in selected),
        "exact_noop_count": sum(bool(row["rollback_exact"]) for row in selected),
        "time_limit_count": sum(row["failure_reason"] == "time_limit" for row in selected),
        "mean_conflict_reduction": statistics.fmean(
            float(row["conflict_reduction"]) for row in selected
        ),
        "mean_normalized_conflict_reduction": statistics.fmean(
            float(row["normalized_conflict_reduction"]) for row in selected
        ),
        "mean_native_replan_seconds": statistics.fmean(
            float(row["native_replan_seconds"]) for row in selected
        ),
    }


def _decision(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    summaries = {name: _algorithm_summary(rows, name) for name in ALGORITHMS}
    state_metrics: dict[str, dict[str, Any]] = {}
    for map_id in sorted({str(row["map_id"]) for row in rows}):
        by_algorithm = {
            name: [
                row
                for row in rows
                if row["map_id"] == map_id and row["repair_algorithm"] == name
            ]
            for name in ALGORITHMS
        }
        state_metrics[map_id] = {
            "strict_drop_count": {
                name: sum(bool(row["strict_drop"]) for row in selected)
                for name, selected in by_algorithm.items()
            },
            "mean_normalized_conflict_reduction": {
                name: statistics.fmean(
                    float(row["normalized_conflict_reduction"])
                    for row in selected
                )
                for name, selected in by_algorithm.items()
            },
            "gcbs_after_fingerprint_count": len(
                {
                    str(row["after_repair_structure_fingerprint"])
                    for row in by_algorithm["GCBS"]
                }
            ),
        }
    gate = dict(config["decision_gate"])
    failures: list[str] = []
    if summaries["GCBS"]["strict_drop_count"] <= summaries["PP"]["strict_drop_count"]:
        failures.append("gcbs_strict_drop_count_not_higher")
    if summaries["GCBS"]["exact_noop_count"] >= summaries["PP"]["exact_noop_count"]:
        failures.append("gcbs_exact_noop_count_not_lower")
    hard_maps = tuple(map(str, gate["hard_state_map_ids"]))
    hard_strict_superiority = 0
    for map_id, metrics in state_metrics.items():
        drops = dict(metrics["strict_drop_count"])
        if drops["GCBS"] < drops["PP"]:
            failures.append(f"gcbs_strict_drop_regressed:{map_id}")
        if map_id in hard_maps:
            if drops["GCBS"] < int(gate["minimum_gcbs_strict_drops_per_hard_state"]):
                failures.append(f"gcbs_hard_state_drop_floor_failed:{map_id}")
            if drops["GCBS"] > drops["PP"]:
                hard_strict_superiority += 1
    if hard_strict_superiority < int(gate["minimum_hard_states_strictly_better"]):
        failures.append("gcbs_hard_state_superiority_breadth_failed")
    positive_map = str(gate["positive_control_map_id"])
    positive_drops = dict(state_metrics[positive_map]["strict_drop_count"])
    if positive_drops["GCBS"] < int(gate["minimum_gcbs_positive_control_drops"]):
        failures.append("positive_control_drop_floor_failed")
    if (
        summaries["GCBS"]["mean_normalized_conflict_reduction"]
        <= summaries["PP"]["mean_normalized_conflict_reduction"]
    ):
        failures.append("gcbs_mean_normalized_reduction_not_higher")
    if summaries["GCBS"]["time_limit_count"] > int(gate["maximum_gcbs_time_limit_count"]):
        failures.append("gcbs_time_limit_count_exceeded")
    return {
        "passed": not failures,
        "decision": (
            "eligible_for_fresh_conditional_fallback_test"
            if not failures
            else "stop_same_set_gcbs_repairer_branch"
        ),
        "failures": failures,
        "state_metrics": state_metrics,
        "summaries": [summaries[name] for name in ALGORITHMS],
    }


def run_screen(
    config_path: str | Path, output: str | Path, *, dry_run: bool = False
) -> dict[str, Any]:
    path, _project_root, config = load_registration(config_path)
    plan = build_plan(path)
    if dry_run:
        return plan
    root = Path(output).resolve()
    if root.exists():
        raise ValueError("multiseed PP/GCBS output already exists")
    root.mkdir(parents=True)
    plan_path = root / "screen_plan.json"
    _write_json(plan_path, plan)
    states = {str(row["map_id"]): row for row in plan["states"]}
    rows: list[dict[str, Any]] = []
    time_limit = float(config["execution"]["action_time_limit_seconds"])
    fuse = float(config["execution"]["pair_process_fuse_seconds"])
    for pair in plan["pairs"]:
        pair_rows = _isolated_pair(
            states[str(pair["map_id"])], pair, time_limit=time_limit, fuse_seconds=fuse
        )
        destination = (
            root
            / "pairs"
            / str(pair["map_id"])
            / f"trial_{int(pair['trial_index']):02d}.json"
        )
        _write_json(destination, {"pair": pair, "rows": pair_rows})
        rows.extend(pair_rows)
    decision = _decision(rows, config)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "config_sha256": plan["config_sha256"],
        "plan_sha256": sha256_file(plan_path),
        "attempt_count": len(rows),
        "process_or_integrity_error_count": 0,
        "decision": decision,
        "paired_results": rows,
        "qualification": plan["qualification"],
        "claim_boundary": plan["claim_boundary"],
        "ttf_claim_allowed": False,
        "runtime_promotion_allowed": False,
    }
    _write_json(root / "same_set_multiseed_report.json", report)
    return report


__all__ = [
    "ALGORITHMS",
    "BRANCH_SCHEMA",
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "PLAN_SCHEMA",
    "QUALIFICATION_SCHEMA",
    "REPORT_SCHEMA",
    "_trial_seed",
    "build_plan",
    "load_registration",
    "run_qualification",
    "run_screen",
]
