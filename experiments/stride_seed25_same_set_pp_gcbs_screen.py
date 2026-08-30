from __future__ import annotations

import multiprocessing
import statistics
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _plain, _read_json, _write_json
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_marginalpool_action_replay import (
    _replay_job as _marginalpool_replay_job,
)
from experiments.stride_repairability_collection import repairability_restore_seed
from experiments.stride_structshell_seed25_replay_diagnostic import (
    _one_manifest,
    build_plan as build_replay_plan,
)
from experiments.trace_replay import restore_repair_state, target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.seed25_same_set_pp_gcbs_screen_registration.v1"
PLAN_SCHEMA = "lns2.stride.seed25_same_set_pp_gcbs_screen_plan.v1"
BRANCH_SCHEMA = "lns2.stride.seed25_same_set_pp_gcbs_screen_branch.v1"
REPORT_SCHEMA = "lns2.stride.seed25_same_set_pp_gcbs_screen_report.v1"
EXPERIMENT_ID = "stride-seed25-same-set-pp-gcbs-screen-v1"
ALGORITHMS = ("PP", "GCBS")


def _resolve(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def load_registration(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    project_root = path.parent.parent
    config = _read_json(path)
    source = dict(config.get("source") or {})
    execution = dict(config.get("execution") or {})
    qualification = dict(config.get("qualification_boundary") or {})
    claims = dict(config.get("claim_boundary") or {})
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or int(source.get("solver_seed", -1)) != 25
        or source.get("selection_rule")
        != "dual_actual_selected_exact_agent_set_at_each_frozen_checkpoint"
        or len(source.get("states") or ()) != 4
        or tuple(execution.get("repair_algorithms") or ()) != ALGORITHMS
        or execution.get("one_step_only") is not True
        or float(execution.get("environment_time_limit_seconds", -1.0)) != 5.0
        or float(execution.get("pp_action_time_limit_seconds", -1.0)) != 5.0
        or float(execution.get("process_fuse_seconds_per_job", -1.0)) != 20.0
        or execution.get("strict_serial") is not True
        or int(execution.get("trial_count_per_state_algorithm", -1)) != 1
        or any(
            execution.get(name) is not False
            for name in ("candidate_generation", "controller_selection", "continuation", "ttf")
        )
        or qualification.get("pbs_included") is not False
        or qualification.get("gcbs_cooperative_action_deadline_validated") is not False
        or qualification.get("gcbs_failure_rollback_validated") is not False
        or qualification.get("gcbs_external_process_fuse_required") is not True
        or qualification.get("uniform_acceptance_semantics") is not False
        or claims.get("mechanism_diagnostic_only") is not True
        or claims.get("outcome_enriched_actions") is not True
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
        raise ValueError("same-set PP/GCBS screen registration contract changed")
    replay_registration = _resolve(project_root, source["replay_registration"])
    if sha256_file(replay_registration) != str(source["replay_registration_sha256"]):
        raise ValueError("source replay registration SHA-256 changed")
    return path, project_root, config


def build_plan(config_path: str | Path) -> dict[str, Any]:
    path, _project_root, config = load_registration(config_path)
    source = dict(config["source"])
    replay_registration = _resolve(path.parent.parent, source["replay_registration"])
    replay_plan = build_replay_plan(replay_registration)
    expected = {str(row["map_id"]): dict(row) for row in source["states"]}
    states: list[dict[str, Any]] = []
    for source_state in replay_plan["states"]:
        map_id = str(source_state["map_id"])
        registered = expected.get(map_id)
        if registered is None:
            raise ValueError(f"unregistered source map: {map_id}")
        matches = [
            dict(candidate)
            for candidate in source_state["candidates"]
            if str(candidate["candidate_id"]) == str(registered["candidate_id"])
            and str(candidate["role"]) == str(registered["role"])
            and list(map(int, candidate["agents"]))
            == list(map(int, source_state["dual_action_agents"]))
        ]
        if len(matches) != 1:
            raise ValueError(f"registered Dual action does not resolve exactly once: {map_id}")
        if (
            int(source_state["decision_index"]) != int(registered["decision_index"])
            or str(source_state["state_fingerprint"])
            != str(registered["state_fingerprint"])
            or int(source_state["conflicts"]) != int(registered["conflicts"])
        ):
            raise ValueError(f"registered source state changed: {map_id}")
        states.append(
            {
                **dict(source_state),
                "selected_action": matches[0],
                "candidates": None,
                "outcome_enriched_selection": True,
            }
        )
    if set(expected) != {str(row["map_id"]) for row in states}:
        raise ValueError("registered source map cohort changed")
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "source_replay_plan_sha256": sha256_file(
            _resolve(path.parent.parent, "build/stride-structshell-seed25-replay-diagnostic-v1/replay_plan.json")
        ),
        "state_count": len(states),
        "branch_count": len(states) * len(ALGORITHMS),
        "repair_algorithms": list(ALGORITHMS),
        "one_step_only": True,
        "process_fuse_seconds_per_job": float(
            config["execution"]["process_fuse_seconds_per_job"]
        ),
        "states": states,
        "claim_boundary": dict(config["claim_boundary"]),
        "qualification_boundary": dict(config["qualification_boundary"]),
    }


def _replay_job(state_row: dict[str, Any], algorithm: str, time_limit: float) -> dict[str, Any]:
    job = _marginalpool_replay_job(
        {
            "source_run_config": str(state_row["source_run_config"]),
            "source_run_config_sha256": str(state_row["source_run_config_sha256"]),
            "task_id": str(state_row["task_id"]),
            "solver_seed": int(state_row["solver_seed"]),
            "split": str(state_row["split"]),
        }
    )
    job["environment"] = dict(
        job["environment"],
        replan_algorithm=str(algorithm),
        time_limit=float(time_limit),
        max_repair_iterations=1,
    )
    return job


def run_branch(state_row: dict[str, Any], algorithm: str) -> dict[str, Any]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unregistered repair algorithm: {algorithm}")
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
        raise RuntimeError("source repair structure changed before repairer screen")
    time_limit = float(state_row["per_action_time_limit_seconds"])
    job = _replay_job(state_row, algorithm, time_limit)
    environment, before = restore_repair_state(
        job,
        source_state,
        seed=repairability_restore_seed(before_repair),
    )
    if repair_structure_fingerprint(before) != before_repair:
        raise RuntimeError("restored repair structure changed")
    pp_seed = int(state_row["first_action_pp_seed"])
    action = _paired_action(list(map(int, candidate["agents"])), pp_seed)
    if algorithm == "PP":
        result = _plain(environment.step_with_time_limit(action, time_limit))
        budget_semantics = "cooperative_action_deadline"
    else:
        result = _plain(environment.step(action))
        budget_semantics = "environment_budget_with_external_process_fuse"
    after, metrics = _validate_native_repair(
        result,
        expected_agents=list(map(int, candidate["agents"])),
        expected_seed=pp_seed,
    )
    if metrics.get("action_valid") is not True or metrics.get("generated") is not True:
        raise RuntimeError("registered explicit neighborhood was not applied")
    after_repair = repair_structure_fingerprint(after)
    before_conflicts = int(before["num_of_colliding_pairs"])
    after_conflicts = int(after["num_of_colliding_pairs"])
    if before_conflicts != int(state_row["conflicts"]):
        raise RuntimeError("restored conflict count changed")
    if after_conflicts > before_conflicts:
        raise RuntimeError("repairer committed a conflict increase")
    rollback_exact = after_repair == before_repair and after_conflicts == before_conflicts
    if not bool(metrics["replan_success"]) and not rollback_exact:
        raise RuntimeError("failed repairer action was not an exact repair-structure no-op")
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
        "agents": list(map(int, candidate["agents"])),
        "repair_algorithm": algorithm,
        "pp_seed_requested": pp_seed,
        "budget_semantics": budget_semantics,
        "configured_seconds": time_limit,
        "action_valid": True,
        "replan_success": bool(metrics["replan_success"]),
        "rollback_exact": rollback_exact,
        "pp_rolled_back_metric": bool(metrics.get("pp_rolled_back", False)),
        "pp_failure_reason": str(metrics.get("pp_failure_reason", "")),
        "conflicts_before": before_conflicts,
        "conflicts_after": after_conflicts,
        "conflict_reduction": before_conflicts - after_conflicts,
        "normalized_conflict_reduction": (before_conflicts - after_conflicts)
        / max(1, before_conflicts),
        "strict_drop": after_conflicts < before_conflicts,
        "after_repair_structure_fingerprint": after_repair,
        "native_replan_seconds": float(metrics.get("native_replan_seconds", 0.0)),
        "native_step_seconds": float(metrics.get("native_step_seconds", 0.0)),
        "feasible": bool(after.get("feasible")),
        "ttf_stored": False,
    }


def _worker(sender: Any, state_row: dict[str, Any], algorithm: str) -> None:
    try:
        sender.send({"ok": True, "result": run_branch(state_row, algorithm)})
    except BaseException as error:  # child boundary must report ordinary Python failures
        sender.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
    finally:
        sender.close()


def _isolated_branch(
    state_row: dict[str, Any], algorithm: str, *, fuse_seconds: float
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker, args=(sender, state_row, algorithm))
    process.start()
    sender.close()
    process.join(float(fuse_seconds))
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        receiver.close()
        return {
            "status": "process_timeout",
            "map_id": str(state_row["map_id"]),
            "repair_algorithm": algorithm,
            "process_exit_code": process.exitcode,
        }
    payload = receiver.recv() if receiver.poll(1.0) else None
    receiver.close()
    if process.exitcode != 0 or not isinstance(payload, dict):
        return {
            "status": "process_error",
            "map_id": str(state_row["map_id"]),
            "repair_algorithm": algorithm,
            "process_exit_code": process.exitcode,
        }
    if payload.get("ok") is not True:
        return {
            "status": "python_error",
            "map_id": str(state_row["map_id"]),
            "repair_algorithm": algorithm,
            "process_exit_code": process.exitcode,
            "error": str(payload.get("error")),
        }
    return dict(payload["result"])


def _summary(rows: list[dict[str, Any]], algorithm: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("repair_algorithm") == algorithm]
    ok = [row for row in selected if row.get("status") == "ok"]
    return {
        "repair_algorithm": algorithm,
        "attempts": len(selected),
        "completed": len(ok),
        "strict_drop_count": sum(bool(row["strict_drop"]) for row in ok),
        "exact_noop_count": sum(bool(row["rollback_exact"]) for row in ok),
        "mean_conflict_reduction": statistics.fmean(
            float(row["conflict_reduction"]) for row in ok
        )
        if ok
        else None,
        "mean_normalized_conflict_reduction": statistics.fmean(
            float(row["normalized_conflict_reduction"]) for row in ok
        )
        if ok
        else None,
        "mean_native_replan_seconds": statistics.fmean(
            float(row["native_replan_seconds"]) for row in ok
        )
        if ok
        else None,
    }


def run_screen(
    config_path: str | Path, output: str | Path, *, dry_run: bool = False
) -> dict[str, Any]:
    plan = build_plan(config_path)
    if dry_run:
        return plan
    root = Path(output).resolve()
    if root.exists():
        raise ValueError("same-set PP/GCBS output already exists")
    root.mkdir(parents=True)
    plan_path = root / "screen_plan.json"
    _write_json(plan_path, plan)
    rows: list[dict[str, Any]] = []
    fuse = float(plan["process_fuse_seconds_per_job"])
    for state_row in plan["states"]:
        for algorithm in ALGORITHMS:
            row = _isolated_branch(state_row, algorithm, fuse_seconds=fuse)
            destination = root / "branches" / str(state_row["map_id"]) / f"{algorithm.lower()}.json"
            _write_json(destination, row)
            rows.append(row)
    errors = [row for row in rows if row.get("status") != "ok"]
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete" if not errors else "completed_with_process_or_integrity_errors",
        "config_sha256": plan["config_sha256"],
        "plan_sha256": sha256_file(plan_path),
        "state_count": plan["state_count"],
        "branch_count": len(rows),
        "branch_error_count": len(errors),
        "summaries": [_summary(rows, algorithm) for algorithm in ALGORITHMS],
        "paired_results": rows,
        "qualification_boundary": plan["qualification_boundary"],
        "claim_boundary": plan["claim_boundary"],
        "ttf_claim_allowed": False,
        "runtime_promotion_allowed": False,
    }
    _write_json(root / "same_set_repairer_report.json", report)
    return report


__all__ = [
    "ALGORITHMS",
    "BRANCH_SCHEMA",
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "PLAN_SCHEMA",
    "REPORT_SCHEMA",
    "build_plan",
    "load_registration",
    "run_branch",
    "run_screen",
]
