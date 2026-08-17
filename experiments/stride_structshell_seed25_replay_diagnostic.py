from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, sha256_file
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_marginalpool_action_replay import (
    _replay_job as _marginalpool_replay_job,
)
from experiments.stride_repairability_collection import repairability_restore_seed
from experiments.trace_replay import restore_repair_state, target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.online_selection import (
    generate_online_candidates,
    score_online_candidates,
)


CONFIG_SCHEMA = "lns2.stride.structshell_seed25_replay_diagnostic_registration.v1"
PLAN_SCHEMA = "lns2.stride.structshell_seed25_replay_diagnostic_plan.v1"
BRANCH_SCHEMA = "lns2.stride.structshell_seed25_replay_diagnostic_branch.v1"
REPORT_SCHEMA = "lns2.stride.structshell_seed25_replay_diagnostic_report.v1"
EXPERIMENT_ID = "stride-structshell-seed25-replay-diagnostic-v1"
PROFILE = "realized_dynamic"
PRIMARY_ROLES = ("v2_anchor", "component16", "hotspot16")
OBSERVED_ROLE = "observed_dual_winner"


def _transition_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [dict(row) for row in events if row.get("event") == "transition"]
    if not rows or [int(row["decision_index"]) for row in rows] != list(
        range(len(rows))
    ):
        raise ValueError("source trace has a non-contiguous transition sequence")
    return rows


def _action_agents(event: dict[str, Any]) -> tuple[int, ...]:
    action = dict(event.get("action") or {})
    agents = tuple(sorted(map(int, action.get("agents") or ())))
    if action.get("mode") != "explicit_neighborhood" or not agents:
        raise ValueError("source transition has no explicit executed neighborhood")
    return agents


def _guard_triggered(event: dict[str, Any]) -> bool:
    proposal = dict(dict(event.get("controller") or {}).get("proposal") or {})
    return (
        proposal.get("hybridstructpool_stall_guard_triggered") is True
        or proposal.get("guardpool_triggered") is True
    )


def select_checkpoint(
    dual_events: Iterable[dict[str, Any]], guard_events: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Select a checkpoint using pre-action fields only.

    The function deliberately does not inspect metrics, deltas, termination, or
    elapsed-time fields.  This keeps checkpoint selection blind to repair outcome.
    """

    dual = {int(row["decision_index"]): row for row in _transition_events(dual_events)}
    guard = {
        int(row["decision_index"]): row for row in _transition_events(guard_events)
    }
    common = sorted(set(dual) & set(guard))
    if 0 not in common:
        raise ValueError("Dual and Guard traces do not share decision zero")

    def common_state(index: int) -> bool:
        return str(dual[index].get("before_fingerprint")) == str(
            guard[index].get("before_fingerprint")
        )

    selected: int | None = None
    rule = ""
    for index in common:
        if common_state(index) and _action_agents(dual[index]) != _action_agents(
            guard[index]
        ):
            selected = index
            rule = "first_equal_pre_action_fingerprint_with_different_executed_agent_set"
            break
    if selected is None:
        for index in common:
            if common_state(index) and _guard_triggered(guard[index]):
                selected = index
                rule = "first_registered_plateau_guard_trigger"
                break
    if selected is None:
        selected = 0
        rule = "decision_zero"
        if not common_state(selected):
            raise ValueError("Dual and Guard decision-zero states differ")

    event = dual[selected]
    fingerprint = str(event.get("before_fingerprint") or "")
    if len(fingerprint) != 64:
        raise ValueError("selected checkpoint lacks a SHA-256 state fingerprint")
    return {
        "decision_index": selected,
        "selection_rule": rule,
        "state_fingerprint": fingerprint,
        "dual_action_agents": list(_action_agents(event)),
        "guard_action_agents": list(_action_agents(guard[selected])),
        "outcome_fields_read": False,
    }


def _candidate_by_id(pool: list[dict[str, Any]], candidate_id: Any) -> dict[str, Any]:
    matches = [row for row in pool if str(row.get("candidate_id")) == str(candidate_id)]
    if len(matches) != 1:
        raise ValueError(f"candidate id must resolve exactly once: {candidate_id}")
    return matches[0]


def _candidate_by_family(pool: list[dict[str, Any]], family: str) -> dict[str, Any]:
    matches = [
        row
        for row in pool
        if family in set(map(str, row.get("selection_families") or ()))
    ]
    if len(matches) != 1:
        raise ValueError(f"structural family must resolve exactly once: {family}")
    return matches[0]


def _candidate_record(role: str, row: dict[str, Any]) -> dict[str, Any]:
    agents = sorted(map(int, row.get("agents") or ()))
    if not agents:
        raise ValueError(f"candidate {role} has no agents")
    return {
        "role": role,
        "candidate_id": str(row["candidate_id"]),
        "agents": agents,
        "actual_size": len(agents),
        "selection_families": sorted(map(str, row.get("selection_families") or ())),
        "source_score": float(row["score"]),
    }


def extract_action_candidates(dual_event: dict[str, Any]) -> list[dict[str, Any]]:
    """Freeze the three causal actions and a distinct observed choice-set action."""

    controller = dict(dual_event.get("controller") or {})
    proposal = dict(controller.get("proposal") or {})
    pool = [dict(row) for row in controller.get("candidate_pool") or ()]
    if not pool:
        raise ValueError("selected Dual event has no recorded candidate pool")
    rows = [
        _candidate_record(
            "v2_anchor",
            _candidate_by_id(pool, proposal.get("hybridstructpool_v2_anchor_candidate_id")),
        ),
        _candidate_record(
            "component16",
            _candidate_by_family(pool, "structpool-conflict-component:16"),
        ),
        _candidate_record(
            "hotspot16",
            _candidate_by_family(pool, "structpool-spatiotemporal-hotspot:16"),
        ),
    ]
    if tuple(row["role"] for row in rows) != PRIMARY_ROLES:
        raise AssertionError("primary candidate ordering changed")
    if any(row["actual_size"] != 16 for row in rows[1:]):
        raise ValueError("registered Component/Hotspot candidates are not size 16")
    primary_sets = {tuple(row["agents"]) for row in rows}
    if len(primary_sets) != len(rows):
        raise ValueError("primary actions are not distinct exact neighborhoods")

    observed = _candidate_record(
        OBSERVED_ROLE,
        _candidate_by_id(pool, controller.get("selected_candidate_id")),
    )
    executed = list(_action_agents(dual_event))
    if observed["agents"] != executed:
        raise ValueError("recorded Dual winner differs from the executed neighborhood")
    if tuple(observed["agents"]) not in primary_sets:
        rows.append(observed)
    return rows


def paired_pp_seed(state_repair_fingerprint: str, step_index: int) -> int:
    if len(str(state_repair_fingerprint)) != 64 or int(step_index) < 0:
        raise ValueError("invalid paired PP seed input")
    digest = _fingerprint(
        {
            "namespace": EXPERIMENT_ID,
            "repair_structure_fingerprint": str(state_repair_fingerprint),
            "step_index": int(step_index),
        }
    )
    return 1 + int(digest[:16], 16) % (2**31 - 2)


def _resolve_input(project_root: Path, value: Any) -> Path:
    path = Path(str(value))
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def load_registration(config_path: str | Path) -> tuple[Path, Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    project_root = path.parent.parent
    config = _read_json(path)
    if config.get("schema") != CONFIG_SCHEMA or config.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected replay diagnostic registration identity")
    source = dict(config.get("source") or {})
    execution = dict(config.get("execution") or {})
    selection = dict(config.get("selection") or {})
    if (
        int(source.get("solver_seed", -1)) != 25
        or source.get("dual_arm") != "dual16"
        or source.get("guard_arm") != "dual16_plateau"
        or len(source.get("maps") or ()) != 4
        or tuple(selection.get("required_candidates") or ()) != PRIMARY_ROLES
        or execution.get("continuation_controller") != "v2-full"
        or int(execution.get("continuation_decisions", -1)) != 6
        or float(execution.get("per_action_time_limit_seconds", -1.0)) != 5.0
        or float(execution.get("total_process_time_limit_seconds", -1.0)) != 600.0
        or execution.get("strict_serial") is not True
        or execution.get("same_first_action_pp_seed_across_arms") is not True
        or execution.get("same_continuation_step_pp_seed_across_arms") is not True
        or execution.get("full_ttf_run") is not False
        or execution.get("official_adaptive_arm") is not False
    ):
        raise ValueError("replay diagnostic causal contract changed")
    quick_report = _resolve_input(project_root, source["quick_report"])
    if sha256_file(quick_report) != str(source["quick_report_sha256"]):
        raise ValueError("source quick report SHA-256 changed")
    return path, project_root, config


def _one_manifest(collection: Path, *, solver_seed: int) -> dict[str, Any]:
    rows = _read_jsonl(collection / "realized_dynamic_manifest.jsonl")
    if len(rows) != 1:
        raise ValueError(f"source collection must have exactly one episode: {collection}")
    row = dict(rows[0])
    if (
        row.get("status") != "ok"
        or row.get("error") is not None
        or row.get("policy") != PROFILE
        or int(row.get("solver_seed", -1)) != int(solver_seed)
    ):
        raise ValueError(f"source manifest identity/status changed: {collection}")
    trace = contained_file(collection, row.get("trace_file"), field="trace_file")
    if sha256_file(trace) != str(row.get("trace_sha256")):
        raise ValueError(f"source trace SHA-256 changed: {collection}")
    return row


def _events(collection: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    trace = contained_file(collection, manifest.get("trace_file"), field="trace_file")
    return read_trace_events(trace)


def build_plan(config_path: str | Path) -> dict[str, Any]:
    path, project_root, config = load_registration(config_path)
    source = dict(config["source"])
    source_root = _resolve_input(project_root, source["output_root"])
    states: list[dict[str, Any]] = []
    for map_id in map(str, source["maps"]):
        dual_root = source_root / "maps" / map_id / str(source["dual_arm"])
        guard_root = source_root / "maps" / map_id / str(source["guard_arm"])
        dual_manifest = _one_manifest(dual_root, solver_seed=int(source["solver_seed"]))
        guard_manifest = _one_manifest(guard_root, solver_seed=int(source["solver_seed"]))
        if (
            dual_manifest["task_id"] != guard_manifest["task_id"]
            or dual_manifest["split"] != guard_manifest["split"]
        ):
            raise ValueError(f"Dual/Guard task identity differs: {map_id}")
        dual_events = _events(dual_root, dual_manifest)
        guard_events = _events(guard_root, guard_manifest)
        checkpoint = select_checkpoint(dual_events, guard_events)
        dual_by_index = {
            int(row["decision_index"]): row for row in _transition_events(dual_events)
        }
        candidates = extract_action_candidates(
            dual_by_index[int(checkpoint["decision_index"])]
        )
        state, _trace = target_state_from_trace(
            dual_root,
            dual_manifest,
            decision_index=int(checkpoint["decision_index"]),
            expected_fingerprint=str(checkpoint["state_fingerprint"]),
        )
        repair_fp = repair_structure_fingerprint(state)
        run_config = dual_root / "run_config.json"
        states.append(
            {
                "map_id": map_id,
                "task_id": str(dual_manifest["task_id"]),
                "split": str(dual_manifest["split"]),
                "solver_seed": int(source["solver_seed"]),
                **checkpoint,
                "conflicts": int(state["num_of_colliding_pairs"]),
                "repair_structure_fingerprint": repair_fp,
                "source_collection": str(dual_root),
                "source_manifest_sha256": sha256_file(
                    dual_root / "realized_dynamic_manifest.jsonl"
                ),
                "source_trace_sha256": str(dual_manifest["trace_sha256"]),
                "source_run_config": str(run_config),
                "source_run_config_sha256": sha256_file(run_config),
                "first_action_pp_seed": paired_pp_seed(repair_fp, 0),
                "continuation_pp_seeds": [
                    paired_pp_seed(repair_fp, step) for step in range(1, 7)
                ],
                "per_action_time_limit_seconds": float(
                    config["execution"]["per_action_time_limit_seconds"]
                ),
                "candidates": candidates,
            }
        )
    branch_count = sum(len(row["candidates"]) for row in states)
    continuation_count = int(config["execution"]["continuation_decisions"])
    return {
        "schema": PLAN_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_path": str(path),
        "config_sha256": sha256_file(path),
        "state_count": len(states),
        "branch_count": branch_count,
        "maximum_native_pp_calls": branch_count * (1 + continuation_count),
        "maximum_v2_continuation_selections": branch_count * continuation_count,
        "per_action_time_limit_seconds": float(
            config["execution"]["per_action_time_limit_seconds"]
        ),
        "total_process_time_limit_seconds": float(
            config["execution"]["total_process_time_limit_seconds"]
        ),
        "strict_serial": True,
        "full_ttf_run": False,
        "official_adaptive_included": False,
        "states": states,
    }


def _portable_path(value: Any) -> Path:
    text = str(value)
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 6:
        return Path(f"{text[5].upper()}:/{text[7:]}")
    return Path(text)


def _pure_v2_runtime(state_row: dict[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    run_path = Path(str(state_row["source_run_config"]))
    if sha256_file(run_path) != str(state_row["source_run_config_sha256"]):
        raise ValueError("source run configuration changed after planning")
    run = _read_json(run_path)
    proposal = dict(run["configuration"]["proposal"])
    proposal.pop("hybridstructpool", None)
    proposal.pop("structpool", None)
    proposal.pop("topology_boundary", None)
    bundle_root = _portable_path(run["configuration"]["controller_bundle"])
    model = load_controller_bundle(bundle_root).main_models[PROFILE]
    state_record = {
        "source_run_config": str(run_path),
        "source_run_config_sha256": str(state_row["source_run_config_sha256"]),
        "task_id": str(state_row["task_id"]),
        "solver_seed": int(state_row["solver_seed"]),
        "split": str(state_row["split"]),
    }
    return proposal, model, _marginalpool_replay_job(state_record)


def _select_v2(
    environment: Any,
    state: dict[str, Any],
    *,
    proposal: dict[str, Any],
    model: Any,
    task_id: str,
    solver_seed: int,
    decision_index: int,
) -> dict[str, Any]:
    key = state_fingerprint(state)
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=task_id,
        solver_seed=solver_seed,
        decision_index=decision_index,
        proposal_config=proposal,
        state_hash=key,
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={PROFILE: PROFILE_FEATURE_NAMES[PROFILE]},
        dense_output=False,
    )
    rows, feature_metrics = engine.realized_rows(candidates, state_hash=key)
    selected, scores, margin = score_online_candidates(rows, model)
    candidate = dict(candidates[selected])
    if candidate.get("structpool_family_groups"):
        raise RuntimeError("pure V2 continuation selected a structural candidate")
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "agents": sorted(map(int, candidate["agents"])),
        "selection_families": sorted(
            map(str, candidate.get("selection_families") or ())
        ),
        "score": float(scores[selected]),
        "score_margin": float(margin),
        "candidate_count": len(candidates),
        "generation": _plain(generation),
        "feature_metrics": _plain(feature_metrics),
    }


def _step_record(
    environment: Any,
    state: dict[str, Any],
    *,
    role: str,
    step_index: int,
    candidate: dict[str, Any],
    pp_seed: int,
    pp_time_limit_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before_repair = repair_structure_fingerprint(state)
    before_conflicts = int(state["num_of_colliding_pairs"])
    result = _plain(
        environment.step_with_time_limit(
            _paired_action(candidate["agents"], pp_seed),
            float(pp_time_limit_seconds),
        )
    )
    after, metrics = _validate_native_repair(
        result, expected_agents=candidate["agents"], expected_seed=pp_seed
    )
    after_repair = repair_structure_fingerprint(after)
    failure_reason = str(metrics.get("pp_failure_reason", ""))
    if not bool(metrics["replan_success"]) and (
        not bool(metrics.get("pp_rolled_back", False))
        or before_repair != after_repair
        or before_conflicts != int(after["num_of_colliding_pairs"])
    ):
        raise RuntimeError("bounded PP failure did not roll back atomically")
    return after, {
        "role": role,
        "step_index": int(step_index),
        "pp_seed": int(pp_seed),
        "candidate_id": str(candidate["candidate_id"]),
        "agents": list(map(int, candidate["agents"])),
        "actual_size": len(candidate["agents"]),
        "selection_families": list(map(str, candidate.get("selection_families") or ())),
        "before_repair_fingerprint": before_repair,
        "after_repair_fingerprint": after_repair,
        "repair_exact_noop": before_repair == after_repair,
        "conflicts_before": before_conflicts,
        "conflicts_after": int(after["num_of_colliding_pairs"]),
        "replan_success": bool(metrics["replan_success"]),
        "pp_rolled_back": bool(metrics.get("pp_rolled_back", False)),
        "pp_failure_reason": failure_reason,
        "time_limit_rollback": (
            failure_reason == "time_limit"
            and bool(metrics.get("pp_rolled_back", False))
            and before_repair == after_repair
        ),
        "requested_pp_time_limit_seconds": float(pp_time_limit_seconds),
        "native_replan_seconds": float(metrics.get("native_replan_seconds", 0.0)),
        "terminal": bool(after.get("done")),
    }


def run_branch(state_row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    proposal, model, replay = _pure_v2_runtime(state_row)
    collection = Path(str(state_row["source_collection"]))
    manifest = _one_manifest(collection, solver_seed=int(state_row["solver_seed"]))
    source_state, _trace = target_state_from_trace(
        collection,
        manifest,
        decision_index=int(state_row["decision_index"]),
        expected_fingerprint=str(state_row["state_fingerprint"]),
    )
    repair_fp = repair_structure_fingerprint(source_state)
    if repair_fp != str(state_row["repair_structure_fingerprint"]):
        raise ValueError("planned repair state changed before execution")
    restore_seed = repairability_restore_seed(repair_fp)
    pp_time_limit = float(state_row["per_action_time_limit_seconds"])
    environment, state = restore_repair_state(replay, source_state, seed=restore_seed)
    transitions: list[dict[str, Any]] = []
    state, first = _step_record(
        environment,
        state,
        role=str(candidate["role"]),
        step_index=0,
        candidate=candidate,
        pp_seed=paired_pp_seed(repair_fp, 0),
        pp_time_limit_seconds=pp_time_limit,
    )
    first["selection_mode"] = "forced_recorded_candidate"
    transitions.append(first)
    for step_index in range(1, 7):
        if bool(state.get("done")):
            break
        selected = _select_v2(
            environment,
            state,
            proposal=proposal,
            model=model,
            task_id=str(state_row["task_id"]),
            solver_seed=int(state_row["solver_seed"]),
            decision_index=int(state_row["decision_index"]) + step_index,
        )
        state, transition = _step_record(
            environment,
            state,
            role=str(candidate["role"]),
            step_index=step_index,
            candidate=selected,
            pp_seed=paired_pp_seed(repair_fp, step_index),
            pp_time_limit_seconds=pp_time_limit,
        )
        transition["selection_mode"] = "fresh_v2_full"
        transition["score"] = selected["score"]
        transition["score_margin"] = selected["score_margin"]
        transition["candidate_count"] = selected["candidate_count"]
        transitions.append(transition)
    return {
        "schema": BRANCH_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "ok",
        "map_id": str(state_row["map_id"]),
        "task_id": str(state_row["task_id"]),
        "solver_seed": int(state_row["solver_seed"]),
        "source_decision_index": int(state_row["decision_index"]),
        "source_state_fingerprint": str(state_row["state_fingerprint"]),
        "source_repair_structure_fingerprint": repair_fp,
        "restore_seed": restore_seed,
        "role": str(candidate["role"]),
        "first_candidate_id": str(candidate["candidate_id"]),
        "first_candidate_agents": list(map(int, candidate["agents"])),
        "requested_transition_count": 7,
        "per_action_time_limit_seconds": pp_time_limit,
        "executed_transition_count": len(transitions),
        "terminal_early_stop": len(transitions) < 7,
        "transitions": transitions,
        "runtime_or_ttf_stored": False,
        "official_counterfactual": False,
    }


def run_diagnostic(
    config_path: str | Path,
    output: str | Path,
    *,
    dry_run: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    plan = build_plan(config_path)
    if dry_run:
        return plan
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / "replay_plan.json"
    if plan_path.exists():
        if not resume or _read_json(plan_path) != plan:
            raise ValueError("output already contains a different replay plan")
    else:
        _write_json(plan_path, plan)
    results: list[dict[str, Any]] = []
    for state_row in plan["states"]:
        for candidate in state_row["candidates"]:
            destination = (
                root
                / "branches"
                / str(state_row["map_id"])
                / f"{candidate['role']}.json"
            )
            expected = {
                "map_id": state_row["map_id"],
                "source_state_fingerprint": state_row["state_fingerprint"],
                "role": candidate["role"],
                "first_candidate_id": candidate["candidate_id"],
                "first_candidate_agents": candidate["agents"],
            }
            if resume and destination.exists():
                result = _read_json(destination)
                if result.get("schema") != BRANCH_SCHEMA or any(
                    result.get(key) != value for key, value in expected.items()
                ):
                    raise ValueError(f"invalid resumed branch: {destination}")
            else:
                if destination.exists():
                    raise ValueError(f"branch output already exists: {destination}")
                result = run_branch(state_row, candidate)
                _write_json(destination, result)
            results.append(result)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "config_sha256": plan["config_sha256"],
        "plan_sha256": sha256_file(plan_path),
        "state_count": plan["state_count"],
        "branch_count": len(results),
        "native_pp_call_count": sum(
            int(row["executed_transition_count"]) for row in results
        ),
        "branch_error_count": 0,
        "total_process_time_limit_seconds": plan[
            "total_process_time_limit_seconds"
        ],
        "time_limit_rollback_count": sum(
            bool(transition["time_limit_rollback"])
            for row in results
            for transition in row["transitions"]
        ),
        "full_ttf_run": False,
        "ttf_claim_allowed": False,
        "official_counterfactual_included": False,
        "results": [
            {
                "map_id": row["map_id"],
                "role": row["role"],
                "executed_transition_count": row["executed_transition_count"],
                "terminal_early_stop": row["terminal_early_stop"],
                "first_replan_success": row["transitions"][0]["replan_success"],
                "first_repair_exact_noop": row["transitions"][0]["repair_exact_noop"],
                "initial_conflicts": row["transitions"][0]["conflicts_before"],
                "final_conflicts": row["transitions"][-1]["conflicts_after"],
            }
            for row in results
        ],
    }
    _write_json(root / "mechanism_report.json", report)
    return report


__all__ = [
    "BRANCH_SCHEMA",
    "CONFIG_SCHEMA",
    "EXPERIMENT_ID",
    "OBSERVED_ROLE",
    "PLAN_SCHEMA",
    "PRIMARY_ROLES",
    "REPORT_SCHEMA",
    "build_plan",
    "extract_action_candidates",
    "load_registration",
    "paired_pp_seed",
    "run_branch",
    "run_diagnostic",
    "select_checkpoint",
]
