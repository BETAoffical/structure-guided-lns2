from __future__ import annotations

import os
import statistics
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from experiments._common import producer_identity, sha256_file
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
from experiments.state_analysis import StaticGridAnalysis, analyze_state, analyze_static_grid
from experiments.stride_collection import _paired_action, _replay_job, _validate_native_repair
from experiments.stride_repairability_collection import (
    _source_target_state,
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_robustaction_label_collection import state_artifact_tree_sha256
from experiments.trace_replay import TARGET_STATE_RESTORE_CONTRACT, restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.safeslot_residual_teacher_config.v1"
STATE_SCHEMA = "lns2.stride.safeslot_residual_teacher_state.v1"
TRIAL_SCHEMA = "lns2.stride.safeslot_residual_teacher_trial.v1"
AGGREGATE_SCHEMA = "lns2.stride.safeslot_residual_teacher_action.v1"
COLLECTION_REPORT_SCHEMA = "lns2.stride.safeslot_residual_teacher_report.v1"
LABEL_SCHEMA = "lns2.stride.safeslot_safe_replace_label.v1"
ANALYSIS_REPORT_SCHEMA = "lns2.stride.safeslot_residual_teacher_analysis.v1"
COLLECTION_ID = "stride-safeslot-residual-teacher-v1"
RISK_COMPONENTS = (
    "conflict_event_ratio_to_before",
    "largest_component_ratio_to_before",
    "bottleneck_event_ratio_to_before",
    "repeated_event_excess_ratio_to_before",
)
FORBIDDEN_FIELDS = {
    "cost_to_go",
    "future_repair_rounds",
    "future_trajectory",
    "native_step_seconds",
    "pp_replan_seconds",
    "receding_q",
    "remaining_repair_rounds",
    "repair_runtime",
    "time_to_feasible",
    "ttf",
}
PRODUCER_FILES = (
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_repairability_collection.py",
    "experiments/stride_safeslot_residual_teacher.py",
    "experiments/trace_replay.py",
    "src/python_bindings.cpp",
)


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file() or sha256_file(path) != str(specification["sha256"]):
        raise ValueError(f"registered SafeSlot residual input changed: {path}")
    return path


def validate_safeslot_residual_teacher_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_outcome_unfiltered_paired_one_step_residual_teacher_collection"
        or config.get("collection_id") != COLLECTION_ID
        or config.get("implementation_id") != "stride-safeslot-v1"
        or config.get("pre_registration_parent_commit")
        != "464fef97126c29f2105d1c582cfd7195fcca3934"
    ):
        raise ValueError("SafeSlot residual-teacher identity changed")
    expected_inputs = {
        "readiness_report",
        "readiness_comparisons",
        "grid_report",
        "grid_manifest",
        "base_repair_trials",
        "grid_repair_trials",
        "slotpool_state_evaluation",
    }
    if set(config.get("inputs") or {}) != expected_inputs:
        raise ValueError("SafeSlot residual-teacher input registry changed")
    if dict(config.get("fixed_action_scope") or {}) != {
        "state_count": 98,
        "base_only_v2_anchor_per_state": 1,
        "slotpool_oof_candidates_per_state": 6,
        "anchor_slotpool_exact_overlap_state_count": 1,
        "unique_action_count": 685,
        "trial_count_per_action": 16,
        "total_trial_count": 10960,
        "selection_is_fixed_before_residual_outcomes": True,
        "outcome_based_candidate_filtering": False,
    }:
        raise ValueError("SafeSlot residual action scope changed")
    metrics = dict(config.get("residual_metrics") or {})
    if (
        tuple(map(str, metrics.get("raw_fields") or ()))
        != (
            "conflict_pair_count",
            "conflict_event_count",
            "active_conflict_agent_count",
            "largest_conflict_component_size",
            "bottleneck_conflict_event_count",
            "repeated_conflict_event_excess",
        )
        or tuple(map(str, metrics.get("risk_components") or ())) != RISK_COMPONENTS
        or metrics.get("risk_aggregation") != "unweighted_arithmetic_mean"
        or metrics.get("bottleneck_definition")
        != "event_touches_static_articulation_or_degree_at_most_two"
        or metrics.get("repeated_event_excess")
        != "conflict_event_count_minus_unique_conflict_pair_count"
    ):
        raise ValueError("SafeSlot residual metrics changed")
    label = dict(config.get("final_label_contract") or {})
    if (
        label.get("requires_pre_residual_positive") is not True
        or float(label.get("mean_candidate_minus_anchor_risk_maximum", 1.0)) != 0.0
        or float(
            label.get("each_fixed_half_candidate_minus_anchor_risk_maximum", 1.0)
        )
        != 0.0
        or float(label.get("maximum_per_component_mean_regression", -1.0)) != 0.02
        or tuple(map(int, label.get("trial_indices") or ())) != tuple(range(16))
        or tuple(map(int, label.get("first_fixed_half") or ())) != tuple(range(8))
        or tuple(map(int, label.get("second_fixed_half") or ()))
        != tuple(range(8, 16))
    ):
        raise ValueError("SafeSlot final label contract changed")
    if dict(config.get("analysis_gates") or {}) != {
        "minimum_final_positive_state_count": 20,
        "minimum_final_positive_candidate_count": 40,
        "minimum_final_positive_map_count": 8,
    }:
        raise ValueError("SafeSlot residual analysis gates changed")
    execution = dict(config.get("execution") or {})
    if execution != {
        "workers": 2,
        "per_state_timeout_seconds": 7200,
        "monitor_interval_minutes": 30,
        "resume_required": True,
    }:
        raise ValueError("SafeSlot residual execution contract changed")
    boundary = dict(config.get("claim_boundary") or {})
    if set(boundary) != {
        "model_training_allowed_before_completed_analysis",
        "runtime_integration_allowed",
        "formal_ttf_claim",
        "runtime_or_ttf_stored",
        "future_trajectory_stored",
        "cost_to_go_stored",
        "remaining_repair_rounds_stored",
        "known_maze_result_used",
    } or any(boundary.values()):
        raise ValueError("SafeSlot residual claim boundary changed")
    if dict(config.get("outputs") or {}) != {
        "collection": "build/stride-safeslot-residual-teacher-v1",
        "analysis": "build/stride-safeslot-residual-teacher-analysis-v1",
    }:
        raise ValueError("SafeSlot residual outputs changed")
    if project_root is not None:
        for specification in dict(config["inputs"]).values():
            _registered(project_root, dict(specification))
        state_root = (
            project_root / str(config["grid_state_artifact_root"])
        ).resolve()
        if state_artifact_tree_sha256(state_root) != str(
            config["grid_state_artifact_tree_sha256"]
        ):
            raise ValueError("SafeSlot residual grid state tree changed")


def residual_structure_metrics(
    state: dict[str, Any], *, static_grid: StaticGridAnalysis | None = None
) -> dict[str, int]:
    analysis = analyze_state(state, static_grid=static_grid)
    active_agents = {agent for pair in analysis.pair_set for agent in pair}
    largest_component = max(
        (len(members) for members in analysis.component_members.values()), default=0
    )
    bottleneck_events = 0
    for event in analysis.events:
        if any(
            cell in analysis.articulation or int(analysis.degrees.get(cell, 0)) <= 2
            for cell in event.cells
        ):
            bottleneck_events += 1
    event_count = len(analysis.events)
    pair_count = len(analysis.pair_set)
    return {
        "conflict_pair_count": pair_count,
        "conflict_event_count": event_count,
        "active_conflict_agent_count": len(active_agents),
        "largest_conflict_component_size": largest_component,
        "bottleneck_conflict_event_count": bottleneck_events,
        "repeated_conflict_event_excess": max(0, event_count - pair_count),
    }


def residual_risk_components(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, float]:
    mapping = {
        "conflict_event_ratio_to_before": "conflict_event_count",
        "largest_component_ratio_to_before": "largest_conflict_component_size",
        "bottleneck_event_ratio_to_before": "bottleneck_conflict_event_count",
        "repeated_event_excess_ratio_to_before": "repeated_conflict_event_excess",
    }
    return {
        name: float(after[field]) / max(1, int(before[field]))
        for name, field in mapping.items()
    }


def compare_residual_teacher(
    candidate: dict[str, Any],
    anchor: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    mean_delta = float(candidate["mean_residual_risk"]) - float(
        anchor["mean_residual_risk"]
    )
    first_delta = float(candidate["first_fixed_half_mean_residual_risk"]) - float(
        anchor["first_fixed_half_mean_residual_risk"]
    )
    second_delta = float(candidate["second_fixed_half_mean_residual_risk"]) - float(
        anchor["second_fixed_half_mean_residual_risk"]
    )
    component_deltas = {
        name: float(candidate["component_mean_residual_risk"][name])
        - float(anchor["component_mean_residual_risk"][name])
        for name in RISK_COMPONENTS
    }
    half_maximum = float(
        contract["each_fixed_half_candidate_minus_anchor_risk_maximum"]
    )
    passed = bool(
        mean_delta
        <= float(contract["mean_candidate_minus_anchor_risk_maximum"])
        and first_delta <= half_maximum
        and second_delta <= half_maximum
        and all(
            delta <= float(contract["maximum_per_component_mean_regression"])
            for delta in component_deltas.values()
        )
    )
    return {
        "residual_teacher_pass": passed,
        "mean_residual_risk_delta": mean_delta,
        "first_fixed_half_residual_risk_delta": first_delta,
        "second_fixed_half_residual_risk_delta": second_delta,
        "component_mean_residual_risk_deltas": component_deltas,
    }


def _forbidden_hits(value: Any) -> list[str]:
    hits: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_FIELDS:
                hits.append(lowered)
            hits.extend(_forbidden_hits(child))
    elif isinstance(value, list):
        for child in value:
            hits.extend(_forbidden_hits(child))
    return hits


def _grid_state_path(state_root: Path, row: dict[str, Any]) -> Path:
    filename = PurePosixPath(str(row["state_file"]).replace("\\", "/")).name
    path = state_root / filename
    if not path.is_file() or sha256_file(path) != str(row["state_file_sha256"]):
        raise ValueError(f"SafeSlot residual grid state changed: {path}")
    return path


def _action_aggregate(
    *, state_id: str, action: dict[str, Any], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    ordered = sorted(trials, key=lambda row: int(row["trial_index"]))
    if tuple(int(row["trial_index"]) for row in ordered) != tuple(range(16)):
        raise ValueError(f"SafeSlot residual trial matrix changed: {state_id}")
    risk = [float(row["residual_risk"]) for row in ordered]
    component_means = {
        name: statistics.fmean(
            float(row["residual_risk_components"][name]) for row in ordered
        )
        for name in RISK_COMPONENTS
    }
    return {
        "schema": AGGREGATE_SCHEMA,
        "state_id": state_id,
        "candidate_id": str(action["candidate_id"]),
        "candidate_role": str(action["candidate_role"]),
        "agents": list(map(int, action["agents"])),
        "actual_size": len(action["agents"]),
        "trial_count": 16,
        "mean_residual_risk": statistics.fmean(risk),
        "first_fixed_half_mean_residual_risk": statistics.fmean(risk[:8]),
        "second_fixed_half_mean_residual_risk": statistics.fmean(risk[8:]),
        "component_mean_residual_risk": component_means,
    }


def _collect_residual_state(job: dict[str, Any]) -> dict[str, Any]:
    grid_path = Path(str(job["grid_path"]))
    grid = _read_json(grid_path)
    state_id = str(grid["state_id"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if (
            existing.get("schema") == STATE_SCHEMA
            and existing.get("run_fingerprint") == run_fingerprint
            and existing.get("complete") is True
            and not _forbidden_hits(existing)
        ):
            return {
                "state_id": state_id,
                "state_file": str(output_path),
                "status": "resumed",
                "action_count": len(existing["action_aggregates"]),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"invalid completed SafeSlot residual state: {output_path}")

    replay = _replay_job(dict(grid["decision"]))
    state, source_manifest, source_trace_path = _source_target_state(
        dict(grid["decision"])
    )
    before_fingerprint = state_fingerprint(state)
    before_repair_fingerprint = repair_structure_fingerprint(state)
    if (
        before_fingerprint != str(grid["before_fingerprint"])
        or before_repair_fingerprint != str(grid["before_repair_fingerprint"])
        or int(state["num_of_colliding_pairs"]) != int(grid["before_conflicts"])
    ):
        raise RuntimeError("SafeSlot residual source state differs from grid")
    static_grid = analyze_static_grid(state)
    before_metrics = residual_structure_metrics(state, static_grid=static_grid)
    restore_seed = repairability_restore_seed(before_repair_fingerprint)
    actions = list(job["actions"])
    expected_trials = dict(job["expected_trials"])
    all_trials: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    for action in actions:
        candidate_id = str(action["candidate_id"])
        agents = list(map(int, action["agents"]))
        expected = sorted(
            list(expected_trials[candidate_id]), key=lambda row: int(row["trial_index"])
        )
        if tuple(int(row["trial_index"]) for row in expected) != tuple(range(16)):
            raise RuntimeError("SafeSlot residual expected trial matrix changed")
        action_trials: list[dict[str, Any]] = []
        for expected_row in expected:
            trial_index = int(expected_row["trial_index"])
            pp_seed = repairability_pp_seed(before_repair_fingerprint, trial_index)
            if int(expected_row["pp_seed"]) != pp_seed:
                raise RuntimeError("SafeSlot residual expected PP seed changed")
            branch_environment, branch_state = restore_repair_state(
                replay, state, seed=restore_seed
            )
            if repair_structure_fingerprint(branch_state) != before_repair_fingerprint:
                raise RuntimeError("SafeSlot residual branch restore changed")
            result = _plain(branch_environment.step(_paired_action(agents, pp_seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            after_repair_fingerprint = repair_structure_fingerprint(after)
            conflicts_after = int(after["num_of_colliding_pairs"])
            if (
                conflicts_after != int(expected_row["conflicts_after"])
                or after_repair_fingerprint
                != str(expected_row["after_repair_fingerprint"])
                or bool(metrics["replan_success"])
                != bool(expected_row["replan_success"])
            ):
                raise RuntimeError("SafeSlot residual rerun outcome differs from label")
            after_metrics = residual_structure_metrics(after, static_grid=static_grid)
            components = residual_risk_components(before_metrics, after_metrics)
            residual_risk = statistics.fmean(components.values())
            action_trials.append(
                {
                    "schema": TRIAL_SCHEMA,
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "candidate_role": str(action["candidate_role"]),
                    "trial_index": trial_index,
                    "pp_seed": pp_seed,
                    "before_fingerprint": before_fingerprint,
                    "before_repair_fingerprint": before_repair_fingerprint,
                    "after_repair_fingerprint": after_repair_fingerprint,
                    "conflicts_after": conflicts_after,
                    "normalized_conflict_reduction": float(
                        expected_row["normalized_conflict_reduction"]
                    ),
                    "replan_success": bool(metrics["replan_success"]),
                    "after_residual_metrics": after_metrics,
                    "residual_risk_components": components,
                    "residual_risk": residual_risk,
                }
            )
        all_trials.extend(action_trials)
        aggregates.append(
            _action_aggregate(
                state_id=state_id, action=action, trials=action_trials
            )
        )
    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": state_id,
        "decision": dict(grid["decision"]),
        "before_fingerprint": before_fingerprint,
        "before_repair_fingerprint": before_repair_fingerprint,
        "before_residual_metrics": before_metrics,
        "state_restore": {
            "contract": TARGET_STATE_RESTORE_CONTRACT,
            "restore_seed": restore_seed,
            "source_trace_file": str(source_manifest["trace_file"]),
            "source_trace_path": str(source_trace_path),
        },
        "trials": all_trials,
        "action_aggregates": aggregates,
        "runtime_or_ttf_stored": False,
        "future_trajectory_stored": False,
    }
    if len(all_trials) != len(actions) * 16 or _forbidden_hits(payload):
        raise RuntimeError("SafeSlot residual state product is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": state_id,
        "state_file": str(output_path),
        "status": "ok",
        "action_count": len(actions),
        "trial_count": len(all_trials),
    }


def _trial_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    return dict(result)


def collect_safeslot_residual_teacher(
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = _read_json(config_path)
    validate_safeslot_residual_teacher_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    readiness = _read_json(inputs["readiness_report"])
    grid_report = _read_json(inputs["grid_report"])
    if (
        readiness.get("readiness_passed") is not True
        or readiness.get("post_state_collection_authorized") is not True
        or grid_report.get("passed") is not True
    ):
        raise ValueError("SafeSlot residual collection is not authorized")

    scope = dict(config["fixed_action_scope"])
    grid_manifest = _read_jsonl(inputs["grid_manifest"])
    slot_rows = {
        str(row["state_id"]): row
        for row in _read_jsonl(inputs["slotpool_state_evaluation"])
    }
    base_trials = _trial_index(_read_jsonl(inputs["base_repair_trials"]))
    grid_trials = _trial_index(_read_jsonl(inputs["grid_repair_trials"]))
    if len(grid_manifest) != int(scope["state_count"]) or len(slot_rows) != int(
        scope["state_count"]
    ):
        raise ValueError("SafeSlot residual state coverage changed")

    identity = {
        "schema": CONFIG_SCHEMA,
        "collection_id": COLLECTION_ID,
        "config_sha256": sha256_file(config_path),
        "input_sha256": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "producer": producer_identity(
            project_root=project_root,
            source_files=PRODUCER_FILES,
            native_required=True,
        ),
    }
    run_fingerprint = _fingerprint(identity)
    output = Path(output).resolve()
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("SafeSlot residual output belongs to another run")
        if not resume:
            raise ValueError("SafeSlot residual output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})

    state_root = (
        project_root / str(config["grid_state_artifact_root"])
    ).resolve()
    jobs: list[dict[str, Any]] = []
    total_actions = 0
    for manifest in sorted(grid_manifest, key=lambda row: str(row["state_id"])):
        grid_path = _grid_state_path(state_root, manifest)
        grid = _read_json(grid_path)
        state_id = str(grid["state_id"])
        slot = slot_rows.get(state_id)
        if slot is None or int(slot["selected_candidate_count"]) != 6:
            raise ValueError(f"SafeSlot residual SlotPool scope changed: {state_id}")
        grid_candidates = {
            str(row["candidate_id"]): row for row in list(grid["candidates"])
        }
        anchor = dict(grid["v2_base_anchor"])
        anchor_id = str(anchor["candidate_id"])
        actions: dict[str, dict[str, Any]] = {
            anchor_id: {
                "candidate_id": anchor_id,
                "candidate_role": "base_only_v2_anchor",
                "agents": list(map(int, anchor["agents"])),
            }
        }
        expected: dict[str, list[dict[str, Any]]] = {
            anchor_id: list(base_trials[(state_id, anchor_id)])
        }
        for candidate_id in map(str, slot["selected_candidate_ids"]):
            candidate = grid_candidates.get(candidate_id)
            if candidate is None:
                raise ValueError(f"SafeSlot residual candidate is absent: {state_id}")
            if candidate_id == anchor_id:
                actions[candidate_id]["candidate_role"] = "base_anchor_slotpool_overlap"
                continue
            actions[candidate_id] = {
                "candidate_id": candidate_id,
                "candidate_role": "slotpool_structural_challenger",
                "agents": list(map(int, candidate["agents"])),
            }
            expected[candidate_id] = list(grid_trials[(state_id, candidate_id)])
        total_actions += len(actions)
        jobs.append(
            {
                "job_id": state_id,
                "state_id": state_id,
                "row": {
                    "split": str(grid["decision"]["split"]),
                    "map_id": str(grid["decision"]["map_id"]),
                    "task_id": str(grid["decision"]["task_id"]),
                    "agent_count": int(grid["decision"]["agent_count"]),
                    "layout_mode": str(grid["decision"]["layout_mode"]),
                    "task_variant": grid["decision"].get("task_variant"),
                },
                "solver_seed": int(grid["decision"]["solver_seed"]),
                "grid_path": str(grid_path),
                "actions": [actions[key] for key in sorted(actions)],
                "expected_trials": expected,
                "output_path": str(
                    output / "states" / f"{_fingerprint({'state_id': state_id})[:20]}.json"
                ),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    if total_actions != int(scope["unique_action_count"]):
        raise ValueError("SafeSlot residual unique action count changed")

    status_path = output / "collection_status.json"
    observed: list[dict[str, Any]] = []

    def update_status(result: dict[str, Any]) -> None:
        observed.append(result)
        failures = [row for row in observed if row.get("status") in {"error", "timeout"}]
        _write_json(
            status_path,
            {
                "schema": COLLECTION_REPORT_SCHEMA,
                "status": "running",
                "requested_state_count": len(jobs),
                "completed_state_count": len(observed) - len(failures),
                "completed_action_count": sum(
                    int(row.get("action_count", 0)) for row in observed
                ),
                "completed_trial_count": sum(
                    int(row.get("trial_count", 0)) for row in observed
                ),
                "error_state_count": sum(row.get("status") == "error" for row in failures),
                "timeout_state_count": sum(row.get("status") == "timeout" for row in failures),
                "active_jobs": max(
                    0,
                    min(
                        int(workers or config["execution"]["workers"]),
                        len(jobs) - len(observed),
                    ),
                ),
                "errors": failures,
            },
        )

    _write_json(
        status_path,
        {
            "schema": COLLECTION_REPORT_SCHEMA,
            "status": "running",
            "requested_state_count": len(jobs),
            "completed_state_count": 0,
            "completed_action_count": 0,
            "completed_trial_count": 0,
            "error_state_count": 0,
            "timeout_state_count": 0,
            "active_jobs": min(
                int(workers or config["execution"]["workers"]), len(jobs)
            ),
            "errors": [],
        },
    )
    observed = _run_jobs(
        _collect_residual_state,
        jobs,
        int(workers or config["execution"]["workers"]),
        phase="stride-safeslot-residual-teacher",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["execution"]["per_state_timeout_seconds"]),
        on_result=update_status,
    )
    failures = [row for row in observed if row.get("status") in {"error", "timeout"}]
    successes = [row for row in observed if row.get("status") in {"ok", "resumed"}]
    trials: list[dict[str, Any]] = []
    aggregates: list[dict[str, Any]] = []
    state_manifest: list[dict[str, Any]] = []
    for result in sorted(successes, key=lambda row: str(row["state_id"])):
        payload = _read_json(Path(str(result["state_file"])))
        trials.extend(payload["trials"])
        aggregates.extend(payload["action_aggregates"])
        state_manifest.append(
            {
                "state_id": str(payload["state_id"]),
                "action_count": len(payload["action_aggregates"]),
                "trial_count": len(payload["trials"]),
                "state_file": str(result["state_file"]),
                "state_file_sha256": sha256_file(Path(str(result["state_file"]))),
            }
        )
    passed = bool(
        len(state_manifest) == int(scope["state_count"])
        and len(aggregates) == int(scope["unique_action_count"])
        and len(trials) == int(scope["total_trial_count"])
        and not failures
        and not _forbidden_hits([trials, aggregates])
    )
    artifacts: dict[str, str] = {}
    if passed:
        trials_path = output / "residual_trials.jsonl"
        aggregates_path = output / "residual_action_aggregates.jsonl"
        manifest_path = output / "state_manifest.jsonl"
        _write_jsonl(trials_path, trials)
        _write_jsonl(aggregates_path, aggregates)
        _write_jsonl(manifest_path, state_manifest)
        artifacts = {
            "residual_trials_sha256": sha256_file(trials_path),
            "residual_action_aggregates_sha256": sha256_file(aggregates_path),
            "state_manifest_sha256": sha256_file(manifest_path),
            "state_artifact_tree_sha256": state_artifact_tree_sha256(output / "states"),
        }
    report = {
        "schema": COLLECTION_REPORT_SCHEMA,
        "scientific_status": "paired_one_step_residual_teacher_complete" if passed else "failed",
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(jobs),
        "completed_state_count": len(state_manifest),
        "action_count": len(aggregates),
        "trial_count": len(trials),
        "error_state_count": sum(row.get("status") == "error" for row in failures),
        "timeout_state_count": sum(row.get("status") == "timeout" for row in failures),
        "exact_prior_outcome_reproduction": passed,
        "runtime_or_ttf_stored": False,
        "future_trajectory_stored": False,
        "model_training_allowed": False,
        "formal_ttf_claim": False,
        "passed": passed,
        "errors": failures,
        "artifacts": artifacts,
    }
    _write_json(output / "collection_report.json", report)
    _write_json(status_path, {**report, "status": "complete" if passed else "failed"})
    return report


def analyze_safeslot_residual_teacher(
    config_path: str | Path,
    collection: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parent.parent
    config = _read_json(config_path)
    validate_safeslot_residual_teacher_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    collection = Path(collection).resolve()
    collection_report = _read_json(collection / "collection_report.json")
    if collection_report.get("passed") is not True:
        raise ValueError("SafeSlot residual collection did not pass")
    readiness_rows = _read_jsonl(inputs["readiness_comparisons"])
    aggregates = {
        (str(row["state_id"]), str(row["candidate_id"])): row
        for row in _read_jsonl(collection / "residual_action_aggregates.jsonl")
    }
    contract = dict(config["final_label_contract"])
    labels: list[dict[str, Any]] = []
    positive_states: set[str] = set()
    positive_maps: set[str] = set()
    final_positive_count = 0
    pre_positive_selected_count = 0
    for row in readiness_rows:
        if not bool(row["slotpool_oof_retained"]):
            continue
        state_id = str(row["state_id"])
        candidate_id = str(row["challenger_candidate_id"])
        anchor_id = str(row["anchor_candidate_id"])
        candidate = aggregates[(state_id, candidate_id)]
        anchor = aggregates[(state_id, anchor_id)]
        residual = compare_residual_teacher(candidate, anchor, contract)
        pre_positive = bool(row["pre_residual_positive"])
        pre_positive_selected_count += int(pre_positive)
        residual_pass = bool(residual["residual_teacher_pass"])
        final_positive = pre_positive and residual_pass
        if final_positive:
            final_positive_count += 1
            positive_states.add(state_id)
            positive_maps.add(str(row["map_id"]))
        labels.append(
            {
                "schema": LABEL_SCHEMA,
                "state_id": state_id,
                "map_id": str(row["map_id"]),
                "layout_mode": str(row["layout_mode"]),
                "anchor_candidate_id": anchor_id,
                "challenger_candidate_id": candidate_id,
                "pre_residual_positive": pre_positive,
                "residual_teacher_pass": residual_pass,
                "final_safe_replace_positive": final_positive,
                "mean_residual_risk_delta": residual["mean_residual_risk_delta"],
                "first_fixed_half_residual_risk_delta": residual[
                    "first_fixed_half_residual_risk_delta"
                ],
                "second_fixed_half_residual_risk_delta": residual[
                    "second_fixed_half_residual_risk_delta"
                ],
                "component_mean_residual_risk_deltas": residual[
                    "component_mean_residual_risk_deltas"
                ],
            }
        )
    gates = dict(config["analysis_gates"])
    checks = {
        "final_positive_state_count": len(positive_states)
        >= int(gates["minimum_final_positive_state_count"]),
        "final_positive_candidate_count": final_positive_count
        >= int(gates["minimum_final_positive_candidate_count"]),
        "final_positive_map_count": len(positive_maps)
        >= int(gates["minimum_final_positive_map_count"]),
    }
    passed = all(checks.values())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "safe_replace_labels.jsonl"
    _write_jsonl(labels_path, labels)
    report = {
        "schema": ANALYSIS_REPORT_SCHEMA,
        "scientific_status": (
            "residual_teacher_labels_ready_for_map_grouped_training_design"
            if passed
            else "residual_teacher_labels_insufficient_stop_before_training"
        ),
        "integrity_passed": True,
        "selected_challenger_count": len(labels),
        "pre_residual_positive_selected_count": pre_positive_selected_count,
        "final_positive_candidate_count": final_positive_count,
        "final_positive_state_count": len(positive_states),
        "final_positive_map_count": len(positive_maps),
        "final_positive_maps": sorted(positive_maps),
        "analysis_checks": checks,
        "analysis_passed": passed,
        "model_training_design_allowed": passed,
        "runtime_integration_allowed": False,
        "formal_ttf_claim": False,
        "next_decision": (
            "preregister_whole_map_grouped_safeslot_gate_training"
            if passed
            else "stop_and_reassess_residual_label_without_outcome_filtering"
        ),
        "artifacts": {"safe_replace_labels_sha256": sha256_file(labels_path)},
    }
    _write_json(output / "residual_teacher_analysis_report.json", report)
    return report
