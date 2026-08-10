from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from experiments._common import producer_identity, sha256_file
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.state_analysis import analyze_state, analyze_static_grid
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_maze_tail_action_replay import _replay_job
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.trace_replay import restore_repair_state, target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


REGISTRATION_SCHEMA = "lns2.stride.residualhazard_registration.v1"
STATE_SCHEMA = "lns2.stride.residualhazard_state.v1"
TRIAL_SCHEMA = "lns2.stride.residualhazard_trial.v1"
STATUS_SCHEMA = "lns2.stride.residualhazard_status.v1"
REPORT_SCHEMA = "lns2.stride.residualhazard_report.v1"
EXPERIMENT_ID = "stride-residualhazard-v1"
REPORT_FILENAME = "residualhazard_report.json"

MEASUREMENTS = (
    "normalized_residual_conflict_count",
    "selected_unselected_residual_pair_ratio",
    "new_residual_pair_ratio",
    "low_degree_residual_event_ratio",
    "largest_residual_component_agent_ratio",
    "residual_conflict_cell_herfindahl",
    "outside_boundary_queue_agent_ratio",
)
STRUCTURAL_MEASUREMENTS = MEASUREMENTS[1:]
TARGET_PRIORITY = (
    "new_residual_pair_ratio",
    "selected_unselected_residual_pair_ratio",
    "low_degree_residual_event_ratio",
    "residual_conflict_cell_herfindahl",
    "largest_residual_component_agent_ratio",
    "outside_boundary_queue_agent_ratio",
)
TAIL_CATEGORIES = frozenset({"adverse", "severe"})
CHALLENGERS = ("v2-plus-structpool", "v2-plus-slotpool")
ROLES = ("v2_action", "challenger_action")
TIE_EPSILON = 1e-12

PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/repair_collection.py",
    "experiments/state_analysis.py",
    "experiments/stride_collection.py",
    "experiments/stride_maze_tail_action_replay.py",
    "experiments/stride_repairability_collection.py",
    "experiments/stride_residualhazard.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _mean(values: Iterable[float | int]) -> float:
    numbers = [float(value) for value in values]
    return float(fmean(numbers)) if numbers else 0.0


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _contained_file(root: Path, value: Any, *, label: str) -> Path:
    text = str(value or "")
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a contained relative file")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes the project root") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    return resolved


def validate_registration_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != REGISTRATION_SCHEMA
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("scientific_status")
        != "preregistered_target_design_without_training"
    ):
        raise ValueError("ResidualHazard registration identity changed")
    parent = dict(config.get("parent") or {})
    if set(parent) != {
        "report",
        "report_sha256",
        "state_selection_sha256",
        "action_replay_trials_sha256",
    }:
        raise ValueError("ResidualHazard parent registry changed")
    cohort = dict(config.get("cohort") or {})
    if cohort != {
        "state_count": 66,
        "actions_per_state": 2,
        "trial_indices": list(range(16)),
        "paired_pp_seeds": True,
        "outcome_filtering": False,
        "training_use": False,
    }:
        raise ValueError("ResidualHazard cohort contract changed")
    if tuple(map(str, config.get("residual_measurements") or ())) != MEASUREMENTS:
        raise ValueError("ResidualHazard measurement registry changed")
    stability = dict(config.get("seed_stability") or {})
    if stability != {
        "minimum_same_direction_trials": 12,
        "fixed_halves": [list(range(8)), list(range(8, 16))],
        "both_halves_same_direction": True,
    }:
        raise ValueError("ResidualHazard seed-stability rule changed")
    readiness = dict(config.get("target_readiness") or {})
    if readiness != {
        "target_priority": list(TARGET_PRIORITY),
        "same_direction_on_all_current_maps": True,
        "same_direction_for_both_challengers": True,
        "minimum_seed_stable_tail_comparisons": 8,
        "minimum_seed_stable_tail_comparisons_with_no_worse_residual_conflict": 4,
        "must_not_be_explained_only_by_total_residual_conflicts": True,
        "weighted_post_hoc_target_forbidden": True,
    }:
        raise ValueError("ResidualHazard target-readiness rule changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary != {
        "model_training_allowed": False,
        "runtime_integration_allowed": False,
        "ttf_claim_allowed": False,
        "future_trajectory_features_allowed": False,
        "runtime_features_allowed": False,
    }:
        raise ValueError("ResidualHazard claim boundary changed")


def _load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], Path, dict[str, Any], list[dict[str, Any]]]:
    path = Path(config_path).resolve()
    root = path.parent.parent.resolve()
    config = _read_json(path)
    validate_registration_config(config)
    parent = dict(config["parent"])
    report_path = _contained_file(root, parent["report"], label="ResidualHazard report")
    if sha256_file(report_path) != str(parent["report_sha256"]):
        raise ValueError("ResidualHazard parent report hash changed")
    report = _read_json(report_path)
    if (
        report.get("schema") != "lns2.stride.maze_tail_action_replay_report.v1"
        or report.get("integrity_passed") is not True
        or report.get("action_stability_passed") is not True
        or report.get("predictor_design_allowed") is not True
        or report.get("model_training_allowed") is not False
        or int(report.get("state_count", -1)) != int(config["cohort"]["state_count"])
        or str(report["inputs"]["state_selection_sha256"])
        != str(parent["state_selection_sha256"])
        or str(report["inputs"]["action_replay_trials_sha256"])
        != str(parent["action_replay_trials_sha256"])
    ):
        raise ValueError("ResidualHazard parent report is ineligible")
    collection = report_path.parent
    expected_hashes = dict(report["inputs"]["state_artifact_sha256"])
    artifacts = sorted((collection / "states").glob("*.json"))
    if {item.name for item in artifacts} != set(expected_hashes):
        raise ValueError("ResidualHazard parent state coverage changed")
    states: list[dict[str, Any]] = []
    for artifact in artifacts:
        if sha256_file(artifact) != str(expected_hashes[artifact.name]):
            raise ValueError(f"ResidualHazard parent state hash changed: {artifact.name}")
        payload = _read_json(artifact)
        if payload.get("complete") is not True or payload.get("error") is not None:
            raise ValueError(f"ResidualHazard parent state is incomplete: {artifact.name}")
        states.append(payload)
    if len(states) != int(config["cohort"]["state_count"]):
        raise ValueError("ResidualHazard parent state count changed")
    if len({str(item["state_id"]) for item in states}) != len(states):
        raise ValueError("ResidualHazard parent state IDs are not unique")
    return path, root, config, report_path, report, states


def residual_structure_metrics(
    before: dict[str, Any], after: dict[str, Any], selected: set[int]
) -> dict[str, Any]:
    known = {int(agent["id"]) for agent in before.get("agents", ())}
    if not selected or not selected <= known:
        raise ValueError("ResidualHazard selected neighborhood is empty or illegal")
    static_grid = analyze_static_grid(before)
    before_analysis = analyze_state(before, static_grid=static_grid)
    after_analysis = analyze_state(after, static_grid=static_grid)
    before_pairs = set(before_analysis.pair_set)
    after_pairs = set(after_analysis.pair_set)
    reported_before = {
        tuple(sorted(map(int, edge))) for edge in before.get("conflict_edges", ())
    }
    reported_after = {
        tuple(sorted(map(int, edge))) for edge in after.get("conflict_edges", ())
    }
    if (
        before_pairs != reported_before
        or after_pairs != reported_after
        or len(before_pairs) != int(before.get("num_of_colliding_pairs", -1))
        or len(after_pairs) != int(after.get("num_of_colliding_pairs", -1))
    ):
        raise ValueError("ResidualHazard reconstructed conflict pairs changed")
    boundary = {
        pair for pair in after_pairs if (pair[0] in selected) != (pair[1] in selected)
    }
    new_pairs = after_pairs - before_pairs
    active = {agent for pair in after_pairs for agent in pair}
    outside_active = active - selected
    outside_boundary = {
        right if left in selected else left for left, right in boundary
    }
    largest_component = max(
        (len(members) for members in after_analysis.component_members.values()),
        default=0,
    )
    low_degree_events = sum(
        any(int(static_grid.degrees.get(cell, 0)) <= 2 for cell in event.cells)
        for event in after_analysis.events
    )
    cell_mass: Counter[int] = Counter()
    for event in after_analysis.events:
        contribution = 1.0 / max(1, len(event.cells))
        for cell in event.cells:
            cell_mass[int(cell)] += contribution
    total_mass = float(sum(cell_mass.values()))
    herfindahl = (
        sum((float(value) / total_mass) ** 2 for value in cell_mass.values())
        if total_mass
        else 0.0
    )
    return {
        "normalized_residual_conflict_count": _ratio(
            len(after_pairs), len(before_pairs)
        ),
        "selected_unselected_residual_pair_ratio": _ratio(
            len(boundary), len(after_pairs)
        ),
        "new_residual_pair_ratio": _ratio(len(new_pairs), len(after_pairs)),
        "low_degree_residual_event_ratio": _ratio(
            low_degree_events, len(after_analysis.events)
        ),
        "largest_residual_component_agent_ratio": _ratio(
            largest_component, len(active)
        ),
        "residual_conflict_cell_herfindahl": herfindahl,
        "outside_boundary_queue_agent_ratio": _ratio(
            len(outside_boundary), len(outside_active)
        ),
        "before_conflict_pair_count": len(before_pairs),
        "after_conflict_pair_count": len(after_pairs),
        "after_conflict_event_count": len(after_analysis.events),
        "new_residual_pair_count": len(new_pairs),
        "selected_unselected_residual_pair_count": len(boundary),
        "residual_active_agent_count": len(active),
        "outside_boundary_queue_agent_count": len(outside_boundary),
    }


def seed_stable_direction(differences: list[float], minimum: int = 12) -> dict[str, Any]:
    if len(differences) != 16:
        raise ValueError("ResidualHazard seed stability requires 16 paired values")
    positive = sum(value > TIE_EPSILON for value in differences)
    negative = sum(value < -TIE_EPSILON for value in differences)
    ties = len(differences) - positive - negative
    first = _mean(differences[:8])
    second = _mean(differences[8:])
    direction = (
        "challenger_higher"
        if positive >= minimum and first > 0.0 and second > 0.0
        else "challenger_lower"
        if negative >= minimum and first < 0.0 and second < 0.0
        else "uncertain"
    )
    return {
        "mean_delta": _mean(differences),
        "median_delta": float(sorted(differences)[7] + sorted(differences)[8]) / 2.0,
        "first_half_mean_delta": first,
        "second_half_mean_delta": second,
        "positive_count": positive,
        "negative_count": negative,
        "tie_count": ties,
        "seed_stable_direction": direction,
    }


def _parent_trial_index(parent: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for row in parent.get("trials", ()):
        key = (str(row["candidate_role"]), int(row["trial_index"]))
        if key in result:
            raise ValueError("ResidualHazard parent trial is duplicated")
        result[key] = dict(row)
    if set(result) != {(role, index) for role in ROLES for index in range(16)}:
        raise ValueError("ResidualHazard parent trial product changed")
    return result


def _state_artifact_valid(
    payload: dict[str, Any], *, state_id: str, run_fingerprint: str
) -> bool:
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("state_id") != state_id
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("complete") is not True
        or payload.get("error") is not None
        or payload.get("parent_reproduction_passed") is not True
    ):
        return False
    trials = payload.get("trials")
    if not isinstance(trials, list) or len(trials) != 32:
        return False
    keys = Counter(
        (str(row.get("candidate_role")), int(row.get("trial_index", -1)))
        for row in trials
    )
    if set(keys) != {(role, index) for role in ROLES for index in range(16)} or any(
        count != 1 for count in keys.values()
    ):
        return False
    for row in trials:
        if row.get("schema") != TRIAL_SCHEMA or any(
            name not in row or not math.isfinite(float(row[name]))
            for name in MEASUREMENTS
        ):
            return False
    return True


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    parent = dict(job["parent"])
    selection = dict(parent["selection"])
    state_id = str(parent["state_id"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        payload = _read_json(output_path)
        if _state_artifact_valid(
            payload, state_id=state_id, run_fingerprint=run_fingerprint
        ):
            return {
                "job_id": state_id,
                "state_id": state_id,
                "status": "resumed",
                "candidate_count": 2,
                "trial_count": 32,
                "error_count": 0,
            }
        raise ValueError("ResidualHazard resume artifact is invalid")
    source_root = Path(str(selection["v2_source_root"])).resolve()
    before, trace_path = target_state_from_trace(
        source_root,
        dict(selection["v2_manifest"]),
        decision_index=int(selection["decision_index"]),
        expected_fingerprint=str(selection["before_fingerprint"]),
    )
    if (
        state_fingerprint(before) != str(selection["before_fingerprint"])
        or int(before["num_of_colliding_pairs"]) != int(selection["before_conflicts"])
    ):
        raise RuntimeError("ResidualHazard reconstructed state changed")
    replay = _replay_job(selection)
    repair_fingerprint = repair_structure_fingerprint(before)
    if repair_fingerprint != str(parent["before_repair_fingerprint"]):
        raise RuntimeError("ResidualHazard parent repair fingerprint changed")
    restore_seed = repairability_restore_seed(repair_fingerprint)
    if restore_seed != int(parent["restore_seed"]):
        raise RuntimeError("ResidualHazard restore seed changed")
    parent_trials = _parent_trial_index(parent)
    actions = {str(row["role"]): dict(row) for row in parent["actions"]}
    if tuple(actions) != ROLES:
        raise RuntimeError("ResidualHazard parent actions changed")
    trials: list[dict[str, Any]] = []
    for role in ROLES:
        action = actions[role]
        agents = list(map(int, action["agents"]))
        for trial_index in range(16):
            environment, restored = restore_repair_state(replay, before, seed=restore_seed)
            if repair_structure_fingerprint(restored) != repair_fingerprint:
                raise RuntimeError("ResidualHazard branch restore changed")
            pp_seed = repairability_pp_seed(repair_fingerprint, trial_index)
            expected = parent_trials[(role, trial_index)]
            if pp_seed != int(expected["pp_seed"]):
                raise RuntimeError("ResidualHazard paired PP seed changed")
            result = _plain(environment.step(_paired_action(agents, pp_seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            after_fingerprint = repair_structure_fingerprint(after)
            after_conflicts = int(after["num_of_colliding_pairs"])
            if (
                after_fingerprint != str(expected["after_repair_fingerprint"])
                or after_conflicts != int(expected["conflicts_after"])
            ):
                raise RuntimeError("ResidualHazard parent repair outcome changed")
            residual = residual_structure_metrics(before, after, set(agents))
            trials.append(
                {
                    "schema": TRIAL_SCHEMA,
                    "state_id": state_id,
                    "candidate_role": role,
                    "candidate_id": str(action["candidate_id"]),
                    "trial_index": trial_index,
                    "pp_seed": pp_seed,
                    "after_repair_fingerprint": after_fingerprint,
                    "replan_success": bool(metrics["replan_success"]),
                    **residual,
                }
            )
    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "error": None,
        "state_id": state_id,
        "selection": selection,
        "source_trace_file": str(trace_path),
        "before_fingerprint": str(selection["before_fingerprint"]),
        "before_repair_fingerprint": repair_fingerprint,
        "restore_seed": restore_seed,
        "actions": [actions[role] for role in ROLES],
        "parent_reproduction_passed": True,
        "trials": trials,
    }
    if not _state_artifact_valid(
        payload, state_id=state_id, run_fingerprint=run_fingerprint
    ):
        raise RuntimeError("ResidualHazard state artifact failed validation")
    _write_json(output_path, payload)
    return {
        "job_id": state_id,
        "state_id": state_id,
        "status": "ok",
        "candidate_count": 2,
        "trial_count": 32,
        "error_count": 0,
    }


def _paired_state_report(payload: dict[str, Any], minimum: int) -> dict[str, Any]:
    selection = dict(payload["selection"])
    trials = list(payload["trials"])
    by_role = {
        role: sorted(
            [row for row in trials if row["candidate_role"] == role],
            key=lambda row: int(row["trial_index"]),
        )
        for role in ROLES
    }
    metric_reports = {}
    for measurement in MEASUREMENTS:
        differences = [
            float(right[measurement]) - float(left[measurement])
            for left, right in zip(
                by_role["v2_action"], by_role["challenger_action"]
            )
        ]
        metric_reports[measurement] = seed_stable_direction(differences, minimum)
    return {
        "state_id": str(payload["state_id"]),
        "map_id": str(selection["map_id"]),
        "task_id": str(selection["task_id"]),
        "solver_seed": int(selection["solver_seed"]),
        "challenger": str(selection["challenger"]),
        "tail_category": str(selection["tail_category"]),
        "decision_index": int(selection["decision_index"]),
        "before_conflicts": int(selection["before_conflicts"]),
        "metrics": metric_reports,
    }


def _tail_control_contrasts(
    states: list[dict[str, Any]], measurement: str, group: str
) -> dict[str, float | None]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        groups[str(state[group])].append(state)
    contrasts: dict[str, float | None] = {}
    for name, rows in sorted(groups.items()):
        tail = [
            float(row["metrics"][measurement]["mean_delta"])
            for row in rows
            if row["tail_category"] in TAIL_CATEGORIES
        ]
        control = [
            float(row["metrics"][measurement]["mean_delta"])
            for row in rows
            if row["tail_category"] not in TAIL_CATEGORIES
        ]
        contrasts[name] = _mean(tail) - _mean(control) if tail and control else None
    return contrasts


def analyze_residualhazard(
    config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, _root, config, parent_report_path, parent_report, parents = (
        _load_registration(config_path)
    )
    collection = Path(collection).resolve()
    run = _read_json(collection / "run_config.json")
    run_fingerprint = str(run["run_fingerprint"])
    expected_ids = {str(parent["state_id"]) for parent in parents}
    states: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted((collection / "states").glob("*.json")):
        payload = _read_json(path)
        state_id = str(payload.get("state_id"))
        if state_id not in expected_ids:
            errors.append(f"unexpected state artifact: {state_id}")
        elif not _state_artifact_valid(
            payload, state_id=state_id, run_fingerprint=run_fingerprint
        ):
            errors.append(f"invalid state artifact: {state_id}")
        else:
            states.append(payload)
    if {str(item["state_id"]) for item in states} != expected_ids:
        errors.append("ResidualHazard state coverage is incomplete")
    minimum = int(config["seed_stability"]["minimum_same_direction_trials"])
    state_reports = [
        _paired_state_report(item, minimum)
        for item in sorted(states, key=lambda row: str(row["state_id"]))
    ]
    readiness_config = dict(config["target_readiness"])
    measurement_reports: dict[str, Any] = {}
    for measurement in STRUCTURAL_MEASUREMENTS:
        map_contrasts = _tail_control_contrasts(
            state_reports, measurement, "map_id"
        )
        challenger_contrasts = _tail_control_contrasts(
            state_reports, measurement, "challenger"
        )
        stable_tail = [
            state
            for state in state_reports
            if state["tail_category"] in TAIL_CATEGORIES
            and state["metrics"][measurement]["seed_stable_direction"]
            == "challenger_higher"
        ]
        stable_tail_no_worse_conflict = [
            state
            for state in stable_tail
            if float(
                state["metrics"]["normalized_residual_conflict_count"]["mean_delta"]
            )
            <= TIE_EPSILON
        ]
        gates = {
            "same_direction_on_all_current_maps": len(map_contrasts) == 3
            and all(value is not None and value > 0.0 for value in map_contrasts.values()),
            "same_direction_for_both_challengers": set(challenger_contrasts)
            == set(CHALLENGERS)
            and all(
                value is not None and value > 0.0
                for value in challenger_contrasts.values()
            ),
            "minimum_seed_stable_tail_comparisons": len(stable_tail)
            >= int(readiness_config["minimum_seed_stable_tail_comparisons"]),
            "minimum_seed_stable_tail_comparisons_with_no_worse_residual_conflict": len(
                stable_tail_no_worse_conflict
            )
            >= int(
                readiness_config[
                    "minimum_seed_stable_tail_comparisons_with_no_worse_residual_conflict"
                ]
            ),
        }
        measurement_reports[measurement] = {
            "target_ready": all(gates.values()),
            "map_tail_minus_control_contrasts": map_contrasts,
            "challenger_tail_minus_control_contrasts": challenger_contrasts,
            "seed_stable_tail_count": len(stable_tail),
            "seed_stable_tail_no_worse_residual_conflict_count": len(
                stable_tail_no_worse_conflict
            ),
            "seed_stable_tail_state_ids": [row["state_id"] for row in stable_tail],
            "gates": gates,
        }
    ready = [
        measurement
        for measurement in TARGET_PRIORITY
        if measurement_reports[measurement]["target_ready"]
    ]
    selected_target = ready[0] if ready else None
    all_trials = [row for state in states for row in state["trials"]]
    integrity = {
        "parent_action_stability_passed": parent_report.get("action_stability_passed")
        is True,
        "exact_parent_state_coverage": {str(item["state_id"]) for item in states}
        == expected_ids,
        "two_actions_per_state": all(len(item["actions"]) == 2 for item in states),
        "sixteen_trials_per_action": len(all_trials) == len(expected_ids) * 32,
        "strictly_paired_pp_seeds": all(
            len(
                {
                    int(row["pp_seed"])
                    for row in item["trials"]
                    if int(row["trial_index"]) == trial_index
                }
            )
            == 1
            for item in states
            for trial_index in range(16)
        ),
        "exact_parent_repair_reproduction": all(
            item.get("parent_reproduction_passed") is True for item in states
        ),
        "zero_collection_errors": not errors,
        "all_registered_categories_retained": len(state_reports) == 66,
        "future_trajectory_features_forbidden": config["claim_boundary"][
            "future_trajectory_features_allowed"
        ]
        is False,
        "runtime_features_forbidden": config["claim_boundary"][
            "runtime_features_allowed"
        ]
        is False,
        "model_training_forbidden": config["claim_boundary"][
            "model_training_allowed"
        ]
        is False,
    }
    integrity_passed = all(integrity.values())
    target_readiness_passed = integrity_passed and selected_target is not None
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trial_path = output / "residual_trials.jsonl"
    state_path = output / "residual_state_metrics.jsonl"
    _write_jsonl(trial_path, all_trials)
    _write_jsonl(state_path, state_reports)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "residual_target_readiness_complete",
        "integrity_passed": integrity_passed,
        "target_readiness_passed": target_readiness_passed,
        "selected_target": selected_target,
        "ready_targets_in_frozen_priority_order": ready,
        "model_training_allowed": False,
        "runtime_integration_allowed": False,
        "formal_ttf_claim": False,
        "state_count": len(state_reports),
        "candidate_count": len(state_reports) * 2,
        "trial_count": len(all_trials),
        "tail_state_count": sum(
            row["tail_category"] in TAIL_CATEGORIES for row in state_reports
        ),
        "control_state_count": sum(
            row["tail_category"] not in TAIL_CATEGORIES for row in state_reports
        ),
        "measurement_reports": measurement_reports,
        "integrity_gates": integrity,
        "errors": errors,
        "states": state_reports,
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "parent_report_sha256": sha256_file(parent_report_path),
            "run_config_sha256": sha256_file(collection / "run_config.json"),
            "residual_trials_sha256": sha256_file(trial_path),
            "residual_state_metrics_sha256": sha256_file(state_path),
            "state_artifact_sha256": {
                path.name: sha256_file(path)
                for path in sorted((collection / "states").glob("*.json"))
            },
        },
        "next_step": (
            "preregister_new_map_training_cohort_before_any_model_fit"
            if target_readiness_passed
            else "stop_residualhazard_without_model_training"
        ),
    }
    _write_json(output / REPORT_FILENAME, report)
    return report


def collect_residualhazard(
    config_path: str | Path, output: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    config_path, root, config, parent_report_path, _report, parents = (
        _load_registration(config_path)
    )
    output = Path(output).resolve()
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    identity = {
        "schema": STATUS_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "parent_report_sha256": sha256_file(parent_report_path),
        "selected_state_ids": sorted(str(item["state_id"]) for item in parents),
        "measurements": list(MEASUREMENTS),
        "producer_identity": producer,
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("ResidualHazard output belongs to another run")
        if not resume:
            raise ValueError("ResidualHazard output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "states").mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    jobs = []
    for parent in sorted(parents, key=lambda row: str(row["state_id"])):
        key = _fingerprint(
            {
                "state_id": parent["state_id"],
                "before_fingerprint": parent["before_fingerprint"],
            }
        )[:20]
        jobs.append(
            {
                "job_id": str(parent["state_id"]),
                "state_id": str(parent["state_id"]),
                "row": dict(parent["selection"]["v2_manifest"]),
                "solver_seed": int(parent["selection"]["solver_seed"]),
                "parent": parent,
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    results = _run_jobs(
        _collect_state,
        jobs,
        1,
        phase="stride-residualhazard",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=1800.0,
    )
    _write_jsonl(output / "collection_manifest.jsonl", results)
    completed = sum(row.get("status") in {"ok", "resumed"} for row in results)
    errors = len(results) - completed
    status = {
        "schema": STATUS_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "total_state_count": len(parents),
        "completed_state_count": completed,
        "completed_candidate_count": completed * 2,
        "completed_trial_count": completed * 32,
        "error_state_count": errors,
        "complete": completed == len(parents) and errors == 0,
    }
    _write_json(output / "collection_status.json", status)
    if status["complete"]:
        report = analyze_residualhazard(config_path, output, output)
        status["report_sha256"] = sha256_file(output / REPORT_FILENAME)
        _write_json(output / "collection_status.json", status)
        return report
    return status


def smoke_residualhazard(config_path: str | Path, output: str | Path) -> dict[str, Any]:
    config_path, _root, _config, parent_report_path, _report, parents = (
        _load_registration(config_path)
    )
    parent = min(parents, key=lambda row: str(row["state_id"]))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "mode": "deterministic_single_state_smoke",
        "config_sha256": sha256_file(config_path),
        "parent_report_sha256": sha256_file(parent_report_path),
        "state_id": str(parent["state_id"]),
    }
    run_fingerprint = _fingerprint(identity)
    key = _fingerprint(
        {
            "state_id": parent["state_id"],
            "before_fingerprint": parent["before_fingerprint"],
        }
    )[:20]
    artifact = output / f"{key}.json"
    result = _collect_state(
        {
            "parent": parent,
            "output_path": str(artifact),
            "run_fingerprint": run_fingerprint,
            "resume": False,
        }
    )
    payload = _read_json(artifact)
    return {
        "schema": "lns2.stride.residualhazard_smoke.v1",
        "smoke_passed": _state_artifact_valid(
            payload,
            state_id=str(parent["state_id"]),
            run_fingerprint=run_fingerprint,
        ),
        "state_id": str(parent["state_id"]),
        "candidate_count": int(result["candidate_count"]),
        "trial_count": int(result["trial_count"]),
        "parent_reproduction_passed": payload.get("parent_reproduction_passed")
        is True,
        "artifact_sha256": sha256_file(artifact),
    }


__all__ = [
    "EXPERIMENT_ID",
    "MEASUREMENTS",
    "REPORT_FILENAME",
    "analyze_residualhazard",
    "collect_residualhazard",
    "residual_structure_metrics",
    "seed_stable_direction",
    "smoke_residualhazard",
    "validate_registration_config",
]
