from __future__ import annotations

import collections
import concurrent.futures
import math
from pathlib import Path
from typing import Any

from experiments._common import (
    _native_filesystem_path,
    registered_input,
    sha256_file,
)
from experiments.closed_loop_trace_storage import (
    TRACE_FORMAT_DELTA_GZIP_V2,
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
)
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_pretail_forced_continuation import (
    _collection_path,
    _paired_first_action_seed,
)
from experiments.stride_tailswitch import _mean
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.successor_state_inventory_registration.v1"
REPORT_SCHEMA = "lns2.stride.successor_state_inventory_report.v1"
ROW_SCHEMA = "lns2.stride.successor_state_inventory_episode.v1"
STATUS_SCHEMA = "lns2.stride.successor_state_inventory_status.v1"
EXPERIMENT_ID = "stride-successor-state-inventory-v1"
EXPECTED_PARENT = "2a0cb3de6370460634f3f701cab1824186b796f0"


def _manifest_index_sha256(collection: Path) -> tuple[int, str]:
    rows = [
        {
            "path": path.relative_to(collection).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(collection.rglob("realized_dynamic_manifest.jsonl"))
    ]
    return len(rows), _fingerprint(rows)


def _edge_signature(state: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(tuple(map(int, edge)) for edge in state["conflict_edges"]))


def _apply_transition(
    state: dict[str, Any], event: dict[str, Any]
) -> dict[str, Any]:
    if state_fingerprint(state) != str(event.get("before_fingerprint")):
        raise ValueError("transition before fingerprint changed")
    after = apply_state_delta(state, dict(event["state_delta"]))
    after.update(apply_extras_delta(state, dict(event["state_extras_delta"])))
    if state_fingerprint(after) != str(event.get("after_fingerprint")):
        raise ValueError("transition after fingerprint changed")
    return after


def successor_dynamics(
    initial_state: dict[str, Any], transitions: list[dict[str, Any]]
) -> dict[str, Any]:
    if not transitions or int(transitions[0].get("decision_index", -1)) != 0:
        raise ValueError("forced transition zero is missing")
    successor = _apply_transition(initial_state, transitions[0])
    successor_full = state_fingerprint(successor)
    successor_repair = repair_structure_fingerprint(successor)
    successor_conflicts = int(successor["num_of_colliding_pairs"])
    successor_edges = _edge_signature(successor)
    state = successor
    exact_noop_streak = 0
    conflict_signature_streak = 0
    first_state_change: int | None = None
    first_strict_drop: int | None = None
    left_successor = False
    returned_after_departure = False
    for offset, event in enumerate(transitions[1:], start=1):
        if int(event.get("decision_index", -1)) != offset:
            raise ValueError("transition decision indices are not contiguous")
        before_full = state_fingerprint(state)
        after = _apply_transition(state, event)
        after_full = state_fingerprint(after)
        if first_state_change is None and after_full != successor_full:
            first_state_change = offset
            left_successor = True
        elif left_successor and before_full == successor_full:
            returned_after_departure = True
        if first_strict_drop is None and int(
            after["num_of_colliding_pairs"]
        ) < successor_conflicts:
            first_strict_drop = offset
        if (
            offset == exact_noop_streak + 1
            and before_full == successor_full
            and after_full == successor_full
        ):
            exact_noop_streak += 1
        if (
            offset == conflict_signature_streak + 1
            and _edge_signature(state) == successor_edges
            and _edge_signature(after) == successor_edges
        ):
            conflict_signature_streak += 1
        state = after
    return {
        "successor_fingerprint": successor_full,
        "successor_repair_fingerprint": successor_repair,
        "successor_conflicts": successor_conflicts,
        "successor_feasible": bool(successor.get("feasible")),
        "forced_state_changed": state_fingerprint(initial_state) != successor_full,
        "forced_repair_state_changed": repair_structure_fingerprint(initial_state)
        != successor_repair,
        "forced_strict_conflict_drop": successor_conflicts
        < int(initial_state["num_of_colliding_pairs"]),
        "forced_conflict_delta": int(initial_state["num_of_colliding_pairs"])
        - successor_conflicts,
        "following_transition_count": max(0, len(transitions) - 1),
        "exact_successor_noop_streak": exact_noop_streak,
        "successor_conflict_signature_noop_streak": conflict_signature_streak,
        "decisions_to_first_state_change": first_state_change,
        "state_change_right_censored": first_state_change is None
        and not bool(successor.get("feasible")),
        "decisions_to_first_strict_conflict_drop": first_strict_drop,
        "strict_drop_right_censored": first_strict_drop is None
        and not bool(successor.get("feasible")),
        "returned_to_exact_successor_after_departure": returned_after_departure,
        "final_reconstructed_fingerprint": state_fingerprint(state),
    }


def _single_manifest(collection: Path, item: dict[str, Any]) -> dict[str, Any]:
    rows = _read_jsonl(collection / "realized_dynamic_manifest.jsonl")
    matching = [
        row
        for row in rows
        if str(row.get("task_id")) == str(item["task_id"])
        and int(row.get("solver_seed", -1)) == int(item["solver_seed"])
    ]
    if len(matching) != 1:
        raise ValueError("schedule entry does not identify exactly one manifest row")
    return dict(matching[0])


def _episode_row(task: dict[str, Any]) -> dict[str, Any]:
    item = dict(task["item"])
    collection = Path(str(task["collection"]))
    checkpoint = dict(task["checkpoint"])
    manifest = _single_manifest(collection, item)
    if manifest.get("status") != "ok" or manifest.get("error") is not None:
        raise ValueError("registered PreTail episode is not successful execution")
    if str(manifest.get("trace_format")) != TRACE_FORMAT_DELTA_GZIP_V2:
        raise ValueError("registered PreTail trace format changed")
    trace_relative = Path(str(manifest["trace_file"]))
    trace_path = (collection / trace_relative).resolve()
    try:
        trace_path.relative_to(collection.resolve())
    except ValueError as error:
        raise ValueError("trace escaped its collection") from error
    trace_read_path = _native_filesystem_path(trace_path)
    if sha256_file(trace_read_path) != str(manifest["trace_sha256"]):
        raise ValueError("trace SHA-256 changed")
    events = read_trace_events(trace_read_path)
    if len(events) < 3 or events[0].get("event") != "initial":
        raise ValueError("trace event sequence is incomplete")
    transitions = [dict(event) for event in events[1:-1]]
    if any(event.get("event") != "transition" for event in transitions):
        raise ValueError("trace contains a non-transition body event")
    initial = dict(events[0])
    blob_relative = Path(str(initial["state_blob"]))
    blob_path = (collection / blob_relative).resolve()
    try:
        blob_path.relative_to(collection.resolve())
    except ValueError as error:
        raise ValueError("initial state blob escaped its collection") from error
    state = read_state_blob(_native_filesystem_path(blob_path))
    extras = initial.get("state_extras")
    if not isinstance(extras, dict):
        raise ValueError("initial state extras changed")
    state.update(extras)
    if (
        state_fingerprint(state) != str(transitions[0]["before_fingerprint"])
        or state_fingerprint(state) != str(manifest["summary"]["initial_fingerprint"])
    ):
        raise ValueError("initial state fingerprint changed")

    first = transitions[0]
    controller = dict(first.get("controller") or {})
    action = dict(first.get("action") or {})
    metrics = dict(first.get("metrics") or {})
    expected_seed = _paired_first_action_seed(
        str(item["case_id"]), int(item["trial_index"])
    )
    agents = list(map(int, action.get("agents") or ()))
    neighborhood = list(map(int, metrics.get("neighborhood") or ()))
    order = list(map(int, metrics.get("repair_order") or ()))
    candidate_matches = str(controller.get("selected_candidate_id")) == str(
        item["candidate_id"]
    ) and str(controller.get("forced_candidate_role")) == str(item["arm"])
    pp_evidence_matches = bool(
        agents
        and agents == neighborhood
        and sorted(order) == sorted(neighborhood)
        and len(order) == len(set(order))
        and int(action.get("pp_random_seed", -1)) == expected_seed
        and int(metrics.get("requested_pp_random_seed", -1)) == expected_seed
        and int(metrics.get("applied_pp_random_seed", -1)) == expected_seed
    )
    if controller.get("forced_first_action") is not True:
        raise ValueError("first transition was not forced")
    if not candidate_matches:
        raise ValueError("forced candidate identity changed")
    if not pp_evidence_matches:
        raise ValueError("forced PP seed or order evidence changed")

    dynamics = successor_dynamics(state, transitions)
    finish = dict(events[-1])
    if (
        finish.get("event") != "finish"
        or dynamics["final_reconstructed_fingerprint"]
        != str(finish.get("final_fingerprint"))
    ):
        raise ValueError("final reconstructed fingerprint changed")
    summary = dict(manifest["summary"])
    return {
        "schema": ROW_SCHEMA,
        "case_id": str(item["case_id"]),
        "state_id": str(item["state_id"]),
        "classification": str(checkpoint["classification"]),
        "map_id": str(checkpoint["map_id"]),
        "challenger": str(item["challenger"]),
        "treatment_policy": str(item["treatment_policy"]),
        "trial_index": int(item["trial_index"]),
        "arm": str(item["arm"]),
        "candidate_id": str(item["candidate_id"]),
        "forced_candidate_matches": candidate_matches,
        "forced_pp_evidence_matches": pp_evidence_matches,
        "forced_neighborhood_size": len(agents),
        "before_fingerprint": str(first["before_fingerprint"]),
        "before_conflicts": int(metrics["conflicts_before"]),
        **dynamics,
        "bounded_success": bool(summary["success"]),
        "bounded_stop_reason": str(summary["stop_reason"]),
        "bounded_right_censored": str(summary["stop_reason"])
        in {"repair_limit", "wall_timeout"},
        "bounded_repair_iterations": int(summary["repair_iterations"]),
        "bounded_final_conflicts": int(summary["final_conflicts"]),
        "bounded_normalized_fixed_auc": float(
            summary["normalized_fixed_budget_conflict_auc"]
        ),
        "invalid_action_count": int(summary["invalid_action_count"]),
        "trace_sha256": str(manifest["trace_sha256"]),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    state_change = [
        int(row["decisions_to_first_state_change"])
        for row in rows
        if row["decisions_to_first_state_change"] is not None
    ]
    strict_drop = [
        int(row["decisions_to_first_strict_conflict_drop"])
        for row in rows
        if row["decisions_to_first_strict_conflict_drop"] is not None
    ]
    noops = [int(row["exact_successor_noop_streak"]) for row in rows]
    return {
        "episode_count": len(rows),
        "forced_repair_state_change_rate": _mean(
            float(row["forced_repair_state_changed"]) for row in rows
        ),
        "forced_strict_conflict_drop_rate": _mean(
            float(row["forced_strict_conflict_drop"]) for row in rows
        ),
        "mean_forced_conflict_delta": _mean(
            float(row["forced_conflict_delta"]) for row in rows
        ),
        "any_exact_successor_noop_rate": _mean(float(value > 0) for value in noops),
        "mean_exact_successor_noop_streak": _mean(map(float, noops)),
        "p95_exact_successor_noop_streak": _percentile(
            list(map(float, noops)), 0.95
        ),
        "maximum_exact_successor_noop_streak": max(noops, default=0),
        "state_change_right_censored_rate": _mean(
            float(row["state_change_right_censored"]) for row in rows
        ),
        "mean_observed_decisions_to_state_change": _mean(map(float, state_change)),
        "strict_drop_right_censored_rate": _mean(
            float(row["strict_drop_right_censored"]) for row in rows
        ),
        "mean_observed_decisions_to_strict_drop": _mean(map(float, strict_drop)),
        "bounded_success_rate": _mean(float(row["bounded_success"]) for row in rows),
        "bounded_right_censored_rate": _mean(
            float(row["bounded_right_censored"]) for row in rows
        ),
        "mean_bounded_normalized_fixed_auc": _mean(
            float(row["bounded_normalized_fixed_auc"]) for row in rows
        ),
        "mean_bounded_final_conflicts": _mean(
            float(row["bounded_final_conflicts"]) for row in rows
        ),
    }


def _summaries(
    rows: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        key = tuple(str(row[field]) for field in fields)
        groups[key].append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        summary = dict(zip(fields, key))
        summary.update(_group_summary(group))
        summaries.append(summary)
    return summaries


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path]]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_existing_forced_transition_inventory"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != EXPECTED_PARENT
    ):
        raise ValueError("successor-state inventory registration changed")
    expected_boundary = {
        "existing_trace_inventory_only": True,
        "new_solver_runs_allowed": False,
        "model_training_allowed": False,
        "runtime_controller_change_allowed": False,
        "future_outcome_used_as_online_feature": False,
        "longtail_prevention_claim": False,
        "ttf_improvement_claim": False,
        "generalization_claim": False,
        "no_result_based_exclusion": True,
    }
    if dict(config["claim_boundary"]) != expected_boundary:
        raise ValueError("successor-state inventory claim boundary changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    collection = (root / str(config["source_collection"]["path"])).resolve()
    manifest_count, manifest_hash = _manifest_index_sha256(collection)
    if (
        manifest_count != int(config["source_collection"]["manifest_file_count"])
        or manifest_hash
        != str(config["source_collection"]["manifest_index_sha256"])
    ):
        raise ValueError("successor-state source manifest index changed")
    return path, collection, config, inputs


def analyze_successor_state_inventory(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, collection, config, inputs = load_registration(config_path)
    parent_report = _read_json(inputs["pretail_report"])
    parent_status = _read_json(inputs["pretail_status"])
    if not (
        parent_report.get("integrity_passed") is True
        and int(parent_report.get("episode_count", -1)) == 270
        and parent_status.get("complete") is True
    ):
        raise ValueError("PreTail source completion changed")
    schedule = _read_jsonl(inputs["execution_schedule"])
    checkpoints = {
        str(row["case_id"]): row
        for row in _read_jsonl(inputs["root_checkpoints"])
        if str(row.get("checkpoint_kind")) == "first_structural_selection"
    }
    tasks = []
    for item in schedule:
        checkpoint = checkpoints.get(str(item["case_id"]))
        if checkpoint is None:
            raise ValueError("schedule case left root checkpoint cohort")
        tasks.append(
            {
                "item": item,
                "checkpoint": checkpoint,
                "collection": str(_collection_path(collection, item)),
            }
        )
    workers = int(config["analysis"]["worker_count"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_episode_row, tasks))
    rows.sort(
        key=lambda row: (
            row["case_id"],
            int(row["trial_index"]),
            row["arm"],
        )
    )
    gates = {
        "episode_count": len(rows)
        == int(config["integrity_gates"]["required_episode_count"]),
        "forced_transition_count": sum(
            bool(row["forced_candidate_matches"]) for row in rows
        )
        == int(config["integrity_gates"]["required_forced_transition_count"]),
        "reconstructed_successor_count": sum(
            bool(row["successor_fingerprint"]) for row in rows
        )
        == int(
            config["integrity_gates"]["required_reconstructed_successor_count"]
        ),
        "trace_sha_mismatch_count": True,
        "state_fingerprint_mismatch_count": True,
        "candidate_mismatch_count": all(
            bool(row["forced_candidate_matches"]) for row in rows
        ),
        "pp_seed_or_order_mismatch_count": all(
            bool(row["forced_pp_evidence_matches"]) for row in rows
        ),
        "invalid_action_count": sum(
            int(row.get("invalid_action_count", 0)) for row in rows
        )
        == int(config["integrity_gates"]["required_invalid_action_count"]),
        "execution_error_count": True,
        "all_registered_episodes_retained": len(rows) == len(schedule),
        "case_count": len({row["case_id"] for row in rows})
        == int(config["cohort"]["case_count"]),
        "arm_set": {row["arm"] for row in rows}
        == set(map(str, config["cohort"]["arms"])),
        "trial_index_set": {int(row["trial_index"]) for row in rows}
        == set(map(int, config["cohort"]["trial_indices"])),
    }
    integrity_passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": integrity_passed,
        "integrity_gates": gates,
        "new_solver_run_count": 0,
        "model_training_performed": False,
        "overall": _group_summary(rows),
        "by_arm": _summaries(rows, ("arm",)),
        "by_classification_arm": _summaries(rows, ("classification", "arm")),
        "by_map_arm": _summaries(rows, ("map_id", "arm")),
        "distinct_successor_fingerprint_count": len(
            {row["successor_fingerprint"] for row in rows}
        ),
        "distinct_successor_repair_fingerprint_count": len(
            {row["successor_repair_fingerprint"] for row in rows}
        ),
        "next_action": (
            config["decision_rule"]["all_integrity_gates_pass"]
            if integrity_passed
            else config["decision_rule"]["any_integrity_gate_fails"]
        ),
        "claim_boundary": dict(config["claim_boundary"]),
        "inputs": {
            "registration_sha256": sha256_file(path),
            "registered_file_sha256": {
                name: sha256_file(value) for name, value in sorted(inputs.items())
            },
            "manifest_index_sha256": str(
                config["source_collection"]["manifest_index_sha256"]
            ),
        },
    }
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    rows_path = output_path / "successor_state_inventory.jsonl"
    _write_jsonl(rows_path, rows)
    report["episode_rows_sha256"] = sha256_file(rows_path)
    report_path = output_path / "successor_state_inventory_report.json"
    _write_json(report_path, report)
    _write_json(
        output_path / "successor_state_inventory_status.json",
        {
            "schema": STATUS_SCHEMA,
            "complete": True,
            "integrity_passed": integrity_passed,
            "next_action": report["next_action"],
            "report_sha256": sha256_file(report_path),
            "episode_rows_sha256": report["episode_rows_sha256"],
        },
    )
    return report


__all__ = [
    "analyze_successor_state_inventory",
    "load_registration",
    "successor_dynamics",
]
