from __future__ import annotations

import collections
import concurrent.futures
import hashlib
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import contained_file, registered_input, sha256_file
from experiments.closed_loop_trace_storage import (
    apply_extras_delta,
    apply_state_delta,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
)
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_tailswitch_sequence_forensics import _selected_candidate


CONFIG_SCHEMA = "lns2.stride.plateau_escape_witness_registration.v1"
REPORT_SCHEMA = "lns2.stride.plateau_escape_witness_report.v1"
WITNESS_SCHEMA = "lns2.stride.plateau_escape_witness.v1"
STATUS_SCHEMA = "lns2.stride.plateau_escape_witness_status.v1"
EXPERIMENT_ID = "stride-plateau-escape-witness-v1"
EXPECTED_PARENT = "98fb4601599bffd6c6679226edecdeebe3953d18"
CLASSIFICATIONS = ("adverse", "beneficial", "neutral")
ROOT_CAUSES = (
    "set_defect",
    "order_defect",
    "set_and_order_joint",
    "residual_pp_instability",
)


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _median(values: Iterable[float]) -> float:
    rows = list(values)
    return float(statistics.median(rows)) if rows else 0.0


def _average_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    left = 0
    while left < len(ordered):
        right = left + 1
        while right < len(ordered) and values[ordered[right]] == values[ordered[left]]:
            right += 1
        rank = (left + 1 + right) / 2.0
        for position in range(left, right):
            ranks[ordered[position]] = rank
        left = right
    return ranks


def _spearman(values: list[float], outcomes: list[float]) -> float:
    if len(values) != len(outcomes) or len(values) < 2:
        return 0.0
    left = _average_ranks(values)
    right = _average_ranks(outcomes)
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


def _edge_set(value: Iterable[Iterable[int]]) -> set[tuple[int, int]]:
    return {tuple(sorted(map(int, edge))) for edge in value}


def _path_map(state: dict[str, Any]) -> dict[int, tuple[int, ...]]:
    return {
        int(agent["id"]): tuple(map(int, agent.get("path") or ()))
        for agent in state.get("agents") or ()
    }


def canonical_artifact_manifest(directory: str | Path) -> tuple[str, list[dict[str, Any]]]:
    root = Path(directory).resolve()
    rows = []
    payload = bytearray()
    for path in sorted(root.glob("*.json"), key=lambda value: value.name):
        digest = sha256_file(path)
        rows.append({"filename": path.name, "sha256": digest})
        payload.extend(path.name.encode("utf-8"))
        payload.extend(b"\0")
        payload.extend(digest.encode("ascii"))
        payload.extend(b"\n")
    return hashlib.sha256(payload).hexdigest(), rows


def identify_trigger_plateau(
    transitions: list[dict[str, Any]],
    trigger_decision: int,
) -> tuple[int, int]:
    positions = {
        int(row["decision_index"]): index for index, row in enumerate(transitions)
    }
    if trigger_decision not in positions:
        raise ValueError(f"trigger decision is absent: {trigger_decision}")
    trigger = positions[trigger_decision]
    level = int(transitions[trigger]["conflicts_before"])
    if trigger < 2:
        raise ValueError(f"registered trigger lacks two prior actions: {trigger_decision}")
    for index in (trigger - 2, trigger - 1):
        row = transitions[index]
        if (
            int(row["conflicts_before"]) != level
            or int(row["conflicts_after"]) != level
        ):
            raise ValueError(
                f"registered pre-trigger noop changed: {trigger_decision}/{index}"
            )
    left = trigger - 1
    while left > 0:
        previous = transitions[left - 1]
        if (
            int(previous["conflicts_before"]) != level
            or int(previous["conflicts_after"]) != level
        ):
            break
        left -= 1
    right = trigger - 1
    while right + 1 < len(transitions):
        following = transitions[right + 1]
        if (
            int(following["conflicts_before"]) != level
            or int(following["conflicts_after"]) != level
        ):
            break
        right += 1
    return left, right


def _dominant_candidate(rows: list[dict[str, Any]]) -> tuple[str, list[int]]:
    counts = collections.Counter(str(row["candidate_id"]) for row in rows)
    first = {}
    agents_by_candidate: dict[str, tuple[int, ...]] = {}
    for index, row in enumerate(rows):
        candidate_id = str(row["candidate_id"])
        first.setdefault(candidate_id, index)
        agents = tuple(map(int, row["selected_agents"]))
        previous = agents_by_candidate.setdefault(candidate_id, agents)
        if previous != agents:
            raise ValueError(f"candidate agent set drifted: {candidate_id}")
    candidate_id = sorted(
        counts,
        key=lambda value: (-counts[value], first[value], value),
    )[0]
    return candidate_id, list(agents_by_candidate[candidate_id])


def build_plateau_witness(
    transitions: list[dict[str, Any]],
    *,
    case: dict[str, Any],
    causal: dict[str, Any],
    finish: dict[str, Any],
    minimum_pre_trigger_flat_length: int,
) -> dict[str, Any]:
    trigger_decision = int(case["first_repeat_stall_decision"])
    left, right = identify_trigger_plateau(transitions, trigger_decision)
    plateau = transitions[left : right + 1]
    if len(plateau) < minimum_pre_trigger_flat_length:
        raise ValueError(f"registered pre-trigger plateau shortened: {case['case_id']}")
    trigger_absolute_position = next(
        index
        for index, row in enumerate(transitions)
        if int(row["decision_index"]) == trigger_decision
    )
    triplet = transitions[
        trigger_absolute_position - 2 : trigger_absolute_position + 1
    ]
    if len(triplet) != 3 or len({row["candidate_id"] for row in triplet}) != 1:
        raise ValueError(f"registered exact triplet changed: {case['case_id']}")
    trigger_position = trigger_absolute_position - left

    dominant_id, dominant_agent_rows = _dominant_candidate(plateau)
    dominant_agents = set(dominant_agent_rows)
    snapshots = [set(row["before_edges"]) for row in plateau]
    snapshots.append(set(plateau[-1]["after_edges"]))
    persistent_edges = set.intersection(*snapshots) if snapshots else set()
    persistent_agents = {agent for edge in persistent_edges for agent in edge}
    exact_noops = [bool(row["exact_repair_noop"]) for row in plateau]
    successful_replans = [bool(row["replan_success"]) for row in plateau]
    blocker_union = set(map(int, causal["external_blocker_union"]))
    entry_transition = transitions[left - 1] if left > 0 else None

    exit_row = transitions[right + 1] if right + 1 < len(transitions) else None
    exit_payload: dict[str, Any]
    if exit_row is None:
        exit_payload = {
            "outcome": "right_censored",
            "decision_index": None,
            "candidate_id": None,
            "agent_set": [],
            "strict_reduction": False,
            "same_candidate": False,
            "same_agent_set": False,
            "set_change": False,
            "order_changed_from_last_attempt": False,
            "added_agents": [],
            "removed_agents": [],
            "added_external_blockers": [],
            "persistent_core_full_coverage": False,
            "conflict_delta": 0,
        }
    else:
        if int(exit_row["conflicts_before"]) != int(plateau[-1]["conflicts_after"]):
            raise ValueError(f"transition chain changed: {case['case_id']}")
        strict = int(exit_row["conflicts_after"]) < int(exit_row["conflicts_before"])
        outcome = "strict_reduction" if strict else "conflict_increase"
        exit_agents = set(map(int, exit_row["selected_agents"]))
        added = exit_agents - dominant_agents
        removed = dominant_agents - exit_agents
        exit_payload = {
            "outcome": outcome,
            "decision_index": int(exit_row["decision_index"]),
            "candidate_id": str(exit_row["candidate_id"]),
            "agent_set": sorted(exit_agents),
            "strict_reduction": strict,
            "same_candidate": str(exit_row["candidate_id"]) == dominant_id,
            "same_agent_set": exit_agents == dominant_agents,
            "set_change": exit_agents != dominant_agents,
            "order_changed_from_last_attempt": tuple(exit_row["repair_order"])
            != tuple(plateau[-1]["repair_order"]),
            "added_agents": sorted(added),
            "removed_agents": sorted(removed),
            "added_external_blockers": sorted(added & blocker_union),
            "persistent_core_full_coverage": persistent_agents <= exit_agents,
            "conflict_delta": int(exit_row["conflicts_before"])
            - int(exit_row["conflicts_after"]),
        }

    first_structural = next(
        row
        for row in transitions
        if int(row["decision_index"]) == int(case["first_structural_decision"])
    )
    if not bool(first_structural["selected_structural"]):
        raise ValueError(f"first structural action changed: {case['case_id']}")
    level = int(plateau[0]["conflicts_before"])
    dominant_count = sum(row["candidate_id"] == dominant_id for row in plateau)
    finish_summary = dict(finish.get("summary") or {})
    return {
        "schema": WITNESS_SCHEMA,
        "case_id": str(case["case_id"]),
        "state_id": str(case["state_id"]),
        "map_id": str(case["map_id"]),
        "task_id": str(case["task_id"]),
        "solver_seed": int(case["solver_seed"]),
        "challenger": str(case["challenger"]),
        "treatment_policy": str(case["treatment_policy"]),
        "classification": str(case["classification"]),
        "root_cause": str(causal["root_cause"]),
        "causal_state_fingerprint": str(causal["state_fingerprint"]),
        "first_structural": {
            "decision_index": int(first_structural["decision_index"]),
            "conflicts_before": int(first_structural["conflicts_before"]),
            "conflicts_after": int(first_structural["conflicts_after"]),
            "strict_reduction": int(first_structural["conflicts_after"])
            < int(first_structural["conflicts_before"]),
        },
        "plateau": {
            "conflict_level": level,
            "start_decision": int(plateau[0]["decision_index"]),
            "trigger_decision": trigger_decision,
            "end_decision": int(plateau[-1]["decision_index"]),
            "action_count": len(plateau),
            "actions_before_registered_trigger": trigger_position,
            "actions_after_registered_trigger": max(
                0, right - trigger_absolute_position
            ),
            "entry_after_strict_progress": bool(
                entry_transition is not None
                and int(entry_transition["conflicts_before"]) > level
                and int(entry_transition["conflicts_after"]) == level
            ),
            "conflict_reduction_before_plateau": int(
                first_structural["conflicts_before"]
            )
            - level,
            "dominant_candidate_id": dominant_id,
            "dominant_candidate_agents": sorted(dominant_agents),
            "dominant_candidate_count": dominant_count,
            "dominant_candidate_share": dominant_count / len(plateau),
            "unique_candidate_count": len({row["candidate_id"] for row in plateau}),
            "unique_agent_set_count": len(
                {tuple(row["selected_agents"]) for row in plateau}
            ),
            "unique_repair_order_count": len(
                {tuple(row["repair_order"]) for row in plateau}
            ),
            "unique_pp_seed_count": len(
                {int(row["pp_random_seed"]) for row in plateau}
            ),
            "exact_repair_noop_count": sum(exact_noops),
            "exact_repair_noop_fraction": _mean(map(float, exact_noops)),
            "successful_replan_count": sum(successful_replans),
            "path_change_action_count": sum(
                int(row["changed_agent_count"] > 0) for row in plateau
            ),
            "conflict_edge_change_action_count": sum(
                int(set(row["before_edges"]) != set(row["after_edges"]))
                for row in plateau
            ),
            "persistent_core_edges": [list(edge) for edge in sorted(persistent_edges)],
            "persistent_core_edge_count": len(persistent_edges),
            "persistent_core_agent_count": len(persistent_agents),
            "dominant_missing_persistent_agents": sorted(
                persistent_agents - dominant_agents
            ),
        },
        "causal_context": {
            "external_blocker_union": sorted(blocker_union),
            "external_blocker_union_count": len(blocker_union),
            "failed_agent_union": sorted(map(int, causal["failed_agent_union"])),
            "baseline_replan_success_rate": float(
                causal["baseline_replan_success_rate"]
            ),
            "baseline_mean_external_blocker_count": float(
                causal["baseline_mean_external_blocker_count"]
            ),
            "outcome_enriched_online_use_allowed": False,
        },
        "exit": exit_payload,
        "episode": {
            "success": bool(finish.get("success")),
            "stop_reason": str(finish_summary.get("stop_reason") or "unknown"),
            "repair_iterations": int(finish_summary.get("repair_iterations", 0)),
            "final_conflicts": int(finish_summary.get("final_conflicts", -1)),
        },
        "trace_sha256": str(case["trace_sha256"]),
    }


def _read_case_trace(task: dict[str, Any]) -> dict[str, Any]:
    collection = Path(task["collection"])
    manifest_path = collection / "realized_dynamic_manifest.jsonl"
    if sha256_file(manifest_path) != str(task["expected_manifest_sha256"]):
        raise ValueError(f"manifest identity changed: {task['identity_key']}")
    manifest_rows = _read_jsonl(manifest_path)
    if len(manifest_rows) != 1 or manifest_rows[0].get("status") != "ok":
        raise ValueError(f"manifest coverage changed: {task['identity_key']}")
    manifest = dict(manifest_rows[0])
    trace_path = contained_file(
        collection, manifest["trace_file"], field="plateau witness trace"
    )
    if sha256_file(trace_path) != str(task["expected_trace_sha256"]):
        raise ValueError(f"trace identity changed: {task['identity_key']}")
    events = read_trace_events(trace_path)
    if len(events) < 2 or events[0].get("event") != "initial":
        raise ValueError(f"trace framing changed: {task['identity_key']}")
    initial = events[0]
    state = read_state_blob(
        resolve_state_blob(trace_path, str(initial["state_blob"]), collection)
    )
    state.update(dict(initial.get("state_extras") or {}))
    transitions: list[dict[str, Any]] = []
    for event in events[1:-1]:
        if event.get("event") != "transition":
            raise ValueError(f"trace event changed: {task['identity_key']}")
        if state_fingerprint(state) != str(event.get("before_fingerprint")):
            raise ValueError(f"before fingerprint changed: {task['identity_key']}")
        action = dict(event.get("action") or {})
        controller = dict(event.get("controller") or {})
        candidate, families = _selected_candidate(controller, action)
        before_paths = _path_map(state)
        before_edges = _edge_set(state.get("conflict_edges") or ())
        after = apply_state_delta(state, dict(event["state_delta"]))
        after.update(apply_extras_delta(state, dict(event["state_extras_delta"])))
        if state_fingerprint(after) != str(event.get("after_fingerprint")):
            raise ValueError(f"after fingerprint changed: {task['identity_key']}")
        after_paths = _path_map(after)
        changed_agents = sorted(
            identifier
            for identifier in set(before_paths) | set(after_paths)
            if before_paths.get(identifier) != after_paths.get(identifier)
        )
        after_edges = _edge_set(after.get("conflict_edges") or ())
        metrics = dict(event.get("metrics") or {})
        before_conflicts = int(metrics.get("conflicts_before", -1))
        after_conflicts = int(metrics.get("conflicts_after", -1))
        if before_conflicts != len(before_edges) or after_conflicts != len(after_edges):
            raise ValueError(f"conflict count changed: {task['identity_key']}")
        transitions.append(
            {
                "decision_index": int(event["decision_index"]),
                "candidate_id": str(candidate["candidate_id"]),
                "selected_agents": sorted(map(int, action.get("agents") or ())),
                "selected_structural": any(
                    str(family).startswith("structpool-") for family in families
                ),
                "selection_families": list(map(str, families)),
                "repair_order": list(map(int, metrics.get("repair_order") or ())),
                "pp_random_seed": int(metrics.get("applied_pp_random_seed", -1)),
                "replan_success": bool(metrics.get("replan_success")),
                "conflicts_before": before_conflicts,
                "conflicts_after": after_conflicts,
                "before_edges": before_edges,
                "after_edges": after_edges,
                "changed_agents": changed_agents,
                "changed_agent_count": len(changed_agents),
                "exact_repair_noop": not changed_agents
                and before_edges == after_edges,
            }
        )
        state = after
    finish = dict(events[-1])
    if finish.get("event") != "finish" or state_fingerprint(state) != str(
        finish.get("final_fingerprint")
    ):
        raise ValueError(f"finish fingerprint changed: {task['identity_key']}")
    return build_plateau_witness(
        transitions,
        case=dict(task["case"]),
        causal=dict(task["causal"]),
        finish=finish,
        minimum_pre_trigger_flat_length=int(
            task["minimum_pre_trigger_flat_length"]
        ),
    )


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    strict = [row for row in rows if row["exit"]["outcome"] == "strict_reduction"]
    return {
        "case_count": len(rows),
        "mean_plateau_action_count": _mean(
            float(row["plateau"]["action_count"]) for row in rows
        ),
        "median_plateau_action_count": _median(
            float(row["plateau"]["action_count"]) for row in rows
        ),
        "maximum_plateau_action_count": max(
            (int(row["plateau"]["action_count"]) for row in rows), default=0
        ),
        "mean_actions_after_registered_trigger": _mean(
            float(row["plateau"]["actions_after_registered_trigger"]) for row in rows
        ),
        "strict_reduction_exit_count": len(strict),
        "conflict_increase_exit_count": sum(
            row["exit"]["outcome"] == "conflict_increase" for row in rows
        ),
        "right_censored_count": sum(
            row["exit"]["outcome"] == "right_censored" for row in rows
        ),
        "same_candidate_strict_exit_count": sum(
            row["exit"]["same_candidate"] for row in strict
        ),
        "same_set_strict_exit_count": sum(row["exit"]["same_agent_set"] for row in strict),
        "set_change_strict_exit_count": sum(row["exit"]["set_change"] for row in strict),
        "order_changed_same_set_strict_exit_count": sum(
            row["exit"]["same_agent_set"]
            and row["exit"]["order_changed_from_last_attempt"]
            for row in strict
        ),
        "strict_exit_with_added_external_blocker_count": sum(
            bool(row["exit"]["added_external_blockers"]) for row in strict
        ),
        "first_structural_strict_reduction_count": sum(
            row["first_structural"]["strict_reduction"] for row in rows
        ),
        "entry_after_strict_progress_count": sum(
            row["plateau"]["entry_after_strict_progress"] for row in rows
        ),
        "exact_repair_noop_action_fraction": (
            sum(int(row["plateau"]["exact_repair_noop_count"]) for row in rows)
            / sum(int(row["plateau"]["action_count"]) for row in rows)
            if rows
            else 0.0
        ),
        "mean_external_blocker_union_count": _mean(
            float(row["causal_context"]["external_blocker_union_count"])
            for row in rows
        ),
        "mean_baseline_external_blockers_per_trial": _mean(
            float(row["causal_context"]["baseline_mean_external_blocker_count"])
            for row in rows
        ),
        "unique_repair_order_fraction": (
            sum(int(row["plateau"]["unique_repair_order_count"]) for row in rows)
            / sum(int(row["plateau"]["action_count"]) for row in rows)
            if rows
            else 0.0
        ),
        "unique_pp_seed_fraction": (
            sum(int(row["plateau"]["unique_pp_seed_count"]) for row in rows)
            / sum(int(row["plateau"]["action_count"]) for row in rows)
            if rows
            else 0.0
        ),
        "mean_dominant_missing_persistent_agent_count": _mean(
            float(len(row["plateau"]["dominant_missing_persistent_agents"]))
            for row in rows
        ),
    }


def _summaries(
    witnesses: list[dict[str, Any]], fields: tuple[str, ...]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in witnesses:
        grouped[tuple(str(row[field]) for field in fields)].append(row)
    return [
        {**dict(zip(fields, key)), **_group_summary(rows)}
        for key, rows in sorted(grouped.items())
    ]


def _repairability_hazard_diagnostic(
    witnesses: list[dict[str, Any]],
) -> dict[str, Any]:
    groups = {
        "all": witnesses,
        "adverse": [
            row for row in witnesses if row["classification"] == "adverse"
        ],
        "uncensored": [
            row
            for row in witnesses
            if row["exit"]["outcome"] != "right_censored"
        ],
        "adverse_uncensored": [
            row
            for row in witnesses
            if row["classification"] == "adverse"
            and row["exit"]["outcome"] != "right_censored"
        ],
    }
    metrics = {
        "baseline_replan_success_rate": lambda row: float(
            row["causal_context"]["baseline_replan_success_rate"]
        ),
        "baseline_external_blockers_per_trial": lambda row: float(
            row["causal_context"]["baseline_mean_external_blocker_count"]
        ),
        "external_blocker_union_count": lambda row: float(
            row["causal_context"]["external_blocker_union_count"]
        ),
        "missing_persistent_agent_count": lambda row: float(
            len(row["plateau"]["dominant_missing_persistent_agents"])
        ),
        "dominant_candidate_share": lambda row: float(
            row["plateau"]["dominant_candidate_share"]
        ),
    }
    correlation = {}
    for name, rows in groups.items():
        lengths = [float(row["plateau"]["action_count"]) for row in rows]
        correlation[name] = {
            "case_count": len(rows),
            "spearman_plateau_length": {
                metric: _spearman([function(row) for row in rows], lengths)
                for metric, function in metrics.items()
            },
        }

    bucket_definitions = (
        ("zero", lambda value: value == 0.0),
        ("greater_than_zero_below_0_25", lambda value: 0.0 < value < 0.25),
        ("at_least_0_25_below_0_5", lambda value: 0.25 <= value < 0.5),
        ("at_least_0_5", lambda value: value >= 0.5),
    )
    buckets = []
    for name, predicate in bucket_definitions:
        rows = [
            row
            for row in witnesses
            if predicate(
                float(row["causal_context"]["baseline_replan_success_rate"])
            )
        ]
        lengths = [int(row["plateau"]["action_count"]) for row in rows]
        buckets.append(
            {
                "baseline_replan_success_rate_bucket": name,
                "case_count": len(rows),
                "mean_plateau_action_count": _mean(lengths),
                "median_plateau_action_count": _median(lengths),
                "maximum_plateau_action_count": max(lengths, default=0),
                "plateau_at_least_8_count": sum(value >= 8 for value in lengths),
                "plateau_at_least_16_count": sum(value >= 16 for value in lengths),
                "right_censored_count": sum(
                    row["exit"]["outcome"] == "right_censored" for row in rows
                ),
            }
        )
    return {
        "status": "posthoc_descriptive_only",
        "correlation": correlation,
        "by_baseline_replan_success_rate": buckets,
        "interpretation_boundary": (
            "The 16-seed causal-audit rate is an offline outcome label, not an "
            "online feature or a runtime probe. Correlation is mechanism evidence, "
            "not a prevention or TTF claim."
        ),
    }


def _trigger_pool_escape_opportunity(
    witnesses: list[dict[str, Any]],
    checkpoints: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    cases = []
    for witness in witnesses:
        if not (
            witness["exit"]["outcome"] == "strict_reduction"
            and witness["exit"]["set_change"]
        ):
            continue
        checkpoint = checkpoints[(str(witness["case_id"]), "first_repeat_stall")]
        exit_agents = set(map(int, witness["exit"]["agent_set"]))
        pool = list(checkpoint.get("candidate_pool") or ())
        exact_set = [
            row for row in pool if set(map(int, row.get("agents") or ())) == exit_agents
        ]
        exact_id = [
            row
            for row in pool
            if str(row.get("candidate_id")) == str(witness["exit"]["candidate_id"])
        ]
        best_jaccard = 0.0
        for candidate in pool:
            agents = set(map(int, candidate.get("agents") or ()))
            union = agents | exit_agents
            best_jaccard = max(
                best_jaccard,
                len(agents & exit_agents) / len(union) if union else 1.0,
            )
        cases.append(
            {
                "case_id": str(witness["case_id"]),
                "classification": str(witness["classification"]),
                "root_cause": str(witness["root_cause"]),
                "map_id": str(witness["map_id"]),
                "plateau_action_count": int(witness["plateau"]["action_count"]),
                "trigger_candidate_pool_count": len(pool),
                "exit_agent_count": len(exit_agents),
                "exact_exit_agent_set_present": bool(exact_set),
                "exact_exit_candidate_id_present": bool(exact_id),
                "best_trigger_pool_jaccard": best_jaccard,
            }
        )
    return {
        "status": "posthoc_descriptive_only",
        "set_change_strict_exit_count": len(cases),
        "exact_exit_agent_set_present_count": sum(
            row["exact_exit_agent_set_present"] for row in cases
        ),
        "exact_exit_candidate_id_present_count": sum(
            row["exact_exit_candidate_id_present"] for row in cases
        ),
        "mean_best_trigger_pool_jaccard": _mean(
            float(row["best_trigger_pool_jaccard"]) for row in cases
        ),
        "cases": cases,
        "interpretation_boundary": (
            "Absence means unavailable in the frozen trigger-decision pool only; "
            "it does not prove that the stochastic generator can never emit the set."
        ),
    }


def _known_regression_flat_runs(
    report: dict[str, Any], minimum_length: int, controllers: list[str]
) -> list[dict[str, Any]]:
    output = []
    controller_results = dict(report["controller_results"])
    for controller in controllers:
        result = dict(controller_results[controller])
        trajectory = list(map(int, result["conflict_trajectory"]))
        decisions = {
            int(row["decision_index"]): dict(row) for row in result["decisions"]
        }
        index = 0
        while index < len(trajectory) - 1:
            level = trajectory[index]
            if trajectory[index + 1] != level:
                index += 1
                continue
            start = index
            while index < len(trajectory) - 1 and trajectory[index + 1] == level:
                index += 1
            end = index - 1
            length = end - start + 1
            if length >= minimum_length:
                rows = [decisions[position] for position in range(start, end + 1)]
                counts = collections.Counter(
                    str(row["selected_candidate_id"]) for row in rows
                )
                first = {
                    str(row["selected_candidate_id"]): position
                    for position, row in reversed(list(enumerate(rows)))
                }
                dominant = sorted(
                    counts,
                    key=lambda value: (-counts[value], first[value], value),
                )[0]
                exit_decision = decisions.get(end + 1)
                exit_after = trajectory[end + 2] if end + 2 < len(trajectory) else None
                output.append(
                    {
                        "controller": controller,
                        "conflict_level": level,
                        "start_decision": start,
                        "end_decision": end,
                        "action_count": length,
                        "dominant_candidate_id": dominant,
                        "dominant_candidate_count": counts[dominant],
                        "dominant_candidate_share": counts[dominant] / length,
                        "exit_outcome": (
                            "right_censored"
                            if exit_decision is None
                            else (
                                "strict_reduction"
                                if int(exit_after) < level
                                else "conflict_increase"
                            )
                        ),
                        "exit_candidate_id": (
                            str(exit_decision["selected_candidate_id"])
                            if exit_decision is not None
                            else None
                        ),
                        "same_candidate_exit": bool(
                            exit_decision is not None
                            and str(exit_decision["selected_candidate_id"]) == dominant
                        ),
                    }
                )
            index += 1
    return output


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# STRIDE Plateau-Escape Witness V1 Report",
        "",
        "## Scope",
        "",
        "This is a frozen, existing-trajectory mechanism audit. It runs no PP, "
        "trains no model, and makes no TTF or long-tail-prevention claim.",
        "",
        "## Registered first-repeat plateaus",
        "",
        "| Classification | Cases | Mean length | Max length | Strict exits | Set-change exits | Same-set exits | Censored | Exact-noop action fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["by_classification"]:
        lines.append(
            "| {classification} | {case_count} | {mean_plateau_action_count:.2f} | "
            "{maximum_plateau_action_count} | {strict_reduction_exit_count} | "
            "{set_change_strict_exit_count} | {same_set_strict_exit_count} | "
            "{right_censored_count} | {exact_repair_noop_action_fraction:.4f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Causal mechanism split",
            "",
            "| Classification | Root cause | Cases | Mean length | Strict exits | Set-change exits | Same-set exits | Censored |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["by_classification_root_cause"]:
        lines.append(
            "| {classification} | {root_cause} | {case_count} | "
            "{mean_plateau_action_count:.2f} | {strict_reduction_exit_count} | "
            "{set_change_strict_exit_count} | {same_set_strict_exit_count} | "
            "{right_censored_count} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Post-hoc repairability mechanism diagnostic",
            "",
            "These rows are descriptive only; they changed no cohort, gate, or claim.",
            "",
            "| Frozen 16-seed baseline replan rate | Cases | Mean plateau | Median | Max | >= 8 | >= 16 | Censored |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["posthoc_repairability_hazard_diagnostic"][
        "by_baseline_replan_success_rate"
    ]:
        lines.append(
            "| {baseline_replan_success_rate_bucket} | {case_count} | "
            "{mean_plateau_action_count:.2f} | {median_plateau_action_count:.2f} | "
            "{maximum_plateau_action_count} | {plateau_at_least_8_count} | "
            "{plateau_at_least_16_count} | {right_censored_count} |".format(**row)
        )
    hazard = report["posthoc_repairability_hazard_diagnostic"]["correlation"]
    opportunity = report["posthoc_trigger_pool_escape_opportunity"]
    lines.extend(
        [
            "",
            "- Spearman rho between baseline replan success rate and plateau length: "
            f"`{hazard['all']['spearman_plateau_length']['baseline_replan_success_rate']:.4f}` "
            "overall and "
            f"`{hazard['adverse']['spearman_plateau_length']['baseline_replan_success_rate']:.4f}` "
            "for adverse cases.",
            "- Of "
            f"`{opportunity['set_change_strict_exit_count']}` set-change exits, the exact "
            "future exit set was already present in only "
            f"`{opportunity['exact_exit_agent_set_present_count']}` frozen trigger pool.",
            "- The rate is an offline label, and the future exit set is an outcome-enriched witness; neither is an online feature.",
            "",
            "## Frozen known Maze regression",
            "",
            "| Controller | Flat runs >= 3 | Longest run | Longest level | Longest exit | Same-candidate exit |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in report["known_regression_summary"]:
        lines.append(
            "| {controller} | {flat_run_count} | {longest_action_count} | "
            "{longest_conflict_level} | {longest_exit_outcome} | "
            "{longest_same_candidate_exit} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Integrity and claim boundary",
            "",
            f"- Integrity passed: `{str(report['integrity_passed']).lower()}`",
            f"- Witness count: `{report['witness_count']}`",
            "- Outcome-enriched blocker identities are forensic evidence only and are not online features.",
            "- Candidate-pool, PP-order, and end-to-end claims remain separate.",
            "",
        ]
    )
    return "\n".join(lines)


def load_registration(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], dict[str, Path], Path]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_existing_trajectory_mechanism_forensics"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("pre_registration_parent_commit") != EXPECTED_PARENT
    ):
        raise ValueError("plateau-escape witness registration changed")
    inputs = {
        name: registered_input(root, dict(specification), label=name)
        for name, specification in dict(config["inputs"]).items()
    }
    causal_directory = (root / config["causal_state_artifacts"]["path"]).resolve()
    if not causal_directory.is_dir():
        raise ValueError("causal state artifact directory is missing")
    expected_claim_boundary = {
        "existing_trajectory_forensics_only": True,
        "new_solver_runs_allowed": False,
        "model_training_allowed": False,
        "runtime_controller_change_allowed": False,
        "ttf_improvement_claim": False,
        "longtail_prevention_claim": False,
        "generalization_claim": False,
        "outcome_enriched_blockers_are_online_features": False,
        "no_result_based_exclusion": True,
    }
    if dict(config["claim_boundary"]) != expected_claim_boundary:
        raise ValueError("plateau-escape witness claim boundary changed")
    return path, root, config, inputs, causal_directory


def analyze_plateau_escape_witnesses(
    config_path: str | Path,
    output: str | Path,
    *,
    workers: int = 16,
) -> dict[str, Any]:
    path, root, config, inputs, causal_directory = load_registration(config_path)
    tailswitch_report = _read_json(inputs["tailswitch_report"])
    tailswitch_status = _read_json(inputs["tailswitch_status"])
    sequence_report = _read_json(inputs["sequence_forensics_report"])
    sequence_status = _read_json(inputs["sequence_forensics_status"])
    root_cases = _read_jsonl(inputs["root_diagnostic_cases"])
    root_checkpoints = _read_jsonl(inputs["root_diagnostic_checkpoints"])
    root_report = _read_json(inputs["root_diagnostic_report"])
    causal_report = _read_json(inputs["causal_report"])
    causal_effects = _read_jsonl(inputs["causal_state_effects"])
    causal_status = _read_json(inputs["causal_collection_status"])
    regression_report = _read_json(inputs["known_maze_regression_report"])
    if not (
        tailswitch_report.get("integrity_passed") is True
        and tailswitch_status.get("complete") is True
        and sequence_report.get("integrity_passed") is True
        and sequence_status.get("complete") is True
        and root_report.get("integrity_passed") is True
        and causal_report.get("integrity_passed") is True
        and causal_status.get("status") == "complete"
        and regression_report.get("integrity_passed") is True
    ):
        raise ValueError("registered source integrity changed")
    expected_counts = dict(config["cohort"]["classification_counts"])
    actual_counts = collections.Counter(str(row["classification"]) for row in root_cases)
    if len(root_cases) != 45 or dict(actual_counts) != expected_counts:
        raise ValueError("registered first-repeat cohort changed")
    if len(causal_effects) != 45:
        raise ValueError("causal state-effect coverage changed")

    artifact_digest, artifact_rows = canonical_artifact_manifest(causal_directory)
    artifact_specification = dict(config["causal_state_artifacts"])
    if (
        len(artifact_rows) != int(artifact_specification["file_count"])
        or artifact_digest
        != str(artifact_specification["canonical_manifest_sha256"])
    ):
        raise ValueError("causal state artifact identity changed")

    checkpoints = {
        (str(row["case_id"]), str(row["checkpoint_kind"])): row
        for row in root_checkpoints
    }
    effects = {str(row["state_fingerprint"]): row for row in causal_effects}
    causal_by_case: dict[str, dict[str, Any]] = {}
    for case in root_cases:
        checkpoint = checkpoints[(str(case["case_id"]), "first_repeat_stall")]
        state_fingerprint_value = str(checkpoint["state_fingerprint"])
        effect = dict(effects[state_fingerprint_value])
        state_path = causal_directory / f"{state_fingerprint_value}.json"
        state_artifact = _read_json(state_path)
        if (
            state_artifact.get("complete") is not True
            or state_artifact.get("state_fingerprint") != state_fingerprint_value
            or state_artifact.get("run_fingerprint")
            != causal_status.get("run_fingerprint")
        ):
            raise ValueError(f"causal state changed: {state_fingerprint_value}")
        baseline_trials = [
            dict(row)
            for row in state_artifact.get("trials") or ()
            if row.get("arm") == "selected_native_order" and row.get("applicable")
        ]
        if len(baseline_trials) != 16:
            raise ValueError(f"causal baseline trial coverage changed: {state_fingerprint_value}")
        blocker_union: set[int] = set()
        failed_agents: set[int] = set()
        for trial in baseline_trials:
            diagnostic = dict(trial.get("pp_diagnostic") or {})
            if diagnostic.get("failed_agent") is not None:
                failed_agents.add(int(diagnostic["failed_agent"]))
            for agent in diagnostic.get("agents") or ():
                blocker_union.update(
                    map(int, dict(agent).get("external_blocker_agents") or ())
                )
        causal_by_case[str(case["case_id"])] = {
            "state_fingerprint": state_fingerprint_value,
            "root_cause": str(effect["root_cause"]),
            "external_blocker_union": sorted(blocker_union),
            "failed_agent_union": sorted(failed_agents),
            "baseline_replan_success_rate": float(
                effect["baseline_replan_success_rate"]
            ),
            "baseline_mean_external_blocker_count": float(
                effect["baseline_mean_external_blocker_count"]
            ),
        }

    trace_hashes = dict(sequence_report["inputs"]["trace_sha256"])
    manifest_hashes = dict(sequence_report["inputs"]["manifest_sha256"])
    tailswitch_root = inputs["tailswitch_report"].parent
    tasks = []
    for case in sorted(root_cases, key=lambda row: str(row["case_id"])):
        state_id = str(case["state_id"])
        policy = str(case["treatment_policy"])
        identity_key = f"{state_id}/{policy}"
        if str(case["trace_sha256"]) != str(trace_hashes[identity_key]):
            raise ValueError(f"root/sequence trace identity changed: {identity_key}")
        collection = (
            tailswitch_root
            / "states"
            / _fingerprint({"state_id": state_id})[:20]
            / policy
        )
        tasks.append(
            {
                "case": case,
                "causal": causal_by_case[str(case["case_id"])],
                "collection": str(collection),
                "identity_key": identity_key,
                "expected_manifest_sha256": manifest_hashes[identity_key],
                "expected_trace_sha256": trace_hashes[identity_key],
                "minimum_pre_trigger_flat_length": int(
                    config["plateau_definition"][
                        "minimum_registered_pre_trigger_flat_length"
                    ]
                ),
            }
        )
    worker_count = min(max(1, int(workers)), 16, len(tasks))
    if worker_count == 1:
        witnesses = [_read_case_trace(task) for task in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count
        ) as executor:
            witnesses = list(executor.map(_read_case_trace, tasks))
    witnesses.sort(key=lambda row: str(row["case_id"]))

    by_classification = _summaries(witnesses, ("classification",))
    by_classification_root = _summaries(
        witnesses, ("classification", "root_cause")
    )
    by_map_classification = _summaries(witnesses, ("map_id", "classification"))
    regression_runs = _known_regression_flat_runs(
        regression_report,
        int(config["known_regression_scope"]["minimum_reported_flat_run_length"]),
        list(map(str, config["known_regression_scope"]["controllers"])),
    )
    known_regression_summary = []
    for controller in config["known_regression_scope"]["controllers"]:
        rows = [row for row in regression_runs if row["controller"] == controller]
        longest = max(rows, key=lambda row: row["action_count"], default=None)
        known_regression_summary.append(
            {
                "controller": controller,
                "flat_run_count": len(rows),
                "longest_action_count": (
                    int(longest["action_count"]) if longest else 0
                ),
                "longest_conflict_level": (
                    int(longest["conflict_level"]) if longest else 0
                ),
                "longest_exit_outcome": (
                    str(longest["exit_outcome"]) if longest else "none"
                ),
                "longest_same_candidate_exit": bool(
                    longest and longest["same_candidate_exit"]
                ),
            }
        )

    root_cause_counts = collections.Counter(row["root_cause"] for row in witnesses)
    integrity_gates = {
        "all_45_registered_cases_present": len(witnesses) == 45,
        "classification_counts_match": dict(actual_counts) == expected_counts,
        "all_causal_states_joined": len(causal_by_case) == 45,
        "causal_root_cause_counts_match": dict(root_cause_counts)
        == dict(causal_report["root_cause_counts"]),
        "every_trigger_has_two_prior_flat_actions": all(
            int(row["plateau"]["action_count"]) >= 2 for row in witnesses
        ),
        "all_trace_hashes_match": len(witnesses) == len(tasks),
        "causal_artifact_manifest_matches": artifact_digest
        == artifact_specification["canonical_manifest_sha256"],
        "no_result_based_exclusion": len(witnesses) == len(root_cases),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "integrity_passed": all(integrity_gates.values()),
        "integrity_gates": integrity_gates,
        "worker_count": worker_count,
        "new_solver_run_count": 0,
        "model_training_performed": False,
        "witness_count": len(witnesses),
        "classification_counts": dict(sorted(actual_counts.items())),
        "root_cause_counts": dict(sorted(root_cause_counts.items())),
        "overall": _group_summary(witnesses),
        "by_classification": by_classification,
        "by_classification_root_cause": by_classification_root,
        "by_map_classification": by_map_classification,
        "posthoc_repairability_hazard_diagnostic": (
            _repairability_hazard_diagnostic(witnesses)
        ),
        "posthoc_trigger_pool_escape_opportunity": (
            _trigger_pool_escape_opportunity(witnesses, checkpoints)
        ),
        "known_regression_flat_runs": regression_runs,
        "known_regression_summary": known_regression_summary,
        "interpretation_boundary": {
            "plateau_exit_is_observed_future_forensics": True,
            "external_blocker_identity_is_outcome_enriched": True,
            "online_feature_or_policy_claim": False,
            "candidate_pool_only_can_cover_all_root_causes": False,
        },
        "inputs": {
            "registration_sha256": sha256_file(path),
            "registered_file_sha256": {
                name: sha256_file(value) for name, value in sorted(inputs.items())
            },
            "causal_state_artifact_manifest_sha256": artifact_digest,
            "causal_state_artifact_sha256": artifact_rows,
        },
        "claim_boundary": dict(config["claim_boundary"]),
    }
    output_path = Path(output).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    witness_path = output_path / "plateau_escape_witnesses.jsonl"
    _write_jsonl(witness_path, witnesses)
    report["witnesses_sha256"] = sha256_file(witness_path)
    report_path = output_path / "plateau_escape_witness_report.json"
    _write_json(report_path, report)
    markdown_path = output_path / "plateau_escape_witness_report.md"
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    _write_json(
        output_path / "plateau_escape_witness_status.json",
        {
            "schema": STATUS_SCHEMA,
            "complete": True,
            "integrity_passed": report["integrity_passed"],
            "witness_count": len(witnesses),
            "worker_count": worker_count,
            "report_sha256": sha256_file(report_path),
            "witnesses_sha256": report["witnesses_sha256"],
            "markdown_sha256": sha256_file(markdown_path),
        },
    )
    return report


__all__ = [
    "analyze_plateau_escape_witnesses",
    "build_plateau_witness",
    "canonical_artifact_manifest",
    "identify_trigger_plateau",
    "load_registration",
]
