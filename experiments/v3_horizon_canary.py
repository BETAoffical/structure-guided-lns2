from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import producer_identity, sha256_file
from experiments.closed_loop_confirmation import _plain
from experiments.repair_aware import classify_repair_outcome
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _utc_now,
    _write_json,
    _write_jsonl,
)
from experiments.stall_guard import repair_structure_fingerprint
from experiments.trace_replay import replay_prefix
from experiments.v3_s3_collection import _source_replay_job


V3_HORIZON_CANARY_SCHEMA = "lns2.v3_horizon_semantic_canary.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/repair_aware.py",
    "experiments/repair_collection.py",
    "experiments/stall_guard.py",
    "experiments/trace_replay.py",
    "experiments/v3_horizon.py",
    "experiments/v3_horizon_canary.py",
    "experiments/v3_s3_collection.py",
    "scripts/audit_v3_horizon_canary.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def select_canary_state_ids(
    decisions: Iterable[dict[str, Any]],
    state_count: int,
    *,
    eligible_state_ids: Iterable[str] | None = None,
) -> list[str]:
    if int(state_count) <= 0:
        raise ValueError("v3-H3 canary state_count must be positive")
    eligible = (
        None if eligible_state_ids is None else set(map(str, eligible_state_ids))
    )
    grouped: dict[tuple[str, int], list[str]] = collections.defaultdict(list)
    for decision in decisions:
        state_id = str(decision["state_id"])
        if eligible is not None and state_id not in eligible:
            continue
        grouped[
            (str(decision["layout_mode"]), int(decision["agent_count"]))
        ].append(state_id)
    for values in grouped.values():
        values.sort()
    selected = []
    while len(selected) < int(state_count):
        advanced = False
        for key in sorted(grouped):
            if grouped[key] and len(selected) < int(state_count):
                selected.append(grouped[key].pop(0))
                advanced = True
        if not advanced:
            break
    if len(selected) != int(state_count):
        raise ValueError(
            f"v3-H3 canary requested {state_count} states but only selected {len(selected)}"
        )
    return selected


def select_canary_rollouts(
    rows: Iterable[dict[str, Any]], selected_state_ids: Iterable[str]
) -> tuple[list[dict[str, Any]], list[str]]:
    selected = set(map(str, selected_state_ids))
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        if str(row["state_id"]) in selected and int(row["trial_index"]) == 0:
            grouped[str(row["state_id"])].append(dict(row))
    result = []
    errors = []
    for state_id in sorted(selected):
        state_rows = grouped.get(state_id, [])
        adaptive = [
            row
            for row in state_rows
            if str(row.get("candidate_id")) == "official_adaptive"
        ]
        if len(adaptive) != 1:
            errors.append(f"{state_id}: expected one Adaptive trial-0 rollout")
        else:
            result.append(adaptive[0])
        for size in (4, 8, 16):
            candidates = sorted(
                (
                    row
                    for row in state_rows
                    if str(row.get("candidate_id")) != "official_adaptive"
                    and int(row.get("actual_size", -1)) == size
                ),
                key=lambda row: str(row["candidate_id"]),
            )
            if not candidates:
                errors.append(f"{state_id}: missing size-{size} trial-0 rollout")
            else:
                result.append(candidates[0])
    result.sort(
        key=lambda row: (
            str(row["state_id"]),
            str(row["candidate_id"]),
        )
    )
    return result, errors


def canary_eligible_state_ids(rows: Iterable[dict[str, Any]]) -> set[str]:
    coverage: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"adaptive": 0, "sizes": set()}
    )
    for row in rows:
        if int(row["trial_index"]) != 0:
            continue
        state_id = str(row["state_id"])
        if str(row.get("candidate_id")) == "official_adaptive":
            coverage[state_id]["adaptive"] += 1
        else:
            coverage[state_id]["sizes"].add(int(row.get("actual_size", -1)))
    return {
        state_id
        for state_id, item in coverage.items()
        if item["adaptive"] == 1 and {4, 8, 16}.issubset(item["sizes"])
    }


def replay_historical_rollout(
    decision: dict[str, Any], historical: dict[str, Any]
) -> dict[str, Any]:
    replay, _configuration = _source_replay_job(decision)
    environment, state = replay_prefix(replay, decision["prefix_actions"])
    mismatches = []
    initial_repair = repair_structure_fingerprint(state)
    if initial_repair != str(historical["initial_repair_fingerprint"]):
        mismatches.append("initial_repair_fingerprint")
    if int(state["num_of_colliding_pairs"]) != int(
        historical["conflict_trajectory"][0]
    ):
        mismatches.append("initial_conflicts")

    step_rows = []
    for position, expected in enumerate(historical["steps"], start=1):
        if bool(state.get("done")):
            mismatches.append(f"step{position}:premature_terminal")
            break
        before = state
        before_repair = repair_structure_fingerprint(before)
        if before_repair != str(expected["before_repair_fingerprint"]):
            mismatches.append(f"step{position}:before_repair_fingerprint")
        result = _plain(environment.step(dict(expected["action"])))
        state = dict(result["observation"])
        metrics = dict(result["metrics"])
        after_repair = repair_structure_fingerprint(state)
        conflicts_before = int(before["num_of_colliding_pairs"])
        conflicts_after = int(state["num_of_colliding_pairs"])
        outcome = classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics.get("replan_success")),
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=bool(state.get("feasible")),
        )
        step_mismatches = []
        if metrics.get("step_applied") is not True:
            step_mismatches.append("step_not_applied")
        if after_repair != str(expected["after_repair_fingerprint"]):
            step_mismatches.append("after_repair_fingerprint")
        if conflicts_before != int(expected["conflicts_before"]):
            step_mismatches.append("conflicts_before")
        if conflicts_after != int(expected["conflicts_after"]):
            step_mismatches.append("conflicts_after")
        if outcome != str(expected["repair_outcome"]):
            step_mismatches.append("repair_outcome")
        action = expected["action"]
        requested_random_seed = int(action.get("random_seed", -1))
        observed_random_seed = int(metrics.get("requested_random_seed", -1))
        observed_requested_pp_seed = int(
            metrics.get("requested_pp_random_seed", -1)
        )
        observed_applied_pp_seed = int(
            metrics.get("applied_pp_random_seed", -1)
        )
        if observed_random_seed != requested_random_seed:
            step_mismatches.append("requested_random_seed")
        repair_order = list(metrics.get("repair_order", ()))
        if "pp_random_seed" in action:
            seed_contract = "explicit-pp-seed"
            requested_pp_seed = int(action["pp_random_seed"])
            if observed_requested_pp_seed != requested_pp_seed:
                step_mismatches.append("requested_pp_seed")
            if repair_order and observed_applied_pp_seed != requested_pp_seed:
                step_mismatches.append("applied_pp_seed")
        else:
            seed_contract = "legacy-random-seed-coupled"
            requested_pp_seed = None
            if observed_requested_pp_seed != -1:
                step_mismatches.append("unexpected_requested_pp_seed")
            if observed_applied_pp_seed != -1:
                step_mismatches.append("unexpected_applied_pp_seed")
        mismatches.extend(f"step{position}:{name}" for name in step_mismatches)
        step_rows.append(
            {
                "step": position,
                "expected_outcome": str(expected["repair_outcome"]),
                "observed_outcome": outcome,
                "expected_conflicts_after": int(expected["conflicts_after"]),
                "observed_conflicts_after": conflicts_after,
                "repair_fingerprint_match": after_repair
                == str(expected["after_repair_fingerprint"]),
                "step_applied": metrics.get("step_applied") is True,
                "seed_contract": seed_contract,
                "requested_random_seed": requested_random_seed,
                "observed_random_seed": observed_random_seed,
                "requested_pp_random_seed": requested_pp_seed,
                "observed_requested_pp_random_seed": observed_requested_pp_seed,
                "observed_applied_pp_random_seed": observed_applied_pp_seed,
                "mismatches": step_mismatches,
            }
        )
    expected_feasible = bool(historical["h3"]["feasible"])
    if bool(state.get("feasible")) != expected_feasible:
        mismatches.append("final_feasible")
    if repair_structure_fingerprint(state) != str(
        historical["final_repair_fingerprint"]
    ):
        mismatches.append("final_repair_fingerprint")
    if len(step_rows) != int(historical["executed_steps"]):
        mismatches.append("executed_step_count")
    return {
        "schema": V3_HORIZON_CANARY_SCHEMA,
        "state_id": str(historical["state_id"]),
        "candidate_id": str(historical["candidate_id"]),
        "route": str(historical["route"]),
        "actual_size": int(historical.get("actual_size", 0)),
        "trial_index": int(historical["trial_index"]),
        "historical_executed_steps": int(historical["executed_steps"]),
        "replayed_steps": len(step_rows),
        "steps": step_rows,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def audit_v3_horizon_canary(
    *, source: str | Path, output: str | Path, state_count: int = 24
) -> dict[str, Any]:
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("v3-H3 canary output is non-empty")
    output_root.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now()
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_HORIZON_CANARY_SCHEMA,
            "status": "running",
            "started_at": started_at,
            "completed_rollouts": 0,
            "error_count": 0,
        },
    )

    source_config = _read_json(source_root / "run_config.json")
    state_source = Path(str(source_config["source"])).resolve()
    decisions = _read_jsonl(
        state_source / "collection" / "state_selection.jsonl"
    )
    decision_by_id = {str(row["state_id"]): row for row in decisions}
    historical_rows = _read_jsonl(source_root / "horizon_manifest.jsonl")
    eligible_ids = canary_eligible_state_ids(historical_rows)
    selected_ids = select_canary_state_ids(
        decisions,
        int(state_count),
        eligible_state_ids=eligible_ids,
    )
    selected_rollouts, coverage_errors = select_canary_rollouts(
        historical_rows, selected_ids
    )
    identity = producer_identity(
        project_root=PROJECT_ROOT,
        source_files=PRODUCER_FILES,
        native_required=True,
        optional_package_names=("numpy", "scikit-learn"),
    )
    results = []
    errors = list(coverage_errors)
    started = time.perf_counter()
    for historical in selected_rollouts:
        state_id = str(historical["state_id"])
        try:
            results.append(
                replay_historical_rollout(decision_by_id[state_id], historical)
            )
        except Exception as error:
            errors.append(f"{state_id}/{historical['candidate_id']}: {type(error).__name__}: {error}")
        _write_json(
            output_root / "status.json",
            {
                "schema": V3_HORIZON_CANARY_SCHEMA,
                "status": "running",
                "started_at": started_at,
                "completed_rollouts": len(results),
                "total_rollouts": len(selected_rollouts),
                "error_count": len(errors),
            },
        )
    _write_jsonl(output_root / "canary_rollouts.jsonl", results)
    semantic_mismatch_count = sum(int(row["mismatch_count"]) for row in results)
    failed_rollout_count = sum(not bool(row["passed"]) for row in results)
    report = {
        "schema": V3_HORIZON_CANARY_SCHEMA,
        "evidence_level": "fixed-historical-action-replay",
        "source": str(source_root),
        "source_schema": str(source_config.get("schema")),
        "source_manifest_sha256": sha256_file(
            source_root / "horizon_manifest.jsonl"
        ),
        "producer_identity": identity,
        "requested_state_count": int(state_count),
        "eligible_state_count": len(eligible_ids),
        "selected_state_count": len(selected_ids),
        "selected_state_ids": selected_ids,
        "expected_rollout_count": int(state_count) * 4,
        "replayed_rollout_count": len(results),
        "replayed_step_count": sum(len(row["steps"]) for row in results),
        "failed_rollout_count": failed_rollout_count,
        "semantic_mismatch_count": semantic_mismatch_count,
        "error_count": len(errors),
        "errors": errors,
        "elapsed_seconds": time.perf_counter() - started,
        "checks": {
            "coverage_complete": len(selected_rollouts) == int(state_count) * 4,
            "all_rollouts_replayed": len(results) == len(selected_rollouts),
            "no_execution_errors": not errors,
            "repair_semantics_match": semantic_mismatch_count == 0,
        },
        "limitations": [
            "The canary replays saved actions and PP seeds; it does not regenerate or rerank candidate pools.",
            "Full-state fingerprints are timing-sensitive and are not a semantic gate; repair-state fingerprints are the gate.",
            "A passing canary preserves historical interpretation but does not promote a controller or replace fresh current-schema training data.",
        ],
    }
    report["passed"] = all(map(bool, report["checks"].values()))
    _write_json(output_root / "v3_horizon_canary_report.json", report)
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_HORIZON_CANARY_SCHEMA,
            "status": "complete" if report["passed"] else "failed",
            "started_at": started_at,
            "completed_rollouts": len(results),
            "total_rollouts": len(selected_rollouts),
            "error_count": len(errors),
            "report": str(output_root / "v3_horizon_canary_report.json"),
        },
    )
    return report


__all__ = [
    "V3_HORIZON_CANARY_SCHEMA",
    "audit_v3_horizon_canary",
    "canary_eligible_state_ids",
    "replay_historical_rollout",
    "select_canary_rollouts",
    "select_canary_state_ids",
]
