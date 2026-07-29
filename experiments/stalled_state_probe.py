from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments.closed_loop_confirmation import (
    feature_range_diagnostic,
    generate_online_candidates,
    score_online_candidates,
)
from experiments._common import (
    atomic_write_csv as _write_csv,
    producer_identity,
    strict_bool as _strict_bool,
)
from experiments.compact_controller_model import load_controller_bundle
from experiments.lns2_bottleneck import validate_manifest_trace
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _low_level_delta,
    _make_environment,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    state_fingerprint,
)
from experiments.trace_replay import decision_rows, replay_prefix
from experiments.run_output_guard import prepare_run_output
from experiments.repair_aware import classify_repair_outcome
from experiments.stall_guard import repair_structure_fingerprint


STALLED_STATE_PROBE_SCHEMA = "lns2.stalled_state_probe.v2"
STALLED_STATE_PROBE_VERSION = 2
TRIAL_STATE_RESTORE = "reset_paths-exact-repair-state-v1"
STALLED_STATE_PROBE_PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/_common.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/compact_controller_model.py",
    "experiments/feature_schema_v2.py",
    "experiments/lns2_bottleneck.py",
    "experiments/neighborhood_candidates.py",
    "experiments/online_feature_engine.py",
    "experiments/repair_collection.py",
    "experiments/run_output_guard.py",
    "experiments/stall_guard.py",
    "experiments/stalled_state_probe.py",
    "experiments/trace_replay.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _median(values: Iterable[float]) -> float | None:
    numbers = list(map(float, values))
    return statistics.median(numbers) if numbers else None


def _strict_int(
    value: Any, *, field: str, minimum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _finite_nonnegative(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _optional_native_seed(value: Any) -> int | None:
    if value is None:
        return None
    seed = _strict_int(value, field="native PP seed")
    return seed if seed >= 0 else None


def _json_int_list(value: Any, *, field: str) -> list[int]:
    try:
        raw = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError as error:
        raise ValueError(f"{field} is not valid JSON") from error
    if not isinstance(raw, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in raw
    ):
        raise ValueError(f"{field} must be a non-negative integer list")
    if len(raw) != len(set(raw)):
        raise ValueError(f"{field} must not contain duplicates")
    return list(raw)


def _candidate_pool_signature(
    candidates: Iterable[dict[str, Any]],
    scores: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in candidates]
    score_values = list(scores) if scores is not None else None
    if score_values is not None and len(score_values) != len(materialized):
        raise ValueError("candidate pool score coverage mismatch")
    signature: list[dict[str, Any]] = []
    for index, candidate in enumerate(materialized):
        agents = sorted(
            _json_int_list(candidate.get("agents", ()), field="candidate agents")
        )
        if not agents:
            raise ValueError("candidate pool contains an invalid neighborhood")
        if _strict_int(
            candidate.get("actual_size"), field="candidate actual_size", minimum=1
        ) != len(agents):
            raise ValueError("candidate pool size differs from its neighborhood")
        raw_score = (
            score_values[index]
            if score_values is not None
            else candidate.get("score")
        )
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError("candidate pool contains a non-numeric score")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("candidate pool contains a non-finite score")
        signature.append(
            {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "agents": agents,
                "actual_size": len(agents),
                "score": round(score, 12),
            }
        )
    if any(not row["candidate_id"] for row in signature):
        raise ValueError("candidate pool is missing a candidate id")
    if len({row["candidate_id"] for row in signature}) != len(signature):
        raise ValueError("candidate pool contains duplicate candidate ids")
    return sorted(signature, key=lambda row: row["candidate_id"])


def _source_v2_selection_ids(controller: dict[str, Any]) -> tuple[str, str]:
    """Return base/effective ids while accepting pre-reranker v2 traces."""

    selected = str(controller.get("selected_candidate_id") or "")
    if not selected:
        raise ValueError("source target is missing its selected candidate id")
    base_raw = controller.get("base_selected_candidate_id")
    base = selected if base_raw is None else str(base_raw)
    if not base:
        raise ValueError("source target has an empty base selected candidate id")
    return base, selected


def _repair_state_unchanged(row: dict[str, Any]) -> bool:
    reported = row.get("repair_state_changed")
    before = row.get("before_repair_fingerprint")
    after = row.get("after_repair_fingerprint")
    if before is not None and after is not None:
        changed = str(before) != str(after)
        if reported is not None and bool(reported) != changed:
            raise ValueError("repair state-change flag disagrees with fingerprints")
        return not changed
    if reported is None:
        raise ValueError("decision lacks repair-structure state-change evidence")
    return not bool(reported)


def _state_paths(state: dict[str, Any]) -> list[list[int]]:
    agents = state.get("agents")
    if not isinstance(agents, list) or not agents:
        raise ValueError("stalled-state target has no agents")
    paths: list[list[int]] = []
    for expected_id, agent in enumerate(agents):
        if (
            not isinstance(agent, dict)
            or _strict_int(agent.get("id"), field="target agent id", minimum=0)
            != expected_id
        ):
            raise ValueError("stalled-state target agents are not contiguous")
        path = agent.get("path")
        if (
            not isinstance(path, list)
            or not path
            or any(
                isinstance(location, bool)
                or not isinstance(location, int)
                or location < 0
                for location in path
            )
        ):
            raise ValueError("stalled-state target contains an invalid path")
        paths.append(list(path))
    return paths


def _restore_trial_state(
    job: dict[str, Any],
    source_state: dict[str, Any],
    *,
    seed: int,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    destroy_strategy = str(job.get("replay_destroy_strategy", "Adaptive"))
    environment = _make_environment(
        job["dataset_root"], job["row"], job["environment"], destroy_strategy
    )
    restored = _plain(environment.reset_paths(_state_paths(source_state), seed=seed))
    source_full = state_fingerprint(source_state)
    restored_full = state_fingerprint(restored)
    source_repair = repair_structure_fingerprint(source_state)
    restored_repair = repair_structure_fingerprint(restored)
    if restored_repair != source_repair:
        raise RuntimeError(
            "stalled-state reset_paths did not restore the exact repair state"
        )
    return environment, restored, {
        "trial_state_restore": TRIAL_STATE_RESTORE,
        "source_before_fingerprint": source_full,
        "restored_before_fingerprint": restored_full,
        "full_state_fingerprint_match": restored_full == source_full,
        "repair_state_fingerprint_match": True,
    }


def find_terminal_stall(decisions: list[dict[str, Any]], minimum: int = 3) -> dict[str, Any]:
    """Return the first decision in the terminal unchanged-state failed-replan run."""

    if minimum <= 0:
        raise ValueError("minimum terminal stall length must be positive")
    start = len(decisions)
    while start > 0:
        row = decisions[start - 1]
        metrics = dict(row.get("actual_metrics") or {})
        failed = not bool(metrics.get("replan_success"))
        if not failed:
            break
        if not _repair_state_unchanged(row):
            break
        start -= 1
    length = len(decisions) - start
    if length < minimum:
        raise ValueError(
            f"source trace has no terminal unchanged-state failed-replan run of at least {minimum} decisions"
        )
    target = dict(decisions[start])
    return {
        "decision": target,
        "start_decision_index": int(target["decision_index"]),
        "length": length,
        "before_fingerprint": str(target["before_fingerprint"]),
    }


def _ranked_candidate_indices(
    candidates: list[dict[str, Any]], scores: list[float]
) -> list[int]:
    if len(candidates) != len(scores) or not candidates:
        raise ValueError("candidate ranking inputs are invalid")
    return sorted(
        range(len(candidates)),
        key=lambda index: (
            -round(float(scores[index]), 12),
            str(candidates[index]["candidate_id"]),
        ),
    )


def choose_probe_branches(
    candidates: list[dict[str, Any]],
    scores: list[float],
    *,
    all_candidates: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Choose and deduplicate the fixed diagnostic alternatives."""

    order = _ranked_candidate_indices(candidates, scores)
    if all_candidates:
        requested = [
            (f"rank_{rank}", index)
            for rank, index in enumerate(order, 1)
        ]
        requested[0] = ("rank1", requested[0][1])
    else:
        requested = [
            ("rank1", order[0]),
            ("rank2", order[1] if len(order) > 1 else order[0]),
        ]
        for cap in (8, 4):
            eligible = [
                index
                for index in order
                if int(candidates[index]["actual_size"]) <= cap
            ]
            if not eligible:
                raise ValueError(
                    f"candidate pool has no neighborhood with actual_size <= {cap}"
                )
            requested.append((f"size_le_{cap}", eligible[0]))
    requested.append(("official_adaptive", None))

    unique: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    alias_to_key: dict[str, str] = {}
    for alias, index in requested:
        if index is None:
            key = "official_adaptive"
            branch = {
                "branch_key": key,
                "aliases": [],
                "mode": "official",
                "candidate": None,
                "rank": None,
                "score": None,
            }
        else:
            candidate = dict(candidates[index])
            key = (
                "neighborhood-"
                + _fingerprint(
                    {"agents": sorted(set(map(int, candidate["agents"])))}
                )[:20]
                if all_candidates
                else str(candidate["candidate_id"])
            )
            branch = {
                "branch_key": key,
                "aliases": [],
                "mode": "explicit_neighborhood",
                "candidate": candidate,
                "rank": order.index(index) + 1,
                "score": float(scores[index]),
            }
        if key not in by_key:
            by_key[key] = branch
            unique.append(branch)
        by_key[key]["aliases"].append(alias)
        alias_to_key[alias] = key
    return unique, alias_to_key


def paired_probe_seed(before_fingerprint: str, trial_index: int) -> int:
    if trial_index < 0:
        raise ValueError("trial index must be non-negative")
    return int(
        _fingerprint(
            {
                "namespace": "stalled-state-paired-repair-v1",
                "before_fingerprint": str(before_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def summarize_probe_trials(
    trials: list[dict[str, Any]], alias_to_key: dict[str, str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for row in trials:
        by_key.setdefault(str(row["branch_key"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for alias, key in alias_to_key.items():
        rows = by_key.get(key, [])
        if not rows:
            raise ValueError(f"probe branch is missing trials: {alias}")
        summary = {
            "branch": alias,
            "branch_key": key,
            "trial_count": len(rows),
            "replan_success_count": sum(bool(row["replan_success"]) for row in rows),
            "replan_success_rate": sum(bool(row["replan_success"]) for row in rows)
            / len(rows),
            "positive_conflict_reduction_count": sum(
                int(row["conflict_delta"]) > 0 for row in rows
            ),
            "mean_conflict_delta": statistics.fmean(
                float(row["conflict_delta"]) for row in rows
            ),
            "median_conflict_delta": _median(
                float(row["conflict_delta"]) for row in rows
            ),
            "median_conflicts_after": _median(
                float(row["conflicts_after"]) for row in rows
            ),
            "median_pp_replan_seconds": _median(
                float(row["pp_replan_seconds"]) for row in rows
            ),
            "median_total_decision_seconds": _median(
                float(row["total_decision_seconds"]) for row in rows
            ),
        }
        summaries.append(summary)
    # The historical probe used a separate 25%/two-positive-outcome rule that
    # could pass the same data that the stable paired Oracle classified as
    # inconclusive.  Keep this field only as an explicit non-gate so one set of
    # trials can no longer produce contradictory decisions.
    diagnostic = {
        "passed": False,
        "status": "oracle_audit_required",
        "supported_alternatives": [],
        "reason": (
            "one-step summaries are descriptive; stable paired classification "
            "is owned by experiments.stall_oracle"
        ),
    }
    return summaries, diagnostic


def _model_candidates(
    environment: Any,
    state: dict[str, Any],
    *,
    task_id: str,
    solver_seed: int,
    decision_index: int,
    configuration: dict[str, Any],
    model: Any,
    model_ranges: dict[str, tuple[float, float]],
) -> tuple[list[dict[str, Any]], list[float], dict[str, Any]]:
    started = time.perf_counter()
    before_fingerprint = state_fingerprint(state)
    optimized = str(configuration.get("controller_runtime", "reference")) == "optimized"
    candidates, proposal = generate_online_candidates(
        environment,
        state,
        task_id=task_id,
        solver_seed=solver_seed,
        decision_index=decision_index,
        proposal_config=dict(configuration["proposal"]),
        state_hash=before_fingerprint,
        verify_full_state=True,
        proposal_backend="optimized" if optimized else "reference",
        shadow_validation=False,
    )
    engine = OnlineFeatureEngine(
        state,
        backend=str(configuration.get("feature_backend", "auto")),
        shadow_validation=False,
        required_features={"realized_dynamic": model.base_feature_names},
        dense_output=optimized,
    )
    rows, feature_metrics = engine.realized_rows(candidates, state_hash=before_fingerprint)
    inference_started = time.perf_counter()
    selected, scores, margin = score_online_candidates(rows, model)
    inference_seconds = time.perf_counter() - inference_started
    diagnostics = [
        feature_range_diagnostic(row, "realized_dynamic", model_ranges)
        for row in rows
    ]
    return candidates, scores, {
        "selected_index": selected,
        "score_margin": margin,
        "selection_seconds": time.perf_counter() - started,
        "candidate_generation_seconds": float(
            proposal.get("candidate_generation_seconds", proposal.get("proposal_seconds", 0.0))
        ),
        "state_check_seconds": float(proposal.get("state_check_seconds", 0.0)),
        "state_analysis_seconds": float(
            engine.last_prepare_metrics.get("state_analysis_seconds", 0.0)
        ),
        "realized_feature_seconds": float(
            feature_metrics.get("realized_feature_seconds", 0.0)
        ),
        "inference_seconds": inference_seconds,
        "candidate_count": len(candidates),
        "selected_feature_out_of_range_fraction": float(
            diagnostics[selected]["outside_fraction"]
        ),
    }


def _checkpoint_valid(
    path: Path,
    *,
    run_fingerprint: str,
    branch: dict[str, Any],
    trial_index: int,
    before_fingerprint: str,
    before_repair_fingerprint: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        row = _read_json(path)
        if str(row.get("schema")) != STALLED_STATE_PROBE_SCHEMA or _strict_int(
            row.get("schema_version"), field="schema_version"
        ) != STALLED_STATE_PROBE_VERSION:
            raise ValueError("schema mismatch")
        if row.get("complete") is not True:
            raise ValueError("checkpoint is not complete")
        if str(row.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("run fingerprint mismatch")
        if str(row.get("branch_key")) != str(branch["branch_key"]):
            raise ValueError("branch key mismatch")
        if _strict_int(row.get("trial_index"), field="trial_index", minimum=0) != (
            trial_index
        ):
            raise ValueError("trial index mismatch")
        if (
            row.get("replay_fingerprint_match") is not True
            or row.get("repair_state_fingerprint_match") is not True
            or str(row.get("trial_state_restore")) != TRIAL_STATE_RESTORE
        ):
            raise ValueError("repair-state restore evidence is invalid")
        restored_before_fingerprint = str(
            row.get("restored_before_fingerprint") or ""
        )
        if not restored_before_fingerprint:
            raise ValueError("restored before-state fingerprint is missing")
        full_state_match = _strict_bool(
            row.get("full_state_fingerprint_match"),
            field="full_state_fingerprint_match",
        )
        if full_state_match != (
            restored_before_fingerprint == before_fingerprint
        ):
            raise ValueError("full-state restore evidence is inconsistent")
        expected_seed = paired_probe_seed(before_fingerprint, trial_index)
        if _strict_int(
            row.get("random_seed"), field="random_seed", minimum=0
        ) != expected_seed or _strict_int(
            row.get("requested_pp_random_seed"),
            field="requested_pp_random_seed",
            minimum=0,
        ) != expected_seed:
            raise ValueError("paired PP seed mismatch")
        raw_applied_seed = row.get("applied_pp_random_seed")
        applied_seed = (
            None
            if raw_applied_seed is None
            else _strict_int(
                raw_applied_seed, field="applied_pp_random_seed", minimum=0
            )
        )
        repair_order = _json_int_list(
            row.get("repair_order", "[]"), field="repair_order"
        )
        if repair_order and applied_seed != expected_seed:
            raise ValueError("applied PP seed mismatch")
        if not repair_order and applied_seed is not None:
            raise ValueError("checkpoint applied an unexpected PP seed")
        if str(row.get("before_fingerprint")) != before_fingerprint or str(
            row.get("before_repair_fingerprint")
        ) != before_repair_fingerprint:
            raise ValueError("before-state fingerprint mismatch")
        if not str(row.get("after_fingerprint", "")) or not str(
            row.get("after_repair_fingerprint", "")
        ):
            raise ValueError("after-state fingerprint is missing")
        replan_success = _strict_bool(
            row.get("replan_success"), field="replan_success"
        )
        terminated = _strict_bool(row.get("terminated"), field="terminated")
        truncated = _strict_bool(row.get("truncated"), field="truncated")
        if terminated and truncated:
            raise ValueError("checkpoint cannot be terminated and truncated")
        conflicts_before = _strict_int(
            row.get("conflicts_before"), field="conflicts_before", minimum=0
        )
        conflicts_after = _strict_int(
            row.get("conflicts_after"), field="conflicts_after", minimum=0
        )
        if terminated != (conflicts_after == 0):
            raise ValueError("checkpoint termination disagrees with the after state")
        expected_outcome = classify_repair_outcome(
            before_fingerprint=before_repair_fingerprint,
            after_fingerprint=str(row["after_repair_fingerprint"]),
            replan_success=replan_success,
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=terminated,
        )
        if str(row.get("repair_outcome")) != expected_outcome:
            raise ValueError("checkpoint repair outcome mismatch")
        if _strict_int(row.get("conflict_delta"), field="conflict_delta") != (
            conflicts_before - conflicts_after
        ):
            raise ValueError("checkpoint conflict delta mismatch")
        for name in (
            "model_selection_seconds",
            "native_neighborhood_generation_seconds",
            "pp_replan_seconds",
            "repair_wall_seconds",
            "total_decision_seconds",
        ):
            _finite_nonnegative(row.get(name), field=name)
        if not math.isclose(
            float(row["total_decision_seconds"]),
            float(row["model_selection_seconds"]) + float(row["repair_wall_seconds"]),
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("checkpoint total decision timing mismatch")
        for name in ("expanded", "generated", "reopened"):
            _strict_int(row.get(name), field=name, minimum=0)
        mode = str(branch["mode"])
        if str(row.get("branch_mode")) != mode:
            raise ValueError("checkpoint branch mode mismatch")
        expected_aliases = ",".join(map(str, branch["aliases"]))
        if str(row.get("branch_aliases")) != expected_aliases:
            raise ValueError("checkpoint branch aliases mismatch")
        _strict_int(row.get("sum_of_costs_delta"), field="sum_of_costs_delta")
        if mode == "explicit_neighborhood":
            candidate = dict(branch["candidate"])
            expected_agents = sorted(
                _json_int_list(candidate["agents"], field="branch candidate agents")
            )
            actual_agents = sorted(
                _json_int_list(row["candidate_agents"], field="candidate_agents")
            )
            if (
                str(row.get("candidate_id")) != str(candidate["candidate_id"])
                or _strict_int(
                    row.get("candidate_rank"), field="candidate_rank", minimum=1
                )
                != _strict_int(branch["rank"], field="branch rank", minimum=1)
                or actual_agents != expected_agents
                or _strict_int(
                    row.get("candidate_size"), field="candidate_size", minimum=1
                )
                != len(expected_agents)
            ):
                raise ValueError("checkpoint candidate metadata mismatch")
            candidate_score = row.get("candidate_score")
            if (
                isinstance(candidate_score, bool)
                or not isinstance(candidate_score, (int, float))
                or not math.isfinite(float(candidate_score))
                or not math.isclose(
                    float(candidate_score),
                    float(branch["score"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("checkpoint candidate score mismatch")
        elif mode == "official":
            actual_agents = sorted(
                _json_int_list(row["candidate_agents"], field="candidate_agents")
            )
            if not actual_agents or _strict_int(
                row.get("candidate_size"), field="candidate_size", minimum=1
            ) != len(actual_agents):
                raise ValueError("checkpoint official neighborhood is invalid")
            if (
                row.get("candidate_id") is not None
                or row.get("candidate_rank") is not None
                or row.get("candidate_score") is not None
            ):
                raise ValueError("checkpoint official branch claims a model candidate")
        else:
            raise ValueError("checkpoint branch mode is unsupported")
        if repair_order and sorted(map(int, repair_order)) != actual_agents:
            raise ValueError("checkpoint repair order differs from its neighborhood")
    except (KeyError, OSError, OverflowError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"completed stalled-state checkpoint is invalid: {path}") from error
    return row


def _run_trial(
    *,
    job: dict[str, Any],
    decision: dict[str, Any],
    source_state: dict[str, Any],
    branch: dict[str, Any],
    trial_index: int,
    selection_seconds: float,
    run_fingerprint: str,
) -> dict[str, Any]:
    source_before_fingerprint = str(decision["before_fingerprint"])
    random_seed = paired_probe_seed(source_before_fingerprint, trial_index)
    environment, before, restore_evidence = _restore_trial_state(
        job, source_state, seed=random_seed
    )
    if (
        str(restore_evidence["source_before_fingerprint"])
        != source_before_fingerprint
    ):
        raise RuntimeError("stalled-state source fingerprint changed before trial")
    if str(branch["mode"]) == "official":
        action = {
            "mode": "official",
            "random_seed": random_seed,
            "pp_random_seed": random_seed,
        }
        model_selection_seconds = 0.0
    else:
        candidate = dict(branch["candidate"])
        action = {
            "mode": "explicit_neighborhood",
            "agents": list(map(int, candidate["agents"])),
            "random_seed": random_seed,
            "pp_random_seed": random_seed,
        }
        model_selection_seconds = selection_seconds
    started = time.perf_counter()
    result = _plain(environment.step(action))
    repair_wall_seconds = time.perf_counter() - started
    after = dict(result["observation"])
    metrics = dict(result["metrics"])
    replan_success = _strict_bool(
        metrics.get("replan_success"), field="native replan_success"
    )
    terminated = _strict_bool(result.get("terminated"), field="native terminated")
    truncated = _strict_bool(result.get("truncated"), field="native truncated")
    if terminated and truncated:
        raise RuntimeError("stalled-state trial cannot terminate and truncate")
    requested_pp_seed = _strict_int(
        metrics.get("requested_pp_random_seed"),
        field="native requested PP seed",
        minimum=0,
    )
    if requested_pp_seed != random_seed:
        raise RuntimeError("stalled-state trial did not retain its paired PP seed")
    repair_order = _json_int_list(
        metrics.get("repair_order", []), field="native repair order"
    )
    applied_pp_seed = _optional_native_seed(metrics.get("applied_pp_random_seed"))
    if repair_order and applied_pp_seed != random_seed:
        raise RuntimeError("stalled-state trial did not apply its paired PP seed")
    if not repair_order and applied_pp_seed is not None:
        raise RuntimeError("stalled-state trial applied an unexpected PP seed")
    conflicts_before = _strict_int(
        before["num_of_colliding_pairs"], field="before conflicts", minimum=0
    )
    conflicts_after = _strict_int(
        after["num_of_colliding_pairs"], field="after conflicts", minimum=0
    )
    if terminated != (conflicts_after == 0):
        raise RuntimeError("stalled-state termination disagrees with the after state")
    before_repair_fingerprint = repair_structure_fingerprint(before)
    after_repair_fingerprint = repair_structure_fingerprint(after)
    repair_outcome = classify_repair_outcome(
        before_fingerprint=before_repair_fingerprint,
        after_fingerprint=after_repair_fingerprint,
        replan_success=replan_success,
        conflicts_before=conflicts_before,
        conflicts_after=conflicts_after,
        feasible=terminated,
    )
    native_selection_seconds = _finite_nonnegative(
        metrics.get("native_neighborhood_generation_seconds", 0.0),
        field="native neighborhood generation seconds",
    )
    low_level_delta = _low_level_delta(before, after)
    native_neighborhood = _json_int_list(
        metrics.get("neighborhood", []), field="native neighborhood"
    )
    if branch["candidate"] is not None:
        expected_neighborhood = sorted(
            _json_int_list(
                dict(branch["candidate"])["agents"], field="candidate agents"
            )
        )
        if sorted(native_neighborhood) != expected_neighborhood:
            raise RuntimeError(
                "stalled-state native neighborhood differs from the requested branch"
            )
    elif not native_neighborhood:
        raise RuntimeError("stalled-state official branch selected no neighborhood")
    return {
        "schema": STALLED_STATE_PROBE_SCHEMA,
        "schema_version": STALLED_STATE_PROBE_VERSION,
        "complete": True,
        "run_fingerprint": run_fingerprint,
        "branch_key": str(branch["branch_key"]),
        "branch_aliases": ",".join(map(str, branch["aliases"])),
        "branch_mode": str(branch["mode"]),
        "candidate_id": (
            str(dict(branch["candidate"])["candidate_id"])
            if branch["candidate"] is not None
            else None
        ),
        "candidate_rank": branch["rank"],
        "candidate_score": branch["score"],
        "candidate_size": (
            _strict_int(
                dict(branch["candidate"])["actual_size"],
                field="candidate actual_size",
                minimum=1,
            )
            if branch["candidate"] is not None
            else len(native_neighborhood)
        ),
        "candidate_agents": (
            json.dumps(
                sorted(map(int, dict(branch["candidate"])["agents"])),
                separators=(",", ":"),
            )
            if branch["candidate"] is not None
            else json.dumps(
                sorted(native_neighborhood),
                separators=(",", ":"),
            )
        ),
        "trial_index": trial_index,
        "random_seed": random_seed,
        "requested_pp_random_seed": requested_pp_seed,
        "applied_pp_random_seed": applied_pp_seed,
        "repair_order": json.dumps(
            repair_order,
            separators=(",", ":"),
        ),
        "before_fingerprint": source_before_fingerprint,
        "before_repair_fingerprint": before_repair_fingerprint,
        **restore_evidence,
        "replay_fingerprint_match": True,
        "after_fingerprint": state_fingerprint(after),
        "after_repair_fingerprint": after_repair_fingerprint,
        "replan_success": replan_success,
        "repair_outcome": repair_outcome,
        "terminated": terminated,
        "truncated": truncated,
        "conflicts_before": conflicts_before,
        "conflicts_after": conflicts_after,
        "conflict_delta": conflicts_before - conflicts_after,
        "sum_of_costs_delta": int(after["sum_of_costs"]) - int(before["sum_of_costs"]),
        "expanded": _strict_int(
            low_level_delta.get("expanded", 0), field="expanded", minimum=0
        ),
        "generated": _strict_int(
            low_level_delta.get("generated", 0), field="generated", minimum=0
        ),
        "reopened": _strict_int(
            low_level_delta.get("reopened", 0), field="reopened", minimum=0
        ),
        "model_selection_seconds": model_selection_seconds,
        "native_neighborhood_generation_seconds": native_selection_seconds,
        "pp_replan_seconds": _finite_nonnegative(
            metrics.get("pp_replan_seconds", 0.0), field="PP replan seconds"
        ),
        "repair_wall_seconds": repair_wall_seconds,
        "total_decision_seconds": model_selection_seconds + repair_wall_seconds,
    }


def _markdown(report: dict[str, Any], summaries: list[dict[str, Any]]) -> str:
    diagnostic = dict(report["diagnostic_signal"])
    lines = [
        "# Stalled-state repair probe",
        "",
        f"- Task: `{report['task_id']}`; solver seed: `{report['solver_seed']}`.",
        f"- Target decision: `{report['decision_index']}`; terminal stagnant run: `{report['terminal_stall_length']}` decisions.",
        f"- Before fingerprint: `{report['before_fingerprint']}`.",
        f"- Trials per unique branch: `{report['trials_per_branch']}`.",
        f"- One-step decision status: `{diagnostic['status']}`.",
        "",
        "| Branch | PP success | Positive reductions | Median conflict delta | Median PP seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {branch} | {replan_success_count}/{trial_count} | "
            "{positive_conflict_reduction_count}/{trial_count} | {median_conflict_delta:.3f} | "
            "{median_pp_replan_seconds:.6f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "The probe executes exactly one repair per branch. It does not use Horizon-4 continuation and does not replace complete-episode evaluation.",
            "",
            "No promotion or selector-failure conclusion is produced here; run the full-pool stable paired Oracle audit.",
            "",
        ]
    )
    return "\n".join(lines)


def run_stalled_state_probe(
    source: str | Path,
    output: str | Path,
    *,
    task_id: str,
    solver_seed: int,
    trials: int = 8,
    auto_terminal_stall: bool = True,
    decision_index: int | None = None,
    all_candidates: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    if auto_terminal_stall == (decision_index is not None):
        raise ValueError("choose exactly one of auto terminal stall or decision index")
    collection_root = Path(source).resolve()
    output_root = Path(output).resolve()
    source_run = _read_json(collection_root / "run_config.json")
    if str(source_run.get("controller")) != "v2-full":
        raise ValueError("stalled-state source must use v2-full")
    configuration = dict(source_run["configuration"])
    manifests = [
        row
        for row in _read_jsonl(collection_root / "realized_dynamic_manifest.jsonl")
        if str(row.get("task_id")) == task_id
        and int(row.get("solver_seed", -1)) == int(solver_seed)
        and str(row.get("status")) in {"ok", "resumed"}
    ]
    if len(manifests) != 1:
        raise ValueError("source collection does not contain exactly one requested episode")
    manifest = dict(manifests[0])
    validate_manifest_trace(
        collection_root,
        manifest,
        run_fingerprint=str(source_run["run_fingerprint"]),
        expected_policy="realized_dynamic",
    )
    decisions, _events = decision_rows(collection_root, manifest)
    if auto_terminal_stall:
        stall = find_terminal_stall(decisions)
        decision = dict(stall["decision"])
        terminal_stall_length = int(stall["length"])
    else:
        matches = [
            row for row in decisions if int(row["decision_index"]) == int(decision_index)
        ]
        if len(matches) != 1:
            raise ValueError("requested decision index is absent from the source trace")
        decision = dict(matches[0])
        terminal_stall_length = 0

    dataset_root = Path(str(source_run["dataset"])).resolve()
    rows = {
        str(row["task_id"]): row
        for row in _load_dataset_rows(dataset_root, [str(configuration["split"])])
    }
    if task_id not in rows:
        raise ValueError("requested task is missing from the source dataset")
    job = {
        "dataset_root": str(dataset_root),
        "row": rows[task_id],
        "environment": dict(configuration["environment"]),
        "solver_seed": int(solver_seed),
    }
    controller_path = Path(str(configuration["controller_bundle"])).resolve()
    bundle = load_controller_bundle(controller_path)
    source_controller_manifest = source_run.get("controller_bundle")
    if not isinstance(source_controller_manifest, dict) or _fingerprint(
        bundle.manifest
    ) != _fingerprint(source_controller_manifest):
        raise ValueError(
            "current controller bundle differs from the source collection"
        )
    model = bundle.main_models["realized_dynamic"]
    model_ranges = bundle.main_ranges["realized_dynamic"]
    environment, before = replay_prefix(job, decision["prefix_actions"])
    before_fingerprint = state_fingerprint(before)
    if before_fingerprint != str(decision["before_fingerprint"]):
        raise RuntimeError("target replay fingerprint does not match the source trace")
    candidates, scores, selection = _model_candidates(
        environment,
        before,
        task_id=task_id,
        solver_seed=int(solver_seed),
        decision_index=int(decision["decision_index"]),
        configuration=configuration,
        model=model,
        model_ranges=model_ranges,
    )
    source_transition_matches = [
        event
        for event in _events
        if str(event.get("event")) == "transition"
        and int(event.get("decision_index", -1)) == int(decision["decision_index"])
    ]
    if len(source_transition_matches) != 1:
        raise ValueError("source trace does not identify exactly one target transition")
    source_controller = source_transition_matches[0].get("controller")
    if not isinstance(source_controller, dict) or str(
        source_controller.get("controller_mode")
    ) != "v2-full":
        raise ValueError("source target is missing its frozen v2 controller evidence")
    source_pool = source_controller.get("candidate_pool")
    if not isinstance(source_pool, list) or not source_pool:
        raise ValueError("source target is missing its candidate pool")
    source_pool_signature = _candidate_pool_signature(source_pool)
    current_pool_signature = _candidate_pool_signature(candidates, scores)
    if source_pool_signature != current_pool_signature:
        raise ValueError(
            "current candidate pool or scores differ from the source transition"
        )
    current_selected_id = str(candidates[int(selection["selected_index"])]["candidate_id"])
    source_base_id, source_selected_id = _source_v2_selection_ids(
        source_controller
    )
    if source_base_id != current_selected_id or source_selected_id != current_selected_id:
        raise ValueError("current v2 rank1 differs from the source transition")
    candidate_pool_fingerprint = _fingerprint(current_pool_signature)
    branches, alias_to_key = choose_probe_branches(
        candidates, scores, all_candidates=all_candidates
    )
    producer = producer_identity(
        project_root=Path(__file__).resolve().parents[1],
        source_files=STALLED_STATE_PROBE_PRODUCER_FILES,
        native_required=True,
        optional_package_names=("numpy", "scikit-learn"),
    )
    identity = {
        "schema": STALLED_STATE_PROBE_SCHEMA,
        "schema_version": STALLED_STATE_PROBE_VERSION,
        "source_collection": str(collection_root),
        "source_run_fingerprint": str(source_run["run_fingerprint"]),
        "source_trace_sha256": str(manifest["trace_sha256"]),
        "source_controller_manifest": source_controller_manifest,
        "candidate_pool_fingerprint": candidate_pool_fingerprint,
        "branch_aliases": dict(sorted(alias_to_key.items())),
        "task_id": task_id,
        "solver_seed": int(solver_seed),
        "decision_index": int(decision["decision_index"]),
        "trials": int(trials),
        "all_candidates": bool(all_candidates),
        "trial_state_restore": TRIAL_STATE_RESTORE,
        "producer_identity": producer,
        "producer_identity_fingerprint": _fingerprint(producer),
    }
    runner = prepare_run_output(output_root, resume=resume, identity=identity)
    run_fingerprint = str(runner["identity_fingerprint"])

    reproduction_environment, reproduction_before = replay_prefix(
        job, decision["prefix_actions"]
    )
    if state_fingerprint(reproduction_before) != before_fingerprint:
        raise RuntimeError("source reproduction before fingerprint mismatch")
    reproduced = _plain(reproduction_environment.step(dict(decision["actual_action"])))
    reproduction_after = dict(reproduced["observation"])
    reproduction = {
        "before_fingerprint_match": True,
        "after_fingerprint_match": state_fingerprint(reproduction_after)
        == str(decision["after_fingerprint"]),
        "replan_success_match": bool(dict(reproduced["metrics"]).get("replan_success"))
        == bool(dict(decision["actual_metrics"]).get("replan_success")),
        "conflicts_after_match": int(reproduction_after["num_of_colliding_pairs"])
        == int(dict(decision["actual_metrics"])["conflicts_after"]),
    }
    if not all(reproduction.values()):
        raise RuntimeError("source decision could not be reproduced exactly")
    _write_json(output_root / "source_reproduction.json", reproduction)

    completed: list[dict[str, Any]] = []
    before_repair_fingerprint = repair_structure_fingerprint(before)
    for branch in branches:
        for trial_index in range(trials):
            checkpoint = (
                output_root
                / "checkpoints"
                / str(branch["branch_key"])
                / f"trial-{trial_index:03d}.json"
            )
            existing = (
                _checkpoint_valid(
                    checkpoint,
                    run_fingerprint=run_fingerprint,
                    branch=branch,
                    trial_index=trial_index,
                    before_fingerprint=before_fingerprint,
                    before_repair_fingerprint=before_repair_fingerprint,
                )
                if resume
                else None
            )
            if existing is not None:
                completed.append(existing)
                continue
            result = _run_trial(
                job=job,
                decision=decision,
                source_state=before,
                branch=branch,
                trial_index=trial_index,
                selection_seconds=float(selection["selection_seconds"]),
                run_fingerprint=run_fingerprint,
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _write_json(checkpoint, result)
            validated = _checkpoint_valid(
                checkpoint,
                run_fingerprint=run_fingerprint,
                branch=branch,
                trial_index=trial_index,
                before_fingerprint=before_fingerprint,
                before_repair_fingerprint=before_repair_fingerprint,
            )
            assert validated is not None
            completed.append(validated)

    summaries, gate = summarize_probe_trials(completed, alias_to_key)
    report = {
        "schema": STALLED_STATE_PROBE_SCHEMA,
        "schema_version": STALLED_STATE_PROBE_VERSION,
        "run_fingerprint": run_fingerprint,
        "source_collection": str(collection_root),
        "source_episode_id": str(manifest["episode_id"]),
        "task_id": task_id,
        "solver_seed": int(solver_seed),
        "decision_index": int(decision["decision_index"]),
        "terminal_stall_length": terminal_stall_length,
        "before_fingerprint": before_fingerprint,
        "before_repair_fingerprint": before_repair_fingerprint,
        "trials_per_branch": int(trials),
        "unique_branch_count": len(branches),
        "all_candidates": bool(all_candidates),
        "trial_state_restore": TRIAL_STATE_RESTORE,
        "candidate_count": len(candidates),
        "candidate_pool_fingerprint": candidate_pool_fingerprint,
        "candidate_pool": current_pool_signature,
        "branch_aliases": dict(sorted(alias_to_key.items())),
        "model_selection": selection,
        "source_reproduction": reproduction,
        "diagnostic_signal": gate,
        "producer_identity": producer,
        "producer_identity_fingerprint": _fingerprint(producer),
    }
    _write_csv(output_root / "stalled_state_trials.csv", completed)
    _write_csv(output_root / "stalled_state_summary.csv", summaries)
    _write_json(output_root / "stalled_state_probe_report.json", report)
    markdown = _markdown(report, summaries)
    partial = output_root / "stalled_state_probe_report.md.partial"
    partial.write_text(markdown, encoding="utf-8")
    os.replace(partial, output_root / "stalled_state_probe_report.md")
    return report


__all__ = [
    "STALLED_STATE_PROBE_SCHEMA",
    "choose_probe_branches",
    "find_terminal_stall",
    "paired_probe_seed",
    "run_stalled_state_probe",
    "summarize_probe_trials",
]
