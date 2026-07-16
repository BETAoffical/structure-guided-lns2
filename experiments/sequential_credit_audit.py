from __future__ import annotations

import collections
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    append_jsonl_fsync as _append_jsonl,
    mean as _mean,
    quantile as _quantile,
    sha256_file as _sha256,
    state_storage_id as _state_storage_id,
    trial_job_id as _trial_job_id,
)
from experiments.closed_loop_confirmation import (
    feature_range_diagnostic,
    fixed_budget_conflict_auc,
    generate_online_candidates,
    load_frozen_policy_bundle,
    online_candidate_rows,
    repair_random_seed,
    score_online_candidates,
)
from experiments.local_representation_audit import analyze_static_grid
from experiments.policy_visited_aggregation import candidate_core
from experiments.realized_neighborhood_probe import evaluation_seed
from experiments.repair_collection import (
    SCHEMA_VERSION,
    CollectionLockError,
    _CollectionRunLock,
    _dataset_fingerprint,
    _fingerprint,
    _policy_train_dataset_lookup as _dataset_lookup,
    _low_level_delta,
    _make_environment,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)


AUDIT_SCHEMA = "lns2.sequential_credit_audit.v1"
STATE_SCHEMA = "lns2.sequential_credit_state.v1"
TRIAL_SCHEMA = "lns2.sequential_credit_trial.v1"
INDEX_SCHEMA = "lns2.sequential_credit_index.v1"
REPORT_SCHEMA = "lns2.sequential_credit_report.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATION_FILES = (
    "experiments/_common.py",
    "experiments/sequential_credit_audit.py",
    "experiments/closed_loop_confirmation.py",
    "experiments/policy_visited_aggregation.py",
    "experiments/repair_collection.py",
)


def _resolve(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    first, second = set(left), set(right)
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    left_mean, right_mean = _mean(left), _mean(right)
    numerator = sum(
        (first - left_mean) * (second - right_mean)
        for first, second in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    denominator = left_scale * right_scale
    return numerator / denominator if denominator else 0.0


def _average_ranks(values: list[tuple[float, ...]]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = rank
        cursor = end
    return ranks


def _spearman_effectiveness(
    left: list[dict[str, Any]], right: list[dict[str, Any]], horizon: str = "h4"
) -> float:
    if horizon not in {"h1", "h4"}:
        raise ValueError("effectiveness horizon must be h1 or h4")
    left_by_id = {str(row["candidate_id"]): row for row in left}
    right_by_id = {str(row["candidate_id"]): row for row in right}
    candidate_ids = sorted(left_by_id)
    if candidate_ids != sorted(right_by_id):
        raise ValueError("split-half candidate sets differ")

    def values(rows: dict[str, dict[str, Any]]) -> list[tuple[float, ...]]:
        result = []
        for key in candidate_ids:
            row = rows[key]
            metrics = row.get(horizon, row)
            result.append(
                (
                    -float(metrics["feasible_rate"]),
                    float(metrics["final_conflicts"]),
                    float(metrics["conflict_auc"]),
                )
            )
        return result

    return _pearson(_average_ranks(values(left_by_id)), _average_ranks(values(right_by_id)))


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported sequential-credit config")
    if str(config.get("continuation_policy")) != "realized_dynamic":
        raise ValueError("continuation policy must be frozen realized_dynamic v1")
    selection = dict(config.get("selection", {}))
    if (
        int(selection.get("map_count", 0)) != 12
        or int(selection.get("states_per_map", 0)) != 8
        or int(selection.get("total_states", 0)) != 96
    ):
        raise ValueError("formal selection must contain 8 states from each of 12 maps")
    if list(selection.get("stratification", [])) != [
        "stage",
        "conflict_severity",
        "task_variant",
        "solver_seed",
    ]:
        raise ValueError("state stratification differs from the preregistration")
    thresholds = list(map(float, selection.get("conflict_density_thresholds", [])))
    if len(thresholds) != 2 or not 0.0 < thresholds[0] < thresholds[1]:
        raise ValueError("conflict-density thresholds are invalid")
    if int(config.get("evaluation_trials", 0)) != 4 or int(config.get("horizon", 0)) != 4:
        raise ValueError("sequential-credit audit requires four trials and Horizon 4")
    if int(config.get("workers", 0)) != 4:
        raise ValueError("formal sequential-credit collection uses four workers")
    if float(config.get("trial_process_timeout_seconds", 0.0)) != 180.0:
        raise ValueError("formal trial timeout must be 180 seconds")
    if int(config.get("bootstrap_samples", 0)) != 5000:
        raise ValueError("formal analysis requires 5,000 map bootstrap samples")
    proposal = dict(config.get("proposal", {}))
    if (
        int(proposal.get("max_seed_agents", 0)) != 4
        or list(map(str, proposal.get("heuristics", [])))
        != ["target", "collision", "random"]
        or list(map(int, proposal.get("neighborhood_sizes", []))) != [4, 8, 16]
        or int(proposal.get("trials", 0)) != 8
        or int(proposal.get("candidates_per_family", 0)) != 2
    ):
        raise ValueError("proposal configuration differs from frozen v1")


def _registered_inputs(config: dict[str, Any]) -> dict[str, Any]:
    source = dict(config["source"])
    source_root = _resolve(source["collection"])
    paths = {
        "selected_states": source_root / str(source["selected_states"]),
        "candidates": source_root / str(source["candidates"]),
        "source_run_config": source_root / str(source["run_config"]),
        "source_evaluation_trials": source_root / str(source["evaluation_trials"]),
        "legacy_quality_report": _resolve(config["legacy_diagnosis"]["quality_report"]),
        "legacy_run_config": _resolve(config["legacy_diagnosis"]["run_config"]),
    }
    expected = {
        "selected_states": str(source["selected_states_sha256"]),
        "candidates": str(source["candidates_sha256"]),
        "source_run_config": str(source["run_config_sha256"]),
        "source_evaluation_trials": str(source["evaluation_trials_sha256"]),
        "legacy_quality_report": str(config["legacy_diagnosis"]["quality_report_sha256"]),
        "legacy_run_config": str(config["legacy_diagnosis"]["run_config_sha256"]),
    }
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"registered input is missing: {path}")
        hashes[name] = _sha256(path)
        if hashes[name] != expected[name]:
            raise ValueError(f"registered input SHA256 mismatch: {name}")
    dataset_root = _resolve(config["dataset"])
    dataset_hash = _dataset_fingerprint(dataset_root)
    if dataset_hash != str(config["dataset_fingerprint"]):
        raise ValueError("policy-visited dataset fingerprint mismatch")
    implementation = {
        relative: _sha256(PROJECT_ROOT / relative)
        for relative in IMPLEMENTATION_FILES
        if (PROJECT_ROOT / relative).is_file()
    }
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "sha256": hashes,
        "dataset": str(dataset_root),
        "dataset_fingerprint": dataset_hash,
        "implementation": implementation,
    }


def diagnose_legacy(config: dict[str, Any] | str | Path) -> dict[str, Any]:
    if not isinstance(config, dict):
        config = _read_json(_resolve(config))
    _validate_config(config)
    registered = _registered_inputs(config)
    report = _read_json(Path(registered["paths"]["legacy_quality_report"]))
    expected = dict(config["legacy_diagnosis"])
    counts = dict(report.get("counts", {}))
    integrity = dict(report.get("integrity", {}))
    overlap = float(report.get("horizon_pareto_overlap", {}).get("mean", math.nan))
    tolerance = float(expected["absolute_tolerance"])
    checks = {
        "state_count": int(counts.get("states", -1)) == int(expected["expected_states"]),
        "outcome_count": int(counts.get("outcomes", -1)) == int(expected["expected_outcomes"]),
        "zero_errors": int(counts.get("errors", -1)) == 0,
        "zero_replay_mismatches": int(integrity.get("replay_mismatches", -1)) == 0,
        "zero_invalid_actions": int(integrity.get("invalid_actions", -1)) == 0,
        "no_test_ood_labels": int(integrity.get("test_ood_states", -1)) == 0,
        "h1_h4_pareto_overlap": math.isclose(
            overlap,
            float(expected["expected_h1_h4_pareto_jaccard"]),
            rel_tol=0.0,
            abs_tol=tolerance,
        ),
    }
    return {
        "schema": AUDIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "passed": all(checks.values()),
        "checks": checks,
        "states": int(counts.get("states", 0)),
        "outcomes": int(counts.get("outcomes", 0)),
        "h1_h4_pareto_jaccard": overlap,
        "registered_inputs": registered,
    }


def conflict_severity(row: dict[str, Any], thresholds: list[float]) -> str:
    conflicts = int(row["state"]["num_of_colliding_pairs"])
    agents = int(row["agent_count"])
    density = 2.0 * conflicts / (agents * (agents - 1)) if agents > 1 else 0.0
    if density <= thresholds[0]:
        return "low"
    if density <= thresholds[1]:
        return "medium"
    return "high"


def select_audit_states(
    candidate_rows: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    selection = dict(config["selection"])
    required_split = str(config["source"]["required_split"])
    thresholds = list(map(float, selection["conflict_density_thresholds"]))
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for source in candidate_rows:
        if str(source.get("split")) != required_split:
            continue
        row = _plain(source)
        row["conflict_severity"] = conflict_severity(row, thresholds)
        row["conflict_density"] = (
            2.0
            * int(row["state"]["num_of_colliding_pairs"])
            / (int(row["agent_count"]) * (int(row["agent_count"]) - 1))
        )
        grouped[str(row["map_id"])].append(row)
    if len(grouped) != int(selection["map_count"]):
        raise ValueError("policy_train source does not contain exactly 12 maps")
    selected: list[dict[str, Any]] = []
    per_map = int(selection["states_per_map"])
    for map_id in sorted(grouped):
        strata: dict[tuple[Any, ...], list[dict[str, Any]]] = collections.defaultdict(list)
        for row in grouped[map_id]:
            key = (
                str(row["stage"]),
                str(row["conflict_severity"]),
                str(row["task_variant"]),
                int(row["solver_seed"]),
            )
            strata[key].append(row)
        for key, values in strata.items():
            values.sort(
                key=lambda value: _fingerprint(
                    {
                        "namespace": "sequential-credit-state-selection-v1",
                        "map_id": map_id,
                        "stratum": key,
                        "state_id": value["state_id"],
                    }
                )
            )
        keys = sorted(
            strata,
            key=lambda key: _fingerprint(
                {
                    "namespace": "sequential-credit-stratum-order-v1",
                    "map_id": map_id,
                    "stratum": key,
                }
            ),
        )
        map_rows = []
        while len(map_rows) < per_map and any(strata.values()):
            for key in keys:
                if strata[key] and len(map_rows) < per_map:
                    map_rows.append(strata[key].pop(0))
        if len(map_rows) != per_map:
            raise ValueError(f"map does not provide {per_map} states: {map_id}")
        selected.extend(map_rows)
    selected.sort(key=lambda row: (str(row["map_id"]), str(row["state_id"])))
    if len(selected) != int(selection["total_states"]):
        raise ValueError("selected state count differs from preregistration")
    if len({str(row["state_id"]) for row in selected}) != len(selected):
        raise ValueError("selected states are not unique")
    return selected


def _source_integrity(
    selected_states: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    state_by_id = {str(row["state_id"]): row for row in selected_states}
    candidate_by_id = {str(row["state_id"]): row for row in candidate_rows}
    if len(state_by_id) != len(selected_states) or len(candidate_by_id) != len(candidate_rows):
        raise ValueError("source contains duplicate state IDs")
    if set(state_by_id) != set(candidate_by_id):
        raise ValueError("selected-state and candidate source IDs differ")
    mismatches = []
    missing_selected = []
    for state_id, state_row in state_by_id.items():
        candidate_row = candidate_by_id[state_id]
        for key in ("state_fingerprint", "map_id", "task_id", "solver_seed", "split"):
            if state_row.get(key) != candidate_row.get(key):
                mismatches.append({"state_id": state_id, "field": key})
        candidate_ids = {str(value["candidate_id"]) for value in candidate_row["candidates"]}
        if str(candidate_row["source_selected_candidate_id"]) not in candidate_ids:
            missing_selected.append(state_id)
    if mismatches or missing_selected:
        raise ValueError("policy-visited source integrity check failed")
    return {
        "state_count": len(state_by_id),
        "candidate_state_count": len(candidate_by_id),
        "mismatch_count": len(mismatches),
        "missing_source_selection_count": len(missing_selected),
    }


def _prepare_path(output_root: Path, state_id: str) -> Path:
    return output_root / "prepared_states" / f"{_state_storage_id(state_id)}.json"


def _prepare_state_worker(job: dict[str, Any]) -> dict[str, Any]:
    source = job["state_row"]
    state_id = str(source["state_id"])
    common = {
        "schema": STATE_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": job["run_fingerprint"],
        "job_id": f"prepare__{_state_storage_id(state_id)}",
        "state_id": state_id,
        "split": str(source["split"]),
        "map_id": str(source["map_id"]),
        "task_id": str(source["task_id"]),
        "solver_seed": int(source["solver_seed"]),
    }
    try:
        environment = _make_environment(
            job["dataset_root"], job["row"], job["environment"], "Adaptive"
        )
        state = _plain(environment.reset(seed=int(source["solver_seed"])))
        for action in source["prefix_actions"]:
            if bool(state["done"]):
                raise RuntimeError("prefix terminated before selected state")
            state = _plain(environment.step(action))["observation"]
        if state_fingerprint(state) != str(source["state_fingerprint"]):
            raise RuntimeError("state fingerprint mismatch during preparation")
        candidates, metrics = generate_online_candidates(
            environment,
            state,
            task_id=str(source["task_id"]),
            solver_seed=int(source["solver_seed"]),
            decision_index=int(source["decision_index"]),
            proposal_config=job["proposal"],
        )
        actual = sorted(
            (candidate_core(value) for value in candidates),
            key=lambda value: str(value["candidate_id"]),
        )
        expected = sorted(
            (candidate_core(value) for value in source["candidates"]),
            key=lambda value: str(value["candidate_id"]),
        )
        if actual != expected:
            raise RuntimeError("regenerated candidate pool differs from source trace")
        return {
            **common,
            "status": "ok",
            "complete": True,
            "fingerprint_match": True,
            "candidate_pool_match": True,
            "candidate_count": len(actual),
            "candidate_pool_fingerprint": _fingerprint(actual),
            "proposal_metrics": metrics,
            "error": None,
        }
    except Exception as error:
        return {
            **common,
            "status": "error",
            "complete": False,
            "fingerprint_match": False,
            "candidate_pool_match": False,
            "candidate_count": 0,
            "error": f"{type(error).__name__}: {error}",
        }


def _trial_path(output_root: Path, job: dict[str, Any]) -> Path:
    return (
        output_root
        / "trials"
        / _state_storage_id(str(job["state_id"]))
        / str(job["candidate_id"])
        / f"trial_{int(job['trial_index']):04d}.json"
    )


def _compact_candidate(candidate: dict[str, Any], score: float | None = None) -> dict[str, Any]:
    value = candidate_core(candidate)
    if score is not None:
        value["score"] = float(score)
    return value


def _apply_explicit_action(
    environment: Any,
    state: dict[str, Any],
    candidate: dict[str, Any],
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int], float]:
    agents = sorted(map(int, candidate["agents"]))
    started = time.perf_counter()
    result = _plain(
        environment.step(
            {
                "mode": "explicit_neighborhood",
                "agents": agents,
                "random_seed": int(random_seed),
            }
        )
    )
    wall_seconds = time.perf_counter() - started
    metrics = dict(result["metrics"])
    actual = sorted(map(int, metrics.get("neighborhood", [])))
    if not bool(metrics.get("action_valid")) or actual != agents:
        raise RuntimeError("explicit candidate was rejected or changed")
    after = dict(result["observation"])
    return after, metrics, _low_level_delta(state, after), wall_seconds


def execute_horizon4_trial(
    environment: Any,
    source: dict[str, Any],
    initial_candidate: dict[str, Any],
    trial_index: int,
    bundle: Any,
    proposal_config: dict[str, Any],
) -> dict[str, Any]:
    state_id = str(source["state_id"])
    candidate_id = str(initial_candidate["candidate_id"])
    proposal_seeds = sorted(map(int, initial_candidate["proposal_seeds"]))
    initial_seed = evaluation_seed(state_id, candidate_id, trial_index, proposal_seeds)
    state = _plain(environment.reset(seed=int(source["solver_seed"])))
    for action in source["prefix_actions"]:
        if bool(state["done"]):
            raise RuntimeError("trial prefix terminated before selected state")
        state = _plain(environment.step(action))["observation"]
    if state_fingerprint(state) != str(source["state_fingerprint"]):
        raise RuntimeError("trial prefix replay fingerprint mismatch")
    static_grid = analyze_static_grid(state)
    trajectory = [int(state["num_of_colliding_pairs"])]
    steps: list[dict[str, Any]] = []
    low_level_totals: collections.Counter[str] = collections.Counter()
    runtime_totals: collections.Counter[str] = collections.Counter()
    h1_feasible = False
    h1_conflicts = trajectory[0]
    for rollout_step in range(4):
        if bool(state["done"]):
            break
        before = state
        before_hash = state_fingerprint(before)
        decision_index = int(source["decision_index"]) + rollout_step
        if rollout_step == 0:
            selected = initial_candidate
            random_seed = initial_seed
            controller = {
                "policy": "counterfactual_initial_candidate",
                "candidate_pool": [_compact_candidate(value) for value in source["candidates"]],
                "selected_candidate_id": candidate_id,
                "evaluation_trial_index": trial_index,
            }
        else:
            proposal_started = time.perf_counter()
            candidates, proposal_metrics = generate_online_candidates(
                environment,
                before,
                task_id=str(source["task_id"]),
                solver_seed=int(source["solver_seed"]),
                decision_index=decision_index,
                proposal_config=proposal_config,
            )
            feature_started = time.perf_counter()
            candidate_rows = online_candidate_rows(before, candidates, static_grid=static_grid)
            feature_seconds = time.perf_counter() - feature_started
            inference_started = time.perf_counter()
            selected_index, scores, margin = score_online_candidates(
                candidate_rows, bundle.models["realized_dynamic"]
            )
            inference_seconds = time.perf_counter() - inference_started
            selected = candidates[selected_index]
            random_seed = repair_random_seed(
                str(source["task_id"]),
                int(source["solver_seed"]),
                before_hash,
                decision_index,
                str(selected["candidate_id"]),
                selected["proposal_seeds"],
            )
            diagnostic = feature_range_diagnostic(
                candidate_rows[selected_index],
                "realized_dynamic",
                bundle.ranges["realized_dynamic"],
            )
            controller = {
                "policy": "frozen_v1_realized_dynamic",
                "candidate_pool": [
                    _compact_candidate(candidate, scores[index])
                    for index, candidate in enumerate(candidates)
                ],
                "selected_candidate_id": str(selected["candidate_id"]),
                "selected_score": float(scores[selected_index]),
                "score_margin": float(margin),
                "selected_feature_range": diagnostic,
                "proposal": proposal_metrics,
                "feature_seconds": feature_seconds,
                "inference_seconds": inference_seconds,
                "controller_seconds_before_repair": time.perf_counter() - proposal_started,
            }
            runtime_totals["proposal"] += float(proposal_metrics["proposal_seconds"])
            runtime_totals["feature"] += feature_seconds
            runtime_totals["inference"] += inference_seconds
        after, metrics, low_level, repair_wall = _apply_explicit_action(
            environment, before, selected, random_seed
        )
        for key, value in low_level.items():
            low_level_totals[key] += int(value)
        runtime_totals["repair_wall"] += repair_wall
        trajectory.append(int(after["num_of_colliding_pairs"]))
        steps.append(
            {
                "rollout_step": rollout_step + 1,
                "decision_index": decision_index,
                "before_fingerprint": before_hash,
                "after_fingerprint": state_fingerprint(after),
                "random_seed": int(random_seed),
                "selected_candidate_id": str(selected["candidate_id"]),
                "selected_agents": sorted(map(int, selected["agents"])),
                "controller": controller,
                "metrics": metrics,
                "low_level_delta": low_level,
                "repair_wall_seconds": repair_wall,
                "conflicts_before": trajectory[-2],
                "conflicts_after": trajectory[-1],
                "feasible_after": bool(after["feasible"]),
            }
        )
        state = after
        if rollout_step == 0:
            h1_feasible = bool(after["feasible"])
            h1_conflicts = trajectory[-1]
    success = bool(state["feasible"])
    padded = list(trajectory)
    padded.extend(([0] if success else [padded[-1]]) * (5 - len(padded)))
    return {
        "initial_evaluation_seed": initial_seed,
        "evaluation_seed_disjoint": initial_seed not in set(proposal_seeds),
        "raw_conflict_trajectory": trajectory,
        "padded_conflict_trajectory": padded,
        "h1": {
            "feasible": h1_feasible,
            "final_conflicts": h1_conflicts,
            "conflict_auc": (trajectory[0] + h1_conflicts) / 2.0,
        },
        "h4": {
            "feasible": success,
            "final_conflicts": padded[4],
            "conflict_auc": fixed_budget_conflict_auc(trajectory, 4, success=success),
        },
        "low_level": dict(low_level_totals),
        "runtime": dict(runtime_totals),
        "steps": steps,
        "final_fingerprint": state_fingerprint(state),
    }


def _trial_worker(job: dict[str, Any]) -> dict[str, Any]:
    source = job["state_row"]
    candidate = job["candidate"]
    common = {
        "schema": TRIAL_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": job["run_fingerprint"],
        "job_id": job["job_id"],
        "state_id": str(source["state_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "trial_index": int(job["trial_index"]),
        "split": str(source["split"]),
        "map_id": str(source["map_id"]),
        "task_id": str(source["task_id"]),
        "solver_seed": int(source["solver_seed"]),
    }
    try:
        environment = _make_environment(
            job["dataset_root"], job["row"], job["environment"], "Adaptive"
        )
        bundle = load_frozen_policy_bundle(job["frozen_models"], job["model_registration"])
        outcome = execute_horizon4_trial(
            environment,
            source,
            candidate,
            int(job["trial_index"]),
            bundle,
            job["proposal"],
        )
        return {
            **common,
            "status": "ok",
            "complete": True,
            "outcome_count": 1,
            "outcome": outcome,
            "error": None,
        }
    except Exception as error:
        return {
            **common,
            "status": "error",
            "complete": False,
            "outcome_count": 0,
            "outcome": None,
            "error": f"{type(error).__name__}: {error}",
        }


def _load_source(config: dict[str, Any], registered: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    states = _read_jsonl(Path(registered["paths"]["selected_states"]))
    candidates = _read_jsonl(Path(registered["paths"]["candidates"]))
    _source_integrity(states, candidates)
    return states, candidates


def _run_metadata(
    config: dict[str, Any], registered: dict[str, Any], selected: list[dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    payload = {
        "configuration_fingerprint": _fingerprint(config),
        "registered_sha256": registered["sha256"],
        "dataset_fingerprint": registered["dataset_fingerprint"],
        "implementation": registered["implementation"],
        "selected_state_ids": [str(row["state_id"]) for row in selected],
    }
    run_fingerprint = _fingerprint(payload)
    return run_fingerprint, {
        "schema": AUDIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        **payload,
        "dataset": registered["dataset"],
        "state_count": len(selected),
        "candidate_count": sum(len(row["candidates"]) for row in selected),
        "trial_count": sum(len(row["candidates"]) for row in selected)
        * int(config["evaluation_trials"]),
    }


def build_dry_run(
    config: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    smoke_states: int | None,
    observed_repair_seconds_min: float = 0.0,
) -> dict[str, Any]:
    active = selected[:smoke_states] if smoke_states is not None else selected
    candidates = sum(len(row["candidates"]) for row in active)
    trials = candidates * int(config["evaluation_trials"])
    horizon = int(config["horizon"])
    # Each trial must perform the explicit first action; continuation can add at most H-1.
    lower_repairs, upper_repairs = trials, trials * horizon
    minimal_trial_row = len(
        json.dumps(
            {
                "state_id": "x" * 80,
                "candidate_id": "x" * 32,
                "trial_index": 0,
                "status": "ok",
                "raw_conflict_trajectory": [0] * (horizon + 1),
            }
        ).encode("utf-8")
    )
    return {
        "schema": AUDIT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": smoke_states is None,
        "state_count": len(active),
        "candidate_count": candidates,
        "trial_jobs": trials,
        "repair_count_lower_bound": lower_repairs,
        "repair_count_upper_bound": upper_repairs,
        "worker_count": int(config["workers"]),
        "trial_timeout_seconds": float(config["trial_process_timeout_seconds"]),
        "serial_native_repair_seconds_empirical_lower_bound": (
            lower_repairs * observed_repair_seconds_min
        ),
        "empirical_min_native_repair_seconds": observed_repair_seconds_min,
        "cpu_bound_note": (
            "Empirical floor from the registered 31,656 Horizon-1 trials; it excludes "
            "process startup, replay, proposal, feature extraction and inference."
        ),
        "result_bytes_lower_bound": trials * minimal_trial_row,
        "result_bytes_note": "Serialized skeleton only; full candidate pools make actual storage larger.",
        "states_by_map": dict(collections.Counter(str(row["map_id"]) for row in active)),
    }


def _prepare_states(
    output_root: Path,
    selected: list[dict[str, Any]],
    dataset_lookup: dict[str, dict[str, Any]],
    config: dict[str, Any],
    run_fingerprint: str,
    *,
    resume: bool,
    workers: int,
) -> list[dict[str, Any]]:
    resumed, pending = [], []
    for source in selected:
        path = _prepare_path(output_root, str(source["state_id"]))
        if resume and path.is_file():
            value = _read_json(path)
            if value.get("run_fingerprint") == run_fingerprint and bool(value.get("complete")):
                resumed.append({**value, "status": "resumed"})
                continue
        pending.append(
            {
                "job_id": f"prepare__{_state_storage_id(str(source['state_id']))}",
                "state_row": source,
                "row": dataset_lookup[str(source["task_id"])],
                "dataset_root": str(_resolve(config["dataset"])),
                "environment": config["environment"],
                "proposal": config["proposal"],
                "run_fingerprint": run_fingerprint,
            }
        )
    by_job = {str(job["job_id"]): job for job in pending}
    completed = []

    def record(result: dict[str, Any]) -> None:
        job = by_job[str(result["job_id"])]
        path = _prepare_path(output_root, str(job["state_row"]["state_id"]))
        _write_json(path, result)
        completed.append(result)
        _append_jsonl(output_root / "progress.jsonl", {"phase": "prepare", **result})

    if pending:
        _run_jobs(
            _prepare_state_worker,
            pending,
            workers,
            phase="sequential-credit-prepare",
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(config["prepare_process_timeout_seconds"]),
            on_result=record,
        )
    rows = sorted(resumed + completed, key=lambda row: str(row["state_id"]))
    _write_jsonl(output_root / "state_preparation_manifest.jsonl", rows)
    return rows


def _collect_trials(
    output_root: Path,
    selected: list[dict[str, Any]],
    dataset_lookup: dict[str, dict[str, Any]],
    config: dict[str, Any],
    run_fingerprint: str,
    *,
    resume: bool,
    workers: int,
) -> list[dict[str, Any]]:
    resumed, pending = [], []
    for source in selected:
        for candidate in source["candidates"]:
            for trial_index in range(int(config["evaluation_trials"])):
                job = {
                    "job_id": _trial_job_id(
                        str(source["state_id"]), str(candidate["candidate_id"]), trial_index
                    ),
                    "state_id": str(source["state_id"]),
                    "candidate_id": str(candidate["candidate_id"]),
                    "trial_index": trial_index,
                    "state_row": source,
                    "candidate": candidate,
                    "row": dataset_lookup[str(source["task_id"])],
                    "solver_seed": int(source["solver_seed"]),
                    "dataset_root": str(_resolve(config["dataset"])),
                    "environment": config["environment"],
                    "proposal": config["proposal"],
                    "frozen_models": str(_resolve(config["frozen_models"])),
                    "model_registration": config["model_registration"],
                    "run_fingerprint": run_fingerprint,
                }
                path = _trial_path(output_root, job)
                if resume and path.is_file():
                    value = _read_json(path)
                    if value.get("run_fingerprint") == run_fingerprint and bool(value.get("complete")):
                        resumed.append({**value, "status": "resumed"})
                        continue
                pending.append(job)
    by_job = {str(job["job_id"]): job for job in pending}
    completed = []

    def record(result: dict[str, Any]) -> None:
        job = by_job[str(result["job_id"])]
        normalized = {
            **result,
            "schema": TRIAL_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "job_id": job["job_id"],
            "state_id": job["state_id"],
            "candidate_id": job["candidate_id"],
            "trial_index": int(job["trial_index"]),
        }
        normalized["complete"] = str(normalized.get("status")) in {"ok", "resumed"}
        normalized["outcome_count"] = int(bool(normalized["complete"]))
        _write_json(_trial_path(output_root, job), normalized)
        completed.append(normalized)
        _append_jsonl(
            output_root / "progress.jsonl",
            {
                "phase": "trial",
                "job_id": normalized["job_id"],
                "status": normalized.get("status"),
                "state_id": normalized["state_id"],
                "candidate_id": normalized["candidate_id"],
                "trial_index": normalized["trial_index"],
            },
        )

    if pending:
        _run_jobs(
            _trial_worker,
            pending,
            workers,
            phase="sequential-credit-trial",
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(config["trial_process_timeout_seconds"]),
            on_result=record,
        )
    rows = sorted(
        resumed + completed,
        key=lambda row: (
            str(row["state_id"]),
            str(row["candidate_id"]),
            int(row["trial_index"]),
        ),
    )
    _write_jsonl(output_root / "trial_manifest.jsonl", rows)
    return rows


def _aggregate_metrics(trials: list[dict[str, Any]], horizon: str) -> dict[str, float]:
    values = [dict(row["outcome"])[horizon] for row in trials]
    return {
        "feasible_rate": _mean(value["feasible"] for value in values),
        "final_conflicts": _mean(value["final_conflicts"] for value in values),
        "conflict_auc": _mean(value["conflict_auc"] for value in values),
    }


def _dominates(
    left: dict[str, Any], right: dict[str, Any], *, compute_aware: bool = False
) -> bool:
    first = (-left["feasible_rate"], left["final_conflicts"], left["conflict_auc"])
    second = (-right["feasible_rate"], right["final_conflicts"], right["conflict_auc"])
    if compute_aware:
        first += (float(left["generated"]),)
        second += (float(right["generated"]),)
    return all(a <= b for a, b in zip(first, second)) and any(a < b for a, b in zip(first, second))


def pareto_ids(
    candidates: list[dict[str, Any]], horizon: str, *, compute_aware: bool = False
) -> set[str]:
    return {
        str(candidate["candidate_id"])
        for candidate in candidates
        if not any(
            other is not candidate
            and _dominates(
                {
                    **dict(other[horizon]),
                    "generated": float(other.get("generated", 0.0)),
                },
                {
                    **dict(candidate[horizon]),
                    "generated": float(candidate.get("generated", 0.0)),
                },
                compute_aware=compute_aware,
            )
            for other in candidates
        )
    }


def best_ids(candidates: list[dict[str, Any]], horizon: str) -> set[str]:
    if not candidates:
        return set()
    key = lambda row: (
        -float(row[horizon]["feasible_rate"]),
        float(row[horizon]["final_conflicts"]),
        float(row[horizon]["conflict_auc"]),
    )
    best = min(key(row) for row in candidates)
    return {str(row["candidate_id"]) for row in candidates if key(row) == best}


def aggregate_trials(
    selected: list[dict[str, Any]], trial_rows: list[dict[str, Any]], expected_trials: int
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trial_rows:
        grouped[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    index = []
    for state in selected:
        candidates = []
        for candidate in state["candidates"]:
            key = (str(state["state_id"]), str(candidate["candidate_id"]))
            trials = sorted(grouped.get(key, []), key=lambda row: int(row["trial_index"]))
            indices = [int(row["trial_index"]) for row in trials]
            if indices != list(range(expected_trials)):
                raise ValueError(f"incomplete trials: {key}")
            if any(not bool(row.get("complete")) or not row.get("outcome") for row in trials):
                raise ValueError(f"unsuccessful trial: {key}")
            candidates.append(
                {
                    **candidate_core(candidate),
                    "h1": _aggregate_metrics(trials, "h1"),
                    "h4": _aggregate_metrics(trials, "h4"),
                    "generated": _mean(
                        row["outcome"]["low_level"].get("generated", 0) for row in trials
                    ),
                    "trial_outcomes": [row["outcome"] for row in trials],
                }
            )
        h1_pareto, h4_pareto = pareto_ids(candidates, "h1"), pareto_ids(candidates, "h4")
        h4_compute_pareto = pareto_ids(candidates, "h4", compute_aware=True)
        h1_best, h4_best = best_ids(candidates, "h1"), best_ids(candidates, "h4")
        for candidate in candidates:
            candidate["labels"] = {
                "h1_effectiveness_pareto": candidate["candidate_id"] in h1_pareto,
                "h4_effectiveness_pareto": candidate["candidate_id"] in h4_pareto,
                "h1_best": candidate["candidate_id"] in h1_best,
                "h4_best": candidate["candidate_id"] in h4_best,
            }
        index.append(
            {
                "schema": INDEX_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "state_id": state["state_id"],
                "map_id": state["map_id"],
                "task_id": state["task_id"],
                "layout_mode": state["layout_mode"],
                "task_variant": state["task_variant"],
                "agent_count": state["agent_count"],
                "solver_seed": state["solver_seed"],
                "stage": state["stage"],
                "conflict_severity": state["conflict_severity"],
                "source_selected_candidate_id": state["source_selected_candidate_id"],
                "candidate_count": len(candidates),
                "candidates": candidates,
                "h1_pareto_ids": sorted(h1_pareto),
                "h4_pareto_ids": sorted(h4_pareto),
                "h4_compute_aware_pareto_ids": sorted(h4_compute_pareto),
                "h1_best_ids": sorted(h1_best),
                "h4_best_ids": sorted(h4_best),
            }
        )
    return index


def _half_candidates(state: dict[str, Any], trial_indices: tuple[int, int], horizon: str) -> list[dict[str, Any]]:
    rows = []
    for candidate in state["candidates"]:
        trials = [
            {"outcome": candidate["trial_outcomes"][index]}
            for index in trial_indices
        ]
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                horizon: _aggregate_metrics(trials, horizon),
            }
        )
    return rows


def _map_bootstrap(values: dict[str, list[float]], samples: int, seed: int) -> dict[str, Any]:
    maps = sorted(values)
    map_means = {key: _mean(values[key]) for key in maps}
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(maps) for _ in maps]
        estimates.append(_mean(map_means[key] for key in selected))
    return {
        "unit": "map_id",
        "samples": samples,
        "map_count": len(maps),
        "mean": _mean(map_means.values()),
        "ci95": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
        "by_map": dict(sorted(map_means.items())),
    }


def analyze_index(index: list[dict[str, Any]], config: dict[str, Any], *, formal: bool) -> dict[str, Any]:
    stability = {"spearman": [], "pareto_jaccard": [], "best_jaccard": []}
    h1_h4_pareto, changed_best = [], []
    effectiveness_compute_jaccard = []
    opportunities = []
    opportunities_by_map: dict[str, list[float]] = collections.defaultdict(list)
    maps_non_worse = set()
    errors = []
    for state in index:
        first = _half_candidates(state, (0, 1), "h4")
        second = _half_candidates(state, (2, 3), "h4")
        stability["spearman"].append(_spearman_effectiveness(first, second))
        stability["pareto_jaccard"].append(
            _jaccard(pareto_ids(first, "h4"), pareto_ids(second, "h4"))
        )
        stability["best_jaccard"].append(
            _jaccard(best_ids(first, "h4"), best_ids(second, "h4"))
        )
        h1_h4_pareto.append(_jaccard(state["h1_pareto_ids"], state["h4_pareto_ids"]))
        effectiveness_compute_jaccard.append(
            _jaccard(state["h4_pareto_ids"], state["h4_compute_aware_pareto_ids"])
        )
        changed_best.append(set(state["h1_best_ids"]) != set(state["h4_best_ids"]))
        selected_id = str(state["source_selected_candidate_id"])
        selected = next(
            (row for row in state["candidates"] if str(row["candidate_id"]) == selected_id),
            None,
        )
        if selected is None:
            errors.append({"state_id": state["state_id"], "error": "source selection missing"})
            continue
        oracle_id = sorted(
            state["candidates"],
            key=lambda row: (
                float(row["h4"]["conflict_auc"]),
                -float(row["h4"]["feasible_rate"]),
                float(row["h4"]["final_conflicts"]),
                str(row["candidate_id"]),
            ),
        )[0]["candidate_id"]
        oracle = next(row for row in state["candidates"] if row["candidate_id"] == oracle_id)
        baseline_auc = float(selected["h4"]["conflict_auc"])
        oracle_auc = float(oracle["h4"]["conflict_auc"])
        improvement = (baseline_auc - oracle_auc) / baseline_auc if baseline_auc > 0.0 else 0.0
        opportunities.append(improvement)
        opportunities_by_map[str(state["map_id"])].append(improvement)
    for map_id, values in opportunities_by_map.items():
        if _mean(values) >= 0.0:
            maps_non_worse.add(map_id)
    bootstrap = _map_bootstrap(
        opportunities_by_map,
        int(config["bootstrap_samples"]),
        int(config["bootstrap_seed"]),
    )
    metrics = {
        "state_count": len(index),
        "candidate_count": sum(len(row["candidates"]) for row in index),
        "trial_count": sum(
            len(candidate["trial_outcomes"])
            for row in index
            for candidate in row["candidates"]
        ),
        "integrity_error_count": len(errors),
        "split_half": {
            "mean_spearman": _mean(stability["spearman"]),
            "mean_pareto_jaccard": _mean(stability["pareto_jaccard"]),
            "mean_best_jaccard": _mean(stability["best_jaccard"]),
        },
        "long_term_difference": {
            "mean_h1_h4_pareto_jaccard": _mean(h1_h4_pareto),
            "changed_best_fraction": _mean(changed_best),
        },
        "compute_aware_sensitivity": {
            "mean_h4_effectiveness_compute_pareto_jaccard": _mean(
                effectiveness_compute_jaccard
            )
        },
        "long_term_opportunity": {
            "mean_oracle_auc_improvement": _mean(opportunities),
            "positive_state_fraction": _mean(value > 0.0 for value in opportunities),
            "maps_non_worse": len(maps_non_worse),
            "map_count": len(opportunities_by_map),
            "bootstrap": bootstrap,
        },
    }
    thresholds = dict(config["thresholds"])
    integrity_passed = (
        not errors
        and (not formal or len(index) == int(config["selection"]["total_states"]))
        and all(len(row["candidates"]) <= 18 for row in index)
        and all(len(candidate["trial_outcomes"]) == 4 for row in index for candidate in row["candidates"])
    )
    stability_passed = (
        metrics["split_half"]["mean_spearman"] >= float(thresholds["minimum_split_half_spearman"])
        and metrics["split_half"]["mean_pareto_jaccard"]
        >= float(thresholds["minimum_split_half_pareto_jaccard"])
        and metrics["split_half"]["mean_best_jaccard"]
        >= float(thresholds["minimum_split_half_best_jaccard"])
    )
    difference_passed = (
        metrics["long_term_difference"]["mean_h1_h4_pareto_jaccard"]
        <= float(thresholds["maximum_h1_h4_pareto_jaccard"])
        and metrics["long_term_difference"]["changed_best_fraction"]
        >= float(thresholds["minimum_changed_best_fraction"])
    )
    opportunity_passed = (
        metrics["long_term_opportunity"]["mean_oracle_auc_improvement"]
        >= float(thresholds["minimum_oracle_auc_improvement"])
        and metrics["long_term_opportunity"]["positive_state_fraction"]
        >= float(thresholds["minimum_positive_opportunity_fraction"])
        and metrics["long_term_opportunity"]["maps_non_worse"]
        >= int(thresholds["minimum_maps_non_worse"])
        and bootstrap["ci95"][0] >= float(thresholds["bootstrap_lower_bound"])
    )
    gates = {
        "integrity": integrity_passed,
        "h4_label_stability": stability_passed,
        "h1_h4_long_term_difference": difference_passed,
        "h4_oracle_opportunity": opportunity_passed,
    }
    if not integrity_passed:
        decision = "stop_integrity_failure"
    elif not stability_passed:
        decision = "stop_h4_labels_unstable"
    elif not difference_passed:
        decision = "reject_one_step_credit_mismatch_hypothesis"
    elif not opportunity_passed:
        decision = "frozen_v1_near_candidate_h4_limit"
    else:
        decision = "advance_to_fixed_long_term_value_ranker"
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "formal": formal,
        "passed": all(gates.values()),
        "decision": decision,
        "gates": gates,
        "thresholds": thresholds,
        "metrics": metrics,
        "errors": errors,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    lines = [
        "# InitLNS Policy-Visited Sequential Credit Audit",
        "",
        f"- Formal: `{report['formal']}`",
        f"- Passed: `{report['passed']}`",
        f"- Decision: `{report['decision']}`",
        f"- States/candidates/trials: {metrics['state_count']} / {metrics['candidate_count']} / {metrics['trial_count']}",
        "",
        "## Stability",
        "",
        f"- Split-half Spearman: {metrics['split_half']['mean_spearman']:.6f}",
        f"- Split-half Pareto Jaccard: {metrics['split_half']['mean_pareto_jaccard']:.6f}",
        f"- Split-half best-set Jaccard: {metrics['split_half']['mean_best_jaccard']:.6f}",
        "",
        "## Sequential Credit",
        "",
        f"- H1/H4 Pareto Jaccard: {metrics['long_term_difference']['mean_h1_h4_pareto_jaccard']:.6f}",
        f"- Changed best-set fraction: {metrics['long_term_difference']['changed_best_fraction']:.6f}",
        f"- H4 oracle AUC improvement: {metrics['long_term_opportunity']['mean_oracle_auc_improvement']:.6f}",
        f"- Positive opportunity fraction: {metrics['long_term_opportunity']['positive_state_fraction']:.6f}",
        f"- Maps non-worse: {metrics['long_term_opportunity']['maps_non_worse']}/{metrics['long_term_opportunity']['map_count']}",
        f"- Map bootstrap 95% CI: {metrics['long_term_opportunity']['bootstrap']['ci95']}",
        f"- Effectiveness/compute-aware Pareto Jaccard: {metrics['compute_aware_sensitivity']['mean_h4_effectiveness_compute_pareto_jaccard']:.6f}",
        "",
        "## Gates",
        "",
    ]
    lines.extend(f"- {name}: `{passed}`" for name, passed in report["gates"].items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sequential_credit_audit(
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
    workers: int | None = None,
    resume: bool = False,
    smoke_states: int | None = None,
) -> dict[str, Any]:
    if phase not in {"diagnose", "collect", "analyze", "all", "dry-run"}:
        raise ValueError(f"unsupported phase: {phase}")
    config = _read_json(_resolve(config_path))
    _validate_config(config)
    if workers is not None and int(workers) != 4:
        raise ValueError("sequential-credit collection requires workers=4")
    worker_count = int(config["workers"])
    if smoke_states is not None and smoke_states <= 0:
        raise ValueError("smoke state count must be positive")
    output_root = _resolve(output)
    output_root.mkdir(parents=True, exist_ok=True)
    diagnosis = diagnose_legacy(config)
    if not diagnosis["passed"]:
        raise RuntimeError("legacy H1/H4 diagnosis did not reproduce")
    _write_json(output_root / "legacy_diagnosis.json", diagnosis)
    if phase == "diagnose":
        return diagnosis
    registered = diagnosis["registered_inputs"]
    source_states, source_candidates = _load_source(config, registered)
    selected = select_audit_states(source_candidates, config)
    full_selection = selected
    if smoke_states is not None:
        selected = selected[:smoke_states]
    run_fingerprint, run_config = _run_metadata(config, registered, selected)
    reference_trials = _read_jsonl(Path(registered["paths"]["source_evaluation_trials"]))
    reference_runtimes = [
        float(row["outcome"]["runtime"])
        for row in reference_trials
        if row.get("outcome") and row["outcome"].get("runtime") is not None
    ]
    if len(reference_runtimes) != 31656 or min(reference_runtimes, default=0.0) <= 0.0:
        raise ValueError("registered Horizon-1 runtime reference is incomplete")
    dry_run = build_dry_run(
        config,
        full_selection,
        smoke_states=smoke_states,
        observed_repair_seconds_min=min(reference_runtimes),
    )
    _write_json(output_root / "dry_run.json", dry_run)
    _write_jsonl(
        output_root / "selected_states.jsonl",
        (
            {
                **row,
                "source_schema": row.get("schema"),
                "schema": STATE_SCHEMA,
                "schema_version": SCHEMA_VERSION,
            }
            for row in selected
        ),
    )
    _write_jsonl(
        output_root / "selected_candidates.jsonl",
        (
            {
                "schema": INDEX_SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "state_id": row["state_id"],
                "map_id": row["map_id"],
                "task_id": row["task_id"],
                "candidate": candidate_core(candidate),
            }
            for row in selected
            for candidate in row["candidates"]
        ),
    )
    if phase == "dry-run":
        return dry_run
    existing_config_path = output_root / "run_config.json"
    if existing_config_path.is_file():
        existing = _read_json(existing_config_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("output run fingerprint differs; refusing incompatible resume")
    _write_json(existing_config_path, run_config)
    formal = smoke_states is None
    if phase in {"collect", "all"}:
        dataset_lookup = _dataset_lookup(Path(registered["dataset"]))
        if any(str(row["task_id"]) not in dataset_lookup for row in selected):
            raise ValueError("selected state is missing from policy_train dataset")
        status = {
            "schema": AUDIT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "phase": "collect",
            "status": "running",
            "formal": formal,
            "state_count": len(selected),
        }
        _write_json(output_root / "run_status.json", status)
        try:
            with _CollectionRunLock(output_root, run_fingerprint, "sequential-credit"):
                prepared = _prepare_states(
                    output_root,
                    selected,
                    dataset_lookup,
                    config,
                    run_fingerprint,
                    resume=resume,
                    workers=worker_count,
                )
                if len(prepared) != len(selected) or any(
                    not bool(row.get("complete")) for row in prepared
                ):
                    raise RuntimeError("state preparation failed; no labels generated")
                trials = _collect_trials(
                    output_root,
                    selected,
                    dataset_lookup,
                    config,
                    run_fingerprint,
                    resume=resume,
                    workers=worker_count,
                )
                index = aggregate_trials(selected, trials, int(config["evaluation_trials"]))
                _write_jsonl(output_root / "sequential_credit_index.jsonl", index)
            status.update(
                {
                    "status": "complete",
                    "prepared_state_count": len(prepared),
                    "trial_count": len(trials),
                    "index_state_count": len(index),
                }
            )
            _write_json(output_root / "run_status.json", status)
            _write_json(output_root / "COLLECTION_COMPLETE.json", status)
        except BaseException as error:
            status.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
            _write_json(output_root / "run_status.json", status)
            raise
        if phase == "collect":
            return status
    index_path = output_root / "sequential_credit_index.jsonl"
    if not index_path.is_file():
        raise ValueError("analysis requires a complete sequential-credit index")
    index = _read_jsonl(index_path)
    report = analyze_index(index, config, formal=formal)
    report.update(
        {
            "run_fingerprint": run_fingerprint,
            "legacy_diagnosis": {
                "states": diagnosis["states"],
                "outcomes": diagnosis["outcomes"],
                "h1_h4_pareto_jaccard": diagnosis["h1_h4_pareto_jaccard"],
            },
        }
    )
    _write_json(output_root / "sequential_credit_audit.json", report)
    _write_markdown(output_root / "sequential_credit_audit.md", report)
    _write_json(
        output_root / "ANALYSIS_COMPLETE.json",
        {
            "schema": AUDIT_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "status": "complete",
            "passed": report["passed"],
            "decision": report["decision"],
        },
    )
    return report


__all__ = [
    "AUDIT_SCHEMA",
    "CollectionLockError",
    "aggregate_trials",
    "analyze_index",
    "best_ids",
    "build_dry_run",
    "conflict_severity",
    "diagnose_legacy",
    "execute_horizon4_trial",
    "pareto_ids",
    "run_sequential_credit_audit",
    "select_audit_states",
]
