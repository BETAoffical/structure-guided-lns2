from __future__ import annotations

import collections
import concurrent.futures
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable

from experiments.closed_loop_confirmation import (
    generate_online_candidates,
    score_online_candidates,
)
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_aware import adaptive_feature_row
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _low_level_delta,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.trace_replay import decision_rows, replay_prefix


HIGH_LOAD_RESCUE_SCHEMA = "lns2.high_load_rescue_collection.v1"


def is_no_progress_decision(row: dict[str, Any]) -> bool:
    metrics = dict(row.get("actual_metrics") or {})
    return not bool(metrics.get("replan_success")) or not bool(
        row.get("repair_state_changed")
    )


def _round_robin_maps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["map_id"])].append(row)
    for values in grouped.values():
        values.sort(key=lambda value: (str(value["task_id"]), int(value["decision_index"])))
    result: list[dict[str, Any]] = []
    while any(grouped.values()):
        for map_id in sorted(grouped):
            if grouped[map_id]:
                result.append(grouped[map_id].pop(0))
    return result


def select_failure_decisions(
    source_root: str | Path, *, maximum: int
) -> list[dict[str, Any]]:
    if maximum <= 0:
        raise ValueError("maximum failure-state count must be positive")
    root = Path(source_root).resolve()
    run = _read_json(root / "run_config.json")
    if str(run.get("controller")) != "v2-full":
        raise ValueError("high-load rescue source must use v2-full")
    split = str(dict(run["configuration"])["split"])
    manifests = _read_jsonl(root / "realized_dynamic_manifest.jsonl")
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for manifest in manifests:
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            continue
        decisions, _events = decision_rows(root, manifest)
        for decision in decisions:
            fingerprint = str(decision["before_repair_fingerprint"])
            if fingerprint in seen or not is_no_progress_decision(decision):
                continue
            seen.add(fingerprint)
            selected.append(
                {
                    **decision,
                    "split": split,
                    "source_root": str(root),
                    "source_run_fingerprint": str(run["run_fingerprint"]),
                    "map_id": str(manifest["map_id"]),
                    "layout_mode": str(manifest.get("layout_mode", "unknown")),
                    "task_id": str(manifest["task_id"]),
                    "solver_seed": int(manifest["solver_seed"]),
                    "episode_id": str(manifest["episode_id"]),
                }
            )
    return _round_robin_maps(selected)[:maximum]


def paired_rescue_seed(repair_fingerprint: str, trial_index: int) -> int:
    return int(
        _fingerprint(
            {
                "namespace": "high-load-rescue-paired-v1",
                "repair_fingerprint": str(repair_fingerprint),
                "trial_index": int(trial_index),
            }
        )[:16],
        16,
    ) % (2**31)


def _feature_payload(row: dict[str, Any]) -> dict[str, float]:
    if "feature_values" in row:
        return dict(
            zip(
                map(str, row["feature_names"]),
                map(float, row["feature_values"]),
            )
        )
    return {
        str(name): float(value)
        for name, value in dict(row["features"]["realized_dynamic"]).items()
    }


def _trial(
    job: dict[str, Any],
    decision: dict[str, Any],
    candidate: dict[str, Any],
    trial_index: int,
) -> dict[str, Any]:
    environment, before = replay_prefix(job, decision["prefix_actions"])
    if str(decision["before_fingerprint"]) != state_fingerprint(before):
        raise RuntimeError("high-load rescue replay fingerprint mismatch")
    random_seed = paired_rescue_seed(
        str(decision["before_repair_fingerprint"]), trial_index
    )
    if str(candidate["route"]) == "official_adaptive":
        action = {"mode": "official", "random_seed": random_seed}
    else:
        action = {
            "mode": "explicit_neighborhood",
            "agents": list(map(int, candidate["agents"])),
            "random_seed": random_seed,
        }
    started = time.perf_counter()
    result = _plain(environment.step(action))
    repair_wall_seconds = time.perf_counter() - started
    after = dict(result["observation"])
    metrics = dict(result["metrics"])
    before_conflicts = int(before["num_of_colliding_pairs"])
    after_conflicts = int(after["num_of_colliding_pairs"])
    low_level = _low_level_delta(before, after)
    return {
        "schema": HIGH_LOAD_RESCUE_SCHEMA,
        "split": str(decision["split"]),
        "state_id": str(decision["state_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "trial_index": int(trial_index),
        "random_seed": random_seed,
        "complete": True,
        "status": "ok",
        "outcome": {
            "conflicts_before": before_conflicts,
            "conflicts_after": after_conflicts,
            "conflict_reduction": max(0, before_conflicts - after_conflicts),
            "replan_success": bool(metrics.get("replan_success")),
            "hard_failure": not bool(metrics.get("replan_success")),
            "feasible": bool(after.get("feasible")),
            "repair_seconds": repair_wall_seconds,
            "pp_replan_seconds": float(metrics.get("pp_replan_seconds", 0.0)),
            "generated": int(low_level.get("generated", 0)),
            "expanded": int(low_level.get("expanded", 0)),
            "reopened": int(low_level.get("reopened", 0)),
        },
    }


def _state_job(job: dict[str, Any]) -> dict[str, Any]:
    decision = dict(job["decision"])
    output = Path(job["state_file"])
    if bool(job["resume"]) and output.is_file():
        existing = _read_json(output)
        if (
            str(existing.get("run_fingerprint")) == str(job["run_fingerprint"])
            and bool(existing.get("complete"))
        ):
            return {"state_file": str(output), "status": "resumed"}
    source_run = _read_json(Path(decision["source_root"]) / "run_config.json")
    configuration = dict(source_run["configuration"])
    dataset_root = Path(str(source_run["dataset"])).resolve()
    row_lookup = {
        str(row["task_id"]): row
        for row in _load_dataset_rows(dataset_root, [str(decision["split"])])
    }
    replay_job = {
        "dataset_root": str(dataset_root),
        "row": row_lookup[str(decision["task_id"])],
        "environment": dict(configuration["environment"]),
        "solver_seed": int(decision["solver_seed"]),
    }
    environment, state = replay_prefix(replay_job, decision["prefix_actions"])
    if state_fingerprint(state) != str(decision["before_fingerprint"]):
        raise RuntimeError("high-load state replay did not reproduce its source")

    bundle = load_controller_bundle(Path(job["controller_bundle"]))
    main_model = bundle.main_models["realized_dynamic"]
    proposal = dict(configuration["proposal"])
    proposal["neighborhood_sizes"] = list(map(int, job["neighborhood_sizes"]))
    candidates, generation = generate_online_candidates(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal_config=proposal,
        state_hash=str(decision["before_fingerprint"]),
        verify_full_state=True,
        proposal_backend=str(job["controller_runtime"]),
        shadow_validation=False,
    )
    engine = OnlineFeatureEngine(
        state,
        backend=str(job["feature_backend"]),
        required_features={
            "realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]
        },
        dense_output=False,
    )
    feature_rows, feature_metrics = engine.realized_rows(
        candidates, state_hash=str(decision["before_fingerprint"])
    )
    _selected, scores, _margin = score_online_candidates(feature_rows, main_model)
    source_agents = tuple(sorted(map(int, decision["actual_action"].get("agents", ()))))
    explicit_rows: list[dict[str, Any]] = []
    explicit_candidates: list[dict[str, Any]] = []
    for candidate, feature_row, score in zip(candidates, feature_rows, scores):
        candidate_id = str(candidate["candidate_id"])
        route_row = {
            **candidate,
            "candidate_id": candidate_id,
            "route": "model",
        }
        explicit_candidates.append(route_row)
        explicit_rows.append(
            {
                "schema": HIGH_LOAD_RESCUE_SCHEMA,
                "split": str(decision["split"]),
                "state_id": str(decision["state_id"]),
                "candidate_id": candidate_id,
                "candidate_key": str(feature_row["candidate_key"]),
                "map_id": str(decision["map_id"]),
                "layout_mode": str(decision.get("layout_mode", "unknown")),
                "task_id": str(decision["task_id"]),
                "agent_count": len(state["agents"]),
                "actual_size": int(candidate["actual_size"]),
                "route": "model",
                "base_selected": tuple(sorted(map(int, candidate["agents"])))
                == source_agents,
                "main_score": float(score),
                "features": {
                    "realized_dynamic": _feature_payload(feature_row)
                },
            }
        )
    base_count = sum(bool(row["base_selected"]) for row in explicit_rows)
    if base_count != 1:
        raise RuntimeError(
            f"source v2 action does not match exactly one regenerated candidate: {base_count}"
        )
    adaptive_row = adaptive_feature_row(feature_rows[0])
    adaptive_candidate = {
        "candidate_id": "official_adaptive",
        "route": "official_adaptive",
        "agents": [],
        "actual_size": 0,
    }
    explicit_rows.append(
        {
            "schema": HIGH_LOAD_RESCUE_SCHEMA,
            "split": str(decision["split"]),
            "state_id": str(decision["state_id"]),
            "candidate_id": "official_adaptive",
            "candidate_key": "official_adaptive",
            "map_id": str(decision["map_id"]),
            "layout_mode": str(decision.get("layout_mode", "unknown")),
            "task_id": str(decision["task_id"]),
            "agent_count": len(state["agents"]),
            "actual_size": 0,
            "route": "official_adaptive",
            "base_selected": False,
            "main_score": -1e30,
            "features": {
                "realized_dynamic": _feature_payload(adaptive_row)
            },
        }
    )
    all_candidates = explicit_candidates + [adaptive_candidate]
    trials: list[dict[str, Any]] = []
    initial_trials = int(job["initial_trials"])
    maximum_trials = int(job["maximum_trials"])
    for candidate in all_candidates:
        for trial_index in range(initial_trials):
            trials.append(_trial(replay_job, decision, candidate, trial_index))

    if maximum_trials > initial_trials:
        by_candidate: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for trial in trials:
            by_candidate[str(trial["candidate_id"])].append(trial)
        ranked = sorted(
            all_candidates,
            key=lambda candidate: -statistics.fmean(
                float(row["outcome"]["conflict_reduction"])
                / max(1e-9, float(row["outcome"]["repair_seconds"]))
                for row in by_candidate[str(candidate["candidate_id"])]
            ),
        )
        top = ranked[:4]
        if not any(str(row["candidate_id"]) == "official_adaptive" for row in top):
            top.append(adaptive_candidate)
        top_values = [
            statistics.fmean(
                float(row["outcome"]["conflict_reduction"])
                / max(1e-9, float(row["outcome"]["repair_seconds"]))
                for row in by_candidate[str(candidate["candidate_id"])]
            )
            for candidate in ranked[:2]
        ]
        ambiguous = len(top_values) < 2 or abs(top_values[0] - top_values[1]) <= 0.10 * max(
            1e-9, abs(top_values[0])
        )
        if ambiguous:
            for candidate in top:
                for trial_index in range(initial_trials, maximum_trials):
                    trials.append(_trial(replay_job, decision, candidate, trial_index))

    payload = {
        "schema": HIGH_LOAD_RESCUE_SCHEMA,
        "run_fingerprint": str(job["run_fingerprint"]),
        "complete": True,
        "state": {
            "split": str(decision["split"]),
            "state_id": str(decision["state_id"]),
            "map_id": str(decision["map_id"]),
            "task_id": str(decision["task_id"]),
            "solver_seed": int(decision["solver_seed"]),
            "decision_index": int(decision["decision_index"]),
            "before_fingerprint": str(decision["before_fingerprint"]),
            "before_repair_fingerprint": str(decision["before_repair_fingerprint"]),
        },
        "generation": generation,
        "feature_metrics": feature_metrics,
        "candidates": explicit_rows,
        "trials": trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output)
    return {"state_file": str(output), "status": "ok"}


def _candidate_actuals(
    feature_rows: list[dict[str, Any]], trial_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    features = {
        (str(row["state_id"]), str(row["candidate_id"])): row
        for row in feature_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trial_rows:
        grouped[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    result = []
    for key, rows in grouped.items():
        feature = features[key]
        outcomes = [dict(row["outcome"]) for row in rows]
        result.append(
            {
                "state_id": key[0],
                "candidate_id": key[1],
                "actual_size": int(feature["actual_size"]),
                "route": str(feature["route"]),
                "progress_rate": statistics.fmean(
                    float(row["conflict_reduction"] > 0) for row in outcomes
                ),
                "mean_reduction": statistics.fmean(
                    float(row["conflict_reduction"]) for row in outcomes
                ),
                "mean_repair_seconds": statistics.fmean(
                    float(row["repair_seconds"]) for row in outcomes
                ),
            }
        )
    return result


def size12_pilot_gate(
    feature_rows: list[dict[str, Any]], trial_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    actuals = _candidate_actuals(feature_rows, trial_rows)
    by_state: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in actuals:
        by_state[str(row["state_id"])].append(row)
    pareto_states = 0
    selected_states = 0
    for rows in by_state.values():
        size12 = [row for row in rows if int(row["actual_size"]) == 12]
        if not size12:
            continue
        if any(
            not any(
                other is not candidate
                and float(other["progress_rate"]) >= float(candidate["progress_rate"])
                and float(other["mean_reduction"]) >= float(candidate["mean_reduction"])
                and float(other["mean_repair_seconds"]) <= float(candidate["mean_repair_seconds"])
                and (
                    float(other["progress_rate"]) > float(candidate["progress_rate"])
                    or float(other["mean_reduction"]) > float(candidate["mean_reduction"])
                    or float(other["mean_repair_seconds"]) < float(candidate["mean_repair_seconds"])
                )
                for other in rows
            )
            for candidate in size12
        ):
            pareto_states += 1
        winner = max(
            rows,
            key=lambda row: (
                float(row["mean_reduction"])
                / max(1e-9, float(row["mean_repair_seconds"])),
                float(row["progress_rate"]),
                -float(row["mean_repair_seconds"]),
            ),
        )
        selected_states += int(int(winner["actual_size"]) == 12)
    return {
        "schema": "lns2.size12_pilot_gate.v1",
        "state_count": len(by_state),
        "size12_pareto_state_count": pareto_states,
        "size12_selected_state_count": selected_states,
        "minimum_required_state_count": 60,
        "minimum_selected_state_count": 3,
        "passed": len(by_state) >= 60 and pareto_states > 0 and selected_states >= 3,
    }


def collect_high_load_rescue_data(
    *,
    source_roots: dict[str, str | Path],
    output: str | Path,
    controller_bundle: str | Path,
    maximum_states: dict[str, int],
    neighborhood_sizes: Iterable[int] = (4, 8, 12, 16),
    initial_trials: int = 2,
    maximum_trials: int = 2,
    workers: int = 4,
    feature_backend: str = "native",
    controller_runtime: str = "optimized",
    resume: bool = False,
) -> dict[str, Any]:
    output_root = Path(output).resolve()
    sources = {str(split): str(Path(path).resolve()) for split, path in source_roots.items()}
    sizes = tuple(map(int, neighborhood_sizes))
    if set(sources) != {"policy_train", "policy_validation"}:
        raise ValueError("high-load collection requires train and validation sources")
    if (
        not sizes
        or tuple(sorted(set(sizes))) != sizes
        or any(value <= 0 for value in sizes)
    ):
        raise ValueError("neighborhood sizes must be sorted unique positive integers")
    if initial_trials <= 0 or maximum_trials < initial_trials:
        raise ValueError("invalid high-load rescue trial counts")
    identity = {
        "schema": HIGH_LOAD_RESCUE_SCHEMA,
        "sources": sources,
        "source_fingerprints": {
            split: _read_json(Path(path) / "run_config.json")["run_fingerprint"]
            for split, path in sources.items()
        },
        "controller_bundle": str(Path(controller_bundle).resolve()),
        "maximum_states": dict(maximum_states),
        "neighborhood_sizes": list(sizes),
        "initial_trials": int(initial_trials),
        "maximum_trials": int(maximum_trials),
        "feature_backend": feature_backend,
        "controller_runtime": controller_runtime,
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        previous = _read_json(run_path)
        if str(previous.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("output contains a different high-load rescue collection")
        if not resume:
            raise ValueError("output already exists; pass --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})

    decisions: list[dict[str, Any]] = []
    for split, root in sources.items():
        selected = select_failure_decisions(
            root, maximum=int(maximum_states[split])
        )
        for index, decision in enumerate(selected):
            decision["state_id"] = (
                f"{split}__{decision['episode_id']}__failure_{index:04d}"
            )
        decisions.extend(selected)
    jobs = []
    for decision in decisions:
        state_key = _fingerprint(
            {
                "split": decision["split"],
                "state_id": decision["state_id"],
                "fingerprint": decision["before_repair_fingerprint"],
            }
        )[:20]
        jobs.append(
            {
                "decision": decision,
                "state_file": str(
                    output_root
                    / "states"
                    / str(decision["split"])
                    / f"{state_key}.json"
                ),
                "controller_bundle": str(Path(controller_bundle).resolve()),
                "neighborhood_sizes": list(sizes),
                "initial_trials": int(initial_trials),
                "maximum_trials": int(maximum_trials),
                "feature_backend": feature_backend,
                "controller_runtime": controller_runtime,
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_state_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "state_id": job["decision"]["state_id"],
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            _write_json(
                output_root / "status.json",
                {
                    "schema": HIGH_LOAD_RESCUE_SCHEMA,
                    "status": "running",
                    "completed_states": len(completed),
                    "total_states": len(jobs),
                    "error_states": len(errors),
                },
            )
    feature_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for row in completed:
        payload = _read_json(Path(row["state_file"]))
        feature_rows.extend(payload["candidates"])
        trial_rows.extend(payload["trials"])
    completed_states_by_split = dict(
        collections.Counter(
            str(_read_json(Path(row["state_file"]))["state"]["split"])
            for row in completed
        )
    )
    feature_rows.sort(key=lambda row: (str(row["split"]), str(row["state_id"]), str(row["candidate_id"])))
    trial_rows.sort(key=lambda row: (str(row["split"]), str(row["state_id"]), str(row["candidate_id"]), int(row["trial_index"])))
    _write_jsonl(output_root / "feature_index.jsonl", feature_rows)
    _write_jsonl(output_root / "trial_manifest.jsonl", trial_rows)
    gate = size12_pilot_gate(feature_rows, trial_rows)
    _write_json(output_root / "size12_pilot_gate.json", gate)
    report = {
        "schema": HIGH_LOAD_RESCUE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(jobs),
        "completed_state_count": len(completed),
        "error_state_count": len(errors),
        "errors": errors,
        "candidate_count": len(feature_rows),
        "trial_count": len(trial_rows),
        "state_count_by_split": completed_states_by_split,
        "candidate_count_by_split": dict(
            collections.Counter(str(row["split"]) for row in feature_rows)
        ),
        "size12_pilot_gate": gate,
        "complete": not errors and len(completed) == len(jobs),
    }
    _write_json(output_root / "collection_report.json", report)
    _write_json(
        output_root / "status.json",
        {
            "schema": HIGH_LOAD_RESCUE_SCHEMA,
            "status": "complete" if report["complete"] else "error",
            "completed_states": len(completed),
            "total_states": len(jobs),
            "error_states": len(errors),
        },
    )
    return report


__all__ = [
    "HIGH_LOAD_RESCUE_SCHEMA",
    "collect_high_load_rescue_data",
    "is_no_progress_decision",
    "paired_rescue_seed",
    "select_failure_decisions",
    "size12_pilot_gate",
]
