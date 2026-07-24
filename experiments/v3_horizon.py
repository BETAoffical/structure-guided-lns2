from __future__ import annotations

import collections
import concurrent.futures
import os
import shutil
import time
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import (
    generate_online_candidates,
    repair_random_seed,
    score_online_candidates,
)
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.high_load_rescue import paired_rescue_seed
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.parallel_runtime import (
    initialize_isolated_worker,
    isolated_lane_cpu_sets,
    parallel_runtime_metadata,
)
from experiments.repair_aware import classify_repair_outcome
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
from experiments.stall_guard import repair_structure_fingerprint
from experiments.trace_replay import TRACE_REPLAY_CONTRACT, replay_prefix
from experiments.v3_controller import load_v3_controller_bundle, v3_candidate_order


V3_HORIZON_COLLECTION_SCHEMA = "lns2.v3_horizon_collection.v1"


def _horizon_environment_configuration(
    raw: dict[str, Any], *, prefix_length: int, horizon: int
) -> dict[str, Any]:
    configuration = dict(raw)
    configuration["max_repair_iterations"] = max(
        int(configuration.get("max_repair_iterations", 0)),
        prefix_length + horizon,
    )
    return configuration


def _candidate_rows(
    environment: Any,
    state: dict[str, Any],
    *,
    task_id: str,
    solver_seed: int,
    decision_index: int,
    proposal: dict[str, Any],
    main_model: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float]]:
    state_hash = state_fingerprint(state)
    candidates, _generation = generate_online_candidates(
        environment,
        state,
        task_id=task_id,
        solver_seed=solver_seed,
        decision_index=decision_index,
        proposal_config=proposal,
        state_hash=state_hash,
        verify_full_state=True,
        proposal_backend="optimized",
        shadow_validation=False,
    )
    engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features={"realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]},
        dense_output=False,
    )
    rows, _metrics = engine.realized_rows(candidates, state_hash=state_hash)
    _selected, scores, _margin = score_online_candidates(rows, main_model)
    return candidates, rows, scores


def select_horizon_candidates(
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    v2_scores: list[float],
    v3_bundle: Any,
) -> list[int]:
    selected: set[int] = set()
    by_size: dict[int, list[int]] = collections.defaultdict(list)
    for index, candidate in enumerate(candidates):
        by_size[int(candidate["actual_size"])].append(index)
    for size in (4, 8, 16):
        ranked = sorted(
            by_size.get(size, ()),
            key=lambda index: (-float(v2_scores[index]), str(candidates[index]["candidate_id"])),
        )
        selected.update(ranked[:2])
    selected.add(
        max(
            range(len(candidates)),
            key=lambda index: (
                float(v2_scores[index]), str(candidates[index]["candidate_id"])
            ),
        )
    )
    if bool(getattr(v3_bundle, "is_horizon", False)):
        raise ValueError("horizon candidate sampling requires the frozen one-step v3")
    predictions = v3_bundle.predict(rows)
    old_order = v3_candidate_order(
        candidates, predictions, v2_scores, v3_bundle.thresholds
    )
    if old_order:
        selected.add(old_order[0])
    return sorted(selected, key=lambda index: str(candidates[index]["candidate_id"]))


def _v2_action(
    environment: Any,
    state: dict[str, Any],
    *,
    task_id: str,
    solver_seed: int,
    decision_index: int,
    proposal: dict[str, Any],
    main_model: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    candidates, rows, scores = _candidate_rows(
        environment,
        state,
        task_id=task_id,
        solver_seed=solver_seed,
        decision_index=decision_index,
        proposal=proposal,
        main_model=main_model,
    )
    selected = max(
        range(len(candidates)),
        key=lambda index: (float(scores[index]), str(candidates[index]["candidate_id"])),
    )
    candidate = candidates[selected]
    state_hash = state_fingerprint(state)
    random_seed = repair_random_seed(
        task_id,
        solver_seed,
        state_hash,
        decision_index,
        str(candidate["candidate_id"]),
        candidate.get("proposal_seeds", ()),
    )
    return (
        {
            "mode": "explicit_neighborhood",
            "agents": list(map(int, candidate["agents"])),
            "random_seed": random_seed,
            "pp_random_seed": random_seed,
        },
        {
            "controller_seconds": time.perf_counter() - started,
            "candidate_id": str(candidate["candidate_id"]),
            "actual_size": int(candidate["actual_size"]),
        },
    )


def _horizon_trial(
    replay_job: dict[str, Any],
    decision: dict[str, Any],
    candidate: dict[str, Any],
    *,
    trial_index: int,
    horizon: int,
    first_selection_seconds: float,
    proposal: dict[str, Any],
    main_model: Any,
) -> dict[str, Any]:
    environment, state = replay_prefix(replay_job, decision["prefix_actions"])
    expected = str(decision["before_fingerprint"])
    if state_fingerprint(state) != expected:
        raise RuntimeError("v3-h3 replay fingerprint mismatch")
    initial_conflicts = int(state["num_of_colliding_pairs"])
    initial_repair_fingerprint = repair_structure_fingerprint(state)
    trajectory = [initial_conflicts]
    steps: list[dict[str, Any]] = []
    total_pp_seconds = 0.0
    total_repair_seconds = 0.0
    total_controller_seconds = max(0.0, float(first_selection_seconds))
    low_level_total = collections.Counter()
    first_outcome: dict[str, Any] | None = None
    for offset in range(horizon):
        before = state
        before_conflicts = int(before["num_of_colliding_pairs"])
        before_repair = repair_structure_fingerprint(before)
        if offset == 0:
            random_seed = paired_rescue_seed(
                str(decision["before_repair_fingerprint"]), trial_index
            )
            if str(candidate["route"]) == "official_adaptive":
                action = {
                    "mode": "official",
                    "random_seed": random_seed,
                    "pp_random_seed": random_seed,
                }
            else:
                action = {
                    "mode": "explicit_neighborhood",
                    "agents": list(map(int, candidate["agents"])),
                    "random_seed": random_seed,
                    "pp_random_seed": random_seed,
                }
            controller_row = {
                "controller_seconds": first_selection_seconds,
                "candidate_id": str(candidate["candidate_id"]),
                "actual_size": int(candidate.get("actual_size", 0)),
            }
        else:
            action, controller_row = _v2_action(
                environment,
                before,
                task_id=str(decision["task_id"]),
                solver_seed=int(decision["solver_seed"]),
                decision_index=int(decision["decision_index"]) + offset,
                proposal=proposal,
                main_model=main_model,
            )
            total_controller_seconds += float(controller_row["controller_seconds"])
        repair_started = time.perf_counter()
        result = _plain(environment.step(action))
        repair_seconds = time.perf_counter() - repair_started
        state = dict(result["observation"])
        metrics = dict(result["metrics"])
        after_conflicts = int(state["num_of_colliding_pairs"])
        after_repair = repair_structure_fingerprint(state)
        low_level = _low_level_delta(before, state)
        for name in ("generated", "expanded", "reopened"):
            low_level_total[name] += int(low_level.get(name, 0))
        pp_seconds = max(0.0, float(metrics.get("pp_replan_seconds", 0.0)))
        total_pp_seconds += pp_seconds
        total_repair_seconds += repair_seconds
        outcome_name = classify_repair_outcome(
            before_fingerprint=before_repair,
            after_fingerprint=after_repair,
            replan_success=bool(metrics.get("replan_success")),
            conflicts_before=before_conflicts,
            conflicts_after=after_conflicts,
            feasible=bool(state.get("feasible")),
        )
        step = {
            "step": offset + 1,
            "candidate_id": controller_row["candidate_id"],
            "actual_size": int(controller_row["actual_size"]),
            "action": action,
            "controller_seconds": float(controller_row["controller_seconds"]),
            "repair_seconds": repair_seconds,
            "pp_replan_seconds": pp_seconds,
            "conflicts_before": before_conflicts,
            "conflicts_after": after_conflicts,
            "repair_outcome": outcome_name,
            "before_repair_fingerprint": before_repair,
            "after_repair_fingerprint": after_repair,
        }
        steps.append(step)
        trajectory.append(after_conflicts)
        if first_outcome is None:
            first_outcome = dict(step)
        if bool(state.get("feasible")):
            break
    assert first_outcome is not None
    minimum_conflicts = min(trajectory)
    h3_no_progress = minimum_conflicts >= initial_conflicts
    return {
        "schema": V3_HORIZON_COLLECTION_SCHEMA,
        "split": str(decision["split"]),
        "state_id": str(decision["state_id"]),
        "candidate_id": str(candidate["candidate_id"]),
        "route": str(candidate["route"]),
        "actual_size": int(candidate.get("actual_size", 0)),
        "trial_index": int(trial_index),
        "status": "ok",
        "complete": True,
        "horizon": int(horizon),
        "executed_steps": len(steps),
        "initial_repair_fingerprint": initial_repair_fingerprint,
        "final_repair_fingerprint": repair_structure_fingerprint(state),
        "conflict_trajectory": trajectory,
        "steps": steps,
        "h1": {
            "effective_progress": first_outcome["repair_outcome"]
            in {"conflict_reduced", "feasible"},
            "no_progress": first_outcome["repair_outcome"]
            in {"hard_failure", "accepted_noop"},
            "conflict_reduction": max(
                0,
                int(first_outcome["conflicts_before"])
                - int(first_outcome["conflicts_after"]),
            ),
            "pp_replan_seconds": float(first_outcome["pp_replan_seconds"]),
        },
        "h3": {
            "conflict_reduction": max(0, initial_conflicts - int(trajectory[-1])),
            "best_conflict_reduction": max(0, initial_conflicts - minimum_conflicts),
            "no_progress": h3_no_progress,
            "feasible": bool(state.get("feasible")),
            "pp_replan_seconds": total_pp_seconds,
            "repair_seconds": total_repair_seconds,
            "controller_seconds": total_controller_seconds,
            "total_seconds": total_controller_seconds + total_repair_seconds,
            "generated": int(low_level_total["generated"]),
            "expanded": int(low_level_total["expanded"]),
            "reopened": int(low_level_total["reopened"]),
        },
    }


def _horizon_state_job(job: dict[str, Any]) -> dict[str, Any]:
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
    environment_configuration = _horizon_environment_configuration(
        dict(configuration["environment"]),
        prefix_length=len(decision["prefix_actions"]),
        horizon=int(job["horizon"]),
    )
    replay_job = {
        "dataset_root": str(dataset_root),
        "row": row_lookup[str(decision["task_id"])],
        "environment": environment_configuration,
        "solver_seed": int(decision["solver_seed"]),
    }
    environment, state = replay_prefix(replay_job, decision["prefix_actions"])
    if state_fingerprint(state) != str(decision["before_fingerprint"]):
        raise RuntimeError("v3-h3 state replay did not reproduce its source")
    main_bundle = load_controller_bundle(Path(job["controller_bundle"]))
    main_model = main_bundle.main_models["realized_dynamic"]
    old_v3 = load_v3_controller_bundle(Path(job["v3_bundle"]))
    proposal = dict(configuration["proposal"])
    proposal["neighborhood_sizes"] = [4, 8, 16]
    first_selection_started = time.perf_counter()
    candidates, rows, scores = _candidate_rows(
        environment,
        state,
        task_id=str(decision["task_id"]),
        solver_seed=int(decision["solver_seed"]),
        decision_index=int(decision["decision_index"]),
        proposal=proposal,
        main_model=main_model,
    )
    selected_indices = select_horizon_candidates(candidates, rows, scores, old_v3)
    first_model_selection_seconds = time.perf_counter() - first_selection_started
    selected_candidates = [
        {
            **candidates[index],
            "route": "model",
            "first_selection_seconds": first_model_selection_seconds,
        }
        for index in selected_indices
    ]
    selected_candidates.append(
        {
            "candidate_id": "official_adaptive",
            "route": "official_adaptive",
            "agents": [],
            "actual_size": 0,
            # Native Adaptive neighborhood generation occurs inside step() and
            # is already included in repair_seconds.  It must not inherit the
            # model controller's candidate/feature/inference time.
            "first_selection_seconds": 0.0,
        }
    )
    trials = [
        _horizon_trial(
            replay_job,
            decision,
            candidate,
            trial_index=trial_index,
            horizon=int(job["horizon"]),
            first_selection_seconds=float(candidate["first_selection_seconds"]),
            proposal=proposal,
            main_model=main_model,
        )
        for candidate in selected_candidates
        for trial_index in range(int(job["trials"]))
    ]
    payload = {
        "schema": V3_HORIZON_COLLECTION_SCHEMA,
        "run_fingerprint": str(job["run_fingerprint"]),
        "complete": True,
        "state_id": str(decision["state_id"]),
        "selected_candidate_ids": [
            str(candidate["candidate_id"]) for candidate in selected_candidates
        ],
        "trials": trials,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output)
    return {"state_file": str(output), "status": "ok"}


def _coverage(
    decisions: list[dict[str, Any]], rows: list[dict[str, Any]], trials: int
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    errors = []
    for decision in decisions:
        state_id = str(decision["state_id"])
        state_rows = grouped.get(state_id, [])
        by_candidate = collections.Counter(str(row["candidate_id"]) for row in state_rows)
        if "official_adaptive" not in by_candidate:
            errors.append(f"{state_id}: missing Adaptive")
        model_sizes = {
            int(row.get("actual_size", -1))
            for row in state_rows
            if str(row.get("candidate_id")) != "official_adaptive"
        }
        if not {4, 8, 16}.issubset(model_sizes):
            errors.append(f"{state_id}: incomplete 4/8/16 candidate coverage")
        if len(by_candidate) < 4:
            errors.append(f"{state_id}: fewer than three model arms plus Adaptive")
        for candidate_id, count in by_candidate.items():
            if count != trials:
                errors.append(f"{state_id}/{candidate_id}: trial_count={count}")
        for row in state_rows:
            if int(row["horizon"]) != 3 or not 1 <= int(row["executed_steps"]) <= 3:
                errors.append(f"{state_id}/{row['candidate_id']}: invalid horizon")
    return {
        "state_count": len(decisions),
        "covered_state_count": len(grouped),
        "trial_count": len(rows),
        "error_count": len(errors),
        "errors": errors,
        "passed": len(grouped) == len(decisions) and not errors,
    }


def _reuse_completed_states(
    source: Path,
    output: Path,
    jobs: list[dict[str, Any]],
    run_fingerprint: str,
    required_candidate_ids: dict[str, str] | None = None,
) -> int:
    source_config = _read_json(source / "run_config.json")
    source_fingerprint = str(source_config.get("run_fingerprint"))
    sampling_upgrade = source_fingerprint != run_fingerprint
    if sampling_upgrade:
        expected = "top2-v2-per-size-plus-v3-winner-and-adaptive"
        upgraded = "top2-v2-per-size-plus-v2-winner-v3-winner-and-adaptive"
        if str(source_config.get("candidate_sampling")) != expected:
            raise ValueError("reused v3-h3 collection belongs to a different run")
        comparable = {
            key: value
            for key, value in source_config.items()
            if key not in {"run_fingerprint", "candidate_sampling"}
        }
        current = _read_json(output / "run_config.json")
        current_comparable = {
            key: value
            for key, value in current.items()
            if key not in {"run_fingerprint", "candidate_sampling"}
        }
        if (
            str(current.get("candidate_sampling")) != upgraded
            or comparable != current_comparable
        ):
            raise ValueError("v3-h3 reuse differs beyond the sampling correction")
    reused = 0
    for job in jobs:
        target = Path(job["state_file"])
        relative = target.relative_to(output)
        previous = source / relative
        if not previous.is_file():
            continue
        payload = _read_json(previous)
        if (
            str(payload.get("run_fingerprint")) != source_fingerprint
            or not bool(payload.get("complete"))
        ):
            raise ValueError(f"invalid reusable v3-h3 state: {previous}")
        state_id = str(payload.get("state_id"))
        required = (required_candidate_ids or {}).get(state_id)
        if required is not None and required not in set(
            map(str, payload.get("selected_candidate_ids", ()))
        ):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if sampling_upgrade:
            _write_json(target, {**payload, "run_fingerprint": run_fingerprint})
        else:
            shutil.copy2(previous, target)
        reused += 1
    return reused


def collect_v3_horizon_data(
    *,
    source: str | Path,
    output: str | Path,
    controller_bundle: str | Path,
    v3_bundle: str | Path,
    horizon: int,
    workers: int,
    resume: bool,
    reuse_collection: str | Path | None = None,
) -> dict[str, Any]:
    if horizon != 3:
        raise ValueError("the registered v3 horizon pilot requires --horizon 3")
    if workers <= 0:
        raise ValueError("workers must be positive")
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    decisions = _read_jsonl(source_root / "collection" / "state_selection.jsonl")
    source_collection = _read_json(source_root / "collection" / "collection_report.json")
    if len(decisions) != 180 or not bool(source_collection.get("complete")):
        raise ValueError("v3-h3 source must be the complete registered 180-state pilot")
    identity = {
        "schema": V3_HORIZON_COLLECTION_SCHEMA,
        "trace_replay_contract": TRACE_REPLAY_CONTRACT,
        "source": str(source_root),
        "source_feature_sha256": sha256_file(
            source_root / "collection" / "feature_index.jsonl"
        ),
        "source_trials_sha256": sha256_file(
            source_root / "collection" / "trial_manifest.jsonl"
        ),
        "controller_bundle": str(Path(controller_bundle).resolve()),
        "controller_manifest_sha256": sha256_file(
            Path(controller_bundle).resolve() / "controller_manifest.json"
        ),
        "v3_bundle": str(Path(v3_bundle).resolve()),
        "v3_manifest_sha256": sha256_file(
            Path(v3_bundle).resolve() / "v3_manifest.json"
        ),
        "implementation": {
            str(path.name): sha256_file(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).resolve().with_name("trace_replay.py"),
                Path(__file__).resolve().with_name("closed_loop_confirmation.py"),
            )
        },
        "horizon": horizon,
        "trials": 2,
        "candidate_sampling": (
            "top2-v2-per-size-plus-v2-winner-v3-winner-and-adaptive"
        ),
        "continuation": "v2-full",
        "parallel_runtime": parallel_runtime_metadata(workers),
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("v3-h3 collection output belongs to a different run")
        if not resume:
            raise ValueError("v3-h3 collection exists; pass --resume")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    jobs = []
    base_candidate_ids: dict[str, str] = {}
    for feature in _read_jsonl(source_root / "collection" / "feature_index.jsonl"):
        if str(feature.get("route")) != "model" or not bool(
            feature.get("base_selected")
        ):
            continue
        state_id = str(feature["state_id"])
        if state_id in base_candidate_ids:
            raise ValueError(f"v3-h3 state has multiple v2 base candidates: {state_id}")
        base_candidate_ids[state_id] = str(feature["candidate_id"])
    if len(base_candidate_ids) != len(decisions):
        raise ValueError("v3-h3 source lacks one v2 base candidate per state")
    for decision in decisions:
        state_key = _fingerprint(
            {
                "state_id": decision["state_id"],
                "fingerprint": decision["before_repair_fingerprint"],
            }
        )[:20]
        jobs.append(
            {
                "decision": decision,
                "state_file": str(
                    output_root / "states" / str(decision["split"]) / f"{state_key}.json"
                ),
                "controller_bundle": str(Path(controller_bundle).resolve()),
                "v3_bundle": str(Path(v3_bundle).resolve()),
                "horizon": horizon,
                "trials": 2,
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    reused_state_count = (
        _reuse_completed_states(
            Path(reuse_collection).resolve(),
            output_root,
            jobs,
            run_fingerprint,
            base_candidate_ids,
        )
        if reuse_collection is not None
        else 0
    )
    if reused_state_count:
        for job in jobs:
            job["resume"] = True
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = time.perf_counter()
    cpu_sets = tuple(isolated_lane_cpu_sets(workers))
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_isolated_worker,
        initargs=(cpu_sets,),
    ) as pool:
        futures = {pool.submit(_horizon_state_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                completed.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "state_id": str(job["decision"]["state_id"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            finished = len(completed) + len(errors)
            elapsed = time.perf_counter() - started
            rate = 60.0 * finished / elapsed if elapsed > 0.0 else 0.0
            _write_json(
                output_root / "status.json",
                {
                    "schema": V3_HORIZON_COLLECTION_SCHEMA,
                    "status": "running",
                    "completed_states": len(completed),
                    "total_states": len(jobs),
                    "error_states": len(errors),
                    "states_per_minute": rate,
                    "estimated_remaining_seconds": (
                        60.0 * (len(jobs) - finished) / rate if rate > 0.0 else None
                    ),
                },
            )
    horizon_rows: list[dict[str, Any]] = []
    for row in completed:
        horizon_rows.extend(_read_json(Path(row["state_file"]))["trials"])
    horizon_rows.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["state_id"]),
            str(row["candidate_id"]),
            int(row["trial_index"]),
        )
    )
    _write_jsonl(output_root / "horizon_manifest.jsonl", horizon_rows)
    coverage = _coverage(decisions, horizon_rows, 2)
    _write_json(output_root / "coverage_report.json", coverage)
    report = {
        "schema": V3_HORIZON_COLLECTION_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "requested_state_count": len(jobs),
        "completed_state_count": len(completed),
        "error_state_count": len(errors),
        "errors": errors,
        "coverage": coverage,
        "reused_state_count": reused_state_count,
        "reuse_collection": (
            str(Path(reuse_collection).resolve())
            if reuse_collection is not None
            else None
        ),
        "horizon_manifest_sha256": sha256_file(output_root / "horizon_manifest.jsonl"),
        "complete": not errors and len(completed) == len(jobs) and coverage["passed"],
    }
    _write_json(output_root / "collection_report.json", report)
    _write_json(
        output_root / "status.json",
        {
            "schema": V3_HORIZON_COLLECTION_SCHEMA,
            "status": "complete" if report["complete"] else "error",
            "completed_states": len(completed),
            "total_states": len(jobs),
            "error_states": len(errors),
            "report": str(output_root / "collection_report.json"),
        },
    )
    return report


__all__ = [
    "V3_HORIZON_COLLECTION_SCHEMA",
    "collect_v3_horizon_data",
    "select_horizon_candidates",
]
