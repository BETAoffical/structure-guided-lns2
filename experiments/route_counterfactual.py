from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

from experiments.balanced_controller import load_balanced_controller
from experiments.candidate_pruning import (
    CandidatePruner,
    expected_families_from_proposal_config,
    no_pruning_metrics,
)
from experiments.closed_loop_confirmation import (
    CLOSED_LOOP_SCHEMA,
    feature_range_diagnostic,
    generate_online_candidates,
    repair_random_seed,
    score_online_candidates,
    validate_closed_loop_trace,
)
from experiments.closed_loop_trace_storage import (
    EPISODE_SCHEMA_V2,
    apply_extras_delta,
    apply_state_delta,
    open_trace_text,
    read_state_blob,
    read_trace_events,
    resolve_state_blob,
    trace_file_metadata,
)
from experiments.compact_controller_model import load_controller_bundle
from experiments.online_feature_engine import OnlineFeatureEngine
from experiments.repair_collection import (
    _CollectionRunLock,
    _dataset_fingerprint,
    _fingerprint,
    _load_dataset_rows,
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


ROUTE_COUNTERFACTUAL_SCHEMA = "lns2.skipped_model_once.v2"
ROUTE_COUNTERFACTUAL_VERSION = 2
COUNTERFACTUAL_SCOPE = "official-routes-model-once"


def _initial_state(
    collection_root: Path, trace_path: Path, event: dict[str, Any]
) -> dict[str, Any]:
    if str(event.get("schema")) != EPISODE_SCHEMA_V2:
        state = event.get("state")
        if not isinstance(state, dict):
            raise ValueError("counterfactual source trace is missing its initial state")
        return dict(state)
    state = read_state_blob(
        resolve_state_blob(trace_path, str(event["state_blob"]), collection_root)
    )
    extras = event.get("state_extras")
    if not isinstance(extras, dict):
        raise ValueError("counterfactual source trace has invalid initial extras")
    state.update(extras)
    return state


def _decision_rows(
    collection_root: Path, manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trace_path = collection_root / str(manifest["trace_file"])
    events = read_trace_events(trace_path)
    state = _initial_state(collection_root, trace_path, events[0])
    prefix: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for event in events[1:-1]:
        controller = event.get("controller")
        if not isinstance(controller, dict):
            raise ValueError("balanced source transition is missing controller data")
        route = str(controller.get("route", ""))
        if route not in {"model", "official_adaptive"}:
            raise ValueError("balanced source transition is missing a valid route")
        before_fingerprint = state_fingerprint(state)
        if before_fingerprint != str(event.get("before_fingerprint")):
            raise ValueError("counterfactual source before fingerprint mismatch")
        if str(event.get("schema")) == EPISODE_SCHEMA_V2:
            after = apply_state_delta(state, event["state_delta"])
            after.update(apply_extras_delta(state, event["state_extras_delta"]))
        else:
            after = dict(event["after"])
        actual_metrics = dict(event["metrics"])
        controller_seconds = float(
            controller.get("controller_seconds_before_repair", 0.0)
        )
        repair_seconds = float(event.get("repair_wall_seconds", 0.0))
        rows.append(
            {
                "decision_index": int(event["decision_index"]),
                "route": route,
                "before_fingerprint": before_fingerprint,
                "after_fingerprint": str(event["after_fingerprint"]),
                "prefix_actions": [dict(action) for action in prefix],
                "actual_action": dict(event["action"]),
                "actual_metrics": actual_metrics,
                "before_conflicts": int(state["num_of_colliding_pairs"]),
                "actual_lns2": {
                    "source": "balanced-main-trace",
                    "action": dict(event["action"]),
                    "metrics": actual_metrics,
                    "after_fingerprint": str(event["after_fingerprint"]),
                    "outcome": {
                        "conflicts_before": int(state["num_of_colliding_pairs"]),
                        "conflicts_after": int(after["num_of_colliding_pairs"]),
                        "conflict_delta": int(state["num_of_colliding_pairs"])
                        - int(after["num_of_colliding_pairs"]),
                        "success": bool(after["feasible"]),
                        "sum_of_costs_delta": int(after["sum_of_costs"])
                        - int(state["sum_of_costs"]),
                        "low_level_delta": dict(event.get("low_level_delta") or {}),
                        "controller_seconds": controller_seconds,
                        "repair_seconds": repair_seconds,
                        "total_decision_seconds": float(
                            controller.get(
                                "total_decision_seconds",
                                controller_seconds + repair_seconds,
                            )
                        ),
                    },
                },
            }
        )
        prefix.append(dict(event["action"]))
        state = after
    return rows, events


def _read_counterfactual_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open_trace_text(path, "r") as stream:
        for line in stream:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("route counterfactual row is not an object")
                rows.append(value)
    return rows


def _checkpoint_path(output_root: Path, episode_id: str, decision_index: int) -> Path:
    return (
        output_root
        / "state_checkpoints"
        / episode_id
        / f"decision-{decision_index:04d}.json.gz"
    )


def _valid_checkpoint(
    path: Path,
    *,
    run_fingerprint: str,
    episode_id: str,
    decision_index: int,
    before_fingerprint: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        rows = _read_counterfactual_rows(path)
        if len(rows) != 1:
            return None
        row = rows[0]
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    model = dict(row.get("counterfactual_model") or {})
    if (
        str(row.get("schema")) != ROUTE_COUNTERFACTUAL_SCHEMA
        or int(row.get("schema_version", -1)) != ROUTE_COUNTERFACTUAL_VERSION
        or str(row.get("run_fingerprint")) != run_fingerprint
        or str(row.get("episode_id")) != episode_id
        or int(row.get("decision_index", -1)) != decision_index
        or str(row.get("before_fingerprint")) != before_fingerprint
        or str(row.get("actual_route")) != "official_adaptive"
        or str(row.get("baseline_source")) != "balanced-main-trace"
        or not isinstance(row.get("actual_lns2"), dict)
        or int(model.get("model_use_count", -1)) != 1
        or not bool(row.get("replay_fingerprint_match"))
    ):
        return None
    return row


def _write_checkpoint(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.unlink(missing_ok=True)
    with open_trace_text(partial, "w") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(partial, path)


def _validate_counterfactual_file(
    path: Path,
    *,
    run_fingerprint: str,
    episode_id: str,
    expected_decision_indices: list[int],
) -> list[dict[str, Any]] | None:
    if not path.is_file():
        return None
    try:
        rows = _read_counterfactual_rows(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if len(rows) != len(expected_decision_indices):
        return None
    for expected_index, row in zip(expected_decision_indices, rows):
        model = dict(row.get("counterfactual_model") or {})
        if (
            str(row.get("schema")) != ROUTE_COUNTERFACTUAL_SCHEMA
            or int(row.get("schema_version", -1)) != ROUTE_COUNTERFACTUAL_VERSION
            or str(row.get("run_fingerprint")) != run_fingerprint
            or str(row.get("episode_id")) != episode_id
            or int(row.get("decision_index", -1)) != expected_index
            or str(row.get("actual_route")) != "official_adaptive"
            or str(row.get("baseline_source")) != "balanced-main-trace"
            or not isinstance(row.get("actual_lns2"), dict)
            or int(model.get("model_use_count", -1)) != 1
            or not bool(row.get("replay_fingerprint_match"))
        ):
            return None
    return rows


def _replay_prefix(job: dict[str, Any], actions: Iterable[dict[str, Any]]) -> tuple[Any, dict[str, Any]]:
    environment = _make_environment(
        job["dataset_root"], job["row"], job["environment"], "Adaptive"
    )
    state = _plain(environment.reset(seed=int(job["solver_seed"])))
    for action in actions:
        if bool(state["done"]):
            raise RuntimeError("counterfactual prefix terminated before target state")
        state = _plain(environment.step(dict(action)))["observation"]
    return environment, state


def _model_action(
    environment: Any,
    state: dict[str, Any],
    *,
    job: dict[str, Any],
    model: Any,
    model_ranges: dict[str, tuple[float, float]],
    candidate_pruner: CandidatePruner | None,
    decision_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    state_hash = state_fingerprint(state)
    candidates, proposal = generate_online_candidates(
        environment,
        state,
        task_id=str(job["row"]["task_id"]),
        solver_seed=int(job["solver_seed"]),
        decision_index=decision_index,
        proposal_config=job["proposal"],
        state_hash=state_hash,
        verify_full_state=(
            str(job.get("proposal_state_verification", "always")) == "always"
            or (
                str(job.get("proposal_state_verification")) == "sampled"
                and decision_index % 20 == 0
            )
        ),
    )
    engine = OnlineFeatureEngine(
        state,
        backend=str(job["feature_backend"]),
        shadow_validation=False,
        required_features={
            "realized_dynamic": model.base_feature_names,
            **(
                {"proposal_dynamic": tuple(candidate_pruner.ranges)}
                if candidate_pruner is not None
                else {}
            ),
        },
    )
    retained_indices = list(range(len(candidates)))
    pruning = no_pruning_metrics(len(candidates))
    proposal_feature_seconds = 0.0
    if candidate_pruner is not None:
        proposal_rows, proposal_metrics = engine.proposal_rows(
            candidates, state_hash=state_hash
        )
        proposal_feature_seconds = float(
            proposal_metrics.get("proposal_feature_seconds", 0.0)
        )
        retained_indices, pruning = candidate_pruner.prune(candidates, proposal_rows)
    retained = [candidates[index] for index in retained_indices]
    rows, realized_metrics = engine.realized_rows(retained, state_hash=state_hash)
    inference_started = time.perf_counter()
    selected_local, scores, margin = score_online_candidates(rows, model)
    inference_seconds = time.perf_counter() - inference_started
    selected_index = retained_indices[selected_local]
    selected = candidates[selected_index]
    repair_seed = repair_random_seed(
        str(job["row"]["task_id"]),
        int(job["solver_seed"]),
        state_hash,
        decision_index,
        str(selected["candidate_id"]),
        selected["proposal_seeds"],
    )
    action = {
        "mode": "explicit_neighborhood",
        "agents": list(map(int, selected["agents"])),
        "random_seed": repair_seed,
    }
    controller_seconds = time.perf_counter() - started
    diagnostic = feature_range_diagnostic(
        rows[selected_local], "realized_dynamic", model_ranges
    )
    return action, {
        "controller_seconds": controller_seconds,
        "proposal_seconds": float(proposal.get("proposal_seconds", 0.0)),
        "state_check_seconds": float(proposal.get("state_check_seconds", 0.0)),
        "state_analysis_seconds": float(
            engine.last_prepare_metrics.get("state_analysis_seconds", 0.0)
        ),
        "proposal_feature_seconds": proposal_feature_seconds,
        "realized_feature_seconds": float(
            realized_metrics.get("realized_feature_seconds", 0.0)
        ),
        "inference_seconds": inference_seconds,
        "pruner_seconds": float(pruning.get("pruner_seconds", 0.0)),
        "candidate_count_before": len(candidates),
        "candidate_count_after": len(retained),
        "pruner_fallback": bool(pruning.get("fallback", False)),
        "selected_candidate_id": str(selected["candidate_id"]),
        "selected_agents": list(map(int, selected["agents"])),
        "selected_score": float(scores[selected_local]),
        "score_margin": float(margin),
        "selected_feature_out_of_range_fraction": float(
            diagnostic["outside_fraction"]
        ),
    }


def _branch(
    name: str,
    *,
    job: dict[str, Any],
    decision: dict[str, Any],
    model: Any,
    model_ranges: dict[str, tuple[float, float]],
    candidate_pruner: CandidatePruner | None,
) -> dict[str, Any]:
    if name != "model":
        raise ValueError("skipped-state counterfactual only executes the model branch")
    environment, before = _replay_prefix(job, decision["prefix_actions"])
    before_hash = state_fingerprint(before)
    if before_hash != str(decision["before_fingerprint"]):
        raise RuntimeError(
            f"counterfactual replay fingerprint mismatch: {before_hash}"
        )
    before_soc = int(before["sum_of_costs"])
    action, controller = _model_action(
        environment,
        before,
        job=job,
        model=model,
        model_ranges=model_ranges,
        candidate_pruner=candidate_pruner,
        decision_index=int(decision["decision_index"]),
    )
    repair_started = time.perf_counter()
    first = _plain(environment.step(action))
    first_repair_seconds = time.perf_counter() - repair_started
    first_after = first["observation"]
    first_metrics = dict(first["metrics"])
    conflicts_before = int(before["num_of_colliding_pairs"])
    conflicts_after = int(first_after["num_of_colliding_pairs"])
    controller_seconds = float(controller["controller_seconds"])
    first_total = controller_seconds + first_repair_seconds
    return {
        "branch": "model",
        "model_use_count": 1,
        "action": action,
        "first_metrics": first_metrics,
        "first_after_fingerprint": state_fingerprint(first_after),
        "controller": controller,
        "outcome": {
            "conflicts_before": conflicts_before,
            "conflicts_after": conflicts_after,
            "conflict_delta": conflicts_before - conflicts_after,
            "success": bool(first_after["feasible"]),
            "sum_of_costs_delta": int(first_after["sum_of_costs"]) - before_soc,
            "low_level_delta": _low_level_delta(before, first_after),
            "controller_seconds": controller_seconds,
            "repair_seconds": first_repair_seconds,
            "total_decision_seconds": first_total,
        },
    }


def _pareto_relation(lns2: dict[str, Any], model: dict[str, Any]) -> str:
    left = dict(lns2)
    right = dict(model)
    l_quality = (
        not bool(left["success"]),
        int(left["conflicts_after"]),
        int(left["sum_of_costs_delta"]),
    )
    m_quality = (
        not bool(right["success"]),
        int(right["conflicts_after"]),
        int(right["sum_of_costs_delta"]),
    )
    l_time = float(left["total_decision_seconds"])
    m_time = float(right["total_decision_seconds"])
    model_no_worse = m_quality <= l_quality and m_time <= l_time
    lns2_no_worse = l_quality <= m_quality and l_time <= m_time
    if model_no_worse and (m_quality < l_quality or m_time < l_time):
        return "model_dominates"
    if lns2_no_worse and (l_quality < m_quality or l_time < m_time):
        return "lns2_dominates"
    if m_quality == l_quality and math.isclose(m_time, l_time, rel_tol=1e-9, abs_tol=1e-12):
        return "tie"
    return "quality_time_tradeoff"


def _route_counterfactual_worker(job: dict[str, Any]) -> dict[str, Any]:
    collection_root = Path(job["collection_root"])
    output_root = Path(job["output_root"])
    manifest = dict(job["manifest"])
    episode_id = str(manifest["episode_id"])
    validate_closed_loop_trace(
        collection_root / str(manifest["trace_file"]),
        str(job["source_run_fingerprint"]),
        expected_episode_id=episode_id,
        expected_policy="realized_dynamic",
        expected_solver_seed=int(manifest["solver_seed"]),
        metric_iteration_budget=int(job["metric_iteration_budget"]),
        collection_root=collection_root,
    )
    all_decisions, _ = _decision_rows(collection_root, manifest)
    decisions = [
        decision
        for decision in all_decisions
        if str(decision["route"]) == "official_adaptive"
    ]
    if any(
        str(dict(decision["actual_action"]).get("mode")) != "official"
        for decision in decisions
    ):
        raise ValueError("an official-route source decision did not use LNS2 Adaptive")
    decision_indices = [int(decision["decision_index"]) for decision in decisions]
    destination = output_root / "episodes" / f"{episode_id}.route-counterfactual.jsonl.gz"
    relative = destination.relative_to(output_root).as_posix()
    if bool(job.get("resume")):
        existing = _validate_counterfactual_file(
            destination,
            run_fingerprint=str(job["run_fingerprint"]),
            episode_id=episode_id,
            expected_decision_indices=decision_indices,
        )
        if existing is not None:
            return {
                "schema": ROUTE_COUNTERFACTUAL_SCHEMA,
                "schema_version": ROUTE_COUNTERFACTUAL_VERSION,
                "episode_id": episode_id,
                "task_id": manifest["task_id"],
                "map_id": manifest["map_id"],
                "layout_mode": manifest["layout_mode"],
                "agent_count": int(manifest["agent_count"]),
                "solver_seed": int(manifest["solver_seed"]),
                "decision_count": len(existing),
                "state_checkpoint_count": len(existing),
                "resumed_state_count": len(existing),
                "model_counterfactual_count": len(existing),
                "extra_lns2_execution_count": 0,
                "source_model_route_count": 0,
                "trace_file": relative,
                **trace_file_metadata(destination),
                "status": "resumed",
                "error": None,
            }
    bundle = load_controller_bundle(job["controller_bundle"])
    model = bundle.main_models["realized_dynamic"]
    model_ranges = bundle.main_ranges["realized_dynamic"]
    balanced = load_balanced_controller(job["balanced_config"])
    candidate_pruner = None
    if balanced.pruner_threshold is not None:
        if bundle.pruner_model is None:
            raise ValueError("balanced counterfactual requests a missing pruner")
        candidate_pruner = CandidatePruner(
            model=bundle.pruner_model,
            threshold=balanced.pruner_threshold,
            ranges=bundle.pruner_ranges,
            expected_families=expected_families_from_proposal_config(job["proposal"]),
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed_rows = []
    resumed_state_count = 0
    for decision in decisions:
        checkpoint = _checkpoint_path(
            output_root, episode_id, int(decision["decision_index"])
        )
        existing_state = (
            _valid_checkpoint(
                checkpoint,
                run_fingerprint=str(job["run_fingerprint"]),
                episode_id=episode_id,
                decision_index=int(decision["decision_index"]),
                before_fingerprint=str(decision["before_fingerprint"]),
            )
            if bool(job.get("resume"))
            else None
        )
        if existing_state is not None:
            completed_rows.append(existing_state)
            resumed_state_count += 1
            continue
        model_result = _branch(
            "model",
            job=job,
            decision=decision,
            model=model,
            model_ranges=model_ranges,
            candidate_pruner=candidate_pruner,
        )
        baseline = dict(decision["actual_lns2"])
        row = {
            "schema": ROUTE_COUNTERFACTUAL_SCHEMA,
            "schema_version": ROUTE_COUNTERFACTUAL_VERSION,
            "run_fingerprint": job["run_fingerprint"],
            "episode_id": episode_id,
            "task_id": manifest["task_id"],
            "map_id": manifest["map_id"],
            "layout_mode": manifest["layout_mode"],
            "agent_count": int(manifest["agent_count"]),
            "solver_seed": int(manifest["solver_seed"]),
            "decision_index": int(decision["decision_index"]),
            "actual_route": decision["route"],
            "before_fingerprint": decision["before_fingerprint"],
            "before_conflicts": int(decision["before_conflicts"]),
            "baseline_source": "balanced-main-trace",
            "replay_fingerprint_match": True,
            "actual_lns2": baseline,
            "counterfactual_model": model_result,
            "pareto_relation": _pareto_relation(
                dict(baseline["outcome"]), dict(model_result["outcome"])
            ),
        }
        _write_checkpoint(checkpoint, row)
        completed_rows.append(row)

    partial = destination.with_name(destination.name + ".partial")
    partial.unlink(missing_ok=True)
    with open_trace_text(partial, "w") as stream:
        for row in completed_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
    os.replace(partial, destination)
    validated = _validate_counterfactual_file(
        destination,
        run_fingerprint=str(job["run_fingerprint"]),
        episode_id=episode_id,
        expected_decision_indices=decision_indices,
    )
    if validated is None:
        raise RuntimeError("new route counterfactual file failed validation")
    return {
        "schema": ROUTE_COUNTERFACTUAL_SCHEMA,
        "schema_version": ROUTE_COUNTERFACTUAL_VERSION,
        "episode_id": episode_id,
        "task_id": manifest["task_id"],
        "map_id": manifest["map_id"],
        "layout_mode": manifest["layout_mode"],
        "agent_count": int(manifest["agent_count"]),
        "solver_seed": int(manifest["solver_seed"]),
        "decision_count": len(validated),
        "state_checkpoint_count": len(completed_rows),
        "resumed_state_count": resumed_state_count,
        "model_counterfactual_count": len(validated),
        "extra_lns2_execution_count": 0,
        "source_model_route_count": 0,
        "trace_file": relative,
        **trace_file_metadata(destination),
        "status": "ok",
        "error": None,
    }


def run_route_counterfactuals(
    collection: str | Path,
    output: str | Path,
    *,
    workers: int = 1,
    resume: bool = False,
) -> dict[str, Any]:
    collection_root = Path(collection).resolve()
    output_root = Path(output).resolve()
    source_run = _read_json(collection_root / "run_config.json")
    if str(source_run.get("schema")) != CLOSED_LOOP_SCHEMA:
        raise ValueError("route counterfactual source is not a closed-loop collection")
    if str(source_run.get("controller")) != "v2-balanced":
        raise ValueError("route counterfactual source must use v2-balanced")
    configuration = dict(source_run["configuration"])
    balanced_config = source_run.get("balanced_config") or configuration.get(
        "balanced_config"
    )
    if not isinstance(balanced_config, dict):
        raise ValueError("balanced source collection is missing its frozen config")
    balanced = load_balanced_controller(balanced_config)
    if str(balanced.source.get("selection_unit")) != "complete_episode":
        raise ValueError(
            "skipped-state counterfactual requires complete-episode route calibration"
        )
    dataset_root = Path(str(source_run["dataset"])).resolve()
    if _dataset_fingerprint(dataset_root) != str(source_run["dataset_fingerprint"]):
        raise ValueError("route counterfactual dataset fingerprint mismatch")
    split = str(configuration["split"])
    dataset_rows = {
        str(row["task_id"]): row for row in _load_dataset_rows(dataset_root, [split])
    }
    manifests = [
        row
        for row in _read_jsonl(collection_root / "realized_dynamic_manifest.jsonl")
        if str(row.get("status")) in {"ok", "resumed"}
    ]
    if not manifests:
        raise ValueError("balanced collection has no successful realized_dynamic episodes")
    controller_bundle_value = configuration.get("controller_bundle")
    if not controller_bundle_value:
        raise ValueError("balanced source collection is missing its controller path")
    controller_bundle = Path(str(controller_bundle_value)).resolve()
    if not (controller_bundle / "controller_manifest.json").is_file():
        raise ValueError("balanced source controller bundle is unavailable")
    implementation_fingerprint = _fingerprint(
        Path(__file__).read_text(encoding="utf-8")
    )
    run_fingerprint = _fingerprint(
        {
            "schema": ROUTE_COUNTERFACTUAL_SCHEMA,
            "source_run_fingerprint": source_run["run_fingerprint"],
            "balanced_config": balanced_config,
            "scope": COUNTERFACTUAL_SCOPE,
            "baseline_source": "balanced-main-trace",
            "continuation_steps": 0,
            "model_use_count": 1,
            "proposal_state_verification": (
                "sampled" if bool(source_run.get("formal")) else "always"
            ),
            "implementation_fingerprint": implementation_fingerprint,
        }
    )
    run_config = {
        "schema": ROUTE_COUNTERFACTUAL_SCHEMA,
        "schema_version": ROUTE_COUNTERFACTUAL_VERSION,
        "run_fingerprint": run_fingerprint,
        "source_collection": str(collection_root),
        "source_run_fingerprint": source_run["run_fingerprint"],
        "dataset": str(dataset_root),
        "controller_bundle": str(controller_bundle),
        "balanced_config": balanced_config,
        "scope": COUNTERFACTUAL_SCOPE,
        "baseline_source": "balanced-main-trace",
        "continuation_steps": 0,
        "model_use_count": 1,
        "proposal_state_verification": (
            "sampled" if bool(source_run.get("formal")) else "always"
        ),
        "implementation_fingerprint": implementation_fingerprint,
        "workers": int(workers),
    }
    run_path = output_root / "run_config.json"
    if run_path.is_file():
        existing = _read_json(run_path)
        if str(existing.get("run_fingerprint")) != run_fingerprint:
            raise ValueError("counterfactual output contains a different run")
        if not resume:
            raise ValueError("counterfactual output already exists; pass resume")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(run_path, run_config)
    jobs = []
    for manifest in manifests:
        task_id = str(manifest["task_id"])
        if task_id not in dataset_rows:
            raise ValueError(f"counterfactual source task is missing: {task_id}")
        jobs.append(
            {
                "manifest": manifest,
                "row": dataset_rows[task_id],
                "solver_seed": int(manifest["solver_seed"]),
                "dataset_root": str(dataset_root),
                "environment": configuration["environment"],
                "proposal": configuration["proposal"],
                "feature_backend": source_run["feature_backend"],
                "controller_bundle": str(controller_bundle),
                "balanced_config": balanced_config,
                "collection_root": str(collection_root),
                "output_root": str(output_root),
                "run_fingerprint": run_fingerprint,
                "source_run_fingerprint": source_run["run_fingerprint"],
                "metric_iteration_budget": int(
                    configuration["metric_iteration_budget"]
                ),
                "proposal_state_verification": (
                    "sampled" if bool(source_run.get("formal")) else "always"
                ),
                "resume": resume,
            }
        )
    with _CollectionRunLock(output_root, run_fingerprint, "route-counterfactual"):
        results = _run_jobs(
            _route_counterfactual_worker,
            jobs,
            int(workers),
            phase="route-counterfactual",
            output_root=output_root,
            run_fingerprint=run_fingerprint,
            timeout_seconds=float(configuration["episode_process_timeout_seconds"])
            * max(2.0, float(configuration["max_decisions"])),
        )
    _write_jsonl(output_root / "counterfactual_manifest.jsonl", results)
    expected_official_routes = sum(
        int(dict(row.get("summary") or {}).get("official_decision_count", 0))
        for row in manifests
    )
    expected_state_count = expected_official_routes
    decision_count = sum(int(row.get("decision_count", 0)) for row in results)
    model_counterfactuals = sum(
        int(row.get("model_counterfactual_count", 0)) for row in results
    )
    extra_lns2_executions = sum(
        int(row.get("extra_lns2_execution_count", 0)) for row in results
    )
    source_model_routes = sum(
        int(row.get("source_model_route_count", 0)) for row in results
    )
    error_count = sum(str(row.get("status")) not in {"ok", "resumed"} for row in results)
    state_checkpoint_count = sum(
        int(row.get("state_checkpoint_count", 0)) for row in results
    )
    resumed_state_count = sum(int(row.get("resumed_state_count", 0)) for row in results)
    coverage_complete = (
        error_count == 0
        and decision_count == expected_state_count
        and state_checkpoint_count == expected_state_count
        and model_counterfactuals == expected_state_count
        and extra_lns2_executions == 0
        and source_model_routes == 0
    )
    summary = {
        "schema": ROUTE_COUNTERFACTUAL_SCHEMA,
        "schema_version": ROUTE_COUNTERFACTUAL_VERSION,
        "run_fingerprint": run_fingerprint,
        "episode_count": len(results),
        "counterfactual_state_count": decision_count,
        "expected_counterfactual_state_count": expected_state_count,
        "model_counterfactual_count": model_counterfactuals,
        "extra_lns2_execution_count": extra_lns2_executions,
        "source_model_route_count": source_model_routes,
        "state_checkpoint_count": state_checkpoint_count,
        "resumed_state_count": resumed_state_count,
        "expected_official_route_count": expected_official_routes,
        "replay_fingerprint_mismatch_count": 0 if error_count == 0 else None,
        "missing_model_result_count": 0 if error_count == 0 else None,
        "coverage_complete": coverage_complete,
        "error_count": error_count,
        "passed": coverage_complete,
    }
    _write_json(output_root / "counterfactual_summary.json", summary)
    return summary


def iter_route_counterfactual_rows(root: str | Path) -> Iterable[dict[str, Any]]:
    output_root = Path(root).resolve()
    for manifest in _read_jsonl(output_root / "counterfactual_manifest.jsonl"):
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            continue
        trace = (output_root / str(manifest["trace_file"])).resolve()
        try:
            trace.relative_to(output_root)
        except ValueError as error:
            raise ValueError("counterfactual trace escapes its output root") from error
        metadata = trace_file_metadata(trace)
        if (
            str(metadata["trace_sha256"]) != str(manifest.get("trace_sha256"))
            or int(metadata["trace_bytes"]) != int(manifest.get("trace_bytes", -1))
        ):
            raise ValueError(f"counterfactual trace metadata mismatch: {trace}")
        yield from _read_counterfactual_rows(trace)


__all__ = [
    "COUNTERFACTUAL_SCOPE",
    "ROUTE_COUNTERFACTUAL_SCHEMA",
    "iter_route_counterfactual_rows",
    "run_route_counterfactuals",
]
