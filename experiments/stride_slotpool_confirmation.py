from __future__ import annotations

import collections
import concurrent.futures
import os
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
from experiments.repair_collection import (
    _fingerprint,
    _plain,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_robustaction_label_collection import _aggregate_candidate
from experiments.stride_slotpool import (
    _evaluate_states,
    candidate_matrix,
    pair_matrix,
    predict_slotpool_model,
    stable_pair_table,
    summarize_slotpool_acceptance,
)
from experiments.stride_structpool_size_ablation import (
    _candidate_jaccard,
    _forbidden_hits,
    _outcome_blind_base_anchor,
)
from experiments.trace_replay import replay_prefix, restore_repair_state
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome
from lns2_selector.runtime.topology_candidates import generate_structpool_candidate_grid


CONFIG_SCHEMA = "lns2.stride.slotpool_fresh_confirmation_config.v1"
COLLECTION_SCHEMA = "lns2.stride.slotpool_fresh_collection.v1"
STATE_SCHEMA = "lns2.stride.slotpool_fresh_state.v1"
TRIAL_SCHEMA = "lns2.stride.slotpool_fresh_trial.v1"
AGGREGATE_SCHEMA = "lns2.stride.slotpool_fresh_current_step_label.v1"
ANALYSIS_SCHEMA = "lns2.stride.slotpool_fresh_confirmation.v1"
TRIAL_INDICES = tuple(range(16))


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"registered SlotPool confirmation input is missing: {path}")
    observed = sha256_file(path)
    if observed != str(specification["sha256"]):
        raise ValueError(
            f"registered SlotPool confirmation input changed: {path}: "
            f"expected {specification['sha256']}, got {observed}"
        )
    return path


def validate_slotpool_confirmation_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("confirmation_id") != "stride-slotpool-fresh-confirmation-v1"
        or config.get("model_id") != "stride-slotpool-v1"
        or config.get("scientific_status")
        != "preregistered_frozen_model_map_disjoint_current_step_confirmation"
    ):
        raise ValueError("SlotPool fresh-confirmation identity changed")
    confirmation = dict(config.get("confirmation") or {})
    if (
        len(set(map(str, confirmation.get("map_ids") or ()))) != 6
        or len(set(map(str, confirmation.get("task_ids") or ()))) != 12
        or tuple(map(int, confirmation.get("solver_seeds") or ())) != (1, 2)
        or int(confirmation.get("expected_map_count", 0)) != 6
        or int(confirmation.get("expected_task_count", 0)) != 12
        or int(confirmation.get("expected_state_count", 0)) != 24
        or int(confirmation.get("expected_states_per_map", 0)) != 4
        or tuple(map(int, confirmation.get("trial_indices") or ())) != TRIAL_INDICES
        or confirmation.get("task_selection_source")
        != "preflight_recommended_tasks_using_initial_conflicts_only"
        or confirmation.get("all_registered_states_required") is not True
        or confirmation.get("result_based_state_filtering") is not False
    ):
        raise ValueError("SlotPool fresh-confirmation cohort changed")
    development = set(map(str, config.get("development_map_ids") or ()))
    fresh = set(map(str, confirmation["map_ids"]))
    if len(development) != 16 or development & fresh:
        raise ValueError("SlotPool fresh maps overlap development maps")
    if dict(config.get("offline_acceptance") or {}) != {
        "global_best_retention_minimum": 0.9,
        "mean_normalized_regret_maximum": 0.02,
        "maximum_map_mean_regret": 0.05,
        "maximum_topology_group_mean_regret": 0.05,
        "first_fixed_half_best_retention_minimum": 0.85,
        "second_fixed_half_best_retention_minimum": 0.85,
        "pairwise_accuracy_minimum": 0.6,
        "maximum_selected_candidates": 6,
        "raw_candidate_count_must_be_lower_than_full_grid": True,
    }:
        raise ValueError("SlotPool fresh-confirmation gates changed")
    if dict(config.get("execution") or {}) != {
        "workers": 2,
        "monitor_interval_minutes": 30,
        "resume_required": True,
    }:
        raise ValueError("SlotPool fresh-confirmation execution changed")
    boundary = dict(config.get("claim_boundary") or {})
    if (
        boundary.get("frozen_model_no_retraining") is not True
        or boundary.get("current_step_labels_only") is not True
        or any(
            bool(boundary.get(name))
            for name in (
                "candidate_repair_results_used_for_state_selection",
                "prior_boundary_quality_results_used_for_state_selection",
                "formal_ttf_claim",
                "runtime_integration_before_acceptance",
                "known_maze_included",
            )
        )
    ):
        raise ValueError("SlotPool fresh-confirmation claim boundary changed")
    if project_root is not None:
        for specification in dict(config.get("inputs") or {}).values():
            _registered(project_root.resolve(), dict(specification))


class _PortableEstimator:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def predict_proba(self, values: Any) -> Any:
        import numpy as np

        positive = predict_slotpool_model(self.payload, values)
        return np.column_stack((1.0 - positive, positive))


def _state_artifact_valid(
    payload: dict[str, Any], *, identity: str, state_id: str
) -> bool:
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("identity") != identity
        or payload.get("state_id") != state_id
        or payload.get("complete") is not True
        or _forbidden_hits(payload)
    ):
        return False
    candidates = payload.get("candidate_aggregates")
    trials = payload.get("trials")
    if not isinstance(candidates, list) or not candidates or not isinstance(trials, list):
        return False
    candidate_ids = {str(row.get("candidate_id")) for row in candidates}
    product = collections.Counter(
        (str(row.get("candidate_id")), int(row.get("trial_index", -1)))
        for row in trials
    )
    return (
        len(trials) == len(candidates) * len(TRIAL_INDICES)
        and set(product) == {(candidate_id, index) for candidate_id in candidate_ids for index in TRIAL_INDICES}
        and all(count == 1 for count in product.values())
    )


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    state_row = dict(job["state_row"])
    state_id = str(state_row["state_id"])
    output_path = Path(str(job["output_path"]))
    identity = str(job["identity"])
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _state_artifact_valid(existing, identity=identity, state_id=state_id):
            return {
                "state_id": state_id,
                "status": "resumed",
                "output_path": str(output_path),
                "candidate_count": len(existing["candidate_aggregates"]),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"invalid existing SlotPool fresh state: {output_path}")
    replay = {
        "dataset_root": str(job["dataset_root"]),
        "row": dict(job["dataset_row"]),
        "environment": dict(job["environment"]),
        "solver_seed": int(state_row["solver_seed"]),
        "replay_destroy_strategy": "Adaptive",
    }
    _, initial = replay_prefix(replay, [])
    before = state_fingerprint(initial)
    before_repair = repair_structure_fingerprint(initial)
    before_conflicts = int(initial["num_of_colliding_pairs"])
    if (
        before != str(state_row["state_fingerprint"])
        or before_conflicts != int(state_row["initial_conflicts"])
        or before_conflicts <= 0
    ):
        raise RuntimeError(f"SlotPool fresh replay mismatch: {state_id}")
    cache = TopologyAnalysisCache(initial, backend="native")
    if cache.analysis is None:
        raise RuntimeError("SlotPool fresh native topology analysis is missing")
    grid = generate_structpool_candidate_grid(initial, cache.analysis)
    engine = OnlineFeatureEngine(
        initial,
        backend="native",
        required_features={"realized_dynamic": PROFILE_FEATURE_NAMES["realized_dynamic"]},
    )
    if cache.last_native_prepared is not None:
        engine.prepare(initial, prepared_native_analysis=cache.last_native_prepared)
    feature_rows, _feature_metrics = engine.realized_rows(grid, state_hash=before)
    features_by_id = {
        str(row["candidate_id"]): dict(row["features"]["realized_dynamic"])
        for row in feature_rows
    }
    bundle = load_controller_bundle(Path(str(job["frozen_v2_manifest"])).parent)
    model = bundle.main_models["realized_dynamic"]
    base_candidates = []
    for row in job["coverage_candidates"]:
        families = tuple(map(str, row.get("selection_families") or ()))
        if families and all(
            family.startswith(("target:", "collision:", "random:"))
            for family in families
        ):
            base_candidates.append({**row, "candidate_kind": "base"})
    anchor = _outcome_blind_base_anchor(
        {"candidates": base_candidates}, model
    )
    candidates = []
    for candidate in grid:
        candidate_id = str(candidate["candidate_id"])
        features = features_by_id[candidate_id]
        if set(features) != set(PROFILE_FEATURE_NAMES["realized_dynamic"]):
            raise RuntimeError("SlotPool fresh feature schema changed")
        candidates.append(
            {
                **candidate,
                "candidate_kind": "structpool-grid",
                "features": features,
                "v2_anchor_jaccard": _candidate_jaccard(
                    candidate["agents"], anchor["agents"]
                ),
            }
        )
    if not candidates or len(candidates) > 24:
        raise RuntimeError("SlotPool fresh grid product is invalid")
    restore_seed = repairability_restore_seed(before_repair)
    trials = []
    aggregates = []
    for candidate in candidates:
        candidate_trials = []
        agents = list(map(int, candidate["agents"]))
        for trial_index in TRIAL_INDICES:
            pp_seed = repairability_pp_seed(before_repair, trial_index)
            environment, branch = restore_repair_state(
                replay, initial, seed=restore_seed
            )
            if repair_structure_fingerprint(branch) != before_repair:
                raise RuntimeError("SlotPool fresh restored structure changed")
            result = _plain(environment.step(_paired_action(agents, pp_seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair = repair_structure_fingerprint(after)
            candidate_trials.append(
                {
                    "schema": TRIAL_SCHEMA,
                    "state_id": state_id,
                    "candidate_id": str(candidate["candidate_id"]),
                    "candidate_kind": "structpool-grid",
                    "trial_index": trial_index,
                    "pp_seed": pp_seed,
                    "before_conflicts": before_conflicts,
                    "before_fingerprint": before,
                    "before_repair_fingerprint": before_repair,
                    "conflicts_after": conflicts_after,
                    "normalized_conflict_reduction": (
                        before_conflicts - conflicts_after
                    ) / max(1, before_conflicts),
                    "replan_success": bool(metrics["replan_success"]),
                    "feasible": bool(after.get("feasible")),
                    "repair_outcome": classify_repair_outcome(
                        before_fingerprint=before_repair,
                        after_fingerprint=after_repair,
                        replan_success=bool(metrics["replan_success"]),
                        conflicts_before=before_conflicts,
                        conflicts_after=conflicts_after,
                        feasible=bool(after.get("feasible")),
                    ),
                    "after_repair_fingerprint": after_repair,
                }
            )
        trials.extend(candidate_trials)
        aggregate = _aggregate_candidate(candidate, candidate_trials)
        aggregate.update(
            {
                "schema": AGGREGATE_SCHEMA,
                "state_id": state_id,
                "task_id": str(state_row["task_id"]),
                "map_id": str(state_row["map_id"]),
                "layout_mode": str(state_row["layout_family"]),
                "solver_seed": int(state_row["solver_seed"]),
                "agent_count": int(job["dataset_row"]["agent_count"]),
                "before_conflicts": before_conflicts,
                "no_progress_rate": 1.0 - float(aggregate["progress_rate"]),
                "repair_success_rate": float(aggregate["replan_success_rate"]),
                "v2_anchor_jaccard": float(candidate["v2_anchor_jaccard"]),
                "structpool_support_count_by_family": dict(
                    candidate["structpool_support_count_by_family"]
                ),
                "structpool_support_ratio_by_family": dict(
                    candidate["structpool_support_ratio_by_family"]
                ),
                "structpool_nominal_size_by_family": dict(
                    candidate["structpool_nominal_size_by_family"]
                ),
                "structpool_grid_pure_family": bool(
                    candidate["structpool_grid_pure_family"]
                ),
                "structpool_grid_duplicate_provenance_count": int(
                    candidate["structpool_grid_duplicate_provenance_count"]
                ),
            }
        )
        aggregates.append(aggregate)
    payload = {
        "schema": STATE_SCHEMA,
        "identity": identity,
        "complete": True,
        "state_id": state_id,
        "state": state_row,
        "before_fingerprint": before,
        "before_repair_fingerprint": before_repair,
        "before_conflicts": before_conflicts,
        "v2_base_anchor": anchor,
        "restore_seed": restore_seed,
        "candidate_aggregates": aggregates,
        "trials": trials,
        "runtime_fields_stored": False,
        "future_trajectory_stored": False,
    }
    if not _state_artifact_valid(payload, identity=identity, state_id=state_id):
        raise RuntimeError("SlotPool fresh state artifact is invalid")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": state_id,
        "status": "ok",
        "output_path": str(output_path),
        "candidate_count": len(aggregates),
        "trial_count": len(trials),
    }


def collect_slotpool_confirmation(
    *, config_path: str | Path, output: str | Path, resume: bool = True
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_slotpool_confirmation_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    development = _read_json(inputs["development_report"])
    model_payload = _read_json(inputs["frozen_model"])
    if (
        development.get("runtime_integration_allowed") is not True
        or model_payload.get("confirmation_labels_seen") is not False
    ):
        raise ValueError("SlotPool frozen development model is not eligible")
    task_ids = set(map(str, config["confirmation"]["task_ids"]))
    solver_seeds = set(map(int, config["confirmation"]["solver_seeds"]))
    state_rows = [
        row
        for row in _read_jsonl(inputs["coverage_states"])
        if str(row["task_id"]) in task_ids and int(row["solver_seed"]) in solver_seeds
    ]
    state_rows.sort(key=lambda row: str(row["state_id"]))
    dataset_rows = {
        str(row["task_id"]): row for row in _read_jsonl(inputs["dataset_manifest"])
    }
    qualification = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in _read_jsonl(inputs["qualification_manifest"])
    }
    coverage_candidates: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in _read_jsonl(inputs["coverage_candidates"]):
        if str(row["state_id"]) in {str(value["state_id"]) for value in state_rows}:
            coverage_candidates[str(row["state_id"])].append(row)
    runtime = _read_json(inputs["runtime_config"])
    expected = int(config["confirmation"]["expected_state_count"])
    maps = collections.Counter(str(row["map_id"]) for row in state_rows)
    if (
        len(state_rows) != expected
        or len({str(row["task_id"]) for row in state_rows}) != 12
        or maps != collections.Counter({name: 4 for name in config["confirmation"]["map_ids"]})
        or set(coverage_candidates) != {str(row["state_id"]) for row in state_rows}
    ):
        raise ValueError("SlotPool fresh registered state product changed")
    for row in state_rows:
        key = (str(row["task_id"]), int(row["solver_seed"]))
        if (
            str(row["task_id"]) not in dataset_rows
            or key not in qualification
            or str(qualification[key]["state_fingerprint"]) != str(row["state_fingerprint"])
        ):
            raise ValueError(f"SlotPool fresh replay provenance changed: {row['state_id']}")
    identity_payload = {
        "schema": COLLECTION_SCHEMA,
        "config_sha256": sha256_file(config_path),
        "frozen_model_sha256": sha256_file(inputs["frozen_model"]),
        "state_ids": [str(row["state_id"]) for row in state_rows],
        "trial_indices": list(TRIAL_INDICES),
        "producer": producer_identity(
            project_root=project_root,
            source_files=(
                "experiments/stride_slotpool_confirmation.py",
                "experiments/stride_slotpool.py",
                "experiments/stride_structpool_size_ablation.py",
                "lns2_selector/runtime/topology_candidates.py",
            ),
            native_required=True,
            package_names=("numpy",),
        ),
    }
    identity = _fingerprint(identity_payload)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run_config.json"
    if run_path.is_file() and _read_json(run_path).get("identity") != identity:
        raise ValueError("SlotPool fresh output belongs to another run")
    _write_json(run_path, {**identity_payload, "identity": identity})
    jobs = []
    for row in state_rows:
        state_id = str(row["state_id"])
        jobs.append(
            {
                "state_row": row,
                "dataset_root": str((project_root / str(config["dataset_root"])).resolve()),
                "dataset_row": dataset_rows[str(row["task_id"])],
                "environment": runtime["environment"],
                "coverage_candidates": coverage_candidates[state_id],
                "frozen_v2_manifest": str(inputs["frozen_v2_manifest"]),
                "identity": identity,
                "resume": bool(resume),
                "output_path": str(
                    output / "states" / f"{_fingerprint({'state_id': state_id})[:20]}.json"
                ),
            }
        )
    results = []
    errors = []
    status_path = output / "collection_status.json"
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(config["execution"]["workers"])
    ) as executor:
        futures = {executor.submit(_collect_state, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            try:
                results.append(future.result())
            except Exception as error:
                errors.append(
                    {
                        "state_id": str(job["state_row"]["state_id"]),
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
            _write_json(
                status_path,
                {
                    "schema": COLLECTION_SCHEMA,
                    "status": "running",
                    "requested_state_count": len(jobs),
                    "completed_state_count": len(results),
                    "completed_candidate_count": sum(int(row["candidate_count"]) for row in results),
                    "completed_trial_count": sum(int(row["trial_count"]) for row in results),
                    "error_state_count": len(errors),
                    "errors": errors,
                },
            )
    aggregates = []
    trials = []
    for result in sorted(results, key=lambda row: str(row["state_id"])):
        payload = _read_json(Path(str(result["output_path"])))
        if not _state_artifact_valid(
            payload, identity=identity, state_id=str(result["state_id"])
        ):
            errors.append({"state_id": str(result["state_id"]), "error": "invalid artifact"})
        else:
            aggregates.extend(payload["candidate_aggregates"])
            trials.extend(payload["trials"])
    complete = len(results) == len(jobs) and not errors
    if complete:
        aggregates.sort(key=lambda row: (str(row["state_id"]), str(row["candidate_id"])))
        trials.sort(
            key=lambda row: (
                str(row["state_id"]), str(row["candidate_id"]), int(row["trial_index"])
            )
        )
        _write_jsonl(output / "candidate_aggregates.jsonl", aggregates)
        _write_jsonl(output / "repair_trials.jsonl", trials)
    report = {
        "schema": COLLECTION_SCHEMA,
        "identity": identity,
        "requested_state_count": len(jobs),
        "completed_state_count": len(results),
        "candidate_count": len(aggregates),
        "trial_count": len(trials),
        "error_state_count": len(errors),
        "errors": errors,
        "complete": complete,
        "formal_ttf_claim": False,
        "artifacts": (
            {
                "candidate_aggregates_sha256": sha256_file(output / "candidate_aggregates.jsonl"),
                "repair_trials_sha256": sha256_file(output / "repair_trials.jsonl"),
            }
            if complete
            else {}
        ),
    }
    _write_json(output / "collection_report.json", report)
    _write_json(status_path, {**report, "status": "complete" if complete else "error"})
    return report


def analyze_slotpool_confirmation(
    *, config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_slotpool_confirmation_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    collection = Path(collection).resolve()
    collection_report = _read_json(collection / "collection_report.json")
    if collection_report.get("complete") is not True:
        raise ValueError("SlotPool fresh analysis requires a complete collection")
    rows = _read_jsonl(collection / "candidate_aggregates.jsonl")
    if _forbidden_hits(rows):
        raise ValueError("forbidden future/runtime field entered SlotPool fresh analysis")
    state_ids = {str(row["state_id"]) for row in rows}
    if len(state_ids) != int(config["confirmation"]["expected_state_count"]):
        raise ValueError("SlotPool fresh analysis state coverage changed")
    values = candidate_matrix(rows)
    pairs = stable_pair_table(rows, minimum_gap=0.01)
    model = _PortableEstimator(_read_json(inputs["frozen_model"]))
    state_rows = _evaluate_states(
        estimator=model,
        candidate_values=values,
        rows=rows,
        state_ids=state_ids,
        maximum_candidates=6,
    )
    probabilities = model.predict_proba(pair_matrix(values, pairs))[:, 1]
    correct_weight = sum(
        float(row["weight"])
        for probability, row in zip(probabilities, pairs)
        if (float(probability) >= 0.5) == bool(row["label"])
    )
    total_weight = sum(float(row["weight"]) for row in pairs)
    acceptance = summarize_slotpool_acceptance(
        rows=state_rows,
        pairwise_accuracy=correct_weight / total_weight,
        config=config,
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "state_evaluation.jsonl"
    _write_jsonl(states_path, state_rows)
    report = {
        "schema": ANALYSIS_SCHEMA,
        "scientific_status": (
            "fresh_map_candidate_quality_passed_runtime_may_be_designed"
            if acceptance["passed"]
            else "fresh_map_candidate_quality_failed_runtime_hard_stop"
        ),
        "model_id": "stride-slotpool-v1",
        "frozen_model_sha256": sha256_file(inputs["frozen_model"]),
        "model_retrained": False,
        "state_count": len(state_rows),
        "map_count": len({str(row["map_id"]) for row in state_rows}),
        "candidate_count": len(rows),
        "stable_pair_count": len(pairs),
        "states_with_stable_pairs": len({str(row["state_id"]) for row in pairs}),
        "development_map_overlap": sorted(
            set(map(str, config["development_map_ids"]))
            & {str(row["map_id"]) for row in state_rows}
        ),
        "acceptance": acceptance,
        "runtime_integration_allowed": bool(acceptance["passed"]),
        "known_maze_included": False,
        "formal_ttf_claim": False,
        "next_decision": (
            "design_guardpool_and_maze_regression_without_retraining_slotpool"
            if acceptance["passed"]
            else "stop_slotpool_runtime_and_report_map_specific_failure"
        ),
        "artifacts": {"state_evaluation_sha256": sha256_file(states_path)},
    }
    _write_json(output / "slotpool_fresh_confirmation_report.json", report)
    return report


__all__ = [
    "analyze_slotpool_confirmation",
    "collect_slotpool_confirmation",
    "validate_slotpool_confirmation_config",
]
