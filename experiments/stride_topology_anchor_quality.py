from __future__ import annotations

import concurrent.futures
import os
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
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
from experiments.stride_lns import assign_structure_scores, post_structure_metrics
from experiments.stride_stability import stride_extended_pp_seed
from experiments.trace_replay import replay_prefix
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint
from lns2_selector.runtime.repair_outcomes import classify_repair_outcome


CONFIG_SCHEMA = "lns2.stride.topology_anchor_quality_pilot_config.v1"
TRIAL_SCHEMA = "lns2.stride.topology_anchor_quality_trial.v1"
STATE_SCHEMA = "lns2.stride.topology_anchor_quality_state.v1"
COLLECTION_SCHEMA = "lns2.stride.topology_anchor_quality_collection.v1"
REPORT_SCHEMA = "lns2.stride.topology_anchor_quality_report.v1"


def _quality_registered_path(project_root: Path, artifact: dict[str, Any]) -> Path:
    path = (project_root / str(artifact["path"])).resolve()
    if sha256_file(path) != str(artifact["sha256"]):
        raise ValueError(f"topology-anchor quality input SHA differs: {artifact['path']}")
    return path


def validate_topology_anchor_quality_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("unexpected topology-anchor quality config")
    if (
        config.get("scientific_status") != "paired_four_seed_immediate_quality_pilot"
        or bool(config.get("formal_speed_claim"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("runtime_export_allowed"))
        or bool(config.get("formal_ood_allowed"))
        or bool(config.get("training_allowed"))
        or bool(config.get("future_repair_rounds_used"))
        or bool(config.get("cost_to_go_used"))
        or bool(config.get("runtime_used_in_label"))
    ):
        raise ValueError("topology-anchor quality must remain immediate and non-promoting")
    if (
        config.get("pilot_id") != "stride-topoanchor-quality-pilot-v1"
        or config.get("candidate_generator_id") != "stride-topoanchor-v1"
        or config.get("baseline_pool_id") != "target-collision-random-v2-frozen"
        or int(config.get("expected_state_count", -1)) != 16
        or int(config.get("expected_candidate_count", -1)) != 360
        or int(config.get("expected_base_candidate_count", -1)) != 288
        or int(config.get("expected_anchor_only_candidate_count", -1)) != 72
        or int(config.get("expected_outcome_count", -1)) != 1440
        or tuple(map(int, config.get("trial_indices") or ())) != (0, 1, 2, 3)
        or tuple(map(int, config.get("first_half_indices") or ())) != (0, 1)
        or tuple(map(int, config.get("second_half_indices") or ())) != (2, 3)
        or int(config.get("workers", 0)) != 4
    ):
        raise ValueError("topology-anchor quality cohort or identity changed")
    if dict(config.get("expected_state_count_by_group") or {}) != {
        "dao_ultra_bottleneck": 5,
        "dao_articulated": 8,
        "dao_low_articulation_control": 3,
    }:
        raise ValueError("topology-anchor quality group registry changed")
    if dict(config.get("label") or {}) != {
        "id": "stride-topoanchor-mean-np100-v1",
        "mode": "mean_np100_current_step",
        "structure_weight": 0.02,
        "no_progress_penalty": 0.10,
        "primary_term": "normalized_current_conflict_reduction",
        "paired_seed_scope": "same_state_and_trial_index_for_every_candidate",
    }:
        raise ValueError("topology-anchor quality label changed")
    if dict(config.get("pilot_gates") or {}) != {
        "minimum_augmented_pool_strict_win_rate": 0.20,
        "minimum_mean_normalized_augmented_pool_gain": 0.01,
        "minimum_anchor_top3_state_rate": 0.35,
        "maximum_mean_anchor_best_normalized_regret": 0.15,
        "minimum_strict_wins_by_group": {
            "dao_ultra_bottleneck": 1,
            "dao_articulated": 1,
            "dao_low_articulation_control": 0,
        },
    }:
        raise ValueError("topology-anchor quality gates changed")
    if not bool(config.get("four_seed_uncertainty_is_diagnostic_only")):
        raise ValueError("four-seed uncertainty may not promote a model")
    if set(config.get("inputs") or {}) != {
        "coverage_report",
        "coverage_state_rows",
        "coverage_candidate_rows",
        "dataset_manifest",
        "qualification_manifest",
        "runtime_config",
    }:
        raise ValueError("topology-anchor quality input registry changed")


def _pilot_artifact_valid(
    payload: dict[str, Any], *, identity: str, state_id: str,
    candidate_ids: list[str], trial_indices: tuple[int, ...],
) -> bool:
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("identity") != identity
        or payload.get("state_id") != state_id
        or payload.get("complete") is not True
        or list(payload.get("candidate_ids") or []) != candidate_ids
    ):
        return False
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return False
    expected = {
        (candidate_id, trial_index)
        for candidate_id in candidate_ids
        for trial_index in trial_indices
    }
    observed = {
        (str(row.get("candidate_id")), int(row.get("trial_index", -1)))
        for row in trials
        if isinstance(row, dict)
    }
    return observed == expected and len(trials) == len(expected)


def _collect_topoanchor_quality_state(job: dict[str, Any]) -> dict[str, Any]:
    state_row = dict(job["state_row"])
    candidates = list(job["candidates"])
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    trial_indices = tuple(map(int, job["trial_indices"]))
    output_path = Path(str(job["output_path"]))
    identity = str(job["identity"])
    state_id = str(state_row["state_id"])
    if bool(job["resume"]) and output_path.is_file():
        existing = _read_json(output_path)
        if _pilot_artifact_valid(
            existing,
            identity=identity,
            state_id=state_id,
            candidate_ids=candidate_ids,
            trial_indices=trial_indices,
        ):
            return {
                "state_id": state_id,
                "status": "resumed",
                "output_path": str(output_path),
                "trial_count": len(existing["trials"]),
            }
        raise ValueError(f"invalid completed topology-anchor Pilot state: {output_path}")
    environment_config = dict(job["environment"])
    environment_config["max_repair_iterations"] = max(
        1, int(environment_config.get("max_repair_iterations", 0))
    )
    replay = {
        "dataset_root": str(job["dataset_root"]),
        "row": dict(job["dataset_row"]),
        "environment": environment_config,
        "solver_seed": int(state_row["solver_seed"]),
        "replay_destroy_strategy": "Adaptive",
    }
    _, initial = replay_prefix(replay, [])
    before_fingerprint = state_fingerprint(initial)
    if before_fingerprint != str(state_row["state_fingerprint"]):
        raise RuntimeError(f"topology-anchor Pilot replay mismatch: {state_id}")
    before_repair = repair_structure_fingerprint(initial)
    before_conflicts = int(initial["num_of_colliding_pairs"])
    if before_conflicts != int(state_row["initial_conflicts"]) or before_conflicts <= 0:
        raise RuntimeError(f"topology-anchor Pilot conflict mismatch: {state_id}")
    trials: list[dict[str, Any]] = []
    for trial_index in trial_indices:
        pp_seed = stride_extended_pp_seed(before_repair, trial_index)
        ordered = candidates if trial_index % 2 == 0 else list(reversed(candidates))
        for candidate in ordered:
            branch_environment, branch = replay_prefix(replay, [])
            if state_fingerprint(branch) != before_fingerprint:
                raise RuntimeError("topology-anchor paired branch replay changed")
            agents = list(map(int, candidate["agents"]))
            result = _plain(branch_environment.step(_paired_action(agents, pp_seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_repair = repair_structure_fingerprint(after)
            trials.append(
                {
                    "schema": TRIAL_SCHEMA,
                    "state_id": state_id,
                    "task_id": str(state_row["task_id"]),
                    "map_id": str(state_row["map_id"]),
                    "layout_family": str(state_row["layout_family"]),
                    "solver_seed": int(state_row["solver_seed"]),
                    "candidate_id": str(candidate["candidate_id"]),
                    "candidate_kind": str(candidate["candidate_kind"]),
                    "selection_families": list(candidate["selection_families"]),
                    "actual_size": int(candidate["actual_size"]),
                    "before_conflicts": before_conflicts,
                    "before_fingerprint": before_fingerprint,
                    "before_repair_fingerprint": before_repair,
                    "trial_index": trial_index,
                    "pp_seed": pp_seed,
                    "replan_success": bool(metrics["replan_success"]),
                    "feasible": bool(after.get("feasible")),
                    "conflicts_after": conflicts_after,
                    "after_fingerprint": state_fingerprint(after),
                    "after_repair_fingerprint": after_repair,
                    "repair_outcome": classify_repair_outcome(
                        before_fingerprint=before_repair,
                        after_fingerprint=after_repair,
                        replan_success=bool(metrics["replan_success"]),
                        conflicts_before=before_conflicts,
                        conflicts_after=conflicts_after,
                        feasible=bool(after.get("feasible")),
                    ),
                    "post_structure": post_structure_metrics(after),
                    "native_step_seconds": float(metrics["native_step_seconds"]),
                    "pp_replan_seconds": float(metrics.get("pp_replan_seconds", 0.0)),
                }
            )
    trials.sort(key=lambda row: (int(row["trial_index"]), str(row["candidate_id"])))
    payload = {
        "schema": STATE_SCHEMA,
        "identity": identity,
        "complete": True,
        "state_id": state_id,
        "state": state_row,
        "candidate_ids": candidate_ids,
        "trials": trials,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".partial")
    _write_json(partial, payload)
    os.replace(partial, output_path)
    return {
        "state_id": state_id,
        "status": "ok",
        "output_path": str(output_path),
        "trial_count": len(trials),
    }


def _quality_inputs(
    config_path: Path, config: dict[str, Any]
) -> tuple[Path, dict[str, Path]]:
    project_root = config_path.parents[1]
    return project_root, {
        name: _quality_registered_path(project_root, dict(artifact))
        for name, artifact in config["inputs"].items()
    }


def _selected_quality_rows(
    config: dict[str, Any], inputs: dict[str, Path]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    state_rows = [
        row
        for row in _read_jsonl(inputs["coverage_state_rows"])
        if bool(row["articulation_relevant"]) or bool(row["low_degree_relevant"])
    ]
    candidate_rows: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    base_prefixes = tuple(config["candidate_families"]["base_prefixes"])
    anchor_prefix = str(config["candidate_families"]["anchor_prefix"])
    selected_ids = {str(row["state_id"]) for row in state_rows}
    for row in _read_jsonl(inputs["coverage_candidate_rows"]):
        state_id = str(row["state_id"])
        if state_id not in selected_ids:
            continue
        families = list(map(str, row["selection_families"]))
        is_base = any(family.startswith(base_prefixes) for family in families)
        is_anchor = any(family.startswith(anchor_prefix) for family in families)
        candidate_rows[state_id].append(
            {
                **row,
                "candidate_kind": (
                    "base_and_anchor"
                    if is_base and is_anchor
                    else "base"
                    if is_base
                    else "anchor_only"
                    if is_anchor
                    else "unknown"
                ),
            }
        )
    state_rows.sort(key=lambda row: str(row["state_id"]))
    for rows in candidate_rows.values():
        rows.sort(key=lambda row: str(row["candidate_id"]))
    return state_rows, dict(candidate_rows)


def collect_topology_anchor_quality_pilot(
    config_path: str | Path, output: str | Path, *, resume: bool = True
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_topology_anchor_quality_config(config)
    project_root, inputs = _quality_inputs(config_path, config)
    if _read_json(inputs["coverage_report"]).get("passed") is not True:
        raise ValueError("quality Pilot requires passed topology-anchor coverage")
    state_rows, candidates_by_state = _selected_quality_rows(config, inputs)
    dataset_rows = {
        str(row["task_id"]): row for row in _read_jsonl(inputs["dataset_manifest"])
    }
    qualification = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in _read_jsonl(inputs["qualification_manifest"])
    }
    runtime = _read_json(inputs["runtime_config"])
    group_counts = Counter(str(row["layout_family"]) for row in state_rows)
    candidate_count = sum(len(candidates_by_state.get(str(row["state_id"]), [])) for row in state_rows)
    base_count = sum(
        candidate["candidate_kind"] in {"base", "base_and_anchor"}
        for rows in candidates_by_state.values()
        for candidate in rows
    )
    anchor_only_count = sum(
        candidate["candidate_kind"] == "anchor_only"
        for rows in candidates_by_state.values()
        for candidate in rows
    )
    if (
        len(state_rows) != int(config["expected_state_count"])
        or dict(group_counts) != dict(config["expected_state_count_by_group"])
        or candidate_count != int(config["expected_candidate_count"])
        or base_count != int(config["expected_base_candidate_count"])
        or anchor_only_count != int(config["expected_anchor_only_candidate_count"])
        or set(candidates_by_state) != {str(row["state_id"]) for row in state_rows}
    ):
        raise ValueError("topology-anchor quality selected cohort differs")
    identity_payload = {
        "schema": COLLECTION_SCHEMA,
        "config_sha256": sha256_file(config_path),
        "state_rows": state_rows,
        "candidate_ids_by_state": {
            state_id: [str(row["candidate_id"]) for row in rows]
            for state_id, rows in sorted(candidates_by_state.items())
        },
        "trial_indices": list(config["trial_indices"]),
    }
    identity = _fingerprint(identity_payload)
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "run_config.json", {**identity_payload, "identity": identity})
    jobs = []
    for row in state_rows:
        state_id = str(row["state_id"])
        task_id = str(row["task_id"])
        key = (task_id, int(row["solver_seed"]))
        if task_id not in dataset_rows or key not in qualification:
            raise ValueError(f"topology-anchor quality replay input missing: {state_id}")
        if str(qualification[key]["state_fingerprint"]) != str(row["state_fingerprint"]):
            raise ValueError(f"topology-anchor quality qualification differs: {state_id}")
        jobs.append(
            {
                "state_row": row,
                "candidates": candidates_by_state[state_id],
                "trial_indices": config["trial_indices"],
                "dataset_root": str((project_root / str(config["dataset_root"])).resolve()),
                "dataset_row": dataset_rows[task_id],
                "environment": runtime["environment"],
                "identity": identity,
                "resume": bool(resume),
                "output_path": str(
                    output_root / "states" / f"{_fingerprint({'state_id': state_id})[:20]}.json"
                ),
            }
        )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=int(config["workers"])
    ) as executor:
        futures = {
            executor.submit(_collect_topoanchor_quality_state, job): job for job in jobs
        }
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
                output_root / "collection_status.json",
                {
                    "schema": COLLECTION_SCHEMA,
                    "identity": identity,
                    "requested_state_count": len(jobs),
                    "completed_state_count": len(results),
                    "error_state_count": len(errors),
                    "errors": errors,
                    "status": "running",
                },
            )
    all_trials = []
    for result in sorted(results, key=lambda row: str(row["state_id"])):
        payload = _read_json(Path(str(result["output_path"])))
        state_id = str(result["state_id"])
        if not _pilot_artifact_valid(
            payload,
            identity=identity,
            state_id=state_id,
            candidate_ids=[str(row["candidate_id"]) for row in candidates_by_state[state_id]],
            trial_indices=tuple(map(int, config["trial_indices"])),
        ):
            errors.append({"state_id": state_id, "error": "invalid completed artifact"})
        else:
            all_trials.extend(payload["trials"])
    complete = len(results) == len(jobs) and not errors
    if complete:
        all_trials.sort(
            key=lambda row: (
                str(row["state_id"]), int(row["trial_index"]), str(row["candidate_id"])
            )
        )
        _write_jsonl(output_root / "repair_trials.jsonl", all_trials)
    report = {
        "schema": COLLECTION_SCHEMA,
        "identity": identity,
        "requested_state_count": len(jobs),
        "completed_state_count": len(results),
        "error_state_count": len(errors),
        "errors": errors,
        "candidate_count": candidate_count,
        "trial_count": len(all_trials),
        "expected_trial_count": int(config["expected_outcome_count"]),
        "new_state_count": sum(row["status"] == "ok" for row in results),
        "resumed_state_count": sum(row["status"] == "resumed" for row in results),
        "complete": complete and len(all_trials) == int(config["expected_outcome_count"]),
    }
    _write_json(output_root / "collection_report.json", report)
    _write_json(
        output_root / "collection_status.json",
        {**report, "status": "complete" if report["complete"] else "error"},
    )
    return report


def _aggregate_quality_scores(
    trials: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, dict[int | str, float | bool]]:
    by_index: defaultdict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    before = int(trials[0]["before_conflicts"])
    for row in trials:
        by_index[int(row["trial_index"])][str(row["candidate_id"])] = row
    expected_indices = set(map(int, config["trial_indices"]))
    if set(by_index) != expected_indices:
        raise ValueError("topology-anchor quality trial indices incomplete")
    candidate_sets = [set(rows) for rows in by_index.values()]
    if any(values != candidate_sets[0] for values in candidate_sets[1:]):
        raise ValueError("topology-anchor quality candidate pool changed across seeds")
    scores: dict[str, dict[int | str, float | bool]] = {
        candidate_id: {} for candidate_id in candidate_sets[0]
    }
    for trial_index, outcomes in sorted(by_index.items()):
        seeds = {int(row["pp_seed"]) for row in outcomes.values()}
        if len(seeds) != 1:
            raise ValueError("topology-anchor quality PP seeds are not paired")
        structural_rows = [
            {
                "candidate_id": candidate_id,
                "mean_post_structure": dict(row["post_structure"]),
            }
            for candidate_id, row in outcomes.items()
        ]
        assign_structure_scores(structural_rows)
        structural = {
            str(row["candidate_id"]): float(row["structural_score"])
            for row in structural_rows
        }
        for candidate_id, row in outcomes.items():
            after = int(row["conflicts_after"])
            scores[candidate_id][trial_index] = (
                (before - after) / max(1, before)
                - float(config["label"]["structure_weight"]) * structural[candidate_id]
            )
            scores[candidate_id][f"progress_{trial_index}"] = after < before
    return scores


def _quality_ranking(
    scores: dict[str, dict[int | str, float | bool]],
    indices: list[int],
    no_progress_penalty: float,
) -> tuple[dict[str, float], list[str]]:
    aggregated = {}
    for candidate_id, values in scores.items():
        seed_scores = [float(values[index]) for index in indices]
        progress = [bool(values[f"progress_{index}"]) for index in indices]
        aggregated[candidate_id] = statistics.fmean(seed_scores) - no_progress_penalty * (
            1.0 - statistics.fmean(map(float, progress))
        )
    ranking = sorted(aggregated, key=lambda value: (-aggregated[value], value))
    return aggregated, ranking


def analyze_topology_anchor_quality_pilot(
    config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = _read_json(config_path)
    validate_topology_anchor_quality_config(config)
    _, inputs = _quality_inputs(config_path, config)
    collection_root = Path(collection).resolve()
    collection_report = _read_json(collection_root / "collection_report.json")
    trials_path = collection_root / "repair_trials.jsonl"
    trials = _read_jsonl(trials_path)
    by_state: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        if row.get("schema") != TRIAL_SCHEMA:
            raise ValueError("unexpected topology-anchor quality trial schema")
        by_state[str(row["state_id"])].append(row)
    state_rows, candidate_rows = _selected_quality_rows(config, inputs)
    state_metadata = {str(row["state_id"]): row for row in state_rows}
    if set(by_state) != set(state_metadata):
        raise ValueError("topology-anchor quality analyzed state set differs")
    state_reports = []
    pair_agreements = 0
    pair_count = 0
    top3_overlaps = []
    winner_agreements = 0
    cross_half_regrets = []
    for state_id in sorted(by_state):
        rows = by_state[state_id]
        scores = _aggregate_quality_scores(rows, config)
        kinds = {
            str(row["candidate_id"]): str(row["candidate_kind"])
            for row in candidate_rows[state_id]
        }
        base_ids = sorted(
            candidate_id
            for candidate_id, kind in kinds.items()
            if kind in {"base", "base_and_anchor"}
        )
        anchor_ids = sorted(
            candidate_id for candidate_id, kind in kinds.items() if kind == "anchor_only"
        )
        full_scores, full_rank = _quality_ranking(
            scores,
            list(map(int, config["trial_indices"])),
            float(config["label"]["no_progress_penalty"]),
        )
        halves = [
            _quality_ranking(
                scores,
                list(map(int, config[name])),
                float(config["label"]["no_progress_penalty"]),
            )
            for name in ("first_half_indices", "second_half_indices")
        ]
        first_scores, first_rank = halves[0]
        second_scores, second_rank = halves[1]
        state_pair_count = 0
        state_pair_agreement = 0
        for left, right in combinations(sorted(scores), 2):
            first_delta = first_scores[left] - first_scores[right]
            second_delta = second_scores[left] - second_scores[right]
            if abs(first_delta) <= 1e-12 and abs(second_delta) <= 1e-12:
                continue
            state_pair_count += 1
            state_pair_agreement += int(first_delta * second_delta > 1e-24)
        top3_overlap = len(set(first_rank[:3]) & set(second_rank[:3])) / 3.0
        half_regrets = []
        for selected, evaluation in (
            (first_rank[0], second_scores),
            (second_rank[0], first_scores),
        ):
            span = max(evaluation.values()) - min(evaluation.values())
            half_regrets.append(
                (max(evaluation.values()) - evaluation[selected]) / span
                if span > 1e-12
                else 0.0
            )
        best_all = full_rank[0]
        best_base = max(base_ids, key=lambda value: (full_scores[value], value))
        best_anchor = max(anchor_ids, key=lambda value: (full_scores[value], value))
        span = max(full_scores.values()) - min(full_scores.values())
        pool_gain = full_scores[best_all] - full_scores[best_base]
        normalized_gain = pool_gain / span if span > 1e-12 else 0.0
        anchor_regret = full_scores[best_all] - full_scores[best_anchor]
        normalized_anchor_regret = anchor_regret / span if span > 1e-12 else 0.0
        strict_win = best_all in anchor_ids and pool_gain > 1e-12
        anchor_top3 = any(candidate_id in anchor_ids for candidate_id in full_rank[:3])
        pair_agreements += state_pair_agreement
        pair_count += state_pair_count
        top3_overlaps.append(top3_overlap)
        winner_agreements += first_rank[0] == second_rank[0]
        cross_half_regrets.extend(half_regrets)
        state_reports.append(
            {
                "state_id": state_id,
                "map_id": str(state_metadata[state_id]["map_id"]),
                "layout_family": str(state_metadata[state_id]["layout_family"]),
                "candidate_count": len(scores),
                "base_candidate_count": len(base_ids),
                "anchor_only_candidate_count": len(anchor_ids),
                "best_all_candidate_id": best_all,
                "best_base_candidate_id": best_base,
                "best_anchor_candidate_id": best_anchor,
                "augmented_pool_strict_win": strict_win,
                "normalized_augmented_pool_gain": normalized_gain,
                "anchor_in_top3": anchor_top3,
                "anchor_best_normalized_regret": normalized_anchor_regret,
                "half_pairwise_consistency": (
                    state_pair_agreement / state_pair_count if state_pair_count else 1.0
                ),
                "half_top3_overlap": top3_overlap,
                "half_winner_agreement": first_rank[0] == second_rank[0],
                "cross_half_normalized_regret": statistics.fmean(half_regrets),
            }
        )
    state_count = len(state_reports)
    strict_wins = sum(bool(row["augmented_pool_strict_win"]) for row in state_reports)
    group_reports = {}
    for group in config["expected_state_count_by_group"]:
        rows = [row for row in state_reports if row["layout_family"] == group]
        group_reports[group] = {
            "state_count": len(rows),
            "strict_win_count": sum(bool(row["augmented_pool_strict_win"]) for row in rows),
            "strict_win_rate": (
                sum(bool(row["augmented_pool_strict_win"]) for row in rows) / len(rows)
                if rows
                else 0.0
            ),
            "mean_normalized_augmented_pool_gain": statistics.fmean(
                float(row["normalized_augmented_pool_gain"]) for row in rows
            ) if rows else 0.0,
        }
    summary = {
        "augmented_pool_strict_win_count": strict_wins,
        "augmented_pool_strict_win_rate": strict_wins / state_count if state_count else 0.0,
        "mean_normalized_augmented_pool_gain": statistics.fmean(
            float(row["normalized_augmented_pool_gain"]) for row in state_reports
        ) if state_reports else 0.0,
        "anchor_top3_state_rate": statistics.fmean(
            float(row["anchor_in_top3"]) for row in state_reports
        ) if state_reports else 0.0,
        "mean_anchor_best_normalized_regret": statistics.fmean(
            float(row["anchor_best_normalized_regret"]) for row in state_reports
        ) if state_reports else 0.0,
    }
    uncertainty = {
        "half_pairwise_consistency": pair_agreements / pair_count if pair_count else 0.0,
        "mean_half_top3_overlap": statistics.fmean(top3_overlaps) if top3_overlaps else 0.0,
        "half_winner_agreement_rate": winner_agreements / state_count if state_count else 0.0,
        "mean_cross_half_normalized_regret": statistics.fmean(cross_half_regrets)
        if cross_half_regrets else 0.0,
        "diagnostic_only": True,
    }
    gates_config = config["pilot_gates"]
    gates = {
        "collection_complete": collection_report.get("complete") is True,
        "zero_collection_errors": int(collection_report.get("error_state_count", -1)) == 0,
        "expected_state_count": state_count == int(config["expected_state_count"]),
        "expected_candidate_count": sum(row["candidate_count"] for row in state_reports)
        == int(config["expected_candidate_count"]),
        "expected_outcome_count": len(trials) == int(config["expected_outcome_count"]),
        "minimum_augmented_pool_strict_win_rate": summary[
            "augmented_pool_strict_win_rate"
        ] >= float(gates_config["minimum_augmented_pool_strict_win_rate"]),
        "minimum_mean_normalized_augmented_pool_gain": summary[
            "mean_normalized_augmented_pool_gain"
        ] >= float(gates_config["minimum_mean_normalized_augmented_pool_gain"]),
        "minimum_anchor_top3_state_rate": summary["anchor_top3_state_rate"]
        >= float(gates_config["minimum_anchor_top3_state_rate"]),
        "maximum_mean_anchor_best_normalized_regret": summary[
            "mean_anchor_best_normalized_regret"
        ] <= float(gates_config["maximum_mean_anchor_best_normalized_regret"]),
    }
    for group, minimum in gates_config["minimum_strict_wins_by_group"].items():
        gates[f"{group}_minimum_strict_wins"] = (
            int(group_reports[group]["strict_win_count"]) >= int(minimum)
        )
    passed = all(gates.values())
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": "paired_four_seed_immediate_quality_pilot",
        "formal_speed_claim": False,
        "training_allowed": False,
        "runtime_used_in_label": False,
        "future_repair_rounds_used": False,
        "cost_to_go_used": False,
        "state_count": state_count,
        "candidate_count": sum(row["candidate_count"] for row in state_reports),
        "outcome_count": len(trials),
        "summary": summary,
        "action_uncertainty": uncertainty,
        "topology_groups": group_reports,
        "states": state_reports,
        "gates": gates,
        "passed": passed,
        "next_decision": config[
            "next_decision_on_pass" if passed else "next_decision_on_failure"
        ],
        "inputs": {
            "config_sha256": sha256_file(config_path),
            "collection_report_sha256": sha256_file(
                collection_root / "collection_report.json"
            ),
            "repair_trials_sha256": sha256_file(trials_path),
            **{
                f"{name}_sha256": sha256_file(path)
                for name, path in sorted(inputs.items())
            },
        },
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "topology_anchor_quality_report.json", report)
    return report


__all__ = [
    "analyze_topology_anchor_quality_pilot",
    "collect_topology_anchor_quality_pilot",
    "validate_topology_anchor_quality_config",
]
