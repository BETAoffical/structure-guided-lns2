from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.repair_collection import (
    _fingerprint,
    _load_dataset_rows,
    _plain,
    _read_json,
    _read_jsonl,
    _run_jobs,
    _write_json,
    _write_jsonl,
    state_fingerprint,
)
from experiments.stride_collection import _paired_action, _validate_native_repair
from experiments.stride_repairability import _robust_winner
from experiments.stride_repairability_collection import (
    repairability_pp_seed,
    repairability_restore_seed,
)
from experiments.stride_robuststep_preflight import _mean
from experiments.stride_stage3 import _project_path
from experiments.trace_replay import restore_repair_state, target_state_from_trace
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint


CONFIG_SCHEMA = "lns2.stride.maprank_override_counterfactual_config.v1"
STATE_SCHEMA = "lns2.stride.maprank_override_counterfactual_state.v1"
TRIAL_SCHEMA = "lns2.stride.maprank_override_counterfactual_trial.v1"
REPORT_SCHEMA = "lns2.stride.maprank_override_counterfactual_report.v1"
COLLECTION_SCHEMA = "lns2.stride.maprank_override_counterfactual_collection.v1"
STATUS_SCHEMA = "lns2.stride.maprank_override_counterfactual_status.v1"
PRODUCER_FILES = (
    "CMakeLists.txt",
    "experiments/repair_collection.py",
    "experiments/stride_collection.py",
    "experiments/stride_maprank_counterfactual.py",
    "experiments/stride_repairability.py",
    "experiments/stride_repairability_collection.py",
    "experiments/trace_replay.py",
    "lns2_selector/runtime/fingerprints.py",
    "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/RepairPolicy.h",
    "third_party/mapf_lns2/src/InitLNS.cpp",
)


def _load_config(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("MapRank override-counterfactual schema changed")
    if (
        config.get("state_selection") != "all_and_only_first_override_states"
        or int(config.get("expected_state_count", 0)) != 6
        or list(config.get("candidate_selection") or ())
        != ["v2_augmented_selected", "maprank_selected"]
        or int(config.get("expected_candidate_count_per_state", 0)) != 2
        or list(map(int, config.get("trial_indices") or ())) != list(range(16))
        or config.get("paired_pp_seeds_within_state_and_trial_index") is not True
    ):
        raise ValueError("MapRank override-counterfactual paired design changed")
    if (
        config.get("primary_outcome")
        != "current_step_conflict_reduction_normalized_by_before_conflicts"
        or config.get("runtime_used_in_label") is not False
        or config.get("repair_runtime_descriptive_only") is not True
        or config.get("future_trajectory_read") is not False
        or config.get("fresh_map_data_allowed") is not False
        or config.get("post_hoc_diagnostic_only") is not True
    ):
        raise ValueError("MapRank override-counterfactual evidence boundary changed")
    robust = dict(config.get("robust_pair_rule") or {})
    if robust != {
        "minimum_paired_win_fraction": 0.75,
        "minimum_absolute_mean_effect": 0.02,
        "require_both_half_mean_directions": True,
        "tie_epsilon": 1e-12,
    }:
        raise ValueError("MapRank override-counterfactual robust rule changed")
    for key in ("evaluation_config", "label_design"):
        artifact = _project_path(root, str(config[key]))
        if sha256_file(artifact) != config.get(f"{key}_sha256"):
            raise ValueError(f"MapRank override-counterfactual {key} changed")
    source = _project_path(root, str(config["source_failure_analysis"]))
    if sha256_file(source) != config.get("source_failure_analysis_sha256"):
        raise ValueError("MapRank failure-analysis source changed")
    analysis = _read_json(source)
    if (
        analysis.get("passed") is not True
        or analysis.get("fresh_map_data_read") is not False
        or int(analysis.get("episode_count", 0)) != 16
    ):
        raise ValueError("MapRank failure-analysis source is invalid")
    return path, root, config, source, analysis


def _selected_states(
    root: Path, config: dict[str, Any], analysis: dict[str, Any]
) -> list[dict[str, Any]]:
    trace_root = _project_path(root, str(config["trace_root"]))
    selected = []
    for episode in analysis["episodes"]:
        override = episode.get("first_override")
        if not isinstance(override, dict):
            continue
        cohort = str(episode["cohort_id"])
        source_root = (
            trace_root
            / "cohorts"
            / cohort
            / "controllers"
            / str(analysis["baseline_controller"])
        )
        matches = [
            row
            for row in _read_jsonl(
                source_root / "realized_dynamic_manifest.jsonl"
            )
            if str(row["task_id"]) == str(episode["task_id"])
            and int(row["solver_seed"]) == int(episode["solver_seed"])
        ]
        if len(matches) != 1:
            raise ValueError("MapRank counterfactual source episode is ambiguous")
        manifest = matches[0]
        selected.append(
            {
                "state_id": (
                    f"{episode['task_id']}::solver_seed_{int(episode['solver_seed'])}"
                    f"::decision_{int(override['decision_index'])}"
                ),
                "cohort_id": cohort,
                "map_id": str(manifest["map_id"]),
                "task_id": str(episode["task_id"]),
                "solver_seed": int(episode["solver_seed"]),
                "decision_index": int(override["decision_index"]),
                "before_fingerprint": str(override["before_fingerprint"]),
                "before_conflicts": int(override["conflicts_before"]),
                "source_root": str(source_root),
                "source_manifest": manifest,
                "baseline_candidate_id": str(
                    override["baseline_candidate"]["candidate_id"]
                ),
                "challenger_candidate_id": str(
                    override["challenger_candidate"]["candidate_id"]
                ),
                "recorded_conflict_delta_difference": int(
                    override["immediate_effect"]["conflict_delta_difference"]
                ),
            }
        )
    selected.sort(key=lambda row: str(row["state_id"]))
    if len(selected) != int(config["expected_state_count"]):
        raise ValueError("MapRank counterfactual selected state count changed")
    return selected


def _replay_job(selection: dict[str, Any]) -> dict[str, Any]:
    source_root = Path(str(selection["source_root"])).resolve()
    run = _read_json(source_root / "run_config.json")
    dataset_root = Path(str(run["dataset"])).resolve()
    matches = [
        row
        for row in _load_dataset_rows(
            dataset_root, [str(selection["source_manifest"]["split"])]
        )
        if str(row["task_id"]) == str(selection["task_id"])
    ]
    if len(matches) != 1:
        raise ValueError("MapRank counterfactual task must resolve exactly once")
    environment = dict(run["configuration"]["environment"])
    environment["max_repair_iterations"] = max(
        int(environment.get("max_repair_iterations", 0)),
        int(selection["decision_index"]) + 1,
    )
    return {
        "dataset_root": str(dataset_root),
        "row": matches[0],
        "environment": environment,
        "solver_seed": int(selection["solver_seed"]),
        "replay_destroy_strategy": "Adaptive",
    }


def _target_candidates(
    source_root: Path, manifest: dict[str, Any], selection: dict[str, Any]
) -> list[dict[str, Any]]:
    trace_path = source_root / str(manifest["trace_file"])
    events = read_trace_events(trace_path)
    transitions = {
        int(event["decision_index"]): event
        for event in events
        if event.get("event") == "transition"
    }
    transition = transitions.get(int(selection["decision_index"]))
    if transition is None:
        raise ValueError("MapRank counterfactual target transition is missing")
    if str(transition.get("before_fingerprint")) != str(
        selection["before_fingerprint"]
    ):
        raise ValueError("MapRank counterfactual target fingerprint changed")
    pool = {
        str(row["candidate_id"]): dict(row)
        for row in transition["controller"]["candidate_pool"]
    }
    result = []
    for role, field in (
        ("v2_augmented_selected", "baseline_candidate_id"),
        ("maprank_selected", "challenger_candidate_id"),
    ):
        candidate_id = str(selection[field])
        candidate = pool.get(candidate_id)
        if candidate is None:
            raise ValueError("MapRank selected counterfactual candidate disappeared")
        result.append(
            {
                "role": role,
                "candidate_id": candidate_id,
                "agents": sorted(map(int, candidate["agents"])),
                "actual_size": int(candidate["actual_size"]),
                "selection_families": sorted(
                    map(str, candidate.get("selection_families") or ())
                ),
            }
        )
    if result[0]["candidate_id"] == result[1]["candidate_id"]:
        raise ValueError("MapRank counterfactual requires an actual override")
    return result


def _state_artifact_valid(
    payload: dict[str, Any], *, run_fingerprint: str, state_id: str
) -> bool:
    if (
        payload.get("schema") != STATE_SCHEMA
        or payload.get("run_fingerprint") != run_fingerprint
        or payload.get("state_id") != state_id
        or payload.get("complete") is not True
    ):
        return False
    candidates = list(payload.get("candidates") or ())
    trials = list(payload.get("trials") or ())
    expected = {
        (str(candidate["candidate_id"]), trial_index)
        for candidate in candidates
        for trial_index in range(16)
    }
    observed = {
        (str(row.get("candidate_id")), int(row.get("trial_index", -1)))
        for row in trials
    }
    return (
        len(candidates) == 2
        and len(trials) == 32
        and observed == expected
        and all(row.get("schema") == TRIAL_SCHEMA for row in trials)
    )


def _collect_state(job: dict[str, Any]) -> dict[str, Any]:
    selection = dict(job["selection"])
    output_path = Path(str(job["output_path"]))
    run_fingerprint = str(job["run_fingerprint"])
    if bool(job["resume"]) and output_path.is_file():
        payload = _read_json(output_path)
        if _state_artifact_valid(
            payload,
            run_fingerprint=run_fingerprint,
            state_id=str(selection["state_id"]),
        ):
            return {
                "job_id": str(selection["state_id"]),
                "state_id": str(selection["state_id"]),
                "status": "resumed",
                "candidate_count": 2,
                "trial_count": 32,
                "error_count": 0,
            }
        raise ValueError("MapRank counterfactual resume artifact is invalid")
    source_root = Path(str(selection["source_root"])).resolve()
    manifest = dict(selection["source_manifest"])
    state, trace_path = target_state_from_trace(
        source_root,
        manifest,
        decision_index=int(selection["decision_index"]),
        expected_fingerprint=str(selection["before_fingerprint"]),
    )
    if (
        state_fingerprint(state) != str(selection["before_fingerprint"])
        or int(state["num_of_colliding_pairs"])
        != int(selection["before_conflicts"])
    ):
        raise RuntimeError("MapRank counterfactual reconstructed state changed")
    replay = _replay_job(selection)
    repair_fingerprint = repair_structure_fingerprint(state)
    restore_seed = repairability_restore_seed(repair_fingerprint)
    candidates = _target_candidates(source_root, manifest, selection)
    trials = []
    before_conflicts = int(selection["before_conflicts"])
    for candidate in candidates:
        agents = list(map(int, candidate["agents"]))
        for trial_index in map(int, job["trial_indices"]):
            environment, restored = restore_repair_state(
                replay, state, seed=restore_seed
            )
            if repair_structure_fingerprint(restored) != repair_fingerprint:
                raise RuntimeError("MapRank counterfactual branch restore changed")
            before_low_level = dict(restored.get("low_level") or {})
            pp_seed = repairability_pp_seed(repair_fingerprint, trial_index)
            result = _plain(environment.step(_paired_action(agents, pp_seed)))
            after, metrics = _validate_native_repair(
                result, expected_agents=agents, expected_seed=pp_seed
            )
            conflicts_after = int(after["num_of_colliding_pairs"])
            after_low_level = dict(after.get("low_level") or {})
            trials.append(
                {
                    "schema": TRIAL_SCHEMA,
                    "state_id": str(selection["state_id"]),
                    "candidate_role": str(candidate["role"]),
                    "candidate_id": str(candidate["candidate_id"]),
                    "trial_index": trial_index,
                    "pp_seed": pp_seed,
                    "before_conflicts": before_conflicts,
                    "conflicts_after": conflicts_after,
                    "normalized_conflict_reduction": (
                        before_conflicts - conflicts_after
                    )
                    / max(1, before_conflicts),
                    "progress": conflicts_after < before_conflicts,
                    "feasible": bool(after.get("feasible")),
                    "replan_success": bool(metrics["replan_success"]),
                    "low_level_generated": int(
                        after_low_level.get("generated", 0)
                    )
                    - int(before_low_level.get("generated", 0)),
                    "low_level_expanded": int(after_low_level.get("expanded", 0))
                    - int(before_low_level.get("expanded", 0)),
                    "pp_replan_seconds": float(
                        metrics.get("pp_replan_seconds", 0.0)
                    ),
                }
            )
    seeds_by_index = {
        trial_index: {
            int(row["pp_seed"])
            for row in trials
            if int(row["trial_index"]) == trial_index
        }
        for trial_index in map(int, job["trial_indices"])
    }
    if any(len(seeds) != 1 for seeds in seeds_by_index.values()):
        raise RuntimeError("MapRank counterfactual PP seeds are not paired")
    payload = {
        "schema": STATE_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "complete": True,
        "state_id": str(selection["state_id"]),
        "selection": selection,
        "source_trace_file": str(trace_path),
        "before_fingerprint": str(selection["before_fingerprint"]),
        "before_repair_fingerprint": repair_fingerprint,
        "before_conflicts": before_conflicts,
        "restore_seed": restore_seed,
        "candidates": candidates,
        "trials": trials,
    }
    _write_json(output_path, payload)
    return {
        "job_id": str(selection["state_id"]),
        "state_id": str(selection["state_id"]),
        "status": "ok",
        "candidate_count": 2,
        "trial_count": len(trials),
        "error_count": 0,
    }


def _candidate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trial_count": len(rows),
        "mean_normalized_conflict_reduction": _mean(
            [float(row["normalized_conflict_reduction"]) for row in rows]
        ),
        "conflict_reduction_std": statistics.pstdev(
            [float(row["normalized_conflict_reduction"]) for row in rows]
        ),
        "progress_rate": _mean([float(bool(row["progress"])) for row in rows]),
        "feasible_rate": _mean([float(bool(row["feasible"])) for row in rows]),
        "replan_success_rate": _mean(
            [float(bool(row["replan_success"])) for row in rows]
        ),
        "mean_low_level_generated": _mean(
            [float(row["low_level_generated"]) for row in rows]
        ),
        "mean_low_level_expanded": _mean(
            [float(row["low_level_expanded"]) for row in rows]
        ),
        "mean_pp_replan_seconds_descriptive": _mean(
            [float(row["pp_replan_seconds"]) for row in rows]
        ),
    }


def analyze_override_counterfactual(
    config_path: str | Path, collection: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config, source_path, analysis = _load_config(config_path)
    collection = Path(collection).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    states = [_read_json(path) for path in sorted((collection / "states").glob("*.json"))]
    selected = _selected_states(root, config, analysis)
    expected_ids = {str(row["state_id"]) for row in selected}
    observed_ids = {str(row.get("state_id")) for row in states}
    errors = []
    if observed_ids != expected_ids:
        errors.append("counterfactual state coverage differs")
    trial_rows = [row for state in states for row in state.get("trials", ())]
    state_reports = []
    robust = dict(config["robust_pair_rule"])
    for state in states:
        by_role = {
            role: sorted(
                [row for row in state["trials"] if row["candidate_role"] == role],
                key=lambda row: int(row["trial_index"]),
            )
            for role in config["candidate_selection"]
        }
        baseline = by_role["v2_augmented_selected"]
        challenger = by_role["maprank_selected"]
        baseline_scores = [
            float(row["normalized_conflict_reduction"]) for row in baseline
        ]
        challenger_scores = [
            float(row["normalized_conflict_reduction"]) for row in challenger
        ]
        direction, diagnostics = _robust_winner(
            baseline_scores,
            challenger_scores,
            minimum_win_fraction=float(robust["minimum_paired_win_fraction"]),
            minimum_effect=float(robust["minimum_absolute_mean_effect"]),
            tie_epsilon=float(robust["tie_epsilon"]),
        )
        selection = dict(state["selection"])
        state_reports.append(
            {
                "state_id": str(state["state_id"]),
                "cohort_id": str(selection["cohort_id"]),
                "task_id": str(selection["task_id"]),
                "solver_seed": int(selection["solver_seed"]),
                "decision_index": int(selection["decision_index"]),
                "before_conflicts": int(state["before_conflicts"]),
                "baseline_candidate": dict(state["candidates"][0]),
                "challenger_candidate": dict(state["candidates"][1]),
                "baseline_summary": _candidate_summary(baseline),
                "challenger_summary": _candidate_summary(challenger),
                "maprank_minus_v2_mean_conflict_reduction": _mean(
                    challenger_scores
                )
                - _mean(baseline_scores),
                "robust_winner": (
                    "v2_augmented_selected"
                    if direction > 0
                    else "maprank_selected" if direction < 0 else "uncertain"
                ),
                "robust_pair_diagnostics": diagnostics,
                "seedwise_maprank_better_count": sum(
                    right > left + float(robust["tie_epsilon"])
                    for left, right in zip(baseline_scores, challenger_scores)
                ),
                "seedwise_equal_count": sum(
                    abs(right - left) <= float(robust["tie_epsilon"])
                    for left, right in zip(baseline_scores, challenger_scores)
                ),
                "seedwise_maprank_worse_count": sum(
                    right < left - float(robust["tie_epsilon"])
                    for left, right in zip(baseline_scores, challenger_scores)
                ),
                "recorded_single_seed": {
                    "conflict_delta_difference": int(
                        selection["recorded_conflict_delta_difference"]
                    ),
                },
            }
        )
    winner_counts = Counter(row["robust_winner"] for row in state_reports)
    per_cohort = {
        cohort: {
            "state_count": len(rows),
            "robust_winner_counts": dict(
                sorted(Counter(row["robust_winner"] for row in rows).items())
            ),
            "mean_maprank_minus_v2_conflict_reduction": _mean(
                [
                    float(row["maprank_minus_v2_mean_conflict_reduction"])
                    for row in rows
                ]
            ),
        }
        for cohort in sorted({str(row["cohort_id"]) for row in state_reports})
        for rows in [[row for row in state_reports if row["cohort_id"] == cohort]]
    }
    integrity = {
        "source_failure_analysis_passed": analysis.get("passed") is True,
        "fresh_map_data_not_read": analysis.get("fresh_map_data_read") is False,
        "expected_state_coverage": observed_ids == expected_ids,
        "two_candidates_per_state": all(
            len(state.get("candidates") or ()) == 2 for state in states
        ),
        "sixteen_trials_per_candidate": len(trial_rows) == 6 * 2 * 16,
        "paired_pp_seeds": all(
            len(
                {
                    int(row["pp_seed"])
                    for row in state["trials"]
                    if int(row["trial_index"]) == trial_index
                }
            )
            == 1
            for state in states
            for trial_index in range(16)
        ),
        "runtime_excluded_from_robust_label": config.get("runtime_used_in_label")
        is False,
        "future_trajectory_not_read": config.get("future_trajectory_read") is False,
    }
    passed = not errors and all(integrity.values())
    report = {
        "schema": COLLECTION_SCHEMA,
        "experiment_id": config["experiment_id"],
        "scientific_status": "post_hoc_mechanism_diagnostic_only",
        "post_hoc_diagnostic_only": True,
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "fresh_map_data_read": False,
        "runtime_used_in_label": False,
        "future_trajectory_read": False,
        "primary_outcome": config["primary_outcome"],
        "state_count": len(states),
        "candidate_count": sum(len(state["candidates"]) for state in states),
        "trial_count": len(trial_rows),
        "summary": {
            "robust_winner_counts": dict(sorted(winner_counts.items())),
            "mean_maprank_minus_v2_conflict_reduction": _mean(
                [
                    float(row["maprank_minus_v2_mean_conflict_reduction"])
                    for row in state_reports
                ]
            ),
        },
        "per_cohort": per_cohort,
        "states": state_reports,
        "integrity_gates": integrity,
        "passed": passed,
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "source_failure_analysis_sha256": sha256_file(source_path),
            "collection_run_config_sha256": sha256_file(
                collection / "run_config.json"
            ),
            "state_artifact_sha256": {
                path.name: sha256_file(path)
                for path in sorted((collection / "states").glob("*.json"))
            },
        },
    }
    _write_jsonl(output / "counterfactual_trials.jsonl", trial_rows)
    report["inputs"]["counterfactual_trials_sha256"] = sha256_file(
        output / "counterfactual_trials.jsonl"
    )
    _write_json(output / "maprank_override_counterfactual_report.json", report)
    return report


def collect_override_counterfactual(
    config_path: str | Path,
    output: str | Path,
    *,
    resume: bool = False,
) -> dict[str, Any]:
    path, root, config, source_path, analysis = _load_config(config_path)
    selected = _selected_states(root, config, analysis)
    output = Path(output).resolve()
    producer = producer_identity(
        project_root=root,
        source_files=PRODUCER_FILES,
        native_required=True,
        package_names=("numpy",),
    )
    identity = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "config_sha256": sha256_file(path),
        "source_failure_analysis_sha256": sha256_file(source_path),
        "selected_state_ids": [str(row["state_id"]) for row in selected],
        "trial_indices": list(map(int, config["trial_indices"])),
        "producer": producer,
    }
    run_fingerprint = _fingerprint(identity)
    run_path = output / "run_config.json"
    if run_path.is_file():
        if _read_json(run_path).get("run_fingerprint") != run_fingerprint:
            raise ValueError("MapRank counterfactual output belongs to another run")
        if not resume:
            raise ValueError("MapRank counterfactual output exists; pass --resume")
    output.mkdir(parents=True, exist_ok=True)
    (output / "states").mkdir(parents=True, exist_ok=True)
    _write_json(run_path, {**identity, "run_fingerprint": run_fingerprint})
    _write_jsonl(output / "state_selection.jsonl", selected)
    jobs = []
    for selection in selected:
        key = _fingerprint(
            {
                "state_id": selection["state_id"],
                "before": selection["before_fingerprint"],
            }
        )[:20]
        jobs.append(
            {
                "job_id": str(selection["state_id"]),
                "state_id": str(selection["state_id"]),
                "row": dict(selection["source_manifest"]),
                "solver_seed": int(selection["solver_seed"]),
                "selection": selection,
                "trial_indices": list(map(int, config["trial_indices"])),
                "output_path": str(output / "states" / f"{key}.json"),
                "run_fingerprint": run_fingerprint,
                "resume": bool(resume),
            }
        )
    results = _run_jobs(
        _collect_state,
        jobs,
        int(config["workers"]),
        phase="maprank-override-counterfactual",
        output_root=output,
        run_fingerprint=run_fingerprint,
        timeout_seconds=float(config["state_timeout_seconds"]),
    )
    _write_jsonl(output / "collection_manifest.jsonl", results)
    completed_count = sum(
        row.get("status") in {"ok", "resumed"} for row in results
    )
    error_count = len(results) - completed_count
    status = {
        "schema": STATUS_SCHEMA,
        "run_fingerprint": run_fingerprint,
        "state_count": len(results),
        "completed_state_count": completed_count,
        "error_state_count": error_count,
        "complete": completed_count == len(selected) and error_count == 0,
    }
    _write_json(output / "collection_status.json", status)
    if status["complete"]:
        report = analyze_override_counterfactual(path, output, output)
        status["report_sha256"] = sha256_file(
            output / "maprank_override_counterfactual_report.json"
        )
        _write_json(output / "collection_status.json", status)
        return report
    return status


__all__ = [
    "analyze_override_counterfactual",
    "collect_override_counterfactual",
]
