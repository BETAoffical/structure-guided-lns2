from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.closed_loop_trace_storage import read_trace_events
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.stride_robuststep_preflight import _mean
from experiments.stride_stage3 import _project_path


CONFIG_SCHEMA = "lns2.stride.maprank_high_load_failure_analysis_config.v1"
REPORT_SCHEMA = "lns2.stride.maprank_high_load_failure_analysis.v1"


def _load_config(
    config_path: str | Path,
) -> tuple[Path, Path, dict[str, Any], Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    root = path.parent.parent
    config = _read_json(path)
    if config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("MapRank failure-analysis schema changed")
    if config.get("post_hoc_diagnostic_only") is not True:
        raise ValueError("MapRank failure analysis must remain diagnostic-only")
    if config.get("fresh_map_data_allowed") is not False:
        raise ValueError("MapRank failure analysis may not read fresh maps")
    if config.get("wall_ttf_used_as_training_label") is not False:
        raise ValueError("MapRank failure analysis may not make TTF a label")
    if config.get("future_trajectory_used_as_training_label") is not False:
        raise ValueError("MapRank failure analysis may not make trajectory a label")
    if int(config.get("registered_episode_count", 0)) != 16:
        raise ValueError("MapRank failure-analysis episode count changed")
    if list(config.get("registered_cohorts") or ()) != ["maze300", "room500"]:
        raise ValueError("MapRank failure-analysis cohorts changed")
    if len(config.get("timing_replicates") or ()) != 3:
        raise ValueError("MapRank failure-analysis timing replicate count changed")
    confirmation_path = _project_path(root, str(config["confirmation_report"]))
    if sha256_file(confirmation_path) != config.get("confirmation_report_sha256"):
        raise ValueError("MapRank confirmation report hash changed")
    confirmation = _read_json(confirmation_path)
    if confirmation.get("integrity_passed") is not True:
        raise ValueError("MapRank confirmation integrity did not pass")
    if confirmation.get("performance_passed") is not False:
        raise ValueError("MapRank failure analysis requires the registered failure")
    if confirmation.get("fresh_map_unlocked") is not False:
        raise ValueError("MapRank fresh maps must remain locked")
    return path, root, config, confirmation_path, confirmation


def _manifest_index(
    collection_root: Path, cohorts: list[str], controller: str
) -> tuple[dict[tuple[str, str, int], dict[str, Any]], dict[str, str]]:
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    hashes = {}
    for cohort in cohorts:
        manifest = (
            collection_root
            / "cohorts"
            / cohort
            / "controllers"
            / controller
            / "realized_dynamic_manifest.jsonl"
        )
        hashes[cohort] = sha256_file(manifest)
        for row in _read_jsonl(manifest):
            key = (cohort, str(row["task_id"]), int(row["solver_seed"]))
            if key in rows:
                raise ValueError(f"duplicate failure-analysis episode: {key}")
            rows[key] = row
    return rows, hashes


def _transitions(collection_root: Path, row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        event
        for event in read_trace_events(collection_root / str(row["trace_file"]))
        if event.get("event") == "transition"
    ]


def _candidate_index(transition: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["candidate_id"]): dict(row)
        for row in transition.get("controller", {}).get("candidate_pool", ())
    }


def _candidate_view(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(candidate["candidate_id"]),
        "actual_size": int(candidate.get("actual_size", 0)),
        "selection_families": list(candidate.get("selection_families") or ()),
        "feature_out_of_range_fraction": float(
            candidate.get("feature_out_of_range_fraction", 0.0)
        ),
    }


def _repair_view(transition: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(transition.get("metrics") or {})
    low_level = dict(transition.get("low_level_delta") or {})
    return {
        "conflicts_before": int(metrics.get("conflicts_before", 0)),
        "conflicts_after": int(metrics.get("conflicts_after", 0)),
        "conflict_delta": int(metrics.get("conflict_delta", 0)),
        "replan_success": bool(metrics.get("replan_success")),
        "pp_replan_seconds": float(metrics.get("pp_replan_seconds", 0.0)),
        "sum_of_costs_delta": int(metrics.get("sum_of_costs_after", 0))
        - int(metrics.get("sum_of_costs_before", 0)),
        "low_level_generated": int(low_level.get("generated", 0)),
        "low_level_expanded": int(low_level.get("expanded", 0)),
        "low_level_reopened": int(low_level.get("reopened", 0)),
    }


def _first_override(
    baseline: list[dict[str, Any]], challenger: list[dict[str, Any]]
) -> tuple[int, dict[str, Any] | None, list[str]]:
    errors = []
    common = 0
    for index, (left, right) in enumerate(zip(baseline, challenger)):
        if left.get("before_fingerprint") != right.get("before_fingerprint"):
            errors.append(f"state diverged before action at decision {index}")
            return common, None, errors
        left_controller = dict(left.get("controller") or {})
        right_controller = dict(right.get("controller") or {})
        left_id = str(left_controller.get("selected_candidate_id"))
        right_id = str(right_controller.get("selected_candidate_id"))
        if left_id == right_id:
            if left.get("after_fingerprint") != right.get("after_fingerprint"):
                errors.append(f"same action changed semantics at decision {index}")
                return common, None, errors
            common += 1
            continue
        left_pool = _candidate_index(left)
        right_pool = _candidate_index(right)
        if set(left_pool) != set(right_pool):
            errors.append(f"candidate pool changed at first override {index}")
            return common, None, errors
        if left_id not in left_pool or right_id not in left_pool:
            errors.append(f"selected candidate missing at first override {index}")
            return common, None, errors
        left_selected_v2 = left_pool[left_id]
        right_selected_v2 = left_pool[right_id]
        left_selected_maprank = right_pool[left_id]
        right_selected_maprank = right_pool[right_id]
        left_repair = _repair_view(left)
        right_repair = _repair_view(right)
        return (
            common,
            {
                "decision_index": index,
                "before_fingerprint": str(left["before_fingerprint"]),
                "conflicts_before": left_repair["conflicts_before"],
                "baseline_candidate": _candidate_view(left_selected_v2),
                "challenger_candidate": _candidate_view(right_selected_v2),
                "cross_model_scores": {
                    "baseline_selected_v2_score": float(
                        left_selected_v2.get("score", 0.0)
                    ),
                    "challenger_selected_v2_score": float(
                        right_selected_v2.get("score", 0.0)
                    ),
                    "challenger_minus_baseline_v2_score": float(
                        right_selected_v2.get("score", 0.0)
                    )
                    - float(left_selected_v2.get("score", 0.0)),
                    "baseline_selected_maprank_score": float(
                        left_selected_maprank.get("score", 0.0)
                    ),
                    "challenger_selected_maprank_score": float(
                        right_selected_maprank.get("score", 0.0)
                    ),
                    "challenger_minus_baseline_maprank_score": float(
                        right_selected_maprank.get("score", 0.0)
                    )
                    - float(left_selected_maprank.get("score", 0.0)),
                },
                "baseline_repair": left_repair,
                "challenger_repair": right_repair,
                "immediate_effect": {
                    "conflict_delta_difference": right_repair["conflict_delta"]
                    - left_repair["conflict_delta"],
                    "low_level_generated_difference": right_repair[
                        "low_level_generated"
                    ]
                    - left_repair["low_level_generated"],
                    "low_level_expanded_difference": right_repair[
                        "low_level_expanded"
                    ]
                    - left_repair["low_level_expanded"],
                    "pp_replan_seconds_difference": right_repair[
                        "pp_replan_seconds"
                    ]
                    - left_repair["pp_replan_seconds"],
                },
            },
            errors,
        )
    if len(baseline) != len(challenger):
        errors.append("identical action prefix ended with different trajectory lengths")
    return common, None, errors


def _direction_counts(values: list[float], *, lower_is_better: bool) -> dict[str, int]:
    if lower_is_better:
        better = sum(value < 0 for value in values)
        worse = sum(value > 0 for value in values)
    else:
        better = sum(value > 0 for value in values)
        worse = sum(value < 0 for value in values)
    return {"better": better, "equal": len(values) - better - worse, "worse": worse}


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    overrides = [row for row in rows if row["first_override"] is not None]
    conflict_effects = [
        float(row["first_override"]["immediate_effect"]["conflict_delta_difference"])
        for row in overrides
    ]
    low_level_effects = [
        float(
            row["first_override"]["immediate_effect"][
                "low_level_generated_difference"
            ]
        )
        for row in overrides
    ]
    round_deltas = [float(row["repair_iterations_delta"]) for row in rows]
    ttf_deltas = [float(row["pooled_raw_ttf_delta_seconds"]) for row in rows]
    transitions = Counter(
        "|".join(row["first_override"]["baseline_candidate"]["selection_families"])
        + " -> "
        + "|".join(
            row["first_override"]["challenger_candidate"]["selection_families"]
        )
        for row in overrides
    )
    return {
        "episode_count": len(rows),
        "first_override_episode_count": len(overrides),
        "identical_action_episode_count": len(rows) - len(overrides),
        "mean_common_action_prefix": _mean(
            [float(row["common_action_prefix_count"]) for row in rows]
        ),
        "first_override_immediate_conflict_effect": _direction_counts(
            conflict_effects, lower_is_better=False
        ),
        "first_override_low_level_generated_effect": _direction_counts(
            low_level_effects, lower_is_better=True
        ),
        "eventual_repair_iteration_effect": _direction_counts(
            round_deltas, lower_is_better=True
        ),
        "pooled_raw_ttf_effect": _direction_counts(
            ttf_deltas, lower_is_better=True
        ),
        "mean_repair_iterations_delta": _mean(round_deltas),
        "mean_pooled_raw_ttf_delta_seconds": _mean(ttf_deltas),
        "mean_first_override_conflict_delta_difference": _mean(conflict_effects),
        "mean_first_override_low_level_generated_difference": _mean(
            low_level_effects
        ),
        "first_override_family_transitions": dict(sorted(transitions.items())),
    }


def analyze_maprank_high_load_failure(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    path, root, config, confirmation_path, confirmation = _load_config(config_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohorts = list(map(str, config["registered_cohorts"]))
    baseline_controller = str(config["baseline_controller"])
    challenger_controller = str(config["challenger_controller"])
    trace_root = _project_path(root, str(config["trace_replicate"]))
    trace_rows: dict[str, dict[tuple[str, str, int], dict[str, Any]]] = {}
    trace_hashes = {}
    for controller in (baseline_controller, challenger_controller):
        trace_rows[controller], trace_hashes[controller] = _manifest_index(
            trace_root, cohorts, controller
        )
    expected = set(trace_rows[baseline_controller])
    errors = []
    if expected != set(trace_rows[challenger_controller]):
        errors.append("trace controller episode coverage differs")
    if len(expected) != int(config["registered_episode_count"]):
        errors.append("registered trace episode count differs")
    timing_rows: dict[
        str, dict[str, dict[tuple[str, str, int], dict[str, Any]]]
    ] = {}
    timing_hashes = {}
    timing_report_hashes = {}
    for replicate_id, relative in enumerate(config["timing_replicates"], start=1):
        replicate = f"r{replicate_id}"
        replicate_root = _project_path(root, str(relative))
        report_path = replicate_root / "maprank_raw_ttf_report.json"
        report = _read_json(report_path)
        timing_report_hashes[replicate] = sha256_file(report_path)
        if report.get("integrity_passed") is not True:
            errors.append(f"{replicate}: source integrity failed")
        timing_rows[replicate] = {}
        timing_hashes[replicate] = {}
        for controller in (baseline_controller, challenger_controller):
            rows, hashes = _manifest_index(replicate_root, cohorts, controller)
            timing_rows[replicate][controller] = rows
            timing_hashes[replicate][controller] = hashes
            if set(rows) != expected:
                errors.append(f"{replicate}/{controller}: timing coverage differs")
    episodes = []
    for key in sorted(expected):
        cohort, task_id, solver_seed = key
        baseline_row = trace_rows[baseline_controller][key]
        challenger_row = trace_rows[challenger_controller][key]
        baseline_trace_root = (
            trace_root / "cohorts" / cohort / "controllers" / baseline_controller
        )
        challenger_trace_root = (
            trace_root / "cohorts" / cohort / "controllers" / challenger_controller
        )
        baseline_transitions = _transitions(baseline_trace_root, baseline_row)
        challenger_transitions = _transitions(challenger_trace_root, challenger_row)
        common, first_override, episode_errors = _first_override(
            baseline_transitions, challenger_transitions
        )
        errors.extend(f"{key}: {message}" for message in episode_errors)
        baseline_summary = dict(baseline_row["summary"])
        challenger_summary = dict(challenger_row["summary"])
        baseline_ttf = []
        challenger_ttf = []
        for replicate in timing_rows.values():
            baseline_ttf.append(
                float(
                    replicate[baseline_controller][key]["summary"][
                        "wall_time_to_feasible"
                    ]
                )
            )
            challenger_ttf.append(
                float(
                    replicate[challenger_controller][key]["summary"][
                        "wall_time_to_feasible"
                    ]
                )
            )
        episodes.append(
            {
                "cohort_id": cohort,
                "task_id": task_id,
                "solver_seed": solver_seed,
                "common_action_prefix_count": common,
                "first_override": first_override,
                "baseline_repair_iterations": int(
                    baseline_summary["repair_iterations"]
                ),
                "challenger_repair_iterations": int(
                    challenger_summary["repair_iterations"]
                ),
                "repair_iterations_delta": int(
                    challenger_summary["repair_iterations"]
                )
                - int(baseline_summary["repair_iterations"]),
                "baseline_conflict_trajectory": list(
                    baseline_summary["conflict_trajectory"]
                ),
                "challenger_conflict_trajectory": list(
                    challenger_summary["conflict_trajectory"]
                ),
                "baseline_mean_raw_ttf": _mean(baseline_ttf),
                "challenger_mean_raw_ttf": _mean(challenger_ttf),
                "pooled_raw_ttf_delta_seconds": _mean(challenger_ttf)
                - _mean(baseline_ttf),
            }
        )
    per_cohort = {
        cohort: _group_summary(
            [row for row in episodes if row["cohort_id"] == cohort]
        )
        for cohort in cohorts
    }
    integrity = {
        "confirmation_integrity_passed": confirmation.get("integrity_passed")
        is True,
        "confirmation_performance_failed": confirmation.get("performance_passed")
        is False,
        "fresh_maps_remained_locked": confirmation.get("fresh_map_unlocked")
        is False,
        "registered_episode_coverage": len(episodes)
        == int(config["registered_episode_count"]),
        "three_timing_replicates": len(timing_rows) == 3,
        "trace_controller_coverage_matches": expected
        == set(trace_rows[challenger_controller]),
        "shared_candidate_pools_at_first_override": not any(
            "candidate pool" in error for error in errors
        ),
        "common_prefix_semantics_match": not any(
            "semantics" in error or "state diverged" in error for error in errors
        ),
        "zero_source_errors": not any("source integrity" in error for error in errors),
    }
    passed = not errors and all(integrity.values())
    report = {
        "schema": REPORT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "scientific_status": "post_hoc_failure_decomposition_only",
        "post_hoc_diagnostic_only": True,
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "fresh_map_data_read": False,
        "wall_ttf_used_as_training_label": False,
        "future_trajectory_used_as_training_label": False,
        "baseline_controller": baseline_controller,
        "challenger_controller": challenger_controller,
        "episode_count": len(episodes),
        "summary": _group_summary(episodes),
        "per_cohort": per_cohort,
        "episodes": episodes,
        "integrity_gates": integrity,
        "passed": passed,
        "errors": errors,
        "inputs": {
            "config_sha256": sha256_file(path),
            "confirmation_report_sha256": sha256_file(confirmation_path),
            "trace_manifest_sha256": trace_hashes,
            "timing_report_sha256": timing_report_hashes,
            "timing_manifest_sha256": timing_hashes,
        },
    }
    _write_json(output / "maprank_high_load_failure_analysis.json", report)
    return report


__all__ = ["analyze_maprank_high_load_failure"]
