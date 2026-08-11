from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Any

from experiments._common import registered_input, sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_marginalpool_action_replay import stable_dominates


CONFIG_SCHEMA = "lns2.stride.historyrank_feature_registration.v1"
REPORT_SCHEMA = "lns2.stride.historyrank_feature_report.v1"
ROW_SCHEMA = "lns2.stride.historyrank_feature_row.v1"
EXPERIMENT_ID = "stride-historyrank-feature-v1"
SCIENTIFIC_STATUS = "preregistered_outcome_blind_history_feature_audit"


def load_registration(path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    config_path = Path(path).resolve()
    root = config_path.parents[1]
    config = _read_json(config_path)
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status") != SCIENTIFIC_STATUS
        or config.get("experiment_id") != EXPERIMENT_ID
        or dict(config.get("cohort") or {})
        != {
            "checkpoint_kind": "first_repeat_stall",
            "required_checkpoint_count": 45,
            "retain_all_registered_checkpoints": True,
            "no_result_based_exclusion": True,
        }
        or dict(config.get("candidate_scopes") or {})
        != {
            "primary": (
                "all runtime-retained alternatives except the exact repeated candidate"
            ),
            "diagnostic": (
                "all generated alternatives except the exact repeated candidate"
            ),
            "primary_scope_controls_readiness": True,
        }
    ):
        raise ValueError("HistoryRank feature registration changed")
    inputs = {
        name: registered_input(root, dict(spec), label=name)
        for name, spec in dict(config["inputs"]).items()
    }
    return config_path, config, inputs


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return float(len(left & right)) / float(len(union)) if union else 1.0


def _feature_distance(
    candidate: dict[str, Any],
    repeated: dict[str, Any],
    ranges: dict[str, float],
) -> tuple[float, float]:
    candidate_features = dict(candidate["features"])
    repeated_features = dict(repeated["features"])
    if (
        int(candidate["feature_count"]) != 124
        or int(repeated["feature_count"]) != 124
        or set(candidate_features) != set(repeated_features)
        or len(candidate_features) != 124
    ):
        raise ValueError("HistoryRank feature schema changed")
    changed = 0
    normalized = []
    for name in sorted(candidate_features):
        delta = abs(float(candidate_features[name]) - float(repeated_features[name]))
        if delta > 1.0e-12:
            changed += 1
        scale = float(ranges[name])
        if scale > 0.0:
            normalized.append(delta / scale)
    return (
        float(changed) / 124.0,
        statistics.fmean(normalized) if normalized else 0.0,
    )


def directed_feature_values(
    *,
    candidate_meta: dict[str, Any],
    repeated_meta: dict[str, Any],
    candidate_quality: dict[str, Any],
    repeated_quality: dict[str, Any],
    ranges: dict[str, float],
) -> dict[str, float]:
    candidate_agents = {int(value) for value in candidate_meta["agents"]}
    repeated_agents = {int(value) for value in repeated_meta["agents"]}
    candidate_families = {str(value) for value in candidate_quality["selection_families"]}
    repeated_families = {str(value) for value in repeated_quality["selection_families"]}
    changed_fraction, standardized_l1 = _feature_distance(
        candidate_quality, repeated_quality, ranges
    )
    candidate_features = dict(candidate_quality["features"])
    repeated_features = dict(repeated_quality["features"])
    candidate_size = max(1, len(candidate_agents))
    repeated_size = max(1, len(repeated_agents))
    candidate_score = candidate_meta.get("score")
    repeated_score = repeated_meta.get("score")
    return {
        "agent_novelty": 1.0 - _jaccard(candidate_agents, repeated_agents),
        "family_novelty": 1.0 - _jaccard(candidate_families, repeated_families),
        "feature_standardized_l1": standardized_l1,
        "feature_changed_fraction": changed_fraction,
        "removed_agent_ratio": float(len(repeated_agents - candidate_agents))
        / float(repeated_size),
        "added_agent_ratio": float(len(candidate_agents - repeated_agents))
        / float(candidate_size),
        "internal_conflict_coverage_delta": float(
            candidate_features["realized.internal_conflict_coverage"]
        )
        - float(repeated_features["realized.internal_conflict_coverage"]),
        "incident_conflict_coverage_delta": float(
            candidate_features["realized.incident_conflict_coverage"]
        )
        - float(repeated_features["realized.incident_conflict_coverage"]),
        "component_coverage_mean_delta": float(
            candidate_features["realized.component_coverage_mean"]
        )
        - float(repeated_features["realized.component_coverage_mean"]),
        "lower_boundary_conflict_edges": float(
            repeated_features["realized.boundary_conflict_edges"]
        )
        - float(candidate_features["realized.boundary_conflict_edges"]),
        "lower_path_overlap_mean": float(
            repeated_features["realized.path_overlap_mean"]
        )
        - float(candidate_features["realized.path_overlap_mean"]),
        "absolute_log_size_ratio": abs(
            math.log(float(candidate_size) / float(repeated_size))
        ),
        "v2_score_margin": (
            float(candidate_score) - float(repeated_score)
            if candidate_score is not None and repeated_score is not None
            else float("nan")
        ),
    }


def choose_by_directed_feature(
    candidates: list[dict[str, Any]], *, feature_name: str
) -> dict[str, Any]:
    eligible = [
        row
        for row in candidates
        if feature_name in dict(row["directed_features"])
        and math.isfinite(float(dict(row["directed_features"])[feature_name]))
    ]
    if not eligible:
        raise ValueError(f"HistoryRank feature {feature_name} has no eligible candidate")

    def key(row: dict[str, Any]) -> tuple[float, float, str]:
        score = row.get("score")
        numeric_score = float(score) if score is not None else float("-inf")
        return (
            -float(dict(row["directed_features"])[feature_name]),
            -numeric_score,
            str(row["candidate_id"]),
        )

    return min(eligible, key=key)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "checkpoint_count": len(rows),
        "stable_improvement_count": sum(bool(row["stable_improvement"]) for row in rows),
        "stable_improvement_fraction": statistics.fmean(
            float(row["stable_improvement"]) for row in rows
        ),
        "mean_seed_improvement": statistics.fmean(
            float(row["seed_mean_delta"]) for row in rows
        ),
        "mean_no_progress_delta": statistics.fmean(
            float(row["no_progress_rate_delta"]) for row in rows
        ),
        "both_halves_positive_fraction": statistics.fmean(
            float(row["both_halves_positive"]) for row in rows
        ),
        "mean_feature_value": statistics.fmean(
            float(row["directed_feature_value"]) for row in rows
        ),
    }


def _gate_summary(
    summary: dict[str, Any],
    by_map: dict[str, dict[str, Any]],
    gates: dict[str, Any],
) -> dict[str, bool]:
    return {
        "stable_improvement_fraction": float(summary["stable_improvement_fraction"])
        >= float(gates["minimum_stable_improvement_fraction"]),
        "mean_seed_improvement": float(summary["mean_seed_improvement"])
        >= float(gates["minimum_mean_seed_improvement"]),
        "mean_no_progress_delta": float(summary["mean_no_progress_delta"])
        <= float(gates["maximum_mean_no_progress_delta"]),
        "both_halves_positive_fraction": float(
            summary["both_halves_positive_fraction"]
        )
        >= float(gates["minimum_both_half_positive_fraction"]),
        "every_map_mean_seed_improvement": all(
            float(group["mean_seed_improvement"])
            > float(gates["minimum_per_map_mean_seed_improvement"])
            for group in by_map.values()
        ),
        "zero_errors": int(gates["required_error_count"]) == 0,
    }


def run_historyrank_feature_audit(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path, config, inputs = load_registration(config_path)
    collection = _read_json(inputs["collection_report"])
    if (
        collection.get("complete") is not True
        or collection.get("integrity_passed") is not True
        or int(collection.get("completed_state_count", -1)) != 78
        or int(collection.get("candidate_count", -1)) != 2502
        or int(collection.get("trial_count", -1)) != 40032
    ):
        raise ValueError("HistoryRank feature input collection is not complete")

    cases = {str(row["case_id"]): row for row in _read_jsonl(inputs["root_cases"])}
    checkpoints = [
        row
        for row in _read_jsonl(inputs["root_checkpoints"])
        if row.get("checkpoint_kind") == "first_repeat_stall"
    ]
    logical = {
        str(row["case_id"]): row
        for row in _read_jsonl(inputs["logical_results"])
        if row.get("checkpoint_kind") == "first_repeat_stall"
    }
    aggregates = {
        (str(row["state_fingerprint"]), str(row["candidate_id"])): row
        for row in _read_jsonl(inputs["candidate_aggregates"])
    }
    required = int(config["cohort"]["required_checkpoint_count"])
    if len(checkpoints) != required or len(logical) != required or len(cases) != required:
        raise ValueError("HistoryRank feature cohort count changed")

    feature_priority = [str(value) for value in config["feature_priority"]]
    row_sets: dict[str, list[dict[str, Any]]] = {
        f"{scope}:{feature}": []
        for scope in ("primary", "diagnostic")
        for feature in feature_priority
    }
    for checkpoint in checkpoints:
        case_id = str(checkpoint["case_id"])
        case = cases[case_id]
        result = logical[case_id]
        repeated_id = str(case["repeated_candidate_id"])
        prior = list(dict(case["pp_response"])["prior_repairs"])
        if (
            dict(case["pp_response"]).get("true_pp_noop") is not True
            or len(prior) != 2
            or any(str(row["candidate_id"]) != repeated_id for row in prior)
            or any(int(row["changed_agent_count"]) != 0 for row in prior)
            or any(dict(row["repair_metadata"])["replan_success"] is not False for row in prior)
            or str(checkpoint["selected_candidate_id"]) != repeated_id
            or str(result["selected_candidate_id"]) != repeated_id
        ):
            raise ValueError("HistoryRank feature activation history changed")

        state_key = str(checkpoint["state_fingerprint"])
        metadata = {
            str(row["candidate_id"]): dict(row) for row in checkpoint["candidate_pool"]
        }
        if set(result["candidate_ids"]) != set(metadata):
            raise ValueError("HistoryRank feature candidate pool changed")
        quality = {
            candidate_id: aggregates[(state_key, candidate_id)]
            for candidate_id in metadata
        }
        repeated_meta = metadata[repeated_id]
        repeated_quality = quality[repeated_id]
        feature_names = sorted(dict(repeated_quality["features"]))
        if len(feature_names) != 124:
            raise ValueError("HistoryRank feature count changed")
        ranges = {
            name: max(float(row["features"][name]) for row in quality.values())
            - min(float(row["features"][name]) for row in quality.values())
            for name in feature_names
        }
        candidates = []
        for candidate_id in sorted(metadata):
            if candidate_id == repeated_id:
                continue
            candidate_meta = metadata[candidate_id]
            candidate_quality = quality[candidate_id]
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "retained": candidate_meta.get("retained") is True,
                    "score": candidate_meta.get("score"),
                    "quality": candidate_quality,
                    "directed_features": directed_feature_values(
                        candidate_meta=candidate_meta,
                        repeated_meta=repeated_meta,
                        candidate_quality=candidate_quality,
                        repeated_quality=repeated_quality,
                        ranges=ranges,
                    ),
                }
            )
        scopes = {
            "primary": [row for row in candidates if row["retained"]],
            "diagnostic": candidates,
        }
        for scope, scoped_candidates in scopes.items():
            if not scoped_candidates:
                raise ValueError(f"HistoryRank feature {scope} pool is empty")
            for feature_name in feature_priority:
                chosen = choose_by_directed_feature(
                    scoped_candidates, feature_name=feature_name
                )
                chosen_quality = dict(chosen["quality"])
                seed_delta = float(chosen_quality["seed_mean"]) - float(
                    repeated_quality["seed_mean"]
                )
                no_progress_delta = float(chosen_quality["no_progress_rate"]) - float(
                    repeated_quality["no_progress_rate"]
                )
                first_delta = float(chosen_quality["first_fixed_half_mean"]) - float(
                    repeated_quality["first_fixed_half_mean"]
                )
                second_delta = float(chosen_quality["second_fixed_half_mean"]) - float(
                    repeated_quality["second_fixed_half_mean"]
                )
                row_sets[f"{scope}:{feature_name}"].append(
                    {
                        "schema": ROW_SCHEMA,
                        "scope": scope,
                        "feature_name": feature_name,
                        "logical_checkpoint_id": str(result["logical_checkpoint_id"]),
                        "case_id": case_id,
                        "state_fingerprint": state_key,
                        "map_id": str(checkpoint["map_id"]),
                        "task_id": str(checkpoint["task_id"]),
                        "solver_seed": int(checkpoint["solver_seed"]),
                        "repeated_candidate_id": repeated_id,
                        "chosen_candidate_id": str(chosen["candidate_id"]),
                        "directed_feature_value": float(
                            dict(chosen["directed_features"])[feature_name]
                        ),
                        "seed_mean_delta": seed_delta,
                        "no_progress_rate_delta": no_progress_delta,
                        "first_half_delta": first_delta,
                        "second_half_delta": second_delta,
                        "both_halves_positive": first_delta > 0.0 and second_delta > 0.0,
                        "stable_improvement": stable_dominates(
                            chosen_quality,
                            repeated_quality,
                            minimum=float(
                                config["quality_rule"][
                                    "minimum_seed_mean_advantage"
                                ]
                            ),
                        ),
                        "runtime_or_ttf_read": False,
                        "future_trajectory_read": False,
                    }
                )

    all_rows = [row for rows in row_sets.values() for row in rows]
    all_rows.sort(
        key=lambda row: (
            str(row["scope"]),
            str(row["feature_name"]),
            str(row["logical_checkpoint_id"]),
        )
    )
    expected_rows = 2 * len(feature_priority) * required
    if len(all_rows) != expected_rows:
        raise ValueError("HistoryRank feature row product changed")

    gates = dict(config["readiness_gates"])
    feature_reports: dict[str, dict[str, Any]] = {}
    passing: list[str] = []
    for scope in ("primary", "diagnostic"):
        for feature_name in feature_priority:
            rows = row_sets[f"{scope}:{feature_name}"]
            summary = _summary(rows)
            by_map = {
                map_id: _summary([row for row in rows if row["map_id"] == map_id])
                for map_id in sorted({str(row["map_id"]) for row in rows})
            }
            gate_results = _gate_summary(summary, by_map, gates)
            passed = all(gate_results.values())
            feature_reports[f"{scope}:{feature_name}"] = {
                "summary": summary,
                "by_map": by_map,
                "gates": gate_results,
                "passed": passed,
            }
            if scope == "primary" and passed:
                passing.append(feature_name)

    selected_target = next(
        (feature for feature in feature_priority if feature in passing), None
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "historyrank_feature_rows.jsonl"
    _write_jsonl(rows_path, all_rows)
    report = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "scientific_status": "offline_checkpoint_local_history_feature_audit",
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
        "checkpoint_count": required,
        "feature_count": len(feature_priority),
        "row_count": len(all_rows),
        "feature_priority": feature_priority,
        "feature_reports": feature_reports,
        "passing_primary_features": passing,
        "selected_target": selected_target,
        "feature_readiness_passed": selected_target is not None,
        "rows_sha256": sha256_file(rows_path),
        "runtime_or_ttf_read": False,
        "future_trajectory_read": False,
        "claim_boundary": dict(config["claim_boundary"]),
        "next_step": (
            "preregister a fresh-map confirmation for the selected target"
            if selected_target is not None
            else "preregister a fresh-map grouped HistoryRank training cohort"
        ),
    }
    report_path = output / "historyrank_feature_report.json"
    _write_json(report_path, report)
    _write_json(
        output / "historyrank_feature_status.json",
        {
            "schema": "lns2.stride.historyrank_feature_status.v1",
            "complete": True,
            "integrity_passed": True,
            "feature_readiness_passed": selected_target is not None,
            "selected_target": selected_target,
            "report_sha256": sha256_file(report_path),
        },
    )
    return report
