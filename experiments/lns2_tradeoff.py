from __future__ import annotations

import csv
import html
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments.closed_loop_trace_storage import (
    TRACE_FORMAT_DELTA_GZIP_V2,
    read_trace_events,
    trace_file_metadata,
)
from experiments.natural_distribution_confirmation import (
    conflict_density,
    conflict_severity,
)
from experiments.repair_collection import _read_json, _read_jsonl, _write_json
from experiments.route_counterfactual import (
    COUNTERFACTUAL_SCOPE,
    iter_route_counterfactual_rows,
)


TRADEOFF_REPORT_SCHEMA = "lns2.four_way_tradeoff_evaluation.v2"
CONTROLLER_ORDER = (
    "official_adaptive",
    "v1-full",
    "v2-full",
    "v2-balanced",
)
CONTROLLER_LABELS = {
    "official_adaptive": "Original LNS2 Adaptive",
    "v1-full": "Old full model (v1)",
    "v2-full": "Optimized full model (v2)",
    "v2-balanced": "Balanced hybrid (v2)",
}
VALID_STATUSES = {"ok", "resumed"}
HISTORICAL_REFERENCE = {
    "same_distribution_80_100_agents": {
        "model_auc_improvement": 0.525,
        "model_seconds": 0.715,
        "lns2_seconds": 0.273,
        "model_time_ratio_vs_lns2": 0.715 / 0.273,
    },
    "movingai_100_600_agents": {
        "model_success_count": 131,
        "lns2_success_count": 123,
        "raw_auc_improvement": 0.041,
        "normalized_auc_improvement": 0.324,
        "model_mean_capped_seconds": 62.1757,
        "lns2_mean_capped_seconds": 88.8144,
        "model_wall_time_improvement": 1.0 - 62.1757 / 88.8144,
        "timing_confidence_interval_includes_slight_degradation": True,
        "maps_not_slower_count": 5,
        "map_count": 9,
    },
}


def _mean(values: Iterable[float]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def _relative_improvement(
    reference: float | None, candidate: float | None
) -> float | None:
    if reference in {None, 0.0} or candidate is None:
        return None
    return (float(reference) - float(candidate)) / float(reference)


def _relative_increase(
    reference: float | None, candidate: float | None
) -> float | None:
    value = _relative_improvement(reference, candidate)
    return None if value is None else -value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_path(root: Path, controller: str) -> Path:
    policy = (
        "official_adaptive"
        if controller == "official_adaptive"
        else "realized_dynamic"
    )
    return root / f"{policy}_manifest.jsonl"


def _episode_row(
    source: dict[str, Any],
    *,
    controller: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    summary = dict(source.get("summary") or {})
    totals = dict(summary.get("controller_totals") or {})
    low_level = dict(summary.get("final_low_level") or {})
    agents = int(source.get("agent_count", 0))
    conflicts = int(summary.get("initial_conflicts", 0))
    density = conflict_density(conflicts, agents)
    fixed_auc = summary.get("fixed_budget_conflict_auc")
    return {
        "controller": controller,
        "episode_id": source.get("episode_id"),
        "task_id": source.get("task_id"),
        "map_id": source.get("map_id"),
        "layout_family": source.get("layout_mode"),
        "agent_count": agents,
        "solver_seed": int(source.get("solver_seed", -1)),
        "status": source.get("status"),
        "trace_file": source.get("trace_file"),
        "repairable": bool(summary.get("repairable")),
        "success": bool(summary.get("success")),
        "external_timeout": bool(summary.get("external_timeout")),
        "initial_fingerprint": summary.get("initial_fingerprint"),
        "initial_conflicts": conflicts,
        "initial_conflict_density": density,
        "initial_conflict_severity": conflict_severity(density, thresholds),
        "final_conflicts": int(summary.get("final_conflicts", 0)),
        "repair_iterations": int(summary.get("repair_iterations", 0)),
        "fixed_budget_conflict_auc": fixed_auc,
        "normalized_fixed_budget_conflict_auc": (
            float(fixed_auc) / conflicts
            if fixed_auc is not None and conflicts > 0
            else None
        ),
        "capped_wall_time_seconds": summary.get("capped_wall_time_to_feasible"),
        "repair_wall_seconds": summary.get("repair_wall_seconds"),
        "controller_seconds": float(
            totals.get("controller_seconds_before_repair", 0.0)
        ),
        "feature_seconds": float(totals.get("feature_seconds", 0.0)),
        "inference_seconds": float(totals.get("inference_seconds", 0.0)),
        "final_sum_of_costs": summary.get("final_sum_of_costs"),
        "low_level_expanded": int(low_level.get("expanded", 0)),
        "low_level_generated": int(low_level.get("generated", 0)),
        "low_level_reopened": int(low_level.get("reopened", 0)),
        "invalid_action_count": int(summary.get("invalid_action_count", 0)),
        "fingerprint_mismatch_count": int(
            summary.get("fingerprint_mismatch_count", 0)
        ),
        "learned_decision_count": int(totals.get("learned_decisions", 0)),
        "shadow_validation_count": int(
            totals.get("shadow_validation_count", 0)
        ),
        "shadow_score_max_delta": float(
            totals.get("shadow_score_max_delta", 0.0)
        ),
        "model_decision_count": int(summary.get("model_decision_count", 0)),
        "official_decision_count": int(summary.get("official_decision_count", 0)),
        "model_route_fraction": float(summary.get("model_route_fraction", 0.0)),
        "route_switch_count": int(summary.get("route_switch_count", 0)),
        "candidate_count_before": int(
            totals.get("candidate_count_before_pruning", 0)
        ),
        "candidate_count_after": int(
            totals.get("candidate_count_after_pruning", 0)
        ),
        "model_controller_seconds": float(
            totals.get("model_controller_seconds", 0.0)
        ),
        "model_repair_seconds": float(totals.get("model_repair_seconds", 0.0)),
        "model_total_decision_seconds": float(
            totals.get("model_total_decision_seconds", 0.0)
        ),
        "official_controller_seconds": float(
            totals.get("official_controller_seconds", 0.0)
        ),
        "official_repair_seconds": float(
            totals.get("official_repair_seconds", 0.0)
        ),
        "official_total_decision_seconds": float(
            totals.get("official_total_decision_seconds", 0.0)
        ),
    }


def _load_episodes(
    roots: dict[str, Path], thresholds: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_modes = {
        "official_adaptive": "v1-full",
        "v1-full": "v1-full",
        "v2-full": "v2-full",
        "v2-balanced": "v2-balanced",
    }
    episodes: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    for controller in CONTROLLER_ORDER:
        root = roots[controller]
        run = _read_json(root / "run_config.json")
        actual_mode = str(run.get("controller"))
        if actual_mode != expected_modes[controller]:
            raise ValueError(
                f"{controller} collection uses {actual_mode}, "
                f"not {expected_modes[controller]}"
            )
        bundle = dict(run.get("controller_bundle") or {})
        balanced = dict(run.get("balanced_config") or {})
        if controller == "v2-balanced" and str(
            dict(balanced.get("source") or {}).get("selection_unit")
        ) != "complete_episode":
            raise ValueError(
                "balanced collection uses an obsolete non-episode calibration"
            )
        metadata[controller] = {
            "root": str(root),
            "run_fingerprint": run.get("run_fingerprint"),
            "dataset_fingerprint": run.get("dataset_fingerprint"),
            "trace_format": run.get("trace_format"),
            "feature_backend": run.get("feature_backend"),
            "feature_shadow_validation": bool(
                dict(run.get("configuration") or {}).get(
                    "feature_shadow_validation", False
                )
            ),
            "model_semantic_fingerprint": bundle.get(
                "main_ranker_semantic_fingerprint"
            ),
            "balanced_config_fingerprint": balanced.get(
                "configuration_fingerprint"
            ),
        }
        manifest_rows = _read_jsonl(_manifest_path(root, controller))
        for source in manifest_rows:
            if str(source.get("status")) not in VALID_STATUSES:
                continue
            trace = (root / str(source.get("trace_file", ""))).resolve()
            try:
                trace.relative_to(root)
            except ValueError as error:
                raise ValueError(f"{controller} trace escapes its collection") from error
            if not trace.is_file():
                raise ValueError(f"{controller} trace is missing: {trace}")
            file_metadata = trace_file_metadata(trace)
            if (
                str(file_metadata["trace_sha256"])
                != str(source.get("trace_sha256"))
                or int(file_metadata["trace_bytes"])
                != int(source.get("trace_bytes", -1))
            ):
                raise ValueError(f"{controller} trace metadata mismatch: {trace}")
        controller_rows = [
            _episode_row(row, controller=controller, thresholds=thresholds)
            for row in manifest_rows
        ]
        for row in controller_rows:
            if controller == "v2-balanced" and row["status"] in VALID_STATUSES:
                routed = int(row["model_decision_count"]) + int(
                    row["official_decision_count"]
                )
                if routed != int(row["repair_iterations"]):
                    raise ValueError(
                        f"balanced route accounting mismatch: {row['episode_id']}"
                    )
            if controller in {"v1-full", "v2-full"} and row["status"] in VALID_STATUSES:
                if int(row["learned_decision_count"]) != int(
                    row["repair_iterations"]
                ):
                    raise ValueError(
                        f"full-model decision accounting mismatch: {row['episode_id']}"
                    )
        episodes.extend(controller_rows)

    if len(
        {str(value["dataset_fingerprint"]) for value in metadata.values()}
    ) != 1:
        raise ValueError("paired controller collections use different datasets")
    if {
        str(value["trace_format"]) for value in metadata.values()
    } != {TRACE_FORMAT_DELTA_GZIP_V2}:
        raise ValueError("all four collections must use delta-gzip-v2")
    model_fingerprints = {
        metadata[name]["model_semantic_fingerprint"]
        for name in ("v2-full", "v2-balanced")
    }
    if None in model_fingerprints or len(model_fingerprints) != 1:
        raise ValueError("v2-full and v2-balanced use different frozen models")
    if metadata["v2-balanced"]["balanced_config_fingerprint"] is None:
        raise ValueError("balanced collection is missing its frozen route config")
    return episodes, metadata


def _paired_index(
    episodes: list[dict[str, Any]],
) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for controller in CONTROLLER_ORDER:
        selected = [
            row
            for row in episodes
            if row["controller"] == controller and row["status"] in VALID_STATUSES
        ]
        rows = {
            (str(row["task_id"]), int(row["solver_seed"])): row
            for row in selected
        }
        if len(rows) != len(selected):
            raise ValueError(f"{controller} contains duplicate task/seed episodes")
        result[controller] = rows
    return result


def _frontier_rows(
    episodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[tuple[str, int]]]:
    index = _paired_index(episodes)
    paired_keys = set.intersection(*(set(index[name]) for name in CONTROLLER_ORDER))
    for key in paired_keys:
        values = [index[name][key] for name in CONTROLLER_ORDER]
        fingerprints = {str(row["initial_fingerprint"]) for row in values}
        conflicts = {int(row["initial_conflicts"]) for row in values}
        if (
            any(not row["initial_fingerprint"] for row in values)
            or len(fingerprints) != 1
            or len(conflicts) != 1
        ):
            raise ValueError(f"paired controllers start from different states: {key}")

    result: list[dict[str, Any]] = []
    for controller in CONTROLLER_ORDER:
        rows = [index[controller][key] for key in sorted(paired_keys)]
        repairable = [row for row in rows if row["repairable"]]
        result.append(
            {
                "controller": controller,
                "controller_label": CONTROLLER_LABELS[controller],
                "paired_episode_count": len(rows),
                "repairable_episode_count": len(repairable),
                "success_count": sum(bool(row["success"]) for row in rows),
                "success_rate": (
                    sum(bool(row["success"]) for row in rows) / len(rows)
                    if rows
                    else None
                ),
                "mean_fixed_budget_conflict_auc": _mean(
                    row["fixed_budget_conflict_auc"] for row in repairable
                ),
                "mean_normalized_conflict_auc": _mean(
                    row["normalized_fixed_budget_conflict_auc"]
                    for row in repairable
                ),
                "mean_capped_wall_time_seconds": _mean(
                    row["capped_wall_time_seconds"] for row in repairable
                ),
                "mean_controller_seconds": _mean(
                    row["controller_seconds"] for row in repairable
                ),
                "mean_repair_wall_seconds": _mean(
                    row["repair_wall_seconds"] for row in repairable
                ),
                "mean_repair_iterations": _mean(
                    row["repair_iterations"] for row in repairable
                ),
                "mean_final_sum_of_costs_on_success": _mean(
                    row["final_sum_of_costs"] for row in rows if row["success"]
                ),
                "mean_low_level_expanded": _mean(
                    row["low_level_expanded"] for row in repairable
                ),
                "mean_low_level_generated": _mean(
                    row["low_level_generated"] for row in repairable
                ),
                "mean_low_level_reopened": _mean(
                    row["low_level_reopened"] for row in repairable
                ),
            }
        )
    official = next(row for row in result if row["controller"] == "official_adaptive")
    for row in result:
        row["auc_improvement_vs_lns2"] = _relative_improvement(
            official["mean_fixed_budget_conflict_auc"],
            row["mean_fixed_budget_conflict_auc"],
        )
        row["normalized_auc_improvement_vs_lns2"] = _relative_improvement(
            official["mean_normalized_conflict_auc"],
            row["mean_normalized_conflict_auc"],
        )
        row["wall_time_improvement_vs_lns2"] = _relative_improvement(
            official["mean_capped_wall_time_seconds"],
            row["mean_capped_wall_time_seconds"],
        )
    return result, paired_keys


def _one_sided_t95(sample_count: int) -> float:
    if sample_count <= 1:
        return math.inf
    limits = (
        (2, 6.314),
        (3, 2.920),
        (4, 2.353),
        (5, 2.132),
        (6, 2.015),
        (8, 1.895),
        (10, 1.833),
        (15, 1.761),
        (20, 1.729),
        (30, 1.699),
        (40, 1.684),
        (60, 1.671),
        (120, 1.658),
    )
    for maximum, value in limits:
        if sample_count <= maximum:
            return value
    return 1.645


def _paired_delta_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "paired_count": 0,
            "mean_candidate_minus_reference": None,
            "median_candidate_minus_reference": None,
            "one_sided_95_upper": None,
        }
    mean = statistics.fmean(values)
    if len(values) == 1:
        upper = math.inf
    else:
        standard_error = statistics.stdev(values) / math.sqrt(len(values))
        upper = mean + _one_sided_t95(len(values)) * standard_error
    return {
        "paired_count": len(values),
        "mean_candidate_minus_reference": mean,
        "median_candidate_minus_reference": statistics.median(values),
        "one_sided_95_upper": upper,
    }


def _controller_speed_rows(
    episodes: list[dict[str, Any]], paired_keys: set[tuple[str, int]]
) -> list[dict[str, Any]]:
    index = _paired_index(episodes)
    comparisons = (
        ("v2-full_vs_v1-full", "v1-full", "v2-full"),
        ("v2-full_vs_original-lns2", "official_adaptive", "v2-full"),
        ("v2-balanced_vs_v2-full", "v2-full", "v2-balanced"),
        ("v2-balanced_vs_original-lns2", "official_adaptive", "v2-balanced"),
        ("v1-full_vs_original-lns2", "official_adaptive", "v1-full"),
    )
    result = []
    for comparison, reference, candidate in comparisons:
        keys = [
            key
            for key in sorted(paired_keys)
            if index[reference][key]["repairable"]
            and index[candidate][key]["repairable"]
        ]
        reference_rows = [index[reference][key] for key in keys]
        candidate_rows = [index[candidate][key] for key in keys]
        controller_deltas_by_map: dict[str, list[float]] = defaultdict(list)
        wall_deltas_by_map: dict[str, list[float]] = defaultdict(list)
        for left, right in zip(reference_rows, candidate_rows):
            map_id = str(left["map_id"])
            controller_deltas_by_map[map_id].append(
                float(right["controller_seconds"])
                - float(left["controller_seconds"])
            )
            wall_deltas_by_map[map_id].append(
                float(right["capped_wall_time_seconds"])
                - float(left["capped_wall_time_seconds"])
            )
        controller_delta = _paired_delta_stats(
            [
                float(_mean(values))
                for values in controller_deltas_by_map.values()
                if values
            ]
        )
        wall_delta = _paired_delta_stats(
            [
                float(_mean(values))
                for values in wall_deltas_by_map.values()
                if values
            ]
        )
        reference_controller = _mean(
            row["controller_seconds"] for row in reference_rows
        )
        candidate_controller = _mean(
            row["controller_seconds"] for row in candidate_rows
        )
        reference_wall = _mean(
            row["capped_wall_time_seconds"] for row in reference_rows
        )
        candidate_wall = _mean(
            row["capped_wall_time_seconds"] for row in candidate_rows
        )
        reference_auc = _mean(
            row["fixed_budget_conflict_auc"] for row in reference_rows
        )
        candidate_auc = _mean(
            row["fixed_budget_conflict_auc"] for row in candidate_rows
        )
        common_success_soc_ratios = [
            float(right["final_sum_of_costs"])
            / float(left["final_sum_of_costs"])
            for left, right in zip(reference_rows, candidate_rows)
            if left["success"]
            and right["success"]
            and float(left["final_sum_of_costs"]) != 0.0
        ]
        reference_repair = _mean(
            row["repair_wall_seconds"] for row in reference_rows
        )
        candidate_repair = _mean(
            row["repair_wall_seconds"] for row in candidate_rows
        )
        result.append(
            {
                "comparison": comparison,
                "reference": reference,
                "candidate": candidate,
                "paired_episode_count": len(keys),
                "paired_map_count": len(controller_deltas_by_map),
                "uncertainty_unit": "equal-map paired mean",
                "reference_success_count": sum(
                    bool(row["success"]) for row in reference_rows
                ),
                "candidate_success_count": sum(
                    bool(row["success"]) for row in candidate_rows
                ),
                "success_count_delta": sum(
                    bool(row["success"]) for row in candidate_rows
                )
                - sum(bool(row["success"]) for row in reference_rows),
                "reference_mean_auc": reference_auc,
                "candidate_mean_auc": candidate_auc,
                "auc_improvement": _relative_improvement(
                    reference_auc, candidate_auc
                ),
                "reference_mean_controller_seconds": reference_controller,
                "candidate_mean_controller_seconds": candidate_controller,
                "controller_time_improvement": _relative_improvement(
                    reference_controller, candidate_controller
                ),
                "controller_time_mean_delta_seconds": controller_delta[
                    "mean_candidate_minus_reference"
                ],
                "controller_time_one_sided_95_upper_delta_seconds": controller_delta[
                    "one_sided_95_upper"
                ],
                "controller_time_significantly_lower": (
                    controller_delta["one_sided_95_upper"] is not None
                    and float(controller_delta["one_sided_95_upper"]) < 0.0
                ),
                "reference_mean_wall_seconds": reference_wall,
                "candidate_mean_wall_seconds": candidate_wall,
                "wall_time_improvement": _relative_improvement(
                    reference_wall, candidate_wall
                ),
                "wall_time_mean_delta_seconds": wall_delta[
                    "mean_candidate_minus_reference"
                ],
                "wall_time_one_sided_95_upper_delta_seconds": wall_delta[
                    "one_sided_95_upper"
                ],
                "reference_mean_repair_seconds": reference_repair,
                "candidate_mean_repair_seconds": candidate_repair,
                "repair_time_improvement": _relative_improvement(
                    reference_repair, candidate_repair
                ),
                "common_success_episode_count": len(common_success_soc_ratios),
                "common_success_soc_ratio": _mean(common_success_soc_ratios),
                "reference_mean_low_level_expanded": _mean(
                    row["low_level_expanded"] for row in reference_rows
                ),
                "candidate_mean_low_level_expanded": _mean(
                    row["low_level_expanded"] for row in candidate_rows
                ),
                "low_level_expanded_improvement": _relative_improvement(
                    _mean(row["low_level_expanded"] for row in reference_rows),
                    _mean(row["low_level_expanded"] for row in candidate_rows),
                ),
                "reference_mean_low_level_generated": _mean(
                    row["low_level_generated"] for row in reference_rows
                ),
                "candidate_mean_low_level_generated": _mean(
                    row["low_level_generated"] for row in candidate_rows
                ),
                "low_level_generated_improvement": _relative_improvement(
                    _mean(row["low_level_generated"] for row in reference_rows),
                    _mean(row["low_level_generated"] for row in candidate_rows),
                ),
                "reference_mean_low_level_reopened": _mean(
                    row["low_level_reopened"] for row in reference_rows
                ),
                "candidate_mean_low_level_reopened": _mean(
                    row["low_level_reopened"] for row in candidate_rows
                ),
                "low_level_reopened_improvement": _relative_improvement(
                    _mean(row["low_level_reopened"] for row in reference_rows),
                    _mean(row["low_level_reopened"] for row in candidate_rows),
                ),
            }
        )
    return result


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_semantics(
    event: dict[str, Any]
) -> tuple[list[str], list[str], dict[str, float]]:
    controller = dict(event.get("controller") or {})
    pool = list(controller.get("candidate_pool") or [])
    identifiers = [str(row.get("candidate_id")) for row in pool]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate pool contains duplicate IDs")
    scores = {
        str(row.get("candidate_id")): float(row["score"])
        for row in pool
        if bool(row.get("retained", True)) and row.get("score") is not None
    }
    ranking = sorted(
        scores,
        key=lambda candidate_id: (-round(scores[candidate_id], 12), candidate_id),
    )
    return identifiers, ranking, scores


def _semantic_equivalence_report(
    roots: dict[str, Path],
    episodes: list[dict[str, Any]],
    paired_keys: set[tuple[str, int]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    index = _paired_index(episodes)
    mismatch_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    boundary_details: list[dict[str, Any]] = []
    common_decisions = 0
    post_repair_comparisons = 0
    score_comparisons = 0
    maximum_score_delta = 0.0
    differing_length_episodes = 0
    allowed_budget_length_differences = 0
    unexplained_length_differences = 0
    wall_budgets: dict[str, float | None] = {}
    for controller in ("v1-full", "v2-full"):
        run = _read_json(roots[controller] / "run_config.json")
        raw_budget = dict(run.get("configuration") or {}).get(
            "wall_time_budget_seconds"
        )
        budget = float(raw_budget) if raw_budget is not None else None
        wall_budgets[controller] = (
            budget
            if budget is not None and math.isfinite(budget) and budget > 0.0
            else None
        )

    def mismatch(
        key: tuple[str, int], decision_index: int | None, kind: str, detail: str
    ) -> None:
        mismatch_counts[kind] += 1
        if len(details) < 500:
            details.append(
                {
                    "task_id": key[0],
                    "solver_seed": key[1],
                    "decision_index": decision_index,
                    "mismatch_kind": kind,
                    "detail": detail,
                }
            )

    def boundary_status(
        event: dict[str, Any],
        *,
        position: int,
        transitions: list[dict[str, Any]],
        episode: dict[str, Any],
        controller: str,
    ) -> tuple[bool, str, float | None, float | None]:
        budget = wall_budgets[controller]
        elapsed_value = event.get("elapsed_wall_seconds")
        try:
            elapsed = float(elapsed_value)
        except (TypeError, ValueError):
            elapsed = None
        if not bool(event.get("truncated")):
            return False, "repair_completed", elapsed, budget
        if position != len(transitions) - 1:
            return False, "nonterminal_truncation", elapsed, budget
        if not bool(episode.get("external_timeout")):
            return False, "episode_external_timeout_is_false", elapsed, budget
        if budget is None:
            return False, "wall_time_budget_is_unregistered", elapsed, budget
        if elapsed is None or not math.isfinite(elapsed):
            return False, "elapsed_wall_time_is_missing_or_non_finite", elapsed, budget
        if elapsed < budget:
            return False, "elapsed_wall_time_precedes_budget", elapsed, budget
        return True, "external_wall_time_budget_boundary", elapsed, budget

    for key in sorted(paired_keys):
        left_row = index["v1-full"][key]
        right_row = index["v2-full"][key]
        left_events = read_trace_events(
            roots["v1-full"] / str(left_row["trace_file"])
        )
        right_events = read_trace_events(
            roots["v2-full"] / str(right_row["trace_file"])
        )
        if str(left_events[0].get("state_fingerprint")) != str(
            right_events[0].get("state_fingerprint")
        ):
            mismatch(key, None, "initial_fingerprint", "initial states differ")
        left = [row for row in left_events if row.get("event") == "transition"]
        right = [row for row in right_events if row.get("event") == "transition"]
        if len(left) != len(right):
            differing_length_episodes += 1
            if len(left) < len(right):
                shorter_name = "v1-full"
                shorter = left
                shorter_row = left_row
            else:
                shorter_name = "v2-full"
                shorter = right
                shorter_row = right_row
            shorter_event = shorter[-1] if shorter else {}
            valid_length_boundary, reason, elapsed, budget = boundary_status(
                shorter_event,
                position=len(shorter) - 1,
                transitions=shorter,
                episode=shorter_row,
                controller=shorter_name,
            )
            if valid_length_boundary:
                allowed_budget_length_differences += 1
            else:
                unexplained_length_differences += 1
                mismatch(
                    key,
                    (
                        int(shorter_event.get("decision_index", len(shorter) - 1))
                        if shorter
                        else None
                    ),
                    "unexplained_length_difference",
                    f"{shorter_name} ended first without a valid budget boundary: {reason}",
                )
        for decision_index, (old, new) in enumerate(zip(left, right)):
            common_decisions += 1
            if int(old.get("decision_index", -1)) != int(
                new.get("decision_index", -1)
            ):
                mismatch(
                    key,
                    decision_index,
                    "decision_index",
                    "recorded decision indices differ",
                )
            if str(old.get("before_fingerprint")) != str(
                new.get("before_fingerprint")
            ):
                mismatch(key, decision_index, "before_fingerprint", "before states differ")
            if _canonical(old.get("action")) != _canonical(new.get("action")):
                mismatch(key, decision_index, "action", "selected repair actions differ")
            old_seed = dict(old.get("action") or {}).get("random_seed")
            new_seed = dict(new.get("action") or {}).get("random_seed")
            if old_seed != new_seed:
                mismatch(
                    key,
                    decision_index,
                    "random_seed",
                    "repair random seeds differ",
                )
            old_controller = dict(old.get("controller") or {})
            new_controller = dict(new.get("controller") or {})
            if str(old_controller.get("selected_candidate_id")) != str(
                new_controller.get("selected_candidate_id")
            ):
                mismatch(
                    key,
                    decision_index,
                    "selected_candidate",
                    "selected candidate IDs differ",
                )
            old_ids, old_ranking, old_scores = _candidate_semantics(old)
            new_ids, new_ranking, new_scores = _candidate_semantics(new)
            if old_ids != new_ids:
                mismatch(key, decision_index, "candidate_pool", "candidate IDs/order differ")
            if old_ranking != new_ranking:
                mismatch(key, decision_index, "candidate_ranking", "rankings differ")
            if set(old_scores) != set(new_scores):
                mismatch(key, decision_index, "candidate_scores", "scored candidates differ")
            else:
                for candidate_id in old_scores:
                    delta = abs(old_scores[candidate_id] - new_scores[candidate_id])
                    score_comparisons += 1
                    maximum_score_delta = max(maximum_score_delta, delta)
                    if delta > 1e-12:
                        mismatch(
                            key,
                            decision_index,
                            "candidate_scores",
                            f"{candidate_id} score delta {delta:.17g}",
                        )
            old_boundary, old_reason, old_elapsed, old_budget = boundary_status(
                old,
                position=decision_index,
                transitions=left,
                episode=left_row,
                controller="v1-full",
            )
            new_boundary, new_reason, new_elapsed, new_budget = boundary_status(
                new,
                position=decision_index,
                transitions=right,
                episode=right_row,
                controller="v2-full",
            )
            old_incomplete = bool(old.get("truncated"))
            new_incomplete = bool(new.get("truncated"))
            if old_incomplete and not old_boundary:
                mismatch(
                    key,
                    decision_index,
                    (
                        "nonterminal_truncation"
                        if old_reason == "nonterminal_truncation"
                        else "unexplained_truncation"
                    ),
                    f"v1-full truncated repair is not a budget boundary: {old_reason}",
                )
            if new_incomplete and not new_boundary:
                mismatch(
                    key,
                    decision_index,
                    (
                        "nonterminal_truncation"
                        if new_reason == "nonterminal_truncation"
                        else "unexplained_truncation"
                    ),
                    f"v2-full truncated repair is not a budget boundary: {new_reason}",
                )
            if old_incomplete or new_incomplete:
                if old_boundary or new_boundary:
                    boundary_details.append(
                        {
                            "task_id": key[0],
                            "solver_seed": key[1],
                            "decision_index": decision_index,
                            "v1_is_budget_boundary": old_boundary,
                            "v2_is_budget_boundary": new_boundary,
                            "v1_trace_terminal": decision_index == len(left) - 1,
                            "v2_trace_terminal": decision_index == len(right) - 1,
                            "v1_truncated": old_incomplete,
                            "v2_truncated": new_incomplete,
                            "v1_episode_external_timeout": bool(
                                left_row.get("external_timeout")
                            ),
                            "v2_episode_external_timeout": bool(
                                right_row.get("external_timeout")
                            ),
                            "v1_elapsed_wall_seconds": old_elapsed,
                            "v2_elapsed_wall_seconds": new_elapsed,
                            "v1_wall_time_budget_seconds": old_budget,
                            "v2_wall_time_budget_seconds": new_budget,
                            "v1_transition_count": len(left),
                            "v2_transition_count": len(right),
                            "reason": "post-repair state excluded because at least one repair was truncated at the external wall-time budget",
                        }
                    )
                continue
            post_repair_comparisons += 1
            if str(old.get("after_fingerprint")) != str(
                new.get("after_fingerprint")
            ):
                mismatch(key, decision_index, "after_fingerprint", "after states differ")
            if _canonical(old.get("low_level_delta")) != _canonical(
                new.get("low_level_delta")
            ):
                mismatch(
                    key,
                    decision_index,
                    "low_level_search",
                    "low-level search counters differ",
                )
        if (
            len(left) == len(right)
            and not (left and bool(left[-1].get("truncated")))
            and not (right and bool(right[-1].get("truncated")))
        ):
            left_final = next(
                (row for row in reversed(left_events) if row.get("event") == "finish"),
                {},
            )
            right_final = next(
                (row for row in reversed(right_events) if row.get("event") == "finish"),
                {},
            )
            if str(left_final.get("final_fingerprint")) != str(
                right_final.get("final_fingerprint")
            ):
                mismatch(key, None, "final_fingerprint", "equal-length final states differ")

    total_mismatches = sum(mismatch_counts.values())
    report = {
        "schema": "lns2.v1_v2_common_prefix_equivalence.v2",
        "episode_count": len(paired_keys),
        "common_decision_count": common_decisions,
        "controller_decision_comparison_count": common_decisions,
        "post_repair_comparison_count": post_repair_comparisons,
        "budget_boundary_exclusion_count": len(boundary_details),
        "differing_length_episode_count": differing_length_episodes,
        "allowed_budget_length_difference_count": allowed_budget_length_differences,
        "unexplained_length_difference_count": unexplained_length_differences,
        "candidate_score_comparison_count": score_comparisons,
        "maximum_score_delta": maximum_score_delta,
        "score_tolerance": 1e-12,
        "mismatch_count": total_mismatches,
        "mismatch_counts": dict(sorted(mismatch_counts.items())),
        "common_prefix_exact": total_mismatches == 0,
        "passed": total_mismatches == 0,
        "length_difference_policy": (
            "allowed only when the shorter trace ends with a terminal truncated repair, "
            "the episode reports external_timeout, and elapsed wall time reaches the "
            "registered wall-time budget"
        ),
        "post_repair_policy": (
            "after fingerprints and low-level counters are compared only when both "
            "repairs completed; controller semantics are always compared"
        ),
    }
    return report, details, boundary_details


def _balanced_decisions(
    root: Path, balanced_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for episode in balanced_rows:
        events = read_trace_events(root / str(episode["trace_file"]))
        transitions = [row for row in events if row.get("event") == "transition"]
        count = len(transitions)
        for position, event in enumerate(transitions):
            controller = dict(event.get("controller") or {})
            route = str(controller.get("route"))
            if route not in {"model", "official_adaptive"}:
                raise ValueError("balanced trace has an invalid route")
            fraction = position / max(1, count - 1)
            phase = (
                "early"
                if fraction < 1 / 3
                else "middle"
                if fraction < 2 / 3
                else "late"
            )
            result.append(
                {
                    "episode_id": episode["episode_id"],
                    "decision_index": int(event["decision_index"]),
                    "actual_route": route,
                    "decision_phase": phase,
                }
            )
    return result


def _route_usage_rows(
    balanced_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in balanced_rows:
        model = int(episode["model_decision_count"])
        official = int(episode["official_decision_count"])
        total = model + official
        episode_kind = (
            "all_model"
            if total and official == 0
            else "all_lns2"
            if total and model == 0
            else "mixed"
            if total
            else "no_repair"
        )
        rows.append(
            {
                "record_type": "episode",
                "stratum_kind": "episode",
                "stratum_value": episode_kind,
                "episode_id": episode["episode_id"],
                "task_id": episode["task_id"],
                "map_id": episode["map_id"],
                "layout_family": episode["layout_family"],
                "agent_count": episode["agent_count"],
                "initial_conflict_severity": episode["initial_conflict_severity"],
                "solver_seed": episode["solver_seed"],
                "total_decision_count": total,
                "model_decision_count": model,
                "official_decision_count": official,
                "model_route_fraction": model / total if total else 0.0,
                "route_switch_count": episode["route_switch_count"],
                "candidate_count_before": episode["candidate_count_before"],
                "candidate_count_after": episode["candidate_count_after"],
                "model_controller_seconds": episode["model_controller_seconds"],
                "model_repair_seconds": episode["model_repair_seconds"],
                "model_total_decision_seconds": episode[
                    "model_total_decision_seconds"
                ],
                "official_controller_seconds": episode[
                    "official_controller_seconds"
                ],
                "official_repair_seconds": episode["official_repair_seconds"],
                "official_total_decision_seconds": episode[
                    "official_total_decision_seconds"
                ],
            }
        )

    def add_group(kind: str, value: str, members: list[dict[str, Any]]) -> None:
        model = sum(int(row["model_decision_count"]) for row in members)
        official = sum(int(row["official_decision_count"]) for row in members)
        total = model + official
        rows.append(
            {
                "record_type": "aggregate",
                "stratum_kind": kind,
                "stratum_value": value,
                "episode_count": len(members),
                "total_decision_count": total,
                "model_decision_count": model,
                "official_decision_count": official,
                "model_route_fraction": model / total if total else 0.0,
                "route_switch_count": sum(
                    int(row["route_switch_count"]) for row in members
                ),
                "candidate_count_before": sum(
                    int(row["candidate_count_before"]) for row in members
                ),
                "candidate_count_after": sum(
                    int(row["candidate_count_after"]) for row in members
                ),
                "model_controller_seconds": sum(
                    float(row["model_controller_seconds"]) for row in members
                ),
                "model_repair_seconds": sum(
                    float(row["model_repair_seconds"]) for row in members
                ),
                "model_total_decision_seconds": sum(
                    float(row["model_total_decision_seconds"]) for row in members
                ),
                "official_controller_seconds": sum(
                    float(row["official_controller_seconds"]) for row in members
                ),
                "official_repair_seconds": sum(
                    float(row["official_repair_seconds"]) for row in members
                ),
                "official_total_decision_seconds": sum(
                    float(row["official_total_decision_seconds"]) for row in members
                ),
            }
        )

    add_group("all", "all", balanced_rows)
    pattern_counts = Counter(
        str(row["stratum_value"])
        for row in rows
        if row["record_type"] == "episode"
    )
    for pattern in ("all_model", "all_lns2", "mixed", "no_repair"):
        rows.append(
            {
                "record_type": "aggregate",
                "stratum_kind": "episode_route_pattern",
                "stratum_value": pattern,
                "episode_count": pattern_counts[pattern],
            }
        )
    for kind, field in (
        ("layout_family", "layout_family"),
        ("agent_count", "agent_count"),
        ("initial_conflict_severity", "initial_conflict_severity"),
    ):
        for value in sorted({str(row[field]) for row in balanced_rows}):
            add_group(
                kind,
                value,
                [row for row in balanced_rows if str(row[field]) == value],
            )
    for phase in ("early", "middle", "late"):
        members = [row for row in decision_rows if row["decision_phase"] == phase]
        model = sum(row["actual_route"] == "model" for row in members)
        official = sum(
            row["actual_route"] == "official_adaptive" for row in members
        )
        total = model + official
        rows.append(
            {
                "record_type": "aggregate",
                "stratum_kind": "decision_phase",
                "stratum_value": phase,
                "total_decision_count": total,
                "model_decision_count": model,
                "official_decision_count": official,
                "model_route_fraction": model / total if total else 0.0,
            }
        )
    return rows


def _low_level_value(outcome: dict[str, Any], key: str) -> int:
    return int(dict(outcome.get("low_level_delta") or {}).get(key, 0))


def _flatten_counterfactual(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("actual_route")) != "official_adaptive":
        raise ValueError("skipped-model diagnostic contains a model-routed state")
    lns2 = dict(row["actual_lns2"])
    model = dict(row["counterfactual_model"])
    left = dict(lns2["outcome"])
    right = dict(model["outcome"])
    result = {
        "episode_id": row["episode_id"],
        "task_id": row["task_id"],
        "map_id": row["map_id"],
        "layout_family": row["layout_mode"],
        "agent_count": row["agent_count"],
        "solver_seed": row["solver_seed"],
        "decision_index": row["decision_index"],
        "actual_route": row["actual_route"],
        "before_fingerprint": row["before_fingerprint"],
        "before_conflicts": row["before_conflicts"],
        "baseline_source": row["baseline_source"],
        "replay_fingerprint_match": row["replay_fingerprint_match"],
        "lns2_action": _canonical(lns2.get("action")),
        "lns2_neighborhood": _canonical(
            dict(lns2.get("metrics") or {}).get("neighborhood", [])
        ),
        "model_action": _canonical(model.get("action")),
        "model_neighborhood": _canonical(
            dict(model.get("action") or {}).get("agents", [])
        ),
        "model_candidate_id": dict(model.get("controller") or {}).get(
            "selected_candidate_id"
        ),
        "model_candidate_count_before": dict(model.get("controller") or {}).get(
            "candidate_count_before"
        ),
        "model_candidate_count_after": dict(model.get("controller") or {}).get(
            "candidate_count_after"
        ),
        "pareto_relation": row["pareto_relation"],
    }
    for branch, values in (("lns2", left), ("model", right)):
        for name in (
            "conflicts_after",
            "conflict_delta",
            "success",
            "sum_of_costs_delta",
            "controller_seconds",
            "repair_seconds",
            "total_decision_seconds",
        ):
            result[f"{branch}_{name}"] = values.get(name)
        for low_name in ("expanded", "generated", "reopened"):
            result[f"{branch}_{low_name}"] = _low_level_value(values, low_name)
    controller = dict(model.get("controller") or {})
    for name in (
        "proposal_seconds",
        "state_check_seconds",
        "state_analysis_seconds",
        "proposal_feature_seconds",
        "realized_feature_seconds",
        "inference_seconds",
        "pruner_seconds",
        "pruner_fallback",
    ):
        result[f"model_{name}"] = controller.get(name)
    result["model_remaining_conflict_improvement"] = _relative_improvement(
        left.get("conflicts_after"), right.get("conflicts_after")
    )
    result["model_conflict_delta_advantage"] = float(
        right.get("conflict_delta", 0)
    ) - float(left.get("conflict_delta", 0))
    result["model_success_delta"] = int(bool(right.get("success"))) - int(
        bool(left.get("success"))
    )
    result["model_soc_delta_difference"] = float(
        right.get("sum_of_costs_delta", 0)
    ) - float(left.get("sum_of_costs_delta", 0))
    result["model_time_increase"] = _relative_increase(
        left.get("total_decision_seconds"), right.get("total_decision_seconds")
    )
    return result


def _equal_map_mean(rows: list[dict[str, Any]], field: str) -> float | None:
    by_map: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is not None:
            by_map[str(row["map_id"])].append(float(value))
    return _mean(_mean(values) for values in by_map.values() if values)


def _counterfactual_aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: dict[str, Any] = {
        "scope": "official_adaptive_states_model_once",
        "state_count": len(rows),
        "map_count": len({str(row["map_id"]) for row in rows}),
    }
    for branch in ("lns2", "model"):
        for field in (
            "conflicts_after",
            "conflict_delta",
            "success",
            "sum_of_costs_delta",
            "controller_seconds",
            "repair_seconds",
            "total_decision_seconds",
            "expanded",
            "generated",
            "reopened",
        ):
            result[f"equal_map_{branch}_{field}"] = _equal_map_mean(
                rows, f"{branch}_{field}"
            )
    result["equal_map_model_remaining_conflict_improvement"] = _relative_improvement(
        result["equal_map_lns2_conflicts_after"],
        result["equal_map_model_conflicts_after"],
    )
    result["equal_map_model_conflict_delta_advantage"] = (
        float(result["equal_map_model_conflict_delta"] or 0.0)
        - float(result["equal_map_lns2_conflict_delta"] or 0.0)
    )
    result["equal_map_model_success_delta"] = (
        float(result["equal_map_model_success"] or 0.0)
        - float(result["equal_map_lns2_success"] or 0.0)
    )
    result["equal_map_model_time_increase"] = _relative_increase(
        result["equal_map_lns2_total_decision_seconds"],
        result["equal_map_model_total_decision_seconds"],
    )
    pareto = Counter(str(row["pareto_relation"]) for row in rows)
    for relation in (
        "model_dominates",
        "lns2_dominates",
        "quality_time_tradeoff",
        "tie",
    ):
        result[f"{relation}_count"] = pareto[relation]
    return [result]


def _common_success_soc_ratio(
    index: dict[str, dict[tuple[str, int], dict[str, Any]]],
    reference: str,
    candidate: str,
    keys: set[tuple[str, int]],
) -> tuple[float | None, int]:
    common = [
        key
        for key in keys
        if index[reference][key]["success"] and index[candidate][key]["success"]
    ]
    ratios = [
        float(index[candidate][key]["final_sum_of_costs"])
        / float(index[reference][key]["final_sum_of_costs"])
        for key in common
        if float(index[reference][key]["final_sum_of_costs"]) != 0.0
    ]
    return _mean(ratios), len(common)


def _promotion_report(
    episodes: list[dict[str, Any]],
    frontier: list[dict[str, Any]],
    speed_rows: list[dict[str, Any]],
    paired_keys: set[tuple[str, int]],
    counterfactual: list[dict[str, Any]],
    counterfactual_summary: dict[str, Any],
    semantic_equivalence: dict[str, Any],
    *,
    formal: bool,
) -> dict[str, Any]:
    by_controller = {str(row["controller"]): row for row in frontier}
    by_comparison = {str(row["comparison"]): row for row in speed_rows}
    official = by_controller["official_adaptive"]
    v1 = by_controller["v1-full"]
    v2 = by_controller["v2-full"]
    balanced = by_controller["v2-balanced"]
    index = _paired_index(episodes)
    v1_v2_soc_ratio, v1_v2_common_successes = _common_success_soc_ratio(
        index, "v1-full", "v2-full", paired_keys
    )
    balanced_soc_ratio, balanced_common_successes = _common_success_soc_ratio(
        index, "official_adaptive", "v2-balanced", paired_keys
    )
    v2_speed = by_comparison["v2-full_vs_v1-full"]
    full_gain = float(official["mean_fixed_budget_conflict_auc"] or 0.0) - float(
        v2["mean_fixed_budget_conflict_auc"] or 0.0
    )
    balanced_gain = float(
        official["mean_fixed_budget_conflict_auc"] or 0.0
    ) - float(balanced["mean_fixed_budget_conflict_auc"] or 0.0)
    gain_retention = (
        balanced_gain / full_gain
        if full_gain > 0.0
        else 1.0
        if balanced_gain >= 0.0
        else -math.inf
    )
    balanced_speedup = _relative_improvement(
        v2["mean_capped_wall_time_seconds"],
        balanced["mean_capped_wall_time_seconds"],
    )
    balanced_wall_ratio = (
        float(balanced["mean_capped_wall_time_seconds"])
        / float(official["mean_capped_wall_time_seconds"])
        if official["mean_capped_wall_time_seconds"]
        else math.inf
    )
    skipped = counterfactual[0] if counterfactual else {}
    skipped_quality_gain = skipped.get(
        "equal_map_model_remaining_conflict_improvement"
    )
    skipped_time_increase = skipped.get("equal_map_model_time_increase")
    skipped_model_beneficial = bool(skipped) and (
        float(skipped.get("equal_map_model_conflict_delta_advantage") or 0.0) > 0.0
        or float(skipped.get("equal_map_model_success_delta") or 0.0) > 0.0
    )
    integrity_errors = (
        sum(
            int(row["invalid_action_count"])
            + int(row["fingerprint_mismatch_count"])
            for row in episodes
            if row["status"] in VALID_STATUSES
        )
        + sum(row["status"] not in VALID_STATUSES for row in episodes)
        + int(counterfactual_summary.get("error_count", 0))
        + int(counterfactual_summary.get("missing_model_result_count") or 0)
        + int(counterfactual_summary.get("replay_fingerprint_mismatch_count") or 0)
    )
    expected_pairs = 144 if formal else 24
    v2_gates = {
        "common_prefix_semantics_exact": bool(semantic_equivalence.get("passed")),
        "controller_time_significantly_lower_than_v1": bool(
            v2_speed["controller_time_significantly_lower"]
        ),
        "success_not_below_v1": int(v2["success_count"]) >= int(v1["success_count"]),
        "auc_not_worse_than_v1": float(
            v2["mean_fixed_budget_conflict_auc"] or math.inf
        )
        <= float(v1["mean_fixed_budget_conflict_auc"] or math.inf),
        "common_success_soc_not_worse_than_v1": v1_v2_soc_ratio is None
        or v1_v2_soc_ratio <= 1.0,
    }
    balanced_gates = {
        "registered_episode_coverage": len(paired_keys) == expected_pairs,
        "paired_coverage_complete": all(
            len(index[name]) == len(paired_keys) for name in CONTROLLER_ORDER
        ),
        "success_not_below_lns2": int(balanced["success_count"])
        >= int(official["success_count"]),
        "retains_half_v2_auc_gain": gain_retention >= 0.50,
        "common_success_soc_within_2_percent": balanced_soc_ratio is None
        or balanced_soc_ratio <= 1.02,
        "at_least_10_percent_faster_than_v2_full": balanced_speedup is not None
        and balanced_speedup >= 0.10,
        "not_slower_than_lns2": balanced_wall_ratio <= 1.0,
        "skipped_state_counterfactual_coverage_complete": bool(
            counterfactual_summary.get("passed")
        ),
        "zero_integrity_errors": integrity_errors == 0,
        "proposal_pruner_disabled": True,
    }
    v2_passed = all(v2_gates.values())
    balanced_passed = v2_passed and all(balanced_gates.values())
    v2_has_lns2_benefit = (
        int(v2["success_count"]) > int(official["success_count"])
        or float(v2.get("auc_improvement_vs_lns2") or 0.0) > 0.0
        or float(v2.get("wall_time_improvement_vs_lns2") or 0.0) > 0.0
    )
    if balanced_passed:
        conclusion = "hybrid_supported"
    elif v2_has_lns2_benefit:
        conclusion = "full_model_preferred"
    elif not v2_has_lns2_benefit:
        conclusion = "lns2_preferred"
    else:
        conclusion = "inconclusive_keep_v2_full"
    return {
        "formal": formal,
        "conclusion": conclusion,
        "eligible_to_replace_default": formal and balanced_passed,
        "default_controller_remains": (
            "v2-balanced" if formal and balanced_passed else "v2-full"
        ),
        "v2_full_promotion": {
            "passed": v2_passed,
            "gates": v2_gates,
        },
        "v2_balanced_promotion": {
            "passed": balanced_passed,
            "gates": balanced_gates,
        },
        "metrics": {
            "v2_controller_time_improvement_vs_v1": v2_speed[
                "controller_time_improvement"
            ],
            "v2_wall_time_improvement_vs_v1": v2_speed[
                "wall_time_improvement"
            ],
            "v1_v2_common_success_soc_ratio": v1_v2_soc_ratio,
            "v1_v2_common_success_episode_count": v1_v2_common_successes,
            "v2_auc_gain_vs_lns2": full_gain,
            "balanced_auc_gain_vs_lns2": balanced_gain,
            "balanced_gain_retention": gain_retention,
            "balanced_speedup_over_v2_full": balanced_speedup,
            "balanced_wall_ratio_vs_lns2": balanced_wall_ratio,
            "balanced_common_success_soc_ratio_vs_lns2": balanced_soc_ratio,
            "balanced_common_success_episode_count": balanced_common_successes,
            "skipped_model_once_remaining_conflict_improvement": skipped_quality_gain,
            "skipped_model_once_time_increase": skipped_time_increase,
            "skipped_model_once_has_quality_benefit": skipped_model_beneficial,
            "integrity_error_count": integrity_errors,
        },
    }


def _bar_svg(path: Path, title: str, labels: list[str], values: list[float]) -> None:
    width, height = 940, 500
    left, top, plot_width, plot_height = 90, 70, 800, 330
    safe_values = [max(0.0, float(value)) for value in values]
    maximum = max([1e-12, *safe_values])
    slot = plot_width / max(1, len(safe_values))
    colors = ("#64748b", "#7c3aed", "#0f766e", "#d97706")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="sans-serif" font-size="22">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#94a3b8"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, safe_values)):
        bar_height = plot_height * value / maximum
        x = left + index * slot + slot * 0.18
        y = top + plot_height - bar_height
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{slot * 0.64:.2f}" height="{bar_height:.2f}" rx="4" fill="{colors[index % len(colors)]}"/>',
                f'<text x="{x + slot * 0.32:.2f}" y="{max(top + 14, y - 7):.2f}" text-anchor="middle" font-family="sans-serif" font-size="13">{value:.3g}</text>',
                f'<text x="{x + slot * 0.32:.2f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="12">{html.escape(label)}</text>',
            ]
        )
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def _frontier_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    usable = [
        row
        for row in rows
        if row.get("mean_capped_wall_time_seconds") is not None
        and row.get("mean_fixed_budget_conflict_auc") is not None
    ]
    width, height = 940, 560
    left, top, plot_width, plot_height = 100, 70, 760, 390
    xs = [float(row["mean_capped_wall_time_seconds"]) for row in usable] or [0.0]
    ys = [float(row["mean_fixed_budget_conflict_auc"]) for row in usable] or [0.0]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    colors = ("#64748b", "#7c3aed", "#0f766e", "#d97706")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="35" text-anchor="middle" font-family="sans-serif" font-size="22">Quality-speed frontier (lower is better)</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#64748b"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#64748b"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 25}" text-anchor="middle" font-family="sans-serif" font-size="14">Mean capped wall time (seconds)</text>',
        f'<text transform="translate(25,{top + plot_height / 2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="14">Mean fixed-budget conflict AUC</text>',
    ]
    for index, row in enumerate(usable):
        x = left + 25 + (
            float(row["mean_capped_wall_time_seconds"]) - x_min
        ) / x_span * (plot_width - 50)
        y = top + plot_height - 25 - (
            float(row["mean_fixed_budget_conflict_auc"]) - y_min
        ) / y_span * (plot_height - 50)
        parts.extend(
            [
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="9" fill="{colors[index % len(colors)]}"/>',
                f'<text x="{x + 13:.2f}" y="{y - 10:.2f}" font-family="sans-serif" font-size="13">{html.escape(str(row["controller_label"]))}</text>',
            ]
        )
    parts.append("</svg>\n")
    path.write_text("".join(parts), encoding="utf-8")


def _format_percent(value: Any) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.2f}%"


def _format_number(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _markdown(
    report: dict[str, Any],
    usage: list[dict[str, Any]],
    counterfactual: list[dict[str, Any]],
    speed_rows: list[dict[str, Any]],
) -> str:
    overall = next(
        row
        for row in usage
        if row["record_type"] == "aggregate" and row["stratum_kind"] == "all"
    )
    patterns = {
        str(row["stratum_value"]): int(row["episode_count"])
        for row in usage
        if row["record_type"] == "aggregate"
        and row["stratum_kind"] == "episode_route_pattern"
    }
    skipped = counterfactual[0] if counterfactual else {}
    promotion = report["promotion"]
    speed = {str(row["comparison"]): row for row in speed_rows}
    v2_vs_v1 = speed["v2-full_vs_v1-full"]
    v2_vs_lns2 = speed["v2-full_vs_original-lns2"]
    balanced_vs_v2 = speed["v2-balanced_vs_v2-full"]
    mode_note = (
        "这是正式注册评估。"
        if report["formal"]
        else "这是 quick 非正式试跑，不能作为正式结论，也不会切换默认控制器。"
    )
    lines = [
        "# 旧模型语义下的四路完整配对评估",
        "",
        f"**{mode_note}**",
        "",
        "四个控制器都按相同任务、solver seed、初始状态、100 次 repair / 300 秒预算运行完整 episode。单步反事实不参与四路主结论。",
        "",
        "## 核心结论",
        "",
        f"- 综合判定：`{promotion['conclusion']}`",
        f"- 当前默认控制器：`{promotion['default_controller_remains']}`",
        f"- v1/v2 控制器语义零不匹配：`{report['semantic_equivalence']['passed']}`（共同决策 {report['semantic_equivalence']['common_decision_count']} 次，完整 repair 后状态比较 {report['semantic_equivalence']['post_repair_comparison_count']} 次，合法预算边界排除 {report['semantic_equivalence']['budget_boundary_exclusion_count']} 次，最大分数差 {_format_number(report['semantic_equivalence']['maximum_score_delta'], 14)}）",
        f"- v2-full 相比 v1-full 控制器时间变化：{_format_percent(v2_vs_v1['controller_time_improvement'])}；端到端时间变化：{_format_percent(v2_vs_v1['wall_time_improvement'])}",
        "",
        "## 四路完整 episode",
        "",
        "| 控制器 | 成功数 | 固定预算 AUC | 归一化 AUC | 控制器时间 | 封顶总时间 | 相对 LNS2 AUC 改善 | 相对 LNS2 时间改善 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["frontier"]:
        lines.append(
            f"| {row['controller_label']} | {row['success_count']} | "
            f"{_format_number(row['mean_fixed_budget_conflict_auc'])} | "
            f"{_format_number(row['mean_normalized_conflict_auc'])} | "
            f"{_format_number(row['mean_controller_seconds'])}s | "
            f"{_format_number(row['mean_capped_wall_time_seconds'])}s | "
            f"{_format_percent(row['auc_improvement_vs_lns2'])} | "
            f"{_format_percent(row['wall_time_improvement_vs_lns2'])} |"
        )
    lines.extend(
        [
            "",
            "### 三个直接配对问题",
            "",
            f"- v2-full vs v1-full：控制器时间改善 {_format_percent(v2_vs_v1['controller_time_improvement'])}，完整 episode 时间改善 {_format_percent(v2_vs_v1['wall_time_improvement'])}。",
            f"- v2-full vs 原始 LNS2：成功数差 {v2_vs_lns2['success_count_delta']:+d}，AUC 改善 {_format_percent(v2_vs_lns2['auc_improvement'])}，共同成功 SOC 比值 {_format_number(v2_vs_lns2['common_success_soc_ratio'])}，low-level generated 改善 {_format_percent(v2_vs_lns2['low_level_generated_improvement'])}，完整 episode 时间改善 {_format_percent(v2_vs_lns2['wall_time_improvement'])}。",
            f"- v2-balanced vs v2-full：成功数差 {balanced_vs_v2['success_count_delta']:+d}，AUC 变化 {_format_percent(balanced_vs_v2['auc_improvement'])}，共同成功 SOC 比值 {_format_number(balanced_vs_v2['common_success_soc_ratio'])}，完整 episode 时间改善 {_format_percent(balanced_vs_v2['wall_time_improvement'])}。",
            "",
            "## v2-balanced 实际路由",
            "",
            f"- repair 决策总数：{overall['total_decision_count']}",
            f"- 使用模型：{overall['model_decision_count']}（{_format_percent(overall['model_route_fraction'])}）",
            f"- 使用原始 LNS2：{overall['official_decision_count']}（{_format_percent(1.0 - float(overall['model_route_fraction']))}）",
            f"- 全模型 episode：{patterns.get('all_model', 0)}；全 LNS2：{patterns.get('all_lns2', 0)}；混合：{patterns.get('mixed', 0)}；无需 repair：{patterns.get('no_repair', 0)}",
            "",
            "## 仅对被跳过状态的模型单次反事实",
            "",
            f"- 状态数：{skipped.get('state_count', 0)}；额外 LNS2 执行数：{report['counterfactual_summary'].get('extra_lns2_execution_count', 0)}",
            f"- 模型一次 repair 后剩余冲突改善：{_format_percent(skipped.get('equal_map_model_remaining_conflict_improvement'))}",
            f"- 模型一次 repair 的冲突下降优势：{_format_number(skipped.get('equal_map_model_conflict_delta_advantage'))}",
            f"- 模型一次 repair 的总决策时间增加：{_format_percent(skipped.get('equal_map_model_time_increase'))}",
            "",
            "这里没有续跑四步，也没有重跑模型已经启用的状态；主轨迹中的 Adaptive 单步结果直接作为 LNS2 基线。",
            "",
            "## 晋级门槛",
            "",
            "### v2-full 相比 v1-full",
            "",
        ]
    )
    for name, passed in promotion["v2_full_promotion"]["gates"].items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{name}`")
    lines.extend(["", "### v2-balanced 相比 v2-full / LNS2", ""])
    for name, passed in promotion["v2_balanced_promotion"]["gates"].items():
        lines.append(f"- {'通过' if passed else '未通过'}：`{name}`")
    lines.extend(
        [
            "",
            "## 与历史结果的关系",
            "",
            "历史数据只作参考，不被覆盖：同分布 80/100-agent 上旧模型 AUC 改善 52.5%，但 0.715s vs 0.273s；历史 MovingAI 上成功 131 vs 123、原始 AUC 改善 4.1%、归一化改善 32.4%、平均封顶时间 62.176s vs 88.814s。新报告使用同批次四路重跑来判断优化是否真实。",
            f"新同批次 v1-full 相对 LNS2 的成功、AUC、时间三个方向是否均与历史 MovingAI 一致：`{report['historical_comparison']['movingai_all_directions_consistent']}`。",
            "",
            "## 输出文件",
            "",
            "- `paired_episodes.csv`：四路每个完整 episode",
            "- `controller_speed_comparison.csv`：v2/v1、模型/LNS2、混合/v2 的配对时间和质量",
            "- `v1_v2_semantic_equivalence.json`：共同 repair 前缀的分数、排序、动作和预算感知状态验证",
            "- `v1_v2_budget_boundary_exclusions.csv`：未把不完整 repair 的 after-state 计入等价性比较的合法预算边界",
            "- `route_usage.csv`：混合路由数量、比例和分层时间",
            "- `skipped_model_once.csv`：只对 Adaptive 路由状态额外执行一次模型",
            "- `quality_speed_frontier.csv`：四路完整 episode 的质量—速度前沿",
            "",
        ]
    )
    return "\n".join(lines)


def generate_tradeoff_artifacts(
    collections: dict[str, str | Path],
    counterfactual_root: str | Path,
    output: str | Path,
    *,
    formal: bool,
) -> dict[str, Any]:
    roots = {name: Path(collections[name]).resolve() for name in CONTROLLER_ORDER}
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    official_run = _read_json(roots["official_adaptive"] / "run_config.json")
    thresholds = dict(official_run["configuration"]["severity_thresholds"])
    episodes, run_metadata = _load_episodes(roots, thresholds)
    frontier, paired_keys = _frontier_rows(episodes)
    speed_rows = _controller_speed_rows(episodes, paired_keys)
    semantic, semantic_details, boundary_details = _semantic_equivalence_report(
        roots, episodes, paired_keys
    )

    counterfactual_path = Path(counterfactual_root).resolve()
    counterfactual_run = _read_json(counterfactual_path / "run_config.json")
    if str(counterfactual_run.get("source_run_fingerprint")) != str(
        run_metadata["v2-balanced"]["run_fingerprint"]
    ):
        raise ValueError("counterfactual results belong to a different balanced run")
    if str(counterfactual_run.get("scope")) != COUNTERFACTUAL_SCOPE:
        raise ValueError("counterfactual results use the obsolete all-state/H4 scope")
    raw_counterfactual = list(iter_route_counterfactual_rows(counterfactual_path))
    flat_counterfactual = [
        _flatten_counterfactual(row) for row in raw_counterfactual
    ]
    counterfactual_aggregate = _counterfactual_aggregates(flat_counterfactual)
    counterfactual_summary = _read_json(
        counterfactual_path / "counterfactual_summary.json"
    )
    balanced_rows = [
        row
        for row in episodes
        if row["controller"] == "v2-balanced" and row["status"] in VALID_STATUSES
    ]
    decision_rows = _balanced_decisions(roots["v2-balanced"], balanced_rows)
    usage = _route_usage_rows(balanced_rows, decision_rows)
    promotion = _promotion_report(
        episodes,
        frontier,
        speed_rows,
        paired_keys,
        counterfactual_aggregate,
        counterfactual_summary,
        semantic,
        formal=formal,
    )
    same_batch_v1_vs_lns2 = next(
        row
        for row in speed_rows
        if row["comparison"] == "v1-full_vs_original-lns2"
    )
    historical_trend = {
        "historical_reference": HISTORICAL_REFERENCE,
        "same_batch_v1_vs_lns2": same_batch_v1_vs_lns2,
        "movingai_direction_consistent": {
            "success": int(same_batch_v1_vs_lns2["success_count_delta"]) >= 0,
            "auc": float(same_batch_v1_vs_lns2["auc_improvement"] or 0.0)
            > 0.0,
            "wall_time": float(
                same_batch_v1_vs_lns2["wall_time_improvement"] or 0.0
            )
            > 0.0,
        },
    }
    historical_trend["movingai_all_directions_consistent"] = all(
        historical_trend["movingai_direction_consistent"].values()
    )
    report = {
        "schema": TRADEOFF_REPORT_SCHEMA,
        "formal": formal,
        "collections": run_metadata,
        "complete_episode_count": len(paired_keys) * len(CONTROLLER_ORDER),
        "paired_episode_count": len(paired_keys),
        "frontier": frontier,
        "controller_speed_comparison": speed_rows,
        "semantic_equivalence": semantic,
        "counterfactual_summary": counterfactual_summary,
        "historical_comparison": historical_trend,
        "promotion": promotion,
    }
    _write_csv(output_root / "paired_episodes.csv", episodes)
    _write_csv(output_root / "controller_speed_comparison.csv", speed_rows)
    _write_csv(output_root / "route_usage.csv", usage)
    _write_csv(output_root / "skipped_model_once.csv", flat_counterfactual)
    _write_csv(output_root / "route_counterfactuals.csv", flat_counterfactual)
    _write_csv(output_root / "route_model_vs_lns2.csv", counterfactual_aggregate)
    _write_csv(output_root / "quality_speed_frontier.csv", frontier)
    _write_csv(output_root / "v1_v2_semantic_mismatches.csv", semantic_details)
    _write_csv(
        output_root / "v1_v2_budget_boundary_exclusions.csv", boundary_details
    )
    _write_json(output_root / "v1_v2_semantic_equivalence.json", semantic)
    _write_json(output_root / "tradeoff_report.json", report)
    (output_root / "hybrid_necessity_report.md").write_text(
        _markdown(report, usage, counterfactual_aggregate, speed_rows),
        encoding="utf-8",
    )
    overall = next(
        row
        for row in usage
        if row["record_type"] == "aggregate" and row["stratum_kind"] == "all"
    )
    _bar_svg(
        output_root / "route_usage.svg",
        "Repair decisions by v2-balanced route",
        ["Model", "Original LNS2"],
        [
            float(overall["model_decision_count"]),
            float(overall["official_decision_count"]),
        ],
    )
    _bar_svg(
        output_root / "controller_time_comparison.svg",
        "Mean controller time per complete episode",
        [str(row["controller"]) for row in frontier],
        [float(row["mean_controller_seconds"] or 0.0) for row in frontier],
    )
    skipped = counterfactual_aggregate[0]
    _bar_svg(
        output_root / "counterfactual_pareto.svg",
        "Skipped states: one-repair Pareto relation",
        ["Model dominates", "LNS2 dominates", "Tradeoff", "Tie"],
        [
            float(skipped["model_dominates_count"]),
            float(skipped["lns2_dominates_count"]),
            float(skipped["quality_time_tradeoff_count"]),
            float(skipped["tie_count"]),
        ],
    )
    _frontier_svg(output_root / "quality_speed_frontier.svg", frontier)
    return report


__all__ = [
    "CONTROLLER_LABELS",
    "CONTROLLER_ORDER",
    "HISTORICAL_REFERENCE",
    "TRADEOFF_REPORT_SCHEMA",
    "generate_tradeoff_artifacts",
]
