from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv
from experiments.repair_collection import _write_json
from experiments.wall_clock_report_utils import csv_boolean, read_csv_rows


REPORT_SCHEMA = "lns2.v2_critical_wall_clock_report.v1"
CONTROLLERS = ("official_adaptive", "v2-full", "v2-critical")
ROOM600_TASK = "room-64-64-16__random_04__agents_0600"


def _number(value: Any) -> float:
    return float(value or 0.0)


def _mean(values: Iterable[Any]) -> float | None:
    available = [float(value) for value in values if value not in (None, "")]
    return statistics.fmean(available) if available else None


def _ratio(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate) / max(1e-12, float(baseline))


def _episode_row(source: dict[str, str]) -> dict[str, Any]:
    iterations = int(source["repair_iterations"])
    no_improvement = int(source["no_improvement_repair_count"])
    success = csv_boolean(source["success"])
    return {
        "track": str(source["track"]),
        "task_id": str(source["task_id"]),
        "map_id": str(source["map_id"]),
        "layout_family": str(source["layout_family"]),
        "agent_count": int(source["agent_count"]),
        "solver_seed": int(source["solver_seed"]),
        "controller": str(source["controller"]),
        "success": success,
        "capped_time_to_feasible": _number(source["restricted_time_to_feasible"]),
        "neighborhood_selection_seconds": _number(
            source["neighborhood_selection_seconds"]
        ),
        "candidate_generation_seconds": _number(
            source["candidate_generation_seconds"]
        ),
        "pp_replan_seconds": _number(source["pp_replan_seconds"]),
        "repair_iterations": iterations,
        "no_improvement_repair_count": no_improvement,
        "no_improvement_fraction": no_improvement / iterations if iterations else 0.0,
        "normalized_wall_clock_conflict_auc": _number(
            source["normalized_wall_clock_conflict_auc"]
        ),
        "soc_at_feasible": (
            int(float(source["budget_final_sum_of_costs"])) if success else None
        ),
        "critical_seed_mean_count": (
            _number(source.get("critical_seed_mean_count"))
            if source.get("critical_seed_mean_count") not in (None, "")
            else None
        ),
        "critical_seed_full_fraction": (
            _number(source.get("critical_seed_full_fraction"))
            if source.get("critical_seed_full_fraction") not in (None, "")
            else None
        ),
    }


def _summary(rows: list[dict[str, Any]], controller: str) -> dict[str, Any]:
    selected = [row for row in rows if row["controller"] == controller]
    successful = [row for row in selected if row["success"]]
    total_repairs = sum(int(row["repair_iterations"]) for row in selected)
    return {
        "controller": controller,
        "episode_count": len(selected),
        "success_count": len(successful),
        "mean_capped_time_to_feasible": _mean(
            row["capped_time_to_feasible"] for row in selected
        ),
        "total_neighborhood_selection_seconds": sum(
            float(row["neighborhood_selection_seconds"]) for row in selected
        ),
        "mean_neighborhood_selection_seconds": _mean(
            row["neighborhood_selection_seconds"] for row in selected
        ),
        "total_candidate_generation_seconds": sum(
            float(row["candidate_generation_seconds"]) for row in selected
        ),
        "total_pp_replan_seconds": sum(
            float(row["pp_replan_seconds"]) for row in selected
        ),
        "mean_pp_replan_seconds": _mean(row["pp_replan_seconds"] for row in selected),
        "total_repair_iterations": total_repairs,
        "mean_repair_iterations": _mean(row["repair_iterations"] for row in selected),
        "no_improvement_fraction": (
            sum(int(row["no_improvement_repair_count"]) for row in selected)
            / total_repairs
            if total_repairs
            else 0.0
        ),
        "mean_normalized_wall_clock_conflict_auc": _mean(
            row["normalized_wall_clock_conflict_auc"] for row in selected
        ),
        "mean_soc_at_feasible": _mean(row["soc_at_feasible"] for row in successful),
        "mean_critical_seed_count": _mean(
            row["critical_seed_mean_count"] for row in selected
        ),
        "mean_critical_full_seed_fraction": _mean(
            row["critical_seed_full_fraction"] for row in selected
        ),
    }


def generate_v2_critical_wall_clock_report(
    timing_csv: Path, output: Path
) -> dict[str, Any]:
    rows = [
        _episode_row(row)
        for row in read_csv_rows(Path(timing_csv))
        if str(row.get("controller")) in CONTROLLERS
        and str(row.get("track", "")).startswith("wall-clock-")
    ]
    if not rows:
        raise ValueError("critical-seed wall-clock report has no episode rows")
    tracks = {str(row["track"]) for row in rows}
    if len(tracks) != 1:
        raise ValueError("critical-seed diagnostic requires one wall-clock track")

    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["task_id"]), int(row["solver_seed"]))
        controller = str(row["controller"])
        if controller in grouped.setdefault(key, {}):
            raise ValueError("critical-seed report contains a duplicate episode")
        grouped[key][controller] = row
    incomplete = [key for key, value in grouped.items() if set(value) != set(CONTROLLERS)]
    if incomplete:
        raise ValueError("critical-seed report has incomplete three-way coverage")

    pair_rows: list[dict[str, Any]] = []
    for (task_id, seed), indexed in sorted(grouped.items()):
        v2 = indexed["v2-full"]
        critical = indexed["v2-critical"]
        pair_rows.append(
            {
                "task_id": task_id,
                "map_id": v2["map_id"],
                "agent_count": v2["agent_count"],
                "solver_seed": seed,
                "v2_success": v2["success"],
                "critical_success": critical["success"],
                "both_success": v2["success"] and critical["success"],
                "v2_capped_ttf": v2["capped_time_to_feasible"],
                "critical_capped_ttf": critical["capped_time_to_feasible"],
                "critical_speedup_vs_v2": float(v2["capped_time_to_feasible"])
                / max(1e-12, float(critical["capped_time_to_feasible"])),
                "v2_selection_seconds": v2["neighborhood_selection_seconds"],
                "critical_selection_seconds": critical[
                    "neighborhood_selection_seconds"
                ],
                "selection_reduction_fraction": 1.0
                - float(critical["neighborhood_selection_seconds"])
                / max(1e-12, float(v2["neighborhood_selection_seconds"])),
                "v2_pp_seconds": v2["pp_replan_seconds"],
                "critical_pp_seconds": critical["pp_replan_seconds"],
                "v2_repairs": v2["repair_iterations"],
                "critical_repairs": critical["repair_iterations"],
                "v2_normalized_wall_auc": v2[
                    "normalized_wall_clock_conflict_auc"
                ],
                "critical_normalized_wall_auc": critical[
                    "normalized_wall_clock_conflict_auc"
                ],
                "v2_soc_at_feasible": v2["soc_at_feasible"],
                "critical_soc_at_feasible": critical["soc_at_feasible"],
                "v2_no_improvement_fraction": v2["no_improvement_fraction"],
                "critical_no_improvement_fraction": critical[
                    "no_improvement_fraction"
                ],
                "critical_mean_seed_count": critical["critical_seed_mean_count"],
                "critical_full_seed_fraction": critical[
                    "critical_seed_full_fraction"
                ],
            }
        )

    summaries = [_summary(rows, controller) for controller in CONTROLLERS]
    indexed_summaries = {row["controller"]: row for row in summaries}
    common_success_keys = {
        key
        for key, value in grouped.items()
        if value["v2-full"]["success"] and value["v2-critical"]["success"]
    }
    common_ttf = {
        controller: _mean(
            grouped[key][controller]["capped_time_to_feasible"]
            for key in common_success_keys
        )
        for controller in ("v2-full", "v2-critical")
    }

    map_rows: list[dict[str, Any]] = []
    for task_id in sorted({str(row["task_id"]) for row in rows}):
        task = [row for row in rows if row["task_id"] == task_id]
        by_controller = {
            controller: [row for row in task if row["controller"] == controller]
            for controller in CONTROLLERS
        }
        lns2_ttf = _mean(
            row["capped_time_to_feasible"]
            for row in by_controller["official_adaptive"]
        )
        for controller in ("v2-full", "v2-critical"):
            candidate = by_controller[controller]
            candidate_ttf = _mean(row["capped_time_to_feasible"] for row in candidate)
            map_rows.append(
                {
                    "task_id": task_id,
                    "map_id": task[0]["map_id"],
                    "agent_count": task[0]["agent_count"],
                    "controller": controller,
                    "success_count": sum(bool(row["success"]) for row in candidate),
                    "mean_capped_ttf": candidate_ttf,
                    "speedup_vs_lns2": (
                        float(lns2_ttf) / max(1e-12, float(candidate_ttf))
                        if lns2_ttf is not None and candidate_ttf is not None
                        else None
                    ),
                    "mean_selection_seconds": _mean(
                        row["neighborhood_selection_seconds"] for row in candidate
                    ),
                    "mean_pp_seconds": _mean(
                        row["pp_replan_seconds"] for row in candidate
                    ),
                    "mean_repairs": _mean(row["repair_iterations"] for row in candidate),
                    "mean_normalized_wall_auc": _mean(
                        row["normalized_wall_clock_conflict_auc"] for row in candidate
                    ),
                    "mean_soc_at_feasible": _mean(
                        row["soc_at_feasible"] for row in candidate if row["success"]
                    ),
                    "no_improvement_fraction": (
                        sum(int(row["no_improvement_repair_count"]) for row in candidate)
                        / max(1, sum(int(row["repair_iterations"]) for row in candidate))
                    ),
                }
            )

    v2 = indexed_summaries["v2-full"]
    critical = indexed_summaries["v2-critical"]
    lns2 = indexed_summaries["official_adaptive"]
    common_ttf_ratio = _ratio(
        common_ttf["v2-critical"], common_ttf["v2-full"]
    )
    selection_ratio = _ratio(
        critical["total_neighborhood_selection_seconds"],
        v2["total_neighborhood_selection_seconds"],
    )
    auc_ratio = _ratio(
        critical["mean_normalized_wall_clock_conflict_auc"],
        v2["mean_normalized_wall_clock_conflict_auc"],
    )
    soc_ratio = _ratio(critical["mean_soc_at_feasible"], v2["mean_soc_at_feasible"])
    no_progress_ratio = _ratio(
        critical["no_improvement_fraction"], v2["no_improvement_fraction"]
    )
    room600 = [row for row in map_rows if row["task_id"] == ROOM600_TASK]
    room600_indexed = {row["controller"]: row for row in room600}
    checks = {
        "success_not_below_lns2": critical["success_count"] >= lns2["success_count"],
        "success_not_below_v2": critical["success_count"] >= v2["success_count"],
        "common_success_ttf_at_least_5pct_faster_than_v2": common_ttf_ratio
        is not None
        and common_ttf_ratio <= 0.95,
        "selection_time_at_least_20pct_lower_than_v2": selection_ratio is not None
        and selection_ratio <= 0.80,
        "wall_auc_degradation_at_most_2pct": auc_ratio is not None
        and auc_ratio <= 1.02,
        "soc_degradation_at_most_2pct": soc_ratio is not None and soc_ratio <= 1.02,
        "no_progress_degradation_at_most_2pct": no_progress_ratio is not None
        and no_progress_ratio <= 1.02,
        "room600_all_seeds_success": len(room600_indexed) == 2
        and room600_indexed["v2-critical"]["success_count"] == 3,
        "room600_ttf_not_slower_than_v2": len(room600_indexed) == 2
        and float(room600_indexed["v2-critical"]["mean_capped_ttf"])
        <= float(room600_indexed["v2-full"]["mean_capped_ttf"]),
    }
    passed = all(checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "decision": (
            "critical_diagnostic_passed_wall_clock_gates"
            if passed
            else "keep_v2_full_critical_diagnostic_failed"
        ),
        "evidence_level": "diagnostic_not_promoted",
        "track": next(iter(tracks)),
        "episode_count": len(rows),
        "paired_cohort_count": len(grouped),
        "common_success_count": len(common_success_keys),
        "common_success_mean_ttf": common_ttf,
        "ratios_critical_over_v2": {
            "common_success_ttf": common_ttf_ratio,
            "total_selection_time": selection_ratio,
            "mean_normalized_wall_auc": auc_ratio,
            "mean_soc_at_feasible": soc_ratio,
            "no_improvement_fraction": no_progress_ratio,
        },
        "summaries": summaries,
        "map_results": map_rows,
        "checks": checks,
        "validation": {
            "complete_three_way_coverage": True,
            "duplicate_episode_count": 0,
            "controller_set": list(CONTROLLERS),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "critical_wall_clock_episodes.csv", rows)
    atomic_write_csv(output / "critical_vs_v2_pairs.csv", pair_rows)
    atomic_write_csv(output / "critical_wall_clock_summary.csv", summaries)
    atomic_write_csv(output / "critical_map_results.csv", map_rows)
    _write_json(output / "critical_wall_clock_report.json", report)

    lines = [
        "# v2 critical-seed wall-clock diagnostic",
        "",
        f"Decision: `{report['decision']}`",
        "",
        "> Diagnostic evidence only. The critical-seed configuration remains unpromoted.",
        "",
        "| Controller | Success | Mean capped TTF | Selection total | PP total | Repairs | Wall AUC | SOC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for controller in CONTROLLERS:
        row = indexed_summaries[controller]
        lines.append(
            f"| {controller} | {row['success_count']}/{row['episode_count']} | "
            f"{float(row['mean_capped_time_to_feasible']):.6f} | "
            f"{float(row['total_neighborhood_selection_seconds']):.6f} | "
            f"{float(row['total_pp_replan_seconds']):.6f} | "
            f"{int(row['total_repair_iterations'])} | "
            f"{float(row['mean_normalized_wall_clock_conflict_auc']):.6f} | "
            + (
                f"{float(row['mean_soc_at_feasible']):.3f} |"
                if row["mean_soc_at_feasible"] is not None
                else "n/a |"
            )
        )
    lines.extend(
        [
            "",
            "## Gate checks",
            "",
            *[f"- {name}: `{value}`" for name, value in checks.items()],
            "",
            "## Per-map capped TTF",
            "",
            "| Task | Agents | Controller | Success | Mean capped TTF | Speedup vs LNS2 |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in map_rows:
        lines.append(
            f"| {row['task_id']} | {row['agent_count']} | {row['controller']} | "
            f"{row['success_count']}/3 | {float(row['mean_capped_ttf']):.6f} | "
            f"{float(row['speedup_vs_lns2']):.4f} |"
        )
    (output / "critical_wall_clock_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return report


__all__ = ["generate_v2_critical_wall_clock_report"]
