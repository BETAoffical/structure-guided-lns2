from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from experiments._common import atomic_write_csv
from experiments.repair_collection import _write_json
from experiments.wall_clock_report_utils import csv_boolean, read_csv_rows


REPORT_SCHEMA = "lns2.v2_cost_top3_wall_clock_report.v1"
CONTROLLERS = (
    "official_adaptive",
    "v2-full",
    "v2-cost-top3-frozen",
)


def _mean(rows: list[dict[str, Any]], name: str) -> float | None:
    values = [float(row[name]) for row in rows if str(row.get(name, "")) != ""]
    return statistics.fmean(values) if values else None


def generate_v2_cost_top3_wall_clock_report(
    timing_csv: Path, output: Path
) -> dict[str, Any]:
    source_rows = read_csv_rows(Path(timing_csv))
    rows = [
        row
        for row in source_rows
        if str(row.get("controller")) in CONTROLLERS
        and str(row.get("track", "")).startswith("wall-clock-")
    ]
    if not rows:
        raise ValueError("cost Top-3 wall-clock report has no episode rows")
    grouped: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["track"]),
            str(row["task_id"]),
            int(row["solver_seed"]),
        )
        controller = str(row["controller"])
        if controller in grouped.setdefault(key, {}):
            raise ValueError("cost Top-3 report contains a duplicate episode")
        grouped[key][controller] = row
    incomplete = [key for key, value in grouped.items() if set(value) != set(CONTROLLERS)]
    if incomplete:
        raise ValueError("cost Top-3 report has incomplete controller coverage")

    episode_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        track, task_id, seed = key
        indexed = grouped[key]
        all_success = all(csv_boolean(indexed[name]["success"]) for name in CONTROLLERS)
        for controller in CONTROLLERS:
            row = indexed[controller]
            episode_rows.append(
                {
                    "track": track,
                    "task_id": task_id,
                    "map_id": str(row["map_id"]),
                    "layout_family": str(row["layout_family"]),
                    "agent_count": int(row["agent_count"]),
                    "solver_seed": seed,
                    "controller": controller,
                    "success": csv_boolean(row["success"]),
                    "three_way_common_success": all_success,
                    "capped_time_to_feasible": float(
                        row["restricted_time_to_feasible"]
                    ),
                    "neighborhood_selection_seconds": float(
                        row["neighborhood_selection_seconds"]
                    ),
                    "pp_replan_seconds": float(row["pp_replan_seconds"]),
                    "repair_iterations": int(row["repair_iterations"]),
                    "soc_at_feasible": (
                        int(float(row["budget_final_sum_of_costs"]))
                        if csv_boolean(row["success"])
                        else None
                    ),
                }
            )

    summaries: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []
    for track in sorted({str(row["track"]) for row in episode_rows}):
        track_rows = [row for row in episode_rows if row["track"] == track]
        common_keys = {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in track_rows
            if bool(row["three_way_common_success"])
        }
        for controller in CONTROLLERS:
            selected = [row for row in track_rows if row["controller"] == controller]
            common = [
                row
                for row in selected
                if (str(row["task_id"]), int(row["solver_seed"])) in common_keys
            ]
            successful = [row for row in selected if bool(row["success"])]
            summaries.append(
                {
                    "track": track,
                    "controller": controller,
                    "episode_count": len(selected),
                    "success_count": len(successful),
                    "mean_capped_time_to_feasible": _mean(
                        selected, "capped_time_to_feasible"
                    ),
                    "common_success_count": len(common),
                    "mean_common_success_time_to_feasible": _mean(
                        common, "capped_time_to_feasible"
                    ),
                    "mean_neighborhood_selection_seconds": _mean(
                        selected, "neighborhood_selection_seconds"
                    ),
                    "mean_pp_replan_seconds": _mean(
                        selected, "pp_replan_seconds"
                    ),
                    "mean_repair_iterations": _mean(
                        selected, "repair_iterations"
                    ),
                    "mean_soc_at_feasible": _mean(successful, "soc_at_feasible"),
                }
            )
        for task_id in sorted({str(row["task_id"]) for row in track_rows}):
            task = [row for row in track_rows if row["task_id"] == task_id]
            lns2 = [row for row in task if row["controller"] == "official_adaptive"]
            lns2_mean = _mean(lns2, "capped_time_to_feasible")
            assert lns2_mean is not None
            for controller in ("v2-full", "v2-cost-top3-frozen"):
                candidate = [row for row in task if row["controller"] == controller]
                candidate_mean = _mean(candidate, "capped_time_to_feasible")
                assert candidate_mean is not None
                map_rows.append(
                    {
                        "track": track,
                        "task_id": task_id,
                        "map_id": str(task[0]["map_id"]),
                        "agent_count": int(task[0]["agent_count"]),
                        "controller": controller,
                        "paired_seed_count": len(candidate),
                        "lns2_success_count": sum(
                            int(bool(row["success"])) for row in lns2
                        ),
                        "candidate_success_count": sum(
                            int(bool(row["success"])) for row in candidate
                        ),
                        "lns2_mean_capped_ttf": lns2_mean,
                        "candidate_mean_capped_ttf": candidate_mean,
                        "speedup_vs_lns2": lns2_mean
                        / max(1e-12, candidate_mean),
                    }
                )

    primary_track = min(
        {str(row["track"]) for row in episode_rows},
        key=lambda value: float(value.rsplit("-", 1)[-1]),
    )
    primary = {
        row["controller"]: row
        for row in summaries
        if row["track"] == primary_track
    }
    cost = primary["v2-cost-top3-frozen"]
    lns2 = primary["official_adaptive"]
    decision = (
        "cost_top3_faster_than_lns2_primary_wall_clock"
        if int(cost["success_count"]) >= int(lns2["success_count"])
        and float(cost["mean_capped_time_to_feasible"])
        < float(lns2["mean_capped_time_to_feasible"])
        else "stop_cost_top3_not_faster_than_lns2"
    )
    report = {
        "schema": REPORT_SCHEMA,
        "decision": decision,
        "primary_track": primary_track,
        "episode_count": len(episode_rows),
        "paired_cohort_count": len(grouped),
        "summaries": summaries,
        "map_speedups": map_rows,
        "validation": {
            "complete_three_way_coverage": True,
            "duplicate_episode_count": 0,
            "controller_set": list(CONTROLLERS),
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "cost_top3_wall_clock_episodes.csv", episode_rows)
    atomic_write_csv(output / "cost_top3_wall_clock_summary.csv", summaries)
    atomic_write_csv(output / "cost_top3_map_speedups.csv", map_rows)
    _write_json(output / "cost_top3_wall_clock_report.json", report)
    lines = [
        "# Frozen v2 cost-aware Top-3 wall-clock report",
        "",
        f"Decision: `{decision}`",
        "",
        f"Primary track: `{primary_track}`",
        "",
        "| Controller | Success | Mean capped TTF | Common-success TTF | Selection | PP | Repairs | SOC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for controller in CONTROLLERS:
        row = primary[controller]
        lines.append(
            "| "
            + " | ".join(
                (
                    controller,
                    f"{row['success_count']}/{row['episode_count']}",
                    f"{float(row['mean_capped_time_to_feasible']):.6f}",
                    f"{float(row['mean_common_success_time_to_feasible']):.6f}"
                    if row["mean_common_success_time_to_feasible"] is not None
                    else "n/a",
                    f"{float(row['mean_neighborhood_selection_seconds']):.6f}",
                    f"{float(row['mean_pp_replan_seconds']):.6f}",
                    f"{float(row['mean_repair_iterations']):.3f}",
                    f"{float(row['mean_soc_at_feasible']):.3f}"
                    if row["mean_soc_at_feasible"] is not None
                    else "n/a",
                )
            )
            + " |"
        )
    (output / "cost_top3_wall_clock_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    return report


__all__ = ["generate_v2_cost_top3_wall_clock_report"]
