from __future__ import annotations

import csv

from experiments.v2_cost_top3_wall_clock_report import (
    CONTROLLERS,
    generate_v2_cost_top3_wall_clock_report,
)


def test_report_uses_real_wall_clock_and_three_way_common_success(tmp_path) -> None:
    source = tmp_path / "timing.csv"
    fieldnames = [
        "track",
        "controller",
        "task_id",
        "map_id",
        "layout_family",
        "agent_count",
        "solver_seed",
        "success",
        "restricted_time_to_feasible",
        "neighborhood_selection_seconds",
        "pp_replan_seconds",
        "repair_iterations",
        "budget_final_sum_of_costs",
    ]
    times = {
        "official_adaptive": 10.0,
        "v2-full": 8.0,
        "v2-cost-top3-frozen": 7.0,
    }
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for controller in CONTROLLERS:
            writer.writerow(
                {
                    "track": "wall-clock-300",
                    "controller": controller,
                    "task_id": "task",
                    "map_id": "map",
                    "layout_family": "room",
                    "agent_count": 400,
                    "solver_seed": 1,
                    "success": True,
                    "restricted_time_to_feasible": times[controller],
                    "neighborhood_selection_seconds": 1.0,
                    "pp_replan_seconds": 2.0,
                    "repair_iterations": 3,
                    "budget_final_sum_of_costs": 100,
                }
            )
    report = generate_v2_cost_top3_wall_clock_report(source, tmp_path / "out")
    assert report["decision"] == "cost_top3_faster_than_lns2_primary_wall_clock"
    assert report["paired_cohort_count"] == 1
    speedup = next(
        row
        for row in report["map_speedups"]
        if row["controller"] == "v2-cost-top3-frozen"
    )
    assert speedup["speedup_vs_lns2"] == 10.0 / 7.0
