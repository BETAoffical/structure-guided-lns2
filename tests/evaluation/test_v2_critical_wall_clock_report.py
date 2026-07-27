from __future__ import annotations

import csv

from experiments.v2_critical_wall_clock_report import (
    CONTROLLERS,
    ROOM600_TASK,
    generate_v2_critical_wall_clock_report,
)


def test_report_compares_complete_wall_clock_cohorts_and_keeps_diagnostic_label(
    tmp_path,
) -> None:
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
        "candidate_generation_seconds",
        "pp_replan_seconds",
        "repair_iterations",
        "no_improvement_repair_count",
        "normalized_wall_clock_conflict_auc",
        "budget_final_sum_of_costs",
        "critical_seed_mean_count",
        "critical_seed_full_fraction",
    ]
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for seed in (1, 2, 3):
            for controller in CONTROLLERS:
                writer.writerow(
                    {
                        "track": "wall-clock-600",
                        "controller": controller,
                        "task_id": ROOM600_TASK,
                        "map_id": "room-64-64-16",
                        "layout_family": "room",
                        "agent_count": 600,
                        "solver_seed": seed,
                        "success": True,
                        "restricted_time_to_feasible": {
                            "official_adaptive": 12.0,
                            "v2-full": 10.0,
                            "v2-critical": 9.0,
                        }[controller],
                        "neighborhood_selection_seconds": {
                            "official_adaptive": 0.1,
                            "v2-full": 2.0,
                            "v2-critical": 1.5,
                        }[controller],
                        "candidate_generation_seconds": 0.5,
                        "pp_replan_seconds": 5.0,
                        "repair_iterations": 10,
                        "no_improvement_repair_count": 1,
                        "normalized_wall_clock_conflict_auc": 0.01,
                        "budget_final_sum_of_costs": 100,
                        "critical_seed_mean_count": (
                            2.5 if controller == "v2-critical" else ""
                        ),
                        "critical_seed_full_fraction": (
                            0.2 if controller == "v2-critical" else ""
                        ),
                    }
                )
    report = generate_v2_critical_wall_clock_report(source, tmp_path / "out")
    assert report["decision"] == "critical_diagnostic_passed_wall_clock_gates"
    assert report["evidence_level"] == "diagnostic_not_promoted"
    assert report["paired_cohort_count"] == 3
    assert all(report["checks"].values())
    assert (tmp_path / "out" / "critical_vs_v2_pairs.csv").is_file()


def test_report_rejects_incomplete_controller_coverage(tmp_path) -> None:
    source = tmp_path / "timing.csv"
    source.write_text(
        "track,controller,task_id,map_id,layout_family,agent_count,solver_seed,"
        "success,restricted_time_to_feasible,neighborhood_selection_seconds,"
        "candidate_generation_seconds,pp_replan_seconds,repair_iterations,"
        "no_improvement_repair_count,normalized_wall_clock_conflict_auc,"
        "budget_final_sum_of_costs,critical_seed_mean_count,"
        "critical_seed_full_fraction\n"
        "wall-clock-600,v2-full,task,map,room,400,1,True,1,1,1,1,1,0,"
        "0.01,100,,\n",
        encoding="utf-8",
    )
    try:
        generate_v2_critical_wall_clock_report(source, tmp_path / "out")
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete coverage must fail")
