from __future__ import annotations

import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


REPORT_SCHEMA = "lns2.stride.robuststep_map_preflight_report.v1"
FORBIDDEN_OUTCOME_FIELDS = {
    "v2_relative_ttf",
    "robuststep_relative_ttf",
    "controller_action",
    "controller_repair_outcome",
}


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def analyze_preflight_rows(
    source: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
    qualification_report: dict[str, Any],
    formal_ood: dict[str, Any],
) -> dict[str, Any]:
    if source.get("role") != "stride_robuststep_outcome_blind_load_preflight":
        raise ValueError("unexpected robust-step preflight source role")
    expected_seeds = set(map(int, source["solver_seeds"]))
    expected_task_ids = {str(row["task_id"]) for row in dataset_rows}
    if len(expected_task_ids) != len(dataset_rows):
        raise ValueError("preflight dataset contains duplicate task ids")
    dataset_by_task = {str(row["task_id"]): row for row in dataset_rows}
    benchmark_index = {
        str(row["id"]): dict(row) for row in source["benchmarks"]
    }
    formal_maps = {str(row["benchmark_id"]) for row in formal_ood.get("cases", [])}
    exceptions = set(map(str, source.get("consumed_development_exceptions", [])))
    overlap = sorted(set(benchmark_index) & formal_maps)
    unapproved_overlap = sorted(set(overlap) - exceptions)

    forbidden_hits = sorted(
        {
            field
            for row in qualification_rows
            for field in FORBIDDEN_OUTCOME_FIELDS
            if field in row
        }
    )
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    row_errors: list[str] = []
    for row in qualification_rows:
        key = (str(row.get("task_id")), int(row.get("solver_seed", -1)))
        if key in indexed:
            row_errors.append(f"duplicate:{key[0]}:{key[1]}")
        indexed[key] = row
        source_row = dataset_by_task.get(key[0])
        if source_row is None or key[1] not in expected_seeds:
            row_errors.append(f"unexpected:{key[0]}:{key[1]}")
            continue
        if (
            str(row.get("status")) != "ok"
            or not bool(row.get("initial_complete"))
            or not bool(row.get("state_fingerprint"))
            or not isinstance(row.get("initial_complexity"), dict)
        ):
            row_errors.append(f"invalid:{key[0]}:{key[1]}")
        if (
            str(row.get("map_id")) != str(source_row["map_id"])
            or int(row.get("agent_count", -1)) != int(source_row["agent_count"])
        ):
            row_errors.append(f"semantic:{key[0]}:{key[1]}")
    expected_jobs = {
        (task_id, seed) for task_id in expected_task_ids for seed in expected_seeds
    }
    missing_jobs = sorted(expected_jobs - set(indexed))

    task_summaries = []
    for task_id in sorted(expected_task_ids):
        source_row = dataset_by_task[task_id]
        rows = [indexed[(task_id, seed)] for seed in sorted(expected_seeds) if (task_id, seed) in indexed]
        conflicts = [float(row["initial_conflicts"]) for row in rows]
        densities = [
            float(row.get("initial_complexity", {}).get("conflict_pair_density", 0.0))
            for row in rows
        ]
        path_costs = [
            float(row.get("initial_complexity", {}).get("mean_path_cost", 0.0))
            for row in rows
        ]
        expansions = [
            float(row.get("initial_complexity", {}).get("initial_low_level_expanded", 0.0))
            for row in rows
        ]
        task_summaries.append(
            {
                "task_id": task_id,
                "map_id": str(source_row["map_id"]),
                "layout_family": str(source_row["layout_mode"]),
                "scenario_index": int(str(source_row["scenario_type"]).rsplit("_", 1)[-1]),
                "agent_count": int(source_row["agent_count"]),
                "seed_count": len(rows),
                "mean_initial_conflicts": _mean(conflicts),
                "min_initial_conflicts": min(conflicts, default=0.0),
                "max_initial_conflicts": max(conflicts, default=0.0),
                "nonzero_seed_rate": _mean([float(value > 0.0) for value in conflicts]),
                "mean_conflict_density": _mean(densities),
                "mean_initial_path_cost": _mean(path_costs),
                "mean_initial_low_level_expanded": _mean(expansions),
            }
        )

    rule = dict(source["selection_rule"])
    minimum = float(rule["minimum_mean_initial_conflicts"])
    targets = list(map(float, rule["target_mean_initial_conflicts"]))
    by_map: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task_summaries:
        by_map[str(row["map_id"])].append(row)
    recommended = []
    map_reports = []
    for map_id in sorted(benchmark_index):
        candidates = by_map.get(map_id, [])
        eligible = [
            row
            for row in candidates
            if row["seed_count"] == len(expected_seeds)
            and float(row["mean_initial_conflicts"]) >= minimum
        ]
        selected: list[dict[str, Any]] = []
        if eligible:
            for target in targets:
                remaining = [row for row in eligible if row not in selected]
                if not remaining or len(selected) >= int(rule["maximum_tasks_per_map"]):
                    break
                chosen = min(
                    remaining,
                    key=lambda row: (
                        abs(float(row["mean_initial_conflicts"]) - target),
                        int(row["agent_count"]),
                        int(row["scenario_index"]),
                        str(row["task_id"]),
                    ),
                )
                selected.append(chosen)
                recommended.append(
                    {**chosen, "selection_target": target, "underloaded_fallback": False}
                )
        elif candidates:
            chosen = max(
                candidates,
                key=lambda row: (
                    float(row["mean_initial_conflicts"]),
                    -int(row["agent_count"]),
                    str(row["task_id"]),
                ),
            )
            selected.append(chosen)
            recommended.append(
                {**chosen, "selection_target": None, "underloaded_fallback": True}
            )
        map_reports.append(
            {
                "map_id": map_id,
                "layout_family": str(benchmark_index[map_id]["layout_family"]),
                "candidate_task_count": len(candidates),
                "eligible_task_count": len(eligible),
                "underloaded": not eligible,
                "recommended_task_ids": [str(row["task_id"]) for row in selected],
            }
        )

    gates = {
        "expected_map_count": len(benchmark_index) == int(source["expected_map_count"]),
        "expected_task_count": len(dataset_rows) == int(source["expected_task_count"]),
        "expected_job_count": len(qualification_rows)
        == int(source["expected_task_count"]) * len(expected_seeds),
        "complete_job_coverage": not missing_jobs,
        "all_rows_valid": not row_errors,
        "qualification_passed": bool(qualification_report.get("passed")),
        "forbidden_outcomes_absent": not forbidden_hits,
        "formal_ood_overlap_authorized": not unapproved_overlap,
        "all_maps_represented": set(by_map) == set(benchmark_index),
    }
    underloaded = [str(row["map_id"]) for row in map_reports if row["underloaded"]]
    return {
        "schema": REPORT_SCHEMA,
        "scientific_status": "development_only_outcome_blind_preflight",
        "controller_outcomes_read": False,
        "formal_speed_claim": False,
        "counts": {
            "map_count": len(benchmark_index),
            "task_count": len(dataset_rows),
            "qualification_job_count": len(qualification_rows),
            "recommended_task_count": len(recommended),
        },
        "formal_ood_overlap": overlap,
        "approved_development_exceptions": sorted(exceptions),
        "unapproved_formal_ood_overlap": unapproved_overlap,
        "forbidden_outcome_fields_found": forbidden_hits,
        "missing_jobs": missing_jobs,
        "row_errors": row_errors,
        "task_summaries": task_summaries,
        "map_reports": map_reports,
        "recommended_tasks": sorted(recommended, key=lambda row: str(row["task_id"])),
        "underloaded_maps": underloaded,
        "gates": gates,
        "passed": all(gates.values()),
        "next_decision": (
            "increase_candidate_loads_for_underloaded_maps"
            if all(gates.values()) and underloaded
            else "register_outcome_blind_pilot_cohort"
            if all(gates.values())
            else "repair_preflight_integrity_before_selection"
        ),
    }


def analyze_robuststep_map_preflight(
    *, source_config: str | Path, dataset: str | Path,
    qualification: str | Path, output: str | Path,
) -> dict[str, Any]:
    source_path = Path(source_config).resolve()
    dataset_root = Path(dataset).resolve()
    qualification_root = Path(qualification).resolve()
    source = _read_json(source_path)
    formal_path = source_path.parents[1] / str(source["formal_ood_config"])
    if sha256_file(formal_path) != str(source["formal_ood_config_sha256"]):
        raise ValueError("formal OOD config SHA differs")
    dataset_manifest = dataset_root / "balanced_wall_clock" / "manifest.jsonl"
    qualification_manifest = qualification_root / "qualification_manifest.jsonl"
    qualification_report_path = qualification_root / "qualification_report.json"
    report = analyze_preflight_rows(
        source,
        _read_jsonl(dataset_manifest),
        _read_jsonl(qualification_manifest),
        _read_json(qualification_report_path),
        _read_json(formal_path),
    )
    report["inputs"] = {
        "source_config_sha256": sha256_file(source_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest),
        "qualification_manifest_sha256": sha256_file(qualification_manifest),
        "qualification_report_sha256": sha256_file(qualification_report_path),
        "formal_ood_config_sha256": sha256_file(formal_path),
    }
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "robuststep_map_preflight_report.json", report)
    return report


__all__ = ["analyze_preflight_rows", "analyze_robuststep_map_preflight"]
