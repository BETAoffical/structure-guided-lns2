from __future__ import annotations

import statistics
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.balanced_wall_clock import (
    SPLIT,
    _derived_endpoint_seed,
    _derived_endpoints,
    _fingerprint,
    _four_neighbor_distances,
    _largest_four_connected_component,
    _map_metrics,
    _movingai_passable_cells,
    _write_derived_scenario,
    _write_jsonl_atomic,
)
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


REPORT_SCHEMA = "lns2.stride.robuststep_map_preflight_report.v1"
FORBIDDEN_OUTCOME_FIELDS = {
    "v2_relative_ttf",
    "robuststep_relative_ttf",
    "controller_action",
    "controller_repair_outcome",
    "candidate_repair_outcome",
}
PREFLIGHT_ROLES = {
    "stride_robuststep_outcome_blind_load_preflight",
    "stride_robuststep_outcome_blind_load_extension",
    "stride_robuststep_outcome_blind_congestion_preflight",
    "stride_robuststep_topology_balanced_preflight",
}


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _scenario_rank(row: dict[str, Any]) -> int:
    if "scenario_index" in row:
        return int(row["scenario_index"])
    if "task_seed" in row:
        return int(row["task_seed"])
    return int(str(row["scenario_type"]).rsplit("_", 1)[-1])


def analyze_preflight_rows(
    source: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    qualification_rows: list[dict[str, Any]],
    qualification_report: dict[str, Any],
    formal_ood: dict[str, Any],
) -> dict[str, Any]:
    if source.get("role") not in PREFLIGHT_ROLES:
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
                "scenario_index": _scenario_rank(source_row),
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
    topology_gates = dict(source.get("topology_group_gates") or {})
    group_reports: dict[str, dict[str, Any]] = {}
    if topology_gates:
        expected_groups = set(map(str, topology_gates["required_groups"]))
        actual_groups = {
            str(row["layout_family"]) for row in benchmark_index.values()
        }
        qualified_maps = {
            str(row["map_id"])
            for row in map_reports
            if int(row["eligible_task_count"]) > 0
        }
        for group in sorted(actual_groups):
            group_maps = {
                map_id
                for map_id, row in benchmark_index.items()
                if str(row["layout_family"]) == group
            }
            group_reports[group] = {
                "map_count": len(group_maps),
                "qualified_map_count": len(group_maps & qualified_maps),
                "underloaded_maps": sorted(group_maps & set(underloaded)),
            }
        minimum_by_group = {
            str(group): int(value)
            for group, value in dict(
                topology_gates["minimum_qualified_maps_by_group"]
            ).items()
        }
        allowed_underloaded_groups = set(
            map(str, topology_gates["underloaded_allowed_only_in_groups"])
        )
        underloaded_groups = {
            str(benchmark_index[map_id]["layout_family"])
            for map_id in underloaded
        }
        gates.update(
            {
                "topology_group_registry_exact": actual_groups == expected_groups,
                "topology_group_threshold_registry_exact": (
                    set(minimum_by_group) == expected_groups
                    and allowed_underloaded_groups <= expected_groups
                ),
                "minimum_topology_qualified_map_count": len(qualified_maps)
                >= int(topology_gates["minimum_qualified_map_count"]),
                "minimum_topology_qualified_maps_by_group": all(
                    int(group_reports.get(group, {}).get("qualified_map_count", 0))
                    >= minimum
                    for group, minimum in minimum_by_group.items()
                ),
                "underloaded_only_in_registered_control_groups": (
                    underloaded_groups <= allowed_underloaded_groups
                ),
            }
        )
    passed = all(gates.values())
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
        "topology_group_reports": group_reports,
        "recommended_tasks": sorted(recommended, key=lambda row: str(row["task_id"])),
        "underloaded_maps": underloaded,
        "gates": gates,
        "passed": passed,
        "next_decision": (
            "register_proposal_only_candidate_coverage"
            if passed and topology_gates
            else "increase_candidate_loads_for_underloaded_maps"
            if passed and underloaded
            else "register_outcome_blind_pilot_cohort"
            if passed
            else "repair_preflight_integrity_before_selection"
        ),
    }


def _registered_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parents[1] / path).resolve()


def prepare_congestion_preflight_dataset(
    *, fetched: str | Path, source_config: str | Path, output: str | Path,
) -> dict[str, Any]:
    """Build deterministic opposite-exchange OD tasks on pinned MovingAI maps."""

    fetched_root = Path(fetched).resolve()
    config_path = Path(source_config).resolve()
    output_root = Path(output).resolve()
    config = _read_json(config_path)
    if config.get("role") != "stride_robuststep_outcome_blind_congestion_preflight":
        raise ValueError("unexpected congestion preflight role")
    predecessor = dict(config["predecessor_report"])
    predecessor_path = _registered_path(config_path, str(predecessor["path"]))
    if sha256_file(predecessor_path) != str(predecessor["sha256"]):
        raise ValueError("congestion preflight predecessor SHA differs")
    fetched_manifest = fetched_root / "manifest.jsonl"
    if sha256_file(fetched_manifest) != str(config["fetched_manifest_sha256"]):
        raise ValueError("congestion preflight fetched manifest SHA differs")
    source_index = {
        str(row["id"]): row for row in _read_jsonl(fetched_manifest)
    }
    fingerprint = _fingerprint(config)
    summary_path = output_root / "dataset_summary.json"
    if summary_path.is_file():
        existing = _read_json(summary_path)
        if existing.get("configuration_fingerprint") != fingerprint:
            raise ValueError("congestion output belongs to a different config")
        return existing
    if output_root.is_dir() and any(output_root.iterdir()):
        raise ValueError("congestion output is non-empty but has no summary")
    task_seeds = list(map(int, config["task_seeds"]))
    variants = list(map(str, config["task_variants"]))
    if not task_seeds or len(task_seeds) != len(set(task_seeds)):
        raise ValueError("congestion task seeds must be unique")
    if variants != ["opposite_exchange"]:
        raise ValueError("congestion preflight requires opposite_exchange only")
    split_root = output_root / SPLIT
    manifest = []
    observed_maps: set[str] = set()
    for raw_case in config["benchmarks"]:
        case = dict(raw_case)
        map_id = str(case["id"])
        if map_id in observed_maps or map_id not in source_index:
            raise ValueError(f"invalid congestion map registration: {map_id}")
        observed_maps.add(map_id)
        source = source_index[map_id]
        if str(source["map_sha256"]) != str(case["map_sha256"]):
            raise ValueError(f"congestion source map SHA differs: {map_id}")
        source_map = fetched_root / str(source["map_file"])
        if sha256_file(source_map) != str(case["map_sha256"]):
            raise ValueError(f"congestion map file SHA differs: {map_id}")
        map_path = split_root / "maps" / source_map.name
        map_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_map, map_path)
        rows, cols, _grid, passable = _movingai_passable_cells(map_path)
        component = _largest_four_connected_component(passable)
        metrics = _map_metrics(map_path)
        metadata_path = split_root / "maps" / f"{map_id}.json"
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "benchmark_id": map_id,
                "source": "MovingAI map with project-derived congestion OD",
                "map_sha256": sha256_file(map_path),
                "largest_four_connected_component": len(component),
                "topology_metrics": metrics,
            },
        )
        for agent_count in map(int, case["agent_counts"]):
            if agent_count <= 0 or agent_count > len(component):
                raise ValueError(f"invalid congestion agent count: {map_id}")
            for task_seed in task_seeds:
                endpoint_seed = _derived_endpoint_seed(
                    int(config["master_seed"]),
                    map_id,
                    task_seed,
                    "opposite_exchange",
                    agent_count,
                )
                starts, goals = _derived_endpoints(
                    component, agent_count, "opposite_exchange", endpoint_seed
                )
                distances = _four_neighbor_distances(passable, starts, goals)
                task_id = (
                    f"{map_id}__derived_opposite_exchange"
                    f"__task_seed_{task_seed:04d}__agents_{agent_count:04d}"
                )
                scenario_path = split_root / "scenarios" / f"{task_id}.scen"
                _write_derived_scenario(
                    scenario_path,
                    map_path.name,
                    rows,
                    cols,
                    starts,
                    goals,
                    distances,
                )
                task_path = split_root / "tasks" / f"{task_id}.json"
                _write_json(
                    task_path,
                    {
                        "schema_version": 1,
                        "task_semantics": "project-derived opposite-axis exchange on an official MovingAI map",
                        "benchmark_id": map_id,
                        "task_seed": task_seed,
                        "endpoint_seed": endpoint_seed,
                        "agent_count": agent_count,
                        "unique_starts": len(set(starts)) == agent_count,
                        "unique_goals": len(set(goals)) == agent_count,
                        "fixed_point_count": sum(
                            start == goal for start, goal in zip(starts, goals)
                        ),
                        "minimum_shortest_distance": min(distances),
                        "maximum_shortest_distance": max(distances),
                        "mean_shortest_distance": statistics.fmean(distances),
                        "scenario_sha256": sha256_file(scenario_path),
                    },
                )
                manifest.append(
                    {
                        "split": SPLIT,
                        "source_group": "movingai",
                        "instance_origin": "movingai_map_project_derived_congestion_od",
                        "map_id": map_id,
                        "task_id": task_id,
                        "map_file": f"maps/{map_path.name}",
                        "scenario_file": f"scenarios/{scenario_path.name}",
                        "map_metadata_file": f"maps/{metadata_path.name}",
                        "task_file": f"tasks/{task_path.name}",
                        "layout_mode": str(case["layout_family"]),
                        "layout_variant": map_id,
                        "scenario_type": "movingai_map_derived_opposite_exchange",
                        "task_variant": f"opposite_exchange_seed_{task_seed}_agents_{agent_count}",
                        "task_seed": task_seed,
                        "agent_count": agent_count,
                        "topology_metrics": metrics,
                        "dominant_flow_ratio": 1.0,
                        "hotspot_skew": 0.0,
                        "required_bottleneck_crossing_ratio": 0.0,
                        "mean_shortest_distance": statistics.fmean(distances),
                    }
                )
    if (
        len(observed_maps) != int(config["expected_map_count"])
        or len(manifest) != int(config["expected_task_count"])
    ):
        raise ValueError("congestion dataset dimensions differ from registration")
    manifest.sort(key=lambda row: str(row["task_id"]))
    _write_jsonl_atomic(split_root / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "dataset_revision": str(config["dataset_revision"]),
        "configuration_fingerprint": fingerprint,
        "source": "official MovingAI maps with project-derived opposite-exchange OD",
        "task_semantics": "derived_not_official_mapf_scenarios",
        "splits": {
            SPLIT: {
                "map_count": len(observed_maps),
                "instance_count": len(manifest),
                "source_counts": {"movingai": len(manifest)},
            }
        },
    }
    _write_json(summary_path, summary)
    return summary


def analyze_robuststep_map_preflight(
    *, source_config: str | Path, dataset: str | Path,
    qualification: str | Path, output: str | Path,
) -> dict[str, Any]:
    source_path = Path(source_config).resolve()
    dataset_root = Path(dataset).resolve()
    qualification_root = Path(qualification).resolve()
    source = _read_json(source_path)
    predecessor_path = None
    if source.get("predecessor_report"):
        predecessor = dict(source["predecessor_report"])
        predecessor_path = _registered_path(source_path, str(predecessor["path"]))
        if sha256_file(predecessor_path) != str(predecessor["sha256"]):
            raise ValueError("robust-step preflight predecessor SHA differs")
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
    if predecessor_path is not None:
        report["inputs"]["predecessor_report_sha256"] = sha256_file(
            predecessor_path
        )
    output_root = Path(output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "robuststep_map_preflight_report.json", report)
    return report


__all__ = [
    "analyze_preflight_rows",
    "analyze_robuststep_map_preflight",
    "prepare_congestion_preflight_dataset",
]
