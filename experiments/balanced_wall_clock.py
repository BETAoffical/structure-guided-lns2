from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import random
import shutil
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


SPLIT = "balanced_wall_clock"
CONTROLLERS = ("official_adaptive", "v2-full", "mixed-full-v2")
STRATA = (("low", 1, 10), ("medium", 11, 100), ("high", 101, 500))


def _fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    partial.replace(path)


def _map_metrics(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    headers: dict[str, str] = {}
    marker = None
    for index, line in enumerate(lines):
        if line.strip().lower() == "map":
            marker = index
            break
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            headers[parts[0].lower()] = parts[1]
    if marker is None or "height" not in headers or "width" not in headers:
        raise ValueError(f"invalid MovingAI map: {path}")
    rows, cols = int(headers["height"]), int(headers["width"])
    grid = lines[marker + 1 : marker + 1 + rows]
    if len(grid) != rows or any(len(row) != cols for row in grid):
        raise ValueError(f"MovingAI map dimensions differ from header: {path}")
    free = {
        row * cols + col
        for row, values in enumerate(grid)
        for col, value in enumerate(values)
        if value in {".", "G", "S"}
    }
    if not free:
        raise ValueError(f"MovingAI map has no free cells: {path}")

    def degree(cell: int) -> int:
        row, col = divmod(cell, cols)
        neighbors = []
        if row:
            neighbors.append((row - 1) * cols + col)
        if row + 1 < rows:
            neighbors.append((row + 1) * cols + col)
        if col:
            neighbors.append(row * cols + col - 1)
        if col + 1 < cols:
            neighbors.append(row * cols + col + 1)
        return sum(value in free for value in neighbors)

    degrees = [degree(cell) for cell in free]
    return {
        "rows": rows,
        "cols": cols,
        "free_cell_count": len(free),
        "obstacle_count": rows * cols - len(free),
        "obstacle_ratio": (rows * cols - len(free)) / (rows * cols),
        "average_free_degree": statistics.fmean(degrees),
        "minimum_free_degree": min(degrees),
        "maximum_free_degree": max(degrees),
        "dead_end_cell_count": sum(value <= 1 for value in degrees),
        "low_degree_cell_ratio": sum(value <= 2 for value in degrees) / len(degrees),
    }


def _scenario_metrics(path: Path, agent_count: int) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = lines[1:] if lines and lines[0].lower().startswith("version") else lines
    if len(rows) < agent_count:
        raise ValueError(f"scenario has fewer than {agent_count} agents: {path}")
    selected = [line.split() for line in rows[:agent_count]]
    if any(len(row) < 9 for row in selected):
        raise ValueError(f"invalid MovingAI scenario row: {path}")
    starts = [(int(row[4]), int(row[5])) for row in selected]
    goals = [(int(row[6]), int(row[7])) for row in selected]
    if len(starts) != len(set(starts)) or len(goals) != len(set(goals)):
        raise ValueError(f"scenario prefix repeats a start or goal: {path}")
    distances = [float(row[8]) for row in selected]
    return {
        "agent_count": agent_count,
        "mean_shortest_distance": statistics.fmean(distances),
        "minimum_shortest_distance": min(distances),
        "maximum_shortest_distance": max(distances),
    }


def prepare_movingai_dataset(
    fetched: str | Path, source_config: str | Path, output: str | Path
) -> dict[str, Any]:
    fetched_root = Path(fetched).resolve()
    config_path = Path(source_config).resolve()
    output_root = Path(output).resolve()
    config = _read_json(config_path)
    source_rows = _read_jsonl(fetched_root / "manifest.jsonl")
    source_index = {str(row["id"]): row for row in source_rows}
    case_index = {str(row["id"]): dict(row) for row in config["benchmarks"]}
    if set(source_index) != set(case_index):
        raise ValueError("fetched MovingAI manifest differs from registration")
    split_root = output_root / SPLIT
    manifest = []
    for map_id in sorted(source_index):
        source = source_index[map_id]
        case = case_index[map_id]
        source_map = fetched_root / str(source["map_file"])
        if sha256_file(source_map) != str(source["map_sha256"]):
            raise ValueError(f"MovingAI map SHA mismatch: {map_id}")
        map_path = split_root / "maps" / source_map.name
        map_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_map, map_path)
        metrics = _map_metrics(map_path)
        metadata_path = split_root / "maps" / f"{map_id}.json"
        _write_json(
            metadata_path,
            {
                "schema_version": 1,
                "benchmark_id": map_id,
                "source": "MovingAI MAPF benchmark",
                "map_sha256": sha256_file(map_path),
                "topology_metrics": metrics,
            },
        )
        scenarios = {int(row["index"]): dict(row) for row in source["scenarios"]}
        for scenario_index in map(int, config["scenario_indices"]):
            scenario = scenarios[scenario_index]
            source_scenario = fetched_root / str(scenario["file"])
            if sha256_file(source_scenario) != str(scenario["sha256"]):
                raise ValueError(f"MovingAI scenario SHA mismatch: {map_id}/{scenario_index}")
            scenario_path = split_root / "scenarios" / source_scenario.name
            scenario_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_scenario, scenario_path)
            for agent_count in map(int, case["agent_counts"]):
                task_id = f"{map_id}__random_{scenario_index:02d}__agents_{agent_count:04d}"
                task_metrics = _scenario_metrics(scenario_path, agent_count)
                task_path = split_root / "tasks" / f"{task_id}.json"
                _write_json(
                    task_path,
                    {
                        "schema_version": 1,
                        "task_semantics": f"static MovingAI random-{scenario_index} prefix",
                        "benchmark_id": map_id,
                        "scenario_index": scenario_index,
                        "scenario_sha256": sha256_file(scenario_path),
                        **task_metrics,
                    },
                )
                manifest.append(
                    {
                        "split": SPLIT,
                        "source_group": "movingai",
                        "map_id": map_id,
                        "task_id": task_id,
                        "map_file": f"maps/{map_path.name}",
                        "scenario_file": f"scenarios/{scenario_path.name}",
                        "map_metadata_file": f"maps/{map_id}.json",
                        "task_file": f"tasks/{task_id}.json",
                        "layout_mode": str(case["layout_family"]),
                        "layout_variant": map_id,
                        "scenario_type": f"movingai_random_{scenario_index}",
                        "task_variant": f"random_{scenario_index}_agents_{agent_count}",
                        "agent_count": agent_count,
                        "topology_metrics": metrics,
                        "dominant_flow_ratio": 0.0,
                        "hotspot_skew": 0.0,
                        "required_bottleneck_crossing_ratio": 0.0,
                        "mean_shortest_distance": task_metrics["mean_shortest_distance"],
                    }
                )
    manifest.sort(key=lambda row: str(row["task_id"]))
    _write_jsonl_atomic(split_root / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "configuration_fingerprint": _fingerprint(
            {
                "source_config_sha256": sha256_file(config_path),
                "fetched_manifest_sha256": sha256_file(fetched_root / "manifest.jsonl"),
            }
        ),
        "source": "MovingAI MAPF benchmark random scenarios",
        "splits": {SPLIT: {"map_count": 6, "instance_count": len(manifest)}},
    }
    _write_json(output_root / "dataset_summary.json", summary)
    return summary


def merge_datasets(
    generated: str | Path, movingai: str | Path, output: str | Path
) -> dict[str, Any]:
    output_root = Path(output).resolve()
    manifest = []
    for source_group, source_root_value in (
        ("generated", generated),
        ("movingai", movingai),
    ):
        source_root = Path(source_root_value).resolve()
        split_root = source_root / SPLIT
        for raw in _read_jsonl(split_root / "manifest.jsonl"):
            row = dict(raw)
            row["source_group"] = source_group
            for field in (
                "map_file",
                "scenario_file",
                "map_metadata_file",
                "task_file",
                "legacy_instance_file",
            ):
                if not row.get(field):
                    continue
                relative = Path(str(row[field]))
                source = split_root / relative
                if not source.is_file():
                    raise ValueError(f"dataset file is missing: {source}")
                destination = output_root / SPLIT / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.is_file() and sha256_file(destination) != sha256_file(source):
                    raise ValueError(f"dataset merge collision: {relative}")
                if not destination.is_file():
                    shutil.copy2(source, destination)
            manifest.append(row)
    if len(manifest) != 72 or len({str(row["task_id"]) for row in manifest}) != 72:
        raise ValueError("balanced wall-clock dataset must contain 72 unique tasks")
    manifest.sort(key=lambda row: str(row["task_id"]))
    _write_jsonl_atomic(output_root / SPLIT / "manifest.jsonl", manifest)
    summary = {
        "schema_version": 1,
        "configuration_fingerprint": _fingerprint(manifest),
        "source": "pinned MovingAI plus generated structured maps",
        "task_semantics": "static OD/scenario tasks",
        "splits": {
            SPLIT: {
                "map_count": len({str(row["map_id"]) for row in manifest}),
                "instance_count": len(manifest),
                "source_counts": dict(
                    sorted(collections.Counter(str(row["source_group"]) for row in manifest).items())
                ),
            }
        },
    }
    _write_json(output_root / "dataset_summary.json", summary)
    return summary


def conflict_stratum(conflicts: int) -> str | None:
    for name, lower, upper in STRATA:
        if lower <= conflicts <= upper:
            return name
    return None


def _agent_band(count: int) -> str:
    return "small" if count <= 200 else "medium" if count <= 400 else "large"


def select_balanced_cohort(
    dataset: str | Path, qualification: str | Path, output: str | Path
) -> dict[str, Any]:
    dataset_root = Path(dataset).resolve()
    qualification_root = Path(qualification).resolve()
    output_root = Path(output).resolve()
    rows = _read_jsonl(dataset_root / SPLIT / "manifest.jsonl")
    tasks = {str(row["task_id"]): row for row in rows}
    qualified = _read_jsonl(qualification_root / "qualification_manifest.jsonl")
    if len(qualified) != 216:
        raise ValueError(f"qualification must contain 216 results, found {len(qualified)}")
    candidates = []
    for result in qualified:
        if str(result.get("status")) != "ok" or not bool(result.get("initial_complete")):
            raise ValueError("qualification contains an invalid reset")
        task = tasks[str(result["task_id"])]
        conflicts = int(result["initial_conflicts"])
        candidates.append(
            {
                "task_id": str(result["task_id"]),
                "solver_seed": int(result["solver_seed"]),
                "map_id": str(result["map_id"]),
                "layout_mode": str(result["layout_mode"]),
                "source_group": str(task["source_group"]),
                "agent_count": int(result["agent_count"]),
                "agent_band": _agent_band(int(result["agent_count"])),
                "initial_conflicts": conflicts,
                "conflict_stratum": conflict_stratum(conflicts),
                "state_fingerprint": str(result["state_fingerprint"]),
            }
        )

    selected = []
    stratum_reports: dict[str, dict[str, Any]] = {}
    selection_passed = True
    for stratum, _lower, _upper in STRATA:
        eligible = [row for row in candidates if row["conflict_stratum"] == stratum]
        eligible.sort(
            key=lambda row: _fingerprint(
                ["balanced-wall-clock-cohort-v1", row["task_id"], row["solver_seed"]]
            )
        )
        chosen: list[dict[str, Any]] = []
        map_counts: collections.Counter[str] = collections.Counter()

        def take(predicate: Any) -> bool:
            for row in eligible:
                if row in chosen or map_counts[row["map_id"]] >= 2 or not predicate(row):
                    continue
                chosen.append(row)
                map_counts[row["map_id"]] += 1
                return True
            return False

        for source in ("generated", "movingai"):
            while sum(row["source_group"] == source for row in chosen) < 4:
                if not take(lambda row, source=source: row["source_group"] == source):
                    break
        while len({row["agent_band"] for row in chosen}) < 2:
            existing = {row["agent_band"] for row in chosen}
            if not take(lambda row, existing=existing: row["agent_band"] not in existing):
                break
        while len(chosen) < 12 and take(lambda _row: True):
            pass
        checks = {
            "count": len(chosen) == 12,
            "generated_minimum": sum(row["source_group"] == "generated" for row in chosen) >= 4,
            "movingai_minimum": sum(row["source_group"] == "movingai" for row in chosen) >= 4,
            "agent_band_count": len({row["agent_band"] for row in chosen}) >= 2,
            "map_cap": max(map_counts.values(), default=0) <= 2,
        }
        stratum_reports[stratum] = {
            "eligible": len(eligible),
            "eligible_by_source": dict(
                sorted(collections.Counter(row["source_group"] for row in eligible).items())
            ),
            "eligible_map_count": len({row["map_id"] for row in eligible}),
            "provisional_selected": len(chosen),
            "checks": checks,
            "passed": all(checks.values()),
        }
        selection_passed &= all(checks.values())
        selected.extend(chosen)

    if selection_passed:
        selected.sort(
            key=lambda row: (
                row["conflict_stratum"],
                row["map_id"],
                row["task_id"],
                row["solver_seed"],
            )
        )
        permutations = list(itertools.permutations(CONTROLLERS))
        schedule = []
        for index, row in enumerate(selected):
            schedule.append(
                {
                    **row,
                    "schedule_group": index % len(permutations),
                    "controller_order": list(permutations[index % len(permutations)]),
                }
            )
        _write_jsonl_atomic(output_root / "cohort.jsonl", selected)
        _write_json(
            output_root / "execution_schedule.json",
            {
                "schema": "lns2.controller_execution_schedule.v1",
                "selection_blind_to_controller_outcomes": True,
                "entries": schedule,
            },
        )
    report = {
        "schema": "lns2.balanced_wall_clock_cohort.v1",
        "qualification_count": len(qualified),
        "selected_count": len(selected),
        "selection_blind_to_controller_outcomes": True,
        "strata": stratum_reports,
        "excluded": {
            "zero_conflict": sum(row["initial_conflicts"] == 0 for row in candidates),
            "over_500": sum(row["initial_conflicts"] > 500 for row in candidates),
        },
        "passed": selection_passed and len(selected) == 36,
        "decision": "eligible_for_formal" if selection_passed else "data_gate_failed",
        "formal_collection_allowed": selection_passed,
    }
    _write_json(output_root / "cohort_report.json", report)
    return report


def collect_scheduled(
    *,
    dataset: str | Path,
    config: str | Path,
    qualification: str | Path,
    schedule_root: str | Path,
    output: str | Path,
    original_bundle: str | Path,
    mixed_bundle: str | Path,
    resume: bool,
    dry_run: bool = False,
) -> dict[str, Any]:
    schedule_root_path = Path(schedule_root).resolve()
    cohort_report = _read_json(schedule_root_path / "cohort_report.json")
    if not bool(cohort_report.get("formal_collection_allowed", False)):
        raise ValueError("balanced wall-clock cohort did not pass the preregistered data gate")
    schedule_path = schedule_root_path / "execution_schedule.json"
    schedule = _read_json(schedule_path)
    entries = list(schedule["entries"])
    output_root = Path(output).resolve()
    progress = []
    bundles = {
        "official_adaptive": original_bundle,
        "v2-full": original_bundle,
        "mixed-full-v2": mixed_bundle,
    }
    phases = {
        "official_adaptive": "official_adaptive",
        "v2-full": "realized_dynamic",
        "mixed-full-v2": "realized_dynamic",
    }
    for group in range(6):
        group_rows = [row for row in entries if int(row["schedule_group"]) == group]
        if not group_rows:
            raise ValueError(f"execution schedule group is empty: {group}")
        order = tuple(map(str, group_rows[0]["controller_order"]))
        if any(tuple(map(str, row["controller_order"])) != order for row in group_rows):
            raise ValueError("execution schedule group has inconsistent controller order")
        keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in group_rows}
        task_ids = sorted({task_id for task_id, _seed in keys})
        for controller_id in order:
            lane = output_root / f"order_{group}" / controller_id
            common = dict(
                dataset=dataset,
                config_path=config,
                output=lane,
                workers=1,
                task_ids=task_ids,
                controller="v2-full",
                feature_backend="auto",
                controller_bundle=bundles[controller_id],
                controller_runtime="optimized",
                verification_profile="deployment",
                job_keys=keys,
                cohort_job_keys=keys,
                stopping_rule="historical",
                use_global_collection_lock=False,
            )
            if dry_run:
                result = run_closed_loop_collection(
                    **common, phase=phases[controller_id], dry_run=True
                )
            else:
                if not (lane / "qualification_manifest.jsonl").is_file():
                    run_closed_loop_collection(
                        **common,
                        phase="qualify",
                        qualification_source=qualification,
                        resume=False,
                    )
                result = run_closed_loop_collection(
                    **common,
                    phase=phases[controller_id],
                    resume=(lane / "run_config.json").is_file() or resume,
                )
            progress.append(
                {
                    "group": group,
                    "controller": controller_id,
                    "job_count": len(keys),
                    "dry_run": dry_run,
                    "summary": result,
                }
            )
            _write_json(output_root / "collection_progress.json", {"entries": progress})
    return {"group_count": 6, "lane_count": len(progress), "entries": progress}


def _mean(values: Iterable[float]) -> float:
    rows = list(map(float, values))
    return statistics.fmean(rows) if rows else 0.0


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _relative_improvement(baseline: float, candidate: float) -> float:
    if abs(baseline) <= 1e-15:
        return 0.0 if abs(candidate) <= 1e-15 else -math.inf
    return 1.0 - candidate / baseline


def _episode_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["task_id"]), int(row["solver_seed"])


def _scientific_summary(row: dict[str, Any]) -> dict[str, Any]:
    summary = dict(row["summary"])
    totals = dict(summary.get("controller_totals", {}))
    return {
        "success": bool(summary["success"]),
        "capped_wall_ttf": float(summary["capped_wall_time_to_feasible"]),
        "fixed_auc": float(summary["fixed_budget_conflict_auc"]),
        "normalized_fixed_auc": float(summary["normalized_fixed_budget_conflict_auc"]),
        "repair_iterations": int(summary["repair_iterations"]),
        "generated_nodes": int(summary["final_low_level"]["generated"]),
        "expanded_nodes": int(summary["final_low_level"]["expanded"]),
        "repair_wall_seconds": float(summary.get("repair_wall_seconds", 0.0)),
        "environment_construct_seconds": float(
            summary.get("environment_construct_seconds", 0.0)
        ),
        "reset_wall_seconds": float(summary.get("reset_wall_seconds", 0.0)),
        "episode_observed_wall_seconds": float(
            summary.get("episode_observed_wall_seconds", 0.0)
        ),
        "proposal_seconds": float(totals.get("proposal_seconds", 0.0)),
        "feature_seconds": float(totals.get("feature_seconds", 0.0)),
        "inference_seconds": float(totals.get("inference_seconds", 0.0)),
        "fingerprint_seconds": float(totals.get("state_fingerprint_seconds", 0.0)),
        "pp_replan_seconds": float(totals.get("pp_replan_seconds", 0.0)),
        "controller_before_repair_seconds": float(
            totals.get("controller_seconds_before_repair", 0.0)
        ),
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_scientific_summary(row) for row in rows]
    numeric_fields = (
        "capped_wall_ttf",
        "fixed_auc",
        "normalized_fixed_auc",
        "repair_iterations",
        "generated_nodes",
        "expanded_nodes",
        "repair_wall_seconds",
        "environment_construct_seconds",
        "reset_wall_seconds",
        "episode_observed_wall_seconds",
        "proposal_seconds",
        "feature_seconds",
        "inference_seconds",
        "fingerprint_seconds",
        "pp_replan_seconds",
        "controller_before_repair_seconds",
    )
    return {
        "episode_count": len(rows),
        "success_count": sum(value["success"] for value in values),
        **{f"mean_{field}": _mean(value[field] for value in values) for field in numeric_fields},
        "invalid_action_count": sum(
            int(row["summary"].get("invalid_action_count", 0)) for row in rows
        ),
        "fingerprint_mismatch_count": sum(
            int(row["summary"].get("fingerprint_mismatch_count", 0)) for row in rows
        ),
        "error_count": sum(str(row.get("status")) not in {"ok", "resumed"} for row in rows),
    }


def _paired_bootstrap_by_map(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    *,
    metric: str,
    samples: int = 5000,
    seed: int = 20270831,
) -> dict[str, Any]:
    by_map: dict[str, list[tuple[float, float]]] = collections.defaultdict(list)
    for key in sorted(baseline):
        left, right = baseline[key], candidate[key]
        if str(left["map_id"]) != str(right["map_id"]):
            raise ValueError("paired controller rows disagree on map id")
        by_map[str(left["map_id"])].append(
            (
                float(_scientific_summary(left)[metric]),
                float(_scientific_summary(right)[metric]),
            )
        )
    map_ids = sorted(by_map)
    generator = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [generator.choice(map_ids) for _ in map_ids]
        left_values = [value[0] for map_id in selected for value in by_map[map_id]]
        right_values = [value[1] for map_id in selected for value in by_map[map_id]]
        estimates.append(_relative_improvement(_mean(left_values), _mean(right_values)))
    left_mean = _mean(value[0] for values in by_map.values() for value in values)
    right_mean = _mean(value[1] for values in by_map.values() for value in values)
    return {
        "map_count": len(map_ids),
        "samples": samples,
        "seed": seed,
        "baseline_mean": left_mean,
        "candidate_mean": right_mean,
        "relative_improvement": _relative_improvement(left_mean, right_mean),
        "improvement_95_ci": [_quantile(estimates, 0.025), _quantile(estimates, 0.975)],
    }


def _paired_group_comparison(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    schedule_index: dict[tuple[str, int], dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = collections.defaultdict(list)
    for key in sorted(baseline):
        groups[str(schedule_index[key][field])].append((baseline[key], candidate[key]))
    result = {}
    for name, pairs in sorted(groups.items()):
        left_values = [_scientific_summary(left) for left, _right in pairs]
        right_values = [_scientific_summary(right) for _left, right in pairs]
        left_ttf = _mean(row["capped_wall_ttf"] for row in left_values)
        right_ttf = _mean(row["capped_wall_ttf"] for row in right_values)
        left_auc = _mean(row["fixed_auc"] for row in left_values)
        right_auc = _mean(row["fixed_auc"] for row in right_values)
        result[name] = {
            "episode_count": len(pairs),
            "baseline_success_count": sum(row["success"] for row in left_values),
            "candidate_success_count": sum(row["success"] for row in right_values),
            "ttf_improvement": _relative_improvement(left_ttf, right_ttf),
            "auc_improvement": _relative_improvement(left_auc, right_auc),
            "candidate_not_worse": (
                sum(row["success"] for row in right_values)
                >= sum(row["success"] for row in left_values)
                and right_ttf <= left_ttf
            ),
        }
    return result


def analyze_scheduled(
    collection: str | Path, schedule_root: str | Path, output: str | Path
) -> dict[str, Any]:
    collection_root = Path(collection).resolve()
    schedule = _read_json(Path(schedule_root).resolve() / "execution_schedule.json")
    schedule_index = {_episode_key(row): dict(row) for row in schedule["entries"]}
    expected = set(schedule_index)
    rows = []
    for group in range(6):
        for controller in CONTROLLERS:
            phase = "official_adaptive" if controller == "official_adaptive" else "realized_dynamic"
            path = collection_root / f"order_{group}" / controller / f"{phase}_manifest.jsonl"
            for row in _read_jsonl(path):
                rows.append({**row, "controller_id": controller})
    by_controller: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_controller[str(row["controller_id"])].append(row)
    if any(
        {(str(row["task_id"]), int(row["solver_seed"])) for row in by_controller[name]}
        != expected
        for name in CONTROLLERS
    ):
        raise ValueError("formal controller coverage differs from the frozen cohort")
    indexed = {
        controller: {_episode_key(row): row for row in by_controller[controller]}
        for controller in CONTROLLERS
    }
    integrity_errors = []
    for key in sorted(expected):
        summaries_at_key = [indexed[name][key]["summary"] for name in CONTROLLERS]
        fingerprints = {str(row["initial_fingerprint"]) for row in summaries_at_key}
        initial_conflicts = {int(row["initial_conflicts"]) for row in summaries_at_key}
        if len(fingerprints) != 1 or len(initial_conflicts) != 1:
            integrity_errors.append({"task_id": key[0], "solver_seed": key[1]})
    summaries = {controller: _aggregate_rows(by_controller[controller]) for controller in CONTROLLERS}
    original = summaries["v2-full"]
    mixed = summaries["mixed-full-v2"]
    adaptive = summaries["official_adaptive"]
    mixed_ttf_improvement = _relative_improvement(
        original["mean_capped_wall_ttf"], mixed["mean_capped_wall_ttf"]
    )
    mixed_auc_change = -_relative_improvement(
        original["mean_fixed_auc"], mixed["mean_fixed_auc"]
    )
    mixed_bootstrap = _paired_bootstrap_by_map(
        indexed["v2-full"], indexed["mixed-full-v2"], metric="capped_wall_ttf"
    )
    comparisons_by_stratum = _paired_group_comparison(
        indexed["v2-full"], indexed["mixed-full-v2"], schedule_index, "conflict_stratum"
    )
    comparisons_by_map = _paired_group_comparison(
        indexed["v2-full"], indexed["mixed-full-v2"], schedule_index, "map_id"
    )
    comparisons_by_source = _paired_group_comparison(
        indexed["v2-full"], indexed["mixed-full-v2"], schedule_index, "source_group"
    )
    comparisons_by_agent_band = _paired_group_comparison(
        indexed["v2-full"], indexed["mixed-full-v2"], schedule_index, "agent_band"
    )
    maps_not_worse = sum(row["candidate_not_worse"] for row in comparisons_by_map.values())
    strata_not_worse = sum(
        row["candidate_not_worse"] for row in comparisons_by_stratum.values()
    )
    gate = {
        "success_not_lower_than_v2": mixed["success_count"] >= original["success_count"],
        "ttf_improvement_at_least_5_percent": mixed_ttf_improvement >= 0.05,
        "auc_not_worse_than_2_percent": mixed_auc_change <= 0.02,
        "at_least_8_of_12_maps_not_worse": maps_not_worse >= 8,
        "at_least_2_of_3_strata_not_worse": strata_not_worse >= 2,
        "map_bootstrap_no_significant_degradation": mixed_bootstrap[
            "improvement_95_ci"
        ][1]
        >= 0.0,
        "identical_initial_states": not integrity_errors,
        "zero_experiment_errors": all(
            summaries[name][field] == 0
            for name in CONTROLLERS
            for field in ("invalid_action_count", "fingerprint_mismatch_count", "error_count")
        ),
    }
    report = {
        "schema": "lns2.balanced_wall_clock_report.v1",
        "evidence_level": "end_to_end_balanced_wall_clock",
        "summaries": summaries,
        "comparisons": {
            "mixed_vs_v2_ttf_improvement": mixed_ttf_improvement,
            "mixed_vs_v2_auc_change": mixed_auc_change,
            "v2_vs_adaptive_ttf_improvement": _relative_improvement(
                adaptive["mean_capped_wall_ttf"], original["mean_capped_wall_ttf"]
            ),
            "mixed_vs_v2_map_bootstrap": mixed_bootstrap,
            "mixed_vs_v2_by_stratum": comparisons_by_stratum,
            "mixed_vs_v2_by_map": comparisons_by_map,
            "mixed_vs_v2_by_source": comparisons_by_source,
            "mixed_vs_v2_by_agent_band": comparisons_by_agent_band,
        },
        "initial_state_integrity": {
            "passed": not integrity_errors,
            "mismatch_count": len(integrity_errors),
            "mismatches": integrity_errors,
        },
        "promotion_gate": gate,
        "decision": "mixed_full_candidate" if all(gate.values()) else "keep_v2_full",
        "note": (
            "Wall TTF includes reset, online control, and repair. Runtime and generated-node "
            "components are diagnostics; promotion follows the preregistered paired gates."
        ),
    }
    output_root = Path(output).resolve()
    _write_json(output_root / "balanced_wall_clock_report.json", report)
    return report


__all__ = [
    "CONTROLLERS",
    "STRATA",
    "analyze_scheduled",
    "collect_scheduled",
    "conflict_stratum",
    "merge_datasets",
    "prepare_movingai_dataset",
    "select_balanced_cohort",
]
