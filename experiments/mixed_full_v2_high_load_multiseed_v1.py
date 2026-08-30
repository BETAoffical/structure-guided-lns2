from __future__ import annotations

import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.balanced_wall_clock import analyze_success_only_ttf
from experiments.closed_loop_confirmation import run_closed_loop_collection
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


SCHEMA = "lns2.mixed_full_v2_high_load_multiseed.v1"
CONTROLLERS = ("official_adaptive", "v2-full", "mixed-full-v2")
ORDERS = (
    ("official_adaptive", "v2-full", "mixed-full-v2"),
    ("official_adaptive", "mixed-full-v2", "v2-full"),
    ("v2-full", "official_adaptive", "mixed-full-v2"),
    ("v2-full", "mixed-full-v2", "official_adaptive"),
    ("mixed-full-v2", "official_adaptive", "v2-full"),
    ("mixed-full-v2", "v2-full", "official_adaptive"),
)


def _mean(values: Iterable[float | int]) -> float:
    items = [float(value) for value in values]
    return statistics.fmean(items) if items else 0.0


def _fingerprint(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).resolve()
    config = _read_json(config_path)
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("unsupported high-load expansion config")
    design = dict(config.get("development_schedule") or {})
    groups = list(design.get("groups") or [])
    if len(groups) != 6:
        raise ValueError("development schedule must contain six groups")
    tasks: list[str] = []
    for index, group in enumerate(groups):
        if int(group.get("schedule_group", -1)) != index:
            raise ValueError("schedule groups must be ordered 0..5")
        if tuple(map(str, group.get("controller_order") or ())) != ORDERS[index]:
            raise ValueError("controller-order rotation differs from registration")
        rows = list(group.get("tasks") or [])
        if len(rows) != 3:
            raise ValueError("each schedule group must contain three task templates")
        for row in rows:
            task_id = str(row.get("task_id", ""))
            if not task_id or str(row.get("expected_cell", "")) not in {
                "extreme_high",
                "extreme_medium",
                "high_medium",
            }:
                raise ValueError("invalid task registration")
            tasks.append(task_id)
    if len(tasks) != 18 or len(set(tasks)) != 18:
        raise ValueError("development schedule must contain 18 unique tasks")
    seeds = tuple(map(int, config.get("solver_seeds") or ()))
    if seeds != (1, 2, 3):
        raise ValueError("the multiseed schedule is frozen to solver seeds 1,2,3")
    if int(design.get("expected_paired_key_count", -1)) != 54:
        raise ValueError("expected paired-key count must be 54")
    if int(config.get("workers", -1)) != 1:
        raise ValueError("timed TTF collection must be serial")
    if float(config.get("wall_time_budget_seconds", 0.0)) != 300.0:
        raise ValueError("wall-time metric budget must remain 300 seconds")
    if float(config.get("episode_process_timeout_seconds", 0.0)) != 360.0:
        raise ValueError("episode process fuse must remain 360 seconds")
    return config_path, config


def _task_registrations(config: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {**dict(row), "schedule_group": int(group["schedule_group"])}
        for group in config["development_schedule"]["groups"]
        for row in group["tasks"]
    ]


def _dataset_rows(dataset: Path, split: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(dataset / split / "manifest.jsonl")
    index = {str(row["task_id"]): dict(row) for row in rows}
    if len(index) != len(rows):
        raise ValueError("dataset manifest contains duplicate task IDs")
    return index


def build_plan(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    original_bundle: str | Path,
    mixed_bundle: str | Path,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = Path(dataset).resolve()
    config_file, config = _load_config(config_path)
    output_root = Path(output).resolve()
    design = config["development_schedule"]
    split = str(config["split"])
    manifest_path = dataset_root / split / "manifest.jsonl"
    if sha256_file(manifest_path) != str(design["dataset_manifest_sha256"]):
        raise ValueError("dataset manifest SHA differs from the registered pool")
    rows = _dataset_rows(dataset_root, split)
    registrations = _task_registrations(config)
    task_ids = [str(row["task_id"]) for row in registrations]
    missing = sorted(set(task_ids) - set(rows))
    if missing:
        raise ValueError(f"registered tasks are missing from dataset: {missing[:3]}")
    for task_id in task_ids:
        task_file = dataset_root / split / str(rows[task_id]["task_file"])
        if not task_file.is_file():
            raise ValueError(f"registered task file is missing: {task_id}")
    old_schedule = project_root / str(design["excluded_schedule"])
    if sha256_file(old_schedule) != str(design["excluded_schedule_sha256"]):
        raise ValueError("excluded development schedule SHA differs")
    old_tasks = {
        str(row["task_id"])
        for row in _read_json(old_schedule).get("entries", [])
    }
    overlap = sorted(set(task_ids) & old_tasks)
    if overlap:
        raise ValueError(f"new schedule reuses old task IDs: {overlap[:3]}")
    original_root = Path(original_bundle).resolve()
    mixed_root = Path(mixed_bundle).resolve()
    original_manifest = original_root / "controller_manifest.json"
    mixed_manifest = mixed_root / "controller_manifest.json"
    if sha256_file(original_manifest) != str(design["original_bundle_manifest_sha256"]):
        raise ValueError("original controller bundle SHA differs")
    if sha256_file(mixed_manifest) != str(design["mixed_bundle_manifest_sha256"]):
        raise ValueError("Mixed controller bundle SHA differs")
    job_keys = sorted(
        (task_id, int(seed))
        for task_id in task_ids
        for seed in config["solver_seeds"]
    )
    source_counts = collections.Counter(str(rows[task]["source_group"]) for task in task_ids)
    map_ids = {str(rows[task]["map_id"]) for task in task_ids}
    plan_identity = {
        "schema": SCHEMA,
        "config_sha256": sha256_file(config_file),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "original_bundle_manifest_sha256": sha256_file(original_manifest),
        "mixed_bundle_manifest_sha256": sha256_file(mixed_manifest),
        "task_ids": sorted(task_ids),
        "solver_seeds": list(config["solver_seeds"]),
        "controller_orders": [list(order) for order in ORDERS],
        "wall_time_budget_seconds": 300.0,
    }
    return {
        "schema": SCHEMA,
        "status": "planned",
        "run_identity": _fingerprint(plan_identity),
        "dataset": str(dataset_root),
        "config": str(config_file),
        "output": str(output_root),
        "task_template_count": len(task_ids),
        "paired_key_count": len(job_keys),
        "timed_episode_count": len(job_keys) * len(CONTROLLERS),
        "qualification_episode_count": len(job_keys),
        "source_counts": dict(sorted(source_counts.items())),
        "map_count": len(map_ids),
        "solver_seeds": list(config["solver_seeds"]),
        "workers": 1,
        "qualification_workers": int(design["qualification_workers"]),
        "controller_orders": [list(order) for order in ORDERS],
        "job_keys": [[task_id, seed] for task_id, seed in job_keys],
        "identity": plan_identity,
    }


def _actual_cell(conflicts: int, generated: int) -> str:
    conflict = "extreme" if conflicts >= 501 else "high" if conflicts >= 101 else "below_high"
    load = "high" if generated >= 1_000_001 else "medium" if generated >= 100_001 else "low"
    return f"{conflict}_{load}"


def _agent_band(count: int) -> str:
    if count <= 160:
        return "small"
    if count <= 400:
        return "medium"
    return "large"


def _materialize_schedule(
    plan: dict[str, Any], config: dict[str, Any], qualification_root: Path
) -> tuple[Path, dict[str, Any]]:
    qualification_path = qualification_root / "qualification_manifest.jsonl"
    qualified = _read_jsonl(qualification_path)
    expected = {(str(task), int(seed)) for task, seed in plan["job_keys"]}
    indexed = {(str(row["task_id"]), int(row["solver_seed"])): row for row in qualified}
    if set(indexed) != expected or len(indexed) != 54:
        raise ValueError("fresh qualification coverage differs from the 54 registered keys")
    dataset_rows = _dataset_rows(Path(plan["dataset"]), str(config["split"]))
    by_task = {str(row["task_id"]): row for row in _task_registrations(config)}
    entries: list[dict[str, Any]] = []
    for group in config["development_schedule"]["groups"]:
        group_id = int(group["schedule_group"])
        order = list(map(str, group["controller_order"]))
        for registration in group["tasks"]:
            task_id = str(registration["task_id"])
            source = dataset_rows[task_id]
            for seed in config["solver_seeds"]:
                result = indexed[(task_id, int(seed))]
                if str(result.get("status")) not in {"ok", "resumed"} or not bool(
                    result.get("initial_complete")
                ):
                    raise ValueError(f"fresh qualification failed: {task_id}/{seed}")
                complexity = dict(result["initial_complexity"])
                conflicts = int(result["initial_conflicts"])
                generated = int(complexity["initial_low_level_generated"])
                if conflicts < 101 or generated < 100_001:
                    raise ValueError(f"registered key is no longer high-load: {task_id}/{seed}")
                actual_cell = _actual_cell(conflicts, generated)
                expected_cell = str(by_task[task_id]["expected_cell"])
                entries.append(
                    {
                        "task_id": task_id,
                        "solver_seed": int(seed),
                        "schedule_group": group_id,
                        "controller_order": order,
                        "selection_expected_cell": expected_cell,
                        "actual_load_cell": actual_cell,
                        "map_id": str(source["map_id"]),
                        "source_group": str(source["source_group"]),
                        "layout_mode": str(source["layout_mode"]),
                        "agent_count": int(result["agent_count"]),
                        "agent_band": _agent_band(int(result["agent_count"])),
                        "initial_conflicts": conflicts,
                        "conflict_stratum": "high",
                        "initial_pp_load_stratum": (
                            "high" if generated >= 1_000_001 else "medium"
                        ),
                        "initial_low_level_generated": generated,
                        "initial_low_level_expanded": int(
                            complexity["initial_low_level_expanded"]
                        ),
                        "conflict_event_count": int(complexity["conflict_event_count"]),
                        "active_conflict_agent_ratio": float(
                            complexity["active_conflict_agent_ratio"]
                        ),
                        "largest_conflict_component_ratio": float(
                            complexity["largest_conflict_component_ratio"]
                        ),
                        "total_path_cost": int(complexity["total_path_cost"]),
                        "state_fingerprint": str(result["state_fingerprint"]),
                    }
                )
    entries.sort(key=lambda row: (int(row["schedule_group"]), str(row["task_id"]), int(row["solver_seed"])))
    counts = collections.Counter(str(row["actual_load_cell"]) for row in entries)
    if counts["extreme_high"] < 12 or counts["extreme_medium"] < 18 or counts["high_medium"] < 12:
        raise ValueError(f"fresh load mix is too weak: {dict(sorted(counts.items()))}")
    schedule = {
        "schema": SCHEMA,
        "run_identity": plan["run_identity"],
        "selection_blind_to_controller_outcomes": True,
        "fresh_qualification_sha256": sha256_file(qualification_path),
        "entries": entries,
    }
    cohort_root = Path(plan["output"]) / "cohort"
    _write_json(cohort_root / "execution_schedule.json", schedule)
    report = {
        "schema": SCHEMA,
        "status": "development_schedule_ready",
        "formal_collection_allowed": False,
        "development_collection_allowed": True,
        "run_identity": plan["run_identity"],
        "paired_key_count": len(entries),
        "timed_episode_count": len(entries) * len(CONTROLLERS),
        "task_template_count": 18,
        "solver_seeds": [1, 2, 3],
        "actual_load_cell_counts": dict(sorted(counts.items())),
        "source_counts": dict(
            sorted(collections.Counter(str(row["source_group"]) for row in entries).items())
        ),
        "map_count": len({str(row["map_id"]) for row in entries}),
        "fresh_qualification_sha256": sha256_file(qualification_path),
        "execution_schedule_sha256": sha256_file(cohort_root / "execution_schedule.json"),
    }
    _write_json(cohort_root / "cohort_report.json", report)
    return cohort_root, report


def _common_collection_kwargs(
    *, dataset: Path, config: Path, output: Path, task_ids: list[str], keys: set[tuple[str, int]], bundle: Path
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "config_path": config,
        "output": output,
        "workers": 1,
        "task_ids": task_ids,
        "controller": "v2-full",
        "feature_backend": "auto",
        "controller_bundle": bundle,
        "controller_runtime": "optimized",
        "verification_profile": "deployment",
        "job_keys": keys,
        "cohort_job_keys": keys,
        "stopping_rule": "wall-clock-fixed-metric",
        "use_global_collection_lock": False,
    }


def _collect(
    plan: dict[str, Any], config: dict[str, Any], cohort_root: Path, original_bundle: Path, mixed_bundle: Path
) -> dict[str, Any]:
    schedule = _read_json(cohort_root / "execution_schedule.json")
    output_root = Path(plan["output"]) / "collection"
    qualification_root = Path(plan["output"]) / "qualification"
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
    progress: list[dict[str, Any]] = []
    for group_id, order in enumerate(ORDERS):
        rows = [row for row in schedule["entries"] if int(row["schedule_group"]) == group_id]
        keys = {(str(row["task_id"]), int(row["solver_seed"])) for row in rows}
        task_ids = sorted({task_id for task_id, _seed in keys})
        if len(keys) != 9 or len(task_ids) != 3:
            raise ValueError(f"schedule group {group_id} is not 3 tasks x 3 seeds")
        for controller_id in order:
            lane = output_root / f"order_{group_id}" / controller_id
            common = _common_collection_kwargs(
                dataset=Path(plan["dataset"]),
                config=Path(plan["config"]),
                output=lane,
                task_ids=task_ids,
                keys=keys,
                bundle=bundles[controller_id],
            )
            if not (lane / "run_config.json").is_file():
                run_closed_loop_collection(
                    **common,
                    phase="qualify",
                    qualification_source=qualification_root,
                    resume=False,
                )
            result = run_closed_loop_collection(
                **common,
                phase=phases[controller_id],
                resume=True,
            )
            progress.append(
                {
                    "schedule_group": group_id,
                    "controller": controller_id,
                    "job_count": len(keys),
                    "summary": result,
                }
            )
            _write_json(
                output_root / "collection_progress.json",
                {"schema": SCHEMA, "run_identity": plan["run_identity"], "entries": progress},
            )
    return {"lane_count": len(progress), "entries": progress}


def _manifest_index(collection_root: Path, schedule: dict[str, Any]) -> dict[str, dict[tuple[str, int], dict[str, Any]]]:
    indexed: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for controller in CONTROLLERS:
        phase = "official_adaptive" if controller == "official_adaptive" else "realized_dynamic"
        rows: list[dict[str, Any]] = []
        for group in range(6):
            rows.extend(_read_jsonl(collection_root / f"order_{group}" / controller / f"{phase}_manifest.jsonl"))
        indexed[controller] = {(str(row["task_id"]), int(row["solver_seed"])): row for row in rows}
    expected = {(str(row["task_id"]), int(row["solver_seed"])) for row in schedule["entries"]}
    if any(set(indexed[controller]) != expected for controller in CONTROLLERS):
        raise ValueError("controller coverage differs from the 54-key schedule")
    return indexed


def _controller_summary(rows: list[dict[str, Any]], cap: float) -> dict[str, Any]:
    summaries = [dict(row["summary"]) for row in rows]
    successes = sum(bool(row["success"]) for row in summaries)
    observed = sum(float(row["episode_observed_wall_seconds"]) for row in summaries)
    capped = [float(row["capped_wall_time_to_feasible"]) for row in summaries]
    return {
        "episode_count": len(rows),
        "success_count": successes,
        "success_rate": successes / len(rows),
        "mean_capped_ttf_seconds": _mean(capped),
        "median_capped_ttf_seconds": statistics.median(capped),
        "total_observed_wall_seconds": observed,
        "successes_per_observed_hour": successes * 3600.0 / observed if observed else 0.0,
        "success_at_seconds": {
            str(int(limit)): sum(
                bool(row["success"]) and float(row["wall_time_to_feasible"]) <= limit
                for row in summaries
            )
            for limit in (30.0, 60.0, 120.0, cap)
        },
        "mean_reset_seconds": _mean(float(row["reset_wall_seconds"]) for row in summaries),
        "mean_repair_wall_seconds": _mean(float(row.get("repair_wall_seconds", 0.0)) for row in summaries),
        "mean_pp_replan_seconds": _mean(
            float(dict(row.get("controller_totals") or {}).get("pp_replan_seconds", 0.0))
            for row in summaries
        ),
        "mean_selection_seconds": _mean(
            float(dict(row.get("controller_totals") or {}).get("neighborhood_selection_seconds", 0.0))
            for row in summaries
        ),
        "mean_repair_iterations": _mean(int(row.get("repair_iterations", 0)) for row in summaries),
    }


def _comparison(
    baseline: dict[tuple[str, int], dict[str, Any]],
    candidate: dict[tuple[str, int], dict[str, Any]],
    keys: list[tuple[str, int]],
    cap: float,
) -> dict[str, Any]:
    left = [dict(baseline[key]["summary"]) for key in keys]
    right = [dict(candidate[key]["summary"]) for key in keys]
    left_ttf = [float(row["capped_wall_time_to_feasible"]) for row in left]
    right_ttf = [float(row["capped_wall_time_to_feasible"]) for row in right]
    wins = [r < l for l, r in zip(left_ttf, right_ttf)]
    losses = [r > l for l, r in zip(left_ttf, right_ttf)]
    by_task: dict[str, list[bool]] = collections.defaultdict(list)
    for key, won in zip(keys, wins):
        by_task[key[0]].append(won)
    baseline_observed = sum(float(row["episode_observed_wall_seconds"]) for row in left)
    candidate_observed = sum(float(row["episode_observed_wall_seconds"]) for row in right)
    baseline_successes = sum(bool(row["success"]) for row in left)
    candidate_successes = sum(bool(row["success"]) for row in right)
    baseline_mean, candidate_mean = _mean(left_ttf), _mean(right_ttf)
    return {
        "paired_key_count": len(keys),
        "candidate_faster_count": sum(wins),
        "tie_count": len(keys) - sum(wins) - sum(losses),
        "candidate_slower_count": sum(losses),
        "candidate_faster_fraction": sum(wins) / len(keys),
        "baseline_success_count": baseline_successes,
        "candidate_success_count": candidate_successes,
        "candidate_only_success_count": sum(
            bool(r["success"]) and not bool(l["success"]) for l, r in zip(left, right)
        ),
        "baseline_only_success_count": sum(
            bool(l["success"]) and not bool(r["success"]) for l, r in zip(left, right)
        ),
        "mean_capped_ttf_improvement": (
            (baseline_mean - candidate_mean) / baseline_mean if baseline_mean else 0.0
        ),
        "successes_per_observed_hour_ratio": (
            (candidate_successes / candidate_observed) / (baseline_successes / baseline_observed)
            if candidate_observed and baseline_observed and baseline_successes
            else None
        ),
        "task_templates_with_at_least_two_of_three_faster": sum(
            sum(values) >= 2 for values in by_task.values()
        ),
        "task_templates_with_three_of_three_faster": sum(
            sum(values) == 3 for values in by_task.values()
        ),
        "success_at_seconds_delta": {
            str(int(limit)): sum(
                bool(row["success"]) and float(row["wall_time_to_feasible"]) <= limit
                for row in right
            )
            - sum(
                bool(row["success"]) and float(row["wall_time_to_feasible"]) <= limit
                for row in left
            )
            for limit in (30.0, 60.0, 120.0, cap)
        },
    }


def analyze(output: str | Path) -> dict[str, Any]:
    output_root = Path(output).resolve()
    cohort_root = output_root / "cohort"
    collection_root = output_root / "collection"
    schedule_path = cohort_root / "execution_schedule.json"
    schedule = _read_json(schedule_path)
    indexed = _manifest_index(collection_root, schedule)
    schedule_index = {
        (str(row["task_id"]), int(row["solver_seed"])): row
        for row in schedule["entries"]
    }
    keys = sorted(schedule_index)
    for key in keys:
        fingerprints = {
            str(indexed[controller][key]["summary"]["initial_fingerprint"])
            for controller in CONTROLLERS
        }
        conflicts = {
            int(indexed[controller][key]["summary"]["initial_conflicts"])
            for controller in CONTROLLERS
        }
        if fingerprints != {str(schedule_index[key]["state_fingerprint"])} or conflicts != {
            int(schedule_index[key]["initial_conflicts"])
        }:
            raise ValueError(f"paired initial state differs from fresh qualification: {key}")
    if any(
        str(indexed[controller][key].get("status")) not in {"ok", "resumed"}
        or not isinstance(indexed[controller][key].get("summary"), dict)
        for controller in CONTROLLERS
        for key in keys
    ):
        raise ValueError("analysis requires complete timed episodes")
    cap = float(indexed["official_adaptive"][keys[0]]["summary"]["wall_time_budget_seconds"])
    controllers = {
        controller: _controller_summary([indexed[controller][key] for key in keys], cap)
        for controller in CONTROLLERS
    }
    comparisons = {
        "mixed_vs_official": _comparison(
            indexed["official_adaptive"], indexed["mixed-full-v2"], keys, cap
        ),
        "mixed_vs_v2": _comparison(indexed["v2-full"], indexed["mixed-full-v2"], keys, cap),
        "v2_vs_official": _comparison(indexed["official_adaptive"], indexed["v2-full"], keys, cap),
    }
    by_cell: dict[str, Any] = {}
    for cell in sorted({str(row["actual_load_cell"]) for row in schedule["entries"]}):
        cell_keys = [key for key in keys if str(schedule_index[key]["actual_load_cell"]) == cell]
        by_cell[cell] = {
            "paired_key_count": len(cell_keys),
            "controllers": {
                controller: _controller_summary(
                    [indexed[controller][key] for key in cell_keys], cap
                )
                for controller in CONTROLLERS
            },
            "mixed_vs_official": _comparison(
                indexed["official_adaptive"], indexed["mixed-full-v2"], cell_keys, cap
            ),
            "mixed_vs_v2": _comparison(
                indexed["v2-full"], indexed["mixed-full-v2"], cell_keys, cap
            ),
        }
    success_only = analyze_success_only_ttf(
        collection_root, cohort_root, output_root / "success_only_ttf"
    )
    report = {
        "schema": SCHEMA,
        "status": "complete",
        "evidence_level": "deterministic_multiseed_sequential_development",
        "formal_promotion_allowed": False,
        "run_identity": str(schedule["run_identity"]),
        "paired_initial_state_integrity": True,
        "task_template_count": 18,
        "solver_seeds": [1, 2, 3],
        "paired_key_count": len(keys),
        "timed_episode_count": len(keys) * len(CONTROLLERS),
        "wall_time_budget_seconds": cap,
        "controllers": controllers,
        "comparisons": comparisons,
        "by_actual_load_cell": by_cell,
        "success_only_report_sha256": sha256_file(
            output_root / "success_only_ttf" / "success_only_ttf_report.json"
        ),
        "input": {
            "schedule_sha256": sha256_file(schedule_path),
            "collection_progress_sha256": sha256_file(
                collection_root / "collection_progress.json"
            ),
        },
    }
    _write_json(output_root / "analysis_report.json", report)
    return {**report, "success_only": success_only}


def run(
    dataset: str | Path,
    config_path: str | Path,
    output: str | Path,
    original_bundle: str | Path,
    mixed_bundle: str | Path,
) -> dict[str, Any]:
    plan = build_plan(dataset, config_path, output, original_bundle, mixed_bundle)
    _config_file, config = _load_config(config_path)
    output_root = Path(output).resolve()
    qualification_root = output_root / "qualification"
    all_task_ids = sorted({str(task) for task, _seed in plan["job_keys"]})
    all_keys = {(str(task), int(seed)) for task, seed in plan["job_keys"]}
    qualification = run_closed_loop_collection(
        Path(plan["dataset"]),
        Path(plan["config"]),
        qualification_root,
        phase="qualify",
        workers=int(config["development_schedule"]["qualification_workers"]),
        resume=(qualification_root / "run_config.json").is_file(),
        task_ids=all_task_ids,
        controller="v2-full",
        feature_backend="auto",
        controller_bundle=Path(original_bundle).resolve(),
        controller_runtime="optimized",
        verification_profile="deployment",
        job_keys=all_keys,
        cohort_job_keys=all_keys,
        stopping_rule="wall-clock-fixed-metric",
        use_global_collection_lock=False,
    )
    cohort_root, cohort = _materialize_schedule(plan, config, qualification_root)
    collection = _collect(
        plan,
        config,
        cohort_root,
        Path(original_bundle).resolve(),
        Path(mixed_bundle).resolve(),
    )
    result = analyze(output_root)
    return {
        "schema": SCHEMA,
        "status": "complete",
        "plan": plan,
        "qualification": qualification,
        "cohort": cohort,
        "collection_lane_count": collection["lane_count"],
        "analysis": result,
    }


__all__ = ["analyze", "build_plan", "run"]
