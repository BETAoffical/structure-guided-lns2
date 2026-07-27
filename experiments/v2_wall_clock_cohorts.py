from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, read_json, sha256_file
from experiments.lns2_bottleneck import load_track
from experiments.repair_collection import _write_json
from experiments.tradeoff_evaluation import _manifest_path as controller_manifest_path


REPORT_SCHEMA = "lns2.v2_lns2_seven_map_report.v1"
TRACK = "wall-clock-600"
CONTROLLERS = ("official_adaptive", "v2-full")
RUN_CONFIG_CONTROLLERS = {
    "official_adaptive": "v1-full",
    "v2-full": "v2-full",
}
EXPECTED_TASKS = (
    "maze-32-32-4__random_05__agents_0100",
    "den312d__random_05__agents_0200",
    "room-64-64-16__random_05__agents_0400",
    "room-64-64-16__random_04__agents_0600",
    "random-32-32-10__random_05__agents_0200",
    "warehouse-10-20-10-2-2__random_05__agents_0500",
    "random-64-64-10__random_04__agents_0600",
)
EXPECTED_SEEDS = (1, 2, 3)
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260727


def _mean(values: Iterable[float | int | None]) -> float | None:
    numbers = [float(value) for value in values if value is not None]
    return statistics.fmean(numbers) if numbers else None


def _ratio_change(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    if abs(float(baseline)) <= 1e-15:
        return 0.0 if abs(float(candidate)) <= 1e-15 else math.inf
    return float(candidate) / float(baseline) - 1.0


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    rows: list[dict[str, Any]],
    *,
    baseline_field: str,
    candidate_field: str,
    seed: int,
) -> dict[str, Any]:
    pairs = [
        (float(row[baseline_field]), float(row[candidate_field]))
        for row in rows
        if row.get(baseline_field) is not None
        and row.get(candidate_field) is not None
    ]
    if not pairs:
        return {
            "pair_count": 0,
            "relative_change": None,
            "ci95_lower": None,
            "ci95_upper": None,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": seed,
        }
    point = _ratio_change(
        statistics.fmean(candidate for _baseline, candidate in pairs),
        statistics.fmean(baseline for baseline, _candidate in pairs),
    )
    generator = random.Random(seed)
    estimates: list[float] = []
    for _index in range(BOOTSTRAP_SAMPLES):
        sample = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        estimate = _ratio_change(
            statistics.fmean(candidate for _baseline, candidate in sample),
            statistics.fmean(baseline for baseline, _candidate in sample),
        )
        if estimate is not None and math.isfinite(estimate):
            estimates.append(estimate)
    return {
        "pair_count": len(pairs),
        "relative_change": point,
        "ci95_lower": _quantile(estimates, 0.025),
        "ci95_upper": _quantile(estimates, 0.975),
        "samples": BOOTSTRAP_SAMPLES,
        "seed": seed,
    }


def _canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _stratum(initial_conflicts: int) -> str:
    if initial_conflicts == 0:
        return "initially_feasible"
    if initial_conflicts <= 3:
        return "low_conflict_1_3"
    return "repair_demanding_ge_4"


def _source_track_root(source: Path) -> Path:
    return source / "tracks" / TRACK


def _required_collection_roots(source: Path) -> dict[str, Path]:
    collection_root = _source_track_root(source) / "collections"
    roots = {controller: collection_root / controller for controller in CONTROLLERS}
    missing = [str(path) for path in roots.values() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("required collection is missing: " + ", ".join(missing))
    return roots


def _source_status(source: Path) -> dict[str, Any]:
    status = read_json(source / "status.json")
    progress = read_json(source / "collection_progress.json")
    if str(status.get("status")) != "complete" or str(status.get("phase")) != "complete":
        raise ValueError(f"source is not complete: {source}")
    if str(progress.get("status")) != "complete" or str(progress.get("phase")) != "complete":
        raise ValueError(f"source collection progress is not complete: {source}")
    if int(progress.get("error_jobs", -1)) != 0:
        raise ValueError(f"source contains collection errors: {source}")
    if int(dict(status.get("bottleneck_validation") or {}).get("error_episode_count", 0)) != 0:
        raise ValueError(f"source contains episode errors: {source}")
    return status


def _collection_identity(root: Path, controller: str) -> dict[str, Any]:
    run = read_json(root / "run_config.json")
    configuration = dict(run.get("configuration") or {})
    environment = dict(configuration.get("environment") or {})
    implementation = dict(run.get("controller_implementation") or {})
    native = dict(implementation.get("native_module") or {})
    if str(run.get("controller")) != RUN_CONFIG_CONTROLLERS[controller]:
        raise ValueError(f"controller run-config mismatch: {root}")
    required_values = {
        "stopping_rule": configuration.get("stopping_rule"),
        "controller_runtime": configuration.get("controller_runtime"),
        "feature_backend": configuration.get("feature_backend"),
        "verification_profile": configuration.get("verification_profile"),
        "replan_algorithm": environment.get("replan_algorithm"),
        "use_sipp": environment.get("use_sipp"),
        "max_repair_iterations": environment.get("max_repair_iterations"),
        "time_limit": environment.get("time_limit"),
        "wall_time_budget_seconds": configuration.get("wall_time_budget_seconds"),
    }
    expected = {
        "stopping_rule": "wall-clock",
        "controller_runtime": "optimized",
        "feature_backend": "native",
        "verification_profile": "deployment",
        "replan_algorithm": "PP",
        "use_sipp": True,
        "max_repair_iterations": 0,
        "time_limit": 600.0,
        "wall_time_budget_seconds": 600.0,
    }
    if required_values != expected:
        raise ValueError(
            f"collection runtime/PP configuration mismatch: {root}: {required_values}"
        )
    run_fingerprint = str(run.get("run_fingerprint") or "")
    implementation_sha256 = str(implementation.get("sha256") or "")
    native_sha256 = str(native.get("sha256") or "")
    dataset_fingerprint = str(run.get("dataset_fingerprint") or "")
    if not all((run_fingerprint, implementation_sha256, native_sha256, dataset_fingerprint)):
        raise ValueError(f"collection identity is incomplete: {root}")
    return {
        "run_fingerprint": run_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "controller_implementation_sha256": implementation_sha256,
        "native_module_sha256": native_sha256,
        "controller_bundle_fingerprint": _canonical_fingerprint(
            run.get("controller_bundle")
        ),
        "replan_algorithm": environment.get("replan_algorithm"),
        "use_sipp": bool(environment.get("use_sipp")),
        "wall_time_budget_seconds": float(configuration.get("wall_time_budget_seconds")),
    }


def _source_identity(source: Path) -> dict[str, Any]:
    status = _source_status(source)
    runner = read_json(source / "runner_config.json")
    identity = dict(runner.get("identity") or {})
    controllers = tuple(map(str, identity.get("controllers") or []))
    if not set(CONTROLLERS).issubset(controllers):
        raise ValueError(f"source lacks required controllers: {source}")
    expected_runner = {
        "controller_runtime": "optimized",
        "feature_backend": "native",
        "verification_profile": "deployment",
        "paired_execution": "strict",
        "wall_clock_seconds": 600.0,
        "tracks": ["wall-clock"],
    }
    observed_runner = {key: identity.get(key) for key in expected_runner}
    if observed_runner != expected_runner:
        raise ValueError(f"runner identity mismatch: {source}: {observed_runner}")
    model_semantic_fingerprint = str(identity.get("model_semantic_fingerprint") or "")
    dataset = str(identity.get("dataset") or "")
    controller_bundle = str(identity.get("controller_bundle") or "")
    if not all((model_semantic_fingerprint, dataset, controller_bundle)):
        raise ValueError(f"runner identity is incomplete: {source}")
    roots = _required_collection_roots(source)
    collections = {
        controller: _collection_identity(root, controller)
        for controller, root in roots.items()
    }
    return {
        "source": str(source),
        "status_episode_count": int(status.get("episode_count", 0)),
        "runner_identity_fingerprint": str(runner.get("identity_fingerprint") or ""),
        "dataset": dataset,
        "controller_bundle": controller_bundle,
        "model_semantic_fingerprint": model_semantic_fingerprint,
        "collections": collections,
    }


def _validate_cross_source_identity(identities: list[dict[str, Any]]) -> None:
    for field in ("dataset", "controller_bundle", "model_semantic_fingerprint"):
        if len({str(identity[field]) for identity in identities}) != 1:
            raise ValueError(f"source {field} mismatch")
    for controller in CONTROLLERS:
        for field in (
            "dataset_fingerprint",
            "controller_implementation_sha256",
            "native_module_sha256",
            "controller_bundle_fingerprint",
            "replan_algorithm",
            "use_sipp",
            "wall_time_budget_seconds",
        ):
            values = {
                identity["collections"][controller][field] for identity in identities
            }
            if len(values) != 1:
                raise ValueError(f"source {controller} {field} mismatch")


def _source_hashes(source: Path) -> dict[str, Any]:
    files = {
        "status.json": source / "status.json",
        "collection_progress.json": source / "collection_progress.json",
        "runner_config.json": source / "runner_config.json",
        f"tracks/{TRACK}/execution_schedule.json": (
            _source_track_root(source) / "execution_schedule.json"
        ),
    }
    roots = _required_collection_roots(source)
    for controller, root in roots.items():
        files[f"{controller}/run_config.json"] = root / "run_config.json"
        files[f"{controller}/manifest.jsonl"] = controller_manifest_path(
            root, controller
        )
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"source evidence files are missing from {source}: {', '.join(missing)}"
        )
    hashes = {name: sha256_file(path) for name, path in sorted(files.items())}
    return {
        "source": str(source),
        "files": hashes,
        "aggregate_sha256": _canonical_fingerprint(hashes),
    }


def _episode_key(row: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(row.get("task_id")),
        int(row.get("solver_seed", -1)),
        str(row.get("controller")),
    )


def _validate_episodes(episodes: list[dict[str, Any]]) -> None:
    expected_keys = {
        (task, seed, controller)
        for task in EXPECTED_TASKS
        for seed in EXPECTED_SEEDS
        for controller in CONTROLLERS
    }
    keys = [_episode_key(row) for row in episodes]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        raise ValueError(f"cohort sources overlap: {duplicates[:3]}")
    if set(keys) != expected_keys:
        missing = sorted(expected_keys - set(keys))
        unexpected = sorted(set(keys) - expected_keys)
        raise ValueError(
            f"seven-map coverage mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    invalid_status = [key for key, row in zip(keys, episodes) if row.get("status") not in {"ok", "resumed"}]
    if invalid_status:
        raise ValueError(f"episode status is not valid: {invalid_status[:3]}")
    invalid_budget = [
        key
        for key, row in zip(keys, episodes)
        if str(row.get("stopping_rule")) != "wall-clock"
        or abs(float(row.get("wall_time_budget_seconds", 0.0)) - 600.0) > 1e-12
    ]
    if invalid_budget:
        raise ValueError(f"episode wall-clock budget mismatch: {invalid_budget[:3]}")
    timing_failures = [
        key
        for key, row in zip(keys, episodes)
        if (
            int(row.get("repair_iterations", 0)) > 0
            and not bool(row.get("timing_instrumentation_complete"))
        )
        or not bool(row.get("finalization_timing_instrumented"))
        or float(row.get("process_timing_closure_error_seconds", math.inf))
        > max(0.02, 0.02 * float(row.get("episode_process_wall_seconds", 0.0)))
    ]
    if timing_failures:
        raise ValueError(f"episode timing closure failed: {timing_failures[:3]}")


def _paired_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {_episode_key(row): row for row in episodes}
    result: list[dict[str, Any]] = []
    for task_id in EXPECTED_TASKS:
        for seed in EXPECTED_SEEDS:
            lns2 = indexed[(task_id, seed, "official_adaptive")]
            v2 = indexed[(task_id, seed, "v2-full")]
            if lns2.get("initial_fingerprint") != v2.get("initial_fingerprint"):
                raise ValueError(f"paired initial fingerprint mismatch: {(task_id, seed)}")
            if int(lns2.get("initial_conflicts", 0)) != int(v2.get("initial_conflicts", 0)):
                raise ValueError(f"paired initial conflict mismatch: {(task_id, seed)}")
            initial_conflicts = int(lns2.get("initial_conflicts", 0))
            lns2_repairs = int(lns2.get("repair_iterations", 0))
            v2_repairs = int(v2.get("repair_iterations", 0))
            lns2_no_progress = int(lns2.get("no_improvement_repair_count", 0))
            v2_no_progress = int(v2.get("no_improvement_repair_count", 0))
            lns2_ttf = float(lns2.get("restricted_time_to_feasible", 0.0))
            v2_ttf = float(v2.get("restricted_time_to_feasible", 0.0))
            lns2_auc = lns2.get("normalized_wall_clock_conflict_auc")
            v2_auc = v2.get("normalized_wall_clock_conflict_auc")
            lns2_soc = int(lns2.get("budget_final_sum_of_costs", 0)) if lns2.get("success") else None
            v2_soc = int(v2.get("budget_final_sum_of_costs", 0)) if v2.get("success") else None
            result.append(
                {
                    "track": TRACK,
                    "task_id": task_id,
                    "map_id": lns2.get("map_id"),
                    "layout_family": lns2.get("layout_family"),
                    "agent_count": int(lns2.get("agent_count", 0)),
                    "solver_seed": seed,
                    "initial_fingerprint": lns2.get("initial_fingerprint"),
                    "initial_conflicts": initial_conflicts,
                    "conflict_stratum": _stratum(initial_conflicts),
                    "lns2_success": bool(lns2.get("success")),
                    "v2_success": bool(v2.get("success")),
                    "common_success": bool(lns2.get("success") and v2.get("success")),
                    "lns2_capped_ttf": lns2_ttf,
                    "v2_capped_ttf": v2_ttf,
                    "v2_ttf_relative_change": _ratio_change(v2_ttf, lns2_ttf),
                    "speedup_lns2_over_v2": lns2_ttf / max(1e-15, v2_ttf),
                    "lns2_initialization_seconds": float(lns2.get("environment_construct_seconds", 0.0)) + float(lns2.get("reset_wall_seconds", 0.0)),
                    "v2_initialization_seconds": float(v2.get("environment_construct_seconds", 0.0)) + float(v2.get("reset_wall_seconds", 0.0)),
                    "lns2_selection_seconds": float(lns2.get("neighborhood_selection_seconds", 0.0)),
                    "v2_selection_seconds": float(v2.get("neighborhood_selection_seconds", 0.0)),
                    "lns2_pp_seconds": float(lns2.get("pp_replan_seconds", 0.0)),
                    "v2_pp_seconds": float(v2.get("pp_replan_seconds", 0.0)),
                    "lns2_repair_iterations": lns2_repairs,
                    "v2_repair_iterations": v2_repairs,
                    "lns2_no_progress_fraction": lns2_no_progress / lns2_repairs if lns2_repairs else 0.0,
                    "v2_no_progress_fraction": v2_no_progress / v2_repairs if v2_repairs else 0.0,
                    "lns2_normalized_wall_auc": lns2_auc,
                    "v2_normalized_wall_auc": v2_auc,
                    "v2_auc_relative_change": _ratio_change(
                        float(v2_auc) if v2_auc is not None else None,
                        float(lns2_auc) if lns2_auc is not None else None,
                    ),
                    "lns2_soc_at_feasible": lns2_soc,
                    "v2_soc_at_feasible": v2_soc,
                    "v2_soc_relative_change": _ratio_change(
                        float(v2_soc) if v2_soc is not None else None,
                        float(lns2_soc) if lns2_soc is not None else None,
                    ),
                }
            )
    return result


def _controller_rows(
    episodes: list[dict[str, Any]], paired: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    common_keys = {
        (row["task_id"], row["solver_seed"])
        for row in paired
        if row["common_success"]
    }
    groups = {
        "all": lambda row: True,
        "repairable": lambda row: int(row.get("initial_conflicts", 0)) > 0,
        "common_success": lambda row: (
            str(row.get("task_id")), int(row.get("solver_seed", -1))
        ) in common_keys,
    }
    result: list[dict[str, Any]] = []
    for group, predicate in groups.items():
        for controller in CONTROLLERS:
            rows = [
                row
                for row in episodes
                if row.get("controller") == controller and predicate(row)
            ]
            repairs = sum(int(row.get("repair_iterations", 0)) for row in rows)
            no_progress = sum(
                int(row.get("no_improvement_repair_count", 0)) for row in rows
            )
            result.append(
                {
                    "group": group,
                    "controller": controller,
                    "episode_count": len(rows),
                    "success_count": sum(bool(row.get("success")) for row in rows),
                    "mean_capped_ttf": _mean(row.get("restricted_time_to_feasible") for row in rows),
                    "mean_initialization_seconds": _mean(float(row.get("environment_construct_seconds", 0.0)) + float(row.get("reset_wall_seconds", 0.0)) for row in rows),
                    "total_selection_seconds": sum(float(row.get("neighborhood_selection_seconds", 0.0)) for row in rows),
                    "total_pp_seconds": sum(float(row.get("pp_replan_seconds", 0.0)) for row in rows),
                    "total_repair_iterations": repairs,
                    "no_progress_fraction": no_progress / repairs if repairs else 0.0,
                    "mean_normalized_wall_auc": _mean(row.get("normalized_wall_clock_conflict_auc") for row in rows if int(row.get("initial_conflicts", 0)) > 0),
                    "mean_soc_at_feasible": _mean(row.get("budget_final_sum_of_costs") for row in rows if row.get("success")),
                }
            )
    return result


def _paired_summary(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    lns2_ttf = _mean(row["lns2_capped_ttf"] for row in rows)
    v2_ttf = _mean(row["v2_capped_ttf"] for row in rows)
    lns2_auc = _mean(row["lns2_normalized_wall_auc"] for row in rows)
    v2_auc = _mean(row["v2_normalized_wall_auc"] for row in rows)
    soc_rows = [row for row in rows if row["common_success"]]
    lns2_soc = _mean(row["lns2_soc_at_feasible"] for row in soc_rows)
    v2_soc = _mean(row["v2_soc_at_feasible"] for row in soc_rows)
    repairs_lns2 = sum(int(row["lns2_repair_iterations"]) for row in rows)
    repairs_v2 = sum(int(row["v2_repair_iterations"]) for row in rows)
    no_progress_lns2 = sum(
        float(row["lns2_no_progress_fraction"]) * int(row["lns2_repair_iterations"])
        for row in rows
    )
    no_progress_v2 = sum(
        float(row["v2_no_progress_fraction"]) * int(row["v2_repair_iterations"])
        for row in rows
    )
    return {
        "group": label,
        "pair_count": len(rows),
        "lns2_success_count": sum(bool(row["lns2_success"]) for row in rows),
        "v2_success_count": sum(bool(row["v2_success"]) for row in rows),
        "common_success_count": len(soc_rows),
        "lns2_mean_capped_ttf": lns2_ttf,
        "v2_mean_capped_ttf": v2_ttf,
        "v2_ttf_relative_change": _ratio_change(v2_ttf, lns2_ttf),
        "speedup_lns2_over_v2": (
            float(lns2_ttf) / max(1e-15, float(v2_ttf))
            if lns2_ttf is not None and v2_ttf is not None
            else None
        ),
        "lns2_total_selection_seconds": sum(float(row["lns2_selection_seconds"]) for row in rows),
        "v2_total_selection_seconds": sum(float(row["v2_selection_seconds"]) for row in rows),
        "lns2_total_pp_seconds": sum(float(row["lns2_pp_seconds"]) for row in rows),
        "v2_total_pp_seconds": sum(float(row["v2_pp_seconds"]) for row in rows),
        "lns2_total_repairs": repairs_lns2,
        "v2_total_repairs": repairs_v2,
        "lns2_no_progress_fraction": no_progress_lns2 / repairs_lns2 if repairs_lns2 else 0.0,
        "v2_no_progress_fraction": no_progress_v2 / repairs_v2 if repairs_v2 else 0.0,
        "lns2_mean_normalized_wall_auc": lns2_auc,
        "v2_mean_normalized_wall_auc": v2_auc,
        "v2_auc_relative_change": _ratio_change(v2_auc, lns2_auc),
        "lns2_mean_soc_at_feasible": lns2_soc,
        "v2_mean_soc_at_feasible": v2_soc,
        "v2_soc_relative_change": _ratio_change(v2_soc, lns2_soc),
    }


def _per_map_rows(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in paired:
        grouped[str(row["task_id"])].append(row)
    result: list[dict[str, Any]] = []
    for task_id in EXPECTED_TASKS:
        rows = grouped[task_id]
        summary = _paired_summary(rows, task_id)
        summary.update(
            {
                "task_id": task_id,
                "map_id": rows[0]["map_id"],
                "layout_family": rows[0]["layout_family"],
                "agent_count": rows[0]["agent_count"],
                "initial_conflict_min": min(int(row["initial_conflicts"]) for row in rows),
                "initial_conflict_mean": statistics.fmean(float(row["initial_conflicts"]) for row in rows),
                "initial_conflict_max": max(int(row["initial_conflicts"]) for row in rows),
            }
        )
        result.append(summary)
    return result


def _gate_summary(rows: list[dict[str, Any]], label: str, seed_offset: int) -> dict[str, Any]:
    summary = _paired_summary(rows, label)
    ttf_bootstrap = _paired_bootstrap(
        rows,
        baseline_field="lns2_capped_ttf",
        candidate_field="v2_capped_ttf",
        seed=BOOTSTRAP_SEED + seed_offset,
    )
    auc_bootstrap = _paired_bootstrap(
        rows,
        baseline_field="lns2_normalized_wall_auc",
        candidate_field="v2_normalized_wall_auc",
        seed=BOOTSTRAP_SEED + 100 + seed_offset,
    )
    gates = {
        "success_not_lower": summary["v2_success_count"] >= summary["lns2_success_count"],
        "mean_capped_ttf_at_least_5pct_faster": (
            summary["v2_ttf_relative_change"] is not None
            and float(summary["v2_ttf_relative_change"]) <= -0.05
        ),
        "ttf_ci95_upper_not_worse_than_2pct": (
            ttf_bootstrap["ci95_upper"] is not None
            and float(ttf_bootstrap["ci95_upper"]) <= 0.02
        ),
        "auc_ci95_upper_not_worse_than_2pct": (
            auc_bootstrap["ci95_upper"] is not None
            and float(auc_bootstrap["ci95_upper"]) <= 0.02
        ),
        "soc_not_worse_than_2pct": (
            summary["v2_soc_relative_change"] is not None
            and float(summary["v2_soc_relative_change"]) <= 0.02
        ),
    }
    return {
        "summary": summary,
        "ttf_bootstrap": ttf_bootstrap,
        "auc_bootstrap": auc_bootstrap,
        "gates": gates,
        "passed": bool(rows) and all(gates.values()),
    }


def _decision(
    repairable: dict[str, Any], heavy: dict[str, Any], low: dict[str, Any]
) -> str:
    if repairable["passed"]:
        return "v2_wall_clock_supported"
    if heavy["passed"] and not low["passed"]:
        return "v2_heavy_load_only"
    summary = repairable["summary"]
    if (
        summary["v2_success_count"] < summary["lns2_success_count"]
        or summary["v2_ttf_relative_change"] is None
        or float(summary["v2_ttf_relative_change"]) >= 0.0
    ):
        return "lns2_preferred"
    return "inconclusive_keep_v2_full"


def _markdown(report: dict[str, Any], per_map: list[dict[str, Any]]) -> str:
    repairable = report["repairable_evaluation"]
    summary = repairable["summary"]
    lines = [
        "# Seven-map v2/LNS2 wall-clock report",
        "",
        f"Decision: `{report['decision']}`.",
        "",
        "Evidence level: diagnostic seven-map quick cohort, direct 600-second wall-clock budget, no repair limit.",
        "",
        "## Repairable paired episodes",
        "",
        f"Pairs: {summary['pair_count']}; success LNS2/v2: {summary['lns2_success_count']}/{summary['v2_success_count']}.",
        f"Mean capped TTF LNS2/v2: {summary['lns2_mean_capped_ttf']:.6f}s / {summary['v2_mean_capped_ttf']:.6f}s; v2 relative change: {summary['v2_ttf_relative_change']:.2%}.",
        f"Mean normalized wall AUC LNS2/v2: {summary['lns2_mean_normalized_wall_auc']:.9f} / {summary['v2_mean_normalized_wall_auc']:.9f}; v2 relative change: {summary['v2_auc_relative_change']:.2%}.",
        f"Mean feasible SOC LNS2/v2: {summary['lns2_mean_soc_at_feasible']:.3f} / {summary['v2_mean_soc_at_feasible']:.3f}; v2 relative change: {summary['v2_soc_relative_change']:.2%}.",
        "",
        "TTF and AUC confidence intervals use 10,000 paired bootstrap resamples of the ratio of cohort means.",
        "",
        "## Per task",
        "",
        "| Task | Initial conflicts | Success LNS2/v2 | Mean capped TTF LNS2/v2 (s) | LNS2/v2 speedup |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in per_map:
        lines.append(
            "| {task_id} | {initial_conflict_min}/{initial_conflict_mean:.1f}/{initial_conflict_max} | "
            "{lns2_success_count}/{v2_success_count} | {lns2_mean_capped_ttf:.4f}/{v2_mean_capped_ttf:.4f} | {speedup_lns2_over_v2:.3f}x |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Gate results",
            "",
            *[
                f"- {name}: `{passed}`"
                for name, passed in repairable["gates"].items()
            ],
            "",
            "Initially feasible episodes are excluded from repair-quality gates and are reported only as initialization/reset overhead.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_v2_wall_clock_cohort_report(
    sources: Iterable[str | Path], output: str | Path
) -> dict[str, Any]:
    source_paths = [Path(value).resolve() for value in sources]
    if len(source_paths) != 2 or len(set(source_paths)) != 2:
        raise ValueError("seven-map report requires exactly two distinct sources")
    output_path = Path(output).resolve()
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(f"report output directory is not empty: {output_path}")

    identities = [_source_identity(source) for source in source_paths]
    _validate_cross_source_identity(identities)
    source_hashes = [_source_hashes(source) for source in source_paths]

    episodes: list[dict[str, Any]] = []
    source_episode_counts: dict[str, int] = {}
    for source in source_paths:
        roots = _required_collection_roots(source)
        loaded_episodes, _iterations, _metadata = load_track(TRACK, roots)
        episodes.extend(loaded_episodes)
        source_episode_counts[str(source)] = len(loaded_episodes)
    _validate_episodes(episodes)
    paired = _paired_rows(episodes)

    controller_summary = _controller_rows(episodes, paired)
    per_map = _per_map_rows(paired)
    strata: list[dict[str, Any]] = []
    for label in ("initially_feasible", "low_conflict_1_3", "repair_demanding_ge_4"):
        rows = [row for row in paired if row["conflict_stratum"] == label]
        strata.append(_paired_summary(rows, label))
    repairable_rows = [row for row in paired if int(row["initial_conflicts"]) > 0]
    low_rows = [row for row in paired if row["conflict_stratum"] == "low_conflict_1_3"]
    heavy_rows = [row for row in paired if row["conflict_stratum"] == "repair_demanding_ge_4"]
    repairable = _gate_summary(repairable_rows, "repairable", 0)
    low = _gate_summary(low_rows, "low_conflict_1_3", 1)
    heavy = _gate_summary(heavy_rows, "repair_demanding_ge_4", 2)
    decision = _decision(repairable, heavy, low)

    report = {
        "schema": REPORT_SCHEMA,
        "evidence_level": "diagnostic_seven_map_quick",
        "decision": decision,
        "track": TRACK,
        "controllers": list(CONTROLLERS),
        "task_count": len(EXPECTED_TASKS),
        "seed_count": len(EXPECTED_SEEDS),
        "episode_count": len(episodes),
        "paired_episode_count": len(paired),
        "source_episode_counts_after_controller_filter": source_episode_counts,
        "source_identities": identities,
        "source_hashes": source_hashes,
        "validation": {
            "passed": True,
            "source_count": len(source_paths),
            "source_complete": True,
            "source_error_count": 0,
            "trace_validation_passed": True,
            "cross_source_identity_passed": True,
            "paired_initial_fingerprint_mismatch_count": 0,
            "duplicate_episode_count": 0,
            "timing_closure_failure_count": 0,
            "expected_episode_count": 42,
            "observed_episode_count": len(episodes),
        },
        "bootstrap": {
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
            "estimand": "paired bootstrap relative change of cohort means (v2/LNS2 - 1)",
        },
        "repairable_evaluation": repairable,
        "low_conflict_evaluation": low,
        "repair_demanding_evaluation": heavy,
        "next_step": {
            "v2_wall_clock_supported": "full_pool_critical_conflict_oracle_audit",
            "v2_heavy_load_only": "low_conflict_static_bypass_audit",
            "lns2_preferred": "cpp_pp_phase_performance_audit",
            "inconclusive_keep_v2_full": "cpp_pp_phase_performance_audit",
        }[decision],
    }

    output_path.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output_path / "paired_episodes.csv", paired)
    atomic_write_csv(output_path / "controller_summary.csv", controller_summary)
    atomic_write_csv(output_path / "per_map_results.csv", per_map)
    atomic_write_csv(output_path / "conflict_strata.csv", strata)
    _write_json(output_path / "v2_lns2_seven_map_report.json", report)
    (output_path / "v2_lns2_seven_map_report.md").write_text(
        _markdown(report, per_map), encoding="utf-8"
    )
    return report


__all__ = [
    "BOOTSTRAP_SAMPLES",
    "CONTROLLERS",
    "EXPECTED_SEEDS",
    "EXPECTED_TASKS",
    "REPORT_SCHEMA",
    "TRACK",
    "generate_v2_wall_clock_cohort_report",
]
