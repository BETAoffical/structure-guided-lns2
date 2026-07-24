#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NATIVE_MODULE_ROOT = PROJECT_ROOT / "build" / "linux" / "project"
if NATIVE_MODULE_ROOT.is_dir() and str(NATIVE_MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(NATIVE_MODULE_ROOT))

from experiments.closed_loop_confirmation import (  # noqa: E402
    run_closed_loop_collection,
)
from experiments.closed_loop_trace_storage import (  # noqa: E402
    TRACE_FORMAT_DELTA_GZIP_V2,
)
from experiments.repair_collection import (  # noqa: E402
    _read_jsonl,
    _utc_now,
    _write_json,
)
from experiments.run_output_guard import prepare_run_output  # noqa: E402
from experiments.v3_s3 import load_v3_s3_bundle  # noqa: E402


DEFAULT_TASKS = (
    "maze-32-32-4__random_05__agents_0100",
    "room-64-64-16__random_05__agents_0400",
    "random-64-64-10__random_04__agents_0600",
    "room-64-64-16__random_04__agents_0600",
)
CONTROLLERS = (
    ("v2-full", "v2-full"),
    ("v3-s3", "v3-s3"),
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _csv_values(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result or len(result) != len(set(result)):
        raise ValueError("comma-separated values must be non-empty and unique")
    return result


def _episode_row(manifest: dict[str, Any], controller: str) -> dict[str, Any]:
    summary = dict(manifest["summary"])
    timings = dict(summary.get("controller_totals") or {})
    sizes = {
        str(name): int(value)
        for name, value in dict(summary.get("selected_size_counts") or {}).items()
    }
    iterations = int(summary["repair_iterations"])
    return {
        "controller": controller,
        "task_id": str(manifest["task_id"]),
        "solver_seed": int(manifest["solver_seed"]),
        "agent_count": int(manifest["agent_count"]),
        "status": str(manifest["status"]),
        "success": bool(summary["success"]),
        "stop_reason": str(summary["stop_reason"]),
        "initial_conflicts": int(summary["initial_conflicts"]),
        "final_conflicts": int(summary["final_conflicts"]),
        "repair_iterations": iterations,
        "wall_time_to_feasible": summary.get("wall_time_to_feasible"),
        "capped_wall_time_to_feasible": float(
            summary["capped_wall_time_to_feasible"]
        ),
        "episode_observed_wall_seconds": float(
            summary["episode_observed_wall_seconds"]
        ),
        "normalized_wall_clock_conflict_auc": summary.get(
            "normalized_wall_clock_conflict_auc"
        ),
        "neighborhood_selection_seconds": float(
            timings.get("neighborhood_selection_seconds", 0.0)
        ),
        "candidate_generation_seconds": float(
            timings.get("candidate_generation_seconds", 0.0)
        ),
        "feature_seconds": float(timings.get("feature_seconds", 0.0)),
        "ranking_inference_seconds": float(
            timings.get("inference_seconds", 0.0)
        ),
        "v3_s3_seconds": float(timings.get("v3_s3_seconds", 0.0)),
        "pp_replan_seconds": float(timings.get("pp_replan_seconds", 0.0)),
        "repair_wall_seconds": float(summary.get("repair_wall_seconds", 0.0)),
        "selection_seconds_per_iteration": (
            float(timings.get("neighborhood_selection_seconds", 0.0))
            / iterations
            if iterations
            else 0.0
        ),
        "pp_seconds_per_iteration": (
            float(timings.get("pp_replan_seconds", 0.0)) / iterations
            if iterations
            else 0.0
        ),
        "size4_count": sizes.get("4", 0),
        "size8_count": sizes.get("8", 0),
        "size16_count": sizes.get("16", 0),
        "other_actual_size_count": sum(
            value for name, value in sizes.items() if name not in {"4", "8", "16"}
        ),
        "v3_s3_planner_calls": int(
            dict(summary.get("v3_s3") or {}).get("planner_call_count", 0)
        ),
        "v3_s3_direct_continuations": int(
            dict(summary.get("v3_s3") or {}).get(
                "direct_continuation_count", 0
            )
        ),
        "v3_s3_deviation_replans": int(
            dict(summary.get("v3_s3") or {}).get(
                "deviation_replan_count", 0
            )
        ),
    }


def _paired_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (str(row["controller"]), str(row["task_id"]), int(row["solver_seed"])): row
        for row in episodes
    }
    pairs = []
    keys = sorted(
        {
            (str(row["task_id"]), int(row["solver_seed"]))
            for row in episodes
        }
    )
    for task_id, seed in keys:
        v2 = indexed[("v2-full", task_id, seed)]
        v3 = indexed[("v3-s3", task_id, seed)]
        v2_time = float(v2["capped_wall_time_to_feasible"])
        v3_time = float(v3["capped_wall_time_to_feasible"])
        pairs.append(
            {
                "task_id": task_id,
                "solver_seed": seed,
                "agent_count": int(v2["agent_count"]),
                "v2_success": bool(v2["success"]),
                "v3_s3_success": bool(v3["success"]),
                "v2_capped_time": v2_time,
                "v3_s3_capped_time": v3_time,
                "v3_s3_time_delta_seconds": v3_time - v2_time,
                "v3_s3_time_change_fraction": (
                    (v3_time - v2_time) / v2_time if v2_time > 0.0 else None
                ),
                "v2_iterations": int(v2["repair_iterations"]),
                "v3_s3_iterations": int(v3["repair_iterations"]),
                "iteration_delta": int(v3["repair_iterations"])
                - int(v2["repair_iterations"]),
                "v2_selection_seconds": float(
                    v2["neighborhood_selection_seconds"]
                ),
                "v3_s3_selection_seconds": float(
                    v3["neighborhood_selection_seconds"]
                ),
                "v2_pp_seconds": float(v2["pp_replan_seconds"]),
                "v3_s3_pp_seconds": float(v3["pp_replan_seconds"]),
                "v2_normalized_wall_auc": v2[
                    "normalized_wall_clock_conflict_auc"
                ],
                "v3_s3_normalized_wall_auc": v3[
                    "normalized_wall_clock_conflict_auc"
                ],
                "v2_final_conflicts": int(v2["final_conflicts"]),
                "v3_s3_final_conflicts": int(v3["final_conflicts"]),
            }
        )
    return pairs


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(
    episodes: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    by_controller = {}
    for controller, _mode in CONTROLLERS:
        rows = [row for row in episodes if row["controller"] == controller]
        by_controller[controller] = {
            "episode_count": len(rows),
            "success_count": sum(bool(row["success"]) for row in rows),
            "mean_capped_wall_time_to_feasible": statistics.fmean(
                float(row["capped_wall_time_to_feasible"]) for row in rows
            ),
            "mean_repair_iterations": statistics.fmean(
                int(row["repair_iterations"]) for row in rows
            ),
            "total_neighborhood_selection_seconds": sum(
                float(row["neighborhood_selection_seconds"]) for row in rows
            ),
            "total_pp_replan_seconds": sum(
                float(row["pp_replan_seconds"]) for row in rows
            ),
            "mean_normalized_wall_clock_conflict_auc": statistics.fmean(
                float(row["normalized_wall_clock_conflict_auc"])
                for row in rows
                if row["normalized_wall_clock_conflict_auc"] is not None
            ),
        }
    common_success = [
        row
        for row in pairs
        if bool(row["v2_success"]) and bool(row["v3_s3_success"])
    ]
    pooled_v2_time = sum(
        float(row["v2_capped_time"]) for row in common_success
    )
    pooled_v3_time = sum(
        float(row["v3_s3_capped_time"]) for row in common_success
    )
    return {
        "schema": "lns2.v3_s3_runtime_comparison.v1",
        "diagnostic_only": True,
        "controllers": by_controller,
        "paired_episode_count": len(pairs),
        "common_success_count": len(common_success),
        "v3_s3_faster_common_success_count": sum(
            float(row["v3_s3_capped_time"]) < float(row["v2_capped_time"])
            for row in common_success
        ),
        "mean_v3_s3_time_change_fraction_common_success": (
            statistics.fmean(
                float(row["v3_s3_time_change_fraction"])
                for row in common_success
            )
            if common_success
            else None
        ),
        "median_v3_s3_time_change_fraction_common_success": (
            statistics.median(
                float(row["v3_s3_time_change_fraction"])
                for row in common_success
            )
            if common_success
            else None
        ),
        "pooled_v3_s3_time_change_fraction_common_success": (
            (pooled_v3_time - pooled_v2_time) / pooled_v2_time
            if pooled_v2_time > 0.0
            else None
        ),
    }


def _markdown(report: dict[str, Any], pairs: list[dict[str, Any]]) -> str:
    aggregate = dict(report["aggregate"])
    controllers = dict(aggregate["controllers"])
    lines = [
        "# v3-S3 vs v2-full real wall-clock diagnostic",
        "",
        "This is a paired diagnostic subset, not a promotion or formal result.",
        "",
        "| Controller | Success | Mean capped time (s) | Mean iterations | Selection total (s) | PP total (s) | Mean normalized wall AUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for controller, _mode in CONTROLLERS:
        row = dict(controllers[controller])
        lines.append(
            f"| {controller} | {row['success_count']}/{row['episode_count']} | "
            f"{row['mean_capped_wall_time_to_feasible']:.6f} | "
            f"{row['mean_repair_iterations']:.3f} | "
            f"{row['total_neighborhood_selection_seconds']:.6f} | "
            f"{row['total_pp_replan_seconds']:.6f} | "
            f"{row['mean_normalized_wall_clock_conflict_auc']:.6f} |"
        )
    pooled = aggregate[
        "pooled_v3_s3_time_change_fraction_common_success"
    ]
    median = aggregate[
        "median_v3_s3_time_change_fraction_common_success"
    ]
    lines.extend(
        [
            "",
            f"- v3-S3 was faster on {aggregate['v3_s3_faster_common_success_count']}/"
            f"{aggregate['common_success_count']} common-success pairs.",
            f"- Pooled completion time change: {float(pooled):+.2%}.",
            f"- Median per-pair completion time change: {float(median):+.2%}.",
        ]
    )
    lines.extend(
        [
            "",
            "| Task | Seed | v2 success/time | v3-S3 success/time | Time change | Iterations v2/v3 | Selection v2/v3 | PP v2/v3 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairs:
        change = row["v3_s3_time_change_fraction"]
        lines.append(
            f"| {row['task_id']} | {row['solver_seed']} | "
            f"{row['v2_success']}/{row['v2_capped_time']:.6f}s | "
            f"{row['v3_s3_success']}/{row['v3_s3_capped_time']:.6f}s | "
            f"{float(change):+.2%} | "
            f"{row['v2_iterations']}/{row['v3_s3_iterations']} | "
            f"{row['v2_selection_seconds']:.6f}/{row['v3_s3_selection_seconds']:.6f} | "
            f"{row['v2_pp_seconds']:.6f}/{row['v3_s3_pp_seconds']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a paired real-wall-clock v2-full/v3-S3 diagnostic."
    )
    parser.add_argument(
        "--dataset", default="build/initlns-movingai-ood-dataset-v1"
    )
    parser.add_argument(
        "--collection-config", default="configs/movingai_ood_collection.json"
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument(
        "--v3-s3-bundle",
        default=(
            "build/initlns-v3-s3-mixed-load-pilot-v5-adaptive/controller"
        ),
    )
    parser.add_argument("--task-ids", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--solver-seeds", default="1")
    parser.add_argument("--wall-clock-seconds", type=float, default=300.0)
    parser.add_argument(
        "--output", default="build/initlns-v3-s3-runtime-diagnostic-v1"
    )
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    if arguments.wall_clock_seconds <= 0.0:
        parser.error("--wall-clock-seconds must be positive")
    try:
        task_ids = list(_csv_values(arguments.task_ids))
        solver_seeds = tuple(
            int(value) for value in _csv_values(arguments.solver_seeds)
        )
    except ValueError as error:
        parser.error(str(error))
    if not solver_seeds or any(seed <= 0 for seed in solver_seeds):
        parser.error("--solver-seeds must contain positive integers")

    dataset = _resolve(arguments.dataset)
    source_config = _resolve(arguments.collection_config)
    controller_bundle = _resolve(arguments.controller_bundle)
    v3_s3_bundle = _resolve(arguments.v3_s3_bundle)
    output = _resolve(arguments.output)
    bundle = load_v3_s3_bundle(v3_s3_bundle)
    identity = {
        "runner": "run_v3_s3_runtime_comparison",
        "schema_version": 1,
        "dataset": str(dataset),
        "collection_config": str(source_config),
        "controller_bundle": str(controller_bundle),
        "v3_s3_bundle": str(v3_s3_bundle),
        "v3_s3_source_fingerprint": str(
            bundle.manifest["source_fingerprint"]
        ),
        "task_ids": task_ids,
        "solver_seeds": list(solver_seeds),
        "wall_clock_seconds": float(arguments.wall_clock_seconds),
        "verification_profile": "deployment",
        "controller_runtime": "optimized",
        "feature_backend": "auto",
        "deterministic_pp_replay": True,
    }
    prepare_run_output(output, resume=arguments.resume, identity=identity)
    config = output / "effective_collection_config.json"
    config_payload = json.loads(source_config.read_text(encoding="utf-8"))
    config_payload["deterministic_pp_replay"] = True
    config_payload["workers"] = 1
    _write_json(config, config_payload)
    status_path = output / "status.json"
    started_at = _utc_now()
    _write_json(
        status_path,
        {
            "schema": "lns2.v3_s3_runtime_comparison_status.v1",
            "status": "running",
            "phase": "qualification",
            "started_at": started_at,
            "updated_at": started_at,
            "completed_episodes": 0,
            "total_episodes": len(task_ids) * len(solver_seeds) * 2,
            "error_episodes": 0,
        },
    )

    roots = {
        name: output / "collections" / name for name, _mode in CONTROLLERS
    }
    cohort = {(task, seed) for task in task_ids for seed in solver_seeds}
    qualification_root: Path | None = None
    for name, mode in CONTROLLERS:
        run_closed_loop_collection(
            dataset,
            config,
            roots[name],
            phase="qualify",
            workers=1,
            resume=arguments.resume or (roots[name] / "run_config.json").is_file(),
            task_ids=task_ids,
            trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
            controller=mode,
            feature_backend="auto",
            controller_runtime="optimized",
            verification_profile="deployment",
            controller_bundle=controller_bundle,
            v3_s3_bundle=v3_s3_bundle if mode == "v3-s3" else None,
            cohort_job_keys=cohort,
            wall_time_budget_seconds=float(arguments.wall_clock_seconds),
            episode_process_timeout_seconds=float(arguments.wall_clock_seconds)
            + 60.0,
            stopping_rule="wall-clock",
            qualification_source=qualification_root,
        )
        if qualification_root is None:
            qualification_root = roots[name]

    schedule = []
    for task_id, seed in sorted(cohort):
        rotation = int(
            hashlib.sha256(f"{task_id}|{seed}".encode()).hexdigest()[:8], 16
        ) % 2
        order = CONTROLLERS[rotation:] + CONTROLLERS[:rotation]
        schedule.append(
            {
                "task_id": task_id,
                "solver_seed": seed,
                "controller_order": [name for name, _mode in order],
            }
        )
    _write_json(
        output / "execution_schedule.json",
        {
            "schema": "lns2.v3_s3_runtime_schedule.v1",
            "method": "sha256-rotated-within-pair",
            "entries": schedule,
        },
    )

    completed = 0
    errors = 0
    for entry in schedule:
        task_id = str(entry["task_id"])
        seed = int(entry["solver_seed"])
        for name in entry["controller_order"]:
            mode = dict(CONTROLLERS)[name]
            run_closed_loop_collection(
                dataset,
                config,
                roots[name],
                phase="realized_dynamic",
                workers=1,
                resume=True,
                task_ids=task_ids,
                trace_format=TRACE_FORMAT_DELTA_GZIP_V2,
                controller=mode,
                feature_backend="auto",
                controller_runtime="optimized",
                verification_profile="deployment",
                controller_bundle=controller_bundle,
                v3_s3_bundle=(
                    v3_s3_bundle if mode == "v3-s3" else None
                ),
                job_keys={(task_id, seed)},
                cohort_job_keys=cohort,
                wall_time_budget_seconds=float(arguments.wall_clock_seconds),
                episode_process_timeout_seconds=float(
                    arguments.wall_clock_seconds
                )
                + 60.0,
                stopping_rule="wall-clock",
            )
            rows = _read_jsonl(
                roots[name] / "realized_dynamic_manifest.jsonl"
            )
            matching = [
                row
                for row in rows
                if str(row.get("task_id")) == task_id
                and int(row.get("solver_seed", -1)) == seed
            ]
            if len(matching) != 1:
                raise RuntimeError(
                    f"{name} did not produce exactly one paired episode"
                )
            completed += 1
            errors += int(
                str(matching[0].get("status")) not in {"ok", "resumed"}
            )
            _write_json(
                status_path,
                {
                    "schema": "lns2.v3_s3_runtime_comparison_status.v1",
                    "status": "running",
                    "phase": "episodes",
                    "started_at": started_at,
                    "updated_at": _utc_now(),
                    "completed_episodes": completed,
                    "total_episodes": len(task_ids) * len(solver_seeds) * 2,
                    "error_episodes": errors,
                },
            )

    if errors:
        _write_json(
            status_path,
            {
                "schema": "lns2.v3_s3_runtime_comparison_status.v1",
                "status": "error",
                "phase": "episode-errors",
                "started_at": started_at,
                "updated_at": _utc_now(),
                "completed_episodes": completed,
                "total_episodes": len(task_ids) * len(solver_seeds) * 2,
                "error_episodes": errors,
                "error": "one or more paired episodes failed",
            },
        )
        return 2

    episodes = []
    for name, _mode in CONTROLLERS:
        manifests = _read_jsonl(
            roots[name] / "realized_dynamic_manifest.jsonl"
        )
        selected = [
            row
            for row in manifests
            if (str(row["task_id"]), int(row["solver_seed"])) in cohort
        ]
        if len(selected) != len(cohort):
            raise RuntimeError(f"{name} paired coverage is incomplete")
        episodes.extend(_episode_row(row, name) for row in selected)
    pairs = _paired_rows(episodes)
    aggregate = _aggregate(episodes, pairs)
    report = {
        "schema": "lns2.v3_s3_runtime_comparison_report.v1",
        "generated_at": _utc_now(),
        "diagnostic_only": True,
        "identity": identity,
        "aggregate": aggregate,
        "episodes": episodes,
        "pairs": pairs,
    }
    _write_csv(output / "episode_runtime.csv", episodes)
    _write_csv(output / "paired_runtime.csv", pairs)
    _write_json(output / "runtime_comparison_report.json", report)
    (output / "runtime_comparison_report.md").write_text(
        _markdown(report, pairs), encoding="utf-8"
    )
    _write_json(
        status_path,
        {
            "schema": "lns2.v3_s3_runtime_comparison_status.v1",
            "status": "complete",
            "phase": "complete",
            "started_at": started_at,
            "updated_at": _utc_now(),
            "completed_episodes": completed,
            "total_episodes": len(task_ids) * len(solver_seeds) * 2,
            "error_episodes": errors,
            "report": str(output / "runtime_comparison_report.md"),
        },
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
