from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _result_from_trace(
    path: Path,
    expected_candidate_generator_profile: str | None = None,
    expected_replan_order_seeds: list[int] | None = None,
    expected_rollout_horizons: list[int] | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    rows = _read_jsonl(path)
    if not rows or rows[-1].get("event_type") != "summary":
        return None
    summary = rows[-1]
    if expected_replan_order_seeds is not None and [
        int(value)
        for value in summary.get("candidate_replan_order_seeds", [])
    ] != expected_replan_order_seeds:
        return None
    if (
        expected_candidate_generator_profile is not None
        and str(
            summary.get("candidate_generator_profile", "full8")
        )
        != expected_candidate_generator_profile
    ):
        return None
    if expected_rollout_horizons is not None and [
        int(value)
        for value in summary.get("candidate_rollout_horizons", [])
    ] != expected_rollout_horizons:
        return None
    required = {
        "success",
        "initial_conflicting_pairs",
        "final_conflicting_pairs",
        "iterations",
        "accepted_iterations",
        "makespan",
        "sum_of_costs",
        "runtime_ms",
        "search_runtime_ms",
        "guidance_runtime_ms",
        "counterfactual_runtime_ms",
        "guidance_requests",
        "guidance_used",
        "guidance_fallbacks",
    }
    if not required.issubset(summary):
        return None
    return {key: summary[key] for key in sorted(required)}


def collect_experience(
    dataset: str | Path,
    solver: str | Path,
    output: str | Path,
    split: str = "train",
    seed: int = 1234,
    seeds: list[int] | None = None,
    neighborhood: int = 6,
    iterations: int = 500,
    time_limit_ms: int = 3000,
    candidate_trials: bool = False,
    candidate_count: int = 8,
    candidate_generator_profile: str = "full8",
    candidate_trial_limit_ms: int = 2000,
    candidate_replan_order_seeds: list[int] | None = None,
    candidate_rollout_horizons: list[int] | None = None,
    layout_modes: list[str] | None = None,
    task_variants: list[str] | None = None,
    max_runs: int | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    dataset_root = Path(dataset).resolve()
    solver_path = Path(solver).resolve()
    output_root = Path(output).resolve()
    manifest_path = dataset_root / split / "manifest.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest does not exist: {manifest_path}")
    if not solver_path.is_file():
        raise ValueError(f"solver does not exist: {solver_path}")
    if (
        neighborhood <= 0
        or iterations < 0
        or time_limit_ms <= 0
        or candidate_count <= 0
        or candidate_trial_limit_ms <= 0
    ):
        raise ValueError("invalid solver limits")

    if candidate_generator_profile not in {"full8", "core5"}:
        raise ValueError("candidate generator profile must be full8 or core5")
    if candidate_trials:
        candidate_count = 5 if candidate_generator_profile == "core5" else 8
    workers = max(1, int(workers))
    if max_runs is not None and max_runs <= 0:
        raise ValueError("max_runs must be positive")

    rows = _read_jsonl(manifest_path)
    allowed_layout_modes = (
        {str(value) for value in layout_modes} if layout_modes else None
    )
    allowed_task_variants = (
        {str(value) for value in task_variants} if task_variants else None
    )
    if allowed_layout_modes is not None:
        rows = [
            row
            for row in rows
            if str(row["layout_mode"]) in allowed_layout_modes
        ]
    if allowed_task_variants is not None:
        rows = [
            row
            for row in rows
            if str(row.get("task_variant", row["scenario_type"]))
            in allowed_task_variants
        ]
    trace_root = output_root / split / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    collected: list[dict[str, Any]] = []

    solver_seeds = [int(value) for value in (seeds or [seed])]
    if not solver_seeds or len(solver_seeds) != len(set(solver_seeds)):
        raise ValueError("solver seeds must be non-empty and unique")
    replan_order_seeds = [
        int(value) for value in (candidate_replan_order_seeds or [0])
    ]
    if (
        not replan_order_seeds
        or len(replan_order_seeds) != len(set(replan_order_seeds))
    ):
        raise ValueError("candidate replan order seeds must be unique")
    rollout_horizons = sorted(
        set(int(value) for value in (candidate_rollout_horizons or []))
    )
    if any(value <= 0 for value in rollout_horizons):
        raise ValueError("candidate rollout horizons must be positive")

    planned_runs: list[tuple[int, dict[str, Any], int]] = []
    for row_index, row in enumerate(rows):
        for solver_seed in solver_seeds:
            planned_runs.append((row_index, row, solver_seed))
    if max_runs is not None:
        planned_runs = planned_runs[:max_runs]

    def collect_one(
        job: tuple[int, dict[str, Any], int]
    ) -> tuple[int, dict[str, Any]]:
        row_index, row, solver_seed = job
        task_id = str(row["task_id"])
        instance_path = (
            dataset_root / split / str(row["instance_file"])
        ).resolve()
        trace_path = (
            trace_root / f"{task_id}__seed_{solver_seed:04d}.jsonl"
        )
        result = _result_from_trace(
            trace_path,
            (
                candidate_generator_profile
                if candidate_trials
                else None
            ),
            replan_order_seeds if candidate_trials else None,
            rollout_horizons if candidate_trials else None,
        )
        if result is not None:
            status = "solved" if result["success"] else "unsolved"
            error = None
            return_code = 0 if result["success"] else 1
            return (
                row_index,
                {
                    "schema_version": 1,
                    "task_id": task_id,
                    "map_id": row["map_id"],
                    "split": split,
                    "layout_mode": row["layout_mode"],
                    "layout_variant": row.get("layout_variant"),
                    "scenario_type": row["scenario_type"],
                    "task_variant": row.get("task_variant"),
                    "instance_file": str(instance_path),
                    "trace_file": str(trace_path),
                    "solver_seed": solver_seed,
                    "neighborhood_size": neighborhood,
                    "max_iterations": iterations,
                    "time_limit_ms": time_limit_ms,
                    "candidate_trials": candidate_trials,
                    "candidate_count": (
                        candidate_count if candidate_trials else 0
                    ),
                    "candidate_generator_profile": (
                        candidate_generator_profile
                        if candidate_trials
                        else None
                    ),
                    "candidate_trial_limit_ms": (
                        candidate_trial_limit_ms
                        if candidate_trials
                        else 0
                    ),
                    "candidate_replan_order_seeds": (
                        replan_order_seeds if candidate_trials else []
                    ),
                    "candidate_rollout_horizons": (
                        rollout_horizons if candidate_trials else []
                    ),
                    "status": status,
                    "return_code": return_code,
                    "result": result,
                    "error": error,
                },
            )

        command = [
            str(solver_path),
            "--instance",
            str(instance_path),
            "--seed",
            str(solver_seed),
            "--neighborhood",
            str(neighborhood),
            "--iterations",
            str(iterations),
            "--time-limit-ms",
            str(time_limit_ms),
            "--trace",
            str(trace_path),
        ]
        if candidate_trials:
            command.extend(
                [
                    "--candidate-mode",
                    "collect",
                    "--candidate-count",
                    str(candidate_count),
                    "--candidate-generator-profile",
                    candidate_generator_profile,
                    "--candidate-trial-limit-ms",
                    str(candidate_trial_limit_ms),
                    "--candidate-replan-order-seeds",
                    ",".join(str(value) for value in replan_order_seeds),
                ]
            )
            if rollout_horizons:
                command.extend(
                    [
                        "--candidate-rollout-horizons",
                        ",".join(str(value) for value in rollout_horizons),
                    ]
                )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=(
                    max(300.0, time_limit_ms / 1000.0 + 30.0)
                    if candidate_trials
                    else max(10.0, time_limit_ms / 1000.0 + 10.0)
                ),
            )
            result = None
            if completed.stdout.strip():
                try:
                    result = json.loads(completed.stdout)
                except json.JSONDecodeError:
                    result = None
            valid_run = (
                completed.returncode in {0, 1}
                and trace_path.is_file()
                and result is not None
            )
            status = (
                "solved"
                if valid_run and completed.returncode == 0
                else "unsolved"
                if valid_run
                else "error"
            )
            error = (
                None
                if valid_run
                else completed.stderr.strip()
                or "solver did not produce valid JSON and a trace"
            )
            return_code = completed.returncode
        except subprocess.TimeoutExpired as error_value:
            status = "error"
            error = f"collector timeout: {error_value}"
            return_code = None
            result = None

        return (
            row_index,
            {
                "schema_version": 1,
                "task_id": task_id,
                "map_id": row["map_id"],
                "split": split,
                "layout_mode": row["layout_mode"],
                "layout_variant": row.get("layout_variant"),
                "scenario_type": row["scenario_type"],
                "task_variant": row.get("task_variant"),
                "instance_file": str(instance_path),
                "trace_file": str(trace_path) if trace_path.is_file() else None,
                "solver_seed": solver_seed,
                "neighborhood_size": neighborhood,
                "max_iterations": iterations,
                "time_limit_ms": time_limit_ms,
                "candidate_trials": candidate_trials,
                "candidate_count": (
                    candidate_count if candidate_trials else 0
                ),
                "candidate_generator_profile": (
                    candidate_generator_profile if candidate_trials else None
                ),
                "candidate_trial_limit_ms": (
                    candidate_trial_limit_ms if candidate_trials else 0
                ),
                "candidate_replan_order_seeds": (
                    replan_order_seeds if candidate_trials else []
                ),
                "candidate_rollout_horizons": (
                    rollout_horizons if candidate_trials else []
                ),
                "status": status,
                "return_code": return_code,
                "result": result,
                "error": error,
            },
        )

    collected_with_index: list[tuple[int, dict[str, Any]]] = []
    if workers == 1:
        for job in planned_runs:
            collected_with_index.append(collect_one(job))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(collect_one, job) for job in planned_runs]
            for future in as_completed(futures):
                collected_with_index.append(future.result())
    collected = [
        row
        for _, row in sorted(
            collected_with_index,
            key=lambda item: (
                item[0],
                int(item[1]["solver_seed"]),
                str(item[1]["task_id"]),
            ),
        )
    ]

    _write_jsonl(output_root / "collection_manifest.jsonl", collected)
    valid_results = [
        row["result"] for row in collected if row["result"] is not None
    ]
    conflict_run_count = sum(
        result["initial_conflicting_pairs"] > 0
        for result in valid_results
    )
    unsolved_count = sum(
        row["status"] == "unsolved" for row in collected
    )
    summary = {
        "schema_version": 1,
        "dataset": str(dataset_root),
        "solver": str(solver_path),
        "split": split,
        "run_count": len(collected),
        "solved_count": sum(
            row["status"] == "solved" for row in collected
        ),
        "unsolved_count": unsolved_count,
        "error_count": sum(
            row["status"] == "error" for row in collected
        ),
        "solver_options": {
            "seeds": solver_seeds,
            "neighborhood_size": neighborhood,
            "max_iterations": iterations,
            "time_limit_ms": time_limit_ms,
            "candidate_trials": candidate_trials,
            "candidate_count": (
                candidate_count if candidate_trials else 0
            ),
            "candidate_generator_profile": (
                candidate_generator_profile if candidate_trials else None
            ),
            "candidate_trial_limit_ms": (
                candidate_trial_limit_ms
                if candidate_trials
                else 0
            ),
            "candidate_replan_order_seeds": (
                replan_order_seeds if candidate_trials else []
            ),
            "candidate_rollout_horizons": (
                rollout_horizons if candidate_trials else []
            ),
            "layout_modes": sorted(allowed_layout_modes or []),
            "task_variants": sorted(allowed_task_variants or []),
            "max_runs": max_runs,
            "workers": workers,
        },
        "pressure_metrics": {
            "initial_conflict_run_count": conflict_run_count,
            "initial_conflict_run_ratio": round(
                conflict_run_count / max(1, len(valid_results)), 6
            ),
            "initial_conflicting_pairs": sum(
                result["initial_conflicting_pairs"]
                for result in valid_results
            ),
            "lns_iterations": sum(
                result["iterations"] for result in valid_results
            ),
            "unsolved_ratio": round(
                unsolved_count / max(1, len(collected)), 6
            ),
        },
        "cost_estimate": {
            "planned_run_count": len(planned_runs),
            "informative_state_count": sum(
                result["iterations"] for result in valid_results
            ),
            "candidate_trial_count": (
                sum(result["iterations"] for result in valid_results)
                * (candidate_count if candidate_trials else 0)
                * (len(replan_order_seeds) if candidate_trials else 0)
            ),
            "rollout_label_count": (
                sum(result["iterations"] for result in valid_results)
                * (candidate_count if candidate_trials else 0)
                * (len(replan_order_seeds) if candidate_trials else 0)
                * len(rollout_horizons)
            ),
        },
    }
    (output_root / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
