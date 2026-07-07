from __future__ import annotations

import json
import subprocess
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
    candidate_trial_limit_ms: int = 2000,
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

    rows = _read_jsonl(manifest_path)
    trace_root = output_root / split / "traces"
    trace_root.mkdir(parents=True, exist_ok=True)
    collected: list[dict[str, Any]] = []

    solver_seeds = [int(value) for value in (seeds or [seed])]
    if not solver_seeds or len(solver_seeds) != len(set(solver_seeds)):
        raise ValueError("solver seeds must be non-empty and unique")

    for row in rows:
        task_id = str(row["task_id"])
        instance_path = (
            dataset_root / split / str(row["instance_file"])
        ).resolve()
        for solver_seed in solver_seeds:
            trace_path = (
                trace_root / f"{task_id}__seed_{solver_seed:04d}.jsonl"
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
                        "--candidate-trial-limit-ms",
                        str(candidate_trial_limit_ms),
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
                        else max(
                            10.0,
                            time_limit_ms / 1000.0 + 10.0,
                        )
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

            collected.append(
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
                    "trace_file": (
                        str(trace_path) if trace_path.is_file() else None
                    ),
                    "solver_seed": solver_seed,
                    "neighborhood_size": neighborhood,
                    "max_iterations": iterations,
                    "time_limit_ms": time_limit_ms,
                    "candidate_trials": candidate_trials,
                    "candidate_count": (
                        candidate_count if candidate_trials else 0
                    ),
                    "candidate_trial_limit_ms": (
                        candidate_trial_limit_ms
                        if candidate_trials
                        else 0
                    ),
                    "status": status,
                    "return_code": return_code,
                    "result": result,
                    "error": error,
                }
            )

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
            "candidate_trial_limit_ms": (
                candidate_trial_limit_ms
                if candidate_trials
                else 0
            ),
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
    }
    (output_root / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary
