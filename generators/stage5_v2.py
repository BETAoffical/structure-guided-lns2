from __future__ import annotations

import json
import subprocess
from itertools import permutations
from pathlib import Path
from typing import Any

from .candidate_guided_solver import (
    CandidateGuide,
    run_candidate_guided_instance,
)
from .stage5 import (
    _breakdowns,
    _comparison,
    _read_jsonl,
    _run_baseline,
    _write_json,
    _write_jsonl,
)


def _run_controlled(
    solver: Path,
    instance: Path,
    trace: Path,
    seed: int,
    neighborhood: int,
    iterations: int,
    time_limit_ms: int,
) -> dict[str, Any]:
    trace.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            str(solver),
            "--instance",
            str(instance),
            "--seed",
            str(seed),
            "--neighborhood",
            str(neighborhood),
            "--iterations",
            str(iterations),
            "--time-limit-ms",
            str(time_limit_ms),
            "--trace",
            str(trace),
            "--candidate-mode",
            "controlled",
            "--candidate-count",
            "8",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=max(15.0, time_limit_ms / 1000.0 + 15.0),
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"controlled solver failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    result = json.loads(completed.stdout)
    result["return_code"] = completed.returncode
    return result


def _guided_effectiveness(trace: Path) -> dict[str, int]:
    rows = [
        row
        for row in _read_jsonl(trace)
        if row["event_type"] == "iteration"
    ]
    guided = [row for row in rows if row.get("guidance_used", False)]
    effective = sum(
        row["candidate_valid"]
        and row["accepted"]
        and (
            row["conflicting_pairs_after"]
            < row["conflicting_pairs_before"]
            or (
                row["conflicting_pairs_after"]
                == row["conflicting_pairs_before"]
                and row["sum_of_costs_after"]
                < row["sum_of_costs_before"]
            )
        )
        for row in guided
    )
    changed = sum(
        set(row["neighborhood"])
        != set(row["baseline_neighborhood"])
        for row in guided
    )
    return {
        "guided_iteration_count": len(guided),
        "effective_guided_iteration_count": effective,
        "changed_neighborhood_count": changed,
    }


def _run_with_retry(action, time_limit_ms: int):
    last_error: Exception | None = None
    for _ in range(2):
        try:
            value = action()
            result = value[0] if isinstance(value, tuple) else value
            if (
                float(
                    result.get(
                        "search_runtime_ms", result["runtime_ms"]
                    )
                )
                > time_limit_ms + 1000
            ):
                raise RuntimeError("search exceeded its wall budget")
            return value
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(f"solver failed after retry: {last_error}")


def run_stage5_v2_experiment(
    dataset: str | Path,
    solver: str | Path,
    index: str | Path,
    config: str | Path,
    output: str | Path,
    split: str,
    seeds: list[int],
    neighborhood: int = 6,
    iterations: int = 500,
    time_limit_ms: int = 5000,
) -> dict[str, Any]:
    if split != "test":
        raise ValueError("Stage 5 v2 final experiment reads Test only")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("solver seeds must be non-empty and unique")
    dataset_root = Path(dataset).resolve()
    solver_path = Path(solver).resolve()
    index_root = Path(index).resolve()
    config_path = Path(config).resolve()
    output_root = Path(output).resolve()
    manifest = _read_jsonl(
        dataset_root / split / "manifest.jsonl"
    )
    legacy_rows = []
    controlled_rows = []
    guided_rows = []
    guides = {
        str(row["task_id"]): CandidateGuide(
            dataset_root,
            split,
            str(row["task_id"]),
            index_root,
            config_path,
        )
        for row in manifest
    }
    strategy_permutations = list(
        permutations(
            (
                "legacy_baseline",
                "controlled_baseline",
                "candidate_guided",
            )
        )
    )
    pair_index = 0

    for row in manifest:
        task_id = str(row["task_id"])
        instance = dataset_root / split / row["instance_file"]
        for seed in seeds:
            run_id = f"{task_id}__seed_{seed:04d}"
            strategy_order = strategy_permutations[
                pair_index % len(strategy_permutations)
            ]
            pair_index += 1
            for order_index, strategy in enumerate(strategy_order):
                trace = (
                    output_root
                    / strategy
                    / "traces"
                    / f"{run_id}.jsonl"
                )
                base_row = {
                    "strategy": strategy,
                    "split": split,
                    "task_id": task_id,
                    "map_id": row["map_id"],
                    "layout_mode": row["layout_mode"],
                    "task_variant": row["task_variant"],
                    "solver_seed": seed,
                    "pair_order": order_index,
                    "trace_file": str(trace),
                }
                if strategy == "legacy_baseline":
                    result = _run_with_retry(
                        lambda: _run_baseline(
                            solver_path,
                            instance,
                            trace,
                            seed,
                            neighborhood,
                            iterations,
                            time_limit_ms,
                        ),
                        time_limit_ms,
                    )
                    legacy_rows.append({**base_row, "result": result})
                elif strategy == "controlled_baseline":
                    result = _run_with_retry(
                        lambda: _run_controlled(
                            solver_path,
                            instance,
                            trace,
                            seed,
                            neighborhood,
                            iterations,
                            time_limit_ms,
                        ),
                        time_limit_ms,
                    )
                    controlled_rows.append(
                        {**base_row, "result": result}
                    )
                else:
                    trace.parent.mkdir(parents=True, exist_ok=True)
                    result, decisions = _run_with_retry(
                        lambda: run_candidate_guided_instance(
                            solver_path,
                            instance,
                            trace,
                            guides[task_id],
                            seed,
                            neighborhood,
                            iterations,
                            time_limit_ms,
                        ),
                        time_limit_ms,
                    )
                    guidance_path = (
                        output_root
                        / strategy
                        / "guidance"
                        / f"{run_id}.jsonl"
                    )
                    _write_jsonl(guidance_path, decisions)
                    guided_rows.append(
                        {
                            **base_row,
                            "guidance_file": str(guidance_path),
                            "result": result,
                            **_guided_effectiveness(trace),
                        }
                    )

    _write_jsonl(output_root / "legacy_runs.jsonl", legacy_rows)
    _write_jsonl(
        output_root / "controlled_runs.jsonl", controlled_rows
    )
    _write_jsonl(output_root / "guided_runs.jsonl", guided_rows)
    primary = _comparison(controlled_rows, guided_rows)
    primary["by_layout"] = _breakdowns(
        controlled_rows, guided_rows, "layout_mode"
    )
    primary["by_task_variant"] = _breakdowns(
        controlled_rows, guided_rows, "task_variant"
    )
    summary = {
        "schema_version": 2,
        "split": split,
        "test_data_read": True,
        "task_count": len(manifest),
        "solver_seeds": seeds,
        "total_run_count": (
            len(legacy_rows) + len(controlled_rows) + len(guided_rows)
        ),
        "solver_options": {
            "neighborhood_size": neighborhood,
            "max_iterations": iterations,
            "time_limit_ms": time_limit_ms,
            "candidate_count": 8,
        },
        "primary_comparison_controlled_vs_guided": primary,
        "order_effect_comparison_legacy_vs_controlled": _comparison(
            legacy_rows, controlled_rows
        ),
        "guidance_integrity": {
            "used_count": sum(
                row["guided_iteration_count"] for row in guided_rows
            ),
            "changed_neighborhood_count": sum(
                row["changed_neighborhood_count"]
                for row in guided_rows
            ),
            "effective_count": sum(
                row["effective_guided_iteration_count"]
                for row in guided_rows
            ),
        },
    }
    _write_json(output_root / "experiment_summary.json", summary)
    return summary
