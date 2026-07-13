from __future__ import annotations

import collections
import json
import math
import random
import subprocess
from pathlib import Path
from typing import Any

from .guided_solver import RepairGuide, run_guided_instance


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_baseline(
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
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=max(15.0, time_limit_ms / 1000.0 + 15.0),
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(
            f"baseline solver failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    result = json.loads(completed.stdout)
    result["return_code"] = completed.returncode
    return result


def _trace_effectiveness(trace: Path) -> dict[str, int]:
    iterations = [
        row
        for row in _read_jsonl(trace)
        if row["event_type"] == "iteration"
    ]
    guided = [row for row in iterations if row.get("guidance_used", False)]
    effective = sum(
        row["accepted"]
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
    return {
        "guided_iteration_count": len(guided),
        "effective_guided_iteration_count": effective,
    }


def _run_guided_with_retry(
    solver: Path,
    instance: Path,
    trace: Path,
    guide: RepairGuide,
    seed: int,
    neighborhood: int,
    iterations: int,
    time_limit_ms: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    last_error: Exception | None = None
    for _ in range(2):
        try:
            result, decisions = run_guided_instance(
                solver,
                instance,
                trace,
                guide,
                seed,
                neighborhood,
                iterations,
                time_limit_ms,
            )
            if result["search_runtime_ms"] > time_limit_ms + 1000:
                raise RuntimeError("guided search exceeded its budget")
            return result, decisions
        except RuntimeError as error:
            last_error = error
    raise RuntimeError(f"guided solver failed after retry: {last_error}")


def _strategy_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    results = [row["result"] for row in rows]
    solved = [result for result in results if result["success"]]
    return {
        "run_count": len(rows),
        "solved_count": len(solved),
        "solved_ratio": round(len(solved) / max(1, len(rows)), 6),
        "final_conflicting_pairs": sum(
            result["final_conflicting_pairs"] for result in results
        ),
        "iterations": sum(result["iterations"] for result in results),
        "accepted_iterations": sum(
            result["accepted_iterations"] for result in results
        ),
        "mean_runtime_ms": round(
            sum(result["runtime_ms"] for result in results)
            / max(1, len(results)),
            6,
        ),
        "mean_search_runtime_ms": round(
            sum(
                result.get("search_runtime_ms", result["runtime_ms"])
                for result in results
            )
            / max(1, len(results)),
            6,
        ),
        "mean_guidance_runtime_ms": round(
            sum(result.get("guidance_runtime_ms", 0.0) for result in results)
            / max(1, len(results)),
            6,
        ),
        "mean_sum_of_costs_solved": (
            round(
                sum(result["sum_of_costs"] for result in solved)
                / len(solved),
                6,
            )
            if solved
            else None
        ),
        "mean_makespan_solved": (
            round(
                sum(result["makespan"] for result in solved) / len(solved),
                6,
            )
            if solved
            else None
        ),
        "guidance_requests": sum(
            result.get("guidance_requests", 0) for result in results
        ),
        "guidance_used": sum(
            result.get("guidance_used", 0) for result in results
        ),
        "guidance_fallbacks": sum(
            result.get("guidance_fallbacks", 0) for result in results
        ),
        "effective_guided_iterations": sum(
            row.get("effective_guided_iteration_count", 0) for row in rows
        ),
    }


def _paired_outcome(
    baseline: dict[str, Any], guided: dict[str, Any]
) -> str:
    left = baseline["result"]
    right = guided["result"]
    if left["success"] != right["success"]:
        return "guided_win" if right["success"] else "baseline_win"
    if left["final_conflicting_pairs"] != right["final_conflicting_pairs"]:
        return (
            "guided_win"
            if right["final_conflicting_pairs"]
            < left["final_conflicting_pairs"]
            else "baseline_win"
        )
    if left["success"] and left["sum_of_costs"] != right["sum_of_costs"]:
        return (
            "guided_win"
            if right["sum_of_costs"] < left["sum_of_costs"]
            else "baseline_win"
        )
    return "tie"


def _bootstrap_interval(
    values: list[float], samples: int = 2000
) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    generator = random.Random(20260706)
    means = []
    for _ in range(samples):
        means.append(
            sum(generator.choice(values) for _ in values) / len(values)
        )
    means.sort()
    low = means[int(0.025 * (samples - 1))]
    high = means[int(0.975 * (samples - 1))]
    return {
        "mean": round(sum(values) / len(values), 6),
        "ci95_low": round(low, 6),
        "ci95_high": round(high, 6),
    }


def _exact_sign_test(first: int, second: int) -> float:
    total = first + second
    if total == 0:
        return 1.0
    tail = sum(
        math.comb(total, index)
        for index in range(min(first, second) + 1)
    ) / (2**total)
    return round(min(1.0, 2.0 * tail), 9)


def _comparison(
    baseline_rows: list[dict[str, Any]],
    guided_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = {
        (row["task_id"], row["solver_seed"]): row
        for row in baseline_rows
    }
    guided = {
        (row["task_id"], row["solver_seed"]): row
        for row in guided_rows
    }
    if set(baseline) != set(guided):
        raise ValueError("baseline and guided runs are not paired")
    outcomes = collections.Counter(
        _paired_outcome(baseline[key], guided[key])
        for key in sorted(baseline)
    )
    ordered_keys = sorted(baseline)
    success_guided_only = sum(
        guided[key]["result"]["success"]
        and not baseline[key]["result"]["success"]
        for key in ordered_keys
    )
    success_baseline_only = sum(
        baseline[key]["result"]["success"]
        and not guided[key]["result"]["success"]
        for key in ordered_keys
    )
    paired_differences = {
        "solved_indicator": [
            float(guided[key]["result"]["success"])
            - float(baseline[key]["result"]["success"])
            for key in ordered_keys
        ],
        "final_conflicting_pairs": [
            guided[key]["result"]["final_conflicting_pairs"]
            - baseline[key]["result"]["final_conflicting_pairs"]
            for key in ordered_keys
        ],
        "iterations": [
            guided[key]["result"]["iterations"]
            - baseline[key]["result"]["iterations"]
            for key in ordered_keys
        ],
        "search_runtime_ms": [
            guided[key]["result"].get(
                "search_runtime_ms", guided[key]["result"]["runtime_ms"]
            )
            - baseline[key]["result"].get(
                "search_runtime_ms", baseline[key]["result"]["runtime_ms"]
            )
            for key in ordered_keys
        ],
        "wall_runtime_ms": [
            guided[key]["result"]["runtime_ms"]
            - baseline[key]["result"]["runtime_ms"]
            for key in ordered_keys
        ],
        "sum_of_costs_both_solved": [
            guided[key]["result"]["sum_of_costs"]
            - baseline[key]["result"]["sum_of_costs"]
            for key in ordered_keys
            if guided[key]["result"]["success"]
            and baseline[key]["result"]["success"]
        ],
    }
    baseline_first = sum(
        baseline[key].get("pair_order", 0)
        < guided[key].get("pair_order", 1)
        for key in ordered_keys
    )
    return {
        "baseline": _strategy_summary(baseline_rows),
        "guided": _strategy_summary(guided_rows),
        "paired_outcomes": {
            key: outcomes.get(key, 0)
            for key in ("guided_win", "baseline_win", "tie")
        },
        "paired_statistics": {
            "difference_definition": "guided_minus_baseline",
            "bootstrap_95_percent": {
                name: _bootstrap_interval(
                    [float(value) for value in values]
                )
                for name, values in paired_differences.items()
            },
            "paired_outcome_sign_test_p": _exact_sign_test(
                outcomes.get("guided_win", 0),
                outcomes.get("baseline_win", 0),
            ),
            "success_mcnemar_exact_p": _exact_sign_test(
                success_guided_only, success_baseline_only
            ),
            "success_discordant_pairs": {
                "guided_only": success_guided_only,
                "baseline_only": success_baseline_only,
            },
            "execution_order": {
                "baseline_first": baseline_first,
                "guided_first": len(ordered_keys) - baseline_first,
            },
        },
    }


def _breakdowns(
    baseline_rows: list[dict[str, Any]],
    guided_rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    values = sorted({str(row[field]) for row in baseline_rows})
    return {
        value: _comparison(
            [
                row
                for row in baseline_rows
                if str(row[field]) == value
            ],
            [
                row
                for row in guided_rows
                if str(row[field]) == value
            ],
        )
        for value in values
    }


def _threshold_key(value: float) -> str:
    return f"{value:.3f}".replace(".", "_")


def run_stage5_experiment(
    dataset: str | Path,
    solver: str | Path,
    index: str | Path,
    evaluation: str | Path,
    output: str | Path,
    split: str,
    seeds: list[int],
    thresholds: list[float] | None = None,
    config: str | Path | None = None,
    neighborhood: int = 6,
    iterations: int = 500,
    time_limit_ms: int = 5000,
) -> dict[str, Any]:
    if split not in {"validation", "test"}:
        raise ValueError("Stage 5 supports only validation or test")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("solver seeds must be non-empty and unique")
    if split == "validation":
        threshold_values = thresholds or [0.5, 0.67, 0.8]
        role_weight = 0.5
        conflict_weight = 0.35
        baseline_weight = 0.15
        if config is not None:
            raise ValueError("Validation selects config; it does not load one")
    else:
        if config is None:
            raise ValueError("Test requires a frozen Validation config")
        frozen = _read_json(Path(config).resolve())
        if (
            frozen.get("selected_on_split") != "validation"
            or frozen.get("test_data_read", True)
        ):
            raise ValueError("Test config is not a frozen Validation config")
        threshold_values = [float(frozen["effective_threshold"])]
        role_weight = float(frozen["role_weight"])
        conflict_weight = float(frozen["conflict_weight"])
        baseline_weight = float(frozen["baseline_weight"])

    dataset_root = Path(dataset).resolve()
    solver_path = Path(solver).resolve()
    index_root = Path(index).resolve()
    evaluation_root = Path(evaluation).resolve()
    output_root = Path(output).resolve()
    manifest = _read_jsonl(dataset_root / split / "manifest.jsonl")
    baseline_rows = []
    guided_by_threshold: dict[float, list[dict[str, Any]]] = {
        threshold: [] for threshold in threshold_values
    }
    guides: dict[tuple[float, str], RepairGuide] = {}
    for row in manifest:
        task_id = str(row["task_id"])
        instance = dataset_root / split / row["instance_file"]
        for threshold in threshold_values:
            guides[(threshold, task_id)] = RepairGuide(
                dataset_root,
                split,
                task_id,
                index_root,
                evaluation_root,
                effective_threshold=threshold,
                role_weight=role_weight,
                conflict_weight=conflict_weight,
                baseline_weight=baseline_weight,
            )
        for seed in seeds:
            run_id = f"{task_id}__seed_{seed:04d}"
            strategies: list[float | None] = [None, *threshold_values]
            rotation = (
                sum(task_id.encode("utf-8")) + seed
            ) % len(strategies)
            strategies = strategies[rotation:] + strategies[:rotation]
            for order_index, threshold in enumerate(strategies):
                if threshold is None:
                    trace = (
                        output_root
                        / "baseline"
                        / "traces"
                        / f"{run_id}.jsonl"
                    )
                    result = _run_baseline(
                        solver_path,
                        instance,
                        trace,
                        seed,
                        neighborhood,
                        iterations,
                        time_limit_ms,
                    )
                    baseline_rows.append(
                        {
                            "strategy": "baseline_lns2",
                            "split": split,
                            "task_id": task_id,
                            "map_id": row["map_id"],
                            "layout_mode": row["layout_mode"],
                            "task_variant": row["task_variant"],
                            "solver_seed": seed,
                            "pair_order": order_index,
                            "trace_file": str(trace),
                            "result": result,
                        }
                    )
                    continue

                threshold_key = _threshold_key(threshold)
                trace = (
                    output_root
                    / f"guided_{threshold_key}"
                    / "traces"
                    / f"{run_id}.jsonl"
                )
                trace.parent.mkdir(parents=True, exist_ok=True)
                result, decisions = _run_guided_with_retry(
                    solver_path,
                    instance,
                    trace,
                    guides[(threshold, task_id)],
                    seed,
                    neighborhood,
                    iterations,
                    time_limit_ms,
                )
                decision_path = (
                    output_root
                    / f"guided_{threshold_key}"
                    / "guidance"
                    / f"{run_id}.jsonl"
                )
                _write_jsonl(decision_path, decisions)
                effectiveness = _trace_effectiveness(trace)
                guided_by_threshold[threshold].append(
                    {
                        "strategy": "guided_lns2",
                        "split": split,
                        "task_id": task_id,
                        "map_id": row["map_id"],
                        "layout_mode": row["layout_mode"],
                        "task_variant": row["task_variant"],
                        "solver_seed": seed,
                        "pair_order": order_index,
                        "effective_threshold": threshold,
                        "trace_file": str(trace),
                        "guidance_file": str(decision_path),
                        "result": result,
                        **effectiveness,
                    }
                )

    comparisons = {}
    all_guided_rows = []
    for threshold in threshold_values:
        guided_rows = guided_by_threshold[threshold]
        all_guided_rows.extend(guided_rows)
        comparison = _comparison(baseline_rows, guided_rows)
        comparison["by_layout"] = _breakdowns(
            baseline_rows, guided_rows, "layout_mode"
        )
        comparison["by_task_variant"] = _breakdowns(
            baseline_rows, guided_rows, "task_variant"
        )
        comparisons[str(threshold)] = comparison

    _write_jsonl(output_root / "baseline_runs.jsonl", baseline_rows)
    _write_jsonl(output_root / "guided_runs.jsonl", all_guided_rows)
    selected_threshold = max(
        threshold_values,
        key=lambda threshold: (
            comparisons[str(threshold)]["guided"]["solved_count"],
            -comparisons[str(threshold)]["guided"][
                "final_conflicting_pairs"
            ],
            comparisons[str(threshold)]["paired_outcomes"]["guided_win"]
            - comparisons[str(threshold)]["paired_outcomes"][
                "baseline_win"
            ],
            -threshold,
        ),
    )
    summary = {
        "schema_version": 1,
        "split": split,
        "task_count": len(manifest),
        "solver_seeds": seeds,
        "solver_options": {
            "neighborhood_size": neighborhood,
            "max_iterations": iterations,
            "time_limit_ms": time_limit_ms,
        },
        "threshold_comparisons": comparisons,
        "selected_threshold": selected_threshold,
        "test_data_read": split == "test",
    }
    _write_json(output_root / "experiment_summary.json", summary)
    if split == "validation":
        selected_config = {
            "schema_version": 1,
            "selected_on_split": "validation",
            "test_data_read": False,
            "effective_threshold": selected_threshold,
            "role_weight": role_weight,
            "conflict_weight": conflict_weight,
            "baseline_weight": baseline_weight,
            "neighborhood_size": neighborhood,
            "max_iterations": iterations,
            "time_limit_ms": time_limit_ms,
        }
        _write_json(output_root / "selected_config.json", selected_config)
    return summary


def analyze_stage5_results(output: str | Path) -> dict[str, Any]:
    output_root = Path(output).resolve()
    summary = _read_json(output_root / "experiment_summary.json")
    baseline_rows = _read_jsonl(output_root / "baseline_runs.jsonl")
    guided_rows = _read_jsonl(output_root / "guided_runs.jsonl")
    comparisons = {}
    thresholds = sorted(
        {float(row["effective_threshold"]) for row in guided_rows}
    )
    for threshold in thresholds:
        selected = [
            row
            for row in guided_rows
            if float(row["effective_threshold"]) == threshold
        ]
        comparison = _comparison(baseline_rows, selected)
        comparison["by_layout"] = _breakdowns(
            baseline_rows, selected, "layout_mode"
        )
        comparison["by_task_variant"] = _breakdowns(
            baseline_rows, selected, "task_variant"
        )
        comparisons[str(threshold)] = comparison
    summary["threshold_comparisons"] = comparisons
    _write_json(output_root / "experiment_summary.json", summary)
    return summary
