from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

from .candidate_guided_solver import (
    CandidateGuide,
    RankerCandidateGuide,
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
from .stage5_v2 import (
    _guided_effectiveness,
    _run_controlled,
    _run_with_retry,
)


def _strategy_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "used_count": sum(row["guided_iteration_count"] for row in rows),
        "changed_neighborhood_count": sum(
            row["changed_neighborhood_count"] for row in rows
        ),
        "effective_count": sum(
            row["effective_guided_iteration_count"] for row in rows
        ),
    }


def run_stage5_v3_experiment(
    dataset: str | Path,
    solver: str | Path,
    ranker: str | Path,
    ranker_config: str | Path,
    output: str | Path,
    split: str,
    seeds: list[int],
    knn_index: str | Path | None = None,
    knn_config: str | Path | None = None,
    neighborhood: int = 6,
    iterations: int = 500,
    time_limit_ms: int = 5000,
) -> dict[str, Any]:
    if split != "test":
        raise ValueError("Stage 5 v3 final experiment reads Test only")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("solver seeds must be non-empty and unique")
    dataset_root = Path(dataset).resolve()
    solver_path = Path(solver).resolve()
    ranker_root = Path(ranker).resolve()
    ranker_config_path = Path(ranker_config).resolve()
    output_root = Path(output).resolve()
    manifest = _read_jsonl(dataset_root / split / "manifest.jsonl")

    use_knn = knn_index is not None and knn_config is not None
    if (knn_index is None) != (knn_config is None):
        raise ValueError("kNN index and config must be provided together")
    knn_index_root = Path(knn_index).resolve() if knn_index else None
    knn_config_path = Path(knn_config).resolve() if knn_config else None

    legacy_rows = []
    controlled_rows = []
    knn_rows = []
    ranker_rows = []
    knn_guides = {}
    if use_knn:
        assert knn_index_root is not None and knn_config_path is not None
        knn_guides = {
            str(row["task_id"]): CandidateGuide(
                dataset_root,
                split,
                str(row["task_id"]),
                knn_index_root,
                knn_config_path,
            )
            for row in manifest
        }
    ranker_guides = {
        str(row["task_id"]): RankerCandidateGuide(
            dataset_root,
            split,
            str(row["task_id"]),
            ranker_root,
            ranker_config_path,
        )
        for row in manifest
    }

    strategies = [
        "legacy_baseline",
        "controlled_baseline",
        "candidate_guided_ranker",
    ]
    if use_knn:
        strategies.insert(2, "candidate_guided_knn")
    strategy_permutations = list(itertools.permutations(strategies))
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
                    controlled_rows.append({**base_row, "result": result})
                else:
                    trace.parent.mkdir(parents=True, exist_ok=True)
                    guide = (
                        knn_guides[task_id]
                        if strategy == "candidate_guided_knn"
                        else ranker_guides[task_id]
                    )
                    result, decisions = _run_with_retry(
                        lambda: run_candidate_guided_instance(
                            solver_path,
                            instance,
                            trace,
                            guide,
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
                    guided_row = {
                        **base_row,
                        "guidance_file": str(guidance_path),
                        "result": result,
                        **_guided_effectiveness(trace),
                    }
                    if strategy == "candidate_guided_knn":
                        knn_rows.append(guided_row)
                    else:
                        ranker_rows.append(guided_row)

    _write_jsonl(output_root / "legacy_runs.jsonl", legacy_rows)
    _write_jsonl(output_root / "controlled_runs.jsonl", controlled_rows)
    if use_knn:
        _write_jsonl(output_root / "knn_guided_runs.jsonl", knn_rows)
    _write_jsonl(output_root / "ranker_guided_runs.jsonl", ranker_rows)

    primary = _comparison(controlled_rows, ranker_rows)
    primary["by_layout"] = _breakdowns(
        controlled_rows, ranker_rows, "layout_mode"
    )
    primary["by_task_variant"] = _breakdowns(
        controlled_rows, ranker_rows, "task_variant"
    )
    summary = {
        "schema_version": 1,
        "split": split,
        "test_data_read": True,
        "task_count": len(manifest),
        "solver_seeds": seeds,
        "total_run_count": (
            len(legacy_rows)
            + len(controlled_rows)
            + len(knn_rows)
            + len(ranker_rows)
        ),
        "solver_options": {
            "neighborhood_size": neighborhood,
            "max_iterations": iterations,
            "time_limit_ms": time_limit_ms,
            "candidate_count": 8,
        },
        "primary_comparison_controlled_vs_ranker": primary,
        "order_effect_comparison_legacy_vs_controlled": _comparison(
            legacy_rows, controlled_rows
        ),
        "ranker_guidance_integrity": _strategy_summary(ranker_rows),
    }
    if use_knn:
        knn_comparison = _comparison(knn_rows, ranker_rows)
        knn_comparison["by_layout"] = _breakdowns(
            knn_rows, ranker_rows, "layout_mode"
        )
        knn_comparison["by_task_variant"] = _breakdowns(
            knn_rows, ranker_rows, "task_variant"
        )
        summary["secondary_comparison_knn_vs_ranker"] = knn_comparison
        summary["knn_guidance_integrity"] = _strategy_summary(knn_rows)
    _write_json(output_root / "experiment_summary.json", summary)
    return summary
