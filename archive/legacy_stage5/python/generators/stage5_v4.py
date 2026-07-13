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


def _integrity(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "used_count": sum(row["guided_iteration_count"] for row in rows),
        "changed_neighborhood_count": sum(
            row["changed_neighborhood_count"] for row in rows
        ),
        "effective_count": sum(
            row["effective_guided_iteration_count"] for row in rows
        ),
    }


def _comparison_with_breakdowns(
    baseline: list[dict[str, Any]],
    guided: list[dict[str, Any]],
) -> dict[str, Any]:
    result = _comparison(baseline, guided)
    result["by_layout"] = _breakdowns(baseline, guided, "layout_mode")
    result["by_task_variant"] = _breakdowns(
        baseline, guided, "task_variant"
    )
    return result


def run_stage5_v4_experiment(
    dataset: str | Path,
    solver: str | Path,
    rollout_ranker: str | Path,
    rollout_config: str | Path,
    output: str | Path,
    split: str,
    seeds: list[int],
    knn_index: str | Path | None = None,
    knn_config: str | Path | None = None,
    v3_ranker: str | Path | None = None,
    v3_config: str | Path | None = None,
    neighborhood: int = 6,
    iterations: int = 500,
    time_limit_ms: int = 5000,
    candidate_generator_profile: str = "core5",
) -> dict[str, Any]:
    if split != "test":
        raise ValueError("Stage 5 v4 final experiment reads Test only")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("solver seeds must be non-empty and unique")
    if (knn_index is None) != (knn_config is None):
        raise ValueError("kNN index and config must be provided together")
    if (v3_ranker is None) != (v3_config is None):
        raise ValueError("v3 ranker and config must be provided together")
    if candidate_generator_profile not in {"core5", "full8"}:
        raise ValueError("candidate generator profile must be core5 or full8")
    candidate_count = 5 if candidate_generator_profile == "core5" else 8

    dataset_root = Path(dataset).resolve()
    solver_path = Path(solver).resolve()
    output_root = Path(output).resolve()
    manifest = _read_jsonl(dataset_root / split / "manifest.jsonl")

    strategies = [
        "legacy_baseline",
        "controlled_baseline",
        "rollout_guided_v4",
    ]
    if knn_index is not None:
        strategies.insert(2, "candidate_guided_knn")
    if v3_ranker is not None:
        strategies.insert(-1, "candidate_guided_ranker_v3")

    knn_guides = {}
    if knn_index is not None and knn_config is not None:
        knn_guides = {
            str(row["task_id"]): CandidateGuide(
                dataset_root,
                split,
                str(row["task_id"]),
                Path(knn_index).resolve(),
                Path(knn_config).resolve(),
            )
            for row in manifest
        }
    v3_guides = {}
    if v3_ranker is not None and v3_config is not None:
        v3_guides = {
            str(row["task_id"]): RankerCandidateGuide(
                dataset_root,
                split,
                str(row["task_id"]),
                Path(v3_ranker).resolve(),
                Path(v3_config).resolve(),
            )
            for row in manifest
        }
    rollout_guides = {
        str(row["task_id"]): RankerCandidateGuide(
            dataset_root,
            split,
            str(row["task_id"]),
            Path(rollout_ranker).resolve(),
            Path(rollout_config).resolve(),
        )
        for row in manifest
    }

    rows_by_strategy: dict[str, list[dict[str, Any]]] = {
        strategy: [] for strategy in strategies
    }
    permutations = list(itertools.permutations(strategies))
    pair_index = 0
    for row in manifest:
        task_id = str(row["task_id"])
        instance = dataset_root / split / row["instance_file"]
        for seed in seeds:
            run_id = f"{task_id}__seed_{seed:04d}"
            strategy_order = permutations[pair_index % len(permutations)]
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
                    rows_by_strategy[strategy].append(
                        {**base_row, "result": result}
                    )
                    continue
                if strategy == "controlled_baseline":
                    result = _run_with_retry(
                        lambda: _run_controlled(
                            solver_path,
                            instance,
                            trace,
                            seed,
                            neighborhood,
                            iterations,
                            time_limit_ms,
                            candidate_generator_profile,
                        ),
                        time_limit_ms,
                    )
                    rows_by_strategy[strategy].append(
                        {**base_row, "result": result}
                    )
                    continue
                if strategy == "candidate_guided_knn":
                    guide = knn_guides[task_id]
                elif strategy == "candidate_guided_ranker_v3":
                    guide = v3_guides[task_id]
                else:
                    guide = rollout_guides[task_id]
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
                        candidate_generator_profile,
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
                rows_by_strategy[strategy].append(
                    {
                        **base_row,
                        "guidance_file": str(guidance_path),
                        "result": result,
                        **_guided_effectiveness(trace),
                    }
                )

    for strategy, rows in rows_by_strategy.items():
        _write_jsonl(output_root / f"{strategy}_runs.jsonl", rows)
    controlled = rows_by_strategy["controlled_baseline"]
    rollout = rows_by_strategy["rollout_guided_v4"]
    summary = {
        "schema_version": 1,
        "split": split,
        "test_data_read": True,
        "task_count": len(manifest),
        "solver_seeds": seeds,
        "total_run_count": sum(len(rows) for rows in rows_by_strategy.values()),
        "solver_options": {
            "neighborhood_size": neighborhood,
            "max_iterations": iterations,
            "time_limit_ms": time_limit_ms,
            "candidate_count": candidate_count,
            "candidate_generator_profile": candidate_generator_profile,
        },
        "primary_comparison_controlled_vs_rollout": (
            _comparison_with_breakdowns(controlled, rollout)
        ),
        "order_effect_comparison_legacy_vs_controlled": _comparison(
            rows_by_strategy["legacy_baseline"], controlled
        ),
        "rollout_guidance_integrity": _integrity(rollout),
    }
    if "candidate_guided_knn" in rows_by_strategy:
        summary["secondary_comparison_knn_vs_rollout"] = (
            _comparison_with_breakdowns(
                rows_by_strategy["candidate_guided_knn"], rollout
            )
        )
        summary["knn_guidance_integrity"] = _integrity(
            rows_by_strategy["candidate_guided_knn"]
        )
    if "candidate_guided_ranker_v3" in rows_by_strategy:
        summary["secondary_comparison_v3_ranker_vs_rollout"] = (
            _comparison_with_breakdowns(
                rows_by_strategy["candidate_guided_ranker_v3"], rollout
            )
        )
        summary["v3_ranker_guidance_integrity"] = _integrity(
            rows_by_strategy["candidate_guided_ranker_v3"]
        )
    _write_json(output_root / "experiment_summary.json", summary)
    return summary
