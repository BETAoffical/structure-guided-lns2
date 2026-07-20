from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.closed_loop_confirmation import (  # noqa: E402
    generate_online_candidates,
    score_online_candidates,
)
from experiments.compact_controller_model import load_controller_bundle  # noqa: E402
from experiments.online_feature_engine import OnlineFeatureEngine  # noqa: E402
from experiments.repair_collection import (  # noqa: E402
    _load_dataset_rows,
    _make_environment,
    _plain,
    _read_json,
    _write_json,
    state_fingerprint,
)


DEFAULT_TASKS = (
    "random-32-32-10__random_05__agents_0200",
    "maze-32-32-4__random_05__agents_0100",
    "room-64-64-16__random_05__agents_0400",
    "warehouse-10-20-10-2-2__random_05__agents_0500",
    "den312d__random_05__agents_0200",
    "maze-128-128-1__random_04__agents_0600",
    "room-64-64-16__random_04__agents_0600",
    "random-64-64-10__random_04__agents_0600",
)


def _median(values: list[float]) -> float:
    return statistics.median(values)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _measure(
    function: Callable[[], Any], *, warmup: int, repeats: int
) -> tuple[Any, list[float]]:
    result = None
    for _ in range(warmup):
        result = function()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        values.append(time.perf_counter() - started)
    return result, values


def _dense_mapping(row: dict[str, Any]) -> dict[str, float]:
    return dict(
        zip(
            map(str, row["feature_names"]),
            map(float, row["feature_values"]),
        )
    )


def benchmark_case(
    *,
    dataset: Path,
    row: dict[str, Any],
    config: dict[str, Any],
    controller_bundle: Any,
    solver_seed: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    environment = _make_environment(
        dataset, row, config["environment"], "Adaptive"
    )
    state = _plain(environment.reset(seed=solver_seed))
    if bool(state["done"]) or not state.get("conflict_edges"):
        return {
            "task_id": row["task_id"],
            "solver_seed": solver_seed,
            "skipped": True,
            "reason": "initial_state_not_repairable",
        }
    state_hash = state_fingerprint(state)
    common = {
        "task_id": str(row["task_id"]),
        "solver_seed": solver_seed,
        "decision_index": 0,
        "proposal_config": config["proposal"],
        "state_hash": state_hash,
        "verify_full_state": False,
    }

    def proposals(runtime: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return generate_online_candidates(
            environment,
            state,
            proposal_backend=runtime,
            shadow_validation=False,
            **common,
        )

    reference_result, reference_proposal_times = _measure(
        lambda: proposals("reference"), warmup=warmup, repeats=repeats
    )
    optimized_result, optimized_proposal_times = _measure(
        lambda: proposals("optimized"), warmup=warmup, repeats=repeats
    )
    reference_candidates = reference_result[0]
    optimized_candidates = optimized_result[0]
    if reference_candidates != optimized_candidates:
        raise RuntimeError("reference and compact candidate pools differ")

    model = controller_bundle.main_models["realized_dynamic"]
    legacy_score_model = SimpleNamespace(
        profile=model.profile,
        feature_names=model.feature_names,
        pair_vector=model.pair_vector,
        predict_positive=model.predict_positive,
    )
    required = {"realized_dynamic": model.base_feature_names}
    dictionary_engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features=required,
    )
    dense_engine = OnlineFeatureEngine(
        state,
        backend="native",
        required_features=required,
        dense_output=True,
    )

    def dictionary_features() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return dictionary_engine.realized_rows(
            reference_candidates, state_hash=state_hash
        )

    def dense_features() -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return dense_engine.realized_rows(
            reference_candidates, state_hash=state_hash
        )

    dictionary_result, dictionary_feature_times = _measure(
        dictionary_features, warmup=warmup, repeats=repeats
    )
    dense_result, dense_feature_times = _measure(
        dense_features, warmup=warmup, repeats=repeats
    )
    dictionary_rows, dictionary_feature_metrics = dictionary_result
    dense_rows, dense_feature_metrics = dense_result
    maximum_feature_delta = 0.0
    for dictionary_row, dense_row in zip(dictionary_rows, dense_rows):
        expected = dictionary_row["features"]["realized_dynamic"]
        actual = _dense_mapping(dense_row)
        if set(expected) != set(actual):
            raise RuntimeError("dense feature schema differs from dictionary output")
        maximum_feature_delta = max(
            maximum_feature_delta,
            max(
                abs(float(expected[name]) - float(actual[name]))
                for name in expected
            ),
        )

    reference_score, reference_score_times = _measure(
        lambda: score_online_candidates(dictionary_rows, legacy_score_model),
        warmup=warmup,
        repeats=repeats,
    )
    optimized_score, optimized_score_times = _measure(
        lambda: score_online_candidates(dense_rows, model),
        warmup=warmup,
        repeats=repeats,
    )
    maximum_score_delta = max(
        (
            abs(float(left) - float(right))
            for left, right in zip(reference_score[1], optimized_score[1])
        ),
        default=0.0,
    )
    if reference_score[0] != optimized_score[0] or maximum_score_delta > 1e-12:
        raise RuntimeError("dense scoring changes ranking or score")

    reference_selection = [
        proposal + feature + score
        for proposal, feature, score in zip(
            reference_proposal_times,
            dictionary_feature_times,
            reference_score_times,
        )
    ]
    optimized_selection = [
        proposal + feature + score
        for proposal, feature, score in zip(
            optimized_proposal_times,
            dense_feature_times,
            optimized_score_times,
        )
    ]
    return {
        "task_id": row["task_id"],
        "map_id": row["map_id"],
        "layout_family": row.get("layout_mode"),
        "agent_count": int(row["agent_count"]),
        "solver_seed": solver_seed,
        "skipped": False,
        "candidate_count": len(reference_candidates),
        "proposal_count": int(reference_result[1]["proposal_count"]),
        "candidate_pool_exact": True,
        "selected_candidate_exact": True,
        "maximum_feature_delta": maximum_feature_delta,
        "maximum_score_delta": maximum_score_delta,
        "reference_proposal_median_seconds": _median(reference_proposal_times),
        "optimized_proposal_median_seconds": _median(optimized_proposal_times),
        "optimized_proposal_p95_seconds": _p95(optimized_proposal_times),
        "proposal_time_reduction": 1.0
        - _median(optimized_proposal_times) / _median(reference_proposal_times),
        "dictionary_feature_median_seconds": _median(dictionary_feature_times),
        "dense_feature_median_seconds": _median(dense_feature_times),
        "dense_feature_p95_seconds": _p95(dense_feature_times),
        "feature_time_reduction": 1.0
        - _median(dense_feature_times) / _median(dictionary_feature_times),
        "dictionary_native_state_analysis_seconds": float(
            dictionary_feature_metrics.get("state_analysis_seconds", 0.0)
        ),
        "dictionary_native_feature_fill_seconds": float(
            dictionary_feature_metrics.get("feature_fill_seconds", 0.0)
        ),
        "dictionary_python_feature_seconds": float(
            dictionary_feature_metrics.get("feature_python_seconds", 0.0)
        ),
        "dense_native_state_analysis_seconds": float(
            dense_feature_metrics.get("state_analysis_seconds", 0.0)
        ),
        "dense_native_state_input_seconds": float(
            dense_feature_metrics.get("state_input_seconds", 0.0)
        ),
        "dense_native_state_conflict_scan_seconds": float(
            dense_feature_metrics.get("state_conflict_scan_seconds", 0.0)
        ),
        "dense_native_state_graph_seconds": float(
            dense_feature_metrics.get("state_graph_seconds", 0.0)
        ),
        "dense_native_state_path_aggregate_seconds": float(
            dense_feature_metrics.get("state_path_aggregate_seconds", 0.0)
        ),
        "dense_native_feature_fill_seconds": float(
            dense_feature_metrics.get("feature_fill_seconds", 0.0)
        ),
        "dense_python_feature_seconds": float(
            dense_feature_metrics.get("feature_python_seconds", 0.0)
        ),
        "reference_selection_median_seconds": _median(reference_selection),
        "optimized_selection_median_seconds": _median(optimized_selection),
        "optimized_selection_p95_seconds": _p95(optimized_selection),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark exact compact-proposal and dense-feature runtime paths."
    )
    parser.add_argument(
        "--dataset", default="build/initlns-movingai-ood-dataset-v1"
    )
    parser.add_argument(
        "--config", default="configs/movingai_ood_collection.json"
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument("--task-id", action="append")
    parser.add_argument("--solver-seed", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", default="build/initlns-exact-runtime-benchmark/report.json"
    )
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats <= 0:
        parser.error("warmup must be non-negative and repeats must be positive")

    dataset = (PROJECT_ROOT / arguments.dataset).resolve()
    config = _read_json((PROJECT_ROOT / arguments.config).resolve())
    selected_ids = set(arguments.task_id or DEFAULT_TASKS)
    rows = [
        row
        for row in _load_dataset_rows(dataset, [str(config["split"])])
        if str(row["task_id"]) in selected_ids
    ]
    if {str(row["task_id"]) for row in rows} != selected_ids:
        missing = sorted(selected_ids - {str(row["task_id"]) for row in rows})
        parser.error(f"dataset is missing benchmark tasks: {missing}")
    bundle = load_controller_bundle(
        (PROJECT_ROOT / arguments.controller_bundle).resolve()
    )
    cases = [
        benchmark_case(
            dataset=dataset,
            row=row,
            config=config,
            controller_bundle=bundle,
            solver_seed=arguments.solver_seed,
            warmup=arguments.warmup,
            repeats=arguments.repeats,
        )
        for row in rows
    ]
    measured = [row for row in cases if not row["skipped"]]
    if not measured:
        raise RuntimeError("no repairable benchmark state was available")
    report = {
        "schema": "lns2.exact_runtime_benchmark.v1",
        "cases": cases,
        "maximum_feature_delta": max(
            float(row["maximum_feature_delta"]) for row in measured
        ),
        "maximum_score_delta": max(
            float(row["maximum_score_delta"]) for row in measured
        ),
        "median_proposal_time_reduction": statistics.median(
            float(row["proposal_time_reduction"]) for row in measured
        ),
        "median_feature_time_reduction": statistics.median(
            float(row["feature_time_reduction"]) for row in measured
        ),
        "median_optimized_selection_seconds": statistics.median(
            float(row["optimized_selection_median_seconds"])
            for row in measured
        ),
    }
    report["gates"] = {
        "exact": report["maximum_feature_delta"] <= 1e-12
        and report["maximum_score_delta"] <= 1e-12
        and all(bool(row["candidate_pool_exact"]) for row in measured),
        "proposal_reduction_at_least_40_percent": report[
            "median_proposal_time_reduction"
        ]
        >= 0.40,
        "feature_reduction_at_least_25_percent": report[
            "median_feature_time_reduction"
        ]
        >= 0.25,
        "selection_at_most_0_15_seconds": report[
            "median_optimized_selection_seconds"
        ]
        <= 0.15,
    }
    report["passed"] = all(report["gates"].values())
    output = (PROJECT_ROOT / arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["gates"]["exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
