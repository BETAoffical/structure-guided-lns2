#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NATIVE_BUILD = PROJECT_ROOT / "build" / "linux" / "project"
sys.path.insert(0, str(PROJECT_ROOT))
if NATIVE_BUILD.is_dir():
    sys.path.insert(0, str(NATIVE_BUILD))

from experiments.closed_loop_confirmation import score_online_candidates  # noqa: E402
from experiments.compact_controller_model import (  # noqa: E402
    load_compact_model,
    load_controller_bundle,
)
from experiments.repair_collection import _read_json  # noqa: E402


def _candidate_rows(path: Path, expected_count: int) -> tuple[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            grouped[str(row["state_id"])].append(row)
    for state_id, rows in grouped.items():
        if len(rows) == expected_count:
            return state_id, rows
    raise ValueError(
        f"candidate index has no state with exactly {expected_count} candidates"
    )


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _ranking(rows: list[dict[str, Any]], scores: list[float]) -> list[str]:
    return [
        str(rows[index]["candidate_key"])
        for index in sorted(
            range(len(rows)),
            key=lambda index: (
                -round(float(scores[index]), 12),
                str(rows[index]["candidate_key"]),
            ),
        )
    ]


def benchmark(
    controller_bundle: Path,
    candidate_index: Path,
    *,
    profile: str,
    candidate_count: int,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    loaded = load_controller_bundle(controller_bundle)
    native_model = loaded.main_models[profile]
    if native_model.inference_backend != "native-portable-tree":
        raise RuntimeError(
            "PortableTreeEnsemble is unavailable; add the Linux module directory "
            "to PYTHONPATH before running this benchmark"
        )
    model_row = dict(loaded.manifest["main_rankers"][profile])
    python_model = load_compact_model(
        _read_json(controller_bundle / str(model_row["file"]))
    )
    state_id, rows = _candidate_rows(candidate_index, candidate_count)

    vectors: list[list[float]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            vectors.append(python_model.pair_vector(rows[left], rows[right]))
            vectors.append(python_model.pair_vector(rows[right], rows[left]))
    python_probabilities = python_model.predict_positive(vectors)
    native_probabilities = native_model.predict_positive(vectors)
    maximum_probability_delta = max(
        (
            abs(left - right)
            for left, right in zip(python_probabilities, native_probabilities)
        ),
        default=0.0,
    )
    python_selection, python_scores, python_margin = score_online_candidates(
        rows, python_model
    )
    native_selection, native_scores, native_margin = score_online_candidates(
        rows, native_model
    )
    maximum_score_delta = max(
        (abs(left - right) for left, right in zip(python_scores, native_scores)),
        default=0.0,
    )
    python_ranking = _ranking(rows, python_scores)
    native_ranking = _ranking(rows, native_scores)
    equivalent = (
        maximum_probability_delta <= 1e-12
        and maximum_score_delta <= 1e-12
        and abs(python_margin - native_margin) <= 1e-12
        and python_ranking == native_ranking
        and python_selection == native_selection
    )
    if not equivalent:
        raise RuntimeError("native and Python portable-tree inference are not equivalent")

    models = {"python": python_model, "native": native_model}
    for _ in range(warmup):
        score_online_candidates(rows, python_model)
        score_online_candidates(rows, native_model)
    samples: dict[str, list[float]] = {"python": [], "native": []}
    for repeat in range(repeats):
        order = ("python", "native") if repeat % 2 == 0 else ("native", "python")
        for backend in order:
            started = time.perf_counter()
            score_online_candidates(rows, models[backend])
            samples[backend].append(time.perf_counter() - started)

    timing = {
        backend: {
            "median_seconds": statistics.median(values),
            "p95_seconds": _p95(values),
            "median_milliseconds": 1000.0 * statistics.median(values),
            "p95_milliseconds": 1000.0 * _p95(values),
        }
        for backend, values in samples.items()
    }
    timing["native"]["median_speedup_vs_python"] = (
        timing["python"]["median_seconds"] / timing["native"]["median_seconds"]
    )
    timing["native"]["p95_speedup_vs_python"] = (
        timing["python"]["p95_seconds"] / timing["native"]["p95_seconds"]
    )
    return {
        "schema": "lns2.portable_tree_inference_microbenchmark.v1",
        "profile": profile,
        "candidate_state_id": state_id,
        "candidate_count": len(rows),
        "unordered_pair_count": len(rows) * (len(rows) - 1) // 2,
        "probability_evaluation_count": len(vectors),
        "warmup_repetitions": warmup,
        "measured_repetitions": repeats,
        "python_version": sys.version,
        "python_backend": python_model.inference_backend,
        "native_backend": native_model.inference_backend,
        "maximum_probability_delta": maximum_probability_delta,
        "maximum_score_delta": maximum_score_delta,
        "margin_delta": abs(python_margin - native_margin),
        "ranking_matches": python_ranking == native_ranking,
        "selection_matches": python_selection == native_selection,
        "selected_candidate_key": str(rows[python_selection]["candidate_key"]),
        "equivalent_within_1e-12": equivalent,
        "timing": timing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Python and native portable-tree pairwise scoring."
    )
    parser.add_argument(
        "--controller-bundle",
        default="artifacts/initlns-closed-loop-controller-v2",
    )
    parser.add_argument("--candidate-index")
    parser.add_argument("--profile", default="realized_dynamic")
    parser.add_argument("--candidate-count", type=int, default=18)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats <= 0 or args.candidate_count <= 1:
        parser.error("warmup must be non-negative; repeats and candidate-count must be positive")
    candidate_index = args.candidate_index
    if candidate_index is None:
        config = _read_json(PROJECT_ROOT / "configs" / "movingai_ood_collection.json")
        candidate_index = str(config["model_registration"]["development_index"])
    controller_bundle = Path(args.controller_bundle)
    if not controller_bundle.is_absolute():
        controller_bundle = PROJECT_ROOT / controller_bundle
    candidate_index_path = Path(candidate_index)
    if not candidate_index_path.is_absolute():
        candidate_index_path = PROJECT_ROOT / candidate_index_path
    result = benchmark(
        controller_bundle.resolve(),
        candidate_index_path.resolve(),
        profile=str(args.profile),
        candidate_count=int(args.candidate_count),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    if args.output:
        output = Path(args.output)
        if not output.is_absolute():
            output = PROJECT_ROOT / output
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(output.name + ".partial")
        partial.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        partial.replace(output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
