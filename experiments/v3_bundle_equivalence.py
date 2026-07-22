from __future__ import annotations

import collections
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from experiments._common import quantile, read_jsonl, write_json
from experiments.compact_controller_model import load_controller_bundle
from experiments.v3_controller import (
    V3ControllerBundle,
    load_v3_controller_bundle,
    v3_candidate_order,
)


V3_BUNDLE_EQUIVALENCE_SCHEMA = "lns2.v3_bundle_equivalence.v1"
PREDICTION_NAMES = (
    "effective_progress_probability",
    "no_progress_probability",
    "conflict_reduction",
    "repair_seconds",
    "utility",
)


def _without_native(bundle: V3ControllerBundle) -> V3ControllerBundle:
    return V3ControllerBundle(
        models={
            name: replace(model, native_predictor=None)
            for name, model in bundle.models.items()
        },
        thresholds=dict(bundle.thresholds),
        selection_overhead_seconds=bundle.selection_overhead_seconds,
        manifest=dict(bundle.manifest),
        report=dict(bundle.report),
    )


def _maximum_prediction_delta(
    left: dict[str, list[float]], right: dict[str, list[float]]
) -> dict[str, float]:
    result = {}
    for name in PREDICTION_NAMES:
        left_values = left[name]
        right_values = right[name]
        if len(left_values) != len(right_values):
            raise ValueError("v3 prediction vectors have different lengths")
        result[name] = max(
            (
                abs(float(left_value) - float(right_value))
                for left_value, right_value in zip(left_values, right_values)
            ),
            default=0.0,
        )
    return result


def _state_rows(feature_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in feature_rows:
        if str(row.get("route")) == "model":
            grouped[str(row["state_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row["candidate_id"]))
    return dict(sorted(grouped.items()))


def _order(
    bundle: V3ControllerBundle,
    rows: list[dict[str, Any]],
    predictions: dict[str, list[float]],
) -> list[str]:
    indices = v3_candidate_order(
        rows,
        predictions,
        [float(row["main_score"]) for row in rows],
        bundle.thresholds,
    )
    return [str(rows[index]["candidate_id"]) for index in indices]


def _timings(
    operation: Callable[[], object], *, warmups: int, repeats: int
) -> dict[str, Any]:
    for _ in range(warmups):
        operation()
    values = []
    for _ in range(repeats):
        started = time.perf_counter()
        operation()
        values.append(time.perf_counter() - started)
    return {
        "warmups": warmups,
        "repeats": repeats,
        "median_seconds": statistics.median(values),
        "p95_seconds": quantile(values, 0.95),
        "minimum_seconds": min(values),
        "maximum_seconds": max(values),
    }


def audit_v3_bundle_equivalence(
    *,
    feature_index: str | Path,
    reference_bundle: str | Path,
    candidate_bundle: str | Path,
    output: str | Path,
    main_controller_bundle: str | Path | None = None,
    require_native: bool = True,
    benchmark_repeats: int = 100,
) -> dict[str, Any]:
    feature_path = Path(feature_index).resolve()
    output_root = Path(output).resolve()
    rows = read_jsonl(feature_path)
    if not rows:
        raise ValueError("v3 equivalence feature index is empty")
    reference = load_v3_controller_bundle(reference_bundle)
    candidate = load_v3_controller_bundle(candidate_bundle)
    reference_python = _without_native(reference)
    candidate_python = _without_native(candidate)

    reference_predictions = reference.predict(rows)
    candidate_predictions = candidate.predict(rows)
    reference_python_predictions = reference_python.predict(rows)
    candidate_python_predictions = candidate_python.predict(rows)
    implementation_delta = _maximum_prediction_delta(
        reference_predictions, candidate_predictions
    )
    python_implementation_delta = _maximum_prediction_delta(
        reference_python_predictions, candidate_python_predictions
    )
    reference_native_delta = _maximum_prediction_delta(
        reference_predictions, reference_python_predictions
    )
    candidate_native_delta = _maximum_prediction_delta(
        candidate_predictions, candidate_python_predictions
    )

    grouped = _state_rows(rows)
    ranking_mismatches = []
    selection_mismatches = []
    for state_id, state_rows in grouped.items():
        left_prediction = reference.predict(state_rows)
        right_prediction = candidate.predict(state_rows)
        left_order = _order(reference, state_rows, left_prediction)
        right_order = _order(candidate, state_rows, right_prediction)
        if left_order != right_order:
            ranking_mismatches.append(state_id)
        if left_order[:1] != right_order[:1]:
            selection_mismatches.append(state_id)

    representative_state_id, representative_rows = min(
        grouped.items(), key=lambda item: (abs(len(item[1]) - 18), item[0])
    )

    def benchmark(bundle: V3ControllerBundle) -> object:
        predictions = bundle.predict(representative_rows)
        return _order(bundle, representative_rows, predictions)

    reference_timing = _timings(
        lambda: benchmark(reference), warmups=5, repeats=benchmark_repeats
    )
    candidate_timing = _timings(
        lambda: benchmark(candidate), warmups=5, repeats=benchmark_repeats
    )
    main_runtime_count = 0
    main_runtime: set[str] = set()
    if main_controller_bundle is not None:
        main_bundle = load_controller_bundle(main_controller_bundle)
        main_runtime = set(
            main_bundle.main_models["realized_dynamic"].base_feature_names
        )
        main_runtime_count = len(main_runtime)
    reference_runtime = set(reference.required_feature_names)
    candidate_runtime = set(candidate.required_feature_names)
    reference_native = reference.inference_backends == ("native-portable-tree",)
    candidate_native = candidate.inference_backends == ("native-portable-tree",)
    maximum_delta = max(
        (
            *implementation_delta.values(),
            *python_implementation_delta.values(),
            *reference_native_delta.values(),
            *candidate_native_delta.values(),
        ),
        default=0.0,
    )
    checks = {
        "state_coverage": len(grouped) == 180,
        "candidate_coverage": len(rows) == 3412,
        "thresholds_equal": reference.thresholds == candidate.thresholds,
        "selection_overhead_equal": reference.selection_overhead_seconds
        == candidate.selection_overhead_seconds,
        "prediction_parity": maximum_delta <= 1e-12,
        "ranking_parity": not ranking_mismatches,
        "selection_parity": not selection_mismatches,
        "reference_native": reference_native or not require_native,
        "candidate_native": candidate_native or not require_native,
    }
    report = {
        "schema": V3_BUNDLE_EQUIVALENCE_SCHEMA,
        "schema_version": 1,
        "feature_index": str(feature_path),
        "reference_bundle": str(Path(reference_bundle).resolve()),
        "candidate_bundle": str(Path(candidate_bundle).resolve()),
        "state_count": len(grouped),
        "candidate_count": len(rows),
        "model_candidate_count": sum(len(state_rows) for state_rows in grouped.values()),
        "prediction_maximum_delta": implementation_delta,
        "python_prediction_maximum_delta": python_implementation_delta,
        "reference_native_python_maximum_delta": reference_native_delta,
        "candidate_native_python_maximum_delta": candidate_native_delta,
        "overall_maximum_delta": maximum_delta,
        "ranking_mismatch_count": len(ranking_mismatches),
        "selection_mismatch_count": len(selection_mismatches),
        "ranking_mismatch_states": ranking_mismatches[:20],
        "selection_mismatch_states": selection_mismatches[:20],
        "reference_inference_backends": list(reference.inference_backends),
        "candidate_inference_backends": list(candidate.inference_backends),
        "runtime_projection": {
            "reference": reference.runtime_projection,
            "candidate": candidate.runtime_projection,
            "v2_runtime_feature_count": main_runtime_count,
            "reference_combined_v2_v3_feature_count": len(
                reference_runtime | main_runtime
            ),
            "candidate_combined_v2_v3_feature_count": len(
                candidate_runtime | main_runtime
            ),
        },
        "selection_microbenchmark": {
            "representative_state_id": representative_state_id,
            "candidate_count": len(representative_rows),
            "scope": "cached_features_plus_four_head_inference_and_ordering",
            "reference": reference_timing,
            "candidate": candidate_timing,
            "median_speedup_fraction": (
                reference_timing["median_seconds"]
                - candidate_timing["median_seconds"]
            )
            / max(1e-12, reference_timing["median_seconds"]),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "v3_bundle_equivalence_report.json", report)
    markdown = [
        "# v3 94-feature bundle equivalence",
        "",
        f"- Passed: `{report['passed']}`",
        f"- Coverage: {len(grouped)} states, {len(rows)} candidates.",
        f"- Maximum prediction delta: {maximum_delta:.3g}.",
        f"- Ranking mismatches: {len(ranking_mismatches)}.",
        f"- Selection mismatches: {len(selection_mismatches)}.",
        "- Declared training features: "
        f"{reference.runtime_projection['declared_feature_count']} -> "
        f"{candidate.runtime_projection['declared_feature_count']}.",
        "- Runtime v3 features: "
        f"{reference.runtime_projection['runtime_feature_count']} -> "
        f"{candidate.runtime_projection['runtime_feature_count']}.",
        "- Combined v2/v3 runtime features: "
        f"{len(reference_runtime | main_runtime)} -> "
        f"{len(candidate_runtime | main_runtime)}.",
        "",
        "The microbenchmark measures cached feature rows plus four-head inference "
        "and ordering. Candidate generation and feature extraction are intentionally "
        "outside this equivalence audit.",
        "",
    ]
    (output_root / "v3_bundle_equivalence_report.md").write_text(
        "\n".join(markdown), encoding="utf-8", newline="\n"
    )
    return report


__all__ = ["V3_BUNDLE_EQUIVALENCE_SCHEMA", "audit_v3_bundle_equivalence"]
