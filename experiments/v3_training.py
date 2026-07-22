from __future__ import annotations

import collections
import math
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v3 import (
    V3_FEATURE_NAMES,
    V3_FEATURE_SCHEMA_ID,
    V3_FEATURE_SCHEMA_SHA256,
    resolve_v3_feature_names,
)
from experiments.high_load_rescue_training import (
    _positive_probability,
    _weights as balanced_trial_weights,
)
from experiments.repair_aware import PortableScalarModel, load_portable_scalar_model
from experiments.repair_aware_training import (
    MODEL_PARAMETERS,
    _balanced_map_folds,
    _portable_payload,
)
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.v3_controller import (
    V3_BUNDLE_SCHEMA,
    load_v3_controller_bundle,
    v3_candidate_order,
)


V3_TRAINING_SCHEMA = "lns2.v3_training.v1"
V3_PILOT_GATE_SCHEMA = "lns2.v3_pilot_gate.wall_clock_v2"
EFFECTIVE_TOLERANCE_GRID = (0.0, 0.05, 0.10)
NO_PROGRESS_TOLERANCE_GRID = (0.0, 0.05, 0.10)
REDUCTION_RETENTION_GRID = (0.90, 0.95, 0.98)


def _outcome_name(outcome: dict[str, Any]) -> str:
    registered = outcome.get("repair_outcome")
    if registered is not None:
        return str(registered)
    if bool(outcome.get("hard_failure")):
        return "hard_failure"
    before = int(outcome["conflicts_before"])
    after = int(outcome["conflicts_after"])
    if after == 0:
        return "feasible"
    if not bool(outcome.get("repair_state_changed", after != before)):
        return "accepted_noop"
    return "conflict_reduced" if after < before else "state_changed_no_reduction"


def _rows(
    feature_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    split: str,
    feature_names: tuple[str, ...],
) -> list[dict[str, Any]]:
    candidates = {
        (str(row["state_id"]), str(row["candidate_id"])): row
        for row in feature_rows
        if str(row.get("split")) == split
    }
    result: list[dict[str, Any]] = []
    for trial in trial_rows:
        if str(trial.get("split")) != split:
            continue
        key = (str(trial["state_id"]), str(trial["candidate_id"]))
        candidate = candidates.get(key)
        if candidate is None:
            continue
        if str(trial.get("status")) not in {"ok", "resumed"} or not bool(
            trial.get("complete")
        ):
            raise ValueError("v3 training data contains an incomplete trial")
        outcome = dict(trial["outcome"])
        before = int(outcome["conflicts_before"])
        after = int(outcome["conflicts_after"])
        name = _outcome_name(outcome)
        features = dict(candidate["features"]["realized_dynamic"])
        result.append(
            {
                "split": split,
                "state_id": key[0],
                "candidate_id": key[1],
                "map_id": str(candidate["map_id"]),
                "layout_mode": str(candidate.get("layout_mode", "unknown")),
                "agent_count": int(candidate.get("agent_count", 0)),
                "route": str(candidate["route"]),
                "actual_size": int(candidate["actual_size"]),
                "base_selected": bool(candidate.get("base_selected")),
                "v2_score": float(candidate.get("main_score", -1e30)),
                "features": [float(features.get(name, 0.0)) for name in feature_names],
                "effective_progress": int(name in {"conflict_reduced", "feasible"}),
                "no_progress": int(name in {"hard_failure", "accepted_noop"}),
                "hard_failure": int(name == "hard_failure"),
                "accepted_noop": int(name == "accepted_noop"),
                "conflict_reduction": float(max(0, before - after)),
                "repair_seconds": max(1e-9, float(outcome["repair_seconds"])),
                "log_repair_seconds": math.log1p(
                    max(1e-9, float(outcome["repair_seconds"]))
                ),
            }
        )
    expected = set(candidates)
    actual = {(str(row["state_id"]), str(row["candidate_id"])) for row in result}
    if not result or expected != actual:
        raise ValueError("v3 training data lacks complete candidate/trial coverage")
    return result


def _fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    values = np.asarray([row["features"] for row in rows], dtype=float)
    weights = np.asarray(balanced_trial_weights(rows), dtype=float)
    models: dict[str, Any] = {}
    for name, target in (
        ("effective_progress_probability", "effective_progress"),
        ("no_progress_probability", "no_progress"),
    ):
        estimator = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
        estimator.fit(
            values,
            np.asarray([row[target] for row in rows], dtype=int),
            sample_weight=weights,
        )
        models[name] = estimator
    for name, target in (
        ("conflict_reduction", "conflict_reduction"),
        ("log_repair_seconds", "log_repair_seconds"),
    ):
        estimator = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
        estimator.fit(
            values,
            np.asarray([row[target] for row in rows], dtype=float),
            sample_weight=weights,
        )
        models[name] = estimator
    return models


def _predict(models: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    import numpy as np

    values = np.asarray([row["features"] for row in rows], dtype=float)
    return {
        "effective_progress_probability": _positive_probability(
            models["effective_progress_probability"], values
        ),
        "no_progress_probability": _positive_probability(
            models["no_progress_probability"], values
        ),
        "conflict_reduction": [
            max(0.0, float(value))
            for value in models["conflict_reduction"].predict(values)
        ],
        "log_repair_seconds": list(
            map(float, models["log_repair_seconds"].predict(values))
        ),
    }


def _with_runtime_predictions(
    predicted: dict[str, list[float]], selection_overhead_seconds: float
) -> dict[str, list[float]]:
    seconds = [
        max(1e-9, math.expm1(min(50.0, value)))
        for value in predicted["log_repair_seconds"]
    ]
    reduction = predicted["conflict_reduction"]
    return {
        **predicted,
        "repair_seconds": seconds,
        "utility": [
            delta / max(1e-9, duration + selection_overhead_seconds)
            for delta, duration in zip(reduction, seconds)
        ],
    }


def _runtime_model_values(name: str, values: list[float]) -> list[float]:
    if name == "conflict_reduction":
        return [max(0.0, float(value)) for value in values]
    return list(map(float, values))


def _index_predictions(
    rows: list[dict[str, Any]], predictions: dict[str, list[float]]
) -> dict[tuple[str, str], dict[str, list[float]]]:
    result: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for index, row in enumerate(rows):
        key = (str(row["state_id"]), str(row["candidate_id"]))
        for name, values in predictions.items():
            result[key][name].append(float(values[index]))
    return {key: dict(values) for key, values in result.items()}


def _states(
    rows: list[dict[str, Any]],
    predictions: dict[tuple[str, str], dict[str, list[float]]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    result = []
    for state_id, state_rows in sorted(grouped.items()):
        by_arm: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in state_rows:
            by_arm[str(row["candidate_id"])].append(row)
        arms = []
        for candidate_id, trials in sorted(by_arm.items()):
            first = trials[0]
            predicted = predictions[(state_id, candidate_id)]
            arms.append(
                {
                    "candidate_id": candidate_id,
                    "route": str(first["route"]),
                    "actual_size": int(first["actual_size"]),
                    "base_selected": bool(first["base_selected"]),
                    "v2_score": float(first["v2_score"]),
                    "predicted": {
                        name: statistics.fmean(values)
                        for name, values in predicted.items()
                    },
                    "actual": {
                        "effective_rate": statistics.fmean(
                            float(row["effective_progress"]) for row in trials
                        ),
                        "no_progress_rate": statistics.fmean(
                            float(row["no_progress"]) for row in trials
                        ),
                        "hard_failure_rate": statistics.fmean(
                            float(row["hard_failure"]) for row in trials
                        ),
                        "accepted_noop_rate": statistics.fmean(
                            float(row["accepted_noop"]) for row in trials
                        ),
                        "conflict_reduction": statistics.fmean(
                            float(row["conflict_reduction"]) for row in trials
                        ),
                        "repair_seconds": statistics.fmean(
                            float(row["repair_seconds"]) for row in trials
                        ),
                    },
                }
            )
        model = [arm for arm in arms if arm["route"] == "model"]
        adaptive = [arm for arm in arms if arm["route"] == "official_adaptive"]
        if sum(bool(arm["base_selected"]) for arm in model) != 1:
            raise ValueError(f"v3 state lacks exactly one v2 base arm: {state_id}")
        if len(adaptive) != 1:
            raise ValueError(f"v3 state lacks exactly one Adaptive arm: {state_id}")
        first = state_rows[0]
        result.append(
            {
                "state_id": state_id,
                "map_id": str(first["map_id"]),
                "layout_mode": str(first["layout_mode"]),
                "agent_count": int(first["agent_count"]),
                "arms": arms,
            }
        )
    return result


def _select_v3(state: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    arms = [arm for arm in state["arms"] if arm["route"] == "model"]
    candidates = [{"candidate_id": arm["candidate_id"]} for arm in arms]
    predictions = {
        name: [float(arm["predicted"][name]) for arm in arms]
        for name in (
            "effective_progress_probability",
            "no_progress_probability",
            "conflict_reduction",
            "repair_seconds",
            "utility",
        )
    }
    order = v3_candidate_order(
        candidates,
        predictions,
        [float(arm["v2_score"]) for arm in arms],
        thresholds,
    )
    if not order:
        raise ValueError(f"v3 has no eligible model arm for state {state['state_id']}")
    return arms[order[0]]


def _selected(state: dict[str, Any], kind: str, thresholds: dict[str, float]) -> dict[str, Any]:
    if kind == "v3":
        return _select_v3(state, thresholds)
    if kind == "v2":
        return next(arm for arm in state["arms"] if bool(arm["base_selected"]))
    if kind == "adaptive":
        return next(
            arm for arm in state["arms"] if arm["route"] == "official_adaptive"
        )
    raise ValueError(f"unknown v3 comparison kind: {kind}")


def _metrics(
    states: list[dict[str, Any]], kind: str, thresholds: dict[str, float], overhead: float
) -> dict[str, float]:
    arms = [_selected(state, kind, thresholds) for state in states]
    actual = [arm["actual"] for arm in arms]
    mean_reduction = statistics.fmean(
        float(row["conflict_reduction"]) for row in actual
    )
    mean_repair = statistics.fmean(float(row["repair_seconds"]) for row in actual)
    return {
        "state_count": float(len(states)),
        "effective_rate": statistics.fmean(
            float(row["effective_rate"]) for row in actual
        ),
        "no_progress_rate": statistics.fmean(
            float(row["no_progress_rate"]) for row in actual
        ),
        "hard_failure_rate": statistics.fmean(
            float(row["hard_failure_rate"]) for row in actual
        ),
        "accepted_noop_rate": statistics.fmean(
            float(row["accepted_noop_rate"]) for row in actual
        ),
        "mean_conflict_reduction": mean_reduction,
        "mean_repair_seconds": mean_repair,
        "conflict_reduction_per_total_second": mean_reduction
        / max(1e-9, mean_repair + overhead),
    }


def _ratios(v3: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        "effective_rate_delta": v3["effective_rate"] - baseline["effective_rate"],
        "no_progress_rate_delta": v3["no_progress_rate"]
        - baseline["no_progress_rate"],
        "conflict_reduction_ratio": v3["mean_conflict_reduction"]
        / max(1e-9, baseline["mean_conflict_reduction"]),
        "efficiency_ratio": v3["conflict_reduction_per_total_second"]
        / max(1e-9, baseline["conflict_reduction_per_total_second"]),
    }


def _cell_gate(
    states: list[dict[str, Any]], thresholds: dict[str, float], overhead: float
) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for state in states:
        grouped[(str(state["layout_mode"]), int(state["agent_count"]))].append(state)
    rows = []
    for (layout, agents), cell_states in sorted(grouped.items()):
        v3 = _metrics(cell_states, "v3", thresholds, overhead)
        v2 = _metrics(cell_states, "v2", thresholds, overhead)
        ratio = v3["conflict_reduction_per_total_second"] / max(
            1e-9, v2["conflict_reduction_per_total_second"]
        )
        rows.append(
            {
                "layout_mode": layout,
                "agent_count": agents,
                "state_count": len(cell_states),
                "efficiency_ratio_vs_v2": ratio,
                "noninferior": ratio + 1e-12 >= 1.0,
            }
        )
    return {
        "cells": rows,
        "cell_count": len(rows),
        "noninferior_cell_count": sum(bool(row["noninferior"]) for row in rows),
        "worst_efficiency_ratio": min(
            (float(row["efficiency_ratio_vs_v2"]) for row in rows), default=0.0
        ),
    }


def _gate(
    states: list[dict[str, Any]], thresholds: dict[str, float], overhead: float
) -> dict[str, Any]:
    v3 = _metrics(states, "v3", thresholds, overhead)
    v2 = _metrics(states, "v2", thresholds, overhead)
    adaptive = _metrics(states, "adaptive", thresholds, overhead)
    comparison = _ratios(v3, v2)
    adaptive_efficiency_ratio = v3["conflict_reduction_per_total_second"] / max(
        1e-9, adaptive["conflict_reduction_per_total_second"]
    )
    cells = _cell_gate(states, thresholds, overhead)
    checks, diagnostics = _gate_checks(
        comparison, adaptive_efficiency_ratio, cells
    )
    return {
        "v3": v3,
        "v2": v2,
        "adaptive": adaptive,
        "v3_vs_v2": comparison,
        "v3_vs_adaptive_efficiency_ratio": adaptive_efficiency_ratio,
        "cell_gate": cells,
        "checks": checks,
        "diagnostics": diagnostics,
        "passed": all(checks.values()),
    }


def _gate_checks(
    comparison: dict[str, float],
    adaptive_efficiency_ratio: float,
    cells: dict[str, Any],
) -> tuple[dict[str, bool], dict[str, bool]]:
    """Separate wall-clock eligibility from per-repair quality diagnostics.

    A cost-aware controller may rationally trade a smaller conflict reduction per
    repair for a larger reduction per wall-clock second.  The former remains a
    visible regression diagnostic, but complete wall-clock episodes decide whether
    that trade is beneficial.
    """
    checks = {
        "effective_rate": comparison["effective_rate_delta"] + 1e-12 >= -0.01,
        "no_progress_rate": comparison["no_progress_rate_delta"] <= 0.01 + 1e-12,
        "v2_efficiency": comparison["efficiency_ratio"] + 1e-12 >= 1.10,
        "adaptive_efficiency": adaptive_efficiency_ratio + 1e-12 >= 1.0,
        "cell_coverage": cells["cell_count"] == 6,
        "cell_noninferiority": cells["cell_count"] == 6
        and cells["noninferior_cell_count"] >= 5,
        "worst_cell": cells["worst_efficiency_ratio"] + 1e-12 >= 0.90,
    }
    diagnostics = {
        "mean_conflict_reduction_retention_at_least_98pct": (
            comparison["conflict_reduction_ratio"] + 1e-12 >= 0.98
        )
    }
    return checks, diagnostics


def _calibrate(states: list[dict[str, Any]], overhead: float) -> dict[str, Any]:
    rows = []
    for effective in EFFECTIVE_TOLERANCE_GRID:
        for no_progress in NO_PROGRESS_TOLERANCE_GRID:
            for retention in REDUCTION_RETENTION_GRID:
                thresholds = {
                    "effective_probability_tolerance": effective,
                    "no_progress_probability_tolerance": no_progress,
                    "conflict_reduction_retention": retention,
                }
                gate = _gate(states, thresholds, overhead)
                rows.append({"thresholds": thresholds, "gate": gate})
    passing = [row for row in rows if bool(row["gate"]["passed"])]
    selected = max(
        passing or rows,
        key=lambda row: (
            float(row["gate"]["v3"]["conflict_reduction_per_total_second"]),
            float(row["gate"]["v3"]["effective_rate"]),
            -float(row["gate"]["v3"]["no_progress_rate"]),
            float(row["thresholds"]["conflict_reduction_retention"]),
        ),
    )
    return {"grid": rows, "selected": selected, "passed": bool(passing)}


def _prediction_metrics(
    rows: list[dict[str, Any]], predictions: dict[str, list[float]]
) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    effective = [int(row["effective_progress"]) for row in rows]
    no_progress = [int(row["no_progress"]) for row in rows]
    return {
        "trial_count": float(len(rows)),
        "effective_progress_auc": (
            float(
                roc_auc_score(
                    effective, predictions["effective_progress_probability"]
                )
            )
            if len(set(effective)) > 1
            else 0.5
        ),
        "no_progress_auc": (
            float(roc_auc_score(no_progress, predictions["no_progress_probability"]))
            if len(set(no_progress)) > 1
            else 0.5
        ),
        "reduction_mae": statistics.fmean(
            abs(float(row["conflict_reduction"]) - value)
            for row, value in zip(rows, predictions["conflict_reduction"])
        ),
        "repair_log_mae": statistics.fmean(
            abs(float(row["log_repair_seconds"]) - value)
            for row, value in zip(rows, predictions["log_repair_seconds"])
        ),
    }


def train_v3_controller(
    *,
    feature_index: str | Path,
    trial_manifest: str | Path,
    controller_bundle: str | Path,
    output: str | Path,
    selection_overhead_seconds: float,
) -> dict[str, Any]:
    feature_path = Path(feature_index).resolve()
    trial_path = Path(trial_manifest).resolve()
    controller_path = Path(controller_bundle).resolve()
    output_root = Path(output).resolve()
    overhead = float(selection_overhead_seconds)
    if not math.isfinite(overhead) or overhead < 0.0:
        raise ValueError("v3 selection overhead must be finite and nonnegative")
    feature_names = V3_FEATURE_NAMES
    feature_rows = _read_jsonl(feature_path)
    trial_rows = _read_jsonl(trial_path)
    train_rows = _rows(feature_rows, trial_rows, "policy_train", feature_names)
    diagnostic_rows = _rows(
        feature_rows, trial_rows, "policy_validation", feature_names
    )
    train_maps = {str(row["map_id"]) for row in train_rows}
    diagnostic_maps = {str(row["map_id"]) for row in diagnostic_rows}
    if train_maps & diagnostic_maps:
        raise ValueError("v3 train and diagnostic maps overlap")

    oof_index: dict[tuple[str, str], dict[str, list[float]]] = {}
    fold_reports = []
    for fold in _balanced_map_folds(train_rows):
        held_maps = set(fold["validation_maps"])
        training = [row for row in train_rows if row["map_id"] not in held_maps]
        held = [row for row in train_rows if row["map_id"] in held_maps]
        models = _fit(training)
        predicted = _with_runtime_predictions(_predict(models, held), overhead)
        indexed = _index_predictions(held, predicted)
        if set(oof_index) & set(indexed):
            raise ValueError("v3 OOF predictions overlap")
        oof_index.update(indexed)
        fold_reports.append(
            {
                **fold,
                "training_trial_count": len(training),
                "validation_trial_count": len(held),
                "prediction_metrics": _prediction_metrics(held, predicted),
            }
        )
    train_states = _states(train_rows, oof_index)
    calibration = _calibrate(train_states, overhead)
    thresholds = dict(calibration["selected"]["thresholds"])

    models = _fit(train_rows)
    diagnostic_raw = _predict(models, diagnostic_rows)
    diagnostic_predictions = _with_runtime_predictions(diagnostic_raw, overhead)
    diagnostic_states = _states(
        diagnostic_rows,
        _index_predictions(diagnostic_rows, diagnostic_predictions),
    )
    diagnostic_gate = _gate(diagnostic_states, thresholds, overhead)
    main_bundle = load_controller_bundle(controller_path)
    source_fingerprint = _fingerprint(
        {
            "feature_index_sha256": sha256_file(feature_path),
            "trial_manifest_sha256": sha256_file(trial_path),
            "main_ranker_semantic_fingerprint": main_bundle.manifest[
                "main_ranker_semantic_fingerprint"
            ],
            "feature_schema_sha256": V3_FEATURE_SCHEMA_SHA256,
            "model_parameters": MODEL_PARAMETERS,
            "threshold_grid": {
                "effective": EFFECTIVE_TOLERANCE_GRID,
                "no_progress": NO_PROGRESS_TOLERANCE_GRID,
                "retention": REDUCTION_RETENTION_GRID,
            },
            "pilot_gate_schema": V3_PILOT_GATE_SCHEMA,
            "selection_overhead_seconds": overhead,
        }
    )

    output_root.mkdir(parents=True, exist_ok=True)
    dense_diagnostic = [
        {
            "feature_profile": "realized_dynamic",
            "feature_names": feature_names,
            "feature_values": tuple(row["features"]),
        }
        for row in diagnostic_rows
    ]
    model_rows = {}
    parity = {}
    native_available = True
    for name, estimator in models.items():
        payload = _portable_payload(name, estimator, list(feature_names), source_fingerprint)
        path = output_root / f"v3__{name}.json"
        _write_json(path, payload)
        portable = load_portable_scalar_model(payload)
        python_portable: PortableScalarModel = replace(portable, native_predictor=None)
        python_values = _runtime_model_values(
            name, python_portable.predict(dense_diagnostic)
        )
        portable_values = _runtime_model_values(
            name, portable.predict(dense_diagnostic)
        )
        reference = diagnostic_raw[name]
        sklearn_python_delta = max(
            (abs(float(left) - float(right)) for left, right in zip(reference, python_values)),
            default=0.0,
        )
        native_python_delta = max(
            (
                abs(float(left) - float(right))
                for left, right in zip(portable_values, python_values)
            ),
            default=0.0,
        )
        native = portable.inference_backend == "native-portable-tree"
        native_available = native_available and native
        if sklearn_python_delta > 1e-12 or native_python_delta > 1e-12:
            raise ValueError(
                "v3 portable model differs from sklearn/native: "
                f"{name} sklearn_python={sklearn_python_delta} "
                f"native_python={native_python_delta}"
            )
        parity[name] = {
            "sklearn_python_maximum_delta": sklearn_python_delta,
            "native_python_maximum_delta": native_python_delta,
            "inference_backend": portable.inference_backend,
        }
        model_rows[name] = {
            "file": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
            "semantic_fingerprint": payload["semantic_fingerprint"],
            "tree_count": len(payload["trees"]),
            "inference_backend": portable.inference_backend,
        }

    parity_maximum = max(
        (
            max(
                float(row["sklearn_python_maximum_delta"]),
                float(row["native_python_maximum_delta"]),
            )
            for row in parity.values()
        ),
        default=0.0,
    )
    pilot_checks = {
        **dict(diagnostic_gate["checks"]),
        "calibration_passed": bool(calibration["passed"]),
        "native_available": native_available,
        "portable_parity": parity_maximum <= 1e-12,
    }
    pilot_passed = all(pilot_checks.values())
    report = {
        "schema": V3_TRAINING_SCHEMA,
        "schema_version": 1,
        "pilot_gate_schema": V3_PILOT_GATE_SCHEMA,
        "training_labels_seen": ["policy_train"],
        "formal_or_movingai_labels_seen": False,
        "diagnostic_split_locked_once": True,
        "feature_schema_id": V3_FEATURE_SCHEMA_ID,
        "feature_schema_sha256": V3_FEATURE_SCHEMA_SHA256,
        "feature_count": len(feature_names),
        "training_state_count": len(train_states),
        "diagnostic_state_count": len(diagnostic_states),
        "training_map_count": len(train_maps),
        "diagnostic_map_count": len(diagnostic_maps),
        "training_agent_counts": sorted({int(row["agent_count"]) for row in train_rows}),
        "model_parameters": MODEL_PARAMETERS,
        "selection_overhead_seconds": overhead,
        "folds": fold_reports,
        "calibration": calibration,
        "diagnostic_prediction_metrics": _prediction_metrics(
            diagnostic_rows, diagnostic_predictions
        ),
        "diagnostic_gate": diagnostic_gate,
        "portable_parity": parity,
        "portable_maximum_delta": parity_maximum,
        "native_available": native_available,
        "pilot_checks": pilot_checks,
        "pilot_passed": pilot_passed,
        "decision": "v3_pilot_passed" if pilot_passed else "v3_pilot_failed",
    }
    report_path = output_root / "training_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema": V3_BUNDLE_SCHEMA,
        "schema_version": 1,
        "pilot_gate_schema": V3_PILOT_GATE_SCHEMA,
        "feature_schema_id": V3_FEATURE_SCHEMA_ID,
        "feature_schema_sha256": V3_FEATURE_SCHEMA_SHA256,
        "profile": "realized_dynamic",
        "feature_names": list(feature_names),
        "models": model_rows,
        "thresholds": thresholds,
        "selection_overhead_seconds": overhead,
        "maximum_distinct_failures": 3,
        "terminal_fallback": "official_adaptive",
        "main_ranker_semantic_fingerprint": main_bundle.manifest[
            "main_ranker_semantic_fingerprint"
        ],
        "training_report": {
            "file": report_path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(report_path),
        },
        "source_fingerprint": source_fingerprint,
        "deployment_promoted": False,
    }
    _write_json(output_root / "v3_manifest.json", manifest)

    # Exercise the public loader before the artifact is reported complete.
    load_v3_controller_bundle(output_root)
    return {**report, "manifest": manifest}


def finalize_v3_native_audit(
    *,
    feature_index: str | Path,
    trial_manifest: str | Path,
    controller_output: str | Path,
) -> dict[str, Any]:
    """Complete native parity without requiring scikit-learn in the runtime.

    Model fitting and sklearn/Python parity may run in the registered Windows
    training environment.  This final audit runs in WSL, where the native
    PortableTreeEnsemble is available, and seals the report/manifest only after
    native/Python parity passes.
    """

    feature_path = Path(feature_index).resolve()
    trial_path = Path(trial_manifest).resolve()
    output_root = Path(controller_output).resolve()
    report_path = output_root / "training_report.json"
    manifest_path = output_root / "v3_manifest.json"
    report = dict(_read_json(report_path))
    manifest = dict(_read_json(manifest_path))
    feature_names = resolve_v3_feature_names(
        str(manifest.get("feature_schema_id")),
        str(manifest.get("feature_schema_sha256")),
    )
    diagnostic_rows = _rows(
        _read_jsonl(feature_path),
        _read_jsonl(trial_path),
        "policy_validation",
        feature_names,
    )
    dense_diagnostic = [
        {
            "feature_profile": "realized_dynamic",
            "feature_names": feature_names,
            "feature_values": tuple(row["features"]),
        }
        for row in diagnostic_rows
    ]
    parity = dict(report.get("portable_parity") or {})
    native_available = True
    for name, raw in dict(manifest.get("models") or {}).items():
        model_path = output_root / str(dict(raw)["file"])
        portable = load_portable_scalar_model(dict(_read_json(model_path)))
        python_portable: PortableScalarModel = replace(
            portable, native_predictor=None
        )
        python_values = _runtime_model_values(
            str(name), python_portable.predict(dense_diagnostic)
        )
        native_values = _runtime_model_values(
            str(name), portable.predict(dense_diagnostic)
        )
        native_delta = max(
            (
                abs(float(left) - float(right))
                for left, right in zip(native_values, python_values)
            ),
            default=0.0,
        )
        native = portable.inference_backend == "native-portable-tree"
        native_available = native_available and native
        if not native:
            raise RuntimeError(f"v3 native predictor is unavailable: {name}")
        if native_delta > 1e-12:
            raise ValueError(f"v3 native model differs from Python: {name}")
        previous = dict(parity.get(name) or {})
        parity[name] = {
            **previous,
            "native_python_maximum_delta": native_delta,
            "inference_backend": portable.inference_backend,
        }
        manifest["models"][name]["inference_backend"] = (
            portable.inference_backend
        )
    parity_maximum = max(
        (
            max(
                float(row.get("sklearn_python_maximum_delta", 0.0)),
                float(row.get("native_python_maximum_delta", 0.0)),
            )
            for row in parity.values()
        ),
        default=0.0,
    )
    pilot_checks = {
        **dict(dict(report["diagnostic_gate"])["checks"]),
        "calibration_passed": bool(dict(report["calibration"])["passed"]),
        "native_available": native_available,
        "portable_parity": parity_maximum <= 1e-12,
    }
    pilot_passed = all(pilot_checks.values())
    report.update(
        {
            "portable_parity": parity,
            "portable_maximum_delta": parity_maximum,
            "native_available": native_available,
            "native_audit_completed": True,
            "pilot_checks": pilot_checks,
            "pilot_passed": pilot_passed,
            "decision": (
                "v3_pilot_passed" if pilot_passed else "v3_pilot_failed"
            ),
        }
    )
    _write_json(report_path, report)
    manifest["training_report"]["sha256"] = sha256_file(report_path)
    manifest["native_audit_completed"] = True
    _write_json(manifest_path, manifest)
    load_v3_controller_bundle(output_root)
    return {**report, "manifest": manifest}


__all__ = [
    "V3_PILOT_GATE_SCHEMA",
    "V3_TRAINING_SCHEMA",
    "_gate_checks",
    "finalize_v3_native_audit",
    "train_v3_controller",
]
