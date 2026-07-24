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
    V3_H3_BUNDLE_SCHEMA,
    load_v3_controller_bundle,
    v3_h3_candidate_order,
)
from experiments.v3_training import (
    _index_predictions,
    _rows as one_step_rows,
)


V3_H3_TRAINING_SCHEMA = "lns2.v3_horizon_training.v1"
V3_H3_GATE_SCHEMA = "lns2.v3_horizon_gate.v1"
EFFECTIVE_TOLERANCE_GRID = (0.0, 0.05, 0.10)
NO_PROGRESS_TOLERANCE_GRID = (0.0, 0.05, 0.10)
UTILITY_IMPROVEMENT_GRID = (0.0, 0.05, 0.10)


def _h1_rows(
    features: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    rows = one_step_rows(features, trials, split, V3_FEATURE_NAMES)
    outcomes = {
        (str(row["state_id"]), str(row["candidate_id"]), int(row["trial_index"])): dict(
            row["outcome"]
        )
        for row in trials
        if str(row.get("split")) == split
    }
    counters: collections.Counter[tuple[str, str]] = collections.Counter()
    result = []
    for row in rows:
        key = (str(row["state_id"]), str(row["candidate_id"]))
        trial_index = counters[key]
        counters[key] += 1
        outcome = outcomes[(key[0], key[1], trial_index)]
        if str(row["route"]) != "model":
            continue
        result.append(
            {
                **row,
                "h1_effective_progress": int(row["effective_progress"]),
                "h1_no_progress": int(row["no_progress"]),
                "h1_conflict_reduction": float(row["conflict_reduction"]),
                "h1_log_pp_seconds": math.log1p(
                    max(1e-9, float(outcome.get("pp_replan_seconds", 0.0)))
                ),
            }
        )
    return result


def _h3_rows(
    feature_rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    features = {
        (str(row["state_id"]), str(row["candidate_id"])): row
        for row in feature_rows
        if str(row.get("split")) == split
    }
    result = []
    for horizon in horizon_rows:
        if str(horizon.get("split")) != split:
            continue
        key = (str(horizon["state_id"]), str(horizon["candidate_id"]))
        feature = features.get(key)
        route = str(horizon.get("route") or "")
        if feature is None and route != "official_adaptive":
            raise ValueError(f"v3-h3 trial lacks candidate features: {key}")
        if feature is None:
            state_features = [
                row
                for (state_id, _candidate_id), row in features.items()
                if state_id == key[0]
            ]
            if not state_features:
                raise ValueError(f"v3-h3 Adaptive trial lacks state metadata: {key}")
            metadata = state_features[0]
            values = {name: 0.0 for name in V3_FEATURE_NAMES}
        else:
            metadata = feature
            values = dict(feature["features"]["realized_dynamic"])
        h1 = dict(horizon["h1"])
        h3 = dict(horizon["h3"])
        result.append(
            {
                "split": split,
                "state_id": key[0],
                "candidate_id": key[1],
                "map_id": str(metadata["map_id"]),
                "layout_mode": str(metadata.get("layout_mode", "unknown")),
                "agent_count": int(metadata.get("agent_count", 0)),
                "route": route or str(metadata["route"]),
                "actual_size": int(
                    horizon.get("actual_size", metadata.get("actual_size", 0))
                ),
                "base_selected": bool(
                    feature is not None and feature.get("base_selected")
                ),
                "v2_score": float(
                    feature.get("main_score", -1e30)
                    if feature is not None
                    else -1e30
                ),
                "features": [float(values.get(name, 0.0)) for name in V3_FEATURE_NAMES],
                "h1_effective_progress": int(bool(h1["effective_progress"])),
                "h1_no_progress": int(bool(h1["no_progress"])),
                "h1_conflict_reduction": float(h1["conflict_reduction"]),
                "h1_log_pp_seconds": math.log1p(
                    max(1e-9, float(h1["pp_replan_seconds"]))
                ),
                "h3_conflict_reduction": float(h3["conflict_reduction"]),
                "h3_log_total_seconds": math.log1p(
                    max(1e-9, float(h3["total_seconds"]))
                ),
                "h3_no_progress": int(bool(h3["no_progress"])),
                "h3_total_seconds": max(1e-9, float(h3["total_seconds"])),
            }
        )
    if not result:
        raise ValueError(f"v3-h3 has no horizon rows for {split}")
    return result


def _fit(
    h1_rows: list[dict[str, Any]], h3_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    models: dict[str, Any] = {}
    h1_values = np.asarray([row["features"] for row in h1_rows], dtype=float)
    h1_weights = np.asarray(balanced_trial_weights(h1_rows), dtype=float)
    for name, target in (
        ("h1_effective_progress_probability", "h1_effective_progress"),
        ("h1_no_progress_probability", "h1_no_progress"),
    ):
        estimator = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
        estimator.fit(
            h1_values,
            np.asarray([row[target] for row in h1_rows], dtype=int),
            sample_weight=h1_weights,
        )
        models[name] = estimator
    for name, target in (
        ("h1_conflict_reduction", "h1_conflict_reduction"),
        ("h1_log_pp_seconds", "h1_log_pp_seconds"),
    ):
        estimator = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
        estimator.fit(
            h1_values,
            np.asarray([row[target] for row in h1_rows], dtype=float),
            sample_weight=h1_weights,
        )
        models[name] = estimator
    model_h3_rows = [row for row in h3_rows if str(row["route"]) == "model"]
    h3_values = np.asarray([row["features"] for row in model_h3_rows], dtype=float)
    h3_weights = np.asarray(balanced_trial_weights(model_h3_rows), dtype=float)
    no_progress = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    no_progress.fit(
        h3_values,
        np.asarray([row["h3_no_progress"] for row in model_h3_rows], dtype=int),
        sample_weight=h3_weights,
    )
    models["h3_no_progress_probability"] = no_progress
    for name, target in (
        ("h3_conflict_reduction", "h3_conflict_reduction"),
        ("h3_log_total_seconds", "h3_log_total_seconds"),
    ):
        estimator = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
        estimator.fit(
            h3_values,
            np.asarray([row[target] for row in model_h3_rows], dtype=float),
            sample_weight=h3_weights,
        )
        models[name] = estimator
    return models


def _predict(models: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    import numpy as np

    values = np.asarray([row["features"] for row in rows], dtype=float)
    h1_effective = _positive_probability(
        models["h1_effective_progress_probability"], values
    )
    h1_no_progress = _positive_probability(
        models["h1_no_progress_probability"], values
    )
    h3_no_progress = _positive_probability(
        models["h3_no_progress_probability"], values
    )
    h1_reduction = [
        max(0.0, float(value))
        for value in models["h1_conflict_reduction"].predict(values)
    ]
    h1_log_pp = list(map(float, models["h1_log_pp_seconds"].predict(values)))
    h3_reduction = [
        max(0.0, float(value))
        for value in models["h3_conflict_reduction"].predict(values)
    ]
    h3_log_total = list(map(float, models["h3_log_total_seconds"].predict(values)))
    h3_total = [max(1e-9, math.expm1(min(50.0, value))) for value in h3_log_total]
    return {
        "effective_progress_probability": h1_effective,
        "no_progress_probability": h1_no_progress,
        "h1_conflict_reduction": h1_reduction,
        "h1_log_pp_seconds": h1_log_pp,
        "h1_pp_seconds": [
            max(1e-9, math.expm1(min(50.0, value))) for value in h1_log_pp
        ],
        "h3_conflict_reduction": h3_reduction,
        "h3_log_total_seconds": h3_log_total,
        "h3_total_seconds": h3_total,
        "h3_no_progress_probability": h3_no_progress,
        "utility": [
            reduction / max(1e-9, duration)
            for reduction, duration in zip(h3_reduction, h3_total)
        ],
    }


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
            predicted = predictions.get((state_id, candidate_id), {})
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
                        "no_progress_rate": statistics.fmean(
                            float(row["h3_no_progress"]) for row in trials
                        ),
                        "conflict_reduction": statistics.fmean(
                            float(row["h3_conflict_reduction"]) for row in trials
                        ),
                        "total_seconds": statistics.fmean(
                            float(row["h3_total_seconds"]) for row in trials
                        ),
                    },
                }
            )
        model = [arm for arm in arms if arm["route"] == "model"]
        adaptive = [arm for arm in arms if arm["route"] == "official_adaptive"]
        if sum(bool(arm["base_selected"]) for arm in model) != 1:
            raise ValueError(f"v3-h3 state lacks one v2 base arm: {state_id}")
        if len(adaptive) != 1:
            raise ValueError(f"v3-h3 state lacks one Adaptive arm: {state_id}")
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


def _selected(
    state: dict[str, Any], kind: str, thresholds: dict[str, float]
) -> dict[str, Any]:
    if kind == "adaptive":
        return next(arm for arm in state["arms"] if arm["route"] == "official_adaptive")
    model = [arm for arm in state["arms"] if arm["route"] == "model"]
    if kind == "v2":
        return next(arm for arm in model if bool(arm["base_selected"]))
    candidates = [{"candidate_id": arm["candidate_id"]} for arm in model]
    names = (
        "effective_progress_probability",
        "no_progress_probability",
        "h3_no_progress_probability",
        "utility",
    )
    predictions = {
        name: [float(arm["predicted"][name]) for arm in model] for name in names
    }
    order = v3_h3_candidate_order(
        candidates,
        predictions,
        [float(arm["v2_score"]) for arm in model],
        thresholds,
    )
    return model[order[0]]


def _metrics(
    states: list[dict[str, Any]], kind: str, thresholds: dict[str, float]
) -> dict[str, float]:
    selected = [_selected(state, kind, thresholds) for state in states]
    actuals = [dict(arm["actual"]) for arm in selected]
    total_seconds = math.fsum(float(row["total_seconds"]) for row in actuals)
    total_reduction = math.fsum(float(row["conflict_reduction"]) for row in actuals)
    return {
        "state_count": float(len(states)),
        "no_progress_rate": statistics.fmean(
            float(row["no_progress_rate"]) for row in actuals
        ),
        "mean_conflict_reduction": statistics.fmean(
            float(row["conflict_reduction"]) for row in actuals
        ),
        "mean_total_seconds": statistics.fmean(
            float(row["total_seconds"]) for row in actuals
        ),
        "conflict_reduction_per_total_second": total_reduction
        / max(1e-9, total_seconds),
    }


def _gate(states: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    h3 = _metrics(states, "h3", thresholds)
    v2 = _metrics(states, "v2", thresholds)
    adaptive = _metrics(states, "adaptive", thresholds)
    efficiency_ratio = h3["conflict_reduction_per_total_second"] / max(
        1e-9, v2["conflict_reduction_per_total_second"]
    )
    reduction_ratio = h3["mean_conflict_reduction"] / max(
        1e-9, v2["mean_conflict_reduction"]
    )
    cells = []
    for layout, agents in sorted(
        {(str(row["layout_mode"]), int(row["agent_count"])) for row in states}
    ):
        selected = [
            row
            for row in states
            if str(row["layout_mode"]) == layout and int(row["agent_count"]) == agents
        ]
        h3_cell = _metrics(selected, "h3", thresholds)
        v2_cell = _metrics(selected, "v2", thresholds)
        ratio = h3_cell["conflict_reduction_per_total_second"] / max(
            1e-9, v2_cell["conflict_reduction_per_total_second"]
        )
        cells.append({"layout_mode": layout, "agent_count": agents, "ratio": ratio})
    checks = {
        "efficiency_at_least_10pct_better": efficiency_ratio + 1e-12 >= 1.10,
        "no_progress_not_up_1pct": h3["no_progress_rate"]
        <= v2["no_progress_rate"] + 0.01 + 1e-12,
        "reduction_retention_at_least_95pct": reduction_ratio + 1e-12 >= 0.95,
        "five_of_six_cells_noninferior": len(cells) == 6
        and sum(float(row["ratio"]) + 1e-12 >= 1.0 for row in cells) >= 5,
        "worst_cell_at_least_90pct": bool(cells)
        and min(float(row["ratio"]) for row in cells) + 1e-12 >= 0.90,
    }
    return {
        "h3": h3,
        "v2": v2,
        "adaptive": adaptive,
        "efficiency_ratio": efficiency_ratio,
        "conflict_reduction_ratio": reduction_ratio,
        "cells": cells,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _calibrate(states: list[dict[str, Any]]) -> dict[str, Any]:
    grid = []
    for effective in EFFECTIVE_TOLERANCE_GRID:
        for no_progress in NO_PROGRESS_TOLERANCE_GRID:
            for improvement in UTILITY_IMPROVEMENT_GRID:
                thresholds = {
                    "h1_effective_probability_tolerance": effective,
                    "h1_no_progress_probability_tolerance": no_progress,
                    "minimum_h3_utility_improvement": improvement,
                }
                grid.append({"thresholds": thresholds, "gate": _gate(states, thresholds)})
    passing = [row for row in grid if bool(row["gate"]["passed"])]
    selected = max(
        passing or grid,
        key=lambda row: (
            float(row["gate"]["h3"]["conflict_reduction_per_total_second"]),
            float(row["gate"]["conflict_reduction_ratio"]),
            -float(row["thresholds"]["minimum_h3_utility_improvement"]),
        ),
    )
    return {"grid": grid, "selected": selected, "passed": bool(passing)}


def _portable_values(name: str, values: list[float]) -> list[float]:
    if name.endswith("conflict_reduction"):
        return [max(0.0, float(value)) for value in values]
    return list(map(float, values))


def train_v3_horizon_controller(
    *,
    feature_index: str | Path,
    one_step_manifest: str | Path,
    horizon_manifest: str | Path,
    controller_bundle: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    feature_path = Path(feature_index).resolve()
    one_step_path = Path(one_step_manifest).resolve()
    horizon_path = Path(horizon_manifest).resolve()
    controller_path = Path(controller_bundle).resolve()
    output_root = Path(output).resolve()
    features = _read_jsonl(feature_path)
    one_step = _read_jsonl(one_step_path)
    horizon = _read_jsonl(horizon_path)
    train_h1 = _h1_rows(features, one_step, "policy_train")
    diagnostic_h1 = _h1_rows(features, one_step, "policy_validation")
    train_h3 = _h3_rows(features, horizon, "policy_train")
    diagnostic_h3 = _h3_rows(features, horizon, "policy_validation")
    train_maps = {str(row["map_id"]) for row in train_h3}
    diagnostic_maps = {str(row["map_id"]) for row in diagnostic_h3}
    if train_maps & diagnostic_maps:
        raise ValueError("v3-h3 train and diagnostic maps overlap")
    oof_index: dict[tuple[str, str], dict[str, list[float]]] = {}
    fold_reports = []
    for fold in _balanced_map_folds(train_h3):
        held_maps = set(fold["validation_maps"])
        models = _fit(
            [row for row in train_h1 if row["map_id"] not in held_maps],
            [row for row in train_h3 if row["map_id"] not in held_maps],
        )
        held = [
            row
            for row in train_h3
            if row["map_id"] in held_maps and row["route"] == "model"
        ]
        indexed = _index_predictions(held, _predict(models, held))
        if set(indexed) & set(oof_index):
            raise ValueError("v3-h3 OOF predictions overlap")
        oof_index.update(indexed)
        fold_reports.append({**fold, "validation_trial_count": len(held)})
    train_states = _states(train_h3, oof_index)
    calibration = _calibrate(train_states)
    thresholds = dict(calibration["selected"]["thresholds"])
    models = _fit(train_h1, train_h3)
    diagnostic_model = [row for row in diagnostic_h3 if row["route"] == "model"]
    diagnostic_predictions = _predict(models, diagnostic_model)
    diagnostic_states = _states(
        diagnostic_h3,
        _index_predictions(diagnostic_model, diagnostic_predictions),
    )
    diagnostic_gate = _gate(diagnostic_states, thresholds)
    main_bundle = load_controller_bundle(controller_path)
    source_fingerprint = _fingerprint(
        {
            "feature_index_sha256": sha256_file(feature_path),
            "one_step_manifest_sha256": sha256_file(one_step_path),
            "horizon_manifest_sha256": sha256_file(horizon_path),
            "main_ranker_semantic_fingerprint": main_bundle.manifest[
                "main_ranker_semantic_fingerprint"
            ],
            "feature_schema_sha256": V3_FEATURE_SCHEMA_SHA256,
            "model_parameters": MODEL_PARAMETERS,
            "threshold_grid": {
                "effective": EFFECTIVE_TOLERANCE_GRID,
                "no_progress": NO_PROGRESS_TOLERANCE_GRID,
                "utility_improvement": UTILITY_IMPROVEMENT_GRID,
            },
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    dense = [
        {
            "feature_profile": "realized_dynamic",
            "feature_names": V3_FEATURE_NAMES,
            "feature_values": tuple(row["features"]),
        }
        for row in diagnostic_model
    ]
    raw_predictions = _predict(models, diagnostic_model)
    model_rows = {}
    parity = {}
    native_available = True
    target_prediction_names = {
        "h1_effective_progress_probability": "effective_progress_probability",
        "h1_no_progress_probability": "no_progress_probability",
        "h1_conflict_reduction": "h1_conflict_reduction",
        "h1_log_pp_seconds": "h1_log_pp_seconds",
        "h3_conflict_reduction": "h3_conflict_reduction",
        "h3_log_total_seconds": "h3_log_total_seconds",
        "h3_no_progress_probability": "h3_no_progress_probability",
    }
    for name, estimator in models.items():
        payload = _portable_payload(name, estimator, list(V3_FEATURE_NAMES), source_fingerprint)
        path = output_root / f"v3_h3__{name}.json"
        _write_json(path, payload)
        portable = load_portable_scalar_model(payload)
        python_portable: PortableScalarModel = replace(portable, native_predictor=None)
        python_values = _portable_values(name, python_portable.predict(dense))
        portable_values = _portable_values(name, portable.predict(dense))
        reference = raw_predictions[target_prediction_names[name]]
        sklearn_delta = max(
            (abs(float(left) - float(right)) for left, right in zip(reference, python_values)),
            default=0.0,
        )
        native_delta = max(
            (abs(float(left) - float(right)) for left, right in zip(portable_values, python_values)),
            default=0.0,
        )
        if sklearn_delta > 1e-12 or native_delta > 1e-12:
            raise ValueError(f"v3-h3 portable parity failed for {name}")
        native = portable.inference_backend == "native-portable-tree"
        native_available = native_available and native
        parity[name] = {
            "sklearn_python_maximum_delta": sklearn_delta,
            "native_python_maximum_delta": native_delta,
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
        "portable_parity": parity_maximum <= 1e-12,
        "native_available": native_available,
    }
    pilot_passed = all(pilot_checks.values())
    report = {
        "schema": V3_H3_TRAINING_SCHEMA,
        "pilot_gate_schema": V3_H3_GATE_SCHEMA,
        "training_labels_seen": ["policy_train"],
        "formal_or_movingai_labels_seen": False,
        "feature_schema_id": V3_FEATURE_SCHEMA_ID,
        "feature_schema_sha256": V3_FEATURE_SCHEMA_SHA256,
        "feature_count": len(V3_FEATURE_NAMES),
        "training_state_count": len(train_states),
        "diagnostic_state_count": len(diagnostic_states),
        "folds": fold_reports,
        "calibration": calibration,
        "diagnostic_gate": diagnostic_gate,
        "portable_parity": parity,
        "portable_maximum_delta": parity_maximum,
        "native_available": native_available,
        "native_audit_completed": native_available,
        "pilot_checks": pilot_checks,
        "pilot_passed": pilot_passed,
        "decision": "v3_h3_pilot_passed" if pilot_passed else "v3_h3_pilot_failed",
    }
    report_path = output_root / "training_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema": V3_H3_BUNDLE_SCHEMA,
        "schema_version": 1,
        "pilot_gate_schema": V3_H3_GATE_SCHEMA,
        "feature_schema_id": V3_FEATURE_SCHEMA_ID,
        "feature_schema_sha256": V3_FEATURE_SCHEMA_SHA256,
        "profile": "realized_dynamic",
        "feature_names": list(V3_FEATURE_NAMES),
        "models": model_rows,
        "thresholds": thresholds,
        "selection_overhead_seconds": 0.0,
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
    load_v3_controller_bundle(output_root)
    return {**report, "manifest": manifest}


def finalize_v3_horizon_native_audit(
    *,
    feature_index: str | Path,
    horizon_manifest: str | Path,
    controller_output: str | Path,
) -> dict[str, Any]:
    feature_path = Path(feature_index).resolve()
    horizon_path = Path(horizon_manifest).resolve()
    output_root = Path(controller_output).resolve()
    report_path = output_root / "training_report.json"
    manifest_path = output_root / "v3_manifest.json"
    report = dict(_read_json(report_path))
    manifest = dict(_read_json(manifest_path))
    diagnostic = [
        row
        for row in _h3_rows(
            _read_jsonl(feature_path),
            _read_jsonl(horizon_path),
            "policy_validation",
        )
        if str(row["route"]) == "model"
    ]
    dense = [
        {
            "feature_profile": "realized_dynamic",
            "feature_names": V3_FEATURE_NAMES,
            "feature_values": tuple(row["features"]),
        }
        for row in diagnostic
    ]
    parity = dict(report.get("portable_parity") or {})
    native_available = True
    for name, raw in dict(manifest.get("models") or {}).items():
        model_path = output_root / str(dict(raw)["file"])
        portable = load_portable_scalar_model(dict(_read_json(model_path)))
        python_portable: PortableScalarModel = replace(
            portable, native_predictor=None
        )
        python_values = _portable_values(str(name), python_portable.predict(dense))
        native_values = _portable_values(str(name), portable.predict(dense))
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
            raise RuntimeError(f"v3-h3 native predictor is unavailable: {name}")
        if native_delta > 1e-12:
            raise ValueError(f"v3-h3 native model differs from Python: {name}")
        parity[name] = {
            **dict(parity.get(name) or {}),
            "native_python_maximum_delta": native_delta,
            "inference_backend": portable.inference_backend,
        }
        manifest["models"][name]["inference_backend"] = portable.inference_backend
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
        "portable_parity": parity_maximum <= 1e-12,
        "native_available": native_available,
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
                "v3_h3_pilot_passed" if pilot_passed else "v3_h3_pilot_failed"
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
    "finalize_v3_horizon_native_audit",
    "train_v3_horizon_controller",
    "V3_H3_TRAINING_SCHEMA",
]
