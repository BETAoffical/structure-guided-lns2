from __future__ import annotations

import collections
import math
import statistics
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import (
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    PROFILE_FEATURE_NAMES,
)
from experiments.repair_aware import (
    REPAIR_AWARE_BUNDLE_SCHEMA,
    load_portable_scalar_model,
)
from experiments.repair_aware_training import (
    MODEL_PARAMETERS,
    _balanced_map_folds,
    _portable_payload,
)
from experiments.repair_collection import _fingerprint, _read_jsonl, _write_json


HIGH_LOAD_RESCUE_TRAINING_SCHEMA = "lns2.high_load_rescue_training.v1"
MAX_RESCUE_GRID = (1, 2, 3)
ADAPTIVE_MARGIN_GRID = (0.0, 0.05, 0.10, 0.20)
MINIMUM_EFFICIENCY_QUANTILES = (0.0, 0.10, 0.25)


def _training_rows(
    feature_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    split: str,
    feature_names: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = {
        (str(row["state_id"]), str(row["candidate_id"])): row
        for row in feature_rows
        if str(row.get("split")) == split
    }
    rows = []
    for trial in trial_rows:
        if str(trial.get("split")) != split:
            continue
        if str(trial.get("status")) not in {"ok", "resumed"} or not bool(
            trial.get("complete")
        ):
            raise ValueError("high-load rescue data contains an incomplete trial")
        key = (str(trial["state_id"]), str(trial["candidate_id"]))
        candidate = candidates.get(key)
        if candidate is None:
            raise ValueError(f"high-load rescue feature/trial join failed: {key}")
        outcome = dict(trial["outcome"])
        before = int(outcome["conflicts_before"])
        after = int(outcome["conflicts_after"])
        repair_seconds = max(1e-9, float(outcome["repair_seconds"]))
        features = dict(candidate["features"]["realized_dynamic"])
        rows.append(
            {
                "split": split,
                "state_id": key[0],
                "candidate_id": key[1],
                "map_id": str(candidate["map_id"]),
                "layout_mode": str(candidate.get("layout_mode", "unknown")),
                "route": str(candidate["route"]),
                "actual_size": int(candidate["actual_size"]),
                "base_selected": bool(candidate.get("base_selected")),
                "main_score": float(candidate.get("main_score", -1e30)),
                "features": [float(features.get(name, 0.0)) for name in feature_names],
                "progress": int(after < before),
                "reduction": float(max(0, before - after)),
                "repair_seconds": repair_seconds,
                "log_repair_seconds": math.log1p(repair_seconds),
                "hard_failure": int(bool(outcome.get("hard_failure"))),
            }
        )
    expected = set(candidates)
    actual = {(str(row["state_id"]), str(row["candidate_id"])) for row in rows}
    if not rows or expected != actual:
        raise ValueError("high-load rescue data lacks complete candidate coverage")
    return list(candidates.values()), rows


def _weights(rows: list[dict[str, Any]]) -> list[float]:
    states_by_map: dict[str, set[str]] = collections.defaultdict(set)
    arms_by_state: dict[str, set[str]] = collections.defaultdict(set)
    trials_by_arm: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in rows:
        states_by_map[str(row["map_id"])].add(str(row["state_id"]))
        arms_by_state[str(row["state_id"])].add(str(row["candidate_id"]))
        trials_by_arm[(str(row["state_id"]), str(row["candidate_id"]))] += 1
    raw = []
    for row in rows:
        state = str(row["state_id"])
        arm = str(row["candidate_id"])
        map_id = str(row["map_id"])
        raw.append(
            1.0
            / max(
                1,
                len(states_by_map)
                * len(states_by_map[map_id])
                * len(arms_by_state[state])
                * trials_by_arm[(state, arm)],
            )
        )
    scale = len(rows) / math.fsum(raw)
    return [value * scale for value in raw]


def _fit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    values = np.asarray([row["features"] for row in rows], dtype=float)
    weights = np.asarray(_weights(rows), dtype=float)
    progress = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    progress.fit(values, np.asarray([row["progress"] for row in rows]), sample_weight=weights)
    failure = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    failure.fit(values, np.asarray([row["hard_failure"] for row in rows]), sample_weight=weights)
    reduction = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
    reduction.fit(values, np.asarray([row["reduction"] for row in rows]), sample_weight=weights)
    seconds = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
    seconds.fit(values, np.asarray([row["log_repair_seconds"] for row in rows]), sample_weight=weights)
    return {
        "progress_probability": progress,
        "conflict_reduction": reduction,
        "log_repair_seconds": seconds,
        "hard_failure_probability": failure,
    }


def _positive_probability(estimator: Any, values: Any) -> list[float]:
    classes = list(map(int, estimator.classes_))
    probabilities = estimator.predict_proba(values)
    if 1 not in classes:
        return [0.0] * len(values)
    return list(map(float, probabilities[:, classes.index(1)]))


def _predict(models: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    import numpy as np

    values = np.asarray([row["features"] for row in rows], dtype=float)
    progress = _positive_probability(models["progress_probability"], values)
    hard_failure = _positive_probability(models["hard_failure_probability"], values)
    reduction = [max(0.0, float(value)) for value in models["conflict_reduction"].predict(values)]
    log_seconds = list(map(float, models["log_repair_seconds"].predict(values)))
    seconds = [max(1e-9, math.expm1(min(value, 50.0))) for value in log_seconds]
    efficiency = [
        p * max(0.0, 1.0 - failure) * delta / duration
        for p, failure, delta, duration in zip(progress, hard_failure, reduction, seconds)
    ]
    return {
        "progress_probability": progress,
        "conflict_reduction": reduction,
        "log_repair_seconds": log_seconds,
        "repair_seconds": seconds,
        "hard_failure_probability": hard_failure,
        "efficiency": efficiency,
    }


def _prediction_index(
    rows: list[dict[str, Any]], predictions: dict[str, list[float]]
) -> dict[tuple[str, str], dict[str, list[float]]]:
    result: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for index, row in enumerate(rows):
        key = (str(row["state_id"]), str(row["candidate_id"]))
        for name in (
            "progress_probability",
            "conflict_reduction",
            "repair_seconds",
            "hard_failure_probability",
            "efficiency",
        ):
            result[key][name].append(float(predictions[name][index]))
    return {key: dict(values) for key, values in result.items()}


def _states(
    rows: list[dict[str, Any]],
    predictions: dict[tuple[str, str], dict[str, list[float]]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    result = []
    for state_id, trials in grouped.items():
        by_arm: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in trials:
            by_arm[str(row["candidate_id"])].append(row)
        arms = []
        for candidate_id, arm_trials in by_arm.items():
            first = arm_trials[0]
            predicted = predictions[(state_id, candidate_id)]
            arms.append(
                {
                    "candidate_id": candidate_id,
                    "route": str(first["route"]),
                    "actual_size": int(first["actual_size"]),
                    "base_selected": bool(first["base_selected"]),
                    "main_score": float(first["main_score"]),
                    "predicted": {
                        name: statistics.fmean(values)
                        for name, values in predicted.items()
                    },
                    "actual": {
                        "progress_rate": statistics.fmean(float(row["progress"]) for row in arm_trials),
                        "reduction": statistics.fmean(float(row["reduction"]) for row in arm_trials),
                        "repair_seconds": statistics.fmean(float(row["repair_seconds"]) for row in arm_trials),
                        "hard_failure_rate": statistics.fmean(float(row["hard_failure"]) for row in arm_trials),
                    },
                }
            )
        if sum(bool(arm["base_selected"]) for arm in arms) != 1:
            raise ValueError(f"state does not identify exactly one v2 base arm: {state_id}")
        if sum(arm["route"] == "official_adaptive" for arm in arms) != 1:
            raise ValueError(f"state does not identify exactly one Adaptive arm: {state_id}")
        result.append(
            {
                "state_id": state_id,
                "map_id": str(trials[0]["map_id"]),
                "arms": arms,
            }
        )
    return result


def _metrics(selected: list[dict[str, Any]]) -> dict[str, float]:
    if not selected:
        return {
            "state_count": 0.0,
            "progress_rate": 0.0,
            "mean_reduction": 0.0,
            "mean_repair_seconds": 0.0,
            "conflict_reduction_per_second": 0.0,
            "hard_failure_rate": 0.0,
            "size12_fraction": 0.0,
            "adaptive_fraction": 0.0,
        }
    actuals = [arm["actual"] for arm in selected]
    mean_seconds = statistics.fmean(float(row["repair_seconds"]) for row in actuals)
    mean_reduction = statistics.fmean(float(row["reduction"]) for row in actuals)
    return {
        "state_count": float(len(selected)),
        "progress_rate": statistics.fmean(float(row["progress_rate"]) for row in actuals),
        "mean_reduction": mean_reduction,
        "mean_repair_seconds": mean_seconds,
        "conflict_reduction_per_second": mean_reduction / max(1e-9, mean_seconds),
        "hard_failure_rate": statistics.fmean(float(row["hard_failure_rate"]) for row in actuals),
        "size12_fraction": statistics.fmean(float(arm["actual_size"] == 12) for arm in selected),
        "adaptive_fraction": statistics.fmean(float(arm["route"] == "official_adaptive") for arm in selected),
    }


def _choose(
    state: dict[str, Any],
    *,
    margin: float,
    minimum_efficiency: float,
    allow_size12: bool,
) -> dict[str, Any]:
    model = [
        arm
        for arm in state["arms"]
        if arm["route"] == "model" and (allow_size12 or arm["actual_size"] != 12)
    ]
    adaptive = next(arm for arm in state["arms"] if arm["route"] == "official_adaptive")
    best = max(
        model,
        key=lambda arm: (
            float(arm["predicted"]["efficiency"]),
            float(arm["predicted"]["progress_probability"]),
            -float(arm["predicted"]["hard_failure_probability"]),
            float(arm["main_score"]),
            str(arm["candidate_id"]),
        ),
    )
    model_efficiency = float(best["predicted"]["efficiency"])
    adaptive_efficiency = float(adaptive["predicted"]["efficiency"])
    return (
        adaptive
        if model_efficiency < minimum_efficiency
        or adaptive_efficiency > model_efficiency * (1.0 + margin)
        else best
    )


def _calibrate_thresholds(states: list[dict[str, Any]]) -> dict[str, Any]:
    adaptive_metrics = _metrics(
        [next(arm for arm in state["arms"] if arm["route"] == "official_adaptive") for state in states]
    )
    efficiencies = sorted(
        max(
            float(arm["predicted"]["efficiency"])
            for arm in state["arms"]
            if arm["route"] == "model"
        )
        for state in states
    )
    minimums = {0.0}
    for quantile in MINIMUM_EFFICIENCY_QUANTILES[1:]:
        index = min(len(efficiencies) - 1, round(quantile * (len(efficiencies) - 1)))
        minimums.add(float(efficiencies[index]))
    grid = []
    for minimum in sorted(minimums):
        for margin in ADAPTIVE_MARGIN_GRID:
            selected = [
                _choose(state, margin=margin, minimum_efficiency=minimum, allow_size12=True)
                for state in states
            ]
            metrics = _metrics(selected)
            metrics.update({"minimum_predicted_efficiency": minimum, "adaptive_efficiency_margin": margin})
            metrics["passed"] = metrics["progress_rate"] + 1e-12 >= adaptive_metrics["progress_rate"] - 0.01
            grid.append(metrics)
    passing = [row for row in grid if row["passed"]]
    selected = max(
        passing or grid,
        key=lambda row: (
            row["conflict_reduction_per_second"],
            row["progress_rate"],
            -row["adaptive_fraction"],
            -row["adaptive_efficiency_margin"],
        ),
    )
    return {
        "adaptive_baseline": adaptive_metrics,
        "grid": grid,
        "selected": selected,
        "passed": bool(passing),
    }


def _max_rescue_calibration(states: list[dict[str, Any]]) -> dict[str, Any]:
    reports = []
    for maximum in MAX_RESCUE_GRID:
        state_values = []
        for state in states:
            adaptive = next(arm for arm in state["arms"] if arm["route"] == "official_adaptive")
            ordered = sorted(
                (arm for arm in state["arms"] if arm["route"] == "model"),
                key=lambda arm: -float(arm["predicted"]["efficiency"]),
            )[:maximum]
            remaining = 1.0
            expected_seconds = 0.0
            expected_reduction = 0.0
            for arm in ordered + [adaptive]:
                actual = arm["actual"]
                expected_seconds += remaining * float(actual["repair_seconds"])
                expected_reduction += remaining * float(actual["reduction"])
                remaining *= 1.0 - float(actual["progress_rate"])
            state_values.append((expected_reduction, expected_seconds, 1.0 - remaining))
        mean_reduction = statistics.fmean(value[0] for value in state_values)
        mean_seconds = statistics.fmean(value[1] for value in state_values)
        reports.append(
            {
                "max_model_rescues": maximum,
                "expected_progress_rate": statistics.fmean(value[2] for value in state_values),
                "expected_reduction": mean_reduction,
                "expected_seconds": mean_seconds,
                "expected_reduction_per_second": mean_reduction / max(1e-9, mean_seconds),
            }
        )
    best_efficiency = max(row["expected_reduction_per_second"] for row in reports)
    eligible = [row for row in reports if row["expected_reduction_per_second"] >= best_efficiency - 1e-12]
    selected = min(eligible, key=lambda row: row["max_model_rescues"])
    return {"grid": reports, "selected": selected}


def _size12_report(
    states: list[dict[str, Any]], thresholds: dict[str, float]
) -> dict[str, Any]:
    with_size12 = [
        _choose(state, margin=thresholds["adaptive_efficiency_margin"], minimum_efficiency=thresholds["minimum_predicted_efficiency"], allow_size12=True)
        for state in states
    ]
    without_size12 = [
        _choose(state, margin=thresholds["adaptive_efficiency_margin"], minimum_efficiency=thresholds["minimum_predicted_efficiency"], allow_size12=False)
        for state in states
    ]
    left = _metrics(with_size12)
    right = _metrics(without_size12)
    improvement = left["conflict_reduction_per_second"] / max(1e-9, right["conflict_reduction_per_second"]) - 1.0
    progress_delta = left["progress_rate"] - right["progress_rate"]
    return {
        "with_size12": left,
        "without_size12": right,
        "efficiency_improvement_fraction": improvement,
        "progress_rate_delta": progress_delta,
        "size12_selected_fraction": left["size12_fraction"],
        "offline_passed": improvement + 1e-12 >= 0.02 and progress_delta >= -0.01 - 1e-12 and left["size12_fraction"] + 1e-12 >= 0.05,
        "selection_overhead_gate": "deferred_to_complete_episode_runtime",
    }


def _prediction_metrics(rows: list[dict[str, Any]], predictions: dict[str, list[float]]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    progress = [int(row["progress"]) for row in rows]
    failure = [int(row["hard_failure"]) for row in rows]
    return {
        "trial_count": float(len(rows)),
        "progress_auc": float(roc_auc_score(progress, predictions["progress_probability"])) if len(set(progress)) > 1 else 0.5,
        "hard_failure_auc": float(roc_auc_score(failure, predictions["hard_failure_probability"])) if len(set(failure)) > 1 else 0.5,
        "reduction_mae": statistics.fmean(abs(float(row["reduction"]) - value) for row, value in zip(rows, predictions["conflict_reduction"])),
        "repair_log_mae": statistics.fmean(abs(float(row["log_repair_seconds"]) - value) for row, value in zip(rows, predictions["log_repair_seconds"])),
    }


def train_high_load_rescue_controller(
    *,
    feature_index: str | Path,
    trial_manifest: str | Path,
    controller_bundle: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    feature_path = Path(feature_index).resolve()
    trial_path = Path(trial_manifest).resolve()
    controller_path = Path(controller_bundle).resolve()
    output_root = Path(output).resolve()
    feature_rows = _read_jsonl(feature_path)
    trial_rows = _read_jsonl(trial_path)
    feature_names = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    _train_candidates, train_rows = _training_rows(feature_rows, trial_rows, "policy_train", feature_names)
    _validation_candidates, validation_rows = _training_rows(feature_rows, trial_rows, "policy_validation", feature_names)
    train_maps = {str(row["map_id"]) for row in train_rows}
    validation_maps = {str(row["map_id"]) for row in validation_rows}
    if train_maps & validation_maps:
        raise ValueError("high-load train and locked validation maps overlap")
    folds = _balanced_map_folds(train_rows)
    oof: dict[tuple[str, str], dict[str, list[float]]] = {}
    fold_reports = []
    for fold in folds:
        training = [row for row in train_rows if row["map_id"] in set(fold["train_maps"])]
        held_out = [row for row in train_rows if row["map_id"] in set(fold["validation_maps"])]
        models = _fit(training)
        predicted = _predict(models, held_out)
        indexed = _prediction_index(held_out, predicted)
        if set(oof) & set(indexed):
            raise ValueError("high-load OOF predictions overlap")
        oof.update(indexed)
        fold_reports.append({**fold, "training_trial_count": len(training), "validation_trial_count": len(held_out), "prediction_metrics": _prediction_metrics(held_out, predicted)})
    train_states = _states(train_rows, oof)
    threshold_calibration = _calibrate_thresholds(train_states)
    max_rescue_calibration = _max_rescue_calibration(train_states)
    thresholds = {
        "minimum_predicted_efficiency": float(threshold_calibration["selected"]["minimum_predicted_efficiency"]),
        "adaptive_efficiency_margin": float(threshold_calibration["selected"]["adaptive_efficiency_margin"]),
    }
    train_size12 = _size12_report(train_states, thresholds)

    final_models = _fit(train_rows)
    validation_predictions = _predict(final_models, validation_rows)
    validation_states = _states(validation_rows, _prediction_index(validation_rows, validation_predictions))
    validation_selected = [
        _choose(state, margin=thresholds["adaptive_efficiency_margin"], minimum_efficiency=thresholds["minimum_predicted_efficiency"], allow_size12=True)
        for state in validation_states
    ]
    validation_metrics = _metrics(validation_selected)
    validation_size12 = _size12_report(validation_states, thresholds)
    size12_promoted = bool(train_size12["offline_passed"] and validation_size12["offline_passed"])

    main_bundle = load_controller_bundle(controller_path)
    source_fingerprint = _fingerprint(
        {
            "feature_index_sha256": sha256_file(feature_path),
            "trial_manifest_sha256": sha256_file(trial_path),
            "controller_semantic_fingerprint": main_bundle.manifest["main_ranker_semantic_fingerprint"],
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "parameters": MODEL_PARAMETERS,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    model_rows = {}
    portable_deltas = {}
    dense_validation = [
        {"feature_profile": "realized_dynamic", "feature_names": feature_names, "feature_values": tuple(row["features"])}
        for row in validation_rows
    ]
    for name, estimator in final_models.items():
        payload = _portable_payload(name, estimator, list(feature_names), source_fingerprint)
        path = output_root / f"repair_aware__{name}.json"
        _write_json(path, payload)
        portable = load_portable_scalar_model(payload)
        values = portable.predict(dense_validation)
        reference_name = name
        reference = validation_predictions[reference_name]
        maximum_delta = max((abs(float(left) - float(right)) for left, right in zip(reference, values)), default=0.0)
        if maximum_delta > 1e-12:
            raise ValueError(f"portable high-load model differs from sklearn: {name} {maximum_delta}")
        portable_deltas[name] = maximum_delta
        model_rows[name] = {
            "file": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
            "semantic_fingerprint": payload["semantic_fingerprint"],
            "tree_count": len(payload["trees"]),
            "inference_backend": portable.inference_backend,
        }
    selected_max = int(max_rescue_calibration["selected"]["max_model_rescues"])
    report = {
        "schema": HIGH_LOAD_RESCUE_TRAINING_SCHEMA,
        "schema_version": 1,
        "training_labels_seen": ["policy_train"],
        "formal_or_movingai_labels_seen": False,
        "validation_locked_once": True,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_count": len(feature_names),
        "training_state_count": len(train_states),
        "validation_state_count": len(validation_states),
        "training_agent_counts": sorted({int(row["features"][feature_names.index("state.agent_count")]) for row in train_rows}),
        "folds": fold_reports,
        "threshold_calibration": threshold_calibration,
        "max_rescue_calibration": max_rescue_calibration,
        "selected_max_model_rescues": selected_max,
        "training_size12": train_size12,
        "validation_size12": validation_size12,
        "size12_promoted_offline": size12_promoted,
        "validation_selection": validation_metrics,
        "validation_prediction_metrics": _prediction_metrics(validation_rows, validation_predictions),
        "portable_maximum_deltas": portable_deltas,
    }
    report_path = output_root / "training_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema": REPAIR_AWARE_BUNDLE_SCHEMA,
        "schema_version": 2,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "profile": "realized_dynamic",
        "feature_names": list(feature_names),
        "main_ranker_semantic_fingerprint": main_bundle.manifest["main_ranker_semantic_fingerprint"],
        "models": model_rows,
        "thresholds": thresholds,
        "selected_max_model_rescues": selected_max,
        "size12_promoted_offline": size12_promoted,
        "guarded_tiebreak_eligible": False,
        "training_report": {"file": report_path.relative_to(output_root).as_posix(), "sha256": sha256_file(report_path)},
        "source_fingerprint": source_fingerprint,
    }
    _write_json(output_root / "repair_aware_manifest.json", manifest)
    return {**report, "manifest": manifest}


__all__ = [
    "HIGH_LOAD_RESCUE_TRAINING_SCHEMA",
    "train_high_load_rescue_controller",
]
