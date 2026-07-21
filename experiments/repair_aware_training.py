from __future__ import annotations

import collections
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file
from experiments.closed_loop_confirmation import score_online_candidates
from experiments.compact_controller_model import load_controller_bundle
from experiments.feature_schema_v2 import FEATURE_SCHEMA_ID, FEATURE_SCHEMA_SHA256
from experiments.repair_aware import (
    LEGACY_REPAIR_AWARE_BUNDLE_SCHEMA,
    PORTABLE_SCALAR_MODEL_SCHEMA,
    guarded_tiebreak_candidate,
    load_portable_scalar_model,
    repair_aware_order,
)
from experiments.repair_collection import _fingerprint, _read_jsonl, _write_json


REPAIR_AWARE_TRAINING_SCHEMA = "lns2.repair_aware_training.v1"
MODEL_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": 20260721,
    "early_stopping": False,
}
RESCUE_TOLERANCE_GRID = (0.0, 0.025, 0.05, 0.10, 0.15, 0.20)
TIE_SCORE_GAP_GRID = (0.0, 0.01, 0.02, 0.05, 0.10)
TIE_PROGRESS_MARGIN_GRID = (0.0, 0.025, 0.05, 0.10)
TIE_MINIMUM_REDUCTION_RATIO = 0.98
TIE_MAXIMUM_GENERATED_RATIO = 0.90


def _grouped(
    rows: Iterable[dict[str, Any]], key: str
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        result[str(row[key])].append(row)
    return dict(result)


def _balanced_map_folds(rows: list[dict[str, Any]], count: int = 4) -> list[dict[str, Any]]:
    map_layout: dict[str, str] = {}
    for row in rows:
        map_id = str(row["map_id"])
        layout = str(row.get("layout_mode", "unknown"))
        previous = map_layout.setdefault(map_id, layout)
        if previous != layout:
            raise ValueError(f"map {map_id} has inconsistent layouts")
    if len(map_layout) < count:
        raise ValueError("repair-aware OOF requires at least four training maps")
    by_layout: dict[str, list[str]] = collections.defaultdict(list)
    for map_id, layout in map_layout.items():
        by_layout[layout].append(map_id)
    validation: list[list[str]] = [[] for _ in range(count)]
    for layout in sorted(by_layout):
        for index, map_id in enumerate(sorted(by_layout[layout])):
            validation[index % count].append(map_id)
    all_maps = set(map_layout)
    folds = []
    for index, maps in enumerate(validation):
        if not maps:
            raise ValueError("repair-aware OOF produced an empty validation fold")
        folds.append(
            {
                "fold": index,
                "train_maps": sorted(all_maps - set(maps)),
                "validation_maps": sorted(maps),
            }
        )
    return folds


def _training_rows(
    feature_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    split: str,
    feature_names: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = {
        (str(row["state_id"]), str(row["candidate_id"])): row
        for row in feature_rows
        if str(row.get("split")) == split
    }
    trials = []
    unknown = []
    for manifest in trial_rows:
        if str(manifest.get("split")) != split:
            continue
        if str(manifest.get("status")) not in {"ok", "resumed"} or not bool(
            manifest.get("complete")
        ):
            raise ValueError(f"incomplete repair-aware training trial: {manifest.get('job_id')}")
        outcome = dict(manifest["outcome"])
        key = (str(manifest["state_id"]), str(manifest["candidate_id"]))
        candidate = candidates.get(key)
        if candidate is None:
            unknown.append(key)
            continue
        features = dict(candidate["features"]["realized_dynamic"])
        before = int(outcome["conflicts_before"])
        after = int(outcome["conflicts_after"])
        generated = int(outcome["generated"])
        trials.append(
            {
                "split": split,
                "state_id": key[0],
                "candidate_id": key[1],
                "candidate_key": str(candidate["candidate_key"]),
                "map_id": str(candidate["map_id"]),
                "layout_mode": str(candidate.get("layout_mode", "unknown")),
                # Sparse one-hot values in the historical index have the same
                # zero-default semantics as the deployed v2 ranker.
                "features": [float(features.get(name, 0.0)) for name in feature_names],
                "progress": int(after < before),
                "reduction": float(max(0, before - after)),
                "log_generated": math.log1p(max(0, generated)),
                "generated": float(max(0, generated)),
            }
        )
    if unknown:
        raise ValueError(f"repair-aware trial/feature join failed for {len(unknown)} rows")
    if not trials or set(candidates) != {
        (str(row["state_id"]), str(row["candidate_id"])) for row in trials
    }:
        raise ValueError("repair-aware training data has incomplete candidate coverage")
    return list(candidates.values()), trials


def _balanced_weights(rows: list[dict[str, Any]]) -> list[float]:
    states_by_map: dict[str, set[str]] = collections.defaultdict(set)
    candidates_by_state: dict[str, set[str]] = collections.defaultdict(set)
    trials_by_candidate: collections.Counter[tuple[str, str]] = collections.Counter()
    for row in rows:
        states_by_map[str(row["map_id"])].add(str(row["state_id"]))
        candidates_by_state[str(row["state_id"])].add(str(row["candidate_id"]))
        trials_by_candidate[(str(row["state_id"]), str(row["candidate_id"]))] += 1
    map_count = len(states_by_map)
    raw = []
    for row in rows:
        state = str(row["state_id"])
        candidate = str(row["candidate_id"])
        map_id = str(row["map_id"])
        raw.append(
            1.0
            / (
                map_count
                * len(states_by_map[map_id])
                * len(candidates_by_state[state])
                * trials_by_candidate[(state, candidate)]
            )
        )
    scale = len(rows) / math.fsum(raw)
    return [value * scale for value in raw]


def _fit_models(rows: list[dict[str, Any]]) -> dict[str, Any]:
    import numpy as np
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    values = np.asarray([row["features"] for row in rows], dtype=float)
    weights = np.asarray(_balanced_weights(rows), dtype=float)
    classifier = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    classifier.fit(
        values,
        np.asarray([row["progress"] for row in rows], dtype=int),
        sample_weight=weights,
    )
    regressors = {}
    for name in ("reduction", "log_generated"):
        estimator = HistGradientBoostingRegressor(**MODEL_PARAMETERS)
        estimator.fit(
            values,
            np.asarray([row[name] for row in rows], dtype=float),
            sample_weight=weights,
        )
        regressors[name] = estimator
    return {
        "progress_probability": classifier,
        "conflict_reduction": regressors["reduction"],
        "log_generated": regressors["log_generated"],
    }


def _predict_models(models: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    import numpy as np

    values = np.asarray([row["features"] for row in rows], dtype=float)
    return {
        "progress_probability": list(
            map(float, models["progress_probability"].predict_proba(values)[:, 1])
        ),
        "conflict_reduction": list(
            map(float, models["conflict_reduction"].predict(values))
        ),
        "generated": [
            max(0.0, math.expm1(float(value)))
            for value in models["log_generated"].predict(values)
        ],
        "log_generated": list(
            map(float, models["log_generated"].predict(values))
        ),
    }


def _hist_trees(estimator: Any) -> list[list[dict[str, Any]]]:
    trees = []
    for stage in estimator._predictors:
        if len(stage) != 1:
            raise ValueError("portable repair-aware model requires one tree per stage")
        nodes = []
        for node in stage[0].nodes:
            if bool(node["is_categorical"]):
                raise ValueError("portable repair-aware model does not support categorical nodes")
            nodes.append(
                {
                    "value": float(node["value"]),
                    "feature_idx": int(node["feature_idx"]),
                    "num_threshold": float(node["num_threshold"]),
                    "missing_go_to_left": bool(node["missing_go_to_left"]),
                    "left": int(node["left"]),
                    "right": int(node["right"]),
                    "is_leaf": bool(node["is_leaf"]),
                }
            )
        trees.append(nodes)
    return trees


def _portable_payload(
    name: str, estimator: Any, feature_names: list[str], source_fingerprint: str
) -> dict[str, Any]:
    transform = "sigmoid" if name.endswith("_probability") else "identity"
    payload = {
        "schema": PORTABLE_SCALAR_MODEL_SCHEMA,
        "schema_version": 1,
        "name": name,
        "profile": "realized_dynamic",
        "feature_names": feature_names,
        "baseline": float(estimator._baseline_prediction[0, 0]),
        "trees": _hist_trees(estimator),
        "transform": transform,
        "source_fingerprint": source_fingerprint,
    }
    payload["semantic_fingerprint"] = _fingerprint(
        {
            "name": name,
            "profile": "realized_dynamic",
            "feature_names": feature_names,
            "baseline": payload["baseline"],
            "trees": payload["trees"],
            "transform": transform,
        }
    )
    return payload


def _candidate_actuals(trials: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    result = {}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in trials:
        grouped[(str(row["state_id"]), str(row["candidate_id"]))].append(row)
    for key, rows in grouped.items():
        result[key] = {
            "effective_rate": statistics.fmean(float(row["progress"]) for row in rows),
            "reduction": statistics.fmean(float(row["reduction"]) for row in rows),
            "generated": statistics.fmean(float(row["generated"]) for row in rows),
        }
    return result


def _candidate_predictions(
    candidates: list[dict[str, Any]],
    trial_predictions: dict[tuple[str, str], dict[str, list[float]]],
) -> dict[str, list[float]]:
    values = {name: [] for name in ("progress_probability", "conflict_reduction", "generated")}
    for candidate in candidates:
        key = (str(candidate["state_id"]), str(candidate["candidate_id"]))
        prediction = trial_predictions[key]
        for name in values:
            values[name].append(statistics.fmean(prediction[name]))
    return values


def _index_predictions(
    rows: list[dict[str, Any]], predictions: dict[str, list[float]]
) -> dict[tuple[str, str], dict[str, list[float]]]:
    result: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for index, row in enumerate(rows):
        key = (str(row["state_id"]), str(row["candidate_id"]))
        for name in ("progress_probability", "conflict_reduction", "generated"):
            result[key][name].append(float(predictions[name][index]))
    return {key: dict(value) for key, value in result.items()}


def _state_records(
    candidate_rows: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    predictions: dict[tuple[str, str], dict[str, list[float]]],
    main_model: Any,
) -> list[dict[str, Any]]:
    actuals = _candidate_actuals(trials)
    states = []
    for state_id, candidates in sorted(_grouped(candidate_rows, "state_id").items()):
        candidates.sort(key=lambda row: str(row["candidate_key"]))
        base, scores, _ = score_online_candidates(candidates, main_model)
        states.append(
            {
                "state_id": state_id,
                "map_id": str(candidates[0]["map_id"]),
                "candidates": candidates,
                "scores": scores,
                "base": base,
                "predictions": _candidate_predictions(candidates, predictions),
                "actuals": [
                    actuals[(state_id, str(candidate["candidate_id"]))]
                    for candidate in candidates
                ],
            }
        )
    return states


def _selection_metrics(states: list[dict[str, Any]], selector: Any) -> dict[str, float]:
    selected = [selector(state) for state in states]
    actuals = [state["actuals"][index] for state, index in zip(states, selected)]
    return {
        "state_count": float(len(states)),
        "effective_rate": statistics.fmean(row["effective_rate"] for row in actuals),
        "mean_conflict_reduction": statistics.fmean(row["reduction"] for row in actuals),
        "mean_generated": statistics.fmean(row["generated"] for row in actuals),
    }


def _calibrate_rescue(states: list[dict[str, Any]]) -> dict[str, Any]:
    failed = [state for state in states if state["actuals"][state["base"]]["effective_rate"] == 0.0]
    reports = []
    for tolerance in RESCUE_TOLERANCE_GRID:
        def selector(state: dict[str, Any]) -> int:
            alternatives = [index for index in range(len(state["candidates"])) if index != state["base"]]
            order = repair_aware_order(
                state["candidates"], state["predictions"], state["scores"],
                probability_tolerance=tolerance, eligible=alternatives,
            )
            return order[0] if order else state["base"]
        reports.append({"probability_tolerance": tolerance, **_selection_metrics(failed, selector)})
    maximum = max((row["effective_rate"] for row in reports), default=0.0)
    eligible = [row for row in reports if row["effective_rate"] >= maximum - 0.01]
    selected = max(
        eligible,
        key=lambda row: (row["mean_conflict_reduction"], -row["mean_generated"], -row["probability_tolerance"]),
    ) if eligible else {"probability_tolerance": 0.05}
    return {"failed_base_state_count": len(failed), "grid": reports, "selected": selected}


def _calibrate_tiebreak(states: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = _selection_metrics(states, lambda state: state["base"])
    reports = []
    for score_gap in TIE_SCORE_GAP_GRID:
        for progress_margin in TIE_PROGRESS_MARGIN_GRID:
            thresholds = {
                "rescue_probability_tolerance": 0.05,
                "tie_score_gap_fraction": score_gap,
                "tie_progress_margin": progress_margin,
                "tie_minimum_reduction_ratio": TIE_MINIMUM_REDUCTION_RATIO,
                "tie_maximum_generated_ratio": TIE_MAXIMUM_GENERATED_RATIO,
            }
            metrics = _selection_metrics(
                states,
                lambda state, frozen=thresholds: guarded_tiebreak_candidate(
                    state["candidates"], state["predictions"], state["scores"], state["base"], frozen
                ),
            )
            metrics.update({"tie_score_gap_fraction": score_gap, "tie_progress_margin": progress_margin})
            metrics["generated_reduction_fraction"] = 1.0 - metrics["mean_generated"] / baseline["mean_generated"]
            metrics["conflict_reduction_ratio"] = metrics["mean_conflict_reduction"] / baseline["mean_conflict_reduction"]
            metrics["passed"] = (
                metrics["effective_rate"] + 1e-12 >= baseline["effective_rate"]
                and metrics["conflict_reduction_ratio"] + 1e-12 >= TIE_MINIMUM_REDUCTION_RATIO
                and metrics["generated_reduction_fraction"] + 1e-12 >= 0.10
            )
            reports.append(metrics)
    passing = [row for row in reports if row["passed"]]
    selected = max(
        passing,
        key=lambda row: (row["generated_reduction_fraction"], row["mean_conflict_reduction"], -row["tie_score_gap_fraction"]),
    ) if passing else None
    return {"baseline": baseline, "grid": reports, "selected": selected, "passed": selected is not None}


def _prediction_metrics(rows: list[dict[str, Any]], predictions: dict[str, list[float]]) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score

    labels = [int(row["progress"]) for row in rows]
    probabilities = predictions["progress_probability"]
    return {
        "trial_count": float(len(rows)),
        "progress_auc": float(roc_auc_score(labels, probabilities)),
        "progress_brier": statistics.fmean((left - right) ** 2 for left, right in zip(labels, probabilities)),
        "reduction_mae": statistics.fmean(abs(float(row["reduction"]) - value) for row, value in zip(rows, predictions["conflict_reduction"])),
        "generated_log_mae": statistics.fmean(abs(float(row["log_generated"]) - value) for row, value in zip(rows, predictions["log_generated"])),
    }


def run_repair_aware_training(
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
    main_bundle = load_controller_bundle(controller_path)
    main_model = main_bundle.main_models["realized_dynamic"]
    feature_names = list(main_model.base_feature_names)
    train_candidates, train_trials = _training_rows(
        feature_rows, trial_rows, "policy_train", feature_names
    )
    validation_candidates, validation_trials = _training_rows(
        feature_rows, trial_rows, "policy_validation", feature_names
    )
    train_maps = {str(row["map_id"]) for row in train_trials}
    validation_maps = {str(row["map_id"]) for row in validation_trials}
    if train_maps & validation_maps:
        raise ValueError("policy_train and policy_validation maps overlap")

    folds = _balanced_map_folds(train_trials)
    oof_predictions: dict[tuple[str, str], dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    fold_reports = []
    for fold in folds:
        training = [row for row in train_trials if row["map_id"] in set(fold["train_maps"])]
        held_out = [row for row in train_trials if row["map_id"] in set(fold["validation_maps"])]
        models = _fit_models(training)
        predicted = _predict_models(models, held_out)
        indexed = _index_predictions(held_out, predicted)
        overlap = set(oof_predictions) & set(indexed)
        if overlap:
            raise ValueError("OOF repair-aware predictions overlap")
        for key, value in indexed.items():
            oof_predictions[key] = collections.defaultdict(list, value)
        fold_reports.append(
            {
                **fold,
                "training_trial_count": len(training),
                "validation_trial_count": len(held_out),
                "prediction_metrics": _prediction_metrics(held_out, predicted),
            }
        )
    oof_index = {key: dict(value) for key, value in oof_predictions.items()}
    train_states = _state_records(
        train_candidates, train_trials, oof_index, main_model
    )
    rescue_calibration = _calibrate_rescue(train_states)
    tiebreak_calibration = _calibrate_tiebreak(train_states)

    final_models = _fit_models(train_trials)
    validation_predicted = _predict_models(final_models, validation_trials)
    validation_index = _index_predictions(validation_trials, validation_predicted)
    validation_states = _state_records(
        validation_candidates, validation_trials, validation_index, main_model
    )
    tolerance = float(rescue_calibration["selected"]["probability_tolerance"])
    selected_tie = tiebreak_calibration.get("selected")
    thresholds = {
        "rescue_probability_tolerance": tolerance,
        "tie_score_gap_fraction": float(selected_tie["tie_score_gap_fraction"]) if selected_tie else 0.0,
        "tie_progress_margin": float(selected_tie["tie_progress_margin"]) if selected_tie else 0.0,
        "tie_minimum_reduction_ratio": TIE_MINIMUM_REDUCTION_RATIO,
        "tie_maximum_generated_ratio": TIE_MAXIMUM_GENERATED_RATIO,
    }
    validation_baseline = _selection_metrics(validation_states, lambda state: state["base"])
    validation_tie = _selection_metrics(
        validation_states,
        lambda state: guarded_tiebreak_candidate(
            state["candidates"], state["predictions"], state["scores"], state["base"], thresholds
        ),
    )
    validation_tie["generated_reduction_fraction"] = 1.0 - validation_tie["mean_generated"] / validation_baseline["mean_generated"]
    validation_tie["conflict_reduction_ratio"] = validation_tie["mean_conflict_reduction"] / validation_baseline["mean_conflict_reduction"]
    validation_tie["passed"] = bool(
        selected_tie is not None
        and validation_tie["effective_rate"] + 1e-12 >= validation_baseline["effective_rate"]
        and validation_tie["conflict_reduction_ratio"] + 1e-12 >= TIE_MINIMUM_REDUCTION_RATIO
        and validation_tie["generated_reduction_fraction"] + 1e-12 >= 0.10
    )

    source_fingerprint = _fingerprint(
        {
            "feature_index_sha256": sha256_file(feature_path),
            "trial_manifest_sha256": sha256_file(trial_path),
            "controller_semantic_fingerprint": main_bundle.manifest[
                "main_ranker_semantic_fingerprint"
            ],
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "parameters": MODEL_PARAMETERS,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    model_rows = {}
    portable_deltas = {}
    for name, estimator in final_models.items():
        payload = _portable_payload(name, estimator, feature_names, source_fingerprint)
        path = output_root / f"repair_aware__{name}.json"
        _write_json(path, payload)
        portable = load_portable_scalar_model(payload)
        portable_values = portable.predict(
            [
                {
                    "feature_profile": "realized_dynamic",
                    "feature_names": tuple(feature_names),
                    "feature_values": tuple(row["features"]),
                }
                for row in validation_trials
            ]
        )
        reference_name = "generated" if name == "log_generated" else name
        reference = validation_predicted[reference_name]
        if name == "log_generated":
            reference = validation_predicted["log_generated"]
        maximum_delta = max(
            (abs(float(left) - float(right)) for left, right in zip(reference, portable_values)),
            default=0.0,
        )
        if maximum_delta > 1e-12:
            raise ValueError(f"portable repair-aware model differs from sklearn: {name} {maximum_delta}")
        portable_deltas[name] = maximum_delta
        model_rows[name] = {
            "file": path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(path),
            "semantic_fingerprint": payload["semantic_fingerprint"],
            "tree_count": len(payload["trees"]),
            "inference_backend": portable.inference_backend,
        }

    report = {
        "schema": REPAIR_AWARE_TRAINING_SCHEMA,
        "schema_version": 1,
        "inputs": {
            "feature_index": str(feature_path),
            "feature_index_sha256": sha256_file(feature_path),
            "trial_manifest": str(trial_path),
            "trial_manifest_sha256": sha256_file(trial_path),
            "controller_bundle": str(controller_path),
        },
        "training_labels_seen": ["policy_train"],
        "formal_or_ood_labels_seen": False,
        "validation_locked_once": True,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_count": len(feature_names),
        "training_state_count": len(train_states),
        "training_trial_count": len(train_trials),
        "training_map_count": len(train_maps),
        "validation_state_count": len(validation_states),
        "validation_trial_count": len(validation_trials),
        "validation_map_count": len(validation_maps),
        "model_parameters": MODEL_PARAMETERS,
        "folds": fold_reports,
        "rescue_calibration": rescue_calibration,
        "guarded_tiebreak_calibration": tiebreak_calibration,
        "validation_prediction_metrics": _prediction_metrics(validation_trials, validation_predicted),
        "validation_baseline": validation_baseline,
        "validation_guarded_tiebreak": validation_tie,
        "guarded_tiebreak_eligible": bool(validation_tie["passed"]),
        "thresholds": thresholds,
        "portable_maximum_deltas": portable_deltas,
    }
    report_path = output_root / "training_report.json"
    _write_json(report_path, report)
    manifest = {
        "schema": LEGACY_REPAIR_AWARE_BUNDLE_SCHEMA,
        "schema_version": 1,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "profile": "realized_dynamic",
        "feature_names": feature_names,
        "main_ranker_semantic_fingerprint": main_bundle.manifest[
            "main_ranker_semantic_fingerprint"
        ],
        "models": model_rows,
        "thresholds": thresholds,
        "guarded_tiebreak_eligible": bool(validation_tie["passed"]),
        "training_report": {
            "file": report_path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(report_path),
        },
        "source_fingerprint": source_fingerprint,
    }
    _write_json(output_root / "repair_aware_manifest.json", manifest)
    return {**report, "manifest": manifest}


__all__ = [
    "MODEL_PARAMETERS",
    "REPAIR_AWARE_TRAINING_SCHEMA",
    "run_repair_aware_training",
]
