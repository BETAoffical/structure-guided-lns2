from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.mixed_full_v2 import _atomic_pickle
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_augcontrol import (
    _input_specifications,
    _load_candidates,
    _load_pair_table,
    _mean_metrics,
    _prediction_records,
    _selection_change_diagnostics,
)
from experiments.stride_certguard import (
    CONTROLLER_ID,
    LABEL_REPORT_SCHEMA,
    LABEL_SCHEMA,
    validate_certguard_design,
    validate_certguard_label_config,
    validate_certguard_training_config,
)
from experiments.stride_guardrank import (
    _balanced_folds,
    _challenger_evidence,
    _fit_maps,
    _topology_comparisons,
)
from experiments.stride_maprank import validate_maprank_design
from experiments.stride_repairability import BUILD_SCHEMA, CONFLICT_ONLY_LABEL_SCHEMA
from experiments.stride_stage3 import _project_path
from experiments.stride_stage4 import _candidate_matrix, _fit_registered_model, _frozen_predictions, _pair_matrix


REPORT_SCHEMA = "lns2.stride.certguard_training.v1"
PREDICTION_SCHEMA = "lns2.stride.certguard_prediction.v1"


def _load_uncertainty_table(
    path: Path,
    candidate_index: dict[tuple[str, str], int],
    *,
    split: str,
) -> dict[str, Any]:
    import numpy as np

    left: list[int] = []
    right: list[int] = []
    labels: list[int] = []
    weights: list[float] = []
    state_ids: list[str] = []
    state_maps: dict[str, str] = {}
    state_weights: Counter[str] = Counter()
    lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("schema") != LABEL_SCHEMA:
            raise ValueError("unexpected CertGuard uncertainty label schema")
        if str(row["split"]) != split:
            continue
        state_id = str(row["state_id"])
        map_id = str(row["map_id"])
        left_id = str(row["left_candidate_id"])
        right_id = str(row["right_candidate_id"])
        if left_id >= right_id:
            raise ValueError("CertGuard unordered pair orientation changed")
        if state_id in state_maps and state_maps[state_id] != map_id:
            raise ValueError("CertGuard pair state changed map")
        state_maps[state_id] = map_id
        key = (state_id, left_id, right_id)
        if key in lookup:
            raise ValueError(f"duplicate CertGuard uncertainty pair: {key}")
        label = int(row["label"])
        winner = row.get("robust_winner_candidate_id")
        if label not in {0, 1} or (label == 0) != (winner is None):
            raise ValueError("CertGuard uncertainty target is inconsistent")
        weight = float(row["sample_weight"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("CertGuard uncertainty weight is invalid")
        left.append(candidate_index[(state_id, left_id)])
        right.append(candidate_index[(state_id, right_id)])
        labels.append(label)
        weights.append(weight)
        state_ids.append(state_id)
        state_weights[state_id] += weight
        lookup[key] = {
            "label": label,
            "winner_candidate_id": None if winner is None else str(winner),
        }
    if not labels or set(labels) != {0, 1}:
        raise ValueError(f"CertGuard {split} uncertainty pairs require both labels")
    if any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9)
        for value in state_weights.values()
    ):
        raise ValueError("CertGuard uncertainty pair weights differ by state")
    return {
        "left": np.asarray(left, dtype=np.int32),
        "right": np.asarray(right, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int8),
        "weights": np.asarray(weights, dtype=np.float64),
        "state_ids": np.asarray(state_ids, dtype=object),
        "maps": np.asarray([state_maps[value] for value in state_ids], dtype=object),
        "state_count": len(state_weights),
        "lookup": lookup,
    }


def _uncertainty_matrix(candidate_values: Any, table: dict[str, Any]) -> Any:
    import numpy as np

    left = candidate_values[table["left"]]
    right = candidate_values[table["right"]]
    return np.concatenate((np.abs(left - right), 0.5 * (left + right)), axis=1).astype(
        np.float32, copy=False
    )


def _fit_uncertainty_maps(
    *,
    maps: set[str],
    pair_values: Any,
    table: dict[str, Any],
    parameters: dict[str, Any],
) -> Any:
    import numpy as np

    mask = np.isin(table["maps"], sorted(maps))
    if not bool(np.any(mask)):
        raise ValueError("CertGuard map split has no uncertainty pairs")
    return _fit_registered_model(
        pair_values[mask], table["labels"][mask], table["weights"][mask], parameters
    )


def _pair_vector(left: dict[str, Any], right: dict[str, Any]) -> list[float]:
    names = PROFILE_FEATURE_NAMES["realized_dynamic"]
    left_values = [float(left["features"][name]) for name in names]
    right_values = [float(right["features"][name]) for name in names]
    return [abs(first - second) for first, second in zip(left_values, right_values)] + [
        0.5 * (first + second) for first, second in zip(left_values, right_values)
    ]


def _certainty_probabilities(
    grouped: dict[str, list[dict[str, Any]]],
    direction_evidence: dict[str, dict[str, Any]],
    estimator: Any,
) -> dict[str, float]:
    result: dict[str, float] = {}
    vectors: list[list[float]] = []
    vector_states: list[str] = []
    for state_id, rows in sorted(grouped.items()):
        evidence = direction_evidence[state_id]
        anchor_id = str(evidence["anchor_candidate_id"])
        challenger_id = str(evidence["challenger_candidate_id"])
        if anchor_id == challenger_id:
            result[state_id] = 1.0
            continue
        by_id = {str(row["candidate_id"]): row for row in rows}
        vectors.append(_pair_vector(by_id[challenger_id], by_id[anchor_id]))
        vector_states.append(state_id)
    if vectors:
        probabilities = estimator.predict_proba(vectors)[:, 1]
        result.update(
            {state_id: float(value) for state_id, value in zip(vector_states, probabilities)}
        )
    return result


def _maprank_predictions(
    evidence: dict[str, dict[str, Any]], thresholds: dict[str, float]
) -> dict[str, str]:
    result: dict[str, str] = {}
    for state_id, row in evidence.items():
        anchor = str(row["anchor_candidate_id"])
        challenger = str(row["challenger_candidate_id"])
        passed = float(row["challenger_evidence"]) + 1e-12 >= float(
            thresholds[str(row["challenger_kind"])]
        )
        result[state_id] = challenger if anchor == challenger or passed else anchor
    return result


def _certguard_predictions(
    *,
    maprank_predictions: dict[str, str],
    anchor_predictions: dict[str, str],
    certainty_probabilities: dict[str, float],
    certainty_threshold: float,
) -> dict[str, str]:
    return {
        state_id: (
            maprank_predictions[state_id]
            if maprank_predictions[state_id] == anchor_predictions[state_id]
            or float(certainty_probabilities[state_id]) + 1e-12 >= certainty_threshold
            else anchor_predictions[state_id]
        )
        for state_id in maprank_predictions
    }


def _lookup_pair(
    lookup: dict[tuple[str, str, str], dict[str, Any]],
    state_id: str,
    first: str,
    second: str,
) -> dict[str, Any]:
    left, right = sorted((str(first), str(second)))
    return lookup[(state_id, left, right)]


def _override_diagnostics(
    *,
    maprank_predictions: dict[str, str],
    certguard_predictions: dict[str, str],
    anchor_predictions: dict[str, str],
    lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    maprank_overrides = [
        state_id
        for state_id in sorted(anchor_predictions)
        if maprank_predictions[state_id] != anchor_predictions[state_id]
    ]
    retained = [
        state_id
        for state_id in maprank_overrides
        if certguard_predictions[state_id] == maprank_predictions[state_id]
    ]
    robust = uncertain = safe = wrong = 0
    for state_id in retained:
        row = _lookup_pair(
            lookup,
            state_id,
            anchor_predictions[state_id],
            maprank_predictions[state_id],
        )
        if int(row["label"]) == 0:
            uncertain += 1
        else:
            robust += 1
            if str(row["winner_candidate_id"]) == maprank_predictions[state_id]:
                safe += 1
            else:
                wrong += 1
    retained_count = len(retained)
    return {
        "state_count": len(anchor_predictions),
        "maprank_override_count": len(maprank_overrides),
        "retained_override_count": retained_count,
        "rejected_override_count": len(maprank_overrides) - retained_count,
        "maprank_override_retention_fraction": (
            retained_count / len(maprank_overrides) if maprank_overrides else 1.0
        ),
        "robust_retained_override_count": robust,
        "uncertain_retained_override_count": uncertain,
        "uncertain_override_fraction": (
            uncertain / retained_count if retained_count else 0.0
        ),
        "directionally_safe_override_count": safe,
        "directionally_wrong_override_count": wrong,
        "directionally_safe_override_precision": (
            safe / retained_count if retained_count else 0.0
        ),
    }


def _calibrate_certainty(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    maprank_predictions: dict[str, str],
    anchor_predictions: dict[str, str],
    certainty_probabilities: dict[str, float],
    lookup: dict[tuple[str, str, str], dict[str, Any]],
    grid: list[float],
    exact_tolerance: float,
    top3_tolerance: float,
) -> dict[str, Any]:
    anchor_records = _prediction_records(
        "v2-full/anchor", anchor_predictions, grouped, evaluation_pool="augmented"
    )
    anchor_metrics = _mean_metrics(anchor_records)
    rows: list[dict[str, Any]] = []
    for threshold in grid:
        predictions = _certguard_predictions(
            maprank_predictions=maprank_predictions,
            anchor_predictions=anchor_predictions,
            certainty_probabilities=certainty_probabilities,
            certainty_threshold=threshold,
        )
        records = _prediction_records(
            CONTROLLER_ID, predictions, grouped, evaluation_pool="augmented"
        )
        metrics = _mean_metrics(records)
        rows.append(
            {
                "certainty_threshold": threshold,
                "metrics": metrics,
                "constraints_passed": (
                    float(metrics["exact_best_rate"]) + exact_tolerance + 1e-12
                    >= float(anchor_metrics["exact_best_rate"])
                    and float(metrics["top3_hit_rate"]) + top3_tolerance + 1e-12
                    >= float(anchor_metrics["top3_hit_rate"])
                ),
                "override_diagnostics": _override_diagnostics(
                    maprank_predictions=maprank_predictions,
                    certguard_predictions=predictions,
                    anchor_predictions=anchor_predictions,
                    lookup=lookup,
                ),
            }
        )
    eligible = [row for row in rows if row["constraints_passed"]]
    if not eligible:
        raise ValueError("CertGuard threshold grid lacks its V2 fallback")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["metrics"]["mean_normalized_repairability_regret"]),
            float(row["override_diagnostics"]["uncertain_override_fraction"]),
            -float(row["certainty_threshold"]),
        ),
    )
    return {
        **selected,
        "grid_size": len(rows),
        "eligible_grid_count": len(eligible),
        "anchor_metrics": anchor_metrics,
        "threshold_sweep": [
            {
                "certainty_threshold": float(row["certainty_threshold"]),
                "constraints_passed": bool(row["constraints_passed"]),
                "mean_normalized_repairability_regret": float(
                    row["metrics"]["mean_normalized_repairability_regret"]
                ),
                "exact_best_rate": float(row["metrics"]["exact_best_rate"]),
                "top3_hit_rate": float(row["metrics"]["top3_hit_rate"]),
                "override_diagnostics": dict(row["override_diagnostics"]),
            }
            for row in rows
        ],
    }


def _uncertainty_metrics(
    labels: Any, probabilities: Any, weights: Any
) -> dict[str, float]:
    import numpy as np
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    weights = np.asarray(weights)
    total = float(np.sum(weights))
    positive_weight = float(np.sum(weights[labels == 1]))
    prevalence = positive_weight / total
    correct = (probabilities >= 0.5) == (labels == 1)
    accuracy = float(np.sum(weights[correct])) / total
    majority = max(prevalence, 1.0 - prevalence)
    brier = float(np.sum(weights * (probabilities - labels) ** 2)) / total
    baseline_brier = float(
        np.sum(weights * (prevalence - labels) ** 2) / total
    )
    return {
        "weighted_accuracy": accuracy,
        "weighted_majority_accuracy": majority,
        "accuracy_gain_over_majority": accuracy - majority,
        "weighted_roc_auc": float(
            roc_auc_score(labels, probabilities, sample_weight=weights)
        ),
        "weighted_brier_score": brier,
        "weighted_constant_prevalence_brier_score": baseline_brier,
        "weighted_brier_improvement": baseline_brier - brier,
        "weighted_positive_prevalence": prevalence,
        "weighted_pair_count": total,
    }


def _nested_train_oof(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    direction_values: Any,
    direction_table: dict[str, Any],
    direction_maps: Any,
    uncertainty_values: Any,
    uncertainty_table: dict[str, Any],
    anchor_predictions: dict[str, str],
    direction_specs: tuple[tuple[str, str], ...],
    direction_parameters: dict[str, Any],
    uncertainty_parameters: dict[str, Any],
    direction_thresholds: dict[str, float],
    outer_count: int,
    inner_count: int,
    certainty_grid: list[float],
    exact_tolerance: float,
    top3_tolerance: float,
) -> dict[str, Any]:
    import numpy as np

    all_maps = {str(rows[0]["map_id"]) for rows in grouped.values()}
    certguard_predictions: dict[str, str] = {}
    maprank_predictions: dict[str, str] = {}
    crossfit_certainties: dict[str, float] = {}
    folds: list[dict[str, Any]] = []
    oof_labels: list[int] = []
    oof_probabilities: list[float] = []
    oof_weights: list[float] = []
    for outer in _balanced_folds(all_maps, outer_count):
        outer_train_maps = set(map(str, outer["train_maps"]))
        outer_validation_maps = set(map(str, outer["validation_maps"]))
        inner_grouped = {
            state_id: rows
            for state_id, rows in grouped.items()
            if str(rows[0]["map_id"]) in outer_train_maps
        }
        inner_anchor = {state_id: anchor_predictions[state_id] for state_id in inner_grouped}
        inner_direction_evidence: dict[str, dict[str, Any]] = {}
        inner_certainties: dict[str, float] = {}
        for inner in _balanced_folds(outer_train_maps, inner_count):
            fit_maps = set(map(str, inner["train_maps"]))
            held_maps = set(map(str, inner["validation_maps"]))
            direction_estimator = _fit_maps(
                maps=fit_maps,
                pair_maps=direction_maps,
                pair_values=direction_values,
                pair_table=direction_table,
                parameters=direction_parameters,
            )
            uncertainty_estimator = _fit_uncertainty_maps(
                maps=fit_maps,
                pair_values=uncertainty_values,
                table=uncertainty_table,
                parameters=uncertainty_parameters,
            )
            held_grouped = {
                state_id: rows
                for state_id, rows in inner_grouped.items()
                if str(rows[0]["map_id"]) in held_maps
            }
            held_anchor = {state_id: inner_anchor[state_id] for state_id in held_grouped}
            evidence = _challenger_evidence(
                held_grouped, direction_estimator, held_anchor, direction_specs
            )
            inner_direction_evidence.update(evidence)
            inner_certainties.update(
                _certainty_probabilities(held_grouped, evidence, uncertainty_estimator)
            )
        if set(inner_direction_evidence) != set(inner_grouped) or set(inner_certainties) != set(inner_grouped):
            raise ValueError("CertGuard inner OOF coverage differs")
        inner_maprank = _maprank_predictions(inner_direction_evidence, direction_thresholds)
        calibration = _calibrate_certainty(
            grouped=inner_grouped,
            maprank_predictions=inner_maprank,
            anchor_predictions=inner_anchor,
            certainty_probabilities=inner_certainties,
            lookup=uncertainty_table["lookup"],
            grid=certainty_grid,
            exact_tolerance=exact_tolerance,
            top3_tolerance=top3_tolerance,
        )
        direction_estimator = _fit_maps(
            maps=outer_train_maps,
            pair_maps=direction_maps,
            pair_values=direction_values,
            pair_table=direction_table,
            parameters=direction_parameters,
        )
        uncertainty_estimator = _fit_uncertainty_maps(
            maps=outer_train_maps,
            pair_values=uncertainty_values,
            table=uncertainty_table,
            parameters=uncertainty_parameters,
        )
        held_grouped = {
            state_id: rows
            for state_id, rows in grouped.items()
            if str(rows[0]["map_id"]) in outer_validation_maps
        }
        held_anchor = {state_id: anchor_predictions[state_id] for state_id in held_grouped}
        evidence = _challenger_evidence(
            held_grouped, direction_estimator, held_anchor, direction_specs
        )
        held_maprank = _maprank_predictions(evidence, direction_thresholds)
        held_certainties = _certainty_probabilities(
            held_grouped, evidence, uncertainty_estimator
        )
        held_certguard = _certguard_predictions(
            maprank_predictions=held_maprank,
            anchor_predictions=held_anchor,
            certainty_probabilities=held_certainties,
            certainty_threshold=float(calibration["certainty_threshold"]),
        )
        maprank_predictions.update(held_maprank)
        certguard_predictions.update(held_certguard)
        crossfit_certainties.update(held_certainties)
        pair_mask = np.isin(uncertainty_table["maps"], sorted(outer_validation_maps))
        probabilities = uncertainty_estimator.predict_proba(
            uncertainty_values[pair_mask]
        )[:, 1]
        oof_labels.extend(map(int, uncertainty_table["labels"][pair_mask]))
        oof_probabilities.extend(map(float, probabilities))
        oof_weights.extend(map(float, uncertainty_table["weights"][pair_mask]))
        folds.append(
            {
                "outer_fold": int(outer["fold"]),
                "training_maps": sorted(outer_train_maps),
                "validation_maps": sorted(outer_validation_maps),
                "validation_state_count": len(held_grouped),
                "certainty_threshold_calibration": calibration,
                "uncertainty_pair_metrics": _uncertainty_metrics(
                    uncertainty_table["labels"][pair_mask],
                    probabilities,
                    uncertainty_table["weights"][pair_mask],
                ),
            }
        )
    if set(certguard_predictions) != set(grouped) or set(maprank_predictions) != set(grouped):
        raise ValueError("CertGuard outer OOF coverage differs")
    final_calibration = _calibrate_certainty(
        grouped=grouped,
        maprank_predictions=maprank_predictions,
        anchor_predictions=anchor_predictions,
        certainty_probabilities=crossfit_certainties,
        lookup=uncertainty_table["lookup"],
        grid=certainty_grid,
        exact_tolerance=exact_tolerance,
        top3_tolerance=top3_tolerance,
    )
    certguard_records = _prediction_records(
        CONTROLLER_ID, certguard_predictions, grouped, evaluation_pool="augmented"
    )
    maprank_records = _prediction_records(
        "stride-maprank-v1/oof-fixed-runtime-threshold",
        maprank_predictions,
        grouped,
        evaluation_pool="augmented",
    )
    anchor_records = _prediction_records(
        "v2-full/anchor", anchor_predictions, grouped, evaluation_pool="augmented"
    )
    return {
        "predictions": certguard_predictions,
        "maprank_predictions": maprank_predictions,
        "records": certguard_records,
        "maprank_records": maprank_records,
        "anchor_records": anchor_records,
        "metrics": _mean_metrics(certguard_records),
        "maprank_metrics": _mean_metrics(maprank_records),
        "anchor_metrics": _mean_metrics(anchor_records),
        "override_diagnostics": _override_diagnostics(
            maprank_predictions=maprank_predictions,
            certguard_predictions=certguard_predictions,
            anchor_predictions=anchor_predictions,
            lookup=uncertainty_table["lookup"],
        ),
        "uncertainty_pair_metrics": _uncertainty_metrics(
            oof_labels, oof_probabilities, oof_weights
        ),
        "folds": folds,
        "final_calibration": final_calibration,
    }


def run_certguard_training(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    import numpy as np

    project_root = Path(__file__).resolve().parents[1]
    path = Path(config_path).resolve()
    output_path = Path(output).resolve()
    config = _read_json(path)
    validate_certguard_training_config(config)
    design_path = _project_path(project_root, config["design"])
    label_config_path = _project_path(project_root, config["label_config"])
    if sha256_file(design_path) != str(config["design_sha256"]):
        raise ValueError("CertGuard design SHA differs")
    if sha256_file(label_config_path) != str(config["label_config_sha256"]):
        raise ValueError("CertGuard label config SHA differs")
    validate_certguard_design(_read_json(design_path))
    validate_certguard_label_config(_read_json(label_config_path))
    labels_path = _project_path(project_root, config["uncertainty_labels"])
    label_report_path = labels_path.parent / "certguard_label_report.json"
    label_report = _read_json(label_report_path)
    if (
        label_report.get("schema") != LABEL_REPORT_SCHEMA
        or str(label_report.get("config_sha256")) != str(config["label_config_sha256"])
        or bool(label_report.get("runtime_used_in_label"))
        or bool(label_report.get("future_trajectory_used_in_label"))
        or bool(label_report.get("high_load_data_read"))
    ):
        raise ValueError("CertGuard label report is invalid")
    maprank_labels = _project_path(project_root, config["maprank_labels"])
    maprank_summary_path = maprank_labels / "label_build_summary.json"
    aggregate_path = maprank_labels / "candidate_aggregates.jsonl"
    direction_path = maprank_labels / "conflict_only_dominance_pairs.jsonl"
    selection_path = _project_path(project_root, config["maprank_selection"])
    registered = {
        maprank_summary_path: config["maprank_label_summary_sha256"],
        aggregate_path: config["maprank_candidate_aggregates_sha256"],
        direction_path: config["maprank_direction_pairs_sha256"],
        selection_path: config["maprank_selection_sha256"],
    }
    for registered_path, digest in registered.items():
        if sha256_file(registered_path) != str(digest):
            raise ValueError(f"CertGuard registered input differs: {registered_path}")
    maprank_summary = _read_json(maprank_summary_path)
    if maprank_summary.get("schema") != BUILD_SCHEMA:
        raise ValueError("CertGuard MapRank label summary schema differs")
    maprank_bundle = _project_path(project_root, config["maprank_bundle"])
    maprank_report_path = _project_path(project_root, config["maprank_training_report"])
    if sha256_file(maprank_bundle / "controller_manifest.json") != str(config["maprank_manifest_sha256"]):
        raise ValueError("CertGuard frozen MapRank manifest differs")
    if sha256_file(maprank_report_path) != str(config["maprank_training_report_sha256"]):
        raise ValueError("CertGuard frozen MapRank training report differs")
    maprank_report = _read_json(maprank_report_path)
    direction_thresholds = {
        name: float(value)
        for name, value in dict(config["frozen_direction"]["runtime_thresholds"]).items()
    }
    if {
        name: float(value) for name, value in dict(maprank_report["final_thresholds"]).items()
    } != direction_thresholds:
        raise ValueError("CertGuard registered direction thresholds differ from MapRank")
    data_design = _read_json(_project_path(project_root, "configs/stride_maprank_design.json"))
    validate_maprank_design(data_design)
    candidates, grouped = _load_candidates(
        aggregate_path, selection_path, topology_threshold=0.06
    )
    candidate_index = {
        (str(row["state_id"]), str(row["candidate_id"])): int(row["candidate_index"])
        for row in candidates
    }
    direction_table = _load_pair_table(
        direction_path,
        candidate_index,
        split="train",
        expected_schema=CONFLICT_ONLY_LABEL_SCHEMA,
    )
    uncertainty_train = _load_uncertainty_table(
        labels_path, candidate_index, split="train"
    )
    uncertainty_validation = _load_uncertainty_table(
        labels_path, candidate_index, split="validation"
    )
    train_grouped = {
        state_id: rows for state_id, rows in grouped.items() if str(rows[0]["split"]) == "train"
    }
    validation_grouped = {
        state_id: rows for state_id, rows in grouped.items() if str(rows[0]["split"]) == "validation"
    }
    train_anchor = _frozen_predictions(
        train_grouped, _project_path(project_root, "artifacts/initlns-closed-loop-controller-v2")
    )
    validation_anchor = _frozen_predictions(
        validation_grouped, _project_path(project_root, "artifacts/initlns-closed-loop-controller-v2")
    )
    _, direction_specs = _input_specifications()
    candidate_values = _candidate_matrix(candidates)
    direction_values = _pair_matrix(candidate_values, direction_table, direction_specs)
    direction_maps = np.asarray(
        [str(candidates[int(index)]["map_id"]) for index in direction_table["left"]],
        dtype=object,
    )
    uncertainty_train_values = _uncertainty_matrix(candidate_values, uncertainty_train)
    gates = dict(config["offline_gates"])
    nested = _nested_train_oof(
        grouped=train_grouped,
        direction_values=direction_values,
        direction_table=direction_table,
        direction_maps=direction_maps,
        uncertainty_values=uncertainty_train_values,
        uncertainty_table=uncertainty_train,
        anchor_predictions=train_anchor,
        direction_specs=direction_specs,
        direction_parameters=dict(maprank_report["model_parameters"]),
        uncertainty_parameters=dict(config["model_parameters"]),
        direction_thresholds=direction_thresholds,
        outer_count=int(config["outer_map_folds"]),
        inner_count=int(config["inner_map_folds"]),
        certainty_grid=list(map(float, config["certainty_threshold_grid"])),
        exact_tolerance=float(gates["exact_best_rate_noninferiority_tolerance_vs_v2"]),
        top3_tolerance=float(gates["top3_hit_rate_noninferiority_tolerance_vs_v2"]),
    )
    final_estimator = _fit_registered_model(
        uncertainty_train_values,
        uncertainty_train["labels"],
        uncertainty_train["weights"],
        dict(config["model_parameters"]),
    )
    from lns2_selector.controllers import load_selector
    from lns2_selector.runtime.contracts import SelectionRequest

    maprank_selector = load_selector("stride-maprank-v1", maprank_bundle)
    validation_maprank: dict[str, str] = {}
    validation_certainties: dict[str, float] = {}
    for state_id, rows in sorted(validation_grouped.items()):
        online_rows = [
            {
                "candidate_id": row["candidate_id"],
                "candidate_key": row["candidate_key"],
                "candidate_kind": row["candidate_kind"],
                "features": {"realized_dynamic": row["features"]},
            }
            for row in rows
        ]
        candidates_view = [
            {
                "candidate_id": row["candidate_id"],
                "selection_families": row["selection_families"],
            }
            for row in rows
        ]
        decision = maprank_selector.select(
            SelectionRequest(
                candidates=candidates_view,
                candidate_rows=online_rows,
                before_fingerprint=state_id,
                profile="realized_dynamic",
            )
        )
        selected = str(rows[int(decision.candidate_index)]["candidate_id"])
        validation_maprank[state_id] = selected
        anchor = validation_anchor[state_id]
        by_id = {str(row["candidate_id"]): row for row in rows}
        validation_certainties[state_id] = (
            1.0
            if selected == anchor
            else float(
                final_estimator.predict_proba([_pair_vector(by_id[selected], by_id[anchor])])[0, 1]
            )
        )
    final_threshold = float(nested["final_calibration"]["certainty_threshold"])
    validation_certguard = _certguard_predictions(
        maprank_predictions=validation_maprank,
        anchor_predictions=validation_anchor,
        certainty_probabilities=validation_certainties,
        certainty_threshold=final_threshold,
    )
    validation_records = _prediction_records(
        CONTROLLER_ID,
        validation_certguard,
        validation_grouped,
        evaluation_pool="augmented",
    )
    validation_maprank_records = _prediction_records(
        "stride-maprank-v1/frozen",
        validation_maprank,
        validation_grouped,
        evaluation_pool="augmented",
    )
    validation_anchor_records = _prediction_records(
        "v2-full/anchor", validation_anchor, validation_grouped, evaluation_pool="augmented"
    )
    metrics = dict(nested["metrics"])
    maprank_metrics = dict(nested["maprank_metrics"])
    anchor_metrics = dict(nested["anchor_metrics"])
    regret_vs_v2 = float(anchor_metrics["mean_normalized_repairability_regret"]) - float(
        metrics["mean_normalized_repairability_regret"]
    )
    relative_regret_vs_v2 = regret_vs_v2 / max(
        1e-12, float(anchor_metrics["mean_normalized_repairability_regret"])
    )
    regret_degradation_vs_maprank = float(metrics["mean_normalized_repairability_regret"]) - float(
        maprank_metrics["mean_normalized_repairability_regret"]
    )
    topology = _topology_comparisons(
        nested["records"],
        nested["anchor_records"],
        float(gates["maximum_topology_group_normalized_regret_degradation_vs_v2"]),
    )
    pair_metrics = dict(nested["uncertainty_pair_metrics"])
    override = dict(nested["override_diagnostics"])
    promotion_gates = {
        "uncertainty_roc_auc": float(pair_metrics["weighted_roc_auc"]) + 1e-12
        >= float(gates["minimum_uncertainty_roc_auc"]),
        "uncertainty_accuracy_gain": float(pair_metrics["accuracy_gain_over_majority"]) + 1e-12
        >= float(gates["minimum_weighted_accuracy_gain_over_majority"]),
        "normalized_regret_improves_v2": relative_regret_vs_v2 + 1e-12
        >= float(gates["minimum_relative_normalized_regret_improvement_over_frozen_v2"]),
        "normalized_regret_noninferior_to_maprank": regret_degradation_vs_maprank
        <= float(gates["maximum_absolute_normalized_regret_degradation_vs_maprank"]) + 1e-12,
        "exact_best_noninferior_to_v2": float(metrics["exact_best_rate"])
        + float(gates["exact_best_rate_noninferiority_tolerance_vs_v2"])
        + 1e-12
        >= float(anchor_metrics["exact_best_rate"]),
        "top3_noninferior_to_v2": float(metrics["top3_hit_rate"])
        + float(gates["top3_hit_rate_noninferiority_tolerance_vs_v2"])
        + 1e-12
        >= float(anchor_metrics["top3_hit_rate"]),
        "uncertain_override_fraction": float(override["uncertain_override_fraction"])
        <= float(gates["maximum_uncertain_override_fraction"]) + 1e-12,
        "directionally_safe_override_precision": float(
            override["directionally_safe_override_precision"]
        )
        + 1e-12
        >= float(gates["minimum_directionally_safe_override_precision"]),
        "maprank_override_retention": float(override["maprank_override_retention_fraction"])
        + 1e-12
        >= float(gates["minimum_maprank_override_retention_fraction"]),
        "topology_groups_noninferior_to_v2": all(row["passed"] for row in topology),
    }
    offline_passed = all(promotion_gates.values())
    output_path.mkdir(parents=True, exist_ok=True)
    prediction_path = output_path / "offline_predictions.jsonl"
    _write_jsonl(
        prediction_path,
        [
            {**row, "schema": PREDICTION_SCHEMA, "evidence_role": "nested_train_map_oof"}
            for row in nested["records"]
        ]
        + [
            {**row, "schema": PREDICTION_SCHEMA, "evidence_role": "legacy_validation_descriptive"}
            for row in validation_records
        ],
    )
    model_path = output_path / "sklearn__uncertainty.pkl"
    _atomic_pickle(model_path, final_estimator)
    report = {
        "schema": REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "scientific_status": (
            "offline_gate_passed_runtime_export_pending"
            if offline_passed
            else "nested_train_oof_gate_failed"
        ),
        "offline_passed": offline_passed,
        "runtime_bundle_exported": False,
        "shadow_eligible": False,
        "fresh_map_eligible": False,
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "legacy_validation_used_for_calibration": False,
        "high_load_used_for_training_or_calibration": False,
        "test_data_read": False,
        "formal_ood_data_read": False,
        "train_state_count": len(train_grouped),
        "train_map_count": len({str(rows[0]["map_id"]) for rows in train_grouped.values()}),
        "legacy_validation_state_count": len(validation_grouped),
        "nested_train_oof": {
            "metrics": metrics,
            "maprank_fixed_threshold_metrics": maprank_metrics,
            "v2_anchor_metrics": anchor_metrics,
            "normalized_regret_improvement_vs_v2": regret_vs_v2,
            "relative_normalized_regret_improvement_vs_v2": relative_regret_vs_v2,
            "normalized_regret_degradation_vs_maprank": regret_degradation_vs_maprank,
            "uncertainty_pair_metrics": pair_metrics,
            "override_diagnostics": override,
            "selection_changes_vs_v2": _selection_change_diagnostics(
                nested["anchor_records"], nested["records"]
            ),
            "selection_changes_vs_maprank": _selection_change_diagnostics(
                nested["maprank_records"], nested["records"]
            ),
            "topology_group_comparisons": topology,
            "folds": nested["folds"],
            "final_calibration": nested["final_calibration"],
        },
        "legacy_validation_descriptive": {
            "metrics": _mean_metrics(validation_records),
            "maprank_metrics": _mean_metrics(validation_maprank_records),
            "v2_anchor_metrics": _mean_metrics(validation_anchor_records),
            "override_diagnostics": _override_diagnostics(
                maprank_predictions=validation_maprank,
                certguard_predictions=validation_certguard,
                anchor_predictions=validation_anchor,
                lookup=uncertainty_validation["lookup"],
            ),
        },
        "promotion_gates": promotion_gates,
        "final_certainty_threshold": final_threshold,
        "frozen_direction_thresholds": direction_thresholds,
        "model_parameters": dict(config["model_parameters"]),
        "source_sha256": {
            "config": sha256_file(path),
            "design": sha256_file(design_path),
            "label_config": sha256_file(label_config_path),
            "label_report": sha256_file(label_report_path),
            "uncertainty_labels": sha256_file(labels_path),
            "maprank_manifest": sha256_file(maprank_bundle / "controller_manifest.json"),
            "maprank_training_report": sha256_file(maprank_report_path),
        },
        "artifacts": {
            "offline_predictions": prediction_path.name,
            "offline_predictions_sha256": sha256_file(prediction_path),
            "sklearn_uncertainty_model": model_path.name,
            "sklearn_uncertainty_model_sha256": sha256_file(model_path),
        },
    }
    _write_json(output_path / "certguard_training_report.json", report)
    return report


__all__ = [
    "_certguard_predictions",
    "_uncertainty_matrix",
    "run_certguard_training",
]
