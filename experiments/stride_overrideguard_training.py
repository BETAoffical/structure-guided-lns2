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
from experiments.stride_certguard_training import (
    _maprank_predictions,
    _uncertainty_metrics,
)
from experiments.stride_guardrank import (
    _balanced_folds,
    _challenger_evidence,
    _fit_maps,
    _topology_comparisons,
)
from experiments.stride_maprank import validate_maprank_design
from experiments.stride_overrideguard import (
    CONTROLLER_ID,
    LABEL_REPORT_SCHEMA,
    LABEL_SCHEMA,
    validate_overrideguard_design,
    validate_overrideguard_label_config,
    validate_overrideguard_training_config,
)
from experiments.stride_repairability import BUILD_SCHEMA, CONFLICT_ONLY_LABEL_SCHEMA
from experiments.stride_stage3 import _project_path
from experiments.stride_stage4 import (
    _candidate_matrix,
    _fit_registered_model,
    _frozen_predictions,
    _pair_matrix,
)


REPORT_SCHEMA = "lns2.stride.overrideguard_training.v1"
PREDICTION_SCHEMA = "lns2.stride.overrideguard_prediction.v1"
REGISTERED_OUTPUT = "build/stride-overrideguard-training-v1"


def _load_action_table(
    path: Path,
    candidate_index: dict[tuple[str, str], int],
    *,
    split: str,
) -> dict[str, Any]:
    import numpy as np

    candidates: list[int] = []
    anchors: list[int] = []
    labels: list[int] = []
    weights: list[float] = []
    state_ids: list[str] = []
    maps: list[str] = []
    state_weights: Counter[str] = Counter()
    state_anchors: dict[str, str] = {}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("schema") != LABEL_SCHEMA:
            raise ValueError("unexpected OverrideGuard action label schema")
        if str(row["split"]) != split:
            continue
        state_id = str(row["state_id"])
        map_id = str(row["map_id"])
        candidate_id = str(row["candidate_id"])
        anchor_id = str(row["anchor_candidate_id"])
        if candidate_id == anchor_id:
            raise ValueError("OverrideGuard action label cannot target its anchor")
        if state_id in state_anchors and state_anchors[state_id] != anchor_id:
            raise ValueError("OverrideGuard state changed V2 anchor")
        state_anchors[state_id] = anchor_id
        key = (state_id, candidate_id)
        if key in lookup:
            raise ValueError(f"duplicate OverrideGuard action label: {key}")
        label = int(row["label"])
        reason = str(row["label_reason"])
        if label not in {0, 1} or (label == 1) != (
            reason == "positive_robust_candidate_win"
        ):
            raise ValueError("OverrideGuard action target is inconsistent")
        if reason not in {
            "positive_robust_candidate_win",
            "negative_uncertain_pair",
            "negative_robust_anchor_win",
        }:
            raise ValueError("OverrideGuard action reason is invalid")
        weight = float(row["sample_weight"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError("OverrideGuard action weight is invalid")
        candidates.append(candidate_index[(state_id, candidate_id)])
        anchors.append(candidate_index[(state_id, anchor_id)])
        labels.append(label)
        weights.append(weight)
        state_ids.append(state_id)
        maps.append(map_id)
        state_weights[state_id] += weight
        lookup[key] = {
            "label": label,
            "label_reason": reason,
            "anchor_candidate_id": anchor_id,
        }
    if not labels or set(labels) != {0, 1}:
        raise ValueError(f"OverrideGuard {split} action labels require both classes")
    if any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9)
        for value in state_weights.values()
    ):
        raise ValueError("OverrideGuard action weights differ by state")
    return {
        "candidate": np.asarray(candidates, dtype=np.int32),
        "anchor": np.asarray(anchors, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int8),
        "weights": np.asarray(weights, dtype=np.float64),
        "state_ids": np.asarray(state_ids, dtype=object),
        "maps": np.asarray(maps, dtype=object),
        "state_count": len(state_weights),
        "lookup": lookup,
    }


def _action_matrix(candidate_values: Any, table: dict[str, Any]) -> Any:
    import numpy as np

    candidate = candidate_values[table["candidate"]]
    anchor = candidate_values[table["anchor"]]
    return np.concatenate((candidate - anchor, 0.5 * (candidate + anchor)), axis=1).astype(
        np.float32, copy=False
    )


def _fit_action_maps(
    *,
    maps: set[str],
    action_values: Any,
    table: dict[str, Any],
    parameters: dict[str, Any],
) -> Any:
    import numpy as np

    mask = np.isin(table["maps"], sorted(maps))
    if not bool(np.any(mask)):
        raise ValueError("OverrideGuard map split has no action labels")
    return _fit_registered_model(
        action_values[mask], table["labels"][mask], table["weights"][mask], parameters
    )


def _action_vector(candidate: dict[str, Any], anchor: dict[str, Any]) -> list[float]:
    names = PROFILE_FEATURE_NAMES["realized_dynamic"]
    candidate_values = [float(candidate["features"][name]) for name in names]
    anchor_values = [float(anchor["features"][name]) for name in names]
    return [
        value - base for value, base in zip(candidate_values, anchor_values)
    ] + [
        0.5 * (value + base) for value, base in zip(candidate_values, anchor_values)
    ]


def _action_probabilities(
    grouped: dict[str, list[dict[str, Any]]],
    maprank_predictions: dict[str, str],
    anchor_predictions: dict[str, str],
    estimator: Any,
) -> dict[str, float]:
    result: dict[str, float] = {}
    vectors: list[list[float]] = []
    states: list[str] = []
    for state_id, rows in sorted(grouped.items()):
        candidate_id = maprank_predictions[state_id]
        anchor_id = anchor_predictions[state_id]
        if candidate_id == anchor_id:
            result[state_id] = 1.0
            continue
        by_id = {str(row["candidate_id"]): row for row in rows}
        vectors.append(_action_vector(by_id[candidate_id], by_id[anchor_id]))
        states.append(state_id)
    if vectors:
        probabilities = estimator.predict_proba(vectors)[:, 1]
        result.update({state: float(value) for state, value in zip(states, probabilities)})
    return result


def _overrideguard_predictions(
    *,
    maprank_predictions: dict[str, str],
    anchor_predictions: dict[str, str],
    safety_probabilities: dict[str, float],
    safety_threshold: float,
) -> dict[str, str]:
    return {
        state_id: (
            maprank_predictions[state_id]
            if maprank_predictions[state_id] == anchor_predictions[state_id]
            or float(safety_probabilities[state_id]) + 1e-12 >= safety_threshold
            else anchor_predictions[state_id]
        )
        for state_id in maprank_predictions
    }


def _override_diagnostics(
    *,
    maprank_predictions: dict[str, str],
    overrideguard_predictions: dict[str, str],
    anchor_predictions: dict[str, str],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    overrides = [
        state_id
        for state_id in sorted(anchor_predictions)
        if maprank_predictions[state_id] != anchor_predictions[state_id]
    ]
    retained = [
        state_id
        for state_id in overrides
        if overrideguard_predictions[state_id] == maprank_predictions[state_id]
    ]
    reasons: Counter[str] = Counter()
    safe = 0
    for state_id in retained:
        candidate_id = maprank_predictions[state_id]
        row = lookup[(state_id, candidate_id)]
        if str(row["anchor_candidate_id"]) != anchor_predictions[state_id]:
            raise ValueError("OverrideGuard diagnostic anchor differs")
        reasons[str(row["label_reason"])] += 1
        safe += int(row["label"]) == 1
    retained_count = len(retained)
    precision = safe / retained_count if retained_count else 0.0
    return {
        "state_count": len(anchor_predictions),
        "maprank_override_count": len(overrides),
        "retained_override_count": retained_count,
        "rejected_override_count": len(overrides) - retained_count,
        "maprank_override_retention_fraction": (
            retained_count / len(overrides) if overrides else 1.0
        ),
        "directionally_safe_override_count": safe,
        "unsafe_override_count": retained_count - safe,
        "unsafe_override_fraction": 1.0 - precision if retained_count else 0.0,
        "directionally_safe_override_precision": precision,
        "retained_label_reason_counts": dict(sorted(reasons.items())),
    }


def _calibrate_safety(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    maprank_predictions: dict[str, str],
    anchor_predictions: dict[str, str],
    safety_probabilities: dict[str, float],
    lookup: dict[tuple[str, str], dict[str, Any]],
    grid: list[float],
    exact_tolerance: float,
    top3_tolerance: float,
    minimum_precision: float,
    minimum_retention: float,
) -> dict[str, Any]:
    anchor_records = _prediction_records(
        "v2-full/anchor", anchor_predictions, grouped, evaluation_pool="augmented"
    )
    anchor_metrics = _mean_metrics(anchor_records)
    rows: list[dict[str, Any]] = []
    for threshold in grid:
        predictions = _overrideguard_predictions(
            maprank_predictions=maprank_predictions,
            anchor_predictions=anchor_predictions,
            safety_probabilities=safety_probabilities,
            safety_threshold=threshold,
        )
        records = _prediction_records(
            CONTROLLER_ID, predictions, grouped, evaluation_pool="augmented"
        )
        metrics = _mean_metrics(records)
        diagnostics = _override_diagnostics(
            maprank_predictions=maprank_predictions,
            overrideguard_predictions=predictions,
            anchor_predictions=anchor_predictions,
            lookup=lookup,
        )
        quality_passed = (
            float(metrics["exact_best_rate"]) + exact_tolerance + 1e-12
            >= float(anchor_metrics["exact_best_rate"])
            and float(metrics["top3_hit_rate"]) + top3_tolerance + 1e-12
            >= float(anchor_metrics["top3_hit_rate"])
        )
        safety_passed = (
            float(diagnostics["directionally_safe_override_precision"]) + 1e-12
            >= minimum_precision
            and float(diagnostics["maprank_override_retention_fraction"]) + 1e-12
            >= minimum_retention
        )
        rows.append(
            {
                "safety_threshold": threshold,
                "metrics": metrics,
                "quality_constraints_passed": quality_passed,
                "safety_constraints_passed": safety_passed,
                "constraints_passed": quality_passed and safety_passed,
                "override_diagnostics": diagnostics,
            }
        )
    eligible = [row for row in rows if row["constraints_passed"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["metrics"]["mean_normalized_repairability_regret"]),
                float(row["override_diagnostics"]["unsafe_override_fraction"]),
                -float(row["override_diagnostics"]["maprank_override_retention_fraction"]),
                -float(row["safety_threshold"]),
            ),
        )
        feasible = True
    else:
        fallback = [row for row in rows if float(row["safety_threshold"]) > 1.0]
        if len(fallback) != 1:
            raise ValueError("OverrideGuard threshold grid lacks a unique V2 fallback")
        selected = fallback[0]
        feasible = False
    return {
        **selected,
        "calibration_feasible": feasible,
        "grid_size": len(rows),
        "eligible_grid_count": len(eligible),
        "anchor_metrics": anchor_metrics,
        "threshold_sweep": [
            {
                "safety_threshold": float(row["safety_threshold"]),
                "constraints_passed": bool(row["constraints_passed"]),
                "quality_constraints_passed": bool(row["quality_constraints_passed"]),
                "safety_constraints_passed": bool(row["safety_constraints_passed"]),
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


def _nested_train_oof(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    direction_values: Any,
    direction_table: dict[str, Any],
    direction_maps: Any,
    action_values: Any,
    action_table: dict[str, Any],
    anchor_predictions: dict[str, str],
    direction_specs: tuple[tuple[str, str], ...],
    direction_parameters: dict[str, Any],
    action_parameters: dict[str, Any],
    direction_thresholds: dict[str, float],
    outer_count: int,
    inner_count: int,
    safety_grid: list[float],
    exact_tolerance: float,
    top3_tolerance: float,
    minimum_precision: float,
    minimum_retention: float,
) -> dict[str, Any]:
    import numpy as np

    all_maps = {str(rows[0]["map_id"]) for rows in grouped.values()}
    guarded_predictions: dict[str, str] = {}
    maprank_predictions: dict[str, str] = {}
    crossfit_probabilities: dict[str, float] = {}
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
        inner_maprank: dict[str, str] = {}
        inner_probabilities: dict[str, float] = {}
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
            action_estimator = _fit_action_maps(
                maps=fit_maps,
                action_values=action_values,
                table=action_table,
                parameters=action_parameters,
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
            held_maprank = _maprank_predictions(evidence, direction_thresholds)
            inner_maprank.update(held_maprank)
            inner_probabilities.update(
                _action_probabilities(
                    held_grouped, held_maprank, held_anchor, action_estimator
                )
            )
        if set(inner_maprank) != set(inner_grouped) or set(inner_probabilities) != set(
            inner_grouped
        ):
            raise ValueError("OverrideGuard inner OOF coverage differs")
        calibration = _calibrate_safety(
            grouped=inner_grouped,
            maprank_predictions=inner_maprank,
            anchor_predictions=inner_anchor,
            safety_probabilities=inner_probabilities,
            lookup=action_table["lookup"],
            grid=safety_grid,
            exact_tolerance=exact_tolerance,
            top3_tolerance=top3_tolerance,
            minimum_precision=minimum_precision,
            minimum_retention=minimum_retention,
        )
        direction_estimator = _fit_maps(
            maps=outer_train_maps,
            pair_maps=direction_maps,
            pair_values=direction_values,
            pair_table=direction_table,
            parameters=direction_parameters,
        )
        action_estimator = _fit_action_maps(
            maps=outer_train_maps,
            action_values=action_values,
            table=action_table,
            parameters=action_parameters,
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
        held_probabilities = _action_probabilities(
            held_grouped, held_maprank, held_anchor, action_estimator
        )
        held_guarded = _overrideguard_predictions(
            maprank_predictions=held_maprank,
            anchor_predictions=held_anchor,
            safety_probabilities=held_probabilities,
            safety_threshold=float(calibration["safety_threshold"]),
        )
        maprank_predictions.update(held_maprank)
        guarded_predictions.update(held_guarded)
        crossfit_probabilities.update(held_probabilities)
        action_mask = np.isin(action_table["maps"], sorted(outer_validation_maps))
        probabilities = action_estimator.predict_proba(action_values[action_mask])[:, 1]
        oof_labels.extend(map(int, action_table["labels"][action_mask]))
        oof_probabilities.extend(map(float, probabilities))
        oof_weights.extend(map(float, action_table["weights"][action_mask]))
        folds.append(
            {
                "outer_fold": int(outer["fold"]),
                "training_maps": sorted(outer_train_maps),
                "validation_maps": sorted(outer_validation_maps),
                "validation_state_count": len(held_grouped),
                "safety_threshold_calibration": calibration,
                "action_classifier_metrics": _uncertainty_metrics(
                    action_table["labels"][action_mask],
                    probabilities,
                    action_table["weights"][action_mask],
                ),
            }
        )
    if set(guarded_predictions) != set(grouped) or set(maprank_predictions) != set(grouped):
        raise ValueError("OverrideGuard outer OOF coverage differs")
    final_calibration = _calibrate_safety(
        grouped=grouped,
        maprank_predictions=maprank_predictions,
        anchor_predictions=anchor_predictions,
        safety_probabilities=crossfit_probabilities,
        lookup=action_table["lookup"],
        grid=safety_grid,
        exact_tolerance=exact_tolerance,
        top3_tolerance=top3_tolerance,
        minimum_precision=minimum_precision,
        minimum_retention=minimum_retention,
    )
    guarded_records = _prediction_records(
        CONTROLLER_ID, guarded_predictions, grouped, evaluation_pool="augmented"
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
        "predictions": guarded_predictions,
        "maprank_predictions": maprank_predictions,
        "records": guarded_records,
        "maprank_records": maprank_records,
        "anchor_records": anchor_records,
        "metrics": _mean_metrics(guarded_records),
        "maprank_metrics": _mean_metrics(maprank_records),
        "anchor_metrics": _mean_metrics(anchor_records),
        "override_diagnostics": _override_diagnostics(
            maprank_predictions=maprank_predictions,
            overrideguard_predictions=guarded_predictions,
            anchor_predictions=anchor_predictions,
            lookup=action_table["lookup"],
        ),
        "maprank_override_diagnostics": _override_diagnostics(
            maprank_predictions=maprank_predictions,
            overrideguard_predictions=maprank_predictions,
            anchor_predictions=anchor_predictions,
            lookup=action_table["lookup"],
        ),
        "action_classifier_metrics": _uncertainty_metrics(
            oof_labels, oof_probabilities, oof_weights
        ),
        "all_fold_calibrations_feasible": all(
            bool(row["safety_threshold_calibration"]["calibration_feasible"])
            for row in folds
        ),
        "folds": folds,
        "final_calibration": final_calibration,
    }


def run_overrideguard_training(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    import numpy as np

    path = Path(config_path).resolve()
    project_root = path.parents[1]
    output_path = Path(output).resolve()
    if output_path != (project_root / REGISTERED_OUTPUT).resolve():
        raise ValueError("STRIDE-OverrideGuard training output differs from registration")
    config = _read_json(path)
    validate_overrideguard_training_config(config)
    design_path = _project_path(project_root, config["design"])
    label_config_path = _project_path(project_root, config["label_config"])
    if sha256_file(design_path) != str(config["design_sha256"]):
        raise ValueError("OverrideGuard design SHA differs")
    if sha256_file(label_config_path) != str(config["label_config_sha256"]):
        raise ValueError("OverrideGuard label config SHA differs")
    validate_overrideguard_design(_read_json(design_path))
    validate_overrideguard_label_config(_read_json(label_config_path))
    labels_path = _project_path(project_root, config["action_labels"])
    label_report_path = labels_path.parent / "overrideguard_label_report.json"
    label_report = _read_json(label_report_path)
    if (
        label_report.get("schema") != LABEL_REPORT_SCHEMA
        or str(label_report.get("config_sha256")) != str(config["label_config_sha256"])
        or str(dict(label_report.get("artifacts") or {}).get("action_safety_pairs_sha256"))
        != sha256_file(labels_path)
        or bool(label_report.get("runtime_used_in_label"))
        or bool(label_report.get("future_trajectory_used_in_label"))
        or bool(label_report.get("high_load_data_read"))
    ):
        raise ValueError("OverrideGuard label report is invalid")

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
            raise ValueError(f"OverrideGuard registered input differs: {registered_path}")
    if _read_json(maprank_summary_path).get("schema") != BUILD_SCHEMA:
        raise ValueError("OverrideGuard MapRank label summary schema differs")
    maprank_bundle = _project_path(project_root, config["maprank_bundle"])
    maprank_report_path = _project_path(project_root, config["maprank_training_report"])
    if sha256_file(maprank_bundle / "controller_manifest.json") != str(
        config["maprank_manifest_sha256"]
    ):
        raise ValueError("OverrideGuard frozen MapRank manifest differs")
    if sha256_file(maprank_report_path) != str(config["maprank_training_report_sha256"]):
        raise ValueError("OverrideGuard frozen MapRank report differs")
    maprank_report = _read_json(maprank_report_path)
    direction_thresholds = {
        name: float(value)
        for name, value in dict(
            config["frozen_proposal"]["runtime_thresholds"]
        ).items()
    }
    if {
        name: float(value)
        for name, value in dict(maprank_report["final_thresholds"]).items()
    } != direction_thresholds:
        raise ValueError("OverrideGuard registered MapRank thresholds differ")
    validate_maprank_design(
        _read_json(_project_path(project_root, "configs/stride_maprank_design.json"))
    )

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
    action_train = _load_action_table(labels_path, candidate_index, split="train")
    action_validation = _load_action_table(
        labels_path, candidate_index, split="validation"
    )
    train_grouped = {
        state_id: rows
        for state_id, rows in grouped.items()
        if str(rows[0]["split"]) == "train"
    }
    validation_grouped = {
        state_id: rows
        for state_id, rows in grouped.items()
        if str(rows[0]["split"]) == "validation"
    }
    v2_bundle = _project_path(
        project_root, "artifacts/initlns-closed-loop-controller-v2"
    )
    train_anchor = _frozen_predictions(train_grouped, v2_bundle)
    validation_anchor = _frozen_predictions(validation_grouped, v2_bundle)
    _, direction_specs = _input_specifications()
    candidate_values = _candidate_matrix(candidates)
    direction_values = _pair_matrix(candidate_values, direction_table, direction_specs)
    direction_maps = np.asarray(
        [str(candidates[int(index)]["map_id"]) for index in direction_table["left"]],
        dtype=object,
    )
    action_train_values = _action_matrix(candidate_values, action_train)
    gates = dict(config["offline_gates"])
    constraints = dict(config["calibration_constraints"])
    nested = _nested_train_oof(
        grouped=train_grouped,
        direction_values=direction_values,
        direction_table=direction_table,
        direction_maps=direction_maps,
        action_values=action_train_values,
        action_table=action_train,
        anchor_predictions=train_anchor,
        direction_specs=direction_specs,
        direction_parameters=dict(maprank_report["model_parameters"]),
        action_parameters=dict(config["model_parameters"]),
        direction_thresholds=direction_thresholds,
        outer_count=int(config["outer_map_folds"]),
        inner_count=int(config["inner_map_folds"]),
        safety_grid=list(map(float, config["safety_threshold_grid"])),
        exact_tolerance=float(
            constraints["exact_best_rate_noninferiority_tolerance_vs_v2"]
        ),
        top3_tolerance=float(
            constraints["top3_hit_rate_noninferiority_tolerance_vs_v2"]
        ),
        minimum_precision=float(
            constraints["minimum_directionally_safe_override_precision"]
        ),
        minimum_retention=float(
            constraints["minimum_maprank_override_retention_fraction"]
        ),
    )
    final_estimator = _fit_registered_model(
        action_train_values,
        action_train["labels"],
        action_train["weights"],
        dict(config["model_parameters"]),
    )

    from lns2_selector.controllers import load_selector
    from lns2_selector.runtime.contracts import SelectionRequest

    maprank_selector = load_selector("stride-maprank-v1", maprank_bundle)
    validation_maprank: dict[str, str] = {}
    validation_probabilities: dict[str, float] = {}
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
        validation_probabilities[state_id] = (
            1.0
            if selected == anchor
            else float(
                final_estimator.predict_proba(
                    [_action_vector(by_id[selected], by_id[anchor])]
                )[0, 1]
            )
        )
    final_threshold = float(nested["final_calibration"]["safety_threshold"])
    validation_guarded = _overrideguard_predictions(
        maprank_predictions=validation_maprank,
        anchor_predictions=validation_anchor,
        safety_probabilities=validation_probabilities,
        safety_threshold=final_threshold,
    )
    validation_records = _prediction_records(
        CONTROLLER_ID,
        validation_guarded,
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
        "v2-full/anchor",
        validation_anchor,
        validation_grouped,
        evaluation_pool="augmented",
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
    regret_degradation_vs_maprank = float(
        metrics["mean_normalized_repairability_regret"]
    ) - float(maprank_metrics["mean_normalized_repairability_regret"])
    topology = _topology_comparisons(
        nested["records"],
        nested["anchor_records"],
        float(gates["maximum_topology_group_normalized_regret_degradation_vs_v2"]),
    )
    classifier = dict(nested["action_classifier_metrics"])
    override = dict(nested["override_diagnostics"])
    promotion_gates = {
        "all_nested_calibrations_feasible": bool(
            nested["all_fold_calibrations_feasible"]
        )
        and bool(nested["final_calibration"]["calibration_feasible"]),
        "safety_roc_auc": float(classifier["weighted_roc_auc"]) + 1e-12
        >= float(gates["minimum_safety_roc_auc"]),
        "safety_accuracy_gain": float(classifier["accuracy_gain_over_majority"])
        + 1e-12
        >= float(gates["minimum_weighted_accuracy_gain_over_majority"]),
        "normalized_regret_improves_v2": relative_regret_vs_v2 + 1e-12
        >= float(gates["minimum_relative_normalized_regret_improvement_over_frozen_v2"]),
        "normalized_regret_noninferior_to_maprank": regret_degradation_vs_maprank
        <= float(gates["maximum_absolute_normalized_regret_degradation_vs_maprank"])
        + 1e-12,
        "exact_best_noninferior_to_v2": float(metrics["exact_best_rate"])
        + float(gates["exact_best_rate_noninferiority_tolerance_vs_v2"])
        + 1e-12
        >= float(anchor_metrics["exact_best_rate"]),
        "top3_noninferior_to_v2": float(metrics["top3_hit_rate"])
        + float(gates["top3_hit_rate_noninferiority_tolerance_vs_v2"])
        + 1e-12
        >= float(anchor_metrics["top3_hit_rate"]),
        "unsafe_override_fraction": float(override["unsafe_override_fraction"])
        <= float(gates["maximum_unsafe_override_fraction"]) + 1e-12,
        "directionally_safe_override_precision": float(
            override["directionally_safe_override_precision"]
        )
        + 1e-12
        >= float(gates["minimum_directionally_safe_override_precision"]),
        "maprank_override_retention": float(
            override["maprank_override_retention_fraction"]
        )
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
            {
                **row,
                "schema": PREDICTION_SCHEMA,
                "evidence_role": "legacy_validation_descriptive",
            }
            for row in validation_records
        ],
    )
    model_path = output_path / "sklearn__action_safety.pkl"
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
        "wall_clock_evidence_collected": False,
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
            "action_classifier_metrics": classifier,
            "override_diagnostics": override,
            "maprank_override_diagnostics": nested[
                "maprank_override_diagnostics"
            ],
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
                overrideguard_predictions=validation_guarded,
                anchor_predictions=validation_anchor,
                lookup=action_validation["lookup"],
            ),
            "maprank_override_diagnostics": _override_diagnostics(
                maprank_predictions=validation_maprank,
                overrideguard_predictions=validation_maprank,
                anchor_predictions=validation_anchor,
                lookup=action_validation["lookup"],
            ),
        },
        "promotion_gates": promotion_gates,
        "final_safety_threshold": final_threshold,
        "frozen_maprank_thresholds": direction_thresholds,
        "model_parameters": dict(config["model_parameters"]),
        "source_sha256": {
            "config": sha256_file(path),
            "design": sha256_file(design_path),
            "label_config": sha256_file(label_config_path),
            "label_report": sha256_file(label_report_path),
            "action_labels": sha256_file(labels_path),
            "maprank_manifest": sha256_file(maprank_bundle / "controller_manifest.json"),
            "maprank_training_report": sha256_file(maprank_report_path),
        },
        "artifacts": {
            "offline_predictions": prediction_path.name,
            "offline_predictions_sha256": sha256_file(prediction_path),
            "sklearn_action_safety_model": model_path.name,
            "sklearn_action_safety_model_sha256": sha256_file(model_path),
        },
    }
    _write_json(output_path / "overrideguard_training_report.json", report)
    return report


__all__ = [
    "_action_matrix",
    "_overrideguard_predictions",
    "run_overrideguard_training",
]
