from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.mixed_full_v2 import _atomic_pickle
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_anchorchoice_training import (
    _anchorchoice_predictions,
    _calibrate_selection,
    _choice_diagnostics,
)
from experiments.stride_augcontrol import (
    _input_specifications,
    _load_candidates,
    _load_pair_table,
    _mean_metrics,
    _prediction_records,
    _selection_change_diagnostics,
)
from experiments.stride_certguard_training import _maprank_predictions
from experiments.stride_guardrank import (
    _balanced_folds,
    _challenger_evidence,
    _fit_maps,
    _topology_comparisons,
)
from experiments.stride_maprank import validate_maprank_design
from experiments.stride_marginchoice import (
    CONTROLLER_ID,
    LABEL_REPORT_SCHEMA,
    LABEL_SCHEMA,
    validate_marginchoice_design,
    validate_marginchoice_label_config,
    validate_marginchoice_training_config,
)
from experiments.stride_overrideguard_training import _action_matrix, _action_vector
from experiments.stride_repairability import BUILD_SCHEMA, CONFLICT_ONLY_LABEL_SCHEMA
from experiments.stride_stage3 import _project_path
from experiments.stride_stage4 import _candidate_matrix, _frozen_predictions, _pair_matrix


REPORT_SCHEMA = "lns2.stride.marginchoice_training.v1"
PREDICTION_SCHEMA = "lns2.stride.marginchoice_prediction.v1"


def _load_margin_table(
    path: Path,
    candidate_index: dict[tuple[str, str], int],
    *,
    split: str,
) -> dict[str, Any]:
    import numpy as np

    candidates: list[int] = []
    anchors: list[int] = []
    targets: list[float] = []
    robust_labels: list[int] = []
    weights: list[float] = []
    state_ids: list[str] = []
    maps: list[str] = []
    state_weights: Counter[str] = Counter()
    state_anchors: dict[str, str] = {}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _read_jsonl(path):
        if row.get("schema") != LABEL_SCHEMA:
            raise ValueError("unexpected MarginChoice label schema")
        if str(row["split"]) != split:
            continue
        state_id = str(row["state_id"])
        map_id = str(row["map_id"])
        candidate_id = str(row["candidate_id"])
        anchor_id = str(row["anchor_candidate_id"])
        if candidate_id == anchor_id:
            raise ValueError("MarginChoice action cannot equal its anchor")
        if state_id in state_anchors and state_anchors[state_id] != anchor_id:
            raise ValueError("MarginChoice state changed V2 anchor")
        state_anchors[state_id] = anchor_id
        key = (state_id, candidate_id)
        if key in lookup:
            raise ValueError(f"duplicate MarginChoice action label: {key}")
        target = float(row["conservative_margin"])
        first = float(row["first_half_mean_effect"])
        second = float(row["second_half_mean_effect"])
        mean = float(row["mean_effect"])
        robust = int(row["robust_candidate_win"])
        weight = float(row["sample_weight"])
        if (
            not all(math.isfinite(value) for value in (target, first, second, mean, weight))
            or not math.isclose(target, min(first, second), rel_tol=0.0, abs_tol=1e-12)
            or robust not in {0, 1}
            or weight <= 0.0
        ):
            raise ValueError("MarginChoice target row is invalid")
        candidates.append(candidate_index[(state_id, candidate_id)])
        anchors.append(candidate_index[(state_id, anchor_id)])
        targets.append(target)
        robust_labels.append(robust)
        weights.append(weight)
        state_ids.append(state_id)
        maps.append(map_id)
        state_weights[state_id] += weight
        lookup[key] = {
            "label": robust,
            "label_reason": (
                "positive_robust_candidate_win"
                if robust
                else "negative_not_robust_candidate_win"
            ),
            "anchor_candidate_id": anchor_id,
            "conservative_margin": target,
        }
    if not targets or set(robust_labels) != {0, 1}:
        raise ValueError(f"MarginChoice {split} labels require both robust classes")
    if any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9)
        for value in state_weights.values()
    ):
        raise ValueError("MarginChoice action weights differ by state")
    return {
        "candidate": np.asarray(candidates, dtype=np.int32),
        "anchor": np.asarray(anchors, dtype=np.int32),
        "targets": np.asarray(targets, dtype=np.float64),
        "robust_labels": np.asarray(robust_labels, dtype=np.int8),
        "weights": np.asarray(weights, dtype=np.float64),
        "state_ids": np.asarray(state_ids, dtype=object),
        "maps": np.asarray(maps, dtype=object),
        "state_count": len(state_weights),
        "lookup": lookup,
    }


def _fit_margin_maps(
    *,
    maps: set[str],
    action_values: Any,
    table: dict[str, Any],
    parameters: dict[str, Any],
) -> Any:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    mask = np.isin(table["maps"], sorted(maps))
    if not bool(np.any(mask)):
        raise ValueError("MarginChoice map split has no labels")
    estimator = HistGradientBoostingRegressor(**parameters)
    estimator.fit(
        action_values[mask],
        table["targets"][mask],
        sample_weight=table["weights"][mask],
    )
    return estimator


def _margin_scores(
    grouped: dict[str, list[dict[str, Any]]],
    anchor_predictions: dict[str, str],
    estimator: Any,
) -> dict[str, list[dict[str, Any]]]:
    vectors: list[list[float]] = []
    keys: list[tuple[str, str]] = []
    for state_id, rows in sorted(grouped.items()):
        anchor_id = anchor_predictions[state_id]
        by_id = {str(row["candidate_id"]): row for row in rows}
        anchor = by_id[anchor_id]
        for candidate_id in sorted(by_id):
            if candidate_id == anchor_id:
                continue
            vectors.append(_action_vector(by_id[candidate_id], anchor))
            keys.append((state_id, candidate_id))
    predictions = estimator.predict(vectors)
    result: dict[str, list[dict[str, Any]]] = {state_id: [] for state_id in grouped}
    for (state_id, candidate_id), value in zip(keys, predictions):
        result[state_id].append(
            {"candidate_id": candidate_id, "safety_probability": float(value)}
        )
    if any(not rows for rows in result.values()):
        raise ValueError("MarginChoice state lacks non-anchor candidates")
    return result


def _margin_regression_metrics(
    targets: Any, predictions: Any, weights: Any
) -> dict[str, float]:
    import numpy as np

    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    target_mean = float(np.sum(weights * targets) / total)
    prediction_mean = float(np.sum(weights * predictions) / total)
    centered_target = targets - target_mean
    centered_prediction = predictions - prediction_mean
    covariance = float(np.sum(weights * centered_target * centered_prediction) / total)
    target_variance = float(np.sum(weights * centered_target**2) / total)
    prediction_variance = float(np.sum(weights * centered_prediction**2) / total)
    denominator = math.sqrt(max(0.0, target_variance * prediction_variance))
    mae = float(np.sum(weights * np.abs(predictions - targets)) / total)
    baseline_mae = float(np.sum(weights * np.abs(target_mean - targets)) / total)
    return {
        "weighted_margin_correlation": covariance / denominator if denominator else 0.0,
        "weighted_mean_absolute_error": mae,
        "weighted_constant_mean_absolute_error": baseline_mae,
        "weighted_mae_improvement": baseline_mae - mae,
        "weighted_target_mean": target_mean,
        "weighted_prediction_mean": prediction_mean,
        "weighted_action_count": total,
    }


def _nested_train_oof(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    action_values: Any,
    margin_table: dict[str, Any],
    direction_values: Any,
    direction_table: dict[str, Any],
    direction_maps: Any,
    anchor_predictions: dict[str, str],
    direction_specs: tuple[tuple[str, str], ...],
    margin_parameters: dict[str, Any],
    direction_parameters: dict[str, Any],
    direction_thresholds: dict[str, float],
    outer_count: int,
    inner_count: int,
    selection_grid: list[float],
    exact_tolerance: float,
    top3_tolerance: float,
    minimum_precision: float,
    minimum_state_override_fraction: float,
) -> dict[str, Any]:
    import numpy as np

    all_maps = {str(rows[0]["map_id"]) for rows in grouped.values()}
    choice_predictions: dict[str, str] = {}
    maprank_predictions: dict[str, str] = {}
    crossfit_scores: dict[str, list[dict[str, Any]]] = {}
    folds: list[dict[str, Any]] = []
    oof_targets: list[float] = []
    oof_predictions: list[float] = []
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
        inner_scores: dict[str, list[dict[str, Any]]] = {}
        for inner in _balanced_folds(outer_train_maps, inner_count):
            fit_maps = set(map(str, inner["train_maps"]))
            held_maps = set(map(str, inner["validation_maps"]))
            estimator = _fit_margin_maps(
                maps=fit_maps,
                action_values=action_values,
                table=margin_table,
                parameters=margin_parameters,
            )
            held_grouped = {
                state_id: rows
                for state_id, rows in inner_grouped.items()
                if str(rows[0]["map_id"]) in held_maps
            }
            held_anchor = {state_id: inner_anchor[state_id] for state_id in held_grouped}
            inner_scores.update(_margin_scores(held_grouped, held_anchor, estimator))
        if set(inner_scores) != set(inner_grouped):
            raise ValueError("MarginChoice inner OOF coverage differs")
        calibration = _calibrate_selection(
            grouped=inner_grouped,
            candidate_scores=inner_scores,
            anchor_predictions=inner_anchor,
            lookup=margin_table["lookup"],
            grid=selection_grid,
            exact_tolerance=exact_tolerance,
            top3_tolerance=top3_tolerance,
            minimum_precision=minimum_precision,
            minimum_state_override_fraction=minimum_state_override_fraction,
        )
        margin_estimator = _fit_margin_maps(
            maps=outer_train_maps,
            action_values=action_values,
            table=margin_table,
            parameters=margin_parameters,
        )
        held_grouped = {
            state_id: rows
            for state_id, rows in grouped.items()
            if str(rows[0]["map_id"]) in outer_validation_maps
        }
        held_anchor = {state_id: anchor_predictions[state_id] for state_id in held_grouped}
        held_scores = _margin_scores(held_grouped, held_anchor, margin_estimator)
        held_predictions = _anchorchoice_predictions(
            candidate_scores=held_scores,
            anchor_predictions=held_anchor,
            selection_threshold=float(calibration["selection_threshold"]),
        )
        choice_predictions.update(held_predictions)
        crossfit_scores.update(held_scores)

        direction_estimator = _fit_maps(
            maps=outer_train_maps,
            pair_maps=direction_maps,
            pair_values=direction_values,
            pair_table=direction_table,
            parameters=direction_parameters,
        )
        evidence = _challenger_evidence(
            held_grouped, direction_estimator, held_anchor, direction_specs
        )
        maprank_predictions.update(_maprank_predictions(evidence, direction_thresholds))

        margin_mask = np.isin(margin_table["maps"], sorted(outer_validation_maps))
        predictions = margin_estimator.predict(action_values[margin_mask])
        oof_targets.extend(map(float, margin_table["targets"][margin_mask]))
        oof_predictions.extend(map(float, predictions))
        oof_weights.extend(map(float, margin_table["weights"][margin_mask]))
        folds.append(
            {
                "outer_fold": int(outer["fold"]),
                "training_maps": sorted(outer_train_maps),
                "validation_maps": sorted(outer_validation_maps),
                "validation_state_count": len(held_grouped),
                "selection_threshold_calibration": calibration,
                "margin_regression_metrics": _margin_regression_metrics(
                    margin_table["targets"][margin_mask],
                    predictions,
                    margin_table["weights"][margin_mask],
                ),
            }
        )
    if (
        set(choice_predictions) != set(grouped)
        or set(maprank_predictions) != set(grouped)
        or set(crossfit_scores) != set(grouped)
    ):
        raise ValueError("MarginChoice outer OOF coverage differs")
    final_calibration = _calibrate_selection(
        grouped=grouped,
        candidate_scores=crossfit_scores,
        anchor_predictions=anchor_predictions,
        lookup=margin_table["lookup"],
        grid=selection_grid,
        exact_tolerance=exact_tolerance,
        top3_tolerance=top3_tolerance,
        minimum_precision=minimum_precision,
        minimum_state_override_fraction=minimum_state_override_fraction,
    )
    choice_records = _prediction_records(
        CONTROLLER_ID, choice_predictions, grouped, evaluation_pool="augmented"
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
        "predictions": choice_predictions,
        "maprank_predictions": maprank_predictions,
        "records": choice_records,
        "maprank_records": maprank_records,
        "anchor_records": anchor_records,
        "metrics": _mean_metrics(choice_records),
        "maprank_metrics": _mean_metrics(maprank_records),
        "anchor_metrics": _mean_metrics(anchor_records),
        "choice_diagnostics": _choice_diagnostics(
            predictions=choice_predictions,
            anchor_predictions=anchor_predictions,
            lookup=margin_table["lookup"],
        ),
        "margin_regression_metrics": _margin_regression_metrics(
            oof_targets, oof_predictions, oof_weights
        ),
        "all_fold_calibrations_feasible": all(
            bool(row["selection_threshold_calibration"]["calibration_feasible"])
            for row in folds
        ),
        "folds": folds,
        "final_calibration": final_calibration,
    }


def run_marginchoice_training(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    import numpy as np

    path = Path(config_path).resolve()
    project_root = path.parents[1]
    config = _read_json(path)
    validate_marginchoice_training_config(config)
    output_path = Path(output).resolve()
    if output_path != (project_root / str(config["registered_output"])).resolve():
        raise ValueError("STRIDE-MarginChoice training output differs from registration")
    design_path = _project_path(project_root, config["design"])
    label_config_path = _project_path(project_root, config["label_config"])
    if sha256_file(design_path) != str(config["design_sha256"]):
        raise ValueError("MarginChoice design SHA differs")
    if sha256_file(label_config_path) != str(config["label_config_sha256"]):
        raise ValueError("MarginChoice label config SHA differs")
    validate_marginchoice_design(_read_json(design_path))
    validate_marginchoice_label_config(_read_json(label_config_path))
    margin_path = _project_path(project_root, config["margin_labels"])
    margin_report_path = margin_path.parent / "marginchoice_label_report.json"
    margin_report = _read_json(margin_report_path)
    if (
        margin_report.get("schema") != LABEL_REPORT_SCHEMA
        or str(margin_report.get("config_sha256")) != str(config["label_config_sha256"])
        or str(
            dict(margin_report.get("artifacts") or {}).get(
                "conservative_margin_actions_sha256"
            )
        )
        != sha256_file(margin_path)
        or bool(margin_report.get("runtime_used_in_label"))
        or bool(margin_report.get("future_trajectory_used_in_label"))
        or bool(margin_report.get("high_load_data_read"))
    ):
        raise ValueError("MarginChoice label report is invalid")

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
            raise ValueError(f"MarginChoice registered input differs: {registered_path}")
    if _read_json(maprank_summary_path).get("schema") != BUILD_SCHEMA:
        raise ValueError("MarginChoice MapRank label summary schema differs")
    maprank_bundle = _project_path(project_root, config["maprank_bundle"])
    maprank_report_path = _project_path(project_root, config["maprank_training_report"])
    if sha256_file(maprank_bundle / "controller_manifest.json") != str(
        config["maprank_manifest_sha256"]
    ):
        raise ValueError("MarginChoice MapRank manifest differs")
    if sha256_file(maprank_report_path) != str(config["maprank_training_report_sha256"]):
        raise ValueError("MarginChoice MapRank report differs")
    maprank_report = _read_json(maprank_report_path)
    direction_thresholds = {
        name: float(value)
        for name, value in dict(maprank_report["final_thresholds"]).items()
    }
    v2_bundle = _project_path(
        project_root, "artifacts/initlns-closed-loop-controller-v2"
    )
    if sha256_file(v2_bundle / "controller_manifest.json") != str(
        config["frozen_v2_manifest_sha256"]
    ):
        raise ValueError("MarginChoice V2 manifest differs")
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
    margin_train = _load_margin_table(margin_path, candidate_index, split="train")
    margin_validation = _load_margin_table(
        margin_path, candidate_index, split="validation"
    )
    direction_table = _load_pair_table(
        direction_path,
        candidate_index,
        split="train",
        expected_schema=CONFLICT_ONLY_LABEL_SCHEMA,
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
    train_anchor = _frozen_predictions(train_grouped, v2_bundle)
    validation_anchor = _frozen_predictions(validation_grouped, v2_bundle)
    candidate_values = _candidate_matrix(candidates)
    action_train_values = _action_matrix(candidate_values, margin_train)
    _, direction_specs = _input_specifications()
    direction_values = _pair_matrix(candidate_values, direction_table, direction_specs)
    direction_maps = np.asarray(
        [str(candidates[int(index)]["map_id"]) for index in direction_table["left"]],
        dtype=object,
    )
    constraints = dict(config["calibration_constraints"])
    gates = dict(config["offline_gates"])
    nested = _nested_train_oof(
        grouped=train_grouped,
        action_values=action_train_values,
        margin_table=margin_train,
        direction_values=direction_values,
        direction_table=direction_table,
        direction_maps=direction_maps,
        anchor_predictions=train_anchor,
        direction_specs=direction_specs,
        margin_parameters=dict(config["model_parameters"]),
        direction_parameters=dict(maprank_report["model_parameters"]),
        direction_thresholds=direction_thresholds,
        outer_count=int(config["outer_map_folds"]),
        inner_count=int(config["inner_map_folds"]),
        selection_grid=list(map(float, config["selection_threshold_grid"])),
        exact_tolerance=float(
            constraints["exact_best_rate_noninferiority_tolerance_vs_v2"]
        ),
        top3_tolerance=float(
            constraints["top3_hit_rate_noninferiority_tolerance_vs_v2"]
        ),
        minimum_precision=float(
            constraints["minimum_directionally_safe_override_precision"]
        ),
        minimum_state_override_fraction=float(
            constraints["minimum_state_override_fraction"]
        ),
    )
    final_estimator = _fit_margin_maps(
        maps={str(rows[0]["map_id"]) for rows in train_grouped.values()},
        action_values=action_train_values,
        table=margin_train,
        parameters=dict(config["model_parameters"]),
    )
    final_threshold = float(nested["final_calibration"]["selection_threshold"])
    validation_scores = _margin_scores(
        validation_grouped, validation_anchor, final_estimator
    )
    validation_predictions = _anchorchoice_predictions(
        candidate_scores=validation_scores,
        anchor_predictions=validation_anchor,
        selection_threshold=final_threshold,
    )

    from lns2_selector.controllers import load_selector
    from lns2_selector.runtime.contracts import SelectionRequest

    maprank_selector = load_selector("stride-maprank-v1", maprank_bundle)
    validation_maprank: dict[str, str] = {}
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
        validation_maprank[state_id] = str(
            rows[int(decision.candidate_index)]["candidate_id"]
        )
    validation_records = _prediction_records(
        CONTROLLER_ID,
        validation_predictions,
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
    regression = dict(nested["margin_regression_metrics"])
    choice = dict(nested["choice_diagnostics"])
    promotion_gates = {
        "all_nested_calibrations_feasible": bool(
            nested["all_fold_calibrations_feasible"]
        )
        and bool(nested["final_calibration"]["calibration_feasible"]),
        "weighted_margin_correlation": float(
            regression["weighted_margin_correlation"]
        )
        + 1e-12
        >= float(gates["minimum_weighted_margin_correlation"]),
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
        "unsafe_override_fraction": float(choice["unsafe_override_fraction"])
        <= float(gates["maximum_unsafe_override_fraction"]) + 1e-12,
        "directionally_safe_override_precision": float(
            choice["directionally_safe_override_precision"]
        )
        + 1e-12
        >= float(gates["minimum_directionally_safe_override_precision"]),
        "state_override_fraction": float(choice["state_override_fraction"]) + 1e-12
        >= float(gates["minimum_state_override_fraction"]),
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
    model_path = output_path / "sklearn__conservative_margin.pkl"
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
            "margin_regression_metrics": regression,
            "choice_diagnostics": choice,
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
            "choice_diagnostics": _choice_diagnostics(
                predictions=validation_predictions,
                anchor_predictions=validation_anchor,
                lookup=margin_validation["lookup"],
            ),
        },
        "promotion_gates": promotion_gates,
        "final_selection_threshold": final_threshold,
        "model_parameters": dict(config["model_parameters"]),
        "source_sha256": {
            "config": sha256_file(path),
            "design": sha256_file(design_path),
            "label_config": sha256_file(label_config_path),
            "margin_labels": sha256_file(margin_path),
            "margin_label_report": sha256_file(margin_report_path),
            "maprank_manifest": sha256_file(maprank_bundle / "controller_manifest.json"),
            "maprank_training_report": sha256_file(maprank_report_path),
            "v2_manifest": sha256_file(v2_bundle / "controller_manifest.json"),
        },
        "artifacts": {
            "offline_predictions": prediction_path.name,
            "offline_predictions_sha256": sha256_file(prediction_path),
            "sklearn_conservative_margin_model": model_path.name,
            "sklearn_conservative_margin_model_sha256": sha256_file(model_path),
        },
    }
    _write_json(output_path / "marginchoice_training_report.json", report)
    return report


__all__ = [
    "_margin_regression_metrics",
    "_margin_scores",
    "run_marginchoice_training",
]
