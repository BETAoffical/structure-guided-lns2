from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.mixed_full_v2 import _atomic_pickle
from experiments.repair_collection import _read_json, _write_json, _write_jsonl
from experiments.stride_anchorchoice import (
    CONTROLLER_ID,
    validate_anchorchoice_design,
    validate_anchorchoice_training_config,
)
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
from experiments.stride_overrideguard import LABEL_REPORT_SCHEMA
from experiments.stride_overrideguard_training import (
    _action_matrix,
    _action_vector,
    _fit_action_maps,
    _load_action_table,
)
from experiments.stride_repairability import BUILD_SCHEMA, CONFLICT_ONLY_LABEL_SCHEMA
from experiments.stride_stage3 import _project_path
from experiments.stride_stage4 import _candidate_matrix, _frozen_predictions, _pair_matrix


REPORT_SCHEMA = "lns2.stride.anchorchoice_training.v1"
PREDICTION_SCHEMA = "lns2.stride.anchorchoice_prediction.v1"


def _candidate_scores(
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
    probabilities = estimator.predict_proba(vectors)[:, 1]
    result: dict[str, list[dict[str, Any]]] = {state_id: [] for state_id in grouped}
    for (state_id, candidate_id), probability in zip(keys, probabilities):
        result[state_id].append(
            {"candidate_id": candidate_id, "safety_probability": float(probability)}
        )
    if any(not rows for rows in result.values()):
        raise ValueError("AnchorChoice state lacks a non-anchor candidate")
    return result


def _anchorchoice_predictions(
    *,
    candidate_scores: dict[str, list[dict[str, Any]]],
    anchor_predictions: dict[str, str],
    selection_threshold: float,
) -> dict[str, str]:
    predictions: dict[str, str] = {}
    for state_id, rows in sorted(candidate_scores.items()):
        best = min(
            rows,
            key=lambda row: (
                -float(row["safety_probability"]),
                str(row["candidate_id"]),
            ),
        )
        predictions[state_id] = (
            str(best["candidate_id"])
            if float(best["safety_probability"]) + 1e-12 >= selection_threshold
            else anchor_predictions[state_id]
        )
    return predictions


def _choice_diagnostics(
    *,
    predictions: dict[str, str],
    anchor_predictions: dict[str, str],
    lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    states = sorted(anchor_predictions)
    overrides = [
        state_id
        for state_id in states
        if predictions[state_id] != anchor_predictions[state_id]
    ]
    opportunity_states = {
        state_id
        for state_id in states
        if any(
            key[0] == state_id and int(row["label"]) == 1
            for key, row in lookup.items()
        )
    }
    safe = 0
    reasons: Counter[str] = Counter()
    for state_id in overrides:
        selected = predictions[state_id]
        row = lookup[(state_id, selected)]
        if str(row["anchor_candidate_id"]) != anchor_predictions[state_id]:
            raise ValueError("AnchorChoice diagnostic anchor differs")
        safe += int(row["label"]) == 1
        reasons[str(row["label_reason"])] += 1
    count = len(overrides)
    precision = safe / count if count else 0.0
    return {
        "state_count": len(states),
        "robust_opportunity_state_count": len(opportunity_states),
        "override_count": count,
        "state_override_fraction": count / len(states),
        "directionally_safe_override_count": safe,
        "unsafe_override_count": count - safe,
        "unsafe_override_fraction": 1.0 - precision if count else 0.0,
        "directionally_safe_override_precision": precision,
        "robust_opportunity_recall": (
            safe / len(opportunity_states) if opportunity_states else 1.0
        ),
        "selected_label_reason_counts": dict(sorted(reasons.items())),
    }


def _calibrate_selection(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    candidate_scores: dict[str, list[dict[str, Any]]],
    anchor_predictions: dict[str, str],
    lookup: dict[tuple[str, str], dict[str, Any]],
    grid: list[float],
    exact_tolerance: float,
    top3_tolerance: float,
    minimum_precision: float,
    minimum_state_override_fraction: float,
) -> dict[str, Any]:
    anchor_records = _prediction_records(
        "v2-full/anchor", anchor_predictions, grouped, evaluation_pool="augmented"
    )
    anchor_metrics = _mean_metrics(anchor_records)
    rows: list[dict[str, Any]] = []
    for threshold in grid:
        predictions = _anchorchoice_predictions(
            candidate_scores=candidate_scores,
            anchor_predictions=anchor_predictions,
            selection_threshold=threshold,
        )
        records = _prediction_records(
            CONTROLLER_ID, predictions, grouped, evaluation_pool="augmented"
        )
        metrics = _mean_metrics(records)
        diagnostics = _choice_diagnostics(
            predictions=predictions,
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
            and float(diagnostics["state_override_fraction"]) + 1e-12
            >= minimum_state_override_fraction
        )
        rows.append(
            {
                "selection_threshold": threshold,
                "metrics": metrics,
                "quality_constraints_passed": quality_passed,
                "safety_constraints_passed": safety_passed,
                "constraints_passed": quality_passed and safety_passed,
                "choice_diagnostics": diagnostics,
            }
        )
    eligible = [row for row in rows if row["constraints_passed"]]
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                float(row["metrics"]["mean_normalized_repairability_regret"]),
                float(row["choice_diagnostics"]["unsafe_override_fraction"]),
                -float(row["choice_diagnostics"]["state_override_fraction"]),
                -float(row["selection_threshold"]),
            ),
        )
        feasible = True
    else:
        fallback = [row for row in rows if float(row["selection_threshold"]) > 1.0]
        if len(fallback) != 1:
            raise ValueError("AnchorChoice grid lacks a unique V2 fallback")
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
                "selection_threshold": float(row["selection_threshold"]),
                "constraints_passed": bool(row["constraints_passed"]),
                "quality_constraints_passed": bool(row["quality_constraints_passed"]),
                "safety_constraints_passed": bool(row["safety_constraints_passed"]),
                "mean_normalized_repairability_regret": float(
                    row["metrics"]["mean_normalized_repairability_regret"]
                ),
                "exact_best_rate": float(row["metrics"]["exact_best_rate"]),
                "top3_hit_rate": float(row["metrics"]["top3_hit_rate"]),
                "choice_diagnostics": dict(row["choice_diagnostics"]),
            }
            for row in rows
        ],
    }


def _nested_train_oof(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    action_values: Any,
    action_table: dict[str, Any],
    direction_values: Any,
    direction_table: dict[str, Any],
    direction_maps: Any,
    anchor_predictions: dict[str, str],
    direction_specs: tuple[tuple[str, str], ...],
    action_parameters: dict[str, Any],
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
        inner_scores: dict[str, list[dict[str, Any]]] = {}
        for inner in _balanced_folds(outer_train_maps, inner_count):
            fit_maps = set(map(str, inner["train_maps"]))
            held_maps = set(map(str, inner["validation_maps"]))
            estimator = _fit_action_maps(
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
            inner_scores.update(_candidate_scores(held_grouped, held_anchor, estimator))
        if set(inner_scores) != set(inner_grouped):
            raise ValueError("AnchorChoice inner OOF coverage differs")
        calibration = _calibrate_selection(
            grouped=inner_grouped,
            candidate_scores=inner_scores,
            anchor_predictions=inner_anchor,
            lookup=action_table["lookup"],
            grid=selection_grid,
            exact_tolerance=exact_tolerance,
            top3_tolerance=top3_tolerance,
            minimum_precision=minimum_precision,
            minimum_state_override_fraction=minimum_state_override_fraction,
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
        held_scores = _candidate_scores(held_grouped, held_anchor, action_estimator)
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
                "selection_threshold_calibration": calibration,
                "action_classifier_metrics": _uncertainty_metrics(
                    action_table["labels"][action_mask],
                    probabilities,
                    action_table["weights"][action_mask],
                ),
            }
        )
    if (
        set(choice_predictions) != set(grouped)
        or set(maprank_predictions) != set(grouped)
        or set(crossfit_scores) != set(grouped)
    ):
        raise ValueError("AnchorChoice outer OOF coverage differs")
    final_calibration = _calibrate_selection(
        grouped=grouped,
        candidate_scores=crossfit_scores,
        anchor_predictions=anchor_predictions,
        lookup=action_table["lookup"],
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
            lookup=action_table["lookup"],
        ),
        "action_classifier_metrics": _uncertainty_metrics(
            oof_labels, oof_probabilities, oof_weights
        ),
        "all_fold_calibrations_feasible": all(
            bool(row["selection_threshold_calibration"]["calibration_feasible"])
            for row in folds
        ),
        "folds": folds,
        "final_calibration": final_calibration,
    }


def run_anchorchoice_training(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    import numpy as np

    path = Path(config_path).resolve()
    project_root = path.parents[1]
    config = _read_json(path)
    validate_anchorchoice_training_config(config)
    output_path = Path(output).resolve()
    registered_output = (project_root / str(config["registered_output"])).resolve()
    if output_path != registered_output:
        raise ValueError("STRIDE-AnchorChoice training output differs from registration")
    design_path = _project_path(project_root, config["design"])
    if sha256_file(design_path) != str(config["design_sha256"]):
        raise ValueError("AnchorChoice design SHA differs")
    validate_anchorchoice_design(_read_json(design_path))

    action_labels_path = _project_path(project_root, config["action_labels"])
    action_report_path = _project_path(project_root, config["action_label_report"])
    if sha256_file(action_labels_path) != str(config["action_labels_sha256"]):
        raise ValueError("AnchorChoice action labels differ")
    if sha256_file(action_report_path) != str(config["action_label_report_sha256"]):
        raise ValueError("AnchorChoice action label report differs")
    action_report = _read_json(action_report_path)
    if (
        action_report.get("schema") != LABEL_REPORT_SCHEMA
        or str(dict(action_report.get("artifacts") or {}).get("action_safety_pairs_sha256"))
        != str(config["action_labels_sha256"])
        or bool(action_report.get("runtime_used_in_label"))
        or bool(action_report.get("future_trajectory_used_in_label"))
        or bool(action_report.get("high_load_data_read"))
    ):
        raise ValueError("AnchorChoice action label report is invalid")

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
            raise ValueError(f"AnchorChoice registered input differs: {registered_path}")
    if _read_json(maprank_summary_path).get("schema") != BUILD_SCHEMA:
        raise ValueError("AnchorChoice MapRank label summary schema differs")
    maprank_bundle = _project_path(project_root, config["maprank_bundle"])
    maprank_report_path = _project_path(project_root, config["maprank_training_report"])
    if sha256_file(maprank_bundle / "controller_manifest.json") != str(
        config["maprank_manifest_sha256"]
    ):
        raise ValueError("AnchorChoice MapRank manifest differs")
    if sha256_file(maprank_report_path) != str(config["maprank_training_report_sha256"]):
        raise ValueError("AnchorChoice MapRank report differs")
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
        raise ValueError("AnchorChoice V2 anchor manifest differs")
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
    action_train = _load_action_table(
        action_labels_path, candidate_index, split="train"
    )
    action_validation = _load_action_table(
        action_labels_path, candidate_index, split="validation"
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
    action_train_values = _action_matrix(candidate_values, action_train)
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
        action_table=action_train,
        direction_values=direction_values,
        direction_table=direction_table,
        direction_maps=direction_maps,
        anchor_predictions=train_anchor,
        direction_specs=direction_specs,
        action_parameters=dict(config["model_parameters"]),
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
    final_estimator = _fit_action_maps(
        maps={str(rows[0]["map_id"]) for rows in train_grouped.values()},
        action_values=action_train_values,
        table=action_train,
        parameters=dict(config["model_parameters"]),
    )
    final_threshold = float(nested["final_calibration"]["selection_threshold"])
    validation_scores = _candidate_scores(
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
    classifier = dict(nested["action_classifier_metrics"])
    choice = dict(nested["choice_diagnostics"])
    promotion_gates = {
        "all_nested_calibrations_feasible": bool(
            nested["all_fold_calibrations_feasible"]
        )
        and bool(nested["final_calibration"]["calibration_feasible"]),
        "safety_roc_auc": float(classifier["weighted_roc_auc"]) + 1e-12
        >= float(gates["minimum_safety_roc_auc"]),
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
    model_path = output_path / "sklearn__direct_action_safety.pkl"
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
                lookup=action_validation["lookup"],
            ),
        },
        "promotion_gates": promotion_gates,
        "final_selection_threshold": final_threshold,
        "model_parameters": dict(config["model_parameters"]),
        "source_sha256": {
            "config": sha256_file(path),
            "design": sha256_file(design_path),
            "action_labels": sha256_file(action_labels_path),
            "action_label_report": sha256_file(action_report_path),
            "maprank_manifest": sha256_file(maprank_bundle / "controller_manifest.json"),
            "maprank_training_report": sha256_file(maprank_report_path),
            "v2_manifest": sha256_file(v2_bundle / "controller_manifest.json"),
        },
        "artifacts": {
            "offline_predictions": prediction_path.name,
            "offline_predictions_sha256": sha256_file(prediction_path),
            "sklearn_direct_action_safety_model": model_path.name,
            "sklearn_direct_action_safety_model_sha256": sha256_file(model_path),
        },
    }
    _write_json(output_path / "anchorchoice_training_report.json", report)
    return report


__all__ = [
    "_anchorchoice_predictions",
    "_candidate_scores",
    "run_anchorchoice_training",
]
