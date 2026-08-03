from __future__ import annotations

import shutil
import statistics
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.compact_controller_model import load_controller_bundle
from experiments.context_audit import PairwiseModel
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import _read_json, _write_json, _write_jsonl
from experiments.stride_augcontrol import (
    _input_specifications,
    _load_candidates,
    _load_pair_table,
    _mean_metrics,
    _model_predictions,
    _prediction_records,
    _selection_change_diagnostics,
    _validate_label_audit_provenance,
)
from experiments.stride_repairability import (
    BUILD_SCHEMA,
    CONFLICT_ONLY_LABEL_SCHEMA,
)
from experiments.stride_stage3 import _project_path
from experiments.stride_stage4 import (
    _candidate_matrix,
    _fit_registered_model,
    _frozen_predictions,
    _pair_matrix,
)
from experiments.stride_stage4r import _export_diagnostic_controller
from lns2_selector.controllers import load_selector
from lns2_selector.runtime.contracts import SelectionRequest
from lns2_selector.runtime.online_selection import pairwise_win_probability
from lns2_selector.training.tree_utils import balanced_map_folds


CONFIG_SCHEMA = "lns2.stride.guardrank_training_config.v1"
REPORT_SCHEMA = "lns2.stride.guardrank_training.v1"
PREDICTION_SCHEMA = "lns2.stride.guardrank_prediction.v1"
CONTROLLER_ID = "stride-guardrank-v1"
STRATEGY_SCHEMA = "lns2.stride.guardrank_strategy.v1"


def validate_guardrank_training_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "post_augcontrol_diagnostic_train_only_design"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("challenger_label_schema")
        != CONFLICT_ONLY_LABEL_SCHEMA
        or config.get("feature_schema") != "lns2.realized_features.v2"
        or int(config.get("base_feature_dimension", -1)) != 124
    ):
        raise ValueError("STRIDE guardrank training identity changed")
    if (
        config.get("model_class")
        != "sklearn.ensemble.HistGradientBoostingClassifier"
        or dict(config.get("model_parameters") or {})
        != {
            "early_stopping": False,
            "l2_regularization": 0.1,
            "learning_rate": 0.05,
            "max_iter": 100,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "random_state": 20260714,
        }
        or int(config.get("outer_map_folds", 0)) != 4
        or int(config.get("inner_map_folds", 0)) != 3
    ):
        raise ValueError("STRIDE guardrank model protocol changed")
    grid = list(map(float, config.get("threshold_grid") or ()))
    if (
        grid != [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.01]
        or list(map(str, config.get("threshold_kinds") or ()))
        != ["base", "boundary_only"]
        or config.get("calibration_objective")
        != "minimize_normalized_regret_subject_to_v2_exact_and_top3_noninferiority"
    ):
        raise ValueError("STRIDE guardrank calibration protocol changed")
    if dict(config.get("offline_gates") or {}) != {
        "minimum_relative_normalized_regret_improvement_over_frozen_v2": 0.05,
        "absolute_normalized_regret_improvement_alternative": 0.02,
        "top3_hit_rate_noninferiority_tolerance": 0.01,
        "exact_best_rate_noninferiority_tolerance": 0.01,
        "maximum_topology_group_normalized_regret_degradation": 0.03,
        "minimum_pairwise_accuracy_gain_over_weighted_majority": 0.03,
    }:
        raise ValueError("STRIDE guardrank offline gates changed")
    if (
        config.get("training_split") != "train"
        or config.get("legacy_validation_split") != "validation"
        or set(map(str, config.get("forbidden_calibration_splits") or ()))
        != {"validation", "test", "formal_ood"}
        or config.get("legacy_validation_is_descriptive_only") is not True
        or config.get("fresh_development_validation_required") is not True
        or bool(config.get("formal_ood_data_allowed"))
        or bool(config.get("test_data_allowed"))
        or bool(config.get("runtime_used_in_label"))
        or bool(config.get("future_trajectory_used_in_label"))
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("STRIDE guardrank evidence boundary changed")
    for field in ("data_design_sha256", "frozen_v2_manifest_sha256"):
        if len(str(config.get(field, ""))) != 64:
            raise ValueError(f"STRIDE guardrank registration hash is invalid: {field}")


def _map_family(map_id: str) -> str:
    lowered = map_id.lower()
    for prefix in ("den", "brc", "lak", "hrt", "ost", "ht_"):
        if lowered.startswith(prefix):
            return prefix.rstrip("_")
    return lowered.split("_")[0]


def _balanced_folds(map_ids: set[str], count: int) -> list[dict[str, Any]]:
    rows = [
        {"map_id": map_id, "layout_mode": _map_family(map_id)}
        for map_id in sorted(map_ids)
    ]
    return balanced_map_folds(rows, count=count)


def _online_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": row["candidate_id"],
            "candidate_key": row["candidate_key"],
            "candidate_kind": row["candidate_kind"],
            "features": {"realized_dynamic": row["features"]},
        }
        for row in rows
    ]


def _challenger_evidence(
    grouped: dict[str, list[dict[str, Any]]],
    estimator: Any,
    anchor_predictions: dict[str, str],
    input_specs: tuple[tuple[str, str], ...],
) -> dict[str, dict[str, Any]]:
    model = PairwiseModel(
        profile="realized_dynamic",
        feature_names=list(PROFILE_FEATURE_NAMES["realized_dynamic"]),
        estimator=estimator,
    )
    challenger = _model_predictions(grouped, estimator, input_specs)
    result = {}
    for state_id, rows in sorted(grouped.items()):
        by_id = {str(row["candidate_id"]): index for index, row in enumerate(rows)}
        challenger_id = challenger[state_id]
        anchor_id = anchor_predictions[state_id]
        challenger_index = by_id[challenger_id]
        anchor_index = by_id[anchor_id]
        probability = (
            1.0
            if challenger_index == anchor_index
            else pairwise_win_probability(
                _online_rows(rows), model, challenger_index, anchor_index
            )
        )
        result[state_id] = {
            "anchor_candidate_id": anchor_id,
            "challenger_candidate_id": challenger_id,
            "challenger_kind": str(rows[challenger_index]["candidate_kind"]),
            "challenger_evidence": probability,
        }
    return result


def _guard_predictions(
    evidence: dict[str, dict[str, Any]], thresholds: dict[str, float]
) -> dict[str, str]:
    result = {}
    for state_id, row in evidence.items():
        same = row["anchor_candidate_id"] == row["challenger_candidate_id"]
        override = float(row["challenger_evidence"]) + 1e-12 >= float(
            thresholds[str(row["challenger_kind"])]
        )
        result[state_id] = str(
            row["challenger_candidate_id"]
            if same or override
            else row["anchor_candidate_id"]
        )
    return result


def _calibrate_thresholds(
    *,
    evidence: dict[str, dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    anchor_predictions: dict[str, str],
    grid: list[float],
    exact_tolerance: float,
    top3_tolerance: float,
) -> dict[str, Any]:
    anchor_records = _prediction_records(
        "v2-full/anchor",
        anchor_predictions,
        grouped,
        evaluation_pool="augmented",
    )
    anchor_metrics = _mean_metrics(anchor_records)
    rows = []
    for base in grid:
        for boundary in grid:
            thresholds = {"base": base, "boundary_only": boundary}
            predictions = _guard_predictions(evidence, thresholds)
            records = _prediction_records(
                CONTROLLER_ID,
                predictions,
                grouped,
                evaluation_pool="augmented",
            )
            metrics = _mean_metrics(records)
            constraints_passed = (
                float(metrics["exact_best_rate"]) + exact_tolerance + 1e-12
                >= float(anchor_metrics["exact_best_rate"])
                and float(metrics["top3_hit_rate"]) + top3_tolerance + 1e-12
                >= float(anchor_metrics["top3_hit_rate"])
            )
            changes = _selection_change_diagnostics(anchor_records, records)
            rows.append(
                {
                    "thresholds": thresholds,
                    "metrics": metrics,
                    "constraints_passed": constraints_passed,
                    "selection_changes": changes,
                }
            )
    eligible = [row for row in rows if row["constraints_passed"]]
    if not eligible:
        raise ValueError("guardrank grid lacks its registered V2 fallback")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["metrics"]["mean_normalized_repairability_regret"]),
            -float(row["metrics"]["exact_best_rate"]),
            -float(row["metrics"]["top3_hit_rate"]),
            int(row["selection_changes"]["selection_change_count"]),
            -float(row["thresholds"]["boundary_only"]),
            -float(row["thresholds"]["base"]),
        ),
    )
    return {
        "thresholds": dict(selected["thresholds"]),
        "metrics": dict(selected["metrics"]),
        "anchor_metrics": anchor_metrics,
        "selection_changes": dict(selected["selection_changes"]),
        "grid_size": len(rows),
        "eligible_grid_count": len(eligible),
    }


def _pairwise_fold_metrics(
    estimator: Any,
    pair_values: Any,
    pair_table: dict[str, Any],
    mask: Any,
) -> tuple[float, float, float]:
    import numpy as np

    probabilities = estimator.predict_proba(pair_values[mask])[:, 1]
    labels = pair_table["labels"][mask]
    weights = pair_table["weights"][mask]
    correct = (probabilities >= 0.5) == (labels == 1)
    total = float(np.sum(weights))
    positive = float(np.sum(weights[labels == 1]))
    return float(np.sum(weights[correct])), total, max(positive, total - positive)


def _fit_maps(
    *,
    maps: set[str],
    pair_maps: Any,
    pair_values: Any,
    pair_table: dict[str, Any],
    parameters: dict[str, Any],
) -> Any:
    import numpy as np

    mask = np.isin(pair_maps, sorted(maps))
    if not bool(np.any(mask)):
        raise ValueError("guardrank map split has no training pairs")
    return _fit_registered_model(
        pair_values[mask],
        pair_table["labels"][mask],
        pair_table["weights"][mask],
        parameters,
    )


def _topology_comparisons(
    challenger_records: list[dict[str, Any]],
    anchor_records: list[dict[str, Any]],
    tolerance: float,
) -> list[dict[str, Any]]:
    result = []
    for group in sorted({str(row["topology_group"]) for row in anchor_records}):
        challenger = [
            row for row in challenger_records if str(row["topology_group"]) == group
        ]
        anchor = [
            row for row in anchor_records if str(row["topology_group"]) == group
        ]
        challenger_regret = statistics.fmean(
            float(row["normalized_repairability_regret"]) for row in challenger
        )
        anchor_regret = statistics.fmean(
            float(row["normalized_repairability_regret"]) for row in anchor
        )
        result.append(
            {
                "topology_group": group,
                "state_count": len(anchor),
                "guardrank_normalized_regret": challenger_regret,
                "v2_normalized_regret": anchor_regret,
                "degradation": challenger_regret - anchor_regret,
                "passed": challenger_regret <= anchor_regret + tolerance + 1e-12,
            }
        )
    return result


def _nested_train_oof(
    *,
    grouped: dict[str, list[dict[str, Any]]],
    pair_values: Any,
    pair_table: dict[str, Any],
    pair_maps: Any,
    anchor_predictions: dict[str, str],
    input_specs: tuple[tuple[str, str], ...],
    parameters: dict[str, Any],
    outer_count: int,
    inner_count: int,
    grid: list[float],
    exact_tolerance: float,
    top3_tolerance: float,
) -> dict[str, Any]:
    import numpy as np

    all_maps = {str(rows[0]["map_id"]) for rows in grouped.values()}
    nested_predictions: dict[str, str] = {}
    crossfit_evidence: dict[str, dict[str, Any]] = {}
    folds = []
    correct_weight = total_weight = majority_weight = 0.0
    for outer in _balanced_folds(all_maps, outer_count):
        outer_train_maps = set(map(str, outer["train_maps"]))
        outer_validation_maps = set(map(str, outer["validation_maps"]))
        inner_evidence: dict[str, dict[str, Any]] = {}
        inner_grouped = {
            state_id: rows
            for state_id, rows in grouped.items()
            if str(rows[0]["map_id"]) in outer_train_maps
        }
        inner_anchor = {
            state_id: anchor_predictions[state_id] for state_id in inner_grouped
        }
        for inner in _balanced_folds(outer_train_maps, inner_count):
            inner_train_maps = set(map(str, inner["train_maps"]))
            inner_validation_maps = set(map(str, inner["validation_maps"]))
            estimator = _fit_maps(
                maps=inner_train_maps,
                pair_maps=pair_maps,
                pair_values=pair_values,
                pair_table=pair_table,
                parameters=parameters,
            )
            held_grouped = {
                state_id: rows
                for state_id, rows in inner_grouped.items()
                if str(rows[0]["map_id"]) in inner_validation_maps
            }
            held_anchor = {
                state_id: inner_anchor[state_id] for state_id in held_grouped
            }
            inner_evidence.update(
                _challenger_evidence(
                    held_grouped, estimator, held_anchor, input_specs
                )
            )
        if set(inner_evidence) != set(inner_grouped):
            raise ValueError("guardrank inner OOF evidence coverage differs")
        calibration = _calibrate_thresholds(
            evidence=inner_evidence,
            grouped=inner_grouped,
            anchor_predictions=inner_anchor,
            grid=grid,
            exact_tolerance=exact_tolerance,
            top3_tolerance=top3_tolerance,
        )
        outer_estimator = _fit_maps(
            maps=outer_train_maps,
            pair_maps=pair_maps,
            pair_values=pair_values,
            pair_table=pair_table,
            parameters=parameters,
        )
        outer_grouped = {
            state_id: rows
            for state_id, rows in grouped.items()
            if str(rows[0]["map_id"]) in outer_validation_maps
        }
        outer_anchor = {
            state_id: anchor_predictions[state_id] for state_id in outer_grouped
        }
        outer_evidence = _challenger_evidence(
            outer_grouped, outer_estimator, outer_anchor, input_specs
        )
        crossfit_evidence.update(outer_evidence)
        nested_predictions.update(
            _guard_predictions(outer_evidence, calibration["thresholds"])
        )
        mask = np.isin(pair_maps, sorted(outer_validation_maps))
        correct, total, majority = _pairwise_fold_metrics(
            outer_estimator, pair_values, pair_table, mask
        )
        correct_weight += correct
        total_weight += total
        majority_weight += majority
        folds.append(
            {
                "outer_fold": int(outer["fold"]),
                "training_maps": sorted(outer_train_maps),
                "validation_maps": sorted(outer_validation_maps),
                "validation_state_count": len(outer_grouped),
                "threshold_calibration": calibration,
                "pairwise_accuracy": correct / total,
                "weighted_majority_accuracy": majority / total,
            }
        )
    if set(nested_predictions) != set(grouped) or set(crossfit_evidence) != set(grouped):
        raise ValueError("guardrank nested OOF state coverage differs")
    records = _prediction_records(
        CONTROLLER_ID,
        nested_predictions,
        grouped,
        evaluation_pool="augmented",
    )
    anchor_records = _prediction_records(
        "v2-full/anchor",
        anchor_predictions,
        grouped,
        evaluation_pool="augmented",
    )
    final_calibration = _calibrate_thresholds(
        evidence=crossfit_evidence,
        grouped=grouped,
        anchor_predictions=anchor_predictions,
        grid=grid,
        exact_tolerance=exact_tolerance,
        top3_tolerance=top3_tolerance,
    )
    return {
        "predictions": nested_predictions,
        "records": records,
        "anchor_records": anchor_records,
        "metrics": _mean_metrics(records),
        "anchor_metrics": _mean_metrics(anchor_records),
        "selection_changes": _selection_change_diagnostics(anchor_records, records),
        "pairwise_accuracy": correct_weight / total_weight,
        "weighted_majority_accuracy": majority_weight / total_weight,
        "folds": folds,
        "final_calibration": final_calibration,
    }


def _export_guardrank_bundle(
    *,
    root: Path,
    estimator: Any,
    thresholds: dict[str, float],
    candidates: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    frozen_bundle: Path,
    parameters: dict[str, Any],
    training_pair_count: int,
    source_hashes: dict[str, str],
    nested_passed: bool,
) -> dict[str, Any]:
    source_manifest = _read_json(frozen_bundle / "controller_manifest.json")
    exported = _export_diagnostic_controller(
        root=root,
        controller_id=CONTROLLER_ID,
        estimator=estimator,
        feature_names=tuple(PROFILE_FEATURE_NAMES["realized_dynamic"]),
        candidates=candidates,
        grouped=grouped,
        source_bundle=frozen_bundle,
        source_manifest=source_manifest,
        parameters=parameters,
        training_pair_count=training_pair_count,
        source_hashes=source_hashes,
    )
    manifest_path = root / "controller_manifest.json"
    manifest = _read_json(manifest_path)
    anchor_rows = {}
    for profile, value in dict(source_manifest["main_rankers"]).items():
        source_row = dict(value)
        source_path = frozen_bundle / str(source_row["file"])
        destination = root / f"anchor__{profile}.json"
        shutil.copyfile(source_path, destination)
        if sha256_file(destination) != str(source_row["sha256"]):
            raise ValueError("guardrank copied V2 anchor ranker differs")
        source_row["file"] = destination.relative_to(root).as_posix()
        source_row["sha256"] = sha256_file(destination)
        anchor_rows[str(profile)] = source_row
    manifest["anchor_rankers"] = anchor_rows
    manifest["anchor_ranges"] = dict(source_manifest["main_ranges"])
    manifest["selection_strategy"] = {
        "schema": STRATEGY_SCHEMA,
        "strategy_id": "v2_anchor_pairwise_guard",
        "applied_profile": "realized_dynamic",
        "anchor_controller_id": "v2-full",
        "challenger_label_schema": CONFLICT_ONLY_LABEL_SCHEMA,
        "challenger_thresholds": thresholds,
        "calibration_split": "train_nested_map_oof",
        "legacy_validation_used_for_calibration": False,
    }
    report_path = root / str(manifest["promotion_report"]["file"])
    evidence = _read_json(report_path)
    evidence.update(
        {
            "nested_train_oof_passed": nested_passed,
            "fresh_development_validation_required": True,
            "shadow_eligible": False,
            "selection_strategy": manifest["selection_strategy"],
        }
    )
    _write_json(report_path, evidence)
    manifest["promotion_report"]["sha256"] = sha256_file(report_path)
    _write_json(manifest_path, manifest)
    loaded = load_controller_bundle(root)
    if set(loaded.anchor_models) != set(PROFILE_FEATURE_NAMES):
        raise ValueError("guardrank exported anchor rankers did not reload")
    return {
        **exported,
        "controller_manifest_sha256": sha256_file(manifest_path),
        "promotion_report_sha256": sha256_file(report_path),
        "anchor_ranker_sha256": {
            profile: row["sha256"] for profile, row in anchor_rows.items()
        },
        "challenger_thresholds": thresholds,
    }


def _runtime_predictions(
    bundle: Path, grouped: dict[str, list[dict[str, Any]]]
) -> dict[str, str]:
    selector = load_selector(CONTROLLER_ID, bundle)
    result = {}
    for state_id, rows in sorted(grouped.items()):
        candidates = [
            {
                "candidate_id": row["candidate_id"],
                "selection_families": row["selection_families"],
            }
            for row in rows
        ]
        decision = selector.select(
            SelectionRequest(
                candidates=candidates,
                candidate_rows=_online_rows(rows),
                before_fingerprint=state_id,
                profile="realized_dynamic",
            )
        )
        result[state_id] = str(rows[int(decision.candidate_index)]["candidate_id"])
    return result


def run_guardrank_training(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    import numpy as np

    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_guardrank_training_config(config)
    data_design_path = _project_path(project_root, str(config["data_design"]))
    if sha256_file(data_design_path) != str(config["data_design_sha256"]):
        raise ValueError("STRIDE guardrank data design differs")
    data_design = _read_json(data_design_path)
    labels = _project_path(project_root, str(config["labels"]))
    selection_path = _project_path(project_root, str(config["selection"]))
    summary_path = labels / "label_build_summary.json"
    summary = _read_json(summary_path)
    if (
        summary.get("schema") != BUILD_SCHEMA
        or bool(summary.get("runtime_used_in_label"))
        or bool(summary.get("future_trajectory_used_in_label"))
    ):
        raise ValueError("STRIDE guardrank label summary is invalid")
    audit_provenance = _validate_label_audit_provenance(summary)
    aggregate_path = labels / "candidate_aggregates.jsonl"
    pair_path = labels / "conflict_only_dominance_pairs.jsonl"
    candidates, grouped = _load_candidates(
        aggregate_path,
        selection_path,
        topology_threshold=float(data_design["topology_boundary_threshold"]),
    )
    candidate_index = {
        (str(row["state_id"]), str(row["candidate_id"])): int(row["candidate_index"])
        for row in candidates
    }
    train_pairs = _load_pair_table(
        pair_path,
        candidate_index,
        split="train",
        expected_schema=CONFLICT_ONLY_LABEL_SCHEMA,
    )
    train_grouped = {
        state_id: rows
        for state_id, rows in grouped.items()
        if str(rows[0]["split"]) == "train"
    }
    legacy_validation_grouped = {
        state_id: rows
        for state_id, rows in grouped.items()
        if str(rows[0]["split"]) == "validation"
    }
    if set(train_grouped) & set(legacy_validation_grouped):
        raise ValueError("guardrank train and legacy validation states overlap")
    frozen_bundle = _project_path(project_root, str(config["frozen_v2_bundle"]))
    if sha256_file(frozen_bundle / "controller_manifest.json") != str(
        config["frozen_v2_manifest_sha256"]
    ):
        raise ValueError("STRIDE guardrank frozen V2 bundle differs")
    train_anchor = _frozen_predictions(train_grouped, frozen_bundle)
    legacy_anchor = _frozen_predictions(legacy_validation_grouped, frozen_bundle)
    names, input_specs = _input_specifications()
    candidate_values = _candidate_matrix(candidates)
    pair_values = _pair_matrix(candidate_values, train_pairs, input_specs)
    pair_maps = np.asarray(
        [str(candidates[int(index)]["map_id"]) for index in train_pairs["left"]],
        dtype=object,
    )
    parameters = dict(config["model_parameters"])
    gates = dict(config["offline_gates"])
    nested = _nested_train_oof(
        grouped=train_grouped,
        pair_values=pair_values,
        pair_table=train_pairs,
        pair_maps=pair_maps,
        anchor_predictions=train_anchor,
        input_specs=input_specs,
        parameters=parameters,
        outer_count=int(config["outer_map_folds"]),
        inner_count=int(config["inner_map_folds"]),
        grid=list(map(float, config["threshold_grid"])),
        exact_tolerance=float(gates["exact_best_rate_noninferiority_tolerance"]),
        top3_tolerance=float(gates["top3_hit_rate_noninferiority_tolerance"]),
    )
    nested_metrics = dict(nested["metrics"])
    anchor_metrics = dict(nested["anchor_metrics"])
    regret_improvement = float(
        anchor_metrics["mean_normalized_repairability_regret"]
    ) - float(nested_metrics["mean_normalized_repairability_regret"])
    relative_improvement = regret_improvement / max(
        1e-12, float(anchor_metrics["mean_normalized_repairability_regret"])
    )
    topology = _topology_comparisons(
        nested["records"],
        nested["anchor_records"],
        float(gates["maximum_topology_group_normalized_regret_degradation"]),
    )
    promotion_gates = {
        "nested_map_oof_pairwise_accuracy": float(nested["pairwise_accuracy"])
        >= float(nested["weighted_majority_accuracy"])
        + float(gates["minimum_pairwise_accuracy_gain_over_weighted_majority"])
        - 1e-12,
        "nested_map_oof_regret_improves_v2": (
            relative_improvement + 1e-12
            >= float(
                gates[
                    "minimum_relative_normalized_regret_improvement_over_frozen_v2"
                ]
            )
            or regret_improvement + 1e-12
            >= float(gates["absolute_normalized_regret_improvement_alternative"])
        ),
        "nested_map_oof_exact_best_noninferior": float(
            nested_metrics["exact_best_rate"]
        )
        + float(gates["exact_best_rate_noninferiority_tolerance"])
        + 1e-12
        >= float(anchor_metrics["exact_best_rate"]),
        "nested_map_oof_top3_noninferior": float(nested_metrics["top3_hit_rate"])
        + float(gates["top3_hit_rate_noninferiority_tolerance"])
        + 1e-12
        >= float(anchor_metrics["top3_hit_rate"]),
        "nested_map_oof_topology_groups_noninferior": all(
            row["passed"] for row in topology
        ),
    }
    nested_passed = all(promotion_gates.values())
    final_estimator = _fit_registered_model(
        pair_values,
        train_pairs["labels"],
        train_pairs["weights"],
        parameters,
    )
    final_thresholds = {
        name: float(value)
        for name, value in dict(
            nested["final_calibration"]["thresholds"]
        ).items()
    }
    legacy_evidence = _challenger_evidence(
        legacy_validation_grouped,
        final_estimator,
        legacy_anchor,
        input_specs,
    )
    legacy_predictions = _guard_predictions(legacy_evidence, final_thresholds)
    legacy_records = _prediction_records(
        CONTROLLER_ID,
        legacy_predictions,
        legacy_validation_grouped,
        evaluation_pool="augmented",
    )
    legacy_anchor_records = _prediction_records(
        "v2-full/anchor",
        legacy_anchor,
        legacy_validation_grouped,
        evaluation_pool="augmented",
    )
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "offline_predictions.jsonl"
    _write_jsonl(
        prediction_path,
        [
            {**row, "schema": PREDICTION_SCHEMA, "evidence_role": "nested_train_oof"}
            for row in nested["records"]
        ]
        + [
            {**row, "schema": PREDICTION_SCHEMA, "evidence_role": "legacy_validation_descriptive"}
            for row in legacy_records
        ],
    )
    source_hashes = {
        "config": sha256_file(config_path),
        "label_summary": sha256_file(summary_path),
        "candidate_aggregates": sha256_file(aggregate_path),
        "conflict_only_pairs": sha256_file(pair_path),
        "selection": sha256_file(selection_path),
        "frozen_v2_manifest": sha256_file(
            frozen_bundle / "controller_manifest.json"
        ),
    }
    export = _export_guardrank_bundle(
        root=output / CONTROLLER_ID,
        estimator=final_estimator,
        thresholds=final_thresholds,
        candidates=[row for row in candidates if str(row["split"]) == "train"],
        grouped=train_grouped,
        frozen_bundle=frozen_bundle,
        parameters=parameters,
        training_pair_count=len(train_pairs["labels"]),
        source_hashes=source_hashes,
        nested_passed=nested_passed,
    )
    reference_train = _guard_predictions(
        _challenger_evidence(
            train_grouped, final_estimator, train_anchor, input_specs
        ),
        final_thresholds,
    )
    runtime_train = _runtime_predictions(output / CONTROLLER_ID, train_grouped)
    runtime_legacy = _runtime_predictions(
        output / CONTROLLER_ID, legacy_validation_grouped
    )
    runtime_equivalence = {
        "train_state_count": len(train_grouped),
        "train_mismatch_count": sum(
            reference_train[state_id] != runtime_train[state_id]
            for state_id in train_grouped
        ),
        "legacy_validation_state_count": len(legacy_validation_grouped),
        "legacy_validation_mismatch_count": sum(
            legacy_predictions[state_id] != runtime_legacy[state_id]
            for state_id in legacy_validation_grouped
        ),
    }
    runtime_equivalence["passed"] = (
        runtime_equivalence["train_mismatch_count"] == 0
        and runtime_equivalence["legacy_validation_mismatch_count"] == 0
    )
    integrity_gates = {
        "label_collection_audited": bool(audit_provenance),
        "feature_dimension": len(names) == 124,
        "pairwise_input_dimension": len(input_specs) == 147,
        "calibration_train_maps_only": True,
        "legacy_validation_not_used_for_calibration": True,
        "formal_ood_not_read": True,
        "test_data_not_read": True,
        "runtime_not_used_in_label": True,
        "future_trajectory_not_used_in_label": True,
        "runtime_bundle_equivalence": bool(runtime_equivalence["passed"]),
    }
    fresh_eligible = all(integrity_gates.values()) and nested_passed
    report = {
        "schema": REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "scientific_status": (
            "fresh_development_eligible"
            if fresh_eligible
            else "nested_train_oof_gate_failed"
        ),
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "formal_ood_data_read": False,
        "test_data_read": False,
        "legacy_validation_used_for_calibration": False,
        "legacy_validation_is_descriptive_only": True,
        "fresh_development_validation_required": True,
        "shadow_eligible": False,
        "fresh_development_eligible": fresh_eligible,
        "train_state_count": len(train_grouped),
        "train_map_count": len({str(rows[0]["map_id"]) for rows in train_grouped.values()}),
        "legacy_validation_state_count": len(legacy_validation_grouped),
        "nested_train_oof": {
            "metrics": nested_metrics,
            "v2_anchor_metrics": anchor_metrics,
            "pairwise_accuracy": nested["pairwise_accuracy"],
            "weighted_majority_accuracy": nested["weighted_majority_accuracy"],
            "normalized_regret_improvement_vs_v2": regret_improvement,
            "relative_normalized_regret_improvement_vs_v2": relative_improvement,
            "selection_changes": nested["selection_changes"],
            "topology_group_comparisons": topology,
            "folds": nested["folds"],
            "final_calibration": nested["final_calibration"],
        },
        "legacy_validation_descriptive": {
            "metrics": _mean_metrics(legacy_records),
            "v2_anchor_metrics": _mean_metrics(legacy_anchor_records),
            "selection_changes": _selection_change_diagnostics(
                legacy_anchor_records, legacy_records
            ),
        },
        "promotion_gates": promotion_gates,
        "integrity_gates": integrity_gates,
        "runtime_equivalence": runtime_equivalence,
        "label_audit_provenance": audit_provenance,
        "final_thresholds": final_thresholds,
        "model_parameters": parameters,
        "export": export,
        "source_sha256": source_hashes,
        "artifacts": {
            "offline_predictions": prediction_path.name,
            "offline_predictions_sha256": sha256_file(prediction_path),
        },
    }
    _write_json(output / "guardrank_training_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "CONTROLLER_ID",
    "REPORT_SCHEMA",
    "_calibrate_thresholds",
    "_guard_predictions",
    "run_guardrank_training",
    "validate_guardrank_training_config",
]
