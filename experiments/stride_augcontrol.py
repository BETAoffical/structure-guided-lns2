from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import (
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)
from experiments.stride_repairability import (
    BUILD_SCHEMA,
    CANDIDATE_SCHEMA,
    CONFLICT_ONLY_LABEL_SCHEMA,
    CONTROLLER_ID,
    LABEL_SCHEMA,
)
from experiments.stride_repairability_selection import (
    validate_repairability_data_design,
)
from experiments.stride_stage4 import (
    _candidate_matrix,
    _fit_registered_model,
    _frozen_predictions,
    _pair_matrix,
)
from experiments.stride_stage4r import _export_diagnostic_controller
from experiments.stride_stage3 import _project_path
from experiments.v2_factorial_audit import _select_model


CONFIG_SCHEMA = "lns2.stride.augcontrol_training_config.v1"
REPORT_SCHEMA = "lns2.stride.augcontrol_training.v1"
PREDICTION_SCHEMA = "lns2.stride.augcontrol_prediction.v1"
CONFLICT_CONTROLLER_ID = "stride-augcontrol-conflict-v1"

def _input_specifications() -> tuple[tuple[str, ...], tuple[tuple[str, str], ...]]:
    names = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
    specifications = tuple(("delta", name) for name in names) + tuple(
        ("shared", name)
        for name in names
        if name.startswith(("state.", "context."))
    )
    return names, specifications


def validate_augcontrol_training_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("scientific_status")
        != "preregistered_before_repairability_label_outcomes"
        or config.get("controller_id") != CONTROLLER_ID
        or config.get("conflict_ablation_id") != CONFLICT_CONTROLLER_ID
        or config.get("feature_schema") != "lns2.realized_features.v2"
        or config.get("label_schema") != LABEL_SCHEMA
        or int(config.get("base_feature_dimension", -1)) != 124
        or bool(config.get("default_replacement_allowed"))
        or bool(config.get("formal_speed_claim"))
    ):
        raise ValueError("STRIDE augcontrol training identity changed")
    pairwise = dict(config.get("pairwise_input") or {})
    _, specifications = _input_specifications()
    if pairwise != {
        "candidate_features": "delta",
        "shared_state_context": "mean",
        "expected_input_dimension": len(specifications),
    }:
        raise ValueError("STRIDE augcontrol pairwise construction changed")
    if dict(config.get("model_parameters") or {}) != {
        "early_stopping": False,
        "l2_regularization": 0.1,
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "random_state": 20260714,
    }:
        raise ValueError("STRIDE augcontrol model capacity changed")
    if (
        config.get("model_class")
        != "sklearn.ensemble.HistGradientBoostingClassifier"
        or config.get("training_split") != "train"
        or config.get("validation_split") != "validation"
        or config.get("split_unit") != "map"
        or config.get("sample_weighting") != "equal_total_weight_per_state"
        or bool(config.get("hyperparameter_tuning"))
        or bool(config.get("formal_ood_data_allowed"))
        or bool(config.get("test_data_allowed"))
    ):
        raise ValueError("STRIDE augcontrol training protocol changed")
    gates = dict(config.get("offline_gates") or {})
    if gates != {
        "minimum_validation_maps": 6,
        "minimum_pairwise_accuracy_gain_over_weighted_majority": 0.03,
        "minimum_relative_normalized_regret_improvement_over_frozen_v2": 0.05,
        "absolute_normalized_regret_improvement_alternative": 0.02,
        "top3_hit_rate_noninferiority_tolerance": 0.01,
        "exact_best_rate_noninferiority_tolerance": 0.01,
        "maximum_topology_group_normalized_regret_degradation": 0.03,
    }:
        raise ValueError("STRIDE augcontrol promotion gates changed")
    if set(map(str, config.get("forbidden_training_inputs") or ())) != {
        "repair_runtime",
        "time_to_feasible",
        "future_repair_rounds",
        "cost_to_go",
        "receding_q",
        "controller_outcome",
    }:
        raise ValueError("STRIDE augcontrol forbidden inputs changed")
    if list(map(str, config.get("primary_offline_metrics") or ())) != [
        "weighted_pairwise_accuracy",
        "exact_best_rate",
        "top3_hit_rate",
        "mean_normalized_repairability_regret",
    ] or list(map(str, config.get("comparators") or ())) != [
        "frozen-v2-full-on-identical-augmented-pool",
        "stride-augcontrol-v1/base-pool-only-ablation",
        "stride-augcontrol-v1/conflict-only-label-ablation",
    ]:
        raise ValueError("STRIDE augcontrol registered evaluation changed")
    for field in (
        "label_design_sha256",
        "data_design_sha256",
        "source_cohort_sha256",
        "frozen_v2_manifest_sha256",
    ):
        if len(str(config.get(field, ""))) != 64:
            raise ValueError(f"STRIDE augcontrol registration hash is invalid: {field}")


def _load_candidates(
    aggregate_path: Path,
    selection_path: Path,
    *,
    topology_threshold: float,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    selection = {str(row["state_id"]): row for row in _read_jsonl(selection_path)}
    expected_features = set(PROFILE_FEATURE_NAMES["realized_dynamic"])
    candidates: list[dict[str, Any]] = []
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for source in _read_jsonl(aggregate_path):
        if source.get("schema") != CANDIDATE_SCHEMA:
            raise ValueError("unexpected STRIDE augcontrol candidate schema")
        state_id = str(source["state_id"])
        candidate_id = str(source["candidate_id"])
        key = (state_id, candidate_id)
        if key in seen:
            raise ValueError(f"duplicate STRIDE augcontrol candidate: {key}")
        seen.add(key)
        metadata = selection.get(state_id)
        if metadata is None:
            raise ValueError(f"augcontrol candidate is absent from selection: {state_id}")
        features = {
            str(name): float(value)
            for name, value in dict(source["features"]).items()
        }
        if set(features) != expected_features or any(
            not math.isfinite(value) for value in features.values()
        ):
            raise ValueError(f"augcontrol candidate feature schema differs: {key}")
        if str(source["split"]) != str(metadata["research_split"]):
            raise ValueError(f"augcontrol candidate research split differs: {key}")
        ratio = float(metadata["static_low_degree_cell_ratio"])
        row = {
            "candidate_index": len(candidates),
            "state_id": state_id,
            "candidate_id": candidate_id,
            "candidate_key": candidate_id,
            "features": features,
            "repairability_score": float(source["mean_repairability_score"]),
            "repairability_score_std": float(source["repairability_score_std"]),
            "mean_conflicts_after": float(source["mean_conflicts_after"]),
            "mean_conflict_reduction_ratio": float(
                source["mean_conflict_reduction_ratio"]
            ),
            "progress_rate": float(source["progress_rate"]),
            "feasible_rate": float(source["feasible_rate"]),
            "candidate_kind": str(source["candidate_kind"]),
            "selection_families": list(map(str, source["selection_families"])),
            "actual_size": int(source["actual_size"]),
            "map_id": str(source["map_id"]),
            "split": str(source["split"]),
            "source_policy": str(source["source_policy"]),
            "decision_stage": str(source["decision_stage"]),
            "agent_count": int(source["agent_count"]),
            "agent_band": str(metadata["agent_band"]),
            "topology_group": (
                "boundary_relevant" if ratio >= topology_threshold else "control"
            ),
        }
        candidates.append(row)
        grouped[state_id].append(row)
    if set(grouped) != set(selection):
        raise ValueError("augcontrol candidate and selected-state coverage differs")
    if any(len(rows) < 2 or len(rows) > 20 for rows in grouped.values()):
        raise ValueError("augcontrol candidate pool cardinality differs")
    return candidates, dict(grouped)


def _load_pair_table(
    pair_path: Path,
    candidate_index: dict[tuple[str, str], int],
    *,
    split: str,
    expected_schema: str,
) -> dict[str, Any]:
    import numpy as np

    left: list[int] = []
    right: list[int] = []
    labels: list[int] = []
    weights: list[float] = []
    states: Counter[str] = Counter()
    for row in _read_jsonl(pair_path):
        if row.get("schema") != expected_schema:
            raise ValueError(f"unexpected augcontrol pair schema in {pair_path}")
        if str(row["split"]) != split:
            continue
        state_id = str(row["state_id"])
        left.append(candidate_index[(state_id, str(row["left_candidate_id"]))])
        right.append(candidate_index[(state_id, str(row["right_candidate_id"]))])
        labels.append(int(row["label"]))
        weight = float(row["sample_weight"])
        weights.append(weight)
        states[state_id] += weight
    if not labels or set(labels) != {0, 1}:
        raise ValueError(f"augcontrol {split} pairs require both labels")
    if any(
        not math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-9)
        for value in states.values()
    ):
        raise ValueError(f"augcontrol {split} pair weights differ by state")
    return {
        "left": np.asarray(left, dtype=np.int32),
        "right": np.asarray(right, dtype=np.int32),
        "labels": np.asarray(labels, dtype=np.int8),
        "weights": np.asarray(weights, dtype=np.float64),
        "state_count": len(states),
    }


def _pairwise_metrics(estimator: Any, values: Any, table: dict[str, Any]) -> dict[str, float]:
    import numpy as np

    probabilities = estimator.predict_proba(values)[:, 1]
    labels = table["labels"]
    weights = table["weights"]
    correct = (probabilities >= 0.5) == (labels == 1)
    total = float(np.sum(weights))
    positive = float(np.sum(weights[labels == 1]))
    majority = max(positive, total - positive) / total
    return {
        "accuracy": float(np.sum(weights[correct])) / total,
        "weighted_majority_accuracy": majority,
        "validation_weight": total,
    }


def _model_predictions(
    grouped: dict[str, list[dict[str, Any]]],
    estimator: Any,
    input_specs: tuple[tuple[str, str], ...],
    *,
    base_only: bool = False,
) -> dict[str, str]:
    predictions = {}
    for state_id, rows in sorted(grouped.items()):
        eligible = [
            row for row in rows if not base_only or row["candidate_kind"] != "boundary_only"
        ]
        if not eligible:
            raise ValueError(f"augcontrol state has no eligible candidates: {state_id}")
        selected = _select_model(eligible, estimator, input_specs)
        predictions[state_id] = str(selected["candidate_id"])
    return predictions


def _prediction_records(
    model_id: str,
    predictions: dict[str, str],
    grouped: dict[str, list[dict[str, Any]]],
    *,
    evaluation_pool: str,
) -> list[dict[str, Any]]:
    records = []
    for state_id, rows in sorted(grouped.items()):
        eligible = (
            rows
            if evaluation_pool == "augmented"
            else [row for row in rows if row["candidate_kind"] != "boundary_only"]
        )
        by_id = {str(row["candidate_id"]): row for row in eligible}
        selected = by_id[predictions[state_id]]
        ranked = sorted(
            eligible,
            key=lambda row: (
                -float(row["repairability_score"]),
                str(row["candidate_id"]),
            ),
        )
        best = float(ranked[0]["repairability_score"])
        worst = float(ranked[-1]["repairability_score"])
        selected_score = float(selected["repairability_score"])
        regret = max(0.0, best - selected_score)
        score_range = best - worst
        records.append(
            {
                "schema": PREDICTION_SCHEMA,
                "model_id": model_id,
                "evaluation_pool": evaluation_pool,
                "state_id": state_id,
                "selected_candidate_id": str(selected["candidate_id"]),
                "selected_candidate_kind": str(selected["candidate_kind"]),
                "map_id": str(selected["map_id"]),
                "split": str(selected["split"]),
                "source_policy": str(selected["source_policy"]),
                "decision_stage": str(selected["decision_stage"]),
                "agent_band": str(selected["agent_band"]),
                "topology_group": str(selected["topology_group"]),
                "candidate_count": len(eligible),
                "exact_best": selected_score + 1e-12 >= best,
                "top3_hit": str(selected["candidate_id"])
                in {str(row["candidate_id"]) for row in ranked[:3]},
                "repairability_score": selected_score,
                "repairability_regret": regret,
                "normalized_repairability_regret": (
                    regret / score_range if score_range > 1e-12 else 0.0
                ),
                "mean_conflicts_after": float(selected["mean_conflicts_after"]),
                "mean_conflict_reduction_ratio": float(
                    selected["mean_conflict_reduction_ratio"]
                ),
                "progress_rate": float(selected["progress_rate"]),
                "feasible_rate": float(selected["feasible_rate"]),
            }
        )
    return records


def _mean_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize empty augcontrol predictions")
    mean = lambda name: statistics.fmean(float(row[name]) for row in records)
    result = {
        "state_count": len(records),
        "exact_best_rate": mean("exact_best"),
        "top3_hit_rate": mean("top3_hit"),
        "mean_repairability_score": mean("repairability_score"),
        "mean_repairability_regret": mean("repairability_regret"),
        "mean_normalized_repairability_regret": mean(
            "normalized_repairability_regret"
        ),
        "mean_conflicts_after": mean("mean_conflicts_after"),
        "mean_conflict_reduction_ratio": mean("mean_conflict_reduction_ratio"),
        "mean_progress_rate": mean("progress_rate"),
        "mean_feasible_rate": mean("feasible_rate"),
    }
    result["subgroups"] = [
        {
            "field": field,
            "value": value,
            **_mean_metrics_without_subgroups(
                [row for row in records if str(row[field]) == value]
            ),
        }
        for field in ("topology_group", "source_policy", "decision_stage", "agent_band")
        for value in sorted({str(row[field]) for row in records})
    ]
    return result


def _mean_metrics_without_subgroups(records: list[dict[str, Any]]) -> dict[str, Any]:
    mean = lambda name: statistics.fmean(float(row[name]) for row in records)
    return {
        "state_count": len(records),
        "exact_best_rate": mean("exact_best"),
        "top3_hit_rate": mean("top3_hit"),
        "mean_normalized_repairability_regret": mean(
            "normalized_repairability_regret"
        ),
    }


def _oracle_pool_opportunity(
    grouped: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    improvements = []
    boundary_best = 0
    for rows in grouped.values():
        base = [row for row in rows if row["candidate_kind"] != "boundary_only"]
        best_all = max(rows, key=lambda row: float(row["repairability_score"]))
        best_base = max(base, key=lambda row: float(row["repairability_score"]))
        improvements.append(
            float(best_all["repairability_score"])
            - float(best_base["repairability_score"])
        )
        boundary_best += int(best_all["candidate_kind"] == "boundary_only")
    return {
        "state_count": len(grouped),
        "boundary_unique_best_rate": boundary_best / len(grouped),
        "mean_best_score_improvement_over_base_pool": statistics.fmean(improvements),
        "positive_opportunity_rate": sum(value > 1e-12 for value in improvements)
        / len(improvements),
    }


def run_augcontrol_training(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    config = _read_json(config_path)
    validate_augcontrol_training_config(config)

    registered_paths = {
        "label_design": _project_path(project_root, str(config["label_design"])),
        "data_design": _project_path(project_root, str(config["data_design"])),
        "source_cohort": _project_path(project_root, str(config["source_cohort"])),
    }
    for name, path in registered_paths.items():
        if sha256_file(path) != str(config[f"{name}_sha256"]):
            raise ValueError(f"STRIDE augcontrol registered {name} differs")
    data_design = _read_json(registered_paths["data_design"])
    validate_repairability_data_design(data_design)
    source_cohort = _read_json(registered_paths["source_cohort"])
    expected_split = {
        str(map_id): split
        for split, map_ids in dict(source_cohort["effective_map_split"]).items()
        for map_id in map_ids
    }

    labels = _project_path(project_root, str(config["labels"]))
    selection_path = _project_path(project_root, str(config["selection"]))
    aggregate_path = labels / "candidate_aggregates.jsonl"
    primary_pair_path = labels / "dominance_pairs.jsonl"
    conflict_pair_path = labels / "conflict_only_dominance_pairs.jsonl"
    label_summary_path = labels / "label_build_summary.json"
    label_summary = _read_json(label_summary_path)
    if (
        label_summary.get("schema") != BUILD_SCHEMA
        or label_summary.get("controller_id") != CONTROLLER_ID
        or label_summary.get("label_schema") != LABEL_SCHEMA
        or bool(label_summary.get("runtime_used_in_label"))
        or bool(label_summary.get("future_trajectory_used_in_label"))
    ):
        raise ValueError("STRIDE augcontrol label summary is invalid")

    candidates, grouped = _load_candidates(
        aggregate_path,
        selection_path,
        topology_threshold=float(data_design["topology_boundary_threshold"]),
    )
    observed_split: defaultdict[str, set[str]] = defaultdict(set)
    for row in candidates:
        observed_split[str(row["map_id"])].add(str(row["split"]))
    if set(observed_split) != set(expected_split) or any(
        values != {expected_split[map_id]}
        for map_id, values in observed_split.items()
    ):
        raise ValueError("STRIDE augcontrol map-held-out split differs")

    candidate_index = {
        (str(row["state_id"]), str(row["candidate_id"])): int(
            row["candidate_index"]
        )
        for row in candidates
    }
    primary_train = _load_pair_table(
        primary_pair_path,
        candidate_index,
        split="train",
        expected_schema=LABEL_SCHEMA,
    )
    primary_validation = _load_pair_table(
        primary_pair_path,
        candidate_index,
        split="validation",
        expected_schema=LABEL_SCHEMA,
    )
    conflict_train = _load_pair_table(
        conflict_pair_path,
        candidate_index,
        split="train",
        expected_schema=CONFLICT_ONLY_LABEL_SCHEMA,
    )
    conflict_validation = _load_pair_table(
        conflict_pair_path,
        candidate_index,
        split="validation",
        expected_schema=CONFLICT_ONLY_LABEL_SCHEMA,
    )

    names, input_specs = _input_specifications()
    candidate_values = _candidate_matrix(candidates)
    parameters = dict(config["model_parameters"])
    primary_train_values = _pair_matrix(
        candidate_values, primary_train, input_specs
    )
    primary_estimator = _fit_registered_model(
        primary_train_values,
        primary_train["labels"],
        primary_train["weights"],
        parameters,
    )
    primary_validation_values = _pair_matrix(
        candidate_values, primary_validation, input_specs
    )
    pairwise = _pairwise_metrics(
        primary_estimator, primary_validation_values, primary_validation
    )
    conflict_train_values = _pair_matrix(
        candidate_values, conflict_train, input_specs
    )
    conflict_estimator = _fit_registered_model(
        conflict_train_values,
        conflict_train["labels"],
        conflict_train["weights"],
        parameters,
    )
    conflict_validation_values = _pair_matrix(
        candidate_values, conflict_validation, input_specs
    )
    conflict_pairwise = _pairwise_metrics(
        conflict_estimator, conflict_validation_values, conflict_validation
    )

    validation_grouped = {
        state_id: rows
        for state_id, rows in grouped.items()
        if str(rows[0]["split"]) == "validation"
    }
    primary_predictions = _model_predictions(
        validation_grouped, primary_estimator, input_specs
    )
    primary_base_predictions = _model_predictions(
        validation_grouped, primary_estimator, input_specs, base_only=True
    )
    conflict_predictions = _model_predictions(
        validation_grouped, conflict_estimator, input_specs
    )
    frozen_bundle = _project_path(project_root, str(config["frozen_v2_bundle"]))
    if sha256_file(frozen_bundle / "controller_manifest.json") != str(
        config["frozen_v2_manifest_sha256"]
    ):
        raise ValueError("STRIDE augcontrol frozen V2 bundle differs")
    frozen_predictions = _frozen_predictions(validation_grouped, frozen_bundle)
    frozen_base_predictions = _frozen_predictions(
        {
            state_id: [
                row for row in rows if row["candidate_kind"] != "boundary_only"
            ]
            for state_id, rows in validation_grouped.items()
        },
        frozen_bundle,
    )

    records_by_model = {
        CONTROLLER_ID: _prediction_records(
            CONTROLLER_ID,
            primary_predictions,
            validation_grouped,
            evaluation_pool="augmented",
        ),
        f"{CONTROLLER_ID}/base-pool-only": _prediction_records(
            f"{CONTROLLER_ID}/base-pool-only",
            primary_base_predictions,
            validation_grouped,
            evaluation_pool="base",
        ),
        CONFLICT_CONTROLLER_ID: _prediction_records(
            CONFLICT_CONTROLLER_ID,
            conflict_predictions,
            validation_grouped,
            evaluation_pool="augmented",
        ),
        "v2-full/augmented-pool": _prediction_records(
            "v2-full/augmented-pool",
            frozen_predictions,
            validation_grouped,
            evaluation_pool="augmented",
        ),
        "v2-full/base-pool": _prediction_records(
            "v2-full/base-pool",
            frozen_base_predictions,
            validation_grouped,
            evaluation_pool="base",
        ),
    }
    metrics = {
        model_id: _mean_metrics(records)
        for model_id, records in records_by_model.items()
    }
    primary_metrics = metrics[CONTROLLER_ID]
    frozen_metrics = metrics["v2-full/augmented-pool"]
    gates_config = dict(config["offline_gates"])
    primary_groups = {
        (row["field"], row["value"]): row
        for row in primary_metrics["subgroups"]
    }
    frozen_groups = {
        (row["field"], row["value"]): row
        for row in frozen_metrics["subgroups"]
    }
    topology_comparisons = []
    for group in ("control", "boundary_relevant"):
        key = ("topology_group", group)
        delta = float(
            primary_groups[key]["mean_normalized_repairability_regret"]
        ) - float(frozen_groups[key]["mean_normalized_repairability_regret"])
        topology_comparisons.append(
            {
                "topology_group": group,
                "state_count": int(primary_groups[key]["state_count"]),
                "normalized_regret_delta_vs_v2": delta,
                "passed": delta
                <= float(
                    gates_config[
                        "maximum_topology_group_normalized_regret_degradation"
                    ]
                )
                + 1e-12,
            }
        )
    regret_improvement = float(
        frozen_metrics["mean_normalized_repairability_regret"]
    ) - float(primary_metrics["mean_normalized_repairability_regret"])
    relative_regret_improvement = regret_improvement / max(
        float(frozen_metrics["mean_normalized_repairability_regret"]), 1e-12
    )
    validation_map_count = len(
        {str(rows[0]["map_id"]) for rows in validation_grouped.values()}
    )
    promotion_gates = {
        "validation_map_coverage": validation_map_count
        >= int(gates_config["minimum_validation_maps"]),
        "pairwise_accuracy_beats_majority": float(pairwise["accuracy"])
        >= float(pairwise["weighted_majority_accuracy"])
        + float(
            gates_config[
                "minimum_pairwise_accuracy_gain_over_weighted_majority"
            ]
        )
        - 1e-12,
        "normalized_regret_improves_v2": (
            relative_regret_improvement + 1e-12
            >= float(
                gates_config[
                    "minimum_relative_normalized_regret_improvement_over_frozen_v2"
                ]
            )
            or regret_improvement + 1e-12
            >= float(
                gates_config[
                    "absolute_normalized_regret_improvement_alternative"
                ]
            )
        ),
        "exact_best_rate_noninferior": float(primary_metrics["exact_best_rate"])
        + float(gates_config["exact_best_rate_noninferiority_tolerance"])
        + 1e-12
        >= float(frozen_metrics["exact_best_rate"]),
        "top3_hit_rate_noninferior": float(primary_metrics["top3_hit_rate"])
        + float(gates_config["top3_hit_rate_noninferiority_tolerance"])
        + 1e-12
        >= float(frozen_metrics["top3_hit_rate"]),
        "topology_groups_noninferior": all(
            row["passed"] for row in topology_comparisons
        ),
    }

    output.mkdir(parents=True, exist_ok=True)
    all_records = [
        row
        for model_id in sorted(records_by_model)
        for row in records_by_model[model_id]
    ]
    prediction_path = output / "validation_predictions.jsonl"
    _write_jsonl(prediction_path, all_records)
    source_manifest = _read_json(frozen_bundle / "controller_manifest.json")
    source_hashes = {
        "config": sha256_file(config_path),
        "selection": sha256_file(selection_path),
        "label_summary": sha256_file(label_summary_path),
        "candidate_aggregates": sha256_file(aggregate_path),
        "dominance_pairs": sha256_file(primary_pair_path),
        "conflict_only_pairs": sha256_file(conflict_pair_path),
        "frozen_v2_manifest": sha256_file(
            frozen_bundle / "controller_manifest.json"
        ),
    }
    exported = _export_diagnostic_controller(
        root=output / CONTROLLER_ID,
        controller_id=CONTROLLER_ID,
        estimator=primary_estimator,
        feature_names=names,
        candidates=candidates,
        grouped=grouped,
        source_bundle=frozen_bundle,
        source_manifest=source_manifest,
        parameters=parameters,
        training_pair_count=len(primary_train["labels"]),
        source_hashes=source_hashes,
    )
    integrity_gates = {
        "label_summary_identity": True,
        "map_held_out_split": True,
        "feature_schema": len(names) == 124,
        "pairwise_input_dimension": len(input_specs) == 147,
        "formal_ood_not_read": True,
        "test_data_not_read": True,
        "runtime_not_used_in_label": True,
        "portable_equivalence": bool(exported["equivalence"]["passed"]),
    }
    report = {
        "schema": REPORT_SCHEMA,
        "controller_id": CONTROLLER_ID,
        "conflict_ablation_id": CONFLICT_CONTROLLER_ID,
        "scientific_status": "shadow_eligible"
        if all(integrity_gates.values()) and all(promotion_gates.values())
        else "offline_gate_failed",
        "default_replacement_allowed": False,
        "formal_speed_claim": False,
        "formal_ood_data_read": False,
        "test_data_read": False,
        "runtime_used_in_label": False,
        "future_trajectory_used_in_label": False,
        "state_count": len(grouped),
        "train_state_count": sum(
            str(rows[0]["split"]) == "train" for rows in grouped.values()
        ),
        "validation_state_count": len(validation_grouped),
        "candidate_count": len(candidates),
        "train_pair_state_count": int(primary_train["state_count"]),
        "validation_pair_state_count": int(primary_validation["state_count"]),
        "primary_pairwise_validation": pairwise,
        "conflict_pairwise_validation": conflict_pairwise,
        "metrics": metrics,
        "oracle_pool_opportunity": _oracle_pool_opportunity(validation_grouped),
        "normalized_regret_improvement_vs_v2": regret_improvement,
        "relative_normalized_regret_improvement_vs_v2": relative_regret_improvement,
        "topology_group_comparisons": topology_comparisons,
        "integrity_gates": integrity_gates,
        "promotion_gates": promotion_gates,
        "offline_passed": all(integrity_gates.values())
        and all(promotion_gates.values()),
        "shadow_eligible": all(integrity_gates.values())
        and all(promotion_gates.values()),
        "export": exported,
        "model_parameters": parameters,
        "input_dimension": len(input_specs),
        "source_sha256": source_hashes,
        "artifacts": {
            "validation_predictions": prediction_path.name,
            "validation_predictions_sha256": sha256_file(prediction_path),
        },
    }
    _write_json(output / "augcontrol_training_report.json", report)
    return report


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "run_augcontrol_training",
    "validate_augcontrol_training_config",
]
