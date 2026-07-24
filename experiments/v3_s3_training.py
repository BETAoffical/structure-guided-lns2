from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.repair_aware import (
    PORTABLE_SCALAR_MODEL_SCHEMA,
    PortableScalarModel,
    load_portable_scalar_model,
)
from experiments.repair_aware_training import _balanced_map_folds, _hist_trees
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
)
from experiments.v3_s3 import (
    S3_ACTION_TEMPLATES,
    S3_HORIZON,
    S3_MODEL_NAMES,
    S3_TEMPORAL_FEATURE_NAMES,
    S3ActionTemplate,
    V3_S3_BUNDLE_SCHEMA,
    V3_S3_FEATURE_SCHEMA_ID,
    V3_S3_FEATURE_SCHEMA_SHA256,
    V3_S3_FULL_FEATURE_NAMES,
    V3_S3_OBJECTIVE_ID,
    V3_S3_PROFILE,
    balanced_sequence_templates,
    load_v3_s3_bundle,
    rank_s3_sequences,
    registered_runtime_sequences,
    sequence_feature_row,
    sequence_id,
)


# Windows hybrid-core topology is not always reported through the legacy WMIC
# query used by joblib. The pilot intentionally caps outer training at ten
# processes, so register that known-safe ceiling before scikit-learn is loaded.
os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT", str(max(1, min(10, os.cpu_count() or 1)))
)


V3_S3_TRAINING_SCHEMA = "lns2.v3_s3_training.v2"
MODEL_FAMILIES = ("hist_gradient_boosting", "extra_trees")
HGB_PARAMETERS = {
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "learning_rate": 0.05,
    "l2_regularization": 0.1,
    "random_state": 20260722,
    "early_stopping": False,
}
EXTRA_TREES_PARAMETERS = {
    "n_estimators": 200,
    "min_samples_leaf": 5,
    "max_features": "sqrt",
    "random_state": 20260722,
    "n_jobs": 1,
}
VALID_THRESHOLD_GRID = (0.40, 0.50, 0.60)
NO_PROGRESS_THRESHOLD_GRID = (0.40, 0.50, 0.60)


class _ConstantEstimator:
    def __init__(self, value: float, *, probability: bool):
        self.value = float(value)
        self.probability = bool(probability)
        self.classes_ = [0, 1]

    def predict(self, values: Any) -> list[float]:
        return [self.value] * len(values)

    def predict_proba(self, values: Any) -> Any:
        import numpy as np

        return np.asarray(
            [[1.0 - self.value, self.value] for _ in range(len(values))],
            dtype=float,
        )


def _target_names() -> tuple[str, ...]:
    return tuple(sorted(S3_MODEL_NAMES))


def _runtime_reachable_steps(trial: dict[str, Any]) -> list[dict[str, Any]]:
    """Return labels reachable before the deployed controller must replan."""

    reachable = []
    for raw in sorted(trial["steps"], key=lambda row: int(row["step"])):
        step = dict(raw)
        reachable.append(step)
        if (
            not bool(step.get("template_valid"))
            or not bool(step.get("executed"))
            or (
                step.get("before_fingerprint") is not None
                and step.get("after_fingerprint") is not None
                and str(step["before_fingerprint"])
                == str(step["after_fingerprint"])
            )
            or str(step.get("repair_outcome"))
            in {"hard_failure", "accepted_noop"}
        ):
            break
    return reachable


def _runtime_prefix_targets(trial: dict[str, Any]) -> dict[str, float]:
    steps = _runtime_reachable_steps(trial)
    trajectory = list(map(int, trial["conflict_trajectory"]))
    initial_conflicts = int(trajectory[0])
    final_conflicts = initial_conflicts
    total_seconds = 0.0
    selection_seconds = 0.0
    positive_progress = False
    feasible = False
    for step in steps:
        if bool(step.get("executed")):
            final_conflicts = int(step["conflicts_after"])
            total_seconds += float(step["total_seconds"])
            selection_seconds += float(step.get("selection_seconds", 0.0))
            positive_progress = positive_progress or float(
                step["conflict_reduction"]
            ) > 0.0
            feasible = feasible or str(step.get("repair_outcome")) == "feasible"
            feasible = feasible or int(step["conflicts_after"]) == 0
        else:
            selected_seconds = float(step.get("selection_seconds", 0.0))
            total_seconds += selected_seconds
            selection_seconds += selected_seconds
    return {
        "sequence_net_conflict_reduction": float(
            initial_conflicts - final_conflicts
        ),
        "sequence_total_seconds": float(total_seconds),
        "sequence_selection_seconds": float(selection_seconds),
        "sequence_log_total_seconds": math.log1p(max(1e-9, total_seconds)),
        "sequence_no_progress_probability": float(not positive_progress),
        "sequence_feasible_probability": float(feasible),
    }


def _sequence_rows(
    feature_path: Path, trial_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_rows = _read_jsonl(feature_path)
    feature_keys = [
        (str(row["state_id"]), str(row["sequence_id"])) for row in feature_rows
    ]
    if len(feature_keys) != len(set(feature_keys)):
        raise ValueError("v3-S3 training features contain duplicate sequences")
    features = {key: row for key, row in zip(feature_keys, feature_rows)}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    trial_keys = []
    for trial in _read_jsonl(trial_path):
        key = (str(trial["state_id"]), str(trial["sequence_id"]))
        trial_key = (*key, int(trial["trial_index"]))
        trial_keys.append(trial_key)
        grouped[key].append(trial)
    if len(trial_keys) != len(set(trial_keys)):
        raise ValueError("v3-S3 training trials contain duplicate paired seeds")
    if set(features) != set(grouped):
        raise ValueError("v3-S3 features and trials have different sequence coverage")
    result = []
    for key, feature in sorted(features.items()):
        trials = grouped[key]
        reachable_by_trial = {
            int(trial["trial_index"]): _runtime_reachable_steps(trial)
            for trial in trials
        }
        names = tuple(map(str, feature["feature_names"]))
        values = tuple(map(float, feature["feature_values"]))
        if names != V3_S3_FULL_FEATURE_NAMES or len(values) != len(names):
            raise ValueError("v3-S3 training feature schema is incomplete")
        targets: dict[str, float | None] = {}
        for step in range(1, S3_HORIZON + 1):
            observed = []
            for trial in trials:
                step_rows = [
                    row
                    for row in reachable_by_trial[int(trial["trial_index"])]
                    if int(row["step"]) == step
                ]
                if step_rows:
                    observed.append(dict(step_rows[0]))
            if not observed:
                targets[f"step{step}_template_valid_probability"] = None
                targets[f"step{step}_no_progress_probability"] = None
                targets[f"step{step}_conflict_reduction"] = None
                targets[f"step{step}_log_total_seconds"] = None
                continue
            targets[f"step{step}_template_valid_probability"] = statistics.fmean(
                float(row["template_valid"]) for row in observed
            )
            executed = [row for row in observed if bool(row.get("executed"))]
            if not executed:
                targets[f"step{step}_no_progress_probability"] = None
                targets[f"step{step}_conflict_reduction"] = None
                targets[f"step{step}_log_total_seconds"] = None
            else:
                targets[f"step{step}_no_progress_probability"] = statistics.fmean(
                    float(
                        str(row["repair_outcome"])
                        in {"hard_failure", "accepted_noop"}
                    )
                    for row in executed
                )
                targets[f"step{step}_conflict_reduction"] = statistics.fmean(
                    float(row["conflict_reduction"]) for row in executed
                )
                targets[f"step{step}_log_total_seconds"] = statistics.fmean(
                    math.log1p(max(1e-9, float(row["total_seconds"])))
                    for row in executed
                )
        prefix_targets = [_runtime_prefix_targets(trial) for trial in trials]
        for name in (
            "sequence_net_conflict_reduction",
            "sequence_log_total_seconds",
            "sequence_no_progress_probability",
        ):
            targets[name] = statistics.fmean(
                float(row[name]) for row in prefix_targets
            )
        registered_ids = {
            sequence_id(templates)
            for templates in balanced_sequence_templates(str(feature["state_id"]))
        }
        prefix_reduction = statistics.fmean(
            float(row["sequence_net_conflict_reduction"])
            for row in prefix_targets
        )
        prefix_seconds = statistics.fmean(
            float(row["sequence_total_seconds"])
            for row in prefix_targets
        )
        prefix_no_progress = statistics.fmean(
            float(row["sequence_no_progress_probability"])
            for row in prefix_targets
        )
        prefix_selection_seconds = statistics.fmean(
            float(row["sequence_selection_seconds"])
            for row in prefix_targets
        )
        prefix_feasible = statistics.fmean(
            float(row["sequence_feasible_probability"])
            for row in prefix_targets
        )
        result.append(
            {
                "split": str(feature["split"]),
                "state_id": key[0],
                "sequence_id": key[1],
                "map_id": str(feature["map_id"]),
                "layout_mode": str(feature["layout_mode"]),
                "agent_count": int(feature["agent_count"]),
                "source_stratum": str(feature["source_stratum"]),
                "templates": [dict(value) for value in feature["templates"]],
                "runtime_registered": str(feature["sequence_id"])
                in registered_ids,
                "features": values,
                "targets": targets,
                "actual": {
                    "conflict_reduction": statistics.fmean(
                        float(row["conflict_reduction"]) for row in trials
                    ),
                    "total_seconds": statistics.fmean(
                        float(row["total_seconds"]) for row in trials
                    ),
                    "no_progress_rate": statistics.fmean(
                        float(row["no_progress"]) for row in trials
                    ),
                    "feasible_rate": statistics.fmean(
                        float(row["feasible"]) for row in trials
                    ),
                    "trials": trials,
                    "runtime_prefix_net_conflict_reduction": prefix_reduction,
                    "runtime_prefix_total_seconds": prefix_seconds,
                    "runtime_prefix_selection_seconds": (
                        prefix_selection_seconds
                    ),
                    "runtime_prefix_no_progress_rate": prefix_no_progress,
                    "runtime_prefix_feasible_rate": prefix_feasible,
                },
            }
        )
    train = [row for row in result if row["split"] == "policy_train"]
    diagnostic = [row for row in result if row["split"] == "policy_validation"]
    if not train or not diagnostic:
        raise ValueError("v3-S3 requires train and diagnostic sequence rows")
    if {row["map_id"] for row in train} & {row["map_id"] for row in diagnostic}:
        raise ValueError("v3-S3 train and diagnostic maps overlap")
    return train, diagnostic


def _balanced_weights(rows: list[dict[str, Any]]) -> list[float]:
    states_by_map: dict[str, set[str]] = collections.defaultdict(set)
    sequences_by_state: dict[str, int] = collections.Counter()
    for row in rows:
        states_by_map[str(row["map_id"])].add(str(row["state_id"]))
        sequences_by_state[str(row["state_id"])] += 1
    raw = [
        1.0
        / (
            len(states_by_map)
            * len(states_by_map[str(row["map_id"])])
            * sequences_by_state[str(row["state_id"])]
        )
        for row in rows
    ]
    scale = len(raw) / max(1e-12, math.fsum(raw))
    return [value * scale for value in raw]


def _is_probability_target(name: str) -> bool:
    return str(name).endswith("_probability")


def _normalize_prediction_values(
    name: str, values: Any
) -> list[float]:
    normalized = list(map(float, values))
    if _is_probability_target(name):
        return [min(1.0, max(0.0, value)) for value in normalized]
    if str(name).startswith("step") and str(name).endswith(
        "conflict_reduction"
    ):
        return [max(0.0, value) for value in normalized]
    return normalized


def _fit_target_model(
    name: str,
    values: Any,
    target: Any,
    target_weights: Any,
    family: str,
) -> tuple[str, Any]:
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

    probability = _is_probability_target(name)
    if len(set(map(float, target))) == 1:
        return name, _ConstantEstimator(float(target[0]), probability=probability)
    # Sequence outcomes are aggregated over two or four paired PP seeds.  A
    # target such as 0.5 is an empirical probability, not a positive class.
    # Binarizing at 0.5 would turn one success plus one failure into certainty.
    # Regress the bounded empirical rate directly; the runtime clamps the tree
    # estimate to [0, 1].  This also keeps each state/sequence at one balanced
    # training row instead of duplicating it by its adaptive seed count.
    estimator = (
        HistGradientBoostingRegressor(**HGB_PARAMETERS)
        if family == "hist_gradient_boosting"
        else ExtraTreesRegressor(**EXTRA_TREES_PARAMETERS)
    )
    estimator.fit(values, target, sample_weight=target_weights)
    return name, estimator


def _fit_models(
    rows: list[dict[str, Any]],
    feature_indices: tuple[int, ...],
    family: str,
    *,
    jobs: int = 1,
) -> dict[str, Any]:
    import numpy as np

    if family not in MODEL_FAMILIES:
        raise ValueError(f"unsupported v3-S3 model family: {family}")
    all_values = np.asarray(
        [[row["features"][index] for index in feature_indices] for row in rows],
        dtype=float,
    )
    weights = _balanced_weights(rows)
    specifications = []
    for name in _target_names():
        indices = [
            index
            for index, row in enumerate(rows)
            if row["targets"].get(name) is not None
        ]
        if not indices:
            raise ValueError(f"v3-S3 target has no observations: {name}")
        target = np.asarray(
            [float(rows[index]["targets"][name]) for index in indices], dtype=float
        )
        target_weights = np.asarray([weights[index] for index in indices], dtype=float)
        values = all_values[indices]
        specifications.append((name, values, target, target_weights, family))
    if int(jobs) == 1:
        fitted = [
            _fit_target_model(*specification) for specification in specifications
        ]
    else:
        from joblib import Parallel, delayed

        fitted = Parallel(n_jobs=int(jobs), prefer="processes")(
            delayed(_fit_target_model)(*specification)
            for specification in specifications
        )
    return dict(fitted)


def _predict_models(
    models: dict[str, Any],
    rows: list[dict[str, Any]],
    feature_indices: tuple[int, ...],
) -> dict[str, list[float]]:
    import numpy as np

    values = np.asarray(
        [[row["features"][index] for index in feature_indices] for row in rows],
        dtype=float,
    )
    result = {}
    for name, estimator in models.items():
        predicted = _normalize_prediction_values(
            name, estimator.predict(values)
        )
        result[name] = list(map(float, predicted))
        if name.endswith("log_total_seconds"):
            step = name.split("_", 1)[0]
            result[f"{step}_total_seconds"] = [
                max(1e-9, math.expm1(min(50.0, float(value))))
                for value in predicted
            ]
    return result


def _strict_feature_projection(
    rows: list[dict[str, Any]], names: tuple[str, ...]
) -> tuple[tuple[int, ...], dict[str, Any]]:
    # Strict removals must hold over the complete training corpus. Sampling can
    # incorrectly classify a sparse feature as constant when its non-zero maps
    # happen to occur later in the manifest.
    sample = rows
    columns = [tuple(float(row["features"][index]) for row in sample) for index in range(len(names))]
    kept = []
    signatures: dict[str, int] = {}
    removed = {}
    for index, column in enumerate(columns):
        minimum = min(column)
        maximum = max(column)
        if abs(maximum - minimum) <= 1e-15:
            removed[names[index]] = {"reason": "constant", "canonical": None}
            continue
        anchor = next((value for value in column if abs(value) > 1e-15), 1.0)
        normalized = tuple(round(value / anchor, 12) for value in column)
        signature = hashlib.sha256(repr(normalized).encode("utf-8")).hexdigest()
        previous = signatures.get(signature)
        if previous is not None:
            scale = anchor / next(
                value for value in columns[previous] if abs(value) > 1e-15
            )
            if all(
                abs(left - scale * right) <= 1e-12
                for left, right in zip(column, columns[previous])
            ):
                removed[names[index]] = {
                    "reason": "linear_equivalent",
                    "canonical": names[previous],
                    "scale": scale,
                }
                continue
        signatures[signature] = index
        kept.append(index)
    return tuple(kept), {
        "source_feature_count": len(names),
        "kept_feature_count": len(kept),
        "removed": removed,
    }


def _used_feature_indices(
    models: dict[str, Any], feature_indices: tuple[int, ...]
) -> tuple[int, ...]:
    used_local = set()
    for estimator in models.values():
        if isinstance(estimator, _ConstantEstimator):
            continue
        predictors = getattr(estimator, "_predictors", None)
        if predictors is not None:
            for stage in predictors:
                for predictor in stage:
                    for node in predictor.nodes:
                        if not bool(node["is_leaf"]):
                            used_local.add(int(node["feature_idx"]))
        for tree in getattr(estimator, "estimators_", ()):
            for node in range(tree.tree_.node_count):
                feature = int(tree.tree_.feature[node])
                if feature >= 0:
                    used_local.add(feature)
    return tuple(feature_indices[index] for index in sorted(used_local))


def _oof_predictions(
    rows: list[dict[str, Any]],
    feature_indices: tuple[int, ...],
    family: str,
    *,
    jobs: int = 1,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], set[int]]:
    result = {name: [math.nan] * len(rows) for name in _target_names()}
    for step in range(1, S3_HORIZON + 1):
        result[f"step{step}_total_seconds"] = [math.nan] * len(rows)
    result["sequence_total_seconds"] = [math.nan] * len(rows)
    index_by_id = {id(row): index for index, row in enumerate(rows)}
    fold_reports = []
    used = set()
    for fold in _balanced_map_folds(rows):
        held_maps = set(fold["validation_maps"])
        training = [row for row in rows if row["map_id"] not in held_maps]
        held = [row for row in rows if row["map_id"] in held_maps]
        models = _fit_models(training, feature_indices, family, jobs=jobs)
        used.update(_used_feature_indices(models, feature_indices))
        predicted = _predict_models(models, held, feature_indices)
        for local, row in enumerate(held):
            target_index = index_by_id[id(row)]
            for name, values in predicted.items():
                result[name][target_index] = float(values[local])
        fold_reports.append(
            {
                **fold,
                "training_sequence_count": len(training),
                "validation_sequence_count": len(held),
            }
        )
    if any(any(not math.isfinite(value) for value in values) for values in result.values()):
        raise ValueError("v3-S3 OOF predictions are incomplete")
    return result, fold_reports, used


def _state_groups(rows: list[dict[str, Any]]) -> list[list[int]]:
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["state_id"])].append(index)
    return [grouped[key] for key in sorted(grouped)]


def _selection_metrics(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[float]],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    selected_rows = []
    size_counts = collections.Counter()
    risk_relaxed_count = 0
    for indices in _state_groups(rows):
        indices = [
            index for index in indices if bool(rows[index]["runtime_registered"])
        ]
        if not indices:
            continue
        sequences = [
            tuple(S3ActionTemplate.from_payload(value) for value in rows[index]["templates"])
            for index in indices
        ]
        local_predictions = {
            name: [values[index] for index in indices]
            for name, values in predictions.items()
        }
        order = rank_s3_sequences(sequences, local_predictions, thresholds)
        if not order:
            order = rank_s3_sequences(
                sequences,
                local_predictions,
                thresholds,
                allow_risk_relaxation=True,
            )
            if order:
                risk_relaxed_count += 1
        if not order:
            continue
        selected = rows[indices[order[0]]]
        selected_rows.append(selected)
        for template in selected["templates"]:
            size_counts[int(template["requested_size"])] += 1
    if not selected_rows:
        return {
            "state_count": 0,
            "selection_coverage": 0.0,
            "effective_rate": 0.0,
            "no_progress_rate": 1.0,
            "feasible_rate": 0.0,
            "mean_conflict_reduction": 0.0,
            "mean_total_seconds": math.inf,
            "mean_selection_seconds": math.inf,
            "conflict_reduction_per_total_second": 0.0,
            "size_fractions": {},
            "risk_relaxed_count": 0,
            "risk_relaxed_fraction": 0.0,
            "mean_runtime_prefix_net_conflict_reduction": 0.0,
            "mean_runtime_prefix_total_seconds": math.inf,
            "runtime_prefix_efficiency": 0.0,
            "runtime_prefix_no_progress_rate": 1.0,
            "by_agent_count": {},
        }
    counterfactual_reduction = math.fsum(
        float(row["actual"]["conflict_reduction"]) for row in selected_rows
    )
    counterfactual_seconds = math.fsum(
        float(row["actual"]["total_seconds"]) for row in selected_rows
    )
    runtime_prefix_reduction = math.fsum(
        float(row["actual"]["runtime_prefix_net_conflict_reduction"])
        for row in selected_rows
    )
    runtime_prefix_seconds = math.fsum(
        float(row["actual"]["runtime_prefix_total_seconds"])
        for row in selected_rows
    )
    by_agents = {}
    for agents in sorted({int(row["agent_count"]) for row in selected_rows}):
        subset = [row for row in selected_rows if int(row["agent_count"]) == agents]
        subtotal_reduction = math.fsum(
            float(row["actual"]["runtime_prefix_net_conflict_reduction"])
            for row in subset
        )
        subtotal_seconds = math.fsum(
            float(row["actual"]["runtime_prefix_total_seconds"])
            for row in subset
        )
        by_agents[str(agents)] = {
            "state_count": len(subset),
            "conflict_reduction_sum": subtotal_reduction,
            "total_seconds_sum": subtotal_seconds,
            "efficiency": subtotal_reduction / max(1e-9, subtotal_seconds),
            "mean_conflict_reduction": statistics.fmean(
                float(row["actual"]["runtime_prefix_net_conflict_reduction"])
                for row in subset
            ),
            "no_progress_rate": statistics.fmean(
                float(row["actual"]["runtime_prefix_no_progress_rate"])
                for row in subset
            ),
        }
    total_templates = sum(size_counts.values())
    return {
        "state_count": len(selected_rows),
        "selection_coverage": len(selected_rows) / len(_state_groups(rows)),
        "effective_rate": statistics.fmean(
            1.0 - float(row["actual"]["runtime_prefix_no_progress_rate"])
            for row in selected_rows
        ),
        "no_progress_rate": statistics.fmean(
            float(row["actual"]["runtime_prefix_no_progress_rate"])
            for row in selected_rows
        ),
        "feasible_rate": statistics.fmean(
            float(row["actual"]["runtime_prefix_feasible_rate"])
            for row in selected_rows
        ),
        "mean_conflict_reduction": statistics.fmean(
            float(row["actual"]["runtime_prefix_net_conflict_reduction"])
            for row in selected_rows
        ),
        "mean_total_seconds": statistics.fmean(
            float(row["actual"]["runtime_prefix_total_seconds"])
            for row in selected_rows
        ),
        "mean_selection_seconds": statistics.fmean(
            float(row["actual"]["runtime_prefix_selection_seconds"])
            for row in selected_rows
        ),
        "conflict_reduction_per_total_second": runtime_prefix_reduction
        / max(1e-9, runtime_prefix_seconds),
        "mean_runtime_prefix_net_conflict_reduction": statistics.fmean(
            float(row["actual"]["runtime_prefix_net_conflict_reduction"])
            for row in selected_rows
        ),
        "mean_runtime_prefix_total_seconds": statistics.fmean(
            float(row["actual"]["runtime_prefix_total_seconds"])
            for row in selected_rows
        ),
        "runtime_prefix_efficiency": runtime_prefix_reduction
        / max(1e-9, runtime_prefix_seconds),
        "runtime_prefix_no_progress_rate": statistics.fmean(
            float(row["actual"]["runtime_prefix_no_progress_rate"])
            for row in selected_rows
        ),
        "counterfactual_three_step_mean_conflict_reduction": (
            counterfactual_reduction / len(selected_rows)
        ),
        "counterfactual_three_step_mean_total_seconds": (
            counterfactual_seconds / len(selected_rows)
        ),
        "counterfactual_three_step_efficiency": counterfactual_reduction
        / max(1e-9, counterfactual_seconds),
        "size_fractions": {
            str(size): count / total_templates for size, count in sorted(size_counts.items())
        },
        "risk_relaxed_count": risk_relaxed_count,
        "risk_relaxed_fraction": risk_relaxed_count / len(selected_rows),
        "by_agent_count": by_agents,
    }


def _calibrate_thresholds(
    rows: list[dict[str, Any]], predictions: dict[str, list[float]]
) -> dict[str, Any]:
    grid = []
    for valid in VALID_THRESHOLD_GRID:
        for no_progress in NO_PROGRESS_THRESHOLD_GRID:
            for sequence_no_progress in NO_PROGRESS_THRESHOLD_GRID:
                thresholds = {
                    "minimum_template_valid_probability": valid,
                    "maximum_no_progress_probability": no_progress,
                    "maximum_sequence_no_progress_probability": (
                        sequence_no_progress
                    ),
                }
                metrics = _selection_metrics(rows, predictions, thresholds)
                grid.append({"thresholds": thresholds, "metrics": metrics})
    selected = max(
        grid,
        key=lambda row: (
            float(row["metrics"]["conflict_reduction_per_total_second"]),
            float(row["metrics"]["mean_conflict_reduction"]),
            -float(row["metrics"]["no_progress_rate"]),
            -float(row["metrics"]["risk_relaxed_fraction"]),
            float(row["metrics"]["selection_coverage"]),
        ),
    )
    return {"grid": grid, "selected": selected}


def _cached_oof_predictions(
    cache: dict[
        tuple[str, tuple[int, ...]],
        tuple[dict[str, list[float]], list[dict[str, Any]], set[int]],
    ],
    rows: list[dict[str, Any]],
    feature_indices: tuple[int, ...],
    family: str,
    *,
    jobs: int,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], set[int]]:
    key = (str(family), tuple(feature_indices))
    if key not in cache:
        cache[key] = _oof_predictions(
            rows,
            tuple(feature_indices),
            family,
            jobs=jobs,
        )
    return cache[key]


def _schema_candidates(
    train: list[dict[str, Any]],
    *,
    jobs: int = 1,
    oof_cache: dict[
        tuple[str, tuple[int, ...]],
        tuple[dict[str, list[float]], list[dict[str, Any]], set[int]],
    ]
    | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = oof_cache if oof_cache is not None else {}
    full = tuple(range(len(V3_S3_FULL_FEATURE_NAMES)))
    deduplicated, audit = _strict_feature_projection(train, V3_S3_FULL_FEATURE_NAMES)
    _predictions, _folds, used = _cached_oof_predictions(
        cache,
        train,
        deduplicated,
        "hist_gradient_boosting",
        jobs=jobs,
    )
    stable = tuple(sorted(used))
    if not stable:
        stable = deduplicated
    candidates = [
        {"name": "full", "indices": full},
        {"name": "strict_deduplicated", "indices": deduplicated},
        {"name": "stable_compact", "indices": stable},
    ]
    # Group ablations are evaluated with the same map-group OOF protocol.
    # They are eligible for deployment only through the common retention gates
    # below; no group is removed merely because a single fitted tree ignored it.
    for group in ("state", "proposal", "realized", "history", "sequence"):
        indices = tuple(
            index
            for index in deduplicated
            if V3_S3_FULL_FEATURE_NAMES[index].split(".", 1)[0] != group
        )
        if indices:
            candidates.append({"name": f"without_{group}", "indices": indices})
    unique = []
    seen = set()
    for row in candidates:
        key = tuple(row["indices"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique, audit


def _choose_feature_schema(
    train: list[dict[str, Any]],
    *,
    jobs: int = 1,
    oof_cache: dict[
        tuple[str, tuple[int, ...]],
        tuple[dict[str, list[float]], list[dict[str, Any]], set[int]],
    ]
    | None = None,
) -> dict[str, Any]:
    cache = oof_cache if oof_cache is not None else {}
    candidates, dedup_audit = _schema_candidates(
        train,
        jobs=jobs,
        oof_cache=cache,
    )
    reports = []
    for candidate in candidates:
        predictions, folds, used = _cached_oof_predictions(
            cache,
            train,
            tuple(candidate["indices"]),
            "hist_gradient_boosting",
            jobs=jobs,
        )
        calibration = _calibrate_thresholds(train, predictions)
        metrics = dict(calibration["selected"]["metrics"])
        reports.append(
            {
                "name": candidate["name"],
                "indices": list(candidate["indices"]),
                "feature_count": len(candidate["indices"]),
                "runtime_used_feature_count": len(used),
                "folds": folds,
                "calibration": calibration,
                "metrics": metrics,
            }
        )
    best = max(
        reports,
        key=lambda row: float(row["metrics"]["conflict_reduction_per_total_second"]),
    )
    eligible = []
    for row in reports:
        metrics = row["metrics"]
        efficiency_ok = float(metrics["conflict_reduction_per_total_second"]) + 1e-12 >= 0.99 * float(
            best["metrics"]["conflict_reduction_per_total_second"]
        )
        reduction_ok = float(metrics["mean_conflict_reduction"]) + 1e-12 >= 0.98 * float(
            best["metrics"]["mean_conflict_reduction"]
        )
        no_progress_ok = float(metrics["no_progress_rate"]) <= float(
            best["metrics"]["no_progress_rate"]
        ) + 0.01 + 1e-12
        scale_ok = all(
            float(dict(metrics["by_agent_count"]).get(str(agents), {}).get("efficiency", 0.0))
            + 1e-12
            >= 0.95
            * float(
                dict(best["metrics"]["by_agent_count"])
                .get(str(agents), {})
                .get("efficiency", 0.0)
            )
            for agents in (80, 100, 200, 400, 600)
        )
        row["checks"] = {
            "efficiency_retention_99pct": efficiency_ok,
            "reduction_retention_98pct": reduction_ok,
            "no_progress_delta_1pct": no_progress_ok,
            "all_agent_scales_efficiency_95pct": scale_ok,
        }
        if all(row["checks"].values()):
            eligible.append(row)
    selected = min(eligible or [best], key=lambda row: (int(row["feature_count"]), row["name"]))
    return {
        "deduplication": dedup_audit,
        "candidates": reports,
        "best_schema": best["name"],
        "selected_schema": selected["name"],
        "selected_indices": selected["indices"],
        "selected_feature_names": [
            V3_S3_FULL_FEATURE_NAMES[index] for index in selected["indices"]
        ],
    }


def _family_report(
    train: list[dict[str, Any]],
    feature_indices: tuple[int, ...],
    family: str,
    *,
    jobs: int = 1,
    oof_cache: dict[
        tuple[str, tuple[int, ...]],
        tuple[dict[str, list[float]], list[dict[str, Any]], set[int]],
    ]
    | None = None,
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    started = time.perf_counter()
    cache = oof_cache if oof_cache is not None else {}
    cache_key = (str(family), tuple(feature_indices))
    cache_hit = cache_key in cache
    predictions, folds, used = _cached_oof_predictions(
        cache,
        train,
        feature_indices,
        family,
        jobs=jobs,
    )
    calibration = _calibrate_thresholds(train, predictions)
    return (
        {
            "family": family,
            "oof_seconds": time.perf_counter() - started,
            "oof_cache_hit": cache_hit,
            "folds": folds,
            "runtime_used_feature_count": len(used),
            "thresholds": dict(calibration["selected"]["thresholds"]),
            "metrics": dict(calibration["selected"]["metrics"]),
        },
        predictions,
    )


def _extra_trees_statistically_preferred(
    reports: dict[str, dict[str, Any]]
) -> bool:
    hgb = reports["hist_gradient_boosting"]["metrics"]
    extra = reports["extra_trees"]["metrics"]
    return (
        float(extra["conflict_reduction_per_total_second"])
        + 1e-12
        >= 1.03 * float(hgb["conflict_reduction_per_total_second"])
        and float(extra["mean_conflict_reduction"]) + 1e-12
        >= 0.98 * float(hgb["mean_conflict_reduction"])
        and float(extra["no_progress_rate"])
        <= float(hgb["no_progress_rate"]) + 0.01 + 1e-12
    )


def _tree_payload(estimator: Any, *, probability: bool) -> tuple[float, list[list[dict[str, Any]]], str]:
    if isinstance(estimator, _ConstantEstimator):
        return estimator.value, [], "identity"
    if probability and hasattr(estimator, "classes_"):
        raise TypeError("v3-S3 probability targets must preserve empirical rates")
    if hasattr(estimator, "_predictors"):
        return (
            float(estimator._baseline_prediction[0, 0]),
            _hist_trees(estimator),
            "identity",
        )
    estimators = list(estimator.estimators_)
    divisor = float(len(estimators))
    trees = []
    for tree in estimators:
        source = tree.tree_
        nodes = []
        for index in range(source.node_count):
            leaf = int(source.children_left[index]) < 0
            if leaf:
                raw = source.value[index][0]
                value = float(raw[0]) / divisor
            else:
                value = 0.0
            nodes.append(
                {
                    "value": value,
                    "feature_idx": 0 if leaf else int(source.feature[index]),
                    "num_threshold": 0.0 if leaf else float(source.threshold[index]),
                    "missing_go_to_left": True,
                    "left": 0 if leaf else int(source.children_left[index]),
                    "right": 0 if leaf else int(source.children_right[index]),
                    "is_leaf": leaf,
                }
            )
        trees.append(nodes)
    return 0.0, trees, "identity"


def _portable_payload(
    name: str,
    estimator: Any,
    feature_names: list[str],
    source_fingerprint: str,
) -> dict[str, Any]:
    baseline, trees, transform = _tree_payload(
        estimator, probability=_is_probability_target(name)
    )
    payload = {
        "schema": PORTABLE_SCALAR_MODEL_SCHEMA,
        "schema_version": 1,
        "name": name,
        "profile": V3_S3_PROFILE,
        "feature_names": feature_names,
        "baseline": baseline,
        "trees": trees,
        "transform": transform,
        "source_fingerprint": source_fingerprint,
    }
    payload["semantic_fingerprint"] = _fingerprint(
        {
            "name": name,
            "profile": V3_S3_PROFILE,
            "feature_names": feature_names,
            "baseline": baseline,
            "trees": trees,
            "transform": transform,
        }
    )
    return payload


def _export_family(
    *,
    models: dict[str, Any],
    family: str,
    rows: list[dict[str, Any]],
    feature_indices: tuple[int, ...],
    feature_names: list[str],
    output: Path,
    source_fingerprint: str,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reference = _predict_models(models, rows, feature_indices)
    model_rows = {}
    parity = {}
    dense = [
        {
            "feature_profile": V3_S3_PROFILE,
            "feature_names": tuple(feature_names),
            "feature_values": tuple(row["features"][index] for index in feature_indices),
        }
        for row in rows
    ]
    for name, estimator in models.items():
        payload = _portable_payload(name, estimator, feature_names, source_fingerprint)
        path = output / f"v3_s3__{name}.json"
        _write_json(path, payload)
        portable = load_portable_scalar_model(payload)
        python_portable: PortableScalarModel = replace(portable, native_predictor=None)
        values = _normalize_prediction_values(
            name, python_portable.predict(dense)
        )
        expected = reference[name]
        delta = max(
            (abs(float(left) - float(right)) for left, right in zip(values, expected)),
            default=0.0,
        )
        if delta > 1e-12:
            raise ValueError(f"v3-S3 portable parity failed: {family}/{name} {delta}")
        parity[name] = {"sklearn_python_maximum_delta": delta}
        model_rows[name] = {
            "file": f"model_candidates/{family}/{path.name}",
            "sha256": sha256_file(path),
            "semantic_fingerprint": payload["semantic_fingerprint"],
            "tree_count": len(payload["trees"]),
        }
    return {"family": family, "models": model_rows, "portable_parity": parity}


def _prediction_intervals(
    rows: list[dict[str, Any]], predictions: dict[str, list[float]]
) -> dict[str, float]:
    reduction_residuals = []
    seconds_residuals = []
    for index, row in enumerate(rows):
        for step in range(1, S3_HORIZON + 1):
            reduction = row["targets"].get(f"step{step}_conflict_reduction")
            log_seconds = row["targets"].get(f"step{step}_log_total_seconds")
            if reduction is not None:
                reduction_residuals.append(
                    abs(float(reduction) - predictions[f"step{step}_conflict_reduction"][index])
                )
            if log_seconds is not None:
                actual_seconds = math.expm1(float(log_seconds))
                seconds_residuals.append(
                    abs(actual_seconds - predictions[f"step{step}_total_seconds"][index])
                )
    def quantile(values: list[float]) -> float:
        if not values:
            return math.inf
        ordered = sorted(values)
        return float(ordered[min(len(ordered) - 1, math.ceil(0.90 * len(ordered)) - 1)])
    return {
        "conflict_reduction": quantile(reduction_residuals),
        "total_seconds": quantile(seconds_residuals),
        "coverage": 0.90,
    }


def _continuation_calibration(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[float]],
) -> dict[str, Any]:
    coverage = 0.90

    def quantile(values: list[float]) -> float:
        if not values:
            return math.inf
        ordered = sorted(map(float, values))
        return float(
            ordered[
                min(
                    len(ordered) - 1,
                    math.ceil(coverage * len(ordered)) - 1,
                )
            ]
        )

    def cell(indices: list[int], step: int) -> dict[str, Any]:
        usable = [
            index
            for index in indices
            if bool(rows[index]["runtime_registered"])
            and rows[index]["targets"].get(
                f"step{step}_conflict_reduction"
            )
            is not None
            and rows[index]["targets"].get(f"step{step}_log_total_seconds")
            is not None
            and rows[index]["targets"].get(
                f"step{step}_no_progress_probability"
            )
            is not None
        ]
        if not usable:
            raise ValueError(
                f"v3-S3 continuation step {step} has no observations"
            )
        thresholds = (0.30, 0.40, 0.50, 0.60, 0.70)

        def accuracy(threshold: float) -> float:
            return statistics.fmean(
                float(
                    (
                        predictions[
                            f"step{step}_no_progress_probability"
                        ][index]
                        >= threshold
                    )
                    == (
                        float(
                            rows[index]["targets"][
                                f"step{step}_no_progress_probability"
                            ]
                        )
                        >= 0.5
                    )
                )
                for index in usable
            )

        no_progress_threshold = max(
            thresholds, key=lambda value: (accuracy(value), -abs(value - 0.5))
        )
        reduction_errors = []
        time_errors = []
        for index in usable:
            predicted_reduction = float(
                predictions[f"step{step}_conflict_reduction"][index]
            )
            actual_reduction = float(
                rows[index]["targets"][f"step{step}_conflict_reduction"]
            )
            reduction_errors.append(
                abs(actual_reduction - predicted_reduction)
                / max(1.0, abs(predicted_reduction))
            )
            time_errors.append(
                abs(
                    float(
                        rows[index]["targets"][
                            f"step{step}_log_total_seconds"
                        ]
                    )
                    - float(
                        predictions[f"step{step}_log_total_seconds"][index]
                    )
                )
            )
        return {
            "observation_count": len(usable),
            "no_progress_threshold": float(no_progress_threshold),
            "no_progress_accuracy": accuracy(no_progress_threshold),
            "reduction_relative_error": quantile(reduction_errors),
            "log_total_seconds_error": quantile(time_errors),
        }

    fallback = {}
    cells = {}
    all_indices = list(range(len(rows)))
    for step in range(1, S3_HORIZON + 1):
        fallback[f"step{step}"] = cell(all_indices, step)
        for agents in sorted({int(row["agent_count"]) for row in rows}):
            indices = [
                index
                for index, row in enumerate(rows)
                if int(row["agent_count"]) == agents
            ]
            try:
                candidate = cell(indices, step)
            except ValueError:
                continue
            if int(candidate["observation_count"]) >= 32:
                cells[f"step{step}:agents{agents}"] = candidate
    return {
        "schema": "lns2.v3_s3_continuation.v1",
        "coverage": coverage,
        "minimum_cell_observations": 32,
        "fallback": fallback,
        "cells": cells,
    }


def _baseline_metrics(
    path: Path,
    split: str,
    controller: str,
    *,
    expected_state_ids: set[str],
) -> dict[str, Any]:
    def runtime_prefix(row: dict[str, Any]) -> dict[str, float]:
        trajectory = list(map(int, row["conflict_trajectory"]))
        initial_conflicts = int(trajectory[0])
        final_conflicts = initial_conflicts
        total_seconds = 0.0
        selection_seconds = 0.0
        positive_progress = False
        feasible = False
        for step in sorted(row["steps"], key=lambda value: int(value["step"])):
            total_seconds += float(step["total_seconds"])
            selection_seconds += float(step.get("selection_seconds", 0.0))
            final_conflicts = int(step["conflicts_after"])
            reduction = int(step["conflicts_before"]) - final_conflicts
            positive_progress = positive_progress or reduction > 0
            outcome = str(step.get("repair_outcome"))
            feasible = feasible or outcome == "feasible" or final_conflicts == 0
            if outcome in {"hard_failure", "accepted_noop"}:
                break
        return {
            "conflict_reduction": float(initial_conflicts - final_conflicts),
            "total_seconds": total_seconds,
            "selection_seconds": selection_seconds,
            "no_progress": float(not positive_progress),
            "feasible": float(feasible),
        }

    rows = [
        row
        for row in _read_jsonl(path)
        if str(row["split"]) == split and str(row["controller"]) == controller
    ]
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    if set(grouped) != set(expected_state_ids):
        missing = sorted(set(expected_state_ids) - set(grouped))
        extra = sorted(set(grouped) - set(expected_state_ids))
        raise ValueError(
            f"v3-S3 {controller} baseline state coverage mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    states = []
    for state_id, state_rows in grouped.items():
        trial_indices = [int(row["trial_index"]) for row in state_rows]
        if sorted(trial_indices) != [0, 1]:
            raise ValueError(
                f"v3-S3 {controller} baseline has invalid paired trials: {state_id}"
            )
        if len({int(row["agent_count"]) for row in state_rows}) != 1:
            raise ValueError(
                f"v3-S3 {controller} baseline has inconsistent agent_count: {state_id}"
            )
        prefixes = [runtime_prefix(row) for row in state_rows]
        states.append(
            {
                "conflict_reduction": statistics.fmean(
                    row["conflict_reduction"] for row in prefixes
                ),
                "total_seconds": statistics.fmean(
                    row["total_seconds"] for row in prefixes
                ),
                "no_progress": statistics.fmean(
                    row["no_progress"] for row in prefixes
                ),
                "feasible": statistics.fmean(
                    row["feasible"] for row in prefixes
                ),
                "agent_count": int(state_rows[0]["agent_count"]),
                "selection_seconds": statistics.fmean(
                    row["selection_seconds"] for row in prefixes
                ),
            }
        )
    reduction = math.fsum(row["conflict_reduction"] for row in states)
    seconds = math.fsum(row["total_seconds"] for row in states)
    return {
        "state_count": len(states),
        "effective_rate": statistics.fmean(1.0 - row["no_progress"] for row in states),
        "no_progress_rate": statistics.fmean(row["no_progress"] for row in states),
        "feasible_rate": statistics.fmean(row["feasible"] for row in states),
        "mean_conflict_reduction": statistics.fmean(row["conflict_reduction"] for row in states),
        "mean_total_seconds": statistics.fmean(row["total_seconds"] for row in states),
        "mean_selection_seconds": statistics.fmean(
            row["selection_seconds"] for row in states
        ),
        "conflict_reduction_per_total_second": reduction / max(1e-9, seconds),
        "by_agent_count": {
            str(agents): {
                "conflict_reduction_sum": math.fsum(
                    row["conflict_reduction"]
                    for row in states
                    if row["agent_count"] == agents
                ),
                "total_seconds_sum": math.fsum(
                    row["total_seconds"]
                    for row in states
                    if row["agent_count"] == agents
                ),
                "efficiency": math.fsum(
                    row["conflict_reduction"] for row in states if row["agent_count"] == agents
                )
                / max(
                    1e-9,
                    math.fsum(
                        row["total_seconds"] for row in states if row["agent_count"] == agents
                    ),
                )
            }
            for agents in sorted({row["agent_count"] for row in states})
        },
    }


def _pilot_gate(
    s3: dict[str, Any], v2: dict[str, Any], continuation_reuse_fraction: float
) -> dict[str, bool]:
    small_scales = (80, 100, 200)
    high_scales = (400, 600)
    def cell(metrics: dict[str, Any], agents: int, name: str) -> float:
        return float(
            dict(metrics.get("by_agent_count", {}))
            .get(str(agents), {})
            .get(name, 0.0)
        )

    return {
        "selection_coverage_complete": float(s3["selection_coverage"]) + 1e-12
        >= 1.0,
        "effective_rate_not_below_v2": float(s3["effective_rate"]) + 1e-12 >= float(v2["effective_rate"]),
        "efficiency_at_least_10pct_better": float(s3["conflict_reduction_per_total_second"]) + 1e-12 >= 1.10 * float(v2["conflict_reduction_per_total_second"]),
        "reduction_retention_98pct": float(s3["mean_conflict_reduction"]) + 1e-12 >= 0.98 * float(v2["mean_conflict_reduction"]),
        "no_progress_delta_1pct": float(s3["no_progress_rate"]) <= float(v2["no_progress_rate"]) + 0.01 + 1e-12,
        "small_load_worst_degradation_2pct": all(
            cell(s3, agents, "efficiency") + 1e-12
            >= 0.98 * cell(v2, agents, "efficiency")
            for agents in small_scales
        ),
        "high_load_efficiency_10pct": math.fsum(
            cell(s3, agents, "conflict_reduction_sum")
            for agents in high_scales
        )
        / max(
            1e-9,
            math.fsum(
                cell(s3, agents, "total_seconds_sum")
                for agents in high_scales
            ),
        )
        + 1e-12
        >= 1.10
        * math.fsum(
            cell(v2, agents, "conflict_reduction_sum")
            for agents in high_scales
        )
        / max(
            1e-9,
            math.fsum(
                cell(v2, agents, "total_seconds_sum")
                for agents in high_scales
            ),
        ),
        "direct_continuation_at_least_50pct": continuation_reuse_fraction + 1e-12 >= 0.50,
        "risk_relaxed_at_most_5pct": float(
            s3.get("risk_relaxed_fraction", 0.0)
        )
        <= 0.05 + 1e-12,
        "v2_runtime_call_count_zero": True,
        "adaptive_runtime_call_count_zero": True,
    }


def _with_planner_overhead(
    metrics: dict[str, Any], planner_seconds_per_state: float
) -> dict[str, Any]:
    result = json.loads(json.dumps(metrics))
    overhead = max(0.0, float(planner_seconds_per_state))
    states = int(result["state_count"])
    if states == 0:
        result["planner_inference_seconds_per_state"] = overhead
        return result
    old_total = float(result["mean_total_seconds"]) * states
    reduction_total = float(result["conflict_reduction_per_total_second"]) * old_total
    new_total = old_total + states * overhead
    result["mean_total_seconds"] = new_total / max(1, states)
    result["mean_selection_seconds"] = (
        float(result["mean_selection_seconds"]) + overhead
    )
    result["conflict_reduction_per_total_second"] = reduction_total / max(
        1e-9, new_total
    )
    for row in dict(result["by_agent_count"]).values():
        count = int(row["state_count"])
        row["total_seconds_sum"] = float(row["total_seconds_sum"]) + count * overhead
        row["efficiency"] = float(row["conflict_reduction_sum"]) / max(
            1e-9, float(row["total_seconds_sum"])
        )
    result["planner_inference_seconds_per_state"] = overhead
    return result


def _continuation_diagnostic(
    rows: list[dict[str, Any]],
    predictions: dict[str, list[float]],
    thresholds: dict[str, float],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    possible = 0
    reused = 0
    for indices in _state_groups(rows):
        indices = [
            index
            for index in indices
            if bool(rows[index]["runtime_registered"])
        ]
        if not indices:
            continue
        sequences = [
            tuple(
                S3ActionTemplate.from_payload(value)
                for value in rows[index]["templates"]
            )
            for index in indices
        ]
        local = {
            name: [values[index] for index in indices]
            for name, values in predictions.items()
        }
        order = rank_s3_sequences(sequences, local, thresholds)
        if not order:
            order = rank_s3_sequences(
                sequences,
                local,
                thresholds,
                allow_risk_relaxation=True,
            )
        if not order:
            continue
        chosen = rows[indices[order[0]]]
        for trial in chosen["actual"]["trials"]:
            steps = [
                dict(row)
                for row in _runtime_reachable_steps(trial)
                if bool(row.get("executed"))
            ]
            for offset, step in enumerate(steps[:-1]):
                possible += 1
                no_progress = str(step["repair_outcome"]) in {
                    "hard_failure",
                    "accepted_noop",
                }
                cell_key = (
                    f"step{offset + 1}:agents{int(chosen['agent_count'])}"
                )
                cell = dict(calibration["cells"]).get(cell_key)
                if cell is None:
                    cell = dict(calibration["fallback"])[
                        f"step{offset + 1}"
                    ]
                cell = dict(cell)
                predicted_no_progress = (
                    local[f"step{offset + 1}_no_progress_probability"][order[0]]
                    >= float(cell["no_progress_threshold"])
                )
                predicted_reduction = float(
                    local[f"step{offset + 1}_conflict_reduction"][order[0]]
                )
                reduction_error = abs(
                    float(step["conflict_reduction"]) - predicted_reduction
                ) / max(1.0, abs(predicted_reduction))
                log_time_error = abs(
                    math.log1p(max(0.0, float(step["total_seconds"])))
                    - float(
                        local[f"step{offset + 1}_log_total_seconds"][
                            order[0]
                        ]
                    )
                )
                if (
                    not no_progress
                    and predicted_no_progress == no_progress
                    and reduction_error
                    <= float(cell["reduction_relative_error"]) + 1e-12
                    and log_time_error
                    <= float(cell["log_total_seconds_error"]) + 1e-12
                ):
                    reused += 1
    return {
        "continuation_possible_count": possible,
        "continuation_reused_count": reused,
        "continuation_reuse_fraction": reused / possible if possible else 0.0,
    }


def train_v3_s3_controller(
    *,
    sequence_features: str | Path,
    sequence_trials: str | Path,
    external_baselines: str | Path,
    output: str | Path,
    training_jobs: int = 1,
) -> dict[str, Any]:
    if int(training_jobs) <= 0:
        raise ValueError("v3-S3 training_jobs must be positive")
    feature_path = Path(sequence_features).resolve()
    trial_path = Path(sequence_trials).resolve()
    baseline_path = Path(external_baselines).resolve()
    output_root = Path(output).resolve()
    train, diagnostic = _sequence_rows(feature_path, trial_path)
    oof_cache: dict[
        tuple[str, tuple[int, ...]],
        tuple[dict[str, list[float]], list[dict[str, Any]], set[int]],
    ] = {}
    feature_selection = _choose_feature_schema(
        train,
        jobs=int(training_jobs),
        oof_cache=oof_cache,
    )
    feature_indices = tuple(map(int, feature_selection["selected_indices"]))
    feature_names = [V3_S3_FULL_FEATURE_NAMES[index] for index in feature_indices]
    family_reports = {}
    family_predictions = {}
    for family in MODEL_FAMILIES:
        report, predictions = _family_report(
            train,
            feature_indices,
            family,
            jobs=int(training_jobs),
            oof_cache=oof_cache,
        )
        family_reports[family] = report
        family_predictions[family] = predictions
    feature_selection["oof_cache_entry_count"] = len(oof_cache)
    feature_selection["selected_hgb_oof_reused"] = bool(
        family_reports["hist_gradient_boosting"]["oof_cache_hit"]
    )
    preferred = (
        "extra_trees"
        if _extra_trees_statistically_preferred(family_reports)
        else "hist_gradient_boosting"
    )
    source_fingerprint = _fingerprint(
        {
            "sequence_features_sha256": sha256_file(feature_path),
            "sequence_trials_sha256": sha256_file(trial_path),
            "external_baselines_sha256": sha256_file(baseline_path),
            "feature_schema_sha256": V3_S3_FEATURE_SCHEMA_SHA256,
            "selected_feature_names": feature_names,
            "model_parameters": {
                "hist_gradient_boosting": HGB_PARAMETERS,
                "extra_trees": EXTRA_TREES_PARAMETERS,
            },
            "training_objective_id": V3_S3_OBJECTIVE_ID,
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    exports = {}
    fitted = {}
    for family in MODEL_FAMILIES:
        models = _fit_models(
            train, feature_indices, family, jobs=int(training_jobs)
        )
        fitted[family] = models
        exports[family] = _export_family(
            models=models,
            family=family,
            rows=diagnostic,
            feature_indices=feature_indices,
            feature_names=feature_names,
            output=output_root / "model_candidates" / family,
            source_fingerprint=source_fingerprint,
        )
    diagnostic_state_ids = {str(row["state_id"]) for row in diagnostic}
    diagnostic_v2 = _baseline_metrics(
        baseline_path,
        "policy_validation",
        "v2-full",
        expected_state_ids=diagnostic_state_ids,
    )
    diagnostic_adaptive = _baseline_metrics(
        baseline_path,
        "policy_validation",
        "official_adaptive",
        expected_state_ids=diagnostic_state_ids,
    )
    family_diagnostics = {}
    for family in MODEL_FAMILIES:
        family_thresholds = dict(family_reports[family]["thresholds"])
        family_intervals = _prediction_intervals(
            train, family_predictions[family]
        )
        family_continuation = _continuation_calibration(
            train, family_predictions[family]
        )
        diagnostic_predictions = _predict_models(
            fitted[family], diagnostic, feature_indices
        )
        diagnostic_s3 = _selection_metrics(
            diagnostic, diagnostic_predictions, family_thresholds
        )
        continuation = _continuation_diagnostic(
            diagnostic,
            diagnostic_predictions,
            family_thresholds,
            family_continuation,
        )
        family_diagnostics[family] = {
            "thresholds": family_thresholds,
            "prediction_intervals": family_intervals,
            "continuation_calibration": family_continuation,
            "v3_s3": diagnostic_s3,
            **continuation,
            "pilot_checks": _pilot_gate(
                diagnostic_s3,
                diagnostic_v2,
                float(continuation["continuation_reuse_fraction"]),
            ),
        }
    selected_diagnostic = dict(family_diagnostics[preferred])
    thresholds = dict(selected_diagnostic["thresholds"])
    intervals = dict(selected_diagnostic["prediction_intervals"])
    continuation_calibration = dict(
        selected_diagnostic["continuation_calibration"]
    )
    diagnostic_s3 = dict(selected_diagnostic["v3_s3"])
    continuation_possible = int(selected_diagnostic["continuation_possible_count"])
    continuation_reused = int(selected_diagnostic["continuation_reused_count"])
    continuation_fraction = float(selected_diagnostic["continuation_reuse_fraction"])
    checks = dict(selected_diagnostic["pilot_checks"])
    report = {
        "schema": V3_S3_TRAINING_SCHEMA,
        "training_state_count": len({row["state_id"] for row in train}),
        "diagnostic_state_count": len({row["state_id"] for row in diagnostic}),
        "training_sequence_count": len(train),
        "diagnostic_sequence_count": len(diagnostic),
        "training_agent_counts": sorted({int(row["agent_count"]) for row in train}),
        "training_jobs": int(training_jobs),
        "training_objective_id": V3_S3_OBJECTIVE_ID,
        "formal_or_movingai_labels_seen": False,
        "feature_selection": feature_selection,
        "declared_feature_count": len(feature_names),
        "model_family_reports": family_reports,
        "model_family_diagnostics": family_diagnostics,
        "provisional_model_family": preferred,
        "model_family_selection_requires_native_audit": True,
        "thresholds": thresholds,
        "prediction_intervals": intervals,
        "continuation_calibration": continuation_calibration,
        "diagnostic": {
            "v3_s3": diagnostic_s3,
            "v2_full": diagnostic_v2,
            "official_adaptive": diagnostic_adaptive,
            "continuation_possible_count": continuation_possible,
            "continuation_reused_count": continuation_reused,
            "continuation_reuse_fraction": continuation_fraction,
        },
        "pilot_checks": checks,
        "pilot_passed_before_native_audit": all(checks.values()),
        "native_audit_completed": False,
        "decision": "awaiting_v3_s3_native_audit",
        "model_exports": exports,
    }
    report_path = output_root / "training_report.json"
    _write_json(report_path, report)
    selected_models = exports[preferred]["models"]
    manifest_models = {
        name: dict(row)
        for name, row in selected_models.items()
    }
    manifest = {
        "schema": V3_S3_BUNDLE_SCHEMA,
        "schema_version": 1,
        "feature_schema_id": V3_S3_FEATURE_SCHEMA_ID,
        "feature_schema_sha256": V3_S3_FEATURE_SCHEMA_SHA256,
        "training_objective_id": V3_S3_OBJECTIVE_ID,
        "profile": V3_S3_PROFILE,
        "feature_names": feature_names,
        "models": manifest_models,
        "model_family": preferred,
        "model_candidates": exports,
        "thresholds": thresholds,
        "prediction_intervals": intervals,
        "continuation_calibration": continuation_calibration,
        "source_fingerprint": source_fingerprint,
        "training_report": {
            "file": report_path.relative_to(output_root).as_posix(),
            "sha256": sha256_file(report_path),
        },
        "runtime_dependencies": [],
        "v2_runtime_call_count": 0,
        "adaptive_runtime_call_count": 0,
        "deployment_promoted": False,
        "native_audit_completed": False,
    }
    _write_json(output_root / "v3_s3_manifest.json", manifest)
    load_v3_s3_bundle(output_root)
    return {**report, "manifest": manifest}


def finalize_v3_s3_native_audit(
    *, controller_output: str | Path, benchmark_rows: int = 36
) -> dict[str, Any]:
    root = Path(controller_output).resolve()
    report_path = root / "training_report.json"
    manifest_path = root / "v3_s3_manifest.json"
    report = dict(_read_json(report_path))
    manifest = dict(_read_json(manifest_path))
    feature_names = list(map(str, manifest["feature_names"]))
    requested_benchmark_rows = int(benchmark_rows)
    if requested_benchmark_rows <= 0:
        raise ValueError("v3-S3 native benchmark_rows must be positive")
    if requested_benchmark_rows > 36:
        raise ValueError(
            "v3-S3 native benchmark_rows cannot exceed the registered "
            "36-sequence deployment schedule"
        )
    parity_probe_count = min(64, requested_benchmark_rows)
    parity_rows = [
        {
            "feature_profile": V3_S3_PROFILE,
            "feature_names": tuple(feature_names),
            "feature_values": tuple(
                0.0
                if probe == 0
                else float(((probe + 1) * (index + 3)) % 19 - 9) / 3.0
                for index, _name in enumerate(feature_names)
            ),
        }
        for probe in range(parity_probe_count)
    ]
    deployment_sequences = list(
        registered_runtime_sequences(
            "native-audit",
            (template.key for template in S3_ACTION_TEMPLATES),
        )
    )
    if len(deployment_sequences) != 36:
        raise RuntimeError(
            "v3-S3 registered deployment schedule must contain 36 sequences"
        )
    runtime_sequences = deployment_sequences[:requested_benchmark_rows]
    base_candidate = {
        "feature_profile": V3_S3_PROFILE,
        "feature_names": V3_S3_FULL_FEATURE_NAMES,
        "feature_values": tuple(0.0 for _ in V3_S3_FULL_FEATURE_NAMES),
    }
    row_started = time.perf_counter()
    runtime_rows = [
        sequence_feature_row(
            base_candidate,
            {name: 0.0 for name in S3_TEMPORAL_FEATURE_NAMES},
            templates,
            agent_count=600,
            feature_names=feature_names,
        )
        for templates in runtime_sequences
    ]
    row_construction_seconds = time.perf_counter() - row_started
    latencies = {}
    planner_seconds = {}
    parity = {}
    for family, export in dict(report["model_exports"]).items():
        family_deltas = {}
        models = {}
        for name, raw in dict(export["models"]).items():
            path = root / str(dict(raw)["file"])
            payload = dict(_read_json(path))
            portable = load_portable_scalar_model(payload)
            python_model: PortableScalarModel = replace(portable, native_predictor=None)
            python_values = python_model.predict(parity_rows)
            native_values = portable.predict(parity_rows)
            delta = max(
                (abs(float(left) - float(right)) for left, right in zip(python_values, native_values)),
                default=0.0,
            )
            if portable.inference_backend != "native-portable-tree" or delta > 1e-12:
                raise RuntimeError(f"v3-S3 native parity failed: {family}/{name}")
            family_deltas[name] = delta
            models[name] = portable
        measurements = []
        thresholds = dict(report["model_family_diagnostics"])[family]["thresholds"]
        for _repeat in range(3):
            started = time.perf_counter()
            predicted = {
                name: _normalize_prediction_values(
                    name, model.predict(runtime_rows)
                )
                for name, model in models.items()
            }
            for step in range(1, S3_HORIZON + 1):
                predicted[f"step{step}_total_seconds"] = [
                    max(1e-9, math.expm1(min(50.0, value)))
                    for value in predicted[f"step{step}_log_total_seconds"]
                ]
            predicted["sequence_total_seconds"] = [
                max(1e-9, math.expm1(min(50.0, value)))
                for value in predicted["sequence_log_total_seconds"]
            ]
            rank_s3_sequences(runtime_sequences, predicted, dict(thresholds))
            measurements.append(time.perf_counter() - started)
        inference_and_ranking = statistics.median(measurements)
        latencies[family] = inference_and_ranking / len(runtime_rows)
        planner_seconds[family] = row_construction_seconds + inference_and_ranking
        parity[family] = family_deltas
    preferred = str(report["provisional_model_family"])
    if preferred == "extra_trees" and latencies["extra_trees"] > 1.10 * latencies["hist_gradient_boosting"]:
        preferred = "hist_gradient_boosting"
    selected_diagnostic = dict(report["model_family_diagnostics"])[preferred]
    adjusted_s3 = _with_planner_overhead(
        dict(selected_diagnostic["v3_s3"]), planner_seconds[preferred]
    )
    report["thresholds"] = dict(selected_diagnostic["thresholds"])
    report["prediction_intervals"] = dict(
        selected_diagnostic["prediction_intervals"]
    )
    report["continuation_calibration"] = dict(
        selected_diagnostic["continuation_calibration"]
    )
    report["diagnostic"] = {
        "v3_s3": adjusted_s3,
        "v2_full": dict(report["diagnostic"])["v2_full"],
        "official_adaptive": dict(report["diagnostic"])["official_adaptive"],
        "continuation_possible_count": int(
            selected_diagnostic["continuation_possible_count"]
        ),
        "continuation_reused_count": int(
            selected_diagnostic["continuation_reused_count"]
        ),
        "continuation_reuse_fraction": float(
            selected_diagnostic["continuation_reuse_fraction"]
        ),
    }
    report["pilot_checks"] = {
        **_pilot_gate(
            adjusted_s3,
            dict(report["diagnostic"])["v2_full"],
            float(selected_diagnostic["continuation_reuse_fraction"]),
        ),
        "selection_overhead_at_most_105pct": float(
            adjusted_s3["mean_selection_seconds"]
        )
        <= 1.05
        * float(dict(report["diagnostic"])["v2_full"]["mean_selection_seconds"])
        + 1e-12,
    }
    export = dict(report["model_exports"])[preferred]
    manifest["models"] = {
        name: {
            **dict(raw),
            "inference_backend": "native-portable-tree",
        }
        for name, raw in dict(export["models"]).items()
    }
    manifest["model_family"] = preferred
    manifest["thresholds"] = dict(selected_diagnostic["thresholds"])
    manifest["prediction_intervals"] = dict(
        selected_diagnostic["prediction_intervals"]
    )
    manifest["continuation_calibration"] = dict(
        selected_diagnostic["continuation_calibration"]
    )
    manifest["native_audit_completed"] = True
    checks = {
        **dict(report["pilot_checks"]),
        "native_python_parity": max(
            (value for family in parity.values() for value in family.values()), default=0.0
        )
        <= 1e-12,
        "native_latency_rule": preferred != "extra_trees"
        or latencies["extra_trees"] <= 1.10 * latencies["hist_gradient_boosting"],
    }
    decision = (
        "v3_s3_mixed_load_pilot_passed"
        if all(checks.values())
        else "v3_s3_mixed_load_pilot_failed"
    )
    report.update(
        {
            "selected_model_family": preferred,
            "native_latency_seconds_per_sequence": latencies,
            "native_benchmark_sequence_count": len(runtime_sequences),
            "native_parity_probe_count": len(parity_rows),
            "sequence_row_construction_seconds": row_construction_seconds,
            "planner_seconds_per_new_state": planner_seconds,
            "native_parity": parity,
            "native_audit_completed": True,
            "pilot_checks": checks,
            "pilot_passed": all(checks.values()),
            "decision": decision,
        }
    )
    _write_json(report_path, report)
    manifest["training_report"]["sha256"] = sha256_file(report_path)
    _write_json(manifest_path, manifest)
    load_v3_s3_bundle(root)
    return {**report, "manifest": manifest}


__all__ = [
    "EXTRA_TREES_PARAMETERS",
    "HGB_PARAMETERS",
    "MODEL_FAMILIES",
    "V3_S3_TRAINING_SCHEMA",
    "finalize_v3_s3_native_audit",
    "train_v3_s3_controller",
]
