from __future__ import annotations

import collections
import itertools
import statistics
from pathlib import Path
from typing import Any

from experiments._common import producer_identity, sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_collection import _read_json, _read_jsonl, _write_json, _write_jsonl
from experiments.stride_scalepool_evaluation import _validate_external_label_audit
from experiments.stride_structpool_size_ablation import _best, _family_variant, _forbidden_hits
from lns2_selector.training.tree_utils import balanced_map_folds


CONFIG_SCHEMA = "lns2.stride.slotpool_registration.v1"
REPORT_SCHEMA = "lns2.stride.slotpool_offline_evaluation.v1"
STATE_SCHEMA = "lns2.stride.slotpool_offline_state.v1"
PAIR_SCHEMA = "lns2.stride.slotpool_stable_pair_summary.v1"
IMPLEMENTATION_ID = "stride-slotpool-v1"
FAMILY_VARIANTS = (
    "bottleneck_crossing",
    "conflict_component",
    "topology_boundary_articulation",
    "topology_boundary_low_degree",
    "spatiotemporal_hotspot",
    "path_overlap",
)
ALLOWED_SIZES = (8, 16, 24, 32)
BASE_FEATURE_NAMES = tuple(PROFILE_FEATURE_NAMES["realized_dynamic"])
STATE_FEATURE_NAMES = tuple(name for name in BASE_FEATURE_NAMES if name.startswith("state."))
DERIVED_FEATURE_NAMES = (
    *(f"slot.family={name}" for name in FAMILY_VARIANTS),
    *(f"slot.size={size}" for size in ALLOWED_SIZES),
    *(f"slot.family_size={family}:{size}" for family in FAMILY_VARIANTS for size in ALLOWED_SIZES),
    "slot.provenance_count",
    "slot.mixed_family",
    "slot.v2_anchor_jaccard",
    "slot.support_count_min",
    "slot.support_count_mean",
    "slot.support_count_max",
    "slot.support_ratio_min",
    "slot.support_ratio_mean",
    "slot.support_ratio_max",
    "slot.size_support_ratio_min",
    "slot.size_support_ratio_mean",
    "slot.size_support_ratio_max",
)
CANDIDATE_FEATURE_NAMES = BASE_FEATURE_NAMES + DERIVED_FEATURE_NAMES
PAIR_FEATURE_NAMES = tuple(f"delta:{name}" for name in CANDIDATE_FEATURE_NAMES) + tuple(
    f"shared:{name}" for name in STATE_FEATURE_NAMES
)


def _registered(project_root: Path, specification: dict[str, Any]) -> Path:
    path = (project_root / str(specification["path"])).resolve()
    if not path.is_file():
        raise ValueError(f"registered SlotPool input is missing: {path}")
    observed = sha256_file(path)
    if observed != str(specification["sha256"]):
        raise ValueError(
            f"registered SlotPool input changed: {path}: expected {specification['sha256']}, got {observed}"
        )
    return path


def validate_slotpool_config(
    config: dict[str, Any], *, project_root: Path | None = None
) -> None:
    if (
        config.get("schema") != CONFIG_SCHEMA
        or config.get("implementation_id") != IMPLEMENTATION_ID
        or config.get("scientific_status")
        != "preregistered_map_grouped_offline_candidate_budget_validation"
    ):
        raise ValueError("SlotPool registration identity changed")
    cohort = dict(config.get("cohort") or {})
    if cohort != {
        "state_count": 98,
        "candidate_count": 1368,
        "trial_count": 21888,
        "map_count": 16,
        "known_maze_long_tail_excluded": True,
        "development_oof_only": True,
        "fresh_map_data_read_before_acceptance": False,
    }:
        raise ValueError("SlotPool development cohort changed")
    features = dict(config.get("features") or {})
    if (
        features.get("base_schema") != "lns2.realized_features.v2"
        or int(features.get("base_dimension", -1)) != len(BASE_FEATURE_NAMES)
        or len(BASE_FEATURE_NAMES) != 124
        or int(features.get("derived_slot_dimension", -1)) != len(DERIVED_FEATURE_NAMES)
        or len(DERIVED_FEATURE_NAMES) != 46
        or int(features.get("candidate_dimension", -1)) != len(CANDIDATE_FEATURE_NAMES)
        or len(CANDIDATE_FEATURE_NAMES) != 170
        or int(features.get("shared_state_dimension", -1)) != len(STATE_FEATURE_NAMES)
        or len(STATE_FEATURE_NAMES) != 23
        or int(features.get("pair_dimension", -1)) != len(PAIR_FEATURE_NAMES)
        or len(PAIR_FEATURE_NAMES) != 193
        or tuple(features.get("family_variants") or ()) != FAMILY_VARIANTS
        or tuple(map(int, features.get("allowed_sizes") or ())) != ALLOWED_SIZES
    ):
        raise ValueError("SlotPool feature contract changed")
    labels = dict(config.get("stable_pair_labels") or {})
    if (
        list(labels.get("first_fixed_half") or ()) != list(range(8))
        or list(labels.get("second_fixed_half") or ()) != list(range(8, 16))
        or float(labels.get("minimum_absolute_gap_per_half", -1.0)) != 0.01
        or labels.get("require_same_order_in_both_halves") is not True
        or labels.get("mirrored_training_pairs") is not True
        or labels.get("uniform_total_weight_per_state") is not True
        or labels.get("within_state_weight") != "mean_absolute_fixed_half_gap"
    ):
        raise ValueError("SlotPool stable-pair label contract changed")
    model = dict(config.get("model") or {})
    if (
        model.get("class") != "sklearn.ensemble.HistGradientBoostingClassifier"
        or dict(model.get("fixed_parameters") or {})
        != {
            "early_stopping": False,
            "learning_rate": 0.05,
            "max_iter": 100,
            "min_samples_leaf": 20,
            "random_state": 20260809,
        }
        or list(model.get("parameter_grid") or ())
        != [
            {"max_leaf_nodes": 7, "l2_regularization": 0.1},
            {"max_leaf_nodes": 7, "l2_regularization": 1.0},
            {"max_leaf_nodes": 15, "l2_regularization": 0.1},
            {"max_leaf_nodes": 15, "l2_regularization": 1.0},
        ]
        or int(model.get("outer_map_folds", 0)) != 4
        or int(model.get("inner_map_folds", 0)) != 3
    ):
        raise ValueError("SlotPool model protocol changed")
    budget = dict(config.get("candidate_budget") or {})
    if budget != {
        "maximum_candidates": 6,
        "ranking": "mean_pairwise_win_probability_borda",
        "tie_break": "candidate_id_ascending",
        "allow_multiple_sizes_per_family": True,
        "exact_agent_set_deduplication": "reuse_stage2_grid",
        "additional_jaccard_filter": False,
        "additional_v2_anchor_filter": False,
    }:
        raise ValueError("SlotPool candidate-budget contract changed")
    gates = dict(config.get("offline_acceptance") or {})
    if gates != {
        "global_best_retention_minimum": 0.9,
        "mean_normalized_regret_maximum": 0.02,
        "maximum_map_mean_regret": 0.05,
        "maximum_topology_group_mean_regret": 0.05,
        "first_fixed_half_best_retention_minimum": 0.85,
        "second_fixed_half_best_retention_minimum": 0.85,
        "pairwise_accuracy_minimum": 0.6,
        "maximum_selected_candidates": 6,
        "raw_candidate_count_must_be_lower_than_full_grid": True,
    }:
        raise ValueError("SlotPool acceptance gates changed")
    boundary = dict(config.get("claim_boundary") or {})
    if boundary.get("current_step_labels_only") is not True or any(
        bool(boundary.get(name))
        for name in (
            "runtime_integration_before_acceptance",
            "formal_ttf_claim",
            "future_trajectory_read",
            "cost_to_go_read",
            "remaining_repair_rounds_read",
            "known_maze_result_used_for_parameters",
        )
    ):
        raise ValueError("SlotPool claim boundary changed")
    if project_root is not None:
        for specification in dict(config.get("inputs") or {}).values():
            _registered(project_root.resolve(), dict(specification))


def _slot_feature_values(row: dict[str, Any]) -> list[float]:
    families: set[str] = set()
    sizes: set[int] = set()
    family_sizes: set[tuple[str, int]] = set()
    supports: list[float] = []
    support_ratios: list[float] = []
    size_support_ratios: list[float] = []
    support_by_family = dict(row["structpool_support_count_by_family"])
    ratio_by_family = dict(row["structpool_support_ratio_by_family"])
    for raw_family in row["selection_families"]:
        family, size, _ = _family_variant(str(raw_family))
        families.add(family)
        sizes.add(size)
        family_sizes.add((family, size))
        support = float(support_by_family[raw_family])
        ratio = float(ratio_by_family[raw_family])
        supports.append(support)
        support_ratios.append(ratio)
        size_support_ratios.append(float(size) / max(1.0, support))
    if not families:
        raise ValueError(f"SlotPool candidate has no family provenance: {row.get('candidate_id')}")
    return [
        *(float(name in families) for name in FAMILY_VARIANTS),
        *(float(size in sizes) for size in ALLOWED_SIZES),
        *(float((family, size) in family_sizes) for family in FAMILY_VARIANTS for size in ALLOWED_SIZES),
        float(len(row["selection_families"])),
        float(len(families) > 1),
        float(row["v2_anchor_jaccard"]),
        min(supports),
        statistics.fmean(supports),
        max(supports),
        min(support_ratios),
        statistics.fmean(support_ratios),
        max(support_ratios),
        min(size_support_ratios),
        statistics.fmean(size_support_ratios),
        max(size_support_ratios),
    ]


def candidate_matrix(rows: list[dict[str, Any]]) -> Any:
    import numpy as np

    values = []
    for row in rows:
        features = dict(row["features"])
        if set(features) != set(BASE_FEATURE_NAMES):
            raise ValueError(f"SlotPool 124-dimensional feature schema changed: {row['candidate_id']}")
        values.append(
            [float(features[name]) for name in BASE_FEATURE_NAMES] + _slot_feature_values(row)
        )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(rows), len(CANDIDATE_FEATURE_NAMES)):
        raise RuntimeError("SlotPool candidate feature dimension changed")
    return result


def stable_pair_table(
    rows: list[dict[str, Any]], *, minimum_gap: float = 0.01
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        grouped[str(row["state_id"])].append((index, row))
    result: list[dict[str, Any]] = []
    for state_id, state_rows in sorted(grouped.items()):
        raw: list[dict[str, Any]] = []
        for (left, left_row), (right, right_row) in itertools.combinations(state_rows, 2):
            first_gap = float(left_row["first_fixed_half_mean"]) - float(
                right_row["first_fixed_half_mean"]
            )
            second_gap = float(left_row["second_fixed_half_mean"]) - float(
                right_row["second_fixed_half_mean"]
            )
            if (
                abs(first_gap) + 1e-12 < minimum_gap
                or abs(second_gap) + 1e-12 < minimum_gap
                or first_gap * second_gap <= 0.0
            ):
                continue
            raw.append(
                {
                    "state_id": state_id,
                    "map_id": str(left_row["map_id"]),
                    "left": left,
                    "right": right,
                    "label": int(first_gap > 0.0),
                    "raw_weight": 0.5 * (abs(first_gap) + abs(second_gap)),
                }
            )
        total = sum(float(row["raw_weight"]) for row in raw)
        if total <= 0.0:
            continue
        for row in raw:
            result.append({**row, "weight": float(row["raw_weight"]) / total})
    return result


def pair_matrix(candidate_values: Any, pairs: list[dict[str, Any]]) -> Any:
    import numpy as np

    shared_indices = [BASE_FEATURE_NAMES.index(name) for name in STATE_FEATURE_NAMES]
    values = np.empty((len(pairs), len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    for pair_index, row in enumerate(pairs):
        left = int(row["left"])
        right = int(row["right"])
        values[pair_index, : len(CANDIDATE_FEATURE_NAMES)] = (
            candidate_values[left] - candidate_values[right]
        )
        values[pair_index, len(CANDIDATE_FEATURE_NAMES) :] = 0.5 * (
            candidate_values[left, shared_indices] + candidate_values[right, shared_indices]
        )
    return values


def _fit_model(
    candidate_values: Any,
    pairs: list[dict[str, Any]],
    parameters: dict[str, Any],
) -> Any:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    if not pairs:
        raise ValueError("SlotPool training split has no stable pairs")
    forward = pair_matrix(candidate_values, pairs)
    mirror_pairs = [{**row, "left": row["right"], "right": row["left"]} for row in pairs]
    mirrored = pair_matrix(candidate_values, mirror_pairs)
    values = np.concatenate((forward, mirrored), axis=0)
    labels = np.asarray(
        [int(row["label"]) for row in pairs] + [1 - int(row["label"]) for row in pairs],
        dtype=np.int8,
    )
    weights = np.asarray(
        [float(row["weight"]) for row in pairs] * 2, dtype=np.float64
    )
    weights *= len(weights) / float(weights.sum())
    estimator = HistGradientBoostingClassifier(**parameters)
    estimator.fit(values, labels, sample_weight=weights)
    return estimator


def _score_state(
    estimator: Any,
    candidate_values: Any,
    indexed_rows: list[tuple[int, dict[str, Any]]],
    maximum_candidates: int,
) -> list[dict[str, Any]]:
    if not indexed_rows:
        raise ValueError("SlotPool cannot score an empty state")
    if len(indexed_rows) == 1:
        return [{"candidate_id": str(indexed_rows[0][1]["candidate_id"]), "score": 1.0}]
    pairs = [
        {"left": left[0], "right": right[0]}
        for left, right in itertools.combinations(indexed_rows, 2)
    ]
    probabilities = estimator.predict_proba(pair_matrix(candidate_values, pairs))[:, 1]
    totals = {index: 0.0 for index, _ in indexed_rows}
    counts = {index: 0 for index, _ in indexed_rows}
    for row, probability in zip(pairs, probabilities):
        left = int(row["left"])
        right = int(row["right"])
        totals[left] += float(probability)
        totals[right] += 1.0 - float(probability)
        counts[left] += 1
        counts[right] += 1
    ranked = sorted(
        (
            {
                "candidate_id": str(row["candidate_id"]),
                "score": totals[index] / float(counts[index]),
            }
            for index, row in indexed_rows
        ),
        key=lambda row: (-float(row["score"]), str(row["candidate_id"])),
    )
    return ranked[:maximum_candidates]


def _evaluate_states(
    *,
    estimator: Any,
    candidate_values: Any,
    rows: list[dict[str, Any]],
    state_ids: set[str],
    maximum_candidates: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        if str(row["state_id"]) in state_ids:
            grouped[str(row["state_id"])].append((index, row))
    result = []
    for state_id, indexed in sorted(grouped.items()):
        ranked = _score_state(estimator, candidate_values, indexed, maximum_candidates)
        selected_ids = {str(row["candidate_id"]) for row in ranked}
        all_rows = [row for _, row in indexed]
        selected = [row for row in all_rows if str(row["candidate_id"]) in selected_ids]
        best = _best(all_rows)
        selected_best = _best(selected)
        first_best = _best(all_rows, key="first_fixed_half_mean")
        second_best = _best(all_rows, key="second_fixed_half_mean")
        result.append(
            {
                "schema": STATE_SCHEMA,
                "state_id": state_id,
                "map_id": str(best["map_id"]),
                "layout_mode": str(best["layout_mode"]),
                "candidate_count": len(all_rows),
                "selected_candidate_count": len(selected),
                "selected_candidate_ids": [str(row["candidate_id"]) for row in ranked],
                "selected_scores": [float(row["score"]) for row in ranked],
                "global_best_candidate_id": str(best["candidate_id"]),
                "selected_best_candidate_id": str(selected_best["candidate_id"]),
                "global_best_retained": str(best["candidate_id"]) in selected_ids,
                "first_fixed_half_best_retained": str(first_best["candidate_id"]) in selected_ids,
                "second_fixed_half_best_retained": str(second_best["candidate_id"]) in selected_ids,
                "normalized_regret": float(best["seed_mean"]) - float(selected_best["seed_mean"]),
            }
        )
    if set(grouped) != state_ids:
        raise ValueError("SlotPool state evaluation coverage changed")
    return result


def _pair_accuracy(
    estimator: Any,
    candidate_values: Any,
    pairs: list[dict[str, Any]],
) -> float:
    if not pairs:
        raise ValueError("SlotPool validation split has no stable pairs")
    probabilities = estimator.predict_proba(pair_matrix(candidate_values, pairs))[:, 1]
    correct = [
        (float(probability) >= 0.5) == bool(row["label"])
        for probability, row in zip(probabilities, pairs)
    ]
    numerator = sum(float(row["weight"]) for row, ok in zip(pairs, correct) if ok)
    denominator = sum(float(row["weight"]) for row in pairs)
    return numerator / denominator


def summarize_slotpool_acceptance(
    *,
    rows: list[dict[str, Any]],
    pairwise_accuracy: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("SlotPool acceptance requires state rows")
    map_regrets: dict[str, list[float]] = collections.defaultdict(list)
    topology_regrets: dict[str, list[float]] = collections.defaultdict(list)
    for row in rows:
        map_regrets[str(row["map_id"])].append(float(row["normalized_regret"]))
        topology_regrets[str(row["layout_mode"])].append(float(row["normalized_regret"]))
    map_means = {name: statistics.fmean(values) for name, values in sorted(map_regrets.items())}
    topology_means = {
        name: statistics.fmean(values) for name, values in sorted(topology_regrets.items())
    }
    full_count = sum(int(row["candidate_count"]) for row in rows)
    selected_count = sum(int(row["selected_candidate_count"]) for row in rows)
    metrics = {
        "global_best_retention": statistics.fmean(bool(row["global_best_retained"]) for row in rows),
        "mean_normalized_regret": statistics.fmean(float(row["normalized_regret"]) for row in rows),
        "maximum_map_mean_regret": max(map_means.values()),
        "maximum_topology_group_mean_regret": max(topology_means.values()),
        "first_fixed_half_best_retention": statistics.fmean(
            bool(row["first_fixed_half_best_retained"]) for row in rows
        ),
        "second_fixed_half_best_retention": statistics.fmean(
            bool(row["second_fixed_half_best_retained"]) for row in rows
        ),
        "pairwise_accuracy": float(pairwise_accuracy),
        "maximum_selected_candidate_count": max(int(row["selected_candidate_count"]) for row in rows),
        "full_candidate_count": full_count,
        "selected_candidate_count": selected_count,
        "candidate_reduction": full_count - selected_count,
        "map_mean_regret": map_means,
        "topology_group_mean_regret": topology_means,
    }
    gates = dict(config["offline_acceptance"])
    checks = {
        "global_best_retention": metrics["global_best_retention"]
        >= float(gates["global_best_retention_minimum"]),
        "mean_normalized_regret": metrics["mean_normalized_regret"]
        <= float(gates["mean_normalized_regret_maximum"]),
        "map_group_regret": metrics["maximum_map_mean_regret"]
        <= float(gates["maximum_map_mean_regret"]),
        "topology_group_regret": metrics["maximum_topology_group_mean_regret"]
        <= float(gates["maximum_topology_group_mean_regret"]),
        "first_fixed_half_best_retention": metrics["first_fixed_half_best_retention"]
        >= float(gates["first_fixed_half_best_retention_minimum"]),
        "second_fixed_half_best_retention": metrics["second_fixed_half_best_retention"]
        >= float(gates["second_fixed_half_best_retention_minimum"]),
        "pairwise_accuracy": metrics["pairwise_accuracy"]
        >= float(gates["pairwise_accuracy_minimum"]),
        "candidate_budget": metrics["maximum_selected_candidate_count"]
        <= int(gates["maximum_selected_candidates"]),
        "candidate_count_reduced": selected_count < full_count,
    }
    return {"metrics": metrics, "checks": checks, "passed": all(checks.values())}


def _map_folds(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    maps = {}
    for row in rows:
        maps[str(row["map_id"])] = str(row["layout_mode"])
    return balanced_map_folds(
        [{"map_id": map_id, "layout_mode": layout} for map_id, layout in sorted(maps.items())],
        count=count,
    )


def _parameters(config: dict[str, Any], grid_row: dict[str, Any]) -> dict[str, Any]:
    return {**dict(config["model"]["fixed_parameters"]), **dict(grid_row)}


def _cross_validate_parameters(
    *,
    config: dict[str, Any],
    candidate_values: Any,
    rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    train_maps: set[str],
) -> tuple[int, list[dict[str, Any]]]:
    inner_rows = [row for row in rows if str(row["map_id"]) in train_maps]
    folds = _map_folds(inner_rows, int(config["model"]["inner_map_folds"]))
    summaries = []
    for parameter_index, grid_row in enumerate(config["model"]["parameter_grid"]):
        state_rows: list[dict[str, Any]] = []
        accuracies: list[float] = []
        for fold in folds:
            fit_maps = set(map(str, fold["train_maps"]))
            validation_maps = set(map(str, fold["validation_maps"]))
            fit_pairs = [row for row in pairs if str(row["map_id"]) in fit_maps]
            validation_pairs = [row for row in pairs if str(row["map_id"]) in validation_maps]
            estimator = _fit_model(candidate_values, fit_pairs, _parameters(config, grid_row))
            validation_states = {
                str(row["state_id"])
                for row in rows
                if str(row["map_id"]) in validation_maps
            }
            state_rows.extend(
                _evaluate_states(
                    estimator=estimator,
                    candidate_values=candidate_values,
                    rows=rows,
                    state_ids=validation_states,
                    maximum_candidates=int(config["candidate_budget"]["maximum_candidates"]),
                )
            )
            accuracies.append(_pair_accuracy(estimator, candidate_values, validation_pairs))
        metrics = summarize_slotpool_acceptance(
            rows=state_rows,
            pairwise_accuracy=statistics.fmean(accuracies),
            config=config,
        )["metrics"]
        summaries.append(
            {
                "parameter_index": parameter_index,
                "parameters": _parameters(config, grid_row),
                "metrics": metrics,
            }
        )
    selected = min(
        summaries,
        key=lambda row: (
            float(row["metrics"]["mean_normalized_regret"]),
            -float(row["metrics"]["global_best_retention"]),
            -min(
                float(row["metrics"]["first_fixed_half_best_retention"]),
                float(row["metrics"]["second_fixed_half_best_retention"]),
            ),
            -float(row["metrics"]["pairwise_accuracy"]),
            int(row["parameter_index"]),
        ),
    )
    return int(selected["parameter_index"]), summaries


def evaluate_slotpool(
    *, config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    project_root = config_path.parents[1]
    config = _read_json(config_path)
    validate_slotpool_config(config, project_root=project_root)
    inputs = {
        name: _registered(project_root, dict(specification))
        for name, specification in dict(config["inputs"]).items()
    }
    label_report = _read_json(inputs["label_report"])
    label_audit = _read_json(inputs["label_audit"])
    observed_artifacts = {
        "repair_trials_sha256": sha256_file(inputs["repair_trials"]),
        "candidate_aggregates_sha256": sha256_file(inputs["candidate_aggregates"]),
        "state_manifest_sha256": sha256_file(inputs["state_manifest"]),
        "state_artifact_tree_sha256": str(label_audit["source_artifacts"]["state_artifact_tree_sha256"]),
    }
    _validate_external_label_audit(
        label_report=label_report,
        audit_report=label_audit,
        observed_artifacts=observed_artifacts,
    )
    rows = _read_jsonl(inputs["candidate_aggregates"])
    if (
        len(rows) != int(config["cohort"]["candidate_count"])
        or len({str(row["state_id"]) for row in rows}) != int(config["cohort"]["state_count"])
        or len({str(row["map_id"]) for row in rows}) != int(config["cohort"]["map_count"])
    ):
        raise ValueError("SlotPool registered cohort counts changed")
    forbidden = sorted(_forbidden_hits(rows))
    if forbidden:
        raise ValueError(f"forbidden future/runtime fields entered SlotPool: {forbidden}")
    candidate_values = candidate_matrix(rows)
    pairs = stable_pair_table(
        rows,
        minimum_gap=float(config["stable_pair_labels"]["minimum_absolute_gap_per_half"]),
    )
    state_ids_with_pairs = {str(row["state_id"]) for row in pairs}
    if len(state_ids_with_pairs) < int(config["cohort"]["state_count"]) // 2:
        raise ValueError("too few SlotPool states have stable pair labels")
    outer_folds = _map_folds(rows, int(config["model"]["outer_map_folds"]))
    all_maps = {str(row["map_id"]) for row in rows}
    state_results: list[dict[str, Any]] = []
    fold_results = []
    weighted_correct = 0.0
    total_weight = 0.0
    for fold in outer_folds:
        train_maps = set(map(str, fold["train_maps"]))
        validation_maps = set(map(str, fold["validation_maps"]))
        if train_maps & validation_maps or train_maps | validation_maps != all_maps:
            raise RuntimeError("SlotPool outer fold map leakage detected")
        parameter_index, inner = _cross_validate_parameters(
            config=config,
            candidate_values=candidate_values,
            rows=rows,
            pairs=pairs,
            train_maps=train_maps,
        )
        train_pairs = [row for row in pairs if str(row["map_id"]) in train_maps]
        validation_pairs = [row for row in pairs if str(row["map_id"]) in validation_maps]
        estimator = _fit_model(
            candidate_values,
            train_pairs,
            _parameters(config, config["model"]["parameter_grid"][parameter_index]),
        )
        validation_states = {
            str(row["state_id"])
            for row in rows
            if str(row["map_id"]) in validation_maps
        }
        fold_states = _evaluate_states(
            estimator=estimator,
            candidate_values=candidate_values,
            rows=rows,
            state_ids=validation_states,
            maximum_candidates=int(config["candidate_budget"]["maximum_candidates"]),
        )
        for row in fold_states:
            row["outer_fold"] = int(fold["fold"])
            row["selected_parameter_index"] = parameter_index
        state_results.extend(fold_states)
        probabilities = estimator.predict_proba(pair_matrix(candidate_values, validation_pairs))[:, 1]
        correct_weight = sum(
            float(row["weight"])
            for probability, row in zip(probabilities, validation_pairs)
            if (float(probability) >= 0.5) == bool(row["label"])
        )
        fold_weight = sum(float(row["weight"]) for row in validation_pairs)
        weighted_correct += correct_weight
        total_weight += fold_weight
        fold_results.append(
            {
                "outer_fold": int(fold["fold"]),
                "train_maps": sorted(train_maps),
                "validation_maps": sorted(validation_maps),
                "train_pair_count": len(train_pairs),
                "validation_pair_count": len(validation_pairs),
                "validation_state_count": len(fold_states),
                "selected_parameter_index": parameter_index,
                "selected_parameters": _parameters(
                    config, config["model"]["parameter_grid"][parameter_index]
                ),
                "inner_parameter_summaries": inner,
                "pairwise_accuracy": correct_weight / fold_weight,
            }
        )
    if len(state_results) != int(config["cohort"]["state_count"]):
        raise RuntimeError("SlotPool OOF state coverage changed")
    acceptance = summarize_slotpool_acceptance(
        rows=state_results,
        pairwise_accuracy=weighted_correct / total_weight,
        config=config,
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "state_evaluation.jsonl"
    pairs_path = output / "stable_pair_summary.jsonl"
    folds_path = output / "fold_diagnostics.jsonl"
    _write_jsonl(states_path, sorted(state_results, key=lambda row: str(row["state_id"])))
    state_pair_counts = collections.Counter(str(row["state_id"]) for row in pairs)
    _write_jsonl(
        pairs_path,
        [
            {
                "schema": PAIR_SCHEMA,
                "state_id": state_id,
                "stable_pair_count": count,
                "total_state_weight": sum(
                    float(row["weight"]) for row in pairs if str(row["state_id"]) == state_id
                ),
            }
            for state_id, count in sorted(state_pair_counts.items())
        ],
    )
    _write_jsonl(folds_path, fold_results)
    report = {
        "schema": REPORT_SCHEMA,
        "scientific_status": (
            "offline_candidate_quality_passed_pending_fresh_map_confirmation"
            if acceptance["passed"]
            else "offline_candidate_quality_failed_runtime_hard_stop"
        ),
        "implementation_id": IMPLEMENTATION_ID,
        "candidate_feature_dimension": len(CANDIDATE_FEATURE_NAMES),
        "pair_feature_dimension": len(PAIR_FEATURE_NAMES),
        "state_count": len(state_results),
        "candidate_count": len(rows),
        "stable_pair_count": len(pairs),
        "states_with_stable_pairs": len(state_ids_with_pairs),
        "outer_fold_count": len(outer_folds),
        "map_disjoint_oof": True,
        "known_maze_long_tail_included": False,
        "fresh_map_data_read": False,
        "forbidden_fields": forbidden,
        "acceptance": acceptance,
        "runtime_integration_allowed": bool(acceptance["passed"]),
        "formal_ttf_claim": False,
        "next_decision": (
            "register_fresh_unseen_map_confirmation"
            if acceptance["passed"]
            else "stop_slotpool_runtime_and_analyze_failure"
        ),
        "producer": producer_identity(
            project_root=project_root,
            source_files=(
                "experiments/stride_slotpool.py",
                "experiments/stride_structpool_size_ablation.py",
                "lns2_selector/training/tree_utils.py",
            ),
            native_required=False,
            package_names=("numpy", "scikit-learn"),
        ),
        "inputs": {
            name: sha256_file(path) for name, path in sorted(inputs.items())
        },
        "artifacts": {
            "state_evaluation_sha256": sha256_file(states_path),
            "stable_pair_summary_sha256": sha256_file(pairs_path),
            "fold_diagnostics_sha256": sha256_file(folds_path),
        },
    }
    _write_json(output / "slotpool_evaluation_report.json", report)
    return report


__all__ = [
    "CANDIDATE_FEATURE_NAMES",
    "DERIVED_FEATURE_NAMES",
    "PAIR_FEATURE_NAMES",
    "candidate_matrix",
    "evaluate_slotpool",
    "stable_pair_table",
    "summarize_slotpool_acceptance",
    "validate_slotpool_config",
]
