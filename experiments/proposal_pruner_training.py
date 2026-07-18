from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments._common import sha256_file as _sha256
from experiments.closed_loop_confirmation import (
    load_frozen_policy_bundle,
    score_online_candidates,
)
from experiments.compact_controller_model import (
    compact_runtime_model,
    export_controller_bundle,
)
from experiments.context_audit import PairwiseModel, _pair_vector
from experiments.feature_schema_v2 import (
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    PROFILE_FEATURE_NAMES,
    PROPOSAL_FAMILIES,
    canonicalize_features,
    validate_redundancies,
)
from experiments.realized_neighborhood_ranking_audit import effectiveness_dominates
from experiments.repair_collection import _read_json, _read_jsonl, _write_json


TRAINING_SCHEMA = "lns2.proposal_pruner_training.v2"
MODEL_PARAMETERS = {
    "learning_rate": 0.05,
    "max_iter": 100,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 20,
    "l2_regularization": 0.1,
    "random_state": 20260714,
    "early_stopping": False,
}
DEFAULT_THRESHOLDS = tuple(round(0.50 + index * 0.01, 2) for index in range(50))


def _canonical_row(row: dict[str, Any]) -> dict[str, Any]:
    features = dict(row.get("features", {}))
    return {
        **row,
        "features": {
            profile: canonicalize_features(dict(features.get(profile, {})), profile)
            for profile in PROFILE_FEATURE_NAMES
        },
    }


def _grouped(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        result[str(row["state_id"])].append(row)
    for candidates in result.values():
        candidates.sort(key=lambda value: str(value["candidate_key"]))
    return dict(result)


def _family_indices(candidates: list[dict[str, Any]]) -> dict[str, list[int]]:
    result = {family: [] for family in PROPOSAL_FAMILIES}
    for index, candidate in enumerate(candidates):
        for family in map(str, candidate.get("selection_families", [])):
            if family in result:
                result[family].append(index)
    return {family: sorted(set(indices)) for family, indices in result.items()}


@dataclass(frozen=True)
class FamilyPair:
    state_id: str
    map_id: str
    layout: str
    family: str
    left: dict[str, Any]
    right: dict[str, Any]
    label: int


def _dominance_family_pairs(rows: list[dict[str, Any]]) -> tuple[list[FamilyPair], dict[str, int]]:
    pairs = []
    diagnostics: collections.Counter[str] = collections.Counter()
    for state_id, candidates in sorted(_grouped(rows).items()):
        family_indices = _family_indices(candidates)
        for family in PROPOSAL_FAMILIES:
            indices = family_indices[family]
            if len(indices) != 2:
                diagnostics[f"family_cardinality={len(indices)}"] += 1
                continue
            left, right = (candidates[index] for index in indices)
            if effectiveness_dominates(left["outcome"], right["outcome"]):
                label = 1
            elif effectiveness_dominates(right["outcome"], left["outcome"]):
                label = 0
            else:
                diagnostics["indistinguishable_or_non_dominating"] += 1
                continue
            pairs.append(
                FamilyPair(
                    state_id=state_id,
                    map_id=str(left["map_id"]),
                    layout=str(left.get("layout_mode", "unknown")),
                    family=family,
                    left=left,
                    right=right,
                    label=label,
                )
            )
            diagnostics["labeled"] += 1
    return pairs, dict(diagnostics)


def _balanced_pair_weights(pairs: list[FamilyPair]) -> list[float]:
    layouts = sorted({pair.layout for pair in pairs})
    maps_by_layout: dict[str, set[str]] = collections.defaultdict(set)
    states_by_map: dict[str, set[str]] = collections.defaultdict(set)
    pairs_by_state: collections.Counter[str] = collections.Counter()
    for pair in pairs:
        maps_by_layout[pair.layout].add(pair.map_id)
        states_by_map[pair.map_id].add(pair.state_id)
        pairs_by_state[pair.state_id] += 1
    raw = [
        1.0
        / (
            len(layouts)
            * len(maps_by_layout[pair.layout])
            * len(states_by_map[pair.map_id])
            * pairs_by_state[pair.state_id]
        )
        for pair in pairs
    ]
    scale = len(pairs) / math.fsum(raw)
    return [value * scale for value in raw]


def train_pruner_model(
    rows: list[dict[str, Any]],
    *,
    model_parameters: dict[str, Any] | None = None,
) -> tuple[PairwiseModel, dict[str, Any]]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    pairs, pair_diagnostics = _dominance_family_pairs(rows)
    if not pairs:
        raise ValueError("no stable within-family dominance labels are available")
    pair_weights = _balanced_pair_weights(pairs)
    names = list(PROFILE_FEATURE_NAMES["proposal_dynamic"])
    examples = []
    labels = []
    weights = []
    for pair, pair_weight in zip(pairs, pair_weights):
        examples.append(_pair_vector(pair.left, pair.right, "proposal_dynamic", names))
        labels.append(pair.label)
        weights.append(pair_weight / 2.0)
        examples.append(_pair_vector(pair.right, pair.left, "proposal_dynamic", names))
        labels.append(1 - pair.label)
        weights.append(pair_weight / 2.0)
    parameters = dict(MODEL_PARAMETERS)
    parameters.update(model_parameters or {})
    parameters["random_state"] = MODEL_PARAMETERS["random_state"]
    parameters["early_stopping"] = False
    estimator = HistGradientBoostingClassifier(**parameters)
    estimator.fit(
        np.asarray(examples, dtype=float),
        np.asarray(labels, dtype=int),
        sample_weight=np.asarray(weights, dtype=float),
    )
    return PairwiseModel("proposal_dynamic", names, estimator), {
        "state_count": len({pair.state_id for pair in pairs}),
        "map_count": len({pair.map_id for pair in pairs}),
        "layout_count": len({pair.layout for pair in pairs}),
        "labeled_family_pair_count": len(pairs),
        "training_example_count": len(examples),
        "pair_diagnostics": pair_diagnostics,
        "model_parameters": parameters,
        "feature_count": len(names),
        "pairwise_input_dimension": len(examples[0]),
        "balanced_by": ["layout", "map", "state", "candidate_family"],
    }


def balanced_map_folds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    map_layout: dict[str, str] = {}
    for row in rows:
        map_id = str(row["map_id"])
        layout = str(row.get("layout_mode", "unknown"))
        if map_layout.setdefault(map_id, layout) != layout:
            raise ValueError(f"map {map_id} has multiple layout labels")
    by_layout: dict[str, list[str]] = collections.defaultdict(list)
    for map_id, layout in map_layout.items():
        by_layout[layout].append(map_id)
    for maps in by_layout.values():
        maps.sort()
    counts = set(map(len, by_layout.values()))
    if counts != {4}:
        raise ValueError(
            "proposal pruner requires four maps per layout for four balanced folds"
        )
    all_maps = set(map_layout)
    folds = []
    for fold in range(4):
        validation_maps = sorted(maps[fold] for maps in by_layout.values())
        folds.append(
            {
                "fold": fold,
                "train_maps": sorted(all_maps - set(validation_maps)),
                "validation_maps": validation_maps,
            }
        )
    return folds


def _predict_positive(model: Any, vectors: list[list[float]]) -> list[float]:
    predict = getattr(model, "predict_positive", None)
    if callable(predict):
        return list(map(float, predict(vectors)))
    import numpy as np

    return list(
        map(
            float,
            model.estimator.predict_proba(np.asarray(vectors, dtype=float))[:, 1],
        )
    )


def _pair_probability(left: dict[str, Any], right: dict[str, Any], model: Any) -> float:
    pair_vector = getattr(model, "pair_vector", None)
    if callable(pair_vector):
        forward_vector = pair_vector(left, right)
        reverse_vector = pair_vector(right, left)
    else:
        forward_vector = _pair_vector(
            left, right, model.profile, list(model.feature_names)
        )
        reverse_vector = _pair_vector(
            right, left, model.profile, list(model.feature_names)
        )
    forward, reverse = _predict_positive(model, [forward_vector, reverse_vector])
    return (forward + (1.0 - reverse)) / 2.0


def family_probabilities(
    rows: list[dict[str, Any]], model: Any
) -> dict[tuple[str, str], float]:
    result = {}
    for state_id, candidates in sorted(_grouped(rows).items()):
        for family, indices in _family_indices(candidates).items():
            if len(indices) == 2:
                result[(state_id, family)] = _pair_probability(
                    candidates[indices[0]], candidates[indices[1]], model
                )
    return result


def out_of_fold_probabilities(
    rows: list[dict[str, Any]], folds: list[dict[str, Any]]
) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]]]:
    predictions = {}
    reports = []
    for fold in folds:
        train_maps = set(map(str, fold["train_maps"]))
        validation_maps = set(map(str, fold["validation_maps"]))
        train = [row for row in rows if str(row["map_id"]) in train_maps]
        validation = [
            row for row in rows if str(row["map_id"]) in validation_maps
        ]
        model, report = train_pruner_model(train)
        fold_predictions = family_probabilities(validation, model)
        overlap = set(predictions) & set(fold_predictions)
        if overlap:
            raise ValueError("OOF predictions contain duplicate state-family keys")
        predictions.update(fold_predictions)
        reports.append({**fold, **report, "prediction_count": len(fold_predictions)})
    return predictions, reports


def _probability_matrix(candidates: list[dict[str, Any]], model: Any) -> list[list[float]]:
    count = len(candidates)
    matrix = [[0.5] * count for _ in range(count)]
    pairs = []
    forward_vectors = []
    reverse_vectors = []
    pair_vector = getattr(model, "pair_vector", None)
    for left in range(count):
        for right in range(left + 1, count):
            if callable(pair_vector):
                forward_vectors.append(pair_vector(candidates[left], candidates[right]))
                reverse_vectors.append(pair_vector(candidates[right], candidates[left]))
            else:
                forward_vectors.append(
                    _pair_vector(
                        candidates[left],
                        candidates[right],
                        model.profile,
                        list(model.feature_names),
                    )
                )
                reverse_vectors.append(
                    _pair_vector(
                        candidates[right],
                        candidates[left],
                        model.profile,
                        list(model.feature_names),
                    )
                )
            pairs.append((left, right))
    if pairs:
        forward = _predict_positive(model, forward_vectors)
        reverse = _predict_positive(model, reverse_vectors)
        for (left, right), first, second in zip(pairs, forward, reverse):
            probability = (first + (1.0 - second)) / 2.0
            matrix[left][right] = probability
            matrix[right][left] = 1.0 - probability
    return matrix


def _select_from_matrix(
    candidates: list[dict[str, Any]], matrix: list[list[float]], retained: list[int]
) -> int:
    scores = {
        index: math.fsum(matrix[index][other] for other in retained if other != index)
        for index in retained
    }
    return min(
        retained,
        key=lambda index: (
            -round(scores[index], 12),
            str(candidates[index]["candidate_key"]),
        ),
    )


@dataclass
class OfflineState:
    state_id: str
    map_id: str
    candidates: list[dict[str, Any]]
    family_indices: dict[str, list[int]]
    family_probabilities: dict[str, float]
    main_matrix: list[list[float]]
    full_winner: int


def _offline_states(
    rows: list[dict[str, Any]],
    probabilities: dict[tuple[str, str], float],
    main_model: Any,
) -> list[OfflineState]:
    states = []
    for state_id, candidates in sorted(_grouped(rows).items()):
        matrix = _probability_matrix(candidates, main_model)
        retained = list(range(len(candidates)))
        states.append(
            OfflineState(
                state_id=state_id,
                map_id=str(candidates[0]["map_id"]),
                candidates=candidates,
                family_indices=_family_indices(candidates),
                family_probabilities={
                    family: probabilities[(state_id, family)]
                    for family, indices in _family_indices(candidates).items()
                    if len(indices) == 2 and (state_id, family) in probabilities
                },
                main_matrix=matrix,
                full_winner=_select_from_matrix(candidates, matrix, retained),
            )
        )
    return states


def _retained_indices(state: OfflineState, threshold: float) -> tuple[list[int], bool]:
    retained: set[int] = set()
    for family in PROPOSAL_FAMILIES:
        indices = state.family_indices.get(family, [])
        if len(indices) == 1:
            retained.add(indices[0])
        elif len(indices) == 2 and family in state.family_probabilities:
            probability = state.family_probabilities[family]
            if max(probability, 1.0 - probability) >= threshold:
                retained.add(indices[0] if probability >= 0.5 else indices[1])
            else:
                retained.update(indices)
        else:
            return list(range(len(state.candidates))), True
    return sorted(retained), False


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1)
    return ordered[max(0, index)]


def _map_bootstrap_upper(
    values: list[tuple[str, float]], samples: int, seed: int
) -> float:
    by_map: dict[str, list[float]] = collections.defaultdict(list)
    for map_id, value in values:
        by_map[map_id].append(float(value))
    map_means = [statistics.fmean(numbers) for numbers in by_map.values()]
    if not map_means:
        return 0.0
    generator = random.Random(seed)
    estimates = [
        statistics.fmean(generator.choice(map_means) for _ in map_means)
        for _ in range(samples)
    ]
    return _quantile(estimates, 0.95)


def evaluate_threshold(
    states: list[OfflineState],
    threshold: float,
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    before_count = 0
    after_count = 0
    winner_retained = 0
    fallback_count = 0
    success_delta = []
    conflict_regret = []
    auc_regret = []
    for state in states:
        retained, fallback = _retained_indices(state, threshold)
        fallback_count += int(fallback)
        before_count += len(state.candidates)
        after_count += len(retained)
        winner_retained += int(state.full_winner in retained)
        selected = _select_from_matrix(state.candidates, state.main_matrix, retained)
        reference_outcome = state.candidates[state.full_winner]["outcome"]
        selected_outcome = state.candidates[selected]["outcome"]
        success_delta.append(
            float(selected_outcome["solved_rate"])
            - float(reference_outcome["solved_rate"])
        )
        conflict_regret.append(
            (
                state.map_id,
                (
                    float(selected_outcome["conflicts_after"])
                    - float(reference_outcome["conflicts_after"])
                )
                / max(abs(float(reference_outcome["conflicts_after"])), 1.0),
            )
        )
        auc_regret.append(
            (
                state.map_id,
                (
                    float(selected_outcome["conflict_auc"])
                    - float(reference_outcome["conflict_auc"])
                )
                / max(abs(float(reference_outcome["conflict_auc"])), 1.0),
            )
        )
    reduction = 1.0 - after_count / before_count if before_count else 0.0
    report = {
        "threshold": float(threshold),
        "state_count": len(states),
        "candidate_count_before": before_count,
        "candidate_count_after": after_count,
        "candidate_reduction_fraction": reduction,
        "full_ranker_winner_retention": winner_retained / len(states) if states else 0.0,
        "mean_solved_rate_delta": statistics.fmean(success_delta) if success_delta else 0.0,
        "conflict_regret_mean": statistics.fmean(value for _, value in conflict_regret),
        "conflict_regret_one_sided_95_upper": _map_bootstrap_upper(
            conflict_regret, bootstrap_samples, 20260714
        ),
        "auc_regret_mean": statistics.fmean(value for _, value in auc_regret),
        "auc_regret_one_sided_95_upper": _map_bootstrap_upper(
            auc_regret, bootstrap_samples, 20260715
        ),
        "fallback_state_count": fallback_count,
    }
    report["passed"] = (
        report["full_ranker_winner_retention"] >= 0.99
        and report["mean_solved_rate_delta"] >= -1e-12
        and report["conflict_regret_one_sided_95_upper"] <= 0.01
        and report["auc_regret_one_sided_95_upper"] <= 0.01
        and report["candidate_reduction_fraction"] >= 0.15
    )
    return report


def calibrate_threshold(
    rows: list[dict[str, Any]],
    probabilities: dict[tuple[str, str], float],
    main_model: Any,
    thresholds: Iterable[float],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    states = _offline_states(rows, probabilities, main_model)
    reports = [
        evaluate_threshold(states, threshold, bootstrap_samples=bootstrap_samples)
        for threshold in thresholds
    ]
    passing = [report for report in reports if bool(report["passed"])]
    selected = (
        max(
            passing,
            key=lambda report: (
                float(report["candidate_reduction_fraction"]),
                float(report["threshold"]),
            ),
        )
        if passing
        else None
    )
    return {
        "threshold_grid": reports,
        "selected": selected,
        "passed": selected is not None,
    }


def _portable_payload(model: PairwiseModel, source_fingerprint: str) -> dict[str, Any]:
    estimator = model.estimator
    if list(map(int, estimator.classes_)) != [0, 1]:
        raise ValueError("proposal pruner exporter requires a binary classifier")
    trees = []
    for stage in estimator._predictors:
        if len(stage) != 1:
            raise ValueError("proposal pruner exporter requires one tree per stage")
        nodes = []
        for node in stage[0].nodes:
            if bool(node["is_categorical"]):
                raise ValueError("proposal pruner does not support categorical nodes")
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
    return {
        "schema": "lns2.portable_pairwise_hist_gbdt.v1",
        "schema_version": 1,
        "profile": "proposal_dynamic",
        "source_model_sha256": source_fingerprint,
        "feature_names": list(model.feature_names),
        "baseline": float(estimator._baseline_prediction[0, 0]),
        "trees": trees,
    }


def _feature_ranges(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    names = PROFILE_FEATURE_NAMES["proposal_dynamic"]
    values = {name: [] for name in names}
    for row in rows:
        features = row["features"]["proposal_dynamic"]
        for name in names:
            values[name].append(float(features[name]))
    return {name: (min(numbers), max(numbers)) for name, numbers in values.items()}


def main_ranker_equivalence(
    development_rows: list[dict[str, Any]], bundle: Any
) -> dict[str, Any]:
    grouped = _grouped(development_rows)
    profiles = {}
    for profile in ("proposal_dynamic", "realized_dynamic"):
        compact = compact_runtime_model(bundle.models[profile])
        mismatches = 0
        maximum_score_delta = 0.0
        for candidates in grouped.values():
            old_index, old_scores, _ = score_online_candidates(
                candidates, bundle.models[profile]
            )
            new_index, new_scores, _ = score_online_candidates(candidates, compact)
            mismatches += old_index != new_index
            maximum_score_delta = max(
                maximum_score_delta,
                max(
                    (abs(left - right) for left, right in zip(old_scores, new_scores)),
                    default=0.0,
                ),
            )
        profiles[profile] = {
            "state_count": len(grouped),
            "selection_mismatch_count": mismatches,
            "maximum_score_delta": maximum_score_delta,
            "pairwise_input_dimension": len(compact.input_features),
            "used_feature_count": len(compact.base_feature_names),
            "passed": mismatches == 0 and maximum_score_delta <= 1e-12,
        }
    return {
        "profiles": profiles,
        "passed": all(bool(value["passed"]) for value in profiles.values()),
    }


def run_proposal_pruner_training(
    config_path: str | Path, output: str | Path
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    config_file = Path(config_path).resolve()
    config = _read_json(config_file)
    if int(config.get("schema_version", -1)) != 2:
        raise ValueError("unsupported proposal pruner training config")

    def resolve(value: str) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (project_root / path).resolve()

    train_path = resolve(str(config["training_index"]))
    validation_path = resolve(str(config["validation_index"]))
    collection_config_path = resolve(str(config["collection_config"]))
    source_bundle_path = resolve(str(config["source_bundle"]))
    development_path = resolve(str(config["development_index"]))
    output_root = Path(output).resolve()

    def registered_path(path: Path) -> str:
        try:
            return path.relative_to(project_root).as_posix()
        except ValueError:
            return f"external/{path.name}"
    raw_training = [
        row
        for row in _read_jsonl(train_path)
        if str(row.get("split")) == "policy_train"
    ]
    raw_validation = _read_jsonl(validation_path)
    if any(str(row.get("split")) != "policy_validation" for row in raw_validation):
        raise ValueError("validation index contains non-policy_validation rows")
    train_maps = {str(row["map_id"]) for row in raw_training}
    validation_maps = {str(row["map_id"]) for row in raw_validation}
    if train_maps & validation_maps:
        raise ValueError("policy train and validation maps overlap")
    training = [_canonical_row(row) for row in raw_training]
    validation = [_canonical_row(row) for row in raw_validation]
    development_rows = _read_jsonl(development_path)
    redundancy = validate_redundancies(
        itertools.chain(raw_training, raw_validation, development_rows),
        "realized_dynamic",
    )
    folds = balanced_map_folds(training)

    collection_config = _read_json(collection_config_path)
    frozen_root = resolve(str(collection_config["frozen_models"]))
    v1_bundle = load_frozen_policy_bundle(
        frozen_root, dict(collection_config["model_registration"])
    )
    main_model = compact_runtime_model(v1_bundle.models["realized_dynamic"])
    equivalence = main_ranker_equivalence(development_rows, v1_bundle)
    bootstrap_samples = int(config.get("bootstrap_samples", 5000))
    thresholds = tuple(map(float, config.get("threshold_grid", DEFAULT_THRESHOLDS)))

    oof_probabilities, fold_reports = out_of_fold_probabilities(training, folds)
    dedicated_calibration = calibrate_threshold(
        training,
        oof_probabilities,
        main_model,
        thresholds,
        bootstrap_samples=bootstrap_samples,
    )
    dedicated_model, full_training_report = train_pruner_model(training)
    dedicated_validation = None
    if dedicated_calibration["selected"] is not None:
        locked_threshold = float(dedicated_calibration["selected"]["threshold"])
        dedicated_validation = evaluate_threshold(
            _offline_states(
                validation,
                family_probabilities(validation, dedicated_model),
                main_model,
            ),
            locked_threshold,
            bootstrap_samples=bootstrap_samples,
        )

    selected_source = None
    selected_model: Any | None = None
    selected_payload = None
    selected_threshold = None
    selected_validation = None
    legacy_calibration = None
    legacy_validation = None
    if dedicated_validation is not None and bool(dedicated_validation["passed"]):
        selected_source = "proposal_pruner_v2_policy_train"
        selected_model = dedicated_model
        selected_threshold = float(dedicated_validation["threshold"])
        selected_validation = dedicated_validation
        source_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "training_index_sha256": _sha256(train_path),
                    "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
                    "parameters": MODEL_PARAMETERS,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        selected_payload = _portable_payload(dedicated_model, source_fingerprint)
    else:
        legacy_model = compact_runtime_model(v1_bundle.models["proposal_dynamic"])
        legacy_oof = family_probabilities(training, legacy_model)
        legacy_calibration = calibrate_threshold(
            training,
            legacy_oof,
            main_model,
            thresholds,
            bootstrap_samples=bootstrap_samples,
        )
        if legacy_calibration["selected"] is not None:
            threshold = float(legacy_calibration["selected"]["threshold"])
            legacy_validation = evaluate_threshold(
                _offline_states(
                    validation,
                    family_probabilities(validation, legacy_model),
                    main_model,
                ),
                threshold,
                bootstrap_samples=bootstrap_samples,
            )
            if bool(legacy_validation["passed"]):
                selected_source = "frozen_proposal_dynamic_fallback"
                selected_model = legacy_model
                selected_threshold = threshold
                selected_validation = legacy_validation
                selected_payload = _read_json(
                    source_bundle_path / "pairwise__proposal_dynamic.json"
                )

    exact_acceleration_passed = bool(equivalence["passed"] and redundancy["passed"])
    offline_pruner_passed = selected_payload is not None
    promotion_report = {
        "exact_acceleration_passed": exact_acceleration_passed,
        "pruner_offline_validation_passed": offline_pruner_passed,
        # End-to-end quick/formal gates are deliberately separate. Until those
        # reports pass, an eligible pruner is available only via --controller.
        "pruning_promotion_passed": False,
        "main_ranker_equivalence": equivalence,
        "redundancy_audit": redundancy,
        "selected_pruner_source": selected_source,
        "selected_threshold": selected_threshold,
        "selected_validation": selected_validation,
        "remaining_promotion_gates": [
            "quick paired controller run",
            "720 formal episodes without errors",
            "quality non-inferiority",
            "controller time reduction >= 25%",
            "end-to-end wall time reduction >= 8%",
        ],
    }
    manifest = export_controller_bundle(
        source_bundle_path,
        output_root,
        pruner_payload=selected_payload,
        pruner_ranges=_feature_ranges(training) if selected_payload is not None else None,
        pruner_threshold=selected_threshold,
        pruner_metadata={
            "training_source": selected_source,
            "training_labels_seen": ["policy_train"],
            "formal_or_ood_labels_seen": False,
            "validation_locked_once": True,
        }
        if selected_payload is not None
        else None,
        promotion_report=promotion_report,
    )
    report = {
        "schema": TRAINING_SCHEMA,
        "schema_version": 2,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "inputs": {
            "training_index": registered_path(train_path),
            "training_index_sha256": _sha256(train_path),
            "validation_index": registered_path(validation_path),
            "validation_index_sha256": _sha256(validation_path),
            "source_bundle": registered_path(source_bundle_path),
            "development_index": registered_path(development_path),
        },
        "training_candidate_count": len(training),
        "training_state_count": len(_grouped(training)),
        "training_map_count": len(train_maps),
        "validation_candidate_count": len(validation),
        "validation_state_count": len(_grouped(validation)),
        "validation_map_count": len(validation_maps),
        "folds": fold_reports,
        "full_training": full_training_report,
        "dedicated_calibration": dedicated_calibration,
        "dedicated_validation": dedicated_validation,
        "legacy_calibration": legacy_calibration,
        "legacy_validation": legacy_validation,
        "selected_pruner_source": selected_source,
        "selected_threshold": selected_threshold,
        "offline_pruner_passed": offline_pruner_passed,
        "main_ranker_equivalence": equivalence,
        "redundancy_audit": redundancy,
        "controller_manifest": manifest,
    }
    _write_json(output_root / "training_report.json", report)
    return report


__all__ = [
    "DEFAULT_THRESHOLDS",
    "MODEL_PARAMETERS",
    "TRAINING_SCHEMA",
    "balanced_map_folds",
    "calibrate_threshold",
    "evaluate_threshold",
    "family_probabilities",
    "main_ranker_equivalence",
    "out_of_fold_probabilities",
    "run_proposal_pruner_training",
    "train_pruner_model",
]
