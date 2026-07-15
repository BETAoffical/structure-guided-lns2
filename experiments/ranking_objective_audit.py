from __future__ import annotations

import collections
import itertools
import math
import pickle
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.closed_loop_confirmation import (
    _sha256,
    score_online_candidates,
)
from experiments.context_audit import PairwiseModel, _pair_vector
from experiments.policy_visited_aggregation_analysis import (
    _portable_model,
    _portable_payload,
    train_equal_state_pairwise_model,
)
from experiments.realized_neighborhood_ranking_audit import (
    _grouped,
    _selection_record,
    effectiveness_dominates,
    summarize_records,
)
from experiments.repair_collection import (
    SCHEMA_VERSION,
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)


SCHEMA = "lns2.ranking_objective_audit.v1"
PROFILE = "realized_dynamic"
METHODS = ("equal_pairwise", "impact_pairwise", "dual_outcome")


def _mean(values: Iterable[float | int]) -> float:
    numbers = list(map(float, values))
    return statistics.fmean(numbers) if numbers else 0.0


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _relative_improvement(baseline: float, challenger: float) -> float:
    if baseline == 0.0:
        return 0.0 if challenger == 0.0 else -float("inf")
    return (baseline - challenger) / baseline


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported ranking-objective audit config")
    if str(config.get("feature_profile")) != PROFILE:
        raise ValueError("objective audit must use realized_dynamic")
    if tuple(map(str, config.get("methods", []))) != METHODS:
        raise ValueError("objective audit methods differ from the registration")
    expected = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": 20260714,
    }
    if dict(config.get("model_parameters", {})) != expected:
        raise ValueError("objective audit model parameters changed")
    if int(config.get("trial_count", 0)) != 4:
        raise ValueError("objective audit requires four trials per candidate")
    if not math.isclose(float(config.get("solved_quantization", 0.0)), 0.25):
        raise ValueError("solved-rate quantization must be 0.25")
    if int(config.get("bootstrap_samples", 0)) != 5000:
        raise ValueError("objective audit requires 5,000 map bootstrap samples")


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({name for row in rows for name in row["features"][PROFILE]})


def _state_conflicts(candidates: list[dict[str, Any]]) -> float:
    values = {
        float(row["features"][PROFILE]["state.colliding_pairs"])
        for row in candidates
    }
    if len(values) != 1 or next(iter(values)) <= 0.0:
        raise ValueError("state candidates do not share a positive conflict count")
    return next(iter(values))


def train_impact_pairwise_model(
    rows: list[dict[str, Any]], model_parameters: dict[str, Any]
) -> tuple[PairwiseModel, dict[str, Any]]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    names = _feature_names(rows)
    examples: list[list[float]] = []
    labels: list[int] = []
    weights: list[float] = []
    state_weights: dict[str, float] = {}
    dominance_count = 0
    for state_id, candidates in sorted(_grouped(rows).items()):
        before = _state_conflicts(candidates)
        pairs: list[tuple[dict[str, Any], dict[str, Any], int, float]] = []
        for left, right in itertools.combinations(candidates, 2):
            if effectiveness_dominates(left["outcome"], right["outcome"]):
                label = 1
            elif effectiveness_dominates(right["outcome"], left["outcome"]):
                label = 0
            else:
                continue
            impact = max(
                abs(
                    float(left["outcome"]["solved_rate"])
                    - float(right["outcome"]["solved_rate"])
                ),
                abs(
                    float(left["outcome"]["conflicts_after"])
                    - float(right["outcome"]["conflicts_after"])
                )
                / max(1.0, before),
            )
            if impact <= 0.0:
                raise ValueError("a dominance pair has zero registered impact")
            pairs.append((left, right, label, impact))
        if not pairs:
            continue
        total = sum(row[3] for row in pairs)
        state_total = 0.0
        for left, right, label, impact in pairs:
            weight = impact / (2.0 * total)
            examples.append(_pair_vector(left, right, PROFILE, names))
            labels.append(label)
            weights.append(weight)
            examples.append(_pair_vector(right, left, PROFILE, names))
            labels.append(1 - label)
            weights.append(weight)
            state_total += 2.0 * weight
        state_weights[state_id] = state_total
        dominance_count += len(pairs)
    if not examples:
        raise ValueError("no impact-weighted dominance pairs are available")
    estimator = HistGradientBoostingClassifier(**model_parameters)
    estimator.fit(
        np.asarray(examples, dtype=float),
        np.asarray(labels, dtype=int),
        sample_weight=np.asarray(weights, dtype=float),
    )
    return PairwiseModel(PROFILE, names, estimator), {
        "state_count": len(state_weights),
        "dominance_pair_count": dominance_count,
        "example_count": len(examples),
        "state_weight_min": min(state_weights.values()),
        "state_weight_max": max(state_weights.values()),
        "equal_state_weight": all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in state_weights.values()
        ),
    }


@dataclass
class DualOutcomeModel:
    profile: str
    feature_names: list[str]
    solved_estimator: Any
    residual_estimator: Any
    solved_quantization: float = 0.25


@dataclass
class PortableRawRegressor:
    feature_names: list[str]
    baseline: float
    trees: list[list[dict[str, Any]]]
    native_predictor: Any | None = None

    def predict(self, vectors: list[list[float]]) -> list[float]:
        if self.native_predictor is not None:
            return list(map(float, self.native_predictor.predict_raw(vectors)))
        predictions = []
        for vector in vectors:
            raw = self.baseline
            for nodes in self.trees:
                index = 0
                while not bool(nodes[index]["is_leaf"]):
                    node = nodes[index]
                    value = float(vector[int(node["feature_idx"])])
                    go_left = (
                        math.isnan(value) and bool(node["missing_go_to_left"])
                    ) or (
                        not math.isnan(value)
                        and value <= float(node["num_threshold"])
                    )
                    index = int(node["left"] if go_left else node["right"])
                raw += float(nodes[index]["value"])
            predictions.append(raw)
        return predictions


@dataclass
class PortableDualOutcomeModel:
    profile: str
    feature_names: list[str]
    solved_estimator: PortableRawRegressor
    residual_estimator: PortableRawRegressor
    solved_quantization: float = 0.25


def _vectors(rows: list[dict[str, Any]], names: list[str]) -> list[list[float]]:
    return [
        [float(row["features"][PROFILE].get(name, 0.0)) for name in names]
        for row in rows
    ]


def train_dual_outcome_model(
    rows: list[dict[str, Any]],
    model_parameters: dict[str, Any],
    solved_quantization: float,
) -> tuple[DualOutcomeModel, dict[str, Any]]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    names = _feature_names(rows)
    solved: list[float] = []
    residual: list[float] = []
    weights: list[float] = []
    grouped = _grouped(rows)
    ordered_rows = [
        row
        for state_id in sorted(grouped)
        for row in grouped[state_id]
    ]
    matrix = np.asarray(_vectors(ordered_rows, names), dtype=float)
    for state_id in sorted(grouped):
        candidates = grouped[state_id]
        before = _state_conflicts(candidates)
        weight = 1.0 / len(candidates)
        for row in candidates:
            solved.append(float(row["outcome"]["solved_rate"]))
            residual.append(float(row["outcome"]["conflicts_after"]) / before)
            weights.append(weight)
    estimators = []
    for targets in (solved, residual):
        estimator = HistGradientBoostingRegressor(**model_parameters)
        estimator.fit(
            matrix,
            np.asarray(targets, dtype=float),
            sample_weight=np.asarray(weights, dtype=float),
        )
        estimators.append(estimator)
    state_weights = {
        state_id: sum(1.0 / len(candidates) for _ in candidates)
        for state_id, candidates in grouped.items()
    }
    return DualOutcomeModel(
        PROFILE, names, estimators[0], estimators[1], solved_quantization
    ), {
        "state_count": len(grouped),
        "candidate_count": len(rows),
        "state_weight_min": min(state_weights.values()),
        "state_weight_max": max(state_weights.values()),
        "equal_state_weight": all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in state_weights.values()
        ),
    }


def _quantized_solved(value: float, quantum: float) -> float:
    clamped = min(1.0, max(0.0, float(value)))
    return math.floor(clamped / quantum + 0.5) * quantum


def score_dual_outcome_candidates(
    rows: list[dict[str, Any]], model: DualOutcomeModel | PortableDualOutcomeModel
) -> tuple[int, dict[str, list[float]]]:
    if not rows:
        raise ValueError("cannot score an empty candidate pool")
    vectors = _vectors(rows, model.feature_names)
    if isinstance(model, PortableDualOutcomeModel):
        solved = model.solved_estimator.predict(vectors)
        residual = model.residual_estimator.predict(vectors)
    else:
        import numpy as np

        matrix = np.asarray(vectors, dtype=float)
        solved = list(map(float, model.solved_estimator.predict(matrix)))
        residual = list(map(float, model.residual_estimator.predict(matrix)))
    quantized = [_quantized_solved(value, model.solved_quantization) for value in solved]
    selected = min(
        range(len(rows)),
        key=lambda index: (
            -quantized[index],
            round(float(residual[index]), 12),
            str(rows[index]["candidate_key"]),
        ),
    )
    return selected, {
        "solved_rate": list(map(float, solved)),
        "quantized_solved_rate": quantized,
        "normalized_residual_conflicts": list(map(float, residual)),
    }


def _evaluate_model(
    rows: list[dict[str, Any]], model: Any, method: str
) -> dict[str, dict[str, Any]]:
    records = {}
    for state_id, candidates in _grouped(rows).items():
        if method == "dual_outcome":
            selected, _ = score_dual_outcome_candidates(candidates, model)
        else:
            selected, _, _ = score_online_candidates(candidates, model)
        records[state_id] = _selection_record(candidates[selected], candidates, method)
    return records


def _train_method(
    method: str, rows: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    parameters = dict(config["model_parameters"])
    if method == "equal_pairwise":
        return train_equal_state_pairwise_model(rows, PROFILE, parameters)
    if method == "impact_pairwise":
        return train_impact_pairwise_model(rows, parameters)
    if method == "dual_outcome":
        return train_dual_outcome_model(
            rows, parameters, float(config["solved_quantization"])
        )
    raise ValueError(f"unknown objective method: {method}")


def _map_bootstrap(
    baseline: dict[str, dict[str, Any]],
    challenger: dict[str, dict[str, Any]],
    samples: int,
) -> dict[str, Any]:
    if set(baseline) != set(challenger):
        raise ValueError("comparison records do not share states")
    by_map: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, row in baseline.items():
        if row["map_id"] != challenger[state_id]["map_id"]:
            raise ValueError("comparison map mismatch")
        by_map[str(row["map_id"])].append(state_id)
    maps = sorted(by_map)
    rng = random.Random(20260716)
    top1_delta = []
    regret_improvement = []
    for _ in range(samples):
        sampled = [rng.choice(maps) for _ in maps]
        states = [state for map_id in sampled for state in by_map[map_id]]
        base_top = _mean(baseline[state]["pareto_hit"] for state in states)
        new_top = _mean(challenger[state]["pareto_hit"] for state in states)
        base_regret = _mean(baseline[state]["conflict_regret"] for state in states)
        new_regret = _mean(challenger[state]["conflict_regret"] for state in states)
        top1_delta.append(new_top - base_top)
        regret_improvement.append(_relative_improvement(base_regret, new_regret))
    return {
        "map_count": len(maps),
        "samples": samples,
        "top1_delta_95_ci": [
            _quantile(top1_delta, 0.025),
            _quantile(top1_delta, 0.975),
        ],
        "conflict_regret_improvement_95_ci": [
            _quantile(regret_improvement, 0.025),
            _quantile(regret_improvement, 0.975),
        ],
    }


def objective_acceptance(
    baseline: dict[str, Any],
    challenger: dict[str, Any],
    bootstrap: dict[str, Any],
    maps_no_worse: int,
    map_count: int,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    top_delta = float(challenger["pareto_top1_hit_rate"]) - float(
        baseline["pareto_top1_hit_rate"]
    )
    regret_improvement = _relative_improvement(
        float(baseline["mean_conflict_regret"]),
        float(challenger["mean_conflict_regret"]),
    )
    top_qualifies = top_delta >= float(thresholds["minimum_top1_improvement"])
    regret_qualifies = regret_improvement >= float(
        thresholds["minimum_conflict_regret_improvement"]
    )
    other_not_degraded = (
        float(challenger["pareto_top1_hit_rate"])
        >= float(baseline["pareto_top1_hit_rate"])
        - float(thresholds["maximum_top1_degradation"])
        and regret_improvement
        >= -float(thresholds["maximum_conflict_regret_degradation"])
    )
    bootstrap_ok = (
        float(bootstrap["top1_delta_95_ci"][0]) >= 0.0
        or float(bootstrap["conflict_regret_improvement_95_ci"][0]) >= 0.0
    )
    minimum_maps = int(
        thresholds[
            "minimum_train_maps_no_worse"
            if map_count == 12
            else "minimum_validation_maps_no_worse"
        ]
    )
    gates = {
        "top1_or_conflict_regret_improves": top_qualifies or regret_qualifies,
        "other_metric_not_degraded": other_not_degraded,
        "map_bootstrap_not_degraded": bootstrap_ok,
        "minimum_maps_no_worse": maps_no_worse >= minimum_maps,
        "no_size_collapse": float(challenger["maximum_size_share"])
        <= float(thresholds["maximum_single_size_share"]),
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "top1_delta": top_delta,
        "conflict_regret_improvement": regret_improvement,
        "maps_no_worse": maps_no_worse,
        "map_count": map_count,
    }


def _maps_no_worse(
    baseline: dict[str, dict[str, Any]], challenger: dict[str, dict[str, Any]]
) -> tuple[int, dict[str, Any]]:
    by_map: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, row in baseline.items():
        by_map[str(row["map_id"])].append(state_id)
    details = {}
    count = 0
    for map_id, states in sorted(by_map.items()):
        base = _mean(baseline[state]["conflict_regret"] for state in states)
        new = _mean(challenger[state]["conflict_regret"] for state in states)
        no_worse = new <= base + 1e-12
        count += no_worse
        details[map_id] = {
            "state_count": len(states),
            "baseline_conflict_regret": base,
            "challenger_conflict_regret": new,
            "no_worse": no_worse,
        }
    return count, details


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0 + 1.0
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_var = sum((value - left_mean) ** 2 for value in left)
    right_var = sum((value - right_mean) ** 2 for value in right)
    if left_var == 0.0 or right_var == 0.0:
        return None
    return sum(
        (first - left_mean) * (second - right_mean)
        for first, second in zip(left, right)
    ) / math.sqrt(left_var * right_var)


def candidate_reliability_report(
    collection_root: Path, offline_root: Path, expected_trials: int
) -> dict[str, Any]:
    manifest = _read_jsonl(collection_root / "evaluation_trial_manifest.jsonl")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in manifest:
        if not bool(row.get("complete")) or row.get("error") is not None:
            raise ValueError("evaluation manifest contains an incomplete trial")
        grouped[(str(row["state_id"]), str(row["candidate_id"]))].append(
            dict(row["outcome"])
        )
    if any(
        len(values) != expected_trials
        or sorted(int(row["evaluation_trial_index"]) for row in values)
        != list(range(expected_trials))
        for values in grouped.values()
    ):
        raise ValueError("candidate trial coverage is incomplete")
    states: dict[str, dict[str, list[dict[str, Any]]]] = collections.defaultdict(dict)
    for (state_id, candidate_id), values in grouped.items():
        states[state_id][candidate_id] = sorted(
            values, key=lambda row: int(row["evaluation_trial_index"])
        )
    eta_squared = []
    split_spearman = []
    split_best_overlap = 0
    oracle_reductions = []
    constant_candidates = 0
    for candidates in states.values():
        values = [
            float(row["conflicts_after"])
            for trials in candidates.values()
            for row in trials
        ]
        grand_mean = _mean(values)
        total = sum((value - grand_mean) ** 2 for value in values)
        between = sum(
            len(trials)
            * (_mean(row["conflicts_after"] for row in trials) - grand_mean) ** 2
            for trials in candidates.values()
        )
        eta_squared.append(between / total if total else 0.0)
        ordered = [candidates[key] for key in sorted(candidates)]
        first = [_mean(row["conflicts_after"] for row in rows[:2]) for rows in ordered]
        second = [_mean(row["conflicts_after"] for row in rows[2:]) for rows in ordered]
        correlation = _correlation(_ranks(first), _ranks(second))
        if correlation is not None:
            split_spearman.append(correlation)
        split_best_overlap += bool(
            {index for index, value in enumerate(first) if value == min(first)}
            & {index for index, value in enumerate(second) if value == min(second)}
        )
        constant_candidates += sum(
            len({float(row["conflicts_after"]) for row in trials}) == 1
            for trials in ordered
        )
        before_values = {
            float(row["conflicts_before"]) for trials in ordered for row in trials
        }
        if len(before_values) != 1:
            raise ValueError("trial state has inconsistent initial conflicts")
        oracle_reductions.append(
            next(iter(before_values))
            - min(_mean(row["conflicts_after"] for row in trials) for trials in ordered)
        )

    def predictions(name: str) -> dict[str, dict[str, Any]]:
        path = offline_root / f"offline_predictions__{name}.jsonl"
        return {str(row["state_id"]): row for row in _read_jsonl(path)}

    v1 = predictions("v1_realized_dynamic")
    v2 = predictions("v2_realized_dynamic")
    changed = {state for state in v1 if v1[state]["candidate_id"] != v2[state]["candidate_id"]}
    directions = collections.Counter()
    for state in changed:
        delta = float(v1[state]["conflict_regret"]) - float(v2[state]["conflict_regret"])
        directions["better" if delta > 1e-12 else "worse" if delta < -1e-12 else "tie"] += 1
    _, map_details = _maps_no_worse(v1, v2)
    return {
        "state_count": len(states),
        "candidate_count": len(grouped),
        "trial_count": len(manifest),
        "trials_per_candidate": expected_trials,
        "candidate_all_trials_same_fraction": constant_candidates / len(grouped),
        "action_eta_squared": {
            "mean": _mean(eta_squared),
            "median": statistics.median(eta_squared),
            "fraction_at_least_0_5": sum(value >= 0.5 for value in eta_squared)
            / len(eta_squared),
        },
        "split_trial_spearman": {
            "valid_state_count": len(split_spearman),
            "mean": _mean(split_spearman),
            "median": statistics.median(split_spearman),
        },
        "split_best_overlap_fraction": split_best_overlap / len(states),
        "oracle_conflict_reduction": {
            "positive_state_fraction": sum(value > 0.0 for value in oracle_reductions)
            / len(oracle_reductions),
            "mean": _mean(oracle_reductions),
            "median": statistics.median(oracle_reductions),
            "maximum": max(oracle_reductions),
        },
        "v1_v2_selection": {
            "state_count": len(v1),
            "same_candidate_count": len(v1) - len(changed),
            "changed_candidate_count": len(changed),
            "changed_directions": dict(sorted(directions.items())),
            "per_map": map_details,
        },
    }


def leave_one_train_map_out(
    train_rows: list[dict[str, Any]], anchors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    maps = sorted({str(row["map_id"]) for row in train_rows})
    if len(maps) != 12:
        raise ValueError("objective audit requires exactly 12 Train maps")
    anchor_states = set(_grouped(anchors))
    folds = []
    for validation_map in maps:
        fit_rows = anchors + [
            row for row in train_rows if str(row["map_id"]) != validation_map
        ]
        held_rows = [
            row for row in train_rows if str(row["map_id"]) == validation_map
        ]
        fit_states = set(_grouped(fit_rows))
        held_states = set(_grouped(held_rows))
        if not held_rows or fit_states & held_states:
            raise ValueError("leave-one-map-out state leakage detected")
        if not anchor_states.issubset(fit_states) or anchor_states & held_states:
            raise ValueError("historical anchors leaked into evaluation")
        if any(str(row["map_id"]) == validation_map for row in fit_rows):
            raise ValueError("leave-one-map-out map leakage detected")
        folds.append(
            {
                "validation_map": validation_map,
                "fit_rows": fit_rows,
                "held_rows": held_rows,
                "anchor_state_count": len(anchor_states),
            }
        )
    return folds


def _cross_validate(
    train_rows: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    maps = sorted({str(row["map_id"]) for row in train_rows})
    folds = leave_one_train_map_out(train_rows, anchors)
    all_records = {method: {} for method in METHODS}
    fold_reports = []
    for fold, split in enumerate(folds):
        validation_map = str(split["validation_map"])
        fit_rows = list(split["fit_rows"])
        held_rows = list(split["held_rows"])
        diagnostics = {}
        for method in METHODS:
            model, diagnostic = _train_method(method, fit_rows, config)
            records = _evaluate_model(held_rows, model, method)
            if set(records) & set(all_records[method]):
                raise ValueError("cross-validation state evaluated more than once")
            all_records[method].update(records)
            diagnostics[method] = diagnostic
        fold_reports.append(
            {
                "fold": fold,
                "validation_map": validation_map,
                "fit_map_count": len(maps) - 1,
                "anchor_state_count": int(split["anchor_state_count"]),
                "validation_state_count": len(_grouped(held_rows)),
                "training_diagnostics": diagnostics,
            }
        )
    summaries = {
        method: summarize_records(records) for method, records in all_records.items()
    }
    comparisons = {}
    for method in METHODS[1:]:
        no_worse, map_details = _maps_no_worse(
            all_records["equal_pairwise"], all_records[method]
        )
        bootstrap = _map_bootstrap(
            all_records["equal_pairwise"],
            all_records[method],
            int(config["bootstrap_samples"]),
        )
        comparisons[method] = {
            "bootstrap": bootstrap,
            "map_details": map_details,
            "acceptance": objective_acceptance(
                summaries["equal_pairwise"],
                summaries[method],
                bootstrap,
                no_worse,
                len(maps),
                dict(config["thresholds"]),
            ),
        }
    eligible = [method for method in METHODS[1:] if comparisons[method]["acceptance"]["passed"]]
    winner = None
    if eligible:
        winner = sorted(
            eligible,
            key=lambda method: (
                -float(summaries[method]["pareto_top1_hit_rate"]),
                float(summaries[method]["mean_conflict_regret"]),
                0 if method == "impact_pairwise" else 1,
            ),
        )[0]
    report = {
        "map_count": len(maps),
        "state_count": len(_grouped(train_rows)),
        "anchor_state_count": len(_grouped(anchors)),
        "validation_labels_used_for_method_selection": False,
        "folds": fold_reports,
        "summaries": summaries,
        "comparisons": comparisons,
        "eligible_challengers": eligible,
        "winner": winner,
        "passed": winner is not None,
    }
    return report, all_records


def _portable_regressor_payload(estimator: Any, feature_names: list[str]) -> dict[str, Any]:
    trees = []
    for stage in estimator._predictors:
        if len(stage) != 1:
            raise ValueError("portable regressor supports one tree per stage")
        nodes = []
        for node in stage[0].nodes:
            if bool(node["is_categorical"]):
                raise ValueError("portable regressor does not support categorical nodes")
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
        "schema": "lns2.portable_hist_gbdt.v2",
        "schema_version": 2,
        "output_transform": "identity",
        "feature_names": feature_names,
        "baseline": float(estimator._baseline_prediction[0, 0]),
        "trees": trees,
    }


def _portable_raw_model(payload: dict[str, Any]) -> PortableRawRegressor:
    native = None
    try:
        import lns2_env

        predictor_type = getattr(lns2_env, "PortableTreeEnsemble", None)
        if predictor_type is not None and hasattr(predictor_type, "predict_raw"):
            native = predictor_type(float(payload["baseline"]), list(payload["trees"]))
    except ImportError:
        pass
    return PortableRawRegressor(
        list(map(str, payload["feature_names"])),
        float(payload["baseline"]),
        list(payload["trees"]),
        native,
    )


def export_objective_model(
    method: str,
    model: Any,
    train_rows: list[dict[str, Any]],
    output: Path,
    solved_quantization: float,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    grouped = _grouped(train_rows)
    if method == "impact_pairwise":
        payload = _portable_payload(model, "unregistered-diagnostic-source")
        path = output / "impact_pairwise.json"
        _write_json(path, payload)
        portable: Any = _portable_model(payload)
        files = [{"role": "pairwise", "file": path.name, "sha256": _sha256(path)}]
    elif method == "dual_outcome":
        solved_payload = _portable_regressor_payload(model.solved_estimator, model.feature_names)
        residual_payload = _portable_regressor_payload(
            model.residual_estimator, model.feature_names
        )
        solved_path = output / "solved_rate.json"
        residual_path = output / "normalized_residual_conflicts.json"
        _write_json(solved_path, solved_payload)
        _write_json(residual_path, residual_payload)
        portable = PortableDualOutcomeModel(
            PROFILE,
            model.feature_names,
            _portable_raw_model(solved_payload),
            _portable_raw_model(residual_payload),
            solved_quantization,
        )
        files = [
            {"role": "solved_rate", "file": solved_path.name, "sha256": _sha256(solved_path)},
            {
                "role": "normalized_residual_conflicts",
                "file": residual_path.name,
                "sha256": _sha256(residual_path),
            },
        ]
    else:
        raise ValueError("only challenger models may be exported")
    mismatch = 0
    maximum_delta = 0.0
    for candidates in grouped.values():
        if method == "impact_pairwise":
            native_index, native_scores, _ = score_online_candidates(candidates, model)
            portable_index, portable_scores, _ = score_online_candidates(candidates, portable)
            maximum_delta = max(
                maximum_delta,
                max(abs(a - b) for a, b in zip(native_scores, portable_scores)),
            )
        else:
            native_index, native_values = score_dual_outcome_candidates(candidates, model)
            portable_index, portable_values = score_dual_outcome_candidates(candidates, portable)
            for name in native_values:
                maximum_delta = max(
                    maximum_delta,
                    max(
                        abs(a - b)
                        for a, b in zip(native_values[name], portable_values[name])
                    ),
                )
        mismatch += native_index != portable_index
    equivalence = {
        "state_count": len(grouped),
        "selection_mismatch_count": mismatch,
        "maximum_prediction_delta": maximum_delta,
        "passed": mismatch == 0 and maximum_delta <= 1e-12,
    }
    if not equivalence["passed"]:
        raise ValueError("objective model portable inference mismatch")
    manifest = {
        "schema": "lns2.ranking_objective_bundle.v1",
        "schema_version": 1,
        "selector_type": method,
        "feature_profile": PROFILE,
        "feature_names": list(model.feature_names),
        "solved_quantization": solved_quantization,
        "files": files,
        "equivalence": equivalence,
        "confirmation_labels_seen": False,
    }
    _write_json(output / "portable_manifest.json", manifest)
    return manifest


def _load_registered_inputs(
    collection_root: Path,
    training_root: Path,
    offline_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    expected = dict(config["registered_inputs"])
    checks = {
        "collection_run_fingerprint": str(
            _read_json(collection_root / "run_config.json")["run_fingerprint"]
        ),
        "aggregate_train_index_sha256": _sha256(
            training_root / "aggregate_train_index.jsonl"
        ),
        "validation_index_sha256": _sha256(training_root / "validation_index.jsonl"),
        "training_report_sha256": _sha256(training_root / "training_report.json"),
        "offline_report_sha256": _sha256(offline_root / "offline_report.json"),
    }
    if checks != expected:
        raise ValueError(f"registered objective-audit inputs changed: {checks}")
    aggregate = _read_jsonl(training_root / "aggregate_train_index.jsonl")
    train = [row for row in aggregate if str(row.get("split")) == "policy_train"]
    anchors = [row for row in aggregate if str(row.get("split")) != "policy_train"]
    validation = _read_jsonl(training_root / "validation_index.jsonl")
    if len(_grouped(train)) != 288 or len(_grouped(anchors)) != 23:
        raise ValueError("registered Train or historical anchor state count changed")
    if len(_grouped(validation)) != 154:
        raise ValueError("registered Validation state count changed")
    return train, anchors, validation


def run_ranking_objective_audit(
    collection: str | Path,
    training: str | Path,
    offline: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
) -> dict[str, Any]:
    if phase not in {"diagnose", "cross_validate", "validate", "all"}:
        raise ValueError("phase must be diagnose, cross_validate, validate, or all")
    collection_root = Path(collection).resolve()
    training_root = Path(training).resolve()
    offline_root = Path(offline).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_config(config)
    train, anchors, validation = _load_registered_inputs(
        collection_root, training_root, offline_root, config
    )
    run_fingerprint = _fingerprint(
        {
            "schema": SCHEMA,
            "configuration": config,
            "implementation": _sha256(Path(__file__)),
        }
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_root / "run_config.json",
        {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "configuration": config,
        },
    )
    reliability = candidate_reliability_report(
        collection_root, offline_root, int(config["trial_count"])
    )
    _write_json(output_root / "candidate_reliability.json", reliability)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "candidate_reliability": reliability,
        "static_context_used": False,
        "rl_trained": False,
        "new_data_collected": False,
    }
    if phase == "diagnose":
        _write_json(output_root / "audit_report.json", result)
        return result
    cross_validation, records = _cross_validate(train, anchors, config)
    result["cross_validation"] = cross_validation
    for method, values in records.items():
        _write_jsonl(
            output_root / f"cross_validation_predictions__{method}.jsonl",
            [values[key] for key in sorted(values)],
        )
    if phase == "cross_validate" or not cross_validation["passed"]:
        result["decision"] = "stop_objective_alignment" if not cross_validation["passed"] else "eligible_for_development_validation"
        result["validation_evaluated"] = False
        result["confirmation_generation_allowed"] = False
        _write_json(output_root / "audit_report.json", result)
        return result
    winner = str(cross_validation["winner"])
    final_model, training_diagnostic = _train_method(winner, anchors + train, config)
    winner_records = _evaluate_model(validation, final_model, winner)
    v1 = {
        str(row["state_id"]): row
        for row in _read_jsonl(
            offline_root / "offline_predictions__v1_realized_dynamic.jsonl"
        )
    }
    v2 = {
        str(row["state_id"]): row
        for row in _read_jsonl(
            offline_root / "offline_predictions__v2_realized_dynamic.jsonl"
        )
    }
    if set(winner_records) != set(v1) or set(v1) != set(v2):
        raise ValueError("development Validation state sets differ")
    winner_summary = summarize_records(winner_records)
    v1_summary = summarize_records(v1)
    no_worse, map_details = _maps_no_worse(v1, winner_records)
    bootstrap = _map_bootstrap(
        v1, winner_records, int(config["bootstrap_samples"])
    )
    acceptance = objective_acceptance(
        v1_summary,
        winner_summary,
        bootstrap,
        no_worse,
        6,
        dict(config["thresholds"]),
    )
    validation_report = {
        "winner": winner,
        "validation_labels_used_for_training": False,
        "state_count": len(winner_records),
        "training_diagnostic": training_diagnostic,
        "summaries": {
            "v1_realized_dynamic": v1_summary,
            "v2_equal_pairwise": summarize_records(v2),
            winner: winner_summary,
        },
        "bootstrap": bootstrap,
        "map_details": map_details,
        "acceptance": acceptance,
    }
    _write_jsonl(
        output_root / f"validation_predictions__{winner}.jsonl",
        [winner_records[key] for key in sorted(winner_records)],
    )
    _write_json(output_root / "validation_report.json", validation_report)
    result["development_validation"] = validation_report
    result["validation_evaluated"] = True
    if acceptance["passed"]:
        model_root = output_root / "frozen_winner"
        model_root.mkdir(parents=True, exist_ok=True)
        with (model_root / f"{winner}.pkl").open("wb") as stream:
            pickle.dump(final_model, stream)
        manifest = export_objective_model(
            winner,
            final_model,
            anchors + train,
            model_root / "portable",
            float(config["solved_quantization"]),
        )
        result["portable_manifest"] = manifest
        result["decision"] = "eligible_for_independent_confirmation"
        result["confirmation_generation_allowed"] = True
    else:
        result["decision"] = "stop_objective_alignment"
        result["confirmation_generation_allowed"] = False
    _write_json(output_root / "audit_report.json", result)
    return result


__all__ = [
    "DualOutcomeModel",
    "METHODS",
    "PortableDualOutcomeModel",
    "PortableRawRegressor",
    "candidate_reliability_report",
    "export_objective_model",
    "leave_one_train_map_out",
    "objective_acceptance",
    "run_ranking_objective_audit",
    "score_dual_outcome_candidates",
    "train_dual_outcome_model",
    "train_impact_pairwise_model",
]
