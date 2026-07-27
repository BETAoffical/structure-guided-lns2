from __future__ import annotations

import collections
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

from experiments._common import atomic_write_csv, sha256_file
from experiments.feature_schema_v3 import V3_FEATURE_NAMES
from experiments.repair_aware_training import _balanced_map_folds
from experiments.repair_collection import _read_jsonl, _write_json
from experiments.v2_cost_tiebreak_audit import (
    TOP_K,
    _comparison,
    _conservative_predictions,
    _diagnostic_checks,
    _margins,
    _metrics,
    _model_arms,
    _oracle_top3,
    _training_checks,
    _uncertainty_calibration,
    _v2_order,
    select_cost_tiebreak,
)
from experiments.v3_training import (
    MODEL_PARAMETERS,
    _fit,
    _index_predictions,
    _outcome_name,
    _predict,
    _rows,
    _states,
    _with_runtime_predictions,
)


V2_COST_TIEBREAK_ERROR_AUDIT_SCHEMA = (
    "lns2.v2_cost_tiebreak_error_audit.v1"
)
PAIRWISE_THRESHOLD_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)
PAIRWISE_QUALITY_RETENTION = 0.98
PAIRWISE_MINIMUM_TIME_IMPROVEMENT = 0.10
PAIRWISE_FEATURE_COUNT = 95


def _positive_probability(estimator: Any, values: Any) -> list[float]:
    classes = list(map(int, estimator.classes_))
    if classes == [0]:
        return [0.0] * len(values)
    if classes == [1]:
        return [1.0] * len(values)
    index = classes.index(1)
    return list(map(float, estimator.predict_proba(values)[:, index]))


def _actual_from_trial(trial: dict[str, Any]) -> dict[str, float]:
    outcome = dict(trial["outcome"])
    before = int(outcome["conflicts_before"])
    after = int(outcome["conflicts_after"])
    name = _outcome_name(outcome)
    return {
        "effective": float(name in {"conflict_reduced", "feasible"}),
        "no_progress": float(name in {"hard_failure", "accepted_noop"}),
        "conflict_reduction": float(max(0, before - after)),
        "repair_seconds": max(1e-9, float(outcome["repair_seconds"])),
    }


def validate_paired_trials(
    feature_rows: list[dict[str, Any]], trial_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    candidates = {
        (str(row["split"]), str(row["state_id"]), str(row["candidate_id"]))
        for row in feature_rows
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    seed_groups: dict[tuple[str, str, int], set[int]] = collections.defaultdict(
        set
    )
    for raw in trial_rows:
        row = dict(raw)
        if str(row.get("status")) not in {"ok", "resumed"} or not bool(
            row.get("complete")
        ):
            raise ValueError("cost-tiebreak error audit found incomplete trial")
        key = (
            str(row["split"]),
            str(row["state_id"]),
            str(row["candidate_id"]),
        )
        grouped[key].append(row)
        seed_groups[(key[0], key[1], int(row["trial_index"]))].add(
            int(row["random_seed"])
        )
    if set(grouped) != candidates:
        raise ValueError("paired trials do not cover every candidate")
    two_seed = 0
    four_seed = 0
    for key, rows in grouped.items():
        indices = [int(row["trial_index"]) for row in rows]
        if len(indices) != len(set(indices)):
            raise ValueError(f"duplicate paired trial: {key}")
        if set(indices) == {0, 1}:
            two_seed += 1
        elif set(indices) == {0, 1, 2, 3}:
            four_seed += 1
        else:
            raise ValueError(f"candidate has invalid paired seed coverage: {key}")
    if any(len(seeds) != 1 for seeds in seed_groups.values()):
        raise ValueError("candidates do not share paired random seeds")
    return {
        "candidate_count": len(grouped),
        "two_seed_candidate_count": two_seed,
        "four_seed_candidate_count": four_seed,
        "paired_state_trial_count": len(seed_groups),
        "all_candidates_have_seed_zero_one": True,
        "all_state_trials_share_random_seed": True,
    }


def _trial_index(
    trial_rows: Iterable[dict[str, Any]], split: str
) -> dict[tuple[str, str], dict[int, dict[str, Any]]]:
    result: dict[tuple[str, str], dict[int, dict[str, Any]]] = (
        collections.defaultdict(dict)
    )
    for raw in trial_rows:
        if str(raw.get("split")) != split:
            continue
        key = (str(raw["state_id"]), str(raw["candidate_id"]))
        trial = int(raw["trial_index"])
        if trial in result[key]:
            raise ValueError("duplicate trial index in paired trial manifest")
        result[key][trial] = {
            "random_seed": int(raw["random_seed"]),
            **_actual_from_trial(raw),
        }
    return dict(result)


def _quality_safe(candidate: dict[str, float], base: dict[str, float]) -> bool:
    return bool(
        float(candidate["conflict_reduction"]) + 1e-12
        >= PAIRWISE_QUALITY_RETENTION * float(base["conflict_reduction"])
        and float(candidate["effective"]) + 1e-12 >= float(base["effective"])
        and float(candidate["no_progress"])
        <= float(base["no_progress"]) + 1e-12
    )


def _preferred(candidate: dict[str, float], base: dict[str, float]) -> bool:
    return bool(
        _quality_safe(candidate, base)
        and float(candidate["repair_seconds"])
        <= (1.0 - PAIRWISE_MINIMUM_TIME_IMPROVEMENT)
        * float(base["repair_seconds"])
        + 1e-12
    )


def _pair_features(
    candidate: list[float],
    base: list[float],
    *,
    v2_score_gap: float,
) -> list[float]:
    if len(candidate) != len(V3_FEATURE_NAMES) or len(base) != len(
        V3_FEATURE_NAMES
    ):
        raise ValueError("pairwise feature vector does not match feature-v3")
    state_start = V3_FEATURE_NAMES.index("state.agent_count")
    values = [
        float(candidate[index]) - float(base[index])
        for index in range(state_start)
    ]
    values.extend(map(float, base[state_start:]))
    values.append(float(v2_score_gap))
    if len(values) != PAIRWISE_FEATURE_COUNT:
        raise ValueError("pairwise feature count changed unexpectedly")
    return values


def build_pair_rows(
    states: list[dict[str, Any]],
    raw_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
    split: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    feature_index: dict[tuple[str, str], list[float]] = {}
    for row in raw_rows:
        key = (str(row["state_id"]), str(row["candidate_id"]))
        values = list(map(float, row["features"]))
        previous = feature_index.setdefault(key, values)
        if previous != values:
            raise ValueError("candidate features differ across paired seeds")
    trials = _trial_index(trial_rows, split)
    pairs = []
    state_rows = []
    for state in states:
        arms = _model_arms(state)
        order = _v2_order(arms)
        base = arms[order[0]]
        base_key = (str(state["state_id"]), str(base["candidate_id"]))
        base_trials = trials[base_key]
        state_pair_labels = []
        for index in order[1:TOP_K]:
            arm = arms[index]
            key = (str(state["state_id"]), str(arm["candidate_id"]))
            candidate_trials = trials[key]
            common = sorted(set(base_trials) & set(candidate_trials))
            if common[:2] != [0, 1]:
                raise ValueError("top-3 candidate lacks paired seed zero and one")
            labels = [
                int(_preferred(candidate_trials[trial], base_trials[trial]))
                for trial in common
            ]
            candidate_actual = dict(arm["actual"])
            base_actual = dict(base["actual"])
            mean_label = int(
                _preferred(
                    {
                        "effective": float(candidate_actual["effective_rate"]),
                        "no_progress": float(
                            candidate_actual["no_progress_rate"]
                        ),
                        "conflict_reduction": float(
                            candidate_actual["conflict_reduction"]
                        ),
                        "repair_seconds": float(
                            candidate_actual["repair_seconds"]
                        ),
                    },
                    {
                        "effective": float(base_actual["effective_rate"]),
                        "no_progress": float(base_actual["no_progress_rate"]),
                        "conflict_reduction": float(
                            base_actual["conflict_reduction"]
                        ),
                        "repair_seconds": float(base_actual["repair_seconds"]),
                    },
                )
            )
            pair = {
                "split": split,
                "state_id": str(state["state_id"]),
                "map_id": str(state["map_id"]),
                "layout_mode": str(state["layout_mode"]),
                "agent_count": int(state["agent_count"]),
                "candidate_id": str(arm["candidate_id"]),
                "base_candidate_id": str(base["candidate_id"]),
                "candidate_size": int(arm["actual_size"]),
                "base_size": int(base["actual_size"]),
                "features": _pair_features(
                    feature_index[key],
                    feature_index[base_key],
                    v2_score_gap=float(arm["v2_score"])
                    - float(base["v2_score"]),
                ),
                "label": mean_label,
                "seed_zero_label": labels[0],
                "seed_one_label": labels[1],
                "seed_label_agreement": int(labels[0] == labels[1]),
                "any_seed_preferred": int(any(labels[:2])),
                "all_available_seed_label_agreement": int(
                    len(set(labels)) == 1
                ),
                "trial_count": len(common),
            }
            pairs.append(pair)
            state_pair_labels.append(pair)
        state_rows.append(
            _state_seed_stability(state, arms, order, trials, state_pair_labels)
        )
    return pairs, state_rows


def _seed_oracle(
    arms: list[dict[str, Any]],
    order: list[int],
    trials: dict[tuple[str, str], dict[int, dict[str, Any]]],
    state_id: str,
    trial_index: int,
) -> str:
    base = arms[order[0]]
    base_actual = trials[(state_id, str(base["candidate_id"]))][trial_index]
    eligible = []
    for index in order[:TOP_K]:
        arm = arms[index]
        actual = trials[(state_id, str(arm["candidate_id"]))][trial_index]
        if _quality_safe(actual, base_actual):
            eligible.append((arm, actual))
    arm, _ = max(
        eligible or [(base, base_actual)],
        key=lambda value: (
            float(value[1]["conflict_reduction"])
            / max(1e-9, float(value[1]["repair_seconds"])),
            float(value[1]["conflict_reduction"]),
            -float(value[1]["repair_seconds"]),
            float(value[0]["v2_score"]),
            str(value[0]["candidate_id"]),
        ),
    )
    return str(arm["candidate_id"])


def _state_seed_stability(
    state: dict[str, Any],
    arms: list[dict[str, Any]],
    order: list[int],
    trials: dict[tuple[str, str], dict[int, dict[str, Any]]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    state_id = str(state["state_id"])
    base_id = str(arms[order[0]]["candidate_id"])
    zero = _seed_oracle(arms, order, trials, state_id, 0)
    one = _seed_oracle(arms, order, trials, state_id, 1)
    return {
        "state_id": state_id,
        "map_id": str(state["map_id"]),
        "layout_mode": str(state["layout_mode"]),
        "agent_count": int(state["agent_count"]),
        "base_candidate_id": base_id,
        "seed_zero_oracle_candidate_id": zero,
        "seed_one_oracle_candidate_id": one,
        "oracle_candidate_agreement": int(zero == one),
        "oracle_route_agreement": int((zero == base_id) == (one == base_id)),
        "disagreeing_pair_count": sum(
            int(not bool(row["seed_label_agreement"])) for row in pairs
        ),
    }


def seed_stability_report(
    pairs: list[dict[str, Any]], states: list[dict[str, Any]]
) -> dict[str, Any]:
    conditional = [row for row in pairs if bool(row["any_seed_preferred"])]
    four_seed = [row for row in pairs if int(row["trial_count"]) == 4]
    return {
        "pair_count": len(pairs),
        "pair_label_agreement_fraction": statistics.fmean(
            float(row["seed_label_agreement"]) for row in pairs
        ),
        "conditional_pair_count": len(conditional),
        "conditional_pair_label_agreement_fraction": (
            statistics.fmean(
                float(row["seed_label_agreement"]) for row in conditional
            )
            if conditional
            else 0.0
        ),
        "pair_preferred_on_both_seed_count": sum(
            int(row["seed_zero_label"] and row["seed_one_label"])
            for row in pairs
        ),
        "pair_preferred_on_neither_seed_count": sum(
            int(not row["seed_zero_label"] and not row["seed_one_label"])
            for row in pairs
        ),
        "pair_seed_disagreement_count": sum(
            int(not bool(row["seed_label_agreement"])) for row in pairs
        ),
        "four_seed_top3_pair_count": len(four_seed),
        "four_seed_all_label_agreement_fraction": (
            statistics.fmean(
                float(row["all_available_seed_label_agreement"])
                for row in four_seed
            )
            if four_seed
            else 0.0
        ),
        "state_count": len(states),
        "state_oracle_candidate_agreement_fraction": statistics.fmean(
            float(row["oracle_candidate_agreement"]) for row in states
        ),
        "state_oracle_route_agreement_fraction": statistics.fmean(
            float(row["oracle_route_agreement"]) for row in states
        ),
    }


def _balanced_pair_weights(rows: list[dict[str, Any]]) -> list[float]:
    class_counts = collections.Counter(int(row["label"]) for row in rows)
    map_counts = collections.Counter(str(row["map_id"]) for row in rows)
    if len(class_counts) != 2:
        raise ValueError("pairwise audit requires both target classes")
    raw = [
        1.0
        / (
            float(class_counts[int(row["label"])])
            * float(map_counts[str(row["map_id"])])
        )
        for row in rows
    ]
    scale = len(raw) / math.fsum(raw)
    return [value * scale for value in raw]


def _fit_pairwise(rows: list[dict[str, Any]]) -> Any:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(**MODEL_PARAMETERS)
    model.fit(
        np.asarray([row["features"] for row in rows], dtype=float),
        np.asarray([row["label"] for row in rows], dtype=int),
        sample_weight=np.asarray(_balanced_pair_weights(rows), dtype=float),
    )
    return model


def _predict_pairwise(model: Any, rows: list[dict[str, Any]]) -> list[float]:
    import numpy as np

    values = np.asarray([row["features"] for row in rows], dtype=float)
    return _positive_probability(model, values)


def _pair_prediction_index(
    rows: list[dict[str, Any]], predictions: list[float]
) -> dict[tuple[str, str], float]:
    if len(rows) != len(predictions):
        raise ValueError("pairwise predictions do not align with rows")
    result = {}
    for row, prediction in zip(rows, predictions):
        key = (str(row["state_id"]), str(row["candidate_id"]))
        if key in result:
            raise ValueError("duplicate pairwise prediction")
        result[key] = float(prediction)
    return result


def evaluate_pairwise_policy(
    states: list[dict[str, Any]],
    predictions: dict[tuple[str, str], float],
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = []
    diagnostics = []
    for state in states:
        arms = _model_arms(state)
        order = _v2_order(arms)
        base = arms[order[0]]
        alternatives = []
        for index in order[1:TOP_K]:
            arm = arms[index]
            probability = predictions[
                (str(state["state_id"]), str(arm["candidate_id"]))
            ]
            if probability + 1e-12 >= threshold:
                alternatives.append((arm, probability))
        arm, probability = max(
            alternatives or [(base, 0.0)],
            key=lambda value: (
                float(value[1]),
                float(value[0]["v2_score"]),
                str(value[0]["candidate_id"]),
            ),
        )
        selected.append(arm)
        diagnostics.append(
            {
                "state_id": str(state["state_id"]),
                "map_id": str(state["map_id"]),
                "layout_mode": str(state["layout_mode"]),
                "agent_count": int(state["agent_count"]),
                "base_candidate_id": str(base["candidate_id"]),
                "selected_candidate_id": str(arm["candidate_id"]),
                "override": int(arm is not base),
                "selected_probability": float(probability),
                "selected_actual_size": int(arm["actual_size"]),
            }
        )
    base = [
        next(arm for arm in _model_arms(state) if bool(arm["base_selected"]))
        for state in states
    ]
    chosen = _metrics(selected)
    baseline = _metrics(base)
    comparison = _comparison(chosen, baseline)
    cell_rows = _pairwise_cells(states, selected)
    return {
        "selected": chosen,
        "v2": baseline,
        "comparison": comparison,
        "override_count": sum(int(row["override"]) for row in diagnostics),
        "override_fraction": statistics.fmean(
            float(row["override"]) for row in diagnostics
        ),
        "cell_gate": {
            "cell_count": len(cell_rows),
            "noninferior_cell_count": sum(
                int(row["noninferior"]) for row in cell_rows
            ),
            "rows": cell_rows,
        },
    }, diagnostics


def _pairwise_cells(
    states: list[dict[str, Any]], selected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[int]] = collections.defaultdict(list)
    for index, state in enumerate(states):
        groups[(str(state["layout_mode"]), int(state["agent_count"]))].append(
            index
        )
    rows = []
    for (layout, agents), indices in sorted(groups.items()):
        chosen = _metrics([selected[index] for index in indices])
        base = _metrics(
            [
                next(
                    arm
                    for arm in _model_arms(states[index])
                    if bool(arm["base_selected"])
                )
                for index in indices
            ]
        )
        comparison = _comparison(chosen, base)
        rows.append(
            {
                "layout_mode": layout,
                "agent_count": agents,
                "state_count": len(indices),
                **comparison,
                "noninferior": int(
                    comparison["conflict_reduction_ratio"] >= 0.95
                    and comparison["efficiency_ratio"] >= 0.95
                ),
            }
        )
    return rows


def calibrate_pairwise_policy(
    states: list[dict[str, Any]],
    predictions: dict[tuple[str, str], float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for threshold in PAIRWISE_THRESHOLD_GRID:
        metrics, _ = evaluate_pairwise_policy(states, predictions, threshold)
        checks = _training_checks(metrics)
        rows.append(
            {
                "threshold": threshold,
                **metrics["comparison"],
                "override_fraction": metrics["override_fraction"],
                "noninferior_cell_count": metrics["cell_gate"][
                    "noninferior_cell_count"
                ],
                "passed_check_count": sum(map(int, checks.values())),
                "passed": int(all(checks.values())),
            }
        )
    passing = [row for row in rows if bool(row["passed"])]
    selected = max(
        passing or rows,
        key=lambda row: (
            bool(row["passed"]),
            int(row["passed_check_count"]),
            float(row["efficiency_ratio"]),
            -float(row["repair_time_ratio"]),
            float(row["conflict_reduction_ratio"]),
            float(row["threshold"]),
        ),
    )
    return {
        "passed": bool(passing),
        "passing_threshold_count": len(passing),
        "selected_threshold": float(selected["threshold"]),
        "selected_summary": selected,
    }, rows


def _mean_absolute(values: Iterable[float]) -> float:
    materialized = list(map(abs, map(float, values)))
    return statistics.fmean(materialized) if materialized else 0.0


def prediction_error_report(states: list[dict[str, Any]]) -> dict[str, Any]:
    arms = [arm for state in states for arm in _model_arms(state)]
    return _prediction_error_for_arms(arms)


def _prediction_error_for_arms(arms: list[dict[str, Any]]) -> dict[str, Any]:
    if not arms:
        raise ValueError("prediction error group is empty")
    effective_errors = [
        float(arm["predicted"]["effective_progress_probability"])
        - float(arm["actual"]["effective_rate"])
        for arm in arms
    ]
    no_progress_errors = [
        float(arm["predicted"]["no_progress_probability"])
        - float(arm["actual"]["no_progress_rate"])
        for arm in arms
    ]
    reduction_errors = [
        float(arm["predicted"]["conflict_reduction"])
        - float(arm["actual"]["conflict_reduction"])
        for arm in arms
    ]
    seconds_errors = [
        float(arm["predicted"]["repair_seconds"])
        - float(arm["actual"]["repair_seconds"])
        for arm in arms
    ]
    actual_seconds = [
        float(arm["actual"]["repair_seconds"]) for arm in arms
    ]
    return {
        "candidate_count": len(arms),
        "effective_probability_mae": _mean_absolute(effective_errors),
        "effective_probability_brier": statistics.fmean(
            value * value for value in effective_errors
        ),
        "no_progress_probability_mae": _mean_absolute(no_progress_errors),
        "no_progress_probability_brier": statistics.fmean(
            value * value for value in no_progress_errors
        ),
        "conflict_reduction_mae": _mean_absolute(reduction_errors),
        "conflict_reduction_mean_error": statistics.fmean(reduction_errors),
        "repair_seconds_mae": _mean_absolute(seconds_errors),
        "repair_seconds_mean_error": statistics.fmean(seconds_errors),
        "repair_seconds_relative_mae": statistics.fmean(
            abs(error) / max(1e-3, actual)
            for error, actual in zip(seconds_errors, actual_seconds)
        ),
    }


def prediction_error_group_rows(
    states: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(
        list
    )
    for state in states:
        for arm in _model_arms(state):
            groups[("overall", "all")].append(arm)
            groups[
                (
                    "layout_agent",
                    f"{state['layout_mode']}:{int(state['agent_count'])}",
                )
            ].append(arm)
            groups[("actual_size", str(int(arm["actual_size"])))].append(arm)
    return [
        {
            "split": split,
            "group_type": group_type,
            "group": group,
            **_prediction_error_for_arms(arms),
        }
        for (group_type, group), arms in sorted(groups.items())
    ]


def pairwise_prediction_quality(
    rows: list[dict[str, Any]],
    predictions: dict[tuple[str, str], float],
) -> dict[str, Any]:
    labels = [int(row["label"]) for row in rows]
    values = [
        float(predictions[(str(row["state_id"]), str(row["candidate_id"]))])
        for row in rows
    ]
    return {
        "pair_count": len(rows),
        "positive_fraction": statistics.fmean(map(float, labels)),
        "roc_auc": (
            _binary_roc_auc(labels, values)
            if len(set(labels)) > 1
            else None
        ),
        "brier": statistics.fmean(
            (prediction - label) ** 2
            for prediction, label in zip(values, labels)
        ),
        "accuracy_at_0_5": statistics.fmean(
            float((prediction >= 0.5) == bool(label))
            for prediction, label in zip(values, labels)
        ),
    }


def _binary_roc_auc(labels: list[int], scores: list[float]) -> float:
    if len(labels) != len(scores) or not labels:
        raise ValueError("ROC AUC labels and scores do not align")
    positives = sum(int(value == 1) for value in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both classes")
    ordered = sorted(
        ((float(score), int(label)) for label, score in zip(labels, scores)),
        key=lambda value: value[0],
    )
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[start][0]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        positive_rank_sum += average_rank * sum(
            label for _, label in ordered[start:stop]
        )
        start = stop
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def _miss_reason(
    state: dict[str, Any],
    oracle: dict[str, Any],
    thresholds: dict[str, float],
    calibration: dict[str, Any],
) -> str:
    arms = _model_arms(state)
    order = _v2_order(arms)
    base = arms[order[0]]
    margins = _margins(calibration, int(state["agent_count"]))
    candidate = _conservative_predictions(oracle, margins)
    baseline = _conservative_predictions(base, margins)
    reasons = []
    if candidate["effective_progress_probability"] < (
        baseline["effective_progress_probability"]
        - float(thresholds["effective_probability_tolerance"])
    ):
        reasons.append("effective_probability_guard")
    if candidate["no_progress_probability"] > (
        baseline["no_progress_probability"]
        + float(thresholds["no_progress_probability_tolerance"])
    ):
        reasons.append("no_progress_probability_guard")
    retention = float(thresholds["conflict_reduction_retention"])
    if candidate["conflict_reduction"] + 1e-12 < (
        retention * baseline["conflict_reduction"]
    ):
        reasons.append("predicted_reduction_guard")
    if baseline["lower_conflict_reduction"] > 0.0 and (
        candidate["lower_conflict_reduction"] + 1e-12
        < retention * baseline["lower_conflict_reduction"]
    ):
        reasons.append("conservative_reduction_guard")
    if candidate["upper_repair_seconds"] > (
        (1.0 - float(thresholds["minimum_time_improvement"]))
        * baseline["upper_repair_seconds"]
        + 1e-12
    ):
        reasons.append("predicted_time_guard")
    if candidate["lower_utility"] + 1e-12 < (
        (1.0 + float(thresholds["minimum_utility_improvement"]))
        * baseline["lower_utility"]
    ):
        reasons.append("predicted_utility_guard")
    return "+".join(reasons) if reasons else "eligible_but_lower_predicted_rank"


def miss_attribution(
    states: list[dict[str, Any]],
    thresholds: dict[str, float],
    calibration: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    for state in states:
        selected, _ = select_cost_tiebreak(state, thresholds, calibration)
        oracle = _oracle_top3(state)
        base = next(
            arm for arm in _model_arms(state) if bool(arm["base_selected"])
        )
        selected_id = str(selected["candidate_id"])
        oracle_id = str(oracle["candidate_id"])
        base_id = str(base["candidate_id"])
        if selected_id == oracle_id:
            category = "policy_matches_oracle"
            reason = "match"
        elif oracle_id == base_id:
            category = "policy_override_when_oracle_base"
            reason = "unnecessary_override"
        elif selected_id == base_id:
            category = "policy_kept_base_missed_oracle"
            reason = _miss_reason(state, oracle, thresholds, calibration)
        else:
            category = "policy_selected_different_override"
            reason = _miss_reason(state, oracle, thresholds, calibration)
        rows.append(
            {
                "state_id": str(state["state_id"]),
                "map_id": str(state["map_id"]),
                "layout_mode": str(state["layout_mode"]),
                "agent_count": int(state["agent_count"]),
                "category": category,
                "reason": reason,
                "base_candidate_id": base_id,
                "selected_candidate_id": selected_id,
                "oracle_candidate_id": oracle_id,
            }
        )
    blockers: collections.Counter[str] = collections.Counter()
    for row in rows:
        for blocker in str(row["reason"]).split("+"):
            if blocker not in {"match", "unnecessary_override"}:
                blockers[blocker] += 1
    return {
        "state_count": len(rows),
        "category_counts": dict(
            sorted(collections.Counter(row["category"] for row in rows).items())
        ),
        "reason_counts": dict(
            sorted(collections.Counter(row["reason"] for row in rows).items())
        ),
        "blocker_counts": dict(sorted(blockers.items())),
    }, rows


def _target_states(
    state_rows: list[dict[str, Any]],
    miss_rows: list[dict[str, Any]],
    *,
    maximum: int = 50,
) -> list[dict[str, Any]]:
    miss = {str(row["state_id"]): row for row in miss_rows}
    candidates = []
    for row in state_rows:
        state_id = str(row["state_id"])
        miss_row = miss[state_id]
        unstable = int(
            not bool(row["oracle_candidate_agreement"])
            or int(row["disagreeing_pair_count"]) > 0
        )
        prediction_miss = int(
            str(miss_row["category"]) != "policy_matches_oracle"
        )
        if unstable or prediction_miss:
            candidates.append(
                {
                    **row,
                    "prediction_miss": prediction_miss,
                    "miss_category": str(miss_row["category"]),
                    "miss_reason": str(miss_row["reason"]),
                    "priority": 2 * unstable + prediction_miss,
                }
            )
    return sorted(
        candidates,
        key=lambda row: (
            -int(row["priority"]),
            -int(row["agent_count"]),
            str(row["map_id"]),
            str(row["state_id"]),
        ),
    )[:maximum]


def _render_report(report: dict[str, Any]) -> str:
    train_seed = report["seed_stability"]["policy_train"]
    diagnostic_seed = report["seed_stability"]["policy_validation"]
    pairwise = report["pairwise_oof"]
    diagnostic = report["pairwise_diagnostic"]
    return "\n".join(
        [
            "# V2 Top-3 prediction error and PP-seed audit",
            "",
            f"Decision: `{report['decision']}`",
            "",
            "## Paired-seed stability",
            "",
            f"- Train pair-label agreement: {train_seed['pair_label_agreement_fraction']:.3%}",
            f"- Train state Oracle exact agreement: {train_seed['state_oracle_candidate_agreement_fraction']:.3%}",
            f"- Diagnostic pair-label agreement: {diagnostic_seed['pair_label_agreement_fraction']:.3%}",
            f"- Diagnostic state Oracle exact agreement: {diagnostic_seed['state_oracle_candidate_agreement_fraction']:.3%}",
            "",
            "## Direct pairwise OOF",
            "",
            f"- Threshold: {pairwise['threshold']:.2f}",
            f"- Conflict-reduction ratio: {pairwise['metrics']['comparison']['conflict_reduction_ratio']:.4f}",
            f"- Repair-time ratio: {pairwise['metrics']['comparison']['repair_time_ratio']:.4f}",
            f"- Efficiency ratio: {pairwise['metrics']['comparison']['efficiency_ratio']:.4f}",
            "",
            "## Diagnostic maps",
            "",
            f"- Conflict-reduction ratio: {diagnostic['metrics']['comparison']['conflict_reduction_ratio']:.4f}",
            f"- Repair-time ratio: {diagnostic['metrics']['comparison']['repair_time_ratio']:.4f}",
            f"- Efficiency ratio: {diagnostic['metrics']['comparison']['efficiency_ratio']:.4f}",
            "",
            "This is an offline one-repair audit. It does not establish complete-episode wall-clock improvement.",
            "",
        ]
    )


def audit_v2_cost_tiebreak_errors(
    *, source: Path, policy_audit: Path, output: Path
) -> dict[str, Any]:
    source = Path(source).resolve()
    policy_audit = Path(policy_audit).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("cost-tiebreak error-audit output is non-empty")
    feature_path = source / "collection" / "feature_index.jsonl"
    trial_path = source / "collection" / "trial_manifest.jsonl"
    policy_report_path = policy_audit / "v2_cost_tiebreak_audit_report.json"
    feature_rows = _read_jsonl(feature_path)
    trial_rows = _read_jsonl(trial_path)
    import json

    policy_report = json.loads(policy_report_path.read_text(encoding="utf-8"))
    thresholds = {
        key: float(value)
        for key, value in dict(policy_report["selected_thresholds"]).items()
    }
    coverage = validate_paired_trials(feature_rows, trial_rows)

    fold_rows = []
    oof_pair_predictions: dict[tuple[str, str], float] = {}
    train_raw = _rows(feature_rows, trial_rows, "policy_train", V3_FEATURE_NAMES)
    diagnostic_raw = _rows(
        feature_rows, trial_rows, "policy_validation", V3_FEATURE_NAMES
    )
    train_maps = {str(row["map_id"]) for row in train_raw}
    diagnostic_maps = {str(row["map_id"]) for row in diagnostic_raw}
    if train_maps & diagnostic_maps:
        raise ValueError("training and diagnostic maps overlap")

    oof_head_predictions: dict[tuple[str, str], dict[str, list[float]]] = {}
    for fold in _balanced_map_folds(train_raw):
        held_maps = set(fold["validation_maps"])
        training = [row for row in train_raw if row["map_id"] not in held_maps]
        held = [row for row in train_raw if row["map_id"] in held_maps]
        head_models = _fit(training)
        head_prediction = _with_runtime_predictions(
            _predict(head_models, held), 0.0
        )
        indexed = _index_predictions(held, head_prediction)
        if set(indexed) & set(oof_head_predictions):
            raise ValueError("head OOF predictions overlap")
        oof_head_predictions.update(indexed)
        held_states = _states(held, indexed)
        held_pairs, _ = build_pair_rows(
            held_states, held, trial_rows, "policy_train"
        )
        train_pair_source = [
            row for row in train_raw if row["map_id"] not in held_maps
        ]
        training_state_predictions = _with_runtime_predictions(
            _predict(head_models, train_pair_source), 0.0
        )
        training_states = _states(
            train_pair_source,
            _index_predictions(train_pair_source, training_state_predictions),
        )
        training_pairs, _ = build_pair_rows(
            training_states, train_pair_source, trial_rows, "policy_train"
        )
        pair_model = _fit_pairwise(training_pairs)
        pair_predictions = _predict_pairwise(pair_model, held_pairs)
        indexed_pairs = _pair_prediction_index(held_pairs, pair_predictions)
        if set(indexed_pairs) & set(oof_pair_predictions):
            raise ValueError("pairwise OOF predictions overlap")
        oof_pair_predictions.update(indexed_pairs)
        fold_rows.append(
            {
                "fold": int(fold["fold"]),
                "training_map_count": len(fold["train_maps"]),
                "validation_map_count": len(fold["validation_maps"]),
                "training_pair_count": len(training_pairs),
                "validation_pair_count": len(held_pairs),
                "positive_training_pair_count": sum(
                    int(row["label"]) for row in training_pairs
                ),
            }
        )

    train_states = _states(train_raw, oof_head_predictions)
    train_pairs, train_state_seed_rows = build_pair_rows(
        train_states, train_raw, trial_rows, "policy_train"
    )
    if set(oof_pair_predictions) != {
        (str(row["state_id"]), str(row["candidate_id"]))
        for row in train_pairs
    }:
        raise ValueError("pairwise OOF coverage is incomplete")
    calibration = _uncertainty_calibration(train_states)
    pair_calibration, threshold_rows = calibrate_pairwise_policy(
        train_states, oof_pair_predictions
    )
    threshold = float(pair_calibration["selected_threshold"])
    train_pair_metrics, train_pair_selection_rows = evaluate_pairwise_policy(
        train_states, oof_pair_predictions, threshold
    )
    train_pair_checks = _training_checks(train_pair_metrics)

    final_head_models = _fit(train_raw)
    diagnostic_head_predictions = _with_runtime_predictions(
        _predict(final_head_models, diagnostic_raw), 0.0
    )
    diagnostic_states = _states(
        diagnostic_raw,
        _index_predictions(diagnostic_raw, diagnostic_head_predictions),
    )
    diagnostic_pairs, diagnostic_state_seed_rows = build_pair_rows(
        diagnostic_states, diagnostic_raw, trial_rows, "policy_validation"
    )
    final_pair_model = _fit_pairwise(train_pairs)
    diagnostic_pair_predictions = _pair_prediction_index(
        diagnostic_pairs,
        _predict_pairwise(final_pair_model, diagnostic_pairs),
    )
    diagnostic_pair_metrics, diagnostic_pair_selection_rows = (
        evaluate_pairwise_policy(
            diagnostic_states, diagnostic_pair_predictions, threshold
        )
    )
    diagnostic_pair_checks = _diagnostic_checks(diagnostic_pair_metrics)
    train_pair_prediction_quality = pairwise_prediction_quality(
        train_pairs, oof_pair_predictions
    )
    diagnostic_pair_prediction_quality = pairwise_prediction_quality(
        diagnostic_pairs, diagnostic_pair_predictions
    )

    train_miss, train_miss_rows = miss_attribution(
        train_states, thresholds, calibration
    )
    diagnostic_miss, diagnostic_miss_rows = miss_attribution(
        diagnostic_states, thresholds, calibration
    )
    train_seed = seed_stability_report(train_pairs, train_state_seed_rows)
    diagnostic_seed = seed_stability_report(
        diagnostic_pairs, diagnostic_state_seed_rows
    )
    seed_checks = {
        "training_pair_agreement_at_least_80pct": (
            train_seed["pair_label_agreement_fraction"] >= 0.80
        ),
        "diagnostic_pair_agreement_at_least_80pct": (
            diagnostic_seed["pair_label_agreement_fraction"] >= 0.80
        ),
        "training_oracle_route_agreement_at_least_80pct": (
            train_seed["state_oracle_route_agreement_fraction"] >= 0.80
        ),
        "diagnostic_oracle_route_agreement_at_least_80pct": (
            diagnostic_seed["state_oracle_route_agreement_fraction"] >= 0.80
        ),
    }
    pairwise_passed = all(train_pair_checks.values()) and all(
        diagnostic_pair_checks.values()
    )
    stable = all(seed_checks.values())
    if pairwise_passed and stable:
        decision = "targeted_four_seed_pilot_justified"
    elif not stable:
        decision = "seed_variance_blocks_pairwise_promotion"
    elif not all(train_pair_checks.values()):
        decision = "pairwise_prediction_insufficient"
    else:
        decision = "pairwise_diagnostic_generalization_insufficient"

    target_rows = _target_states(
        diagnostic_state_seed_rows,
        diagnostic_miss_rows,
        maximum=50,
    )
    report = {
        "schema": V2_COST_TIEBREAK_ERROR_AUDIT_SCHEMA,
        "decision": decision,
        "diagnostic_only": True,
        "coverage": {
            **coverage,
            "training_state_count": len(train_states),
            "diagnostic_state_count": len(diagnostic_states),
            "training_pair_count": len(train_pairs),
            "diagnostic_pair_count": len(diagnostic_pairs),
            "pairwise_feature_count": PAIRWISE_FEATURE_COUNT,
        },
        "seed_stability": {
            "policy_train": train_seed,
            "policy_validation": diagnostic_seed,
            "checks": seed_checks,
        },
        "prediction_errors": {
            "policy_train_oof": prediction_error_report(train_states),
            "policy_validation": prediction_error_report(diagnostic_states),
        },
        "pairwise_prediction_quality": {
            "policy_train_oof": train_pair_prediction_quality,
            "policy_validation": diagnostic_pair_prediction_quality,
        },
        "existing_policy_miss_attribution": {
            "policy_train_oof": train_miss,
            "policy_validation": diagnostic_miss,
        },
        "pairwise_calibration": pair_calibration,
        "pairwise_oof": {
            "threshold": threshold,
            "metrics": train_pair_metrics,
            "checks": train_pair_checks,
        },
        "pairwise_diagnostic": {
            "threshold": threshold,
            "metrics": diagnostic_pair_metrics,
            "checks": diagnostic_pair_checks,
        },
        "candidate_four_seed_target_count": len(target_rows),
        "four_seed_collection_justified": bool(
            decision == "targeted_four_seed_pilot_justified"
        ),
        "four_seed_collection_started": False,
        "folds": fold_rows,
        "source": {
            "path": str(source),
            "feature_index_sha256": sha256_file(feature_path),
            "trial_manifest_sha256": sha256_file(trial_path),
            "policy_report_sha256": sha256_file(policy_report_path),
        },
        "limitations": [
            "The policy-validation maps were inspected previously and remain diagnostic.",
            "Most candidates have two paired PP trials; 219 candidates have four.",
            "Oracle and mean-label comparisons are one-repair diagnostics, not end-to-end solver results.",
            "No additional solver execution or four-seed collection was started.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(output / "map_fold_results.csv", fold_rows)
    atomic_write_csv(
        output / "prediction_error_by_group.csv",
        prediction_error_group_rows(train_states, "policy_train_oof")
        + prediction_error_group_rows(diagnostic_states, "policy_validation"),
    )
    atomic_write_csv(output / "pairwise_threshold_grid.csv", threshold_rows)
    atomic_write_csv(
        output / "training_seed_pair_stability.csv",
        [
            {key: value for key, value in row.items() if key != "features"}
            for row in train_pairs
        ],
    )
    atomic_write_csv(
        output / "diagnostic_seed_pair_stability.csv",
        [
            {key: value for key, value in row.items() if key != "features"}
            for row in diagnostic_pairs
        ],
    )
    atomic_write_csv(output / "training_state_seed_stability.csv", train_state_seed_rows)
    atomic_write_csv(
        output / "diagnostic_state_seed_stability.csv", diagnostic_state_seed_rows
    )
    atomic_write_csv(
        output / "training_pairwise_selections.csv", train_pair_selection_rows
    )
    atomic_write_csv(
        output / "diagnostic_pairwise_selections.csv",
        diagnostic_pair_selection_rows,
    )
    atomic_write_csv(output / "training_miss_attribution.csv", train_miss_rows)
    atomic_write_csv(
        output / "diagnostic_miss_attribution.csv", diagnostic_miss_rows
    )
    if target_rows:
        atomic_write_csv(output / "candidate_four_seed_targets.csv", target_rows)
    _write_json(output / "v2_cost_tiebreak_error_audit_report.json", report)
    (output / "v2_cost_tiebreak_error_audit_report.md").write_text(
        _render_report(report), encoding="utf-8", newline="\n"
    )
    return report
