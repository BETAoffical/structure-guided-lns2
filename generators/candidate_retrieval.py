from __future__ import annotations

import collections
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable


K_OPTIONS = (3, 5, 7, 11)
GROUP_WEIGHT_OPTIONS = (
    {"map": 1.0, "task": 1.0, "state": 1.0, "candidate": 1.0},
    {"map": 1.0, "task": 1.0, "state": 1.0, "candidate": 2.0},
    {"map": 1.0, "task": 1.0, "state": 2.0, "candidate": 1.0},
    {"map": 0.5, "task": 0.5, "state": 1.0, "candidate": 2.0},
    {"map": 0.5, "task": 0.5, "state": 1.5, "candidate": 3.0},
)
MARGIN_OPTIONS = (0.0, 0.25, 0.5, 1.0)
FEATURE_PROFILES = {
    "full": None,
    "dedup20": (
        "map.shelf_coverage",
        "map.gate_ratio",
        "map.dead_end_ratio",
        "map.maximum_prior",
        "task.agent_count",
        "task.density_service",
        "task.hotspot_skew",
        "task.mean_shortest_distance",
        "state.conflict_density",
        "state.maximum_conflict_degree",
        "state.vertex_ratio",
        "state.mean_conflict_time",
        "candidate.conflict_edge_coverage",
        "candidate.induced_edge_density",
        "candidate.path_conflict_overlap",
        "candidate.temporal_overlap",
        "candidate.start_goal_blocker_ratio",
        "candidate.mean_path_stretch",
        "candidate.mean_structural_prior",
        "candidate.cross_zone_ratio",
    ),
    "core12": (
        "map.gate_ratio",
        "map.dead_end_ratio",
        "map.maximum_prior",
        "task.agent_count",
        "task.hotspot_skew",
        "task.mean_shortest_distance",
        "state.conflict_density",
        "state.maximum_conflict_degree",
        "state.mean_conflict_time",
        "candidate.conflict_edge_coverage",
        "candidate.path_conflict_overlap",
        "candidate.temporal_overlap",
        "candidate.mean_path_stretch",
    ),
    "rollout22": (
        "map.shelf_coverage",
        "map.gate_ratio",
        "map.dead_end_ratio",
        "map.maximum_prior",
        "task.agent_count",
        "task.density_service",
        "task.hotspot_skew",
        "task.mean_shortest_distance",
        "state.conflict_density",
        "state.maximum_conflict_degree",
        "state.vertex_ratio",
        "state.mean_conflict_time",
        "candidate.conflict_edge_coverage",
        "candidate.induced_edge_density",
        "candidate.path_conflict_overlap",
        "candidate.temporal_overlap",
        "candidate.start_goal_blocker_ratio",
        "candidate.mean_path_stretch",
        "candidate.mean_structural_prior",
        "candidate.cross_zone_ratio",
        "candidate.one_step_conflict_reduction",
        "rollout.horizon",
    ),
}


def _feature_names(
    cases: list[dict[str, Any]], feature_profile: str
) -> list[str]:
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(f"unknown candidate feature profile: {feature_profile}")
    all_names = sorted(
        set(
            itertools.chain.from_iterable(
                case["features"].keys() for case in cases
            )
        )
    )
    selected = FEATURE_PROFILES[feature_profile]
    if selected is None:
        return all_names
    return [name for name in selected if name in set(all_names)]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path, rows: Iterable[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def fit_normalizer(
    cases: list[dict[str, Any]],
    feature_profile: str = "full",
) -> dict[str, Any]:
    names = _feature_names(cases, feature_profile)
    entries = []
    for name in names:
        values = [float(case["features"].get(name, 0.0)) for case in cases]
        mean = sum(values) / max(1, len(values))
        variance = sum((value - mean) ** 2 for value in values) / max(
            1, len(values)
        )
        standard_deviation = math.sqrt(variance)
        if standard_deviation <= 1e-12:
            continue
        entries.append(
            {
                "name": name,
                "group": name.split(".", 1)[0],
                "mean": mean,
                "standard_deviation": standard_deviation,
            }
        )
    if not entries:
        raise ValueError("all candidate features have zero variance")
    return {
        "fit_split": "train",
        "feature_profile": feature_profile,
        "available_feature_profiles": sorted(FEATURE_PROFILES),
        "feature_count_before_filter": len(names),
        "feature_count": len(entries),
        "zero_variance_features": [
            name
            for name in names
            if name not in {entry["name"] for entry in entries}
        ],
        "features": entries,
    }


def vectorize(
    features: dict[str, float], normalizer: dict[str, Any]
) -> list[float]:
    return [
        (
            float(features.get(entry["name"], 0.0))
            - float(entry["mean"])
        )
        / float(entry["standard_deviation"])
        for entry in normalizer["features"]
    ]


def grouped_distance(
    left: list[float],
    right: list[float],
    normalizer: dict[str, Any],
    group_weights: dict[str, float],
) -> float:
    weighted = 0.0
    weight_sum = 0.0
    for index, entry in enumerate(normalizer["features"]):
        weight = float(group_weights.get(entry["group"], 1.0))
        weighted += weight * (left[index] - right[index]) ** 2
        weight_sum += weight
    return math.sqrt(weighted / max(1e-12, weight_sum))


def rank_neighbors(
    query: list[float],
    entries: list[dict[str, Any]],
    normalizer: dict[str, Any],
    group_weights: dict[str, float],
    k: int,
    exclude_state: str | None = None,
) -> list[tuple[dict[str, Any], float]]:
    ranked = sorted(
        (
            (
                entry,
                grouped_distance(
                    query,
                    entry["vector"],
                    normalizer,
                    group_weights,
                ),
            )
            for entry in entries
            if entry["state_id"] != exclude_state
        ),
        key=lambda item: (item[1], str(item[0]["case_id"])),
    )
    selected: list[tuple[dict[str, Any], float]] = []
    states: set[str] = set()
    task_counts: collections.Counter[str] = collections.Counter()
    for entry, distance in ranked:
        state_id = str(entry["state_id"])
        task_id = str(entry["task_id"])
        if state_id in states or task_counts[task_id] >= 2:
            continue
        selected.append((entry, distance))
        states.add(state_id)
        task_counts[task_id] += 1
        if len(selected) >= k:
            break
    return selected


def _weighted_mean(
    neighbors: list[tuple[dict[str, Any], float]],
    value,
) -> float:
    if not neighbors:
        return 0.0
    weights = [1.0 / max(1e-6, distance) for _, distance in neighbors]
    return sum(
        weight * float(value(entry))
        for weight, (entry, _) in zip(weights, neighbors)
    ) / sum(weights)


def prediction_from_neighbors(
    neighbors: list[tuple[dict[str, Any], float]],
) -> dict[str, Any]:
    valid_probability = _weighted_mean(
        neighbors,
        lambda entry: entry["outcome"].get(
            "valid_probability",
            entry["outcome"]["candidate_valid"],
        ),
    )
    conflict_reduction = _weighted_mean(
        neighbors,
        lambda entry: (
            entry["outcome"]["conflict_reduction"]
            if entry["outcome"]["candidate_valid"]
            else 0.0
        ),
    )
    cost_improvement = _weighted_mean(
        neighbors,
        lambda entry: (
            entry["outcome"]["cost_improvement"]
            if entry["outcome"]["candidate_valid"]
            else 0.0
        ),
    )
    runtime_ms = _weighted_mean(
        neighbors,
        lambda entry: entry["outcome"].get(
            "total_runtime_ms",
            entry["outcome"]["replan_runtime_ms"],
        ),
    )
    utility = (
        4.0 * (valid_probability - 0.5)
        + 2.0 * conflict_reduction
        + 0.02 * cost_improvement
        - 0.0001 * runtime_ms
    )
    return {
        "valid_probability": valid_probability,
        "conflict_reduction": conflict_reduction,
        "cost_improvement": cost_improvement,
        "runtime_ms": runtime_ms,
        "utility": utility,
        "nearest_distance": (
            neighbors[0][1] if neighbors else math.inf
        ),
        "neighbors": [
            {
                "case_id": entry["case_id"],
                "state_id": entry["state_id"],
                "task_id": entry["task_id"],
                "distance": distance,
            }
            for entry, distance in neighbors
        ],
    }


def predict_candidate(
    query: list[float],
    entries: list[dict[str, Any]],
    normalizer: dict[str, Any],
    group_weights: dict[str, float],
    k: int,
) -> dict[str, Any]:
    return prediction_from_neighbors(
        rank_neighbors(
            query, entries, normalizer, group_weights, k
        )
    )


def actual_utility(outcome: dict[str, Any]) -> float:
    if "valid_probability" in outcome:
        solved_bonus = 4.0 * float(outcome.get("solved_probability", 0.0))
        return (
            solved_bonus
            + 4.0 * (float(outcome["valid_probability"]) - 0.5)
            + 2.0 * float(outcome["conflict_reduction"])
            + 0.02 * float(outcome["cost_improvement"])
            - min(
                2000.0,
                float(
                    outcome.get(
                        "total_runtime_ms",
                        outcome["replan_runtime_ms"],
                    )
                ),
            )
            / 20000.0
        )
    if not outcome["candidate_valid"]:
        return -4.0
    return (
        2.0 * float(outcome["conflict_reduction"])
        + 0.02 * float(outcome["cost_improvement"])
        - min(
            2000.0,
            float(
                outcome.get(
                    "total_runtime_ms",
                    outcome["replan_runtime_ms"],
                )
            ),
        )
        / 20000.0
    )


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[max(0, min(index, len(ordered) - 1))]


def ood_threshold(
    entries: list[dict[str, Any]],
    normalizer: dict[str, Any],
    group_weights: dict[str, float],
) -> float:
    distances = []
    by_candidate: collections.defaultdict[
        int, list[dict[str, Any]]
    ] = collections.defaultdict(list)
    for entry in entries:
        by_candidate[int(entry["candidate_index"])].append(entry)
    for entry in entries:
        nearest = min(
            (
                grouped_distance(
                    entry["vector"],
                    other["vector"],
                    normalizer,
                    group_weights,
                )
                for other in by_candidate[
                    int(entry["candidate_index"])
                ]
                if other["state_id"] != entry["state_id"]
            ),
            default=math.inf,
        )
        if math.isfinite(nearest):
            distances.append(nearest)
    return _percentile(distances, 0.95)


def build_candidate_index(
    memory: str | Path,
    output: str | Path,
    feature_profile: str = "full",
) -> dict[str, Any]:
    memory_root = Path(memory).resolve()
    output_root = Path(output).resolve()
    summary = _read_json(memory_root / "candidate_summary.json")
    cases = _read_jsonl(memory_root / "candidate_cases.jsonl")
    if (
        summary.get("split") != "train"
        or summary.get("usage") != "memory"
        or any(case.get("split") != "train" for case in cases)
    ):
        raise ValueError("candidate index may only use Train memory")
    normalizer = fit_normalizer(cases, feature_profile)
    entries = [
        {
            "case_id": case["case_id"],
            "state_id": case["state_id"],
            "run_id": case["run_id"],
            "map_id": case["map_id"],
            "task_id": case["task_id"],
            "solver_seed": case["solver_seed"],
            "candidate_index": case["candidate_index"],
            "vector": vectorize(case["features"], normalizer),
            "outcome": case["outcome"],
        }
        for case in cases
    ]
    entries.sort(key=lambda entry: str(entry["case_id"]))
    _write_json(output_root / "normalizer.json", normalizer)
    _write_jsonl(output_root / "candidate_index.jsonl", entries)
    result = {
        "schema_version": 1,
        "fit_split": "train",
        "case_count": len(entries),
        "state_count": len({entry["state_id"] for entry in entries}),
        "map_count": len({entry["map_id"] for entry in entries}),
        "task_count": len({entry["task_id"] for entry in entries}),
        "feature_profile": normalizer["feature_profile"],
        "feature_count": normalizer["feature_count"],
        "feature_count_before_filter": normalizer[
            "feature_count_before_filter"
        ],
        "feature_names": [
            entry["name"] for entry in normalizer["features"]
        ],
        "excluded_feature_classes": [
            "agent_id",
            "absolute_coordinates",
            "generator_name",
            "post_repair_paths",
            "outcome_labels",
        ],
    }
    _write_json(output_root / "index_summary.json", result)
    return result


def _state_metrics(
    predictions: list[tuple[dict[str, Any], dict[str, Any]]],
    margin: float,
    threshold: float,
) -> dict[str, Any]:
    by_index = {
        int(case["candidate_index"]): (case, prediction)
        for case, prediction in predictions
    }
    baseline_case, baseline_prediction = by_index[0]
    alternatives = [
        value for index, value in by_index.items() if index != 0
    ]
    best_case, best_prediction = max(
        alternatives,
        key=lambda value: (
            value[1]["utility"],
            -int(value[0]["candidate_index"]),
        ),
    )
    reason = ""
    if best_prediction["nearest_distance"] > threshold:
        reason = "out_of_distribution"
    elif best_prediction["valid_probability"] < 0.5:
        reason = "low_valid_probability"
    elif (
        best_prediction["utility"] - baseline_prediction["utility"]
        < margin
    ):
        reason = "insufficient_margin"
    selected_case = baseline_case if reason else best_case
    actual_values = {
        int(case["candidate_index"]): actual_utility(case["outcome"])
        for case, _ in predictions
    }
    predicted_values = {
        int(case["candidate_index"]): prediction["utility"]
        for case, prediction in predictions
    }
    oracle_index = max(
        actual_values,
        key=lambda index: (actual_values[index], -index),
    )
    predicted_index = max(
        predicted_values,
        key=lambda index: (predicted_values[index], -index),
    )
    pair_correct = 0
    pair_total = 0
    for left, right in itertools.combinations(
        sorted(actual_values), 2
    ):
        actual_delta = actual_values[left] - actual_values[right]
        predicted_delta = (
            predicted_values[left] - predicted_values[right]
        )
        if actual_delta == 0:
            continue
        pair_total += 1
        pair_correct += (actual_delta > 0) == (predicted_delta > 0)
    selected_index = int(selected_case["candidate_index"])
    return {
        "state_id": baseline_case["state_id"],
        "selected_candidate_index": selected_index,
        "predicted_best_candidate_index": predicted_index,
        "oracle_candidate_index": oracle_index,
        "fallback_reason": reason,
        "used_guidance": selected_index != 0,
        "selected_actual_utility": actual_values[selected_index],
        "baseline_actual_utility": actual_values[0],
        "oracle_actual_utility": actual_values[oracle_index],
        "top1_exact": predicted_index == oracle_index,
        "baseline_is_oracle": oracle_index == 0,
        "pair_correct": pair_correct,
        "pair_total": pair_total,
        "predictions": [
            {
                "candidate_index": int(case["candidate_index"]),
                **prediction,
                "actual_utility": actual_values[
                    int(case["candidate_index"])
                ],
            }
            for case, prediction in sorted(
                predictions,
                key=lambda value: int(value[0]["candidate_index"]),
            )
        ],
    }


def _aggregate_metrics(states: list[dict[str, Any]]) -> dict[str, Any]:
    count = max(1, len(states))
    return {
        "state_count": len(states),
        "top1_gain": sum(
            state["selected_actual_utility"]
            - state["baseline_actual_utility"]
            for state in states
        )
        / count,
        "baseline_win_rate": sum(
            state["baseline_is_oracle"] for state in states
        )
        / count,
        "oracle_regret": sum(
            state["oracle_actual_utility"]
            - state["selected_actual_utility"]
            for state in states
        )
        / count,
        "top1_accuracy": sum(
            state["top1_exact"] for state in states
        )
        / count,
        "ranking_accuracy": sum(
            state["pair_correct"] for state in states
        )
        / max(1, sum(state["pair_total"] for state in states)),
        "guidance_use_rate": sum(
            state["used_guidance"] for state in states
        )
        / count,
    }


def evaluate_candidate_retrieval(
    index: str | Path,
    queries: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    index_root = Path(index).resolve()
    query_root = Path(queries).resolve()
    output_root = Path(output).resolve()
    query_summary = _read_json(
        query_root / "candidate_summary.json"
    )
    cases = _read_jsonl(query_root / "candidate_cases.jsonl")
    if (
        query_summary.get("split") != "validation"
        or query_summary.get("usage") != "evaluation"
        or any(case.get("split") != "validation" for case in cases)
    ):
        raise ValueError("candidate tuning requires Validation queries")
    normalizer = _read_json(index_root / "normalizer.json")
    if normalizer.get("fit_split") != "train":
        raise ValueError("candidate normalizer was not fit on Train")
    feature_profile = str(normalizer.get("feature_profile", "full"))
    entries = _read_jsonl(
        index_root / "candidate_index.jsonl"
    )
    grouped: collections.defaultdict[str, list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for case in cases:
        grouped[str(case["state_id"])].append(case)
    for state_id, values in grouped.items():
        if len(values) < 2:
            raise ValueError(
                f"Validation state {state_id} has fewer than two candidates"
            )
        indices = sorted(int(row["candidate_index"]) for row in values)
        if indices != list(range(len(values))):
            raise ValueError(
                f"Validation state {state_id} candidate indices are invalid"
            )
        values.sort(key=lambda row: int(row["candidate_index"]))

    evaluations = []
    best_key: tuple[float, ...] | None = None
    selected: dict[str, Any] | None = None
    selected_states: list[dict[str, Any]] = []
    for weights in GROUP_WEIGHT_OPTIONS:
        threshold = ood_threshold(entries, normalizer, weights)
        neighbor_cache = {
            str(case["case_id"]): rank_neighbors(
                vectorize(case["features"], normalizer),
                entries,
                normalizer,
                weights,
                max(K_OPTIONS),
            )
            for case in cases
        }
        for k in K_OPTIONS:
            state_predictions = {}
            for state_id, state_cases in sorted(grouped.items()):
                state_predictions[state_id] = [
                    (
                        case,
                        prediction_from_neighbors(
                            neighbor_cache[str(case["case_id"])][
                                :k
                            ]
                        ),
                    )
                    for case in state_cases
                ]
            for margin in MARGIN_OPTIONS:
                states = [
                    _state_metrics(
                        state_predictions[state_id],
                        margin,
                        threshold,
                    )
                    for state_id in sorted(state_predictions)
                ]
                metrics = _aggregate_metrics(states)
                row = {
                    "k": k,
                    "group_weights": weights,
                    "minimum_margin": margin,
                    "ood_distance_threshold": threshold,
                    "metrics": metrics,
                }
                evaluations.append(row)
                key = (
                    metrics["top1_gain"],
                    -metrics["oracle_regret"],
                    metrics["ranking_accuracy"],
                    metrics["top1_accuracy"],
                    -margin,
                    -k,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    selected = row
                    selected_states = states
    assert selected is not None
    _write_jsonl(
        output_root / "candidate_guidance.jsonl", selected_states
    )
    selected_config = {
        "schema_version": 1,
        "selected_on_split": "validation",
        "test_data_read": False,
        "feature_profile": feature_profile,
        "k": selected["k"],
        "group_weights": selected["group_weights"],
        "minimum_margin": selected["minimum_margin"],
        "ood_distance_threshold": selected[
            "ood_distance_threshold"
        ],
        "minimum_valid_probability": 0.5,
    }
    _write_json(output_root / "selected_config.json", selected_config)
    summary = {
        "schema_version": 1,
        "index_split": "train",
        "query_split": "validation",
        "test_data_read": False,
        "feature_profile": feature_profile,
        "feature_count": normalizer["feature_count"],
        "configuration_count": len(evaluations),
        "selected_parameters": selected_config,
        "selected_metrics": selected["metrics"],
        "all_configurations": evaluations,
    }
    _write_json(output_root / "evaluation_summary.json", summary)
    return summary
