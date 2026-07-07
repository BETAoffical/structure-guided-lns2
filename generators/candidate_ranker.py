from __future__ import annotations

import collections
import itertools
import json
import math
import pickle
from pathlib import Path
from typing import Any, Iterable

from .candidate_retrieval import (
    FEATURE_PROFILES,
    actual_utility,
    fit_normalizer,
    vectorize,
)


MARGIN_OPTIONS = (0.0, 0.25, 0.5, 1.0)
LINEAR_EPOCHS = 80
LINEAR_LEARNING_RATE = 0.05
LINEAR_L2 = 0.0001


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


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sigmoid(value: float) -> float:
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _subtract(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _group_by_state(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: collections.defaultdict[str, list[dict[str, Any]]] = (
        collections.defaultdict(list)
    )
    for case in cases:
        grouped[str(case["state_id"])].append(case)
    for state_id, rows in grouped.items():
        if len(rows) != 8:
            raise ValueError(f"state {state_id} does not contain eight candidates")
        indices = sorted(int(row["candidate_index"]) for row in rows)
        if indices != list(range(8)):
            raise ValueError(f"state {state_id} candidate indices are invalid")
        rows.sort(key=lambda row: int(row["candidate_index"]))
    return dict(sorted(grouped.items()))


def _prepare_cases(
    cases: list[dict[str, Any]],
    normalizer: dict[str, Any],
) -> list[dict[str, Any]]:
    prepared = []
    for case in cases:
        prepared.append(
            {
                **case,
                "vector": vectorize(case["features"], normalizer),
                "utility": actual_utility(case["outcome"]),
            }
        )
    return prepared


def _pairwise_examples(
    grouped: dict[str, list[dict[str, Any]]],
) -> list[tuple[list[float], float]]:
    examples = []
    for state_cases in grouped.values():
        for left, right in itertools.combinations(state_cases, 2):
            delta = float(left["utility"]) - float(right["utility"])
            if abs(delta) <= 1e-12:
                continue
            if delta > 0:
                examples.append((_subtract(left["vector"], right["vector"]), 1.0))
                examples.append((_subtract(right["vector"], left["vector"]), -1.0))
            else:
                examples.append((_subtract(right["vector"], left["vector"]), 1.0))
                examples.append((_subtract(left["vector"], right["vector"]), -1.0))
    examples.sort(key=lambda item: tuple(round(value, 12) for value in item[0]))
    return examples


def _train_pairwise_linear(
    grouped: dict[str, list[dict[str, Any]]],
    feature_count: int,
) -> dict[str, Any]:
    weights = [0.0] * feature_count
    examples = _pairwise_examples(grouped)
    if not examples:
        raise ValueError("no non-tied candidate pairs are available")
    for _ in range(LINEAR_EPOCHS):
        for features, label in examples:
            margin = label * _dot(weights, features)
            if margin < 1.0:
                for index, value in enumerate(features):
                    weights[index] = (
                        (1.0 - LINEAR_LEARNING_RATE * LINEAR_L2)
                        * weights[index]
                        + LINEAR_LEARNING_RATE * label * value
                    )
            else:
                for index in range(len(weights)):
                    weights[index] = (
                        1.0 - LINEAR_LEARNING_RATE * LINEAR_L2
                    ) * weights[index]
    return {
        "model_type": "pairwise_linear",
        "weights": weights,
        "bias": 0.0,
        "epochs": LINEAR_EPOCHS,
        "learning_rate": LINEAR_LEARNING_RATE,
        "l2": LINEAR_L2,
        "pair_count": len(examples),
    }


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
    except Exception:
        return False
    return True


def _train_sklearn_pairwise(
    model_name: str,
    grouped: dict[str, list[dict[str, Any]]],
    model_path: Path,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    examples = []
    labels = []
    for state_cases in grouped.values():
        for left, right in itertools.combinations(state_cases, 2):
            delta = float(left["utility"]) - float(right["utility"])
            if abs(delta) <= 1e-12:
                continue
            examples.append(_subtract(left["vector"], right["vector"]))
            labels.append(1 if delta > 0 else 0)
            examples.append(_subtract(right["vector"], left["vector"]))
            labels.append(0 if delta > 0 else 1)
    if not examples:
        raise ValueError("no non-tied candidate pairs are available")
    if model_name == "sklearn_logistic":
        estimator = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=20260708,
            solver="liblinear",
        )
    elif model_name == "sklearn_forest":
        estimator = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=3,
            random_state=20260708,
        )
    elif model_name == "sklearn_gbdt":
        estimator = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=2,
            learning_rate=0.05,
            random_state=20260708,
        )
    else:
        raise ValueError(f"unknown sklearn ranker model: {model_name}")
    estimator.fit(np.asarray(examples, dtype=float), np.asarray(labels, dtype=int))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump(estimator, stream)
    return {
        "model_type": model_name,
        "path": str(model_path.name),
        "pair_count": len(examples),
    }


def train_candidate_ranker(
    memory: str | Path,
    output: str | Path,
    feature_profile: str = "dedup20",
    models: list[str] | None = None,
) -> dict[str, Any]:
    if feature_profile not in FEATURE_PROFILES:
        raise ValueError(f"unknown candidate feature profile: {feature_profile}")
    requested = models or ["pairwise_linear"]
    allowed = {
        "pairwise_linear",
        "sklearn_logistic",
        "sklearn_forest",
        "sklearn_gbdt",
    }
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"unknown ranker model(s): {', '.join(unknown)}")
    memory_root = Path(memory).resolve()
    output_root = Path(output).resolve()
    summary = _read_json(memory_root / "candidate_summary.json")
    cases = _read_jsonl(memory_root / "candidate_cases.jsonl")
    if (
        summary.get("split") != "train"
        or summary.get("usage") != "memory"
        or any(case.get("split") != "train" for case in cases)
    ):
        raise ValueError("candidate ranker may only train on Train memory")
    normalizer = fit_normalizer(cases, feature_profile)
    prepared = _prepare_cases(cases, normalizer)
    grouped = _group_by_state(prepared)
    _write_json(output_root / "normalizer.json", normalizer)

    trained = []
    skipped = []
    for model_name in requested:
        if model_name == "pairwise_linear":
            model = _train_pairwise_linear(grouped, normalizer["feature_count"])
            _write_json(output_root / "pairwise_linear.json", model)
            trained.append(
                {
                    "model_type": "pairwise_linear",
                    "path": "pairwise_linear.json",
                    "pair_count": model["pair_count"],
                }
            )
            continue
        if not _sklearn_available():
            skipped.append(
                {
                    "model_type": model_name,
                    "reason": "sklearn_unavailable",
                }
            )
            continue
        trained.append(
            _train_sklearn_pairwise(
                model_name,
                grouped,
                output_root / f"{model_name}.pkl",
            )
        )
    if not trained:
        raise ValueError("no ranker models were trained")

    result = {
        "schema_version": 1,
        "fit_split": "train",
        "test_data_read": False,
        "feature_profile": feature_profile,
        "feature_count": normalizer["feature_count"],
        "feature_names": [entry["name"] for entry in normalizer["features"]],
        "state_count": len(grouped),
        "case_count": len(prepared),
        "model_count": len(trained),
        "models": trained,
        "skipped_models": skipped,
        "excluded_feature_classes": [
            "agent_id",
            "absolute_coordinates",
            "generator_name",
            "post_repair_paths",
            "outcome_labels",
        ],
    }
    _write_json(output_root / "ranker_summary.json", result)
    return result


class CandidateRanker:
    def __init__(
        self,
        ranker_root: str | Path,
        model_type: str,
    ) -> None:
        self.root = Path(ranker_root).resolve()
        self.normalizer = _read_json(self.root / "normalizer.json")
        if self.normalizer.get("fit_split") != "train":
            raise ValueError("candidate ranker normalizer is not Train-only")
        self.model_type = model_type
        if model_type == "pairwise_linear":
            model = _read_json(self.root / "pairwise_linear.json")
            self.weights = [float(value) for value in model["weights"]]
            self.bias = float(model.get("bias", 0.0))
            self.estimator = None
        elif model_type.startswith("sklearn_"):
            path = self.root / f"{model_type}.pkl"
            with path.open("rb") as stream:
                self.estimator = pickle.load(stream)
            self.weights = []
            self.bias = 0.0
        else:
            raise ValueError(f"unknown ranker model: {model_type}")

    def vectorize(self, features: dict[str, float]) -> list[float]:
        return vectorize(features, self.normalizer)

    def _linear_scores(self, vectors: list[list[float]]) -> list[float]:
        return [self.bias + _dot(self.weights, vector) for vector in vectors]

    def _sklearn_scores(self, vectors: list[list[float]]) -> list[float]:
        import numpy as np

        assert self.estimator is not None
        count = len(vectors)
        scores = [0.0] * count
        for left, right in itertools.permutations(range(count), 2):
            diff = np.asarray([_subtract(vectors[left], vectors[right])])
            if hasattr(self.estimator, "predict_proba"):
                probability = float(self.estimator.predict_proba(diff)[0][1])
            elif hasattr(self.estimator, "decision_function"):
                probability = _sigmoid(
                    float(self.estimator.decision_function(diff)[0])
                )
            else:
                probability = float(self.estimator.predict(diff)[0])
            scores[left] += probability
        return [score / max(1, count - 1) for score in scores]

    def score_vectors(self, vectors: list[list[float]]) -> list[float]:
        if self.model_type == "pairwise_linear":
            return self._linear_scores(vectors)
        return self._sklearn_scores(vectors)


def _state_metrics(
    state_cases: list[dict[str, Any]],
    scores: dict[int, float],
    margin: float,
) -> dict[str, Any]:
    actual_values = {
        int(case["candidate_index"]): actual_utility(case["outcome"])
        for case in state_cases
    }
    baseline_score = scores[0]
    alternatives = [index for index in scores if index != 0]
    best_index = max(alternatives, key=lambda index: (scores[index], -index))
    reason = ""
    if scores[best_index] - baseline_score < margin:
        reason = "insufficient_margin"
    selected_index = 0 if reason else best_index
    oracle_index = max(
        actual_values,
        key=lambda index: (actual_values[index], -index),
    )
    predicted_index = max(scores, key=lambda index: (scores[index], -index))
    pair_correct = 0
    pair_total = 0
    for left, right in itertools.combinations(sorted(actual_values), 2):
        actual_delta = actual_values[left] - actual_values[right]
        predicted_delta = scores[left] - scores[right]
        if abs(actual_delta) <= 1e-12:
            continue
        pair_total += 1
        pair_correct += (actual_delta > 0) == (predicted_delta > 0)
    return {
        "state_id": state_cases[0]["state_id"],
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
                "score": scores[int(case["candidate_index"])],
                "actual_utility": actual_values[int(case["candidate_index"])],
            }
            for case in state_cases
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
        "top1_accuracy": sum(state["top1_exact"] for state in states) / count,
        "ranking_accuracy": sum(state["pair_correct"] for state in states)
        / max(1, sum(state["pair_total"] for state in states)),
        "guidance_use_rate": sum(state["used_guidance"] for state in states)
        / count,
    }


def evaluate_candidate_ranker(
    ranker: str | Path,
    queries: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    ranker_root = Path(ranker).resolve()
    query_root = Path(queries).resolve()
    output_root = Path(output).resolve()
    ranker_summary = _read_json(ranker_root / "ranker_summary.json")
    query_summary = _read_json(query_root / "candidate_summary.json")
    cases = _read_jsonl(query_root / "candidate_cases.jsonl")
    if (
        query_summary.get("split") != "validation"
        or query_summary.get("usage") != "evaluation"
        or any(case.get("split") != "validation" for case in cases)
    ):
        raise ValueError("candidate ranker tuning requires Validation queries")
    grouped = _group_by_state(cases)

    evaluations = []
    selected: dict[str, Any] | None = None
    selected_states: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    for model in ranker_summary["models"]:
        model_type = str(model["model_type"])
        candidate_ranker = CandidateRanker(ranker_root, model_type)
        for margin in MARGIN_OPTIONS:
            states = []
            for state_id, state_cases in grouped.items():
                vectors = [
                    candidate_ranker.vectorize(case["features"])
                    for case in state_cases
                ]
                raw_scores = candidate_ranker.score_vectors(vectors)
                scores = {
                    int(case["candidate_index"]): float(score)
                    for case, score in zip(state_cases, raw_scores)
                }
                states.append(_state_metrics(state_cases, scores, margin))
            metrics = _aggregate_metrics(states)
            row = {
                "model_type": model_type,
                "minimum_margin": margin,
                "metrics": metrics,
            }
            evaluations.append(row)
            key = (
                metrics["top1_gain"],
                -metrics["oracle_regret"],
                metrics["ranking_accuracy"],
                metrics["top1_accuracy"],
                -margin,
                model_type,
            )
            if best_key is None or key > best_key:
                best_key = key
                selected = row
                selected_states = states
    assert selected is not None
    selected_config = {
        "schema_version": 1,
        "selected_on_split": "validation",
        "test_data_read": False,
        "guide_type": "ranker",
        "feature_profile": ranker_summary["feature_profile"],
        "model_type": selected["model_type"],
        "minimum_margin": selected["minimum_margin"],
    }
    _write_jsonl(output_root / "ranker_guidance.jsonl", selected_states)
    _write_json(output_root / "selected_config.json", selected_config)
    summary = {
        "schema_version": 1,
        "ranker_split": "train",
        "query_split": "validation",
        "test_data_read": False,
        "feature_profile": ranker_summary["feature_profile"],
        "feature_count": ranker_summary["feature_count"],
        "configuration_count": len(evaluations),
        "selected_parameters": selected_config,
        "selected_metrics": selected["metrics"],
        "all_configurations": evaluations,
    }
    _write_json(output_root / "evaluation_summary.json", summary)
    return summary


def predict_ranker_state(
    ranker: CandidateRanker,
    candidate_features: list[dict[str, float]],
    candidate_indices: list[int],
    minimum_margin: float,
) -> dict[str, Any]:
    vectors = [ranker.vectorize(features) for features in candidate_features]
    scores = ranker.score_vectors(vectors)
    indexed_scores = {
        int(index): float(score)
        for index, score in zip(candidate_indices, scores)
    }
    baseline = indexed_scores[0]
    alternatives = [
        index for index in sorted(indexed_scores) if int(index) != 0
    ]
    best = max(alternatives, key=lambda index: (indexed_scores[index], -index))
    margin = indexed_scores[best] - baseline
    reason = ""
    if margin < minimum_margin:
        reason = "insufficient_margin"
    return {
        "use_guidance": not reason,
        "candidate_index": int(best if not reason else 0),
        "fallback_reason": reason,
        "predicted_margin": float(margin),
        "predicted_score": float(indexed_scores[best]),
        "baseline_score": float(baseline),
        "predictions": [
            {
                "candidate_index": int(index),
                "score": float(indexed_scores[int(index)]),
            }
            for index in sorted(indexed_scores)
        ],
    }
