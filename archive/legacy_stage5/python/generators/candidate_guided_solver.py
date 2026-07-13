from __future__ import annotations

import json
import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from .candidate_experience import candidate_raw_features
from .candidate_ranker import (
    CandidateRanker,
    predict_ranker_state,
)
from .candidate_retrieval import (
    predict_candidate,
    vectorize,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


class CandidateGuide:
    def __init__(
        self,
        dataset: str | Path,
        split: str,
        task_id: str,
        index: str | Path,
        config: str | Path,
    ) -> None:
        if split not in {"validation", "test"}:
            raise ValueError("candidate guidance cannot read Train")
        self.dataset_root = Path(dataset).resolve()
        self.split = split
        self.task_id = task_id
        self.index_root = Path(index).resolve()
        rows = _read_jsonl(
            self.dataset_root / split / "manifest.jsonl"
        )
        try:
            self.manifest = next(
                row for row in rows if row["task_id"] == task_id
            )
        except StopIteration as error:
            raise ValueError(f"unknown task: {task_id}") from error
        self.map_document = _read_json(
            self.dataset_root / split / self.manifest["map_file"]
        )
        self.task_document = _read_json(
            self.dataset_root / split / self.manifest["task_file"]
        )
        self.normalizer = _read_json(
            self.index_root / "normalizer.json"
        )
        if self.normalizer.get("fit_split") != "train":
            raise ValueError("candidate normalizer is not Train-only")
        self.entries = _read_jsonl(
            self.index_root / "candidate_index.jsonl"
        )
        if any(
            not str(entry["task_id"]).startswith("train_")
            for entry in self.entries
        ):
            raise ValueError("candidate index contains non-Train data")
        parameters = _read_json(Path(config).resolve())
        if (
            parameters.get("selected_on_split") != "validation"
            or parameters.get("test_data_read", True)
        ):
            raise ValueError(
                "candidate guidance requires frozen Validation parameters"
            )
        index_profile = str(
            self.normalizer.get("feature_profile", "full")
        )
        config_profile = str(
            parameters.get("feature_profile", "full")
        )
        if index_profile != config_profile:
            raise ValueError(
                "candidate index/config feature profiles do not match"
            )
        self.k = int(parameters["k"])
        self.group_weights = {
            key: float(value)
            for key, value in parameters["group_weights"].items()
        }
        self.minimum_margin = float(parameters["minimum_margin"])
        self.ood_threshold = float(
            parameters["ood_distance_threshold"]
        )
        self.minimum_valid_probability = float(
            parameters["minimum_valid_probability"]
        )

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        predictions = []
        for candidate in request["candidates"]:
            features = candidate_raw_features(
                self.map_document,
                self.task_document,
                self.manifest,
                request["conflict_events"],
                request["paths"],
                request["seed_conflict"],
                candidate,
            )
            prediction = predict_candidate(
                vectorize(features, self.normalizer),
                self.entries,
                self.normalizer,
                self.group_weights,
                self.k,
            )
            predictions.append(
                {
                    "candidate_index": int(
                        candidate["candidate_index"]
                    ),
                    **prediction,
                }
            )
        baseline = next(
            prediction
            for prediction in predictions
            if prediction["candidate_index"] == 0
        )
        best = max(
            (
                prediction
                for prediction in predictions
                if prediction["candidate_index"] != 0
            ),
            key=lambda prediction: (
                prediction["utility"],
                -prediction["candidate_index"],
            ),
        )
        reason = ""
        if not math.isfinite(best["nearest_distance"]):
            reason = "no_neighbors"
        elif best["nearest_distance"] > self.ood_threshold:
            reason = "out_of_distribution"
        elif (
            best["valid_probability"]
            < self.minimum_valid_probability
        ):
            reason = "low_valid_probability"
        elif (
            best["utility"] - baseline["utility"]
            < self.minimum_margin
        ):
            reason = "insufficient_margin"
        return {
            "use_guidance": not reason,
            "candidate_index": (
                int(best["candidate_index"]) if not reason else 0
            ),
            "out_of_distribution": reason == "out_of_distribution",
            "fallback_reason": reason,
            "predicted_valid_probability": float(
                best["valid_probability"]
            ),
            "predicted_conflict_reduction": float(
                best["conflict_reduction"]
            ),
            "predicted_cost_improvement": float(
                best["cost_improvement"]
            ),
            "predicted_runtime_ms": float(best["runtime_ms"]),
            "nearest_distance": float(best["nearest_distance"]),
            "predicted_margin": float(
                best["utility"] - baseline["utility"]
            ),
            "predictions": predictions,
            "python_runtime_ms": (
                time.perf_counter() - started
            )
            * 1000.0,
        }


class RankerCandidateGuide:
    def __init__(
        self,
        dataset: str | Path,
        split: str,
        task_id: str,
        ranker: str | Path,
        config: str | Path,
    ) -> None:
        if split not in {"validation", "test"}:
            raise ValueError("candidate guidance cannot read Train")
        self.dataset_root = Path(dataset).resolve()
        self.split = split
        self.task_id = task_id
        self.ranker_root = Path(ranker).resolve()
        rows = _read_jsonl(
            self.dataset_root / split / "manifest.jsonl"
        )
        try:
            self.manifest = next(
                row for row in rows if row["task_id"] == task_id
            )
        except StopIteration as error:
            raise ValueError(f"unknown task: {task_id}") from error
        self.map_document = _read_json(
            self.dataset_root / split / self.manifest["map_file"]
        )
        self.task_document = _read_json(
            self.dataset_root / split / self.manifest["task_file"]
        )
        self.ranker_summary = _read_json(
            self.ranker_root / "ranker_summary.json"
        )
        if self.ranker_summary.get("fit_split") != "train":
            raise ValueError("candidate ranker is not Train-only")
        parameters = _read_json(Path(config).resolve())
        if (
            parameters.get("selected_on_split") != "validation"
            or parameters.get("test_data_read", True)
        ):
            raise ValueError(
                "candidate guidance requires frozen Validation parameters"
            )
        if parameters.get("guide_type") != "ranker":
            raise ValueError("ranker guidance requires a ranker config")
        ranker_profile = str(
            self.ranker_summary.get("feature_profile", "full")
        )
        config_profile = str(
            parameters.get("feature_profile", "full")
        )
        if ranker_profile != config_profile:
            raise ValueError(
                "candidate ranker/config feature profiles do not match"
            )
        self.minimum_margin = float(parameters["minimum_margin"])
        self.regular_beltway_margin_bonus = float(
            parameters.get("regular_beltway_margin_bonus", 0.0)
        )
        self.low_conflict_margin_bonus = float(
            parameters.get("low_conflict_margin_bonus", 0.0)
        )
        self.low_conflict_threshold = int(
            parameters.get("low_conflict_threshold", 0)
        )
        self.clustered_margin_bonus = float(
            parameters.get("clustered_margin_bonus", 0.0)
        )
        self.model_type = str(parameters["model_type"])
        trained_models = {
            str(model["model_type"])
            for model in self.ranker_summary.get("models", [])
        }
        if self.model_type not in trained_models:
            raise ValueError("selected ranker model was not trained")
        self.ranker = CandidateRanker(self.ranker_root, self.model_type)

    def decide(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        candidate_features = []
        candidate_indices = []
        for candidate in request["candidates"]:
            candidate_indices.append(int(candidate["candidate_index"]))
            candidate_features.append(
                candidate_raw_features(
                    self.map_document,
                    self.task_document,
                    self.manifest,
                    request["conflict_events"],
                    request["paths"],
                    request["seed_conflict"],
                    candidate,
                )
            )
        decision = predict_ranker_state(
            self.ranker,
            candidate_features,
            candidate_indices,
            self._effective_margin(request),
        )
        best_score = float(decision["predicted_score"])
        return {
            "use_guidance": bool(decision["use_guidance"]),
            "candidate_index": int(decision["candidate_index"]),
            "out_of_distribution": False,
            "fallback_reason": decision["fallback_reason"],
            "predicted_valid_probability": 1.0 / (
                1.0 + math.exp(-max(-40.0, min(40.0, best_score)))
            ),
            "predicted_conflict_reduction": best_score,
            "predicted_cost_improvement": 0.0,
            "predicted_runtime_ms": 0.0,
            "nearest_distance": 0.0,
            "predicted_margin": float(decision["predicted_margin"]),
            "predictions": decision["predictions"],
            "python_runtime_ms": (
                time.perf_counter() - started
            )
            * 1000.0,
        }

    def _effective_margin(self, request: dict[str, Any]) -> float:
        margin = self.minimum_margin
        if self.manifest.get("layout_mode") == "regular_beltway":
            margin += self.regular_beltway_margin_bonus
        if self.manifest.get("task_variant") == "balanced_clustered":
            margin += self.clustered_margin_bonus
        if (
            self.low_conflict_threshold > 0
            and len(request.get("conflict_events", []))
            <= self.low_conflict_threshold
        ):
            margin += self.low_conflict_margin_bonus
        return margin


def run_candidate_guided_instance(
    solver: str | Path,
    instance: str | Path,
    trace: str | Path,
    guide: CandidateGuide,
    seed: int,
    neighborhood: int = 6,
    iterations: int = 500,
    time_limit_ms: int = 5000,
    candidate_generator_profile: str = "full8",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_count = 5 if candidate_generator_profile == "core5" else 8
    command = [
        str(Path(solver).resolve()),
        "--instance",
        str(Path(instance).resolve()),
        "--seed",
        str(seed),
        "--neighborhood",
        str(neighborhood),
        "--iterations",
        str(iterations),
        "--time-limit-ms",
        str(time_limit_ms),
        "--trace",
        str(Path(trace).resolve()),
        "--candidate-count",
        str(candidate_count),
        "--candidate-generator-profile",
        candidate_generator_profile,
        "--candidate-guidance-stdio",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    watchdog = threading.Timer(
        max(30.0, time_limit_ms / 1000.0 + 30.0),
        process.kill,
    )
    watchdog.daemon = True
    watchdog.start()
    if process.stdin is None or process.stdout is None:
        process.kill()
        raise RuntimeError("failed to open candidate guidance pipes")
    decisions = []
    result = None
    try:
        for line in process.stdout:
            line = line.strip()
            if line.startswith("CANDIDATE_REQUEST "):
                request = json.loads(
                    line[len("CANDIDATE_REQUEST ") :]
                )
                try:
                    decision = guide.decide(request)
                except Exception as error:
                    decision = {
                        "use_guidance": False,
                        "candidate_index": 0,
                        "out_of_distribution": False,
                        "fallback_reason": "python_error",
                        "predicted_valid_probability": -1.0,
                        "predicted_conflict_reduction": 0.0,
                        "predicted_cost_improvement": 0.0,
                        "predicted_runtime_ms": -1.0,
                        "nearest_distance": -1.0,
                        "predicted_margin": 0.0,
                        "predictions": [],
                        "python_runtime_ms": 0.0,
                        "error": str(error),
                    }
                decisions.append(
                    {"iteration": request["iteration"], **decision}
                )
                reason = decision["fallback_reason"] or "-"
                response = [
                    "CANDIDATE",
                    "1" if decision["use_guidance"] else "0",
                    str(decision["candidate_index"]),
                    f"{decision['predicted_valid_probability']:.12g}",
                    f"{decision['predicted_conflict_reduction']:.12g}",
                    f"{decision['predicted_cost_improvement']:.12g}",
                    f"{decision['predicted_runtime_ms']:.12g}",
                    f"{decision['nearest_distance']:.12g}",
                    (
                        "1"
                        if decision["out_of_distribution"]
                        else "0"
                    ),
                    reason,
                ]
                process.stdin.write(" ".join(response) + "\n")
                process.stdin.flush()
            elif line.startswith("RESULT "):
                result = json.loads(line[len("RESULT ") :])
        process.stdin.close()
        return_code = process.wait()
    finally:
        watchdog.cancel()
    stderr = process.stderr.read() if process.stderr is not None else ""
    if result is None or return_code not in {0, 1}:
        raise RuntimeError(
            f"candidate-guided solver failed ({return_code}): "
            f"{stderr.strip()}"
        )
    result["return_code"] = return_code
    return result, decisions
