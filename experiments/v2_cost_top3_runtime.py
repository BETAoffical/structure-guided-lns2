from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


V2_COST_TOP3_CONFIG_SCHEMA = "lns2.v2_cost_top3_config.v1"
PREDICTION_NAMES = {
    "effective_progress_probability",
    "no_progress_probability",
    "conflict_reduction",
    "repair_seconds",
}
THRESHOLD_NAMES = {
    "effective_probability_tolerance",
    "no_progress_probability_tolerance",
    "conflict_reduction_retention",
    "minimum_time_improvement",
    "minimum_utility_improvement",
}


@dataclass(frozen=True)
class FrozenV2CostTop3Config:
    top_k: int
    thresholds: dict[str, float]
    uncertainty_calibration: dict[str, Any]
    source: dict[str, Any]
    raw: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.raw, sort_keys=True))


def _finite_nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return number


def _validated_margins(raw: dict[str, Any], name: str) -> dict[str, float]:
    return {
        "reduction_overprediction": _finite_nonnegative(
            raw.get("reduction_overprediction"),
            f"{name}.reduction_overprediction",
        ),
        "seconds_underprediction": _finite_nonnegative(
            raw.get("seconds_underprediction"),
            f"{name}.seconds_underprediction",
        ),
    }


def load_v2_cost_top3_config(
    source: str | Path | dict[str, Any],
) -> FrozenV2CostTop3Config:
    if isinstance(source, dict):
        raw = json.loads(json.dumps(source))
    else:
        raw = json.loads(Path(source).read_text(encoding="utf-8"))
    if str(raw.get("schema")) != V2_COST_TOP3_CONFIG_SCHEMA:
        raise ValueError("unsupported v2 cost Top-3 config schema")
    top_k = int(raw.get("top_k", 0))
    if top_k != 3:
        raise ValueError("the frozen cost-aware controller requires top_k=3")
    thresholds_raw = dict(raw.get("thresholds") or {})
    if set(thresholds_raw) != THRESHOLD_NAMES:
        raise ValueError("v2 cost Top-3 thresholds are incomplete")
    thresholds = {
        name: _finite_nonnegative(thresholds_raw[name], name)
        for name in sorted(THRESHOLD_NAMES)
    }
    for name in (
        "effective_probability_tolerance",
        "no_progress_probability_tolerance",
        "conflict_reduction_retention",
        "minimum_time_improvement",
    ):
        if thresholds[name] > 1.0:
            raise ValueError(f"{name} must not exceed one")
    calibration_raw = dict(raw.get("uncertainty_calibration") or {})
    global_margins = _validated_margins(
        dict(calibration_raw.get("global") or {}), "calibration.global"
    )
    by_agent_count = {
        str(int(agent_count)): _validated_margins(
            dict(values), f"calibration.by_agent_count.{agent_count}"
        )
        for agent_count, values in dict(
            calibration_raw.get("by_agent_count") or {}
        ).items()
    }
    calibration = {
        "quantile": float(calibration_raw.get("quantile", 0.0)),
        "global": global_margins,
        "by_agent_count": by_agent_count,
    }
    if not 0.0 <= calibration["quantile"] <= 1.0:
        raise ValueError("uncertainty quantile must be between zero and one")
    return FrozenV2CostTop3Config(
        top_k=top_k,
        thresholds=thresholds,
        uncertainty_calibration=calibration,
        source=dict(raw.get("source") or {}),
        raw=raw,
    )


def _candidate_key(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_key", candidate["candidate_id"]))


def _margins(
    config: FrozenV2CostTop3Config, agent_count: int
) -> dict[str, float]:
    calibration = config.uncertainty_calibration
    return dict(
        calibration["by_agent_count"].get(
            str(int(agent_count)), calibration["global"]
        )
    )


def _conservative_prediction(
    predictions: dict[str, list[float]],
    index: int,
    margins: dict[str, float],
) -> dict[str, float]:
    reduction = max(0.0, float(predictions["conflict_reduction"][index]))
    seconds = max(1e-9, float(predictions["repair_seconds"][index]))
    lower_reduction = max(
        0.0, reduction - float(margins["reduction_overprediction"])
    )
    upper_seconds = seconds + float(margins["seconds_underprediction"])
    return {
        "effective_progress_probability": float(
            predictions["effective_progress_probability"][index]
        ),
        "no_progress_probability": float(
            predictions["no_progress_probability"][index]
        ),
        "conflict_reduction": reduction,
        "repair_seconds": seconds,
        "lower_conflict_reduction": lower_reduction,
        "upper_repair_seconds": upper_seconds,
        "lower_utility": lower_reduction / max(1e-9, upper_seconds),
    }


def select_v2_cost_top3(
    candidates: list[dict[str, Any]],
    v2_scores: list[float],
    predictions: dict[str, list[float]],
    config: FrozenV2CostTop3Config,
    *,
    agent_count: int,
) -> tuple[int, dict[str, Any]]:
    if len(candidates) < config.top_k or len(v2_scores) != len(candidates):
        raise ValueError("v2 cost Top-3 inputs do not match the candidate pool")
    if not PREDICTION_NAMES <= set(predictions) or any(
        len(predictions[name]) != len(candidates) for name in PREDICTION_NAMES
    ):
        raise ValueError("v2 cost Top-3 predictions are incomplete")
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -round(float(v2_scores[index]), 12),
            _candidate_key(candidates[index]),
        ),
    )
    base_index = order[0]
    top_indices = order[: config.top_k]
    margins = _margins(config, agent_count)
    conservative = {
        index: _conservative_prediction(predictions, index, margins)
        for index in top_indices
    }
    base = conservative[base_index]
    thresholds = config.thresholds
    eligible: list[int] = []
    for index in top_indices:
        prediction = conservative[index]
        if prediction["effective_progress_probability"] < (
            base["effective_progress_probability"]
            - thresholds["effective_probability_tolerance"]
        ):
            continue
        if prediction["no_progress_probability"] > (
            base["no_progress_probability"]
            + thresholds["no_progress_probability_tolerance"]
        ):
            continue
        retention = thresholds["conflict_reduction_retention"]
        if prediction["conflict_reduction"] + 1e-12 < (
            retention * base["conflict_reduction"]
        ):
            continue
        if base["lower_conflict_reduction"] > 0.0 and (
            prediction["lower_conflict_reduction"] + 1e-12
            < retention * base["lower_conflict_reduction"]
        ):
            continue
        if prediction["upper_repair_seconds"] > (
            (1.0 - thresholds["minimum_time_improvement"])
            * base["upper_repair_seconds"]
            + 1e-12
        ):
            continue
        if prediction["lower_utility"] + 1e-12 < (
            (1.0 + thresholds["minimum_utility_improvement"])
            * base["lower_utility"]
        ):
            continue
        eligible.append(index)
    # Preserve the frozen offline audit's exact tie semantics.
    selected = max(
        eligible or [base_index],
        key=lambda index: (
            conservative[index]["lower_utility"],
            conservative[index]["lower_conflict_reduction"],
            -conservative[index]["upper_repair_seconds"],
            float(v2_scores[index]),
            str(candidates[index]["candidate_id"]),
        ),
    )
    return selected, {
        "schema": V2_COST_TOP3_CONFIG_SCHEMA,
        "base_candidate_id": str(candidates[base_index]["candidate_id"]),
        "selected_candidate_id": str(candidates[selected]["candidate_id"]),
        "override": selected != base_index,
        "selected_v2_rank": order.index(selected) + 1,
        "eligible_candidate_count": len(eligible),
        "top3_candidate_ids": [
            str(candidates[index]["candidate_id"]) for index in top_indices
        ],
        "base_prediction": base,
        "selected_prediction": conservative[selected],
        "agent_count_calibration": (
            str(int(agent_count))
            if str(int(agent_count))
            in config.uncertainty_calibration["by_agent_count"]
            else "global"
        ),
        "route": "model",
    }


__all__ = [
    "FrozenV2CostTop3Config",
    "V2_COST_TOP3_CONFIG_SCHEMA",
    "load_v2_cost_top3_config",
    "select_v2_cost_top3",
]
