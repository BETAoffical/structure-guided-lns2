from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from experiments.feature_schema_v2 import PROPOSAL_FAMILIES, unsupported_actual_size


PRUNER_SCHEMA = "lns2.proposal_pruner.v2"
DEFAULT_OOD_FRACTION = 0.10


def _outside_fraction(
    row: dict[str, Any], ranges: dict[str, tuple[float, float]]
) -> float:
    features = dict(row["features"]["proposal_dynamic"])
    if not ranges:
        return 0.0
    outside = sum(
        float(features.get(name, 0.0)) < float(bounds[0])
        or float(features.get(name, 0.0)) > float(bounds[1])
        for name, bounds in ranges.items()
    )
    return outside / len(ranges)


def _pair_probability(
    left: dict[str, Any], right: dict[str, Any], model: Any
) -> float:
    pair_vector = getattr(model, "pair_vector", None)
    if callable(pair_vector):
        forward_vector = pair_vector(left, right)
        reverse_vector = pair_vector(right, left)
    else:
        from research.studies.context.context_audit import _pair_vector

        forward_vector = _pair_vector(
            left, right, model.profile, model.feature_names
        )
        reverse_vector = _pair_vector(
            right, left, model.profile, model.feature_names
        )
    predict = getattr(model, "predict_positive", None)
    if callable(predict):
        forward = float(predict([forward_vector])[0])
        reverse = float(predict([reverse_vector])[0])
    else:
        import numpy as np

        forward = float(
            model.estimator.predict_proba(
                np.asarray([forward_vector], dtype=float)
            )[0, 1]
        )
        reverse = float(
            model.estimator.predict_proba(
                np.asarray([reverse_vector], dtype=float)
            )[0, 1]
        )
    return (forward + (1.0 - reverse)) / 2.0


@dataclass
class CandidatePruner:
    model: Any
    threshold: float
    ranges: dict[str, tuple[float, float]]
    expected_families: tuple[str, ...] = PROPOSAL_FAMILIES
    maximum_outside_fraction: float = DEFAULT_OOD_FRACTION
    pruner_id: str = "proposal_pruner_v2"

    def __post_init__(self) -> None:
        if not 0.5 <= float(self.threshold) <= 1.0:
            raise ValueError("pruner threshold must be in [0.5, 1.0]")
        if not 0.0 <= float(self.maximum_outside_fraction) <= 1.0:
            raise ValueError("invalid pruner OOD fraction")

    def prune(
        self,
        candidates: list[dict[str, Any]],
        proposal_rows: list[dict[str, Any]],
    ) -> tuple[list[int], dict[str, Any]]:
        started = time.perf_counter()

        def fallback(reason: str) -> tuple[list[int], dict[str, Any]]:
            return list(range(len(candidates))), {
                "pruner_id": self.pruner_id,
                "enabled": True,
                "fallback": True,
                "fallback_reason": reason,
                "candidate_count_before": len(candidates),
                "candidate_count_after": len(candidates),
                "reduction_fraction": 0.0,
                "pruner_seconds": time.perf_counter() - started,
                "family_decisions": [],
            }

        if len(candidates) != len(proposal_rows) or not candidates:
            return fallback("candidate_row_mismatch")

        for row in proposal_rows:
            values = dict(row["features"]["proposal_dynamic"])
            if any(not math.isfinite(float(value)) for value in values.values()):
                return fallback("non_finite_feature")
            if unsupported_actual_size(values):
                return fallback("unsupported_actual_size")
            if _outside_fraction(row, self.ranges) > self.maximum_outside_fraction:
                return fallback("feature_out_of_range")

        by_family: dict[str, list[int]] = {family: [] for family in self.expected_families}
        for index, candidate in enumerate(candidates):
            for family in map(str, candidate.get("selection_families", [])):
                if family not in by_family:
                    return fallback("unexpected_family")
                by_family[family].append(index)
        if any(len(set(indices)) not in {1, 2} for indices in by_family.values()):
            return fallback("invalid_family_cardinality")

        retained: set[int] = set()
        decisions = []
        for family in self.expected_families:
            indices = sorted(set(by_family[family]))
            if len(indices) == 1:
                retained.add(indices[0])
                decisions.append(
                    {
                        "family": family,
                        "candidate_indices": indices,
                        "retained_indices": indices,
                        "confidence": 1.0,
                        "pruned": False,
                    }
                )
                continue
            left, right = indices
            probability = _pair_probability(
                proposal_rows[left], proposal_rows[right], self.model
            )
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                return fallback("non_finite_prediction")
            confidence = max(probability, 1.0 - probability)
            if confidence >= self.threshold:
                winner = left if probability >= 0.5 else right
                retained.add(winner)
                retained_indices = [winner]
                pruned = True
            else:
                retained.update(indices)
                retained_indices = indices
                pruned = False
            decisions.append(
                {
                    "family": family,
                    "candidate_indices": indices,
                    "retained_indices": retained_indices,
                    "probability_left": probability,
                    "confidence": confidence,
                    "pruned": pruned,
                }
            )
        if not retained:
            return fallback("empty_retained_pool")
        retained_indices = sorted(retained)
        return retained_indices, {
            "pruner_id": self.pruner_id,
            "enabled": True,
            "fallback": False,
            "fallback_reason": None,
            "candidate_count_before": len(candidates),
            "candidate_count_after": len(retained_indices),
            "reduction_fraction": 1.0 - len(retained_indices) / len(candidates),
            "pruner_seconds": time.perf_counter() - started,
            "family_decisions": decisions,
        }


def no_pruning_metrics(candidate_count: int, reason: str = "disabled") -> dict[str, Any]:
    return {
        "pruner_id": None,
        "enabled": False,
        "fallback": False,
        "fallback_reason": reason,
        "candidate_count_before": int(candidate_count),
        "candidate_count_after": int(candidate_count),
        "reduction_fraction": 0.0,
        "pruner_seconds": 0.0,
        "family_decisions": [],
    }


def expected_families_from_proposal_config(config: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{heuristic}:{size}"
            for heuristic in map(str, config["heuristics"])
            for size in map(int, config["neighborhood_sizes"])
        )
    )


__all__ = [
    "CandidatePruner",
    "DEFAULT_OOD_FRACTION",
    "PRUNER_SCHEMA",
    "expected_families_from_proposal_config",
    "no_pruning_metrics",
]
