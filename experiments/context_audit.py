"""Runtime compatibility for the frozen pairwise GBDT.

The canonical v1 sklearn pickle records
``experiments.context_audit.PairwiseModel``. Keep the class at this qualified
name after the historical context audit leaves the active tree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _vector(row: dict[str, Any], profile: str, names: list[str]) -> list[float]:
    values = row["features"][profile]
    return [float(values.get(name, 0.0)) for name in names]


def _pair_vector(
    left: dict[str, Any],
    right: dict[str, Any],
    profile: str,
    names: list[str],
) -> list[float]:
    left_vector = _vector(left, profile, names)
    right_vector = _vector(right, profile, names)
    shared = [
        (first + second) / 2.0
        for name, first, second in zip(names, left_vector, right_vector)
        if name.startswith(("state.", "context."))
    ]
    return [
        first - second for first, second in zip(left_vector, right_vector)
    ] + shared


@dataclass
class PairwiseModel:
    profile: str
    feature_names: list[str]
    estimator: Any

    def select(self, candidates: list[dict[str, Any]]) -> int:
        import numpy as np

        comparisons: list[list[float]] = []
        reverse_comparisons: list[list[float]] = []
        pairs: list[tuple[int, int]] = []
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                comparisons.append(
                    _pair_vector(
                        candidates[left],
                        candidates[right],
                        self.profile,
                        self.feature_names,
                    )
                )
                reverse_comparisons.append(
                    _pair_vector(
                        candidates[right],
                        candidates[left],
                        self.profile,
                        self.feature_names,
                    )
                )
                pairs.append((left, right))
        if not comparisons:
            return 0
        forward = self.estimator.predict_proba(
            np.asarray(comparisons, dtype=float)
        )[:, 1]
        reverse = self.estimator.predict_proba(
            np.asarray(reverse_comparisons, dtype=float)
        )[:, 1]
        probabilities = (forward + (1.0 - reverse)) / 2.0
        scores = [0.0] * len(candidates)
        for probability, (left, right) in zip(probabilities, pairs):
            scores[left] += float(probability)
            scores[right] += 1.0 - float(probability)
        return min(
            range(len(candidates)),
            key=lambda index: (
                -scores[index],
                str(candidates[index]["candidate_key"]),
            ),
        )


__all__ = ["PairwiseModel", "_pair_vector"]
