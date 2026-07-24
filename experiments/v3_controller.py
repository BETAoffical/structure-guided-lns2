from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.feature_schema_v3 import resolve_v3_feature_names
from experiments.repair_aware import (
    PortableScalarModel,
    classify_repair_outcome,
    load_portable_scalar_model,
)


V3_BUNDLE_SCHEMA = "lns2.v3_controller_bundle.v1"
V3_H3_BUNDLE_SCHEMA = "lns2.v3_horizon_controller_bundle.v1"
V3_MODEL_NAMES = {
    "effective_progress_probability",
    "no_progress_probability",
    "conflict_reduction",
    "log_repair_seconds",
}
V3_THRESHOLD_NAMES = {
    "effective_probability_tolerance",
    "no_progress_probability_tolerance",
    "conflict_reduction_retention",
}
V3_H3_MODEL_NAMES = {
    "h1_effective_progress_probability",
    "h1_no_progress_probability",
    "h1_conflict_reduction",
    "h1_log_pp_seconds",
    "h3_conflict_reduction",
    "h3_log_total_seconds",
    "h3_no_progress_probability",
}
V3_H3_THRESHOLD_NAMES = {
    "h1_effective_probability_tolerance",
    "h1_no_progress_probability_tolerance",
    "minimum_h3_utility_improvement",
}


@dataclass(frozen=True)
class V3ControllerBundle:
    models: dict[str, PortableScalarModel]
    thresholds: dict[str, float]
    selection_overhead_seconds: float
    manifest: dict[str, Any]
    report: dict[str, Any]

    @property
    def schema(self) -> str:
        return str(self.manifest.get("schema", V3_BUNDLE_SCHEMA))

    @property
    def is_horizon(self) -> bool:
        return self.schema == V3_H3_BUNDLE_SCHEMA

    @property
    def inference_backends(self) -> tuple[str, ...]:
        return tuple(sorted({model.inference_backend for model in self.models.values()}))

    @property
    def required_feature_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    feature
                    for model in self.models.values()
                    for feature in model.feature_names
                }
            )
        )

    @property
    def runtime_projection(self) -> dict[str, Any]:
        declared = len(tuple(self.manifest.get("feature_names", ())))
        return {
            "declared_feature_count": declared,
            "runtime_feature_count": len(self.required_feature_names),
            "removed_runtime_feature_count": declared
            - len(self.required_feature_names),
            "model_runtime_feature_counts": {
                name: len(model.feature_names)
                for name, model in sorted(self.models.items())
            },
        }

    def predict(self, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
        if self.is_horizon:
            h1_effective = self.models["h1_effective_progress_probability"].predict(rows)
            h1_no_progress = self.models["h1_no_progress_probability"].predict(rows)
            h1_reduction = self.models["h1_conflict_reduction"].predict(rows)
            h1_log_pp = self.models["h1_log_pp_seconds"].predict(rows)
            h3_reduction = self.models["h3_conflict_reduction"].predict(rows)
            h3_log_total = self.models["h3_log_total_seconds"].predict(rows)
            h3_no_progress = self.models["h3_no_progress_probability"].predict(rows)
            h1_pp_seconds = [
                max(1e-9, math.expm1(min(50.0, float(value))))
                for value in h1_log_pp
            ]
            h3_total_seconds = [
                max(1e-9, math.expm1(min(50.0, float(value))))
                for value in h3_log_total
            ]
            h3_reductions = [max(0.0, float(value)) for value in h3_reduction]
            return {
                "effective_progress_probability": [
                    min(1.0, max(0.0, float(value))) for value in h1_effective
                ],
                "no_progress_probability": [
                    min(1.0, max(0.0, float(value))) for value in h1_no_progress
                ],
                "h1_conflict_reduction": [
                    max(0.0, float(value)) for value in h1_reduction
                ],
                "h1_pp_seconds": h1_pp_seconds,
                "h3_conflict_reduction": h3_reductions,
                "h3_total_seconds": h3_total_seconds,
                "h3_no_progress_probability": [
                    min(1.0, max(0.0, float(value)))
                    for value in h3_no_progress
                ],
                "utility": [
                    reduction / max(1e-9, duration)
                    for reduction, duration in zip(
                        h3_reductions, h3_total_seconds
                    )
                ],
            }
        effective = self.models["effective_progress_probability"].predict(rows)
        no_progress = self.models["no_progress_probability"].predict(rows)
        reduction = self.models["conflict_reduction"].predict(rows)
        log_seconds = self.models["log_repair_seconds"].predict(rows)
        seconds = [
            max(1e-9, math.expm1(min(50.0, float(value))))
            for value in log_seconds
        ]
        reductions = [max(0.0, float(value)) for value in reduction]
        return {
            "effective_progress_probability": [
                min(1.0, max(0.0, float(value))) for value in effective
            ],
            "no_progress_probability": [
                min(1.0, max(0.0, float(value))) for value in no_progress
            ],
            "conflict_reduction": reductions,
            "repair_seconds": seconds,
            "utility": [
                delta / max(1e-9, duration + self.selection_overhead_seconds)
                for delta, duration in zip(reductions, seconds)
            ],
        }


def load_v3_controller_bundle(path: str | Path) -> V3ControllerBundle:
    root = Path(path).resolve()
    manifest = dict(read_json(root / "v3_manifest.json"))
    schema = str(manifest.get("schema"))
    if schema not in {V3_BUNDLE_SCHEMA, V3_H3_BUNDLE_SCHEMA}:
        raise ValueError("unexpected v3 controller bundle schema")
    expected_feature_names = resolve_v3_feature_names(
        str(manifest.get("feature_schema_id")),
        str(manifest.get("feature_schema_sha256")),
    )
    if tuple(map(str, manifest.get("feature_names", ()))) != expected_feature_names:
        raise ValueError("v3 controller feature declaration mismatch")
    models: dict[str, PortableScalarModel] = {}
    for name, raw in dict(manifest.get("models", {})).items():
        row = dict(raw)
        model_path = root / str(row["file"])
        if sha256_file(model_path) != str(row["sha256"]):
            raise ValueError(f"v3 model SHA256 mismatch: {name}")
        payload = dict(read_json(model_path))
        if tuple(map(str, payload.get("feature_names", ()))) != expected_feature_names:
            raise ValueError(f"v3 model feature declaration mismatch: {name}")
        model = load_portable_scalar_model(payload, compact_features=True)
        if model.name != str(name):
            raise ValueError(f"v3 model name mismatch: {name}")
        models[str(name)] = model
    expected_models = V3_H3_MODEL_NAMES if schema == V3_H3_BUNDLE_SCHEMA else V3_MODEL_NAMES
    if set(models) != expected_models:
        raise ValueError("v3 controller bundle has an incomplete model set")
    thresholds = {
        str(name): float(value)
        for name, value in dict(manifest.get("thresholds", {})).items()
    }
    expected_thresholds = (
        V3_H3_THRESHOLD_NAMES if schema == V3_H3_BUNDLE_SCHEMA else V3_THRESHOLD_NAMES
    )
    if set(thresholds) != expected_thresholds:
        raise ValueError("v3 controller bundle has incomplete thresholds")
    if schema == V3_H3_BUNDLE_SCHEMA:
        if not 0.0 <= thresholds["h1_effective_probability_tolerance"] <= 1.0:
            raise ValueError("v3-h3 effective probability tolerance is invalid")
        if not 0.0 <= thresholds["h1_no_progress_probability_tolerance"] <= 1.0:
            raise ValueError("v3-h3 no-progress probability tolerance is invalid")
        if not 0.0 <= thresholds["minimum_h3_utility_improvement"] <= 1.0:
            raise ValueError("v3-h3 utility improvement threshold is invalid")
    else:
        if not 0.0 <= thresholds["effective_probability_tolerance"] <= 1.0:
            raise ValueError("v3 effective probability tolerance is invalid")
        if not 0.0 <= thresholds["no_progress_probability_tolerance"] <= 1.0:
            raise ValueError("v3 no-progress probability tolerance is invalid")
        if not 0.0 < thresholds["conflict_reduction_retention"] <= 1.0:
            raise ValueError("v3 conflict-reduction retention is invalid")
    selection_overhead = float(manifest.get("selection_overhead_seconds", -1.0))
    if not math.isfinite(selection_overhead) or selection_overhead < 0.0:
        raise ValueError("v3 selection overhead is invalid")
    report_row = dict(manifest.get("training_report", {}))
    report_path = root / str(report_row["file"])
    if sha256_file(report_path) != str(report_row["sha256"]):
        raise ValueError("v3 training report SHA256 mismatch")
    return V3ControllerBundle(
        models=models,
        thresholds=thresholds,
        selection_overhead_seconds=selection_overhead,
        manifest=manifest,
        report=dict(read_json(report_path)),
    )


def v3_candidate_order(
    candidates: list[dict[str, Any]],
    predictions: dict[str, list[float]],
    v2_scores: list[float],
    thresholds: dict[str, float],
    *,
    eligible: Iterable[int] | None = None,
) -> list[int]:
    allowed = list(range(len(candidates))) if eligible is None else list(eligible)
    if not allowed:
        return []
    if len(v2_scores) != len(candidates) or any(
        len(values) != len(candidates) for values in predictions.values()
    ):
        raise ValueError("v3 candidate predictions do not match the candidate pool")
    effective = predictions["effective_progress_probability"]
    no_progress = predictions["no_progress_probability"]
    reduction = predictions["conflict_reduction"]
    utility = predictions["utility"]
    maximum_effective = max(effective[index] for index in allowed)
    effective_shortlist = [
        index
        for index in allowed
        if effective[index]
        >= maximum_effective - thresholds["effective_probability_tolerance"]
    ]
    minimum_no_progress = min(no_progress[index] for index in effective_shortlist)
    risk_shortlist = [
        index
        for index in effective_shortlist
        if no_progress[index]
        <= minimum_no_progress + thresholds["no_progress_probability_tolerance"]
    ]
    maximum_reduction = max(reduction[index] for index in risk_shortlist)
    quality_floor = thresholds["conflict_reduction_retention"] * maximum_reduction
    quality_shortlist = [
        index for index in risk_shortlist if reduction[index] + 1e-12 >= quality_floor
    ]
    return sorted(
        quality_shortlist,
        key=lambda index: (
            -round(float(utility[index]), 12),
            -round(float(effective[index]), 12),
            round(float(no_progress[index]), 12),
            -round(float(v2_scores[index]), 12),
            str(candidates[index]["candidate_id"]),
        ),
    )


def v3_h3_candidate_order(
    candidates: list[dict[str, Any]],
    predictions: dict[str, list[float]],
    v2_scores: list[float],
    thresholds: dict[str, float],
    *,
    eligible: Iterable[int] | None = None,
) -> list[int]:
    """Rank candidates by conservative three-repair utility.

    The frozen v2 winner remains the action unless the best safe H3 candidate
    clears the calibrated relative-utility margin.  This prevents the horizon
    model from changing an established decision for a negligible predicted win.
    """

    allowed = list(range(len(candidates))) if eligible is None else list(eligible)
    if not allowed:
        return []
    if len(v2_scores) != len(candidates) or any(
        len(values) != len(candidates) for values in predictions.values()
    ):
        raise ValueError("v3-h3 predictions do not match the candidate pool")
    effective = predictions["effective_progress_probability"]
    no_progress = predictions["no_progress_probability"]
    h3_no_progress = predictions["h3_no_progress_probability"]
    utility = predictions["utility"]
    maximum_effective = max(effective[index] for index in allowed)
    effective_shortlist = [
        index
        for index in allowed
        if effective[index]
        >= maximum_effective
        - thresholds["h1_effective_probability_tolerance"]
    ]
    minimum_no_progress = min(no_progress[index] for index in effective_shortlist)
    safe = [
        index
        for index in effective_shortlist
        if no_progress[index]
        <= minimum_no_progress
        + thresholds["h1_no_progress_probability_tolerance"]
    ]
    ranked = sorted(
        safe,
        key=lambda index: (
            -round(float(utility[index]), 12),
            round(float(h3_no_progress[index]), 12),
            -round(float(effective[index]), 12),
            -round(float(v2_scores[index]), 12),
            str(candidates[index]["candidate_id"]),
        ),
    )
    # Match score_online_candidates exactly: scores are rounded to 1e-12 and
    # the lexicographically smallest candidate key wins a numerical tie.
    # Using max(score, candidate_id) here used to reverse that final rule.
    v2_winner = min(
        allowed,
        key=lambda index: (
            -round(float(v2_scores[index]), 12),
            str(
                candidates[index].get(
                    "candidate_key", candidates[index]["candidate_id"]
                )
            ),
        ),
    )
    best = ranked[0]
    if best != v2_winner:
        best_utility = float(utility[best])
        v2_utility = float(utility[v2_winner])
        required = (1.0 + thresholds["minimum_h3_utility_improvement"]) * max(
            0.0, v2_utility
        )
        if (
            best_utility <= v2_utility + 1e-12
            or best_utility + 1e-12 < required
        ):
            return [v2_winner] + [index for index in ranked if index != v2_winner]
    return ranked


@dataclass
class V3ControllerState:
    bundle: V3ControllerBundle
    maximum_distinct_failures: int = 3
    state_anchor_fingerprint: str | None = None
    failed_candidate_ids: set[str] = field(default_factory=set)
    failed_neighborhoods: set[tuple[int, ...]] = field(default_factory=set)
    adaptive_fallback_active: bool = False
    pending_candidate_id: str | None = None
    pending_neighborhood: tuple[int, ...] | None = None
    pending_route: str | None = None
    model_decision_count: int = 0
    adaptive_fallback_decision_count: int = 0
    no_progress_count: int = 0
    hard_failure_count: int = 0
    accepted_noop_count: int = 0
    state_changed_no_reduction_count: int = 0
    conflict_reduced_count: int = 0
    feasible_count: int = 0
    blacklist_addition_count: int = 0
    cache_hit_count: int = 0
    rescued_state_count: int = 0
    longest_unchanged_streak: int = 0
    current_unchanged_streak: int = 0

    def __post_init__(self) -> None:
        if self.maximum_distinct_failures <= 0:
            raise ValueError("v3 failure limit must be positive")

    def _reset_for_state(self, fingerprint: str) -> None:
        self.state_anchor_fingerprint = str(fingerprint)
        self.failed_candidate_ids.clear()
        self.failed_neighborhoods.clear()
        self.adaptive_fallback_active = False
        self.pending_candidate_id = None
        self.pending_neighborhood = None
        self.pending_route = None
        self.current_unchanged_streak = 0

    def begin_state(self, fingerprint: str) -> bool:
        changed = self.state_anchor_fingerprint != str(fingerprint)
        if changed:
            self._reset_for_state(str(fingerprint))
        return changed

    def predictions_required(self, before_fingerprint: str) -> bool:
        self.begin_state(before_fingerprint)
        return not self.adaptive_fallback_active

    def note_cache_hit(self) -> None:
        self.cache_hit_count += 1

    def select(
        self,
        candidates: list[dict[str, Any]],
        v2_scores: list[float],
        predictions: dict[str, list[float]] | None,
        *,
        before_fingerprint: str,
    ) -> tuple[int | None, dict[str, Any]]:
        self.begin_state(before_fingerprint)
        selected: int | None = None
        if not self.adaptive_fallback_active:
            if predictions is None:
                raise ValueError("v3 predictions are required for every model decision")
            eligible = [
                index
                for index, candidate in enumerate(candidates)
                if tuple(sorted(map(int, candidate.get("agents", ()))))
                not in self.failed_neighborhoods
            ]
            if (
                len(self.failed_neighborhoods) >= self.maximum_distinct_failures
                or not eligible
            ):
                self.adaptive_fallback_active = True
            else:
                order = (
                    v3_h3_candidate_order(
                        candidates,
                        predictions,
                        v2_scores,
                        self.bundle.thresholds,
                        eligible=eligible,
                    )
                    if self.bundle.is_horizon
                    else v3_candidate_order(
                        candidates,
                        predictions,
                        v2_scores,
                        self.bundle.thresholds,
                        eligible=eligible,
                    )
                )
                if order:
                    selected = order[0]
                else:
                    self.adaptive_fallback_active = True
        if self.adaptive_fallback_active:
            self.pending_candidate_id = None
            self.pending_neighborhood = None
            self.pending_route = "official_adaptive"
            self.adaptive_fallback_decision_count += 1
            selection_kind = "official_fallback"
        else:
            assert selected is not None
            self.pending_candidate_id = str(candidates[selected]["candidate_id"])
            self.pending_neighborhood = tuple(
                sorted(map(int, candidates[selected].get("agents", ())))
            )
            self.pending_route = "model"
            self.model_decision_count += 1
            selection_kind = "v3"
        return selected, {
            "schema": self.bundle.schema,
            "state_anchor_fingerprint": self.state_anchor_fingerprint,
            "selection_kind": selection_kind,
            "effective_selected_candidate_id": self.pending_candidate_id,
            "failed_candidate_ids": sorted(self.failed_candidate_ids),
            "failed_candidate_count": len(self.failed_candidate_ids),
            "excluded_candidate_count": len(candidates) - len(eligible)
            if not self.adaptive_fallback_active
            else sum(
                tuple(sorted(map(int, candidate.get("agents", ()))))
                in self.failed_neighborhoods
                for candidate in candidates
            ),
            "blacklisted_neighborhood_count": len(self.failed_neighborhoods),
            "maximum_distinct_failures": self.maximum_distinct_failures,
            "adaptive_fallback_active": self.adaptive_fallback_active,
            "route": self.pending_route,
            "predictions": (
                {
                    str(candidate["candidate_id"]): {
                        name: float(values[index])
                        for name, values in predictions.items()
                    }
                    for index, candidate in enumerate(candidates)
                }
                if predictions is not None
                else {}
            ),
        }

    def observe(
        self,
        *,
        before_fingerprint: str,
        after_fingerprint: str,
        replan_success: bool,
        conflicts_before: int,
        conflicts_after: int,
        feasible: bool,
    ) -> dict[str, Any]:
        if self.pending_route is None:
            raise RuntimeError("v3 observation has no pending decision")
        outcome = classify_repair_outcome(
            before_fingerprint=before_fingerprint,
            after_fingerprint=after_fingerprint,
            replan_success=replan_success,
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=feasible,
        )
        setattr(self, f"{outcome}_count", getattr(self, f"{outcome}_count") + 1)
        no_progress = outcome in {"hard_failure", "accepted_noop"}
        if no_progress:
            self.no_progress_count += 1
            self.current_unchanged_streak += 1
            self.longest_unchanged_streak = max(
                self.longest_unchanged_streak, self.current_unchanged_streak
            )
            if (
                self.pending_route == "model"
                and self.pending_candidate_id is not None
                and self.pending_neighborhood is not None
            ):
                if self.pending_candidate_id not in self.failed_candidate_ids:
                    self.failed_candidate_ids.add(self.pending_candidate_id)
                if self.pending_neighborhood not in self.failed_neighborhoods:
                    self.failed_neighborhoods.add(self.pending_neighborhood)
                    self.blacklist_addition_count += 1
                if len(self.failed_neighborhoods) >= self.maximum_distinct_failures:
                    self.adaptive_fallback_active = True
            else:
                self.adaptive_fallback_active = True
        else:
            if self.failed_candidate_ids or self.pending_route == "official_adaptive":
                self.rescued_state_count += 1
            self._reset_for_state(str(after_fingerprint))
        return {
            "repair_outcome": outcome,
            "no_progress": no_progress,
            "state_unchanged": str(before_fingerprint) == str(after_fingerprint),
            "failed_candidate_count_after": len(self.failed_candidate_ids),
            "blacklisted_neighborhood_count_after": len(
                self.failed_neighborhoods
            ),
            "adaptive_fallback_active": self.adaptive_fallback_active,
        }

    def summary(self) -> dict[str, Any]:
        total = self.model_decision_count + self.adaptive_fallback_decision_count
        return {
            "schema": self.bundle.schema,
            "inference_backends": list(self.bundle.inference_backends),
            "model_decision_count": self.model_decision_count,
            "adaptive_fallback_decision_count": self.adaptive_fallback_decision_count,
            "adaptive_fallback_fraction": (
                self.adaptive_fallback_decision_count / total if total else 0.0
            ),
            "no_progress_count": self.no_progress_count,
            "hard_failure_count": self.hard_failure_count,
            "accepted_noop_count": self.accepted_noop_count,
            "state_changed_no_reduction_count": self.state_changed_no_reduction_count,
            "conflict_reduced_count": self.conflict_reduced_count,
            "feasible_count": self.feasible_count,
            "blacklist_addition_count": self.blacklist_addition_count,
            "blacklisted_neighborhood_count": len(self.failed_neighborhoods),
            "cache_hit_count": self.cache_hit_count,
            "rescued_state_count": self.rescued_state_count,
            "longest_unchanged_streak": self.longest_unchanged_streak,
        }


__all__ = [
    "V3_BUNDLE_SCHEMA",
    "V3_H3_BUNDLE_SCHEMA",
    "V3ControllerBundle",
    "V3ControllerState",
    "load_v3_controller_bundle",
    "v3_candidate_order",
    "v3_h3_candidate_order",
]
