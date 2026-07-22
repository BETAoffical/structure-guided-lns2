from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.feature_schema_v2 import FEATURE_SCHEMA_ID, FEATURE_SCHEMA_SHA256
from experiments.repair_collection import _fingerprint


REPAIR_AWARE_CONFIG_SCHEMA = "lns2.repair_aware_controller.v2"
LEGACY_REPAIR_AWARE_CONFIG_SCHEMA = "lns2.repair_aware_controller.v1"
REPAIR_AWARE_BUNDLE_SCHEMA = "lns2.repair_aware_bundle.v2"
LEGACY_REPAIR_AWARE_BUNDLE_SCHEMA = "lns2.repair_aware_bundle.v1"
PORTABLE_SCALAR_MODEL_SCHEMA = "lns2.portable_scalar_hist_gbdt.v1"
REPAIR_AWARE_MODES = ("rescue-only", "shadow", "guarded-tiebreak")
REPAIR_OUTCOMES = (
    "hard_failure",
    "accepted_noop",
    "state_changed_no_reduction",
    "conflict_reduced",
    "feasible",
)


def _row_features(row: dict[str, Any], profile: str) -> dict[str, float]:
    if "feature_values" in row:
        if str(row.get("feature_profile")) != profile:
            raise ValueError("dense row profile does not match repair-aware model")
        names = tuple(map(str, row.get("feature_names", ())))
        values = tuple(map(float, row.get("feature_values", ())))
        if len(names) != len(values) or len(names) != len(set(names)):
            raise ValueError("invalid dense feature row")
        return dict(zip(names, values))
    return {
        str(name): float(value)
        for name, value in dict(row["features"][profile]).items()
    }


def adaptive_feature_row(row: dict[str, Any], profile: str = "realized_dynamic") -> dict[str, Any]:
    """Build the deterministic state-only pseudo row used for Adaptive routing."""

    features = _row_features(row, profile)
    values = {
        name: float(value) if name.startswith(("state.", "context.")) else 0.0
        for name, value in features.items()
    }
    return {
        "candidate_id": "official_adaptive",
        "candidate_key": "official_adaptive",
        "feature_profile": profile,
        "feature_names": tuple(values),
        "feature_values": tuple(values.values()),
    }


@dataclass(frozen=True)
class RepairAwareConfig:
    mode: str
    max_model_rescues: int | None
    same_candidate_attempt_limit: int
    lazy_neighborhood_sizes: tuple[int, ...]
    terminal_fallback: str
    fallback_until_state_change: bool
    reset_on_state_fingerprint_change: bool

    def payload(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_AWARE_CONFIG_SCHEMA,
            "mode": self.mode,
            "max_model_rescues": self.max_model_rescues,
            "same_candidate_attempt_limit": self.same_candidate_attempt_limit,
            "lazy_neighborhood_sizes": list(self.lazy_neighborhood_sizes),
            "terminal_fallback": self.terminal_fallback,
            "fallback_until_state_change": self.fallback_until_state_change,
            "reset_on_state_fingerprint_change": self.reset_on_state_fingerprint_change,
        }


def load_repair_aware_config(
    value: str | Path | dict[str, Any],
) -> RepairAwareConfig:
    raw = dict(read_json(Path(value).resolve())) if not isinstance(value, dict) else dict(value)
    schema = str(raw.get("schema"))
    if schema not in {
        REPAIR_AWARE_CONFIG_SCHEMA,
        LEGACY_REPAIR_AWARE_CONFIG_SCHEMA,
    }:
        raise ValueError("unexpected repair-aware controller schema")
    mode = str(raw.get("mode", "rescue-only"))
    if mode not in REPAIR_AWARE_MODES:
        raise ValueError(f"unsupported repair-aware mode: {mode}")
    maximum_raw = raw.get("max_model_rescues")
    maximum = int(maximum_raw) if maximum_raw is not None else None
    if maximum is not None and maximum <= 0:
        raise ValueError("max_model_rescues must be positive")
    attempt_limit = int(raw.get("same_candidate_attempt_limit", 1 if schema == LEGACY_REPAIR_AWARE_CONFIG_SCHEMA else 2))
    if attempt_limit <= 0:
        raise ValueError("same_candidate_attempt_limit must be positive")
    lazy_sizes = tuple(map(int, raw.get("lazy_neighborhood_sizes", ())))
    if len(lazy_sizes) != len(set(lazy_sizes)) or any(value <= 0 for value in lazy_sizes):
        raise ValueError("lazy_neighborhood_sizes must be unique positive integers")
    fallback = str(raw.get("terminal_fallback", "official_adaptive"))
    if fallback != "official_adaptive":
        raise ValueError("repair-aware fallback must be official_adaptive")
    fallback_until_change = bool(
        raw.get(
            "fallback_until_state_change",
            not bool(raw.get("refresh_after_fallback", False)),
        )
    )
    if schema == REPAIR_AWARE_CONFIG_SCHEMA and not fallback_until_change:
        raise ValueError("repair-aware fallback must persist until the state changes")
    if not bool(raw.get("reset_on_state_fingerprint_change", True)):
        raise ValueError("repair-aware state must reset after a fingerprint change")
    return RepairAwareConfig(
        mode=mode,
        max_model_rescues=maximum,
        same_candidate_attempt_limit=attempt_limit,
        lazy_neighborhood_sizes=lazy_sizes,
        terminal_fallback=fallback,
        fallback_until_state_change=fallback_until_change,
        reset_on_state_fingerprint_change=True,
    )


def classify_repair_outcome(
    *,
    before_fingerprint: str,
    after_fingerprint: str,
    replan_success: bool,
    conflicts_before: int,
    conflicts_after: int,
    feasible: bool = False,
) -> str:
    before_conflicts = int(conflicts_before)
    after_conflicts = int(conflicts_after)
    changed = str(before_fingerprint) != str(after_fingerprint)
    if not replan_success and changed:
        raise ValueError("failed PP changed the solver state instead of rolling back")
    if not changed and before_conflicts != after_conflicts:
        raise ValueError("unchanged state fingerprint has different conflict counts")
    if feasible or after_conflicts == 0:
        if not changed and before_conflicts > 0:
            raise ValueError("an unchanged conflicting state cannot become feasible")
        return "feasible"
    if not replan_success:
        return "hard_failure"
    if not changed:
        return "accepted_noop"
    if after_conflicts < before_conflicts:
        return "conflict_reduced"
    return "state_changed_no_reduction"


@dataclass
class PortableScalarModel:
    name: str
    profile: str
    feature_names: list[str]
    baseline: float
    trees: list[list[dict[str, Any]]]
    transform: str
    semantic_fingerprint: str
    native_predictor: Any | None = None
    declared_feature_count: int | None = None

    @property
    def source_feature_count(self) -> int:
        return int(self.declared_feature_count or len(self.feature_names))

    @property
    def inference_backend(self) -> str:
        return (
            "native-portable-tree"
            if self.native_predictor is not None
            else "python-portable-tree"
        )

    def _vectors(self, rows: Iterable[dict[str, Any]]) -> list[list[float]]:
        vectors = []
        for row in rows:
            features = _row_features(row, self.profile)
            missing = set(self.feature_names) - set(features)
            if missing:
                raise ValueError(
                    f"repair-aware row is missing model features: {sorted(missing)}"
                )
            vector = [float(features[name]) for name in self.feature_names]
            if any(not math.isfinite(value) for value in vector):
                raise ValueError("repair-aware model received non-finite features")
            vectors.append(vector)
        return vectors

    def _python_raw(self, vectors: list[list[float]]) -> list[float]:
        outputs = []
        for vector in vectors:
            raw = float(self.baseline)
            for nodes in self.trees:
                index = 0
                while not bool(nodes[index]["is_leaf"]):
                    node = nodes[index]
                    value = vector[int(node["feature_idx"])]
                    go_left = (
                        math.isnan(value) and bool(node["missing_go_to_left"])
                    ) or (
                        not math.isnan(value)
                        and value <= float(node["num_threshold"])
                    )
                    index = int(node["left"] if go_left else node["right"])
                raw += float(nodes[index]["value"])
            outputs.append(raw)
        return outputs

    def predict(self, rows: list[dict[str, Any]]) -> list[float]:
        vectors = self._vectors(rows)
        if self.native_predictor is not None:
            method = (
                self.native_predictor.predict_positive
                if self.transform == "sigmoid"
                else self.native_predictor.predict_raw
            )
            return list(map(float, method(vectors)))
        raw = self._python_raw(vectors)
        if self.transform == "identity":
            return raw
        values = []
        for value in raw:
            if value >= 0.0:
                values.append(1.0 / (1.0 + math.exp(-value)))
            else:
                exponential = math.exp(value)
                values.append(exponential / (1.0 + exponential))
        return values


def _connect_native(model: PortableScalarModel) -> PortableScalarModel:
    try:
        import lns2_env
    except ImportError:
        return model
    predictor = getattr(lns2_env, "PortableTreeEnsemble", None)
    if predictor is None:
        return model
    model.native_predictor = predictor(model.baseline, model.trees)
    return model


def compact_portable_scalar_model(
    model: PortableScalarModel,
) -> PortableScalarModel:
    """Project a fitted scalar ensemble onto features used by its split nodes."""

    used_indices = sorted(
        {
            int(node["feature_idx"])
            for tree in model.trees
            for node in tree
            if not bool(node["is_leaf"])
        }
    )
    if any(index < 0 or index >= len(model.feature_names) for index in used_indices):
        raise ValueError("portable scalar model references an invalid feature index")
    if used_indices == list(range(len(model.feature_names))):
        return _connect_native(model)
    compact_index = {
        source_index: target_index
        for target_index, source_index in enumerate(used_indices)
    }
    compact_trees = []
    for source_tree in model.trees:
        output_tree = []
        for source_node in source_tree:
            node = dict(source_node)
            node["feature_idx"] = (
                0
                if bool(node["is_leaf"])
                else compact_index[int(node["feature_idx"])]
            )
            output_tree.append(node)
        compact_trees.append(output_tree)
    return _connect_native(
        PortableScalarModel(
            name=model.name,
            profile=model.profile,
            feature_names=[model.feature_names[index] for index in used_indices],
            baseline=model.baseline,
            trees=compact_trees,
            transform=model.transform,
            semantic_fingerprint=model.semantic_fingerprint,
            declared_feature_count=model.source_feature_count,
        )
    )


def load_portable_scalar_model(
    payload: dict[str, Any], *, compact_features: bool = False
) -> PortableScalarModel:
    if str(payload.get("schema")) != PORTABLE_SCALAR_MODEL_SCHEMA:
        raise ValueError("unexpected portable scalar model schema")
    transform = str(payload.get("transform"))
    if transform not in {"identity", "sigmoid"}:
        raise ValueError("portable scalar model has an invalid transform")
    names = list(map(str, payload.get("feature_names", ())))
    if not names or len(names) != len(set(names)):
        raise ValueError("portable scalar model has invalid feature names")
    expected = _fingerprint(
        {
            "name": payload.get("name"),
            "profile": payload.get("profile"),
            "feature_names": names,
            "baseline": payload.get("baseline"),
            "trees": payload.get("trees"),
            "transform": transform,
        }
    )
    if str(payload.get("semantic_fingerprint")) != expected:
        raise ValueError("portable scalar model semantic fingerprint mismatch")
    model = PortableScalarModel(
        name=str(payload["name"]),
        profile=str(payload["profile"]),
        feature_names=names,
        baseline=float(payload["baseline"]),
        trees=list(payload["trees"]),
        transform=transform,
        semantic_fingerprint=expected,
        declared_feature_count=len(names),
    )
    return (
        compact_portable_scalar_model(model)
        if compact_features
        else _connect_native(model)
    )


@dataclass(frozen=True)
class RepairAwareBundle:
    models: dict[str, PortableScalarModel]
    thresholds: dict[str, float]
    guarded_tiebreak_eligible: bool
    manifest: dict[str, Any]
    report: dict[str, Any]

    @property
    def selected_max_model_rescues(self) -> int:
        return int(self.manifest.get("selected_max_model_rescues", 3))

    @property
    def is_wall_clock_bundle(self) -> bool:
        return "log_repair_seconds" in self.models

    def predict(self, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
        progress = self.models["progress_probability"].predict(rows)
        reduction = self.models["conflict_reduction"].predict(rows)
        result = {
            "progress_probability": [min(1.0, max(0.0, value)) for value in progress],
            "conflict_reduction": [max(0.0, value) for value in reduction],
        }
        if self.is_wall_clock_bundle:
            log_seconds = self.models["log_repair_seconds"].predict(rows)
            hard_failure = self.models["hard_failure_probability"].predict(rows)
            result["repair_seconds"] = [
                max(1e-9, math.expm1(min(value, 50.0))) for value in log_seconds
            ]
            result["hard_failure_probability"] = [
                min(1.0, max(0.0, value)) for value in hard_failure
            ]
            result["efficiency"] = [
                max(0.0, probability)
                * max(0.0, 1.0 - failure)
                * max(0.0, expected_reduction)
                / max(1e-9, seconds)
                for probability, failure, expected_reduction, seconds in zip(
                    result["progress_probability"],
                    result["hard_failure_probability"],
                    result["conflict_reduction"],
                    result["repair_seconds"],
                )
            ]
        else:
            log_generated = self.models["log_generated"].predict(rows)
            result["generated"] = [
                max(0.0, math.expm1(min(value, 50.0))) for value in log_generated
            ]
        return result


def load_repair_aware_bundle(path: str | Path) -> RepairAwareBundle:
    root = Path(path).resolve()
    manifest = dict(read_json(root / "repair_aware_manifest.json"))
    schema = str(manifest.get("schema"))
    if schema not in {REPAIR_AWARE_BUNDLE_SCHEMA, LEGACY_REPAIR_AWARE_BUNDLE_SCHEMA}:
        raise ValueError("unexpected repair-aware bundle schema")
    if str(manifest.get("feature_schema_id")) != FEATURE_SCHEMA_ID or str(
        manifest.get("feature_schema_sha256")
    ) != FEATURE_SCHEMA_SHA256:
        raise ValueError("repair-aware bundle feature schema mismatch")
    models = {}
    for name, raw in dict(manifest.get("models", {})).items():
        row = dict(raw)
        model_path = root / str(row["file"])
        if sha256_file(model_path) != str(row["sha256"]):
            raise ValueError(f"repair-aware model SHA256 mismatch: {name}")
        model = load_portable_scalar_model(dict(read_json(model_path)))
        if model.name != name:
            raise ValueError(f"repair-aware model name mismatch: {name}")
        models[name] = model
    required = (
        {
            "progress_probability",
            "conflict_reduction",
            "log_repair_seconds",
            "hard_failure_probability",
        }
        if schema == REPAIR_AWARE_BUNDLE_SCHEMA
        else {"progress_probability", "conflict_reduction", "log_generated"}
    )
    if set(models) != required:
        raise ValueError("repair-aware bundle has an incomplete model set")
    report_row = dict(manifest.get("training_report", {}))
    report_path = root / str(report_row["file"])
    if sha256_file(report_path) != str(report_row["sha256"]):
        raise ValueError("repair-aware training report SHA256 mismatch")
    thresholds = {
        str(name): float(value)
        for name, value in dict(manifest.get("thresholds", {})).items()
    }
    expected_thresholds = (
        {
            "minimum_predicted_efficiency",
            "adaptive_efficiency_margin",
        }
        if schema == REPAIR_AWARE_BUNDLE_SCHEMA
        else {
            "rescue_probability_tolerance",
            "tie_score_gap_fraction",
            "tie_progress_margin",
            "tie_minimum_reduction_ratio",
            "tie_maximum_generated_ratio",
        }
    )
    if set(thresholds) != expected_thresholds:
        raise ValueError("repair-aware bundle has incomplete frozen thresholds")
    return RepairAwareBundle(
        models=models,
        thresholds=thresholds,
        guarded_tiebreak_eligible=bool(
            manifest.get("guarded_tiebreak_eligible", False)
        ),
        manifest=manifest,
        report=dict(read_json(report_path)),
    )


def repair_aware_order(
    candidates: list[dict[str, Any]],
    predictions: dict[str, list[float]],
    main_scores: list[float],
    *,
    probability_tolerance: float = 0.0,
    eligible: Iterable[int] | None = None,
) -> list[int]:
    allowed = list(range(len(candidates))) if eligible is None else list(eligible)
    if not allowed:
        return []
    if "efficiency" in predictions:
        efficiency = predictions["efficiency"]
        probabilities = predictions["progress_probability"]
        hard_failure = predictions["hard_failure_probability"]
        return sorted(
            allowed,
            key=lambda index: (
                -round(efficiency[index], 12),
                -round(probabilities[index], 12),
                round(hard_failure[index], 12),
                -round(main_scores[index], 12),
                str(candidates[index]["candidate_id"]),
            ),
        )
    probabilities = predictions["progress_probability"]
    maximum = max(probabilities[index] for index in allowed)
    shortlist = [index for index in allowed if probabilities[index] >= maximum - probability_tolerance]
    generated = predictions["generated"]
    reduction = predictions["conflict_reduction"]
    return sorted(
        shortlist,
        key=lambda index: (
            -round(reduction[index], 12),
            round(generated[index], 12),
            -round(main_scores[index], 12),
            str(candidates[index]["candidate_id"]),
        ),
    )


def guarded_tiebreak_candidate(
    candidates: list[dict[str, Any]],
    predictions: dict[str, list[float]],
    main_scores: list[float],
    base_index: int,
    thresholds: dict[str, float],
) -> int:
    stable = [round(float(value), 12) for value in main_scores]
    order = sorted(
        range(len(candidates)),
        key=lambda index: (-stable[index], str(candidates[index]["candidate_id"])),
    )
    scale = max(1.0, float(len(candidates) - 1))
    progress = predictions["progress_probability"]
    reduction = predictions["conflict_reduction"]
    generated = predictions["generated"]
    eligible = []
    for index in order[:3]:
        if index == base_index:
            continue
        score_gap = (stable[base_index] - stable[index]) / scale
        if score_gap > thresholds["tie_score_gap_fraction"]:
            continue
        if progress[index] <= progress[base_index] + thresholds["tie_progress_margin"]:
            continue
        if reduction[index] < (
            thresholds["tie_minimum_reduction_ratio"] * reduction[base_index]
        ):
            continue
        if generated[index] > (
            thresholds["tie_maximum_generated_ratio"] * generated[base_index]
        ):
            continue
        eligible.append(index)
    ordered = repair_aware_order(
        candidates,
        predictions,
        main_scores,
        probability_tolerance=thresholds["rescue_probability_tolerance"],
        eligible=eligible,
    )
    return ordered[0] if ordered else base_index


@dataclass
class RepairAwareState:
    config: RepairAwareConfig
    bundle: RepairAwareBundle
    state_anchor_fingerprint: str | None = None
    candidate_attempt_counts: dict[str, int] = field(default_factory=dict)
    rescue_attempts: int = 0
    needs_rescue: bool = False
    refresh_required: bool = False
    adaptive_fallback_active: bool = False
    pending_candidate_id: str | None = None
    pending_route: str | None = None
    pending_selection_kind: str | None = None
    no_progress_count: int = 0
    hard_failure_count: int = 0
    accepted_noop_count: int = 0
    state_changed_no_reduction_count: int = 0
    conflict_reduced_count: int = 0
    feasible_count: int = 0
    rescue_selection_count: int = 0
    fallback_count: int = 0
    cache_hit_count: int = 0
    cache_refresh_count: int = 0
    shadow_difference_count: int = 0
    tiebreak_override_count: int = 0
    rescued_state_count: int = 0
    longest_unchanged_streak: int = 0
    current_unchanged_streak: int = 0

    def _reset_for_state(self, fingerprint: str) -> None:
        self.state_anchor_fingerprint = str(fingerprint)
        self.candidate_attempt_counts.clear()
        self.rescue_attempts = 0
        self.needs_rescue = False
        self.refresh_required = False
        self.adaptive_fallback_active = False
        self.pending_candidate_id = None
        self.pending_route = None
        self.pending_selection_kind = None
        self.current_unchanged_streak = 0

    def begin_state(self, fingerprint: str) -> bool:
        changed = self.state_anchor_fingerprint != str(fingerprint)
        if changed:
            self._reset_for_state(str(fingerprint))
        return changed

    def consume_refresh(self) -> bool:
        value = self.refresh_required
        self.refresh_required = False
        if value:
            self.cache_refresh_count += 1
        return value

    def note_cache_hit(self) -> None:
        self.cache_hit_count += 1

    def predictions_required(self, before_fingerprint: str) -> bool:
        """Return whether the auxiliary models can affect this decision."""
        self.begin_state(before_fingerprint)
        return bool(
            self.needs_rescue and not self.adaptive_fallback_active
            or self.config.mode == "shadow"
            or (
                self.config.mode == "guarded-tiebreak"
                and self.bundle.guarded_tiebreak_eligible
            )
        )

    def select(
        self,
        candidates: list[dict[str, Any]],
        main_scores: list[float],
        base_index: int,
        predictions: dict[str, list[float]] | None,
        *,
        before_fingerprint: str,
        adaptive_prediction: dict[str, float] | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        predictions_required = self.predictions_required(before_fingerprint)
        if predictions_required and predictions is None:
            raise ValueError("repair-aware predictions are required for this decision")
        shadow_index: int | None = None
        if predictions is not None:
            shadow_order = repair_aware_order(
                candidates,
                predictions,
                main_scores,
                probability_tolerance=self.bundle.thresholds.get(
                    "rescue_probability_tolerance", 0.0
                ),
            )
            shadow_index = shadow_order[0] if shadow_order else base_index
            if shadow_index != base_index:
                self.shadow_difference_count += 1

        selection_kind = "base"
        selected_index: int | None = base_index
        guarded_enabled = (
            self.config.mode == "guarded-tiebreak"
            and self.bundle.guarded_tiebreak_eligible
        )
        if self.adaptive_fallback_active:
            selected_index = None
            selection_kind = "official_fallback"
            self.fallback_count += 1
        elif self.needs_rescue:
            assert predictions is not None
            remaining = [
                index
                for index, candidate in enumerate(candidates)
                if self.candidate_attempt_counts.get(str(candidate["candidate_id"]), 0)
                < self.config.same_candidate_attempt_limit
            ]
            maximum = self.config.max_model_rescues or self.bundle.selected_max_model_rescues
            if self.rescue_attempts < maximum and remaining:
                ordered = repair_aware_order(
                    candidates,
                    predictions,
                    main_scores,
                    probability_tolerance=self.bundle.thresholds.get(
                        "rescue_probability_tolerance", 0.0
                    ),
                    eligible=remaining,
                )
                selected_index = ordered[0] if ordered else None
                if (
                    selected_index is not None
                    and "efficiency" in predictions
                    and adaptive_prediction is not None
                ):
                    selected_efficiency = float(
                        predictions["efficiency"][selected_index]
                    )
                    adaptive_efficiency = float(adaptive_prediction["efficiency"])
                    minimum = float(
                        self.bundle.thresholds["minimum_predicted_efficiency"]
                    )
                    margin = float(
                        self.bundle.thresholds["adaptive_efficiency_margin"]
                    )
                    if (
                        selected_efficiency < minimum
                        or adaptive_efficiency > selected_efficiency * (1.0 + margin)
                    ):
                        selected_index = None
                selection_kind = "rescue"
                if selected_index is None:
                    selection_kind = "official_fallback"
                    self.adaptive_fallback_active = True
                    self.fallback_count += 1
                else:
                    self.rescue_selection_count += 1
            else:
                selected_index = None
                selection_kind = "official_fallback"
                self.adaptive_fallback_active = True
                self.fallback_count += 1
        elif guarded_enabled:
            assert predictions is not None
            selected_index = guarded_tiebreak_candidate(
                candidates,
                predictions,
                main_scores,
                base_index,
                self.bundle.thresholds,
            )
            if selected_index != base_index:
                selection_kind = "guarded_tiebreak"
                self.tiebreak_override_count += 1

        self.pending_candidate_id = (
            str(candidates[selected_index]["candidate_id"])
            if selected_index is not None
            else None
        )
        self.pending_route = "model" if selected_index is not None else "official_adaptive"
        self.pending_selection_kind = selection_kind
        return selected_index, {
            "mode": self.config.mode,
            "guarded_tiebreak_eligible": self.bundle.guarded_tiebreak_eligible,
            "guarded_tiebreak_enabled": guarded_enabled,
            "state_anchor_fingerprint": self.state_anchor_fingerprint,
            "selection_kind": selection_kind,
            "base_selected_candidate_id": str(candidates[base_index]["candidate_id"]),
            "effective_selected_candidate_id": self.pending_candidate_id,
            "shadow_selected_candidate_id": (
                str(candidates[shadow_index]["candidate_id"])
                if shadow_index is not None
                else None
            ),
            "base_selection_preserved": selected_index == base_index,
            "failed_candidate_count": sum(
                count >= self.config.same_candidate_attempt_limit
                for count in self.candidate_attempt_counts.values()
            ),
            "candidate_attempt_counts": dict(sorted(self.candidate_attempt_counts.items())),
            "rescue_attempts": self.rescue_attempts,
            "adaptive_fallback_active": self.adaptive_fallback_active,
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
            "route": self.pending_route,
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
            raise RuntimeError("repair-aware observation has no pending decision")
        outcome = classify_repair_outcome(
            before_fingerprint=before_fingerprint,
            after_fingerprint=after_fingerprint,
            replan_success=replan_success,
            conflicts_before=conflicts_before,
            conflicts_after=conflicts_after,
            feasible=feasible,
        )
        setattr(self, f"{outcome}_count", getattr(self, f"{outcome}_count") + 1)
        unchanged = str(before_fingerprint) == str(after_fingerprint)
        no_progress = outcome in {"hard_failure", "accepted_noop"}
        if no_progress:
            self.no_progress_count += 1
            self.current_unchanged_streak += 1
            self.longest_unchanged_streak = max(
                self.longest_unchanged_streak, self.current_unchanged_streak
            )
            if self.pending_route == "model" and self.pending_candidate_id is not None:
                self.candidate_attempt_counts[self.pending_candidate_id] = (
                    self.candidate_attempt_counts.get(self.pending_candidate_id, 0) + 1
                )
                if self.pending_selection_kind == "rescue":
                    self.rescue_attempts += 1
                self.needs_rescue = True
            else:
                self.needs_rescue = True
                self.adaptive_fallback_active = True
        else:
            if self.pending_selection_kind in {"rescue", "official_fallback"}:
                self.rescued_state_count += 1
            self._reset_for_state(str(after_fingerprint))
        return {
            "repair_outcome": outcome,
            "no_progress": no_progress,
            "state_unchanged": unchanged,
            "failed_candidate_count_after": sum(
                count >= self.config.same_candidate_attempt_limit
                for count in self.candidate_attempt_counts.values()
            ),
            "rescue_attempts_after": self.rescue_attempts,
            "refresh_required": self.refresh_required,
            "adaptive_fallback_active": self.adaptive_fallback_active,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "schema": REPAIR_AWARE_CONFIG_SCHEMA,
            "mode": self.config.mode,
            "no_progress_count": self.no_progress_count,
            "hard_failure_count": self.hard_failure_count,
            "accepted_noop_count": self.accepted_noop_count,
            "state_changed_no_reduction_count": self.state_changed_no_reduction_count,
            "conflict_reduced_count": self.conflict_reduced_count,
            "feasible_count": self.feasible_count,
            "rescue_selection_count": self.rescue_selection_count,
            "fallback_count": self.fallback_count,
            "cache_hit_count": self.cache_hit_count,
            "cache_refresh_count": self.cache_refresh_count,
            "shadow_difference_count": self.shadow_difference_count,
            "tiebreak_override_count": self.tiebreak_override_count,
            "rescued_state_count": self.rescued_state_count,
            "longest_unchanged_streak": self.longest_unchanged_streak,
        }


__all__ = [
    "LEGACY_REPAIR_AWARE_BUNDLE_SCHEMA",
    "PORTABLE_SCALAR_MODEL_SCHEMA",
    "REPAIR_AWARE_BUNDLE_SCHEMA",
    "REPAIR_AWARE_CONFIG_SCHEMA",
    "REPAIR_AWARE_MODES",
    "REPAIR_OUTCOMES",
    "PortableScalarModel",
    "RepairAwareBundle",
    "RepairAwareConfig",
    "RepairAwareState",
    "adaptive_feature_row",
    "classify_repair_outcome",
    "compact_portable_scalar_model",
    "guarded_tiebreak_candidate",
    "load_portable_scalar_model",
    "load_repair_aware_bundle",
    "load_repair_aware_config",
    "repair_aware_order",
]
