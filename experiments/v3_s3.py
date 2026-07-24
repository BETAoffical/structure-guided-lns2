from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments._common import read_json, sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES
from experiments.repair_aware import PortableScalarModel, load_portable_scalar_model
from experiments.repair_collection import _fingerprint


V3_S3_BUNDLE_SCHEMA = "lns2.v3_s3_controller_bundle.v2"
V3_S3_LEGACY_FEATURE_SCHEMA_ID = "lns2.v3_s3_features.v1"
V3_S3_FEATURE_SCHEMA_ID = "lns2.v3_s3_features.v2"
V3_S3_LEGACY_OBJECTIVE_ID = "v3-s3-runtime-reachable-sequence-v2"
V3_S3_OBJECTIVE_ID = "v3-s3-runtime-reachable-sequence-v3"
V3_S3_PROFILE = "v3_s3"
S3_FAMILIES = ("collision", "target", "random")
S3_SIZES = (4, 8, 16)
S3_REPRESENTATIVES = (0, 1)
S3_HORIZON = 3


@dataclass(frozen=True, order=True)
class S3ActionTemplate:
    family: str
    requested_size: int
    representative: int

    def __post_init__(self) -> None:
        if self.family not in S3_FAMILIES:
            raise ValueError(f"unsupported S3 family: {self.family}")
        if int(self.requested_size) not in S3_SIZES:
            raise ValueError(f"unsupported S3 size: {self.requested_size}")
        if int(self.representative) not in S3_REPRESENTATIVES:
            raise ValueError(
                f"unsupported S3 representative: {self.representative}"
            )

    @property
    def family_key(self) -> str:
        return f"{self.family}:{int(self.requested_size)}"

    @property
    def key(self) -> str:
        return (
            f"{self.family}:size{int(self.requested_size)}:"
            f"rep{int(self.representative)}"
        )

    def payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "requested_size": int(self.requested_size),
            "representative": int(self.representative),
            "template_key": self.key,
        }

    @classmethod
    def from_payload(cls, value: dict[str, Any]) -> "S3ActionTemplate":
        return cls(
            family=str(value["family"]),
            requested_size=int(value["requested_size"]),
            representative=int(value["representative"]),
        )


S3_ACTION_TEMPLATES = tuple(
    S3ActionTemplate(family, size, representative)
    for family in S3_FAMILIES
    for size in S3_SIZES
    for representative in S3_REPRESENTATIVES
)

S3_LEGACY_TEMPORAL_FEATURE_NAMES = (
    "history.available_steps",
    "history.recent_no_progress_length",
    "history.last_conflict_reduction",
    "history.mean3_conflict_reduction",
    "history.last_repair_seconds",
    "history.mean3_repair_seconds",
    "history.state_change_rate3",
    "history.previous_size_ratio_agents",
)
S3_WALL_TIME_FEATURE_NAMES = (
    "history.last_repair_seconds",
    "history.mean3_repair_seconds",
)
S3_TEMPORAL_FEATURE_NAMES = tuple(
    name
    for name in S3_LEGACY_TEMPORAL_FEATURE_NAMES
    if name not in S3_WALL_TIME_FEATURE_NAMES
)


def s3_temporal_context(
    history: list[dict[str, Any]],
    agent_count: int,
    *,
    include_wall_time: bool = False,
) -> dict[str, float]:
    """Summarize deterministic information available before the next decision.

    ``include_wall_time`` exists only to keep historical v1 bundles replayable.
    New v2 feature bundles never include measured wall-clock time as an input.
    """

    recent = history[-3:]
    reductions = [float(row["conflict_reduction"]) for row in recent]
    no_progress_length = 0
    for row in reversed(history):
        if not bool(row["no_progress"]):
            break
        no_progress_length += 1
    previous_size = int(history[-1]["neighborhood_size"]) if history else 0
    result = {
        "history.available_steps": float(len(recent)),
        "history.recent_no_progress_length": float(no_progress_length),
        "history.last_conflict_reduction": reductions[-1] if reductions else 0.0,
        "history.mean3_conflict_reduction": (
            statistics.fmean(reductions) if reductions else 0.0
        ),
        "history.state_change_rate3": (
            statistics.fmean(float(row["state_changed"]) for row in recent)
            if recent
            else 0.0
        ),
        "history.previous_size_ratio_agents": (
            float(previous_size) / max(1, int(agent_count))
        ),
    }
    if include_wall_time:
        seconds = [float(row["repair_seconds"]) for row in recent]
        result.update(
            {
                "history.last_repair_seconds": seconds[-1] if seconds else 0.0,
                "history.mean3_repair_seconds": (
                    statistics.fmean(seconds) if seconds else 0.0
                ),
            }
        )
    return result


def _template_feature_names(step: int) -> tuple[str, ...]:
    prefix = f"sequence.step{int(step)}"
    return (
        *(f"{prefix}.family={family}" for family in S3_FAMILIES),
        *(f"{prefix}.size={size}" for size in S3_SIZES),
        *(f"{prefix}.representative={value}" for value in S3_REPRESENTATIVES),
        f"{prefix}.requested_size",
        f"{prefix}.requested_size_ratio_agents",
    )


S3_TEMPLATE_FEATURE_NAMES = tuple(
    name
    for step in range(1, S3_HORIZON + 1)
    for name in _template_feature_names(step)
)
V3_S3_FULL_FEATURE_NAMES = tuple(
    (
        *PROFILE_FEATURE_NAMES["realized_dynamic"],
        *S3_TEMPORAL_FEATURE_NAMES,
        *S3_TEMPLATE_FEATURE_NAMES,
    )
)
V3_S3_LEGACY_FULL_FEATURE_NAMES = tuple(
    (
        *PROFILE_FEATURE_NAMES["realized_dynamic"],
        *S3_LEGACY_TEMPORAL_FEATURE_NAMES,
        *S3_TEMPLATE_FEATURE_NAMES,
    )
)


def _schema_payload() -> dict[str, Any]:
    return {
        "schema": V3_S3_FEATURE_SCHEMA_ID,
        "schema_version": 2,
        "profile": V3_S3_PROFILE,
        "base_profile": "realized_dynamic",
        "base_feature_names": list(PROFILE_FEATURE_NAMES["realized_dynamic"]),
        "temporal_feature_names": list(S3_TEMPORAL_FEATURE_NAMES),
        "template_feature_names": list(S3_TEMPLATE_FEATURE_NAMES),
        "feature_names": list(V3_S3_FULL_FEATURE_NAMES),
        "horizon": S3_HORIZON,
        "action_template_count": len(S3_ACTION_TEMPLATES),
        "wall_clock_features": [],
    }


V3_S3_FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        _schema_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()


def _legacy_schema_payload() -> dict[str, Any]:
    return {
        "schema": V3_S3_LEGACY_FEATURE_SCHEMA_ID,
        "schema_version": 1,
        "profile": V3_S3_PROFILE,
        "base_profile": "realized_dynamic",
        "base_feature_names": list(PROFILE_FEATURE_NAMES["realized_dynamic"]),
        "temporal_feature_names": list(S3_LEGACY_TEMPORAL_FEATURE_NAMES),
        "template_feature_names": list(S3_TEMPLATE_FEATURE_NAMES),
        "feature_names": list(V3_S3_LEGACY_FULL_FEATURE_NAMES),
        "horizon": S3_HORIZON,
        "action_template_count": len(S3_ACTION_TEMPLATES),
    }


V3_S3_LEGACY_FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        _legacy_schema_payload(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()


def v3_s3_feature_schema_manifest() -> dict[str, Any]:
    return {**_schema_payload(), "sha256": V3_S3_FEATURE_SCHEMA_SHA256}


def candidate_template_indices(
    candidates: list[dict[str, Any]],
) -> dict[str, int]:
    """Map stable action templates to generated representative candidates."""

    result: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        ranks = {
            str(name): int(value)
            for name, value in dict(
                candidate.get("selection_rank_by_family", {})
            ).items()
        }
        for family_key in map(str, candidate.get("selection_families", ())):
            family, separator, size_text = family_key.partition(":")
            if not separator or family not in S3_FAMILIES:
                continue
            try:
                size = int(size_text)
            except ValueError:
                continue
            rank = ranks.get(family_key)
            if size not in S3_SIZES or rank not in S3_REPRESENTATIVES:
                continue
            template = S3ActionTemplate(family, size, int(rank))
            previous = result.setdefault(template.key, index)
            if previous != index:
                raise ValueError(f"duplicate S3 candidate template: {template.key}")
    return result


def balanced_sequence_templates(
    state_key: str, *, count: int = 36
) -> tuple[tuple[S3ActionTemplate, ...], ...]:
    """Create the registered 36-plan pilot schedule.

    Every one of the 18 first-step actions appears exactly twice.  Continuation
    templates are deterministically rotated by state so maps do not all observe
    the same two tails.
    """

    if count != 2 * len(S3_ACTION_TEMPLATES):
        raise ValueError("the registered S3 pilot requires 36 base sequences")
    rotation = int(_fingerprint({"namespace": "v3-s3-sequences", "state": state_key})[:8], 16)
    total = len(S3_ACTION_TEMPLATES)
    sequences = []
    for first_index, first in enumerate(S3_ACTION_TEMPLATES):
        for repeat in range(2):
            second_index = (first_index + rotation + 5 + 7 * repeat) % total
            third_index = (first_index + rotation + 11 + 5 * repeat) % total
            if second_index == first_index:
                second_index = (second_index + 1) % total
            if third_index in {first_index, second_index}:
                third_index = (third_index + 2) % total
            sequences.append(
                (
                    first,
                    S3_ACTION_TEMPLATES[second_index],
                    S3_ACTION_TEMPLATES[third_index],
                )
            )
    return tuple(sequences)


def all_runtime_sequences(
    available_first_templates: Iterable[str],
) -> tuple[tuple[S3ActionTemplate, ...], ...]:
    """Return the exhaustive 18^3 diagnostic sequence space.

    The deployed controller deliberately does not maximize over this space:
    policy_train observes only a sparse subset, so exhaustive maximization
    would combine high inference cost with unsupported-tail extrapolation.
    """

    available = set(map(str, available_first_templates))
    first = [
        template
        for template in S3_ACTION_TEMPLATES
        if template.key in available
    ]
    return tuple(
        (head, second, third)
        for head in first
        for second in S3_ACTION_TEMPLATES
        for third in S3_ACTION_TEMPLATES
    )


def registered_runtime_sequences(
    state_key: str,
    available_first_templates: Iterable[str],
) -> tuple[tuple[S3ActionTemplate, ...], ...]:
    """Return the bounded runtime schedule used by collection and deployment.

    Each available first-step template appears at most twice.  With the
    registered complete 18-template pool this produces exactly 36 sequences.
    """

    available = set(map(str, available_first_templates))
    return tuple(
        templates
        for templates in balanced_sequence_templates(str(state_key))
        if templates[0].key in available
    )


def sequence_id(templates: Iterable[S3ActionTemplate]) -> str:
    values = tuple(template.key for template in templates)
    if len(values) != S3_HORIZON:
        raise ValueError("S3 sequence must contain exactly three templates")
    return f"s3-{_fingerprint(values)[:20]}"


def _feature_dict(row: dict[str, Any]) -> dict[str, float]:
    if "feature_values" in row:
        names = tuple(map(str, row.get("feature_names", ())))
        values = tuple(map(float, row.get("feature_values", ())))
        if len(names) != len(values) or len(names) != len(set(names)):
            raise ValueError("invalid dense S3 source feature row")
        return dict(zip(names, values))
    return {
        str(name): float(value)
        for name, value in dict(row["features"]["realized_dynamic"]).items()
    }


def sequence_feature_row(
    candidate_row: dict[str, Any],
    temporal_context: dict[str, Any],
    templates: tuple[S3ActionTemplate, ...],
    *,
    agent_count: int,
    feature_names: Iterable[str] = V3_S3_FULL_FEATURE_NAMES,
) -> dict[str, Any]:
    if len(templates) != S3_HORIZON:
        raise ValueError("S3 sequence feature row requires three templates")
    agents = int(agent_count)
    if agents <= 0:
        raise ValueError("S3 sequence requires a positive agent count")
    selected_names = tuple(map(str, feature_names))
    source_values = _feature_dict(candidate_row)
    required_base = set(selected_names) & set(
        PROFILE_FEATURE_NAMES["realized_dynamic"]
    )
    missing_base = required_base - set(source_values)
    if missing_base:
        raise ValueError(
            f"S3 source row is missing required base features: {sorted(missing_base)}"
        )
    supported_names = set(V3_S3_LEGACY_FULL_FEATURE_NAMES)
    unknown_names = set(selected_names) - supported_names
    if unknown_names:
        raise ValueError(f"unknown S3 feature projection: {sorted(unknown_names)}")
    required_temporal = set(selected_names) & set(
        S3_LEGACY_TEMPORAL_FEATURE_NAMES
    )
    missing_temporal = required_temporal - set(temporal_context)
    if missing_temporal:
        raise ValueError(
            "S3 temporal context is missing required features: "
            f"{sorted(missing_temporal)}"
        )
    values = {name: 0.0 for name in V3_S3_LEGACY_FULL_FEATURE_NAMES}
    for name, value in source_values.items():
        if name in values:
            values[name] = float(value)
    for name in S3_LEGACY_TEMPORAL_FEATURE_NAMES:
        values[name] = float(temporal_context.get(name, 0.0))
    for step, template in enumerate(templates, 1):
        prefix = f"sequence.step{step}"
        values[f"{prefix}.family={template.family}"] = 1.0
        values[f"{prefix}.size={int(template.requested_size)}"] = 1.0
        values[f"{prefix}.representative={int(template.representative)}"] = 1.0
        values[f"{prefix}.requested_size"] = float(template.requested_size)
        values[f"{prefix}.requested_size_ratio_agents"] = (
            float(template.requested_size) / agents
        )
    return {
        "candidate_id": sequence_id(templates),
        "candidate_key": sequence_id(templates),
        "feature_profile": V3_S3_PROFILE,
        "feature_names": selected_names,
        "feature_values": tuple(values[name] for name in selected_names),
    }


S3_MODEL_NAMES = frozenset(
    (
        *(
            f"step{step}_{target}"
            for step in range(1, S3_HORIZON + 1)
            for target in (
                "conflict_reduction",
                "log_total_seconds",
                "no_progress_probability",
                "template_valid_probability",
            )
        ),
        "sequence_net_conflict_reduction",
        "sequence_log_total_seconds",
        "sequence_no_progress_probability",
    )
)


@dataclass(frozen=True)
class V3S3Bundle:
    models: dict[str, PortableScalarModel]
    thresholds: dict[str, float]
    feature_names: tuple[str, ...]
    prediction_intervals: dict[str, float]
    continuation_calibration: dict[str, Any]
    manifest: dict[str, Any]
    report: dict[str, Any]

    @property
    def inference_backends(self) -> tuple[str, ...]:
        return tuple(sorted({model.inference_backend for model in self.models.values()}))

    @property
    def required_feature_names(self) -> tuple[str, ...]:
        required = {
            name for model in self.models.values() for name in model.feature_names
        }
        return tuple(name for name in self.feature_names if name in required)

    @property
    def wall_time_history_required(self) -> bool:
        return bool(set(self.feature_names) & set(S3_WALL_TIME_FEATURE_NAMES))

    def predict(self, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        for name, model in self.models.items():
            values = list(map(float, model.predict(rows)))
            if name.endswith("_probability"):
                values = [min(1.0, max(0.0, value)) for value in values]
            elif name.startswith("step") and name.endswith(
                "conflict_reduction"
            ):
                values = [max(0.0, value) for value in values]
            result[name] = values
        for step in range(1, S3_HORIZON + 1):
            logs = result[f"step{step}_log_total_seconds"]
            result[f"step{step}_total_seconds"] = [
                max(1e-9, math.expm1(min(50.0, value))) for value in logs
            ]
        result["sequence_total_seconds"] = [
            max(1e-9, math.expm1(min(50.0, value)))
            for value in result["sequence_log_total_seconds"]
        ]
        return result


def load_v3_s3_bundle(path: str | Path) -> V3S3Bundle:
    root = Path(path).resolve()
    manifest = dict(read_json(root / "v3_s3_manifest.json"))
    if str(manifest.get("schema")) != V3_S3_BUNDLE_SCHEMA:
        raise ValueError("unexpected v3-S3 bundle schema")
    forbidden = {
        "main_ranker_semantic_fingerprint",
        "v2_bundle",
        "v2_scores",
        "terminal_fallback",
        "adaptive_fallback",
    }
    if forbidden & set(manifest):
        raise ValueError("v3-S3 bundle contains a forbidden v2/Adaptive dependency")
    feature_schema_id = str(manifest.get("feature_schema_id"))
    schema_contracts = {
        V3_S3_FEATURE_SCHEMA_ID: (
            V3_S3_FEATURE_SCHEMA_SHA256,
            V3_S3_OBJECTIVE_ID,
            set(V3_S3_FULL_FEATURE_NAMES),
        ),
        V3_S3_LEGACY_FEATURE_SCHEMA_ID: (
            V3_S3_LEGACY_FEATURE_SCHEMA_SHA256,
            V3_S3_LEGACY_OBJECTIVE_ID,
            set(V3_S3_LEGACY_FULL_FEATURE_NAMES),
        ),
    }
    if feature_schema_id not in schema_contracts:
        raise ValueError("v3-S3 feature schema id mismatch")
    expected_schema_sha256, expected_objective_id, allowed_names = (
        schema_contracts[feature_schema_id]
    )
    if str(manifest.get("feature_schema_sha256")) != expected_schema_sha256:
        raise ValueError("v3-S3 feature schema SHA256 mismatch")
    if str(manifest.get("training_objective_id")) != expected_objective_id:
        raise ValueError("v3-S3 training objective mismatch")
    declared_names = tuple(map(str, manifest.get("feature_names", ())))
    if not declared_names or len(declared_names) != len(set(declared_names)):
        raise ValueError("v3-S3 feature declaration is invalid")
    if not set(declared_names) <= allowed_names:
        raise ValueError("v3-S3 bundle declares unknown features")
    models = {}
    for name, raw in dict(manifest.get("models", {})).items():
        model_path = root / str(dict(raw)["file"])
        if sha256_file(model_path) != str(dict(raw)["sha256"]):
            raise ValueError(f"v3-S3 model SHA256 mismatch: {name}")
        model = load_portable_scalar_model(
            dict(read_json(model_path)), compact_features=True
        )
        if model.name != str(name) or model.profile != V3_S3_PROFILE:
            raise ValueError(f"v3-S3 model identity mismatch: {name}")
        if not set(model.feature_names) <= set(declared_names):
            raise ValueError(f"v3-S3 model uses undeclared features: {name}")
        models[str(name)] = model
    if set(models) != S3_MODEL_NAMES:
        raise ValueError("v3-S3 bundle has an incomplete model set")
    thresholds = {
        str(name): float(value)
        for name, value in dict(manifest.get("thresholds", {})).items()
    }
    if set(thresholds) != {
        "minimum_template_valid_probability",
        "maximum_no_progress_probability",
        "maximum_sequence_no_progress_probability",
    }:
        raise ValueError("v3-S3 thresholds are incomplete")
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("v3-S3 probability threshold is invalid")
    intervals = {
        str(name): float(value)
        for name, value in dict(manifest.get("prediction_intervals", {})).items()
    }
    if set(intervals) != {"conflict_reduction", "total_seconds", "coverage"}:
        raise ValueError("v3-S3 prediction intervals are incomplete")
    if any(not math.isfinite(value) or value < 0.0 for value in intervals.values()):
        raise ValueError("v3-S3 prediction interval is invalid")
    if intervals["coverage"] > 1.0:
        raise ValueError("v3-S3 prediction interval coverage is invalid")
    continuation = dict(manifest.get("continuation_calibration", {}))
    if str(continuation.get("schema")) not in {
        "lns2.v3_s3_continuation.v1",
        "lns2.v3_s3_continuation.v2",
    }:
        raise ValueError("v3-S3 continuation calibration is missing")
    if not 0.0 < float(continuation.get("coverage", 0.0)) <= 1.0:
        raise ValueError("v3-S3 continuation coverage is invalid")
    if not isinstance(continuation.get("fallback"), dict) or not isinstance(
        continuation.get("cells"), dict
    ):
        raise ValueError("v3-S3 continuation calibration is incomplete")
    report_ref = dict(manifest.get("training_report", {}))
    if set(report_ref) != {"file", "sha256"}:
        raise ValueError("v3-S3 training report reference is incomplete")
    report_path = root / str(report_ref["file"])
    if sha256_file(report_path) != str(report_ref["sha256"]):
        raise ValueError("v3-S3 training report SHA256 mismatch")
    report = dict(read_json(report_path))
    return V3S3Bundle(
        models=models,
        thresholds=thresholds,
        feature_names=declared_names,
        prediction_intervals=intervals,
        continuation_calibration=continuation,
        manifest=manifest,
        report=report,
    )


def rank_s3_sequences(
    sequences: list[tuple[S3ActionTemplate, ...]],
    predictions: dict[str, list[float]],
    thresholds: dict[str, float],
    *,
    allow_risk_relaxation: bool = False,
) -> list[int]:
    if not sequences:
        return []
    count = len(sequences)
    if any(len(values) != count for values in predictions.values()):
        raise ValueError("v3-S3 prediction count does not match sequence count")
    valid_minimum = float(thresholds["minimum_template_valid_probability"])
    no_progress_maximum = float(thresholds["maximum_no_progress_probability"])
    sequence_no_progress_maximum = float(
        thresholds["maximum_sequence_no_progress_probability"]
    )
    scored = []
    for index, templates in enumerate(sequences):
        maximum_risk = 0.0
        minimum_validity = 1.0
        for step in range(1, S3_HORIZON + 1):
            valid = min(
                1.0,
                max(
                    0.0,
                    float(predictions[f"step{step}_template_valid_probability"][index]),
                ),
            )
            no_progress = min(
                1.0,
                max(
                    0.0,
                    float(predictions[f"step{step}_no_progress_probability"][index]),
                ),
            )
            maximum_risk = max(maximum_risk, no_progress)
            minimum_validity = min(minimum_validity, valid)
        expected_reduction = float(
            predictions["sequence_net_conflict_reduction"][index]
        )
        expected_seconds = max(
            1e-9, float(predictions["sequence_total_seconds"][index])
        )
        sequence_no_progress = min(
            1.0,
            max(
                0.0,
                float(predictions["sequence_no_progress_probability"][index]),
            ),
        )
        utility = expected_reduction / expected_seconds
        scored.append(
            {
                "index": index,
                "utility": utility,
                "reduction": expected_reduction,
                "seconds": expected_seconds,
                "risk": maximum_risk,
                "sequence_risk": sequence_no_progress,
                "validity": minimum_validity,
                "safe": minimum_validity + 1e-12 >= valid_minimum
                and maximum_risk <= no_progress_maximum + 1e-12
                and sequence_no_progress
                <= sequence_no_progress_maximum + 1e-12,
                "sequence_id": sequence_id(templates),
            }
        )
    eligible = [row for row in scored if bool(row["safe"])]
    if not eligible:
        if not allow_risk_relaxation:
            return []
        eligible = scored
        eligible.sort(
            key=lambda row: (
                -round(float(row["validity"]), 12),
                round(float(row["sequence_risk"]), 12),
                round(float(row["risk"]), 12),
                -round(float(row["utility"]), 12),
                -round(float(row["reduction"]), 12),
                str(row["sequence_id"]),
            )
        )
    else:
        eligible.sort(
            key=lambda row: (
                -round(float(row["utility"]), 12),
                -round(float(row["reduction"]), 12),
                round(float(row["sequence_risk"]), 12),
                round(float(row["risk"]), 12),
                str(row["sequence_id"]),
            )
        )
    return [int(row["index"]) for row in eligible]


@dataclass
class S3ActivePlan:
    templates: tuple[S3ActionTemplate, ...]
    predictions: dict[str, tuple[float, ...]]
    sequence_id: str
    step_index: int = 0


class V3S3ControllerState:
    """Independent open-loop-three controller with calibrated replanning."""

    def __init__(self, bundle: V3S3Bundle):
        self.bundle = bundle
        self.active_plan: S3ActivePlan | None = None
        self.blacklisted_neighborhoods: set[tuple[int, ...]] = set()
        self.pending_agents: tuple[int, ...] | None = None
        self.pending_prediction: dict[str, float] | None = None
        self.pending_before_fingerprint: str | None = None
        self.pending_agent_count: int | None = None
        self.pending_step: int | None = None
        self.planner_call_count = 0
        self.direct_continuation_count = 0
        self.deviation_replan_count = 0
        self.stalled_no_candidate_count = 0
        self.risk_relaxed_count = 0
        self.cache_hit_count = 0

    @property
    def continuation_template(self) -> S3ActionTemplate | None:
        if self.active_plan is None or self.active_plan.step_index <= 0:
            return None
        if self.active_plan.step_index >= S3_HORIZON:
            return None
        return self.active_plan.templates[self.active_plan.step_index]

    def _candidate_for_template(
        self, candidates: list[dict[str, Any]], template: S3ActionTemplate
    ) -> int | None:
        index = candidate_template_indices(candidates).get(template.key)
        if index is None:
            return None
        agents = tuple(sorted(map(int, candidates[index]["agents"])))
        return None if agents in self.blacklisted_neighborhoods else index

    def candidate_generation_request(self) -> dict[str, Any]:
        """Describe the cheapest candidate pool required by the next decision."""

        template = self.continuation_template
        if template is None:
            return {"mode": "full"}
        return {
            "mode": "restricted",
            "heuristics": [template.family],
            "neighborhood_sizes": [int(template.requested_size)],
            "candidates_per_family": 2,
            "template": template.payload(),
        }

    def note_cache_hit(self) -> None:
        self.cache_hit_count += 1

    def _continuation_cell(self) -> dict[str, float]:
        if self.pending_step is None or self.pending_agent_count is None:
            raise RuntimeError("v3-S3 continuation context is missing")
        calibration = self.bundle.continuation_calibration
        key = f"step{int(self.pending_step)}:agents{int(self.pending_agent_count)}"
        raw = dict(calibration.get("cells", {})).get(key)
        if raw is None:
            raw = dict(calibration.get("fallback", {})).get(
                f"step{int(self.pending_step)}"
            )
        if raw is None:
            raise ValueError(f"v3-S3 continuation calibration has no cell: {key}")
        return {str(name): float(value) for name, value in dict(raw).items()}

    def select(
        self,
        candidates: list[dict[str, Any]],
        candidate_rows: list[dict[str, Any]],
        *,
        temporal_context: dict[str, Any],
        before_fingerprint: str,
        agent_count: int | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        if len(candidates) != len(candidate_rows):
            raise ValueError("v3-S3 candidates and features differ in length")
        resolved_agent_count = int(agent_count) if agent_count is not None else None
        observed_agent_counts = set()
        for row in candidate_rows:
            source = _feature_dict(row)
            if "state.agent_count" in source:
                observed_agent_counts.add(
                    int(round(float(source["state.agent_count"])))
                )
        if len(observed_agent_counts) > 1:
            raise ValueError("v3-S3 candidate rows disagree on agent_count")
        if candidates and resolved_agent_count is None:
            if len(observed_agent_counts) != 1:
                raise ValueError(
                    "v3-S3 selection requires an explicit, consistent agent_count"
                )
            resolved_agent_count = next(iter(observed_agent_counts))
        if (
            resolved_agent_count is not None
            and observed_agent_counts
            and observed_agent_counts != {resolved_agent_count}
        ):
            raise ValueError("v3-S3 explicit agent_count disagrees with candidate rows")
        if resolved_agent_count is not None and resolved_agent_count <= 0:
            raise ValueError("v3-S3 selection requires a positive agent_count")
        if self.active_plan is not None and self.active_plan.step_index > 0:
            template = self.active_plan.templates[self.active_plan.step_index]
            selected = self._candidate_for_template(candidates, template)
            if selected is not None:
                step = self.active_plan.step_index + 1
                self.direct_continuation_count += 1
                prediction = {
                    name: float(values[self.active_plan.step_index])
                    for name, values in self.active_plan.predictions.items()
                }
                self.pending_agents = tuple(sorted(map(int, candidates[selected]["agents"])))
                self.pending_prediction = prediction
                self.pending_before_fingerprint = str(before_fingerprint)
                self.pending_agent_count = int(resolved_agent_count)
                self.pending_step = int(step)
                return selected, {
                    "schema": V3_S3_BUNDLE_SCHEMA,
                    "route": "v3-s3",
                    "selection_kind": "direct-continuation",
                    "sequence_id": self.active_plan.sequence_id,
                    "sequence_step": step,
                    "template": template.payload(),
                    "full_pool_scored": False,
                    "v2_call_count": 0,
                    "adaptive_call_count": 0,
                }
            self.active_plan = None
            self.deviation_replan_count += 1

        template_indices = candidate_template_indices(candidates)
        sequences = list(
            registered_runtime_sequences(
                str(before_fingerprint), template_indices
            )
        )
        rows = []
        active_sequences = []
        for templates in sequences:
            candidate_index = template_indices[templates[0].key]
            agents = tuple(sorted(map(int, candidates[candidate_index]["agents"])))
            if agents in self.blacklisted_neighborhoods:
                continue
            rows.append(
                sequence_feature_row(
                    candidate_rows[candidate_index],
                    temporal_context,
                    templates,
                    agent_count=int(resolved_agent_count),
                    feature_names=getattr(
                        self.bundle,
                        "required_feature_names",
                        self.bundle.feature_names,
                    ),
                )
            )
            active_sequences.append(templates)
        if not active_sequences:
            self.stalled_no_candidate_count += 1
            return None, {
                "schema": V3_S3_BUNDLE_SCHEMA,
                "route": "v3-s3",
                "selection_kind": "v3_stalled_no_candidate",
                "full_pool_scored": True,
                "v2_call_count": 0,
                "adaptive_call_count": 0,
            }
        predictions = self.bundle.predict(rows)
        order = rank_s3_sequences(
            active_sequences, predictions, self.bundle.thresholds
        )
        self.planner_call_count += 1
        risk_relaxed = False
        if not order:
            order = rank_s3_sequences(
                active_sequences,
                predictions,
                self.bundle.thresholds,
                allow_risk_relaxation=True,
            )
            risk_relaxed = bool(order)
            if risk_relaxed:
                self.risk_relaxed_count += 1
        if not order:
            self.stalled_no_candidate_count += 1
            return None, {
                "schema": V3_S3_BUNDLE_SCHEMA,
                "route": "v3-s3",
                "selection_kind": "v3_exhausted",
                "full_pool_scored": True,
                "v2_call_count": 0,
                "adaptive_call_count": 0,
            }
        chosen = order[0]
        templates = active_sequences[chosen]
        plan_predictions = {
            target: tuple(
                float(predictions[f"step{step}_{target}"][chosen])
                if target != "total_seconds"
                else float(predictions[f"step{step}_total_seconds"][chosen])
                for step in range(1, S3_HORIZON + 1)
            )
            for target in (
                "conflict_reduction",
                "total_seconds",
                "log_total_seconds",
                "no_progress_probability",
                "template_valid_probability",
            )
        }
        self.active_plan = S3ActivePlan(
            templates=templates,
            predictions=plan_predictions,
            sequence_id=sequence_id(templates),
        )
        selected = template_indices[templates[0].key]
        self.pending_agents = tuple(sorted(map(int, candidates[selected]["agents"])))
        self.pending_prediction = {
            name: float(values[0]) for name, values in plan_predictions.items()
        }
        self.pending_before_fingerprint = str(before_fingerprint)
        self.pending_agent_count = int(resolved_agent_count)
        self.pending_step = 1
        return selected, {
            "schema": V3_S3_BUNDLE_SCHEMA,
            "route": "v3-s3",
            "selection_kind": "new-plan",
            "sequence_id": self.active_plan.sequence_id,
            "sequence_step": 1,
            "templates": [template.payload() for template in templates],
            "scored_sequence_count": len(active_sequences),
            "full_pool_scored": True,
            "risk_relaxed": risk_relaxed,
            "v2_call_count": 0,
            "adaptive_call_count": 0,
        }

    def observe(
        self,
        *,
        before_fingerprint: str,
        after_fingerprint: str,
        repair_outcome: str,
        conflict_reduction: float,
        total_seconds: float,
        feasible: bool,
    ) -> bool:
        if self.pending_agents is None or self.pending_prediction is None:
            raise RuntimeError("v3-S3 observe called without a pending action")
        if str(before_fingerprint) != str(self.pending_before_fingerprint):
            raise ValueError("v3-S3 pending state fingerprint mismatch")
        no_progress = str(repair_outcome) in {"hard_failure", "accepted_noop"}
        hard_stop = no_progress or str(before_fingerprint) == str(after_fingerprint)
        calibration = self._continuation_cell()
        predicted_no_progress = (
            float(self.pending_prediction["no_progress_probability"])
            >= float(calibration["no_progress_threshold"])
        )
        predicted_reduction = float(
            self.pending_prediction["conflict_reduction"]
        )
        reduction_scale = max(1.0, abs(predicted_reduction))
        reduction_error = (
            abs(float(conflict_reduction) - predicted_reduction)
            / reduction_scale
        )
        predicted_lower_reduction = predicted_reduction - float(
            calibration["reduction_relative_error"]
        ) * reduction_scale
        expected = (
            not hard_stop
            and predicted_no_progress == no_progress
            and reduction_error
            <= float(calibration["reduction_relative_error"]) + 1e-12
            and not (
                float(conflict_reduction) <= 0.0
                and predicted_lower_reduction > 0.0
            )
        )
        if (
            expected
            and str(self.bundle.continuation_calibration.get("schema"))
            == "lns2.v3_s3_continuation.v1"
        ):
            predicted_log_seconds = float(
                self.pending_prediction["log_total_seconds"]
            )
            time_error = abs(
                math.log1p(max(0.0, float(total_seconds)))
                - predicted_log_seconds
            )
            expected = (
                time_error
                <= float(calibration["log_total_seconds_error"]) + 1e-12
            )
        state_changed = str(before_fingerprint) != str(after_fingerprint)
        if state_changed:
            self.blacklisted_neighborhoods.clear()
        elif hard_stop:
            self.blacklisted_neighborhoods.add(self.pending_agents)
        if feasible:
            self.active_plan = None
            self.blacklisted_neighborhoods.clear()
        elif expected and self.active_plan is not None:
            self.active_plan.step_index += 1
            if self.active_plan.step_index >= S3_HORIZON:
                self.active_plan = None
        else:
            if self.active_plan is not None:
                self.deviation_replan_count += 1
            self.active_plan = None
        self.pending_agents = None
        self.pending_prediction = None
        self.pending_before_fingerprint = None
        self.pending_agent_count = None
        self.pending_step = None
        return expected

    def summary(self) -> dict[str, Any]:
        return {
            "schema": V3_S3_BUNDLE_SCHEMA,
            "planner_call_count": self.planner_call_count,
            "direct_continuation_count": self.direct_continuation_count,
            "deviation_replan_count": self.deviation_replan_count,
            "stalled_no_candidate_count": self.stalled_no_candidate_count,
            "risk_relaxed_count": self.risk_relaxed_count,
            "cache_hit_count": self.cache_hit_count,
            "v2_call_count": 0,
            "adaptive_call_count": 0,
            "blacklisted_neighborhood_count": len(self.blacklisted_neighborhoods),
        }


__all__ = [
    "S3_ACTION_TEMPLATES",
    "S3_HORIZON",
    "S3_LEGACY_TEMPORAL_FEATURE_NAMES",
    "S3_TEMPORAL_FEATURE_NAMES",
    "S3_WALL_TIME_FEATURE_NAMES",
    "S3ActionTemplate",
    "V3_S3_BUNDLE_SCHEMA",
    "V3_S3_FEATURE_SCHEMA_ID",
    "V3_S3_FEATURE_SCHEMA_SHA256",
    "V3_S3_FULL_FEATURE_NAMES",
    "V3_S3_LEGACY_FULL_FEATURE_NAMES",
    "V3_S3_OBJECTIVE_ID",
    "V3S3Bundle",
    "V3S3ControllerState",
    "all_runtime_sequences",
    "balanced_sequence_templates",
    "candidate_template_indices",
    "load_v3_s3_bundle",
    "rank_s3_sequences",
    "registered_runtime_sequences",
    "s3_temporal_context",
    "sequence_feature_row",
    "sequence_id",
    "v3_s3_feature_schema_manifest",
]
