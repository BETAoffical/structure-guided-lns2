from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from experiments._common import sha256_file
from experiments.feature_schema_v2 import PROFILE_FEATURE_NAMES


MODEL_SCHEMA = "lns2.stride.slotpool_pairwise_hist_gbdt.v1"
IMPLEMENTATION_ID = "stride-slotpool-v1"
PROFILE = "realized_dynamic"
FAMILY_VARIANTS = (
    "bottleneck_crossing",
    "conflict_component",
    "topology_boundary_articulation",
    "topology_boundary_low_degree",
    "spatiotemporal_hotspot",
    "path_overlap",
)
ALLOWED_SIZES = (8, 16, 24, 32)
BASE_FEATURE_NAMES = tuple(PROFILE_FEATURE_NAMES[PROFILE])
STATE_FEATURE_NAMES = tuple(
    name for name in BASE_FEATURE_NAMES if name.startswith("state.")
)
DERIVED_FEATURE_NAMES = (
    *(f"slot.family={name}" for name in FAMILY_VARIANTS),
    *(f"slot.size={size}" for size in ALLOWED_SIZES),
    *(
        f"slot.family_size={family}:{size}"
        for family in FAMILY_VARIANTS
        for size in ALLOWED_SIZES
    ),
    "slot.provenance_count",
    "slot.mixed_family",
    "slot.v2_anchor_jaccard",
    "slot.support_count_min",
    "slot.support_count_mean",
    "slot.support_count_max",
    "slot.support_ratio_min",
    "slot.support_ratio_mean",
    "slot.support_ratio_max",
    "slot.size_support_ratio_min",
    "slot.size_support_ratio_mean",
    "slot.size_support_ratio_max",
)
CANDIDATE_FEATURE_NAMES = BASE_FEATURE_NAMES + DERIVED_FEATURE_NAMES
PAIR_FEATURE_NAMES = tuple(
    f"delta:{name}" for name in CANDIDATE_FEATURE_NAMES
) + tuple(f"shared:{name}" for name in STATE_FEATURE_NAMES)
FAMILY_PREFIXES = {
    "structpool-bottleneck-crossing": "bottleneck_crossing",
    "structpool-conflict-component": "conflict_component",
    "structpool-boundary-articulation": "topology_boundary_articulation",
    "structpool-boundary-low_degree": "topology_boundary_low_degree",
    "structpool-spatiotemporal-hotspot": "spatiotemporal_hotspot",
    "structpool-path-overlap": "path_overlap",
}
SEMANTIC_MODEL_KEYS = (
    "schema",
    "implementation_id",
    "candidate_feature_names",
    "pair_feature_names",
    "candidate_feature_dimension",
    "pair_feature_dimension",
    "selected_parameter_index",
    "parameters",
    "baseline",
    "trees",
)


def load_slotpool_model(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"SlotPool model is missing: {path}")
    observed = sha256_file(path)
    if expected_sha256 is not None and observed != str(expected_sha256).lower():
        raise ValueError(
            f"SlotPool model SHA-256 changed: expected {expected_sha256}, got {observed}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != MODEL_SCHEMA
        or payload.get("implementation_id") != IMPLEMENTATION_ID
        or tuple(payload.get("candidate_feature_names") or ())
        != CANDIDATE_FEATURE_NAMES
        or tuple(payload.get("pair_feature_names") or ()) != PAIR_FEATURE_NAMES
        or int(payload.get("candidate_feature_dimension", -1))
        != len(CANDIDATE_FEATURE_NAMES)
        or int(payload.get("pair_feature_dimension", -1)) != len(PAIR_FEATURE_NAMES)
    ):
        raise ValueError("SlotPool portable model schema changed")
    semantic_sha = str(payload.get("semantic_sha256", ""))
    semantic_payload = {key: payload[key] for key in SEMANTIC_MODEL_KEYS}
    encoded = json.dumps(
        semantic_payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != semantic_sha:
        raise ValueError("SlotPool portable model semantic hash changed")
    return payload


def _family_variant(raw_family: str) -> tuple[str, int]:
    prefix, separator, raw_size = raw_family.rpartition(":")
    if not separator or prefix not in FAMILY_PREFIXES:
        raise ValueError(f"unsupported SlotPool family: {raw_family}")
    size = int(raw_size)
    if size not in ALLOWED_SIZES:
        raise ValueError(f"unsupported SlotPool size: {raw_family}")
    return FAMILY_PREFIXES[prefix], size


def _candidate_jaccard(left: list[int], right: list[int]) -> float:
    first = set(map(int, left))
    second = set(map(int, right))
    return len(first & second) / max(1, len(first | second))


def _row_features(row: dict[str, Any]) -> dict[str, float]:
    if "feature_values" in row:
        if str(row.get("feature_profile")) != PROFILE:
            raise ValueError("dense SlotPool feature row has the wrong profile")
        names = tuple(map(str, row.get("feature_names") or ()))
        values = tuple(map(float, row.get("feature_values") or ()))
        if len(names) != len(values) or len(names) != len(set(names)):
            raise ValueError("dense SlotPool feature row is invalid")
        features = dict(zip(names, values))
    else:
        features = dict(dict(row.get("features") or {}).get(PROFILE) or {})
    if set(features) != set(BASE_FEATURE_NAMES):
        raise ValueError("SlotPool runtime requires the exact 124-feature schema")
    if any(not math.isfinite(float(value)) for value in features.values()):
        raise ValueError("SlotPool runtime feature is non-finite")
    return features


def _derived_values(
    candidate: dict[str, Any], *, anchor_agents: list[int]
) -> list[float]:
    families: set[str] = set()
    sizes: set[int] = set()
    family_sizes: set[tuple[str, int]] = set()
    supports: list[float] = []
    support_ratios: list[float] = []
    size_support_ratios: list[float] = []
    support_by_family = dict(candidate["structpool_support_count_by_family"])
    ratio_by_family = dict(candidate["structpool_support_ratio_by_family"])
    provenance = list(map(str, candidate["selection_families"]))
    for raw_family in provenance:
        family, size = _family_variant(raw_family)
        families.add(family)
        sizes.add(size)
        family_sizes.add((family, size))
        support = float(support_by_family[raw_family])
        ratio = float(ratio_by_family[raw_family])
        supports.append(support)
        support_ratios.append(ratio)
        size_support_ratios.append(float(size) / max(1.0, support))
    if not families:
        raise ValueError("SlotPool structural candidate has no provenance")
    return [
        *(float(name in families) for name in FAMILY_VARIANTS),
        *(float(size in sizes) for size in ALLOWED_SIZES),
        *(
            float((family, size) in family_sizes)
            for family in FAMILY_VARIANTS
            for size in ALLOWED_SIZES
        ),
        float(len(provenance)),
        float(len(families) > 1),
        _candidate_jaccard(list(candidate["agents"]), anchor_agents),
        min(supports),
        statistics.fmean(supports),
        max(supports),
        min(support_ratios),
        statistics.fmean(support_ratios),
        max(support_ratios),
        min(size_support_ratios),
        statistics.fmean(size_support_ratios),
        max(size_support_ratios),
    ]


def _candidate_matrix(
    candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    anchor_agents: list[int],
) -> Any:
    import numpy as np

    values = []
    for index in indices:
        features = _row_features(rows[index])
        values.append(
            [float(features[name]) for name in BASE_FEATURE_NAMES]
            + _derived_values(candidates[index], anchor_agents=anchor_agents)
        )
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(indices), len(CANDIDATE_FEATURE_NAMES)):
        raise RuntimeError("SlotPool runtime candidate feature dimension changed")
    return result


def _pair_matrix(candidate_values: Any, pairs: list[tuple[int, int]]) -> Any:
    import numpy as np

    shared = [BASE_FEATURE_NAMES.index(name) for name in STATE_FEATURE_NAMES]
    values = np.empty((len(pairs), len(PAIR_FEATURE_NAMES)), dtype=np.float32)
    for pair_index, (left, right) in enumerate(pairs):
        values[pair_index, : len(CANDIDATE_FEATURE_NAMES)] = (
            candidate_values[left] - candidate_values[right]
        )
        values[pair_index, len(CANDIDATE_FEATURE_NAMES) :] = 0.5 * (
            candidate_values[left, shared] + candidate_values[right, shared]
        )
    return values


def _predict(payload: dict[str, Any], matrix: Any) -> Any:
    import numpy as np

    values = np.asarray(matrix, dtype=np.float32)
    raw = np.full(values.shape[0], float(payload["baseline"]), dtype=np.float64)
    for tree in payload["trees"]:
        for row_index, row in enumerate(values):
            node_index = 0
            while not bool(tree[node_index]["is_leaf"]):
                node = tree[node_index]
                value = float(row[int(node["feature_idx"])])
                go_left = (
                    bool(node["missing_go_to_left"])
                    if math.isnan(value)
                    else value <= float(node["num_threshold"])
                )
                node_index = int(node["left"] if go_left else node["right"])
            raw[row_index] += float(tree[node_index]["value"])
    result = np.empty_like(raw)
    positive = raw >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-raw[positive]))
    exponential = np.exp(raw[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def reduce_slotpool_candidates(
    *,
    candidates: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    model_payload: dict[str, Any],
    v2_anchor_index: int,
    maximum_challengers: int = 6,
) -> dict[str, Any]:
    if len(candidates) != len(candidate_rows) or not candidates:
        raise ValueError("SlotPool runtime candidate rows do not align")
    if not 0 <= v2_anchor_index < len(candidates):
        raise ValueError("SlotPool V2 anchor index is out of range")
    structural = [
        index
        for index, candidate in enumerate(candidates)
        if bool(candidate.get("structpool_family_groups"))
    ]
    base = [index for index in range(len(candidates)) if index not in set(structural)]
    if v2_anchor_index not in base:
        raise ValueError("SlotPool V2 anchor must come from the base pool")
    if maximum_challengers != 6:
        raise ValueError("SlotPool runtime requires a six-challenger budget")
    if not structural:
        return {
            "retained_indices": base,
            "base_indices": base,
            "structural_indices": [],
            "selected_structural_indices": [],
            "selected_candidate_ids": [],
            "selected_scores": [],
            "v2_anchor_candidate_id": str(candidates[v2_anchor_index]["candidate_id"]),
            "raw_structural_candidate_count": 0,
        }
    values = _candidate_matrix(
        candidates,
        candidate_rows,
        structural,
        anchor_agents=list(candidates[v2_anchor_index]["agents"]),
    )
    pairs = list(itertools.combinations(range(len(structural)), 2))
    if pairs:
        probabilities = _predict(model_payload, _pair_matrix(values, pairs))
        totals = [0.0] * len(structural)
        counts = [0] * len(structural)
        for (left, right), probability in zip(pairs, probabilities):
            totals[left] += float(probability)
            totals[right] += 1.0 - float(probability)
            counts[left] += 1
            counts[right] += 1
        scores = [totals[index] / counts[index] for index in range(len(structural))]
    else:
        scores = [1.0]
    order = sorted(
        range(len(structural)),
        key=lambda index: (
            -float(scores[index]),
            str(candidates[structural[index]]["candidate_id"]),
        ),
    )
    selected_local = order[:maximum_challengers]
    selected = [structural[index] for index in selected_local]
    selected_set = set(selected)
    retained = [
        index
        for index in range(len(candidates))
        if index in set(base) or index in selected_set
    ]
    return {
        "retained_indices": retained,
        "base_indices": base,
        "structural_indices": structural,
        "selected_structural_indices": selected,
        "selected_candidate_ids": [
            str(candidates[index]["candidate_id"]) for index in selected
        ],
        "selected_scores": [float(scores[index]) for index in selected_local],
        "v2_anchor_candidate_id": str(candidates[v2_anchor_index]["candidate_id"]),
        "raw_structural_candidate_count": len(structural),
    }


__all__ = [
    "load_slotpool_model",
    "reduce_slotpool_candidates",
]
