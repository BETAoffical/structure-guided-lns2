from __future__ import annotations

import json
import time
from typing import Any, Iterable

from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.hybridstructpool import HybridStructPoolResult
from lns2_selector.runtime.structshell_dual16 import (
    STRUCTSHELL_DUAL16_POOL_ID,
    STRUCTSHELL_DUAL16_RUNTIME_ID,
    _merge_structshell_dual16_candidates,
    structshell_dual16_ablation_gate,
    structshell_dual16_augmentation,
)
from lns2_selector.runtime.topology_candidates import (
    generate_structpool_candidate_subset,
)


STRUCTSHELL_COMPONENT16_POOL_ID = "stride-structshell-component16-v1"
STRUCTSHELL_COMPONENT16_RUNTIME_ID = "stride-structshell-component16-runtime-v1"

_COMPONENT16_FAMILY = "structpool-conflict-component:16"
_SOURCE_FAMILY_SIZES = {
    "conflict_component": [16],
    "spatiotemporal_hotspot": [16],
}
_COMPONENT16_ACTIVATION_GATE = {
    "gate_id": "component16_ablation_all_states",
}


def structshell_component16_augmentation() -> dict[str, Any]:
    """Return the immutable V2 + Component16 runtime contract.

    Both Dual16 source cells are generated so Component16 stays bit-for-bit on
    the existing optimized source path.  Only the row carrying Component16
    provenance is admitted to the frozen V2 ranking pool.
    """

    result = {
        "enabled": True,
        "pool_id": STRUCTSHELL_COMPONENT16_POOL_ID,
        "runtime_id": STRUCTSHELL_COMPONENT16_RUNTIME_ID,
        "source_pool_id": STRUCTSHELL_DUAL16_POOL_ID,
        "source_runtime_id": STRUCTSHELL_DUAL16_RUNTIME_ID,
        "full_union_required": False,
        "full_union_audit_preserved": True,
        "runtime_filter_id": "component_family_fixed16_v1",
        "source_mode": "structshell_component16",
        "nominal_size": 16,
        "runtime_structural_family_sizes": _SOURCE_FAMILY_SIZES,
        "retained_selection_family": _COMPONENT16_FAMILY,
        "maximum_generated_candidates": 2,
        "maximum_added_candidates": 1,
        "maximum_total_candidates": 64,
        "static_grid_cache": True,
        "activation_gate": _COMPONENT16_ACTIVATION_GATE,
    }
    return json.loads(json.dumps(result))


def validate_structshell_component16_augmentation(
    value: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    result = dict(value)
    if result != structshell_component16_augmentation():
        raise ValueError("unsupported Component16 StructShell runtime augmentation")
    return result


def structshell_component16_ablation_gate(
    state: dict[str, Any], value: dict[str, Any]
) -> dict[str, Any]:
    """Enable Component16 on every valid, non-terminal conflicted state."""

    config = validate_structshell_component16_augmentation(value)
    assert config is not None
    started = time.perf_counter()
    result = structshell_dual16_ablation_gate(
        state,
        structshell_dual16_augmentation(),
    )
    gate_id = str(config["activation_gate"]["gate_id"])
    result["seconds"] = time.perf_counter() - started
    result["gate_id"] = gate_id
    if result["passed"]:
        result["reason"] = gate_id
    return result


def generate_structshell_component16_runtime_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_candidates: Iterable[dict[str, Any]],
    config: dict[str, Any],
) -> HybridStructPoolResult:
    """Merge the complete V2 pool with at most one Component16 candidate."""

    specification = validate_structshell_component16_augmentation(config)
    assert specification is not None
    base = list(v2_candidates)
    if not base:
        raise ValueError("Component16 StructShell requires the full V2 pool")

    dual16_sizes = dict(
        structshell_dual16_augmentation()["runtime_structural_family_sizes"]
    )
    if dual16_sizes != dict(specification["runtime_structural_family_sizes"]):
        raise RuntimeError("Component16 source contract differs from active Dual16")
    family_sizes = {
        str(family): tuple(map(int, sizes))
        for family, sizes in dual16_sizes.items()
    }

    started = time.perf_counter()
    generated = generate_structpool_candidate_subset(
        state,
        analysis,
        family_sizes=family_sizes,
    )
    maximum_generated = int(specification["maximum_generated_candidates"])
    if len(generated) > maximum_generated:
        raise RuntimeError(
            "Component16 StructShell source generated more than two candidates"
        )
    retained_family = str(specification["retained_selection_family"])
    structural = [
        row
        for row in generated
        if retained_family in set(map(str, row.get("selection_families") or ()))
    ]
    if len(structural) > int(specification["maximum_added_candidates"]):
        raise RuntimeError(
            "Component16 StructShell generated multiple Component16 candidates"
        )
    structural_seconds = time.perf_counter() - started

    result = _merge_structshell_dual16_candidates(base, structural)
    if len(result.challengers) > int(specification["maximum_added_candidates"]):
        raise RuntimeError(
            "Component16 StructShell retained more than one challenger"
        )
    result.structural_generation_seconds = structural_seconds
    result.causal_generation_seconds = 0.0
    return result


__all__ = [
    "STRUCTSHELL_COMPONENT16_POOL_ID",
    "STRUCTSHELL_COMPONENT16_RUNTIME_ID",
    "generate_structshell_component16_runtime_candidates",
    "structshell_component16_ablation_gate",
    "structshell_component16_augmentation",
    "validate_structshell_component16_augmentation",
]
