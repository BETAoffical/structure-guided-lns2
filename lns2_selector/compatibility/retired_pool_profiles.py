"""Read-only profiles for historical SlotPool and GuardPool artifacts.

These constructors preserve the exact JSON-compatible augmentation payloads
written by the retired experiments.  They are intentionally isolated from the
active runtime: ``validate_structpool_augmentation`` rejects both payloads and
the closed-loop executor must never use them for a new episode.
"""

from __future__ import annotations

import json
from typing import Any


_RETIRED_SLOTPOOL_AUGMENTATION = {
    "enabled": True,
    "pool_id": "stride-slotpool-v1",
    "runtime_id": "stride-slotpool-runtime-v1",
    "neighborhood_sizes": [8, 16, 24, 32],
    "maximum_added_candidates": 24,
    "maximum_slotpool_candidates": 6,
    "maximum_total_candidates": 48,
    "static_grid_cache": True,
    "slotpool_model": {
        "path": "build/stride-slotpool-v1-offline-evaluation/slotpool_pairwise_model.json",
        "sha256": "e0a42c7a21cb341a6c5ce13ad6d89e6f5279c04378535fb9c44ce76a1dfe2319",
    },
    "activation_gate": {
        "gate_id": "stride-highstress-state-v1",
        "minimum_conflict_pair_count": 16,
        "any_of": {
            "minimum_agent_count": 96,
            "minimum_active_conflict_agent_count": 32,
            "minimum_largest_conflict_component_size": 16,
        },
    },
}
_RETIRED_GUARDPOOL_AUGMENTATION = {
    **_RETIRED_SLOTPOOL_AUGMENTATION,
    "runtime_id": "stride-guardpool-v1",
    "stall_guard": {
        "no_progress_limit": 8,
        "recovery_controller": "v2-full",
        "release_condition": "strict_conflict_decrease",
        "wall_time_condition": None,
    },
}


def slotpool_runtime_augmentation() -> dict[str, Any]:
    """Return a detached copy of the retired SlotPool artifact profile."""

    return json.loads(json.dumps(_RETIRED_SLOTPOOL_AUGMENTATION))


def guardpool_runtime_augmentation() -> dict[str, Any]:
    """Return a detached copy of the retired GuardPool artifact profile."""

    return json.loads(json.dumps(_RETIRED_GUARDPOOL_AUGMENTATION))


__all__ = [
    "guardpool_runtime_augmentation",
    "slotpool_runtime_augmentation",
]
