from __future__ import annotations

import hashlib
import json
from typing import Any


REPAIR_STRUCTURE_KEYS = (
    "initialized",
    "initial_solution_complete",
    "feasible",
    "rows",
    "cols",
    "sum_of_costs",
    "num_of_colliding_pairs",
    "obstacles",
    "conflict_edges",
    "agents",
)


def semantic_fingerprint(value: Any) -> str:
    """Return the repository's canonical JSON SHA-256 fingerprint."""

    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def repair_structure_fingerprint(state: dict[str, Any]) -> str:
    """Hash repair-relevant state while excluding attempts and counters."""

    missing = [key for key in REPAIR_STRUCTURE_KEYS if key not in state]
    if missing:
        raise ValueError(f"repair state is missing structural fields: {missing}")
    return semantic_fingerprint({key: state[key] for key in REPAIR_STRUCTURE_KEYS})
