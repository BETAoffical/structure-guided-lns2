from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable

from experiments.neighborhood_candidates import candidate_id
from experiments.state_analysis import StateAnalysis
from lns2_selector.runtime.causalclosurepool import generate_causalclosure_candidates
from lns2_selector.runtime.topology_candidates import generate_structpool_candidate_grid


HYBRIDSTRUCTPOOL_ID = "stride-hybridstructpool-v1"
STRUCTURAL_SIZES = (8, 16, 24, 32)


@dataclass
class HybridStructPoolResult:
    """Outcome-blind union of V2, the complete StructShell, and CausalClosure."""

    candidates: list[dict[str, Any]]
    challengers: list[dict[str, Any]]
    provenance_by_candidate_id: dict[str, tuple[str, ...]]
    base_candidate_count: int
    structural_candidate_count: int
    causal_candidate_count: int
    exact_duplicate_count: int
    causal_attempts: list[dict[str, Any]]


def _normalized_candidate(row: dict[str, Any]) -> tuple[tuple[int, ...], str]:
    agents = tuple(sorted(set(map(int, row.get("agents") or ()))))
    if not agents:
        raise ValueError("HybridStructPool candidates must contain agents")
    identity = str(row.get("candidate_id") or "")
    expected = candidate_id(agents)
    if identity != expected:
        raise ValueError("HybridStructPool candidate identity does not match agents")
    return agents, identity


def merge_hybridstructpool_candidates(
    base_candidates: Iterable[dict[str, Any]],
    structural_candidates: Iterable[dict[str, Any]],
    causal_candidates: Iterable[dict[str, Any]],
) -> HybridStructPoolResult:
    """Merge exact agent sets while retaining source provenance outside rows.

    Input rows are copied without adding outcome or selector fields.  V2 rows
    win exact-set ties so the frozen base representation remains authoritative.
    """

    sources = (
        ("v2_base", [dict(row) for row in base_candidates]),
        ("structshell_equal_four_size", [dict(row) for row in structural_candidates]),
        ("causalclosure_v2", [dict(row) for row in causal_candidates]),
    )
    merged: dict[tuple[int, ...], dict[str, Any]] = {}
    provenance: dict[tuple[int, ...], set[str]] = {}
    source_counts: dict[str, int] = {}
    duplicate_count = 0
    for source, rows in sources:
        source_counts[source] = len(rows)
        seen_in_source: set[tuple[int, ...]] = set()
        for row in rows:
            agents, _identity = _normalized_candidate(row)
            if agents in seen_in_source:
                raise ValueError(f"HybridStructPool source contains duplicate sets: {source}")
            seen_in_source.add(agents)
            provenance.setdefault(agents, set()).add(source)
            if agents in merged:
                duplicate_count += 1
                continue
            merged[agents] = copy.deepcopy(row)

    ordered = sorted(merged.values(), key=lambda row: str(row["candidate_id"]))
    provenance_by_id = {
        str(merged[agents]["candidate_id"]): tuple(sorted(values))
        for agents, values in sorted(
            provenance.items(), key=lambda item: str(merged[item[0]]["candidate_id"])
        )
    }
    challengers = [
        copy.deepcopy(row)
        for row in ordered
        if "v2_base" not in provenance_by_id[str(row["candidate_id"])]
    ]
    return HybridStructPoolResult(
        candidates=copy.deepcopy(ordered),
        challengers=challengers,
        provenance_by_candidate_id=provenance_by_id,
        base_candidate_count=source_counts["v2_base"],
        structural_candidate_count=source_counts["structshell_equal_four_size"],
        causal_candidate_count=source_counts["causalclosure_v2"],
        exact_duplicate_count=duplicate_count,
        causal_attempts=[],
    )


def generate_hybridstructpool_candidates(
    state: dict[str, Any],
    analysis: StateAnalysis,
    *,
    v2_candidates: Iterable[dict[str, Any]],
    v2_anchors: Iterable[dict[str, Any]],
    structural_sizes: Iterable[int] = STRUCTURAL_SIZES,
    maximum_causal_candidates: int = 12,
    maximum_causal_neighborhood_size: int = 64,
    causal_temporal_window: int = 2,
    maximum_causal_jaccard: float = 0.9,
) -> HybridStructPoolResult:
    """Generate the full candidate contract without a runtime budget reducer.

    The function intentionally exposes every exact-deduplicated four-size
    structural action.  It does not rank, score with outcomes, or compress the
    union to a runtime budget.
    """

    sizes = tuple(sorted(set(map(int, structural_sizes))))
    if sizes != STRUCTURAL_SIZES:
        raise ValueError("HybridStructPool requires symmetric sizes 8, 16, 24, and 32")
    base = [dict(row) for row in v2_candidates]
    anchors = [dict(row) for row in v2_anchors]
    if not base or not anchors:
        raise ValueError("HybridStructPool requires the full V2 pool and a V2 anchor")
    structural = generate_structpool_candidate_grid(
        state, analysis, neighborhood_sizes=sizes
    )
    causal = generate_causalclosure_candidates(
        state,
        analysis,
        v2_anchors=anchors,
        maximum_candidates=maximum_causal_candidates,
        maximum_neighborhood_size=maximum_causal_neighborhood_size,
        temporal_window=causal_temporal_window,
        maximum_jaccard_similarity=maximum_causal_jaccard,
    )
    result = merge_hybridstructpool_candidates(base, structural, causal.candidates)
    result.causal_attempts = copy.deepcopy(causal.attempts)
    return result


__all__ = [
    "HYBRIDSTRUCTPOOL_ID",
    "STRUCTURAL_SIZES",
    "HybridStructPoolResult",
    "generate_hybridstructpool_candidates",
    "merge_hybridstructpool_candidates",
]
