from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Iterable

from experiments._common import ratio as _ratio
from experiments.repair_collection import _fingerprint, _read_jsonl


def _candidate_id(agents: Iterable[int]) -> str:
    return f"neighborhood-{_fingerprint(sorted(map(int, agents)))[:16]}"


def _jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return _ratio(len(left_set & right_set), len(union)) if union else 1.0


def _distance(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    return 1.0 - _jaccard(left, right)


def select_representative_neighborhoods(
    proposals: list[dict[str, Any]], candidates_per_family: int
) -> list[dict[str, Any]]:
    if candidates_per_family <= 0:
        raise ValueError("candidates_per_family must be positive")
    by_agents: dict[tuple[int, ...], dict[str, Any]] = {}
    by_family: dict[str, collections.Counter[tuple[int, ...]]] = (
        collections.defaultdict(collections.Counter)
    )
    for proposal in proposals:
        agents = tuple(sorted(int(value) for value in proposal["agents"]))
        if not agents or len(agents) != len(set(agents)):
            raise ValueError("proposal agents must be a non-empty unique set")
        family = str(proposal["family"])
        by_family[family][agents] += 1
        record = by_agents.setdefault(
            agents,
            {
                "candidate_id": _candidate_id(agents),
                "agents": list(agents),
                "proposal_count_by_family": collections.Counter(),
                "proposal_seeds": set(),
                "seed_agents": set(),
                "selection_families": set(),
            },
        )
        record["proposal_count_by_family"][family] += 1
        record["proposal_seeds"].add(int(proposal["proposal_seed"]))
        record["seed_agents"].add(int(proposal["seed_agent"]))

    for family, counts in sorted(by_family.items()):
        remaining = set(counts)
        chosen: list[tuple[int, ...]] = []
        while remaining and len(chosen) < candidates_per_family:
            if not chosen:
                selected = min(
                    remaining,
                    key=lambda agents: (-counts[agents], _candidate_id(agents)),
                )
            else:
                selected = min(
                    remaining,
                    key=lambda agents: (
                        -min(_distance(agents, previous) for previous in chosen),
                        -counts[agents],
                        _candidate_id(agents),
                    ),
                )
            chosen.append(selected)
            remaining.remove(selected)
            by_agents[selected]["selection_families"].add(family)

    selected_rows = []
    for agents, record in by_agents.items():
        if not record["selection_families"]:
            continue
        selected_rows.append(
            {
                "candidate_id": record["candidate_id"],
                "agents": list(agents),
                "actual_size": len(agents),
                "selection_families": sorted(record["selection_families"]),
                "proposal_count_by_family": dict(
                    sorted(record["proposal_count_by_family"].items())
                ),
                "proposal_seeds": sorted(record["proposal_seeds"]),
                "seed_agents": sorted(record["seed_agents"]),
            }
        )
    return sorted(selected_rows, key=lambda row: str(row["candidate_id"]))


def conflict_density(conflicts: int, agent_count: int) -> float:
    if agent_count < 2:
        return 0.0
    return 2.0 * float(conflicts) / float(agent_count * (agent_count - 1))


def conflict_severity(density: float, thresholds: dict[str, Any]) -> str:
    low = float(thresholds["low_max"])
    medium = float(thresholds["medium_max"])
    if not 0.0 <= low < medium:
        raise ValueError("conflict severity thresholds must satisfy 0 <= low < medium")
    if density <= low:
        return "low"
    if density <= medium:
        return "medium"
    return "high"


def no_pruning_metrics(
    candidate_count: int, reason: str = "disabled"
) -> dict[str, Any]:
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


def _reference_rows(reference: Path) -> list[dict[str, Any]]:
    if not reference.is_dir():
        raise ValueError(f"missing reference dataset: {reference}")
    rows: list[dict[str, Any]] = []
    for manifest in sorted(reference.glob("*/manifest.jsonl")):
        rows.extend(_read_jsonl(manifest))
    if not rows:
        raise ValueError(f"reference dataset has no manifests: {reference}")
    return rows


def _seed_isolation(
    rows: list[dict[str, Any]], references: list[str], project_root: Path
) -> dict[str, Any]:
    map_seeds = {int(row["map_seed"]) for row in rows}
    task_seeds = {int(row["task_seed"]) for row in rows}
    overlap: dict[str, Any] = {}
    for value in references:
        path = Path(value)
        if not path.is_absolute():
            path = project_root / path
        reference = _reference_rows(path.resolve())
        map_overlap = sorted(map_seeds & {int(row["map_seed"]) for row in reference})
        task_overlap = sorted(task_seeds & {int(row["task_seed"]) for row in reference})
        if map_overlap or task_overlap:
            overlap[str(value)] = {
                "map_seeds": map_overlap,
                "task_seeds": task_overlap,
            }
    return {
        "passed": not overlap,
        "map_seed_count": len(map_seeds),
        "task_seed_count": len(task_seeds),
        "overlap": overlap,
        "references": list(references),
    }
