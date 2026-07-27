from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments.repair_collection import _read_json, select_seed_agents


CRITICAL_CONFIG_SCHEMA = "lns2.v2_critical_seed_config.v1"
CRITICAL_PROFILES = ("graph", "temporal", "topology")


def _edge(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def conflict_edges(state: dict[str, Any]) -> set[tuple[int, int]]:
    return {
        _edge(int(value[0]), int(value[1]))
        for value in state.get("conflict_edges", [])
        if int(value[0]) != int(value[1])
    }


def update_edge_ages(
    previous: dict[tuple[int, int], int], state: dict[str, Any]
) -> dict[tuple[int, int], int]:
    return {edge: int(previous.get(edge, 0)) + 1 for edge in conflict_edges(state)}


def _conflict_graph(
    state: dict[str, Any],
) -> tuple[dict[int, set[int]], set[tuple[int, int]], set[int]]:
    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    for left, right in conflict_edges(state):
        adjacency[left].add(right)
        adjacency[right].add(left)
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    bridges: set[tuple[int, int]] = set()
    articulation: set[int] = set()
    clock = 0

    def visit(node: int) -> None:
        nonlocal clock
        discovery[node] = low[node] = clock
        clock += 1
        children = 0
        for neighbor in sorted(adjacency[node]):
            if neighbor not in discovery:
                parent[neighbor] = node
                children += 1
                visit(neighbor)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(_edge(node, neighbor))
                if parent.get(node) is None and children > 1:
                    articulation.add(node)
                if parent.get(node) is not None and low[neighbor] >= discovery[node]:
                    articulation.add(node)
            elif neighbor != parent.get(node):
                low[node] = min(low[node], discovery[neighbor])

    for node in sorted(adjacency):
        if node not in discovery:
            parent[node] = None
            visit(node)
    return dict(adjacency), bridges, articulation


def _component_sizes(adjacency: dict[int, set[int]]) -> dict[int, int]:
    result: dict[int, int] = {}
    visited: set[int] = set()
    for root in sorted(adjacency):
        if root in visited:
            continue
        queue = [root]
        component: list[int] = []
        visited.add(root)
        while queue:
            node = queue.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        for node in component:
            result[node] = len(component)
    return result


def _path_bottleneck_ratio(state: dict[str, Any], path: Iterable[int]) -> float:
    rows = int(state.get("rows", 0))
    cols = int(state.get("cols", 0))
    obstacles = list(map(int, state.get("obstacles", [])))
    if rows <= 0 or cols <= 0 or len(obstacles) != rows * cols:
        return 0.0
    values = list(map(int, path))
    if not values:
        return 0.0
    bottlenecks = 0
    for cell in values:
        row, col = divmod(cell, cols)
        degree = 0
        for next_row, next_col in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if (
                0 <= next_row < rows
                and 0 <= next_col < cols
                and not obstacles[next_row * cols + next_col]
            ):
                degree += 1
        bottlenecks += int(degree <= 2)
    return bottlenecks / len(values)


def critical_agent_features(
    state: dict[str, Any],
    *,
    edge_ages: dict[tuple[int, int], int] | None = None,
    include_path_bottleneck: bool = True,
) -> dict[int, dict[str, float]]:
    adjacency, bridges, articulation = _conflict_graph(state)
    components = _component_sizes(adjacency)
    ages = edge_ages or {edge: 1 for edge in conflict_edges(state)}
    agents = {int(row["id"]): row for row in state.get("agents", [])}
    result: dict[int, dict[str, float]] = {}
    for agent_id, neighbors in adjacency.items():
        two_hop = set(neighbors)
        for neighbor in neighbors:
            two_hop.update(adjacency.get(neighbor, ()))
        two_hop.discard(agent_id)
        incident = {_edge(agent_id, neighbor) for neighbor in neighbors}
        agent = agents.get(agent_id, {})
        result[agent_id] = {
            "degree": float(len(neighbors)),
            "bridge_incidence": float(len(incident & bridges)),
            "articulation": float(agent_id in articulation),
            "component_size": float(components.get(agent_id, 1)),
            "two_hop_reach": float(len(two_hop)),
            "persistence": float(sum(ages.get(edge, 1) for edge in incident)),
            "delay": float(agent.get("delay", 0.0)),
            "path_bottleneck_ratio": (
                _path_bottleneck_ratio(state, agent.get("path", []))
                if include_path_bottleneck
                else 0.0
            ),
        }
    return result


def _normalize(
    rows: dict[int, dict[str, float]], name: str
) -> dict[int, float]:
    maximum = max((float(value[name]) for value in rows.values()), default=0.0)
    if maximum <= 0.0:
        return {agent_id: 0.0 for agent_id in rows}
    return {
        agent_id: float(value[name]) / maximum for agent_id, value in rows.items()
    }


def critical_agent_scores(
    state: dict[str, Any],
    *,
    profile: str,
    edge_ages: dict[tuple[int, int], int] | None = None,
) -> dict[int, float]:
    if profile not in CRITICAL_PROFILES:
        raise ValueError(f"unsupported critical-conflict profile: {profile}")
    rows = critical_agent_features(
        state,
        edge_ages=edge_ages,
        include_path_bottleneck=profile == "topology",
    )
    normalized = {
        name: _normalize(rows, name)
        for name in (
            "degree",
            "bridge_incidence",
            "component_size",
            "two_hop_reach",
            "persistence",
            "delay",
            "path_bottleneck_ratio",
        )
    }
    result = {}
    for agent_id, values in rows.items():
        score = (
            2.0 * normalized["degree"][agent_id]
            + 2.0 * normalized["bridge_incidence"][agent_id]
            + 1.5 * values["articulation"]
            + normalized["component_size"][agent_id]
            + normalized["two_hop_reach"][agent_id]
            + 0.5 * normalized["delay"][agent_id]
        )
        if profile in {"temporal", "topology"}:
            score += 1.5 * normalized["persistence"][agent_id]
        if profile == "topology":
            score += normalized["path_bottleneck_ratio"][agent_id]
        result[agent_id] = float(score)
    return result


def select_critical_seed_agents(
    state: dict[str, Any],
    *,
    profile: str,
    margin_threshold: float,
    minimum_seeds: int = 2,
    maximum_seeds: int = 4,
    legacy_maximum_seeds: int = 4,
    state_hash: str | None = None,
    edge_ages: dict[tuple[int, int], int] | None = None,
) -> tuple[list[int], dict[str, Any]]:
    if not 1 <= minimum_seeds <= maximum_seeds:
        raise ValueError("critical seed bounds are invalid")
    if legacy_maximum_seeds < maximum_seeds:
        raise ValueError("legacy seed universe cannot be smaller than the output cap")
    if not math.isfinite(margin_threshold) or margin_threshold < 0.0:
        raise ValueError("critical seed margin must be finite and non-negative")
    # Always rank inside the same legacy-v2 seed universe.  In particular, the
    # two-seed audit must mean "best two of the original four", not "rerun the
    # legacy sampler with a different maximum" (which changes the shuffle).
    legacy = select_seed_agents(
        state, legacy_maximum_seeds, state_hash=state_hash
    )
    scores = critical_agent_scores(state, profile=profile, edge_ages=edge_ages)
    ranked = sorted(legacy, key=lambda value: (-scores.get(value, 0.0), value))
    count = min(minimum_seeds, maximum_seeds, len(ranked))
    maximum_score = max((scores.get(value, 0.0) for value in ranked), default=1.0)
    scale = maximum_score if maximum_score > 0.0 else 1.0
    while count < len(ranked) and count < maximum_seeds:
        previous = scores.get(ranked[count - 1], 0.0)
        following = scores.get(ranked[count], 0.0)
        if (previous - following) / scale > margin_threshold:
            break
        count += 1
    selected = ranked[:count]
    return selected, {
        "profile": profile,
        "margin_threshold": float(margin_threshold),
        "legacy_maximum_seeds": int(legacy_maximum_seeds),
        "legacy_seed_agents": legacy,
        "ranked_seed_agents": ranked,
        "selected_seed_agents": selected,
        "seed_scores": {str(value): scores.get(value, 0.0) for value in ranked},
        "fallback_to_full": len(selected) == len(legacy),
    }


@dataclass(frozen=True)
class CriticalSeedConfig:
    profile: str
    margin_threshold: float
    minimum_seeds: int
    maximum_seeds: int
    diagnostic_only: bool
    source: dict[str, Any]
    raw: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        return dict(self.raw)


def load_critical_seed_config(
    value: str | Path | dict[str, Any],
    *,
    allow_unpromoted_diagnostic: bool = False,
) -> CriticalSeedConfig:
    payload = _read_json(Path(value)) if isinstance(value, (str, Path)) else dict(value)
    if str(payload.get("schema")) != CRITICAL_CONFIG_SCHEMA:
        raise ValueError("invalid v2-critical seed config schema")
    diagnostic_only = bool(payload.get("diagnostic_only", False))
    if not bool(payload.get("deployment_promoted")) and not (
        allow_unpromoted_diagnostic and diagnostic_only
    ):
        raise ValueError("v2-critical seed config did not pass its offline audit")
    profile = str(payload.get("profile"))
    if profile not in CRITICAL_PROFILES:
        raise ValueError("v2-critical config has an invalid profile")
    minimum = int(payload.get("minimum_seeds", 0))
    maximum = int(payload.get("maximum_seeds", 0))
    if not 1 <= minimum <= maximum:
        raise ValueError("v2-critical config has invalid seed bounds")
    margin = float(payload["margin_threshold"])
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("v2-critical config has an invalid margin threshold")
    return CriticalSeedConfig(
        profile=profile,
        margin_threshold=margin,
        minimum_seeds=minimum,
        maximum_seeds=maximum,
        diagnostic_only=diagnostic_only,
        source=dict(payload.get("source") or {}),
        raw=payload,
    )
