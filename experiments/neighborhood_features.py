from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable

from experiments._common import (
    add_categorical_feature as _categorical,
    mean as _mean,
    population_std as _std,
    ratio as _ratio,
)
from experiments.state_analysis import StateAnalysis


FORBIDDEN_FEATURE_FRAGMENTS = (
    "outcome.",
    "label.",
    "post_repair",
    "conflicts_after",
    "conflict_reduction",
    "solved_rate",
    "branch_runtime",
    "step_runtime",
)


def _aggregate(prefix: str, values: Iterable[float | int]) -> dict[str, float]:
    numbers = [float(value) for value in values]
    return {
        f"{prefix}_mean": _mean(numbers),
        f"{prefix}_std": _std(numbers),
        f"{prefix}_min": min(numbers, default=0.0),
        f"{prefix}_max": max(numbers, default=0.0),
        f"{prefix}_sum": sum(numbers),
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing JSONL file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _path_wait_ratio(path: list[int]) -> float:
    return _ratio(
        sum(left == right for left, right in zip(path, path[1:])),
        max(0, len(path) - 1),
    )


def _path_values(path: Iterable[int], values: dict[int, float | int]) -> list[float]:
    return [float(values.get(cell, 0.0)) for cell in path]


def state_dynamic_features(
    state: dict[str, Any], analysis: StateAnalysis
) -> dict[str, float]:
    agents = list(state["agents"])
    agent_count = len(agents)
    conflict_degrees = [float(agent.get("conflict_degree", 0)) for agent in agents]
    delays = [float(agent.get("delay", 0)) for agent in agents]
    path_costs = [float(agent.get("path_cost", 0)) for agent in agents]
    shortest = [float(agent.get("shortest_path_cost", 0)) for agent in agents]
    paths = [[int(cell) for cell in agent["path"]] for agent in agents]
    active_agents = {value for edge in analysis.pair_set for value in edge}
    component_sizes = [len(value) for value in analysis.component_members.values()]
    event_times = [event.time for event in analysis.events]
    low_level = dict(state.get("low_level", {}))
    features = {
        "state.agent_count": float(agent_count),
        "state.iteration": float(state.get("iteration", 0)),
        "state.colliding_pairs": float(state.get("num_of_colliding_pairs", 0)),
        "state.conflict_edge_density": _ratio(len(analysis.pair_set), agent_count),
        "state.conflict_event_count": float(len(analysis.events)),
        "state.vertex_event_ratio": _ratio(
            sum(event.kind == "vertex" for event in analysis.events),
            len(analysis.events),
        ),
        "state.conflicting_agent_ratio": _ratio(len(active_agents), agent_count),
        "state.component_count": float(len(component_sizes)),
        "state.largest_component": max(component_sizes, default=0.0),
        "state.largest_component_ratio": _ratio(
            max(component_sizes, default=0.0), agent_count
        ),
        "state.degree_mean": _mean(conflict_degrees),
        "state.degree_std": _std(conflict_degrees),
        "state.degree_max": max(conflict_degrees, default=0.0),
        "state.delay_mean": _mean(delays),
        "state.delay_std": _std(delays),
        "state.delay_max": max(delays, default=0.0),
        "state.path_cost_mean": _mean(path_costs),
        "state.path_cost_std": _std(path_costs),
        "state.path_stretch_mean": _ratio(sum(path_costs), max(1.0, sum(shortest))),
        "state.path_wait_ratio_mean": _mean(_path_wait_ratio(path) for path in paths),
        "state.conflict_time_mean": _mean(event_times),
        "state.conflict_time_std": _std(event_times),
        "state.sum_of_costs_per_agent": _ratio(
            state.get("sum_of_costs", 0), agent_count
        ),
        "state.low_level_generated_per_agent": _ratio(
            low_level.get("generated", 0), agent_count
        ),
        "state.low_level_runs_per_agent": _ratio(low_level.get("runs", 0), agent_count),
    }
    return features


@dataclass
class CandidateFeatureCache:
    by_id: dict[int, dict[str, Any]]
    active_agents: set[int]
    paths: dict[int, list[int]]
    path_sets: dict[int, set[int]]


def candidate_feature_cache(
    state: dict[str, Any], analysis: StateAnalysis
) -> CandidateFeatureCache:
    by_id = {int(agent["id"]): agent for agent in state["agents"]}
    if len(by_id) != len(state["agents"]):
        raise ValueError("state contains duplicate agent ids")
    paths = {
        agent_id: [int(cell) for cell in agent["path"]]
        for agent_id, agent in by_id.items()
    }
    return CandidateFeatureCache(
        by_id=by_id,
        active_agents={value for edge in analysis.pair_set for value in edge},
        paths=paths,
        path_sets={agent_id: set(path) for agent_id, path in paths.items()},
    )


def proposal_features(
    state: dict[str, Any],
    analysis: StateAnalysis,
    candidate: dict[str, Any],
    *,
    feature_cache: CandidateFeatureCache | None = None,
) -> dict[str, float]:
    agents = list(state["agents"])
    by_id = (
        candidate_feature_cache(state, analysis).by_id
        if feature_cache is None
        else feature_cache.by_id
    )
    selected = [int(value) for value in candidate["agents"]]
    if len(selected) != len(set(selected)) or not selected:
        raise ValueError("candidate neighborhood must be non-empty and unique")
    if any(agent_id not in by_id for agent_id in selected):
        raise ValueError("candidate neighborhood references an unknown agent")
    seed_ids = sorted({int(value) for value in candidate.get("seed_agents", [])})
    if any(agent_id not in by_id for agent_id in seed_ids):
        raise ValueError("candidate provenance references an unknown seed agent")
    counts = {
        str(name).lower(): int(value)
        for name, value in dict(candidate.get("proposal_count_by_family", {})).items()
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("proposal family counts must be non-negative")
    total_count = sum(counts.values())
    selection_families = sorted(
        {str(value).lower() for value in candidate.get("selection_families", [])}
    )
    seed_agents = [by_id[agent_id] for agent_id in seed_ids]
    seed_component_sizes = [
        len(analysis.component_members.get(analysis.component_id.get(agent_id), set()))
        for agent_id in seed_ids
    ]
    actual_size = len(selected)
    features = {
        "proposal.actual_size": float(actual_size),
        "proposal.actual_size_ratio_agents": _ratio(actual_size, len(agents)),
        "proposal.total_count": float(total_count),
        "proposal.unique_proposal_seed_count": float(
            len(set(map(int, candidate.get("proposal_seeds", []))))
        ),
        "proposal.seed_agent_count": float(len(seed_ids)),
        "proposal.selection_family_count": float(len(selection_families)),
        "proposal.support_family_count": float(sum(value > 0 for value in counts.values())),
    }
    _categorical(features, "proposal.actual_size", actual_size)
    for family in selection_families:
        features[f"proposal.selection_family={family}"] = 1.0
    for family, count in sorted(counts.items()):
        features[f"proposal.family_count={family}"] = float(count)
        features[f"proposal.family_ratio={family}"] = _ratio(count, total_count)
    features.update(
        _aggregate(
            "proposal.seed_conflict_degree",
            [agent.get("conflict_degree", 0) for agent in seed_agents],
        )
    )
    features.update(
        _aggregate(
            "proposal.seed_delay", [agent.get("delay", 0) for agent in seed_agents]
        )
    )
    features.update(
        _aggregate(
            "proposal.seed_path_cost",
            [agent.get("path_cost", 0) for agent in seed_agents],
        )
    )
    features.update(_aggregate("proposal.seed_component_size", seed_component_sizes))
    return features


def explicit_neighborhood_features(
    state: dict[str, Any],
    analysis: StateAnalysis,
    neighborhood: list[int],
    *,
    feature_cache: CandidateFeatureCache | None = None,
) -> dict[str, float]:
    if not neighborhood or len(neighborhood) != len(set(neighborhood)):
        raise ValueError("explicit neighborhood must be non-empty and unique")
    feature_cache = (
        candidate_feature_cache(state, analysis)
        if feature_cache is None
        else feature_cache
    )
    by_id = feature_cache.by_id
    if any(agent_id not in by_id for agent_id in neighborhood):
        raise ValueError("explicit neighborhood contains an unknown agent")
    selected = set(neighborhood)
    internal_edges = [
        edge for edge in analysis.pair_set if edge[0] in selected and edge[1] in selected
    ]
    boundary_edges = [
        edge for edge in analysis.pair_set if (edge[0] in selected) != (edge[1] in selected)
    ]
    active_agents = feature_cache.active_agents
    component_ids = {
        analysis.component_id[agent_id]
        for agent_id in selected
        if agent_id in analysis.component_id
    }
    component_coverages = [
        _ratio(
            len(selected & analysis.component_members[component_id]),
            len(analysis.component_members[component_id]),
        )
        for component_id in sorted(component_ids)
    ]
    selected_agents = [by_id[agent_id] for agent_id in sorted(selected)]
    delays = [float(agent.get("delay", 0)) for agent in selected_agents]
    conflicts = [float(agent.get("conflict_degree", 0)) for agent in selected_agents]
    path_costs = [float(agent.get("path_cost", 0)) for agent in selected_agents]
    stretches = [
        _ratio(agent.get("path_cost", 0), max(1, agent.get("shortest_path_cost", 0)))
        for agent in selected_agents
    ]
    selected_ids = sorted(selected)
    paths = [feature_cache.paths[agent_id] for agent_id in selected_ids]
    path_sets = [feature_cache.path_sets[agent_id] for agent_id in selected_ids]
    overlaps = [
        _ratio(len(left & right), len(left | right))
        for left, right in itertools.combinations(path_sets, 2)
    ]
    union = set().union(*path_sets) if path_sets else set()
    if union:
        coordinates = [divmod(cell, analysis.cols) for cell in union]
        span_rows = max(row for row, _ in coordinates) - min(row for row, _ in coordinates) + 1
        span_cols = max(col for _, col in coordinates) - min(col for _, col in coordinates) + 1
    else:
        span_rows = span_cols = 0
    flattened = [cell for path in paths for cell in path]
    degrees = _path_values(flattened, analysis.degrees)
    obstacle_2 = _path_values(flattened, analysis.obstacle_rate_2)
    obstacle_4 = _path_values(flattened, analysis.obstacle_rate_4)
    internal_events = sum(
        event.left in selected and event.right in selected for event in analysis.events
    )
    incident_events = sum(
        event.left in selected or event.right in selected for event in analysis.events
    )
    features = {
        "realized.actual_size": float(len(selected)),
        "realized.actual_size_ratio_agents": _ratio(len(selected), len(by_id)),
        "realized.conflicting_agent_ratio": _ratio(len(selected & active_agents), len(selected)),
        "realized.conflicting_agent_coverage": _ratio(len(selected & active_agents), len(active_agents)),
        "realized.component_count": float(len(component_ids)),
        "realized.component_coverage_mean": _mean(component_coverages),
        "realized.component_coverage_max": max(component_coverages, default=0.0),
        "realized.internal_conflict_edges": float(len(internal_edges)),
        "realized.boundary_conflict_edges": float(len(boundary_edges)),
        "realized.incident_conflict_coverage": _ratio(
            len(internal_edges) + len(boundary_edges), len(analysis.pair_set)
        ),
        "realized.internal_conflict_coverage": _ratio(
            len(internal_edges), len(analysis.pair_set)
        ),
        "realized.internal_event_coverage": _ratio(internal_events, len(analysis.events)),
        "realized.incident_event_coverage": _ratio(incident_events, len(analysis.events)),
        "realized.path_overlap_mean": _mean(overlaps),
        "realized.path_overlap_max": max(overlaps, default=0.0),
        "realized.path_union_cell_ratio": _ratio(len(union), len(analysis.free_cells)),
        "realized.path_bbox_area_ratio": _ratio(
            span_rows * span_cols, analysis.rows * analysis.cols
        ),
        "realized.path_degree_mean": _mean(degrees),
        "realized.path_low_degree_ratio": _ratio(
            sum(value <= 2 for value in degrees), len(degrees)
        ),
        "realized.path_articulation_ratio": _ratio(
            sum(cell in analysis.articulation for cell in flattened), len(flattened)
        ),
        "realized.path_visit_heat_mean": _mean(
            _path_values(flattened, analysis.visit_heat)
        ),
        "realized.path_obstacle_rate_r2": _mean(obstacle_2),
        "realized.path_obstacle_rate_r4": _mean(obstacle_4),
        "realized.path_wait_ratio_mean": _mean(_path_wait_ratio(path) for path in paths),
    }
    features.update(_aggregate("realized.delay", delays))
    features.update(_aggregate("realized.conflict_degree", conflicts))
    features.update(_aggregate("realized.path_cost", path_costs))
    features.update(_aggregate("realized.path_stretch", stretches))
    return features


def static_context_features(state: dict[str, Any]) -> dict[str, float]:
    context = dict(state.get("context", {}))
    topology = dict(context.get("topology_metrics", {}))
    rows = int(state.get("rows", 0))
    cols = int(state.get("cols", 0))
    obstacles = [int(value) for value in state.get("obstacles", [])]
    free_cells = len(obstacles) - sum(obstacles)
    agent_count = len(state.get("agents", []))
    features = {
        "context.agent_count": float(agent_count),
        "context.rows": float(rows),
        "context.cols": float(cols),
        "context.free_cells": float(free_cells),
        "context.free_cell_ratio": _ratio(free_cells, rows * cols),
        "context.agent_density": _ratio(agent_count, free_cells),
        "context.mean_shortest_distance": float(context.get("mean_shortest_distance", 0)),
        "context.dominant_flow_ratio": float(context.get("dominant_flow_ratio", 0)),
        "context.hotspot_skew": float(context.get("hotspot_skew", 0)),
        "context.required_bottleneck_crossing_ratio": float(
            context.get("required_bottleneck_crossing_ratio", 0)
        ),
        "context.required_intersection_crossing_ratio": float(
            context.get("required_intersection_crossing_ratio", 0)
        ),
        "context.shared_corridor_ratio": float(context.get("shared_corridor_ratio", 0)),
        "context.swap_pair_ratio": float(context.get("swap_pair_ratio", 0)),
        "context.articulation_count": float(topology.get("articulation_count", 0)),
        "context.average_free_degree": float(topology.get("average_free_degree", 0)),
        "context.dead_end_cell_count": float(topology.get("dead_end_cell_count", 0)),
        "context.route_redundancy_proxy": float(
            topology.get("route_redundancy_proxy", 0)
        ),
    }
    for name in ("layout_mode", "layout_variant", "scenario_type", "task_variant"):
        _categorical(features, f"context.{name}", context.get(name))
    return features


def _feature_profiles(
    state: dict[str, Any], analysis: StateAnalysis, candidate: dict[str, Any]
) -> dict[str, dict[str, float]]:
    return _feature_profiles_from_shared(state, analysis, candidate)


def _feature_profiles_from_shared(
    state: dict[str, Any],
    analysis: StateAnalysis,
    candidate: dict[str, Any],
    *,
    dynamic: dict[str, float] | None = None,
    context: dict[str, float] | None = None,
    feature_cache: CandidateFeatureCache | None = None,
) -> dict[str, dict[str, float]]:
    dynamic = state_dynamic_features(state, analysis) if dynamic is None else dynamic
    proposal = proposal_features(
        state, analysis, candidate, feature_cache=feature_cache
    )
    realized = explicit_neighborhood_features(
        state,
        analysis,
        [int(value) for value in candidate["agents"]],
        feature_cache=feature_cache,
    )
    context = static_context_features(state) if context is None else context
    profiles = {
        "proposal_dynamic": dynamic | proposal,
        "realized_dynamic": dynamic | proposal | realized,
        "realized_context": dynamic | proposal | realized | context,
    }
    for profile, features in profiles.items():
        leaking = sorted(
            name
            for name in features
            if any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
        )
        if leaking:
            raise ValueError(f"feature leakage in {profile}: {leaking}")
        if any(not math.isfinite(float(value)) for value in features.values()):
            raise ValueError(f"non-finite feature value in {profile}")
    return profiles
