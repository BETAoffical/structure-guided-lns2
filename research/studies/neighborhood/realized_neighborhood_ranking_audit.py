from __future__ import annotations

import collections
import hashlib
import itertools
import json
import math
import pickle
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    add_categorical_feature as _categorical,
    feature_names as _feature_names,
    mean as _mean,
    population_std as _std,
    ratio as _ratio,
    resolve_within as _resolve,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)
from research.studies.context.context_audit import MODEL_SEED, PairwiseModel, _pair_vector
from research.studies.representation.local_representation_audit import StateAnalysis, analyze_state


SCHEMA_VERSION = 1
INDEX_SCHEMA = "lns2.realized_neighborhood_ranking_index.v1"
FEATURE_PROFILES = (
    "proposal_dynamic",
    "realized_dynamic",
    "realized_context",
)
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


def effectiveness_values(outcome: dict[str, Any]) -> tuple[float, float]:
    return (-float(outcome["solved_rate"]), float(outcome["conflicts_after"]))


def effectiveness_dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_values = effectiveness_values(left)
    right_values = effectiveness_values(right)
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _sensitivity_dominates(
    left: dict[str, Any], right: dict[str, Any], metric: str
) -> bool:
    left_values = effectiveness_values(left) + (float(left[metric]),)
    right_values = effectiveness_values(right) + (float(right[metric]),)
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def _aggregate_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "trial_count": len(rows),
        "solved_rate": _mean(bool(row["solved"]) for row in rows),
        "conflicts_after": _mean(row["conflicts_after"] for row in rows),
        "conflict_auc": _mean(row["conflict_auc"] for row in rows),
        "generated": _mean(row["generated"] for row in rows),
        "runtime": _mean(row["runtime"] for row in rows),
    }


def _label_rows(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    for candidates in grouped.values():
        for candidate in candidates:
            outcome = candidate["outcome"]
            candidate["labels"] = {
                "effectiveness_pareto": not any(
                    other is not candidate
                    and effectiveness_dominates(other["outcome"], outcome)
                    for other in candidates
                ),
                "compute_aware_pareto": not any(
                    other is not candidate
                    and _sensitivity_dominates(
                        other["outcome"], outcome, "generated"
                    )
                    for other in candidates
                ),
                "runtime_sensitive_pareto": not any(
                    other is not candidate
                    and _sensitivity_dominates(other["outcome"], outcome, "runtime")
                    for other in candidates
                ),
            }


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


def build_ranking_index(
    collection: str | Path,
    *,
    expected_states: int | None = 23,
    expected_candidates: int | None = 412,
    expected_outcomes: int | None = 3296,
    expected_trials: int = 8,
    expected_maps: int | None = 6,
    expected_split: str = "probe",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = Path(collection).resolve()
    run_config = _read_json(root / "run_config.json")
    candidate_rows = _read_jsonl(root / "candidates.jsonl")
    manifests = _read_jsonl(root / "collection_manifest.jsonl")
    if int(run_config.get("configuration", {}).get("evaluation_trials", -1)) != expected_trials:
        raise ValueError("collection evaluation trial count does not match the audit")
    if len({str(row["state_id"]) for row in candidate_rows}) != len(candidate_rows):
        raise ValueError("collection contains duplicate state ids")
    forbidden_splits = sorted(
        {
            str(row.get("split", "unknown"))
            for row in candidate_rows
            if str(row.get("split", "unknown")) != expected_split
        }
    )
    if forbidden_splits:
        raise ValueError(
            f"ranking audit contains Test/OOD or non-{expected_split} splits: {forbidden_splits}"
        )
    manifest_by_state: dict[str, dict[str, Any]] = {}
    outcomes_by_key: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    total_outcomes = 0
    for manifest in manifests:
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            raise ValueError("collection manifest contains an unsuccessful state")
        state_id = str(manifest["state_id"])
        if state_id in manifest_by_state:
            raise ValueError(f"duplicate manifest state: {state_id}")
        manifest_by_state[state_id] = manifest
        error_rows = _read_jsonl(_resolve(root, str(manifest["errors_file"])))
        if error_rows or int(manifest.get("error_count", 0)):
            raise ValueError(f"collection state contains explicit-evaluation errors: {state_id}")
        outcome_rows = _read_jsonl(_resolve(root, str(manifest["outcomes_file"])))
        total_outcomes += len(outcome_rows)
        for outcome in outcome_rows:
            outcomes_by_key[(str(outcome["state_id"]), str(outcome["candidate_id"]))].append(
                outcome
            )
    state_ids = {str(row["state_id"]) for row in candidate_rows}
    if set(manifest_by_state) != state_ids:
        raise ValueError("candidate states and collection manifest states do not match")

    index: list[dict[str, Any]] = []
    candidate_keys: set[tuple[str, str]] = set()
    for source in candidate_rows:
        state_id = str(source["state_id"])
        state = source["state"]
        analysis = analyze_state(state)
        expected_pairs = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in state.get("conflict_edges", [])
        }
        if analysis.pair_set != expected_pairs:
            raise ValueError(f"reconstructed conflicts disagree for state {state_id}")
        if int(state.get("num_of_colliding_pairs", len(expected_pairs))) != len(expected_pairs):
            raise ValueError(f"recorded conflict count disagrees for state {state_id}")
        candidates = list(source["candidates"])
        if int(source.get("candidate_count", len(candidates))) != len(candidates):
            raise ValueError(f"candidate count disagrees for state {state_id}")
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            key = (state_id, candidate_id)
            if key in candidate_keys:
                raise ValueError(f"duplicate explicit candidate: {key}")
            candidate_keys.add(key)
            trial_rows = outcomes_by_key.get(key, [])
            if len(trial_rows) != expected_trials:
                raise ValueError(
                    f"candidate {candidate_id} has {len(trial_rows)} trials, expected {expected_trials}"
                )
            trial_indices = sorted(int(row["evaluation_trial_index"]) for row in trial_rows)
            if trial_indices != list(range(expected_trials)):
                raise ValueError(f"candidate {candidate_id} has invalid trial indices")
            evaluation_seeds = [int(row["evaluation_seed"]) for row in trial_rows]
            if len(evaluation_seeds) != len(set(evaluation_seeds)):
                raise ValueError(f"candidate {candidate_id} repeats an evaluation seed")
            agents = sorted(map(int, candidate["agents"]))
            for outcome in trial_rows:
                if not bool(outcome.get("action_valid")):
                    raise ValueError(f"candidate {candidate_id} contains an invalid action")
                if not bool(outcome.get("evaluation_seed_disjoint")):
                    raise ValueError(f"candidate {candidate_id} reuses a proposal seed")
                actual = sorted(map(int, outcome.get("actual_neighborhood", [])))
                if actual != agents or sorted(map(int, outcome.get("agents", []))) != agents:
                    raise ValueError(f"candidate {candidate_id} changed its explicit neighborhood")
                if int(outcome.get("conflicts_before", -1)) != len(expected_pairs):
                    raise ValueError(f"candidate {candidate_id} has the wrong source conflict count")
            profiles = _feature_profiles(state, analysis, candidate)
            context = dict(state.get("context", {}))
            index.append(
                {
                    "schema": INDEX_SCHEMA,
                    "schema_version": SCHEMA_VERSION,
                    "state_id": state_id,
                    "candidate_id": candidate_id,
                    "candidate_key": candidate_id,
                    "map_id": str(source["map_id"]),
                    "task_id": str(source["task_id"]),
                    "split": str(source["split"]),
                    "layout_mode": str(source.get("layout_mode", context.get("layout_mode", "unknown"))),
                    "task_variant": str(source.get("task_variant", context.get("task_variant", "unknown"))),
                    "agent_count": int(source.get("agent_count", len(state["agents"]))),
                    "actual_size": len(agents),
                    "agents": agents,
                    "selection_families": sorted(
                        map(str, candidate.get("selection_families", []))
                    ),
                    "seed_agents": sorted(map(int, candidate.get("seed_agents", []))),
                    "trial_count": len(trial_rows),
                    "features": profiles,
                    "outcome": _aggregate_outcomes(trial_rows),
                    "neighborhood_sha256": hashlib.sha256(
                        ",".join(map(str, agents)).encode("ascii")
                    ).hexdigest(),
                }
            )
    orphan_outcomes = sorted(set(outcomes_by_key) - candidate_keys)
    if orphan_outcomes:
        raise ValueError(f"collection contains {len(orphan_outcomes)} orphan outcomes")
    _label_rows(index)
    index.sort(key=lambda row: (str(row["state_id"]), str(row["candidate_id"])))
    map_count = len({str(row["map_id"]) for row in index})
    checks = {
        "state_count": len(state_ids),
        "map_count": map_count,
        "candidate_count": len(index),
        "outcome_count": total_outcomes,
        "trials_per_candidate": expected_trials,
        "forbidden_split_rows": 0,
        "orphan_outcomes": 0,
    }
    expected = {
        "state_count": expected_states,
        "map_count": expected_maps,
        "candidate_count": expected_candidates,
        "outcome_count": expected_outcomes,
    }
    mismatches = {
        name: {"expected": value, "actual": checks[name]}
        for name, value in expected.items()
        if value is not None and checks[name] != value
    }
    if mismatches:
        raise ValueError(f"ranking collection size mismatch: {mismatches}")
    checks["passed"] = True
    checks["source_run_fingerprint"] = str(run_config.get("run_fingerprint", ""))
    return index, checks


def _grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: str(row["candidate_id"]))
    return grouped


def leave_one_map_out_folds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = _grouped(rows)
    state_map = {state_id: str(candidates[0]["map_id"]) for state_id, candidates in grouped.items()}
    maps = sorted(set(state_map.values()))
    if len(maps) < 2:
        raise ValueError("leave-one-map-out evaluation requires at least two maps")
    folds = []
    all_states = set(grouped)
    for fold, validation_map in enumerate(maps):
        validation_states = {
            state_id for state_id, map_id in state_map.items() if map_id == validation_map
        }
        train_states = all_states - validation_states
        train_maps = sorted(set(maps) - {validation_map})
        folds.append(
            {
                "fold": fold,
                "train_maps": train_maps,
                "validation_maps": [validation_map],
                "train_states": train_states,
                "validation_states": validation_states,
            }
        )
    for fold in folds:
        if set(fold["train_maps"]) & set(fold["validation_maps"]):
            raise ValueError("map leakage in leave-one-map-out folds")
        if fold["train_states"] & fold["validation_states"]:
            raise ValueError("state leakage in leave-one-map-out folds")
    return folds


DominancePair = tuple[dict[str, Any], dict[str, Any], int]


def dominance_pairs(rows: list[dict[str, Any]]) -> list[DominancePair]:
    result: list[DominancePair] = []
    for candidates in _grouped(rows).values():
        for left, right in itertools.combinations(candidates, 2):
            if effectiveness_dominates(left["outcome"], right["outcome"]):
                result.append((left, right, 1))
            elif effectiveness_dominates(right["outcome"], left["outcome"]):
                result.append((left, right, 0))
    if not result:
        raise ValueError("no effectiveness dominance pairs are available")
    return result


def train_pairwise_model(
    rows: list[dict[str, Any]],
    profile: str,
    model_parameters: dict[str, Any] | None = None,
) -> tuple[PairwiseModel, int]:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    pairs = dominance_pairs(rows)
    names = _feature_names(rows, profile)
    examples: list[list[float]] = []
    labels: list[int] = []
    for left, right, label in pairs:
        examples.append(_pair_vector(left, right, profile, names))
        labels.append(label)
        examples.append(_pair_vector(right, left, profile, names))
        labels.append(1 - label)
    parameters = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": MODEL_SEED,
    }
    parameters.update(model_parameters or {})
    parameters["random_state"] = MODEL_SEED
    estimator = HistGradientBoostingClassifier(**parameters)
    estimator.fit(np.asarray(examples, dtype=float), np.asarray(labels, dtype=int))
    return PairwiseModel(profile, names, estimator), len(pairs)


def batch_model_selections(
    grouped_candidates: dict[str, list[dict[str, Any]]], model: PairwiseModel
) -> dict[str, int]:
    import numpy as np

    forward_vectors: list[list[float]] = []
    reverse_vectors: list[list[float]] = []
    references: list[tuple[str, int, int]] = []
    for state_id, candidates in sorted(grouped_candidates.items()):
        for left in range(len(candidates)):
            for right in range(left + 1, len(candidates)):
                forward_vectors.append(
                    _pair_vector(
                        candidates[left],
                        candidates[right],
                        model.profile,
                        model.feature_names,
                    )
                )
                reverse_vectors.append(
                    _pair_vector(
                        candidates[right],
                        candidates[left],
                        model.profile,
                        model.feature_names,
                    )
                )
                references.append((state_id, left, right))
    scores = {
        state_id: [0.0] * len(candidates)
        for state_id, candidates in grouped_candidates.items()
    }
    if forward_vectors:
        forward = model.estimator.predict_proba(
            np.asarray(forward_vectors, dtype=float)
        )[:, 1]
        reverse = model.estimator.predict_proba(
            np.asarray(reverse_vectors, dtype=float)
        )[:, 1]
        probabilities = (forward + (1.0 - reverse)) / 2.0
        for probability, (state_id, left, right) in zip(probabilities, references):
            scores[state_id][left] += float(probability)
            scores[state_id][right] += 1.0 - float(probability)
    return {
        state_id: min(
            range(len(candidates)),
            key=lambda index: (
                -scores[state_id][index], str(candidates[index]["candidate_key"])
            ),
        )
        for state_id, candidates in grouped_candidates.items()
    }


def pairwise_accuracy(rows: list[dict[str, Any]], model: PairwiseModel) -> float:
    import numpy as np

    pairs = dominance_pairs(rows)
    forward_vectors = [
        _pair_vector(left, right, model.profile, model.feature_names)
        for left, right, _ in pairs
    ]
    reverse_vectors = [
        _pair_vector(right, left, model.profile, model.feature_names)
        for left, right, _ in pairs
    ]
    forward = model.estimator.predict_proba(
        np.asarray(forward_vectors, dtype=float)
    )[:, 1]
    reverse = model.estimator.predict_proba(
        np.asarray(reverse_vectors, dtype=float)
    )[:, 1]
    probabilities = (forward + (1.0 - reverse)) / 2.0
    correct = sum(
        int(probability >= 0.5) == label
        for probability, (_, _, label) in zip(probabilities, pairs)
    )
    return _ratio(correct, len(pairs))


def _selection_record(
    selected: dict[str, Any], candidates: list[dict[str, Any]], selector: str
) -> dict[str, Any]:
    best_conflicts = min(float(row["outcome"]["conflicts_after"]) for row in candidates)
    best_solved = max(float(row["outcome"]["solved_rate"]) for row in candidates)
    minimum_auc = min(float(row["outcome"]["conflict_auc"]) for row in candidates)
    minimum_generated = min(float(row["outcome"]["generated"]) for row in candidates)
    outcome = selected["outcome"]
    return {
        "state_id": str(selected["state_id"]),
        "map_id": str(selected["map_id"]),
        "task_id": str(selected["task_id"]),
        "selector": selector,
        "candidate_id": str(selected["candidate_id"]),
        "pareto_hit": float(bool(selected["labels"]["effectiveness_pareto"])),
        "compute_aware_pareto_hit": float(
            bool(selected["labels"]["compute_aware_pareto"])
        ),
        "runtime_sensitive_pareto_hit": float(
            bool(selected["labels"]["runtime_sensitive_pareto"])
        ),
        "conflict_regret": _ratio(
            float(outcome["conflicts_after"]) - best_conflicts,
            max(1.0, abs(best_conflicts)),
        ),
        "solved_rate_regret": best_solved - float(outcome["solved_rate"]),
        "auc_regret": _ratio(
            float(outcome["conflict_auc"]) - minimum_auc,
            max(1.0, abs(minimum_auc)),
        ),
        "generated_regret": _ratio(
            float(outcome["generated"]) - minimum_generated,
            max(1.0, abs(minimum_generated)),
        ),
        "selected_size": int(selected["actual_size"]),
        "selection_families": list(selected["selection_families"]),
        "selected_conflicts": float(outcome["conflicts_after"]),
        "selected_solved_rate": float(outcome["solved_rate"]),
        "selected_generated": float(outcome["generated"]),
        "selected_runtime": float(outcome["runtime"]),
    }


def evaluate_model(
    rows: list[dict[str, Any]], model: PairwiseModel, selector: str
) -> dict[str, dict[str, Any]]:
    records = {}
    grouped = _grouped(rows)
    selections = batch_model_selections(grouped, model)
    for state_id, candidates in grouped.items():
        selected = candidates[selections[state_id]]
        records[state_id] = _selection_record(selected, candidates, selector)
    return records


def _fractional_distribution(
    records: Iterable[dict[str, Any]], name: str
) -> dict[str, float]:
    counter: collections.Counter[str] = collections.Counter()
    count = 0
    for record in records:
        values = list(record[name])
        if not values:
            values = ["unknown"]
        for value in values:
            counter[str(value)] += 1.0 / len(values)
        count += 1
    return {key: _ratio(value, count) for key, value in sorted(counter.items())}


def summarize_records(
    records: dict[str, dict[str, Any]], pair_accuracy: float | None = None
) -> dict[str, Any]:
    values = list(records.values())
    sizes = collections.Counter(int(row["selected_size"]) for row in values)
    return {
        "state_count": len(values),
        "pareto_top1_hit_rate": _mean(row["pareto_hit"] for row in values),
        "compute_aware_pareto_top1_hit_rate": _mean(
            row["compute_aware_pareto_hit"] for row in values
        ),
        "runtime_sensitive_pareto_top1_hit_rate": _mean(
            row["runtime_sensitive_pareto_hit"] for row in values
        ),
        "mean_conflict_regret": _mean(row["conflict_regret"] for row in values),
        "mean_solved_rate_regret": _mean(row["solved_rate_regret"] for row in values),
        "mean_auc_regret": _mean(row["auc_regret"] for row in values),
        "mean_generated_regret": _mean(row["generated_regret"] for row in values),
        "mean_selected_runtime": _mean(row["selected_runtime"] for row in values),
        "pairwise_accuracy": pair_accuracy,
        "selected_size_counts": {str(key): value for key, value in sorted(sizes.items())},
        "maximum_size_share": _ratio(max(sizes.values(), default=0), len(values)),
        "selection_family_fraction": _fractional_distribution(values, "selection_families"),
    }


def uniform_random_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records = {}
    for state_id, candidates in _grouped(rows).items():
        best_conflicts = min(float(row["outcome"]["conflicts_after"]) for row in candidates)
        best_solved = max(float(row["outcome"]["solved_rate"]) for row in candidates)
        minimum_auc = min(float(row["outcome"]["conflict_auc"]) for row in candidates)
        minimum_generated = min(float(row["outcome"]["generated"]) for row in candidates)
        size_weights: collections.Counter[int] = collections.Counter(
            int(row["actual_size"]) for row in candidates
        )
        family_weights: collections.Counter[str] = collections.Counter()
        for candidate in candidates:
            families = list(candidate["selection_families"]) or ["unknown"]
            for family in families:
                family_weights[str(family)] += 1.0 / len(families)
        records[state_id] = {
            "state_id": state_id,
            "map_id": str(candidates[0]["map_id"]),
            "task_id": str(candidates[0]["task_id"]),
            "selector": "uniform_random_expectation",
            "candidate_id": None,
            "pareto_hit": _mean(
                float(row["labels"]["effectiveness_pareto"]) for row in candidates
            ),
            "compute_aware_pareto_hit": _mean(
                float(row["labels"]["compute_aware_pareto"]) for row in candidates
            ),
            "runtime_sensitive_pareto_hit": _mean(
                float(row["labels"]["runtime_sensitive_pareto"]) for row in candidates
            ),
            "conflict_regret": _mean(
                _ratio(
                    float(row["outcome"]["conflicts_after"]) - best_conflicts,
                    max(1.0, abs(best_conflicts)),
                )
                for row in candidates
            ),
            "solved_rate_regret": _mean(
                best_solved - float(row["outcome"]["solved_rate"]) for row in candidates
            ),
            "auc_regret": _mean(
                _ratio(
                    float(row["outcome"]["conflict_auc"]) - minimum_auc,
                    max(1.0, abs(minimum_auc)),
                )
                for row in candidates
            ),
            "generated_regret": _mean(
                _ratio(
                    float(row["outcome"]["generated"]) - minimum_generated,
                    max(1.0, abs(minimum_generated)),
                )
                for row in candidates
            ),
            "selected_size": min(size_weights, default=0),
            "selection_families": [],
            "selected_runtime": _mean(row["outcome"]["runtime"] for row in candidates),
            "expected_size_distribution": {
                str(key): _ratio(value, len(candidates)) for key, value in size_weights.items()
            },
            "expected_family_distribution": {
                key: _ratio(value, len(candidates)) for key, value in family_weights.items()
            },
        }
    return records


def _uniform_summary(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = list(records.values())
    size_distribution: collections.Counter[str] = collections.Counter()
    family_distribution: collections.Counter[str] = collections.Counter()
    for row in values:
        size_distribution.update(row["expected_size_distribution"])
        family_distribution.update(row["expected_family_distribution"])
    return {
        "state_count": len(values),
        "pareto_top1_hit_rate": _mean(row["pareto_hit"] for row in values),
        "compute_aware_pareto_top1_hit_rate": _mean(
            row["compute_aware_pareto_hit"] for row in values
        ),
        "runtime_sensitive_pareto_top1_hit_rate": _mean(
            row["runtime_sensitive_pareto_hit"] for row in values
        ),
        "mean_conflict_regret": _mean(row["conflict_regret"] for row in values),
        "mean_solved_rate_regret": _mean(row["solved_rate_regret"] for row in values),
        "mean_auc_regret": _mean(row["auc_regret"] for row in values),
        "mean_generated_regret": _mean(row["generated_regret"] for row in values),
        "mean_selected_runtime": _mean(row["selected_runtime"] for row in values),
        "pairwise_accuracy": 0.5,
        "selected_size_fraction": {
            key: _ratio(value, len(values))
            for key, value in sorted(size_distribution.items())
        },
        "selection_family_fraction": {
            key: _ratio(value, len(values))
            for key, value in sorted(family_distribution.items())
        },
        "maximum_size_share": _ratio(
            max(size_distribution.values(), default=0.0), len(values)
        ),
    }


def internal_coverage_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records = {}
    for state_id, candidates in _grouped(rows).items():
        selected = min(
            candidates,
            key=lambda row: (
                -float(
                    row["features"]["realized_dynamic"].get(
                        "realized.internal_conflict_coverage", 0.0
                    )
                ),
                str(row["candidate_id"]),
            ),
        )
        records[state_id] = _selection_record(
            selected, candidates, "maximum_internal_conflict_coverage"
        )
    return records


def oracle_records(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    records = {}
    for state_id, candidates in _grouped(rows).items():
        pareto = [row for row in candidates if row["labels"]["effectiveness_pareto"]]
        selected = min(
            pareto,
            key=lambda row: (
                -float(row["outcome"]["solved_rate"]),
                float(row["outcome"]["conflicts_after"]),
                str(row["candidate_id"]),
            ),
        )
        records[state_id] = _selection_record(selected, candidates, "oracle")
    return records


def cross_validate(
    rows: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    model_parameters: dict[str, Any] | None = None,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, list[PairwiseModel]],
    dict[str, Any],
]:
    records: dict[str, dict[str, dict[str, Any]]] = {
        profile: {} for profile in FEATURE_PROFILES
    }
    models: dict[str, list[PairwiseModel]] = {profile: [] for profile in FEATURE_PROFILES}
    diagnostics: dict[str, Any] = {profile: [] for profile in FEATURE_PROFILES}
    for profile in FEATURE_PROFILES:
        for fold in folds:
            train = [row for row in rows if row["state_id"] in fold["train_states"]]
            validation = [
                row for row in rows if row["state_id"] in fold["validation_states"]
            ]
            model, pair_count = train_pairwise_model(train, profile, model_parameters)
            models[profile].append(model)
            fold_records = evaluate_model(validation, model, profile)
            records[profile].update(fold_records)
            diagnostics[profile].append(
                {
                    "fold": int(fold["fold"]),
                    "train_dominance_pairs": pair_count,
                    "symmetric_train_examples": 2 * pair_count,
                    "validation_dominance_pairs": len(dominance_pairs(validation)),
                    "validation_pairwise_accuracy": pairwise_accuracy(validation, model),
                }
            )
    return records, models, diagnostics


def _profile_pairwise_accuracy(diagnostics: list[dict[str, Any]]) -> float:
    numerator = sum(
        float(row["validation_pairwise_accuracy"]) * int(row["validation_dominance_pairs"])
        for row in diagnostics
    )
    denominator = sum(int(row["validation_dominance_pairs"]) for row in diagnostics)
    return _ratio(numerator, denominator)


def compare_records(
    baseline: dict[str, dict[str, Any]],
    improved: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    baseline_summary = summarize_records(baseline)
    improved_summary = summarize_records(improved)
    baseline_regret = float(baseline_summary["mean_conflict_regret"])
    by_map: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, record in baseline.items():
        if state_id in improved:
            by_map[str(record["map_id"])].append(state_id)
    per_map = {}
    no_worse = 0
    for map_id, states in sorted(by_map.items()):
        baseline_value = _mean(baseline[state]["conflict_regret"] for state in states)
        improved_value = _mean(improved[state]["conflict_regret"] for state in states)
        if improved_value <= baseline_value + 1e-12:
            no_worse += 1
        per_map[map_id] = {
            "baseline_conflict_regret": baseline_value,
            "improved_conflict_regret": improved_value,
            "improvement": baseline_value - improved_value,
        }
    return {
        "pareto_top1_gain": (
            float(improved_summary["pareto_top1_hit_rate"])
            - float(baseline_summary["pareto_top1_hit_rate"])
        ),
        "relative_conflict_regret_reduction": _ratio(
            baseline_regret - float(improved_summary["mean_conflict_regret"]),
            max(1e-12, baseline_regret),
        ),
        "absolute_conflict_regret_improvement": (
            baseline_regret - float(improved_summary["mean_conflict_regret"])
        ),
        "maps_no_worse": no_worse,
        "map_count": len(by_map),
        "per_map": per_map,
    }


def map_bootstrap(
    baseline: dict[str, dict[str, Any]],
    improved: dict[str, dict[str, Any]],
    samples: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("bootstrap sample count must be positive")
    by_map: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, row in baseline.items():
        if state_id in improved:
            by_map[str(row["map_id"])].append(state_id)
    maps = sorted(by_map)
    if not maps:
        raise ValueError("bootstrap has no shared maps")
    hit_by_map = {
        map_id: _mean(
            improved[state]["pareto_hit"] - baseline[state]["pareto_hit"]
            for state in states
        )
        for map_id, states in by_map.items()
    }
    conflict_by_map = {
        map_id: _mean(
            baseline[state]["conflict_regret"] - improved[state]["conflict_regret"]
            for state in states
        )
        for map_id, states in by_map.items()
    }
    rng = random.Random(MODEL_SEED ^ 0x4E4248)
    hit_values = []
    conflict_values = []
    for _ in range(samples):
        selected = [rng.choice(maps) for _ in maps]
        hit_values.append(_mean(hit_by_map[map_id] for map_id in selected))
        conflict_values.append(_mean(conflict_by_map[map_id] for map_id in selected))

    def interval(values: list[float]) -> list[float]:
        values.sort()
        return [
            values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))],
        ]

    return {
        "samples": samples,
        "unit": "map_id",
        "hit_gain_95_ci": interval(hit_values),
        "conflict_improvement_95_ci": interval(conflict_values),
    }


def _context_bundle(row: dict[str, Any]) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in row["features"]["realized_context"].items()
        if name.startswith("context.")
    }


def replace_context_bundle(
    candidates: list[dict[str, Any]], bundle: dict[str, float]
) -> list[dict[str, Any]]:
    changed = []
    for source in candidates:
        row = dict(source)
        row["features"] = dict(source["features"])
        values = {
            name: float(value)
            for name, value in source["features"]["realized_context"].items()
            if not name.startswith("context.")
        }
        values.update(bundle)
        row["features"]["realized_context"] = values
        changed.append(row)
    return changed


def context_permutation_test(
    rows: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    models: list[PairwiseModel],
    dynamic_records: dict[str, dict[str, Any]],
    real_context_records: dict[str, dict[str, Any]],
    permutations: int,
) -> dict[str, Any]:
    if permutations <= 0:
        raise ValueError("permutation count must be positive")
    grouped = _grouped(rows)
    task_bundles: dict[str, dict[str, float]] = {}
    for state_id, candidates in grouped.items():
        task_id = str(candidates[0]["task_id"])
        bundle = _context_bundle(candidates[0])
        if task_id in task_bundles and task_bundles[task_id] != bundle:
            raise ValueError(f"task {task_id} has inconsistent static context")
        task_bundles[task_id] = bundle
    tasks = sorted(task_bundles)
    real_comparison = compare_records(dynamic_records, real_context_records)
    rng = random.Random(MODEL_SEED ^ 0xC07E57)
    cached_records: dict[tuple[str, str], dict[str, Any]] = {}
    for donor_task in tasks:
        for fold, model in zip(folds, models):
            changed_by_state = {
                state_id: replace_context_bundle(
                    grouped[state_id], task_bundles[donor_task]
                )
                for state_id in sorted(fold["validation_states"])
            }
            selections = batch_model_selections(changed_by_state, model)
            for state_id, changed in changed_by_state.items():
                selected = changed[selections[state_id]]
                cached_records[(state_id, donor_task)] = _selection_record(
                    selected, changed, "realized_context_permuted"
                )
    hit_values = []
    conflict_values = []
    for _ in range(permutations):
        donors = list(tasks)
        rng.shuffle(donors)
        donor_by_task = dict(zip(tasks, donors))
        permuted_records = {
            state_id: cached_records[
                (state_id, donor_by_task[str(candidates[0]["task_id"])])
            ]
            for state_id, candidates in grouped.items()
        }
        comparison = compare_records(dynamic_records, permuted_records)
        hit_values.append(float(comparison["pareto_top1_gain"]))
        conflict_values.append(
            float(comparison["relative_conflict_regret_reduction"])
        )
    real_hit = float(real_comparison["pareto_top1_gain"])
    real_conflict = float(real_comparison["relative_conflict_regret_reduction"])
    return {
        "permutations": permutations,
        "unit": "task_id",
        "real_hit_gain": real_hit,
        "real_conflict_regret_reduction": real_conflict,
        "hit_gain_percentile": _ratio(sum(value <= real_hit for value in hit_values), permutations),
        "conflict_reduction_percentile": _ratio(
            sum(value <= real_conflict for value in conflict_values), permutations
        ),
        "null_hit_gain_mean": _mean(hit_values),
        "null_conflict_reduction_mean": _mean(conflict_values),
        "cached_state_context_evaluations": len(cached_records),
    }


def _profile_diagnostics(rows: list[dict[str, Any]], profile: str) -> dict[str, Any]:
    names = _feature_names(rows, profile)
    first: dict[str, float] = {}
    constant = {name: True for name in names}
    digests = {name: hashlib.sha256() for name in names}
    for row in rows:
        values = row["features"][profile]
        for name in names:
            value = float(values.get(name, 0.0))
            if name not in first:
                first[name] = value
            elif first[name] != value:
                constant[name] = False
            digests[name].update(struct.pack("<d", value))
    by_digest: dict[str, list[str]] = collections.defaultdict(list)
    for name, digest in digests.items():
        by_digest[digest.hexdigest()].append(name)
    return {
        "feature_count": len(names),
        "constant_features": [name for name in names if constant[name]],
        "duplicate_feature_groups": [
            group for group in sorted(by_digest.values()) if len(group) > 1
        ],
    }


def feature_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        profile: _profile_diagnostics(rows, profile) for profile in FEATURE_PROFILES
    }


def _oracle_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sizes: collections.Counter[int] = collections.Counter()
    pareto_counts = []
    for candidates in _grouped(rows).values():
        pareto = [row for row in candidates if row["labels"]["effectiveness_pareto"]]
        pareto_counts.append(len(pareto))
        sizes.update({int(row["actual_size"]) for row in pareto})
    return {
        "supported_sizes": {str(key): value for key, value in sorted(sizes.items())},
        "multiple_sizes_supported": len(sizes) > 1,
        "mean_pareto_candidates_per_state": _mean(pareto_counts),
        "maximum_pareto_candidates_per_state": max(pareto_counts, default=0),
    }


def _beats_simple_baseline(
    learned: dict[str, Any], baseline: dict[str, Any]
) -> bool:
    no_worse = (
        float(learned["pareto_top1_hit_rate"])
        >= float(baseline["pareto_top1_hit_rate"]) - 1e-12
        and float(learned["mean_conflict_regret"])
        <= float(baseline["mean_conflict_regret"]) + 1e-12
    )
    strict = (
        float(learned["pareto_top1_hit_rate"])
        > float(baseline["pareto_top1_hit_rate"]) + 1e-12
        or float(learned["mean_conflict_regret"])
        < float(baseline["mean_conflict_regret"]) - 1e-12
    )
    return no_worse and strict


def acceptance_report(
    summaries: dict[str, Any],
    comparisons: dict[str, Any],
    bootstraps: dict[str, Any],
    permutation: dict[str, Any],
    oracle_support: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    realized = comparisons["realized_dynamic_vs_proposal_dynamic"]
    realized_bootstrap = bootstraps["realized_dynamic_vs_proposal_dynamic"]
    context = comparisons["realized_context_vs_realized_dynamic"]
    context_bootstrap = bootstraps["realized_context_vs_realized_dynamic"]
    realized_summary = summaries["realized_dynamic"]
    realized_pass = (
        realized["pareto_top1_gain"] >= float(thresholds["minimum_top1_gain"])
        and realized["relative_conflict_regret_reduction"]
        >= float(thresholds["minimum_conflict_regret_reduction"])
        and realized_bootstrap["hit_gain_95_ci"][1] >= 0.0
        and realized_bootstrap["conflict_improvement_95_ci"][1] >= 0.0
        and realized["maps_no_worse"] >= int(thresholds["minimum_maps_no_worse"])
        and not (
            oracle_support["multiple_sizes_supported"]
            and realized_summary["maximum_size_share"]
            > float(thresholds["maximum_size_share"])
        )
    )
    context_pass = (
        context["pareto_top1_gain"] >= float(thresholds["minimum_context_top1_gain"])
        and context["relative_conflict_regret_reduction"]
        >= float(thresholds["minimum_context_conflict_regret_reduction"])
        and permutation["hit_gain_percentile"]
        >= float(thresholds["minimum_context_permutation_percentile"])
        and permutation["conflict_reduction_percentile"]
        >= float(thresholds["minimum_context_permutation_percentile"])
        and context_bootstrap["hit_gain_95_ci"][1] >= 0.0
        and context_bootstrap["conflict_improvement_95_ci"][1] >= 0.0
    )
    simple_baselines = {
        name: _beats_simple_baseline(realized_summary, summaries[name])
        for name in ("uniform_random", "internal_conflict_coverage")
    }
    proposal_beats_simple = all(
        _beats_simple_baseline(summaries["proposal_dynamic"], summaries[name])
        for name in ("uniform_random", "internal_conflict_coverage")
    )
    if realized_pass and context_pass:
        decision = "collect_independent_ranking_confirmation_and_restore_transfer_claim"
    elif realized_pass:
        decision = "advance_dynamic_realized_ranking_and_shrink_static_transfer_claim"
    elif proposal_beats_simple:
        decision = "retain_proposal_metadata_ranking_only"
    elif not any(simple_baselines.values()):
        decision = "stop_learning_expansion_and_redesign_candidate_or_repair_order_action"
    else:
        decision = "realized_signal_is_inconclusive_do_not_expand_or_train_rl"
    return {
        "passed": bool(realized_pass),
        "static_transfer_passed": bool(context_pass),
        "gates": {
            "realized_ranking": {
                "passed": bool(realized_pass),
                "comparison": realized,
                "bootstrap": realized_bootstrap,
                "requirement": "top-1 +5pp, conflict-regret -5%, no significant bootstrap degradation, >=4/6 maps no worse, no unsupported >80% size collapse",
            },
            "static_context": {
                "passed": bool(context_pass),
                "comparison": context,
                "bootstrap": context_bootstrap,
                "permutation": permutation,
                "requirement": "top-1 +5pp, conflict-regret -5%, both permutation percentiles >=95%, no significant bootstrap degradation",
            },
            "learned_realized_beats_simple_baselines": {
                "passed": all(simple_baselines.values()),
                "comparisons": simple_baselines,
            },
        },
        "decision": decision,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# InitLNS Realized-Neighborhood Ranking Audit",
        "",
        f"Registered ranking gate: **{'PASS' if report['acceptance']['passed'] else 'FAIL'}**",
        f"Static-context gate: **{'PASS' if report['acceptance']['static_transfer_passed'] else 'FAIL'}**",
        "",
        "## Data",
        "",
        f"- Independent states: {report['integrity']['state_count']}",
        f"- Maps: {report['integrity']['map_count']}",
        f"- Explicit candidates: {report['integrity']['candidate_count']}",
        f"- PP-order outcomes: {report['integrity']['outcome_count']}",
        f"- Trials aggregated per candidate: {report['integrity']['trials_per_candidate']}",
        "",
        "## Leave-one-map-out results",
        "",
        "| Selector | Pareto top-1 | Conflict regret | AUC regret | Generated regret | Pairwise accuracy | Max size share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in (
        "uniform_random",
        "internal_conflict_coverage",
        "proposal_dynamic",
        "realized_dynamic",
        "realized_context",
        "oracle",
    ):
        value = report["summaries"][name]
        accuracy = value.get("pairwise_accuracy")
        accuracy_text = "n/a" if accuracy is None else f"{accuracy:.4f}"
        lines.append(
            f"| {name} | {value['pareto_top1_hit_rate']:.4f} | "
            f"{value['mean_conflict_regret']:.4f} | "
            f"{value['mean_auc_regret']:.4f} | "
            f"{value['mean_generated_regret']:.4f} | {accuracy_text} | "
            f"{value['maximum_size_share']:.4f} |"
        )
    lines.extend(["", "## Registered decisions", ""])
    for name, gate in report["acceptance"]["gates"].items():
        lines.append(f"- {name}: **{'PASS' if gate['passed'] else 'FAIL'}**")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"`{report['acceptance']['decision']}`",
            "",
            "The model never sees post-repair outcomes as features. All eight PP-order "
            "trials are aggregated before labeling, and entire maps are held out during "
            "evaluation. This is a small diagnostic audit over six maps, not final "
            "generalization evidence.",
            "",
            "## Timings",
            "",
        ]
    )
    for name, value in report["timings_seconds"].items():
        lines.append(f"- {name}: {value:.3f} s")
    return "\n".join(lines) + "\n"


def run_realized_neighborhood_ranking_audit(
    collection: str | Path,
    config_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    config = _read_json(Path(config_path).resolve())
    expected = dict(config["expected"])
    stage = time.perf_counter()
    rows, integrity = build_ranking_index(
        collection,
        expected_states=int(expected["states"]),
        expected_candidates=int(expected["candidates"]),
        expected_outcomes=int(expected["outcomes"]),
        expected_trials=int(expected["trials_per_candidate"]),
        expected_maps=int(expected["maps"]),
    )
    timings["index_build"] = time.perf_counter() - stage
    folds = leave_one_map_out_folds(rows)
    if len(folds) != int(expected["maps"]):
        raise ValueError("leave-one-map-out fold count does not match expected maps")
    stage = time.perf_counter()
    diagnostics = feature_diagnostics(rows)
    timings["feature_diagnostics"] = time.perf_counter() - stage
    stage = time.perf_counter()
    records, models, pair_diagnostics = cross_validate(
        rows, folds, dict(config.get("model", {}))
    )
    timings["pairwise_training_and_evaluation"] = time.perf_counter() - stage
    stage = time.perf_counter()
    random_records = uniform_random_records(rows)
    heuristic_records = internal_coverage_records(rows)
    perfect_records = oracle_records(rows)
    baseline_records = {
        "uniform_random": random_records,
        "internal_conflict_coverage": heuristic_records,
        "oracle": perfect_records,
    }
    summaries: dict[str, Any] = {
        "uniform_random": _uniform_summary(random_records),
        "internal_conflict_coverage": summarize_records(heuristic_records),
        "oracle": summarize_records(perfect_records),
    }
    for profile in FEATURE_PROFILES:
        summaries[profile] = summarize_records(
            records[profile], _profile_pairwise_accuracy(pair_diagnostics[profile])
        )
    timings["baseline_evaluation"] = time.perf_counter() - stage
    stage = time.perf_counter()
    comparisons = {
        "realized_dynamic_vs_proposal_dynamic": compare_records(
            records["proposal_dynamic"], records["realized_dynamic"]
        ),
        "realized_context_vs_realized_dynamic": compare_records(
            records["realized_dynamic"], records["realized_context"]
        ),
    }
    bootstrap_samples = int(config["evaluation"]["bootstrap_samples"])
    bootstraps = {
        "realized_dynamic_vs_proposal_dynamic": map_bootstrap(
            records["proposal_dynamic"], records["realized_dynamic"], bootstrap_samples
        ),
        "realized_context_vs_realized_dynamic": map_bootstrap(
            records["realized_dynamic"], records["realized_context"], bootstrap_samples
        ),
    }
    timings["map_bootstrap"] = time.perf_counter() - stage
    stage = time.perf_counter()
    permutation = context_permutation_test(
        rows,
        folds,
        models["realized_context"],
        records["realized_dynamic"],
        records["realized_context"],
        int(config["evaluation"]["context_permutations"]),
    )
    timings["context_permutation"] = time.perf_counter() - stage
    oracle_support = _oracle_support(rows)
    acceptance = acceptance_report(
        summaries,
        comparisons,
        bootstraps,
        permutation,
        oracle_support,
        dict(config["thresholds"]),
    )
    digest = hashlib.sha256(
        "\n".join(
            f"{row['state_id']}|{row['candidate_id']}|{row['neighborhood_sha256']}|{row['trial_count']}"
            for row in rows
        ).encode("utf-8")
    ).hexdigest()
    configuration_fingerprint = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output_root = Path(output).resolve()
    stage = time.perf_counter()
    _write_jsonl(output_root / "ranking_index.jsonl", rows)
    prediction_rows = []
    for selector, selector_records in (baseline_records | records).items():
        prediction_rows.extend(
            dict(record) for _, record in sorted(selector_records.items())
        )
    _write_jsonl(output_root / "predictions.jsonl", prediction_rows)
    models_root = output_root / "models"
    for profile, profile_models in models.items():
        for fold, model in enumerate(profile_models):
            path = models_root / f"pairwise__{profile}__fold_{fold}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                pickle.dump(model, stream)
    _write_json(
        output_root / "run_config.json",
        {
            "schema_version": SCHEMA_VERSION,
            "collection": str(Path(collection).resolve()),
            "source_run_fingerprint": integrity["source_run_fingerprint"],
            "configuration": config,
            "configuration_fingerprint": configuration_fingerprint,
            "index_sha256": digest,
        },
    )
    timings["artifact_write"] = time.perf_counter() - stage
    timings["total"] = time.perf_counter() - total_started
    report = {
        "schema_version": SCHEMA_VERSION,
        "model_seed": MODEL_SEED,
        "configuration_fingerprint": configuration_fingerprint,
        "collection": str(Path(collection).resolve()),
        "index_sha256": digest,
        "integrity": integrity,
        "independent_unit": "state nested within map; map is the cross-validation and bootstrap unit",
        "folds": [
            {
                key: sorted(value) if isinstance(value, set) else value
                for key, value in fold.items()
            }
            for fold in folds
        ],
        "pre_registration": {
            "feature_profiles": list(FEATURE_PROFILES),
            "primary_objectives": ["maximize solved_rate", "minimize conflicts_after"],
            "conflict_auc_role": "reported only; redundant with remaining conflicts at Horizon 1",
            "generated_role": "compute-aware sensitivity only",
            "runtime_role": "machine-sensitivity diagnostic only",
            "learner": "fixed pairwise HistGradientBoostingClassifier",
            "cross_validation": "leave-one-map-out",
            "bootstrap_samples": bootstrap_samples,
            "context_permutations": int(config["evaluation"]["context_permutations"]),
        },
        "feature_diagnostics": diagnostics,
        "pairwise_training": pair_diagnostics,
        "summaries": summaries,
        "comparisons": comparisons,
        "map_bootstraps": bootstraps,
        "context_permutation": permutation,
        "oracle_support": oracle_support,
        "acceptance": acceptance,
        "timings_seconds": timings,
        "limitations": [
            "Only 23 independent states from six maps are available.",
            "Candidate neighborhoods were proposed by existing Target/Collision/Random generators.",
            "All source states come from official Adaptive trajectories.",
            "Passing this audit permits independent confirmation; it is not final transfer evidence.",
        ],
    }
    _write_json(output_root / "realized_neighborhood_ranking_audit.json", report)
    (output_root / "realized_neighborhood_ranking_audit.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report


__all__ = [
    "FEATURE_PROFILES",
    "acceptance_report",
    "batch_model_selections",
    "build_ranking_index",
    "context_permutation_test",
    "effectiveness_dominates",
    "explicit_neighborhood_features",
    "leave_one_map_out_folds",
    "replace_context_bundle",
    "run_realized_neighborhood_ranking_audit",
    "train_pairwise_model",
]
