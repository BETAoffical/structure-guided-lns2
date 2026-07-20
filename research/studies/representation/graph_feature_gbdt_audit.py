from __future__ import annotations

import collections
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments._common import mean as _mean, ratio as _ratio
from experiments.closed_loop_confirmation import _sha256, score_online_candidates
from research.studies.representation.local_representation_audit import ConflictEvent, analyze_state
from research.studies.policy.policy_visited_aggregation_analysis import (
    _portable_model,
    _portable_payload,
    train_equal_state_pairwise_model,
)
from research.studies.policy.ranking_objective_audit import (
    _evaluate_model,
    _map_bootstrap,
    _maps_no_worse,
    leave_one_train_map_out,
    objective_acceptance,
)
from research.studies.neighborhood.realized_neighborhood_ranking_audit import (
    _grouped,
    dominance_pairs,
    pairwise_accuracy,
    summarize_records,
)
from experiments.repair_collection import (
    _fingerprint,
    _read_json,
    _read_jsonl,
    _write_json,
    _write_jsonl,
)


SCHEMA = "lns2.graph_feature_gbdt_audit.v1"
SCHEMA_VERSION = 1
PROFILE = "graph_augmented"
BASELINE_PROFILE = "realized_dynamic"
CHALLENGER = "graph_augmented_gbdt"

STRUCTURAL_FEATURE_NAMES = (
    "graph.selected_active_count",
    "graph.induced_edge_density",
    "graph.induced_component_count",
    "graph.induced_largest_component_ratio",
    "graph.boundary_vertex_count",
    "graph.boundary_vertex_ratio",
    "graph.cut_conductance",
    "graph.one_hop_agent_coverage",
    "graph.two_hop_agent_coverage",
    "graph.one_hop_edge_coverage",
    "graph.two_hop_edge_coverage",
    "graph.articulation_agent_coverage",
    "graph.internal_bridge_coverage",
    "graph.incident_bridge_coverage",
    "graph.selected_core_mean",
    "graph.selected_core_max",
    "graph.selected_minus_unselected_core_mean",
    "graph.selected_harmonic_mean",
    "graph.selected_harmonic_max",
    "graph.selected_minus_unselected_harmonic_mean",
    "graph.top_degree_agent_coverage",
    "graph.top_harmonic_agent_coverage",
    "graph.selected_minus_unselected_degree_mean",
)
TEMPORAL_SCOPES = ("internal", "boundary", "incident")
TEMPORAL_METRICS = (
    "event_mass_ratio",
    "pair_repeat_excess_ratio",
    "maximum_pair_mass_ratio",
    "first_time_ratio",
    "mean_time_ratio",
    "last_time_ratio",
    "early_event_ratio",
    "middle_event_ratio",
    "late_event_ratio",
    "vertex_event_ratio",
)
TEMPORAL_FEATURE_NAMES = tuple(
    f"temporal.{scope}.{metric}"
    for scope in TEMPORAL_SCOPES
    for metric in TEMPORAL_METRICS
    if not (
        metric == "event_mass_ratio" and scope in {"internal", "incident"}
    )
)
GRAPH_FEATURE_NAMES = STRUCTURAL_FEATURE_NAMES + TEMPORAL_FEATURE_NAMES
FORBIDDEN_FEATURE_FRAGMENTS = (
    "outcome",
    "label",
    "after",
    "runtime",
    "generated",
    "layout",
    "task_variant",
    "agent_density",
    "context.",
)
COMPACT_RETAINED_FIELDS = (
    "state_id",
    "candidate_id",
    "candidate_key",
    "map_id",
    "task_id",
    "split",
    "agents",
    "actual_size",
    "selection_families",
    "outcome",
    "labels",
    "trial_count",
    "neighborhood_sha256",
)


def _validate_config(config: dict[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("unsupported graph-feature GBDT audit config")
    if str(config.get("baseline_profile")) != BASELINE_PROFILE:
        raise ValueError("graph-feature audit baseline profile changed")
    if str(config.get("feature_profile")) != PROFILE:
        raise ValueError("graph-feature audit profile changed")
    if str(config.get("challenger")) != CHALLENGER:
        raise ValueError("graph-feature audit challenger changed")
    expected = {
        "learning_rate": 0.05,
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 20,
        "l2_regularization": 0.1,
        "random_state": 20260714,
    }
    if dict(config.get("model_parameters", {})) != expected:
        raise ValueError("graph-feature GBDT parameters changed")
    if int(config.get("bootstrap_samples", 0)) != 5000:
        raise ValueError("graph-feature audit requires 5,000 bootstrap samples")


def _registered_paths(project_root: Path) -> dict[str, Path]:
    return {
        "graph_audit_report_sha256": project_root
        / "build/initlns-graph-representation-audit-v2/graph_representation_audit.json",
        "graph_equivalence_report_sha256": project_root
        / "build/initlns-graph-representation-audit-v2/equivalence_report.json",
        "current_lomo_predictions_sha256": project_root
        / "build/initlns-model-capacity-audit-v1/lomo_predictions__current.jsonl",
        "aggregate_train_index_sha256": project_root
        / "build/initlns-policy-visited-natural-v2-training/aggregate_train_index.jsonl",
        "validation_index_sha256": project_root
        / "build/initlns-policy-visited-natural-v2-training/validation_index.jsonl",
        "policy_candidates_sha256": project_root
        / "build/initlns-policy-visited-natural-v2-collection/candidates.jsonl",
        "historical_candidates_sha256": project_root
        / "build/realized-neighborhood-stability-probe-v1/candidates.jsonl",
        "validation_gbdt_predictions_sha256": project_root
        / "build/initlns-policy-visited-natural-v2-offline/offline_predictions__v2_realized_dynamic.jsonl",
    }


def validate_registered_inputs(
    project_root: Path, config: dict[str, Any]
) -> dict[str, str]:
    actual = {
        name: _sha256(path) for name, path in _registered_paths(project_root).items()
    }
    expected = {
        str(name): str(value).lower()
        for name, value in dict(config["registered_inputs"]).items()
    }
    if actual != expected:
        raise ValueError(f"registered graph-feature inputs changed: {actual}")
    graph_report = _read_json(
        _registered_paths(project_root)["graph_audit_report_sha256"]
    )
    equivalence = _read_json(
        _registered_paths(project_root)["graph_equivalence_report_sha256"]
    )
    if (
        str(graph_report.get("decision"))
        != "stop_supervised_representation_expansion"
        or bool(graph_report.get("validation_evaluated"))
        or not bool(equivalence.get("passed"))
    ):
        raise ValueError("graph-feature audit prerequisite result changed")
    return actual


@dataclass(frozen=True)
class ConflictGraph:
    agent_ids: tuple[int, ...]
    active: frozenset[int]
    adjacency: dict[int, frozenset[int]]
    edges: frozenset[tuple[int, int]]
    events: tuple[ConflictEvent, ...]
    articulation: frozenset[int]
    bridges: frozenset[tuple[int, int]]
    core: dict[int, int]
    harmonic: dict[int, float]
    horizon: int


def _articulation_and_bridges(
    nodes: Iterable[int], adjacency: dict[int, set[int]]
) -> tuple[set[int], set[tuple[int, int]]]:
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    child_count: collections.Counter[int] = collections.Counter()
    articulation: set[int] = set()
    bridges: set[tuple[int, int]] = set()
    clock = 0
    for root in sorted(nodes):
        if root in discovery:
            continue
        parent[root] = None
        discovery[root] = low[root] = clock
        clock += 1
        stack: list[tuple[int, Any]] = [(root, iter(sorted(adjacency[root])))]
        while stack:
            node, neighbors = stack[-1]
            try:
                neighbor = next(neighbors)
            except StopIteration:
                stack.pop()
                ancestor = parent[node]
                if ancestor is None:
                    if child_count[node] > 1:
                        articulation.add(node)
                else:
                    low[ancestor] = min(low[ancestor], low[node])
                    edge = tuple(sorted((ancestor, node)))
                    if low[node] > discovery[ancestor]:
                        bridges.add(edge)
                    if (
                        parent[ancestor] is not None
                        and low[node] >= discovery[ancestor]
                    ):
                        articulation.add(ancestor)
                continue
            if neighbor not in discovery:
                parent[neighbor] = node
                child_count[node] += 1
                discovery[neighbor] = low[neighbor] = clock
                clock += 1
                stack.append((neighbor, iter(sorted(adjacency[neighbor]))))
            elif parent[node] != neighbor:
                low[node] = min(low[node], discovery[neighbor])
    return articulation, bridges


def _core_numbers(nodes: Iterable[int], adjacency: dict[int, set[int]]) -> dict[int, int]:
    remaining = set(nodes)
    result: dict[int, int] = {}
    level = 0
    while remaining:
        level = max(
            level,
            min(sum(neighbor in remaining for neighbor in adjacency[node]) for node in remaining),
        )
        queue = collections.deque(
            sorted(
                node
                for node in remaining
                if sum(neighbor in remaining for neighbor in adjacency[node]) <= level
            )
        )
        queued = set(queue)
        while queue:
            node = queue.popleft()
            if node not in remaining:
                continue
            remaining.remove(node)
            result[node] = level
            for neighbor in sorted(adjacency[node] & remaining):
                if (
                    neighbor not in queued
                    and sum(value in remaining for value in adjacency[neighbor]) <= level
                ):
                    queue.append(neighbor)
                    queued.add(neighbor)
    return result


def _harmonic_centrality(
    nodes: Iterable[int], adjacency: dict[int, set[int]]
) -> dict[int, float]:
    ordered = sorted(nodes)
    denominator = max(1, len(ordered) - 1)
    result = {}
    for start in ordered:
        distances = {start: 0}
        queue = collections.deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(adjacency[node]):
                if neighbor not in distances:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        result[start] = sum(
            1.0 / distance for node, distance in distances.items() if node != start
        ) / denominator
    return result


def build_conflict_graph(state: dict[str, Any]) -> ConflictGraph:
    analysis = analyze_state(state)
    agent_ids = tuple(sorted(int(agent["id"]) for agent in state["agents"]))
    adjacency = {agent_id: set() for agent_id in agent_ids}
    for left, right in analysis.pair_set:
        adjacency[left].add(right)
        adjacency[right].add(left)
    active = {node for node in agent_ids if adjacency[node]}
    articulation, bridges = _articulation_and_bridges(agent_ids, adjacency)
    horizon = max(
        (len(agent["path"]) - 1 for agent in state["agents"]), default=0
    )
    return ConflictGraph(
        agent_ids=agent_ids,
        active=frozenset(active),
        adjacency={node: frozenset(values) for node, values in adjacency.items()},
        edges=frozenset(analysis.pair_set),
        events=tuple(analysis.events),
        articulation=frozenset(articulation),
        bridges=frozenset(bridges),
        core=_core_numbers(agent_ids, adjacency),
        harmonic=_harmonic_centrality(agent_ids, adjacency),
        horizon=horizon,
    )


def _component_sizes(nodes: set[int], graph: ConflictGraph) -> list[int]:
    remaining = set(nodes)
    sizes = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        size = 0
        stack = [start]
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in graph.adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def _expand_hops(selected: set[int], graph: ConflictGraph, hops: int) -> set[int]:
    reached = set(selected) & set(graph.active)
    frontier = set(reached)
    for _ in range(hops):
        frontier = {
            neighbor
            for node in frontier
            for neighbor in graph.adjacency[node]
            if neighbor not in reached
        }
        reached.update(frontier)
    return reached


def _top_agents(values: dict[int, float], active: set[int]) -> set[int]:
    if not active:
        return set()
    count = max(1, math.ceil(len(active) * 0.1))
    return set(
        sorted(active, key=lambda node: (-float(values[node]), node))[:count]
    )


def _event_scope_features(
    scope: str,
    events: list[ConflictEvent],
    total_events: int,
    horizon: int,
) -> dict[str, float]:
    prefix = f"temporal.{scope}."
    pair_counts = collections.Counter(
        tuple(sorted((event.left, event.right))) for event in events
    )
    times = [int(event.time) for event in events]
    scale = max(1, horizon)
    early = sum(time * 3 <= scale for time in times)
    late = sum(time * 3 > 2 * scale for time in times)
    middle = len(times) - early - late
    result = {
        prefix + "event_mass_ratio": _ratio(len(events), total_events),
        prefix + "pair_repeat_excess_ratio": _ratio(
            sum(max(0, count - 1) for count in pair_counts.values()), len(events)
        ),
        prefix + "maximum_pair_mass_ratio": _ratio(
            max(pair_counts.values(), default=0), len(events)
        ),
        prefix + "first_time_ratio": _ratio(min(times, default=0), scale),
        prefix + "mean_time_ratio": _ratio(_mean(times), scale),
        prefix + "last_time_ratio": _ratio(max(times, default=0), scale),
        prefix + "early_event_ratio": _ratio(early, len(events)),
        prefix + "middle_event_ratio": _ratio(middle, len(events)),
        prefix + "late_event_ratio": _ratio(late, len(events)),
        prefix + "vertex_event_ratio": _ratio(
            sum(event.kind == "vertex" for event in events), len(events)
        ),
    }
    if scope in {"internal", "incident"}:
        del result[prefix + "event_mass_ratio"]
    return result


def extract_graph_features(
    graph: ConflictGraph, selected_agents: Iterable[int]
) -> dict[str, float]:
    selected = set(map(int, selected_agents))
    known = set(graph.agent_ids)
    if not selected or not selected.issubset(known):
        raise ValueError("graph feature candidate contains an unknown agent")
    selected_active = selected & set(graph.active)
    unselected_active = set(graph.active) - selected
    internal = {
        edge for edge in graph.edges if edge[0] in selected and edge[1] in selected
    }
    boundary = {
        edge for edge in graph.edges if (edge[0] in selected) != (edge[1] in selected)
    }
    boundary_vertices = {
        node for edge in boundary for node in edge if node not in selected
    }
    component_sizes = _component_sizes(selected_active, graph)
    one_hop = _expand_hops(selected, graph, 1)
    two_hop = _expand_hops(selected, graph, 2)
    volume_selected = sum(len(graph.adjacency[node]) for node in selected)
    volume_unselected = sum(len(graph.adjacency[node]) for node in known - selected)
    degrees = {node: float(len(graph.adjacency[node])) for node in graph.agent_ids}
    top_degree = _top_agents(degrees, set(graph.active))
    top_harmonic = _top_agents(graph.harmonic, set(graph.active))
    features = {
        "graph.selected_active_count": float(len(selected_active)),
        "graph.induced_edge_density": _ratio(
            2 * len(internal), len(selected_active) * max(0, len(selected_active) - 1)
        ),
        "graph.induced_component_count": float(len(component_sizes)),
        "graph.induced_largest_component_ratio": _ratio(
            max(component_sizes, default=0), len(selected_active)
        ),
        "graph.boundary_vertex_count": float(len(boundary_vertices)),
        "graph.boundary_vertex_ratio": _ratio(
            len(boundary_vertices), len(graph.active)
        ),
        "graph.cut_conductance": _ratio(
            len(boundary), min(volume_selected, volume_unselected)
        ),
        "graph.one_hop_agent_coverage": _ratio(len(one_hop), len(graph.active)),
        "graph.two_hop_agent_coverage": _ratio(len(two_hop), len(graph.active)),
        "graph.one_hop_edge_coverage": _ratio(
            sum(left in one_hop and right in one_hop for left, right in graph.edges),
            len(graph.edges),
        ),
        "graph.two_hop_edge_coverage": _ratio(
            sum(left in two_hop and right in two_hop for left, right in graph.edges),
            len(graph.edges),
        ),
        "graph.articulation_agent_coverage": _ratio(
            len(selected & set(graph.articulation)), len(graph.articulation)
        ),
        "graph.internal_bridge_coverage": _ratio(
            sum(left in selected and right in selected for left, right in graph.bridges),
            len(graph.bridges),
        ),
        "graph.incident_bridge_coverage": _ratio(
            sum(left in selected or right in selected for left, right in graph.bridges),
            len(graph.bridges),
        ),
        "graph.selected_core_mean": _mean(graph.core[node] for node in selected),
        "graph.selected_core_max": max(
            (float(graph.core[node]) for node in selected), default=0.0
        ),
        "graph.selected_minus_unselected_core_mean": _mean(
            graph.core[node] for node in selected_active
        )
        - _mean(graph.core[node] for node in unselected_active),
        "graph.selected_harmonic_mean": _mean(
            graph.harmonic[node] for node in selected
        ),
        "graph.selected_harmonic_max": max(
            (graph.harmonic[node] for node in selected), default=0.0
        ),
        "graph.selected_minus_unselected_harmonic_mean": _mean(
            graph.harmonic[node] for node in selected_active
        )
        - _mean(graph.harmonic[node] for node in unselected_active),
        "graph.top_degree_agent_coverage": _ratio(
            len(selected & top_degree), len(top_degree)
        ),
        "graph.top_harmonic_agent_coverage": _ratio(
            len(selected & top_harmonic), len(top_harmonic)
        ),
        "graph.selected_minus_unselected_degree_mean": _mean(
            degrees[node] for node in selected_active
        )
        - _mean(degrees[node] for node in unselected_active),
    }
    scopes = {
        "internal": [
            event
            for event in graph.events
            if event.left in selected and event.right in selected
        ],
        "boundary": [
            event
            for event in graph.events
            if (event.left in selected) != (event.right in selected)
        ],
        "incident": [
            event
            for event in graph.events
            if event.left in selected or event.right in selected
        ],
    }
    for scope, events in scopes.items():
        features.update(
            _event_scope_features(
                scope, events, len(graph.events), graph.horizon
            )
        )
    if set(features) != set(GRAPH_FEATURE_NAMES):
        raise ValueError("graph feature schema changed")
    if any(not math.isfinite(float(value)) for value in features.values()):
        raise ValueError("graph feature extraction produced a non-finite value")
    return features


def _raw_states(project_root: Path) -> dict[str, dict[str, Any]]:
    paths = _registered_paths(project_root)
    raw = {}
    for path in (
        paths["historical_candidates_sha256"],
        paths["policy_candidates_sha256"],
    ):
        for row in _read_jsonl(path):
            state_id = str(row["state_id"])
            if state_id in raw:
                raise ValueError(f"duplicate raw graph-feature state: {state_id}")
            raw[state_id] = row
    return raw


def build_graph_feature_index(
    project_root: Path, config: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    registered = validate_registered_inputs(project_root, config)
    paths = _registered_paths(project_root)
    source_rows = _read_jsonl(paths["aggregate_train_index_sha256"]) + _read_jsonl(
        paths["validation_index_sha256"]
    )
    by_state = _grouped(source_rows)
    raw = _raw_states(project_root)
    if set(by_state) != set(raw):
        raise ValueError("graph-feature raw states and indexes differ")
    indexed = []
    for state_id in sorted(by_state):
        source = raw[state_id]
        graph = build_conflict_graph(source["state"])
        raw_candidates = {
            str(candidate["candidate_id"]): candidate
            for candidate in source["candidates"]
        }
        for row in sorted(by_state[state_id], key=lambda value: str(value["candidate_id"])):
            candidate_id = str(row["candidate_id"])
            if candidate_id not in raw_candidates:
                raise ValueError("graph-feature candidate is missing from raw state")
            agents = sorted(map(int, row["agents"]))
            if agents != sorted(map(int, raw_candidates[candidate_id]["agents"])):
                raise ValueError("graph-feature candidate agents changed")
            graph_features = extract_graph_features(graph, agents)
            base = {
                str(name): float(value)
                for name, value in row["features"][BASELINE_PROFILE].items()
            }
            if set(base) & set(graph_features):
                raise ValueError("graph features overwrite registered dynamic features")
            augmented = {
                name: row[name] for name in COMPACT_RETAINED_FIELDS if name in row
            }
            augmented["schema"] = "lns2.graph_feature_candidate.v2"
            augmented["schema_version"] = SCHEMA_VERSION
            augmented["features"] = {PROFILE: {**base, **graph_features}}
            indexed.append(augmented)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "graph_feature_index.jsonl"
    grouped = _grouped(indexed)
    state_split = {
        state_id: str(rows[0]["split"]) for state_id, rows in grouped.items()
    }
    feature_names = sorted(
        {name for row in indexed for name in row["features"][PROFILE]}
    )
    if any(
        fragment in name.lower()
        for name in GRAPH_FEATURE_NAMES
        for fragment in FORBIDDEN_FEATURE_FRAGMENTS
    ):
        raise ValueError("graph-feature input leakage detected")
    duplicate_base_columns = _duplicate_base_columns(indexed)
    if duplicate_base_columns:
        raise ValueError(
            f"graph-feature columns duplicate registered inputs: {duplicate_base_columns}"
        )
    compact_rows = []
    for row in indexed:
        compact = {name: value for name, value in row.items() if name != "features"}
        compact["feature_values"] = [
            float(row["features"][PROFILE].get(name, 0.0)) for name in feature_names
        ]
        compact_rows.append(compact)
    _write_jsonl(index_path, compact_rows)
    manifest = {
        "schema": "lns2.graph_feature_index_manifest.v2",
        "schema_version": 2,
        "state_count": len(grouped),
        "candidate_count": len(indexed),
        "trial_count": sum(int(row["outcome"]["trial_count"]) for row in indexed),
        "policy_trial_count": sum(
            int(row["outcome"]["trial_count"])
            for row in indexed
            if str(row["split"]) in {"policy_train", "policy_validation"}
        ),
        "train_state_count": sum(
            split == "policy_train" for split in state_split.values()
        ),
        "validation_state_count": sum(
            split == "policy_validation" for split in state_split.values()
        ),
        "anchor_state_count": sum(
            split not in {"policy_train", "policy_validation"}
            for split in state_split.values()
        ),
        "map_count": len({str(row["map_id"]) for row in indexed}),
        "base_feature_count": len(feature_names) - len(GRAPH_FEATURE_NAMES),
        "structural_feature_names": list(STRUCTURAL_FEATURE_NAMES),
        "temporal_feature_names": list(TEMPORAL_FEATURE_NAMES),
        "feature_names": feature_names,
        "index_encoding": "ordered_feature_vector",
        "duplicate_base_columns": duplicate_base_columns,
        "registered_inputs": registered,
        "index_sha256": _sha256(index_path),
        "static_context_used": False,
        "outcome_features_used": False,
    }
    if (
        manifest["state_count"],
        manifest["candidate_count"],
        manifest["trial_count"],
        manifest["policy_trial_count"],
        manifest["anchor_state_count"],
        manifest["train_state_count"],
        manifest["validation_state_count"],
    ) != (465, 8326, 34952, 31656, 23, 288, 154):
        raise ValueError("graph-feature registered data counts changed")
    _write_json(output_root / "index_manifest.json", manifest)
    return manifest


def load_graph_feature_index(index_root: Path) -> list[dict[str, Any]]:
    manifest = _read_json(index_root / "index_manifest.json")
    if (
        str(manifest.get("schema")) != "lns2.graph_feature_index_manifest.v2"
        or str(manifest.get("index_encoding")) != "ordered_feature_vector"
    ):
        raise ValueError("unsupported graph-feature index encoding")
    names = list(map(str, manifest["feature_names"]))
    rows = []
    for compact in _read_jsonl(index_root / "graph_feature_index.jsonl"):
        values = list(map(float, compact.pop("feature_values")))
        if len(values) != len(names):
            raise ValueError("graph-feature vector length differs from manifest")
        compact["features"] = {PROFILE: dict(zip(names, values))}
        rows.append(compact)
    if len(rows) != int(manifest["candidate_count"]):
        raise ValueError("graph-feature compact index row count changed")
    return rows


def _baseline_records(path: Path, *, expected_states: int, expected_maps: int) -> dict[str, dict[str, Any]]:
    records = {str(row["state_id"]): row for row in _read_jsonl(path)}
    if len(records) != expected_states or len(
        {str(row["map_id"]) for row in records.values()}
    ) != expected_maps:
        raise ValueError("registered graph-feature baseline predictions changed")
    return records


def _weighted_pairwise_accuracy(
    inputs: list[tuple[list[dict[str, Any]], Any]]
) -> float:
    weighted = 0.0
    count = 0
    for rows, model in inputs:
        pair_count = len(dominance_pairs(rows))
        weighted += pairwise_accuracy(rows, model) * pair_count
        count += pair_count
    return weighted / count if count else 0.0


def _feature_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = {
        name: [float(row["features"][PROFILE][name]) for row in rows]
        for name in GRAPH_FEATURE_NAMES
    }
    constants = [name for name, column in values.items() if len(set(column)) == 1]
    return {
        "candidate_count": len(rows),
        "structural_feature_count": len(STRUCTURAL_FEATURE_NAMES),
        "temporal_feature_count": len(TEMPORAL_FEATURE_NAMES),
        "constant_graph_features": constants,
        "varying_graph_feature_count": len(GRAPH_FEATURE_NAMES) - len(constants),
        "groups_are_diagnostic_only": True,
    }


def _duplicate_base_columns(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    base_names = sorted(
        {
            name
            for row in rows
            for name in row["features"][PROFILE]
            if name not in GRAPH_FEATURE_NAMES
        }
    )
    base_columns: dict[tuple[float, ...], list[str]] = collections.defaultdict(list)
    for name in base_names:
        base_columns[
            tuple(
                float(row["features"][PROFILE].get(name, 0.0))
                for row in rows
            )
        ].append(name)
    duplicates = []
    for name in GRAPH_FEATURE_NAMES:
        column = tuple(float(row["features"][PROFILE][name]) for row in rows)
        for base_name in base_columns.get(column, []):
            duplicates.append({"graph_feature": name, "base_feature": base_name})
    return duplicates


def run_train_audit(
    project_root: Path,
    index_root: Path,
    config: dict[str, Any],
    output_root: Path,
) -> tuple[dict[str, Any], Any | None]:
    rows = load_graph_feature_index(index_root)
    train = [row for row in rows if str(row["split"]) == "policy_train"]
    anchors = [
        row
        for row in rows
        if str(row["split"]) not in {"policy_train", "policy_validation"}
    ]
    if len(_grouped(train)) != 288 or len(_grouped(anchors)) != 23:
        raise ValueError("graph-feature Train or anchor state count changed")
    folds = leave_one_train_map_out(train, anchors)
    records = {}
    accuracy_inputs = []
    fold_reports = []
    for fold_number, split in enumerate(folds):
        print(
            f"[graph-feature-gbdt] fold={fold_number} map={split['validation_map']}",
            flush=True,
        )
        model, diagnostics = train_equal_state_pairwise_model(
            list(split["fit_rows"]), PROFILE, dict(config["model_parameters"])
        )
        held = list(split["held_rows"])
        selected = _evaluate_model(held, model, CHALLENGER)
        if set(records) & set(selected):
            raise ValueError("graph-feature LOMO evaluated a state more than once")
        records.update(selected)
        accuracy_inputs.append((held, model))
        fold_reports.append(
            {
                "fold": fold_number,
                "validation_map": split["validation_map"],
                "validation_state_count": len(_grouped(held)),
                "anchor_state_count": split["anchor_state_count"],
                "training_diagnostics": diagnostics,
            }
        )
    summary = summarize_records(records, _weighted_pairwise_accuracy(accuracy_inputs))
    baseline = _baseline_records(
        _registered_paths(project_root)["current_lomo_predictions_sha256"],
        expected_states=288,
        expected_maps=12,
    )
    baseline_summary = summarize_records(baseline)
    expected = config["expected_current_lomo"]
    if not math.isclose(
        float(baseline_summary["pareto_top1_hit_rate"]),
        float(expected["pareto_top1_hit_rate"]),
        rel_tol=0.0,
        abs_tol=float(expected["absolute_tolerance"]),
    ) or not math.isclose(
        float(baseline_summary["mean_conflict_regret"]),
        float(expected["mean_conflict_regret"]),
        rel_tol=0.0,
        abs_tol=float(expected["absolute_tolerance"]),
    ):
        raise ValueError("graph-feature current GBDT baseline did not reproduce")
    bootstrap = _map_bootstrap(
        baseline, records, int(config["bootstrap_samples"])
    )
    maps_no_worse, map_details = _maps_no_worse(baseline, records)
    comparison = objective_acceptance(
        baseline_summary,
        summary,
        bootstrap,
        maps_no_worse,
        12,
        dict(config["thresholds"]),
    )
    _write_jsonl(
        output_root / "lomo_predictions__graph_augmented_gbdt.jsonl",
        [records[state_id] for state_id in sorted(records)],
    )
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "baseline": baseline_summary,
        "challenger": summary,
        "comparison": comparison,
        "bootstrap": bootstrap,
        "map_details": map_details,
        "folds": fold_reports,
        "feature_diagnostics": _feature_diagnostics(train + anchors),
        "passed": comparison["passed"],
        "validation_labels_used_for_selection": False,
        "static_context_used": False,
    }
    _write_json(output_root / "train_report.json", report)
    if not comparison["passed"]:
        return report, None
    full_model, _ = train_equal_state_pairwise_model(
        anchors + train, PROFILE, dict(config["model_parameters"])
    )
    return report, full_model


def run_validation_audit(
    project_root: Path,
    index_root: Path,
    config: dict[str, Any],
    output_root: Path,
    model: Any,
) -> dict[str, Any]:
    rows = load_graph_feature_index(index_root)
    validation = [row for row in rows if str(row["split"]) == "policy_validation"]
    if len(_grouped(validation)) != 154 or len(
        {str(row["map_id"]) for row in validation}
    ) != 6:
        raise ValueError("graph-feature Validation design changed")
    records = _evaluate_model(validation, model, CHALLENGER)
    summary = summarize_records(records, pairwise_accuracy(validation, model))
    baseline = _baseline_records(
        _registered_paths(project_root)["validation_gbdt_predictions_sha256"],
        expected_states=154,
        expected_maps=6,
    )
    baseline_summary = summarize_records(baseline)
    bootstrap = _map_bootstrap(
        baseline, records, int(config["bootstrap_samples"])
    )
    maps_no_worse, map_details = _maps_no_worse(baseline, records)
    comparison = objective_acceptance(
        baseline_summary,
        summary,
        bootstrap,
        maps_no_worse,
        6,
        dict(config["thresholds"]),
    )
    _write_jsonl(
        output_root / "validation_predictions__graph_augmented_gbdt.jsonl",
        [records[state_id] for state_id in sorted(records)],
    )
    portable = None
    if comparison["passed"]:
        payload = _portable_payload(model, "graph-feature-gbdt-audit")
        model_path = output_root / "models/graph_augmented_gbdt.json"
        _write_json(model_path, payload)
        portable_model = _portable_model(payload)
        mismatches = 0
        for candidates in _grouped(validation).values():
            native, _, _ = score_online_candidates(candidates, model)
            portable_selected, _, _ = score_online_candidates(
                candidates, portable_model
            )
            mismatches += native != portable_selected
        portable = {
            "model_sha256": _sha256(model_path),
            "selection_mismatch_count": mismatches,
            "passed": mismatches == 0,
        }
        if not portable["passed"]:
            raise ValueError("graph-feature portable model selection mismatch")
    report = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "baseline": baseline_summary,
        "challenger": summary,
        "comparison": comparison,
        "bootstrap": bootstrap,
        "map_details": map_details,
        "passed": comparison["passed"],
        "portable_model": portable,
        "validation_labels_used_for_training": False,
        "validation_labels_used_for_model_selection": False,
    }
    _write_json(output_root / "validation_report.json", report)
    return report


def run_graph_feature_gbdt_audit(
    project_root: str | Path,
    config_path: str | Path,
    output: str | Path,
    *,
    phase: str = "all",
) -> dict[str, Any]:
    if phase not in {"index", "train", "all"}:
        raise ValueError("phase must be index, train, or all")
    root = Path(project_root).resolve()
    output_root = Path(output).resolve()
    config = _read_json(Path(config_path).resolve())
    _validate_config(config)
    implementation_sha256 = _sha256(Path(__file__))
    run_fingerprint = _fingerprint(
        {"configuration": config, "implementation_sha256": implementation_sha256}
    )
    output_root.mkdir(parents=True, exist_ok=True)
    run_config_path = output_root / "run_config.json"
    run_config = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "configuration": config,
        "implementation_sha256": implementation_sha256,
        "run_fingerprint": run_fingerprint,
    }
    if run_config_path.exists() and str(
        _read_json(run_config_path).get("run_fingerprint")
    ) != run_fingerprint:
        raise ValueError("graph-feature output belongs to a different run fingerprint")
    if (output_root / "completed.json").exists():
        raise ValueError("graph-feature output is already complete")
    _write_json(run_config_path, run_config)
    started = time.perf_counter()
    (output_root / "completed.json").unlink(missing_ok=True)
    _write_json(
        output_root / "run_status.json",
        {
            "status": "running",
            "phase": phase,
            "pid": os.getpid(),
            "run_fingerprint": run_fingerprint,
        },
    )
    try:
        manifest = build_graph_feature_index(root, config, output_root / "index")
        if phase == "index":
            final = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "index": manifest,
                "phase": phase,
                "elapsed_seconds": time.perf_counter() - started,
            }
        else:
            train_report, model = run_train_audit(
                root, output_root / "index", config, output_root
            )
            validation = (
                run_validation_audit(
                    root, output_root / "index", config, output_root, model
                )
                if model is not None and phase == "all"
                else None
            )
            decision = (
                "eligible_for_independent_graph_feature_confirmation"
                if validation is not None and validation["passed"]
                else "train_passed_awaiting_validation"
                if train_report["passed"] and validation is None
                else "stop_after_development_validation"
                if validation is not None
                else "stop_supervised_representation_expansion"
            )
            final = {
                "schema": SCHEMA,
                "schema_version": SCHEMA_VERSION,
                "index": manifest,
                "train": train_report,
                "validation": validation,
                "decision": decision,
                "validation_evaluated": validation is not None,
                "new_solver_data_collected": False,
                "static_context_used": False,
                "rl_trained": False,
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_json(output_root / "graph_feature_gbdt_audit.json", final)
        status = {
            "status": "completed",
            "phase": phase,
            "run_fingerprint": run_fingerprint,
            "decision": final.get("decision"),
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(output_root / "run_status.json", status)
        _write_json(output_root / "completed.json", status)
        return final
    except BaseException as error:
        _write_json(
            output_root / "run_status.json",
            {
                "status": "failed",
                "phase": phase,
                "run_fingerprint": run_fingerprint,
                "error": repr(error),
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        raise


__all__ = [
    "CHALLENGER",
    "GRAPH_FEATURE_NAMES",
    "STRUCTURAL_FEATURE_NAMES",
    "TEMPORAL_FEATURE_NAMES",
    "build_conflict_graph",
    "build_graph_feature_index",
    "extract_graph_features",
    "load_graph_feature_index",
    "run_graph_feature_gbdt_audit",
    "validate_registered_inputs",
]
