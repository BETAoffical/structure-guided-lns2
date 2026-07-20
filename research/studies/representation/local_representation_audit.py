from __future__ import annotations

import collections
import hashlib
import itertools
import math
import pickle
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from experiments._common import (
    mean as _mean,
    population_std as _std,
    ratio as _ratio,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)
from research.studies.context.context_audit import (
    MODEL_SEED,
    PairwiseModel,
    _average_outcomes,
    _candidate_key,
    _dataset_contexts,
    _dataset_root,
    _feature_names,
    _horizon,
    _pair_vector,
    _read_jsonl,
    _resolve,
    _stage_labels,
    _vector,
    candidate_features,
)


LOCAL_AUDIT_SCHEMA_VERSION = 1
INDEX_SCHEMA_VERSION = 1
FEATURE_PROFILES = (
    "dynamic_action",
    "local_pre",
    "local_pre_context",
    "realized",
    "realized_context",
)
ACTION_PROFILES = FEATURE_PROFILES[:3]
REALIZED_PROFILES = FEATURE_PROFILES[3:]
OBJECTIVES = ("effectiveness", "compute_aware", "runtime_sensitivity")
HORIZONS = (1, 4)
FORBIDDEN_FEATURE_FRAGMENTS = (
    "conflicts_after",
    "conflict_reduction",
    "cost_improvement",
    "branch_runtime",
    "step_runtime",
    "replan_success",
    "outcome",
    "post_repair",
)


@dataclass(frozen=True)
class ConflictEvent:
    time: int
    kind: str
    left: int
    right: int
    cells: tuple[int, ...]


@dataclass
class StateAnalysis:
    rows: int
    cols: int
    free_cells: set[int]
    degrees: dict[int, int]
    articulation: set[int]
    obstacle_rate_2: dict[int, float]
    obstacle_rate_4: dict[int, float]
    visit_heat: collections.Counter[int]
    agent_heat: collections.Counter[int]
    events: list[ConflictEvent]
    pair_set: set[tuple[int, int]]
    component_id: dict[int, int]
    component_members: dict[int, set[int]]


@dataclass
class StaticGridAnalysis:
    rows: int
    cols: int
    obstacles: tuple[int, ...]
    free_cells: set[int]
    degrees: dict[int, int]
    articulation: set[int]
    obstacle_rate_2: dict[int, float]
    obstacle_rate_4: dict[int, float]


def reconstruct_conflicts(agents: list[dict[str, Any]]) -> list[ConflictEvent]:
    agent_ids = [int(agent["id"]) for agent in agents]
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("state contains duplicate agent ids")
    paths = {int(agent["id"]): [int(cell) for cell in agent["path"]] for agent in agents}
    if any(not path for path in paths.values()):
        raise ValueError("state contains an empty agent path")
    horizon = max(map(len, paths.values()), default=0)
    events: list[ConflictEvent] = []
    for time in range(horizon):
        occupancy: dict[int, list[int]] = collections.defaultdict(list)
        for agent_id, path in paths.items():
            occupancy[path[min(time, len(path) - 1)]].append(agent_id)
        for cell, occupants in occupancy.items():
            for left, right in itertools.combinations(sorted(occupants), 2):
                events.append(ConflictEvent(time, "vertex", left, right, (cell,)))
        if time == 0:
            continue
        transitions: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
        for agent_id, path in paths.items():
            previous = path[min(time - 1, len(path) - 1)]
            current = path[min(time, len(path) - 1)]
            if previous != current:
                transitions[(previous, current)].append(agent_id)
        for (previous, current), forward_agents in sorted(transitions.items()):
            if previous >= current:
                continue
            for left in forward_agents:
                for right in transitions.get((current, previous), []):
                    first, second = sorted((left, right))
                    events.append(
                        ConflictEvent(time, "edge", first, second, (previous, current))
                    )
    events.sort(key=lambda event: (event.time, event.kind, event.left, event.right))
    return events


def _grid_neighbors(cell: int, rows: int, cols: int, free: set[int]) -> list[int]:
    row, col = divmod(cell, cols)
    result = []
    for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        next_row, next_col = row + delta_row, col + delta_col
        next_cell = next_row * cols + next_col
        if 0 <= next_row < rows and 0 <= next_col < cols and next_cell in free:
            result.append(next_cell)
    return result


def articulation_cells(rows: int, cols: int, obstacles: list[int]) -> set[int]:
    free = {index for index, blocked in enumerate(obstacles) if not int(blocked)}
    adjacency = {cell: _grid_neighbors(cell, rows, cols, free) for cell in free}
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    parent: dict[int, int | None] = {}
    child_count: collections.Counter[int] = collections.Counter()
    articulation: set[int] = set()
    clock = 0

    for root in sorted(free):
        if root in discovery:
            continue
        parent[root] = None
        discovery[root] = low[root] = clock
        clock += 1
        stack: list[tuple[int, Any]] = [(root, iter(adjacency[root]))]
        while stack:
            cell, neighbors = stack[-1]
            try:
                neighbor = next(neighbors)
            except StopIteration:
                stack.pop()
                ancestor = parent[cell]
                if ancestor is None:
                    if child_count[cell] > 1:
                        articulation.add(cell)
                else:
                    low[ancestor] = min(low[ancestor], low[cell])
                    if parent[ancestor] is not None and low[cell] >= discovery[ancestor]:
                        articulation.add(ancestor)
                continue
            if neighbor not in discovery:
                parent[neighbor] = cell
                child_count[cell] += 1
                discovery[neighbor] = low[neighbor] = clock
                clock += 1
                stack.append((neighbor, iter(adjacency[neighbor])))
            elif parent[cell] != neighbor:
                low[cell] = min(low[cell], discovery[neighbor])
    return articulation


def _obstacle_rates(
    rows: int, cols: int, obstacles: list[int], radius: int
) -> dict[int, float]:
    rates = {}
    for cell, blocked in enumerate(obstacles):
        if int(blocked):
            continue
        row, col = divmod(cell, cols)
        total = 0
        obstacle_count = 0
        for nearby_row in range(max(0, row - radius), min(rows, row + radius + 1)):
            for nearby_col in range(max(0, col - radius), min(cols, col + radius + 1)):
                total += 1
                obstacle_count += int(obstacles[nearby_row * cols + nearby_col])
        rates[cell] = _ratio(obstacle_count, total)
    return rates


def _conflict_components(
    agent_ids: Iterable[int], edges: Iterable[tuple[int, int]]
) -> tuple[dict[int, int], dict[int, set[int]]]:
    adjacency = {int(agent_id): set() for agent_id in agent_ids}
    active: set[int] = set()
    for left, right in edges:
        if left not in adjacency or right not in adjacency:
            raise ValueError(f"conflict edge references unknown agent: {(left, right)}")
        adjacency[left].add(right)
        adjacency[right].add(left)
        active.update((left, right))
    component_id: dict[int, int] = {}
    component_members: dict[int, set[int]] = {}
    for start in sorted(active):
        if start in component_id:
            continue
        identifier = len(component_members)
        members = {start}
        stack = [start]
        component_id[start] = identifier
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor not in component_id:
                    component_id[neighbor] = identifier
                    members.add(neighbor)
                    stack.append(neighbor)
        component_members[identifier] = members
    return component_id, component_members


def analyze_static_grid(state: dict[str, Any]) -> StaticGridAnalysis:
    rows = int(state["rows"])
    cols = int(state["cols"])
    if rows <= 0 or cols <= 0:
        raise ValueError("grid rows and cols must be positive")
    obstacles = tuple(int(value) for value in state["obstacles"])
    if len(obstacles) != rows * cols:
        raise ValueError("obstacle grid dimensions do not match rows and cols")
    if any(value not in {0, 1} for value in obstacles):
        raise ValueError("obstacle grid values must be 0 or 1")
    free = {index for index, blocked in enumerate(obstacles) if not blocked}
    return StaticGridAnalysis(
        rows=rows,
        cols=cols,
        obstacles=obstacles,
        free_cells=free,
        degrees={cell: len(_grid_neighbors(cell, rows, cols, free)) for cell in free},
        articulation=articulation_cells(rows, cols, list(obstacles)),
        obstacle_rate_2=_obstacle_rates(rows, cols, list(obstacles), 2),
        obstacle_rate_4=_obstacle_rates(rows, cols, list(obstacles), 4),
    )


def analyze_state(
    state: dict[str, Any], *, static_grid: StaticGridAnalysis | None = None
) -> StateAnalysis:
    rows = int(state["rows"])
    cols = int(state["cols"])
    obstacles = tuple(int(value) for value in state["obstacles"])
    if static_grid is None:
        static_grid = analyze_static_grid(state)
    elif (
        rows != static_grid.rows
        or cols != static_grid.cols
        or obstacles != static_grid.obstacles
    ):
        raise ValueError("cached static grid does not match solver state")
    free = static_grid.free_cells
    agents = list(state.get("agents", []))
    if not agents:
        raise ValueError("state must contain at least one agent")
    agent_ids = [int(agent["id"]) for agent in agents]
    if len(agent_ids) != len(set(agent_ids)):
        raise ValueError("state contains duplicate agent ids")
    known_agents = set(agent_ids)
    for agent in agents:
        agent_id = int(agent["id"])
        path = [int(cell) for cell in agent.get("path", [])]
        if not path:
            raise ValueError(f"agent {agent_id} has an empty path")
        if any(cell < 0 or cell >= rows * cols for cell in path):
            raise ValueError(f"agent {agent_id} path contains an out-of-range cell")
        if any(cell not in free for cell in path):
            raise ValueError(f"agent {agent_id} path enters an obstacle")
        if int(agent.get("start", path[0])) != path[0]:
            raise ValueError(f"agent {agent_id} path does not start at its start cell")
        if int(agent.get("goal", path[-1])) != path[-1]:
            raise ValueError(f"agent {agent_id} path does not end at its goal cell")
        for previous, current in zip(path, path[1:]):
            previous_row, previous_col = divmod(previous, cols)
            current_row, current_col = divmod(current, cols)
            distance = abs(previous_row - current_row) + abs(previous_col - current_col)
            if distance not in {0, 1}:
                raise ValueError(f"agent {agent_id} path contains a non-adjacent move")
    for edge in state.get("conflict_edges", []):
        if len(edge) != 2:
            raise ValueError(f"invalid conflict edge: {edge}")
        left, right = int(edge[0]), int(edge[1])
        if left == right:
            raise ValueError(f"conflict edge contains the same agent twice: {edge}")
        if left not in known_agents or right not in known_agents:
            raise ValueError(f"conflict edge references unknown agent: {edge}")
    visit_heat: collections.Counter[int] = collections.Counter()
    agent_heat: collections.Counter[int] = collections.Counter()
    for agent in agents:
        path = [int(cell) for cell in agent["path"]]
        visit_heat.update(path)
        agent_heat.update(set(path))
    events = reconstruct_conflicts(agents)
    pair_set = {(event.left, event.right) for event in events}
    component_id, component_members = _conflict_components(agent_ids, pair_set)
    return StateAnalysis(
        rows=rows,
        cols=cols,
        free_cells=free,
        degrees=static_grid.degrees,
        articulation=static_grid.articulation,
        obstacle_rate_2=static_grid.obstacle_rate_2,
        obstacle_rate_4=static_grid.obstacle_rate_4,
        visit_heat=visit_heat,
        agent_heat=agent_heat,
        events=events,
        pair_set=pair_set,
        component_id=component_id,
        component_members=component_members,
    )


def _path_wait_ratio(path: list[int]) -> float:
    return _ratio(sum(left == right for left, right in zip(path, path[1:])), len(path) - 1)


def _path_values(path: list[int], values: dict[int, float | int]) -> list[float]:
    return [float(values.get(cell, 0.0)) for cell in path]


def seed_local_features(
    state: dict[str, Any], analysis: StateAnalysis, seed_id: int, size: int
) -> dict[str, float]:
    by_id = {int(agent["id"]): agent for agent in state["agents"]}
    seed = by_id[seed_id]
    path = [int(cell) for cell in seed["path"]]
    events = [
        event
        for event in analysis.events
        if event.left == seed_id or event.right == seed_id
    ]
    times = [event.time for event in events]
    degrees = _path_values(path, analysis.degrees)
    visits = _path_values(path, analysis.visit_heat)
    agent_visits = _path_values(path, analysis.agent_heat)
    obstacle_2 = _path_values(path, analysis.obstacle_rate_2)
    obstacle_4 = _path_values(path, analysis.obstacle_rate_4)
    component = analysis.component_id.get(seed_id)
    component_size = len(analysis.component_members.get(component, set()))
    path_cost = max(1, int(seed.get("path_cost", len(path) - 1)))
    start_row, start_col = divmod(int(seed["start"]), analysis.cols)
    goal_row, goal_col = divmod(int(seed["goal"]), analysis.cols)
    return {
        "local.requested_size_ratio_agents": _ratio(size, len(state["agents"])),
        "local.requested_size_ratio_component": _ratio(size, component_size),
        "local.seed_event_count": float(len(events)),
        "local.seed_vertex_event_count": float(sum(e.kind == "vertex" for e in events)),
        "local.seed_edge_event_count": float(sum(e.kind == "edge" for e in events)),
        "local.seed_unique_conflict_cell_count": float(
            len({cell for event in events for cell in event.cells})
        ),
        "local.seed_conflict_time_first_ratio": _ratio(min(times, default=0), path_cost),
        "local.seed_conflict_time_mean_ratio": _ratio(_mean(times), path_cost),
        "local.seed_conflict_time_last_ratio": _ratio(max(times, default=0), path_cost),
        "local.seed_wait_ratio": _path_wait_ratio(path),
        "local.seed_unique_path_cell_ratio": _ratio(len(set(path)), len(path)),
        "local.seed_path_visit_heat_mean": _mean(visits),
        "local.seed_path_visit_heat_max": max(visits, default=0.0),
        "local.seed_path_agent_heat_mean": _mean(agent_visits),
        "local.seed_path_agent_heat_max": max(agent_visits, default=0.0),
        "local.seed_path_degree_mean": _mean(degrees),
        "local.seed_path_degree_min": min(degrees, default=0.0),
        "local.seed_path_low_degree_ratio": _ratio(sum(value <= 2 for value in degrees), len(degrees)),
        "local.seed_path_articulation_ratio": _ratio(
            sum(cell in analysis.articulation for cell in path), len(path)
        ),
        "local.seed_path_obstacle_rate_r2": _mean(obstacle_2),
        "local.seed_path_obstacle_rate_r4": _mean(obstacle_4),
        "local.seed_start_goal_row_distance": _ratio(
            abs(goal_row - start_row), max(1, analysis.rows - 1)
        ),
        "local.seed_start_goal_col_distance": _ratio(
            abs(goal_col - start_col), max(1, analysis.cols - 1)
        ),
    }


def _aggregate(prefix: str, values: list[float]) -> dict[str, float]:
    return {
        f"{prefix}_mean": _mean(values),
        f"{prefix}_std": _std(values),
        f"{prefix}_max": max(values, default=0.0),
        f"{prefix}_sum": sum(values),
    }


def realized_neighborhood_features(
    state: dict[str, Any],
    analysis: StateAnalysis,
    seed_id: int,
    requested_size: int,
    neighborhood: list[int],
) -> dict[str, float]:
    if len(neighborhood) != len(set(neighborhood)):
        raise ValueError("realized neighborhood contains duplicate agent ids")
    by_id = {int(agent["id"]): agent for agent in state["agents"]}
    if any(agent_id not in by_id for agent_id in neighborhood):
        raise ValueError("realized neighborhood contains an unknown agent id")
    selected = set(neighborhood)
    internal = sum(left in selected and right in selected for left, right in analysis.pair_set)
    boundary = sum((left in selected) != (right in selected) for left, right in analysis.pair_set)
    component_ids = {
        analysis.component_id[agent_id]
        for agent_id in selected
        if agent_id in analysis.component_id
    }
    seed_component = analysis.component_id.get(seed_id)
    seed_members = analysis.component_members.get(seed_component, set())
    selected_agents = [by_id[agent_id] for agent_id in sorted(selected)]
    delays = [float(agent.get("delay", 0)) for agent in selected_agents]
    conflicts = [float(agent.get("conflict_degree", 0)) for agent in selected_agents]
    path_costs = [float(agent.get("path_cost", 0)) for agent in selected_agents]
    stretches = [
        _ratio(agent.get("path_cost", 0), max(1, agent.get("shortest_path_cost", 0)))
        for agent in selected_agents
    ]
    paths = [[int(cell) for cell in agent["path"]] for agent in selected_agents]
    path_sets = [set(path) for path in paths]
    overlaps = []
    for left, right in itertools.combinations(path_sets, 2):
        overlaps.append(_ratio(len(left & right), len(left | right)))
    union = set().union(*path_sets) if path_sets else set()
    if union:
        coordinates = [divmod(cell, analysis.cols) for cell in union]
        span_rows = max(row for row, _ in coordinates) - min(row for row, _ in coordinates) + 1
        span_cols = max(col for _, col in coordinates) - min(col for _, col in coordinates) + 1
    else:
        span_rows = span_cols = 0
    flattened = [cell for path in paths for cell in path]
    degrees = _path_values(flattened, analysis.degrees)
    features = {
        "realized.actual_size": float(len(selected)),
        "realized.actual_size_ratio_agents": _ratio(len(selected), len(state["agents"])),
        "realized.actual_size_ratio_component": _ratio(len(selected), len(seed_members)),
        "realized.requested_size_match": float(len(selected) == requested_size),
        "realized.seed_included": float(seed_id in selected),
        "realized.conflicting_agent_ratio": _ratio(
            sum(float(agent.get("conflict_degree", 0)) > 0 for agent in selected_agents),
            len(selected_agents),
        ),
        "realized.component_count": float(len(component_ids)),
        "realized.seed_component_coverage": _ratio(len(selected & seed_members), len(seed_members)),
        "realized.internal_conflict_edges": float(internal),
        "realized.boundary_conflict_edges": float(boundary),
        "realized.incident_conflict_coverage": _ratio(
            internal + boundary, len(analysis.pair_set)
        ),
        "realized.internal_conflict_coverage": _ratio(internal, len(analysis.pair_set)),
        "realized.path_overlap_mean": _mean(overlaps),
        "realized.path_overlap_max": max(overlaps, default=0.0),
        "realized.path_union_cell_ratio": _ratio(len(union), len(analysis.free_cells)),
        "realized.path_bbox_area_ratio": _ratio(
            span_rows * span_cols, analysis.rows * analysis.cols
        ),
        "realized.path_degree_mean": _mean(degrees),
        "realized.path_low_degree_ratio": _ratio(sum(value <= 2 for value in degrees), len(degrees)),
        "realized.path_articulation_ratio": _ratio(
            sum(cell in analysis.articulation for cell in flattened), len(flattened)
        ),
        "realized.path_visit_heat_mean": _mean(
            _path_values(flattened, analysis.visit_heat)
        ),
    }
    features.update(_aggregate("realized.delay", delays))
    features.update(_aggregate("realized.conflict_degree", conflicts))
    features.update(_aggregate("realized.path_cost", path_costs))
    features.update(_aggregate("realized.path_stretch", stretches))
    return features


def _actual_neighborhood(outcome: dict[str, Any]) -> list[int]:
    steps = [step for step in outcome.get("steps", []) if int(step.get("step", -1)) == 1]
    if len(steps) != 1:
        raise ValueError("outcome must contain exactly one candidate step")
    neighborhood = steps[0].get("metrics", {}).get("neighborhood")
    if not isinstance(neighborhood, list) or not neighborhood:
        raise ValueError("candidate step is missing its realized neighborhood")
    return [int(agent_id) for agent_id in neighborhood]


def _profiles(
    state_row: dict[str, Any],
    outcome: dict[str, Any],
    stage: str,
    dataset_context: dict[str, Any] | None,
    analysis: StateAnalysis,
    neighborhood: list[int],
) -> dict[str, dict[str, float]]:
    action = outcome["candidate_action"]
    known_agents = {int(agent["id"]) for agent in state_row["state"]["agents"]}
    seed_id = int(action["seed_agent"])
    if seed_id not in known_agents:
        raise ValueError(f"candidate seed references unknown agent: {seed_id}")
    size = int(action["neighborhood_size"])
    if size <= 0:
        raise ValueError("candidate neighborhood size must be positive")
    base = candidate_features(state_row, outcome, stage, dataset_context)
    dynamic = dict(base["dynamic"])
    dynamic.pop("action.neighborhood_size", None)
    dynamic[f"action.neighborhood_size={size}"] = 1.0
    local = seed_local_features(state_row["state"], analysis, seed_id, size)
    context = {
        name: float(value)
        for name, value in base["full_context"].items()
        if name.startswith("context.")
    }
    realized_features = realized_neighborhood_features(
        state_row["state"],
        analysis,
        seed_id,
        size,
        neighborhood,
    )
    local_pre = dynamic | local
    realized = local_pre | realized_features
    profiles = {
        "dynamic_action": dynamic,
        "local_pre": local_pre,
        "local_pre_context": local_pre | context,
        "realized": realized,
        "realized_context": realized | context,
    }
    for profile, features in profiles.items():
        leaking = sorted(
            name
            for name in features
            if any(fragment in name.lower() for fragment in FORBIDDEN_FEATURE_FRAGMENTS)
        )
        if leaking:
            raise ValueError(f"feature leakage in {profile}: {leaking}")
    return profiles


def _metadata(
    state_row: dict[str, Any], key: tuple[int, str, int], stage: str
) -> dict[str, Any]:
    context = state_row["state"].get("context", {})
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "state_id": str(state_row["state_id"]),
        "episode_id": str(state_row["episode_id"]),
        "split": str(context.get("split", "unknown")),
        "map_id": str(context.get("map_id", "unknown")),
        "task_id": str(context.get("task_id", "unknown")),
        "decision_index": int(state_row["decision_index"]),
        "stage": stage,
        "candidate_key": f"{key[0]}:{key[1]}:{key[2]}",
        "candidate_action": {
            "seed_agent": key[0],
            "heuristic": key[1],
            "neighborhood_size": key[2],
        },
    }


def build_local_indexes(
    collection: str | Path,
    dataset: str | Path | None = None,
    expected_outcomes: int | None = 7344,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(collection).resolve()
    dataset_contexts = _dataset_contexts(_dataset_root(root, dataset))
    manifests = _read_jsonl(root / "counterfactual_manifest.jsonl")
    states: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for manifest in manifests:
        if str(manifest.get("status")) not in {"ok", "resumed"}:
            continue
        states.extend(_read_jsonl(_resolve(root, str(manifest["states_file"]))))
        outcomes.extend(_read_jsonl(_resolve(root, str(manifest["outcomes_file"]))))
    if expected_outcomes is not None and len(outcomes) != expected_outcomes:
        raise ValueError(
            f"expected {expected_outcomes} counterfactual outcomes, found {len(outcomes)}"
        )
    unexpected = sorted(
        {
            str(row["state"].get("context", {}).get("split", "unknown"))
            for row in states
        }
        - {"train", "validation"}
    )
    if unexpected:
        raise ValueError(f"local audit contains forbidden Test/OOD splits: {unexpected}")
    state_index = {str(row["state_id"]): row for row in states}
    if len(state_index) != len(states):
        raise ValueError("duplicate state ids in collection")
    stages = _stage_labels(states)
    analyses: dict[str, StateAnalysis] = {}
    mismatch: list[str] = []
    for state_id, state_row in state_index.items():
        analysis = analyze_state(state_row["state"])
        expected_pairs = {
            tuple(sorted((int(edge[0]), int(edge[1]))))
            for edge in state_row["state"].get("conflict_edges", [])
        }
        if analysis.pair_set != expected_pairs:
            mismatch.append(state_id)
        analyses[state_id] = analysis
    if mismatch:
        raise ValueError(
            f"reconstructed conflict pairs disagree with state edges for {len(mismatch)} states"
        )

    realized_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, tuple[int, str, int]], list[dict[str, Any]]] = collections.defaultdict(list)
    missing_neighborhood = 0
    for outcome in outcomes:
        if not bool(outcome.get("action_valid", False)):
            raise ValueError("collection contains an invalid candidate action")
        if any(_horizon(outcome, horizon) is None for horizon in HORIZONS):
            raise ValueError("collection outcome is missing Horizon 1 or Horizon 4")
        state_id = str(outcome["state_id"])
        if state_id not in state_index:
            raise ValueError(f"outcome references unknown state: {state_id}")
        try:
            neighborhood = _actual_neighborhood(outcome)
        except ValueError:
            missing_neighborhood += 1
            continue
        state_row = state_index[state_id]
        context = state_row["state"].get("context", {})
        key = _candidate_key(outcome)
        profiles = _profiles(
            state_row,
            outcome,
            stages.get(state_id, "unknown"),
            dataset_contexts.get(str(context.get("task_id", ""))),
            analyses[state_id],
            neighborhood,
        )
        row = _metadata(state_row, key, stages.get(state_id, "unknown"))
        row.update(
            {
                "trial_index": int(outcome.get("trial_index", 0)),
                "trial_seed": int(outcome.get("trial_seed", 0)),
                "trial_key": f"{row['candidate_key']}:{int(outcome.get('trial_index', 0))}",
                "realized_neighborhood": neighborhood,
                "realized_neighborhood_sha256": hashlib.sha256(
                    ",".join(map(str, neighborhood)).encode("ascii")
                ).hexdigest(),
                "features": profiles,
                "outcomes": {
                    str(horizon): _average_outcomes([outcome], horizon)
                    for horizon in HORIZONS
                },
            }
        )
        realized_rows.append(row)
        grouped[(state_id, key)].append(outcome)
    if missing_neighborhood:
        raise ValueError(f"{missing_neighborhood} outcomes lack a realized neighborhood")
    if len(realized_rows) != len(outcomes):
        raise ValueError("not all counterfactual outcomes were indexed")

    first_realized = {
        (str(row["state_id"]), str(row["candidate_key"])): row
        for row in realized_rows
    }
    action_rows: list[dict[str, Any]] = []
    for (state_id, key), trial_rows in sorted(grouped.items()):
        source = first_realized[(state_id, f"{key[0]}:{key[1]}:{key[2]}")]
        row = {name: value for name, value in source.items() if name not in {
            "trial_index", "trial_seed", "trial_key", "realized_neighborhood",
            "realized_neighborhood_sha256", "features", "outcomes"
        }}
        row["trial_count"] = len(trial_rows)
        row["features"] = {
            profile: source["features"][profile] for profile in ACTION_PROFILES
        }
        row["outcomes"] = {
            str(horizon): _average_outcomes(trial_rows, horizon) for horizon in HORIZONS
        }
        action_rows.append(row)

    action_rows.sort(key=lambda row: (row["state_id"], row["candidate_key"]))
    realized_rows.sort(key=lambda row: (row["state_id"], row["trial_key"]))
    integrity = {
        "outcome_rows": len(outcomes),
        "action_rows": len(action_rows),
        "realized_rows": len(realized_rows),
        "state_rows": len(states),
        "map_count": len({row["map_id"] for row in action_rows}),
        "task_count": len({row["task_id"] for row in action_rows}),
        "missing_realized_neighborhoods": missing_neighborhood,
        "conflict_pair_mismatch_states": len(mismatch),
        "forbidden_split_rows": 0,
        "passed": True,
    }
    return action_rows, realized_rows, integrity


def _objective_values(outcome: dict[str, Any], mode: str) -> tuple[float, ...]:
    if mode not in OBJECTIVES:
        raise ValueError(f"unknown objective mode: {mode}")
    values = (
        -float(outcome.get("solved_rate", float(bool(outcome["solved"])))),
        float(outcome["conflicts_after"]),
        float(outcome["conflict_auc"]),
    )
    if mode in {"compute_aware", "runtime_sensitivity"}:
        values += (float(outcome["generated"]),)
    if mode == "runtime_sensitivity":
        values += (float(outcome["branch_runtime"]),)
    return values


def _dominates(left: dict[str, Any], right: dict[str, Any], mode: str) -> bool:
    left_values = _objective_values(left, mode)
    right_values = _objective_values(right, mode)
    return all(left <= right for left, right in zip(left_values, right_values)) and any(
        left < right for left, right in zip(left_values, right_values)
    )


def relabel(
    rows: list[dict[str, Any]], horizon: int, mode: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for source in rows:
        row = dict(source)
        row["outcome"] = source["outcomes"][str(horizon)]
        grouped[str(row["state_id"])].append(row)
    result = []
    for candidates in grouped.values():
        for candidate in candidates:
            candidate["pareto"] = not any(
                other is not candidate
                and _dominates(other["outcome"], candidate["outcome"], mode)
                for other in candidates
            )
            candidate["objective_mode"] = mode
            candidate["horizon"] = horizon
            result.append(candidate)
    return sorted(result, key=lambda row: (row["state_id"], row.get("trial_key", row["candidate_key"])))


def _grouped(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    for candidates in grouped.values():
        candidates.sort(key=lambda row: str(row.get("trial_key", row["candidate_key"])))
    return grouped


DominancePair = tuple[dict[str, Any], dict[str, Any], int]


def _dominance_pairs(rows: list[dict[str, Any]], mode: str) -> list[DominancePair]:
    pairs: list[DominancePair] = []
    for candidates in _grouped(rows).values():
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                if _dominates(left["outcome"], right["outcome"], mode):
                    pairs.append((left, right, 1))
                elif _dominates(right["outcome"], left["outcome"], mode):
                    pairs.append((left, right, 0))
    if not pairs:
        raise ValueError("no dominance pairs are available for pairwise training")
    return pairs


def _pairwise_examples(
    pairs: list[DominancePair], profile: str, names: list[str]
) -> tuple[list[list[float]], list[int]]:
    examples: list[list[float]] = []
    labels: list[int] = []
    for left, right, label in pairs:
        examples.append(_pair_vector(left, right, profile, names))
        labels.append(label)
        examples.append(_pair_vector(right, left, profile, names))
        labels.append(1 - label)
    return examples, labels


def _train_pairwise(
    rows: list[dict[str, Any]], profile: str, pairs: list[DominancePair]
) -> PairwiseModel:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    names = _feature_names(rows, profile)
    examples, labels = _pairwise_examples(pairs, profile, names)
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=MODEL_SEED,
    )
    estimator.fit(np.asarray(examples, dtype=float), np.asarray(labels, dtype=int))
    return PairwiseModel(profile, names, estimator)


def _selection_record(selected: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    minimum_auc = min(float(row["outcome"]["conflict_auc"]) for row in candidates)
    minimum_conflicts = min(float(row["outcome"]["conflicts_after"]) for row in candidates)
    selected_auc = float(selected["outcome"]["conflict_auc"])
    selected_conflicts = float(selected["outcome"]["conflicts_after"])
    action = selected["candidate_action"]
    return {
        "map_id": str(selected["map_id"]),
        "task_id": str(selected["task_id"]),
        "pareto_hit": float(bool(selected["pareto"])),
        "auc_regret": _ratio(selected_auc - minimum_auc, max(1.0, abs(minimum_auc))),
        "conflict_regret": _ratio(
            selected_conflicts - minimum_conflicts, max(1.0, abs(minimum_conflicts))
        ),
        "selected_family": f"{action['heuristic']}:{int(action['neighborhood_size'])}",
        "selected_size": int(action["neighborhood_size"]),
    }


def _evaluate_pairwise(
    rows: list[dict[str, Any]], model: PairwiseModel
) -> dict[str, dict[str, Any]]:
    records = {}
    for state_id, candidates in _grouped(rows).items():
        records[state_id] = _selection_record(
            candidates[model.select(candidates)], candidates
        )
    return records


def _summarize(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    sizes = collections.Counter(int(row["selected_size"]) for row in records.values())
    families = collections.Counter(str(row["selected_family"]) for row in records.values())
    return {
        "state_count": len(records),
        "pareto_top1_hit_rate": _mean(row["pareto_hit"] for row in records.values()),
        "mean_auc_regret": _mean(row["auc_regret"] for row in records.values()),
        "mean_conflict_regret": _mean(row["conflict_regret"] for row in records.values()),
        "selected_sizes": {str(key): value for key, value in sorted(sizes.items())},
        "maximum_size_share": _ratio(max(sizes.values(), default=0), len(records)),
        "selected_action_families": dict(sorted(families.items())),
    }


def _map_folds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from sklearn.model_selection import GroupKFold

    states = sorted(_grouped(rows).values(), key=lambda candidates: candidates[0]["state_id"])
    state_ids = [str(candidates[0]["state_id"]) for candidates in states]
    groups = [str(candidates[0]["map_id"]) for candidates in states]
    if len(set(groups)) < 3:
        raise ValueError("local audit requires at least three maps")
    result = []
    splitter = GroupKFold(n_splits=3)
    for fold, (train_indices, validation_indices) in enumerate(splitter.split(state_ids, groups=groups)):
        result.append(
            {
                "fold": fold,
                "train_states": {state_ids[index] for index in train_indices},
                "validation_states": {state_ids[index] for index in validation_indices},
                "train_maps": sorted({groups[index] for index in train_indices}),
                "validation_maps": sorted({groups[index] for index in validation_indices}),
            }
        )
    return result


def _cross_validate(
    labeled: list[dict[str, Any]],
    profile: str,
    folds: list[dict[str, Any]],
    fold_pairs: list[list[DominancePair]],
) -> tuple[dict[str, dict[str, Any]], list[PairwiseModel]]:
    records: dict[str, dict[str, Any]] = {}
    models = []
    for fold, pairs in zip(folds, fold_pairs):
        train = [row for row in labeled if row["state_id"] in fold["train_states"]]
        validation = [row for row in labeled if row["state_id"] in fold["validation_states"]]
        model = _train_pairwise(train, profile, pairs)
        models.append(model)
        records.update(_evaluate_pairwise(validation, model))
    return records, models


def _comparison(
    baseline: dict[str, dict[str, Any]], improved: dict[str, dict[str, Any]]
) -> dict[str, float]:
    baseline_summary = _summarize(baseline)
    improved_summary = _summarize(improved)
    baseline_auc = float(baseline_summary["mean_auc_regret"])
    return {
        "pareto_top1_gain": (
            float(improved_summary["pareto_top1_hit_rate"])
            - float(baseline_summary["pareto_top1_hit_rate"])
        ),
        "relative_auc_regret_reduction": _ratio(
            baseline_auc - float(improved_summary["mean_auc_regret"]),
            max(1e-12, baseline_auc),
        ),
    }


def _map_bootstrap(
    baseline: dict[str, dict[str, Any]],
    improved: dict[str, dict[str, Any]],
    samples: int,
) -> dict[str, list[float]]:
    by_map: dict[str, list[str]] = collections.defaultdict(list)
    for state_id, row in baseline.items():
        if state_id in improved:
            by_map[str(row["map_id"])].append(state_id)
    maps = sorted(by_map)
    map_hit = {
        map_id: _mean(improved[state]["pareto_hit"] - baseline[state]["pareto_hit"] for state in states)
        for map_id, states in by_map.items()
    }
    map_auc = {
        map_id: _mean(baseline[state]["auc_regret"] - improved[state]["auc_regret"] for state in states)
        for map_id, states in by_map.items()
    }
    rng = random.Random(MODEL_SEED ^ 0x10CA1)
    hit_values = []
    auc_values = []
    for _ in range(samples):
        selected = [rng.choice(maps) for _ in maps]
        hit_values.append(_mean(map_hit[map_id] for map_id in selected))
        auc_values.append(_mean(map_auc[map_id] for map_id in selected))

    def interval(values: list[float]) -> list[float]:
        values.sort()
        return [
            values[int(0.025 * (len(values) - 1))],
            values[int(0.975 * (len(values) - 1))],
        ]

    return {"hit_gain_95_ci": interval(hit_values), "auc_improvement_95_ci": interval(auc_values)}


def _context_bundle(row: dict[str, Any], profile: str) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in row["features"][profile].items()
        if name.startswith("context.")
    }


def _with_context_bundle(
    candidates: list[dict[str, Any]],
    profile: str,
    bundle: dict[str, float],
) -> list[dict[str, Any]]:
    changed = []
    for source in candidates:
        row = dict(source)
        row["features"] = dict(source["features"])
        features = {
            name: value
            for name, value in source["features"][profile].items()
            if not name.startswith("context.")
        }
        features.update(bundle)
        row["features"][profile] = features
        changed.append(row)
    return changed


def _permuted_fold_records(
    validation: list[dict[str, Any]],
    model: PairwiseModel,
    fold_number: int,
    permutations: int,
    *,
    use_cache: bool,
) -> list[dict[str, dict[str, Any]]]:
    grouped = _grouped(validation)
    task_sources: dict[str, dict[str, Any]] = {}
    for row in validation:
        task_sources.setdefault(str(row["task_id"]), row)
    tasks = sorted(task_sources)
    bundles = {
        task: _context_bundle(task_sources[task], "local_pre_context")
        for task in tasks
    }
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    if use_cache:
        for state_id, candidates in grouped.items():
            for donor_task, bundle in bundles.items():
                changed = _with_context_bundle(
                    candidates, "local_pre_context", bundle
                )
                cached[(state_id, donor_task)] = _evaluate_pairwise(
                    changed, model
                )[state_id]

    records = [dict() for _ in range(permutations)]
    for permutation in range(permutations):
        donors = list(tasks)
        rng = random.Random(
            MODEL_SEED + permutation * 1009 + int(fold_number) * 9176
        )
        rng.shuffle(donors)
        donor_for = dict(zip(tasks, donors))
        for state_id, candidates in grouped.items():
            source_task = str(candidates[0]["task_id"])
            donor_task = donor_for[source_task]
            if use_cache:
                record = cached[(state_id, donor_task)]
            else:
                changed = _with_context_bundle(
                    candidates,
                    "local_pre_context",
                    bundles[donor_task],
                )
                record = _evaluate_pairwise(changed, model)[state_id]
            records[permutation][state_id] = record
    return records


def _permutation_test(
    rows: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    baseline_records: dict[str, dict[str, Any]],
    real_records: dict[str, dict[str, Any]],
    models: list[PairwiseModel],
    permutations: int,
) -> dict[str, Any]:
    null_records = [dict() for _ in range(permutations)]
    labeled = relabel(rows, 1, "effectiveness")
    for fold, model in zip(folds, models):
        validation = [row for row in labeled if row["state_id"] in fold["validation_states"]]
        fold_records = _permuted_fold_records(
            validation,
            model,
            int(fold["fold"]),
            permutations,
            use_cache=True,
        )
        for permutation, records in enumerate(fold_records):
            null_records[permutation].update(records)
    real = _comparison(baseline_records, real_records)
    null = [_comparison(baseline_records, records) for records in null_records]
    hit_values = [value["pareto_top1_gain"] for value in null]
    auc_values = [value["relative_auc_regret_reduction"] for value in null]
    return {
        "count": permutations,
        "unit": "task_id",
        "real_hit_gain": real["pareto_top1_gain"],
        "real_auc_regret_reduction": real["relative_auc_regret_reduction"],
        "hit_gain_percentile": _ratio(sum(real["pareto_top1_gain"] > value for value in hit_values), permutations),
        "auc_reduction_percentile": _ratio(
            sum(real["relative_auc_regret_reduction"] > value for value in auc_values), permutations
        ),
        "null_hit_gain_range": [min(hit_values, default=0.0), max(hit_values, default=0.0)],
        "null_auc_reduction_range": [min(auc_values, default=0.0), max(auc_values, default=0.0)],
    }


def _oracle(rows: list[dict[str, Any]], horizon: int, mode: str) -> dict[str, Any]:
    labeled = relabel(rows, horizon, mode)
    grouped = _grouped(labeled)
    sizes = sorted({int(row["candidate_action"]["neighborhood_size"]) for row in labeled})
    coverage = {
        str(size): _ratio(
            sum(
                any(row["pareto"] and int(row["candidate_action"]["neighborhood_size"]) == size for row in candidates)
                for candidates in grouped.values()
            ),
            len(grouped),
        )
        for size in sizes
    }
    supported = [size for size, value in coverage.items() if value >= 0.10]
    return {
        "state_count": len(grouped),
        "size_state_coverage": coverage,
        "supported_sizes": supported,
        "multiple_sizes_supported": len(supported) >= 2,
        "mean_pareto_candidates": _mean(sum(bool(row["pareto"]) for row in candidates) for candidates in grouped.values()),
    }


@dataclass
class MetricRegressors:
    profile: str
    feature_names: list[str]
    estimators: dict[str, Any]


def _train_regressors(rows: list[dict[str, Any]], profile: str) -> MetricRegressors:
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor

    names = _feature_names(rows, profile)
    matrix = np.asarray([_vector(row, profile, names) for row in rows], dtype=float)
    targets = {
        "conflicts_after": [float(row["outcome"]["conflicts_after"]) for row in rows],
        "conflict_auc": [float(row["outcome"]["conflict_auc"]) for row in rows],
        "log1p_generated": [math.log1p(float(row["outcome"]["generated"])) for row in rows],
    }
    estimators = {}
    for name, values in targets.items():
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=MODEL_SEED,
        )
        estimator.fit(matrix, np.asarray(values, dtype=float))
        estimators[name] = estimator
    return MetricRegressors(profile, names, estimators)


def _evaluate_regressors(
    rows: list[dict[str, Any]], model: MetricRegressors
) -> dict[str, dict[str, Any]]:
    import numpy as np

    records = {}
    for state_id, candidates in _grouped(rows).items():
        matrix = np.asarray([_vector(row, model.profile, model.feature_names) for row in candidates], dtype=float)
        predicted = {
            name: estimator.predict(matrix) for name, estimator in model.estimators.items()
        }
        fronts = []
        for index in range(len(candidates)):
            value = (
                float(predicted["conflicts_after"][index]),
                float(predicted["conflict_auc"][index]),
                float(predicted["log1p_generated"][index]),
            )
            dominated = False
            for other in range(len(candidates)):
                if other == index:
                    continue
                other_value = (
                    float(predicted["conflicts_after"][other]),
                    float(predicted["conflict_auc"][other]),
                    float(predicted["log1p_generated"][other]),
                )
                if all(left <= right for left, right in zip(other_value, value)) and any(
                    left < right for left, right in zip(other_value, value)
                ):
                    dominated = True
                    break
            if not dominated:
                fronts.append(index)
        selected = min(
            fronts,
            key=lambda index: (
                float(predicted["conflicts_after"][index]),
                float(predicted["conflict_auc"][index]),
                str(candidates[index].get("trial_key", candidates[index]["candidate_key"])),
            ),
        )
        records[state_id] = _selection_record(candidates[selected], candidates)
    return records


def _regression_diagnostics(
    action_rows: list[dict[str, Any]],
    realized_rows: list[dict[str, Any]],
    folds: list[dict[str, Any]],
) -> dict[str, Any]:
    result = {}
    for profile in FEATURE_PROFILES:
        source = action_rows if profile in ACTION_PROFILES else realized_rows
        labeled = relabel(source, 1, "effectiveness")
        records = {}
        for fold in folds:
            train = [row for row in labeled if row["state_id"] in fold["train_states"]]
            validation = [row for row in labeled if row["state_id"] in fold["validation_states"]]
            model = _train_regressors(train, profile)
            records.update(_evaluate_regressors(validation, model))
        result[profile] = _summarize(records)
    return result


def _run_objective(
    action_rows: list[dict[str, Any]],
    realized_rows: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    horizon: int,
    mode: str,
    bootstrap_samples: int,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, list[PairwiseModel]],
    dict[str, Any],
]:
    labeled_sources = {
        "action": relabel(action_rows, horizon, mode),
        "realized": relabel(realized_rows, horizon, mode),
    }
    pair_cache: dict[str, list[list[DominancePair]]] = {}
    pair_diagnostics: dict[str, Any] = {}
    for source_name, labeled in labeled_sources.items():
        fold_pairs = []
        for fold in folds:
            train = [
                row for row in labeled if row["state_id"] in fold["train_states"]
            ]
            fold_pairs.append(_dominance_pairs(train, mode))
        pair_cache[source_name] = fold_pairs
        pair_diagnostics[source_name] = {
            "dominance_pairs_by_fold": [len(pairs) for pairs in fold_pairs],
            "symmetric_examples_by_fold": [2 * len(pairs) for pairs in fold_pairs],
            "dominance_pairs_total": sum(map(len, fold_pairs)),
        }
    records = {}
    models = {}
    for profile in FEATURE_PROFILES:
        source_name = "action" if profile in ACTION_PROFILES else "realized"
        records[profile], models[profile] = _cross_validate(
            labeled_sources[source_name],
            profile,
            folds,
            pair_cache[source_name],
        )
    comparisons = {}
    for baseline, improved in (
        ("dynamic_action", "local_pre"),
        ("local_pre", "local_pre_context"),
        ("local_pre", "realized"),
        ("realized", "realized_context"),
    ):
        key = f"{improved}_vs_{baseline}"
        comparisons[key] = _comparison(records[baseline], records[improved])
        comparisons[key].update(
            _map_bootstrap(records[baseline], records[improved], bootstrap_samples)
        )
    return (
        {
            "horizon": horizon,
            "objective": mode,
            "profiles": {profile: _summarize(value) for profile, value in records.items()},
            "comparisons": comparisons,
            "oracle": _oracle(action_rows, horizon, mode),
        },
        records,
        models,
        pair_diagnostics,
    )


def _score_objective_with_models(
    action_rows: list[dict[str, Any]],
    realized_rows: list[dict[str, Any]],
    folds: list[dict[str, Any]],
    models: dict[str, list[PairwiseModel]],
    horizon: int,
    mode: str,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    records: dict[str, dict[str, dict[str, Any]]] = {}
    for profile in FEATURE_PROFILES:
        source = action_rows if profile in ACTION_PROFILES else realized_rows
        labeled = relabel(source, horizon, mode)
        profile_records: dict[str, dict[str, Any]] = {}
        for fold, model in zip(folds, models[profile]):
            validation = [
                row
                for row in labeled
                if row["state_id"] in fold["validation_states"]
            ]
            profile_records.update(_evaluate_pairwise(validation, model))
        records[profile] = profile_records
    comparisons = {}
    for baseline, improved in (
        ("dynamic_action", "local_pre"),
        ("local_pre", "local_pre_context"),
        ("local_pre", "realized"),
        ("realized", "realized_context"),
    ):
        key = f"{improved}_vs_{baseline}"
        comparisons[key] = _comparison(records[baseline], records[improved])
        comparisons[key].update(
            _map_bootstrap(records[baseline], records[improved], bootstrap_samples)
        )
    return (
        {
            "horizon": horizon,
            "objective": mode,
            "policy_training_label": "Horizon 1 effectiveness Pareto",
            "profiles": {
                profile: _summarize(value) for profile, value in records.items()
            },
            "comparisons": comparisons,
            "oracle": _oracle(action_rows, horizon, mode),
        },
        records,
    )


def _gate_report(
    main: dict[str, Any],
    horizon4: dict[str, Any],
    permutation: dict[str, Any],
) -> dict[str, Any]:
    comparisons = main["comparisons"]
    observable = comparisons["realized_vs_local_pre"]
    local = comparisons["local_pre_vs_dynamic_action"]
    context = comparisons["local_pre_context_vs_local_pre"]
    oracle = main["oracle"]
    context_profile = main["profiles"]["local_pre_context"]
    gates = {
        "observability": {
            "actual": observable,
            "requirement": "top-1 gain >= 0.05, AUC regret reduction >= 0.05, bootstrap upper bounds >= 0",
            "passed": (
                observable["pareto_top1_gain"] >= 0.05
                and observable["relative_auc_regret_reduction"] >= 0.05
                and observable["hit_gain_95_ci"][1] >= 0
                and observable["auc_improvement_95_ci"][1] >= 0
            ),
        },
        "local_representation": {
            "actual": local,
            "requirement": "top-1 gain >= 0.03 or AUC reduction >= 0.05; other metric cannot materially regress",
            "passed": (
                (local["pareto_top1_gain"] >= 0.03 and local["relative_auc_regret_reduction"] >= -0.05)
                or (local["relative_auc_regret_reduction"] >= 0.05 and local["pareto_top1_gain"] >= -0.03)
            ),
        },
        "static_context": {
            "actual": {"comparison": context, "permutation": permutation},
            "requirement": "top-1 gain and AUC reduction >= 0.05; both permutation percentiles >= 0.95",
            "passed": (
                context["pareto_top1_gain"] >= 0.05
                and context["relative_auc_regret_reduction"] >= 0.05
                and permutation["hit_gain_percentile"] >= 0.95
                and permutation["auc_reduction_percentile"] >= 0.95
            ),
        },
        "no_size_collapse": {
            "actual": {
                "oracle_supported_sizes": oracle["supported_sizes"],
                "local_pre_context_maximum_size_share": context_profile["maximum_size_share"],
            },
            "requirement": "maximum selected-size share <= 0.90 when oracle supports multiple sizes",
            "passed": not (
                oracle["multiple_sizes_supported"]
                and context_profile["maximum_size_share"] > 0.90
            ),
        },
    }
    h4_local = horizon4["comparisons"]["local_pre_vs_dynamic_action"]
    h4_local_pass = (
        (h4_local["pareto_top1_gain"] >= 0.03 and h4_local["relative_auc_regret_reduction"] >= -0.05)
        or (h4_local["relative_auc_regret_reduction"] >= 0.05 and h4_local["pareto_top1_gain"] >= -0.03)
    )
    local_pass = gates["local_representation"]["passed"]
    context_pass = gates["static_context"]["passed"]
    observed_pass = gates["observability"]["passed"]
    if local_pass and not h4_local_pass:
        decision = "increase_trials_for_long_horizon_variance"
    elif observed_pass and not local_pass:
        decision = "rank_realized_neighborhoods"
    elif local_pass and context_pass:
        decision = "retain_transfer_high_level_policy_and_collect_confirmation"
    elif local_pass:
        decision = "retain_dynamic_high_level_policy_and_shrink_static_transfer_claim"
    else:
        decision = "run_movingai_mechanism_probe_before_more_collection"
    return {
        "passed": all(value["passed"] for value in gates.values()),
        "gates": gates,
        "horizon1_local_passed": local_pass,
        "horizon4_local_passed": h4_local_pass,
        "decision": decision,
    }


def _profile_feature_diagnostics(
    rows: list[dict[str, Any]], profile: str
) -> dict[str, Any]:
    names = sorted({name for row in rows for name in row["features"][profile]})
    first_values: dict[str, float] = {}
    constant = {name: True for name in names}
    digests = {name: hashlib.sha256() for name in names}
    for row in rows:
        features = row["features"][profile]
        for name in names:
            value = float(features.get(name, 0.0))
            if name not in first_values:
                first_values[name] = value
            elif value != first_values[name]:
                constant[name] = False
            digests[name].update(struct.pack("<d", value))
    by_digest: dict[str, list[str]] = collections.defaultdict(list)
    for name, digest in digests.items():
        by_digest[digest.hexdigest()].append(name)
    return {
        "feature_count": len(names),
        "constant_features": [
            {"name": name, "value": first_values[name]}
            for name in names
            if constant[name]
        ],
        "duplicate_feature_groups": [
            group for group in sorted(by_digest.values()) if len(group) > 1
        ],
    }


def _index_storage_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stored_values = 0
    materialized_values = 0
    for row in rows:
        merged: dict[str, float] = {}
        for features in row["features"].values():
            stored_values += len(features)
            for name, raw_value in features.items():
                value = float(raw_value)
                if name in merged and merged[name] != value:
                    raise ValueError(
                        f"feature {name} has inconsistent values across profiles"
                    )
                merged[name] = value
        materialized_values += len(merged)
    redundant_values = stored_values - materialized_values
    return {
        "row_count": len(rows),
        "stored_feature_values": stored_values,
        "unique_materialized_feature_values": materialized_values,
        "redundant_feature_values": redundant_values,
        "redundant_fraction": _ratio(redundant_values, stored_values),
    }


def feature_diagnostics(
    action_rows: list[dict[str, Any]], realized_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    profiles = {}
    for profile in FEATURE_PROFILES:
        source = action_rows if profile in ACTION_PROFILES else realized_rows
        profiles[profile] = _profile_feature_diagnostics(source, profile)
    return {
        "profiles": profiles,
        "index_storage": {
            "action_index": _index_storage_diagnostics(action_rows),
            "realized_index": _index_storage_diagnostics(realized_rows),
        },
        "compact_index_v2_deferred": True,
    }


def scientific_result_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Return report fields whose values define the registered experiment result."""
    return {
        name: report[name]
        for name in (
            "schema_version",
            "model_seed",
            "index_sha256",
            "integrity",
            "folds",
            "pre_registration",
            "analyses",
            "context_permutation",
            "auxiliary_metric_regressors",
            "acceptance",
        )
    }


def render_markdown(report: dict[str, Any]) -> str:
    main = report["analyses"]["h1_effectiveness"]
    lines = [
        "# InitLNS Local Representation Audit",
        "",
        f"Pre-registered gates: **{'PASS' if report['acceptance']['passed'] else 'FAIL'}**",
        "",
        "## Integrity",
        "",
        f"- Parsed outcomes: {report['integrity']['outcome_rows']}",
        f"- States: {report['integrity']['state_rows']}",
        f"- Missing realized neighborhoods: {report['integrity']['missing_realized_neighborhoods']}",
        f"- Conflict-pair mismatches: {report['integrity']['conflict_pair_mismatch_states']}",
        "",
        "## Horizon 1 effectiveness",
        "",
        "| Profile | Pareto top-1 | AUC regret | Conflict regret | Max size share |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for profile in FEATURE_PROFILES:
        value = main["profiles"][profile]
        lines.append(
            f"| {profile} | {value['pareto_top1_hit_rate']:.4f} | "
            f"{value['mean_auc_regret']:.4f} | {value['mean_conflict_regret']:.4f} | "
            f"{value['maximum_size_share']:.4f} |"
        )
    lines.extend(["", "## Gates", ""])
    for name, gate in report["acceptance"]["gates"].items():
        lines.append(f"- {name}: **{'PASS' if gate['passed'] else 'FAIL'}**")
    lines.extend(["", "## Feature diagnostics", ""])
    for profile in FEATURE_PROFILES:
        value = report["feature_diagnostics"]["profiles"][profile]
        lines.append(
            f"- {profile}: {value['feature_count']} features, "
            f"{len(value['constant_features'])} constant, "
            f"{len(value['duplicate_feature_groups'])} duplicate groups"
        )
    lines.extend(["", "## Stage timings", ""])
    for name, value in report["timings_seconds"].items():
        lines.append(f"- {name}: {value:.3f} s")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"`{report['acceptance']['decision']}`",
            "",
            "The effectiveness label excludes generated nodes and runtime. Horizon 4, "
            "compute-aware, and runtime-sensitive results are sensitivity analyses only.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_local_representation_audit(
    collection: str | Path,
    output: str | Path,
    dataset: str | Path | None = None,
    expected_outcomes: int | None = 7344,
    bootstrap_samples: int = 2000,
    permutations: int = 500,
) -> dict[str, Any]:
    if bootstrap_samples <= 0 or permutations <= 0:
        raise ValueError("bootstrap samples and permutations must be positive")
    total_started = time.perf_counter()
    timings: dict[str, float] = {}
    output_root = Path(output).resolve()
    collection_root = Path(collection).resolve()
    resolved_dataset = _dataset_root(collection_root, dataset)
    stage_started = time.perf_counter()
    action_rows, realized_rows, integrity = build_local_indexes(
        collection_root, resolved_dataset, expected_outcomes
    )
    timings["index_build"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    diagnostics = feature_diagnostics(action_rows, realized_rows)
    timings["feature_diagnostics"] = time.perf_counter() - stage_started
    folds = _map_folds(action_rows)
    stage_started = time.perf_counter()
    main_analysis, main_records, main_models, pair_diagnostics = _run_objective(
        action_rows,
        realized_rows,
        folds,
        1,
        "effectiveness",
        bootstrap_samples,
    )
    timings["main_pairwise_training_and_evaluation"] = (
        time.perf_counter() - stage_started
    )
    stage_started = time.perf_counter()
    models_root = output_root / "models"
    for profile, profile_models in main_models.items():
        for fold, model in enumerate(profile_models):
            path = models_root / f"pairwise__{profile}__fold_{fold}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as stream:
                pickle.dump(model, stream)
    timings["model_artifact_write"] = time.perf_counter() - stage_started
    analyses: dict[str, Any] = {"h1_effectiveness": main_analysis}
    stage_started = time.perf_counter()
    for horizon, mode, name in (
        (4, "effectiveness", "h4_effectiveness"),
        (1, "compute_aware", "h1_compute_aware"),
        (1, "runtime_sensitivity", "h1_runtime_sensitivity"),
    ):
        analysis, _ = _score_objective_with_models(
            action_rows,
            realized_rows,
            folds,
            main_models,
            horizon,
            mode,
            bootstrap_samples,
        )
        analyses[name] = analysis
    timings["sensitivity_scoring"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    permutation = _permutation_test(
        action_rows,
        folds,
        main_records["local_pre"],
        main_records["local_pre_context"],
        main_models["local_pre_context"],
        permutations,
    )
    timings["context_permutation"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    auxiliary = _regression_diagnostics(action_rows, realized_rows, folds)
    timings["auxiliary_regression"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    acceptance = _gate_report(
        analyses["h1_effectiveness"], analyses["h4_effectiveness"], permutation
    )
    digest = hashlib.sha256(
        "\n".join(
            f"{row['state_id']}|{row['candidate_key']}|{row['trial_count']}"
            for row in action_rows
        ).encode("utf-8")
    ).hexdigest()
    timings["report_assembly"] = time.perf_counter() - stage_started
    report = {
        "schema_version": LOCAL_AUDIT_SCHEMA_VERSION,
        "model_seed": MODEL_SEED,
        "collection": str(collection_root),
        "dataset": str(resolved_dataset) if resolved_dataset else None,
        "index_sha256": digest,
        "integrity": integrity,
        "folds": [
            {
                key: sorted(value) if isinstance(value, set) else value
                for key, value in fold.items()
            }
            for fold in folds
        ],
        "pre_registration": {
            "feature_profiles": list(FEATURE_PROFILES),
            "primary_label": "Horizon 1 effectiveness Pareto",
            "effectiveness_objectives": ["solved_rate", "conflicts_after", "conflict_auc"],
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_unit": "map_id",
            "permutations": permutations,
            "permutation_unit": "task_id",
            "learner": "fixed pairwise HistGradientBoostingClassifier",
        },
        "analyses": analyses,
        "context_permutation": permutation,
        "auxiliary_metric_regressors": auxiliary,
        "pairwise_training": pair_diagnostics,
        "feature_diagnostics": diagnostics,
        "acceptance": acceptance,
        "timings_seconds": timings,
    }
    stage_started = time.perf_counter()
    _write_jsonl(output_root / "action_index.jsonl", action_rows)
    _write_jsonl(output_root / "realized_index.jsonl", realized_rows)
    timings["index_artifact_write"] = time.perf_counter() - stage_started
    report["timings_seconds"] = timings
    stage_started = time.perf_counter()
    _write_json(output_root / "local_representation_audit.json", report)
    (output_root / "local_representation_audit.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    timings["report_artifact_write"] = time.perf_counter() - stage_started
    timings["total"] = time.perf_counter() - total_started
    report["timings_seconds"] = timings
    _write_json(output_root / "local_representation_audit.json", report)
    (output_root / "local_representation_audit.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    return report
