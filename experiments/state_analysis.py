from __future__ import annotations

import collections
import itertools
from dataclasses import dataclass
from typing import Any

from experiments._common import ratio as _ratio


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
