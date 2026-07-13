from __future__ import annotations

import collections

from .models import Cell, MapData, TaskData

_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _cells(values: list[list[int]]) -> set[Cell]:
    return {(int(row), int(col)) for row, col in values}


def _distance_map(map_data: MapData, start: Cell) -> dict[Cell, int]:
    distances = {start: 0}
    queue: collections.deque[Cell] = collections.deque([start])
    while queue:
        row, col = queue.popleft()
        for dr, dc in _DIRECTIONS:
            neighbor = (row + dr, col + dc)
            if map_data.traversable(neighbor) and neighbor not in distances:
                distances[neighbor] = distances[(row, col)] + 1
                queue.append(neighbor)
    return distances


def validate_map(map_data: MapData) -> None:
    if map_data.rows == 0 or map_data.cols == 0:
        raise ValueError("map is empty")
    if any(len(row) != map_data.cols for row in map_data.grid):
        raise ValueError("map rows have inconsistent lengths")
    if any(character not in ".@" for row in map_data.grid for character in row):
        raise ValueError("map grid contains unsupported characters")

    free = map_data.free_cells()
    if not free:
        raise ValueError("map has no traversable cells")
    visited = {free[0]}
    open_cells: collections.deque[Cell] = collections.deque([free[0]])
    while open_cells:
        row, col = open_cells.popleft()
        for dr, dc in _DIRECTIONS:
            neighbor = (row + dr, col + dc)
            if map_data.traversable(neighbor) and neighbor not in visited:
                visited.add(neighbor)
                open_cells.append(neighbor)
    if len(visited) != len(free):
        raise ValueError("traversable map cells are not fully connected")

    for key in ("service_cells", "stations", "semantic_cell_types"):
        if key not in map_data.metadata:
            raise ValueError(f"map metadata is missing {key}")
    semantic = map_data.metadata["semantic_cell_types"]
    if len(semantic) != map_data.rows or any(
        len(row) != map_data.cols for row in semantic
    ):
        raise ValueError("semantic layer dimensions do not match map")
    if int(map_data.metadata.get("schema_version", 1)) >= 2:
        zones = map_data.metadata.get("zones", {})
        for zone in (
            "left_storage",
            "center_storage",
            "right_storage",
            "top_storage",
            "bottom_storage",
        ):
            if not zones.get(zone):
                raise ValueError(f"map metadata is missing non-empty {zone}")


def validate_task(map_data: MapData, task_data: TaskData) -> None:
    if task_data.map_id != map_data.map_id:
        raise ValueError("task references a different map")
    if not task_data.starts or len(task_data.starts) != len(task_data.goals):
        raise ValueError("task has invalid agent endpoint counts")
    if len(set(task_data.starts)) != len(task_data.starts):
        raise ValueError("agent starts are not unique")
    if len(set(task_data.goals)) != len(task_data.goals):
        raise ValueError("agent goals are not unique")
    for index, (start, goal) in enumerate(
        zip(task_data.starts, task_data.goals)
    ):
        if not map_data.traversable(start) or not map_data.traversable(goal):
            raise ValueError(f"agent {index} endpoint is not traversable")
        if start == goal:
            raise ValueError(f"agent {index} has identical start and goal")

    metadata = task_data.metadata
    if int(metadata.get("task_semantics_version", 1)) < 2:
        return
    if int(metadata.get("schema_version", -1)) != 2:
        raise ValueError("task semantics v2 requires task schema_version 2")
    assignments = list(metadata.get("flow_assignments", []))
    if len(assignments) != task_data.agent_count:
        raise ValueError("flow_assignments length does not match agent count")

    quota_counts = metadata.get("od_quota_counts")
    if quota_counts is not None and dict(collections.Counter(assignments)) != {
        str(name): int(count) for name, count in dict(quota_counts).items()
    }:
        raise ValueError("realized flow assignments do not match OD quotas")

    scenario = metadata.get("scenario_type")
    zones = {
        name.removesuffix("_storage"): _cells(values)
        for name, values in map_data.metadata["zones"].items()
        if name.endswith("_storage")
    }
    exact_scenarios = {
        "cross_zone_exchange": {
            "left_to_center",
            "center_to_left",
            "center_to_right",
            "right_to_center",
            "left_to_right",
            "right_to_left",
        },
        "intersection_crossing": {
            "left_to_right",
            "right_to_left",
            "top_to_bottom",
            "bottom_to_top",
        },
    }
    if scenario in exact_scenarios and quota_counts is not None:
        allowed = exact_scenarios[str(scenario)]
        for index, (start, goal, assignment) in enumerate(
            zip(task_data.starts, task_data.goals, assignments)
        ):
            if assignment not in allowed:
                raise ValueError(
                    f"agent {index} has invalid exact OD assignment {assignment}"
                )
            origin, destination = assignment.split("_to_", 1)
            if start not in zones[origin] or goal not in zones[destination]:
                raise ValueError(
                    f"agent {index} endpoints violate {assignment}"
                )

    if scenario != "intersection_crossing":
        return
    required_values = list(metadata.get("required_intersections", []))
    component_ids = list(
        metadata.get("required_intersection_component_ids", [])
    )
    if (
        len(required_values) != task_data.agent_count
        or len(component_ids) != task_data.agent_count
    ):
        raise ValueError("intersection requirement arrays have invalid lengths")
    selected = {
        str(component["id"]): _cells(component["cells"])
        for component in metadata.get("selected_intersection_components", [])
    }
    expected_count = round(
        float(metadata["required_intersection_crossing_ratio"])
        * task_data.agent_count
    )
    actual_count = sum(value is not None for value in required_values)
    if actual_count != expected_count:
        raise ValueError("realized intersection count does not match request")
    if round(actual_count / task_data.agent_count, 6) != float(
        metadata["realized_intersection_crossing_ratio"]
    ):
        raise ValueError("realized intersection ratio is inconsistent")

    start_distances: dict[Cell, dict[Cell, int]] = {}
    intersection_distances: dict[Cell, dict[Cell, int]] = {}
    semantic = map_data.metadata["semantic_cell_types"]
    for index, (start, goal, value, component_id) in enumerate(
        zip(
            task_data.starts,
            task_data.goals,
            required_values,
            component_ids,
        )
    ):
        if value is None:
            if component_id is not None:
                raise ValueError(
                    f"agent {index} has a component without an intersection"
                )
            continue
        intersection = (int(value[0]), int(value[1]))
        if component_id not in selected or intersection not in selected[component_id]:
            raise ValueError(
                f"agent {index} references an unselected intersection"
            )
        if semantic[intersection[0]][intersection[1]] != "X":
            raise ValueError(
                f"agent {index} required intersection is not semantic X"
            )
        if start not in start_distances:
            start_distances[start] = _distance_map(map_data, start)
        if intersection not in intersection_distances:
            intersection_distances[intersection] = _distance_map(
                map_data, intersection
            )
        shortest = start_distances[start].get(goal)
        through = start_distances[start].get(intersection)
        remaining = intersection_distances[intersection].get(goal)
        if (
            shortest is None
            or through is None
            or remaining is None
            or through + remaining != shortest
        ):
            raise ValueError(
                f"agent {index} has no shortest path through its intersection"
            )
