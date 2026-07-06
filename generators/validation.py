from __future__ import annotations

import collections

from .models import Cell, MapData, TaskData

_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
