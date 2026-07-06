from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import MapData, TaskData
from .visualization import ascii_preview, svg_preview


def map_document(map_data: MapData) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "map_id": map_data.map_id,
        "seed": map_data.seed,
        "rows": map_data.rows,
        "cols": map_data.cols,
        "grid": map_data.grid,
        "metadata": map_data.metadata,
    }


def task_document(task_data: TaskData) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task_id": task_data.task_id,
        "map_id": task_data.map_id,
        "seed": task_data.seed,
        "starts": [list(cell) for cell in task_data.starts],
        "goals": [list(cell) for cell in task_data.goals],
        "metadata": task_data.metadata,
    }


def write_json(path: str | Path, document: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_mapf(
    path: str | Path, map_data: MapData, task_data: TaskData
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{map_data.rows} {map_data.cols}\n")
        for row in map_data.grid:
            stream.write(row + "\n")
        stream.write(f"{task_data.agent_count}\n")
        for start, goal in zip(task_data.starts, task_data.goals):
            stream.write(
                f"{start[0]} {start[1]} {goal[0]} {goal[1]}\n"
            )


def write_map_bundle(directory: str | Path, map_data: MapData) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / f"{map_data.map_id}.json", map_document(map_data))
    (destination / f"{map_data.map_id}.txt").write_text(
        ascii_preview(map_data), encoding="utf-8"
    )
    (destination / f"{map_data.map_id}.svg").write_text(
        svg_preview(map_data), encoding="utf-8"
    )


def write_instance_bundle(
    directory: str | Path, map_data: MapData, task_data: TaskData
) -> None:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    name = task_data.task_id
    write_mapf(destination / f"{name}.mapf", map_data, task_data)
    write_json(destination / f"{name}.json", task_document(task_data))
    (destination / f"{name}.txt").write_text(
        ascii_preview(map_data, task_data), encoding="utf-8"
    )
    (destination / f"{name}.svg").write_text(
        svg_preview(map_data, task_data), encoding="utf-8"
    )
