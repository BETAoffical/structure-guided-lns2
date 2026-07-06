from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Cell = tuple[int, int]


@dataclass
class MapData:
    map_id: str
    seed: int
    grid: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def traversable(self, cell: Cell) -> bool:
        row, col = cell
        return (
            0 <= row < self.rows
            and 0 <= col < self.cols
            and self.grid[row][col] == "."
        )

    def free_cells(self) -> list[Cell]:
        return [
            (row, col)
            for row in range(self.rows)
            for col in range(self.cols)
            if self.grid[row][col] == "."
        ]


@dataclass
class TaskData:
    task_id: str
    map_id: str
    seed: int
    starts: list[Cell]
    goals: list[Cell]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_count(self) -> int:
        return len(self.starts)
