from __future__ import annotations

import collections
import random
import sys
from typing import Any

from .config import (
    sample_choice,
    sample_float,
    sample_int,
    weighted_choice,
)
from .models import Cell, MapData

_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _cells_to_json(cells: list[Cell] | set[Cell]) -> list[list[int]]:
    return [[row, col] for row, col in sorted(cells)]


def _band_starts(
    limit: int, margin: int, block_size: int, aisle_width: int
) -> list[int]:
    available = limit - 2 * margin
    if available < block_size:
        return []
    count = 1 + (available - block_size) // (block_size + aisle_width)
    used = count * block_size + (count - 1) * aisle_width
    offset = max(0, (available - used) // 2)
    return [
        margin + offset + index * (block_size + aisle_width)
        for index in range(count)
    ]


def _int_bounds(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        low, high = value
        if low <= high:
            return low, high
    raise ValueError(f"{name} must be an integer or [minimum, maximum]")


def _variable_band_starts(
    limit: int,
    margin: int,
    block_size: int,
    aisle_width_range: Any,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    minimum, maximum = _int_bounds(
        aisle_width_range, "variable_vertical_aisle_width"
    )
    if minimum <= 0:
        raise ValueError("variable aisle widths must be positive")
    widths = list(range(minimum, maximum + 1))
    rng.shuffle(widths)
    relative_starts: list[int] = []
    sampled_gaps: list[int] = []
    cursor = 0
    gap_index = 0
    available = limit - 2 * margin
    while cursor + block_size <= available:
        relative_starts.append(cursor)
        gap = widths[gap_index % len(widths)]
        gap_index += 1
        next_cursor = cursor + block_size + gap
        if next_cursor + block_size <= available:
            sampled_gaps.append(gap)
        else:
            break
        cursor = next_cursor
    if not relative_starts:
        return [], []
    used = relative_starts[-1] + block_size
    offset = max(0, (available - used) // 2)
    starts = [margin + offset + start for start in relative_starts]
    return starts, sampled_gaps


def _free_neighbors(grid: list[list[str]], cell: Cell) -> list[Cell]:
    rows = len(grid)
    cols = len(grid[0])
    row, col = cell
    result = []
    for dr, dc in _DIRECTIONS:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == ".":
            result.append((nr, nc))
    return result


def _is_connected(grid: list[list[str]]) -> bool:
    free = [
        (row, col)
        for row in range(len(grid))
        for col in range(len(grid[0]))
        if grid[row][col] == "."
    ]
    if not free:
        return False
    visited = {free[0]}
    open_cells: collections.deque[Cell] = collections.deque([free[0]])
    while open_cells:
        cell = open_cells.popleft()
        for neighbor in _free_neighbors(grid, cell):
            if neighbor not in visited:
                visited.add(neighbor)
                open_cells.append(neighbor)
    return len(visited) == len(free)


def _try_block_cells(
    grid: list[list[str]],
    obstacle_types: list[list[str]],
    cells: list[Cell],
    obstacle_type: str,
) -> bool:
    changed = [
        cell for cell in cells if grid[cell[0]][cell[1]] == "."
    ]
    for row, col in changed:
        grid[row][col] = "@"
        obstacle_types[row][col] = obstacle_type
    if changed and _is_connected(grid):
        return True
    for row, col in changed:
        grid[row][col] = "."
        obstacle_types[row][col] = "."
    return False


def _sample_gap_coordinates(
    gaps: list[tuple[int, int]],
    count: int,
    rng: random.Random,
) -> list[int]:
    if count < 0 or count > len(gaps):
        return []
    selected = rng.sample(gaps, count)
    return [rng.randrange(start, end) for start, end in selected]


def _sample_gate_coordinates(
    gaps: list[tuple[int, int]],
    count: int,
    forbidden: set[int],
    rng: random.Random,
) -> list[int]:
    candidates = [
        [value for value in range(start, end) if value not in forbidden]
        for start, end in gaps
    ]
    candidates = [values for values in candidates if values]
    if count > len(candidates):
        return []
    selected = rng.sample(candidates, count)
    return [rng.choice(values) for values in selected]


def _sample_layout_mode(config: dict[str, Any], rng: random.Random) -> str:
    mode = config.get("layout_mode", "regular_beltway")
    if mode != "mixed":
        return sample_choice(mode, rng, "layout_mode")
    return weighted_choice(
        dict(
            config.get(
                "layout_mixture",
                {
                    "regular_beltway": 1.0,
                    "partial_beltway": 1.0,
                    "wall_shelves": 1.0,
                    "dead_end_aisles": 1.0,
                    "partial_cross_aisles": 1.0,
                    "compartmentalized": 1.0,
                    "mixed_width": 1.0,
                    "asymmetric": 1.0,
                    "station_centric": 1.0,
                },
            )
        ),
        rng,
        "layout_mixture",
    )


def _weighted_station_values(
    count: int, distribution: str
) -> list[float]:
    if count <= 0:
        return []
    if distribution == "uniform":
        return [round(1.0 / count, 6)] * count
    if distribution == "zipf":
        raw = [1.0 / (index + 1) for index in range(count)]
        total = sum(raw)
        return [round(value / total, 6) for value in raw]
    raise ValueError(
        "station_demand_distribution must be 'uniform' or 'zipf'"
    )


def _topology_metrics(grid: list[list[str]]) -> dict[str, float | int]:
    degrees = [
        len(_free_neighbors(grid, (row, col)))
        for row in range(len(grid))
        for col in range(len(grid[0]))
        if grid[row][col] == "."
    ]
    dead_ends = sum(degree == 1 for degree in degrees)
    return {
        "dead_end_cell_count": dead_ends,
        "average_free_degree": round(sum(degrees) / len(degrees), 6),
        "minimum_free_degree": min(degrees),
        "maximum_free_degree": max(degrees),
    }


def _articulation_cells(grid: list[list[str]]) -> set[Cell]:
    sys.setrecursionlimit(max(10_000, len(grid) * len(grid[0]) * 2))
    discovery: dict[Cell, int] = {}
    low: dict[Cell, int] = {}
    parent: dict[Cell, Cell | None] = {}
    articulation: set[Cell] = set()
    clock = 0

    def visit(cell: Cell) -> None:
        nonlocal clock
        clock += 1
        discovery[cell] = low[cell] = clock
        children = 0
        for neighbor in _free_neighbors(grid, cell):
            if neighbor not in discovery:
                parent[neighbor] = cell
                children += 1
                visit(neighbor)
                low[cell] = min(low[cell], low[neighbor])
                if parent[cell] is None and children > 1:
                    articulation.add(cell)
                if (
                    parent[cell] is not None
                    and low[neighbor] >= discovery[cell]
                ):
                    articulation.add(cell)
            elif neighbor != parent[cell]:
                low[cell] = min(low[cell], discovery[neighbor])

    for row in range(len(grid)):
        for col in range(len(grid[0])):
            cell = (row, col)
            if grid[row][col] == "." and cell not in discovery:
                parent[cell] = None
                visit(cell)
    return articulation


def _obstacle_clearance(grid: list[list[str]]) -> list[list[int]]:
    rows = len(grid)
    cols = len(grid[0])
    distance = [[rows + cols for _ in range(cols)] for _ in range(rows)]
    open_cells: collections.deque[Cell] = collections.deque()
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == "@":
                distance[row][col] = 0
                open_cells.append((row, col))
    if not open_cells:
        return [[rows + cols for _ in range(cols)] for _ in range(rows)]
    while open_cells:
        row, col = open_cells.popleft()
        for dr, dc in _DIRECTIONS:
            nr, nc = row + dr, col + dc
            if (
                0 <= nr < rows
                and 0 <= nc < cols
                and distance[nr][nc] > distance[row][col] + 1
            ):
                distance[nr][nc] = distance[row][col] + 1
                open_cells.append((nr, nc))
    return distance


def _station_candidates(
    rows: int, cols: int, sides: list[str], beltway: int
) -> list[Cell]:
    offset = max(0, beltway // 2)
    candidates: list[Cell] = []
    for side in sides:
        if side == "left":
            candidates.extend((row, offset) for row in range(1, rows - 1))
        elif side == "right":
            candidates.extend(
                (row, cols - 1 - offset) for row in range(1, rows - 1)
            )
        elif side == "top":
            candidates.extend((offset, col) for col in range(1, cols - 1))
        elif side == "bottom":
            candidates.extend(
                (rows - 1 - offset, col) for col in range(1, cols - 1)
            )
        else:
            raise ValueError(f"unsupported station side: {side}")
    return list(dict.fromkeys(candidates))


def _generate_warehouse_once(
    config: dict[str, Any], seed: int, map_id: str
) -> MapData:
    rng = random.Random(seed)
    layout_mode = _sample_layout_mode(config, rng)
    rows = sample_int(config.get("rows", 40), rng, "rows")
    cols = sample_int(config.get("cols", 60), rng, "cols")
    beltway = sample_int(
        config.get("outer_beltway_width", [1, 3]),
        rng,
        "outer_beltway_width",
    )
    beltway_mode = sample_choice(
        config.get(
            "beltway_mode",
            "partial" if layout_mode == "partial_beltway" else "full",
        ),
        rng,
        "beltway_mode",
    )
    if beltway_mode == "none":
        beltway = 0
    wall_clearance = sample_int(
        config.get(
            "wall_clearance",
            0 if layout_mode == "wall_shelves" else 1,
        ),
        rng,
        "wall_clearance",
    )
    block_height = sample_int(
        config.get("shelf_block_height", [3, 6]),
        rng,
        "shelf_block_height",
    )
    block_width = sample_int(
        config.get("shelf_block_width", [5, 10]),
        rng,
        "shelf_block_width",
    )
    horizontal_aisle = sample_int(
        config.get("horizontal_aisle_width", [1, 3]),
        rng,
        "horizontal_aisle_width",
    )
    vertical_aisle = sample_int(
        config.get("vertical_aisle_width", [1, 3]),
        rng,
        "vertical_aisle_width",
    )
    station_count = sample_int(
        config.get("station_count", [2, 6]), rng, "station_count"
    )
    station_clearance = sample_int(
        config.get("station_clearance", 2), rng, "station_clearance"
    )
    layout_jitter = sample_float(
        config.get("layout_jitter", [0.0, 0.15]), rng, "layout_jitter"
    )
    beltway_open_fraction = sample_float(
        config.get("beltway_open_fraction", [0.55, 1.0]),
        rng,
        "beltway_open_fraction",
    )
    dead_end_probability = sample_float(
        config.get(
            "dead_end_aisle_probability",
            [0.15, 0.35] if layout_mode == "dead_end_aisles" else 0.0,
        ),
        rng,
        "dead_end_aisle_probability",
    )
    dead_end_depth = sample_int(
        config.get("dead_end_depth", [4, 12]), rng, "dead_end_depth"
    )
    cross_removal_probability = sample_float(
        config.get(
            "cross_aisle_removal_probability",
            [0.1, 0.3] if layout_mode == "partial_cross_aisles" else 0.0,
        ),
        rng,
        "cross_aisle_removal_probability",
    )
    narrow_probability = sample_float(
        config.get(
            "narrow_aisle_probability",
            [0.2, 0.5] if layout_mode == "mixed_width" else 0.0,
        ),
        rng,
        "narrow_aisle_probability",
    )
    narrow_segment_length = sample_int(
        config.get("narrow_segment_length", [3, 12]),
        rng,
        "narrow_segment_length",
    )
    wall_contact_probability = sample_float(
        config.get(
            "shelf_wall_contact_probability",
            [0.1, 0.3] if layout_mode == "wall_shelves" else 0.0,
        ),
        rng,
        "shelf_wall_contact_probability",
    )
    gate_count = sample_int(
        config.get(
            "gate_count",
            [1, 4] if layout_mode == "compartmentalized" else 0,
        ),
        rng,
        "gate_count",
    )
    gate_width = sample_int(
        config.get("gate_width", [1, 3]), rng, "gate_width"
    )
    divider_template = sample_choice(
        config.get("divider_template", "single"),
        rng,
        "divider_template",
    )
    buffer_zone_count = sample_int(
        config.get("buffer_zone_count", [0, 2]),
        rng,
        "buffer_zone_count",
    )
    station_placement_config = config.get(
        "station_centric_placement", "clustered"
    ) if layout_mode == "station_centric" else config.get(
        "station_placement", "distributed"
    )
    station_placement = sample_choice(
        station_placement_config,
        rng,
        "station_placement",
    )
    station_cluster_count = sample_int(
        config.get("station_cluster_count", [1, 2]),
        rng,
        "station_cluster_count",
    )
    station_demand_distribution = sample_choice(
        config.get("station_demand_distribution", "uniform"),
        rng,
        "station_demand_distribution",
    )
    station_entrance_width = sample_int(
        config.get("station_entrance_width", [1, 3]),
        rng,
        "station_entrance_width",
    )
    station_queue_depth = sample_int(
        config.get("station_queue_depth", [1, 5]),
        rng,
        "station_queue_depth",
    )
    station_queue_width = sample_int(
        config.get("station_queue_width", [1, 4]),
        rng,
        "station_queue_width",
    )
    combine_layout_features = bool(
        config.get("combine_layout_features", False)
    )
    beltway_narrow_segment_count = sample_int(
        config.get("beltway_narrow_segment_count", [1, 2]),
        rng,
        "beltway_narrow_segment_count",
    )
    beltway_blocked_segment_count = sample_int(
        config.get("beltway_blocked_segment_count", [0, 1]),
        rng,
        "beltway_blocked_segment_count",
    )
    beltway_segment_length = sample_int(
        config.get("beltway_segment_length", [6, 12]),
        rng,
        "beltway_segment_length",
    )
    beltway_narrow_width = sample_int(
        config.get("beltway_narrow_width", 1),
        rng,
        "beltway_narrow_width",
    )
    wall_shelf_extension_count = sample_int(
        config.get("wall_shelf_extension_count", [1, 2]),
        rng,
        "wall_shelf_extension_count",
    )
    dead_end_aisle_count = sample_int(
        config.get("dead_end_aisle_count", [1, 3]),
        rng,
        "dead_end_aisle_count",
    )
    orientation_counts_config = dict(
        config.get("dead_end_orientation_counts", {})
    )
    dead_end_vertical_count = sample_int(
        orientation_counts_config.get(
            "vertical", dead_end_aisle_count
        ),
        rng,
        "dead_end_orientation_counts.vertical",
    )
    dead_end_horizontal_count = sample_int(
        orientation_counts_config.get("horizontal", 0),
        rng,
        "dead_end_orientation_counts.horizontal",
    )
    if orientation_counts_config:
        dead_end_aisle_count = (
            dead_end_vertical_count + dead_end_horizontal_count
        )
    cross_aisle_closure_count = sample_int(
        config.get("cross_aisle_closure_count", [1, 2]),
        rng,
        "cross_aisle_closure_count",
    )
    cross_aisle_span_blocks = sample_int(
        config.get("cross_aisle_span_blocks", [1, 2]),
        rng,
        "cross_aisle_span_blocks",
    )
    divider_orientation = sample_choice(
        config.get("divider_orientation", ["vertical", "horizontal"]),
        rng,
        "divider_orientation",
    )
    divider_thickness = sample_int(
        config.get("divider_thickness", 1),
        rng,
        "divider_thickness",
    )
    asymmetric_trim_width = sample_int(
        config.get("asymmetric_trim_width", [1, 2]),
        rng,
        "asymmetric_trim_width",
    )
    if min(rows, cols) < 12:
        raise ValueError("warehouse dimensions must both be at least 12")
    if beltway < 0 or wall_clearance < 0:
        raise ValueError("beltway and wall clearance cannot be negative")
    if beltway_mode not in {"full", "partial", "none"}:
        raise ValueError("beltway_mode must be full, partial, or none")
    if not 0.0 <= beltway_open_fraction <= 1.0:
        raise ValueError("beltway_open_fraction must be between 0 and 1")
    if beltway_narrow_width < 1:
        raise ValueError("beltway_narrow_width must be positive")
    if divider_orientation not in {"vertical", "horizontal"}:
        raise ValueError(
            "divider_orientation must be vertical or horizontal"
        )
    if divider_thickness < 1:
        raise ValueError("divider_thickness must be positive")
    if divider_template not in {
        "single",
        "cross_four_gate",
        "double_horizontal",
        "double_vertical",
    }:
        raise ValueError("unsupported divider_template")
    if divider_template != "single" and (
        gate_count != 2 or gate_width != 1 or divider_thickness != 1
    ):
        raise ValueError(
            "combined divider templates require two single-cell gates "
            "and one-cell wall thickness"
        )

    margin = beltway + wall_clearance
    row_starts = _band_starts(
        rows, margin, block_height, horizontal_aisle
    )
    sampled_vertical_aisle_widths: list[int] = []
    if layout_mode == "mixed_width":
        variable_width = config.get(
            "variable_vertical_aisle_width",
            [1, max(3, vertical_aisle)],
        )
        col_starts, sampled_vertical_aisle_widths = (
            _variable_band_starts(
                cols, margin, block_width, variable_width, rng
            )
        )
    else:
        col_starts = _band_starts(
            cols, margin, block_width, vertical_aisle
        )
        sampled_vertical_aisle_widths = [
            col_starts[index + 1]
            - (col_starts[index] + block_width)
            for index in range(len(col_starts) - 1)
        ]
    if not row_starts or not col_starts:
        raise ValueError("map is too small for the sampled warehouse structure")

    grid = [["." for _ in range(cols)] for _ in range(rows)]
    obstacle_types = [["." for _ in range(cols)] for _ in range(rows)]
    shelf_blocks: list[dict[str, Any]] = []
    asymmetric_adjustments: list[dict[str, Any]] = []
    for block_row, top in enumerate(row_starts):
        for block_col, left in enumerate(col_starts):
            trim_bottom = (
                1 if block_height > 2 and rng.random() < layout_jitter else 0
            )
            trim_right = (
                1 if block_width > 2 and rng.random() < layout_jitter else 0
            )
            if layout_mode == "asymmetric" and left >= cols // 2:
                mode_trim = min(
                    asymmetric_trim_width, max(1, block_width - 2)
                )
                trim_right = max(trim_right, mode_trim)
                asymmetric_adjustments.append(
                    {
                        "shelf_id": (
                            f"shelf_{block_row:02d}_{block_col:02d}"
                        ),
                        "side": "right",
                        "trimmed_columns": mode_trim,
                    }
                )
            bottom = top + block_height - trim_bottom
            right = left + block_width - trim_right
            cells: list[Cell] = []
            for row in range(top, bottom):
                for col in range(left, right):
                    grid[row][col] = "@"
                    obstacle_types[row][col] = "S"
                    cells.append((row, col))
            shelf_blocks.append(
                {
                    "id": f"shelf_{block_row:02d}_{block_col:02d}",
                    "bounds": [top, left, bottom - 1, right - 1],
                    "cells": _cells_to_json(cells),
                    "active": True,
                }
            )

    structural_changes: dict[str, list[dict[str, Any]]] = {
        "buffer_zones": [],
        "beltway_closures": [],
        "wall_extensions": [],
        "dead_end_caps": [],
        "cross_aisle_closures": [],
        "narrow_segments": [],
        "compartment_gates": [],
        "asymmetric_shelf_adjustments": asymmetric_adjustments,
        "variable_aisles": [],
    }
    if layout_mode == "mixed_width":
        structural_changes["variable_aisles"] = [
            {
                "between_shelf_columns": [index, index + 1],
                "left_col": col_starts[index] + block_width,
                "right_col": col_starts[index + 1] - 1,
                "width": width,
            }
            for index, width in enumerate(
                sampled_vertical_aisle_widths
            )
        ]

    active_blocks = list(shelf_blocks)
    rng.shuffle(active_blocks)
    for block in active_blocks[: min(buffer_zone_count, len(active_blocks))]:
        removed_cells = [tuple(cell) for cell in block["cells"]]
        for row, col in removed_cells:
            grid[row][col] = "."
            obstacle_types[row][col] = "."
        block["active"] = False
        structural_changes["buffer_zones"].append(
            {"source_shelf": block["id"], "cells": block["cells"]}
        )

    if (
        beltway > 0
        and (
            layout_mode == "partial_beltway"
            or (combine_layout_features and beltway_mode == "partial")
        )
    ):
        configured_sides = list(
            config.get(
                "beltway_affected_sides",
                ["top", "bottom", "left", "right"],
            )
        )
        invalid_sides = set(configured_sides) - {
            "top",
            "bottom",
            "left",
            "right",
        }
        if invalid_sides:
            raise ValueError(
                f"unsupported beltway sides: {sorted(invalid_sides)}"
            )
        if not configured_sides:
            raise ValueError("beltway_affected_sides cannot be empty")
        segment_kinds = (
            ["narrow"] * beltway_narrow_segment_count
            + ["blocked"] * beltway_blocked_segment_count
        )
        rng.shuffle(segment_kinds)
        side_order = configured_sides[:]
        rng.shuffle(side_order)
        for index, kind in enumerate(segment_kinds):
            effective_kind = (
                "blocked" if kind == "narrow" and beltway == 1 else kind
            )
            side = side_order[index % len(side_order)]
            axis_length = cols if side in {"top", "bottom"} else rows
            available = max(1, axis_length - 2 * beltway)
            length = min(beltway_segment_length, available)
            minimum_start = beltway
            maximum_start = max(
                minimum_start, axis_length - beltway - length
            )
            start = rng.randint(minimum_start, maximum_start)
            end = start + length
            open_width = (
                min(beltway, beltway_narrow_width)
                if effective_kind == "narrow"
                else 0
            )
            if side == "top":
                cells = [
                    (row, col)
                    for row in range(0, beltway - open_width)
                    for col in range(start, end)
                ]
            elif side == "bottom":
                cells = [
                    (row, col)
                    for row in range(
                        rows - beltway + open_width, rows
                    )
                    for col in range(start, end)
                ]
            elif side == "left":
                cells = [
                    (row, col)
                    for row in range(start, end)
                    for col in range(0, beltway - open_width)
                ]
            else:
                cells = [
                    (row, col)
                    for row in range(start, end)
                    for col in range(
                        cols - beltway + open_width, cols
                    )
                ]
            if _try_block_cells(grid, obstacle_types, cells, "W"):
                structural_changes["beltway_closures"].append(
                    {
                        "kind": effective_kind,
                        "side": side,
                        "start": start,
                        "length": length,
                        "original_width": beltway,
                        "open_width": open_width,
                        "cells": _cells_to_json(cells),
                    }
                )

    wall_feature_enabled = (
        layout_mode == "wall_shelves" or combine_layout_features
    )
    if wall_feature_enabled:
        boundary_blocks = [
            block
            for block in shelf_blocks
            if block["active"]
            and (
                block["bounds"][0] == min(row_starts)
                or block["bounds"][0] == max(row_starts)
                or block["bounds"][1] == min(col_starts)
                or block["bounds"][1] == max(col_starts)
            )
        ]
        rng.shuffle(boundary_blocks)
        target_extensions = (
            wall_shelf_extension_count
            if layout_mode == "wall_shelves"
            else round(
                wall_contact_probability * len(boundary_blocks)
            )
        )
        for block in boundary_blocks:
            if (
                len(structural_changes["wall_extensions"])
                >= target_extensions
            ):
                break
            top, left, bottom, right = block["bounds"]
            distances = {
                "top": top,
                "bottom": rows - 1 - bottom,
                "left": left,
                "right": cols - 1 - right,
            }
            side = min(distances, key=distances.get)
            cells: list[Cell] = []
            if side == "top":
                cells = [
                    (row, col)
                    for row in range(0, top)
                    for col in range(left, right + 1)
                ]
            elif side == "bottom":
                cells = [
                    (row, col)
                    for row in range(bottom + 1, rows)
                    for col in range(left, right + 1)
                ]
            elif side == "left":
                cells = [
                    (row, col)
                    for row in range(top, bottom + 1)
                    for col in range(0, left)
                ]
            else:
                cells = [
                    (row, col)
                    for row in range(top, bottom + 1)
                    for col in range(right + 1, cols)
                ]
            if _try_block_cells(grid, obstacle_types, cells, "S"):
                structural_changes["wall_extensions"].append(
                    {
                        "source_shelf": block["id"],
                        "side": side,
                        "cells": _cells_to_json(cells),
                    }
                )

    row_gaps = [
        (row_starts[index] + block_height, row_starts[index + 1])
        for index in range(len(row_starts) - 1)
    ]
    col_gaps = [
        (col_starts[index] + block_width, col_starts[index + 1])
        for index in range(len(col_starts) - 1)
    ]

    cross_feature_enabled = (
        layout_mode == "partial_cross_aisles"
        or combine_layout_features
    )
    if cross_feature_enabled and row_gaps and col_starts:
        cross_candidates: list[tuple[int, int, int]] = []
        for gap_index in range(len(row_gaps)):
            for start_block in range(len(col_starts)):
                maximum_span = min(
                    cross_aisle_span_blocks,
                    len(col_starts) - start_block,
                )
                if maximum_span > 0:
                    cross_candidates.append(
                        (gap_index, start_block, maximum_span)
                    )
        rng.shuffle(cross_candidates)
        target_cross_closures = (
            cross_aisle_closure_count
            if layout_mode == "partial_cross_aisles"
            else round(
                cross_removal_probability * len(row_gaps)
            )
        )
        for gap_index, start_block, span in cross_candidates:
            if (
                len(structural_changes["cross_aisle_closures"])
                >= target_cross_closures
            ):
                break
            gap_top, gap_bottom = row_gaps[gap_index]
            final_block = start_block + span - 1
            attached_columns = [
                col
                for block_index in range(
                    start_block, final_block + 1
                )
                for col in range(
                    col_starts[block_index],
                    min(
                        cols,
                        col_starts[block_index] + block_width,
                    ),
                )
                if obstacle_types[gap_top - 1][col] == "S"
                and obstacle_types[gap_bottom][col] == "S"
            ]
            if not attached_columns:
                continue
            cells = [
                (row, col)
                for row in range(gap_top, gap_bottom)
                for col in attached_columns
            ]
            if _try_block_cells(grid, obstacle_types, cells, "W"):
                structural_changes["cross_aisle_closures"].append(
                    {
                        "cross_aisle_index": gap_index,
                        "row_range": [gap_top, gap_bottom - 1],
                        "shelf_block_span": [
                            start_block,
                            final_block,
                        ],
                        "attached_shelf_columns": attached_columns,
                        "cells": _cells_to_json(cells),
                    }
                )

    dead_end_feature_enabled = (
        layout_mode == "dead_end_aisles" or combine_layout_features
    )
    if dead_end_feature_enabled and col_gaps and row_starts:
        vertical_candidates = [
            (gap_index, band_index)
            for gap_index in range(len(col_gaps))
            for band_index in range(len(row_starts))
        ]
        rng.shuffle(vertical_candidates)
        target_vertical = (
            dead_end_vertical_count
            if layout_mode == "dead_end_aisles"
            else round(dead_end_probability * len(col_gaps))
        )
        vertical_added = 0
        for gap_index, band_index in vertical_candidates:
            if (
                vertical_added >= target_vertical
            ):
                break
            gap_left, gap_right = col_gaps[gap_index]
            band_start = row_starts[band_index]
            cap_row = min(
                band_start + block_height - 1,
                band_start + max(1, dead_end_depth // 2),
            )
            left_shelf = (cap_row, gap_left - 1)
            right_shelf = (cap_row, gap_right)
            if (
                obstacle_types[left_shelf[0]][left_shelf[1]] != "S"
                or obstacle_types[right_shelf[0]][right_shelf[1]] != "S"
            ):
                continue
            cells = [
                (cap_row, col) for col in range(gap_left, gap_right)
            ]
            if _try_block_cells(grid, obstacle_types, cells, "W"):
                structural_changes["dead_end_caps"].append(
                    {
                        "orientation": "vertical",
                        "vertical_aisle_index": gap_index,
                        "shelf_band_index": band_index,
                        "depth_target": dead_end_depth,
                        "left_shelf_cell": list(left_shelf),
                        "right_shelf_cell": list(right_shelf),
                        "cells": _cells_to_json(cells),
                    }
                )
                vertical_added += 1

    if (
        dead_end_feature_enabled
        and row_gaps
        and col_starts
        and layout_mode == "dead_end_aisles"
    ):
        horizontal_candidates = [
            (gap_index, band_index)
            for gap_index in range(len(row_gaps))
            for band_index in range(len(col_starts))
        ]
        rng.shuffle(horizontal_candidates)
        horizontal_added = 0
        for gap_index, band_index in horizontal_candidates:
            if horizontal_added >= dead_end_horizontal_count:
                break
            gap_top, gap_bottom = row_gaps[gap_index]
            band_start = col_starts[band_index]
            cap_col = min(
                band_start + block_width - 1,
                band_start + max(1, dead_end_depth // 2),
            )
            top_shelf = (gap_top - 1, cap_col)
            bottom_shelf = (gap_bottom, cap_col)
            if (
                obstacle_types[top_shelf[0]][top_shelf[1]] != "S"
                or obstacle_types[bottom_shelf[0]][bottom_shelf[1]]
                != "S"
            ):
                continue
            cells = [
                (row, cap_col) for row in range(gap_top, gap_bottom)
            ]
            if _try_block_cells(grid, obstacle_types, cells, "W"):
                structural_changes["dead_end_caps"].append(
                    {
                        "orientation": "horizontal",
                        "horizontal_aisle_index": gap_index,
                        "shelf_band_index": band_index,
                        "depth_target": dead_end_depth,
                        "top_shelf_cell": list(top_shelf),
                        "bottom_shelf_cell": list(bottom_shelf),
                        "cells": _cells_to_json(cells),
                    }
                )
                horizontal_added += 1

    if combine_layout_features:
        for gap_left, gap_right in col_gaps:
            if (
                gap_right - gap_left <= 1
                or rng.random() >= narrow_probability
            ):
                continue
            side_col = rng.choice([gap_left, gap_right - 1])
            start_row = rng.randrange(
                max(0, margin), max(margin + 1, rows - margin)
            )
            end_row = min(rows - margin, start_row + narrow_segment_length)
            cells = [
                (row, side_col) for row in range(start_row, end_row)
            ]
            if _try_block_cells(grid, obstacle_types, cells, "N"):
                structural_changes["narrow_segments"].append(
                    {
                        "orientation": "vertical",
                        "cells": _cells_to_json(cells),
                    }
                )

    if layout_mode == "compartmentalized" and gate_count > 0:
        template_counts = {
            "cross_four_gate": (1, 1),
            "double_horizontal": (0, 2),
            "double_vertical": (2, 0),
            "single": (
                (1, 0)
                if divider_orientation == "vertical"
                else (0, 1)
            ),
        }
        vertical_count, horizontal_count = template_counts[
            divider_template
        ]
        vertical_columns = _sample_gap_coordinates(
            col_gaps, vertical_count, rng
        )
        horizontal_rows = _sample_gap_coordinates(
            row_gaps, horizontal_count, rng
        )
        divider_records: list[dict[str, Any]] = []
        all_wall_cells: set[Cell] = set()

        for column in vertical_columns:
            gate_rows = _sample_gate_coordinates(
                row_gaps, gate_count, set(horizontal_rows), rng
            )
            gate_cells = {(row, column) for row in gate_rows}
            wall_cells = {
                (row, column)
                for row in range(rows)
                if (row, column) not in gate_cells
            }
            all_wall_cells.update(wall_cells)
            divider_records.append(
                {
                    "template": divider_template,
                    "orientation": "vertical",
                    "divider_columns": [column],
                    "gate_cells": _cells_to_json(gate_cells),
                    "wall_cells": _cells_to_json(wall_cells),
                }
            )

        for row in horizontal_rows:
            gate_columns = _sample_gate_coordinates(
                col_gaps, gate_count, set(vertical_columns), rng
            )
            gate_cells = {(row, col) for col in gate_columns}
            wall_cells = {
                (row, col)
                for col in range(cols)
                if (row, col) not in gate_cells
            }
            all_wall_cells.update(wall_cells)
            divider_records.append(
                {
                    "template": divider_template,
                    "orientation": "horizontal",
                    "divider_rows": [row],
                    "gate_cells": _cells_to_json(gate_cells),
                    "wall_cells": _cells_to_json(wall_cells),
                }
            )

        expected_wall_count = vertical_count + horizontal_count
        if (
            len(divider_records) == expected_wall_count
            and all(
                len(record["gate_cells"]) == gate_count
                for record in divider_records
            )
            and _try_block_cells(
                grid,
                obstacle_types,
                sorted(all_wall_cells),
                "W",
            )
        ):
            structural_changes["compartment_gates"].extend(
                divider_records
            )

    service_cells: set[Cell] = set()
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] != ".":
                continue
            if any(
                0 <= row + dr < rows
                and 0 <= col + dc < cols
                and obstacle_types[row + dr][col + dc] == "S"
                for dr, dc in _DIRECTIONS
            ):
                service_cells.add((row, col))

    sides = list(config.get("station_sides", ["left", "right"]))
    if station_placement == "single_side":
        sides = [rng.choice(sides)]
    elif station_placement == "opposite_sides" and len(sides) < 2:
        raise ValueError("opposite_sides requires at least two station_sides")
    elif station_placement not in {
        "distributed",
        "clustered",
        "single_side",
        "opposite_sides",
    }:
        raise ValueError(f"unsupported station_placement: {station_placement}")
    if station_placement == "opposite_sides":
        side_candidates = [
            _station_candidates(rows, cols, [side], beltway)
            for side in sides
        ]
        for values in side_candidates:
            rng.shuffle(values)
        candidates = [
            cell
            for offset in range(max(map(len, side_candidates), default=0))
            for values in side_candidates
            if offset < len(values)
            for cell in [values[offset]]
        ]
    else:
        candidates = _station_candidates(rows, cols, sides, beltway)
    if station_placement == "clustered":
        centers = rng.sample(
            candidates, min(station_cluster_count, len(candidates))
        )
        candidates.sort(
            key=lambda cell: min(
                abs(cell[0] - center[0]) + abs(cell[1] - center[1])
                for center in centers
            )
        )
    elif station_placement != "opposite_sides":
        rng.shuffle(candidates)
    stations: list[Cell] = []
    for candidate in candidates:
        if grid[candidate[0]][candidate[1]] != ".":
            continue
        if all(
            abs(candidate[0] - row) + abs(candidate[1] - col)
            >= station_clearance
            for row, col in stations
        ):
            stations.append(candidate)
            if len(stations) == station_count:
                break
    if len(stations) != station_count:
        raise ValueError("unable to place requested number of stations")

    station_entrances: dict[str, list[Cell]] = {}
    station_zones: dict[str, list[Cell]] = {}
    for index, station in enumerate(stations):
        if station[1] <= beltway or station[1] >= cols - 1 - beltway:
            entrance = [
                (row, station[1])
                for row in range(
                    max(0, station[0] - station_entrance_width // 2),
                    min(
                        rows,
                        station[0] + (station_entrance_width + 1) // 2,
                    ),
                )
                if grid[row][station[1]] == "."
            ]
        else:
            entrance = [
                (station[0], col)
                for col in range(
                    max(0, station[1] - station_entrance_width // 2),
                    min(
                        cols,
                        station[1] + (station_entrance_width + 1) // 2,
                    ),
                )
                if grid[station[0]][col] == "."
            ]
        station_key = f"station_{index:02d}"
        station_entrances[station_key] = entrance
        zone: list[Cell] = []
        for row in range(rows):
            for col in range(cols):
                if (
                    grid[row][col] == "."
                    and abs(row - station[0]) + abs(col - station[1])
                    <= station_clearance
                ):
                    zone.append((row, col))
        if station[1] <= beltway:
            queue_cells = [
                (row, col)
                for row in range(
                    max(0, station[0] - station_queue_width // 2),
                    min(
                        rows,
                        station[0] + (station_queue_width + 1) // 2,
                    ),
                )
                for col in range(
                    station[1],
                    min(cols, station[1] + station_queue_depth + 1),
                )
            ]
        elif station[1] >= cols - 1 - beltway:
            queue_cells = [
                (row, col)
                for row in range(
                    max(0, station[0] - station_queue_width // 2),
                    min(
                        rows,
                        station[0] + (station_queue_width + 1) // 2,
                    ),
                )
                for col in range(
                    max(0, station[1] - station_queue_depth),
                    station[1] + 1,
                )
            ]
        elif station[0] <= beltway:
            queue_cells = [
                (row, col)
                for row in range(
                    station[0],
                    min(rows, station[0] + station_queue_depth + 1),
                )
                for col in range(
                    max(0, station[1] - station_queue_width // 2),
                    min(
                        cols,
                        station[1] + (station_queue_width + 1) // 2,
                    ),
                )
            ]
        else:
            queue_cells = [
                (row, col)
                for row in range(
                    max(0, station[0] - station_queue_depth),
                    station[0] + 1,
                )
                for col in range(
                    max(0, station[1] - station_queue_width // 2),
                    min(
                        cols,
                        station[1] + (station_queue_width + 1) // 2,
                    ),
                )
            ]
        zone.extend(
            cell
            for cell in queue_cells
            if grid[cell[0]][cell[1]] == "."
        )
        zone = list(dict.fromkeys(zone))
        zone.extend(entrance)
        station_zones[station_key] = list(dict.fromkeys(zone))
    station_region = {
        cell for zone in station_zones.values() for cell in zone
    }

    row_bands = [
        range(start, start + block_height) for start in row_starts
    ]
    col_bands = [
        range(start, start + block_width) for start in col_starts
    ]
    station_set = {
        cell for entrance in station_entrances.values() for cell in entrance
    }
    semantic = [
        [
            (
                "@"
                if obstacle_types[row][col] == "S"
                else obstacle_types[row][col]
                if obstacle_types[row][col] in {"W", "N"}
                else "."
            )
            for col in range(cols)
        ]
        for row in range(rows)
    ]
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == "@":
                continue
            if (
                row < beltway
                or row >= rows - beltway
                or col < beltway
                or col >= cols - beltway
            ):
                semantic[row][col] = "B"
                continue
            in_block_row = any(row in band for band in row_bands)
            in_block_col = any(col in band for band in col_bands)
            if not in_block_row and not in_block_col:
                semantic[row][col] = "X"
            elif not in_block_row:
                semantic[row][col] = "H"
            elif not in_block_col:
                semantic[row][col] = "V"
    for row, col in service_cells:
        semantic[row][col] = "S"
    for row, col in station_set:
        semantic[row][col] = "P"

    articulation = _articulation_cells(grid)
    topology_metrics = _topology_metrics(grid)
    topology_metrics["articulation_count"] = len(articulation)
    topology_metrics["route_redundancy_proxy"] = round(
        max(0.0, float(topology_metrics["average_free_degree"]) - 2.0),
        6,
    )
    clearance = _obstacle_clearance(grid)
    prior: list[list[float]] = []
    for row in range(rows):
        prior_row = []
        for col in range(cols):
            if grid[row][col] == "@":
                prior_row.append(0.0)
                continue
            cell = (row, col)
            degree = len(_free_neighbors(grid, cell))
            score = 1.0 / (1.0 + clearance[row][col])
            if degree <= 2:
                score += 0.2
            if cell in articulation:
                score += 0.5
            if semantic[row][col] == "X":
                score += 0.1
            if cell in station_region:
                score += 0.1
            prior_row.append(round(min(1.0, score), 4))
        prior.append(prior_row)

    left_services = [
        cell for cell in service_cells if cell[1] < cols / 3
    ]
    center_services = [
        cell for cell in service_cells if cols / 3 <= cell[1] < 2 * cols / 3
    ]
    right_services = [
        cell for cell in service_cells if cell[1] >= 2 * cols / 3
    ]
    station_zone_cells = sorted(station_region)

    metadata = {
        "schema_version": 1,
        "map_type": "warehouse",
        "sampled_parameters": {
            "layout_mode": layout_mode,
            "rows": rows,
            "cols": cols,
            "shelf_block_height": block_height,
            "shelf_block_width": block_width,
            "horizontal_aisle_width": horizontal_aisle,
            "vertical_aisle_width": vertical_aisle,
            "outer_beltway_width": beltway,
            "beltway_mode": beltway_mode,
            "beltway_open_fraction": round(beltway_open_fraction, 4),
            "wall_clearance": wall_clearance,
            "shelf_wall_contact_probability": round(
                wall_contact_probability, 4
            ),
            "dead_end_aisle_probability": round(
                dead_end_probability, 4
            ),
            "dead_end_depth": dead_end_depth,
            "cross_aisle_removal_probability": round(
                cross_removal_probability, 4
            ),
            "narrow_aisle_probability": round(narrow_probability, 4),
            "narrow_segment_length": narrow_segment_length,
            "combine_layout_features": combine_layout_features,
            "beltway_narrow_segment_count": (
                beltway_narrow_segment_count
            ),
            "beltway_blocked_segment_count": (
                beltway_blocked_segment_count
            ),
            "beltway_segment_length": beltway_segment_length,
            "beltway_narrow_width": beltway_narrow_width,
            "wall_shelf_extension_count": wall_shelf_extension_count,
            "dead_end_aisle_count": dead_end_aisle_count,
            "dead_end_orientation_counts": {
                "vertical": dead_end_vertical_count,
                "horizontal": dead_end_horizontal_count,
            },
            "cross_aisle_closure_count": cross_aisle_closure_count,
            "cross_aisle_span_blocks": cross_aisle_span_blocks,
            "variable_vertical_aisle_widths": (
                sampled_vertical_aisle_widths
            ),
            "gate_count": gate_count,
            "gate_width": gate_width,
            "divider_template": divider_template,
            "divider_orientation": divider_orientation,
            "divider_thickness": divider_thickness,
            "asymmetric_trim_width": asymmetric_trim_width,
            "buffer_zone_count": buffer_zone_count,
            "station_count": station_count,
            "station_clearance": station_clearance,
            "station_sides": sides,
            "station_placement": station_placement,
            "station_cluster_count": station_cluster_count,
            "station_entrance_width": station_entrance_width,
            "station_queue_depth": station_queue_depth,
            "station_queue_width": station_queue_width,
            "station_demand_distribution": station_demand_distribution,
            "layout_jitter": round(layout_jitter, 4),
        },
        "structural_changes": structural_changes,
        "obstacle_type_layer": [
            "".join(row) for row in obstacle_types
        ],
        "shelf_blocks": shelf_blocks,
        "stations": [
            {
                "id": f"station_{index:02d}",
                "cell": list(cell),
                "entrance_cells": _cells_to_json(
                    station_entrances[f"station_{index:02d}"]
                ),
                "demand_weight": _weighted_station_values(
                    len(stations), station_demand_distribution
                )[index],
                "entrance_width": station_entrance_width,
            }
            for index, cell in enumerate(stations)
        ],
        "station_zones": {
            key: _cells_to_json(value)
            for key, value in station_zones.items()
        },
        "service_cells": _cells_to_json(service_cells),
        "zones": {
            "left_storage": _cells_to_json(left_services),
            "center_storage": _cells_to_json(center_services),
            "right_storage": _cells_to_json(right_services),
            "station_approach": _cells_to_json(station_zone_cells),
        },
        "semantic_cell_types": ["".join(row) for row in semantic],
        "semantic_legend": {
            "@": "shelf",
            "W": "wall_or_closure",
            "N": "narrowing_obstacle",
            ".": "unclassified_free",
            "B": "outer_beltway",
            "H": "horizontal_aisle",
            "V": "vertical_aisle",
            "X": "intersection",
            "S": "storage_service",
            "P": "picking_station",
        },
        "articulation_cells": _cells_to_json(articulation),
        "topology_metrics": topology_metrics,
        "structural_congestion_prior": prior,
        "free_cell_count": sum(row.count(".") for row in grid),
    }
    return MapData(
        map_id=map_id,
        seed=seed,
        grid=["".join(row) for row in grid],
        metadata=metadata,
    )


def _matches_topology_constraints(
    map_data: MapData, constraints: dict[str, Any]
) -> bool:
    metrics = map_data.metadata["topology_metrics"]
    checks = {
        "minimum_articulation_count": (
            metrics["articulation_count"],
            lambda actual, expected: actual >= expected,
        ),
        "maximum_articulation_count": (
            metrics["articulation_count"],
            lambda actual, expected: actual <= expected,
        ),
        "minimum_dead_end_count": (
            metrics["dead_end_cell_count"],
            lambda actual, expected: actual >= expected,
        ),
        "maximum_dead_end_count": (
            metrics["dead_end_cell_count"],
            lambda actual, expected: actual <= expected,
        ),
        "minimum_average_degree": (
            metrics["average_free_degree"],
            lambda actual, expected: actual >= expected,
        ),
        "maximum_average_degree": (
            metrics["average_free_degree"],
            lambda actual, expected: actual <= expected,
        ),
    }
    return all(
        key not in constraints or predicate(actual, constraints[key])
        for key, (actual, predicate) in checks.items()
    )


def _has_primary_layout_feature(map_data: MapData) -> bool:
    parameters = map_data.metadata["sampled_parameters"]
    changes = map_data.metadata["structural_changes"]
    mode = parameters["layout_mode"]
    requirements = {
        "partial_beltway": bool(changes["beltway_closures"]),
        "wall_shelves": bool(changes["wall_extensions"]),
        "dead_end_aisles": bool(changes["dead_end_caps"]),
        "partial_cross_aisles": bool(
            changes["cross_aisle_closures"]
        ),
        "compartmentalized": bool(changes["compartment_gates"]),
        "mixed_width": len(
            set(parameters["variable_vertical_aisle_widths"])
        )
        > 1,
        "asymmetric": bool(changes["asymmetric_shelf_adjustments"]),
        "station_centric": parameters["station_placement"] == "clustered",
    }
    return requirements.get(mode, True)


def _matches_layout_requirements(map_data: MapData) -> bool:
    parameters = map_data.metadata["sampled_parameters"]
    changes = map_data.metadata["structural_changes"]
    mode = parameters["layout_mode"]
    if mode == "dead_end_aisles":
        expected = parameters["dead_end_orientation_counts"]
        actual = collections.Counter(
            cap.get("orientation", "vertical")
            for cap in changes["dead_end_caps"]
        )
        return all(
            actual[orientation] == count
            for orientation, count in expected.items()
        )
    if mode == "compartmentalized":
        dividers = changes["compartment_gates"]
        template = parameters["divider_template"]
        expected_wall_count = 1 if template == "single" else 2
        if len(dividers) != expected_wall_count:
            return False
        expected_gate_cells = (
            parameters["gate_count"]
            if template != "single"
            else parameters["gate_count"] * parameters["gate_width"]
        )
        return all(
            len(divider["gate_cells"]) == expected_gate_cells
            for divider in dividers
        )
    return True


def generate_warehouse(
    config: dict[str, Any], seed: int, map_id: str
) -> MapData:
    constraints = dict(config.get("topology_constraints", {}))
    attempts = int(config.get("generation_attempts", 20))
    if attempts <= 0:
        raise ValueError("generation_attempts must be positive")
    for attempt in range(attempts):
        generation_seed = seed + attempt * 1_000_003
        map_data = _generate_warehouse_once(
            config, generation_seed, map_id
        )
        if (
            _has_primary_layout_feature(map_data)
            and _matches_layout_requirements(map_data)
            and _matches_topology_constraints(map_data, constraints)
        ):
            map_data.seed = seed
            map_data.metadata["requested_seed"] = seed
            map_data.metadata["generation_seed"] = generation_seed
            map_data.metadata["generation_attempt"] = attempt + 1
            map_data.metadata["topology_constraints"] = constraints
            return map_data
    raise ValueError(
        f"unable to satisfy topology_constraints after {attempts} attempts"
    )
