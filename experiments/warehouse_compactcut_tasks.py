"""Pure-geometry compact Warehouse maps and paired cut-demand tasks.

The module deliberately does not import or invoke the native solver.  It
constructs eight deterministic 28x39 maps, proves a two-cell divider cut from
the realised grid, and builds paired 120-agent tasks.  The structured task
places every origin/destination pair on opposite sides of the cut.  The
diagnostic task keeps every pair inside one cut partition; it is a diagnostic
control, not an exactly difficulty-matched causal control.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from generators.models import MapData
from generators.warehouse import generate_warehouse


MAP_BANK_SCHEMA = "lns2.warehouse_compactcut_map_bank.v1"
GEOMETRY_AUDIT_SCHEMA = "lns2.warehouse_compactcut_geometry_audit.v1"
TASK_PAIR_SCHEMA = "lns2.warehouse_compactcut_task_pair.v1"
TASK_AUDIT_SCHEMA = "lns2.warehouse_compactcut_task_audit.v1"
SERIALIZED_WITNESS_SCHEMA = "lns2.prioritized_joint_mapf_feasibility_witness.v1"

STRUCTURED_VARIANT = "bidirectional_mandatory_cut_exchange"
DIAGNOSTIC_VARIANT = "diagnostic_within_partition_exchange"
TASK_VARIANTS = (STRUCTURED_VARIANT, DIAGNOSTIC_VARIANT)
AGENT_COUNT = 120
PARTITION_FLOW_COUNT = AGENT_COUNT // 2
DEFAULT_TASK_SEEDS = (521, 557)
DISTANCE_MEAN_RELATIVE_GATE = 0.05
DISTANCE_P95_RELATIVE_GATE = 0.10
DIAGNOSTIC_TARGET_OFFSETS = tuple(
    [0, *(offset for step in range(1, 21) for offset in (-step, step))]
)

MAP_SPECS: tuple[dict[str, Any], ...] = (
    *(
        {
            "map_id": f"whcc_cfg_{index:02d}",
            "divider_template": "cross_four_gate",
            "map_seed": seed,
        }
        for index, seed in enumerate(range(2026081701, 2026081705), 1)
    ),
    *(
        {
            "map_id": f"whcc_dh_{index:02d}",
            "divider_template": "double_horizontal",
            "map_seed": seed,
        }
        for index, seed in enumerate(
            (2026081711, 2026081712, 2026081713, 2026081715), 1
        )
    ),
)
_MAP_SPEC_BY_ID = {str(spec["map_id"]): dict(spec) for spec in MAP_SPECS}

_MAP_CONFIG_BASE: dict[str, Any] = {
    "layout_mode": "compartmentalized",
    "rows": 28,
    "cols": 39,
    "shelf_block_height": 2,
    "shelf_block_width": 6,
    "horizontal_aisle_width": 2,
    "vertical_aisle_width": 2,
    "outer_beltway_width": 0,
    "beltway_mode": "none",
    "wall_clearance": 0,
    "gate_count": 2,
    "gate_width": 1,
    "divider_thickness": 1,
    "combine_layout_features": False,
    "buffer_zone_count": 0,
    "station_count": 2,
    "station_sides": ["left", "right"],
    "station_clearance": 3,
    "station_placement": "opposite_sides",
    "station_cluster_count": 1,
    "station_entrance_width": 1,
    "station_queue_depth": 2,
    "station_queue_width": 1,
    "station_demand_distribution": "uniform",
    "layout_jitter": 0.0,
    "dead_end_aisle_probability": 0.0,
    "dead_end_aisle_count": 0,
    "dead_end_orientation_counts": {"vertical": 0, "horizontal": 0},
    "cross_aisle_removal_probability": 0.0,
    "narrow_aisle_probability": 0.0,
    "generation_attempts": 40,
    "topology_constraints": {"minimum_average_degree": 1.5},
}

_DIRECTIONS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _grid_sha(grid: Sequence[str]) -> str:
    return _canonical_sha({"height": len(grid), "width": len(grid[0]), "grid": list(grid)})


def _free_cells(grid: Sequence[str]) -> set[tuple[int, int]]:
    return {
        (row, col)
        for row, line in enumerate(grid)
        for col, value in enumerate(line)
        if value == "."
    }


def _neighbors(cell: tuple[int, int]) -> Iterable[tuple[int, int]]:
    row, col = cell
    for dr, dc in _DIRECTIONS:
        yield row + dr, col + dc


def _components(
    grid: Sequence[str], removed: Iterable[tuple[int, int]] = ()
) -> list[set[tuple[int, int]]]:
    remaining = _free_cells(grid) - set(removed)
    result: list[set[tuple[int, int]]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue: collections.deque[tuple[int, int]] = collections.deque([start])
        while queue:
            cell = queue.popleft()
            for neighbor in _neighbors(cell):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        result.append(component)
    return sorted(result, key=lambda cells: min(cells))


def _distance_map(
    grid: Sequence[str],
    start: tuple[int, int],
    *,
    removed: Iterable[tuple[int, int]] = (),
    occupied: Iterable[tuple[int, int]] = (),
) -> dict[tuple[int, int], int]:
    blocked = set(removed) | set(occupied)
    blocked.discard(start)
    if start not in _free_cells(grid):
        return {}
    distance = {start: 0}
    queue: collections.deque[tuple[int, int]] = collections.deque([start])
    while queue:
        cell = queue.popleft()
        for neighbor in _neighbors(cell):
            row, col = neighbor
            if (
                0 <= row < len(grid)
                and 0 <= col < len(grid[0])
                and grid[row][col] == "."
                and neighbor not in blocked
                and neighbor not in distance
            ):
                distance[neighbor] = distance[cell] + 1
                queue.append(neighbor)
    return distance


def _shortest_path(
    grid: Sequence[str],
    start: tuple[int, int],
    goal: tuple[int, int],
    *,
    removed: Iterable[tuple[int, int]] = (),
    occupied: Iterable[tuple[int, int]] = (),
) -> list[tuple[int, int]] | None:
    blocked = set(removed) | set(occupied)
    blocked.discard(start)
    if goal in blocked:
        return None
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: collections.deque[tuple[int, int]] = collections.deque([start])
    while queue and goal not in parent:
        cell = queue.popleft()
        for neighbor in _neighbors(cell):
            row, col = neighbor
            if (
                0 <= row < len(grid)
                and 0 <= col < len(grid[0])
                and grid[row][col] == "."
                and neighbor not in blocked
                and neighbor not in parent
            ):
                parent[neighbor] = cell
                queue.append(neighbor)
    if goal not in parent:
        return None
    path = [goal]
    while path[-1] != start:
        predecessor = parent[path[-1]]
        assert predecessor is not None
        path.append(predecessor)
    path.reverse()
    return path


def _p95(values: Sequence[int]) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _rng(*parts: Any) -> random.Random:
    material = "|".join(map(str, (TASK_PAIR_SCHEMA, *parts))).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def _spec(map_id: str) -> dict[str, Any]:
    if map_id not in _MAP_SPEC_BY_ID:
        raise ValueError(f"unregistered compact-cut map_id: {map_id}")
    return dict(_MAP_SPEC_BY_ID[map_id])


def generate_compactcut_map_bank() -> dict[str, MapData]:
    """Generate the eight checksum-stable maps; no reset or solver is used."""

    bank: dict[str, MapData] = {}
    for registered in MAP_SPECS:
        config = dict(_MAP_CONFIG_BASE)
        config["divider_template"] = registered["divider_template"]
        map_data = generate_warehouse(
            config, int(registered["map_seed"]), str(registered["map_id"])
        )
        map_data.metadata["compactcut_registration"] = {
            "schema": MAP_BANK_SCHEMA,
            **dict(registered),
            "agent_count": AGENT_COUNT,
            "cut_capacity": 2,
            "grid_sha256": _grid_sha(map_data.grid),
        }
        audit = audit_compactcut_geometry(map_data)
        if not audit["passed"]:
            raise ValueError(
                f"registered compact-cut map failed Q0: {map_data.map_id}: "
                + ",".join(audit["errors"])
            )
        bank[map_data.map_id] = map_data
    grid_hashes = {
        str(map_data.metadata["compactcut_registration"]["grid_sha256"])
        for map_data in bank.values()
    }
    if len(grid_hashes) != len(MAP_SPECS):
        raise ValueError("registered compact-cut map bank contains duplicate grids")
    return bank


def _divider_candidates(map_data: MapData) -> list[dict[str, Any]]:
    grid = map_data.grid
    records = list(
        map_data.metadata.get("structural_changes", {}).get("compartment_gates", ())
    )
    candidates: list[dict[str, Any]] = []
    for record_index, raw in enumerate(records):
        record = dict(raw)
        orientation = str(record.get("orientation", ""))
        gates = sorted(tuple(map(int, cell)) for cell in record.get("gate_cells", ()))
        if orientation == "vertical":
            coordinates = list(map(int, record.get("divider_columns", ())))
            line = (
                [(row, coordinates[0]) for row in range(len(grid))]
                if len(coordinates) == 1
                else []
            )
        elif orientation == "horizontal":
            coordinates = list(map(int, record.get("divider_rows", ())))
            line = (
                [(coordinates[0], col) for col in range(len(grid[0]))]
                if len(coordinates) == 1
                else []
            )
        else:
            coordinates, line = [], []
        realised_free = sorted(cell for cell in line if grid[cell[0]][cell[1]] == ".")
        line_exact = bool(line) and realised_free == gates and len(gates) == 2
        components = _components(grid, gates) if line_exact else []
        valid_partition = (
            len(components) == 2
            and min(map(len, components)) >= 2 * PARTITION_FLOW_COUNT
        )
        gate_subset_connectivity = []
        if valid_partition:
            representatives = [min(component) for component in components]
            for retained in gates:
                reachable = _distance_map(
                    grid,
                    representatives[0],
                    removed=set(gates) - {retained},
                )
                gate_subset_connectivity.append(representatives[1] in reachable)
        candidates.append(
            {
                "record_index": record_index,
                "orientation": orientation,
                "coordinate": coordinates[0] if len(coordinates) == 1 else None,
                "cut_cells": [list(cell) for cell in gates],
                "line_exact_on_realised_grid": line_exact,
                "component_count_after_cut": len(components),
                "component_sizes_after_cut": sorted(
                    (len(component) for component in components), reverse=True
                ),
                "both_gate_subsets_restore_connectivity": bool(
                    gate_subset_connectivity and all(gate_subset_connectivity)
                ),
                "valid_two_cell_cut": bool(
                    line_exact
                    and valid_partition
                    and gate_subset_connectivity
                    and all(gate_subset_connectivity)
                ),
            }
        )
    return candidates


def audit_compactcut_geometry(map_data: MapData) -> dict[str, Any]:
    """Audit identity and a real-grid, minimal two-cell divider cut."""

    errors: list[str] = []
    try:
        registered = _spec(map_data.map_id)
    except ValueError as exc:
        return {"schema": GEOMETRY_AUDIT_SCHEMA, "passed": False, "errors": [str(exc)]}
    parameters = dict(map_data.metadata.get("sampled_parameters") or {})
    registration = dict(map_data.metadata.get("compactcut_registration") or {})
    expected_registration = {
        "schema": MAP_BANK_SCHEMA,
        **registered,
        "agent_count": AGENT_COUNT,
        "cut_capacity": 2,
        "grid_sha256": _grid_sha(map_data.grid),
    }
    if registration != expected_registration:
        errors.append("registration_mismatch")
    if len(map_data.grid) != 28 or any(len(row) != 39 for row in map_data.grid):
        errors.append("map_dimensions_mismatch")
    if any(value not in ".@" for row in map_data.grid for value in row):
        errors.append("unsupported_grid_cell")
    exact_parameters = {
        "layout_mode": "compartmentalized",
        "rows": 28,
        "cols": 39,
        "outer_beltway_width": 0,
        "beltway_mode": "none",
        "gate_count": 2,
        "gate_width": 1,
        "divider_thickness": 1,
        "divider_template": registered["divider_template"],
    }
    if any(parameters.get(key) != value for key, value in exact_parameters.items()):
        errors.append("map_parameter_mismatch")
    if (
        int(map_data.seed) != int(registered["map_seed"])
        or int(map_data.metadata.get("requested_seed", -1)) != int(registered["map_seed"])
    ):
        errors.append("map_seed_mismatch")
    if len(_components(map_data.grid)) != 1:
        errors.append("realised_grid_not_connected")

    candidates = _divider_candidates(map_data)
    valid = [candidate for candidate in candidates if candidate["valid_two_cell_cut"]]
    selected: dict[str, Any] | None = None
    if not valid:
        errors.append("no_registered_capacity_two_cut")
    else:
        selected = max(
            valid,
            key=lambda candidate: (
                min(candidate["component_sizes_after_cut"]),
                1 if candidate["orientation"] == "horizontal" else 0,
                -int(candidate["coordinate"]),
            ),
        )
    partition_hashes: list[str] = []
    if selected is not None:
        cut = {tuple(cell) for cell in selected["cut_cells"]}
        components = _components(map_data.grid, cut)
        partition_hashes = [
            _canonical_sha([list(cell) for cell in sorted(component)])
            for component in components
        ]
        if len(cut) != 2 or len(components) != 2:
            errors.append("selected_cut_bypass_detected")

    return {
        "schema": GEOMETRY_AUDIT_SCHEMA,
        "passed": not errors,
        "errors": errors,
        "map_id": map_data.map_id,
        "map_seed": map_data.seed,
        "divider_template": registered["divider_template"],
        "grid_sha256": _grid_sha(map_data.grid),
        "height": len(map_data.grid),
        "width": len(map_data.grid[0]) if map_data.grid else 0,
        "cut_capacity": 2,
        "candidate_cuts": candidates,
        "selected_cut": selected,
        "partition_membership_sha256": partition_hashes,
        "minimum_endpoint_capacity_per_partition": 2 * PARTITION_FLOW_COUNT,
        "bypass_after_selected_cut_removal": False if selected is not None else None,
    }


def _hungarian(cost: Sequence[Sequence[int]]) -> list[int]:
    """Return a deterministic minimum-cost square assignment."""

    size = len(cost)
    if size == 0 or any(len(row) != size for row in cost):
        raise ValueError("assignment cost matrix must be non-empty and square")
    u = [0] * (size + 1)
    v = [0] * (size + 1)
    matched_row = [0] * (size + 1)
    previous_column = [0] * (size + 1)
    infinity = 10**30
    for row_index in range(1, size + 1):
        matched_row[0] = row_index
        minimum = [infinity] * (size + 1)
        used = [False] * (size + 1)
        column = 0
        while True:
            used[column] = True
            row = matched_row[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = cost[row - 1][candidate - 1] - u[row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            prior = previous_column[column]
            matched_row[column] = matched_row[prior]
            column = prior
            if column == 0:
                break
    assignment = [-1] * size
    for column in range(1, size + 1):
        assignment[matched_row[column] - 1] = column - 1
    if sorted(assignment) != list(range(size)):
        raise AssertionError("Hungarian assignment is not a permutation")
    return assignment


def _hungarian_rectangular(cost: Sequence[Sequence[int]]) -> list[int]:
    """Deterministic Hungarian assignment for rows <= columns."""

    row_count = len(cost)
    column_count = len(cost[0]) if cost else 0
    if (
        row_count == 0
        or column_count < row_count
        or any(len(row) != column_count for row in cost)
    ):
        raise ValueError("rectangular assignment requires 0 < rows <= columns")
    u = [0] * (row_count + 1)
    v = [0] * (column_count + 1)
    matched_row = [0] * (column_count + 1)
    previous_column = [0] * (column_count + 1)
    infinity = 10**30
    for row_index in range(1, row_count + 1):
        matched_row[0] = row_index
        minimum = [infinity] * (column_count + 1)
        used = [False] * (column_count + 1)
        column = 0
        while True:
            used[column] = True
            row = matched_row[column]
            delta = infinity
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = cost[row - 1][candidate - 1] - u[row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[matched_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if matched_row[column] == 0:
                break
        while True:
            prior = previous_column[column]
            matched_row[column] = matched_row[prior]
            column = prior
            if column == 0:
                break
    assignment = [-1] * row_count
    for column in range(1, column_count + 1):
        if matched_row[column] != 0:
            assignment[matched_row[column] - 1] = column - 1
    if any(column < 0 for column in assignment) or len(set(assignment)) != row_count:
        raise AssertionError("rectangular Hungarian assignment is incomplete")
    return assignment


def _distance_to_cut(
    component: set[tuple[int, int]], cut: set[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    frontier = sorted(
        cell
        for cell in component
        if any(neighbor in cut for neighbor in _neighbors(cell))
    )
    if not frontier:
        raise ValueError("cut has no neighbor in partition")
    distance = {cell: 0 for cell in frontier}
    queue: collections.deque[tuple[int, int]] = collections.deque(frontier)
    while queue:
        cell = queue.popleft()
        for neighbor in _neighbors(cell):
            if neighbor in component and neighbor not in distance:
                distance[neighbor] = distance[cell] + 1
                queue.append(neighbor)
    if len(distance) != len(component):
        raise ValueError("partition is disconnected from registered cut")
    return distance


def _assignment_for_target(
    starts: Sequence[tuple[int, int]],
    goals: Sequence[tuple[int, int]],
    distance_maps: Mapping[tuple[int, int], Mapping[tuple[int, int], int]],
    target_twice: int,
    *,
    forbid_fixed_points: bool = False,
) -> tuple[list[tuple[int, int]], list[int]]:
    matrix: list[list[int]] = []
    for start in starts:
        row: list[int] = []
        for goal_index, goal in enumerate(goals):
            distance = int(distance_maps[start][goal])
            # The tiny column term makes otherwise equal optima stable without
            # changing the primary squared-distance objective.
            fixed_penalty = 10**15 if forbid_fixed_points and start == goal else 0
            row.append(
                fixed_penalty
                + (2 * distance - target_twice) ** 2 * 1000
                + goal_index
            )
        matrix.append(row)
    assignment = _hungarian(matrix)
    assigned_goals = [goals[index] for index in assignment]
    distances = [
        int(distance_maps[start][goal])
        for start, goal in zip(starts, assigned_goals)
    ]
    return assigned_goals, distances


def _rectangular_assignment_for_target(
    starts: Sequence[tuple[int, int]],
    goal_pool: Sequence[tuple[int, int]],
    distance_maps: Mapping[tuple[int, int], Mapping[tuple[int, int], int]],
    target_twice: int,
    *,
    forbid_fixed_points: bool = False,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Choose distinct goals from a larger frozen parking pool."""

    if len(goal_pool) < len(starts):
        raise ValueError("rectangular assignment has fewer goals than starts")
    matrix: list[list[int]] = []
    for start in starts:
        row = []
        for goal_index, goal in enumerate(goal_pool):
            distance = int(distance_maps[start][goal])
            fixed_penalty = 10**18 if forbid_fixed_points and start == goal else 0
            row.append(
                fixed_penalty
                + (2 * distance - target_twice) ** 2 * 1_000_000
                + goal_index
            )
        matrix.append(row)
    assignment = _hungarian_rectangular(matrix)
    assigned_goals = [goal_pool[index] for index in assignment]
    distances = [
        int(distance_maps[start][goal])
        for start, goal in zip(starts, assigned_goals)
    ]
    return assigned_goals, distances


def _exact_cardinality_parking_assignment(
    parking_order: Sequence[tuple[int, int]],
    distance_maps: Mapping[tuple[int, int], Mapping[tuple[int, int], int]],
    target_distance: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]:
    """Select exactly 60 distinct starts and goals from one parking pool.

    The square augmented assignment has ``n`` real rows/columns and ``n-60``
    dummy rows/columns.  Prohibiting dummy-to-dummy matches forces exactly 60
    real-to-real assignments without an outcome-derived endpoint subset.
    """

    parking = list(parking_order)
    size = len(parking)
    if size < PARTITION_FLOW_COUNT:
        raise ValueError("diagnostic parking pool has fewer than 60 cells")
    dummy_count = size - PARTITION_FLOW_COUNT
    augmented_size = size + dummy_count
    prohibited = 10**12
    cost = np.full(
        (augmented_size, augmented_size), prohibited, dtype=np.int64
    )
    for start_rank, start in enumerate(parking):
        for goal_rank, goal in enumerate(parking):
            distance = int(distance_maps[start][goal])
            value = (distance - int(target_distance)) ** 2 * 1000 + goal_rank
            if start == goal:
                value += prohibited
            cost[start_rank, goal_rank] = value
        if dummy_count:
            cost[start_rank, size:] = 0
    if dummy_count:
        cost[size:, :size] = 0
    row_indices, column_indices = linear_sum_assignment(cost)
    real_matches = sorted(
        (int(row), int(column))
        for row, column in zip(row_indices, column_indices)
        if row < size and column < size
    )
    if len(real_matches) != PARTITION_FLOW_COUNT:
        raise ValueError("exact-cardinality assignment did not select 60 pairs")
    starts = [parking[row] for row, _ in real_matches]
    goals = [parking[column] for _, column in real_matches]
    if any(start == goal for start, goal in zip(starts, goals)):
        raise ValueError("exact-cardinality assignment selected a fixed point")
    distances = [
        int(distance_maps[start][goal]) for start, goal in zip(starts, goals)
    ]
    return starts, goals, distances


def _shortest_path_in_cells(
    cells: set[tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    if start not in cells or goal not in cells:
        return None
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue: collections.deque[tuple[int, int]] = collections.deque([start])
    while queue and goal not in parent:
        cell = queue.popleft()
        for neighbor in sorted(_neighbors(cell)):
            if neighbor in cells and neighbor not in parent:
                parent[neighbor] = cell
                queue.append(neighbor)
    if goal not in parent:
        return None
    path = [goal]
    while path[-1] != start:
        predecessor = parent[path[-1]]
        assert predecessor is not None
        path.append(predecessor)
    path.reverse()
    return path


def _connected_dominating_routing_core(
    partition: set[tuple[int, int]],
    cut: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    """Build the frozen lexicographic connected dominating routing core."""

    gate_neighbors = sorted(
        cell for cell in partition if any(neighbor in cut for neighbor in _neighbors(cell))
    )
    if len(gate_neighbors) != len(cut):
        raise ValueError("partition does not expose one neighbor per cut gate")
    initial_path = _shortest_path_in_cells(
        partition, gate_neighbors[0], gate_neighbors[1]
    )
    if initial_path is None:
        raise ValueError("gate neighbors are disconnected inside partition")
    core = set(initial_path)

    def dominated_cells() -> set[tuple[int, int]]:
        return {
            cell
            for cell in partition
            if cell in core or any(neighbor in core for neighbor in _neighbors(cell))
        }

    dominated = dominated_cells()
    while len(dominated) != len(partition):
        frontier = sorted(
            cell
            for cell in partition - core
            if any(neighbor in core for neighbor in _neighbors(cell))
        )
        if not frontier:
            raise ValueError("connected dominating core growth stalled")

        def new_dominated_count(cell: tuple[int, int]) -> int:
            closed = {cell} | {neighbor for neighbor in _neighbors(cell) if neighbor in partition}
            return len(closed - dominated)

        chosen = min(frontier, key=lambda cell: (-new_dominated_count(cell), cell))
        core.add(chosen)
        dominated = dominated_cells()
    return core


def _routing_registration(
    map_data: MapData,
    geometry: Mapping[str, Any],
    task_seed: int,
) -> dict[str, Any]:
    cut = {tuple(cell) for cell in dict(geometry["selected_cut"])["cut_cells"]}
    partitions = _components(map_data.grid, cut)
    records: list[dict[str, Any]] = []
    for partition_index, partition in enumerate(partitions):
        core = _connected_dominating_routing_core(partition, cut)
        parking_pool = sorted(
            cell
            for cell in partition - core
            if any(neighbor in core for neighbor in _neighbors(cell))
        )
        if len(parking_pool) < PARTITION_FLOW_COUNT + 1:
            raise ValueError("routing core leaves fewer than 61 off-core cells")
        distance_to_cut = _distance_to_cut(partition, cut)
        structured_pool = sorted(
            parking_pool, key=lambda cell: (distance_to_cut[cell], cell)
        )[: min(len(parking_pool), 90)]
        rng = _rng(
            map_data.map_id,
            geometry["grid_sha256"],
            int(task_seed),
            1234,
            partition_index,
        )
        sampled = rng.sample(structured_pool, PARTITION_FLOW_COUNT + 1)
        diagnostic_pool = list(parking_pool)
        rng.shuffle(diagnostic_pool)
        records.append(
            {
                "partition_index": partition_index,
                "partition": partition,
                "core": core,
                "parking_pool": parking_pool,
                "structured_pool": structured_pool,
                "parking": sampled[:PARTITION_FLOW_COUNT],
                "buffer": sampled[-1],
                "diagnostic_pool": diagnostic_pool,
            }
        )
    return {"cut": cut, "partitions": partitions, "records": records}


def _registered_task_assignments(
    map_data: MapData,
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    records = list(registration["records"])
    starts_a = list(records[0]["parking"])
    starts_b = list(records[1]["parking"])
    structured_starts = starts_a + starts_b
    all_parking = sorted(
        set(records[0]["parking_pool"]) | set(records[1]["parking_pool"])
    )
    distance_maps = {
        start: _distance_map(map_data.grid, start) for start in all_parking
    }
    if any(
        len(distance_maps[start]) != len(_free_cells(map_data.grid))
        for start in all_parking
    ):
        raise ValueError("generated parking cell cannot reach the realised grid")

    structured_goals_a, structured_distances_a = _assignment_for_target(
        starts_a, starts_b, distance_maps, 1
    )
    structured_goals_b, structured_distances_b = _assignment_for_target(
        starts_b, starts_a, distance_maps, 1
    )
    structured_distances = structured_distances_a + structured_distances_b
    structured = {
        "target_twice": 1,
        "starts": structured_starts,
        "goals": structured_goals_a + structured_goals_b,
        "distances": structured_distances,
        "mean": sum(structured_distances) / AGENT_COUNT,
        "p95": _p95(structured_distances),
    }

    rounded_target = int(round(float(structured["mean"])))
    diagnostic: dict[str, Any] | None = None
    best_failure: tuple[float, float] | None = None
    for target_offset in DIAGNOSTIC_TARGET_OFFSETS:
        target_distance = rounded_target + target_offset
        diagnostic_starts_a, diagnostic_goals_a, diagnostic_distances_a = (
            _exact_cardinality_parking_assignment(
                records[0]["diagnostic_pool"], distance_maps, target_distance
            )
        )
        diagnostic_starts_b, diagnostic_goals_b, diagnostic_distances_b = (
            _exact_cardinality_parking_assignment(
                records[1]["diagnostic_pool"], distance_maps, target_distance
            )
        )
        diagnostic_distances = diagnostic_distances_a + diagnostic_distances_b
        diagnostic_mean = sum(diagnostic_distances) / AGENT_COUNT
        diagnostic_p95 = _p95(diagnostic_distances)
        mean_delta = abs(diagnostic_mean - structured["mean"]) / structured["mean"]
        p95_delta = abs(diagnostic_p95 - structured["p95"]) / structured["p95"]
        if best_failure is None or (mean_delta, p95_delta) < best_failure:
            best_failure = (mean_delta, p95_delta)
        if (
            mean_delta <= DISTANCE_MEAN_RELATIVE_GATE
            and p95_delta <= DISTANCE_P95_RELATIVE_GATE
        ):
            diagnostic = {
                "target_distance": target_distance,
                "target_offset": target_offset,
                "starts": diagnostic_starts_a + diagnostic_starts_b,
                "goals": diagnostic_goals_a + diagnostic_goals_b,
                "distances": diagnostic_distances,
                "mean": diagnostic_mean,
                "p95": diagnostic_p95,
                "mean_delta": mean_delta,
                "p95_delta": p95_delta,
            }
            break
    if diagnostic is None:
        raise ValueError(
            "unable to construct registered compact-cut distance match; "
            f"best_distance_delta={best_failure}"
        )
    return {
        "structured": structured,
        "diagnostic": diagnostic,
        "distance_assignment": {
            "structured_target_twice": structured["target_twice"],
            "diagnostic_target_distance": diagnostic["target_distance"],
            "diagnostic_target_offset": diagnostic["target_offset"],
            "diagnostic_search_offsets": list(DIAGNOSTIC_TARGET_OFFSETS),
        },
    }


def _routing_registration_payload(
    registration: Mapping[str, Any],
) -> dict[str, Any]:
    records = list(registration["records"])
    cores = [set(record["core"]) for record in records]
    return {
        "schema": "lns2.compactcut_connected_dominating_routing_core.v1",
        "construction": "lex_bfs_gate_path_then_max_new_dominated_lexicographic",
        "sample_namespace": [1234],
        "structured_pool_rule": "nearest_to_cut_then_lexicographic_first_at_most_90",
        "diagnostic_pool_rule": "all_off_core_parking_rng_shuffled_exact_cardinality_60",
        "partition_core_sha256": [
            _canonical_sha([list(cell) for cell in sorted(core)]) for core in cores
        ],
        "partition_core_sizes": [len(core) for core in cores],
        "partition_parking_pool_sizes": [
            len(record["parking_pool"]) for record in records
        ],
        "partition_structured_pool_sizes": [
            len(record["structured_pool"]) for record in records
        ],
        "partition_buffers": [list(record["buffer"]) for record in records],
    }


def _core_route(
    source: tuple[int, int],
    destination: tuple[int, int],
    core: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    if source in core or destination in core or source == destination:
        raise ValueError("core route endpoints must be distinct off-core cells")
    entries = sorted(neighbor for neighbor in _neighbors(source) if neighbor in core)
    exits = sorted(neighbor for neighbor in _neighbors(destination) if neighbor in core)
    candidates: list[list[tuple[int, int]]] = []
    for entry in entries:
        for exit_cell in exits:
            middle = _shortest_path_in_cells(core, entry, exit_cell)
            if middle is not None:
                candidates.append([source, *middle, destination])
    if not candidates:
        raise ValueError("off-core endpoints are not connected through routing core")
    return min(candidates, key=lambda path: (len(path), path))


def _permutation_paths_and_cycles(
    starts: Sequence[tuple[int, int]], goals: Sequence[tuple[int, int]]
) -> tuple[list[list[int]], list[list[int]]]:
    """Decompose an injective parking transfer into paths and cycles."""

    if len(set(starts)) != len(starts) or len(set(goals)) != len(goals):
        raise ValueError("routing witness requires unique starts and goals")
    index_by_start = {cell: index for index, cell in enumerate(starts)}
    successor = [index_by_start.get(goal) for goal in goals]
    if any(index == target for index, target in enumerate(successor)):
        raise ValueError("routing witness requires a fixed-point-free mapping")
    unseen = set(range(len(starts)))
    goal_cells = set(goals)
    path_heads = sorted(
        index for index, start in enumerate(starts) if start not in goal_cells
    )
    paths: list[list[int]] = []
    for head in path_heads:
        if head not in unseen:
            continue
        path: list[int] = []
        current: int | None = head
        while current is not None:
            if current not in unseen:
                raise ValueError("routing path merges into another component")
            path.append(current)
            unseen.remove(current)
            current = successor[current]
        paths.append(path)

    cycles: list[list[int]] = []
    while unseen:
        first = min(unseen)
        cycle: list[int] = []
        current = first
        while current not in cycle:
            if current not in unseen:
                raise ValueError("goal mapping is not a disjoint permutation")
            cycle.append(current)
            unseen.remove(current)
            next_agent = successor[current]
            if next_agent is None:
                raise ValueError("routing path was not reached from a source-only head")
            current = next_agent
        if current != first or len(cycle) < 2:
            raise ValueError("invalid permutation cycle")
        cycles.append(cycle)
    return paths, cycles


def _cycle_rotation_witness(
    grid: Sequence[str],
    starts: Sequence[tuple[int, int]],
    goals: Sequence[tuple[int, int]],
    *,
    cut: set[tuple[int, int]],
    partition_cores: Sequence[set[tuple[int, int]]],
    partition_buffers: Sequence[tuple[int, int]],
    partition_membership: Mapping[tuple[int, int], int],
    require_cut_crossing: bool,
) -> dict[str, Any]:
    """Rotate permutation cycles through empty connected routing cores.

    Only one agent moves per tick.  All other agents remain on off-core parking
    cells, so every interior core cell is empty at each segment boundary.  A
    cycle first moves its last agent to an empty buffer, shifts the remaining
    agents into their registered goals in reverse order, and finally moves the
    buffered agent into the sole empty parking cell.
    """

    free = _free_cells(grid)
    positions = list(starts)
    occupancy = {cell: index for index, cell in enumerate(positions)}
    fixed: set[int] = set()
    events: list[dict[str, Any]] = []
    goal_reservations: list[dict[str, int]] = []
    traversed_cut = [False] * len(starts)
    paths, cycles = _permutation_paths_and_cycles(starts, goals)
    combined_core = set().union(*partition_cores, cut)
    structured_gate_by_agent: dict[int, tuple[int, int]] = {}
    structured_direction_by_agent: dict[int, str] = {}
    structured_route_core_by_agent: dict[int, set[tuple[int, int]]] = {}
    if require_cut_crossing:
        gates = sorted(cut)
        direction_groups: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
        for agent, (start, goal) in enumerate(zip(starts, goals)):
            direction = (
                int(partition_membership[start]),
                int(partition_membership[goal]),
            )
            direction_groups[direction].append(agent)
        if (
            len(gates) != 2
            or set(direction_groups) != {(0, 1), (1, 0)}
            or any(len(agents) != PARTITION_FLOW_COUNT for agents in direction_groups.values())
        ):
            raise ValueError("structured gate balancing requires 60 agents per direction")
        for direction in ((0, 1), (1, 0)):
            for rank, agent in enumerate(sorted(direction_groups[direction])):
                gate = gates[0 if rank < PARTITION_FLOW_COUNT // 2 else 1]
                structured_gate_by_agent[agent] = gate
                structured_direction_by_agent[agent] = f"{direction[0]}_to_{direction[1]}"
                structured_route_core_by_agent[agent] = set().union(
                    *partition_cores, {gate}
                )

    def move_segment(agent: int, destination: tuple[int, int], core: set[tuple[int, int]]) -> None:
        source = positions[agent]
        if destination in occupancy:
            raise ValueError("cycle rotation destination is occupied")
        route_core = structured_route_core_by_agent.get(agent, core)
        path = _core_route(source, destination, route_core)
        for next_cell in path[1:]:
            previous = positions[agent]
            if next_cell not in free or next_cell in occupancy:
                raise ValueError("cycle route is not collision free")
            occupancy.pop(previous)
            occupancy[next_cell] = agent
            positions[agent] = next_cell
            traversed_cut[agent] = (
                traversed_cut[agent] or previous in cut or next_cell in cut
            )
            events.append(
                {
                    "time": len(events),
                    "agent_index": agent,
                    "from": list(previous),
                    "to": list(next_cell),
                }
            )

    path_records: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths):
        path_partition = int(partition_membership[starts[path[0]]])
        if require_cut_crossing:
            core = combined_core
        else:
            if any(
                int(partition_membership[starts[agent]]) != path_partition
                or int(partition_membership[goals[agent]]) != path_partition
                for agent in path
            ):
                raise ValueError("diagnostic transfer path crosses partition")
            core = partition_cores[path_partition]
        for offset in range(len(path) - 1, -1, -1):
            agent = path[offset]
            move_segment(agent, goals[agent], core)
        reservation_time = len(events)
        for agent in sorted(path):
            if positions[agent] != goals[agent]:
                raise AssertionError("path rotation did not place registered goal")
            fixed.add(agent)
            goal_reservations.append(
                {"time": reservation_time, "agent_index": agent}
            )
        path_records.append(
            {
                "path_index": path_index,
                "agent_indices": path,
                "reservation_time": reservation_time,
            }
        )

    cycle_records: list[dict[str, Any]] = []
    for cycle_index, cycle in enumerate(cycles):
        cycle_partition = int(partition_membership[starts[cycle[0]]])
        if require_cut_crossing:
            core = combined_core
            buffer = partition_buffers[cycle_partition]
        else:
            if any(
                int(partition_membership[starts[agent]]) != cycle_partition
                for agent in cycle
            ):
                raise ValueError("diagnostic permutation cycle crosses partition")
            core = partition_cores[cycle_partition]
            routing_cells = {
                cell
                for cell, index in partition_membership.items()
                if int(index) == cycle_partition
                and cell not in core
                and any(neighbor in core for neighbor in _neighbors(cell))
            }
            empty_routing_cells = sorted(routing_cells - set(occupancy))
            if not empty_routing_cells:
                raise ValueError("diagnostic cycle has no empty off-core buffer")
            buffer = empty_routing_cells[0]
        if buffer in occupancy:
            raise ValueError("registered routing buffer is occupied")
        last = cycle[-1]
        move_segment(last, buffer, core)
        for offset in range(len(cycle) - 2, -1, -1):
            agent = cycle[offset]
            move_segment(agent, goals[agent], core)
        move_segment(last, goals[last], core)
        reservation_time = len(events)
        for agent in sorted(cycle):
            if positions[agent] != goals[agent]:
                raise AssertionError("cycle rotation did not place registered goal")
            fixed.add(agent)
            goal_reservations.append(
                {"time": reservation_time, "agent_index": agent}
            )
        cycle_records.append(
            {
                "cycle_index": cycle_index,
                "agent_indices": cycle,
                "buffer": list(buffer),
                "reservation_time": reservation_time,
            }
        )
    if require_cut_crossing and not all(traversed_cut):
        raise ValueError("structured cycle witness did not traverse registered cut")
    if not require_cut_crossing and any(traversed_cut):
        raise ValueError("diagnostic cycle witness traversed registered cut")
    return {
        "schema": SERIALIZED_WITNESS_SCHEMA,
        "semantics": "deterministic_cycle_rotation_over_connected_dominating_core",
        "collision_free_by_construction": True,
        "priority_goal_reservation": "permanent_after_completed_permutation_cycle",
        "event_count": len(events),
        "makespan": len(events),
        "move_events": events,
        "goal_reservations": goal_reservations,
        "transfer_paths": path_records,
        "permutation_cycles": cycle_records,
        "partition_buffers": [list(cell) for cell in partition_buffers],
        "partition_core_sha256": [
            _canonical_sha([list(cell) for cell in sorted(core)])
            for core in partition_cores
        ],
        "structured_gate_assignment": (
            [
                {
                    "agent_index": agent,
                    "direction": structured_direction_by_agent[agent],
                    "gate": list(structured_gate_by_agent[agent]),
                }
                for agent in range(len(starts))
            ]
            if require_cut_crossing
            else None
        ),
        "all_structured_agents_traversed_cut": (
            all(traversed_cut) if require_cut_crossing else None
        ),
        "diagnostic_cut_traversal_count": (
            sum(traversed_cut) if not require_cut_crossing else None
        ),
        "final_positions_sha256": _canonical_sha([list(cell) for cell in positions]),
    }


def _task_id(map_id: str, variant: str, task_seed: int) -> str:
    code = "bmcx" if variant == STRUCTURED_VARIANT else "dwpx"
    return f"{map_id}__{code}__t{int(task_seed):04d}__n{AGENT_COUNT:04d}"


def _variant_payload(
    *,
    map_id: str,
    variant: str,
    task_seed: int,
    starts: Sequence[tuple[int, int]],
    goals: Sequence[tuple[int, int]],
    distances: Sequence[int],
    witness: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "task_id": _task_id(map_id, variant, task_seed),
        "map_id": map_id,
        "variant": variant,
        "task_seed": int(task_seed),
        "agent_count": AGENT_COUNT,
        "starts": [list(cell) for cell in starts],
        "goals": [list(cell) for cell in goals],
        "shortest_path_distances": list(map(int, distances)),
        "distance_mean": sum(distances) / len(distances),
        "distance_p95": _p95(distances),
        "prioritized_joint_mapf_witness": dict(witness),
    }


def generate_compactcut_tasks(
    map_data: MapData, task_seed: int
) -> dict[str, Any]:
    """Generate the paired N=120 tasks and a joint feasibility witness."""

    task_seed = int(task_seed)
    if task_seed not in DEFAULT_TASK_SEEDS:
        raise ValueError(f"unregistered compact-cut task_seed: {task_seed}")
    geometry = audit_compactcut_geometry(map_data)
    if not geometry["passed"]:
        raise ValueError("compact-cut map failed geometry audit")
    registration = _routing_registration(map_data, geometry, task_seed)
    cut = set(registration["cut"])
    records = list(registration["records"])
    assignments = _registered_task_assignments(map_data, registration)
    structured = dict(assignments["structured"])
    diagnostic = dict(assignments["diagnostic"])

    cores = [set(record["core"]) for record in records]
    buffers = [tuple(record["buffer"]) for record in records]
    membership = {
        cell: partition_index
        for partition_index, partition in enumerate(registration["partitions"])
        for cell in partition
    }
    structured_witness = _cycle_rotation_witness(
        map_data.grid,
        structured["starts"],
        structured["goals"],
        cut=cut,
        partition_cores=cores,
        partition_buffers=buffers,
        partition_membership=membership,
        require_cut_crossing=True,
    )
    diagnostic_witness = _cycle_rotation_witness(
        map_data.grid,
        diagnostic["starts"],
        diagnostic["goals"],
        cut=cut,
        partition_cores=cores,
        partition_buffers=buffers,
        partition_membership=membership,
        require_cut_crossing=False,
    )
    variants = {
        STRUCTURED_VARIANT: _variant_payload(
            map_id=map_data.map_id,
            variant=STRUCTURED_VARIANT,
            task_seed=task_seed,
            starts=structured["starts"],
            goals=structured["goals"],
            distances=structured["distances"],
            witness=structured_witness,
        ),
        DIAGNOSTIC_VARIANT: _variant_payload(
            map_id=map_data.map_id,
            variant=DIAGNOSTIC_VARIANT,
            task_seed=task_seed,
            starts=diagnostic["starts"],
            goals=diagnostic["goals"],
            distances=diagnostic["distances"],
            witness=diagnostic_witness,
        ),
    }
    payload = {
        "schema": TASK_PAIR_SCHEMA,
        "pairing_semantics": "independent_registered_endpoints_distance_near_matched_secondary_control",
        "causal_control_claim": False,
        "control_interpretation": "within-partition diagnostic control",
        "map_id": map_data.map_id,
        "map_seed": map_data.seed,
        "map_grid_sha256": geometry["grid_sha256"],
        "task_seed": task_seed,
        "agent_count": AGENT_COUNT,
        "geometry": geometry,
        "routing_registration": _routing_registration_payload(registration),
        "distance_assignment": dict(assignments["distance_assignment"]),
        "variants": variants,
    }
    audit = audit_compactcut_tasks(map_data, payload)
    if not audit["passed"]:
        raise ValueError("constructed compact-cut task failed audit: " + ",".join(audit["errors"]))
    return payload


def _audit_serialized_witness(
    grid: Sequence[str],
    starts: Sequence[tuple[int, int]],
    goals: Sequence[tuple[int, int]],
    raw_witness: Mapping[str, Any],
    *,
    cut: set[tuple[int, int]],
    require_cut_crossing: bool,
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    witness = dict(raw_witness or {})
    if (
        witness.get("schema") != SERIALIZED_WITNESS_SCHEMA
        or witness.get("semantics")
        != "deterministic_cycle_rotation_over_connected_dominating_core"
        or witness.get("collision_free_by_construction") is not True
        or witness.get("priority_goal_reservation")
        != "permanent_after_completed_permutation_cycle"
    ):
        errors.append("witness_contract_mismatch")
    positions = list(starts)
    occupancy = {cell: index for index, cell in enumerate(positions)}
    fixed: set[int] = set()
    cut_traversed: set[int] = set()
    first_cut_cell: dict[int, tuple[int, int]] = {}
    raw_reservations = list(witness.get("goal_reservations") or ())
    reservations_by_time: dict[int, list[int]] = collections.defaultdict(list)
    for raw in raw_reservations:
        try:
            reservation = dict(raw or {})
            reservations_by_time[int(reservation.get("time", -1))].append(
                int(reservation.get("agent_index", -1))
            )
        except (TypeError, ValueError):
            errors.append("malformed_goal_reservation")
    events = list(witness.get("move_events") or ())
    for time, raw_event in enumerate(events):
        for agent in reservations_by_time.get(time, ()):
            if not 0 <= agent < len(starts) or positions[agent] != goals[agent] or agent in fixed:
                errors.append(f"time_{time}:invalid_goal_reservation")
            else:
                fixed.add(agent)
        event = dict(raw_event or {})
        try:
            agent = int(event.get("agent_index", -1))
            source = tuple(map(int, event.get("from") or ()))
            destination = tuple(map(int, event.get("to") or ()))
            recorded_time = int(event.get("time", -1))
        except (TypeError, ValueError):
            errors.append(f"time_{time}:malformed_event")
            continue
        if not 0 <= agent < len(starts) or agent in fixed:
            errors.append(f"time_{time}:agent_identity_or_goal_reservation")
            continue
        if recorded_time != time or source != positions[agent]:
            errors.append(f"time_{time}:source_or_clock")
            continue
        if (
            destination not in _free_cells(grid)
            or destination in occupancy
            or abs(source[0] - destination[0]) + abs(source[1] - destination[1]) != 1
        ):
            errors.append(f"time_{time}:vertex_collision_obstacle_or_non_unit_step")
            continue
        occupancy.pop(source)
        occupancy[destination] = agent
        positions[agent] = destination
        if source in cut or destination in cut:
            cut_traversed.add(agent)
            encountered = source if source in cut else destination
            first_cut_cell.setdefault(agent, encountered)
    for agent in reservations_by_time.get(len(events), ()):
        if not 0 <= agent < len(starts) or positions[agent] != goals[agent] or agent in fixed:
            errors.append("terminal_invalid_goal_reservation")
        else:
            fixed.add(agent)
    if set(reservations_by_time) - set(range(len(events) + 1)):
        errors.append("goal_reservation_time_out_of_range")
    if fixed != set(range(len(starts))):
        errors.append("witness_goal_reservation_coverage")
    if positions != list(goals):
        errors.append("witness_final_positions")
    if require_cut_crossing and cut_traversed != set(range(len(starts))):
        errors.append("witness_structured_cut_coverage")
    if not require_cut_crossing and cut_traversed:
        errors.append("witness_diagnostic_cut_traversal")
    if int(witness.get("event_count", -1)) != len(events):
        errors.append("witness_event_count")
    if int(witness.get("makespan", -1)) != len(events):
        errors.append("witness_makespan")
    if witness.get("final_positions_sha256") != _canonical_sha(
        [list(cell) for cell in positions]
    ):
        errors.append("witness_final_positions_sha256")
    structured_gate_direction_counts: dict[str, int] | None = None
    if require_cut_crossing:
        partitions = _components(grid, cut)
        membership = {
            cell: index
            for index, partition in enumerate(partitions)
            for cell in partition
        }
        gates = sorted(cut)
        structured_gate_direction_counts = {
            f"{source}_to_{destination}@{gate[0]},{gate[1]}": 0
            for source, destination in ((0, 1), (1, 0))
            for gate in gates
        }
        for agent, (start, goal) in enumerate(zip(starts, goals)):
            direction = (membership.get(start, -1), membership.get(goal, -1))
            gate = first_cut_cell.get(agent)
            key = (
                f"{direction[0]}_to_{direction[1]}@{gate[0]},{gate[1]}"
                if gate is not None
                else "missing"
            )
            if key in structured_gate_direction_counts:
                structured_gate_direction_counts[key] += 1
        if (
            len(structured_gate_direction_counts) != 4
            or any(count != PARTITION_FLOW_COUNT // 2 for count in structured_gate_direction_counts.values())
        ):
            errors.append("witness_structured_gate_direction_balance")
    return errors, {
        "event_count": len(events),
        "makespan": len(events),
        "goal_reservation_count": len(fixed),
        "all_structured_agents_traversed_cut": (
            len(cut_traversed) == len(starts) if require_cut_crossing else None
        ),
        "diagnostic_cut_traversal_count": (
            len(cut_traversed) if not require_cut_crossing else None
        ),
        "structured_gate_direction_counts": structured_gate_direction_counts,
        "collision_free": not errors,
    }


def audit_compactcut_tasks(
    map_data: MapData, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute pair identity, cut semantics, distances, and joint witnesses."""

    errors: list[str] = []
    geometry = audit_compactcut_geometry(map_data)
    if not geometry["passed"]:
        return {
            "schema": TASK_AUDIT_SCHEMA,
            "passed": False,
            "errors": [f"geometry:{item}" for item in geometry["errors"]],
        }
    selected = dict(geometry["selected_cut"])
    cut = {tuple(cell) for cell in selected["cut_cells"]}
    partitions = _components(map_data.grid, cut)
    membership = {
        cell: index for index, partition in enumerate(partitions) for cell in partition
    }
    if payload.get("schema") != TASK_PAIR_SCHEMA:
        errors.append("task_pair_schema_mismatch")
    if (
        payload.get("pairing_semantics")
        != "independent_registered_endpoints_distance_near_matched_secondary_control"
        or payload.get("causal_control_claim") is not False
        or payload.get("control_interpretation")
        != "within-partition diagnostic control"
    ):
        errors.append("pairing_contract_mismatch")
    if (
        payload.get("map_id") != map_data.map_id
        or int(payload.get("map_seed", -1)) != int(map_data.seed)
        or payload.get("map_grid_sha256") != geometry["grid_sha256"]
        or int(payload.get("agent_count", -1)) != AGENT_COUNT
        or dict(payload.get("geometry") or {}) != geometry
    ):
        errors.append("pair_identity_mismatch")
    try:
        task_seed = int(payload.get("task_seed", -1))
    except (TypeError, ValueError):
        task_seed = -1
        errors.append("task_seed_malformed")

    expected_variants: dict[str, dict[str, Any]] = {}
    if task_seed not in DEFAULT_TASK_SEEDS:
        errors.append("unregistered_task_seed")
    else:
        try:
            expected_registration = _routing_registration(
                map_data, geometry, task_seed
            )
            expected_assignments = _registered_task_assignments(
                map_data, expected_registration
            )
            if dict(payload.get("routing_registration") or {}) != (
                _routing_registration_payload(expected_registration)
            ):
                errors.append("routing_registration_mismatch")
            if dict(payload.get("distance_assignment") or {}) != dict(
                expected_assignments["distance_assignment"]
            ):
                errors.append("distance_assignment_mismatch")
            expected_records = list(expected_registration["records"])
            expected_cores = [set(record["core"]) for record in expected_records]
            expected_buffers = [tuple(record["buffer"]) for record in expected_records]
            expected_membership = {
                cell: partition_index
                for partition_index, partition in enumerate(
                    expected_registration["partitions"]
                )
                for cell in partition
            }
            for expected_variant, assignment_key in (
                (STRUCTURED_VARIANT, "structured"),
                (DIAGNOSTIC_VARIANT, "diagnostic"),
            ):
                expected_assignment = dict(expected_assignments[assignment_key])
                expected_variants[expected_variant] = {
                    "starts": list(expected_assignment["starts"]),
                    "goals": list(expected_assignment["goals"]),
                    "distances": list(expected_assignment["distances"]),
                    "witness": _cycle_rotation_witness(
                        map_data.grid,
                        expected_assignment["starts"],
                        expected_assignment["goals"],
                        cut=set(expected_registration["cut"]),
                        partition_cores=expected_cores,
                        partition_buffers=expected_buffers,
                        partition_membership=expected_membership,
                        require_cut_crossing=expected_variant == STRUCTURED_VARIANT,
                    ),
                }
        except (AssertionError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"registered_construction_failed:{type(exc).__name__}")

    variants = dict(payload.get("variants") or {})
    parsed: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    free = _free_cells(map_data.grid)
    for variant in TASK_VARIANTS:
        task = dict(variants.get(variant) or {})
        try:
            starts = [tuple(map(int, cell)) for cell in task.get("starts") or ()]
            goals = [tuple(map(int, cell)) for cell in task.get("goals") or ()]
            recorded_distances = list(
                map(int, task.get("shortest_path_distances") or ())
            )
        except (TypeError, ValueError):
            starts, goals, recorded_distances = [], [], []
            errors.append(f"{variant}:malformed_endpoints")
        if (
            task.get("task_id") != _task_id(map_data.map_id, variant, task_seed)
            or task.get("map_id") != map_data.map_id
            or task.get("variant") != variant
            or int(task.get("task_seed", -1)) != task_seed
            or int(task.get("agent_count", -1)) != AGENT_COUNT
        ):
            errors.append(f"{variant}:identity_mismatch")
        expected = expected_variants.get(variant)
        if expected is not None and (
            starts != expected["starts"]
            or goals != expected["goals"]
            or recorded_distances != expected["distances"]
        ):
            errors.append(f"{variant}:registered_endpoint_or_assignment_mismatch")
        endpoints_valid = (
            len(starts) == AGENT_COUNT
            and len(goals) == AGENT_COUNT
            and len(set(starts)) == AGENT_COUNT
            and len(set(goals)) == AGENT_COUNT
            and all(start != goal for start, goal in zip(starts, goals))
            and all(cell in free and cell not in cut for cell in starts + goals)
        )
        if not endpoints_valid:
            errors.append(f"{variant}:endpoint_contract")
        distances: list[int] = []
        cross_count = 0
        within_count = 0
        if endpoints_valid:
            for start, goal in zip(starts, goals):
                start_partition = membership.get(start, -1)
                goal_partition = membership.get(goal, -2)
                if start_partition == goal_partition:
                    within_count += 1
                else:
                    cross_count += 1
                distance = _distance_map(map_data.grid, start).get(goal)
                if distance is None:
                    errors.append(f"{variant}:unreachable_pair")
                    break
                distances.append(distance)
        if distances != recorded_distances:
            errors.append(f"{variant}:shortest_distance_mismatch")
        if variant == STRUCTURED_VARIANT and (
            cross_count != AGENT_COUNT
            or sum(membership.get(start) == 0 for start in starts)
            != PARTITION_FLOW_COUNT
            or sum(membership.get(goal) == 0 for goal in goals)
            != PARTITION_FLOW_COUNT
        ):
            errors.append(f"{variant}:mandatory_bidirectional_cut_flow")
        if variant == DIAGNOSTIC_VARIANT and within_count != AGENT_COUNT:
            errors.append(f"{variant}:not_within_partition")
        if endpoints_valid and any(
            sum(membership.get(cell) == partition_index for cell in starts)
            != PARTITION_FLOW_COUNT
            or sum(membership.get(cell) == partition_index for cell in goals)
            != PARTITION_FLOW_COUNT
            for partition_index in range(2)
        ):
            errors.append(f"{variant}:partition_endpoint_balance")
        distance_mean = sum(distances) / len(distances) if distances else None
        distance_p95 = _p95(distances) if distances else None
        if (
            distance_mean != task.get("distance_mean")
            or distance_p95 != task.get("distance_p95")
        ):
            errors.append(f"{variant}:distance_summary_mismatch")
        raw_witness = dict(task.get("prioritized_joint_mapf_witness") or {})
        if expected is not None and raw_witness != expected["witness"]:
            errors.append(f"{variant}:deterministic_witness_mismatch")
        witness_errors, witness_metrics = _audit_serialized_witness(
            map_data.grid,
            starts,
            goals,
            raw_witness,
            cut=cut,
            require_cut_crossing=variant == STRUCTURED_VARIANT,
        )
        errors.extend(f"{variant}:{item}" for item in witness_errors)
        parsed[variant] = {"starts": starts, "goals": goals, "distances": distances}
        metrics[variant] = {
            "cross_partition_count": cross_count,
            "within_partition_count": within_count,
            "distance_mean": distance_mean,
            "distance_p95": distance_p95,
            "prioritized_joint_mapf_witness": witness_metrics,
        }

    if all(variant in parsed for variant in TASK_VARIANTS):
        structured = parsed[STRUCTURED_VARIANT]
        diagnostic = parsed[DIAGNOSTIC_VARIANT]
        structured_mean = metrics[STRUCTURED_VARIANT]["distance_mean"]
        diagnostic_mean = metrics[DIAGNOSTIC_VARIANT]["distance_mean"]
        structured_p95 = metrics[STRUCTURED_VARIANT]["distance_p95"]
        diagnostic_p95 = metrics[DIAGNOSTIC_VARIANT]["distance_p95"]
        mean_delta = (
            abs(diagnostic_mean - structured_mean) / structured_mean
            if structured_mean and diagnostic_mean is not None
            else math.inf
        )
        p95_delta = (
            abs(diagnostic_p95 - structured_p95) / structured_p95
            if structured_p95 and diagnostic_p95 is not None
            else math.inf
        )
        metrics["paired_distance"] = {
            "mean_relative_difference": mean_delta,
            "p95_relative_difference": p95_delta,
            "mean_gate_maximum": DISTANCE_MEAN_RELATIVE_GATE,
            "p95_gate_maximum": DISTANCE_P95_RELATIVE_GATE,
        }
        if mean_delta > DISTANCE_MEAN_RELATIVE_GATE:
            errors.append("paired_distance_mean_gate_failed")
        if p95_delta > DISTANCE_P95_RELATIVE_GATE:
            errors.append("paired_distance_p95_gate_failed")

    return {
        "schema": TASK_AUDIT_SCHEMA,
        "passed": not errors,
        "errors": errors,
        "map_id": map_data.map_id,
        "map_grid_sha256": geometry["grid_sha256"],
        "task_seed": task_seed,
        "agent_count": AGENT_COUNT,
        "geometry": geometry,
        "metrics": metrics,
        "control_interpretation": "within-partition diagnostic control",
        "causal_control_claim": False,
    }


__all__ = [
    "AGENT_COUNT",
    "DEFAULT_TASK_SEEDS",
    "DIAGNOSTIC_VARIANT",
    "DISTANCE_MEAN_RELATIVE_GATE",
    "DISTANCE_P95_RELATIVE_GATE",
    "GEOMETRY_AUDIT_SCHEMA",
    "MAP_BANK_SCHEMA",
    "MAP_SPECS",
    "PARTITION_FLOW_COUNT",
    "SERIALIZED_WITNESS_SCHEMA",
    "STRUCTURED_VARIANT",
    "TASK_AUDIT_SCHEMA",
    "TASK_PAIR_SCHEMA",
    "TASK_VARIANTS",
    "audit_compactcut_geometry",
    "audit_compactcut_tasks",
    "generate_compactcut_map_bank",
    "generate_compactcut_tasks",
]
