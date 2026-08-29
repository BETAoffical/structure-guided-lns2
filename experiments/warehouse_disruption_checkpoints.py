"""Controller-independent Warehouse path-delay checkpoint helpers.

The functions in this module are deliberately pure: they do not import the
native solver, inspect controller traces, or write files.  A caller supplies a
complete feasible path set together with frozen Warehouse semantic metadata.
The module then deterministically selects a preregistered fraction of agents at
one global time, inserts fixed waits near station/intersection hotspots, and
returns the evidence needed to materialize and qualify a native checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any

from experiments.state_analysis import reconstruct_conflicts


CHECKPOINT_SCHEMA = "lns2.warehouse_disruption_checkpoint.v1"
DISTURBANCE_SCHEMA = "lns2.warehouse_global_hotspot_delay.v1"
HOTSPOT_SCHEMA = "lns2.warehouse_station_intersection_hotspot.v1"
QUALIFICATION_GATE_ID = "lns2.warehouse_disruption_gate.16-32-16.v1"

MINIMUM_DELAY_FRACTION = 0.10
MAXIMUM_DELAY_FRACTION = 0.15
SUPPORTED_DELAY_TICKS = (2, 3, 4)
DEFAULT_GATE_THRESHOLDS = {
    "minimum_conflict_pair_count": 16,
    "minimum_active_conflict_agent_count": 32,
    "minimum_largest_conflict_component_size": 16,
}

_BLOCKED_SEMANTIC_TYPES = frozenset({"@", "W", "N"})


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _registered_sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return normalized


def _semantic_shape(semantic_cell_types: Sequence[str]) -> tuple[int, int]:
    rows = list(semantic_cell_types)
    if not rows or not all(isinstance(row, str) and row for row in rows):
        raise ValueError("semantic_cell_types must be non-empty strings")
    cols = len(rows[0])
    if any(len(row) != cols for row in rows):
        raise ValueError("semantic_cell_types must be rectangular")
    return len(rows), cols


def _coordinate(value: Any, rows: int, cols: int, *, label: str) -> tuple[int, int]:
    if type(value) is int:
        if not 0 <= value < rows * cols:
            raise ValueError(f"{label} cell id is outside the grid")
        return divmod(value, cols)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(type(item) is int for item in value)
    ):
        row, col = int(value[0]), int(value[1])
        if not (0 <= row < rows and 0 <= col < cols):
            raise ValueError(f"{label} coordinate is outside the grid")
        return row, col
    raise ValueError(f"{label} must be a row-major cell id or [row, col]")


def _normalize_complete_paths(
    paths: Sequence[Sequence[Any]], rows: int, cols: int
) -> list[list[int]]:
    if isinstance(paths, (str, bytes)) or not paths:
        raise ValueError("paths must contain at least one agent path")
    normalized: list[list[int]] = []
    for agent_id, raw_path in enumerate(paths):
        if isinstance(raw_path, (str, bytes)) or not raw_path:
            raise ValueError(f"agent {agent_id} has an empty path")
        path = []
        for time, value in enumerate(raw_path):
            row, col = _coordinate(
                value, rows, cols, label=f"agent {agent_id} time {time}"
            )
            path.append(row * cols + col)
        for previous, current in zip(path, path[1:]):
            old_row, old_col = divmod(previous, cols)
            new_row, new_col = divmod(current, cols)
            if abs(old_row - new_row) + abs(old_col - new_col) > 1:
                raise ValueError(f"agent {agent_id} path contains a non-local move")
        normalized.append(path)
    return normalized


def warehouse_hotspot_definition(
    station_zones: Mapping[str, Sequence[Any]],
    semantic_cell_types: Sequence[str],
    *,
    neighborhood_radius: int = 1,
) -> dict[str, Any]:
    """Return station-zone and semantic-X cells expanded by Manhattan radius."""

    rows, cols = _semantic_shape(semantic_cell_types)
    if type(neighborhood_radius) is not int or neighborhood_radius < 1:
        raise ValueError("neighborhood_radius must be a positive integer")
    if not isinstance(station_zones, Mapping) or not station_zones:
        raise ValueError("station_zones must be a non-empty mapping")

    normalized_zones: dict[str, list[list[int]]] = {}
    station_cells: set[tuple[int, int]] = set()
    for station_id in sorted(map(str, station_zones)):
        values = station_zones[station_id]
        if isinstance(values, (str, bytes)) or not values:
            raise ValueError(f"station zone {station_id} is empty")
        cells = {
            _coordinate(value, rows, cols, label=f"station zone {station_id}")
            for value in values
        }
        for row, col in cells:
            if semantic_cell_types[row][col] in _BLOCKED_SEMANTIC_TYPES:
                raise ValueError(f"station zone {station_id} contains a blocked cell")
        normalized_zones[station_id] = [list(cell) for cell in sorted(cells)]
        station_cells.update(cells)

    semantic_x_cells = {
        (row, col)
        for row, line in enumerate(semantic_cell_types)
        for col, value in enumerate(line)
        if value == "X"
    }
    base_cells = station_cells | semantic_x_cells
    if not base_cells:
        raise ValueError("Warehouse hotspot definition has no station or X cells")

    hotspot_cells: set[tuple[int, int]] = set()
    for base_row, base_col in base_cells:
        for row in range(
            max(0, base_row - neighborhood_radius),
            min(rows, base_row + neighborhood_radius + 1),
        ):
            remaining = neighborhood_radius - abs(row - base_row)
            for col in range(max(0, base_col - remaining), min(cols, base_col + remaining + 1)):
                if semantic_cell_types[row][col] not in _BLOCKED_SEMANTIC_TYPES:
                    hotspot_cells.add((row, col))

    result = {
        "schema": HOTSPOT_SCHEMA,
        "definition": "station_zones_union_semantic_X_manhattan_neighborhood",
        "rows": rows,
        "cols": cols,
        "neighborhood_radius": neighborhood_radius,
        "station_zone_count": len(normalized_zones),
        "station_zone_cell_count": len(station_cells),
        "semantic_x_cell_count": len(semantic_x_cells),
        "base_cell_count": len(base_cells),
        "hotspot_cell_count": len(hotspot_cells),
        "station_zones_sha256": _canonical_sha256(normalized_zones),
        "semantic_cell_types_sha256": _canonical_sha256(list(semantic_cell_types)),
        "station_zone_cells": [list(cell) for cell in sorted(station_cells)],
        "semantic_x_cells": [list(cell) for cell in sorted(semantic_x_cells)],
        "hotspot_cells": [list(cell) for cell in sorted(hotspot_cells)],
    }
    result["hotspot_definition_sha256"] = _canonical_sha256(result)
    return result


def summarize_conflict_graph(
    conflict_edges: Sequence[Sequence[int]],
    *,
    agent_count: int,
    conflict_event_count: int | None = None,
    vertex_conflict_event_count: int | None = None,
    edge_conflict_event_count: int | None = None,
) -> dict[str, Any]:
    """Summarize unique pair edges and their active connected components."""

    if type(agent_count) is not int or agent_count <= 0:
        raise ValueError("agent_count must be positive")
    edges: list[tuple[int, int]] = []
    for raw_edge in conflict_edges:
        if (
            isinstance(raw_edge, (str, bytes))
            or len(raw_edge) != 2
            or any(type(value) is not int for value in raw_edge)
        ):
            raise ValueError("conflict edges must contain two integer agent ids")
        left, right = sorted((int(raw_edge[0]), int(raw_edge[1])))
        if left == right or not (0 <= left < agent_count and 0 <= right < agent_count):
            raise ValueError("conflict edge contains an invalid agent id")
        edges.append((left, right))
    if len(edges) != len(set(edges)):
        raise ValueError("conflict_edges contains duplicates")
    edges.sort()

    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(adjacency)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component = {start}
        queue: deque[int] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(adjacency[current]):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda members: (-len(members), members))
    active_count = len(adjacency)

    def _optional_count(value: int | None, label: str) -> int | None:
        if value is None:
            return None
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} must be a nonnegative integer")
        return value

    event_count = _optional_count(conflict_event_count, "conflict_event_count")
    vertex_count = _optional_count(
        vertex_conflict_event_count, "vertex_conflict_event_count"
    )
    edge_count = _optional_count(edge_conflict_event_count, "edge_conflict_event_count")
    if (
        event_count is not None
        and vertex_count is not None
        and edge_count is not None
        and event_count != vertex_count + edge_count
    ):
        raise ValueError("conflict event type counts do not sum to the total")
    denominator = agent_count * (agent_count - 1)
    return {
        "agent_count": agent_count,
        "conflict_pair_count": len(edges),
        "initial_conflicts": len(edges),
        "conflict_event_count": event_count,
        "vertex_conflict_event_count": vertex_count,
        "edge_conflict_event_count": edge_count,
        "conflict_pair_density": (2.0 * len(edges) / denominator if denominator else 0.0),
        "active_conflict_agent_count": active_count,
        "active_conflict_agent_ratio": active_count / agent_count,
        "conflict_component_count": len(components),
        "conflict_component_sizes": [len(component) for component in components],
        "largest_conflict_component_size": len(components[0]) if components else 0,
        "largest_conflict_component_ratio": (
            len(components[0]) / agent_count if components else 0.0
        ),
        "conflict_edges": [list(edge) for edge in edges],
    }


def summarize_path_conflicts(paths: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Reconstruct vertex/edge conflicts using the repository's path semantics."""

    if isinstance(paths, (str, bytes)) or not paths:
        raise ValueError("paths must contain at least one agent path")
    agents = []
    for agent_id, path in enumerate(paths):
        if isinstance(path, (str, bytes)) or not path:
            raise ValueError(f"agent {agent_id} has an empty path")
        if any(type(cell) is not int for cell in path):
            raise ValueError("summarize_path_conflicts requires integer cell ids")
        agents.append({"id": agent_id, "path": list(path)})
    events = reconstruct_conflicts(agents)
    event_types = Counter(event.kind for event in events)
    pairs = sorted({(event.left, event.right) for event in events})
    return summarize_conflict_graph(
        pairs,
        agent_count=len(paths),
        conflict_event_count=len(events),
        vertex_conflict_event_count=int(event_types["vertex"]),
        edge_conflict_event_count=int(event_types["edge"]),
    )


def warehouse_checkpoint_gate(
    conflict_summary: Mapping[str, Any],
    *,
    minimum_conflict_pair_count: int = 16,
    minimum_active_conflict_agent_count: int = 32,
    minimum_largest_conflict_component_size: int = 16,
) -> dict[str, Any]:
    """Evaluate the preregistered 16/32/16 disturbed-state supply gate."""

    thresholds = {
        "minimum_conflict_pair_count": minimum_conflict_pair_count,
        "minimum_active_conflict_agent_count": minimum_active_conflict_agent_count,
        "minimum_largest_conflict_component_size": minimum_largest_conflict_component_size,
    }
    if any(type(value) is not int or value <= 0 for value in thresholds.values()):
        raise ValueError("checkpoint gate thresholds must be positive integers")
    observed = {
        "conflict_pair_count": int(conflict_summary.get("conflict_pair_count", -1)),
        "active_conflict_agent_count": int(
            conflict_summary.get("active_conflict_agent_count", -1)
        ),
        "largest_conflict_component_size": int(
            conflict_summary.get("largest_conflict_component_size", -1)
        ),
    }
    if any(value < 0 for value in observed.values()):
        raise ValueError("conflict_summary lacks nonnegative gate metrics")
    failures = [
        metric
        for metric, threshold_key in (
            ("conflict_pair_count", "minimum_conflict_pair_count"),
            ("active_conflict_agent_count", "minimum_active_conflict_agent_count"),
            (
                "largest_conflict_component_size",
                "minimum_largest_conflict_component_size",
            ),
        )
        if observed[metric] < thresholds[threshold_key]
    ]
    return {
        "gate_id": QUALIFICATION_GATE_ID,
        "passed": not failures,
        "thresholds": thresholds,
        "observed": observed,
        "failed_metrics": failures,
        "controller_outcomes_consulted": False,
    }


def _selection_count(agent_count: int, requested_fraction: float) -> int:
    if (
        isinstance(requested_fraction, bool)
        or not isinstance(requested_fraction, (int, float))
        or not math.isfinite(float(requested_fraction))
        or not MINIMUM_DELAY_FRACTION
        <= float(requested_fraction)
        <= MAXIMUM_DELAY_FRACTION
    ):
        raise ValueError("delayed_agent_fraction must be between 0.10 and 0.15")
    minimum = math.ceil(agent_count * MINIMUM_DELAY_FRACTION)
    maximum = math.floor(agent_count * MAXIMUM_DELAY_FRACTION)
    if minimum > maximum:
        raise ValueError("agent_count is too small for an exact 10-15% selection")
    nearest = math.floor(agent_count * float(requested_fraction) + 0.5)
    return min(max(nearest, minimum), maximum)


def inject_global_hotspot_delays(
    paths: Sequence[Sequence[Any]],
    *,
    semantic_cell_types: Sequence[str],
    station_zones: Mapping[str, Sequence[Any]],
    global_time: int,
    delay_ticks: int,
    delayed_agent_fraction: float,
    selection_seed: int,
    neighborhood_radius: int = 1,
) -> dict[str, Any]:
    """Insert fixed waits for a deterministic hotspot-local agent sample.

    The source paths must be collision-free under goal-waiting MAPF semantics.
    Selection depends only on the registered inputs, path occupancy at the
    registered global time, and ``selection_seed``.  No controller result is an
    input to this function.
    """

    rows, cols = _semantic_shape(semantic_cell_types)
    normalized = _normalize_complete_paths(paths, rows, cols)
    source_conflicts = summarize_path_conflicts(normalized)
    if source_conflicts["conflict_pair_count"] != 0:
        raise ValueError("source paths must be feasible and conflict-free")
    if type(global_time) is not int or global_time < 0:
        raise ValueError("global_time must be a nonnegative integer")
    if type(delay_ticks) is not int or delay_ticks not in SUPPORTED_DELAY_TICKS:
        raise ValueError("delay_ticks must be one of 2, 3, or 4")
    if type(selection_seed) is not int or selection_seed < 0:
        raise ValueError("selection_seed must be a nonnegative integer")

    hotspot = warehouse_hotspot_definition(
        station_zones,
        semantic_cell_types,
        neighborhood_radius=neighborhood_radius,
    )
    hotspot_ids = {
        int(cell[0]) * cols + int(cell[1]) for cell in hotspot["hotspot_cells"]
    }
    eligible = [
        agent_id
        for agent_id, path in enumerate(normalized)
        if global_time < len(path) - 1 and path[global_time] in hotspot_ids
    ]
    target_count = _selection_count(len(normalized), float(delayed_agent_fraction))
    if len(eligible) < target_count:
        raise ValueError(
            "registered global time has fewer hotspot agents than the delayed fraction"
        )

    seed_material = {
        "schema": DISTURBANCE_SCHEMA,
        "selection_seed": selection_seed,
        "global_time": global_time,
        "delay_ticks": delay_ticks,
        "delayed_agent_fraction": float(delayed_agent_fraction),
        "source_paths_sha256": _canonical_sha256(normalized),
        "hotspot_definition_sha256": hotspot["hotspot_definition_sha256"],
    }
    random_seed = int.from_bytes(
        hashlib.sha256(
            json.dumps(seed_material, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).digest()[:16],
        "big",
    )
    rng = random.Random(random_seed)
    selected = sorted(rng.sample(sorted(eligible), target_count))
    selected_set = set(selected)
    disturbed = []
    for agent_id, path in enumerate(normalized):
        if agent_id not in selected_set:
            disturbed.append(list(path))
            continue
        disturbed.append(
            list(path[: global_time + 1])
            + [path[global_time]] * delay_ticks
            + list(path[global_time + 1 :])
        )

    conflict_summary = summarize_path_conflicts(disturbed)
    qualification = warehouse_checkpoint_gate(conflict_summary)
    result = {
        "schema": DISTURBANCE_SCHEMA,
        "path_encoding": "row_major_cell_id",
        "rows": rows,
        "cols": cols,
        "agent_count": len(normalized),
        "global_time": global_time,
        "delay_ticks": delay_ticks,
        "delayed_agent_fraction_requested": float(delayed_agent_fraction),
        "delayed_agent_count": target_count,
        "delayed_agent_fraction_realized": target_count / len(normalized),
        "selection_seed": selection_seed,
        "selection_seed_domain_sha256": _canonical_sha256(seed_material),
        "selection_semantics": (
            "preregistered_seed_sample_from_agents_present_in_frozen_hotspot_at_"
            "global_time_without_controller_outcomes"
        ),
        "eligible_agent_count": len(eligible),
        "eligible_agent_ids": sorted(eligible),
        "eligible_agent_ids_sha256": _canonical_sha256(sorted(eligible)),
        "selected_agent_ids": selected,
        "selected_agent_ids_sha256": _canonical_sha256(selected),
        "hotspot_definition": hotspot,
        "source_paths_sha256": _canonical_sha256(normalized),
        "disturbed_paths_sha256": _canonical_sha256(disturbed),
        "source_conflicts": source_conflicts,
        "conflicts": conflict_summary,
        "qualification": qualification,
        "controller_outcomes_consulted": False,
        "paths": disturbed,
    }
    return result


def compute_checkpoint_identity_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash a complete checkpoint manifest row, excluding its self-hash field."""

    payload = dict(manifest)
    payload.pop("checkpoint_identity_sha256", None)
    return _canonical_sha256(payload)


def checkpoint_manifest_fields(
    *,
    map_id: str,
    task_id: str,
    map_sha256: str,
    task_sha256: str,
    source_paths: Sequence[Sequence[Any]],
    disturbance: Mapping[str, Any],
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    """Build the immutable, pre-native manifest row for one disturbed path set."""

    if not str(map_id) or not str(task_id):
        raise ValueError("map_id and task_id must be non-empty")
    if disturbance.get("schema") != DISTURBANCE_SCHEMA:
        raise ValueError("disturbance has an unsupported schema")
    if disturbance.get("controller_outcomes_consulted") is not False:
        raise ValueError("disturbance must be controller-outcome blind")
    rows = int(disturbance.get("rows", 0))
    cols = int(disturbance.get("cols", 0))
    normalized_source = _normalize_complete_paths(source_paths, rows, cols)
    disturbed_paths = _normalize_complete_paths(
        list(disturbance.get("paths") or ()), rows, cols
    )
    if len(normalized_source) != len(disturbed_paths):
        raise ValueError("source and disturbed path counts differ")
    if _canonical_sha256(normalized_source) != disturbance.get("source_paths_sha256"):
        raise ValueError("source paths differ from the disturbance evidence")
    if _canonical_sha256(disturbed_paths) != disturbance.get("disturbed_paths_sha256"):
        raise ValueError("disturbed paths differ from the disturbance evidence")

    hotspot = dict(disturbance.get("hotspot_definition") or {})
    compact_disturbance = {
        key: disturbance[key]
        for key in (
            "schema",
            "global_time",
            "delay_ticks",
            "delayed_agent_fraction_requested",
            "delayed_agent_count",
            "delayed_agent_fraction_realized",
            "selection_seed",
            "selection_seed_domain_sha256",
            "selection_semantics",
            "eligible_agent_count",
            "eligible_agent_ids",
            "eligible_agent_ids_sha256",
            "selected_agent_ids",
            "selected_agent_ids_sha256",
            "controller_outcomes_consulted",
        )
    }
    compact_disturbance["hotspot"] = {
        key: hotspot[key]
        for key in (
            "schema",
            "definition",
            "neighborhood_radius",
            "station_zone_count",
            "station_zone_cell_count",
            "semantic_x_cell_count",
            "base_cell_count",
            "hotspot_cell_count",
            "station_zones_sha256",
            "semantic_cell_types_sha256",
            "hotspot_definition_sha256",
        )
    }
    base = {
        "schema": CHECKPOINT_SCHEMA,
        "map_id": str(map_id),
        "task_id": str(task_id),
        "map_sha256": _registered_sha256(map_sha256, "map_sha256"),
        "task_sha256": _registered_sha256(task_sha256, "task_sha256"),
        "path_encoding": "row_major_cell_id",
        "rows": rows,
        "cols": cols,
        "agent_count": len(normalized_source),
        "source_paths_sha256": str(disturbance["source_paths_sha256"]),
        "disturbed_paths_sha256": str(disturbance["disturbed_paths_sha256"]),
        "disturbance": compact_disturbance,
        "conflicts": dict(disturbance["conflicts"]),
        "qualification": dict(disturbance["qualification"]),
        "controller_outcomes_consulted": False,
        "native_solver_or_controller_invoked": False,
    }
    if checkpoint_id is None:
        identifier = f"warehouse-disruption-{_canonical_sha256(base)[:24]}"
    else:
        identifier = str(checkpoint_id)
        if not identifier or identifier.strip() != identifier:
            raise ValueError("checkpoint_id must be a non-empty trimmed string")
    result = {**base, "checkpoint_id": identifier}
    result["checkpoint_identity_sha256"] = compute_checkpoint_identity_sha256(result)
    return result


__all__ = [
    "CHECKPOINT_SCHEMA",
    "DEFAULT_GATE_THRESHOLDS",
    "DISTURBANCE_SCHEMA",
    "HOTSPOT_SCHEMA",
    "MAXIMUM_DELAY_FRACTION",
    "MINIMUM_DELAY_FRACTION",
    "QUALIFICATION_GATE_ID",
    "SUPPORTED_DELAY_TICKS",
    "checkpoint_manifest_fields",
    "compute_checkpoint_identity_sha256",
    "inject_global_hotspot_delays",
    "summarize_conflict_graph",
    "summarize_path_conflicts",
    "warehouse_checkpoint_gate",
    "warehouse_hotspot_definition",
]
