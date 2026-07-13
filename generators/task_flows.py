from __future__ import annotations

import collections
import math
import random
from typing import Any

from .config import (
    sample_choice,
    sample_float,
    sample_int,
    weighted_choice,
)
from .models import Cell, MapData, TaskData

_DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_CROSS_ZONE_FLOWS = (
    "left_to_center",
    "center_to_left",
    "center_to_right",
    "right_to_center",
    "left_to_right",
    "right_to_left",
)
_INTERSECTION_FLOWS = (
    "left_to_right",
    "right_to_left",
    "top_to_bottom",
    "bottom_to_top",
)


def _cells(values: list[list[int]]) -> list[Cell]:
    return [(int(row), int(col)) for row, col in values]


def _distance_map(map_data: MapData, start: Cell) -> dict[Cell, int]:
    distance = {start: 0}
    open_cells: collections.deque[Cell] = collections.deque([start])
    while open_cells:
        row, col = open_cells.popleft()
        for dr, dc in _DIRECTIONS:
            neighbor = (row + dr, col + dc)
            if map_data.traversable(neighbor) and neighbor not in distance:
                distance[neighbor] = distance[(row, col)] + 1
                open_cells.append(neighbor)
    return distance


def _candidate_pools(map_data: MapData) -> dict[str, list[Cell]]:
    zones = map_data.metadata["zones"]
    storage = _cells(map_data.metadata["service_cells"])
    station = _cells(zones["station_approach"])
    left = _cells(zones["left_storage"])
    center = _cells(zones["center_storage"])
    right = _cells(zones["right_storage"])
    top = _cells(zones.get("top_storage", [])) or [
        cell for cell in storage if cell[0] < map_data.rows / 3
    ]
    bottom = _cells(zones.get("bottom_storage", [])) or [
        cell for cell in storage if cell[0] >= 2 * map_data.rows / 3
    ]
    free = map_data.free_cells()
    return {
        "free": free,
        "storage": storage,
        "station": station,
        "left": left,
        "center": center,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


def _flow_pools(
    flow_type: str,
    pools: dict[str, list[Cell]],
    rng: random.Random,
) -> tuple[list[Cell], list[Cell], str]:
    if flow_type == "random":
        return pools["free"], pools["free"], "random"
    if flow_type == "storage_to_station":
        return pools["storage"], pools["station"], "storage_to_station"
    if flow_type == "station_to_storage":
        return pools["station"], pools["storage"], "station_to_storage"
    if flow_type == "one_way":
        return pools["left"], pools["right"], "left_to_right"
    if flow_type == "left_to_right":
        return pools["left"], pools["right"], "left_to_right"
    if flow_type == "right_to_left":
        return pools["right"], pools["left"], "right_to_left"
    if flow_type == "left_to_center":
        return pools["left"], pools["center"], "left_to_center"
    if flow_type == "center_to_left":
        return pools["center"], pools["left"], "center_to_left"
    if flow_type == "center_to_right":
        return pools["center"], pools["right"], "center_to_right"
    if flow_type == "right_to_center":
        return pools["right"], pools["center"], "right_to_center"
    if flow_type == "top_to_bottom":
        return pools["top"], pools["bottom"], "top_to_bottom"
    if flow_type == "bottom_to_top":
        return pools["bottom"], pools["top"], "bottom_to_top"
    if flow_type == "bidirectional":
        if rng.random() < 0.5:
            return pools["left"], pools["right"], "left_to_right"
        return pools["right"], pools["left"], "right_to_left"
    if flow_type == "hub_spoke":
        if rng.random() < 0.5:
            return pools["storage"], pools["station"], "spoke_to_hub"
        return pools["station"], pools["storage"], "hub_to_spoke"
    raise ValueError(f"unsupported flow type: {flow_type}")


def _clustered_pool(
    pool: list[Cell],
    cluster_count: int,
    radius: int,
    rng: random.Random,
) -> list[Cell]:
    if not pool or cluster_count <= 0:
        return pool
    centers = rng.sample(pool, min(cluster_count, len(pool)))
    clustered = [
        cell
        for cell in pool
        if any(
            abs(cell[0] - center[0]) + abs(cell[1] - center[1])
            <= radius
            for center in centers
        )
    ]
    return clustered or pool


def _choose_candidate(
    cells: list[Cell],
    distribution: str,
    rng: random.Random,
    rank_cache: dict[tuple[Cell, ...], dict[Cell, int]],
    demand_weights: dict[Cell, float] | None = None,
) -> Cell:
    demand_weights = demand_weights or {}
    if distribution == "uniform" and not any(
        cell in demand_weights for cell in cells
    ):
        return rng.choice(cells)
    if distribution not in {"uniform", "zipf"}:
        raise ValueError(
            "hotspot_distribution must be 'uniform' or 'zipf'"
        )
    key = tuple(cells)
    if key not in rank_cache:
        ranked = list(cells)
        rng.shuffle(ranked)
        rank_cache[key] = {
            cell: index + 1 for index, cell in enumerate(ranked)
        }
    ranks = rank_cache[key]
    weights = []
    for cell in cells:
        weight = demand_weights.get(cell, 1.0)
        if distribution == "zipf":
            weight *= 1.0 / ranks[cell]
        weights.append(weight)
    return rng.choices(cells, weights=weights, k=1)[0]


def _station_demand_weights(map_data: MapData) -> dict[Cell, float]:
    station_weights = {
        station["id"]: float(station.get("demand_weight", 1.0))
        for station in map_data.metadata["stations"]
    }
    result: dict[Cell, float] = {}
    for station_id, values in map_data.metadata["station_zones"].items():
        for cell in _cells(values):
            result[cell] = result.get(cell, 0.0) + station_weights[station_id]
    return result


def _od_flow(
    od_matrix: dict[str, float],
    pools: dict[str, list[Cell]],
    rng: random.Random,
) -> tuple[list[Cell], list[Cell], str]:
    selected = weighted_choice(od_matrix, rng, "od_matrix")
    if "->" not in selected:
        raise ValueError("OD keys must use 'origin->destination'")
    origin, destination = selected.split("->", 1)
    if origin not in pools or destination not in pools:
        raise ValueError(f"OD matrix references unknown zone: {selected}")
    return pools[origin], pools[destination], selected


def _scenario_flow(
    scenario: str,
    index: int,
    dominant_ratio: float,
    background_flow: str,
    opposing_ratio: float,
    config: dict[str, Any],
    rng: random.Random,
) -> str:
    if rng.random() > dominant_ratio:
        return background_flow
    if scenario == "uniform_random":
        return "random"
    if scenario == "dominant_one_way":
        return "one_way"
    if scenario == "balanced_bidirectional":
        return "right_to_left" if index % 2 else "left_to_right"
    if scenario == "bottleneck_pressure":
        return "right_to_left" if rng.random() < opposing_ratio else "left_to_right"
    if scenario == "station_rush":
        return "storage_to_station"
    if scenario == "station_release":
        return "station_to_storage"
    if scenario == "dead_end_turnover":
        return "storage_to_station" if index % 2 == 0 else "station_to_storage"
    if scenario == "mixed_background":
        return str(config.get("primary_flow", "bidirectional"))
    raise ValueError(f"unsupported scenario_type: {scenario}")


def _bottleneck_candidates(map_data: MapData, mode: str) -> list[Cell]:
    articulation = _cells(map_data.metadata.get("articulation_cells", []))
    if mode == "articulation" and articulation:
        return articulation
    prior = map_data.metadata["structural_congestion_prior"]
    ranked = sorted(
        (
            (prior[row][col], (row, col))
            for row in range(map_data.rows)
            for col in range(map_data.cols)
            if map_data.traversable((row, col))
        ),
        reverse=True,
    )
    return [cell for _, cell in ranked[: max(10, len(ranked) // 50)]]


def _balanced_flow_schedule(
    flows: tuple[str, ...], count: int, seed: int, salt: int
) -> list[str]:
    ordered = list(flows)
    random.Random(seed ^ salt).shuffle(ordered)
    return [ordered[index % len(ordered)] for index in range(count)]


def _intersection_components(map_data: MapData) -> list[dict[str, Any]]:
    semantic = map_data.metadata["semantic_cell_types"]
    unvisited = {
        (row, col)
        for row in range(map_data.rows)
        for col in range(map_data.cols)
        if semantic[row][col] == "X" and map_data.traversable((row, col))
    }
    prior = map_data.metadata["structural_congestion_prior"]
    components: list[dict[str, Any]] = []
    while unvisited:
        first = min(unvisited)
        unvisited.remove(first)
        queue: collections.deque[Cell] = collections.deque([first])
        cells = [first]
        while queue:
            row, col = queue.popleft()
            for dr, dc in _DIRECTIONS:
                neighbor = (row + dr, col + dc)
                if neighbor in unvisited:
                    unvisited.remove(neighbor)
                    queue.append(neighbor)
                    cells.append(neighbor)
        cells.sort()
        score = max(float(prior[row][col]) for row, col in cells)
        components.append(
            {
                "id": f"intersection_{cells[0][0]}_{cells[0][1]}",
                "cells": cells,
                "score": round(score, 6),
            }
        )
    return sorted(
        components,
        key=lambda item: (-float(item["score"]), item["cells"][0]),
    )


def _shortest_path_intersections(
    start: Cell,
    goal: Cell,
    component: dict[str, Any],
    distance_cache: dict[Cell, dict[Cell, int]],
    intersection_distance_cache: dict[Cell, dict[Cell, int]],
) -> list[Cell]:
    distance = distance_cache[start].get(goal)
    if distance is None:
        return []
    return [
        cell
        for cell in component["cells"]
        if (
            cell in distance_cache[start]
            and goal in intersection_distance_cache[cell]
            and distance_cache[start][cell]
            + intersection_distance_cache[cell][goal]
            == distance
        )
    ]


def _component_flow_capacity(
    component: dict[str, Any],
    start_pool: list[Cell],
    goal_pool: list[Cell],
    minimum_distance: int,
    maximum_distance: int | None,
    map_data: MapData,
    distance_cache: dict[Cell, dict[Cell, int]],
    intersection_distance_cache: dict[Cell, dict[Cell, int]],
) -> dict[str, int]:
    valid_starts: set[Cell] = set()
    valid_goals: set[Cell] = set()
    pair_count = 0
    for start in start_pool:
        if start not in distance_cache:
            distance_cache[start] = _distance_map(map_data, start)
        for goal in goal_pool:
            distance = distance_cache[start].get(goal)
            if distance is None or distance < minimum_distance:
                continue
            if maximum_distance is not None and distance > maximum_distance:
                continue
            if _shortest_path_intersections(
                start,
                goal,
                component,
                distance_cache,
                intersection_distance_cache,
            ):
                valid_starts.add(start)
                valid_goals.add(goal)
                pair_count += 1
    return {
        "starts": len(valid_starts),
        "goals": len(valid_goals),
        "pairs": pair_count,
        "endpoint_capacity": min(len(valid_starts), len(valid_goals)),
    }


def _intersection_candidates(
    map_data: MapData,
    pools: dict[str, list[Cell]],
    minimum_distance: int,
    maximum_distance: int | None,
    distance_cache: dict[Cell, dict[Cell, int]],
) -> tuple[list[dict[str, Any]], dict[Cell, dict[Cell, int]]]:
    components = _intersection_components(map_data)
    intersection_distance_cache = {
        cell: _distance_map(map_data, cell)
        for component in components
        for cell in component["cells"]
    }
    for component in components:
        supported_flows = []
        flow_capacities: dict[str, dict[str, int]] = {}
        for flow in _INTERSECTION_FLOWS:
            start_pool, goal_pool, _ = _flow_pools(flow, pools, random.Random(0))
            capacity = _component_flow_capacity(
                component,
                start_pool,
                goal_pool,
                minimum_distance,
                maximum_distance,
                map_data,
                distance_cache,
                intersection_distance_cache,
            )
            flow_capacities[flow] = capacity
            if capacity["pairs"] > 0:
                supported_flows.append(flow)
        component["supported_flows"] = supported_flows
        component["flow_capacities"] = flow_capacities
    return components, intersection_distance_cache


def _choose_mixed_flow(
    mixture: dict[str, float], rng: random.Random
) -> str:
    names = list(mixture)
    weights = [float(mixture[name]) for name in names]
    if not names or any(weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("flow_mixture must contain positive weights")
    return rng.choices(names, weights=weights, k=1)[0]


def generate_tasks(
    map_data: MapData,
    config: dict[str, Any],
    seed: int,
    task_id: str,
) -> TaskData:
    rng = random.Random(seed)
    pools = _candidate_pools(map_data)
    station_demand_weights = _station_demand_weights(map_data)
    if "agent_count" in config:
        agent_count = sample_int(
            config["agent_count"], rng, "agent_count"
        )
    else:
        density = sample_float(
            config.get("agent_density", [0.05, 0.2]),
            rng,
            "agent_density",
        )
        density_reference = sample_choice(
            config.get("density_reference", "service_cells"),
            rng,
            "density_reference",
        )
        reference_sizes = {
            "free_cells": len(pools["free"]),
            "service_cells": len(pools["storage"]),
            "aisle_cells": sum(
                character in "HVXB"
                for row in map_data.metadata["semantic_cell_types"]
                for character in row
            ),
        }
        if density_reference not in reference_sizes:
            raise ValueError(f"unsupported density_reference: {density_reference}")
        agent_count = max(1, round(density * reference_sizes[density_reference]))
    if agent_count <= 0:
        raise ValueError("agent_count must be positive")
    if agent_count > len(pools["free"]):
        raise ValueError("agent_count exceeds traversable cell count")

    scenario_value = config.get("scenario_type")
    flow_type = str(
        config.get(
            "flow_type",
            "scenario_controlled"
            if scenario_value is not None
            else "random",
        )
    )
    mixture = dict(config.get("flow_mixture", {}))
    if scenario_value == "mixed":
        scenario = weighted_choice(
            dict(config.get("scenario_mixture", {})),
            rng,
            "scenario_mixture",
        )
    elif scenario_value is None:
        scenario = "legacy_flow"
    else:
        scenario = sample_choice(
            scenario_value, rng, "scenario_type"
        )
    dominant_ratio = sample_float(
        config.get("dominant_flow_ratio", 1.0),
        rng,
        "dominant_flow_ratio",
    )
    background_flow = str(config.get("background_flow", "random"))
    opposing_flow_ratio = sample_float(
        config.get("opposing_flow_ratio", 0.5),
        rng,
        "opposing_flow_ratio",
    )
    origin_cluster_count = sample_int(
        config.get("origin_cluster_count", 0),
        rng,
        "origin_cluster_count",
    )
    goal_cluster_count = sample_int(
        config.get("goal_cluster_count", 0),
        rng,
        "goal_cluster_count",
    )
    cluster_radius = sample_int(
        config.get("cluster_radius", [1, 5]), rng, "cluster_radius"
    )
    hotspot_skew = sample_float(
        config.get("hotspot_skew", 0.0), rng, "hotspot_skew"
    )
    hotspot_distribution = sample_choice(
        config.get("hotspot_distribution", "uniform"),
        rng,
        "hotspot_distribution",
    )
    bottleneck_ratio = sample_float(
        config.get("required_bottleneck_crossing_ratio", 0.0),
        rng,
        "required_bottleneck_crossing_ratio",
    )
    shared_corridor_ratio = sample_float(
        config.get("shared_corridor_ratio", 0.0),
        rng,
        "shared_corridor_ratio",
    )
    swap_pair_ratio = sample_float(
        config.get("swap_pair_ratio", 0.0), rng, "swap_pair_ratio"
    )
    intersection_ratio = sample_float(
        config.get("required_intersection_crossing_ratio", 0.0),
        rng,
        "required_intersection_crossing_ratio",
    )
    target_intersection_count = sample_int(
        config.get("target_intersection_count", 2),
        rng,
        "target_intersection_count",
    )
    od_matrix = {
        str(key): float(value)
        for key, value in dict(config.get("od_matrix", {})).items()
    }
    for name, value in (
        ("dominant_flow_ratio", dominant_ratio),
        ("opposing_flow_ratio", opposing_flow_ratio),
        ("hotspot_skew", hotspot_skew),
        ("required_bottleneck_crossing_ratio", bottleneck_ratio),
        ("shared_corridor_ratio", shared_corridor_ratio),
        ("swap_pair_ratio", swap_pair_ratio),
        ("required_intersection_crossing_ratio", intersection_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if intersection_ratio > 0.0 and scenario != "intersection_crossing":
        raise ValueError(
            "required_intersection_crossing_ratio requires "
            "scenario_type='intersection_crossing'"
        )
    if intersection_ratio > 0.0 and target_intersection_count <= 0:
        raise ValueError("target_intersection_count must be positive")
    if intersection_ratio > 0.0 and od_matrix:
        raise ValueError(
            "od_matrix cannot be combined with an intersection constraint"
        )
    if (
        scenario in {"cross_zone_exchange", "intersection_crossing"}
        and not od_matrix
        and swap_pair_ratio > 0.0
    ):
        raise ValueError(
            "swap_pair_ratio cannot be combined with an exact OD scenario"
        )

    bottleneck_mode = str(
        config.get("target_bottleneck_mode", "highest_prior")
    )
    bottlenecks = _bottleneck_candidates(map_data, bottleneck_mode)
    bottleneck_agent_count = round(
        max(bottleneck_ratio, shared_corridor_ratio) * agent_count
    )
    minimum_distance = int(config.get("minimum_shortest_distance", 1))
    maximum_value = config.get("maximum_shortest_distance")
    maximum_distance = (
        int(maximum_value) if maximum_value is not None else None
    )
    max_attempts = int(config.get("max_sampling_attempts", 10_000))

    exact_flow_schedule: list[str] | None = None
    if not od_matrix and scenario == "cross_zone_exchange":
        exact_flow_schedule = _balanced_flow_schedule(
            _CROSS_ZONE_FLOWS, agent_count, seed, 0x43524F53
        )
    elif not od_matrix and scenario == "intersection_crossing":
        exact_flow_schedule = _balanced_flow_schedule(
            _INTERSECTION_FLOWS, agent_count, seed, 0x494E5458
        )
    requested_flow_counts = (
        dict(collections.Counter(exact_flow_schedule))
        if exact_flow_schedule is not None
        else None
    )

    if exact_flow_schedule is not None:
        start_counts: collections.Counter[str] = collections.Counter()
        goal_counts: collections.Counter[str] = collections.Counter()
        for flow in exact_flow_schedule:
            origin, destination = flow.split("_to_", 1)
            start_counts[origin] += 1
            goal_counts[destination] += 1
        for zone, count in start_counts.items():
            if count > len(pools[zone]):
                raise ValueError(
                    f"exact OD schedule requires {count} starts in {zone}, "
                    f"but only {len(pools[zone])} are available"
                )
        for zone, count in goal_counts.items():
            if count > len(pools[zone]):
                raise ValueError(
                    f"exact OD schedule requires {count} goals in {zone}, "
                    f"but only {len(pools[zone])} are available"
                )

    starts: list[Cell] = []
    goals: list[Cell] = []
    used_starts: set[Cell] = set()
    used_goals: set[Cell] = set()
    assignments: list[str] = []
    distances: list[int] = []
    required_bottlenecks: list[Cell | None] = []
    distance_cache: dict[Cell, dict[Cell, int]] = {}
    intersection_components: list[dict[str, Any]] = []
    selected_intersection_components: list[dict[str, Any]] = []
    intersection_distance_cache: dict[Cell, dict[Cell, int]] = {}
    intersection_agent_count = round(intersection_ratio * agent_count)
    minimum_intersection_endpoint_capacity = 0
    if scenario == "intersection_crossing" and intersection_agent_count:
        (
            intersection_components,
            intersection_distance_cache,
        ) = _intersection_candidates(
            map_data,
            pools,
            minimum_distance,
            maximum_distance,
            distance_cache,
        )
        minimum_intersection_endpoint_capacity = 2 * math.ceil(
            intersection_agent_count
            / (len(_INTERSECTION_FLOWS) * target_intersection_count)
        )
        feasible_components = [
            component
            for component in intersection_components
            if (
                set(component["supported_flows"])
                == set(_INTERSECTION_FLOWS)
                and all(
                    component["flow_capacities"][flow][
                        "endpoint_capacity"
                    ]
                    >= minimum_intersection_endpoint_capacity
                    for flow in _INTERSECTION_FLOWS
                )
            )
        ]
        feasible_components.sort(
            key=lambda component: (
                -float(component["score"]),
                -min(
                    component["flow_capacities"][flow][
                        "endpoint_capacity"
                    ]
                    for flow in _INTERSECTION_FLOWS
                ),
                str(component["id"]),
            )
        )
        if len(feasible_components) < target_intersection_count:
            raise ValueError(
                "intersection_crossing requires "
                f"{target_intersection_count} feasible intersection "
                "components supporting all four flows; found "
                f"{len(feasible_components)} with endpoint capacity "
                f">= {minimum_intersection_endpoint_capacity}"
            )
        selected_intersection_components = feasible_components[
            :target_intersection_count
        ]
    required_intersections: list[Cell | None] = []
    required_intersection_component_ids: list[str | None] = []
    origin_hotspot_cache: dict[int, list[Cell]] = {}
    goal_hotspot_cache: dict[int, list[Cell]] = {}
    endpoint_rank_cache: dict[tuple[Cell, ...], dict[Cell, int]] = {}
    bottleneck_distance_cache = {
        bottleneck: _distance_map(map_data, bottleneck)
        for bottleneck in bottlenecks
    }

    for agent in range(agent_count):
        found = False
        for _ in range(max_attempts):
            if od_matrix:
                start_pool, goal_pool, assignment = _od_flow(
                    od_matrix, pools, rng
                )
            else:
                if exact_flow_schedule is not None:
                    selected_flow = exact_flow_schedule[agent]
                elif scenario == "legacy_flow":
                    selected_flow = (
                        _choose_mixed_flow(mixture, rng)
                        if flow_type == "mixed"
                        else flow_type
                    )
                else:
                    selected_flow = _scenario_flow(
                        scenario,
                        agent,
                        dominant_ratio,
                        background_flow,
                        opposing_flow_ratio,
                        config,
                        rng,
                    )
                start_pool, goal_pool, assignment = _flow_pools(
                    selected_flow, pools, rng
                )
            if rng.random() < hotspot_skew:
                start_key = id(start_pool)
                goal_key = id(goal_pool)
                if start_key not in origin_hotspot_cache:
                    origin_hotspot_cache[start_key] = _clustered_pool(
                        start_pool,
                        origin_cluster_count,
                        cluster_radius,
                        rng,
                    )
                if goal_key not in goal_hotspot_cache:
                    goal_hotspot_cache[goal_key] = _clustered_pool(
                        goal_pool,
                        goal_cluster_count,
                        cluster_radius,
                        rng,
                    )
                start_pool = origin_hotspot_cache[start_key]
                goal_pool = goal_hotspot_cache[goal_key]
            available_starts = [
                cell for cell in start_pool if cell not in used_starts
            ]
            available_goals = [
                cell for cell in goal_pool if cell not in used_goals
            ]
            if not available_starts or not available_goals:
                continue
            start = _choose_candidate(
                available_starts,
                hotspot_distribution,
                rng,
                endpoint_rank_cache,
                station_demand_weights,
            )
            goal = _choose_candidate(
                available_goals,
                hotspot_distribution,
                rng,
                endpoint_rank_cache,
                station_demand_weights,
            )
            if start == goal:
                continue
            if start not in distance_cache:
                distance_cache[start] = _distance_map(map_data, start)
            distance = distance_cache[start].get(goal)
            if distance is None or distance < minimum_distance:
                continue
            if maximum_distance is not None and distance > maximum_distance:
                continue
            required_bottleneck: Cell | None = None
            if agent < bottleneck_agent_count and bottlenecks:
                candidate_bottlenecks = [
                    bottleneck
                    for bottleneck in bottlenecks
                    if (
                        bottleneck in distance_cache[start]
                        and goal in bottleneck_distance_cache[bottleneck]
                        and distance_cache[start][bottleneck]
                        + bottleneck_distance_cache[bottleneck][goal]
                        == distance
                    )
                ]
                if not candidate_bottlenecks:
                    continue
                required_bottleneck = rng.choice(candidate_bottlenecks)
            required_intersection: Cell | None = None
            required_component_id: str | None = None
            if agent < intersection_agent_count:
                component = selected_intersection_components[
                    agent % len(selected_intersection_components)
                ]
                candidate_intersections = _shortest_path_intersections(
                    start,
                    goal,
                    component,
                    distance_cache,
                    intersection_distance_cache,
                )
                if not candidate_intersections:
                    continue
                required_intersection = rng.choice(candidate_intersections)
                required_component_id = str(component["id"])
            starts.append(start)
            goals.append(goal)
            assignments.append(assignment)
            distances.append(distance)
            required_bottlenecks.append(required_bottleneck)
            required_intersections.append(required_intersection)
            required_intersection_component_ids.append(required_component_id)
            used_starts.add(start)
            used_goals.add(goal)
            found = True
            break
        if not found:
            raise ValueError(
                f"unable to sample valid endpoints for agent {agent}; "
                "reduce agent_count or distance constraints"
            )

    swap_indices = list(range(agent_count))
    rng.shuffle(swap_indices)
    target_swap_agent_count = min(
        agent_count - agent_count % 2,
        2 * round((swap_pair_ratio * agent_count) / 2),
    )
    swapped_pairs: list[list[int]] = []
    for offset in range(0, len(swap_indices) - 1, 2):
        if len(swapped_pairs) * 2 >= target_swap_agent_count:
            break
        first = swap_indices[offset]
        second = swap_indices[offset + 1]
        first_goal = goals[second]
        second_goal = goals[first]
        first_distance = distance_cache[starts[first]].get(first_goal)
        second_distance = distance_cache[starts[second]].get(second_goal)
        if (
            starts[first] == first_goal
            or starts[second] == second_goal
            or first_distance is None
            or second_distance is None
            or first_distance < minimum_distance
            or second_distance < minimum_distance
            or (
                maximum_distance is not None
                and (
                    first_distance > maximum_distance
                    or second_distance > maximum_distance
                )
            )
        ):
            continue
        goals[first], goals[second] = goals[second], goals[first]
        distances[first] = first_distance
        distances[second] = second_distance
        required_bottlenecks[first] = None
        required_bottlenecks[second] = None
        required_intersections[first] = None
        required_intersections[second] = None
        required_intersection_component_ids[first] = None
        required_intersection_component_ids[second] = None
        assignments[first] += "_goal_swap"
        assignments[second] += "_goal_swap"
        swapped_pairs.append([first, second])

    realized_od: dict[str, int] = {}
    for assignment in assignments:
        realized_od[assignment] = realized_od.get(assignment, 0) + 1
    metadata = {
        "schema_version": 2,
        "task_semantics_version": 2,
        "flow_type": flow_type,
        "flow_mixture": mixture if flow_type == "mixed" else None,
        "scenario_type": scenario,
        "scenario_mixture": (
            config.get("scenario_mixture")
            if scenario_value == "mixed"
            else None
        ),
        "dominant_flow_ratio": round(dominant_ratio, 6),
        "background_flow": background_flow,
        "opposing_flow_ratio": round(opposing_flow_ratio, 6),
        "od_matrix": od_matrix or None,
        "od_quota_counts": requested_flow_counts,
        "realized_flow_counts": realized_od,
        "origin_cluster_count": origin_cluster_count,
        "goal_cluster_count": goal_cluster_count,
        "cluster_radius": cluster_radius,
        "hotspot_skew": round(hotspot_skew, 6),
        "hotspot_distribution": hotspot_distribution,
        "required_bottleneck_crossing_ratio": round(
            bottleneck_ratio, 6
        ),
        "shared_corridor_ratio": round(shared_corridor_ratio, 6),
        "target_bottleneck_mode": bottleneck_mode,
        "required_bottlenecks": [
            list(cell) if cell is not None else None
            for cell in required_bottlenecks
        ],
        "required_intersection_crossing_ratio": round(
            intersection_ratio, 6
        ),
        "realized_intersection_crossing_ratio": round(
            sum(cell is not None for cell in required_intersections)
            / agent_count,
            6,
        ),
        "target_intersection_count": target_intersection_count,
        "minimum_intersection_endpoint_capacity": (
            minimum_intersection_endpoint_capacity
        ),
        "candidate_intersection_components": [
            {
                "id": str(component["id"]),
                "cells": [list(cell) for cell in component["cells"]],
                "score": component["score"],
                "supported_flows": list(component["supported_flows"]),
                "flow_capacities": component["flow_capacities"],
            }
            for component in intersection_components
        ],
        "selected_intersection_components": [
            {
                "id": str(component["id"]),
                "cells": [list(cell) for cell in component["cells"]],
                "score": component["score"],
                "supported_flows": list(component["supported_flows"]),
                "flow_capacities": component["flow_capacities"],
            }
            for component in selected_intersection_components
        ],
        "required_intersections": [
            list(cell) if cell is not None else None
            for cell in required_intersections
        ],
        "required_intersection_component_ids": (
            required_intersection_component_ids
        ),
        "swap_pair_ratio": round(swap_pair_ratio, 6),
        "swapped_agent_pairs": swapped_pairs,
        "agent_count": agent_count,
        "agent_density_free_cells": round(
            agent_count / len(pools["free"]), 6
        ),
        "agent_density_service_cells": round(
            agent_count / max(1, len(pools["storage"])), 6
        ),
        "minimum_shortest_distance": minimum_distance,
        "maximum_shortest_distance": maximum_distance,
        "actual_shortest_distances": distances,
        "mean_shortest_distance": round(sum(distances) / len(distances), 4),
        "flow_assignments": assignments,
    }
    return TaskData(
        task_id=task_id,
        map_id=map_data.map_id,
        seed=seed,
        starts=starts,
        goals=goals,
        metadata=metadata,
    )
