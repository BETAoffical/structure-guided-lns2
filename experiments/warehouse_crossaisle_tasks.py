"""Deterministic paired Warehouse cross-aisle development tasks.

This module is deliberately geometry-only.  It neither imports the native
solver nor runs reset/repair.  The two variants share the exact start sequence
and goal multiset; the matched variant is a distance-near-matched secondary
control, not an exactly difficulty-matched causal control.
"""

from __future__ import annotations

import hashlib
import math
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


TASK_PAIR_SCHEMA = "lns2.warehouse_crossaisle_task_pair.v1"
Q0_AUDIT_SCHEMA = "lns2.warehouse_crossaisle_q0_geometry_audit.v1"
STRUCTURED_VARIANT = "reciprocal_cross_aisle_exchange"
MATCHED_VARIANT = "matched_shifted_exchange"
TASK_VARIANTS = (STRUCTURED_VARIANT, MATCHED_VARIANT)
CROSS_AISLE_GROUP_COUNT = 12
SUPPORTED_Q = (16, 20)
DEFAULT_TASK_SEEDS = (419, 463)

REGISTERED_WAREHOUSE_MAP_SHA256 = {
    "warehouse-10-20-10-2-1": "c8d1b2f24788ed6bd1ccf45065b96b4ce82d65f88c72de750e03e2758637bff0",
    "warehouse-10-20-10-2-2": "4f06e82c2b87238daa8e308086afdba701112bf023e94e740e5bf9508a6adec3",
    "warehouse-20-40-10-2-1": "bd3bec2d1c20a8bbf900583cb4c2cf9dc16101470fbfae93b272ee0fb575d3ec",
    "warehouse-20-40-10-2-2": "eae2a3f5298b1e113bfc32b5b4404f5de5105365d237df63cc9dc9e2d1c73713",
}
REGISTERED_AISLE_WIDTHS = {
    "warehouse-10-20-10-2-1": 1,
    "warehouse-10-20-10-2-2": 2,
    "warehouse-20-40-10-2-1": 1,
    "warehouse-20-40-10-2-2": 2,
}
MAP_CODES = {
    "warehouse-10-20-10-2-1": "w1020a",
    "warehouse-10-20-10-2-2": "w1020b",
    "warehouse-20-40-10-2-1": "w2040a",
    "warehouse-20-40-10-2-2": "w2040b",
}

# These are protocol pins, not values inferred after a reset.
EXPECTED_GEOMETRY = {
    "warehouse-10-20-10-2-1": {
        "internal_group_count": 19,
        "excluded_central_group": [31],
        "chosen_groups": [[13], [16], [19], [22], [25], [28], [34], [37], [40], [43], [46], [49]],
        "staging_capacity": 25,
    },
    "warehouse-10-20-10-2-2": {
        "internal_group_count": 19,
        "excluded_central_group": [41, 42],
        "chosen_groups": [
            [17, 18], [21, 22], [25, 26], [29, 30], [33, 34], [37, 38],
            [45, 46], [49, 50], [53, 54], [57, 58], [61, 62], [65, 66],
        ],
        "staging_capacity": 25,
    },
    "warehouse-20-40-10-2-1": {
        "internal_group_count": 39,
        "excluded_central_group": [61],
        "chosen_groups": [[43], [46], [49], [52], [55], [58], [64], [67], [70], [73], [76], [79]],
        "staging_capacity": 50,
    },
    "warehouse-20-40-10-2-2": {
        "internal_group_count": 39,
        "excluded_central_group": [81, 82],
        "chosen_groups": [
            [57, 58], [61, 62], [65, 66], [69, 70], [73, 74], [77, 78],
            [85, 86], [89, 90], [93, 94], [97, 98], [101, 102], [105, 106],
        ],
        "staging_capacity": 50,
    },
}

RESET_QUALIFICATION_GATES = {
    STRUCTURED_VARIANT: {
        "initial_feasible": False,
        "minimum_initial_conflicts": 16,
        "minimum_active_conflict_agents": 32,
        "minimum_largest_component": 16,
    },
    MATCHED_VARIANT: {
        "initial_feasible": False,
        "minimum_initial_conflicts": 1,
    },
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_map(path: str | Path, map_id: str) -> tuple[list[str], str]:
    path = Path(path).resolve()
    if map_id not in REGISTERED_WAREHOUSE_MAP_SHA256:
        raise ValueError(f"unregistered Warehouse map: {map_id}")
    actual_sha = _sha256_file(path)
    if actual_sha != REGISTERED_WAREHOUSE_MAP_SHA256[map_id]:
        raise ValueError(f"Warehouse map checksum mismatch for {map_id}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5 or lines[0].strip() != "type octile" or lines[3].strip() != "map":
        raise ValueError("invalid MovingAI octile map header")
    height = int(lines[1].split()[1])
    width = int(lines[2].split()[1])
    grid = lines[4:]
    if len(grid) != height or any(len(row) != width for row in grid):
        raise ValueError("MovingAI map dimensions do not match header")
    if any(char not in ".GST@OWT" for row in grid for char in row):
        raise ValueError("unsupported MovingAI map cell")
    return grid, actual_sha


def _passable(char: str) -> bool:
    return char in ".GS"


def _consecutive_groups(rows: Iterable[int]) -> list[list[int]]:
    groups: list[list[int]] = []
    for row in sorted(rows):
        if not groups or row != groups[-1][-1] + 1:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def _side_capacity(grid: Sequence[str], row: int, column: int, step: int) -> int:
    capacity = 0
    while 0 < column < len(grid[0]) - 1 and _passable(grid[row][column]):
        capacity += 1
        column += step
    return capacity


def detect_cross_aisle_geometry(map_path: str | Path, map_id: str) -> dict[str, Any]:
    """Detect and hard-audit the frozen central mirrored 12-group geometry."""

    grid, map_sha = _read_map(map_path, map_id)
    height, width = len(grid), len(grid[0])
    aisle_width = REGISTERED_AISLE_WIDTHS[map_id]
    obstacle_cells = [
        (row, col)
        for row in range(1, height - 1)
        for col in range(1, width - 1)
        if not _passable(grid[row][col])
    ]
    if not obstacle_cells:
        raise ValueError("Warehouse map has no internal shelf cells")
    shelf_left = min(col for _, col in obstacle_cells)
    shelf_right = max(col for _, col in obstacle_cells)

    full_rows = [
        row
        for row in range(1, height - 1)
        if all(_passable(grid[row][col]) for col in range(1, width - 1))
    ]
    raw_groups = _consecutive_groups(full_rows)
    eligible: list[list[int]] = []
    group_audits: list[dict[str, Any]] = []
    for group in raw_groups:
        above, below = group[0] - 1, group[-1] + 1
        is_internal = above >= 1 and below <= height - 2
        shelf_above = is_internal and any(
            not _passable(grid[above][col]) for col in range(shelf_left, shelf_right + 1)
        )
        shelf_below = is_internal and any(
            not _passable(grid[below][col]) for col in range(shelf_left, shelf_right + 1)
        )
        if is_internal and shelf_above and shelf_below and len(group) == aisle_width:
            eligible.append(group)
            left_capacity = min(
                _side_capacity(grid, row, shelf_left - 1, -1) for row in group
            )
            right_capacity = min(
                _side_capacity(grid, row, shelf_right + 1, 1) for row in group
            )
            group_audits.append(
                {
                    "rows": list(group),
                    "full_width_traversal": True,
                    "shelf_band_above": True,
                    "shelf_band_below": True,
                    "left_staging_capacity": left_capacity,
                    "right_staging_capacity": right_capacity,
                }
            )

    expected = EXPECTED_GEOMETRY[map_id]
    if len(eligible) != expected["internal_group_count"] or len(eligible) % 2 != 1:
        raise ValueError("registered internal cross-aisle group count changed")
    middle = len(eligible) // 2
    central = eligible[middle]
    upper = eligible[middle - 6 : middle]
    lower = eligible[middle + 1 : middle + 7]
    if len(upper) != 6 or len(lower) != 6:
        raise ValueError("fewer than six mirrored groups on each side of center")
    chosen = upper + lower
    central_center2 = central[0] + central[-1]
    mirrored = all(
        (up[0] + up[-1]) + (down[0] + down[-1]) == 2 * central_center2
        for up, down in zip(upper, reversed(lower))
    )
    if not mirrored:
        raise ValueError("central cross-aisle selection is not geometrically mirrored")
    if central != expected["excluded_central_group"] or chosen != expected["chosen_groups"]:
        raise ValueError("registered central/chosen cross-aisle rows changed")

    selected_audits = [group_audits[eligible.index(group)] for group in chosen]
    left_capacity = min(item["left_staging_capacity"] for item in selected_audits)
    right_capacity = min(item["right_staging_capacity"] for item in selected_audits)
    if (
        left_capacity != expected["staging_capacity"]
        or right_capacity != expected["staging_capacity"]
        or min(left_capacity, right_capacity) < max(SUPPORTED_Q)
    ):
        raise ValueError("registered cross-aisle endpoint capacity changed")

    return {
        "map_id": map_id,
        "map_sha256": map_sha,
        "height": height,
        "width": width,
        "shelf_left": shelf_left,
        "shelf_right": shelf_right,
        "aisle_width": aisle_width,
        "eligible_groups": [list(group) for group in eligible],
        "internal_group_count": len(eligible),
        "excluded_central_group": list(central),
        "chosen_groups": [list(group) for group in chosen],
        "group_count": len(chosen),
        "mirrored_about_excluded_center": mirrored,
        "left_staging_capacity": left_capacity,
        "right_staging_capacity": right_capacity,
        "chosen_group_audits": selected_audits,
    }


def crossaisle_task_specs(
    task_seeds: Sequence[int] = DEFAULT_TASK_SEEDS,
    q_values: Sequence[int] = SUPPORTED_Q,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for map_id in REGISTERED_WAREHOUSE_MAP_SHA256:
        width = REGISTERED_AISLE_WIDTHS[map_id]
        for q in q_values:
            if int(q) not in SUPPORTED_Q:
                raise ValueError(f"unsupported q: {q}")
            agent_count = 2 * CROSS_AISLE_GROUP_COUNT * width * int(q)
            for task_seed in task_seeds:
                for variant in TASK_VARIANTS:
                    code = "rcx" if variant == STRUCTURED_VARIANT else "msx"
                    specs.append(
                        {
                            "task_id": f"{MAP_CODES[map_id]}__{code}__t{int(task_seed):04d}__q{int(q):02d}__n{agent_count:04d}",
                            "map_id": map_id,
                            "map_sha256": REGISTERED_WAREHOUSE_MAP_SHA256[map_id],
                            "variant": variant,
                            "task_seed": int(task_seed),
                            "q": int(q),
                            "aisle_width": width,
                            "group_count": CROSS_AISLE_GROUP_COUNT,
                            "agent_count": agent_count,
                        }
                    )
    return specs


def _rng(map_sha: str, task_seed: int, label: str) -> random.Random:
    material = f"{TASK_PAIR_SCHEMA}|{map_sha}|{int(task_seed)}|{label}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:16], "big"))


def _expected_task_id(
    map_id: str, variant: str, task_seed: int, q: int, agent_count: int
) -> str:
    code = "rcx" if variant == STRUCTURED_VARIANT else "msx"
    return f"{MAP_CODES[map_id]}__{code}__t{task_seed:04d}__q{q:02d}__n{agent_count:04d}"


def _monotone_path_is_free(
    grid: Sequence[str], start: tuple[int, int], goal: tuple[int, int]
) -> bool:
    def segment(a: tuple[int, int], b: tuple[int, int]) -> Iterable[tuple[int, int]]:
        if a[0] == b[0]:
            step = 1 if b[1] >= a[1] else -1
            return ((a[0], col) for col in range(a[1], b[1] + step, step))
        step = 1 if b[0] >= a[0] else -1
        return ((row, a[1]) for row in range(a[0], b[0] + step, step))

    bend_a = (goal[0], start[1])
    bend_b = (start[0], goal[1])
    paths = (
        list(segment(start, bend_a)) + list(segment(bend_a, goal))[1:],
        list(segment(start, bend_b)) + list(segment(bend_b, goal))[1:],
    )
    return any(all(_passable(grid[row][col]) for row, col in path) for path in paths)


def _task_payload(
    *,
    map_id: str,
    variant: str,
    task_seed: int,
    q: int,
    starts: Sequence[tuple[int, int]],
    mapping: Mapping[tuple[int, int], tuple[int, int]],
    grid: Sequence[str],
) -> dict[str, Any]:
    goals = [mapping[start] for start in starts]
    distances: list[int] = []
    for start, goal in zip(starts, goals):
        if not _monotone_path_is_free(grid, start, goal):
            raise ValueError("task endpoint pair has no certified Manhattan path")
        distances.append(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))
    return {
        "task_id": _expected_task_id(
            map_id, variant, task_seed, q, len(starts)
        ),
        "map_id": map_id,
        "variant": variant,
        "task_seed": task_seed,
        "q": q,
        "agent_count": len(starts),
        "starts": [list(cell) for cell in starts],
        "goals": [list(cell) for cell in goals],
        "shortest_path_distances": distances,
    }


def generate_paired_crossaisle_tasks(
    map_path: str | Path, map_id: str, task_seed: int, q: int
) -> dict[str, Any]:
    """Generate a paired structured/shifted task payload without a solver call."""

    q = int(q)
    if q not in SUPPORTED_Q:
        raise ValueError(f"unsupported q: {q}")
    geometry = detect_cross_aisle_geometry(map_path, map_id)
    grid, map_sha = _read_map(map_path, map_id)
    chosen = geometry["chosen_groups"]
    shelf_left, shelf_right = geometry["shelf_left"], geometry["shelf_right"]
    lanes = [(group_index, lane_index, row) for group_index, group in enumerate(chosen) for lane_index, row in enumerate(group)]

    left: dict[tuple[int, int], list[tuple[int, int]]] = {}
    right: dict[tuple[int, int], list[tuple[int, int]]] = {}
    permutations: dict[tuple[int, int], list[int]] = {}
    for group_index, lane_index, row in lanes:
        key = (group_index, lane_index)
        left[key] = [(row, shelf_left - q + slot) for slot in range(q)]
        right[key] = [(row, shelf_right + 1 + slot) for slot in range(q)]
        permutation = list(range(q))
        _rng(map_sha, task_seed, f"slots|{group_index}|{lane_index}|q{q}").shuffle(permutation)
        permutations[key] = permutation

    structured: dict[tuple[int, int], tuple[int, int]] = {}
    shifted: dict[tuple[int, int], tuple[int, int]] = {}
    group_shift = {base + offset: base + ((offset + 1) % 3) for base in range(0, 12, 3) for offset in range(3)}
    for group_index, lane_index, _ in lanes:
        key = (group_index, lane_index)
        destination = (group_shift[group_index], lane_index)
        for slot, permuted_slot in enumerate(permutations[key]):
            structured[left[key][slot]] = right[key][permuted_slot]
            structured[right[key][permuted_slot]] = left[key][slot]
            shifted[left[key][slot]] = right[destination][permuted_slot]
            shifted[right[key][permuted_slot]] = left[destination][slot]

    starts = list(structured)
    _rng(map_sha, task_seed, f"start-order|q{q}").shuffle(starts)
    expected_count = 2 * CROSS_AISLE_GROUP_COUNT * geometry["aisle_width"] * q
    if len(starts) != expected_count or set(starts) != set(shifted):
        raise AssertionError("paired task endpoint construction is incomplete")
    variants = {
        STRUCTURED_VARIANT: _task_payload(
            map_id=map_id,
            variant=STRUCTURED_VARIANT,
            task_seed=int(task_seed),
            q=q,
            starts=starts,
            mapping=structured,
            grid=grid,
        ),
        MATCHED_VARIANT: _task_payload(
            map_id=map_id,
            variant=MATCHED_VARIANT,
            task_seed=int(task_seed),
            q=q,
            starts=starts,
            mapping=shifted,
            grid=grid,
        ),
    }
    return {
        "schema": TASK_PAIR_SCHEMA,
        "pairing_semantics": "same_start_sequence_same_goal_multiset_distance_near_matched_secondary_control",
        "causal_control_claim": False,
        "map_id": map_id,
        "map_sha256": map_sha,
        "task_seed": int(task_seed),
        "q": q,
        "agent_count": expected_count,
        "geometry": geometry,
        "variants": variants,
    }


def _nearest_rank_p95(values: Sequence[int]) -> float:
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _reciprocal_ratio(starts: Sequence[tuple[int, int]], goals: Sequence[tuple[int, int]]) -> float:
    mapping = dict(zip(starts, goals))
    return sum(mapping.get(mapping[start]) == start for start in starts) / len(starts)


def audit_paired_crossaisle_tasks(
    map_path: str | Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute Q0 geometry and task invariants; returns errors instead of raising."""

    errors: list[str] = []
    map_id = str(payload.get("map_id", ""))
    try:
        geometry = detect_cross_aisle_geometry(map_path, map_id)
        grid, map_sha = _read_map(map_path, map_id)
    except (OSError, ValueError) as exc:
        return {"schema": Q0_AUDIT_SCHEMA, "passed": False, "errors": [str(exc)]}
    if payload.get("schema") != TASK_PAIR_SCHEMA:
        errors.append("task_pair_schema_mismatch")
    if (
        payload.get("pairing_semantics")
        != "same_start_sequence_same_goal_multiset_distance_near_matched_secondary_control"
        or payload.get("causal_control_claim") is not False
    ):
        errors.append("pairing_semantics_mismatch")
    if payload.get("map_sha256") != map_sha:
        errors.append("payload_map_sha256_mismatch")
    if dict(payload.get("geometry") or {}) != geometry:
        errors.append("geometry_payload_mismatch")
    q = int(payload.get("q", -1))
    task_seed = int(payload.get("task_seed", -1))
    expected_count = 2 * CROSS_AISLE_GROUP_COUNT * geometry["aisle_width"] * q
    if q not in SUPPORTED_Q:
        errors.append("unsupported_q")
    if int(payload.get("agent_count", -1)) != expected_count:
        errors.append("agent_count_formula_mismatch")

    variants = dict(payload.get("variants") or {})
    metrics: dict[str, Any] = {}
    parsed: dict[str, tuple[list[tuple[int, int]], list[tuple[int, int]], list[int]]] = {}
    chosen_row_to_group = {
        row: group_index
        for group_index, group in enumerate(geometry["chosen_groups"])
        for row in group
    }
    chosen_rows = set(chosen_row_to_group)
    shelf_left, shelf_right = geometry["shelf_left"], geometry["shelf_right"]
    for variant in TASK_VARIANTS:
        task = dict(variants.get(variant) or {})
        malformed = False
        try:
            starts = [tuple(map(int, cell)) for cell in task.get("starts") or ()]
            goals = [tuple(map(int, cell)) for cell in task.get("goals") or ()]
            recorded_distances = list(
                map(int, task.get("shortest_path_distances") or ())
            )
        except (TypeError, ValueError):
            starts, goals, recorded_distances = [], [], []
            malformed = True
            errors.append(f"{variant}:malformed_endpoint_payload")
        expected_task_id = _expected_task_id(
            map_id, variant, task_seed, q, expected_count
        )
        if (
            task.get("variant") != variant
            or task.get("map_id") != map_id
            or task.get("task_id") != expected_task_id
            or int(task.get("task_seed", -1)) != task_seed
            or int(task.get("q", -1)) != q
            or int(task.get("agent_count", -1)) != expected_count
        ):
            errors.append(f"{variant}:identity_mismatch")
        if len(starts) != expected_count or len(goals) != expected_count:
            errors.append(f"{variant}:endpoint_count_mismatch")
        if len(set(starts)) != len(starts) or len(set(goals)) != len(goals):
            errors.append(f"{variant}:endpoints_not_unique")
        if any(start == goal for start, goal in zip(starts, goals)):
            errors.append(f"{variant}:fixed_point")
        valid_endpoint = lambda cell: (
            len(cell) == 2
            and 0 <= cell[0] < len(grid)
            and 0 <= cell[1] < len(grid[0])
            and cell[0] in chosen_rows
            and (shelf_left - q <= cell[1] < shelf_left or shelf_right < cell[1] <= shelf_right + q)
        )
        endpoints_valid = not malformed and all(
            valid_endpoint(cell) for cell in starts + goals
        )
        if not endpoints_valid:
            errors.append(f"{variant}:endpoint_outside_registered_lanes")
        distances: list[int] = []
        if endpoints_valid:
            for start, goal in zip(starts, goals):
                if not _monotone_path_is_free(grid, start, goal):
                    errors.append(f"{variant}:uncertified_reachability")
                    break
                distances.append(
                    abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                )
        if distances != recorded_distances:
            errors.append(f"{variant}:distance_payload_mismatch")
        mapping = dict(zip(starts, goals))
        same_group_ratio = (
            sum(chosen_row_to_group.get(s[0]) == chosen_row_to_group.get(g[0]) for s, g in zip(starts, goals)) / len(starts)
            if starts else 0.0
        )
        reciprocal_ratio = _reciprocal_ratio(starts, goals) if starts else 0.0
        expected_reciprocal = 1.0 if variant == STRUCTURED_VARIANT else 0.0
        expected_same_group = 1.0 if variant == STRUCTURED_VARIANT else 0.0
        if reciprocal_ratio != expected_reciprocal:
            errors.append(f"{variant}:reciprocal_ratio_mismatch")
        if same_group_ratio != expected_same_group:
            errors.append(f"{variant}:same_group_ratio_mismatch")
        group_counts = {index: 0 for index in range(CROSS_AISLE_GROUP_COUNT)}
        row_counts = {row: 0 for row in chosen_rows}
        for start in starts:
            if len(start) == 2 and start[0] in chosen_row_to_group:
                group_counts[chosen_row_to_group[start[0]]] += 1
                row_counts[start[0]] += 1
        if any(count != 2 * geometry["aisle_width"] * q for count in group_counts.values()):
            errors.append(f"{variant}:group_coverage_mismatch")
        if any(count != 2 * q for count in row_counts.values()):
            errors.append(f"{variant}:physical_lane_coverage_mismatch")
        metrics[variant] = {
            "reciprocal_ratio": reciprocal_ratio,
            "same_group_ratio": same_group_ratio,
            "distance_mean": sum(distances) / len(distances) if distances else None,
            "distance_p95": _nearest_rank_p95(distances) if distances else None,
            "covered_group_count": sum(count > 0 for count in group_counts.values()),
            "covered_physical_lane_count": sum(count > 0 for count in row_counts.values()),
        }
        parsed[variant] = (starts, goals, distances)

    if all(variant in parsed for variant in TASK_VARIANTS):
        structured_starts, structured_goals, _ = parsed[STRUCTURED_VARIANT]
        matched_starts, matched_goals, _ = parsed[MATCHED_VARIANT]
        if structured_starts != matched_starts:
            errors.append("paired_start_sequence_mismatch")
        if sorted(structured_goals) != sorted(matched_goals):
            errors.append("paired_goal_multiset_mismatch")
        s_mean = metrics[STRUCTURED_VARIANT]["distance_mean"]
        m_mean = metrics[MATCHED_VARIANT]["distance_mean"]
        s_p95 = metrics[STRUCTURED_VARIANT]["distance_p95"]
        m_p95 = metrics[MATCHED_VARIANT]["distance_p95"]
        mean_delta = (
            abs(m_mean - s_mean) / s_mean
            if s_mean and m_mean is not None
            else math.inf
        )
        p95_delta = (
            abs(m_p95 - s_p95) / s_p95
            if s_p95 and m_p95 is not None
            else math.inf
        )
        metrics["paired_distance"] = {
            "mean_relative_difference": mean_delta,
            "p95_relative_difference": p95_delta,
            "mean_gate_maximum": 0.05,
            "p95_gate_maximum": 0.10,
        }
        if mean_delta > 0.05:
            errors.append("paired_distance_mean_gate_failed")
        if p95_delta > 0.10:
            errors.append("paired_distance_p95_gate_failed")

    return {
        "schema": Q0_AUDIT_SCHEMA,
        "passed": not errors,
        "errors": errors,
        "map_id": map_id,
        "map_sha256": map_sha,
        "task_seed": task_seed,
        "q": q,
        "agent_count": expected_count,
        "geometry": geometry,
        "metrics": metrics,
        "qualification_gates_for_future_reset_only_q1": RESET_QUALIFICATION_GATES,
        "control_interpretation": "same-start/same-goal-multiset distance-near-matched secondary control",
    }


__all__ = [
    "CROSS_AISLE_GROUP_COUNT",
    "DEFAULT_TASK_SEEDS",
    "EXPECTED_GEOMETRY",
    "MATCHED_VARIANT",
    "Q0_AUDIT_SCHEMA",
    "REGISTERED_AISLE_WIDTHS",
    "REGISTERED_WAREHOUSE_MAP_SHA256",
    "RESET_QUALIFICATION_GATES",
    "STRUCTURED_VARIANT",
    "SUPPORTED_Q",
    "TASK_PAIR_SCHEMA",
    "TASK_VARIANTS",
    "audit_paired_crossaisle_tasks",
    "crossaisle_task_specs",
    "detect_cross_aisle_geometry",
    "generate_paired_crossaisle_tasks",
]
