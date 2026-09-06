"""Static case preparation only: never import a solver or execute a controller."""

from __future__ import annotations

import collections
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any

from experiments._common import (
    CLOSED_LOOP_IMPLEMENTATION_FILES, atomic_write_text, read_json, read_jsonl,
    sha256_file, write_json, write_jsonl,
)


def contained(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or PureWindowsPath(relative).drive:
        raise ValueError("expected a repository-relative path")
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()) or target == root.resolve():
        raise ValueError("path escapes repository")
    return target


def checked_input(root: Path, specification: dict) -> Path:
    path = contained(root, specification["manifest"])
    if sha256_file(path) != specification["sha256"]:
        raise ValueError(f"input SHA mismatch: {specification['manifest']}")
    return path


def read_grid(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if len(lines) < 5 or lines[0] != "type octile" or lines[3] != "map":
        raise ValueError("invalid MovingAI map header")
    if not lines[1].startswith("height ") or not lines[2].startswith("width "):
        raise ValueError("invalid map dimensions")
    height, width = int(lines[1].split()[1]), int(lines[2].split()[1])
    grid = lines[4:]
    if height < 1 or width < 1 or len(grid) != height or any(len(r) != width for r in grid):
        raise ValueError("map dimensions do not match grid")
    if set("".join(grid)) - set(".@TOWGS"):
        raise ValueError("unsupported MovingAI cell")
    return grid


def free(grid: list[str], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] in ".GS"


def distances(grid: list[str], start: tuple[int, int]) -> dict[tuple[int, int], int]:
    found = {start: 0}
    queue = collections.deque([start])
    while queue:
        r, c = queue.popleft()
        for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nxt not in found and free(grid, nxt):
                found[nxt] = found[(r, c)] + 1
                queue.append(nxt)
    return found


def scenario_prefix(map_path: Path, scenario: Path, grid: list[str], count: int) -> list[list[str]]:
    lines = scenario.read_text(encoding="utf-8-sig").splitlines()
    if not lines or lines[0].strip() != "version 1":
        raise ValueError("invalid scenario version")
    rows = [line.split() for line in lines[1:] if line.strip()]
    if len(rows) < count:
        raise ValueError("scenario prefix is too short")
    if any(len(row) != 9 for row in rows[:count]):
        raise ValueError("invalid scenario fields")
    for row in rows[:count]:
        if row[1] != map_path.name or [int(row[2]), int(row[3])] != [len(grid[0]), len(grid)]:
            raise ValueError("scenario map identity/dimensions mismatch")
    return rows[:count]


def audit_task(map_path: Path, scenario: Path, task_path: Path, count: int) -> dict:
    if type(count) is not int or count <= 0:
        raise ValueError("invalid agent count")
    grid = read_grid(map_path)
    task = read_json(task_path)
    rows = scenario_prefix(map_path, scenario, grid, count)
    if "starts" in task or "goals" in task:
        starts, goals = task["starts"], task["goals"]
    else:
        if task.get("agent_count") != count or task.get("benchmark_id") != map_path.stem:
            raise ValueError("MovingAI sidecar identity mismatch")
        if task.get("scenario_sha256") != sha256_file(scenario):
            raise ValueError("MovingAI sidecar scenario SHA mismatch")
        starts = [[int(row[5]), int(row[4])] for row in rows[:count]]
        goals = [[int(row[7]), int(row[6])] for row in rows[:count]]
    if len(starts) != count or len(goals) != count:
        raise ValueError("task agent count mismatch")
    for cells in (starts, goals):
        for cell in cells:
            if not isinstance(cell, list) or len(cell) != 2 or any(type(x) is not int for x in cell):
                raise ValueError("invalid task coordinate")
            if not free(grid, tuple(cell)):
                raise ValueError("blocked or out-of-bounds endpoint")
        if len(set(map(tuple, cells))) != count:
            raise ValueError("duplicate starts or goals")
    lengths = []
    reference_distance_differences = []
    for i, row in enumerate(rows[:count]):
        if len(row) != 9 or row[1] != map_path.name:
            raise ValueError("scenario map name or fields mismatch")
        if [int(row[2]), int(row[3])] != [len(grid[0]), len(grid)]:
            raise ValueError("scenario map dimensions mismatch")
        if [int(row[5]), int(row[4])] != starts[i] or [int(row[7]), int(row[6])] != goals[i]:
            raise ValueError("scenario prefix and task disagree")
        length = distances(grid, tuple(starts[i])).get(tuple(goals[i]))
        if length is None:
            raise ValueError("unreachable OD pair")
        claimed = float(row[8])
        if not math.isfinite(claimed) or claimed < 0:
            raise ValueError("invalid scenario reference distance")
        if abs(claimed - length) > 0.001:
            reference_distance_differences.append({"agent": i, "reference": claimed, "four_neighbor": length})
        lengths.append(length)
    return {
        "rows": len(grid), "cols": len(grid[0]),
        "free_cells": sum(ch in ".GS" for row in grid for ch in row),
        "agent_count": count, "sum_shortest_distances": sum(lengths),
        "max_shortest_distance": max(lengths), "static_valid": True,
        "distance_reference": "recomputed_four_neighbor_BFS_not_scenario_reference_column",
        "scenario_distance_difference_count": len(reference_distance_differences),
        "scenario_distance_difference_examples": reference_distance_differences[:3],
        "od_prefix_sha256": hashlib.sha256(json.dumps([starts, goals], separators=(",", ":")).encode()).hexdigest(),
        "joint_feasibility_proven": False, "native_reset_performed": False,
    }


def audit_paths(grid: list[str], starts: list, goals: list, paths: list[list[int]]) -> dict:
    """Future output audit, with stay-at-goal collisions and terminal padding removed."""
    if not paths or len(paths) != len(starts) or len(paths) != len(goals):
        raise ValueError("path agent count mismatch")
    width = len(grid[0])
    normalized = []
    for i, raw in enumerate(paths):
        if not raw or any(type(x) is not int or not free(grid, divmod(x, width)) for x in raw):
            raise ValueError("empty, invalid or blocked path")
        if divmod(raw[0], width) != tuple(starts[i]) or divmod(raw[-1], width) != tuple(goals[i]):
            raise ValueError("path endpoints mismatch")
        for a, b in zip(raw, raw[1:]):
            ar, ac = divmod(a, width)
            br, bc = divmod(b, width)
            if abs(ar - br) + abs(ac - bc) > 1:
                raise ValueError("illegal move")
        path = list(raw)
        while len(path) > 1 and path[-2] == path[-1]:
            path.pop()
        normalized.append(path)
    makespan = max(len(path) - 1 for path in normalized)
    pairs = set()
    for t in range(makespan + 1):
        occupied: dict[int, list[int]] = collections.defaultdict(list)
        moves: dict[tuple[int, int], list[int]] = collections.defaultdict(list)
        for i, path in enumerate(normalized):
            pos = path[min(t, len(path) - 1)]
            pairs.update((j, i) for j in occupied[pos])
            occupied[pos].append(i)
            if t:
                prev = path[min(t - 1, len(path) - 1)]
                if prev != pos:
                    pairs.update((j, i) for j in moves[(pos, prev)])
                    moves[(prev, pos)].append(i)
    return {
        "feasible": not pairs, "colliding_pairs": len(pairs),
        "soc_steps": sum(len(p) - 1 for p in normalized), "makespan_steps": makespan,
        "wait_steps": sum(a == b for p in normalized for a, b in zip(p, p[1:])),
        "terminal_padding_removed": True,
    }


def prepare(root: Path, config_path: Path) -> dict[str, Any]:
    root = root.resolve()
    config = read_json(config_path)
    if config.get("schema") != "lns2.path_quality_preflight.v1" or config.get("allow_solver_execution") is not False:
        raise ValueError("preflight cannot authorize solver execution")
    seeds = config["solver_seeds"]
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("invalid solver seeds")
    if config["controllers"] != ["official_adaptive", "v2-full", "dual16"]:
        raise ValueError("frozen three-arm design changed")
    files: dict[str, str] = {}

    def register(path: Path) -> str:
        relative = path.resolve().relative_to(root).as_posix()
        files[relative] = sha256_file(path)
        return relative

    register(config_path)
    runtime_template = checked_input(root, config["runtime_template"])
    register(runtime_template)
    bundle_manifest = checked_input(root, config["frozen_bundle"])
    register(bundle_manifest)

    def verify_bundle(value: Any) -> None:
        if isinstance(value, dict):
            if "file" in value and "sha256" in value:
                path = contained(bundle_manifest.parent, value["file"])
                if sha256_file(path) != value["sha256"]:
                    raise ValueError("frozen bundle member SHA mismatch")
                register(path)
            for item in value.values():
                verify_bundle(item)
        elif isinstance(value, list):
            for item in value:
                verify_bundle(item)

    verify_bundle(read_json(bundle_manifest))
    cases = []
    task_ids = set()
    for source in config["sources"]:
        manifest = checked_input(root, source)
        register(manifest)
        rows = read_jsonl(manifest)
        expected_ids = None
        if not source.get("all_tasks"):
            expected_ids = {
                f"{name}__random_{scenario:02d}__agents_{count:04d}"
                for name, count in source["map_agent_counts"].items()
                for scenario in source["scenario_indices"]
            }
            rows = [row for row in rows if row["task_id"] in expected_ids]
        if len(rows) != source["expected_tasks"] or (expected_ids is not None and {r['task_id'] for r in rows} != expected_ids):
            raise ValueError("case roster missing or duplicated")
        for row in sorted(rows, key=lambda r: r["task_id"]):
            if row["task_id"] in task_ids:
                raise ValueError("duplicate task ID across sources")
            task_ids.add(row["task_id"])
            paths = {key: contained(manifest.parent, row[key]) for key in ("map_file", "scenario_file", "task_file")}
            static = audit_task(paths["map_file"], paths["scenario_file"], paths["task_file"], row["agent_count"])
            reason = config["quarantined_maps"].get(row["map_id"])
            cases.append({
                "task_id": row["task_id"], "map_id": row["map_id"],
                "family": "warehouse" if source["name"] == "warehouse" else row["layout_mode"],
                "input_mode": "fresh_start_goal_task_not_checkpoint", "solver_seeds": seeds,
                "status": "quarantined" if reason else "static_ready_runtime_unverified",
                "quarantine_reason": reason, "static_audit": static,
                "files": {key: register(path) for key, path in paths.items()},
            })
    checkpoint_path = checked_input(root, config["supplementary_checkpoints"])
    register(checkpoint_path)
    checkpoint_rows = read_jsonl(checkpoint_path)
    if len(checkpoint_rows) != config["supplementary_checkpoints"]["expected_count"]:
        raise ValueError("checkpoint count mismatch")
    if len({r["checkpoint_id"] for r in checkpoint_rows}) != len(checkpoint_rows):
        raise ValueError("duplicate checkpoint ID")
    supplementary = []
    by_task = {case["task_id"]: case for case in cases}
    for row in checkpoint_rows:
        blob = contained(checkpoint_path.parent, row["state_blob"])
        if sha256_file(blob) != row["state_blob_sha256"]:
            raise ValueError("checkpoint blob SHA mismatch")
        case = by_task[row["task_id"]]
        if case["map_id"] != row["map_id"] or case["static_audit"]["agent_count"] != row["agent_count"]:
            raise ValueError("checkpoint task binding mismatch")
        if files[case["files"]["map_file"]] != row["map_sha256"] or files[case["files"]["task_file"]] != row["task_sha256"]:
            raise ValueError("checkpoint map/task SHA mismatch")
        supplementary.append({"checkpoint_id": row["checkpoint_id"], "task_id": row["task_id"],
                              "state_blob": register(blob), "role": config["supplementary_checkpoints"]["role"]})
    for relative in (*CLOSED_LOOP_IMPLEMENTATION_FILES,
                     "third_party/mapf_lns2/src/LNS.cpp", "third_party/mapf_lns2/inc/LNS.h",
                     "third_party/mapf_lns2/src/driver.cpp",
                     "lns2_selector/evaluation/path_quality_preflight.py",
                     "lns2_selector/evaluation/anytime_handoff.py",
                     "lns2_selector/evaluation/path_quality_execution.py",
                     "src/python_bindings.cpp", "CMakeLists.txt",
                     "scripts/prepare_path_quality_cases.py"):
        register(contained(root, relative))
    quarantined = [c for c in cases if c["status"] == "quarantined"]
    eligible = len(cases) - len(quarantined)
    from lns2_selector.evaluation.path_quality_execution import execution_schedule
    schedule = execution_schedule(cases, config["protocol"])
    report = {
        "schema": "lns2.path_quality_preparation.v1", "scientific_status": config["scientific_status"],
        "status": "static_preparation_complete_timed_execution_blocked",
        "solver_calls": 0, "timed_episodes_started": 0, "training_calls": 0,
        "historical_controller_outcomes_used_for_selection": False,
        "map_count": len({c["map_id"] for c in cases}), "task_count": len(cases),
        "quarantined_tasks": len(quarantined), "eligible_tasks": eligible,
        "eligible_instance_seeds": eligible * len(seeds),
        "proposed_first_feasible_episode_count": eligible * len(seeds) * len(config["controllers"]),
        "proposed_total_episode_count": len(schedule), "execution_schedule": schedule,
        "runtime_template": config["runtime_template"],
        "supplementary_checkpoints": supplementary, "cases": cases, "protocol": config["protocol"],
        "input_sha256": dict(sorted(files.items())),
        "blocking_items": [
            "User has not authorized timed execution; no collect entry point is provided.",
            "Native binary loaded identity, reset parity and complete final path export must be verified separately before timing.",
            "Scheduling, path journals and isolated episode supervision exist; production native registration and controller-level pairing admission still require separate checks. No cohort launcher is enabled.",
            "Physical motion duration is uncalibrated; modeled completion times are sensitivity analysis, not real robot results.",
            "Historical checkpoints are full-path repair inputs, not committed-prefix execution recovery.",
        ],
    }
    report["fingerprint"] = hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return report


def publish(root: Path, report: dict, output: str, *, verify: bool = False) -> None:
    destination = contained(root, output)
    if not destination.is_relative_to((root / "build").resolve()):
        raise ValueError("preflight output must be under build")
    target = destination / "preflight_report.json"
    roster = destination / "cases.jsonl"
    schedule_path = destination / "execution_schedule.jsonl"
    if verify or target.exists() or (destination.exists() and any(destination.iterdir())):
        if not target.is_file() or read_json(target) != report or not roster.is_file() or read_jsonl(roster) != report["cases"]:
            raise ValueError("existing preflight differs; use a new output directory")
        if "execution_schedule" in report and (not schedule_path.is_file() or read_jsonl(schedule_path) != report["execution_schedule"]):
            raise ValueError("existing execution schedule differs")
        return
    write_jsonl(roster, report["cases"])
    if "execution_schedule" in report:
        write_jsonl(schedule_path, report["execution_schedule"])
    lines = ["# Path quality preparation (no solver execution)", "", report["status"], "",
             "| Map | Agents | Status |", "|---|---:|---|"]
    lines.extend(f"| {r['task_id']} | {r['static_audit']['agent_count']} | {r['status']} |" for r in report["cases"])
    lines.extend(["", "## Blocking items", *[f"- {s}" for s in report["blocking_items"]], "",
                  f"Fingerprint: `{report['fingerprint']}`", ""])
    atomic_write_text(destination / "preflight_report.md", "\n".join(lines))
    write_json(target, report)
