"""Inventory retained inputs and generate the frozen cohort without a solver."""
import argparse
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json, write_jsonl
from generators.dataset import generate_dataset
from generators.models import MapData, TaskData
from generators.validation import _distance_map, validate_map, validate_task
from scripts.prepare_warehouse_repair_confirmation import CONFIG, dataset_config, validate

BASE = ROOT / "build/warehouse-repair-confirmation-v1"
SKIP = {".git", "venv-graph", "site-packages", "__pycache__", "node_modules",
        "linux", "windows"}
SEED_KEYS = {"master_seed", "map_seed", "task_seed", "map_seeds", "task_seeds"}


def seeds_in(value):
    result = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SEED_KEYS:
                values = item if isinstance(item, list) else [item]
                result.update(v for v in values if type(v) is int)
            result.update(seeds_in(item))
    elif isinstance(value, list):
        for item in value:
            result.update(seeds_in(item))
    return result


def geometry_hash(text):
    lines = text.splitlines()
    if not lines or not lines[0].lower().startswith("type "):
        return None  # Older non-MovingAI formats are recorded, not interpreted.
    if len(lines) < 5 or lines[3].lower() != "map":
        raise ValueError("invalid MovingAI map header")
    rows, cols = int(lines[1].split()[1]), int(lines[2].split()[1])
    grid = lines[4:]
    if len(grid) != rows or any(len(row) != cols for row in grid):
        raise ValueError("invalid MovingAI map dimensions")
    return hashlib.sha256((f"{rows},{cols}\n" + "\n".join(grid)).encode()).hexdigest()


def input_paths(root):
    for name in ("build", "configs", "artifacts"):
        for directory, folders, files in os.walk(root / name, followlinks=False):
            parent = Path(directory)
            folders[:] = sorted(d for d in folders if d not in SKIP
                                and not (parent / d).is_symlink()
                                and (parent / d) != root / "build/warehouse-repair-confirmation-v1")
            for file in sorted(files):
                path = parent / file
                if path.is_symlink():
                    continue
                if (path.suffix == ".map" or file in {"manifest.jsonl", "dataset_summary.json"}
                        or (path.suffix == ".json" and
                            (name == "configs" or parent.name in {"maps", "instances"}))):
                    yield path


def inspect_input(path, root):
    record = {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
    seeds = set()
    if path.suffix == ".map":
        record["geometry_sha256"] = geometry_hash(path.read_text(encoding="utf-8-sig"))
    elif path.suffix == ".jsonl":
        # Read only manifests, never episode traces or outcome labels.
        with path.open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    seeds.update(seeds_in(json.loads(line)))
    else:
        doc = read_json(path)
        seeds.update(seeds_in(doc))
        if isinstance(doc, dict) and ("map_id" in doc or "task_id" in doc):
            if type(doc.get("seed")) is int:
                seeds.add(doc["seed"])
    record["seeds"] = sorted(seeds)
    return record


def seed_schedule(config):
    rng = random.Random(config["master_seed"])
    result, used = [], set()
    for _ in range(8):
        seed = rng.randrange(1, 2**31)
        while seed in used:
            seed = rng.randrange(1, 2**31)
        used.add(seed)
        result.append({"map_seed": seed, "task_seeds": [rng.randrange(1, 2**31) for _ in range(2)]})
    return result


def assert_disjoint(expanded, history):
    schedule = seed_schedule(expanded)
    values = [expanded["master_seed"]]
    for row in schedule:
        values.extend([row["map_seed"], *row["task_seeds"]])
    if len(values) != len(set(values)):
        raise ValueError("new seed schedule contains duplicates")
    overlap = set(values) & {s for row in history for s in row["seeds"]}
    if overlap:
        raise ValueError(f"retained historical seed collision: {sorted(overlap)}")
    return schedule


def verify_task(row, split_root):
    m = read_json(split_root / row["map_metadata_file"])
    t = read_json(split_root / row["task_file"])
    map_data = MapData(m["map_id"], m["seed"], m["grid"], m["metadata"])
    task = TaskData(t["task_id"], t["map_id"], t["seed"],
                    [tuple(v) for v in t["starts"]], [tuple(v) for v in t["goals"]], t["metadata"])
    validate_map(map_data)
    validate_task(map_data, task)
    if (m["map_id"], m["seed"], t["task_id"], t["seed"], task.agent_count) != (
            row["map_id"], row["map_seed"], row["task_id"], row["task_seed"], row["agent_count"]):
        raise ValueError("manifest/sidecar identity mismatch")
    if task.agent_count != round(len(map_data.free_cells()) * 0.15):
        raise ValueError("agent density mismatch")
    distances = [_distance_map(map_data, s).get(g) for s, g in zip(task.starts, task.goals)]
    if any(d is None or not 8 <= d <= 100 for d in distances):
        raise ValueError("endpoint distance outside frozen limits")
    map_path = split_root / row["map_file"]
    expected_map = f"type octile\nheight {map_data.rows}\nwidth {map_data.cols}\nmap\n" + "\n".join(map_data.grid)
    if geometry_hash(map_path.read_text()) != geometry_hash(expected_map):
        raise ValueError("map/sidecar geometry mismatch")
    scenario = (split_root / row["scenario_file"]).read_text().splitlines()
    if scenario[0] != "version 1" or len(scenario) != task.agent_count + 1:
        raise ValueError("invalid scenario size")
    for line, start, goal, distance in zip(scenario[1:], task.starts, task.goals, distances):
        parts = line.split()
        if (len(parts) != 9 or parts[1] != map_path.name or
                list(map(int, parts[2:8])) != [map_data.cols, map_data.rows, start[1], start[0], goal[1], goal[0]]
                or float(parts[8]) != distance):
            raise ValueError("scenario/sidecar endpoints or distance mismatch")
    return {"map_id": map_data.map_id, "task_id": task.task_id, "map_seed": map_data.seed,
            "task_seed": task.seed, "agents": task.agent_count, "free_cells": len(map_data.free_cells()),
            "shortest_distance_min": min(distances), "shortest_distance_max": max(distances),
            "geometry_sha256": geometry_hash(expected_map)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["inventory", "generate", "verify"])
    args = parser.parse_args()
    config = read_json(ROOT / CONFIG)
    validate(config)
    preflight = read_json(BASE / "preparation/preflight.json")
    for path, digest in preflight["input_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("frozen input changed: " + path)
    expanded_path = BASE / "preparation/expanded_dataset_config.json"
    if sha256_file(expanded_path) != preflight["expanded_dataset_sha256"]:
        raise ValueError("expanded dataset config changed")
    expanded = read_json(expanded_path)
    if expanded != dataset_config(config, read_json(ROOT / config["base_dataset_config"])):
        raise ValueError("derived config mismatch")
    audit = BASE / "dataset-audit"
    history_path = audit / "historical_inputs.jsonl"
    seal_path = audit / "inventory.json"
    if args.phase == "inventory":
        audit.mkdir(exist_ok=False)
        paths = sorted(input_paths(ROOT))
        print(f"Inventory: {len(paths)} input files, 20 read workers", flush=True)
        with ThreadPoolExecutor(max_workers=20) as pool:
            records = list(pool.map(lambda p: inspect_input(p, ROOT), paths))
        # Exclude the newly preregistered config, not older inputs using its seeds.
        records = [r for r in records if r["path"] != CONFIG]
        schedule = assert_disjoint(expanded, records)
        write_jsonl(history_path, records)
        seal = {"schema": "lns2.confirmation_input_inventory.v1", "input_count": len(records),
                "map_count": sum(r.get("geometry_sha256") is not None for r in records),
                "scope": "accessible retained input files only; not deleted or archived Git history; no labels",
                "excluded_directory_names": sorted(SKIP), "seed_schedule": schedule,
                "history_sha256": sha256_file(history_path), "config_sha256": sha256_file(ROOT / CONFIG)}
        write_json(seal_path, seal)
        print(json.dumps(seal, indent=2))
        return
    seal = read_json(seal_path)
    registration = read_json(ROOT / "artifacts/warehouse-repair-confirmation-v1/dataset_inventory.json")
    if seal["history_sha256"] != registration["inventory_sha256"]:
        raise ValueError("inventory differs from preregistered artifact")
    if seal["history_sha256"] != sha256_file(history_path) or seal["config_sha256"] != sha256_file(ROOT / CONFIG):
        raise ValueError("historical inventory seal mismatch")
    history = [json.loads(line) for line in history_path.read_text().splitlines()]
    expected = assert_disjoint(expanded, history)
    destination = ROOT / config["dataset"]["output"]
    if args.phase == "generate":
        if destination.exists():
            raise ValueError("refusing overwrite or seed reroll; use verify on completed output")
        # An interrupted/failed generation deliberately leaves its output for audit.
        write_json(audit / "generation_status.json", {"status": "started", "solver_calls": 0})
        try:
            print("Generating fixed 8-map / 16-task cohort; no solver calls", flush=True)
            generate_dataset(expanded)
        except BaseException as error:
            write_json(audit / "generation_status.json", {"status": "failed", "error": str(error), "solver_calls": 0})
            raise
        write_json(audit / "generation_status.json", {"status": "generated_pending_validation", "solver_calls": 0})
    split_root = destination / "confirmation"
    rows = [json.loads(line) for line in (split_root / "manifest.jsonl").read_text().splitlines()]
    if len(rows) != 16 or len({r["task_id"] for r in rows}) != 16 or len({r["map_id"] for r in rows}) != 8:
        raise ValueError("cohort size mismatch")
    for index, spec in enumerate(expected):
        pair = rows[2 * index:2 * index + 2]
        if ([r["task_seed"] for r in pair] != spec["task_seeds"] or
                any(r["map_seed"] != spec["map_seed"] for r in pair) or
                len({r["map_id"] for r in pair}) != 1):
            raise ValueError("generated seed schedule mismatch")
    print("Validating 16 task bundles with up to 20 independent workers", flush=True)
    with ProcessPoolExecutor(max_workers=20) as pool:
        tasks = list(pool.map(verify_task, rows, repeat(split_root)))
    maps = {t["map_id"]: t["geometry_sha256"] for t in tasks}
    if len(set(maps.values())) != 8 or set(maps.values()) & {r.get("geometry_sha256") for r in history}:
        raise ValueError("new or retained historical geometry duplicated; no replacement permitted")
    files = sorted(p for p in destination.rglob("*") if p.is_file())
    report = {"schema": "lns2.confirmation_dataset_validation.v1", "status": "dataset_validated_not_ready_for_timing",
              "map_count": 8, "task_count": 16, "solver_calls": 0, "timed_episodes": 0,
              "historical_inventory_sha256": sha256_file(history_path), "tasks": tasks,
              "files": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in files},
              "seed_overlap": 0, "geometry_overlap": 0,
              "remaining": ["construct_and_validate_checkpoints", "runner_clock_tests", "timing_authorization"]}
    report_path = audit / "dataset_validation.json"
    if report_path.exists() and read_json(report_path) != report:
        raise ValueError("validated dataset changed; refusing to overwrite frozen validation report")
    write_json(report_path, report)
    write_json(audit / "generation_status.json", {"status": "validated", "solver_calls": 0})
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
