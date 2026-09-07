"""Reset-only Pilot diagnostics. No controller steps or performance comparison."""

from __future__ import annotations

import multiprocessing
from pathlib import Path

from experiments._common import json_fingerprint, read_json, sha256_file, write_json, write_jsonl
from experiments.repair_collection import _CollectionRunLock, state_fingerprint
from lns2_selector.evaluation.path_quality_preflight import audit_paths, contained, read_grid
from lns2_selector.evaluation.path_quality_pressure import prepare_pressure
from lns2_selector.solver.native import load_native_module, native_identity


def pressure_metrics(state, task, grid):
    agents = sorted(state["agents"], key=lambda a: a["id"])
    count = len(agents)
    if [a["id"] for a in agents] != list(range(count)) or count != len(task["starts"]):
        raise ValueError("agent identity mismatch")
    if state["iteration"] != 0 or not state["initial_solution_complete"]:
        raise ValueError("reset returned incomplete paths or performed repairs")
    paths = [a["path"] for a in agents]
    quality = audit_paths(grid, task["starts"], task["goals"], paths)
    if quality["colliding_pairs"] != state["num_of_colliding_pairs"]:
        raise ValueError("reconstructed conflicts disagree with native")
    adjacency = {i: set() for i in range(count)}
    edges = set()
    for a, b in state["conflict_edges"]:
        if a not in adjacency or b not in adjacency or a == b:
            raise ValueError("invalid conflict edge")
        edges.add(tuple(sorted((a, b))))
        adjacency[a].add(b)
        adjacency[b].add(a)
    if len(edges) != quality["colliding_pairs"]:
        raise ValueError("native graph/count mismatch")
    active = {i for i, neighbors in adjacency.items() if neighbors}
    unseen, components = set(active), []
    while unseen:
        frontier = [unseen.pop()]
        size = 0
        while frontier:
            current = frontier.pop()
            size += 1
            neighbors = adjacency[current] & unseen
            unseen.difference_update(neighbors)
            frontier.extend(neighbors)
        components.append(size)
    required = task["metadata"]["required_bottlenecks"]
    cols = len(grid[0])
    constrained = [(path, cell) for path, cell in zip(paths, required) if cell is not None]
    visited = sum(cell[0] * cols + cell[1] in path for path, cell in constrained)
    return {**quality, "agent_count": count, "conflict_density": 2 * len(edges) / max(1, count * (count - 1)),
            "active_conflict_agents": len(active), "conflict_component_sizes": sorted(components, reverse=True),
            "largest_conflict_component": max(components, default=0),
            "constrained_agents": len(constrained), "agents_visiting_designated_bottleneck": visited,
            "designated_bottleneck_visit_fraction": visited / len(constrained) if constrained else None}


def _reset_worker(job):
    root = Path(job["root"])
    try:
        module = load_native_module()
        identity = native_identity(module)
        config = job["config"]
        if (identity["sha256"] != config["native_sha256"]
                or Path(identity["path"]).resolve() != contained(root, config["native_file"])):
            raise ValueError("loaded native identity mismatch")
        row = job["row"]
        files = row["files"]
        environment = module.LNS2RepairEnv(
            str(contained(root, files["map_file"])), str(contained(root, files["scenario_file"])),
            agent_count=row["quality"]["agent_count"], time_limit=config["initial_planning_budget_seconds"],
            neighborhood_size=8, destroy_strategy="Adaptive", replan_algorithm="PP", use_sipp=True,
            max_repair_iterations=0, screen=0, context={})
        state = environment.reset(seed=job["seed"])
        quality = pressure_metrics(state, read_json(contained(root, files["task_file"])),
                                   read_grid(contained(root, files["map_file"])))
        result = {"status": "ok", "observation": state, "quality": quality,
                  "state_fingerprint": state_fingerprint(state)}
    except Exception as error:
        result = {"status": "error", "error": str(error), "error_type": type(error).__name__}
    result.update(binding=job["binding"], key=job["key"], repairs_executed=0)
    write_json(Path(job["output"]), result)


def run_resets(root, config_path):
    config = read_json(config_path)
    if (config["schema"] != "lns2.pressure_reset_only.v1" or config["workers"] != 1
            or config["repairs_allowed"] is not False or config["formal_timing_allowed"] is not False):
        raise ValueError("reset-only contract changed")
    preparation_config = contained(root, config["preparation_config"])
    prepared = prepare_pressure(root, preparation_config, verify=True)
    if prepared["generation_errors"] or prepared["status"] != "static_ready_reset_pending":
        raise ValueError("static Pilot has unresolved generation errors")
    source_config = read_json(preparation_config)
    seeds = source_config["solver_seeds_for_later_reset"]
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("invalid reset seeds")
    identity = native_identity()
    if (identity["sha256"] != config["native_sha256"]
            or Path(identity["path"]).resolve() != contained(root, config["native_file"])):
        raise ValueError("wrong native loaded before reset")
    output = contained(root, config["output"])
    if not output.is_relative_to((root / "build").resolve()):
        raise ValueError("reset output must be in build")
    registration = {"preparation_fingerprint": prepared["fingerprint"],
                    "prepared_report_sha256": sha256_file(contained(root, source_config["output"]) / "preparation_report.json"),
                    "config_sha256": sha256_file(config_path), "native_sha256": identity["sha256"],
                    "implementation_sha256": sha256_file(Path(__file__)),
                    "dependency_sha256": {relative: sha256_file(root / relative) for relative in (
                        "experiments/repair_collection.py", "lns2_selector/solver/native.py")}}
    binding = json_fingerprint(registration)
    ctx, rows = multiprocessing.get_context("spawn"), []
    expected = len(prepared["cases"]) * len(seeds)
    with _CollectionRunLock(output, binding, "pressure-reset-only"):
        registry = output / "registration.json"
        if registry.exists() and read_json(registry) != registration:
            raise ValueError("reset registration changed; create new run revision")
        write_json(registry, registration)
        write_json(output / "run_status.json", {"status": "running", "expected": expected, "binding": binding})
        try:
            for row in prepared["cases"]:
                for seed in seeds:
                    if (output / "STOP_AFTER_RESET").exists():
                        raise InterruptedError("safe stop requested between resets")
                    key = json_fingerprint([row["case"]["task_id"], seed])[:24]
                    path = output / "resets" / f"{key}.json"
                    if not path.exists():
                        job = {"root": str(root), "row": row, "seed": seed, "key": key,
                               "output": str(path), "config": config, "binding": binding}
                        process = ctx.Process(target=_reset_worker, args=(job,))
                        process.start()
                        try:
                            process.join(config["reset_process_timeout_seconds"])
                            if process.is_alive():
                                process.terminate()
                                process.join(5)
                                if process.is_alive():
                                    process.kill()
                                    process.join()
                                write_json(path, {"status": "error", "error": "reset process timeout",
                                                  "binding": binding, "key": key, "repairs_executed": 0})
                            elif process.exitcode != 0 or not path.exists():
                                write_json(path, {"status": "error", "error": "abnormal reset exit",
                                                  "binding": binding, "key": key, "repairs_executed": 0})
                        finally:
                            if process.is_alive():
                                process.terminate()
                                process.join()
                            process.close()
                    saved = read_json(path)
                    if saved.get("binding") != binding or saved.get("key") != key or saved.get("repairs_executed") != 0:
                        raise ValueError("reset result identity mismatch")
                    entry = {"key": key, "case": row["case"], "solver_seed": seed,
                             "reset_file": path.relative_to(root).as_posix(), "sha256": sha256_file(path),
                             "status": saved["status"], "quality": saved.get("quality")}
                    rows.append(entry)
                    write_jsonl(output / "manifest.jsonl", rows)
                    if saved["status"] != "ok":
                        raise ValueError(f"reset failed: {saved.get('error')}")
                    if state_fingerprint(saved["observation"]) != saved["state_fingerprint"]:
                        raise ValueError("saved reset state fingerprint mismatch")
                    print(f"reset {len(rows)}/{expected}: {row['case']['task_id']} seed={seed} conflicts={saved['quality']['colliding_pairs']}", flush=True)
                    write_json(output / "run_status.json", {"status": "running", "completed": len(rows), "expected": expected, "binding": binding})
        except BaseException as error:
            write_json(output / "run_status.json", {"status": "paused" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                       "completed": len(rows), "expected": expected, "binding": binding, "error": str(error)})
            raise
        report = {"schema": "lns2.pressure_reset_report.v1", "binding": binding, "expected": expected,
                  "completed": len(rows), "zero_conflict": sum(r["quality"]["feasible"] for r in rows),
                  "status": "complete_reset_only", "repairs_executed": 0, "timed_episodes_started": 0,
                  "manifest_sha256": sha256_file(output / "manifest.jsonl"), "rows": rows}
        write_json(output / "report.json", report)
        write_json(output / "run_status.json", {"status": "complete", "completed": len(rows), "binding": binding})
    return report
