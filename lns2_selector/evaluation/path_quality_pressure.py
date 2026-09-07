"""Static pressure Pilot preparation. No native solver or timed runner is imported."""

from __future__ import annotations

from pathlib import Path

from experiments._common import json_fingerprint, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from generators.io import task_document, write_movingai_scen
from generators.models import MapData
from generators.task_flows import generate_tasks
from generators.validation import validate_task
from lns2_selector.evaluation.path_quality_preflight import audit_task, checked_input, contained, distances, read_grid


def case_plan(config: dict, sources: list[dict]) -> list[dict]:
    if (config.get("schema") != "lns2.path_quality_pressure_pilot.v1"
            or config.get("allow_solver_execution") is not False
            or config.get("role") != "reused_map_design_pilot_not_confirmation"):
        raise ValueError("static-only Pilot contract required")
    if len(sources) != config["expected_maps"] or len({r["map_id"] for r in sources}) != len(sources):
        raise ValueError("source must contain exactly one task per expected map")
    densities, modes = config["densities"], config["modes"]
    if (not densities or len(set(densities)) != len(densities)
            or any(type(d) not in (float, int) or not 0 < d < 1 for d in densities)):
        raise ValueError("invalid density grid")
    if not modes or len({m["name"] for m in modes}) != len(modes):
        raise ValueError("duplicate or empty modes")
    for mode in modes:
        if not 0 <= mode["required_bottleneck_crossing_ratio"] <= 1:
            raise ValueError("invalid bottleneck ratio")
    result, seeds = [], {r["task_seed"] for r in sources}
    for row in sorted(sources, key=lambda r: r["map_id"]):
        for density in densities:
            for mode in modes:
                identity = {"master_seed": config["master_seed"], "map_id": row["map_id"],
                            "density": density, "mode": mode}
                digest = json_fingerprint(identity)
                seed = int(digest[:8], 16) & 0x7fffffff
                if seed in seeds:
                    raise ValueError("task seed collision; do not silently resample")
                seeds.add(seed)
                result.append({**identity, "task_seed": seed,
                               "task_id": f"{row['map_id']}__pressure_{digest[:12]}"})
    return result


def verify_outputs(root: Path, report: dict) -> None:
    for relative, expected in report["output_sha256"].items():
        if sha256_file(contained(root, relative)) != expected:
            raise ValueError(f"prepared output SHA mismatch: {relative}")


def prepare_pressure(root: Path, config_path: Path, *, verify: bool = False) -> dict:
    config = read_json(config_path)
    source_path = checked_input(root, config["source_manifest"])
    sources = read_jsonl(source_path)
    cases = case_plan(config, sources)
    output = contained(root, config["output"])
    if not output.is_relative_to((root / "build").resolve()):
        raise ValueError("Pilot output must be inside build")
    inputs = {config_path.relative_to(root).as_posix(): sha256_file(config_path),
              source_path.relative_to(root).as_posix(): sha256_file(source_path)}
    for directory in (root / "generators",):
        for path in sorted(directory.rglob("*.py")):
            inputs[path.relative_to(root).as_posix()] = sha256_file(path)
    for relative in ("lns2_selector/evaluation/path_quality_pressure.py",
                     "lns2_selector/evaluation/path_quality_preflight.py", "experiments/_common.py"):
        inputs[relative] = sha256_file(root / relative)
    maps = {}
    for row in sources:
        paths = {}
        for field in ("map_file", "map_metadata_file"):
            relative = (source_path.parent.relative_to(root) / row[field]).as_posix()
            paths[field] = contained(root, relative)
            inputs[relative] = sha256_file(paths[field])
        document = read_json(paths["map_metadata_file"])
        if document["map_id"] != row["map_id"] or document["grid"] != read_grid(paths["map_file"]):
            raise ValueError("source map and metadata disagree")
        maps[row["map_id"]] = (MapData(document["map_id"], document["seed"], document["grid"], document["metadata"]), paths)
    registration = {"schema": "lns2.pressure_pilot_registration.v1", "inputs": inputs, "cases": cases}
    fingerprint = json_fingerprint(registration)
    registry_path = output / "registration.json"
    if registry_path.exists():
        if read_json(registry_path) != registration:
            raise ValueError("Pilot inputs changed; use a new output revision")
    elif verify:
        raise ValueError("missing Pilot registration")
    else:
        write_json(registry_path, registration)
    report_path = output / "preparation_report.json"
    if verify:
        report = read_json(report_path)
        if report["fingerprint"] != fingerprint:
            raise ValueError("Pilot report binding mismatch")
        verify_outputs(root, report)
        markers = [read_json(output / "cases" / f"{case['task_id']}.json") for case in cases]
        if markers != report["cases"]:
            raise ValueError("Pilot report/markers disagree")
        return report
    results, output_hashes = [], {}
    for index, case in enumerate(cases):
        marker = output / "cases" / f"{case['task_id']}.json"
        if marker.exists():
            result = read_json(marker)
            if result["fingerprint"] != fingerprint or result["case"] != case:
                raise ValueError("case binding changed")
            verify_outputs(root, result)
        else:
            map_data, paths = maps[case["map_id"]]
            result = {"case": case, "fingerprint": fingerprint, "output_sha256": {}}
            task_config = {**config["task"], "agent_density": case["density"],
                           "required_bottleneck_crossing_ratio": case["mode"]["required_bottleneck_crossing_ratio"]}
            try:
                task = generate_tasks(map_data, task_config, case["task_seed"], case["task_id"])
                validate_task(map_data, task)
                required = task.metadata["required_bottlenecks"]
                expected = round(task.agent_count * case["mode"]["required_bottleneck_crossing_ratio"])
                if sum(c is not None for c in required) != expected:
                    raise ValueError("bottleneck requirement silently dropped")
                # An eligible shortest route is not proof that PP will choose that route.
                caches = {}
                for start, goal, cell, length in zip(task.starts, task.goals, required, task.metadata["actual_shortest_distances"]):
                    if cell is not None:
                        point = tuple(cell)
                        if point not in caches:
                            caches[point] = distances(map_data.grid, point)
                        if caches[point].get(start, -1) + caches[point].get(goal, -1) != length:
                            raise ValueError("bottleneck shortest-route constraint violated")
                task_path = output / "instances" / f"{task.task_id}.json"
                scen_path = task_path.with_suffix(".scen")
                write_json(task_path, task_document(task))
                write_movingai_scen(scen_path, map_data, task)
                quality = audit_task(paths["map_file"], scen_path, task_path, task.agent_count)
                result.update(status="static_valid", quality=quality, constrained_agents=expected,
                              files={"map_file": paths["map_file"].relative_to(root).as_posix(),
                                     "task_file": task_path.relative_to(root).as_posix(),
                                     "scenario_file": scen_path.relative_to(root).as_posix()})
                result["output_sha256"] = {p.relative_to(root).as_posix(): sha256_file(p) for p in (task_path, scen_path)}
            except ValueError as error:
                result.update(status="generation_error", error=str(error))
            write_json(marker, result)
        results.append(result)
        output_hashes.update(result["output_sha256"])
        output_hashes[marker.relative_to(root).as_posix()] = sha256_file(marker)
        print(f"prepare {index + 1}/{len(cases)} {case['task_id']} {result['status']}", flush=True)
    write_jsonl(output / "manifest.jsonl", results)
    output_hashes[(output / "manifest.jsonl").relative_to(root).as_posix()] = sha256_file(output / "manifest.jsonl")
    errors = sum(r["status"] != "static_valid" for r in results)
    report = {"schema": "lns2.pressure_pilot_report.v1", "fingerprint": fingerprint,
              "status": "static_ready_reset_pending" if not errors else "blocked_generation_errors",
              "map_count": len(maps), "task_count": len(cases), "generation_errors": errors,
              "planned_reset_count": len(cases) * len(config["solver_seeds_for_later_reset"]),
              "solver_calls": 0, "timed_episodes_started": 0, "timing_authorized": False,
              "conflict_pressure_verified": False, "joint_feasibility_proven": False,
              "cases": results, "output_sha256": output_hashes}
    write_json(report_path, report)
    return report
