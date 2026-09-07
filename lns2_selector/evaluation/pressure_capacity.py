"""Read-only rejection-sampler diagnostics; never repair or replace a task."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from experiments._common import read_json, read_jsonl, sha256_file, write_json
from generators.models import MapData
from generators.task_flows import _distance_map, generate_tasks
from lns2_selector.evaluation.path_quality_preflight import contained
from lns2_selector.evaluation.path_quality_pressure import prepare_pressure


def eligible_edges(map_data, pools, od_matrix, bottlenecks, minimum, maximum):
    """Enumerate every allowed OD pair under the original shortest-route rule."""
    hotspot_distances = [_distance_map(map_data, cell) for cell in bottlenecks]
    cache, by_flow = {}, {}
    for flow, weight in sorted(od_matrix.items()):
        if weight <= 0:
            continue
        origin, destination = flow.split("->")
        edges = set()
        for start in pools[origin]:
            if start not in cache:
                cache[start] = _distance_map(map_data, start)
            for goal in pools[destination]:
                length = cache[start].get(goal)
                if start == goal or length is None or length < minimum or (maximum is not None and length > maximum):
                    continue
                if any(start in d and goal in d and d[start] + d[goal] == length for d in hotspot_distances):
                    edges.add((start, goal))
        by_flow[flow] = edges
    return by_flow


def matching_capacity(edges):
    # scipy is already installed in the Windows research environment.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching

    if not edges:
        return 0
    starts = {cell: i for i, cell in enumerate(sorted({a for a, _ in edges}))}
    goals = {cell: i for i, cell in enumerate(sorted({b for _, b in edges}))}
    ordered = sorted(edges)
    graph = csr_matrix(([1] * len(ordered), ([starts[a] for a, _ in ordered],
                                            [goals[b] for _, b in ordered])),
                       shape=(len(starts), len(goals)))
    return int((maximum_bipartite_matching(graph, perm_type="column") >= 0).sum())


def failure_frame(map_data, task_config, case, expected_error):
    try:
        generate_tasks(map_data, task_config, case["task_seed"], case["task_id"])
    except ValueError as error:
        if str(error) != expected_error:
            raise ValueError("original generation failure did not reproduce") from error
        trace = error.__traceback__
        while trace is not None:
            if trace.tb_frame.f_code is generate_tasks.__code__:
                return dict(trace.tb_frame.f_locals)
            trace = trace.tb_next
        raise ValueError("missing generator traceback frame") from error
    raise ValueError("expected failure unexpectedly succeeded")


def analyze_failure(map_data, task_config, case, expected_error):
    frame = failure_frame(map_data, task_config, case, expected_error)
    if frame["hotspot_skew"] != 0 or not frame["od_matrix"] or frame["agent"] >= frame["bottleneck_agent_count"]:
        raise ValueError("unsupported diagnostic domain")
    by_flow = eligible_edges(map_data, frame["pools"], frame["od_matrix"], frame["bottlenecks"],
                             frame["minimum_distance"], frame["maximum_distance"])
    full = set().union(*by_flow.values())
    remaining = {(a, b) for a, b in full if a not in frame["used_starts"] and b not in frame["used_goals"]}
    total_capacity, remaining_capacity = matching_capacity(full), matching_capacity(remaining)
    required = frame["bottleneck_agent_count"]
    if total_capacity < required:
        decision = "constrained_endpoint_capacity_insufficient"
    elif not remaining:
        decision = "greedy_prefix_exhausted_candidates_despite_global_constrained_capacity"
    else:
        decision = "rejection_sampling_missed_existing_valid_pair"
    return {
        "case": case, "reproduced_error": expected_error, "failed_agent_index": frame["agent"],
        "agent_count": frame["agent_count"], "required_constrained_agents": required,
        "accepted_flow_counts": dict(Counter(frame["assignments"])),
        "full_eligible_od_pairs": len(full), "full_constrained_matching_capacity": total_capacity,
        "remaining_eligible_od_pairs": len(remaining), "remaining_matching_capacity": remaining_capacity,
        "valid_next_pair_exists_but_was_not_sampled": bool(remaining),
        "constrained_agents_still_needed": required - frame["agent"],
        "by_flow": {name: {"full_pairs": len(edges), "remaining_pairs": len(edges & remaining)}
                    for name, edges in by_flow.items()},
        "decision": decision,
        "boundary": "matching covers constrained endpoints only; it does not prove a complete task or MAPF feasibility",
        "replacement_task_generated": False,
    }


def diagnose(root: Path, config_path: Path, output: Path):
    if not output.resolve().is_relative_to((root / "build").resolve()):
        raise ValueError("diagnostic output must be inside build")
    prepared = prepare_pressure(root, config_path, verify=True)
    config = read_json(config_path)
    registration = read_json(contained(root, config["output"]) / "registration.json")
    sources = {row["map_id"]: row for row in read_jsonl(
        contained(root, config["source_manifest"]["manifest"]))}
    source_root = contained(root, config["source_manifest"]["manifest"]).parent
    results = []
    for failed in prepared["cases"]:
        if failed["status"] != "generation_error":
            continue
        case = failed["case"]
        document = read_json(source_root / sources[case["map_id"]]["map_metadata_file"])
        map_data = MapData(document["map_id"], document["seed"], document["grid"], document["metadata"])
        task_config = {**config["task"], "agent_density": case["density"],
                       "required_bottleneck_crossing_ratio": case["mode"]["required_bottleneck_crossing_ratio"]}
        result = analyze_failure(map_data, task_config, case, failed["error"])
        results.append(result)
        print(f"{case['task_id']}: {result['decision']}; remaining_pairs={result['remaining_eligible_od_pairs']}", flush=True)
    import scipy
    report = {"schema": "lns2.pressure_capacity_diagnostic.v1", "source_fingerprint": prepared["fingerprint"],
              "input_sha256": registration["inputs"], "implementation_sha256": sha256_file(Path(__file__)),
              "scipy_version": scipy.__version__, "results": results,
              "solver_calls": 0, "timed_episodes_started": 0}
    write_json(output / "capacity_report.json", report)
    return report
