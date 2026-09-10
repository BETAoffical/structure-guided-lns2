"""Inspect feedback reuse in retained traces; never invoke a solver or policy."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, read_jsonl, sha256_file, write_json
from experiments.closed_loop_trace_storage import (
    apply_state_delta, iter_trace_events, read_state_blob,
)
from experiments.repair_collection import state_fingerprint

BASE = ROOT / "build/warehouse-repair-confirmation-v1"
SOURCE_SHA = "c5fc7e0a14d579e7d411bb613e1cef3b13704ce0b21e038015f195df26b81c8c"


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def checked(condition, message):
    if not condition:
        raise ValueError(message)


def condition_keys(state, agents, budget, protocol, instance):
    """Observable conditions, NOT a proof of native Markov-state equivalence."""
    ids = [a["id"] for a in state["agents"]]
    checked(len(ids) == len(set(ids)), "duplicate agent IDs")
    checked(len(agents) == len(set(agents)) and set(agents) <= set(ids),
            "invalid selected agents")
    checked(bool(agents), "empty selected agents")
    ordered = sorted(state["agents"], key=lambda a: a["id"])
    membership = sorted(agents)
    paths = digest([(a["id"], a["start"], a["goal"], a["path"]) for a in ordered])
    edges = sorted(sorted(e) for e in state["conflict_edges"])
    coarse = digest([instance, membership, edges, protocol])
    full = digest([instance, membership, paths, protocol])
    members = set(membership)
    external = [(a["id"], a["start"], a["goal"], a["path"])
                for a in ordered if a["id"] not in members]
    endpoints = [(a["id"], a["start"], a["goal"])
                 for a in ordered if a["id"] in members]
    bound = sum(bool(set(e) & members) for e in edges)
    # InitLNS removes selected paths before PP; old affected pair COUNT is
    # its acceptance bound. Native cache/tie-order equivalence remains untested.
    repair = digest([instance, external, endpoints, bound, protocol])
    return {"coarse": coarse, "full_paths": full,
            "full_paths_budget": digest([full, budget]), "paths": paths,
            "repair_problem": repair, "repair_problem_budget": digest([repair, budget])}


def summarize(rows):
    counts = Counter(steps=len(rows))
    seen = {name: {} for name in ("coarse", "full_paths", "full_paths_budget",
                                 "repair_problem", "repair_problem_budget")}
    for row in rows:
        drop = row["after_conflicts"] < row["before_conflicts"]
        counts["strict_drops"] += drop
        counts["path_unchanged"] += row["path_unchanged"]
        counts["equal_conflicts_changed_paths"] += (
            row["before_conflicts"] == row["after_conflicts"] and not row["path_unchanged"])
        counts["censored"] += row["censored"]
        counts["repair_problem_repeat_after_nondrop_uncensored"] += (
            not row["censored"] and any(not p["drop"] and not p["censored"]
                                       for p in seen["repair_problem"].get(row["repair_problem"], [])))
        for name, history in seen.items():
            previous = history.get(row[name], [])
            counts[name + "_repeat_steps"] += bool(previous)
            counts[name + "_success_after_prior_nondrop"] += (
                drop and any(not old["drop"] and not old["censored"] for old in previous))
            if name == "coarse":
                counts["coarse_repeat_without_any_full_path_match"] += (
                    bool(previous) and all(p["paths"] != row["paths"] for p in previous))
            history.setdefault(row[name], []).append({
                "drop": drop, "censored": row["censored"], "paths": row["paths"],
            })
    # Include empty categories so zero-iteration episodes have a stable schema.
    for name in seen:
        counts.setdefault(name + "_repeat_steps", 0)
        counts.setdefault(name + "_success_after_prior_nondrop", 0)
    counts.setdefault("coarse_repeat_without_any_full_path_match", 0)
    return dict(counts)


def total_stats(summaries):
    totals = Counter()
    for stats in summaries:
        totals.update(stats)
    return dict(totals)


def inspect(job):
    item, expected = job["item"], job["expected"]
    directory = BASE / "timed" / item["job_id"]
    result_path = directory / "first_phase_result.json"
    result = read_json(result_path)["payload"]
    trace = directory / "first_phase" / result["trace_file"]
    checked(sha256_file(trace) == result["trace_sha256"], "trace SHA mismatch")
    state, instance, finish = None, None, False
    rows = []
    plateau, longest = [], []
    for event in iter_trace_events(trace):
        if event["event"] == "initial":
            checked(state is None, "duplicate initial event")
            state = read_state_blob(directory / "first_phase" / event["state_blob"])
            checked(state_fingerprint(state) == event["state_fingerprint"], "initial identity")
            checked(event["state_fingerprint"] == expected["solver_summary"]["initial_fingerprint"],
                    "report initial identity")
            instance = digest([item["checkpoint_identity_sha256"], state["rows"],
                               state["cols"], state["obstacles"]])
        elif event["event"] == "transition":
            checked(state is not None and not finish, "transition outside episode")
            checked(state_fingerprint(state) == event["before_fingerprint"], "before identity")
            after = apply_state_delta(state, event["state_delta"])
            checked(state_fingerprint(after) == event["after_fingerprint"], "after identity")
            metrics, action = event["metrics"], event["action"]
            checked(metrics["action_valid"] and metrics["step_applied"], "invalid action")
            selected = metrics["neighborhood"]
            checked(action["mode"] == "explicit_neighborhood" and
                    sorted(action["agents"]) == sorted(selected), "changed explicit action")
            checked(action.get("random_seed") is not None, "uncontrolled random stream")
            pool = event["controller"]["candidate_pool"]
            chosen = [c for c in pool if c["candidate_id"] == event["controller"]["selected_candidate_id"]]
            checked(len(chosen) == 1 and sorted(chosen[0]["agents"]) == sorted(selected),
                    "candidate identity")
            before_n, after_n = state["num_of_colliding_pairs"], after["num_of_colliding_pairs"]
            checked(metrics["conflicts_before"] == before_n and metrics["conflicts_after"] == after_n,
                    "conflict metric mismatch")
            budget = metrics["requested_pp_time_limit_seconds"]
            protocol = {k: v for k, v in action.items() if k not in ("random_seed", "agents")}
            keys = condition_keys(state, selected, budget, protocol, instance)
            after_keys = condition_keys(after, selected, budget, protocol, instance)
            reason = metrics["pp_failure_reason"]
            row = {**keys, "decision": event["decision_index"], "agents": sorted(selected),
                   "before_conflicts": before_n, "after_conflicts": after_n,
                   "path_unchanged": keys["paths"] == after_keys["paths"],
                   "censored": reason not in ("none", "conflict_bound_exceeded") or
                   metrics["native_replan_seconds"] >= budget}
            rows.append(row)
            if before_n == after_n:
                plateau.append(row)
                if len(plateau) > len(longest):
                    longest = list(plateau)
            else:
                plateau = []
            state = after
        elif event["event"] == "finish":
            checked(state is not None and not finish, "invalid finish")
            if "final_fingerprint" in event:
                checked(state_fingerprint(state) == event["final_fingerprint"], "finish identity")
            finish = True
    checked(finish, "missing finish")
    checked(len(rows) == expected["solver_summary"]["repair_iterations"], "iteration count")
    checked(state["num_of_colliding_pairs"] == expected["solver_summary"]["final_conflicts"],
            "final conflict count")
    checked(bool(state["feasible"]) == expected["success"], "final feasibility")
    return {"checkpoint": item["checkpoint_id"], "controller": item["controller"],
            "trace": trace.relative_to(ROOT).as_posix(), "trace_sha256": result["trace_sha256"],
            "result_sha256": sha256_file(result_path), "all": summarize(rows),
            "low_conflicts_le3": summarize([r for r in rows if 0 < r["before_conflicts"] <= 3]),
            "longest_plateau": {"first_decision": longest[0]["decision"] if longest else None,
                                **summarize(longest)}, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--output", type=Path, default=ROOT / "build/feedback-memory-audit-v2")
    args = parser.parse_args()
    checked(1 <= args.workers <= 20, "workers must be 1..20")
    checked(not args.output.exists(), "output exists; preserve earlier audit")
    source = BASE / "timed/confirmation_report.json"
    registry = ROOT / "artifacts/warehouse-repair-confirmation-v1/timing_result.json"
    checked(sha256_file(source) == read_json(registry)["report_sha256"] == SOURCE_SHA,
            "frozen formal report changed")
    expected = {(r["checkpoint_id"], r["controller"]): r for r in read_json(source)["rows"]}
    specs_path = BASE / "readiness/episode_specs.jsonl"
    jobs = [{"item": s["item"], "expected": expected[(s["item"]["checkpoint_id"], s["item"]["controller"])]}
            for s in read_jsonl(specs_path) if s["item"]["controller"] in ("dual16", "v2-full")]
    checked(len(jobs) == 64 and len({j["item"]["job_id"] for j in jobs}) == 64, "cohort mismatch")
    cases = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(inspect, jobs):
            cases.append(result)
            print(f"verified {len(cases)}/64 {result['checkpoint']} {result['controller']}", flush=True)
    checked(sha256_file(source) == SOURCE_SHA, "source changed during audit")
    totals = {}
    for controller in ("v2-full", "dual16"):
        selected = [c for c in cases if c["controller"] == controller]
        totals[controller] = {scope: total_stats(c[scope] for c in selected)
                              for scope in ("all", "low_conflicts_le3")}
    source_files = ["third_party/mapf_lns2/src/InitLNS.cpp",
                    "third_party/mapf_lns2/src/PathTable.cpp",
                    "experiments/closed_loop_trace_storage.py", "experiments/repair_collection.py"]
    report = {"schema": "lns2.feedback_memory_observability.v2", "new_solver_calls": 0,
              "new_ttf_episodes": 0, "source_sha256": SOURCE_SHA,
              "registry_sha256": sha256_file(registry), "specs_sha256": sha256_file(specs_path),
              "implementation_sha256": sha256_file(Path(__file__)), "workers": args.workers,
              "inspected_source_sha256": {p: sha256_file(ROOT / p) for p in source_files},
              "totals": totals, "cases": cases,
              "decision": "no_controller_promotion_observability_only",
              "limitations": ["Recorded chosen actions only; no off-policy rollout inference.",
                              "Coarse keys intentionally merge different path conditions.",
                              "Full-path keys do not certify all hidden native state is equivalent.",
                              "Repair-problem keys drop selected OLD paths because PP removes them; native equivalence still untested.",
                              "Seeds are omitted to inspect repeated stochastic trials, not deterministic caching.",
                              "Full-path counts ignoring budget are optimistic reuse counts.",
                              "Equal-conflict path changes are not labeled failed recovery.",
                              "No reward, learned model, intervention threshold or scientific gate fitted."]}
    write_json(args.output / "report.json", report)
    print(json.dumps({"totals": totals, "report_sha256": sha256_file(args.output / "report.json")}, indent=2))


if __name__ == "__main__":
    main()
