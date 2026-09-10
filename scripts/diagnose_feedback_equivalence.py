"""Bounded native contract diagnostic; not a controller or a TTF benchmark."""

import argparse
import json
from collections import defaultdict
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.closed_loop_trace_storage import apply_state_delta, iter_trace_events, read_state_blob
from experiments.repair_collection import (
    _CollectionRunLock, _make_environment, _plain, _run_jobs, state_fingerprint,
)
from scripts.audit_feedback_memory import BASE, SOURCE_SHA, checked, condition_keys, digest
from scripts.run_warehouse_repair_confirmation import environment_config, install_native, task_plan
from scripts.warehouse_repair_confirmation_runtime import verify_prepared

OUT = ROOT / "build/feedback-native-equivalence-v1"
AUDIT = ROOT / "build/feedback-memory-audit-v2/report.json"
AUDIT_SHA = "3457c9b43ebeb5f045ef3101e5f146674767c7c5cb506aeacb1a28267e1d38a1"
PP_SECONDS = 5.0
WORKERS = 20


def snapshot_blob_path(item, relative_path):
    root = (BASE / "timed" / item["job_id"] / "first_phase").resolve()
    path = (root / relative_path).resolve()
    checked(path.is_relative_to(root), "state blob escapes episode directory")
    return path


def select_pairs(cases):
    pairs = []
    for checkpoint in ("repair-confirm-09", "repair-confirm-13", "repair-confirm-22"):
        case = next(c for c in cases if c["checkpoint"] == checkpoint and c["controller"] == "dual16")
        found = None
        for right in case["rows"]:
            left = next((r for r in case["rows"] if r["decision"] < right["decision"] and
                         r["repair_problem"] == right["repair_problem"] and r["paths"] != right["paths"]), None)
            if left:
                found = (left, right)
                break
        checked(found is not None, "different-incumbent pair unavailable")
        pairs.append((case, *found, "slow_different_incumbents"))
    for case in sorted(cases, key=lambda c: (c["checkpoint"], c["controller"])):
        if case["checkpoint"] in ("repair-confirm-09", "repair-confirm-13", "repair-confirm-22"):
            continue
        found = None
        for right in case["rows"]:
            if right["after_conflicts"] >= right["before_conflicts"] or right["censored"]:
                continue
            left = next((r for r in case["rows"] if r["decision"] < right["decision"] and
                         r["repair_problem"] == right["repair_problem"] and not r["censored"] and
                         r["after_conflicts"] == r["before_conflicts"]), None)
            if left:
                found = (left, right)
                break
        if found:
            pairs.append((case, *found, "historical_recovery_control"))
        if len(pairs) == 5:
            break
    checked(len(pairs) == 5, "exactly two recovery controls required")
    return pairs


def prepare():
    checked(not (OUT / "plan.json").exists(), "plan exists; use collect/resume")
    checked(sha256_file(AUDIT) == AUDIT_SHA, "audit changed")
    checked(sha256_file(BASE / "timed/confirmation_report.json") == SOURCE_SHA, "formal report changed")
    config, specs = verify_prepared(require_ready=True)
    groups = []
    for index, (case, left, right, role) in enumerate(select_pairs(read_json(AUDIT)["cases"])):
        spec = next(s for s in specs if s["item"]["checkpoint_id"] == case["checkpoint"] and
                    s["item"]["controller"] == case["controller"])
        trace = ROOT / case["trace"]
        checked(sha256_file(trace) == case["trace_sha256"], "trace changed")
        snapshots, prefix, state, initial = {}, [], None, None
        for event in iter_trace_events(trace):
            if event["event"] == "initial":
                initial = state = read_state_blob(snapshot_blob_path(spec["item"], event["state_blob"]))
                checked(state_fingerprint(state) == event["state_fingerprint"], "initial mismatch")
            elif event["event"] == "transition":
                checked(state_fingerprint(state) == event["before_fingerprint"], "before mismatch")
                if event["decision_index"] in (left["decision"], right["decision"]):
                    snapshots[str(event["decision_index"])] = {"state": state, "prefix": list(prefix)}
                after = apply_state_delta(state, event["state_delta"])
                checked(state_fingerprint(after) == event["after_fingerprint"], "after mismatch")
                prefix.append({k: event[k] for k in ("action", "metrics", "before_fingerprint", "after_fingerprint")})
                state = after
                if len(snapshots) == 2:
                    break
        checked(len(snapshots) == 2, "snapshot missing")
        a, b = [snapshots[str(r["decision"])]["state"] for r in (left, right)]
        protocol = {"mode": "explicit_neighborhood"}
        checked(condition_keys(a, left["agents"], PP_SECONDS, protocol, index)["repair_problem"] ==
                condition_keys(b, right["agents"], PP_SECONDS, protocol, index)["repair_problem"],
                "repair conditions differ")
        groups.append({"group_id": index, "checkpoint": case["checkpoint"], "controller": case["controller"],
                       "role": role, "agents": left["agents"], "snapshots": snapshots, "initial": initial,
                       "restore_seed": spec["worker_job"]["episode_override"]["initial_restore"]["restore_seed"],
                       "task_id": spec["item"]["task_id"], "trace": case["trace"],
                       "trace_sha256": case["trace_sha256"], "result_sha256": case["result_sha256"]})
    plan = {"schema": "lns2.feedback_native_equivalence.v1", "audit_sha256": AUDIT_SHA,
            "source_sha256": SOURCE_SHA, "config": config, "groups": groups, "jobs": 80,
            "workers": WORKERS, "pp_seconds": PP_SECONDS, "process_fuse_seconds": 120,
            "new_ttf_episodes": 0, "diagnostic_only": True,
            "implementation_sha256": sha256_file(Path(__file__)),
            "memory_key_implementation_sha256": sha256_file(ROOT / "scripts/audit_feedback_memory.py"),
            "selection_rule": "first differing-incumbent repeated repair key for each of three slow cases; first two other checkpoint/controller pairs with historical recovery",
            "gate": "all 20 group/order/seed cells uncensored and outcome/path equal across two snapshots and two restore modes; search counters diagnostic only; rollback compares restored inputs not equality across different old paths"}
    write_json(OUT / "plan.json", plan)
    return {"jobs": 80, "maximum_new_repairs": 80, "prefix_replay_repairs": sum(
        len(s["prefix"]) * 4 for g in groups for s in g["snapshots"].values()),
        "groups": [{k: g[k] for k in ("group_id", "checkpoint", "controller", "role")} |
                   {"decisions": sorted(g["snapshots"])} for g in groups],
        "plan_sha256": sha256_file(OUT / "plan.json")}


def jobs_for(plan):
    for group in plan["groups"]:
        for snapshot, restore, reverse, trial in product(sorted(group["snapshots"]),
                                                         ("direct", "prefix"), (False, True), range(2)):
            seed = int(digest(["feedback-native-equivalence-v1", group["group_id"], trial])[:8], 16) & 0x7fffffff
            yield {"job_id": f"g{group['group_id']}-d{snapshot}-{restore}-r{int(reverse)}-t{trial}",
                   "group": group, "snapshot": snapshot, "restore": restore, "trial": trial,
                   "order": sorted(group["agents"], reverse=reverse), "seed": seed,
                   "config": plan["config"], "plan_sha256": sha256_file(OUT / "plan.json")}


def paths(state, members=None):
    return [(a["id"], a["path"]) for a in sorted(state["agents"], key=lambda a: a["id"])
            if members is None or a["id"] in members]


def worker(job):
    config, group = job["config"], job["group"]
    install_native(config)
    task = next(t["row"] for t in task_plan(config) if t["row"]["task_id"] == group["task_id"])
    env = _make_environment(str(ROOT / config["dataset"]["output"]), task,
                            {**environment_config(config), "time_limit": 600}, "Adaptive")
    snapshot = group["snapshots"][job["snapshot"]]
    target = snapshot["state"]
    restored = target if job["restore"] == "direct" else group["initial"]
    state = _plain(env.reset_paths([p for _, p in paths(restored)], seed=group["restore_seed"]))
    checked(paths(state) == paths(restored), "reset paths mismatch")
    if job["restore"] == "prefix":
        checked(state_fingerprint(state) == state_fingerprint(restored), "reset full identity")
        for event in snapshot["prefix"]:
            checked(state_fingerprint(state) == event["before_fingerprint"], "prefix before identity")
            step = _plain(env.step_with_time_limit(event["action"], event["metrics"]["requested_pp_time_limit_seconds"]))
            state = step["observation"]
            checked(state_fingerprint(state) == event["after_fingerprint"] and
                    step["metrics"]["repair_order"] == event["metrics"]["repair_order"], "prefix replay mismatch")
        checked(state_fingerprint(state) == state_fingerprint(target), "target full identity")
    action = {"mode": "explicit_neighborhood", "agents": group["agents"], "repair_order": job["order"],
              "random_seed": job["seed"], "pp_random_seed": job["seed"]}
    protocol = {"mode": "explicit_neighborhood", "repair_order": job["order"]}
    key = condition_keys(state, group["agents"], PP_SECONDS, protocol, group["group_id"])["repair_problem_budget"]
    checked(key == condition_keys(target, group["agents"], PP_SECONDS, protocol, group["group_id"])["repair_problem_budget"],
            "restored repair condition mismatch")
    step = _plain(env.step_with_time_limit(action, PP_SECONDS))
    metrics, after = step["metrics"], step["observation"]
    checked(metrics["action_valid"] and metrics["step_applied"] and
            sorted(metrics["neighborhood"]) == sorted(group["agents"]), "explicit action modified")
    checked(metrics["repair_order"] == job["order"] and metrics["applied_pp_random_seed"] == job["seed"],
            "fixed order or random stream changed")
    members = set(group["agents"])
    external = {a["id"] for a in state["agents"]} - members
    checked(paths(state, external) == paths(after, external), "external paths modified")
    if metrics["pp_rolled_back"]:
        checked(paths(state) == paths(after), "rollback changed original paths")
    reason = metrics["pp_failure_reason"]
    censored = reason not in ("none", "conflict_bound_exceeded") or metrics["native_replan_seconds"] >= PP_SECONDS
    signature = {"accepted": metrics["replan_success"], "rollback": metrics["pp_rolled_back"],
                 "failure_reason": reason, "attempt_conflicts": metrics["pp_attempt_conflict_pair_count"],
                 "failed_agent": metrics["pp_failed_agent"], "attempted_agents": metrics["pp_attempted_agent_count"],
                 "after_conflicts": after["num_of_colliding_pairs"], "feasible": after["feasible"],
                 "accepted_paths": digest(paths(after, members)) if metrics["replan_success"] else None}
    return {"status": "ok", "job_id": job["job_id"], "plan_sha256": job["plan_sha256"],
            "group_id": group["group_id"], "snapshot": job["snapshot"], "restore": job["restore"],
            "order": job["order"], "seed": job["seed"], "trial": job["trial"], "condition_key": key,
            "signature": signature, "censored": censored, "metrics": metrics,
            "search_delta": {k: after["low_level"][k] - state["low_level"][k] for k in ("generated", "expanded", "runs")},
            "selected_paths_before": paths(state, members), "selected_paths_after": paths(after, members)}


def verified_plan():
    plan = read_json(OUT / "plan.json")
    checked(sha256_file(Path(__file__)) == plan["implementation_sha256"], "implementation changed")
    checked(sha256_file(ROOT / "scripts/audit_feedback_memory.py") == plan["memory_key_implementation_sha256"], "key implementation changed")
    checked(sha256_file(AUDIT) == AUDIT_SHA and sha256_file(BASE / "timed/confirmation_report.json") == SOURCE_SHA,
            "source changed")
    config, _ = verify_prepared(require_ready=True)
    checked(config == plan["config"], "frozen config changed")
    for group in plan["groups"]:
        trace = ROOT / group["trace"]
        checked(sha256_file(trace) == group["trace_sha256"], "source trace changed")
    return plan


def validate_result(result, job):
    checked(result["status"] == "ok", "failed outcome; inspect before resume")
    expected = {"job_id": job["job_id"], "plan_sha256": job["plan_sha256"],
                "group_id": job["group"]["group_id"], "snapshot": job["snapshot"],
                "restore": job["restore"], "order": job["order"], "seed": job["seed"],
                "trial": job["trial"]}
    checked(all(result.get(k) == v for k, v in expected.items()), "outcome identity mismatch")


def collect():
    plan = verified_plan()
    jobs = list(jobs_for(plan))
    pending = []
    for job in jobs:
        output = OUT / "outcomes" / (job["job_id"] + ".json")
        if output.exists():
            old = read_json(output)
            validate_result(old, job)
        else:
            pending.append(job)
    def save(result):
        write_json(OUT / "outcomes" / (result["job_id"] + ".json"), result)
        print(result["job_id"], result["status"], flush=True)
    def failure(job, status, error):
        return {"job_id": job["job_id"], "plan_sha256": job["plan_sha256"], "status": status, "error": error}
    with _CollectionRunLock(OUT, sha256_file(OUT / "plan.json"), "feedback-equivalence"):
        for offset in range(0, len(pending), WORKERS):
            if (OUT / "STOP").exists():
                return {"status": "safe_stop", "remaining": len(pending) - offset}
            _run_jobs(worker, pending[offset:offset + WORKERS], WORKERS, phase="feedback-equivalence",
                      output_root=OUT, run_fingerprint=sha256_file(OUT / "plan.json"), timeout_seconds=120,
                      on_result=save, failure_result=failure, stop_on_failure=True)
    return {"status": "complete", "jobs": len(jobs)}


def compare_cell(values):
    checked(len(values) == 4, "missing paired conditions")
    snapshots = {v["snapshot"] for v in values}
    checked(len(snapshots) == 2 and {(v["snapshot"], v["restore"]) for v in values} ==
            set(product(snapshots, ("direct", "prefix"))), "duplicate or invalid paired condition")
    return {"censored": any(v["censored"] for v in values),
            "same_condition": len({v["condition_key"] for v in values}) == 1,
            "same_outcome": len({digest(v["signature"]) for v in values}) == 1,
            "same_search_counters": len({digest(v["search_delta"]) for v in values}) == 1}


def analyze():
    plan = verified_plan()
    cells = defaultdict(list)
    inputs = {}
    for job in jobs_for(plan):
        path = OUT / "outcomes" / (job["job_id"] + ".json")
        result = read_json(path)
        validate_result(result, job)
        inputs[path.relative_to(OUT).as_posix()] = sha256_file(path)
        cells[(result["group_id"], tuple(result["order"]), result["trial"])].append(result)
    comparisons = [{"group_id": key[0], "order": list(key[1]), "trial": key[2], **compare_cell(values)}
                   for key, values in sorted(cells.items())]
    checked(len(comparisons) == 20, "cell count mismatch")
    passed = all(c["same_condition"] and c["same_outcome"] and not c["censored"] for c in comparisons)
    report = {"schema": plan["schema"], "plan_sha256": sha256_file(OUT / "plan.json"),
              "jobs": len(inputs), "new_ttf_episodes": 0, "comparisons": comparisons,
              "outcome_sha256": inputs,
              "decision": "bounded_native_equivalence_supported_not_controller_promotion" if passed else
                          "stop_feedback_implementation_explain_native_difference",
              "limits": "Selected retrospective states and fixed orders/seeds only; not proof of equality for all states or of feedback-controller benefit."}
    write_json(OUT / "report.json", report)
    return {"decision": report["decision"], "jobs": len(inputs), "cells": len(comparisons),
            "same_outcome_cells": sum(c["same_outcome"] for c in comparisons),
            "same_search_counter_cells": sum(c["same_search_counters"] for c in comparisons),
            "censored_cells": sum(c["censored"] for c in comparisons),
            "report_sha256": sha256_file(OUT / "report.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "collect": collect, "analyze": analyze}[args.phase](), indent=2))
