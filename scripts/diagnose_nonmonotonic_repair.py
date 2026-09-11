"""Isolated native acceptance pilot. No model training or end-to-end timing."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.nonmonotonic_repair import ARMS, acceptance_draw, temperature, validate_transition, summarize
from experiments.repair_collection import _CollectionRunLock, _plain, _run_jobs, state_fingerprint
from scripts.audit_feedback_memory import checked, digest
from scripts.run_feedback_exploration_diagnostics import (
    FrozenPool, check_source_pool, paths, padded_auc, restore, validate_final,
)

OUT = ROOT / "build/nonmonotonic-repair-pilot-v1"
SOURCE = ROOT / "build/feedback-exploration-diagnostics-v1/plan.json"
SOURCE_SHA = "83af0fd6d70a24517a4f41e2cfdec13c433a841fd118188ddba4111e825dd2a6"
NATIVE = ROOT / "build/linux/nonmonotonic-repair-v1/lns2_env.cpython-310-x86_64-linux-gnu.so"
HORIZON, TRIALS, WORKERS = 64, 4, 20
PP_SECONDS, EPISODE_SECONDS, FUSE_SECONDS, TOTAL_SECONDS = 5.0, 120, 180, 1800


def seed(*parts):
    return int(digest(["nonmonotonic-v1", *parts])[:8], 16) & 0x7fffffff


def prepare():
    checked(not (OUT / "plan.json").exists(), "plan exists; use dry-run/resume")
    checked(sha256_file(SOURCE) == SOURCE_SHA, "source registration changed")
    source = read_json(SOURCE)
    checked(len(source["discovery"]) == 5, "fixed source supply changed")
    config = deepcopy(source["config"])
    config["frozen"]["native_path"] = NATIVE.relative_to(ROOT).as_posix()
    config["frozen"]["native_sha256"] = sha256_file(NATIVE)
    inputs = {SOURCE.relative_to(ROOT).as_posix(): SOURCE_SHA,
              NATIVE.relative_to(ROOT).as_posix(): sha256_file(NATIVE)}
    legacy = ROOT / source["config"]["frozen"]["native_path"]
    checked(sha256_file(legacy) == source["config"]["frozen"]["native_sha256"], "legacy native changed")
    inputs[legacy.relative_to(ROOT).as_posix()] = sha256_file(legacy)
    for folder in ("src", "include", "third_party/mapf_lns2", "lns2_selector", "experiments", "scripts",
                   "artifacts/initlns-closed-loop-controller-v2"):
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    for name in ("docs/NONMONOTONIC_REPAIR_PROTOCOL_ZH.md", "CMakeLists.txt"):
        inputs[name] = sha256_file(ROOT / name)
    for case in source["discovery"]:
        checked(sha256_file(ROOT / case["trace"]) == case["trace_sha256"], "source trace changed")
        inputs[case["trace"]] = case["trace_sha256"]
    plan = {"schema": "lns2.nonmonotonic_repair_pilot.v1", "config": config,
            "legacy_config": source["config"], "cases": source["discovery"], "inputs": inputs,
            "arms": ARMS, "horizon": HORIZON, "trials": TRIALS, "workers": WORKERS,
            "pp_seconds": PP_SECONDS, "episode_seconds": EPISODE_SECONDS,
            "fuse_seconds": FUSE_SECONDS, "total_seconds": TOTAL_SECONDS,
            "temperature": {"initial": 1000, "cooling": 0.99, "offset": "source_native_iteration"},
            "no_ttf_or_training": True}
    write_json(OUT / "plan.json", plan)
    return describe(plan)


def verify():
    plan = read_json(OUT / "plan.json")
    for p, expected in plan["inputs"].items():
        checked(sha256_file(ROOT / p) == expected, "registered input changed: " + p)
    return plan


def describe(plan):
    return {"states": len(plan["cases"]), "jobs": 60, "maximum_new_repairs": 3840,
            "prefix_replays": sum(len(c["prefix"]) for c in plan["cases"]) * TRIALS * len(ARMS),
            "workers": WORKERS, "total_resource_cap_seconds": TOTAL_SECONDS,
            "no_ttf": True, "plan_sha256": sha256_file(OUT / "plan.json")}


def jobs(plan, smoke=False):
    values = []
    for case in plan["cases"]:
        for trial in range(1 if smoke else TRIALS):
            for arm in (*ARMS, "legacy") if smoke else ARMS:
                values.append({"job_id": f"{case['case_id']}-{arm}-{trial}",
                    "case": case, "trial": trial, "arm": arm,
                    "config": plan["legacy_config"] if arm == "legacy" else plan["config"],
                    "horizon": 2 if smoke else HORIZON,
                    "plan_sha256": sha256_file(OUT / "plan.json")})
    return values


def worker(job):
    env, state = restore(job)
    case, arm = job["case"], job["arm"]
    selector = FrozenPool(case)
    index, pool = selector.select(env, state, case["decision"])
    check_source_pool(case, index, pool)
    initial = state
    transitions, conflicts = [], [state["num_of_colliding_pairs"]]
    began = time.monotonic()
    censored, stop = False, "step_limit"
    for decision in range(job["horizon"]):
        if state["feasible"]:
            stop = "feasible"
            break
        if time.monotonic() - began >= EPISODE_SECONDS - PP_SECONDS:
            censored, stop = True, "resource_budget"
            break
        if decision:
            index, pool = selector.select(env, state, case["decision"] + decision)
        candidate = pool[index]
        action = {"mode": "explicit_neighborhood", "agents": candidate["agents"],
                  "random_seed": seed(case["case_id"], job["trial"], decision, "pp")}
        temp = temperature(initial["iteration"] + decision)
        draw = acceptance_draw(seed(case["case_id"], job["trial"], decision, "accept"))
        before = state
        if arm in ("standard", "legacy"):
            step = _plain(env.step_with_time_limit(action, PP_SECONDS))
        else:
            step = _plain(env.step_experimental_pp(action, PP_SECONDS, arm, temp, draw))
        state, metrics = step["observation"], step["metrics"]
        validate_transition(before, state, metrics, candidate["agents"],
                            "standard" if arm == "legacy" else arm, temp, draw)
        validate_final(state)
        transitions.append({"action": action, "metrics": metrics, "candidate_pool": pool,
                            "selected_id": candidate["candidate_id"], "temperature": temp, "uniform": draw,
                            "before_fingerprint": state_fingerprint(before), "after_fingerprint": state_fingerprint(state)})
        conflicts.append(state["num_of_colliding_pairs"])
        if metrics["pp_failure_reason"] == "time_limit" or step["truncated"]:
            censored, stop = True, "native_censor"
            break
    if state["feasible"]:
        stop = "feasible"
    return {"status": "ok", "job_id": job["job_id"], "plan_sha256": job["plan_sha256"],
            "case_id": case["case_id"], "trial": job["trial"], "arm": arm, "map_id": case["map_id"],
            "horizon": job["horizon"], "conflicts": conflicts, "transitions": transitions,
            "initial_fingerprint": state_fingerprint(initial), "final_state": state,
            "feasible": state["feasible"], "censored": censored, "stop": stop,
            "auc": padded_auc(conflicts, job["horizon"]),
            "generated": state["low_level"]["generated"] - initial["low_level"]["generated"],
            "accepted_increases": sum(b > a for a, b in zip(conflicts, conflicts[1:])),
            "diagnostic_wall_seconds": time.monotonic() - began}


def load_result(path, job):
    r = read_json(path)
    checked(r["status"] == "ok" and all(r[k] == job[k] for k in
        ("job_id", "plan_sha256", "arm", "trial", "horizon")), "result identity/error")
    checked(r["initial_fingerprint"] == state_fingerprint(job["case"]["state"]), "source mismatch")
    checked(len(r["conflicts"]) == len(r["transitions"]) + 1 and
            r["auc"] == padded_auc(r["conflicts"], job["horizon"]) and
            r["feasible"] == (r["conflicts"][-1] == 0), "result summary mismatch")
    fp = r["initial_fingerprint"]
    for event in r["transitions"]:
        checked(event["before_fingerprint"] == fp, "broken fingerprint chain")
        fp = event["after_fingerprint"]
    checked(fp == state_fingerprint(r["final_state"]), "final identity mismatch")
    return r


def verify_smoke(plan):
    schedule = jobs(plan, True)
    rows = {r["job_id"]: r for r in (load_result(OUT / "smoke" / (j["job_id"] + ".json"), j) for j in schedule)}
    for c in plan["cases"]:
        a, b = (rows[f"{c['case_id']}-{arm}-0"] for arm in ("legacy", "standard"))
        checked(not a["censored"] and not b["censored"], "censored parity")
        checked(state_fingerprint(a["final_state"]) == state_fingerprint(b["final_state"]), "legacy parity mismatch")
        checked([(e["candidate_pool"], e["selected_id"]) for e in a["transitions"]] ==
                [(e["candidate_pool"], e["selected_id"]) for e in b["transitions"]], "candidate/scorer parity mismatch")
    return {"status": "passed", "legacy_new_pairs": len(plan["cases"])}


def run(smoke=False):
    plan = verify()
    if not smoke:
        verify_smoke(plan)
    phase = "smoke" if smoke else "outcomes"
    schedule, began = jobs(plan, smoke), time.monotonic()
    directory = OUT / phase
    with _CollectionRunLock(OUT, sha256_file(OUT / "plan.json"), phase):
        try:
            pending = []
            for j in schedule:
                p = directory / (j["job_id"] + ".json")
                if p.exists():
                    load_result(p, j)
                else:
                    pending.append(j)
            def save(r):
                write_json(directory / (r["job_id"] + ".json"), r)
                print(r["job_id"], r["status"], flush=True)
            def failure(j, status, error):
                return {"job_id": j["job_id"], "status": status, "error": error}
            for offset in range(0, len(pending), WORKERS):
                if (OUT / "STOP").exists() or time.monotonic() - began + FUSE_SECONDS >= TOTAL_SECONDS:
                    raise InterruptedError("safe stop or total budget")
                write_json(OUT / "run_status.json", {"status": "running", "phase": phase,
                    "completed": len(schedule) - len(pending) + offset, "total": len(schedule)})
                _run_jobs(worker, pending[offset:offset+WORKERS], WORKERS, phase=phase,
                    output_root=directory, run_fingerprint=sha256_file(OUT / "plan.json"),
                    timeout_seconds=FUSE_SECONDS, on_result=save, failure_result=failure, stop_on_failure=True)
            report = verify_smoke(plan) if smoke else analyze(plan)
            write_json(OUT / "run_status.json", {"status": "complete", "phase": phase,
                "completed": len(schedule), "elapsed_seconds": time.monotonic() - began})
            return report
        except BaseException as e:
            write_json(OUT / "run_status.json", {"status": "paused" if isinstance(e, InterruptedError) else "failed", "error": str(e)})
            raise


def analyze(plan):
    rows = [load_result(OUT / "outcomes" / (j["job_id"] + ".json"), j) for j in jobs(plan)]
    report = summarize(rows)
    report["plan_sha256"] = sha256_file(OUT / "plan.json")
    report["outcome_sha256"] = {p.name: sha256_file(p) for p in sorted((OUT / "outcomes").glob("*.json"))}
    write_json(OUT / "report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "dry-run", "smoke", "collect", "analyze"))
    args = parser.parse_args()
    if args.phase == "prepare": result = prepare()
    elif args.phase == "dry-run": result = describe(verify())
    elif args.phase == "analyze": result = analyze(verify())
    else: result = run(args.phase == "smoke")
    print(json.dumps(result, indent=2))
