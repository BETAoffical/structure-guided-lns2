"""Bounded, gated author-reference ablations on retained task OD, not saved paths."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from scripts import verify_cplns_observation as bridge
from scripts import verify_cplns_stationary_count as reference

CONFIG = ROOT / "configs/cplns_real_task_mechanism_v1.json"
BASE = "author_no_sa_no_restart"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def task_files(case):
    state = case["state"]
    h, w, obstacles = state["rows"], state["cols"], state["obstacles"]
    agents = sorted(state["agents"], key=lambda a: a["id"])
    if h <= 0 or w <= 0 or len(obstacles) != h * w or not agents:
        raise ValueError("invalid task grid")
    if [a["id"] for a in agents] != list(range(len(agents))):
        raise ValueError("author CLI requires contiguous ordered IDs")
    for key in ("start", "goal"):
        cells = [a[key] for a in agents]
        if len(set(cells)) != len(cells) or any(not 0 <= c < h * w or obstacles[c] for c in cells):
            raise ValueError("invalid or duplicate task endpoints")
    grid = f"type octile\nheight {h}\nwidth {w}\nmap\n" + "\n".join(
        "".join("@" if obstacles[y * w + x] else "." for x in range(w)) for y in range(h)) + "\n"
    scen = "version 1\n" + "".join(
        f"0\t{case['case_id']}.map\t{w}\t{h}\t{a['start'] % w}\t{a['start'] // w}\t"
        f"{a['goal'] % w}\t{a['goal'] // w}\t0\n" for a in agents)
    return grid, scen, len(agents)


def schedule(config, cases, ref):
    jobs = []
    for case in cases:
        base = Path(config["output"]) / "tasks" / case["case_id"]
        for seed in config["seeds"]:
            for profile, settings in ref["profiles"].items():
                options = {**ref["common_options"], **settings, "screen": "1", "seed": str(seed),
                           "cutoffTime": str(config["solver_seconds"]),
                           "map": base.with_suffix(".map").as_posix(),
                           "agents": base.with_suffix(".scen").as_posix(), "agentNum": str(case["agent_count"])}
                jobs.append({"job_id": f"{case['case_id']}-{seed}-{profile}", "case_id": case["case_id"],
                             "map_id": case["map_id"], "seed": seed, "profile": profile, "lane": "observer_on",
                             "output": config["output"], "external_seconds": config["external_seconds"],
                             "argv": [(reference.OUT / "build/plns").relative_to(ROOT).as_posix(),
                                      *[v for k, value in options.items() for v in ("--" + k, value)]]})
    return jobs


def prepare():
    config = read_json(CONFIG)
    _, ref = reference.verify()
    source = ROOT / config["source"]
    ref_report = reference.OUT / "report.json"
    if sha256_file(source) != config["source_sha256"] or sha256_file(ref_report) != config["reference_report_sha256"]:
        raise ValueError("source or reference prerequisite changed")
    out = ROOT / config["output"]
    if out.exists():
        raise ValueError("output exists; use verify/resume, never overwrite")
    supplied = {c["case_id"]: c for c in read_json(source)["discovery"]}
    cases, files = [], {}
    for name in config["cases"]:
        case = supplied[name]
        if sha256_file(ROOT / case["trace"]) != case["trace_sha256"]:
            raise ValueError("historical trace changed")
        grid, scen, count = task_files(case)
        for suffix, content in ((".map", grid), (".scen", scen)):
            files[out / "tasks" / (name + suffix)] = content
        cases.append({"case_id": name, "map_id": case["map_id"], "agent_count": count,
                      "saved_state_conflicts": case["state"]["num_of_colliding_pairs"],
                      "saved_state_paths_used": False})
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="ascii", newline="\n")
    tracked = [CONFIG, source, ref_report, reference.OUT / "registry.json", reference.OUT / "build/plns",
               Path(__file__), Path(bridge.__file__), Path(reference.__file__),
               ROOT / "docs/CPLNS_REAL_TASK_MECHANISM_ZH.md",
               ROOT / "experiments/_common.py", ROOT / "experiments/repair_collection.py", *files]
    plan = {"schema": config["schema"], "config": config, "cases": cases,
            "jobs": schedule(config, cases, ref),
            "inputs": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in tracked}}
    write_json(out / "plan.json", plan)
    return {"qualification_jobs": 20, "maximum_jobs": len(plan["jobs"]), "workers": config["workers"],
            "maximum_solver_resource_seconds": len(plan["jobs"]) * config["solver_seconds"],
            "scope": config["scope"], "cases": cases}


def verify():
    config = read_json(CONFIG)
    plan = read_json(ROOT / config["output"] / "plan.json")
    if plan["config"] != config:
        raise ValueError("config changed")
    for path, expected in plan["inputs"].items():
        if sha256_file(ROOT / path) != expected:
            raise ValueError("input changed: " + path)
    _, ref = reference.verify()
    if plan["jobs"] != schedule(config, plan["cases"], ref):
        raise ValueError("job schedule changed")
    return plan


def event_summary(events):
    complete = [e for e in events if e["event"] == "init" or e["event"] == "initial_pp" and e["conflicts"] == 0]
    if not complete:
        raise ValueError("no complete initial state; cannot admit this task")
    first = complete[0]
    initial = {"agents": first["agents"], "conflicts": first["conflicts"], "cost": first["cost"]}
    sequence = [first["conflicts"]]
    best, debt, escapes = first["conflicts"], None, 0
    worse, rejected, identical, restarts = 0, 0, 0, set()
    previous = first["agents"]
    for event in events:
        if event["event"] == "init":
            restarts.add(event["restart"])
            debt = None
            best = min(best, event["conflicts"])
            previous = event["agents"]
            continue
        if event["event"] != "step":
            continue
        sequence.append(event["conflicts"])
        rejected += event["accepted"] == 0
        identical += event["agents"] == previous
        if event["accepted"] and event["new_pairs"] > event["old_pairs"]:
            worse += 1
            if debt is None:
                debt = best
        if debt is not None and event["conflicts"] < debt:
            escapes += 1
            debt = None
        best = min(best, event["conflicts"])
        previous = event["agents"]
    return {"initial_digest": digest(initial), "initial_conflicts": first["conflicts"],
            "conflicts": sequence, "observed_best_conflicts": best,
            "observed_steps": len(sequence) - 1, "accepted_worse": worse, "rejected": rejected,
            "unchanged_paths": identical, "worse_then_new_best_same_restart": escapes,
            "observed_restart_ids": sorted(restarts), "feasible_within_observed_steps": best == 0,
            "prefix_truncated": events[-1]["decisions_seen"] > 128}


def analyze_job(job):
    out = ROOT / job["output"]
    path = out / "jobs" / (job["job_id"] + ".json")
    row = read_json(path)
    bridge.verify_job(row, job)
    text = path.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
    parsed = bridge.parse_output(text, row["returncode"], row["timed_out"])
    if row["status"] != "complete" or not parsed["valid"]:
        raise ValueError("incomplete/invalid author process: " + job["job_id"])
    events = [json.loads(line) for line in path.with_suffix(".events.jsonl").read_text().splitlines()]
    task = out / "tasks" / job["case_id"]
    metrics = bridge.validate_events(events, task.with_suffix(".map").read_text(), task.with_suffix(".scen").read_text())
    if metrics["final_conflicts"] != parsed["conflicts"]:
        raise ValueError("final path/log conflict mismatch")
    return {"status": "ok", **{k: job[k] for k in ("job_id", "case_id", "map_id", "seed", "profile")},
            **event_summary(events), "metrics": metrics, "final_feasible": metrics["final_conflicts"] == 0,
            "upstream_restart_counter": parsed["upstream_restart_counter"],
            "job_record_sha256": sha256_file(path)}


def qualification(rows, config):
    if len(rows) != len(config["cases"]) * len(config["seeds"]) or any(r["profile"] != BASE for r in rows):
        raise ValueError("qualification coverage mismatch")
    if len({(r["case_id"], r["seed"]) for r in rows}) != len(rows):
        raise ValueError("duplicate qualification")
    if {(r["case_id"], r["seed"]) for r in rows} != {(c, s) for c in config["cases"] for s in config["seeds"]}:
        raise ValueError("qualification task/seed mismatch")
    active = [r for r in rows if r["initial_conflicts"] > 0]
    return {"total": len(rows), "repair_episodes": len(active), "repair_maps": len({r["map_id"] for r in active}),
            "passed": len(active) >= config["minimum_repair_episodes"] and
                      len({r["map_id"] for r in active}) >= config["minimum_repair_maps"]}


def checked_rows(plan, jobs):
    rows = _run_jobs(analyze_job, jobs, plan["config"]["workers"], phase="cplns-path-audit", timeout_seconds=180)
    errors = [r for r in rows if r.get("status") != "ok"]
    if errors:
        write_json(ROOT / plan["config"]["output"] / "analysis_errors.json", errors)
        raise ValueError("path/process audit failed; see analysis_errors.json")
    return sorted(rows, key=lambda r: r["job_id"])


def collect(plan, jobs):
    out = ROOT / plan["config"]["output"]
    pending = []
    for job in jobs:
        path = out / "jobs" / (job["job_id"] + ".json")
        if path.exists():
            row = read_json(path)
            bridge.verify_job(row, job)
            if row["status"] != "complete":
                raise ValueError("failed job retained; no automatic retry")
        else:
            if any(path.with_suffix(suffix).exists() for suffix in (".log", ".events.jsonl")):
                raise ValueError("orphaned job output; inspect before resuming")
            pending.append(job)
    def save(row):
        write_json(out / "jobs" / (row["job_id"] + ".json"), row)
        print(row["job_id"], row["status"], flush=True)
    for start in range(0, len(pending), plan["config"]["workers"]):
        if (out / "STOP_AFTER_BATCH").exists():
            raise InterruptedError("safe stop before next batch")
        _run_jobs(bridge.run_job, pending[start:start + plan["config"]["workers"]], plan["config"]["workers"],
                  phase="cplns-real-task", output_root=out / "jobs", run_fingerprint=sha256_file(out / "plan.json"),
                  timeout_seconds=90, on_result=save, stop_on_failure=True)


def report(plan, rows, gate):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case_id"], row["seed"])].append(row)
    mismatches = [list(key) for key, group in groups.items() if len({r["initial_digest"] for r in group}) != 1]
    summaries = {}
    for profile in sorted({r["profile"] for r in rows}):
        group = [r for r in rows if r["profile"] == profile]
        summaries[profile] = {"jobs": len(group), "final_feasible": sum(r["final_feasible"] for r in group),
                              "prefix_feasible": sum(r["feasible_within_observed_steps"] for r in group),
                              "accepted_worse": sum(r["accepted_worse"] for r in group),
                              "worse_then_new_best": sum(r["worse_then_new_best_same_restart"] for r in group),
                              "truncated_prefixes": sum(r["prefix_truncated"] for r in group)}
    result = {"schema": plan["schema"], "scope": plan["config"]["scope"], "qualification": gate,
              "initial_mismatches": mismatches, "profiles": summaries, "rows": rows,
              "plan_sha256": sha256_file(ROOT / plan["config"]["output"] / "plan.json"),
              "decision": "insufficient_repair_pressure" if not gate["passed"] else
                          "initial_pairing_failure" if mismatches else "bounded_mechanism_observed_no_promotion",
              "no_ttf_or_promotion": True, "saved_paths_replayed": False}
    write_json(ROOT / plan["config"]["output"] / "report.json", result)
    return {k: result[k] for k in ("qualification", "profiles", "initial_mismatches", "decision")}


def run():
    plan = verify()
    out = ROOT / plan["config"]["output"]
    with _CollectionRunLock(out, sha256_file(out / "plan.json"), "cplns-real-task"):
        write_json(out / "run_status.json", {"status": "running"})
        try:
            base_jobs = [j for j in plan["jobs"] if j["profile"] == BASE]
            collect(plan, base_jobs)
            rows = checked_rows(plan, base_jobs)
            gate = qualification(rows, plan["config"])
            write_json(out / "qualification.json", {"gate": gate, "rows": rows})
            if gate["passed"]:
                collect(plan, [j for j in plan["jobs"] if j["profile"] != BASE])
                rows = checked_rows(plan, plan["jobs"])
            result = report(plan, rows, gate)
            write_json(out / "run_status.json", {"status": "complete", "decision": result["decision"]})
            return result
        except BaseException as exc:
            write_json(out / "run_status.json", {"status": "interrupted_or_failed", "error": str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "run"))
    args = parser.parse_args()
    result = prepare() if args.phase == "prepare" else run() if args.phase == "run" else {"verified": bool(verify())}
    print(json.dumps(result, indent=2))
