"""Allocation-free observation audit; all earlier sources, binaries and reports stay frozen."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from scripts import verify_cplns_observation as old
from scripts.verify_cplns_reference import verify as reference, check_archive

CONFIG = ROOT / "configs/cplns_noalloc_audit_v1.json"
HEADER = ROOT / "experiments/cplns_observer_noalloc.h"
HARNESS = ROOT / "tests/evaluation/cplns_noalloc_harness.cpp"
COUNT_TEMPLATE = ROOT / "tests/evaluation/cplns_count_harness.cpp.in"
COUNT_OUT = ROOT / "build/cplns-count-function-audit-v1"


def expected_files(config, ref):
    out = ROOT / config["output"]
    source = ROOT / ref["source"]
    files = {}
    for rel in check_archive(ref):
        p = source / rel
        files[out / "source" / rel] = (old.instrument(rel, p.read_text(encoding="utf-8")).encode()
                                      if rel in ("src/LNS.cpp", "src/InitLNS.cpp") else p.read_bytes())
    files[out / "source/src/cplns_observer.h"] = HEADER.read_bytes()
    for name in config["fixtures"]:
        grid, scen, _ = old.fixture(name, config["fixture_seed"])
        for suffix, content in ((".map", grid), (".scen", scen)):
            files[out / "fixtures" / (name + suffix)] = content.encode("ascii")
    return files


def prepare():
    config, ref = read_json(CONFIG), reference()
    out = ROOT / config["output"]
    if out.exists():
        raise ValueError("output already exists; do not overwrite")
    planned = expected_files(config, ref)
    for p, contents in planned.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(contents)
    return {"prepared": True, "jobs": len(schedule(config, ref))}


def schedule(config, ref):
    jobs = []
    for name in config["fixtures"]:
        for profile, settings in ref["profiles"].items():
            for seed in config["seeds"]:
                for lane in ("upstream", "observer_off", "observer_on"):
                    base = Path(config["output"]) / "fixtures" / name
                    options = {**ref["common_options"], **settings, "screen": str(config["screen"]),
                               "seed": str(seed), "cutoffTime": str(config["solver_seconds"]),
                               "map": base.with_suffix(".map").as_posix(), "agents": base.with_suffix(".scen").as_posix(),
                               "agentNum": str(old.fixture(name, config["fixture_seed"])[2])}
                    binary = ref["binary"] if lane == "upstream" else config["output"] + "/build/plns"
                    jobs.append({"job_id": f"{name}-{profile}-{seed}-{lane}", "fixture": name, "profile": profile,
                                 "seed": seed, "lane": lane, "output": config["output"],
                                 "external_seconds": config["external_seconds"],
                                 "argv": [binary, *[v for k, val in options.items() for v in ("--" + k, val)]]})
    return jobs


def bound_inputs(config, ref):
    out = ROOT / config["output"]
    paths = [CONFIG, HEADER, HARNESS, Path(__file__), Path(old.__file__),
             ROOT / "scripts/verify_cplns_reference.py", ROOT / ref["output"] / "registry.json",
             ROOT / ref["binary"], out / "build/plns", out / "noalloc_harness"]
    paths += [p for directory in (out / "source", out / "fixtures") for p in directory.rglob("*") if p.is_file()]
    if any(p.is_symlink() for p in paths):
        raise ValueError("symlink input")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def register():
    config, ref = read_json(CONFIG), reference()
    out = ROOT / config["output"]
    if (out / "registry.json").exists():
        raise ValueError("registry exists")
    for p, contents in expected_files(config, ref).items():
        if p.read_bytes() != contents:
            raise ValueError("source/fixture differs from exact patch: " + str(p))
    expected_source = {p for p in expected_files(config, ref) if out / "source" in p.parents}
    if expected_source != {p for p in (out / "source").rglob("*") if p.is_file()}:
        raise ValueError("unexpected source file set")
    write_json(out / "registry.json", {"schema": "lns2.cplns_noalloc_registry.v1", "inputs": bound_inputs(config, ref)})
    return {"registered": True, "jobs": len(schedule(config, ref))}


def verify():
    config, ref = read_json(CONFIG), reference()
    if read_json(ROOT / config["output"] / "registry.json")["inputs"] != bound_inputs(config, ref):
        raise ValueError("registered noalloc input changed")
    return config, ref


def harness():
    config, _ = verify()
    out = ROOT / config["output"]
    result_file = out / "harness_result.json"
    if result_file.exists():
        raise ValueError("harness already run; preserve result")
    path = out / "harness.events.jsonl"
    env = dict(os.environ)
    env["CPLNS_OBSERVER_PATH"] = str(path)
    on = subprocess.run([str(out / "noalloc_harness")], env=env, timeout=15)
    if on.returncode:
        raise ValueError("allocation trap or observer I/O failed")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    if len(events) != 258 or events[-1]["decisions_seen"] != 129 or len(events[0]["agents"][0]["path"]) != 12000:
        raise ValueError("buffer/cap/final contract failed")
    env.pop("CPLNS_OBSERVER_PATH")
    off = subprocess.run([str(out / "noalloc_harness")], env=env, timeout=15)
    existing = subprocess.run([str(out / "noalloc_harness")], env={**env, "CPLNS_OBSERVER_PATH": str(path)}, timeout=15)
    failed_io = subprocess.run([str(out / "noalloc_harness")], env={**env, "CPLNS_OBSERVER_PATH": str(out / "absent/trace")}, timeout=15)
    result = {"on_exit": on.returncode, "off_exit": off.returncode, "existing_exit": existing.returncode,
              "failed_io_exit": failed_io.returncode, "events": len(events),
              "event_sha256": sha256_file(path), "cpp_new_trap": True, "no_ttf": True}
    result["passed"] = off.returncode == 0 and existing.returncode == failed_io.returncode == 86
    write_json(result_file, result)
    if not result["passed"]:
        raise ValueError("harness error handling failed")
    return result


def initial_paths(text, count):
    found = {}
    for line in text.splitlines():
        match = re.fullmatch(r"Agent (\d+):\s*(.*)", line)
        if match:
            aid = int(match[1])
            if aid in found:
                raise ValueError("duplicate agent before complete initialization")
            found[aid] = [int(v) for v in match[2].split()]
            if len(found) == count:
                return found
    return None


def compare_logs(left, right, count, max_records):
    a, b = old.scientific_log(left), old.scientific_log(right)
    length = max(0, min(len(a), len(b), max_records) - 8)
    index = next((i for i in range(length) if a[i] != b[i]), None)
    ap, bp = initial_paths(left, count), initial_paths(right, count)
    return {"prefix_equal": length > 0 and index is None, "first_difference": index,
            "compared_records": length, "initial_paths_present": ap is not None and bp is not None,
            "initial_paths_equal": ap is not None and ap == bp,
            "first_left": a[index] if index is not None else None,
            "first_right": b[index] if index is not None else None}


def collect():
    config, ref = verify()
    out = ROOT / config["output"]
    harness_result = read_json(out / "harness_result.json")
    if not harness_result["passed"] or harness_result["event_sha256"] != sha256_file(out / "harness.events.jsonl"):
        raise ValueError("no valid allocation harness result")
    fingerprint = sha256_file(out / "registry.json")
    with _CollectionRunLock(out, fingerprint, "noalloc"):
        pending = []
        for job in schedule(config, ref):
            p = out / "jobs" / (job["job_id"] + ".json")
            if p.exists():
                previous = read_json(p)
                old.verify_job(previous, job)
                if previous["status"] != "complete":
                    raise ValueError("previous process failure requires inspection")
            else:
                pending.append(job)
        def save(row):
            write_json(out / "jobs" / (row["job_id"] + ".json"), row)
            print(row["job_id"], row["status"], flush=True)
        write_json(out / "run_status.json", {"status": "running", "pending": len(pending)})
        try:
            _run_jobs(old.run_job, pending, config["workers"], phase="noalloc", output_root=out / "jobs",
                      run_fingerprint=fingerprint, timeout_seconds=35, on_result=save,
                      failure_result=lambda j, s, e: {"job_id": j["job_id"], "status": s, "error": e}, stop_on_failure=True)
            result = report()
            write_json(out / "run_status.json", {"status": "complete", "decision": result["decision"]})
            return result
        except BaseException as exc:
            write_json(out / "run_status.json", {"status": "interrupted_or_failed", "error": str(exc)})
            raise


def report():
    config, ref = verify()
    out = ROOT / config["output"]
    groups, diagnostics, hashes = {}, [], {}
    for job in schedule(config, ref):
        p = out / "jobs" / (job["job_id"] + ".json")
        row = read_json(p)
        old.verify_job(row, job)
        hashes[p.relative_to(ROOT).as_posix()] = sha256_file(p)
        text = p.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
        parsed = old.parse_output(text, row["returncode"], row["timed_out"])
        item = {"job_id": job["job_id"], "process_valid": parsed["valid"]}
        if job["lane"] == "observer_on":
            events = [json.loads(line) for line in p.with_suffix(".events.jsonl").read_text().splitlines()]
            grid, scen, count = old.fixture(job["fixture"], config["fixture_seed"])
            try:
                metrics = old.validate_events(events, grid, scen)
                if metrics["final_conflicts"] != parsed.get("conflicts"):
                    raise ValueError("final path/log mismatch")
                exported_init = next((e for e in events if e["event"] == "init"), None)
                if exported_init is not None and initial_paths(text, count) != {a["id"]: a["path"] for a in exported_init["agents"]}:
                    raise ValueError("exported init and raw path log disagree")
                item.update(paths_valid=True, metrics=metrics)
            except ValueError as exc:
                item.update(paths_valid=False, error=str(exc))
        diagnostics.append(item)
        groups.setdefault((job["fixture"], job["profile"], job["seed"]), {})[job["lane"]] = (text, parsed)
    comparisons = []
    for (name, profile, seed), lanes in groups.items():
        count = old.fixture(name, config["fixture_seed"])[2]
        for left, right in (("upstream", "observer_off"), ("upstream", "observer_on"), ("observer_off", "observer_on")):
            a, ar = lanes[left]
            b, br = lanes[right]
            comp = compare_logs(a, b, count, config["prefix_records"])
            pp_direct = name == "open"
            if ar.get("conflicts") == br.get("conflicts") == 0:
                comp["prefix_equal"] = old.scientific_log(a) == old.scientific_log(b) and ar["soc"] == br["soc"]
            comp["passed"] = comp["prefix_equal"] and (pp_direct or comp["initial_paths_equal"])
            comparisons.append({"fixture": name, "profile": profile, "seed": seed,
                                "left_lane": left, "right_lane": right, "pp_direct": pp_direct, **comp})
    parity = all(c["passed"] for c in comparisons)
    integrity = all(d["process_valid"] and d.get("paths_valid", True) for d in diagnostics)
    result = {"schema": "lns2.cplns_noalloc_report.v1", "comparisons": comparisons, "diagnostics": diagnostics,
              "parity_passed": parity, "integrity_passed": integrity, "job_record_hashes": hashes,
              "registry_sha256": sha256_file(out / "registry.json"), "no_ttf_or_promotion": True,
              "decision": "bounded_noalloc_pass" if parity and integrity else "blocked_before_real_cases"}
    write_json(out / "report.json", result)
    return {"jobs": len(diagnostics), "comparisons_passed": sum(c["passed"] for c in comparisons),
            "comparisons": len(comparisons), "path_failures": sum(d.get("paths_valid") is False for d in diagnostics),
            "decision": result["decision"]}


def target_pairs(paths, target):
    path = paths[target]
    if not path:
        return []
    pairs = []
    for aid, other in paths.items():
        if aid == target or not other:
            continue
        for t in range(max(len(path), len(other))):
            p, q = path[min(t, len(path) - 1)], other[min(t, len(other) - 1)]
            vertex = p == q
            swap = t > 0 and p == other[min(t - 1, len(other) - 1)] and q == path[min(t - 1, len(path) - 1)]
            if vertex or swap:
                pairs.append(sorted((target, aid)))
                break
    return sorted(pairs)


def extract_counters(text):
    start = "bool InitLNS::updateCollidingPairs("
    end = "void InitLNS::chooseDestroyHeuristicbyALNS()"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError("counter extraction anchor changed")
    function = text[text.index(start):text.index(end)]
    original = function.replace("InitLNS::updateCollidingPairs", "CounterFixture::original", 1)
    guard = "if (path.size() < 2) return succ;"
    if function.count(guard) != 1:
        raise ValueError("stationary-path guard changed")
    corrected = function.replace("InitLNS::updateCollidingPairs", "CounterFixture::corrected", 1).replace(guard, "if (path.empty()) return succ;", 1)
    return original + "\n" + corrected


def counter_prepare():
    ref = reference()
    if COUNT_OUT.exists():
        raise ValueError("counter audit already exists")
    audit = ROOT / "build/cplns-observer-failure-audit-v1/report.json"
    if sha256_file(audit) != "02dd4cf8f3e8e01f981e332b9d7081d0edd0783202bf5de871143fcc0b05691d":
        raise ValueError("historical failure audit changed")
    failure = read_json(audit)
    prefix = "build/cplns-observer-divergence-v1/jobs/two_door-Collision-observer_on"
    record_path = ROOT / (prefix + ".json")
    if sha256_file(record_path) != failure["job_record_bindings"][record_path.relative_to(ROOT).as_posix()]:
        raise ValueError("historical witness record changed")
    events_path = ROOT / (prefix + ".events.jsonl")
    if sha256_file(events_path) != read_json(record_path)["events_sha256"]:
        raise ValueError("historical witness paths changed")
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    witnesses = [e for e in events if e["event"] == "step" and e["restart"] == 0 and e["iteration"] == 60]
    if len(witnesses) != 1:
        raise ValueError("missing/ambiguous count witness")
    paths = {a["id"]: a["path"] for a in witnesses[0]["agents"]}
    synthetic = [
        ("stationary_blocked", {7: [1], 14: [0, 1, 2, 3]}, 7, 4),
        ("stationary_clear", {7: [1], 14: [2, 3]}, 7, 4),
        ("swap", {0: [0, 1], 1: [1, 0]}, 0, 2),
        ("vertex", {0: [0, 1, 2], 1: [2, 1, 0]}, 0, 3),
        ("goal_wait", {0: [0, 1], 1: [3, 2, 1, 2]}, 0, 4),
        ("repeated_pair_once", {0: [1], 1: [0, 1, 2, 1, 2]}, 0, 3),
        ("external_stationary", {0: [0, 1, 2], 1: [1]}, 0, 3),
        ("empty_robustness", {0: [], 1: [1]}, 0, 2),
    ]
    cases = [{"name": name, "paths": p, "target": target, "cells": cells, "expected": target_pairs(p, target)}
             for name, p, target, cells in synthetic]
    cases += [{"name": f"saved_state_agent_{aid}", "paths": paths, "target": aid, "cells": 64,
               "expected": target_pairs(paths, aid)} for aid in sorted(paths)]
    source_file = ROOT / ref["source"] / "src/InitLNS.cpp"
    functions = extract_counters(source_file.read_text(encoding="utf-8"))
    template = COUNT_TEMPLATE.read_text(encoding="utf-8")
    if template.count("/* EXTRACTED_FUNCTIONS */") != 1:
        raise ValueError("counter template changed")
    COUNT_OUT.mkdir(parents=True)
    (COUNT_OUT / "counter.cpp").write_text(template.replace("/* EXTRACTED_FUNCTIONS */", functions), encoding="utf-8", newline="\n")
    lines = [str(len(cases))]
    for case in cases:
        lines.append(f"{case['cells']} {len(case['paths'])} {case['target']}")
        for aid, path in sorted(case["paths"].items()):
            lines.append(" ".join(map(str, [aid, len(path), *path])))
    (COUNT_OUT / "cases.txt").write_text("\n".join(lines) + "\n", encoding="ascii")
    write_json(COUNT_OUT / "cases.json", cases)
    bound = [Path(__file__), COUNT_TEMPLATE, source_file, audit, record_path, events_path,
             COUNT_OUT / "counter.cpp", COUNT_OUT / "cases.txt", COUNT_OUT / "cases.json"]
    write_json(COUNT_OUT / "plan.json", {"schema": "lns2.cplns_count_function.v1",
               "inputs": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in bound},
               "cases": len(cases), "only_change": "path.size()<2 becomes path.empty()",
               "scope": "extracted_function_with_table_fixture_not_end_to_end_PP", "no_ttf_or_promotion": True})
    return {"cases": len(cases), "prepared": True}


def counter_run():
    plan = read_json(COUNT_OUT / "plan.json")
    for p, digest in plan["inputs"].items():
        if sha256_file(ROOT / p) != digest:
            raise ValueError("counter input changed: " + p)
    if (COUNT_OUT / "report.json").exists():
        raise ValueError("counter results already exist")
    binary = COUNT_OUT / "counter"
    with (COUNT_OUT / "cases.txt").open() as stream:
        run = subprocess.run([str(binary)], stdin=stream, capture_output=True, text=True, timeout=20)
    if run.returncode:
        raise ValueError("counter harness failed: " + run.stderr)
    rows = [json.loads(line) for line in run.stdout.splitlines()]
    cases = read_json(COUNT_OUT / "cases.json")
    if len(rows) != len(cases) or [r["case"] for r in rows] != list(range(len(cases))):
        raise ValueError("incomplete counter results")
    for row, case in zip(rows, cases):
        row.update(name=case["name"], expected=case["expected"],
                   original_matches=row["original"] == case["expected"], corrected_matches=row["corrected"] == case["expected"])
    result = {"schema": "lns2.cplns_count_function_report.v1", "rows": rows,
              "corrected_passed": all(r["corrected_matches"] for r in rows),
              "original_mismatches": sum(not r["original_matches"] for r in rows),
              "binary_sha256": sha256_file(binary), "plan_sha256": sha256_file(COUNT_OUT / "plan.json"),
              "no_ttf_or_promotion": True, "native_PP_modified": False}
    write_json(COUNT_OUT / "report.json", result)
    return {k: v for k, v in result.items() if k != "rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "register", "harness", "collect", "report", "verify", "counter-prepare", "counter-run"))
    args = parser.parse_args()
    value = {"verified": bool(verify())} if args.phase == "verify" else {
        "prepare": prepare, "register": register, "harness": harness, "collect": collect, "report": report,
        "counter-prepare": counter_prepare, "counter-run": counter_run}[args.phase]()
    print(json.dumps(value, indent=2))
