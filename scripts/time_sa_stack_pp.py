"""Serial frozen-prefix PP component timing; never an end-to-end TTF run."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.sa_single_check_runtime import SingleFullCheckPool
from scripts import probe_sa_stack_neighbors as probe

OUT = ROOT / "build/sa-stack-pp-timing-v1"
VARIANTS = ("reference", "stack_neighbors")
PILOT_SHA = "1bd8c0109f8dac067900a76834393700656766b925513b7ebf6e0832ea7f6275"
GATE = dict(minimum_pp_reduction_percent=5.0, minimum_cases_faster=6,
            bootstrap_reduction_lower_minimum=0.0, bootstrap_draws=5000,
            bootstrap_seed=20260913, repeats=2, workers=1, job_timeout_seconds=300)
pilot = probe.pilot


def positions(length):
    if type(length) is not int or length < 1:
        raise ValueError("nonempty trajectory required")
    return sorted({0, (length - 1) // 2, length - 1})


def prerequisite():
    p = probe.verify()
    report_path = probe.OUT / "report.json"
    if sha256_file(report_path) != PILOT_SHA:
        raise ValueError("native equivalence report changed")
    report = read_json(report_path)
    if (report["decision"], report["complete"], report["steps"]) != (
            "eligible_for_bounded_pp_timing", True, 792):
        raise ValueError("native prerequisite not met")
    for name, h in report["files"].items():
        if sha256_file(probe.OUT / "replay" / name) != h:
            raise ValueError("native replay evidence changed")
    b = read_json(probe.OUT / "build.json")
    if b["plan_sha256"] != sha256_file(probe.OUT / "plan.json"):
        raise ValueError("native plan binding changed")
    if sha256_file(ROOT / b["native_path"]) != b["native_sha256"]:
        raise ValueError("stack native changed")
    for name, field in (("SIPP.cpp", "generated_cpp_sha256"), ("SIPP.o", "object_sha256")):
        if sha256_file(probe.OUT / name) != b[field]:
            raise ValueError("generated native evidence changed")
    return p, b


def register():
    if (OUT / "QUARANTINED.json").exists():
        raise ValueError("quarantined batch; use a newly registered output directory")
    if (OUT / "registration.json").exists():
        raise ValueError("registration exists")
    p, b = prerequisite()
    lengths = {}
    for case in p["cases"]:
        record = p["records"][case["case_id"]]
        if sha256_file(ROOT / record["path"]) != record["sha256"]:
            raise ValueError("trace changed")
        lengths[case["case_id"]] = len(read_json(ROOT / record["path"])["events"])
    if len(lengths) != 8 or sum(lengths.values()) != 792:
        raise ValueError("unexpected case coverage")
    names = ["scripts/time_sa_stack_pp.py", "tests/evaluation/test_sa_stack_pp_timing.py",
             "docs/SA_STACK_PP_TIMING_PROTOCOL_ZH.md"]
    r = dict(schema="lns2.sa_stack_pp_timing.v1", cases=p["cases"], records=p["records"],
             config=p["config"], native=b, lengths=lengths, gate=GATE,
             input_sha256={n: sha256_file(ROOT / n) for n in names},
             prior_report_sha256=PILOT_SHA, no_ttf=True, default_changed=False,
             source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    write_json(OUT / "registration.json", r)
    return dict(jobs=32, workers=1, paired_pp_calls=1584, total_repair_calls=3168,
                diagnostic_sample_pairs=48, no_ttf=True)


def verify():
    if (OUT / "QUARANTINED.json").exists():
        raise ValueError("quarantined batch cannot be resumed or analyzed")
    p, b = prerequisite()
    r = read_json(OUT / "registration.json")
    for name, h in r["input_sha256"].items():
        if sha256_file(ROOT / name) != h:
            raise ValueError("registered source changed: " + name)
    if (r["gate"], r["cases"], r["records"], r["config"], r["native"]) != (
            GATE, p["cases"], p["records"], p["config"], b):
        raise ValueError("registered protocol changed")
    if not r["no_ttf"] or r["default_changed"] or r["prior_report_sha256"] != PILOT_SHA:
        raise ValueError("scope changed")
    lengths = {}
    for k, v in r["records"].items():
        if sha256_file(ROOT / v["path"]) != v["sha256"]:
            raise ValueError("registered trace changed")
        lengths[k] = len(read_json(ROOT / v["path"])["events"])
    if lengths != r["lengths"]:
        raise ValueError("trace length changed")
    return r


def schedule(r):
    jobs = []
    for repeat in range(GATE["repeats"]):
        for index, case in enumerate(r["cases"]):
            variants = VARIANTS if (repeat + index) % 2 == 0 else VARIANTS[::-1]
            for variant in variants:
                jobs.append(dict(job_id=f"r{repeat}-{case['case_id']}-{variant}",
                    case=case, config=r["config"], record=r["records"][case["case_id"]],
                    native=r["native"], repeat=repeat, variant=variant,
                    length=r["lengths"][case["case_id"]]))
    return jobs


def seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("invalid component timer")
    return float(value)


def repair_equal(state, expected, metrics, expected_metrics):
    if pilot.state_fingerprint(state) != pilot.state_fingerprint(expected):
        raise ValueError("path/conflict/search-counter mismatch")
    for key in ("repair_order", "neighborhood", "replan_success", "pp_failure_reason",
                "pp_attempted_agent_count", "pp_inserted_agent_count", "requested_collect_pp_diagnostics"):
        if metrics[key] != expected_metrics[key]:
            raise ValueError("PP semantic mismatch: " + key)
    if metrics["requested_collect_pp_diagnostics"]:
        raise ValueError("detailed PP diagnostics unexpectedly active")


def worker(job):
    config = deepcopy(job["config"])
    if job["variant"] == VARIANTS[1]:
        config["frozen"].update(native_path=job["native"]["native_path"],
                                native_sha256=job["native"]["native_sha256"])
    if sha256_file(ROOT / job["record"]["path"]) != job["record"]["sha256"]:
        raise ValueError("trace changed")
    saved = read_json(ROOT / job["record"]["path"])
    env = pilot.make_env(dict(job, config=config, budget=3000.))
    state = pilot._plain(env.reset(seed=job["case"]["solver_seed"]))
    expected = saved["initial_state"]
    if pilot.state_fingerprint(state) != pilot.state_fingerprint(expected):
        raise ValueError("initial fingerprint mismatch")
    selector = SingleFullCheckPool(job["case"])
    sampled = positions(len(saved["events"]))
    rows = []
    for d, event in enumerate(saved["events"]):
        if selector.select(env, state, d) != (event["selected_index"], event["pool"]):
            raise ValueError(f"candidate/score/action mismatch at {d}")
        started = time.perf_counter()
        raw = env.step_experimental_pp(event["action"], 300., "annealed", event["temperature"], event["uniform"])
        outer_seconds = time.perf_counter() - started
        step = pilot._plain(raw)
        state = step["observation"]
        expected = pilot.apply_state_delta(expected, event["delta"])
        repair_equal(state, expected, step["metrics"], event["metrics"])
        pp = seconds(step["metrics"]["native_replan_seconds"])
        if pp > outer_seconds + 1e-6:
            raise ValueError("nested PP timer exceeds outer call")
        rows.append(dict(decision=d, sampled=d in sampled, pp_seconds=pp,
                         outer_step_seconds=outer_seconds, fingerprint=pilot.state_fingerprint(state)))
    pilot.validate_final(state)
    return dict(status="ok", job_id=job["job_id"], registration_sha256=job["registration_sha256"],
        case_id=job["case"]["case_id"], map_id=job["case"]["map_id"], repeat=job["repeat"],
        variant=job["variant"], native_sha256=config["frozen"]["native_sha256"],
        full_prefix_equal=True, rows=rows, no_ttf=True)


def validate_job(job, row):
    expected_native = (job["config"]["frozen"]["native_sha256"] if job["variant"] == VARIANTS[0]
                       else job["native"]["native_sha256"])
    expected_identity = (job["job_id"], job["registration_sha256"], job["case"]["case_id"],
                         job["case"]["map_id"], job["repeat"], job["variant"], expected_native)
    actual = tuple(row.get(k) for k in ("job_id", "registration_sha256", "case_id", "map_id",
                                      "repeat", "variant", "native_sha256"))
    if row.get("status") != "ok" or actual != expected_identity or not row.get("full_prefix_equal") or not row.get("no_ttf"):
        raise ValueError("incomplete or changed job identity")
    if [e["decision"] for e in row["rows"]] != list(range(job["length"])):
        raise ValueError("missing/duplicate PP call")
    if [e["decision"] for e in row["rows"] if e["sampled"]] != positions(job["length"]):
        raise ValueError("sample selection changed")
    for e in row["rows"]:
        pp, outer = seconds(e["pp_seconds"]), seconds(e["outer_step_seconds"])
        if pp > outer + 1e-6 or not isinstance(e["fingerprint"], str) or len(e["fingerprint"]) != 64:
            raise ValueError("invalid PP measurement")


def collect(resume=False):
    r = verify()
    identity = sha256_file(OUT / "registration.json")
    with pilot._CollectionRunLock(OUT, identity, "stack-pp-timing"):
        path = OUT / "manifest.json"
        if path.exists() and not resume:
            raise ValueError("run already exists; use resume after inspection")
        manifest = read_json(path) if path.exists() else dict(registration_sha256=identity, files={})
        if manifest["registration_sha256"] != identity:
            raise ValueError("run identity changed")
        if not set(manifest["files"]).issubset({j["job_id"] for j in schedule(r)}):
            raise ValueError("unexpected completed job")
        write_json(path, manifest)
        try:
            for job in schedule(r):
                job["registration_sha256"] = identity
                result_path = OUT / "jobs" / (job["job_id"] + ".json")
                marker = OUT / "started" / (job["job_id"] + ".json")
                if job["job_id"] in manifest["files"]:
                    if sha256_file(result_path) != manifest["files"][job["job_id"]]:
                        raise ValueError("saved job changed")
                    validate_job(job, read_json(result_path))
                    continue
                if (OUT / "STOP_AFTER_JOB.json").exists():
                    write_json(OUT / "status.json", dict(status="stopped", completed=len(manifest["files"]), total=32))
                    return dict(stopped=True, completed=len(manifest["files"]))
                if marker.exists() or result_path.exists():
                    raise ValueError("incomplete prior job; inspect, do not silently retry: " + job["job_id"])
                write_json(marker, dict(registration_sha256=identity, job_id=job["job_id"]))
                write_json(OUT / "status.json", dict(status="running", current=job["job_id"],
                                                    completed=len(manifest["files"]), total=32))
                print("START " + job["job_id"], flush=True)
                def failed(j, status, error):
                    return dict(status=status, job_id=j["job_id"], error=str(error))
                rows = pilot._run_jobs(worker, [job], workers=1, phase="pp-component",
                    output_root=OUT / "progress" / job["job_id"], run_fingerprint=identity,
                    timeout_seconds=GATE["job_timeout_seconds"], failure_result=failed,
                    on_result=lambda row: write_json(result_path, row), stop_on_failure=True)
                if len(rows) != 1:
                    raise ValueError("worker did not return exactly one result")
                validate_job(job, rows[0])
                manifest["files"][job["job_id"]] = sha256_file(result_path)
                write_json(path, manifest)
                print(f"COMPLETE {len(manifest['files'])}/32", flush=True)
            verify()
            write_json(OUT / "status.json", dict(status="complete", completed=32, total=32))
            return dict(complete=True, jobs=32, no_ttf=True)
        except BaseException:
            write_json(OUT / "status.json", dict(status="failed_or_interrupted", completed=len(manifest["files"]), total=32))
            raise


def reduction(a, b):
    if a <= 0:
        raise ValueError("zero reference timing")
    return 100.0 * (1.0 - b / a)


def gate_decision(percent, cases_faster, ci_lower):
    if (percent >= GATE["minimum_pp_reduction_percent"] and cases_faster >= GATE["minimum_cases_faster"]
            and ci_lower >= GATE["bootstrap_reduction_lower_minimum"]):
        return "eligible_for_separate_ttf_confirmation"
    return "pp_gain_unconfirmed_do_not_expand_this_timing"


def summarize(rows):
    paired = {}
    for row in rows:
        key = (row["case_id"], row["repeat"])
        if row["variant"] in paired.setdefault(key, {}):
            raise ValueError("duplicate runtime job")
        paired[key][row["variant"]] = row
    cases = {}
    sample = {v: 0.0 for v in VARIANTS}
    for (case, _), values in paired.items():
        if set(values) != set(VARIANTS):
            raise ValueError("unpaired runtime job")
        a, b = (values[v] for v in VARIANTS)
        if a["map_id"] != b["map_id"] or [(e["decision"], e["fingerprint"]) for e in a["rows"]] != [
                (e["decision"], e["fingerprint"]) for e in b["rows"]]:
            raise ValueError("paired trajectories differ")
        item = cases.setdefault(case, dict(case_id=case, map_id=a["map_id"], calls=0,
            reference=0., stack_neighbors=0., outer_reference=0., outer_stack_neighbors=0.))
        item["calls"] += len(a["rows"])
        for variant in VARIANTS:
            for e in values[variant]["rows"]:
                item[variant] += seconds(e["pp_seconds"])
                item["outer_" + variant] += seconds(e["outer_step_seconds"])
                if e["sampled"]:
                    sample[variant] += e["pp_seconds"]
    maps = {}
    for case in cases.values():
        case["pp_reduction_percent"] = reduction(case[VARIANTS[0]], case[VARIANTS[1]])
        totals = maps.setdefault(case["map_id"], {v: 0. for v in VARIANTS})
        for v in VARIANTS:
            totals[v] += case[v]
    if len(maps) != 4 or len(cases) != 8 or len(paired) != 16:
        raise ValueError("incomplete experimental coverage")
    totals = {v: sum(c[v] for c in cases.values()) for v in VARIANTS}
    rng = random.Random(GATE["bootstrap_seed"])
    draws = []
    keys = sorted(maps)
    for _ in range(GATE["bootstrap_draws"]):
        chosen = [maps[rng.choice(keys)] for _ in keys]
        draws.append(reduction(sum(c[VARIANTS[0]] for c in chosen), sum(c[VARIANTS[1]] for c in chosen)))
    draws.sort()
    ci = [draws[int(.025 * (len(draws) - 1))], draws[int(.975 * (len(draws) - 1))]]
    percent = reduction(totals[VARIANTS[0]], totals[VARIANTS[1]])
    faster = sum(c[VARIANTS[1]] < c[VARIANTS[0]] for c in cases.values())
    return dict(cases=list(cases.values()), maps=maps, pp_seconds=totals, paired_calls=sum(c["calls"] for c in cases.values()),
        pp_reduction_percent=percent, cases_faster=faster, paired_map_bootstrap_reduction_ci95=ci,
        sampled_pp_seconds=sample, sampled_pp_reduction_percent=reduction(sample[VARIANTS[0]], sample[VARIANTS[1]]),
        decision=gate_decision(percent, faster, ci[0]), no_ttf=True, default_changed=False)


def analyze():
    r = verify()
    identity = sha256_file(OUT / "registration.json")
    manifest = read_json(OUT / "manifest.json")
    jobs = schedule(r)
    if manifest["registration_sha256"] != identity or set(manifest["files"]) != {j["job_id"] for j in jobs}:
        raise ValueError("incomplete or changed manifest")
    rows = []
    for job in jobs:
        job["registration_sha256"] = identity
        path = OUT / "jobs" / (job["job_id"] + ".json")
        if sha256_file(path) != manifest["files"][job["job_id"]]:
            raise ValueError("timed result changed")
        row = read_json(path)
        validate_job(job, row)
        rows.append(row)
    report = summarize(rows)
    if report["paired_calls"] != 1584:
        raise ValueError("unexpected repair call coverage")
    report.update(schema="lns2.sa_stack_pp_timing_report.v1", registration_sha256=identity,
                  manifest_sha256=sha256_file(OUT / "manifest.json"), gate=GATE)
    write_json(OUT / "report.json", report)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "verify", "collect", "resume", "stop", "analyze"))
    p.add_argument("--output", default=OUT.relative_to(ROOT).as_posix())
    args = p.parse_args()
    requested = (ROOT / args.output).resolve()
    if requested.parent != (ROOT / "build").resolve() or not requested.name.startswith("sa-stack-pp-timing-"):
        p.error("output must be a sa-stack-pp-timing-* directory directly under build")
    OUT = requested
    phase = args.phase
    if phase == "stop":
        write_json(OUT / "STOP_AFTER_JOB.json", dict(requested=True))
        result = dict(stop_after_job=True)
    elif phase == "resume":
        result = collect(True)
    elif phase == "verify":
        result = dict(verified=bool(verify()))
    else:
        result = globals()[phase]()
    print(json.dumps(result, indent=2))
