"""Frozen-runtime equivalence and small serial paired TTF engineering check."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.sa_single_check_runtime import SingleFullCheckPool, isolated_worker
from scripts import run_sa_supervised_confirmation as supervisor

source, pilot = supervisor.source, supervisor.pilot
OUT = ROOT / "build/sa-single-check-runtime-v1"
VARIANTS = ("reference_double_check", "single_full_check")


def cohort(plan):
    maps = sorted({c["map_id"] for c in plan["cases"]})
    chosen = {maps[i] for i in (1, 3, 5, 7)}
    cases = [c for c in plan["cases"] if c["map_id"] in chosen and c["solver_seed"] == 103]
    if len(cases) != 8:
        raise ValueError("fixed engineering cohort changed")
    return cases


def register():
    if (OUT / "registration.json").exists():
        raise ValueError("registration already exists; use verify")
    plan = source.verify()
    cases = cohort(plan)
    manifest_path = supervisor.OUT / "timed_report.json"
    manifest = read_json(manifest_path)
    records = {}
    for name, sha in manifest["files"].items():
        if any(Path(name).name == c["case_id"] + "-dual16_sa.json" for c in cases):
            if sha256_file(ROOT / name) != sha:
                raise ValueError("historical trace changed")
            row = read_json(ROOT / name)
            records[row["case_id"]] = dict(path=name, sha256=sha)
    if set(records) != {c["case_id"] for c in cases}:
        raise ValueError("historical traces missing")
    names = ["experiments/sa_single_check_runtime.py", "scripts/confirm_sa_single_check_runtime.py",
             "tests/evaluation/test_sa_single_check_runtime.py", "docs/SA_SINGLE_CHECK_RUNTIME_PROTOCOL_ZH.md",
             manifest_path.relative_to(ROOT).as_posix(), source.OUT.relative_to(ROOT).as_posix() + "/plan.json"]
    payload = dict(schema="lns2.sa_single_check.v1", cases=cases, config=plan["config"],
        variants=list(VARIANTS), rounds=2, budget=60., solver_fuse=180, audit_fuse=1800,
        workers=1, diagnostic_workers=8, max_repair_iterations=0, records=records,
        inputs={n: sha256_file(ROOT / n) for n in names},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        development_only=True, default_changed=False, timing_jobs=32)
    write_json(OUT / "registration.json", payload)
    return dict(registered=True, cases=8, timing_jobs=32, diagnostic_jobs=8)


def verify():
    source.verify()
    r = read_json(OUT / "registration.json")
    for n, sha in r["inputs"].items():
        if sha256_file(ROOT / n) != sha:
            raise ValueError("registered input changed: " + n)
    if (r["variants"], r["rounds"], r["budget"], r["timing_jobs"], r["workers"]) != (list(VARIANTS), 2, 60., 32, 1):
        raise ValueError("protocol changed")
    if r["cases"] != cohort(read_json(source.OUT / "plan.json")):
        raise ValueError("cohort changed")
    return r


def schedule(r):
    identity = sha256_file(OUT / "registration.json")
    result = []
    for repeat in range(r["rounds"]):
        for i, case in enumerate(r["cases"]):
            order = VARIANTS if (i + repeat) % 2 == 0 else VARIANTS[::-1]
            for variant in order:
                jid = f"{case['case_id']}-r{repeat}-{variant}"
                result.append(dict(job_id=jid, case=case, config=r["config"], arm="dual16_sa",
                    phase="timed", max_steps=None, budget=r["budget"], runtime_variant=variant,
                    repeat=repeat, plan_sha256=identity, registration_sha256=identity,
                    capture_path=(OUT / "raw" / (jid + ".json")).relative_to(ROOT).as_posix()))
    return result


def solver_worker(job):
    path = ROOT / job["capture_path"]
    if path.exists():
        raise ValueError("raw exists; audit instead of retrying")
    selectors = dict(zip(VARIANTS, (pilot.TimedPool, SingleFullCheckPool)))
    selector = selectors[job["runtime_variant"]]
    captured = []
    def capture(row, request):
        if request is not job or captured:
            raise ValueError("unexpected capture boundary")
        captured.append(True)
        row.update(runtime_variant=job["runtime_variant"], repeat=job["repeat"],
                   validation_state="pending_full_audit")
        row["integrity_sha256"] = pilot.digest(row)
        write_json(path, row)
    isolated_worker(pilot.worker, selector, capture)(job)
    if len(captured) != 1:
        raise ValueError("capture boundary missing")
    return dict(status="ok", job_id=job["job_id"], raw_sha256=sha256_file(path))


def replay_worker(job):
    record = job["record"]
    if sha256_file(ROOT / record["path"]) != record["sha256"]:
        raise ValueError("historical trace changed")
    saved = read_json(ROOT / record["path"])
    env = pilot.make_env(dict(job, budget=3000))
    state = pilot._plain(env.reset(seed=job["case"]["solver_seed"]))
    expected = saved["initial_state"]
    reference, optimized = pilot.TimedPool(job["case"]), SingleFullCheckPool(job["case"])
    if pilot.state_fingerprint(state) != pilot.state_fingerprint(expected):
        raise ValueError("replay initial mismatch")
    for d, event in enumerate(saved["events"]):
        a = reference.select(env, deepcopy(state), d)
        b = optimized.select(env, deepcopy(state), d)
        if a != b or a != (event["selected_index"], event["pool"]):
            raise ValueError(f"candidate/score/choice mismatch at {d}")
        step = pilot._plain(env.step_experimental_pp(event["action"], 300., "annealed",
                                                   event["temperature"], event["uniform"]))
        state = step["observation"]
        expected = pilot.apply_state_delta(expected, event["delta"])
        if pilot.state_fingerprint(state) != pilot.state_fingerprint(expected):
            raise ValueError(f"full prefix replay mismatch at {d}")
        for field in ("repair_order", "neighborhood", "replan_success", "pp_failure_reason"):
            if step["metrics"][field] != event["metrics"][field]:
                raise ValueError("PP behavior mismatch: " + field)
    return dict(status="ok", job_id=job["job_id"], states=len(saved["events"]),
                complete_prefix_equal=True, final_fingerprint=pilot.state_fingerprint(state),
                registration_sha256=job["registration_sha256"])


def preflight():
    r = verify()
    identity = sha256_file(OUT / "registration.json")
    jobs = [dict(job_id=c["case_id"], case=c, config=r["config"], arm="dual16_sa",
                 record=r["records"][c["case_id"]], registration_sha256=identity,
                 plan_sha256=identity, supervisor_stage="replay") for c in r["cases"]]
    with pilot._CollectionRunLock(OUT, identity, "full-prefix-equivalence"):
        for j in jobs:
            if (OUT / "preflight" / (j["job_id"] + ".json")).exists():
                raise ValueError("preflight output exists; inspect rather than overwrite")
        rows = pilot._run_jobs(replay_worker, jobs, workers=r["diagnostic_workers"],
            phase="replay", output_root=OUT, run_fingerprint=identity, timeout_seconds=1800,
            failure_result=supervisor.failure_result, stop_on_failure=True,
            on_result=lambda x: write_json(OUT / "preflight" / (x["job_id"] + ".json"), x))
        if len(rows) != 8 or any(x["status"] != "ok" for x in rows):
            raise ValueError("equivalence failed; timing prohibited")
        report = dict(complete=True, registration_sha256=identity, cases=8,
            states=sum(x["states"] for x in rows),
            files={x["job_id"] + ".json": sha256_file(OUT / "preflight" / (x["job_id"] + ".json")) for x in rows})
        write_json(OUT / "preflight_report.json", report)
        return report


def load_validated(job):
    path = ROOT / job["capture_path"]
    receipt = read_json(OUT / "audits" / (job["job_id"] + ".json"))
    if (receipt.get("status") != "ok" or not receipt.get("full_audit")
            or receipt.get("job_id") != job["job_id"]
            or receipt.get("raw_sha256") != sha256_file(path)
            or receipt.get("registration_sha256") != job["registration_sha256"]):
        raise ValueError("invalid full audit receipt")
    row = supervisor.checked_row(path, job)
    if (row.get("runtime_variant"), row.get("repeat")) != (job["runtime_variant"], job["repeat"]):
        raise ValueError("runtime/repeat identity mismatch")
    return row


def stage_run(job, stage, worker, timeout):
    folder = OUT / "attempts" / job["job_id"] / stage
    path = folder / f"{len(list(folder.glob('*.json'))) + 1:03d}.json"
    rows = pilot._run_jobs(worker, [dict(job, supervisor_stage=stage)], workers=1,
        phase=stage, output_root=OUT, run_fingerprint=job["registration_sha256"],
        timeout_seconds=timeout, failure_result=supervisor.failure_result,
        on_result=lambda x: write_json(path, x), stop_on_failure=True)
    if len(rows) != 1 or rows[0]["status"] != "ok":
        raise ValueError("stage failed; inspect preserved attempt: " + stage)
    return rows[0]


def collect(resume=False):
    r = verify()
    identity = sha256_file(OUT / "registration.json")
    pre = read_json(OUT / "preflight_report.json")
    if not pre["complete"] or pre["cases"] != 8 or pre["registration_sha256"] != identity:
        raise ValueError("preflight incomplete")
    for name, sha in pre["files"].items():
        if sha256_file(OUT / "preflight" / name) != sha:
            raise ValueError("preflight changed")
    with pilot._CollectionRunLock(OUT, identity, "serial-runtime-comparison"):
        if resume:
            (OUT / "STOP_AFTER_EPISODE.json").unlink(missing_ok=True)
        scheduled = schedule(r)
        for i, job in enumerate(scheduled):
            receipt = OUT / "audits" / (job["job_id"] + ".json")
            if receipt.exists():
                load_validated(job)
                continue
            if (OUT / "STOP_AFTER_EPISODE.json").exists():
                write_json(OUT / "run_status.json", dict(status="paused", completed=i, total=len(scheduled)))
                return dict(paused=True, completed=i)
            path = ROOT / job["capture_path"]
            try:
                if not path.exists():
                    if (OUT / "attempts" / job["job_id"] / "solver").exists():
                        raise ValueError("prior attempt without capture; no automatic solver retry")
                    write_json(OUT / "run_status.json", dict(status="running", stage="solver", completed=i,
                        total=len(scheduled), active_job=job["job_id"]))
                    print(f"solver {i+1}/{len(scheduled)} {job['job_id']}", flush=True)
                    stage_run(job, "solver", solver_worker, r["solver_fuse"])
                supervisor.checked_row(path, job)
                job["raw_sha256"] = sha256_file(path)
                write_json(OUT / "run_status.json", dict(status="running", stage="audit", completed=i,
                    total=len(scheduled), active_job=job["job_id"]))
                result = stage_run(job, "audit", supervisor.audit_worker, r["audit_fuse"])
                write_json(receipt, result)
                row = load_validated(job)
                print(json.dumps(dict(completed=i+1, success=row["success_within_budget"],
                    ttf=row["ttf_seconds"], steps=len(row["events"]))), flush=True)
            except BaseException as error:
                write_json(OUT / "run_status.json", dict(status="error_or_interrupted", completed=i,
                    active_job=job["job_id"], error=f"{type(error).__name__}: {error}", raw_preserved=path.exists()))
                raise
        files = {}
        for j in scheduled:
            load_validated(j)
            for p in (ROOT / j["capture_path"], OUT / "audits" / (j["job_id"] + ".json")):
                files[p.relative_to(ROOT).as_posix()] = sha256_file(p)
        result = dict(complete=True, jobs=len(scheduled), registration_sha256=identity, files=files)
        write_json(OUT / "timed_report.json", result)
        write_json(OUT / "run_status.json", dict(status="complete", completed=len(scheduled)))
        return dict(complete=True, jobs=len(scheduled))


def compare_trajectories(a, b):
    sa, sb = a["initial_state"], b["initial_state"]
    if pilot.state_fingerprint(sa) != pilot.state_fingerprint(sb):
        raise ValueError("paired initial mismatch")
    equal = 0
    for x, y in zip(a["events"], b["events"]):
        for key in ("pool", "selected_index", "action", "temperature", "uniform"):
            if x[key] != y[key]:
                raise ValueError("paired choice mismatch: " + key)
        for key in ("repair_order", "neighborhood"):
            if x["metrics"][key] != y["metrics"][key]:
                raise ValueError("paired PP order/neighborhood mismatch")
        sa, sb = pilot.apply_state_delta(sa, x["delta"]), pilot.apply_state_delta(sb, y["delta"])
        if pilot.state_fingerprint(sa) != pilot.state_fingerprint(sb):
            if "time_limit" in (x["metrics"]["pp_failure_reason"], y["metrics"]["pp_failure_reason"]):
                return dict(common_steps_equal=equal, full_trajectory_equal=False, deadline_divergence=True)
            raise ValueError("paired path/counter mismatch without PP deadline")
        equal += 1
    return dict(common_steps_equal=equal, full_trajectory_equal=len(a["events"]) == len(b["events"]),
                deadline_divergence=len(a["events"]) != len(b["events"]))


def analyze():
    r = verify()
    report = read_json(OUT / "timed_report.json")
    if not report["complete"] or report["jobs"] != 32 or report["registration_sha256"] != sha256_file(OUT / "registration.json"):
        raise ValueError("incomplete timing")
    for name, sha in report["files"].items():
        if sha256_file(ROOT / name) != sha:
            raise ValueError("timing evidence changed")
    rows = [load_validated(j) for j in schedule(r)]
    pairs = []
    for repeat in range(2):
        for c in r["cases"]:
            a, b = [next(x for x in rows if x["case_id"] == c["case_id"] and x["repeat"] == repeat
                         and x["runtime_variant"] == v) for v in VARIANTS]
            pairs.append(dict(case_id=c["case_id"], map_id=c["map_id"], repeat=repeat,
                initial_conflicts=a["initial_state"]["num_of_colliding_pairs"],
                reference_ttf=a["ttf_seconds"], optimized_ttf=b["ttf_seconds"],
                reference_success=a["success_within_budget"], optimized_success=b["success_within_budget"],
                capped_difference=pilot.capped_time(b)-pilot.capped_time(a),
                **compare_trajectories(a,b)))
    summary = {v: dict(episodes=16, successes=sum(x["success_within_budget"] for x in rows if x["runtime_variant"] == v),
        mean_capped_ttf=statistics.mean(pilot.capped_time(x) for x in rows if x["runtime_variant"] == v),
        selection_seconds=sum(x["selection_seconds"] for x in rows if x["runtime_variant"] == v),
        pp_seconds=sum(x["pp_seconds"] for x in rows if x["runtime_variant"] == v)) for v in VARIANTS}
    improvement=100*(1-summary[VARIANTS[1]]["mean_capped_ttf"]/summary[VARIANTS[0]]["mean_capped_ttf"])
    map_differences = {m:statistics.mean(p["capped_difference"] for p in pairs if p["map_id"]==m)
                       for m in sorted({p["map_id"] for p in pairs})}
    rng = random.Random(20260912)
    values = list(map_differences.values())
    samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(5000))
    common = [p for p in pairs if p["reference_success"] and p["optimized_success"]]
    result=dict(schema="lns2.sa_single_check_analysis.v1", summary=summary, pairs=pairs,
        mean_capped_ttf_improvement_percent=improvement,
        maps=map_differences, paired_seconds_map_bootstrap_ci95=[samples[124], samples[4874]],
        common_success_pairs=len(common),
        common_success_mean_ttf={v:statistics.mean(p[k] for p in common) if common else None
            for v,k in zip(VARIANTS,("reference_ttf","optimized_ttf"))},
        registration_sha256=sha256_file(OUT / "registration.json"), development_only=True,
        independent_maps=4, paired_cases=8, repeated_pairs=16, no_default_promotion=True)
    write_json(OUT / "analysis.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("register", "verify", "preflight", "collect", "resume", "stop", "analyze"))
    args = p.parse_args()
    if args.phase == "stop":
        write_json(OUT / "STOP_AFTER_EPISODE.json", dict(requested=True))
        result = dict(stop_after_current_episode=True)
    elif args.phase in ("collect", "resume"):
        result = collect(args.phase == "resume")
    elif args.phase == "verify":
        result = dict(verified=bool(verify()))
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
