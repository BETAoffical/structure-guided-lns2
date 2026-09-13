"""Paired high-pressure first-feasible replay; no training or phase two."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_path_quality as q

OUT = ROOT / "build/sa-pressure-first-feasible-v1"
SOURCE = "build/path-quality-pressure-evaluation-v1/preflight.json"
PROTOCOL = dict(first_feasible_budget_seconds=120, total_planning_budgets_seconds=[])
REGISTERED = ("scripts/run_sa_pressure_first_feasible.py",
              "tests/evaluation/test_sa_pressure_first_feasible.py",
              "docs/SA_PRESSURE_FIRST_FEASIBLE_PROTOCOL_ZH.md")


def schedule(cases):
    rows = q.execution.execution_schedule(cases, PROTOCOL)
    mapping = dict(zip(q.execution.CONTROLLERS, q.CONTROLLERS))
    for row in rows:
        row["controller"] = mapping[row["controller"]]
        row["job_id"] = q.json_fingerprint({k: row[k] for k in
            ("task_id", "solver_seed", "controller", "protocol", "budget_seconds")})[:24]
    return rows


def check_cases(cases):
    if len(cases) != 48 or len({c["map_id"] for c in cases}) != 8:
        raise ValueError("expected complete 48-task/eight-map cohort")
    if any(c["solver_seeds"] != [61, 62] or c["family"] != "warehouse" or
           c["status"] != "static_ready_runtime_unverified" for c in cases):
        raise ValueError("cohort identity changed")
    for map_id in {c["map_id"] for c in cases}:
        cells = Counter((c["pressure_design"]["density"], c["pressure_design"]["mode"]["name"])
                        for c in cases if c["map_id"] == map_id)
        if cells != Counter({(d, od): 1 for d in (.15, .2, .25)
                             for od in ("balanced", "bottleneck_eligible")}):
            raise ValueError("density/OD coverage changed")


def prepare():
    if (OUT / "registration.json").exists():
        raise ValueError("registration already exists")
    original = q.verify()
    old = q.read_json(ROOT / SOURCE)
    if q.json_fingerprint({k:v for k,v in old.items() if k != "fingerprint"}) != old["fingerprint"]:
        raise ValueError("historical preflight corrupt")
    cases = old["cases"]
    check_cases(cases)
    inputs = dict(original["inputs"])
    inputs[SOURCE] = q.sha256_file(ROOT/SOURCE)
    for c in cases:
        for name in c["files"].values():
            if q.sha256_file(ROOT/name) != old["input_sha256"][name]:
                raise ValueError("historical input changed")
            inputs[name] = old["input_sha256"][name]
        if q.audit_task(ROOT/c["files"]["map_file"], ROOT/c["files"]["scenario_file"],
                        ROOT/c["files"]["task_file"], c["static_audit"]["agent_count"]) != c["static_audit"]:
            raise ValueError("static audit changed")
    for name in REGISTERED:
        inputs[name] = q.sha256_file(ROOT/name)
    r = dict(schema="lns2.sa_pressure_first_feasible.v1", cases=cases, schedule=schedule(cases),
             template=original["template"], inputs=inputs, native_sha256=q.NATIVE_SHA,
             native_file=q.NATIVE, protocol=PROTOCOL, workers=1, controllers=list(q.CONTROLLERS),
             sa=original["sa"], engineering_variant_used=False, promotion_allowed=False,
             source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    r["fingerprint"] = q.json_fingerprint(r)
    q.write_json(OUT/"registration.json", r)
    return dict(prepared=True, tasks=48, maps=8, resets=96, episodes=288)


def verify(native=False):
    r = q.read_json(OUT/"registration.json")
    if q.json_fingerprint({k:v for k,v in r.items() if k != "fingerprint"}) != r["fingerprint"]:
        raise ValueError("registration corrupt")
    check_cases(r["cases"])
    if (r["protocol"], r["native_sha256"], r["native_file"], r["workers"], r["controllers"],
        r["sa"], r["engineering_variant_used"]) != (PROTOCOL, q.NATIVE_SHA, q.NATIVE, 1,
        list(q.CONTROLLERS), dict(initial_temperature=1000., cooling=.99), False):
        raise ValueError("frozen protocol changed")
    if r["schedule"] != schedule(r["cases"]):
        raise ValueError("schedule changed")
    for name, digest in r["inputs"].items():
        if q.sha256_file(ROOT/name) != digest:
            raise ValueError("input changed: " + name)
    if native and q.native_identity()["sha256"] != q.NATIVE_SHA:
        raise ValueError("incorrect shared native")
    return r


def reset_worker(job):
    if q.native_identity()["sha256"] != q.NATIVE_SHA:
        raise ValueError("reset native mismatch")
    c = job["case"]
    j = q.worker_job(c, job["item"], job["template"], OUT/"admission", job["binding"])
    env = q._make_environment(j["dataset_root"], j["row"], j["environment"], "Adaptive")
    state = q._plain(env.reset(seed=j["solver_seed"]))
    if not state["initial_solution_complete"]:
        raise ValueError("initial PP incomplete")
    journal = q.execution.PathJournal(OUT/"admission", job["binding"], ROOT/c["files"]["map_file"],
                                      ROOT/c["files"]["scenario_file"])
    return dict(status="ok", job_id=job["job_id"], key=job["job_id"], binding=job["binding"],
                state_fingerprint=q.state_fingerprint(state), initial_quality=journal.quality(state),
                paths=[a["path"] for a in sorted(state["agents"], key=lambda a:a["id"])])


def validate():
    r = verify(True)
    with q._CollectionRunLock(OUT, r["fingerprint"], "pressure-admission"):
        if (OUT/"admission_started.json").exists():
            raise ValueError("admission attempt exists; inspect instead of automatic retry")
        jobs = [dict(job_id=q.admission_key(i), item=i,
                     case=next(c for c in r["cases"] if c["task_id"] == i["task_id"]),
                     template=r["template"], binding=r["fingerprint"])
                for i in r["schedule"] if i["controller"] == "official_adaptive"]
        q.write_json(OUT/"admission_started.json", dict(jobs=len(jobs)))
        def failed(job, status, error):
            return dict(status=status, job_id=job["job_id"], error=str(error))
        rows = q._run_jobs(reset_worker, jobs, workers=4, phase="pressure-admission",
            output_root=OUT/"admission-progress", run_fingerprint=r["fingerprint"], timeout_seconds=240,
            failure_result=failed, stop_on_failure=True,
            on_result=lambda row:q.write_json(OUT/"admission"/(row["job_id"]+".json"), row))
        result = dict(passed=len(rows)==96 and all(x["status"]=="ok" for x in rows),
                      completed=len(rows), binding=r["fingerprint"],
                      files={x["job_id"]:q.sha256_file(OUT/"admission"/(x["job_id"]+".json")) for x in rows})
        q.write_json(OUT/"admission.json", result)
        return result


def anchors(r):
    a = q.read_json(OUT/"admission.json")
    expected = {q.admission_key(i) for i in r["schedule"]}
    if not a["passed"] or a["binding"] != r["fingerprint"] or set(a["files"]) != expected:
        raise ValueError("admission incomplete")
    result = {}
    for key, digest in a["files"].items():
        p = OUT/"admission"/(key+".json")
        if q.sha256_file(p) != digest:
            raise ValueError("anchor changed")
        result[key] = q.read_json(p)
    return result


def spec_for(r, item, a, folder=None):
    return q.spec_for(r, item, a[q.admission_key(item)], folder or OUT/"episodes"/item["job_id"])


def inspect(r, item, a):
    spec = spec_for(r, item, a)
    return q.inspect_episode(ROOT, spec["case"], item, Path(spec["output"]), spec["binding"], a[q.admission_key(item)])


def smoke():
    r = verify(True)
    a = anchors(r)
    if (OUT/"smoke_started.json").exists():
        raise ValueError("smoke attempt exists")
    q.write_json(OUT/"smoke_started.json", dict(binding=r["fingerprint"]))
    results = []
    for original in r["schedule"][:3]:
        item = dict(original, budget_seconds=2., job_id="smoke-"+original["controller"])
        spec = q.spec_for(r, item, a[q.admission_key(original)], OUT/"smoke"/item["job_id"])
        result = q.execution.supervise_episode(spec, authorized=True, child_entry=q.child)
        if result["error"] or result["status"] == "external_timeout":
            raise ValueError("smoke error; inspect preserved output")
        q.audit_trace(spec)
        results.append(dict(controller=item["controller"], status=result["status"]))
    report = dict(passed=True, binding=r["fingerprint"], not_performance_evidence=True, jobs=results)
    q.write_json(OUT/"smoke.json", report)
    return report


def collect(resume=False):
    r = verify(True)
    a = anchors(r)
    smoke_result = q.read_json(OUT/"smoke.json")
    if not smoke_result["passed"] or smoke_result["binding"] != r["fingerprint"]:
        raise ValueError("smoke not passed")
    with q._CollectionRunLock(OUT, r["fingerprint"], "pressure-first-feasible"):
        mp = OUT/"manifest.json"
        if mp.exists() and not resume:
            raise ValueError("run exists; explicit resume required")
        manifest = q.read_json(mp) if mp.exists() else dict(binding=r["fingerprint"], jobs={})
        if manifest["binding"] != r["fingerprint"] or not set(manifest["jobs"]).issubset({i["job_id"] for i in r["schedule"]}):
            raise ValueError("manifest identity/keys invalid")
        q.write_json(OUT/("environment-resume-"+str(len(manifest["jobs"]))+".json"), q.runtime_environment())
        try:
            for item in r["schedule"]:
                spec = spec_for(r, item, a)
                folder = Path(spec["output"])
                old = manifest["jobs"].get(item["job_id"])
                if old:
                    for name, digest in old["files"].items():
                        if q.sha256_file(folder/name) != digest:
                            raise ValueError("completed artifact changed")
                    continue
                if (OUT/"STOP_AFTER_EPISODE.json").exists():
                    q.write_json(OUT/"status.json", dict(status="stopped", completed=len(manifest["jobs"]), total=288))
                    return dict(stopped=True, completed=len(manifest["jobs"]))
                if folder.exists():
                    raise ValueError("uncommitted episode exists; audit interruption before retry")
                q.write_json(OUT/"status.json", dict(status="running", completed=len(manifest["jobs"]), total=288, current=item))
                print(f"START {item['schedule_index']+1}/288 {item['controller']} {item['task_id']} seed={item['solver_seed']}", flush=True)
                result = q.execution.supervise_episode(spec, authorized=True, child_entry=q.child)
                if result["error"]:
                    raise ValueError("episode error; inspect retained artifacts")
                row = inspect(r, item, a)
                q.audit_trace(spec)
                files = {p.relative_to(folder).as_posix():q.sha256_file(p) for p in folder.rglob("*.json*") if p.is_file()}
                manifest["jobs"][item["job_id"]] = dict(status=row["status"], success=row["success"], files=files)
                q.write_json(mp, manifest)
                print(f"COMPLETE {len(manifest['jobs'])}/288 {row['status']} ttf={row['ttf_seconds']}", flush=True)
            q.write_json(OUT/"status.json", dict(status="complete", completed=288, total=288))
            return dict(complete=True, episodes=288)
        except BaseException:
            q.write_json(OUT/"status.json", dict(status="failed_or_interrupted", completed=len(manifest["jobs"]), total=288))
            raise


def analyze():
    r = verify()
    a = anchors(r)
    manifest = q.read_json(OUT/"manifest.json")
    if manifest["binding"] != r["fingerprint"] or set(manifest["jobs"]) != {i["job_id"] for i in r["schedule"]}:
        raise ValueError("incomplete results")
    rows = []
    for item in r["schedule"]:
        for name, digest in manifest["jobs"][item["job_id"]]["files"].items():
            if q.sha256_file(OUT/"episodes"/item["job_id"]/name) != digest:
                raise ValueError("result changed")
        row = inspect(r, item, a)
        case = next(c for c in r["cases"] if c["task_id"] == item["task_id"])
        row.update(design_density=case["pressure_design"]["density"], od_mode=case["pressure_design"]["mode"]["name"])
        rows.append(row)
    stats = q.summarize(rows, controllers=q.CONTROLLERS, comparisons=q.COMPARISONS)
    strata = {f"density:{d}":q.summarize([x for x in rows if x["design_density"]==d], controllers=q.CONTROLLERS,
                                       comparisons=q.COMPARISONS) for d in (.15,.2,.25)}
    report = dict(schema="lns2.sa_pressure_report.v1", complete=True, scheduled=288, binding=r["fingerprint"],
                  native_sha256=q.NATIVE_SHA, promotion_allowed=False, statistics=stats, density_strata=strata, episodes=rows)
    q.publish_analysis(OUT/"analysis", rows, report)
    return dict(complete=True, scheduled=288, successes=sum(x["success"] for x in rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "validate", "smoke", "run", "resume", "stop", "analyze"))
    args = parser.parse_args()
    if args.phase == "stop":
        q.write_json(OUT/"STOP_AFTER_EPISODE.json", dict(requested=True))
        result = dict(stop_after_episode=True)
    elif args.phase in {"run", "resume"}:
        result = collect(args.phase == "resume")
        if result.get("complete"):
            result = analyze()
    elif args.phase == "verify":
        result = dict(verified=bool(verify(True)))
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, indent=2))
