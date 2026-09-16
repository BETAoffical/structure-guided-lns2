"""Prepare an Official+SA supplement; formal timing requires separate authorization."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_pressure_first_feasible as historical
from scripts import run_sa_path_quality as q
from experiments._common import atomic_write_text

OUT = ROOT / "build/official-sa-pressure-supplement-v1"
OLD = ROOT / "build/sa-pressure-first-feasible-v1"
OLD_BINDING = "bece1ef3f35cef210efcf0d99407061cfe5343b82e3473fc1c97acd7dc6a7181"
REPORT_SHA = "16a16a350e1ffa61e384ab5ba6a6f24f896c5e73d4382a34905c5226238e611a"
CHANGED_SOURCE = {
    "scripts/run_sa_path_quality.py", "src/python_bindings.cpp",
    "third_party/mapf_lns2/inc/CBS/PBS.h", "third_party/mapf_lns2/inc/InitLNS.h",
    "third_party/mapf_lns2/inc/RepairPolicy.h", "third_party/mapf_lns2/src/CBS/PBS.cpp",
    "third_party/mapf_lns2/src/ConstraintTable.cpp", "third_party/mapf_lns2/src/InitLNS.cpp",
}
OWN_FILES = ("scripts/prepare_official_sa_pressure.py",
             "tests/evaluation/test_official_sa_pressure.py",
             "docs/OFFICIAL_SA_PRESSURE_SUPPLEMENT_PROTOCOL_ZH.md")


def schedule(cases):
    rows = []
    for old in historical.schedule(cases):
        if old["controller"] != "official_adaptive":
            continue
        item = dict(old, controller="official_sa", schedule_index=len(rows), within_pair_position=0)
        item["job_id"] = q.json_fingerprint({k: item[k] for k in
            ("task_id", "solver_seed", "controller", "protocol", "budget_seconds")})[:24]
        rows.append(item)
    return rows


def check_fingerprint(r):
    if q.json_fingerprint({k: v for k, v in r.items() if k != "fingerprint"}) != r["fingerprint"]:
        raise ValueError("registration fingerprint mismatch")


def contained(base, relative):
    path = (base / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(base.resolve()):
        raise ValueError("artifact path escapes root")
    return path


def prepare():
    if (OUT / "registration.json").exists():
        raise ValueError("registration exists; do not overwrite")
    old = q.read_json(OLD / "registration.json")
    check_fingerprint(old)
    historical.check_cases(old["cases"])
    if (old["fingerprint"], old["native_sha256"], old["sa"], old["schedule"]) != (
            OLD_BINDING, q.NATIVE_SHA, dict(initial_temperature=1000., cooling=.99),
            historical.schedule(old["cases"])):
        raise ValueError("historical protocol changed")
    if q.sha256_file(OLD / "analysis/report.json") != REPORT_SHA:
        raise ValueError("historical report changed")
    inputs, drift = {}, {}
    for name, digest in old["inputs"].items():
        current = q.sha256_file(contained(ROOT, name))
        if current != digest:
            if name not in CHANGED_SOURCE:
                raise ValueError("unreviewed historical input drift: " + name)
            drift[name] = dict(historical=digest, current=current,
                              role="runner extension" if name.endswith(".py") else "not compiled; frozen binary retained")
        inputs[name] = current
    if inputs[q.NATIVE] != q.NATIVE_SHA:
        raise ValueError("frozen binary changed")
    for name in OWN_FILES:
        inputs[name] = q.sha256_file(ROOT / name)
    for c in old["cases"]:
        if q.audit_task(ROOT/c["files"]["map_file"], ROOT/c["files"]["scenario_file"],
                        ROOT/c["files"]["task_file"], c["static_audit"]["agent_count"]) != c["static_audit"]:
            raise ValueError("static task audit changed")
    a = q.read_json(OLD / "admission.json")
    manifest = q.read_json(OLD / "manifest.json")
    if not a["passed"] or a["binding"] != OLD_BINDING or manifest["binding"] != OLD_BINDING:
        raise ValueError("historical admission/manifest mismatch")
    if set(manifest["jobs"]) != {i["job_id"] for i in old["schedule"]}:
        raise ValueError("historical run incomplete")
    evidence = {}
    for name in ("registration.json", "admission.json", "manifest.json", "analysis/report.json"):
        path = OLD / name
        evidence[path.relative_to(ROOT).as_posix()] = q.sha256_file(path)
    if set(a["files"]) != {q.admission_key(i) for i in schedule(old["cases"])}:
        raise ValueError("historical anchor coverage mismatch")
    for key, digest in a["files"].items():
        evidence[(OLD / "admission" / (key+".json")).relative_to(ROOT).as_posix()] = digest
    for i in old["schedule"]:
        if i["controller"] != "dual16_sa":
            continue
        files = manifest["jobs"][i["job_id"]]["files"]
        if not {"binding.json", "initial.json", "terminal.json", "supervisor.json"}.issubset(files):
            raise ValueError("historical episode evidence incomplete")
        folder = OLD / "episodes" / i["job_id"]
        for name, digest in files.items():
            evidence[contained(folder, name).relative_to(ROOT).as_posix()] = digest
    for name, digest in evidence.items():
        if q.sha256_file(contained(ROOT, name)) != digest:
            raise ValueError("historical evidence changed: " + name)
    r = dict(schema="lns2.official_sa_pressure_supplement.v1", cases=old["cases"],
             template=old["template"], schedule=schedule(old["cases"]), inputs=inputs,
             evidence=evidence, source_drift=drift, historical_binding=OLD_BINDING,
             native_file=q.NATIVE, native_sha256=q.NATIVE_SHA,
             controllers=["official_sa"], workers=1, restart=False, repair_iteration_cap=None,
             sa=old["sa"], comparison="cross_batch_exploratory_not_concurrent_timing",
             promotion_allowed=False, source_commit=subprocess.check_output(
                 ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    r["fingerprint"] = q.json_fingerprint(r)
    q.write_json(OUT / "registration.json", r)
    return dict(prepared=True, episodes=96, historical_dual16_sa=96, source_drift=drift)


def verify(native=False, evidence=False):
    r = q.read_json(OUT / "registration.json")
    check_fingerprint(r)
    historical.check_cases(r["cases"])
    if (r["schedule"], r["controllers"], r["workers"], r["restart"], r["repair_iteration_cap"],
        r["native_file"], r["native_sha256"], r["sa"], r["historical_binding"]) != (
        schedule(r["cases"]), ["official_sa"], 1, False, None, q.NATIVE, q.NATIVE_SHA,
        dict(initial_temperature=1000., cooling=.99), OLD_BINDING):
        raise ValueError("frozen supplement protocol changed")
    for name, digest in {**r["inputs"], **(r["evidence"] if evidence else {})}.items():
        if q.sha256_file(contained(ROOT, name)) != digest:
            raise ValueError("registered file changed: " + name)
    if native and q.native_identity()["sha256"] != q.NATIVE_SHA:
        raise ValueError("loaded native identity mismatch")
    return r


def old_anchors(r):
    a = q.read_json(OLD / "admission.json")
    result = {}
    for key, digest in a["files"].items():
        path = OLD / "admission" / (key+".json")
        if q.sha256_file(path) != digest:
            raise ValueError("historical anchor changed")
        result[key] = q.read_json(path)
    return result


def reset_worker(job):
    if q.native_identity()["sha256"] != q.NATIVE_SHA:
        raise ValueError("reset native mismatch")
    c, i = job["case"], job["item"]
    j = q.worker_job(c, i, job["template"], OUT/"admission", job["binding"])
    env = q._make_environment(j["dataset_root"], j["row"], j["environment"], "Adaptive")
    state = q._plain(env.reset(seed=i["solver_seed"]))
    expected = job["anchor"]
    if not state["initial_solution_complete"] or q.state_fingerprint(state) != expected["state_fingerprint"]:
        raise ValueError("reset fingerprint mismatch")
    if [a["path"] for a in sorted(state["agents"], key=lambda a: a["id"])] != expected["paths"]:
        raise ValueError("reset paths mismatch")
    return dict(status="ok", job_id=job["job_id"], binding=job["binding"],
                state_fingerprint=q.state_fingerprint(state), initial_conflicts=state["num_of_colliding_pairs"])


def validate(workers=20):
    r = verify(True)
    a = old_anchors(r)
    with q._CollectionRunLock(OUT, r["fingerprint"], "supplement-admission"):
        if (OUT/"admission_started.json").exists():
            raise ValueError("admission attempt exists; inspect before retry")
        jobs = [dict(job_id=q.admission_key(i), item=i, template=r["template"], binding=r["fingerprint"],
                     anchor=a[q.admission_key(i)], case=next(c for c in r["cases"] if c["task_id"]==i["task_id"]))
                for i in r["schedule"]]
        q.write_json(OUT/"admission_started.json", dict(workers=workers, jobs=96))
        def failed(job, status, error):
            return dict(status=status, job_id=job["job_id"], error=str(error))
        rows = q._run_jobs(reset_worker, jobs, workers=workers, phase="official-sa-admission",
            output_root=OUT/"admission-progress", run_fingerprint=r["fingerprint"],
            timeout_seconds=240, failure_result=failed, stop_on_failure=True,
            on_result=lambda row: q.write_json(OUT/"admission"/(row["job_id"]+".json"), row))
        result = dict(passed=len(rows)==96 and all(x["status"]=="ok" for x in rows),
                      binding=r["fingerprint"], completed=len(rows), not_performance_evidence=True,
                      files={x["job_id"]: q.sha256_file(OUT/"admission"/(x["job_id"]+".json")) for x in rows})
        q.write_json(OUT/"admission.json", result)
        return result


def check_admission(r):
    a = q.read_json(OUT/"admission.json")
    if not a["passed"] or a["binding"] != r["fingerprint"] or set(a["files"]) != {q.admission_key(i) for i in r["schedule"]}:
        raise ValueError("admission incomplete")
    for key, digest in a["files"].items():
        if q.sha256_file(OUT/"admission"/(key+".json")) != digest:
            raise ValueError("admission artifact changed")


def spec_for(r, item, anchors, folder=None):
    return q.spec_for(r, item, anchors[anchor_key(item)], folder or OUT/"episodes"/item["job_id"])


def anchor_key(item):
    # Smoke changes only the execution budget, not the registered reset identity.
    return q.admission_key(dict(item, budget_seconds=120.))


def inspect(r, item, anchors, spec):
    return q.inspect_episode(ROOT, spec["case"], item, Path(spec["output"]), spec["binding"], anchors[anchor_key(item)])


def workload(folder):
    binding = q.read_json(folder/"binding.json")["binding"]
    first = q.execution.read_artifact(folder/"initial.json", binding)["observation"]["low_level"]
    last = q.execution.read_artifact(folder/"terminal.json", binding)["observation"]["low_level"]
    phase = q.execution.read_artifact(folder/"first_phase_result.json", binding)
    return dict(pp_seconds=phase["summary"]["pp_seconds"],
                selection_seconds=phase["summary"]["selection_seconds"],
                low_level_delta={k:last[k]-first[k] for k in first if isinstance(first[k], (int,float))})


def smoke():
    r = verify(True)
    check_admission(r)
    a = old_anchors(r)
    with q._CollectionRunLock(OUT, r["fingerprint"], "supplement-smoke"):
        if (OUT/"smoke_started.json").exists():
            raise ValueError("smoke attempt exists; do not retry silently")
        q.write_json(OUT/"smoke_started.json", dict(binding=r["fingerprint"]))
        rows, files = [], {}
        for density in (.15, .25):
            task = next(c for c in r["cases"] if c["pressure_design"]["density"]==density)
            original = next(i for i in r["schedule"] if i["task_id"]==task["task_id"])
            item = dict(original, budget_seconds=2., job_id="smoke-"+original["job_id"])
            spec = spec_for(r, item, a, OUT/"smoke"/item["job_id"])
            result = q.execution.supervise_episode(spec, authorized=True, child_entry=q.child)
            if result["error"] or result["status"]=="external_timeout":
                raise ValueError("smoke error; inspect retained artifacts")
            q.audit_trace(spec)
            rows.append(inspect(r, item, a, spec))
        for path in (OUT/"smoke").rglob("*.json*"):
            files[path.relative_to(OUT).as_posix()] = q.sha256_file(path)
        result = dict(passed=True, binding=r["fingerprint"], jobs=rows, files=files,
                      not_performance_evidence=True)
        q.write_json(OUT/"smoke.json", result)
        return dict(passed=True, jobs=len(rows), not_performance_evidence=True)


def ready(native=True):
    r = verify(native, evidence=True)
    check_admission(r)
    s = q.read_json(OUT/"smoke.json")
    if not s["passed"] or s["binding"] != r["fingerprint"] or len(s["jobs"]) != 2:
        raise ValueError("smoke not passed")
    for name, digest in s["files"].items():
        if q.sha256_file(contained(OUT, name)) != digest:
            raise ValueError("smoke artifact changed")
    return r


def collect(*, authorized=False, resume=False, clear_stop=False):
    if not authorized:
        raise PermissionError("separate timing authorization required")
    if clear_stop and not resume:
        raise ValueError("clear-stop requires explicit resume")
    r = ready()
    a = old_anchors(r)
    with q._CollectionRunLock(OUT, r["fingerprint"], "official-sa-formal"):
        mp, stop = OUT/"manifest.json", OUT/"STOP_AFTER_EPISODE.json"
        if mp.exists() and not resume:
            raise ValueError("run exists; explicit resume required")
        manifest = q.read_json(mp) if mp.exists() else dict(binding=r["fingerprint"], jobs={})
        if manifest["binding"] != r["fingerprint"] or not set(manifest["jobs"]).issubset({i["job_id"] for i in r["schedule"]}):
            raise ValueError("manifest identity mismatch")
        if clear_stop and stop.exists():
            stop.unlink()
        q.write_json(OUT/("environment-"+str(len(manifest["jobs"]))+".json"), q.runtime_environment())
        try:
            for item in r["schedule"]:
                if stop.exists():
                    q.write_json(OUT/"status.json", dict(status="stopped", completed=len(manifest["jobs"]), total=96))
                    return dict(stopped=True, completed=len(manifest["jobs"]))
                spec = spec_for(r, item, a)
                folder = Path(spec["output"])
                if item["job_id"] in manifest["jobs"]:
                    for name, digest in manifest["jobs"][item["job_id"]]["files"].items():
                        if q.sha256_file(contained(folder, name)) != digest:
                            raise ValueError("completed artifact changed")
                    inspect(r, item, a, spec)
                    continue
                if folder.exists():
                    raise ValueError("uncommitted episode; audit interruption before retry")
                q.write_json(OUT/"status.json", dict(status="running", completed=len(manifest["jobs"]), total=96, current=item))
                print(f"START {item['schedule_index']+1}/96 {item['task_id']} seed={item['solver_seed']}", flush=True)
                result = q.execution.supervise_episode(spec, authorized=True, child_entry=q.child)
                if result["error"] or result["status"]=="external_timeout":
                    raise ValueError("episode error/outer fuse; inspect before resuming")
                row = inspect(r, item, a, spec)
                q.audit_trace(spec)
                files = {p.relative_to(folder).as_posix(): q.sha256_file(p) for p in folder.rglob("*.json*")}
                manifest["jobs"][item["job_id"]] = dict(status=row["status"], success=row["success"], files=files)
                q.write_json(mp, manifest)
                print(f"COMPLETE {len(manifest['jobs'])}/96 {row['status']} ttf={row['ttf_seconds']}", flush=True)
            q.write_json(OUT/"status.json", dict(status="complete", completed=96, total=96))
            return dict(complete=True, episodes=96)
        except BaseException:
            q.write_json(OUT/"status.json", dict(status="failed_or_interrupted", completed=len(manifest["jobs"]), total=96))
            raise


def analyze():
    r = ready(False)
    a = old_anchors(r)
    m = q.read_json(OUT/"manifest.json")
    if m["binding"] != r["fingerprint"] or set(m["jobs"]) != {i["job_id"] for i in r["schedule"]}:
        raise ValueError("results incomplete")
    rows = []
    for item in r["schedule"]:
        spec = spec_for(r, item, a)
        for name, digest in m["jobs"][item["job_id"]]["files"].items():
            if q.sha256_file(contained(Path(spec["output"]), name)) != digest:
                raise ValueError("result artifact changed")
        row = inspect(r, item, a, spec)
        row.update(design_density=spec["case"]["pressure_design"]["density"],
                   od_mode=spec["case"]["pressure_design"]["mode"]["name"],
                   source_batch="supplement", workload=workload(Path(spec["output"])))
        rows.append(row)
    baseline = [deepcopy(x) for x in q.read_json(OLD/"analysis/report.json")["episodes"] if x["controller"]=="dual16_sa"]
    keys = lambda rs: {(x["task_id"], x["solver_seed"]): x["initial_fingerprint"] for x in rs}
    if len(baseline)!=96 or keys(baseline)!=keys(rows):
        raise ValueError("cross-batch pair/fingerprint mismatch")
    for row in baseline:
        row.update(source_batch="historical", workload=workload(OLD/"episodes"/row["job_id"]))
    rows += baseline
    kw = dict(controllers=("official_sa", "dual16_sa"), comparisons=(("official_sa", "dual16_sa"),))
    report = dict(schema="lns2.official_sa_pressure_comparison.v1", comparison=r["comparison"],
                  promotion_allowed=False, statistics=q.summarize(rows, **kw), episodes=rows,
                  density_strata={str(d):q.summarize([x for x in rows if x["design_density"]==d], **kw)
                                  for d in (.15,.2,.25)})
    q.publish_analysis(OUT/"analysis", rows, report)
    lines = ["# Official+SA 高压力补充对照", "",
             "这是跨批次探索性比较，不是同期串行计时确认。不得自动晋级或声称因果加速。",
             "Official+SA为本轮96项；Dual16+SA为历史96项。失败不填零，原始TTF仅在共同成功项配对。", "",
             "|分层|共同成功|Official+SA平均TTF/秒|Dual16+SA平均TTF/秒|", "|---|---:|---:|---:|"]
    for c in report["statistics"]["comparisons"]:
        metric = c["metrics"]["ttf_seconds"]
        values = (f"{metric['baseline_mean']:.4f}", f"{metric['challenger_mean']:.4f}") if metric else ("NA", "NA")
        lines.append(f"|{c['group']}|{c['common_success_pairs']}|{values[0]}|{values[1]}|")
    lines += ["", "全组成功数、失败列表、地图bootstrap、路径质量、PP工作量见report.json；环境/批次差异不能由bootstrap消除。"]
    atomic_write_text(OUT/"analysis/report_zh.md", "\n".join(lines)+"\n")
    return dict(complete=True, new_episodes=96, historical_episodes=96, comparison=r["comparison"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "validate", "smoke", "ready", "collect", "resume", "stop", "analyze"))
    parser.add_argument("--workers", type=int, default=20, choices=range(1,21))
    parser.add_argument("--authorize-timing", action="store_true")
    parser.add_argument("--clear-stop", action="store_true")
    args = parser.parse_args()
    if args.phase in {"collect", "resume"}:
        result = collect(authorized=args.authorize_timing, resume=args.phase=="resume", clear_stop=args.clear_stop)
    elif args.phase == "stop":
        q.write_json(OUT/"STOP_AFTER_EPISODE.json", dict(requested=True))
        result = dict(stop_after_episode=True)
    elif args.phase == "validate":
        result = validate(args.workers)
    elif args.phase == "verify":
        result = dict(verified=bool(verify(True, evidence=True)))
    elif args.phase == "ready":
        r = ready()
        result = dict(ready=True, binding=r["fingerprint"], episodes=96, timed_workers=1,
                      timing_authorized=False, formal_started=(OUT/"manifest.json").exists(), comparison=r["comparison"])
        q.write_json(OUT/"readiness.json", result)
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
