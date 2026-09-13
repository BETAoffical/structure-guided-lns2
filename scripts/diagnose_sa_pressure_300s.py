"""Fixed 300-second diagnostic on the union of Official/SA 120s failures."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_pressure_first_feasible as p

q = p.q
OUT = ROOT / "build/sa-pressure-300s-diagnostic-v1"
SOURCE = "build/sa-pressure-first-feasible-v1/analysis/report.json"
SOURCE_SHA = "16a16a350e1ffa61e384ab5ba6a6f24f896c5e73d4382a34905c5226238e611a"
CONTROLLERS = ("official_adaptive", "dual16_sa")
COMPARISONS = (("official_adaptive", "dual16_sa"),)
REGISTERED = ("scripts/diagnose_sa_pressure_300s.py", "tests/evaluation/test_sa_pressure_300s.py",
              "docs/SA_PRESSURE_300S_PROTOCOL_ZH.md")


def failure_union(report):
    return sorted({(x["task_id"], x["solver_seed"]) for x in report["episodes"]
                   if x["controller"] in CONTROLLERS and not x["success"]})


def schedule(old, report):
    rows = []
    for index, (task, seed) in enumerate(failure_union(report)):
        order = CONTROLLERS if index % 2 == 0 else tuple(reversed(CONTROLLERS))
        for position, controller in enumerate(order):
            original = next(x for x in old["schedule"] if
                            (x["task_id"],x["solver_seed"],x["controller"]) == (task,seed,controller))
            item = dict(original, budget_seconds=300., schedule_index=len(rows), within_pair_position=position)
            item["job_id"] = q.json_fingerprint({k:item[k] for k in
                ("task_id","solver_seed","controller","protocol","budget_seconds")})[:24]
            rows.append(item)
    return rows


def prepare():
    if (OUT/"registration.json").exists():
        raise ValueError("registration exists")
    old = p.verify()
    if q.sha256_file(ROOT/SOURCE) != SOURCE_SHA:
        raise ValueError("source result changed")
    report = q.read_json(ROOT/SOURCE)
    if not report["complete"] or report["scheduled"] != 288 or len(failure_union(report)) != 11:
        raise ValueError("expected complete source and 11-condition union")
    inputs = dict(old["inputs"], **{SOURCE:SOURCE_SHA})
    source_manifest = p.OUT/"manifest.json"
    inputs[source_manifest.relative_to(ROOT).as_posix()] = q.sha256_file(source_manifest)
    manifest = q.read_json(source_manifest)
    history_inputs = {}
    source_rows = [x for x in report["episodes"] if x["controller"] in CONTROLLERS and
                   (x["task_id"],x["solver_seed"]) in failure_union(report)]
    for x in source_rows:
        for name, digest in manifest["jobs"][x["job_id"]]["files"].items():
            path = p.OUT/"episodes"/x["job_id"]/name
            if q.sha256_file(path) != digest:
                raise ValueError("historical episode changed")
            history_inputs[path.relative_to(ROOT).as_posix()] = digest
    for name in REGISTERED:
        inputs[name] = q.sha256_file(ROOT/name)
    r = dict(schema="lns2.sa_pressure_300s.v1", cases=old["cases"], template=old["template"],
             source_rows=source_rows, schedule=schedule(old, report), inputs=inputs, history_inputs=history_inputs,
             controllers=list(CONTROLLERS), budget_seconds=300., workers=1,
             native_sha256=q.NATIVE_SHA, native_file=q.NATIVE, promotion_allowed=False,
             source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    r["fingerprint"] = q.json_fingerprint(r)
    q.write_json(OUT/"registration.json",r)
    return dict(prepared=True, conditions=11, episodes=22, budget_seconds=300)


def verify(native=False):
    r = q.read_json(OUT/"registration.json")
    if q.json_fingerprint({k:v for k,v in r.items() if k!="fingerprint"}) != r["fingerprint"]:
        raise ValueError("registration changed")
    if (r["controllers"],r["budget_seconds"],r["workers"],r["native_sha256"],r["native_file"]) != (
            list(CONTROLLERS),300.,1,q.NATIVE_SHA,q.NATIVE):
        raise ValueError("protocol changed")
    if len(r["schedule"])!=22 or r["schedule"] != schedule(p.q.read_json(p.OUT/"registration.json"),q.read_json(ROOT/SOURCE)):
        raise ValueError("selected cohort changed")
    for name,digest in {**r["inputs"],**r["history_inputs"]}.items():
        if q.sha256_file(ROOT/name)!=digest:
            raise ValueError("registered input changed: "+name)
    if native and q.native_identity()["sha256"]!=q.NATIVE_SHA:
        raise ValueError("native changed")
    return r


def reset_worker(job):
    if q.native_identity()["sha256"]!=q.NATIVE_SHA:
        raise ValueError("incorrect reset native")
    c=job["case"]
    j=q.worker_job(c,job["item"],job["template"],OUT/"admission",job["binding"])
    env=q._make_environment(j["dataset_root"],j["row"],j["environment"],"Adaptive")
    state=q._plain(env.reset(seed=j["solver_seed"]))
    if not state["initial_solution_complete"] or q.state_fingerprint(state)!=job["expected"]:
        raise ValueError("300s reset differs from historical initial state")
    journal=q.execution.PathJournal(OUT/"admission",job["binding"],ROOT/c["files"]["map_file"],ROOT/c["files"]["scenario_file"])
    return dict(status="ok",job_id=job["job_id"],state_fingerprint=q.state_fingerprint(state),
                initial_quality=journal.quality(state),binding=job["binding"])


def validate():
    r=verify(True)
    with q._CollectionRunLock(OUT,r["fingerprint"],"300s-admission"):
        if (OUT/"admission_started.json").exists():
            raise ValueError("admission attempt exists")
        jobs=[]
        for i in r["schedule"]:
            if i["controller"]!="official_adaptive": continue
            old=next(x for x in r["source_rows"] if (x["task_id"],x["solver_seed"],x["controller"]) ==
                     (i["task_id"],i["solver_seed"],i["controller"]))
            jobs.append(dict(job_id=q.admission_key(i),item=i,template=r["template"],binding=r["fingerprint"],
                        case=next(c for c in r["cases"] if c["task_id"]==i["task_id"]),expected=old["initial_fingerprint"]))
        q.write_json(OUT/"admission_started.json",dict(jobs=len(jobs)))
        def failed(job,status,error):return dict(status=status,job_id=job["job_id"],error=str(error))
        rows=q._run_jobs(reset_worker,jobs,workers=4,phase="300s-reset",output_root=OUT/"admission-progress",
            run_fingerprint=r["fingerprint"],timeout_seconds=420,failure_result=failed,stop_on_failure=True,
            on_result=lambda row:q.write_json(OUT/"admission"/(row["job_id"]+".json"),row))
        a=dict(passed=len(rows)==11 and all(x["status"]=="ok" for x in rows),binding=r["fingerprint"],
               files={x["job_id"]:q.sha256_file(OUT/"admission"/(x["job_id"]+".json")) for x in rows})
        q.write_json(OUT/"admission.json",a)
        return dict(passed=a["passed"],resets=len(rows))


def anchors(r):
    a=q.read_json(OUT/"admission.json")
    if not a["passed"] or a["binding"]!=r["fingerprint"] or set(a["files"])!={q.admission_key(i) for i in r["schedule"]}:
        raise ValueError("incomplete admission")
    result={}
    for key,digest in a["files"].items():
        path=OUT/"admission"/(key+".json")
        if q.sha256_file(path)!=digest:raise ValueError("anchor changed")
        result[key]=q.read_json(path)
    return result


def spec_for(r,item,a):
    return q.spec_for(r,item,a[q.admission_key(item)],OUT/"episodes"/item["job_id"])


def inspect(r,item,a):
    spec=spec_for(r,item,a)
    return q.inspect_episode(ROOT,spec["case"],item,Path(spec["output"]),spec["binding"],a[q.admission_key(item)])


def collect(resume=False):
    r=verify(True)
    a=anchors(r)
    with q._CollectionRunLock(OUT,r["fingerprint"],"300s-diagnostic"):
        mp=OUT/"manifest.json"
        if mp.exists() and not resume:raise ValueError("explicit resume required")
        m=q.read_json(mp) if mp.exists() else dict(binding=r["fingerprint"],jobs={})
        if m["binding"]!=r["fingerprint"] or not set(m["jobs"]).issubset({i["job_id"] for i in r["schedule"]}):
            raise ValueError("manifest changed")
        q.write_json(OUT/("environment-"+str(len(m["jobs"]))+".json"),q.runtime_environment())
        try:
            for i in r["schedule"]:
                folder=OUT/"episodes"/i["job_id"]
                if i["job_id"] in m["jobs"]:
                    for name,digest in m["jobs"][i["job_id"]]["files"].items():
                        if q.sha256_file(folder/name)!=digest:raise ValueError("result changed")
                    continue
                if (OUT/"STOP_AFTER_EPISODE.json").exists():
                    q.write_json(OUT/"status.json",dict(status="stopped",completed=len(m["jobs"]),total=22))
                    return dict(stopped=True,completed=len(m["jobs"]))
                if folder.exists():raise ValueError("interrupted episode; audit before retry")
                q.write_json(OUT/"status.json",dict(status="running",completed=len(m["jobs"]),total=22,current=i))
                print(f"START {i['schedule_index']+1}/22 {i['controller']} {i['task_id']} seed={i['solver_seed']}",flush=True)
                spec=spec_for(r,i,a)
                result=q.execution.supervise_episode(spec,authorized=True,child_entry=q.child)
                if result["error"]:raise ValueError("episode error; retained for inspection")
                row=inspect(r,i,a)
                q.audit_trace(spec)
                m["jobs"][i["job_id"]]=dict(status=row["status"],success=row["success"],files={
                    path.relative_to(folder).as_posix():q.sha256_file(path) for path in folder.rglob("*.json*") if path.is_file()})
                q.write_json(mp,m)
                print(f"COMPLETE {len(m['jobs'])}/22 {row['status']} ttf={row['ttf_seconds']}",flush=True)
            q.write_json(OUT/"status.json",dict(status="complete",completed=22,total=22))
            return dict(complete=True,episodes=22)
        except BaseException:
            q.write_json(OUT/"status.json",dict(status="failed_or_interrupted",completed=len(m["jobs"]),total=22))
            raise


def scientific_event(event):
    delta=dict(event["delta"])
    delta["top_set"]={k:v for k,v in delta["top_set"].items() if k!="runtime"}
    selected=event["selected_index"]
    return dict(action=event["action"],delta=delta,repair_order=event["metrics"]["repair_order"],
                temperature=event["temperature"],uniform=event["uniform"],
                selected=None if selected is None else event["pool"][selected]["candidate_id"])


def trajectory_summary(folder,initial_conflicts):
    seq=[(0.,initial_conflicts)]
    events=[]
    with (folder/"first_phase/trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            e=json.loads(line)
            seq.append((e["elapsed_seconds"],e["metrics"]["conflicts_after"]))
            events.append((q.json_fingerprint(scientific_event(e)),e["metrics"]["pp_failure_reason"]))
    return seq,events


def analyze():
    r=verify()
    a=anchors(r)
    m=q.read_json(OUT/"manifest.json")
    if m["binding"]!=r["fingerprint"] or set(m["jobs"])!={i["job_id"] for i in r["schedule"]}:
        raise ValueError("incomplete results")
    rows=[]
    diagnostics=[]
    for i in r["schedule"]:
        folder=OUT/"episodes"/i["job_id"]
        for name,digest in m["jobs"][i["job_id"]]["files"].items():
            if q.sha256_file(folder/name)!=digest:raise ValueError("result changed")
        row=inspect(r,i,a)
        rows.append(row)
        old=next(x for x in r["source_rows"] if (x["task_id"],x["solver_seed"],x["controller"]) ==
                 (i["task_id"],i["solver_seed"],i["controller"]))
        seq,events=trajectory_summary(folder,row["initial_conflicts"])
        _,old_events=trajectory_summary(p.OUT/"episodes"/old["job_id"],old["initial_conflicts"])
        safe_old=old_events[:-1] if old_events and old_events[-1][1]=="time_limit" else old_events
        matched=0
        for left,right in zip(safe_old,events):
            if left[0]!=right[0]:break
            matched+=1
        best=min(c for _,c in seq)
        diagnostics.append(dict(job_id=i["job_id"],task_id=i["task_id"],solver_seed=i["solver_seed"],controller=i["controller"],
            old_success=old["success"],old_ttf_seconds=old["ttf_seconds"],new_success=row["success"],new_ttf_seconds=row["ttf_seconds"],
            matched_scientific_prefix_steps=matched,old_non_timeout_steps=len(safe_old),
            prefix_equal=matched==len(safe_old),best_conflicts=best,first_best_seconds=next(t for t,c in seq if c==best),
            conflicts_by_cutoff={str(cut):next(c for t,c in reversed(seq) if t<=cut) for cut in [60,120,180,240,300]}))
    report=dict(schema="lns2.sa_pressure_300s_report.v1",complete=True,scheduled=22,binding=r["fingerprint"],
                promotion_allowed=False,statistics=q.summarize(rows,controllers=CONTROLLERS,comparisons=COMPARISONS),
                diagnostics=diagnostics,episodes=rows)
    q.publish_analysis(OUT/"analysis",rows,report)
    return dict(complete=True,successes={c:sum(x["success"] for x in rows if x["controller"]==c) for c in CONTROLLERS})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","validate","run","resume","stop","analyze"))
    args=parser.parse_args()
    if args.phase=="stop":
        q.write_json(OUT/"STOP_AFTER_EPISODE.json",dict(requested=True)); result=dict(stop_after_episode=True)
    elif args.phase=="verify":result=dict(verified=bool(verify(True)))
    elif args.phase in {"run","resume"}:
        result=collect(args.phase=="resume")
        if result.get("complete"):result=analyze()
    else:result=globals()[args.phase]()
    print(json.dumps(result,indent=2))
