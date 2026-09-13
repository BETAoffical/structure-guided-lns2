"""Fixed, isolated PBS correctness and same-neighborhood mechanism rounds."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_pbs_repair as old

OUT = ROOT / "build/pbs-repair-mechanism-v1"
BUILDS = {"admission": "build/linux/pbs-fixed-asan-v1", "mechanism": "build/linux/pbs-mechanism-v1"}
BUILDS.update(extended=BUILDS["mechanism"], tail=BUILDS["mechanism"], tail_extended=BUILDS["mechanism"])


def stage_report(stage):
    folder=OUT/stage
    report=old.read(folder/"report.json")
    old.require(report["errors"]==0,"previous stage has errors")
    for name,sha in report["files"].items():
        old.require(old.sha256_file(folder/name)==sha,"previous result changed")
    return report


def extended_cases(selected, source="mechanism"):
    report=stage_report(source)
    names=[]
    for case in selected:
        rows=[r for r in report["rows"] if r["job"]["case_id"]==case["id"] and r["job"]["algorithm"]=="PBS"]
        metrics=[old.read(OUT/source/"jobs"/(r["job"]["id"]+".json"))["metrics"] for r in rows]
        if rows and all(not r["summary"]["strict_drop"] for r in rows) and any(m["pbs_stop_reason"]=="node_limit" for m in metrics):
            names.append(case["id"])
    old.require(bool(names),"no node-limit iteration admitted")
    return [c for c in selected if c["id"] in names]


def tail_cases():
    from experiments._common import read_jsonl
    from experiments.closed_loop_trace_storage import apply_state_delta
    from experiments.repair_collection import state_fingerprint
    plan_path=ROOT/"build/sa-plateau-candidate-audit-v1/plan.json"
    plan=old.read(plan_path)
    old.require(plan["fingerprint"]==old.semantic_fingerprint({k:v for k,v in plan.items() if k!="fingerprint"}),"tail plan changed")
    source=ROOT/"build/sa-pressure-300s-diagnostic-v1"
    for name,key in (("registration.json","source_registration_sha256"),("manifest.json","source_manifest_sha256"),("analysis/report.json","source_report_sha256")):
        old.require(old.sha256_file(source/name)==plan[key],"tail source changed")
    registration=old.read(source/"registration.json")
    manifest=old.read(source/"manifest.json")
    result=[]
    for target in plan["targets"]:
        folder=source/"episodes"/target["job_id"]
        source_files=manifest["jobs"][target["job_id"]]["files"]
        for name,sha in source_files.items():
            old.require(old.sha256_file(folder/name)==sha,"tail episode changed")
        initial=old.read(folder/"initial.json")
        state=initial["observation"] if "observation" in initial else initial["payload"]["observation"]
        events=list(read_jsonl(folder/"first_phase/trace.jsonl"))
        for event in events[:target["decision"]]: state=apply_state_delta(state,event["delta"])
        old.require(state_fingerprint(state)==target["state_fingerprint"],"tail reconstruction mismatch")
        case=next(c for c in registration["cases"] if c["task_id"]==target["item"]["task_id"])
        files={k:case["files"][k] for k in ("map_file","scenario_file")}
        for name in files.values():
            old.require(old.sha256_file(ROOT/name)==registration["inputs"][name],"tail task changed")
        candidate=target["pool"][target["selected_index"]]
        result.append(dict(id=target["job_id"],files=files,paths=[a["path"] for a in sorted(state["agents"],key=lambda a:a["id"])],
            agents=candidate["agents"],seed=17,expected_structure=old.repair_structure_fingerprint(state),
            state_fingerprint=target["state_fingerprint"],selected_candidate_id=candidate["candidate_id"],
            source_plan_sha256=old.sha256_file(plan_path)))
    return result


def cases():
    plan = old.read(old.OUT / "plan.json")
    old.verify(plan, full=True)
    return plan["cases"]


def validate_result(before, result, agents):
    after, metrics = result["observation"], result["metrics"]
    old.validate_transition(before, after, metrics, agents)
    members = set(agents)
    outside = {tuple(e) for e in before["conflict_edges"] if not (set(e) & members)}
    outside_after = {tuple(e) for e in after["conflict_edges"] if not (set(e) & members)}
    old.require(outside == outside_after, "external conflicts changed")
    return dict(before=before["num_of_colliding_pairs"], after=after["num_of_colliding_pairs"],
                external_pairs=len(outside), incident_before=before["num_of_colliding_pairs"]-len(outside),
                incident_after=after["num_of_colliding_pairs"]-len(outside), feasible=after["feasible"],
                strict_drop=after["num_of_colliding_pairs"] < before["num_of_colliding_pairs"],
                before_structure=old.repair_structure_fingerprint(before),
                after_structure=old.repair_structure_fingerprint(after),
                nodes={k:after["low_level"][k]-before["low_level"][k] for k in ("generated","expanded","runs")})


def worker(folder, job_id):
    import faulthandler
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
    faulthandler.enable()
    registration = old.read(folder/"registration.json")
    old.require(registration["runner_sha256"] == old.sha256_file(Path(__file__)), "runner changed")
    job = next(j for j in registration["jobs"] if j["id"] == job_id)
    case = next(c for c in registration["cases"] if c["id"] == job["case_id"])
    binary = ROOT/registration["native"]
    old.require(old.sha256_file(binary) == registration["native_sha256"], "native changed")
    sys.path.insert(0,str(binary.parent))
    import lns2_env
    old.require(Path(lns2_env.__file__).resolve() == binary.resolve(), "wrong native loaded")
    old.require(hasattr(lns2_env,"pbs_diagnostic_schema"), "not a diagnostic binary")
    env = lns2_env.LNS2RepairEnv(str(ROOT/case["files"]["map_file"]), str(ROOT/case["files"]["scenario_file"]),
        len(case["paths"]), time_limit=5 if registration["stage"]=="admission" else 60,
        replan_algorithm=job["algorithm"], use_sipp=True)
    before = env.reset_paths(case["paths"], seed=job["seed"])
    if "expected_structure" in case:
        old.require(old.repair_structure_fingerprint(before)==case["expected_structure"],"restored mismatch")
    print("restored; entering repair",flush=True)
    action=dict(mode="explicit_neighborhood", agents=case["agents"], random_seed=job["seed"])
    if registration["stage"] == "admission":
        result=env.step(action)
    elif job["algorithm"] == "PP":
        result=env.step_with_time_limit(action,job["seconds"])
    else:
        result=env.step_diagnostic_pbs(action,job["seconds"],job["node_limit"],job["warm_root"])
    summary=validate_result(before,result,case["agents"])
    old.write(folder/"jobs"/(job_id+".json"),dict(status="ok", binding=registration["binding"],job=job,
        summary=summary,metrics=result["metrics"],final_state=result["observation"]))
    print("valid return",flush=True)


def run(stage):
    folder=OUT/stage
    old.require(not (folder/"registration.json").exists(),"stage exists; preserve prior evidence")
    selected=cases()
    prerequisite = None
    if stage != "admission":
        prerequisite = OUT/"admission/report.json"
        old.require(old.read(prerequisite)["errors"] == 0, "PBS correctness admission failed")
    if stage=="extended": selected=extended_cases(selected)
    if stage=="tail_extended":
        selected=extended_cases(old.read(OUT/"tail/registration.json")["cases"],"tail")
    if stage=="tail":
        previous=stage_report("mechanism")
        old.require(any(r["summary"]["strict_drop"] for r in previous["rows"]
                        if r["job"]["algorithm"]=="PBS" and r["job"]["case_id"] not in ("open","corridor")),"no real-state PBS opportunity")
        selected=tail_cases()
    binary,=(ROOT/BUILDS[stage]).glob("lns2_env*.so")
    jobs=[]
    for case in selected:
        for algorithm in (("PBS",) if stage.endswith("extended") else ("PP","PBS")):
            for trial in range(2 if stage=="admission" else 4):
                seed=case["seed"] if stage=="admission" else int(old.semantic_fingerprint([case["id"],20260914,trial])[:7],16)
                jobs.append(dict(id=f"{case['id']}-{algorithm}-{trial}",case_id=case["id"],algorithm=algorithm,
                    trial=trial,seed=seed,seconds=5.,node_limit=len(case["agents"])*(4 if stage.endswith("extended") else 1),warm_root=False))
    registration=dict(schema="lns2.pbs_mechanism_round.v1",stage=stage,cases=selected,jobs=jobs,
        native=binary.relative_to(ROOT).as_posix(),native_sha256=old.sha256_file(binary),
        source_plan_sha256=old.sha256_file(old.OUT/"plan.json"),runner_sha256=old.sha256_file(Path(__file__)),
        prerequisite_sha256=None if prerequisite is None else old.sha256_file(prerequisite),
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        workers=min(20,len(jobs)),process_fuse_seconds=60 if stage=="admission" else 30,no_ttf=True)
    registration["binding"]=old.semantic_fingerprint(registration)
    old.write(folder/"registration.json",registration)
    env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    if stage=="admission":
        env.update(LD_PRELOAD=subprocess.check_output(["g++","-print-file-name=libasan.so"],text=True).strip(),
                   ASAN_OPTIONS="detect_leaks=0:disable_coredump=1")
    def execute(job):
        log=folder/"logs"/(job["id"]+".log")
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("w") as stream:
            proc=subprocess.Popen([sys.executable,"-B",str(Path(__file__)),stage,"--worker",job["id"]],
                                  stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:
                code=proc.wait(timeout=registration["process_fuse_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
                code="timeout_unknown"
        path=folder/"jobs"/(job["id"]+".json")
        if code==0 and path.exists():
            row=old.read(path)
            old.require(row["binding"]==registration["binding"],"output binding mismatch")
        else:
            row=dict(status="native_or_worker_error",returncode=code,job=job,binding=registration["binding"])
            old.write(path,row)
        return {k:v for k,v in row.items() if k not in ("metrics","final_state")}
    # Only children load native; the thread pool supervises independent processes.
    with ThreadPoolExecutor(max_workers=registration["workers"]) as pool:
        rows=list(pool.map(execute,jobs))
    report=dict(schema="lns2.pbs_mechanism_report.v1",binding=registration["binding"],rows=rows,
                errors=sum(r["status"]!="ok" for r in rows),no_ttf=True,
                files={p.relative_to(folder).as_posix():old.sha256_file(p) for p in sorted((folder/"jobs").glob("*.json"))})
    old.write(folder/"report.json",report)
    print(dict(stage=stage,jobs=len(rows),errors=report["errors"]))


def analyze():
    reports={}
    inputs={}
    for stage in BUILDS:
        folder=OUT/stage
        reg=old.read(folder/"registration.json")
        old.require(reg["binding"]==old.semantic_fingerprint({k:v for k,v in reg.items() if k!="binding"}),"registration changed")
        old.require(old.sha256_file(ROOT/reg["native"])==reg["native_sha256"],"registered binary changed")
        report=stage_report(stage)
        old.require(report["binding"]==reg["binding"],"report binding mismatch")
        groups=defaultdict(list)
        for job in reg["jobs"]:
            row=old.read(folder/"jobs"/(job["id"]+".json"))
            old.require(row["binding"]==reg["binding"] and row["job"]==job,"job identity mismatch")
            groups[(job["case_id"],job["algorithm"])].append(row)
        stats=[]
        for (case,algorithm),rows in groups.items():
            stats.append(dict(case=case,algorithm=algorithm,trials=len(rows),
                strict_drops=sum(r["summary"]["strict_drop"] for r in rows),feasible=sum(r["summary"]["feasible"] for r in rows),
                before=rows[0]["summary"]["before"],external=rows[0]["summary"]["external_pairs"],
                remaining=[r["summary"]["after"] for r in rows],
                mean_drop=mean(r["summary"]["before"]-r["summary"]["after"] for r in rows),
                mean_generated=mean(r["summary"]["nodes"]["generated"] for r in rows),
                root_improvements=sum(r["metrics"].get("pbs_hl_expanded",-1)==0 and r["summary"]["strict_drop"] for r in rows),
                stop_reasons=dict(Counter(r["metrics"].get("pbs_stop_reason",r["metrics"]["pp_failure_reason"]) for r in rows))))
        reports[stage]=dict(jobs=len(reg["jobs"]),errors=report["errors"],groups=stats)
        inputs[stage]=dict(registration=old.sha256_file(folder/"registration.json"),report=old.sha256_file(folder/"report.json"))
    tail_groups=[g for stage in ("tail","tail_extended") for g in reports[stage]["groups"] if g["algorithm"]=="PBS"]
    decision=("SA_tail_local_opportunity_requires_independent_validation" if any(g["strict_drops"] for g in tail_groups)
              else "local_opportunity_but_no_SA_tail_recovery_within_registered_budget")
    result=dict(schema="lns2.pbs_mechanism_summary.v1",stages=reports,inputs=inputs,
        decision=decision,no_ttf=True,
        iteration="single_node_limit_multiplier_4",formal_controller_changed=False,
        caveats=["historical posthoc states", "parallel wall budgets not solver speed evidence",
                 "time_limit does not prove infeasibility", "PP nonincrease versus PBS strict decrease acceptance"])
    old.write(OUT/"summary.json",result)
    print(result)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=(*BUILDS,"analyze"))
    parser.add_argument("--worker")
    args=parser.parse_args()
    if args.worker: worker(OUT/args.stage,args.worker)
    elif args.stage=="analyze": analyze()
    else: run(args.stage)
