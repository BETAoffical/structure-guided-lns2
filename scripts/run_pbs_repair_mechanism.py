"""Fixed, isolated PBS correctness and same-neighborhood mechanism rounds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_pbs_repair as old

OUT = ROOT / "build/pbs-repair-mechanism-v1"
BUILDS = {"admission": "build/linux/pbs-fixed-asan-v1", "mechanism": "build/linux/pbs-mechanism-v1"}


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
    binary,=(ROOT/BUILDS[stage]).glob("lns2_env*.so")
    jobs=[]
    for case in selected:
        for algorithm in ("PP","PBS"):
            for trial in range(2 if stage=="admission" else 4):
                seed=case["seed"] if stage=="admission" else int(old.semantic_fingerprint([case["id"],20260914,trial])[:7],16)
                jobs.append(dict(id=f"{case['id']}-{algorithm}-{trial}",case_id=case["id"],algorithm=algorithm,
                    trial=trial,seed=seed,seconds=5.,node_limit=len(case["agents"]),warm_root=False))
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


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=tuple(BUILDS))
    parser.add_argument("--worker")
    args=parser.parse_args()
    worker(OUT/args.stage,args.worker) if args.worker else run(args.stage)
