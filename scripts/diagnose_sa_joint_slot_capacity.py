"""One bounded coverage follow-up on previously unproved saved states."""
from pathlib import Path
import sys
import time
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import validate_sa_certificate_generalization as g
from experiments.temporal_slot_scan import scan
from experiments.joint_slot_capacity import scan_capacity
from experiments.diagnostic_integrity import snapshot_sources,read_bound_plan

io=g.io
OUT=ROOT/"build/sa-joint-slot-capacity-v1"


def worker(job):
    case=g.load_case(job["source"])
    started=time.monotonic()
    pairs=scan(case["state"],case["selected"],max_pairs=len(case["state"]["conflict_edges"]))
    joint=scan_capacity(case["state"],case["selected"])
    return dict(id=job["id"],status="ok",cohort=case["origin"]["cohort"],all_pairs=pairs,
                joint=joint,diagnostic_seconds=time.monotonic()-started)


def main():
    io.require(not (OUT/"plan.json").exists(),"preserve previous joint-capacity run")
    rows=[r for stage in ("scan","post") for r in g.read_stage(stage) if r.get("case_path") and r["certificate_status"]=="not_proved"]
    jobs=[dict(id=r["id"],job_id=r["id"],source=r) for r in rows]
    names=["scripts/diagnose_sa_joint_slot_capacity.py","experiments/joint_slot_capacity.py",
        "experiments/temporal_slot_scan.py","experiments/local_path_search.py",
        "scripts/validate_sa_certificate_generalization.py","docs/SA_JOINT_SLOT_CAPACITY_PROTOCOL_ZH.md"]
    names += [r["case_path"] for r in rows]
    plan=dict(schema="lns2.sa_joint_slot_capacity.v1",inputs={n:io.sha256_file(ROOT/n) for n in names},jobs=jobs,
              workers=20,horizon=512,work=10000000,no_solver=True,no_ttf=True)
    plan["binding"]=io.semantic_fingerprint(plan)
    snapshot_sources(ROOT,plan["inputs"],OUT/"registered_sources")
    io.write(OUT/"plan.json",plan)
    def failed(job,status,error): return dict(id=job["id"],status=status,error=error)
    def save(row):
        row["binding"]=plan["binding"]
        io.write(OUT/"rows"/(row["id"]+".json"),row)
        print(row["id"],row["status"],flush=True)
    with g.q._CollectionRunLock(OUT,plan["binding"],"joint-capacity"):
        results=g.q._run_jobs(worker,jobs,workers=20,phase="joint-capacity",output_root=OUT/"progress",
            run_fingerprint=plan["binding"],timeout_seconds=180.,failure_result=failed,on_result=save,stop_on_failure=True)
    io.require(len(results)==len(jobs) and all(r["status"]=="ok" for r in results),"incomplete capacity follow-up")
    read_bound_plan(ROOT,OUT)
    io.write(OUT/"report.json",dict(binding=plan["binding"],rows=results,no_ttf=True,controller_promoted=False,
        files={j["id"]:io.sha256_file(OUT/"rows"/(j["id"]+".json")) for j in jobs}))
    print(dict(jobs=len(jobs),all_pair_status=dict(Counter(r["all_pairs"]["status"] for r in results)),
               joint_status=dict(Counter(r["joint"]["status"] for r in results))))


if __name__=="__main__": main()
