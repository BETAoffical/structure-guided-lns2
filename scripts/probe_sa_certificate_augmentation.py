"""Bounded, certificate-derived membership interventions; no production changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import diagnose_sa_pair_compatibility as source
from scripts.diagnose_sa_external_obstruction import reachable_prefix
from experiments.pair_compatibility import diagnose
from experiments import local_path_search as ref
from experiments.diagnostic_integrity import pair_absent,verify_inputs,snapshot_sources

io=source.io
OUT=ROOT/"build/sa-certificate-augmentation-v1"


def candidates(case, blockers):
    selected=set(case["selected"])
    agents={a["id"] for a in case["state"]["agents"]}
    blockers=sorted(set(blockers))
    io.require(blockers and len(blockers)<=8 and not set(blockers)&selected and set(blockers)<=agents,"invalid boundary")
    eligible=sorted(agents-selected-set(blockers),key=lambda a:io.semantic_fingerprint([case["id"],20260914,"control",a]))
    io.require(len(eligible)>=len(blockers),"no matched controls")
    entries=[("baseline",[],"baseline")]+[(f"add-{a}",[a],"certificate_single") for a in blockers]
    entries += [("boundary_union",blockers,"certificate_union"),("control_one",eligible[:1],"control"),
                ("control_union",eligible[:len(blockers)],"control")]
    return [dict(id=name,agents=sorted(selected|set(extra)),added=extra,kind=kind) for name,extra,kind in entries]


def prepare(certificate_file="build/sa-external-obstruction-v1/report.json"):
    io.require(not (OUT/"plan.json").exists(),"plan exists")
    old=source.verify(evidence_only=True)
    certificate_path=ROOT/certificate_file
    certificate=io.read(certificate_path)
    for name,sha in certificate["inputs"].items():
        io.require(io.sha256_file(ROOT/name)==sha,"certificate input changed")
    plateau=io.read(ROOT/"build/sa-plateau-candidate-audit-v1/plan.json")
    cases=[]
    for proof in certificate["proofs"]:
        if proof["status"] not in ("forced_prefix_collision","forced_goal_slot"):
            continue
        case=next(c for c in old["cases"] if c["id"]==proof["job"]["case_id"])
        graph,agents=ref.validate_state(case["state"])
        fixed={aid:a["path"] for aid,a in agents.items() if aid not in case["selected"]}
        pair=proof["job"]["pair"]
        if proof["status"]=="forced_prefix_collision":
            prefixes=[reachable_prefix(graph,agents[a]["start"],fixed,proof["event"]["time"]) for a in pair]
            io.require(prefixes==proof["prefixes"],"certificate reconstruction mismatch")
        else:
            from experiments.goal_slot_certificate import prove_goal_slot
            io.require(prove_goal_slot(case["state"],case["selected"],pair,proof["event"]["time"])==proof["certificate"],"goal certificate reconstruction mismatch")
        target=next(t for t in plateau["targets"] if t["job_id"]==case["id"])
        cases.append(dict(**case,pair=pair,candidates=candidates(case,proof["blocker_candidates"]),
                          temperature=target["temperature"]))
    io.require(bool(cases),"no certified obstruction")
    inputs={name:io.sha256_file(ROOT/name) for name in old["inputs"]}
    inputs[certificate_path.relative_to(ROOT).as_posix()]=io.sha256_file(certificate_path)
    for name in ("scripts/probe_sa_certificate_augmentation.py","scripts/diagnose_sa_external_obstruction.py",
                 "tests/evaluation/test_sa_external_obstruction.py","docs/SA_CERTIFICATE_AUGMENTATION_PROTOCOL_ZH.md",
                 "build/sa-plateau-candidate-audit-v1/plan.json","scripts/probe_sa_rejection_branches.py"):
        inputs[name]=io.sha256_file(ROOT/name)
    for name in ("experiments/diagnostic_integrity.py","experiments/goal_slot_certificate.py",
                 "scripts/diagnose_sa_pair_compatibility.py","build/sa-pair-compatibility-v1/plan.json"):
        inputs[name]=io.sha256_file(ROOT/name)
    if any(p["status"]=="forced_goal_slot" for p in certificate["proofs"]):
        name="docs/SA_GOAL_CERTIFICATE_AUGMENTATION_PROTOCOL_ZH.md"
        inputs[name]=io.sha256_file(ROOT/name)
    plan=dict(schema="lns2.sa_certificate_augmentation.v1",cases=cases,inputs=inputs,workers=20,
              reference_seconds=45.,reference_nodes=250000,pp_seconds=5.,fuse=90,trials=4,no_ttf=True)
    plan["binding"]=io.semantic_fingerprint(plan)
    snapshot_sources(ROOT,inputs,OUT/"registered_sources")
    io.write(OUT/"plan.json",plan)
    print(dict(cases=len(cases),reference_jobs=sum(len(c["candidates"]) for c in cases),
               native_jobs=sum(len(c["candidates"])*4 for c in cases),workers=20))


def verify(evidence_only=False):
    plan=io.read(OUT/"plan.json")
    io.require(plan["binding"]==io.semantic_fingerprint({k:v for k,v in plan.items() if k!="binding"}),"plan changed")
    verify_inputs(ROOT,plan["inputs"],OUT/"registered_sources" if evidence_only else None)
    return plan


def jobs(plan,stage):
    return [dict(id=f"{c['id']}-{a['id']}-{trial}",case_id=c["id"],candidate=a,trial=trial)
            for c in plan["cases"] for a in c["candidates"] for trial in (range(4) if stage=="native" else [-1])]


def worker(stage,job_id):
    plan=verify()
    job=next(j for j in jobs(plan,stage) if j["id"]==job_id)
    case=next(c for c in plan["cases"] if c["id"]==job["case_id"])
    members=job["candidate"]["agents"]
    if stage=="reference":
        result=diagnose(case["state"],members,case["pair"],True,"astar",plan["reference_seconds"],plan["reference_nodes"])
    else:
        binary=ROOT/io.NATIVE
        io.require(io.sha256_file(binary)==io.NATIVE_SHA,"native changed")
        sys.path.insert(0,str(binary.parent))
        import lns2_env
        from scripts import run_sa_wall_clock as sa
        io.require(Path(lns2_env.__file__).resolve()==binary.resolve(),"wrong native")
        agents=sorted(case["state"]["agents"],key=lambda a:a["id"])
        env=lns2_env.LNS2RepairEnv(str(ROOT/case["files"]["map_file"]),str(ROOT/case["files"]["scenario_file"]),
                                  len(agents),time_limit=60,use_sipp=True)
        before=env.reset_paths([a["path"] for a in agents],seed=17)
        io.require(io.repair_structure_fingerprint(before)==io.repair_structure_fingerprint(case["state"]),"restore mismatch")
        seed=int(io.semantic_fingerprint([case["id"],20260914,job["trial"]])[:7],16)
        uniform=random.Random(seed).random()
        raw=env.step_experimental_pp(dict(mode="explicit_neighborhood",agents=members,random_seed=seed),
                                     plan["pp_seconds"],"annealed",case["temperature"],uniform)
        after,metrics=raw["observation"],raw["metrics"]
        io.require(metrics["action_valid"] and sorted(metrics["neighborhood"])==members,"invalid explicit action")
        io.require([a["path"] for a in before["agents"] if a["id"] not in members]==
                   [a["path"] for a in after["agents"] if a["id"] not in members],"external paths changed")
        ref.validate_state(after)
        # Reuse frozen SA acceptance validation, without altering the runtime.
        sa.validate_transition(before,after,metrics,members,"annealed",case["temperature"],uniform)
        result=dict(before=before["num_of_colliding_pairs"],after=after["num_of_colliding_pairs"],
                    feasible=after["feasible"],strict_drop=after["num_of_colliding_pairs"]<before["num_of_colliding_pairs"],
                    target_pair_cleared=pair_absent(after["conflict_edges"],case["pair"]),metrics=metrics,final_state=after,
                    censored=metrics["pp_failure_reason"]=="time_limit",
                    generated=after["low_level"]["generated"]-before["low_level"]["generated"])
    io.write(OUT/stage/"jobs"/(job_id+".json"),dict(binding=plan["binding"],job=job,stage=stage,status="ok",result=result))


def read_stage(plan,stage):
    manifest=io.read(OUT/stage/"manifest.json")
    schedule=jobs(plan,stage)
    io.require(manifest["binding"]==plan["binding"] and set(manifest["files"])=={j["id"] for j in schedule},"manifest mismatch")
    rows=[]
    for job in schedule:
        path=OUT/stage/"jobs"/(job["id"]+".json")
        io.require(io.sha256_file(path)==manifest["files"][job["id"]],"result changed")
        row=io.read(path)
        io.require(row["job"]==job and row["binding"]==plan["binding"] and row["stage"]==stage and row["status"]=="ok","invalid row")
        rows.append(row)
    return rows


def collect(stage):
    plan=verify()
    folder=OUT/stage
    io.require(not folder.exists(),"preserve previous stage")
    folder.mkdir(parents=True)
    schedule=jobs(plan,stage)
    def execute(job):
        path=folder/(job["id"]+".log")
        with path.open("w") as log:
            proc=subprocess.Popen([sys.executable,"-B",str(Path(__file__)),stage,"--worker",job["id"],
                                   "--output",OUT.relative_to(ROOT).as_posix()],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
            try:
                code=proc.wait(timeout=plan["fuse"])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
                code="external_timeout"
        io.require(code==0,f"worker error {job['id']}: {code}")
    with ThreadPoolExecutor(max_workers=min(plan["workers"],len(schedule))) as pool:
        list(pool.map(execute,schedule))
    io.write(folder/"manifest.json",dict(binding=plan["binding"],files={j["id"]:
             io.sha256_file(folder/"jobs"/(j["id"]+".json")) for j in schedule}))
    print(dict(stage=stage,jobs=len(read_stage(plan,stage)),errors=0))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","reference","native"))
    parser.add_argument("--worker")
    parser.add_argument("--output",default="build/sa-certificate-augmentation-v1")
    parser.add_argument("--certificate",default="build/sa-external-obstruction-v1/report.json")
    args=parser.parse_args()
    OUT=ROOT/args.output
    io.require(OUT.resolve().is_relative_to((ROOT/"build").resolve()),"output must remain inside build")
    if args.worker:
        worker(args.stage,args.worker)
    elif args.stage=="prepare":
        prepare(args.certificate)
    else:
        collect(args.stage)
