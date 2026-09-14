"""Bounded unseen-map and post-intervention certificate verification, never TTF."""
import argparse
from collections import Counter,defaultdict
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import diagnose_pbs_repair as io
from scripts.audit_sa_cooling_traces import q
from experiments.temporal_slot_scan import scan
from experiments.diagnostic_integrity import read_bound_plan,snapshot_sources
from experiments import local_path_search as ref
from experiments.nonmonotonic_repair import validate_transition
from generators.models import MapData,TaskData
from generators.io import write_movingai_map,write_movingai_scen

OUT=ROOT/"build/sa-certificate-generalization-v2"
PARENT="build/sa-training-readiness-v1/plan.json"
OLD="build/sa-certificate-results-v1/report.json"


def map_key(state):
    return io.semantic_fingerprint([state["rows"],state["cols"],state["obstacles"]])


def choose_episodes(jobs,excluded):
    unique={}
    for job in sorted(jobs,key=lambda j:(j["cohort"]!="full_120",j["job_id"])):
        if job["map_hash"] not in excluded:
            unique.setdefault((job["task_id"],job["solver_seed"]),job)
    counts=Counter()
    selected=[]
    for job in sorted(unique.values(),key=lambda j:io.semantic_fingerprint([20260915,j["task_id"],j["solver_seed"]])):
        if len(selected)>=24: break
        if counts[job["map_hash"]]>=3: continue
        selected.append(job)
        counts[job["map_hash"]]+=1
    return selected


def candidate_sets(case,mode):
    selected=set(case["selected"])
    blockers=case["certificate"]["blockers"]
    # An over-wide hint list remains a recorded coverage failure, not a resampled case.
    if not blockers or len(blockers)>8: return []
    others={a["id"] for a in case["state"]["agents"]}-selected-set(blockers)
    control=sorted(others,key=lambda a:io.semantic_fingerprint([case["id"],20260915,a]))[:len(blockers)]
    if len(control)!=len(blockers): return []
    padding=sorted(selected-{a for p in case["state"]["conflict_edges"] for a in p})
    rows=[dict(id="baseline",members=sorted(selected),kind="baseline")]
    changes=[(f"certificate-{a}",[a],"certificate") for a in blockers]
    changes += [("certificate-union",blockers,"certificate_union"),("random-one",control[:1],"control"),("random-union",control,"control_union")]
    for name,extra,kind in changes:
        if mode=="replace" and len(padding)<len(extra): continue
        members=(selected-set(padding[:len(extra)]) if mode=="replace" else selected)|set(extra)
        rows.append(dict(id=name,members=sorted(members),kind=kind))
    return rows


def prepare():
    io.require(not (OUT/"plan.json").exists(),"preserve existing run")
    parent=io.read(ROOT/PARENT)
    io.require(parent["fingerprint"]==q.json_fingerprint({k:v for k,v in parent.items() if k!="fingerprint"}),"parent plan changed")
    old=io.read(ROOT/OLD)
    for name,sha in old["sources"].items():
        io.require(io.sha256_file(ROOT/name)==sha,"previous evidence changed")
    prior=io.read(ROOT/"build/sa-pair-compatibility-v1/plan.json")
    excluded={map_key(c["state"]) for c in prior["cases"]}
    jobs=[]
    for job in parent["jobs"]:
        folder=ROOT/job["folder"]
        io.require(io.sha256_file(folder/"initial.json")==job["files"]["initial.json"],"initial changed")
        state=q.execution.read_artifact(folder/"initial.json",job["binding"])["observation"]
        io.require(q.state_fingerprint(state)==job["initial_fingerprint"],"initial identity changed")
        clean={k:v for k,v in job.items() if k!="success"}
        jobs.append(dict(**clean,map_hash=map_key(state)))
    jobs=choose_episodes(jobs,excluded)
    names=[PARENT,OLD,"build/sa-pair-compatibility-v1/plan.json",io.NATIVE,
        "artifacts/initlns-closed-loop-controller-v2/main__realized_dynamic.json",
        "scripts/validate_sa_certificate_generalization.py","experiments/temporal_slot_scan.py",
        "experiments/diagnostic_integrity.py","experiments/local_path_search.py",
        "experiments/nonmonotonic_repair.py","experiments/repair_collection.py",
        "scripts/diagnose_sa_external_obstruction.py","docs/SA_CERTIFICATE_GENERALIZATION_PROTOCOL_ZH.md"]
    plan=dict(schema="lns2.sa_certificate_generalization.v1",jobs=jobs,excluded_maps=sorted(excluded),
              inputs={n:io.sha256_file(ROOT/n) for n in names},workers=20,trials=4,pp_seconds=5.,fuse=180.,no_ttf=True)
    plan["binding"]=io.semantic_fingerprint(plan)
    snapshot_sources(ROOT,plan["inputs"],OUT/"registered_sources")
    io.write(OUT/"plan.json",plan)
    print(dict(episodes=len(jobs),maps=len({j["map_hash"] for j in jobs}),excluded_maps=len(excluded)))


def materialize(state,selected,case_id,temperature,origin):
    ref.validate_state(state)
    folder=OUT/"cases"/case_id
    io.require(not (folder/"case.json").exists(),"case exists without verified resume")
    started=time.monotonic()
    certificate=scan(state,selected)
    diagnostic_seconds=time.monotonic()-started
    agents=sorted(state["agents"],key=lambda a:a["id"])
    io.require([a["id"] for a in agents]==list(range(len(agents))),"native adapter requires original consecutive IDs")
    n=state["cols"]
    grid=["".join("@" if x else "." for x in state["obstacles"][r*n:(r+1)*n]) for r in range(state["rows"])]
    data=MapData(case_id,0,grid)
    task=TaskData(case_id,case_id,0,[divmod(a["start"],n) for a in agents],[divmod(a["goal"],n) for a in agents])
    write_movingai_map(folder/(case_id+".map"),data)
    write_movingai_scen(folder/(case_id+".scen"),data,task)
    files={suffix:(folder/(case_id+suffix)).relative_to(ROOT).as_posix() for suffix in (".map",".scen")}
    case=dict(id=case_id,state=state,selected=selected,temperature=temperature,origin=origin,map_hash=map_key(state),
              certificate=certificate,diagnostic_seconds=diagnostic_seconds,files=files,
              file_hashes={p:io.sha256_file(ROOT/p) for p in files.values()})
    io.write(folder/"case.json",case)
    return dict(status="ok",id=case_id,case_path=(folder/"case.json").relative_to(ROOT).as_posix(),
                case_sha=io.sha256_file(folder/"case.json"),proofs=len(certificate["proofs"]),
                certificate_status=certificate["status"],blockers=len(certificate["blockers"]))


def scan_episode(job):
    folder=ROOT/job["folder"]
    for name in ("initial.json","first_phase/trace.jsonl"):
        io.require(io.sha256_file(folder/name)==job["files"][name],"trace input changed")
    state=q.execution.read_artifact(folder/"initial.json",job["binding"])["observation"]
    best=state["num_of_colliding_pairs"]
    gap=0
    for d,e in enumerate(q.read_jsonl(folder/"first_phase/trace.jsonl")):
        io.require(e["decision"]==d and not state["feasible"],"invalid source order")
        if d>=64 and gap>=16 and 0<state["num_of_colliding_pairs"]<=32 and len(state["agents"])<=600:
            return materialize(state,e["action"]["agents"],job["id"],e["temperature"],dict(cohort="unseen_map_development",source=job["folder"],decision=d))
        after=q.apply_state_delta(state,e["delta"])
        gap=0 if after["num_of_colliding_pairs"]<best else gap+1
        best=min(best,after["num_of_colliding_pairs"])
        state=after
    return dict(status="ok",id=job["id"],reason="no_eligible_prefix_state")


def scan_post(job):
    old=io.read(ROOT/OLD)
    file=ROOT/job["file"]
    io.require(io.sha256_file(file)==old["sources"][job["file"]],"continuation changed")
    row=io.read(file)
    root=ROOT/"build/sa-certificate-continuation-v1/roots/ce45f43b848696ee701f8b60.json"
    io.require(io.sha256_file(root)==old["sources"][root.relative_to(ROOT).as_posix()],"root changed")
    state=io.read(root)
    for offset,e in enumerate(row["events"]):
        if offset>=8 and e["metrics"]["pp_rolled_back"]:
            return materialize(state,e["action"]["agents"],job["id"],e["temperature"],dict(cohort="post_intervention_development",source=job["file"],decision=e["decision"],offset=offset))
        state=q.apply_state_delta(state,e["delta"])
    return dict(status="ok",id=job["id"],reason="no_late_rejection")


def load_case(row):
    io.require(io.sha256_file(ROOT/row["case_path"])==row["case_sha"],"case changed")
    case=io.read(ROOT/row["case_path"])
    for path,sha in case["file_hashes"].items():
        io.require(io.sha256_file(ROOT/path)==sha,"mirrored map/scenario changed")
    return case


def probe(job):
    case=load_case(job["source"])
    binary=ROOT/io.NATIVE
    io.require(io.sha256_file(binary)==io.NATIVE_SHA,"native changed")
    sys.path.insert(0,str(binary.parent))
    import lns2_env
    io.require(Path(lns2_env.__file__).resolve()==binary.resolve(),"wrong native")
    agents=sorted(case["state"]["agents"],key=lambda a:a["id"])
    env=lns2_env.LNS2RepairEnv(str(ROOT/case["files"][".map"]),str(ROOT/case["files"][".scen"]),len(agents),time_limit=60,use_sipp=True)
    before=q._plain(env.reset_paths([a["path"] for a in agents],seed=17))
    io.require(io.repair_structure_fingerprint(before)==io.repair_structure_fingerprint(case["state"]),"mirror reset mismatch")
    seed=int(io.semantic_fingerprint([case["id"],20260915,job["trial"]])[:7],16)
    draw=random.Random(seed).random()
    members=job["candidate"]["members"]
    raw=q._plain(env.step_experimental_pp(dict(mode="explicit_neighborhood",agents=members,random_seed=seed),5.,"annealed",case["temperature"],draw))
    after,m=raw["observation"],raw["metrics"]
    ref.validate_state(after)
    validate_transition(before,after,m,members,"annealed",case["temperature"],draw)
    io.require(after["feasible"]==(after["num_of_colliding_pairs"]==0),"feasible mismatch")
    return dict(id=job["id"],status="ok",case_id=case["id"],trial=job["trial"],candidate=job["candidate"],
                before=before["num_of_colliding_pairs"],after=after["num_of_colliding_pairs"],
                generated=after["low_level"]["generated"],censored=m["pp_failure_reason"]=="time_limit",
                metrics=m,final_state=after)


def read_stage(stage):
    folder=OUT/stage
    m=io.read(folder/"manifest.json")
    plan=read_bound_plan(ROOT,OUT,evidence_only=True)
    io.require(m["binding"]==plan["binding"],"stage binding")
    io.require(io.sha256_file(folder/"schedule.json")==m["schedule_sha"],"stage schedule changed")
    scheduled=io.read(folder/"schedule.json")
    io.require(scheduled["binding"]==plan["binding"] and set(m["files"])=={j["id"] for j in scheduled["jobs"]},"stage coverage mismatch")
    rows=[]
    for name,sha in m["files"].items():
        path=folder/"rows"/(name+".json")
        io.require(io.sha256_file(path)==sha,"saved row changed")
        row=io.read(path)
        io.require(row["id"]==name and row["binding"]==plan["binding"] and row["status"]=="ok","row identity/error")
        rows.append(row)
    return rows


def replacement_allowed(rows):
    baselines={(r["case_id"],r["trial"]):r for r in rows if r["candidate"]["id"]=="baseline"}
    return any(r["candidate"]["kind"]=="certificate" and not r["censored"]
               and not baselines[(r["case_id"],r["trial"])]["censored"]
               and r["after"]<baselines[(r["case_id"],r["trial"])]["after"] for r in rows)


def process_jobs(jobs):
    # The shared scheduler reads job_id before registering the child for cleanup.
    return [dict(job,job_id=job["id"]) for job in jobs]


def schedule(plan,stage):
    if stage=="scan": return [dict(j,id="unseen-"+j["job_id"]) for j in plan["jobs"]]
    if stage=="post":
        return [dict(id=f"post-{arm}-{trial}",file=f"build/sa-certificate-continuation-v1/branches/ce45f43b848696ee701f8b60/{arm}-{trial}.json")
                for arm in ("add_certified","replace_padding") for trial in range(4)]
    if stage=="replace" and not replacement_allowed(read_stage("add")): return []
    cases=[r for phase in ("scan","post") for r in read_stage(phase) if r.get("case_path")]
    jobs=[]
    seen=set()
    for row in cases:
        case=load_case(row)
        key=io.semantic_fingerprint([io.repair_structure_fingerprint(case["state"]),case["selected"],case["temperature"]])
        if key in seen: continue
        seen.add(key)
        for candidate in candidate_sets(case,stage):
            for trial in range(4):
                jobs.append(dict(id=f"{case['id']}-{candidate['id']}-{trial}",source=row,candidate=candidate,trial=trial))
    return jobs


def collect(stage,resume=False):
    plan=read_bound_plan(ROOT,OUT)
    folder=OUT/stage
    jobs=schedule(plan,stage)
    with q._CollectionRunLock(folder,plan["binding"],stage):
        if (folder/"manifest.json").exists():
            io.require(resume,"completed phase exists")
            rows=read_stage(stage)
            io.require({r["id"] for r in rows}=={j["id"] for j in jobs},"schedule mismatch")
            print(dict(stage=stage,verified_resume=len(rows)))
            return
        pending=[]
        for j in jobs:
            path=folder/"rows"/(j["id"]+".json")
            if path.exists():
                io.require(resume,"existing partial row")
                receipt=io.read(folder/"receipts"/(j["id"]+".json"))
                io.require(receipt==dict(binding=plan["binding"],job=io.semantic_fingerprint(j),sha=io.sha256_file(path)),"partial receipt mismatch")
                io.require(io.read(path)["status"]=="ok","failed job requires inspection")
            else: pending.append(j)
        lookup={j["id"]:j for j in jobs}
        def failed(job,status,error): return dict(id=job["id"],status=status,error=error)
        def save(row):
            row["binding"]=plan["binding"]
            path=folder/"rows"/(row["id"]+".json")
            io.write(path,row)
            io.write(folder/"receipts"/(row["id"]+".json"),dict(binding=plan["binding"],job=io.semantic_fingerprint(lookup[row["id"]]),sha=io.sha256_file(path)))
            print(stage,row["id"],row["status"],flush=True)
        io.write(folder/"schedule.json",dict(binding=plan["binding"],jobs=jobs))
        rows=q._run_jobs(scan_episode if stage=="scan" else scan_post if stage=="post" else probe,process_jobs(pending),
            workers=20,phase=stage,output_root=folder/"progress",run_fingerprint=plan["binding"],timeout_seconds=180.,
            failure_result=failed,on_result=save,stop_on_failure=True)
        io.require(len(rows)==len(pending) and all(r["status"]=="ok" for r in rows),"phase incomplete; inspect errors")
        io.write(folder/"manifest.json",dict(binding=plan["binding"],schedule_sha=io.sha256_file(folder/"schedule.json"),files={j["id"]:io.sha256_file(folder/"rows"/(j["id"]+".json")) for j in jobs}))
        print(dict(stage=stage,jobs=len(jobs),errors=0))


def analyze():
    plan=read_bound_plan(ROOT,OUT,evidence_only=True)
    coverage=[]
    for phase in ("scan","post"):
        rows=read_stage(phase)
        cases=[load_case(r) for r in rows if r.get("case_path")]
        coverage.append(dict(cohort=phase,episodes=len(rows),states=len(cases),maps=len({c["map_hash"] for c in cases}),
            proved=sum(c["certificate"]["status"]=="proved" for c in cases),statuses=dict(Counter(c["certificate"]["status"] for c in cases))))
    results={}
    for phase in ("add","replace"):
        if not (OUT/phase/"manifest.json").exists(): continue
        rows=read_stage(phase)
        jobs=schedule(plan,phase)
        io.require({j["id"] for j in jobs}=={r["id"] for r in rows},"probe schedule incomplete")
        lookup={j["id"]:j for j in jobs}
        grouped=defaultdict(list)
        for r in rows:
            job=lookup[r["id"]]
            case=load_case(job["source"])
            io.require(r["candidate"]==job["candidate"] and r["trial"]==job["trial"],"probe identity mismatch")
            after=r["final_state"]
            ref.validate_state(after)
            draw=random.Random(int(io.semantic_fingerprint([case["id"],20260915,r["trial"]])[:7],16)).random()
            validate_transition(case["state"],after,r["metrics"],r["candidate"]["members"],"annealed",case["temperature"],draw)
            io.require(r["after"]==after["num_of_colliding_pairs"] and after["feasible"]==(r["after"]==0),"outcome mismatch")
            grouped[r["case_id"]].append(r)
        tables=[]
        for case_id,group in grouped.items():
            baseline={r["trial"]:r for r in group if r["candidate"]["id"]=="baseline"}
            io.require(set(baseline)==set(range(4)),"missing baseline trials")
            for name in sorted({r["candidate"]["id"] for r in group}):
                xs=sorted([r for r in group if r["candidate"]["id"]==name],key=lambda r:r["trial"])
                io.require([r["trial"] for r in xs]==list(range(4)),"trial completeness")
                tables.append(dict(case_id=case_id,candidate=name,before=xs[0]["before"],after=[r["after"] for r in xs],
                    better=sum(r["after"]<baseline[r["trial"]]["after"] for r in xs),worse=sum(r["after"]>baseline[r["trial"]]["after"] for r in xs),
                    completed=sum(r["after"]==0 for r in xs),censored=sum(r["censored"] for r in xs),
                    generated_mean=sum(r["generated"] for r in xs)/4,
                    attempt_deltas=[r["metrics"]["pp_attempt_conflict_pair_count"]-r["metrics"]["pp_old_conflict_pair_count"] for r in xs]))
        results[phase]=dict(jobs=len(rows),tables=tables)
    io.write(OUT/"report.json",dict(schema="lns2.sa_certificate_generalization_report.v1",binding=plan["binding"],coverage=coverage,
        results=results,no_ttf=True,controller_promoted=False,no_training=True,
        sources={p.relative_to(ROOT).as_posix():io.sha256_file(p) for p in OUT.rglob("*.json") if p.name!="report.json"}))
    print(coverage)
    print({k:dict(jobs=v["jobs"],improved_candidates=sum(t["better"]>0 for t in v["tables"]),completed=sum(t["completed"] for t in v["tables"])) for k,v in results.items()})


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","scan","post","add","replace","analyze"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.stage=="prepare": prepare()
    elif args.stage=="analyze": analyze()
    else: collect(args.stage,args.resume)
