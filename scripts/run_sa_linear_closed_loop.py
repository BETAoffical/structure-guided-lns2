"""New-map, fixed-work SA selector comparison. Diagnostic only, not TTF."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key]="1"
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json,read_jsonl,sha256_file,json_fingerprint
from experiments.sa_paired_completion import require,validate_dataset,MODEL_PARAMS
from experiments.sa_paired_closed_loop import portable_model,portable_payload,stop_reason,subset
from experiments.sa_linear_closed_loop import ARMS,LinearRanker,fit_full,choose,summarize
from scripts.run_sa_paired_closed_loop import (
    once,sealed,check_seal,reset_case,reset_worker,paired_tasks,features_for,receipt_valid,
)
from scripts.audit_sa_history_information import atomic

CONFIG=ROOT/"configs/sa_linear_closed_loop.json"


def prepare():
    from scripts.run_sa_low_complexity_ranker import verify as verify_pilot
    from scripts.run_sa_paired_closed_loop import verify as verify_old
    from scripts.run_sa_independent_confirmation import historical_inventory
    cfg=read_json(CONFIG)
    out=ROOT/cfg["output"]
    require(not out.exists(),"existing registration; no overwrite")
    pilot,pilot_out=verify_pilot()
    old,_=verify_old()
    require(pilot_out==ROOT/cfg["source"],"source identity")
    data=validate_dataset(read_json(ROOT/cfg["training_index"]))
    require(len(data["states"])==47 and len({s["map_id"] for s in data["states"]})==8,"training cohort changed")
    inputs=dict(old["inputs"])
    inputs.update(pilot["files"])
    paths=[CONFIG,Path(__file__),ROOT/"experiments/sa_linear_closed_loop.py",
        ROOT/"tests/evaluation/test_sa_linear_closed_loop.py",ROOT/"docs/SA_LINEAR_CLOSED_LOOP_PROTOCOL_ZH.md",
        pilot_out/"plan.json",pilot_out/"report.json",ROOT/cfg["training_index"],
        ROOT/"scripts/run_sa_low_complexity_ranker.py",ROOT/"scripts/run_sa_paired_closed_loop.py"]
    for p in paths:
        inputs[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    history=historical_inventory(exclude=out)
    inputs.update(history["manifests"])
    rng=random.Random(cfg["master_seed"])
    masters=[rng.randrange(1,2**31) for _ in range(cfg["maps"])]
    require(len(set(masters))==8 and not set(masters)&set(history["seeds"]),"master seed overlap")
    require(cfg["arms"]==list(ARMS) and not cfg["formal_ttf"] and cfg["workers"]<=20,"protocol scope")
    for name,digest in inputs.items():
        require(sha256_file(ROOT/name)==digest,"source changed: "+name)
    plan=dict(config=cfg,inputs=inputs,history=history,map_masters=masters,model=MODEL_PARAMS,
        native_file=old["native_file"],native_sha256=old["native_sha256"],
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        role="independent_map_bounded_diagnostic",no_ttf=True)
    plan["binding"]=json_fingerprint(plan)
    once(out/"plan.json",plan)
    return dict(binding=plan["binding"],maps=8,paired_tasks=32,episodes=96,maximum_repairs=96*512,
        workers=cfg["workers"],wave_fuse_minutes=5*cfg["fuse_seconds"]/60,no_ttf=True)


def verify():
    cfg=read_json(CONFIG)
    out=ROOT/cfg["output"]
    p=read_json(out/"plan.json")
    require(p["config"]==cfg and p["binding"]==json_fingerprint({k:v for k,v in p.items() if k!="binding"}),"plan changed")
    for name,digest in p["inputs"].items():
        require(sha256_file(ROOT/name)==digest,"registered input changed: "+name)
    return p,out


def model_receipt(plan,out):
    r=check_seal(read_json(out/"model/receipt.json"))
    require(r["binding"]==plan["binding"],"model binding")
    for name,digest in r["files"].items():
        require(sha256_file(out/"model"/name)==digest,"model artifact changed")
    return r


def models_load(out,native=True):
    meta=read_json(out/"model/metadata.json")
    return dict(linear=LinearRanker(read_json(out/"model/linear.json")),
        gbdt=portable_model(read_json(out/"model/gbdt.json"),meta["feature_names"],native=native))


def freeze():
    from lns2_selector.runtime.fingerprints import semantic_fingerprint
    from experiments.sa_low_complexity_ranker import predict
    p,out=verify()
    if (out/"model/receipt.json").exists():
        return dict(resumed=True,receipt=model_receipt(p,out))
    require(not (out/"model").exists(),"partial model publication; audit required")
    data=read_json(ROOT/p["config"]["training_index"])
    linear,gbdt=fit_full(data)
    payload=portable_payload(gbdt)
    payload["name"]="paired_completion_map_balanced_47"
    payload["semantic_fingerprint"]=semantic_fingerprint({k:v for k,v in payload.items() if k not in {"schema","semantic_fingerprint"}})
    portable=portable_model(payload,data["feature_names"],native=False)
    scalar=LinearRanker(linear)
    fixtures=[]
    for s in data["states"]:
        require(gbdt.rank(s)==portable.rank(s),"GBDT portable mismatch")
        a,b=predict(linear,s),scalar.rank(s)
        require(a["selected"]==b["selected"] and all(abs(a["scores"][k]-v)<=1e-12 for k,v in b["scores"].items()),"linear numerical mismatch")
        clean={k:s[k] for k in ("state_id","anchor_id","agent_ids")}
        clean["candidates"]=[{k:c[k] for k in ("candidate_id","agents","features")} for c in s["candidates"]]
        fixtures.append(dict(state=clean,linear=b,gbdt=gbdt.rank(s)))
    ranges={n:[min(c["features"][n] for s in data["states"] for c in s["candidates"]),
               max(c["features"][n] for s in data["states"] for c in s["candidates"])] for n in data["feature_names"]}
    once(out/"model/linear.json",linear)
    once(out/"model/gbdt.json",payload)
    once(out/"model/fixtures.json",fixtures)
    once(out/"model/metadata.json",dict(feature_names=data["feature_names"],ranges=ranges,
        training_sha256=sha256_file(ROOT/p["config"]["training_index"]),train_ids=linear["train_ids"],train_maps=linear["train_maps"],
        gbdt_params=MODEL_PARAMS,linear_alpha=1.,production_allowed=False))
    once(out/"model/receipt.json",sealed(dict(binding=p["binding"],files={f.name:sha256_file(f) for f in sorted((out/"model").iterdir())})))
    return dict(training_states=47,training_maps=8,model_sha256=model_receipt(p,out)["files"])


def verify_models():
    p,out=verify()
    model_receipt(p,out)
    models=models_load(out)
    for r in read_json(out/"model/fixtures.json"):
        for arm in ("linear","gbdt"):
            require(models[arm].rank(r["state"])==r[arm],"portable fixture mismatch: "+arm)
    backend=models["gbdt"].estimator.model.inference_backend
    record=dict(binding=p["binding"],backend=backend,fixtures=47,exact=True,
        receipt_sha256=sha256_file(out/"model/receipt.json"))
    once(out/f"model-verification-{backend}.json",record)
    return record


def generate_one(job):
    from generators.dataset import generate_dataset
    p,index=job["plan"],job["index"]
    cfg=p["config"]
    folder=ROOT/cfg["output"]/f"dataset/shards/{index:02d}"
    if (folder/"receipt.json").exists():
        r=check_seal(read_json(folder/"receipt.json"))
        require(r["binding"]==p["binding"],"shard binding")
        for name,digest in r["files"].items():
            require(sha256_file(ROOT/name)==digest,"generated file changed")
        return r
    require(not folder.exists(),"partial dataset shard; no automatic retry")
    base=read_json(ROOT/cfg["dataset_base"])
    base.update(master_seed=p["map_masters"][index],map_id_prefix=f"sa_linear_v1_m{index:02d}",
        tasks_per_map=2,splits={"confirmation":{"layout_counts":{"station_centric":1}}})
    template=deepcopy(base["task_variants"][0])
    base["task_variants"]=[]
    for density in cfg["densities"]:
        v=deepcopy(template)
        v["name"]=f"bottleneck_d{int(density*100)}"
        v["task"].update(agent_density=density,required_bottleneck_crossing_ratio=cfg["bottleneck_ratio"])
        base["task_variants"].append(v)
    generate_dataset(base,output_override=folder.relative_to(ROOT))
    rows=read_jsonl(folder/"confirmation/manifest.jsonl")
    for row in rows:
        for k in ("map_file","map_metadata_file","task_file","scenario_file","instance_file","legacy_instance_file"):
            row[k]=(folder/"confirmation"/row[k]).relative_to(ROOT).as_posix()
    r=sealed(dict(binding=p["binding"],rows=rows,files={f.relative_to(ROOT).as_posix():sha256_file(f) for f in sorted(folder.rglob("*")) if f.is_file()}))
    once(folder/"receipt.json",r)
    return r


def generate():
    from lns2_selector.evaluation.path_quality_preflight import audit_task
    p,out=verify()
    cfg=p["config"]
    model_receipt(p,out)
    v=read_json(out/"model-verification-native-portable-tree.json")
    require(v["binding"]==p["binding"] and v["receipt_sha256"]==sha256_file(out/"model/receipt.json"),"native model check missing")
    once(out/"generation_registration.json",dict(binding=p["binding"],model_receipt_sha256=sha256_file(out/"model/receipt.json")))
    with ProcessPoolExecutor(max_workers=8) as pool:
        shards=list(pool.map(generate_one,[dict(plan=p,index=i) for i in range(8)]))
    rows=[r for s in shards for r in s["rows"]]
    seeds={r["map_seed"] for r in rows}|{r["task_seed"] for r in rows}
    hashes={r["map_id"]:sha256_file(ROOT/r["map_file"]) for r in rows}
    require(len(rows)==16 and len({r["task_seed"] for r in rows})==16 and len(seeds)==24,"case/seed counts")
    require(len(hashes)==8 and len(set(hashes.values()))==8,"duplicate geometry")
    require(not seeds&set(p["history"]["seeds"]) and not set(hashes.values())&set(p["history"]["map_hashes"]),"historical overlap")
    cases=[]
    for r in rows:
        audit=audit_task(ROOT/r["map_file"],ROOT/r["scenario_file"],ROOT/r["task_file"],r["agent_count"])
        cases.append(dict(task_id=r["task_id"],map_id=r["map_id"],family="warehouse",solver_seeds=cfg["solver_seeds"],
            static_audit=audit,files={k:r[k] for k in ("map_file","scenario_file","task_file")},
            density=r.get("agent_density"),task_variant=r["task_variant"]))
    once(out/"cases.json",sealed(dict(binding=p["binding"],cases=cases,files={k:v for s in shards for k,v in s["files"].items()},
        map_sha256=hashes,map_seeds=sorted({r["map_seed"] for r in rows}),task_seeds=sorted({r["task_seed"] for r in rows}))))
    return dict(maps=8,tasks=16,seed_overlap=0,geometry_overlap=0)


def cases_verified(p,out):
    r=check_seal(read_json(out/"cases.json"))
    freeze=read_json(out/"generation_registration.json")
    require(r["binding"]==p["binding"]==freeze["binding"] and
        freeze["model_receipt_sha256"]==sha256_file(out/"model/receipt.json"),"cases/model binding")
    for name,digest in r["files"].items():
        require(sha256_file(ROOT/name)==digest,"case changed")
    return r["cases"]


def qualify():
    from experiments.repair_collection import _run_jobs,_CollectionRunLock
    p,out=verify()
    cases=cases_verified(p,out)
    if (out/"qualification.json").exists():
        return check_seal(read_json(out/"qualification.json"))
    cfg=p["config"]
    jobs=[dict(job_id=k,case=c,solver_seed=s,plan=p) for c,s,k in paired_tasks(cases,cfg)]
    with _CollectionRunLock(out,p["binding"],"linear-reset"):
        results=_run_jobs(reset_worker,jobs,cfg["workers"],phase="reset",output_root=out/"reset-progress",
            run_fingerprint=p["binding"],timeout_seconds=cfg["fuse_seconds"],stop_on_failure=True)
        active=[r for r in results if r.get("conflicts",0)>0]
        passed=len(results)==32 and all(r["status"]=="ok" for r in results) and len(active)>=cfg["minimum_active_episodes"] and len({r["map_id"] for r in active})>=cfg["minimum_active_maps"]
        r=sealed(dict(binding=p["binding"],passed=passed,results=results,active=len(active),active_maps=len({r["map_id"] for r in active})))
        once(out/"qualification.json",r)
    return r


def episode_worker(job):
    from scripts import run_sa_path_quality as q
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from scripts.train_sa_history_selector import die_with_parent
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    die_with_parent(job["parent_pid"])
    p,case,arm=job["plan"],job["case"],job["arm"]
    cfg=p["config"]
    out=ROOT/cfg["output"]
    folder=out/"episodes"/job["job_id"]
    if (folder/"receipt.json").exists():
        r=receipt_valid(folder,p)
        return dict(status="ok",job_id=job["job_id"],stop=r["stop"],resumed=True)
    require(not folder.exists(),"partial episode; inspect before retry")
    model_receipt(p,out)
    models=models_load(out)
    require(models["gbdt"].estimator.model.inference_backend=="native-portable-tree","native GBDT required")
    meta=read_json(out/"model/metadata.json")
    env,state,source=reset_case(case,job["solver_seed"],p)
    initial=q.state_fingerprint(state)
    require(initial==job["expected_initial"],"paired reset mismatch")
    validate_final(state)
    once(folder/"initial.json",dict(binding=p["binding"],state=state))
    initial_nodes=state["low_level"]["generated"]
    history=History(state)
    selector=q.SingleFullCheckPool(source)
    engine=OnlineFeatureEngine(state,backend="native")
    began=time.monotonic()
    count=0
    seen=set()
    revisits=0
    trajectory=[state["num_of_colliding_pairs"]]
    with (folder/"trace.jsonl").open("x",encoding="utf8") as stream:
        while True:
            stop=stop_reason(state,count,state["low_level"]["generated"]-initial_nodes,time.monotonic()-began,cfg)
            if stop:
                break
            fp=q.state_fingerprint(state)
            anchor_index,pool=selector.select(env,state,count)
            require(q.state_fingerprint(env.get_state())==fp,"proposal changed state")
            anchor=pool[anchor_index]["candidate_id"]
            occurrence=f"{job['pair_id']}-d{count:04d}-{fp}"
            ids=subset(pool,anchor,occurrence,cfg["candidate_seed"],cfg["candidate_limit"])
            candidates=[next(c for c in pool if c["candidate_id"]==i) for i in ids]
            temp=q.temperature(count)
            features=features_for(state,candidates,engine,history,temp,fp)
            ranking=choose(arm,candidates,anchor,models,features,[a["id"] for a in state["agents"]])
            index=next(i for i,c in enumerate(pool) if c["candidate_id"]==ranking["selected"])
            pp=q.seed(source["case_id"],0,count,"pp")
            draw=q.acceptance_draw(q.seed(source["case_id"],0,count,"accept"))
            action=dict(mode="explicit_neighborhood",agents=pool[index]["agents"],random_seed=pp)
            remaining=cfg["episode_seconds"]-(time.monotonic()-began)
            if remaining<=0:
                stop="wall_safety"
                break
            raw=q._plain(env.step_experimental_pp(action,min(cfg["pp_seconds"],remaining),"annealed",temp,draw))
            after,m=raw["observation"],raw["metrics"]
            q.validate_transition(state,after,m,action["agents"],"annealed",temp,draw)
            physical=json_fingerprint([(a["id"],a["path"]) for a in state["agents"]])
            revisits+=physical in seen
            seen.add(physical)
            outside=sum(v<meta["ranges"][n][0] or v>meta["ranges"][n][1] for f in features for n,v in f.items())
            event=dict(decision=count,before=fp,action=action,temperature=temp,uniform=draw,metrics=m,pool=pool,
                subset=ids,anchor_id=anchor,selected_index=index,ranking=ranking,features=features,
                feature_outside_fraction=outside/(len(features)*len(meta["feature_names"])),physical=physical,
                delta=q.encode_state_delta(state,after))
            stream.write(json.dumps(event,separators=(",",":"),allow_nan=False)+"\n")
            stream.flush()
            history.observe(state,event,after)
            state=after
            count+=1
            trajectory.append(state["num_of_colliding_pairs"])
            if m["pp_failure_reason"]=="time_limit" or not m["acceptance_evaluated"]:
                stop="incomplete_pp"
                break
    validate_final(state)
    r=dict(binding=p["binding"],job_id=job["job_id"],pair_id=job["pair_id"],map_id=case["map_id"],arm=arm,
        solver_seed=job["solver_seed"],initial_fingerprint=initial,final_fingerprint=q.state_fingerprint(state),
        stop=stop,success=state["feasible"],decisions=count,generated=state["low_level"]["generated"]-initial_nodes,
        node_overshoot=max(0,state["low_level"]["generated"]-initial_nodes-cfg["node_budget"]),conflicts=trajectory,
        physical_revisits=revisits,final_soc=state["sum_of_costs"],final_makespan=max(len(a["path"])-1 for a in state["agents"]),
        diagnostic_seconds=time.monotonic()-began,no_ttf=True)
    once(folder/"result.json",r)
    once(folder/"receipt.json",dict(binding=p["binding"],files={n:sha256_file(folder/n) for n in ("initial.json","trace.jsonl","result.json")}))
    return dict(status="ok",job_id=job["job_id"],stop=stop,success=r["success"],decisions=count)


def collect(resume=False,limit=None):
    from experiments.repair_collection import _run_jobs,_CollectionRunLock
    p,out=verify()
    cfg=p["config"]
    qualification=check_seal(read_json(out/"qualification.json"))
    require(qualification["binding"]==p["binding"] and qualification["passed"],"qualification failed; no seed redraw")
    require(resume or not (out/"episodes").exists(),"existing episodes require --resume")
    model_receipt(p,out)
    cases=cases_verified(p,out)
    anchors={r["job_id"]:r["initial"] for r in qualification["results"]}
    once(out/"execution_registration.json",dict(binding=p["binding"],qualification_sha256=sha256_file(out/"qualification.json"),
        cases_sha256=sha256_file(out/"cases.json"),model_receipt_sha256=sha256_file(out/"model/receipt.json")))
    jobs=[]
    for case,seed,key in paired_tasks(cases,cfg):
        for arm in ARMS:
            jobs.append(dict(job_id=f"{key}-{arm}",pair_id=key,case=case,solver_seed=seed,arm=arm,
                expected_initial=anchors[key],plan=p,parent_pid=os.getpid()))
    if limit is not None:
        require(0<limit<=len(jobs),"invalid limit")
        jobs=jobs[:limit]
    pending=[j for j in jobs if not (out/"episodes"/j["job_id"]/"receipt.json").exists()]
    for j in jobs:
        if (out/"episodes"/j["job_id"]/"receipt.json").exists():
            receipt_valid(out/"episodes"/j["job_id"],p)
    with _CollectionRunLock(out,p["binding"],"linear-closed-loop"):
        atomic(out/"run_status.json",dict(status="running",binding=p["binding"]))
        try:
            for offset in range(0,len(pending),cfg["workers"]):
                if (out/"STOP_AFTER_BATCH").exists():
                    break
                batch=pending[offset:offset+cfg["workers"]]
                def record(row):
                    with (out/"progress.jsonl").open("a",encoding="utf8") as f:
                        f.write(json.dumps(row)+"\n")
                    print("EPISODE",row,flush=True)
                result=_run_jobs(episode_worker,batch,cfg["workers"],phase=f"batch{offset}",output_root=out/f"progress/{offset}",
                    run_fingerprint=p["binding"],timeout_seconds=cfg["fuse_seconds"],stop_on_failure=True,on_result=record)
                require(len(result)==len(batch) and all(r["status"]=="ok" for r in result),"batch error; no automatic retry")
            completed=sum((out/"episodes"/f"{key}-{arm}"/"receipt.json").exists() for _,_,key in paired_tasks(cases,cfg) for arm in ARMS)
            atomic(out/"run_status.json",dict(status="completed" if completed==96 else "paused",completed=completed,binding=p["binding"]))
        except BaseException as exc:
            atomic(out/"run_status.json",dict(status="error",error=repr(exc),binding=p["binding"]))
            raise
    return dict(completed=completed,total=96,no_ttf=True)


def audit_episode(job):
    from scripts import run_sa_path_quality as q
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.sa_history_selector import History
    from experiments.online_feature_engine import OnlineFeatureEngine
    p,folder=job["plan"],Path(job["folder"])
    cfg=p["config"]
    result=receipt_valid(folder,p)
    initial=read_json(folder/"initial.json")
    require(initial["binding"]==p["binding"] and result["binding"]==p["binding"],"trace binding")
    state=initial["state"]
    require(q.state_fingerprint(state)==result["initial_fingerprint"],"initial mismatch")
    out=ROOT/cfg["output"]
    models=models_load(out,native=False)
    meta=read_json(out/"model/metadata.json")
    history=History(state)
    engine=OnlineFeatureEngine(state,backend="native")
    nodes=state["low_level"]["generated"]
    conflicts=[state["num_of_colliding_pairs"]]
    count=changed=revisits=rollbacks=0
    outside=[]
    seen=set()
    for e in read_jsonl(folder/"trace.jsonl"):
        require(e["decision"]==count and e["before"]==q.state_fingerprint(state),"trace sequence")
        key=f"{result['pair_id']}-d{count:04d}-{e['before']}"
        ids=subset(e["pool"],e["anchor_id"],key,cfg["candidate_seed"],cfg["candidate_limit"])
        require(ids==e["subset"],"candidate subset changed")
        candidates=[next(c for c in e["pool"] if c["candidate_id"]==i) for i in ids]
        f=features_for(state,candidates,engine,history,e["temperature"],e["before"])
        require(f==e["features"],"online feature mismatch")
        ranking=choose(result["arm"],candidates,e["anchor_id"],models,f,[a["id"] for a in state["agents"]])
        require(ranking==e["ranking"] and e["pool"][e["selected_index"]]["candidate_id"]==ranking["selected"],"model choice mismatch")
        case_id=result["pair_id"].rsplit("-s",1)[0]+f"-seed{result['solver_seed']}"
        require(e["action"]["random_seed"]==q.seed(case_id,0,count,"pp") and
            e["uniform"]==q.acceptance_draw(q.seed(case_id,0,count,"accept")) and e["temperature"]==q.temperature(count),"PP/SA stream mismatch")
        after=q.apply_state_delta(state,e["delta"])
        require(e["action"]["agents"]==e["pool"][e["selected_index"]]["agents"],"altered action")
        q.validate_transition(state,after,e["metrics"],e["action"]["agents"],"annealed",e["temperature"],e["uniform"])
        physical=json_fingerprint([(a["id"],a["path"]) for a in state["agents"]])
        require(physical==e["physical"],"physical fingerprint")
        actual_outside=sum(v<meta["ranges"][n][0] or v>meta["ranges"][n][1] for row in f for n,v in row.items())/(len(f)*len(meta["feature_names"]))
        require(actual_outside==e["feature_outside_fraction"],"feature range diagnostic changed")
        outside.append(actual_outside)
        revisits+=physical in seen
        seen.add(physical)
        changed+=ranking["selected"]!=e["anchor_id"]
        rollbacks+=bool(e["metrics"]["pp_rolled_back"])
        history.observe(state,e,after)
        state=after
        count+=1
        conflicts.append(state["num_of_colliding_pairs"])
    validate_final(state)
    require(result["final_fingerprint"]==q.state_fingerprint(state) and result["conflicts"]==conflicts and
        result["decisions"]==count and result["generated"]==state["low_level"]["generated"]-nodes and
        result["physical_revisits"]==revisits and result["success"]==state["feasible"],"result mismatch")
    require(result["final_soc"]==state["sum_of_costs"] and result["final_makespan"]==max(len(a["path"])-1 for a in state["agents"]),"path quality mismatch")
    if result["stop"] not in {"wall_safety","incomplete_pp"}:
        require(stop_reason(state,count,result["generated"],0,cfg)==result["stop"],"stop mismatch")
    return result|dict(changed_from_anchor=changed,rollbacks=rollbacks,mean_feature_outside=sum(outside)/max(1,len(outside)))


def analyze():
    p,out=verify()
    model_receipt(p,out)
    cases=cases_verified(p,out)
    qualification=check_seal(read_json(out/"qualification.json"))
    anchors={r["job_id"]:r["initial"] for r in qualification["results"]}
    execution=read_json(out/"execution_registration.json")
    require(execution==dict(binding=p["binding"],qualification_sha256=sha256_file(out/"qualification.json"),
        cases_sha256=sha256_file(out/"cases.json"),model_receipt_sha256=sha256_file(out/"model/receipt.json")),"execution identity")
    jobs=[dict(folder=str(out/"episodes"/f"{k}-{arm}"),plan=p) for _,_,k in paired_tasks(cases,p["config"]) for arm in ARMS]
    require(all((Path(j["folder"])/"receipt.json").exists() for j in jobs),"incomplete cohort")
    with ProcessPoolExecutor(max_workers=p["config"]["workers"]) as pool:
        episodes=list(pool.map(audit_episode,jobs))
    for r,j in zip(episodes,jobs,strict=True):
        require(r["job_id"]==Path(j["folder"]).name and r["job_id"]==r["pair_id"]+"-"+r["arm"]
            and r["initial_fingerprint"]==anchors[r["pair_id"]],"schedule identity")
    report=summarize(episodes,p["config"])
    report.update(binding=p["binding"],episodes=episodes,files={
        (Path(j["folder"])/n).relative_to(out).as_posix():sha256_file(Path(j["folder"])/n)
        for j in jobs for n in ("receipt.json","initial.json","trace.jsonl","result.json")})
    verify()
    once(out/"report.json",sealed(report))
    return {k:v for k,v in report.items() if k not in {"files","episodes","paired_cases"}}


def main():
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","freeze","verify-model","generate","qualify","collect","analyze","verify","request-stop"))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.phase=="collect":
        result=collect(args.resume,args.limit)
    elif args.phase=="request-stop":
        p,out=verify()
        once(out/"STOP_AFTER_BATCH",dict(binding=p["binding"],requested=True))
        result=dict(stop_after_current_batch=True)
    elif args.phase=="verify":
        p,out=verify()
        model_receipt(p,out)
        if (out/"report.json").exists():
            report=check_seal(read_json(out/"report.json"))
            require(report["binding"]==p["binding"],"report binding")
            for name,digest in report["files"].items():
                require(sha256_file(out/name)==digest,"result file changed")
        result=dict(binding=p["binding"],inputs=len(p["inputs"]),verified=True)
    else:
        result={"prepare":prepare,"freeze":freeze,"verify-model":verify_models,"generate":generate,
            "qualify":qualify,"analyze":analyze}[args.phase]()
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
