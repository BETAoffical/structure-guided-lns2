"""Recover capped Train terminals; at most one fixed condition-baseline update."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
from scripts import recover_sa_onpolicy as old
from scripts import run_sa_crossfit_update as crossfit
from scripts.compare_sa_uncapped import prefix_check
from experiments import sa_uncapped_runtime as runtime
from experiments import sa_uncapped_training_contract as credit

CONFIG = "configs/sa_uncapped_training.json"
CODE = (CONFIG, "scripts/recover_sa_uncapped_training.py", "experiments/sa_uncapped_training_contract.py",
        "tests/evaluation/test_sa_uncapped_training.py", "docs/SA_UNCAPPED_TRAINING_PROTOCOL_ZH.md",
        "experiments/sa_uncapped_runtime.py", "scripts/compare_sa_uncapped.py")
PHASE = "train-0"


def runtime_plan(source, cfg):
    run.require((cfg["max_decisions"],cfg["decision_feature_reference"],cfg["node_budget"],cfg["pp_safety_seconds"],
                 cfg["episode_safety_seconds"],cfg["process_fuse_seconds"],cfg["workers"],cfg["maximum_updates"],
                 cfg["formal_ttf"],cfg["heldout_evaluation"],cfg["automatic_promotion"]) ==
                (None,256,25000000,20.,900.,960.,20,1,False,False,False),"registered scope")
    run.require(cfg["update"] == run.read_json(ROOT/crossfit.CONFIG)["update"] and cfg["update_arm"] == "uncapped_condition", "fixed update")
    return dict(source, config=dict(source["config"],output=cfg["output"]), proposal=dict(source["proposal"],
        max_decisions=None,decision_feature_reference=256,episode_safety_seconds=900.,process_fuse_seconds=960.))


def partition(batch):
    entries = {}
    for entry, folder, row in batch:
        job=entry["job"]
        run.require(row["split"] == job["split"] == "train" and job["phase"] == PHASE and job["iteration"] == 0,
                    "Train actor-0 only")
        run.require(row["status"] == "ok" and row["stop"] in ("feasible","node_budget","decision_budget"),"complete source only")
        run.require(row["job_id"] not in entries,"duplicate episode")
        entries[row["job_id"]] = dict(job=job,expected_initial=entry["expected_initial"],
            origin="extend" if row["stop"] == "decision_budget" else "reuse",
            source_folder=folder.relative_to(ROOT).as_posix(),source_sha256=run.sha256_file(folder/"result.json"),
            original_stop=row["stop"],original_success=row["success"],original_decisions=row["decisions"])
    return entries


def folder_for(reg, e):
    return ROOT/e["source_folder"] if e["origin"] == "reuse" else ROOT/reg["config"]["output"]/PHASE/e["job"]["job_id"]


def prepare():
    source_reg, source, src = old.verify()
    cfg=run.read_json(ROOT/CONFIG)
    run.require(src == ROOT/cfg["source"],"source identity")
    plan=runtime_plan(source,cfg)
    batch=old.audited_batch(source_reg,source,src)
    entries=partition(batch)
    run.require(len(entries)==cfg["expected_episodes"]==96 and Counter(e["origin"] for e in entries.values()) ==
                {"reuse":cfg["expected_reused"],"extend":cfg["expected_extended"]} == {"reuse":80,"extend":16},"scope partition")
    out=ROOT/cfg["output"]
    run.require(not out.exists(),"existing recovery; use verify/resume")
    # Only Train artifacts are read. The prior heldout comparison is not an input.
    cross_reg, _, _, cross_out=crossfit.verify()
    cache=run.check_seal(run.read_json(ROOT/cfg["source_cache"]))
    run.require(cache["binding"]==cross_reg["binding"] and len(cache["rows"])==96,"source cache binding")
    caches={r["episode"]["job_id"]:r for r in cache["rows"]}
    run.require(set(caches)==set(entries),"source cache coverage")
    inputs={n:run.sha256_file(ROOT/n) for n in (*CODE,crossfit.CONFIG,cfg["source_cache"])}
    for name in ("registration.json","batch.audit.json","models/actor-0.json","models/receipt-0.json"):
        inputs[(src/name).relative_to(ROOT).as_posix()]=run.sha256_file(src/name)
    inputs[(cross_out/"registration.json").relative_to(ROOT).as_posix()]=run.sha256_file(cross_out/"registration.json")
    for jid,e in entries.items():
        row=run.result_read(ROOT/e["source_folder"],source)
        run.require(caches[jid]["source_sha256"]==e["source_sha256"],"cache/result identity")
        for name in ("result.json",*row["files"]):
            path=ROOT/e["source_folder"]/name
            inputs[path.relative_to(ROOT).as_posix()]=run.sha256_file(path)
        run.require(run.sha256_file(ROOT/caches[jid]["cache"])==caches[jid]["sha256"],"source cache changed")
        inputs[caches[jid]["cache"]]=caches[jid]["sha256"]
    body=dict(schema=cfg["schema"],config=cfg,entries=entries,inputs=inputs,scientific_binding=source["binding"],
        source_binding=source_reg["binding"],source_policy_sha256=run.validate_bundle(run.actor_load(src,source,0)),
        source_commit=run.subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        no_ttf=True,no_heldout=True)
    body["binding"]=run.json_fingerprint(body)
    with old.strict_lock(out,body["binding"],"uncapped-train-prepare"):
        for name in ("models/actor-0.json","models/receipt-0.json"):
            (out/name).parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src/name,out/name)
            run.require(run.sha256_file(src/name)==run.sha256_file(out/name),"model copy")
        run.once(out/"registration.json",run.sealed(body))
    return dict(registered=True,binding=body["binding"],reused=80,rerun=16,max_decisions=None)


def verify():
    source_reg,source,_=old.verify()
    cfg=run.read_json(ROOT/CONFIG)
    out=ROOT/cfg["output"]
    reg=run.check_seal(run.read_json(out/"registration.json"))
    run.require(reg["binding"]==run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding","integrity")}),"registration hash")
    run.require(reg["config"]==cfg and reg["scientific_binding"]==source["binding"] and
                reg["source_binding"]==source_reg["binding"],"source identity")
    for name,digest in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT,name,field="training input"))==digest,"changed input: "+name)
    p=runtime_plan(source,cfg)
    run.require(run.validate_bundle(run.actor_load(out,p,0))==reg["source_policy_sha256"],"actor-0 identity")
    expected={j["job_id"]:j for j in run.jobs_for(source,PHASE,0)}
    run.require({k:e["job"] for k,e in reg["entries"].items()}==expected,"Train schedule changed")
    return reg,p,out


def record(reg,p,e):
    folder=folder_for(reg,e)
    row=run.result_read(folder,p)
    j=e["job"]
    run.require(all(row[k]==j[k] for k in ("job_id","pair_id","replica","arm","split")) and
                row["map_id"]==j["case"]["map_id"] and row["initial_fingerprint"]==e["expected_initial"] and
                row["policy_sha256"]==reg["source_policy_sha256"],"episode identity")
    if e["origin"]=="reuse":
        run.require(run.sha256_file(folder/"result.json")==e["source_sha256"],"reused result changed")
    else:
        receipt=run.check_seal(run.read_json(folder/"recovery_receipt.json"))
        run.require(row["execution_binding"]==reg["binding"] and receipt==run.sealed(dict(
            binding=reg["binding"],result_sha256=run.sha256_file(folder/"result.json"))),"execution receipt")
    credit.terminal_return(row,max_decisions=None,node_budget=p["proposal"]["node_budget"])
    return folder,row


def job_for(reg,p,e):
    return dict(e["job"],plan=p,parent_pid=os.getpid(),expected_initial=e["expected_initial"],
                comparison_binding=reg["binding"],entry=e)


def worker(job):
    result=runtime.episode_worker(job)
    folder=ROOT/job["plan"]["config"]["output"]/PHASE/job["job_id"]
    run.once(folder/"recovery_receipt.json",run.sealed(dict(binding=job["comparison_binding"],
        result_sha256=run.sha256_file(folder/"result.json"))))
    return result


def collect(resume=False):
    from experiments.repair_collection import _run_jobs
    run.require(os.name!="nt","use frozen WSL native")
    reg,p,out=verify()
    with old.strict_lock(out,reg["binding"],"uncapped-train-collect"):
        run.require(resume or not (out/"run_status.json").exists(),"explicit resume required")
        pending=[]
        for e in reg["entries"].values():
            folder=folder_for(reg,e)
            run.require(not (out/"failures"/(e["job"]["job_id"]+".json")).exists(),"recorded failure requires inspection")
            if (folder/"result.json").exists():
                run.require(record(reg,p,e)[1]["status"]=="ok","censored result requires inspection")
            else:
                run.require(e["origin"]=="extend" and not folder.exists(),"partial or missing protected source")
                pending.append(job_for(reg,p,e))
        if resume:(out/"STOP_AFTER_BATCH").unlink(missing_ok=True)
        if (out/"STOP_AFTER_BATCH").exists():return dict(status="paused")
        def progress(row):
            with (out/"progress.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
        def failure(j,status,error):
            row=dict(status="censored" if status=="timeout" else "error",job_id=j["job_id"],error=error)
            run.once(out/"failures"/(j["job_id"]+".json"),run.sealed(dict(binding=reg["binding"],**row)))
            return row
        run.write_json(out/"run_status.json",dict(status="collecting",pending=len(pending),binding=reg["binding"]))
        try:
            rows=_run_jobs(worker,pending,reg["config"]["workers"],phase="uncapped-train",output_root=out/"progress",
                run_fingerprint=reg["binding"],timeout_seconds=960.,on_result=progress,failure_result=failure,stop_on_failure=False) if pending else []
            if len(rows)!=len(pending) or any(r["status"]!="ok" for r in rows):
                run.write_json(out/"run_status.json",dict(status="needs_inspection",no_update=True))
                return dict(status="needs_inspection",no_update=True)
            files={}
            for jid,e in reg["entries"].items():
                folder,row=record(reg,p,e)
                run.require(row["status"]=="ok","unknown terminal")
                files[jid]=run.sha256_file(folder/"result.json")
            run.once(out/"collection.complete.json",run.sealed(dict(binding=reg["binding"],files=files)))
            run.write_json(out/"run_status.json",dict(status="collected",episodes=96,binding=reg["binding"]))
        except BaseException as exc:
            run.write_json(out/"run_status.json",dict(status="interrupted_or_error",error=repr(exc),no_update=True))
            raise
    return dict(collected=96,reused=80,new=16,no_update=True)


def audit_worker(job):
    result=runtime.audit_worker(job)
    e=job["entry"]
    previous=run.result_read(ROOT/e["source_folder"],job["plan"])
    current=run.folder_for(ROOT/job["plan"]["config"]["output"],PHASE,job)
    row=run.result_read(current,job["plan"])
    run.require(run.sha256_file(ROOT/e["source_folder"]/"result.json")==e["source_sha256"],"old source changed")
    result["prefix_steps"]=prefix_check(run.trace_read(ROOT/e["source_folder"]),run.trace_read(current),previous,row)
    return result


def audit():
    from experiments.repair_collection import _run_jobs
    reg,p,out=verify()
    complete=run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(complete["binding"]==reg["binding"] and set(complete["files"])==set(reg["entries"]),"collection coverage")
    jobs,rows=[],[]
    for jid,e in reg["entries"].items():
        folder,row=record(reg,p,e)
        digest=run.sha256_file(folder/"result.json")
        run.require(digest==complete["files"][jid] and row["status"]=="ok","complete result changed")
        if e["origin"]=="reuse":rows.append(dict(status="ok",job_id=jid,result_sha256=digest,reused_source_audit=True))
        else:jobs.append(job_for(reg,p,e))
    with old.strict_lock(out,reg["binding"],"uncapped-train-audit"):
        fresh=_run_jobs(audit_worker,jobs,reg["config"]["workers"],phase="uncapped-train-audit",output_root=out/"audit-progress",
                        run_fingerprint=reg["binding"],timeout_seconds=960.)
        run.require(len(fresh)==16 and all(r["status"]=="ok" for r in fresh),"new audit failed")
        run.once(out/"batch.audit.json",run.sealed(dict(binding=reg["binding"],results=rows+fresh)))
    return dict(audited=96,reused_audits=80,new_audits=16,prefix_steps=sum(r["prefix_steps"] for r in fresh))


def audited_batch(reg,p,out):
    proof=run.check_seal(run.read_json(out/"batch.audit.json"))
    run.require(proof["binding"]==reg["binding"] and len(proof["results"])==96,"audit coverage")
    hashes={r["job_id"]:r["result_sha256"] for r in proof["results"] if r["status"]=="ok"}
    run.require(set(hashes)==set(reg["entries"]),"audit identities")
    rows=[]
    for jid,e in reg["entries"].items():
        folder,row=record(reg,p,e)
        run.require(run.sha256_file(folder/"result.json")==hashes[jid],"stale audit")
        rows.append((e,folder,row))
    return rows


def analyze():
    reg,p,out=verify()
    batch=audited_batch(reg,p,out)
    old_rows,new_rows=[],[]
    for e,folder,row in batch:
        previous=run.result_read(ROOT/e["source_folder"],p)
        def steps(path):return [{k:v[k] for k in ("decision","policy_sha256","probabilities","selected_id","behavior_log_probability")}
                               for v in run.trace_read(path)]
        new_steps=steps(folder)
        old_steps=new_steps if e["origin"]=="reuse" else new_steps[:previous["decisions"]]
        old_rows.append(dict(previous,steps=old_steps))
        new_rows.append(dict(row,steps=new_steps))
    groups={e["job"]["pair_id"]:e["job"]["case"]["map_id"] for e in reg["entries"].values()}
    common=dict(policy_sha256=reg["source_policy_sha256"],expected_groups=groups,replicas=4,node_budget=25000000)
    old_credit=run.gradient_coefficients(old_rows,max_decisions=256,**common)
    new_credit=credit.gradient_coefficients(new_rows,max_decisions=None,**common)
    change=credit.signal_change(old_rows,new_rows,old_credit,new_credit)
    def mixed(rows):
        return {g:sum(r["success"] for r in rows if r["pair_id"]==g) for g in groups
                if 0<sum(r["success"] for r in rows if r["pair_id"]==g)<4}
    body=dict(binding=reg["binding"],audit_sha256=run.sha256_file(out/"batch.audit.json"),episodes=96,reused=80,new=16,
        old_stops=dict(Counter(r["stop"] for r in old_rows)),new_stops=dict(Counter(r["stop"] for r in new_rows)),
        old_mixed_conditions=mixed(old_rows),new_mixed_conditions=mixed(new_rows),signal=change,
        old_credit=old_credit,new_credit=new_credit,new_decisions=sum(r["decisions"] for r in new_rows),
        added_decisions=sum(b["decisions"]-a["decisions"] for a,b in zip(old_rows,new_rows)),
        outcomes=[{k:r[k] for k in ("episode_id","map_id","pair_id","replica","stop","success","decisions","generated","final_conflicts")} for r in new_rows],
        no_ttf=True,no_heldout=True,no_update_yet=True)
    with old.strict_lock(out,reg["binding"],"uncapped-train-analyze"):
        run.once(out/"terminal_analysis.json",run.sealed(body))
    return {k:body[k] for k in ("old_stops","new_stops","signal","new_decisions","added_decisions")}


def analysis_gate(reg,out):
    result=run.check_seal(run.read_json(out/"terminal_analysis.json"))
    run.require(result["binding"]==reg["binding"] and result["audit_sha256"]==run.sha256_file(out/"batch.audit.json"),"analysis identity")
    return result


def extract():
    from concurrent.futures import ProcessPoolExecutor
    reg,p,out=verify()
    analysis=analysis_gate(reg,out)
    if not analysis["signal"]["possible_gradient_change"]:return dict(skipped="unchanged gradient information")
    originals=run.check_seal(run.read_json(ROOT/reg["config"]["source_cache"]))
    caches={r["episode"]["job_id"]:r for r in originals["rows"]}
    weights={r["episode_id"]:r for r in analysis["new_credit"]}
    rows,jobs=[],[]
    with old.strict_lock(out,reg["binding"],"uncapped-train-extract"):
        run.require(not (out/"cache").exists() and not (out/"cache.json").exists(),"partial extraction requires inspection")
        actor=run.actor_load(out,p,0)
        for e,folder,row in audited_batch(reg,p,out):
            jid=row["job_id"]
            if e["origin"]=="reuse":
                c=caches[jid]
                run.require(c["source_sha256"]==run.sha256_file(folder/"result.json") and
                            run.sha256_file(ROOT/c["cache"])==c["sha256"],"reused cache identity")
                rows.append(dict(c,episode=row,credit=weights[jid],reused_cache=True))
            else:jobs.append(dict(folder=folder.relative_to(ROOT).as_posix(),plan=p,actor=actor,
                                 output=reg["config"]["output"],result_sha256=run.sha256_file(folder/"result.json")))
        with ProcessPoolExecutor(max_workers=reg["config"]["workers"]) as pool:
            for r in pool.map(crossfit.extract_worker,jobs):
                r.pop("steps")
                rows.append(dict(r,credit=weights[r["episode"]["job_id"]],reused_cache=False))
        run.require(len(rows)==96 and len({r["episode"]["episode_id"] for r in rows})==96,"cache coverage")
        run.once(out/"cache.json",run.sealed(dict(binding=reg["binding"],rows=sorted(rows,key=lambda r:r["episode"]["job_id"]))))
    return dict(cached=96,reused=80,extracted=16)


def train():
    import numpy as np
    import torch
    from experiments.sa_crossfit_update import padded_episode,tensor_distribution,guarded_update
    from experiments.sa_onpolicy_actor import torch_actor
    torch.set_num_threads(1)
    reg,p,out=verify()
    analysis=analysis_gate(reg,out)
    with old.strict_lock(out,reg["binding"],"uncapped-train-update"):
        run.require(not (out/"update.json").exists() and not run.actor_file(out,1).exists(),"one fixed update only")
        if not analysis["signal"]["possible_gradient_change"]:
            result=dict(binding=reg["binding"],decision="no_new_gradient_information_skip_retraining",updated=False)
        else:
            actor=run.actor_load(out,p,0)
            run.require(torch.__version__==actor["training_library"],"frozen Torch version")
            rows=crossfit.cached(reg,out)
            run.require(all(r["episode"]["split"]=="train" and r["episode"]["policy_sha256"]==reg["source_policy_sha256"] for r,_ in rows),"Train actor-0 only")
            model=torch_actor(actor)
            gradient=[torch.zeros_like(v) for v in model.parameters()]
            packs=[]
            expected={r["episode_id"]:r for r in analysis["new_credit"]}
            for i,(r,data) in enumerate(rows):
                run.require(r["credit"]==expected[r["episode"]["episode_id"]],"training credit changed")
                padded=padded_episode(data,actor)
                logp=tensor_distribution(model,actor,padded)
                run.require(np.max(np.abs(logp.exp().detach().numpy()-padded[1]))<1e-12,"behavior replay")
                selected=logp[torch.arange(len(data["lengths"])),torch.as_tensor(data["selected"],dtype=torch.long)]
                advantage=torch.as_tensor(float(r["episode"]["success"])-r["credit"]["baseline"],dtype=torch.float64)
                loss=-(advantage*selected).sum()
                raw=torch.autograd.grad(loss,tuple(model.parameters()))
                for total,g in zip(gradient,raw):total+=r["credit"]["episode_weight"]*g
                packs.append(dict(padded=padded,weight=r["credit"]["episode_weight"],draws=data["draws"],lengths=data["lengths"],selected=data["selected"]))
                if (i+1)%12==0:print(f"Gradient {i+1}/96",flush=True)
            magnitude=sum(float((g*g).sum()) for g in gradient)**.5
            if magnitude==0:
                result=dict(binding=reg["binding"],decision="zero_gradient_no_update",updated=False)
            else:
                bundle,diagnostic=guarded_update(actor,gradient,packs,reg["config"]["update"],reg["binding"],reg["config"]["update_arm"])
                result=dict(binding=reg["binding"],updated=bundle is not None,diagnostic=diagnostic,
                    cache_sha256=run.sha256_file(out/"cache.json"),analysis_sha256=run.sha256_file(out/"terminal_analysis.json"),
                    decision="fixed_update_not_evaluated" if bundle is not None else "no_step_within_budget",
                    no_heldout=True,no_ttf=True,automatic_promotion=False)
                if bundle is not None:
                    run.publish_actor(out,p,bundle,result)
                    result.update(model_sha256=run.sha256_file(run.actor_file(out,1)),policy_sha256=run.validate_bundle(bundle))
        run.once(out/"update.json",run.sealed(result))
    return result


def parity(torch_side=False):
    from experiments.sa_onpolicy_actor import NumpyActor,torch_actor,torch_distribution
    reg,p,out=verify()
    update=run.check_seal(run.read_json(out/"update.json"))
    run.require(update["binding"]==reg["binding"] and update["updated"],"no updated actor")
    actor=run.actor_load(out,p,1)
    run.require(run.sha256_file(run.actor_file(out,1))==update["model_sha256"] and run.validate_bundle(actor)==update["policy_sha256"],"updated actor identity")
    portable=NumpyActor(actor)
    with old.strict_lock(out,reg["binding"],"uncapped-train-parity"):
        fixtures=[]
        if torch_side:
            model=torch_actor(actor)
            for e,folder,row in audited_batch(reg,p,out):
                for event in run.trace_read(folder):
                    if event["decision"] not in {0,32,128,255,256,row["decisions"]-1}:continue
                    ids=event["candidate_ids"]
                    values=torch_distribution(model,actor,ids,event["anchor_id"],event["features"]).probs.detach().numpy().tolist()
                    fixtures.append(dict(job_id=row["job_id"],decision=event["decision"],candidate_ids=ids,
                        anchor_id=event["anchor_id"],features=event["features"],probabilities=dict(zip(ids,values))))
        else:
            run.native_runtime(p)
            ref=run.check_seal(run.read_json(out/"torch-parity.json"))
            run.require(ref["binding"]==reg["binding"] and ref["policy_sha256"]==portable.sha and
                        ref["update_sha256"]==run.sha256_file(out/"update.json"),"parity identity")
            fixtures=ref["fixtures"]
        error=0.
        for f in fixtures:
            actual=portable.probabilities(f["candidate_ids"],f["anchor_id"],f["features"])
            error=max(error,max(abs(actual[k]-f["probabilities"][k]) for k in actual))
            run.require(error<=1e-12,"portable probability mismatch")
            for i in range(16):
                draw=run.random.Random(i).random()
                run.require(run.select_with_draw(actual,draw)==run.select_with_draw(f["probabilities"],draw),"portable action mismatch")
        result=dict(binding=reg["binding"],policy_sha256=portable.sha,update_sha256=run.sha256_file(out/"update.json"),
                    max_error=error,fixture_count=len(fixtures),decisions_after_256=sum(f["decision"]>=256 for f in fixtures))
        run.once(out/("torch-parity.json" if torch_side else "parity.json"),run.sealed(dict(result,fixtures=fixtures) if torch_side else result))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","collect","audit","analyze","extract","train","parity","stop"))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--torch",action="store_true")
    args=parser.parse_args()
    if args.phase=="collect":result=collect(args.resume)
    elif args.phase=="parity":result=parity(args.torch)
    elif args.phase=="verify":result=dict(binding=verify()[0]["binding"],verified=True)
    elif args.phase=="stop":
        cfg=run.read_json(ROOT/CONFIG)
        run.write_json(ROOT/cfg["output"]/"STOP_AFTER_BATCH",dict(requested=True))
        result=dict(stop_after_current_batch=True)
    else:result=globals()[args.phase]()
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
