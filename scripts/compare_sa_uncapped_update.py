"""Evaluate one frozen uncapped actor; reuse verified baselines without rerunning them."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import compare_sa_uncapped as previous
from scripts import recover_sa_uncapped_training as training
from scripts import run_sa_onpolicy as run
from scripts import recover_sa_onpolicy as recovery
from scripts.register_sa_crossfit_report_fix import first_difference
from experiments import sa_uncapped_runtime as runtime

CONFIG="configs/sa_uncapped_update_comparison.json"
ARM="uncapped_condition"
CODE=(CONFIG,"scripts/compare_sa_uncapped_update.py","tests/evaluation/test_sa_uncapped_update_comparison.py",
      "docs/SA_UNCAPPED_UPDATE_COMPARISON_PROTOCOL_ZH.md","scripts/register_sa_crossfit_report_fix.py")


def configuration(old, override):
    run.require((override["arm"],override["expected_new"],override["expected_reused"],override["max_decisions"],
        override["decision_feature_reference"],override["node_budget"],override["pp_safety_seconds"],
        override["episode_safety_seconds"],override["process_fuse_seconds"],override["workers"],
        override["formal_ttf"],override["training"],override["automatic_promotion"]) ==
        (ARM,16,80,None,256,25000000,20.,900.,960.,20,False,False,False),"registered comparison scope")
    run.require(override["source_evaluation"]==old["config"]["output"],"baseline source")
    spec=dict(path=override["source_training"]+"/models/actor-1.json",file_sha256=override["model_sha256"],
              policy_sha256=override["policy_sha256"],iteration=1)
    return dict(old["config"],**override,arms=[ARM],models={ARM:spec})


def schedule(old, cfg):
    base=[j for j in old["jobs"] if j["comparison_arm"]=="untrained_exploration"]
    run.require(len(base)==16 and len({j["pair_id"] for j in base})==8 and
                len({j["case"]["map_id"] for j in base})==2,"paired development scope")
    run.require(all(j["split"]=="development_holdout" and j["phase"]==cfg["phase"] for j in base),"heldout phase")
    run.require(Counter(j["replica"] for j in base)=={0:8,1:8},"replica scope")
    return [dict(j,comparison_arm=ARM,arm="trained_actor",iteration=1,
        job_id=run.json_fingerprint([j["phase"],j["pair_id"],j["replica"],ARM])[:24]) for j in base]


def check_training(cfg, scientific_binding):
    reg,p,out=training.verify()
    run.require(out==ROOT/cfg["source_training"] and p["binding"]==scientific_binding,"training source identity")
    update=run.check_seal(run.read_json(out/"update.json"))
    parity=run.check_seal(run.read_json(out/"parity.json"))
    run.require(run.sha256_file(out/"update.json")==cfg["update_sha256"] and update["updated"] and
                update["binding"]==reg["binding"],"frozen update")
    run.require(parity["binding"]==reg["binding"] and parity["update_sha256"]==cfg["update_sha256"] and
                parity["policy_sha256"]==cfg["policy_sha256"] and parity["max_error"]<=1e-12,"frozen parity")
    actor=run.actor_load(out,p,1)
    run.require(run.sha256_file(run.actor_file(out,1))==cfg["model_sha256"] and
                run.validate_bundle(actor)==cfg["policy_sha256"] and actor["prototype_arm"]==ARM and
                actor["update_binding"]==reg["binding"] and actor["parent_policy"]==reg["source_policy_sha256"],"model identity")
    return reg,actor,out


def source_results(old, source, out):
    complete=run.check_seal(run.read_json(out/"collection.complete.json"))
    audit=run.check_seal(run.read_json(out/"audit.json"))
    run.require(complete["binding"]==audit["binding"]==old["binding"] and complete["jobs"]==80 and
                len(audit["results"])==len(old["jobs"])==80,"complete baseline evidence")
    hashes={r["job_id"]:r["result_sha256"] for r in audit["results"] if r["status"]=="ok"}
    run.require(set(hashes)==set(complete["files"])=={j["job_id"] for j in old["jobs"]},"baseline audit IDs")
    rows=[]
    for j in old["jobs"]:
        folder=previous.prior.folder_for(old["config"],j)
        row=previous.read_result(old,source,j)
        run.require(row["status"]=="ok" and run.sha256_file(folder/"result.json")==hashes[j["job_id"]]==
                    complete["files"][j["job_id"]],"baseline result changed")
        rows.append((j,folder,row))
    return rows


def prepare():
    old,source,baseline_out=previous.verify()
    cfg=configuration(old,run.read_json(ROOT/CONFIG))
    treg,actor,tout=check_training(cfg,source["binding"])
    jobs=schedule(old,cfg)
    train_maps={e["job"]["case"]["map_id"] for e in treg["entries"].values()}
    run.require(not train_maps & {j["case"]["map_id"] for j in jobs},"training map leakage")
    out=ROOT/cfg["output"]
    run.require(not out.exists(),"existing comparison; use verify/resume")
    inputs={n:run.sha256_file(ROOT/n) for n in CODE}
    refs=[]
    for j,folder,row in source_results(old,source,baseline_out):
        refs.append(dict(job=j,folder=folder.relative_to(ROOT).as_posix(),sha256=run.sha256_file(folder/"result.json")))
        for name in ("result.json","comparison_receipt.json",*row["files"]):
            path=folder/name
            inputs[path.relative_to(ROOT).as_posix()]=run.sha256_file(path)
    for parent,names in ((baseline_out,("registration.json","collection.complete.json","audit.json","report.json")),
                         (tout,("registration.json","update.json","parity.json","models/actor-1.json","models/receipt-1.json"))):
        for name in names:
            path=parent/name
            inputs[path.relative_to(ROOT).as_posix()]=run.sha256_file(path)
    body=dict(schema=cfg["schema"],config=cfg,inputs=inputs,jobs=jobs,references=refs,
        scientific_binding=source["binding"],source_binding=old["binding"],training_binding=treg["binding"],
        source_commit=run.subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        role="viewed_development_holdout",no_training=True,no_ttf=True)
    body["binding"]=run.json_fingerprint(body)
    with recovery.strict_lock(out,body["binding"],"uncapped-update-prepare"):
        p=previous.runtime_plan(source,cfg,ARM)
        target=run.actor_file(ROOT/p["config"]["output"],1)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(run.actor_file(tout,1),target)
        run.require(run.sha256_file(target)==cfg["model_sha256"],"model copy")
        run.once(target.parent/"receipt-1.json",run.sealed(dict(binding=source["binding"],
            policy_sha256=run.validate_bundle(actor),file_sha256=run.sha256_file(target),
            metadata=dict(comparison_binding=body["binding"],copied_not_trained=True))))
        run.once(out/"registration.json",run.sealed(body))
    return dict(registered=True,new_jobs=16,reused=80,max_decisions=None,binding=body["binding"])


def verify():
    old,source,_=previous.verify()
    cfg=configuration(old,run.read_json(ROOT/CONFIG))
    treg,actor,_=check_training(cfg,source["binding"])
    out=ROOT/cfg["output"]
    reg=run.check_seal(run.read_json(out/"registration.json"))
    run.require(reg["binding"]==run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding","integrity")}),"registration hash")
    run.require(reg["config"]==cfg and reg["jobs"]==schedule(old,cfg) and reg["source_binding"]==old["binding"] and
                reg["scientific_binding"]==source["binding"] and reg["training_binding"]==treg["binding"],"comparison identity")
    for name,digest in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT,name,field="comparison input"))==digest,"changed input: "+name)
    p=previous.runtime_plan(source,cfg,ARM)
    run.require(run.actor_load(ROOT/p["config"]["output"],p,1)==actor,"copied model differs")
    return reg,source,out,old


def collect(resume=False):
    from experiments.repair_collection import _run_jobs
    run.require(os.name!="nt","use frozen WSL native")
    reg,source,out,_=verify()
    with recovery.strict_lock(out,reg["binding"],"uncapped-update-collect"):
        p=previous.runtime_plan(source,reg["config"],ARM)
        pending=previous.prior.pending_jobs(reg,p,out,resume)
        if resume:(out/"STOP_AFTER_BATCH").unlink(missing_ok=True)
        if (out/"STOP_AFTER_BATCH").exists():return dict(status="paused")
        def progress(row):
            with (out/"progress.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(row)+"\n")
            print(json.dumps(row),flush=True)
        def failure(j,status,error):
            row=dict(status="censored" if status=="timeout" else "error",job_id=j["job_id"],error=error)
            run.once(out/"failures"/(j["job_id"]+".json"),run.sealed(dict(binding=reg["binding"],**row)))
            return row
        run.write_json(out/"run_status.json",dict(status="running",pending=len(pending),binding=reg["binding"]))
        try:
            rows=_run_jobs(previous.worker,pending,reg["config"]["workers"],phase="uncapped-update",output_root=out/"progress",
                run_fingerprint=reg["binding"],timeout_seconds=960.,on_result=progress,failure_result=failure,stop_on_failure=False) if pending else []
            if len(rows)!=len(pending) or any(r["status"]!="ok" for r in rows):
                run.write_json(out/"run_status.json",dict(status="needs_inspection",binding=reg["binding"]))
                return dict(status="needs_inspection",no_automatic_retry=True)
            files={}
            for j in reg["jobs"]:
                run.require(previous.read_result(reg,source,j)["status"]=="ok","unknown terminal")
                files[j["job_id"]]=run.sha256_file(previous.prior.folder_for(reg["config"],j)/"result.json")
            run.once(out/"collection.complete.json",run.sealed(dict(binding=reg["binding"],jobs=16,files=files)))
            run.write_json(out/"run_status.json",dict(status="collected",jobs=16,binding=reg["binding"]))
        except BaseException as exc:
            run.write_json(out/"run_status.json",dict(status="interrupted_or_error",binding=reg["binding"],error=repr(exc)))
            raise
    return dict(collected=16,reused=80,no_training=True,no_ttf=True)


def audit():
    from experiments.repair_collection import _run_jobs
    reg,source,out,_=verify()
    complete=run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(complete["binding"]==reg["binding"] and complete["jobs"]==16 and
                set(complete["files"])=={j["job_id"] for j in reg["jobs"]},"collection coverage")
    for j in reg["jobs"]:
        run.require(previous.read_result(reg,source,j)["status"]=="ok" and
                    run.sha256_file(previous.prior.folder_for(reg["config"],j)/"result.json")==complete["files"][j["job_id"]],"collection changed")
    with recovery.strict_lock(out,reg["binding"],"uncapped-update-audit"):
        rows=_run_jobs(runtime.audit_worker,[previous.augmented(reg,source,j) for j in reg["jobs"]],20,
            phase="uncapped-update-audit",output_root=out/"audit-progress",run_fingerprint=reg["binding"],timeout_seconds=960.)
        run.require(len(rows)==16 and all(r["status"]=="ok" for r in rows),"audit failed")
        run.once(out/"audit.json",run.sealed(dict(binding=reg["binding"],results=rows)))
    return dict(audited=16)


def pair_key(row):
    return row["pair_id"],row["replica"]


def validate_pairs(data):
    keys=set(data[ARM])
    run.require(len(keys)==16 and all(set(rows)==keys for rows in data.values()),"paired denominator")
    for key in keys:
        rows=[r[key] for r in data.values()]
        run.require(len({r["initial_fingerprint"] for r in rows})==len({r["rng_stream_id"] for r in rows})==1,"paired initial/RNG")
        run.require(all(r["split"]=="development_holdout" and r["status"]=="ok" for r in rows),"complete heldout only")
        run.require(len({r["map_id"] for r in rows})==1,"paired map identity")


def interpretation(comparisons):
    delta={a:v["overall"]["net_success_bounds"] for a,v in comparisons.items()}
    run.require(all(low==high for low,high in delta.values()),"unknown outcomes cannot guide continuation")
    old,dual,explore=(delta[a][0] for a in ("bounded_condition","dual16_sa","untrained_exploration"))
    if old>0 and dual>=0 and explore>0:return "development_net_gain_needs_fresh_stream_confirmation"
    if old<0 and dual<0:return "development_regression_do_not_promote"
    return "mixed_or_no_net_gain_do_not_promote"


def report():
    reg,source,out,old=verify()
    proof=run.check_seal(run.read_json(out/"audit.json"))
    run.require(proof["binding"]==reg["binding"] and len(proof["results"])==16,"audit coverage")
    hashes={r["job_id"]:r["result_sha256"] for r in proof["results"] if r["status"]=="ok"}
    run.require(set(hashes)=={j["job_id"] for j in reg["jobs"]},"audit identities")
    data={a:{} for a in (*previous.prior.ARMS,ARM)}
    folders={a:{} for a in data}
    for j,folder,row in source_results(old,source,ROOT/reg["config"]["source_evaluation"]):
        data[j["comparison_arm"]][pair_key(row)]=row
        folders[j["comparison_arm"]][pair_key(row)]=folder
    for j in reg["jobs"]:
        row=previous.read_result(reg,source,j)
        folder=previous.prior.folder_for(reg["config"],j)
        run.require(run.sha256_file(folder/"result.json")==hashes[j["job_id"]],"stale audit")
        data[ARM][pair_key(row)]=row
        folders[ARM][pair_key(row)]=folder
    validate_pairs(data)
    maps=sorted({r["map_id"] for r in data[ARM].values()})
    comparisons={base:dict(overall=previous.prior.paired_comparison(data[ARM],data[base]),
        by_map={m:previous.prior.paired_comparison({k:r for k,r in data[ARM].items() if r["map_id"]==m},
                                                {k:r for k,r in data[base].items() if r["map_id"]==m}) for m in maps})
        for base in previous.prior.ARMS}
    changes={}
    for base in ("bounded_condition","untrained_exploration"):
        rows=[]
        for key,left in data[ARM].items():
            right=data[base][key]
            common,first=first_difference(run.trace_read(folders[ARM][key]),run.trace_read(folders[base][key]))
            if first is None:
                run.require(left["decisions"]==right["decisions"] and left["final_fingerprint"]==right["final_fingerprint"],"unexplained divergence")
            rows.append(dict(pair_id=key[0],replica=key[1],map_id=left["map_id"],common_prefix_decisions=common,
                first_difference=first,success=left["success"],base_success=right["success"],
                final_conflicts=left["final_conflicts"],base_final_conflicts=right["final_conflicts"]))
        changes[base]=dict(changed_trajectories=sum(r["first_difference"] is not None for r in rows),pairs=rows)
    summaries={a:dict(episodes=len(rows),success=sum(r["success"] for r in rows.values()),
        stops=dict(Counter(r["stop"] for r in rows.values()))) for a,rows in data.items()}
    result=dict(schema="lns2.sa.uncapped_update_comparison_report.v1",binding=reg["binding"],
        audit_sha256=run.sha256_file(out/"audit.json"),summaries=summaries,comparisons=comparisons,behavior=changes,
        episodes=[dict(comparison_arm=a,**r) for a,rows in data.items() for r in rows.values()],
        decision=interpretation(comparisons),new_episodes=16,reused_episodes=80,max_decisions=None,node_budget=25000000,
        no_training=True,no_ttf=True,automatic_promotion=False,independent_generalization=False,
        role="viewed_development_holdout",uncertainty="two previously viewed maps; no confirmatory bootstrap claim")
    with recovery.strict_lock(out,reg["binding"],"uncapped-update-report"):
        run.once(out/"report.json",run.sealed(result))
        run.write_json(out/"run_status.json",dict(status="completed",new_episodes=16,reused=80,binding=reg["binding"],
                                                  report_sha256=run.sha256_file(out/"report.json")))
    return dict(summaries=summaries,comparisons={k:v["overall"] for k,v in comparisons.items()},decision=result["decision"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","collect","audit","report","stop"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.phase=="collect":result=collect(args.resume)
    elif args.phase=="verify":result=dict(binding=verify()[0]["binding"],verified=True)
    elif args.phase=="stop":
        cfg=run.read_json(ROOT/CONFIG)
        run.write_json(ROOT/cfg["output"]/"STOP_AFTER_BATCH",dict(requested=True))
        result=dict(stop_after_current_batch=True)
    else:result=globals()[args.phase]()
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
