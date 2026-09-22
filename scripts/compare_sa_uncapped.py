"""Remove only the execution decision cap; rerun the same frozen 80 episodes."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import compare_sa_crossfit as prior
from scripts import run_sa_onpolicy as run
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_uncapped_runtime as runtime

CONFIG = "configs/sa_crossfit_uncapped.json"
CODE = (CONFIG, "scripts/compare_sa_uncapped.py", "experiments/sa_uncapped_runtime.py",
        "tests/evaluation/test_sa_uncapped.py", "docs/SA_UNCAPPED_PROTOCOL_ZH.md")


def configuration(old):
    override = run.read_json(ROOT/CONFIG)
    run.require((override["max_decisions"], override["decision_feature_reference"], override["node_budget"],
                 override["pp_safety_seconds"], override["episode_safety_seconds"], override["process_fuse_seconds"],
                 override["workers"], override["formal_ttf"], override["automatic_promotion"]) ==
                (None, 256, 25000000, 20., 900., 960., 20, False, False), "uncapped preregistration")
    run.require(override["source"] == old["config"]["output"], "source comparison identity")
    return dict(old["config"], **override)


def runtime_plan(source, cfg, arm):
    plan = prior.runtime_plan(source, cfg, arm)
    return dict(plan, proposal=dict(plan["proposal"], max_decisions=None,
                decision_feature_reference=cfg["decision_feature_reference"], node_budget=cfg["node_budget"]))


def prepare():
    old, source, old_out = prior.verify()
    cfg = configuration(old)
    out = ROOT/cfg["output"]
    run.require(not out.exists(), "existing output; use verify/resume")
    run.require(source["template"]["environment"]["max_repair_iterations"] == 0, "hidden native cap")
    completion = run.check_seal(run.read_json(old_out/"collection.complete.json"))
    audit = run.check_seal(run.read_json(old_out/"audit.json"))
    run.require(completion["jobs"] == 80 and audit["binding"] == old["binding"] and not audit["missing"], "source complete audit")
    audited = {r["job_id"]:r["result_sha256"] for r in audit["results"]}
    inputs = {name:run.sha256_file(ROOT/name) for name in CODE}
    for name in ("registration.json", "collection.complete.json", "audit.json", "report.json"):
        inputs[(old_out/name).relative_to(ROOT).as_posix()] = run.sha256_file(old_out/name)
    for j in old["jobs"]:
        row = prior.read_result(old, source, j)
        folder = prior.folder_for(old["config"], j)
        run.require(row["status"] == "ok" and run.sha256_file(folder/"result.json") ==
                    completion["files"][j["job_id"]] == audited[j["job_id"]], "source result audit")
        for name in ("result.json", "comparison_receipt.json", *row["files"]):
            inputs[(folder/name).relative_to(ROOT).as_posix()] = run.sha256_file(folder/name)
    for spec in cfg["models"].values():
        inputs[spec["path"]] = spec["file_sha256"]
    body = dict(schema=cfg["schema"], config=cfg, inputs=inputs, jobs=old["jobs"],
                source_binding=old["binding"], scientific_binding=source["binding"],
                source_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                no_training=True, no_ttf=True, role="viewed_development_holdout")
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out, body["binding"], "uncapped-prepare"):
        for arm in cfg["models"]:
            bundle = prior.check_model(cfg, arm, source["binding"])
            arm_out = ROOT/runtime_plan(source, cfg, arm)["config"]["output"]
            target = run.actor_file(arm_out, bundle["iteration"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/cfg["models"][arm]["path"], target)
            run.require(run.sha256_file(target) == cfg["models"][arm]["file_sha256"], "model copy")
            run.once(arm_out/"models"/f"receipt-{bundle['iteration']}.json", run.sealed(dict(
                binding=source["binding"], policy_sha256=run.validate_bundle(bundle), file_sha256=run.sha256_file(target),
                metadata=dict(comparison_binding=body["binding"], comparison_arm=arm, copied_not_trained=True))))
        run.once(out/"registration.json", run.sealed(body))
    return dict(registered=True, jobs=80, binding=body["binding"], max_decisions=None, no_ttf=True)


def verify():
    old, source, _ = prior.verify()
    cfg = configuration(old)
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/"registration.json"))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding", "integrity")}), "registration hash")
    run.require(reg["config"] == cfg and reg["jobs"] == old["jobs"] and reg["source_binding"] == old["binding"] and
                reg["scientific_binding"] == source["binding"], "execution identity")
    for name, digest in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT, name, field="uncapped input")) == digest, f"changed input: {name}")
    for arm in cfg["models"]:
        p = runtime_plan(source,cfg,arm)
        bundle = run.actor_load(ROOT/p["config"]["output"], p, cfg["models"][arm]["iteration"])
        run.require(bundle == prior.check_model(cfg, arm, source["binding"]), "copied model differs")
    return reg, source, out


def read_result(reg, source, job):
    row = prior.read_result(reg, source, job)
    run.require(row["execution_binding"] == reg["binding"], "wrong execution contract")
    runtime.validate_terminal(row, runtime_plan(source,reg["config"],job["comparison_arm"])["proposal"])
    return row


def worker(job):
    row = runtime.episode_worker(job)
    folder = prior.folder_for(job["comparison_config"], job)
    run.once(folder/"comparison_receipt.json", run.sealed(dict(comparison_binding=job["comparison_binding"],
                comparison_arm=job["comparison_arm"], result_sha256=run.sha256_file(folder/"result.json"))))
    return dict(row, comparison_arm=job["comparison_arm"])


def augmented(reg, source, j):
    return dict(j, plan=runtime_plan(source, reg["config"], j["comparison_arm"]), parent_pid=os.getpid(),
                comparison_config=reg["config"], comparison_binding=reg["binding"])


def collect(resume=False):
    from experiments.repair_collection import _run_jobs
    run.require(os.name != "nt", "use frozen WSL native")
    reg, source, out = verify()
    cfg = reg["config"]
    with recovery.strict_lock(out, reg["binding"], "uncapped-collect"):
        run.require(resume or not (out/"run_status.json").exists(), "explicit resume required")
        pending = []
        for j in reg["jobs"]:
            folder = prior.folder_for(cfg,j)
            run.require(not (out/"failures"/(j["job_id"]+".json")).exists(), "failure needs inspection")
            if (folder/"result.json").exists():
                run.require(read_result(reg,source,j)["status"] == "ok", "censored episode needs inspection")
            else:
                run.require(not folder.exists(), "partial episode needs inspection")
                pending.append(augmented(reg,source,j))
        if resume:
            (out/"STOP_AFTER_BATCH").unlink(missing_ok=True)
        def progress(row):
            with (out/"progress.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
        def failure(job,status,error):
            row = dict(status="censored" if status == "timeout" else "error", job_id=job["job_id"],
                       comparison_arm=job["comparison_arm"], stop="external_timeout" if status == "timeout" else "exception", error=error)
            run.once(out/"failures"/(job["job_id"]+".json"), run.sealed(dict(binding=reg["binding"], **row)))
            return row
        run.write_json(out/"run_status.json", dict(status="running", pending=len(pending), binding=reg["binding"]))
        try:
            for offset in range(0,len(pending),cfg["workers"]):
                if (out/"STOP_AFTER_BATCH").exists():
                    run.write_json(out/"run_status.json",dict(status="paused",binding=reg["binding"]))
                    return dict(status="paused")
                batch = pending[offset:offset+cfg["workers"]]
                rows = _run_jobs(worker,batch,cfg["workers"],phase="uncapped-collect",timeout_seconds=cfg["process_fuse_seconds"],
                    output_root=out/"progress"/batch[0]["job_id"],run_fingerprint=reg["binding"],on_result=progress,
                    failure_result=failure,stop_on_failure=False)
                if len(rows) != len(batch) or any(r["status"] != "ok" for r in rows):
                    run.write_json(out/"run_status.json",dict(status="needs_inspection",binding=reg["binding"]))
                    return dict(status="needs_inspection",no_automatic_retry=True)
            records = [read_result(reg,source,j) for j in reg["jobs"]]
            run.require(all(r["status"] == "ok" for r in records), "incomplete collection")
            run.once(out/"collection.complete.json",run.sealed(dict(binding=reg["binding"],jobs=len(records),
                files={j["job_id"]:run.sha256_file(prior.folder_for(cfg,j)/"result.json") for j in reg["jobs"]})))
            run.write_json(out/"run_status.json",dict(status="completed",jobs=len(records),binding=reg["binding"]))
        except BaseException as exc:
            run.write_json(out/"run_status.json",dict(status="interrupted_or_error",error=repr(exc),binding=reg["binding"]))
            raise
    return dict(status="completed",jobs=len(records),no_ttf=True)


def audit():
    from experiments.repair_collection import _run_jobs
    reg,source,out=verify()
    complete=run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(complete["binding"] == reg["binding"] and complete["jobs"] == len(reg["jobs"]), "collection coverage")
    for j in reg["jobs"]:
        read_result(reg,source,j)
        run.require(complete["files"][j["job_id"]] == run.sha256_file(prior.folder_for(reg["config"],j)/"result.json"), "collection result changed")
    with recovery.strict_lock(out,reg["binding"],"uncapped-audit"):
        rows=_run_jobs(runtime.audit_worker,[augmented(reg,source,j) for j in reg["jobs"]],reg["config"]["workers"],
            phase="uncapped-audit",output_root=out/"audit-progress",run_fingerprint=reg["binding"],timeout_seconds=960.)
        run.require(len(rows)==80 and all(r["status"]=="ok" for r in rows), "audit failed")
        run.once(out/"audit.json",run.sealed(dict(binding=reg["binding"],results=rows)))
    return dict(audited=len(rows))


def prefix_check(old_events, new_events, before, after):
    new_events=iter(new_events)
    count=0
    for old in old_events:
        new=next(new_events,None)
        run.require(new is not None and {k:v for k,v in old.items() if k != "metrics"} ==
                    {k:v for k,v in new.items() if k != "metrics"}, "old action/path/feature prefix changed")
        count+=1
    run.require(count == before["decisions"], "source trace length")
    if before["stop"] == "decision_budget":
        run.require(after["decisions"] > count, "old cap not removed")
    else:
        run.require(next(new_events,None) is None and all(before[k]==after[k] for k in
            ("stop","success","decisions","generated","final_fingerprint","soc","makespan","wait_steps")), "unaffected episode changed")
    return count


def report():
    reg,source,out=verify()
    receipt=run.check_seal(run.read_json(out/"audit.json"))
    run.require(receipt["binding"]==reg["binding"] and len(receipt["results"])==80,"audit coverage")
    hashes={r["job_id"]:r["result_sha256"] for r in receipt["results"]}
    old=run.check_seal(run.read_json(ROOT/reg["config"]["source"]/"registration.json"))
    data={a:{} for a in prior.ARMS}
    prefixes=[]
    for j in reg["jobs"]:
        row=read_result(reg,source,j)
        folder=prior.folder_for(reg["config"],j)
        run.require(hashes[j["job_id"]]==run.sha256_file(folder/"result.json"),"stale audit")
        previous=prior.read_result(old,source,j)
        n=prefix_check(run.trace_read(prior.folder_for(old["config"],j)),run.trace_read(folder),previous,row)
        prefixes.append(dict(job_id=j["job_id"],arm=j["comparison_arm"],prefix_steps=n,previous_stop=previous["stop"],
                             current_stop=row["stop"],previous_decisions=previous["decisions"],current_decisions=row["decisions"]))
        data[j["comparison_arm"]][(j["pair_id"],j["replica"])]=row
    for key in data[prior.ARMS[0]]:
        run.require(len({data[a][key]["initial_fingerprint"] for a in prior.ARMS})==1 and
                    len({data[a][key]["rng_stream_id"] for a in prior.ARMS})==1,"cross-arm pairing")
    summaries={a:dict(episodes=len(rows),success=sum(r["success"] for r in rows.values()),
        stops=dict(Counter(r["stop"] for r in rows.values())),beyond_training_horizon=sum(r["decisions"]>256 for r in rows.values()),
        max_decisions_observed=max(r["decisions"] for r in rows.values())) for a,rows in data.items()}
    maps=sorted({r["map_id"] for r in data[prior.ARMS[0]].values()})
    comparisons={}
    for i,arm in enumerate(prior.ARMS):
        for base in prior.ARMS[:i]:
            a,b=data[arm],data[base]
            comparisons[f"{arm}_vs_{base}"]=dict(overall=prior.paired_comparison(a,b),by_map={m:prior.paired_comparison(
                {k:v for k,v in a.items() if v["map_id"]==m},{k:v for k,v in b.items() if v["map_id"]==m}) for m in maps})
    body=dict(schema="lns2.sa.uncapped_report.v1",binding=reg["binding"],summaries=summaries,comparisons=comparisons,
        prefix_equivalence=prefixes,episodes=[dict(comparison_arm=a,**r) for a,rows in data.items() for r in rows.values()],
        legal_noops=sum(r["legal_noops"] for r in receipt["results"]),max_decisions=None,node_budget=25000000,
        decision_feature_reference=256,no_ttf=True,no_training=True,automatic_promotion=False,
        decision="uncapped_development_comparison_only",independent_generalization=False)
    with recovery.strict_lock(out,reg["binding"],"uncapped-report"):
        run.once(out/"report.json",run.sealed(body))
    return dict(summaries=summaries,prefixes_verified=len(prefixes),no_ttf=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","verify","collect","audit","report","stop"))
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    if args.phase=="collect": result=collect(args.resume)
    elif args.phase=="verify":
        reg,_,_=verify()
        result=dict(verified=True,binding=reg["binding"],jobs=len(reg["jobs"]),max_decisions=None)
    elif args.phase=="stop":
        cfg=run.read_json(ROOT/CONFIG)
        run.write_json(ROOT/cfg["output"]/"STOP_AFTER_BATCH",dict(requested=True))
        result=dict(safe_stop_after_current_batch=True)
    else: result=globals()[args.phase]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
