"""Train-map-only late-state discovery and independent-trial confirmation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, json_fingerprint
from experiments.sa_paired_completion import require
from experiments.sa_policy_aligned_update import budget_features, work_stop
from experiments.sa_late_state_signal import assign_phases, choose_discovery, describe, summarize
from scripts import run_sa_policy_aligned_update as base
from scripts import recover_sa_policy_aligned_update_evaluation as parent
from scripts.run_sa_paired_closed_loop import once, sealed, check_seal
from scripts.audit_sa_history_information import atomic

CONFIG = "configs/sa_late_state_signal.json"
CODE = [CONFIG, "experiments/sa_late_state_signal.py", "scripts/run_sa_late_state_signal.py",
        "tests/evaluation/test_sa_late_state_signal.py", "docs/SA_LATE_STATE_SIGNAL_PROTOCOL_ZH.md"]


def parent_verified():
    previous = base.CONFIG
    base.CONFIG = parent.CONFIG
    try:
        return base.verify()
    finally:
        base.CONFIG = previous


def prepare():
    from scripts import run_sa_path_quality as q
    cfg = read_json(ROOT/CONFIG)
    p, folder = parent_verified()
    require(folder == ROOT/cfg["parent"] and cfg["source"] == p["config"]["source"], "parent identity")
    require(cfg["model_fits"] == 0 and not cfg["formal_ttf"], "diagnostic scope changed")
    for k in ("max_decisions", "node_budget", "pp_seconds", "branch_seconds"):
        require(cfg[k] == p["config"][k], "work/PP budget changed")
    require(cfg["continuations"] == ["frozen"] and cfg["trials"] == 16 and
            cfg["discovery_trials"] == list(range(8)) and cfg["confirmation_trials"] == list(range(8,16)), "trial contract")
    phases = assign_phases(p["split"]["train_maps"], p["split"]["validation_maps"], cfg)
    out = ROOT/cfg["output"]
    require(not out.exists(), "refuse to replace prior output")
    inputs = dict(p["inputs"])
    model = base.load_updated_model(folder, p, native=False)
    roots = []
    for episode in p["split"]["training_episode_ids"]:
        old_entry = next(r for r in p["roots"] if r["id"] == episode+"-d0000")
        old = base.root_read(old_entry, p)
        phase = phases[old["map_id"]]
        source = ROOT/cfg["source"]/"episodes"/episode
        state, prefix = old["initial"], []
        root = None
        with (source/"trace.jsonl").open(encoding="utf8") as stream:
            for line in stream:
                event = json.loads(line)
                require(event["decision"] == len(prefix) and event["before"] == q.state_fingerprint(state), "source discontinuity")
                if event["decision"] == phase:
                    used = state["low_level"]["generated"]-old["initial_nodes"]
                    require(work_stop(state["feasible"], phase, used, cfg) is None, "ineligible root; do not replace")
                    pool = {c["candidate_id"]:c for c in event["pool"]}
                    cs = [pool[c] for c in event["subset"]]
                    require(len(cs) == 4, "four-candidate coverage required")
                    fs = [budget_features(f, phase, used, cfg) for f in event["features"]]
                    ranking = model.rank(dict(anchor_id=event["anchor_id"], agent_ids=[a["id"] for a in state["agents"]],
                                              candidates=[dict(c,features=f) for c,f in zip(cs,fs,strict=True)]))
                    root = dict(id=episode+f"-d{phase:04d}", map_id=old["map_id"], case=old["case"],
                        pair_id=old["pair_id"], solver_seed=old["solver_seed"], decision=phase,
                        initial=old["initial"], state=state, prefix=prefix, source_event=event,
                        initial_nodes=old["initial_nodes"], generated_at_root=used,
                        anchor_id=event["anchor_id"], candidates=cs, features=fs, updated_ranking=ranking)
                    break
                prefix.append(event)
                state = q.apply_state_delta(state, event["delta"])
        require(root is not None, "missing registered phase; do not replace episode")
        roots.append(root)
    entries = []
    for r in roots:
        path = out/"inputs"/(r["id"]+".json")
        once(path, sealed(r))
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        entries.append({k:r[k] for k in ("id", "map_id", "decision", "pair_id", "generated_at_root")} |
            dict(file=path.relative_to(ROOT).as_posix(), candidates=[c["candidate_id"] for c in r["candidates"]]))
    for name in [*CODE, cfg["parent"]+"/plan.json", *[cfg["parent"]+"/model/"+n for n in
                 ("bundle.json", "fixtures.json", "training.json", "receipt.json")]]:
        inputs[name] = sha256_file(ROOT/name)
    plan = dict(config=cfg, source_plan=p["source_plan"], split=p["split"], roots=entries, inputs=inputs,
        parent_binding=p["binding"], cases=[r["case"] for r in roots],
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        label_jobs=len(roots)*4*cfg["trials"], maximum_label_repairs=sum((128-r["decision"])*64 for r in roots),
        replay_repairs=sum(r["decision"]*64 for r in roots), no_ttf=True, model_fits=0)
    plan["binding"] = json_fingerprint(plan)
    once(out/"plan.json", plan)
    return {k:plan[k] for k in ("binding", "label_jobs", "maximum_label_repairs", "replay_repairs")}


def jobs_for(p, stage):
    trials = p["config"][stage+"_trials"]
    return [dict(j, parent_pid=os.getpid(), plan=p) for j in base.jobs_for(p,"labels") if j["trial"] in trials]


def verify_predictions(native):
    p,out = base.verify()
    pp,folder = parent_verified()
    model = base.load_updated_model(folder, pp, native=native)
    for e in p["roots"]:
        r = base.root_read(e,p)
        state = dict(anchor_id=r["anchor_id"],agent_ids=[a["id"] for a in r["state"]["agents"]],
                     candidates=[dict(c,features=f) for c,f in zip(r["candidates"],r["features"],strict=True)])
        require(model.rank(state) == r["updated_ranking"], "registered prediction mismatch")
    base.record(out/("predictions.native.json" if native else "predictions.python.json"),
                dict(binding=p["binding"], roots=len(p["roots"]), native=native))
    return dict(prediction_parity=True, roots=len(p["roots"]), native=native)


def collect(stage, resume, limit):
    from experiments.repair_collection import _CollectionRunLock, _run_jobs
    p,out = base.verify()
    for file in ("preflight.complete.json", "predictions.native.json", "predictions.python.json"):
        require(check_seal(read_json(out/file))["binding"] == p["binding"], "preflight/prediction check missing")
    if stage == "confirmation":
        choice = check_seal(read_json(out/"choices.json"))
        require(choice["binding"] == p["binding"] and choice["complete"], "freeze discovery first")
        require(choice["discovery_audit_sha256"] == sha256_file(out/"discovery.audit.json"), "discovery audit changed")
        start = dict(binding=p["binding"], choice_sha256=sha256_file(out/"choices.json"))
        path = out/"confirmation.start.json"
        if path.exists(): require(read_json(path) == start, "confirmation choice changed")
        else: once(path,start)
    jobs = jobs_for(p,stage)
    if limit is not None:
        require(0 < limit <= len(jobs), "limit out of range")
        jobs = jobs[:limit]
    pending = []
    for job in jobs:
        folder = out/"labels"/job["job_id"]
        if (folder/"result.json").exists():
            require(resume, "resume required")
            base.check_result(folder,p)
        else:
            require(not folder.exists(), "partial job requires interruption audit")
            pending.append(job)
    with _CollectionRunLock(out,p["binding"],stage):
        atomic(out/"run_status.json",dict(stage=stage,status="running",pending=len(pending)))
        try:
            for offset in range(0,len(pending),p["config"]["workers"]):
                if (out/"STOP_AFTER_BATCH").exists(): break
                batch = pending[offset:offset+p["config"]["workers"]]
                def progress(row):
                    with (out/"progress.jsonl").open("a",encoding="utf8") as f:
                        f.write(json.dumps(dict(stage=stage,**row))+"\n")
                    print(stage,row,flush=True)
                results = _run_jobs(base.worker,batch,p["config"]["workers"],phase=stage,
                    output_root=out/"progress"/f"{stage}-{offset}",run_fingerprint=p["binding"],
                    timeout_seconds=p["config"]["fuse_seconds"],on_result=progress,
                    failure_result=base.failure,stop_on_failure=True)
                require(len(results)==len(batch) and all(r["status"]=="ok" for r in results), "job error; inspect before resume")
            all_jobs = jobs_for(p,stage)
            done = sum((out/"labels"/j["job_id"]/"result.json").exists() for j in all_jobs)
            if done == len(all_jobs):
                for j in all_jobs: base.check_result(out/"labels"/j["job_id"],p)
                base.record(out/(stage+".complete.json"),dict(binding=p["binding"],jobs=done))
            atomic(out/"run_status.json",dict(stage=stage,status="complete" if done==len(all_jobs) else "paused",done=done,total=len(all_jobs)))
        except BaseException as exc:
            atomic(out/"run_status.json",dict(stage=stage,status="error",error=repr(exc)))
            raise
    return dict(stage=stage,done=done,total=len(all_jobs))


def audit_stage(p,out,stage):
    require(check_seal(read_json(out/(stage+".complete.json")))["binding"] == p["binding"], "stage incomplete")
    jobs = jobs_for(p,stage)
    path = out/(stage+".audit.json")
    if path.exists():
        audit = check_seal(read_json(path))
        require(audit["binding"] == p["binding"] and len(audit["results"]) == len(jobs), "audit identity")
        for j in jobs:
            require(audit["files"][j["job_id"]] == sha256_file(out/"labels"/j["job_id"]/"result.json"), "result changed")
            base.check_result(out/"labels"/j["job_id"],p)
        return audit["results"]
    with ProcessPoolExecutor(max_workers=p["config"]["workers"]) as pool:
        results = []
        for i,r in enumerate(pool.map(base.audit_job,jobs),1):
            results.append(r)
            if i % 20 == 0 or i == len(jobs): print(stage,"audited",i,"/",len(jobs),flush=True)
    base.record(path,dict(binding=p["binding"],results=results,
        files={j["job_id"]:sha256_file(out/"labels"/j["job_id"]/"result.json") for j in jobs}))
    return results


def freeze():
    p,out = base.verify()
    require(not (out/"confirmation.start.json").exists(), "cannot select after confirmation starts")
    results = audit_stage(p,out,"discovery")
    roots = [base.root_read(e,p) for e in p["roots"]]
    choices = [choose_discovery(r,[x for x in results if x["root_id"]==r["id"]],p["config"]) for r in roots]
    data = dict(binding=p["binding"],choices=choices,complete=all(c["complete"] for c in choices),
                discovery_audit_sha256=sha256_file(out/"discovery.audit.json"),model_fits=0)
    base.record(out/"choices.json",data)
    return dict(complete=data["complete"],changed=sum(c["selected"]!=r["anchor_id"] for c,r in zip(choices,roots)),roots=len(roots))


def analyze():
    p,out = base.verify()
    choices = check_seal(read_json(out/"choices.json"))
    require(read_json(out/"confirmation.start.json")["choice_sha256"]==sha256_file(out/"choices.json"), "choice changed")
    require(choices["binding"]==p["binding"] and choices["discovery_audit_sha256"]==sha256_file(out/"discovery.audit.json"), "choice identity")
    audit_stage(p,out,"discovery")
    results = audit_stage(p,out,"confirmation")
    roots = [base.root_read(e,p) for e in p["roots"]]
    by_id = {c["root_id"]:c for c in choices["choices"]}
    rows = [describe(r,by_id[r["id"]],[x for x in results if x["root_id"]==r["id"]],p["config"]) for r in roots]
    summary = summarize(rows,p["config"])
    base.record(out/"report.json",dict(binding=p["binding"],summary=summary,roots=rows,
        evidence={n:sha256_file(out/n) for n in ("plan.json","choices.json","confirmation.start.json","discovery.audit.json","confirmation.audit.json")},
        train_maps_only=True,no_ttf=True,model_fits=0))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase",choices=("prepare","verify","dry-run","verify-predictions","preflight","discovery","freeze","confirmation","analyze","request-stop"))
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--limit",type=int)
    ap.add_argument("--python-portable",action="store_true")
    args = ap.parse_args()
    base.CONFIG = CONFIG
    if args.phase == "prepare": result = prepare()
    elif args.phase in ("verify","dry-run"):
        p,_=base.verify(); result={k:p[k] for k in ("binding","label_jobs","maximum_label_repairs","replay_repairs")}
    elif args.phase == "verify-predictions": result=verify_predictions(not args.python_portable)
    elif args.phase == "preflight": result=base.collect("preflight",args.resume,args.limit)
    elif args.phase in ("discovery","confirmation"): result=collect(args.phase,args.resume,args.limit)
    elif args.phase == "freeze": result=freeze()
    elif args.phase == "analyze": result=analyze()
    else:
        _,out=base.verify(); once(out/"STOP_AFTER_BATCH",dict(requested=True)); result=dict(safe_stop=True)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == "__main__":
    main()
