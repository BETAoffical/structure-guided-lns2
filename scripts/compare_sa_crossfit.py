"""Frozen crossfit forks: bounded development completion, never TTF or fitting."""
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
from scripts import recover_sa_onpolicy as recovery
from scripts import run_sa_crossfit_update as crossfit
from experiments.sa_paired_completion import require
from experiments import sa_onpolicy_noop_runtime as noop

CONFIG = "configs/sa_crossfit_comparison.json"
CODE = (CONFIG, "scripts/compare_sa_crossfit.py", "tests/evaluation/test_sa_crossfit_comparison.py",
        "docs/SA_CROSSFIT_COMPARISON_PROTOCOL_ZH.md", "experiments/sa_onpolicy_noop_runtime.py")
ARMS = ("official_sa", "dual16_sa", "untrained_exploration", "bounded_condition", "bounded_state")


def engine_arm(arm):
    require(arm in ARMS, "unknown comparison arm")
    return "trained_actor" if arm.startswith("bounded_") else arm


def runtime_plan(source, cfg, arm):
    # Model/scientific identity is unchanged; the enclosing registration binds this comparison.
    require(arm in cfg["arms"], "arm not registered")
    return dict(source, config=dict(source["config"], output=f"{cfg['output']}/runs/{arm}"),
                proposal=dict(source["proposal"], episode_safety_seconds=cfg["episode_safety_seconds"],
                              process_fuse_seconds=cfg["process_fuse_seconds"]))


def schedule(source, cfg):
    require(tuple(cfg["arms"]) == ARMS and cfg["replicas"] == 2 and cfg["conditions"] == 8, "scope changed")
    conditions = run.paired_conditions(source, "development_holdout")
    require(len(conditions) == cfg["conditions"] and len({c["case"]["map_id"] for c in conditions}) == 2,
            "development holdout scope")
    jobs = []
    for c in conditions:
        for replica in range(cfg["replicas"]):
            for arm in cfg["arms"]:
                jobs.append(dict(c, comparison_arm=arm, arm=engine_arm(arm), phase=cfg["phase"], replica=replica,
                                 iteration=cfg["models"].get(arm, {}).get("iteration", 0),
                                 job_id=run.json_fingerprint([cfg["phase"], c["pair_id"], replica, arm])[:24]))
    return jobs


def folder_for(cfg, job):
    return ROOT/cfg["output"]/"runs"/job["comparison_arm"]/job["phase"]/job["job_id"]


def check_model(cfg, arm, scientific_binding):
    spec = cfg["models"][arm]
    path = run.contained_file(ROOT, spec["path"], field="comparison model")
    require(run.sha256_file(path) == spec["file_sha256"], "changed model bytes")
    bundle = run.read_json(path)
    require(run.validate_bundle(bundle) == spec["policy_sha256"] and
            bundle["binding"] == scientific_binding and bundle["iteration"] == spec["iteration"], "model identity")
    if arm.startswith("bounded_"):
        require(bundle["prototype_arm"] == arm and bundle["parent_policy"] ==
                cfg["models"]["untrained_exploration"]["policy_sha256"], "fork identity")
    return bundle


def prepare():
    cross_reg, source, _, cross_out = crossfit.verify()
    cfg = run.read_json(ROOT/CONFIG)
    out = ROOT/cfg["output"]
    require(not out.exists(), "existing comparison; use verify/resume")
    require((cfg["workers"], cfg["episode_safety_seconds"], cfg["process_fuse_seconds"],
             cfg["formal_ttf"], cfg["automatic_promotion"]) == (20, 900., 960., False, False), "resource contract")
    require((source["proposal"]["max_decisions"], source["proposal"]["node_budget"],
             source["proposal"]["pp_safety_seconds"]) == (256, 25000000, 20.), "scientific work budget")
    source_plan, source_out = run.verify()
    inputs = {name: run.sha256_file(ROOT/name) for name in CODE}
    previous = ROOT/cfg["supersedes"]
    require(run.sha256_file(previous/"registration.json") == cfg["superseded_registration_sha256"], "previous registration")
    old = run.check_seal(run.read_json(previous/"registration.json"))
    inputs[(previous/"registration.json").relative_to(ROOT).as_posix()] = cfg["superseded_registration_sha256"]
    prior_count = 0
    for job in old["jobs"]:
        folder = folder_for(old["config"], job)
        if not (folder/"result.json").exists():
            require(not folder.exists(), "unexplained partial previous episode")
            continue
        prior = read_result(old, source, job)
        for name in ("result.json", "comparison_receipt.json", *prior["files"]):
            inputs[(folder/name).relative_to(ROOT).as_posix()] = run.sha256_file(folder/name)
        prior_count += 1
    require(prior_count == 20, "expected first batch only")
    for name in ("registration.json", "update.json", "parity.json", "torch-parity.json"):
        inputs[(cross_out/name).relative_to(ROOT).as_posix()] = run.sha256_file(cross_out/name)
    parity = run.check_seal(run.read_json(cross_out/"parity.json"))
    require(parity["binding"] == cross_reg["binding"] and parity["update_sha256"] ==
            run.sha256_file(cross_out/"update.json"), "crossfit parity binding")
    for name in ("parity-0.json", "micro.json", "qualify.complete.json"):
        require(run.check_seal(run.read_json(source_out/name))["binding"] == source_plan["binding"], "source qualification/parity")
        inputs[(source_out/name).relative_to(ROOT).as_posix()] = run.sha256_file(source_out/name)
    initials = {}
    for j in run.jobs_for(source_plan, "qualify"):
        if j["split"] != "development_holdout":
            continue
        folder = run.folder_for(source_out, "qualify", j)
        row = run.result_read(folder, source_plan)
        initials[j["pair_id"]] = row["initial_fingerprint"]
        for name in ("result.json", *row["files"]):
            inputs[(folder/name).relative_to(ROOT).as_posix()] = run.sha256_file(folder/name)
    bundles = {arm: check_model(cfg, arm, source["binding"]) for arm in cfg["models"]}
    require(all(b["update_binding"] == cross_reg["binding"] for a,b in bundles.items() if a.startswith("bounded_")), "fork update binding")
    for spec in cfg["models"].values():
        inputs[spec["path"]] = spec["file_sha256"]
    jobs = [dict(j, expected_initial=initials[j["pair_id"]]) for j in schedule(source, cfg)]
    body = dict(schema=cfg["schema"], config=cfg, scientific_binding=source["binding"],
                crossfit_binding=cross_reg["binding"], inputs=inputs, jobs=jobs,
                source_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                no_ttf=True, no_training=True, role="viewed_development_holdout")
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out, body["binding"], "comparison-prepare"):
        for arm, bundle in bundles.items():
            arm_out = ROOT/runtime_plan(source, cfg, arm)["config"]["output"]
            target = run.actor_file(arm_out, bundle["iteration"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/cfg["models"][arm]["path"], target)
            require(run.sha256_file(target) == cfg["models"][arm]["file_sha256"], "copy changed model")
            run.once(arm_out/"models"/f"receipt-{bundle['iteration']}.json", run.sealed(dict(
                binding=source["binding"], policy_sha256=run.validate_bundle(bundle), file_sha256=run.sha256_file(target),
                metadata=dict(comparison_binding=body["binding"], comparison_arm=arm, copied_not_trained=True))))
        run.once(out/"registration.json", run.sealed(body))
    return dict(registered=True, jobs=len(jobs), conditions=len(initials), binding=body["binding"], no_ttf=True)


def verify():
    cross_reg, source, _, _ = crossfit.verify()
    cfg = run.read_json(ROOT/CONFIG)
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/"registration.json"))
    require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding", "integrity")}), "registration hash")
    require(reg["config"] == cfg and reg["scientific_binding"] == source["binding"] and
            reg["crossfit_binding"] == cross_reg["binding"], "comparison identity")
    for name, digest in reg["inputs"].items():
        require(run.sha256_file(run.contained_file(ROOT, name, field="comparison input")) == digest, "changed input: "+name)
    require([{k:v for k,v in j.items() if k != "expected_initial"} for j in reg["jobs"]] == schedule(source, cfg), "schedule changed")
    for arm in cfg["models"]:
        bundle = check_model(cfg, arm, source["binding"])
        p = runtime_plan(source, cfg, arm)
        arm_out = ROOT/p["config"]["output"]
        loaded = run.actor_load(arm_out, p, bundle["iteration"])
        require(loaded == bundle, "copied model differs")
        receipt = run.check_seal(run.read_json(arm_out/"models"/f"receipt-{bundle['iteration']}.json"))
        require(receipt["metadata"] == dict(comparison_binding=reg["binding"], comparison_arm=arm, copied_not_trained=True), "model registration")
    return reg, source, out


def expected_policy(cfg, job):
    return cfg["models"].get(job["comparison_arm"], {}).get("policy_sha256", job["arm"])


def read_result(reg, source, job):
    cfg = reg["config"]
    folder = folder_for(cfg, job)
    p = runtime_plan(source, cfg, job["comparison_arm"])
    row = run.result_read(folder, p)
    require(all(row[k] == job[k] for k in ("job_id", "pair_id", "replica", "arm", "split")), "episode identity")
    require(row["map_id"] == job["case"]["map_id"] and row["initial_fingerprint"] == job["expected_initial"], "paired initial")
    require(row["policy_sha256"] == expected_policy(cfg, job), "wrong comparison policy")
    receipt = run.check_seal(run.read_json(folder/"comparison_receipt.json"))
    require(receipt == run.sealed(dict(comparison_binding=reg["binding"], comparison_arm=job["comparison_arm"],
             result_sha256=run.sha256_file(folder/"result.json"))), "comparison receipt")
    return row


def comparison_worker(job):
    summary = noop.episode_worker(job)
    folder = folder_for(job["comparison_config"], job)
    run.once(folder/"comparison_receipt.json", run.sealed(dict(comparison_binding=job["comparison_binding"],
             comparison_arm=job["comparison_arm"], result_sha256=run.sha256_file(folder/"result.json"))))
    return dict(summary, comparison_arm=job["comparison_arm"])


def pending_jobs(reg, source, out, resume):
    require(resume or not (out/"run_status.json").exists(), "explicit resume required")
    pending = []
    for j in reg["jobs"]:
        folder = folder_for(reg["config"], j)
        failure = out/"failures"/(j["job_id"]+".json")
        require(not failure.exists(), "recorded failure requires inspection, no automatic retry")
        if (folder/"result.json").exists():
            require(read_result(reg, source, j)["status"] == "ok", "censored result requires inspection")
        else:
            require(not folder.exists(), "partial episode requires inspection, no automatic retry")
            pending.append(dict(j, plan=runtime_plan(source, reg["config"], j["comparison_arm"]),
                                parent_pid=os.getpid(), comparison_config=reg["config"], comparison_binding=reg["binding"]))
    return pending


def collect(resume=False):
    from experiments.repair_collection import _run_jobs
    require(os.name != "nt", "use frozen WSL native")
    reg, source, out = verify()
    cfg = reg["config"]
    with recovery.strict_lock(out, reg["binding"], "comparison-collect"):
        pending = pending_jobs(reg, source, out, resume)
        if resume:
            (out/"STOP_AFTER_BATCH").unlink(missing_ok=True)
        def progress(row):
            with (out/"progress.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row)+"\n")
            print(row, flush=True)
        def failure(job, status, error):
            row = dict(status="censored" if status == "timeout" else "error", job_id=job["job_id"],
                       comparison_arm=job["comparison_arm"], stop="external_timeout" if status == "timeout" else "exception", error=error)
            run.once(out/"failures"/(job["job_id"]+".json"), run.sealed(dict(binding=reg["binding"], **row)))
            return row
        run.write_json(out/"run_status.json", dict(status="running", pending=len(pending), binding=reg["binding"]))
        try:
            for offset in range(0, len(pending), cfg["workers"]):
                if (out/"STOP_AFTER_BATCH").exists():
                    run.write_json(out/"run_status.json", dict(status="paused", binding=reg["binding"]))
                    return dict(status="paused")
                batch = pending[offset:offset+cfg["workers"]]
                rows = _run_jobs(comparison_worker, batch, cfg["workers"], phase=cfg["phase"],
                    output_root=out/"progress"/f"batch-{batch[0]['job_id']}", run_fingerprint=reg["binding"],
                    timeout_seconds=cfg["process_fuse_seconds"], on_result=progress, failure_result=failure,
                    stop_on_failure=False)
                if len(rows) != len(batch) or any(r["status"] != "ok" for r in rows):
                    run.write_json(out/"run_status.json", dict(status="needs_inspection", binding=reg["binding"]))
                    return dict(status="needs_inspection", no_automatic_retry=True)
            records = [read_result(reg, source, j) for j in reg["jobs"]]
            require(all(r["status"] == "ok" for r in records), "incomplete comparison")
            run.once(out/"collection.complete.json", run.sealed(dict(binding=reg["binding"], jobs=len(records),
                files={j["job_id"]:run.sha256_file(folder_for(cfg,j)/"result.json") for j in reg["jobs"]})))
            run.write_json(out/"run_status.json", dict(status="completed", jobs=len(records), binding=reg["binding"]))
        except BaseException as exc:
            run.write_json(out/"run_status.json", dict(status="interrupted_or_error", error=repr(exc), binding=reg["binding"]))
            raise
    return dict(status="completed", jobs=len(records), no_ttf=True)


def audit_worker(job):
    # The unchanged engine auditor reconstructs features, scores, draws and all state deltas.
    row = run.audit_worker(job)
    folder = folder_for(job["comparison_config"], job)
    result = run.result_read(folder, job["plan"])
    final = run.read_json(folder/"final.json")
    paths = [a["path"] for a in final["agents"]]
    require(result["soc"] == sum(len(p)-1 for p in paths) and
            result["makespan"] == max(len(p)-1 for p in paths) and
            result["wait_steps"] == sum(a == b for p in paths for a,b in zip(p,p[1:])), "path-quality summary")
    expected = expected_policy(job["comparison_config"], job)
    require(result["rng_stream_id"] == run.json_fingerprint([job["plan"]["config"]["stream_seed"],
            job["phase"],job["pair_id"],job["replica"]]), "RNG identity")
    state = run.read_json(folder/"initial.json")
    q = run.native_runtime(job["plan"])
    noops = 0
    for event in run.trace_read(folder):
        require(event["policy_sha256"] == expected, "trace model changed")
        after = q.apply_state_delta(state, event["delta"])
        incomplete = noop.pp_incomplete(state, after, event["metrics"])
        if incomplete:
            require(event["decision"] + 1 == result["decisions"] and result["stop"] == "incomplete_pp", "continued incomplete PP")
        noops += event["metrics"]["pp_failure_reason"] == "not_run"
        state = after
    return dict(row, legal_noops=noops)


def audit():
    from experiments.repair_collection import _run_jobs
    reg, source, out = verify()
    jobs, missing = [], []
    for j in reg["jobs"]:
        folder = folder_for(reg["config"],j)
        if not (folder/"comparison_receipt.json").exists():
            missing.append(j["job_id"])
            continue
        read_result(reg,source,j)
        jobs.append(dict(j, plan=runtime_plan(source,reg["config"],j["comparison_arm"]), comparison_config=reg["config"]))
    with recovery.strict_lock(out, reg["binding"], "comparison-audit"):
        rows = _run_jobs(audit_worker,jobs,reg["config"]["workers"],phase="comparison-audit",
                        output_root=out/"audit-progress",run_fingerprint=reg["binding"],timeout_seconds=960.)
        require(len(rows)==len(jobs) and all(r["status"]=="ok" for r in rows), "trace audit failed")
        run.once(out/("audit.json" if not missing else f"audit-partial-{len(rows)}.json"),run.sealed(
            dict(binding=reg["binding"],results=rows,missing=missing)))
    return dict(audited=len(rows),missing=len(missing))


def prefix_audit(reg, source):
    old = run.check_seal(run.read_json(ROOT/reg["config"]["supersedes"]/"registration.json"))
    current = {j["job_id"]:j for j in reg["jobs"]}
    rows = []
    for j in old["jobs"]:
        folder = folder_for(old["config"],j)
        if not (folder/"result.json").exists(): continue
        prior = read_result(old,source,j)
        new_folder = folder_for(reg["config"],current[j["job_id"]])
        events = iter(run.trace_read(new_folder))
        count = 0
        for event in run.trace_read(folder):
            new = next(events,None)
            require(new is not None and recovery.event_projection(event) == recovery.event_projection(new), "changed old action/path prefix")
            count += 1
        now = read_result(reg,source,current[j["job_id"]])
        if prior["status"] == "ok":
            require(next(events,None) is None and all(prior[k] == now[k] for k in
                ("stop","success","decisions","generated","final_fingerprint","soc","makespan","wait_steps")), "unchanged arm diverged")
        else:
            require(prior["stop"] == "incomplete_pp" and event["metrics"]["pp_failure_reason"] == "not_run", "unexpected prior censor")
            require(now["decisions"] > prior["decisions"], "legal no-op was not continued")
        rows.append(dict(job_id=j["job_id"],prefix_steps=count,previous_status=prior["status"]))
    require(len(rows)==20,"missing prior prefixes")
    return rows


def value(row):
    return int(row["success"]) if row["status"] == "ok" else None


def paired_comparison(left, right):
    require(set(left)==set(right), "unpaired denominator")
    wins=losses=ties=unknown=0
    low=high=0
    common=[]
    for key in sorted(left):
        a,b=value(left[key]),value(right[key])
        low += (0 if a is None else a) - (1 if b is None else b)
        high += (1 if a is None else a) - (0 if b is None else b)
        if a is None or b is None:
            unknown+=1
        elif a>b: wins+=1
        elif a<b: losses+=1
        else: ties+=1
        if a==b==1: common.append((left[key],right[key]))
    means={k:dict(left=sum(a[k] for a,b in common)/len(common),right=sum(b[k] for a,b in common)/len(common))
           for k in ("decisions","generated","soc","makespan","wait_steps")} if common else {}
    return dict(pairs=len(left),wins=wins,losses=losses,ties=ties,unknown_pairs=unknown,
                net_success_bounds=[low,high],common_success=len(common),common_success_means=means)


def report():
    reg,source,out=verify()
    audit_path=out/"audit.json"
    require(audit_path.exists(), "complete trace audit required; partial diagnostics are not a complete report")
    receipt=run.check_seal(run.read_json(audit_path))
    require(receipt["binding"]==reg["binding"] and not receipt["missing"], "audit identity")
    hashes={r["job_id"]:r["result_sha256"] for r in receipt["results"]}
    data={arm:{} for arm in ARMS}
    for j in reg["jobs"]:
        row=read_result(reg,source,j)
        require(hashes[j["job_id"]]==run.sha256_file(folder_for(reg["config"],j)/"result.json"), "stale audit")
        data[j["comparison_arm"]][(j["pair_id"],j["replica"])]=row
    for key in data[ARMS[0]]:
        require(len({data[a][key]["initial_fingerprint"] for a in ARMS})==1 and
                len({data[a][key]["rng_stream_id"] for a in ARMS})==1,"cross-arm pairing")
    summaries={a:dict(episodes=len(rows),success=sum(value(r)==1 for r in rows.values()),
                     unknown=sum(value(r) is None for r in rows.values()),stops=dict(Counter(r["stop"] for r in rows.values())))
               for a,rows in data.items()}
    comparisons={}
    maps=sorted({r["map_id"] for r in data[ARMS[0]].values()})
    for arm in ARMS[2:]:
        for base in ARMS[:3]:
            if arm==base: continue
            a,b=data[arm],data[base]
            comparisons[f"{arm}_vs_{base}"]=dict(overall=paired_comparison(a,b),by_map={m:paired_comparison(
                {k:v for k,v in a.items() if v["map_id"]==m},{k:v for k,v in b.items() if v["map_id"]==m}) for m in maps})
    body=dict(binding=reg["binding"],schema="lns2.sa.crossfit_comparison_report.v1",no_ttf=True,
              role="viewed_development_holdout",independent_generalization=False,automatic_promotion=False,
              summaries=summaries,comparisons=comparisons,
              prefix_equivalence=prefix_audit(reg,source),
              legal_noops=sum(r["legal_noops"] for r in receipt["results"]),
              episodes=[dict(comparison_arm=a,**r) for a,rows in data.items() for r in rows.values()],
              decision="bounded_development_comparison_only")
    with recovery.strict_lock(out,reg["binding"],"comparison-report"):
        run.once(out/"report.json",run.sealed(body))
    return dict(summaries=summaries,comparisons={k:v["overall"] for k,v in comparisons.items()},no_ttf=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","collect","audit","report","stop"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    if args.phase=="verify":
        reg,_,_=verify()
        result=dict(verified=True,jobs=len(reg["jobs"]),binding=reg["binding"])
    elif args.phase=="collect": result=collect(args.resume)
    elif args.phase=="stop":
        cfg=run.read_json(ROOT/CONFIG)
        run.write_json(ROOT/cfg["output"]/"STOP_AFTER_BATCH",dict(requested=True))
        result=dict(safe_stop_after_current_batch=True)
    else: result=globals()[args.phase]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
