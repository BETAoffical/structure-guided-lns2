"""One Train-only sampling intervention; no new actor training or timing study."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import collect_sa_second_batch as batch
from scripts.register_sa_crossfit_report_fix import first_difference
from experiments.sa_terminal_sampler import make_sampler, validate_sampler, sampler_runtime, TerminalSampler, coverage

run, compare, recovery, runtime = batch.run, batch.compare, batch.recovery, batch.runtime
CONFIG = "configs/sa_terminal_sampling.json"
REGISTRATION = "sampling_registration.json"
ARM = "half_uniform_sampling"
CODE = (CONFIG, "scripts/probe_sa_terminal_sampling.py", "experiments/sa_terminal_sampler.py",
        "tests/evaluation/test_sa_terminal_sampling.py", "docs/SA_TERMINAL_SAMPLING_PROTOCOL_ZH.md")


def selected_controls(jobs, cfg, train_maps):
    selected = []
    for j in jobs:
        run.require(j["split"] == "train" and j["case"]["map_id"] in train_maps, "Train-only sampling")
        if j["comparison_arm"] == cfg["parent_arm"] and j["case"]["task_variant"] == cfg["task_variant"]:
            selected.append(j)
    groups = {}
    for j in selected:
        groups.setdefault(j["pair_id"], []).append(j)
    run.require(len(selected) == len({j["job_id"] for j in selected}) == 48 and len(groups) == 12,
                "complete high-density scope")
    run.require({j["case"]["map_id"] for j in selected} == set(train_maps) and len(train_maps) == 6,
                "all six Train maps required")
    run.require(all(len(js) == 4 and {j["replica"] for j in js} == set(range(4)) for js in groups.values()),
                "four paired replicas per condition")
    run.require(Counter(j["case"]["map_id"] for j in selected) == {m:8 for m in train_maps} and
                all(j["solver_seed"] in (233,239) for j in selected), "two initial seeds per map")
    return selected


def context():
    old, source, old_out = batch.verify()
    cfg = run.read_json(ROOT/CONFIG)
    fixed = dict(parent_arm="uncapped_condition", task_variant="bottleneck_d25", uniform_mix=.5,
        conditions=12, replicas=4, expected_jobs=48, max_decisions=None, decision_feature_reference=256,
        node_budget=25000000, pp_safety_seconds=20., episode_safety_seconds=900., process_fuse_seconds=960.,
        workers=20, formal_ttf=False, training=False, automatic_promotion=False)
    run.require(all(cfg[k] == v for k,v in fixed.items()) and old_out == ROOT/cfg["source"], "frozen sampling scope")
    run.require(run.sha256_file(old_out/"report.json") == cfg["source_report_sha256"], "source report changed")
    complete = run.check_seal(run.read_json(old_out/"collection.complete.json"))
    audit = run.check_seal(run.read_json(old_out/"audit.json"))
    run.require(complete["binding"] == audit["binding"] == old["binding"] and complete["jobs"] == 192 and
                len(audit["results"]) == 192 and all(r["status"] == "ok" for r in audit["results"]) and
                {r["job_id"]:r["result_sha256"] for r in audit["results"]} == complete["files"], "source audit coverage")
    controls = selected_controls(batch.ready_jobs(old,source,old_out),cfg,source["split"]["train_maps"])
    for j in controls:
        row = compare.read_result(old,source,j)
        run.require(row["status"] == "ok" and run.sha256_file(compare.prior.folder_for(old["config"],j)/"result.json") ==
                    complete["files"][j["job_id"]], "source control changed")
    parent = compare.prior.check_model(old["config"],cfg["parent_arm"],source["binding"])
    bundle = make_sampler(parent)
    cfg = dict(cfg, arms=[ARM], phase=old["config"]["phase"], models={ARM:dict(iteration=bundle["iteration"],
        policy_sha256=validate_sampler(bundle))})
    jobs = [dict(j, comparison_arm=ARM, control_job_id=j["job_id"],
        job_id=run.json_fingerprint([j["phase"],j["pair_id"],j["replica"],ARM])[:24]) for j in controls]
    return old, source, old_out, cfg, controls, jobs, bundle


def prepare():
    old,source,old_out,cfg,controls,jobs,bundle = context()
    out = ROOT/cfg["output"]
    run.require(not out.exists(), "existing output; use verify/resume")
    inputs = {n:run.sha256_file(ROOT/n) for n in CODE}
    for name in ("registration.json","qualification.json","collection.complete.json","audit.json","report.json"):
        inputs[(old_out/name).relative_to(ROOT).as_posix()] = run.sha256_file(old_out/name)
    for j in controls:
        folder = compare.prior.folder_for(old["config"],j)
        row = compare.read_result(old,source,j)
        for name in ("result.json","comparison_receipt.json",*row["files"]):
            inputs[(folder/name).relative_to(ROOT).as_posix()] = run.sha256_file(folder/name)
    body = dict(schema=cfg["schema"], config=cfg, inputs=inputs, jobs=jobs, control_jobs=controls,
        source_binding=old["binding"], scientific_binding=source["binding"],
        source_commit=run.subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        role="paired_train_sampling_intervention_not_new_initial_conditions",
        no_training=True, no_ttf=True, no_heldout=True, parent_policy=bundle["parent_policy"])
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out,body["binding"],"terminal-sampling-prepare"):
        plan = compare.runtime_plan(source,cfg,ARM)
        target = run.actor_file(ROOT/plan["config"]["output"],bundle["iteration"])
        run.once(target,bundle)
        run.once(target.parent/f"receipt-{bundle['iteration']}.json",run.sealed(dict(binding=source["binding"],
            file_sha256=run.sha256_file(target),policy_sha256=validate_sampler(bundle),
            metadata=dict(sampling_binding=body["binding"],non_trainable=True))))
        run.once(out/REGISTRATION,run.sealed(body))
    return dict(registered=True,binding=body["binding"],new_episodes=48,reused_controls=48,max_decisions=None)


def verify():
    old,source,_,cfg,controls,jobs,bundle = context()
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/REGISTRATION))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding","integrity")}),
                "registration hash")
    run.require(reg["config"] == cfg and reg["jobs"] == jobs and reg["control_jobs"] == controls and
                reg["source_binding"] == old["binding"] and reg["scientific_binding"] == source["binding"],
                "sampling registration identity")
    for n,digest in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT,n,field="sampling input")) == digest,"changed input: "+n)
    plan = compare.runtime_plan(source,cfg,ARM)
    with sampler_runtime():
        run.require(run.actor_load(ROOT/plan["config"]["output"],plan,bundle["iteration"]) == bundle,"sampler changed")
    return reg,source,out,old


def worker(job):
    with sampler_runtime():
        return compare.worker(job)


def audit_worker(job):
    with sampler_runtime():
        return runtime.audit_worker(job)


def check_parity(receipts, binding):
    run.require(len(receipts) == 2 and all(r["binding"] == binding and 0 <= r["max_error"] <= 1e-12 for r in receipts),
                "cross-platform sampler parity required")
    # Both errors are measured against the same sealed WSL parent probabilities.
    # Their sum bounds cross-platform error without requiring equal float JSON hashes.
    run.require(sum(r["max_error"] for r in receipts) <= 1e-12,"cross-platform probability tolerance")
    def projected(r):
        return [{k:s[k] for k in ("job_id","decision","selections")} for s in r["samples"]]
    run.require(projected(receipts[0]) == projected(receipts[1]) and len(receipts[0]["samples"]) >= 48,
                "cross-platform candidate selections differ")


def collect(resume=False):
    reg,source,out,_ = verify()
    receipts = [run.check_seal(run.read_json(out/f"parity-{platform}.json")) for platform in ("windows","wsl")]
    check_parity(receipts,reg["binding"])
    with recovery.strict_lock(out,reg["binding"],"terminal-sampling-collect"):
        done = batch.execute_batches(reg,out,[compare.augmented(reg,source,j) for j in reg["jobs"]],worker,
            lambda j:compare.prior.folder_for(reg["config"],j),lambda j:compare.read_result(reg,source,j),
            "collection",resume,reg["config"]["process_fuse_seconds"])
    return dict(status="collected" if done else "paused_or_needs_inspection",no_training=True)


def completed(reg,source,out):
    proof = run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(proof["binding"] == reg["binding"] and proof["jobs"] == 48 and
                set(proof["files"]) == {j["job_id"] for j in reg["jobs"]}, "complete collection required")
    for j in reg["jobs"]:
        row = compare.read_result(reg,source,j)
        run.require(row["status"] == "ok" and run.sha256_file(compare.prior.folder_for(reg["config"],j)/"result.json") ==
                    proof["files"][j["job_id"]], "changed/unknown terminal result")
    return proof


def audit():
    from experiments.repair_collection import _run_jobs
    reg,source,out,_ = verify()
    proof = completed(reg,source,out)
    with recovery.strict_lock(out,reg["binding"],"terminal-sampling-audit"):
        rows = _run_jobs(audit_worker,[compare.augmented(reg,source,j) for j in reg["jobs"]],20,
            phase="terminal-sampling-audit",output_root=out/"audit-progress",run_fingerprint=reg["binding"],timeout_seconds=960.)
        run.require(len(rows) == 48 and all(r["status"] == "ok" for r in rows) and
                    {r["job_id"]:r["result_sha256"] for r in rows} == proof["files"], "full trace audit")
        run.once(out/"audit.json",run.sealed(dict(binding=reg["binding"],results=rows)))
    return dict(audited=48)


def summarize_trace(folder):
    total, nonanchor, p_anchor, minimum, conflicts = 0, 0, 0., None, []
    for e in run.trace_read(folder):
        total += 1
        nonanchor += e["selected_id"] != e["anchor_id"]
        p_anchor += e["probabilities"][e["anchor_id"]]
        after = e["metrics"]["conflicts_after"]
        minimum = after if minimum is None else min(minimum,after)
        conflicts.append(after)
    return dict(decisions=total,nonanchor=nonanchor,anchor_probability_sum=p_anchor,
                minimum_conflicts=minimum,last50_conflicts=conflicts[-50:])


def parity():
    reg,source,out,old = verify()
    plan = compare.runtime_plan(source,reg["config"],ARM)
    with sampler_runtime():
        bundle = run.actor_load(ROOT/plan["config"]["output"],plan,reg["config"]["models"][ARM]["iteration"])
    sampler = TerminalSampler(bundle)
    samples, error = [], 0.
    for j in reg["control_jobs"]:
        for e in run.trace_read(compare.prior.folder_for(old["config"],j)):
            if e["decision"] not in (0,256):
                continue
            p = sampler.probabilities(e["candidate_ids"],e["anchor_id"],e["features"])
            target = {cid:.5*v+.5/len(p) for cid,v in e["probabilities"].items()}
            error = max(error,max(abs(p[cid]-v) for cid,v in target.items()))
            draws = [0.,.01,.1,.25,.5,.75,.9,.999999]
            samples.append(dict(job_id=j["job_id"],decision=e["decision"],
                probabilities_sha256=run.json_fingerprint(p),
                selections=[run.select_with_draw(p,d) for d in draws]))
    run.require(error <= 1e-12 and len(samples) >= 48,"parent probability replay / sampling mismatch")
    platform = "windows" if os.name == "nt" else "wsl"
    with recovery.strict_lock(out,reg["binding"],"sampler-parity"):
        run.once(out/f"parity-{platform}.json",run.sealed(dict(binding=reg["binding"],samples=samples,max_error=error)))
    return dict(platform=platform,samples=len(samples),max_error=error,no_solver=True)


def report():
    reg,source,out,old = verify()
    proof = completed(reg,source,out)
    audit_proof = run.check_seal(run.read_json(out/"audit.json"))
    run.require(audit_proof["binding"] == reg["binding"] and len(audit_proof["results"]) == 48 and
                all(r["status"] == "ok" for r in audit_proof["results"]) and
                {r["job_id"]:r["result_sha256"] for r in audit_proof["results"]} == proof["files"],"matching full audit")
    parent, sampled, behavior = {}, {}, []
    for j,c in zip(reg["jobs"],reg["control_jobs"]):
        key = j["pair_id"],j["replica"]
        a,b = compare.read_result(reg,source,j),compare.read_result(old,source,c)
        run.require(j["control_job_id"] == c["job_id"] and all(a[f] == b[f] for f in
            ("pair_id","replica","map_id","split","initial_fingerprint","rng_stream_id")), "paired control identity")
        left,right = compare.prior.folder_for(reg["config"],j),compare.prior.folder_for(old["config"],c)
        count,first = first_difference(run.trace_read(left),run.trace_read(right))
        if first is None:
            run.require(a["decisions"] == b["decisions"] and a["final_fingerprint"] == b["final_fingerprint"],"unexplained divergence")
        parent[key],sampled[key] = b,a
        behavior.append(dict(pair_id=key[0],replica=key[1],map_id=a["map_id"],common_prefix=count,first_difference=first,
                             parent=summarize_trace(right),sampled=summarize_trace(left)))
    groups = sorted({k[0] for k in parent})
    counts = [{g:sum(r["success"] for (p,_),r in rows.items() if p == g) for g in groups} for rows in (parent,sampled)]
    information = coverage(*counts)
    maps = sorted({r["map_id"] for r in parent.values()})
    summaries = {name:dict(episodes=len(rows),success=sum(r["success"] for r in rows.values()),
        by_map={m:sum(r["success"] for r in rows.values() if r["map_id"] == m) for m in maps},
        stops=dict(Counter(r["stop"] for r in rows.values())),
        beyond_256=sum(r["decisions"] > 256 for r in rows.values()),max_decisions=max(r["decisions"] for r in rows.values()))
        for name,rows in (("parent",parent),("sampled",sampled))}
    comparisons = dict(overall=compare.prior.paired_comparison(sampled,parent),by_map={m:
        compare.prior.paired_comparison({k:r for k,r in sampled.items() if r["map_id"] == m},
                                       {k:r for k,r in parent.items() if r["map_id"] == m}) for m in maps})
    body = dict(schema="lns2.sa.terminal_sampling_report.v1",binding=reg["binding"],summaries=summaries,
        comparisons=comparisons,terminal_information=information,condition_success=dict(parent=counts[0],sampled=counts[1]),
        behavior=behavior,episodes=[dict(sampling_arm=n,**r) for n,rows in (("parent",parent),("sampled",sampled)) for r in rows.values()],
        decision=information["decision"],audit_sha256=run.sha256_file(out/"audit.json"),
        no_training=True,no_ttf=True,no_heldout=True,automatic_promotion=False,
        uncertainty="six seen Train maps, 12 paired conditions, 48 paired streams; not independent model validation")
    with recovery.strict_lock(out,reg["binding"],"terminal-sampling-report"):
        run.once(out/"report.json",run.sealed(body))
        run.write_json(out/"run_status.json",dict(status="completed",binding=reg["binding"],report_sha256=run.sha256_file(out/"report.json")))
    return dict(summaries=summaries,paired=comparisons["overall"],terminal_information=information)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","dry-run","parity","collect","audit","report","stop"))
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if args.phase == "collect": result = collect(args.resume)
    elif args.phase in ("verify","dry-run"):
        reg,_,_,_ = verify()
        result = dict(binding=reg["binding"],new_jobs=len(reg["jobs"]),reused_controls=len(reg["control_jobs"]),
            workers=20,max_decisions=None,no_training=True,node_budget_per_job=25000000,
            external_safety_max_seconds=3*960,not_a_runtime_prediction=True)
    elif args.phase == "stop":
        cfg = run.read_json(ROOT/CONFIG)
        run.write_json(ROOT/cfg["output"]/"STOP_AFTER_BATCH",dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else: result = globals()[args.phase]()
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
