"""One fixed work-aligned cooling schedule versus decision cooling, Train only."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
from scripts import collect_sa_second_batch as batch
from scripts import compare_sa_uncapped as compare
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_uncapped_runtime as runtime
from experiments import sa_budget_cooling as cooling

CONFIG = "configs/sa_budget_cooling.json"
CODE = (CONFIG, "experiments/sa_budget_cooling.py", "scripts/probe_sa_budget_cooling.py",
        "tests/evaluation/test_sa_budget_cooling.py", "docs/SA_BUDGET_COOLING_PROTOCOL_ZH.md")


def configuration():
    cfg = run.read_json(ROOT/CONFIG)
    expected = dict(phase="budget-cooling-train-20260924", replicas=[0,1], clocks=list(cooling.CLOCKS),
        actors=list(batch.ARMS), conditions=24, expected_jobs=192, workers=20, max_decisions=None,
        node_budget=25000000, endpoint_unit_worsening_probability=.05, initial_temperature=1000.,
        pp_safety_seconds=20., episode_safety_seconds=900., process_fuse_seconds=960.,
        no_training=True, formal_ttf=False, no_heldout=True, automatic_promotion=False)
    run.require(all(cfg[k] == v for k,v in expected.items()), "fixed cooling probe scope")
    return cfg


def schedule(conditions, cfg, expected):
    run.require(len(conditions)==24 and all(c["split"]=="train" for c in conditions), "Train conditions only")
    return [dict(c,phase=cfg["phase"],replica=r,clock=clock,comparison_arm=actor,
        arm="trained_actor" if actor==batch.ARMS[1] else actor, iteration=int(actor==batch.ARMS[1]),
        expected_initial=expected[c["pair_id"]],
        job_id=run.json_fingerprint([cfg["phase"],c["pair_id"],r,clock,actor])[:24])
        for c in conditions for r in cfg["replicas"] for actor in cfg["actors"] for clock in cfg["clocks"]]


def plan_for(reg, source, job):
    p = compare.runtime_plan(source,reg["source_config"],job["comparison_arm"])
    output = reg["config"]["output"] + "/" + ("controls" if job.get("control") else "runs")
    output += "/" + job["clock"] + "/" + job["comparison_arm"]
    return dict(p, config=dict(p["config"], output=output))


def folder(reg, source, job):
    p = plan_for(reg,source,job)
    return run.folder_for(ROOT/p["config"]["output"],job["phase"],job)


def augmented(reg,source,job):
    return dict(job,plan=plan_for(reg,source,job),parent_pid=os.getpid(),comparison_binding=reg["binding"])


def verify():
    cfg=configuration()
    out=ROOT/cfg["output"]
    reg=run.check_seal(run.read_json(out/"comparison_registration.json"))
    run.require(reg["binding"]==run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding","integrity")}), "registration hash")
    run.require(reg["config"]==cfg, "changed configuration")
    for name,sha in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT,name,field="cooling input"))==sha,"changed input: "+name)
    return reg,reg["source_plan"],out


def prepare():
    cfg=configuration()
    out=ROOT/cfg["output"]
    run.require(not out.exists(), "existing output; explicit resume only")
    _,source,_,old_cfg,conditions,bundles=batch.context()
    src=ROOT/cfg["source"]
    run.require(run.sha256_file(src/"report.json")==cfg["source_report_sha256"],"frozen source report")
    old=run.check_seal(run.read_json(src/"registration.json"))
    report=run.check_seal(run.read_json(src/"report.json"))
    run.require(old["config"]==old_cfg and old["scientific_binding"]==source["binding"] and
                report["binding"]==old["binding"] and old["conditions"]==conditions, "source identity")
    expected={}
    originals={r["job_id"]:r for r in report["episodes"]}
    for j in old["jobs"]:
        row=originals[j["job_id"]]
        run.require(row["split"]=="train" and row["status"]=="ok", "source coverage")
        expected.setdefault(j["pair_id"],row["initial_fingerprint"])
        run.require(expected[j["pair_id"]]==row["initial_fingerprint"],"source pairing")
    jobs=schedule(conditions,cfg,expected)
    run.require(len(jobs)==len({j["job_id"] for j in jobs})==cfg["expected_jobs"], "job count")
    controls=[]
    for actor in cfg["actors"]:
        row=min((r for r in originals.values() if r["comparison_arm"]==actor),key=lambda r:(r["decisions"],r["job_id"]))
        j=next(j for j in old["jobs"] if j["job_id"]==row["job_id"])
        controls.append(dict(j,clock="decision",control=True,expected_initial=row["initial_fingerprint"],
            original_folder=(src/"runs"/actor/j["phase"]/j["job_id"]).relative_to(ROOT).as_posix()))
    inputs={name:run.sha256_file(ROOT/name) for name in CODE}
    # Bind runtime implementation without importing or modifying historical results.
    for directory in ("scripts","experiments","lns2_selector"):
        for path in (ROOT/directory).rglob("*.py"):
            inputs[path.relative_to(ROOT).as_posix()]=run.sha256_file(path)
    for case in conditions:
        for name in case["case"]["files"].values():
            inputs[name]=run.sha256_file(ROOT/name)
    for name in ("registration.json","report.json","audit.json","collection.complete.json"):
        inputs[(src/name).relative_to(ROOT).as_posix()]=run.sha256_file(src/name)
    for j in controls:
        row=run.result_read(ROOT/j["original_folder"],source)
        for name in (*row["files"],"result.json"):
            path=ROOT/j["original_folder"]/name
            inputs[path.relative_to(ROOT).as_posix()]=run.sha256_file(path)
    inputs[source["native_file"]]=source["config"]["native_sha256"]
    for spec in old_cfg["models"].values(): inputs[spec["path"]]=spec["file_sha256"]
    for path in (ROOT/"artifacts/initlns-closed-loop-controller-v2").glob("*.json"):
        inputs[path.relative_to(ROOT).as_posix()]=run.sha256_file(path)
    body=dict(schema=cfg["schema"],config=cfg,source_config=old_cfg,source_plan=source,
        inputs=inputs,jobs=jobs,controls=controls,source_commit=run.subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),no_ttf=True,no_training=True,no_heldout=True)
    body["binding"]=run.json_fingerprint(body)
    with recovery.strict_lock(out,body["binding"],"cooling-prepare"):
        destinations={}
        for j in jobs+controls:
            p=plan_for(body,source,j)
            destinations[p["config"]["output"]]=(p,j["comparison_arm"])
        for destination,(p,actor) in destinations.items():
            bundle=bundles[actor]
            target=run.actor_file(ROOT/destination,bundle["iteration"])
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/old_cfg["models"][actor]["path"],target)
            run.once(target.parent/f"receipt-{bundle['iteration']}.json",run.sealed(dict(binding=source["binding"],
                policy_sha256=run.validate_bundle(bundle),file_sha256=run.sha256_file(target),
                metadata=dict(copied_not_trained=True,cooling_binding=body["binding"]))))
        run.once(out/"comparison_registration.json",run.sealed(body))
    return dict(registered=True,controls=2,jobs=192,workers=20,no_training=True,no_ttf=True)


def worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job["parent_pid"])
    p=job["plan"]
    out=ROOT/p["config"]["output"]
    target=run.folder_for(out,job["phase"],job)
    run.require(not target.exists(),"partial episode requires inspection")
    q,env,state,ctx=run.reset(job)
    run.require(q.state_fingerprint(state)==job["expected_initial"],"initial mismatch")
    bundle=run.actor_load(out,p,job["iteration"])
    clock=cooling.WorkClock(job["clock"],state["low_level"]["generated"])
    wrapped=cooling.ClockedEnvironment(env,clock,q._plain)
    with cooling.clock_scope(q,clock):
        row=runtime.episode_loop(job,q,wrapped,state,ctx,target,bundle)
    run.require(clock.decision==row["decisions"] and clock.used==row["generated"],"observed work mismatch")
    return {k:row[k] for k in ("status","job_id","stop","success","decisions")}


def read_result(reg,source,j):
    row=run.result_read(folder(reg,source,j),plan_for(reg,source,j))
    run.require(row["execution_binding"]==reg["binding"] and row["job_id"]==j["job_id"] and
        row["initial_fingerprint"]==j["expected_initial"] and row["policy_sha256"]==
        reg["source_config"]["models"][j["comparison_arm"]]["policy_sha256"],"episode identity")
    run.require(all(row[k]==j[k] for k in ("pair_id","replica","arm","split")) and row["map_id"]==j["case"]["map_id"],"task identity")
    runtime.validate_terminal(row,plan_for(reg,source,j)["proposal"])
    return row


def audit_worker(job):
    q=run.native_runtime(job["plan"])
    out=ROOT/job["plan"]["config"]["output"]
    f=run.folder_for(out,job["phase"],job)
    state=run.read_json(f/"initial.json")
    clock=cooling.WorkClock(job["clock"],state["low_level"]["generated"])
    temps={}
    stats=Counter()
    oor=[]
    for event in run.trace_read(f):
        d=event["decision"]
        temps[d]=clock(d)
        run.require(event["temperature"]==temps[d],"actual cooling differs")
        m=event["metrics"]
        worse=m["acceptance_evaluated"] and m["pp_attempt_conflict_pair_count"]>m["pp_old_conflict_pair_count"]
        stats.update(dict(steps=1,worse=int(worse),worse_accepted=int(worse and m["replan_success"]),
            rollback=int(m["pp_rolled_back"]),late_worse=int(worse and clock.used>=.9*cooling.NODE_BUDGET),
            late_worse_accepted=int(worse and clock.used>=.9*cooling.NODE_BUDGET and m["replan_success"])))
        oor.append(event["out_of_range_fraction"])
        state=q.apply_state_delta(state,event["delta"])
        clock.observe(state)
    # The existing full auditor rechecks features, policy draws, PP, paths and work.
    with cooling.clock_scope(q,temps.__getitem__):
        result=runtime.audit_worker(job)
    return dict(result,diagnostics=dict(counts=dict(stats),last_temperature=temps[max(temps)] if temps else None,
        mean_out_of_range_fraction=sum(oor)/len(oor) if oor else None))


def controls(resume=False):
    reg,source,out=verify()
    with recovery.strict_lock(out,reg["binding"],"cooling-controls"):
        jobs=[augmented(reg,source,j) for j in reg["controls"]]
        batch.execute_batches(reg,out,jobs,worker,
            lambda j:folder(reg,source,j),lambda j:read_result(reg,source,j),"controls",resume,960.)
        results=[]
        for j in jobs:
            row=read_result(reg,source,j)
            old=run.result_read(ROOT/j["original_folder"],source)
            n=compare.prefix_check(run.trace_read(ROOT/j["original_folder"]),run.trace_read(folder(reg,source,j)),old,row)
            check=audit_worker(j)
            results.append(dict(job_id=j["job_id"],decisions=n,audit=check))
        run.once(out/"controls.parity.json",run.sealed(dict(binding=reg["binding"],results=results,exact=True)))
    return dict(controls=2,exact=True,no_ttf=True)


def collect(resume=False):
    reg,source,out=verify()
    proof=run.check_seal(run.read_json(out/"controls.parity.json"))
    run.require(proof["binding"]==reg["binding"] and proof["exact"] and len(proof["results"])==2,"controls not admitted")
    with recovery.strict_lock(out,reg["binding"],"cooling-collect"):
        batch.execute_batches(reg,out,[augmented(reg,source,j) for j in reg["jobs"]],worker,
            lambda j:folder(reg,source,j),lambda j:read_result(reg,source,j),"collection",resume,960.)
    return dict(status=run.read_json(out/"collection.status.json"),no_ttf=True)


def audit():
    from experiments.repair_collection import _run_jobs
    reg,source,out=verify()
    complete=run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(complete["binding"]==reg["binding"] and complete["jobs"]==192,"collection incomplete")
    for j in reg["jobs"]:
        read_result(reg,source,j)
        run.require(complete["files"][j["job_id"]]==run.sha256_file(folder(reg,source,j)/"result.json"),"collection SHA")
    with recovery.strict_lock(out,reg["binding"],"cooling-audit"):
        rows=_run_jobs(audit_worker,[augmented(reg,source,j) for j in reg["jobs"]],20,
            phase="cooling-audit",output_root=out/"audit-progress",run_fingerprint=reg["binding"],timeout_seconds=960.)
        run.require(len(rows)==192 and all(r["status"]=="ok" for r in rows),"audit failed")
        run.once(out/"audit.json",run.sealed(dict(binding=reg["binding"],results=rows)))
    return dict(audited=192)


def contrasts(rows):
    groups=defaultdict(dict)
    for r in rows:
        key=(r["pair_id"],r["replica"])
        run.require(r["clock"] not in groups[key],"duplicate clock")
        groups[key][r["clock"]]=r
    wins,losses,common=[],[],[]
    for key,pair in sorted(groups.items()):
        run.require(set(pair)==set(cooling.CLOCKS),"incomplete clock pairing")
        a,b=(pair[c] for c in cooling.CLOCKS)
        run.require(all(a[k]==b[k] for k in ("initial_fingerprint","rng_stream_id","map_id","policy_sha256")),"pair identity")
        if b["success"] and not a["success"]: wins.append(list(key))
        if a["success"] and not b["success"]: losses.append(list(key))
        if a["success"] and b["success"]: common.append((a,b))
    return dict(pairs=len(groups),gains=wins,losses=losses,net_success=len(wins)-len(losses),
        common_success=len(common),common_success_totals={clock:{k:sum(pair[i][k] for pair in common)
            for k in ("generated","decisions","soc","makespan","wait_steps")} for i,clock in enumerate(cooling.CLOCKS)})


def map_bootstrap(by_map):
    values=[by_map[k] for k in sorted(by_map)]
    rng=random.Random(20260924)
    samples=[]
    for _ in range(5000):
        drawn=[rng.choice(values) for _ in values]
        samples.append(sum(v["net_success"] for v in drawn)/sum(v["pairs"] for v in drawn))
    samples.sort()
    return dict(resamples=5000,unit="map",maps=len(values),seed=20260924,
                success_rate_delta_ci95=[samples[125],samples[4874]],descriptive_not_gate=True)


def analyze():
    reg,source,out=verify()
    proof=run.check_seal(run.read_json(out/"audit.json"))
    audited={r["job_id"]:r["result_sha256"] for r in proof["results"]}
    diagnostics={r["job_id"]:r["diagnostics"] for r in proof["results"]}
    run.require(proof["binding"]==reg["binding"] and len(audited)==192,"complete audit required")
    rows=[]
    for j in reg["jobs"]:
        row=read_result(reg,source,j)
        run.require(audited[j["job_id"]]==run.sha256_file(folder(reg,source,j)/"result.json"),"stale audit")
        rows.append(dict(row,clock=j["clock"],comparison_arm=j["comparison_arm"],diagnostics=diagnostics[j["job_id"]]))
    summaries={}
    for actor in batch.ARMS:
        group=[r for r in rows if r["comparison_arm"]==actor]
        summaries[actor]=dict(contrast=contrasts(group),by_clock={c:dict(episodes=48,successes=sum(r["success"] for r in group if r["clock"]==c)) for c in cooling.CLOCKS},
            by_map={m:contrasts([r for r in group if r["map_id"]==m]) for m in sorted({r["map_id"] for r in group})})
        summaries[actor]["bootstrap"]=map_bootstrap(summaries[actor]["by_map"])
        for clock in cooling.CLOCKS:
            totals=Counter()
            subset=[r for r in group if r["clock"]==clock]
            for r in subset: totals.update(r["diagnostics"]["counts"])
            summaries[actor]["by_clock"][clock]["diagnostic_counts"]=dict(totals)
            summaries[actor]["by_clock"][clock]["stops"]=dict(Counter(r["stop"] for r in subset))
    main=summaries[batch.ARMS[1]]
    positive=sum(v["net_success"]>0 for v in main["by_map"].values())
    net=main["contrast"]["net_success"]
    decision=("distributed_development_completion_signal_needs_independent_check" if positive>=2 else
              "localized_development_completion_signal_needs_independent_check") if net>0 else "no_net_completion_gain_stop_schedule"
    report=dict(schema="lns2.sa.budget_cooling_report.v1",binding=reg["binding"],summaries=summaries,episodes=rows,
        decision=decision,
        no_training=True,no_ttf=True,no_promotion=True,audit_sha256=run.sha256_file(out/"audit.json"))
    run.once(out/"report.json",run.sealed(report))
    run.write_json(out/"run_status.json",dict(status="complete",decision=report["decision"],episodes=192))
    return dict(decision=report["decision"],summaries=summaries)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","controls","collect","audit","analyze"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    result=prepare() if args.phase=="prepare" else controls(args.resume) if args.phase=="controls" else collect(args.resume) if args.phase=="collect" else audit() if args.phase=="audit" else analyze() if args.phase=="analyze" else dict(verified=True,binding=verify()[0]["binding"])
    print(json.dumps(result),flush=True)
