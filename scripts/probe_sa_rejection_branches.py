"""Bounded one-decision acceptance interventions; not TTF or model training."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json,sha256_file,write_json
from experiments.sa_single_check_runtime import SingleFullCheckPool
from scripts import run_sa_supervised_confirmation as supervision
from scripts import audit_sa_trace_opportunities as audit

source,pilot=supervision.source,supervision.pilot
OUT=ROOT/"build/sa-rejection-branch-probe-v1"
AUDIT="build/sa-trace-opportunity-audit-v1/report.json"
AUDIT_SHA="faae54d725157af9d72eab2fca054ecd10983067e216cfa9517a1b685998336e"
MODES=("keep_rejection","accept_once")
HORIZON=64
NODE_BUDGET=5_000_000
FOLLOWUP_SECONDS=120.


def selected_events(report):
    rows=[r for r in report["attempts"] if r["arm"]=="dual16_sa" and not r["accepted"]]
    rows=sorted(rows,key=lambda r:(r["case_id"],r["decision"]))
    if len(rows)!=9 or len({r["case_id"] for r in rows})!=5 or len({(r["case_id"],r["decision"]) for r in rows})!=9:
        raise ValueError("fixed rejection set changed")
    return rows


def prepare():
    if (OUT/"plan.json").exists(): raise ValueError("plan exists; use verify")
    if sha256_file(ROOT/AUDIT)!=AUDIT_SHA: raise ValueError("audit changed")
    plan=source.verify()
    cases={c["case_id"]:c for c in plan["cases"]}
    manifest=read_json(ROOT/audit.SOURCE)
    if sha256_file(ROOT/audit.SOURCE)!=audit.SOURCE_SHA: raise ValueError("trace manifest changed")
    records={}
    for n,h in manifest["files"].items():
        if n.endswith("-dual16_sa.json"):
            records[Path(n).name.removesuffix("-dual16_sa.json")]=dict(path=n,sha256=h)
    targets=[]
    for item in selected_events(read_json(ROOT/AUDIT)):
        record=records[item["case_id"]]
        if sha256_file(ROOT/record["path"])!=record["sha256"]: raise ValueError("trace changed")
        saved=read_json(ROOT/record["path"])
        e=saved["events"][item["decision"]]
        m=e["metrics"]
        if m["pp_failure_reason"]!="acceptance_rejected" or m["replan_success"] or m["acceptance_probability"]<=0:
            raise ValueError("not a reproducible positive-probability rejection")
        targets.append(dict(target_id=f"{item['case_id']}-d{item['decision']:04d}",
            case=cases[item["case_id"]],decision=item["decision"],record=record))
    names=[AUDIT,audit.SOURCE,"scripts/probe_sa_rejection_branches.py",
        "tests/evaluation/test_sa_rejection_branches.py","docs/SA_REJECTION_BRANCH_PROTOCOL_ZH.md",
        "experiments/sa_single_check_runtime.py", "experiments/nonmonotonic_repair.py"]
    payload=dict(schema="lns2.sa_rejection_branch.v1",config=plan["config"],targets=targets,
        modes=list(MODES),horizon=HORIZON,node_budget=NODE_BUDGET,followup_seconds=FOLLOWUP_SECONDS,
        node_limit_checked_between_repairs=True,per_pp_seconds=30.,replay_pp_seconds=300.,
        environment_seconds=3000.,workers=20,job_fuse_seconds=900.,jobs=18,
        source_plan_sha256=sha256_file(source.OUT/"plan.json"),
        inputs={n:sha256_file(ROOT/n) for n in names},
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        development_only=True,no_ttf=True,no_default_change=True)
    write_json(OUT/"plan.json",payload)
    return dict(targets=9,branches=18,smoke_branches=2,max_followup_repairs=18*64,
                workers=20,active_workers_at_most=18,timing_experiment=False)


def verify():
    frozen=source.verify()
    p=read_json(OUT/"plan.json")
    for n,h in p["inputs"].items():
        if sha256_file(ROOT/n)!=h: raise ValueError("registered source changed: "+n)
    if p["config"]!=frozen["config"] or p["source_plan_sha256"]!=sha256_file(source.OUT/"plan.json"):
        raise ValueError("frozen configuration mismatch")
    if (p["horizon"],p["node_budget"],p["followup_seconds"],p["modes"],p["workers"])!=(HORIZON,NODE_BUDGET,FOLLOWUP_SECONDS,list(MODES),20):
        raise ValueError("budget or mode changed")
    expected=[(r["case_id"],r["decision"]) for r in selected_events(read_json(ROOT/AUDIT))]
    if [(t["case"]["case_id"],t["decision"]) for t in p["targets"]]!=expected: raise ValueError("target set mismatch")
    return p


def jobs(plan,phase):
    targets=plan["targets"][:1] if phase=="smoke" else plan["targets"]
    return [dict(job_id=t["target_id"]+"-"+m,case=t["case"],target=t,mode=m,config=plan["config"],
        plan=plan,phase=phase,horizon=2 if phase=="smoke" else plan["horizon"],
        plan_sha256=sha256_file(OUT/"plan.json")) for t in targets for m in MODES]


def check_attempt(actual,expected):
    for k in ("repair_order","neighborhood","pp_attempted_agent_count","pp_inserted_agent_count",
              "pp_old_conflict_pair_count","pp_attempt_conflict_pair_count"):
        if actual[k]!=expected[k]: raise ValueError("PP attempt mismatch: "+k)


def forced_draw(mode,event):
    if mode not in MODES: raise ValueError("unknown intervention")
    if event["metrics"]["acceptance_probability"]<=0: raise ValueError("cannot force zero-probability attempt")
    return 0. if mode=="accept_once" else event["uniform"]


def worker(job):
    if (OUT/"STOP_AFTER_JOB.json").exists():
        return dict(status="paused",job_id=job["job_id"],plan_sha256=job["plan_sha256"])
    began=time.monotonic()
    t,p=job["target"],job["plan"]
    record=t["record"]
    if sha256_file(ROOT/record["path"])!=record["sha256"]: raise ValueError("source trace changed")
    saved=read_json(ROOT/record["path"])
    env=pilot.make_env(dict(job,budget=p["environment_seconds"]))
    state=pilot._plain(env.reset(seed=job["case"]["solver_seed"]))
    expected=saved["initial_state"]
    if pilot.state_fingerprint(state)!=pilot.state_fingerprint(expected): raise ValueError("initial mismatch")
    for d in range(t["decision"]):
        e=saved["events"][d]
        step=pilot._plain(env.step_experimental_pp(e["action"],p["replay_pp_seconds"],"annealed",e["temperature"],e["uniform"]))
        state=step["observation"]
        expected=pilot.apply_state_delta(expected,e["delta"])
        if pilot.state_fingerprint(state)!=pilot.state_fingerprint(expected): raise ValueError(f"prefix mismatch at {d}")
        check_attempt(step["metrics"],e["metrics"])
    selector=SingleFullCheckPool(job["case"])
    e=saved["events"][t["decision"]]
    index,pool=selector.select(env,state,t["decision"])
    if (index,pool)!=(e["selected_index"],e["pool"]): raise ValueError("target candidate/score mismatch")
    before=deepcopy(state)
    draw=forced_draw(job["mode"],e)
    step=pilot._plain(env.step_experimental_pp(e["action"],p["replay_pp_seconds"],"annealed",e["temperature"],draw))
    state,m=step["observation"],step["metrics"]
    check_attempt(m,e["metrics"])
    original_after=pilot.apply_state_delta(expected,e["delta"])
    if state["low_level"]!=original_after["low_level"]: raise ValueError("target low-level counters mismatch")
    if job["mode"]=="keep_rejection":
        if pilot.state_fingerprint(state)!=pilot.state_fingerprint(original_after): raise ValueError("original rejection mismatch")
    elif not m["replan_success"] or m["pp_rolled_back"]:
        raise ValueError("intervention not accepted")
    pilot.validate_transition(before,state,m,m["neighborhood"],"annealed",e["temperature"],draw)
    pilot.validate_final(before)
    pilot.validate_final(state)
    target_after=deepcopy(state)
    followup_started=time.monotonic()
    start_nodes=state["low_level"]["generated"]
    events=[]
    stop="horizon"
    conflicts=[state["num_of_colliding_pairs"]]
    while len(events)<job["horizon"]:
        if state["feasible"]: stop="feasible"; break
        if state["low_level"]["generated"]-start_nodes>=p["node_budget"]: stop="node_budget"; break
        remaining=p["followup_seconds"]-(time.monotonic()-followup_started)
        if remaining<=0: stop="wall_safety_limit"; break
        d=t["decision"]+1+len(events)
        index,pool=selector.select(env,state,d)
        remaining=p["followup_seconds"]-(time.monotonic()-followup_started)
        if remaining<=0: stop="wall_safety_limit"; break
        action=pilot.action_for("dual16_sa",job["case"],d,pool,index)
        temp=pilot.temperature(d)
        u=pilot.acceptance_draw(pilot.seed(job["case"]["case_id"],0,d,"accept"))
        old=state
        step=pilot._plain(env.step_experimental_pp(action,min(p["per_pp_seconds"],remaining),"annealed",temp,u))
        state=step["observation"]
        events.append(dict(decision=d,action=action,temperature=temp,uniform=u,
            metrics=step["metrics"],selected_index=index,pool=pool,delta=pilot.encode_state_delta(old,state)))
        conflicts.append(state["num_of_colliding_pairs"])
        if state["feasible"]: stop="feasible"; break
        if step["metrics"]["pp_failure_reason"]=="time_limit" or step["truncated"]:
            stop="pp_safety_limit"; break
    elapsed=time.monotonic()-followup_started
    # Validate outside the rollout safety clock; never trade checks for horizon.
    checked=target_after
    for event in events:
        after=pilot.apply_state_delta(checked,event["delta"])
        pilot.validate_transition(checked,after,event["metrics"],event["metrics"]["neighborhood"],
            "annealed",event["temperature"],event["uniform"])
        pilot.validate_final(after)
        checked=after
    if pilot.state_fingerprint(checked)!=pilot.state_fingerprint(state): raise ValueError("saved rollout mismatch")
    return dict(status="ok",job_id=job["job_id"],phase=job["phase"],mode=job["mode"],target_id=t["target_id"],
        case_id=job["case"]["case_id"],map_id=job["case"]["map_id"],decision=t["decision"],
        plan_sha256=job["plan_sha256"],source_sha256=record["sha256"],prefix_verified=True,
        before=before,target_after=target_after,target_metrics=m,original_uniform=e["uniform"],intervention_uniform=draw,
        events=events,final=state,conflicts=conflicts,stop=stop,feasible=state["feasible"],
        followup_steps=len(events),followup_generated=state["low_level"]["generated"]-start_nodes,
        node_cap_overshoot=max(0,state["low_level"]["generated"]-start_nodes-p["node_budget"]),
        followup_safety_seconds=elapsed,job_seconds=time.monotonic()-began,not_ttf=True)


def failure(job,status,error):
    return dict(status=status,job_id=job["job_id"],plan_sha256=job["plan_sha256"],error=str(error),automatic_retry=False)


def load(job,phase):
    row=read_json(OUT/phase/(job["job_id"]+".json"))
    if row.get("integrity_sha256")!=pilot.digest({k:v for k,v in row.items() if k!="integrity_sha256"}): raise ValueError("result hash mismatch")
    for k in ("job_id","plan_sha256","phase","mode"):
        if row.get(k)!=job[k]: raise ValueError("result identity mismatch: "+k)
    if row["status"]!="ok": raise ValueError("prior error needs inspection; no automatic retry")
    return row


def collect(phase,resume=False):
    plan=verify()
    if phase=="branches":
        smoke=read_json(OUT/"smoke_report.json")
        if not smoke["complete"] or smoke["plan_sha256"]!=sha256_file(OUT/"plan.json"): raise ValueError("smoke incomplete")
        for n,h in smoke["files"].items():
            if sha256_file(OUT/"smoke"/n)!=h: raise ValueError("smoke changed")
    scheduled=jobs(plan,phase)
    with pilot._CollectionRunLock(OUT,sha256_file(OUT/"plan.json"),phase):
        if resume: (OUT/"STOP_AFTER_JOB.json").unlink(missing_ok=True)
        pending=[]
        for job in scheduled:
            if (OUT/phase/(job["job_id"]+".json")).exists(): load(job,phase)
            else: pending.append(job)
        def save(r):
            if r["status"]=="paused":
                write_json(OUT/"paused"/(r["job_id"]+".json"),r)
                return
            r["integrity_sha256"]=pilot.digest(r)
            write_json(OUT/phase/(r["job_id"]+".json"),r)
            print(json.dumps({k:r.get(k) for k in ("job_id","status","feasible","followup_steps","stop","error")}),flush=True)
        pilot._run_jobs(worker,pending,workers=plan["workers"],phase=phase,output_root=OUT,
            run_fingerprint=sha256_file(OUT/"plan.json"),timeout_seconds=plan["job_fuse_seconds"],
            failure_result=failure,on_result=save,stop_on_failure=True)
        if any(not (OUT/phase/(j["job_id"]+".json")).exists() for j in scheduled):
            return dict(paused_or_incomplete=True)
        rows=[load(j,phase) for j in scheduled]
        comparisons=compare(rows)
        report=dict(complete=True,phase=phase,plan_sha256=sha256_file(OUT/"plan.json"),jobs=len(rows),comparisons=comparisons,
            files={j["job_id"]+".json":sha256_file(OUT/phase/(j["job_id"]+".json")) for j in scheduled},
            no_ttf=True,no_default_promotion=True)
        write_json(OUT/(phase+"_report.json"),report)
        return report


def compare(rows):
    groups={}
    for r in rows:
        group=groups.setdefault(r["target_id"],{})
        if r["mode"] in group: raise ValueError("duplicate branch")
        group[r["mode"]]=r
    results=[]
    for key,g in sorted(groups.items()):
        if set(g)!=set(MODES): raise ValueError("missing paired branch")
        a,b=[g[m] for m in MODES]
        if pilot.state_fingerprint(a["before"])!=pilot.state_fingerprint(b["before"]): raise ValueError("branch root mismatch")
        check_attempt(a["target_metrics"],b["target_metrics"])
        if a["target_after"]["low_level"]!=b["target_after"]["low_level"]: raise ValueError("paired PP nodes differ")
        results.append(dict(target_id=key,map_id=a["map_id"],case_id=a["case_id"],decision=a["decision"],
            root_conflicts=a["before"]["num_of_colliding_pairs"],
            results={m:dict(feasible=g[m]["feasible"],stop=g[m]["stop"],steps=g[m]["followup_steps"],
                generated=g[m]["followup_generated"],final_conflicts=g[m]["conflicts"][-1],
                full_horizon_or_success=g[m]["stop"] in ("horizon","feasible")) for m in MODES}))
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","verify","smoke","collect","resume","stop"))
    args=p.parse_args()
    if args.phase=="stop":
        write_json(OUT/"STOP_AFTER_JOB.json",dict(requested=True)); result=dict(stop_after_running_jobs=True)
    elif args.phase in ("smoke","collect","resume"): result=collect("smoke" if args.phase=="smoke" else "branches",args.phase=="resume")
    elif args.phase=="verify": result=dict(verified=bool(verify()))
    else: result=prepare()
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
