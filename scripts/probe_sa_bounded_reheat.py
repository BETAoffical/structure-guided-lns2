"""One finite, prefix-scaled reheating pulse; diagnostic only, never a controller default."""
import argparse
import json
import math
import multiprocessing
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import diagnose_sa_pressure_300s as source
from scripts.probe_sa_rejection_branches import check_attempt
from scripts.run_feedback_exploration_diagnostics import validate_final

q=source.q
OUT=ROOT/"build/sa-bounded-reheat-probe-v1"
AUDIT="build/sa-cooling-trace-audit-v2/report.json"
AUDIT_SHA="23ac28f030a8f1c9811ad3e7ef5add9c704f9ffe5a752b43114a35eaab400098"
ARMS=("frozen","pulse16")
HORIZON=128
PULSE=16


def choose_targets(rows):
    group=[r for r in rows if r["cohort"]=="failure_union_300"]
    failures=[r for r in group if not r["success"]]
    eligible=[]
    for r in failures:
        cold=[a for a in r["attempts"] if a["probability"]<.01 and not a["accepted"] and a["best_gap_before"]>=32]
        if cold: eligible.append((len(cold),r,cold[0]["decision"]))
    primary=sorted(eligible,key=lambda x:(-x[0],x[1]["job_id"]))[:3]
    negative=[r for r in failures if not r["bins"]["all"].get("worse_rejected",0)]
    if len(primary)!=3 or len(negative)!=1: raise ValueError("fixed primary/control availability changed")
    result=[dict(role="primary",row=r,decision=d) for _,r,d in primary]
    result.append(dict(role="negative",row=negative[0],decision=1000))
    successes=sorted((r for r in group if r["success"]),key=lambda r:r["job_id"])
    if len(successes)!=5: raise ValueError("expected five retained successes")
    result.extend(dict(role="success_control",row=r,decision=max(0,r["bins"]["all"]["steps"]-64)) for r in successes)
    return result


def prefix_scale(events,decision):
    # Use the 32 preceding attempts, never the selected rejection or a successful witness.
    changes=[]
    for e in events[max(0,decision-32):decision]:
        m=e["metrics"]
        delta=m["pp_attempt_conflict_pair_count"]-m["pp_old_conflict_pair_count"]
        if m["acceptance_evaluated"] and delta>0: changes.append(delta)
    return float(statistics.median(changes)) if changes else 1.


def selected_temperature(arm,decision,offset,scale):
    if arm not in ARMS or offset<0 or scale<=0: raise ValueError("invalid pulse inputs")
    original=q.temperature(decision)
    if arm=="pulse16" and offset<PULSE:
        return max(original,scale/(-math.log(.5))*.99**offset)
    return original


def random_action(case,decision,trial,pool,index):
    action=q.action_for("dual16_sa",case,decision,pool,index)
    if trial:
        action["random_seed"]=q.seed(case["case_id"],trial,decision,"pp")
    uniform=q.acceptance_draw(q.seed(case["case_id"],trial,decision,"accept"))
    return action,uniform


def history(r,item,anchors):
    spec=source.spec_for(r,item,anchors)
    folder=source.OUT/"episodes"/item["job_id"]
    manifest=q.read_json(source.OUT/"manifest.json")
    for name,sha in manifest["jobs"][item["job_id"]]["files"].items():
        if q.sha256_file(folder/name)!=sha: raise ValueError("source episode changed")
    initial=q.execution.read_artifact(folder/"initial.json",spec["binding"])["observation"]
    return spec,initial,list(q.read_jsonl(folder/"first_phase/trace.jsonl"))


def prepare():
    if (OUT/"plan.json").exists(): raise ValueError("plan exists; preserve prior work")
    if sys.platform!="linux": raise ValueError("prepare in the original WSL checkout for source binding consistency")
    r=source.verify()
    if q.sha256_file(ROOT/AUDIT)!=AUDIT_SHA: raise ValueError("audit changed")
    anchors=source.anchors(r)
    targets=[]
    for selected in choose_targets(q.read_json(ROOT/AUDIT)["episodes"]):
        row,d=selected["row"],selected["decision"]
        identity=row["job_id"].removeprefix("failure_union_300-")
        item=next(i for i in r["schedule"] if i["job_id"]==identity)
        _,state,events=history(r,item,anchors)
        if d>=len(events): raise ValueError("target past trace")
        for e in events[:d]: state=q.apply_state_delta(state,e["delta"])
        scale=prefix_scale(events,d)
        targets.append(dict(target_id=identity,item=item,role=selected["role"],decision=d,scale=scale,
            root_fingerprint=q.state_fingerprint(state),root_conflicts=state["num_of_colliding_pairs"],
            original_temperature=q.temperature(d),pulse_temperature=selected_temperature("pulse16",d,0,scale),
            pool=events[d]["pool"],selected_index=events[d]["selected_index"]))
    names=["scripts/probe_sa_bounded_reheat.py","tests/evaluation/test_sa_bounded_reheat.py",
           "docs/SA_BOUNDED_REHEAT_PROTOCOL_ZH.md",AUDIT,
           "docs/SA_REJECTION_BRANCH_RESULTS_ZH.md","docs/STABLE_CANDIDATE_CONTINUATION_RESULT_ZH.md"]
    plan=dict(schema="lns2.sa_bounded_reheat.v1",targets=targets,arms=list(ARMS),trials=4,horizon=HORIZON,
        pulse=PULSE,workers=20,branch_workers_per_root=8,followup_seconds=180.,node_budget=50_000_000,
        pp_seconds=30.,replay_pp_seconds=300.,branch_fuse_seconds=300.,
        source_registration_sha256=q.sha256_file(source.OUT/"registration.json"),
        source_manifest_sha256=q.sha256_file(source.OUT/"manifest.json"),
        inputs={n:q.sha256_file(ROOT/n) for n in names},no_ttf=True,no_promotion=True,
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    plan["fingerprint"]=q.json_fingerprint(plan)
    q.write_json(OUT/"plan.json",plan)
    return dict(states=len(targets),branches=72,smoke_branches=2,max_followup_repairs=72*128+4,
        replay_steps=sum(t["decision"] for t in targets),targets=[{k:t[k] for k in
        ("target_id","role","decision","root_conflicts","scale","original_temperature","pulse_temperature")} for t in targets])


def verify(native=False):
    plan=q.read_json(OUT/"plan.json")
    if plan["fingerprint"]!=q.json_fingerprint({k:v for k,v in plan.items() if k!="fingerprint"}): raise ValueError("plan changed")
    for n,h in plan["inputs"].items():
        if q.sha256_file(ROOT/n)!=h: raise ValueError("registered source changed: "+n)
    for n,k in (("registration.json","source_registration_sha256"),("manifest.json","source_manifest_sha256")):
        if q.sha256_file(source.OUT/n)!=plan[k]: raise ValueError("historical registration changed")
    return plan,source.verify(native)


def branch(env,root,case,t,history_events,plan,arm,trial,horizon,path):
    try:
        if q.state_fingerprint(env.get_state())!=t["root_fingerprint"]: raise ValueError("fork root mismatch")
        selector=q.SingleFullCheckPool(case)
        state=root
        events=[]
        counts=[state["num_of_colliding_pairs"]]
        nodes0=state["low_level"]["generated"]
        started=time.monotonic()
        stop="horizon"
        source_equal_steps=0
        for offset in range(horizon):
            if state["feasible"]: stop="feasible";break
            if state["low_level"]["generated"]-nodes0>=plan["node_budget"]: stop="node_budget";break
            remaining=plan["followup_seconds"]-(time.monotonic()-started)
            if remaining<=0: stop="wall_safety";break
            d=t["decision"]+offset
            index,pool=selector.select(env,state,d)
            if offset==0 and (index,pool)!=(t["selected_index"],t["pool"]): raise ValueError("first pool/score mismatch")
            action,u=random_action(case,d,trial,pool,index)
            temp=selected_temperature(arm,d,offset,t["scale"])
            remaining=plan["followup_seconds"]-(time.monotonic()-started)
            if remaining<=0: stop="wall_safety";break
            raw=q._plain(env.step_experimental_pp(action,min(plan["pp_seconds"],remaining),"annealed",temp,u))
            after,m=raw["observation"],raw["metrics"]
            if not m["action_valid"] or not m["step_applied"]: raise ValueError("invalid action")
            if arm=="frozen" and trial==0 and d<len(history_events) and history_events[d]["metrics"]["pp_failure_reason"]!="time_limit":
                original=history_events[d]
                if (action,index,pool,u,temp)!=(original["action"],original["selected_index"],original["pool"],original["uniform"],original["temperature"]):
                    raise ValueError("frozen continuation differs from source")
                check_attempt(m,original["metrics"])
                if q.state_fingerprint(after)!=q.state_fingerprint(q.apply_state_delta(state,original["delta"])):
                    raise ValueError("original continuation state mismatch")
                source_equal_steps+=1
            events.append(dict(decision=d,action=action,metrics=m,pool=pool,selected_index=index,
                temperature=temp,uniform=u,delta=q.encode_state_delta(state,after)))
            state=after
            counts.append(state["num_of_colliding_pairs"])
            if state["feasible"]: stop="feasible";break
            if raw["truncated"] or m["pp_failure_reason"]=="time_limit": stop="pp_safety";break
        elapsed=time.monotonic()-started
        checked=root
        for e in events:
            after=q.apply_state_delta(checked,e["delta"])
            q.validate_transition(checked,after,e["metrics"],e["action"]["agents"],"annealed",e["temperature"],e["uniform"])
            checked=after
        validate_final(state)
        if q.state_fingerprint(checked)!=q.state_fingerprint(state): raise ValueError("saved rollout mismatch")
        padded=counts+[0 if state["feasible"] else counts[-1]]*(horizon+1-len(counts))
        row=dict(status="ok",target_id=t["target_id"],role=t["role"],arm=arm,trial=trial,horizon=horizon,
            root_fingerprint=t["root_fingerprint"],events=events,counts=counts,feasible=state["feasible"],
            final_fingerprint=q.state_fingerprint(state),stop=stop,generated=state["low_level"]["generated"]-nodes0,
            node_cap_overshoot=max(0,state["low_level"]["generated"]-nodes0-plan["node_budget"]),
            diagnostic_seconds=elapsed,source_equal_steps=source_equal_steps,
            full_window=stop in ("horizon","feasible"),descriptive_padded_auc=sum((a+b)/2 for a,b in zip(padded,padded[1:])),
            temperature_exposed_steps=sum(e["temperature"]>q.temperature(e["decision"]) for e in events),no_ttf=True)
    except Exception as exc:
        row=dict(status="error",target_id=t["target_id"],arm=arm,trial=trial,error=repr(exc))
    row["binding"]=plan["fingerprint"]
    row["integrity_sha256"]=q.json_fingerprint(row)
    q.write_json(path,row)


def checked_row(path,plan):
    row=q.read_json(path)
    if row.get("integrity_sha256")!=q.json_fingerprint({k:v for k,v in row.items() if k!="integrity_sha256"}): raise ValueError("branch SHA changed")
    if row["binding"]!=plan["fingerprint"] or row["status"]!="ok": raise ValueError("branch failed: "+str(path))
    return row


def fork_batch(env,root,case,t,events,plan,phase,resume):
    if sys.platform!="linux" or len(list(Path("/proc/self/task").iterdir()))!=1: raise ValueError("fork requires single-threaded Linux parent")
    ctx=multiprocessing.get_context("fork")
    folder=OUT/phase/t["target_id"]
    folder.mkdir(parents=True,exist_ok=True)
    pending=[]
    for trial in ([0] if phase=="smoke" else range(4)):
        for arm in ARMS:
            path=folder/f"{arm}-t{trial}.json"
            if path.exists():
                if not resume: raise ValueError("explicit resume required")
                checked_row(path,plan)
            else: pending.append((arm,trial,path))
    active=[]
    try:
        while pending or active:
            while pending and len(active)<plan["workers"] and not (OUT/"STOP_AFTER_JOB.json").exists():
                arm,trial,path=pending.pop(0)
                process=ctx.Process(target=branch,args=(env,root,case,t,events,plan,arm,trial,2 if phase=="smoke" else HORIZON,path))
                process.start();active.append((process,time.monotonic(),path))
            for entry in list(active):
                process,began,path=entry
                if process.is_alive() and time.monotonic()-began<=plan["branch_fuse_seconds"]: continue
                if process.is_alive():
                    process.terminate();process.join()
                    q.write_json(path,dict(status="hard_timeout",binding=plan["fingerprint"]))
                    raise ValueError("branch hard timeout; retain as unknown")
                process.join()
                if process.exitcode!=0: raise ValueError("branch process failed")
                row=checked_row(path,plan)
                print(phase,t["target_id"],row["arm"],row["trial"],row["stop"],row["counts"][-1],flush=True)
                active.remove(entry)
            q.write_json(OUT/"status.json",dict(status="running",phase=phase,target=t["target_id"],active=len(active),pending=len(pending)))
            if not active and pending and (OUT/"STOP_AFTER_JOB.json").exists(): return False
            time.sleep(1.)
    finally:
        for process,_,_ in active:
            if process.is_alive(): process.terminate()
            process.join()
    if q.state_fingerprint(env.get_state())!=t["root_fingerprint"]: raise ValueError("child changed parent")
    return True


def collect(resume=False):
    plan,r=verify(True)
    anchors=source.anchors(r)
    with q._CollectionRunLock(OUT,plan["fingerprint"],"bounded-reheat"):
        try:
            for number,t in enumerate(plan["targets"]):
                if (OUT/"STOP_AFTER_JOB.json").exists(): return dict(paused=True)
                spec,expected,events=history(r,t["item"],anchors)
                job=spec["worker_job"]
                env=q._make_environment(job["dataset_root"],job["row"],dict(job["environment"],time_limit=100000.),"Adaptive")
                state=q._plain(env.reset(seed=job["solver_seed"]))
                if q.state_fingerprint(state)!=q.state_fingerprint(expected): raise ValueError("reset mismatch")
                print("REPLAY",number+1,len(plan["targets"]),t["role"],t["decision"],flush=True)
                for d,e in enumerate(events[:t["decision"]]):
                    raw=q._plain(env.step_experimental_pp(e["action"],plan["replay_pp_seconds"],"annealed",e["temperature"],e["uniform"]))
                    state=raw["observation"];expected=q.apply_state_delta(expected,e["delta"])
                    check_attempt(raw["metrics"],e["metrics"])
                    if q.state_fingerprint(state)!=q.state_fingerprint(expected): raise ValueError(f"prefix mismatch {d}")
                    if d%200==0: print("PREFIX",d,flush=True)
                if q.state_fingerprint(state)!=t["root_fingerprint"]: raise ValueError("root mismatch")
                validate_final(state)
                case=dict(case_id=job["sa_case_id"],task_id=job["row"]["task_id"],solver_seed=job["solver_seed"],proposal=job["sa_proposal"])
                if q.SingleFullCheckPool(case).select(env,state,t["decision"])!=(t["selected_index"],t["pool"]): raise ValueError("root pool mismatch")
                q.write_json(OUT/"roots"/(t["target_id"]+".json"),state)
                if number==0 and not fork_batch(env,state,case,t,events,plan,"smoke",resume): return dict(paused=True)
                if not fork_batch(env,state,case,t,events,plan,"branches",resume): return dict(paused=True)
            result=analyze()
            q.write_json(OUT/"status.json",dict(status="complete",branches=72))
            return result
        except BaseException as exc:
            q.write_json(OUT/"status.json",dict(status="failed_or_interrupted",error=repr(exc)))
            raise


def pair_summary(a,b):
    if a["root_fingerprint"]!=b["root_fingerprint"] or a["trial"]!=b["trial"]: raise ValueError("pair identity mismatch")
    return dict(trial=a["trial"],frozen_feasible=a["feasible"],pulse_feasible=b["feasible"],
        gain=not a["feasible"] and b["feasible"],loss=a["feasible"] and not b["feasible"],
        paired_full_window=a["full_window"] and b["full_window"],
        frozen_final=a["counts"][-1],pulse_final=b["counts"][-1],
        frozen_generated=a["generated"],pulse_generated=b["generated"],
        frozen_steps=len(a["events"]),pulse_steps=len(b["events"]),
        temperature_exposed_steps=b["temperature_exposed_steps"],
        same_final_fingerprint=a["final_fingerprint"]==b["final_fingerprint"])


def analyze():
    plan,r=verify()
    results=[];files={}
    for t in plan["targets"]:
        root=q.read_json(OUT/"roots"/(t["target_id"]+".json"))
        if q.state_fingerprint(root)!=t["root_fingerprint"]: raise ValueError("stored root changed")
        pairs=[]
        for trial in range(4):
            rows=[]
            for arm in ARMS:
                path=OUT/"branches"/t["target_id"]/f"{arm}-t{trial}.json"
                row=checked_row(path,plan);state=root;counts=[root["num_of_colliding_pairs"]]
                if (row["target_id"],row["arm"],row["trial"],row["horizon"])!=(t["target_id"],arm,trial,HORIZON): raise ValueError("branch identity mismatch")
                case=dict(case_id=f"{t['item']['task_id']}-seed{t['item']['solver_seed']}")
                for offset,e in enumerate(row["events"]):
                    d=t["decision"]+offset
                    if e["decision"]!=d or e["temperature"]!=selected_temperature(arm,d,offset,t["scale"]): raise ValueError("pulse schedule changed")
                    if (e["action"],e["uniform"])!=random_action(case,d,trial,e["pool"],e["selected_index"]): raise ValueError("paired RNG changed")
                    after=q.apply_state_delta(state,e["delta"])
                    q.validate_transition(state,after,e["metrics"],e["action"]["agents"],"annealed",e["temperature"],e["uniform"])
                    state=after;counts.append(state["num_of_colliding_pairs"])
                validate_final(state)
                if counts!=row["counts"] or q.state_fingerprint(state)!=row["final_fingerprint"]: raise ValueError("outcome trace changed")
                files[path.relative_to(OUT).as_posix()]=q.sha256_file(path);rows.append(row)
            pairs.append(pair_summary(*rows))
        results.append(dict(target_id=t["target_id"],role=t["role"],task_id=t["item"]["task_id"],
            solver_seed=t["item"]["solver_seed"],decision=t["decision"],root_conflicts=t["root_conflicts"],scale=t["scale"],pairs=pairs))
    summary={}
    for role in ("primary","negative","success_control"):
        targets=[t for t in results if t["role"]==role];pairs=[p for t in targets for p in t["pairs"]]
        summary[role]=dict(states=len(targets),pairs=len(pairs),frozen_success=sum(p["frozen_feasible"] for p in pairs),
            pulse_success=sum(p["pulse_feasible"] for p in pairs),gains=sum(p["gain"] for p in pairs),losses=sum(p["loss"] for p in pairs),
            censored_pairs=sum(not p["paired_full_window"] for p in pairs),
            exposed_pairs=sum(p["temperature_exposed_steps"]>0 for p in pairs),
            gain_states=sum(any(p["gain"] for p in t["pairs"]) for t in targets))
    primary=summary["primary"]
    decision=("resource_inconclusive" if any(v["censored_pairs"] for v in summary.values()) else
        "mechanism_signal_requires_confirmation" if primary["gains"]-primary["losses"]>=2 and primary["gain_states"]>=2 else
        "localized_completion_signal" if primary["gains"]>primary["losses"] else "no_completion_benefit")
    report=dict(schema="lns2.sa_bounded_reheat_report.v1",complete=True,binding=plan["fingerprint"],
        states=results,summary=summary,decision=decision,files=files,no_ttf=True,no_promotion=True)
    q.write_json(OUT/"report.json",report)
    return dict(complete=True,branches=72,summary=summary,decision=decision)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","collect","analyze"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    print(json.dumps(prepare() if args.phase=="prepare" else collect(args.resume) if args.phase=="collect" else analyze()),flush=True)
