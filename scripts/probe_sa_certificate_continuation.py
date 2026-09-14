"""One certified membership intervention followed by the unchanged frozen policy."""
import argparse
import multiprocessing
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import probe_sa_certificate_augmentation as previous
from scripts import diagnose_sa_pressure_300s as history
from scripts.probe_sa_rejection_branches import check_attempt
from scripts.run_feedback_exploration_diagnostics import validate_final

io=previous.io
q=history.q
OUT=ROOT/"build/sa-certificate-continuation-v1"
ARMS=("frozen","add_certified","replace_padding")


def paired_action(case,target,trial,offset,pool,index,arm,added):
    d=target["decision"]+offset
    action=q.action_for("dual16_sa",case,d,pool,index)
    if offset==0:
        selected=set(target["pool"][target["selected_index"]]["agents"])
        if arm=="add_certified":
            selected.add(added)
        elif arm=="replace_padding":
            conflict_agents={a for edge in target["edges"] for a in edge}
            padding=sorted(selected-conflict_agents)
            if not padding: raise ValueError("no nonconflicting padding available")
            selected.remove(padding[0])
            selected.add(added)
        action=dict(mode="explicit_neighborhood",agents=sorted(selected))
        seed=int(io.semantic_fingerprint([target["job_id"],20260914,trial])[:7],16)
        uniform=random.Random(seed).random()
    else:
        seed=q.seed(case["case_id"],trial,d,"pp")
        uniform=q.acceptance_draw(q.seed(case["case_id"],trial,d,"accept"))
    action["random_seed"]=seed
    return action,uniform


def prepare():
    io.require(not (OUT/"plan.json").exists(),"plan exists")
    p=previous.read_bound_plan(ROOT,previous.OUT)
    rows=previous.read_stage(p,"native")
    cases=[]
    plateau=io.read(ROOT/"build/sa-plateau-candidate-audit-v1/plan.json")
    for case in p["cases"]:
        qualified=[]
        for candidate in case["candidates"]:
            group=[r["result"] for r in rows if r["job"]["case_id"]==case["id"] and r["job"]["candidate"]["id"]==candidate["id"]]
            if (candidate["kind"]=="certificate_single" and len(group)==4 and not any(r["censored"] for r in group)
                    and sum(r["strict_drop"] for r in group)>=2):
                qualified.append((sum(r["after"] for r in group),candidate["id"],candidate))
        if qualified:
            candidate=min(qualified)[2]
            target=next(t for t in plateau["targets"] if t["job_id"]==case["id"])
            cases.append(dict(target=dict(target,edges=case["state"]["conflict_edges"]),added=candidate["added"][0],
                              selected_after_looking_at_development_results=True))
    io.require(bool(cases),"no stable native intervention")
    inputs=dict(p["inputs"])
    for name in ("scripts/probe_sa_certificate_continuation.py","tests/evaluation/test_sa_certificate_continuation.py",
                 "docs/SA_CERTIFICATE_CONTINUATION_PROTOCOL_ZH.md","build/sa-certificate-augmentation-v1/native/manifest.json",
                 "build/sa-pressure-300s-diagnostic-v1/registration.json","build/sa-pressure-300s-diagnostic-v1/manifest.json",
                 "experiments/sa_single_check_runtime.py","lns2_selector/runtime/structshell_dual16.py",
                 "lns2_selector/runtime/online_selection.py","experiments/online_feature_engine.py",
                 "scripts/run_sa_path_quality.py","scripts/run_sa_wall_clock.py",
                 "artifacts/initlns-closed-loop-controller-v2/main__realized_dynamic.json"):
        inputs[name]=io.sha256_file(ROOT/name)
    plan=dict(schema="lns2.sa_certificate_continuation.v1",cases=cases,inputs=inputs,arms=list(ARMS),trials=4,
              horizon=64,seconds=120.,node_budget=50000000,pp_seconds=5.,workers=20,fuse=180.,no_ttf=True,
              source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    plan["binding"]=io.semantic_fingerprint(plan)
    io.write(OUT/"plan.json",plan)
    print(dict(states=len(cases),branches=len(cases)*12,horizon=64,prefix=sum(c["target"]["decision"] for c in cases),no_ttf=True))


def verify():
    p=io.read(OUT/"plan.json")
    io.require(p["binding"]==io.semantic_fingerprint({k:v for k,v in p.items() if k!="binding"}),"plan changed")
    for name,sha in p["inputs"].items():
        io.require(io.sha256_file(ROOT/name)==sha,"input changed: "+name)
    return p


def branch(env,root,case,target,added,plan,arm,trial,path):
    state=root
    events=[]
    stop="horizon"
    started=time.monotonic()
    nodes0=state["low_level"]["generated"]
    selector=q.SingleFullCheckPool(case)
    for offset in range(plan["horizon"]):
        if state["feasible"]: stop="feasible";break
        if state["low_level"]["generated"]-nodes0>=plan["node_budget"]: stop="node_budget";break
        remaining=plan["seconds"]-(time.monotonic()-started)
        if remaining<=0: stop="wall_safety";break
        d=target["decision"]+offset
        index,pool=selector.select(env,state,d)
        if offset==0:
            io.require((index,pool)==(target["selected_index"],target["pool"]),"root model/pool mismatch")
        action,uniform=paired_action(case,target,trial,offset,pool,index,arm,added)
        remaining=plan["seconds"]-(time.monotonic()-started)
        if remaining<=0: stop="wall_safety";break
        temp=q.temperature(d)
        raw=q._plain(env.step_experimental_pp(action,min(plan["pp_seconds"],remaining),"annealed",temp,uniform))
        after,m=raw["observation"],raw["metrics"]
        q.validate_transition(state,after,m,action["agents"],"annealed",temp,uniform)
        events.append(dict(decision=d,action=action,temperature=temp,uniform=uniform,metrics=m,pool=pool,
                           selected_index=index,delta=q.encode_state_delta(state,after)))
        state=after
        if m["pp_failure_reason"]=="time_limit": stop="pp_safety";break
    if state["feasible"]: stop="feasible"
    validate_final(state)
    check=root
    for event in events:
        check=q.apply_state_delta(check,event["delta"])
    io.require(q.state_fingerprint(check)==q.state_fingerprint(state),"saved delta mismatch")
    row=dict(binding=plan["binding"],arm=arm,trial=trial,status="ok",events=events,
             root_fingerprint=q.state_fingerprint(root),final_fingerprint=q.state_fingerprint(state),
             final_conflicts=state["num_of_colliding_pairs"],feasible=state["feasible"],stop=stop,
             full_window=stop in ("horizon","feasible"),generated=state["low_level"]["generated"]-nodes0,
             diagnostic_seconds=time.monotonic()-started,no_ttf=True)
    row["integrity"]=io.semantic_fingerprint(row)
    io.write(path,row)


def collect():
    plan=verify()
    io.require(not (OUT/"roots").exists(),"collection exists; preserve evidence")
    sys.path.insert(0,str((ROOT/io.NATIVE).parent))
    import lns2_env
    io.require(Path(lns2_env.__file__).resolve()==(ROOT/io.NATIVE).resolve(),"wrong native")
    r=io.read(history.OUT/"registration.json")
    io.require(r["fingerprint"]==q.json_fingerprint({k:v for k,v in r.items() if k!="fingerprint"}),"source registration changed")
    anchors=history.anchors(r)
    ctx=multiprocessing.get_context("fork")
    for entry in plan["cases"]:
        target=entry["target"]
        spec=history.spec_for(r,target["item"],anchors)
        job=spec["worker_job"]
        folder=history.OUT/"episodes"/target["job_id"]
        manifest=io.read(history.OUT/"manifest.json")
        for name,sha in manifest["jobs"][target["job_id"]]["files"].items():
            io.require(io.sha256_file(folder/name)==sha,"source episode changed")
        expected=q.execution.read_artifact(folder/"initial.json",spec["binding"])["observation"]
        events=list(q.read_jsonl(folder/"first_phase/trace.jsonl"))
        env=q._make_environment(job["dataset_root"],job["row"],dict(job["environment"],time_limit=100000.),"Adaptive")
        state=q._plain(env.reset(seed=job["solver_seed"]))
        io.require(q.state_fingerprint(state)==q.state_fingerprint(expected),"initial reset mismatch")
        for d,event in enumerate(events[:target["decision"]]):
            raw=q._plain(env.step_experimental_pp(event["action"],300.,"annealed",event["temperature"],event["uniform"]))
            state=raw["observation"]
            expected=q.apply_state_delta(expected,event["delta"])
            check_attempt(raw["metrics"],event["metrics"])
            io.require(q.state_fingerprint(state)==q.state_fingerprint(expected),f"prefix mismatch {d}")
            if d%100==0: print("PREFIX",d,target["decision"],flush=True)
        io.require(q.state_fingerprint(state)==target["state_fingerprint"],"root mismatch")
        case=dict(case_id=job["sa_case_id"],task_id=job["row"]["task_id"],solver_seed=job["solver_seed"],proposal=job["sa_proposal"])
        io.require(q.SingleFullCheckPool(case).select(env,state,target["decision"])==(target["selected_index"],target["pool"]),"root pool differs")
        io.require(len(list(Path("/proc/self/task").iterdir()))==1,"fork parent must be single-threaded")
        io.write(OUT/"roots"/(target["job_id"]+".json"),state)
        active=[]
        try:
            for arm in ARMS:
                for trial in range(4):
                    path=OUT/"branches"/target["job_id"]/f"{arm}-{trial}.json"
                    proc=ctx.Process(target=branch,args=(env,state,case,target,entry["added"],plan,arm,trial,path))
                    proc.start()
                    active.append((proc,time.monotonic(),path))
            while active:
                for proc,began,path in list(active):
                    if proc.is_alive() and time.monotonic()-began<=plan["fuse"]: continue
                    if proc.is_alive():
                        proc.terminate();proc.join()
                        raise ValueError("branch hard timeout; preserve diagnostic")
                    proc.join()
                    io.require(proc.exitcode==0 and path.exists(),"branch failed")
                    row=io.read(path)
                    print("BRANCH",row["arm"],row["trial"],row["stop"],row["final_conflicts"],flush=True)
                    active.remove((proc,began,path))
                time.sleep(.5)
        finally:
            for proc,_,_ in active:
                if proc.is_alive(): proc.terminate()
                proc.join()
        io.require(q.state_fingerprint(env.get_state())==target["state_fingerprint"],"fork changed parent")
    paths=sorted((OUT/"branches").rglob("*.json"))
    io.require(len(paths)==len(plan["cases"])*12,"incomplete branches")
    io.write(OUT/"manifest.json",dict(binding=plan["binding"],files={p.relative_to(OUT).as_posix():io.sha256_file(p) for p in paths}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","collect"))
    args=p.parse_args()
    prepare() if args.stage=="prepare" else collect()
