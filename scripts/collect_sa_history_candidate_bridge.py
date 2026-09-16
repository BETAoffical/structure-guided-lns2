"""Isolated paired SA continuations after a frozen candidate preflight."""
import argparse
from itertools import combinations
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json,sha256_file,json_fingerprint
from experiments.sa_history_information import future_labels
from scripts.audit_sa_history_information import atomic,require
from scripts.preflight_sa_history_candidate_bridge import CONFIG
from scripts.train_sa_history_selector import die_with_parent


def locations():
    cfg=read_json(CONFIG)
    return cfg,ROOT/cfg["output"]


def prepare():
    cfg,out=locations()
    require(not (out/"collection_plan.json").exists(),"collection plan exists")
    pre=read_json(out/"preflight.json")
    require(pre["admission"]["passed"],"candidate contrast admission failed")
    base=read_json(out/"preflight_plan.json")
    require(base["binding"]==json_fingerprint({k:v for k,v in base.items() if k!="binding"}),"preflight plan identity")
    require(pre["binding"]==base["binding"] and cfg==base["config"],"preflight/config binding")
    inputs=dict(base["inputs"])
    for name in ("preflight.json","preflight_plan.json"):
        p=out/name
        inputs[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    for p in [Path(__file__),ROOT/"tests/evaluation/test_sa_history_bridge_collection.py"]:
        inputs[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    plan=dict(config=cfg,inputs=inputs,roots=pre["roots"],
              commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    plan["binding"]=json_fingerprint(plan)
    atomic(out/"collection_plan.json",plan)
    verify()
    return dict(roots=len(plan["roots"]),trials=sum(len(r["selected"]) for r in plan["roots"])*cfg["trials"],
                max_repairs=sum(len(r["selected"]) for r in plan["roots"])*cfg["trials"]*cfg["horizon"],
                prefix_steps=sum(r["target"]["decision"] for r in plan["roots"]),workers=cfg["workers"],no_ttf=True)


def verify():
    cfg,out=locations()
    plan=read_json(out/"collection_plan.json")
    require(plan["binding"]==json_fingerprint({k:v for k,v in plan.items() if k!="binding"}),"plan fingerprint")
    require(plan["config"]==cfg,"config changed")
    for name,digest in plan["inputs"].items():
        require(sha256_file(ROOT/name)==digest,"registered input changed: "+name)
    return plan,out


def randomization(root_id,trial,absolute_decision,cfg):
    from scripts import run_sa_path_quality as q
    pp=q.seed(root_id,cfg["seed"]+trial,absolute_decision,"pp")
    uniform=q.acceptance_draw(q.seed(root_id,cfg["seed"]+trial,absolute_decision,"accept"))
    return pp,uniform


def frozen_index(pool):
    return min(range(len(pool)),key=lambda i:(-round(pool[i]["score"],12),pool[i]["candidate_id"]))


def branch(env,root,case,entry,candidate,trial,plan,path,parent_pid):
    die_with_parent(parent_pid)
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from scripts.run_feedback_exploration_diagnostics import validate_final
    cfg=plan["config"]
    try:
        state=root["state"]
        require(q.state_fingerprint(env.get_state())==root["state_fingerprint"],"fork root changed")
        selector=q.SingleFullCheckPool(case)
        events=[]
        begun=time.monotonic()
        initial_nodes=state["low_level"]["generated"]
        stop="horizon"
        for offset in range(1 if trial==-1 else cfg["horizon"]):
            if state["feasible"]: stop="feasible";break
            if state["low_level"]["generated"]-initial_nodes>=cfg["node_budget"]: stop="node_budget";break
            if time.monotonic()-begun>=cfg["branch_seconds"]: stop="wall_safety";break
            d=entry["target"]["decision"]+offset
            if offset==0:
                pool=root["control_event"]["pool"]
                index=next(i for i,c in enumerate(pool) if c["candidate_id"]==candidate["candidate_id"])
            else:
                index,pool=selector.select(env,state,d)
            pp,uniform=randomization(entry["target"]["id"],trial,d,cfg)
            temp=q.temperature(d)
            action=dict(mode="explicit_neighborhood",agents=pool[index]["agents"],random_seed=pp)
            if trial==-1:
                event=root["control_event"]
                action,temp,uniform=event["action"],event["temperature"],event["uniform"]
            remaining=cfg["branch_seconds"]-(time.monotonic()-begun)
            if remaining<=0: stop="wall_safety";break
            raw=q._plain(env.step_experimental_pp(action,min(cfg["pp_seconds"],remaining),"annealed",temp,uniform))
            after,m=raw["observation"],raw["metrics"]
            q.validate_transition(state,after,m,action["agents"],"annealed",temp,uniform)
            events.append(dict(decision=d,action=action,temperature=temp,uniform=uniform,metrics=m,pool=pool,
                               selected_index=index,delta=q.encode_state_delta(state,after)))
            state=after
            if trial==-1:
                check_attempt(m,event["metrics"])
                require(q.state_fingerprint(state)==q.state_fingerprint(q.apply_state_delta(root["state"],event["delta"])),"original control mismatch")
                stop="control"
            if m["pp_failure_reason"]=="time_limit" or not m["acceptance_evaluated"]: stop="incomplete_pp";break
        if trial!=-1 and state["feasible"]: stop="feasible"
        validate_final(state)
        row=dict(status="ok",binding=plan["binding"],root_id=entry["target"]["id"],candidate_id=candidate["candidate_id"],
                 trial=trial,events=events,stop=stop,root_fingerprint=root["state_fingerprint"],
                 final_fingerprint=q.state_fingerprint(state),final_conflicts=state["num_of_colliding_pairs"],
                 final_cost=state["sum_of_costs"],generated=state["low_level"]["generated"]-initial_nodes,
                 diagnostic_seconds=time.monotonic()-begun,no_ttf=True)
    except Exception as exc:
        row=dict(status="error",binding=plan["binding"],error=repr(exc))
    atomic(path,row)


def expected_files(entry,cfg):
    return {"control.json"}|{f"{c}-t{t}.json" for c in entry["selected"] for t in range(cfg["trials"])}


def check_root_receipt(folder,entry,plan):
    receipt=read_json(folder/"receipt.json")
    require(receipt["binding"]==plan["binding"],"root receipt binding")
    require(set(receipt["files"])==expected_files(entry,plan["config"]),"root receipt incomplete")
    for name,digest in receipt["files"].items():
        require(sha256_file(folder/name)==digest,"branch bytes changed")
        require(read_json(folder/name)["binding"]==plan["binding"],"branch binding")


def root_worker(job):
    die_with_parent(job["parent_pid"])
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    plan,entry=job["plan"],job["entry"]
    cfg,out=locations()
    target=entry["target"]
    folder=out/"branches"/target["id"]
    if (folder/"receipt.json").exists():
        check_root_receipt(folder,entry,plan)
        return dict(status="ok",id=target["id"],resumed=True)
    require(not folder.exists(),"partial root requires audit before retry")
    sampling_cfg=read_json(ROOT/cfg["sampling_config"])
    source=ROOT/sampling_cfg["source"]
    root_path=ROOT/sampling_cfg["output"]/"states"/target["id"]/"root.json"
    require(sha256_file(root_path)==entry["root_sha"],"root input changed")
    root=read_json(root_path)
    require(q.native_identity()["sha256"]==q.NATIVE_SHA,"wrong frozen native")
    reg=read_json(source/"registration.json")
    case=next(c for c in reg["cases"] if c["task_id"]==target["item"]["task_id"])
    worker=q.worker_job(case,target["item"],reg["template"],folder/"unused",plan["binding"])
    original=source/"episodes"/target["item"]["job_id"]
    expected=read_json(original/"initial.json")["payload"]["observation"]
    env=q._make_environment(worker["dataset_root"],worker["row"],dict(worker["environment"],time_limit=100000.),"Adaptive")
    state=q._plain(env.reset(seed=worker["solver_seed"]))
    require(q.state_fingerprint(state)==q.state_fingerprint(expected),"initial mismatch")
    with (original/"first_phase/trace.jsonl").open(encoding="utf8") as f:
        for line in f:
            e=json.loads(line)
            if e["decision"]==target["decision"]: break
            raw=q._plain(env.step_experimental_pp(e["action"],sampling_cfg["replay_pp_seconds"],"annealed",e["temperature"],e["uniform"]))
            state=raw["observation"]
            expected=q.apply_state_delta(expected,e["delta"])
            check_attempt(raw["metrics"],e["metrics"])
            require(q.state_fingerprint(state)==q.state_fingerprint(expected),"prefix mismatch")
    require(q.state_fingerprint(state)==entry["state_fingerprint"],"root fingerprint mismatch")
    source_case=dict(case_id=worker["sa_case_id"],task_id=case["task_id"],solver_seed=worker["solver_seed"],proposal=worker["sa_proposal"])
    require(q.SingleFullCheckPool(source_case).select(env,state,target["decision"])==
            (root["control_event"]["selected_index"],root["control_event"]["pool"]),"root candidate mismatch")
    require(len(list(Path("/proc/self/task").iterdir()))==1,"fork parent must be single threaded")
    folder.mkdir(parents=True)
    lookup={c["candidate_id"]:c for c in root["candidates"]}
    jobs=[(lookup[root["old_selected_id"]],-1)]+[(lookup[c],t) for c in entry["selected"] for t in range(cfg["trials"])]
    active=[]
    context=multiprocessing.get_context("fork")
    try:
        while jobs or active:
            while jobs and len(active)<cfg["workers"]:
                candidate,trial=jobs.pop(0)
                path=folder/("control.json" if trial==-1 else f"{candidate['candidate_id']}-t{trial}.json")
                proc=context.Process(target=branch,args=(env,root,source_case,entry,candidate,trial,plan,path,os.getpid()))
                proc.start()
                active.append((proc,time.monotonic(),path))
            for item in list(active):
                proc,start,path=item
                if proc.is_alive() and time.monotonic()-start<cfg["branch_fuse_seconds"]: continue
                if proc.is_alive():
                    proc.terminate();proc.join()
                    atomic(path,dict(status="censored",binding=plan["binding"],stop="hard_fuse"))
                else:
                    proc.join()
                    require(proc.exitcode==0 and path.exists(),"branch crashed")
                    require(read_json(path)["status"]=="ok","branch reported error: "+str(path))
                active.remove(item)
                print("BRANCH",target["id"],path.name,read_json(path)["stop"],flush=True)
            time.sleep(.05)
    finally:
        for proc,_,_ in active:
            if proc.is_alive(): proc.terminate()
            proc.join()
    require(q.state_fingerprint(env.get_state())==entry["state_fingerprint"],"parent mutated by branch")
    require(read_json(folder/"control.json").get("stop")=="control","original-action control incomplete")
    files=expected_files(entry,cfg)
    require({p.name for p in folder.glob("*.json")}==files,"unexpected branch files")
    atomic(folder/"receipt.json",dict(binding=plan["binding"],files={n:sha256_file(folder/n) for n in sorted(files)}))
    check_root_receipt(folder,entry,plan)
    return dict(status="ok",id=target["id"],trials=len(files)-1)


def collect(resume=False,limit=None):
    from scripts import run_sa_path_quality as q
    plan,out=verify()
    require(resume or not (out/"branches").exists(),"existing results require resume")
    done=[]
    with q._CollectionRunLock(out,plan["binding"],"history-candidate-bridge"):
        atomic(out/"run_status.json",dict(status="running",binding=plan["binding"]))
        try:
            for entry in plan["roots"][:limit]:
                if (out/"STOP_AFTER_ROOT").exists(): break
                job=dict(job_id=entry["target"]["id"],entry=entry,plan=plan,parent_pid=os.getpid())
                result=q._run_jobs(root_worker,[job],workers=1,phase=job["job_id"],output_root=out/"progress"/job["job_id"],
                    run_fingerprint=plan["binding"],timeout_seconds=600.,stop_on_failure=True)
                require(len(result)==1 and result[0]["status"]=="ok","root failed")
                done.extend(result)
                atomic(out/"run_status.json",dict(status="running",completed_this_call=len(done),last=job["job_id"],binding=plan["binding"]))
            verify()
            all_done=all((out/"branches"/e["target"]["id"]/"receipt.json").exists() for e in plan["roots"])
            atomic(out/"run_status.json",dict(status="completed" if all_done else "paused",binding=plan["binding"],roots_this_call=len(done)))
        except BaseException as exc:
            atomic(out/"run_status.json",dict(status="error",binding=plan["binding"],error=repr(exc)))
            raise
    return dict(roots_this_call=len(done),all_complete=all_done)


def branch_labels(row,root,entry,cfg):
    from scripts import run_sa_path_quality as q
    require(row["root_fingerprint"]==root["state_fingerprint"],"branch root identity")
    state=root["state"]
    counts=[]
    for offset,e in enumerate(row["events"]):
        require(e["decision"]==entry["target"]["decision"]+offset,"branch decision sequence")
        pp,uniform=randomization(entry["target"]["id"],row["trial"],e["decision"],cfg)
        require(e["action"]["random_seed"]==pp and e["uniform"]==uniform and e["temperature"]==q.temperature(e["decision"]),"branch random/temperature schedule")
        if offset==0:
            require(e["pool"][e["selected_index"]]["candidate_id"]==row["candidate_id"],"forced candidate identity")
        else:
            require(e["selected_index"]==frozen_index(e["pool"]),"non-frozen continuation choice")
        require(e["action"]["agents"]==e["pool"][e["selected_index"]]["agents"],"action membership changed")
        after=q.apply_state_delta(state,e["delta"])
        q.validate_transition(state,after,e["metrics"],e["action"]["agents"],"annealed",e["temperature"],uniform)
        state=after
        counts.append(state["num_of_colliding_pairs"])
    require(q.state_fingerprint(state)==row["final_fingerprint"] and state["num_of_colliding_pairs"]==row["final_conflicts"],"branch final identity")
    if row["stop"] not in ("horizon","feasible"): return None
    require(row["stop"]!="horizon" or len(counts)==cfg["horizon"],"short horizon")
    require(row["stop"]!="feasible" or counts[-1]==0,"false feasibility")
    return future_labels(counts,0,entry["previous_best"],cfg["horizon"],cfg["sustain"])


def stability(values):
    ids=sorted(values)
    require(len(ids)>=2 and all(len(values[c])==8 for c in ids),"paired trial coverage")
    halves=[{c:sum(values[c][start:start+4])/4 for c in ids} for start in (0,4)]
    best=[{c for c in ids if h[c]==max(h.values())} for h in halves]
    directions=[]
    for a,b in combinations(ids,2):
        d=[h[a]-h[b] for h in halves]
        directions.append(dict(pair=[a,b],half_differences=d,strict_both=all(x!=0 for x in d),
                               any_strict=any(x!=0 for x in d),agree=d[0]*d[1]>0))
    return dict(best_jaccard=len(best[0]&best[1])/len(best[0]|best[1]),pairs=directions,halves=halves)


def analyze_state(job):
    plan,entry=job
    cfg,out=locations()
    folder=out/"branches"/entry["target"]["id"]
    check_root_receipt(folder,entry,plan)
    sampling=read_json(ROOT/cfg["sampling_config"])
    root=read_json(ROOT/sampling["output"]/"states"/entry["target"]["id"]/"root.json")
    labels={}
    censored=[]
    for candidate in entry["selected"]:
        labels[candidate]=[]
        for trial in range(cfg["trials"]):
            r=read_json(folder/f"{candidate}-t{trial}.json")
            if r["status"]=="censored": label=None
            else:
                require((r["root_id"],r["candidate_id"],r["trial"])==(entry["target"]["id"],candidate,trial),"branch metadata")
                label=branch_labels(r,root,entry,cfg)
            labels[candidate].append(label)
            if label is None: censored.append(dict(candidate_id=candidate,trial=trial,stop=r["stop"]))
    result=dict(id=entry["target"]["id"],map_id=entry["target"]["map_id"],stratum=entry["target"]["stratum"],labels=labels,censored=censored)
    if not censored:
        result["stability"]={t:stability({c:[v[t] for v in vs] for c,vs in labels.items()}) for t in ("sustained_progress","completion")}
        selected={"frozen":entry["old_selected_id"],"ordered":entry["choices"]["ordered"],"temporal_bag":entry["choices"]["temporal_bag"]}
        result["policy_rates"]={p:{t:sum(v[t] for v in labels[c])/cfg["trials"] for t in ("sustained_progress","completion")} for p,c in selected.items()}
        result["policy_rates"]["uniform"]={t:sum(v[t] for vs in labels.values() for v in vs)/(len(labels)*cfg["trials"]) for t in ("sustained_progress","completion")}
    return result


def summarize(states,cfg):
    import numpy as np
    complete=[s for s in states if not s["censored"]]
    report=dict(states=states,complete_states=len(complete),censored_trials=sum(len(s["censored"]) for s in states),no_ttf=True,no_policy_promotion=True)
    require(complete,"all roots censored")
    summary={}
    for target in ("sustained_progress","completion"):
        pairs=[p|{"map_id":s["map_id"]} for s in complete for p in s["stability"][target]["pairs"]]
        strict=[p for p in pairs if p["strict_both"]]
        any_strict=sum(p["any_strict"] for p in pairs)
        jac=sum(s["stability"][target]["best_jaccard"] for s in complete)/len(complete)
        agreement=sum(p["agree"] for p in strict)/len(strict) if strict else None
        passed=(len(complete)==cfg["states"] and jac>=.5 and any_strict>=len(pairs)/2 and len(strict)>=8
                and len({p["map_id"] for p in strict})>=4 and agreement>=.75)
        rates={p:sum(s["policy_rates"][p][target] for s in complete)/len(complete) for p in ("frozen","ordered","temporal_bag","uniform")}
        contrasts={}
        for policy in ("ordered","temporal_bag"):
            values=np.asarray([s["policy_rates"][policy][target]-s["policy_rates"]["frozen"][target] for s in complete])
            draws=np.random.default_rng(cfg["seed"]).integers(0,len(complete),size=(cfg["bootstrap"],len(complete)))
            contrasts[policy]=dict(delta=float(values.mean()),ci95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist(),
                                    wins=int(sum(values>0)),losses=int(sum(values<0)),ties=int(sum(values==0)))
        summary[target]=dict(best_jaccard=jac,total_pairs=len(pairs),any_strict_pairs=any_strict,strict_both_pairs=len(strict),
                             strict_maps=len({p["map_id"] for p in strict}),direction_agreement=agreement,stability_passed=passed,
                             policy_rates=rates,vs_frozen=contrasts)
    return report|dict(summary=summary,decision="candidate_labels_stable_only" if summary["sustained_progress"]["stability_passed"] else "stop_unstable_or_insufficient_candidate_labels")


def analyze():
    from concurrent.futures import ProcessPoolExecutor
    plan,out=verify()
    with ProcessPoolExecutor(max_workers=min(8,plan["config"]["workers"])) as pool:
        states=list(pool.map(analyze_state,[(plan,e) for e in plan["roots"]]))
    report=summarize(states,plan["config"])|dict(binding=plan["binding"])
    verify()
    if (out/"report.json").exists(): require(read_json(out/"report.json")==report,"recomputed report differs")
    else: atomic(out/"report.json",report)
    return dict(decision=report["decision"],summary=report["summary"],sha256=sha256_file(out/"report.json"))


def main():
    for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[n]="1"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","collect","analyze","request-stop"))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    if args.phase=="request-stop":
        _,out=locations()
        atomic(out/"STOP_AFTER_ROOT",dict(requested=True))
        return
    result=prepare() if args.phase=="prepare" else analyze() if args.phase=="analyze" else collect(args.resume,args.limit)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__=="__main__": main()
