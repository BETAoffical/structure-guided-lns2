"""Outcome-blind roots and map-held-out candidate contrast before new rollouts."""
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json,json_fingerprint,sha256_file
from experiments.sa_history_information import OrderedHistory,profile_features,episode_weights
from experiments.sa_history_selector import History
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.repair_collection import state_fingerprint
from scripts.audit_sa_history_information import verify as verify_information,receipt_result,atomic,require
from scripts.audit_sa_history_matched_controls import matched_features
from scripts.train_sa_history_selector import verify as verify_sampling

CONFIG=ROOT/"configs/sa_history_candidate_bridge.json"
MODELS={"dynamic_progress":("dynamic","sustained_progress"),"dynamic_completion":("dynamic","completion"),
        "ordered":("ordered","sustained_progress"),"temporal_bag":("temporal_bag","completion")}


def select_roots(targets,cfg):
    maps=sorted({t["map_id"] for t in targets})
    history_maps=[m for m in maps if any(t["map_id"]==m and t["stratum"].startswith("history_") for t in targets)]
    history_maps=sorted(history_maps,key=lambda m:json_fingerprint([cfg["seed"],m]))[:cfg["history_states"]]
    chosen=[]
    for m in maps:
        pool=[t for t in targets if t["map_id"]==m and (t["stratum"].startswith("history_") if m in history_maps else t["stratum"]=="progress")]
        require(bool(pool),"missing frozen stratum")
        chosen.append(min(pool,key=lambda t:json_fingerprint([cfg["seed"],t["id"]])))
    require(len(chosen)==cfg["states"] and len(history_maps)==cfg["history_states"],"root quota")
    return chosen


def features(profile,row):
    return matched_features(row,profile) if profile=="temporal_bag" else profile_features(row,profile)


def root_rows(job):
    cfg,source_plan,target=job
    source=ROOT/source_plan["config"]["source"]/"episodes"/target["item"]["job_id"]
    folder=ROOT/source_plan["config"]["output"]/"states"/target["id"]
    root=read_json(folder/"root.json")
    receipt=read_json(folder/"receipt.json")
    require(sha256_file(folder/"root.json")==receipt["files"]["root.json"],"root bytes changed")
    state=read_json(source/"initial.json")["payload"]["observation"]
    history,ordered=History(state,32),OrderedHistory()
    for line in (source/"first_phase/trace.jsonl").read_text(encoding="utf8").splitlines():
        event=json.loads(line)
        if event["decision"]==target["decision"]:
            break
        after=apply_state_delta(state,event["delta"])
        ordered.observe(state,event,after,history.best)
        history.observe(state,event,after)
        state=after
    require(history.decision==target["decision"],"prefix missing")
    require(state_fingerprint(state)==root["state_fingerprint"],"root fingerprint")
    rows=[]
    for c,base in zip(root["candidates"],root["feature_rows"]):
        hf=history.features(state,c,root["control_event"]["temperature"])
        require(all(base[k]==v for k,v in hf.items()),"history mismatch")
        rows.append(dict(id=target["id"]+"/"+c["candidate_id"],state_id=target["id"],map_id=target["map_id"],
                         episode=target["item"]["job_id"],base=base,records=ordered.records(state,c),
                         candidate_id=c["candidate_id"],agents=c["agents"]))
    return dict(target=target,rows=rows,old_selected_id=root["old_selected_id"],previous_best=history.best,
                state_fingerprint=root["state_fingerprint"],root_sha=sha256_file(folder/"root.json"))


def predict(job):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    train,root,model_cfg=job
    train=[r for r in train if r["map_id"]!=root["target"]["map_id"]]
    require(all(r["episode"]!=root["target"]["item"]["job_id"] for r in train),"episode leakage")
    result={}
    for name,(profile,label) in MODELS.items():
        a=[features(profile,r) for r in train]
        b=[features(profile,r) for r in root["rows"]]
        names=sorted({k for v in a for k in v})
        require(all(set(v)<=set(names) for v in b),"unexpected feature schema")
        def matrix(vs): return np.array([[v.get(k,0.) for k in names] for v in vs],dtype=float)
        y=[r["labels"][label] for r in train]
        require(len(set(y))==2,"single class training fold")
        model=HistGradientBoostingClassifier(**model_cfg).fit(matrix(a),y,sample_weight=episode_weights(train))
        result[name]=model.predict_proba(matrix(b))[:,1].tolist()
    return root | {"predictions":result,"train_episodes":len({r["episode"] for r in train}),
                   "train_maps":sorted({r["map_id"] for r in train})}


def choice(candidates,values):
    require(len(candidates)==len(values)>0,"candidate prediction shape")
    return candidates[min(range(len(values)),key=lambda i:(-values[i],candidates[i]))]


def describe(root,cfg):
    ids=[r["candidate_id"] for r in root["rows"]]
    chosen={k:choice(ids,v) for k,v in root["predictions"].items()}
    selected=[]
    for c in [root["old_selected_id"],chosen["ordered"],chosen["temporal_bag"]]+sorted(ids,key=lambda c:json_fingerprint([cfg["seed"],root["target"]["id"],c])):
        if c not in selected and len(selected)<cfg["max_candidates"]:
            selected.append(c)
    return root | {"choices":chosen,"selected":selected,
                   "spreads":{k:max(v)-min(v) for k,v in root["predictions"].items()}}


def admission(roots):
    different_frozen=sum(any(r["choices"][p]!=r["old_selected_id"] for p in ("ordered","temporal_bag")) for r in roots)
    different_dynamic=sum(r["choices"]["ordered"]!=r["choices"]["dynamic_progress"] or r["choices"]["temporal_bag"]!=r["choices"]["dynamic_completion"] for r in roots)
    varying=sum(max(r["spreads"][p] for p in ("ordered","temporal_bag"))>1e-12 for r in roots)
    return dict(maps=len({r["target"]["map_id"] for r in roots}),different_frozen=different_frozen,
                different_dynamic=different_dynamic,varying=varying,
                passed=len(roots)==8 and different_frozen>=4 and different_dynamic>=3 and varying>=4)


def main():
    for n in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ[n]="1"
    cfg=read_json(CONFIG)
    out=ROOT/cfg["output"]
    require(not (out/"preflight.json").exists(),"preflight exists; preserve frozen predictions")
    info,source=verify_information()
    sampling=verify_sampling(ROOT/cfg["sampling_config"])
    require(sha256_file(source/"report.json")==cfg["information_report_sha"],"information report changed")
    require(sha256_file(ROOT/"build/sa-history-information-matched-v1/report.json")==cfg["matched_report_sha"],"matched report changed")
    inputs=dict(info["inputs"])
    for p in [CONFIG,Path(__file__),ROOT/"docs/SA_HISTORY_CANDIDATE_BRIDGE_ZH.md",source/"index.json",source/"report.json",
              ROOT/"build/sa-history-information-matched-v1/report.json",ROOT/sampling["config"]["output"]/"plan.json"]:
        inputs[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    targets=select_roots(sampling["targets"],cfg)
    train=[r for r in receipt_result(source/"index.json",info["binding"])["rows"] if r["labels"] is not None]
    plan=dict(config=cfg,targets=targets,inputs=inputs,information_binding=info["binding"],sampling_binding=sampling["binding"],
              commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    plan["binding"]=json_fingerprint(plan)
    atomic(out/"preflight_plan.json",plan)
    with ProcessPoolExecutor(max_workers=min(cfg["workers"],8)) as pool:
        roots=list(pool.map(root_rows,[(cfg,sampling,t) for t in targets]))
        results=list(pool.map(predict,[(train,r,info["config"]["model"]) for r in roots]))
    results=[describe(r,cfg) for r in results]
    verify_information()
    verify_sampling(ROOT/cfg["sampling_config"])
    report=dict(binding=plan["binding"],roots=results,admission=admission(results),no_solver_calls=True,no_ttf=True)
    atomic(out/"preflight.json",report)
    print(json.dumps(dict(admission=report["admission"],root_decisions=[r["target"]["decision"] for r in results],
                          max_trials=sum(len(r["selected"]) for r in results)*cfg["trials"],sha256=sha256_file(out/"preflight.json"))))


if __name__=="__main__": main()
