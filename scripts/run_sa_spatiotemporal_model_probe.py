"""Frozen offline representation comparison, with no solver or live policy use."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

from experiments._common import contained_file, json_fingerprint, read_json, sha256_file, write_json
from experiments.sa_paired_completion import MODEL_PARAMS, PairedCompletionModel, require, training_matrix
from experiments.sa_spatiotemporal_model import encode_model_input, fit_scaler, make_model, select
from experiments.sa_state_coverage import merge_data
from experiments.sa_unbalanced_coverage import state_weights
from scripts.audit_sa_spatiotemporal_input import verify as verify_input

CONFIG = "configs/sa_spatiotemporal_model_probe.json"
CODE = ["experiments/sa_spatiotemporal_model.py", "scripts/run_sa_spatiotemporal_model_probe.py",
        "tests/evaluation/test_sa_spatiotemporal_model.py", "docs/SA_SPATIOTEMPORAL_MODEL_PROTOCOL_ZH.md",
        "experiments/sa_paired_completion.py", "experiments/sa_state_coverage.py",
        "experiments/sa_unbalanced_coverage.py", "experiments/sa_spatiotemporal_input.py",
        "experiments/_common.py"]


def data_input(cfg):
    receipt = verify_input(cfg["input_audit"])
    require(receipt["binding"]==cfg["input_binding"],"input audit changed")
    settings = read_json(ROOT/cfg["input_audit"])
    datasets = [read_json(ROOT/s["directory"]/"training_index.json") for s in settings["sources"]]
    data = merge_data(*datasets,cfg["input_binding"])
    data["states"].sort(key=lambda s:s["state_id"])
    for s in data["states"]:
        s["candidates"].sort(key=lambda c:c["candidate_id"])
    require(len(data["states"])==cfg["expected_states"] and len({s["map_id"] for s in data["states"]})==cfg["expected_maps"],
            "cohort changed")
    require(data["trial_count"]==cfg["trials"]==8 and data["horizon"]==cfg["horizon"]==32,"label budget changed")
    require(all(t["completed"] is not None for s in data["states"] for c in s["candidates"] for t in c["trials"]),
            "censored labels cannot train")
    return data,settings


def encode_job(job):
    path,memberships = job
    with gzip.open(path,"rt") as stream:
        payload = json.load(stream)
    return encode_model_input(payload,memberships)


def tensor_arrays(encoded):
    n,c,k,t = max(x["base"].shape[0] for x in encoded),len(encoded[0]["indices"]),max(
        len(a) for x in encoded for a in x["indices"]),max(x["length"] for x in encoded)
    require(all(len(x["indices"])==c for x in encoded),"candidate count mismatch")
    result = dict(base=np.zeros((len(encoded),n,t,27),np.float32),
                  condition=np.zeros((len(encoded),c,k,t,32),np.float32),
                  indices=np.zeros((len(encoded),c,k),np.int64),
                  agent_mask=np.zeros((len(encoded),c,k),np.float32),
                  time_mask=np.zeros((len(encoded),t),np.float32))
    for i,x in enumerate(encoded):
        nn,tt,_ = x["base"].shape
        result["base"][i,:nn,:tt] = x["base"]
        result["time_mask"][i,:tt] = 1
        for j,idx in enumerate(x["indices"]):
            result["indices"][i,j,:len(idx)] = idx
            result["agent_mask"][i,j,:len(idx)] = 1
            result["condition"][i,j,:len(idx),:tt] = x["condition"][j]
    return result


def prepare():
    cfg = read_json(ROOT/CONFIG)
    require(cfg["new_solver_calls"]==0 and not cfg["runtime_integration_allowed"] and
            not cfg["model_promotion_allowed"] and cfg["role"]=="previously_viewed_development","boundary changed")
    data,settings = data_input(cfg)
    out = ROOT/cfg["output"]
    require(not out.exists() and out.resolve().is_relative_to((ROOT/"build").resolve()),"existing or unsafe output")
    inputs = {p:sha256_file(ROOT/p) for p in [CONFIG,cfg["input_audit"]]+CODE}
    audited = read_json(ROOT/settings["output"]/"complete.json")
    inputs.update(audited["inputs"])
    inputs.update({settings["output"]+"/"+p:h for p,h in audited["files"].items()})
    out.mkdir(parents=True)
    jobs = [(str(ROOT/settings["output"]/"states"/(s["state_id"]+".json.gz")),[c["agents"] for c in s["candidates"]])
            for s in data["states"]]
    write_json(out/"run_status.json",dict(status="preparing"))
    with ProcessPoolExecutor(max_workers=min(cfg["workers"],len(jobs))) as pool:
        arrays = tensor_arrays(list(pool.map(encode_job,jobs)))
    with (out/"tokens.npz").open("xb") as stream:
        np.savez_compressed(stream,**arrays)
    write_json(out/"development_index.json",data)
    inputs[(out/"tokens.npz").relative_to(ROOT).as_posix()] = sha256_file(out/"tokens.npz")
    inputs[(out/"development_index.json").relative_to(ROOT).as_posix()] = sha256_file(out/"development_index.json")
    for p,h in inputs.items():
        require(sha256_file(ROOT/p)==h,"input changed during preparation: "+p)
    plan = dict(config=cfg,inputs=inputs,source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
                maps=sorted({s["map_id"] for s in data["states"]}),data_fingerprint=json_fingerprint(data),
                tensor_shapes={k:list(v.shape) for k,v in arrays.items()},raw_tensor_bytes=sum(a.nbytes for a in arrays.values()))
    plan["binding"] = json_fingerprint(plan)
    write_json(out/"plan.json",plan)
    write_json(out/"run_status.json",dict(status="prepared",binding=plan["binding"]))
    return dict(binding=plan["binding"],states=len(data["states"]),maps=len(plan["maps"]),
                neural_fits=len(plan["maps"])*len(cfg["profiles"])*len(cfg["seeds"]),gbdt_fits=len(plan["maps"]),
                raw_tensor_bytes=plan["raw_tensor_bytes"],tensor_shapes=plan["tensor_shapes"])


def verify():
    cfg = read_json(ROOT/CONFIG)
    out = ROOT/cfg["output"]
    plan = read_json(out/"plan.json")
    require(plan["config"]==cfg and plan["binding"]==json_fingerprint({k:v for k,v in plan.items() if k!="binding"}),
            "plan changed")
    for p,h in plan["inputs"].items():
        require(sha256_file(contained_file(ROOT,p,field="probe input"))==h,"registered input changed: "+p)
    return plan,out


def result_file(out,method,held,seed):
    return out/"fits"/(method+"-"+held+"-"+str(seed)+".json")


def sealed_fit(path,binding):
    value = read_json(path)
    require(value["binding"]==binding and value["integrity"]==json_fingerprint({k:v for k,v in value.items() if k!="integrity"}),
            "fit identity mismatch")
    return value


def save_fit(path,value):
    require(not path.exists(),"fit already exists")
    write_json(path,dict(value,integrity=json_fingerprint(value)))


def gbdt_job(job):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    plan,data,held = job
    require(sklearn.__version__=="1.5.0","registered sklearn required")
    start = time.monotonic()
    matrix = training_matrix(data,{held})
    weights = state_weights(data["states"],held)
    estimator = HistGradientBoostingRegressor(loss="squared_error",**MODEL_PARAMS).fit(
        matrix["x"],matrix["y"],sample_weight=[w*weights[s] for w,s in zip(matrix["weights"],matrix["state_ids"])])
    model = PairedCompletionModel(data["feature_names"],estimator)
    rows = [dict(state_id=s["state_id"],selected=model.rank(s)["selected"],scores=model.rank(s)["scores"],
                 held=s["map_id"]==held) for s in data["states"]]
    return dict(binding=plan["binding"],method="gbdt",seed=MODEL_PARAMS["random_state"],held=held,
                train_ids=sorted(weights),rows=rows,seconds=time.monotonic()-start,sklearn=sklearn.__version__)


def gbdt():
    plan,out = verify()
    data = read_json(out/"development_index.json")
    jobs = [(plan,data,m) for m in plan["maps"] if not result_file(out,"gbdt",m,MODEL_PARAMS["random_state"]).exists()]
    with ProcessPoolExecutor(max_workers=min(plan["config"]["workers"],max(1,len(jobs)))) as pool:
        for result in pool.map(gbdt_job,jobs):
            save_fit(result_file(out,"gbdt",result["held"],result["seed"]),result)
            print("GBDT",result["held"],flush=True)
    for m in plan["maps"]:
        sealed_fit(result_file(out,"gbdt",m,MODEL_PARAMS["random_state"]),plan["binding"])
    return dict(gbdt_fits=len(plan["maps"]))


def neural_fit(plan,data,held,profile,seed,raw,device):
    import torch
    cfg = plan["config"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(profile,len(data["feature_names"]),cfg["hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    mean,std = fit_scaler(data["states"],held,data["feature_names"])
    flat_values = np.array([[[c["features"][n] for n in data["feature_names"]] for c in s["candidates"]]
                           for s in data["states"]],np.float32)
    flat = torch.as_tensor((flat_values-mean)/std,device=device)
    labels = torch.tensor([[np.mean([int(t["completed"]) for t in c["trials"]]) for c in s["candidates"]]
                           for s in data["states"]],dtype=torch.float32,device=device)
    train = [i for i,s in enumerate(data["states"]) if s["map_id"]!=held]
    weights = state_weights(data["states"],held)
    weight = torch.tensor([weights.get(s["state_id"],0) for s in data["states"]],device=device)
    pairs = torch.triu_indices(flat.shape[1],flat.shape[1],1,device=device)
    rng = np.random.default_rng(seed)
    start = time.monotonic()
    history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        order = rng.permutation(train)
        total = 0.
        for offset in range(0,len(order),cfg["batch_states"]):
            ids = torch.tensor(order[offset:offset+cfg["batch_states"]],device=device)
            batch = {} if profile=="flat" else {k:v[ids] for k,v in raw.items()}
            scores = model(batch,flat[ids])
            error = (scores[:,pairs[0]]-scores[:,pairs[1]])-(labels[ids][:,pairs[0]]-labels[ids][:,pairs[1]])
            loss = (error.square().mean(1)*weight[ids]).sum()/cfg["batch_states"]
            require(torch.isfinite(loss).item(),"nonfinite training loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip"])
            opt.step()
            total += float(loss.detach())*cfg["batch_states"]
        history.append(total/len(train))
        if (epoch+1)%20==0:
            print("EPOCH",held,profile,seed,epoch+1,"loss",round(history[-1],6),flush=True)
        require(time.monotonic()-start<=cfg["maximum_fit_seconds"],"fit safety budget exceeded; do not change epochs")
    model.eval()
    rows = []
    with torch.no_grad():
        for offset in range(0,len(data["states"]),cfg["batch_states"]):
            ids = torch.arange(offset,min(len(data["states"]),offset+cfg["batch_states"]),device=device)
            batch = {} if profile=="flat" else {k:v[ids] for k,v in raw.items()}
            values = model(batch,flat[ids]).cpu().numpy()
            for i,scores in zip(ids.tolist(),values):
                s = data["states"][i]
                candidates = [c["candidate_id"] for c in s["candidates"]]
                chosen = select(scores,candidates,s["anchor_id"])
                rows.append(dict(state_id=s["state_id"],selected=candidates[chosen],
                    scores=dict(zip(candidates,map(float,scores))),held=s["map_id"]==held))
    result = dict(binding=plan["binding"],method=profile,seed=seed,held=held,train_ids=sorted(weights),rows=rows,
                  losses=history,seconds=time.monotonic()-start,parameters=sum(p.numel() for p in model.parameters()),
                  scaler=dict(mean=mean.tolist(),std=std.tolist()),torch=torch.__version__,device=str(device))
    return result


def train():
    import torch
    plan,out = verify()
    cfg = plan["config"]
    require(torch.cuda.is_available(),"registered GPU unavailable; do not silently change runtime")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    data = read_json(out/"development_index.json")
    with np.load(out/"tokens.npz",allow_pickle=False) as arrays:
        raw = {k:torch.tensor(arrays[k],device=device) for k in arrays.files}
    start = time.monotonic()
    lock = out/"train.lock"
    with lock.open("x"):
        try:
            done = 0
            write_json(out/"run_status.json",dict(status="training",binding=plan["binding"],completed=done))
            for held in plan["maps"]:
                for profile in cfg["profiles"]:
                    for seed in cfg["seeds"]:
                        path = result_file(out,profile,held,seed)
                        if path.exists():
                            sealed_fit(path,plan["binding"])
                        else:
                            require(time.monotonic()-start<cfg["maximum_training_seconds"],"training safety budget exceeded")
                            result = neural_fit(plan,data,held,profile,seed,raw,device)
                            save_fit(path,result)
                        done += 1
                        print("FIT",done,"/",len(plan["maps"])*len(cfg["profiles"])*len(cfg["seeds"]),held,profile,seed,flush=True)
                        write_json(out/"run_status.json",dict(status="training",binding=plan["binding"],completed=done))
            verify()
            write_json(out/"run_status.json",dict(status="trained",binding=plan["binding"],completed=done))
        except BaseException as exc:
            write_json(out/"run_status.json",dict(status="error",binding=plan["binding"],completed=done,error=repr(exc)))
            raise
    lock.unlink()
    return dict(neural_fits=done)


def analyze():
    plan,out = verify()
    cfg = plan["config"]
    data = read_json(out/"development_index.json")
    states = {s["state_id"]:s for s in data["states"]}
    values,train_values,details = {},{},[]
    profiles = ["gbdt"]+cfg["profiles"]
    for held in plan["maps"]:
        for profile in profiles:
            seeds = [MODEL_PARAMS["random_state"]] if profile=="gbdt" else cfg["seeds"]
            for seed in seeds:
                result = sealed_fit(result_file(out,profile,held,seed),plan["binding"])
                expected = sorted(sid for sid,s in states.items() if s["map_id"]!=held)
                require(result["train_ids"]==expected and result["held"]==held,"map leakage")
                require(len(result["rows"])==len(states) and {r["state_id"] for r in result["rows"]}==set(states),"prediction coverage")
                for r in result["rows"]:
                    s = states[r["state_id"]]
                    require(r["held"]==(s["map_id"]==held),"held flag mismatch")
                    rates = {c["candidate_id"]:np.mean([int(t["completed"]) for t in c["trials"]]) for c in s["candidates"]}
                    require(r["selected"] in rates and set(r["scores"])==set(rates),"prediction candidate mismatch")
                    row = dict(state_id=s["state_id"],map_id=s["map_id"],method=profile,seed=seed,fold=held,
                               held=r["held"],rate=float(rates[r["selected"]]),selected=r["selected"],
                               regret=float(max(rates.values())-rates[r["selected"]]),
                               oracle_hit=bool(rates[r["selected"]]==max(rates.values())),
                               size=len(next(c["agents"] for c in s["candidates"] if c["candidate_id"]==r["selected"])))
                    details.append(row)
                    target = values if r["held"] else train_values
                    target.setdefault((profile,seed,held),[]).append(row["rate"])
    per_map = {m:{} for m in plan["maps"]}
    for m in plan["maps"]:
        for profile in profiles:
            per_map[m][profile] = float(np.mean([np.mean(v) for (p,seed,held),v in values.items() if p==profile and held==m]))
        subset = [s for s in data["states"] if s["map_id"]==m]
        for name in ("frozen","uniform","oracle"):
            v = []
            for s in subset:
                rates = {c["candidate_id"]:np.mean([int(t["completed"]) for t in c["trials"]]) for c in s["candidates"]}
                v.append(rates[s["anchor_id"]] if name=="frozen" else np.mean(list(rates.values())) if name=="uniform" else max(rates.values()))
            per_map[m][name] = float(np.mean(v))
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    draws = rng.integers(0,len(per_map),(cfg["bootstrap"],len(per_map)))
    contrasts = {}
    for baseline in ("gbdt","flat","time_bag","frozen","uniform"):
        delta = np.array([v["path_time"]-v[baseline] for v in per_map.values()])
        seed_deltas = []
        for seed in cfg["seeds"]:
            seed_deltas.append(float(np.mean([np.mean(values[("path_time",seed,m)])-(
                np.mean(values[(baseline,seed,m)]) if baseline in cfg["profiles"] else per_map[m][baseline]) for m in plan["maps"]])))
        contrasts[baseline] = dict(delta=float(delta.mean()),ci95=np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
                                   map_wins=int(sum(delta>0)),seed_deltas=seed_deltas)
    checks = dict(gbdt_gain=contrasts["gbdt"]["delta"]>=cfg["minimum_gbdt_gain"],
                  gbdt_ci=contrasts["gbdt"]["ci95"][0]>=0,
                  gbdt_map_wins=contrasts["gbdt"]["map_wins"]>=cfg["minimum_map_wins"],
                  gbdt_seed_wins=sum(d>0 for d in contrasts["gbdt"]["seed_deltas"])>=cfg["minimum_seed_wins"],
                  beats_flat=contrasts["flat"]["delta"]>0 and contrasts["flat"]["ci95"][0]>=0,
                  beats_bag=contrasts["time_bag"]["delta"]>0,
                  beats_frozen=contrasts["frozen"]["delta"]>0,
                  beats_uniform=contrasts["uniform"]["delta"]>0)
    diagnostics = {}
    for profile in profiles:
        selected = [r for r in details if r["method"]==profile and r["held"]]
        diagnostics[profile] = dict(
            mean_regret=float(np.mean([np.mean([r["regret"] for r in selected if r["map_id"]==m]) for m in plan["maps"]])),
            empirical_oracle_hit=float(np.mean([np.mean([r["oracle_hit"] for r in selected if r["map_id"]==m]) for m in plan["maps"]])),
            size_counts={str(k):sum(r["size"]==k for r in selected) for k in sorted({r["size"] for r in selected})})
    report = dict(binding=plan["binding"],per_map=per_map,diagnostics=diagnostics,
        map_equal={p:float(np.mean([v[p] for v in per_map.values()])) for p in per_map[plan["maps"][0]]},
        train_fit={p:float(np.mean([np.mean(v) for (method,seed,held),v in train_values.items() if method==p])) for p in profiles},
        contrasts=contrasts,checks=checks,rows=details,
        decision="development_representation_signal" if all(checks.values()) else "no_stable_incremental_representation_signal",
        model_promotion_allowed=False,formal_ttf_allowed=False,independent_generalization_established=False)
    write_json(out/"report.json",report)
    write_json(out/"complete.json",dict(binding=plan["binding"],files={p.relative_to(out).as_posix():sha256_file(p)
        for p in [out/"report.json"]+sorted((out/"fits").glob("*.json"))}))
    write_json(out/"run_status.json",dict(status="completed",binding=plan["binding"]))
    return {k:v for k,v in report.items() if k not in ("rows","per_map")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("prepare","gbdt","train","analyze","verify"))
    args = parser.parse_args()
    if args.stage=="verify":
        plan,out = verify()
        result = dict(binding=plan["binding"],status="inputs_verified")
    else:
        result = globals()[args.stage]()
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
