"""Read-only fit identity/metric audit and one fixed determinism replication."""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_sa_spatiotemporal_model_probe as probe
from experiments._common import read_json, sha256_file, write_json
from experiments.sa_paired_completion import MODEL_PARAMS, require
from experiments.sa_spatiotemporal_model import select
import numpy as np


def verify_fit_identity(result, profile, held, seed, states, epochs):
    require((result["method"],result["held"],result["seed"])==(profile,held,seed),"fit filename identity mismatch")
    require(result["train_ids"]==sorted(s["state_id"] for s in states if s["map_id"]!=held),"train map leakage")
    require(len(result["rows"])==len(states) and {r["state_id"] for r in result["rows"]}==
            {s["state_id"] for s in states},"fit state coverage")
    if profile!="gbdt":
        require(len(result["losses"])==epochs and all(math.isfinite(v) and v>=0 for v in result["losses"]),"epoch coverage")


def audit(repeat_one=False):
    plan,out = probe.verify()
    cfg = plan["config"]
    require(not (out/"train.lock").exists(),"training still active or interrupted")
    receipt = read_json(out/"complete.json")
    expected = {"report.json"}
    for m in plan["maps"]:
        for profile in ["gbdt"]+cfg["profiles"]:
            for seed in ([MODEL_PARAMS["random_state"]] if profile=="gbdt" else cfg["seeds"]):
                expected.add(probe.result_file(out,profile,m,seed).relative_to(out).as_posix())
    require(receipt["binding"]==plan["binding"] and set(receipt["files"])==expected,"output coverage")
    for name,digest in receipt["files"].items():
        require(sha256_file(out/name)==digest,"result SHA mismatch: "+name)
    data = read_json(out/"development_index.json")
    states = {s["state_id"]:s for s in data["states"]}
    rates = {sid:{c["candidate_id"]:sum(int(t["completed"]) for t in c["trials"])/8
                  for c in s["candidates"]} for sid,s in states.items()}
    observed,train_hit,parameter_counts,seconds = {},{}, {},0.
    checks = 0
    for name in sorted(expected-{"report.json"}):
        result = probe.sealed_fit(out/name,plan["binding"])
        profile,held,seed = result["method"],result["held"],result["seed"]
        require((out/name)==probe.result_file(out,profile,held,seed),"filename differs from recorded identity")
        verify_fit_identity(result,profile,held,seed,data["states"],cfg["epochs"])
        seconds += result["seconds"]
        if profile!="gbdt":
            parameter_counts.setdefault(profile,set()).add(result["parameters"])
        for row in result["rows"]:
            s = states[row["state_id"]]
            ids = [c["candidate_id"] for c in s["candidates"]]
            require(set(row["scores"])==set(ids) and row["held"]==(s["map_id"]==held),"row identity")
            selected = ids[select([row["scores"][i] for i in ids],ids,s["anchor_id"])]
            require(selected==row["selected"],"score/selection mismatch")
            rate = rates[s["state_id"]][selected]
            if row["held"]:
                observed.setdefault((profile,held),[]).append(rate)
            else:
                train_hit.setdefault(profile,[]).append(rate==max(rates[s["state_id"]].values()))
            checks += 1
    report = read_json(out/"report.json")
    for (profile,held),values in observed.items():
        require(abs(float(np.mean(values))-report["per_map"][held][profile])<1e-12,"reported metric mismatch")
    result = dict(binding=plan["binding"],verified_files=len(expected),prediction_rows_checked=checks,
                  train_empirical_oracle_hit={k:float(np.mean(v)) for k,v in train_hit.items()},
                  parameters={k:sorted(v) for k,v in parameter_counts.items()},
                  summed_fit_seconds=seconds,verifier_sha256=sha256_file(Path(__file__)),
                  original_report_sha256=sha256_file(out/"report.json"),repeat_one=None)
    if repeat_one:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        with np.load(out/"tokens.npz",allow_pickle=False) as arrays:
            raw = {k:torch.tensor(arrays[k],device=device) for k in arrays.files}
        held,seed,profile = plan["maps"][0],cfg["seeds"][0],"path_time"
        previous = probe.sealed_fit(probe.result_file(out,profile,held,seed),plan["binding"])
        again = probe.neural_fit(plan,data,held,profile,seed,raw,device)
        keys = ("rows","losses","scaler","train_ids","parameters")
        require(all(again[k]==previous[k] for k in keys),"fixed neural training did not reproduce exactly")
        result["repeat_one"] = dict(held=held,seed=seed,profile=profile,exact_scores_and_selection=True,
                                   training_losses_identical=True,seconds=again["seconds"])
    require(sha256_file(out/"report.json")==result["original_report_sha256"],"audit changed original report")
    filename = "verification_repeat.json" if repeat_one else "verification.json"
    path = out/filename
    require(not path.exists(),"preserve previous verification")
    write_json(path,result)
    return result


if __name__=="__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat-one",action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.repeat_one),indent=2))
