"""Fixed 16-fit pilot using existing H32 data; no solver or production changes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json,sha256_file,json_fingerprint
from experiments.sa_paired_completion import MODEL_PARAMS,require,validate_dataset
from experiments.sa_low_complexity_ranker import METHODS,fit,predict,completion_rates,summarize
from scripts import run_sa_spatiotemporal_model_probe as previous
from scripts.run_sa_paired_closed_loop import once

CONFIG = ROOT/"configs/sa_low_complexity_ranker.json"


def prepare():
    cfg = read_json(CONFIG)
    out = ROOT/cfg["output"]
    require(not out.exists(),"existing output; no overwrite")
    source,folder = previous.verify()
    require(folder==ROOT/cfg["source"],"source directory changed")
    data = validate_dataset(read_json(folder/"development_index.json"))
    require(len(data["states"])==cfg["states"] and len({s["map_id"] for s in data["states"]})==cfg["maps"],"cohort changed")
    require(data["horizon"]==32 and data["trial_count"]==8,"target/trial definition changed")
    require(all(t["completed"] is not None for s in data["states"] for c in s["candidates"] for t in c["trials"]),"censored labels")
    files = dict(source["inputs"])
    additions = [CONFIG,Path(__file__),ROOT/"experiments/sa_low_complexity_ranker.py",
        ROOT/"tests/evaluation/test_sa_low_complexity_ranker.py",ROOT/"docs/SA_LOW_COMPLEXITY_RANKER_PROTOCOL_ZH.md",
        folder/"plan.json",ROOT/"scripts/run_sa_spatiotemporal_model_probe.py",ROOT/"scripts/run_sa_paired_closed_loop.py"]
    baseline = {}
    for held in source["maps"]:
        path = previous.result_file(folder,"gbdt",held,MODEL_PARAMS["random_state"])
        record = previous.sealed_fit(path,source["binding"])
        require(record["train_ids"]==sorted(s["state_id"] for s in data["states"] if s["map_id"]!=held),"baseline map leakage")
        baseline[held] = path.relative_to(ROOT).as_posix()
        additions.append(path)
    for path in additions:
        name,digest = path.relative_to(ROOT).as_posix(),sha256_file(path)
        require(name not in files or files[name]==digest,"source identity conflict")
        files[name] = digest
    plan = dict(config=cfg,files=files,baseline=baseline,maps=source["maps"],source_binding=source["binding"],
        source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    plan["binding"] = json_fingerprint(plan)
    once(out/"plan.json",plan)
    return dict(binding=plan["binding"],fits=2*len(plan["maps"]),workers=min(cfg["workers"],2*len(plan["maps"])),new_solver_calls=0)


def verify():
    cfg = read_json(CONFIG)
    out = ROOT/cfg["output"]
    plan = read_json(out/"plan.json")
    require(plan["config"]==cfg and plan["binding"]==json_fingerprint({k:v for k,v in plan.items() if k!="binding"}),"plan changed")
    for name,digest in plan["files"].items():
        require(sha256_file(ROOT/name)==digest,"input changed: "+name)
    return plan,out


def worker(job):
    data,held,method,plan = job
    result = fit(data,held,method,plan["config"])
    return dict(binding=plan["binding"],held=held,method=method,**result)


def result_path(out,method,held):
    return out/"fits"/(method+"-"+held+".json")


def checked_result(out,method,held,plan):
    value = read_json(result_path(out,method,held))
    require(value["binding"]==plan["binding"] and value["integrity"]==json_fingerprint({k:v for k,v in value.items() if k!="integrity"}),"fit changed")
    require(value["held"]==held and value["method"]==method and value["model"]["held"]==held
            and value["model"]["method"]==method and value["model"]["config"]==plan["config"],"fit identity")
    return value


def train(resume=False):
    plan,out = verify()
    require(resume or not (out/"fits").exists(),"existing fits require --resume")
    data = read_json(ROOT/plan["config"]["source"]/"development_index.json")
    jobs = [(data,m,method,plan) for m in plan["maps"] for method in METHODS if not result_path(out,method,m).exists()]
    lock = out/"TRAINING.lock"
    with lock.open("x",encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        if jobs:
            with ProcessPoolExecutor(max_workers=min(plan["config"]["workers"],len(jobs))) as pool:
                for i,result in enumerate(pool.map(worker,jobs),1):
                    once(result_path(out,result["method"],result["held"]),dict(result,integrity=json_fingerprint(result)))
                    print(json.dumps(dict(completed=i,total=len(jobs),held=result["held"],method=result["method"])),flush=True)
        for held in plan["maps"]:
            for method in METHODS:
                checked_result(out,method,held,plan)
        verify()
        return dict(completed=2*len(plan["maps"]),new_fits=len(jobs))
    finally:
        lock.unlink()


def analyze():
    plan,out = verify()
    data = read_json(ROOT/plan["config"]["source"]/"development_index.json")
    states = {s["state_id"]:s for s in data["states"]}
    predictions = {sid:{} for sid in states}
    training = {m:[] for m in ["gbdt",*METHODS]}
    result_files = {}
    for held in plan["maps"]:
        baseline = previous.sealed_fit(ROOT/plan["baseline"][held],plan["source_binding"])
        for method in ["gbdt",*METHODS]:
            result = baseline if method=="gbdt" else checked_result(out,method,held,plan)
            train_ids = result["train_ids"] if method=="gbdt" else result["model"]["train_ids"]
            require(train_ids==sorted(sid for sid,s in states.items() if s["map_id"]!=held),"train/held-out leakage")
            require(len(result["rows"])==len(states) and {r["state_id"] for r in result["rows"]}==set(states),"row coverage")
            for row in result["rows"]:
                state = states[row["state_id"]]
                require(row["held"]==(state["map_id"]==held),"held flag")
                if method!="gbdt":
                    require(predict(result["model"],state)=={k:row[k] for k in ("selected","scores")},"model reloaded prediction changed")
                if row["held"]:
                    require(method not in predictions[row["state_id"]],"duplicate held prediction")
                    predictions[row["state_id"]][method] = row["selected"]
                else:
                    rates = completion_rates(state)
                    training[method].append(dict(rate=rates[row["selected"]],oracle_hit=rates[row["selected"]]==max(rates.values())))
            if method!="gbdt":
                path = result_path(out,method,held)
                result_files[path.relative_to(out).as_posix()] = sha256_file(path)
    report = summarize(data,predictions,plan["config"])
    require(abs(report["means"]["gbdt"]-plan["config"]["expected_gbdt_rate"])<1e-12,"historical baseline not reproduced")
    report.update(binding=plan["binding"],files=result_files,new_solver_calls=0,
        train_fit={k:dict(rows=len(v),completion=sum(r["rate"] for r in v)/len(v),
            oracle_hit=sum(r["oracle_hit"] for r in v)/len(v)) for k,v in training.items()})
    report["integrity"] = json_fingerprint(report)
    verify()
    once(out/"report.json",report)
    return {k:v for k,v in report.items() if k not in {"files","rows"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","train","analyze","verify"))
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if args.phase=="prepare":
        result = prepare()
    elif args.phase=="train":
        result = train(args.resume)
    elif args.phase=="analyze":
        result = analyze()
    else:
        plan,out = verify()
        count = 0
        if (out/"report.json").exists():
            report = read_json(out/"report.json")
            require(report["binding"]==plan["binding"],"report binding")
            require(report["integrity"]==json_fingerprint({k:v for k,v in report.items() if k!="integrity"}),"report changed")
            for name,digest in report["files"].items():
                require(sha256_file(out/name)==digest,"output changed")
                count += 1
        result = dict(binding=plan["binding"],inputs=len(plan["files"]),outputs=count)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
