"""Recompute saved model decisions only; no solver calls or fitting."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, json_fingerprint
from experiments.sa_paired_completion import require, validate_dataset
from experiments.sa_pairwise_consistency import inspect, evaluate, counts, label_summary
from scripts import run_sa_continuation_value as continuation
from scripts import run_sa_paired_closed_loop as loop
from scripts.collect_sa_history_candidate_bridge import frozen_index

CONFIG = ROOT/"configs/sa_pairwise_consistency.json"
MODEL = None


def prepare():
    cfg = read_json(CONFIG)
    out = ROOT/cfg["output"]
    require(not out.exists(), "existing output; do not overwrite")
    source, source_out = continuation.verify()
    require(source_out == ROOT/cfg["source"], "source path changed")
    require(sha256_file(source_out/"report.json") == cfg["source_report_sha256"], "source report changed")
    report = read_json(source_out/"report.json")
    require(report["binding"] == source["binding"], "source report binding")
    files = dict(source["files"])
    for name,digest in report["files"].items():
        path = source_out/name
        require(sha256_file(path) == digest, "source output changed: "+name)
        files[path.relative_to(ROOT).as_posix()] = digest
    model_out = ROOT/source["config"]["model_source"]
    loop.model_files(model_out,source["model_source"])
    index_path = ROOT/source["config"]["source"]/"training_index.json"
    data = validate_dataset(read_json(index_path))
    meta = read_json(model_out/"model/metadata.json")
    require(len(data["states"]) == cfg["expected_labeled_states"], "state coverage")
    require(data["feature_names"] == meta["feature_names"], "model schema")
    require(not set(s["state_id"] for s in data["states"]) & set(meta["train_state_ids"]), "training state overlap")
    train = read_json(ROOT/source["model_source"]["config"]["training_index"])
    require(not set(s["episode"] for s in data["states"]) & set(s["episode"] for s in train["states"]), "training episode overlap")
    additions = [CONFIG,Path(__file__),ROOT/"experiments/sa_pairwise_consistency.py",
        ROOT/"tests/evaluation/test_sa_pairwise_consistency.py",ROOT/"docs/SA_PAIRWISE_CONSISTENCY_PROTOCOL_ZH.md",
        source_out/"plan.json",source_out/"report.json",index_path]
    for p in additions:
        name = p.relative_to(ROOT).as_posix()
        digest = sha256_file(p)
        require(name not in files or files[name] == digest,"conflicting identity")
        files[name] = digest
    plan = dict(config=cfg,files=files,source_binding=source["binding"],
        source_index=index_path.relative_to(ROOT).as_posix(),model=str(model_out.relative_to(ROOT).as_posix())+"/model/model.pkl",
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        new_solver_calls=0,training=False,production_changed=False)
    plan["binding"] = json_fingerprint(plan)
    loop.once(out/"plan.json",plan)
    return dict(binding=plan["binding"],registered_files=len(files),new_solver_calls=0)


def verify():
    cfg = read_json(CONFIG)
    out = ROOT/cfg["output"]
    plan = read_json(out/"plan.json")
    require(plan["config"] == cfg and plan["binding"] == json_fingerprint({k:v for k,v in plan.items() if k != "binding"}), "plan binding")
    for name,digest in plan["files"].items():
        require(sha256_file(ROOT/name) == digest,"registered file changed: "+name)
    return plan,out


def initialize(model_path):
    global MODEL
    import sklearn
    require(sklearn.__version__ == "1.5.0", "frozen sklearn version required")
    # This local pickle is hash-verified before worker creation.
    with (ROOT/model_path).open("rb") as stream:
        MODEL = pickle.load(stream)


def audit_root(job):
    state = job["state"]
    row = dict(inspect(MODEL,state,job["expected"]),anchor_id=state["anchor_id"],
        state_id=state["state_id"],map_id=state["map_id"],decision=state["decision"])
    values = {c["candidate_id"]:[r["completed"] for r in sorted(c["trials"],key=lambda r:r["trial"])]
              for c in state["candidates"]}
    row["labels"] = dict(frozen=evaluate(row,values))
    if job["new_values"] is not None:
        require(values == job["old_values"],"source index vs formal report labels mismatch")
        row["labels"]["paired"] = evaluate(row,job["new_values"])
    return dict(kind="root",id=state["state_id"],rows=[row])


def audit_trace(job):
    branch = loop.check_seal(read_json(ROOT/job["path"]))
    require(branch["binding"] == job["binding"] and branch["status"] == "ok" and branch["arm"] == "paired", "branch identity")
    rows = []
    for offset,event in enumerate(branch["events"]):
        if offset == 0:
            require(event["ranking"] is None,"first action must be forced")
            continue
        pool = event["pool"]
        anchor = pool[frozen_index(pool)]["candidate_id"]
        candidates = [next(c for c in pool if c["candidate_id"] == cid) for cid in event["subset"]]
        state = dict(anchor_id=anchor,agent_ids=job["agent_ids"],candidates=[
            dict(candidate_id=c["candidate_id"],agents=c["agents"],features=f)
            for c,f in zip(candidates,event["features"],strict=True)])
        result = inspect(MODEL,state,event["ranking"])
        require(result["selected"] == pool[event["selected_index"]]["candidate_id"],"logged action mismatch")
        rows.append(dict(result,root_id=branch["root_id"],trial=branch["trial"],root_candidate=branch["candidate_id"],
            map_id=job["map_id"],decision=event["decision"],before=event["before"],anchor_id=anchor))
    return dict(kind="trace",id=job["id"],rows=rows)


def worker(job):
    return audit_root(job) if job["kind"] == "root" else audit_trace(job)


def jobs(plan):
    source_out = ROOT/plan["config"]["source"]
    report = read_json(source_out/"report.json")
    source = read_json(source_out/"plan.json")
    data = validate_dataset(read_json(ROOT/plan["source_index"]))
    found = {r["state_id"]:r for r in report["states"]}
    result = []
    for state in sorted(data["states"],key=lambda s:s["state_id"]):
        prior = found.get(state["state_id"])
        result.append(dict(kind="root",state=state,expected=prior["prediction"] if prior else None,
            old_values=prior["values"]["frozen"] if prior else None,new_values=prior["values"]["paired"] if prior else None))
    for entry in source["entries"]:
        target = entry["target"]
        root = read_json(ROOT/source["config"]["source"]/"roots"/target["id"]/"root.json")
        for cid in sorted(target["selected"]):
            for trial in range(8):
                identity = target["id"]+"-"+cid+"-t"+str(trial)
                path = source_out/"roots"/target["id"]/f"{cid}-t{trial}.json"
                result.append(dict(kind="trace",id=identity,path=path.relative_to(ROOT).as_posix(),
                    binding=source["binding"],map_id=target["map_id"],agent_ids=[a["id"] for a in root["state"]["agents"]]))
    return result


def run():
    plan,out = verify()
    require(not (out/"report.json").exists(),"report exists; use verify, not rerun")
    schedule = jobs(plan)
    lock = out/"RUNNING.lock"
    with lock.open("x",encoding="ascii") as stream:
        stream.write(str(os.getpid()))
    try:
        all_rows = {"root":[],"trace":[]}
        with ProcessPoolExecutor(max_workers=plan["config"]["workers"],initializer=initialize,initargs=(plan["model"],)) as pool:
            for i,result in enumerate(pool.map(worker,schedule),1):
                loop.once(out/result["kind"]/(result["id"]+".json"),dict(binding=plan["binding"],**result))
                all_rows[result["kind"]].extend(result["rows"])
                if i % 20 == 0 or i == len(schedule):
                    print(json.dumps(dict(completed=i,total=len(schedule))),flush=True)
        require(len(all_rows["trace"]) == plan["config"]["expected_continuation_decisions"],"trace coverage")
        roots = sorted(all_rows["root"],key=lambda r:r["state_id"])
        paired = [r for r in roots if "paired" in r["labels"]]
        cfg = plan["config"]
        report = dict(binding=plan["binding"],root_counts=counts(roots),trace_counts=counts(all_rows["trace"]),
            trace_by_map={m:counts([r for r in all_rows["trace"] if r["map_id"]==m]) for m in sorted({r["map_id"] for r in all_rows["trace"]})},
            old_continuation=label_summary(roots,"frozen",cfg["seed"],cfg["bootstrap"]),
            new_continuation=label_summary(paired,"paired",cfg["seed"],cfg["bootstrap"]),
            roots=roots,exploratory=True,promotion=False,new_solver_calls=0,training=False,production_changed=False)
        verify()
        report["files"] = {p.relative_to(out).as_posix():sha256_file(p) for kind in ("root","trace") for p in sorted((out/kind).glob("*.json"))}
        loop.once(out/"report.json",report)
        return {k:v for k,v in report.items() if k not in {"roots","files","trace_by_map"}}
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","analyze","verify"))
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare()
    elif args.phase == "analyze":
        result = run()
    else:
        plan,out = verify()
        count = 0
        if (out/"report.json").exists():
            report = read_json(out/"report.json")
            require(report["binding"] == plan["binding"],"report binding")
            for name,digest in report["files"].items():
                require(sha256_file(out/name) == digest,"result file changed")
                count += 1
        result = dict(binding=plan["binding"],inputs=len(plan["files"]),outputs=count)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
