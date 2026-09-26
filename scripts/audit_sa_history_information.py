"""Frozen observational audit; no solver, selector export, or timing runs."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import json_fingerprint, read_json, sha256_file
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.repair_collection import state_fingerprint
from experiments.sa_history_selector import History
from experiments.sa_history_information import OrderedHistory, future_labels, profile_features, episode_weights, metric

CONFIG = ROOT / "configs/sa_history_information_audit.json"


from lns2_selector.runtime.contracts import require


def source_finished(receipt):
    return receipt["status"] in ("completed", "no_feasible_solution")


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    require(not tmp.exists(), "interrupted atomic output: " + str(tmp))
    tmp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf8")
    os.replace(tmp, path)


def progress(out, phase, **kwargs):
    row = dict(phase=phase, **kwargs)
    with (out / "progress.jsonl").open("a", encoding="utf8") as f:
        f.write(json.dumps(row, sort_keys=True)+"\n")
    print(json.dumps(row, sort_keys=True), flush=True)


def prepare():
    from experiments.online_feature_engine import _native_batch_function
    cfg = read_json(CONFIG)
    out, source = ROOT/cfg["output"], ROOT/cfg["source"]
    require(not out.exists(), "output exists")
    ref = ROOT/cfg["reference_plan"]
    require(sha256_file(ref) == cfg["reference_plan_sha"], "reference plan changed")
    reg, manifest = (read_json(source/n) for n in ("registration.json", "manifest.json"))
    require(reg["fingerprint"] == json_fingerprint({k:v for k,v in reg.items() if k != "fingerprint"}), "registration fingerprint")
    require(manifest["binding"] == reg["fingerprint"], "source manifest binding")
    jobs = []
    inputs = {}
    for item in reg["schedule"]:
        if item["controller"] != "dual16_sa":
            continue
        receipt = manifest["jobs"][item["job_id"]]
        require(source_finished(receipt), "source episode not complete")
        files = {}
        for name in ("initial.json", "first_phase/trace.jsonl", "result.json"):
            path = source/"episodes"/item["job_id"]/name
            require(sha256_file(path) == receipt["files"][name], "source changed")
            relative = path.relative_to(ROOT).as_posix()
            inputs[relative] = receipt["files"][name]
            files[name] = relative
        jobs.append(dict(episode=item["job_id"], map_id=item["map_id"], task_id=item["task_id"], files=files))
    require(len(jobs) == 96 and len({j["map_id"] for j in jobs}) == 8, "unexpected cohort")
    native = _native_batch_function()
    require(native is not None, "feature-only native unavailable")
    native_path = Path(sys.modules[native.__module__].__file__).resolve()
    inputs[native_path.relative_to(ROOT).as_posix()] = sha256_file(native_path)
    for directory in ("experiments", "scripts"):
        for p in (ROOT/directory).glob("*.py"):
            inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    for p in [CONFIG, ref, source/"registration.json", source/"manifest.json",
              ROOT/"docs/SA_HISTORY_INFORMATION_AUDIT_ZH.md"]:
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    # All old frozen artifacts and reference roots remain protected, even though no policy is fitted.
    for name, digest in read_json(ref)["inputs"].items():
        if name.startswith("artifacts/") or name.endswith(".so"):
            require(sha256_file(ROOT/name) == digest, "frozen artifact changed")
            inputs[name] = digest
    for t in read_json(ref)["targets"]:
        p = ref.parent/"states"/t["id"]/"root.json"
        receipt = read_json(p.parent/"receipt.json")
        require(sha256_file(p) == receipt["files"]["root.json"], "root changed")
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    plan = dict(config=cfg, inputs=inputs, jobs=jobs,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = json_fingerprint(plan)
    atomic(out/"plan.json", plan)
    return dict(episodes=len(jobs), max_landmarks=len(jobs)*len(cfg["decisions"]),
                max_fits=len(cfg["profiles"])*8*len(cfg["targets"]), new_solver_jobs=0)


def verify():
    cfg = read_json(CONFIG)
    out = ROOT/cfg["output"]
    plan = read_json(out/"plan.json")
    require(plan["binding"] == json_fingerprint({k:v for k,v in plan.items() if k != "binding"}), "plan fingerprint")
    require(plan["config"] == cfg, "config changed")
    for name, digest in plan["inputs"].items():
        require(sha256_file(ROOT/name) == digest, "input changed: " + name)
    return plan, out


def extract_episode(job):
    from experiments.online_feature_engine import OnlineFeatureEngine
    cfg = job["config"]
    state = read_json(ROOT/job["files"]["initial.json"])["payload"]["observation"]
    initial_count = state["num_of_colliding_pairs"]
    history, ordered = History(state, 32), OrderedHistory()
    rows, counts = [], []
    max_decision = max(cfg["decisions"]) + cfg["future_horizon"] - 1
    with (ROOT/job["files"]["first_phase/trace.jsonl"]).open(encoding="utf8") as f:
        for line in f:
            event = json.loads(line)
            d = event["decision"]
            if d > max_decision:
                break
            require(d == history.decision, "source sequence discontinuity")
            if d in cfg["decisions"]:
                candidate = event["pool"][event["selected_index"]]
                require(sorted(candidate["agents"]) == sorted(event["action"]["agents"]), "chosen candidate mismatch")
                engine = OnlineFeatureEngine(state, backend="native")
                extracted, _ = engine.realized_rows([candidate], state_hash=state_fingerprint(state))
                base = extracted[0]["features"]["realized_dynamic"] | {"source.score": candidate["score"]}
                base |= history.features(state, candidate, event["temperature"])
                rows.append(dict(id=f"{job['episode']}-d{d:04d}", episode=job["episode"], map_id=job["map_id"],
                    decision=d, current_conflicts=state["num_of_colliding_pairs"], previous_best=history.best,
                    stall=history.since_best, candidate_id=candidate["candidate_id"], base=base,
                    records=ordered.records(state, candidate), state_fingerprint=state_fingerprint(state)))
            after = apply_state_delta(state, event["delta"])
            require(event["metrics"]["action_valid"] and event["metrics"]["step_applied"], "source invalid action")
            ordered.observe(state, event, after, history.best)
            history.observe(state, event, after)
            counts.append(after["num_of_colliding_pairs"])
            state = after
    for row in rows:
        row["labels"] = future_labels(counts, row["decision"], row["previous_best"], cfg["future_horizon"], cfg["sustain"])
    return dict(episode=job["episode"], map_id=job["map_id"], initial_conflicts=initial_count, rows=rows,
                observed_steps=len(counts), observed_terminal=bool(counts and counts[-1] == 0),
                missing_decisions=[d for d in cfg["decisions"] if d >= len(counts)])


def parity(plan):
    from experiments.online_feature_engine import OnlineFeatureEngine
    ref = ROOT/plan["config"]["reference_plan"]
    errors = []
    for t in read_json(ref)["targets"]:
        root = read_json(ref.parent/"states"/t["id"]/"root.json")
        extracted, _ = OnlineFeatureEngine(root["state"], backend="native").realized_rows(
            root["candidates"], state_hash=root["state_fingerprint"])
        for row, expected in zip(extracted, root["feature_rows"]):
            values = row["features"]["realized_dynamic"]
            for k,v in values.items():
                require(k in expected and math.isclose(v, expected[k], rel_tol=1e-10, abs_tol=1e-10), "feature parity: "+k)
                errors.append(abs(v-expected[k]))
    return dict(roots=len(read_json(ref)["targets"]), max_absolute_error=max(errors))


def receipt_result(path, binding):
    value = read_json(path)
    require(value["binding"] == binding, "result binding")
    require(read_json(path.with_suffix(".receipt.json"))["sha256"] == sha256_file(path), "result changed")
    return value["result"]


def save_result(path, binding, result):
    require(not path.exists(), "refuse overwrite result")
    atomic(path, dict(binding=binding, result=result))
    atomic(path.with_suffix(".receipt.json"), dict(sha256=sha256_file(path)))


def build(plan, out):
    parity_result = parity(plan)
    progress(out, "parity", **parity_result)
    cfg = plan["config"]
    results = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        pending = {}
        for job in plan["jobs"]:
            path = out/"episodes"/(job["episode"]+".json")
            if path.exists():
                results.append(receipt_result(path, plan["binding"]))
            else:
                pending[pool.submit(extract_episode, job | {"config":cfg})] = path
        for future in as_completed(pending):
            result = future.result()
            save_result(pending[future], plan["binding"], result)
            results.append(result)
            progress(out, "extract", complete=len(results), total=len(plan["jobs"]), episode=result["episode"])
    results.sort(key=lambda r:r["episode"])
    rows = [r for result in results for r in result["rows"]]
    coverage = dict(episodes=len(results), landmarks=len(rows), eligible=sum(r["labels"] is not None for r in rows),
                    censored=sum(r["labels"] is None for r in rows), absent=sum(len(r["missing_decisions"]) for r in results),
                    parity=parity_result)
    if (out/"index.json").exists():
        require(receipt_result(out/"index.json",plan["binding"]) == dict(rows=rows, coverage=coverage), "rebuilt index changed")
    else:
        save_result(out/"index.json", plan["binding"], dict(rows=rows, coverage=coverage))
    return coverage


def fit_fold(job):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    rows, held = job["rows"], job["map_id"]
    train = [r for r in rows if r["map_id"] != held]
    test = [r for r in rows if r["map_id"] == held]
    require(set(r["episode"] for r in train).isdisjoint(r["episode"] for r in test), "episode leakage")
    profile, target = job["profile"], job["target"]
    # Feature names encode a fixed schema only; no fitted imputer or selector sees test values.
    vectors = {r["id"]:profile_features(r,profile) for r in rows}
    names = sorted({k for v in vectors.values() for k in v})
    def x(part):
        return np.asarray([[vectors[r["id"]].get(k,0) for k in names] for r in part],dtype=float)
    y = np.asarray([r["labels"][target] for r in train])
    w = np.asarray(episode_weights(train))
    prior = float(np.average(y,weights=w))
    if len(set(y)) == 1:
        predictions = [prior]*len(test)
    else:
        model = HistGradientBoostingClassifier(**job["model"]).fit(x(train),y,sample_weight=w)
        predictions = model.predict_proba(x(test))[:,1].tolist()
    return dict(map_id=held, profile=profile, target=target, train_episodes=len({r["episode"] for r in train}),
        features=len(names), constant_train_columns=int(np.sum(np.ptp(x(train),axis=0)==0)),
        training_single_class=len(set(y))==1, rows=[dict(id=r["id"],prediction=p,prior=prior) for r,p in zip(test,predictions)])


def comparison(rows, new, old, target, repetitions, seed):
    import numpy as np
    maps = sorted({r["map_id"] for r in rows})
    per_map = {}
    for name in maps:
        subset = [r for r in rows if r["map_id"] == name]
        per_map[name] = dict(delta=metric(subset,[new[r["id"]] for r in subset],target)["brier"]-
                                  metric(subset,[old[r["id"]] for r in subset],target)["brier"],
                             episodes=len({r["episode"] for r in subset}))
    rng = np.random.default_rng(seed)
    chosen = rng.integers(0,len(maps),size=(repetitions,len(maps)))
    values = np.array([per_map[k]["delta"] for k in maps])
    weights = np.array([per_map[k]["episodes"] for k in maps])
    boot = np.sum(values[chosen]*weights[chosen],axis=1)/np.sum(weights[chosen],axis=1)
    a=metric(rows,[new[r["id"]] for r in rows],target)["brier"]
    b=metric(rows,[old[r["id"]] for r in rows],target)["brier"]
    return dict(delta=a-b, relative_improvement=(b-a)/b if b else None,
                ci95=np.quantile(boot,[.025,.975]).tolist(), nonworse_maps=sum(v["delta"]<=0 for v in per_map.values()),
                per_map=per_map)


def summarize(rows, folds, cfg):
    predictions = {}
    for f in folds:
        key = (f["profile"],f["target"])
        predictions.setdefault(key,{}).update({r["id"]:r["prediction"] for r in f["rows"]})
        predictions.setdefault(("prior",f["target"]),{}).update({r["id"]:r["prior"] for r in f["rows"]})
    metrics = {t:{p:metric(rows,[predictions[p,t][r["id"]] for r in rows],t)
                  for p in cfg["profiles"]+["prior"]} for t in cfg["targets"]}
    comparisons = {}
    pairs=[("aggregate","dynamic"),("resource_bag","aggregate"),("ordered","aggregate"),
           ("resource_ordered","aggregate"),("resource_ordered","resource_bag")]
    pairs += [("resource_ordered",f"shuffled_{i}") for i in range(5)]
    for t in cfg["targets"]:
        comparisons[t]={a+" vs "+b:comparison(rows,predictions[a,t],predictions[b,t],t,cfg["bootstrap"],cfg["seed"])
                        for a,b in pairs}
    strata={}
    for name,predicate in (("stalled_16",lambda r:r["stall"]>=16),("not_stalled_16",lambda r:r["stall"]<16),
                           ("conflicts_1_10",lambda r:r["current_conflicts"]<=10),
                           ("conflicts_gt10",lambda r:r["current_conflicts"]>10)):
        part=[r for r in rows if predicate(r)]
        strata[name]=dict(states=len(part),episodes=len({r["episode"] for r in part}),
            metrics={t:{p:metric(part,[predictions[p,t][r["id"]] for r in part],t) for p in cfg["profiles"]}
                     for t in cfg["targets"]} if part else {})
    primary=comparisons["sustained_progress"]
    exploratory=[]
    for p in ("resource_bag","ordered","resource_ordered"):
        c=primary[p+" vs aggregate"]
        if c["relative_improvement"] is not None and c["relative_improvement"]>=cfg["signal_relative_brier_improvement"] and c["nonworse_maps"]>=6 and c["ci95"][0]<=0:
            exploratory.append(p)
    order_tests=[primary["resource_ordered vs "+s] for s in ["resource_bag"]+[f"shuffled_{i}" for i in range(5)]]
    return dict(metrics=metrics,comparisons=comparisons,strata=strata,exploratory_profiles=exploratory,
        stable_order_signal=all(c["delta"]<0 and c["ci95"][1]<0 for c in order_tests),
        decision="exploratory_information_only" if exploratory else "no_incremental_information_signal",
        no_causal_action_ranking_claim=True,no_policy_promotion=True)


def analyze(plan,out):
    cfg=plan["config"]
    index=receipt_result(out/"index.json",plan["binding"])
    rows=[r for r in index["rows"] if r["labels"] is not None]
    minimum=cfg["minimum"]
    support={t:dict(positive=sum(r["labels"][t] for r in rows),negative=sum(1-r["labels"][t] for r in rows)) for t in cfg["targets"]}
    eligible=(len({r["map_id"] for r in rows})>=minimum["maps"] and len({r["episode"] for r in rows})>=minimum["episodes"]
              and support["sustained_progress"]["positive"]>=minimum["positives"]
              and support["sustained_progress"]["negative"]>=minimum["negatives"])
    progress(out,"label_support",support=support,eligible=eligible)
    if not eligible:
        report=dict(decision="insufficient_support",support=support,coverage=index["coverage"],no_policy_promotion=True)
    else:
        jobs=[dict(rows=rows,map_id=m,profile=p,target=t,model=cfg["model"])
              for m in sorted({r["map_id"] for r in rows}) for p in cfg["profiles"] for t in cfg["targets"]]
        folds=[]
        with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
            pending={}
            for job in jobs:
                name=json_fingerprint([job["map_id"],job["profile"],job["target"]])[:20]
                path=out/"folds"/(name+".json")
                if path.exists():
                    folds.append(receipt_result(path,plan["binding"]))
                else:
                    pending[pool.submit(fit_fold,job)]=path
            for future in as_completed(pending):
                result=future.result()
                save_result(pending[future],plan["binding"],result)
                folds.append(result)
                progress(out,"fit",complete=len(folds),total=len(jobs),profile=result["profile"],target=result["target"],map_id=result["map_id"])
        folds.sort(key=lambda f:(f["map_id"],f["profile"],f["target"]))
        report=summarize(rows,folds,cfg) | dict(support=support,coverage=index["coverage"],
               effective_maps=len({r["map_id"] for r in rows}),effective_episodes=len({r["episode"] for r in rows}),
               folds=len(folds),censoring=[{k:r[k] for k in ("id","map_id","decision","current_conflicts","stall")}
                                           for r in index["rows"] if r["labels"] is None])
    report |= dict(binding=plan["binding"],index_sha256=sha256_file(out/"index.json"),schema=cfg["schema"])
    if (out/"report.json").exists():
        require(read_json(out/"report.json")==report,"recomputed report changed")
    else:
        atomic(out/"report.json",report)
    return dict(decision=report["decision"],report_sha256=sha256_file(out/"report.json"))


def main():
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[name]="1"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","build","analyze"))
    args=parser.parse_args()
    cfg=read_json(CONFIG)
    out=ROOT/cfg["output"]
    out.mkdir(parents=True,exist_ok=True) if args.phase != "prepare" else None
    if args.phase=="prepare":
        print(json.dumps(prepare(),sort_keys=True))
        return
    plan,out=verify()
    lock=out/"run.lock"
    with lock.open("x",encoding="utf8") as f:
        f.write(str(os.getpid()))
    start=time.monotonic()
    try:
        atomic(out/"run_status.json",dict(status="running",phase=args.phase,pid=os.getpid(),binding=plan["binding"]))
        value=build(plan,out) if args.phase=="build" else analyze(plan,out)
        verify()
        atomic(out/"run_status.json",dict(status="completed",phase=args.phase,seconds=time.monotonic()-start,binding=plan["binding"]))
        print(json.dumps(value,sort_keys=True))
    except BaseException as e:
        atomic(out/"run_status.json",dict(status="failed",phase=args.phase,error=repr(e),binding=plan["binding"]))
        raise
    finally:
        lock.unlink()


if __name__=="__main__":
    main()
