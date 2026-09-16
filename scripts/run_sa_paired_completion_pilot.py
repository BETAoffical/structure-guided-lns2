"""Bounded input-selected labels and map-held-out prototype evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, write_json, sha256_file, json_fingerprint
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.repair_collection import state_fingerprint
from experiments.sa_history_selector import History
from experiments.sa_history_information import profile_features
from experiments.sa_paired_sampling import choose_candidates, choose_roots, work_budget
from experiments.sa_paired_completion import SCHEMA, MODEL_PARAMS, PairedCompletionModel, inspect_dataset, training_matrix, require
from scripts.train_sa_history_selector import die_with_parent, verify as verify_sampling
from scripts.collect_sa_history_candidate_bridge import branch, branch_labels, stability

CONFIG = ROOT / "configs/sa_paired_completion_pilot.json"


def history_fingerprint(history):
    return json_fingerprint(dict(decision=history.decision, recent=list(history.recent),
        ages=sorted((list(k), v) for k, v in history.ages.items()), best=history.best, since_best=history.since_best))


def scan_source(job):
    cfg, item = job["config"], job["item"]
    folder = ROOT / cfg["source"] / "episodes" / item["job_id"]
    state = read_json(folder / "initial.json")["payload"]["observation"]
    history = History(state)
    found = []
    with (folder / "first_phase/trace.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            e = json.loads(line)
            d = e["decision"]
            require(d == history.decision, "source sequence changed")
            if d in cfg["milestones"] and len(e["pool"]) >= 2:
                sid = f"{item['job_id']}-d{d:04d}"
                anchor = e["pool"][e["selected_index"]]["candidate_id"]
                chosen = choose_candidates(e["pool"], anchor, sid, cfg["seed"], cfg["max_candidates"])
                found.append(dict(id=sid, item=item, map_id=job["map_id"], decision=d,
                    phase="early" if d == 4 else "continuing", selected=chosen, anchor_id=anchor,
                    previous_best=history.best, state_fingerprint=state_fingerprint(state),
                    history_fingerprint=history_fingerprint(history), pool_sha=json_fingerprint(e["pool"]),
                    selected_index=e["selected_index"], initial_conflicts=state["num_of_colliding_pairs"],
                    sampled_sizes=[len(next(c for c in e["pool"] if c["candidate_id"] == cid)["agents"]) for cid in chosen],
                    source_pool_count=len(e["pool"])))
            if d >= max(cfg["milestones"]):
                break
            after = apply_state_delta(state, e["delta"])
            history.observe(state, e, after)
            state = after
    return found


def prepare():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    require(not out.exists(), "output exists; do not overwrite previous preparation")
    require((cfg["maps"], cfg["states"], cfg["max_candidates"], cfg["trials"], cfg["horizon"], cfg["workers"]) ==
            (8, 16, 4, 8, 32, 20), "registered pilot budget changed")
    source_plan = verify_sampling(ROOT / cfg["old_sampling_config"])
    inputs = dict(source_plan["inputs"])
    source = ROOT / cfg["source"]
    reg = read_json(source / "registration.json")
    excluded = read_json(ROOT / cfg["excluded_bridge"])
    require(excluded["binding"] == json_fingerprint({k:v for k,v in excluded.items() if k != "binding"}), "exclusion binding")
    episodes = {r["target"]["item"]["job_id"] for r in excluded["roots"]}
    cases = {c["task_id"]: c for c in reg["cases"]}
    jobs = [dict(config=cfg, item=item, map_id=cases[item["task_id"]]["map_id"])
            for item in reg["schedule"] if item["controller"] == "dual16_sa" and item["job_id"] not in episodes]
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        available = [r for rows in pool.map(scan_source, jobs) for r in rows]
    maps = {c["map_id"] for c in reg["cases"]}
    roots, coverage = choose_roots(available, maps, episodes, cfg["seed"])
    paths = [CONFIG, Path(__file__), ROOT / cfg["excluded_bridge"],
             ROOT / "experiments/sa_paired_sampling.py", ROOT / "experiments/sa_paired_completion.py",
             ROOT / "tests/evaluation/test_sa_paired_completion_pilot.py",
             ROOT / "docs/SA_PAIRED_COMPLETION_PILOT_ZH.md",
             ROOT / "scripts/collect_sa_history_candidate_bridge.py"]
    for path in paths:
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for path, digest in inputs.items():
        require(sha256_file(ROOT / path) == digest, "source changed during preflight: " + path)
    plan = dict(config=cfg, roots=roots, coverage=coverage, inputs=inputs,
        excluded_episodes=sorted(episodes), model=MODEL_PARAMS, available_occurrences=len(available),
        native_file=source_plan["native_file"], native_sha256=source_plan["native_sha256"],
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        admission=len(maps) == cfg["maps"] and len(roots) == cfg["states"] and all(c["selected"] == 2 for c in coverage))
    plan["binding"] = json_fingerprint(plan)
    write_json(out / "plan.json", plan)
    return dict(binding=plan["binding"], admission=plan["admission"], budget=work_budget(roots, cfg), coverage=coverage)


def verify():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg and plan["binding"] == json_fingerprint({k:v for k,v in plan.items() if k != "binding"}), "plan changed")
    require(plan["model"] == MODEL_PARAMS, "training parameters changed")
    for name, digest in plan["inputs"].items():
        require(sha256_file(ROOT / name) == digest, "registered input changed: " + name)
    return plan, out


def expected_files(target, cfg):
    return {"root.json", "control.json"} | {f"{cid}-t{t}.json" for cid in target["selected"] for t in range(cfg["trials"])}


def verify_receipt(folder, target, plan):
    receipt = read_json(folder / "receipt.json")
    require(receipt["binding"] == plan["binding"] and set(receipt["files"]) == expected_files(target, plan["config"]), "receipt coverage")
    require({p.name for p in folder.glob("*.json")} == set(receipt["files"]) | {"receipt.json"}, "unexpected result files")
    for name, digest in receipt["files"].items():
        require(sha256_file(folder/name) == digest, "saved result changed")
        require(read_json(folder/name)["binding"] == plan["binding"], "result binding changed")


def root_worker(job):
    die_with_parent(job["parent_pid"])
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from experiments.online_feature_engine import OnlineFeatureEngine
    plan, target = job["plan"], job["target"]
    cfg = plan["config"]
    folder = ROOT / cfg["output"] / "roots" / target["id"]
    if (folder / "receipt.json").exists():
        verify_receipt(folder, target, plan)
        return dict(status="ok", id=target["id"], resumed=True)
    require(not folder.exists(), "partial root needs audit; no automatic retry")
    require(q.native_identity()["sha256"] == plan["native_sha256"], "wrong native")
    reg = read_json(ROOT / cfg["source"] / "registration.json")
    case = next(c for c in reg["cases"] if c["task_id"] == target["item"]["task_id"])
    worker = q.worker_job(case, target["item"], reg["template"], folder / "unused", plan["binding"])
    original = ROOT / cfg["source"] / "episodes" / target["item"]["job_id"]
    expected = read_json(original / "initial.json")["payload"]["observation"]
    env = q._make_environment(worker["dataset_root"], worker["row"], dict(worker["environment"], time_limit=100000.), "Adaptive")
    state = q._plain(env.reset(seed=worker["solver_seed"]))
    require(q.state_fingerprint(state) == q.state_fingerprint(expected), "initial mismatch")
    history = History(state)
    with (original / "first_phase/trace.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["decision"] == target["decision"]:
                break
            raw = q._plain(env.step_experimental_pp(event["action"], cfg["replay_pp_seconds"], "annealed", event["temperature"], event["uniform"]))
            after = raw["observation"]
            expected = q.apply_state_delta(expected, event["delta"])
            check_attempt(raw["metrics"], event["metrics"])
            require(q.state_fingerprint(after) == q.state_fingerprint(expected), "prefix mismatch")
            history.observe(state, event, after)
            state = after
    require(history.decision == target["decision"] and event["decision"] == target["decision"], "missing root occurrence")
    require(q.state_fingerprint(state) == target["state_fingerprint"] and history_fingerprint(history) == target["history_fingerprint"], "root/history mismatch")
    source_case = dict(case_id=worker["sa_case_id"], task_id=case["task_id"], solver_seed=worker["solver_seed"], proposal=worker["sa_proposal"])
    index, pool = q.SingleFullCheckPool(source_case).select(env, state, target["decision"])
    require((index, pool) == (event["selected_index"], event["pool"]), "candidate pool replay mismatch")
    require(json_fingerprint(pool) == target["pool_sha"] and q.state_fingerprint(env.get_state()) == target["state_fingerprint"], "proposal changed state or pool")
    require(choose_candidates(pool, target["anchor_id"], target["id"], cfg["seed"], cfg["max_candidates"]) == target["selected"], "blind sampling mismatch")
    candidates = [next(c for c in pool if c["candidate_id"] == cid) for cid in target["selected"]]
    features, _ = OnlineFeatureEngine(state, backend="native").realized_rows(candidates, state_hash=target["state_fingerprint"])
    values = [r["features"]["realized_dynamic"] | {"source.score": c["score"]} |
              history.features(state, c, event["temperature"]) for r,c in zip(features, candidates)]
    require(len(values) == len(candidates) and all(len(profile_features(dict(base=v), "dynamic")) == 127
                                                for v in values), "dynamic feature contract changed")
    root = dict(binding=plan["binding"], state=state, state_fingerprint=target["state_fingerprint"],
        source=target, control_event=event, candidates=candidates, feature_rows=values,
        old_selected_id=target["anchor_id"], previous_best=history.best)
    require(len(list(Path("/proc/self/task").iterdir())) == 1, "fork requires single-threaded parent")
    folder.mkdir(parents=True)
    write_json(folder / "root.json", root)
    lookup = {c["candidate_id"]: c for c in candidates}
    pending = [(lookup[target["anchor_id"]], -1)] + [(c,t) for c in candidates for t in range(cfg["trials"])]
    active = []
    ctx = multiprocessing.get_context("fork")
    entry = dict(target=target, previous_best=history.best)
    try:
        while pending or active:
            while pending and len(active) < cfg["workers"]:
                candidate, trial = pending.pop(0)
                path = folder / ("control.json" if trial == -1 else f"{candidate['candidate_id']}-t{trial}.json")
                p = ctx.Process(target=branch, args=(env, root, source_case, entry, candidate, trial, plan, path, os.getpid()))
                p.start()
                active.append((p, time.monotonic(), path, candidate, trial))
            for item in list(active):
                p, start, path, candidate, trial = item
                if p.is_alive() and time.monotonic()-start < cfg["branch_fuse_seconds"]:
                    continue
                if p.is_alive():
                    p.terminate()
                    p.join()
                    write_json(path, dict(status="censored", binding=plan["binding"], root_id=target["id"],
                        candidate_id=candidate["candidate_id"], trial=trial, stop="external_timeout"))
                else:
                    p.join()
                    require(p.exitcode == 0 and path.exists() and read_json(path)["status"] == "ok", "branch crashed or error")
                active.remove(item)
            time.sleep(.05)
    finally:
        for p, *_ in active:
            if p.is_alive():
                p.terminate()
            p.join()
    require(q.state_fingerprint(env.get_state()) == target["state_fingerprint"], "branch mutated parent")
    require(read_json(folder / "control.json").get("stop") == "control", "original action control incomplete")
    files = expected_files(target, cfg)
    require({p.name for p in folder.glob("*.json")} == files, "unexpected branch files")
    write_json(folder / "receipt.json", dict(binding=plan["binding"], files={n:sha256_file(folder/n) for n in sorted(files)}))
    verify_receipt(folder, target, plan)
    return dict(status="ok", id=target["id"], branches=len(files)-2)


def collect(resume=False, limit=None):
    from scripts import run_sa_path_quality as q
    plan, out = verify()
    require(plan["admission"], "source coverage gate failed")
    require(resume or not (out / "roots").exists(), "existing collection requires resume")
    if limit is not None:
        require(0 < limit <= len(plan["roots"]), "invalid root limit")
    with q._CollectionRunLock(out, plan["binding"], "paired-completion-pilot"):
        write_json(out / "run_status.json", dict(status="running", binding=plan["binding"]))
        try:
            for target in plan["roots"][:limit]:
                if (out / "STOP_AFTER_ROOT").exists():
                    break
                job = dict(job_id=target["id"], target=target, plan=plan, parent_pid=os.getpid())
                results = q._run_jobs(root_worker, [job], workers=1, phase=target["id"],
                    output_root=out/"progress"/target["id"], run_fingerprint=plan["binding"],
                    timeout_seconds=plan["config"]["root_fuse_seconds"], stop_on_failure=True)
                require(len(results) == 1 and results[0]["status"] == "ok", "root failed")
                n = sum((out/"roots"/r["id"]/"receipt.json").exists() for r in plan["roots"])
                write_json(out / "run_status.json", dict(status="running", binding=plan["binding"], completed_roots=n))
                print("ROOT", n, "/", len(plan["roots"]), target["id"], flush=True)
            verify()
            complete = all((out/"roots"/r["id"]/"receipt.json").exists() for r in plan["roots"])
            write_json(out / "run_status.json", dict(status="completed" if complete else "paused", binding=plan["binding"]))
        except BaseException as exc:
            write_json(out / "run_status.json", dict(status="error", binding=plan["binding"], error=repr(exc)))
            raise
    return dict(complete=complete)


def save_once(path, value):
    if path.exists():
        require(read_json(path) == value, "existing output differs: " + path.name)
    else:
        write_json(path, value)


def analyze():
    plan, out = verify()
    cfg, states, names = plan["config"], [], None
    from scripts import run_sa_path_quality as q
    for target in plan["roots"]:
        folder = out / "roots" / target["id"]
        verify_receipt(folder, target, plan)
        root = read_json(folder / "root.json")
        require(root["source"] == target and q.state_fingerprint(root["state"]) == target["state_fingerprint"], "root identity")
        require([c["candidate_id"] for c in root["candidates"]] == target["selected"] and
                root["old_selected_id"] == target["anchor_id"], "candidate/anchor identity")
        candidates = []
        for c, base in zip(root["candidates"], root["feature_rows"], strict=True):
            features = profile_features(dict(base=base), "dynamic")
            if names is None:
                names = sorted(features)
            require(sorted(features) == names, "feature schema changed")
            trials = []
            for t in range(cfg["trials"]):
                row = read_json(folder / f"{c['candidate_id']}-t{t}.json")
                require((row["root_id"], row["candidate_id"], row["trial"]) == (target["id"], c["candidate_id"], t), "trial identity")
                labels = None if row["status"] == "censored" else branch_labels(row, root, dict(target=target, previous_best=root["previous_best"]), cfg)
                trials.append(dict(trial=t, randomization_key=json_fingerprint([plan["binding"], target["id"], t]),
                    stop=row["stop"], steps=len(row.get("events", [])),
                    final_conflicts=row.get("final_conflicts", root["state"]["num_of_colliding_pairs"]),
                    completed=None if labels is None else bool(labels["completion"])))
            candidates.append(dict(candidate_id=c["candidate_id"], agents=c["agents"], features=features, trials=trials))
        states.append(dict(state_id=target["id"], map_id=target["map_id"], episode=target["item"]["job_id"],
            decision=target["decision"], anchor_id=target["anchor_id"], phase=target["phase"],
            state_fingerprint=target["state_fingerprint"], history_fingerprint=target["history_fingerprint"],
            agent_ids=[a["id"] for a in root["state"]["agents"]], candidates=candidates))
    data = dict(schema=SCHEMA, role="prospective_development", sampling="outcome_blind_same_state",
        source_kind="registered_prospective_collection", continuation_binding=plan["binding"],
        horizon=cfg["horizon"], trial_count=cfg["trials"], feature_names=names, states=states)
    report = inspect_dataset(data)
    report.update(binding=plan["binding"], no_ttf=True, data_fingerprint=json_fingerprint(data))
    verify()
    save_once(out / "training_index.json", data)
    save_once(out / "label_report.json", report)
    return {k:v for k,v in report.items() if k != "states"}


def fit_fold(job):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    data, held = job["data"], job["map_id"]
    paired = PairedCompletionModel.fit(data, {held})
    names = data["feature_names"]
    training = sorted((s for s in data["states"] if s["map_id"] != held), key=lambda s:s["state_id"])
    x, y, w = [], [], []
    for s in training:
        for c in sorted(s["candidates"], key=lambda c:c["candidate_id"]):
            x.append([c["features"][n] for n in names])
            y.append(sum(t["completed"] for t in c["trials"])/len(c["trials"]))
            w.append(1/len(s["candidates"]))
    direct = HistGradientBoostingRegressor(loss="squared_error", **MODEL_PARAMS).fit(x,y,sample_weight=w)
    predictions = []
    for s in sorted((s for s in data["states"] if s["map_id"] == held), key=lambda s:s["state_id"]):
        ordered = sorted(s["candidates"], key=lambda c:c["candidate_id"])
        values = direct.predict(np.array([[c["features"][n] for n in names] for c in ordered]))
        ds = {c["candidate_id"]: float(v) for c,v in zip(ordered,values)}
        selected = min(ds, key=lambda c:(-ds[c], c != s["anchor_id"], c))
        predictions.append(dict(state_id=s["state_id"], paired=paired.rank(s), pointwise=dict(selected=selected, scores=ds)))
    return dict(map_id=held, train_ids=[s["state_id"] for s in training], predictions=predictions)


def _fit_evaluate():
    import numpy as np
    plan, out = verify()
    data = read_json(out / "training_index.json")
    labels = read_json(out / "label_report.json")
    require(labels["binding"] == plan["binding"] and labels["data_fingerprint"] == json_fingerprint(data), "data identity")
    require(inspect_dataset(data) == {k:v for k,v in labels.items() if k not in ("binding","no_ttf","data_fingerprint")}, "label report changed")
    maps = sorted({s["map_id"] for s in data["states"]})
    input_sha = sha256_file(out / "training_index.json")
    receipt_path = out / "training_receipt.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        require(receipt["binding"] == plan["binding"] and receipt["input_sha256"] == input_sha and
                receipt["report_sha256"] == sha256_file(out/"model_report.json"), "training receipt changed")
        report = read_json(out/"model_report.json")
        return dict(decision=report["decision"], summary=report.get("summary"), resumed=True)
    require(not (out / "model_report.json").exists(), "incomplete training publication needs audit")
    issues = {}
    for m in maps:
        try:
            training_matrix(data, {m})
        except ValueError as exc:
            issues[m] = str(exc)
    if issues:
        report = dict(binding=plan["binding"], input_sha256=input_sha, decision="not_trained_data_contract_or_fold_support",
                      issues=issues, real_data_fits=0, promotion_allowed=False)
        save_once(out/"model_report.json", report)
        write_json(receipt_path, dict(binding=plan["binding"], input_sha256=input_sha, report_sha256=sha256_file(out/"model_report.json")))
        return report
    jobs = [dict(data=data, map_id=m) for m in maps]
    with ProcessPoolExecutor(max_workers=min(plan["config"]["workers"],len(maps))) as pool:
        folds = []
        for fold in pool.map(fit_fold,jobs):
            folds.append(fold)
            print("TRAIN_FOLD", len(folds), "/", len(maps), fold["map_id"], flush=True)
    predicted = {p["state_id"]:p for f in folds for p in f["predictions"]}
    states = []
    for s in data["states"]:
        rate = {c["candidate_id"]:sum(t["completed"] for t in c["trials"])/len(c["trials"]) for c in s["candidates"]}
        p = predicted[s["state_id"]]
        rates = dict(frozen=rate[s["anchor_id"]], uniform=sum(rate.values())/len(rate), oracle=max(rate.values()),
                     paired=rate[p["paired"]["selected"]], pointwise=rate[p["pointwise"]["selected"]])
        trial_stability = stability({c["candidate_id"]:[int(t["completed"]) for t in sorted(c["trials"],key=lambda r:r["trial"])] for c in s["candidates"]})
        states.append(dict(state_id=s["state_id"], map_id=s["map_id"], phase=s["phase"], rates=rates, predictions=p,
                           trial_stability=trial_stability))
    summary = {k:sum(s["rates"][k] for s in states)/len(states) for k in states[0]["rates"]}
    contrasts = {}
    for baseline in ("frozen","uniform","pointwise"):
        values = np.array([sum(s["rates"]["paired"]-s["rates"][baseline] for s in states if s["map_id"]==m)/sum(s["map_id"]==m for s in states) for m in maps])
        draws = np.random.default_rng(plan["config"]["seed"]).integers(0,len(maps),(plan["config"]["bootstrap"],len(maps)))
        contrasts[baseline] = dict(delta=float(values.mean()), ci95=np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist(),
            map_wins=int(sum(values>0)), map_losses=int(sum(values<0)), map_ties=int(sum(values==0)))
    exploratory_signal = contrasts["frozen"]["delta"] >= .05 and contrasts["frozen"]["ci95"][0] >= 0 and contrasts["frozen"]["map_wins"] >= 5 and contrasts["uniform"]["delta"] > 0 and contrasts["pointwise"]["delta"] >= 0
    report = dict(binding=plan["binding"], input_sha256=input_sha, folds=folds, states=states, summary=summary,
        paired_contrasts=contrasts, decision="limited_offline_signal_only" if exploratory_signal else "no_reliable_paired_model_advantage",
        promotion_allowed=False, runtime_integration_allowed=False, no_ttf=True, independent_confirmation=False)
    verify()
    require(sha256_file(out/"training_index.json") == input_sha, "data changed during fitting")
    save_once(out/"model_report.json", report)
    write_json(receipt_path, dict(binding=plan["binding"], input_sha256=input_sha, report_sha256=sha256_file(out/"model_report.json")))
    return dict(summary=summary, paired_contrasts=contrasts, decision=report["decision"], promotion_allowed=False)


def fit_evaluate():
    from scripts import run_sa_path_quality as q
    plan, out = verify()
    with q._CollectionRunLock(out, plan["binding"], "paired-completion-training"):
        write_json(out / "training_status.json", dict(status="running", binding=plan["binding"]))
        try:
            result = _fit_evaluate()
            write_json(out / "training_status.json", dict(status="completed", binding=plan["binding"],
                                                         decision=result["decision"]))
            return result
        except BaseException as exc:
            write_json(out / "training_status.json", dict(status="error", binding=plan["binding"], error=repr(exc)))
            raise


def main():
    for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[k] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare","dry-run","collect","analyze","fit-evaluate","request-stop"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare()
    elif args.phase == "dry-run":
        plan,_ = verify()
        result = dict(admission=plan["admission"], budget=work_budget(plan["roots"],plan["config"]))
    elif args.phase == "collect":
        result = collect(args.resume,args.limit)
    elif args.phase == "analyze":
        result = analyze()
    elif args.phase == "fit-evaluate":
        result = fit_evaluate()
    else:
        plan,out = verify()
        write_json(out/"STOP_AFTER_ROOT",dict(binding=plan["binding"],requested=True))
        result = dict(stop_after_current_root=True)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
