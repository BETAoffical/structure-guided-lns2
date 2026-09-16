"""Independent SA history pilot: prepare, bounded counterfactuals, offline training."""
import argparse
from collections import Counter
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, write_json, write_jsonl, sha256_file, json_fingerprint
from experiments.sa_history_selector import History, choose_candidates, targets, aggregate_trials, select_prediction, pareto_indices

CONFIG = ROOT / "configs/sa_history_selector_pilot.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def locations(config_path=None):
    config = read_json(config_path or CONFIG)
    return config, ROOT / config["output"], ROOT / config["source"]


def prepare(config_path=None):
    cfg, out, source = locations(config_path)
    require(not (out / "plan.json").exists(), "plan exists; preserve it")
    reg = read_json(source / "registration.json")
    require(reg["fingerprint"] == json_fingerprint({k: v for k, v in reg.items() if k != "fingerprint"}), "source registration corrupt")
    manifest = read_json(source / "manifest.json")
    require(manifest["binding"] == reg["fingerprint"], "source manifest binding")
    cases = {c["task_id"]: c for c in reg["cases"]}
    available = []
    inputs = {}
    scan_jobs = []
    stratified = cfg.get("sampling", {}).get("mode") == "history_stratified"
    for item in reg["schedule"]:
        if item["controller"] != "dual16_sa":
            continue
        folder = source / "episodes" / item["job_id"]
        receipt = manifest["jobs"][item["job_id"]]
        for name in ("initial.json", "first_phase/trace.jsonl"):
            path = folder / name
            require(sha256_file(path) == receipt["files"][name], "source bytes changed")
            inputs[path.relative_to(ROOT).as_posix()] = receipt["files"][name]
        if stratified:
            scan_jobs.append(dict(root=str(ROOT), item=item, map_id=cases[item["task_id"]]["map_id"], config=cfg,
                initial=(folder/"initial.json").relative_to(ROOT).as_posix(), initial_sha=receipt["files"]["initial.json"],
                trace=(folder/"first_phase/trace.jsonl").relative_to(ROOT).as_posix(), trace_sha=receipt["files"]["first_phase/trace.jsonl"]))
            continue
        with (folder / "first_phase/trace.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                e = json.loads(line)
                d = e["decision"]
                if d in cfg["decisions"]:
                    available.append(dict(item=item, map_id=cases[item["task_id"]]["map_id"], decision=d,
                                          id=f"{item['job_id']}-d{d:04d}"))
                if d >= max(cfg["decisions"]):
                    break
    chosen = []
    coverage = []
    for map_id in sorted({c["map_id"] for c in reg["cases"]}):
        for decision in cfg.get("decisions", []):
            pool = [t for t in available if t["map_id"] == map_id and t["decision"] == decision]
            selected = sorted(pool, key=lambda t: json_fingerprint([cfg["seed"], t["id"]]))[:cfg["states_per_map_per_decision"]]
            chosen.extend(selected)
            coverage.append(dict(map_id=map_id, decision=decision, available=len(pool), selected=len(selected)))
    admission = {}
    if stratified:
        from experiments.sa_history_sampling import build_history_sample
        chosen, coverage, admission = build_history_sample(scan_jobs, cfg, {c["map_id"] for c in reg["cases"]})
    require(0 < len(chosen) <= cfg["max_states"], "state budget")
    # Inventory all retained reports; the prose review distinguishes deep reading from inventory.
    inventory = []
    for p in sorted((ROOT / "docs").rglob("*.md")):
        content = p.read_text(encoding="utf-8-sig")
        inventory.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=sha256_file(p),
                              title=content.splitlines()[0] if content else "",
                              failure_terms_found=any(w in content.lower() for w in ("failed", "not promoted", "no_go", "未通过", "停止"))))
    write_json(out / "history_inventory.json", inventory)
    names = [Path(config_path or CONFIG).resolve().relative_to(ROOT).as_posix(), "experiments/sa_history_selector.py",
             "scripts/train_sa_history_selector.py", "tests/evaluation/test_sa_history_selector.py",
             "docs/SA_HISTORY_SELECTOR_PILOT_ZH.md"]
    names += cfg.get("extra_registered", [])
    if cfg.get("protocol"):
        names.append(cfg["protocol"])
    names += [(source / n).relative_to(ROOT).as_posix() for n in ("registration.json", "manifest.json")]
    # Pin current dependencies independently; original-action replay verifies historical equivalence.
    for directory in ("experiments", "lns2_selector", "scripts"):
        names += [p.relative_to(ROOT).as_posix() for p in (ROOT / directory).rglob("*.py")]
    names += [reg["native_file"]]
    for name, digest in reg["inputs"].items():
        if name.startswith("artifacts/"):
            require(sha256_file(ROOT / name) == digest, "frozen scorer changed: " + name)
            inputs[name] = digest
    for c in reg["cases"]:
        names += list(c["files"].values())
    for name in names:
        inputs[name] = sha256_file(ROOT / name)
    require(inputs[reg["native_file"]] == reg["native_sha256"], "native changed")
    inputs[(out / "history_inventory.json").relative_to(ROOT).as_posix()] = sha256_file(out / "history_inventory.json")
    plan = dict(schema=cfg["schema"], config=cfg, targets=chosen, coverage=coverage, inputs=inputs,
                source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                development_only=True, no_ttf=True, native_file=reg["native_file"], native_sha256=reg["native_sha256"],
                historical_reports_inventoried=len(inventory), original_control_required=True,
                sampling_admission=admission)
    plan["binding"] = json_fingerprint(plan)
    write_json(out / "plan.json", plan)
    return dict(states=len(chosen), maps=len({t["map_id"] for t in chosen}),
                max_new_trials=len(chosen)*cfg["max_candidates"]*cfg["trials"],
                original_controls=len(chosen), prefix_steps=sum(t["decision"] for t in chosen), workers=cfg["workers"],
                sampling_admission=admission, coverage=coverage if stratified else None)


def verify(config_path=None):
    cfg, out, _ = locations(config_path)
    plan = read_json(out / "plan.json")
    require(plan["binding"] == json_fingerprint({k:v for k,v in plan.items() if k != "binding"}), "plan changed")
    require(plan["config"] == cfg, "config changed")
    for name, digest in plan["inputs"].items():
        require(sha256_file(ROOT / name) == digest, "input changed: " + name)
    return plan


def die_with_parent(parent_pid):
    """Kernel-enforced cleanup also covers death while blocked inside native code."""
    import ctypes
    require(sys.platform == "linux", "parent-death guard requires Linux")
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    libc = ctypes.CDLL(None, use_errno=True)
    require(libc.prctl(1, signal.SIGKILL, 0, 0, 0) == 0, "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGKILL)


def validate_receipt(folder, plan, target):
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    receipt = read_json(folder / "receipt.json")
    require(receipt["binding"] == plan["binding"], "receipt binding")
    root = read_json(folder / "root.json")
    require(root["binding"] == plan["binding"] and root["source"] == target, "root identity")
    candidates = root["candidates"]
    require(0 < len(candidates) <= plan["config"]["max_candidates"], "candidate budget")
    require(len(candidates) == len(root["feature_rows"]), "feature count")
    require(len({c["candidate_id"] for c in candidates}) == len(candidates), "duplicate candidate id")
    require(root["old_selected_id"] in {c["candidate_id"] for c in candidates}, "missing frozen action")
    expected = {"root.json", "control.json"} | {
        f"{c['candidate_id']}-t{i}.json" for c in candidates for i in range(plan["config"]["trials"])}
    require(set(receipt["files"]) == expected, "incomplete receipt file set")
    require({p.name for p in folder.glob("*.json")} == expected | {"receipt.json"}, "unexpected state file")
    for name, digest in receipt["files"].items():
        require(sha256_file(folder / name) == digest, "completed state changed: " + name)
    before = root["state"]
    require(q.state_fingerprint(before) == root["state_fingerprint"], "root fingerprint")
    old = next(c for c in candidates if c["candidate_id"] == root["old_selected_id"])
    branches = [("control.json", old, -1)] + [
        (f"{c['candidate_id']}-t{i}.json", c, i) for c in candidates for i in range(plan["config"]["trials"])]
    for name, candidate, trial in branches:
        row = read_json(folder / name)
        require((row["status"], row["binding"], row["target_id"], row["candidate_id"], row["trial"]) ==
                ("ok", plan["binding"], target["id"], candidate["candidate_id"], trial), "branch identity")
        require(row["members"] == candidate["agents"] and sorted(row["metrics"]["neighborhood"]) == sorted(row["members"]), "branch members")
        after = q.apply_state_delta(before, row["delta"])
        require(q.state_fingerprint(after) == row["final_fingerprint"], "branch fingerprint")
        require(targets(before, after, root["edge_weights"], row["metrics"]) == row["target"], "target changed")
        if trial == -1:
            event = root["control_event"]
            check_attempt(row["metrics"], event["metrics"])
            require(row["final_fingerprint"] == q.state_fingerprint(q.apply_state_delta(before, event["delta"])), "original control changed")
    return root


def save_branch(env, state, event, target, candidate, trial, plan, path, weights, parent_pid):
    die_with_parent(parent_pid)
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from scripts.run_feedback_exploration_diagnostics import validate_final
    try:
        require(q.state_fingerprint(env.get_state()) == q.state_fingerprint(state), "fork root changed")
        pp_seed = int(json_fingerprint([plan["config"]["seed"], target["id"], trial, "pp"])[:7], 16)
        draw_seed = int(json_fingerprint([plan["config"]["seed"], target["id"], trial, "accept"])[:7], 16)
        action = event["action"] if trial == -1 else dict(mode="explicit_neighborhood", agents=candidate["agents"], random_seed=pp_seed)
        pp_seed = action.get("random_seed")
        uniform = event["uniform"] if trial == -1 else random.Random(draw_seed).random()
        raw = q._plain(env.step_experimental_pp(action, plan["config"]["pp_seconds"], "annealed", event["temperature"], uniform))
        after, metrics = raw["observation"], raw["metrics"]
        require(metrics["action_valid"] and metrics["step_applied"], "invalid repair")
        q.validate_transition(state, after, metrics, candidate["agents"], "annealed", event["temperature"], uniform)
        validate_final(after)
        if trial == -1:
            check_attempt(metrics, event["metrics"])
            require(q.state_fingerprint(after) == q.state_fingerprint(q.apply_state_delta(state, event["delta"])), "original action mismatch")
        row = dict(status="ok", binding=plan["binding"], target_id=target["id"], candidate_id=candidate["candidate_id"],
                   trial=trial, members=candidate["agents"], pp_seed=pp_seed, uniform=uniform,
                   requested_action=action,
                   target=targets(state, after, weights, metrics), metrics=metrics,
                   delta=q.encode_state_delta(state, after), final_fingerprint=q.state_fingerprint(after))
    except Exception as exc:
        row = dict(status="error", binding=plan["binding"], error=repr(exc))
    write_json(path, row)


def root_worker(job):
    die_with_parent(job["parent_pid"])
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    plan, t = job["plan"], job["target"]
    cfg = plan["config"]
    out, source = ROOT/cfg["output"], ROOT/cfg["source"]
    folder = out / "states" / t["id"]
    receipt_path = folder / "receipt.json"
    if receipt_path.exists():
        validate_receipt(folder, plan, t)
        return dict(status="ok", id=t["id"], resumed=True)
    require(not folder.exists(), "partial state needs explicit audit; no automatic retry")
    folder.mkdir(parents=True)
    require(q.native_identity()["sha256"] == plan["native_sha256"], "wrong native")
    reg = read_json(source / "registration.json")
    case = next(c for c in reg["cases"] if c["task_id"] == t["item"]["task_id"])
    worker = q.worker_job(case, t["item"], reg["template"], folder / "unused", plan["binding"])
    original = source / "episodes" / t["item"]["job_id"]
    envelope = read_json(original / "initial.json")
    expected = q.execution.read_artifact(original / "initial.json", envelope["binding"])["observation"]
    env = q._make_environment(worker["dataset_root"], worker["row"], dict(worker["environment"], time_limit=100000.0), "Adaptive")
    state = q._plain(env.reset(seed=worker["solver_seed"]))
    require(q.state_fingerprint(state) == q.state_fingerprint(expected), "initial mismatch")
    history = History(state, cfg["history_window"])
    with (original / "first_phase/trace.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["decision"] == t["decision"]:
                break
            raw = q._plain(env.step_experimental_pp(event["action"], cfg["replay_pp_seconds"], "annealed", event["temperature"], event["uniform"]))
            after = raw["observation"]
            expected = q.apply_state_delta(expected, event["delta"])
            require(q.state_fingerprint(after) == q.state_fingerprint(expected), "prefix mismatch")
            check_attempt(raw["metrics"], event["metrics"])
            history.observe(state, event, after)
            state = after
    require(event["decision"] == t["decision"], "missing occurrence")
    source_case = dict(case_id=worker["sa_case_id"], task_id=case["task_id"], solver_seed=worker["solver_seed"], proposal=worker["sa_proposal"])
    index, pool = q.SingleFullCheckPool(source_case).select(env, state, t["decision"])
    require((index, pool) == (event["selected_index"], event["pool"]), "candidate replay mismatch")
    sampling_features = None
    if cfg.get("sampling"):
        from experiments.sa_history_sampling import candidate_history, choose_history_candidates, classify
        require(classify(history, state, pool, cfg) == t["stratum"], "sampling stratum changed")
        sampling_features = candidate_history(history, state, pool, event["temperature"])
        indices = choose_history_candidates(pool, index, [cfg["seed"], t["id"]], sampling_features, cfg["max_candidates"])
        preflight = t["preflight"]
        require(preflight["state_fingerprint"] == q.state_fingerprint(state), "preflight state mismatch")
        require(preflight["pool_sha"] == json_fingerprint(pool) and preflight["selected_index"] == index, "preflight pool mismatch")
        require(preflight["candidate_ids"] == [pool[i]["candidate_id"] for i in indices], "sampled candidates changed")
        require(preflight["history_features"] == [sampling_features[i] for i in indices], "causal history features changed")
    else:
        indices = choose_candidates(pool, index, [cfg["seed"], t["id"]], cfg["max_candidates"])
    candidates = [pool[i] for i in indices]
    engine = OnlineFeatureEngine(state, backend="native")
    feature_rows, _ = engine.realized_rows(candidates, state_hash=q.state_fingerprint(state))
    feature_rows = [r["features"]["realized_dynamic"] | {"source.score": c["score"]} |
                    history.features(state, c, event["temperature"]) for r, c in zip(feature_rows, candidates)]
    validate_final(state)
    root = dict(binding=plan["binding"], state=state, source=t, candidates=candidates, feature_rows=feature_rows,
                control_event=event,
                old_selected_id=pool[index]["candidate_id"], edge_weights=history.edge_weights(),
                state_fingerprint=q.state_fingerprint(state))
    write_json(folder / "root.json", root)
    require(sys.platform == "linux" and len(list(Path("/proc/self/task").iterdir())) == 1, "fork parent must be single-threaded Linux")
    ctx = multiprocessing.get_context("fork")
    pending = [(pool[index], -1)] + [(c, trial) for c in candidates for trial in range(cfg["trials"])]
    active = []
    try:
        while pending or active:
            while pending and len(active) < cfg["workers"]:
                c, trial = pending.pop(0)
                path = folder / ("control.json" if trial == -1 else f"{c['candidate_id']}-t{trial}.json")
                process = ctx.Process(target=save_branch, args=(env, state, event, t, c, trial, plan, path, root["edge_weights"], os.getpid()))
                process.start()
                active.append((process, time.monotonic(), path))
            for entry in list(active):
                process, started, path = entry
                if process.is_alive() and time.monotonic() - started < cfg["branch_fuse_seconds"]:
                    continue
                if process.is_alive():
                    process.terminate()
                    process.join()
                    raise TimeoutError("branch fuse; unknown outcome, do not label failure")
                process.join()
                require(process.exitcode == 0 and path.exists(), "branch missing")
                require(read_json(path)["status"] == "ok", "branch error: " + str(path))
                active.remove(entry)
            time.sleep(.05)
    finally:
        for process, _, _ in active:
            if process.is_alive():
                process.terminate()
            process.join()
    require(q.state_fingerprint(env.get_state()) == root["state_fingerprint"], "fork changed parent")
    files = {p.name: sha256_file(p) for p in folder.glob("*.json")}
    write_json(receipt_path, dict(binding=plan["binding"], files=files))
    validate_receipt(folder, plan, t)
    print("STATE", t["id"], "complete", len(candidates)*cfg["trials"], "labels", flush=True)
    return dict(status="ok", id=t["id"], trials=len(candidates)*cfg["trials"])


def collect(resume=False, limit=None, config_path=None):
    from scripts import run_sa_path_quality as q
    plan = verify(config_path)
    require(all(plan.get("sampling_admission", {}).values()), "sampling coverage failed; no new labels")
    _, out, _ = locations(config_path)
    require(resume or not (out / "states").exists(), "existing collection requires resume")
    jobs = [dict(job_id=t["id"], target=t, plan=plan, parent_pid=os.getpid()) for t in plan["targets"][:limit]]
    with q._CollectionRunLock(out, plan["binding"], "history-counterfactual"):
        def failure(job, status, error):
            return dict(status=status, id=job["job_id"], error=str(error))
        rows = q._run_jobs(root_worker, jobs, workers=1, phase="history-roots", output_root=out / "progress",
                           run_fingerprint=plan["binding"], timeout_seconds=600, failure_result=failure,
                           stop_on_failure=True, on_result=lambda row: print(json.dumps(row), flush=True))
    require(all(r["status"] == "ok" for r in rows), "collection error; inspect before resume")
    return dict(states=len(rows), full_scope=len(rows) == len(plan["targets"]), no_ttf=True)


def index_rows(plan):
    out = ROOT / plan["config"]["output"]
    data, excluded = [], []
    for t in plan["targets"]:
        folder = out / "states" / t["id"]
        root = validate_receipt(folder, plan, t)
        state_rows = []
        for candidate, features in zip(root["candidates"], root["feature_rows"]):
            trials = [read_json(folder / f"{candidate['candidate_id']}-t{i}.json") for i in range(plan["config"]["trials"])]
            require(all(r["binding"] == plan["binding"] and r["target_id"] == t["id"] and r["candidate_id"] == candidate["candidate_id"] for r in trials), "trial identity")
            labels = aggregate_trials(trials, plan["config"]["trials"])
            if labels is None:
                excluded.append(dict(state=t["id"], map_id=t["map_id"], decision=t["decision"], reason="incomplete_trial"))
                state_rows = []
                break
            state_rows.append(dict(state_id=t["id"], map_id=t["map_id"], decision=t["decision"],
                                   stratum=t.get("stratum", "fixed_decision"),
                                   candidate_id=candidate["candidate_id"], size=len(candidate["agents"]),
                                   initial_conflicts=root["state"]["num_of_colliding_pairs"], features=features,
                                   labels=labels, old_selected=candidate["candidate_id"] == root["old_selected_id"],
                                   trial_labels=[r["target"] for r in trials]))
        data.extend(state_rows)
    return data, excluded


def train_fold(job):
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
    rows, heldout, cfg = job
    train = [r for r in rows if r["map_id"] != heldout]
    test = [r for r in rows if r["map_id"] == heldout]
    require(train and test, "empty map fold")
    require(not {r["state_id"] for r in train} & {r["state_id"] for r in test}, "map leakage")
    counts = Counter(r["state_id"] for r in train)
    weights = np.array([1 / counts[r["state_id"]] for r in train])
    weights *= len(weights) / weights.sum()
    result = []
    with threadpool_limits(limits=1):
        for profile in ("dynamic", "history"):
            names = sorted(k for k in train[0]["features"] if profile == "history" or not k.startswith("history."))
            x = np.array([[r["features"][k] for k in names] for r in train])
            xt = np.array([[r["features"][k] for k in names] for r in test])
            heads = {}
            for label in ("conflicts", "persistence", "feasible"):
                model = HistGradientBoostingRegressor(**cfg["model"])
                model.fit(x, np.array([r["labels"][label] for r in train]), sample_weight=weights)
                heads[label] = model.predict(xt)
            for i, row in enumerate(test):
                result.append(dict(state_id=row["state_id"], candidate_id=row["candidate_id"], profile=profile,
                                   prediction={k: float(v[i]) for k, v in heads.items()}))
    print("FOLD", heldout, "complete", flush=True)
    return result


def evaluate(rows, predictions, cfg):
    import numpy as np
    look = {(p["profile"], p["state_id"], p["candidate_id"]): p["prediction"] for p in predictions}
    evaluated = []
    for sid in sorted({r["state_id"] for r in rows}):
        group = sorted([r for r in rows if r["state_id"] == sid], key=lambda r: r["candidate_id"])
        labels = [r["labels"] for r in group]
        pareto = pareto_indices(labels)
        choices = {"frozen": next(i for i, r in enumerate(group) if r["old_selected"])}
        for profile in ("dynamic", "history"):
            pred = [look[profile, sid, r["candidate_id"]] for r in group]
            choices[profile] = select_prediction(pred, [r["candidate_id"] for r in group], group[0]["initial_conflicts"], cfg["quality_tolerance_pairs"])
            if profile == "history":
                choices["history_conflict_only"] = select_prediction(pred, [r["candidate_id"] for r in group], group[0]["initial_conflicts"], risk=False)
        row = dict(state_id=sid, map_id=group[0]["map_id"], decision=group[0]["decision"], choices={})
        for method, i in choices.items():
            row["choices"][method] = dict(candidate_id=group[i]["candidate_id"], size=group[i]["size"],
                **labels[i], pareto_hit=float(i in pareto), conflict_regret=labels[i]["conflicts"]-min(l["conflicts"] for l in labels))
        row["choices"]["uniform"] = {k: sum(l[k] for l in labels)/len(labels) for k in labels[0]}
        row["choices"]["uniform"].update(pareto_hit=len(pareto)/len(labels), conflict_regret=row["choices"]["uniform"]["conflicts"]-min(l["conflicts"] for l in labels))
        evaluated.append(row)
    summary = {}
    for method in evaluated[0]["choices"]:
        summary[method] = {metric: float(np.mean([r["choices"][method][metric] for r in evaluated])) for metric in ("conflicts", "conflict_regret", "persistence", "feasible", "pareto_hit")}
    maps = sorted({r["map_id"] for r in evaluated})
    rng = np.random.default_rng(cfg["seed"])
    comparisons = {}
    for base in ("dynamic", "frozen"):
        comparisons[base] = {}
        for metric in ("conflict_regret", "feasible", "pareto_hit", "persistence"):
            differences = np.array([np.mean([r["choices"]["history"][metric]-r["choices"][base][metric] for r in evaluated if r["map_id"] == m]) for m in maps])
            sizes = np.array([sum(r["map_id"] == m for r in evaluated) for m in maps])
            sampled = rng.integers(0, len(maps), size=(cfg["bootstrap_samples"], len(maps)))
            boots = (differences[sampled]*sizes[sampled]).sum(axis=1)/sizes[sampled].sum(axis=1)
            comparisons[base][metric] = dict(delta=float(np.average(differences, weights=sizes)),
                map_equal_delta=float(differences.mean()), ci95=np.quantile(boots, [.025,.975]).tolist(), map_deltas=dict(zip(maps, differences.tolist())))
    errors = {}
    for profile in ("dynamic", "history"):
        errors[profile] = {label: float(np.mean([np.mean([
            abs(look[profile, r["state_id"], r["candidate_id"]][label]-r["labels"][label])
            for r in rows if r["state_id"] == sid]) for sid in {r["state_id"] for r in rows}]))
            for label in ("conflicts", "persistence", "feasible")}
    from scipy.stats import spearmanr
    stability = []
    for sid in sorted({r["state_id"] for r in rows}):
        group = [r for r in rows if r["state_id"] == sid]
        halves = [[{k: sum(r["trial_labels"][i][k] for i in indices)/2
                    for k in ("conflicts", "persistence", "feasible")} for r in group]
                  for indices in ((0,1),(2,3))]
        a,b = [[p["conflicts"] for p in half] for half in halves]
        rho = None if len(set(a)) < 2 or len(set(b)) < 2 else float(spearmanr(a,b).statistic)
        pa,pb = [set(pareto_indices(half)) for half in halves]
        stability.append(dict(state_id=sid, conflict_spearman=rho, pareto_jaccard=len(pa&pb)/len(pa|pb)))
    size_counts = {m: dict(Counter(r["choices"][m]["size"] for r in evaluated)) for m in summary if m != "uniform"}
    return dict(summary=summary, comparisons=comparisons, state_predictions=evaluated,
                predictive_mae=errors, split_trial_stability=stability, selected_sizes=size_counts,
                estimand="equal_state_means_on_states_with_all_candidate_trials_complete")


def train(config_path=None):
    import numpy as np
    import pickle
    import sklearn
    from sklearn.ensemble import HistGradientBoostingRegressor
    from concurrent.futures import ProcessPoolExecutor
    from threadpoolctl import threadpool_limits
    plan = verify(config_path)
    require(all(plan.get("sampling_admission", {}).values()), "sampling coverage failed; no training")
    cfg, out, _ = locations(config_path)
    directory = out / "training"
    require(not directory.exists(), "training already attempted; preserve it")
    directory.mkdir()
    write_json(directory / "status.json", dict(status="running", binding=plan["binding"]))
    try:
        data, excluded = index_rows(plan)
        require(len({r["map_id"] for r in data}) == 8, "training requires all eight maps")
        require(len({r["state_id"] for r in data}) >= 24, "too few complete states")
        write_jsonl(directory / "index.jsonl", data)
        maps = sorted({r["map_id"] for r in data})
        with ProcessPoolExecutor(max_workers=min(8, cfg["workers"])) as pool:
            folds = list(pool.map(train_fold, [(data, m, cfg) for m in maps]))
        predictions = [p for fold in folds for p in fold]
        write_jsonl(directory / "oof_predictions.jsonl", predictions)
        report = evaluate(data, predictions, cfg)
        if cfg.get("sampling"):
            report["by_stratum"] = {s:evaluate([r for r in data if r["stratum"] == s], predictions, cfg)
                                    for s in sorted({r["stratum"] for r in data})}
        names = sorted(data[0]["features"])
        require(all(set(r["features"]) == set(names) for r in data), "feature schema drift")
        counts = Counter(r["state_id"] for r in data)
        weights = np.array([1/counts[r["state_id"]] for r in data]); weights *= len(weights)/weights.sum()
        bundles = {}
        with threadpool_limits(limits=1):
            for profile in ("dynamic", "history"):
                features = [n for n in names if profile == "history" or not n.startswith("history.")]
                x = np.array([[r["features"][n] for n in features] for r in data])
                models = {}
                for label in ("conflicts", "persistence", "feasible"):
                    model = HistGradientBoostingRegressor(**cfg["model"])
                    model.fit(x, np.array([r["labels"][label] for r in data]), sample_weight=weights)
                    models[label] = model
                bundle = dict(schema="lns2.sa_history_transition_model.v1", diagnostic_only=True, runtime_allowed=False,
                              profile=profile, feature_names=features, models=models, config=cfg,
                              data_sha256=sha256_file(directory / "index.jsonl"), binding=plan["binding"], sklearn_version=sklearn.__version__)
                path = directory / f"{profile}.pkl"
                with path.open("xb") as f:
                    pickle.dump(bundle, f)
                bundles[profile] = dict(path=path.relative_to(ROOT).as_posix(), sha256=sha256_file(path))
        diagnostic = {n:dict(min=min(r["features"][n] for r in data), max=max(r["features"][n] for r in data)) for n in names}
        report.update(schema="lns2.sa_history_pilot_result.v1", binding=plan["binding"], states=len(counts), candidates=len(data),
                      scheduled_states=len(plan["targets"]), excluded_state_count=len(excluded),
                      maps=maps, excluded=excluded, bundles=bundles, features=diagnostic, sklearn_version=sklearn.__version__,
                      decision="development_pilot_only_no_runtime_promotion", frozen_controller_changed=False, no_ttf=True,
                      target_boundary="accepted_one_step_transition_not_long_term_completion", data_sha256=sha256_file(directory/"index.jsonl"))
        write_json(directory / "report.json", report)
        write_json(directory / "status.json", dict(status="complete", binding=plan["binding"], report_sha256=sha256_file(directory/"report.json")))
        return {k:report[k] for k in ("states", "candidates", "summary", "decision")}
    except BaseException:
        write_json(directory / "status.json", dict(status="failed_or_interrupted", binding=plan["binding"]))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "collect", "train"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("positive limit required")
    result = collect(args.resume, args.limit, args.config) if args.phase == "collect" else globals()[args.phase](args.config)
    print(json.dumps(result if args.phase != "verify" else {"verified": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
