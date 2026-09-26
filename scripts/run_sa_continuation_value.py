"""Isolate continuation policy at fixed candidate/state/trial; no training/TTF."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, json_fingerprint, verify_registered_plan
from experiments.sa_paired_completion import require
from experiments.sa_continuation_value import select_roots, state_summary, summarize
from experiments.sa_paired_closed_loop import portable_model, choose, subset
from experiments.sa_history_information import profile_features
from scripts import run_sa_paired_closed_loop as loop
from scripts import run_sa_unbalanced_coverage as coverage
from scripts import run_sa_paired_completion_pilot as pilot
from scripts import collect_sa_history_candidate_bridge as bridge
from scripts.audit_sa_history_information import atomic

CONFIG = ROOT / "configs/sa_continuation_value.json"


def record(path, payload):
    loop.once(path, loop.sealed(payload))


def read_record(path, plan):
    result = loop.check_seal(read_json(path))
    require(result["binding"] == plan["binding"], "branch binding")
    return result


def prepare():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    require(not out.exists(), "existing output; no overwrite")
    source, source_out = coverage.verify()
    model_source, model_out = loop.verify()
    require(source_out == ROOT/cfg["source"] and model_out == ROOT/cfg["model_source"], "source directory mismatch")
    require(sha256_file(model_out/"model/bundle.json") == cfg["bundle_sha256"], "model identity")
    payload = loop.model_files(model_out, model_source)
    meta = read_json(model_out/"model/metadata.json")
    model = portable_model(payload, meta["feature_names"], native=False)
    files = dict(source["inputs"])
    for name, digest in model_source["inputs"].items():
        require(name not in files or files[name] == digest, "inconsistent frozen dependencies")
        files[name] = digest
    additions = [CONFIG, Path(__file__), ROOT/"experiments/sa_continuation_value.py",
        ROOT/"tests/evaluation/test_sa_continuation_value.py", ROOT/"docs/SA_CONTINUATION_VALUE_PROTOCOL_ZH.md",
        source_out/"plan.json", model_out/"plan.json", source_out/"training_index.json"]
    additions += list((model_out/"model").glob("*"))
    train_rows = []
    old_index = ROOT/model_source["config"]["training_index"]
    training_episodes = {s["episode"] for s in read_json(old_index)["states"]}
    for sid in meta["train_state_ids"]:
        path = old_index.parent/"roots"/sid/"root.json"
        root = read_json(path)
        additions.append(path)
        train_rows.append(dict(decision=root["source"]["decision"], agents=len(root["state"]["agents"]),
                               conflicts=root["state"]["num_of_colliding_pairs"]))
    ranges = {k:[min(r[k] for r in train_rows), max(r[k] for r in train_rows)] for k in cfg["support_ranges"]}
    require(ranges == cfg["support_ranges"], "training support ranges changed")
    rows, roots = [], {}
    for target in source["roots"]:
        path = source_out/"roots"/target["id"]/"root.json"
        root = read_json(path)
        require(root["source"] == target, "root source mismatch")
        roots[target["id"]] = root
        additions.append(path)
        rows.append(dict(target, agents=len(root["state"]["agents"]), conflicts=root["state"]["num_of_colliding_pairs"]))
    selected = select_roots(rows, ranges, cfg["selection_seed"], set(meta["train_state_ids"]))
    entries = []
    for sid in selected:
        target = next(t for t in source["roots"] if t["id"] == sid)
        root = roots[sid]
        require(root["source"]["item"]["job_id"] not in training_episodes, "training episode overlap")
        pilot.verify_receipt(source_out/"roots"/sid, target, source)
        additions += list((source_out/"roots"/sid).glob("*.json"))
        features = [profile_features(dict(base=f), "dynamic") for f in root["feature_rows"]]
        prediction = choose("paired", root["candidates"], root["old_selected_id"], model, features,
                            [a["id"] for a in root["state"]["agents"]], sid, cfg["selection_seed"])
        entries.append(dict(target=target, prediction=prediction))
    require(len(entries) == cfg["roots"] and all(len(e["target"]["selected"]) == cfg["candidates"] for e in entries), "cohort size changed")
    require((source["config"]["trials"], source["config"]["horizon"]) == (cfg["trials"], cfg["horizon"]), "source trial/horizon mismatch")
    for path in additions:
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    plan = dict(config=cfg, source=source, model_source=model_source, files=files, entries=entries,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                no_ttf=True, no_training=True, production_changed=False)
    plan["binding"] = json_fingerprint(plan)
    loop.once(out/"plan.json", plan)
    return dict(binding=plan["binding"], roots=8, new_candidate_branches=256, control_replays=8,
        maximum_branch_repairs=264*32, prefix_repairs=sum(e["target"]["decision"] for e in entries),
        workers=cfg["workers"], root_safety_upper_seconds=8*source["config"]["root_fuse_seconds"],
        ideal_branch_fuse_batches_seconds=8*(1+2)*source["config"]["branch_fuse_seconds"],
        selected=[dict(id=e["target"]["id"], map_id=e["target"]["map_id"], phase=e["target"]["phase"]) for e in entries])


def verify():
    return verify_registered_plan(ROOT, CONFIG)


def replay_root(plan, entry):
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from experiments.sa_history_selector import History
    source, target = plan["source"], entry["target"]
    cfg = source["config"]
    require(q.native_identity()["sha256"] == source["native_sha256"], "native identity")
    root = read_json(ROOT/plan["config"]["source"]/"roots"/target["id"]/"root.json")
    reg = read_json(ROOT/cfg["source"]/"registration.json")
    case = next(c for c in reg["cases"] if c["task_id"] == target["item"]["task_id"])
    worker = q.worker_job(case, target["item"], reg["template"], ROOT/plan["config"]["output"]/"unused", plan["binding"])
    original = ROOT/cfg["source"]/"episodes"/target["item"]["job_id"]
    expected = read_json(original/"initial.json")["payload"]["observation"]
    env = q._make_environment(worker["dataset_root"], worker["row"], dict(worker["environment"], time_limit=100000.), "Adaptive")
    state = q._plain(env.reset(seed=worker["solver_seed"]))
    require(q.state_fingerprint(state) == q.state_fingerprint(expected), "initial mismatch")
    history = History(state)
    with (original/"first_phase/trace.jsonl").open(encoding="utf8") as stream:
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
    require(history.decision == target["decision"] and pilot.history_fingerprint(history) == target["history_fingerprint"], "history mismatch")
    require(q.state_fingerprint(state) == target["state_fingerprint"] == q.state_fingerprint(root["state"]), "root mismatch")
    case = dict(case_id=worker["sa_case_id"], task_id=case["task_id"], solver_seed=worker["solver_seed"], proposal=worker["sa_proposal"])
    index, pool = q.SingleFullCheckPool(case).select(env, state, target["decision"])
    require((index, pool) == (root["control_event"]["selected_index"], root["control_event"]["pool"]), "root pool mismatch")
    require(q.state_fingerprint(env.get_state()) == target["state_fingerprint"], "proposal mutated root")
    return env, root, case, history


def branch(env, root, case, history, entry, cid, trial, arm, plan, path, parent_pid):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    die_with_parent(parent_pid)
    cfg, target = plan["source"]["config"], entry["target"]
    try:
        state = root["state"]
        require(q.state_fingerprint(env.get_state()) == target["state_fingerprint"], "fork root changed")
        original = read_json(ROOT/plan["config"]["source"]/"roots"/target["id"]/f"{cid}-t{trial}.json")
        selector = q.SingleFullCheckPool(case)
        engine = model = None
        if arm == "paired":
            folder = ROOT/plan["config"]["model_source"]
            model = portable_model(loop.model_files(folder, plan["model_source"]), read_json(folder/"model/metadata.json")["feature_names"])
            require(model.estimator.model.inference_backend == "native-portable-tree", "native inference required")
            engine = OnlineFeatureEngine(state, backend="native")
        started = time.monotonic()
        initial_nodes = state["low_level"]["generated"]
        events = []
        stop = "horizon"
        for offset in range(cfg["horizon"]):
            if state["feasible"]:
                stop = "feasible"; break
            if state["low_level"]["generated"]-initial_nodes >= cfg["node_budget"]:
                stop = "node_budget"; break
            if time.monotonic()-started >= cfg["branch_seconds"]:
                stop = "wall_safety"; break
            d = target["decision"]+offset
            fp = q.state_fingerprint(state)
            ranking = features = ids = None
            if offset == 0:
                pool = root["control_event"]["pool"]
                index = next(i for i,c in enumerate(pool) if c["candidate_id"] == cid)
            else:
                index, pool = selector.select(env, state, d)
                require(q.state_fingerprint(env.get_state()) == fp, "proposal changed state")
                if arm == "paired":
                    anchor = pool[index]["candidate_id"]
                    key = f"{case['task_id']}-s{case['solver_seed']}-d{d:04d}-{fp}"
                    mc = plan["model_source"]["config"]
                    ids = subset(pool, anchor, key, mc["candidate_seed"], mc["candidate_limit"])
                    candidates = [next(c for c in pool if c["candidate_id"] == i) for i in ids]
                    features = loop.features_for(state, candidates, engine, history, q.temperature(d), fp)
                    ranking = choose("paired", candidates, anchor, model, features, [a["id"] for a in state["agents"]], key, mc["candidate_seed"])
                    index = next(i for i,c in enumerate(pool) if c["candidate_id"] == ranking["selected"])
            pp, draw = bridge.randomization(target["id"], trial, d, cfg)
            temp = q.temperature(d)
            action = dict(mode="explicit_neighborhood", agents=pool[index]["agents"], random_seed=pp)
            remaining = cfg["branch_seconds"]-(time.monotonic()-started)
            if remaining <= 0:
                stop = "wall_safety"; break
            raw = q._plain(env.step_experimental_pp(action, min(cfg["pp_seconds"], remaining), "annealed", temp, draw))
            after, metrics = raw["observation"], raw["metrics"]
            q.validate_transition(state, after, metrics, action["agents"], "annealed", temp, draw)
            event = dict(decision=d, before=fp, action=action, temperature=temp, uniform=draw, metrics=metrics,
                         pool=pool, selected_index=index, ranking=ranking, features=features, subset=ids,
                         delta=q.encode_state_delta(state, after))
            incomplete = metrics["pp_failure_reason"] == "time_limit" or not metrics["acceptance_evaluated"]
            if not incomplete and (offset == 0 or arm == "frozen"):
                require(offset < len(original["events"]), "old comparison is incomplete")
                reference = original["events"][offset]
                require(action == reference["action"] and pool == reference["pool"], "paired action/pool mismatch")
                check_attempt(metrics, reference["metrics"])
                require(q.state_fingerprint(after) == q.state_fingerprint(q.apply_state_delta(state, reference["delta"])), "paired successor mismatch")
            events.append(event)
            history.observe(state, event, after)
            state = after
            if incomplete:
                stop = "incomplete_pp"; break
        if state["feasible"]:
            stop = "feasible"
        validate_final(state)
        row = dict(binding=plan["binding"], status="ok", root_id=target["id"], candidate_id=cid, trial=trial, arm=arm,
            root_fingerprint=target["state_fingerprint"], events=events, stop=stop,
            final_fingerprint=q.state_fingerprint(state), final_conflicts=state["num_of_colliding_pairs"],
            generated=state["low_level"]["generated"]-initial_nodes, diagnostic_seconds=time.monotonic()-started, no_ttf=True)
    except Exception as exc:
        row = dict(binding=plan["binding"], status="error", root_id=target["id"], candidate_id=cid, trial=trial, arm=arm, error=repr(exc))
    record(path, row)


def branch_jobs(entry):
    target = entry["target"]
    return [("control.json", target["anchor_id"], 0, "frozen")] + [
        (f"{cid}-t{trial}.json", cid, trial, "paired") for cid in target["selected"] for trial in range(8)]


def check_receipt(folder, entry, plan):
    r = read_json(folder/"receipt.json")
    require(r["binding"] == plan["binding"] and set(r["files"]) == {j[0] for j in branch_jobs(entry)}, "receipt coverage")
    for name, digest in r["files"].items():
        require(sha256_file(folder/name) == digest, "branch file changed")
        row = read_record(folder/name, plan)
        require(row["root_id"] == entry["target"]["id"] and row["status"] in {"ok", "censored"}, "branch identity/status")


def root_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts import run_sa_path_quality as q
    die_with_parent(job["parent_pid"])
    plan, entry = job["plan"], job["entry"]
    folder = ROOT/plan["config"]["output"]/"roots"/entry["target"]["id"]
    if (folder/"receipt.json").exists():
        check_receipt(folder, entry, plan)
        return dict(status="ok", job_id=job["job_id"], resumed=True)
    env, root, case, history = replay_root(plan, entry)
    require(len(list(Path("/proc/self/task").iterdir())) == 1, "fork requires single-threaded parent")
    folder.mkdir(parents=True, exist_ok=True)
    journal = folder/"started.json"
    launched = read_json(journal) if journal.exists() else []
    schedule = branch_jobs(entry)
    for name in launched:
        require((folder/name).exists(), "interrupted branch requires audit; no automatic retry")
    require(set(launched) <= {j[0] for j in schedule}, "unknown scheduled branch")
    context = multiprocessing.get_context("fork")
    active = []
    try:
        for group in ([schedule[0]], schedule[1:]):
            pending = []
            for spec in group:
                if (folder/spec[0]).exists():
                    row = read_record(folder/spec[0], plan)
                    require(row["status"] in {"ok", "censored"}, "previous branch error; inspect before resume")
                else:
                    pending.append(spec)
            while pending or active:
                while pending and len(active) < plan["config"]["workers"]:
                    name, cid, trial, arm = pending.pop(0)
                    launched.append(name)
                    atomic(journal, launched)
                    path = folder/name
                    p = context.Process(target=branch, args=(env, root, case, history, entry, cid, trial, arm, plan, path, os.getpid()))
                    p.start()
                    active.append((p, time.monotonic(), path, cid, trial, arm))
                for item in list(active):
                    p, started, path, cid, trial, arm = item
                    if p.is_alive() and time.monotonic()-started < plan["source"]["config"]["branch_fuse_seconds"]:
                        continue
                    if p.is_alive():
                        p.terminate(); p.join(5)
                        if p.is_alive():
                            p.kill(); p.join()
                        if not path.exists():
                            record(path, dict(binding=plan["binding"], status="censored", root_id=entry["target"]["id"],
                                candidate_id=cid, trial=trial, arm=arm, stop="external_timeout", no_ttf=True))
                    else:
                        p.join()
                        require(p.exitcode == 0 and path.exists(), "branch process crash")
                    row = read_record(path, plan)
                    require(row["status"] in {"ok", "censored"}, "branch error: " + row.get("error", "unknown"))
                    active.remove(item)
                time.sleep(.05)
            control = read_record(folder/"control.json", plan)
            original = read_json(ROOT/plan["config"]["source"]/"roots"/entry["target"]["id"]/f"{entry['target']['anchor_id']}-t0.json")
            require(control["status"] == "ok" and control["stop"] in {"feasible", "horizon"} and
                    control["final_fingerprint"] == original["final_fingerprint"] and len(control["events"]) == len(original["events"]), "old continuation control failed")
    finally:
        for p, *_ in active:
            if p.is_alive():
                p.terminate(); p.join(5)
                if p.is_alive():
                    p.kill()
            p.join()
    require(q.state_fingerprint(env.get_state()) == entry["target"]["state_fingerprint"], "child mutated parent")
    loop.once(folder/"receipt.json", dict(binding=plan["binding"], files={name:sha256_file(folder/name) for name, *_ in schedule}))
    check_receipt(folder, entry, plan)
    return dict(status="ok", job_id=job["job_id"], branches=len(schedule), prefix=entry["target"]["decision"])


def collect(resume=False, limit=None):
    from experiments.repair_collection import _run_jobs, _CollectionRunLock
    plan, out = verify()
    require(resume or not (out/"roots").exists(), "resume required")
    if limit is not None:
        require(0 < limit <= len(plan["entries"]), "invalid limit")
    with _CollectionRunLock(out, plan["binding"], "continuation-value"):
        atomic(out/"run_status.json", dict(status="running", binding=plan["binding"]))
        try:
            for entry in plan["entries"][:limit]:
                if (out/"STOP_AFTER_ROOT").exists():
                    break
                job = dict(job_id=entry["target"]["id"], entry=entry, plan=plan, parent_pid=os.getpid())
                rows = _run_jobs(root_worker, [job], 1, phase=job["job_id"], output_root=out/"progress"/job["job_id"],
                    run_fingerprint=plan["binding"], timeout_seconds=plan["source"]["config"]["root_fuse_seconds"], stop_on_failure=True)
                require(len(rows) == 1 and rows[0]["status"] == "ok", "root failed; audit before retry")
                with (out/"progress.jsonl").open("a", encoding="utf8") as f:
                    f.write(json.dumps(rows[0])+"\n")
                print("ROOT", rows[0], flush=True)
            count = sum((out/"roots"/e["target"]["id"]/"receipt.json").exists() for e in plan["entries"])
            atomic(out/"run_status.json", dict(status="completed" if count == len(plan["entries"]) else "paused", completed_roots=count, binding=plan["binding"]))
        except BaseException as exc:
            atomic(out/"run_status.json", dict(status="error", error=repr(exc), binding=plan["binding"]))
            raise
    return dict(completed_roots=count, total=len(plan["entries"]))


def audit_branch(row, root, history, case, entry, plan):
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    if row["status"] == "censored":
        require(row["stop"] == "external_timeout", "unknown censoring")
        return None
    require(row["status"] == "ok" and row["root_fingerprint"] == root["state_fingerprint"], "branch status/root")
    cfg, target = plan["source"]["config"], entry["target"]
    state = root["state"]
    initial_nodes = state["low_level"]["generated"]
    model_out = ROOT/plan["config"]["model_source"]
    model = portable_model(loop.model_files(model_out, plan["model_source"]), read_json(model_out/"model/metadata.json")["feature_names"], native=False)
    engine = OnlineFeatureEngine(state, backend="native")
    original = read_json(ROOT/plan["config"]["source"]/"roots"/target["id"]/f"{row['candidate_id']}-t{row['trial']}.json")
    for offset, event in enumerate(row["events"]):
        d = target["decision"]+offset
        fp = q.state_fingerprint(state)
        require(not state["feasible"] and event["decision"] == d and event["before"] == fp, "branch trace sequence")
        pp, draw = bridge.randomization(target["id"], row["trial"], d, cfg)
        require(event["action"]["random_seed"] == pp and event["uniform"] == draw and event["temperature"] == q.temperature(d), "branch random schedule")
        pool = event["pool"]
        if offset == 0:
            require(pool == root["control_event"]["pool"], "forced pool changed")
            expected = row["candidate_id"]
            require(event["features"] is None and event["ranking"] is None, "root is forced, not model selected")
        elif row["arm"] == "paired":
            anchor = pool[bridge.frozen_index(pool)]["candidate_id"]
            key = f"{case['task_id']}-s{case['solver_seed']}-d{d:04d}-{fp}"
            mc = plan["model_source"]["config"]
            ids = subset(pool, anchor, key, mc["candidate_seed"], mc["candidate_limit"])
            require(ids == event["subset"], "continuation subset changed")
            candidates = [next(c for c in pool if c["candidate_id"] == cid) for cid in ids]
            features = loop.features_for(state, candidates, engine, history, event["temperature"], fp)
            require(features == event["features"], "feature mismatch")
            rank = choose("paired", candidates, anchor, model, features, [a["id"] for a in state["agents"]], key, mc["candidate_seed"])
            require(rank == event["ranking"], "Python/native inference mismatch")
            expected = rank["selected"]
        else:
            expected = pool[bridge.frozen_index(pool)]["candidate_id"]
        require(pool[event["selected_index"]]["candidate_id"] == expected and event["action"]["agents"] == pool[event["selected_index"]]["agents"], "action mismatch")
        after = q.apply_state_delta(state, event["delta"])
        m = event["metrics"]
        q.validate_transition(state, after, m, event["action"]["agents"], "annealed", event["temperature"], draw)
        incomplete = m["pp_failure_reason"] == "time_limit" or not m["acceptance_evaluated"]
        if not incomplete and (offset == 0 or row["arm"] == "frozen"):
            reference = original["events"][offset]
            require(event["action"] == reference["action"] and pool == reference["pool"], "reference action/pool mismatch")
            check_attempt(m, reference["metrics"])
            require(q.state_fingerprint(after) == q.state_fingerprint(q.apply_state_delta(state, reference["delta"])), "reference successor mismatch")
        history.observe(state, event, after)
        state = after
        require(not incomplete or offset == len(row["events"])-1, "continued after incomplete PP")
    validate_final(state)
    require(len(row["events"]) <= cfg["horizon"] and q.state_fingerprint(state) == row["final_fingerprint"] and state["num_of_colliding_pairs"] == row["final_conflicts"], "final state mismatch")
    require(row["generated"] == state["low_level"]["generated"]-initial_nodes, "node delta mismatch")
    if row["stop"] == "feasible":
        require(state["feasible"], "false success")
        return 1
    require(not state["feasible"], "feasible state marked incomplete")
    if row["stop"] == "horizon":
        require(len(row["events"]) == cfg["horizon"], "short horizon")
        return 0
    require(row["stop"] in {"node_budget", "wall_safety", "incomplete_pp"}, "unknown stop")
    if row["stop"] == "node_budget":
        require(row["generated"] >= cfg["node_budget"], "false node censoring")
    if row["stop"] == "incomplete_pp":
        require(row["events"] and incomplete, "false PP censoring")
    return None


def analyze_root(job):
    from scripts import run_sa_path_quality as q
    from experiments.sa_history_selector import History
    from experiments.online_feature_engine import OnlineFeatureEngine
    plan, entry = job
    target = entry["target"]
    folder = ROOT/plan["config"]["output"]/"roots"/target["id"]
    check_receipt(folder, entry, plan)
    root = read_json(ROOT/plan["config"]["source"]/"roots"/target["id"]/"root.json")
    cfg = plan["source"]["config"]
    source_episode = ROOT/cfg["source"]/"episodes"/target["item"]["job_id"]
    state = read_json(source_episode/"initial.json")["payload"]["observation"]
    history = History(state)
    with (source_episode/"first_phase/trace.jsonl").open(encoding="utf8") as f:
        for line in f:
            event = json.loads(line)
            if event["decision"] == target["decision"]:
                break
            after = q.apply_state_delta(state, event["delta"])
            history.observe(state, event, after)
            state = after
    require(q.state_fingerprint(state) == target["state_fingerprint"] and pilot.history_fingerprint(history) == target["history_fingerprint"], "audit root/history mismatch")
    case = dict(task_id=target["item"]["task_id"], solver_seed=target["item"]["solver_seed"])
    features = loop.features_for(state, root["candidates"], OnlineFeatureEngine(state, backend="native"), history,
                                 q.temperature(target["decision"]), target["state_fingerprint"])
    require(features == [profile_features(dict(base=f), "dynamic") for f in root["feature_rows"]], "root features changed")
    model_out = ROOT/plan["config"]["model_source"]
    model = portable_model(loop.model_files(model_out, plan["model_source"]), read_json(model_out/"model/metadata.json")["feature_names"], native=False)
    prediction = choose("paired", root["candidates"], root["old_selected_id"], model, features,
                        [a["id"] for a in state["agents"]], target["id"], plan["config"]["selection_seed"])
    require(prediction == entry["prediction"], "root frozen prediction changed")
    control = read_record(folder/"control.json", plan)
    require(control["arm"] == "frozen" and control["candidate_id"] == target["anchor_id"] and control["trial"] == 0, "control identity")
    audit_branch(control, root, copy.deepcopy(history), case, entry, plan)
    values = {a:{cid:[] for cid in target["selected"]} for a in ("frozen", "paired")}
    steps = changed = 0
    for cid in target["selected"]:
        for trial in range(8):
            new = read_record(folder/f"{cid}-t{trial}.json", plan)
            require((new["root_id"], new["candidate_id"], new["trial"], new["arm"]) == (target["id"], cid, trial, "paired"), "trial identity")
            values["paired"][cid].append(audit_branch(new, root, copy.deepcopy(history), case, entry, plan))
            old = read_json(ROOT/plan["config"]["source"]/"roots"/target["id"]/f"{cid}-t{trial}.json")
            require((old["root_id"], old["candidate_id"], old["trial"]) == (target["id"], cid, trial), "source trial identity")
            bridge.branch_labels(old, root, dict(target=target, previous_best=root["previous_best"]), cfg)
            values["frozen"][cid].append(1 if old["stop"] == "feasible" else 0 if old["stop"] == "horizon" else None)
            steps += len(new.get("events", []))
            changed += sum(e["selected_index"] != bridge.frozen_index(e["pool"]) for e in new.get("events", [])[1:])
    return dict(state_id=target["id"], map_id=target["map_id"], phase=target["phase"], decision=target["decision"],
        values=values, prediction=entry["prediction"], steps=steps, changed_continuation_actions=changed,
        **state_summary(values, target["anchor_id"], entry["prediction"]["selected"]))


def analyze():
    plan, out = verify()
    require(all((out/"roots"/e["target"]["id"]/"receipt.json").exists() for e in plan["entries"]), "incomplete cohort")
    with ProcessPoolExecutor(max_workers=min(20, len(plan["entries"]))) as pool:
        rows = list(pool.map(analyze_root, [(plan, e) for e in plan["entries"]]))
    report = dict(binding=plan["binding"], states=rows,
        summary=summarize(rows, plan["config"]["selection_seed"], plan["config"]["bootstrap"]),
        no_ttf=True, no_training=True, production_changed=False,
        files={p.relative_to(out).as_posix():sha256_file(p) for p in sorted((out/"roots").rglob("*.json"))})
    verify()
    loop.once(out/"report.json", report)
    return report["summary"]


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "analyze", "verify", "request-stop"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "collect":
        result = collect(args.resume, args.limit)
    elif args.phase in {"verify", "request-stop"}:
        plan, out = verify()
        if args.phase == "request-stop":
            loop.once(out/"STOP_AFTER_ROOT", dict(binding=plan["binding"]))
        result = dict(binding=plan["binding"], inputs=len(plan["files"]), phase=args.phase)
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
