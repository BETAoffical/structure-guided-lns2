"""One developmental policy update with episode-relative work budgets."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, json_fingerprint, contained_file
from experiments.sa_paired_completion import require
from experiments.sa_paired_closed_loop import subset, choose, portable_payload, portable_model
from experiments.sa_policy_aligned_update import (
    ARMS, CENSORED, split_sources, work_stop, budget_features, aggregate, training_matrix, fit_once,
)
from scripts import run_sa_linear_closed_loop as source_loop
from scripts.run_sa_paired_closed_loop import once, sealed, check_seal, reset_case, features_for
from scripts.audit_sa_history_information import atomic

CONFIG = "configs/sa_policy_aligned_update.json"
CODE = ["experiments/sa_policy_aligned_update.py", "scripts/run_sa_policy_aligned_update.py",
        "tests/evaluation/test_sa_policy_aligned_update.py", "docs/SA_POLICY_ALIGNED_UPDATE_PROTOCOL_ZH.md"]


def prepare():
    from scripts import run_sa_path_quality as q
    cfg = read_json(ROOT/CONFIG)
    out = ROOT/cfg["output"]
    require(not out.exists(), "output exists; use verify/resume, never overwrite")
    require(cfg["maximum_updates"] == 1 and not cfg["formal_ttf"] and not cfg["automatic_promotion"], "scope changed")
    source, folder = source_loop.verify()
    require(folder == ROOT/cfg["source"], "source identity")
    require(sha256_file(folder/"report.json") == cfg["source_report_sha256"], "source report changed")
    report = check_seal(read_json(folder/"report.json"))
    cases = [source_loop.runtime_case(c) for c in source_loop.cases_verified(source, folder)]
    source_loop.model_receipt(source, folder)
    split = split_sources(report["episodes"], cfg)
    inputs = dict(source["inputs"])
    for name, digest in report["files"].items():
        path = contained_file(folder, name, field="source output")
        require(sha256_file(path) == digest, "source output changed")
        inputs[path.relative_to(ROOT).as_posix()] = digest
    for p in [CONFIG, *CODE, cfg["source"]+"/plan.json", cfg["source"]+"/report.json", cfg["source"]+"/cases.json"]:
        inputs[p] = sha256_file(ROOT/p)
    for p in (folder/"model").glob("*.json"):
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    roots = []
    for job_id in split["training_episode_ids"]:
        episode = next(e for e in report["episodes"] if e["job_id"] == job_id)
        case = next(c for c in cases if episode["pair_id"] == f"{c['task_id']}-s{episode['solver_seed']}")
        initial = read_json(folder/"episodes"/job_id/"initial.json")["state"]
        state, prefix = initial, []
        with (folder/"episodes"/job_id/"trace.jsonl").open(encoding="utf8") as stream:
            for line in stream:
                event = json.loads(line)
                d = event["decision"]
                require(event["before"] == q.state_fingerprint(state), "source trace discontinuity")
                if d in cfg["root_decisions"]:
                    generated = state["low_level"]["generated"]-initial["low_level"]["generated"]
                    require(work_stop(state["feasible"], d, generated, cfg) is None, "root outside registered work budget")
                    pool = {c["candidate_id"]: c for c in event["pool"]}
                    root = dict(id=job_id+f"-d{d:04d}", map_id=case["map_id"], case=case,
                        pair_id=episode["pair_id"], solver_seed=episode["solver_seed"], decision=d,
                        initial=initial, state=state, prefix=list(prefix), source_event=event,
                        initial_nodes=initial["low_level"]["generated"], generated_at_root=generated,
                        anchor_id=event["anchor_id"], candidates=[pool[c] for c in event["subset"]],
                        features=[budget_features(f, d, generated, cfg) for f in event["features"]])
                    roots.append(root)
                if d >= max(cfg["root_decisions"]):
                    break
                prefix.append(event)
                state = q.apply_state_delta(state, event["delta"])
    require(len(roots) == cfg["train_maps"]*len(cfg["root_decisions"]), "missing visited root; no source replacement")
    require(all(len(r["candidates"]) == 4 for r in roots), "four-candidate coverage missing")
    root_records = []
    for r in roots:
        path = out/"inputs"/(r["id"]+".json")
        once(path, sealed(r))
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        root_records.append({k:r[k] for k in ("id", "map_id", "decision", "generated_at_root", "pair_id")} |
                            dict(file=path.relative_to(ROOT).as_posix(), candidates=[c["candidate_id"] for c in r["candidates"]]))
    for name, digest in inputs.items():
        require(sha256_file(ROOT/name) == digest, "input changed during preparation")
    plan = dict(config=cfg, source_plan=source, split=split, roots=root_records, cases=cases, inputs=inputs,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), no_ttf=True,
        label_jobs=sum(len(r["candidates"])*cfg["trials"]*2 for r in roots),
        maximum_label_repairs=sum((cfg["max_decisions"]-r["decision"])*len(r["candidates"])*cfg["trials"]*2 for r in roots),
        replay_repairs=sum(r["decision"]*len(r["candidates"])*cfg["trials"]*2 for r in roots),
        validation_episodes=3*len(split["validation_pair_ids"]))
    plan["binding"] = json_fingerprint(plan)
    once(out/"plan.json", plan)
    return {k:plan[k] for k in ("binding", "label_jobs", "maximum_label_repairs", "replay_repairs", "validation_episodes")}


def verify():
    cfg = read_json(ROOT/CONFIG)
    out = ROOT/cfg["output"]
    p = read_json(out/"plan.json")
    require(p["config"] == cfg and p["binding"] == json_fingerprint({k:v for k,v in p.items() if k != "binding"}), "plan changed")
    for name, digest in p["inputs"].items():
        require(sha256_file(contained_file(ROOT, name, field="registered input")) == digest, "input changed: "+name)
    return p, out


def root_read(entry, plan):
    path = contained_file(ROOT, entry["file"], field="root")
    require(sha256_file(path) == plan["inputs"][entry["file"]], "root bytes changed")
    return check_seal(read_json(path))


def replay(root, plan):
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from experiments.sa_history_selector import History
    env, state, case = reset_case(root["case"], root["solver_seed"], plan["source_plan"])
    require(q.state_fingerprint(state) == q.state_fingerprint(root["initial"]), "initial mismatch")
    history = History(state)
    for e in root["prefix"]:
        require(history.decision == e["decision"], "prefix order")
        raw = q._plain(env.step_experimental_pp(e["action"], plan["config"]["pp_seconds"], "annealed", e["temperature"], e["uniform"]))
        after = raw["observation"]
        check_attempt(raw["metrics"], e["metrics"])
        require(q.state_fingerprint(after) == q.state_fingerprint(q.apply_state_delta(state, e["delta"])), "prefix mismatch")
        history.observe(state, e, after)
        state = after
    require(q.state_fingerprint(state) == q.state_fingerprint(root["state"]), "root mismatch")
    index, pool = q.SingleFullCheckPool(case).select(env, state, history.decision)
    e = root["source_event"]
    require(pool == e["pool"] and pool[index]["candidate_id"] == e["anchor_id"], "source candidate pool mismatch")
    return env, state, case, history


def selected_pool(env, state, history, selector, engine, case, plan):
    from scripts import run_sa_path_quality as q
    fp, d = q.state_fingerprint(state), history.decision
    index, pool = selector.select(env, state, d)
    require(q.state_fingerprint(env.get_state()) == fp, "proposal changed state")
    anchor = pool[index]["candidate_id"]
    key = f"{case['task_id']}-s{case['solver_seed']}-d{d:04d}-{fp}"
    cfg = plan["source_plan"]["config"]
    ids = subset(pool, anchor, key, cfg["candidate_seed"], cfg["candidate_limit"])
    candidates = [next(c for c in pool if c["candidate_id"] == cid) for cid in ids]
    fs = features_for(state, candidates, engine, history, q.temperature(d), fp)
    return pool, anchor, ids, candidates, fs


def record(path, row):
    once(path, sealed(row))


def check_result(folder, plan):
    row = check_seal(read_json(folder/"result.json"))
    require(row["binding"] == plan["binding"] and row["status"] in ("ok", "censored"), "invalid result")
    for name, digest in row.get("files", {}).items():
        require(sha256_file(contained_file(folder, name, field="result file")) == digest, "result file changed")
    return row


def load_updated_model(out, plan, native=True):
    manifest = check_seal(read_json(out/"model/receipt.json"))
    training_path = out/"model/training.json"
    training = check_seal(read_json(training_path))
    require(manifest["binding"] == training["binding"] == plan["binding"], "updated model identity")
    require(manifest["training_sha256"] == sha256_file(training_path), "training receipt changed")
    require(manifest["files"] == training["files"] and manifest["parity_verified"], "model parity receipt")
    path = out/"model/bundle.json"
    require(sha256_file(path) == manifest["files"]["bundle.json"], "updated bundle changed")
    # The bundle names are delta/mean columns; the ranker needs base candidate names.
    return portable_model(read_json(path), training["feature_names"], native=native)


def worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    die_with_parent(job["parent_pid"])
    p, cfg = job["plan"], job["plan"]["config"]
    folder = ROOT/cfg["output"]/job["phase"]/job["job_id"]
    if (folder/"result.json").exists():
        return dict(status="ok", job_id=job["job_id"], stop=check_result(folder, p)["stop"], resumed=True)
    require(not folder.exists(), "partial job; inspect interruption before resume")
    if job["phase"] in ("preflight", "labels"):
        root = root_read(job["root"], p)
        env, state, case, history = replay(root, p)
        initial_nodes = root["initial_nodes"]
    else:
        env, state, case = reset_case(job["case"], job["solver_seed"], p["source_plan"])
        history = History(state)
        initial_nodes = state["low_level"]["generated"]
        root = None
    start_fp = q.state_fingerprint(state)
    models = source_loop.models_load(ROOT/cfg["source"])
    old_model = models["gbdt"]
    model = old_model
    if job["arm"] == "updated":
        model = load_updated_model(ROOT/cfg["output"], p)
    require(old_model.estimator.model.inference_backend == "native-portable-tree", "native portable inference required")
    selector = q.SingleFullCheckPool(case)
    engine = OnlineFeatureEngine(state, backend="native")
    folder.mkdir(parents=True)
    record(folder/"initial.json", dict(binding=p["binding"], state=state))
    began = time.monotonic()
    first_successor, count = None, 0
    with (folder/"trace.jsonl.gz").open("xb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, filename="", mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf8") as stream:
                while True:
                    d = history.decision
                    used = state["low_level"]["generated"]-initial_nodes
                    stop = work_stop(state["feasible"], d, used, cfg)
                    if stop:
                        break
                    if time.monotonic()-began >= cfg["branch_seconds"]:
                        stop = "wall_safety"; break
                    pool, anchor, ids, candidates, fs = selected_pool(env, state, history, selector, engine, case, p)
                    supplied = [budget_features(f, d, used, cfg) for f in fs] if job["arm"] == "updated" else fs
                    ranking = choose("frozen" if job["arm"] == "frozen" else "paired", candidates, anchor,
                                     model, supplied, [a["id"] for a in state["agents"]], "unused", 0)
                    cid = job["candidate_id"] if count == 0 and job["phase"] == "labels" else ranking["selected"]
                    index = next(i for i,c in enumerate(pool) if c["candidate_id"] == cid)
                    random_key = root["id"] if root else job["pair_id"]
                    trial = job.get("trial", 0)
                    pp = q.seed(random_key, trial, d, "pp")
                    draw = q.acceptance_draw(q.seed(random_key, trial, d, "accept"))
                    temp = q.temperature(d)
                    action = dict(mode="explicit_neighborhood", agents=pool[index]["agents"], random_seed=pp)
                    if job["phase"] == "preflight":
                        ref = root["source_event"]
                        require(ids == ref["subset"] and fs == ref["features"] and ranking == ref["ranking"], "source inference mismatch")
                        action, temp, draw = ref["action"], ref["temperature"], ref["uniform"]
                        index = ref["selected_index"]
                    remaining = cfg["branch_seconds"]-(time.monotonic()-began)
                    if remaining <= 0:
                        stop = "wall_safety"; break
                    raw = q._plain(env.step_experimental_pp(action, min(cfg["pp_seconds"], remaining), "annealed", temp, draw))
                    after, m = raw["observation"], raw["metrics"]
                    q.validate_transition(state, after, m, action["agents"], "annealed", temp, draw)
                    event = dict(decision=d, before=q.state_fingerprint(state), action=action, temperature=temp, uniform=draw,
                        metrics=m, pool=pool, anchor_id=anchor, subset=ids, features=supplied, ranking=ranking,
                        selected_index=index, delta=q.encode_state_delta(state, after))
                    stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False)+"\n")
                    stream.flush()
                    if count == 0:
                        first_successor = q.state_fingerprint(after)
                    history.observe(state, event, after)
                    state, count = after, count+1
                    if job["phase"] == "preflight":
                        check_attempt(m, root["source_event"]["metrics"])
                        require(q.state_fingerprint(after) == q.state_fingerprint(q.apply_state_delta(root["state"], root["source_event"]["delta"])), "control mismatch")
                        stop = "control"; break
                    if m["pp_failure_reason"] == "time_limit" or not m["acceptance_evaluated"]:
                        stop = "incomplete_pp"; break
    validate_final(state)
    result = dict(binding=p["binding"], status="ok", job_id=job["job_id"], phase=job["phase"],
        arm=job["arm"], pair_id=job.get("pair_id"), root_id=root["id"] if root else None,
        map_id=(root["case"] if root else job["case"])["map_id"], candidate_id=job.get("candidate_id"), trial=job.get("trial"),
        stop=stop, success=state["feasible"], initial_fingerprint=start_fp, first_successor=first_successor,
        final_fingerprint=q.state_fingerprint(state), final_conflicts=state["num_of_colliding_pairs"],
        decisions=count, absolute_decision=history.decision, initial_nodes=initial_nodes,
        accumulated_generated=state["low_level"]["generated"]-initial_nodes,
        seconds_diagnostic=time.monotonic()-began, no_ttf=True,
        files={n:sha256_file(folder/n) for n in ("initial.json", "trace.jsonl.gz")})
    record(folder/"result.json", result)
    return dict(status="ok", job_id=job["job_id"], stop=stop, success=state["feasible"], decisions=count)


def failure(job, status, error):
    return dict(status=status, job_id=job["job_id"], error=error)


def jobs_for(plan, phase):
    cfg = plan["config"]
    if phase == "preflight":
        return [dict(job_id=r["id"], root=r, arm="gbdt", phase=phase) for r in plan["roots"]]
    if phase == "labels":
        return [dict(job_id=f"{r['id']}-{cid}-t{t}-{arm}", root=r, arm=arm, phase=phase, candidate_id=cid, trial=t)
                for r in plan["roots"] for cid in r["candidates"] for t in range(cfg["trials"]) for arm in cfg["continuations"]]
    require(phase == "evaluation", "unknown collection phase")
    result = []
    for key in plan["split"]["validation_pair_ids"]:
        task, seed = key.rsplit("-s", 1)
        case = next(c for c in plan["cases"] if c["task_id"] == task)
        for arm in ARMS:
            result.append(dict(job_id=key+"-"+arm, pair_id=key, solver_seed=int(seed), case=case, arm=arm, phase=phase))
    return result


def collect(phase, resume=False, limit=None):
    from experiments.repair_collection import _run_jobs, _CollectionRunLock
    plan, out = verify()
    if phase != "preflight":
        require(read_json(out/"preflight.complete.json")["binding"] == plan["binding"], "preflight required")
    if phase == "evaluation":
        receipt = check_seal(read_json(out/"model/receipt.json"))
        require(receipt["binding"] == plan["binding"] and receipt["parity_verified"], "model freeze/parity required")
    require(resume or not (out/phase).exists(), "resume required")
    jobs = jobs_for(plan, phase)
    if limit is not None:
        require(0 < limit <= len(jobs), "invalid limit")
        jobs = jobs[:limit]
    pending = []
    for job in jobs:
        if (out/phase/job["job_id"]/"result.json").exists():
            check_result(out/phase/job["job_id"], plan)
        else:
            pending.append(dict(job, plan=plan, parent_pid=os.getpid()))
    with _CollectionRunLock(out, plan["binding"], "policy-aligned-"+phase):
        atomic(out/"run_status.json", dict(phase=phase, status="running", pending=len(pending)))
        try:
            for offset in range(0, len(pending), plan["config"]["workers"]):
                if (out/"STOP_AFTER_BATCH").exists():
                    break
                batch = pending[offset:offset+plan["config"]["workers"]]
                def progress(row):
                    with (out/"progress.jsonl").open("a", encoding="utf8") as f:
                        f.write(json.dumps(dict(phase=phase, **row))+"\n")
                    print(phase, row, flush=True)
                rows = _run_jobs(worker, batch, plan["config"]["workers"], phase=f"{phase}-{offset}",
                    output_root=out/"progress"/f"{phase}-{offset}", run_fingerprint=plan["binding"],
                    timeout_seconds=plan["config"]["fuse_seconds"], on_result=progress,
                    failure_result=failure, stop_on_failure=True)
                require(len(rows) == len(batch) and all(r["status"] == "ok" for r in rows), "job error; inspect before resuming")
            all_jobs = jobs_for(plan, phase)
            done = sum((out/phase/j["job_id"]/"result.json").exists() for j in all_jobs)
            if done == len(all_jobs):
                verified = [check_result(out/phase/j["job_id"], plan) for j in all_jobs]
                if phase == "preflight":
                    require(all(r["stop"] == "control" for r in verified), "preflight control failed")
                record(out/(phase+".complete.json"), dict(binding=plan["binding"], phase=phase, jobs=done,
                    files={(Path(phase)/j["job_id"]/"result.json").as_posix():sha256_file(out/phase/j["job_id"]/"result.json") for j in all_jobs}))
            atomic(out/"run_status.json", dict(phase=phase, status="completed" if done == len(all_jobs) else "paused", done=done, total=len(all_jobs)))
        except BaseException as exc:
            atomic(out/"run_status.json", dict(phase=phase, status="error", error=repr(exc)))
            raise
    return dict(phase=phase, done=done, total=len(all_jobs))


def audit_job(job):
    from scripts import run_sa_path_quality as q
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.sa_history_selector import History
    from experiments.online_feature_engine import OnlineFeatureEngine
    p, cfg = job["plan"], job["plan"]["config"]
    folder = ROOT/cfg["output"]/job["phase"]/job["job_id"]
    result = check_result(folder, p)
    state = read_json(folder/"initial.json")["state"]
    root = root_read(job["root"], p) if job["phase"] == "labels" else None
    if root:
        history = History(root["initial"])
        current = root["initial"]
        for e in root["prefix"]:
            after = q.apply_state_delta(current, e["delta"])
            history.observe(current, e, after)
            current = after
        require(q.state_fingerprint(current) == q.state_fingerprint(state), "audit prefix/root mismatch")
    else:
        history = History(state)
    require(result["initial_fingerprint"] == q.state_fingerprint(state), "initial audit identity")
    model = source_loop.models_load(ROOT/cfg["source"])["gbdt"]
    if job["arm"] == "updated":
        model = load_updated_model(ROOT/cfg["output"], p)
    engine = OnlineFeatureEngine(state, backend="native")
    count = 0
    with gzip.open(folder/"trace.jsonl.gz", "rt", encoding="utf8") as stream:
        for line in stream:
            e = json.loads(line)
            d = history.decision
            used = state["low_level"]["generated"]-result["initial_nodes"]
            require(work_stop(state["feasible"], d, used, cfg) is None, "action after work budget")
            require(e["decision"] == d and e["before"] == q.state_fingerprint(state), "trace discontinuity")
            key = f"{(root or job)['pair_id']}-d{d:04d}-{e['before']}"
            sc = p["source_plan"]["config"]
            ids = subset(e["pool"], e["anchor_id"], key, sc["candidate_seed"], sc["candidate_limit"])
            require(ids == e["subset"], "candidate sampling mismatch")
            cs = [next(c for c in e["pool"] if c["candidate_id"] == cid) for cid in ids]
            fs = features_for(state, cs, engine, history, q.temperature(d), e["before"])
            supplied = [budget_features(f, d, used, cfg) for f in fs] if job["arm"] == "updated" else fs
            require(supplied == e["features"], "online/offline feature mismatch")
            ranking = choose("frozen" if job["arm"] == "frozen" else "paired", cs, e["anchor_id"], model,
                             supplied, [a["id"] for a in state["agents"]], "unused", 0)
            require(ranking == e["ranking"], "model score/choice mismatch")
            chosen = job["candidate_id"] if root and count == 0 else ranking["selected"]
            selected = e["pool"][e["selected_index"]]
            require(selected["candidate_id"] == chosen and e["action"]["agents"] == selected["agents"], "action mismatch")
            random_key = root["id"] if root else job["pair_id"]
            pp = q.seed(random_key, job.get("trial", 0), d, "pp")
            draw = q.acceptance_draw(q.seed(random_key, job.get("trial", 0), d, "accept"))
            require(e["action"]["random_seed"] == pp and e["uniform"] == draw and e["temperature"] == q.temperature(d), "randomization mismatch")
            after = q.apply_state_delta(state, e["delta"])
            q.validate_transition(state, after, e["metrics"], e["action"]["agents"], "annealed", e["temperature"], draw)
            if count == 0:
                require(q.state_fingerprint(after) == result["first_successor"], "first successor mismatch")
            history.observe(state, e, after)
            state, count = after, count+1
    validate_final(state)
    require(count == result["decisions"] and history.decision == result["absolute_decision"] and
            q.state_fingerprint(state) == result["final_fingerprint"] and state["feasible"] == result["success"] and
            state["num_of_colliding_pairs"] == result["final_conflicts"] and
            state["low_level"]["generated"]-result["initial_nodes"] == result["accumulated_generated"], "final result mismatch")
    if result["stop"] not in CENSORED:
        require(work_stop(state["feasible"], history.decision, result["accumulated_generated"], cfg) == result["stop"], "budget stop mismatch")
    return result


def label_audit():
    plan, out = verify()
    require((out/"labels.complete.json").exists(), "collection incomplete")
    roots = [root_read(e, plan) for e in plan["roots"]]
    jobs = jobs_for(plan, "labels")
    with ProcessPoolExecutor(max_workers=plan["config"]["workers"]) as pool:
        results = list(pool.map(audit_job, [dict(j, plan=plan) for j in jobs]))
    labels = [aggregate(r, [x for x in results if x["root_id"] == r["id"]], plan["config"]) for r in roots]
    report = dict(binding=plan["binding"], roots=labels, complete=all(r["complete"] for r in labels),
        censored=sum(r["censored"] for r in labels), no_ttf=True, model_fits=0,
        results={j["job_id"]:sha256_file(out/"labels"/j["job_id"]/"result.json") for j in jobs})
    record(out/"labels.audit.json", report)
    return dict(complete=report["complete"], censored=report["censored"], roots=len(labels))


def train():
    plan, out = verify()
    require(not (out/"model").exists(), "one update only; no refit")
    audit = check_seal(read_json(out/"labels.audit.json"))
    require(audit["binding"] == plan["binding"] and audit["complete"], "unknown labels: no silent filtering")
    for jid, digest in audit["results"].items():
        require(sha256_file(out/"labels"/jid/"result.json") == digest, "label changed")
    roots = [root_read(e, plan) for e in plan["roots"]]
    matrix = training_matrix(roots, audit["roots"], plan["config"], plan["split"]["train_maps"], plan["split"]["validation_maps"])
    model = fit_once(matrix)
    payload = portable_payload(model)
    payload["name"] = "sa_policy_aligned_update_development_v1"
    from lns2_selector.runtime.fingerprints import semantic_fingerprint
    payload["semantic_fingerprint"] = semantic_fingerprint({k:v for k,v in payload.items() if k not in {"schema", "semantic_fingerprint"}})
    portable = portable_model(payload, matrix["names"], native=False)
    fixtures = []
    for r in roots:
        state = dict(anchor_id=r["anchor_id"], agent_ids=[a["id"] for a in r["state"]["agents"]],
            candidates=[dict(c, features=f) for c, f in zip(r["candidates"], r["features"], strict=True)])
        ranking = model.rank(state)
        require(ranking == portable.rank(state), "sklearn/portable mismatch")
        fixtures.append(dict(state=state, ranking=ranking))
    once(out/"model/bundle.json", payload)
    once(out/"model/fixtures.json", fixtures)
    record(out/"model/training.json", dict(binding=plan["binding"], train_roots=[r["id"] for r in roots],
        train_maps=plan["split"]["train_maps"], label_sha256=sha256_file(out/"labels.audit.json"), model_fits=1,
        feature_names=matrix["names"], production_allowed=False,
        files={n:sha256_file(out/"model"/n) for n in ("bundle.json", "fixtures.json")}))
    return dict(model_fits=1, features=len(matrix["names"]), rows=len(matrix["x"]), production_allowed=False)


def verify_model():
    plan, out = verify()
    training = check_seal(read_json(out/"model/training.json"))
    require(training["binding"] == plan["binding"], "model binding")
    for n,h in training["files"].items():
        require(sha256_file(out/"model"/n) == h, "model changed")
    model = portable_model(read_json(out/"model/bundle.json"), training["feature_names"])
    require(model.estimator.model.inference_backend == "native-portable-tree", "native verification required")
    for f in read_json(out/"model/fixtures.json"):
        require(model.rank(f["state"]) == f["ranking"], "native fixture mismatch")
    record(out/"model/receipt.json", dict(binding=plan["binding"], files=training["files"],
        training_sha256=sha256_file(out/"model/training.json"), parity_verified=True, production_allowed=False))
    return dict(native_parity=True, fixtures=len(read_json(out/"model/fixtures.json")))


def analyze():
    plan, out = verify()
    require((out/"evaluation.complete.json").exists(), "evaluation incomplete")
    with ProcessPoolExecutor(max_workers=plan["config"]["workers"]) as pool:
        rows = list(pool.map(audit_job, [dict(j, plan=plan) for j in jobs_for(plan, "evaluation")]))
    for key in plan["split"]["validation_pair_ids"]:
        group = [r for r in rows if r["pair_id"] == key]
        require(len(group) == 3 and len({r["initial_fingerprint"] for r in group}) == 1, "unpaired evaluation")
    summary = {a:dict(episodes=sum(r["arm"] == a for r in rows), success=sum(r["success"] for r in rows if r["arm"] == a),
        censored=sum(r["stop"] in CENSORED for r in rows if r["arm"] == a),
        generated=sum(r["accumulated_generated"] for r in rows if r["arm"] == a)) for a in ARMS}
    contrasts = {}
    for base in ("frozen", "gbdt"):
        pairs = [(next(r for r in rows if r["pair_id"] == k and r["arm"] == "updated"),
                  next(r for r in rows if r["pair_id"] == k and r["arm"] == base)) for k in plan["split"]["validation_pair_ids"]]
        common = [(a,b) for a,b in pairs if a["success"] and b["success"]]
        contrasts[base] = dict(observed_net_success=sum(int(a["success"])-int(b["success"]) for a,b in pairs),
            common_success=len(common), common_nodes_ratio=sum(a["accumulated_generated"] for a,b in common)/sum(b["accumulated_generated"] for a,b in common) if common else None,
            per_map={m:sum(int(a["success"])-int(b["success"]) for a,b in pairs if a["map_id"] == m) for m in plan["split"]["validation_maps"]})
    censored = sum(s["censored"] for s in summary.values())
    positive = contrasts["frozen"]["observed_net_success"] > 0 or (
        contrasts["frozen"]["observed_net_success"] == 0 and contrasts["frozen"]["common_nodes_ratio"] is not None and contrasts["frozen"]["common_nodes_ratio"] < 1)
    decision = "resource_censored_inconclusive" if censored else "development_signal" if positive else "no_net_signal_do_not_repeat_parameters"
    result = dict(binding=plan["binding"], summary=summary, contrasts=contrasts, episodes=rows, decision=decision,
        no_ttf=True, independent_confirmation=False, automatic_promotion=False, maximum_updates=1)
    record(out/"report.json", result)
    return {k:v for k,v in result.items() if k != "episodes"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=("prepare", "dry-run", "preflight", "collect", "audit-labels", "train", "verify-model", "evaluate", "analyze", "verify", "request-stop"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    if args.phase == "prepare": result = prepare()
    elif args.phase in ("verify", "dry-run"):
        p, _ = verify()
        result = {k:p[k] for k in ("binding", "label_jobs", "maximum_label_repairs", "replay_repairs", "validation_episodes")}
    elif args.phase in ("preflight", "collect", "evaluate"):
        result = collect({"preflight":"preflight", "collect":"labels", "evaluate":"evaluation"}[args.phase], args.resume, args.limit)
    elif args.phase == "audit-labels": result = label_audit()
    elif args.phase == "train": result = train()
    elif args.phase == "verify-model": result = verify_model()
    elif args.phase == "analyze": result = analyze()
    else:
        _, out = verify()
        once(out/"STOP_AFTER_BATCH", dict(requested=True))
        result = dict(safe_stop_after_active_batch=True)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
