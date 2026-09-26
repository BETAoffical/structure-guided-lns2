"""One-intervention versus continuous takeover; frozen native, no TTF/training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, read_jsonl, sha256_file, json_fingerprint
from experiments.sa_paired_completion import require
from experiments.sa_paired_closed_loop import choose, subset, portable_model, stop_reason
from experiments.sa_intervention_frequency import select_once, first_disagreement, horizon_outcome, completion_contrast, interpretation
from scripts import run_sa_paired_closed_loop as old
from scripts.audit_sa_history_information import atomic

CONFIG = ROOT / "configs/sa_intervention_frequency.json"


def trace_prefix(folder, decision):
    events = []
    with (folder / "trace.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            event = json.loads(line)
            events.append(event)
            if decision is not None and event["decision"] >= decision:
                break
    return events


def prepare():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    require(not out.exists(), "output already exists; no overwrite")
    source, source_out = old.verify()
    require(source_out == ROOT / cfg["source_output"] and source["binding"] == cfg["source_binding"], "wrong source")
    require(sha256_file(source_out / "report.json") == cfg["source_report_sha256"], "source report changed")
    old.model_files(source_out, source)
    cases = old.cases_verified(source, source_out)
    qualification = old.check_seal(read_json(source_out / "qualification.json"))
    require(qualification["passed"] and qualification["binding"] == source["binding"], "qualification mismatch")
    files = dict(source["inputs"])
    additions = [CONFIG, Path(__file__), ROOT / "experiments/sa_intervention_frequency.py",
        ROOT / "tests/evaluation/test_sa_intervention_frequency.py",
        ROOT / "docs/SA_INTERVENTION_FREQUENCY_PROTOCOL_ZH.md",
        source_out / "report.json", source_out / "plan.json", source_out / "cases.json",
        source_out / "qualification.json", source_out / "model/receipt.json"]
    additions += list((source_out / "model").glob("*"))
    jobs = []
    for case, seed, key in old.paired_tasks(cases, source["config"]):
        baselines = {}
        for arm in ("frozen", "paired"):
            folder = source_out / "episodes" / f"{key}-{arm}"
            result = old.receipt_valid(folder, source)
            require(result["pair_id"] == key and result["arm"] == arm, "baseline identity")
            baselines[arm] = result
            additions += [folder / name for name in ("receipt.json", "result.json", "initial.json", "trace.jsonl")]
        require(baselines["frozen"]["initial_fingerprint"] == baselines["paired"]["initial_fingerprint"], "baseline pairing")
        events = read_jsonl(source_out / "episodes" / f"{key}-paired" / "trace.jsonl")
        decision = first_disagreement(events)
        jobs.append(dict(job_id=key + "-single", pair_id=key, case=case, solver_seed=seed,
                         expected_initial=baselines["frozen"]["initial_fingerprint"], intervention=decision))
    require(len(jobs) == cfg["expected_pairs"] and len({j["case"]["map_id"] for j in jobs}) == cfg["expected_maps"], "cohort mismatch")
    for path in additions:
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    plan = dict(config=cfg, source=source, files=files, jobs=jobs,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        no_ttf=True, no_training=True, production_changed=False)
    plan["binding"] = json_fingerprint(plan)
    old.once(out / "plan.json", plan)
    return dict(binding=plan["binding"], jobs=len(jobs), maps=cfg["expected_maps"],
        maximum_repairs=len(jobs)*source["config"]["max_decisions"],
        first_disagreements={str(d):sum(j["intervention"] == d for j in jobs) for d in sorted({j["intervention"] for j in jobs}, key=str)},
        maximum_workers=cfg["workers"], sequential_safety_seconds=len(jobs)*source["config"]["fuse_seconds"],
        ideal_batch_safety_seconds=((len(jobs)+cfg["workers"]-1)//cfg["workers"])*source["config"]["fuse_seconds"])


def verify():
    from experiments._common import verify_registered_plan
    return verify_registered_plan(ROOT, CONFIG)


def check_prefix(event, before, after, paired, frozen, at_intervention):
    from scripts import run_sa_path_quality as q
    keys = ("decision", "before", "pool", "subset", "anchor_id", "features", "temperature", "uniform")
    for reference in (paired, frozen):
        require(all(event[k] == reference[k] for k in keys), "common-prefix input mismatch")
        require(event["ranking"]["scores"] == reference["ranking"]["scores"], "common-prefix scores mismatch")
    require(event["action"] == paired["action"], "paired action mismatch")
    if not at_intervention:
        require(event["action"] == frozen["action"], "frozen prefix action mismatch")
    expected = q.apply_state_delta(before, paired["delta"])
    require(q.state_fingerprint(after) == q.state_fingerprint(expected), "first-action successor mismatch")


def worker(job):
    from scripts import run_sa_path_quality as q
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from scripts.train_sa_history_selector import die_with_parent
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    die_with_parent(job["parent_pid"])
    plan = job["plan"]
    cfg = plan["source"]["config"]
    out = ROOT / plan["config"]["output"]
    source_out = ROOT / plan["config"]["source_output"]
    folder = out / "episodes" / job["job_id"]
    if (folder / "receipt.json").exists():
        result = old.receipt_valid(folder, plan)
        return dict(status="ok", job_id=job["job_id"], resumed=True, stop=result["stop"])
    require(not folder.exists(), "partial episode: audit required, no automatic retry")
    model = portable_model(old.model_files(source_out, plan["source"]), read_json(source_out / "model/metadata.json")["feature_names"])
    require(model.estimator.model.inference_backend == "native-portable-tree", "native inference required")
    env, state, source = old.reset_case(job["case"], job["solver_seed"], plan["source"])
    require(q.state_fingerprint(state) == job["expected_initial"], "initial state mismatch")
    validate_final(state)
    prefixes = {arm:trace_prefix(source_out / "episodes" / f"{job['pair_id']}-{arm}", job["intervention"])
                for arm in ("paired", "frozen")}
    history = History(state)
    selector = q.SingleFullCheckPool(source)
    engine = OnlineFeatureEngine(state, backend="native")
    initial_nodes = state["low_level"]["generated"]
    folder.mkdir(parents=True)
    old.once(folder / "initial.json", dict(binding=plan["binding"], state=state))
    decisions = 0
    used = False
    intervention = None
    trajectory = [state["num_of_colliding_pairs"]]
    began = time.monotonic()
    with (folder / "trace.jsonl").open("x", encoding="utf8") as stream:
        while True:
            stop = stop_reason(state, decisions, state["low_level"]["generated"] - initial_nodes, time.monotonic()-began, cfg)
            if stop:
                break
            fp = q.state_fingerprint(state)
            anchor_index, pool = selector.select(env, state, decisions)
            require(q.state_fingerprint(env.get_state()) == fp, "proposal changed state")
            anchor = pool[anchor_index]["candidate_id"]
            key = f"{job['pair_id']}-d{decisions:04d}-{fp}"
            ids = subset(pool, anchor, key, cfg["candidate_seed"], cfg["candidate_limit"])
            candidates = [next(c for c in pool if c["candidate_id"] == i) for i in ids]
            temp = q.temperature(decisions)
            features = old.features_for(state, candidates, engine, history, temp, fp)
            ranking = choose("paired", candidates, anchor, model, features, [a["id"] for a in state["agents"]], key, cfg["candidate_seed"])
            selected, next_used, used_now = select_once(anchor, ranking["selected"], used)
            index = next(i for i,c in enumerate(pool) if c["candidate_id"] == selected)
            draw = q.acceptance_draw(q.seed(source["case_id"], 0, decisions, "accept"))
            action = dict(mode="explicit_neighborhood", agents=pool[index]["agents"], random_seed=q.seed(source["case_id"], 0, decisions, "pp"))
            remaining = cfg["episode_seconds"] - (time.monotonic()-began)
            if remaining <= 0:
                stop = "wall_safety"
                break
            raw = q._plain(env.step_experimental_pp(action, min(cfg["pp_seconds"], remaining), "annealed", temp, draw))
            after, metrics = raw["observation"], raw["metrics"]
            q.validate_transition(state, after, metrics, action["agents"], "annealed", temp, draw)
            event = dict(decision=decisions, before=fp, action=action, temperature=temp, uniform=draw,
                pool=pool, subset=ids, anchor_id=anchor, selected_index=index, ranking=ranking,
                applied_selected=selected, intervention=used_now, features=features, metrics=metrics,
                delta=q.encode_state_delta(state, after))
            if not used:
                require(decisions < len(prefixes["paired"]) and decisions < len(prefixes["frozen"]), "source prefix missing")
                check_prefix(event, state, after, prefixes["paired"][decisions], prefixes["frozen"][decisions], used_now)
            if used_now:
                require(decisions == job["intervention"], "intervention occurrence changed")
                intervention = decisions
            stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            history.observe(state, event, after)
            used = next_used
            state = after
            decisions += 1
            trajectory.append(state["num_of_colliding_pairs"])
            if metrics["pp_failure_reason"] == "time_limit" or not metrics["acceptance_evaluated"]:
                stop = "incomplete_pp"
                break
    validate_final(state)
    result = dict(binding=plan["binding"], job_id=job["job_id"], pair_id=job["pair_id"], map_id=job["case"]["map_id"],
        arm="single", solver_seed=job["solver_seed"], initial_fingerprint=job["expected_initial"],
        final_fingerprint=q.state_fingerprint(state), stop=stop, success=state["feasible"], decisions=decisions,
        generated=state["low_level"]["generated"]-initial_nodes,
        node_overshoot=max(0, state["low_level"]["generated"]-initial_nodes-cfg["node_budget"]),
        conflicts=trajectory, intervention=intervention, final_soc=state["sum_of_costs"],
        final_makespan=max(len(a["path"])-1 for a in state["agents"]), diagnostic_seconds=time.monotonic()-began, no_ttf=True)
    old.once(folder / "result.json", result)
    old.once(folder / "receipt.json", dict(binding=plan["binding"], files={name:sha256_file(folder/name) for name in ("initial.json", "trace.jsonl", "result.json")}))
    return dict(status="ok", job_id=job["job_id"], success=result["success"], stop=stop, decisions=decisions, intervention=intervention)


def collect(resume=False, limit=None):
    from experiments.repair_collection import _run_jobs, _CollectionRunLock
    plan, out = verify()
    require(resume or not (out / "episodes").exists(), "resume required")
    jobs = [dict(j, plan=plan, parent_pid=os.getpid()) for j in plan["jobs"]]
    if limit is not None:
        require(0 < limit <= len(jobs), "invalid limit")
        jobs = jobs[:limit]
    size = plan["config"]["workers"]
    with _CollectionRunLock(out, plan["binding"], "intervention-frequency"):
        atomic(out / "run_status.json", dict(status="running", binding=plan["binding"]))
        try:
            for offset in range(0, len(jobs), size):
                if (out / "STOP_AFTER_BATCH").exists():
                    break
                def record(row):
                    with (out / "progress.jsonl").open("a", encoding="utf8") as stream:
                        stream.write(json.dumps(row) + "\n")
                    print("EPISODE", row, flush=True)
                batch = jobs[offset:offset+size]
                rows = _run_jobs(worker, batch, size, phase=f"batch{offset}", output_root=out/f"progress/{offset}",
                    run_fingerprint=plan["binding"], timeout_seconds=plan["source"]["config"]["fuse_seconds"],
                    stop_on_failure=True, on_result=record)
                require(len(rows) == len(batch) and all(r["status"] == "ok" for r in rows), "batch failed; inspect before retry")
            count = sum((out / "episodes" / j["job_id"] / "receipt.json").exists() for j in plan["jobs"])
            atomic(out / "run_status.json", dict(status="completed" if count == len(plan["jobs"]) else "paused", completed=count, binding=plan["binding"]))
        except BaseException as exc:
            atomic(out / "run_status.json", dict(status="error", error=repr(exc), binding=plan["binding"]))
            raise
    return dict(completed=count, total=len(plan["jobs"]))


def audit_worker(job):
    from scripts import run_sa_path_quality as q
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    plan = job["plan"]
    cfg = plan["source"]["config"]
    folder = ROOT / plan["config"]["output"] / "episodes" / job["job_id"]
    result = old.receipt_valid(folder, plan)
    require(result["job_id"] == job["job_id"] and result["pair_id"] == job["pair_id"] and result["map_id"] == job["case"]["map_id"], "result identity")
    state = read_json(folder / "initial.json")["state"]
    require(q.state_fingerprint(state) == job["expected_initial"] == result["initial_fingerprint"], "initial identity")
    source_out = ROOT / plan["config"]["source_output"]
    model = portable_model(old.model_files(source_out, plan["source"]), read_json(source_out / "model/metadata.json")["feature_names"], native=False)
    engine = OnlineFeatureEngine(state, backend="native")
    history = History(state)
    nodes = state["low_level"]["generated"]
    counts = [state["num_of_colliding_pairs"]]
    prefixes = {a:trace_prefix(source_out/"episodes"/f"{job['pair_id']}-{a}", job["intervention"]) for a in ("frozen", "paired")}
    used = False
    intervention = None
    count = 0
    for event in read_jsonl(folder / "trace.jsonl"):
        fp = q.state_fingerprint(state)
        require(event["decision"] == count and event["before"] == fp, "trace sequence")
        key = f"{job['pair_id']}-d{count:04d}-{fp}"
        ids = subset(event["pool"], event["anchor_id"], key, cfg["candidate_seed"], cfg["candidate_limit"])
        require(ids == event["subset"], "subset mismatch")
        candidates = [next(c for c in event["pool"] if c["candidate_id"] == i) for i in ids]
        features = old.features_for(state, candidates, engine, history, event["temperature"], fp)
        require(features == event["features"], "pre-action features changed")
        prediction = choose("paired", candidates, event["anchor_id"], model, features, [a["id"] for a in state["agents"]], key, cfg["candidate_seed"])
        require(prediction == event["ranking"], "Python/native ranking mismatch")
        selected, next_used, used_now = select_once(event["anchor_id"], prediction["selected"], used)
        require(selected == event["applied_selected"] == event["pool"][event["selected_index"]]["candidate_id"] and used_now == event["intervention"], "one-intervention policy mismatch")
        case_id = job["pair_id"].rsplit("-s", 1)[0] + f"-seed{job['solver_seed']}"
        require(event["action"]["random_seed"] == q.seed(case_id, 0, count, "pp") and event["temperature"] == q.temperature(count) and event["uniform"] == q.acceptance_draw(q.seed(case_id, 0, count, "accept")), "random stream mismatch")
        require(event["action"]["agents"] == event["pool"][event["selected_index"]]["agents"], "explicit action changed")
        after = q.apply_state_delta(state, event["delta"])
        q.validate_transition(state, after, event["metrics"], event["action"]["agents"], "annealed", event["temperature"], event["uniform"])
        if not used:
            check_prefix(event, state, after, prefixes["paired"][count], prefixes["frozen"][count], used_now)
        if used_now:
            intervention = count
        history.observe(state, event, after)
        state, used = after, next_used
        counts.append(state["num_of_colliding_pairs"])
        count += 1
    validate_final(state)
    require(count == result["decisions"] and counts == result["conflicts"] and q.state_fingerprint(state) == result["final_fingerprint"], "final trajectory mismatch")
    require(result["success"] == state["feasible"] and result["generated"] == state["low_level"]["generated"]-nodes, "outcome mismatch")
    require(result["final_soc"] == state["sum_of_costs"] and result["final_makespan"] == max(len(a["path"])-1 for a in state["agents"]), "quality mismatch")
    require(intervention == result["intervention"], "intervention receipt mismatch")
    if result["stop"] not in {"wall_safety", "incomplete_pp"}:
        require(intervention == job["intervention"] and stop_reason(state, count, result["generated"], 0, cfg) == result["stop"], "termination mismatch")
    if result["stop"] == "incomplete_pp":
        require(count > 0 and (event["metrics"]["pp_failure_reason"] == "time_limit" or not event["metrics"]["acceptance_evaluated"]), "unexplained PP censoring")
    return result


def analyze():
    plan, out = verify()
    cfg = plan["config"]
    require(all((out/"episodes"/j["job_id"]/"receipt.json").exists() for j in plan["jobs"]), "incomplete cohort")
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        results = list(pool.map(audit_worker, [dict(j, plan=plan) for j in plan["jobs"]]))
    paired = []
    episodes = []
    for job, single in zip(plan["jobs"], results, strict=True):
        arms = {a:old.receipt_valid(ROOT/cfg["source_output"]/"episodes"/f"{job['pair_id']}-{a}", plan["source"]) for a in ("frozen", "paired")}
        arms["single"] = single
        require(len({r["initial_fingerprint"] for r in arms.values()}) == 1, "paired reset mismatch")
        paired.append(dict(pair_id=job["pair_id"], map_id=single["map_id"], intervention=single["intervention"],
            success={a:int(r["success"]) for a,r in arms.items()},
            unknown={a:int(not r["success"] and r["stop"] in {"incomplete_pp", "wall_safety", "external_timeout"}) for a,r in arms.items()},
            h32={a:horizon_outcome(r, single["intervention"], cfg["horizon"]) for a,r in arms.items()}))
        episodes.extend(arms.values())
    summary = {a:dict(success=sum(r["success"] for r in episodes if r["arm"] == a),
        unknown=sum(p["unknown"][a] for p in paired), episodes=len(paired),
        decisions=sum(r["decisions"] for r in episodes if r["arm"] == a),
        generated=sum(r["generated"] for r in episodes if r["arm"] == a)) for a in ("frozen", "paired", "single")}
    contrasts = {a:completion_contrast(paired, a, cfg["bootstrap"], cfg["seed"]) for a in ("frozen", "paired")}
    common = {}
    for baseline in ("frozen", "paired"):
        keys = {p["pair_id"] for p in paired if p["success"]["single"] and p["success"][baseline]}
        common[baseline] = dict(pairs=len(keys), totals={a:{metric:sum(r[metric] for r in episodes if r["arm"] == a and r["pair_id"] in keys)
            for metric in ("generated", "decisions", "final_soc", "final_makespan")} for a in ("single", baseline)})
    report = dict(binding=plan["binding"], summary=summary, contrasts=contrasts, common_success=common,
        paired_cases=paired, decision=interpretation(contrasts), no_ttf=True, production_changed=False,
        files={p.relative_to(out).as_posix():sha256_file(p) for p in sorted((out/"episodes").rglob("*")) if p.is_file()})
    verify()
    old.once(out / "report.json", report)
    return {k:v for k,v in report.items() if k not in ("paired_cases", "files")}


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "collect", "analyze", "request-stop"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "collect":
        result = collect(args.resume, args.limit)
    elif args.phase == "verify":
        plan, _ = verify()
        result = dict(binding=plan["binding"], registered_files=len(plan["files"]))
    elif args.phase == "request-stop":
        plan, out = verify()
        old.once(out/"STOP_AFTER_BATCH", dict(binding=plan["binding"]))
        result = dict(stop_requested=True)
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
