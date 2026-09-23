"""Terminal support from full-task states, retaining prefix work and SA history."""
import argparse
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_sa_parent_update as parent
from experiments import sa_uncapped_runtime as runtime
from experiments.sa_history_selector import History

run, batch, recovery = parent.run, parent.collection, parent.recovery
CONFIG = "configs/sa_original_terminal_support.json"
REGISTRATION = "support_registration.json"
CODE = (CONFIG, "scripts/probe_sa_original_terminal_support.py",
        "tests/evaluation/test_sa_original_terminal_support.py",
        "docs/SA_ORIGINAL_TERMINAL_SUPPORT_PROTOCOL_ZH.md")


def config():
    cfg = run.read_json(ROOT / CONFIG)
    expected = dict(threshold=10, roots_per_map=1, trials=4, workers=20, node_budget=25000000,
                    max_decisions=None, training=False, formal_ttf=False, automatic_promotion=False,
                    episode_safety_seconds=900., process_fuse_seconds=1860.)
    run.require(all(cfg[k] == v for k, v in expected.items()), "fixed support scope")
    return cfg


def first_crossing(records, threshold, budget):
    # First observed crossing, not a hindsight minimum or a successful suffix.
    for r in records:
        if 0 < r["conflicts"] <= threshold and r["generated"] < budget:
            return r
    return None


def select_roots(records, maps):
    run.require(len({r["episode_id"] for r in records}) == len(records), "unique source episodes")
    run.require(all(r["split"] == "train" and r["map_id"] in maps for r in records), "Train only")
    chosen = {}
    for r in sorted(records, key=lambda r: (r["pair_id"], r["replica"], r["episode_id"])):
        if r["root"] is not None and r["map_id"] not in chosen:
            chosen[r["map_id"]] = r
    return [chosen[m] for m in sorted(chosen)]


def history_signature(history):
    return run.json_fingerprint(dict(decision=history.decision, best=history.best,
        since_best=history.since_best, recent=list(history.recent),
        ages=[(list(e), age) for e, age in sorted(history.ages.items())]))


def prepare():
    from scripts import run_sa_path_quality as q
    cfg = config()
    _, sr, source, _, actor, entries = parent.evidence()
    run.require(sr["config"]["output"] == cfg["source"] and
                run.sha256_file(ROOT / cfg["actor"]) == cfg["actor_sha256"], "source identity")
    out = ROOT / cfg["output"]
    run.require(not out.exists(), "existing output requires inspection")
    jobs = {j["job_id"]: j for j in sr["jobs"]}
    records = []
    for e in sorted(entries, key=lambda e: e["job_id"]):
        job = jobs[e["job_id"]]
        if job["case"]["task_variant"] != "bottleneck_d25":
            continue
        initial = run.read_json(ROOT / e["folder"] / "initial.json")
        state = initial
        history = History(state)
        points = []
        for event in run.trace_read(ROOT / e["folder"]):
            run.require(event["decision"] == history.decision and
                        event["before"] == q.state_fingerprint(state), "source continuity")
            # Only pre-action states can supply the original next-action control.
            points.append(dict(decision=history.decision, conflicts=state["num_of_colliding_pairs"],
                generated=state["low_level"]["generated"] - initial["low_level"]["generated"],
                fingerprint=event["before"], history=history_signature(history)))
            after = q.apply_state_delta(state, event["delta"])
            history.observe(state, event, after)
            state = after
        run.require(q.state_fingerprint(state) == e["episode"]["final_fingerprint"], "source final")
        records.append(dict(episode_id=e["job_id"], map_id=e["episode"]["map_id"], split="train",
            pair_id=job["pair_id"], replica=job["replica"], folder=e["folder"], source_job=job,
            source_success=e["episode"]["success"],
            root=first_crossing(points, cfg["threshold"], cfg["node_budget"])))
    run.require(len(records) == 48, "complete original d25 corpus")
    roots = select_roots(records, source["split"]["train_maps"])
    inputs = {n: run.sha256_file(ROOT / n) for n in CODE}
    inputs.update(sr["inputs"])
    inputs[cfg["actor"]] = cfg["actor_sha256"]
    for n in ("registration.json", "report.json", "audit.json", "collection.complete.json"):
        p = ROOT / cfg["source"] / n
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    for r in records:
        for n in ("result.json", "initial.json", "final.json", "trace.jsonl.gz"):
            p = ROOT / r["folder"] / n
            inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    body = dict(config=cfg, source=source, source_config=sr["config"], parent_policy=run.validate_bundle(actor),
        records=records, roots=roots, inputs=inputs, no_training=True, no_ttf=True,
        root_selection_exploratory=True, no_new_initial_tasks=True,
        source_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out, body["binding"], "support-prepare"):
        run.once(out / REGISTRATION, run.sealed(body))
    return dry_run(body)


def verify():
    cfg = config()
    out = ROOT / cfg["output"]
    reg = run.check_seal(run.read_json(out / REGISTRATION))
    run.require(reg["config"] == cfg and reg["binding"] == run.json_fingerprint(
        {k: v for k, v in reg.items() if k not in ("binding", "integrity")}), "registration")
    for p, sha in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT, p, field="support input")) == sha, "input changed: " + p)
    run.require(select_roots(reg["records"], reg["source"]["split"]["train_maps"]) == reg["roots"], "root selection")
    run.require(run.validate_bundle(run.read_json(ROOT / cfg["actor"])) == reg["parent_policy"], "model semantics")
    return reg, out


def dry_run(reg):
    roots, cfg = reg["roots"], reg["config"]
    return dict(roots=len(roots), controls=len(roots), branches=len(roots) * cfg["trials"],
        source_episodes=48, missing_maps=sorted(set(reg["source"]["split"]["train_maps"]) - {r["map_id"] for r in roots}),
        replay_repairs=sum(r["root"]["decision"] for r in roots) * (cfg["trials"] + 1),
        remaining_node_allowance=sum(cfg["node_budget"] - r["root"]["generated"] for r in roots) * cfg["trials"],
        note="Node allowance can overshoot by the last atomic PP call; no decision bound or TTF estimate.",
        rows=[dict(map_id=r["map_id"], source_replica=r["replica"], **r["root"],
                   remaining_nodes=cfg["node_budget"] - r["root"]["generated"]) for r in roots])


def jobs_for(reg, phase):
    jobs = []
    for r in reg["roots"]:
        for trial in ([-1] if phase == "control" else range(reg["config"]["trials"])):
            key = run.json_fingerprint([reg["binding"], r["episode_id"], phase, trial])[:24]
            jobs.append(dict(job_id=key, phase=phase, trial=trial, root=r, registration=reg, parent_pid=os.getpid()))
    return jobs


def folder(j):
    return ROOT / j["registration"]["config"]["output"] / j["phase"] / j["job_id"]


def context(j):
    reg = j["registration"]
    plan = batch.compare.runtime_plan(reg["source"], reg["source_config"], "uncapped_condition")
    job = dict(j["root"]["source_job"], plan=plan)
    actor = run.NumpyActor(run.read_json(ROOT / reg["config"]["actor"]))
    return plan, job, actor


def replay(j):
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_onpolicy_noop_runtime import pp_incomplete
    plan, job, actor = context(j)
    q, env, state, ctx = run.reset(job)
    initial = run.read_json(ROOT / j["root"]["folder"] / "initial.json")
    run.require(q.state_fingerprint(state) == q.state_fingerprint(initial), "reset fingerprint")
    ctx = dict(ctx, initial_nodes=state["low_level"]["generated"])
    history = History(state)
    selector = q.SingleFullCheckPool(ctx)
    engine = OnlineFeatureEngine(state, backend="native")
    expected = None
    for old in run.trace_read(ROOT / j["root"]["folder"]):
        event = run.selection(env, state, ctx, history, engine, selector, runtime.feature_plan(plan), actor)
        run.require(all(event[k] == old[k] for k in event), "prefix pool/features/probabilities")
        run.require(old["before"] == q.state_fingerprint(state), "prefix fingerprint")
        if history.decision == j["root"]["root"]["decision"]:
            expected = old
            break
        raw = q._plain(env.step_experimental_pp(old["action"], plan["proposal"]["pp_safety_seconds"],
                                               "annealed", old["temperature"], old["uniform"]))
        after, metrics = raw["observation"], raw["metrics"]
        q.validate_transition(state, after, metrics, old["action"]["agents"], "annealed", old["temperature"], old["uniform"])
        run.require(not pp_incomplete(state, after, metrics) and q.state_fingerprint(after) ==
                    q.state_fingerprint(q.apply_state_delta(state, old["delta"])), "native prefix replay")
        history.observe(state, old, after)
        state = after
    root = j["root"]["root"]
    run.require(expected is not None and q.state_fingerprint(state) == root["fingerprint"] and
                history_signature(history) == root["history"] and
                state["low_level"]["generated"] - ctx["initial_nodes"] == root["generated"], "complete root context")
    return q, env, state, ctx, history, selector, engine, actor, plan, expected


def execute_step(j, q, env, state, ctx, history, selector, engine, actor, plan, remaining):
    cfg = j["registration"]["config"]
    d = history.decision
    event = run.selection(env, state, ctx, history, engine, selector, runtime.feature_plan(plan), actor)
    rng = (plan, cfg["phase"], j["root"]["episode_id"], j["trial"], d)
    draw = run.stream_draw(*rng, "select")
    cid = run.select_with_draw(event["probabilities"], draw)
    selected = next(c["agents"] for c in event["pool"] if c["candidate_id"] == cid)
    action = dict(mode="explicit_neighborhood", agents=selected, random_seed=run.stream_draw(*rng, "pp"))
    temperature, uniform = q.temperature(d), run.stream_draw(*rng, "accept")
    raw = q._plain(env.step_experimental_pp(action, min(plan["proposal"]["pp_safety_seconds"], remaining),
                                           "annealed", temperature, uniform))
    after, metrics = raw["observation"], raw["metrics"]
    q.validate_transition(state, after, metrics, selected, "annealed", temperature, uniform)
    event.update(decision=d, selected_id=cid, behavior_log_probability=math.log(event["probabilities"][cid]),
        selection_draw=draw, policy_sha256=actor.sha, before=q.state_fingerprint(state), action=action,
        temperature=temperature, uniform=uniform, metrics=metrics, delta=q.encode_state_delta(state, after))
    return event, after


def worker(j):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.sa_onpolicy_noop_runtime import pp_incomplete
    die_with_parent(j["parent_pid"])
    out, cfg = folder(j), j["registration"]["config"]
    run.require(not out.exists(), "partial work requires inspection")
    q, env, state, ctx, history, selector, engine, actor, plan, expected = replay(j)
    run.once(out / "root.json", state)
    began = time.monotonic()
    stop = None
    if j["phase"] == "control":
        raw = q._plain(env.step_experimental_pp(expected["action"], plan["proposal"]["pp_safety_seconds"],
            "annealed", expected["temperature"], expected["uniform"]))
        after, metrics = raw["observation"], raw["metrics"]
        q.validate_transition(state, after, metrics, expected["action"]["agents"], "annealed", expected["temperature"], expected["uniform"])
        run.require(not pp_incomplete(state, after, metrics) and q.state_fingerprint(after) ==
                    q.state_fingerprint(q.apply_state_delta(state, expected["delta"])), "original next-action control")
        run.once(out / "control.json", dict(before=expected["before"], after=q.state_fingerprint(after)))
        history.observe(state, expected, after)
        state, stop = after, "control_validated"
    else:
        with gzip.open(out / "trace.jsonl.gz", "xt", encoding="utf8", compresslevel=3) as stream:
            while True:
                used = state["low_level"]["generated"] - ctx["initial_nodes"]
                stop = runtime.work_stop(state["feasible"], history.decision, used, plan["proposal"])
                if stop:
                    break
                remaining = cfg["episode_safety_seconds"] - (time.monotonic() - began)
                if remaining <= 0:
                    stop = "wall_safety"
                    break
                event, after = execute_step(j, q, env, state, ctx, history, selector, engine, actor, plan, remaining)
                stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
                stream.flush()
                incomplete = pp_incomplete(state, after, event["metrics"])
                history.observe(state, event, after)
                state = after
                if incomplete:
                    stop = "incomplete_pp"
                    break
    validate_final(state)
    run.once(out / "final.json", state)
    paths = [a["path"] for a in state["agents"]]
    row = dict(binding=j["registration"]["binding"], job_id=j["job_id"], phase=j["phase"], trial=j["trial"],
        root_episode=j["root"]["episode_id"], map_id=j["root"]["map_id"], split="train_state_support_diagnostic",
        policy_sha256=actor.sha, status="censored" if stop in {"wall_safety", "incomplete_pp"} else "ok",
        stop=stop, success=state["feasible"], root_decision=j["root"]["root"]["decision"], absolute_decisions=history.decision,
        prefix_generated=j["root"]["root"]["generated"], generated=state["low_level"]["generated"] - ctx["initial_nodes"],
        final_conflicts=state["num_of_colliding_pairs"], final_fingerprint=q.state_fingerprint(state),
        soc=sum(len(p)-1 for p in paths), makespan=max(len(p)-1 for p in paths),
        wait_steps=sum(a == b for p in paths for a, b in zip(p, p[1:])), no_ttf=True,
        files={p.name: run.sha256_file(p) for p in out.iterdir() if p.is_file()})
    if j["phase"] != "control":
        runtime.validate_terminal(dict(row, decisions=history.decision), plan["proposal"])
    run.once(out / "result.json", run.sealed(row))
    return {k: row[k] for k in ("job_id", "status", "stop", "success", "absolute_decisions")}


def read_result(j):
    row = run.result_read(folder(j), j["registration"])
    run.require(row["job_id"] == j["job_id"] and row["root_episode"] == j["root"]["episode_id"] and
                row["trial"] == j["trial"] and row["phase"] == j["phase"] and
                row["policy_sha256"] == j["registration"]["parent_policy"] and
                row["split"] == "train_state_support_diagnostic", "result identity")
    return row


def audit_worker(j):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.compact_controller_model import load_controller_bundle
    from lns2_selector.runtime.online_selection import score_online_candidates
    from experiments.sa_onpolicy_noop_runtime import pp_incomplete
    die_with_parent(j["parent_pid"])
    row = read_result(j)
    plan, _, actor = context(j)
    q = run.native_runtime(plan)
    source = ROOT / j["root"]["folder"]
    state = run.read_json(source / "initial.json")
    initial_nodes = state["low_level"]["generated"]
    history = History(state)
    for e in run.trace_read(source):
        if history.decision == j["root"]["root"]["decision"]:
            break
        run.require(e["before"] == q.state_fingerprint(state), "audit source before")
        after = q.apply_state_delta(state, e["delta"])
        history.observe(state, e, after)
        state = after
    run.require(q.state_fingerprint(state) == j["root"]["root"]["fingerprint"] ==
        q.state_fingerprint(run.read_json(folder(j) / "root.json")) and
        history_signature(history) == j["root"]["root"]["history"], "audit inherited root")
    engine = OnlineFeatureEngine(state, backend="native")
    model = load_controller_bundle(ROOT / "artifacts/initlns-closed-loop-controller-v2").main_models["realized_dynamic"]
    minimum = state["num_of_colliding_pairs"]
    for event in run.trace_read(folder(j)):
        d = history.decision
        used = state["low_level"]["generated"] - initial_nodes
        run.require(runtime.work_stop(state["feasible"], d, used, plan["proposal"]) is None, "action after terminal")
        run.require(event["decision"] == d and event["before"] == q.state_fingerprint(state) and
                    event["policy_sha256"] == actor.sha, "trace identity")
        pool, ids = event["pool"], event["candidate_ids"]
        known = {a["id"] for a in state["agents"]}
        run.require(ids == sorted(set(ids)) == [c["candidate_id"] for c in pool] and
            all(c["agents"] and len(set(c["agents"])) == len(c["agents"]) and set(c["agents"]) <= known
                for c in pool), "candidate identity")
        run.require(set(event["proposal_order"]) == set(ids) and len(event["proposal_order"]) == len(ids), "proposal order")
        by_id = {c["candidate_id"]: c for c in pool}
        original = [by_id[cid] for cid in event["proposal_order"]]
        feature_rows, _ = engine.realized_rows(original, state_hash=event["before"])
        index, scores, _ = score_online_candidates(feature_rows, model)
        run.require(original[index]["candidate_id"] == event["anchor_id"] and
                    all(abs(c["score"] - score) <= 1e-12 for c, score in zip(original, scores)), "anchor score")
        features = run.features_for(state, pool, engine, history, q.temperature(d), event["before"])
        features = [run.budget_features(f, d, used, runtime.feature_plan(plan)["proposal"]) for f in features]
        run.require(features == event["features"], "inherited online/offline features")
        probabilities = actor.probabilities(ids, event["anchor_id"], features)
        run.require(probabilities == event["probabilities"], "behavior probabilities")
        rng = (plan, j["registration"]["config"]["phase"], j["root"]["episode_id"], j["trial"], d)
        draw = run.stream_draw(*rng, "select")
        cid = run.select_with_draw(probabilities, draw)
        expected_action = dict(mode="explicit_neighborhood", agents=by_id[cid]["agents"], random_seed=run.stream_draw(*rng, "pp"))
        run.require(event["action"] == expected_action and event["selected_id"] == cid and event["selection_draw"] == draw and
                    event["behavior_log_probability"] == math.log(probabilities[cid]), "action stream")
        run.require(event["temperature"] == q.temperature(d) and event["uniform"] == run.stream_draw(*rng, "accept"), "SA stream")
        after = q.apply_state_delta(state, event["delta"])
        q.validate_transition(state, after, event["metrics"], expected_action["agents"], "annealed", event["temperature"], event["uniform"])
        if pp_incomplete(state, after, event["metrics"]):
            run.require(row["stop"] == "incomplete_pp" and d + 1 == row["absolute_decisions"], "incomplete PP")
        history.observe(state, event, after)
        state = after
        minimum = min(minimum, state["num_of_colliding_pairs"])
    validate_final(state)
    final = run.read_json(folder(j) / "final.json")
    run.require(q.state_fingerprint(state) == q.state_fingerprint(final) == row["final_fingerprint"], "final state")
    run.require(history.decision == row["absolute_decisions"] and
        state["low_level"]["generated"] - initial_nodes == row["generated"] and
        state["num_of_colliding_pairs"] == row["final_conflicts"] and state["feasible"] == row["success"], "terminal accounting")
    paths = [a["path"] for a in state["agents"]]
    run.require(row["soc"] == sum(len(p)-1 for p in paths) and row["makespan"] == max(len(p)-1 for p in paths) and
                row["wait_steps"] == sum(a == b for p in paths for a, b in zip(p, p[1:])), "path quality")
    runtime.validate_terminal(dict(row, decisions=history.decision), plan["proposal"])
    return dict(status="ok", job_id=j["job_id"], steps=history.decision-j["root"]["root"]["decision"],
                minimum_conflicts=minimum, result_sha256=run.sha256_file(folder(j) / "result.json"))


def support_summary(roots, rows, trials):
    run.require(len(rows) == len(roots) * trials and len({r["job_id"] for r in rows}) == len(rows), "complete branch grid")
    result = []
    for root in roots:
        values = [r for r in rows if r["root_episode"] == root["episode_id"]]
        run.require(sorted(r["trial"] for r in values) == list(range(trials)), "trial coverage")
        run.require(all(r["status"] == "ok" and r["stop"] in {"feasible", "node_budget"} for r in values), "censored is unknown")
        n = sum(r["success"] for r in values)
        result.append(dict(map_id=root["map_id"], root_episode=root["episode_id"], successes=n, trials=trials,
            mixed=0 < n < trials, source_success=root["source_success"], root=root["root"],
            branches=sorted(values, key=lambda r: r["trial"])))
    return dict(roots=result, completed=sum(r["successes"] for r in result), branches=len(rows),
        mixed_roots=sum(r["mixed"] for r in result), recovered_failed_sources=sum(not r["source_success"] and r["successes"] > 0 for r in result),
        decision="conditional_terminal_support_not_controller_improvement" if any(r["mixed"] or
            (not r["source_success"] and r["successes"] > 0) for r in result) else "no_new_mixed_terminal_support_stop_probe",
        no_training=True, no_ttf=True, automatic_promotion=False)


def analyze():
    from experiments.repair_collection import _run_jobs
    reg, out = verify()
    jobs = jobs_for(reg, "branches")
    complete = run.check_seal(run.read_json(out / "branches.complete.json"))
    run.require(complete["binding"] == reg["binding"] and complete["jobs"] == len(jobs), "complete branches required")
    for j in jobs:
        run.require(complete["files"][j["job_id"]] == run.sha256_file(folder(j) / "result.json"), "completed result changed")
    with recovery.strict_lock(out, reg["binding"], "support-audit"):
        audit_path = out / "audit.json"
        if audit_path.exists():
            audit = run.check_seal(run.read_json(audit_path))
            results = audit["results"]
            run.require(audit["binding"] == reg["binding"] and len(results) == len(jobs) and
                {r["job_id"]: r["result_sha256"] for r in results} == complete["files"], "audit identity")
        else:
            results = _run_jobs(audit_worker, jobs, reg["config"]["workers"], phase="support-audit",
                output_root=out / "audit-progress", run_fingerprint=reg["binding"], timeout_seconds=960.)
            run.require(len(results) == len(jobs) and all(r["status"] == "ok" for r in results), "full trace audit")
            run.once(audit_path, run.sealed(dict(binding=reg["binding"], results=results)))
        rows = [read_result(j) for j in jobs]
        report = support_summary(reg["roots"], rows, reg["config"]["trials"])
        report.update(binding=reg["binding"], audit_sha256=run.sha256_file(audit_path),
                      collection_sha256=run.sha256_file(out / "branches.complete.json"), audit=results)
        report = json.loads(json.dumps(report))
        run.once(out / "report.json", run.sealed(report))
        run.write_json(out / "run_status.json", dict(status="complete", report_sha256=run.sha256_file(out / "report.json")))
    return {k: report[k] for k in ("completed", "branches", "mixed_roots", "recovered_failed_sources", "decision")}


def collect(phase, resume=False):
    reg, out = verify()
    if phase == "branches":
        proof = run.check_seal(run.read_json(out / "control.complete.json"))
        controls = jobs_for(reg, "control")
        run.require(proof["binding"] == reg["binding"] and proof["jobs"] == len(controls), "controls first")
        for j in controls:
            run.require(read_result(j)["stop"] == "control_validated" and proof["files"][j["job_id"]] ==
                        run.sha256_file(folder(j) / "result.json"), "control proof")
    with recovery.strict_lock(out, reg["binding"], phase):
        done = batch.execute_batches(reg, out, jobs_for(reg, phase), worker, folder, read_result,
                                     phase, resume, reg["config"]["process_fuse_seconds"])
    return dict(status="complete" if done else "paused_or_needs_inspection", phase=phase)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "verify", "dry-run", "control", "branches", "analyze", "stop"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.phase == "prepare":
        result = prepare()
    elif args.phase in {"control", "branches"}:
        result = collect(args.phase, args.resume)
    elif args.phase == "analyze":
        result = analyze()
    else:
        reg, out = verify()
        if args.phase == "stop":
            run.write_json(out / "STOP_AFTER_BATCH", dict(requested=True))
            result = dict(safe_stop_after_current_batch=True)
        else:
            result = dry_run(reg)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
