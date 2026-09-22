"""Uncapped evaluation of frozen actors; historical training remains immutable."""
import gzip
import json
import math
import time

from scripts import run_sa_onpolicy as run
from experiments.sa_onpolicy_noop_runtime import pp_incomplete


def feature_plan(plan):
    cfg = plan["proposal"]
    run.require(cfg["max_decisions"] is None and cfg["decision_feature_reference"] == 256,
                "uncapped execution / frozen feature contract")
    # This reference is an input scaling constant, never a stopping condition.
    return dict(plan, proposal=dict(cfg, max_decisions=cfg["decision_feature_reference"]))


def work_stop(feasible, decision, generated, cfg):
    run.require(cfg["max_decisions"] is None and decision >= 0 and generated >= 0, "work contract")
    if feasible:
        return "feasible"
    if generated >= cfg["node_budget"]:
        return "node_budget"
    return None


def validate_terminal(row, cfg):
    run.require(row["decisions"] >= 0 and row["generated"] >= 0, "negative work")
    expected = work_stop(row["success"], row["decisions"], row["generated"], cfg)
    if row["status"] == "ok":
        run.require(expected == row["stop"] and expected in ("feasible", "node_budget"), "terminal work mismatch")
    else:
        run.require(row["status"] == "censored" and row["stop"] in ("wall_safety", "incomplete_pp"), "unknown censor")


def episode_loop(job, q, env, state, ctx, folder, bundle):
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    plan, cfg = job["plan"], job["plan"]["proposal"]
    features_plan = feature_plan(plan)
    initial = state
    ctx = dict(ctx, initial_nodes=state["low_level"]["generated"])
    history = History(state)
    actor = run.NumpyActor(bundle) if job["arm"] in {"trained_actor", "untrained_exploration"} else None
    selector = q.SingleFullCheckPool(ctx) if job["arm"] != "official_sa" else None
    engine = OnlineFeatureEngine(state, backend="native") if selector else None
    run.once(folder / "initial.json", state)
    began = time.monotonic()
    with gzip.open(folder / "trace.jsonl.gz", "xt", encoding="utf8", compresslevel=3) as stream:
        while True:
            d = history.decision
            used = state["low_level"]["generated"] - ctx["initial_nodes"]
            stop = work_stop(state["feasible"], d, used, cfg)
            if stop:
                break
            if time.monotonic() - began >= cfg["episode_safety_seconds"]:
                stop = "wall_safety"
                break
            event = run.selection(env, state, ctx, history, engine, selector, features_plan, actor) if selector else {}
            draw = run.stream_draw(plan, job["phase"], job["pair_id"], job["replica"], d, "select")
            if selector:
                cid = run.select_with_draw(event["probabilities"], draw) if actor else event["anchor_id"]
                chosen = next(c for c in event["pool"] if c["candidate_id"] == cid)
                action = dict(mode="explicit_neighborhood", agents=chosen["agents"],
                              random_seed=run.stream_draw(plan, job["phase"], job["pair_id"], job["replica"], d, "pp"))
                event.update(selected_id=cid, behavior_log_probability=math.log(event["probabilities"][cid]), selection_draw=draw)
            else:
                action = dict(mode="official")
            remaining = cfg["episode_safety_seconds"] - (time.monotonic() - began)
            if remaining <= 0:
                stop = "wall_safety"
                break
            temp = q.temperature(d)
            uniform = run.stream_draw(plan, job["phase"], job["pair_id"], job["replica"], d, "accept")
            raw = q._plain(env.step_experimental_pp(action, min(cfg["pp_safety_seconds"], remaining), "annealed", temp, uniform))
            after, metrics = raw["observation"], raw["metrics"]
            selected = action.get("agents", metrics["neighborhood"])
            q.validate_transition(state, after, metrics, selected, "annealed", temp, uniform)
            incomplete = pp_incomplete(state, after, metrics)
            event.update(decision=d, policy_sha256=actor.sha if actor else job["arm"],
                         before=q.state_fingerprint(state), action=action, temperature=temp, uniform=uniform,
                         metrics=metrics, delta=q.encode_state_delta(state, after))
            stream.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()
            history.observe(state, event, after)
            state = after
            if incomplete:
                stop = "incomplete_pp"
                break
    validate_final(state)
    paths = [a["path"] for a in state["agents"]]
    run.require(sum(len(p) - 1 for p in paths) == state["sum_of_costs"], "SOC reconstruction")
    run.once(folder / "final.json", state)
    status = "censored" if stop in {"wall_safety", "incomplete_pp"} else "ok"
    row = dict(binding=plan["binding"], execution_binding=job["comparison_binding"], status=status,
               job_id=job["job_id"], episode_id=job["job_id"], pair_id=job["pair_id"],
               map_id=job["case"]["map_id"], split=job["split"], replica=job["replica"], arm=job["arm"],
               policy_sha256=run.validate_bundle(bundle) if bundle else job["arm"],
               rng_stream_id=run.json_fingerprint([plan["config"]["stream_seed"], job["phase"], job["pair_id"], job["replica"]]),
               initial_fingerprint=q.state_fingerprint(initial), final_fingerprint=q.state_fingerprint(state),
               stop=stop, success=state["feasible"], final_conflicts=state["num_of_colliding_pairs"],
               decisions=history.decision, generated=state["low_level"]["generated"] - ctx["initial_nodes"],
               soc=state["sum_of_costs"], makespan=max(len(p) - 1 for p in paths),
               wait_steps=sum(a == b for p in paths for a, b in zip(p, p[1:])),
               seconds_diagnostic=time.monotonic() - began, no_ttf=True,
               files={n: run.sha256_file(folder / n) for n in ("initial.json", "final.json", "trace.jsonl.gz")})
    validate_terminal(row, cfg)
    run.once(folder / "result.json", run.sealed(row))
    return row


def episode_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job["parent_pid"])
    plan = job["plan"]
    out = run.ROOT / plan["config"]["output"]
    folder = run.folder_for(out, job["phase"], job)
    run.require(not folder.exists(), "partial episode requires inspection")
    q, env, state, ctx = run.reset(job)
    run.require(q.state_fingerprint(state) == job["expected_initial"], "paired initial mismatch")
    bundle = run.actor_load(out, plan, job["iteration"]) if job["arm"] in {"trained_actor", "untrained_exploration"} else None
    row = episode_loop(job, q, env, state, ctx, folder, bundle)
    return {k: row[k] for k in ("status", "job_id", "stop", "success", "decisions")}


def audit_worker(job):
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    from scripts.run_feedback_exploration_diagnostics import validate_final
    p = job["plan"]
    fp = feature_plan(p)
    out = run.ROOT / p["config"]["output"]
    q = run.native_runtime(p)
    folder = run.folder_for(out, job["phase"], job)
    result = run.result_read(folder, p)
    run.require(result["execution_binding"] == job["comparison_binding"], "execution identity")
    state = run.read_json(folder / "initial.json")
    history = History(state)
    initial_nodes = state["low_level"]["generated"]
    actor = run.NumpyActor(run.actor_load(out, p, job["iteration"])) if job["arm"] in {"trained_actor", "untrained_exploration"} else None
    engine = OnlineFeatureEngine(state, backend="native") if job["arm"] != "official_sa" else None
    if engine:
        from experiments.compact_controller_model import load_controller_bundle
        from lns2_selector.runtime.online_selection import score_online_candidates
        anchor_model = load_controller_bundle(run.ROOT / "artifacts/initlns-closed-loop-controller-v2").main_models["realized_dynamic"]
    run.require(q.state_fingerprint(state) == result["initial_fingerprint"] == job["expected_initial"], "audit initial")
    noops = 0
    for event in run.trace_read(folder):
        d = history.decision
        used = state["low_level"]["generated"] - initial_nodes
        run.require(work_stop(state["feasible"], d, used, p["proposal"]) is None, "action after terminal")
        run.require(event["decision"] == d and event["before"] == q.state_fingerprint(state), "audit trace order")
        run.require(event["policy_sha256"] == result["policy_sha256"], "trace policy")
        if engine:
            pool = event["pool"]
            ids = [c["candidate_id"] for c in pool]
            run.require(ids == sorted(set(ids)) == event["candidate_ids"], "audit pool IDs")
            known = {a["id"] for a in state["agents"]}
            run.require(all(c["agents"] and len(c["agents"]) == len(set(c["agents"])) and set(c["agents"]) <= known for c in pool), "candidate agents")
            fs = run.features_for(state, pool, engine, history, q.temperature(d), event["before"])
            run.require(set(event["proposal_order"]) == set(ids) and len(event["proposal_order"]) == len(ids), "proposal order")
            lookup = {c["candidate_id"]: c for c in pool}
            original = [lookup[cid] for cid in event["proposal_order"]]
            score_rows, _ = engine.realized_rows(original, state_hash=event["before"])
            index, scores, _ = score_online_candidates(score_rows, anchor_model)
            run.require(original[index]["candidate_id"] == event["anchor_id"] and all(abs(c["score"]-s) <= 1e-12 for c, s in zip(original, scores)), "anchor score mismatch")
            supplied = [run.budget_features(f, d, used, fp["proposal"]) for f in fs]
            run.require(supplied == event["features"], "online/offline feature mismatch")
            probabilities = actor.probabilities(ids, event["anchor_id"], supplied) if actor else {c: float(c == event["anchor_id"]) for c in ids}
            run.require(probabilities == event["probabilities"], "actor probabilities")
            draw = run.stream_draw(p, job["phase"], job["pair_id"], job["replica"], d, "select")
            cid = run.select_with_draw(probabilities, draw) if actor else event["anchor_id"]
            selected = next(c["agents"] for c in pool if c["candidate_id"] == cid)
            run.require(cid == event["selected_id"] and selected == event["action"]["agents"] and event["selection_draw"] == draw, "selection")
            run.require(event["behavior_log_probability"] == math.log(probabilities[cid]), "behavior likelihood")
            run.require(event["action"]["random_seed"] == run.stream_draw(p, job["phase"], job["pair_id"], job["replica"], d, "pp"), "PP RNG")
        else:
            run.require(event["action"] == {"mode": "official"}, "official RNG changed")
            selected = event["metrics"]["neighborhood"]
        run.require(event["temperature"] == q.temperature(d) and event["uniform"] == run.stream_draw(p, job["phase"], job["pair_id"], job["replica"], d, "accept"), "SA stream")
        after = q.apply_state_delta(state, event["delta"])
        q.validate_transition(state, after, event["metrics"], selected, "annealed", event["temperature"], event["uniform"])
        if pp_incomplete(state, after, event["metrics"]):
            run.require(d + 1 == result["decisions"] and result["stop"] == "incomplete_pp", "continued incomplete PP")
        noops += event["metrics"]["pp_failure_reason"] == "not_run"
        history.observe(state, event, after)
        state = after
    validate_final(state)
    final = run.read_json(folder / "final.json")
    run.require(q.state_fingerprint(state) == q.state_fingerprint(final) == result["final_fingerprint"] and
                {k:v for k,v in state.items() if k not in {"runtime", "context"}} ==
                {k:v for k,v in final.items() if k not in {"runtime", "context"}}, "final paths")
    run.require(history.decision == result["decisions"] and state["low_level"]["generated"]-initial_nodes == result["generated"], "audit work")
    run.require(state["num_of_colliding_pairs"] == result["final_conflicts"] and state["feasible"] == result["success"], "audit outcome")
    paths = [a["path"] for a in final["agents"]]
    run.require(result["soc"] == sum(len(p)-1 for p in paths) and result["makespan"] == max(len(p)-1 for p in paths) and
                result["wait_steps"] == sum(a == b for p in paths for a,b in zip(p,p[1:])), "quality summary")
    run.require(result["rng_stream_id"] == run.json_fingerprint([p["config"]["stream_seed"], job["phase"], job["pair_id"], job["replica"]]), "RNG identity")
    validate_terminal(result, p["proposal"])
    return dict(status="ok", job_id=job["job_id"], legal_noops=noops, result_sha256=run.sha256_file(folder / "result.json"))
