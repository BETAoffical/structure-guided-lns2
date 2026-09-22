"""Versioned engine copy: distinguish a legal native no-op from interrupted PP.

The original engine remains hash-frozen for training reproducibility. An AST
regression test restricts this copy to the interruption-classification change.
"""
import gzip
import json
import math
import time

from scripts.run_sa_onpolicy import (NumpyActor, once, work_stop, selection, stream_draw,
    select_with_draw, require, validate_bundle, json_fingerprint, sha256_file,
    terminal_return, sealed)


def pp_incomplete(before, after, metrics):
    if metrics["pp_failure_reason"] == "time_limit":
        return True
    if metrics["acceptance_evaluated"]:
        return False
    if metrics["pp_failure_reason"] != "not_run":
        return True
    require(metrics["action_valid"] and metrics["step_applied"] and
            metrics["pp_attempted_agent_count"] == metrics["pp_inserted_agent_count"] == 0 and
            not metrics["pp_rolled_back"] and not metrics["repair_order"], "invalid native no-op")
    selected = set(metrics["neighborhood"])
    agents = {a["id"]:a for a in before["agents"]}
    require(selected <= set(agents), "no-op unknown agent")
    if metrics["generated"]:
        require(all(agents[a]["conflict_degree"] == 0 for a in selected), "generated no-op touches conflict")
    require(after["iteration"] == before["iteration"] + 1 and
            all(before[k] == after[k] for k in ("agents", "low_level", "conflict_edges", "sum_of_costs", "num_of_colliding_pairs")),
            "no-op changed paths, conflicts or search work")
    return False


def episode_loop(job, q, env, state, ctx, folder, bundle):
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.sa_history_selector import History
    plan, cfg = job["plan"], job["plan"]["proposal"]
    initial = state
    ctx = dict(ctx, initial_nodes=state["low_level"]["generated"])
    history = History(state)
    actor = NumpyActor(bundle) if job["arm"] in {"trained_actor", "untrained_exploration"} else None
    selector = q.SingleFullCheckPool(ctx) if job["arm"] != "official_sa" else None
    engine = OnlineFeatureEngine(state, backend="native") if selector else None
    once(folder / "initial.json", state)
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
            event = selection(env, state, ctx, history, engine, selector, plan, actor) if selector else {}
            draw = stream_draw(plan, job["phase"], job["pair_id"], job["replica"], d, "select")
            if selector:
                cid = select_with_draw(event["probabilities"], draw) if actor else event["anchor_id"]
                chosen = next(c for c in event["pool"] if c["candidate_id"] == cid)
                action = dict(mode="explicit_neighborhood", agents=chosen["agents"],
                              random_seed=stream_draw(plan, job["phase"], job["pair_id"], job["replica"], d, "pp"))
                event.update(selected_id=cid, behavior_log_probability=math.log(event["probabilities"][cid]), selection_draw=draw)
            else:
                action = dict(mode="official")
            remaining = cfg["episode_safety_seconds"] - (time.monotonic() - began)
            if remaining <= 0:
                stop = "wall_safety"
                break
            temp = q.temperature(d)
            uniform = stream_draw(plan, job["phase"], job["pair_id"], job["replica"], d, "accept")
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
    require(sum(len(p) - 1 for p in paths) == state["sum_of_costs"], "SOC reconstruction")
    once(folder / "final.json", state)
    status = "censored" if stop in {"wall_safety", "incomplete_pp"} else "ok"
    row = dict(binding=plan["binding"], status=status, job_id=job["job_id"], episode_id=job["job_id"],
               pair_id=job["pair_id"], map_id=job["case"]["map_id"], split=job["split"], replica=job["replica"],
               arm=job["arm"], policy_sha256=validate_bundle(bundle) if bundle else job["arm"],
               rng_stream_id=json_fingerprint([plan["config"]["stream_seed"], job["phase"], job["pair_id"], job["replica"]]),
               initial_fingerprint=q.state_fingerprint(initial), final_fingerprint=q.state_fingerprint(state),
               stop=stop, success=state["feasible"], final_conflicts=state["num_of_colliding_pairs"],
               decisions=history.decision, generated=state["low_level"]["generated"] - ctx["initial_nodes"],
               soc=state["sum_of_costs"], makespan=max(len(p) - 1 for p in paths),
               wait_steps=sum(a == b for p in paths for a, b in zip(p, p[1:])),
               seconds_diagnostic=time.monotonic() - began, no_ttf=True,
               files={n: sha256_file(folder / n) for n in ("initial.json", "final.json", "trace.jsonl.gz")})
    terminal_return(row, max_decisions=cfg["max_decisions"], node_budget=cfg["node_budget"])
    once(folder / "result.json", sealed(row))
    return row


def episode_worker(job):
    from scripts import run_sa_onpolicy as run
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job["parent_pid"])
    plan = job["plan"]
    out = run.ROOT / plan["config"]["output"]
    folder = run.folder_for(out, job["phase"], job)
    require(not folder.exists(), "partial episode requires inspection, never silent retry")
    q, env, state, ctx = run.reset(job)
    require(q.state_fingerprint(state) == job["expected_initial"], "paired initial mismatch")
    bundle = run.actor_load(out, plan, job["iteration"]) if job["arm"] in {"trained_actor", "untrained_exploration"} else None
    row = episode_loop(job, q, env, state, ctx, folder, bundle)
    return {k: row[k] for k in ("status", "job_id", "stop", "success", "decisions")}
