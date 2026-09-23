"""One fixed work-budget extension of four previously audited M04 suffixes."""
import argparse
from collections import Counter
import gzip
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_sa_original_terminal_support as support

run, runtime, recovery = support.run, support.runtime, support.recovery
CONFIG = "configs/sa_terminal_budget_diagnostic.json"
CODE = (CONFIG, "scripts/probe_sa_terminal_budget.py",
        "tests/evaluation/test_sa_terminal_budget.py",
        "docs/SA_TERMINAL_BUDGET_DIAGNOSTIC_PROTOCOL_ZH.md")
METRIC_CLOCKS = frozenset(("runtime_before", "runtime_after", "step_runtime", "episode_runtime_delta_seconds",
    "native_step_seconds", "native_neighborhood_generation_seconds", "native_replan_seconds", "pp_replan_seconds",
    "native_state_snapshot_seconds", "native_repair_bookkeeping_seconds", "native_residual_seconds",
    "binding_solver_call_seconds", "binding_state_snapshot_seconds", "state_to_python_seconds",
    "metrics_to_python_seconds", "binding_total_seconds", "binding_residual_seconds"))


def config():
    cfg = run.read_json(ROOT / CONFIG)
    fixed = dict(original_node_budget=25000000, node_budget=30000000, max_decisions=None,
                 trials=4, workers=4, episode_safety_seconds=900., process_fuse_seconds=1860.,
                 training=False, formal_ttf=False, automatic_promotion=False)
    run.require(all(cfg[k] == v for k, v in fixed.items()), "fixed budget diagnostic scope")
    return cfg


def stop_config(cfg):
    return dict(node_budget=cfg["node_budget"], max_decisions=None)


def source_jobs(source, cfg):
    run.require(source["config"]["node_budget"] == cfg["original_node_budget"], "original budget")
    jobs = [j for j in support.jobs_for(source, "branches") if j["root"]["episode_id"] == cfg["root_episode"]]
    run.require(len(jobs) == 4 and sorted(j["trial"] for j in jobs) == list(range(4)) and
                all(j["root"]["split"] == "train" for j in jobs), "all four original Train suffixes")
    return jobs


def prepare():
    cfg = config()
    source, source_out = support.verify()
    run.require(source_out == ROOT / cfg["source"], "source directory")
    audit = run.check_seal(run.read_json(source_out / "audit.json"))
    report = run.check_seal(run.read_json(source_out / "report.json"))
    run.require(audit["binding"] == report["binding"] == source["binding"], "audited source")
    jobs = source_jobs(source, cfg)
    records = []
    inputs = {n: run.sha256_file(ROOT / n) for n in CODE}
    # Pin current amended audit code as well as the untouched scientific inputs.
    inputs.update({n: run.sha256_file(ROOT / n) for n in source["inputs"]})
    for p in source_out.glob("*.json"):
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    proofs = {r["job_id"]: r for r in audit["results"]}
    for j in jobs:
        row = support.read_result(j)
        run.require(row["status"] == "ok" and row["stop"] == "node_budget" and not row["success"] and
                    cfg["original_node_budget"] <= row["generated"] < cfg["node_budget"], "failed original budget branch")
        path = support.folder(j)
        run.require(proofs[j["job_id"]]["status"] == "ok" and proofs[j["job_id"]]["result_sha256"] ==
                    run.sha256_file(path / "result.json"), "branch audit proof")
        for p in path.iterdir():
            if p.is_file():
                inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
        records.append(dict(source_job_id=j["job_id"], trial=j["trial"], row=row))
    out = ROOT / cfg["output"]
    run.require(not out.exists(), "existing output requires inspection")
    body = dict(config=cfg, source=source, records=records, inputs=inputs,
                source_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                no_training=True, no_ttf=True, root_selection_exploratory=True)
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out, body["binding"], "budget-prepare"):
        run.once(out / "registration.json", run.sealed(body))
    return dry_run(body)


def verify():
    cfg = config()
    out = ROOT / cfg["output"]
    reg = run.check_seal(run.read_json(out / "registration.json"))
    run.require(reg["config"] == cfg and reg["binding"] == run.json_fingerprint(
        {k: v for k, v in reg.items() if k not in {"binding", "integrity"}}), "registration identity")
    source, _ = support.verify()
    run.require(source == reg["source"], "source registration identity")
    for p, sha in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT, p, field="budget input")) == sha, "input changed: " + p)
    return reg, out


def dry_run(reg):
    return dict(jobs=len(reg["records"]), workers=reg["config"]["workers"],
        replay_repairs=sum(r["row"]["absolute_decisions"] for r in reg["records"]),
        added_node_allowance=sum(reg["config"]["node_budget"] - r["row"]["generated"] for r in reg["records"]),
        original_successes=0, max_decisions=None, no_ttf=True, rows=[dict(trial=r["trial"],
            boundary_conflicts=r["row"]["final_conflicts"], boundary_generated=r["row"]["generated"],
            boundary_decision=r["row"]["absolute_decisions"]) for r in reg["records"]])


def jobs_for(reg):
    old = {j["job_id"]: j for j in source_jobs(reg["source"], reg["config"])}
    return [dict(job_id=run.json_fingerprint([reg["binding"], r["source_job_id"]])[:24],
                 registration=reg, record=r, source_job=old[r["source_job_id"]], parent_pid=os.getpid())
            for r in reg["records"]]


def folder(j):
    return ROOT / j["registration"]["config"]["output"] / "extensions" / j["job_id"]


def event_science(event):
    return dict(event, metrics={k: v for k, v in event["metrics"].items() if k not in METRIC_CLOCKS})


def physical_signature(state):
    return run.json_fingerprint(sorted((a["id"], a["path"]) for a in state["agents"]))


def prefix_replay(j):
    from experiments.sa_onpolicy_noop_runtime import pp_incomplete
    old = j["source_job"]
    q, env, state, ctx, h, selector, engine, actor, plan, _ = support.replay(old)
    for expected in run.trace_read(support.folder(old)):
        event, after = support.execute_step(old, q, env, state, ctx, h, selector, engine, actor, plan,
                                            plan["proposal"]["pp_safety_seconds"])
        run.require(event_science(event) == event_science(expected) and
                    not pp_incomplete(state, after, event["metrics"]), "original suffix science changed")
        h.observe(state, event, after)
        state = after
    row = j["record"]["row"]
    run.require(q.state_fingerprint(state) == row["final_fingerprint"] and h.decision == row["absolute_decisions"] and
                state["low_level"]["generated"] - ctx["initial_nodes"] == row["generated"], "boundary mismatch")
    return q, env, state, ctx, h, selector, engine, actor, plan


def quality(state):
    paths = [a["path"] for a in state["agents"]]
    return dict(soc=sum(len(p)-1 for p in paths), makespan=max(len(p)-1 for p in paths),
                wait_steps=sum(a == b for p in paths for a, b in zip(p, p[1:])))


def worker(j):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.sa_onpolicy_noop_runtime import pp_incomplete
    die_with_parent(j["parent_pid"])
    cfg, out = j["registration"]["config"], folder(j)
    run.require(not out.exists(), "partial work requires inspection")
    q, env, state, ctx, h, selector, engine, actor, plan = prefix_replay(j)
    run.once(out / "boundary.json", state)
    receipt = dict(fingerprint=q.state_fingerprint(state), history=support.history_signature(h),
                   replay_repairs=h.decision, generated=state["low_level"]["generated"] - ctx["initial_nodes"])
    run.once(out / "prefix_receipt.json", receipt)
    began = time.monotonic()
    with gzip.open(out / "trace.jsonl.gz", "xt", encoding="utf8", compresslevel=3) as f:
        while True:
            used = state["low_level"]["generated"] - ctx["initial_nodes"]
            stop = runtime.work_stop(state["feasible"], h.decision, used, stop_config(cfg))
            if stop:
                break
            remaining = cfg["episode_safety_seconds"] - (time.monotonic() - began)
            if remaining <= 0:
                stop = "wall_safety"
                break
            event, after = support.execute_step(j["source_job"], q, env, state, ctx, h, selector, engine, actor, plan, remaining)
            f.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")
            f.flush()
            incomplete = pp_incomplete(state, after, event["metrics"])
            h.observe(state, event, after)
            state = after
            if incomplete:
                stop = "incomplete_pp"
                break
    validate_final(state)
    run.once(out / "final.json", state)
    row = dict(binding=j["registration"]["binding"], job_id=j["job_id"], source_job_id=j["record"]["source_job_id"],
        trial=j["record"]["trial"], policy_sha256=actor.sha, status="censored" if stop in {"wall_safety", "incomplete_pp"} else "ok",
        stop=stop, success=state["feasible"], original_budget_success=False, no_ttf=True, no_training=True,
        split="train_budget_diagnostic", generated=state["low_level"]["generated"] - ctx["initial_nodes"],
        decisions=h.decision, extension_steps=h.decision-receipt["replay_repairs"],
        extension_generated=state["low_level"]["generated"] - ctx["initial_nodes"] - receipt["generated"],
        final_fingerprint=q.state_fingerprint(state), final_conflicts=state["num_of_colliding_pairs"], **quality(state),
        files={p.name: run.sha256_file(p) for p in out.iterdir() if p.is_file()})
    runtime.validate_terminal(row, stop_config(cfg))
    run.once(out / "result.json", run.sealed(row))
    return {k: row[k] for k in ("job_id", "status", "stop", "success", "decisions", "final_conflicts")}


def read_result(j):
    row = run.result_read(folder(j), j["registration"])
    run.require(row["job_id"] == j["job_id"] and row["source_job_id"] == j["record"]["source_job_id"] and
        row["trial"] == j["record"]["trial"] and row["policy_sha256"] == j["registration"]["source"]["parent_policy"] and
        row["split"] == "train_budget_diagnostic" and row["original_budget_success"] is False, "result identity")
    runtime.validate_terminal(row, stop_config(j["registration"]["config"]))
    return row


def audit_worker(j):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    from experiments.online_feature_engine import OnlineFeatureEngine
    from experiments.compact_controller_model import load_controller_bundle
    from lns2_selector.runtime.online_selection import score_online_candidates
    from experiments.sa_onpolicy_noop_runtime import pp_incomplete
    die_with_parent(j["parent_pid"])
    row, old, cfg = read_result(j), j["source_job"], j["registration"]["config"]
    plan, _, actor = support.context(old)
    q = run.native_runtime(plan)
    source = ROOT / old["root"]["folder"]
    state = run.read_json(source / "initial.json")
    initial_nodes = state["low_level"]["generated"]
    h = support.History(state)
    for e in run.trace_read(source):
        if h.decision == old["root"]["root"]["decision"]:
            break
        run.require(e["before"] == q.state_fingerprint(state) and e["decision"] == h.decision, "source prefix")
        after = q.apply_state_delta(state, e["delta"])
        h.observe(state, e, after)
        state = after
    for e in run.trace_read(support.folder(old)):
        run.require(e["before"] == q.state_fingerprint(state) and e["decision"] == h.decision, "source suffix")
        after = q.apply_state_delta(state, e["delta"])
        h.observe(state, e, after)
        state = after
    receipt = run.read_json(folder(j) / "prefix_receipt.json")
    run.require(receipt == dict(fingerprint=q.state_fingerprint(state), history=support.history_signature(h),
        replay_repairs=h.decision, generated=state["low_level"]["generated"] - initial_nodes), "prefix receipt")
    run.require(receipt["fingerprint"] == j["record"]["row"]["final_fingerprint"] ==
                q.state_fingerprint(run.read_json(folder(j) / "boundary.json")), "old boundary")
    engine = OnlineFeatureEngine(state, backend="native")
    model = load_controller_bundle(ROOT / "artifacts/initlns-closed-loop-controller-v2").main_models["realized_dynamic"]
    minimum, curve = state["num_of_colliding_pairs"], []
    previous_path = physical_signature(state)
    paths, memberships = Counter([previous_path]), Counter()
    persistent_edges = {tuple(sorted(e)) for e in state["conflict_edges"]}
    self_loops, accepted_increases = 0, 0
    for event in run.trace_read(folder(j)):
        d, used = h.decision, state["low_level"]["generated"] - initial_nodes
        run.require(runtime.work_stop(state["feasible"], d, used, stop_config(cfg)) is None, "post-terminal action")
        run.require(event["decision"] == d and event["before"] == q.state_fingerprint(state) and
                    event["policy_sha256"] == actor.sha, "trace identity")
        pool, ids = event["pool"], event["candidate_ids"]
        known = {a["id"] for a in state["agents"]}
        run.require(ids == sorted(set(ids)) == [c["candidate_id"] for c in pool] and
            all(c["agents"] and len(set(c["agents"])) == len(c["agents"]) and set(c["agents"]) <= known for c in pool), "candidate identity")
        run.require(set(event["proposal_order"]) == set(ids) and len(event["proposal_order"]) == len(ids), "proposal order")
        by_id = {c["candidate_id"]: c for c in pool}
        original = [by_id[cid] for cid in event["proposal_order"]]
        engine.prepare(state)
        feature_rows, _ = engine.realized_rows(original, state_hash=event["before"])
        index, scores, _ = score_online_candidates(feature_rows, model)
        run.require(original[index]["candidate_id"] == event["anchor_id"] and
            all(abs(c["score"] - score) <= 1e-12 for c, score in zip(original, scores)), "anchor score")
        features = run.features_for(state, pool, engine, h, q.temperature(d), event["before"])
        features = [run.budget_features(f, d, used, runtime.feature_plan(plan)["proposal"]) for f in features]
        run.require(features == event["features"] and all(f["budget.remaining_node_fraction"] == 0 for f in features), "frozen budget features")
        probabilities = actor.probabilities(ids, event["anchor_id"], features)
        run.require(probabilities == event["probabilities"] and actor.out_of_range_fraction(features) == event["out_of_range_fraction"], "actor outputs")
        rng = (plan, old["registration"]["config"]["phase"], old["root"]["episode_id"], old["trial"], d)
        draw = run.stream_draw(*rng, "select")
        cid = run.select_with_draw(probabilities, draw)
        action = dict(mode="explicit_neighborhood", agents=by_id[cid]["agents"], random_seed=run.stream_draw(*rng, "pp"))
        run.require(event["selected_id"] == cid and event["action"] == action and event["selection_draw"] == draw and
            event["behavior_log_probability"] == math.log(probabilities[cid]), "original selection stream")
        run.require(event["temperature"] == q.temperature(d) and event["uniform"] == run.stream_draw(*rng, "accept"), "original SA stream")
        after = q.apply_state_delta(state, event["delta"])
        q.validate_transition(state, after, event["metrics"], action["agents"], "annealed", event["temperature"], event["uniform"])
        if pp_incomplete(state, after, event["metrics"]):
            run.require(row["stop"] == "incomplete_pp" and d+1 == row["decisions"], "incomplete PP")
        signature = physical_signature(after)
        self_loops += signature == previous_path
        paths[signature] += 1
        memberships[run.json_fingerprint(sorted(action["agents"]))] += 1
        accepted_increases += after["num_of_colliding_pairs"] > state["num_of_colliding_pairs"]
        persistent_edges.intersection_update(tuple(sorted(e)) for e in after["conflict_edges"])
        previous_path = signature
        h.observe(state, event, after)
        state = after
        minimum = min(minimum, state["num_of_colliding_pairs"])
        curve.append(dict(decision=h.decision, generated=state["low_level"]["generated"]-initial_nodes,
                          conflicts=state["num_of_colliding_pairs"], out_of_range_fraction=event["out_of_range_fraction"]))
    validate_final(state)
    run.require(q.state_fingerprint(state) == q.state_fingerprint(run.read_json(folder(j) / "final.json")) == row["final_fingerprint"], "final state")
    run.require(state["feasible"] == row["success"] and state["num_of_colliding_pairs"] == row["final_conflicts"] and
        h.decision == row["decisions"] and state["low_level"]["generated"]-initial_nodes == row["generated"] and
        row["extension_steps"] == h.decision-receipt["replay_repairs"] and
        row["extension_generated"] == row["generated"]-receipt["generated"] and
        quality(state) == {k: row[k] for k in quality(state)}, "final accounting")
    runtime.validate_terminal(row, stop_config(cfg))
    return dict(job_id=j["job_id"], status="ok", minimum_conflicts=minimum, curve=curve,
                unique_path_configurations=len(paths), physical_self_loops=self_loops,
                recurring_nonconsecutive_states=sum(paths.values())-len(paths)-self_loops,
                unique_memberships=len(memberships), maximum_membership_count=max(memberships.values(), default=0),
                accepted_conflict_increases=accepted_increases, persistent_edges=sorted(persistent_edges),
                result_sha256=run.sha256_file(folder(j) / "result.json"))


def summary(rows):
    run.require(len(rows) == 4 and sorted(r["trial"] for r in rows) == list(range(4)), "complete trials")
    run.require(all(r["status"] == "ok" and r["stop"] in {"feasible", "node_budget"} and not r["original_budget_success"] for r in rows), "unknown is not failure")
    successes = sum(r["success"] for r in rows)
    return dict(original_25m_successes=0, extended_30m_successes=successes, trials=4, independent_roots=1,
        decision="conditional_extra_work_witness_not_model_gain" if successes else "no_recovery_with_fixed_extra_work_stop_budget_scan",
        no_training=True, no_ttf=True, automatic_promotion=False)


def collect(resume=False):
    reg, out = verify()
    with recovery.strict_lock(out, reg["binding"], "extension"):
        done = support.batch.execute_batches(reg, out, jobs_for(reg), worker, folder, read_result,
            "extensions", resume, reg["config"]["process_fuse_seconds"])
    return dict(status="complete" if done else "paused_or_needs_inspection")


def analyze():
    from experiments.repair_collection import _run_jobs
    reg, out = verify()
    jobs = jobs_for(reg)
    complete = run.check_seal(run.read_json(out / "extensions.complete.json"))
    run.require(complete["binding"] == reg["binding"] and complete["jobs"] == len(jobs) and
        complete["files"] == {j["job_id"]: run.sha256_file(folder(j) / "result.json") for j in jobs}, "completion identity")
    with recovery.strict_lock(out, reg["binding"], "audit"):
        if (out / "audit.json").exists():
            proof = run.check_seal(run.read_json(out / "audit.json"))
            run.require(proof["binding"] == reg["binding"] and {r["job_id"]: r["result_sha256"] for r in proof["results"]} == complete["files"], "audit identity")
            audited = proof["results"]
        else:
            audited = _run_jobs(audit_worker, jobs, 4, phase="extension-audit", output_root=out / "audit-progress",
                run_fingerprint=reg["binding"], timeout_seconds=960.,
                failure_result=lambda j, status, error: dict(job_id=j["job_id"], status=status, error=error))
            run.once(out / "audit-attempt.json", run.sealed(dict(binding=reg["binding"], results=audited)))
            run.require(len(audited) == 4 and all(r["status"] == "ok" for r in audited), "complete extension audit")
            run.once(out / "audit.json", run.sealed(dict(binding=reg["binding"], results=audited)))
        rows = [read_result(j) for j in jobs]
        report = dict(summary(rows), binding=reg["binding"], rows=rows, audit=audited,
                      source_rows=reg["records"], audit_sha256=run.sha256_file(out / "audit.json"))
        run.once(out / "report.json", run.sealed(json.loads(json.dumps(report))))
        run.write_json(out / "run_status.json", dict(status="complete", report_sha256=run.sha256_file(out / "report.json")))
    return summary(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "verify", "dry-run", "collect", "analyze", "stop"))
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if args.phase == "prepare":
        result = prepare()
    elif args.phase == "collect":
        result = collect(args.resume)
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
