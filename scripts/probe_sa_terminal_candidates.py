"""Existing-candidate feasibility at first occurrences of three terminal pairs."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_sa_terminal_budget as identity
from scripts import run_sa_path_quality as q
from scripts.run_feedback_exploration_diagnostics import validate_final
from scripts.train_sa_history_selector import die_with_parent
from experiments.sa_onpolicy_noop_runtime import pp_incomplete

budget, run = identity.probe, identity.run
CONFIG = "configs/sa_terminal_candidates.json"
CODE = (CONFIG, "scripts/probe_sa_terminal_candidates.py", "tests/evaluation/test_sa_terminal_candidates.py",
        "docs/SA_TERMINAL_CANDIDATES_PROTOCOL_ZH.md")


def config():
    cfg = run.read_json(ROOT / CONFIG)
    fixed = dict(trials=4, workers=20, pp_seconds=20., branch_fuse_seconds=90., group_fuse_seconds=1860.,
                 max_decisions=None, training=False, formal_ttf=False, automatic_expansion=False)
    run.require(all(cfg[k] == v for k, v in fixed.items()), "fixed candidate scope")
    return cfg


def select_targets(occurrences):
    selected = {}
    for r in sorted(occurrences, key=lambda r: (r["trial"], r["decision"])):
        run.require(r["split"] == "train" and len(r["edge"]) == 2 and r["edge"][0] != r["edge"][1], "Train pair only")
        key = (r["root_episode"], tuple(sorted(r["edge"])))
        selected.setdefault(key, r)
    return list(selected.values())


def boundary_history(job):
    old = job["source_job"]
    source = ROOT / old["root"]["folder"]
    state = run.read_json(source / "initial.json")
    history = budget.support.History(state)
    for event in run.trace_read(source):
        if history.decision == old["root"]["root"]["decision"]: break
        after = q.apply_state_delta(state, event["delta"])
        history.observe(state, event, after)
        state = after
    for event in run.trace_read(budget.support.folder(old)):
        after = q.apply_state_delta(state, event["delta"])
        history.observe(state, event, after)
        state = after
    receipt = run.read_json(budget.folder(job) / "prefix_receipt.json")
    run.require(q.state_fingerprint(state) == receipt["fingerprint"] and
        budget.support.history_signature(history) == receipt["history"], "audited boundary history")
    return history


def prepare():
    cfg = config()
    source, source_out, proof = identity.verify()
    run.require(source_out == ROOT / cfg["source"] and run.sha256_file(source_out / "report.json") ==
                cfg["source_report_sha256"], "source identity")
    report = run.check_seal(run.read_json(source_out / "report.json"))
    run.require(report["binding"] == source["binding"] and report["extended_30m_successes"] == 0, "source result")
    occurrences = []
    for j in budget.jobs_for(source):
        budget.read_result(j)
        state = run.read_json(budget.folder(j) / "boundary.json")
        history = boundary_history(j)
        for e in run.trace_read(budget.folder(j)):
            run.require(e["before"] == q.state_fingerprint(state) and e["decision"] == history.decision, "source continuity")
            if state["num_of_colliding_pairs"] == 1:
                run.require(len(state["conflict_edges"]) == 1 and len(state["agents"]) == 500, "full single-pair task")
                rid = run.json_fingerprint([source["binding"], j["job_id"], e["decision"]])[:24]
                occurrences.append(dict(id=rid, source_job_id=j["job_id"], trial=j["record"]["trial"],
                    decision=e["decision"], root_episode=j["source_job"]["root"]["episode_id"], split="train",
                    edge=state["conflict_edges"][0], fingerprint=e["before"],
                    history=budget.support.history_signature(history), event=e))
            after = q.apply_state_delta(state, e["delta"])
            history.observe(state, e, after)
            state = after
    targets = select_targets(occurrences)
    run.require(len(occurrences) == 6 and len(targets) == 3 and
                all(len(t["event"]["pool"]) == 19 for t in targets), "registered state/candidate inventory")
    old_path = "build/sa-plateau-candidate-audit-v1/plan.json"
    old = run.read_json(ROOT / old_path)
    run.require(not ({t["fingerprint"] for t in targets} & {t["state_fingerprint"] for t in old["targets"]}), "duplicate historical state")
    inputs = dict(source["inputs"], **proof["files"])
    commit = run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for path, data in identity.source_python(commit).items():
        run.require(identity.same_source((ROOT / path).read_bytes(), data), "uncommitted Python: " + path)
        inputs[path] = run.sha256_file(ROOT / path)
    for n in (*CODE, old_path):
        inputs[n] = run.sha256_file(ROOT / n)
    for n in ("registration.json", "runtime_identity_supplement.json", "extensions.complete.json", "audit.json", "report.json"):
        p = source_out / n
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    for j in budget.jobs_for(source):
        for p in budget.folder(j).iterdir():
            if p.is_file(): inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    body = dict(config=cfg, source=source, occurrences=occurrences, targets=targets, inputs=inputs,
                source_commit=commit, exploratory=True, no_training=True, no_ttf=True)
    body["binding"] = run.json_fingerprint(body)
    out = ROOT / cfg["output"]
    run.require(not out.exists(), "existing output requires inspection")
    with budget.recovery.strict_lock(out, body["binding"], "candidate-prepare"):
        run.once(out / "registration.json", run.sealed(body))
    return dry_run(body)


def verify():
    cfg = config()
    out = ROOT / cfg["output"]
    reg = run.check_seal(run.read_json(out / "registration.json"))
    run.require(reg["config"] == cfg and reg["binding"] == run.json_fingerprint(
        {k: v for k, v in reg.items() if k not in {"binding", "integrity"}}), "registration identity")
    identity.verify_files(reg["inputs"])
    source, _, _ = identity.verify()
    run.require(source == reg["source"] and select_targets(reg["occurrences"]) == reg["targets"], "source/selection changed")
    return reg, out


def dry_run(reg):
    ends = {}
    for t in reg["targets"]: ends[t["source_job_id"]] = max(ends.get(t["source_job_id"], 0), t["decision"])
    return dict(occurrences=len(reg["occurrences"]), roots=len(reg["targets"]),
        candidates=sum(len(t["event"]["pool"]) for t in reg["targets"]),
        counterfactual_jobs=sum(len(t["event"]["pool"])*reg["config"]["trials"] for t in reg["targets"]),
        controls=len(reg["targets"]), replay_repairs=sum(ends.values()), workers=reg["config"]["workers"],
        independent_original_tasks=1, no_ttf=True, no_training=True,
        targets=[{k: t[k] for k in ("id", "trial", "decision", "edge")} for t in reg["targets"]])


def branch_jobs(reg, target):
    event = target["event"]
    jobs = [dict(job_id="control", candidate_id=event["selected_id"], trial=-1, action=event["action"])]
    for trial in range(reg["config"]["trials"]):
        seed = int(run.json_fingerprint([reg["config"]["seed_namespace"], target["fingerprint"], trial])[:8], 16) & 0x7fffffff
        for index, c in enumerate(event["pool"]):
            jobs.append(dict(job_id=f"c{index:02d}-t{trial}", candidate_id=c["candidate_id"], trial=trial,
                action=dict(mode="explicit_neighborhood", agents=c["agents"], random_seed=seed)))
    return jobs


def branch_folder(reg, target, job):
    return ROOT / reg["config"]["output"] / "branches" / target["id"] / job["job_id"]


def canonical_state(state):
    return {k: v for k, v in state.items() if k not in {"runtime", "context"}}


def check_branch(before, row, target, job):
    run.require(row["job"] == job and row["root_id"] == target["id"] and
                row["before"] == target["fingerprint"] == q.state_fingerprint(before), "branch identity")
    e = target["event"]
    after = q.apply_state_delta(before, row["delta"])
    q.validate_transition(before, after, row["metrics"], job["action"]["agents"], "annealed", e["temperature"], e["uniform"])
    known = {a["id"]: a["path"] for a in before["agents"]}
    selected = set(job["action"]["agents"])
    run.require(all(a["path"] == known[a["id"]] for a in after["agents"] if a["id"] not in selected), "outsider paths changed")
    validate_final(after)
    incomplete = pp_incomplete(before, after, row["metrics"])
    run.require(row["status"] == ("censored" if incomplete or row["truncated"] else "ok"), "censor semantics")
    run.require(row["after"] == q.state_fingerprint(after) and row["feasible"] == after["feasible"] and
        row["conflicts"] == after["num_of_colliding_pairs"] and row["generated"] ==
        after["low_level"]["generated"]-before["low_level"]["generated"] and row["quality"] == budget.quality(after), "branch accounting")
    if job["trial"] == -1:
        run.require(row["status"] == "ok" and budget.event_science(dict(metrics=row["metrics"])) ==
                    budget.event_science(dict(metrics=e["metrics"])) and row["delta"] == e["delta"], "original action control")
    return after


def branch_worker(env, before, target, job, reg, parent_pid):
    die_with_parent(parent_pid)
    out = branch_folder(reg, target, job)
    try:
        run.require(not out.exists() and q.state_fingerprint(q._plain(env.get_state())) == target["fingerprint"], "fork root")
        event = target["event"]
        raw = q._plain(env.step_experimental_pp(job["action"], reg["config"]["pp_seconds"],
                                               "annealed", event["temperature"], event["uniform"]))
        after, metrics = raw["observation"], raw["metrics"]
        incomplete = pp_incomplete(before, after, metrics) or bool(raw["truncated"])
        row = dict(binding=reg["binding"], job=job, root_id=target["id"], status="censored" if incomplete else "ok",
            before=q.state_fingerprint(before), after=q.state_fingerprint(after), metrics=metrics,
            delta=q.encode_state_delta(before, after), feasible=after["feasible"], conflicts=after["num_of_colliding_pairs"],
            generated=after["low_level"]["generated"]-before["low_level"]["generated"],
            truncated=bool(raw["truncated"]), quality=budget.quality(after), no_ttf=True, files={})
        check_branch(before, row, target, job)
        if after["feasible"]:
            run.once(out / "witness.json", canonical_state(after))
            row["files"]["witness.json"] = run.sha256_file(out / "witness.json")
        run.once(out / "result.json", run.sealed(row))
    except Exception as error:
        run.once(out / "error.json", run.sealed(dict(binding=reg["binding"], job=job, root_id=target["id"],
                                                    status="error", error=repr(error))))
        raise


def read_branch(reg, target, job):
    row = run.result_read(branch_folder(reg, target, job), reg)
    run.require(row["job"] == job and row["root_id"] == target["id"], "saved branch identity")
    return row


def run_forks(env, state, target, reg, jobs, resume):
    run.require(sys.platform == "linux" and len(list(Path("/proc/self/task").iterdir())) == 1,
                "single-threaded Linux fork required")
    out = ROOT / reg["config"]["output"]
    pending = []
    for job in jobs:
        path = branch_folder(reg, target, job)
        if (path / "result.json").exists():
            run.require(resume and read_branch(reg, target, job)["status"] == "ok", "existing result requires verified resume")
        else:
            run.require(not path.exists(), "partial/error branch requires inspection")
            pending.append(job)
    active = []
    ctx = multiprocessing.get_context("fork")
    try:
        while pending or active:
            while pending and len(active) < reg["config"]["workers"] and not (out / "STOP_AFTER_JOB").exists():
                job = pending.pop(0)
                process = ctx.Process(target=branch_worker, args=(env, state, target, job, reg, os.getpid()))
                process.start()
                active.append((process, time.monotonic(), job))
            for item in list(active):
                process, began, job = item
                if process.is_alive() and time.monotonic()-began <= reg["config"]["branch_fuse_seconds"]:
                    continue
                if process.is_alive():
                    process.terminate()
                    process.join()
                    run.once(branch_folder(reg, target, job) / "timeout.json", run.sealed(dict(
                        binding=reg["binding"], job=job, status="censored", stop="external_timeout")))
                    raise ValueError("branch timeout is unknown; inspect before resuming")
                process.join()
                run.require(process.exitcode == 0 and read_branch(reg, target, job)["status"] == "ok", "branch failed or censored")
                active.remove(item)
                with (out / "progress.jsonl").open("a", encoding="utf8") as f:
                    f.write(json.dumps(dict(root=target["id"], job=job["job_id"], status="ok"))+"\n")
            run.write_json(out / "progress.json", dict(root=target["id"], pending=len(pending), active=len(active)))
            if pending and not active and (out / "STOP_AFTER_JOB").exists(): return False
            time.sleep(.1)
    finally:
        for process, _, _ in active:
            if process.is_alive(): process.terminate()
            process.join()
    run.require(q.state_fingerprint(q._plain(env.get_state())) == target["fingerprint"], "fork changed parent")
    return True


def group_worker(job):
    die_with_parent(job["parent_pid"])
    reg, source_job = job["registration"], job["source_job"]
    out = ROOT / reg["config"]["output"]
    print("REPLAY", source_job["record"]["trial"], flush=True)
    _, env, state, context, h, selector, engine, actor, plan = budget.prefix_replay(source_job)
    targets = {t["decision"]: t for t in job["targets"]}
    for e in run.trace_read(budget.folder(source_job)):
        run.require(h.decision == e["decision"] and e["before"] == q.state_fingerprint(state), "extension prefix")
        if h.decision in targets:
            if (out / "STOP_AFTER_JOB").exists(): return dict(job_id=job["job_id"], status="paused")
            target = targets[h.decision]
            before = q.state_fingerprint(state)
            event = run.selection(env, state, context, h, engine, selector, budget.runtime.feature_plan(plan), actor)
            run.require(all(event[k] == e[k] for k in event) and q.state_fingerprint(q._plain(env.get_state())) == before,
                        "full candidate/features/probabilities replay")
            run.require(before == target["fingerprint"] and budget.support.history_signature(h) == target["history"], "target fingerprint/history")
            validate_final(state)
            run.once(out / "roots" / (target["id"]+".json"), canonical_state(state))
            run.once(out / "roots" / (target["id"]+"-receipt.json"), dict(fingerprint=before,
                history=budget.support.history_signature(h), decision=h.decision,
                generated=state["low_level"]["generated"]-context["initial_nodes"],
                event_sha256=run.json_fingerprint(budget.event_science(e))))
            jobs = branch_jobs(reg, target)
            print("CONTROL", target["id"], h.decision, flush=True)
            if not run_forks(env, state, target, reg, jobs[:1], job["resume"]):
                return dict(job_id=job["job_id"], status="paused")
            check_branch(state, read_branch(reg, target, jobs[0]), target, jobs[0])
            print("CANDIDATES", target["id"], len(jobs)-1, flush=True)
            if not run_forks(env, state, target, reg, jobs[1:], job["resume"]):
                return dict(job_id=job["job_id"], status="paused")
            print("ROOT_DONE", target["id"], flush=True)
        if h.decision == max(targets): break
        actual, after = budget.support.execute_step(source_job["source_job"], q, env, state, context,
            h, selector, engine, actor, plan, plan["proposal"]["pp_safety_seconds"])
        run.require(budget.event_science(actual) == budget.event_science(e), "continuation science changed")
        h.observe(state, actual, after)
        state = after
    run.require(h.decision == max(targets), "missing final target")
    return dict(job_id=job["job_id"], status="ok")


def collect(resume=False):
    from experiments.repair_collection import _run_jobs
    reg, out = verify()
    run.require(os.name != "nt", "use frozen Linux native")
    run.require(resume or not (out / "run_status.json").exists(), "explicit resume required")
    by_source = {j["job_id"]: j for j in budget.jobs_for(reg["source"])}
    groups = {}
    for t in reg["targets"]: groups.setdefault(t["source_job_id"], []).append(t)
    with budget.recovery.strict_lock(out, reg["binding"], "terminal-candidates"):
        if resume: (out / "STOP_AFTER_JOB").unlink(missing_ok=True)
        run.write_json(out / "run_status.json", dict(status="running", binding=reg["binding"]))
        try:
            for key, targets in groups.items():
                if (out / "STOP_AFTER_JOB").exists():
                    run.write_json(out / "run_status.json", dict(status="paused", binding=reg["binding"]))
                    return dict(status="paused")
                job = dict(job_id=key, registration=reg, source_job=by_source[key], targets=targets,
                           parent_pid=os.getpid(), resume=resume)
                attempt = len(list((out / "group-attempts").glob("*.json")))
                result = _run_jobs(group_worker, [job], 1, phase="terminal-candidate-group",
                    output_root=out / "group-progress" / str(attempt), run_fingerprint=reg["binding"],
                    timeout_seconds=reg["config"]["group_fuse_seconds"],
                    failure_result=lambda j, status, error: dict(job_id=j["job_id"], status=status, error=error))
                run.once(out / "group-attempts" / f"{attempt:04d}.json", run.sealed(dict(binding=reg["binding"], results=result)))
                if len(result) == 1 and result[0]["status"] == "paused":
                    run.write_json(out / "run_status.json", dict(status="paused", binding=reg["binding"]))
                    return dict(status="paused")
                run.require(len(result) == 1 and result[0]["status"] == "ok", "group error; inspect retained attempt")
            files = {}
            for t in reg["targets"]:
                for job in branch_jobs(reg, t):
                    run.require(read_branch(reg, t, job)["status"] == "ok", "complete uncensored branches")
                    p = branch_folder(reg, t, job) / "result.json"
                    files[p.relative_to(out).as_posix()] = run.sha256_file(p)
                for suffix in (".json", "-receipt.json"):
                    p = out / "roots" / (t["id"]+suffix)
                    files[p.relative_to(out).as_posix()] = run.sha256_file(p)
            identity.verify_files(reg["inputs"])
            run.once(out / "collection.complete.json", run.sealed(dict(binding=reg["binding"], files=files)))
            run.write_json(out / "run_status.json", dict(status="collected", binding=reg["binding"]))
        except BaseException:
            run.write_json(out / "run_status.json", dict(status="needs_inspection", binding=reg["binding"]))
            raise
    return dict(status="collected", counterfactuals=228, controls=3)


def summarize_target(target, rows):
    candidates = []
    for c in target["event"]["pool"]:
        values = sorted((r for r in rows if r["job"]["candidate_id"] == c["candidate_id"] and r["job"]["trial"] >= 0),
                        key=lambda r: r["job"]["trial"])
        run.require([r["job"]["trial"] for r in values] == [0, 1, 2, 3] and all(r["status"] == "ok" for r in values), "four complete trials")
        candidates.append(dict(candidate_id=c["candidate_id"], agents=c["agents"],
            selected=c["candidate_id"] == target["event"]["selected_id"],
            both_endpoints=set(target["edge"]) <= set(c["agents"]),
            feasible=sum(r["feasible"] for r in values), conflicts=[r["conflicts"] for r in values],
            generated=[r["generated"] for r in values]))
    chosen = next(c for c in candidates if c["selected"])
    best = max(c["feasible"] for c in candidates)
    return dict(id=target["id"], edge=target["edge"], trial=target["trial"], decision=target["decision"],
        candidates=candidates, selected_feasible=chosen["feasible"], best_feasible=best,
        total_feasible=sum(c["feasible"] for c in candidates),
        interpretation="no_observed_terminal_option" if best == 0 else
            "selected_option_also_order_sensitive" if chosen["feasible"] else "unselected_terminal_witness")


def analyze():
    reg, out = verify()
    complete = run.check_seal(run.read_json(out / "collection.complete.json"))
    run.require(complete["binding"] == reg["binding"], "completion binding")
    identity.verify_files(complete["files"], out)
    summaries, expected_files = [], set()
    with budget.recovery.strict_lock(out, reg["binding"], "terminal-candidates-audit"):
        for t in reg["targets"]:
            before = run.read_json(out / "roots" / (t["id"]+".json"))
            run.require(before["num_of_colliding_pairs"] == 1 and len(before["agents"]) == 500 and
                        q.state_fingerprint(before) == t["fingerprint"], "root state")
            receipt = run.read_json(out / "roots" / (t["id"]+"-receipt.json"))
            run.require(receipt["fingerprint"] == t["fingerprint"] and receipt["decision"] == t["decision"] and
                        receipt["history"] == t["history"] and
                        receipt["generated"] >= 25000000 and receipt["event_sha256"] ==
                        run.json_fingerprint(budget.event_science(t["event"])), "inherited receipt")
            rows = []
            for job in branch_jobs(reg, t):
                row = read_branch(reg, t, job)
                run.require(row["status"] == "ok", "censored is unknown")
                after = check_branch(before, row, t, job)
                if row["feasible"]:
                    run.require(set(row["files"]) == {"witness.json"} and q.state_fingerprint(
                        run.read_json(branch_folder(reg, t, job) / "witness.json")) == q.state_fingerprint(after), "full witness")
                else: run.require(not row["files"], "unexpected witness")
                expected_files.add((branch_folder(reg, t, job) / "result.json").relative_to(out).as_posix())
                rows.append(row)
            for suffix in (".json", "-receipt.json"): expected_files.add("roots/"+t["id"]+suffix)
            summaries.append(summarize_target(t, rows))
        run.require(set(complete["files"]) == expected_files, "complete exact manifest")
        successes = sum(r["total_feasible"] for r in summaries)
        report = dict(binding=reg["binding"], roots=summaries, counterfactuals=228, controls=3,
            feasible_counterfactuals=successes, no_ttf=True, no_training=True, automatic_promotion=False,
            collection_sha256=run.sha256_file(out / "collection.complete.json"),
            decision="conditional_terminal_witness_requires_new_contract" if successes else "no_sampled_terminal_witness_no_ranker_training")
        run.once(out / "report.json", run.sealed(json.loads(json.dumps(report))))
        run.write_json(out / "run_status.json", dict(status="complete", report_sha256=run.sha256_file(out / "report.json")))
    return {k: report[k] for k in ("counterfactuals", "controls", "feasible_counterfactuals", "decision")}


def main(actions=None, description=None):
    actions = actions or dict(prepare=prepare, collect=collect, analyze=analyze, verify=verify, dry_run=dry_run)
    parser = argparse.ArgumentParser(description=description or __doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "dry-run", "collect", "analyze", "stop"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare": result = actions['prepare']()
    elif args.phase == "collect": result = actions['collect'](args.resume)
    elif args.phase == "analyze": result = actions['analyze']()
    else:
        reg, out = actions['verify']()
        if args.phase == "stop":
            run.write_json(out / "STOP_AFTER_JOB", dict(requested=True))
            result = dict(safe_stop_after_current_jobs=True)
        else: result = actions['dry_run'](reg)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
