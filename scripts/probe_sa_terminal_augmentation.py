"""Fixed posthoc set augmentation, with paired PP seeds and reused controls."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_sa_terminal_candidates as source
from scripts import audit_sa_terminal_resources as resources

run, q, budget, identity = source.run, source.q, source.budget, source.identity
CONFIG = "configs/sa_terminal_augmentation.json"
CODE = (CONFIG, "scripts/probe_sa_terminal_augmentation.py",
        "tests/evaluation/test_sa_terminal_augmentation.py", "docs/SA_TERMINAL_AUGMENTATION_PROTOCOL_ZH.md")


def config():
    cfg = run.read_json(ROOT / CONFIG)
    fixed = dict(trials=4, workers=20, pp_seconds=20., branch_fuse_seconds=90.,
                 group_fuse_seconds=1860., max_decisions=None, training=False,
                 formal_ttf=False, automatic_expansion=False)
    run.require(all(cfg[k] == v for k, v in fixed.items()), "fixed augmentation scope")
    return cfg


def candidate_sets(target, state, cfg):
    known = {a["id"] for a in state["agents"]}
    original = set(target["event"]["action"]["agents"])
    resources_set = set(cfg["resource_agents"])
    run.require(len(known) == len(state["agents"]) and original <= known and resources_set <= known,
                "unique known agents")
    added = resources_set - original
    run.require(added, "targeted addition must change candidate")
    external = sorted(known - original, key=lambda aid: (
        run.json_fingerprint([cfg["random_namespace"], target["fingerprint"], aid]), aid))
    random_set = original | set(external[:len(added)])
    targeted = original | added
    run.require(len(random_set) == len(targeted) and random_set != targeted, "distinct equal-size control")
    return [dict(arm=arm, candidate_id="augmentation-"+run.json_fingerprint(sorted(agents))[:16],
                 agents=sorted(agents), additions=sorted(agents-original))
            for arm, agents in (("targeted", targeted), ("random", random_set))]


def baseline_jobs(old, target):
    jobs = [j for j in source.branch_jobs(old, target)
            if j["trial"] >= 0 and j["candidate_id"] == target["event"]["selected_id"]]
    run.require([j["trial"] for j in jobs] == [0, 1, 2, 3] and all(
        j["action"]["agents"] == target["event"]["action"]["agents"] for j in jobs), "original paired trials")
    return jobs


def branch_jobs(reg, target):
    old = reg["source"]
    jobs = [source.branch_jobs(old, target)[0]]
    for baseline in baseline_jobs(old, target):
        for c in reg["candidates"][target["id"]]:
            jobs.append(dict(job_id=f"{c['arm']}-t{baseline['trial']}", candidate_id=c["candidate_id"],
                trial=baseline["trial"], action=dict(mode="explicit_neighborhood", agents=c["agents"],
                    random_seed=baseline["action"]["random_seed"])))
    return jobs


def prepare():
    cfg = config()
    old, src = source.verify()
    resource_reg = resources.verify()
    run.require(run.sha256_file(src / "report.json") == cfg["source_report_sha256"] and
                run.sha256_file(resources.OUT / "report.json") == cfg["resource_report_sha256"], "prior evidence SHA")
    targets = [t for rid in cfg["target_ids"] for t in old["targets"] if t["id"] == rid]
    run.require(len(targets) == 2 and [t["decision"] for t in targets] == [241, 249] and
                len({t["source_job_id"] for t in targets}) == 1 and
                all(t["split"] == "train" for t in targets), "two fixed Train roots")
    inputs = dict(resource_reg["inputs"])
    candidates = {}
    for t in targets:
        state = run.read_json(src / "roots" / (t["id"]+".json"))
        run.require(q.state_fingerprint(state) == t["fingerprint"] and len(state["agents"]) == 500, "full root")
        candidates[t["id"]] = candidate_sets(t, state, cfg)
        for j in baseline_jobs(old, t):
            row = source.read_branch(old, t, j)
            run.require(row["status"] == "ok", "baseline trial complete")
            source.check_branch(state, row, t, j)
    commit = run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for path, content in identity.source_python(commit).items():
        run.require(identity.same_source((ROOT/path).read_bytes(), content), "uncommitted Python: "+path)
        inputs[path] = run.sha256_file(ROOT/path)
    for name in CODE:
        frozen = run.subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
        run.require(identity.same_source((ROOT/name).read_bytes(), frozen), "uncommitted protocol: "+name)
        inputs[name] = run.sha256_file(ROOT/name)
    for name in ("registration.json", "report.json", "resource_exchange_candidates.json", "independent_event_check.json"):
        path = resources.OUT / name
        inputs[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    body = dict(config=cfg, source=old, targets=targets, candidates=candidates, inputs=inputs,
        source_commit=commit, exploratory_posthoc=True, no_training=True, no_ttf=True,
        qualification="two correlated states of one task; not a success-rate experiment")
    body["binding"] = run.json_fingerprint(body)
    out = ROOT / cfg["output"]
    run.require(not out.exists(), "preserve prior output")
    with budget.recovery.strict_lock(out, body["binding"], "augmentation-prepare"):
        run.once(out / "registration.json", run.sealed(body))
    return dry_run(body)


def verify():
    cfg = config()
    out = ROOT / cfg["output"]
    reg = run.check_seal(run.read_json(out / "registration.json"))
    run.require(reg["config"] == cfg and reg["binding"] == run.json_fingerprint(
        {k: v for k, v in reg.items() if k not in {"binding", "integrity"}}), "registration identity")
    identity.verify_files(reg["inputs"])
    old, src = source.verify()
    run.require(reg["source"] == old and reg["targets"] == [
        t for rid in cfg["target_ids"] for t in old["targets"] if t["id"] == rid], "source roots changed")
    for t in reg["targets"]:
        state = run.read_json(src / "roots" / (t["id"]+".json"))
        run.require(reg["candidates"][t["id"]] == candidate_sets(t, state, cfg), "blind candidate plan changed")
    return reg, out


def dry_run(reg):
    return dict(new_branches=16, original_controls=2, reused_baseline_trials=8, replay_repairs=249,
        workers=reg["config"]["workers"], concurrent_branches_per_root=8, max_decisions=None,
        no_ttf=True, no_training=True, candidates=reg["candidates"])


def group_worker(job):
    source.die_with_parent(job["parent_pid"])
    reg, source_job = job["registration"], job["source_job"]
    out = ROOT / reg["config"]["output"]
    print("REPLAY", source_job["record"]["trial"], flush=True)
    _, env, state, context, h, selector, engine, actor, plan = budget.prefix_replay(source_job)
    targets = {t["decision"]: t for t in reg["targets"]}
    for e in run.trace_read(budget.folder(source_job)):
        run.require(h.decision == e["decision"] and e["before"] == q.state_fingerprint(state), "extension prefix")
        if h.decision in targets:
            if (out / "STOP_AFTER_JOB").exists(): return dict(job_id=job["job_id"], status="paused")
            t = targets[h.decision]
            event = run.selection(env, state, context, h, engine, selector, budget.runtime.feature_plan(plan), actor)
            run.require(all(event[k] == e[k] for k in event) and
                q.state_fingerprint(q._plain(env.get_state())) == t["fingerprint"] and
                budget.support.history_signature(h) == t["history"], "root/candidates/history replay")
            source.validate_final(state)
            run.once(out / "roots" / (t["id"]+".json"), source.canonical_state(state))
            run.once(out / "roots" / (t["id"]+"-receipt.json"), dict(fingerprint=t["fingerprint"],
                history=t["history"], decision=h.decision, event_sha256=run.json_fingerprint(budget.event_science(e))))
            jobs = branch_jobs(reg, t)
            print("CONTROL", t["id"], flush=True)
            if not source.run_forks(env, state, t, reg, jobs[:1], job["resume"]):
                return dict(job_id=job["job_id"], status="paused")
            source.check_branch(state, source.read_branch(reg, t, jobs[0]), t, jobs[0])
            print("AUGMENTATIONS", t["id"], len(jobs)-1, flush=True)
            if not source.run_forks(env, state, t, reg, jobs[1:], job["resume"]):
                return dict(job_id=job["job_id"], status="paused")
            print("ROOT_DONE", t["id"], flush=True)
        if h.decision == max(targets): break
        actual, after = budget.support.execute_step(source_job["source_job"], q, env, state, context,
            h, selector, engine, actor, plan, plan["proposal"]["pp_safety_seconds"])
        run.require(budget.event_science(actual) == budget.event_science(e), "continuation science changed")
        h.observe(state, actual, after)
        state = after
    run.require(h.decision == max(targets), "missing final target")
    return dict(job_id=job["job_id"], status="ok")


def collection_files(reg, out):
    files = {}
    for t in reg["targets"]:
        for j in branch_jobs(reg, t):
            row = source.read_branch(reg, t, j)
            run.require(row["status"] == "ok", "incomplete branch is unknown")
            path = source.branch_folder(reg, t, j) / "result.json"
            files[path.relative_to(out).as_posix()] = run.sha256_file(path)
        for suffix in (".json", "-receipt.json"):
            path = out / "roots" / (t["id"]+suffix)
            files[path.relative_to(out).as_posix()] = run.sha256_file(path)
    return files


def collect(resume=False):
    from experiments.repair_collection import _run_jobs
    reg, out = verify()
    run.require(sys.platform == "linux", "use frozen Linux native")
    run.require(resume or not (out / "run_status.json").exists(), "explicit resume required")
    with budget.recovery.strict_lock(out, reg["binding"], "terminal-augmentation"):
        if resume: (out / "STOP_AFTER_JOB").unlink(missing_ok=True)
        run.write_json(out / "run_status.json", dict(status="running", binding=reg["binding"]))
        try:
            by_source = {j["job_id"]: j for j in budget.jobs_for(reg["source"]["source"])}
            key = reg["targets"][0]["source_job_id"]
            job = dict(job_id=key, registration=reg, source_job=by_source[key], parent_pid=os.getpid(), resume=resume)
            attempt = len(list((out / "attempts").glob("*.json")))
            result = _run_jobs(group_worker, [job], 1, phase="augmentation-group",
                output_root=out / "group-progress" / str(attempt), run_fingerprint=reg["binding"],
                timeout_seconds=reg["config"]["group_fuse_seconds"],
                failure_result=lambda j, status, error: dict(job_id=j["job_id"], status=status, error=error))
            run.once(out / "attempts" / f"{attempt:04d}.json", run.sealed(dict(binding=reg["binding"], results=result)))
            if len(result) == 1 and result[0]["status"] == "paused":
                run.write_json(out / "run_status.json", dict(status="paused", binding=reg["binding"]))
                return dict(status="paused")
            run.require(len(result) == 1 and result[0]["status"] == "ok", "group failed; inspect, do not auto-retry")
            files = collection_files(reg, out)
            identity.verify_files(reg["inputs"])
            run.once(out / "collection.complete.json", run.sealed(dict(binding=reg["binding"], files=files)))
            run.write_json(out / "run_status.json", dict(status="collected", binding=reg["binding"]))
        except BaseException:
            run.write_json(out / "run_status.json", dict(status="needs_inspection", binding=reg["binding"]))
            raise
    return dict(status="collected", new_branches=16, controls=2)


def summarize(rows):
    rows = sorted(rows, key=lambda r: r["trial"])
    run.require([r["trial"] for r in rows] == [0, 1, 2, 3] and all(
        r["status"] == "ok" and bool(r["feasible"]) == (r["conflicts"] == 0) for r in rows), "four complete trials")
    return dict(feasible=sum(r["feasible"] for r in rows), conflicts=[r["conflicts"] for r in rows],
        generated=[r["generated"] for r in rows], old_pair_removed=sum(r["old_pair_removed"] for r in rows),
        distinct_orders=len({tuple(r["repair_order"]) for r in rows}), trials=rows)


def analyze():
    reg, out = verify()
    complete = run.check_seal(run.read_json(out / "collection.complete.json"))
    run.require(complete["binding"] == reg["binding"] and complete["files"] == collection_files(reg, out), "complete exact manifest")
    reports = []
    with budget.recovery.strict_lock(out, reg["binding"], "augmentation-audit"):
        for t in reg["targets"]:
            before = run.read_json(out / "roots" / (t["id"]+".json"))
            run.require(q.state_fingerprint(before) == t["fingerprint"] and len(before["agents"]) == 500, "root identity")
            receipt = run.read_json(out / "roots" / (t["id"]+"-receipt.json"))
            run.require(receipt == dict(fingerprint=t["fingerprint"], history=t["history"], decision=t["decision"],
                event_sha256=run.json_fingerprint(budget.event_science(t["event"]))), "prefix receipt")
            jobs = [(reg["source"], j, "original") for j in baseline_jobs(reg["source"], t)]
            by_candidate = {c["candidate_id"]: c["arm"] for c in reg["candidates"][t["id"]]}
            jobs += [(reg, j, "control" if j["trial"] == -1 else by_candidate[j["candidate_id"]]) for j in branch_jobs(reg, t)]
            arms = {name: [] for name in ("original", "targeted", "random")}
            for registration, job, arm in jobs:
                row = source.read_branch(registration, t, job)
                after = source.check_branch(before, row, t, job)
                if row["feasible"]:
                    witness = source.branch_folder(registration, t, job) / "witness.json"
                    run.require(set(row["files"]) == {"witness.json"} and
                        q.state_fingerprint(run.read_json(witness)) == q.state_fingerprint(after), "full witness")
                else: run.require(not row["files"], "unexpected witness")
                if arm == "control": continue
                events = resources.describe_events(before, after, row["metrics"]["repair_order"])
                edges = [list(pair) for pair in sorted({tuple(e["pair"]) for e in events})]
                arms[arm].append(dict(trial=job["trial"], status=row["status"], feasible=row["feasible"],
                    conflicts=row["conflicts"], generated=row["generated"], quality=row["quality"],
                    old_pair_removed=sorted(t["edge"]) not in edges, edges=edges, events=events,
                    repair_order=row["metrics"]["repair_order"], random_seed=job["action"]["random_seed"],
                    result_path=(source.branch_folder(registration,t,job)/"result.json").relative_to(ROOT).as_posix()))
            reports.append(dict(id=t["id"], decision=t["decision"], original_agents=t["event"]["action"]["agents"],
                candidates=reg["candidates"][t["id"]], arms={k: summarize(v) for k,v in arms.items()}))
        successes = {a: sum(r["arms"][a]["feasible"] for r in reports) for a in ("original", "targeted", "random")}
        decision = ("no_sampled_terminal_recovery" if not any(successes.values()) else
                    "targeted_witness_posthoc_not_online_policy" if successes["targeted"] > successes["random"] else
                    "no_targeted_increment_over_random")
        report = dict(binding=reg["binding"], roots=reports, successes=successes, decision=decision,
            new_branches=16, controls=2, reused_baselines=8, independent_tasks=1, no_ttf=True, no_training=True,
            automatic_promotion=False, exploratory_posthoc=True,
            collection_sha256=run.sha256_file(out/"collection.complete.json"))
        identity.verify_files(reg["inputs"])
        run.once(out / "report.json", run.sealed(report))
        run.write_json(out / "run_status.json", dict(status="complete", report_sha256=run.sha256_file(out/"report.json")))
    return dict(successes=successes, decision=decision, new_branches=16, independent_tasks=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "dry-run", "collect", "analyze", "stop"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare": result = prepare()
    elif args.phase == "collect": result = collect(args.resume)
    elif args.phase == "analyze": result = analyze()
    else:
        reg, out = verify()
        if args.phase == "stop":
            run.write_json(out / "STOP_AFTER_JOB", dict(requested=True))
            result = dict(safe_stop_after_current_jobs=True)
        else: result = dry_run(reg)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__": main()
