"""Bounded resource recovery and first on-policy update, with immutable sources."""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
from experiments._common import contained_file
from experiments.sa_paired_completion import require

CONFIG = "configs/sa_onpolicy_resource_recovery.json"
CODE = (CONFIG, "scripts/recover_sa_onpolicy.py", "tests/evaluation/test_sa_onpolicy_recovery.py",
        "docs/SA_ONPOLICY_RESOURCE_RECOVERY_PROTOCOL_ZH.md")
PHASE = "train-0"


@contextmanager
def strict_lock(out, binding, phase, control_root=None):
    # Fail closed on existing locks. Never use Windows os.kill(pid, 0).
    control = control_root if control_root is not None else ROOT / "build/.repair_collection_control"
    owner = dict(run_id=uuid.uuid4().hex, pid=os.getpid(), host=socket.gethostname()+"|strict-recovery",
                 binding=binding, phase=phase, output_root=str(out))
    held = []
    try:
        for path in (control / "active.lock", out / ".collection.lock"):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf8") as f:
                held.append(path)
                json.dump(owner, f)
                f.flush()
                os.fsync(f.fileno())
        yield
    finally:
        for path in reversed(held):
            if path.exists() and run.read_json(path).get("run_id") == owner["run_id"]:
                path.unlink()


def runtime_plan(source, cfg, registration):
    require((cfg["episode_safety_seconds"], cfg["process_fuse_seconds"], cfg["workers"],
             cfg["maximum_updates_this_stage"], cfg["change_normalizer"], cfg["formal_ttf"], cfg["automatic_promotion"])
            == (900., 960., 20, 1, False, False, False), "recovery scope changed")
    require((source["proposal"]["max_decisions"], source["proposal"]["node_budget"],
             source["proposal"]["pp_safety_seconds"]) == (256, 25000000, 20.), "scientific work contract")
    # Preserve scientific/policy binding. A separate receipt identifies resources.
    return dict(source, config=dict(source["config"], output=cfg["output"]),
                proposal=dict(source["proposal"], episode_safety_seconds=cfg["episode_safety_seconds"],
                              process_fuse_seconds=cfg["process_fuse_seconds"]), resource_registration=registration)


def prepare():
    source, src = run.verify()
    cfg = run.read_json(ROOT / CONFIG)
    require(src == ROOT / cfg["source_output"], "source output")
    for name, key in (("plan.json", "source_plan_sha256"), ("train-0.partial-audit.json", "source_audit_sha256"),
                      ("models/actor-0.json", "source_actor_sha256")):
        require(run.sha256_file(src/name) == cfg[key], "source changed: " + name)
    require(not (src/"models/actor-1.json").exists(), "original policy already updated")
    runtime_plan(source, cfg, "pending")
    out = ROOT / cfg["output"]
    require(not out.exists(), "existing recovery; use verify/resume")
    audited = run.check_seal(run.read_json(src/"train-0.partial-audit.json"))
    require(audited["binding"] == source["binding"] and len(audited["results"]) == 80, "partial audit scope")
    old_hashes = {x["job_id"]: x["result_sha256"] for x in audited["results"] if x["status"] == "ok"}
    initial = {}
    for j in run.jobs_for(source, "qualify"):
        r = run.result_read(run.folder_for(src, "qualify", j), source)
        initial[r["pair_id"]] = r["initial_fingerprint"]
    entries = {}
    for j in run.jobs_for(source, PHASE, 0):
        old = run.folder_for(src, PHASE, j)
        entry = dict(job=j, expected_initial=initial[j["pair_id"]])
        if (old/"result.json").exists():
            row = run.result_read(old, source)
            require(old_hashes[j["job_id"]] == run.sha256_file(old/"result.json"), "old audit changed")
            require(row["initial_fingerprint"] == entry["expected_initial"], "original initial identity")
            entry.update(original_folder=old.relative_to(ROOT).as_posix(), original_sha256=old_hashes[j["job_id"]])
            if row["status"] == "ok":
                entry.update(origin="reused", folder=entry["original_folder"])
            else:
                require(j["job_id"] == cfg["censored_job_id"] and row["status"] == "censored", "unregistered censor")
                entry.update(origin="retry", folder=(out/PHASE/j["job_id"]).relative_to(ROOT).as_posix())
        else:
            require(not old.exists(), "unexpected partial source")
            entry.update(origin="new", folder=(out/PHASE/j["job_id"]).relative_to(ROOT).as_posix())
        entries[j["job_id"]] = entry
    require(Counter(e["origin"] for e in entries.values()) == {"reused": 79, "retry": 1, "new": 16}, "source partition")
    copies = ("models/actor-0.json", "models/receipt-0.json")
    inputs = {n: run.sha256_file(ROOT/n) for n in CODE}
    for name in ("plan.json", "train-0.partial-audit.json", "parity-0.json", "micro.json", *copies):
        inputs[(src/name).relative_to(ROOT).as_posix()] = run.sha256_file(src/name)
    registration = dict(schema=cfg["schema"], config=cfg, scientific_binding=source["binding"], inputs=inputs,
                        source_policy_sha256=run.validate_bundle(run.actor_load(src, source, 0)), entries=entries,
                        source_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        no_ttf=True, no_automatic_retries=True)
    registration["binding"] = run.json_fingerprint(registration)
    with strict_lock(out, registration["binding"], "prepare"):
        for name in copies:
            (out/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src/name, out/name)
            require(run.sha256_file(out/name) == run.sha256_file(src/name), "copy changed actor")
        run.once(out/"registration.json", run.sealed(registration))
    return dict(registered=True, binding=registration["binding"], reused=79, scheduled=17, maximum_updates=1)


def verify():
    source, src = run.verify()
    cfg = run.read_json(ROOT/CONFIG)
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/"registration.json"))
    body = {k: v for k, v in reg.items() if k not in {"binding", "integrity"}}
    require(reg["binding"] == run.json_fingerprint(body) and reg["config"] == cfg, "recovery registration")
    require(source["binding"] == reg["scientific_binding"] and src == ROOT/cfg["source_output"], "source contract")
    require(not (src/"models/actor-1.json").exists(), "original branch has another update")
    for name, digest in reg["inputs"].items():
        require(run.sha256_file(contained_file(ROOT, name, field="recovery input")) == digest, "changed recovery input: "+name)
    p = runtime_plan(source, cfg, reg["binding"])
    require(run.validate_bundle(run.actor_load(out, p, 0)) == reg["source_policy_sha256"], "changed actor0")
    return reg, p, out


def event_projection(event, after=True):
    keys = ("decision", "policy_sha256", "before", "pool", "proposal_order", "candidate_ids", "anchor_id",
            "features", "probabilities", "selected_id", "selection_draw", "behavior_log_probability", "action", "temperature", "uniform")
    return {k: event[k] for k in ((*keys, "delta") if after else keys)}


def compare_prefix(old_events, new_events):
    old_events, new_events = iter(old_events), iter(new_events)
    complete, censored = 0, 0
    for old in old_events:
        require(censored == 0, "event after old truncation")
        new = next(new_events, None)
        require(new is not None, "short recovery prefix")
        valid = old["metrics"]["acceptance_evaluated"] and old["metrics"]["pp_failure_reason"] != "time_limit"
        require(event_projection(old, valid) == event_projection(new, valid), "recovery prefix mismatch")
        if valid:
            complete += 1
        else:
            require(old["metrics"]["pp_rolled_back"] and new["metrics"]["acceptance_evaluated"], "incomplete recovery repair")
            censored += 1
    require(censored == 1, "expected exactly one old interrupted repair")
    return dict(exact_completed_steps=complete, exact_interrupted_pre_action=True)


def result_entry(reg, p, entry):
    folder = contained_file(ROOT, entry["folder"]+"/result.json", field="episode reference").parent
    row = run.result_read(folder, p)
    require(row["job_id"] == entry["job"]["job_id"] and row["initial_fingerprint"] == entry["expected_initial"], "episode identity")
    require(row["policy_sha256"] == reg["source_policy_sha256"] and row["status"] == "ok", "unknown/stale episode")
    if entry["origin"] == "reused":
        require(run.sha256_file(folder/"result.json") == entry["original_sha256"], "reused result changed")
    else:
        receipt = run.check_seal(run.read_json(folder/"resource_receipt.json"))
        require(receipt["recovery_binding"] == reg["binding"] and
                receipt["result_sha256"] == run.sha256_file(folder/"result.json"), "resource provenance")
    return folder, row


def recovery_worker(job):
    summary = run.episode_worker(job)
    folder = run.folder_for(ROOT/job["plan"]["config"]["output"], PHASE, job)
    run.once(folder/"resource_receipt.json", run.sealed(dict(recovery_binding=job["plan"]["resource_registration"],
         scientific_binding=job["plan"]["binding"], result_sha256=run.sha256_file(folder/"result.json"))))
    return summary


def collect(resume=False):
    from experiments.repair_collection import _run_jobs
    reg, p, out = verify()
    require(os.name != "nt", "collection uses frozen WSL native")
    with strict_lock(out, reg["binding"], "collect"):
        require(resume or not (out/PHASE).exists(), "use explicit resume")
        if resume:
            (out/"STOP_AFTER_BATCH").unlink(missing_ok=True)
        if (out/"STOP_AFTER_BATCH").exists():
            return dict(status="paused", no_update=True)
        pending = []
        for entry in reg["entries"].values():
            folder = ROOT/entry["folder"]
            if (folder/"result.json").exists():
                result_entry(reg, p, entry)
            else:
                jid = entry["job"]["job_id"]
                require(entry["origin"] != "reused" and not folder.exists() and not (out/"failures"/(jid+".json")).exists(), "partial/error: no automatic retry")
                pending.append(dict(entry["job"], plan=p, parent_pid=os.getpid(), expected_initial=entry["expected_initial"]))
        require(len(pending) <= reg["config"]["expected_new"], "unexpected job expansion")
        run.write_json(out/"run_status.json", dict(status="collecting", pending=len(pending)))
        def progress(row):
            with (out/"progress.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps(row)+"\n")
            print(row, flush=True)
        def failure(job, status, error):
            row = dict(status="censored" if status == "timeout" else "error", job_id=job["job_id"], error=error)
            run.once(out/"failures"/(job["job_id"]+".json"), run.sealed(row))
            return row
        rows = _run_jobs(recovery_worker, pending, reg["config"]["workers"], phase="resource-recovery", output_root=out/"progress",
            run_fingerprint=reg["binding"], timeout_seconds=p["proposal"]["process_fuse_seconds"],
            on_result=progress, failure_result=failure, stop_on_failure=True) if pending else []
        if len(rows) != len(pending) or any(x["status"] != "ok" for x in rows):
            run.write_json(out/"run_status.json", dict(status="needs_inspection", no_update=True))
            return dict(status="needs_inspection", no_update=True)
        records = {}
        for jid, entry in reg["entries"].items():
            folder, _ = result_entry(reg, p, entry)
            records[jid] = run.sha256_file(folder/"result.json")
        run.once(out/"collection.complete.json", run.sealed(dict(binding=reg["binding"], records=records)))
        run.write_json(out/"run_status.json", dict(status="collected", episodes=len(records)))
    return dict(collected=len(records), reused=79, new=17, no_update=True)


def audit_worker(job):
    row = run.audit_worker(job)
    if job["origin"] == "retry":
        folder = run.folder_for(ROOT/job["plan"]["config"]["output"], PHASE, job)
        old = ROOT/job["original_folder"]
        run.result_read(old, job["plan"])
        require(run.sha256_file(old/"result.json") == job["original_sha256"], "censored source changed")
        row["prefix"] = compare_prefix(run.trace_read(old), run.trace_read(folder))
    return row


def audit():
    from experiments.repair_collection import _run_jobs
    reg, p, out = verify()
    complete = run.check_seal(run.read_json(out/"collection.complete.json"))
    require(complete["binding"] == reg["binding"] and set(complete["records"]) == set(reg["entries"]), "collection coverage")
    with strict_lock(out, reg["binding"], "audit"):
        jobs, rows = [], []
        for jid, entry in reg["entries"].items():
            folder, _ = result_entry(reg, p, entry)
            digest = run.sha256_file(folder/"result.json")
            require(digest == complete["records"][jid], "collection changed")
            if entry["origin"] == "reused":
                rows.append(dict(status="ok", job_id=jid, result_sha256=digest, reused_source_audit=True))
            else:
                jobs.append(dict(entry["job"], plan=p, origin=entry["origin"], original_folder=entry.get("original_folder"),
                                 original_sha256=entry.get("original_sha256")))
        fresh = _run_jobs(audit_worker, jobs, reg["config"]["workers"], phase="recovery-audit", timeout_seconds=960, stop_on_failure=True)
        require(len(fresh) == len(jobs) and all(r["status"] == "ok" for r in fresh), "recovery audit failed")
        rows += fresh
        run.once(out/"batch.audit.json", run.sealed(dict(binding=reg["binding"], scientific_binding=p["binding"], results=rows)))
    return dict(audited=len(rows), reused_audits=79, new_audits=len(fresh), prefix=[r["prefix"] for r in fresh if "prefix" in r])


def audited_batch(reg, p, out):
    proof = run.check_seal(run.read_json(out/"batch.audit.json"))
    require(proof["binding"] == reg["binding"], "audit binding")
    require(len(proof["results"]) == 96 and all(r["status"] == "ok" for r in proof["results"]), "audit results")
    hashes = {r["job_id"]: r["result_sha256"] for r in proof["results"] if r["status"] == "ok"}
    require(set(hashes) == set(reg["entries"]) and len(hashes) == 96, "audit coverage")
    result = []
    for jid, entry in reg["entries"].items():
        folder, row = result_entry(reg, p, entry)
        require(run.sha256_file(folder/"result.json") == hashes[jid], "audited result changed")
        result.append((entry, folder, row))
    return result


def update():
    reg, p, out = verify()
    with strict_lock(out, reg["binding"], "update-0"):
        require(not (out/"update-0.json").exists() and not run.actor_file(out, 1).exists(), "one update only")
        rows, folders = [], {}
        for _, folder, row in audited_batch(reg, p, out):
            row["steps"] = [{k: e[k] for k in ("decision", "policy_sha256", "probabilities", "selected_id", "behavior_log_probability")}
                            for e in run.trace_read(folder)]
            rows.append(row)
            folders[row["episode_id"]] = folder
        coefficients = run.gradient_coefficients(rows, policy_sha256=reg["source_policy_sha256"],
            expected_groups={e["job"]["pair_id"]: e["job"]["case"]["map_id"] for e in reg["entries"].values()},
            replicas=4, max_decisions=256, node_budget=25000000)
        require(any(c["coefficient"] != 0 for c in coefficients), "no signal: do not fit")
        actor, diagnostic = run.update_once(run.actor_load(out, p, 0), coefficients,
            lambda key: run.trace_read(folders[key]), run.sha256_file(out/"batch.audit.json"))
        run.publish_actor(out, p, actor, dict(recovery_binding=reg["binding"], **diagnostic))
        result = dict(binding=reg["binding"], **diagnostic, episodes=len(rows),
                      nonzero_credit_episodes=sum(c["coefficient"] != 0 for c in coefficients), no_evaluation=True)
        run.once(out/"update-0.json", run.sealed(result))
    return result


def parity(torch_side=False):
    reg, p, out = verify()
    bundle = run.actor_load(out, p, 1)
    with strict_lock(out, reg["binding"], "parity-1"):
        if torch_side:
            from experiments.sa_onpolicy_actor import torch_actor, torch_distribution
            model, rows = torch_actor(bundle), []
            for _, folder, _ in audited_batch(reg, p, out):
                for e in run.trace_read(folder):
                    if e["decision"] not in (0, 32, 128, 255):
                        continue
                    ids = e["candidate_ids"]
                    scores = torch_distribution(model, bundle, ids, e["anchor_id"], e["features"]).probs.detach().numpy().tolist()
                    rows.append(dict(candidate_ids=ids, anchor_id=e["anchor_id"], features=e["features"], probabilities=dict(zip(ids, scores))))
            run.once(out/"torch-parity-1.json", run.sealed(dict(binding=reg["binding"], policy_sha256=run.validate_bundle(bundle), rows=rows)))
            return dict(fixtures=len(rows))
        run.native_runtime(p)
        ref = run.check_seal(run.read_json(out/"torch-parity-1.json"))
        actor = run.NumpyActor(bundle)
        require(ref["binding"] == reg["binding"] and ref["policy_sha256"] == actor.sha, "parity model")
        error = 0.
        for row in ref["rows"]:
            actual = actor.probabilities(row["candidate_ids"], row["anchor_id"], row["features"])
            error = max(error, max(abs(actual[k]-row["probabilities"][k]) for k in actual))
            require(error <= 1e-12, "portable probabilities differ")
            for i in range(16):
                draw = run.random.Random(i).random()
                require(run.select_with_draw(actual, draw) == run.select_with_draw(row["probabilities"], draw), "portable actions differ")
        receipt = dict(binding=reg["binding"], fixtures=len(ref["rows"]), max_error=error, policy_sha256=actor.sha)
        run.once(out/"parity-1.json", run.sealed(receipt))
    return receipt


def report():
    import math
    reg, p, out = verify()
    require(run.check_seal(run.read_json(out/"parity-1.json"))["binding"] == reg["binding"], "updated portable parity missing")
    actor = run.NumpyActor(run.actor_load(out, p, 1))
    totals, tvs, changed, max_change = Counter(), [], 0, 0.
    batch = audited_batch(reg, p, out)
    for _, folder, row in batch:
        totals[row["stop"]] += 1
        for e in run.trace_read(folder):
            actual = actor.probabilities(e["candidate_ids"], e["anchor_id"], e["features"])
            delta = [abs(actual[k]-e["probabilities"][k]) for k in actual]
            max_change = max(max_change, max(delta))
            tvs.append(.5*math.fsum(delta))
            changed += run.select_with_draw(actual, e["selection_draw"]) != e["selected_id"]
    result = dict(binding=reg["binding"], completed_batch=96, reused=79, new=17, stop_counts=dict(totals),
                  gradient_updates=1, decisions_replayed=len(tvs), same_draw_changed_actions=changed,
                  mean_total_variation=math.fsum(tvs)/len(tvs), max_probability_change=max_change,
                  decision="first_on_policy_update_verified_not_closed_loop_evidence", no_ttf=True,
                  no_heldout_evaluation=True, normalizer_unchanged=True, saturation_risk_remains=True)
    run.once(out/"report.json", run.sealed(result))
    run.write_json(out/"run_status.json", dict(status="first_update_verified", no_active_workers=True,
        next_stage="separate second on-policy batch; no additional fitting on this old batch"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "collect", "audit", "update", "parity", "report", "stop"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--torch", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare": result = prepare()
    elif args.phase == "verify": result = dict(binding=verify()[0]["binding"], verified=True)
    elif args.phase == "collect": result = collect(args.resume)
    elif args.phase == "audit": result = audit()
    elif args.phase == "update": result = update()
    elif args.phase == "parity": result = parity(args.torch)
    elif args.phase == "report": result = report()
    else:
        cfg = run.read_json(ROOT/CONFIG)
        run.write_json(ROOT/cfg["output"]/"STOP_AFTER_BATCH", dict(requested=True))
        result = dict(stop_after_current_batch=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if not isinstance(error, SystemExit):
            output = ROOT/run.read_json(ROOT/CONFIG)["output"]
            if (output/"registration.json").exists():
                run.write_json(output/"run_status.json", dict(status="error", error=repr(error), no_automatic_retry=True))
        raise
