"""Independent-reset, source/size-matched SA diagnostic; never trains a model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import json_fingerprint, read_json, sha256_file
from scripts.audit_sa_history_information import atomic, require
from scripts.collect_sa_history_candidate_bridge import branch, branch_labels
from scripts.run_sa_paired_closed_loop import once

CONFIG = ROOT / "configs/sa_source_matched_collection.json"
NATIVE = "build/linux/sa-wall-clock-v1/lns2_env.cpython-310-x86_64-linux-gnu.so"
NATIVE_SHA = "5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d"


def prepared(config):
    path = ROOT / config["preparation"]
    require(sha256_file(path) == config["preparation_sha256"], "preparation bytes changed")
    plan = read_json(path)
    require(plan["binding"] == json_fingerprint({k: v for k, v in plan.items() if k != "binding"}), "preparation binding")
    frozen = plan["config"]
    for key, other in (("seed", "trial_seed"), ("trials", "planned_trials"),
                       ("horizon", "planned_horizon"), ("workers", "planned_workers"),
                       ("pp_seconds", "planned_pp_safety_seconds"),
                       ("trial_seconds", "planned_trial_timeout_seconds")):
        require(config[key] == frozen[other], "prepared budget changed: " + key)
    require(config["no_ttf"] and not config["training_allowed"], "diagnostic boundary changed")
    return plan


def schedule(plan):
    # Interleave the pair and anchor within each trial, rather than group by action.
    return [dict(job_id=f"{r['state_id']}-{c['candidate_id']}-t{t}", root=r,
                 candidate_id=c["candidate_id"], trial=t)
            for r in plan["roots"] for t in range(plan["config"]["trials"])
            for c in r["candidates"]]


def merge_pinned(inputs, pins):
    for rel, digest in pins.items():
        require(rel not in inputs or inputs[rel] == digest, "conflicting historical hash: " + rel)
        require(sha256_file(ROOT / rel) == digest, "historical runtime/input changed: " + rel)
        inputs[rel] = digest


def build_plan():
    cfg = read_json(CONFIG)
    prep = prepared(cfg)
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()), "output outside build")
    inputs = dict(prep["inputs"])
    registration = read_json(ROOT / cfg["source"] / "registration.json")
    # Keep model bytes and actual Python runtime from the historical lane;
    # current C++ sources are not the identity of the already-frozen binary.
    historical = {p: h for p, h in registration["inputs"].items()
                  if p.startswith(("artifacts/", "experiments/", "lns2_selector/")) or p == NATIVE}
    cases = {c["task_id"]: c for c in registration["cases"]}
    for root in prep["roots"]:
        saved = read_json(ROOT / root["source_root"])
        for path in cases[saved["source"]["item"]["task_id"]]["files"].values():
            historical[path] = registration["inputs"][path]
    for name in ("scripts/run_sa_wall_clock.py", "scripts/diagnose_nonmonotonic_repair.py",
                 "scripts/run_feedback_exploration_diagnostics.py"):
        historical[name] = registration["inputs"][name]
    merge_pinned(inputs, historical)
    paths = [CONFIG, Path(__file__), ROOT / cfg["preparation"], ROOT / NATIVE,
             ROOT / "experiments/sa_source_matched_analysis.py",
             ROOT / "tests/evaluation/test_sa_source_matched_collection.py",
             ROOT / "tests/evaluation/test_sa_source_matched_analysis.py",
             ROOT / "docs/SA_SOURCE_MATCHED_COLLECTION_ZH.md",
             ROOT / "scripts/collect_sa_history_candidate_bridge.py",
             ROOT / "scripts/run_sa_paired_completion_pilot.py",
             ROOT / "scripts/run_sa_path_quality.py",
             ROOT / "scripts/train_sa_history_selector.py",
             ROOT / "experiments/sa_history_information.py",
             ROOT / "experiments/sa_history_selector.py",
             ROOT / "scripts/probe_sa_rejection_branches.py",
             ROOT / "scripts/run_feedback_exploration_diagnostics.py",
             ROOT / "experiments/repair_collection.py",
             ROOT / cfg["source"] / "registration.json"]
    for r in prep["roots"]:
        saved = read_json(ROOT / r["source_root"])
        require(saved["binding"] == r["source_binding"], "source root binding changed")
        folder = ROOT / cfg["source"] / "episodes" / saved["source"]["item"]["job_id"]
        paths.extend([folder / "initial.json", folder / "first_phase/trace.jsonl"])
    for path in paths:
        rel = path.relative_to(ROOT).as_posix()
        digest = sha256_file(path)
        require(rel not in inputs or inputs[rel] == digest, "prepared input changed: " + rel)
        inputs[rel] = digest
    for rel, digest in inputs.items():
        require(sha256_file(ROOT / rel) == digest, "registered input changed: " + rel)
    require(inputs[NATIVE] == NATIVE_SHA, "frozen native changed")
    result = dict(config=cfg, roots=prep["roots"], inputs=inputs, budget=prep["budget"],
                  preparation_binding=prep["binding"], native_sha256=NATIVE_SHA,
                  role="previously_viewed_development", collection_ready=True,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    result["binding"] = json_fingerprint(result)
    require(len(schedule(result)) == 376, "frozen job count changed")
    return result, out


def prepare():
    plan, out = build_plan()
    once(out / "plan.json", plan)
    return dict(binding=plan["binding"], budget=plan["budget"], collection_ready=True)


def verify():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg, "collection config changed")
    require(plan["binding"] == json_fingerprint({k: v for k, v in plan.items() if k != "binding"}), "collection binding")
    prep = prepared(cfg)
    require(plan["preparation_binding"] == prep["binding"] and plan["roots"] == prep["roots"], "blind root/pair selection changed")
    for rel, digest in plan["inputs"].items():
        require(sha256_file(ROOT / rel) == digest, "registered input changed: " + rel)
    return plan, out


def receipt(folder, job, plan):
    saved = read_json(folder / "receipt.json")
    require(saved["binding"] == plan["binding"] and saved["job_id"] == job["job_id"], "receipt identity")
    require(saved["result_sha256"] == sha256_file(folder / "result.json"), "trial result bytes changed")
    require(saved["started_sha256"] == sha256_file(folder / "started.json"), "trial start bytes changed")
    row = read_json(folder / "result.json")
    require(row["binding"] == plan["binding"], "result binding")
    require((row.get("root_id"), row.get("candidate_id"), row.get("trial")) ==
            (job["root"]["state_id"], job["candidate_id"], job["trial"]), "result identity")
    require({p.name for p in folder.iterdir()} == {"started.json", "result.json", "receipt.json"}, "incomplete/unexpected trial files")
    require(row["status"] in ("ok", "censored"), "failed trial requires review")
    return row


class BudgetExpired(Exception):
    pass


def remaining(deadline, clock=time.monotonic):
    seconds = deadline - clock()
    if seconds <= 0:
        raise BudgetExpired("trial budget exhausted")
    return seconds


def restore(job, plan, deadline):
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from scripts.run_sa_paired_completion_pilot import history_fingerprint
    from experiments.sa_history_selector import History
    cfg, entry = plan["config"], job["root"]
    require(q.native_identity()["sha256"] == plan["native_sha256"], "wrong native loaded")
    root = read_json(ROOT / entry["source_root"])
    target = root["source"]
    reg = read_json(ROOT / cfg["source"] / "registration.json")
    case = next(c for c in reg["cases"] if c["task_id"] == target["item"]["task_id"])
    worker = q.worker_job(case, target["item"], reg["template"], ROOT / cfg["output"] / "unused", plan["binding"])
    original = ROOT / cfg["source"] / "episodes" / target["item"]["job_id"]
    expected = read_json(original / "initial.json")["payload"]["observation"]
    remaining(deadline)
    env = q._make_environment(worker["dataset_root"], worker["row"], dict(worker["environment"], time_limit=100000.), "Adaptive")
    state = q._plain(env.reset(seed=worker["solver_seed"]))
    require(q.state_fingerprint(state) == q.state_fingerprint(expected), "initial fingerprint mismatch")
    history = History(state)
    event = None
    with (original / "first_phase/trace.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            event = json.loads(line)
            require(event["decision"] == history.decision, "prefix decision gap")
            if event["decision"] == target["decision"]:
                break
            seconds = min(cfg["pp_seconds"], remaining(deadline))
            raw = q._plain(env.step_experimental_pp(event["action"], seconds, "annealed", event["temperature"], event["uniform"]))
            if raw["metrics"]["pp_failure_reason"] == "time_limit":
                raise BudgetExpired("prefix PP safety limit; not a replay mismatch")
            after = raw["observation"]
            expected = q.apply_state_delta(expected, event["delta"])
            check_attempt(raw["metrics"], event["metrics"])
            require(q.state_fingerprint(after) == q.state_fingerprint(expected), "prefix fingerprint mismatch")
            history.observe(state, event, after)
            state = after
    require(event is not None and history.decision == target["decision"] == event["decision"], "missing root occurrence")
    require(q.state_fingerprint(state) == entry["root_fingerprint"] == root["state_fingerprint"], "root fingerprint mismatch")
    require(history_fingerprint(history) == target["history_fingerprint"], "history fingerprint mismatch")
    source_case = dict(case_id=worker["sa_case_id"], task_id=case["task_id"], solver_seed=worker["solver_seed"], proposal=worker["sa_proposal"])
    remaining(deadline)
    index, pool = q.SingleFullCheckPool(source_case).select(env, state, target["decision"])
    require((index, pool) == (event["selected_index"], event["pool"]) ==
            (root["control_event"]["selected_index"], root["control_event"]["pool"]), "candidate pool mismatch")
    require(q.state_fingerprint(env.get_state()) == entry["root_fingerprint"], "proposal changed state")
    lookup = {c["candidate_id"]: c for c in pool}
    for candidate in entry["candidates"]:
        actual = lookup[candidate["candidate_id"]]
        require(actual["agents"] == candidate["agents"], "candidate membership changed")
        require(sorted(actual["selection_families"]) == sorted(candidate["selection_families"]), "candidate source changed")
    remaining(deadline)
    return env, root, source_case, lookup[job["candidate_id"]]


def mark_receipt(folder, job, plan):
    once(folder / "receipt.json", dict(binding=plan["binding"], job_id=job["job_id"],
         result_sha256=sha256_file(folder / "result.json"), started_sha256=sha256_file(folder / "started.json")))


def archive_interrupted_writes(folder, job):
    """Only called after the scheduler has joined the timed-out child."""
    for name in ("started.json.tmp", "result.json.tmp", "receipt.json.tmp"):
        path = folder / name
        if not path.exists():
            continue
        require(not path.is_symlink(), "unexpected symlink in atomic output")
        digest = sha256_file(path)
        destination = folder.parent.parent / "interrupted_writes" / job["job_id"] / (name + "-" + digest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        require(not destination.exists(), "duplicate interrupted write requires review")
        path.rename(destination)


def trial_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job["parent_pid"])
    plan = job["plan"]
    folder = Path(job["folder"])
    require(not folder.exists(), "partial trial requires audit before retry")
    began = time.monotonic()
    deadline = began + plan["config"]["trial_seconds"]
    atomic(folder / "started.json", dict(binding=plan["binding"], job_id=job["job_id"], pid=os.getpid()))
    try:
        env, root, case, candidate = restore(job, plan, deadline)
        # The existing rollout is reused unchanged. Its budget is only the time
        # left after reset/replay, and the parent also enforces a whole-job fuse.
        cfg = dict(plan["config"], branch_seconds=remaining(deadline), node_budget=2**63-1)
        branch(env, root, case, dict(target=root["source"], previous_best=root["previous_best"]),
               candidate, job["trial"], dict(plan, config=cfg), folder / "result.json", job["parent_pid"])
        row = read_json(folder / "result.json")
        require(row["status"] == "ok", "rollout error: " + str(row.get("error")))
        label = validate_trial(row, job, plan)
        mark_receipt(folder, job, plan)
        return dict(status="ok", job_id=job["job_id"], stop=row["stop"], completion=label,
                    diagnostic_seconds=time.monotonic()-began)
    except BudgetExpired as exc:
        atomic(folder / "result.json", dict(status="censored", binding=plan["binding"], root_id=job["root"]["state_id"],
             candidate_id=job["candidate_id"], trial=job["trial"], stop="prefix_or_total_safety", reason=str(exc)))
        mark_receipt(folder, job, plan)
        return dict(status="ok", job_id=job["job_id"], stop="prefix_or_total_safety", completion=None)


def validate_trial(row, job, plan):
    require((row.get("root_id"), row.get("candidate_id"), row.get("trial")) ==
            (job["root"]["state_id"], job["candidate_id"], job["trial"]), "trial identity mismatch")
    if row["status"] == "censored":
        require(row["stop"] in ("hard_fuse", "prefix_or_total_safety"), "unknown censor reason")
        return None
    root = read_json(ROOT / job["root"]["source_root"])
    require(row["status"] == "ok", "trial failed")
    if row["events"]:
        require(row["events"][0]["pool"] == root["control_event"]["pool"], "initial pool identity")
    else:
        require(row["stop"] == "wall_safety", "empty uncensored branch")
    label = branch_labels(row, root, dict(target=root["source"], previous_best=root["previous_best"]), plan["config"])
    return None if label is None else bool(label["completion"])


def _failure(job, status, error):
    folder, plan = Path(job["folder"]), job["plan"]
    if status == "timeout":
        archive_interrupted_writes(folder, job)
        # An already committed result wins a race with the worker's exit.
        if (folder / "receipt.json").exists():
            row = receipt(folder, job, plan)
            return dict(status="ok", job_id=job["job_id"], stop=row["stop"], completion=validate_trial(row, job, plan))
        if (folder / "result.json").exists():
            row = read_json(folder / "result.json")
            label = validate_trial(row, job, plan)
            mark_receipt(folder, job, plan)
            return dict(status="ok", job_id=job["job_id"], stop=row["stop"], completion=label)
        if not (folder / "started.json").exists():
            atomic(folder / "started.json", dict(binding=plan["binding"], job_id=job["job_id"], startup_timeout=True))
        atomic(folder / "result.json", dict(status="censored", binding=plan["binding"], root_id=job["root"]["state_id"],
             candidate_id=job["candidate_id"], trial=job["trial"], stop="hard_fuse", reason=error))
        mark_receipt(folder, job, plan)
        return dict(status="ok", job_id=job["job_id"], stop="hard_fuse", completion=None)
    return dict(status="error", job_id=job["job_id"], error=error)


def failure(job, status, error):
    try:
        return _failure(job, status, error)
    except Exception as exc:
        # A corrupt timeout result must halt the NEXT batch, not kill unrelated
        # live children before they can seal their own independent outputs.
        return dict(status="error", job_id=job["job_id"], error=repr(exc), original_failure=error)


def existing_jobs(plan, out):
    jobs = schedule(plan)
    known = {j["job_id"] for j in jobs}
    base = out / "trials"
    if base.exists():
        require({p.name for p in base.iterdir()} <= known, "unregistered trial directory")
    pending, completed = [], []
    for job in jobs:
        folder = base / job["job_id"]
        if folder.exists():
            require((folder / "receipt.json").exists(), "interrupted trial requires review: " + job["job_id"])
            receipt(folder, job, plan)
            completed.append(job)
        else:
            pending.append(dict(job, folder=str(folder), plan=plan, parent_pid=os.getpid()))
    return pending, completed


def collect(resume=False, limit=None, smoke=False):
    from experiments.repair_collection import _CollectionRunLock, _run_jobs
    plan, out = verify()
    if smoke:
        out = out / "smoke"
        limit = 3 if limit is None else limit
    require(limit is None or limit > 0, "positive job limit required")
    with _CollectionRunLock(out, plan["binding"], "source-matched"):
        require(resume or not (out / "trials").exists(), "existing collection requires --resume")
        if resume and (out / "STOP_AFTER_BATCH").exists():
            (out / "STOP_AFTER_BATCH").unlink()
        pending, completed = existing_jobs(plan, out)
        pending = pending[:limit]
        atomic(out / "run_status.json", dict(status="running", binding=plan["binding"], complete=len(completed), total=len(schedule(plan))))
        try:
            for offset in range(0, len(pending), plan["config"]["workers"]):
                if (out / "STOP_AFTER_BATCH").exists():
                    break
                batch = pending[offset:offset+plan["config"]["workers"]]
                def progress(result):
                    with (out / "progress.jsonl").open("a", encoding="utf8") as stream:
                        stream.write(json.dumps(result, sort_keys=True) + "\n")
                    print(json.dumps(result, sort_keys=True), flush=True)
                rows = _run_jobs(trial_worker, batch, workers=plan["config"]["workers"],
                    phase="source-matched", output_root=out / "progress", run_fingerprint=plan["binding"],
                    timeout_seconds=plan["config"]["trial_seconds"], on_result=progress,
                    failure_result=failure, stop_on_failure=False)
                require(all(r["status"] == "ok" for r in rows), "trial error; batch drained, review before resume")
                _, completed = existing_jobs(plan, out)
                atomic(out / "run_status.json", dict(status="running", binding=plan["binding"], complete=len(completed), total=len(schedule(plan))))
                verify()
            verify()
            _, completed = existing_jobs(plan, out)
            status = "completed" if len(completed) == len(schedule(plan)) else "paused"
            atomic(out / "run_status.json", dict(status=status, binding=plan["binding"], complete=len(completed), total=len(schedule(plan))))
            return read_json(out / "run_status.json")
        except BaseException as exc:
            atomic(out / "run_status.json", dict(status="error", binding=plan["binding"], error=repr(exc)))
            raise


def analyze():
    from experiments.sa_source_matched_analysis import summarize
    plan, out = verify()
    pending, completed = existing_jobs(plan, out)
    require(not pending and len(completed) == 376, "collection incomplete")
    records, inputs, stops = [], {}, {}
    jobs = {j["job_id"]: j for j in completed}
    for r in plan["roots"]:
        values = {}
        for candidate in r["candidates"]:
            cid = candidate["candidate_id"]
            values[cid] = []
            for t in range(plan["config"]["trials"]):
                job = jobs[f"{r['state_id']}-{cid}-t{t}"]
                folder = out / "trials" / job["job_id"]
                row = receipt(folder, job, plan)
                values[cid].append(validate_trial(row, job, plan))
                stops[row["stop"]] = stops.get(row["stop"], 0) + 1
                for p in folder.iterdir():
                    inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
        records.append(dict(state_id=r["state_id"], map_id=r["map_id"], pair_ids=r["pair_ids"],
                            anchor_id=r["anchor_id"], values=values))
    archived = sorted((out / "interrupted_writes").glob("*/*"))
    inputs.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in archived})
    report = summarize(records, bootstrap_samples=plan["config"]["bootstrap"], seed=plan["config"]["bootstrap_seed"])
    report.update(binding=plan["binding"], no_ttf=True, training_allowed=False, trial_jobs=376,
                  role=plan["role"], stops=stops, input_hashes=inputs, interrupted_atomic_writes=len(archived))
    once(out / "analysis_records.json", records)
    once(out / "report.json", report)
    return {k: v for k, v in report.items() if k not in ("input_hashes", "rows")}


def request_stop(smoke=False):
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()), "stop output outside build")
    plan = read_json(out / "plan.json")
    require(plan["binding"] == json_fingerprint({k: v for k, v in plan.items() if k != "binding"}), "stop plan identity")
    require(plan["config"]["output"] == cfg["output"], "stop output changed")
    if smoke:
        out = out / "smoke"
    atomic(out / "STOP_AFTER_BATCH", dict(binding=plan["binding"]))
    return dict(stop_requested=True, drain_active_jobs=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "dry-run", "collect", "analyze", "stop"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit-jobs", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare()
    elif args.phase in ("verify", "dry-run"):
        plan, out = verify()
        pending, completed = existing_jobs(plan, out)
        result = dict(binding=plan["binding"], budget=plan["budget"], pending=len(pending), complete=len(completed))
    elif args.phase == "collect":
        result = collect(args.resume, args.limit_jobs, args.smoke)
    elif args.phase == "analyze":
        result = analyze()
    else:
        result = request_stop(args.smoke)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
