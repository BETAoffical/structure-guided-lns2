"""Sealed, lower-concurrency recovery of registered unknown SA trials only."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import json_fingerprint, read_json, sha256_file
from scripts import collect_sa_source_matched as old
from scripts.audit_sa_history_information import atomic, require
from scripts.collect_sa_history_candidate_bridge import randomization
from scripts.run_sa_paired_closed_loop import once

CONFIG = ROOT / "configs/sa_source_matched_recovery.json"
TIMING_FIELDS = frozenset((
    "diagnostic_seconds", "binding_residual_seconds", "binding_solver_call_seconds",
    "binding_state_snapshot_seconds", "binding_total_seconds", "episode_runtime_delta_seconds",
    "metrics_to_python_seconds", "native_neighborhood_generation_seconds",
    "native_repair_bookkeeping_seconds", "native_replan_seconds", "native_residual_seconds",
    "native_state_snapshot_seconds", "native_step_seconds", "pp_replan_seconds",
    "runtime_after", "runtime_before", "state_to_python_seconds", "step_runtime",
))


def scope(original, records, cfg):
    by_id = {r["state_id"]: r for r in records}
    jobs, controls = [], []
    for job in old.schedule(original):
        r = by_id[job["root"]["state_id"]]
        value = r["values"][job["candidate_id"]][job["trial"]]
        if value is None:
            require(r["state_id"] == cfg["root_id"], "unknown outside recovery scope")
            jobs.append(dict(job, role="recovery"))
        elif r["state_id"] == cfg["root_id"] and job["trial"] == cfg["control_trial"]:
            controls.append(dict(job, role="control"))
    require(len(jobs) == cfg["unknown_jobs"] and len(controls) == 3, "recovery scope changed")
    return controls + jobs


def build_plan():
    cfg = read_json(CONFIG)
    original, original_out = old.verify()
    require(original_out == ROOT / cfg["original"], "original output changed")
    for name, key in (("plan.json", "original_plan_sha256"), ("report.json", "original_report_sha256"),
                      ("analysis_records.json", "original_records_sha256")):
        require(sha256_file(original_out / name) == cfg[key], "original evidence changed: " + name)
    for key in ("trial_seconds", "pp_seconds", "horizon", "no_ttf", "training_allowed"):
        require(cfg[key] == original["config"][key], "scientific/resource boundary changed: " + key)
    require(cfg["workers"] == 4 and cfg["control_trial"] == 0 and cfg["unknown_jobs"] == 16,
            "registered recovery allocation changed")
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()) and out != original_out, "unsafe recovery output")
    inputs = dict(original["inputs"])
    old.merge_pinned(inputs, read_json(original_out / "report.json")["input_hashes"])
    paths = [CONFIG, Path(__file__), original_out / "plan.json", original_out / "report.json",
             original_out / "analysis_records.json", ROOT / "tests/evaluation/test_sa_source_matched_recovery.py",
             ROOT / "docs/SA_SOURCE_MATCHED_RECOVERY_ZH.md"]
    old.merge_pinned(inputs, {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths})
    jobs = scope(original, read_json(original_out / "analysis_records.json"), cfg)
    for job in jobs:
        row = old.receipt(original_out / "trials" / job["job_id"], job, original)
        require((row["status"] == "censored") == (job["role"] == "recovery"), "old label/status disagreement")
        if job["role"] == "recovery":
            require(row["stop"] == "hard_fuse", "unregistered censor reason")
    plan = dict(config=cfg, runtime_config=dict(original["config"], output=cfg["output"]), inputs=inputs,
                jobs=jobs, original_binding=original["binding"], native_sha256=original["native_sha256"],
                role="posthoc_resource_recovery_development", commit=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = json_fingerprint(plan)
    return plan, out


def verify():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg, "recovery config changed")
    require(plan["binding"] == json_fingerprint({k: v for k, v in plan.items() if k != "binding"}), "plan binding")
    for rel, digest in plan["inputs"].items():
        require(sha256_file(ROOT / rel) == digest, "registered input changed: " + rel)
    return plan, out


class Phases:
    def __init__(self, folder, binding, job_id, clock=time.monotonic):
        self.folder, self.clock = folder, clock
        self.start = clock()
        self.identity = dict(binding=binding, job_id=job_id)

    def mark(self, phase, **extra):
        self.folder.mkdir(parents=True, exist_ok=True)
        with (self.folder / "phases.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(dict(self.identity, phase=phase, elapsed=self.clock()-self.start,
                                        **extra), sort_keys=True, allow_nan=False) + "\n")


def runtime(plan):
    return dict(plan, config=plan["runtime_config"])


def restore(job, plan, deadline, phases):
    """Same frozen replay checks as v1; stage observations do not touch RNG."""
    from scripts import run_sa_path_quality as q
    from scripts.probe_sa_rejection_branches import check_attempt
    from scripts.run_sa_paired_completion_pilot import history_fingerprint
    from experiments.sa_history_selector import History
    cfg, entry = plan["runtime_config"], job["root"]
    phases.mark("load")
    require(q.native_identity()["sha256"] == plan["native_sha256"], "wrong native loaded")
    root = read_json(ROOT / entry["source_root"])
    target = root["source"]
    reg = read_json(ROOT / cfg["source"] / "registration.json")
    case = next(c for c in reg["cases"] if c["task_id"] == target["item"]["task_id"])
    worker = q.worker_job(case, target["item"], reg["template"], ROOT / cfg["output"] / "unused", plan["binding"])
    original = ROOT / cfg["source"] / "episodes" / target["item"]["job_id"]
    expected = read_json(original / "initial.json")["payload"]["observation"]
    old.remaining(deadline)
    phases.mark("reset")
    env = q._make_environment(worker["dataset_root"], worker["row"], dict(worker["environment"], time_limit=100000.), "Adaptive")
    state = q._plain(env.reset(seed=worker["solver_seed"]))
    phases.mark("initial_validation")
    require(q.state_fingerprint(state) == q.state_fingerprint(expected), "initial fingerprint mismatch")
    history = History(state)
    event = None
    phases.mark("prefix")
    with (original / "first_phase/trace.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            event = json.loads(line)
            require(event["decision"] == history.decision, "prefix decision gap")
            if event["decision"] == target["decision"]:
                break
            phases.mark("prefix_step", decision=event["decision"])
            seconds = min(cfg["pp_seconds"], old.remaining(deadline))
            raw = q._plain(env.step_experimental_pp(event["action"], seconds, "annealed", event["temperature"], event["uniform"]))
            if raw["metrics"]["pp_failure_reason"] == "time_limit":
                raise old.BudgetExpired("prefix PP safety limit")
            after = raw["observation"]
            expected = q.apply_state_delta(expected, event["delta"])
            check_attempt(raw["metrics"], event["metrics"])
            require(q.state_fingerprint(after) == q.state_fingerprint(expected), "prefix fingerprint mismatch")
            history.observe(state, event, after)
            state = after
    phases.mark("root_validation")
    require(event is not None and history.decision == target["decision"] == event["decision"], "missing root occurrence")
    require(q.state_fingerprint(state) == entry["root_fingerprint"] == root["state_fingerprint"], "root fingerprint mismatch")
    require(history_fingerprint(history) == target["history_fingerprint"], "history fingerprint mismatch")
    case = dict(case_id=worker["sa_case_id"], task_id=case["task_id"], solver_seed=worker["solver_seed"], proposal=worker["sa_proposal"])
    old.remaining(deadline)
    phases.mark("root_pool")
    index, pool = q.SingleFullCheckPool(case).select(env, state, target["decision"])
    require((index, pool) == (event["selected_index"], event["pool"]) ==
            (root["control_event"]["selected_index"], root["control_event"]["pool"]), "candidate pool mismatch")
    require(q.state_fingerprint(env.get_state()) == entry["root_fingerprint"], "proposal changed state")
    lookup = {c["candidate_id"]: c for c in pool}
    for candidate in entry["candidates"]:
        actual = lookup[candidate["candidate_id"]]
        require(actual["agents"] == candidate["agents"], "candidate membership changed")
        require(sorted(actual["selection_families"]) == sorted(candidate["selection_families"]), "candidate source changed")
    old.remaining(deadline)
    return env, root, case, lookup[job["candidate_id"]]


def rollout(env, root, case, candidate, job, plan, deadline, phases):
    from scripts import run_sa_path_quality as q
    from scripts.run_feedback_exploration_diagnostics import validate_final
    cfg = plan["runtime_config"]
    state = root["state"]
    require(q.state_fingerprint(env.get_state()) == root["state_fingerprint"], "rollout root changed")
    phases.mark("selector_init")
    selector = q.SingleFullCheckPool(case)
    events, initial_nodes, stop = [], state["low_level"]["generated"], "horizon"
    phases.mark("rollout")
    began = time.monotonic()
    for offset in range(cfg["horizon"]):
        if state["feasible"]:
            stop = "feasible"
            break
        if time.monotonic() >= deadline:
            stop = "wall_safety"
            break
        d = root["source"]["decision"] + offset
        phases.mark("proposal", decision=d)
        if offset == 0:
            pool = root["control_event"]["pool"]
            index = next(i for i, c in enumerate(pool) if c["candidate_id"] == candidate["candidate_id"])
        else:
            index, pool = selector.select(env, state, d)
        pp, uniform = randomization(root["source"]["id"], job["trial"], d, cfg)
        temp = q.temperature(d)
        action = dict(mode="explicit_neighborhood", agents=pool[index]["agents"], random_seed=pp)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop = "wall_safety"
            break
        phases.mark("repair", decision=d)
        raw = q._plain(env.step_experimental_pp(action, min(cfg["pp_seconds"], remaining), "annealed", temp, uniform))
        after, metrics = raw["observation"], raw["metrics"]
        phases.mark("transition_validation", decision=d)
        q.validate_transition(state, after, metrics, action["agents"], "annealed", temp, uniform)
        events.append(dict(decision=d, action=action, temperature=temp, uniform=uniform, metrics=metrics,
                           pool=pool, selected_index=index, delta=q.encode_state_delta(state, after)))
        state = after
        if metrics["pp_failure_reason"] == "time_limit" or not metrics["acceptance_evaluated"]:
            stop = "incomplete_pp"
            break
    if state["feasible"]:
        stop = "feasible"
    phases.mark("final_validation")
    validate_final(state)
    return dict(status="ok", binding=plan["binding"], root_id=job["root"]["state_id"],
                candidate_id=job["candidate_id"], trial=job["trial"], events=events, stop=stop,
                root_fingerprint=root["state_fingerprint"], final_fingerprint=q.state_fingerprint(state),
                final_conflicts=state["num_of_colliding_pairs"], final_cost=state["sum_of_costs"],
                generated=state["low_level"]["generated"]-initial_nodes,
                diagnostic_seconds=time.monotonic()-began, no_ttf=True)


def seal(folder, job, plan, timeout_salvaged=False):
    require({p.name for p in folder.iterdir()} == {"started.json", "result.json", "phases.jsonl"}, "unsealed files")
    once(folder / "receipt.json", dict(binding=plan["binding"], job_id=job["job_id"], timeout_salvaged=timeout_salvaged,
         files={p.name: sha256_file(p) for p in folder.iterdir()}))


def receipt(folder, job, plan):
    saved = read_json(folder / "receipt.json")
    require(saved["binding"] == plan["binding"] and saved["job_id"] == job["job_id"], "receipt identity")
    require(set(saved["files"]) == {"started.json", "result.json", "phases.jsonl"}, "receipt schema")
    require({p.name for p in folder.iterdir()} == set(saved["files"]) | {"receipt.json"}, "unexpected trial files")
    for name, digest in saved["files"].items():
        require(sha256_file(folder / name) == digest, "recovery bytes changed: " + name)
    row = read_json(folder / "result.json")
    require(row["binding"] == plan["binding"] and (row["root_id"], row["candidate_id"], row["trial"]) ==
            (job["root"]["state_id"], job["candidate_id"], job["trial"]), "trial identity")
    require(row["status"] in ("ok", "censored"), "failed trial requires review")
    return row


def summary(row, job, plan):
    label = old.validate_trial(row, job, runtime(plan))
    return dict(status="ok", job_id=job["job_id"], role=job["role"], stop=row["stop"], completion=label,
                valid=label is not None, censored=label is None)


def trial_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job["parent_pid"])
    plan, folder = job["plan"], Path(job["folder"])
    require(not folder.exists(), "partial recovery requires audit")
    phases = Phases(folder, plan["binding"], job["job_id"])
    deadline = phases.start + plan["config"]["trial_seconds"]
    atomic(folder / "started.json", dict(binding=plan["binding"], job_id=job["job_id"], pid=os.getpid()))
    try:
        env, root, case, candidate = restore(job, plan, deadline, phases)
        row = rollout(env, root, case, candidate, job, plan, deadline, phases)
    except old.BudgetExpired as exc:
        row = dict(status="censored", binding=plan["binding"], root_id=job["root"]["state_id"],
                   candidate_id=job["candidate_id"], trial=job["trial"], stop="prefix_or_total_safety", reason=str(exc))
    phases.mark("result_write")
    atomic(folder / "result.json", row)
    phases.mark("label_validation")
    result = summary(row, job, plan)
    phases.mark("seal")
    seal(folder, job, plan)
    return result


def failure(job, status, error):
    try:
        require(status == "timeout", "worker failed: " + str(error))
        folder, plan = Path(job["folder"]), job["plan"]
        old.archive_interrupted_writes(folder, job)
        if (folder / "receipt.json").exists():
            return summary(receipt(folder, job, plan), job, plan)
        if not (folder / "started.json").exists():
            atomic(folder / "started.json", dict(binding=plan["binding"], job_id=job["job_id"], startup_timeout=True))
        if not (folder / "phases.jsonl").exists():
            Phases(folder, plan["binding"], job["job_id"]).mark("startup_timeout")
        if (folder / "result.json").exists():
            row = read_json(folder / "result.json")
        else:
            row = dict(status="censored", binding=plan["binding"], root_id=job["root"]["state_id"],
                       candidate_id=job["candidate_id"], trial=job["trial"], stop="hard_fuse", reason=error)
            atomic(folder / "result.json", row)
        result = summary(row, job, plan)
        seal(folder, job, plan, timeout_salvaged=True)
        return result
    except Exception as exc:
        return dict(status="error", job_id=job["job_id"], error=repr(exc), original_failure=error)


def inventory(plan, out):
    base, jobs = out / "trials", plan["jobs"]
    if base.exists():
        require({p.name for p in base.iterdir()} <= {j["job_id"] for j in jobs}, "unregistered trial")
    pending, done = [], []
    for job in jobs:
        folder = base / job["job_id"]
        if folder.exists():
            require((folder / "receipt.json").exists(), "interrupted recovery requires review: " + job["job_id"])
            receipt(folder, job, plan)
            done.append(job)
        else:
            pending.append(dict(job, folder=str(folder), plan=plan, parent_pid=os.getpid()))
    return pending, done


def semantic_row(row):
    """Exclude only timing metadata, never paths, pools, scores or search counts."""
    if isinstance(row, dict):
        return {k: semantic_row(v) for k, v in row.items()
                if k != "binding" and k not in TIMING_FIELDS}
    if isinstance(row, list):
        return [semantic_row(v) for v in row]
    return row


def controls_pass(plan, out):
    for job in (j for j in plan["jobs"] if j["role"] == "control"):
        row = receipt(out / "trials" / job["job_id"], job, plan)
        original = read_json(ROOT / plan["config"]["original"] / "trials" / job["job_id"] / "result.json")
        require(semantic_row(row) == semantic_row(original), "control semantics changed: " + job["job_id"])
        require(old.validate_trial(row, job, runtime(plan)) is not None, "control censored")
    return True


def counts(results):
    return dict(sealed=len(results), valid=sum(r["completion"] is not None for r in results),
                feasible=sum(r["completion"] is True for r in results),
                horizon_nonfeasible=sum(r["completion"] is False for r in results),
                censored=sum(r["completion"] is None for r in results))


def collect(resume=False):
    from experiments.repair_collection import _CollectionRunLock, _run_jobs
    plan, out = verify()
    with _CollectionRunLock(out, plan["binding"], "source-matched-recovery"):
        require(resume or not (out / "trials").exists(), "existing recovery requires --resume")
        if resume and (out / "STOP_AFTER_BATCH").exists():
            (out / "STOP_AFTER_BATCH").unlink()
        try:
            for role in ("control", "recovery"):
                pending, _ = inventory(plan, out)
                pending = [j for j in pending if j["role"] == role]
                if role == "recovery" and not (out / "STOP_AFTER_BATCH").exists():
                    controls_pass(plan, out)
                for offset in range(0, len(pending), plan["config"]["workers"]):
                    if (out / "STOP_AFTER_BATCH").exists():
                        break
                    atomic(out / "run_status.json", dict(status="running", binding=plan["binding"], phase=role))
                    def progress(result):
                        with (out / "progress.jsonl").open("a", encoding="utf8") as stream:
                            stream.write(json.dumps(result, sort_keys=True) + "\n")
                        print(json.dumps(result, sort_keys=True), flush=True)
                    rows = _run_jobs(trial_worker, pending[offset:offset+plan["config"]["workers"]],
                        workers=plan["config"]["workers"], phase="source-matched-recovery",
                        output_root=out / "progress", run_fingerprint=plan["binding"],
                        timeout_seconds=plan["config"]["trial_seconds"], failure_result=failure,
                        on_result=progress, stop_on_failure=False)
                    require(all(r["status"] == "ok" for r in rows), "recovery error; batch drained")
                    _, done = inventory(plan, out)
                    summaries = [summary(receipt(out / "trials" / j["job_id"], j, plan), j, plan) for j in done]
                    atomic(out / "run_status.json", dict(status="running", binding=plan["binding"], **counts(summaries)))
                    verify()
            pending, done = inventory(plan, out)
            summaries = [summary(receipt(out / "trials" / j["job_id"], j, plan), j, plan) for j in done]
            atomic(out / "run_status.json", dict(status="paused" if pending else "completed", binding=plan["binding"], **counts(summaries)))
            return read_json(out / "run_status.json")
        except BaseException as exc:
            atomic(out / "run_status.json", dict(status="error", binding=plan["binding"], error=repr(exc)))
            raise


def merge_records(original, replacements):
    records = copy.deepcopy(original)
    lookup = {r["state_id"]: r for r in records}
    seen = set()
    for job, label in replacements:
        key = (job["root"]["state_id"], job["candidate_id"], job["trial"])
        require(key not in seen, "duplicate replacement")
        seen.add(key)
        values = lookup[key[0]]["values"][key[1]]
        require(values[key[2]] is None, "cannot replace observed label")
        values[key[2]] = label
    return records


def phase_records(path, timeout_salvaged):
    lines = path.read_text(encoding="utf8").splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            require(timeout_salvaged and i == len(lines)-1 and not line.endswith("\n"), "corrupt phase log")
            return dict(rows=rows, interrupted_final_line=True)
    return dict(rows=rows, interrupted_final_line=False)


def analyze():
    from experiments.sa_source_matched_analysis import summarize
    plan, out = verify()
    pending, done = inventory(plan, out)
    require(not pending, "recovery incomplete")
    controls_pass(plan, out)
    replacements, provenance, results, stages = [], [], [], {}
    for job in done:
        folder = out / "trials" / job["job_id"]
        row = receipt(folder, job, plan)
        result = summary(row, job, plan)
        results.append(result)
        stages[job["job_id"]] = phase_records(folder / "phases.jsonl", read_json(folder / "receipt.json")["timeout_salvaged"])
        if job["role"] == "recovery":
            replacements.append((job, result["completion"]))
        provenance.append(dict(job_id=job["job_id"], role=job["role"], completion=result["completion"],
                               files={p.relative_to(ROOT).as_posix(): sha256_file(p) for p in folder.iterdir()}))
    original = read_json(ROOT / plan["config"]["original"] / "analysis_records.json")
    records = merge_records(original, replacements)
    cfg = plan["runtime_config"]
    report = summarize(records, bootstrap_samples=cfg["bootstrap"], seed=cfg["bootstrap_seed"])
    report.update(binding=plan["binding"], role=plan["role"], no_ttf=True, training_allowed=False,
                  recovery_counts=counts([r for r in results if r["role"] == "recovery"]),
                  control_counts=counts([r for r in results if r["role"] == "control"]),
                  controls_exact=True, replacement_count=len(replacements),
                  provenance=provenance, original_report_sha256=plan["config"]["original_report_sha256"],
                  interrupted_write_hashes={p.relative_to(ROOT).as_posix(): sha256_file(p)
                                            for p in sorted((out / "interrupted_writes").glob("*/*"))},
                  independent_confirmation=False)
    once(out / "analysis_records.json", records)
    once(out / "phase_records.json", stages)
    once(out / "report.json", report)
    return {k: v for k, v in report.items() if k not in ("rows", "provenance")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "dry-run", "collect", "analyze", "stop"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        plan, out = build_plan()
        once(out / "plan.json", plan)
        result = dict(binding=plan["binding"], jobs=len(plan["jobs"]), controls=3, recovery=16)
    elif args.phase in ("verify", "dry-run"):
        plan, out = verify()
        pending, done = inventory(plan, out)
        result = dict(binding=plan["binding"], jobs=len(plan["jobs"]), pending=len(pending), sealed=len(done),
                      workers=4, max_prefix_steps=19*128, max_rollout_steps=19*32, hard_seconds_per_job=180,
                      serial_hard_minutes=57, ideal_parallel_batch_hard_minutes=15, no_ttf=True)
    elif args.phase == "collect":
        result = collect(args.resume)
    elif args.phase == "analyze":
        result = analyze()
    else:
        cfg = read_json(CONFIG)
        out = ROOT / cfg["output"]
        require(out.resolve().is_relative_to((ROOT / "build").resolve()), "unsafe stop output")
        plan = read_json(out / "plan.json")
        require(plan["binding"] == json_fingerprint({k: v for k, v in plan.items() if k != "binding"}), "stop plan identity")
        atomic(out / "STOP_AFTER_BATCH", dict(binding=plan["binding"]))
        result = dict(stop_requested=True, drain_active_jobs=True)
    print(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
