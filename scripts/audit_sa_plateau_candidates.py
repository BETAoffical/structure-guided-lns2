"""Bounded existing-candidate audit at three recorded SA plateaus; never TTF."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_sa_pressure_300s as source
from scripts.probe_sa_rejection_branches import check_attempt
from scripts.run_feedback_exploration_diagnostics import validate_final

q = source.q
OUT = ROOT / "build/sa-plateau-candidate-audit-v1"
TARGETS = ("3adb808c48cb7fd455e4fbe3", "0decb088a8e0529a3c77c2bf", "ce45f43b848696ee701f8b60")
REPORT_SHA = "b5d18477c724fc7b867fa45995cd691e2f9c5f24dfff53388bd041e6eb492945"


def target_index(events):
    return next(i for i, e in enumerate(events) if e["elapsed_seconds"] >= 120.)


def coverage(state, candidate):
    members = set(candidate["agents"])
    edges = state["conflict_edges"]
    return dict(internal_pairs=sum(a in members and b in members for a, b in edges),
                incident_pairs=sum(a in members or b in members for a, b in edges),
                nonconflicting_members=sorted(members - {a for e in edges for a in e}))


def load_source(r, item):
    folder = source.OUT / "episodes" / item["job_id"]
    manifest = q.read_json(source.OUT / "manifest.json")
    for name, sha in manifest["jobs"][item["job_id"]]["files"].items():
        if q.sha256_file(folder / name) != sha:
            raise ValueError("source episode changed")
    spec = source.spec_for(r, item, source.anchors(r))
    q.audit_trace(spec)
    state = q.execution.read_artifact(folder / "initial.json", spec["binding"])["observation"]
    events = list(q.read_jsonl(folder / "first_phase/trace.jsonl"))
    return spec, state, events


def prepare():
    if (OUT / "plan.json").exists():
        raise ValueError("plan exists; no overwrite")
    r = source.verify()
    if q.sha256_file(source.OUT / "analysis/report.json") != REPORT_SHA:
        raise ValueError("source report changed")
    targets = []
    for identity in TARGETS:
        item = next(i for i in r["schedule"] if i["job_id"] == identity)
        _, state, events = load_source(r, item)
        d = target_index(events)
        for e in events[:d]:
            state = q.apply_state_delta(state, e["delta"])
        e = events[d]
        late = [x for x in events if x["elapsed_seconds"] >= 120.]
        ids = Counter(x["pool"][x["selected_index"]]["candidate_id"] for x in late)
        targets.append(dict(job_id=identity, item=item, decision=d, conflicts=state["num_of_colliding_pairs"],
            state_fingerprint=q.state_fingerprint(state), selected_index=e["selected_index"],
            pool=e["pool"], temperature=e["temperature"], uniform=e["uniform"],
            coverage=[coverage(state, c) for c in e["pool"]],
            late_decisions=len(late), late_selected_counts=dict(ids),
            late_strict_improvements=sum(x["metrics"]["conflicts_after"] < x["metrics"]["conflicts_before"] for x in late)))
    names = ["scripts/audit_sa_plateau_candidates.py", "tests/evaluation/test_sa_plateau_candidates.py",
             "docs/SA_PLATEAU_CANDIDATE_PROTOCOL_ZH.md"]
    plan = dict(schema="lns2.sa_plateau_candidates.v1", targets=targets, trials=4, workers=20,
        pp_seconds=30., branch_fuse_seconds=90., replay_pp_seconds=300.,
        source_report_sha256=REPORT_SHA, source_registration_sha256=q.sha256_file(source.OUT / "registration.json"),
        source_manifest_sha256=q.sha256_file(source.OUT / "manifest.json"),
        inputs={n:q.sha256_file(ROOT/n) for n in names},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        no_ttf=True, no_controller_change=True, no_continuation=True)
    plan["fingerprint"] = q.json_fingerprint(plan)
    q.write_json(OUT / "plan.json", plan)
    return dict(states=3, candidates=sum(len(t["pool"]) for t in targets),
                jobs=sum(len(t["pool"])*4+1 for t in targets), replay_steps=sum(t["decision"] for t in targets),
                workers=20, repetitions_per_candidate=4, timing_experiment=False)


def verify(native=False):
    plan = q.read_json(OUT / "plan.json")
    if plan["fingerprint"] != q.json_fingerprint({k:v for k,v in plan.items() if k != "fingerprint"}):
        raise ValueError("plan changed")
    for name, sha in plan["inputs"].items():
        if q.sha256_file(ROOT/name) != sha:
            raise ValueError("audit implementation changed")
    for name, key in (("registration.json", "source_registration_sha256"),
                      ("manifest.json", "source_manifest_sha256"), ("analysis/report.json", "source_report_sha256")):
        if q.sha256_file(source.OUT/name) != plan[key]:
            raise ValueError("source identity changed")
    return plan, source.verify(native)


def branch_specs(target, event):
    result = [dict(job_id="control", index=target["selected_index"], trial=-1,
                   action=event["action"], uniform=event["uniform"])]
    for trial in range(4):
        pp_seed = q.seed(target["job_id"], 20260913, trial, "plateau-pp")
        for index, c in enumerate(target["pool"]):
            result.append(dict(job_id=f"c{index:02d}-t{trial}", index=index, trial=trial,
                action=dict(mode="explicit_neighborhood", agents=c["agents"], random_seed=pp_seed),
                uniform=event["uniform"]))
    return result


def branch(env, before, target, event, job, plan, path):
    try:
        if q.state_fingerprint(env.get_state()) != target["state_fingerprint"]:
            raise ValueError("fork root changed")
        raw = q._plain(env.step_experimental_pp(job["action"], plan["pp_seconds"], "annealed",
                                               target["temperature"], job["uniform"]))
        after, metrics = raw["observation"], raw["metrics"]
        if not metrics["action_valid"] or not metrics["step_applied"]:
            raise ValueError("invalid action")
        q.validate_transition(before, after, metrics, job["action"]["agents"], "annealed",
                              target["temperature"], job["uniform"])
        validate_final(after)
        if job["trial"] == -1:
            check_attempt(metrics, event["metrics"])
            if q.state_fingerprint(after) != q.state_fingerprint(q.apply_state_delta(before, event["delta"])):
                raise ValueError("original action failed exact replay")
        row = dict(status="ok", **job, root_fingerprint=target["state_fingerprint"],
                   metrics=metrics, delta=q.encode_state_delta(before, after),
                   final_fingerprint=q.state_fingerprint(after), feasible=after["feasible"],
                   conflicts=after["num_of_colliding_pairs"], censored=raw["truncated"] or metrics["pp_failure_reason"]=="time_limit")
    except Exception as exc:
        row = dict(status="error", **job, error=repr(exc))
    row["binding"] = plan["fingerprint"]
    q.write_json(path, row)


def run_forks(env, state, target, event, plan, resume):
    # Fork only a single-threaded parent: each repair gets identical native memory and isolated rand().
    if sys.platform != "linux" or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise ValueError("native fork requires a single-threaded Linux parent")
    ctx = multiprocessing.get_context("fork")
    folder = OUT / "branches" / target["job_id"]
    folder.mkdir(parents=True, exist_ok=True)
    pending = []
    for job in branch_specs(target, event):
        path = folder / (job["job_id"]+".json")
        if path.exists():
            row = q.read_json(path)
            if not resume or row["binding"] != plan["fingerprint"] or row["status"] != "ok":
                raise ValueError("existing branch needs audit")
        else:
            pending.append(job)
    active = []
    try:
        while pending or active:
            while pending and len(active) < plan["workers"] and not (OUT/"STOP_AFTER_JOB.json").exists():
                job = pending.pop(0)
                path = folder / (job["job_id"]+".json")
                process = ctx.Process(target=branch, args=(env,state,target,event,job,plan,path))
                process.start()
                active.append((process,time.monotonic(),path))
            for entry in list(active):
                process, began, path = entry
                if process.is_alive() and time.monotonic()-began <= plan["branch_fuse_seconds"]:
                    continue
                if process.is_alive():
                    process.terminate()
                    process.join()
                    q.write_json(path, dict(status="timeout", binding=plan["fingerprint"]))
                    raise ValueError("branch hard timeout; unknown, not infeasible")
                process.join()
                if process.exitcode != 0 or not path.exists() or q.read_json(path)["status"] != "ok":
                    raise ValueError("branch failure: "+str(path))
                active.remove(entry)
            q.write_json(OUT/"status.json",dict(status="branches", target=target["job_id"], pending=len(pending), active=len(active)))
            if pending and not active and (OUT/"STOP_AFTER_JOB.json").exists():
                return False
            time.sleep(.1)
    finally:
        for process, _, _ in active:
            if process.is_alive(): process.terminate()
            process.join()
    if q.state_fingerprint(env.get_state()) != target["state_fingerprint"]:
        raise ValueError("child modified parent")
    return True


def collect(resume=False):
    plan, r = verify(True)
    with q._CollectionRunLock(OUT,plan["fingerprint"],"plateau-candidates"):
        for t in plan["targets"]:
            if (OUT/"STOP_AFTER_JOB.json").exists(): return dict(paused=True)
            spec, expected, events = load_source(r,t["item"])
            job = spec["worker_job"]
            config = dict(job["environment"],time_limit=100000.)
            env = q._make_environment(job["dataset_root"],job["row"],config,"Adaptive")
            state = q._plain(env.reset(seed=job["solver_seed"]))
            if q.state_fingerprint(state) != q.state_fingerprint(expected): raise ValueError("reset mismatch")
            print("REPLAY", t["job_id"], t["decision"], flush=True)
            for d,e in enumerate(events[:t["decision"]]):
                result = q._plain(env.step_experimental_pp(e["action"],plan["replay_pp_seconds"],"annealed",e["temperature"],e["uniform"]))
                state = result["observation"]
                expected = q.apply_state_delta(expected,e["delta"])
                if q.state_fingerprint(state) != q.state_fingerprint(expected): raise ValueError(f"prefix mismatch {d}")
                check_attempt(result["metrics"],e["metrics"])
                if d % 200 == 0: print("PREFIX",d,flush=True)
            if q.state_fingerprint(state) != t["state_fingerprint"]: raise ValueError("target mismatch")
            case = dict(case_id=job["sa_case_id"],task_id=job["row"]["task_id"],solver_seed=job["solver_seed"],proposal=job["sa_proposal"])
            index,pool = q.SingleFullCheckPool(case).select(env,state,t["decision"])
            if (index,pool) != (t["selected_index"],t["pool"]): raise ValueError("pool or score mismatch")
            validate_final(state)
            q.write_json(OUT/"roots"/(t["job_id"]+".json"),state)
            if not run_forks(env,state,t,events[t["decision"]],plan,resume): return dict(paused=True)
            print("DONE",t["job_id"],flush=True)
        report = analyze()
        q.write_json(OUT/"status.json",dict(status="complete",jobs=report["jobs"]))
        return report


def analyze():
    plan, _ = verify()
    results, files = [], {}
    for t in plan["targets"]:
        before = q.read_json(OUT/"roots"/(t["job_id"]+".json"))
        if q.state_fingerprint(before) != t["state_fingerprint"]: raise ValueError("saved root changed")
        rows = []
        for path in sorted((OUT/"branches"/t["job_id"]).glob("*.json")):
            row = q.read_json(path)
            if row["status"] != "ok" or row["binding"] != plan["fingerprint"]: raise ValueError("invalid result")
            after = q.apply_state_delta(before,row["delta"])
            if q.state_fingerprint(after) != row["final_fingerprint"] or after["num_of_colliding_pairs"] != row["conflicts"]:
                raise ValueError("saved branch mismatch")
            q.validate_transition(before,after,row["metrics"],row["action"]["agents"],"annealed",t["temperature"],row["uniform"])
            validate_final(after)
            files[path.relative_to(OUT).as_posix()] = q.sha256_file(path)
            rows.append(row)
        expected={(i,trial) for i in range(len(t["pool"])) for trial in range(4)}|{(t["selected_index"],-1)}
        if len(rows)!=len(expected) or {(x["index"],x["trial"]) for x in rows}!=expected: raise ValueError("trial coverage mismatch")
        candidates=[]
        for i,c in enumerate(t["pool"]):
            trials=sorted((x for x in rows if x["index"]==i and x["trial"]>=0),key=lambda x:x["trial"])
            candidates.append(dict(candidate_id=c["candidate_id"], agents=c["agents"], families=c["selection_families"],
                score=c["score"],selected=i==t["selected_index"],coverage=t["coverage"][i],
                conflicts=[x["conflicts"] for x in trials],feasible=sum(x["feasible"] for x in trials),
                censored=sum(x["censored"] for x in trials)))
        results.append(dict(job_id=t["job_id"],task_id=t["item"]["task_id"],solver_seed=t["item"]["solver_seed"],
            decision=t["decision"],initial_conflicts=t["conflicts"],candidates=candidates,
            original_action_replayed=True,late_decisions=t["late_decisions"],late_selected_counts=t["late_selected_counts"],
            late_strict_improvements=t["late_strict_improvements"]))
    report=dict(schema="lns2.sa_plateau_candidate_report.v1",complete=True,jobs=len(files),states=results,
        binding=plan["fingerprint"],files=files,no_ttf=True,no_promotion=True,no_continuation=True)
    q.write_json(OUT/"report.json",report)
    return dict(complete=True,jobs=len(files),summary=[dict(job_id=t["job_id"],conflicts=t["initial_conflicts"],
        selected=next(c["conflicts"] for c in t["candidates"] if c["selected"]),
        improving_candidates=sum(any(x<t["initial_conflicts"] for x in c["conflicts"]) for c in t["candidates"]),
        feasible_candidates=sum(c["feasible"]>0 for c in t["candidates"])) for t in results])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","collect","analyze"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    result=prepare() if args.phase=="prepare" else collect(args.resume) if args.phase=="collect" else analyze()
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    main()
