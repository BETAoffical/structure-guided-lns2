"""Read retained SA traces for feedback availability; no solver, fitting or policy replay."""
import argparse
from collections import Counter
from pathlib import Path
import json
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import audit_sa_cooling_traces as cooling

q = cooling.q
OUT = ROOT / "build/sa-training-readiness-v1"
PLATEAU = "build/sa-plateau-candidate-audit-v1/report.json"
FROZEN = {
    "build/sa-cooling-trace-audit-v2/report.json": "23ac28f030a8f1c9811ad3e7ef5add9c704f9ffe5a752b43114a35eaab400098",
    PLATEAU: "ac9edd853c82bfcaa908936f453e659009d6bb5acee74ae23bfcab1f7b8ab7ea",
    "build/sa-bounded-reheat-probe-v1/report.json": "4bc5d186aa6e6aa3b0322af4fe3f822dd5c22c2f60f0b24ec6bb4ad5b1c549b7",
}


class RepairConditions(cooling.PhysicalIdentity):
    """Visible PP inputs excluding sampled order, deadline and hidden native history."""
    def __init__(self, state, instance):
        super().__init__(state)
        self.instance = instance
        self.ids = sorted(self.paths)
        self.endpoints = {a["id"]: (a["start"], a["goal"]) for a in state["agents"]}

    def problem(self, state, agents, protocol):
        members = set(agents)
        if not members or len(members) != len(agents) or not members <= self.paths.keys():
            raise ValueError("invalid neighborhood members")
        bound = sum(bool(set(edge) & members) for edge in state["conflict_edges"])
        return q.json_fingerprint([
            self.instance, self.static, protocol,
            [(i, self.paths[i]) for i in self.ids if i not in members],
            [(i, self.endpoints[i]) for i in sorted(members)], bound,
        ])


def before_features(history, key):
    past = history.get(key, Counter())
    return {k: past[k] for k in ("attempts", "complete", "drop", "nondrop", "censored")}


def record(history, key, complete, drop):
    past = history.setdefault(key, Counter())
    past["attempts"] += 1
    past["complete"] += complete
    past["censored"] += not complete
    if complete:
        past["drop" if drop else "nondrop"] += 1


def check_saved(row, binding):
    payload = {k: v for k, v in row.items() if k != "integrity_sha256"}
    if row.get("integrity_sha256") != q.json_fingerprint(payload):
        raise ValueError("saved audit changed")
    if row.get("status") != "ok" or row.get("binding") != binding:
        raise ValueError("saved audit failed or belongs to another plan")


def alternative_summary(state):
    candidates = state["candidates"]
    if len({c["candidate_id"] for c in candidates}) != len(candidates):
        raise ValueError("duplicate candidate")
    selected = [c for c in candidates if c["selected"]]
    if len(selected) != 1:
        raise ValueError("selected candidate missing")
    for c in candidates:
        if len(c["conflicts"]) != 4 or c["censored"]:
            raise ValueError("expected four complete trials")
        if c["feasible"] != sum(x == 0 for x in c["conflicts"]):
            raise ValueError("feasibility disagreement")
    baseline = selected[0]["conflicts"]
    rows = []
    for c in candidates:
        xs = c["conflicts"]
        rows.append(dict(candidate_id=c["candidate_id"], selected=c["selected"],
            conflicts=xs, feasible=c["feasible"],
            paired_better=sum(x < b for x, b in zip(xs, baseline)),
            paired_worse=sum(x > b for x, b in zip(xs, baseline)),
            both_half_mean_better=all(statistics.mean(xs[j:j+2]) < statistics.mean(baseline[j:j+2]) for j in (0, 2)),
            mean_remaining=statistics.mean(xs)))
    return dict(task_id=state["task_id"], solver_seed=state["solver_seed"],
        root_id=state["job_id"], candidates=len(rows), trials=len(rows)*4,
        baseline=baseline, feasible_trials=sum(c["feasible"] for c in rows),
        candidates_both_halves_better=sum(c["both_half_mean_better"] for c in rows),
        candidates_all_four_better=sum(c["paired_better"] == 4 for c in rows), rows=rows)


def prepare():
    if (OUT / "plan.json").exists():
        raise ValueError("output exists; preserve it")
    old = cooling.verify()
    for name, sha in FROZEN.items():
        if q.sha256_file(ROOT / name) != sha:
            raise ValueError("frozen evidence changed: " + name)
    names = ["scripts/audit_sa_training_readiness.py", "tests/evaluation/test_sa_training_readiness.py",
        "docs/SA_TRAINING_READINESS_PROTOCOL_ZH.md", "docs/FEEDBACK_EXPLORATION_RESULT_ZH.md",
        "docs/STRIDE_HISTORYRANK_FEATURE_V1_REPORT.md", "docs/FEEDBACK_MEMORY_FEASIBILITY_AUDIT_ZH.md",
        "artifacts/initlns-closed-loop-controller-v2/main__realized_dynamic.json",
        "build/sa-cooling-trace-audit-v2/plan.json"]
    plan = dict(schema="lns2.sa_training_readiness.v1", jobs=old["jobs"], workers=20,
        inputs={**FROZEN, **{n: q.sha256_file(ROOT/n) for n in names}},
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        no_solver=True, no_training=True, no_threshold_selection=True)
    plan["fingerprint"] = q.json_fingerprint(plan)
    q.write_json(OUT / "plan.json", plan)
    return dict(episodes=len(plan["jobs"]), workers=20, no_solver=True)


def verify():
    plan = q.read_json(OUT / "plan.json")
    if plan["fingerprint"] != q.json_fingerprint({k: v for k, v in plan.items() if k != "fingerprint"}):
        raise ValueError("plan changed")
    for name, sha in plan["inputs"].items():
        if q.sha256_file(ROOT / name) != sha:
            raise ValueError("input changed: " + name)
    cooling.verify()
    return plan


def episode(job):
    folder = ROOT / job["folder"]
    for name, sha in job["files"].items():
        if q.sha256_file(folder/name) != sha:
            raise ValueError("trace source changed")
    state = q.execution.read_artifact(folder/"initial.json", job["binding"])["observation"]
    if q.state_fingerprint(state) != job["initial_fingerprint"]:
        raise ValueError("initial fingerprint")
    conditions = RepairConditions(state, job["binding"])
    histories = {k: {} for k in ("agents", "problem", "temperature_budget")}
    counts, rows = Counter(), []
    best, gap = state["num_of_colliding_pairs"], 0
    for d, e in enumerate(q.read_jsonl(folder/"first_phase/trace.jsonl")):
        if e["decision"] != d or state["feasible"]:
            raise ValueError("invalid event order")
        m, action, pool = e["metrics"], e["action"], e["pool"]
        selected = e["selected_index"]
        if pool[selected]["agents"] != action["agents"]:
            raise ValueError("candidate/action disagreement")
        protocol = {k: v for k, v in action.items() if k not in ("agents", "random_seed")}
        budget = m["requested_pp_time_limit_seconds"]
        keys = [conditions.problem(state, c["agents"], protocol) for c in pool]
        chosen_keys = dict(agents=q.json_fingerprint(sorted(action["agents"])), problem=keys[selected],
            temperature_budget=q.json_fingerprint([keys[selected], e["temperature"], budget]))
        features = {k: before_features(histories[k], key) for k, key in chosen_keys.items()}
        pool_history = [before_features(histories["problem"], key) for key in keys]
        # All history queries precede the current outcome and its insertion.
        row = dict(decision=d, temperature=e["temperature"], best_gap_before=gap,
            before_conflicts=state["num_of_colliding_pairs"], candidate_count=len(pool),
            prior=features, prior_observed_slots=sum(h["attempts"] > 0 for h in pool_history),
            prior_positive_slots=sum(h["drop"] > 0 for h in pool_history))
        after = q.apply_state_delta(state, e["delta"])
        cooling.check_acceptance(e, state["num_of_colliding_pairs"], after["num_of_colliding_pairs"])
        complete = bool(m["acceptance_evaluated"])
        drop = after["num_of_colliding_pairs"] < state["num_of_colliding_pairs"]
        row["observed"] = dict(complete=complete, strict_drop=drop,
            after_conflicts=after["num_of_colliding_pairs"], accepted=bool(m["replan_success"]))
        counts.update(dict(steps=1, candidate_slots=len(pool), complete=complete, censored=not complete,
            strict_drops=drop, steps_with_prior_positive_slot=row["prior_positive_slots"] > 0,
            prior_observed_slots=row["prior_observed_slots"], prior_positive_slots=row["prior_positive_slots"]))
        for kind, history in histories.items():
            prev = features[kind]
            counts[kind+"_repeated"] += prev["attempts"] > 0
            counts[kind+"_with_prior_drop"] += prev["drop"] > 0
            counts[kind+"_drop_after_prior_drop"] += complete and drop and prev["drop"] > 0
            counts[kind+"_drop_after_prior_nondrop"] += complete and drop and prev["nondrop"] > 0
            record(history, chosen_keys[kind], complete, drop)
        gap = 0 if after["num_of_colliding_pairs"] < best else gap+1
        best = min(best, after["num_of_colliding_pairs"])
        conditions.update(e["delta"], after)
        rows.append(row)
        state = after
    terminal = q.execution.read_artifact(folder/"terminal.json", job["binding"])
    if q.state_fingerprint(state) != terminal["state_fingerprint"] or bool(state["feasible"]) != job["success"]:
        raise ValueError("terminal mismatch")
    return dict(status="ok", job_id=job["job_id"], cohort=job["cohort"], task_id=job["task_id"],
        map_hash=conditions.static, success=job["success"], binding=job["audit_binding"], counts=dict(counts), rows=rows)


def run(resume=False):
    plan = verify()
    with q._CollectionRunLock(OUT, plan["fingerprint"], "sa-training-readiness"):
        pending = []
        for j in plan["jobs"]:
            path = OUT/"episodes"/(j["job_id"]+".json")
            if path.exists():
                r = q.read_json(path)
                if not resume:
                    raise ValueError("existing result needs inspection")
                check_saved(r, plan["fingerprint"])
            else:
                pending.append(dict(j, audit_binding=plan["fingerprint"]))
        def failed(job, status, error):
            return dict(status=status, job_id=job["job_id"], error=str(error), binding=plan["fingerprint"])
        def save(row):
            row["integrity_sha256"] = q.json_fingerprint(row)
            q.write_json(OUT/"episodes"/(row["job_id"]+".json"), row)
            print(row["job_id"], row["status"], flush=True)
        q.write_json(OUT/"status.json", dict(status="running", jobs=len(pending)))
        try:
            q._run_jobs(episode, pending, workers=20, phase="sa-readiness", output_root=OUT/"progress",
                run_fingerprint=plan["fingerprint"], timeout_seconds=600,
                failure_result=failed, on_result=save, stop_on_failure=True)
            rows = [q.read_json(OUT/"episodes"/(j["job_id"]+".json")) for j in plan["jobs"]]
            original = {r["job_id"]: r for r in q.read_json(ROOT/"build/sa-cooling-trace-audit-v2/report.json")["episodes"]}
            for row, job in zip(rows, plan["jobs"]):
                check_saved(row, plan["fingerprint"])
                old = original[job["job_id"]]
                if row["job_id"] != job["job_id"] or len(row["rows"]) != old["bins"].get("all", {}).get("steps", 0):
                    raise ValueError("episode identity or count changed")
            totals = {}
            for cohort in ("full_120", "failure_union_300"):
                group = [r for r in rows if r["cohort"] == cohort]
                totals[cohort] = {}
                for label, subset in (("all", group), ("success", [r for r in group if r["success"]]),
                                      ("failure", [r for r in group if not r["success"]])):
                    c = Counter()
                    for r in subset:
                        c.update(r["counts"])
                    totals[cohort][label] = dict(episodes=len(subset), maps=len({r["map_hash"] for r in subset}), **c)
            frozen = q.read_json(ROOT/PLATEAU)
            for name, sha in frozen["files"].items():
                if q.sha256_file((ROOT/PLATEAU).parent/name) != sha:
                    raise ValueError("counterfactual source changed")
            candidates = [alternative_summary(s) for s in frozen["states"]]
            verify()
            report = dict(schema="lns2.sa_training_readiness_report.v1", complete=True, binding=plan["fingerprint"],
                cohorts=totals, counterfactual=candidates, new_solver_calls=0, new_models=0,
                decision="not_ready_for_sa_retraining_chosen_action_feedback_is_not_counterfactual_labels",
                files={"episodes/"+j["job_id"]+".json": q.sha256_file(OUT/"episodes"/(j["job_id"]+".json")) for j in plan["jobs"]})
            q.write_json(OUT/"report.json", report)
            q.write_json(OUT/"status.json", dict(status="complete", episodes=len(rows)))
            return dict(complete=True, cohorts=totals, decision=report["decision"])
        except BaseException as exc:
            q.write_json(OUT/"status.json", dict(status="failed_or_interrupted", error=repr(exc)))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "run"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare() if args.phase == "prepare" else run(args.resume)))
