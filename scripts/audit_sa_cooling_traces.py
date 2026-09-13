"""Read-only cooling/physical-recurrence audit; never invokes a solver."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_sa_pressure_300s as extended
from scripts.audit_sa_trace_opportunities import recovery_window
from experiments.nonmonotonic_repair import probability

p, q = extended.p, extended.q
OUT = ROOT / "build/sa-cooling-trace-audit-v2"
REPORTS = {
    "full_120": (p.OUT, extended.SOURCE_SHA, 96),
    "failure_union_300": (extended.OUT, "b5d18477c724fc7b867fa45995cd691e2f9c5f24dfff53388bd041e6eb492945", 11),
}


def iteration_bin(decision):
    if decision < 0: raise ValueError("negative decision")
    return "0-99" if decision < 100 else "100-499" if decision < 500 else "500-999" if decision < 1000 else "1000+"


def conflict_bin(count):
    if count <= 0: raise ValueError("terminal state has no decision")
    return "1-5" if count <= 5 else "6-20" if count <= 20 else "21+"


def require_source_binding(expected, saved):
    if expected != saved:
        raise ValueError("source binding differs: prepare in the original WSL checkout path; do not weaken identity checks")


class PhysicalIdentity:
    """Analysis-only identity. Never replaces the solver/RNG replay fingerprint."""
    def __init__(self, state):
        self.paths = {a["id"]: q.json_fingerprint(a["path"]) for a in state["agents"]}
        if len(self.paths) != len(state["agents"]): raise ValueError("duplicate agent")
        self.static = q.json_fingerprint([state["rows"], state["cols"], state["obstacles"]])

    def update(self, delta, after):
        change = delta["agents"]
        if change["mode"] == "replace":
            self.paths = {a["id"]: q.json_fingerprint(a["path"]) for a in after["agents"]}
        elif change["mode"] == "patch":
            for patch in change["patches"]:
                if "path" in patch["set"]:
                    self.paths[patch["id"]] = q.json_fingerprint(patch["set"]["path"])
        else: raise ValueError("unknown agent delta")
        return self.key(after)

    def key(self, state):
        payload = self.static + "".join(f"{i}:{h};" for i,h in sorted(self.paths.items()))
        payload += q.json_fingerprint(state["conflict_edges"])
        return hashlib.sha256(payload.encode("ascii")).hexdigest()


def check_acceptance(e, before, after):
    m = e["metrics"]
    if not m["action_valid"] or not m["step_applied"]: raise ValueError("invalid action")
    if sorted(e["action"]["agents"]) != sorted(m["neighborhood"]): raise ValueError("modified action")
    if (before,after) != (m["conflicts_before"],m["conflicts_after"]): raise ValueError("conflict discontinuity")
    if (e["temperature"],e["uniform"]) != (m["acceptance_temperature"],m["acceptance_uniform"]):
        raise ValueError("acceptance inputs differ")
    if e["temperature"] != q.temperature(e["decision"]): raise ValueError("cooling changed")
    delta = m["pp_attempt_conflict_pair_count"]-m["pp_old_conflict_pair_count"]
    if m["acceptance_evaluated"]:
        expected = probability(delta,e["temperature"])
        if not math.isclose(expected,m["acceptance_probability"],rel_tol=1e-12,abs_tol=1e-14):
            raise ValueError("probability mismatch")
        if m["replan_success"] != (delta <= 0 or e["uniform"] < expected): raise ValueError("acceptance mismatch")
    if m["replan_success"]:
        if m["pp_rolled_back"] or after-before != delta: raise ValueError("accepted delta mismatch")
    elif after != before: raise ValueError("rejection failed rollback")
    return delta


def prepare():
    if (OUT/"plan.json").exists(): raise ValueError("plan exists; preserve prior output")
    registrations = {"full_120":p.verify(),"failure_union_300":extended.verify()}
    jobs,inputs=[],{}
    for cohort,(folder,sha,expected) in REPORTS.items():
        path=folder/"analysis/report.json"
        if q.sha256_file(path)!=sha: raise ValueError("report changed")
        r=registrations[cohort]
        report,manifest=q.read_json(path),q.read_json(folder/"manifest.json")
        if not report["complete"] or manifest["binding"]!=r["fingerprint"]: raise ValueError("incomplete source")
        module=p if cohort=="full_120" else extended
        anchors=module.anchors(r)
        items=[i for i in r["schedule"] if i["controller"]=="dual16_sa"]
        if len(items)!=expected: raise ValueError("cohort size changed")
        for i in items:
            row=next(x for x in report["episodes"] if x["job_id"]==i["job_id"])
            spec=module.spec_for(r,i,anchors)
            initial=folder/"episodes"/i["job_id"]/"initial.json"
            require_source_binding(spec["binding"],q.read_json(initial)["binding"])
            q.execution.read_artifact(initial,spec["binding"])
            jobs.append(dict(job_id=cohort+"-"+i["job_id"],cohort=cohort,source_job_id=i["job_id"],
                task_id=i["task_id"],solver_seed=i["solver_seed"],success=row["success"],
                folder=(folder/"episodes"/i["job_id"]).relative_to(ROOT).as_posix(),
                binding=spec["binding"],initial_fingerprint=row["initial_fingerprint"],
                files=manifest["jobs"][i["job_id"]]["files"]))
        for name in ("registration.json","manifest.json","analysis/report.json"):
            inputs[(folder/name).relative_to(ROOT).as_posix()]=q.sha256_file(folder/name)
    names=["scripts/audit_sa_cooling_traces.py","tests/evaluation/test_sa_cooling_traces.py",
        "docs/SA_COOLING_TRACE_PROTOCOL_ZH.md","docs/SA_REJECTION_BRANCH_RESULTS_ZH.md",
        "docs/SA_PLATEAU_CANDIDATE_RESULTS_ZH.md"]
    inputs.update({n:q.sha256_file(ROOT/n) for n in names})
    plan=dict(schema="lns2.sa_cooling_trace.v1",jobs=jobs,inputs=inputs,workers=20,
              no_solver=True,no_policy_change=True,no_counterfactual_prediction=True,
              source_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    plan["fingerprint"]=q.json_fingerprint(plan)
    q.write_json(OUT/"plan.json",plan)
    return dict(episodes=len(jobs),cohorts={k:v[2] for k,v in REPORTS.items()},workers=20,no_solver=True)


def verify():
    plan=q.read_json(OUT/"plan.json")
    if plan["fingerprint"]!=q.json_fingerprint({k:v for k,v in plan.items() if k!="fingerprint"}):
        raise ValueError("plan changed")
    for name,sha in plan["inputs"].items():
        if q.sha256_file(ROOT/name)!=sha: raise ValueError("registered input changed: "+name)
    return plan


def episode(job):
    folder=ROOT/job["folder"]
    for name,sha in job["files"].items():
        if q.sha256_file(folder/name)!=sha: raise ValueError("source episode changed")
    state=q.execution.read_artifact(folder/"initial.json",job["binding"])["observation"]
    if q.state_fingerprint(state)!=job["initial_fingerprint"]: raise ValueError("initial fingerprint mismatch")
    physical=PhysicalIdentity(state)
    key=physical.key(state)
    seen={key}
    seen_actions=set()
    best=state["num_of_colliding_pairs"]
    best_gap=0
    counts=[best]
    bins={}
    attempts=[]
    for d,e in enumerate(q.read_jsonl(folder/"first_phase/trace.jsonl")):
        if e["decision"]!=d or counts[-1]==0: raise ValueError("trace sequence invalid")
        chosen=e["pool"][e["selected_index"]]
        if chosen["agents"]!=e["action"]["agents"]: raise ValueError("candidate differs from action")
        case_id=f"{job['task_id']}-seed{job['solver_seed']}"
        if e["uniform"]!=q.acceptance_draw(q.seed(case_id,0,d,"accept")): raise ValueError("draw identity changed")
        after=q.apply_state_delta(state,e["delta"])
        c=after["num_of_colliding_pairs"]
        delta=check_acceptance(e,counts[-1],c)
        newkey=physical.update(e["delta"],after)
        m=e["metrics"]
        complete=m["acceptance_evaluated"]
        worse=complete and delta>0
        actionkey=(key,tuple(chosen["agents"]))
        tags=("all", "iteration/"+iteration_bin(d),"conflicts/"+conflict_bin(counts[-1]),
              "best_gap/"+("32+" if best_gap>=32 else "0-31"))
        flags=dict(steps=1,complete=complete,incomplete=not complete,worse=worse,
                   worse_accepted=worse and m["replan_success"],worse_rejected=worse and not m["replan_success"],
                   strict_decrease=c<counts[-1],strict_best=c<best,equal=c==counts[-1],
                   physical_unchanged=key==newkey,physical_revisited=newkey in seen,
                   repeated_state_candidate=actionkey in seen_actions,
                   probability_below_01=worse and m["acceptance_probability"]<.01)
        for tag in tags: bins.setdefault(tag,Counter()).update(flags)
        if worse:
            attempts.append(dict(decision=d,before=counts[-1],after=c,delta=delta,
                temperature=e["temperature"],probability=m["acceptance_probability"],accepted=m["replan_success"],
                relative_increase=delta/max(1,counts[-1]),best_gap_before=best_gap,
                repeated_state_candidate=actionkey in seen_actions,physical_unchanged=key==newkey))
        seen_actions.add(actionkey)
        seen.add(newkey)
        best_gap=0 if c<best else best_gap+1
        best=min(best,c)
        key,state=newkey,after
        counts.append(c)
    terminal=q.execution.read_artifact(folder/"terminal.json",job["binding"])
    if q.state_fingerprint(state)!=terminal["state_fingerprint"]: raise ValueError("terminal fingerprint mismatch")
    feasible=state["feasible"]
    for attempt in attempts:
        if attempt["accepted"]:
            future=counts[attempt["decision"]+2:]
            attempt["observed_followup"]={str(h):recovery_window(future,attempt["before"],h,feasible) for h in (4,16,64)}
    return dict(status="ok",job_id=job["job_id"],cohort=job["cohort"],task_id=job["task_id"],
        solver_seed=job["solver_seed"],success=job["success"],binding=job["audit_binding"],
        final_conflicts=counts[-1],best_conflicts=best,unique_physical_states=len(seen),
        bins={k:dict(v) for k,v in bins.items()},attempts=attempts,no_solver=True)


def aggregate(rows):
    totals={}
    for cohort in REPORTS:
        group=[r for r in rows if r["cohort"]==cohort]
        if len(group)!=REPORTS[cohort][2]: raise ValueError("missing cohort episodes")
        totals[cohort]={}
        for label,subset in (("all",group),("success",[r for r in group if r["success"]]),("failure",[r for r in group if not r["success"]])):
            bins={}
            attempts=[a for r in subset for a in r["attempts"]]
            for r in subset:
                for k,c in r["bins"].items(): bins.setdefault(k,Counter()).update(c)
            accepted=[a for a in attempts if a["accepted"]]
            totals[cohort][label]=dict(episodes=len(subset),bins={k:dict(v) for k,v in bins.items()},
                worse_delta_median=statistics.median(a["delta"] for a in attempts) if attempts else None,
                probability_median=statistics.median(a["probability"] for a in attempts) if attempts else None,
                accepted_followup={str(h):dict(Counter(a["observed_followup"][str(h)] for a in accepted)) for h in (4,16,64)},
                failures_without_rejected_worse=[r["job_id"] for r in subset if not r["success"] and not r["bins"].get("all",{}).get("worse_rejected",0)])
    return totals


def run(resume=False):
    plan=verify()
    with q._CollectionRunLock(OUT,plan["fingerprint"],"read-only-cooling"):
        pending=[]
        for job in plan["jobs"]:
            path=OUT/"episodes"/(job["job_id"]+".json")
            if path.exists():
                row=q.read_json(path)
                if not resume or row["binding"]!=plan["fingerprint"] or row["status"]!="ok": raise ValueError("prior audit needs inspection")
            else: pending.append(dict(job,audit_binding=plan["fingerprint"]))
        def failed(job,status,error): return dict(status=status,job_id=job["job_id"],error=str(error),binding=plan["fingerprint"])
        def save(row):
            q.write_json(OUT/"episodes"/(row["job_id"]+".json"),row)
            print(row["job_id"],row["status"],flush=True)
        completed=q._run_jobs(episode,pending,workers=20,phase="read-only-cooling",output_root=OUT/"progress",
            run_fingerprint=plan["fingerprint"],timeout_seconds=300,failure_result=failed,on_result=save,stop_on_failure=True)
        missing=[j["job_id"] for j in plan["jobs"] if not (OUT/"episodes"/(j["job_id"]+".json")).exists()]
        errors=[r for r in completed if r["status"]!="ok"]
        if missing or errors:
            q.write_json(OUT/"status.json",dict(status="incomplete_or_failed",missing=missing,errors=errors))
            raise ValueError("read-only audit stopped; inspect saved error; no automatic retry")
        rows=[q.read_json(OUT/"episodes"/(j["job_id"]+".json")) for j in plan["jobs"]]
        if any(r["status"]!="ok" for r in rows): raise ValueError("audit error; no conclusions")
        result=dict(schema="lns2.sa_cooling_report.v1",complete=True,binding=plan["fingerprint"],
            cohorts=aggregate(rows),episodes=rows,no_solver=True,no_causal_claim=True,
            files={"episodes/"+j["job_id"]+".json":q.sha256_file(OUT/"episodes"/(j["job_id"]+".json")) for j in plan["jobs"]})
        q.write_json(OUT/"report.json",result)
        q.write_json(OUT/"status.json",dict(status="complete",episodes=len(rows)))
        return dict(complete=True,episodes=len(rows),no_solver=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","run"))
    parser.add_argument("--resume",action="store_true")
    args=parser.parse_args()
    print(json.dumps(prepare() if args.phase=="prepare" else run(args.resume)),flush=True)
