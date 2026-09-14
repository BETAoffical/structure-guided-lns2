"""Validate full saved trajectories; preserve legacy flags instead of overwriting them."""
import argparse
from collections import Counter
from pathlib import Path
import random
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import diagnose_pbs_repair as io
from scripts import diagnose_sa_pair_compatibility as pair_runner
from scripts import probe_sa_certificate_augmentation as augmentation
from scripts import probe_sa_certificate_continuation as continuation
from experiments import local_path_search as ref
from experiments.diagnostic_integrity import pair_absent, snapshot_sources, verify_inputs
from experiments.repair_collection import state_fingerprint
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.nonmonotonic_repair import validate_transition,temperature

OUT=ROOT/"build/sa-certificate-results-v1"
GOAL=ROOT/"build/sa-certificate-augmentation-goal-v1"


def checked_feasible(state):
    feasible=state["num_of_colliding_pairs"]==0
    io.require(type(state["feasible"]) is bool and state["feasible"]==feasible,"feasible flag disagrees with conflicts")
    return feasible


def recurrence_after_clear(presence):
    cleared=next((i for i,present in enumerate(presence) if not present),None)
    return None if cleared is None else any(presence[cleared+1:])


def repetition_metrics(fingerprints):
    seen=set()
    revisits=0
    for value in fingerprints:
        revisits+=value in seen
        seen.add(value)
    return dict(unique_physical_states=len(seen),revisited_states=revisits,
                unchanged_steps=sum(a==b for a,b in zip(fingerprints,fingerprints[1:])))


def checked_window(row,plan,counts):
    n=len(row["events"])
    io.require(len(counts)==n+1 and n<=plan["horizon"] and all(c>0 for c in counts[:-1]),"invalid continuation length or post-feasible action")
    feasible=counts[-1]==0
    stop=row["stop"]
    io.require(stop in ("horizon","feasible","wall_safety","node_budget","pp_safety"),"unknown continuation stop")
    io.require(row["feasible"]==feasible and (stop=="feasible")==feasible,"stop feasibility mismatch")
    timeouts=[i for i,e in enumerate(row["events"]) if e["metrics"]["pp_failure_reason"]=="time_limit"]
    timed_out=bool(timeouts)
    if stop=="horizon":
        io.require(n==plan["horizon"] and not timed_out,"incomplete horizon")
    if stop=="pp_safety":
        io.require(bool(row["events"]) and row["events"][-1]["metrics"]["pp_failure_reason"]=="time_limit","missing PP censor")
    if stop=="node_budget":
        io.require(row["generated"]>=plan["node_budget"],"node budget not reached")
    if stop=="wall_safety":
        io.require(row["diagnostic_seconds"]>=plan["seconds"],"wall budget not reached")
    io.require(not timed_out or (stop=="pp_safety" and timeouts==[n-1]),"continued after censored repair")
    full_window=stop in ("horizon","feasible")
    io.require(row["full_window"]==full_window,"full-window flag mismatch")
    return full_window


def diagnosis_decision(obstructions, rows, branches):
    if not obstructions:
        return "no_membership_obstruction_proved"
    if any(b["feasible"] and b["arm"]!="frozen" for b in branches):
        return "bounded_development_completion_not_controller_promotion"
    if any(r["kind"].startswith("certificate") and r["after"]<r["before"] for r in rows):
        return "proved_membership_obstruction_and_local_recovery_not_full_escape"
    return "proved_membership_obstruction_without_observed_native_recovery"


def audit_augmentation(directory,plan):
    previous=augmentation.OUT
    try:
        augmentation.OUT=directory
        native=augmentation.read_stage(plan,"native")
        reference=augmentation.read_stage(plan,"reference")
    finally:
        augmentation.OUT=previous
    revised=[]
    for row in native:
        case=next(x for x in plan["cases"] if x["id"]==row["job"]["case_id"])
        after=row["result"]["final_state"]
        ref.validate_state(after)
        checked_feasible(after)
        metrics=row["result"]["metrics"]
        candidate=row["job"]["candidate"]
        seed=int(io.semantic_fingerprint([case["id"],20260914,row["job"]["trial"]])[:7],16)
        io.require(metrics["requested_random_seed"]==seed,"trial seed mismatch")
        validate_transition(case["state"],after,metrics,candidate["agents"],"annealed",case["temperature"],
                            random.Random(seed).random())
        before=case["state"]["num_of_colliding_pairs"]
        io.require(row["result"]["before"]==before and row["result"]["after"]==after["num_of_colliding_pairs"],"native summary mismatch")
        io.require(row["result"]["strict_drop"]==(after["num_of_colliding_pairs"]<before),"drop mismatch")
        io.require(row["result"]["feasible"]==after["feasible"],"feasibility mismatch")
        io.require(row["result"]["generated"]==after["low_level"]["generated"],"reset-path node delta mismatch")
        io.require(row["result"]["censored"]==(metrics["pp_failure_reason"]=="time_limit"),"censor mismatch")
        cleared=pair_absent(after["conflict_edges"],case["pair"])
        revised.append(dict(candidate=candidate["id"],kind=candidate["kind"],case_id=case["id"],trial=row["job"]["trial"],
            before=before,after=after["num_of_colliding_pairs"],target_pair_cleared=cleared,
            legacy_flag_wrong=cleared!=row["result"]["target_pair_cleared"],
            generated=row["result"]["generated"],censored=row["result"]["censored"]))
    return dict(rows=revised,reference=[dict(candidate=r["job"]["candidate"]["id"],
                status=r["result"]["status"],reason=r["result"].get("reason")) for r in reference])


def freeze():
    for directory in (pair_runner.OUT,augmentation.OUT,continuation.OUT):
        plan=io.read(directory/"plan.json")
        snapshot_sources(ROOT,plan["inputs"],directory/"registered_sources")
    print("registered source snapshots verified")


def audit():
    p=io.read(pair_runner.OUT/"plan.json")
    a=io.read(augmentation.OUT/"plan.json")
    c=io.read(continuation.OUT/"plan.json")
    used={}
    for directory,plan in ((pair_runner.OUT,p),(augmentation.OUT,a),(continuation.OUT,c)):
        io.require(plan["binding"]==io.semantic_fingerprint({k:v for k,v in plan.items() if k!="binding"}),"plan mismatch")
        used[directory.name]=verify_inputs(ROOT,plan["inputs"],directory/"registered_sources")
    bfs,astar=pair_runner.read_stage(p,"bfs"),pair_runner.read_stage(p,"astar")
    early=audit_augmentation(augmentation.OUT,a)
    revised=early["rows"]
    goal_plan=io.read(GOAL/"plan.json")
    io.require(goal_plan["binding"]==io.semantic_fingerprint({k:v for k,v in goal_plan.items() if k!="binding"}),"goal plan mismatch")
    used[GOAL.name]=verify_inputs(ROOT,goal_plan["inputs"],GOAL/"registered_sources")
    goal=audit_augmentation(GOAL,goal_plan)
    io.require(not any(r["legacy_flag_wrong"] for r in goal["rows"]),"new flag regression")
    m=io.read(continuation.OUT/"manifest.json")
    io.require(m["binding"]==c["binding"],"continuation manifest mismatch")
    expected={f"branches/{entry['target']['job_id']}/{arm}-{trial}.json" for entry in c["cases"] for arm in c["arms"] for trial in range(c["trials"])}
    io.require(set(m["files"])==expected,"continuation schedule incomplete")
    branches=[]
    for entry in c["cases"]:
        target=entry["target"]
        root=io.read(continuation.OUT/"roots"/(target["job_id"]+".json"))
        io.require(state_fingerprint(root)==target["state_fingerprint"],"root changed")
        ref.validate_state(root)
        checked_feasible(root)
        pair=next(x["pair"] for x in a["cases"] if x["id"]==target["job_id"])
        for arm in c["arms"]:
            for trial in range(c["trials"]):
                name=f"branches/{target['job_id']}/{arm}-{trial}.json"
                path=continuation.OUT/name
                io.require(io.sha256_file(path)==m["files"][name],"branch changed")
                row=io.read(path)
                io.require(row["integrity"]==io.semantic_fingerprint({k:v for k,v in row.items() if k!="integrity"}),"branch integrity")
                io.require(row["binding"]==c["binding"] and row["arm"]==arm and row["trial"]==trial,"branch identity")
                io.require(row["status"]=="ok" and row["no_ttf"] is True and row["root_fingerprint"]==state_fingerprint(root),"invalid branch root/status")
                state=root
                counts=[root["num_of_colliding_pairs"]]
                physical=[io.repair_structure_fingerprint(root)]
                recurrence=[]
                case=dict(case_id=f"{target['item']['task_id']}-seed{target['item']['solver_seed']}")
                for offset,event in enumerate(row["events"]):
                    action,u=continuation.paired_action(case,target,trial,offset,event["pool"],event["selected_index"],arm,entry["added"])
                    io.require(event["action"]==action and event["uniform"]==u and event["temperature"]==temperature(target["decision"]+offset),"action protocol mismatch")
                    after=apply_state_delta(state,event["delta"])
                    validate_transition(state,after,event["metrics"],action["agents"],"annealed",event["temperature"],u)
                    checked_feasible(after)
                    state=after
                    counts.append(state["num_of_colliding_pairs"])
                    physical.append(io.repair_structure_fingerprint(state))
                    recurrence.append(not pair_absent(state["conflict_edges"],pair))
                ref.validate_state(state)
                io.require(state_fingerprint(state)==row["final_fingerprint"] and counts[-1]==row["final_conflicts"],"final mismatch")
                io.require(state["low_level"]["generated"]-root["low_level"]["generated"]==row["generated"],"node count mismatch")
                full_window=checked_window(row,c,counts)
                branches.append(dict(arm=arm,trial=trial,counts=counts,final=counts[-1],feasible=state["feasible"],
                    full_window=full_window,generated=row["generated"],stop=row["stop"],
                    final_conflict_edges=state["conflict_edges"],repetition=repetition_metrics(physical),
                    target_present_at_end=recurrence[-1] if recurrence else True,
                    target_returned_after_first_clear=recurrence_after_clear(recurrence)))
    sources={}
    for directory in (pair_runner.OUT,augmentation.OUT,continuation.OUT,GOAL,
                      ROOT/"build/sa-external-obstruction-v1",ROOT/"build/sa-goal-obstruction-v1"):
        for file in sorted(directory.rglob("*.json")):
            if "registered_sources" not in file.parts:
                sources[file.relative_to(ROOT).as_posix()]=io.sha256_file(file)
    result=dict(schema="lns2.sa_certificate_results.v1",sources=sources,source_snapshots_used=used,
        pair_jobs=len(bfs)+len(astar),pair_statuses=dict(Counter(r["result"]["status"] for r in {**bfs,**astar}.values())),
        necessary_obstructions=[r["job"] for r in {**bfs,**astar}.values() if r["result"]["proves_selected_insufficient"]],
        reference_jobs=len(early["reference"])+len(goal["reference"]),native_rows=revised,continuations=branches,
        early_reference=early["reference"],goal_augmentation=goal,
        corrected_legacy_flag_rows=sum(r["legacy_flag_wrong"] for r in revised),no_ttf=True,no_training=True,
        controller_promoted=False,auditor_sha=io.sha256_file(Path(__file__)))
    result["decision"]=diagnosis_decision(result["necessary_obstructions"],revised+goal["rows"],branches)
    io.write(OUT/"report.json",result)
    print(dict(pair_jobs=result["pair_jobs"],necessary=len(result["necessary_obstructions"]),
        native_jobs=len(revised)+len(goal["rows"]),corrected_flags=result["corrected_legacy_flag_rows"],branches=len(branches),
        completed=sum(b["feasible"] for b in branches),full_windows=sum(b["full_window"] for b in branches)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","audit"))
    args=p.parse_args()
    freeze() if args.stage=="freeze" else audit()
