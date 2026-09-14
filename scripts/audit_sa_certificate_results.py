"""Validate full saved trajectories; preserve legacy flags instead of overwriting them."""
import argparse
from collections import Counter
from pathlib import Path
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
    native=augmentation.read_stage(a,"native")
    reference=augmentation.read_stage(a,"reference")
    revised=[]
    for row in native:
        case=next(x for x in a["cases"] if x["id"]==row["job"]["case_id"])
        after=row["result"]["final_state"]
        ref.validate_state(after)
        members=row["job"]["candidate"]["agents"]
        validate_transition(case["state"],after,row["result"]["metrics"],members,"annealed",case["temperature"],
                            row["result"]["metrics"]["acceptance_uniform"])
        cleared=pair_absent(after["conflict_edges"],case["pair"])
        revised.append(dict(candidate=row["job"]["candidate"]["id"],trial=row["job"]["trial"],
            before=row["result"]["before"],after=after["num_of_colliding_pairs"],target_pair_cleared=cleared,
            legacy_flag_wrong=cleared!=row["result"]["target_pair_cleared"],
            generated=row["result"]["generated"],censored=row["result"]["censored"]))
    m=io.read(continuation.OUT/"manifest.json")
    io.require(m["binding"]==c["binding"],"continuation manifest mismatch")
    expected={f"branches/{entry['target']['job_id']}/{arm}-{trial}.json" for entry in c["cases"] for arm in c["arms"] for trial in range(c["trials"])}
    io.require(set(m["files"])==expected,"continuation schedule incomplete")
    branches=[]
    for entry in c["cases"]:
        target=entry["target"]
        root=io.read(continuation.OUT/"roots"/(target["job_id"]+".json"))
        io.require(state_fingerprint(root)==target["state_fingerprint"],"root changed")
        pair=next(x["pair"] for x in a["cases"] if x["id"]==target["job_id"])
        for arm in c["arms"]:
            for trial in range(c["trials"]):
                name=f"branches/{target['job_id']}/{arm}-{trial}.json"
                path=continuation.OUT/name
                io.require(io.sha256_file(path)==m["files"][name],"branch changed")
                row=io.read(path)
                io.require(row["integrity"]==io.semantic_fingerprint({k:v for k,v in row.items() if k!="integrity"}),"branch integrity")
                io.require(row["binding"]==c["binding"] and row["arm"]==arm and row["trial"]==trial,"branch identity")
                state=root
                counts=[root["num_of_colliding_pairs"]]
                recurrence=[]
                case=dict(case_id=f"{target['item']['task_id']}-seed{target['item']['solver_seed']}")
                for offset,event in enumerate(row["events"]):
                    action,u=continuation.paired_action(case,target,trial,offset,event["pool"],event["selected_index"],arm,entry["added"])
                    io.require(event["action"]==action and event["uniform"]==u and event["temperature"]==temperature(target["decision"]+offset),"action protocol mismatch")
                    after=apply_state_delta(state,event["delta"])
                    validate_transition(state,after,event["metrics"],action["agents"],"annealed",event["temperature"],u)
                    state=after
                    counts.append(state["num_of_colliding_pairs"])
                    recurrence.append(not pair_absent(state["conflict_edges"],pair))
                ref.validate_state(state)
                io.require(state_fingerprint(state)==row["final_fingerprint"] and counts[-1]==row["final_conflicts"],"final mismatch")
                io.require(state["low_level"]["generated"]-root["low_level"]["generated"]==row["generated"],"node count mismatch")
                branches.append(dict(arm=arm,trial=trial,counts=counts,final=counts[-1],feasible=state["feasible"],
                    full_window=row["full_window"],generated=row["generated"],stop=row["stop"],
                    target_present_at_end=recurrence[-1] if recurrence else True,
                    target_returned_after_first_clear=any(recurrence[1:]) if recurrence and not recurrence[0] else None))
    sources={}
    for directory in (pair_runner.OUT,augmentation.OUT,continuation.OUT):
        for file in sorted(directory.rglob("*.json")):
            if "registered_sources" not in file.parts:
                sources[file.relative_to(ROOT).as_posix()]=io.sha256_file(file)
    result=dict(schema="lns2.sa_certificate_results.v1",sources=sources,source_snapshots_used=used,
        pair_jobs=len(bfs)+len(astar),pair_statuses=dict(Counter(r["result"]["status"] for r in {**bfs,**astar}.values())),
        necessary_obstructions=[r["job"] for r in {**bfs,**astar}.values() if r["result"]["proves_selected_insufficient"]],
        reference_jobs=len(reference),native_rows=revised,continuations=branches,
        corrected_legacy_flag_rows=sum(r["legacy_flag_wrong"] for r in revised),no_ttf=True,no_training=True,
        controller_promoted=False,decision="proved_membership_obstruction_and_local_recovery_not_full_escape")
    io.write(OUT/"report.json",result)
    print(dict(pair_jobs=result["pair_jobs"],necessary=len(result["necessary_obstructions"]),
        native_jobs=len(revised),corrected_flags=result["corrected_legacy_flag_rows"],branches=len(branches),
        completed=sum(b["feasible"] for b in branches),full_windows=sum(b["full_window"] for b in branches)))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","audit"))
    args=p.parse_args()
    freeze() if args.stage=="freeze" else audit()
