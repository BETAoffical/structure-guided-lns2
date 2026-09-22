"""Register an analysis-only projection fix after collection has completed."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_sa_onpolicy as run
from scripts.recover_sa_onpolicy import strict_lock

RUNNER="scripts/compare_sa_crossfit.py"
SUPPORT=("scripts/register_sa_crossfit_report_fix.py","tests/evaluation/test_sa_crossfit_report_fix.py")
BACKUP="runner-before-report-fix.py"
RECEIPT="report-only-amendment.json"
OLD_VERIFY='''    for name, digest in reg["inputs"].items():
        require(run.sha256_file(run.contained_file(ROOT, name, field="comparison input")) == digest, "changed input: "+name)'''
NEW_VERIFY='''    changes = {name: run.sha256_file(run.contained_file(ROOT, name, field="comparison input"))
               for name, digest in reg["inputs"].items()
               if run.sha256_file(run.contained_file(ROOT, name, field="comparison input")) != digest}
    if changes:
        from scripts.register_sa_crossfit_report_fix import validate_amendment
        validate_amendment(ROOT, reg, out, changes)'''
OLD_PAIR='recovery.event_projection(event) == recovery.event_projection(new)'
NEW_PAIR='{k:v for k,v in event.items() if k != "metrics"} == {k:v for k,v in new.items() if k != "metrics"}'


def prove_scope(before,after):
    run.require(before.count(OLD_VERIFY)==before.count(OLD_PAIR)==1,"unexpected source")
    run.require(before.replace(OLD_VERIFY,NEW_VERIFY).replace(OLD_PAIR,NEW_PAIR)==after,"not report-only change")


def validate_amendment(root,reg,out,changes):
    receipt=run.check_seal(run.read_json(out/RECEIPT))
    run.require(receipt["binding"]==reg["binding"] and receipt["old_sha256"]==reg["inputs"][RUNNER],"amendment identity")
    run.require(changes=={RUNNER:receipt["new_sha256"]},"other registered input changed")
    run.require(run.sha256_file(out/BACKUP)==receipt["old_sha256"],"backup changed")
    run.require(receipt["support"]=={n:run.sha256_file(root/n) for n in SUPPORT},"support changed")
    prove_scope((out/BACKUP).read_text(encoding="utf8"),(root/RUNNER).read_text(encoding="utf8"))
    run.require(run.sha256_file(out/"collection.complete.json")==receipt["collection_sha256"],"collection receipt changed")


def register():
    cfg=run.read_json(ROOT/"configs/sa_crossfit_comparison.json")
    out=ROOT/cfg["output"]
    reg=run.check_seal(run.read_json(out/"registration.json"))
    with strict_lock(out,reg["binding"],"report-only-amendment"):
        run.require(run.read_json(out/"run_status.json")["status"]=="completed","collection must finish first")
        complete=run.check_seal(run.read_json(out/"collection.complete.json"))
        run.require(complete["binding"]==reg["binding"] and complete["jobs"]==80,"incomplete collection")
        run.require(run.sha256_file(out/BACKUP)==reg["inputs"][RUNNER],"original backup")
        prove_scope((out/BACKUP).read_text(encoding="utf8"),(ROOT/RUNNER).read_text(encoding="utf8"))
        for name,digest in reg["inputs"].items():
            if name != RUNNER:run.require(run.sha256_file(ROOT/name)==digest,"other input changed")
        for j in reg["jobs"]:
            path=out/"runs"/j["comparison_arm"]/j["phase"]/j["job_id"]/"result.json"
            run.require(run.sha256_file(path)==complete["files"][j["job_id"]],"result changed")
        run.once(out/RECEIPT,run.sealed(dict(binding=reg["binding"],old_sha256=reg["inputs"][RUNNER],
             new_sha256=run.sha256_file(ROOT/RUNNER),support={n:run.sha256_file(ROOT/n) for n in SUPPORT},
             collection_sha256=run.sha256_file(out/"collection.complete.json"),
             changed_episodes=False,changed_models=False,
             reason="Official trace lacks actor-only keys; compare all non-metrics fields, not an explicit-only projection.")))
    from scripts.compare_sa_crossfit import verify
    verify()
    return dict(registered=True,unchanged_episodes=80,changed_science=False)


def first_difference(left_trace,right_trace):
    common=0
    for a,b in zip(left_trace,right_trace):
        run.require(a["before"]==b["before"],"unexplained state divergence before action difference")
        if a["action"]!=b["action"]:
            run.require(a["pool"]==b["pool"] and a["features"]==b["features"] and
                        a["selection_draw"]==b["selection_draw"],"different pre-action conditions")
            first=dict(decision=a["decision"],selected=a["selected_id"],base_selected=b["selected_id"],
                       same_pool_state_and_draw=True,
                       total_variation=.5*sum(abs(a["probabilities"][c]-b["probabilities"][c]) for c in a["probabilities"]))
            return common,first
        run.require(a["delta"]==b["delta"],"same action produced different solver state")
        common+=1
    return common,None


def behavior_diagnostics():
    from scripts import compare_sa_crossfit as compare
    reg,source,out=compare.verify()
    report=run.check_seal(run.read_json(out/"report.json"))
    run.require(report["binding"]==reg["binding"],"report identity")
    jobs={(j["comparison_arm"],j["pair_id"],j["replica"]):j for j in reg["jobs"]}
    results={}
    for arm in ("bounded_condition","bounded_state"):
        pairs=[]
        for key,job in jobs.items():
            if key[0]!=arm:continue
            base=jobs[("untrained_exploration",key[1],key[2])]
            left=compare.read_result(reg,source,job)
            right=compare.read_result(reg,source,base)
            common,first=first_difference(run.trace_read(compare.folder_for(reg["config"],job)),
                                          run.trace_read(compare.folder_for(reg["config"],base)))
            if first is None:
                run.require(left["decisions"]==right["decisions"] and left["final_fingerprint"]==right["final_fingerprint"],"unexplained length divergence")
            pairs.append(dict(pair_id=key[1],replica=key[2],map_id=left["map_id"],common_prefix_decisions=common,
                              first_difference=first,success=left["success"],base_success=right["success"],
                              final_conflicts=left["final_conflicts"],base_final_conflicts=right["final_conflicts"],
                              stop=left["stop"],base_stop=right["stop"]))
        results[arm]=dict(pairs=pairs,changed_trajectories=sum(p["first_difference"] is not None for p in pairs))
    body=run.sealed(dict(binding=reg["binding"],report_sha256=run.sha256_file(out/"report.json"),
                        posthoc_descriptive_only=True,no_solver=True,no_training=True,results=results))
    with strict_lock(out,reg["binding"],"behavior-diagnostics"):
        run.once(out/"behavior_diagnostics.json",body)
    return {a:dict(changed_trajectories=v["changed_trajectories"],pairs=len(v["pairs"])) for a,v in results.items()}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("register","diagnose"),default="register",nargs="?")
    print(register() if p.parse_args().phase=="register" else behavior_diagnostics())
