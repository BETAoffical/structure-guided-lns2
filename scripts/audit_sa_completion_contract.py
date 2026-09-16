"""Audit existing paired labels without solver calls or model training."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments._common import read_json, write_json, sha256_file, json_fingerprint
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.sa_paired_completion import require
from experiments.sa_completion_contract import signatures, pool_signature, split_diagnostics, compare_branches, aggregate
from scripts.run_sa_paired_completion_pilot import verify, verify_receipt
from scripts.collect_sa_history_candidate_bridge import randomization
from experiments.nonmonotonic_repair import temperature

OUT = ROOT / "build/sa-completion-label-contract-audit-v1"
FILES = ["experiments/sa_completion_contract.py", "scripts/audit_sa_completion_contract.py",
         "tests/evaluation/test_sa_completion_contract.py", "docs/SA_COMPLETION_CONTRACT_PROTOCOL_ZH.md"]


def audit_root(job):
    plan,target,index_state,source = job
    folder = Path(source)/"roots"/target["id"]
    verify_receipt(folder,target,plan)
    root = read_json(folder/"root.json")
    require(root["source"] == target, "root provenance")
    initial = signatures(root["state"])
    require(initial["full"] == target["state_fingerprint"], "root fingerprint")
    traces,values = {},{}
    branches=[]
    for c in index_state["candidates"]:
        cid=c["candidate_id"]
        values[cid]=[]
        for t in sorted(c["trials"],key=lambda r:r["trial"]):
            trial=t["trial"]
            row=read_json(folder/f"{cid}-t{trial}.json")
            require((row["status"],row["root_id"],row["candidate_id"],row["trial"]) == ("ok",target["id"],cid,trial), "branch identity")
            require(row["root_fingerprint"] == initial["full"], "branch root")
            require(row["stop"] in ("horizon","feasible") and t["completed"] == (row["stop"]=="feasible"), "label mismatch/censoring")
            require(len(row["events"]) == t["steps"] and row["final_conflicts"]==t["final_conflicts"], "outcome/index mismatch")
            state=root["state"]
            states=[]
            seen={initial["physical"]}
            recurrence=False
            for offset,e in enumerate(row["events"]):
                d=target["decision"]+offset
                pp,u=randomization(target["id"],trial,d,plan["config"])
                require((e["decision"],e["action"]["random_seed"],e["uniform"],e["temperature"]) == (d,pp,u,temperature(d)), "randomization contract")
                require(e["action"]["agents"]==e["pool"][e["selected_index"]]["agents"], "action changed")
                state=apply_state_delta(state,e["delta"])
                sig=signatures(state)
                recurrence |= sig["conflicts"]>0 and sig["physical"] in seen
                seen.add(sig["physical"])
                states.append(sig)
            require(states[-1]["full"]==row["final_fingerprint"], "final fingerprint")
            require(states[-1]["conflicts"] == row["final_conflicts"], "final conflict count")
            rec=dict(candidate_id=cid,trial=trial,completed=t["completed"], states=states,
                next_pool=pool_signature(row["events"][1]) if len(row["events"])>1 else None,
                unchanged_first_paths=states[0]["physical"]==initial["physical"],
                first_accepted=row["events"][0]["metrics"]["replan_success"],physical_recurrence=recurrence)
            traces[cid,trial]=rec
            branches.append({k:v for k,v in rec.items() if k not in ("states","next_pool")})
            values[cid].append(int(t["completed"]))
    pairs=[compare_branches(traces[a,t],traces[b,t]) for t in range(8) for a,b in combinations(sorted(values),2)]
    split=split_diagnostics(values,index_state["anchor_id"])
    verify_receipt(folder,target,plan)
    return dict(id=target["id"],map_id=target["map_id"],phase=target["phase"],split=split,pairs=pairs,branches=branches)


def run(workers=20):
    from scripts.run_sa_path_quality import _CollectionRunLock
    require(1<=workers<=20,"workers must be 1..20")
    plan,source=verify()
    paths={str((source/name).relative_to(ROOT)).replace("\\","/"):sha256_file(source/name)
           for name in ("training_index.json","label_report.json","model_report.json","training_receipt.json","posthoc_diagnostics.json")}
    inputs=paths | {name:sha256_file(ROOT/name) for name in FILES}
    for target in plan["roots"]:
        path=source/"roots"/target["id"]/"receipt.json"
        inputs[path.relative_to(ROOT).as_posix()]=sha256_file(path)
    registration=dict(schema="lns2.sa.completion_contract.v1",source_binding=plan["binding"],inputs=inputs,
        commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        no_solver=True,no_training=True,no_ttf=True,changes_no_gate=True,workers=workers)
    registration["binding"]=json_fingerprint(registration)
    with _CollectionRunLock(OUT,registration["binding"],"completion-label-contract-audit"):
        require(not (OUT/"registration.json").exists(),"audit already registered; do not overwrite")
        write_json(OUT/"registration.json",registration)
        write_json(OUT/"run_status.json",dict(status="running",binding=registration["binding"]))
        try:
            data=read_json(source/"training_index.json")
            label_report=read_json(source/"label_report.json")
            receipt=read_json(source/"training_receipt.json")
            require(receipt["binding"]==label_report["binding"]==plan["binding"],"source binding")
            require(receipt["input_sha256"]==sha256_file(source/"training_index.json") and
                    receipt["report_sha256"]==sha256_file(source/"model_report.json"),"training receipt changed")
            require(label_report["data_fingerprint"]==json_fingerprint(data),"index content changed")
            lookup={s["state_id"]:s for s in data["states"]}
            require(set(lookup)=={t["id"] for t in plan["roots"]},"index root set")
            jobs=[(plan,t,lookup[t["id"]],str(source)) for t in plan["roots"]]
            roots=[]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for row in pool.map(audit_root,jobs):
                    roots.append(row)
                    write_json(OUT/"roots"/(row["id"]+".json"),row)
                    print("AUDITED",len(roots),"/",len(jobs),row["id"],flush=True)
            summary=aggregate(roots)
            old=read_json(source/"posthoc_diagnostics.json")
            require(summary["partitions"][0]["informative_jaccard"]==old["informative_best_jaccard"] and
                    summary["partitions"][0]["cross_half_rate"]==old["cross_half_oracle_rate"],"old split did not reproduce")
            verify()
            for name,digest in inputs.items():
                require(sha256_file(ROOT/name)==digest,"input changed during audit: "+name)
            report=dict(binding=registration["binding"],summary=summary,no_new_labels=True,
                        no_model_fits=True,exploratory_only=True,promotion_allowed=False)
            write_json(OUT/"report.json",report)
            write_json(OUT/"run_status.json",dict(status="completed",binding=registration["binding"],report_sha256=sha256_file(OUT/"report.json")))
            return {k:v for k,v in summary.items() if k!="partitions"}
        except BaseException as exc:
            write_json(OUT/"run_status.json",dict(status="error",binding=registration["binding"],error=repr(exc)))
            raise


if __name__ == "__main__":
    import json
    for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
        os.environ[name]="1"
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers",type=int,default=20)
    args=parser.parse_args()
    print(json.dumps(run(args.workers),indent=2))
