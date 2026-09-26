"""Read-only label coverage, stream grouping and fold accounting; never fit."""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import mean
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, json_fingerprint, read_json, sha256_file
from experiments.sa_paired_completion import require, validate_dataset
from experiments.sa_source_matched_analysis import analyze_record, partitions
from scripts import reconstruct_sa_recent_model as reconstruction
from scripts.collect_sa_history_candidate_bridge import randomization
from scripts.run_sa_paired_closed_loop import once

CONFIG = ROOT / "configs/sa_matched_training_coverage.json"


def prepare():
    cfg = read_json(CONFIG)
    require(cfg["solver_calls"] == cfg["model_fits"] == 0 and not cfg["automatic_training_allowed"] and
            not cfg["automatic_collection_allowed"] and not cfg["formal_ttf_allowed"], "audit boundary changed")
    base,_ = reconstruction.verify()
    out = ROOT / cfg["output"]
    require(not out.exists() and out.resolve().is_relative_to((ROOT/"build").resolve()), "existing or unsafe output")
    inputs = dict(base["inputs"])
    require(sha256_file(ROOT/cfg["reconstruction_evidence"]) == cfg["reconstruction_evidence_sha256"], "evidence drift")
    evidence = read_json(ROOT/cfg["reconstruction_evidence"])
    for name,digest in evidence["files"].items():
        require(sha256_file(ROOT/name) == digest and (name not in inputs or inputs[name] == digest), "evidence input drift")
        inputs[name] = digest
    sources = []
    paths = [CONFIG, Path(__file__), ROOT/cfg["reconstruction_evidence"],
             ROOT/"tests/evaluation/test_sa_matched_training_coverage.py",
             ROOT/"docs/SA_MATCHED_TRAINING_COVERAGE_PROTOCOL_ZH.md",
             ROOT/cfg["matched_collection"]/"plan.json"]
    for directory in cfg["old_collections"]:
        source = read_json(ROOT/directory/"plan.json")
        require(source["binding"] == json_fingerprint({k:v for k,v in source.items() if k != "binding"}), "old source binding")
        sources.append(dict(directory=directory, binding=source["binding"], config=source["config"],
                            ids=[r["id"] for r in source["roots"]]))
        paths.append(ROOT/directory/"plan.json")
        for sid in sources[-1]["ids"]:
            path=ROOT/directory/"roots"/sid/"root.json"
            require(path.relative_to(ROOT).as_posix() in inputs, "old root not previously pinned")
    for path in paths:
        name=path.relative_to(ROOT).as_posix(); digest=sha256_file(path)
        require(name not in inputs or inputs[name] == digest, "conflicting pin")
        inputs[name]=digest
    plan=dict(config=cfg, inputs=inputs, old_sources=sources, old_index=base["training_index"],
              records=base["label_records"],
              commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    plan["binding"]=json_fingerprint(plan)
    once(out/"plan.json",plan)
    verify()
    return dict(binding=plan["binding"], model_fits=0, solver_calls=0)


def verify():
    cfg=read_json(CONFIG); out=ROOT/cfg["output"]
    require(out.resolve().is_relative_to((ROOT/"build").resolve()),"unsafe output")
    plan=read_json(out/"plan.json")
    require(plan["config"]==cfg and plan["binding"]==json_fingerprint({k:v for k,v in plan.items() if k!="binding"}),"plan identity")
    for name,digest in plan["inputs"].items():
        require(sha256_file(contained_file(ROOT,name,field="coverage input"))==digest,"input drift: "+name)
    return plan,out


def comparison_group(state, source_binding, candidates, trial_keys, kind, family=None):
    require(len(trial_keys)==8 and len(set(trial_keys))==8,"trial-key coverage")
    ids=[c["candidate_id"] for c in candidates]
    require(len(ids)>=2 and len(set(ids))==len(ids),"duplicate or insufficient candidates")
    members=[tuple(sorted(c["agents"])) for c in candidates]
    require(len(set(members))==len(members) and all(m and len(set(m))==len(m) for m in members),"duplicate or invalid membership")
    require(all(len(c["values"])==8 and all(type(v) is bool for v in c["values"]) for c in candidates),"incomplete labels")
    if kind=="matched_pair":
        require(len(ids)==2 and family in ("target:16","collision:16","random:16") and
                all(len(c["agents"])==16 and c["selection_families"]==[family] for c in candidates),"matched family/size contract")
    else:
        require(kind=="old_grid","unknown group kind")
    return dict(group_id=json_fingerprint([source_binding,state["state_id"],kind]),
                state_id=state["state_id"],map_id=state["map_id"],source_binding=source_binding,kind=kind,
                family=family,trial_keys=trial_keys,candidates=sorted(candidates,key=lambda c:c["candidate_id"]))


def pair_rows(groups):
    require(len({g["group_id"] for g in groups})==len(groups),"duplicate comparison group")
    counts=Counter(g["state_id"] for g in groups)
    state_maps={}
    result=[]
    for group in groups:
        sid=group["state_id"]
        require(state_maps.setdefault(sid,group["map_id"])==group["map_id"],"root crosses maps")
        pairs=list(combinations(group["candidates"],2))
        for a,b in pairs:
            delta=[int(x)-int(y) for x,y in zip(a["values"],b["values"])]
            result.append(dict(group_id=group["group_id"],state_id=sid,map_id=group["map_id"],kind=group["kind"],
                left=a["candidate_id"],right=b["candidate_id"],target=mean(delta),
                a_wins=delta.count(1),b_wins=delta.count(-1),ties=delta.count(0),
                state_budget_weight=1/(counts[sid]*len(pairs))))
    require(all(abs(math.fsum(r["state_budget_weight"] for r in result if r["state_id"]==sid)-1)<1e-12
                for sid in counts),"duplicated root training weight")
    return result


def training_fold(rows, held):
    maps={r["map_id"] for r in rows}
    require(held in maps and len(maps)>1,"invalid held map")
    train=[r for r in rows if r["map_id"]!=held]
    counts={m:len({r["state_id"] for r in train if r["map_id"]==m}) for m in maps-{held}}
    n=sum(counts.values())
    return [r|dict(map_balanced_weight=r["state_budget_weight"]*n/(len(counts)*counts[r["map_id"]])) for r in train]


def neutral_crossfit(record):
    """When a training half ties, use uniform evaluation expectation, not IDs."""
    a,b=sorted(record["pair_ids"]); values=record["values"]; gains=[]
    for left,right in partitions():
        for train,test in ((left,right),(right,left)):
            train_delta=sum(int(values[a][t])-int(values[b][t]) for t in train)
            test_delta=mean(int(values[a][t])-int(values[b][t]) for t in test)
            gains.append(0. if train_delta==0 else ((train_delta>0)-(train_delta<0))*test_delta/2)
    return math.fsum(gains)/len(gains)


def summary(rows):
    maps=sorted({r["map_id"] for r in rows})
    mass=math.fsum(abs(r["all_trial_difference"]) for r in rows)
    nonzero=[r for r in rows if r["all_trial_difference"]!=0]
    discordant=sum(r["paired_counts"]["a_wins"]+r["paired_counts"]["b_wins"] for r in rows)
    by_map={m:[r for r in rows if r["map_id"]==m] for m in maps}
    return dict(states=len(rows),maps=len(maps),nonzero_states=len(nonzero),
                nonzero_maps=len({r["map_id"] for r in nonzero}),paired_trial_positions=8*len(rows),
                discordant_positions=discordant,zero_difference_states=len(rows)-len(nonzero),
                universally_same_half_direction=sum(r["half_direction_counts"]["strict_same_direction"]==35 for r in rows),
                original_positive_crossfit_states=sum(r["crossfit_gain"]>0 for r in rows),
                absolute_net_trial_mass=mass*8,
                max_state_net_mass_fraction=max((abs(r["all_trial_difference"])/mass for r in rows),default=0) if mass else 0,
                original_crossfit_mean=mean(mean(r["crossfit_gain"] for r in by_map[m]) for m in maps),
                tie_neutral_crossfit_mean=mean(mean(r["tie_neutral_crossfit"] for r in by_map[m]) for m in maps))


def coverage(records, roots, groups):
    require(len({r["state_id"] for r in records})==len(records)==len(roots),"coverage cohort")
    rows=[]
    for record in records:
        root=roots[record["state_id"]]
        require(record["map_id"]==root["map_id"] and record["pair_ids"]==root["pair_ids"],"root label identity")
        row=analyze_record(record)
        require(row["pair_complete"],"censoring cannot train")
        row={k:v for k,v in row.items() if k!="partitions"}
        row.update(family=root["family"],decision=root["decision"],tie_neutral_crossfit=neutral_crossfit(record))
        rows.append(row)
    rows.sort(key=lambda r:(r["map_id"],r["state_id"]))
    comparisons=pair_rows(groups)
    folds=[]
    for held in sorted({r["map_id"] for r in rows}):
        train=[r for r in rows if r["map_id"]!=held]
        matrix=training_fold(comparisons,held)
        folds.append(dict(held=held,matched_train=summary(train),
                          train_state_ids=sorted({r["state_id"] for r in matrix}),
                          train_group_ids=sorted({r["group_id"] for r in matrix}),
                          unordered_comparisons=len(matrix),
                          weight_sum=math.fsum(r["map_balanced_weight"] for r in matrix)))
    return dict(rows=rows,summary=summary(rows),by_family={f:summary([r for r in rows if r["family"]==f])
                for f in sorted({r["family"] for r in rows})},folds=folds,
                actual_training_performed=False,solver_calls=0,independent_confirmation=False,
                numerical_fit_is_possible=all(f["matched_train"]["nonzero_states"]>0 for f in folds),
                automatic_promotion=False,thresholds_applied=False)


def stream_hashes(root, cfg):
    return [json_fingerprint([randomization(root["state_id"],t,root["decision"]+d,cfg)
                             for d in range(32)]) for t in range(8)]


def audit():
    plan,out=verify()
    data=validate_dataset(read_json(ROOT/plan["old_index"]))
    old_by_id={s["state_id"]:s for s in data["states"]}
    source_by_id={sid:s for s in plan["old_sources"] for sid in s["ids"]}
    require(set(source_by_id)==set(old_by_id) and sum(len(s["ids"]) for s in plan["old_sources"])==len(old_by_id),"old source coverage")
    collection=read_json(ROOT/plan["config"]["matched_collection"]/"plan.json")
    require(collection["binding"]==json_fingerprint({k:v for k,v in collection.items() if k!="binding"}),"matched collection binding")
    roots={r["state_id"]:r for r in collection["roots"]}
    records=read_json(ROOT/plan["records"]); labels={r["state_id"]:r for r in records}
    require(set(labels)==set(roots) and len(labels)==len(records),"matched record coverage")
    groups=[]; stream_rows=[]
    for sid,s in sorted(old_by_id.items()):
        source=source_by_id[sid]
        root=read_json(ROOT/source["directory"]/"roots"/sid/"root.json")
        require(root["state_fingerprint"]==s["state_fingerprint"] and root["source"]["map_id"]==s["map_id"],"old root identity")
        members={c["candidate_id"]:c for c in root["candidates"]}
        keys=[json_fingerprint([source["binding"],sid,t]) for t in range(8)]
        candidates=[]
        for c in s["candidates"]:
            trials=sorted(c["trials"],key=lambda r:r["trial"])
            require([r["randomization_key"] for r in trials]==keys and c["agents"]==members[c["candidate_id"]]["agents"],"old trial stream or members")
            candidates.append(dict(candidate_id=c["candidate_id"],agents=c["agents"],values=[r["completed"] for r in trials]))
        groups.append(comparison_group(s,source["binding"],candidates,keys,"old_grid"))
        if sid not in roots:
            continue
        entry=roots[sid]
        require((entry["root_fingerprint"],entry["map_id"],entry["episode"],entry["decision"])==
                (s["state_fingerprint"],s["map_id"],s["episode"],s["decision"]),"matched/old root differs")
        newkeys=[json_fingerprint([collection["binding"],sid,t]) for t in range(8)]
        selected=[c|dict(values=labels[sid]["values"][c["candidate_id"]]) for c in entry["candidates"] if c["candidate_id"] in entry["pair_ids"]]
        groups.append(comparison_group(s,collection["binding"],selected,newkeys,"matched_pair",entry["family"]))
        old_stream=stream_hashes(entry,source["config"]); new_stream=stream_hashes(entry,collection["config"])
        require(not set(old_stream)&set(new_stream),"unexpected old/new stream overlap; review grouping")
        stream_rows.append(dict(state_id=sid,old_seed=source["config"]["seed"],new_seed=collection["config"]["seed"],
                                old_trial_keys=keys,new_trial_keys=newkeys,old_stream_hashes=old_stream,new_stream_hashes=new_stream,
                                same_trial_number_is_not_shared_randomization=True))
    report=coverage(records,roots,groups)
    comparisons=pair_rows(groups)
    index=dict(schema="lns2.sa.comparison_groups.v1",binding=plan["binding"],groups=groups,pairs=comparisons,
               old_feature_index=plan["old_index"],new_feature_directory="build/sa-source-matched-frozen-prediction-v1/features",
               training_authorized=False,anchor_in_matched_training=False,
               weight_contract="map_balanced_root_budget_then_equal_groups_then_equal_unordered_pairs")
    once(out/"comparison_groups.json",index)
    once(out/"stream_audit.json",dict(binding=plan["binding"],rows=stream_rows,raw_rollouts_replayed=False))
    outcomes=lambda kind:sum(len(c["values"]) for g in groups if g["kind"]==kind for c in g["candidates"])
    total_new=sum(len(v) for r in records for v in r["values"].values())
    require((len(old_by_id),len(records),len(groups),len(comparisons),outcomes("old_grid"),outcomes("matched_pair"),total_new)==
            (47,16,63,298,1504,256,376),"registered cohort changed")
    report.update(binding=plan["binding"],role=plan["config"]["role"],unique_roots=len(old_by_id),groups=len(groups),
                  old_groups=sum(g["kind"]=="old_grid" for g in groups),matched_groups=len(records),
                  unordered_comparisons=len(comparisons),old_candidate_outcomes=outcomes("old_grid"),matched_pair_outcomes=outcomes("matched_pair"),
                  matched_anchor_outcomes_excluded=total_new-outcomes("matched_pair"),
                  group_index_sha256=sha256_file(out/"comparison_groups.json"),
                  stream_audit_sha256=sha256_file(out/"stream_audit.json"))
    report["integrity"]=json_fingerprint(report)
    once(out/"report.json",report)
    verify()
    return {k:v for k,v in report.items() if k!="rows"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","audit","verify"))
    args=parser.parse_args()
    if args.phase=="verify":
        plan,out=verify(); result=dict(binding=plan["binding"],inputs=len(plan["inputs"]))
        if (out/"report.json").exists():
            report=read_json(out/"report.json")
            require(report["binding"]==plan["binding"] and report["integrity"]==
                    json_fingerprint({k:v for k,v in report.items() if k!="integrity"}),"report binding/integrity")
            for name,key in (("comparison_groups.json","group_index_sha256"),("stream_audit.json","stream_audit_sha256")):
                require(sha256_file(out/name)==report[key],"output drift")
    else:
        result=globals()[args.phase]()
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
