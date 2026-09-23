"""One fixed Train-only difficulty transition, with unchanged terminal actor/PP/SA."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import collect_sa_second_batch as batch
from experiments.sa_terminal_transition import retained_ids, subset_document, signal_decision
from experiments._common import atomic_write_text
from lns2_selector.evaluation.path_quality_preflight import audit_task, read_grid, scenario_prefix

run,compare,recovery,runtime = batch.run,batch.compare,batch.recovery,batch.runtime
CONFIG = "configs/sa_terminal_transition.json"
REGISTRATION = "curriculum_registration.json"
ARM = "uncapped_condition"
CODE = (CONFIG,"scripts/probe_sa_terminal_transition.py","experiments/sa_terminal_transition.py",
        "tests/evaluation/test_sa_terminal_transition.py","docs/SA_TERMINAL_TRANSITION_PROTOCOL_ZH.md")


def parent_cases(old,source,cfg):
    cases = {c["case"]["task_id"]:c["case"] for c in old["conditions"] if c["case"]["task_variant"] == cfg["source_variant"]}
    run.require(len(cases) == 6 and {c["map_id"] for c in cases.values()} == set(source["split"]["train_maps"]),
                "all six original Train maps only")
    run.require(all(c["task_id"] == c["map_id"]+"__task_0001" for c in cases.values()), "original high-load tasks")
    return [cases[k] for k in sorted(cases)]


def context():
    old,source,src = batch.verify()
    cfg = run.read_json(ROOT/CONFIG)
    fixed = dict(subset_seed=2026092301,source_variant="bottleneck_d25",retained_ratio=.9,solver_seeds=[233,239],
        replicas=4,conditions=12,expected_jobs=48,max_decisions=None,decision_feature_reference=256,node_budget=25000000,
        pp_safety_seconds=20.,episode_safety_seconds=900.,process_fuse_seconds=960.,qualification_fuse_seconds=180.,
        workers=20,training=False,formal_ttf=False,automatic_promotion=False)
    run.require(all(cfg[k] == v for k,v in fixed.items()) and src == ROOT/cfg["source"], "fixed transition scope")
    run.require(run.sha256_file(src/"report.json") == cfg["source_report_sha256"] and
                run.sha256_file(ROOT/cfg["reference_report"]) == cfg["reference_report_sha256"], "source evidence changed")
    run.check_seal(run.read_json(ROOT/cfg["reference_report"]))
    complete = run.check_seal(run.read_json(src/"collection.complete.json"))
    aud = run.check_seal(run.read_json(src/"audit.json"))
    run.require(complete["binding"] == aud["binding"] == old["binding"] and complete["jobs"] == 192 and
                len(aud["results"]) == 192 and all(r["status"] == "ok" for r in aud["results"]) and
                {r["job_id"]:r["result_sha256"] for r in aud["results"]} == complete["files"], "source collection audit")
    cfg = dict(cfg,arms=[ARM],models={ARM:old["config"]["models"][ARM]})
    bundle = compare.prior.check_model(cfg,ARM,source["binding"])
    run.require(source["template"]["environment"]["max_repair_iterations"] == 0 and
                source["proposal"]["pp_safety_seconds"] == cfg["pp_safety_seconds"], "unchanged native/PP contract")
    return old,source,src,cfg,bundle,parent_cases(old,source,cfg)


def derive(parent,cfg):
    task_path = ROOT/parent["files"]["task_file"]
    task = run.read_json(task_path)
    run.require(task["map_id"] == parent["map_id"] and task["task_id"] == parent["task_id"], "task identity")
    ids,strata = retained_ids(task,cfg["subset_seed"])
    child = subset_document(task,ids,strata,seed=cfg["subset_seed"],free_cells=parent["static_audit"]["free_cells"],
                            source_sha256=run.sha256_file(task_path))
    map_path = ROOT/parent["files"]["map_file"]
    grid = read_grid(map_path)
    rows = scenario_prefix(map_path,ROOT/parent["files"]["scenario_file"],grid,len(task["starts"]))
    for i,row in enumerate(rows):
        run.require([int(row[5]),int(row[4])] == task["starts"][i] and
                    [int(row[7]),int(row[6])] == task["goals"][i], "parent scenario/task disagreement")
    scen = "version 1\n"+"".join("\t".join(rows[i])+"\n" for i in ids)
    return child,scen


def condition_schedule(cases,cfg):
    return [dict(case=c,split="train",solver_seed=s,pair_id=f"{c['task_id']}-s{s}")
            for c in cases for s in cfg["solver_seeds"]]


def jobs_for(conditions,cfg):
    return [dict(c,phase=cfg["phase"],comparison_arm=ARM,arm="trained_actor",iteration=1,replica=r,
                 job_id=run.json_fingerprint([cfg["phase"],c["pair_id"],r,ARM])[:24])
            for c in conditions for r in range(cfg["replicas"])]


def prepare():
    old,source,src,cfg,bundle,parents = context()
    out = ROOT/cfg["output"]
    run.require(not out.exists(), "existing output or interrupted preparation; inspect, do not overwrite")
    inputs = {p:run.sha256_file(ROOT/p) for p in CODE}
    for p in (src/"registration.json",src/"qualification.json",src/"collection.complete.json",src/"audit.json",src/"report.json",
              ROOT/cfg["reference_report"],ROOT/cfg["models"][ARM]["path"]):
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    for c in parents:
        for p in c["files"].values(): inputs[p] = run.sha256_file(ROOT/p)
    with recovery.strict_lock(out,run.json_fingerprint(cfg),"transition-prepare"):
        cases,generated = [],{}
        for parent in parents:
            task,scen = derive(parent,cfg)
            task_path = out/"dataset"/(task["task_id"]+".json")
            scen_path = task_path.with_suffix(".scen")
            run.once(task_path,task)
            run.require(not scen_path.exists(), "scenario already exists")
            atomic_write_text(scen_path,scen)
            audit = audit_task(ROOT/parent["files"]["map_file"],scen_path,task_path,len(task["starts"]))
            run.require(audit["scenario_distance_difference_count"] == 0, "subset shortest distances changed")
            c = dict(parent,task_id=task["task_id"],task_variant="bottleneck_d25_retain90",solver_seeds=cfg["solver_seeds"],
                density=task["metadata"]["agent_density_free_cells"],static_audit=audit,
                files=dict(parent["files"],task_file=task_path.relative_to(ROOT).as_posix(),
                           scenario_file=scen_path.relative_to(ROOT).as_posix()))
            cases.append(c)
            for p in (task_path,scen_path): generated[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
        conditions = condition_schedule(cases,cfg)
        batch.require_unused(conditions,out)
        body = dict(schema=cfg["schema"],config=cfg,inputs=inputs,generated=generated,parents=parents,cases=cases,
            conditions=conditions,jobs=jobs_for(conditions,cfg),source_binding=old["binding"],scientific_binding=source["binding"],
            source_commit=run.subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            role="train_endpoint_preserving_subset_not_new_maps_or_algorithm_comparison",no_training=True,no_ttf=True)
        body["binding"] = run.json_fingerprint(body)
        p = compare.runtime_plan(source,cfg,ARM)
        target = run.actor_file(ROOT/p["config"]["output"],1)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/cfg["models"][ARM]["path"],target)
        run.require(run.sha256_file(target) == cfg["models"][ARM]["file_sha256"], "frozen actor copy")
        run.once(target.parent/"receipt-1.json",run.sealed(dict(binding=source["binding"],policy_sha256=run.validate_bundle(bundle),
            file_sha256=run.sha256_file(target),metadata=dict(transition_binding=body["binding"],copied_not_trained=True))))
        run.once(out/REGISTRATION,run.sealed(body))
    return dict(binding=body["binding"],maps=6,qualification=12,episodes=48,max_decisions=None)


def check_inputs(reg):
    for p,digest in {**reg["inputs"],**reg["generated"]}.items():
        run.require(run.sha256_file(run.contained_file(ROOT,p,field="transition input")) == digest,"changed input: "+p)


def verify():
    old,source,_,cfg,bundle,parents = context()
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/REGISTRATION))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding","integrity")}), "registration hash")
    run.require(reg["config"] == cfg and reg["parents"] == parents and reg["source_binding"] == old["binding"] and
                reg["scientific_binding"] == source["binding"], "transition identity")
    check_inputs(reg)
    for c,parent in zip(reg["cases"],parents):
        task,scen = derive(parent,cfg)
        run.require(run.read_json(ROOT/c["files"]["task_file"]) == task and
                    (ROOT/c["files"]["scenario_file"]).read_text(encoding="utf8") == scen, "derived task changed")
    run.require(len(reg["cases"]) == 6 and reg["conditions"] == condition_schedule(reg["cases"],cfg) and
                reg["jobs"] == jobs_for(reg["conditions"],cfg), "complete schedule")
    batch.require_unused(reg["conditions"],out)
    p = compare.runtime_plan(source,cfg,ARM)
    run.require(run.actor_load(ROOT/p["config"]["output"],p,1) == bundle,"model copy changed")
    return reg,source,out,old


def qualification_jobs(reg,source):
    p = dict(source,config=dict(source["config"],output=reg["config"]["output"]+"/qualification"))
    return [dict(c,phase="qualify-retain90",job_id=run.json_fingerprint(["qualify-retain90",c["pair_id"]])[:24],
                 plan=p,parent_pid=os.getpid(),collection_binding=reg["binding"]) for c in reg["conditions"]]


def qualify(resume=False,verified=None):
    reg,source,out,_ = verified or verify()
    check_inputs(reg)
    jobs = qualification_jobs(reg,source)
    with recovery.strict_lock(out,reg["binding"],"transition-qualify"):
        done = batch.execute_batches(reg,out,jobs,batch.qualification_worker,batch.qfolder,
            lambda j:batch.read_qualification(reg,j),"qualification",resume,reg["config"]["qualification_fuse_seconds"])
        if not done: return dict(status="paused_or_needs_inspection")
        rows = [batch.read_qualification(reg,j) for j in jobs]
        run.once(out/"qualification.json",run.sealed(dict(binding=reg["binding"],rows=rows,valid=len(rows),
            nonzero=sum(not r["feasible"] for r in rows),unique_initial_paths=len({r["paths_signature"] for r in rows}),
            complete_sha256=run.sha256_file(out/"qualification.complete.json"),no_outcome_filter=True)))
    return dict(status="qualified",valid=len(rows),nonzero=sum(not r["feasible"] for r in rows))


def ready_jobs(reg,source,out):
    proof = run.check_seal(run.read_json(out/"qualification.json"))
    completion = run.check_seal(run.read_json(out/"qualification.complete.json"))
    qjobs = qualification_jobs(reg,source)
    rows = [batch.read_qualification(reg,j) for j in qjobs]
    run.require(proof["binding"] == completion["binding"] == reg["binding"] and proof["rows"] == rows and
                proof["valid"] == completion["jobs"] == len(rows) == 12 and
                proof["complete_sha256"] == run.sha256_file(out/"qualification.complete.json") and
                set(completion["files"]) == {j["job_id"] for j in qjobs}, "qualification coverage")
    for j in qjobs:
        run.require(completion["files"][j["job_id"]] == run.sha256_file(batch.qfolder(j)/"result.json"), "qualification changed")
    initials = {r["pair_id"]:r["initial_fingerprint"] for r in rows}
    return [dict(j,expected_initial=initials[j["pair_id"]]) for j in reg["jobs"]]


def collect(resume=False,verified=None):
    reg,source,out,_ = verified or verify()
    check_inputs(reg)
    jobs = ready_jobs(reg,source,out)
    with recovery.strict_lock(out,reg["binding"],"transition-collect"):
        done = batch.execute_batches(reg,out,[compare.augmented(reg,source,j) for j in jobs],compare.worker,
            lambda j:compare.prior.folder_for(reg["config"],j),lambda j:compare.read_result(reg,source,j),
            "collection",resume,reg["config"]["process_fuse_seconds"])
    return dict(status="collected" if done else "paused_or_needs_inspection")


def completed(reg,source,out):
    jobs = ready_jobs(reg,source,out)
    proof = run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(proof["binding"] == reg["binding"] and proof["jobs"] == 48 and
                set(proof["files"]) == {j["job_id"] for j in jobs}, "complete collection")
    for j in jobs:
        row = compare.read_result(reg,source,j)
        run.require(row["status"] == "ok" and proof["files"][j["job_id"]] ==
                    run.sha256_file(compare.prior.folder_for(reg["config"],j)/"result.json"), "unknown/changed terminal")
    return jobs,proof


def audit(verified=None):
    from experiments.repair_collection import _run_jobs
    reg,source,out,_ = verified or verify()
    check_inputs(reg)
    jobs,proof = completed(reg,source,out)
    with recovery.strict_lock(out,reg["binding"],"transition-audit"):
        rows = _run_jobs(runtime.audit_worker,[compare.augmented(reg,source,j) for j in jobs],20,
            phase="transition-audit",output_root=out/"audit-progress",run_fingerprint=reg["binding"],timeout_seconds=960.)
        run.require(len(rows) == 48 and all(r["status"] == "ok" for r in rows) and
                    {r["job_id"]:r["result_sha256"] for r in rows} == proof["files"], "full trace audit")
        run.once(out/"audit.json",run.sealed(dict(binding=reg["binding"],results=rows)))
    return dict(audited=48)


def report(verified=None):
    reg,source,out,old = verified or verify()
    check_inputs(reg)
    jobs,complete = completed(reg,source,out)
    aud = run.check_seal(run.read_json(out/"audit.json"))
    run.require(aud["binding"] == reg["binding"] and len(aud["results"]) == 48 and
                all(r["status"] == "ok" for r in aud["results"]) and
                {r["job_id"]:r["result_sha256"] for r in aud["results"]} == complete["files"], "matching audit")
    training = []
    for j in jobs:
        row = compare.read_result(reg,source,j)
        folder = compare.prior.folder_for(reg["config"],j)
        steps = [{k:e[k] for k in ("decision","policy_sha256","probabilities","selected_id","behavior_log_probability")}
                 for e in run.trace_read(folder)]
        training.append(dict(row,steps=steps))
    groups = {c["pair_id"]:c["case"]["map_id"] for c in reg["conditions"]}
    info = batch.credit_summary(training,reg["config"]["models"][ARM]["policy_sha256"],groups)
    hard = "sa_linear_v1_m04_station_centric_0000"
    hard_success = sum(r["success"] for r in training if r["map_id"] == hard)
    interpretation = signal_decision(info,hard_success)
    original = run.check_seal(run.read_json(ROOT/old["config"]["output"]/"report.json"))
    counts = []
    for c,parent in zip(reg["cases"],reg["parents"]):
        for seed in reg["config"]["solver_seeds"]:
            key = f"{c['task_id']}-s{seed}"
            previous = [r for r in original["episodes"] if r["comparison_arm"] == ARM and
                        r["pair_id"] == f"{parent['task_id']}-s{seed}"]
            run.require(len(previous) == 4, "source terminal denominator")
            current = [r for r in training if r["pair_id"] == key]
            counts.append(dict(pair_id=key,map_id=c["map_id"],solver_seed=seed,source_agents=parent["static_audit"]["agent_count"],
                retained_agents=c["static_audit"]["agent_count"],previous_success=sum(r["success"] for r in previous),
                current_success=sum(r["success"] for r in current),replicas=4,not_paired_algorithm_comparison=True))
    body = dict(schema="lns2.sa.terminal_transition_report.v1",binding=reg["binding"],interpretation=interpretation,
        condition_outcomes=counts,terminal_information=info,success=sum(r["success"] for r in training),episodes=48,
        decisions=sum(r["decisions"] for r in training),stops=dict(Counter(r["stop"] for r in training)),
        beyond_256=sum(r["decisions"]>256 for r in training),hard_subset_success=hard_success,
        outcomes=[{k:v for k,v in r.items() if k != "steps"} for r in training],
        audit_sha256=run.sha256_file(out/"audit.json"),qualification_sha256=run.sha256_file(out/"qualification.json"),
        no_training=True,no_ttf=True,no_heldout=True,automatic_promotion=False,
        caveat="same Train maps, stratified deletion intervention, different PP initial paths and RNG streams; no hard-task recovery claim")
    # All map/count keys are strings before sealing, including nested summaries.
    artifact = run.sealed(body)
    run.check_seal(json.loads(json.dumps(artifact)))
    with recovery.strict_lock(out,reg["binding"],"transition-report"):
        run.once(out/"report.json",artifact)
        run.write_json(out/"run_status.json",dict(status="completed",binding=reg["binding"],report_sha256=run.sha256_file(out/"report.json")))
    return dict(success=body["success"],episodes=48,decisions=body["decisions"],conditions=counts,
                information={k:v for k,v in info.items() if k != "coefficients"},interpretation=interpretation)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","verify","dry-run","qualify","collect","audit","report","all","stop"))
    p.add_argument("--resume",action="store_true")
    args = p.parse_args()
    if args.phase == "all":
        verified = verify()
        result = qualify(args.resume,verified)
        print(json.dumps(result),flush=True)
        if result["status"] == "qualified":
            result = collect(args.resume,verified)
            if result["status"] == "collected":
                print(json.dumps(audit(verified)),flush=True)
                result = report(verified)
    elif args.phase in ("qualify","collect"): result = globals()[args.phase](args.resume)
    elif args.phase in ("verify","dry-run"):
        reg,_,_,_ = verify()
        result = dict(binding=reg["binding"],qualification=12,episodes=48,workers=20,max_decisions=None,
            repair_node_budget_per_episode=25000000,episode_waves=3,safety_bound_seconds=3*960+180,no_training=True,no_ttf=True)
    elif args.phase == "stop":
        run.write_json(ROOT/run.read_json(ROOT/CONFIG)["output"]/"STOP_AFTER_BATCH",dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else: result = globals()[args.phase]()
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
