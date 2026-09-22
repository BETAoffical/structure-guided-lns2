"""Train-only complete on-policy trajectories and a paired exploration control."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import recover_sa_uncapped_training as source_training
from scripts import compare_sa_uncapped as compare
from scripts import run_sa_onpolicy as run
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_uncapped_runtime as runtime
from experiments import sa_uncapped_training_contract as credit

CONFIG = "configs/sa_second_onpolicy_batch.json"
ARMS = ("untrained_exploration", "uncapped_condition")
CODE = (CONFIG, "scripts/collect_sa_second_batch.py", "tests/evaluation/test_sa_second_batch.py",
        "docs/SA_SECOND_ONPOLICY_BATCH_PROTOCOL_ZH.md")


def conditions(old, cfg):
    expected = dict(phase="train-1-new-conditions", solver_seeds=[233,239], replicas=4, conditions=24,
        expected_jobs=192, arms=list(ARMS), max_decisions=None, decision_feature_reference=256,
        node_budget=25000000, pp_safety_seconds=20., episode_safety_seconds=900., process_fuse_seconds=960.,
        qualification_fuse_seconds=180., workers=20, maximum_updates_this_stage=0,
        formal_ttf=False, heldout_evaluation=False, automatic_promotion=False)
    run.require(all(cfg[k] == v for k,v in expected.items()), "fixed collection scope")
    roots = {}
    for e in old["entries"].values():
        j = e["job"]
        run.require(j["split"] == "train", "Train only")
        roots[j["case"]["task_id"]] = j
    run.require(len(roots) == 12 and len({j["case"]["map_id"] for j in roots.values()}) == 6,
                "six original Train maps and twelve tasks")
    result = []
    for task, root in sorted(roots.items()):
        run.require(not set(cfg["solver_seeds"]) & set(root["case"]["solver_seeds"]), "old solver seed reused")
        for seed in cfg["solver_seeds"]:
            result.append(dict(case=dict(root["case"],solver_seeds=cfg["solver_seeds"]), split="train",
                solver_seed=seed, pair_id=f"{task}-s{seed}"))
    return result


def schedule(roots, cfg):
    jobs = [dict(c, phase=cfg["phase"], replica=r, comparison_arm=arm,
        arm="trained_actor" if arm == ARMS[1] else arm, iteration=cfg["models"][arm]["iteration"],
        job_id=run.json_fingerprint([cfg["phase"],c["pair_id"],r,arm])[:24])
        for c in roots for r in range(cfg["replicas"]) for arm in ARMS]
    run.require(len(jobs) == len({j["job_id"] for j in jobs}) == 192, "complete job schedule")
    return jobs


def pair_inventory(value):
    if isinstance(value, dict):
        if isinstance(value.get("pair_id"), str):
            yield value["pair_id"]
        for v in value.values():
            yield from pair_inventory(v)
    elif isinstance(value, list):
        for v in value:
            yield from pair_inventory(v)


def require_unused(roots, out):
    pairs = {c["pair_id"] for c in roots}
    for path in sorted((ROOT/"build").glob("sa-*/registration.json")):
        if path.parent != out:
            run.require(not pairs & set(pair_inventory(run.read_json(path))),
                        "previously registered condition: " + path.parent.name)


def context():
    old, source, src = source_training.verify()
    cfg = run.read_json(ROOT/CONFIG)
    run.require(src == ROOT/cfg["source_training"], "source training identity")
    roots = conditions(old,cfg)
    run.require({c["case"]["map_id"] for c in roots} == set(source["split"]["train_maps"]), "Train map identity")
    run.require(source["template"]["environment"]["max_repair_iterations"] == 0 and
                source["proposal"]["pp_safety_seconds"] == cfg["pp_safety_seconds"], "repair contract")
    bundles = {a:compare.prior.check_model(cfg,a,source["binding"]) for a in ARMS}
    update = run.check_seal(run.read_json(src/"update.json"))
    parity = run.check_seal(run.read_json(src/"parity.json"))
    analysis = run.check_seal(run.read_json(src/"terminal_analysis.json"))
    run.require(run.sha256_file(src/"update.json") == cfg["update_sha256"] and
                update["analysis_sha256"] == run.sha256_file(src/"terminal_analysis.json") and
                analysis["audit_sha256"] == run.sha256_file(src/"batch.audit.json"),"parent training evidence")
    run.require(update["binding"] == parity["binding"] == old["binding"] and update["updated"] and
                update["model_sha256"] == cfg["models"][ARMS[1]]["file_sha256"] and
                update["policy_sha256"] == parity["policy_sha256"] == run.validate_bundle(bundles[ARMS[1]]) and
                parity["update_sha256"] == run.sha256_file(src/"update.json"), "frozen parent update and parity")
    run.require(bundles[ARMS[1]]["parent_policy"] == run.validate_bundle(bundles[ARMS[0]]) and
                bundles[ARMS[1]]["prototype_arm"] == ARMS[1], "parent fork identity")
    return old,source,src,cfg,roots,bundles


def paths_signature(state, map_id):
    return run.json_fingerprint([map_id, sorted((a["id"],a["path"]) for a in state["agents"])])


def prepare():
    old,source,src,cfg,roots,bundles = context()
    out = ROOT/cfg["output"]
    run.require(not out.exists(), "existing output; use verify/resume")
    require_unused(roots,out)
    inputs = {name:run.sha256_file(ROOT/name) for name in CODE}
    previous_initials = {}
    for e,folder,row in source_training.audited_batch(old,source,src):
        j = e["job"]
        if j["pair_id"] in previous_initials:
            continue
        path = folder/"initial.json"
        state = run.read_json(path)
        previous_initials[j["pair_id"]] = dict(map_id=j["case"]["map_id"],task_id=j["case"]["task_id"],
            fingerprint=e["expected_initial"],paths_signature=paths_signature(state,j["case"]["map_id"]))
        inputs[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    for name in ("registration.json","update.json","parity.json","terminal_analysis.json","batch.audit.json"):
        inputs[(src/name).relative_to(ROOT).as_posix()] = run.sha256_file(src/name)
    for spec in cfg["models"].values():
        inputs[spec["path"]] = spec["file_sha256"]
    body = dict(schema=cfg["schema"],config=cfg,inputs=inputs,conditions=roots,jobs=schedule(roots,cfg),
        scientific_binding=source["binding"],source_binding=old["binding"],previous_initials=previous_initials,
        source_commit=run.subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        role="new_initial_conditions_on_seen_train_maps",no_ttf=True,no_update=True,no_heldout=True)
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out,body["binding"],"second-batch-prepare"):
        for arm,bundle in bundles.items():
            p = compare.runtime_plan(source,cfg,arm)
            target = run.actor_file(ROOT/p["config"]["output"],bundle["iteration"])
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/cfg["models"][arm]["path"],target)
            run.require(run.sha256_file(target) == cfg["models"][arm]["file_sha256"], "model copy")
            run.once(target.parent/f"receipt-{bundle['iteration']}.json",run.sealed(dict(binding=source["binding"],
                policy_sha256=run.validate_bundle(bundle),file_sha256=run.sha256_file(target),
                metadata=dict(collection_binding=body["binding"],copied_not_trained=True))))
        run.once(out/"registration.json",run.sealed(body))
    return dict(registered=True,binding=body["binding"],qualification=24,episodes=192,max_decisions=None,no_update=True)


def verify():
    old,source,_,cfg,roots,bundles = context()
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/"registration.json"))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding","integrity")}),"registration hash")
    run.require(reg["config"] == cfg and reg["conditions"] == roots and reg["jobs"] == schedule(roots,cfg) and
                reg["scientific_binding"] == source["binding"] and reg["source_binding"] == old["binding"],"registration identity")
    for name,digest in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT,name,field="second batch input")) == digest,"changed input: "+name)
    require_unused(roots,out)
    for a,b in bundles.items():
        p = compare.runtime_plan(source,cfg,a)
        run.require(run.actor_load(ROOT/p["config"]["output"],p,b["iteration"]) == b,"copied actor changed")
    return reg,source,out


def qualification_jobs(reg, source):
    cfg = reg["config"]
    p = dict(source,config=dict(source["config"],output=cfg["output"]+"/qualification"))
    return [dict(c,phase="qualify-second-train",job_id=run.json_fingerprint(["qualify-second-train",c["pair_id"]])[:24],
        plan=p,parent_pid=os.getpid(),collection_binding=reg["binding"]) for c in reg["conditions"]]


def qfolder(j):
    return run.folder_for(ROOT/j["plan"]["config"]["output"],j["phase"],j)


def qualification_worker(j):
    from scripts.train_sa_history_selector import die_with_parent
    from scripts.run_feedback_exploration_diagnostics import validate_final
    die_with_parent(j["parent_pid"])
    run.require(not qfolder(j).exists(),"partial qualification")
    q,env,state,_ = run.reset(j)
    validate_final(state)
    run.once(qfolder(j)/"initial.json",state)
    row = dict(binding=j["plan"]["binding"],execution_binding=j["collection_binding"],status="ok",job_id=j["job_id"],
        pair_id=j["pair_id"],map_id=j["case"]["map_id"],split=j["split"],initial_fingerprint=q.state_fingerprint(state),
        initial_conflicts=state["num_of_colliding_pairs"],feasible=state["feasible"],
        paths_signature=paths_signature(state,j["case"]["map_id"]),files={"initial.json":run.sha256_file(qfolder(j)/"initial.json")})
    run.once(qfolder(j)/"result.json",run.sealed(row))
    return dict(status="ok",job_id=j["job_id"],conflicts=row["initial_conflicts"])


def read_qualification(reg, j):
    row = run.result_read(qfolder(j),j["plan"])
    run.require(row["execution_binding"] == reg["binding"] and row["status"] == "ok" and
                row["split"] == "train" and row["pair_id"] == j["pair_id"] and row["job_id"] == j["job_id"] and
                row["map_id"] == j["case"]["map_id"],"qualification identity")
    state = run.read_json(qfolder(j)/"initial.json")
    from scripts.run_sa_path_quality import state_fingerprint
    run.require(row["initial_fingerprint"] == state_fingerprint(state) and row["paths_signature"] ==
                paths_signature(state,row["map_id"]) and row["feasible"] == state["feasible"] and
                row["initial_conflicts"] == state["num_of_colliding_pairs"],"initial state integrity")
    return row


def execute_batches(reg,out,jobs,worker,folder,reader,phase,resume,timeout):
    from experiments.repair_collection import _run_jobs
    run.require(os.name != "nt","use frozen WSL native")
    run.require(resume or not (out/f"{phase}.status.json").exists(),"explicit resume required")
    pending = []
    for j in jobs:
        run.require(not (out/"failures"/(j["job_id"]+".json")).exists(),"recorded failure needs inspection")
        path = folder(j)
        if (path/"result.json").exists():
            run.require(reader(j)["status"] == "ok","unknown episode needs inspection")
        else:
            run.require(not path.exists(),"partial episode needs inspection")
            pending.append(j)
    if resume:(out/"STOP_AFTER_BATCH").unlink(missing_ok=True)
    def progress(row):
        with (out/"progress.jsonl").open("a",encoding="utf8") as f:f.write(json.dumps(dict(phase=phase,**row))+"\n")
        print(json.dumps(dict(phase=phase,**row)),flush=True)
    def failure(j,status,error):
        row = dict(status="censored" if status == "timeout" else "error",job_id=j["job_id"],error=error)
        run.once(out/"failures"/(j["job_id"]+".json"),run.sealed(dict(binding=reg["binding"],phase=phase,**row)))
        return row
    def status(value):
        run.write_json(out/f"{phase}.status.json",dict(status=value,binding=reg["binding"]))
        run.write_json(out/"run_status.json",dict(status=value,phase=phase,binding=reg["binding"]))
    status("running")
    try:
        for start in range(0,len(pending),reg["config"]["workers"]):
            if (out/"STOP_AFTER_BATCH").exists():
                status("paused")
                return False
            batch = pending[start:start+reg["config"]["workers"]]
            rows = _run_jobs(worker,batch,reg["config"]["workers"],phase=phase,
                output_root=out/"progress"/phase/batch[0]["job_id"],run_fingerprint=reg["binding"],
                timeout_seconds=timeout,on_result=progress,failure_result=failure,stop_on_failure=False)
            if len(rows) != len(batch) or any(r["status"] != "ok" for r in rows):
                status("needs_inspection")
                return False
        files = {}
        for j in jobs:
            run.require(reader(j)["status"] == "ok","unknown terminal")
            files[j["job_id"]] = run.sha256_file(folder(j)/"result.json")
        run.once(out/f"{phase}.complete.json",run.sealed(dict(binding=reg["binding"],jobs=len(jobs),files=files)))
        status("complete")
    except BaseException:
        status("interrupted_or_error")
        raise
    return True


def qualify(resume=False):
    reg,source,out = verify()
    jobs = qualification_jobs(reg,source)
    with recovery.strict_lock(out,reg["binding"],"second-batch-qualify"):
        done = execute_batches(reg,out,jobs,qualification_worker,qfolder,lambda j:read_qualification(reg,j),
                               "qualification",resume,reg["config"]["qualification_fuse_seconds"])
        if not done:return dict(status="paused_or_needs_inspection")
        rows = [read_qualification(reg,j) for j in jobs]
        old_paths = {r["paths_signature"] for r in reg["previous_initials"].values()}
        new_paths = {r["paths_signature"] for r in rows}
        novel = new_paths-old_paths
        report = dict(binding=reg["binding"],rows=rows,valid=len(rows),nonzero=sum(not r["feasible"] for r in rows),
            unique_initial_paths=len(new_paths),novel_initial_paths=len(novel),
            novel_maps=sorted({r["map_id"] for r in rows if r["paths_signature"] in novel}),
            collection_allowed=bool(novel),no_seed_resampling=True,
            complete_sha256=run.sha256_file(out/"qualification.complete.json"))
        run.once(out/"qualification.json",run.sealed(report))
    return {k:report[k] for k in ("valid","nonzero","unique_initial_paths","novel_initial_paths","collection_allowed")}


def ready_jobs(reg,source,out):
    proof = run.check_seal(run.read_json(out/"qualification.json"))
    complete = run.check_seal(run.read_json(out/"qualification.complete.json"))
    qjobs = qualification_jobs(reg,source)
    rows = [read_qualification(reg,j) for j in qjobs]
    run.require(proof["binding"] == complete["binding"] == reg["binding"] and proof["rows"] == rows and
                proof["complete_sha256"] == run.sha256_file(out/"qualification.complete.json") and
                complete["jobs"] == proof["valid"] == len(rows) == 24 and
                set(complete["files"]) == {j["job_id"] for j in qjobs},"qualification proof")
    for j in qjobs:
        run.require(complete["files"][j["job_id"]] == run.sha256_file(qfolder(j)/"result.json"),"changed qualification")
    novel = {r["paths_signature"] for r in rows}-{r["paths_signature"] for r in reg["previous_initials"].values()}
    run.require(proof["collection_allowed"] and proof["novel_initial_paths"] == len(novel) > 0,
                "no new initial conditions; do not collect duplicate batch")
    initial = {r["pair_id"]:r["initial_fingerprint"] for r in rows}
    return [dict(j,expected_initial=initial[j["pair_id"]]) for j in reg["jobs"]]


def collect(resume=False):
    reg,source,out = verify()
    jobs = ready_jobs(reg,source,out)
    with recovery.strict_lock(out,reg["binding"],"second-batch-collect"):
        done = execute_batches(reg,out,[compare.augmented(reg,source,j) for j in jobs],compare.worker,
            lambda j:compare.prior.folder_for(reg["config"],j),lambda j:compare.read_result(reg,source,j),
            "collection",resume,reg["config"]["process_fuse_seconds"])
    return dict(collected=192 if done else None,status="complete" if done else "paused_or_needs_inspection",no_update=True)


def audit():
    from experiments.repair_collection import _run_jobs
    reg,source,out = verify()
    jobs = ready_jobs(reg,source,out)
    complete = run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(complete["binding"] == reg["binding"] and complete["jobs"] == 192 and
                set(complete["files"]) == {j["job_id"] for j in jobs},"complete collection")
    for j in jobs:
        row = compare.read_result(reg,source,j)
        run.require(row["status"] == "ok" and complete["files"][j["job_id"]] ==
                    run.sha256_file(compare.prior.folder_for(reg["config"],j)/"result.json"),"complete result changed")
    with recovery.strict_lock(out,reg["binding"],"second-batch-audit"):
        rows = _run_jobs(runtime.audit_worker,[compare.augmented(reg,source,j) for j in jobs],20,
            phase="second-batch-audit",output_root=out/"audit-progress",run_fingerprint=reg["binding"],timeout_seconds=960.)
        run.require(len(rows) == 192 and all(r["status"] == "ok" for r in rows),"full trajectory audit")
        run.once(out/"audit.json",run.sealed(dict(binding=reg["binding"],results=rows)))
    return dict(audited=192)


def credit_summary(rows, policy, groups):
    coefficients = credit.gradient_coefficients(rows,policy_sha256=policy,expected_groups=groups,
        replicas=4,max_decisions=None,node_budget=25000000)
    counts = {g:sum(r["success"] for r in rows if r["pair_id"] == g) for g in groups}
    mixed = {g:n for g,n in counts.items() if 0 < n < 4}
    by_id = {r["episode_id"]:r for r in rows}
    nonzero = [c for c in coefficients if c["coefficient"] != 0]
    return dict(coefficients=coefficients,mixed_conditions=mixed,effective_maps=sorted({groups[g] for g in mixed}),
        nonzero_credit_episodes=len(nonzero),credited_decisions=sum(by_id[c["episode_id"]]["decisions"] for c in nonzero),
        decisions=sum(r["decisions"] for r in rows),one_update_eligible=bool(nonzero),
        decision="fresh_terminal_credit_available_not_trained" if nonzero else "no_within_condition_credit_do_not_train")


def analyze():
    reg,source,out = verify()
    jobs = ready_jobs(reg,source,out)
    proof = run.check_seal(run.read_json(out/"audit.json"))
    hashes = {r["job_id"]:r["result_sha256"] for r in proof["results"] if r["status"] == "ok"}
    run.require(proof["binding"] == reg["binding"] and len(proof["results"]) == 192 and
                set(hashes) == {j["job_id"] for j in jobs},"audit coverage")
    data = {a:{} for a in ARMS}
    train = []
    for j in jobs:
        folder = compare.prior.folder_for(reg["config"],j)
        row = compare.read_result(reg,source,j)
        run.require(hashes[j["job_id"]] == run.sha256_file(folder/"result.json") and row["status"] == "ok","stale audit")
        data[j["comparison_arm"]][(row["pair_id"],row["replica"])] = row
        if j["comparison_arm"] == ARMS[1]:
            steps = [{k:e[k] for k in ("decision","policy_sha256","probabilities","selected_id","behavior_log_probability")}
                     for e in run.trace_read(folder)]
            train.append(dict(row,steps=steps))
    run.require(set(data[ARMS[0]]) == set(data[ARMS[1]]) and len(train) == 96,"paired coverage")
    for key in data[ARMS[1]]:
        a,b = (data[name][key] for name in ARMS)
        run.require(a["initial_fingerprint"] == b["initial_fingerprint"] and a["rng_stream_id"] == b["rng_stream_id"],"paired initial/RNG")
    groups = {c["pair_id"]:c["case"]["map_id"] for c in reg["conditions"]}
    info = credit_summary(train,reg["config"]["models"][ARMS[1]]["policy_sha256"],groups)
    maps = sorted(set(groups.values()))
    summaries = {a:dict(episodes=len(rows),success=sum(r["success"] for r in rows.values()),
        stops=dict(Counter(r["stop"] for r in rows.values())),
        by_map={m:sum(r["success"] for r in rows.values() if r["map_id"] == m) for m in maps}) for a,rows in data.items()}
    comparison = dict(overall=compare.prior.paired_comparison(data[ARMS[1]],data[ARMS[0]]),by_map={m:
        compare.prior.paired_comparison({k:r for k,r in data[ARMS[1]].items() if r["map_id"] == m},
                                       {k:r for k,r in data[ARMS[0]].items() if r["map_id"] == m}) for m in maps})
    old = run.check_seal(run.read_json(ROOT/reg["config"]["source_training"]/"terminal_analysis.json"))
    old_mixed = old["new_mixed_conditions"]
    result = dict(schema="lns2.sa.second_onpolicy_batch_report.v1",binding=reg["binding"],summaries=summaries,
        paired_exploration_control=comparison,training_information=info,
        previous_actor0_information=dict(mixed_conditions=len(old_mixed),
            effective_maps=len({r["map_id"] for r in old["outcomes"] if r["pair_id"] in old_mixed}),
            source_sha256=run.sha256_file(ROOT/reg["config"]["source_training"]/"terminal_analysis.json"),
            not_a_paired_policy_comparison=True),
        episodes=[dict(comparison_arm=a,**r) for a,rows in data.items() for r in rows.values()],
        qualification_sha256=run.sha256_file(out/"qualification.json"),audit_sha256=run.sha256_file(out/"audit.json"),
        decision=info["decision"],training_arm=ARMS[1],control_excluded_from_gradient=True,
        no_update=True,no_ttf=True,no_heldout=True,independent_maps=False,automatic_promotion=False)
    with recovery.strict_lock(out,reg["binding"],"second-batch-analyze"):
        run.once(out/"report.json",run.sealed(result))
        run.write_json(out/"run_status.json",dict(status="completed",binding=reg["binding"],report_sha256=run.sha256_file(out/"report.json")))
    return dict(summaries=summaries,paired_control=comparison["overall"],information={k:v for k,v in info.items() if k != "coefficients"},
                no_update=True,no_ttf=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","verify","qualify","collect","audit","analyze","stop"))
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args()
    if args.phase in ("qualify","collect"):result = globals()[args.phase](args.resume)
    elif args.phase == "verify":result = dict(verified=True,binding=verify()[0]["binding"])
    elif args.phase == "stop":
        cfg = run.read_json(ROOT/CONFIG)
        run.write_json(ROOT/cfg["output"]/"STOP_AFTER_BATCH",dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else:result = globals()[args.phase]()
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":main()
