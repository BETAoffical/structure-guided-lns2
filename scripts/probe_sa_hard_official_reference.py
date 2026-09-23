"""Same-initial-state Official+SA reference on two previously studied Train maps."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_sa_terminal_sampling as sampling

batch, run, compare, recovery, runtime = sampling.batch, sampling.run, sampling.compare, sampling.recovery, sampling.runtime
CONFIG = "configs/sa_hard_official_reference.json"
REGISTRATION = "reference_registration.json"
ARM = "official_sa"
CODE = (CONFIG, "scripts/probe_sa_hard_official_reference.py", "tests/evaluation/test_sa_hard_official_reference.py",
        "docs/SA_HARD_OFFICIAL_REFERENCE_PROTOCOL_ZH.md")


def select_controls(reg, cfg):
    maps = {cfg["hard_map"], cfg["control_map"]}
    run.require(len(maps) == 2, "distinct hard and control maps")
    selected = [j for j in reg["jobs"] if j["case"]["map_id"] in maps]
    run.require(len(selected) == len({j["job_id"] for j in selected}) == 16, "sixteen reference positions")
    groups = {}
    for j in selected:
        c = j["case"]
        run.require(j["split"] == "train" and c["task_variant"] == cfg["task_variant"] and
                    c["task_id"] == c["map_id"] + "__task_0001" and
                    j["pair_id"] == f"{c['task_id']}-s{j['solver_seed']}" and
                    j["solver_seed"] in cfg["solver_seeds"], "exact Train condition identity")
        groups.setdefault((c["map_id"], j["solver_seed"]), []).append(j)
    run.require(set(groups) == {(m, s) for m in maps for s in cfg["solver_seeds"]}, "map/seed coverage")
    run.require(all(len(js) == 4 and {j["replica"] for j in js} == set(range(4)) and
                    len({j["expected_initial"] for j in js}) == 1 for js in groups.values()), "paired replicas")
    return selected


def schedule(controls):
    return [dict(j, comparison_arm=ARM, arm=ARM, iteration=0, control_job_id=j["job_id"],
                 job_id=run.json_fingerprint([j["phase"], j["pair_id"], j["replica"], ARM])[:24]) for j in controls]


def context():
    parent, source, parent_out, old = sampling.verify()
    cfg = run.read_json(ROOT / CONFIG)
    fixed = dict(task_variant="bottleneck_d25", solver_seeds=[233,239], replicas=4, expected_jobs=16,
        max_decisions=None, decision_feature_reference=256, node_budget=25000000, pp_safety_seconds=20.,
        episode_safety_seconds=900., process_fuse_seconds=960., workers=20, formal_ttf=False,
        training=False, automatic_promotion=False)
    run.require(all(cfg[k] == v for k,v in fixed.items()), "frozen reference scope")
    run.require(parent_out == ROOT/cfg["source"] and run.sha256_file(parent_out/"report.json") ==
                cfg["source_report_sha256"], "frozen sampling result")
    proof = sampling.completed(parent, source, parent_out)
    audit = run.check_seal(run.read_json(parent_out/"audit.json"))
    run.require(audit["binding"] == parent["binding"] and len(audit["results"]) == 48 and
                all(r["status"] == "ok" for r in audit["results"]) and
                {r["job_id"]:r["result_sha256"] for r in audit["results"]} == proof["files"], "source audit")
    controls = select_controls(parent,cfg)
    run.require({j["case"]["map_id"] for j in controls} <= set(source["split"]["train_maps"]), "no heldout data")
    cfg = dict(cfg, arms=[ARM], models={}, phase=parent["config"]["phase"])
    plan = compare.runtime_plan(source,cfg,ARM)
    run.require(plan["template"]["environment"]["max_repair_iterations"] == 0 and
                plan["proposal"]["pp_safety_seconds"] == cfg["pp_safety_seconds"], "frozen PP and uncapped runtime")
    return parent, source, parent_out, old, cfg, controls


def prepare():
    parent, source, parent_out, old, cfg, controls = context()
    out = ROOT/cfg["output"]
    run.require(not out.exists(), "existing reference; use verify/resume")
    inputs = {p:run.sha256_file(ROOT/p) for p in CODE}
    for name in (sampling.REGISTRATION, "collection.complete.json", "audit.json", "report.json"):
        inputs[(parent_out/name).relative_to(ROOT).as_posix()] = run.sha256_file(parent_out/name)
    for j in controls:
        folder = compare.prior.folder_for(parent["config"],j)
        row = compare.read_result(parent,source,j)
        for name in ("result.json", "comparison_receipt.json", *row["files"]):
            inputs[(folder/name).relative_to(ROOT).as_posix()] = run.sha256_file(folder/name)
    body = dict(schema=cfg["schema"], config=cfg, inputs=inputs, jobs=schedule(controls), control_jobs=controls,
        source_binding=parent["binding"], scientific_binding=source["binding"],
        source_commit=run.subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        role="same_initial_train_reference_not_selector_only_ablation", no_training=True, no_ttf=True, no_heldout=True,
        rng_contract="paired SA draws; preserve official native RNG, not paired PP orders")
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out,body["binding"],"hard-reference-prepare"):
        run.once(out/REGISTRATION,run.sealed(body))
    return dict(registered=True, binding=body["binding"], new_jobs=16, no_decision_cap=True)


def check_inputs(reg):
    for name,digest in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT,name,field="reference input")) == digest,
                    "changed reference input: " + name)


def verify():
    parent, source, _, old, cfg, controls = context()
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/REGISTRATION))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding","integrity")}),
                "reference registration hash")
    run.require(reg["config"] == cfg and reg["jobs"] == schedule(controls) and reg["control_jobs"] == controls and
                reg["source_binding"] == parent["binding"] and reg["scientific_binding"] == source["binding"],
                "reference registration identity")
    check_inputs(reg)
    return reg, source, out, parent, old


def collect(resume=False, verified=None):
    reg,source,out,_,_ = verified or verify()
    check_inputs(reg)
    with recovery.strict_lock(out,reg["binding"],"hard-reference-collect"):
        done = batch.execute_batches(reg,out,[compare.augmented(reg,source,j) for j in reg["jobs"]],compare.worker,
            lambda j:compare.prior.folder_for(reg["config"],j),lambda j:compare.read_result(reg,source,j),
            "collection",resume,reg["config"]["process_fuse_seconds"])
    return dict(status="collected" if done else "paused_or_needs_inspection", no_ttf=True)


def completed(reg,source,out):
    proof = run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(proof["binding"] == reg["binding"] and proof["jobs"] == 16 and
                set(proof["files"]) == {j["job_id"] for j in reg["jobs"]}, "complete reference required")
    for j in reg["jobs"]:
        row = compare.read_result(reg,source,j)
        run.require(row["status"] == "ok" and run.sha256_file(compare.prior.folder_for(reg["config"],j)/"result.json") ==
                    proof["files"][j["job_id"]], "unknown or changed reference")
    return proof


def audit(verified=None):
    from experiments.repair_collection import _run_jobs
    reg,source,out,_,_ = verified or verify()
    check_inputs(reg)
    proof = completed(reg,source,out)
    with recovery.strict_lock(out,reg["binding"],"hard-reference-audit"):
        rows = _run_jobs(runtime.audit_worker,[compare.augmented(reg,source,j) for j in reg["jobs"]],20,
            phase="hard-reference-audit",output_root=out/"audit-progress",run_fingerprint=reg["binding"],timeout_seconds=960.)
        run.require(len(rows) == 16 and all(r["status"] == "ok" for r in rows) and
                    {r["job_id"]:r["result_sha256"] for r in rows} == proof["files"], "full reference trace audit")
        run.once(out/"audit.json",run.sealed(dict(binding=reg["binding"],results=rows)))
    return dict(audited=16)


def semantic_step(e):
    # Ignore unused random draws and wall time, but retain paths, work, order and rollback outcomes.
    delta = dict(e["delta"], top_set={k:v for k,v in e["delta"].get("top_set",{}).items() if k not in ("runtime","context")})
    m = e["metrics"]
    return dict(before=e["before"], delta=delta, neighborhood=m["neighborhood"], repair_order=m["repair_order"],
                failure=m["pp_failure_reason"], applied=m["step_applied"], after=m["conflicts_after"])


def trace_summary(folder,row):
    digest = hashlib.sha256(row["initial_fingerprint"].encode())
    conflicts, sizes, rules, increased = [], Counter(), Counter(), 0
    for e in run.trace_read(folder):
        digest.update(run.json_fingerprint(semantic_step(e)).encode())
        m = e["metrics"]
        conflicts.append(m["conflicts_after"])
        sizes[len(m["neighborhood"])] += 1
        rules[m["applied_heuristic"]] += 1
        increased += m["conflicts_after"] > m["conflicts_before"]
    run.require(len(conflicts) == row["decisions"], "summary trace length")
    return dict(trajectory_sha256=digest.hexdigest(), minimum_post_conflicts=min(conflicts) if conflicts else row["final_conflicts"],
                last50_conflicts=conflicts[-50:], sizes=dict(sizes), heuristics=dict(rules), accepted_increases=increased)


def reference_decision(official, cfg):
    hard = [r for r in official.values() if r["map_id"] == cfg["hard_map"]]
    control = [r for r in official.values() if r["map_id"] == cfg["control_map"]]
    if any(r["status"] != "ok" for r in (*hard,*control)) or not hard or not control:
        return "incomplete_reference_no_causal_conclusion"
    if any(r["success"] for r in hard):
        return "official_success_witness_available_compare_candidate_coverage_next"
    if all(r["success"] for r in control):
        return "hard_failure_shared_at_fixed_budget_no_success_labels_for_retraining"
    return "reference_also_loses_control_inspect_configuration_before_expansion"


def report(verified=None):
    reg,source,out,parent,old = verified or verify()
    check_inputs(reg)
    proof = completed(reg,source,out)
    aud = run.check_seal(run.read_json(out/"audit.json"))
    run.require(aud["binding"] == reg["binding"] and len(aud["results"]) == 16 and
                all(r["status"] == "ok" for r in aud["results"]) and
                {r["job_id"]:r["result_sha256"] for r in aud["results"]} == proof["files"], "matching reference audit")
    data = {a:{} for a in (ARM,"actor1","sampler","actor0")}
    behavior = []
    old_lookup = {(j["pair_id"],j["replica"],j["comparison_arm"]):j for j in batch.ready_jobs(old,source,ROOT/old["config"]["output"])}
    for j,c in zip(reg["jobs"],reg["control_jobs"]):
        key = j["pair_id"],j["replica"]
        jobs = [(ARM,reg,j),("sampler",parent,c)] + [(name,old,old_lookup[(*key,arm)]) for name,arm in
                                                   (("actor1","uncapped_condition"),("actor0","untrained_exploration"))]
        for name,registration,job in jobs:
            row = compare.read_result(registration,source,job)
            data[name][key] = row
            folder = compare.prior.folder_for(registration["config"],job)
            behavior.append(dict(reference_arm=name,pair_id=key[0],replica=key[1],map_id=row["map_id"],
                                 **trace_summary(folder,row)))
        run.require(len({data[a][key]["initial_fingerprint"] for a in data}) == 1 and
                    len({data[a][key]["rng_stream_id"] for a in data}) == 1, "initial/SA stream mismatch")
    maps = [reg["config"]["control_map"],reg["config"]["hard_map"]]
    summaries = {a:dict(episodes=len(rows),success=sum(r["success"] for r in rows.values()),
        by_map={m:sum(r["success"] for r in rows.values() if r["map_id"] == m) for m in maps},
        stops=dict(Counter(r["stop"] for r in rows.values())),max_decisions_observed=max(r["decisions"] for r in rows.values()))
        for a,rows in data.items()}
    distinct = {a:{p:len({b["trajectory_sha256"] for b in behavior if b["reference_arm"] == a and b["pair_id"] == p})
                  for p,_ in data[a]} for a in data}
    comparisons = {a:{m:compare.prior.paired_comparison(
        {k:r for k,r in data[ARM].items() if r["map_id"] == m},
        {k:r for k,r in data[a].items() if r["map_id"] == m}) for m in maps} for a in data if a != ARM}
    body = dict(schema="lns2.sa.hard_official_reference_report.v1",binding=reg["binding"],summaries=summaries,
        comparisons_official_minus_other=comparisons,distinct_trajectories_per_condition=distinct,behavior=behavior,
        episodes=[dict(reference_arm=a,**r) for a,rows in data.items() for r in rows.values()],
        decision=reference_decision(data[ARM],reg["config"]),audit_sha256=run.sha256_file(out/"audit.json"),
        max_decisions=None,node_budget=25000000,no_training=True,no_ttf=True,no_heldout=True,automatic_promotion=False,
        caveat="two selected Train maps; four initial conditions; replicas alter SA draws, not official PP reseeding; not selector-only")
    with recovery.strict_lock(out,reg["binding"],"hard-reference-report"):
        run.once(out/"report.json",run.sealed(body))
        run.write_json(out/"run_status.json",dict(status="completed",binding=reg["binding"],report_sha256=run.sha256_file(out/"report.json")))
    return dict(summaries=summaries,distinct_trajectories=distinct,decision=body["decision"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("prepare","verify","dry-run","collect","audit","report","all","stop"))
    p.add_argument("--resume",action="store_true")
    args = p.parse_args()
    if args.phase == "all":
        verified = verify()
        result = collect(args.resume,verified)
        if result["status"] == "collected":
            print(json.dumps(audit(verified)),flush=True)
            result = report(verified)
    elif args.phase == "collect": result = collect(args.resume)
    elif args.phase in ("verify","dry-run"):
        reg,_,_,_,_ = verify()
        result = dict(binding=reg["binding"],new_jobs=16,reused_controls=48,initial_conditions=4,workers=20,
                      effective_workers=16,max_decisions=None,node_budget=25000000,external_safety_seconds=960,no_ttf=True)
    elif args.phase == "stop":
        run.write_json(ROOT/run.read_json(ROOT/CONFIG)["output"]/"STOP_AFTER_BATCH",dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else: result = globals()[args.phase]()
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__": main()
