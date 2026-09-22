"""One fresh-stream, uncapped development comparison of the frozen actor-2."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import compare_sa_fresh_stream as fresh
from scripts import train_sa_parent_update as training
from scripts.register_sa_crossfit_report_fix import first_difference

run, previous, recovery, runtime = fresh.run, fresh.previous, fresh.recovery, fresh.runtime
CONFIG = "configs/sa_second_update_comparison.json"
ARM = "second_condition"
ARMS = ("dual16_sa", "untrained_exploration", "uncapped_condition", ARM)
CODE = (CONFIG, "scripts/compare_sa_second_update.py", "tests/evaluation/test_sa_second_update.py",
        "docs/SA_SECOND_UPDATE_COMPARISON_PROTOCOL_ZH.md", "scripts/register_sa_crossfit_report_fix.py")


def configuration(old, override):
    expected = dict(phase="crossfit-comparison", replica_ids=[4, 5], conditions=8, expected_jobs=64,
        arms=list(ARMS), max_decisions=None, decision_feature_reference=256, node_budget=25000000,
        pp_safety_seconds=20., episode_safety_seconds=900., process_fuse_seconds=960., workers=20,
        formal_ttf=False, training=False, automatic_promotion=False)
    run.require(all(override[k] == v for k, v in expected.items()), "fixed second comparison scope")
    run.require(override["source_evaluation"] == old["config"]["output"], "source evaluation identity")
    models = {a: old["config"]["models"][a] for a in ARMS if a in old["config"]["models"]}
    models[ARM] = dict(path=override["source_training"]+"/models/actor-2.json", iteration=2,
        file_sha256=override["model_sha256"], policy_sha256=override["policy_sha256"])
    return dict(old["config"], **override, models=models)


def schedule(old, cfg):
    roots = {}
    for j in old["jobs"]:
        run.require(j["phase"] == cfg["phase"] and j["split"] == "development_holdout", "viewed development only")
        identity = {k: j[k] for k in ("case", "pair_id", "solver_seed", "expected_initial", "split", "phase")}
        run.require(j["pair_id"] not in roots or roots[j["pair_id"]] == identity, "inconsistent source condition")
        roots[j["pair_id"]] = identity
    run.require(len(roots) == 8 and len({j["case"]["map_id"] for j in roots.values()}) == 2, "unchanged eight conditions")
    jobs = [dict(root, replica=r, comparison_arm=a, arm="trained_actor" if a in ("uncapped_condition", ARM) else a,
        iteration=cfg["models"].get(a, {}).get("iteration", 0),
        job_id=run.json_fingerprint([cfg["phase"], pair, r, a])[:24])
        for pair, root in sorted(roots.items()) for r in cfg["replica_ids"] for a in ARMS]
    run.require(len(jobs) == len({j["job_id"] for j in jobs}) == 64, "complete paired schedule")
    run.require(not set(map(fresh.stream_key, jobs)) & set(map(fresh.stream_key, old["jobs"])), "reused stream")
    return jobs


def context():
    treg, source, tout, parent = training.verify()
    old, old_source, old_out = fresh.verify()
    cfg = configuration(old, run.read_json(ROOT/CONFIG))
    run.require(tout == ROOT/cfg["source_training"] and old_out == ROOT/cfg["source_evaluation"] and
        source["binding"] == old_source["binding"], "training/evaluation lineage")
    update = run.check_seal(run.read_json(tout/"update.json"))
    parity = run.check_seal(run.read_json(tout/"parity.json"))
    run.require(run.sha256_file(tout/"update.json") == cfg["update_sha256"] and update["updated"] and
        update["binding"] == parity["binding"] == treg["binding"] and
        parity["update_sha256"] == cfg["update_sha256"] and parity["policy_sha256"] == cfg["policy_sha256"] and
        parity["max_error"] <= 1e-12, "frozen update and portable evidence")
    bundles = {a: previous.prior.check_model(cfg, a, source["binding"]) for a in cfg["models"]}
    child = bundles[ARM]
    run.require(bundles["uncapped_condition"] == parent and child == run.actor_load(tout, source, 2) and
        child["parent_policy"] == run.validate_bundle(parent) and child["update_binding"] == treg["binding"] and
        child["prototype_arm"] == ARM, "frozen parent/child lineage")
    jobs = schedule(old, cfg)
    run.require(not {j["case"]["map_id"] for j in jobs} & {e["episode"]["map_id"] for e in treg["entries"]}, "Train map leakage")
    run.require(source["template"]["environment"]["max_repair_iterations"] == 0 and
        source["proposal"]["pp_safety_seconds"] == cfg["pp_safety_seconds"], "unchanged native repair contract")
    return treg, source, tout, old, old_out, cfg, jobs, bundles


def prepare():
    treg, source, tout, old, old_out, cfg, jobs, bundles = context()
    out = ROOT/cfg["output"]
    run.require(not out.exists(), "existing output; use verify/resume")
    streams = fresh.inventory(jobs, out)
    inputs = {n: run.sha256_file(ROOT/n) for n in CODE}
    inputs.update(streams["files"])
    for folder, names in ((tout, (training.REGISTRATION, "update.json", "parity.json")),
                          (old_out, ("registration.json", "audit.json", "report.json", "collection.complete.json"))):
        for name in names:
            run.require(run.check_seal(run.read_json(folder/name))["binding"] in (treg["binding"], old["binding"]), "source proof binding")
            inputs[(folder/name).relative_to(ROOT).as_posix()] = run.sha256_file(folder/name)
    for spec in cfg["models"].values():
        inputs[spec["path"]] = spec["file_sha256"]
    body = dict(schema=cfg["schema"], config=cfg, inputs=inputs, jobs=jobs, streams=streams,
        scientific_binding=source["binding"], training_binding=treg["binding"], source_binding=old["binding"],
        source_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        role="new_random_streams_on_viewed_development_maps", no_training=True, no_ttf=True)
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out, body["binding"], "second-update-prepare"):
        for arm, bundle in bundles.items():
            plan = previous.runtime_plan(source, cfg, arm)
            target = run.actor_file(ROOT/plan["config"]["output"], bundle["iteration"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/cfg["models"][arm]["path"], target)
            run.require(run.sha256_file(target) == cfg["models"][arm]["file_sha256"], "model copy")
            run.once(target.parent/f"receipt-{bundle['iteration']}.json", run.sealed(dict(binding=source["binding"],
                policy_sha256=run.validate_bundle(bundle), file_sha256=run.sha256_file(target),
                metadata=dict(comparison_binding=body["binding"], comparison_arm=arm, copied_not_trained=True))))
        run.once(out/"registration.json", run.sealed(body))
    return dict(binding=body["binding"], jobs=64, streams=16, max_decisions=None, no_training=True, no_ttf=True)


def verify():
    treg, source, _, old, _, cfg, jobs, bundles = context()
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/"registration.json"))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding", "integrity")}), "registration hash")
    run.require(reg["config"] == cfg and reg["jobs"] == jobs and reg["scientific_binding"] == source["binding"] and
        reg["source_binding"] == old["binding"] and reg["training_binding"] == treg["binding"], "comparison identity")
    for name, digest in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT, name, field="second comparison input")) == digest, "changed input: "+name)
    fresh.inventory(jobs, out)
    for arm, bundle in bundles.items():
        p = previous.runtime_plan(source, cfg, arm)
        run.require(run.actor_load(ROOT/p["config"]["output"], p, bundle["iteration"]) == bundle, "copied actor changed")
    return reg, source, out


def collect(resume=False):
    from experiments.repair_collection import _run_jobs
    run.require(os.name != "nt", "use frozen WSL native")
    reg, source, out = verify()
    cfg = reg["config"]
    with recovery.strict_lock(out, reg["binding"], "second-update-collect"):
        pending = fresh.pending_jobs(reg, source, out, resume)
        if resume:
            (out/"STOP_AFTER_BATCH").unlink(missing_ok=True)
        def progress(row):
            with (out/"progress.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row)+"\n")
            print(json.dumps(row), flush=True)
        def failure(j, status, error):
            row = dict(status="censored" if status == "timeout" else "error", job_id=j["job_id"],
                comparison_arm=j["comparison_arm"], error=error)
            run.once(out/"failures"/(j["job_id"]+".json"), run.sealed(dict(binding=reg["binding"], **row)))
            return row
        run.write_json(out/"run_status.json", dict(status="running", binding=reg["binding"], pending=len(pending)))
        try:
            for offset in range(0, len(pending), cfg["workers"]):
                if (out/"STOP_AFTER_BATCH").exists():
                    run.write_json(out/"run_status.json", dict(status="paused", binding=reg["binding"]))
                    return dict(status="paused")
                batch = pending[offset:offset+cfg["workers"]]
                rows = _run_jobs(previous.worker, batch, cfg["workers"], phase="second-update-comparison",
                    output_root=out/"progress"/batch[0]["job_id"], run_fingerprint=reg["binding"],
                    timeout_seconds=cfg["process_fuse_seconds"], on_result=progress, failure_result=failure, stop_on_failure=False)
                if len(rows) != len(batch) or any(r["status"] != "ok" for r in rows):
                    run.write_json(out/"run_status.json", dict(status="needs_inspection", binding=reg["binding"]))
                    return dict(status="needs_inspection", no_automatic_retry=True)
            files = {}
            for j in reg["jobs"]:
                run.require(previous.read_result(reg, source, j)["status"] == "ok", "unknown terminal")
                files[j["job_id"]] = run.sha256_file(previous.prior.folder_for(cfg, j)/"result.json")
            run.once(out/"collection.complete.json", run.sealed(dict(binding=reg["binding"], jobs=64, files=files)))
            run.write_json(out/"run_status.json", dict(status="collected", binding=reg["binding"], jobs=64))
        except BaseException as exc:
            run.write_json(out/"run_status.json", dict(status="interrupted_or_error", binding=reg["binding"], error=repr(exc)))
            raise
    return dict(collected=64, no_ttf=True)


def completed(reg, source, out):
    proof = run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(proof["binding"] == reg["binding"] and proof["jobs"] == 64 and
        set(proof["files"]) == {j["job_id"] for j in reg["jobs"]}, "complete result coverage")
    for j in reg["jobs"]:
        run.require(previous.read_result(reg, source, j)["status"] == "ok" and proof["files"][j["job_id"]] ==
            run.sha256_file(previous.prior.folder_for(reg["config"], j)/"result.json"), "collection result changed")
    return proof


def audit():
    from experiments.repair_collection import _run_jobs
    reg, source, out = verify()
    complete = completed(reg, source, out)
    with recovery.strict_lock(out, reg["binding"], "second-update-audit"):
        rows = _run_jobs(runtime.audit_worker, [previous.augmented(reg, source, j) for j in reg["jobs"]], 20,
            phase="second-update-audit", output_root=out/"audit-progress", run_fingerprint=reg["binding"], timeout_seconds=960.)
        run.require(len(rows) == 64 and all(r["status"] == "ok" for r in rows) and
            {r["job_id"]:r["result_sha256"] for r in rows} == complete["files"], "complete trace audit")
        run.once(out/"audit.json", run.sealed(dict(binding=reg["binding"], results=rows)))
    return dict(audited=64)


def validate_pairs(data):
    run.require(set(data) == set(ARMS), "four registered arms")
    keys = set(data[ARM])
    run.require(len(keys) == 16 and all(set(rows) == keys for rows in data.values()), "complete paired denominator")
    for key in keys:
        rows = [r[key] for r in data.values()]
        run.require(all(r["status"] == "ok" and r["split"] == "development_holdout" for r in rows), "complete development only")
        for field in ("initial_fingerprint", "rng_stream_id", "map_id"):
            run.require(len({r[field] for r in rows}) == 1, "unpaired "+field)


def interpretation(comparisons):
    bounds = {a:v["overall"]["net_success_bounds"] for a,v in comparisons.items()}
    run.require(all(lo == hi for lo,hi in bounds.values()), "unknown outcomes need inspection")
    parent, explore, dual = (bounds[a][0] for a in ("uncapped_condition", "untrained_exploration", "dual16_sa"))
    if parent > 0 and explore > 0 and dual >= 0:
        return "second_update_development_net_gain_needs_confirmation"
    if parent < 0 and dual < 0:
        return "second_update_development_regression_do_not_promote"
    return "second_update_mixed_or_no_net_gain_do_not_promote"


def report():
    reg, source, out = verify()
    complete = completed(reg, source, out)
    audit_proof = run.check_seal(run.read_json(out/"audit.json"))
    run.require(audit_proof["binding"] == reg["binding"] and len(audit_proof["results"]) == 64 and
        all(r["status"] == "ok" for r in audit_proof["results"]) and
        {r["job_id"]:r["result_sha256"] for r in audit_proof["results"]} == complete["files"], "matching full audit")
    data, folders = {a:{} for a in ARMS}, {a:{} for a in ARMS}
    for j in reg["jobs"]:
        row = previous.read_result(reg, source, j)
        key = row["pair_id"], row["replica"]
        run.require(key not in data[j["comparison_arm"]], "duplicate pair")
        data[j["comparison_arm"]][key] = row
        folders[j["comparison_arm"]][key] = previous.prior.folder_for(reg["config"], j)
    validate_pairs(data)
    maps = sorted({r["map_id"] for r in data[ARM].values()})
    comparisons = {a:dict(overall=previous.prior.paired_comparison(data[ARM], data[a]),
        by_map={m:previous.prior.paired_comparison({k:r for k,r in data[ARM].items() if r["map_id"] == m},
                                                {k:r for k,r in data[a].items() if r["map_id"] == m}) for m in maps})
        for a in ARMS if a != ARM}
    summaries = {a:dict(episodes=16, success=sum(r["success"] for r in rows.values()),
        stops=dict(Counter(r["stop"] for r in rows.values())),
        by_map={m:sum(r["success"] for r in rows.values() if r["map_id"] == m) for m in maps},
        beyond_256=sum(r["decisions"] > 256 for r in rows.values()),
        max_observed_decisions=max(r["decisions"] for r in rows.values())) for a,rows in data.items()}
    behavior = []
    for key, left in sorted(data[ARM].items()):
        right = data["uncapped_condition"][key]
        count, first = first_difference(run.trace_read(folders[ARM][key]), run.trace_read(folders["uncapped_condition"][key]))
        if first is None:
            run.require(left["decisions"] == right["decisions"] and left["final_fingerprint"] == right["final_fingerprint"], "unexplained parent divergence")
        behavior.append(dict(pair_id=key[0], replica=key[1], map_id=left["map_id"], common_prefix_decisions=count,
            first_difference=first, success=left["success"], parent_success=right["success"],
            final_conflicts=left["final_conflicts"], parent_final_conflicts=right["final_conflicts"]))
    result = dict(schema="lns2.sa.second_update_report.v1", binding=reg["binding"],
        audit_sha256=run.sha256_file(out/"audit.json"), summaries=summaries, comparisons=comparisons,
        behavior=behavior, episodes=[dict(comparison_arm=a, **r) for a,rows in data.items() for r in rows.values()],
        decision=interpretation(comparisons), max_decisions=None, node_budget=25000000,
        no_training=True, no_ttf=True, automatic_promotion=False, independent_generalization=False,
        uncertainty="16 paired streams, eight conditions, two viewed development maps; not 64 independent tasks")
    with recovery.strict_lock(out, reg["binding"], "second-update-report"):
        run.once(out/"report.json", run.sealed(result))
        run.write_json(out/"run_status.json", dict(status="completed", binding=reg["binding"], jobs=64,
            report_sha256=run.sha256_file(out/"report.json")))
    return dict(summaries=summaries, comparisons={a:v["overall"] for a,v in comparisons.items()}, decision=result["decision"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "dry-run", "collect", "audit", "report", "stop"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "collect":
        result = collect(args.resume)
    elif args.phase in ("verify", "dry-run"):
        reg, _, _ = verify()
        result = dict(verified=True, binding=reg["binding"], jobs=64, streams=16, workers=20, batches=4,
                      max_decisions=None, solver_called=False, process_budget_upper_minutes=64)
    elif args.phase == "stop":
        cfg = run.read_json(ROOT/CONFIG)
        run.require((ROOT/cfg["output"]/"registration.json").exists(), "unregistered run")
        run.write_json(ROOT/cfg["output"]/"STOP_AFTER_BATCH", dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
