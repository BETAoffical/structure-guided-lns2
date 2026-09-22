"""Frozen-policy repeat on unused streams, with no decision-count stopping rule."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import compare_sa_uncapped_update as last

run, previous, recovery, runtime = last.run, last.previous, last.recovery, last.runtime
CONFIG = "configs/sa_uncapped_fresh_stream.json"
ARMS = ("official_sa", "dual16_sa", "untrained_exploration", "bounded_condition", last.ARM)
CODE = (CONFIG, "scripts/compare_sa_fresh_stream.py", "tests/evaluation/test_sa_fresh_stream.py",
        "docs/SA_FRESH_STREAM_PROTOCOL_ZH.md")


def configuration(prior, baseline, override):
    expected = dict(phase="crossfit-comparison", replica_ids=[2, 3], conditions=8, expected_jobs=80,
                    arms=list(ARMS), max_decisions=None, decision_feature_reference=256,
                    node_budget=25000000, pp_safety_seconds=20., episode_safety_seconds=900.,
                    process_fuse_seconds=960., workers=20, formal_ttf=False, training=False,
                    automatic_promotion=False)
    run.require(all(override[k] == v for k, v in expected.items()), "frozen repeat scope")
    run.require(override["source_comparison"] == prior["config"]["output"], "previous comparison identity")
    models = {a: baseline["config"]["models"][a] for a in ARMS if a in baseline["config"]["models"]}
    models[last.ARM] = prior["config"]["models"][last.ARM]
    cfg = dict(prior["config"], **override, models=models)
    for key in ("arm", "expected_new", "expected_reused"):
        cfg.pop(key, None)
    return cfg


def stream_key(job):
    return job["phase"], job["pair_id"], job["replica"]


def schedule(prior, cfg):
    roots = {j["pair_id"]: j for j in prior["jobs"] if j["replica"] == 0}
    run.require(len(roots) == 8 and len({j["case"]["map_id"] for j in roots.values()}) == 2,
                "unchanged eight conditions on two development maps")
    run.require(all(j["split"] == "development_holdout" and j["phase"] == cfg["phase"]
                    for j in roots.values()), "development scope")
    jobs = []
    for pair, root in sorted(roots.items()):
        for replica in cfg["replica_ids"]:
            for arm in cfg["arms"]:
                jobs.append(dict(root, replica=replica, comparison_arm=arm,
                    arm="trained_actor" if arm in ("bounded_condition", last.ARM) else arm,
                    iteration=cfg["models"].get(arm, {}).get("iteration", 0),
                    job_id=run.json_fingerprint([cfg["phase"], pair, replica, arm])[:24]))
    run.require(len(jobs) == len({j["job_id"] for j in jobs}) == cfg["expected_jobs"], "schedule count")
    run.require(not {stream_key(j) for j in jobs} & {stream_key(j) for j in prior["jobs"]}, "reused stream")
    return jobs


def registry_streams(value):
    if isinstance(value, dict):
        if all(k in value for k in ("phase", "pair_id", "replica")):
            yield stream_key(value)
        for item in value.values():
            yield from registry_streams(item)
    elif isinstance(value, list):
        for item in value:
            yield from registry_streams(item)


def inventory(jobs, out):
    # Inspect registration metadata, never select streams by outcomes.
    fresh = {stream_key(j) for j in jobs}
    files, seen = {}, set()
    for path in sorted((ROOT/"build").glob("sa-*/registration.json")):
        if path.parent == out:
            continue
        keys = set(registry_streams(run.read_json(path)))
        run.require(not fresh & keys, "previously registered stream: " + path.parent.name)
        relevant = {k for k in keys if k[0] == "crossfit-comparison"}
        if relevant:
            files[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
            seen.update(relevant)
    run.require(seen and len(fresh) == 16, "missing historical stream inventory")
    return dict(files=files, prior_streams=[list(k) for k in sorted(seen)],
                fresh_streams=[list(k) for k in sorted(fresh)])


def context():
    prior, source, prior_out, baseline = last.verify()
    cfg = configuration(prior, baseline, run.read_json(ROOT/CONFIG))
    treg, _, _ = last.training.verify()
    jobs = schedule(prior, cfg)
    train_jobs = [e["job"] for e in treg["entries"].values()]
    run.require(not {stream_key(j) for j in jobs} & {stream_key(j) for j in train_jobs}, "training RNG leakage")
    run.require(not {j["case"]["map_id"] for j in jobs} & {j["case"]["map_id"] for j in train_jobs},
                "training map leakage")
    run.require(source["template"]["environment"]["max_repair_iterations"] == 0 and
                source["proposal"]["pp_safety_seconds"] == cfg["pp_safety_seconds"], "frozen repair contract")
    bundles = {a: previous.prior.check_model(cfg, a, source["binding"]) for a in cfg["models"]}
    return prior, source, prior_out, cfg, jobs, bundles


def prepare():
    prior, source, prior_out, cfg, jobs, bundles = context()
    out = ROOT/cfg["output"]
    run.require(not out.exists(), "existing output; use verify/resume")
    streams = inventory(jobs, out)
    proof = run.check_seal(run.read_json(prior_out/"report.json"))
    run.require(proof["binding"] == prior["binding"] and proof["decision"] ==
                "development_net_gain_needs_fresh_stream_confirmation", "repeat prerequisite")
    inputs = {name: run.sha256_file(ROOT/name) for name in CODE}
    inputs.update(streams["files"])
    for name in ("registration.json", "report.json", "audit.json", "collection.complete.json"):
        path = prior_out/name
        inputs[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    for spec in cfg["models"].values():
        inputs[spec["path"]] = spec["file_sha256"]
    body = dict(schema=cfg["schema"], config=cfg, inputs=inputs, jobs=jobs, streams=streams,
                scientific_binding=source["binding"], source_binding=prior["binding"],
                source_commit=run.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                role="fresh_stream_repeat_on_viewed_development_maps", no_training=True, no_ttf=True)
    body["binding"] = run.json_fingerprint(body)
    with recovery.strict_lock(out, body["binding"], "fresh-stream-prepare"):
        for arm, bundle in bundles.items():
            p = previous.runtime_plan(source, cfg, arm)
            target = run.actor_file(ROOT/p["config"]["output"], bundle["iteration"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/cfg["models"][arm]["path"], target)
            run.require(run.sha256_file(target) == cfg["models"][arm]["file_sha256"], "copied model bytes")
            run.once(target.parent/f"receipt-{bundle['iteration']}.json", run.sealed(dict(
                binding=source["binding"], policy_sha256=run.validate_bundle(bundle), file_sha256=run.sha256_file(target),
                metadata=dict(comparison_binding=body["binding"], comparison_arm=arm, copied_not_trained=True))))
        run.once(out/"registration.json", run.sealed(body))
    return dict(binding=body["binding"], jobs=len(jobs), streams=16, max_decisions=None, no_ttf=True)


def verify():
    prior, source, _, cfg, jobs, bundles = context()
    out = ROOT/cfg["output"]
    reg = run.check_seal(run.read_json(out/"registration.json"))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding", "integrity")}),
                "registration hash")
    run.require(reg["config"] == cfg and reg["jobs"] == jobs and reg["scientific_binding"] == source["binding"] and
                reg["source_binding"] == prior["binding"], "repeat identity")
    for name, digest in reg["inputs"].items():
        run.require(run.sha256_file(run.contained_file(ROOT, name, field="repeat input")) == digest, "changed input: " + name)
    inventory(jobs, out)
    for arm, bundle in bundles.items():
        p = previous.runtime_plan(source, cfg, arm)
        run.require(run.actor_load(ROOT/p["config"]["output"], p, bundle["iteration"]) == bundle, "copied model identity")
    return reg, source, out


def pending_jobs(reg, source, out, resume):
    run.require(resume or not (out/"run_status.json").exists(), "explicit resume required")
    pending = []
    for j in reg["jobs"]:
        folder = previous.prior.folder_for(reg["config"], j)
        run.require(not (out/"failures"/(j["job_id"]+".json")).exists(), "recorded failure needs inspection")
        if (folder/"result.json").exists():
            run.require(previous.read_result(reg, source, j)["status"] == "ok", "censored episode needs inspection")
        else:
            run.require(not folder.exists(), "partial episode needs inspection")
            pending.append(previous.augmented(reg, source, j))
    return pending


def collect(resume=False):
    from experiments.repair_collection import _run_jobs
    run.require(os.name != "nt", "use frozen WSL native")
    reg, source, out = verify()
    cfg = reg["config"]
    with recovery.strict_lock(out, reg["binding"], "fresh-stream-collect"):
        pending = pending_jobs(reg, source, out, resume)
        if resume:
            (out/"STOP_AFTER_BATCH").unlink(missing_ok=True)
        def progress(row):
            with (out/"progress.jsonl").open("a", encoding="utf8") as f:
                f.write(json.dumps(row)+"\n")
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
                rows = _run_jobs(previous.worker, batch, cfg["workers"], phase="fresh-stream",
                    output_root=out/"progress"/batch[0]["job_id"], run_fingerprint=reg["binding"],
                    timeout_seconds=cfg["process_fuse_seconds"], on_result=progress,
                    failure_result=failure, stop_on_failure=False)
                if len(rows) != len(batch) or any(r["status"] != "ok" for r in rows):
                    run.write_json(out/"run_status.json", dict(status="needs_inspection", binding=reg["binding"]))
                    return dict(status="needs_inspection", no_automatic_retry=True)
            files = {}
            for j in reg["jobs"]:
                run.require(previous.read_result(reg, source, j)["status"] == "ok", "unknown terminal")
                files[j["job_id"]] = run.sha256_file(previous.prior.folder_for(cfg, j)/"result.json")
            run.once(out/"collection.complete.json", run.sealed(dict(binding=reg["binding"], jobs=len(files), files=files)))
            run.write_json(out/"run_status.json", dict(status="collected", binding=reg["binding"], jobs=len(files)))
        except BaseException as exc:
            run.write_json(out/"run_status.json", dict(status="interrupted_or_error", binding=reg["binding"], error=repr(exc)))
            raise
    return dict(collected=len(files), no_ttf=True)


def audit():
    from experiments.repair_collection import _run_jobs
    reg, source, out = verify()
    complete = run.check_seal(run.read_json(out/"collection.complete.json"))
    run.require(complete["binding"] == reg["binding"] and complete["jobs"] == len(reg["jobs"]) and
                set(complete["files"]) == {j["job_id"] for j in reg["jobs"]}, "collection coverage")
    for j in reg["jobs"]:
        run.require(previous.read_result(reg, source, j)["status"] == "ok" and complete["files"][j["job_id"]] ==
                    run.sha256_file(previous.prior.folder_for(reg["config"], j)/"result.json"), "collection changed")
    with recovery.strict_lock(out, reg["binding"], "fresh-stream-audit"):
        rows = _run_jobs(runtime.audit_worker, [previous.augmented(reg, source, j) for j in reg["jobs"]], 20,
            phase="fresh-stream-audit", output_root=out/"audit-progress", run_fingerprint=reg["binding"], timeout_seconds=960.)
        run.require(len(rows) == 80 and all(r["status"] == "ok" for r in rows), "full audit failed")
        run.once(out/"audit.json", run.sealed(dict(binding=reg["binding"], results=rows)))
    return dict(audited=80)


def interpretation(comparisons):
    decision = last.interpretation(comparisons)
    if decision == "development_net_gain_needs_fresh_stream_confirmation":
        return "net_gain_repeated_on_fresh_streams_not_independent_maps"
    if decision == "development_regression_do_not_promote":
        return "fresh_stream_regression_do_not_promote"
    return "net_gain_not_repeated_do_not_promote"


def report():
    reg, source, out = verify()
    proof = run.check_seal(run.read_json(out/"audit.json"))
    hashes = {r["job_id"]: r["result_sha256"] for r in proof["results"] if r["status"] == "ok"}
    run.require(proof["binding"] == reg["binding"] and len(proof["results"]) == 80 and
                set(hashes) == {j["job_id"] for j in reg["jobs"]}, "audit coverage")
    data = {a:{} for a in ARMS}
    for j in reg["jobs"]:
        row = previous.read_result(reg, source, j)
        run.require(hashes[j["job_id"]] == run.sha256_file(previous.prior.folder_for(reg["config"], j)/"result.json"), "stale audit")
        data[j["comparison_arm"]][last.pair_key(row)] = row
    last.validate_pairs(data)
    maps = sorted({r["map_id"] for r in data[last.ARM].values()})
    comparisons = {base:dict(overall=previous.prior.paired_comparison(data[last.ARM], data[base]),
        by_map={m:previous.prior.paired_comparison({k:r for k,r in data[last.ARM].items() if r["map_id"] == m},
                                                {k:r for k,r in data[base].items() if r["map_id"] == m}) for m in maps})
        for base in ARMS if base != last.ARM}
    summaries = {a:dict(episodes=len(rows), success=sum(r["success"] for r in rows.values()),
        by_map={m:sum(r["success"] for r in rows.values() if r["map_id"] == m) for m in maps},
        stops=dict(Counter(r["stop"] for r in rows.values())),
        beyond_256=sum(r["decisions"] > 256 for r in rows.values()),
        max_observed_decisions=max(r["decisions"] for r in rows.values())) for a,rows in data.items()}
    old = run.check_seal(run.read_json(ROOT/reg["config"]["source_comparison"]/"report.json"))
    result = dict(schema="lns2.sa.fresh_stream_report.v1", binding=reg["binding"],
        audit_sha256=run.sha256_file(out/"audit.json"), summaries=summaries, comparisons=comparisons,
        episodes=[dict(comparison_arm=a, **r) for a,rows in data.items() for r in rows.values()],
        previous_batch_descriptive={a:old["summaries"][a] for a in ARMS},
        pooled_descriptive={a:dict(episodes=32, success=summaries[a]["success"]+old["summaries"][a]["success"])
                            for a in ARMS},
        decision=interpretation(comparisons), max_decisions=None, node_budget=25000000,
        no_training=True, no_ttf=True, automatic_promotion=False, independent_generalization=False,
        role=reg["role"], uncertainty="16 paired streams, eight conditions, two previously viewed maps; not 80 independent tasks")
    with recovery.strict_lock(out, reg["binding"], "fresh-stream-report"):
        run.once(out/"report.json", run.sealed(result))
        run.write_json(out/"run_status.json", dict(status="completed", binding=reg["binding"],
            jobs=80, report_sha256=run.sha256_file(out/"report.json")))
    return dict(summaries=summaries, comparisons={k:v["overall"] for k,v in comparisons.items()}, decision=result["decision"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "collect", "audit", "report", "stop"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "collect":
        result = collect(args.resume)
    elif args.phase == "verify":
        result = dict(verified=True, binding=verify()[0]["binding"], jobs=80)
    elif args.phase == "stop":
        cfg = run.read_json(ROOT/CONFIG)
        run.write_json(ROOT/cfg["output"]/"STOP_AFTER_BATCH", dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else:
        result = globals()[args.phase]()
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
