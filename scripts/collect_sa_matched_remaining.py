"""Pair-only H32 collection using the frozen PP/SA worker and two-stage budget."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, json_fingerprint, read_json, sha256_file
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from experiments import sa_source_matched_analysis as stats
from scripts import collect_sa_source_matched as old
from scripts import prepare_sa_matched_remaining as prep
from scripts.audit_sa_history_information import atomic, require
from scripts.audit_sa_matched_training_coverage import neutral_crossfit
from scripts.recover_sa_source_matched import semantic_row
from scripts.run_sa_paired_closed_loop import once

CONFIG = ROOT / "configs/sa_matched_remaining_collection.json"


def check_contract(cfg, prepared):
    fixed = prepared["config"]
    for key, other in (("seed", "trial_seed"), ("trials", "planned_trials"), ("horizon", "planned_horizon"),
                       ("workers", "planned_workers"), ("pp_seconds", "planned_pp_safety_seconds"),
                       ("trial_seconds", "planned_trial_timeout_seconds")):
        require(cfg[key] == fixed[other], "prepared budget changed: " + key)
    require(cfg["schema"] == "lns2.sa.matched_remaining_collection.v1" and cfg["no_ttf"] and
            not cfg["training_allowed"] and not cfg["automatic_promotion"] and cfg["sustain"] == 8 and
            cfg["preflight_workers"] == 4 and cfg["role"] == "previously_viewed_development", "collection boundary")
    require(cfg["source"] == fixed["source"] and cfg["bootstrap"] == 5000 and cfg["bootstrap_seed"] == 202609183,
            "source/analysis contract changed")
    out = (ROOT / cfg["output"]).resolve()
    require(out != (ROOT / "build").resolve() and out.is_relative_to((ROOT / "build").resolve()) and
            out != (ROOT / fixed["output"]).resolve(), "unsafe collection output")
    roots = prepared["roots"]
    require(len(roots) == 31 and len({r["state_id"] for r in roots}) == 31, "root count")
    for root in roots:
        ids = [c["candidate_id"] for c in root["candidates"]]
        require(len(ids) == 2 and set(ids) == set(root["pair_ids"]) and len(set(ids)) == 2, "pair-only candidates")
        require(root["family"] in fixed["allowed_families"] and
                all(prep.historical.signature(c) == (16, (root["family"],)) for c in root["candidates"]), "family or size mismatch")


def build_plan():
    cfg = read_json(CONFIG)
    path = contained_file(ROOT, cfg["preparation"], field="preparation")
    require(sha256_file(path) == cfg["preparation_sha256"], "preparation bytes changed")
    prepared = read_json(path)
    require(prepared["binding"] == json_fingerprint({k:v for k,v in prepared.items() if k != "binding"}), "preparation binding")
    check_contract(cfg, prepared)
    inputs = dict(prepared["inputs"])
    paths = [CONFIG, Path(__file__), path, ROOT / "tests/evaluation/test_sa_matched_remaining_collection.py",
             ROOT / "docs/SA_MATCHED_REMAINING_COLLECTION_ZH.md", ROOT / "scripts/recover_sa_source_matched.py"]
    old.merge_pinned(inputs, {p.relative_to(ROOT).as_posix():sha256_file(p) for p in paths})
    for rel, digest in inputs.items():
        require(sha256_file(contained_file(ROOT, rel, field="collection input")) == digest, "input drift: " + rel)
    jobs, phases = prep.planned_batches(prepared["roots"], prepared["config"])
    require(phases == prepared["phases"] and len(jobs) == 496, "staged schedule drift")
    result = dict(config=cfg, roots=prepared["roots"], phases=phases, inputs=inputs, budget=prepared["budget"],
                  preparation_binding=prepared["binding"], native_sha256=old.NATIVE_SHA,
                  role=cfg["role"], adapter_registered=True, native_preflight_required=True,
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    result["binding"] = json_fingerprint(result)
    return result


def verify(runtime_only=False):
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()) and out.resolve() != (ROOT / "build").resolve(), "unsafe output")
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg and plan["binding"] == json_fingerprint({k:v for k,v in plan.items() if k != "binding"}), "plan/config drift")
    for rel, digest in plan["inputs"].items():
        if not runtime_only or not rel.startswith("build/") or rel == old.NATIVE:
            require(sha256_file(contained_file(ROOT, rel, field="registered input")) == digest, "input drift: " + rel)
    return plan, out


def prepare():
    plan = build_plan()
    once(ROOT / plan["config"]["output"] / "plan.json", plan)
    return dict(binding=plan["binding"], jobs=496, native_preflight_passed=False)


def preflight_batches(plan):
    roots = plan["roots"]
    short = min(roots, key=lambda r:(r["decision"], r["state_id"]))
    long = min(roots, key=lambda r:(-r["decision"], r["state_id"]))
    require(short["state_id"] != long["state_id"], "preflight requires two roots")
    primary, repeat = [], []
    for root in (short, long):
        ids = [f"{root['state_id']}-{c}-t0" for c in root["pair_ids"]]
        primary.extend(ids)
        repeat.append(ids[0])
    return primary, repeat


def inspect_jobs(plan, out, allowed):
    pending, completed = old.existing_jobs(plan, out)
    require({j["job_id"] for j in completed} <= allowed, "unexpected completed job in this phase")
    return {j["job_id"]:j for j in pending if j["job_id"] in allowed}, {j["job_id"] for j in completed}


def execute(plan, out, phases, *, resume=False, limit_batches=None):
    require(limit_batches is None or type(limit_batches) is int and limit_batches > 0, "invalid batch limit")
    allowed = {jid for phase in phases for batch in phase["batches"] for jid in batch}
    require(len(allowed) == sum(len(b) for p in phases for b in p["batches"]), "duplicated job across batches")
    with _CollectionRunLock(out, plan["binding"], "matched-remaining"):
        require(resume or not (out / "trials").exists(), "existing jobs require --resume")
        marker = out / "STOP_AFTER_BATCH"
        if resume and marker.exists():
            require(read_json(marker)["binding"] == plan["binding"], "foreign stop marker")
            marker.unlink()
        pending, completed = inspect_jobs(plan, out, allowed)
        launched = 0
        def status(state, **extra):
            value = dict(status=state, binding=plan["binding"], total=len(allowed), complete=len(completed), **extra)
            atomic(out / "run_status.json", value)
            return value
        def progress(row):
            with (out / "progress.jsonl").open("a", encoding="utf8") as stream:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        try:
            status("running")
            for phase in phases:
                for index, ids in enumerate(phase["batches"]):
                    if marker.exists() or limit_batches is not None and launched >= limit_batches:
                        return status("paused")
                    batch = [pending[jid] for jid in ids if jid in pending]
                    if not batch:
                        continue
                    verify(runtime_only=True)
                    status("running", phase=phase["name"], batch=index, workers=phase["workers"])
                    rows = _run_jobs(old.trial_worker, batch, workers=phase["workers"], phase="matched-remaining",
                                     output_root=out / "progress", run_fingerprint=plan["binding"],
                                     timeout_seconds=plan["config"]["trial_seconds"], on_result=progress,
                                     failure_result=old.failure, stop_on_failure=False)
                    require(len(rows) == len(batch) and {r["job_id"] for r in rows} == {j["job_id"] for j in batch}, "scheduler result coverage")
                    require(all(r["status"] == "ok" for r in rows), "job error; active batch drained; review required")
                    for job in batch:
                        old.receipt(out / "trials" / job["job_id"], job, plan)
                        completed.add(job["job_id"])
                        pending.pop(job["job_id"])
                    launched += 1
                    state = status("running", phase=phase["name"], batch=index, workers=phase["workers"])
                    print(json.dumps(state), flush=True)
            verify()
            _, completed = inspect_jobs(plan, out, allowed)
            require(completed == allowed, "end-of-run job coverage")
            return status("completed")
        except BaseException as exc:
            status("error", error=repr(exc))
            raise


def preflight_report(plan, out):
    primary, repeat = preflight_batches(plan)
    lookup = {j["job_id"]:j for j in old.schedule(plan)}
    pins, rows = {}, {}
    for sub, ids in (("preflight", primary), ("preflight/repeat", repeat)):
        _, complete = inspect_jobs(plan, out / sub, set(ids))
        require(complete == set(ids), "native preflight incomplete")
        for jid in ids:
            folder = out / sub / "trials" / jid
            row = old.receipt(folder, lookup[jid], plan)
            require(old.validate_trial(row, lookup[jid], plan) is not None, "native preflight censored")
            rows[(sub,jid)] = row
            for path in folder.iterdir():
                pins[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for jid in repeat:
        require(semantic_row(rows[("preflight",jid)]) == semantic_row(rows[("preflight/repeat",jid)]), "native repeat semantics differ")
    return dict(binding=plan["binding"], passed=True, primary_jobs=primary, repeat_jobs=repeat,
                input_hashes=pins, new_scientific_labels=0, model_fits=0, no_ttf=True)


def preflight(resume=False):
    plan, out = verify()
    primary, repeat = preflight_batches(plan)
    for sub, ids in (("preflight", primary), ("preflight/repeat", repeat)):
        result = execute(plan, out / sub, [dict(name=sub, workers=4, batches=[ids])], resume=resume)
        if result["status"] != "completed":
            return result
    result = preflight_report(plan, out)
    once(out / "preflight_report.json", result)
    return {k:v for k,v in result.items() if k != "input_hashes"}


def collect(resume=False, limit_batches=None):
    plan, out = verify()
    require(read_json(out / "preflight_report.json") == preflight_report(plan, out), "preflight receipt changed")
    return execute(plan, out, plan["phases"], resume=resume, limit_batches=limit_batches)


def pair_record(record):
    pair = sorted(record["pair_ids"])
    values = record["values"]
    require(len(pair) == 2 and len(set(pair)) == 2 and set(values) == set(pair), "pair-only values")
    require(all(len(v) == 8 and all(type(x) is bool or x is None for x in v) for v in values.values()), "eight bool/unknown trials required")
    a,b = pair
    complete = all(v is not None for x in values.values() for v in x)
    counts = Counter("unknown" if x is None or y is None else "both_success" if x and y else
                     "both_failure" if not x and not y else "a_wins" if x else "b_wins" for x,y in zip(values[a],values[b]))
    rates = {cid:stats._candidate_stats(v) for cid,v in values.items()}
    partitions, directions = [], Counter()
    if complete:
        for left,right in stats.partitions():
            signs = [stats._direction(stats._rate(values,a,h)-stats._rate(values,b,h)) for h in (left,right)]
            category = "both_tied" if signs == [0,0] else "one_tied" if 0 in signs else "strict_same_direction" if signs[0] == signs[1] else "strict_opposite_direction"
            directions[category] += 1
            partitions.append(sum(stats._crossfit_direction(values,pair,s,t)["gain"] for s,t in ((left,right),(right,left)))/2)
    delta = (sum(values[a])-sum(values[b]))/8 if complete else None
    return dict(state_id=record["state_id"], map_id=record["map_id"], pair_ids=pair, family=record["family"],
                anchor_id=record["anchor_id"], anchor_sampled=record["anchor_id"] in values, pair_complete=complete,
                candidates=rates, paired_counts={k:counts[k] for k in ("a_wins","b_wins","both_success","both_failure","unknown")},
                all_trial_difference=delta, absolute_all_trial_difference=abs(delta) if complete else None,
                pair_uniform_rate=(sum(values[a])+sum(values[b]))/16 if complete else None,
                crossfit_gain=math.fsum(partitions)/35 if complete else None,
                tie_neutral_crossfit=neutral_crossfit(record) if complete else None,
                half_direction_counts={k:directions[k] for k in ("strict_same_direction","strict_opposite_direction","both_tied","one_tied")} if complete else None)


def summarize(records, cfg):
    require(records and len({r["state_id"] for r in records}) == len(records), "empty/duplicate cohort")
    rows = [pair_record(r) for r in records]
    maps = sorted({r["map_id"] for r in rows})
    rng = random.Random(cfg["bootstrap_seed"])
    draws = [tuple(rng.randrange(len(maps)) for _ in maps) for _ in range(cfg["bootstrap"])]
    metrics = {field:stats._aggregate(rows, field, maps, draws) for field in
               ("pair_uniform_rate","absolute_all_trial_difference","crossfit_gain","tie_neutral_crossfit")}
    return dict(states=len(rows), maps=len(maps), rows=rows, metrics=metrics,
                complete_states=sum(r["pair_complete"] for r in rows),
                nonzero_contrast_states=sum(r["all_trial_difference"] is not None and r["all_trial_difference"] != 0 for r in rows),
                discordant_positions=sum(r["paired_counts"]["a_wins"]+r["paired_counts"]["b_wins"] for r in rows),
                missing_policy="retain_unknown_no_imputation_no_complete_case_selection",
                weighting="equal_maps_then_equal_states", bootstrap_unit="map",
                independent_confirmation=False, model_fits=0, automatic_promotion=False, no_ttf=True)


def analyze():
    plan, out = verify()
    pending, completed = old.existing_jobs(plan, out)
    require(not pending and len(completed) == 496, "collection incomplete")
    lookup = {j["job_id"]:j for j in completed}
    records, pins, stops = [], {}, Counter()
    for root in plan["roots"]:
        values = {}
        for cid in root["pair_ids"]:
            values[cid] = []
            for trial in range(8):
                jid = f"{root['state_id']}-{cid}-t{trial}"
                folder = out / "trials" / jid
                row = old.receipt(folder, lookup[jid], plan)
                values[cid].append(old.validate_trial(row, lookup[jid], plan))
                stops[row["stop"]] += 1
                for path in folder.iterdir():
                    pins[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        records.append({k:root[k] for k in ("state_id","map_id","pair_ids","anchor_id","family","decision")} | dict(values=values))
    report = summarize(records, plan["config"])
    report.update(binding=plan["binding"], input_hashes=pins, stops=dict(stops), trial_jobs=496,
                  valid_trials=sum(v is not None for r in records for vs in r["values"].values() for v in vs),
                  native_preflight=preflight_report(plan,out)["passed"])
    once(out / "analysis_records.json", records)
    once(out / "report.json", report)
    return {k:v for k,v in report.items() if k not in ("input_hashes","rows")}


def stop(preflight_only=False):
    cfg = read_json(CONFIG)
    out = (ROOT / cfg["output"]).resolve()
    require(out.is_relative_to((ROOT/"build").resolve()) and out != (ROOT/"build").resolve(), "unsafe stop")
    plan = read_json(out / "plan.json")
    require(plan["binding"] == json_fingerprint({k:v for k,v in plan.items() if k != "binding"}), "stop plan identity")
    for target in ([out/"preflight",out/"preflight/repeat"] if preflight_only else [out]):
        atomic(target / "STOP_AFTER_BATCH", dict(binding=plan["binding"]))
    return dict(stop_requested=True, drain_active_batch=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare","verify","dry-run","preflight","collect","analyze","stop"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare": result = prepare()
    elif args.phase == "preflight": result = preflight(args.resume)
    elif args.phase == "collect": result = collect(args.resume, args.limit_batches)
    elif args.phase == "analyze": result = analyze()
    elif args.phase == "stop": result = stop(args.preflight_only)
    else:
        plan,out = verify()
        pending,complete = old.existing_jobs(plan,out)
        result = dict(binding=plan["binding"], pending=len(pending), complete=len(complete), budget=plan["budget"])
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
