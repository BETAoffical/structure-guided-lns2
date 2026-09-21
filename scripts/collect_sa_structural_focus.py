"""Fixed second-core intervention followed by unchanged Dual16+SA, not TTF."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import FunctionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, json_fingerprint, read_json, sha256_file
from experiments.neighborhood_candidates import candidate_id
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from scripts import collect_sa_source_matched as old
from scripts import probe_sa_structural_focus as focus
from scripts.audit_sa_history_information import atomic, require
from scripts.collect_sa_history_candidate_bridge import branch_labels
from scripts.recover_sa_source_matched import semantic_row
from scripts.run_sa_paired_closed_loop import once

CONFIG = ROOT / "configs/sa_structural_focus_continuation.json"
CODE = ("scripts/collect_sa_structural_focus.py", "configs/sa_structural_focus_continuation.json",
        "tests/evaluation/test_sa_structural_focus_continuation.py",
        "docs/SA_STRUCTURAL_FOCUS_CONTINUATION_PROTOCOL_ZH.md")


def intervention_roots(source, report):
    entries = {r["state_id"]: r for r in source["roots"]}
    roots = []
    for row in report["rows"]:
        changed = [a for a in row["attempts"] if a["status"] == "different_members"]
        if not changed:
            continue
        require(len(changed) == 1, "not the frozen single-variant scope")
        variant = changed[0]
        require(not variant["in_original_pool"] and not variant["labeled_with_anchor"], "variant scope drift")
        entry = entries[row["state_id"]]
        require(all(entry[k] == row[k] for k in ("anchor_id", "decision", "map_id")), "root metadata drift")
        saved = read_json(ROOT / entry["source_root"])
        require(saved["state_fingerprint"] == entry["root_fingerprint"] and
                saved["old_selected_id"] == entry["anchor_id"], "source identity")
        anchor = next(c for c in saved["control_event"]["pool"] if c["candidate_id"] == entry["anchor_id"])
        alt = dict(candidate_id=variant["candidate_id"], agents=variant["agents"], actual_size=16,
                   selection_families=["diagnostic-second-core:" + variant["family"]])
        require(len(alt["agents"]) == len(set(alt["agents"])) == 16 and
                candidate_id(alt["agents"]) == alt["candidate_id"], "variant identity")
        roots.append({k: entry[k] for k in ("state_id", "map_id", "decision", "episode", "source_root",
                      "source_binding", "root_fingerprint", "anchor_id")} |
                     dict(candidates=[anchor, alt], alternate_id=alt["candidate_id"], variant=variant,
                          action_input_snapshot=row["action_input_snapshot"]))
    require(len(roots) == 15 and len({r["state_id"] for r in roots}) == 15 and
            len({r["map_id"] for r in roots}) == 8, "frozen 15-root scope changed")
    return roots


def prepare():
    cfg = read_json(CONFIG)
    source = focus.verify_plan()
    require(source["binding"] == cfg["focus_binding"], "focus binding changed")
    report_path = ROOT / cfg["focus"] / "report.json"
    require(sha256_file(report_path) == cfg["focus_report_sha256"], "focus report changed")
    report = read_json(report_path)
    require(report["binding"] == source["binding"], "focus report binding")
    previous = read_json(ROOT / cfg["runtime_plan"])
    require(previous["binding"] == json_fingerprint({k: v for k, v in previous.items() if k != "binding"}), "runtime plan binding")
    for key in ("source", "trials", "horizon", "sustain", "workers", "pp_seconds", "trial_seconds"):
        require(cfg[key] == previous["config"][key], "frozen runtime budget drift: " + key)
    require(cfg["no_ttf"] and not cfg["training_allowed"] and not cfg["automatic_promotion"], "diagnostic scope")
    inputs = dict(source["inputs"])
    old.merge_pinned(inputs, previous["inputs"])
    for name in CODE + (cfg["runtime_plan"], cfg["focus"] + "/plan.json", cfg["focus"] + "/report.json",
                        "scripts/recover_sa_source_matched.py", "scripts/collect_sa_source_matched.py"):
        old.merge_pinned(inputs, {name: sha256_file(ROOT / name)})
    roots = intervention_roots(source, report)
    prefix = sum(r["decision"] for r in roots) * cfg["trials"] * 2
    plan = dict(schema=cfg["schema"], config=cfg, roots=roots, inputs=inputs,
                native_sha256=old.NATIVE_SHA, role=cfg["role"],
                budget=dict(jobs=240, max_h32_repairs=7680, max_prefix_repairs=prefix,
                            total_serial_fuse_budget_seconds=240 * cfg["trial_seconds"],
                            ideal_20_worker_budget_seconds=12 * cfg["trial_seconds"]),
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    require(inputs[old.NATIVE] == old.NATIVE_SHA and len(old.schedule(plan)) == 240, "native/jobs drift")
    plan["binding"] = json_fingerprint(plan)
    out = output(cfg)
    once(out / "plan.json", plan)
    return dict(binding=plan["binding"], budget=plan["budget"])


def output(cfg):
    out = ROOT / cfg["output"]
    require(out.resolve().is_relative_to((ROOT / "build").resolve()) and not out.is_symlink(), "unsafe output")
    return out


def verify(runtime_only=False):
    cfg = read_json(CONFIG)
    out = output(cfg)
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg and plan["binding"] == json_fingerprint(
            {k: v for k, v in plan.items() if k != "binding"}), "plan/config drift")
    for name, digest in plan["inputs"].items():
        if runtime_only and name.startswith("build/") and name != old.NATIVE:
            continue
        require(sha256_file(contained_file(ROOT, name, field="focus continuation input")) == digest,
                "pinned input changed: " + name)
    return plan, out


def forced_root(root, entry, cid):
    """Only the forced first-action pool changes; continuation uses frozen code."""
    require(cid in (entry["anchor_id"], entry["alternate_id"]), "unregistered intervention")
    pool = root["control_event"]["pool"]
    anchor, alt = entry["candidates"]
    require(anchor["candidate_id"] == entry["anchor_id"] and alt["candidate_id"] == entry["alternate_id"], "arm order")
    require(pool[root["control_event"]["selected_index"]] == anchor, "anchor is not the original choice")
    require(alt["candidate_id"] not in {c["candidate_id"] for c in pool}, "alternate was in original pool")
    require(candidate_id(alt["agents"]) == alt["candidate_id"] and len(set(alt["agents"])) == len(alt["agents"]) == 16,
            "alternate membership identity")
    require(set(alt["agents"]) <= {a["id"] for a in root["state"]["agents"]}, "unknown alternate agent")
    if cid == entry["anchor_id"]:
        return root
    return dict(root, control_event=dict(root["control_event"], pool=pool + [deepcopy(alt)]))


def restore(job, plan, deadline):
    from scripts import run_sa_path_quality as q
    entry = job["root"]
    anchor_job = dict(job, candidate_id=entry["anchor_id"],
                      root=dict(entry, candidates=[entry["candidates"][0]]))
    env, root, case, anchor = old.restore(anchor_job, plan, deadline)
    require(anchor == entry["candidates"][0], "restored anchor metadata changed")
    state = q._plain(env.get_state())
    require(json_fingerprint(focus.action_input(state)) == entry["action_input_snapshot"], "actual root input drift")
    material = focus.materialize(state, root["control_event"]["pool"], entry["anchor_id"], [])
    actual = [a for a in material["attempts"] if a["status"] == "different_members"]
    require(actual == [entry["variant"]], "actual second-core neighborhood differs from preregistration")
    require(q.state_fingerprint(env.get_state()) == root["state_fingerprint"], "materialization mutated state")
    old.remaining(deadline)
    root = forced_root(root, entry, job["candidate_id"])
    candidate = next(c for c in entry["candidates"] if c["candidate_id"] == job["candidate_id"])
    return env, root, case, candidate


def validate_trial(row, job, plan):
    from scripts import run_sa_path_quality as q
    from scripts.run_feedback_exploration_diagnostics import validate_final
    require((row.get("root_id"), row.get("candidate_id"), row.get("trial")) ==
            (job["root"]["state_id"], job["candidate_id"], job["trial"]), "trial identity")
    if row["status"] == "censored":
        require(row["stop"] in ("hard_fuse", "prefix_or_total_safety"), "unknown censor reason")
        return None
    require(row["status"] == "ok" and row["no_ttf"], "failed trial/scope")
    root = forced_root(read_json(ROOT / job["root"]["source_root"]), job["root"], job["candidate_id"])
    require(len(row["events"]) <= plan["config"]["horizon"], "excess continuation")
    if row["events"]:
        require(row["events"][0]["pool"] == root["control_event"]["pool"], "forced pool drift")
    else:
        require(row["stop"] == "wall_safety", "empty non-censored continuation")
    label = branch_labels(row, root, dict(target=root["source"], previous_best=root["previous_best"]), plan["config"])
    state = root["state"]
    for i, event in enumerate(row["events"]):
        require(not state["feasible"], "continued after feasibility")
        metrics = event["metrics"]
        incomplete = metrics["pp_failure_reason"] == "time_limit" or not metrics["acceptance_evaluated"]
        require(not incomplete or (i == len(row["events"]) - 1 and row["stop"] == "incomplete_pp"), "incomplete PP used as label")
        state = q.apply_state_delta(state, event["delta"])
    require(state["sum_of_costs"] == row["final_cost"] and
            state["low_level"]["generated"] - root["state"]["low_level"]["generated"] == row["generated"], "cost/node summary drift")
    validate_final(state)
    require(bool(state["feasible"]) == (row["stop"] == "feasible"), "feasibility/stop mismatch")
    require(row["stop"] in ("horizon", "feasible", "incomplete_pp", "wall_safety"), "unknown stop")
    return None if label is None else bool(label["completion"])


def private_call(fn, bindings, *args):
    local = FunctionType(fn.__code__, dict(fn.__globals__, **bindings), fn.__name__, fn.__defaults__, fn.__closure__)
    local.__kwdefaults__ = fn.__kwdefaults__
    return local(*args)


def trial_worker(job):
    return private_call(old.trial_worker, dict(restore=restore, validate_trial=validate_trial), job)


def failure(job, status, error):
    try:
        return private_call(old._failure, dict(validate_trial=validate_trial), job, status, error)
    except Exception as exc:
        return dict(status="error", job_id=job["job_id"], error=repr(exc), original_failure=error)


def inventory(plan, out, allowed):
    pending, done = old.existing_jobs(plan, out)
    require({j["job_id"] for j in done} <= set(allowed), "out-of-scope completed job")
    for job in done:
        validate_trial(old.receipt(out / "trials" / job["job_id"], job, plan), job, plan)
    return [j for j in pending if j["job_id"] in allowed], done


def execute(plan, out, allowed, workers, resume=False):
    require(len(allowed) == len(set(allowed)), "duplicate jobs")
    with _CollectionRunLock(out, plan["binding"], "focus-continuation"):
        require(resume or not (out / "trials").exists(), "existing outputs require --resume")
        marker = out / "STOP_AFTER_BATCH"
        if marker.exists() and resume:
            require(read_json(marker)["binding"] == plan["binding"], "stop marker binding")
            marker.unlink()
        pending, done = inventory(plan, out, allowed)
        try:
            for offset in range(0, len(pending), workers):
                if marker.exists():
                    break
                verify(runtime_only=True)
                atomic(out / "run_status.json", dict(status="running", binding=plan["binding"],
                       complete=len(done), total=len(allowed), workers=workers))
                batch = pending[offset:offset + workers]
                def progress(result):
                    with (out / "progress.jsonl").open("a", encoding="utf8") as stream:
                        stream.write(json.dumps(result, sort_keys=True) + "\n")
                    print(json.dumps(result, sort_keys=True), flush=True)
                rows = _run_jobs(trial_worker, batch, workers=workers, phase="focus-continuation",
                    output_root=out / "progress", run_fingerprint=plan["binding"],
                    timeout_seconds=plan["config"]["trial_seconds"], on_result=progress,
                    failure_result=failure, stop_on_failure=False)
                require(len(rows) == len(batch) and {r["job_id"] for r in rows} == {j["job_id"] for j in batch}, "batch coverage")
                require(all(r["status"] == "ok" for r in rows), "batch drained; error requires review before resume")
                # Validate the new batch now; completed receipts are checked again at exit.
                for job in batch:
                    validate_trial(old.receipt(out / "trials" / job["job_id"], job, plan), job, plan)
                done.extend(batch)
                print("BATCH", len(done), "/", len(allowed), flush=True)
            verify()
            pending, done = inventory(plan, out, allowed)
            result = dict(status="paused" if pending else "completed", binding=plan["binding"], complete=len(done), total=len(allowed))
            atomic(out / "run_status.json", result)
            return result
        except BaseException as exc:
            atomic(out / "run_status.json", dict(status="error", binding=plan["binding"], error=repr(exc)))
            raise


def preflight_jobs(plan):
    roots = sorted(plan["roots"], key=lambda r: (r["decision"], r["state_id"]))
    selected = {roots[0]["state_id"], roots[-1]["state_id"]}
    return [j["job_id"] for j in old.schedule(plan) if j["root"]["state_id"] in selected and j["trial"] == 0]


def preflight_report(plan, out):
    lookup = {j["job_id"]: j for j in old.schedule(plan)}
    ids, inputs = preflight_jobs(plan), {}
    require(len(ids) == 4, "preflight scope")
    for jid in ids:
        rows = []
        for name in ("preflight", "preflight-repeat"):
            folder = out / name / "trials" / jid
            row = old.receipt(folder, lookup[jid], plan)
            require(validate_trial(row, lookup[jid], plan) is not None, "preflight unknown")
            rows.append(row)
            inputs.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in folder.iterdir()})
        require(semantic_row(rows[0]) == semantic_row(rows[1]), "repeat scientific result mismatch")
    return dict(binding=plan["binding"], passed=True, independent_jobs=8, scientific_jobs=0, input_hashes=inputs)


def summarize(records, cfg):
    import numpy as np
    maps = sorted({r["map_id"] for r in records})
    by_map, rows = {}, []
    totals = {arm: dict(feasible=0, horizon_nonfeasible=0, unknown=0) for arm in ("anchor", "alternate")}
    for record in records:
        r = {k: record[k] for k in ("state_id", "map_id", "decision", "family", "removed_members")}
        for arm in totals:
            values = record[arm]
            require(len(values) == cfg["trials"], "incomplete trial coverage")
            count = dict(feasible=sum(v is True for v in values), horizon_nonfeasible=sum(v is False for v in values),
                         unknown=sum(v is None for v in values))
            for k, v in count.items():
                totals[arm][k] += v
            r[arm] = count | dict(rate_lower=count["feasible"] / len(values),
                                  rate_upper=(count["feasible"] + count["unknown"]) / len(values))
        r["delta_bounds"] = [r["alternate"]["rate_lower"] - r["anchor"]["rate_upper"],
                               r["alternate"]["rate_upper"] - r["anchor"]["rate_lower"]]
        rows.append(r)
    for m in maps:
        subset = [r for r in rows if r["map_id"] == m]
        by_map[m] = dict(roots=len(subset), delta_bounds=np.mean([r["delta_bounds"] for r in subset], axis=0).tolist())
    bounds = np.asarray([by_map[m]["delta_bounds"] for m in maps])
    draws = np.random.default_rng(cfg["bootstrap_seed"]).integers(0, len(maps), size=(cfg["bootstrap"], len(maps)))
    samples = bounds[draws].mean(axis=1)
    ci = [float(np.quantile(samples[:, 0], .025)), float(np.quantile(samples[:, 1], .975))]
    known = not any(v["unknown"] for v in totals.values())
    delta = bounds.mean(axis=0).tolist()
    decision = "unknown_resource_censoring" if not known else (
        "positive_development_signal" if delta[0] > 0 and ci[0] > 0 else
        "negative_development_signal" if delta[1] < 0 and ci[1] < 0 else "mixed_or_inconclusive_development")
    return dict(roots=len(rows), maps=len(maps), arms=totals, by_map=by_map, rows=rows,
                root_weighted_delta_bounds=np.mean([r["delta_bounds"] for r in rows], axis=0).tolist(),
                map_balanced_delta_bounds=delta, map_bootstrap_ci95=ci,
                paired_trials=dict(gains=sum(a is False and b is True for r in records for a, b in zip(r["anchor"], r["alternate"])),
                                   losses=sum(a is True and b is False for r in records for a, b in zip(r["anchor"], r["alternate"])),
                                   unknown=sum(a is None or b is None for r in records for a, b in zip(r["anchor"], r["alternate"]))),
                decision=decision, no_ttf=True, training_allowed=False, controller_promotion=False)


def analyze():
    from scripts import run_sa_path_quality as q
    plan, out = verify()
    allowed = [j["job_id"] for j in old.schedule(plan)]
    pending, done = inventory(plan, out, allowed)
    require(not pending and len(done) == 240, "collection incomplete")
    lookup = {j["job_id"]: j for j in done}
    records, details, inputs, stops = [], [], {}, {}
    for entry in plan["roots"]:
        record = {k: entry[k] for k in ("state_id", "map_id", "decision")}
        record.update(family=entry["variant"]["family"], removed_members=len(entry["variant"]["removed"]))
        for arm, cid in (("anchor", entry["anchor_id"]), ("alternate", entry["alternate_id"])):
            record[arm] = []
            for t in range(plan["config"]["trials"]):
                jid = f"{entry['state_id']}-{cid}-t{t}"
                folder = out / "trials" / jid
                row = old.receipt(folder, lookup[jid], plan)
                value = validate_trial(row, lookup[jid], plan)
                record[arm].append(value)
                stops[row["stop"]] = stops.get(row["stop"], 0) + 1
                inputs.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in folder.iterdir()})
                detail = dict(job_id=jid, state_id=entry["state_id"], arm=arm, trial=t, completion=value, stop=row["stop"])
                if row["status"] == "ok":
                    state = read_json(ROOT / entry["source_root"])["state"]
                    counts = [state["num_of_colliding_pairs"]]
                    for event in row["events"]:
                        state = q.apply_state_delta(state, event["delta"])
                        counts.append(state["num_of_colliding_pairs"])
                    detail.update(conflicts=counts, generated=row["generated"], final_cost=row["final_cost"],
                                  repair_steps=len(row["events"]), diagnostic_seconds=row["diagnostic_seconds"])
                details.append(detail)
        records.append(record)
    result = summarize(records, plan["config"]) | dict(binding=plan["binding"], input_hashes=inputs, stops=stops)
    once(out / "analysis_records.json", records)
    once(out / "trial_details.json", details)
    once(out / "report.json", result)
    return {k: v for k, v in result.items() if k not in ("input_hashes", "rows", "by_map")}


def main():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "dry-run", "preflight", "collect", "analyze", "stop"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare()
    elif args.phase == "stop":
        out = output(read_json(CONFIG))
        plan = read_json(out / "plan.json")
        atomic(out / "STOP_AFTER_BATCH", dict(binding=plan["binding"]))
        result = dict(stop_requested=True, drain_active_jobs=True)
    else:
        plan, out = verify()
        if args.phase == "preflight":
            ids = preflight_jobs(plan)
            for name in ("preflight", "preflight-repeat"):
                execute(plan, out / name, ids, plan["config"]["preflight_workers"], args.resume)
            result = preflight_report(plan, out)
            once(out / "preflight.json", result)
            result = {k: v for k, v in result.items() if k != "input_hashes"}
        elif args.phase == "collect":
            require(read_json(out / "preflight.json") == preflight_report(plan, out), "preflight changed")
            result = execute(plan, out, [j["job_id"] for j in old.schedule(plan)], plan["config"]["workers"], args.resume)
        elif args.phase == "analyze":
            result = analyze()
        else:
            pending, done = inventory(plan, out, [j["job_id"] for j in old.schedule(plan)])
            result = dict(binding=plan["binding"], budget=plan["budget"], pending=len(pending), complete=len(done))
    print(json.dumps(result, sort_keys=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
