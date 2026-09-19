"""Freeze the unlabelled matched-pair complement; no solver or fitting entry."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, json_fingerprint, read_json, sha256_file
from experiments.sa_paired_completion import require
from scripts import prepare_sa_source_matched as historical
from scripts.collect_sa_history_candidate_bridge import randomization
from scripts.collect_sa_source_matched import NATIVE, NATIVE_SHA, schedule
from scripts.run_sa_paired_closed_loop import once

CONFIG = "configs/sa_matched_remaining_preparation.json"
OWN_INPUTS = [CONFIG, "scripts/prepare_sa_matched_remaining.py",
              "tests/evaluation/test_sa_matched_remaining_preparation.py",
              "docs/SA_MATCHED_REMAINING_PREPARATION_ZH.md"]


def check_config(cfg):
    require(cfg["schema"] == "lns2.sa.matched_remaining_preparation.v1", "schema changed")
    require(cfg["new_solver_calls"] == 0 and not cfg["training_allowed"] and
            not cfg["collection_ready"] and not cfg["formal_ttf_allowed"], "preparation boundary changed")
    fixed = dict(universe_roots=47, excluded_roots=16, remaining_roots=31, maps=8,
                 actual_size=16, planned_trials=8, planned_horizon=32,
                 planned_trial_timeout_seconds=180, planned_pp_safety_seconds=5,
                 planned_workers=20, long_prefix_from_decision=32,
                 planned_long_prefix_workers=4, extra_anchor_jobs=0,
                 sampling_seed=202609181, trial_seed=202609182)
    for key, value in fixed.items():
        require(type(cfg[key]) is int and cfg[key] == value, "fixed contract changed: " + key)
    require(cfg["role"] == "previously_viewed_development" and
            sorted(cfg["allowed_families"]) == ["collision:16", "random:16", "target:16"], "role/families changed")
    output = (ROOT / cfg["output"]).resolve()
    require(output != (ROOT / "build").resolve() and output.is_relative_to((ROOT / "build").resolve()),
            "unsafe output directory")


def select_complement(records, excluded_ids, cfg):
    records = [historical.blind_root(r) for r in records]
    ids = {r["state_id"] for r in records}
    excluded = set(excluded_ids)
    require(len(ids) == len(records) == cfg["universe_roots"], "universe identity/count")
    require(len(excluded) == len(excluded_ids) == cfg["excluded_roots"] and excluded <= ids,
            "excluded roots are not a unique registered subset")
    require(len({r["episode"] for r in records}) == len(records), "repeated source episode")
    chosen = []
    for record in sorted(records, key=lambda r: (r["map_id"], r["state_id"])):
        if record["state_id"] in excluded:
            continue
        # Reuse the frozen family/member hash exactly; only root quota and
        # extra-anchor scheduling differ from the sixteen-root experiment.
        row = historical.select([record], cfg | dict(maps=1, roots_per_map=1))[0]
        row["candidates"] = [c for c in row["candidates"] if c["candidate_id"] in row["pair_ids"]]
        require(len(row["candidates"]) == 2, "pair-only contract")
        chosen.append(row)
    require({r["state_id"] for r in chosen} == ids - excluded and len(chosen) == cfg["remaining_roots"],
            "complement changed")
    require(len({r["map_id"] for r in chosen}) == cfg["maps"], "remaining map coverage")
    return chosen


def planned_batches(roots, cfg):
    scheduled = schedule(dict(roots=roots, config=dict(trials=cfg["planned_trials"])))
    require(len({j["job_id"] for j in scheduled}) == len(scheduled), "duplicate scheduled job")
    phases = []
    for long_prefix in (False, True):
        jobs = [j for j in scheduled if (j["root"]["decision"] >= cfg["long_prefix_from_decision"]) == long_prefix]
        workers = cfg["planned_long_prefix_workers"] if long_prefix else cfg["planned_workers"]
        phases.append(dict(name="long_prefix" if long_prefix else "short_prefix", workers=workers,
            roots=len({j["root"]["state_id"] for j in jobs}), jobs=len(jobs),
            batches=[[j["job_id"] for j in jobs[i:i+workers]] for i in range(0, len(jobs), workers)]))
    return scheduled, phases


def streams(state_id, decision, seed, horizon=32, trials=8):
    return [json_fingerprint([randomization(state_id, t, decision + d, dict(seed=seed))
                             for d in range(horizon)]) for t in range(trials)]


def audit_streams(root, old_config, cfg):
    new = streams(root["state_id"], root["decision"], cfg["trial_seed"])
    old = streams(root["state_id"], root["decision"], old_config["seed"])
    require(len(set(new)) == cfg["planned_trials"] and not set(new) & set(old), "old/new trial stream collision")
    return dict(state_id=root["state_id"], old_seed=old_config["seed"], new_seed=cfg["trial_seed"],
                old_streams=old, new_streams=new, overlap=0,
                candidates_share_stream_within_trial=True)


def prefix_event(path, root):
    """Check saved trace coverage, not native replay or future branch outcomes."""
    target = root["source"]["decision"]
    require(type(target) is int and target >= 0, "invalid prefix decision")
    with path.open(encoding="utf8") as source:
        for decision, line in enumerate(source):
            event = json.loads(line)
            require(event["decision"] == decision, "saved prefix decision gap")
            if decision == target:
                require(event == root["control_event"], "saved pool/action event changed")
                return decision + 1
    raise ValueError("missing saved root occurrence")


def build():
    cfg, digest = historical.json_snapshot(ROOT / CONFIG)
    check_config(cfg)
    inputs = {CONFIG: digest}

    def pin(name, expected=None):
        path = contained_file(ROOT, name, field="remaining preparation input")
        actual = sha256_file(path)
        require(expected is None or actual == expected, "input changed: " + name)
        require(name not in inputs or inputs[name] == actual, "conflicting input: " + name)
        inputs[name] = actual
        return path

    def frozen_plan(name, digest):
        result = read_json(pin(name, digest))
        require(result["binding"] == json_fingerprint({k:v for k,v in result.items() if k != "binding"}),
                "source plan binding")
        return result

    coverage = frozen_plan(cfg["coverage_plan"], cfg["coverage_plan_sha256"])
    for name, sha in coverage["inputs"].items():
        pin(name, sha)
    matched = frozen_plan(cfg["matched_plan"], cfg["matched_plan_sha256"])
    prior = frozen_plan(cfg["historical_preparation"], cfg["historical_preparation_sha256"])
    require(matched["preparation_binding"] == prior["binding"] and matched["roots"] == prior["roots"],
            "historical exclusion cohort changed")
    require(matched["config"]["source"] == cfg["source"], "original trajectory source changed")
    for name in OWN_INPUTS:
        pin(name)
    sources = {}
    for entry in coverage["old_sources"]:
        origin = entry["directory"]
        source = frozen_plan(origin + "/plan.json", inputs[origin + "/plan.json"])
        require(source["binding"] == entry["binding"] and source["config"] == entry["config"], "old source identity")
        for sid in entry["ids"]:
            require(sid not in sources, "ambiguous source root")
            sources[sid] = entry
    data = read_json(pin(coverage["old_index"], inputs[coverage["old_index"]]))
    require(set(sources) == {s["state_id"] for s in data["states"]}, "source/index root coverage")
    records, roots = [], {}
    for state in data["states"]:
        sid = state["state_id"]
        name = sources[sid]["directory"] + "/roots/" + sid + "/root.json"
        root = read_json(pin(name, inputs[name]))
        historical.check_root_identity(state, root)
        require(root["binding"] == sources[sid]["binding"], "root source binding")
        for candidate in root["control_event"]["pool"]:
            require(set(candidate["agents"]) <= set(state["agent_ids"]), "unknown candidate agent")
        records.append({k:state[k] for k in ("state_id", "map_id", "episode", "decision", "anchor_id")} |
                       dict(pool=root["control_event"]["pool"]))
        roots[sid] = (name, root)

    excluded = [r["state_id"] for r in matched["roots"]]
    selected = select_complement(records, excluded, cfg)
    registration_name = cfg["source"] + "/registration.json"
    reg = read_json(pin(registration_name, inputs[registration_name]))
    cases = {c["task_id"]:c for c in reg["cases"]}
    stream_audit, source_audit = [], []
    for row in selected:
        name, root = roots[row["state_id"]]
        row.update(source_root=name, source_binding=root["binding"], root_fingerprint=root["state_fingerprint"])
        episode = cfg["source"] + "/episodes/" + row["episode"]
        initial = pin(episode + "/initial.json")
        trace = pin(episode + "/first_phase/trace.jsonl")
        lines = prefix_event(trace, root)
        case = cases[root["source"]["item"]["task_id"]]
        require(case["map_id"] == row["map_id"], "case/root map identity")
        for file in case["files"].values():
            pin(file, reg["inputs"][file])
        stream_audit.append(audit_streams(row, sources[row["state_id"]]["config"], cfg))
        source_audit.append(dict(state_id=row["state_id"], checked_trace_lines=lines,
                                 initial_bytes=initial.stat().st_size, trace_bytes=trace.stat().st_size))
    pin(NATIVE, NATIVE_SHA)
    jobs, phases = planned_batches(selected, cfg)
    budget = historical.work_budget(selected, cfg)
    budget["all_20_worker_ideal_budget_seconds"] = budget.pop("ideal_parallel_budget_seconds")
    budget.update(batch_timeout_budget_seconds=sum(len(p["batches"]) for p in phases) * cfg["planned_trial_timeout_seconds"],
                  maximum_total_repairs=budget["maximum_prefix_replay_repairs"] + budget["maximum_rollout_repairs"],
                  includes_reset_prefix_and_rollout=True, scheduler_overhead_in_budget=False)
    require(len(jobs) == budget["trial_jobs"] == 496, "pair-only job budget mismatch")
    # Byte-size estimates are historical storage metadata, never label-based sampling.
    old_dir = Path(cfg["matched_plan"]).parent.as_posix()
    sizes = sorted(pin(old_dir + "/trials/" + j["job_id"] + "/result.json").stat().st_size for j in schedule(matched))
    p95 = sizes[math.ceil(.95 * len(sizes)) - 1]
    disk = dict(reference_results=len(sizes), reference_p95_result_bytes=p95,
                reference_max_result_bytes=max(sizes), estimated_results_bytes_at_reference_p95=496*p95,
                estimated_results_bytes_at_reference_max=496*max(sizes),
                hard_upper_bound=False, excludes_roots_metadata_and_temporary_files=True)
    historical.verify_inputs(inputs)
    plan = dict(schema=cfg["schema"], config=cfg, inputs=inputs, roots=selected, excluded_state_ids=sorted(excluded),
                budget=budget, phases=phases, native_file=NATIVE, native_sha256=NATIVE_SHA,
                source_audit=source_audit, stream_audit=stream_audit, disk_estimate=disk,
                coverage=dict(by_map=dict(sorted(Counter(r["map_id"] for r in selected).items())),
                              by_family=dict(sorted(Counter(r["family"] for r in selected).items())),
                              by_decision=dict(sorted(Counter(str(r["decision"]) for r in selected).items()))),
                collection_ready=False, solver_calls=0, model_fits=0, selection_uses_outcomes=False,
                native_replay_performed=False, independent_confirmation=False,
                blockers=["Pair-only staged collector adapter not registered; old collector requires 376 jobs and anchors",
                          "Native prefix/pool replay must pass before counting any new branch label"],
                old_model_and_runtime_unchanged=True)
    plan["binding"] = json_fingerprint(plan)
    return plan


def compact(plan):
    return {k:plan[k] for k in ("binding", "coverage", "budget", "disk_estimate", "collection_ready", "solver_calls",
                                "model_fits", "native_replay_performed", "blockers")} | dict(
        phases=[{k:v for k,v in p.items() if k != "batches"} | dict(batch_count=len(p["batches"])) for p in plan["phases"]],
        registered_inputs=len(plan["inputs"]))


def run(phase):
    plan = build()
    out = ROOT / plan["config"]["output"]
    jobs, _ = planned_batches(plan["roots"], plan["config"])
    manifest = [dict(job_id=j["job_id"], state_id=j["root"]["state_id"], candidate_id=j["candidate_id"], trial=j["trial"],
                     stream=next(r for r in plan["stream_audit"] if r["state_id"]==j["root"]["state_id"])["new_streams"][j["trial"]])
                for j in jobs]
    files = {"plan.json":plan, "job_manifest.json":manifest, "report.json":compact(plan)}
    for name, value in files.items():
        if phase == "prepare":
            once(out / name, value)
        else:
            require(read_json(out / name) == value, "preparation output changed: " + name)
    return compact(plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "dry-run"))
    args = parser.parse_args()
    print(json.dumps(run(args.phase), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
