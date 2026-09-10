"""Three bounded diagnostic rounds. No training, production mutation, or TTF."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, read_jsonl, sha256_file, write_json
from experiments.closed_loop_trace_storage import apply_state_delta, iter_trace_events, read_state_blob
from experiments.feedback_exploration_diagnostic import FeedbackSelection, padded_auc, paired_summary, classify_rounds
from experiments.repair_collection import _CollectionRunLock, _make_environment, _plain, _run_jobs, state_fingerprint
from scripts.audit_feedback_memory import BASE, SOURCE_SHA, checked, condition_keys, digest
from scripts.run_warehouse_repair_confirmation import environment_config, install_native, task_plan
from scripts.warehouse_repair_confirmation_runtime import verify_prepared

OUT = ROOT / "build/feedback-exploration-diagnostics-v1"
HORIZON, TRIALS, WORKERS = 32, 4, 20
PP_SECONDS, EPISODE_SECONDS, FUSE_SECONDS, TOTAL_SECONDS = 5.0, 120, 180, 5400
FILES = ["scripts/run_feedback_exploration_diagnostics.py", "experiments/feedback_exploration_diagnostic.py",
         "scripts/audit_feedback_memory.py", "docs/FEEDBACK_EXPLORATION_PROTOCOL_ZH.md"]


def seed(*parts):
    return int(digest(["feedback-exploration-v1", *parts])[:8], 16) & 0x7fffffff


def paths(state):
    return [(a["id"], a["path"]) for a in sorted(state["agents"], key=lambda a: a["id"])]


def repair_keys(state, candidates, instance):
    return [condition_keys(state, c["agents"], PP_SECONDS, {"mode": "explicit_neighborhood"}, instance)[
        "repair_problem_budget"] for c in candidates]


def snapshot(spec, decision=None):
    item = spec["item"]
    directory = BASE / "timed" / item["job_id"] / "first_phase"
    result = read_json(directory.parent / "first_phase_result.json")["payload"]
    trace = directory / result["trace_file"]
    checked(sha256_file(trace) == result["trace_sha256"], "trace SHA mismatch")
    initial, state, prefix = None, None, []
    for event in iter_trace_events(trace):
        if event["event"] == "initial":
            initial = state = read_state_blob(directory / event["state_blob"])
            checked(state_fingerprint(state) == event["state_fingerprint"], "initial identity")
        elif event["event"] == "transition":
            checked(state_fingerprint(state) == event["before_fingerprint"], "prefix before identity")
            if (decision is None and 0 < state["num_of_colliding_pairs"] <= 3) or event["decision_index"] == decision:
                return {"case_id": item["checkpoint_id"], "map_id": item["map_id"],
                        "instance": item["checkpoint_identity_sha256"], "task_id": item["task_id"],
                        "solver_seed": spec["worker_job"]["solver_seed"], "proposal": spec["worker_job"]["proposal"],
                        "restore_seed": spec["worker_job"]["episode_override"]["initial_restore"]["restore_seed"],
                        "initial": initial, "state": state, "prefix": prefix,
                        "decision": event["decision_index"], "candidates": event["controller"]["candidate_pool"],
                        "selected_id": event["controller"]["selected_candidate_id"],
                        "trace": trace.relative_to(ROOT).as_posix(), "trace_sha256": result["trace_sha256"]}
            after = apply_state_delta(state, event["state_delta"])
            checked(state_fingerprint(after) == event["after_fingerprint"], "prefix after identity")
            prefix.append({k: event[k] for k in ("action", "metrics", "before_fingerprint", "after_fingerprint")})
            state = after
    if decision is None:
        return None
    raise ValueError("required nonterminal source unavailable; no substitution")


def prepare():
    checked(not (OUT / "plan.json").exists(), "plan already exists")
    config, specs = verify_prepared(require_ready=True)
    equivalence = ROOT / "build/feedback-native-equivalence-v1/report.json"
    checked(sha256_file(equivalence) == "d460bf29c41af271f04fd12f537293c4d8f8fe21becad8ce439489e52b2b2563",
            "native equivalence evidence changed")
    opportunity = read_json(BASE / "opportunity/plan.json")
    checked(sha256_file(BASE / "opportunity/plan.json") == "5f151f4c3fae8e8a67e7eec509731a017f2036015751532aa3da0de3392cc417",
            "discovery source selection changed")
    dual = {s["item"]["checkpoint_id"]: s for s in specs if s["item"]["controller"] == "dual16"}
    discovery = [snapshot(dual[c["checkpoint"]], c["decision"]) for c in opportunity["cases"]]
    maps = {c["map_id"] for c in discovery}
    validation, per_map, unavailable = [], Counter(), []
    for spec in sorted(dual.values(), key=lambda s: s["item"]["checkpoint_id"]):
        map_id = spec["item"]["map_id"]
        if map_id in maps or per_map[map_id] >= 2:
            continue
        source = snapshot(spec)
        if source is None:
            unavailable.append(spec["item"]["checkpoint_id"])
            continue
        validation.append(source)
        per_map[map_id] += 1
    checked(len(discovery) == 5 and len(validation) == 6 and len(per_map) == 3,
            "fixed 5/6 state supply or map isolation failed")
    plan = {"schema": "lns2.feedback_exploration_diagnostics.v1", "config": config,
            "discovery": discovery, "validation": validation, "unavailable_low_conflict_sources": unavailable,
            "horizon": HORIZON, "trials": TRIALS,
            "arms": list(FeedbackSelection.ARMS), "workers": WORKERS, "pp_seconds": PP_SECONDS,
            "episode_seconds": EPISODE_SECONDS, "job_fuse_seconds": FUSE_SECONDS, "total_seconds": TOTAL_SECONDS,
            "source_sha256": SOURCE_SHA, "files": {p: sha256_file(ROOT / p) for p in FILES},
            "round1_episodes": 60, "round2_max_states": 2, "round2_max_jobs": 160, "round3_episodes": 72,
            "no_ttf": True, "no_training": True, "no_parameter_search": True}
    write_json(OUT / "plan.json", plan)
    return {"discovery": [(c["case_id"], c["decision"]) for c in discovery],
            "validation": [(c["case_id"], c["decision"]) for c in validation],
            "round1_episodes": 60, "round3_episodes": 72, "round2_max_jobs": 160,
            "maximum_new_repairs": 60 * HORIZON + 72 * HORIZON + 160,
            "plan_sha256": sha256_file(OUT / "plan.json")}


def verify():
    plan = read_json(OUT / "plan.json")
    for p, expected in plan["files"].items():
        checked(sha256_file(ROOT / p) == expected, "diagnostic implementation changed: " + p)
    config, _ = verify_prepared(require_ready=True)
    checked(config == plan["config"] and sha256_file(BASE / "timed/confirmation_report.json") == SOURCE_SHA,
            "frozen source changed")
    for case in plan["discovery"] + plan["validation"]:
        checked(sha256_file(ROOT / case["trace"]) == case["trace_sha256"], "source trace changed")
    return plan


class FrozenPool:
    def __init__(self, case):
        from experiments.compact_controller_model import load_controller_bundle
        self.case = case
        self.model = load_controller_bundle(ROOT / "artifacts/initlns-closed-loop-controller-v2").main_models["realized_dynamic"]
        checked(self.model.inference_backend == "native-portable-tree", "native scorer required")

    def select(self, env, state, decision):
        from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
        from lns2_selector.runtime.online_selection import generate_online_candidates, score_online_candidates
        from lns2_selector.runtime.structshell_dual16 import structshell_dual16_augmentation, generate_structshell_dual16_runtime_candidates
        before = state_fingerprint(state)
        engine = OnlineFeatureEngine(state, backend="native", dense_output=True,
                                     required_features={"realized_dynamic": set(self.model.base_feature_names)})
        candidates, _ = generate_online_candidates(env, state, task_id=self.case["task_id"],
            solver_seed=self.case["solver_seed"], decision_index=decision, proposal_config=self.case["proposal"],
            state_hash=before, verify_full_state=True, proposal_backend="optimized")
        topology = TopologyAnalysisCache(state, static_grid=engine.static_grid, backend="native")
        candidates = list(generate_structshell_dual16_runtime_candidates(state, topology.analysis,
            v2_candidates=candidates, config=structshell_dual16_augmentation()).candidates)
        features, _ = engine.realized_rows(candidates, state_hash=before)
        index, scores, _ = score_online_candidates(features, self.model)
        checked(state_fingerprint(env.get_state()) == before, "proposal changed state")
        pool = [{**c, "score": float(s)} for c, s in zip(candidates, scores)]
        return int(index), pool


def check_source_pool(case, index, pool):
    old = {c["candidate_id"]: c for c in case["candidates"]}
    checked(set(old) == {c["candidate_id"] for c in pool}, "source pool IDs changed")
    for c in pool:
        previous = old[c["candidate_id"]]
        checked(sorted(c["agents"]) == sorted(previous["agents"]) and
                sorted(c["selection_families"]) == sorted(previous["selection_families"]) and
                math.isclose(c["score"], previous["score"], rel_tol=0, abs_tol=1e-10), "source pool/scorer changed")
    checked(pool[index]["candidate_id"] == case["selected_id"], "source winner changed")


def restore(job):
    case, config = job["case"], job["config"]
    install_native(config)
    task = next(t["row"] for t in task_plan(config) if t["row"]["task_id"] == case["task_id"])
    env = _make_environment(str(ROOT / config["dataset"]["output"]), task,
                            {**environment_config(config), "time_limit": 600}, "Adaptive")
    state = _plain(env.reset_paths([p for _, p in paths(case["initial"])], seed=case["restore_seed"]))
    checked(state_fingerprint(state) == state_fingerprint(case["initial"]), "initial restore mismatch")
    for event in case["prefix"]:
        checked(state_fingerprint(state) == event["before_fingerprint"], "prefix before mismatch")
        step = _plain(env.step_with_time_limit(event["action"], event["metrics"]["requested_pp_time_limit_seconds"]))
        state = step["observation"]
        checked(state_fingerprint(state) == event["after_fingerprint"] and
                step["metrics"]["repair_order"] == event["metrics"]["repair_order"], "prefix replay mismatch")
    checked(state_fingerprint(state) == state_fingerprint(case["state"]), "source snapshot mismatch")
    return env, state


def validate_step(before, step, agents):
    metrics, after = step["metrics"], step["observation"]
    checked(metrics["action_valid"] and metrics["step_applied"] and
            sorted(metrics["neighborhood"]) == sorted(agents), "invalid/modified action")
    selected = set(agents)
    old = dict(paths(before))
    for agent, path in paths(after):
        checked(agent in selected or old[agent] == path, "external path modified")
    if metrics["pp_rolled_back"]:
        checked(paths(before) == paths(after), "rollback mismatch")
    checked(after["num_of_colliding_pairs"] <= before["num_of_colliding_pairs"], "conflicts increased")
    return metrics["pp_failure_reason"] not in ("none", "conflict_bound_exceeded") or metrics["native_replan_seconds"] >= PP_SECONDS


def validate_final(state):
    from experiments.state_analysis import summarize_initial_state_complexity
    summarize_initial_state_complexity(state)
    cols = state["cols"]
    checked(len(state["obstacles"]) == state["rows"] * cols, "invalid obstacle dimensions")
    for a in state["agents"]:
        path = a["path"]
        checked(path[0] == a["start"] and path[-1] == a["goal"], "invalid path endpoints")
        checked(all(0 <= p < len(state["obstacles"]) and not state["obstacles"][p] for p in path), "invalid path cell")
        checked(all(abs(x // cols - y // cols) + abs(x % cols - y % cols) <= 1
                    for x, y in zip(path, path[1:])), "invalid path move")
    checked(state["feasible"] == (state["num_of_colliding_pairs"] == 0), "invalid feasibility")


def rollout_worker(job):
    env, state = restore(job)
    case = job["case"]
    selector = FrozenPool(case)
    base, pool = selector.select(env, state, case["decision"])
    check_source_pool(case, base, pool)
    feedback = FeedbackSelection(job["arm"])
    transitions, conflicts = [], [state["num_of_colliding_pairs"]]
    began = time.monotonic()
    stop, censored = "step_limit", False
    for decision in range(job.get("horizon", HORIZON)):
        if state["feasible"]:
            stop = "feasible"
            break
        if time.monotonic() - began >= EPISODE_SECONDS - PP_SECONDS:
            stop, censored = "resource_budget", True
            break
        if decision:
            base, pool = selector.select(env, state, case["decision"] + decision)
        keys = repair_keys(state, pool, case["instance"])
        chosen, diagnostic = feedback.select(keys, [c["candidate_id"] for c in pool], base,
                                             seed(job["phase"], case["case_id"], job["trial"], decision, "selection"))
        candidate = pool[chosen]
        action = {"mode": "explicit_neighborhood", "agents": candidate["agents"],
                  "random_seed": seed(job["phase"], case["case_id"], job["trial"], decision, "pp")}
        before = state
        step = _plain(env.step_with_time_limit(action, PP_SECONDS))
        native_censor = validate_step(before, step, candidate["agents"])
        state = step["observation"]
        feedback.observe(keys[chosen], conflicts[-1], state["num_of_colliding_pairs"], native_censor)
        event = {"action": action, "metrics": step["metrics"], "before_fingerprint": state_fingerprint(before),
                 "after_fingerprint": state_fingerprint(state), "candidate_id": candidate["candidate_id"],
                 "base_id": pool[base]["candidate_id"], "candidate_pool": pool, "keys": keys,
                 "feedback": diagnostic, "selected_key": keys[chosen]}
        transitions.append(event)
        conflicts.append(state["num_of_colliding_pairs"])
        if native_censor:
            stop, censored = "native_censor", True
            break
    if state["feasible"]:
        stop = "feasible"
    validate_final(state)
    result = {"status": "ok", "job_id": job["job_id"], "plan_sha256": job["plan_sha256"],
              "case_id": case["case_id"], "map_id": case["map_id"], "trial": job["trial"], "arm": job["arm"],
              "phase": job["phase"], "initial_fingerprint": state_fingerprint(case["state"]),
              "conflicts": conflicts, "feasible": state["feasible"], "censored": censored, "stop": stop,
              "auc": padded_auc(conflicts, job.get("horizon", HORIZON)), "transitions": transitions, "final_state": state}
    return result


def probe_worker(job):
    env, state = restore(job)
    candidate = job["candidate"]
    key = repair_keys(state, [candidate], job["case"]["instance"])[0]
    checked(key == job["repair_key"], "probe repair condition changed")
    action = {"mode": "explicit_neighborhood", "agents": candidate["agents"],
              "random_seed": seed("probe", job["case"]["case_id"], key, job["trial"])}
    step = _plain(env.step_with_time_limit(action, PP_SECONDS))
    censored = validate_step(state, step, candidate["agents"])
    validate_final(step["observation"])
    return {"status": "ok", "job_id": job["job_id"], "plan_sha256": job["plan_sha256"],
            "case_id": job["case"]["case_id"], "repair_key": key, "trial": job["trial"],
            "candidate_id": candidate["candidate_id"], "agents": candidate["agents"],
            "before_conflicts": state["num_of_colliding_pairs"],
            "after_conflicts": step["observation"]["num_of_colliding_pairs"],
            "censored": censored, "metrics": step["metrics"]}


def stage_jobs(plan, cases, phase, arms=FeedbackSelection.ARMS, trials=TRIALS, horizon=HORIZON):
    return [{"job_id": f"{c['case_id']}-{arm}-{trial}", "phase": phase, "arm": arm,
             "trial": trial, "case": c, "config": plan["config"], "horizon": horizon,
             "plan_sha256": sha256_file(OUT / "plan.json")}
            for c in cases for trial in range(trials) for arm in arms]


def load_result(path, job):
    result = read_json(path)
    checked(result["status"] == "ok" and result["job_id"] == job["job_id"] and
            result["plan_sha256"] == job["plan_sha256"], "result error/identity mismatch")
    if "arm" in job:
        checked(all(result[k] == job[k] for k in ("arm", "trial", "phase")) and
                result["case_id"] == job["case"]["case_id"] and
                result["initial_fingerprint"] == state_fingerprint(job["case"]["state"]), "rollout pairing mismatch")
        checked(result["feasible"] == (result["conflicts"][-1] == 0) and
                result["auc"] == padded_auc(result["conflicts"], job["horizon"]), "rollout summary mismatch")
        fingerprint = result["initial_fingerprint"]
        checked(len(result["transitions"]) + 1 == len(result["conflicts"]), "transition count mismatch")
        for event in result["transitions"]:
            checked(event["before_fingerprint"] == fingerprint, "broken result fingerprint chain")
            fingerprint = event["after_fingerprint"]
        checked(fingerprint == state_fingerprint(result["final_state"]), "final fingerprint mismatch")
    else:
        checked(all(result[k] == job[k] for k in ("trial", "repair_key")) and
                result["candidate_id"] == job["candidate"]["candidate_id"] and
                result["agents"] == job["candidate"]["agents"], "probe identity mismatch")
    return result


def collect_stage(phase, jobs, worker, began):
    directory = OUT / phase
    schedule_path = directory / "schedule.json"
    if schedule_path.exists():
        checked(read_json(schedule_path) == jobs, "stage schedule changed")
    else:
        write_json(schedule_path, jobs)
    pending = [j for j in jobs if not (directory / "outcomes" / (j["job_id"] + ".json")).exists()]
    for j in jobs:
        p = directory / "outcomes" / (j["job_id"] + ".json")
        if p.exists():
            load_result(p, j)
    completed = len(jobs) - len(pending)
    def save(result):
        nonlocal completed
        write_json(directory / "outcomes" / (result["job_id"] + ".json"), result)
        completed += 1
        write_json(OUT / "run_status.json", {"phase": phase, "status": "running", "completed": completed,
                  "stage_total": len(jobs), "elapsed_seconds": time.monotonic() - began})
        print(f"{phase} {completed}/{len(jobs)} {result['status']}", flush=True)
    def failure(job, status, error):
        return {"status": status, "error": error, "job_id": job["job_id"], "plan_sha256": job["plan_sha256"]}
    for offset in range(0, len(pending), WORKERS):
        if (OUT / "STOP").exists() or time.monotonic() - began >= TOTAL_SECONDS:
            raise InterruptedError("safe stop or total resource budget")
        _run_jobs(worker, pending[offset:offset + WORKERS], WORKERS, phase=phase, output_root=directory,
                  run_fingerprint=sha256_file(OUT / "plan.json"), timeout_seconds=FUSE_SECONDS,
                  on_result=save, failure_result=failure, stop_on_failure=True)
    return [load_result(directory / "outcomes" / (j["job_id"] + ".json"), j) for j in jobs]


def screen_sources(plan, round1):
    """Select at most two unresolved successor states with unmeasured repair keys."""
    known = set()
    for c in plan["discovery"]:
        known.update(repair_keys(c["state"], c["candidates"], c["instance"]))
    for r in round1:
        known.update(e["selected_key"] for e in r["transitions"])
    selected, used = [], set()
    ranking = {"feedback": 0, "uniform": 1, "frozen": 2}
    for r in sorted(round1, key=lambda r: (ranking[r["arm"]], r["case_id"], r["trial"])):
        if r["feasible"] or r["censored"] or r["case_id"] in used:
            continue
        used.add(r["case_id"])
        base = next(c for c in plan["discovery"] if c["case_id"] == r["case_id"])
        case = {**base, "case_id": base["case_id"] + "-successor", "state": r["final_state"],
                "prefix": base["prefix"] + [{k: e[k] for k in ("action", "metrics", "before_fingerprint", "after_fingerprint")}
                                             for e in r["transitions"]],
                "decision": base["decision"] + len(r["transitions"])}
        # Generate only the successor's current pool, never search to choose the probe.
        env, state = restore({"case": case, "config": plan["config"]})
        index, pool = FrozenPool(case).select(env, state, case["decision"])
        keys = repair_keys(state, pool, case["instance"])
        eligible = [(c, k) for c, k in zip(pool, keys) if k not in known]
        if eligible:
            selected.append({"case": case, "eligible": eligible, "source_job": r["job_id"],
                             "current_pool_count": len(pool), "base_id": pool[index]["candidate_id"]})
            known.update(k for _, k in eligible)
        if len(selected) == 2:
            break
    return selected


def run(smoke=False):
    plan = verify()
    status_path = OUT / "run_status.json"
    prior_elapsed = read_json(status_path).get("elapsed_seconds", 0) if status_path.exists() and not smoke else 0
    began = time.monotonic() - prior_elapsed
    with _CollectionRunLock(OUT, sha256_file(OUT / "plan.json"), "feedback-exploration"):
        try:
            if smoke:
                jobs = stage_jobs(plan, plan["discovery"], "smoke", arms=("frozen",), trials=1, horizon=2)
                rows = collect_stage("smoke", jobs, rollout_worker, began)
                write_json(status_path, {"status": "smoke_complete", "elapsed_seconds": 0})
                return {"status": "smoke_pass", "source_pools_verified": len(rows)}
            smoke_jobs = stage_jobs(plan, plan["discovery"], "smoke", arms=("frozen",), trials=1, horizon=2)
            for j in smoke_jobs:
                load_result(OUT / "smoke/outcomes" / (j["job_id"] + ".json"), j)
            rows1 = collect_stage("round1", stage_jobs(plan, plan["discovery"], "round1"), rollout_worker, began)
            summary1 = paired_summary(rows1)
            write_json(OUT / "round1/report.json", summary1)
            checked(sum(r["censored"] for r in rows1) < len(rows1) / 4, "too many resource-censored episodes; no expansion")
            supply_path = OUT / "round2/sources.json"
            if supply_path.exists():
                sources = read_json(supply_path)
            else:
                sources = screen_sources(plan, rows1)
                write_json(supply_path, sources)
            jobs2 = [{"job_id": f"{s['case']['case_id']}-{c['candidate_id']}-{t}", "case": s["case"],
                      "candidate": c, "repair_key": k, "trial": t, "config": plan["config"],
                      "plan_sha256": sha256_file(OUT / "plan.json")}
                     for s in sources for c, k in s["eligible"] for t in range(TRIALS)]
            checked(len(jobs2) <= 160, "round2 budget exceeded")
            rows2 = collect_stage("round2", jobs2, probe_worker, began)
            write_json(OUT / "round2/report.json", {"jobs": len(rows2), "sources": len(sources),
                "strict_drop_jobs": sum(r["after_conflicts"] < r["before_conflicts"] for r in rows2),
                "feasible_jobs": sum(r["after_conflicts"] == 0 for r in rows2),
                "censored": sum(r["censored"] for r in rows2), "interpretation": "H1 supply only, no controller promotion"})
            rows3 = collect_stage("round3", stage_jobs(plan, plan["validation"], "round3"), rollout_worker, began)
            summary3 = paired_summary(rows3)
            write_json(OUT / "round3/report.json", summary3)
            report = {"schema": plan["schema"], "plan_sha256": sha256_file(OUT / "plan.json"),
                      "round1": summary1, "round2_jobs": len(rows2), "round3": summary3,
                      "new_ttf_episodes": 0, "no_controller_promotion": True,
                      "decision": classify_rounds(summary1, summary3),
                      "outcome_sha256": {p.relative_to(OUT).as_posix(): sha256_file(p)
                                          for phase in ("round1", "round2", "round3")
                                          for p in sorted((OUT / phase / "outcomes").glob("*.json"))},
                      "claim": "Retrospective bounded sequential diagnostic; no independent TTF or new-map evidence."}
            write_json(OUT / "report.json", report)
            write_json(status_path, {"status": "complete", "elapsed_seconds": time.monotonic() - began,
                                    "rounds": 3, "episodes": len(rows1) + len(rows3), "probe_jobs": len(rows2)})
            return report
        except BaseException as error:
            write_json(status_path, {"status": "paused" if isinstance(error, InterruptedError) else "failed",
                       "error": str(error), "elapsed_seconds": time.monotonic() - began})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "smoke", "run"))
    args = parser.parse_args()
    print(json.dumps(prepare() if args.phase == "prepare" else run(args.phase == "smoke"), indent=2))
