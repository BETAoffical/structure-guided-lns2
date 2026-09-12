"""Serial development TTF comparison, with an explicit stop-after-episode file."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.closed_loop_trace_storage import apply_state_delta, encode_state_delta
from experiments.nonmonotonic_repair import temperature, acceptance_draw, validate_transition
from experiments.repair_collection import _CollectionRunLock, _make_environment, _plain, _run_jobs, state_fingerprint
from scripts.audit_feedback_memory import digest
from scripts.diagnose_nonmonotonic_repair import seed
from scripts.run_feedback_exploration_diagnostics import FrozenPool, validate_final
from scripts.run_warehouse_repair_confirmation import install_native, environment_config, task_plan

OUT = ROOT / "build/sa-wall-clock-pilot-v1"
SOURCE = ROOT / "build/uniform-runtime-official-probe-v1/plan.json"
SOURCE_SHA = "4a448262e641f394c090250c5bf95d714d32bd1adb32843e804ea271a25f2015"
NATIVE = "build/linux/sa-wall-clock-v1/lns2_env.cpython-310-x86_64-linux-gnu.so"
ARMS = ("official", "official_sa", "dual16", "dual16_sa")
ORDERS = ((0, 1, 3, 2), (1, 2, 0, 3), (2, 3, 1, 0), (3, 0, 2, 1))
BUDGET, FUSE = 60.0, 180.0


class TimedPool(FrozenPool):
    """Reuse the production static/prepared caches; retain frozen actions."""

    def __init__(self, case):
        super().__init__(case)
        self.engine = self.topology = None

    def select(self, env, state, decision):
        from experiments.online_feature_engine import OnlineFeatureEngine, TopologyAnalysisCache
        from lns2_selector.runtime.online_selection import generate_online_candidates, score_online_candidates
        from lns2_selector.runtime.structshell_dual16 import structshell_dual16_augmentation, generate_structshell_dual16_runtime_candidates
        before = state_fingerprint(state)
        if self.engine is None:
            self.engine = OnlineFeatureEngine(state, backend="native", dense_output=True,
                required_features={"realized_dynamic": set(self.model.base_feature_names)})
        if self.topology is None:
            self.topology = TopologyAnalysisCache(state, static_grid=self.engine.static_grid, backend="native")
        else:
            self.topology.prepare(state)
        candidates, _ = generate_online_candidates(env, state, task_id=self.case["task_id"],
            solver_seed=self.case["solver_seed"], decision_index=decision,
            proposal_config=self.case["proposal"], state_hash=before,
            verify_full_state=True, proposal_backend="optimized")
        pool = list(generate_structshell_dual16_runtime_candidates(state, self.topology.analysis,
            v2_candidates=candidates, config=structshell_dual16_augmentation()).candidates)
        self.engine.prepare(state, prepared_native_analysis=self.topology.last_native_prepared)
        features, _ = self.engine.realized_rows(pool, state_hash=before)
        index, scores, _ = score_online_candidates(features, self.model)
        if state_fingerprint(env.get_state()) != before:
            raise ValueError("proposal mutated state")
        return int(index), [{**c, "score": float(s)} for c, s in zip(pool, scores)]


def prepare():
    if OUT.exists():
        raise ValueError("output already exists; do not overwrite registration")
    if sha256_file(SOURCE) != SOURCE_SHA:
        raise ValueError("source identity changed")
    source = read_json(SOURCE)
    config = deepcopy(source["config"])
    legacy_config = deepcopy(config)
    config["frozen"].update(native_path=NATIVE, native_sha256=sha256_file(ROOT / NATIVE))
    tasks = {t["row"]["task_id"]: t["row"] for t in task_plan(config)}
    inputs = {SOURCE.relative_to(ROOT).as_posix(): SOURCE_SHA,
              NATIVE: sha256_file(ROOT / NATIVE),
              legacy_config["frozen"]["native_path"]: legacy_config["frozen"]["native_sha256"]}
    cases = []
    for old in sorted(source["cases"], key=lambda c: c["case_id"]):
        c = {k: old[k] for k in ("case_id", "map_id", "task_id", "solver_seed", "proposal")}
        c["row"] = tasks[c["task_id"]]
        c["expected_initial_fingerprint"] = state_fingerprint(old["raw_initial"])
        c["expected_endpoints"] = old["expected_endpoints"]
        cases.append(c)
        for key in ("map_file", "map_metadata_file", "task_file", "scenario_file"):
            p = ROOT / config["dataset"]["output"] / c["row"]["split"] / c["row"][key]
            inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    if len(cases) != 16 or len({c["map_id"] for c in cases}) != 8:
        raise ValueError("fixed cohort mismatch")
    for folder in ("src", "include", "third_party/mapf_lns2", "lns2_selector", "experiments", "scripts",
                   "artifacts/initlns-closed-loop-controller-v2"):
        for p in (ROOT / folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    for name in ("docs/SA_WALL_CLOCK_PROTOCOL_ZH.md", "CMakeLists.txt"):
        inputs[name] = sha256_file(ROOT / name)
    plan = dict(schema="lns2.sa_wall_clock.v1", config=config, legacy_config=legacy_config,
        cases=cases, inputs=inputs, arms=list(ARMS), orders=ORDERS, budget=BUDGET, fuse=FUSE,
        max_repair_iterations=0, workers=1, bootstrap_samples=5000, bootstrap_seed=20260912,
        initial_temperature=1000, cooling=.99, development_only=True,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    write_json(OUT / "plan.json", plan)
    return {"cases": 16, "maps": 8, "timed_jobs": 64, "budget_seconds": BUDGET,
            "serial_budget_upper_seconds": 64 * BUDGET, "includes_old_development_tasks": True}


def verify():
    p = read_json(OUT / "plan.json")
    if (p["arms"], p["budget"], p["fuse"], p["max_repair_iterations"], p["workers"]) != (list(ARMS), BUDGET, FUSE, 0, 1):
        raise ValueError("protocol changed")
    for name, sha in p["inputs"].items():
        if sha256_file(ROOT / name) != sha:
            raise ValueError("frozen file changed: " + name)
    return p


def jobs(plan, phase):
    sha = sha256_file(OUT / "plan.json")
    result = []
    for i, case in enumerate(plan["cases"]):
        arms = ("admission",) if phase == "admission" else tuple(ARMS[k] for k in ORDERS[i % 4])
        if phase in ("smoke", "parity") and i >= 2:
            continue
        if phase == "parity":
            for native in ("legacy", "current"):
                for arm in ("official", "dual16"):
                    result.append(dict(job_id=case["case_id"] + "-" + arm + "-" + native,
                        case=case, config=plan["legacy_config"] if native == "legacy" else plan["config"],
                        plan_sha256=sha, arm=arm, budget=BUDGET, phase=phase, max_steps=3))
            continue
        for arm in arms:
            result.append(dict(job_id=case["case_id"] + "-" + arm, case=case, config=plan["config"],
                plan_sha256=sha, arm=arm, budget=BUDGET, phase=phase,
                max_steps=3 if phase == "smoke" else None))
    return result


def make_env(job):
    install_native(job["config"])
    return _make_environment(str(ROOT / job["config"]["dataset"]["output"]), job["case"]["row"],
        {**environment_config(job["config"]), "time_limit": job["budget"]}, "Adaptive")


def action_for(arm, case, decision, pool, index):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if arm.startswith("official"):
        return {"mode": "official"}
    return {"mode": "explicit_neighborhood", "agents": pool[index]["agents"],
            "random_seed": seed(case["case_id"], 0, decision, "pp")}


def capped_time(row, budget=BUDGET):
    t = row["ttf_seconds"]
    if t is not None and (not math.isfinite(t) or t < 0):
        raise ValueError("invalid TTF")
    return min(t, budget) if t is not None else budget


def worker(job):
    cold = time.monotonic()
    case, arm = job["case"], job["arm"]
    env = make_env(job)
    selector = TimedPool(case) if arm.startswith("dual16") else None
    setup = time.monotonic() - cold
    start = time.monotonic()
    state = _plain(env.reset(seed=case["solver_seed"]))
    reset_seconds = time.monotonic() - start
    initial = state
    if not state["initial_solution_complete"]:
        raise ValueError("incomplete reset; cannot compare registered initial state")
    if state_fingerprint(state) != case["expected_initial_fingerprint"]:
        raise ValueError("raw PP initial fingerprint mismatch")
    if [[a["path"][0], a["path"][-1]] for a in sorted(state["agents"], key=lambda a: a["id"])] != case["expected_endpoints"]:
        raise ValueError("task endpoints mismatch")
    if arm == "admission":
        validate_final(state)
        return dict(status="ok", job_id=job["job_id"], plan_sha256=job["plan_sha256"],
            initial_fingerprint=state_fingerprint(state), conflicts=state["num_of_colliding_pairs"])
    ttf = reset_seconds if state["feasible"] else None
    events, stop = [], "feasible" if state["feasible"] else "deadline"
    while not state["feasible"]:
        remaining = job["budget"] - (time.monotonic() - start)
        if remaining <= 0:
            break
        d = len(events)
        if job["max_steps"] is not None and d >= job["max_steps"]:
            stop = "smoke_step_limit"
            break
        select_start = time.monotonic()
        index, pool = selector.select(env, state, d) if selector else (None, [])
        if job["phase"] == "parity" and selector:
            reference_index, reference_pool = FrozenPool(case).select(env, state, d)
            if index != reference_index or pool != reference_pool:
                raise ValueError("cached selector differs from frozen diagnostic selector")
        select_seconds = time.monotonic() - select_start
        remaining = job["budget"] - (time.monotonic() - start)
        if remaining <= 0:
            stop = "selection_deadline"
            break
        action = action_for(arm, case, d, pool, index)
        temp, draw = temperature(d), acceptance_draw(seed(case["case_id"], 0, d, "accept"))
        before = state
        step = _plain(env.step_experimental_pp(action, remaining, "annealed", temp, draw)
                      if arm.endswith("_sa") else env.step_with_time_limit(action, remaining))
        elapsed = time.monotonic() - start
        state, m = step["observation"], step["metrics"]
        if state["feasible"]:
            ttf, stop = elapsed, "feasible"
        if not m["action_valid"] or not m["step_applied"]:
            raise ValueError("invalid or unapplied action")
        events.append(dict(action=action, metrics=m, pool=pool, selected_index=index,
            temperature=temp, uniform=draw, delta=encode_state_delta(before, state),
            elapsed_seconds=elapsed, selection_seconds=select_seconds))
        if state["feasible"]:
            break
        if m["pp_failure_reason"] == "time_limit" or step["truncated"]:
            stop = "deadline"
            break
    loop_end = time.monotonic() - start
    row = dict(status="ok", job_id=job["job_id"], plan_sha256=job["plan_sha256"],
        case_id=case["case_id"], map_id=case["map_id"], arm=arm, phase=job["phase"],
        initial_state=initial, final_state=state, events=events, stop_reason=stop,
        setup_seconds=setup, reset_seconds=reset_seconds, ttf_seconds=ttf,
        loop_end_seconds=loop_end, success_within_budget=ttf is not None and ttf <= job["budget"],
        generated=state["low_level"]["generated"] - initial["low_level"]["generated"],
        pp_calls=sum(e["metrics"]["pp_attempted_agent_count"] > 0 for e in events),
        selection_seconds=sum(e["selection_seconds"] for e in events),
        pp_seconds=sum(e["metrics"]["native_replan_seconds"] for e in events))
    # Expensive scientific validation is outside first-feasible TTF for all arms.
    validate_result(row, job)
    row["validated_seconds"] = time.monotonic() - start
    row["cold_worker_seconds"] = time.monotonic() - cold
    return row


def validate_result(row, job):
    if row["status"] != "ok" or any(row[k] != job[k] for k in ("job_id", "plan_sha256", "arm", "phase")):
        raise ValueError("result identity/error")
    state = row["initial_state"]
    if state_fingerprint(state) != job["case"]["expected_initial_fingerprint"]:
        raise ValueError("initial mismatch")
    validate_final(state)
    previous_time = row["reset_seconds"]
    for d, event in enumerate(row["events"]):
        after = apply_state_delta(state, event["delta"])
        if event["elapsed_seconds"] < previous_time or state["feasible"]:
            raise ValueError("time order or post-feasible step")
        pool, idx = event["pool"], event["selected_index"]
        if event["action"] != action_for(row["arm"], job["case"], d, pool, idx):
            raise ValueError("action identity mismatch")
        if row["arm"].startswith("official"):
            if pool or idx is not None:
                raise ValueError("official candidate override")
        elif (not pool or pool[idx]["score"] < max(c["score"] for c in pool)):
            raise ValueError("nonmaximum model selection")
        temp, draw = temperature(d), acceptance_draw(seed(job["case"]["case_id"], 0, d, "accept"))
        if (temp, draw) != (event["temperature"], event["uniform"]):
            raise ValueError("SA random stream mismatch")
        mode = "annealed" if row["arm"].endswith("_sa") else "standard"
        validate_transition(state, after, event["metrics"], event["metrics"]["neighborhood"], mode, temp, draw)
        if pool and sorted(pool[idx]["agents"]) != sorted(event["metrics"]["neighborhood"]):
            raise ValueError("explicit neighborhood modified")
        validate_final(after)
        state, previous_time = after, event["elapsed_seconds"]
    if state_fingerprint(state) != state_fingerprint(row["final_state"]):
        raise ValueError("final state mismatch")
    expected_ttf = previous_time if state["feasible"] else None
    if row["ttf_seconds"] != expected_ttf:
        raise ValueError("TTF not anchored to first feasible transition")
    if row["success_within_budget"] != (expected_ttf is not None and expected_ttf <= job["budget"]):
        raise ValueError("success deadline mismatch")
    if row["generated"] != state["low_level"]["generated"] - row["initial_state"]["low_level"]["generated"]:
        raise ValueError("node mismatch")
    if not state["feasible"] and row["stop_reason"] not in ("deadline", "selection_deadline", "smoke_step_limit"):
        raise ValueError("unexplained termination")
    capped_time(row, job["budget"])


def saved_result(path, job):
    row = read_json(path)
    if row.get("integrity_sha256") != digest({k: v for k, v in row.items() if k != "integrity_sha256"}):
        raise ValueError("result hash mismatch")
    if job["arm"] == "admission":
        if row["status"] != "ok" or row["plan_sha256"] != job["plan_sha256"] or row["initial_fingerprint"] != job["case"]["expected_initial_fingerprint"]:
            raise ValueError("admission mismatch")
    else:
        validate_result(row, job)
    return row


def execute(plan, phase):
    scheduled = jobs(plan, phase)
    if phase == "timed":
        for stage in ("admission", "parity", "smoke"):
            r = read_json(OUT / (stage + "_report.json"))
            if not r["complete"] or r["plan_sha256"] != sha256_file(OUT / "plan.json"):
                raise ValueError("admission/smoke incomplete")
            for job in jobs(plan, stage):
                p = OUT / stage / (job["job_id"] + ".json")
                if sha256_file(p) != r["files"][p.name]:
                    raise ValueError("admission/smoke evidence changed")
                saved_result(p, job)
    directory = OUT / phase
    def save(row):
        row["integrity_sha256"] = digest(row)
        write_json(directory / (row["job_id"] + ".json"), row)
    with _CollectionRunLock(OUT, sha256_file(OUT / "plan.json"), phase):
        for i, job in enumerate(scheduled):
            path = directory / (job["job_id"] + ".json")
            if path.exists():
                saved_result(path, job)
                continue
            if (OUT / "STOP_AFTER_EPISODE.json").exists():
                write_json(OUT / "run_status.json", {"status": "paused", "phase": phase, "next_job": job["job_id"]})
                return {"paused": True, "completed": i, "total": len(scheduled)}
            write_json(OUT / "run_status.json", {"status": "running", "phase": phase,
                "active_job": job["job_id"], "completed": i, "total": len(scheduled)})
            print(f"{phase} {i+1}/{len(scheduled)} {job['job_id']}", flush=True)
            try:
                rows = _run_jobs(worker, [job], workers=1, phase=phase, output_root=OUT,
                    run_fingerprint=job["plan_sha256"], timeout_seconds=FUSE, on_result=save, stop_on_failure=True)
            except BaseException:
                write_json(OUT / "run_status.json", {"status": "error_or_interrupted", "job": job["job_id"]})
                raise
            if len(rows) != 1 or rows[0]["status"] != "ok":
                write_json(OUT / "run_status.json", {"status": "error", "job": job["job_id"]})
                raise ValueError("worker error; inspect before retry")
            result = saved_result(path, job)
            print(json.dumps({k: result[k] for k in ("ttf_seconds", "success_within_budget", "pp_calls") if k in result}), flush=True)
    report = dict(complete=True, jobs=len(scheduled), phase=phase, plan_sha256=sha256_file(OUT / "plan.json"),
        files={j["job_id"] + ".json": sha256_file(directory / (j["job_id"] + ".json")) for j in scheduled})
    if phase == "parity":
        checked = 0
        for job in scheduled:
            if not job["job_id"].endswith("-current"):
                continue
            a = read_json(directory / (job["job_id"] + ".json"))
            b = read_json(directory / (job["job_id"].replace("-current", "-legacy") + ".json"))
            if len(a["events"]) != len(b["events"]):
                raise ValueError("native parity length mismatch")
            sa, sb = a["initial_state"], b["initial_state"]
            if state_fingerprint(sa) != state_fingerprint(sb):
                raise ValueError("native reset parity mismatch")
            for ea, eb in zip(a["events"], b["events"]):
                sa, sb = apply_state_delta(sa, ea["delta"]), apply_state_delta(sb, eb["delta"])
                if state_fingerprint(sa) != state_fingerprint(sb) or ea["pool"] != eb["pool"] or ea["action"] != eb["action"]:
                    raise ValueError("native transition parity mismatch")
                for field in ("repair_order", "neighborhood", "replan_success", "pp_failure_reason"):
                    if ea["metrics"][field] != eb["metrics"][field]:
                        raise ValueError("native repair parity mismatch")
                checked += 1
        report["matched_transitions"] = checked
    write_json(OUT / (phase + "_report.json"), report)
    write_json(OUT / "run_status.json", {"status": "complete", "phase": phase, "jobs": len(scheduled)})
    return report


def comparison(rows, base, challenger):
    table = {(r["case_id"], r["arm"]): r for r in rows}
    cases = sorted({r["case_id"] for r in rows})
    pairs = [(table[c, base], table[c, challenger]) for c in cases]
    maps = sorted({a["map_id"] for a, _ in pairs})
    by_map = {m: [(a, b) for a, b in pairs if a["map_id"] == m] for m in maps}
    rng = random.Random(20260912)
    draws, success_draws = [], []
    for _ in range(5000):
        sample = [p for m in rng.choices(maps, k=len(maps)) for p in by_map[m]]
        draws.append(statistics.mean(capped_time(b) - capped_time(a) for a, b in sample))
        success_draws.append(statistics.mean(int(b["success_within_budget"]) - int(a["success_within_budget"]) for a, b in sample))
    def interval(xs):
        xs = sorted(xs)
        return [xs[124], xs[4874]]
    base_mean = statistics.mean(capped_time(a) for a, _ in pairs)
    new_mean = statistics.mean(capped_time(b) for _, b in pairs)
    successes = [sum(r["success_within_budget"] for r in side) for side in zip(*pairs)]
    ci = interval(draws)
    if new_mean < base_mean and successes[1] < successes[0]:
        decision = "speed_reliability_tradeoff"
    elif new_mean < base_mean and ci[1] < 0:
        decision = "development_timing_signal"
    elif new_mean < base_mean:
        decision = "positive_point_estimate_uncertain"
    else:
        decision = "no_mean_timing_gain_this_pilot"
    return dict(base=base, challenger=challenger, successes=successes,
        mean_capped_seconds=[base_mean, new_mean], improvement_percent=100*(base_mean-new_mean)/base_mean,
        paired_seconds_ci95=ci, paired_success_difference_ci95=interval(success_draws), decision=decision,
        gained=[a["case_id"] for a, b in pairs if b["success_within_budget"] and not a["success_within_budget"]],
        lost=[a["case_id"] for a, b in pairs if a["success_within_budget"] and not b["success_within_budget"]],
        maps={m: {"base": statistics.mean(capped_time(a) for a, b in ps),
                   "challenger": statistics.mean(capped_time(b) for a, b in ps)} for m, ps in by_map.items()})


def analyze(plan):
    report = read_json(OUT / "timed_report.json")
    rows = []
    for job in jobs(plan, "timed"):
        path = OUT / "timed" / (job["job_id"] + ".json")
        if sha256_file(path) != report["files"][path.name]:
            raise ValueError("timed output changed")
        rows.append(saved_result(path, job))
    summary = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        summary[arm] = dict(episodes=len(selected), successes=sum(r["success_within_budget"] for r in selected),
            mean_capped_ttf=statistics.mean(capped_time(r) for r in selected),
            median_capped_ttf=statistics.median(capped_time(r) for r in selected),
            generated=sum(r["generated"] for r in selected), pp_calls=sum(r["pp_calls"] for r in selected),
            selection_seconds=sum(r["selection_seconds"] for r in selected),
            pp_seconds=sum(r["pp_seconds"] for r in selected))
    result = dict(schema="lns2.sa_wall_clock_analysis.v1", plan_sha256=sha256_file(OUT / "plan.json"),
        summary=summary, comparisons=[comparison(rows, a, b) for a, b in
            (("official", "official_sa"), ("dual16", "dual16_sa"), ("official", "dual16"),
             ("official", "dual16_sa"), ("official_sa", "dual16_sa"))],
        development_only=True, no_default_promotion=True,
        details=[{k:r[k] for k in ("case_id", "map_id", "arm", "ttf_seconds", "success_within_budget", "pp_calls", "generated", "reset_seconds", "selection_seconds", "pp_seconds", "validated_seconds", "stop_reason")} for r in rows])
    write_json(OUT / "analysis.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "admission", "parity", "smoke", "collect", "analyze", "stop", "resume"))
    args = parser.parse_args()
    if args.phase == "stop":
        write_json(OUT / "STOP_AFTER_EPISODE.json", {"requested": True})
        result = {"stop_after_current_episode": True}
    elif args.phase == "prepare":
        result = prepare()
    else:
        plan = verify()
        if args.phase == "resume":
            (OUT / "STOP_AFTER_EPISODE.json").unlink(missing_ok=True)
        if args.phase in ("collect", "resume", "admission", "parity", "smoke"):
            result = execute(plan, "timed" if args.phase in ("collect", "resume") else args.phase)
        elif args.phase == "analyze":
            result = analyze(plan)
        else:
            result = {"verified": True, "jobs": 64}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
