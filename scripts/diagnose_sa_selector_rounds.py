"""Frozen acceptance/selector factorial; finite diagnostic rounds, never TTF."""

import argparse
from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.nonmonotonic_repair import probability, validate_transition
from experiments.repair_collection import _CollectionRunLock, _make_environment, _plain, _run_jobs, state_fingerprint
from scripts import diagnose_dual16_sa_transfer as transfer
from scripts import diagnose_nonmonotonic_repair as pilot
from scripts.run_feedback_exploration_diagnostics import FrozenPool, check_source_pool, padded_auc, restore, validate_final
from scripts.run_warehouse_repair_confirmation import environment_config, install_native, task_plan

OUT = ROOT / "build/sa-selector-factorial-rounds-v1"
ARMS = ("standard", "rank_greedy", "rank_sa", "uniform_greedy", "uniform_sa")
SEEDS = {"round2": (7, 11), "round3": (13, 17)}
OLD_SHA = "6308b86dd713ad6ba5757d0c405bab262e52c3c0238ed3065f394d7e0d4f02d8"
HORIZON, WORKERS, PHASE_SECONDS = 256, 20, 1800


def acceptance_arm(arm):
    if arm not in (*ARMS, "legacy"):
        raise ValueError("unknown arm")
    return "standard" if arm in ("standard", "legacy") else "annealed" if arm.endswith("_sa") else "complete_greedy"


def choose_index(arm, case_id, decision, best, pool):
    acceptance_arm(arm)
    if not pool or len({c["candidate_id"] for c in pool}) != len(pool):
        raise ValueError("invalid candidate pool")
    if not arm.startswith("uniform_"):
        return best
    order = sorted(range(len(pool)), key=lambda i: pool[i]["candidate_id"])
    return random.Random(pilot.seed(case_id, 0, decision, "uniform-choice")).choice(order)


def prepare():
    old = transfer.verify()
    if OUT.exists() or sha256_file(transfer.OUT / "report.json") != OLD_SHA:
        raise ValueError("output exists or source changed")
    cases = []
    for phase, seeds in SEEDS.items():
        for source in old["cases"]:
            if source["source_seed"] != 0:
                continue
            for seed in seeds:
                cases.append({k: source[k] for k in ("source_case_id", "map_id", "task_id", "proposal", "expected_grid")} | {
                    "case_id": source["source_case_id"] + "-native-" + str(seed), "source_seed": seed,
                    "solver_seed": seed, "restore_seed": seed, "decision": 0, "prefix": [], "phase": phase,
                    "expected_endpoints": [[p[0], p[-1]] for p in source["imported_paths"]],
                    "instance": transfer.source.digest([phase, source["source_case_id"], seed, "native-pp-import-v1"])})
    if len(cases) != 20 or len({c["case_id"] for c in cases}) != 20:
        raise ValueError("fixed initial schedule mismatch")
    inputs = dict(old["inputs"])
    for p in (Path(__file__), ROOT / "docs/SA_SELECTOR_FACTORIAL_PROTOCOL_ZH.md",
              transfer.OUT / "plan.json", transfer.OUT / "report.json"):
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    for name, sha in read_json(transfer.OUT / "report.json")["outcome_sha256"].items():
        p = transfer.OUT / "outcomes" / name
        if sha256_file(p) != sha:
            raise ValueError("old outcome changed")
        inputs[p.relative_to(ROOT).as_posix()] = sha
    plan = {"schema": "lns2.sa_selector_rounds.v1", "config": old["config"], "legacy_config": old["legacy_config"],
            "cases": cases, "inputs": inputs, "arms": list(ARMS), "horizon": HORIZON,
            "workers": WORKERS, "phase_seconds": PHASE_SECONDS, "seeds": {k: list(v) for k, v in SEEDS.items()},
            "no_ttf_or_promotion": True}
    write_json(OUT / "plan.json", plan)
    return {"admissions": 20, "smoke": 12, "round2": 50, "round3": 50, "maximum_rollout_repairs": 25600}


def verify():
    plan = read_json(OUT / "plan.json")
    for p, sha in plan["inputs"].items():
        if sha256_file(ROOT / p) != sha:
            raise ValueError("frozen input changed: " + p)
    if (plan["arms"], plan["horizon"], plan["workers"], plan["phase_seconds"], plan["seeds"]) != (
            list(ARMS), HORIZON, WORKERS, PHASE_SECONDS, {k: list(v) for k, v in SEEDS.items()}):
        raise ValueError("frozen protocol changed")
    return plan


def audit_rows(rows):
    report = {}
    for arm in pilot.ARMS:
        counts, probs = Counter(), []
        for row in rows:
            if row["arm"] != arm:
                continue
            for e in row["transitions"]:
                m = e["metrics"]
                chosen = next(c for c in e["candidate_pool"] if c["candidate_id"] == e["selected_id"])
                if chosen["score"] < max(c["score"] for c in e["candidate_pool"]) - 1e-10:
                    raise ValueError("selected candidate is not model maximum")
                if sorted(chosen["agents"]) != sorted(e["action"]["agents"]) or not m["action_valid"]:
                    raise ValueError("invalid requested action")
                counts["steps"] += 1
                counts["zero_delta"] += m["conflicts_before"] == m["conflicts_after"]
                delta = m["pp_attempt_conflict_pair_count"] - m["pp_old_conflict_pair_count"]
                if arm != "standard" and m["acceptance_evaluated"]:
                    p = probability(delta, e["temperature"] if arm == "annealed" else 0)
                    if not math.isclose(p, m["acceptance_probability"], abs_tol=1e-12):
                        raise ValueError("acceptance probability mismatch")
                    if m["replan_success"] != (delta <= 0 or e["uniform"] < p):
                        raise ValueError("acceptance decision mismatch")
                    if delta > 0:
                        probs.append(p)
                        counts["worse_attempts"] += 1
                        counts["worse_accepted"] += m["replan_success"]
        report[arm] = {**counts, "worse_probability_min": min(probs, default=None),
                       "worse_probability_mean": statistics.mean(probs) if probs else None}
    return report


def audit(plan):
    rows = [read_json(p) for p in sorted((transfer.OUT / "outcomes").glob("repair-*.json"))]
    if len(rows) != 60:
        raise ValueError("expected all 60 old outcomes")
    result = audit_rows(rows)
    state = rows[0]["final_state"]
    altered = deepcopy(state)
    altered["low_level"]["generated"] += 1
    result["metadata_changes_full_fingerprint"] = state_fingerprint(state) != state_fingerprint(altered)
    result["not_a_proven_bug"] = True
    result["source_report_sha256"] = OLD_SHA
    write_json(OUT / "audit.json", result)
    return result


def admit(job):
    case, config = deepcopy(job["case"]), job["config"]
    install_native(config)
    row = next(t["row"] for t in task_plan(config) if t["row"]["task_id"] == case["task_id"])
    env = _make_environment(str(ROOT / config["dataset"]["output"]), row,
                            {**environment_config(config), "time_limit": 120}, "Adaptive")
    raw = _plain(env.reset(seed=case["source_seed"]))
    validate_final(raw)
    paths = [a["path"] for a in sorted(raw["agents"], key=lambda a: a["id"])]
    if not raw["initial_solution_complete"] or [[p[0], p[-1]] for p in paths] != case["expected_endpoints"]:
        raise ValueError("incomplete initialization or task identity mismatch")
    del env
    env = _make_environment(str(ROOT / config["dataset"]["output"]), row,
                            {**environment_config(config), "time_limit": 600}, "Adaptive")
    state = _plain(env.reset_paths(paths, seed=case["restore_seed"]))
    validate_final(state)
    if any(state[k] != v for k, v in case["expected_grid"].items()) or state["iteration"] != 0:
        raise ValueError("native initialization grid/iteration mismatch")
    if transfer.path_array(state) != paths or state["num_of_colliding_pairs"] != raw["num_of_colliding_pairs"]:
        raise ValueError("canonical import changed paths/conflicts")
    if state["feasible"]:
        pool, selected = [], None
    else:
        best, pool = FrozenPool(case).select(env, state, 0)
        selected = pool[best]["candidate_id"]
    case.update(initial=state, state=state, candidates=pool, selected_id=selected,
                imported_paths=paths, raw_initial=raw)
    return {"status": "ok", "job_id": job["job_id"], "plan_sha256": job["plan_sha256"], "case": case}


def load_admission(path, job):
    row = read_json(path)
    if row["status"] != "ok" or any(row[k] != job[k] for k in ("job_id", "plan_sha256")):
        raise ValueError("admission identity/error")
    case = row["case"]
    if any(case[k] != v for k, v in job["case"].items()):
        raise ValueError("admission request changed")
    state = case["state"]
    validate_final(state)
    if state != case["initial"] or transfer.path_array(state) != case["imported_paths"]:
        raise ValueError("admission state changed")
    if any(state[k] != v for k, v in case["expected_grid"].items()) or [[p[0], p[-1]] for p in case["imported_paths"]] != case["expected_endpoints"]:
        raise ValueError("admission geometry changed")
    return row


def worker(job):
    env, state = restore(job)
    initial, case = state, job["case"]
    arm, mode = job["arm"], acceptance_arm(job["arm"])
    selector = None if state["feasible"] else FrozenPool(case)
    events, conflicts, began = [], [state["num_of_colliding_pairs"]], time.monotonic()
    censored, stop = False, "step_limit"
    for decision in range(job["horizon"]):
        if state["feasible"]:
            stop = "feasible"
            break
        if time.monotonic() - began >= pilot.EPISODE_SECONDS - pilot.PP_SECONDS:
            censored, stop = True, "resource_budget"
            break
        best, pool = selector.select(env, state, decision)
        if not decision:
            check_source_pool(case, best, pool)
        index = choose_index(arm, case["case_id"], decision, best, pool)
        candidate = pool[index]
        action = {"mode": "explicit_neighborhood", "agents": candidate["agents"],
                  "random_seed": pilot.seed(case["case_id"], 0, decision, "pp")}
        temp = pilot.temperature(decision)
        draw = pilot.acceptance_draw(pilot.seed(case["case_id"], 0, decision, "accept"))
        before = state
        step = _plain(env.step_with_time_limit(action, pilot.PP_SECONDS) if mode == "standard" else
                      env.step_experimental_pp(action, pilot.PP_SECONDS, mode, temp, draw))
        state, metrics = step["observation"], step["metrics"]
        validate_transition(before, state, metrics, candidate["agents"], mode, temp, draw)
        validate_final(state)
        events.append({"action": action, "metrics": metrics, "candidate_pool": pool,
                       "selected_id": candidate["candidate_id"], "rank_best_id": pool[best]["candidate_id"],
                       "temperature": temp, "uniform": draw, "before_fingerprint": state_fingerprint(before),
                       "after_fingerprint": state_fingerprint(state)})
        conflicts.append(state["num_of_colliding_pairs"])
        if metrics["pp_failure_reason"] == "time_limit" or step["truncated"]:
            censored, stop = True, "native_censor"
            break
    if state["feasible"]:
        stop = "feasible"
    return {"status": "ok", "job_id": job["job_id"], "plan_sha256": job["plan_sha256"],
            "case_id": case["case_id"], "trial": 0, "arm": arm, "map_id": case["map_id"], "horizon": job["horizon"],
            "initial_fingerprint": state_fingerprint(initial), "final_state": state, "transitions": events,
            "conflicts": conflicts, "feasible": state["feasible"], "censored": censored, "stop": stop,
            "auc": padded_auc(conflicts, job["horizon"]), "generated": state["low_level"]["generated"] - initial["low_level"]["generated"],
            "accepted_increases": sum(b > a for a, b in zip(conflicts, conflicts[1:])),
            "diagnostic_wall_seconds": time.monotonic() - began}


def load_result(path, job):
    row = pilot.load_result(path, job)
    validate_final(row["final_state"])
    for d, e in enumerate(row["transitions"]):
        pool = e["candidate_pool"]
        best = next(i for i, c in enumerate(pool) if c["candidate_id"] == e["rank_best_id"])
        if pool[best]["score"] < max(c["score"] for c in pool) - 1e-10:
            raise ValueError("model maximum mismatch")
        index = choose_index(job["arm"], row["case_id"], d, best, pool)
        if pool[index]["candidate_id"] != e["selected_id"] or sorted(pool[index]["agents"]) != sorted(e["action"]["agents"]):
            raise ValueError("selection policy mismatch")
        if e["action"]["random_seed"] != pilot.seed(row["case_id"], 0, d, "pp"):
            raise ValueError("PP random stream changed")
    return row


def execute(phase, jobs, function, loader):
    directory, began = OUT / phase, time.monotonic()
    pending = []
    for job in jobs:
        path = directory / (job["job_id"] + ".json")
        if path.exists():
            loader(path, job)
        else:
            pending.append(job)
    def save(row):
        write_json(directory / (row["job_id"] + ".json"), row)
        print(phase, row["job_id"], row["status"], flush=True)
    for offset in range(0, len(pending), WORKERS):
        if (OUT / "STOP").exists() or time.monotonic() - began + pilot.FUSE_SECONDS >= PHASE_SECONDS:
            raise InterruptedError("phase resource cap or safe stop")
        _run_jobs(function, pending[offset:offset + WORKERS], WORKERS, phase=phase, output_root=directory,
                  run_fingerprint=sha256_file(OUT / "plan.json"), timeout_seconds=pilot.FUSE_SECONDS,
                  on_result=save, failure_result=lambda j, s, e: {"job_id": j["job_id"], "status": s, "error": e}, stop_on_failure=True)
    return [loader(directory / (j["job_id"] + ".json"), j) for j in jobs]


def jobs(plan, cases, phase):
    selected = [c for c in cases if c["phase"] == ("round2" if phase == "smoke" else phase)]
    if phase == "smoke":
        selected = selected[:2]
    return [{"job_id": c["case_id"] + "-" + a, "case": c, "arm": a, "trial": 0,
             "horizon": 2 if phase == "smoke" else HORIZON,
             "config": plan["legacy_config"] if a == "legacy" else plan["config"],
             "plan_sha256": sha256_file(OUT / "plan.json")}
            for c in selected for a in ((*ARMS, "legacy") if phase == "smoke" else ARMS)]


def smoke_gate(rows):
    if len(rows) != 12 or any(r["censored"] for r in rows):
        raise ValueError("incomplete/censored smoke")
    for case in {r["case_id"] for r in rows}:
        group = {r["arm"]: r for r in rows if r["case_id"] == case}
        if set(group) != {*ARMS, "legacy"}:
            raise ValueError("smoke arms changed")
        a, b = group["standard"], group["legacy"]
        if state_fingerprint(a["final_state"]) != state_fingerprint(b["final_state"]) or [
            (e["candidate_pool"], e["selected_id"]) for e in a["transitions"]] != [
            (e["candidate_pool"], e["selected_id"]) for e in b["transitions"]]:
            raise ValueError("default native/controller parity changed")
    return {"passed": True, "pairs": 2}


def compare(table, arm, base):
    keys = sorted(table[base])
    gains = [k for k in keys if table[arm][k]["feasible"] and not table[base][k]["feasible"]]
    losses = [k for k in keys if table[base][k]["feasible"] and not table[arm][k]["feasible"]]
    common = [k for k in keys if table[base][k]["feasible"] and table[arm][k]["feasible"]]
    return {"gains": gains, "losses": losses, "common_success": len(common),
            "common_success_repairs": {a: sum(len(table[a][k]["transitions"]) for k in common) for a in (base, arm)}}


def portfolio(a, b, budget):
    # Fixed alternation, no access to future success when allocating a turn.
    if a["feasible"] and not a["transitions"]:
        return {"feasible": True, "repair_calls": 0, "selected_lane": "standard"}
    if b["feasible"] and not b["transitions"]:
        return {"feasible": True, "repair_calls": 0, "selected_lane": "rank_sa"}
    for turn in range(1, budget + 1):
        row = a if turn % 2 else b
        used = (turn + 1) // 2 if turn % 2 else turn // 2
        if row["feasible"] and len(row["transitions"]) <= used:
            return {"feasible": True, "repair_calls": turn, "selected_lane": "standard" if turn % 2 else "rank_sa"}
    return {"feasible": False, "repair_calls": budget, "selected_lane": None}


def summarize(rows):
    tables = {a: {} for a in ARMS}
    for r in rows:
        if r["arm"] not in tables or r["case_id"] in tables[r["arm"]]:
            raise ValueError("duplicate/unknown result")
        tables[r["arm"]][r["case_id"]] = r
    keys = set(tables["standard"])
    if not keys or any(set(t) != keys for t in tables.values()):
        raise ValueError("unpaired outcomes")
    for k in keys:
        if len({tables[a][k]["initial_fingerprint"] for a in ARMS}) != 1:
            raise ValueError("initial states differ")
    arms = {a: {"episodes": len(t), "feasible": sum(r["feasible"] for r in t.values()),
                 "censored": sum(r["censored"] for r in t.values()), "generated": sum(r["generated"] for r in t.values()),
                 "auc_sum": sum(r["auc"] for r in t.values()),
                 "repair_calls": sum(len(r["transitions"]) for r in t.values()),
                 "accepted_increases": sum(r["accepted_increases"] for r in t.values())} for a, t in tables.items()}
    comparisons = {a + "_vs_standard": compare(tables, a, "standard") for a in ARMS[1:]}
    for a, b in (("rank_sa", "rank_greedy"), ("uniform_sa", "uniform_greedy"), ("uniform_sa", "rank_sa"), ("uniform_greedy", "rank_greedy")):
        comparisons[a + "_vs_" + b] = compare(tables, a, b)
    ports = {str(budget): {k: portfolio(tables["standard"][k], tables["rank_sa"][k], budget) for k in sorted(keys)} for budget in (256, 512)}
    maps = sorted({r["map_id"] for r in rows})
    strata = {m: {a: {"feasible": sum(r["feasible"] for r in tables[a].values() if r["map_id"] == m),
                      "auc": sum(r["auc"] for r in tables[a].values() if r["map_id"] == m)} for a in ARMS} for m in maps}
    intervals = {}
    for arm in ARMS[1:]:
        rng = random.Random(20260912)
        samples = sorted(sum(strata[m]["standard"]["auc"] - strata[m][arm]["auc"]
                             for m in rng.choices(maps, k=len(maps))) / len(keys) for _ in range(5000))
        intervals[arm] = [samples[124], samples[4874]]
    return {"arms": arms, "comparisons": comparisons, "fixed_alternation_portfolio": ports,
            "by_map": strata, "map_bootstrap_auc_reduction_ci95_descriptive": intervals,
            "no_ttf_or_promotion": True, "paired": len(keys)}


def backup(phase):
    paths = [OUT / "plan.json", OUT / (phase + "_report.json"), *sorted((OUT / phase).glob("*.json"))]
    target = OUT / (phase + "_backup.zip")
    if target.exists():
        with zipfile.ZipFile(target) as archive:
            if any(archive.read(p.relative_to(OUT).as_posix()) != p.read_bytes() for p in paths):
                raise ValueError("backup differs; do not overwrite")
        return
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in paths:
            archive.write(p, p.relative_to(OUT).as_posix())
    with zipfile.ZipFile(target) as archive:
        if any(archive.read(p.relative_to(OUT).as_posix()) != p.read_bytes() for p in paths):
            raise ValueError("backup byte mismatch")


def run(phase):
    plan = verify()
    with _CollectionRunLock(OUT, sha256_file(OUT / "plan.json"), phase):
        write_json(OUT / "run_status.json", {"phase": phase, "status": "running"})
        try:
            if phase == "audit":
                result = audit(plan)
            else:
                if not (OUT / "audit.json").exists():
                    raise ValueError("audit required")
                aj = [{"job_id": c["case_id"], "case": c, "config": plan["config"], "plan_sha256": sha256_file(OUT / "plan.json")} for c in plan["cases"]]
                if phase == "admit":
                    records = execute("admission", aj, admit, load_admission)
                    result = {"admitted": len(records), "zero_conflict": sum(r["case"]["state"]["feasible"] for r in records)}
                else:
                    cases = [load_admission(OUT / "admission" / (j["job_id"] + ".json"), j)["case"] for j in aj]
                    if phase == "smoke":
                        result = smoke_gate(execute(phase, jobs(plan, cases, phase), worker, load_result))
                    else:
                        smoke_gate([load_result(OUT / "smoke" / (j["job_id"] + ".json"), j) for j in jobs(plan, cases, "smoke")])
                        if phase == "round3" and not (OUT / "round2_report.json").exists():
                            raise ValueError("round2 review must finish first")
                        rows = execute(phase, jobs(plan, cases, phase), worker, load_result)
                        result = summarize(rows)
                        result["plan_sha256"] = sha256_file(OUT / "plan.json")
                        result["outcome_sha256"] = {r["job_id"]: sha256_file(OUT / phase / (r["job_id"] + ".json")) for r in rows}
                        write_json(OUT / (phase + "_report.json"), result)
                        backup(phase)
            write_json(OUT / "run_status.json", {"phase": phase, "status": "complete", "summary": result})
            return result
        except BaseException as exc:
            write_json(OUT / "run_status.json", {"phase": phase, "status": "failed_or_paused", "error": str(exc)})
            raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "audit", "admit", "smoke", "round2", "round3", "verify"))
    phase = p.parse_args().phase
    print(json.dumps(prepare() if phase == "prepare" else {"verified": bool(verify())} if phase == "verify" else run(phase), indent=2))
