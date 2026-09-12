"""Unscored candidate equivalence and canonical-state Official comparison; not TTF."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import time
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _plain, _run_jobs, state_fingerprint
from experiments.nonmonotonic_repair import validate_transition
from scripts import diagnose_sa_independent_lanes as source
from scripts.run_feedback_exploration_diagnostics import restore, validate_final

OUT = ROOT / "build/uniform-runtime-official-probe-v1"
ARMS = ("official", "uniform_standard", "uniform_complete")
HORIZON, WORKERS = 256, 20


class UnscoredPool:
    def __init__(self, case):
        self.case = case

    def select(self, env, state, decision):
        from experiments.online_feature_engine import TopologyAnalysisCache
        from lns2_selector.runtime.online_selection import generate_online_candidates
        from lns2_selector.runtime.structshell_dual16 import structshell_dual16_augmentation, generate_structshell_dual16_runtime_candidates
        before = state_fingerprint(state)
        candidates, _ = generate_online_candidates(env, state, task_id=self.case["task_id"],
            solver_seed=self.case["solver_seed"], decision_index=decision, proposal_config=self.case["proposal"],
            state_hash=before, verify_full_state=True, proposal_backend="optimized")
        topology = TopologyAnalysisCache(state, backend="native")
        pool = list(generate_structshell_dual16_runtime_candidates(state, topology.analysis,
            v2_candidates=candidates, config=structshell_dual16_augmentation()).candidates)
        if state_fingerprint(env.get_state()) != before:
            raise ValueError("proposal changed state")
        index = source.old.choose_index("uniform_greedy", self.case["case_id"], decision, 0, pool)
        return index, pool


def unscored(pool):
    return [{k: v for k, v in candidate.items() if k != "score"} for candidate in pool]


def no_model_guard():
    stack = ExitStack()
    for name in ("experiments.compact_controller_model.load_controller_bundle",
                 "lns2_selector.runtime.online_selection.score_online_candidates",
                 "experiments.online_feature_engine.OnlineFeatureEngine.realized_rows"):
        stack.enter_context(patch(name, side_effect=AssertionError("model/feature scoring called in unscored runtime")))
    return stack


def prepare():
    old = source.verify()
    source.analyze()
    if OUT.exists():
        raise ValueError("output exists")
    inputs, cases, references = dict(old["inputs"]), [], {}
    for path in (Path(__file__), ROOT / "docs/UNIFORM_RUNTIME_OFFICIAL_PROTOCOL_ZH.md", source.OUT / "plan.json",
                 source.OUT / "analysis.json", source.OUT / "admission_report.json"):
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    admission_hashes = read_json(source.OUT / "admission_report.json")["outcome_sha256"]
    for request in old["cases"]:
        path = source.OUT / "admission" / (request["case_id"] + ".json")
        if sha256_file(path) != admission_hashes[path.name]:
            raise ValueError("admission changed")
        case = read_json(path)["case"]
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        reference = source.OUT / case["phase"] / (case["case_id"] + "-uniform_greedy.json")
        report = read_json(source.OUT / (case["phase"] + "_report.json"))
        if sha256_file(reference) != report["outcome_sha256"][reference.name]:
            raise ValueError("reference changed")
        inputs[reference.relative_to(ROOT).as_posix()] = sha256_file(reference)
        references[case["case_id"]] = reference.relative_to(ROOT).as_posix()
        cases.append(case)
    if len(cases) != 16 or len({c["map_id"] for c in cases}) != 8:
        raise ValueError("source coverage mismatch")
    plan = {"schema": "lns2.uniform_runtime_official.v1", "config": old["config"], "cases": cases,
            "inputs": inputs, "references": references, "arms": list(ARMS), "horizon": HORIZON,
            "workers": WORKERS, "no_ttf_or_promotion": True}
    write_json(OUT / "plan.json", plan)
    return {"parity_jobs": 16, "baseline_jobs": 32, "expected_parity_repairs": 445,
            "baseline_repair_cap": 8192, "not_new_data": True}


def verify():
    plan = read_json(OUT / "plan.json")
    if (plan["arms"], plan["horizon"], plan["workers"]) != (list(ARMS), HORIZON, WORKERS):
        raise ValueError("protocol changed")
    for path, sha in plan["inputs"].items():
        if sha256_file(ROOT / path) != sha:
            raise ValueError("frozen input changed: " + path)
    return plan


def action_for(arm, case, decision, pool=None, index=None):
    if arm == "official":
        return {"mode": "official"}
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return {"mode": "explicit_neighborhood", "agents": pool[index]["agents"],
            "random_seed": source.old.pilot.seed(case["case_id"], 0, decision, "pp")}


def worker(job):
    began = time.monotonic()
    env, state = restore(job)
    initial, case, arm = state, job["case"], job["arm"]
    pooler = UnscoredPool(case) if arm != "official" else None
    events, conflicts, censored = [], [state["num_of_colliding_pairs"]], False
    with no_model_guard():
        for decision in range(job["horizon"]):
            if state["feasible"]:
                break
            if time.monotonic() - began >= source.old.pilot.EPISODE_SECONDS - source.old.pilot.PP_SECONDS:
                censored = True
                break
            index, pool = pooler.select(env, state, decision) if pooler else (None, [])
            action = action_for(arm, case, decision, pool, index)
            before = state
            temp = source.old.pilot.temperature(decision)
            draw = source.old.pilot.acceptance_draw(source.old.pilot.seed(case["case_id"], 0, decision, "accept"))
            mode = "complete_greedy" if arm == "uniform_complete" else "standard"
            step = _plain(env.step_experimental_pp(action, source.old.pilot.PP_SECONDS, mode, temp, draw)
                          if mode == "complete_greedy" else env.step_with_time_limit(action, source.old.pilot.PP_SECONDS))
            state, m = step["observation"], step["metrics"]
            selected = m["neighborhood"] if arm == "official" else pool[index]["agents"]
            validate_transition(before, state, m, selected, mode, temp, draw)
            validate_final(state)
            event = {"action": action, "candidate_pool": pool, "selected_id": pool[index]["candidate_id"] if pool else None,
                     "metrics": m, "before_fingerprint": state_fingerprint(before), "after_fingerprint": state_fingerprint(state),
                     "generated_delta": state["low_level"]["generated"] - before["low_level"]["generated"]}
            events.append(event)
            conflicts.append(state["num_of_colliding_pairs"])
            if m["pp_failure_reason"] == "time_limit" or step["truncated"]:
                censored = True
                break
    return {"status": "ok", "job_id": job["job_id"], "plan_sha256": job["plan_sha256"], "arm": arm,
            "case_id": case["case_id"], "map_id": case["map_id"], "horizon": job["horizon"],
            "initial_fingerprint": state_fingerprint(initial), "final_state": state, "transitions": events,
            "conflicts": conflicts, "feasible": state["feasible"], "censored": censored,
            "decisions": len(events), "pp_calls": sum(e["metrics"]["pp_attempted_agent_count"] > 0 for e in events),
            "generated": state["low_level"]["generated"] - initial["low_level"]["generated"],
            "model_scoring_forbidden": True, "diagnostic_wall_seconds": time.monotonic() - began}


def load_result(path, job):
    row = read_json(path)
    integrity = row.pop("integrity_sha256", None)
    if source.digest(row) != integrity or row["status"] != "ok":
        raise ValueError("result integrity/error")
    if any(row[k] != job[k] for k in ("job_id", "plan_sha256", "arm", "horizon")) or any(row[k] != job["case"][k] for k in ("case_id", "map_id")):
        raise ValueError("result identity mismatch")
    previous = state_fingerprint(job["case"]["state"])
    if previous != row["initial_fingerprint"] or row["conflicts"][0] != job["case"]["state"]["num_of_colliding_pairs"]:
        raise ValueError("initial state mismatch")
    events = row["transitions"]
    if len(row["conflicts"]) != len(events) + 1 or row["decisions"] != len(events) or len(events) > job["horizon"]:
        raise ValueError("budget/sequence mismatch")
    for d, event in enumerate(events):
        pool, m = event["candidate_pool"], event["metrics"]
        if row["arm"] == "official":
            if pool or event["selected_id"] is not None or event["action"] != {"mode": "official"}:
                raise ValueError("official RNG/action changed")
        else:
            if any("score" in c for c in pool):
                raise ValueError("score in unscored pool")
            index = source.old.choose_index("uniform_greedy", row["case_id"], d, 0, pool)
            if (event["selected_id"] != pool[index]["candidate_id"] or
                    event["action"] != action_for(row["arm"], job["case"], d, pool, index)):
                raise ValueError("uniform selection mismatch")
        if (previous != event["before_fingerprint"] or not m["action_valid"] or
                [m["conflicts_before"], m["conflicts_after"]] != row["conflicts"][d:d+2]):
            raise ValueError("transition mismatch")
        previous = event["after_fingerprint"]
    validate_final(row["final_state"])
    if (previous != state_fingerprint(row["final_state"]) or row["feasible"] != row["final_state"]["feasible"] or
            row["generated"] != sum(e["generated_delta"] for e in events) or
            row["generated"] != row["final_state"]["low_level"]["generated"] - job["case"]["state"]["low_level"]["generated"]):
        raise ValueError("final state or nodes mismatch")
    if not row["feasible"] and not row["censored"] and len(events) != job["horizon"]:
        raise ValueError("unexplained termination")
    return row


def parity(plan, rows):
    comparisons = []
    for row in rows:
        reference = read_json(ROOT / plan["references"][row["case_id"]])["lanes"]["uniform_greedy"]
        if (len(row["transitions"]) != len(reference["transitions"]) or row["conflicts"] != reference["conflicts"]
                or row["generated"] != reference["generated"] or row["censored"]
                or state_fingerprint(row["final_state"]) != state_fingerprint(reference["final_state"])):
            raise ValueError("trajectory equivalence failed: " + row["case_id"])
        for d, (actual, saved) in enumerate(zip(row["transitions"], reference["transitions"])):
            fields = ("action", "selected_id", "before_fingerprint", "after_fingerprint", "generated_delta")
            metrics = ("repair_order", "replan_success", "pp_failure_reason", "pp_attempt_conflict_pair_count", "pp_old_conflict_pair_count", "acceptance_probability")
            if (unscored(actual["candidate_pool"]) != unscored(saved["candidate_pool"]) or
                    any(actual[k] != saved[k] for k in fields) or any(actual["metrics"][k] != saved["metrics"][k] for k in metrics)):
                raise ValueError(f"candidate/action equivalence failed: {row['case_id']} step {d}")
        comparisons.append({"case_id": row["case_id"], "decisions": row["decisions"], "passed": True})
    if len(comparisons) != 16:
        raise ValueError("incomplete parity")
    return {"passed": True, "comparisons": comparisons, "decisions": sum(r["decisions"] for r in rows)}


def jobs(plan, phase):
    arms = ("uniform_complete",) if phase == "parity" else ("official", "uniform_standard")
    return [{"job_id": c["case_id"] + "-" + arm, "case": c, "config": plan["config"], "arm": arm,
             "horizon": HORIZON, "plan_sha256": sha256_file(OUT / "plan.json")} for c in plan["cases"] for arm in arms]


def execute(plan, phase):
    scheduled, directory, began = jobs(plan, phase), OUT / phase, time.monotonic()
    pending = []
    for job in scheduled:
        path = directory / (job["job_id"] + ".json")
        if path.exists():
            load_result(path, job)
        else:
            pending.append(job)
    def save(row):
        if row["status"] == "ok":
            row["integrity_sha256"] = source.digest(row)
        write_json(directory / (row["job_id"] + ".json"), row)
        print(phase, row["job_id"], row["status"], flush=True)
    for offset in range(0, len(pending), WORKERS):
        if (OUT / "STOP").exists() or time.monotonic() - began + 180 >= 1800:
            raise InterruptedError("safe stop or phase budget")
        _run_jobs(worker, pending[offset:offset+WORKERS], WORKERS, phase=phase, output_root=directory,
                  run_fingerprint=sha256_file(OUT / "plan.json"), timeout_seconds=180, on_result=save,
                  failure_result=lambda j,s,e: {"job_id": j["job_id"], "status": s, "error": e}, stop_on_failure=True)
    return [load_result(directory / (j["job_id"] + ".json"), j) for j in scheduled]


def summarize(rows):
    table = {a: {r["case_id"]: r for r in rows if r["arm"] == a} for a in ARMS}
    keys = set(table["official"])
    if len(keys) != 16 or len(rows) != 48 or any(set(t) != keys for t in table.values()):
        raise ValueError("incomplete pairs")
    for key in keys:
        if len({table[a][key]["initial_fingerprint"] for a in ARMS}) != 1:
            raise ValueError("unpaired initial paths")
    def totals(group):
        return {"episodes": len(group), "success": sum(r["feasible"] for r in group),
                **{k: sum(r[k] for r in group) for k in ("decisions", "pp_calls", "generated", "censored")}}
    arms = {a: totals(list(t.values())) for a,t in table.items()}
    maps = sorted({r["map_id"] for r in rows})
    by_map = {m: {a: totals([r for r in t.values() if r["map_id"] == m]) for a,t in table.items()} for m in maps}
    comparisons = {}
    for a in ARMS[1:]:
        common = [k for k in keys if table[a][k]["feasible"] and table["official"][k]["feasible"]]
        gains = sorted(k for k in keys if table[a][k]["feasible"] and not table["official"][k]["feasible"])
        losses = sorted(k for k in keys if not table[a][k]["feasible"] and table["official"][k]["feasible"])
        not_worse = [m for m in maps if by_map[m][a]["generated"] <= by_map[m]["official"]["generated"]]
        gates = {"integrity": not any(r["censored"] for r in rows), "no_success_loss": not losses,
                 "nodes_reduce_5pct": arms[a]["generated"] <= .95 * arms["official"]["generated"],
                 "maps_not_worse_4of8": len(not_worse) >= 4}
        comparisons[a] = {"gains": gains, "losses": losses, "maps_not_worse": not_worse,
                          "common_success": {b: totals([table[b][k] for k in common]) for b in ("official",a)},
                          "gates": gates, "passed": all(gates.values())}
    return {"arms": arms, "by_map": by_map, "comparisons": comparisons, "no_ttf_or_promotion": True,
            "per_case": {k: {a: {field: table[a][k][field] for field in ("feasible", "decisions", "generated", "censored")} for a in ARMS} for k in sorted(keys)},
            "decision": "new_task_confirmation_only" if any(c["passed"] for c in comparisons.values()) else "stop_uniform_timing_extension"}


def backup(phase):
    paths = [OUT / "plan.json", OUT / (phase + "_report.json"), *sorted((OUT / phase).glob("*.json"))]
    target = OUT / (phase + "_backup.zip")
    if not target.exists():
        with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, path.relative_to(OUT).as_posix())
    with zipfile.ZipFile(target) as archive:
        if any(archive.read(p.relative_to(OUT).as_posix()) != p.read_bytes() for p in paths):
            raise ValueError("backup differs")


def run(phase):
    plan = verify()
    with _CollectionRunLock(OUT, sha256_file(OUT / "plan.json"), phase):
        write_json(OUT / "run_status.json", {"phase": phase, "status": "running"})
        try:
            if phase == "parity":
                result = parity(plan, execute(plan, phase))
            else:
                parity_rows = [load_result(OUT / "parity" / (j["job_id"] + ".json"), j) for j in jobs(plan,"parity")]
                parity(plan, parity_rows)
                rows = execute(plan,phase)
                result = summarize(parity_rows + rows)
            result["plan_sha256"] = sha256_file(OUT / "plan.json")
            result["outcome_sha256"] = {p.name: sha256_file(p) for p in sorted((OUT / phase).glob("*.json")) if p.name != "collection_progress.json"}
            write_json(OUT / (phase + "_report.json"), result)
            backup(phase)
            write_json(OUT / "run_status.json", {"phase": phase, "status": "complete", "summary": result})
            return result
        except BaseException as exc:
            write_json(OUT / "run_status.json", {"phase": phase, "status": "failed_or_paused", "error": str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "parity", "baseline"))
    phase = parser.parse_args().phase
    print(json.dumps(prepare() if phase == "prepare" else {"verified": bool(verify())} if phase == "verify" else run(phase), indent=2))
