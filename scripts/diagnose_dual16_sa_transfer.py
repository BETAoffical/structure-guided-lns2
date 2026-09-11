"""Transfer existing experimental SA to frozen Dual16 at author initial path states."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _make_environment, _plain, _run_jobs, state_fingerprint
from scripts import diagnose_nonmonotonic_repair as pilot
from scripts import diagnose_cplns_real_tasks as source
from scripts.run_feedback_exploration_diagnostics import FrozenPool, validate_final
from scripts.run_warehouse_repair_confirmation import environment_config, install_native, task_plan

OUT = ROOT / "build/dual16-sa-initial-transfer-v1"
PROTOCOL = ROOT / "docs/DUAL16_SA_INITIAL_TRANSFER_PROTOCOL_ZH.md"
NATIVE_SHA = "f715b2c4157c901d5dcc4ef2b008197efff6a39da95438a9fa679d20dd8c0632"
OLD_REPORT_SHA = "d9b560e2eb342d6989b1d17f8202e37ee70e9c97795e7a545a0af3f782a691a4"
AUTHOR_REPORT_SHA = "49d0eae69d3c62f76fd806df4733cd997f8a43cabb6096c2c86c6b660736b4c6"
HORIZON, WORKERS, TOTAL_SECONDS = 256, 20, 1800


def initial_event(path):
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "init":
                return event
    raise ValueError("required author initial event missing")


def path_array(event):
    agents = sorted(event["agents"], key=lambda a: a["id"])
    if [a["id"] for a in agents] != list(range(len(agents))) or not agents:
        raise ValueError("invalid imported agent IDs")
    if any(not a["path"] for a in agents):
        raise ValueError("empty imported path")
    return [a["path"] for a in agents]


def prepare():
    real = source.verify()
    base = ROOT / real["config"]["output"]
    if OUT.exists():
        raise ValueError("output exists; do not overwrite")
    if sha256_file(base / "report.json") != AUTHOR_REPORT_SHA or sha256_file(pilot.OUT / "report.json") != OLD_REPORT_SHA:
        raise ValueError("reference evidence changed")
    if sha256_file(pilot.SOURCE) != pilot.SOURCE_SHA or sha256_file(pilot.NATIVE) != NATIVE_SHA:
        raise ValueError("discovery/native identity changed")
    supplied = read_json(pilot.SOURCE)
    legacy = supplied["config"]
    config = deepcopy(legacy)
    config["frozen"].update(native_path=pilot.NATIVE.relative_to(ROOT).as_posix(), native_sha256=NATIVE_SHA)
    cases_by_id = {c["case_id"]: c for c in supplied["discovery"]}
    registered = {r["job_id"]: r for r in read_json(base / "report.json")["rows"]}
    paths = [Path(__file__), PROTOCOL, Path(pilot.__file__), Path(source.__file__),
             ROOT / "scripts/run_feedback_exploration_diagnostics.py", ROOT / "scripts/audit_feedback_memory.py",
             ROOT / "scripts/run_warehouse_repair_confirmation.py", ROOT / "scripts/prepare_warehouse_repair_confirmation.py",
             pilot.SOURCE, pilot.OUT / "report.json", pilot.NATIVE, ROOT / legacy["frozen"]["native_path"],
             base / "plan.json", base / "report.json"]
    cases = []
    for job in real["jobs"]:
        if job["profile"] != source.BASE:
            continue
        record_path = base / "jobs" / (job["job_id"] + ".json")
        record = read_json(record_path)
        if sha256_file(record_path) != registered[job["job_id"]]["job_record_sha256"]:
            raise ValueError("source job identity changed")
        source.bridge.verify_job(record, job)
        event_path = record_path.with_suffix(".events.jsonl")
        event = initial_event(event_path)
        old = cases_by_id[job["case_id"]]
        cases.append({"case_id": job["case_id"] + "-initial-" + str(job["seed"]),
                      "source_case_id": job["case_id"], "source_seed": job["seed"],
                      "map_id": old["map_id"], "task_id": old["task_id"],
                      "instance": source.digest({"source_job": job["job_id"], "events_sha256": record["events_sha256"],
                                                 "scope": "author_initial_import"}),
                      "solver_seed": job["seed"], "restore_seed": job["seed"], "proposal": old["proposal"],
                      "decision": 0, "prefix": [], "imported_paths": path_array(event),
                      "expected_conflicts": event["conflicts"], "expected_cost": event["cost"],
                      "expected_grid": {k: old["state"][k] for k in ("rows", "cols", "obstacles")}})
        paths += [record_path, event_path]
    if len(cases) != 20 or len({c["case_id"] for c in cases}) != 20:
        raise ValueError("expected 20 fixed initial states")
    for directory in ("experiments", "lns2_selector", "artifacts/initlns-closed-loop-controller-v2", legacy["dataset"]["output"]):
        paths += [p for p in (ROOT / directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    if sha256_file(ROOT / legacy["frozen"]["native_path"]) != legacy["frozen"]["native_sha256"]:
        raise ValueError("legacy native changed")
    plan = {"schema": "lns2.dual16_sa_initial_transfer.v1", "config": config, "legacy_config": legacy,
            "cases": cases, "horizon": HORIZON, "workers": WORKERS,
            "pp_seconds": pilot.PP_SECONDS, "episode_seconds": pilot.EPISODE_SECONDS,
            "fuse_seconds": pilot.FUSE_SECONDS, "total_seconds": TOTAL_SECONDS,
            "inputs": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths},
            "no_ttf_or_promotion": True, "not_old_stalled_paths": True}
    write_json(OUT / "plan.json", plan)
    return {"initial_states": 20, "admission_jobs": 20, "two_step_smoke_jobs": 20,
            "rollout_jobs": 60, "horizon": HORIZON, "maximum_rollout_repairs": 60 * HORIZON, "workers": WORKERS}


def verify():
    plan = read_json(OUT / "plan.json")
    for p, expected in plan["inputs"].items():
        if sha256_file(ROOT / p) != expected:
            raise ValueError("frozen input changed: " + p)
    if (plan["horizon"], plan["workers"], plan["pp_seconds"], plan["episode_seconds"], plan["fuse_seconds"]) != (
        HORIZON, WORKERS, pilot.PP_SECONDS, pilot.EPISODE_SECONDS, pilot.FUSE_SECONDS
    ):
        raise ValueError("resource bounds changed")
    return plan


def admit(job):
    config, case = job["config"], deepcopy(job["case"])
    install_native(config)
    row = next(t["row"] for t in task_plan(config) if t["row"]["task_id"] == case["task_id"])
    env = _make_environment(str(ROOT / config["dataset"]["output"]), row,
                            {**environment_config(config), "time_limit": 600}, "Adaptive")
    state = _plain(env.reset_paths(case["imported_paths"], seed=case["restore_seed"]))
    validate_final(state)
    if any(state[k] != value for k, value in case["expected_grid"].items()):
        raise ValueError("grid differs from author task")
    if [a["path"] for a in sorted(state["agents"], key=lambda a: a["id"])] != case["imported_paths"]:
        raise ValueError("native import altered paths")
    if (state["num_of_colliding_pairs"], state["sum_of_costs"]) != (case["expected_conflicts"], case["expected_cost"]):
        raise ValueError("author/native imported path metrics differ")
    if state["iteration"] != 0:
        raise ValueError("import must start before first repair")
    index, pool = FrozenPool(case).select(env, state, 0)
    case.update(initial=state, state=state, candidates=pool, selected_id=pool[index]["candidate_id"])
    return {"status": "ok", "job_id": job["job_id"], "plan_sha256": job["plan_sha256"], "case": case}


def schedule(plan, admitted, smoke=False):
    result = []
    for case in admitted:
        if smoke and case["source_seed"] != 0:
            continue
        for arm in (*pilot.ARMS, "legacy") if smoke else pilot.ARMS:
            result.append({"job_id": case["case_id"] + "-" + arm, "case": case, "trial": 0, "arm": arm,
                           "horizon": 2 if smoke else HORIZON,
                           "config": plan["legacy_config"] if arm == "legacy" else plan["config"],
                           "plan_sha256": sha256_file(OUT / "plan.json")})
    return result


def execute(plan, phase, jobs, worker, loader):
    directory = OUT / phase
    began, pending = time.monotonic(), []
    for job in jobs:
        path = directory / (job["job_id"] + ".json")
        if path.exists():
            loader(path, job)
        else:
            pending.append(job)
    def save(row):
        write_json(directory / (row["job_id"] + ".json"), row)
        print(row["job_id"], row["status"], flush=True)
    for offset in range(0, len(pending), WORKERS):
        if (OUT / "STOP").exists() or time.monotonic() - began + pilot.FUSE_SECONDS >= TOTAL_SECONDS:
            raise InterruptedError("safe stop or resource cap")
        _run_jobs(worker, pending[offset:offset + WORKERS], WORKERS, phase=phase, output_root=directory,
                  run_fingerprint=sha256_file(OUT / "plan.json"), timeout_seconds=pilot.FUSE_SECONDS,
                  on_result=save, failure_result=lambda j, s, e: {"job_id": j["job_id"], "status": s, "error": e},
                  stop_on_failure=True)
    return [loader(directory / (j["job_id"] + ".json"), j) for j in jobs]


def load_admission(path, job):
    row = read_json(path)
    if row["status"] != "ok" or row["job_id"] != job["job_id"] or row["plan_sha256"] != job["plan_sha256"]:
        raise ValueError("admission identity/error")
    expected = job["case"]
    if any(row["case"][k] != v for k, v in expected.items()):
        raise ValueError("admission source mismatch")
    state = row["case"]["state"]
    validate_final(state)
    if state != row["case"]["initial"] or [a["path"] for a in sorted(state["agents"], key=lambda a: a["id"])] != expected["imported_paths"]:
        raise ValueError("admission paths changed")
    return row


def smoke_gate(rows):
    groups = {}
    for row in rows:
        group = groups.setdefault(row["case_id"], {})
        if row["arm"] in group:
            raise ValueError("duplicate smoke arm")
        group[row["arm"]] = row
    for group in groups.values():
        if set(group) != {*pilot.ARMS, "legacy"} or any(r["censored"] for r in group.values()):
            raise ValueError("incomplete/censored smoke")
        a, b = group["legacy"], group["standard"]
        if state_fingerprint(a["final_state"]) != state_fingerprint(b["final_state"]):
            raise ValueError("legacy/experimental default parity failure")
        if [(e["candidate_pool"], e["selected_id"]) for e in a["transitions"]] != [
            (e["candidate_pool"], e["selected_id"]) for e in b["transitions"]
        ]:
            raise ValueError("default controller candidate/scorer parity failure")
    return {"passed": True, "legacy_pairs": len(groups)}


def summarize(rows):
    old_gate = pilot.summarize(rows)
    tables = {a: {r["case_id"]: r for r in rows if r["arm"] == a} for a in pilot.ARMS}
    metrics = {a: {"feasible": sum(r["feasible"] for r in t.values()),
                   "mean_executed_repairs": statistics.mean(len(r["transitions"]) for r in t.values()),
                   "generated": sum(r["generated"] for r in t.values())} for a, t in tables.items()}
    comparisons = {}
    for base in ("standard", "complete_greedy"):
        common = [k for k in tables[base] if tables[base][k]["feasible"] and tables["annealed"][k]["feasible"]]
        sums = {a: sum(len(tables[a][k]["transitions"]) for k in common) for a in (base, "annealed")}
        comparisons[base] = {"common_success": len(common), "common_success_repairs": sums,
                             "fewer_repairs": sum(len(tables["annealed"][k]["transitions"]) < len(tables[base][k]["transitions"]) for k in common),
                             "more_repairs": sum(len(tables["annealed"][k]["transitions"]) > len(tables[base][k]["transitions"]) for k in common)}
    return {**old_gate, "effort": metrics, "common_success_comparisons": comparisons,
            "scope": "frozen_Dual16_on_imported_author_initial_paths_not_late_stall_replay",
            "no_ttf_or_promotion": True}


def run(phase):
    plan = verify()
    with _CollectionRunLock(OUT, sha256_file(OUT / "plan.json"), phase):
        write_json(OUT / "run_status.json", {"status": "running", "phase": phase})
        try:
            admission_jobs = [{"job_id": c["case_id"], "case": c, "config": plan["config"],
                               "plan_sha256": sha256_file(OUT / "plan.json")} for c in plan["cases"]]
            if phase == "admit":
                rows = execute(plan, "admission", admission_jobs, admit, load_admission)
                result = {"admitted": len(rows), "path_imports_exact": True}
            else:
                admitted = [load_admission(OUT / "admission" / (j["job_id"] + ".json"), j)["case"] for j in admission_jobs]
                if phase == "smoke":
                    rows = execute(plan, "smoke", schedule(plan, admitted, True), pilot.worker, pilot.load_result)
                    result = smoke_gate(rows)
                else:
                    smoke_gate([pilot.load_result(OUT / "smoke" / (j["job_id"] + ".json"), j) for j in schedule(plan, admitted, True)])
                    rows = execute(plan, "outcomes", schedule(plan, admitted), pilot.worker, pilot.load_result)
                    result = summarize(rows)
                    result["plan_sha256"] = sha256_file(OUT / "plan.json")
                    result["outcome_sha256"] = {p.name: sha256_file(p) for p in sorted((OUT / "outcomes").glob("*.json"))}
                    write_json(OUT / "report.json", result)
            write_json(OUT / "run_status.json", {"status": "complete", "phase": phase, "summary": result})
            return result
        except BaseException as exc:
            write_json(OUT / "run_status.json", {"status": "failed_or_paused", "phase": phase, "error": str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "admit", "smoke", "collect"))
    phase = parser.parse_args().phase
    print(json.dumps(prepare() if phase == "prepare" else {"verified": bool(verify())} if phase == "verify" else run(phase), indent=2))
