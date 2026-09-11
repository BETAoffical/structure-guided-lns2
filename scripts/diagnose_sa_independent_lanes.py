"""Fixed-budget independent search lanes; diagnostic effort, never TTF."""

import argparse
import ctypes
import json
import multiprocessing
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _plain, _run_jobs, state_fingerprint
from experiments.nonmonotonic_repair import validate_transition
from scripts import diagnose_sa_selector_rounds as old
from scripts.run_feedback_exploration_diagnostics import FrozenPool, check_source_pool, restore, validate_final

OUT = ROOT / "build/sa-independent-lanes-v1"
ARMS = ("standard", "rank_sa", "uniform_greedy", "portfolio")
LANES = ("rank_sa", "uniform_greedy")
SEEDS = {"round1": 19, "round2": 23}
HORIZON, WORKERS = 256, 20


def digest(value):
    return old.transfer.source.digest(value)


def schedule_index(turn):
    if turn < 0:
        raise ValueError("negative turn")
    return turn % 2


def trace_portfolio(a, b, budget=HORIZON):
    if a["initial_fingerprint"] != b["initial_fingerprint"] or a["censored"] or b["censored"]:
        raise ValueError("unpaired/censored source")
    used = [0, 0]
    for turn in range(budget):
        if any(r["feasible"] and not r["transitions"] for r in (a, b)):
            return {"feasible": True, "repair_calls": 0, "used": used}
        i = schedule_index(turn)
        r = (a, b)[i]
        used[i] += 1
        if used[i] > len(r["transitions"]):
            raise ValueError("source prefix exhausted")
        if r["feasible"] and used[i] == len(r["transitions"]):
            return {"feasible": True, "repair_calls": turn + 1, "used": used}
    return {"feasible": False, "repair_calls": budget, "used": used}


def prepare():
    source = old.verify()
    if OUT.exists():
        raise ValueError("output exists")
    used = {c["task_id"] for c in source["cases"]}
    chosen = {}
    for item in old.task_plan(source["config"]):
        row = item["row"]
        if row["task_id"] not in used:
            chosen.setdefault(row["map_id"], row)
    if len(chosen) != 8:
        raise ValueError("eight unused tasks required")
    inputs = dict(source["inputs"])
    for p in (Path(__file__), ROOT / "docs/SA_INDEPENDENT_LANES_PROTOCOL_ZH.md", old.OUT / "plan.json"):
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    cases = []
    for phase, seed in SEEDS.items():
        for index, row in enumerate(chosen.values()):
            base = ROOT / source["config"]["dataset"]["output"] / row["split"]
            for key in ("map_file", "map_metadata_file", "task_file", "scenario_file"):
                path = base / row[key]
                inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
            task, grid = read_json(base / row["task_file"]), read_json(base / row["map_metadata_file"])
            cols = grid["cols"]
            cases.append({"case_id": f"independent-{index:02d}-native-{seed}", "map_id": row["map_id"],
                "task_id": row["task_id"], "source_seed": seed, "solver_seed": seed, "restore_seed": seed,
                "decision": 0, "prefix": [], "phase": phase, "instance": digest([row["task_id"], seed]),
                "proposal": source["cases"][0]["proposal"],
                "expected_grid": {"rows": grid["rows"], "cols": cols,
                                  "obstacles": [c != "." for line in grid["grid"] for c in line]},
                "expected_endpoints": [[a[0] * cols + a[1], b[0] * cols + b[1]] for a, b in zip(task["starts"], task["goals"])]})
    exploratory = []
    for phase in ("round2", "round3"):
        report = read_json(old.OUT / (phase + "_report.json"))
        for p in sorted((old.OUT / phase).glob("*-rank_sa.json")):
            a, bpath = read_json(p), p.with_name(p.name.replace("rank_sa", "uniform_greedy"))
            b = read_json(bpath)
            for path in (p, bpath):
                if sha256_file(path) != report["outcome_sha256"][path.stem]:
                    raise ValueError("source outcome changed")
                inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
            exploratory.append({"case_id": a["case_id"], **trace_portfolio(a, b),
                                "uniform_feasible": b["feasible"], "uniform_calls": len(b["transitions"])})
    if len(exploratory) != 20:
        raise ValueError("source coverage mismatch")
    gate = (sum(r["feasible"] for r in exploratory) >= sum(r["uniform_feasible"] for r in exploratory)
            and sum(r["repair_calls"] for r in exploratory) <= .95 * sum(r["uniform_calls"] for r in exploratory))
    smoke = []
    for name in ("repair-confirm-04-native-11", "repair-confirm-09-native-17"):
        path = old.OUT / "admission" / (name + ".json")
        inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        smoke.append(read_json(path)["case"])
    plan = {"schema": "lns2.sa_independent_lanes.v1", "config": source["config"], "inputs": inputs,
            "arms": list(ARMS), "lanes": list(LANES), "seeds": SEEDS, "horizon": HORIZON, "workers": WORKERS,
            "cases": cases, "smoke": smoke, "exploratory": exploratory, "exploratory_gate": gate,
            "no_ttf_or_promotion": True}
    write_json(OUT / "plan.json", plan)
    return {"exploratory_gate": gate, "admissions": 16, "smoke": 10, "new_jobs": 64,
            "maximum_new_repairs": 16384, "not_independent_map_confirmation": True}


def verify():
    plan = read_json(OUT / "plan.json")
    for path, sha in plan["inputs"].items():
        if sha256_file(ROOT / path) != sha:
            raise ValueError("frozen input changed: " + path)
    if (plan["arms"], plan["lanes"], plan["seeds"], plan["horizon"], plan["workers"]) != (
            list(ARMS), list(LANES), SEEDS, HORIZON, WORKERS) or not plan["exploratory_gate"]:
        raise ValueError("protocol mismatch or exploratory gate failed")
    return plan


class Lane:
    def __init__(self, job, arm):
        self.job, self.arm = job, arm
        self.env, self.state = restore(job)
        self.initial = self.state
        self.selector = None if self.state["feasible"] else FrozenPool(job["case"])
        self.events, self.conflicts, self.censored = [], [self.state["num_of_colliding_pairs"]], False

    def step(self):
        if self.state["feasible"] or self.censored:
            raise ValueError("step on finished lane")
        d, case = len(self.events), self.job["case"]
        best, pool = self.selector.select(self.env, self.state, d)
        if not d:
            check_source_pool(case, best, pool)
        candidate = pool[old.choose_index(self.arm, case["case_id"], d, best, pool)]
        action = {"mode": "explicit_neighborhood", "agents": candidate["agents"],
                  "random_seed": old.pilot.seed(case["case_id"], 0, d, "pp")}
        temp = old.pilot.temperature(d)
        draw = old.pilot.acceptance_draw(old.pilot.seed(case["case_id"], 0, d, "accept"))
        before, mode = self.state, old.acceptance_arm(self.arm)
        step = _plain(self.env.step_with_time_limit(action, old.pilot.PP_SECONDS) if mode == "standard" else
                      self.env.step_experimental_pp(action, old.pilot.PP_SECONDS, mode, temp, draw))
        self.state, m = step["observation"], step["metrics"]
        validate_transition(before, self.state, m, candidate["agents"], mode, temp, draw)
        validate_final(self.state)
        event = {"action": action, "metrics": m, "candidate_pool": pool, "selected_id": candidate["candidate_id"],
                 "rank_best_id": pool[best]["candidate_id"], "temperature": temp, "uniform": draw,
                 "before_fingerprint": state_fingerprint(before), "after_fingerprint": state_fingerprint(self.state),
                 "generated_delta": self.state["low_level"]["generated"] - before["low_level"]["generated"]}
        self.events.append(event)
        self.conflicts.append(self.state["num_of_colliding_pairs"])
        self.censored = m["pp_failure_reason"] == "time_limit" or step["truncated"]
        return {"feasible": self.state["feasible"], "censored": self.censored,
                "after_fingerprint": event["after_fingerprint"]}

    def result(self):
        return {"arm": self.arm, "initial_fingerprint": state_fingerprint(self.initial),
                "final_state": self.state, "transitions": self.events, "conflicts": self.conflicts,
                "feasible": self.state["feasible"], "censored": self.censored,
                "generated": self.state["low_level"]["generated"] - self.initial["low_level"]["generated"]}


def parent_death_guard(parent_pid):
    # The outer collector can kill a timed-out worker; do not orphan its native lanes.
    if sys.platform != "linux":
        raise RuntimeError("independent native lanes require Linux parent-death protection")
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_PDEATHSIG failed")
    if os.getppid() != parent_pid:
        raise RuntimeError("lane parent exited during spawn")


def lane_server(connection, job, arm, parent_pid):
    try:
        parent_death_guard(parent_pid)
        lane = Lane(job, arm)
        connection.send({"ready": state_fingerprint(lane.initial), "feasible": lane.state["feasible"]})
        while True:
            command = connection.recv()
            if command == "step":
                connection.send(lane.step())
            elif command == "finish":
                connection.send(lane.result())
                break
            else:
                raise ValueError("invalid lane command")
    except BaseException:
        try:
            connection.send({"error": traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def receive(connection, seconds):
    if not connection.poll(seconds):
        raise TimeoutError("lane IPC timeout")
    value = connection.recv()
    if "error" in value:
        raise RuntimeError(value["error"])
    return value


def stop_children(children):
    for process, connection in children:
        connection.close()
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        if process.is_alive():
            raise RuntimeError("native lane did not terminate")


def worker(job):
    began = time.monotonic()
    children, schedule, censored = [], [], False
    try:
        if job["arm"] == "portfolio":
            context = multiprocessing.get_context("spawn")
            ready = []
            for arm in LANES:
                parent, child = context.Pipe()
                process = context.Process(target=lane_server, args=(child, job, arm, os.getpid()), daemon=True)
                process.start()
                child.close()
                children.append((process, parent))
            for _, connection in children:
                ready.append(receive(connection, 30))
            if any(r["ready"] != state_fingerprint(job["case"]["state"]) for r in ready):
                raise ValueError("lane initialization mismatch")
            feasible = any(r["feasible"] for r in ready)
            for turn in range(job["horizon"]):
                if feasible:
                    break
                if time.monotonic() - began >= old.pilot.EPISODE_SECONDS - old.pilot.PP_SECONDS:
                    censored = True
                    break
                i = schedule_index(turn)
                connection = children[i][1]
                connection.send("step")
                response = receive(connection, 30)
                schedule.append(LANES[i])
                feasible, censored = response["feasible"], response["censored"]
                if censored:
                    break
            lanes = {}
            for arm, (_, connection) in zip(LANES, children):
                connection.send("finish")
                lanes[arm] = receive(connection, 30)
        else:
            lane = Lane(job, job["arm"])
            for _ in range(job["horizon"]):
                if lane.state["feasible"]:
                    break
                if time.monotonic() - began >= old.pilot.EPISODE_SECONDS - old.pilot.PP_SECONDS:
                    censored = True
                    break
                response = lane.step()
                schedule.append(job["arm"])
                censored = response["censored"]
                if censored:
                    break
            lanes = {job["arm"]: lane.result()}
        return {"status": "ok", "job_id": job["job_id"], "plan_sha256": job["plan_sha256"],
                "case_id": job["case"]["case_id"], "map_id": job["case"]["map_id"], "arm": job["arm"],
                "horizon": job["horizon"], "schedule": schedule, "lanes": lanes,
                "feasible": any(l["feasible"] for l in lanes.values()), "censored": censored,
                "repair_calls": len(schedule), "generated": sum(l["generated"] for l in lanes.values()),
                "diagnostic_wall_seconds": time.monotonic() - began}
    finally:
        stop_children(children)


def validate_result(row, job):
    if row["status"] != "ok" or any(row[k] != job[k] for k in ("job_id", "plan_sha256", "arm", "horizon")):
        raise ValueError("result identity/error")
    lanes = LANES if job["arm"] == "portfolio" else (job["arm"],)
    if set(row["lanes"]) != set(lanes) or row["case_id"] != job["case"]["case_id"] or row["map_id"] != job["case"]["map_id"]:
        raise ValueError("lane/task identity mismatch")
    expected_schedule = [LANES[schedule_index(t)] if job["arm"] == "portfolio" else job["arm"] for t in range(row["repair_calls"])]
    if row["schedule"] != expected_schedule or row["repair_calls"] > job["horizon"]:
        raise ValueError("budget/schedule mismatch")
    for arm, lane in row["lanes"].items():
        events = lane["transitions"]
        if len(events) != row["schedule"].count(arm) or len(lane["conflicts"]) != len(events) + 1:
            raise ValueError("prefix size mismatch")
        previous = state_fingerprint(job["case"]["state"])
        if lane["initial_fingerprint"] != previous or lane["conflicts"][0] != job["case"]["state"]["num_of_colliding_pairs"]:
            raise ValueError("initial state mismatch")
        for d, event in enumerate(events):
            pool = event["candidate_pool"]
            best = next(i for i, c in enumerate(pool) if c["candidate_id"] == event["rank_best_id"])
            if pool[best]["score"] < max(c["score"] for c in pool) - 1e-10:
                raise ValueError("model maximum mismatch")
            selected = pool[old.choose_index(arm, job["case"]["case_id"], d, best, pool)]
            m = event["metrics"]
            if (event["before_fingerprint"] != previous or selected["candidate_id"] != event["selected_id"]
                    or selected["agents"] != event["action"]["agents"] or not m["action_valid"]
                    or sorted(m["neighborhood"]) != sorted(selected["agents"])
                    or event["action"]["random_seed"] != old.pilot.seed(job["case"]["case_id"], 0, d, "pp")
                    or [m["conflicts_before"], m["conflicts_after"]] != lane["conflicts"][d:d + 2]):
                raise ValueError("invalid action/chain")
            previous = event["after_fingerprint"]
        validate_final(lane["final_state"])
        if (previous != state_fingerprint(lane["final_state"]) or lane["feasible"] != lane["final_state"]["feasible"]
                or lane["generated"] != sum(e["generated_delta"] for e in events)
                or lane["generated"] != lane["final_state"]["low_level"]["generated"] - job["case"]["state"]["low_level"]["generated"]):
            raise ValueError("final state/node accounting mismatch")
    if row["feasible"] != any(l["feasible"] for l in row["lanes"].values()) or row["generated"] != sum(l["generated"] for l in row["lanes"].values()):
        raise ValueError("aggregate mismatch")
    if not row["feasible"] and not row["censored"] and row["repair_calls"] != job["horizon"]:
        raise ValueError("unexplained early stop")
    return row


def load_result(path, job):
    row = read_json(path)
    integrity = row.pop("integrity_sha256", None)
    if digest(row) != integrity:
        raise ValueError("result integrity mismatch")
    return validate_result(row, job)


def execute(phase, jobs, function=worker, loader=load_result):
    directory, began = OUT / phase, time.monotonic()
    pending = []
    for job in jobs:
        path = directory / (job["job_id"] + ".json")
        if path.exists():
            loader(path, job)
        else:
            pending.append(job)
    def save(row):
        if function == worker and row["status"] == "ok":
            row["integrity_sha256"] = digest(row)
        write_json(directory / (row["job_id"] + ".json"), row)
        print(phase, row["job_id"], row["status"], flush=True)
    for offset in range(0, len(pending), WORKERS):
        if (OUT / "STOP").exists() or time.monotonic() - began + old.pilot.FUSE_SECONDS >= old.PHASE_SECONDS:
            raise InterruptedError("safe stop or phase cap")
        _run_jobs(function, pending[offset:offset + WORKERS], WORKERS, phase=phase, output_root=directory,
                  run_fingerprint=sha256_file(OUT / "plan.json"), timeout_seconds=old.pilot.FUSE_SECONDS,
                  on_result=save, failure_result=lambda j, s, e: {"job_id": j["job_id"], "status": s, "error": e}, stop_on_failure=True)
    return [loader(directory / (j["job_id"] + ".json"), j) for j in jobs]


def jobs(plan, cases, horizon=HORIZON):
    return [{"job_id": c["case_id"] + "-" + arm, "case": c, "config": plan["config"],
             "plan_sha256": sha256_file(OUT / "plan.json"), "arm": arm, "horizon": horizon, "trial": 0}
            for c in cases for arm in ARMS]


def summary(rows):
    tables = {a: {r["case_id"]: r for r in rows if r["arm"] == a} for a in ARMS}
    keys = set(tables["standard"])
    if not keys or len(rows) != len(keys) * len(ARMS) or any(set(t) != keys for t in tables.values()):
        raise ValueError("unpaired report")
    totals = {a: {"success": sum(r["feasible"] for r in t.values()), "episodes": len(t),
                   "repair_calls": sum(r["repair_calls"] for r in t.values()), "generated": sum(r["generated"] for r in t.values()),
                   "censored": sum(r["censored"] for r in t.values())} for a, t in tables.items()}
    pairs = {a: {"gains": sorted(k for k in keys if tables["portfolio"][k]["feasible"] and not tables[a][k]["feasible"]),
                 "losses": sorted(k for k in keys if not tables["portfolio"][k]["feasible"] and tables[a][k]["feasible"])} for a in ARMS[:-1]}
    mismatches, failure_causes = [], []
    for k in sorted(keys):
        for arm in LANES:
            actual = tables["portfolio"][k]["lanes"][arm]
            solo = tables[arm][k]["lanes"][arm]
            if actual["initial_fingerprint"] != solo["initial_fingerprint"]:
                raise ValueError("pair initialization differs")
            for d, event in enumerate(actual["transitions"]):
                if d >= len(solo["transitions"]) or scientific_event(event) != scientific_event(solo["transitions"][d]):
                    mismatches.append([k, arm, d])
        if not tables["portfolio"][k]["feasible"]:
            solved = [a for a in LANES if tables[a][k]["feasible"]]
            failure_causes.append({"case_id": k, "cause": "budget_split" if solved else "both_single_lanes_failed",
                                   "solo_success": solved})
    gate = {"integrity": not mismatches and not any(t["censored"] for t in totals.values()),
            "no_success_losses": all(not p["losses"] for p in pairs.values()),
            "new_success_over_rank_sa": bool(pairs["rank_sa"]["gains"]),
            "calls_reduce_5pct_vs_uniform": totals["portfolio"]["repair_calls"] <= .95 * totals["uniform_greedy"]["repair_calls"],
            "nodes_within_110pct_uniform": totals["portfolio"]["generated"] <= 1.1 * totals["uniform_greedy"]["generated"]}
    return {"arms": totals, "portfolio_pairs": pairs, "prefix_mismatches": mismatches,
            "failure_causes": failure_causes, "gates": gate, "passed": all(gate.values()),
            "no_ttf_or_promotion": True,
            "per_case": {k: {a: {v: tables[a][k][v] for v in ("feasible", "repair_calls", "generated", "map_id")} for a in ARMS} for k in sorted(keys)}}


def scientific_event(event):
    fields = ("action", "candidate_pool", "selected_id", "rank_best_id", "temperature", "uniform", "before_fingerprint", "after_fingerprint")
    metrics = ("repair_order", "replan_success", "pp_attempt_conflict_pair_count", "pp_old_conflict_pair_count", "acceptance_probability", "pp_failure_reason")
    return {k: event[k] for k in fields} | {"metrics": {k: event["metrics"][k] for k in metrics}}


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
            admission_jobs = [{"job_id": c["case_id"], "case": c, "config": plan["config"],
                               "plan_sha256": sha256_file(OUT / "plan.json")} for c in plan["cases"]]
            if phase == "admission":
                rows = execute(phase, admission_jobs, old.admit, old.load_admission)
                result = {"admitted": len(rows), "zero_conflict": sum(r["case"]["state"]["feasible"] for r in rows)}
            elif phase == "smoke":
                smoke_jobs = jobs(plan, plan["smoke"], 48)
                rows = execute(phase, smoke_jobs)
                references = [j for j in smoke_jobs if j["arm"] == "rank_sa"]
                refrows = execute("reference", references, old.worker, old.load_result)
                for reference in refrows:
                    actual = next(r for r in rows if r["job_id"] == reference["job_id"])["lanes"]["rank_sa"]
                    if [scientific_event(e) for e in actual["transitions"]] != [scientific_event(e) for e in reference["transitions"]]:
                        raise ValueError("new lane implementation changed frozen worker")
                result = summary(rows)
                if not result["gates"]["integrity"]:
                    raise ValueError("smoke integrity failed")
            else:
                if not read_json(OUT / "smoke_report.json")["gates"]["integrity"]:
                    raise ValueError("smoke required")
                if phase == "round2" and not read_json(OUT / "round1_report.json")["gates"]["integrity"]:
                    raise ValueError("first round integrity required")
                cases = [old.load_admission(OUT / "admission" / (j["job_id"] + ".json"), j)["case"] for j in admission_jobs]
                rows = execute(phase, jobs(plan, [c for c in cases if c["phase"] == phase]))
                result = summary(rows)
            result["plan_sha256"] = sha256_file(OUT / "plan.json")
            result["outcome_sha256"] = {p.name: sha256_file(p) for p in sorted((OUT / phase).glob("*.json")) if p.name != "collection_progress.json"}
            write_json(OUT / (phase + "_report.json"), result)
            backup(phase)
            write_json(OUT / "run_status.json", {"phase": phase, "status": "complete", "summary": result})
            return result
        except BaseException as exc:
            write_json(OUT / "run_status.json", {"phase": phase, "status": "failed_or_paused", "error": str(exc)})
            raise


def analyze():
    plan = verify()
    rows = []
    for phase in SEEDS:
        report = read_json(OUT / (phase + "_report.json"))
        for name, sha in report["outcome_sha256"].items():
            if sha256_file(OUT / phase / name) != sha:
                raise ValueError("registered result changed")
        cases = [read_json(OUT / "admission" / (c["case_id"] + ".json"))["case"] for c in plan["cases"] if c["phase"] == phase]
        batch = [load_result(OUT / phase / (j["job_id"] + ".json"), j) for j in jobs(plan, cases)]
        recalculated = summary(batch)
        if any(report[k] != value for k, value in recalculated.items()):
            raise ValueError("report does not reproduce")
        backup(phase)
        rows.extend(batch)
    result = summary(rows)
    result["plan_sha256"] = sha256_file(OUT / "plan.json")
    result["source_reports"] = {p: sha256_file(OUT / (p + "_report.json")) for p in SEEDS}
    result["decision"] = "mechanism_gate_only" if result["passed"] else "stop_portfolio_extension"
    write_json(OUT / "analysis.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "admission", "smoke", "round1", "round2", "analyze"))
    phase = parser.parse_args().phase
    print(json.dumps(prepare() if phase == "prepare" else {"verified": bool(verify())} if phase == "verify"
                     else analyze() if phase == "analyze" else run(phase), indent=2))
