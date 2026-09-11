"""Output-only CPLNS bridge: fixed small fixtures, no TTF comparison or promotion."""

import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from scripts.verify_cplns_reference import check_archive, fixture_content, parse_output, verify as verify_reference

CONFIG = ROOT / "configs/cplns_observation_bridge_v1.json"
HEADER = ROOT / "experiments/cplns_observer.h"


def insert_once(text, anchor, added):
    if text.count(anchor) != 1:
        raise ValueError("upstream anchor missing or ambiguous: " + anchor[:90])
    return text.replace(anchor, anchor + added, 1)


def instrument(name, text):
    if name not in ("src/InitLNS.cpp", "src/LNS.cpp"):
        return text
    text = '#include "cplns_observer.h"\n' + text
    if name == "src/LNS.cpp":
        text = insert_once(text, "bool LNS::run() {", "\n    cplns_observer::begin();")
        text = insert_once(text, "\n    succ_phase_collision = getInitialSolution();",
            '\n    cplns_observer::state("initial_pp", agents, neighbor.agents, -1, 0, '
            'succ_phase_collision ? 0 : -1, succ_phase_collision ? sum_of_costs : -1);')
        return insert_once(text, "        run_initial_solver();\n        run_collision_lns();",
            '\n        cplns_observer::state("final", agents, std::vector<int>{}, -1, -1, '
            'succ_phase_collision ? 0 : -1, succ_phase_collision ? sum_of_costs : -1, '
            '-1, -1, -1, -1, true);')
    text = insert_once(text, "            bool succ = getInitialSolution();",
        '\n            cplns_observer::state("init", agents, neighbor.agents, num_of_restart, '
        'num_of_iteration, num_of_colliding_pairs, sum_of_costs);')
    text = insert_once(text, "    auto shuffled_agents = neighbor.agents;\n    std::random_shuffle(shuffled_agents.begin(), shuffled_agents.end());",
        '\n    cplns_observer::order(shuffled_agents, num_of_restart, num_of_iteration, '
        'simulated_annealing::sa_iteration_T);')
    text = insert_once(text, "            sum_of_costs += neighbor.sum_of_costs - neighbor.old_sum_of_costs;",
        '\n            cplns_observer::state("step", agents, neighbor.agents, num_of_restart, '
        'num_of_iteration, num_of_colliding_pairs, sum_of_costs, succ, '
        'neighbor.old_colliding_pairs.size(), neighbor.colliding_pairs.size(), '
        'simulated_annealing::sa_iteration_T);')
    return text


def fixture(name, seed):
    if name in ("open", "order_reversal_corridor"):
        grid, scenario = fixture_content(name)
        return grid, scenario, 8
    width = height = 8
    if name not in ("dense_open", "two_door"):
        raise ValueError("unregistered fixture")
    rows = [["." for _ in range(width)] for _ in range(height)]
    if name == "two_door":
        for y in range(height):
            if y not in (2, 5):
                rows[y][3] = "@"
    cells = [y * width + x for y in range(height) for x in range(width) if rows[y][x] == "."]
    rng = random.Random(seed)
    starts, goals = cells[:], cells[:]
    rng.shuffle(starts)
    rng.shuffle(goals)
    count = 60 if name == "dense_open" else 48
    grid = f"type octile\nheight {height}\nwidth {width}\nmap\n" + "\n".join(map("".join, rows)) + "\n"
    scenario = "version 1\n" + "".join(
        f"0\t{name}.map\t{width}\t{height}\t{s % width}\t{s // width}\t{g % width}\t{g // width}\t0\n"
        for s, g in zip(starts[:count], goals[:count]))
    return grid, scenario, count


def prepare():
    ref = verify_reference()
    config = read_json(CONFIG)
    out = ROOT / config["output"]
    if (out / "registry.json").exists() or (out / "build").exists():
        raise ValueError("prepared build or registry exists; do not overwrite a bridge run")
    files = check_archive(ref)
    source = ROOT / ref["source"]
    planned = {}
    for relative in files:
        target = out / "source" / relative
        if relative in ("src/InitLNS.cpp", "src/LNS.cpp"):
            planned[target] = instrument(relative, (source / relative).read_text(encoding="utf-8")).encode("utf-8")
        else:
            planned[target] = (source / relative).read_bytes()
    planned[out / "source/src/cplns_observer.h"] = HEADER.read_bytes()
    for name in config["fixtures"]:
        grid, scen, _ = fixture(name, config["fixture_seed"])
        for suffix, content in ((".map", grid), (".scen", scen)):
            p = out / "fixtures" / (name + suffix)
            planned[p] = content.encode("ascii")
    # Recover a partial preparation only if every existing file is byte-identical.
    for p in out.rglob("*"):
        if p.is_symlink() or p.is_file() and (p not in planned or p.read_bytes() != planned[p]):
            raise ValueError("unexpected partial preparation contents")
    for p, content in planned.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            p.write_bytes(content)
    return {"prepared": True, "jobs_after_build": len(config["fixtures"]) * len(ref["profiles"]) * len(config["seeds"]) * 3}


def inputs(config, ref):
    out = ROOT / config["output"]
    paths = [CONFIG, HEADER, Path(__file__), ROOT / config["reference_config"],
             ROOT / ref["output"] / "registry.json", ROOT / ref["binary"], out / "build/plns"]
    paths += [p for directory in (out / "source", out / "fixtures") for p in directory.rglob("*") if p.is_file()]
    if any(p.is_symlink() for p in paths):
        raise ValueError("symlink in bridge inputs")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def register():
    ref = verify_reference()
    config = read_json(CONFIG)
    out = ROOT / config["output"]
    if (out / "registry.json").exists():
        raise ValueError("registry already exists")
    for rel in check_archive(ref):
        original = ROOT / ref["source"] / rel
        actual = out / "source" / rel
        expected = instrument(rel, original.read_text(encoding="utf-8")).encode() if rel in ("src/InitLNS.cpp", "src/LNS.cpp") else original.read_bytes()
        if actual.read_bytes() != expected:
            raise ValueError("unexpected source edit: " + rel)
    expected_names = set(check_archive(ref)) | {"src/cplns_observer.h"}
    if {p.relative_to(out / "source").as_posix() for p in (out / "source").rglob("*") if p.is_file()} != expected_names:
        raise ValueError("unexpected source file set")
    if (out / "source/src/cplns_observer.h").read_bytes() != HEADER.read_bytes():
        raise ValueError("observer header mismatch")
    registry = {"schema": "lns2.cplns_observer_registry.v1", "inputs": inputs(config, ref),
                "changed_author_files": ["src/InitLNS.cpp", "src/LNS.cpp"], "no_ttf_or_promotion": True}
    write_json(out / "registry.json", registry)
    return {"registered": True, "jobs": len(schedule(config, ref))}


def verify():
    ref = verify_reference()
    config = read_json(CONFIG)
    registry = read_json(ROOT / config["output"] / "registry.json")
    if registry["inputs"] != inputs(config, ref):
        raise ValueError("bridge inputs changed")
    return config, ref


def schedule(config, ref):
    jobs = []
    for name in config["fixtures"]:
        for profile, settings in ref["profiles"].items():
            for seed in config["seeds"]:
                for lane in ("upstream", "observer_off", "observer_on"):
                    base = Path(config["output"]) / "fixtures" / name
                    options = {**ref["common_options"], **settings, "screen": "3", "seed": str(seed),
                               "cutoffTime": str(config["solver_seconds"]), "map": base.with_suffix(".map").as_posix(),
                               "agents": base.with_suffix(".scen").as_posix(),
                               "agentNum": str(fixture(name, config["fixture_seed"])[2])}
                    binary = ref["binary"] if lane == "upstream" else config["output"] + "/build/plns"
                    jobs.append({"job_id": f"{name}-{profile}-{seed}-{lane}", "fixture": name, "profile": profile,
                                 "seed": seed, "lane": lane, "output": config["output"],
                                 "external_seconds": config["external_seconds"],
                                 "argv": [binary, *[v for k, val in options.items() for v in ("--" + k, val)]]})
    return jobs


def run_job(job):
    folder = ROOT / job["output"] / "jobs"
    folder.mkdir(parents=True, exist_ok=True)
    log, events = [folder / (job["job_id"] + suffix) for suffix in (".log", ".events.jsonl")]
    env = dict(os.environ)
    env.pop("CPLNS_OBSERVER_PATH", None)
    if job["lane"] == "observer_on":
        env["CPLNS_OBSERVER_PATH"] = str(events)
    timed_out = False
    with log.open("wb") as stream:
        proc = subprocess.Popen(job["argv"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            proc.wait(timeout=job["external_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except BaseException:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    return {"job_id": job["job_id"], "argv": job["argv"], "returncode": proc.returncode,
            "timed_out": timed_out, "log_sha256": sha256_file(log),
            "events_sha256": sha256_file(events) if events.exists() else None,
            "status": "error" if timed_out or proc.returncode else "complete"}


def verify_job(row, job):
    folder = ROOT / job["output"] / "jobs"
    if row["job_id"] != job["job_id"] or row["argv"] != job["argv"]:
        raise ValueError("job identity mismatch")
    for key, suffix in (("log_sha256", ".log"), ("events_sha256", ".events.jsonl")):
        path = folder / (job["job_id"] + suffix)
        actual = sha256_file(path) if path.exists() else None
        if actual != row[key] or (key == "log_sha256" and actual is None):
            raise ValueError("job output hash mismatch")


def scientific_log(text):
    # Upstream screen=3 already prints these values; strip only wall-clock fields.
    result = []
    for line in text.splitlines():
        if line.startswith(("Neighbors:", "Neighbors_set:", "Old colliding pairs", "New colliding pairs", "Agent ")):
            result.append(line)
        elif line.startswith("After agent "):
            result.append(line.split(", remaining time =")[0])
        elif line.startswith("df "):
            result.append(line)
    return result


def path_metrics(agents, width, rows, expected):
    ids = [a["id"] for a in agents]
    if len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError("invalid agent identities")
    paths = {}
    for a in agents:
        if (a["start"], a["goal"]) != expected[a["id"]]:
            raise ValueError("task mismatch")
        path = a["path"]
        if not path or path[0] != a["start"] or path[-1] != a["goal"]:
            raise ValueError("empty or wrong-endpoint path")
        for p in path:
            if not isinstance(p, int) or not 0 <= p < width * len(rows) or rows[p // width][p % width] in "@TOW":
                raise ValueError("invalid path cell")
        for p, q in zip(path, path[1:]):
            if abs(p // width - q // width) + abs(p % width - q % width) > 1:
                raise ValueError("nonadjacent path step")
        paths[a["id"]] = path
    conflicts = set()
    horizon = max(map(len, paths.values()))
    previous = {}
    for t in range(horizon):
        occupancy, edges = {}, {}
        for aid, path in paths.items():
            cell = path[min(t, len(path) - 1)]
            for other in occupancy.get(cell, ()):
                conflicts.add(tuple(sorted((aid, other))))
            occupancy.setdefault(cell, []).append(aid)
            if t:
                edge = (previous[aid], cell)
                if edge[0] != edge[1]:
                    for other in edges.get((edge[1], edge[0]), ()):
                        conflicts.add(tuple(sorted((aid, other))))
                edges.setdefault(edge, []).append(aid)
            previous[aid] = cell
    return len(conflicts), sum(len(p) - 1 for p in paths.values()), paths


def validate_events(events, grid, scenario):
    lines = grid.splitlines()
    width, rows = int(lines[2].split()[1]), lines[4:]
    expected = {i: (int(v[5]) * width + int(v[4]), int(v[7]) * width + int(v[6]))
                for i, v in enumerate(line.split() for line in scenario.splitlines()[1:])}
    if not events or events[0]["event"] != "initial_pp" or events[-1]["event"] != "final":
        raise ValueError("missing initial/final event")
    previous = None
    previous_conflicts = None
    last_time = -1
    accepted_worse = 0
    pending_order = None
    step_count = 0
    for event in events:
        if event["seconds"] < last_time:
            raise ValueError("nonmonotonic clock")
        last_time = event["seconds"]
        kind = event["event"]
        if kind == "pp_order":
            if pending_order is not None or len(set(event["order"])) != len(event["order"]) or not set(event["order"]) <= set(expected):
                raise ValueError("invalid PP order")
            pending_order = event
            continue
        if kind not in ("initial_pp", "init", "step", "final"):
            raise ValueError("unknown observer event")
        if kind == "initial_pp" and event["conflicts"] == -1:
            # The original outer PP may stop before all agents are planned.
            continue
        conflicts, cost, paths = path_metrics(event["agents"], width, rows, expected)
        if event["conflicts"] >= 0 and conflicts != event["conflicts"]:
            raise ValueError("reconstructed conflict mismatch")
        if event["cost"] >= 0 and cost != event["cost"]:
            raise ValueError("reconstructed SOC mismatch")
        if kind == "step":
            if previous is None or pending_order is None or set(pending_order["order"]) != set(event["selected"]):
                raise ValueError("missing or mismatched PP order/state")
            if (pending_order["restart"], pending_order["iteration"]) != (event["restart"], event["iteration"]):
                raise ValueError("step identity mismatch")
            selected = set(event["selected"])
            if any(paths[aid] != previous[aid] for aid in paths if aid not in selected):
                raise ValueError("external path changed")
            if event["accepted"] == 0 and paths != previous:
                raise ValueError("rollback mismatch")
            if event["accepted"] not in (0, 1):
                raise ValueError("invalid acceptance result")
            expected_conflicts = previous_conflicts + (event["new_pairs"] - event["old_pairs"] if event["accepted"] else 0)
            if conflicts != expected_conflicts:
                raise ValueError("accepted conflict delta mismatch")
            accepted_worse += int(event["accepted"] == 1 and event["new_pairs"] > event["old_pairs"])
            pending_order = None
            step_count += 1
        previous = paths
        previous_conflicts = conflicts
    if pending_order is not None:
        raise ValueError("incomplete observed repair")
    return {"states_checked": sum(e["event"] in ("init", "step", "final") for e in events),
            "steps_checked": step_count, "accepted_worse": accepted_worse,
            "final_conflicts": conflicts, "final_soc": cost,
            "observed_prefix_truncated": events[-1]["decisions_seen"] > 128,
            "first_observed_feasible_seconds": next((e["seconds"] for e in events if e.get("conflicts") == 0), None)}


def collect():
    config, ref = verify()
    out = ROOT / config["output"]
    fingerprint = sha256_file(out / "registry.json")
    with _CollectionRunLock(out, fingerprint, "observation"):
        pending = []
        for job in schedule(config, ref):
            file = out / "jobs" / (job["job_id"] + ".json")
            if file.exists():
                previous = read_json(file)
                verify_job(previous, job)
                if previous["status"] != "complete":
                    raise ValueError("previous failure requires inspection; no automatic retry")
            else:
                pending.append(job)
        def save(row):
            write_json(out / "jobs" / (row["job_id"] + ".json"), row)
            print(row["job_id"], row["status"], flush=True)
        write_json(out / "run_status.json", {"status": "running", "pending": len(pending)})
        try:
            _run_jobs(run_job, pending, config["workers"], phase="observation", output_root=out / "jobs",
                      run_fingerprint=fingerprint, timeout_seconds=config["external_seconds"] + 10,
                      on_result=save, failure_result=lambda j, s, e: {"job_id": j["job_id"], "status": s, "error": e},
                      stop_on_failure=True)
            result = report()
            write_json(out / "run_status.json", {"status": "complete", "decision": result["decision"]})
            return result
        except BaseException as exc:
            write_json(out / "run_status.json", {"status": "interrupted_or_failed", "error": str(exc)})
            raise


def report():
    config, ref = verify()
    out = ROOT / config["output"]
    grouped, diagnostics = {}, []
    for job in schedule(config, ref):
        row = read_json(out / "jobs" / (job["job_id"] + ".json"))
        verify_job(row, job)
        text = (out / "jobs" / (job["job_id"] + ".log")).read_text(encoding="utf-8", errors="replace")
        parsed = parse_output(text, row["returncode"], row["timed_out"])
        if not parsed["valid"]:
            raise ValueError("invalid upstream output: " + job["job_id"])
        key = (job["fixture"], job["profile"], job["seed"])
        grouped.setdefault(key, {})[job["lane"]] = (scientific_log(text), parsed)
        if job["lane"] == "observer_on":
            events = [json.loads(line) for line in (out / "jobs" / (job["job_id"] + ".events.jsonl")).read_text().splitlines()]
            grid, scenario, _ = fixture(job["fixture"], config["fixture_seed"])
            checked = validate_events(events, grid, scenario)
            if checked["final_conflicts"] != parsed["conflicts"]:
                raise ValueError("final log and exported paths disagree")
            diagnostics.append({"job_id": job["job_id"], **checked})
    comparisons = []
    for key, lanes in grouped.items():
        base, base_result = lanes["upstream"]
        for lane in ("observer_off", "observer_on"):
            candidate, result = lanes[lane]
            exact = base_result["conflicts"] == 0 and result["conflicts"] == 0
            # A wall-time interruption can truncate the last low-level call; compare a bounded prefix only.
            count = len(base) if exact else max(0, min(256, len(base), len(candidate)) - 8)
            equivalent = base == candidate if exact else count > 0 and base[:count] == candidate[:count]
            if exact:
                equivalent = equivalent and base_result["soc"] == result["soc"]
            comparisons.append({"fixture": key[0], "profile": key[1], "seed": key[2], "lane": lane,
                                "complete_solved_sequence": exact, "compared_log_records": count, "equivalent": equivalent})
    equivalence = all(c["equivalent"] for c in comparisons)
    acceptance = sum(d["accepted_worse"] for d in diagnostics)
    result = {"schema": "lns2.cplns_observation_report.v1", "jobs": len(schedule(config, ref)),
              "comparisons": comparisons, "diagnostics": diagnostics, "equivalence_passed": equivalence,
              "positive_delta_accepts_observed": acceptance, "independent_path_validation": True,
              "decision": "bounded_observation_pass" if equivalence and acceptance else "investigate_before_real_cases",
              "no_ttf_or_promotion": True, "registry_sha256": sha256_file(out / "registry.json")}
    write_json(out / "report.json", result)
    return {k: v for k, v in result.items() if k not in ("comparisons", "diagnostics")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "register", "verify", "collect", "report", "dry-run"))
    args = parser.parse_args()
    if args.phase == "verify":
        verify()
        value = {"verified": True}
    elif args.phase == "dry-run":
        config, ref = verify()
        value = {"jobs": len(schedule(config, ref)), "workers": config["workers"], "solver_seconds": config["solver_seconds"], "no_ttf": True}
    else:
        value = {"prepare": prepare, "register": register, "collect": collect, "report": report}[args.phase]()
    print(json.dumps(value, indent=2))
