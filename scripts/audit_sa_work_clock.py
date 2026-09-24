"""Read-only node-budget/cooling exposure audit; no solver or training calls."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as io
from scripts.audit_sa_cooling_traces import check_acceptance
from scripts import run_sa_path_quality as q
from scripts.audit_sa_history_information import atomic

SOURCE = ROOT / "build/sa-second-onpolicy-batch-v1"
OUT = ROOT / "build/sa-work-clock-audit-v1"
SOURCE_SHA = "869ae98b2fbbd3b7d9f1aa014f46576a11b8e9599fd118d91b206d4cbea628c1"
ARMS = ("untrained_exploration", "uncapped_condition")
FRACTIONS = (25, 50, 75, 90, 100)
BUDGET = 25000000
require = io.require


def describe(values):
    values = sorted(values)
    return dict(n=len(values), minimum=values[0] if values else None,
                median=statistics.median(values) if values else None,
                mean=statistics.mean(values) if values else None,
                maximum=values[-1] if values else None)


def milestones(steps, success, generated, budget=BUDGET):
    """Use actual crossing actions, not an interpolated or future temperature."""
    result = {}
    for pct in FRACTIONS:
        threshold = budget * pct // 100
        crossing = next((s for s in steps if s["nodes_before"] < threshold <= s["nodes_after"]), None)
        if crossing:
            result[str(pct)] = dict(crossing, threshold=threshold,
                status="success_on_crossing" if crossing["after"] == 0 else "crossed",
                overshoot=crossing["nodes_after"] - threshold)
        else:
            require(success and generated < threshold, "missing work crossing in unfinished episode")
            result[str(pct)] = dict(status="already_feasible", threshold=threshold,
                                    nodes_at_success=generated, temperature=None)
    return result


def work_bin(nodes, budget=BUDGET):
    require(0 <= nodes < budget, "decision started after work budget")
    return next(str(pct) for pct in FRACTIONS if nodes < budget * pct // 100)


def paired(rows):
    groups = defaultdict(dict)
    for row in rows:
        require(row["split"] == "train" and row["arm"] in ARMS, "non-Train or unknown arm")
        key = (row["pair_id"], row["replica"])
        require(row["arm"] not in groups[key], "duplicate paired arm")
        groups[key][row["arm"]] = row
    result = []
    for key, group in sorted(groups.items()):
        require(set(group) == set(ARMS), "missing paired arm")
        a, b = (group[arm] for arm in ARMS)
        for field in ("map_id", "task_id", "solver_seed", "initial_fingerprint", "rng_stream_id"):
            require(a[field] == b[field], "paired identity mismatch: " + field)
        result.append((a, b))
    return result


def prepare():
    require(not OUT.exists(), "audit already exists; verify or resume")
    require(io.sha256_file(SOURCE / "report.json") == SOURCE_SHA, "source report changed")
    reg, report, audit, complete = (io.check_seal(io.read_json(SOURCE / name)) for name in
        ("registration.json", "report.json", "audit.json", "collection.complete.json"))
    require(reg["binding"] == io.json_fingerprint({k:v for k,v in reg.items() if k not in ("binding", "integrity")}), "source registration hash")
    require(all(r["binding"] == reg["binding"] for r in (report, audit, complete)), "source identity")
    require(report["no_ttf"] and report["no_heldout"] and report["no_update"], "source scope")
    require(reg["config"]["node_budget"] == BUDGET and reg["config"]["max_decisions"] is None, "source budget")
    require(report["audit_sha256"] == io.sha256_file(SOURCE / "audit.json"), "source audit changed")
    jobs = reg["jobs"]
    require(len(jobs) == len({j["job_id"] for j in jobs}) == 192, "source job count")
    old = {r["job_id"]: r for r in report["episodes"]}
    checked = {r["job_id"]: r for r in audit["results"]}
    ids = {j["job_id"] for j in jobs}
    require(ids == set(old) == set(checked) == set(complete["files"]), "source missing jobs")
    require(len(old) == len(report["episodes"]) and len(checked) == len(audit["results"]), "duplicate source results")
    inputs, selected = {}, []
    for job in jobs:
        require(job["split"] == "train", "sealed evaluation data forbidden")
        folder = SOURCE / "runs" / job["comparison_arm"] / job["phase"] / job["job_id"]
        row = io.result_read(folder, {"binding":reg["scientific_binding"]})
        require(row["execution_binding"] == reg["binding"] and row["status"] == "ok", "source execution")
        require(dict(row, comparison_arm=job["comparison_arm"]) == old[job["job_id"]], "source report row")
        digest = io.sha256_file(folder / "result.json")
        require(digest == checked[job["job_id"]]["result_sha256"] == complete["files"][job["job_id"]], "source result SHA")
        require(checked[job["job_id"]]["status"] == "ok", "failed source audit")
        require(row["stop"] in ("feasible", "node_budget"), "censored source")
        for name in (*row["files"], "result.json"):
            path = folder / name
            inputs[path.relative_to(ROOT).as_posix()] = io.sha256_file(path)
        selected.append(dict(job_id=job["job_id"], arm=job["comparison_arm"], split=job["split"],
            pair_id=job["pair_id"], replica=job["replica"], map_id=job["case"]["map_id"],
            task_id=job["case"]["task_id"], task_variant=job["case"]["task_variant"], solver_seed=job["solver_seed"],
            folder=folder.relative_to(ROOT).as_posix(), result=row,
            initial_fingerprint=row["initial_fingerprint"], rng_stream_id=row["rng_stream_id"],
            expected_noops=checked[job["job_id"]]["legal_noops"]))
    require(len(paired(selected)) == 96 and len({j["map_id"] for j in selected}) == 6, "pair/map count")
    paths = [SOURCE / n for n in ("registration.json", "report.json", "audit.json", "collection.complete.json")]
    paths += [ROOT / n for n in ("scripts/audit_sa_work_clock.py", "tests/evaluation/test_sa_work_clock.py",
        "docs/SA_WORK_CLOCK_PROTOCOL_ZH.md", "scripts/audit_sa_cooling_traces.py",
        "experiments/nonmonotonic_repair.py", "experiments/closed_loop_trace_storage.py",
        "build/linux/sa-wall-clock-v1/lns2_env.cpython-310-x86_64-linux-gnu.so")]
    require(io.sha256_file(paths[-1]) == "5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d", "frozen native changed")
    for model in reg["config"]["models"].values():
        require(io.sha256_file(ROOT / model["path"]) == model["file_sha256"], "frozen actor changed")
        paths.append(ROOT / model["path"])
    paths += [ROOT / n for n in ("scripts/run_sa_onpolicy.py", "scripts/run_sa_paired_closed_loop.py",
        "scripts/run_sa_path_quality.py", "scripts/audit_sa_history_information.py", "experiments/_common.py")]
    for path in paths:
        inputs[path.relative_to(ROOT).as_posix()] = io.sha256_file(path)
    plan = dict(schema="lns2.sa_work_clock.v1", source_report_sha256=SOURCE_SHA,
        inputs=inputs, jobs=selected, node_budget=BUDGET, fractions=list(FRACTIONS), workers=20,
        no_solver=True, no_training=True, no_counterfactual=True, no_ttf=True,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = io.json_fingerprint(plan)
    io.once(OUT / "audit_registration.json", io.sealed(plan))
    return dict(jobs=len(selected), paired_episodes=96, maps=6, no_solver=True)


def verify_inputs(plan):
    for name, digest in plan["inputs"].items():
        require(io.sha256_file(ROOT / name) == digest, "registered input changed: " + name)


def verify():
    plan = io.check_seal(io.read_json(OUT / "audit_registration.json"))
    require(plan["binding"] == io.json_fingerprint({k:v for k,v in plan.items() if k not in ("binding", "integrity")}), "audit binding")
    verify_inputs(plan)
    completed = OUT / "audit.complete.json"
    if completed.exists():
        done = io.check_seal(io.read_json(completed))
        require(done["binding"] == plan["binding"] and done["episodes"] == 192 and
            done["report_sha256"] == io.sha256_file(OUT / "report.json"), "audit completion changed")
        report = io.check_seal(io.read_json(OUT / "report.json"))
        rows = [io.check_seal(io.read_json(OUT / "episodes" / (j["job_id"] + ".json"))) for j in plan["jobs"]]
        require(report["binding"] == plan["binding"] and report["episodes"] == rows and
                report["summary"] == aggregate(rows), "audit output changed")
    return plan


def episode(job):
    folder = ROOT / job["folder"]
    row = job["result"]
    for name, digest in row["files"].items():
        require(io.sha256_file(folder / name) == digest, "episode changed")
    state = io.read_json(folder / "initial.json")
    initial_nodes = state["low_level"]["generated"]
    require(q.state_fingerprint(state) == row["initial_fingerprint"], "initial identity")
    steps, bins, sizes = [], {}, {}
    for d, event in enumerate(io.trace_read(folder)):
        before = state["num_of_colliding_pairs"]
        used = state["low_level"]["generated"] - initial_nodes
        tag = work_bin(used)
        require(before > 0 and event["decision"] == d and event["before"] == q.state_fingerprint(state), "trace discontinuity")
        require(event["policy_sha256"] == row["policy_sha256"], "trace actor changed")
        chosen = next(c for c in event["pool"] if c["candidate_id"] == event["selected_id"])
        require(chosen["agents"] == event["action"]["agents"], "selected candidate changed")
        after = q.apply_state_delta(state, event["delta"])
        count = after["num_of_colliding_pairs"]
        attempted_delta = check_acceptance(event, before, count)
        m = event["metrics"]
        nodes = after["low_level"]["generated"] - initial_nodes
        require(nodes >= used and after["feasible"] == (count == 0), "invalid work/feasibility")
        complete = m["acceptance_evaluated"]
        worse = complete and attempted_delta > 0
        stats = dict(steps=1, generated=nodes-used, complete=int(complete),
            noops=int(m["pp_failure_reason"] == "not_run"),
            worse=int(worse), worse_accepted=int(worse and m["replan_success"]),
            worse_rejected=int(worse and not m["replan_success"]),
            rollback=int(m["pp_rolled_back"]), strict_decrease=int(count < before),
            terminal=int(count == 0))
        for key in ("all", tag, "low_conflict" if before <= 10 else "higher_conflict"):
            bins.setdefault(key, Counter()).update(stats)
        size = str(len(chosen["agents"]))
        sizes.setdefault(size, Counter()).update(stats)
        steps.append(dict(decision=d, nodes_before=used, nodes_after=nodes,
            before=before, after=count, temperature=event["temperature"], actual_size=int(size),
            complete=complete, attempted_global_conflicts=before+attempted_delta if complete else None,
            worse=worse, worse_accepted=worse and m["replan_success"], rollback=m["pp_rolled_back"]))
        state = after
    require(len(steps) == row["decisions"] and state["low_level"]["generated"]-initial_nodes == row["generated"], "terminal work mismatch")
    require(q.state_fingerprint(state) == row["final_fingerprint"] == q.state_fingerprint(io.read_json(folder / "final.json")), "terminal identity")
    require(state["num_of_colliding_pairs"] == row["final_conflicts"] and state["feasible"] == row["success"], "terminal outcome")
    require(bins.get("all", {}).get("noops", 0) == job["expected_noops"], "source no-op mismatch")
    require(row["stop"] == ("feasible" if row["success"] else "node_budget") and
        (row["success"] or row["generated"] >= BUDGET), "incomplete terminal")
    return dict({k:job[k] for k in ("job_id", "arm", "split", "pair_id", "replica", "map_id", "task_id",
        "task_variant", "solver_seed", "initial_fingerprint", "rng_stream_id")},
        status="ok", success=row["success"], final_conflicts=row["final_conflicts"],
        generated=row["generated"], decisions=len(steps), milestones=milestones(steps, row["success"], row["generated"]),
        bins={k:dict(v) for k,v in bins.items()}, sizes={k:dict(v) for k,v in sizes.items()},
        last_temperature=steps[-1]["temperature"] if steps else None)


def summarize(rows):
    result = dict(episodes=len(rows), successes=sum(r["success"] for r in rows),
        maps=len({r["map_id"] for r in rows}), decisions=describe([r["decisions"] for r in rows]),
        generated=describe([r["generated"] for r in rows]), milestones={}, bins={}, sizes={})
    for pct in FRACTIONS:
        all_marks = [r["milestones"][str(pct)] for r in rows]
        crossed = [m for m in all_marks if m["status"] != "already_feasible"]
        result["milestones"][str(pct)] = dict(statuses=dict(Counter(m["status"] for m in all_marks)),
            **{field:describe([m[field] for m in crossed]) for field in ("temperature", "decision", "after", "overshoot")})
    for group in ("bins", "sizes"):
        totals = defaultdict(Counter)
        for r in rows:
            for key, counts in r[group].items():
                totals[key].update(counts)
        result[group] = {k:dict(v) for k,v in sorted(totals.items())}
    return result


def aggregate(rows):
    pairs = paired(rows)
    contrasts = {}
    for pct in FRACTIONS:
        both = [(a["milestones"][str(pct)], b["milestones"][str(pct)]) for a,b in pairs
                if a["milestones"][str(pct)]["status"] != "already_feasible" and
                   b["milestones"][str(pct)]["status"] != "already_feasible"]
        contrasts[str(pct)] = dict(both_crossed=len(both), pairs=len(pairs),
            actor1_minus_actor0_temperature=describe([b["temperature"]-a["temperature"] for a,b in both]),
            actor1_minus_actor0_decisions=describe([b["decision"]-a["decision"] for a,b in both]))
    return dict(paired_exposure=contrasts,
        by_arm={arm:summarize([r for r in rows if r["arm"]==arm]) for arm in ARMS},
        by_map={m:{a:summarize([r for r in rows if r["map_id"]==m and r["arm"]==a]) for a in ARMS}
                for m in sorted({r["map_id"] for r in rows})},
        by_task={t:{a:summarize([r for r in rows if r["task_variant"]==t and r["arm"]==a]) for a in ARMS}
                for t in sorted({r["task_variant"] for r in rows})},
        failure_descriptive={a:summarize([r for r in rows if r["arm"]==a and not r["success"]]) for a in ARMS})


def analyze(resume=False, workers=20):
    require(1 <= workers <= 20, "workers outside read-only budget")
    plan = verify()
    with q._CollectionRunLock(OUT, plan["binding"], "read-only-work-clock"):
        pending = []
        for job in plan["jobs"]:
            path = OUT / "episodes" / (job["job_id"] + ".json")
            if path.exists():
                old = io.check_seal(io.read_json(path))
                require(resume and old["binding"] == plan["binding"] and old["status"] == "ok", "prior audit requires inspection")
            else:
                pending.append(job)
        atomic(OUT / "run_status.json", dict(status="running", pending=len(pending), workers=workers))
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(episode, j): j["job_id"] for j in pending}
                for n, future in enumerate(as_completed(futures), 1):
                    row = dict(future.result(), binding=plan["binding"])
                    io.once(OUT / "episodes" / (row["job_id"] + ".json"), io.sealed(row))
                    if n % 12 == 0 or n == len(pending):
                        print(json.dumps(dict(completed=n, scheduled=len(pending), no_solver=True)), flush=True)
            verify()
            rows = [io.check_seal(io.read_json(OUT / "episodes" / (j["job_id"] + ".json"))) for j in plan["jobs"]]
            require(len(rows) == 192 and all(r["status"] == "ok" for r in rows), "missing results")
            report = dict(schema="lns2.sa_work_clock_report.v1", binding=plan["binding"], complete=True,
                decision="work_exposure_descriptive_no_causal_gain", no_solver=True, no_training=True,
                no_ttf=True, no_promotion=True, summary=aggregate(rows), episodes=rows)
            io.once(OUT / "report.json", io.sealed(report))
            io.once(OUT / "audit.complete.json", io.sealed(dict(binding=plan["binding"],
                report_sha256=io.sha256_file(OUT / "report.json"), episodes=len(rows))))
            atomic(OUT / "run_status.json", dict(status="complete", episodes=len(rows), no_solver=True))
            return dict(complete=True, episodes=len(rows), decision=report["decision"])
        except BaseException as error:
            atomic(OUT / "run_status.json", dict(status="failed_or_interrupted", error=repr(error)))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "analyze", "verify"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    value = prepare() if args.phase == "prepare" else analyze(args.resume, args.workers) if args.phase == "analyze" else dict(verified=True, binding=verify()["binding"])
    print(json.dumps(value), flush=True)
