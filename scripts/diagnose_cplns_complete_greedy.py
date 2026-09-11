"""One additional arm: finish PP but reject worsening, using 20 frozen real tasks."""

import argparse
from collections import defaultdict
import csv
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from scripts import diagnose_cplns_real_tasks as real

OUT = ROOT / "build/cplns-complete-greedy-ablation-v1"
PROTOCOL = ROOT / "docs/CPLNS_COMPLETE_GREEDY_PROTOCOL_ZH.md"
ARM = "complete_greedy_no_restart"
BASE = real.BASE
SA = "author_sa_no_restart"
SOURCE_PLAN_SHA = "7f9f035c713345541de8a2c3e7287e5c475b5ac1affc31332a4aa6c086a66844"
SOURCE_REPORT_SHA = "49d0eae69d3c62f76fd806df4733cd997f8a43cabb6096c2c86c6b660736b4c6"
EARLY_STOP = """        if (!tl_use_simulated_annealing)  // || neighbor_size <= 4)
        {
            if (neighbor.colliding_pairs.size() >=
                neighbor.old_colliding_pairs.size())
                break;
        }
"""


def patch_pp(text):
    start = text.index("bool InitLNS::runPP() {")
    end = text.index("bool InitLNS::getInitialSolution() {", start)
    body = text[start:end]
    if body.count(EARLY_STOP) != 1:
        raise ValueError("PP early-stop anchor missing or ambiguous")
    return text[:start] + body.replace(EARLY_STOP, "", 1) + text[end:]


def source():
    plan = real.verify()
    folder = ROOT / plan["config"]["output"]
    if sha256_file(folder / "plan.json") != SOURCE_PLAN_SHA or sha256_file(folder / "report.json") != SOURCE_REPORT_SHA:
        raise ValueError("frozen real-task evidence changed")
    return plan


def jobs(plan):
    result = []
    for original in plan["jobs"]:
        if original["profile"] != BASE:
            continue
        argv = original["argv"][:]
        argv[0] = (OUT / "build/plns").relative_to(ROOT).as_posix()
        result.append({**original, "job_id": original["job_id"].replace(BASE, ARM),
                       "source_job_id": original["job_id"], "profile": ARM,
                       "output": OUT.relative_to(ROOT).as_posix(), "argv": argv})
    if len(result) != 20:
        raise ValueError("expected exactly 20 additional jobs")
    return result


def expected_files(plan):
    files = {}
    old_source = real.reference.OUT / "source"
    for path in old_source.rglob("*"):
        if path.is_file():
            if path.is_symlink():
                raise ValueError("source symlink")
            rel = path.relative_to(old_source)
            files[OUT / "source" / rel] = (patch_pp(path.read_text()).encode("utf-8")
                if rel.as_posix() == "src/InitLNS.cpp" else path.read_bytes())
    for path in (ROOT / plan["config"]["output"] / "tasks").iterdir():
        if path.suffix in (".map", ".scen"):
            files[OUT / "tasks" / path.name] = path.read_bytes()
    return files


def prepare():
    plan = source()
    if OUT.exists():
        raise ValueError("output exists; do not overwrite")
    for path, data in expected_files(plan).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return {"new_jobs": len(jobs(plan)), "reused_jobs": 40, "workers": 20,
            "solver_seconds_per_job": 30, "formal_ttf": False, "changed_logic": "remove_PP_conflict_early_stop_only"}


def bindings(plan):
    files = expected_files(plan)
    actual = {p for d in (OUT / "source", OUT / "tasks") for p in d.rglob("*") if p.is_file()}
    if actual != set(files):
        raise ValueError("unexpected source/task file set")
    for path, data in files.items():
        if path.is_symlink() or path.read_bytes() != data:
            raise ValueError("source/task changed")
    base = ROOT / plan["config"]["output"]
    paths = [*files, OUT / "build/plns", Path(__file__), Path(real.__file__), PROTOCOL,
             base / "plan.json", base / "report.json"]
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def register():
    plan = source()
    if (OUT / "plan.json").exists():
        raise ValueError("already registered")
    registered = {"schema": "lns2.cplns_complete_greedy.v1", "inputs": bindings(plan), "jobs": jobs(plan),
                  "config": {**plan["config"], "output": OUT.relative_to(ROOT).as_posix()},
                  "no_ttf_or_promotion": True, "old_evidence_reused_not_rerun": True}
    write_json(OUT / "plan.json", registered)
    return {"registered": True, "jobs": 20}


def verify():
    old = source()
    plan = read_json(OUT / "plan.json")
    if plan["inputs"] != bindings(old) or plan["jobs"] != jobs(old) or plan["config"] != {
        **old["config"], "output": OUT.relative_to(ROOT).as_posix()
    }:
        raise ValueError("registered inputs or schedule changed")
    return plan, old


def analyze(job):
    row = real.analyze_job(job)
    first_order, final = None, None
    path = ROOT / job["output"] / "jobs" / (job["job_id"] + ".events.jsonl")
    with path.open() as stream:
        for line in stream:
            event = json.loads(line)
            if first_order is None and event["event"] == "pp_order":
                first_order = event["order"]
            if event["event"] == "final":
                final = event
    if final is None or not isinstance(final["decisions_seen"], int) or final["decisions_seen"] < row["observed_steps"]:
        raise ValueError("missing/inconsistent full PP call count")
    row.update(total_pp_calls=final["decisions_seen"], first_pp_order=first_order)
    if job["profile"] == ARM:
        if row["accepted_worse"] or any(b > a for a, b in zip(row["conflicts"], row["conflicts"][1:])):
            raise ValueError("greedy accepted worsening")
        if row["initial_conflicts"] > 0 and row["upstream_restart_counter"] != 1:
            raise ValueError("unexpected restart")
    return row


def compare(rows):
    groups = defaultdict(dict)
    for row in rows:
        key = row["case_id"], row["seed"]
        if row["profile"] in groups[key]:
            raise ValueError("duplicate paired row")
        groups[key][row["profile"]] = row
    paired = []
    for key, group in sorted(groups.items()):
        if set(group) != {BASE, SA, ARM}:
            raise ValueError("incomplete triplet")
        paired.append({"case_id": key[0], "seed": key[1], "map_id": group[BASE]["map_id"],
                       "initial_equal": len({r["initial_digest"] for r in group.values()}) == 1,
                       "first_order_equal": all(r["first_pp_order"] == group[BASE]["first_pp_order"] for r in group.values()),
                       "pp_calls": {arm: r["total_pp_calls"] for arm, r in group.items()},
                       "feasible": {arm: r["final_feasible"] for arm, r in group.items()}})
    profiles = {}
    for arm in (BASE, ARM, SA):
        values = [r for r in rows if r["profile"] == arm]
        profiles[arm] = {"jobs": len(values), "feasible": sum(r["final_feasible"] for r in values),
                         "prefix_feasible": sum(r["feasible_within_observed_steps"] for r in values),
                         "mean_pp_calls": statistics.mean(r["total_pp_calls"] for r in values),
                         "max_pp_calls": max(r["total_pp_calls"] for r in values),
                         "accepted_worse_in_prefix": sum(r["accepted_worse"] for r in values)}
    comparisons = {}
    for challenger, baseline in ((ARM, BASE), (SA, ARM)):
        valid = [p for p in paired if p["feasible"][challenger] and p["feasible"][baseline]]
        comparisons[challenger + "_vs_" + baseline] = {
            "common_success": len(valid),
            "fewer_pp_calls": sum(p["pp_calls"][challenger] < p["pp_calls"][baseline] for p in valid),
            "more_pp_calls": sum(p["pp_calls"][challenger] > p["pp_calls"][baseline] for p in valid),
            "equal_pp_calls": sum(p["pp_calls"][challenger] == p["pp_calls"][baseline] for p in valid)}
    integrity = all(p["initial_equal"] and p["first_order_equal"] for p in paired)
    return {"profiles": profiles, "paired": paired, "comparisons": comparisons,
            "decision": "bounded_decomposition_complete_no_promotion" if integrity else "pairing_failure_stop"}


def report(plan, old):
    source_rows = {r["job_id"]: r for r in read_json(ROOT / old["config"]["output"] / "report.json")["rows"]}
    old_jobs = [j for j in old["jobs"] if j["profile"] in (BASE, SA)]
    for job in old_jobs:
        path = ROOT / job["output"] / "jobs" / (job["job_id"] + ".json")
        if sha256_file(path) != source_rows[job["job_id"]]["job_record_sha256"]:
            raise ValueError("old job record changed")
    rows = _run_jobs(analyze, [*plan["jobs"], *old_jobs], 20, phase="complete-greedy-analysis", timeout_seconds=180)
    errors = [r for r in rows if r.get("status") != "ok"]
    if errors:
        write_json(OUT / "analysis_errors.json", errors)
        raise ValueError("integrity analysis failed")
    rows.sort(key=lambda r: r["job_id"])
    if len(rows) != 60:
        raise ValueError("expected 20 paired triplets")
    result = {"schema": plan["schema"], **compare(rows), "rows": rows, "no_ttf_or_promotion": True,
              "plan_sha256": sha256_file(OUT / "plan.json"), "source_report_sha256": SOURCE_REPORT_SHA}
    write_json(OUT / "report.json", result)
    table = io.StringIO(newline="")
    writer = csv.writer(table)
    writer.writerow(["case_id", "seed", "map_id", BASE, ARM, SA, "all_feasible", "initial_equal", "first_order_equal"])
    for p in result["paired"]:
        writer.writerow([p["case_id"], p["seed"], p["map_id"], *[p["pp_calls"][a] for a in (BASE, ARM, SA)],
                         all(p["feasible"].values()), p["initial_equal"], p["first_order_equal"]])
    (OUT / "paired_pp_calls.csv").write_text(table.getvalue(), encoding="utf-8")
    return {k: result[k] for k in ("profiles", "comparisons", "decision")}


def run():
    plan, old = verify()
    with _CollectionRunLock(OUT, sha256_file(OUT / "plan.json"), "cplns-complete-greedy"):
        write_json(OUT / "run_status.json", {"status": "running"})
        try:
            real.collect(plan, plan["jobs"])
            result = report(plan, old)
            write_json(OUT / "run_status.json", {"status": "complete", "decision": result["decision"]})
            return result
        except BaseException as exc:
            write_json(OUT / "run_status.json", {"status": "interrupted_or_failed", "error": str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "register", "verify", "run"))
    phase = parser.parse_args().phase
    print(json.dumps({"prepare": prepare, "register": register, "verify": lambda: {"verified": bool(verify())}, "run": run}[phase](), indent=2))
