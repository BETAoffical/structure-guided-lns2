"""Bounded source-vs-observer divergence localization; never changes author logic."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from scripts import verify_cplns_observation as bridge

OUT = ROOT / "build/cplns-observer-divergence-v1"


def first_difference(left, right):
    count = min(128, len(left), len(right))
    index = next((i for i in range(count) if left[i] != right[i]), None)
    return {"compared_records": count, "equal_prefix": index is None and count > 0,
            "first_difference": index,
            "left": left[index] if index is not None else None,
            "right": right[index] if index is not None else None}


def jobs(config, ref):
    result = []
    for original in bridge.schedule(config, ref):
        if original["fixture"] not in ("dense_open", "two_door") or original["profile"] != "author_sa_no_restart" or original["seed"] != 0:
            continue
        lanes = ("upstream", "upstream_repeat") if original["lane"] == "upstream" else (original["lane"],)
        for rule in ("Adaptive", "Target", "Collision", "Random"):
            for lane in lanes:
                job = dict(original)
                argv = original["argv"][:]
                argv[argv.index("--initDestoryStrategy") + 1] = rule
                job.update(job_id=f"{original['fixture']}-{rule}-{lane}", lane=lane,
                           output=OUT.relative_to(ROOT).as_posix(), argv=argv, rule=rule)
                result.append(job)
    return result


def binding():
    config, ref = bridge.verify()
    paths = [Path(__file__), ROOT / config["output"] / "registry.json", ROOT / config["output"] / "report.json"]
    return config, ref, {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def prepare():
    config, ref, inputs = binding()
    if OUT.exists():
        raise ValueError("divergence run already exists")
    plan = {"schema": "lns2.cplns_observer_divergence.v1", "inputs": inputs, "jobs": jobs(config, ref),
            "workers": 20, "no_ttf_or_promotion": True,
            "hypothesis": "Target auxiliary search depends on heap pointer order; do not repair or change scientific gates."}
    write_json(OUT / "plan.json", plan)
    return {"jobs": len(plan["jobs"]), "workers": 20, "solver_seconds": 3}


def verify():
    config, ref, inputs = binding()
    plan = read_json(OUT / "plan.json")
    if plan["inputs"] != inputs or plan["jobs"] != jobs(config, ref) or plan["workers"] != 20:
        raise ValueError("divergence run identity changed")
    return plan


def collect():
    plan = verify()
    fingerprint = sha256_file(OUT / "plan.json")
    with _CollectionRunLock(OUT, fingerprint, "localize"):
        pending = []
        for job in plan["jobs"]:
            path = OUT / "jobs" / (job["job_id"] + ".json")
            if path.exists():
                row = read_json(path)
                bridge.verify_job(row, job)
                if row["status"] != "complete":
                    raise ValueError("previous failure; no automatic retry")
            else:
                pending.append(job)
        def save(row):
            write_json(OUT / "jobs" / (row["job_id"] + ".json"), row)
            print(row["job_id"], row["status"], flush=True)
        write_json(OUT / "run_status.json", {"status": "running", "pending": len(pending)})
        try:
            _run_jobs(bridge.run_job, pending, 20, phase="localize", output_root=OUT / "jobs",
                      run_fingerprint=fingerprint, timeout_seconds=35, on_result=save,
                      failure_result=lambda j, s, e: {"job_id": j["job_id"], "status": s, "error": e}, stop_on_failure=True)
            result = report()
            write_json(OUT / "run_status.json", {"status": "complete"})
            return result
        except BaseException as exc:
            write_json(OUT / "run_status.json", {"status": "interrupted_or_failed", "error": str(exc)})
            raise


def report():
    plan = verify()
    groups = {}
    for job in plan["jobs"]:
        row = read_json(OUT / "jobs" / (job["job_id"] + ".json"))
        bridge.verify_job(row, job)
        text = (OUT / "jobs" / (job["job_id"] + ".log")).read_text(encoding="utf-8", errors="replace")
        result = bridge.parse_output(text, row["returncode"], row["timed_out"])
        if not result["valid"]:
            raise ValueError("invalid CLI result")
        if job["lane"] == "observer_on":
            events = [json.loads(line) for line in (OUT / "jobs" / (job["job_id"] + ".events.jsonl")).read_text().splitlines()]
            grid, scen, _ = bridge.fixture(job["fixture"], 917)
            metrics = bridge.validate_events(events, grid, scen)
            if metrics["final_conflicts"] != result["conflicts"]:
                raise ValueError("path/log mismatch")
        groups.setdefault((job["fixture"], job["rule"]), {})[job["lane"]] = bridge.scientific_log(text)
    comparisons = []
    for (fixture, rule), lanes in groups.items():
        for left, right in (("upstream", "upstream_repeat"), ("upstream", "observer_off"),
                            ("upstream", "observer_on"), ("observer_off", "observer_on")):
            comparisons.append({"fixture": fixture, "rule": rule, "left_lane": left, "right_lane": right,
                                **first_difference(lanes[left], lanes[right])})
    result = {"schema": "lns2.cplns_observer_divergence_report.v1", "jobs": len(plan["jobs"]),
              "comparisons": comparisons, "no_ttf_or_promotion": True,
              "decision": "localization_only_original_gate_unchanged", "plan_sha256": sha256_file(OUT / "plan.json")}
    write_json(OUT / "report.json", result)
    return {"jobs": result["jobs"], "decision": result["decision"],
            "by_rule": {rule: {"equal": sum(c["equal_prefix"] for c in comparisons if c["rule"] == rule),
                               "total": sum(c["rule"] == rule for c in comparisons)}
                        for rule in ("Adaptive", "Target", "Collision", "Random")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "report"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "collect": collect, "report": report}[args.phase](), indent=2))
