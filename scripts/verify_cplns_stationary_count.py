"""Separate corrected author reference: stationary-path counting only, no performance claims."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _run_jobs
from scripts import verify_cplns_noalloc as noalloc
from scripts import verify_cplns_observation as bridge

OUT = ROOT / "build/cplns-stationary-count-native-v1"
PREREQUISITES = {
    "build/cplns-noalloc-audit-v1/report.json": "3e65ab812c0cc188bd3df64a497ba5a5f433b0167ac80a40f429d1fc6a4dd2eb",
    "build/cplns-count-function-audit-v1/report.json": "14db21d044b9af36ca6c4c12af7005959a7683afd3407b70ec93b3c98869910b",
}


def corrected_source(text):
    guard = "if (path.size() < 2) return succ;"
    if text.count(guard) != 1:
        raise ValueError("stationary guard missing or ambiguous")
    return text.replace(guard, "if (path.empty()) return succ;", 1)


def context():
    config, ref = noalloc.verify()
    for path, digest in PREREQUISITES.items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError("prerequisite report changed")
    return config, ref


def files(config, ref):
    expected = noalloc.expected_files({**config, "output": OUT.relative_to(ROOT).as_posix()}, ref)
    p = OUT / "source/src/InitLNS.cpp"
    expected[p] = corrected_source(expected[p].decode("utf-8")).encode("utf-8")
    return expected


def jobs(config, ref):
    result = []
    for source in noalloc.schedule(config, ref):
        if source["lane"] != "observer_on":
            continue
        job = dict(source)
        argv = source["argv"][:]
        argv[0] = (OUT / "build/plns").relative_to(ROOT).as_posix()
        # Reuse the exact registered fixture paths/CLI from the noalloc control.
        job.update(output=OUT.relative_to(ROOT).as_posix(), argv=argv, source_job_id=source["job_id"])
        result.append(job)
    return result


def prepare():
    config, ref = context()
    if OUT.exists():
        raise ValueError("corrected reference directory already exists")
    for p, data in files(config, ref).items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return {"jobs": len(jobs(config, ref)), "only_change": "stationary-path counter guard", "official_or_dual16_modified": False}


def bindings(config, ref):
    paths = [Path(__file__), Path(noalloc.__file__), OUT / "build/plns"]
    paths += [ROOT / p for p in PREREQUISITES]
    paths += [p for directory in (OUT / "source", OUT / "fixtures") for p in directory.rglob("*") if p.is_file()]
    for p, expected in files(config, ref).items():
        if p.is_symlink() or p.read_bytes() != expected:
            raise ValueError("unexpected corrected source/fixture change")
    source_files = {p for p in (OUT / "source").rglob("*") if p.is_file()}
    if source_files != {p for p in files(config, ref) if OUT / "source" in p.parents}:
        raise ValueError("unexpected corrected source file set")
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths}


def register():
    config, ref = context()
    if (OUT / "registry.json").exists():
        raise ValueError("corrected registry already exists")
    write_json(OUT / "registry.json", {"schema": "lns2.cplns_stationary_corrected.v1",
               "inputs": bindings(config, ref), "jobs": jobs(config, ref), "workers": 20,
               "no_ttf_or_promotion": True, "policy": "author_stationary_count_corrected_not_official"})
    return {"registered": True, "jobs": len(jobs(config, ref))}


def verify():
    config, ref = context()
    registered = read_json(OUT / "registry.json")
    if registered["inputs"] != bindings(config, ref) or registered["jobs"] != jobs(config, ref):
        raise ValueError("corrected registry mismatch")
    return config, ref


def collect():
    config, ref = verify()
    fingerprint = sha256_file(OUT / "registry.json")
    with _CollectionRunLock(OUT, fingerprint, "stationary-counter"):
        pending = []
        for job in jobs(config, ref):
            p = OUT / "jobs" / (job["job_id"] + ".json")
            if p.exists():
                row = read_json(p)
                bridge.verify_job(row, job)
                if row["status"] != "complete":
                    raise ValueError("failed job cannot be automatically retried")
            else:
                pending.append(job)
        def save(row):
            write_json(OUT / "jobs" / (row["job_id"] + ".json"), row)
            print(row["job_id"], row["status"], flush=True)
        write_json(OUT / "run_status.json", {"status": "running", "pending": len(pending)})
        try:
            _run_jobs(bridge.run_job, pending, 20, phase="stationary-counter", output_root=OUT / "jobs",
                      run_fingerprint=fingerprint, timeout_seconds=35, on_result=save,
                      failure_result=lambda j, s, e: {"job_id": j["job_id"], "status": s, "error": e}, stop_on_failure=True)
            result = report()
            write_json(OUT / "run_status.json", {"status": "complete", "decision": result["decision"]})
            return result
        except BaseException as exc:
            write_json(OUT / "run_status.json", {"status": "interrupted_or_failed", "error": str(exc)})
            raise


def report():
    config, ref = verify()
    base = ROOT / config["output"]
    rows, hashes = [], {}
    for job in jobs(config, ref):
        p = OUT / "jobs" / (job["job_id"] + ".json")
        row = read_json(p)
        bridge.verify_job(row, job)
        hashes[p.relative_to(ROOT).as_posix()] = sha256_file(p)
        text = p.with_suffix(".log").read_text(encoding="utf-8", errors="replace")
        parsed = bridge.parse_output(text, row["returncode"], row["timed_out"])
        events = [json.loads(line) for line in p.with_suffix(".events.jsonl").read_text().splitlines()]
        grid, scen, count = bridge.fixture(job["fixture"], config["fixture_seed"])
        item = {"job_id": job["job_id"], "process_valid": parsed["valid"]}
        try:
            metrics = bridge.validate_events(events, grid, scen)
            if metrics["final_conflicts"] != parsed.get("conflicts"):
                raise ValueError("corrected final path/log mismatch")
            item.update(paths_valid=True, metrics=metrics)
        except ValueError as exc:
            item.update(paths_valid=False, error=str(exc))
        original_log = base / "jobs" / (job["source_job_id"] + ".log")
        original_text = original_log.read_text(encoding="utf-8", errors="replace")
        original_record = original_log.with_suffix(".json")
        if sha256_file(original_record) != read_json(base / "report.json")["job_record_hashes"][original_record.relative_to(ROOT).as_posix()]:
            raise ValueError("original job record changed")
        if sha256_file(original_log) != read_json(original_record)["log_sha256"]:
            raise ValueError("original comparison log changed")
        comparison = noalloc.compare_logs(original_text, text, count, config["prefix_records"])
        if job["fixture"] == "open":
            comparison["initial_paths_equal"] = True  # No InitLNS path block in either PP-direct job.
            comparison["prefix_equal"] = bridge.scientific_log(original_text) == bridge.scientific_log(text)
        item["neutral_control"] = job["fixture"] != "two_door"
        item["comparison"] = comparison
        item["passed"] = item["process_valid"] and item["paths_valid"] and comparison["initial_paths_equal"] and (
            not item["neutral_control"] or comparison["prefix_equal"])
        rows.append(item)
    passed = all(row["passed"] for row in rows)
    result = {"schema": "lns2.cplns_stationary_corrected_report.v1", "rows": rows, "job_record_hashes": hashes,
              "registry_sha256": sha256_file(OUT / "registry.json"), "no_ttf_or_promotion": True,
              "official_or_dual16_modified": False,
              "decision": "bounded_native_counter_pass" if passed else "investigate_counter_before_real_cases"}
    write_json(OUT / "report.json", result)
    return {"jobs": len(rows), "passed": sum(row["passed"] for row in rows),
            "path_failures": sum(not row["paths_valid"] for row in rows), "decision": result["decision"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "register", "collect", "report"))
    args = parser.parse_args()
    print(json.dumps({"prepare": prepare, "register": register, "collect": collect, "report": report}[args.phase](), indent=2))
