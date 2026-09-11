"""Verify an unmodified, pinned upstream CLI; not a performance experiment."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import read_json, sha256_file, write_json
from experiments.repair_collection import _CollectionRunLock, _run_jobs

CONFIG = ROOT / "configs/cplns_sequential_reference_v1.json"


def check_archive(config):
    archive = ROOT / config["output"] / "upstream-e3fbcc82.zip"
    if sha256_file(archive) != config["archive_sha256"]:
        raise ValueError("archive hash mismatch")
    source = ROOT / config["source"]
    expected = {}
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or path.parts[0] != "cplns-" + config["commit"]:
                raise ValueError("unsafe archive member")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("archive symlink")
            if info.is_dir():
                continue
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in expected:
                raise ValueError("duplicate archive member")
            expected[relative] = hashlib.sha256(z.read(info)).hexdigest()
    actual = {}
    for p in source.rglob("*"):
        if p.is_symlink():
            raise ValueError("source symlink")
        if p.is_file():
            actual[p.relative_to(source).as_posix()] = sha256_file(p)
    if expected != actual:
        raise ValueError("upstream source modified or incomplete")
    return actual


def fixture_content(name):
    if name == "open":
        width, height, starts, goals = 8, 8, list(range(8)), list(range(56, 64))
    elif name == "order_reversal_corridor":
        width, height, starts, goals = 10, 1, list(range(8)), list(reversed(range(8)))
    else:
        raise ValueError("unknown fixture")
    grid = f"type octile\nheight {height}\nwidth {width}\nmap\n" + ("." * width + "\n") * height
    scen = "version 1\n" + "".join(
        f"0\t{name}.map\t{width}\t{height}\t{s % width}\t{s // width}\t{g % width}\t{g // width}\t"
        f"{abs(s % width - g % width) + abs(s // width - g // width)}\n"
        for s, g in zip(starts, goals))
    return grid, scen


def command(config, name, profile, seed):
    if name not in config["fixtures"] or profile not in config["profiles"] or seed not in config["seeds"]:
        raise ValueError("unregistered smoke job")
    fixture = Path(config["output"]) / "fixtures" / name
    options = {**config["common_options"], **config["profiles"][profile],
               "map": str(fixture.with_suffix(".map")), "agents": str(fixture.with_suffix(".scen")),
               "agentNum": "8", "seed": str(seed), "cutoffTime": str(config["solver_seconds"])}
    if options["numSolver"] != "1" or options["astar_wh"] != "1" or options["maxIterations"] != "0":
        raise ValueError("not an unweighted single-solver feasibility reference")
    return [config["binary"], *[v for k, value in options.items() for v in ("--" + k, value)]]


def parse_output(text, returncode, timed_out=False):
    if timed_out:
        return {"status": "external_timeout", "valid": False}
    if returncode != 0:
        return {"status": "worker_error", "valid": False, "returncode": returncode}
    final = re.findall(r"^final colliding (\d+) costs (\d+) costs_wc (\d+) restart (\d+) .*? iter (\d+) (\d+) .*? t1 (\d+) t2 (\d+) .*? total ([\d.]+)$", text, re.MULTILINE)
    if len(final) != 1:
        return {"status": "missing_or_duplicate_final", "valid": False}
    c, cost, cost_wc, restarts, init_iter, cost_iter, t1, t2, total = final[0]
    has_validation = "validateSolution LNS:491" in text
    if int(c) == 0 and not has_validation:
        return {"status": "missing_upstream_path_validation", "valid": False}
    return {"status": "reported_feasible" if int(c) == 0 else "reported_incomplete", "valid": True,
            "conflicts": int(c), "soc": int(cost), "soc_wc": int(cost_wc),
            "upstream_restart_counter": int(restarts), "init_iterations": int(init_iter),
            "cost_iterations": int(cost_iter), "coarse_t1_seconds": int(t1), "coarse_t2_seconds": int(t2),
            "diagnostic_total_seconds": float(total), "upstream_validation_invoked": has_validation,
            "restart_log_events": len(re.findall(r"g \d+ pe \d+ Restart \d+ Iteration", text)),
            "annealed_accept_log_events": len(re.findall(r"^df [\d.]+ T ", text, re.MULTILINE)),
            "independent_path_validation": False, "no_ttf_or_promotion": True}


def prepare():
    config = read_json(CONFIG)
    out = ROOT / config["output"]
    if (out / "registry.json").exists():
        raise ValueError("registry exists; use verify/smoke, never overwrite")
    source_hashes = check_archive(config)
    inputs = {CONFIG.relative_to(ROOT).as_posix(): sha256_file(CONFIG),
              Path(__file__).relative_to(ROOT).as_posix(): sha256_file(Path(__file__)),
              config["binary"]: sha256_file(ROOT / config["binary"])}
    for name in config["fixtures"]:
        for suffix, content in zip((".map", ".scen"), fixture_content(name)):
            p = out / "fixtures" / (name + suffix)
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                raise ValueError("unexpected existing fixture")
            p.write_text(content, encoding="ascii", newline="\n")
            inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    help_run = subprocess.run([config["binary"], "--help"], cwd=ROOT, capture_output=True, text=True, timeout=20)
    if help_run.returncode != 1 or "Allowed options" not in help_run.stdout:
        raise ValueError("upstream help contract changed")
    (out / "help.log").write_text(help_run.stdout + help_run.stderr, encoding="utf-8")
    registry = {"schema": "lns2.cplns_reference_registry.v1", "source_hashes": source_hashes,
                "inputs": inputs, "commit": config["commit"], "help_exit": help_run.returncode,
                "build_cache_sha256": sha256_file(out / "build/CMakeCache.txt"),
                "unmodified_upstream": True, "no_ttf_or_promotion": True}
    write_json(out / "registry.json", registry)
    return {"source_files": len(source_hashes), "jobs": len(schedule(config)), "binary_sha256": inputs[config["binary"]]}


def verify():
    config = read_json(CONFIG)
    registry = read_json(ROOT / config["output"] / "registry.json")
    for p, digest in registry["inputs"].items():
        if sha256_file(ROOT / p) != digest:
            raise ValueError("registered input changed: " + p)
    if check_archive(config) != registry["source_hashes"]:
        raise ValueError("source registry mismatch")
    return config


def schedule(config):
    return [{"job_id": f"{name}-{profile}-{seed}", "fixture": name, "profile": profile,
             "seed": seed, "argv": command(config, name, profile, seed),
             "directory": config["output"], "external_seconds": config["external_seconds"]}
            for name in config["fixtures"] for profile in config["profiles"] for seed in config["seeds"]]


def worker(job):
    log = ROOT / job["directory"] / "smoke" / (job["job_id"] + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with log.open("wb") as stream:
        proc = subprocess.Popen(job["argv"], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
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
    result = parse_output(log.read_text(encoding="utf-8", errors="replace"), proc.returncode, timed_out)
    result.update(job_id=job["job_id"], fixture=job["fixture"], profile=job["profile"], seed=job["seed"],
                  argv=job["argv"], log_sha256=sha256_file(log),
                  process_returncode=proc.returncode, process_timed_out=timed_out)
    return result


def validate_result(row, job):
    log = ROOT / job["directory"] / "smoke" / (job["job_id"] + ".log")
    if any(row.get(k) != job[k] for k in ("job_id", "fixture", "profile", "seed", "argv")) or row.get("log_sha256") != sha256_file(log):
        raise ValueError("smoke identity/log mismatch")
    parsed = parse_output(log.read_text(encoding="utf-8", errors="replace"),
                          row["process_returncode"], row["process_timed_out"])
    if any(row.get(k) != v for k, v in parsed.items()):
        raise ValueError("smoke summary differs from log")


def run_smoke():
    config = verify()
    out = ROOT / config["output"]
    fingerprint = sha256_file(out / "registry.json")
    with _CollectionRunLock(out, fingerprint, "smoke"):
        pending = []
        for job in schedule(config):
            previous = out / "smoke" / (job["job_id"] + ".json")
            if previous.exists():
                validate_result(read_json(previous), job)
            else:
                pending.append(job)
        def save(row):
            write_json(out / "smoke" / (row["job_id"] + ".json"), row)
            print(row["job_id"], row["status"], flush=True)
        write_json(out / "run_status.json", {"status": "running", "pending": len(pending)})
        _run_jobs(worker, pending, config["workers"], phase="smoke", output_root=out / "smoke",
                  run_fingerprint=fingerprint, timeout_seconds=config["external_seconds"] + 10,
                  on_result=save, failure_result=lambda j, s, e: {"job_id": j["job_id"], "status": s, "valid": False, "error": e},
                  stop_on_failure=True)
    result = report()
    write_json(out / "run_status.json", {"status": "complete", "passed": result["passed"]})
    return result


def report():
    config = verify()
    out = ROOT / config["output"]
    rows = []
    for job in schedule(config):
        row = read_json(out / "smoke" / (job["job_id"] + ".json"))
        validate_result(row, job)
        rows.append(row)
    passed = all(r["valid"] and ((r["status"] == "reported_feasible") == (r["fixture"] == "open")) for r in rows)
    result = {"schema": "lns2.cplns_reference_smoke.v1", "passed": passed,
              "decision": "cli_smoke_only" if passed else "investigate_upstream_or_adapter",
              "jobs": len(rows), "rows": rows, "registry_sha256": sha256_file(out / "registry.json"),
              "no_ttf_or_promotion": True, "independent_path_validation": False}
    write_json(out / "report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "verify", "smoke", "report"))
    args = parser.parse_args()
    value = prepare() if args.phase == "prepare" else ({"verified": True} if args.phase == "verify" and verify()
            else run_smoke() if args.phase == "smoke" else report())
    print(json.dumps(value, indent=2))
