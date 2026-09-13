"""Capture isolated AddressSanitizer PBS crashes without changing frozen runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_pbs_repair as audit

BUILD = ROOT / "build/linux/pbs-admission-asan-v1"
OUT = audit.OUT / "asan"


def worker(case_id):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    plan = audit.read(audit.OUT / "plan.json")
    audit.verify(plan)
    case = next(c for c in plan["cases"] if c["id"] == case_id)
    sys.path.insert(0, str(BUILD))
    import lns2_env
    audit.require(Path(lns2_env.__file__).resolve().parent == BUILD.resolve(), "wrong diagnostic module")
    env = lns2_env.LNS2RepairEnv(str(ROOT/case["files"]["map_file"]), str(ROOT/case["files"]["scenario_file"]),
                               len(case["paths"]), time_limit=5, replan_algorithm="PBS", use_sipp=True)
    print("restore", flush=True)
    before = env.reset_paths(case["paths"], seed=case["seed"])
    if "expected_structure" in case:
        audit.require(audit.repair_structure_fingerprint(before) == case["expected_structure"], "source mismatch")
    print("step", flush=True)
    result = env.step(dict(mode="explicit_neighborhood", agents=case["agents"], random_seed=case["seed"]))
    audit.validate_transition(before, result["observation"], result["metrics"], case["agents"])
    print("returned valid", flush=True)


def collect():
    audit.require(not (OUT / "report.json").exists(), "diagnostic exists; do not overwrite")
    plan = audit.read(audit.OUT / "plan.json")
    audit.verify(plan, full=True)
    binary, = BUILD.glob("lns2_env*.so")
    runtime = subprocess.check_output(["g++", "-print-file-name=libasan.so"], text=True).strip()
    audit.require(Path(runtime).is_file(), "ASan runtime not available")
    OUT.mkdir(parents=True, exist_ok=True)
    def run(case):
        name = case["id"]
        log = OUT / (name + ".log")
        with log.open("w") as stream:
            try:
                result = subprocess.run([sys.executable, "-B", str(Path(__file__)), "--worker", name],
                    stdout=stream, stderr=subprocess.STDOUT, timeout=60,
                    env={**os.environ, "LD_PRELOAD":runtime, "ASAN_OPTIONS":"detect_leaks=0:disable_coredump=1",
                         "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1"})
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = "timeout_unknown"
        return dict(case_id=name, returncode=code, log=log.relative_to(audit.OUT).as_posix(), sha256=audit.sha256_file(log))
    # Native execution is isolated in children; these threads only supervise.
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(run, plan["cases"]))
    report = dict(schema="lns2.pbs_asan_diagnostic.v1", source_binding=plan["binding"],
                  native_sha256=audit.sha256_file(binary), cmake_cache_sha256=audit.sha256_file(BUILD/"CMakeCache.txt"),
                  script_sha256=audit.sha256_file(Path(__file__)), jobs=rows, no_ttf=True, frozen_native_unchanged=True)
    audit.verify(plan, full=True)
    audit.write(OUT / "report.json", report)
    print(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    args = parser.parse_args()
    worker(args.worker) if args.worker else collect()
