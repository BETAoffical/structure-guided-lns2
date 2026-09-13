"""Isolated PBS admission checks, not a controller or TTF benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import sha256_file
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint, semantic_fingerprint

SOURCE = ROOT / "build/stride-seed25-same-set-pp-gcbs-screen-v1/screen_plan.json"
NATIVE = "build/linux/sa-wall-clock-v1/lns2_env.cpython-310-x86_64-linux-gnu.so"
NATIVE_SHA = "5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d"
OUT = ROOT / "build/pbs-repair-admission-v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare():
    from experiments.stride_successor_repairability import _single_manifest, _source_replay
    from experiments.trace_replay import target_state_from_trace
    require(not (OUT / "plan.json").exists(), "plan already exists")
    require(sha256_file(ROOT / NATIVE) == NATIVE_SHA, "frozen native changed")
    inputs = {SOURCE.relative_to(ROOT).as_posix(): sha256_file(SOURCE)}
    cases = []
    for entry in read(SOURCE)["states"]:
        folder = Path(entry["source_collection"])
        for name, key in (("run_config.json", "source_run_config_sha256"),
                          ("realized_dynamic_manifest.jsonl", "source_manifest_sha256")):
            require(sha256_file(folder / name) == entry[key], "historical source changed")
            inputs[(folder / name).relative_to(ROOT).as_posix()] = entry[key]
        manifest = _single_manifest(folder, entry)
        state, trace = target_state_from_trace(folder, manifest, decision_index=entry["decision_index"],
                                               expected_fingerprint=entry["state_fingerprint"])
        require(sha256_file(trace) == entry["source_trace_sha256"], "historical trace changed")
        require(repair_structure_fingerprint(state) == entry["repair_structure_fingerprint"], "source structure changed")
        inputs[trace.relative_to(ROOT).as_posix()] = entry["source_trace_sha256"]
        job = _source_replay(folder / "run_config.json", task_id=entry["task_id"], solver_seed=entry["solver_seed"])
        base = Path(job["dataset_root"]) / job["row"]["split"]
        files = {k: (base / job["row"][k]).relative_to(ROOT).as_posix() for k in ("map_file", "scenario_file")}
        for path in files.values():
            inputs[path] = sha256_file(ROOT / path)
        cases.append(dict(id=entry["map_id"], files=files, source_state=state,
                          expected_structure=entry["repair_structure_fingerprint"],
                          paths=[a["path"] for a in sorted(state["agents"], key=lambda a:a["id"])],
                          agents=entry["selected_action"]["agents"], seed=entry["first_action_pp_seed"]))
    for name in ("open", "corridor"):
        files = {"map_file": f"tests/data/pbs_{name}.map", "scenario_file": f"tests/data/pbs_{name}.scen"}
        for path in files.values():
            inputs[path] = sha256_file(ROOT / path)
        cases.append(dict(id=name, files=files, paths=([[3,4,5],[5,4,3],[0]] if name == "open" else [[0,1,2],[2,1,0]]),
                          agents=[0,1], seed=17))
    inputs["scripts/diagnose_pbs_repair.py"] = sha256_file(Path(__file__))
    plan = dict(schema="lns2.pbs_admission.v1", cases=cases, inputs=inputs, native=NATIVE,
                native_sha256=NATIVE_SHA, workers=20, fuse_seconds=20, solver_seconds=5,
                repeats=2, no_ttf=True, no_sa=True, original_crash_exact_call_unknown=True,
                source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = semantic_fingerprint(plan)
    write(OUT / "plan.json", plan)
    return dict(cases=len(cases), jobs=len(cases)*2*2, max_workers=20, no_ttf=True)


def verify(plan, *, full=False):
    require(plan["binding"] == semantic_fingerprint({k:v for k,v in plan.items() if k != "binding"}), "plan changed")
    require(sha256_file(ROOT / plan["native"]) == plan["native_sha256"], "native changed")
    require(sha256_file(Path(__file__)) == plan["inputs"]["scripts/diagnose_pbs_repair.py"], "implementation changed")
    if full:
        for name, sha in plan["inputs"].items():
            require(sha256_file(ROOT / name) == sha, "input changed: " + name)


def validate_transition(before, after, metrics, agents):
    from scripts.run_feedback_exploration_diagnostics import validate_final
    validate_final(after)
    require(metrics["action_valid"] and metrics["step_applied"], "invalid action")
    require(sorted(metrics["neighborhood"]) == sorted(agents), "explicit set changed")
    old = {a["id"]: a["path"] for a in before["agents"]}
    require(all(a["path"] == old[a["id"]] for a in after["agents"] if a["id"] not in agents), "external paths changed")
    if not metrics["replan_success"]:
        require(all(a["path"] == old[a["id"]] for a in after["agents"]), "failed repair did not roll back")
        require(after["conflict_edges"] == before["conflict_edges"], "rollback conflicts changed")


def worker(case_id, algo, repeat):
    import faulthandler
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    faulthandler.enable()
    plan = read(OUT / "plan.json")
    verify(plan)
    case = next(c for c in plan["cases"] if c["id"] == case_id)
    sys.path.insert(0, str((ROOT / plan["native"]).parent))
    import lns2_env
    require(sha256_file(Path(lns2_env.__file__)) == plan["native_sha256"], "wrong loaded native")
    from experiments.repair_collection import _plain
    print("construct", flush=True)
    env = lns2_env.LNS2RepairEnv(str(ROOT/case["files"]["map_file"]), str(ROOT/case["files"]["scenario_file"]),
        len(case["paths"]), time_limit=plan["solver_seconds"], replan_algorithm=algo, use_sipp=True)
    print("restore", flush=True)
    before = _plain(env.reset_paths(case["paths"], seed=case["seed"]))
    if "expected_structure" in case:
        require(repair_structure_fingerprint(before) == case["expected_structure"], "restored structure mismatch")
    from scripts.run_feedback_exploration_diagnostics import validate_final
    validate_final(before)
    print("step", flush=True)
    result = _plain(env.step(dict(mode="explicit_neighborhood", agents=case["agents"], random_seed=case["seed"])))
    print("validate", flush=True)
    after, metrics = result["observation"], result["metrics"]
    validate_transition(before, after, metrics, case["agents"])
    row = dict(status="ok", binding=plan["binding"], case_id=case_id, algorithm=algo, repeat=repeat,
               before=before["num_of_colliding_pairs"], after=after["num_of_colliding_pairs"],
               before_structure=repair_structure_fingerprint(before), after_structure=repair_structure_fingerprint(after),
               metrics=metrics, final_state=after, feasible=after["feasible"], no_ttf=True)
    write(OUT / "jobs" / f"{case_id}-{algo}-{repeat}.json", row)
    print("completed", flush=True)


def collect():
    plan = read(OUT / "plan.json")
    verify(plan, full=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "run.lock").open("x") as lock:
        lock.write(str(os.getpid()))
    try:
        def run(job):
            case, algo, repeat = job
            stem = f"{case}-{algo}-{repeat}"
            path = OUT / "jobs" / (stem + ".json")
            if path.exists():
                row = read(path)
                require(row["binding"] == plan["binding"], "resume binding mismatch")
                return row
            log = OUT / "logs" / (stem + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as stream:
                try:
                    p = subprocess.run([sys.executable, "-B", str(Path(__file__)), "worker", "--case", case,
                        "--algorithm", algo, "--repeat", str(repeat)], stdout=stream, stderr=subprocess.STDOUT,
                        timeout=plan["fuse_seconds"], env={**os.environ, "OMP_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1"})
                    status = "crash" if p.returncode < 0 else "error"
                    code = p.returncode
                except subprocess.TimeoutExpired:
                    status, code = "timeout_unknown", None
            if code == 0 and path.exists():
                return read(path)
            row = dict(status=status, returncode=code, binding=plan["binding"], case_id=case, algorithm=algo, repeat=repeat)
            write(path, row)
            return row
        jobs = [(c["id"], a, t) for c in plan["cases"] for a in ("PP", "PBS") for t in range(plan["repeats"])]
        # Threads only supervise independent native subprocesses; no native calls in this parent.
        with ThreadPoolExecutor(max_workers=min(plan["workers"], len(jobs))) as pool:
            rows = list(pool.map(run, jobs))
        errors = [r for r in rows if r["status"] != "ok"]
        reproducible = all(len({r.get("after_structure") for r in rows if r["case_id"] == c["id"] and r["algorithm"] == a}) == 1
                           for c in plan["cases"] for a in ("PP", "PBS"))
        report = dict(schema="lns2.pbs_admission_report.v1", binding=plan["binding"], jobs=len(rows),
            rows=[{k:v for k,v in r.items() if k not in ("metrics", "final_state")} for r in rows],
            errors=len(errors), repeated_outcomes_equal=reproducible,
            decision="stop_native_admission_failure" if errors or not reproducible else "bounded_checks_passed_not_runtime_ready",
            no_ttf=True, acceptance_not_matched=True,
            files={p.relative_to(OUT).as_posix():sha256_file(p) for p in sorted((OUT/"jobs").glob("*.json"))})
        verify(plan, full=True)
        write(OUT / "report.json", report)
        return {k:v for k,v in report.items() if k not in ("rows", "files")}
    finally:
        (OUT / "run.lock").unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "collect", "worker"))
    parser.add_argument("--case")
    parser.add_argument("--algorithm", choices=("PP", "PBS"))
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    result = worker(args.case, args.algorithm, args.repeat) if args.phase == "worker" else globals()[args.phase]()
    print(json.dumps(result, sort_keys=True))
