"""Versioned, bounded hard-pair compatibility diagnosis of retained SA tails."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import diagnose_pbs_repair as io
from scripts import run_pbs_repair_mechanism as prior
from experiments.pair_compatibility import diagnose
from experiments.diagnostic_integrity import read_bound_plan

OUT = ROOT / "build/sa-pair-compatibility-v1"
SOURCE_SHA = "fa69839b63a9af7bec13672bc978d4a01a5deefbbbf021794d255e107bea5c74"


def verify(evidence_only=False):
    return read_bound_plan(ROOT,OUT,evidence_only)


def prepare():
    io.require(not (OUT / "plan.json").exists(), "existing plan must not be overwritten")
    io.require(io.sha256_file(prior.OUT / "summary.json") == SOURCE_SHA, "PBS source changed")
    prior.stage_report("tail")
    reg = io.read(prior.OUT / "tail/registration.json")
    source_cases = prior.tail_cases()
    cases, inputs, jobs = [], {}, []
    names = ["scripts/diagnose_sa_pair_compatibility.py", "experiments/pair_compatibility.py",
             "experiments/diagnostic_integrity.py",
             "experiments/local_path_search.py", "experiments/state_analysis.py",
             "tests/evaluation/test_sa_pair_compatibility.py", "docs/SA_PAIR_COMPATIBILITY_PROTOCOL_ZH.md",
             "build/pbs-repair-mechanism-v1/summary.json", "build/pbs-repair-mechanism-v1/tail/registration.json",
             "build/pbs-repair-mechanism-v1/tail/report.json", io.NATIVE]
    for old_case in reg["cases"]:
        # PP may accept equal-conflict paths; the failed PBS branch is fully rolled back.
        job_file = prior.OUT / "tail/jobs" / (old_case["id"] + "-PBS-0.json")
        row = io.read(job_file)
        state = row["final_state"]
        reconstructed = next(c for c in source_cases if c["id"] == old_case["id"])
        io.require(state["conflict_edges"] and row["summary"]["before_structure"] == row["summary"]["after_structure"], "source is not the restored root")
        io.require(io.repair_structure_fingerprint(state) == reconstructed["expected_structure"], "source replay mismatch")
        cases.append(dict(id=old_case["id"], state=state, selected=old_case["agents"], files=old_case["files"]))
        names.append(job_file.relative_to(ROOT).as_posix())
        names.extend(old_case["files"].values())
        for pair in state["conflict_edges"]:
            for relaxed in (False, True):
                jobs.append(dict(id=f"{old_case['id']}-{pair[0]}-{pair[1]}-{'relaxed' if relaxed else 'fixed'}",
                                 case_id=old_case["id"], pair=pair, relax_selected=relaxed))
    for name in sorted(set(names)):
        inputs[name] = io.sha256_file(ROOT / name)
    plan = dict(schema="lns2.sa_pair_compatibility.v1", cases=cases, jobs=jobs, inputs=inputs,
                seconds=45., max_expanded=250000, process_fuse=90, workers=20, no_ttf=True,
                source_commit=subprocess.check_output(["git","rev-parse","HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = io.semantic_fingerprint(plan)
    io.write(OUT / "plan.json", plan)
    print(dict(jobs=len(jobs), states=len(cases), workers=20, seconds_per_job=45, no_ttf=True))


def jobs_for(plan, backend):
    if backend == "bfs":
        return plan["jobs"]
    base = read_stage(plan, "bfs")
    return [j for j in plan["jobs"] if base[j["id"]]["result"]["status"] == "unknown"]


def read_stage(plan, backend):
    manifest = io.read(OUT / backend / "manifest.json")
    io.require(manifest["binding"] == plan["binding"], "manifest identity mismatch")
    expected = jobs_for(plan, backend) if backend != "bfs" else plan["jobs"]
    io.require(set(manifest["files"]) == {j["id"] for j in expected}, "incomplete manifest")
    rows = {}
    for job in expected:
        path = OUT / backend / "jobs" / (job["id"] + ".json")
        io.require(io.sha256_file(path) == manifest["files"][job["id"]], "result changed")
        row = io.read(path)
        io.require(row["binding"] == plan["binding"] and row["job"] == job and row["backend"] == backend, "job identity mismatch")
        io.require(row["status"] == "ok", "unexplained worker error")
        rows[job["id"]] = row
    return rows


def worker(backend, job_id):
    plan = verify()
    job = next(j for j in plan["jobs"] if j["id"] == job_id)
    case = next(c for c in plan["cases"] if c["id"] == job["case_id"])
    result = diagnose(case["state"], case["selected"], job["pair"], job["relax_selected"],
                      backend, plan["seconds"], plan["max_expanded"])
    io.write(OUT / backend / "jobs" / (job_id + ".json"),
             dict(status="ok", binding=plan["binding"], job=job, backend=backend, result=result))


def collect(backend, resume):
    folder = OUT / backend
    if (folder/"manifest.json").exists():
        io.require(resume,"completed stage exists; use explicit resume")
        plan=verify(evidence_only=True)
        rows=read_stage(plan,backend)
        print(dict(stage=backend,verified_existing=len(rows),rerun_jobs=0))
        return
    plan = verify()
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / "RUNNING.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        jobs = jobs_for(plan, backend)
        def execute(job):
            path = folder / "jobs" / (job["id"] + ".json")
            if path.exists():
                io.require(resume, "result exists; use explicit resume")
                receipt=folder/"receipts"/(job["id"]+".json")
                io.require(receipt.exists(),"unsealed result requires manual audit")
                seal=io.read(receipt)
                io.require(seal["binding"]==plan["binding"] and seal["sha256"]==io.sha256_file(path),"resume result changed")
                row = io.read(path)
                io.require(row["binding"] == plan["binding"] and row["job"] == job
                           and row["backend"] == backend and row["status"] == "ok", "invalid resume")
                return
            log = folder / "logs" / (job["id"] + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as stream:
                proc = subprocess.Popen([sys.executable, "-B", str(Path(__file__)), backend, "--worker", job["id"]],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ,"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"})
                try:
                    code = proc.wait(timeout=plan["process_fuse"])
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    code = "external_timeout"
            io.require(code == 0 and path.exists(), f"worker error {job['id']}: {code}")
            io.write(folder/"receipts"/(job["id"]+".json"),dict(binding=plan["binding"],sha256=io.sha256_file(path)))
        with ThreadPoolExecutor(max_workers=min(plan["workers"], max(1,len(jobs)))) as pool:
            list(pool.map(execute, jobs))
        io.write(folder / "manifest.json", dict(binding=plan["binding"], files={j["id"]:
                 io.sha256_file(folder / "jobs" / (j["id"] + ".json")) for j in jobs}))
        rows = read_stage(plan, backend)
        print(dict(stage=backend, jobs=len(rows), results={k:sum(r["result"]["status"]==k for r in rows.values())
                                                        for k in ("feasible","infeasible","unknown")}))
    finally:
        lock.unlink()


def analyze(native=False):
    plan = verify(evidence_only=True)
    bfs, astar = read_stage(plan,"bfs"), read_stage(plan,"astar")
    rows = {**bfs,**astar}
    final = []
    for job in plan["jobs"]:
        row = rows[job["id"]]
        result = row["result"]
        final.append(dict(job=job, backend=row["backend"], status=result["status"],
            reason=result.get("reason"),expanded=result["expanded"],
            proves_selected_insufficient=result["proves_selected_insufficient"],
            full_conflicts=result.get("full_state_conflicts_if_injected"),
            new_conflicts=result.get("new_full_state_conflict_edges")))
    native_rows = []
    if native:
        binary = ROOT / io.NATIVE
        io.require(io.sha256_file(binary)==io.NATIVE_SHA,"frozen native changed")
        sys.path.insert(0,str(binary.parent))
        import lns2_env
        io.require(Path(lns2_env.__file__).resolve()==binary.resolve(),"wrong native loaded")
        for job in plan["jobs"]:
            row = rows[job["id"]]
            if row["result"]["status"] != "feasible":
                continue
            case = next(c for c in plan["cases"] if c["id"]==job["case_id"])
            agents = sorted(case["state"]["agents"],key=lambda a:a["id"])
            io.require([a["id"] for a in agents]==list(range(len(agents))),"native source IDs must be contiguous")
            env = lns2_env.LNS2RepairEnv(str(ROOT/case["files"]["map_file"]),str(ROOT/case["files"]["scenario_file"]),len(agents),time_limit=60)
            before = env.reset_paths([a["path"] for a in agents],seed=17)
            io.require(io.repair_structure_fingerprint(before)==io.repair_structure_fingerprint(case["state"]),"native restoration mismatch")
            replacements = row["result"]["paths"]
            paths = [replacements.get(str(a["id"]),a["path"]) for a in agents]
            after = env.reset_paths(paths,seed=17)
            io.require([a["path"] for a in after["agents"]]==paths,"native altered witness")
            actual = sorted([sorted(e) for e in after["conflict_edges"]])
            io.require(actual==row["result"]["full_state_conflicts_if_injected"],"native conflict mismatch")
            native_rows.append(dict(job_id=job["id"],before=before["num_of_colliding_pairs"],
                after=after["num_of_colliding_pairs"],feasible=after["feasible"],
                status="validated_path_injection_not_native_repair"))
    report=dict(schema="lns2.sa_pair_compatibility_report.v1",binding=plan["binding"],rows=final,
                bfs_jobs=len(bfs),astar_jobs=len(astar),native_witnesses=native_rows,
                inputs={b:io.sha256_file(OUT/b/"manifest.json") for b in ("bfs","astar")},no_ttf=True,
                necessary_obstructions=sum(r["proves_selected_insufficient"] for r in final),
                complete_pair_witnesses=sum(r["status"]=="feasible" for r in final))
    io.write(OUT/("native_report.json" if native else "report.json"),report)
    print({k:v for k,v in report.items() if k not in ("rows","native_witnesses","inputs")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "bfs", "astar", "analyze", "verify-native"))
    parser.add_argument("--worker")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.stage, args.worker)
    elif args.stage == "prepare":
        prepare()
    elif args.stage in ("analyze","verify-native"):
        analyze(native=args.stage=="verify-native")
    else:
        collect(args.stage, args.resume)
