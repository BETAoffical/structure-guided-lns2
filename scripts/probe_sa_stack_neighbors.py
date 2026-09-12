"""Bounded neighbor enumeration microbenchmark and isolated native replay."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json
from experiments.sa_single_check_runtime import SingleFullCheckPool
from scripts import confirm_sa_single_check_runtime as prior

OUT = ROOT / "build/sa-stack-neighbors-probe-v1"
FROZEN = ROOT / "build/linux/sa-wall-clock-v1"
SO = "lns2_env.cpython-310-x86_64-linux-gnu.so"
pilot = prior.pilot
PATTERN = "for (int next_location : instance.getNeighbors(curr->location)) // move to neighboring locations\n        {"
REPLACEMENT = """const int neighbor_candidates[4] = {curr->location + 1, curr->location - 1,
            curr->location + instance.num_of_cols, curr->location - instance.num_of_cols};
        for (int next_location : neighbor_candidates)
        {
            if (!instance.validMove(curr->location, next_location)) continue;"""


def transformed(text):
    start = text.index("Path SIPP::findPath(const ConstraintTable& constraint_table, double time_limit_seconds)")
    end = text.index("Path SIPP::findOptimalPath", start)
    part = text[start:end]
    if part.count(PATTERN) != 1:
        raise ValueError("SIPPS source pattern changed")
    return text[:start] + part.replace(PATTERN, REPLACEMENT) + text[end:]


def execute(args, cwd=FROZEN):
    result = subprocess.run([str(a) for a in args], cwd=cwd, check=True, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
    return result.stdout


def register():
    if (OUT / "plan.json").exists(): raise ValueError("plan exists")
    r = prior.verify()
    names = ["scripts/probe_sa_stack_neighbors.py", "tests/sa_stack_neighbors_probe.cpp",
             "tests/evaluation/test_sa_stack_neighbors.py", "docs/SA_STACK_NEIGHBORS_PROTOCOL_ZH.md",
             "third_party/mapf_lns2/src/SIPP.cpp", "CMakeLists.txt"]
    build_names = [SO, "libmapf_lns2_core.a", "libstructure_guided_instance_validation.a",
                   "CMakeFiles/mapf_lns2_core.dir/flags.make", "CMakeFiles/lns2_env.dir/link.txt",
                   "CMakeFiles/lns2_env.dir/src/python_bindings.cpp.o", "CMakeFiles/lns2_env.dir/src/online_features.cpp.o"]
    names += [(FROZEN / n).relative_to(ROOT).as_posix() for n in build_names]
    inputs = {n: sha256_file(ROOT / n) for n in names}
    write_json(OUT / "plan.json", dict(schema="lns2.sa_stack_neighbors.v1", inputs=inputs,
        cases=r["cases"], records=r["records"], config=r["config"], workers=8,
        no_ttf=True, default_changed=False, minimum_micro_reduction_percent=10,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()))
    return dict(registered=True, replay_cases=8, maximum_replay_steps=792, no_ttf=True)


def verify():
    r = prior.verify()
    p = read_json(OUT / "plan.json")
    for n, h in p["inputs"].items():
        if sha256_file(ROOT / n) != h: raise ValueError("registered source/build changed: " + n)
    if (p["cases"], p["records"], p["config"]) != (r["cases"], r["records"], r["config"]):
        raise ValueError("cohort changed")
    if (p["workers"], p["minimum_micro_reduction_percent"], p["no_ttf"], p["default_changed"]) != (8, 10, True, False):
        raise ValueError("protocol changed")
    return p


def build():
    p = verify()
    if (OUT / "build.json").exists(): raise ValueError("build exists; no overwrite")
    OUT.mkdir(parents=True, exist_ok=True)
    flags = {}
    for line in (FROZEN / "CMakeFiles/mapf_lns2_core.dir/flags.make").read_text().splitlines():
        if line.startswith("CXX_"):
            name, value = line.split("=", 1); flags[name.strip()] = shlex.split(value)
    compile_flags = flags["CXX_DEFINES"] + flags["CXX_INCLUDES"] + flags["CXX_FLAGS"]
    executable = OUT / "neighbor_probe"
    execute(["/usr/bin/c++", *compile_flags, ROOT / "tests/sa_stack_neighbors_probe.cpp",
             FROZEN / "libmapf_lns2_core.a", "-o", executable])
    observations = []
    maps = set()
    for case in p["cases"]:
        if case["map_id"] in maps: continue
        maps.add(case["map_id"])
        root = ROOT / p["config"]["dataset"]["output"] / case["row"]["split"]
        output = execute([executable, root / case["row"]["map_file"], root / case["row"]["scenario_file"]])
        for line in output.splitlines():
            if not line.startswith("RESULT "): continue
            _, cells, a, b = line.split()
            observations.append(dict(map_id=case["map_id"], cells=int(cells), reference=float(a), stack=float(b)))
    if len(observations) != 16: raise ValueError("microbenchmark coverage missing")
    reduction = 100 * (1 - sum(r["stack"] for r in observations) / sum(r["reference"] for r in observations))
    write_json(OUT / "micro.json", dict(plan_sha256=sha256_file(OUT / "plan.json"), rows=observations,
        reduction_percent=reduction, no_ttf=True, executable_sha256=sha256_file(executable)))
    if reduction < p["minimum_micro_reduction_percent"]:
        return dict(micro_passed=False, reduction_percent=reduction, native_not_built=True)
    # Generate exactly one mechanical source substitution outside the source tree.
    cpp = OUT / "SIPP.cpp"
    cpp.write_text(transformed((ROOT / "third_party/mapf_lns2/src/SIPP.cpp").read_text()), encoding="utf-8")
    obj = OUT / "SIPP.o"
    execute(["/usr/bin/c++", *compile_flags, "-c", cpp, "-o", obj])
    link = shlex.split((FROZEN / "CMakeFiles/lns2_env.dir/link.txt").read_text())
    link[link.index("-o") + 1] = str(OUT / SO)
    link.insert(link.index("libmapf_lns2_core.a"), str(obj))
    execute(link)
    write_json(OUT / "build.json", dict(plan_sha256=sha256_file(OUT / "plan.json"),
        native_path=(OUT / SO).relative_to(ROOT).as_posix(), native_sha256=sha256_file(OUT / SO),
        generated_cpp_sha256=sha256_file(cpp), object_sha256=sha256_file(obj),
        micro_sha256=sha256_file(OUT / "micro.json"), frozen_native_unchanged=True))
    verify()
    return dict(built=True, micro_reduction_percent=reduction, no_ttf=True)


def replay_worker(job):
    config = deepcopy(job["config"])
    config["frozen"].update(native_path=job["build"]["native_path"], native_sha256=job["build"]["native_sha256"])
    record = job["record"]
    if sha256_file(ROOT / record["path"]) != record["sha256"]: raise ValueError("trace changed")
    saved = read_json(ROOT / record["path"])
    env = pilot.make_env(dict(job, config=config, budget=3000.))
    state = pilot._plain(env.reset(seed=job["case"]["solver_seed"]))
    expected = saved["initial_state"]
    if pilot.state_fingerprint(state) != pilot.state_fingerprint(expected): raise ValueError("reset mismatch")
    selector = SingleFullCheckPool(job["case"])
    for d, event in enumerate(saved["events"]):
        if selector.select(env, state, d) != (event["selected_index"], event["pool"]):
            raise ValueError(f"candidate/score/action mismatch {d}")
        step = pilot._plain(env.step_experimental_pp(event["action"], 300., "annealed", event["temperature"], event["uniform"]))
        state = step["observation"]
        expected = pilot.apply_state_delta(expected, event["delta"])
        if pilot.state_fingerprint(state) != pilot.state_fingerprint(expected): raise ValueError(f"path/counter mismatch {d}")
        for key in ("repair_order", "neighborhood", "replan_success", "pp_failure_reason"):
            if step["metrics"][key] != event["metrics"][key]: raise ValueError("PP mismatch: " + key)
    pilot.validate_final(state)
    return dict(status="ok", job_id=job["job_id"], steps=len(saved["events"]),
                plan_sha256=job["plan_sha256"], native_sha256=job["build"]["native_sha256"],
                full_prefix_equal=True, no_ttf=True)


def replay():
    p = verify()
    identity = sha256_file(OUT / "plan.json")
    b = read_json(OUT / "build.json")
    if b["plan_sha256"] != identity or sha256_file(ROOT / b["native_path"]) != b["native_sha256"]:
        raise ValueError("diagnostic native changed")
    for name, field in (("SIPP.cpp", "generated_cpp_sha256"), ("SIPP.o", "object_sha256"), ("micro.json", "micro_sha256")):
        if sha256_file(OUT / name) != b[field]: raise ValueError("diagnostic build evidence changed")
    with pilot._CollectionRunLock(OUT, identity, "neighbor-replay"):
        if (OUT / "replay_started.json").exists(): raise ValueError("attempt exists; inspect before rerun")
        write_json(OUT / "replay_started.json", dict(plan_sha256=identity))
        jobs = [dict(job_id=c["case_id"], case=c, config=p["config"], record=p["records"][c["case_id"]],
                     build=b, plan_sha256=identity) for c in p["cases"]]
        def failed(j, status, error):
            return dict(status=status, job_id=j["job_id"], plan_sha256=identity, error=str(error))
        rows = pilot._run_jobs(replay_worker, jobs, workers=8, phase="neighbor-replay", output_root=OUT,
            run_fingerprint=identity, timeout_seconds=900, failure_result=failed,
            on_result=lambda r: write_json(OUT / "replay" / (r["job_id"] + ".json"), r), stop_on_failure=True)
        complete = len(rows) == 8 and all(r["status"] == "ok" for r in rows)
        result = dict(complete=complete, cases=len(rows), steps=sum(r.get("steps", 0) for r in rows),
            plan_sha256=identity, native_sha256=b["native_sha256"], no_ttf=True,
            files={r["job_id"] + ".json": sha256_file(OUT / "replay" / (r["job_id"] + ".json")) for r in rows},
            decision="eligible_for_bounded_pp_timing" if complete else "stop_native_variant")
        if complete and result["steps"] != 792: raise ValueError("replay step coverage changed")
        write_json(OUT / "report.json", result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "verify", "build", "replay"))
    phase = parser.parse_args().phase
    print(json.dumps(globals()[phase](), indent=2))
