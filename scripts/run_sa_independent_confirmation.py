"""Frozen SA confirmation on new maps; reuse the registered pilot timing engine."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, write_json, write_jsonl
from experiments.repair_collection import _CollectionRunLock, _load_dataset_rows, _plain, _run_jobs, state_fingerprint
from generators.dataset import generate_dataset
from scripts import run_sa_wall_clock as pilot

OUT = ROOT / "build/sa-independent-confirmation-v1"
CONFIG = ROOT / "configs/sa_independent_confirmation_v1.json"
PILOT = ROOT / "build/sa-wall-clock-pilot-v1"
FILE_KEYS = ("map_file", "map_metadata_file", "task_file", "scenario_file", "instance_file", "legacy_instance_file")


@contextmanager
def output_context():
    previous = pilot.OUT
    pilot.OUT = OUT
    try:
        yield
    finally:
        pilot.OUT = previous


def configuration():
    c = read_json(CONFIG)
    if (c["map_count"], c["densities"], c["solver_seeds"], c["timed_workers"], c["budget_seconds"],
        c["fuse_seconds"], c["max_repair_iterations"], c["initial_temperature"], c["cooling"]) != (
            8, [.15, .25], [101, 103], 1, 60, 180, 0, 1000, .99):
        raise ValueError("frozen protocol changed")
    if c["replacement_permitted"] or not c["zero_conflict_retained"] or c["automatic_default_promotion"]:
        raise ValueError("selection/promotion boundary changed")
    return c


def historical_inventory(root=ROOT / "build", exclude=OUT):
    seeds, map_hashes, manifests, missing = set(), set(), {}, set()
    for parent, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {"venv-graph", "linux", "windows", ".git", "__pycache__"}
                   and not d.startswith("venv") and not (Path(parent) / d).is_symlink()
                   and (Path(parent) / d).resolve() != exclude.resolve()]
        if "manifest.jsonl" not in files:
            continue
        path = Path(parent) / "manifest.jsonl"
        manifests[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            for key in ("map_seed", "task_seed"):
                if isinstance(row.get(key), int):
                    seeds.add(row[key])
            if not row.get("map_file"):
                continue
            file = path.parent / row["map_file"]
            if file.is_file():
                map_hashes.add(sha256_file(file))
            else:
                missing.add(file.relative_to(ROOT).as_posix())
    return dict(seeds=sorted(seeds), map_hashes=sorted(map_hashes), manifests=manifests,
                missing_map_files=sorted(missing), scope="retained local manifest.jsonl datasets; unavailable historical files listed")


def register():
    if OUT.exists():
        raise ValueError("registration exists; use resume phases, never overwrite")
    c = configuration()
    for p, h in [(ROOT / c["dataset_base"], c["dataset_base_sha256"]),
                 (PILOT / "plan.json", c["pilot_plan_sha256"]),
                 (PILOT / "analysis.json", c["pilot_analysis_sha256"])]:
        if sha256_file(p) != h:
            raise ValueError("frozen source changed: " + str(p))
    old = read_json(PILOT / "plan.json")
    inputs = dict(old["inputs"])
    for name, h in inputs.items():
        if sha256_file(ROOT / name) != h:
            raise ValueError("pilot runtime/input changed: " + name)
    additions = [CONFIG, Path(__file__), ROOT / "docs/SA_INDEPENDENT_CONFIRMATION_PROTOCOL_ZH.md",
                 ROOT / c["dataset_base"], PILOT / "plan.json", PILOT / "analysis.json"]
    additions += [p for p in (ROOT / "generators").rglob("*.py")]
    for p in additions:
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    history = historical_inventory()
    rng = random.Random(c["master_seed"])
    map_masters = [rng.randrange(1, 2**31) for _ in range(c["map_count"])]
    if len(set(map_masters)) != 8 or set(map_masters) & set(history["seeds"]):
        raise ValueError("new generator seeds collide; stop, do not redraw")
    config = deepcopy(old["config"])
    config["dataset"]["output"] = OUT.relative_to(ROOT).as_posix() + "/dataset"
    registration = dict(schema="lns2.sa_independent_registration.v1", protocol=c, inputs=inputs,
        history=history, map_master_seeds=map_masters, config=config, legacy_config=old["legacy_config"],
        proposal=old["cases"][0]["proposal"],
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    write_json(OUT / "registration.json", registration)
    return dict(registered=True, maps=8, tasks=16, paired_cases=32, timed_episodes=128,
                history_manifests=len(history["manifests"]), missing_historical_maps=len(history["missing_map_files"]),
                maximum_solver_minutes=128, maximum_fuse_minutes=384)


def registration():
    r = read_json(OUT / "registration.json")
    if r["protocol"] != configuration():
        raise ValueError("registration/config mismatch")
    for name, h in r["inputs"].items():
        if sha256_file(ROOT / name) != h:
            raise ValueError("registered input changed: " + name)
    return r


def generation_worker(job):
    folder = Path(job["output"])
    if folder.exists():
        raise ValueError("partial generated shard exists; audit required")
    c = deepcopy(job["base"])
    c.update(master_seed=job["seed"], map_id_prefix=f"sa_independent_v1_m{job['index']:02d}",
             tasks_per_map=2, splits={"confirmation": {"layout_counts": {"station_centric": 1}}})
    template = next(v for v in c["task_variants"] if v["name"] == "balanced_od_d15")
    c["task_variants"] = []
    for d in (.15, .25):
        v = deepcopy(template)
        v.update(name=f"balanced_od_d{int(d*100)}", load_band="ordinary" if d == .15 else "higher_density")
        v["task"]["agent_density"] = d
        c["task_variants"].append(v)
    generate_dataset(c, output_override=folder)
    rows = _load_dataset_rows(folder, ["confirmation"])
    files = {}
    for row in rows:
        for key in FILE_KEYS:
            file = folder / "confirmation" / row[key]
            files[file.relative_to(ROOT).as_posix()] = sha256_file(file)
            row[key] = f"shards/{job['index']:02d}/confirmation/" + row[key]
    return dict(status="ok", job_id=job["job_id"], rows=rows, files=files)


def check_isolation(rows, history, dataset):
    map_seeds = {r["map_seed"] for r in rows}
    task_seeds = [r["task_seed"] for r in rows]
    if len(map_seeds) != 8 or len(set(task_seeds)) != 16 or map_seeds & set(task_seeds):
        raise ValueError("within-study seed duplication")
    if (map_seeds | set(task_seeds)) & set(history["seeds"]):
        raise ValueError("historical seed overlap; no replacement allowed")
    hashes = {r["map_id"]: sha256_file(dataset / r["split"] / r["map_file"]) for r in rows}
    if len(set(hashes.values())) != 8 or set(hashes.values()) & set(history["map_hashes"]):
        raise ValueError("map geometry duplicate; no replacement allowed")
    for m in hashes:
        rs = [r for r in rows if r["map_id"] == m]
        if len(rs) != 2 or {r["task_variant"] for r in rs} != {"balanced_od_d15", "balanced_od_d25"}:
            raise ValueError("unpaired density tasks")
    return hashes


def parallel_phase(phase, worker, jobs, timeout, r):
    root = OUT / phase
    identity = sha256_file(OUT / "registration.json")
    existing, pending = [], []
    with _CollectionRunLock(OUT, identity, phase):
        for job in jobs:
            path = root / (job["job_id"] + ".json")
            if path.exists():
                row = read_json(path)
                if row.get("registration_sha256") != identity or row.get("status") != "ok":
                    raise ValueError("prior preparation error/identity mismatch requires audit")
                if row.get("integrity_sha256") != pilot.digest({k:v for k,v in row.items() if k != "integrity_sha256"}):
                    raise ValueError("preparation result tampered")
                existing.append(row)
            else:
                pending.append(job)
        def save(row):
            row["registration_sha256"] = identity
            row["integrity_sha256"] = pilot.digest(row)
            write_json(root / (row["job_id"] + ".json"), row)
            print(f"{phase}: {row['job_id']} {row['status']}", flush=True)
        result = existing + _run_jobs(worker, pending, workers=r["protocol"]["preparation_workers"],
            phase=phase, output_root=OUT, run_fingerprint=identity, timeout_seconds=timeout,
            on_result=save, stop_on_failure=True)
    if len(result) != len(jobs) or any(x["status"] != "ok" for x in result):
        raise ValueError("preparation incomplete")
    return result


def generate():
    r = registration()
    ds = OUT / "dataset"
    jobs = [dict(job_id=f"map-{i:02d}", index=i, seed=seed,
                 base=read_json(ROOT / r["protocol"]["dataset_base"]),
                 output=str(ds / "confirmation/shards" / f"{i:02d}"))
            for i, seed in enumerate(r["map_master_seeds"])]
    results = parallel_phase("generation", generation_worker, jobs, 300, r)
    rows = sorted([row for result in results for row in result["rows"]], key=lambda x:x["task_id"])
    files = {k:v for result in results for k,v in result["files"].items()}
    for name, h in files.items():
        if sha256_file(ROOT / name) != h:
            raise ValueError("generated asset changed")
    hashes = check_isolation(rows, r["history"], ds)
    write_jsonl(ds / "confirmation/manifest.jsonl", rows)
    files[(ds / "confirmation/manifest.jsonl").relative_to(ROOT).as_posix()] = sha256_file(ds / "confirmation/manifest.jsonl")
    report = dict(complete=True, maps=len(hashes), tasks=len(rows), files=files, map_hashes=hashes,
                  history_missing_files=r["history"]["missing_map_files"], replacement=False)
    write_json(OUT / "dataset_report.json", report)
    return {k:v for k,v in report.items() if k not in ("files", "map_hashes", "history_missing_files")}


def reset_worker(job):
    env = pilot.make_env(job)
    s = _plain(env.reset(seed=job["case"]["solver_seed"]))
    if not s["initial_solution_complete"]:
        raise ValueError("incomplete PP reset; no task deletion allowed")
    pilot.validate_final(s)
    case = dict(job["case"], expected_initial_fingerprint=state_fingerprint(s),
        expected_endpoints=[[a["path"][0], a["path"][-1]] for a in sorted(s["agents"], key=lambda a:a["id"])])
    return dict(status="ok", job_id=job["job_id"], case=case, conflicts=s["num_of_colliding_pairs"],
                agent_count=len(s["agents"]), zero_conflict=s["feasible"])


def admit():
    r = registration()
    report = read_json(OUT / "dataset_report.json")
    for name, h in report["files"].items():
        if sha256_file(ROOT / name) != h:
            raise ValueError("dataset changed")
    rows = _load_dataset_rows(OUT / "dataset", ["confirmation"])
    jobs = []
    for row in rows:
        for s in r["protocol"]["solver_seeds"]:
            case = dict(case_id=f"{row['task_id']}-seed{s}", map_id=row["map_id"], task_id=row["task_id"],
                        solver_seed=s, row=row, proposal=r["proposal"])
            jobs.append(dict(job_id=case["case_id"], case=case, config=r["config"], budget=pilot.BUDGET))
    results = sorted(parallel_phase("reset", reset_worker, jobs, pilot.FUSE, r), key=lambda x:x["job_id"])
    if len(results) != 32:
        raise ValueError("fixed reset cohort incomplete")
    summary = dict(complete=True, resets=32, errors=0, zero_conflict=sum(x["zero_conflict"] for x in results),
        cases=[{k:x[k] for k in ("job_id", "conflicts", "agent_count", "zero_conflict")} for x in results])
    write_json(OUT / "reset_report.json", summary)
    if (OUT / "plan.json").exists():
        plan = read_json(OUT / "plan.json")
        if plan["cases"] != [x["case"] for x in results]:
            raise ValueError("existing bound cases changed")
        return summary
    inputs = dict(r["inputs"], **report["files"])
    for p in [OUT / "registration.json", OUT / "reset_report.json", OUT / "dataset_report.json"]:
        inputs[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    legacy = deepcopy(r["legacy_config"])
    legacy["dataset"]["output"] = r["config"]["dataset"]["output"]
    plan = dict(schema="lns2.sa_independent_bound.v1", config=r["config"], legacy_config=legacy,
        cases=[x["case"] for x in results], inputs=inputs, arms=list(pilot.ARMS), orders=pilot.ORDERS,
        budget=pilot.BUDGET, fuse=pilot.FUSE, workers=1, max_repair_iterations=0,
        initial_temperature=1000, cooling=.99, development_only=False,
        claim_scope="new layouts within station_centric generator only; no cross-family or success guarantee")
    write_json(OUT / "plan.json", plan)
    return summary


def verify():
    registration()
    with output_context():
        p = pilot.verify()
    if len(p["cases"]) != 32 or len({c["map_id"] for c in p["cases"]}) != 8:
        raise ValueError("bound cohort changed")
    return p


def analyze():
    p = verify()
    with output_context():
        result = pilot.analyze(p)
    result.update(schema="lns2.sa_independent_analysis.v1", development_only=False,
                  no_default_promotion=True, claim_scope=p["claim_scope"])
    primary = next(x for x in result["comparisons"] if x["base"] == "dual16" and x["challenger"] == "dual16_sa")
    primary["confirmation_mean_ttf_passed"] = (primary["improvement_percent"] >= 5 and
        primary["paired_seconds_ci95"][1] < 0 and primary["successes"][1] >= primary["successes"][0])
    result["success_noninferiority_established"] = False
    write_json(OUT / "analysis.json", result)
    return dict(summary=result["summary"], primary=primary, no_default_promotion=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("register", "generate", "admit", "admission", "parity", "smoke",
                                         "verify", "collect", "resume", "stop", "analyze"))
    args = parser.parse_args()
    if args.phase == "stop":
        write_json(OUT / "STOP_AFTER_EPISODE.json", {"requested": True})
        result = {"stop_after_current_episode": True}
    elif args.phase in ("register", "generate", "admit", "analyze"):
        result = globals()[args.phase]()
    else:
        p = verify()
        if args.phase == "resume":
            (OUT / "STOP_AFTER_EPISODE.json").unlink(missing_ok=True)
        with output_context():
            result = ({"verified": True, "timed_jobs": len(pilot.jobs(p, "timed"))}
                      if args.phase == "verify" else pilot.execute(p, "timed" if args.phase in ("collect", "resume") else args.phase))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
