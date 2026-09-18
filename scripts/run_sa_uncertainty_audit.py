"""Existing-data map-bootstrap audit; no solver, native, or production policy."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, json_fingerprint, contained_file
from experiments.sa_paired_completion import require, validate_dataset, MODEL_PARAMS
from experiments.sa_uncertainty_audit import fit_member, map_draws, state_result, summarize
from scripts.run_sa_paired_closed_loop import once, sealed, check_seal
from scripts.audit_sa_history_information import atomic

CONFIG = ROOT / "configs/sa_uncertainty_audit.json"
CODE = ["experiments/sa_uncertainty_audit.py", "scripts/run_sa_uncertainty_audit.py",
        "tests/evaluation/test_sa_uncertainty_audit.py", "docs/SA_UNCERTAINTY_AUDIT_PROTOCOL_ZH.md"]


def prepare():
    from scripts.run_sa_low_complexity_ranker import verify as verify_source
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    require(not out.exists(), "existing output; never overwrite")
    require(cfg["members"] == 20 and cfg["workers"] == 20 and not cfg["new_solver_calls"]
            and not cfg["automatic_promotion"], "registered scope changed")
    source, folder = verify_source()
    require(folder == ROOT / cfg["source"], "source identity")
    data = validate_dataset(read_json(ROOT / cfg["dataset"]))
    require(len(data["states"]) == cfg["states"] and data["horizon"] == 32 and data["trial_count"] == 8,
            "dataset contract changed")
    require(all(t["completed"] is not None for s in data["states"] for c in s["candidates"] for t in c["trials"]),
            "censored labels; no imputation")
    maps = sorted({s["map_id"] for s in data["states"]})
    require(maps == source["maps"] and len(maps) == cfg["maps"], "map cohort changed")
    inputs = dict(source["files"])
    for path in [CONFIG, *[ROOT / p for p in CODE], ROOT / cfg["dataset"], folder / "plan.json", folder / "report.json",
                 ROOT / "experiments/sa_paired_completion.py", ROOT / "experiments/_common.py"]:
        name, digest = path.relative_to(ROOT).as_posix(), sha256_file(path)
        require(name not in inputs or inputs[name] == digest, "conflicting source SHA")
        inputs[name] = digest
    plan = dict(config=cfg, inputs=inputs, maps=maps, baseline=source["baseline"],
                baseline_binding=source["source_binding"], params=MODEL_PARAMS,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = json_fingerprint(plan)
    once(out / "plan.json", plan)
    return dict(binding=plan["binding"], fits=len(maps)*(cfg["members"]+1), workers=cfg["workers"], new_solver_calls=0)


def verify():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    plan = read_json(out / "plan.json")
    require(plan["config"] == cfg and plan["binding"] == json_fingerprint({k: v for k, v in plan.items() if k != "binding"}),
            "registration changed")
    for name, digest in plan["inputs"].items():
        require(sha256_file(contained_file(ROOT, name, field="input")) == digest, "input changed: " + name)
    require(plan["params"] == MODEL_PARAMS, "model parameters changed")
    return plan, out


def result_path(out, held, member):
    return out / "fits" / f"{held}-{member+1:02d}.json"


def check_fit(plan, data, held, member, row):
    check_seal(row)
    require(row["binding"] == plan["binding"] and row["held"] == held and row["member"] == member,
            "fit binding changed")
    draws = map_draws(data, held, member, plan["config"])
    require(row["map_draws"] == draws and row["train_maps"] == sorted(set(draws)), "bootstrap membership changed")
    require(row["train_ids"] == sorted(s["state_id"] for s in data["states"] if s["map_id"] in draws), "training leakage")
    require([r["state_id"] for r in row["predictions"]] == sorted(s["state_id"] for s in data["states"] if s["map_id"] == held),
            "held prediction coverage")
    if member == -1:
        old = check_seal(read_json(ROOT / plan["baseline"][held]))
        require(old["binding"] == plan["baseline_binding"] and old["held"] == held, "baseline binding")
        expected = {r["state_id"]: r for r in old["rows"] if r["held"]}
        for r in row["predictions"]:
            ref = expected[r["state_id"]]
            require(r["selected"] == ref["selected"] and r["scores"] == ref["scores"], "baseline reproduction mismatch")
    return row


def worker(job):
    data, held, member, plan = job
    start = time.monotonic()
    result = fit_member(data, held, member, plan["config"])
    return sealed(dict(result, binding=plan["binding"], seconds_diagnostic=time.monotonic()-start))


def train(resume=False):
    from experiments.repair_collection import _CollectionRunLock
    plan, out = verify()
    data = read_json(ROOT / plan["config"]["dataset"])
    require(resume or not (out / "fits").exists(), "existing fits require --resume")
    keys = [(held, member) for held in plan["maps"] for member in range(-1, plan["config"]["members"])]
    with _CollectionRunLock(out, plan["binding"], "uncertainty-training"):
        pending = []
        for held, member in keys:
            path = result_path(out, held, member)
            if path.exists():
                check_fit(plan, data, held, member, read_json(path))
            else:
                pending.append((data, held, member, plan))
        done = len(keys)-len(pending)
        atomic(out / "run_status.json", dict(binding=plan["binding"], status="running", done=done, total=len(keys)))
        try:
            with ProcessPoolExecutor(max_workers=plan["config"]["workers"]) as pool:
                for offset in range(0, len(pending), plan["config"]["workers"]):
                    if (out / "STOP_AFTER_BATCH").exists():
                        break
                    batch = pending[offset:offset+plan["config"]["workers"]]
                    for row in pool.map(worker, batch):
                        check_fit(plan, data, row["held"], row["member"], row)
                        once(result_path(out, row["held"], row["member"]), row)
                        done += 1
                        progress = dict(done=done, total=len(keys), held=row["held"], member=row["member"])
                        with (out / "progress.jsonl").open("a", encoding="utf8") as stream:
                            stream.write(json.dumps(progress)+"\n")
                        atomic(out / "run_status.json", dict(binding=plan["binding"], status="running", **progress))
                        print(json.dumps(progress), flush=True)
            verify()
            if done == len(keys):
                for held, member in keys:
                    check_fit(plan, data, held, member, read_json(result_path(out, held, member)))
                once(out / "training.complete.json", sealed(dict(binding=plan["binding"], fits=done)))
            atomic(out / "run_status.json", dict(binding=plan["binding"], status="completed" if done == len(keys) else "paused",
                                                 done=done, total=len(keys)))
        except BaseException as exc:
            atomic(out / "run_status.json", dict(binding=plan["binding"], status="error", done=done, error=repr(exc)))
            raise
    return dict(done=done, total=len(keys), new_solver_calls=0)


def analyze():
    plan, out = verify()
    require(check_seal(read_json(out / "training.complete.json"))["binding"] == plan["binding"], "training incomplete")
    data = read_json(ROOT / plan["config"]["dataset"])
    rows, files = [], {}
    for held in plan["maps"]:
        fits = {}
        for member in range(-1, plan["config"]["members"]):
            path = result_path(out, held, member)
            result = check_fit(plan, data, held, member, read_json(path))
            fits[member] = {r["state_id"]: r for r in result["predictions"]}
            files[path.relative_to(out).as_posix()] = sha256_file(path)
        for s in sorted(data["states"], key=lambda s: s["state_id"]):
            if s["map_id"] == held:
                rows.append(state_result(s, [fits[i][s["state_id"]] for i in range(plan["config"]["members"])],
                                         fits[-1][s["state_id"]], plan["config"]))
    report = summarize(rows, plan["config"])
    require(abs(report["means"]["gbdt"]-plan["config"]["expected_baseline"]) < 1e-12, "historical aggregate mismatch")
    report.update(binding=plan["binding"], files=files, fits=len(files))
    verify()
    once(out / "report.json", sealed(report))
    return {k: v for k, v in report.items() if k not in {"rows", "files", "per_map"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "analyze", "verify"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.phase == "prepare":
        result = prepare()
    elif args.phase == "train":
        result = train(args.resume)
    elif args.phase == "analyze":
        result = analyze()
    else:
        plan, _ = verify()
        result = dict(binding=plan["binding"], registered_files=len(plan["inputs"]))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
