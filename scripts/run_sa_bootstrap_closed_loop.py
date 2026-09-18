"""Bounded four-arm bootstrap selection diagnostic, never formal TTF."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, read_jsonl, sha256_file, json_fingerprint
from experiments.sa_paired_completion import require, validate_dataset
from experiments.sa_paired_closed_loop import portable_payload, portable_model, stop_reason
from experiments.sa_bootstrap_closed_loop import ARMS, choose, fit_full_members, summarize
from scripts import run_sa_linear_closed_loop as previous
from scripts.run_sa_paired_closed_loop import (
    once, sealed, check_seal, paired_tasks, reset_worker, receipt_valid,
)
from scripts.audit_sa_history_information import atomic

CONFIG = ROOT / "configs/sa_bootstrap_closed_loop.json"


def select_cases(cases, cfg):
    maps = sorted({c["map_id"] for c in cases})
    require(len(maps) == cfg["maps"], "map count")
    return [min((c for c in cases if c["map_id"] == m),
                key=lambda c: (json_fingerprint([cfg["case_seed"], m, c["task_id"]]), c["task_id"]))
            for m in maps]


def prepare():
    from scripts import run_sa_uncertainty_audit as uncertainty
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    require(not out.exists(), "existing registration; no overwrite")
    source, source_out = previous.verify()
    require(source_out == ROOT / cfg["source"], "source identity")
    u, uout = uncertainty.verify()
    require(uout == ROOT / cfg["uncertainty_source"], "uncertainty source identity")
    uncertainty.verify_outputs(u, uout)
    previous.model_receipt(source, source_out)
    cases = select_cases(previous.cases_verified(source, source_out), cfg)
    data = validate_dataset(read_json(ROOT / cfg["training_index"]))
    require(not {s["map_id"] for s in data["states"]} & {c["map_id"] for c in cases}, "training map overlap")
    require(cfg["arms"] == list(ARMS) and cfg["workers"] == 20 and not cfg["formal_ttf"], "scope")
    require(cfg["candidate_seed"] == source["config"]["candidate_seed"] and cfg["candidate_limit"] == 4,
            "candidate definition changed")
    q = check_seal(read_json(source_out / "qualification.json"))
    keys = {key for _, _, key in paired_tasks(cases, cfg)}
    require(len(keys) == cfg["paired_tasks"], "paired task count")
    anchors = {r["job_id"]: r["initial"] for r in q["results"] if r["job_id"] in keys}
    require(q["passed"] and set(anchors) == keys, "source qualification incomplete")
    inputs = dict(source["inputs"])
    inputs.update(u["inputs"])
    for receipt in (check_seal(read_json(source_out / "cases.json")),):
        inputs.update(receipt["files"])
    paths = [CONFIG, Path(__file__), ROOT / "experiments/sa_bootstrap_closed_loop.py",
             ROOT / "tests/evaluation/test_sa_bootstrap_closed_loop.py",
             ROOT / "tests/evaluation/test_sa_bootstrap_runner.py",
             ROOT / "docs/SA_BOOTSTRAP_CLOSED_LOOP_PROTOCOL_ZH.md",
             source_out / "plan.json", source_out / "cases.json", source_out / "qualification.json",
             source_out / "model/receipt.json", source_out / "model/gbdt.json",
             source_out / "model/metadata.json", uout / "plan.json", uout / "report.json"]
    inputs.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths})
    for name, digest in inputs.items():
        require(sha256_file(ROOT / name) == digest, "input changed: " + name)
    plan = dict(config=cfg, inputs=inputs, cases=cases, anchors=anchors,
                native_file=source["native_file"], native_sha256=source["native_sha256"],
                source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                no_ttf=True, no_new_independent_confirmation=True)
    plan["binding"] = json_fingerprint(plan)
    once(out / "plan.json", plan)
    return dict(binding=plan["binding"], maps=len(cases), paired_tasks=len(keys), episodes=4*len(keys),
                maximum_repairs=4*len(keys)*cfg["max_decisions"], no_ttf=True)


def verify():
    cfg = read_json(CONFIG)
    out = ROOT / cfg["output"]
    p = read_json(out / "plan.json")
    require(p["config"] == cfg and p["binding"] == json_fingerprint({k:v for k,v in p.items() if k != "binding"}),
            "plan identity changed")
    for name, digest in p["inputs"].items():
        require(sha256_file(ROOT / name) == digest, "input changed: " + name)
    return p, out


def models_load(out, native=True):
    names = read_json(out / "model/metadata.json")["feature_names"]
    models = dict(gbdt=portable_model(read_json(out / "model/gbdt.json"), names, native=native),
                  members=[portable_model(read_json(out / f"model/member-{i:02d}.json"), names, native=native)
                           for i in range(20)])
    if native:
        require(all(m.estimator.model.inference_backend == "native-portable-tree"
                    for m in [models["gbdt"]] + models["members"]), "native inference required for every model")
    return models


def model_receipt(p, out):
    r = previous.model_receipt(p, out)
    expected = {"gbdt.json", "metadata.json", "fixtures.json"} | {f"member-{i:02d}.json" for i in range(20)}
    require(set(r["files"]) == expected, "incomplete model receipt")
    require({f.name for f in (out / "model").iterdir()} == expected | {"receipt.json"}, "unexpected model files")
    return r


def freeze():
    from lns2_selector.runtime.fingerprints import semantic_fingerprint
    p, out = verify()
    if (out / "model/receipt.json").exists():
        return model_receipt(p, out)
    require(not (out / "model").exists(), "partial model publication requires audit")
    data = read_json(ROOT / p["config"]["training_index"])
    fitted = fit_full_members(data, p["config"])
    source = ROOT / p["config"]["source"] / "model"
    payload = read_json(source / "gbdt.json")
    gbdt = portable_model(payload, data["feature_names"], native=False)
    once(out / "model/gbdt.json", payload)
    require(sha256_file(out / "model/gbdt.json") == sha256_file(source / "gbdt.json"), "baseline changed")
    portable = []
    for i, model in enumerate(fitted["models"]):
        payload = portable_payload(model)
        payload["name"] = f"sa_map_bootstrap_member_{i:02d}"
        payload["semantic_fingerprint"] = semantic_fingerprint({k:v for k,v in payload.items()
                                                               if k not in {"schema", "semantic_fingerprint"}})
        portable.append(portable_model(payload, data["feature_names"], native=False))
        once(out / f"model/member-{i:02d}.json", payload)
    fixtures = []
    for state in data["states"]:
        expected = [m.rank(state) for m in fitted["models"]]
        require(expected == [m.rank(state) for m in portable], "sklearn/Python portable scores mismatch")
        clean = {k:state[k] for k in ("state_id", "anchor_id", "agent_ids")}
        clean["candidates"] = [{k:c[k] for k in ("candidate_id", "agents", "features")} for c in state["candidates"]]
        fixtures.append(dict(state=clean, gbdt=gbdt.rank(state), members=expected))
    meta = read_json(source / "metadata.json")
    require(meta["training_sha256"] == sha256_file(ROOT / p["config"]["training_index"]), "training identity")
    meta = {k:meta[k] for k in ("feature_names", "ranges", "training_sha256")}
    meta.update(bootstrap=fitted["metadata"], production_allowed=False)
    once(out / "model/metadata.json", meta)
    once(out / "model/fixtures.json", fixtures)
    once(out / "model/receipt.json", sealed(dict(binding=p["binding"], files={
        f.name:sha256_file(f) for f in sorted((out / "model").iterdir())})))
    return dict(members=20, training_states=47, fixtures=47, new_labels=0)


def verify_models():
    p, out = verify()
    model_receipt(p, out)
    models = models_load(out)
    for row in read_json(out / "model/fixtures.json"):
        require(models["gbdt"].rank(row["state"]) == row["gbdt"], "baseline portable mismatch")
        require([m.rank(row["state"]) for m in models["members"]] == row["members"], "member portable mismatch")
    backend = models["gbdt"].estimator.model.inference_backend
    require(all(m.estimator.model.inference_backend == backend for m in models["members"]), "mixed inference backend")
    r = dict(binding=p["binding"], backend=backend, fixtures=47, models=21, exact=True,
             receipt_sha256=sha256_file(out / "model/receipt.json"))
    once(out / f"model-verification-{backend}.json", r)
    return r


def readiness(p, out):
    model_receipt(p, out)
    r = read_json(out / "model-verification-native-portable-tree.json")
    require(r["binding"] == p["binding"] and r["receipt_sha256"] == sha256_file(out / "model/receipt.json")
            and r["exact"], "native model verification required")


def protected_reset(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job["parent_pid"])
    return reset_worker(job)


def qualify():
    from experiments.repair_collection import _run_jobs, _CollectionRunLock
    p, out = verify()
    readiness(p, out)
    if (out / "qualification.json").exists():
        return qualification(p, out)
    jobs = [dict(job_id=k, case=c, solver_seed=s, plan=p, parent_pid=os.getpid())
            for c,s,k in paired_tasks(p["cases"], p["config"])]
    with _CollectionRunLock(out, p["binding"], "bootstrap-reset"):
        rows = _run_jobs(protected_reset, jobs, p["config"]["workers"], phase="reset", output_root=out / "reset-progress",
                         run_fingerprint=p["binding"], timeout_seconds=p["config"]["fuse_seconds"],
                         stop_on_failure=True, failure_result=previous.job_failure)
        require(len(rows) == len(jobs) and all(r["status"] == "ok" for r in rows), "reset error")
        require({r["job_id"]:r["initial"] for r in rows} == p["anchors"], "historical reset mismatch")
        once(out / "qualification.json", sealed(dict(binding=p["binding"], passed=True, results=rows)))
    return qualification(p, out)


def qualification(p, out):
    r = check_seal(read_json(out / "qualification.json"))
    require(r["binding"] == p["binding"] and r["passed"] and len(r["results"]) == len(p["anchors"])
            and all(x["status"] == "ok" for x in r["results"])
            and {x["job_id"]:x["initial"] for x in r["results"]} == p["anchors"], "qualification identity")
    return r


@contextmanager
def adapted_runner(job):
    """Only in isolated workers: reuse the unchanged PP/SA and trace implementation."""
    old_load, old_choose = previous.models_load, previous.choose
    decision = 0
    def select(arm, candidates, anchor, models, features, known_ids):
        nonlocal decision
        result = choose(arm, candidates, anchor, models["gbdt"], models["members"], features, known_ids,
                        job["pair_id"], decision, job["plan"]["config"]["selection_seed"])
        decision += 1
        return result
    previous.models_load, previous.choose = models_load, select
    try:
        yield
    finally:
        previous.models_load, previous.choose = old_load, old_choose


def episode_worker(job):
    model_receipt(job["plan"], ROOT / job["plan"]["config"]["output"])
    with adapted_runner(job):
        return previous.episode_worker(job)


def validate_result_identity(result, job):
    require(result["binding"] == job["plan"]["binding"]
            and result["job_id"] == job["job_id"] and result["pair_id"] == job["pair_id"]
            and result["arm"] == job["arm"] and result["solver_seed"] == job["solver_seed"]
            and result["map_id"] == job["case"]["map_id"]
            and result["initial_fingerprint"] == job["expected_initial"], "result schedule identity")


def incomplete(event):
    return event is not None and (event["metrics"]["pp_failure_reason"] == "time_limit"
                                 or not event["metrics"]["acceptance_evaluated"])


def audit_episode(job):
    from scripts import run_sa_path_quality as q
    p, folder = job["plan"], Path(job["folder"])
    result = receipt_valid(folder, p)
    validate_result_identity(result, job)
    state = read_json(folder / "initial.json")["state"]
    nodes = state["low_level"]["generated"]
    last = None
    for count, event in enumerate(read_jsonl(folder / "trace.jsonl")):
        require(not incomplete(last), "repair continued after incomplete PP")
        require(stop_reason(state, count, state["low_level"]["generated"]-nodes, 0, p["config"]) is None,
                "repair executed after work budget or feasibility")
        state = q.apply_state_delta(state, event["delta"])
        last = event
    require((result["stop"] == "incomplete_pp") == incomplete(last), "invalid PP censoring")
    if result["stop"] == "wall_safety":
        require(result["diagnostic_seconds"] >= p["config"]["episode_seconds"], "invalid wall censoring")
    with adapted_runner(job):
        return previous.audit_episode(job)


def schedule(p):
    return [dict(job_id=f"{k}-{arm}", pair_id=k, case=c, solver_seed=s, arm=arm, expected_initial=p["anchors"][k],
                 plan=p, parent_pid=os.getpid())
            for c,s,k in paired_tasks(p["cases"], p["config"]) for arm in ARMS]


def execution(p, out):
    return dict(binding=p["binding"], qualification_sha256=sha256_file(out / "qualification.json"),
                model_receipt_sha256=sha256_file(out / "model/receipt.json"))


def smoke_record(p, out):
    return sealed(dict(binding=p["binding"], execution=execution(p, out), audited=4, files={
        (Path("episodes") / j["job_id"] / name).as_posix():sha256_file(out / "episodes" / j["job_id"] / name)
        for j in schedule(p)[:4] for name in ("initial.json", "trace.jsonl", "result.json", "receipt.json")}))


def require_smoke(p, out):
    require((out / "smoke-audit.json").exists(), "four-arm smoke audit required before expansion")
    require(read_json(out / "smoke-audit.json") == smoke_record(p, out), "smoke evidence changed")


def collect(resume=False, limit=None):
    from experiments.repair_collection import _run_jobs, _CollectionRunLock
    p, out = verify()
    readiness(p, out)
    qualification(p, out)
    cfg = p["config"]
    require(resume or not (out / "episodes").exists(), "existing episodes require --resume")
    once(out / "execution_registration.json", execution(p, out))
    jobs = schedule(p)
    if limit is not None:
        require(0 < limit <= len(jobs) and limit % len(ARMS) == 0, "limit must preserve full paired arms")
        jobs = jobs[:limit]
    if len(jobs) > 4:
        require_smoke(p, out)
    pending = []
    for j in jobs:
        folder = out / "episodes" / j["job_id"]
        if (folder / "receipt.json").exists():
            r = receipt_valid(folder, p)
            validate_result_identity(r, j)
        else:
            require(not folder.exists(), "partial episode requires inspection, not automatic retry")
            pending.append(j)
    with _CollectionRunLock(out, p["binding"], "bootstrap-closed-loop"):
        atomic(out / "run_status.json", dict(status="running", binding=p["binding"]))
        try:
            for offset in range(0, len(pending), cfg["workers"]):
                if (out / "STOP_AFTER_BATCH").exists():
                    break
                batch = pending[offset:offset+cfg["workers"]]
                def record(row):
                    with (out / "progress.jsonl").open("a", encoding="utf8") as f:
                        f.write(json.dumps(row)+"\n")
                    print("EPISODE", row, flush=True)
                token = json_fingerprint([j["job_id"] for j in batch])[:16]
                rows = _run_jobs(episode_worker, batch, cfg["workers"], phase="batch-"+token,
                                 output_root=out / "progress" / token, run_fingerprint=p["binding"],
                                 timeout_seconds=cfg["fuse_seconds"], stop_on_failure=True, on_result=record,
                                 failure_result=previous.job_failure)
                require(len(rows) == len(batch) and all(r["status"] == "ok" for r in rows), "batch error; audit before retry")
            completed = sum((out / "episodes" / j["job_id"] / "receipt.json").exists() for j in schedule(p))
            atomic(out / "run_status.json", dict(status="completed" if completed == len(schedule(p)) else "paused",
                                                 completed=completed, binding=p["binding"]))
        except BaseException as exc:
            atomic(out / "run_status.json", dict(status="error", error=repr(exc), binding=p["binding"]))
            raise
    return dict(completed=completed, total=len(schedule(p)), no_ttf=True)


def analyze(limit=None):
    p, out = verify()
    readiness(p, out)
    qualification(p, out)
    require(read_json(out / "execution_registration.json") == execution(p, out), "execution identity")
    jobs = schedule(p)
    if limit is not None:
        require(0 < limit <= len(jobs) and limit % len(ARMS) == 0, "invalid audit limit")
        jobs = jobs[:limit]
    for job in jobs:
        job["folder"] = str(out / "episodes" / job["job_id"])
    with ProcessPoolExecutor(max_workers=p["config"]["workers"]) as pool:
        rows = list(pool.map(audit_episode, jobs))
    for r, j in zip(rows, jobs, strict=True):
        require(r["job_id"] == j["job_id"] and r["arm"] == j["arm"] and r["map_id"] == j["case"]["map_id"]
                and r["pair_id"] == j["pair_id"] and r["solver_seed"] == j["solver_seed"]
                and r["initial_fingerprint"] == j["expected_initial"], "episode identity mismatch")
    if limit is not None:
        if limit == 4:
            once(out / "smoke-audit.json", smoke_record(p, out))
        return dict(audited=len(rows), smoke_only=True, no_ttf=True)
    report = summarize(rows, p["config"])
    report.update(binding=p["binding"], episodes=rows, files={
        (Path(j["folder"]) / n).relative_to(out).as_posix():sha256_file(Path(j["folder"]) / n)
        for j in jobs for n in ("initial.json", "trace.jsonl", "result.json", "receipt.json")})
    verify()
    once(out / "report.json", sealed(report))
    return {k:v for k,v in report.items() if k not in {"files", "episodes", "paired_cases"}}


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "freeze", "verify-model", "qualify", "collect", "analyze", "verify", "request-stop"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.phase == "collect":
        result = collect(args.resume, args.limit)
    elif args.phase == "analyze":
        result = analyze(args.limit)
    elif args.phase == "verify":
        p, out = verify()
        model_receipt(p, out)
        result = analyze() if (out / "report.json").exists() else dict(inputs_verified=len(p["inputs"]))
    elif args.phase == "request-stop":
        p, out = verify()
        once(out / "STOP_AFTER_BATCH", dict(binding=p["binding"], requested=True))
        result = dict(stop_after_current_batch=True)
    else:
        result = {"prepare":prepare, "freeze":freeze, "verify-model":verify_models, "qualify":qualify}[args.phase]()
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
