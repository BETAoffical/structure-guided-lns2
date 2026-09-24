"""Prepare/replay an isolated policy contract; no MAPF collection or optimization."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments import sa_raw_residual_actor as raw
from experiments.sa_onpolicy_actor import select_with_draw
from scripts import audit_sa_policy_expressivity as source
from scripts import run_sa_onpolicy as io
from scripts.audit_sa_history_information import atomic

OUT = ROOT / "build/sa-raw-residual-contract-v1"
ARM = "uncapped_condition"
DRAW_GRID = [i / 16 for i in range(16)]
require = io.require


def prepare():
    require(not OUT.exists(), "output exists; inspect before resume")
    prior = source.verify()
    require((source.OUT / "audit.complete.json").is_file(), "completed expressivity audit required")
    jobs = [j for j in prior["jobs"] if j["arm"] == ARM]
    require(len(jobs) == 96 and len({j["map_id"] for j in jobs}) == 6, "Train scope")
    parent = io.read_json(ROOT / prior["models"][ARM]["path"])
    prototype = raw.initial_bundle(parent, "sa-raw-residual-contract-v1")
    for job in jobs:
        require(source.checked_job(job)["policy_sha256"] == prototype["base_policy_sha256"], "behavior identity")
    inputs = dict(prior["inputs"])
    files = [source.OUT / n for n in ("audit_registration.json", "report.json", "audit.complete.json")]
    files += [ROOT / n for n in ("experiments/sa_raw_residual_actor.py", "scripts/check_sa_raw_residual_contract.py",
        "tests/evaluation/test_sa_raw_residual_actor.py", "docs/SA_RAW_RESIDUAL_CONTRACT_PROTOCOL_ZH.md",
        "experiments/sa_crossfit_update.py")]
    for path in files:
        inputs[path.relative_to(ROOT).as_posix()] = io.sha256_file(path)
    representatives = [min(j["job_id"] for j in jobs if j["map_id"] == m) for m in sorted({j["map_id"] for j in jobs})]
    plan = dict(schema="lns2.sa.raw_contract_check.v1", source_binding=prior["binding"], inputs=inputs,
        jobs=jobs, representative_jobs=representatives, prototype=prototype,
        prototype_sha256=raw.validate_bundle(prototype), no_solver=True, no_training=True, no_ttf=True,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = io.json_fingerprint(plan)
    io.once(OUT / "prototype.initial.json", prototype)
    io.once(OUT / "contract_registration.json", io.sealed(plan))
    return dict(episodes=96, maps=6, no_solver=True, binding=plan["binding"])


def verify():
    plan = io.check_seal(io.read_json(OUT / "contract_registration.json"))
    require(plan["binding"] == io.json_fingerprint({k:v for k,v in plan.items() if k not in ("binding", "integrity")}), "contract binding")
    source.source.verify_inputs(plan)
    require(io.read_json(OUT / "prototype.initial.json") == plan["prototype"] and
            raw.validate_bundle(plan["prototype"]) == plan["prototype_sha256"], "prototype changed")
    if (OUT / "replay.complete.json").exists():
        done = io.check_seal(io.read_json(OUT / "replay.complete.json"))
        require(done["binding"] == plan["binding"] and len(done["files"]) == 96, "replay completeness")
        for job in plan["jobs"]:
            require(done["files"][job["job_id"]] == io.sha256_file(OUT / "episodes" / (job["job_id"] + ".json")), "replay record changed")
        require(done["fixtures_sha256"] == io.sha256_file(OUT / "fixtures.json"), "fixtures changed")
    if (OUT / "contract.complete.json").exists():
        done = io.check_seal(io.read_json(OUT / "contract.complete.json"))
        require(done["binding"] == plan["binding"], "completion binding")
        for name, digest in done["files"].items():
            require(io.sha256_file(OUT / name) == digest, "completed evidence changed")
    return plan


def replay_episode(job, prototype, representative):
    row = source.checked_job(job)
    folder = ROOT / job["folder"]
    for name, digest in row["files"].items():
        require(io.sha256_file(folder / name) == digest, "source changed")
    actor = raw.RawResidualActor(prototype)
    require(actor.base.sha == row["policy_sha256"], "frozen source actor")
    fixtures, errors, count = [], [], 0
    positions = {0, row["decisions"] // 2, row["decisions"] - 1} if representative else set()
    for index, event in enumerate(io.trace_read(folder)):
        require(event["decision"] == index and event["policy_sha256"] == actor.base.sha, "trace discontinuity/identity")
        ids, anchor, fs = event["candidate_ids"], event["anchor_id"], event["features"]
        require(ids == sorted(set(ids)), "canonical source candidates")
        p = actor.probabilities(ids, anchor, fs)
        require(p == actor.base.probabilities(ids, anchor, fs), "initial policy not exactly parent")
        error = max(abs(p[c] - event["probabilities"][c]) for c in ids)
        require(error <= 1e-12 and select_with_draw(p, event["selection_draw"]) == event["selected_id"], "behavior replay mismatch")
        errors.append(error)
        if index in positions:
            draws = [*DRAW_GRID, event["selection_draw"]]
            fixtures.append(dict(id=job["job_id"] + ":" + str(index), synthetic=False, ids=ids, anchor=anchor,
                features=fs, probabilities=p, draws=draws, choices=[select_with_draw(p,d) for d in draws],
                bundle=prototype))
        count += 1
    require(count == row["decisions"], "missing source decisions")
    return dict(job_id=job["job_id"], map_id=job["map_id"], decisions=count, exact_parent=True,
                replay_max_error=max(errors, default=0.), fixtures=fixtures, status="ok")


def constructive_fixtures(prototype):
    base, k = prototype["base"], 20
    names = base["feature_names"]
    x = np.broadcast_to(base["mean"], (k,129)).copy()
    for i in range(k):
        x[i,i] += base["scale"][i]
    fs = [dict(zip(names, row.tolist())) for row in x]
    ids = ["synthetic-" + str(i).zfill(2) for i in range(k)]
    actor = raw.RawResidualActor(prototype)
    base_p = actor.probabilities(ids, ids[0], fs)
    fixtures = []
    for target in range(k):
        bundle = deepcopy(prototype)
        w1, w2 = np.zeros((32,129)), np.zeros((1,32))
        w1[0,target] = 1.
        boost = math.log(999. * (1.-base_p[ids[target]]) / base_p[ids[target]])
        w2[0,0] = boost / math.tanh(1.)
        bundle["correction"] = dict(w1=w1.tolist(), b1=[0.]*32, w2=w2.tolist(), b2=[0.])
        bundle.update(iteration=1, parent_policy=raw.validate_bundle(prototype), update_binding="constructive-not-trained")
        p = raw.RawResidualActor(bundle).probabilities(ids,ids[0],fs)
        require(max(p, key=p.get) == ids[target] and p[ids[target]] > .998, "raw targeted expressivity failed")
        state = dict(ids=ids, anchor=ids[0], features=fs, probabilities=base_p, draw=.5, selected=select_with_draw(base_p,.5))
        episode = dict(episode_id="synthetic", split="train", policy_sha256=raw.validate_bundle(prototype),
                       weight=1., states=[state])
        guard = raw.evaluate_step(prototype,bundle,[episode])
        require(not guard["within_step_budget"], "large constructive jump should not pass update guard")
        fixtures.append(dict(id="construction-" + str(target), synthetic=True, target=ids[target],
            ids=ids, anchor=ids[0], features=fs, probabilities=p, draws=DRAW_GRID,
            choices=[select_with_draw(p,d) for d in DRAW_GRID], bundle=bundle, guard=guard))
    return fixtures


def replay(resume=False, workers=20):
    require(type(workers) is int and 1 <= workers <= 20, "read-only worker limit")
    plan = verify()
    if (OUT / "replay.complete.json").exists():
        require(resume, "completed replay; use resume/verify")
        return dict(verified=True, no_solver=True)
    with source.source.q._CollectionRunLock(OUT, plan["binding"], "raw-contract-replay"):
        pending = []
        for j in plan["jobs"]:
            path = OUT / "episodes" / (j["job_id"] + ".json")
            if path.exists():
                old = io.check_seal(io.read_json(path))
                require(resume and old["binding"] == plan["binding"] and old["job_id"] == j["job_id"] and old["status"] == "ok", "inspect prior replay")
            else:
                pending.append(j)
        atomic(OUT / "run_status.json", dict(status="replaying", pending=len(pending), workers=workers))
        try:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(replay_episode,j,plan["prototype"],j["job_id"] in plan["representative_jobs"]) for j in pending]
                for n, future in enumerate(as_completed(futures), 1):
                    row = dict(future.result(), binding=plan["binding"])
                    io.once(OUT / "episodes" / (row["job_id"] + ".json"), io.sealed(row))
                    if n % 24 == 0 or n == len(pending):
                        print(json.dumps(dict(replayed=n,scheduled=len(pending),no_solver=True)),flush=True)
            source.source.verify_inputs(plan)
            paths = [OUT / "episodes" / (j["job_id"] + ".json") for j in plan["jobs"]]
            rows = [io.check_seal(io.read_json(p)) for p in paths]
            fixtures = [f for r in rows for f in r["fixtures"]] + constructive_fixtures(plan["prototype"])
            require(len(fixtures) == 38, "fixed real/synthetic fixture coverage")
            io.once(OUT / "fixtures.json", io.sealed(dict(binding=plan["binding"], cases=fixtures)))
            io.once(OUT / "replay.complete.json", io.sealed(dict(binding=plan["binding"],
                files={p.stem:io.sha256_file(p) for p in paths}, fixtures_sha256=io.sha256_file(OUT / "fixtures.json"),
                decisions=sum(r["decisions"] for r in rows), replay_max_error=max(r["replay_max_error"] for r in rows))))
            atomic(OUT / "run_status.json",dict(status="replayed_pending_portable_checks", episodes=96))
            return dict(episodes=96,decisions=sum(r["decisions"] for r in rows),fixtures=len(fixtures),no_solver=True)
        except BaseException as error:
            atomic(OUT / "run_status.json",dict(status="failed_or_interrupted",error=repr(error)))
            raise


def portable_check(label):
    require(label in ("windows-numpy", "wsl-numpy", "windows-torch"), "registered portable target")
    require((os.name == "nt") == label.startswith("windows"), "incorrect platform label")
    plan = verify()
    require((OUT / "replay.complete.json").exists(), "replay required")
    cases = io.check_seal(io.read_json(OUT / "fixtures.json"))["cases"]
    errors, choices = [], 0
    if label == "windows-torch":
        import torch
        torch.set_num_threads(1)
    for case in cases:
        bundle, ids = case["bundle"], case["ids"]
        if label == "windows-torch":
            model = raw.torch_correction(bundle)
            with torch.no_grad():
                values = raw.torch_log_distribution(model,bundle,ids,case["anchor"],case["features"]).exp().numpy()
            actual = dict(zip(ids,values.tolist()))
        else:
            actual = raw.RawResidualActor(bundle).probabilities(ids,case["anchor"],case["features"])
        error = max(abs(actual[c]-case["probabilities"][c]) for c in ids)
        require(error <= 1e-12 and [select_with_draw(actual,d) for d in case["draws"]] == case["choices"], "portable probability/choice mismatch")
        if case["synthetic"]:
            require(max(actual,key=actual.get) == case["target"], "portable target mode")
        choices += len(case["draws"])
        errors.append(error)
    result = dict(binding=plan["binding"], label=label, cases=len(cases), choices=choices,
                  max_error=max(errors), numpy=np.__version__, python=sys.version.split()[0],
                  torch=torch.__version__ if label == "windows-torch" else None, no_training=True)
    io.once(OUT / (label + ".json"),io.sealed(result))
    return result


def report():
    plan = verify()
    replayed = io.check_seal(io.read_json(OUT / "replay.complete.json"))
    proofs = [io.check_seal(io.read_json(OUT / (label + ".json"))) for label in ("windows-numpy", "wsl-numpy", "windows-torch")]
    require(all(p["binding"] == plan["binding"] and p["cases"] == 38 for p in proofs), "portable proof coverage")
    result = dict(schema="lns2.sa.raw_contract_report.v1", binding=plan["binding"],
        decision="expression_contract_verified_not_trained", episodes=96, maps=6,
        decisions=replayed["decisions"], exact_parent=True, replay_max_error=replayed["replay_max_error"],
        portable=proofs, synthetic_target_witnesses=20, synthetic_large_jumps_rejected=20,
        no_training=True,no_solver=True,no_ttf=True,no_promotion=True,
        no_counterfactual_success_labels=True, future_training_requires_new_registration=True)
    io.once(OUT / "report.json",io.sealed(result))
    names = ("prototype.initial.json", "replay.complete.json", "fixtures.json", "report.json",
             "windows-numpy.json", "wsl-numpy.json", "windows-torch.json")
    io.once(OUT / "contract.complete.json",io.sealed(dict(binding=plan["binding"], files={n:io.sha256_file(OUT/n) for n in names})))
    atomic(OUT / "run_status.json",dict(status="complete",decision=result["decision"]))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","replay","portable-check","report","verify"))
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--workers",type=int,default=20)
    parser.add_argument("--label",choices=("windows-numpy","wsl-numpy","windows-torch"))
    args = parser.parse_args()
    value = prepare() if args.phase=="prepare" else replay(args.resume,args.workers) if args.phase=="replay" else portable_check(args.label) if args.phase=="portable-check" else report() if args.phase=="report" else dict(verified=True,binding=verify()["binding"])
    print(json.dumps(value),flush=True)
