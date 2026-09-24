"""Read-only frozen actor expressivity audit; never fits or calls a MAPF solver."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.sa_onpolicy_actor import NumpyActor, select_with_draw, vectorize
from scripts import audit_sa_work_clock as source
from scripts import run_sa_onpolicy as io
from scripts.audit_sa_history_information import atomic

OUT = ROOT / "build/sa-policy-expressivity-audit-v1"
SCHEMA = "lns2.sa_policy_expressivity.v1"
require = io.require


def family_bounds(count):
    require(type(count) is int and count >= 1, "positive integer candidate count")
    if count == 1:
        return dict(candidates=1, anchor_probability_min=1., challenger_probability_max=None,
                    required_residual_gap=None, minimum_anchor_log_odds=None,
                    challenger_can_outrank=False, nontrivial_restriction=False)
    # q(anchor)/q(other) = 9K+1; each bounded residual is in [-2, 2].
    ratio, gain = 9 * count + 1, math.exp(4.)
    margin = math.log(ratio) - 4.
    return dict(candidates=count, required_residual_gap=math.log(ratio),
                minimum_anchor_log_odds=margin,
                anchor_probability_min=ratio / (ratio + (count - 1) * gain),
                challenger_probability_max=gain / (gain + 10 * count - 1),
                challenger_can_outrank=margin < 0,
                nontrivial_restriction=margin >= 0)


def measure_event(actor, event):
    ids, anchor = event["candidate_ids"], event["anchor_id"]
    require(ids == sorted(set(ids)) and anchor in ids, "noncanonical candidates")
    require(event["policy_sha256"] == actor.sha, "wrong frozen policy")
    require(set(event["probabilities"]) == set(ids), "probability coverage")
    p = actor.probabilities(ids, anchor, event["features"])
    error = max(abs(p[c] - event["probabilities"][c]) for c in ids)
    require(error <= 1e-12, "frozen probability replay mismatch")
    require(select_with_draw(p, event["selection_draw"]) == event["selected_id"], "selection mismatch")
    x = vectorize(event["features"], actor.bundle["feature_names"])
    hidden = np.tanh(((x - actor.mean) / actor.scale) @ actor.w1.T + actor.b1)
    residual = (2. * np.tanh(hidden @ actor.w2.T + actor.b2))[:, 0]
    i = ids.index(anchor)
    other = [j for j in range(len(ids)) if j != i]
    bounds = family_bounds(len(ids))
    gap = max(float(residual[j] - residual[i]) for j in other) if other else None
    maximum = max(p[c] for c in ids if c != anchor) if other else 0.
    if other:
        log_odds = math.log(p[anchor] / maximum)
        require(abs(log_odds - (bounds["required_residual_gap"] - gap)) <= 1e-12, "log-odds identity")
        require(log_odds >= bounds["minimum_anchor_log_odds"] - 1e-12, "family bound violated")
        require(maximum <= bounds["challenger_probability_max"] + 1e-12, "challenger bound violated")
    else:
        log_odds = None
    prior = {c: .1 / len(ids) + (.9 if c == anchor else 0.) for c in ids}
    return dict(decision=event["decision"], candidates=len(ids),
                locked=bool(bounds["nontrivial_restriction"]),
                anchor_strict_mode=p[anchor] > maximum,
                selected_nonanchor=event["selected_id"] != anchor,
                anchor_probability=p[anchor], maximum_challenger_probability=maximum,
                anchor_log_odds=log_odds, challenger_residual_gap=gap,
                residual_span=float(np.ptp(residual)),
                prior_total_variation=.5 * math.fsum(abs(p[c] - prior[c]) for c in ids),
                same_draw_changed=select_with_draw(prior, event["selection_draw"]) != event["selected_id"],
                replay_error=error)


def checked_job(job):
    require(job["split"] == "train" and job["arm"] in source.ARMS, "only registered Train actors")
    row = job["result"]
    require(row["status"] == "ok" and row["stop"] in ("feasible", "node_budget"), "censored source")
    return row


def episode(job, bundle):
    row = checked_job(job)
    folder = ROOT / job["folder"]
    for name, digest in row["files"].items():
        require(io.sha256_file(folder / name) == digest, "source episode changed")
    actor = NumpyActor(bundle)
    require(actor.sha == row["policy_sha256"], "job policy mismatch")
    steps = []
    for index, event in enumerate(io.trace_read(folder)):
        require(event["decision"] == index, "decision discontinuity")
        require(sorted(c["candidate_id"] for c in event["pool"]) == event["candidate_ids"], "pool alignment")
        steps.append(measure_event(actor, event))
    require(len(steps) == row["decisions"], "incomplete trajectory")
    return dict(job_id=job["job_id"], arm=job["arm"], map_id=job["map_id"],
                pair_id=job["pair_id"], replica=job["replica"], success=row["success"],
                status="ok", steps=steps)


def describe(values):
    values = [v for v in values if v is not None]
    return dict(n=len(values), minimum=min(values) if values else None,
                mean=statistics.mean(values) if values else None,
                maximum=max(values) if values else None)


def summarize(rows):
    steps = [s for row in rows for s in row["steps"]]
    return dict(episodes=len(rows), source_successes=sum(r["success"] for r in rows),
                maps=len({r["map_id"] for r in rows}), steps=len(steps),
                counts={k:sum(s[k] for s in steps) for k in
                        ("locked", "anchor_strict_mode", "selected_nonanchor", "same_draw_changed")},
                candidate_counts=dict(sorted(Counter(str(s["candidates"]) for s in steps).items())),
                metrics={k:describe([s[k] for s in steps]) for k in
                         ("anchor_probability", "maximum_challenger_probability", "anchor_log_odds",
                          "challenger_residual_gap", "residual_span", "prior_total_variation", "replay_error")})


def report_for(plan, rows):
    require(len(rows) == len(plan["jobs"]) == 192, "missing episodes")
    require([r["job_id"] for r in rows] == [j["job_id"] for j in plan["jobs"]], "job order")
    counts = sorted({s["candidates"] for r in rows for s in r["steps"]})
    return dict(schema=SCHEMA, binding=plan["binding"], no_solver=True, no_training=True,
                no_ttf=True, no_causal_or_promotion_claim=True,
                decision="document_policy_class_restriction_not_failure_cause",
                independent_maps=6, episodes=192, family_bounds=[family_bounds(k) for k in counts],
                by_arm={a:summarize([r for r in rows if r["arm"] == a]) for a in source.ARMS},
                by_map={m:{a:summarize([r for r in rows if r["arm"] == a and r["map_id"] == m])
                    for a in source.ARMS} for m in sorted({r["map_id"] for r in rows})})


def prepare():
    require(not OUT.exists(), "output already exists; use verify/resume")
    prior = source.verify()
    require((source.OUT / "audit.complete.json").is_file(), "completed source audit required")
    original = io.check_seal(io.read_json(source.SOURCE / "registration.json"))
    models = original["config"]["models"]
    require(set(models) == set(source.ARMS), "source model arms")
    for job in prior["jobs"]:
        checked_job(job)
    inputs = dict(prior["inputs"])
    for path in [source.OUT / n for n in ("audit_registration.json", "report.json", "audit.complete.json")] + [
        ROOT / n for n in ("scripts/audit_sa_policy_expressivity.py", "scripts/audit_sa_work_clock.py",
        "tests/evaluation/test_sa_policy_expressivity.py", "docs/SA_POLICY_EXPRESSIVITY_PROTOCOL_ZH.md",
        "experiments/sa_onpolicy_actor.py", "experiments/sa_onpolicy_contract.py")]:
        inputs[path.relative_to(ROOT).as_posix()] = io.sha256_file(path)
    plan = dict(schema=SCHEMA, source_binding=prior["binding"], inputs=inputs, jobs=prior["jobs"],
                models=models, no_solver=True, no_training=True, no_ttf=True,
                commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    plan["binding"] = io.json_fingerprint(plan)
    io.once(OUT / "audit_registration.json", io.sealed(plan))
    return dict(episodes=192, maps=6, no_solver=True, binding=plan["binding"])


def verify():
    plan = io.check_seal(io.read_json(OUT / "audit_registration.json"))
    require(plan["schema"] == SCHEMA and plan["binding"] == io.json_fingerprint(
        {k:v for k,v in plan.items() if k not in ("binding", "integrity")}), "audit registration")
    source.verify_inputs(plan)
    if (OUT / "audit.complete.json").exists():
        done = io.check_seal(io.read_json(OUT / "audit.complete.json"))
        require(done["binding"] == plan["binding"] and len(done["files"]) == 192, "completion binding")
        rows = []
        for job in plan["jobs"]:
            path = OUT / "episodes" / (job["job_id"] + ".json")
            require(io.sha256_file(path) == done["files"][job["job_id"]], "audit episode changed")
            row = io.check_seal(io.read_json(path))
            require(row["binding"] == plan["binding"] and row["status"] == "ok", "audit episode binding")
            rows.append(row)
        require(done["report_sha256"] == io.sha256_file(OUT / "report.json"), "report changed")
        actual = io.check_seal(io.read_json(OUT / "report.json"))
        require(actual == io.sealed(report_for(plan, rows)), "report recomputation")
    return plan


def analyze(resume=False, workers=20):
    require(type(workers) is int and 1 <= workers <= 20, "workers outside read-only budget")
    plan = verify()
    if (OUT / "audit.complete.json").exists():
        require(resume, "completed audit; use verify or resume")
        return dict(verified=True, complete=True, no_solver=True)
    models = {a:io.read_json(ROOT / m["path"]) for a,m in plan["models"].items()}
    with source.q._CollectionRunLock(OUT, plan["binding"], "readonly-expressivity"):
        pending = []
        for job in plan["jobs"]:
            path = OUT / "episodes" / (job["job_id"] + ".json")
            if path.exists():
                row = io.check_seal(io.read_json(path))
                require(resume and row["binding"] == plan["binding"] and row["job_id"] == job["job_id"]
                        and row["status"] == "ok", "inspect prior output before resume")
            else:
                pending.append(job)
        atomic(OUT / "run_status.json", dict(status="running", pending=len(pending), workers=workers))
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(episode, job, models[job["arm"]]) for job in pending]
                for n, future in enumerate(as_completed(futures), 1):
                    row = dict(future.result(), binding=plan["binding"])
                    io.once(OUT / "episodes" / (row["job_id"] + ".json"), io.sealed(row))
                    if n % 24 == 0 or n == len(pending):
                        print(json.dumps(dict(completed=n, scheduled=len(pending), no_solver=True)), flush=True)
            source.verify_inputs(plan)
            paths = [OUT / "episodes" / (j["job_id"] + ".json") for j in plan["jobs"]]
            rows = [io.check_seal(io.read_json(p)) for p in paths]
            io.once(OUT / "report.json", io.sealed(report_for(plan, rows)))
            io.once(OUT / "audit.complete.json", io.sealed(dict(binding=plan["binding"],
                report_sha256=io.sha256_file(OUT / "report.json"),
                files={p.stem:io.sha256_file(p) for p in paths})))
            atomic(OUT / "run_status.json", dict(status="complete", episodes=len(rows), no_solver=True))
            return dict(complete=True, episodes=len(rows), no_solver=True)
        except BaseException as error:
            atomic(OUT / "run_status.json", dict(status="failed_or_interrupted", error=repr(error)))
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "analyze", "verify"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    result = prepare() if args.phase == "prepare" else analyze(args.resume, args.workers) if args.phase == "analyze" else dict(verified=True, binding=verify()["binding"])
    print(json.dumps(result), flush=True)
