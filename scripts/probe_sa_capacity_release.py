"""Bounded release diagnostics; no native PP, training, or wall-clock experiment."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_sa_certificate_generalization as g
from scripts import diagnose_sa_joint_slot_capacity as joint
from experiments.capacity_release import propose, evaluate
from experiments.diagnostic_integrity import read_bound_plan, snapshot_sources

io = g.io
OUT = ROOT / "build/sa-capacity-release-v1"


def prepare():
    io.require(not (OUT/"plan.json").exists(), "preserve registered plan")
    parent = read_bound_plan(ROOT, joint.OUT, evidence_only=True)
    report = io.read(joint.OUT/"report.json")
    io.require(report["binding"] == parent["binding"], "parent binding mismatch")
    lookup = {j["id"]: j for j in parent["jobs"]}
    jobs, proposals = [], []
    names = ["scripts/probe_sa_capacity_release.py", "experiments/capacity_release.py",
             "experiments/temporal_slot_scan.py", "experiments/joint_slot_capacity.py",
             "experiments/goal_slot_certificate.py", "experiments/local_path_search.py",
             "experiments/diagnostic_integrity.py", "experiments/repair_collection.py",
             "scripts/validate_sa_certificate_generalization.py",
             "docs/SA_CAPACITY_RELEASE_PROTOCOL_ZH.md",
             (joint.OUT/"report.json").relative_to(ROOT).as_posix(),
             (joint.OUT/"plan.json").relative_to(ROOT).as_posix()]
    for row in sorted(report["rows"], key=lambda r: r["id"]):
        path = joint.OUT/"rows"/(row["id"]+".json")
        io.require(io.sha256_file(path) == report["files"][row["id"]] and io.read(path) == row, "parent row changed")
        if row["joint"]["status"] != "proved":
            continue
        source = lookup[row["id"]]["source"]
        case = g.load_case(source)
        proof = row["joint"]
        proposal = propose(case["state"], case["selected"], proof["witness"]["agents"], proof["time"], case["id"])
        proposals.append(dict(case_id=case["id"], **proposal))
        names.append(source["case_path"])
        for candidate in proposal["candidates"]:
            jid = case["id"]+"-"+candidate["id"]
            jobs.append(dict(id=jid, job_id=jid, source=source, proof=proof, candidate=candidate))
    plan = dict(schema="lns2.capacity_release.v1", inputs={n: io.sha256_file(ROOT/n) for n in names},
                jobs=jobs, proposals=proposals, workers=20, timeout=180, no_solver=True)
    plan["binding"] = io.semantic_fingerprint(plan)
    snapshot_sources(ROOT, plan["inputs"], OUT/"registered_sources")
    io.write(OUT/"plan.json", plan)
    print(dict(cases=len(proposals), jobs=len(jobs), workers=20, solver_calls=0), flush=True)


def worker(job):
    case = g.load_case(job["source"])
    return dict(id=job["id"], status="ok", case_id=case["id"],
                **evaluate(case, job["proof"], job["candidate"]))


def collect(resume):
    plan = read_bound_plan(ROOT, OUT)
    jobs = {j["id"]: j for j in plan["jobs"]}
    with g.q._CollectionRunLock(OUT, plan["binding"], "release"):
        pending = []
        for name, job in jobs.items():
            path = OUT/"rows"/(name+".json")
            if path.exists():
                io.require(resume, "resume required")
                receipt = io.read(OUT/"receipts"/(name+".json"))
                io.require(receipt == dict(binding=plan["binding"], job=io.semantic_fingerprint(job), sha=io.sha256_file(path)), "receipt mismatch")
                io.require(io.read(path)["status"] == "ok", "inspect failed job before resume")
            else:
                pending.append(job)
        def save(row):
            row["binding"] = plan["binding"]
            path = OUT/"rows"/(row["id"]+".json")
            io.write(path, row)
            io.write(OUT/"receipts"/(row["id"]+".json"), dict(binding=plan["binding"], job=io.semantic_fingerprint(jobs[row["id"]]), sha=io.sha256_file(path)))
            print(row["id"], row["status"], flush=True)
        def failed(job, status, error):
            return dict(id=job["id"], status=status, error=error)
        rows = g.q._run_jobs(worker, pending, workers=20, phase="release", output_root=OUT/"progress",
                             run_fingerprint=plan["binding"], timeout_seconds=180,
                             failure_result=failed, on_result=save, stop_on_failure=True)
        io.require(len(rows) == len(pending) and all(r["status"] == "ok" for r in rows), "incomplete release phase")
        io.write(OUT/"manifest.json", dict(binding=plan["binding"], files={n: io.sha256_file(OUT/"rows"/(n+".json")) for n in jobs}))


def analyze():
    plan = read_bound_plan(ROOT, OUT, evidence_only=True)
    manifest = io.read(OUT/"manifest.json")
    io.require(manifest["binding"] == plan["binding"] and set(manifest["files"]) == {j["id"] for j in plan["jobs"]}, "coverage mismatch")
    grouped = defaultdict(list)
    for job in plan["jobs"]:
        path = OUT/"rows"/(job["id"]+".json")
        io.require(io.sha256_file(path) == manifest["files"][job["id"]], "result changed")
        row = io.read(path)
        io.require(row["id"] == job["id"] and row["binding"] == plan["binding"] and row["status"] == "ok" and row["candidate"] == job["candidate"], "result identity")
        grouped[(row["case_id"], row["candidate"]["family"])].append(row)
    table = [dict(case_id=case, family=family, count=len(rows),
                  relaxed_clear=sum(r["original_certificate_cleared"] for r in rows),
                  augmented_not_proved=sum(r["augmented"]["status"] == "not_proved" for r in rows),
                  augmented_proved=sum(r["augmented"]["status"] == "proved" for r in rows),
                  augmented_unknown=sum(r["augmented"]["status"] == "resource_unknown" for r in rows))
             for (case, family), rows in sorted(grouped.items())]
    io.write(OUT/"report.json", dict(binding=plan["binding"], tables=table, jobs=len(plan["jobs"]),
             manifest_sha=io.sha256_file(OUT/"manifest.json"), no_ttf=True, controller_promoted=False))
    print(table, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "collect": collect(args.resume)
    else: analyze()
