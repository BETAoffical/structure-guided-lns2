"""Capacity-label discrimination with frozen PP and matched candidate sizes."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_sa_capacity_release as release
from scripts import validate_sa_certificate_generalization as g
from scripts.audit_sa_capacity_followup import validate_scalars
from experiments.diagnostic_integrity import read_bound_plan, snapshot_sources

io = g.io
OUT = ROOT / "build/sa-capacity-native-v1"


def condition(row):
    if row["augmented"]["status"] == "not_proved":
        return "pass"
    if not row["original_certificate_cleared"] or row["augmented"]["status"] == "proved":
        return "fail"
    return "unknown"


def select_candidates(case, rows):
    requests = [("baseline", "baseline", None), ("raw_single", "directed_single", None)]
    for size in ("single", "pair"):
        requests += [("pass_"+size, "directed_"+size, "pass"),
                     ("fail_"+size, "directed_"+size, "fail"),
                     ("random_"+size, "random_"+size, None)]
    unique = {}
    missing = []
    for role, family, status in requests:
        choices = [r for r in rows if r["candidate"]["family"] == family and
                   (status is None or condition(r) == status)]
        if not choices:
            missing.append(role)
            continue
        row = choices[0]
        added = row["candidate"]["members"]
        members = tuple(sorted(set(case["selected"]) | set(added)))
        if members not in unique:
            unique[members] = dict(id=io.semantic_fingerprint(members)[:16], members=list(members),
                                   added=added, roles=[], condition=condition(row), source=row["id"])
        unique[members]["roles"].append(role)
    return list(unique.values()), missing


def prepare():
    io.require(not (OUT/"plan.json").exists(), "preserve plan")
    io.require(io.sha256_file(ROOT/io.NATIVE) == io.NATIVE_SHA, "frozen native changed")
    parent = read_bound_plan(ROOT, release.OUT)
    manifest = io.read(release.OUT/"manifest.json")
    io.require(manifest["binding"] == parent["binding"] and set(manifest["files"]) == {j["id"] for j in parent["jobs"]}, "parent coverage")
    grouped, sources = defaultdict(list), {}
    for job in parent["jobs"]:
        path = release.OUT/"rows"/(job["id"]+".json")
        io.require(io.sha256_file(path) == manifest["files"][job["id"]], "parent row changed")
        row = io.read(path)
        case_id = row["case_id"]
        io.require(row["id"] == job["id"] and row["binding"] == parent["binding"] and row["status"] == "ok" and row["candidate"] == job["candidate"], "parent identity")
        grouped[case_id].append(row)
        sources[case_id] = job["source"]
    entries, jobs = [], []
    inputs = dict(parent["inputs"])
    for name, digest in manifest["files"].items():
        inputs[(release.OUT/"rows"/(name+".json")).relative_to(ROOT).as_posix()] = digest
    names = [io.NATIVE, "artifacts/initlns-closed-loop-controller-v2/main__realized_dynamic.json",
             "scripts/probe_sa_capacity_native.py", "scripts/audit_sa_capacity_followup.py",
             "scripts/diagnose_pbs_repair.py", "experiments/nonmonotonic_repair.py",
             "docs/SA_CAPACITY_NATIVE_PROTOCOL_ZH.md", "tests/evaluation/test_sa_capacity_native.py",
             (release.OUT/"manifest.json").relative_to(ROOT).as_posix()]
    for case_id, rows in grouped.items():
        case = g.load_case(sources[case_id])
        candidates, missing = select_candidates(case, rows)
        entries.append(dict(id=case_id, source=sources[case_id], candidates=candidates, unavailable_roles=missing))
        inputs.update(case["file_hashes"])
        for c in candidates:
            for trial in range(4):
                jid = case_id+"-"+c["id"]+"-"+str(trial)
                jobs.append(dict(id=jid, job_id=jid, source=sources[case_id], candidate=c, trial=trial))
    for name in names:
        inputs[name] = io.sha256_file(ROOT/name)
    io.require(len({j["id"] for j in jobs}) == len(jobs), "duplicate native job")
    plan = dict(schema="lns2.sa_capacity_native.v1", inputs=inputs, cases=entries, jobs=jobs,
                workers=20, trial_seeds="existing case/20260915/trial hash", pp_seconds=5., fuse=180.,
                no_ttf=True, no_continuation=True)
    plan["binding"] = io.semantic_fingerprint(plan)
    snapshot_sources(ROOT, inputs, OUT/"registered_sources")
    io.write(OUT/"plan.json", plan)
    print(dict(states=len(entries), actions=sum(len(c["candidates"]) for c in entries), jobs=len(jobs)), flush=True)


def collect(resume):
    plan = read_bound_plan(ROOT, OUT)
    jobs = {j["id"]: j for j in plan["jobs"]}
    with g.q._CollectionRunLock(OUT, plan["binding"], "capacity-native"):
        pending = []
        for jid, job in jobs.items():
            path = OUT/"rows"/(jid+".json")
            if path.exists():
                io.require(resume, "resume required")
                io.require(io.read(OUT/"receipts"/(jid+".json")) == dict(binding=plan["binding"], job=io.semantic_fingerprint(job), sha=io.sha256_file(path)), "receipt mismatch")
                io.require(io.read(path)["status"] == "ok", "inspect error before resume")
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
        rows = g.q._run_jobs(g.probe, pending, workers=20, phase="capacity-native", output_root=OUT/"progress",
                             run_fingerprint=plan["binding"], timeout_seconds=180., failure_result=failed,
                             on_result=save, stop_on_failure=True)
        io.require(len(rows) == len(pending) and all(r["status"] == "ok" for r in rows), "incomplete native phase")
        io.write(OUT/"manifest.json", dict(binding=plan["binding"], files={n: io.sha256_file(OUT/"rows"/(n+".json")) for n in jobs}))


def local_gate(groups):
    winners = []
    for case_id in sorted({key[0] for key in groups}):
        for size in ("single", "pair"):
            keys = [(case_id, r) for r in ("baseline", "pass_"+size, "fail_"+size, "random_"+size)]
            if not all(k in groups for k in keys):
                continue
            rows = [groups[k] for k in keys]
            if any(len(x) != 4 or any(r["censored"] for r in x) for x in rows):
                continue
            means = [sum(r["after"] for r in x)/4 for x in rows]
            baseline = {r["trial"]: r["after"] for r in rows[0]}
            wins = sum(r["after"] < baseline[r["trial"]] for r in rows[1])
            if wins >= 2 and all(means[1] < means[i] for i in (0, 2, 3)):
                winners.append(dict(case_id=case_id, size=size, means=means, wins=wins))
    return dict(qualifying=winners, permit_continuation_diagnostic=len({r["case_id"] for r in winners}) >= 2,
                controller_promoted=False)


def analyze():
    plan = read_bound_plan(ROOT, OUT, evidence_only=True)
    manifest = io.read(OUT/"manifest.json")
    io.require(manifest["binding"] == plan["binding"] and set(manifest["files"]) == {j["id"] for j in plan["jobs"]}, "native coverage")
    groups = defaultdict(list)
    for job in plan["jobs"]:
        path = OUT/"rows"/(job["id"]+".json")
        io.require(io.sha256_file(path) == manifest["files"][job["id"]], "native row changed")
        row = io.read(path)
        io.require(row["id"] == job["id"] and row["candidate"] == job["candidate"] and row["trial"] == job["trial"] and row["binding"] == plan["binding"] and row["status"] == "ok", "native identity")
        case = g.load_case(job["source"])
        validate_scalars(row, case)
        g.ref.validate_state(row["final_state"])
        seed = row["metrics"]["requested_random_seed"]
        g.validate_transition(case["state"], row["final_state"], row["metrics"], row["candidate"]["members"],
                              "annealed", case["temperature"], g.random.Random(seed).random())
        io.require(row["final_state"]["feasible"] == (row["after"] == 0), "feasible mismatch")
        for role in row["candidate"]["roles"]:
            groups[(row["case_id"], role)].append(row)
    tables = []
    for (case_id, role), rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["trial"])
        io.require([r["trial"] for r in rows] == list(range(4)), "trial coverage")
        tables.append(dict(case_id=case_id, role=role, added=rows[0]["candidate"]["added"],
                           after=[r["after"] for r in rows], generated_mean=sum(r["generated"] for r in rows)/4,
                           censored=sum(r["censored"] for r in rows), feasible=sum(r["after"] == 0 for r in rows),
                           rolled_back=sum(r["metrics"]["pp_rolled_back"] for r in rows)))
    gate = local_gate(groups)
    io.write(OUT/"report.json", dict(binding=plan["binding"], jobs=len(plan["jobs"]), tables=tables, gate=gate,
             no_ttf=True, manifest_sha=io.sha256_file(OUT/"manifest.json")))
    print(gate, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "analyze"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.stage == "prepare": prepare()
    elif args.stage == "collect": collect(args.resume)
    else: analyze()
