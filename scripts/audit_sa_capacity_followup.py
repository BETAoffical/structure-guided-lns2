"""Audit saved trials and capacity witnesses without launching a solver."""
from pathlib import Path
from itertools import combinations
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import validate_sa_certificate_generalization as g
from scripts import diagnose_sa_joint_slot_capacity as joint
from experiments.goal_slot_certificate import goal_slot
from experiments.diagnostic_integrity import read_bound_plan

io = g.io


def smallest_deficiency(members, domains):
    """Minimum witness within the reported set, not over all possible agents."""
    for size in range(1, len(members) + 1):
        for indices in combinations(range(len(members)), size):
            cells = set().union(*(domains[i] for i in indices))
            if len(cells) < size:
                return dict(agents=[members[i] for i in indices], cells=sorted(cells), deficit=size-len(cells))
    return None


def validate_scalars(row, case):
    metrics = row["metrics"]
    seed = int(io.semantic_fingerprint([case["id"], 20260915, row["trial"]])[:7], 16)
    io.require(row["case_id"] == case["id"], "case identity mismatch")
    io.require(row["before"] == case["state"]["num_of_colliding_pairs"], "initial count mismatch")
    io.require(row["after"] == row["final_state"]["num_of_colliding_pairs"], "final count mismatch")
    io.require(row["generated"] == row["final_state"]["low_level"]["generated"], "node count mismatch")
    io.require(row["censored"] == (metrics["pp_failure_reason"] == "time_limit"), "censor mismatch")
    io.require(metrics["requested_random_seed"] == seed, "random seed mismatch")
    io.require(metrics["requested_pp_time_limit_seconds"] == 5., "repair budget mismatch")


def verify_hall(case, proof):
    graph, agents = g.ref.validate_state(case["state"])
    selected = set(case["selected"])
    witness = proof["witness"]
    members = witness["agents"]
    io.require(len(set(members)) == len(members) and set(members) <= selected, "invalid witness agents")
    fixed = {a: v["path"] for a, v in agents.items() if a not in selected}
    # Recompute the critical tick using the earlier single-tick implementation.
    domains = [set(goal_slot(graph, agents[a]["start"], agents[a]["goal"], fixed,
                            proof["time"]).get("viable_at_tick", [])) for a in members]
    cells = set().union(*domains)
    io.require(sorted(cells) == witness["cells"], "independent capacity cells mismatch")
    io.require(len(members) - len(cells) == witness["deficit"] > 0, "invalid capacity deficit")
    return smallest_deficiency(members, domains)


def main():
    plan = read_bound_plan(ROOT, g.OUT, evidence_only=True)
    sources = {}
    baseline = {}
    trial_count = paired = 0
    for phase in ("add", "replace"):
        rows = g.read_stage(phase)
        jobs = {j["id"]: j for j in g.schedule(plan, phase)}
        io.require(set(jobs) == {r["id"] for r in rows}, "trial coverage mismatch")
        for row in rows:
            job = jobs[row["id"]]
            case = g.load_case(job["source"])
            io.require(row["candidate"] == job["candidate"] and row["trial"] == job["trial"], "candidate mismatch")
            validate_scalars(row, case)
            g.ref.validate_state(row["final_state"])
            seed = int(io.semantic_fingerprint([case["id"], 20260915, row["trial"]])[:7], 16)
            g.validate_transition(case["state"], row["final_state"], row["metrics"], row["candidate"]["members"],
                                  "annealed", case["temperature"], g.random.Random(seed).random())
            if row["candidate"]["id"] == "baseline":
                key = (row["case_id"], row["trial"])
                signature = [io.repair_structure_fingerprint(row["final_state"]), row["generated"],
                             row["metrics"]["repair_order"], row["after"]]
                if phase == "add":
                    baseline[key] = signature
                else:
                    io.require(baseline[key] == signature, "cross-stage baseline changed")
                    paired += 1
            trial_count += 1
        path = g.OUT / phase / "manifest.json"
        sources[path.relative_to(ROOT).as_posix()] = io.sha256_file(path)

    old = ROOT / "build/sa-certificate-generalization-v1"
    old_manifest = io.read(old / "scan/manifest.json")
    current = {r["id"]: r for r in g.read_stage("scan")}
    io.require(set(current) == set(old_manifest["files"]), "scan schedule changed")
    for name, digest in old_manifest["files"].items():
        path = old / "scan/rows" / (name + ".json")
        io.require(io.sha256_file(path) == digest, "old scan row changed")
        before, after = io.read(path), current[name]
        if before.get("case_path"):
            a, b = g.load_case(before), g.load_case(after)
            for field in ("state", "selected", "temperature", "origin", "certificate", "map_hash"):
                io.require(a[field] == b[field], "scan scientific fields changed: " + field)
        else:
            io.require(before["reason"] == after["reason"], "eligibility changed")
    sources[(old / "scan/manifest.json").relative_to(ROOT).as_posix()] = io.sha256_file(old / "scan/manifest.json")

    jp = read_bound_plan(ROOT, joint.OUT, evidence_only=True)
    report = io.read(joint.OUT / "report.json")
    io.require(report["binding"] == jp["binding"], "capacity binding mismatch")
    jobs = {j["id"]: j for j in jp["jobs"]}
    io.require(set(jobs) == set(report["files"]) == {r["id"] for r in report["rows"]}, "capacity coverage")
    verified = 0
    minimum_witnesses = {}
    for row in report["rows"]:
        path = joint.OUT / "rows" / (row["id"] + ".json")
        io.require(io.sha256_file(path) == report["files"][row["id"]] and io.read(path) == row, "capacity row changed")
        io.require(row["status"] == "ok" and row["binding"] == jp["binding"], "capacity error")
        if row["joint"]["status"] == "proved":
            minimum_witnesses[row["id"]] = verify_hall(g.load_case(jobs[row["id"]]["source"]), row["joint"])
            verified += 1
    sources[(joint.OUT / "report.json").relative_to(ROOT).as_posix()] = io.sha256_file(joint.OUT / "report.json")
    sources[Path(__file__).relative_to(ROOT).as_posix()] = io.sha256_file(Path(__file__))
    result = dict(schema="lns2.sa_capacity_followup_audit.v1", trials=trial_count,
                  identical_baselines=paired, identical_scan_episodes=len(current),
                  capacity_rows=len(jobs), independently_verified_proofs=verified,
                  smallest_deficiency_within_reported_set=minimum_witnesses,
                  errors=0, no_solver=True, no_ttf=True, sources=sources)
    io.write(joint.OUT / "independent_audit.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
