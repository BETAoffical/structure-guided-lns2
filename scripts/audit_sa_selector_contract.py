"""Audit retained training/closed-loop contracts without fits or solver calls."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, read_json, sha256_file, json_fingerprint
from experiments.sa_paired_completion import require
from experiments.sa_selector_contract_audit import feature_support, contrast_diagnostic, label_curves
from scripts.run_sa_paired_closed_loop import once, sealed, check_seal

SOURCE = "build/sa-linear-closed-loop-v1a"
OFFLINE = "build/sa-low-complexity-ranker-v1/report.json"
SOURCE_SHA = "23fce23f4c2c77f8ad80e1d47919280b6b79c87b8433324b0435d35a718c0713"
OFFLINE_SHA = "bfe153b6281f8a9afa77f62d92501697feaca86551e7c0560f520dcafc5cbca6"
OUTPUT = "build/sa-selector-contract-audit-v1"


def episode_job(job):
    episode, support, digest = job
    path = ROOT/SOURCE/"episodes"/episode["job_id"]/"trace.jsonl"
    require(sha256_file(path) == digest, "trace changed")
    with path.open(encoding="utf8") as stream:
        events = (json.loads(line) for line in stream)
        return audit_events(events, episode, support, path, digest)


def audit_events(events, episode, support, path, digest):
    changed = violating = novel = count = 0
    columns = Counter()
    abs_sum = 0.
    first = None
    for e in events:
        require(e["decision"] == count, "trace discontinuity")
        d = contrast_diagnostic(e["features"], e["subset"], e["anchor_id"], e["ranking"]["selected"], support)
        count += 1
        changed += d["changed"]
        violating += bool(d["contrast_outside"])
        novel += bool(d["previously_unvarying"])
        abs_sum += d["absolute_outside_fraction"]
        columns.update(d["contrast_outside"])
        if first is None:
            first = dict(before=e["before"], pool_hash=json_fingerprint(e["pool"]),
                         subset=e["subset"], features_hash=json_fingerprint(e["features"]),
                         selected=e["ranking"]["selected"], **d)
    require(count == episode["decisions"], "missing trace events")
    require(sha256_file(path) == digest, "trace changed during audit")
    return dict(job_id=episode["job_id"], pair_id=episode["pair_id"], arm=episode["arm"],
                decisions=count, changed=changed, contrast_outside_decisions=violating,
                novel_contrast_decisions=novel, columns=dict(columns), absolute_fraction_sum=abs_sum, first=first)


def run(workers):
    require(1 <= workers <= 20, "workers must be 1..20")
    out = ROOT/OUTPUT
    require(not out.exists(), "audit exists; verify it instead of overwriting")
    from scripts.run_sa_linear_closed_loop import verify, model_receipt
    plan, source = verify()
    require(source == ROOT/SOURCE, "source identity")
    model_receipt(plan, source)
    require(sha256_file(source/"report.json") == SOURCE_SHA and sha256_file(ROOT/OFFLINE) == OFFLINE_SHA,
            "registered reports changed")
    report = check_seal(read_json(source/"report.json"))
    old = check_seal(read_json(ROOT/OFFLINE))
    require(report["binding"] == plan["binding"], "report binding")
    inputs = dict(plan["inputs"])
    for p, h in report["files"].items():
        path = contained_file(source, p, field="retained output")
        require(sha256_file(path) == h, "changed output: " + p)
        inputs[path.relative_to(ROOT).as_posix()] = h
    for p in [SOURCE+"/report.json", SOURCE+"/model/linear.json", OFFLINE,
              "experiments/sa_selector_contract_audit.py", "scripts/audit_sa_selector_contract.py",
              "tests/evaluation/test_sa_selector_contract_audit.py"]:
        inputs[p] = sha256_file(ROOT/p)
    data = read_json(ROOT/plan["config"]["training_index"])
    support = feature_support(data, read_json(source/"model/linear.json"))
    prediction = {r["state_id"]: {k: v for k, v in r["selected"].items() if k in ("linear_difference", "gbdt")}
                  for r in old["rows"]}
    curves = label_curves(data, prediction)
    for method in curves["map_macro"]:
        require(abs(curves["map_macro"][method]["32"]-old["means"][method]) < 1e-12, "H32 did not reproduce")
    jobs = [(e, support, report["files"]["episodes/"+e["job_id"]+"/trace.jsonl"]) for e in report["episodes"]]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        rows = list(pool.map(episode_job, jobs))
    for pair in sorted({r["pair_id"] for r in rows}):
        group = [r for r in rows if r["pair_id"] == pair]
        require(len(group) == 3 and {r["arm"] for r in group} == {"frozen", "linear", "gbdt"}, "pair coverage")
        require(all(r["first"] is not None for r in group), "unexpected zero-conflict task")
        signatures = [json_fingerprint({k: r["first"][k] for k in ("before", "pool_hash", "subset", "features_hash")}) for r in group]
        require(len(set(signatures)) == 1, "first-decision input mismatch")
    summary = {}
    for arm in ("frozen", "linear", "gbdt"):
        rs = [r for r in rows if r["arm"] == arm]
        counts = Counter()
        for r in rs:
            counts.update(r["columns"])
        summary[arm] = {k: sum(r[k] for r in rs) for k in
                        ("decisions", "changed", "contrast_outside_decisions", "novel_contrast_decisions")}
        summary[arm].update(columns=dict(counts.most_common()),
                            absolute_outside_fraction=sum(r["absolute_fraction_sum"] for r in rs)/sum(r["decisions"] for r in rs),
                            first_changed=sum(r["first"]["changed"] for r in rs),
                            first_contrast_outside=sum(bool(r["first"]["contrast_outside"]) for r in rs))
    for p, h in inputs.items():
        require(sha256_file(contained_file(ROOT, p, field="input")) == h, "input changed during audit")
    value = dict(schema="lns2.sa.selector_contract_audit.v1", inputs=inputs,
                 source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 source_result_sha256=SOURCE_SHA, feature_support=support, label_curves=curves,
                 online_summary=summary, episodes=rows, initial_pair_checks=32,
                 new_solver_calls=0, model_fits=0, controller_changed=False,
                 interpretation="posthoc_descriptive_not_causal_not_a_new_gate")
    once(out/"report.json", sealed(value))
    print(json.dumps(dict(output=OUTPUT, states=len(data["states"]), episodes=len(rows),
                          decisions=sum(r["decisions"] for r in rows), report_sha256=sha256_file(out/"report.json"))))


def verify_output():
    r = check_seal(read_json(ROOT/OUTPUT/"report.json"))
    for p, h in r["inputs"].items():
        require(sha256_file(contained_file(ROOT, p, field="input")) == h, "audit input changed: " + p)
    print(json.dumps(dict(verified=True, inputs=len(r["inputs"]), sha256=sha256_file(ROOT/OUTPUT/"report.json"))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("run", "verify"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    run(args.workers) if args.phase == "run" else verify_output()
