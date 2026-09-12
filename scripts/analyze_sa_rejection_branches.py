"""Read-only, post-hoc state reconvergence analysis of frozen SA branches."""

from collections import Counter
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_sa_rejection_branches as probe


def state_chain(row):
    state = row["target_after"]
    yield state
    for event in row["events"]:
        state = probe.pilot.apply_state_delta(state, event["delta"])
        yield state
    if probe.pilot.state_fingerprint(state) != probe.pilot.state_fingerprint(row["final"]):
        raise ValueError("saved final state mismatch")


def reconvergence(left, right):
    a, b = list(state_chain(left)), list(state_chain(right))
    equal = [probe.pilot.state_fingerprint(x) == probe.pilot.state_fingerprint(y)
             for x, y in zip(a, b)]
    first = next((i for i, value in enumerate(equal) if value), None)
    # State index zero is immediately after the one-decision intervention.
    controls_equal = None
    if first is not None:
        controls_equal = all(
            all(x[k] == y[k] for k in ("decision", "action", "temperature", "uniform", "pool", "selected_index"))
            for x, y in zip(left["events"][first:], right["events"][first:]))
    return dict(first_equal_after_followup_steps=first,
                common_followup_steps=min(len(a), len(b)) - 1,
                equality_persists=None if first is None else all(equal[first:]),
                subsequent_controls_equal=controls_equal,
                equal_final_fingerprint=probe.pilot.state_fingerprint(a[-1]) == probe.pilot.state_fingerprint(b[-1]))


def classify(left, right):
    if left["feasible"] and right["feasible"]:
        return "both_feasible"
    if left["feasible"]:
        return "rejection_only_feasible_other_horizon" if right["stop"] == "horizon" else "rejection_only_feasible_other_resource_unknown"
    if right["feasible"]:
        return "acceptance_only_feasible_other_horizon" if left["stop"] == "horizon" else "acceptance_only_feasible_other_resource_unknown"
    return "neither_feasible_within_bound"


def analyze():
    root = probe.OUT
    report = probe.read_json(root / "branches_report.json")
    plan = probe.read_json(root / "plan.json")
    if not report["complete"] or report["jobs"] != 18 or report["plan_sha256"] != probe.sha256_file(root / "plan.json"):
        raise ValueError("incomplete or changed collection")
    for name, digest in plan["inputs"].items():
        if probe.sha256_file(ROOT / name) != digest:
            raise ValueError("registered source changed: " + name)
    scheduled = probe.jobs(plan, "branches")
    if set(report["files"]) != {j["job_id"] + ".json" for j in scheduled}:
        raise ValueError("unexpected result set")
    rows = []
    for job in scheduled:
        name = job["job_id"] + ".json"
        if probe.sha256_file(root / "branches" / name) != report["files"][name]:
            raise ValueError("branch SHA mismatch")
        row = probe.load(job, "branches")
        if row["followup_steps"] != len(row["events"]) or row["conflicts"] != [s["num_of_colliding_pairs"] for s in state_chain(row)]:
            raise ValueError("trajectory summary mismatch")
        rows.append(row)
    comparisons = probe.compare(rows)
    if comparisons != report["comparisons"]:
        raise ValueError("paired summary mismatch")
    by_id = {r["job_id"]: r for r in rows}
    for item in comparisons:
        left, right = [by_id[item["target_id"] + "-" + m] for m in probe.MODES]
        item["classification"] = classify(left, right)
        item["reconvergence"] = reconvergence(left, right)
    result = dict(
        schema="lns2.sa_rejection_branch_analysis.v1", posthoc_readonly=True,
        no_ttf=True, no_default_promotion=True, no_additional_solver_runs=True,
        source_report_sha256=probe.sha256_file(root / "branches_report.json"),
        plan_sha256=probe.sha256_file(root / "plan.json"),
        analysis_source_sha256=probe.sha256_file(Path(__file__)),
        result_files=report["files"], comparisons=comparisons,
        targets=len(comparisons), trajectories=len({r["case_id"] for r in rows}),
        maps=len({r["map_id"] for r in rows}), branches=len(rows),
        successes_by_mode={m:sum(bool(r["feasible"]) for r in rows if r["mode"] == m) for m in probe.MODES},
        stops=dict(Counter(r["stop"] for r in rows)),
        classifications=dict(Counter(c["classification"] for c in comparisons)),
        max_node_cap_overshoot=max(r["node_cap_overshoot"] for r in rows),
        total_followup_steps=sum(r["followup_steps"] for r in rows),
        total_followup_generated=sum(r["followup_generated"] for r in rows),
        decision="retain_frozen_sa_rejection_no_acceptance_rule_promotion")
    probe.write_json(root / "analysis.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2))
