"""Read-only sealed-result audit and explicitly post-hoc focus diagnostics."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import json_fingerprint, read_json, sha256_file
from scripts import collect_sa_structural_focus as c
from scripts.audit_sa_history_information import require
from scripts.run_sa_path_quality import apply_state_delta


def audit():
    plan, out = c.verify()
    report = read_json(out / "report.json")
    records = read_json(out / "analysis_records.json")
    details = read_json(out / "trial_details.json")
    status = read_json(out / "run_status.json")
    require(status == dict(status="completed", binding=plan["binding"], complete=240, total=240), "unfinished collection")
    require(read_json(out / "preflight.json")["passed"], "preflight not passed")
    calculated = c.summarize(records, plan["config"])
    require(all(report[k] == v for k, v in calculated.items()), "summary does not reproduce")
    require(report["binding"] == plan["binding"], "report binding drift")
    for rel, digest in report["input_hashes"].items():
        require(sha256_file(ROOT / rel) == digest, "result bytes changed: " + rel)
    jobs = {j["job_id"]: j for j in c.old.schedule(plan)}
    require(len(details) == len(jobs) == 240 and {d["job_id"] for d in details} == set(jobs), "detail coverage")
    by_job, raw, stops = {}, {}, Counter()
    for d in details:
        j = jobs[d["job_id"]]
        row = c.old.receipt(out / "trials" / d["job_id"], j, plan)
        require(row["status"] == "ok" and row["stop"] in ("feasible", "horizon"), "unexpected unknown; this result audit requires complete H32 labels")
        arm = "anchor" if j["candidate_id"] == j["root"]["anchor_id"] else "alternate"
        require((d["state_id"], d["arm"], d["trial"], d["completion"], d["stop"]) ==
                (j["root"]["state_id"], arm, j["trial"], row["stop"] == "feasible", row["stop"]), "detail identity/label")
        initial = read_json(ROOT / j["root"]["source_root"])["state"]["num_of_colliding_pairs"]
        require(d["conflicts"] == [initial] + [e["metrics"]["conflicts_after"] for e in row["events"]], "conflict detail drift")
        require(d["generated"] == row["generated"] and d["final_cost"] == row["final_cost"] and
                d["repair_steps"] == len(row["events"]), "work detail drift")
        raw[d["job_id"]], by_job[d["job_id"]] = row, d
        stops[row["stop"]] += 1
    require(dict(stops) == report["stops"], "stop totals drift")
    for r in records:
        e = next(e for e in plan["roots"] if e["state_id"] == r["state_id"])
        for arm, cid in (("anchor", e["anchor_id"]), ("alternate", e["alternate_id"])):
            require(r[arm] == [by_job[f"{e['state_id']}-{cid}-t{t}"]["completion"] for t in range(8)], "aggregated labels drift")

    aggregate = {}
    for arm in ("anchor", "alternate"):
        rows = [d for d in details if d["arm"] == arm]
        aggregate[arm] = dict(mean_first_conflicts=sum(d["conflicts"][1] for d in rows) / 120,
            mean_final_conflicts=sum(d["conflicts"][-1] for d in rows) / 120,
            generated=sum(d["generated"] for d in rows), repairs=sum(d["repair_steps"] for d in rows),
            remaining_1_to_3=sum(1 <= d["conflicts"][-1] <= 3 for d in rows),
            last8_constant_nonzero=sum(d["stop"] == "horizon" and len(set(d["conflicts"][-8:])) == 1 for d in rows))
    first_better, better_but_lost, worse_but_gained = 0, 0, 0
    for e in plan["roots"]:
        for t in range(8):
            a, b = (by_job[f"{e['state_id']}-{e[key]}-t{t}"] for key in ("anchor_id", "alternate_id"))
            first_better += b["conflicts"][1] < a["conflicts"][1]
            better_but_lost += b["conflicts"][1] < a["conflicts"][1] and a["completion"] and not b["completion"]
            worse_but_gained += b["conflicts"][1] > a["conflicts"][1] and not a["completion"] and b["completion"]

    # Largest observed loss is selected AFTER collection, for explanation only.
    worst = min(records, key=lambda r: (sum(r["alternate"]) - sum(r["anchor"]), r["state_id"]))
    entry = next(e for e in plan["roots"] if e["state_id"] == worst["state_id"])
    root = read_json(ROOT / entry["source_root"])
    cases = []
    for t in range(8):
        row = raw[f"{entry['state_id']}-{entry['alternate_id']}-t{t}"]
        if row["stop"] == "feasible":
            continue
        state, snapshots = root["state"], []
        for ev in row["events"]:
            state = apply_state_delta(state, ev["delta"])
            snapshots.append({a["id"]: tuple(a["path"]) for a in state["agents"]})
        last = row["events"][-8:]
        members = last[-1]["action"]["agents"]
        cases.append(dict(trial=t, final_pairs=state["conflict_edges"], last8_path_snapshots=len({json_fingerprint(s) for s in snapshots[-8:]}),
            last8_accepted=sum(e["metrics"]["replan_success"] for e in last), members=members,
            last8_selected_ids=dict(Counter(e["pool"][e["selected_index"]]["candidate_id"] for e in last)),
            last8_distinct_paths_by_member={str(i): len({s[i] for s in snapshots[-8:]}) for i in members}))
    files = ("plan.json", "preflight.json", "run_status.json", "report.json", "analysis_records.json", "trial_details.json")
    return dict(schema="lns2.sa.structural_focus_evidence.v1", binding=plan["binding"], preregistration_commit=plan["commit"],
        files={str((out / name).relative_to(ROOT)).replace("\\", "/"): sha256_file(out / name) for name in files},
        auditor_sha256=sha256_file(Path(__file__)), native_sha256=plan["native_sha256"], pinned_inputs=len(plan["inputs"]),
        summary=calculated, stops=dict(stops), no_ttf=True, training_allowed=False, automatic_promotion=False,
        posthoc=dict(used_for_selection=False, aggregates=aggregate, first_better_pairs=first_better,
            first_better_but_completion_lost=better_but_lost, first_worse_but_completion_gained=worse_but_gained,
            worst_root=entry["state_id"], added=entry["variant"]["added"], removed=entry["variant"]["removed"], branches=cases))


if __name__ == "__main__":
    evidence = audit()
    saved = ROOT / "artifacts/sa-structural-focus-continuation-v1/evidence.json"
    if saved.exists():
        require(read_json(saved) == evidence, "registered evidence does not reproduce")
        print(json.dumps(dict(verified=True, evidence_sha256=sha256_file(saved), decision=evidence["summary"]["decision"]), sort_keys=True))
    else:
        print(json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False))
