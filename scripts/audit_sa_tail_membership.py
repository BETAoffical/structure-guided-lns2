"""Retrospective membership audit of sealed H32 traces; never calls a solver."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._common import json_fingerprint, read_json, sha256_file, write_json
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.state_analysis import reconstruct_conflicts
from scripts import collect_sa_structural_focus as source
from scripts.audit_sa_history_information import require

EVIDENCE = ROOT / "artifacts/sa-structural-focus-continuation-v1/evidence.json"
OUTPUT = ROOT / "build/sa-tail-membership-audit-v1"
WINDOW = 8


def closed_cross(cell, rows, cols):
    """Grid distance at most one, without row wrap; not a reachability proof."""
    y, x = divmod(cell, cols)
    return {ny * cols + nx for ny, nx in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
            if 0 <= ny < rows and 0 <= nx < cols}


def contacts(state, events, conflict_members):
    paths = {int(a["id"]): a["path"] for a in state["agents"]}
    near, parked = set(), set()
    # Nearby current occupancy is only an exposure descriptor, not a blocker.
    queries = set((e.time, cell) for e in events for cell in e.cells)
    for time, cell in queries:
        cells = closed_cross(cell, state["rows"], state["cols"])
        for agent, path in paths.items():
            if agent in conflict_members:
                continue
            if path[min(time, len(path) - 1)] in cells:
                near.add(agent)
                if time >= len(path) - 1:
                    parked.add(agent)
    core_cells = set(cell for agent in conflict_members for cell in paths[agent])
    overlap = {agent for agent, path in paths.items()
               if agent not in conflict_members and core_cells.intersection(path)}
    return near, parked, overlap


def audit_decision(before, after, event):
    paths = {int(a["id"]): a["path"] for a in before["agents"]}
    after_paths = {int(a["id"]): a["path"] for a in after["agents"]}
    require(len(paths) == len(before["agents"]) and set(paths) == set(after_paths), "agent identity")
    events = reconstruct_conflicts(before["agents"])
    pairs = sorted({(e.left, e.right) for e in events})
    require(pairs == sorted(tuple(sorted(edge)) for edge in before["conflict_edges"]), "conflict reconstruction")
    require(len(pairs) == before["num_of_colliding_pairs"], "conflict count")
    active = {a for pair in pairs for a in pair}
    selected = set(event["action"]["agents"])
    require(len(selected) == len(event["action"]["agents"]) and selected <= paths.keys(), "invalid members")
    require(selected == set(event["metrics"]["neighborhood"]), "applied members changed")
    require(all(paths[a] == after_paths[a] for a in paths.keys() - selected), "external path changed")
    pool = event["pool"]
    chosen = pool[event["selected_index"]]
    require(set(chosen["agents"]) == selected, "pool selection mismatch")
    require(len({p["candidate_id"] for p in pool}) == len(pool), "duplicate candidate id")
    for p in pool:
        require(len(set(p["agents"])) == len(p["agents"]) and set(p["agents"]) <= paths.keys(), "bad pool members")
    fillers = selected - active
    all_covered = active <= selected
    low_id_expected = active | set(sorted(paths.keys() - active)[:max(0, 16 - len(active))])
    fill_applicable = bool(active) and all_covered and len(selected) == 16 and len(active) < 16
    near, parked, overlap = contacts(before, events, active)
    alternatives = [p for p in pool if set(p["agents"]) != selected
                    and len(p["agents"]) == len(selected) and active <= set(p["agents"])]
    # Scores are recorded preferences only. Unchosen candidates have no labels here.
    alt_details = [dict(candidate_id=p["candidate_id"], members=p["agents"],
                        score=p["score"], near_fillers=sorted((set(p["agents"]) - active) & near))
                   for p in alternatives]
    return dict(decision=event["decision"], conflicts_before=len(pairs),
                conflicts_after=after["num_of_colliding_pairs"], pairs=[list(p) for p in pairs],
                selected_id=chosen["candidate_id"], families=chosen["selection_families"],
                selected_members=sorted(selected), active_members=sorted(active),
                all_conflict_endpoints_selected=all_covered,
                filler_members=sorted(fillers), filler_count=len(fillers),
                low_id_fill_applicable=fill_applicable,
                exact_low_id_fill=fill_applicable and selected == low_id_expected,
                structural=any(f.startswith("structpool-") for f in chosen["selection_families"]),
                near_fillers=sorted(fillers & near), parked_near_fillers=sorted(fillers & parked),
                near_outsiders=sorted(near - selected), parked_near_outsiders=sorted(parked - selected),
                spatial_overlap_fillers=sorted(fillers & overlap),
                spatial_overlap_outsiders=sorted(overlap - selected),
                changed_fillers=sorted(a for a in fillers if paths[a] != after_paths[a]),
                changed_selected_conflict_members=sorted(a for a in selected & active if paths[a] != after_paths[a]),
                accepted=event["metrics"]["replan_success"], selected_score=chosen["score"],
                same_size_full_coverage_alternatives=alt_details,
                path_signature=json_fingerprint(sorted(paths.items())))


def inspect_branch(item):
    job, plan, out = item
    row = source.old.receipt(out / "trials" / job["job_id"], job, plan)
    require(row["status"] == "ok" and row["stop"] in ("feasible", "horizon"), "unknown source outcome")
    entry = job["root"]
    state = read_json(ROOT / entry["source_root"])["state"]
    decisions = []
    start = max(0, len(row["events"]) - WINDOW)
    for i, event in enumerate(row["events"]):
        after = apply_state_delta(state, event["delta"])
        require(event["metrics"]["conflicts_before"] == state["num_of_colliding_pairs"] and
                event["metrics"]["conflicts_after"] == after["num_of_colliding_pairs"], "transition counts")
        if i >= start:
            decisions.append(audit_decision(state, after, event))
        state = after
    require(state["num_of_colliding_pairs"] == row["final_conflicts"], "final conflicts")
    require((row["stop"] == "feasible") == (row["final_conflicts"] == 0), "completion label")
    return dict(job_id=job["job_id"], state_id=entry["state_id"], map_id=entry["map_id"],
                arm="anchor" if job["candidate_id"] == entry["anchor_id"] else "alternate",
                trial=job["trial"], stop=row["stop"], final_conflicts=row["final_conflicts"],
                repairs=len(row["events"]), decisions=decisions)


def summarize(rows):
    groups = {}
    for arm in ("anchor", "alternate"):
        for outcome in ("all", "feasible", "horizon"):
            branches = [r for r in rows if r["arm"] == arm and (outcome == "all" or r["stop"] == outcome)]
            ds = [d for r in branches for d in r["decisions"]]
            fills = [d for d in ds if d["low_id_fill_applicable"]]
            structural = [d for d in fills if d["structural"]]
            groups[arm + "/" + outcome] = dict(
                branches=len(branches), tail_decisions=len(ds), fill_applicable=len(fills),
                exact_low_id_fill=sum(d["exact_low_id_fill"] for d in fills),
                structural_fill_applicable=len(structural),
                structural_exact_low_id_fill=sum(d["exact_low_id_fill"] for d in structural),
                with_same_size_coverage_alternative=sum(bool(d["same_size_full_coverage_alternatives"]) for d in fills),
                with_near_filler=sum(bool(d["near_fillers"]) for d in fills),
                with_near_outsider=sum(bool(d["near_outsiders"]) for d in fills),
                with_parked_near_outsider=sum(bool(d["parked_near_outsiders"]) for d in fills),
                filler_slots=sum(d["filler_count"] for d in fills),
                near_filler_slots=sum(len(d["near_fillers"]) for d in fills),
                spatial_overlap_filler_slots=sum(len(d["spatial_overlap_fillers"]) for d in fills),
                changed_filler_slots=sum(len(d["changed_fillers"]) for d in fills),
                accepted=sum(d["accepted"] for d in fills))
    recurring = []
    for r in rows:
        ds = r["decisions"]
        if r["stop"] == "horizon" and 1 <= r["final_conflicts"] <= 3:
            recurring.append(dict(job_id=r["job_id"], state_id=r["state_id"], map_id=r["map_id"],
                arm=r["arm"], trial=r["trial"], final_conflicts=r["final_conflicts"],
                unique_selected_sets=len({tuple(d["selected_members"]) for d in ds}),
                unique_pair_sets=len({json_fingerprint(d["pairs"]) for d in ds}),
                unique_path_states=len({d["path_signature"] for d in ds}),
                decisions=len(ds), all_accepted=all(d["accepted"] for d in ds),
                always_full_coverage=all(d["all_conflict_endpoints_selected"] for d in ds),
                near_filler_union=sorted({a for d in ds for a in d["near_fillers"]}),
                near_outsider_union=sorted({a for d in ds for a in d["near_outsiders"]}),
                alternative_available_decisions=sum(bool(d["same_size_full_coverage_alternatives"]) for d in ds)))
    return dict(groups=groups, low_residual_branches=recurring,
                maps=len({r["map_id"] for r in rows}), roots=len({r["state_id"] for r in rows}),
                branches=len(rows), stops=dict(Counter(r["stop"] for r in rows)),
                decision="observational_only_no_controller_promotion", no_ttf=True, training_allowed=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--verify", action="store_true", help="verify receipts and deterministically reproduce this audit")
    args = parser.parse_args()
    require(1 <= args.workers <= 20, "workers must be 1..20")
    plan, out = source.verify()
    evidence = read_json(EVIDENCE)
    require(evidence["binding"] == plan["binding"], "evidence binding")
    inputs = {str(EVIDENCE.relative_to(ROOT)).replace("\\", "/"): sha256_file(EVIDENCE),
              str(Path(__file__).relative_to(ROOT)).replace("\\", "/"): sha256_file(Path(__file__)),
              **evidence["files"], **plan["inputs"]}
    jobs = source.old.schedule(plan)
    for job in jobs:
        for name in ("result.json", "receipt.json", "started.json"):
            path = out / "trials" / job["job_id"] / name
            inputs[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    require(len(jobs) == 240, "sealed cohort changed")
    for name, digest in inputs.items():
        require(sha256_file(ROOT / name) == digest, "input drift: " + name)
    items = [(j, plan, out) for j in jobs]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for i, row in enumerate(executor.map(inspect_branch, items), 1):
            rows.append(row)
            if i % 20 == 0 or i == len(jobs):
                print(f"audited {i}/{len(jobs)} existing branches; no solver calls", flush=True)
    rows.sort(key=lambda row: row["job_id"])
    for name, digest in inputs.items():
        require(sha256_file(ROOT / name) == digest, "input changed during audit: " + name)
    report = dict(schema="lns2.sa.tail_membership_audit.v1", source_binding=plan["binding"],
                  retrospective=True, window=WINDOW, inputs=inputs,
                  rows_fingerprint=json_fingerprint(rows), summary=summarize(rows),
                  limitations=["Tail windows are retrospectively aligned to termination or H32 truncation.",
                               "Decisions and trials on one root/map are not independent samples.",
                               "Proximity and spatial overlap do not prove blockers or dispensable fillers.",
                               "Unchosen candidate outcomes are unknown; current scores are not causal labels."])
    if args.verify:
        require(read_json(OUTPUT / "rows.json") == rows, "audit rows drift")
        require(read_json(OUTPUT / "report.json") == report, "audit report drift")
        print("verified unchanged source and identical membership audit", flush=True)
    else:
        require(not (OUTPUT / "report.json").exists(), "audit exists; use --verify")
        write_json(OUTPUT / "rows.json", rows)
        write_json(OUTPUT / "report.json", report)
        print(report["summary"]["groups"], flush=True)


if __name__ == "__main__":
    main()
