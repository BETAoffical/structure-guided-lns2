"""Post-hoc, read-only attribution of logged selections; no solver or model fit."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, sha256_file, json_fingerprint
from experiments.sa_paired_completion import require, validate_dataset
from scripts.run_sa_paired_closed_loop import once

SOURCE = "build/sa-bootstrap-closed-loop-v1"
REPORT_SHA = "7c007daba518be9039c8a334994e9ea4c9bcf9784efaa4c2b0817208dc307732"
INDEX = "build/sa-spatiotemporal-model-probe-v1/development_index.json"
INDEX_SHA = "16a0b511e8e28cfcf0e24befd6da90f577a5dd1533e2b8c04d9685b87fa11e36"
OUTPUT = "build/sa-selection-coverage-audit-v1"
ARMS = ("frozen", "gbdt", "posterior", "uniform")
COVERAGE = tuple("realized." + name for name in (
    "internal_conflict_coverage", "incident_conflict_coverage",
    "internal_event_coverage", "incident_event_coverage"))


def relation(value):
    return "lower" if value < 0 else "higher" if value > 0 else "equal"


def candidate_features(pool, ids, features):
    require(len(ids) == len(set(ids)) == len(features) and ids, "subset/feature count")
    by_id = {c["candidate_id"]: c for c in pool}
    require(len(by_id) == len(pool) and set(ids) <= set(by_id), "pool identity")
    rows = dict(zip(ids, features, strict=True))
    for cid, row in rows.items():
        agents = by_id[cid]["agents"]
        require(agents and len(agents) == len(set(agents)), "candidate agents")
        require(row["proposal.actual_size"] == len(agents), "actual size mismatch")
        for key in COVERAGE:
            require(type(row[key]) in (float, int) and math.isfinite(row[key])
                    and 0 <= row[key] <= 1, "invalid coverage")
        require(type(row["state.colliding_pairs"]) in (float, int)
                and math.isfinite(row["state.colliding_pairs"])
                and row["state.colliding_pairs"] > 0, "invalid conflict count")
    require(len({r["state.colliding_pairs"] for r in rows.values()}) == 1, "different states")
    return rows


def inspect_event(event):
    pool = event["pool"]
    rows = candidate_features(pool, event["subset"], event["features"])
    # selected_index addresses the FULL pool, not the four-candidate subset.
    index = event["selected_index"]
    require(type(index) is int and 0 <= index < len(pool), "invalid selected index")
    selected = pool[index]["candidate_id"]
    anchor = event["anchor_id"]
    require(selected in rows and anchor in rows, "choice not in subset")
    require(selected == event["ranking"]["selected"], "ranking/action identity")
    require(event["action"]["agents"] == pool[index]["agents"], "altered explicit action")
    decision = event["decision"]
    require(type(decision) is int and 0 <= decision < 128, "decision outside registered budget")
    s, a = rows[selected], rows[anchor]
    return dict(decision=decision, before=event["before"], selected=selected, anchor=anchor,
                changed=selected != anchor, selected_size=int(s["proposal.actual_size"]),
                anchor_size=int(a["proposal.actual_size"]),
                size_relation=relation(s["proposal.actual_size"] - a["proposal.actual_size"]),
                conflicts=s["state.colliding_pairs"],
                nonanchor_in_subset=len(rows) - 1,
                nonanchor_not_lower_internal_coverage=sum(
                    cid != anchor and f[COVERAGE[0]] >= a[COVERAGE[0]] for cid, f in rows.items()),
                coverage_delta={k: s[k] - a[k] for k in COVERAGE})


def stats(rows):
    return dict(decisions=len(rows), episodes=len({r["job_id"] for r in rows}),
                maps=len({r["map_id"] for r in rows}), changed=sum(r["changed"] for r in rows),
                selected_sizes=dict(sorted(Counter(str(r["selected_size"]) for r in rows).items())),
                size_relations=dict(sorted(Counter(r["size_relation"] for r in rows).items())),
                subset_with_no_nonanchor_at_anchor_internal_coverage=sum(
                    r["nonanchor_not_lower_internal_coverage"] == 0 for r in rows),
                coverage={k: dict(mean_delta=statistics.fmean(r["coverage_delta"][k] for r in rows)
                                 if rows else None,
                                 counts=dict(Counter(relation(r["coverage_delta"][k]) for r in rows)))
                          for k in COVERAGE})


def grouped_stats(rows):
    filters = {
        "all": lambda r: True,
        "first": lambda r: r["decision"] == 0,
        "early_0_31": lambda r: r["decision"] < 32,
        "middle_32_63": lambda r: 32 <= r["decision"] < 64,
        "late_64_127": lambda r: r["decision"] >= 64,
        "changed_same_size": lambda r: r["changed"] and r["size_relation"] == "equal",
        "changed_smaller": lambda r: r["changed"] and r["size_relation"] == "lower",
        "changed_larger": lambda r: r["changed"] and r["size_relation"] == "higher",
        "first_changed_same_size": lambda r: r["decision"] == 0 and r["changed"]
                                                and r["size_relation"] == "equal",
    }
    return {name: stats([r for r in rows if predicate(r)]) for name, predicate in filters.items()}


def label_comparisons(data):
    """Existing H32 labels only; each half is descriptive, not a confirmation test."""
    rows = []
    for state in data["states"]:
        candidates = state["candidates"]
        features = candidate_features(candidates, [c["candidate_id"] for c in candidates],
                                      [c["features"] for c in candidates])
        require(state["anchor_id"] in features, "missing training anchor")
        values = {}
        for c in candidates:
            trials = sorted(c["trials"], key=lambda t: t["trial"])
            require([t["trial"] for t in trials] == list(range(8)), "missing/duplicate trials")
            require(all(type(t["completed"]) is bool for t in trials), "invalid completion")
            values[c["candidate_id"]] = [int(t["completed"]) for t in trials]
        aid = state["anchor_id"]
        a = features[aid]
        for cid in sorted(values):
            if cid == aid:
                continue
            diffs = [v - b for v, b in zip(values[cid], values[aid], strict=True)]
            halves = [sum(diffs[:4]) / 4, sum(diffs[4:]) / 4]
            rows.append(dict(state_id=state["state_id"], map_id=state["map_id"], candidate_id=cid,
                anchor=aid, size_relation=relation(features[cid]["proposal.actual_size"] - a["proposal.actual_size"]),
                coverage_relation=relation(features[cid][COVERAGE[0]] - a[COVERAGE[0]]),
                completion_delta=sum(diffs) / 8, half_deltas=halves,
                both_halves="win" if min(halves) > 0 else "loss" if max(halves) < 0 else "other"))
    def summarize(rs):
        per_state = {sid: statistics.fmean(r["completion_delta"] for r in rs if r["state_id"] == sid)
                     for sid in sorted({r["state_id"] for r in rs})}
        state_maps = {r["state_id"]: r["map_id"] for r in rs}
        per_map = {m: statistics.fmean(v for sid, v in per_state.items() if state_maps[sid] == m)
                   for m in sorted(set(state_maps.values()))}
        return dict(comparisons=len(rs), states=len(per_state), maps=len(per_map),
            outcome_counts=dict(Counter(relation(r["completion_delta"]) for r in rs)),
            both_halves=dict(Counter(r["both_halves"] for r in rs)),
            per_state_mean_delta=per_state, per_map_state_equal_mean_delta=per_map,
            map_state_equal_mean_delta=statistics.fmean(per_map.values()) if per_map else None)

    summary = {g: summarize([r for r in rows if r["coverage_relation"] == g])
               for g in ("lower", "equal", "higher")}
    same_size = {g: summarize([r for r in rows if r["coverage_relation"] == g
                              and r["size_relation"] == "equal"])
                 for g in ("lower", "equal", "higher")}
    return dict(summary=summary, same_size=same_size, comparisons=rows)


def checked_path(base, name):
    path = (base / name).resolve()
    require(path.is_relative_to(base.resolve()), "manifest path escapes source")
    return path


def episode_worker(job):
    base, episode, expected_sha = job
    path = Path(base) / "episodes" / episode["job_id"] / "trace.jsonl"
    require(sha256_file(path) == expected_sha, "trace changed")
    rows = []
    first_signature = None
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            row = inspect_event(event)
            require(row["decision"] == len(rows), "nonsequential decisions")
            require(row["conflicts"] == episode["conflicts"][len(rows)], "state conflict mismatch")
            if not rows:
                first_signature = json_fingerprint({k: event[k] for k in
                    ("before", "pool", "subset", "anchor_id", "features")})
            row.update(job_id=episode["job_id"], pair_id=episode["pair_id"],
                       map_id=episode["map_id"], arm=episode["arm"])
            rows.append(row)
    require(len(rows) == episode["decisions"] > 0, "trace length mismatch")
    require(sum(r["changed"] for r in rows) == episode["changed_from_anchor"], "changed count mismatch")
    return dict(rows=rows, first_signature=first_signature)


def audit(workers):
    require(type(workers) is int and 1 <= workers <= 20, "workers must be 1..20")
    source = ROOT / SOURCE
    require(sha256_file(source / "report.json") == REPORT_SHA, "source report changed")
    require(sha256_file(ROOT / INDEX) == INDEX_SHA, "development index changed")
    report = read_json(source / "report.json")
    inputs = {SOURCE + "/report.json": REPORT_SHA, INDEX: INDEX_SHA}
    # These are the exact bytes covered by the previous full native/trace audit.
    for name, digest in sorted(report["files"].items()):
        path = checked_path(source, name)
        require(sha256_file(path) == digest, "source output changed: " + name)
        inputs[path.relative_to(ROOT).as_posix()] = digest
    episodes = report["episodes"]
    require(len(episodes) == len({e["job_id"] for e in episodes}) == 64, "episode identity")
    jobs = [(str(source), e, report["files"]["episodes/" + e["job_id"] + "/trace.jsonl"])
            for e in sorted(episodes, key=lambda e: e["job_id"])]
    if workers == 1:
        results = list(map(episode_worker, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(episode_worker, jobs))
    rows = [row for result in results for row in result["rows"]]
    require(len(rows) == 7414, "decision count")
    first = {}
    for job, result in zip(jobs, results, strict=True):
        e = job[1]
        require(e["arm"] not in first.setdefault(e["pair_id"], {}), "duplicate pair arm")
        first[e["pair_id"]][e["arm"]] = result["first_signature"]
    require(len(first) == 16 and all(set(arms) == set(ARMS) and len(set(arms.values())) == 1
                                   for arms in first.values()), "initial choice sets are not paired")
    data = validate_dataset(read_json(ROOT / INDEX))
    require(len(data["states"]) == 47 and data["horizon"] == 32, "development scope")
    labels = label_comparisons(data)
    limits = [c["features"]["state.colliding_pairs"] for s in data["states"] for c in s["candidates"]]
    result = dict(schema="lns2.sa.selection_coverage_audit.v1", source_binding=report["binding"],
        inputs=inputs, implementation_sha256=sha256_file(Path(__file__)),
        tests_sha256=sha256_file(ROOT / "tests/evaluation/test_sa_selection_coverage.py"),
        descriptive_only=True, new_solver_calls=0, training=False, controller_changed=False,
        evidence_limits=["Post-hoc association, not causal attribution", "No unchosen online outcomes",
                         "Steps are correlated within maps and episodes", "No coverage threshold or new policy selected",
                         "Development half-trial agreement is not an independent confirmation"],
        paired_initial_choice_sets=len(first), decisions=len(rows), labels=labels,
        training_conflicts_range=[min(limits), max(limits)],
        by_arm={arm: grouped_stats([r for r in rows if r["arm"] == arm]) for arm in ARMS},
        by_episode={e["job_id"]: stats([r for r in rows if r["job_id"] == e["job_id"]]) for e in episodes},
        by_map={m: {a: grouped_stats([r for r in rows if r["map_id"] == m and r["arm"] == a])
                    for a in ARMS} for m in sorted({r["map_id"] for r in rows})},
        decisions_outside_training_conflict_range={a: sum(not min(limits) <= r["conflicts"] <= max(limits)
                  for r in rows if r["arm"] == a) for a in ARMS},
        rows=rows)
    result["fingerprint"] = json_fingerprint(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--verify", action="store_true", help="Recompute and compare; never rewrite an output")
    args = parser.parse_args()
    result = audit(args.workers)
    target = ROOT / OUTPUT / "report.json"
    if args.verify:
        require(read_json(target) == result, "audit output mismatch")
    else:
        once(target, result)
    print(json.dumps(dict(fingerprint=result["fingerprint"], decisions=result["decisions"],
                          paired_initial_choice_sets=result["paired_initial_choice_sets"],
                          new_solver_calls=0, verified=args.verify), sort_keys=True))


if __name__ == "__main__":
    main()
