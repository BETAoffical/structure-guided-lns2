"""Post-hoc coverage diagnosis only; no solver, retraining, or controller changes."""
import json
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import read_json, write_json, sha256_file
from scripts.train_sa_history_selector import verify, locations, require


def scan_events(events, window=32):
    recent = deque(maxlen=window)
    best = None
    since_best = 0
    for event in events:
        count = Counter(recent)
        pool = event["pool"]
        current = event["metrics"]["conflicts_before"]
        if best is None:
            best = current
        seen = [count[tuple(sorted(c["agents"]))] for c in pool]
        yield dict(decision=event["decision"], pool_has_seen=any(n > 0 for n in seen),
                   pool_has_twice_tried=any(n >= 2 for n in seen),
                   chosen_seen=seen[event["selected_index"]] > 0,
                   seen_candidates=sum(n > 0 for n in seen), since_best=since_best,
                   prospective_history_opportunity=event["decision"] >= window and
                       since_best >= 16 and any(n >= 2 for n in seen))
        recent.append(tuple(sorted(event["metrics"]["neighborhood"])))
        after = event["metrics"]["conflicts_after"]
        since_best = 0 if after < best else since_best + 1
        best = min(best, after)


def scan_episode(job):
    name, digest, map_id, picked = job
    path = ROOT / name
    require(sha256_file(path) == digest, "source changed")
    phases, selected = {}, []
    with path.open(encoding="utf-8") as stream:
        for row in scan_events(json.loads(line) for line in stream):
            phase = "0-64" if row["decision"] <= 64 else "65-255" if row["decision"] <= 255 else "256+"
            bucket = phases.setdefault(phase, Counter())
            bucket["decisions"] += 1
            for key in ("pool_has_seen", "pool_has_twice_tried", "chosen_seen", "prospective_history_opportunity"):
                bucket[key] += int(row[key])
            if row["decision"] in picked:
                selected.append(row)
    require(sha256_file(path) == digest, "source changed during scan")
    return dict(path=name, map_id=map_id, phases={k:dict(v) for k,v in phases.items()}, selected=selected)


def main():
    plan = verify()
    cfg, out, source = locations()
    destination = out / "diagnosis.json"
    require(not destination.exists(), "preserve existing diagnosis")
    status = read_json(out / "training/status.json")
    require(status["status"] == "complete" and sha256_file(out / "training/report.json") == status["report_sha256"], "training result identity")
    reg = read_json(source / "registration.json")
    cases = {c["task_id"]:c for c in reg["cases"]}
    jobs = []
    for item in reg["schedule"]:
        if item["controller"] != "dual16_sa":
            continue
        path = (source/"episodes"/item["job_id"]/"first_phase/trace.jsonl").relative_to(ROOT).as_posix()
        picked = [t["decision"] for t in plan["targets"] if t["item"]["job_id"] == item["job_id"]]
        jobs.append((path, plan["inputs"][path], cases[item["task_id"]]["map_id"], picked))
    with ProcessPoolExecutor(max_workers=20) as executor:
        episodes = list(executor.map(scan_episode, jobs))
    phases = {}
    for episode in episodes:
        for phase, values in episode["phases"].items():
            phases.setdefault(phase, Counter()).update(values)
    selected = [r for e in episodes for r in e["selected"]]
    rows = [json.loads(line) for line in (out/"training/index.jsonl").read_text(encoding="utf-8").splitlines()]
    states = sorted({r["state_id"] for r in rows})
    coverage = {}
    for key in ("history.window_fill", "history.same_set_count", "history.valid_context_count", "history.stale_context_count"):
        coverage[key] = dict(candidates=sum(r["features"][key] > 0 for r in rows),
                             states=sum(any(r["features"][key] > 0 for r in rows if r["state_id"] == s) for s in states))
    result = dict(schema="lns2.sa_history_coverage_diagnosis.v1", post_hoc=True, binding=plan["binding"],
                  implementation_sha256=sha256_file(Path(__file__)), training_report_sha256=status["report_sha256"],
                  episodes=episodes, phases={k:dict(v) for k,v in phases.items()}, selected_state_count=len(selected),
                  selected_full_pools_with_history=sum(r["pool_has_seen"] for r in selected),
                  selected_full_pools_with_twice_tried=sum(r["pool_has_twice_tried"] for r in selected),
                  labeled_candidate_history=coverage,
                  boundary="same_membership_history_is_not_same_external_context; exploratory_inventory_not_new_cohort_selection",
                  new_solver_calls=0, new_training_runs=0)
    write_json(destination, result)
    print(json.dumps({k:v for k,v in result.items() if k != "episodes"}, indent=2))


if __name__ == "__main__":
    main()
