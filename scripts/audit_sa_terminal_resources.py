"""Explain saved terminal conflicts; never search, replan, or train."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import probe_sa_terminal_candidates as source
from experiments.state_analysis import reconstruct_conflicts, _grid_neighbors

run = source.run
OUT = ROOT / "build/sa-terminal-resource-audit-v1"
REPORT_SHA = "0174a93e33f4787fea6cc48f01bb00462a6890699d5a5bdeac3197ca37f6f702"
HISTORY = (
    "build/sa-pair-compatibility-v1/plan.json",
    "build/sa-external-obstruction-v1/report.json",
    "build/sa-goal-obstruction-v1/report.json",
    "docs/SA_CERTIFICATE_REPAIR_RESULTS_ZH.md",
    "docs/LOCAL_PATH_COMPATIBILITY_V1_RESULT.md",
    "docs/SA_PLATEAU_CANDIDATE_RESULTS_ZH.md",
)
CODE = ("scripts/audit_sa_terminal_resources.py", "tests/evaluation/test_sa_terminal_resources.py",
        "docs/SA_TERMINAL_RESOURCES_PROTOCOL_ZH.md")
STATES = {}


def map_signature(state):
    return run.json_fingerprint({k: state[k] for k in ("rows", "cols", "obstacles")})


def task_signature(state, paths=False):
    fields = ("id", "start", "goal", "path") if paths else ("id", "start", "goal")
    return run.json_fingerprint(dict(map=map_signature(state), agents=[
        {k: a[k] for k in fields} for a in sorted(state["agents"], key=lambda a: a["id"])]))


def phase(path, tick):
    run.require(path and tick >= 0, "invalid path/time")
    end = len(path)-1
    if tick > end: return "goal_hold"
    if tick == end: return "goal_arrival"
    if tick == 0: return "start"
    return "wait" if path[tick] == path[tick-1] else "move"


def checked_events(state):
    events = reconstruct_conflicts(state["agents"])
    edges = {tuple(sorted((e.left, e.right))) for e in events}
    reported = {tuple(sorted(e)) for e in state["conflict_edges"]}
    run.require(edges == reported and len(edges) == state["num_of_colliding_pairs"], "conflict edge mismatch")
    run.require(bool(state["feasible"]) == (not edges), "feasible flag mismatch")
    return events


def describe_events(before, after, order):
    agents = {a["id"]: a for a in after["agents"]}
    old = {a["id"]: a for a in before["agents"]}
    selected = set(order)
    run.require(len(order) == len(selected) and selected <= set(agents) and set(agents) == set(old), "agent identity")
    run.require(all(a["path"] == old[i]["path"] for i, a in agents.items() if i not in selected), "external path changed")
    indices = {aid: n for n, aid in enumerate(order)}
    original = {tuple(sorted(e)) for e in before["conflict_edges"]}
    rows = []
    for e in checked_events(after):
        pair = (e.left, e.right)
        endpoints = []
        for aid in pair:
            path = agents[aid]["path"]
            endpoints.append(dict(id=aid, selected=aid in selected, phase=phase(path, e.time),
                current=path[min(e.time, len(path)-1)], previous=path[min(max(0, e.time-1), len(path)-1)],
                terminal_time=len(path)-1, goal=agents[aid]["goal"],
                path_changed=path != old[aid]["path"], repair_order_index=indices.get(aid)))
        selected_count = sum(a["selected"] for a in endpoints)
        rows.append(dict(pair=list(pair), time=e.time, kind=e.kind, cells=list(e.cells),
            coordinates=[list(divmod(c, after["cols"])) for c in e.cells],
            relation=("external", "boundary", "internal")[selected_count],
            original_pair=pair in original, endpoints=endpoints,
            goal_hold=any(a["phase"] == "goal_hold" for a in endpoints),
            goal_arrival=any(a["phase"] == "goal_arrival" for a in endpoints)))
    return rows


def goal_context(state, aid):
    agents = {a["id"]: a for a in state["agents"]}
    run.require(aid in agents, "unknown target")
    goals = {}
    for a in agents.values(): goals.setdefault(a["goal"], []).append(a["id"])
    free = {i for i, v in enumerate(state["obstacles"]) if not v}
    goal = agents[aid]["goal"]
    neighbors = _grid_neighbors(goal, state["rows"], state["cols"], free)
    return dict(agent=aid, start=agents[aid]["start"], goal=goal,
        goal_coordinates=list(divmod(goal, state["cols"])), terminal_time=len(agents[aid]["path"])-1,
        path_suffix=agents[aid]["path"][-16:], neighbors=[dict(cell=c,
            coordinates=list(divmod(c, state["cols"])), final_owners=[dict(agent=i,
                terminal_time=len(agents[i]["path"])-1) for i in sorted(goals.get(c, []))]) for c in neighbors],
        all_free_neighbors_have_final_owners=bool(neighbors) and all(goals.get(c) for c in neighbors),
        not_an_infeasibility_certificate=True)


def history_comparison(states):
    plan = run.read_json(ROOT / HISTORY[0])
    cases = {c["id"]: c for c in plan["cases"]}
    rows = []
    for path in HISTORY[1:3]:
        for proof in run.read_json(ROOT / path)["proofs"]:
            if proof["status"] not in {"forced_prefix_collision", "forced_goal_slot"}: continue
            old = cases[proof["job"]["case_id"]]["state"]
            for rid, state in states.items():
                rows.append(dict(root_id=rid, history_file=path, old_case=proof["job"]["case_id"],
                    mechanism=proof["status"], old_pair=proof["job"]["pair"], old_event=proof["event"],
                    same_map=map_signature(state) == map_signature(old),
                    same_task=task_signature(state) == task_signature(old),
                    same_paths=task_signature(state, True) == task_signature(old, True),
                    old_agents=len(old["agents"]), current_agents=len(state["agents"]),
                    old_selected=cases[proof["job"]["case_id"]]["selected"],
                    reusable_certificate=False))
    run.require(all(not r["same_paths"] for r in rows), "matched old state requires explicit selected-set comparison")
    return rows


def tally(rows):
    event_kinds, relations, pairs, times, parked_agents = Counter(), Counter(), Counter(), Counter(), Counter()
    result = dict(branches=len(rows), transferred=sum(r["transferred"] for r in rows),
        events=sum(len(r["events"]) for r in rows), branches_with_goal_hold=0,
        branches_with_external_goal_hold=0, branches_with_internal_goal_hold=0,
        branches_with_new_pairs=0, rolled_back=0)
    for row in rows:
        events = row["events"]
        result["branches_with_goal_hold"] += int(any(e["goal_hold"] for e in events))
        result["branches_with_external_goal_hold"] += int(any(
            a["phase"] == "goal_hold" and not a["selected"] for e in events for a in e["endpoints"]))
        result["branches_with_internal_goal_hold"] += int(any(e["goal_hold"] and e["relation"] == "internal" for e in events))
        result["branches_with_new_pairs"] += int(any(not e["original_pair"] for e in events))
        result["rolled_back"] += int(row["rolled_back"])
        event_kinds.update(e["kind"] for e in events)
        relations.update(e["relation"] for e in events)
        times.update(str(e["time"]) for e in events)
        pairs.update({tuple(e["pair"]) for e in events if not e["original_pair"]})
        parked_agents.update({a["id"] for e in events for a in e["endpoints"] if a["phase"] == "goal_hold"})
    result.update(event_kinds=dict(event_kinds), event_relations=dict(relations), event_times=dict(times),
        new_pairs_by_branch=[dict(pair=list(k), branches=v) for k,v in sorted(pairs.items(), key=lambda kv: (-kv[1],kv[0]))],
        parked_agents_by_branch=[dict(agent=k, branches=v) for k,v in sorted(parked_agents.items(), key=lambda kv: (-kv[1],kv[0]))])
    return result


def init_worker(states):
    global STATES
    STATES = states


def audit_branch(job):
    before = STATES[job["root_id"]]
    path = ROOT / job["path"]
    run.require(run.sha256_file(path) == job["sha256"], "branch SHA")
    row = run.check_seal(run.read_json(path))
    run.require(row["status"] == "ok" and row["binding"] == job["source_binding"] and
        row["root_id"] == job["root_id"] and row["job"] == job["job"], "branch identity")
    run.require(row["before"] == source.q.state_fingerprint(before), "before fingerprint")
    after = source.q.apply_state_delta(before, row["delta"])
    run.require(row["after"] == source.q.state_fingerprint(after), "after fingerprint")
    order = row["metrics"]["repair_order"]
    run.require(set(order) == set(row["job"]["action"]["agents"]), "actual order members")
    events = describe_events(before, after, order)
    original = {tuple(sorted(e)) for e in before["conflict_edges"]}
    remaining = {tuple(e["pair"]) for e in events}
    return dict(root_id=job["root_id"], job_id=row["job"]["job_id"], candidate_id=row["job"]["candidate_id"],
        trial=row["job"]["trial"], selected_agents=row["job"]["action"]["agents"], repair_order=order,
        original_pair=list(next(iter(original))), conflicts=len(remaining),
        transferred=not (original & remaining) and bool(remaining), rolled_back=row["metrics"]["pp_rolled_back"],
        attempt_paths_available=not row["metrics"]["pp_rolled_back"], events=events)


def prepare():
    reg, src = source.verify()
    run.require(run.sha256_file(src / "report.json") == REPORT_SHA, "source report changed")
    complete = run.check_seal(run.read_json(src / "collection.complete.json"))
    run.require(complete["binding"] == reg["binding"], "collection identity")
    source.identity.verify_files(complete["files"], src)
    inputs = dict(reg["inputs"])
    for name in ("registration.json", "report.json", "collection.complete.json"):
        inputs[(src/name).relative_to(ROOT).as_posix()] = run.sha256_file(src/name)
    for name, digest in complete["files"].items(): inputs[(src/name).relative_to(ROOT).as_posix()] = digest
    for name in (*CODE, *HISTORY): inputs[name] = run.sha256_file(ROOT/name)
    jobs = []
    for target in reg["targets"]:
        for job in source.branch_jobs(reg, target):
            if job["trial"] < 0: continue
            path = source.branch_folder(reg,target,job)/"result.json"
            jobs.append(dict(root_id=target["id"], source_binding=reg["binding"], job=job,
                path=path.relative_to(ROOT).as_posix(), sha256=run.sha256_file(path)))
    body = dict(schema="lns2.sa.terminal_resources.v1", source_commit=run.subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), inputs=inputs,
        source_folder=src.relative_to(ROOT).as_posix(), roots=[dict(id=t["id"], decision=t["decision"], edge=t["edge"]) for t in reg["targets"]],
        jobs=jobs, no_solver_calls=True, no_training=True, no_ttf=True, exploratory_posthoc=True,
        workers=20, automatic_expansion=False)
    body["binding"] = run.json_fingerprint(body)
    run.require(not OUT.exists(), "preserve prior output")
    run.once(OUT/"registration.json", run.sealed(body))
    return dict(status="prepared", jobs=len(jobs), no_solver_calls=True)


def verify():
    reg = run.check_seal(run.read_json(OUT/"registration.json"))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in {"binding","integrity"}}), "binding")
    source.identity.verify_files(reg["inputs"])
    return reg


def analyze():
    reg = verify()
    with source.budget.recovery.strict_lock(OUT, reg["binding"], "terminal-resource-audit"):
        if (OUT/"report.json").exists():
            report = run.check_seal(run.read_json(OUT/"report.json"))
            run.require(report["binding"] == reg["binding"], "saved report binding")
            source.identity.verify_files(report["files"], OUT)
            return dict(status="verified_existing", branches=report["all"]["branches"])
        states = {r["id"]: run.read_json(ROOT/reg["source_folder"]/"roots"/(r["id"]+".json")) for r in reg["roots"]}
        contexts = {}
        for target in reg["roots"]:
            state = states[target["id"]]
            run.require(len(checked_events(state)) == 1, "expected single-event roots")
            contexts[target["id"]] = dict(target=target, events=describe_events(state,state,[]),
                goals=[goal_context(state,aid) for aid in target["edge"]],
                map_sha256=map_signature(state), task_sha256=task_signature(state))
        rows = []
        run.write_json(OUT/"run_status.json",dict(status="running",binding=reg["binding"]))
        try:
            with ProcessPoolExecutor(max_workers=reg["workers"], initializer=init_worker, initargs=(states,)) as pool:
                futures = [pool.submit(audit_branch,j) for j in reg["jobs"]]
                for future in as_completed(futures):
                    row = future.result()
                    run.once(OUT/"branches"/(row["root_id"]+"-"+row["job_id"]+".json"),run.sealed(row))
                    rows.append(row)
                    if len(rows)%20 == 0: print("AUDITED",len(rows),"/",len(reg["jobs"]),flush=True)
            rows.sort(key=lambda r:(r["root_id"],r["job_id"]))
            report = dict(binding=reg["binding"], roots=contexts, all=tally(rows),
                transferred=tally([r for r in rows if r["transferred"]]),
                by_root={rid:dict(all=tally([r for r in rows if r["root_id"]==rid]),
                    transferred=tally([r for r in rows if r["root_id"]==rid and r["transferred"]])) for rid in sorted(states)},
                historical_certificates=history_comparison(states),
                no_solver_calls=True,no_training=True,no_ttf=True,exploratory_posthoc=True,
                files={p.relative_to(OUT).as_posix():run.sha256_file(p) for p in sorted((OUT/"branches").glob("*.json"))})
            run.require(len(rows)==228 and report["transferred"]["branches"]==66 and len(report["files"])==228, "expected source inventory")
            run.require(report["all"]["rolled_back"]==9 and all(r["conflicts"]>0 for r in rows), "original outcome preservation")
            source.identity.verify_files(reg["inputs"])
            run.once(OUT/"report.json",run.sealed(report))
            run.write_json(OUT/"run_status.json",dict(status="complete",report_sha256=run.sha256_file(OUT/"report.json")))
        except BaseException:
            run.write_json(OUT/"run_status.json",dict(status="needs_inspection",binding=reg["binding"]))
            raise
    return dict(status="complete",branches=len(rows),transferred={k:v for k,v in report["transferred"].items()
        if isinstance(v,int)},no_solver_calls=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","analyze","verify"))
    args = parser.parse_args()
    result = prepare() if args.phase=="prepare" else analyze() if args.phase=="analyze" else dict(binding=verify()["binding"],status="verified")
    print(run.json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__": main()
