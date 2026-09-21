"""Outcome-blind second-core materialization, not a controller or solver trial."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments._common import contained_file, json_fingerprint, read_json, sha256_file, write_json
from experiments.neighborhood_candidates import candidate_id
from experiments.repair_collection import _CollectionRunLock
from experiments.sa_paired_completion import require
from experiments.state_analysis import analyze_state
from lns2_selector.runtime import topology_candidates as topology
from scripts import audit_sa_member_contract as contract
from scripts.prepare_sa_source_matched import grouped

OUT = ROOT / "build/sa-structural-focus-v1"
FAMILIES = ("structpool-conflict-component:16", "structpool-spatiotemporal-hotspot:16")
CODE = ("scripts/probe_sa_structural_focus.py", "tests/evaluation/test_sa_structural_focus.py",
        "docs/SA_STRUCTURAL_FOCUS_PROTOCOL_ZH.md", "scripts/audit_sa_member_contract.py",
        "scripts/prepare_sa_source_matched.py", "experiments/_common.py",
        "experiments/state_analysis.py", "experiments/neighborhood_candidates.py",
        "lns2_selector/runtime/topology_candidates.py")


def action_input(state):
    """No trial outcome, future path, model score, or post-action metric is read."""
    fields = ("id", "start", "goal", "path", "conflict_degree")
    return {k: state[k] for k in ("rows", "cols", "obstacles", "conflict_edges", "num_of_colliding_pairs")} | {
        "agents": [{k: a[k] for k in fields if k in a} for a in state["agents"]]}


def two_cores(analysis):
    internal = Counter(analysis.component_id[e.left] for e in analysis.events)
    components = sorted((c for c in analysis.component_members if internal[c] > 0),
                        key=lambda c: (-internal[c], -len(analysis.component_members[c]), c))
    weights = topology._event_weights(analysis.events)
    component = [dict(key=[c], seeds=sorted(analysis.component_members[c]),
                      priority=dict(weights), events=internal[c]) for c in components[:2]]
    buckets = defaultdict(list)
    for event in analysis.events:
        for cell in event.cells:
            buckets[(event.time // 4, cell)].append(event)
    seen, hotspot = set(), []
    for key, events in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
        seeds = tuple(sorted({agent for e in events for agent in (e.left, e.right)}))
        if seeds in seen:
            continue
        seen.add(seeds)
        hotspot.append(dict(key=list(key), seeds=list(seeds), priority=dict(topology._event_weights(events)), events=len(events)))
        if len(hotspot) == 2:
            break
    result = dict(zip(FAMILIES, (component, hotspot)))
    for family, original in zip(FAMILIES, (topology._conflict_component_seed_data(analysis), topology._hotspot_seed_data(analysis))):
        require(bool(result[family]) == (original is not None), "primary core availability changed")
        if original is not None:
            require(set(result[family][0]["seeds"]) == original[0] and
                    result[family][0]["priority"] == original[1], "primary core differs from frozen generator")
    return result


def core_record(core):
    return {k: core[k] for k in ("key", "seeds", "events")} | {
        "priority": [[a, w] for a, w in sorted(core["priority"].items())]}


def materialize(state, pool, anchor_id, label_groups):
    before = json_fingerprint(state)
    state = action_input(state)
    state_before = json_fingerprint(state)
    grouped(pool, 16)
    by_id = {c["candidate_id"]: c for c in pool}
    require(anchor_id in by_id, "missing anchor")
    anchor = by_id[anchor_id]
    known = {a["id"] for a in state["agents"]}
    require(len(known) >= 16 and all(set(c["agents"]) <= known for c in pool), "invalid state/pool agents")
    require(all(candidate_id(c["agents"]) == c["candidate_id"] for c in pool), "candidate ID mismatch")
    analysis = analyze_state(state)
    require(analysis.pair_set == {tuple(sorted(e)) for e in state["conflict_edges"]} and
            len(analysis.pair_set) == state["num_of_colliding_pairs"], "conflict reconstruction mismatch")
    cores = two_cores(analysis)
    context = topology._neighborhood_context(state)
    primary, attempts = {}, []
    for family in FAMILIES:
        require(cores[family], "conflicted state has no structural core")
        core = cores[family][0]
        agents = topology._ranked_seed_neighborhood(state, analysis, core["seeds"], size=16,
                                                   priority=core["priority"], context=context)
        cid = candidate_id(agents)
        require(cid in by_id and by_id[cid]["agents"] == agents, "frozen structural candidate not reproduced")
        primary[family] = dict(candidate_id=cid, agents=agents, core=core_record(core))
        if family not in anchor["selection_families"]:
            continue
        require(cid == anchor_id, "structural anchor provenance mismatch")
        if len(cores[family]) == 1:
            attempts.append(dict(family=family, status="no_second_distinct_core", primary_core=core_record(core)))
            continue
        alternate = cores[family][1]
        members = topology._ranked_seed_neighborhood(state, analysis, alternate["seeds"], size=16,
                                                    priority=alternate["priority"], context=context)
        new_id = candidate_id(members)
        require(len(members) == len(set(members)) == 16 and set(members) <= known, "invalid alternate members")
        old, new = set(anchor["agents"]), set(members)
        attempts.append(dict(family=family, status="same_after_fill" if new_id == anchor_id else "different_members",
                             primary_core=core_record(core), alternate_core=core_record(alternate), candidate_id=new_id, agents=members,
                             removed=sorted(old - new), added=sorted(new - old), jaccard=len(old & new)/len(old | new),
                             in_original_pool=new_id in by_id,
                             labeled_with_anchor=any(anchor_id in g and new_id in g for g in label_groups)))
    require(json_fingerprint(state) == state_before, "action input mutated")
    return dict(input_snapshot=before, action_input_snapshot=state_before, anchor_id=anchor_id,
                anchor_families=anchor["selection_families"], primary=primary, attempts=attempts,
                structural_anchor=bool(set(anchor["selection_families"]) & set(FAMILIES)))


def once(path, value):
    require(not path.is_symlink() and not path.parent.is_symlink(), "unsafe output")
    require(path.resolve().is_relative_to((ROOT / "build").resolve()), "output outside build")
    if path.exists():
        require(read_json(path) == value, "existing artifact differs")
    else:
        write_json(path, value)


def prepare():
    require(not OUT.exists(), "existing output; use materialize or verify")
    evidence = contract.audit()
    require(read_json(ROOT / contract.OUTPUT) == evidence, "member audit changed")
    source = read_json(ROOT / contract.SOURCE / "plan.json")
    inputs = dict(evidence["inputs"])
    for name in CODE + (contract.OUTPUT,):
        require(name not in inputs or inputs[name] == sha256_file(ROOT / name), "conflicting source")
        inputs[name] = sha256_file(ROOT / name)
    plan = dict(schema="lns2.sa.structural_focus_plan.v1", inputs=inputs, roots=source["roots"],
                member_audit_sha=sha256_file(ROOT / contract.OUTPUT), workers=20,
                size=16, distinct_core_rank=2, hotspot_time_bin=4,
                solver_calls=0, model_fits=0, automatic_collection=False)
    plan["binding"] = json_fingerprint(plan)
    once(OUT / "plan.json", plan)
    return plan


def verify_plan():
    plan = read_json(OUT / "plan.json")
    require(plan["binding"] == json_fingerprint({k: v for k, v in plan.items() if k != "binding"}), "plan binding")
    for name, digest in plan["inputs"].items():
        require(sha256_file(contained_file(ROOT, name, field="focus input")) == digest, "input drift: " + name)
    require((plan["size"], plan["distinct_core_rank"], plan["hotspot_time_bin"], plan["workers"]) == (16, 2, 4, 20), "protocol changed")
    require(plan["solver_calls"] == plan["model_fits"] == 0 and not plan["automatic_collection"], "execution boundary")
    return plan


def job(data):
    entry, digest, groups = data
    root = contract.load_pinned(ROOT, entry["source_root"], digest, {})
    require(root["state_fingerprint"] == entry["root_fingerprint"] and
            root["old_selected_id"] == entry["anchor_id"] and root["source"]["id"] == entry["state_id"], "root identity")
    # Strip scores/outcomes before invoking the candidate-only computation.
    pool = [{k: c[k] for k in ("candidate_id", "agents", "actual_size", "selection_families")}
            for c in root["control_event"]["pool"]]
    before = json_fingerprint(root)
    row = materialize(root["state"], pool, entry["anchor_id"], groups)
    require(json_fingerprint(root) == before, "source root mutated")
    return row | {k: entry[k] for k in ("state_id", "map_id", "decision")}


def summarize(rows):
    attempts = [a for r in rows for a in r["attempts"]]
    changed = [a for a in attempts if a["status"] == "different_members"]
    return dict(roots=len(rows), maps=len({r["map_id"] for r in rows}),
                primary_candidates_reproduced=sum(len(r["primary"]) for r in rows),
                structural_anchor_roots=sum(r["structural_anchor"] for r in rows),
                attempts=len(attempts), statuses=dict(sorted(Counter(a["status"] for a in attempts).items())),
                roots_with_different_members=sum(any(a["status"] == "different_members" for a in r["attempts"]) for r in rows),
                unique_root_alternates=sum(len({a["candidate_id"] for a in r["attempts"] if a["status"] == "different_members"}) for r in rows),
                new_to_pool_attempts=sum(not a["in_original_pool"] for a in changed),
                already_in_pool_attempts=sum(a["in_original_pool"] for a in changed),
                labeled_with_anchor_attempts=sum(a["labeled_with_anchor"] for a in changed))


def run(verify=False, workers=20):
    plan = verify_plan()
    index = read_json(ROOT / contract.SOURCE / "training_index.json")
    jobs = [(r, plan["inputs"][r["source_root"]],
             [[c["candidate_id"] for c in g["candidates"]] for g in index["groups"] if g["state_id"] == r["state_id"]])
            for r in plan["roots"]]
    with _CollectionRunLock(OUT, plan["binding"], "structural-focus-materialization"):
        rows = []
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for number, future in enumerate(as_completed([pool.submit(job, data) for data in jobs]), 1):
                row = future.result()
                rows.append(row)
                print("FOCUS", number, "/", len(jobs), row["state_id"], flush=True)
        rows.sort(key=lambda r: (r["map_id"], r["state_id"]))
        verify_plan()
        report = dict(schema="lns2.sa.structural_focus_report.v1", binding=plan["binding"], summary=summarize(rows), rows=rows,
                      by_map={m: summarize([r for r in rows if r["map_id"] == m]) for m in sorted({r["map_id"] for r in rows})},
                      solver_calls=0, model_fits=0, new_labels=0, runtime_modified=False,
                      performance_evaluated=False, training_ready=False, automatic_collection=False)
        if verify:
            require(read_json(OUT / "report.json") == report, "report does not reproduce")
        else:
            once(OUT / "report.json", report)
        print(report["summary"], flush=True)
        print("REPORT_SHA256", sha256_file(OUT / "report.json"), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "materialize", "verify"))
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()
    require(1 <= args.workers <= 20, "workers must be 1..20")
    if args.stage == "prepare":
        print(prepare()["binding"])
    else:
        run(args.stage == "verify", args.workers)


if __name__ == "__main__":
    main()
