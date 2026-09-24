"""One five-agent hard-constraint reference, conditional on failed PP augmentation."""
import argparse
import heapq
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import probe_sa_terminal_augmentation as parent
from experiments import local_path_search as ref
from experiments.pair_compatibility import astar_path

run, q = parent.run, parent.q
OUT = ROOT / "build/sa-terminal-augmentation-reference-v1"
CODE = ("scripts/check_sa_terminal_joint_reference.py", "tests/evaluation/test_sa_terminal_joint_reference.py",
        "docs/SA_TERMINAL_JOINT_REFERENCE_PROTOCOL_ZH.md")


def solve(state, selected, budget):
    """CBS over all selected agents; fixed outsiders are hard constraints throughout."""
    graph, agents = ref.validate_state(state)
    selected = sorted(selected)
    run.require(1 <= len(selected) <= 5 and len(set(selected)) == len(selected) and
                set(selected) <= agents.keys(), "one to five unique known agents")
    external = [a["path"] for i,a in agents.items() if i not in selected]
    nodes, serial, high, initial_searches = [], 0, 0, []
    try:
        paths = []
        for aid in selected:
            a = agents[aid]
            found = astar_path(graph,a["start"],a["goal"],external,budget)
            initial_searches.append(dict(agent=aid, status=found["status"], expanded=found["expanded"],
                                         reason=found.get("reason"), cost=found.get("cost")))
            if found["status"] != "feasible":
                return dict(status="infeasible", reason="single_agent_fixed_outside_exhausted", agent=aid,
                            expanded=budget.expanded, high_expanded=0, initial_searches=initial_searches)
            paths.append(found["path"])
        constraints = tuple(() for _ in selected)
        seen = {constraints}
        heapq.heappush(nodes,(sum(len(p)-1 for p in paths),serial,constraints,paths))
        while nodes:
            budget.check()
            _,_,constraints,paths = heapq.heappop(nodes)
            high += 1
            events = ref.reconstruct_conflicts([dict(id=aid,path=p) for aid,p in zip(selected,paths)])
            if not events:
                replacements = {str(aid):p for aid,p in zip(selected,paths)}
                checked = ref.validate_witness(state,replacements)
                return dict(status="feasible",paths=replacements,witness_validation=checked,
                            expanded=budget.expanded,high_expanded=high,initial_searches=initial_searches)
            event = min(events,key=lambda e:(e.time,e.left,e.right,e.kind,e.cells))
            for aid in (event.left,event.right):
                index = selected.index(aid)
                branch = list(constraints)
                branch[index] = tuple(sorted(set(branch[index]) | {ref.constraint_for(event,paths[index])}))
                key = tuple(branch)
                if key in seen: continue
                seen.add(key)
                a = agents[aid]
                found = astar_path(graph,a["start"],a["goal"],external,budget,key[index])
                if found["status"] != "feasible": continue
                changed = list(paths)
                changed[index] = found["path"]
                serial += 1
                heapq.heappush(nodes,(sum(len(p)-1 for p in changed),serial,key,changed))
        return dict(status="infeasible",reason="cbs_open_exhausted",expanded=budget.expanded,
                    high_expanded=high,initial_searches=initial_searches)
    except ref.LimitReached as error:
        return dict(status="unknown",reason=str(error),expanded=budget.expanded,high_expanded=high,
                    initial_searches=initial_searches,open_nodes=len(nodes))


def prepare():
    reg,out = parent.verify()
    report = run.check_seal(run.read_json(out/"report.json"))
    run.require(report["binding"] == reg["binding"] and report["successes"] == dict(original=0,targeted=0,random=0),
                "conditional failed augmentation only")
    target = min(reg["targets"],key=lambda t:t["decision"])
    selected = next(c["agents"] for c in reg["candidates"][target["id"]] if c["arm"] == "targeted")
    run.require(len(selected) == 5 and target["decision"] == 241, "one fixed earliest five-agent set")
    state_path = (out/"roots"/(target["id"]+".json")).relative_to(ROOT).as_posix()
    inputs = dict(reg["inputs"])
    commit = run.subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    for path,content in parent.identity.source_python(commit).items():
        run.require(parent.identity.same_source((ROOT/path).read_bytes(),content), "uncommitted Python: "+path)
        inputs[path] = run.sha256_file(ROOT/path)
    for name in (*CODE,state_path,(out/"report.json").relative_to(ROOT).as_posix(),
                 (out/"registration.json").relative_to(ROOT).as_posix(),
                 (out/"collection.complete.json").relative_to(ROOT).as_posix()):
        inputs[name] = run.sha256_file(ROOT/name)
    body = dict(schema="lns2.sa.terminal_joint_reference.v1",source_commit=commit,source_binding=reg["binding"],
                inputs=inputs,state_path=state_path,root_fingerprint=target["fingerprint"],selected=selected,
                seconds=180.,nodes=1000000,fuse_seconds=210.,jobs=1,workers=1,
                no_ttf=True,no_training=True,automatic_expansion=False,posthoc_diagnostic=True)
    body["binding"] = run.json_fingerprint(body)
    run.require(not OUT.exists(), "preserve reference output")
    run.once(OUT/"registration.json",run.sealed(body))
    return dict(jobs=1,selected=selected,seconds=180.,nodes=1000000,scope="fixed outsiders; all five agents jointly")


def verify():
    reg = run.check_seal(run.read_json(OUT/"registration.json"))
    run.require(reg["binding"] == run.json_fingerprint({k:v for k,v in reg.items() if k not in {"binding","integrity"}}),"binding")
    parent.identity.verify_files(reg["inputs"])
    state = run.read_json(ROOT/reg["state_path"])
    run.require(q.state_fingerprint(state) == reg["root_fingerprint"],"reference root")
    return reg,state


def worker(job):
    parent.source.die_with_parent(job["parent_pid"])
    reg,state = job["registration"],job["state"]
    result = solve(state,reg["selected"],ref.Budget(seconds=reg["seconds"],max_expanded=reg["nodes"]))
    return dict(job_id="only-reference",status="ok",binding=reg["binding"],result=result)


def collect():
    from experiments.repair_collection import _run_jobs
    reg,state = verify()
    with parent.budget.recovery.strict_lock(OUT,reg["binding"],"five-agent-reference"):
        run.require(not (OUT/"run_status.json").exists(),"single reference attempt; no automatic retry")
        run.write_json(OUT/"run_status.json",dict(status="running",binding=reg["binding"]))
        try:
            rows = _run_jobs(worker,[dict(job_id="only-reference",registration=reg,state=state,parent_pid=os.getpid())],1,
                phase="five-agent-reference",output_root=OUT/"progress",run_fingerprint=reg["binding"],
                timeout_seconds=reg["fuse_seconds"],
                failure_result=lambda j,status,error:dict(job_id=j["job_id"],status=status,error=error))
            run.once(OUT/"attempt.json",run.sealed(dict(binding=reg["binding"],results=rows)))
            run.require(len(rows) == 1,"one reference job")
            row = rows[0]
            if row["status"] == "timeout":
                result = dict(status="unknown",reason="external_fuse")
            else:
                run.require(row["status"] == "ok" and row["binding"] == reg["binding"],"reference error requires inspection")
                result = row["result"]
            if result["status"] == "feasible":
                run.require(set(map(int,result["paths"])) == set(reg["selected"]),"all five paths")
                run.require(ref.validate_witness(state,result["paths"])["globally_feasible"],"full task witness")
            parent.identity.verify_files(reg["inputs"])
            report = dict(binding=reg["binding"],result=result,no_ttf=True,no_training=True,
                          native_pp_unchanged=True,selected=reg["selected"],automatic_promotion=False)
            run.once(OUT/"report.json",run.sealed(report))
            run.write_json(OUT/"run_status.json",dict(status="complete",report_sha256=run.sha256_file(OUT/"report.json")))
        except BaseException:
            run.write_json(OUT/"run_status.json",dict(status="needs_inspection",binding=reg["binding"]))
            raise
    return {k:v for k,v in result.items() if k not in {"paths"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("prepare","collect","verify"))
    args = parser.parse_args()
    result = prepare() if args.phase == "prepare" else collect() if args.phase == "collect" else dict(binding=verify()[0]["binding"])
    print(run.json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__": main()
