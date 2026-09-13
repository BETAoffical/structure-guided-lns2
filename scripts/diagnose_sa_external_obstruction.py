"""Independently expose an early forced collision before considering blocker changes."""
import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import diagnose_sa_pair_compatibility as pairs
from experiments.pair_compatibility import restricted_state, astar_path
from experiments import local_path_search as ref

io = pairs.io
OUT = ROOT / "build/sa-external-obstruction-v1"


def reachable_prefix(graph, start, fixed, ticks):
    """Uncompressed exact reachability; no goals, pair constraints or heuristics."""
    frontier = {start}
    if any(p[0]==start for p in fixed.values()):
        frontier=set()
    layers, blocked, transitions = [sorted(frontier)], [], []
    for t in range(1,ticks+1):
        vertices, edges = defaultdict(set), defaultdict(set)
        for aid,path in fixed.items():
            now,prev = ref.at(path,t),ref.at(path,t-1)
            vertices[now].add(aid)
            if now != prev:
                edges[(now,prev)].add(aid)
        nxt, stopped, allowed = set(),[],[]
        for u in sorted(frontier):
            for v in graph[u]:
                owners=vertices[v] | edges[(u,v)]
                if owners:
                    stopped.append(dict(time=t,source=u,destination=v,owners=sorted(owners),
                        vertex_owners=sorted(vertices[v]),swap_owners=sorted(edges[(u,v)])))
                else:
                    nxt.add(v)
                    allowed.append([u,v])
        layers.append(sorted(nxt))
        blocked.append(stopped)
        transitions.append(allowed)
        frontier=nxt
    return dict(layers=layers,blocked=blocked,transitions=transitions)


def expose():
    io.require(not (OUT/"report.json").exists(),"preserve prior certificate")
    plan=pairs.verify()
    outcomes={**pairs.read_stage(plan,"bfs"),**pairs.read_stage(plan,"astar")}
    proofs=[]
    for job in plan["jobs"]:
        if not outcomes[job["id"]]["result"]["proves_selected_insufficient"]:
            continue
        case=next(c for c in plan["cases"] if c["id"]==job["case_id"])
        view=restricted_state(case["state"],case["selected"],job["pair"],True)
        graph,agents=ref.validate_state(view)
        fixed={aid:a["path"] for aid,a in agents.items() if aid not in job["pair"]}
        budget=ref.Budget(seconds=45,max_expanded=250000)
        roots=[astar_path(graph,agents[aid]["start"],agents[aid]["goal"],list(fixed.values()),budget)
               for aid in job["pair"]]
        io.require(all(r["status"]=="feasible" for r in roots),"original paths already witness individual feasibility")
        event=ref.pair_events(roots[0]["path"],roots[1]["path"])[0]
        row=dict(job=job,root_paths=[r["path"] for r in roots],event=dict(time=event.time,kind=event.kind,cells=list(event.cells)))
        if event.time>8:
            row.update(status="no_early_prefix_certificate",blocker_candidates=[])
        else:
            prefixes=[reachable_prefix(graph,agents[aid]["start"],fixed,event.time) for aid in job["pair"]]
            forced_vertex=(event.kind=="vertex" and prefixes[0]["layers"][-1]==prefixes[1]["layers"][-1]
                           and len(prefixes[0]["layers"][-1])==1)
            forced_swap=(event.kind=="edge" and all(len(p["transitions"][-1])==1 for p in prefixes)
                         and prefixes[0]["transitions"][-1][0]==list(reversed(prefixes[1]["transitions"][-1][0])))
            blockers=sorted({aid for p in prefixes for layer in p["blocked"] for b in layer for aid in b["owners"]})
            row.update(status="forced_prefix_collision" if forced_vertex or forced_swap else "no_forced_prefix_certificate",
                       prefixes=prefixes,blocker_candidates=blockers)
        proofs.append(row)
    inputs={p.relative_to(ROOT).as_posix():io.sha256_file(p) for p in
            (Path(__file__),pairs.OUT/"plan.json",pairs.OUT/"report.json")}
    report=dict(schema="lns2.sa_external_obstruction.v1",inputs=inputs,proofs=proofs,no_ttf=True,
                candidate_rule="external owners of unavailable transitions in exact prefix, no frequency ranking")
    io.write(OUT/"report.json",report)
    print([{k:v for k,v in p.items() if k not in ("prefixes","root_paths")} for p in proofs])


if __name__=="__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    expose()
