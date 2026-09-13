from collections import deque
import random

import pytest

from experiments import local_path_search as ref
from experiments.pair_compatibility import astar_path, diagnose, restricted_state
from experiments.state_analysis import reconstruct_conflicts
from scripts import diagnose_sa_pair_compatibility as runner


def state(paths, ids=None, rows=2, cols=3):
    ids = list(range(len(paths))) if ids is None else ids
    agents = [dict(id=i, path=p, start=p[0], goal=p[-1]) for i,p in zip(ids,paths)]
    edges = sorted({(e.left,e.right) for e in reconstruct_conflicts(agents)})
    return dict(rows=rows,cols=cols,obstacles=[False]*(rows*cols),agents=agents,
                conflict_edges=[list(e) for e in edges],num_of_colliding_pairs=len(edges))


def independent_cost(graph, start, goal, fixed, constraints, cap):
    constraints = set(constraints)
    h = max([0] + [len(p) for p in fixed] + [t+1 for _,t,_,_ in constraints])
    bound = min(cap, h+len(graph)) if cap is not None else h+len(graph)
    if any(p[0] == start or p[-1] == goal for p in fixed) or ("vertex",0,start,start) in constraints:
        return None
    queue, seen = deque([(start,0)]), {(start,0)}
    while queue:
        u,t = queue.popleft()
        if (u == goal and all(ref.at(p,q) != goal for p in fixed for q in range(t,h+1))
                and not any(q >= t and k == "vertex" and v == goal or
                            q > t and k == "edge" and a == v == goal for k,q,a,v in constraints)):
            return t
        if t == bound:
            continue
        for v in graph[u]:
            nt = t+1
            if ((v,nt) in seen or ("vertex",nt,v,v) in constraints or ("edge",nt,u,v) in constraints
                    or any(ref.at(p,nt) == v or (ref.at(p,t) == v and ref.at(p,nt) == u) for p in fixed)):
                continue
            seen.add((v,nt))
            queue.append((v,nt))
    return None


def test_astar_matches_bfs_and_independent_dynamic_grid():
    graph = ref.graph_of(state([[0]]))
    rng = random.Random(20260914)
    for _ in range(400):
        fixed = []
        for _ in range(rng.randrange(3)):
            path = [rng.randrange(6)]
            for _ in range(rng.randrange(5)):
                path.append(rng.choice(graph[path[-1]]))
            fixed.append(path)
        start,goal = rng.sample(range(6),2)
        constraints = []
        for _ in range(rng.randrange(3)):
            t,u = rng.randrange(6),rng.randrange(6)
            constraints.append(("vertex",t,u,u) if rng.randrange(2) else ("edge",t,u,rng.choice(graph[u])))
        cap = rng.choice([None,None,3,7])
        expected = independent_cost(graph,start,goal,fixed,constraints,cap)
        for search in (ref.low_level,astar_path):
            result = search(graph,start,goal,fixed,ref.Budget(seconds=5,max_expanded=10000),constraints,cap)
            assert result.get("cost") == expected, (search.__name__,fixed,constraints,start,goal,cap,result,expected)


def test_static_tail_real_swap_and_wait():
    graph = ref.graph_of(state([[0]],rows=2,cols=2))
    result = astar_path(graph,1,0,[[0,1]],ref.Budget())
    assert result["cost"] == 3 and result["path"] == [1,3,2,0]


def test_relaxation_drops_only_other_selected_agents():
    s = state([[0,1,2],[2,1,0],[3],[5]],ids=[10,20,30,90])
    relaxed = restricted_state(s,[10,20,30],[10,20],True)
    assert [a["id"] for a in relaxed["agents"]] == [10,20,90]
    assert len(s["agents"]) == 4
    assert len(restricted_state(s,[10,20,30],[10,20],False)["agents"]) == 4
    with pytest.raises(ValueError): restricted_state(s,[10],[10,20],True)


@pytest.mark.parametrize("backend",["bfs","astar"])
def test_pair_witness_is_valid_and_helper_restored(backend):
    s = state([[0,1,2],[2,1,0]],ids=[10,20])
    original = ref.low_level
    result = diagnose(s,[10,20],[10,20],True,backend,5.,10000)
    assert result["status"] == "feasible"
    assert result["proves_full_state_feasible"]
    assert not result["proves_selected_insufficient"]
    assert ref.low_level is original


def test_budget_unknown_is_not_infeasibility():
    s = state([[0,1,2],[2,1,0]])
    original = ref.low_level
    result = diagnose(s,[0,1],[0,1],True,"astar",0.,10000)
    assert result["status"] == "unknown" and not result["proves_selected_insufficient"]
    assert ref.low_level is original


def test_infeasible_relaxation_is_only_necessary_certificate():
    s = state([[0,1,2],[2,1,0],[1]],ids=[10,20,90],rows=1,cols=3)
    result = diagnose(s,[10,20],[10,20],True,"astar",5.,10000)
    assert result["status"] == "infeasible" and result["proves_selected_insufficient"]
    fixed = diagnose(s,[10,20],[10,20],False,"astar",5.,10000)
    assert not fixed["proves_selected_insufficient"]


def test_resume_rejects_missing_and_changed_results(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,"OUT",tmp_path)
    job = dict(id="one")
    plan = dict(binding="identity",jobs=[job])
    runner.io.write(tmp_path/"bfs/manifest.json",dict(binding="identity",files={}))
    with pytest.raises(ValueError,match="incomplete manifest"): runner.read_stage(plan,"bfs")
    path=tmp_path/"bfs/jobs/one.json"
    runner.io.write(path,dict(status="ok",binding="identity",job=job,backend="bfs",result={}))
    runner.io.write(tmp_path/"bfs/manifest.json",dict(binding="identity",files={"one":runner.io.sha256_file(path)}))
    assert runner.read_stage(plan,"bfs")["one"]["status"] == "ok"
    runner.io.write(path,{"tampered":True})
    with pytest.raises(ValueError,match="result changed"): runner.read_stage(plan,"bfs")


def test_astar_gate_does_not_retry_proved_cases(monkeypatch):
    jobs=[dict(id=str(i)) for i in range(3)]
    monkeypatch.setattr(runner,"read_stage",lambda *a:{str(i):dict(result=dict(status=s))
                        for i,s in enumerate(["feasible","unknown","infeasible"])})
    assert runner.jobs_for(dict(jobs=jobs),"astar") == [jobs[1]]
