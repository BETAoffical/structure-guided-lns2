from experiments.goal_slot_certificate import goal_slot
from experiments.local_path_search import at
import random


def test_static_tail_allows_unbounded_late_arrival():
    graph={0:[0,1],1:[0,1,2],2:[1,2]}
    result=goal_slot(graph,0,2,{},0)
    assert result["viable_at_tick"]==[0]
    assert result["terminal_component"]==[0,1,2]


def test_temporary_goal_occupancy_requires_future_departure():
    graph={0:[0,1],1:[0,1,2],2:[1,2]}
    result=goal_slot(graph,0,1,{10:[1,2]},1)
    assert result["viable_at_tick"]==[0,1]
    assert result["terminal_component"]==[0,1]
    assert result["boundary_owners"]==[10]


def test_permanent_occupied_goal_has_no_witness():
    assert not goal_slot({0:[0,1],1:[0,1]},0,1,{9:[1]},0)["individually_feasible"]


def test_goal_conditioned_cells_match_uncompressed_path_enumeration():
    rng=random.Random(20260914)
    graph={i:sorted({i,max(0,i-1),min(5,i+1)}) for i in range(6)}
    for _ in range(150):
        fixed={}
        for aid in (8,13):
            path=[rng.randrange(6)]
            for _ in range(rng.randrange(5)):
                path.append(rng.choice(graph[path[-1]]))
            fixed[aid]=path
        start,goal,tick=rng.randrange(6),rng.randrange(6),rng.randrange(5)
        result=goal_slot(graph,start,goal,fixed,tick)
        horizon=max(tick,max(len(p) for p in fixed.values()))+len(graph)+2
        states={(start,start if tick==0 else None)} if all(at(p,0)!=start for p in fixed.values()) else set()
        for t in range(1,horizon+1):
            states={(v,v if t==tick else saved) for u,saved in states for v in graph[u]
                    if all(v!=at(p,t) and not (u==at(p,t) and v==at(p,t-1)) for p in fixed.values())}
        expected={saved for cell,saved in states if cell==goal}
        assert set(result.get("viable_at_tick",[]))==expected
