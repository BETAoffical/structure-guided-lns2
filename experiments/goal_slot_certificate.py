"""Exact finite-time forward/backward feasibility with an unrestricted static tail."""
from collections import deque

from experiments import local_path_search as ref


def goal_slot(graph,start,goal,fixed,tick):
    horizon=max([tick+1,1]+[len(p)+1 for p in fixed.values()])
    occupancy=[{ref.at(p,t) for p in fixed.values()} for t in range(horizon+1)]
    swaps=[{(ref.at(p,t),ref.at(p,t-1)) for p in fixed.values() if t and ref.at(p,t)!=ref.at(p,t-1)}
           for t in range(horizon+1)]
    if start in occupancy[0] or goal in occupancy[-1]:
        return dict(individually_feasible=False)
    component={goal}
    queue=deque([goal])
    while queue:
        u=queue.popleft()
        for v in graph[u]:
            if v not in occupancy[-1] and v not in component:
                component.add(v)
                queue.append(v)
    forward={start}
    for t in range(1,tick+1):
        forward={v for u in forward for v in graph[u] if v not in occupancy[t] and (u,v) not in swaps[t]}
    backward=component
    for t in range(horizon-1,tick-1,-1):
        backward={u for u in graph if u not in occupancy[t] and any(
                    v in backward and (u,v) not in swaps[t+1] for v in graph[u])}
    viable=sorted(forward&backward)
    boundary={v for u in component for v in graph[u] if v not in component}
    owners=sorted(a for a,p in fixed.items() if p[-1] in boundary)
    return dict(individually_feasible=bool(viable),horizon=horizon,forward_count=len(forward),
                backward_count=len(backward),viable_at_tick=viable,terminal_component=sorted(component),
                terminal_boundary=sorted(boundary),boundary_owners=owners)


def prove_goal_slot(state,selected,pair,tick):
    graph,agents=ref.validate_state(state)
    if len(set(pair))!=2 or not set(pair)<=set(selected)<=agents.keys() or tick<0:
        raise ValueError("invalid goal-slot problem")
    fixed={a:v["path"] for a,v in agents.items() if a not in selected}
    rows=[goal_slot(graph,agents[a]["start"],agents[a]["goal"],fixed,tick) for a in pair]
    forced=(all(r["individually_feasible"] and len(r["viable_at_tick"])==1 for r in rows)
            and rows[0]["viable_at_tick"]==rows[1]["viable_at_tick"])
    return dict(pair=pair,time=tick,forced_collision=forced,agents=rows,
                blockers=sorted({a for r in rows for a in r.get("boundary_owners",[])}))
