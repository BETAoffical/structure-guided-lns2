"""Automatic necessary pair certificates with hard outsiders and a complete static tail."""
from collections import deque

from experiments import local_path_search as ref
from scripts.diagnose_sa_external_obstruction import reachable_prefix


def viable_layers(graph,start,goal,occupancy,swaps):
    horizon=len(occupancy)-1
    if start in occupancy[0] or goal in occupancy[-1]:
        return [set() for _ in occupancy],set()
    component={goal}
    queue=deque([goal])
    while queue:
        for v in graph[queue.popleft()]:
            if v not in occupancy[-1] and v not in component:
                component.add(v)
                queue.append(v)
    layers=[{start}]
    for t in range(1,horizon+1):
        layers.append({v for u in layers[-1] for v in graph[u]
                       if v not in occupancy[t] and (u,v) not in swaps[t]})
    backward=component
    layers[-1]&=backward
    for t in range(horizon-1,-1,-1):
        backward={u for u in graph if u not in occupancy[t]
                  and any(v in backward and (u,v) not in swaps[t+1] for v in graph[u])}
        layers[t]&=backward
    return layers,component


def forced_event(left,right):
    for t,(a,b) in enumerate(zip(left,right)):
        if len(a)==len(b)==1 and a==b:
            return dict(kind="vertex",time=t,cells=sorted(a))
        if t and all(len(x)==1 for x in (left[t-1],right[t-1],a,b)):
            if left[t-1]==b and right[t-1]==a and a!=b:
                return dict(kind="edge",time=t,cells=[next(iter(b)),next(iter(a))])
    return None


def scan(state,selected,max_pairs=2,max_horizon=512,max_work=10000000):
    graph,agents=ref.validate_state(state)
    selected=set(selected)
    if not selected or not selected<=agents.keys():
        raise ValueError("invalid selected agents")
    pairs=[p for p in ref.select_pairs(state,len(state["conflict_edges"])) if set(p)<=selected][:max_pairs]
    fixed={aid:a["path"] for aid,a in agents.items() if aid not in selected}
    horizon=max([1]+[len(p)+1 for p in fixed.values()])
    needed={aid for p in pairs for aid in p}
    work=horizon*len(graph)*len(needed)
    if horizon>max_horizon or work>max_work:
        return dict(status="resource_unknown",horizon=horizon,work=work,proofs=[],blockers=[])
    occupancy=[{ref.at(p,t) for p in fixed.values()} for t in range(horizon+1)]
    swaps=[{(ref.at(p,t),ref.at(p,t-1)) for p in fixed.values() if t and ref.at(p,t)!=ref.at(p,t-1)}
           for t in range(horizon+1)]
    cache={aid:viable_layers(graph,agents[aid]["start"],agents[aid]["goal"],occupancy,swaps) for aid in needed}
    proofs=[]
    for aid,(layers,component) in cache.items():
        if layers[0]: continue
        boundary={v for u in component for v in graph[u] if v not in component}
        owners={a for a,p in fixed.items() if p[-1] in boundary}
        prefix=reachable_prefix(graph,agents[aid]["start"],fixed,min(8,horizon))
        early={a for layer in prefix["blocked"] for b in layer for a in b["owners"]} if not prefix["layers"][-1] else set()
        proofs.append(dict(pair=[aid],event=dict(kind="individual_infeasible",time=None,cells=[]),
                           blocker_hints=sorted(early or owners),hint_kind="prefix_owners" if early else "terminal_boundary",
                           terminal_component_size=len(component)))
    for pair in pairs:
        left,right=(cache[a][0] for a in pair)
        if not left[0] or not right[0]: continue
        event=forced_event(left,right)
        if event is None:
            continue
        components=set().union(*(cache[a][1] for a in pair))
        boundary={v for u in components for v in graph[u] if v not in components}
        owners={a for a,p in fixed.items() if p[-1] in boundary}
        prefix_owners=set()
        if event["time"]<=8:
            for aid in pair:
                prefix=reachable_prefix(graph,agents[aid]["start"],fixed,event["time"])
                prefix_owners.update(a for layer in prefix["blocked"] for b in layer for a in b["owners"])
        hints=prefix_owners if prefix_owners else owners
        proofs.append(dict(pair=pair,event=event,blocker_hints=sorted(hints),
                           hint_kind="prefix_owners" if prefix_owners else "terminal_boundary",
                           terminal_component_size=len(components)))
    blockers=sorted({a for p in proofs for a in p["blocker_hints"]})
    return dict(status="proved" if proofs else "not_proved",horizon=horizon,work=work,
                pairs=pairs,proofs=proofs,blockers=blockers)
