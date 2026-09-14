"""Necessary instantaneous capacity checks, not a complete MAPF solver."""
from collections import deque

from experiments import local_path_search as ref
from experiments.temporal_slot_scan import viable_layers


def hall_witness(domains):
    """Return a deficient left set from a maximum bipartite matching, or None."""
    owner={}
    ordered={a:sorted(cells) for a,cells in domains.items()}
    def augment(a,seen):
        for cell in ordered[a]:
            if cell in seen: continue
            seen.add(cell)
            if cell not in owner or augment(owner[cell],seen):
                owner[cell]=a
                return True
        return False
    for a in sorted(domains,key=lambda a:(len(domains[a]),a)):
        augment(a,set())
    missing=set(domains)-set(owner.values())
    if not missing: return None
    left=set(missing)
    right=set()
    queue=deque(sorted(missing))
    while queue:
        a=queue.popleft()
        for cell in ordered[a]:
            right.add(cell)
            if cell in owner and owner[cell] not in left:
                left.add(owner[cell])
                queue.append(owner[cell])
    if len(left)<=len(right): raise ValueError("invalid Hall deficiency")
    return dict(agents=sorted(left),cells=sorted(right),deficit=len(left)-len(right))


def scan_capacity(state,selected,max_horizon=512,max_work=10000000):
    graph,agents=ref.validate_state(state)
    selected=sorted(set(selected))
    if not selected or not set(selected)<=agents.keys(): raise ValueError("invalid joint members")
    fixed={a:v["path"] for a,v in agents.items() if a not in selected}
    horizon=max([1]+[len(p)+1 for p in fixed.values()])
    work=horizon*len(graph)*len(selected)
    if horizon>max_horizon or work>max_work:
        return dict(status="resource_unknown",horizon=horizon,work=work)
    occupancy=[{ref.at(p,t) for p in fixed.values()} for t in range(horizon+1)]
    swaps=[{(ref.at(p,t),ref.at(p,t-1)) for p in fixed.values() if t and ref.at(p,t)!=ref.at(p,t-1)} for t in range(horizon+1)]
    layers={a:viable_layers(graph,agents[a]["start"],agents[a]["goal"],occupancy,swaps)[0] for a in selected}
    for t in range(horizon+1):
        witness=hall_witness({a:layers[a][t] for a in selected})
        if witness is not None:
            return dict(status="proved",time=t,witness=witness,horizon=horizon,work=work)
    return dict(status="not_proved",horizon=horizon,work=work)
