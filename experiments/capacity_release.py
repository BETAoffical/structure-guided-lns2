"""Input-only outsider proposals and necessary-condition relaxation checks."""
from collections import Counter, defaultdict
from itertools import combinations

from experiments import local_path_search as ref
from experiments.temporal_slot_scan import viable_layers
from experiments.goal_slot_certificate import goal_slot
from experiments.joint_slot_capacity import hall_witness, scan_capacity
from lns2_selector.runtime.fingerprints import semantic_fingerprint


def domain_at_tick(state, selected, witness, tick, released=()):
    graph, agents = ref.validate_state(state)
    selected, released = set(selected), set(released)
    if not selected <= agents.keys() or not set(witness) <= selected or not released <= agents.keys()-selected:
        raise ValueError("invalid release membership")
    fixed = {a: v["path"] for a, v in agents.items() if a not in selected | released}
    return {a: set(goal_slot(graph, agents[a]["start"], agents[a]["goal"], fixed, tick)
                   .get("viable_at_tick", [])) for a in witness}


def propose(state, selected, witness, tick, case_id, limit=8):
    graph, agents = ref.validate_state(state)
    selected = set(selected)
    if not selected <= agents.keys() or not set(witness) <= selected or not witness:
        raise ValueError("invalid witness")
    fixed = {a: v["path"] for a, v in agents.items() if a not in selected}
    horizon = max([tick+1, 1]+[len(p)+1 for p in fixed.values()])
    if horizon > 512 or horizon*len(graph)*len(witness) > 10000000:
        return dict(status="resource_unknown", candidates=[], scores=[])
    owners, edge_owners = [], []
    for t in range(horizon+1):
        vertices, edges = defaultdict(set), defaultdict(set)
        for aid, path in fixed.items():
            v, u = ref.at(path, t), ref.at(path, max(0, t-1))
            vertices[v].add(aid)
            if t and u != v:
                edges[(v, u)].add(aid)
        owners.append(vertices)
        edge_owners.append(edges)
    occupancy = [set(row) for row in owners]
    swaps = [set(row) for row in edge_owners]
    near, total = Counter(), Counter()
    for aid in sorted(witness):
        layers, _ = viable_layers(graph, agents[aid]["start"], agents[aid]["goal"], occupancy, swaps)
        for t in range(1, horizon+1):
            for u in layers[t-1]:
                for v in graph[u]:
                    blockers = owners[t].get(v, set()) | edge_owners[t].get((u, v), set())
                    for blocker in blockers:
                        total[blocker] += 1
                        if abs(t-tick) <= 8:
                            near[blocker] += 1
    ranked = sorted(total, key=lambda a: (-near[a], -total[a], a))[:limit]
    control = sorted(set(fixed)-set(ranked), key=lambda a: semantic_fingerprint([case_id, 20260916, a]))[:len(ranked)]
    if len(control) != len(ranked):
        return dict(status="insufficient_controls", candidates=[], scores=[])
    candidates = [dict(id="baseline", members=[], family="baseline")]
    for label, members in (("directed", ranked), ("random", control)):
        candidates += [dict(id=f"{label}-single-{i}", members=[a], family=label+"_single") for i, a in enumerate(members)]
        candidates += [dict(id=f"{label}-pair-{i}", members=sorted(pair), family=label+"_pair")
                       for i, pair in enumerate(combinations(members[:4], 2))]
    return dict(status="ok", candidates=candidates,
                scores=[dict(agent=a, near=near[a], total=total[a]) for a in ranked])


def evaluate(case, proof, candidate):
    selected, state = case["selected"], case["state"]
    members, tick = proof["witness"]["agents"], proof["time"]
    old = domain_at_tick(state, selected, members, tick)
    domains = domain_at_tick(state, selected, members, tick, candidate["members"])
    if any(not old[a] <= domains[a] for a in members):
        raise ValueError("removing constraints shrank a viable domain")
    deficit = hall_witness(domains)
    if hall_witness(old) is None:
        raise ValueError("source capacity proof not reproduced")
    augmented = (scan_capacity(state, sorted(set(selected) | set(candidate["members"])))
                 if deficit is None else dict(status="not_checked_original_deficit"))
    return dict(candidate=candidate, original_certificate_cleared=deficit is None,
                relaxed_witness=deficit, domains={str(a): sorted(v) for a, v in domains.items()},
                augmented=augmented, no_feasibility_claim=True)
