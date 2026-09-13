"""Necessary pair relaxations; hard reference constraints, never production repair."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import heapq

from experiments import local_path_search as reference
from experiments.state_analysis import reconstruct_conflicts


def restricted_state(state, selected, pair, relax_selected):
    _, agents = reference.validate_state(state)
    selected, pair = set(selected), set(pair)
    if len(pair) != 2 or not pair <= selected <= agents.keys():
        raise ValueError("invalid pair or selected set")
    removed = selected - pair if relax_selected else set()
    result = deepcopy(state)
    result["agents"] = [a for a in result["agents"] if a["id"] not in removed]
    edges = sorted({(e.left, e.right) for e in reconstruct_conflicts(result["agents"])})
    result.update(conflict_edges=[list(e) for e in edges], num_of_colliding_pairs=len(edges))
    return result


def astar_path(graph, start, goal, fixed, budget, constraints=(), max_cost=None, tie_seed=None):
    """Shortest hard-constrained path with a finite stationary tail and spatial h."""
    if tie_seed is not None:
        raise ValueError("random tie mode is not part of this reference")
    budget.check()
    before = budget.expanded
    forbidden = set(map(tuple, constraints))
    if any(t < 0 or k not in ("vertex", "edge") for k, t, _, _ in forbidden):
        raise ValueError("invalid constraint")
    # One stationary layer after the final external move or explicit constraint.
    horizon = max([0] + [len(p) for p in fixed] + [t + 1 for _, t, _, _ in forbidden])
    occupancy, swaps = [], []
    for t in range(horizon + 1):
        budget.check()
        occupancy.append({reference.at(p, t) for p in fixed})
        swaps.append({(reference.at(p, t), reference.at(p, t - 1)) for p in fixed
                      if t and reference.at(p, t) != reference.at(p, t - 1)})
    last_goal = max([t for t, cells in enumerate(occupancy) if goal in cells]
                    + [t for k, t, _, v in forbidden if k == "vertex" and v == goal] + [-1])
    failure = dict(status="infeasible", expanded=0, horizon=horizon)
    if start not in graph or goal not in graph or start in occupancy[0]:
        return dict(failure, reason="blocked_start")
    if any(p[-1] == goal for p in fixed) or ("vertex", 0, start, start) in forbidden:
        return dict(failure, reason="permanent_goal_or_root_constraint")
    distance, queue = {goal: 0}, deque([goal])
    while queue:
        budget.check()
        cell = queue.popleft()
        for v in graph[cell]:
            if v not in distance:
                distance[v] = distance[cell] + 1
                queue.append(v)
    if start not in distance:
        return dict(failure, reason="static_disconnection")
    root = (0, start)
    parents, best = {root: None}, {root: 0}
    heap = [(distance[start], 0, root)]
    while heap:
        budget.consume()
        _, negative_g, key = heapq.heappop(heap)
        tick, cell = -negative_g, key[1]
        if tick != best[key]:
            continue
        if cell == goal and tick > last_goal and not any(
                k == "edge" and u == v == goal and t > tick for k, t, u, v in forbidden):
            path = []
            while key is not None:
                path.append(key[1])
                key = parents[key]
            return dict(status="feasible", path=path[::-1], cost=tick, expanded=budget.expanded-before)
        if max_cost is not None and tick >= max_cost:
            continue
        arrival, layer = tick + 1, min(tick + 1, horizon)
        for dest in graph[cell]:
            nxt = (layer, dest)
            if (dest not in distance or dest in occupancy[layer] or (cell, dest) in swaps[layer]
                    or ("vertex", arrival, dest, dest) in forbidden
                    or ("edge", arrival, cell, dest) in forbidden):
                continue
            if arrival < best.get(nxt, float("inf")):
                best[nxt], parents[nxt] = arrival, key
                heapq.heappush(heap, (arrival + distance[dest], -arrival, nxt))
    return dict(failure, reason="cost_bounded_exhausted" if max_cost is not None else "finite_graph_exhausted",
                expanded=budget.expanded-before)


def diagnose(state, selected, pair, relax_selected, backend, seconds, nodes):
    view = restricted_state(state, selected, pair, relax_selected)
    budget = reference.Budget(seconds=seconds, max_expanded=nodes)
    original = reference.low_level
    if backend not in ("bfs", "astar"):
        raise ValueError("unknown reference backend")
    # Each call runs in its own process. Restore the historical helper even on failure.
    try:
        if backend == "astar":
            reference.low_level = astar_path
        try:
            result = reference.cbs_pair(view, pair, budget)
        except reference.LimitReached as exc:
            result = dict(status="unknown", reason=str(exc))
    finally:
        reference.low_level = original
    if result["status"] == "feasible":
        result["witness_validation"] = reference.validate_witness(view, result["paths"])
        replacement = [dict(a, path=result["paths"].get(str(a["id"]), a["path"])) for a in state["agents"]]
        edges = sorted({(e.left, e.right) for e in reconstruct_conflicts(replacement)})
        result["full_state_conflicts_if_injected"] = [list(e) for e in edges]
        result["new_full_state_conflict_edges"] = [list(e) for e in edges if list(e) not in state["conflict_edges"]]
    result.update(expanded=budget.expanded, relax_selected=relax_selected, backend=backend,
                  proves_selected_insufficient=relax_selected and result["status"] == "infeasible",
                  proves_full_state_feasible=result["status"] == "feasible" and not result["full_state_conflicts_if_injected"])
    return result
