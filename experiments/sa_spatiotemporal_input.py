"""Lossless, pre-action path inputs for a separate SA representation study.

This is a data adapter, not NNS, a learned policy, or an online repairer.
Agent IDs are join keys; models must use membership masks, not numeric IDs.
"""
from collections import defaultdict
from itertools import groupby

from experiments.sa_paired_completion import require
from experiments.state_analysis import reconstruct_conflicts


SCHEMA = "lns2.sa.spatiotemporal_input.v1"


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, "invalid " + name)
    return value


def path_runs(path):
    """Exclusive interval ends; null denotes persistent final occupancy."""
    runs, time = [], 0
    for cell, values in groupby(path):
        end = time + sum(1 for _ in values)
        runs.append([cell, time, end])
        time = end
    runs[-1][2] = None
    return runs


def decode_paths(encoded):
    agents = []
    for a in encoded["agents"]:
        path = []
        for cell, start, end in a["occupancy"]:
            require(start == len(path), "noncontiguous occupancy")
            stop = a["path_cost"] + 1 if end is None else end
            require(stop > start, "empty occupancy interval")
            path.extend([cell] * (stop - start))
        require(len(path) == a["path_cost"] + 1, "path length mismatch")
        agents.append(dict(id=a["id"], start=a["start"], goal=a["goal"], path=path))
    return agents


def interval_events(encoded):
    """Reconstruct conflicts by interval overlap, independently of path scan."""
    occupancy, moves = defaultdict(list), defaultdict(list)
    stop = encoded["horizon"] + 1
    for a in encoded["agents"]:
        for cell, start, end in a["occupancy"]:
            occupancy[cell].append((start, stop if end is None else end, a["id"]))
        for before, after in zip(a["occupancy"], a["occupancy"][1:]):
            moves[(after[1], before[0], after[0])].append(a["id"])
    events = []
    for cell, spans in occupancy.items():
        active = []
        for start, end, aid in sorted(spans):
            active = [span for span in active if span[1] > start]
            for other_start, other_end, bid in active:
                if aid == bid:
                    continue
                left, right = sorted((aid, bid))
                for time in range(max(start, other_start), min(end, other_end)):
                    events.append([time, "vertex", left, right, [cell]])
            active.append((start, end, aid))
    for (time, previous, current), forward in moves.items():
        if previous >= current:
            continue
        for aid in forward:
            for bid in moves.get((time, current, previous), []):
                left, right = sorted((aid, bid))
                events.append([time, "edge", left, right, [previous, current]])
    return sorted(events, key=lambda e: tuple(e[:4]))


def encode_state(state):
    rows, cols = (integer(state[k], k, 1) for k in ("rows", "cols"))
    obstacles = state["obstacles"]
    require(len(obstacles) == rows * cols and
            all(type(v) is int and v in (0, 1) for v in obstacles), "invalid obstacle grid")
    agents, ids, starts, goals = [], set(), set(), set()
    require(state["agents"], "empty agent list")
    for source in state["agents"]:
        aid = integer(source["id"], "agent ID")
        require(aid not in ids, "duplicate agent ID")
        ids.add(aid)
        path = source["path"]
        require(path and all(type(c) is int and 0 <= c < rows * cols and not obstacles[c]
                             for c in path), "invalid path cells")
        start, goal = integer(source["start"], "start"), integer(source["goal"], "goal")
        require(path[0] == start and path[-1] == goal, "path endpoint mismatch")
        require(start not in starts and goal not in goals, "duplicate starts or goals")
        starts.add(start)
        goals.add(goal)
        for previous, current in zip(path, path[1:]):
            pr, pc = divmod(previous, cols)
            cr, cc = divmod(current, cols)
            require(abs(pr - cr) + abs(pc - cc) <= 1, "nonadjacent path step")
        cost = len(path) - 1
        if "path_cost" in source:
            require(type(source["path_cost"]) is int and source["path_cost"] == cost, "path cost mismatch")
        agents.append(dict(id=aid, start=start, goal=goal, path_cost=cost, occupancy=path_runs(path)))
    agents.sort(key=lambda a: a["id"])
    encoded = dict(schema=SCHEMA, rows=rows, cols=cols, obstacles=list(obstacles), agents=agents,
                   horizon=max(a["path_cost"] for a in agents), stay_at_goal=True)
    rebuilt = decode_paths(encoded)
    require({a["id"]: a["path"] for a in rebuilt} ==
            {a["id"]: a["path"] for a in state["agents"]}, "path roundtrip mismatch")
    expected = [[e.time, e.kind, e.left, e.right, list(e.cells)] for e in reconstruct_conflicts(state["agents"])]
    events = interval_events(encoded)
    require(events == expected, "interval/path conflict reconstruction mismatch")
    pairs = {tuple(e[2:4]) for e in events}
    edges = []
    for edge in state["conflict_edges"]:
        require(len(edge) == 2 and all(type(a) is int and a in ids for a in edge) and edge[0] != edge[1],
                "invalid conflict endpoints")
        edges.append(tuple(sorted(edge)))
    require(len(edges) == len(set(edges)) and set(edges) == pairs, "reported conflict edges disagree")
    require(integer(state["num_of_colliding_pairs"], "conflict count") == len(pairs), "conflict count mismatch")
    encoded["events"] = events
    return encoded


def candidate_input(encoded, selected):
    known = [a["id"] for a in encoded["agents"]]
    require(selected and all(type(a) is int and a in known for a in selected) and
            len(selected) == len(set(selected)), "invalid candidate membership")
    chosen = set(selected)
    return dict(selected_mask=[a in chosen for a in known])


def candidate_diagnostics(encoded, selected):
    """Descriptive input coverage, not learned features or feasibility proofs."""
    mask = candidate_input(encoded, selected)["selected_mask"]
    chosen = {a["id"] for a, flag in zip(encoded["agents"], mask) if flag}
    internal, boundary, external = [], [], []
    for e in encoded["events"]:
        n = int(e[2] in chosen) + int(e[3] in chosen)
        (external, boundary, internal)[n].append(e)
    tails = {a["id"]: a["occupancy"][-1][1] for a in encoded["agents"]}
    tail_boundary = [e for e in boundary if e[1] == "vertex" and
                     any(a not in chosen and e[0] >= tails[a] for a in e[2:4])]
    # A geometric one-cell halo is a descriptive exposure set, not a truncation:
    # every outsider's entire path remains in the model input.
    cells = {run[0] for a in encoded["agents"] if a["id"] in chosen for run in a["occupancy"]}
    halo = set(cells)
    for cell in cells:
        row, col = divmod(cell, encoded["cols"])
        for rr, cc in ((row-1,col), (row+1,col), (row,col-1), (row,col+1)):
            if 0 <= rr < encoded["rows"] and 0 <= cc < encoded["cols"]:
                other = rr * encoded["cols"] + cc
                if not encoded["obstacles"][other]:
                    halo.add(other)
    near = {a["id"] for a in encoded["agents"] if a["id"] not in chosen and
            any(run[0] in halo for run in a["occupancy"])}
    colliding_outsiders = {aid for e in boundary for aid in e[2:4] if aid not in chosen}
    pairs = {tuple(e[2:4]) for e in internal + boundary}
    return dict(size=len(chosen), internal_events=len(internal), boundary_events=len(boundary),
                external_events=len(external), outsider_goal_tail_events=len(tail_boundary),
                incident_pairs=len(pairs), incident_events=len(internal)+len(boundary),
                incident_times=sorted({e[0] for e in internal+boundary}),
                nearby_outsiders=len(near), currently_colliding_outsiders=len(colliding_outsiders),
                nearby_noncolliding_outsiders=len(near-colliding_outsiders))
