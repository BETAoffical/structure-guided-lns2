"""Bounded reference searches, deliberately separate from InitLNS soft-conflict PP."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import heapq
import random
import time

from experiments.state_analysis import reconstruct_conflicts


class LimitReached(Exception):
    pass


@dataclass
class Budget:
    seconds: float = 180.0
    max_expanded: int = 1_000_000
    expanded: int = 0
    started: float = field(default_factory=time.monotonic)

    def check(self):
        if time.monotonic() - self.started >= self.seconds:
            raise LimitReached("wall_budget")
        if self.expanded >= self.max_expanded:
            raise LimitReached("node_budget")

    def consume(self):
        self.check()
        self.expanded += 1


def graph_of(state):
    rows, cols, blocked = state['rows'], state['cols'], state['obstacles']
    if rows <= 0 or cols <= 0 or len(blocked) != rows * cols:
        raise ValueError('invalid obstacle grid')
    graph = {}
    for cell, obstacle in enumerate(blocked):
        if obstacle:
            continue
        row, col = divmod(cell, cols)
        neighbors = [cell]
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = row + dr, col + dc
            if 0 <= r < rows and 0 <= c < cols and not blocked[r * cols + c]:
                neighbors.append(r * cols + c)
        graph[cell] = sorted(neighbors)
    return graph


def validate_state(state):
    graph = graph_of(state)
    agents = {a['id']: a for a in state['agents']}
    if len(agents) != len(state['agents']):
        raise ValueError('duplicate agent ID')
    for a in agents.values():
        path = a['path']
        if not path or path[0] != a['start'] or path[-1] != a['goal']:
            raise ValueError('path endpoints')
        if any(p not in graph for p in path):
            raise ValueError('path on obstacle or out of bounds')
        if any(v not in graph[u] for u, v in zip(path, path[1:])):
            raise ValueError('nonadjacent path step')
    edges = {(e.left, e.right) for e in reconstruct_conflicts(state['agents'])}
    if edges != {tuple(sorted(e)) for e in state['conflict_edges']}:
        raise ValueError('conflict edge mismatch')
    if len(edges) != state['num_of_colliding_pairs']:
        raise ValueError('conflict count mismatch')
    return graph, agents


def at(path, tick):
    return path[min(tick, len(path) - 1)]


def select_pairs(state, count=2):
    frequencies = Counter((e.left, e.right) for e in reconstruct_conflicts(state['agents']))
    return [list(pair) for pair in sorted(frequencies, key=lambda p: (-frequencies[p], p))[:count]]


def low_level(graph, start, goal, fixed, budget, constraints=(), max_cost=None, tie_seed=None):
    """BFS with a complete static tail, hard external paths and vertex/edge constraints.

    A constraint is (kind, arrival_tick, source, destination). Vertex uses source=destination.
    No arbitrary time horizon is used to conclude infeasibility.
    """
    budget.check()
    before = budget.expanded
    forbidden = set(map(tuple, constraints))
    if any(t < 0 or kind not in ('vertex', 'edge') for kind, t, _, _ in forbidden):
        raise ValueError('invalid constraint')
    horizon = max([0] + [len(p)-1 for p in fixed] + [t+1 for _, t, _, _ in forbidden])
    occupancy, swaps = [], []
    for tick in range(horizon+1):
        budget.check()
        occupancy.append({at(p, tick) for p in fixed})
        swaps.append({(at(p, tick), at(p, tick-1)) for p in fixed
                      if tick and at(p, tick) != at(p, tick-1)})
    last_goal = max([t for t, cells in enumerate(occupancy) if goal in cells]
                    + [t for kind, t, _, v in forbidden if kind == 'vertex' and v == goal] + [-1])
    failure = dict(status='infeasible', expanded=0, horizon=horizon, last_goal_occupancy=last_goal)
    if start not in graph or goal not in graph or start in occupancy[0]:
        return dict(failure, reason='blocked_start')
    if ('vertex', 0, start, start) in forbidden or any(p[-1] == goal for p in fixed):
        return dict(failure, reason='permanent_goal_or_root_constraint')
    root = (0, start)
    parents, queue = {root: None}, deque([(root, 0)])
    rng = random.Random(tie_seed) if tie_seed is not None else None
    first_goal = None
    while queue:
        budget.consume()
        key, tick = queue.popleft()
        cell = key[1]
        if cell == goal and first_goal is None:
            first_goal = tick
        future_wait_forbidden = any(kind == 'edge' and u == v == goal and t > tick
                                    for kind, t, u, v in forbidden)
        if cell == goal and tick > last_goal and not future_wait_forbidden:
            path = []
            while key is not None:
                path.append(key[1])
                key = parents[key]
            return dict(status='feasible', path=path[::-1], cost=tick,
                        expanded=budget.expanded-before, first_goal_visit=first_goal)
        if max_cost is not None and tick >= max_cost:
            continue
        layer = min(tick+1, horizon)
        neighbors = graph[cell]
        if rng is not None:
            neighbors = list(neighbors)
            rng.shuffle(neighbors)
        for dest in neighbors:
            nxt = (layer, dest)
            if (dest in occupancy[layer] or (cell, dest) in swaps[layer]
                    or ('vertex', tick+1, dest, dest) in forbidden
                    or ('edge', tick+1, cell, dest) in forbidden):
                continue
            if nxt not in parents:
                parents[nxt] = key
                queue.append((nxt, tick+1))
    return dict(failure, expanded=budget.expanded-before,
                reason='cost_bounded_exhausted' if max_cost is not None else 'finite_graph_exhausted',
                first_goal_visit=first_goal)


def pair_events(first, second):
    return reconstruct_conflicts([dict(id=0, path=first), dict(id=1, path=second)])


def constraint_for(event, path):
    t = event.time
    if event.kind == 'vertex':
        return ('vertex', t, at(path, t), at(path, t))
    return ('edge', t, at(path, t-1), at(path, t))


def directed_constraints(first, relaxed_second, count=4):
    constraints = []
    for event in pair_events(first, relaxed_second):
        value = constraint_for(event, first)
        if value not in constraints:
            constraints.append(value)
        if len(constraints) == count:
            break
    return constraints


def sequential(graph, agents, external, order, budget, constraints=(), max_cost=None, tie_seed=None):
    first, second = (agents[i] for i in order)
    a = low_level(graph, first['start'], first['goal'], external, budget,
                  constraints, max_cost, tie_seed)
    if a['status'] != 'feasible':
        return dict(status='not_found', order=order, first=a, second=None)
    b = low_level(graph, second['start'], second['goal'], external + [a['path']], budget)
    result = dict(status='feasible' if b['status'] == 'feasible' else 'not_found',
                  order=order, first=a, second=b)
    if result['status'] == 'feasible':
        result['paths'] = {str(order[0]): a['path'], str(order[1]): b['path']}
    return result


def sequential_method(state, pair, method, budget, seed=20260909):
    graph, agents = validate_state(state)
    external = [a['path'] for aid, a in agents.items() if aid not in pair]
    results = []
    for order in (list(pair), list(reversed(pair))):
        record = dict(order=order, attempts=[], constraints=[], status='not_found')
        results.append(record)
        try:
            base = sequential(graph, agents, external, order, budget)
            record['attempts'].append(dict(kind='ordinary', result=base))
            if base['status'] == 'feasible':
                record.update(status='feasible', paths=base['paths'], recovery='ordinary')
                continue
            if method == 'ordinary' or base['first']['status'] != 'feasible':
                continue
            first_cost = base['first']['cost']
            if method == 'random_paths':
                attempts = [(None, seed+i) for i in range(4)]
            elif method == 'directed_paths':
                second = agents[order[1]]
                relaxed = low_level(graph, second['start'], second['goal'], external, budget)
                record['relaxed_second'] = relaxed
                if relaxed['status'] != 'feasible':
                    record['reason'] = 'second_infeasible_even_without_first'
                    continue
                constraints = directed_constraints(base['first']['path'], relaxed['path'])
                record['constraints'] = [list(c) for c in constraints]
                attempts = [(constraint, None) for constraint in constraints]
            else:
                raise ValueError(method)
            for constraint, tie_seed in attempts:
                attempt = sequential(graph, agents, external, order, budget,
                    [constraint] if constraint is not None else (), first_cost, tie_seed)
                record['attempts'].append(dict(kind=method, constraint=constraint,
                                               tie_seed=tie_seed, result=attempt))
                if attempt['status'] == 'feasible':
                    record.update(status='feasible', paths=attempt['paths'], recovery=method)
                    break
        except LimitReached as exc:
            record.update(status='unknown', reason=str(exc))
    return dict(status='feasible' if any(r['status'] == 'feasible' for r in results)
                else ('unknown' if any(r['status'] == 'unknown' for r in results) else 'not_found'),
                orders=results)


def cbs_pair(state, pair, budget):
    """Reference two-agent CBS. Budget exhaustion is never an infeasibility proof."""
    graph, agents = validate_state(state)
    external = [a['path'] for aid, a in agents.items() if aid not in pair]
    open_nodes, serial, visited, high_expanded = [], 0, set(), 0
    paths = []
    for aid in pair:
        found = low_level(graph, agents[aid]['start'], agents[aid]['goal'], external, budget)
        if found['status'] != 'feasible':
            return dict(status='infeasible', reason='single_agent_infeasible_without_partner', agent=aid)
        paths.append(found['path'])
    heapq.heappush(open_nodes, (sum(len(p)-1 for p in paths), serial, ((), ()), paths))
    while open_nodes:
        budget.check()
        _, _, constraints, paths = heapq.heappop(open_nodes)
        high_expanded += 1
        conflicts = pair_events(*paths)
        if not conflicts:
            return dict(status='feasible', high_level_expanded=high_expanded,
                        paths={str(aid): p for aid, p in zip(pair, paths)})
        event = conflicts[0]
        for index, aid in enumerate(pair):
            branch = list(constraints)
            branch[index] = tuple(sorted(set(branch[index]) | {constraint_for(event, paths[index])}))
            key = tuple(branch)
            if key in visited:
                continue
            visited.add(key)
            a = agents[aid]
            found = low_level(graph, a['start'], a['goal'], external, budget, branch[index])
            if found['status'] != 'feasible':
                continue
            child_paths = list(paths)
            child_paths[index] = found['path']
            serial += 1
            heapq.heappush(open_nodes, (sum(len(p)-1 for p in child_paths), serial, key, child_paths))
    return dict(status='infeasible', reason='cbs_open_exhausted', high_level_expanded=high_expanded)


def validate_witness(state, paths):
    graph, agents = validate_state(state)
    selected = {int(i) for i in paths}
    if not selected or not selected <= agents.keys():
        raise ValueError('unknown witness agent')
    replacement = [dict(a, path=paths.get(str(a['id']), a['path'])) for a in state['agents']]
    for a in replacement:
        p = a['path']
        if not p or p[0] != a['start'] or p[-1] != a['goal'] or any(v not in graph for v in p):
            raise ValueError('invalid witness endpoints or cells')
        if any(v not in graph[u] for u, v in zip(p, p[1:])):
            raise ValueError('invalid witness move')
    before = {(e.left, e.right) for e in reconstruct_conflicts(state['agents'])
              if e.left not in selected and e.right not in selected}
    after = {(e.left, e.right) for e in reconstruct_conflicts(replacement)}
    if after != before:
        raise ValueError('witness introduces or leaves a selected-agent conflict')
    return dict(external_conflict_pairs=[list(p) for p in sorted(before)],
                final_conflict_pairs=len(after), globally_feasible=not after,
                sum_of_costs=sum(len(a['path'])-1 for a in replacement))


def witness_paths(result):
    if 'paths' in result:
        yield result['paths']
    for order in result.get('orders', []):
        if 'paths' in order:
            yield order['paths']
