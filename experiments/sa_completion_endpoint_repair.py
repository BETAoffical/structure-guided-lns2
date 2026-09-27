"""Experiment-only endpoint matching; never changes the frozen task sampler."""
from collections import Counter
from copy import deepcopy
import random

from generators.models import TaskData
from generators.task_flows import _candidate_pools, _bottleneck_candidates, _distance_map
from generators.validation import validate_task


METHOD = 'capacity_matching_v1'


def assign_endpoints(legal, constrained, count, quota, seed, attempts=16):
    """Match the constrained prefix, then match all remaining unique endpoints.

    Exhausting the bounded tie orders is unknown, not an infeasibility proof.
    """
    import numpy as np
    from scipy.sparse.csgraph import maximum_bipartite_matching

    if not 0 <= quota <= count or count > min(legal.shape):
        raise ValueError('invalid endpoint quota/count')
    if legal.shape != constrained.shape or (constrained - constrained.multiply(legal)).nnz:
        raise ValueError('constrained edges must be legal')
    rng = random.Random(seed)
    for attempt in range(attempts):
        rows, cols = list(range(legal.shape[0])), list(range(legal.shape[1]))
        rng.shuffle(rows)
        rng.shuffle(cols)
        match = maximum_bipartite_matching(constrained[rows][:, cols], perm_type='column')
        prefix = [(rows[i], cols[j]) for i, j in enumerate(match) if j >= 0]
        capacity = len(prefix)
        if capacity < quota:
            raise ValueError(f'bottleneck endpoint capacity {capacity} below quota {quota}')
        rng.shuffle(prefix)
        prefix = prefix[:quota]
        used_rows, used_cols = {a for a, _ in prefix}, {b for _, b in prefix}
        rr = [i for i in rows if i not in used_rows]
        cc = [i for i in cols if i not in used_cols]
        match = maximum_bipartite_matching(legal[rr][:, cc], perm_type='column')
        suffix = [(rr[i], cc[j]) for i, j in enumerate(match) if j >= 0]
        rng.shuffle(suffix)
        if len(suffix) >= count - quota:
            return prefix + suffix[:count-quota], dict(
                method=METHOD, matching_attempt=attempt+1, bottleneck_capacity=capacity,
                remaining_capacity=len(suffix), requested_agents=count, required_agents=quota,
                numpy_version=np.__version__, task_seed=seed,
                original_sampling_distribution_preserved=False,
                joint_mapf_feasibility_proven=False)
    raise ValueError('bounded endpoint matching exhausted; full task feasibility unknown')


def repair_task(map_data, config, seed, task_id, metadata_template):
    """Supported only for the registered scalar, no-clustering bottleneck OD task."""
    import numpy as np
    from scipy.sparse import csr_matrix
    import scipy

    expected = dict(density_reference='free_cells', hotspot_distribution='zipf',
        minimum_shortest_distance=8, maximum_shortest_distance=100, max_sampling_attempts=100000,
        agent_density=.25, required_bottleneck_crossing_ratio=.4, hotspot_skew=0.,
        od_matrix={'storage->station':.35,'station->storage':.35,'left->right':.15,'right->left':.15})
    if config != expected:
        raise ValueError('unsupported task configuration for frozen amendment')
    template = deepcopy(metadata_template)
    for key, value in dict(scenario_type='legacy_flow', origin_cluster_count=0,
        goal_cluster_count=0, swap_pair_ratio=0., shared_corridor_ratio=0.,
        required_intersection_crossing_ratio=0., target_bottleneck_mode='highest_prior',
        od_matrix=config['od_matrix'], od_quota_counts=None).items():
        if template.get(key) != value:
            raise ValueError('unsupported metadata template: '+key)
    pools = _candidate_pools(map_data)
    cells = sorted(pools['free'])
    ids = {c:i for i,c in enumerate(cells)}
    count = round(config['agent_density'] * len(cells))
    quota = round(config['required_bottleneck_crossing_ratio'] * count)
    bottlenecks = sorted(_bottleneck_candidates(map_data, 'highest_prior'))
    bd = {b:_distance_map(map_data,b) for b in bottlenecks}
    distances = {}
    legal, constrained = set(), set()
    for flow in sorted(config['od_matrix']):
        origin, destination = flow.split('->')
        for start in sorted(pools[origin]):
            if start not in distances:
                distances[start] = _distance_map(map_data,start)
            ds = distances[start]
            for goal in sorted(pools[destination]):
                d = ds.get(goal)
                if start == goal or d is None or not 8 <= d <= 100:
                    continue
                edge = (ids[start],ids[goal])
                legal.add(edge)
                if any(ds.get(b,10**9)+bd[b].get(goal,10**9)==d for b in bottlenecks):
                    constrained.add(edge)

    def matrix(edges):
        edges = sorted(edges)
        rr, cc = zip(*edges) if edges else ([],[])
        return csr_matrix((np.ones(len(edges), dtype=np.int8),(rr,cc)), shape=(len(cells),len(cells)))

    pairs, proof = assign_endpoints(matrix(legal),matrix(constrained),count,quota,seed)
    starts, goals = [cells[a] for a,_ in pairs], [cells[b] for _,b in pairs]
    assignments, required, actual_distances = [], [], []
    rng = random.Random(seed ^ 0x454E4450)
    pool_sets = {k:set(v) for k,v in pools.items()}
    for index, (start,goal) in enumerate(zip(starts,goals)):
        flows = [f for f in sorted(config['od_matrix'])
                 if start in pool_sets[f.split('->')[0]] and goal in pool_sets[f.split('->')[1]]]
        assignments.append(rng.choices(flows,weights=[config['od_matrix'][f] for f in flows],k=1)[0])
        d = distances[start][goal]
        actual_distances.append(d)
        candidates = [b for b in bottlenecks if distances[start].get(b,10**9)+bd[b].get(goal,10**9)==d]
        required.append(list(rng.choice(candidates)) if index < quota else None)
    proof.update(scipy_version=scipy.__version__, legal_edges=len(legal), constrained_edges=len(constrained))
    template.update(agent_count=count, agent_density_free_cells=round(count/len(cells),6),
        agent_density_service_cells=round(count/max(1,len(pools['storage'])),6),
        actual_shortest_distances=actual_distances, mean_shortest_distance=round(sum(actual_distances)/count,4),
        flow_assignments=assignments, realized_flow_counts=dict(Counter(assignments)),
        required_bottlenecks=required, required_intersections=[None]*count,
        required_intersection_component_ids=[None]*count, endpoint_generation_amendment=proof)
    task = TaskData(task_id=task_id,map_id=map_data.map_id,seed=seed,starts=starts,goals=goals,metadata=template)
    validate_task(map_data,task)
    validate_constraints(map_data,task,config)
    return task


def validate_constraints(map_data, task, config):
    """Independently recompute distance, OD and bottleneck obligations for every agent."""
    validate_task(map_data, task)
    pools = {k:set(v) for k,v in _candidate_pools(map_data).items()}
    bottlenecks = set(_bottleneck_candidates(map_data,'highest_prior'))
    cache = {}
    def distances(cell):
        if cell not in cache:
            cache[cell] = _distance_map(map_data,cell)
        return cache[cell]
    count = round(config['agent_density'] * len(pools['free']))
    quota = round(config['required_bottleneck_crossing_ratio'] * count)
    if task.agent_count != count:
        raise ValueError('agent count changed')
    requirements = task.metadata['required_bottlenecks']
    if len(requirements) != count or sum(b is not None for b in requirements) != quota:
        raise ValueError('bottleneck quota changed')
    actual = []
    for start,goal,flow,b in zip(task.starts,task.goals,task.metadata['flow_assignments'],requirements,strict=True):
        if flow not in config['od_matrix']:
            raise ValueError('unsupported OD')
        origin,destination = flow.split('->')
        if start not in pools[origin] or goal not in pools[destination]:
            raise ValueError('OD endpoint mismatch')
        d = distances(start).get(goal)
        if d is None or not config['minimum_shortest_distance'] <= d <= config['maximum_shortest_distance']:
            raise ValueError('invalid shortest distance')
        actual.append(d)
        if b is not None:
            b = tuple(b)
            if b not in bottlenecks or distances(start).get(b,10**9)+distances(b).get(goal,10**9) != d:
                raise ValueError('no shortest path through required bottleneck')
    if actual != task.metadata['actual_shortest_distances'] or dict(Counter(task.metadata['flow_assignments'])) != task.metadata['realized_flow_counts']:
        raise ValueError('stale task metadata')
