"""Read-only replay of failed endpoint generation, without invoking a solver."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from generators.models import MapData
from generators.task_flows import generate_tasks, _distance_map
from generators.config import merge_dicts
from scripts import run_sa_onpolicy as run


def inspect(index):
    out=ROOT/'build/sa-completion-independent-ttf-v1'
    reg=run.check_seal(run.read_json(out/'design.json'))
    base=run.read_json(ROOT/reg['config']['dataset_base'])
    folder=out/f'dataset/shards/{index:02d}/confirmation'
    saved=run.read_json(next((folder/'maps').glob('*.json')))
    md=MapData(**{k:saved[k] for k in ('map_id','seed','grid','metadata')})
    rng=random.Random(reg['map_masters'][index])
    assert rng.randrange(1,2**31)==md.seed
    low_seed=rng.randrange(1,2**31)
    low=run.read_json(next((folder/'instances').glob('*task_0000.json')))
    assert low['seed']==low_seed
    seed=rng.randrange(1,2**31)
    config=merge_dicts(base['task'],base['task_variants'][0]['task'])
    config.update(agent_density=.25,required_bottleneck_crossing_ratio=.4)
    try:
        task=generate_tasks(md,config,seed,md.map_id+'__task_0001')
        return dict(index=index,status='unexpected_success',agents=task.agent_count)
    except ValueError as error:
        tb=error.__traceback__
        while tb and tb.tb_frame.f_code.co_name!='generate_tasks':tb=tb.tb_next
        if tb is None:raise
        v=tb.tb_frame.f_locals
        pools=v['pools']
        starts,goals=v['used_starts'],v['used_goals']
        pairs=0
        by_flow={}
        allowed_starts=set()
        allowed_goals=set()
        requires_bottleneck=v['agent']<v['bottleneck_agent_count'] and bool(v['bottlenecks'])
        for flow in config['od_matrix']:
            origin,destination=flow.split('->')
            count=0
            for start in pools[origin]:
                if start in starts:continue
                if start not in v['distance_cache']:v['distance_cache'][start]=_distance_map(md,start)
                ds=v['distance_cache'][start]
                for goal in pools[destination]:
                    if goal in goals or start==goal:continue
                    distance=ds.get(goal)
                    if distance is None or distance<v['minimum_distance'] or distance>v['maximum_distance']:continue
                    if requires_bottleneck and not any(ds.get(b,10**9)+v['bottleneck_distance_cache'][b].get(goal,10**9)==distance for b in v['bottlenecks']):continue
                    count+=1
                    allowed_starts.add(start)
                    allowed_goals.add(goal)
            by_flow[flow]=count
            pairs+=count
        return dict(index=index,status='reproduced_generation_failure',error=str(error),task_seed=seed,
            map_id=md.map_id,agents_requested=v['agent_count'],agents_assigned=len(starts),
            bottleneck_quota=v['bottleneck_agent_count'],failing_agent_requires_bottleneck=requires_bottleneck,
            pool_sizes={k:len(p) for k,p in pools.items()},
            remaining_starts={k:len(set(p)-starts) for k,p in pools.items()},
            remaining_goals={k:len(set(p)-goals) for k,p in pools.items()},
            admissible_remaining_pairs_by_flow=by_flow,admissible_remaining_pairs=pairs,
            admissible_remaining_starts=len(allowed_starts),admissible_remaining_goals=len(allowed_goals),
            joint_mapf_infeasibility_proven=False,global_endpoint_infeasibility_proven=False,
            generator_unchanged=True,solver_calls=0)


def capacity(index):
    """Upper bound for constrained endpoint assignment, not joint MAPF feasibility."""
    from generators.task_flows import _candidate_pools, _bottleneck_candidates
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    import numpy as np
    folder=ROOT/f'build/sa-completion-independent-ttf-v1/dataset/shards/{index:02d}/confirmation/maps'
    saved=run.read_json(next(folder.glob('*.json')))
    md=MapData(**{k:saved[k] for k in ('map_id','seed','grid','metadata')})
    pools=_candidate_pools(md)
    cells=pools['free']
    ids={cell:i for i,cell in enumerate(cells)}
    bottlenecks=_bottleneck_candidates(md,'highest_prior')
    bd={b:_distance_map(md,b) for b in bottlenecks}
    distances={}
    rr,cc=[],[]
    for origin,destination in (('storage','station'),('station','storage'),('left','right'),('right','left')):
        for start in pools[origin]:
            if start not in distances:distances[start]=_distance_map(md,start)
            ds=distances[start]
            for goal in pools[destination]:
                d=ds.get(goal)
                if start==goal or d is None or not 8<=d<=100:continue
                if any(ds.get(b,10**9)+bd[b].get(goal,10**9)==d for b in bottlenecks):
                    rr.append(ids[start])
                    cc.append(ids[goal])
    graph=csr_matrix((np.ones(len(rr),dtype=bool),(rr,cc)),shape=(len(cells),len(cells)))
    matched=int(np.count_nonzero(maximum_bipartite_matching(graph,perm_type='column')>=0))
    requested=round(.4*round(.25*len(cells)))
    return dict(index=index,requested_bottleneck_agents=requested,maximum_bottleneck_endpoint_matching=matched,
        unique_admissible_pairs=int(graph.nnz),infeasible_bottleneck_quota_proven=matched<requested,
        caveat='Matching is only an endpoint capacity bound; not a joint MAPF feasibility test.',solver_calls=0)


if __name__=='__main__':
    capacity_only='--capacity-only' in sys.argv
    with ProcessPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(capacity if capacity_only else inspect,(7,10)))
    name='capacity' if capacity_only else 'diagnostic'
    output=ROOT/f'build/sa-completion-generation-{name}-20260928.json'
    run.once(output,run.sealed(dict(results=rows,solver_calls=0,timed_episodes=0)))
    print(json.dumps(rows,ensure_ascii=False,indent=2),flush=True)
