"""One final paired-metadata and collision check; no solver replay or fitting."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_sa_onpolicy as run
from experiments.sa_raw_selection_fast import _bind
from experiments import sa_raw_timing_metrics as metrics


def check_collision_free(state):
    agents=state['agents'];paths=[a['path'] for a in agents]
    run.require(all(p and p[0]==a['start'] and p[-1]==a['goal'] for p,a in zip(paths,agents)),
                'final path endpoints')
    run.require(sum(len(p)-1 for p in paths)==state['sum_of_costs'],'final SOC')
    for t in range(max(map(len,paths))):
        vertices=set();edges=set()
        for p in paths:
            now=p[min(t,len(p)-1)]
            run.require(now not in vertices,'final vertex collision');vertices.add(now)
            if t:
                before=p[min(t-1,len(p)-1)]
                run.require((now,before) not in edges,'final edge-swap collision')
                edges.add((before,now))


def main():
    cfg=run.read_json(ROOT/'configs/sa_direct_onpolicy.json');out=ROOT/cfg['output']
    report=run.check_seal(run.read_json(out/'report.json'))
    rows=report['episodes']
    groups=_bind(metrics._validate,ARMS=('updated','frozen','dual'))(rows)
    run.require(len(rows)==36 and len(groups)==12,'paired cohort coverage')
    verified=0
    for row in rows:
        folder=out/'episodes'/row['job_id']
        run.require(run.check_seal(run.read_json(folder/'result.json'))==row,'reported row mismatch')
        run.require(run.sha256_file(folder/'final.json')==row['files']['final.json'],'final artifact changed')
        state=run.read_json(folder/'final.json')
        if row['success_within_budget']:
            check_collision_free(state);verified+=1
    run.require(run.sha256_file(ROOT/cfg['parent_model'])==cfg['parent_sha256'],'frozen parent changed')
    proof=dict(paired_conditions=len(groups),episodes=len(rows),success_paths_collision_free=verified,
               vertex_and_edge_swaps_checked=True,stay_at_goal=True,full_trace_replay=False,
               frozen_parent_unchanged=True,report_sha256=run.sha256_file(out/'report.json'),
               models={p.name:run.sha256_file(p) for p in sorted((out/'models').glob('round-*.json'))})
    run.once(out/'final-check.json',run.sealed(proof))
    print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':main()
