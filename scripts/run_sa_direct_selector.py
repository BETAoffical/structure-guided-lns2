"""Parallel distillation followed by a small serial three-arm solver experiment."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import random
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_sa_onpolicy as run
from scripts import run_sa_raw_ttf as serial
from experiments import sa_direct_selector as rt
from experiments import sa_raw_timing_metrics as metrics
from experiments import sa_terminal_efficiency_timing as comparison

CONFIG='configs/sa_direct_selector.json'
CODE=(CONFIG,'scripts/run_sa_direct_selector.py','experiments/sa_direct_selector.py',
      'tests/evaluation/test_sa_direct_selector.py','docs/SA_DIRECT_SELECTOR_PROTOCOL_ZH.md')


def config(): return run.read_json(ROOT/CONFIG)


def prepare():
    cfg=config();out=ROOT/cfg['output']
    run.require(not out.exists(),'existing experiment; use train/evaluate to resume completed phases')
    source=run.read_json(ROOT/cfg['training_source']/'registration.json')
    run.require(run.read_json(ROOT/cfg['training_source']/'audit.complete.json')['jobs']==72,'source incomplete')
    evaluation=run.read_json(ROOT/cfg['evaluation_source']/'registration.json')
    run.require(run.sha256_file(ROOT/cfg['teacher'])==cfg['teacher_sha256'],'teacher changed')
    maps=sorted({j['case']['map_id'] for j in source['jobs']})
    run.require(len(maps)==cfg['train_maps']+cfg['validation_maps'],'training map count')
    teacher_spec=deepcopy(next(j['model'] for j in source['jobs'] if j['comparison_arm']=='original'))
    eligible=[j for j in evaluation['jobs'] if j['comparison_arm']=='completion' and
              j['solver_seed']==cfg['evaluation_solver_seed']]
    evalmaps=sorted({j['case']['map_id'] for j in eligible})[:cfg['evaluation_maps']]
    eligible=sorted([j for j in eligible if j['case']['map_id'] in evalmaps],key=lambda j:j['pair_id'])
    run.require(len(eligible)==2*cfg['evaluation_maps'],'evaluation pairing')
    trainhash={run.sha256_file(ROOT/j['case']['files']['map_file']) for j in source['jobs']}
    evalhash={run.sha256_file(ROOT/j['case']['files']['map_file']) for j in eligible}
    run.require(not trainhash&evalhash,'map geometry overlap')
    extraction=[dict(index=i,map_id=m,split='train' if i<cfg['train_maps'] else 'validation',config=cfg,
        jobs=[j for j in source['jobs'] if j['case']['map_id']==m]) for i,m in enumerate(maps)]
    inputs={p:run.sha256_file(ROOT/p) for p in CODE}
    for p in (cfg['teacher'],source['jobs'][0]['plan']['native_file'],
              cfg['training_source']+'/registration.json',cfg['training_source']+'/audit.complete.json',
              cfg['evaluation_source']+'/registration.json'):
        inputs[p]=run.sha256_file(ROOT/p)
    body=dict(config=cfg,inputs=inputs,extraction=extraction,evaluation_templates=eligible,
              teacher_spec=teacher_spec,train_maps=maps[:cfg['train_maps']],validation_maps=maps[cfg['train_maps']:],
              evaluation_maps=evalmaps,source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              backup='../backups/20261004-before-direct-selector',development_test=True)
    body['binding']=run.json_fingerprint(body)
    out.mkdir()
    for name in ('data','models'): (out/name).mkdir()
    run.once(out/'design.json',run.sealed(body))
    return body,out


def verify():
    cfg=config();out=ROOT/cfg['output'];r=run.check_seal(run.read_json(out/'design.json'))
    run.require(r['config']==cfg,'config changed')
    for p,h in r['inputs'].items(): run.require(run.sha256_file(ROOT/p)==h,'changed input: '+p)
    return r,out


def train():
    r,out=verify();cfg=r['config']
    if not (out/'dataset.json').exists():
        completed={int(p.stem):run.read_json(p) for p in (out/'data').glob('*.json')}
        pending=[j for j in r['extraction'] if j['index'] not in completed]
        with ProcessPoolExecutor(max_workers=min(cfg['workers'],len(pending) or 1)) as pool:
            futures={pool.submit(rt.extract_map,j):j['index'] for j in pending}
            for f in as_completed(futures):
                entry=f.result();completed[futures[f]]=entry
                print(json.dumps(dict(phase='extract',map=entry['map_id'],states=entry['states'],
                                      available=entry['available']),ensure_ascii=False),flush=True)
        entries=[completed[i] for i in range(len(r['extraction']))]
        run.once(out/'dataset.json',dict(binding=r['binding'],maps=entries,feature_names=entries[0]['feature_names']))
        raw,_,_=rt.load_data(out,'train')
        scale=raw.std(axis=0);scale[scale<1e-6]=1.
        run.once(out/'normalization.json',dict(feature_names=entries[0]['feature_names'],
                  mean=raw.mean(axis=0).tolist(),scale=scale.tolist(),training_only=True))
    jobs=[dict(width=w,seed=s,config=cfg) for w in cfg['widths'] for s in cfg['seeds']]
    pending=[j for j in jobs if not (out/'models'/f"h{j['width']}-s{j['seed']}-result.json").exists()]
    with ProcessPoolExecutor(max_workers=min(cfg['workers'],len(pending) or 1)) as pool:
        futures=[pool.submit(rt.train_worker,j) for j in pending]
        for f in as_completed(futures):
            v=f.result();print(json.dumps(dict(phase='train',id=v['id'],epoch=v['epoch'],
                validation=v['validation'],seconds=v['seconds'])),flush=True)
    results=[run.read_json(out/'models'/f"h{j['width']}-s{j['seed']}-result.json") for j in jobs]
    winner=min(results,key=lambda v:(v['validation']['cross_entropy'],v['id']))
    run.require(len(results)==20,'model coverage')
    if not (out/'selection.json').exists():
        run.once(out/'selection.json',dict(binding=r['binding'],criterion='minimum_validation_cross_entropy',
                selected=winner,all_models=results,model_sha256=run.sha256_file(ROOT/winner['path'])))
    print(json.dumps(dict(phase='selected',model=winner),ensure_ascii=False),flush=True)
    return winner


def jobs_for(r,out):
    selected=run.read_json(out/'selection.json');winner=selected['selected']
    run.require(run.sha256_file(ROOT/winner['path'])==selected['model_sha256'],'selected model changed')
    spec=dict(path=winner['path'],sha256=selected['model_sha256'])
    orders=list(itertools.permutations(('direct','a','dual')))
    jobs=[]
    for index,template in enumerate(r['evaluation_templates']):
        for arm in orders[index%6]:
            j=deepcopy(template);j.pop('schedule_index',None)
            j.update(comparison_arm=arm,arm='trained_actor' if arm!='dual' else 'dual16_sa',
                model=spec if arm=='direct' else r['teacher_spec'] if arm=='a' else None,
                teacher_spec=r['teacher_spec'],iteration=2,phase='direct-selector-test-20261004',
                job_id=run.json_fingerprint(['direct-selector-v1',template['pair_id'],arm])[:24],
                output=r['config']['output'],timing_binding=r['binding'],parent_pid=os.getpid())
            j['plan']['config']['stream_seed']=r['config']['stream_seed']
            jobs.append(j)
    return jobs


def evaluate():
    r,out=verify();jobs=jobs_for(r,out)
    if not (out/'smoke.json').exists():
        smokejobs=[j for j in jobs if j['comparison_arm']=='direct'][:2]
        with ProcessPoolExecutor(max_workers=2) as pool: proof=list(pool.map(rt.smoke,smokejobs))
        run.once(out/'smoke.json',dict(binding=r['binding'],checks=proof))
        print(json.dumps(dict(phase='smoke',passed=len(proof))),flush=True)
    for i,j in enumerate(jobs):
        result=out/'episodes'/j['job_id']/'result.json'
        if result.exists():
            row=run.check_seal(run.read_json(result));run.require(row['binding']==r['binding'] and row['status']=='ok','resume row')
            continue
        run.require(not result.parent.exists(),'partial episode; inspect')
        print(f"START {i+1}/{len(jobs)} {j['comparison_arm']} {j['pair_id']}",flush=True)
        run.write_json(out/'status.json',dict(phase='evaluate',completed=i,total=len(jobs),active=j['comparison_arm']))
        serial.execute(r,out,[j],rt.timed_worker,'collect',1,240.)
    run.write_json(out/'status.json',dict(phase='complete',completed=len(jobs),total=len(jobs)))
    return report()


def report():
    r,out=verify();jobs=jobs_for(r,out)
    rows=[run.check_seal(run.read_json(out/'episodes'/j['job_id']/'result.json')) for j in jobs]
    run.require(len(rows)==36 and len({x['job_id'] for x in rows})==36 and all(x['status']=='ok' for x in rows),'coverage')
    for j,row in zip(jobs,rows):
        run.require(row['binding']==r['binding'] and row['initial_fingerprint']==j['expected_initial'],'result identity')
    pairs=defaultdict(dict)
    for row in rows: pairs[row['pair_id']][row['comparison_arm']]=row
    run.require(len(pairs)==12 and all(set(g)=={'direct','a','dual'} for g in pairs.values()),'paired coverage')
    maps=sorted({x['map_id'] for x in rows});rng=random.Random(r['config']['seed'])
    draws=[Counter(rng.choices(maps,k=len(maps))) for _ in range(r['config']['bootstrap'])]
    result=dict(binding=r['binding'],selection=run.read_json(out/'selection.json')['selected'],
        arms={a:metrics._arm([x for x in rows if x['comparison_arm']==a]) for a in ('direct','a','dual')},
        contrasts={a:comparison.contrast([dict(parent=g[a],completion=g['direct']) for g in pairs.values()],draws)
                   for a in ('a','dual')},
        by_density={density:{a:dict(success=sum(x['success_within_budget'] for j,x in zip(jobs,rows)
                    if j['case']['task_variant']==density and x['comparison_arm']==a),denominator=6)
                    for a in ('direct','a','dual')} for density in ('bottleneck_d20','bottleneck_d25')},
        independent_of_student_training_maps=True,historically_viewed_development_test=True,
        full_trace_replay=False,terminal_paths_checked=True,dual_runtime_calls_direct=0,
        no_outcome_filtered_cases=True,episodes=rows)
    if not (out/'report.json').exists():run.once(out/'report.json',run.sealed(result))
    print(json.dumps(dict(phase='complete',success={a:v['success_count'] for a,v in result['arms'].items()},
                         report=str(out/'report.json'))),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','train','evaluate','report','all'))
    args=parser.parse_args()
    if args.phase=='all':
        if not (ROOT/config()['output']/'design.json').exists():prepare()
        train();evaluate()
    elif args.phase=='prepare': prepare()
    else:globals()[args.phase]()


if __name__=='__main__':main()
