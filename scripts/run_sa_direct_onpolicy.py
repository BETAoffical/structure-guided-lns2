"""Twenty-worker complete-episode training and a map-disjoint serial comparison."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import random
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_sa_onpolicy as run
from scripts import run_sa_raw_ttf as executor
from experiments import sa_direct_onpolicy as rt
from experiments import sa_raw_timing_metrics as metrics
from experiments import sa_terminal_efficiency_timing as comparison

CONFIG='configs/sa_direct_onpolicy.json'
CODE=(CONFIG,'scripts/run_sa_direct_onpolicy.py','experiments/sa_direct_onpolicy.py',
      'tests/evaluation/test_sa_direct_onpolicy.py','docs/SA_DIRECT_ONPOLICY_PROTOCOL_ZH.md')


def config():return run.read_json(ROOT/CONFIG)


def compact(job):
    j=deepcopy(job)
    j['plan']={k:j['plan'][k] for k in ('config','native_file','proposal','template','binding')}
    for k in ('source_files','source_folder','source_job_id','prior_mode','schedule_index'):j.pop(k,None)
    return j


def prepare():
    cfg=config();out=ROOT/cfg['output'];run.require(not out.exists(),'existing on-policy experiment')
    old=run.read_json(ROOT/cfg['source']/'design.json')
    src=run.read_json(ROOT/cfg['training_source']/'registration.json')
    ev=run.read_json(ROOT/cfg['evaluation_source']/'registration.json')
    run.require(run.sha256_file(ROOT/cfg['parent_model'])==cfg['parent_sha256'],'parent changed')
    train=[compact(j) for j in src['jobs'] if j['comparison_arm']=='original' and j['case']['map_id'] in old['train_maps']]
    maps=sorted({j['case']['map_id'] for j in ev['jobs']})[6:]
    tests=[compact(j) for j in ev['jobs'] if j['comparison_arm']=='completion' and
           j['solver_seed']==271 and j['case']['map_id'] in maps]
    run.require(len(train)==16 and len(tests)==12 and len(maps)==6,'fixed scope')
    trainhash={run.sha256_file(ROOT/j['case']['files']['map_file']) for j in train}
    testhash={run.sha256_file(ROOT/j['case']['files']['map_file']) for j in tests}
    run.require(not trainhash&testhash,'train/test geometry overlap')
    inputs={p:run.sha256_file(ROOT/p) for p in CODE}
    for p in (cfg['parent_model'],cfg['source']+'/design.json',cfg['training_source']+'/registration.json',
              cfg['evaluation_source']+'/registration.json',train[0]['plan']['native_file'],
              'experiments/sa_direct_selector.py'):
        inputs[p]=run.sha256_file(ROOT/p)
    for j in train+tests:
        for p in j['case']['files'].values():inputs[p]=run.sha256_file(ROOT/p)
    body=dict(config=cfg,inputs=inputs,training_templates=sorted(train,key=lambda j:j['pair_id']),
        evaluation_templates=sorted(tests,key=lambda j:j['pair_id']),train_maps=old['train_maps'],
        evaluation_maps=maps,backup='../backups/20261004-before-direct-onpolicy',development_test=True)
    body['binding']=run.json_fingerprint(body);out.mkdir();(out/'models').mkdir()
    run.once(out/'design.json',run.sealed(body))
    run.once(out/'models/round-0.json',run.read_json(ROOT/cfg['parent_model']))
    print(json.dumps(dict(phase='prepared',rounds=2,training_episodes=128,workers=20,test_episodes=36)),flush=True)


def verify():
    cfg=config();out=ROOT/cfg['output'];r=run.check_seal(run.read_json(out/'design.json'))
    run.require(r['config']==cfg,'config changed')
    for p,h in r['inputs'].items():run.require(run.sha256_file(ROOT/p)==h,'changed input '+p)
    return r,out


def model_spec(out,index):
    p=out/'models'/f'round-{index}.json'
    return dict(path=p.relative_to(ROOT).as_posix(),sha256=run.sha256_file(p))


def training_jobs(r,out,index):
    cfg=r['config'];spec=model_spec(out,index-1);jobs=[]
    for template in r['training_templates']:
        for replica in range(cfg['replicas']):
            j=deepcopy(template)
            j.update(output=cfg['output'],model=spec,arm='trained_actor',comparison_arm='direct',
                round=index,replica=replica,config=cfg,parent_pid=os.getpid(),timing_binding=r['binding'],
                phase=f'direct-onpolicy-round-{index}',
                job_id=run.json_fingerprint(['direct-onpolicy',index,template['pair_id'],replica])[:24])
            j['plan']['config']['stream_seed']=cfg['stream_seed'];jobs.append(j)
    return jobs


def train():
    r,out=verify();cfg=r['config']
    if (out/'training.complete.json').exists():return
    for index in range(1,cfg['rounds']+1):
        receipt=out/'models'/f'update-{index}.json'
        if receipt.exists():continue
        jobs=training_jobs(r,out,index)
        pending=[j for j in jobs if not (out/'rollouts'/j['job_id']/'result.json').exists()]
        if pending:
            print(f'COLLECT ROUND {index}: {len(pending)}/64, {cfg["workers"]} workers',flush=True)
            run.write_json(out/'status.json',dict(phase='train',round=index,total=64,workers=cfg['workers']))
            executor.execute(r,out,pending,rt.rollout_worker,f'train-{index}',cfg['workers'],cfg['process_fuse_seconds'])
        rows=[run.check_seal(run.read_json(out/'rollouts'/j['job_id']/'result.json')) for j in jobs]
        parent=run.read_json(ROOT/model_spec(out,index-1)['path'])
        groups={j['pair_id']:j['case']['map_id'] for j in jobs}
        model,proof=rt.update(parent,rows,out,cfg,groups,index)
        run.once(out/'models'/f'round-{index}.json',model)
        summary=dict(binding=r['binding'],round=index,episodes=len(rows),
            success=sum(x['success'] for x in rows if x['status']=='ok'),
            known=sum(x['status']=='ok' for x in rows),censored=sum(x['status']=='censored' for x in rows),
            proof=proof,model=model_spec(out,index))
        run.once(receipt,run.sealed(summary))
        print(json.dumps(dict(phase='update',round=index,success=summary['success'],known=summary['known'],
              censored=summary['censored'],updated=proof['updated'],decision=proof['decision'],
              selected=proof.get('selected'),gradient_norm=proof.get('gradient_norm'))),flush=True)
        if summary['censored']:
            # Preserve every unknown and stop further training; never relabel as failure.
            run.once(out/'training.complete.json',dict(rounds=index,stopped_on_censored_batch=True))
            return
    if not (out/'training.complete.json').exists():run.once(out/'training.complete.json',dict(rounds=cfg['rounds']))


def test_templates(r):
    result=[]
    for t in r['evaluation_templates']:
        j=deepcopy(t);seed=r['config']['evaluation_solver_seed']
        j.update(solver_seed=seed,pair_id=f"{j['case']['task_id']}-s{seed}",
            job_id=run.json_fingerprint(['direct-onpolicy-reset',j['case']['task_id'],seed])[:24],
            output=r['config']['output'],timing_binding=r['binding'])
        j['case']['solver_seeds']=[seed];j['plan']['config']['stream_seed']=r['config']['stream_seed']
        result.append(j)
    return result


def qualify(r,out):
    if (out/'qualification.json').exists():return run.read_json(out/'qualification.json')
    with ProcessPoolExecutor(max_workers=min(r['config']['workers'],12)) as pool:
        rows=list(pool.map(rt.qualification_worker,test_templates(r)))
    run.once(out/'qualification.json',dict(binding=r['binding'],rows=rows))
    print(json.dumps(dict(phase='new_seed_initials',valid=len(rows),all_retained=True)),flush=True)
    return dict(rows=rows)


def test_jobs(r,out):
    proofs={x['pair_id']:x['initial'] for x in qualify(r,out)['rows']}
    final=run.read_json(out/'training.complete.json')['rounds'];orders=list(itertools.permutations(('updated','frozen','dual')))
    jobs=[]
    for i,t in enumerate(test_templates(r)):
        for arm in orders[i%6]:
            j=deepcopy(t)
            j.update(comparison_arm=arm,arm='trained_actor' if arm!='dual' else 'dual16_sa',
                model=model_spec(out,final if arm=='updated' else 0) if arm!='dual' else None,
                expected_initial=proofs[t['pair_id']],parent_pid=os.getpid(),iteration=final,
                job_id=run.json_fingerprint(['direct-onpolicy-test',t['pair_id'],arm])[:24],
                phase='direct-onpolicy-test-20261004')
            jobs.append(j)
    return jobs


def evaluate():
    r,out=verify();jobs=test_jobs(r,out)
    for i,j in enumerate(jobs):
        result=out/'episodes'/j['job_id']/'result.json'
        if result.exists():
            row=run.check_seal(run.read_json(result));run.require(row['binding']==r['binding'],'resume identity');continue
        run.require(not result.parent.exists(),'partial test episode; inspect')
        print(f"START {i+1}/{len(jobs)} {j['comparison_arm']} {j['pair_id']}",flush=True)
        run.write_json(out/'status.json',dict(phase='evaluate',completed=i,total=len(jobs),active=j['comparison_arm']))
        executor.execute(r,out,[j],rt.timed_worker,'collect',1,240.)
    run.write_json(out/'status.json',dict(phase='complete',completed=36,total=36))
    report()


def report():
    r,out=verify();jobs=test_jobs(r,out)
    rows=[run.check_seal(run.read_json(out/'episodes'/j['job_id']/'result.json')) for j in jobs]
    run.require(len(rows)==36 and all(x['status']=='ok' for x in rows),'test coverage')
    pairs=defaultdict(dict)
    for j,x in zip(jobs,rows):
        run.require(x['initial_fingerprint']==j['expected_initial'] and x['binding']==r['binding'],'test identity')
        pairs[x['pair_id']][x['comparison_arm']]=x
    run.require(len(pairs)==12 and all(set(g)=={'updated','frozen','dual'} for g in pairs.values()),'paired coverage')
    rng=random.Random(r['config']['seed']);maps=sorted({x['map_id'] for x in rows})
    draws=[Counter(rng.choices(maps,k=len(maps))) for _ in range(r['config']['bootstrap'])]
    rounds=run.read_json(out/'training.complete.json')['rounds']
    result=dict(binding=r['binding'],training=[run.check_seal(run.read_json(out/'models'/f'update-{i}.json')) for i in range(1,rounds+1)],
        arms={a:metrics._arm([x for x in rows if x['comparison_arm']==a]) for a in ('updated','frozen','dual')},
        contrasts={a:comparison.contrast([dict(parent=g[a],completion=g['updated']) for g in pairs.values()],draws)
                   for a in ('frozen','dual')},
        by_density={den:{a:dict(success=sum(x['success_within_budget'] for j,x in zip(jobs,rows)
                if j['case']['task_variant']==den and x['comparison_arm']==a),denominator=6)
                for a in ('updated','frozen','dual')} for den in ('bottleneck_d20','bottleneck_d25')},
        episodes=rows,student_train_map_disjoint=True,solver_seed=307,historically_viewed_maps=True,
        training_work_reward_not_ttf_reward=True,full_trace_replay=False,terminal_paths_checked=True)
    if not (out/'report.json').exists():run.once(out/'report.json',run.sealed(result))
    print(json.dumps(dict(phase='complete',success={a:v['success_count'] for a,v in result['arms'].items()},
                          report=str(out/'report.json'))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('prepare','train','evaluate','report','all'))
    args=p.parse_args()
    if args.phase=='all':
        if not (ROOT/config()['output']/'design.json').exists():prepare()
        train();evaluate()
    elif args.phase=='prepare':prepare()
    else:globals()[args.phase]()


if __name__=='__main__':main()
