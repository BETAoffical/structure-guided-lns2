"""Frozen A on four existing Train conditions, sixteen fresh replicas each; no training or TTF."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_sa_completion_continuation as prior
from scripts import audit_sa_completion_credit_transfer as credit_source
from experiments import sa_completion_replicability as statistics
from experiments.sa_completion_credit_transfer import trajectory_score,cosine
from experiments.sa_raw_selection_fast import _bind

run,raw,require = prior.run,prior.raw,prior.require
CONFIG = 'configs/sa_completion_replicability.json'


def configuration():
    cfg = run.read_json(ROOT/CONFIG)
    fixed = dict(replicas=16,workers=20,max_decisions=None,node_budget=25000000,
        pp_safety_seconds=20.,episode_safety_seconds=900.,process_fuse_seconds=960.,
        bootstrap=1000,bootstrap_seed=2026100101,formal_ttf=False,automatic_promotion=False,
        maximum_updates_per_arm=0,phase='completion-credit-replicability-20261001',
        parent_sha256='81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062')
    require(all(cfg[k]==v for k,v in fixed.items()),'fixed replication budget changed')
    require(len(cfg['pairs'])==len(set(cfg['pairs']))==4 and cfg['pairs'][0]==cfg['primary_pair'], 'fixed four-condition scope')
    return cfg


def schedule(conditions,phase,replicas):
    require(len(conditions)==4 and len({c['pair_id'] for c in conditions})==4,'four unique conditions')
    return [dict(c,phase=phase,replica=i,comparison_arm='parent',arm='trained_actor',
                 job_id=run.json_fingerprint([phase,c['pair_id'],i,'parent'])[:24])
            for i in range(replicas) for c in conditions]


def prepare():
    cfg = configuration()
    out = ROOT/cfg['output']
    require(not out.exists(),'existing output; verify/resume instead')
    source,src = prior.verify()
    proof = credit_source.verify_report()
    require(src==ROOT/cfg['source'] and credit_source.configuration()['output']==cfg['credit_source'], 'source paths')
    require(source['config']['parent_sha256']==cfg['parent_sha256'] and source['parent']['iteration']==2, 'frozen A lineage')
    all_rows = list(prior.audited_rows(source,src,'train'))
    selected = [(j,r) for j,r in all_rows if j['pair_id'] in cfg['pairs']]
    require(len(selected)==16 and {j['pair_id'] for j,_ in selected}==set(cfg['pairs']), 'original four replicas')
    require(all(j['split']=='train' and j['comparison_arm']=='parent' for j,_ in selected), 'Train A only')
    require(cfg['phase'] not in {j['phase'] for j,_ in all_rows}, 'reused random phase')
    conditions=[]
    for pair in cfg['pairs']:
        group=[(j,r) for j,r in selected if j['pair_id']==pair]
        require(len(group)==4 and 0<sum(r['success'] for _,r in group)<4, 'controls originally mixed')
        j=group[0][0]
        require(len({r['initial_fingerprint'] for _,r in group})==1,'changed original initial')
        conditions.append({k:j[k] for k in ('case','pair_id','solver_seed','split','expected_initial')})
    inputs={}
    def register(path,expected=None):
        path=Path(path)
        digest=run.sha256_file(path)
        require(expected is None or digest==expected,'changed frozen source: '+str(path))
        inputs[path.relative_to(ROOT).as_posix()]=digest
    for j,r in selected:
        for name in j['case']['files'].values():
            register(ROOT/name,source['inputs'][name])
        folder=prior.old.previous.folder(j)
        register(folder/'result.json')
        for name,sha in r['files'].items():
            register(run.contained_file(folder,name,field='original trace'),sha)
    plan=deepcopy(source['plans']['train'])
    require(plan['template']['environment']['max_repair_iterations']==0,'hidden native iteration cap')
    register(ROOT/plan['native_file'],plan['config']['native_sha256'])
    register(ROOT/cfg['parent_model'],cfg['parent_sha256'])
    for p in (ROOT/CONFIG,Path(__file__),Path(statistics.__file__),
              ROOT/'docs/SA_COMPLETION_REPLICABILITY_PROTOCOL_ZH.md',
              ROOT/'tests/evaluation/test_sa_completion_replicability.py',
              src/'registration.json',src/'train.audit.json',
              ROOT/cfg['credit_source']/'registration.json',ROOT/cfg['credit_source']/'report.json',
              ROOT/'build/sa-frozen-confirmation-ttf-v1/report.json'):
        register(p)
    for directory in ('scripts','experiments','lns2_selector'):
        for p in (ROOT/directory).rglob('*.py'):
            register(p)
    for p in (ROOT/'artifacts/initlns-closed-loop-controller-v2').glob('*.json'):
        register(p)
    for name,sha in run.check_seal(run.read_json(ROOT/cfg['credit_source']/'report.json'))['episode_files'].items():
        if name.startswith('episodes/A2/'):
            register(ROOT/cfg['credit_source']/name,sha)
    body=dict(schema=cfg['schema'],config=cfg,inputs=inputs,parent=source['parent'],plans={'train':plan},
        conditions=conditions,train_jobs=schedule(conditions,cfg['phase'],16),original_jobs=[j for j,_ in selected],
        original_rng_streams=[r['rng_stream_id'] for _,r in selected],
        source_binding=source['binding'],credit_report_sha256=proof['report_sha256'],
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        no_training=True,no_ttf=True,no_promotion=True,development_diagnostic=True)
    body['binding']=run.json_fingerprint(body)
    with prior.old.recovery.strict_lock(out,body['binding'],'credit-replication-prepare'):
        run.once(out/'registration.json',run.sealed(body))
    return dry_run_body(body)


def verify():
    return _bind(prior.old.verify,ROOT=ROOT,configuration=configuration)()


def model_spec(reg,arm):
    require(arm=='parent','no challenger or trained model allowed')
    cfg=reg['config']
    return dict(path=cfg['parent_model'],sha256=cfg['parent_sha256'],policy_sha256=raw.validate_bundle(reg['parent']))


def jobs_for(reg,lane='train'):
    require(lane=='train','no validation/evaluation lane')
    jobs=_bind(prior.old.jobs_for,ROOT=ROOT,configuration=configuration,model_spec=model_spec)(reg,lane)
    for j in jobs:
        j['iteration']=reg['parent']['iteration']
    return jobs


def call(function,*args):
    return _bind(function,ROOT=ROOT,configuration=configuration,verify=verify,
                 model_spec=model_spec,jobs_for=jobs_for,audited_rows=audited_rows)(*args)


def audited_rows(reg,out,lane='train'):
    return call(prior.old.audited_rows,reg,out,lane)


def collect(resume=False):
    return call(prior.old.collect,'train',resume)


def audit():
    return call(prior.old.audit,'train')


def dry_run_body(reg):
    cfg=reg['config']
    count=len(reg['train_jobs'])
    require(count==64,'fixed episode count')
    return dict(binding=reg['binding'],conditions=4,episodes=count,workers=20,
        max_decisions=None,node_budget_per_episode=cfg['node_budget'],
        total_node_boundary=count*cfg['node_budget'],atomic_repair_may_overshoot=True,
        process_fuse_batch_upper_seconds=((count+19)//20)*cfg['process_fuse_seconds'],
        maximum_parameter_updates=0,formal_ttf=False,
        runtime_estimate_note='Historical parallel diagnostic durations suggest roughly 5-20 minutes plus audit, not TTF.')


def score_worker(job):
    row,pack=prior.old.previous.extract_worker(job)
    require(row['split']=='train' and row['status']=='ok' and pack is not None,'complete Train trajectories only')
    score,_=trajectory_score(prior.old.previous.load_model(job),pack)
    return dict({k:row[k] for k in ('episode_id','job_id','pair_id','map_id','replica','split','status',
                    'success','decisions','generated','initial_fingerprint','policy_sha256','rng_stream_id','stop')},
                score=score.tolist())


def old_statistics(reg):
    root=ROOT/reg['config']['credit_source']
    rows=[]
    all_vectors=[]
    for p in sorted((root/'episodes/A2').glob('*.json')):
        doc=run.check_seal(run.read_json(p))
        all_vectors.append(np.asarray(doc['loss_gradient']))
        if doc['pair_id'] in reg['config']['pairs']:
            require(doc['coefficient']!=0,'original selected conditions must have credit')
            r=doc['credit_row']
            rows.append(dict({k:r[k] for k in ('episode_id','pair_id','map_id','replica','split','status','success',
                'decisions','generated','initial_fingerprint','policy_sha256','rng_stream_id','stop')},
                score=(-np.asarray(doc['loss_gradient'])/doc['coefficient']).tolist()))
    require(len(rows)==16 and len(all_vectors)==96,'original score coverage')
    return rows,np.sum(all_vectors,axis=0)


def summarize(reg,original,fresh,full_old_gradient):
    cfg=reg['config']
    require(len(fresh)==64 and len({r['episode_id'] for r in fresh})==64 and
            len({r['rng_stream_id'] for r in fresh})==64,'new batch completeness/independence')
    require(not set(reg['original_rng_streams']) & {r['rng_stream_id'] for r in fresh},'reused old RNG streams')
    conditions={}
    old_parts,new_parts=[],[]
    for i,pair in enumerate(cfg['pairs']):
        a=sorted([r for r in original if r['pair_id']==pair],key=lambda r:r['replica'])
        b=sorted([r for r in fresh if r['pair_id']==pair],key=lambda r:r['replica'])
        require([r['replica'] for r in a]==list(range(4)) and [r['replica'] for r in b]==list(range(16)), 'replica inventory')
        summary,go,gn=statistics.condition_report(a,b,bootstrap=cfg['bootstrap'],seed=cfg['bootstrap_seed']+i)
        conditions[pair]=summary
        old_parts.append(go/24)
        new_parts.append(gn/24)
    # Keep all original 24 condition masses; substitute only estimates for these four.
    hybrid=full_old_gradient-np.sum(old_parts,axis=0)+np.sum(new_parts,axis=0)
    return dict(schema=cfg['schema']+'.report',binding=reg['binding'],conditions=conditions,
        primary_pair=cfg['primary_pair'],episodes=64,errors=0,censored=0,
        successes=sum(r['success'] for r in fresh),failures=sum(not r['success'] for r in fresh),
        synthetic_replacement=dict(original_condition_mass=1/24,conditions_replaced=4,
            untouched_conditions=20,original_gradient_norm=float(np.linalg.norm(full_old_gradient)),
            replacement_gradient_norm=float(np.linalg.norm(hybrid)),
            alignment_with_original=cosine(hybrid,full_old_gradient),
            no_parameter_update=True,not_an_independent_batch=True),
        decision='conditional_credit_replication_not_controller_performance',
        no_training=True,no_ttf=True,no_promotion=True)


def analyze():
    reg,out=verify()
    jobs=[j for j,_ in audited_rows(reg,out)]
    require(len(jobs)==64,'complete audited collection required; unknown is not failure')
    with prior.old.recovery.strict_lock(out,reg['binding'],'credit-replication-analysis'):
        require(not (out/'report.json').exists(),'existing analysis; verify instead')
        with ProcessPoolExecutor(max_workers=20) as pool:
            fresh=[]
            for row in pool.map(score_worker,jobs):
                fresh.append(row)
                if len(fresh)%8==0:
                    print(json.dumps(dict(phase='score_replay',done=len(fresh),total=64)),flush=True)
        original,old_gradient=old_statistics(reg)
        result=summarize(reg,original,fresh,old_gradient)
        result.update(audit_sha256=run.sha256_file(out/'train.audit.json'),
                      complete_sha256=run.sha256_file(out/'train.complete.json'))
        run.once(out/'scores.json',run.sealed(dict(binding=reg['binding'],rows=fresh)))
        result['scores_sha256']=run.sha256_file(out/'scores.json')
        verify()
        run.once(out/'report.json',run.sealed(result))
        run.write_json(out/'run_status.json',dict(status='complete',binding=reg['binding'],decision=result['decision']))
    return result


def verify_report():
    reg,out=verify()
    r=run.check_seal(run.read_json(out/'report.json'))
    require(r['binding']==reg['binding'] and r['no_training'] and r['no_ttf'] and r['no_promotion'], 'report scope')
    for key,path in (('audit_sha256','train.audit.json'),('complete_sha256','train.complete.json'),('scores_sha256','scores.json')):
        require(r[key]==run.sha256_file(out/path),'changed completed evidence')
    require(len(list(audited_rows(reg,out)))==64,'report episode coverage')
    return dict(verified=True,report_sha256=run.sha256_file(out/'report.json'),episodes=64,decision=r['decision'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','dry-run','collect','audit','analyze','verify','stop'))
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.phase=='collect': result=collect(args.resume)
    elif args.phase=='dry-run': result=dry_run_body(verify()[0])
    elif args.phase=='verify':
        result=verify_report() if (ROOT/configuration()['output']/'report.json').exists() else dict(verified=True,binding=verify()[0]['binding'])
    elif args.phase=='stop':
        reg,out=verify()
        run.write_json(out/'STOP_AFTER_BATCH',dict(requested=True))
        result=dict(stop_after_current_batch=True)
    else: result=globals()[args.phase]()
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
