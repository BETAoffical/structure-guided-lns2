"""Frozen raw selector serial TTF: prepare, preflight, collect, audit, report."""
import argparse
import itertools
import json
import os
from pathlib import Path
import platform
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import confirm_sa_raw_independent as source
from scripts import run_sa_onpolicy as run
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_raw_timed_runtime as rt
from experiments.sa_raw_confirmation import ARMS

CONFIG = 'configs/sa_raw_serial_ttf.json'
CODE = (CONFIG,'scripts/run_sa_raw_ttf.py','experiments/sa_raw_timed_runtime.py',
        'experiments/sa_raw_timing_metrics.py','tests/evaluation/test_sa_raw_timing_metrics.py',
        'tests/evaluation/test_sa_raw_timed_runtime.py','docs/SA_RAW_SERIAL_TTF_PROTOCOL_ZH.md')
require = run.require


def config():
    c=run.read_json(ROOT/CONFIG)
    require(c['workers']==1 and c['audit_workers']==20 and c['budget_seconds']==120. and
        c['max_decisions'] is None and c['execution_node_budget'] is None and
        c['feature_decision_reference']==256 and c['feature_node_reference']==25000000 and
        c['no_training'] and not c['automatic_promotion'],'timing scope changed')
    return c


def schedule(jobs):
    groups={}
    for j in jobs:
        group=groups.setdefault((j['pair_id'],j['replica']),{})
        require(j['comparison_arm'] not in group,'duplicate arm')
        group[j['comparison_arm']]=j
    orders=list(itertools.permutations(ARMS))
    result=[]
    for i,(_,group) in enumerate(sorted(groups.items())):
        require(set(group)==set(ARMS),'missing paired arm')
        for arm in orders[i%len(orders)]:
            result.append(dict(group[arm],schedule_index=len(result)))
    return result


def prepare():
    c=config()
    old,src=source.verify()
    require(src==ROOT/c['source'] and run.sha256_file(src/'report.json')==c['source_report_sha256'],'source report')
    out=ROOT/c['output']
    require(not out.exists(),'registration exists; verify/resume')
    source.audited(old,src)
    jobs=source.ready_jobs(old,src)
    for j in jobs:
        recorded=source.read_result(j)
        j.pop('parent_pid',None)
        j.update(source_folder=source.prior.folder(j).relative_to(ROOT).as_posix(),output=c['output'],
                 budget_seconds=c['budget_seconds'],preflight_steps=c['preflight_steps'],
                 source_files=recorded['files'],source_decisions=recorded['decisions'])
    inputs=dict(old['inputs'])
    for n in CODE:
        inputs[n]=run.sha256_file(ROOT/n)
    for n in ('registration.json','qualification.json','cases.json','report.json','comparison.complete.json','comparison.audit.json'):
        inputs[(src/n).relative_to(ROOT).as_posix()]=run.sha256_file(src/n)
    body=dict(schema='lns2.sa_raw_ttf.registration.v1',config=c,inputs=inputs,jobs=schedule(jobs),
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        interpretation='reused_six_same_family_maps_timing_confirmation_not_new_generalization',
        no_training=True,no_promotion=True)
    body['binding']=run.json_fingerprint(body)
    with recovery.strict_lock(out,body['binding'],'timing-prepare'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(episodes=len(jobs),pairs=48,workers=1,budget=120,serial_planning_upper_hours=6.4,
                external_fuse_upper_hours=12.8,safe_stop_after_episode=True)


def verify():
    c=config()
    out=ROOT/c['output']
    r=run.check_seal(run.read_json(out/'registration.json'))
    require(r['config']==c and r['binding']==run.json_fingerprint({k:v for k,v in r.items() if k not in ('binding','integrity')}),'registration')
    for n,h in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,n,field='timing input'))==h,'changed input: '+n)
    require(r['jobs']==schedule(r['jobs']) and len(r['jobs'])==192,'timing schedule')
    return r,out


def jobs_for(r):
    return [dict(j,parent_pid=os.getpid(),timing_binding=r['binding']) for j in r['jobs']]


def execute(r,out,jobs,worker,phase,workers,timeout):
    from experiments.repair_collection import _run_jobs
    def failed(j,status,error):
        row=dict(status=status,job_id=j['job_id'],error=error,binding=r['binding'],phase=phase)
        run.once(out/'failures'/f"{phase}-{j['job_id']}.json",run.sealed(row))
        return row
    def progress(row):
        run.once(out/phase/(row['job_id']+'.json'),run.sealed(dict(row,binding=r['binding'])))
        print(json.dumps(dict(phase=phase,**row)),flush=True)
    rows=_run_jobs(worker,jobs,workers,phase=phase,output_root=out/'progress'/phase,
        run_fingerprint=r['binding'],timeout_seconds=timeout,on_result=progress,
        failure_result=failed,stop_on_failure=True)
    require(len(rows)==len(jobs) and all(x['status']=='ok' for x in rows),'incomplete phase; inspect preserved output')
    return rows


def preflight():
    r,out=verify()
    with recovery.strict_lock(out,r['binding'],'timing-prefix-preflight'):
        jobs=jobs_for(r)
        require(not (out/'preflight').exists(),'preflight already attempted; inspect')
        rows=execute(r,out,jobs,rt.preflight_worker,'preflight',r['config']['audit_workers'],r['config']['preflight_fuse_seconds'])
        require(sum(x['prefix_steps'] for x in rows)==576,'incomplete 192-by-3 proof')
        run.once(out/'preflight.complete.json',run.sealed(dict(binding=r['binding'],jobs=len(rows),
            states=sum(x['prefix_steps'] for x in rows),files={x['job_id']:run.sha256_file(out/'preflight'/(x['job_id']+'.json')) for x in rows})))
    return dict(verified=len(rows),states=sum(x['prefix_steps'] for x in rows),timing_allowed=True)


def check_complete(r,out,name,jobs):
    proof=run.check_seal(run.read_json(out/(name+'.complete.json')))
    require(proof['binding']==r['binding'] and proof['jobs']==len(jobs) and set(proof['files'])=={j['job_id'] for j in jobs},'phase coverage')
    for j in jobs:
        file=out/name/(j['job_id']+'.json')
        require(run.sha256_file(file)==proof['files'][j['job_id']],'phase proof changed')
    return proof


def read_result(r,out,j):
    row=run.check_seal(run.read_json(out/'episodes'/j['job_id']/'result.json'))
    require(row['binding']==r['binding'] and all(row[k]==j[k] for k in
        ('job_id','pair_id','replica','comparison_arm','budget_seconds')) and
        row['initial_fingerprint']==j['expected_initial'] and row['map_id']==j['case']['map_id'],'timed result identity')
    require(row['policy_sha256']==(j['model']['policy_sha256'] if j['model'] else j['arm']),'model identity')
    for n,h in row['files'].items():
        require(run.sha256_file(run.contained_file(out/'episodes'/j['job_id'],n,field='timed output'))==h,'timed output changed')
    return row


def collect(resume=False):
    r,out=verify()
    require(os.name!='nt','frozen WSL native required')
    jobs=jobs_for(r)
    check_complete(r,out,'preflight',jobs)
    require(run.check_seal(run.read_json(out/'preflight.complete.json'))['states']==576,'shortened prefix proof')
    with recovery.strict_lock(out,r['binding'],'serial-raw-ttf'):
        if resume:(out/'STOP_AFTER_EPISODE').unlink(missing_ok=True)
        else:require(not (out/'collect.status.json').exists(),'explicit resume required')
        if not (out/'environment.json').exists():
            run.once(out/'environment.json',dict(platform=platform.platform(),python=sys.version,
                cpu_count=os.cpu_count(),thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                workers=1,external_interference_not_automatically_detectable=True))
        def status(value,i,**extra):
            body=dict(status=value,completed=i,total=len(jobs),binding=r['binding'],**extra)
            run.write_json(out/'collect.status.json',body)
            run.write_json(out/'run_status.json',body)
        i=0
        try:
            for i,j in enumerate(jobs):
                proof=out/'collect'/(j['job_id']+'.json')
                if proof.exists():
                    require(run.check_seal(run.read_json(proof))['status']=='ok','failed attempt; no automatic retry')
                    read_result(r,out,j)
                    continue
                if (out/'STOP_AFTER_EPISODE').exists():
                    status('paused',i)
                    return dict(paused=True,completed=i,total=len(jobs))
                require(not (out/'episodes'/j['job_id']).exists(),'partial prior attempt; inspect before resume')
                status('running',i,active_job=j['job_id'],arm=j['comparison_arm'])
                print(f"START {i+1}/{len(jobs)} {j['comparison_arm']} {j['pair_id']} r{j['replica']}",flush=True)
                execute(r,out,[j],rt.timed_worker,'collect',1,r['config']['process_fuse_seconds'])
                read_result(r,out,j)
                status('running',i+1)
            files={j['job_id']:run.sha256_file(out/'collect'/(j['job_id']+'.json')) for j in jobs}
            run.once(out/'collect.complete.json',run.sealed(dict(binding=r['binding'],jobs=len(jobs),files=files)))
            status('collected_pending_audit',len(jobs))
        except BaseException as e:
            status('interrupted_or_error',i,error=f'{type(e).__name__}: {e}')
            raise
    return dict(collected=len(jobs),no_training=True)


def audit():
    r,out=verify()
    jobs=jobs_for(r)
    check_complete(r,out,'collect',jobs)
    with recovery.strict_lock(out,r['binding'],'timing-full-audit'):
        require(not (out/'audit').exists(),'audit already attempted; inspect')
        rows=execute(r,out,jobs,rt.audit_worker,'audit',r['config']['audit_workers'],r['config']['audit_fuse_seconds'])
        run.once(out/'audit.complete.json',run.sealed(dict(binding=r['binding'],jobs=len(rows),
            files={x['job_id']:run.sha256_file(out/'audit'/(x['job_id']+'.json')) for x in rows})))
    return dict(audited=len(rows))


def report():
    from experiments.sa_raw_timing_metrics import summarize
    r,out=verify()
    jobs=jobs_for(r)
    check_complete(r,out,'collect',jobs)
    check_complete(r,out,'audit',jobs)
    rows=[read_result(r,out,j) for j in jobs]
    for j in jobs:
        a=run.check_seal(run.read_json(out/'audit'/(j['job_id']+'.json')))
        require(a['result_sha256']==run.sha256_file(out/'episodes'/j['job_id']/'result.json'),'audit result changed')
    result=summarize(rows,bootstrap=r['config']['bootstrap'],seed=r['config']['bootstrap_seed'])
    result.update(schema='lns2.sa_raw_ttf.report.v1',binding=r['binding'],episodes=rows,
                  reused_confirmation_maps=True,no_training=True,no_promotion=True)
    run.once(out/'report.json',run.sealed(result))
    run.write_json(out/'run_status.json',dict(status='complete',completed=len(jobs),total=len(jobs),binding=r['binding']))
    return {k:v for k,v in result.items() if k!='episodes'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','verify','preflight','collect','resume','audit','report','stop','all'))
    args=parser.parse_args()
    if args.phase=='stop':
        out=ROOT/config()['output']
        require(out.exists(),'unregistered timing')
        run.write_json(out/'STOP_AFTER_EPISODE',dict(requested=True))
        result=dict(stop_after_current_episode=True)
    elif args.phase=='verify':
        r,_=verify()
        result=dict(verified=True,inputs=len(r['inputs']),jobs=len(r['jobs']))
    elif args.phase=='resume':result=collect(True)
    elif args.phase=='all':
        result=collect()
        if not result.get('paused'):
            audit()
            result=report()
    else:result=globals()[args.phase]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
