"""Finish the frozen raw-fast/Dual16 cohort without rerunning completed pairs."""
import argparse
from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_raw_fast_ttf as first
from scripts import run_sa_raw_ttf as source
from scripts import run_sa_onpolicy as run
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_raw_fast_timing as rt
from experiments.sa_raw_selection_fast import _bind

ARMS = ('raw_fast', 'dual16_sa')
CONFIG = 'configs/sa_raw_fast_remaining_ttf.json'
CODE = (CONFIG, 'scripts/run_sa_raw_fast_remaining.py',
        'tests/evaluation/test_sa_raw_fast_remaining.py', 'docs/SA_RAW_FAST_REMAINING_PROTOCOL_ZH.md')
require = run.require
# Reuse the same metrics with two real arms, never manufacture a third arm.
summarize = _bind(rt.summarize, ARMS=ARMS, _validate=_bind(rt._validate, ARMS=ARMS))


def config():
    c = run.read_json(ROOT/CONFIG)
    require(c['conditions'] == [[251,1],[257,0],[257,1]] and c['workers'] == 1 and
            c['audit_workers'] == 20 and c['budget_seconds'] == 120. and
            c['process_fuse_seconds'] == 240. and c['preflight_steps'] == 3 and
            c['preflight_fuse_seconds'] == 300. and c['audit_fuse_seconds'] == 900. and
            c['bootstrap'] == 5000 and c['bootstrap_seed'] == 2026092607 and
            c['max_decisions'] is None and c['execution_node_budget'] is None and
            c['no_training'] and not c['automatic_promotion'], 'remaining scope changed')
    return c


def schedule(jobs):
    groups = {}
    for j in jobs:
        g = groups.setdefault((j['pair_id'],j['replica']),{})
        require(j['comparison_arm'] not in g, 'duplicate arm')
        g[j['comparison_arm']] = j
    result = []
    for i, (_, g) in enumerate(sorted(groups.items())):
        require(set(g) == set(ARMS), 'missing arm')
        for arm in ARMS if i%2 == 0 else ARMS[::-1]:
            result.append(dict(g[arm],schedule_index=len(result)))
    return result


def selected_jobs(old_jobs, c):
    jobs = []
    for old in old_jobs:
        if [old['solver_seed'],old['replica']] not in c['conditions']:
            continue
        if old['comparison_arm'] not in ('raw_updated','dual16_sa'):
            continue
        j = deepcopy(old)
        arm = 'raw_fast' if old['comparison_arm']=='raw_updated' else 'dual16_sa'
        j.update(source_timed_job_id=old['job_id'],comparison_arm=arm,output=c['output'],
                 runtime_variant='fast' if arm=='raw_fast' else 'reference',
                 job_id=run.json_fingerprint(['remaining-fast-ttf-v1',old['job_id'],arm])[:24])
        jobs.append(j)
    jobs = schedule(jobs)
    require(len(jobs)==72 and len({(j['pair_id'],j['replica']) for j in jobs})==36, 'expected 36 pairs')
    for arm in ARMS:
        group = [j for j in jobs if j['comparison_arm']==arm]
        require(sorted(Counter(j['case']['map_id'] for j in group).values())==[6]*6, 'six maps')
        for m in {j['case']['map_id'] for j in group}:
            actual = {(j['case']['task_variant'],j['solver_seed'],j['replica']) for j in group if j['case']['map_id']==m}
            expected = {(v,s,r) for v in ('bottleneck_d20','bottleneck_d25') for s,r in c['conditions']}
            require(actual==expected, 'map/task/seed coverage')
    return jobs


def result_key(row):
    return row['pair_id'],row['replica'],row['comparison_arm']


def first_rows(c):
    path=ROOT/c['first_batch']/'report.json'
    require(run.sha256_file(path)==c['first_report_sha256'],'first report changed')
    report=run.check_seal(run.read_json(path))
    rows=[r for r in report['episodes'] if r['comparison_arm'] in ARMS]
    require(len(rows)==24 and all(r['solver_seed']==251 and r['replica']==0 for r in rows),'first cohort changed')
    return rows


def combine_rows(previous, remaining):
    old_keys={result_key(r) for r in previous}
    new_keys={result_key(r) for r in remaining}
    require(len(old_keys)==len(previous)==24 and len(new_keys)==len(remaining)==72 and
            old_keys.isdisjoint(new_keys), 'duplicate or incomplete cohort')
    rows=previous+remaining
    for r in rows:
        require(r['comparison_arm'] in ARMS,'unexpected third arm')
    for arm in ARMS:
        group=[r for r in rows if r['comparison_arm']==arm]
        require(len({r['map_id'] for r in group})==6,'map coverage')
        for m in {r['map_id'] for r in group}:
            tasks={r['task_id'] for r in group if r['map_id']==m}
            require(len(tasks)==2,'two tasks per map')
            for t in tasks:
                require({(r['solver_seed'],r['replica']) for r in group if r['map_id']==m and r['task_id']==t}
                        =={(251,0),(251,1),(257,0),(257,1)},'full frozen cohort coverage')
    return rows


def prepare():
    c=config()
    previous,previous_out=first.verify()
    require(previous_out==ROOT/c['first_batch'],'first batch root')
    old=run.check_seal(run.read_json(ROOT/'build/sa-raw-serial-ttf-v1/registration.json'))
    old_out=ROOT/old['config']['output']
    source.check_complete(old,old_out,'preflight',source.jobs_for(old))
    source.check_complete(previous,previous_out,'audit',source.jobs_for(previous))
    jobs=selected_jobs(old['jobs'],c)
    combine_rows(first_rows(c),[dict(j,map_id=j['case']['map_id'],task_id=j['case']['task_id']) for j in jobs])
    out=ROOT/c['output']
    require(not out.exists(),'already registered; verify/resume')
    inputs=dict(previous['inputs'])
    for n in CODE:
        inputs[n]=run.sha256_file(ROOT/n)
    for n in ('registration.json','report.json','preflight.complete.json','collect.complete.json','audit.complete.json','pair.complete.json'):
        inputs[(previous_out/n).relative_to(ROOT).as_posix()]=run.sha256_file(previous_out/n)
    reusable={}
    for j in jobs:
        if j['comparison_arm']!='dual16_sa':
            continue
        p=old_out/'preflight'/(j['source_timed_job_id']+'.json')
        relative=p.relative_to(ROOT).as_posix()
        inputs[relative]=run.sha256_file(p)
        reusable[j['job_id']]=dict(path=relative,binding=old['binding'],job_id=j['source_timed_job_id'])
    body=dict(schema='lns2.sa_raw_fast_remaining.registration.v1',config=c,inputs=inputs,jobs=jobs,
              reusable_preflight=reusable,first_binding=previous['binding'],no_training=True,no_promotion=True,
              source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    body['binding']=run.json_fingerprint(body)
    with recovery.strict_lock(out,body['binding'],'remaining-prepare'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(new_episodes=72,new_pairs=36,reused_pairs=12,workers=1,
                new_native_preflight_steps=108,reused_baseline_preflights=36,
                planning_budget_upper_minutes=144,process_fuse_upper_minutes=288)


def verify():
    c=config()
    out=ROOT/c['output']
    r=run.check_seal(run.read_json(out/'registration.json'))
    require(r['config']==c and r['binding']==run.json_fingerprint(
        {k:v for k,v in r.items() if k not in ('binding','integrity')}),'registration changed')
    for n,h in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,n,field='remaining input'))==h,'changed input: '+n)
    old=run.check_seal(run.read_json(ROOT/'build/sa-raw-serial-ttf-v1/registration.json'))
    require(r['jobs']==selected_jobs(old['jobs'],c),'schedule/model/stream changed')
    require(set(r['reusable_preflight'])=={j['job_id'] for j in r['jobs'] if j['comparison_arm']=='dual16_sa'},'reuse coverage')
    first_rows(c)
    return r,out


def reused_preflight(job, r):
    require(job['comparison_arm']=='dual16_sa' and job['model'] is None,'only frozen baseline may reuse proof')
    identity=r['reusable_preflight'][job['job_id']]
    path=run.contained_file(ROOT,identity['path'],field='reused proof')
    require(run.sha256_file(path)==r['inputs'][identity['path']],'changed prefix proof')
    proof=run.check_seal(run.read_json(path))
    require(proof['binding']==identity['binding'] and proof['job_id']==identity['job_id']==job['source_timed_job_id'] and
            proof['status']=='ok' and proof['prefix_steps']==3 and proof['action_feature_path_equal'],'invalid reused preflight')
    return dict(status='ok',job_id=job['job_id'],prefix_steps=3,reused=True,native_steps_executed=0,
                proof_path=identity['path'],proof_sha256=r['inputs'][identity['path']])


def preflight_worker(job):
    require(job['comparison_arm']=='raw_fast','unexpected new prefix execution')
    row=rt.preflight_worker(job)
    return dict(row,reused=False,native_steps_executed=row['prefix_steps'])


def phase(name):
    require(name in ('preflight','audit'),'unknown phase')
    r,out=verify()
    jobs=source.jobs_for(r)
    if name=='audit':source.check_complete(r,out,'collect',jobs)
    with recovery.strict_lock(out,r['binding'],'remaining-'+name):
        require(not (out/name).exists(),'phase already attempted; inspect')
        if name=='preflight':
            reused=[reused_preflight(j,r) for j in jobs if j['comparison_arm']=='dual16_sa']
            for row in reused:
                run.once(out/name/(row['job_id']+'.json'),run.sealed(dict(row,binding=r['binding'])))
            new=source.execute(r,out,[j for j in jobs if j['comparison_arm']=='raw_fast'],preflight_worker,
                               name,20,r['config']['preflight_fuse_seconds'])
            require(len(reused)==36 and sum(x['native_steps_executed'] for x in new)==108,'incomplete preflight')
        else:
            source.execute(r,out,jobs,first.reference.audit_worker,name,20,r['config']['audit_fuse_seconds'])
        run.once(out/(name+'.complete.json'),run.sealed(dict(binding=r['binding'],jobs=len(jobs),
            files={j['job_id']:run.sha256_file(out/name/(j['job_id']+'.json')) for j in jobs})))
    return dict(phase=name,verified=len(jobs))


# The collector, clock and pause/resume behavior are identical to the first batch.
collect=_bind(first.collect,verify=verify)


def report():
    r,out=verify()
    jobs=source.jobs_for(r)
    for name in ('collect','audit'):source.check_complete(r,out,name,jobs)
    rows=[source.read_result(r,out,j) for j in jobs]
    for j in jobs:
        proof=run.check_seal(run.read_json(out/'audit'/(j['job_id']+'.json')))
        require(proof['result_sha256']==run.sha256_file(out/'episodes'/j['job_id']/'result.json'),'stale audit')
    prior=first_rows(r['config'])
    combined=combine_rows(prior,rows)
    def summary(rs):return summarize(rs,r['config']['bootstrap'],r['config']['bootstrap_seed'])
    body=dict(schema='lns2.sa_raw_fast_remaining.report.v1',binding=r['binding'],
              remaining=summary(rows),combined=summary(combined),previous=summary(prior),episodes=rows,
              by_condition={f's{s}-r{rep}':summary([x for x in combined if x['solver_seed']==s and x['replica']==rep])
                            for s in (251,257) for rep in (0,1)},
              by_density={v:summary([x for x in combined if x['task_id'] in
                 {j['case']['task_id'] for j in jobs if j['case']['task_variant']==v}])
                 for v in ('bottleneck_d20','bottleneck_d25')},
              first_report_sha256=r['config']['first_report_sha256'],no_training=True,no_promotion=True,
              combined_is_multi_session_development_evidence=True,no_new_maps=True)
    run.once(out/'report.json',run.sealed(body))
    run.write_json(out/'run_status.json',dict(status='complete',completed=72,total=72,binding=r['binding']))
    return dict(completed=72,total_paired_conditions=48,report=(out/'report.json').relative_to(ROOT).as_posix())


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
        result=dict(inputs=len(r['inputs']),episodes=len(r['jobs']))
    elif args.phase in ('preflight','audit'):result=phase(args.phase)
    elif args.phase=='resume':result=collect(True)
    elif args.phase=='all':
        result=collect()
        if not result.get('paused'):
            phase('audit')
            result=report()
    else:result=globals()[args.phase]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
