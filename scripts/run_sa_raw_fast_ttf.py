"""Small three-arm serial TTF run, safe stop after an episode, no training."""
import argparse
from collections import Counter
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import platform
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_raw_ttf as source
from scripts import audit_sa_raw_selection_cost as engineering
from scripts import run_sa_onpolicy as run
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_raw_fast_timing as rt
from experiments import sa_raw_timed_runtime as reference

CONFIG = 'configs/sa_raw_fast_ttf.json'
CODE = (CONFIG, 'scripts/run_sa_raw_fast_ttf.py', 'experiments/sa_raw_fast_timing.py',
        'tests/evaluation/test_sa_raw_fast_timing.py', 'docs/SA_RAW_FAST_TTF_PROTOCOL_ZH.md')
require = run.require


def config():
    c = run.read_json(ROOT/CONFIG)
    require(c['workers'] == 1 and c['audit_workers'] == 20 and c['budget_seconds'] == 120. and
            c['max_decisions'] is None and c['execution_node_budget'] is None and
            c['solver_seed'] == 251 and c['replica'] == 0 and c['no_training'] and
            not c['automatic_promotion'], 'frozen timing scope changed')
    return c


def schedule(jobs):
    groups = {}
    for j in jobs:
        g = groups.setdefault((j['pair_id'], j['replica']), {})
        require(j['comparison_arm'] not in g, 'duplicate arm')
        g[j['comparison_arm']] = j
    result = []
    orders = list(itertools.permutations(rt.ARMS))
    for i, (_, g) in enumerate(sorted(groups.items())):
        require(set(g) == set(rt.ARMS), 'missing arm')
        for arm in orders[i % len(orders)]:
            result.append(dict(g[arm], schedule_index=len(result)))
    return result


def selected_jobs(old_jobs, c):
    jobs = []
    for old in old_jobs:
        if old['solver_seed'] != c['solver_seed'] or old['replica'] != c['replica']:
            continue
        names = ('raw_reference', 'raw_fast') if old['comparison_arm'] == 'raw_updated' else (
            ('dual16_sa',) if old['comparison_arm'] == 'dual16_sa' else ())
        for name in names:
            j = deepcopy(old)
            j.update(source_timed_job_id=old['job_id'], comparison_arm=name, output=c['output'],
                     runtime_variant='fast' if name == 'raw_fast' else 'reference',
                     job_id=run.json_fingerprint(['fast-ttf-v1', old['job_id'], name])[:24])
            jobs.append(j)
    jobs = schedule(jobs)
    require(len(jobs) == 36 and len({j['pair_id'] for j in jobs}) == 12, 'expected 12 triplets')
    for arm in rt.ARMS:
        group = [j for j in jobs if j['comparison_arm'] == arm]
        require(sorted(Counter(j['case']['map_id'] for j in group).values()) == [2]*6 and
                Counter(j['case']['task_variant'] for j in group) ==
                {'bottleneck_d20':6, 'bottleneck_d25':6}, 'map/density coverage')
        for map_id in {j['case']['map_id'] for j in group}:
            require({j['case']['task_variant'] for j in group if j['case']['map_id']==map_id} ==
                    {'bottleneck_d20','bottleneck_d25'}, 'unpaired map densities')
    return jobs


def prepare():
    c = config()
    old, src = source.verify()
    er, eo = engineering.verify_registration()
    require(run.sha256_file(src/'report.json') == c['source_report_sha256'] and
            run.sha256_file(eo/'report.json') == c['engineering_report_sha256'], 'source reports changed')
    require(not (ROOT/c['output']).exists(), 'already registered')
    source.check_complete(old, src, 'audit', source.jobs_for(old))
    inputs = old['inputs'] | er['inputs']
    for n in CODE:
        inputs[n] = run.sha256_file(ROOT/n)
    for n in ('registration.json', 'report.json', 'native.complete.json', 'verify.complete.json'):
        inputs[(eo/n).relative_to(ROOT).as_posix()] = run.sha256_file(eo/n)
    for n in ('registration.json', 'report.json', 'audit.complete.json'):
        inputs[(src/n).relative_to(ROOT).as_posix()] = run.sha256_file(src/n)
    body = dict(schema='lns2.sa_raw_fast_ttf.registration.v1', config=c, inputs=inputs,
                jobs=selected_jobs(old['jobs'], c), no_training=True, no_promotion=True,
                source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with recovery.strict_lock(ROOT/c['output'], body['binding'], 'fast-timing-prepare'):
        run.once(ROOT/c['output']/'registration.json', run.sealed(body))
    return dict(episodes=36, pairs=12, workers=1, planning_budget_minutes=72,
                process_fuse_upper_minutes=144, safe_stop_after_episode=True)


def verify():
    c = config()
    out = ROOT/c['output']
    r = run.check_seal(run.read_json(out/'registration.json'))
    require(r['config'] == c and r['binding'] == run.json_fingerprint(
        {k:v for k,v in r.items() if k not in ('binding','integrity')}), 'registration changed')
    for n,h in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,n,field='fast timing input')) == h, 'changed input: '+n)
    old = run.check_seal(run.read_json(ROOT/'build/sa-raw-serial-ttf-v1/registration.json'))
    require(r['jobs'] == selected_jobs(old['jobs'], c), 'schedule/stream identity changed')
    return r,out


def phase_jobs(r, phase):
    jobs = source.jobs_for(r)
    if phase != 'pair':
        return jobs
    by_pair = {}
    for j in jobs:
        by_pair.setdefault(j['pair_id'], {})[j['comparison_arm']] = j
    return [dict(g['raw_fast'], runtime_jobs=[g[a]['job_id'] for a in ('raw_reference','raw_fast')])
            for _, g in sorted(by_pair.items())]


def phase(phase_name):
    r,out = verify()
    jobs = phase_jobs(r, phase_name)
    for prior in (() if phase_name == 'preflight' else ('collect',)):
        source.check_complete(r,out,prior,source.jobs_for(r))
    worker = dict(preflight=rt.preflight_worker,audit=reference.audit_worker,pair=rt.pair_worker)[phase_name]
    with recovery.strict_lock(out,r['binding'],'fast-'+phase_name):
        require(not (out/phase_name).exists(),'phase already attempted; inspect')
        rows = source.execute(r,out,jobs,worker,phase_name,20,
                              r['config']['preflight_fuse_seconds' if phase_name=='preflight' else 'audit_fuse_seconds'])
        if phase_name == 'preflight':
            require(sum(x['prefix_steps'] for x in rows)==108,'shortened native prefix proof')
        run.once(out/(phase_name+'.complete.json'),run.sealed(dict(binding=r['binding'],jobs=len(rows),
            files={j['job_id']:run.sha256_file(out/phase_name/(j['job_id']+'.json')) for j in jobs})))
    return dict(phase=phase_name,verified=len(rows))


def collect(resume=False):
    r,out = verify()
    require(os.name != 'nt','use frozen WSL native')
    jobs = source.jobs_for(r)
    source.check_complete(r,out,'preflight',jobs)
    with recovery.strict_lock(out,r['binding'],'fast-serial-ttf'):
        if resume:
            (out/'STOP_AFTER_EPISODE').unlink(missing_ok=True)
        else:
            require(not (out/'collect.status.json').exists(),'explicit resume required')
        if not (out/'environment.json').exists():
            run.once(out/'environment.json',dict(platform=platform.platform(),python=sys.version,
                workers=1,thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                external_interference_not_automatically_detectable=True))
        def status(value,completed,**extra):
            body=dict(status=value,completed=completed,total=len(jobs),binding=r['binding'],**extra)
            run.write_json(out/'collect.status.json',body)
            run.write_json(out/'run_status.json',body)
        i=0
        try:
            for i,j in enumerate(jobs):
                proof=out/'collect'/(j['job_id']+'.json')
                if proof.exists():
                    require(run.check_seal(run.read_json(proof))['status']=='ok','failed attempt; no automatic retry')
                    source.read_result(r,out,j)
                    continue
                if (out/'STOP_AFTER_EPISODE').exists():
                    status('paused',i)
                    return dict(paused=True,completed=i,total=len(jobs))
                require(not (out/'episodes'/j['job_id']).exists(),'partial attempt; inspect before resume')
                status('running',i,active_job=j['job_id'],arm=j['comparison_arm'])
                print(f"START {i+1}/{len(jobs)} {j['comparison_arm']} {j['pair_id']}",flush=True)
                source.execute(r,out,[j],rt.timed_worker,'collect',1,r['config']['process_fuse_seconds'])
                source.read_result(r,out,j)
                status('running',i+1)
            run.once(out/'collect.complete.json',run.sealed(dict(binding=r['binding'],jobs=len(jobs),
                files={j['job_id']:run.sha256_file(out/'collect'/(j['job_id']+'.json')) for j in jobs})))
            status('collected_pending_audit',len(jobs))
        except BaseException as e:
            status('interrupted_or_error',i,error=f'{type(e).__name__}: {e}')
            raise
    return dict(collected=len(jobs))


def report():
    r,out=verify()
    jobs=source.jobs_for(r)
    for p in ('collect','audit','pair'):
        source.check_complete(r,out,p,phase_jobs(r,p))
    rows=[source.read_result(r,out,j) for j in jobs]
    for j in jobs:
        audit=run.check_seal(run.read_json(out/'audit'/(j['job_id']+'.json')))
        require(audit['result_sha256']==run.sha256_file(out/'episodes'/j['job_id']/'result.json'),'stale audit')
    for j in phase_jobs(r,'pair'):
        audit=run.check_seal(run.read_json(out/'pair'/(j['job_id']+'.json')))
        require(audit['result_hashes']==[run.sha256_file(out/'episodes'/x/'result.json') for x in j['runtime_jobs']], 'stale pair audit')
    result=rt.summarize(rows,r['config']['bootstrap'],r['config']['bootstrap_seed'])
    result.update(schema='lns2.sa_raw_fast_ttf.report.v1',binding=r['binding'],episodes=rows)
    run.once(out/'report.json',run.sealed(result))
    run.write_json(out/'run_status.json',dict(status='complete',completed=len(jobs),total=len(jobs),binding=r['binding']))
    return {k:v for k,v in result.items() if k not in ('episodes','by_map')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','verify','preflight','collect','resume','audit','pair','report','stop','all'))
    args=parser.parse_args()
    if args.phase=='stop':
        out=ROOT/config()['output']
        require(out.exists(),'unregistered timing')
        run.write_json(out/'STOP_AFTER_EPISODE',dict(requested=True))
        result=dict(stop_after_current_episode=True)
    elif args.phase=='verify':
        r,_=verify()
        result=dict(inputs=len(r['inputs']),episodes=len(r['jobs']))
    elif args.phase in ('preflight','audit','pair'):
        result=phase(args.phase)
    elif args.phase=='resume':
        result=collect(True)
    elif args.phase=='all':
        result=collect()
        if not result.get('paused'):
            phase('audit')
            phase('pair')
            result=report()
    else:
        result=globals()[args.phase]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
