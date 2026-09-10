"""Frozen same-state PP subset diagnostic; no saved witness paths as actions."""
from __future__ import annotations

from collections import Counter
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.dont_write_bytecode = True
from scripts.audit_preentry_trace import REVIEW, read, sha, native_path
from scripts.audit_recovery_provenance import write
from scripts.audit_witness_path_ablation import load_states, full_edges
from experiments.repair_collection import state_fingerprint

OUT = ROOT/'build/initlns-compressed-neighborhood-native-v1'
CASE = 'case-6542dfb0660a'
ARMS = ('original16','compressed12','compressed11')


def seed(label, trial):
    import hashlib
    return int(hashlib.sha256(f'compressed-native-v1/20260914/{label}/{trial}'.encode()).hexdigest()[:8],16) % (2**31)


def make_jobs(sets):
    jobs = []
    for trial in range(8):
        order = list(sets['original16'])
        random.Random(seed('order',trial)).shuffle(order)
        for arm in ARMS:
            jobs.append(dict(id=f'{trial:02d}-{arm}', arm=arm, trial=trial,
                action=dict(mode='explicit_neighborhood', agents=sets[arm],
                    repair_order=[a for a in order if a in sets[arm]],
                    random_seed=seed('action',trial), pp_random_seed=seed('pp',trial))))
    return jobs


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT/'plan.json').exists():
        raise RuntimeError('Plan exists; use verify/qualify/collect/analyze, do not reprepare')
    e = read(ROOT/'artifacts/initlns-witness-path-ablation-v1/evidence.json')
    assert sha(ROOT/e['report_file']) == e['report_sha256']
    earlier = read(ROOT/'artifacts/initlns-preentry-trace-audit-v1/evidence.json')
    assert sha(ROOT/earlier['report_file']) == earlier['report_sha256']
    detail = next(c for c in read(ROOT/earlier['report_file'])['cases'] if c['case_id']==CASE)
    before,after,event = load_states(detail)
    original_plan = read(REVIEW/'continuation-diagnostic-v2/plan.json')
    case = next(c for c in original_plan['cases'] if c['case_id']==CASE)
    conf = read(ROOT/'configs/path_quality_pressure_deadline_recovery_v1.json')
    prefix = []
    with gzip.open(native_path(ROOT/detail['source_trace']),'rt',encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            if r['event']=='transition' and r['decision_index']<event['decision_index']:
                prefix.append(dict(action=r['action'], before=r['before_fingerprint'],after=r['after_fingerprint'],
                    order=r['metrics']['repair_order'], budget=r['metrics']['requested_pp_time_limit_seconds']))
    assert len(prefix)==253 and before['num_of_colliding_pairs']==42
    sets = dict(original16=sorted(event['action']['agents']),
        compressed12=e['dual16_minimum_paths_for_exact_post_edges'],
        compressed11=e['dual16_minimum_paths_for_34_conflicts'])
    assert set(sets['compressed11']) < set(sets['compressed12']) < set(sets['original16'])
    for name,value in (('before',before),('historical_after',after),('historical_event',event),('prefix',prefix)):
        write(OUT/(name+'.json'),value)
    files = [case[k] for k in ('map_file','scenario_file','source_trace')]
    for path in files+[conf['native_file']]:
        assert sha(ROOT/path)==original_plan['files'][path]
    files += [conf['native_file'], 'scripts/probe_compressed_neighborhood.py',
        'scripts/audit_preentry_trace.py','scripts/audit_recovery_provenance.py',
        'scripts/audit_witness_path_ablation.py','experiments/state_analysis.py',
        'experiments/repair_collection.py','docs/COMPRESSED_NEIGHBORHOOD_NATIVE_PROTOCOL_ZH.md',
        'tests/test_compressed_neighborhood_probe.py',e['report_file'],earlier['report_file']]
    files += [(OUT/(n+'.json')).relative_to(ROOT).as_posix() for n in ('before','historical_after','historical_event','prefix')]
    plan = dict(schema='lns2.compressed_neighborhood_native.v1',case=case,sets=sets,
        native_file=conf['native_file'], native_sha256=conf['native_sha256'],
        source_fingerprint=event['before_fingerprint'], jobs=make_jobs(sets),
        workers=20, process_fuse_seconds=300, pp_seconds=10,
        controlled_relative_order=True, trials=8, files={p:sha(ROOT/p) for p in files},
        gate='At least 6/8 outcomes <=34 conflicts and mean remaining conflicts no worse than original16; zero censoring or errors. Mechanism signal only.')
    write(OUT/'plan.json',plan)
    print(json.dumps(dict(jobs=24,prefix_steps_per_job=253,workers=20,plan_sha256=sha(OUT/'plan.json'))),flush=True)


def verify():
    registry = read(ROOT/'artifacts/initlns-compressed-neighborhood-native-v1/registration.json')
    assert sha(OUT/'plan.json')==registry['plan_sha256']
    plan=read(OUT/'plan.json')
    assert plan['jobs']==make_jobs(plan['sets'])
    for path,digest in plan['files'].items():
        assert sha(ROOT/path)==digest, 'bound file changed: '+path
    return plan


def check_result(result, job):
    assert result['status']=='ok' and result['job']==job
    assert sha(ROOT/result['state_file'])==result['state_sha256']
    state=read(ROOT/result['state_file']); before=read(OUT/'before.json')
    old={a['id']:a for a in before['agents']}; current={a['id']:a for a in state['agents']}
    assert len(current)==len(state['agents'])==len(old) and old.keys()==current.keys()
    assert state_fingerprint(state)==result['after_fingerprint']
    changed=[]
    for aid,a in current.items():
        path=a['path']; cols=state['cols']
        assert path and path[0]==old[aid]['start'] and path[-1]==old[aid]['goal']
        assert all(0<=p<len(state['obstacles']) and not state['obstacles'][p] for p in path)
        assert all(abs(x//cols-y//cols)+abs(x%cols-y%cols)<=1 for x,y in zip(path,path[1:]))
        if path!=old[aid]['path']: changed.append(aid)
    assert set(changed)<=set(job['action']['agents'])
    edges=full_edges({i:a['path'] for i,a in current.items()})
    m=result['metrics']
    assert edges=={tuple(p) for p in state['conflict_edges']}
    assert len(edges)==state['num_of_colliding_pairs']==m['conflicts_after']
    assert sum(len(a['path'])-1 for a in current.values())==state['sum_of_costs']
    assert m['action_valid'] and m['step_applied'] and m['conflicts_before']==42
    assert sorted(m['neighborhood'])==sorted(job['action']['agents'])
    if 'repair_order' in job['action']:
        assert m['repair_order']==job['action']['repair_order']
    if m['pp_rolled_back']:
        assert not changed and edges=={tuple(p) for p in before['conflict_edges']}


def child(job_id):
    plan=verify(); historical=job_id=='historical'
    event=read(OUT/'historical_event.json')
    job=dict(id='historical',arm='historical',action=event['action']) if historical else next(j for j in plan['jobs'] if j['id']==job_id)
    case=plan['case']; native=ROOT/plan['native_file']
    sys.path.insert(0,str(native.parent)); import lns2_env
    assert Path(lns2_env.__file__).resolve()==native.resolve()
    env=lns2_env.LNS2RepairEnv(str(ROOT/case['map_file']),str(ROOT/case['scenario_file']),
        case['agent_count'],time_limit=3600,neighborhood_size=8,destroy_strategy='Adaptive',
        replan_algorithm='PP',use_sipp=True,max_repair_iterations=0,screen=0)
    state=env.reset(case['solver_seed'])
    for p in read(OUT/'prefix.json'):
        assert state_fingerprint(state)==p['before']
        step=env.step_with_time_limit(p['action'],p['budget']); state=step['observation']
        assert state_fingerprint(state)==p['after'] and step['metrics']['repair_order']==p['order']
    assert state_fingerprint(state)==plan['source_fingerprint']
    budget=event['metrics']['requested_pp_time_limit_seconds'] if historical else plan['pp_seconds']
    step=env.step_with_time_limit(job['action'],budget); state=step['observation']; m=step['metrics']
    if historical:
        assert state_fingerprint(state)==event['after_fingerprint']
        for key in ('repair_order','neighborhood','conflicts_after','replan_success','pp_rolled_back'):
            assert m[key]==event['metrics'][key]
    output=OUT/(job_id+'-state.json'); write(output,state)
    before=read(OUT/'before.json')
    result=dict(job=job,status='ok',prefix_steps=253,source_fingerprint=plan['source_fingerprint'],
        after_fingerprint=state_fingerprint(state),metrics=m,
        low_level_delta={k:state['low_level'][k]-before['low_level'][k] for k in state['low_level']},
        state_file=output.relative_to(ROOT).as_posix(),state_sha256=sha(output))
    check_result(result,job)
    write(OUT/(job_id+'.json'),result)


def qualify():
    verify()
    if (OUT/'historical.json').exists():
        raise RuntimeError('Historical result already exists; do not overwrite')
    with (OUT/'historical.log').open('w') as log:
        subprocess.run([sys.executable,'-B',__file__,'_child','historical'],stdout=log,stderr=subprocess.STDOUT,
            timeout=300,check=True)
    r=read(OUT/'historical.json'); check_result(r,r['job'])
    write(OUT/'qualification.json',dict(passed=True,historical_sha256=sha(OUT/'historical.json')))
    print('Historical full16 action and after-state reproduced',flush=True)


def collect():
    plan=verify(); qualification=read(OUT/'qualification.json')
    assert qualification['passed'] and sha(OUT/'historical.json')==qualification['historical_sha256']
    lock=OUT/'run.lock'; fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.close(fd)
    active=[]; completed=[]; pending=[]
    def status(kind,error=None):
        write(OUT/'run_status.json',dict(status=kind,total=24,completed=len(completed),active=len(active),error=error))
    try:
        for job in plan['jobs']:
            path=OUT/(job['id']+'.json')
            if path.exists(): check_result(read(path),job); completed.append(job['id'])
            else: pending.append(job)
        while pending or active:
            while pending and len(active)<20 and not (OUT/'STOP_AFTER_JOB').exists():
                job=pending.pop(0); log=(OUT/(job['id']+'.log')).open('w')
                proc=subprocess.Popen([sys.executable,'-B',__file__,'_child',job['id']],stdout=log,stderr=subprocess.STDOUT)
                active.append((proc,log,time.monotonic(),job))
            for item in list(active):
                proc,log,start,job=item
                if time.monotonic()-start>300: raise TimeoutError(job['id'])
                if proc.poll() is None: continue
                log.close(); active.remove(item)
                if proc.returncode: raise RuntimeError('child failed: '+job['id'])
                check_result(read(OUT/(job['id']+'.json')),job)
                completed.append(job['id']); print(json.dumps(dict(completed=len(completed),job=job['id'])),flush=True)
            status('running')
            if not active and (OUT/'STOP_AFTER_JOB').exists(): break
            time.sleep(0.3)
        status('complete' if len(completed)==24 else 'paused')
    except BaseException as exc:
        status('error',repr(exc)); raise
    finally:
        for proc,log,_,_ in active:
            if proc.poll() is None: proc.kill()
            proc.wait(); log.close()
        lock.unlink()


def analyze():
    plan=verify(); results=[]
    for job in plan['jobs']:
        r=read(OUT/(job['id']+'.json')); check_result(r,job); results.append(r)
    summaries={}
    for arm in ARMS:
        rows=[r for r in results if r['job']['arm']==arm]; m=[r['metrics'] for r in rows]
        values=[x['conflicts_after'] for x in m]
        summaries[arm]=dict(conflicts=values,mean=sum(values)/8,at_most_34=sum(v<=34 for v in values),
            strict_improvements=sum(v<42 for v in values),rollbacks=sum(x['pp_rolled_back'] for x in m),
            reasons=dict(Counter(x['pp_failure_reason'] for x in m)),
            failed_agents=dict(Counter(str(x['pp_failed_agent']) for x in m if x['pp_failed_agent']>=0)),
            node_deltas=[r['low_level_delta'] for r in rows])
    censored=any('time' in str(r['metrics']['pp_failure_reason']).lower() for r in results)
    passed=[a for a in ARMS[1:] if not censored and summaries[a]['at_most_34']>=6
            and summaries[a]['mean']<=summaries['original16']['mean']]
    report=dict(schema='lns2.compressed_neighborhood_native_result.v1',summaries=summaries,
        decision='bounded_mechanism_signal_only' if passed else 'no_go' if not censored else 'inconclusive_budget',
        passed_arms=passed,censored=censored,errors=0,plan_sha256=sha(OUT/'plan.json'),
        result_sha256={r['job']['id']:sha(OUT/(r['job']['id']+'.json')) for r in results},
        qualification_sha256=sha(OUT/'qualification.json'),
        boundary='One hindsight-selected successful source. Controlled common relative order, not default official shuffle or TTF evidence.')
    write(OUT/'report.json',report); print(json.dumps(report),flush=True)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='_child': child(sys.argv[2])
    else: {'prepare':prepare,'verify':verify,'qualify':qualify,'collect':collect,'analyze':analyze}[mode]()
