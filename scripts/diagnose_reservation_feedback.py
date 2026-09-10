"""Outcome-blind feedback selection with true-occupancy acceptance guards."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
from experiments._common import read_json, write_json
from experiments.native_path_compatibility import modules, paths_of, edge_set, seal, check_seal
from experiments.full_neighborhood_recovery import (
    SearchBudget, run_sequence, scientific_signature, check_paths, needs_recovery,
)
from scripts.diagnose_reservation_mediation import released_pp, sha256_file, require

OUT=ROOT/'build/initlns-reservation-feedback-v1'
SOURCE=ROOT/'build/initlns-full-neighborhood-recovery-v1-final'
REG=ROOT/'artifacts/initlns-reservation-feedback-v1/registration.json'
METHODS=('fresh_retry','directed_release','random_release')


def stable_seed(label):
    return int(hashlib.sha256(('reservation-feedback-v1/'+label).encode()).hexdigest()[:8],16) % (2**31)


def feedback_ranking(base, order):
    selected=set(order); counts=Counter(); victims=defaultdict(set); earliest={}
    for record in base['records']:
        for event in record['incident_events']:
            left,right=event['left'],event['right']
            if (left in selected)==(right in selected): continue
            other=right if left in selected else left
            victim=left if left in selected else right
            counts[other]+=1; victims[other].add(victim)
            earliest[other]=min(earliest.get(other,event['time']),event['time'])
    ids=sorted(counts,key=lambda i:(-len(victims[i]),-counts[i],earliest[i],i))
    return [dict(agent=i,victims=sorted(victims[i]),events=counts[i],first_time=earliest[i]) for i in ids]


def choices(base, order, key):
    ranked=feedback_ranking(base,order)
    return dict(ranked=ranked,directed=ranked[0]['agent'] if ranked else None,
                random=random.Random(stable_seed(key+'/selection')).choice(sorted(r['agent'] for r in ranked)) if ranked else None,
                retry_seed=stable_seed(key+'/pp'))


def prepare():
    m=read_json(SOURCE/'manifest.json'); check_seal(m)
    report=read_json(SOURCE/'report.json'); check_seal(report)
    ev=read_json(ROOT/'artifacts/initlns-full-neighborhood-recovery-v1/evidence.json')
    require(sha256_file(SOURCE/'report.json')==ev['files']['report.json'],'source report SHA')
    require(sha256_file(SOURCE/'manifest.json')==ev['files']['manifest.json'],'source manifest SHA')
    require(m['content_sha256']==ev['manifest_content_sha256'],'source manifest identity')
    first=read_json(ROOT/'artifacts/initlns-reservation-mediation-v1/evidence.json')
    require(sha256_file(ROOT/first['round2_report'])==first['round2_sha256'],'round2 changed')
    require(read_json(ROOT/first['round2_report'])['passed'],'posthoc opportunity gate')
    jobs=[]; files={}
    for c in m['cases']:
        order=read_json(ROOT/c['historical_transition_file'])['metrics']['repair_order']
        for s in m['config']['pp_seeds']:
            old=next(j for j in m['jobs'] if j['case_id']==c['case_id'] and j['seed']==s and j['method']=='ordinary')
            path=(SOURCE/'results'/(old['job_id']+'.json')).relative_to(ROOT).as_posix()
            require(sha256_file(ROOT/path)==report['result_hashes'][path],'source row SHA')
            r=read_json(ROOT/path); check_seal(r)
            require(r['status']=='ok' and r['job']==old and not r['budget_exhausted'],'source condition')
            key=f"{c['case_id']}/{s}"
            jobs.append(dict(id=hashlib.sha256(key.encode()).hexdigest()[:16],case=c,seed=s,
                             order=order,source_result=path,selection=choices(r['base'],order,key)))
            files[path]=report['result_hashes'][path]
        for k in ('state_file','map_file','scenario_file','historical_transition_file'):
            p=c[k]; require(sha256_file(ROOT/p)==m['files'][p], 'source file changed: '+p); files[p]=m['files'][p]
    require(len(jobs)==48 and len(m['cases'])==24,'source counts')
    code=['scripts/diagnose_reservation_feedback.py','tests/test_reservation_feedback.py',
          'docs/RESERVATION_FEEDBACK_PROTOCOL_ZH.md','scripts/diagnose_reservation_mediation.py',
          'experiments/full_neighborhood_recovery.py','experiments/native_path_compatibility.py',
          'experiments/local_path_compatibility.py','experiments/local_path_search.py',
          'experiments/state_analysis.py','experiments/_common.py']
    code += [m['config'][k] for k in ('native_file','probe_file','core_file')]
    for p in code: files[p]=sha256_file(ROOT/p)
    for k in ('native_file','probe_file','core_file'):
        require(files[m['config'][k]]==m['files'][m['config'][k]],'source binary changed')
    for p in (SOURCE/'report.json',SOURCE/'manifest.json',ROOT/first['round2_report']):
        files[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    plan=dict(schema='lns2.reservation_feedback.v1',jobs=jobs,files=files,workers=20,
              job_fuse=180,config=m['config'],scope='24_viewed_development_states_no_ttf')
    OUT.mkdir(exist_ok=True); require(not (OUT/'plan.json').exists(),'existing plan')
    write_json(OUT/'plan.json',plan)
    triggered=sum(needs_recovery(read_json(ROOT/j['source_result'])['base']) for j in jobs)
    usable=sum(needs_recovery(read_json(ROOT/j['source_result'])['base']) and bool(j['selection']['ranked']) for j in jobs)
    print(json.dumps(dict(jobs=48,triggered=triggered,feedback_usable=usable,
                         maximum_new_branches=triggered+2*usable,plan_sha256=sha256_file(OUT/'plan.json'))))


def load():
    require(sha256_file(OUT/'plan.json')==read_json(REG)['plan_sha256'],'plan registration')
    m=read_json(OUT/'plan.json')
    for p,h in m['files'].items(): require(sha256_file(ROOT/p)==h,'input changed: '+p)
    return m


def normalize(r,state):
    if r['status'] in ('accepted','rolled_back'):
        return dict(status='ok',rolled_back=r['status']=='rolled_back',paths=r['paths'],
                    attempted_pairs=r['attempted_pairs'],
                    diagnostics=[dict(agent=x['agent'],**x['search']) for x in r['records']])
    return r


def worker(job_id):
    m=load(); j=next(j for j in m['jobs'] if j['id']==job_id); c=j['case']
    state=read_json(ROOT/c['state_file']); source=read_json(ROOT/j['source_result'])
    require(choices(source['base'],j['order'],f"{c['case_id']}/{j['seed']}")==j['selection'],'selection mismatch')
    native, module=modules(m['config'])
    def probe():
        return module.NativePathProbe(str(ROOT/c['map_file']),str(ROOT/c['scenario_file']),paths_of(state))
    base=run_sequence(probe(),state,j['order'],j['seed'],SearchBudget(30,5))
    require(scientific_signature(base)==scientific_signature(source['base']),'baseline signature mismatch')
    require(base['status'] in ('accepted','rolled_back'),'baseline unknown')
    base_count=len(edge_set(dict(enumerate(base['paths'])))); branches={}; triggered=needs_recovery(base)
    env=native.LNS2RepairEnv(str(ROOT/c['map_file']),str(ROOT/c['scenario_file']),len(state['agents']),time_limit=60)
    for method in METHODS:
        chosen=j['selection']['directed' if method=='directed_release' else 'random'] if method!='fresh_retry' else None
        applicable=triggered and (method=='fresh_retry' or chosen is not None)
        if not applicable:
            branches[method]=dict(applicable=False,final_conflicts=base_count,recovered=False,
                                  reason='not_triggered' if not triggered else 'no_external_feedback')
            continue
        if method=='fresh_retry':
            raw=normalize(run_sequence(probe(),state,j['order'],j['selection']['retry_seed'],SearchBudget(30,5)),state)
        else:
            raw=released_pp(probe(),state,j['order'],j['selection']['retry_seed'],chosen,30,5)
        require(raw['status']=='ok','branch unknown: '+method)
        check=check_paths(state,raw['paths'],j['order'])
        restored=env.reset_paths(raw['paths'],seed=j['seed'])
        require(paths_of(restored)==raw['paths'] and restored['num_of_colliding_pairs']==check['conflicts'], 'native path validation')
        recovered=check['conflicts']<base_count
        branches[method]=dict(applicable=True,chosen_agent=chosen,raw=raw,path_check=check,
                              final_conflicts=check['conflicts'] if recovered else base_count,recovered=recovered,
                              final_paths=raw['paths'] if recovered else base['paths'],
                              generated=sum(d['generated'] for d in raw['diagnostics']),
                              expanded=sum(d['expanded'] for d in raw['diagnostics']),
                              native_paths_verified=True)
    write_json(OUT/'results'/f'{job_id}.json',seal(dict(schema='lns2.reservation_feedback_result.v1',
        job=j,plan_sha256=sha256_file(OUT/'plan.json'),baseline_verified=True,
        base_conflicts=base_count,triggered=triggered,branches=branches,status='ok')))


def read_result(j):
    r=read_json(OUT/'results'/f"{j['id']}.json"); check_seal(r)
    require(r['job']==j and r['status']=='ok' and r['plan_sha256']==sha256_file(OUT/'plan.json'),'result identity')
    return r


def collect():
    m=load(); dest=OUT/'results'; dest.mkdir(exist_ok=True)
    lock=OUT/'run.lock'
    with lock.open('x') as f: json.dump(dict(pid=__import__('os').getpid()),f)
    active=[]; done=[]; pending=[]
    def status(kind,error=None):
        write_json(OUT/'run_status.json',dict(status=kind,completed=len(done),total=48,active=len(active),error=error))
    try:
        for j in m['jobs']:
            if (dest/f"{j['id']}.json").exists(): read_result(j); done.append(j['id'])
            else: pending.append(j)
        while pending or active:
            while pending and len(active)<m['workers'] and not (OUT/'STOP_AFTER_JOB').exists():
                j=pending.pop(0); log=(dest/f"{j['id']}.log").open('w')
                p=subprocess.Popen([sys.executable,'-B',__file__,'_worker',j['id']],stdout=log,stderr=subprocess.STDOUT)
                active.append((p,log,time.monotonic(),j))
            for item in list(active):
                p,log,start,j=item
                if p.poll() is None:
                    if time.monotonic()-start>m['job_fuse']: raise TimeoutError(j['id'])
                    continue
                log.close(); active.remove(item); require(p.returncode==0,'worker failed: '+j['id'])
                read_result(j); done.append(j['id']); print(json.dumps(dict(completed=len(done),job=j['id'])),flush=True)
            status('running')
            if not active and (OUT/'STOP_AFTER_JOB').exists(): break
            time.sleep(.3)
        status('complete' if len(done)==48 else 'paused')
    except BaseException as e:
        status('error',repr(e)); raise
    finally:
        for p,log,_,_ in active:
            if p.poll() is None: p.kill()
            p.wait(); log.close()
        lock.unlink()


def gate(rows):
    groups=defaultdict(list)
    for r in rows: groups[r['job']['case']['case_id']].append(r)
    stable=[]; unique=[]
    for case,rs in groups.items():
        if len(rs)!=2 or not all(r['triggered'] and r['branches']['directed_release']['applicable'] for r in rs): continue
        if all(r['branches']['directed_release']['recovered'] for r in rs):
            stable.append(case)
            if all(all(r['branches']['directed_release']['final_conflicts']<r['branches'][b]['final_conflicts']
                       for b in ('fresh_retry','random_release')) for r in rs): unique.append(case)
    maps={groups[c][0]['job']['case']['map_id'] for c in stable}
    failed=[c for c in stable if groups[c][0]['job']['case']['role']=='failed_long_unchanged']
    return dict(stable_states=stable,stable_maps=sorted(maps),stable_failed_states=failed,unique_states=unique,
                passed=len(stable)>=3 and len(maps)>=2 and len(failed)>=1 and len(unique)>=2)


def analyze():
    m=load(); rows=[read_result(j) for j in m['jobs']]; summaries={}
    for method in METHODS:
        bs=[r['branches'][method] for r in rows]
        summaries[method]=dict(applicable=sum(b['applicable'] for b in bs),
            recovered=sum(b['recovered'] for b in bs),
            failed_tail_recovered=sum(r['branches'][method]['recovered'] for r in rows if r['job']['case']['role']=='failed_long_unchanged'),
            generated=sum(b.get('generated',0) for b in bs),expanded=sum(b.get('expanded',0) for b in bs),
            rollback=sum(b['raw']['rolled_back'] for b in bs if b['applicable']))
    g=gate(rows)
    report=dict(schema='lns2.reservation_feedback_report.v1',conditions=48,states=24,errors=0,
                baseline_verified=sum(r['baseline_verified'] for r in rows),gate=g,summaries=summaries,
                decision='bounded_development_signal_only' if g['passed'] else 'no_go',
                result_sha256={j['id']:sha256_file(OUT/'results'/f"{j['id']}.json") for j in m['jobs']})
    write_json(OUT/'report.json',report); print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('command',choices=('prepare','collect','analyze','verify','_worker')); p.add_argument('job',nargs='?'); a=p.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='collect': collect()
    elif a.command=='analyze': analyze()
    elif a.command=='verify': load(); print('Inputs verified')
    else: worker(a.job)


if __name__=='__main__': main()
