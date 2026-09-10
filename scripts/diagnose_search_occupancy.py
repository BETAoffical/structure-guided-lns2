"""Bounded search-observation readiness, never a candidate intervention."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.dont_write_bytecode=True
from experiments._common import read_json,write_json
from experiments.full_neighborhood_recovery import SearchBudget,run_sequence,check_paths
from experiments.native_path_compatibility import paths_of,seal,check_seal
from experiments.search_occupancy_observation import ObservingProbe,rank_contacts
from experiments.local_path_compatibility import sha256_file
from scripts.diagnose_reservation_mediation import require

OUT=ROOT/'build/initlns-search-occupancy-observer-v1'
REG=ROOT/'artifacts/initlns-search-occupancy-observer-v1/registration.json'
MODULE='build/linux/search-occupancy-observer-v1/lns2_search_observer_native.cpython-310-x86_64-linux-gnu.so'


def prepare():
    from scripts.diagnose_reservation_mediation import load as old_load,OUT as FIRST,OLD
    old=old_load(); files=dict(old['files']); jobs=[]
    evidence=read_json(ROOT/'artifacts/initlns-reservation-mediation-v1/evidence.json')
    require(sha256_file(ROOT/evidence['round1_report'])==evidence['round1_sha256'],'first report changed')
    rr=read_json(ROOT/evidence['round1_report'])
    for j in old['jobs']:
        filename=(FIRST/'round1'/f"{j['id']}.json").relative_to(ROOT).as_posix()
        require(sha256_file(ROOT/filename)==rr['result_sha256'][j['id']],'witness result changed')
        action=read_json(OLD/f"{j['trial']:02d}-compressed11.json")['job']['action']
        jobs.append(dict(id=f"witness-{j['trial']:02d}",phase='witness',seed=action['pp_random_seed'],
            order=action['repair_order'],expected_file=filename,case_id=old['case']['case_id'],
            role='successful_long_unchanged',map_id=old['case'].get('map_id','witness'),
            map_file=old['case']['map_file'],scenario_file=old['case']['scenario_file'],
            state_file=(OLD/'before.json').relative_to(ROOT).as_posix()))
        files[filename]=sha256_file(ROOT/filename)
    from scripts.diagnose_reservation_feedback import load as feedback_load
    development=feedback_load()
    for j in development['jobs']:
        c=j['case']
        jobs.append(dict(id='development-'+j['id'],phase='development',seed=j['seed'],order=j['order'],
                         expected_file=j['source_result'],**{k:c[k] for k in
                         ('case_id','role','map_id','map_file','scenario_file','state_file')}))
        for key in ('expected_file','map_file','scenario_file','state_file'):
            p=jobs[-1][key]; files[p]=development['files'][p]
    for directory in ('third_party/mapf_lns2/inc','third_party/mapf_lns2/src','src/search_observer'):
        for p in (ROOT/directory).rglob('*'):
            if p.is_file(): files[p.relative_to(ROOT).as_posix()]=sha256_file(p)
    extra=[MODULE,'src/path_probe/native_path_probe.cpp','scripts/diagnose_search_occupancy.py',
           'experiments/search_occupancy_observation.py','tests/test_search_occupancy_observation.py',
           'docs/SEARCH_OCCUPANCY_OBSERVER_PROTOCOL_ZH.md',
           'build/linux/search-occupancy-observer-v1/CMakeCache.txt']
    for p in extra: files[p]=sha256_file(ROOT/p)
    plan=dict(schema='lns2.search_occupancy.v1',module_file=MODULE,jobs=jobs,files=files,
              witness_agent=old['released_agent'],workers=20,fuse_seconds=180,
              search_seconds=30,call_seconds=5,event_cap=65536,
              build=dict(compiler='GNU 11.4.0',python='3.10.12',pybind11='2.9.1',
                         source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))
    OUT.mkdir(exist_ok=True); require(not (OUT/'plan.json').exists(),'plan exists')
    write_json(OUT/'plan.json',plan)
    print(json.dumps(dict(jobs=dict(Counter(j['phase'] for j in jobs)),
                         plan_sha256=sha256_file(OUT/'plan.json'),module_sha256=files[MODULE])))


def load():
    require(sha256_file(OUT/'plan.json')==read_json(REG)['plan_sha256'],'plan registration changed')
    plan=read_json(OUT/'plan.json')
    for p,h in plan['files'].items(): require(sha256_file(ROOT/p)==h,'bound file changed: '+p)
    return plan


def signature(sequence):
    if 'diagnostics' in sequence:
        diagnostics=sequence['diagnostics']; rolled=sequence['rolled_back']
    else:
        require(sequence['status'] in ('accepted','rolled_back'),'unexpected search status')
        diagnostics=[dict(agent=r['agent'],**r['search']) for r in sequence['records']]
        rolled=sequence['status']=='rolled_back'
    return dict(paths=sequence['paths'],rolled_back=rolled,attempted_pairs=sequence['attempted_pairs'],
                diagnostics=[{k:r[k] for k in ('agent','status','path','cost','expanded','generated','low_level_collisions')}
                             for r in diagnostics])


def write_gzip(path,value):
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:
            compressed.write(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
    temporary.replace(path)


def worker(phase,job_id):
    m=load(); j=next(j for j in m['jobs'] if j['id']==job_id and j['phase']==phase)
    expected_source=read_json(ROOT/j['expected_file'])
    expected=(expected_source['data']['sequences']['compressed11'] if phase=='witness' else expected_source['base'])
    state=read_json(ROOT/j['state_file']); original={a['id']:a['path'] for a in state['agents']}
    sys.path.insert(0,str((ROOT/m['module_file']).parent)); module=importlib.import_module('lns2_search_observer_native')
    require(Path(module.__file__).resolve()==(ROOT/m['module_file']).resolve(),'wrong module')
    require(module.observer_schema=='lns2.offered_soft_intervals.v1','observer schema')
    runs={}; proxy=None
    for mode in ('off','on'):
        probe=module.NativePathProbe(str(ROOT/j['map_file']),str(ROOT/j['scenario_file']),paths_of(state))
        proxy=ObservingProbe(module,probe,original,set(original)-set(j['order']),mode=='on',m['event_cap'])
        sequence=run_sequence(proxy,state,j['order'],j['seed'],SearchBudget(m['search_seconds'],m['call_seconds']))
        require(signature(sequence)==signature(expected),'frozen/off/on signature mismatch: '+mode)
        check_paths(state,sequence['paths'],j['order'])
        runs[mode]=sequence
    require(proxy is not None,'missing observer run')
    truncated=sum(c['capture']['truncated'] for c in proxy.captures)
    unmapped=sum(len(s['unmatched']) for s in proxy.summaries)
    ranked=rank_contacts(proxy.summaries); ids=[r['agent'] for r in ranked]
    raw=OUT/phase/f'{job_id}.json.gz'
    write_gzip(raw,dict(job=j,on=runs['on'],off=runs['off'],captures=proxy.captures,summaries=proxy.summaries))
    report=dict(schema='lns2.search_occupancy_result.v1',job=j,status='ok',parity_passed=True,
                plan_sha256=sha256_file(OUT/'plan.json'),raw_file=raw.relative_to(ROOT).as_posix(),raw_sha256=sha256_file(raw),
                truncated_queries=truncated,unmapped_flags=unmapped,
                flagged=sum(s['flagged'] for s in proxy.summaries),
                queries=sum(c['capture']['queries'] for c in proxy.captures),
                records=sum(len(c['capture']['events']) for c in proxy.captures),
                searches=len(proxy.captures),ranking=ranked,
                witness_rank=(ids.index(m['witness_agent'])+1 if m['witness_agent'] in ids else None) if phase=='witness' else None)
    write_json(OUT/phase/f'{job_id}.json',seal(report))


def read_result(j):
    r=read_json(OUT/j['phase']/f"{j['id']}.json"); check_seal(r)
    require(r['status']=='ok' and r['job']==j and r['plan_sha256']==sha256_file(OUT/'plan.json'),'result identity')
    require(sha256_file(ROOT/r['raw_file'])==r['raw_sha256'],'raw observation changed')
    return r


def collect(phase):
    m=load(); jobs=[j for j in m['jobs'] if j['phase']==phase]
    if phase=='development':
        report=read_json(OUT/'witness-report.json')
        require(report['next_phase_allowed'],'witness gate not passed')
        for j in m['jobs']:
            if j['phase']=='witness':
                read_result(j)
                require(sha256_file(OUT/'witness'/f"{j['id']}.json")==report['result_sha256'][j['id']], 'witness evidence changed')
    directory=OUT/phase; directory.mkdir(exist_ok=True)
    lock=OUT/'run.lock'
    with lock.open('x') as stream: json.dump(dict(pid=os.getpid(),phase=phase),stream)
    active=[]; done=[]; pending=[]
    def status(kind,error=None):
        write_json(OUT/'run_status.json',dict(status=kind,phase=phase,completed=len(done),total=len(jobs),active=len(active),error=error))
    try:
        for j in jobs:
            if (directory/f"{j['id']}.json").exists(): read_result(j); done.append(j['id'])
            else: pending.append(j)
        while pending or active:
            while pending and len(active)<m['workers'] and not (OUT/'STOP_AFTER_JOB').exists():
                j=pending.pop(0); log=(directory/f"{j['id']}.log").open('w')
                p=subprocess.Popen([sys.executable,'-B',__file__,'_worker',phase,j['id']],stdout=log,stderr=subprocess.STDOUT)
                active.append((p,log,time.monotonic(),j))
            for item in list(active):
                p,log,start,j=item
                if p.poll() is None:
                    if time.monotonic()-start>m['fuse_seconds']: raise TimeoutError(j['id'])
                    continue
                log.close(); active.remove(item); require(p.returncode==0,'worker error: '+j['id'])
                read_result(j); done.append(j['id']); print(json.dumps(dict(phase=phase,completed=len(done),job=j['id'])),flush=True)
            status('running')
            if not active and (OUT/'STOP_AFTER_JOB').exists(): break
            time.sleep(.3)
        status('complete' if len(done)==len(jobs) else 'paused')
    except BaseException as e:
        status('error',repr(e)); raise
    finally:
        for p,log,_,_ in active:
            if p.poll() is None: p.kill()
            p.wait(); log.close()
        lock.unlink()


def analyze(phase):
    m=load(); rows=[read_result(j) for j in m['jobs'] if j['phase']==phase]
    integrity=all(r['parity_passed'] and not r['truncated_queries'] and not r['unmapped_flags'] for r in rows)
    coverage=sum(r['witness_rank'] is not None for r in rows) if phase=='witness' else None
    report=dict(schema='lns2.search_occupancy_report.v1',phase=phase,jobs=len(rows),integrity_passed=integrity,
                truncated_queries=sum(r['truncated_queries'] for r in rows),
                unmapped_flags=sum(r['unmapped_flags'] for r in rows),
                total_flagged=sum(r['flagged'] for r in rows),total_records=sum(r['records'] for r in rows),
                searches=sum(r['searches'] for r in rows),candidate_owner_counts=[len(r['ranking']) for r in rows],
                witness_ranks=[r['witness_rank'] for r in rows] if phase=='witness' else None,
                witness_captured=coverage,
                next_phase_allowed=integrity and phase=='witness' and coverage>=6,
                decision='observation_readiness_only' if integrity else 'no_go_observation_integrity',
                result_sha256={r['job']['id']:sha256_file(OUT/phase/f"{r['job']['id']}.json") for r in rows})
    if phase=='witness' and integrity and coverage<6: report['decision']='no_go_witness_visibility'
    write_json(OUT/f'{phase}-report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','collect','analyze','_worker'))
    p.add_argument('phase',nargs='?',choices=('witness','development')); p.add_argument('job',nargs='?')
    a=p.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='verify': load(); print('Registered observer inputs verified')
    elif a.command=='collect': collect(a.phase)
    elif a.command=='analyze': analyze(a.phase)
    else: worker(a.phase,a.job)


if __name__=='__main__': main()
