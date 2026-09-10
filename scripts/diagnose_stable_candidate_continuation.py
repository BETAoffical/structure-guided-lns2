"""One registered candidate, two paired arms; no controller or model changes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
OLD = ROOT / 'build/path-quality-pressure-deadline-recovery-v1/review-20260909'
sys.path.insert(0, str(OLD))
import diagnose_continuation_v2 as legacy

q = legacy.q
OUT = ROOT / 'build/initlns-stable-candidate-continuation-v1'
CID = 'case-3723c8bd272e'
CANDIDATE = 'neighborhood-0d3103de10e25097'
ARMS = ('frozen', 'alternative_first')
STOP = False


def trial_seed(trial):
    return int(q.digest(['stable-candidate-continuation-v1', 20260911, CID, trial])[:12], 16) % (2**31)


def schedule(plan):
    jobs = []
    for trial in range(8):
        for arm in ARMS[::1 if trial % 2 == 0 else -1]:
            job = dict(case_id=CID, trial=trial, arm=arm, trial_seed=trial_seed(trial),
                       fingerprint=plan['fingerprint'], smoke=False)
            job['job_id'] = q.digest(job)[:20]
            jobs.append(job)
    return jobs


def prepare():
    old = legacy.verify()
    case = next(c for c in old['cases'] if c['case_id'] == CID)
    prior = OLD / 'mechanism-collection-16workers-v1/analysis.json'
    analysis = q.read(prior)
    q.require(q.sha(prior) == q.read(prior.parent / 'analysis_identity.json')['analysis_sha256'], 'prior hash')
    row = next(r for r in analysis['candidates_summary'] if r['case_id'] == CID and r['candidate_id'] == CANDIDATE)
    q.require(row['after_conflicts'] == [39]*8 and row['size'] == 4 and not row['selected'], 'candidate evidence changed')
    candidates = q.read(ROOT / case['candidates_file'])
    chosen = next(c for c in candidates if c['candidate_id'] == CANDIDATE)
    q.require(len(chosen['agents']) == 4, 'candidate membership')
    old_seeds = {j['trial_seed'] for j in q.read(OLD/'continuation-diagnostic-v2/schedule.json')}
    main_jobs = q.rows(q.PACKAGE/'main_jobs.jsonl')
    aux_jobs = q.rows(q.PACKAGE/'auxiliary_jobs.jsonl')
    for j in main_jobs + aux_jobs:
        old_seeds.update(v for k,v in j.get('action',{}).items() if 'seed' in k and isinstance(v,int))
    seeds = [trial_seed(i) for i in range(8)]
    q.require(len(set(seeds)) == 8 and not (set(seeds) & old_seeds), 'trial stream collision')
    files = dict(old['files'])
    for p in (Path(__file__), ROOT/'docs/STABLE_CANDIDATE_CONTINUATION_PROTOCOL_ZH.md',
              ROOT/'tests/test_stable_candidate_continuation.py', OLD/'continuation-diagnostic-v2/plan.json',
              OLD/'continuation-diagnostic-v2/schedule.json', prior):
        files[p.relative_to(ROOT).as_posix()] = q.sha(p)
    plan = dict(old, schema='lns2.stable_candidate_continuation.v1', cases=[case],
                alternatives={CID:dict(candidate_id=CANDIDATE, agents=chosen['agents'],
                    rule='Fixed after viewing all eight historical trials; exploratory, not independent confirmation')},
                arms=list(ARMS), trials=8, workers=16, files=files, seed_roots=seeds,
                diagnostic_success_rule='At least two more feasible trials, no paired feasible-to-infeasible loss; or median terminal conflicts lower by at least two, AUC lower in at least six pairs, zero feasible-to-infeasible loss. Descriptive gate only.',
                no_automatic_next_stage=True)
    plan.pop('fingerprint')
    plan['fingerprint'] = q.digest(plan)
    OUT.mkdir(exist_ok=True)
    target = OUT/'plan.json'
    if target.exists():
        q.require(q.read(target) == plan, 'registered plan changed')
    else:
        q.write(target, plan)
    q.write(OUT/'schedule.json', schedule(plan))
    print(json.dumps(dict(jobs=16, workers=16, decision_budget=64, continuation_seconds=60,
                          fuse_seconds=300, fingerprint=plan['fingerprint'])), flush=True)


def verify():
    plan = q.read(OUT/'plan.json')
    q.require(plan['fingerprint'] == q.digest({k:v for k,v in plan.items() if k!='fingerprint'}), 'plan fingerprint')
    for p,h in plan['files'].items():
        q.require(q.sha(ROOT/p)==h, 'registered file changed: '+p)
    q.require(q.read(OUT/'schedule.json')==schedule(plan), 'schedule changed')
    return plan


def child(path):
    request=q.read(path)
    q.require(request in schedule(verify()), 'unknown request')
    legacy.OUT=OUT
    legacy.verify=verify
    return legacy.child(path)


def stop_signal(*_):
    global STOP
    STOP=True


def collect(resume=False):
    plan=verify()
    for name in ('results','states','traces','requests','receipts'):
        (OUT/name).mkdir(exist_ok=True)
    lock=OUT/'run.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode()); os.close(fd)
    active={}; completed={}; pending=[]; error=None
    signal.signal(signal.SIGINT,stop_signal); signal.signal(signal.SIGTERM,stop_signal)
    def status(kind):
        q.write(OUT/'run_status.json',dict(status=kind,completed=len(completed),total=16,
                active=len(active),error=error,fingerprint=plan['fingerprint']))
    try:
        for job in schedule(plan):
            p=OUT/'results'/(job['job_id']+'.json')
            if p.exists():
                q.require(resume,'existing result requires resume')
                r=q.read(p); check_result(r,job,check_trace=True)
                q.require(q.read(OUT/'receipts'/(job['job_id']+'.json'))['sha256']==q.sha(p),'receipt changed')
                completed[job['job_id']]=r
            else:
                pending.append(job)
        while pending or active:
            stopping=STOP or (OUT/'STOP_AFTER_EPISODE').exists() or error is not None
            while pending and len(active)<16 and not stopping:
                job=pending.pop(0); jid=job['job_id']; req=OUT/'requests'/(jid+'.json'); q.write(req,job)
                log=(OUT/'results'/(jid+'.log')).open('w')
                proc=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),'_child',str(req)],
                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                     env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                active[jid]=(proc,log,time.monotonic(),job)
            for jid,(proc,log,began,job) in list(active.items()):
                timeout=proc.poll() is None and time.monotonic()-began>plan['process_fuse_seconds']
                if timeout: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                if proc.poll() is None: continue
                log.close(); del active[jid]
                path=OUT/'results'/(jid+'.json')
                try:
                    if timeout:
                        q.write(OUT/'receipts'/(jid+'.timeout.json'),dict(job=job,status='censored'))
                    q.require(not timeout and proc.returncode==0,'child error or fuse: '+jid)
                    r=q.read(path); check_result(r,job)
                    q.write(OUT/'receipts'/(jid+'.json'),dict(sha256=q.sha(path)))
                    completed[jid]=r
                except Exception as exc:
                    error=str(exc)
                with (OUT/'progress.jsonl').open('a') as f:
                    f.write(json.dumps(dict(job_id=jid,returncode=proc.returncode,timeout=timeout,error=error))+'\n')
                print(json.dumps(dict(completed=len(completed),total=16,error=error)),flush=True)
            status('draining_error' if error else 'running')
            if stopping and not active: break
            time.sleep(.2)
        status('failed' if error else 'complete' if len(completed)==16 else 'paused')
        return 2 if error else 0
    finally:
        for proc,log,_,_ in active.values():
            if proc.poll() is None: os.killpg(proc.pid,signal.SIGKILL)
            proc.wait(); log.close()
        lock.unlink()


def check_result(r,job,check_trace=False):
    q.require(r['job']==job and r['status']=='ok','result identity/status')
    q.require(r['source_selection_verified'],'source pool not verified')
    q.require(r['conflicts'][0]==40 and r['final_conflicts']==r['conflicts'][-1],'conflict sequence')
    q.require(len(r['conflicts'])==r['decisions']+1 and r['decisions']<=64,'decision count')
    q.require(r['feasible']==(r['final_conflicts']==0),'feasibility mismatch')
    padded=r['conflicts']+[0 if r['feasible'] else r['final_conflicts']]*(65-len(r['conflicts']))
    q.require(r['fixed64_conflict_auc']==sum((a+b)/2 for a,b in zip(padded,padded[1:])), 'AUC mismatch')
    for file,expected in ((r['trace_file'],r['trace_sha256']),(r['final_state_file'],r['final_state_sha256'])):
        q.require(q.sha(ROOT/file)==expected,'saved output hash')
    if check_trace:
        source=q.read(ROOT/verify()['cases'][0]['state_file'])
        previous=q.state_fingerprint(source)
        from experiments.state_analysis import reconstruct_conflicts
        values=[40]
        for event in q.rows(ROOT/r['trace_file']):
            if event['event']=='initial':
                q.require(event['fingerprint']==previous,'initial state')
                continue
            q.require(event['before_fingerprint']==previous,'trace chain')
            m=event['metrics']; q.require(m['action_valid'],'illegal action')
            q.require(sorted(event['action']['agents'])==sorted(m['neighborhood']),'modified explicit action')
            q.require(set(map(int,event['changed_paths']))<=set(m['neighborhood']),'unselected paths changed')
            previous=event['after_fingerprint']; values.append(m['conflicts_after'])
        q.require(values==r['conflicts'],'trace conflict sequence')
        final=q.read(ROOT/r['final_state_file']); legacy.validate_paths(final,source)
        q.require(q.state_fingerprint(final)==previous,'final fingerprint')
        events=reconstruct_conflicts(final['agents'])
        pairs={tuple(sorted((e.left,e.right))) for e in events}
        q.require(pairs=={tuple(sorted(e)) for e in final['conflict_edges']},'reconstructed final conflict edges')


def summarize(rows):
    paired=[(next(r for r in rows if r['job']['trial']==i and r['job']['arm']=='frozen'),
             next(r for r in rows if r['job']['trial']==i and r['job']['arm']=='alternative_first')) for i in range(8)]
    losses=sum(a['feasible'] and not b['feasible'] for a,b in paired)
    feasible_gain=sum(b['feasible'] for a,b in paired)-sum(a['feasible'] for a,b in paired)
    med=statistics.median(a['final_conflicts']-b['final_conflicts'] for a,b in paired)
    auc_wins=sum(b['fixed64_conflict_auc']<a['fixed64_conflict_auc'] for a,b in paired)
    uncensored=all(r['stop_reason']!='wall_budget' and r['native_timeouts']==0 for r in rows)
    go=uncensored and losses==0 and (feasible_gain>=2 or (med>=2 and auc_wins>=6))
    return dict(decision='bounded_signal_only' if go else 'no_go_or_inconclusive',
                no_budget_censoring=uncensored,paired_feasibility_losses=losses,feasible_gain=feasible_gain,
                median_final_conflict_improvement=med,auc_wins=auc_wins,
                arms={arm:dict(feasible=sum(r['feasible'] for r in rows if r['job']['arm']==arm),
                    mean_final=statistics.mean(r['final_conflicts'] for r in rows if r['job']['arm']==arm),
                    mean_auc=statistics.mean(r['fixed64_conflict_auc'] for r in rows if r['job']['arm']==arm)) for arm in ARMS})


def analyze():
    plan=verify(); rows=[]
    for job in schedule(plan):
        p=OUT/'results'/(job['job_id']+'.json'); r=q.read(p); check_result(r,job,True); rows.append(r)
        q.require(q.read(OUT/'receipts'/(job['job_id']+'.json'))['sha256']==q.sha(p),'sealed result changed')
    for trial in range(8):
        pair=[r for r in rows if r['job']['trial']==trial]
        first=[next(e for e in q.rows(ROOT/r['trace_file']) if e['event']=='transition') for r in pair]
        q.require(first[0]['action']['random_seed']==first[1]['action']['random_seed'],'first action seed mismatch')
        for r,e in zip(pair,first):
            wanted=CANDIDATE if r['job']['arm']=='alternative_first' else plan['cases'][0]['selected_candidate_id']
            q.require(e['candidate_id']==wanted,'first action selection mismatch')
            for event in q.rows(ROOT/r['trace_file']):
                if event['event']!='transition' or event['decision']==0: continue
                pool=event['candidate_pool']; best=max(c['score'] for c in pool)
                selected=next(c for c in pool if c['candidate_id']==event['candidate_id'])
                q.require(selected['score']==best,'continuation did not use frozen ranking')
    report=dict(summarize(rows),fingerprint=plan['fingerprint'],rows=rows,
                result_hashes={r['job']['job_id']:q.sha(OUT/'results'/(r['job']['job_id']+'.json')) for r in rows},
                inference='One post-hoc state; no independent generalization, serial TTF, or controller promotion')
    q.write(OUT/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','result_hashes')},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','collect','analyze','_child'))
    parser.add_argument('request',nargs='?'); parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    elif args.command=='collect': raise SystemExit(collect(args.resume))
    elif args.command=='analyze': analyze()
    else: raise SystemExit(child(Path(args.request)))
