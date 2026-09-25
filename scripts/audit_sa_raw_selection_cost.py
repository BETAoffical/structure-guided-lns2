"""Frozen raw selector cost/equivalence audit, never an end-to-end TTF run."""
import argparse
import cProfile
from copy import deepcopy
import json
import os
from pathlib import Path
import pstats
import statistics
import sys
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_raw_ttf as source
from scripts import run_sa_onpolicy as run
from scripts import recover_sa_onpolicy as recovery
from experiments import sa_raw_timed_runtime as rt
from experiments.sa_raw_selection_fast import FastPolicy

OUTPUT = 'build/sa-raw-selection-engineering-v1'
SOURCE_SHA = '3995070a223e787817ef4cc9cf72f95ab60f5c2a392b9794e6857474e9f26f69'
CODE = ('scripts/audit_sa_raw_selection_cost.py', 'experiments/sa_raw_selection_fast.py',
        'tests/evaluation/test_sa_raw_selection_fast.py', 'docs/SA_RAW_SELECTION_ENGINEERING_PROTOCOL_ZH.md')
require = run.require


def indices(count):
    require(count > 0, 'empty raw episode')
    return sorted({0, (count - 1) // 2, count - 1})


def sample_jobs(jobs):
    return [j for j in jobs if j['comparison_arm'] == 'raw_updated'
            and j['solver_seed'] == 251 and j['replica'] == 0]


class SavedPool:
    """Immutable archived proposals; no native proposal or repair is measured."""
    def __init__(self, event):
        self.event = event

    def select(self, env, state, decision):
        require(decision == self.event['decision'], 'saved pool decision mismatch')
        by_id = {c['candidate_id']: c for c in self.event['pool']}
        pool = [by_id[c] for c in self.event['proposal_order']]
        return next(i for i, c in enumerate(pool) if c['candidate_id'] == self.event['anchor_id']), pool


def equal_event(actual, expected):
    require(actual.keys() <= expected.keys(), 'unknown selection field')
    for name, value in actual.items():
        require(value == expected[name], 'selection mismatch: ' + name)


def context(job):
    return dict(task_id=job['case']['task_id'], solver_seed=job['solver_seed'],
                case_id=f"{job['case']['task_id']}-seed{job['solver_seed']}",
                proposal=job['plan']['template']['proposal'])


def loaded(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    q = run.native_runtime(job['plan'])
    folder = ROOT / job['output'] / 'episodes' / job['job_id']
    row = run.check_seal(run.read_json(folder / 'result.json'))
    require(run.sha256_file(folder / 'result.json') == job['result_sha256'], 'changed result')
    for name, digest in row['files'].items():
        require(run.sha256_file(folder / name) == digest, 'changed source file')
    initial = run.read_json(folder / 'initial.json')
    require(q.state_fingerprint(initial) == job['expected_initial'], 'initial identity')
    return q, folder, row, initial


def profile_rows(profiler):
    result = []
    for (file, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(profiler).stats.items():
        if 'structure-guided-lns2/' in file.replace('\\', '/'):
            file = file.replace('\\', '/').split('structure-guided-lns2/', 1)[1]
        else:
            file = Path(file).name
        result.append(dict(file=file, line=line, name=name, calls=calls,
                           self_seconds=own, cumulative_seconds=cumulative))
    return sorted(result, key=lambda r: (-r['cumulative_seconds'], r['file'], r['line']))[:50]


def replay_worker(job):
    q, folder, row, initial = loaded(job)
    fast = FastPolicy(job, q, context(job))
    fast.start(initial)
    state, previous = initial, initial
    samples, count, candidate_count = [], 0, 0
    wanted = set(indices(row['decisions']))
    for event in run.trace_read(folder):
        d = event['decision']
        require(d == count and event['before'] == q.state_fingerprint(state), 'trace continuity')
        if job['mode'] == 'verify':
            fast.selector = SavedPool(event)
            equal_event(fast.choose(None, state, d), event)
            candidate_count += len(event['pool'])
        elif d in wanted:
            observations = []
            modes = ['reference'] if job['mode'] == 'profile' else ['reference', 'fast']
            rounds = 1 if job['mode'] == 'profile' else 3
            for repeat in range(rounds):
                order = modes if (d + repeat + job['sample_index']) % 2 == 0 else list(reversed(modes))
                for name in order:
                    policy_type = rt.Policy if name == 'reference' else FastPolicy
                    policy = policy_type(job, q, context(job))
                    policy.start(initial)
                    policy.history = deepcopy(fast.history)
                    policy.engine.prepare(previous)
                    policy.selector = SavedPool(event)
                    snapshot = deepcopy(state)
                    before = q.state_fingerprint(snapshot)
                    if job['mode'] == 'profile':
                        profiler = cProfile.Profile()
                        answer = profiler.runcall(policy.choose, None, snapshot, d)
                        measured = dict(profile=profile_rows(profiler))
                    else:
                        began = time.perf_counter()
                        answer = policy.choose(None, snapshot, d)
                        measured = dict(seconds=time.perf_counter()-began)
                    equal_event(answer, event)
                    require(q.state_fingerprint(snapshot) == before, 'selection mutated state')
                    observations.append(dict(mode=name, repeat=repeat, **measured))
            samples.append(dict(decision=d, candidates=len(event['pool']), observations=observations))
        after = q.apply_state_delta(state, event['delta'])
        fast.observe(state, event, after)
        previous, state = state, after
        count += 1
    require(count == row['decisions'] and q.state_fingerprint(state) == row['final_fingerprint'], 'final continuity')
    return dict(status='ok', job_id=job['job_id'], comparison_arm=job['comparison_arm'],
                map_id=job['case']['map_id'], decisions=count, candidates=candidate_count,
                samples=samples, no_new_repairs=True)


def native_worker(job):
    from scripts.train_sa_history_selector import die_with_parent
    die_with_parent(job['parent_pid'])
    q, folder, row, _ = loaded(job)
    q, env, ctx = rt.prepare_environment(job)
    policy = FastPolicy(job, q, ctx)
    state = q._plain(env.reset(seed=job['solver_seed']))
    require(q.state_fingerprint(state) == job['expected_initial'], 'native initial')
    policy.start(state)
    count = 0
    for event in run.trace_read(folder):
        answer = policy.choose(env, state, count)
        equal_event(answer, event)
        after, metrics, temperature, uniform = rt.transition(job, q, env, state, answer, count, 20.)
        require(q.state_fingerprint(after) == q.state_fingerprint(q.apply_state_delta(state,event['delta'])),
                'native paths/counters differ')
        for key in ('neighborhood', 'repair_order', 'pp_failure_reason', 'replan_success'):
            require(metrics[key] == event['metrics'][key], 'native metric differs: ' + key)
        require(temperature == event['temperature'] and uniform == event['uniform'], 'SA drift')
        rt.finish_event(q,state,after,answer,count,metrics,temperature,uniform,policy.sha)
        policy.observe(state,answer,after)
        state = after
        count += 1
        if count == 3:
            break
    require(count == min(3,row['decisions']), 'short native prefix')
    return dict(status='ok',job_id=job['job_id'],steps=count,no_ttf=True)


def prepare():
    old, src = source.verify()
    require(run.sha256_file(src/'report.json') == SOURCE_SHA, 'source report changed')
    for phase in ('preflight','collect','audit'):
        source.check_complete(old,src,phase,source.jobs_for(old))
    jobs = [dict(j, result_sha256=run.sha256_file(src/'episodes'/j['job_id']/'result.json'))
            for j in old['jobs'] if j['comparison_arm'].startswith('raw_')]
    require(len(jobs)==96 and len(sample_jobs(jobs))==12, 'scope changed')
    inputs = {n:run.sha256_file(ROOT/n) for n in CODE}
    body = dict(schema='lns2.sa.raw_selection_engineering.v1',source_binding=old['binding'],
                source_report_sha256=SOURCE_SHA,inputs=inputs,jobs=jobs,
                verify_workers=20,component_workers=1,repeats=3,no_training=True,no_ttf=True)
    body['binding']=run.json_fingerprint(body)
    out=ROOT/OUTPUT
    require(not out.exists(),'existing registration')
    with recovery.strict_lock(out,body['binding'],'raw-cost-prepare'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(episodes=96,component_episodes=12,component_states=36,native_prefix_steps=36,no_ttf=True)


def verify_registration():
    old,src=source.verify()
    r=run.check_seal(run.read_json(ROOT/OUTPUT/'registration.json'))
    require(r['binding']==run.json_fingerprint({k:v for k,v in r.items() if k not in ('binding','integrity')}), 'audit registration')
    require(r['source_binding']==old['binding'] and run.sha256_file(src/'report.json')==SOURCE_SHA,'source identity')
    for n,h in r['inputs'].items():require(run.sha256_file(ROOT/n)==h,'changed audit input: '+n)
    return r,ROOT/OUTPUT


def execute(phase):
    r,out=verify_registration()
    jobs=r['jobs'] if phase=='verify' else sample_jobs(r['jobs'])
    if phase in ('benchmark','native'):
        source.check_complete(r,out,'verify',r['jobs'])
    jobs=[dict(j,parent_pid=os.getpid(),timing_binding=r['source_binding'],mode=phase,sample_index=i)
          for i,j in enumerate(jobs)]
    with recovery.strict_lock(out,r['binding'],'raw-selection-'+phase):
        require(not (out/(phase+'.complete.json')).exists(),'phase already complete')
        pending=[]
        for j in jobs:
            p=out/phase/(j['job_id']+'.json')
            if p.exists():
                receipt=run.check_seal(run.read_json(p))
                require(receipt['status']=='ok' and receipt['binding']==r['binding'],'failed receipt; inspect before retry')
            else:pending.append(j)
        workers=1 if phase in ('profile','benchmark') else r['verify_workers']
        source.execute(r,out,pending,native_worker if phase=='native' else replay_worker,
                       phase,workers,900.)
        files={j['job_id']:run.sha256_file(out/phase/(j['job_id']+'.json')) for j in jobs}
        run.once(out/(phase+'.complete.json'),run.sealed(dict(binding=r['binding'],jobs=len(jobs),files=files)))
    return dict(phase=phase,jobs=len(jobs),no_ttf=True)


def report():
    r,out=verify_registration()
    results={}
    for phase in ('profile','verify','benchmark','native'):
        jobs=r['jobs'] if phase=='verify' else sample_jobs(r['jobs'])
        source.check_complete(r,out,phase,jobs)
        results[phase]=[run.check_seal(run.read_json(out/phase/(j['job_id']+'.json'))) for j in jobs]
    states=[]
    for receipt in results['benchmark']:
        for sample in receipt['samples']:
            times={m:statistics.median(o['seconds'] for o in sample['observations'] if o['mode']==m)
                   for m in ('reference','fast')}
            states.append(dict(job_id=receipt['job_id'],map_id=receipt['map_id'],decision=sample['decision'],
                               **times,reduction=1-times['fast']/times['reference']))
    base=statistics.mean(s['reference'] for s in states)
    fast=statistics.mean(s['fast'] for s in states)
    report=dict(schema='lns2.sa.raw_selection_engineering.report.v1',binding=r['binding'],
                source_report_sha256=SOURCE_SHA,episodes=96,
                decisions=sum(x['decisions'] for x in results['verify']),
                candidates=sum(x['candidates'] for x in results['verify']),
                native_prefix_steps=sum(x['steps'] for x in results['native']),
                states=states,reference_mean_seconds=base,fast_mean_seconds=fast,
                ratio_of_means_reduction=1-fast/base,
                median_state_reduction=statistics.median(s['reduction'] for s in states),
                faster_states=sum(s['fast']<s['reference'] for s in states),
                no_new_training=True,no_default_change=True,no_ttf=True,
                measured_scope='post_proposal_selection_only; archived_pool; not_whole_selection_or_TTF')
    run.once(out/'report.json',run.sealed(report))
    return {k:v for k,v in report.items() if k!='states'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','profile','verify','benchmark','native','report'))
    a=p.parse_args()
    value=prepare() if a.phase=='prepare' else report() if a.phase=='report' else execute(a.phase)
    print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':main()
