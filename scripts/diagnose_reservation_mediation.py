"""Bounded retrospective CAT interventions, isolated from production controllers."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from experiments._common import read_json, write_json
from experiments.native_path_compatibility import edge_set, modules, paths_of, soft_pp
from experiments.full_neighborhood_recovery import check_paths
from experiments.local_path_compatibility import sha256_file

OUT = ROOT / 'build/initlns-reservation-mediation-v1'
OLD = ROOT / 'build/initlns-compressed-neighborhood-native-v1'
REG = ROOT / 'artifacts/initlns-reservation-mediation-v1/registration.json'
ARMS = ('original16', 'compressed12', 'compressed11')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def query_seed(trial, index):
    value = f'reservation-mediation-v1/20260910/{trial}/{index}'
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % (2**31)


def prepare():
    from scripts.probe_compressed_neighborhood import verify
    old = verify()
    evidence = read_json(ROOT/'artifacts/initlns-compressed-neighborhood-native-v1/evidence.json')
    require(sha256_file(ROOT/evidence['report_file']) == evidence['report_sha256'], 'old report SHA')
    config = read_json(ROOT/'configs/native_path_compatibility_v1.json')
    prior = read_json(ROOT/'build/initlns-native-path-compatibility-v1/manifest.json')
    for key in ('native_file', 'probe_file', 'core_file'):
        require(sha256_file(ROOT/config[key]) == prior['files'][config[key]], key+' SHA')
    delta = sorted(set(old['sets']['compressed12'])-set(old['sets']['compressed11']))
    require(len(delta) == 1, 'expected singleton subset difference')
    jobs = [dict(id=f'trial-{i:02d}', trial=i, query_seeds=[query_seed(i,j) for j in range(3)]) for i in range(8)]
    files = [str(p.relative_to(ROOT).as_posix()) for p in (OLD/'plan.json', OLD/'report.json', OLD/'before.json')]
    for trial in range(8):
        for arm in ARMS:
            p = OLD/f'{trial:02d}-{arm}.json'
            r = read_json(p)
            files.extend([p.relative_to(ROOT).as_posix(), r['state_file']])
    files += [old['case']['map_file'], old['case']['scenario_file']]
    files += [config[k] for k in ('native_file','probe_file','core_file')]
    files += ['scripts/diagnose_reservation_mediation.py', 'tests/test_reservation_mediation.py',
              'docs/RESERVATION_MEDIATION_PROTOCOL_ZH.md', 'experiments/native_path_compatibility.py',
              'experiments/full_neighborhood_recovery.py', 'experiments/state_analysis.py',
              'experiments/_common.py', 'experiments/local_path_compatibility.py',
              'experiments/local_path_search.py']
    plan = dict(schema='lns2.reservation_mediation.v1', config=config, case=old['case'],
                released_agent=delta[0], jobs=jobs, workers=20, job_fuse=180,
                pp_seconds=30, query_seconds=5, source_sha=evidence['report_sha256'],
                files={p:sha256_file(ROOT/p) for p in sorted(set(files))})
    OUT.mkdir(exist_ok=True)
    require(not (OUT/'plan.json').exists(), 'plan already exists')
    write_json(OUT/'plan.json', plan)
    print(json.dumps(dict(jobs_per_round=8, workers=8, round1_pp=24, round1_queries_max=48,
                         round2_pp=8, job_fuse_seconds=180, plan_sha=sha256_file(OUT/'plan.json'))))


def load():
    require(sha256_file(OUT/'plan.json') == read_json(REG)['plan_sha256'], 'registration mismatch')
    p = read_json(OUT/'plan.json')
    for f, expected in p['files'].items():
        require(sha256_file(ROOT/f) == expected, 'bound file changed: '+f)
    return p


def context_before(state, order, records, target):
    index = order.index(target)
    require([r['agent'] for r in records[:index]] == order[:index], 'incomplete prefix')
    current = {a['id']:a['path'] for a in state['agents'] if a['id'] not in order}
    current.update({r['agent']:r['path'] for r in records[:index]})
    return current


def compare_contexts(left, right, removed):
    require({i:p for i,p in left.items() if i != removed} ==
            {i:p for i,p in right.items() if i != removed}, 'non-target context differs')


def released_pp(probe, state, order, seed, released, seconds=30, call_seconds=5):
    """Only search guidance omits a path; acceptance always includes real paths."""
    original = {a['id']:a['path'] for a in state['agents']}
    require(released in original and released not in order, 'release must be external')
    require(len(order)==len(set(order)) and set(order) <= set(original), 'invalid order')
    visible = {i:p for i,p in original.items() if i not in order}
    old_pairs = edge_set(original, order)
    pairs, records, overrides = set(), [], {}
    started = time.monotonic()
    probe.seed_rng(seed)
    for aid in order:
        remaining = seconds-(time.monotonic()-started)
        if remaining <= 0:
            return dict(status='unknown', reason='job_budget', diagnostics=records)
        found = probe.plan(aid, [i for i in visible if i != released], overrides, False,
                           seconds=min(call_seconds, remaining))
        records.append(dict(agent=aid, **found))
        if found['status'] != 'path':
            return dict(status='unknown' if found['status']=='unknown' else 'not_found', diagnostics=records)
        overrides[aid] = found['path']
        visible[aid] = found['path']
        pairs |= edge_set(visible, [aid])
        if len(pairs) > len(old_pairs):
            return dict(status='ok', rolled_back=True, paths=paths_of(state),
                        attempted_pairs=len(pairs), diagnostics=records)
    final = [overrides.get(i,p) for i,p in original.items()]
    require(final[released] == original[released], 'external changed')
    check_paths(state, final, order)
    return dict(status='ok', rolled_back=False, paths=final,
                attempted_pairs=len(pairs), diagnostics=records)


def phase1(plan, job, state, make_probe):
    trial = job['trial']
    sequences, actions = {}, {}
    for arm in ARMS:
        ref = read_json(OLD/f'{trial:02d}-{arm}.json')
        action = ref['job']['action']; actions[arm] = action
        s = soft_pp(make_probe(), state, action['repair_order'], action['pp_random_seed'], plan['pp_seconds'])
        require(s['status']=='ok', 'shadow PP unknown')
        expected = paths_of(read_json(ROOT/ref['state_file']))
        require(s['paths']==expected, 'shadow PP paths differ: '+arm)
        require(s['rolled_back']==ref['metrics']['pp_rolled_back'], 'rollback mismatch')
        require(s['attempted_pairs']==ref['metrics']['pp_attempt_conflict_pair_count'], 'pair mismatch')
        check_paths(state, s['paths'], action['repair_order'])
        sequences[arm] = s
    short, long = sequences['compressed11'], sequences['compressed12']
    order = actions['compressed11']['repair_order']
    differing = [i for i in order if short['paths'][i] != long['paths'][i]]
    queries = []
    if differing:
        target = differing[0]
        left = context_before(state, order, short['diagnostics'], target)
        right = context_before(state, actions['compressed12']['repair_order'], long['diagnostics'], target)
        compare_contexts(left, right, plan['released_agent'])
        for seed in job['query_seeds']:
            row = dict(seed=seed)
            for name, ctx in (('old_occupancy',left), ('mediated_occupancy',right)):
                p = make_probe(); p.seed_rng(seed)
                found = p.plan(target, sorted(ctx), ctx, False, seconds=plan['query_seconds'])
                require(found['status']=='path', 'single query unknown/empty')
                row[name] = found
            row['path_changed'] = row['old_occupancy']['path'] != row['mediated_occupancy']['path']
            queries.append(row)
        explanation = dict(first_divergent_agent=target, released_agent=plan['released_agent'],
                           old_present=plan['released_agent'] in left,
                           mediated_present=plan['released_agent'] in right,
                           contexts_differ_only_at_released_agent=True)
    else:
        explanation = dict(first_divergent_agent=None)
    return dict(sequences=sequences, queries=queries, explanation=explanation)


def worker(phase, job_id):
    plan=load(); job=next(j for j in plan['jobs'] if j['id']==job_id)
    state=read_json(OLD/'before.json')
    _, module=modules(plan['config'])
    def make_probe():
        return module.NativePathProbe(str(ROOT/plan['case']['map_file']),
                                      str(ROOT/plan['case']['scenario_file']), paths_of(state))
    if phase=='round1':
        data=phase1(plan, job, state, make_probe)
    else:
        first=read_result('round1',job)
        a=read_json(OLD/f"{job['trial']:02d}-compressed11.json")['job']['action']
        result=released_pp(make_probe(),state,a['repair_order'],a['pp_random_seed'],plan['released_agent'],
                           plan['pp_seconds'],plan['query_seconds'])
        require(result['status']=='ok', 'release PP unknown/not_found')
        check=check_paths(state,result['paths'],a['repair_order'])
        data=dict(result=result, path_check=check,
                  baseline_conflicts={k:len(edge_set(dict(enumerate(v['paths']))))
                                      for k,v in first['data']['sequences'].items()},
                  source_round1_sha=sha256_file(OUT/'round1'/f"{job_id}.json"))
    write_json(OUT/phase/f'{job_id}.json', dict(schema='lns2.reservation_mediation_result.v1',
               phase=phase, job=job, plan_sha256=sha256_file(OUT/'plan.json'), status='ok', data=data))


def read_result(phase, job):
    r=read_json(OUT/phase/f"{job['id']}.json")
    require(r['phase']==phase and r['job']==job and r['status']=='ok', 'result identity')
    require(r['plan_sha256']==sha256_file(OUT/'plan.json'), 'result plan mismatch')
    return r


def collect(phase):
    plan=load()
    if phase=='round2':
        r=read_json(OUT/'round1-report.json')
        require(r['parity_passed']==24 and r['errors']==0, 'round1 parity gate')
        for j in plan['jobs']:
            require(sha256_file(OUT/'round1'/f"{j['id']}.json")==r['result_sha256'][j['id']], 'round1 changed')
    dest=OUT/phase; dest.mkdir(exist_ok=True)
    lock=OUT/'run.lock'
    with lock.open('x',encoding='utf-8') as f:
        json.dump(dict(pid=os.getpid(),phase=phase),f)
    active=[]; done=[]; pending=[]
    def status(kind,error=None):
        write_json(OUT/'run_status.json',dict(status=kind,phase=phase,completed=len(done),
                   total=len(plan['jobs']),active=len(active),error=error))
    try:
        for j in plan['jobs']:
            if (dest/f"{j['id']}.json").exists(): read_result(phase,j); done.append(j['id'])
            else: pending.append(j)
        while pending or active:
            while pending and len(active)<plan['workers'] and not (OUT/'STOP_AFTER_JOB').exists():
                j=pending.pop(0); log=(dest/f"{j['id']}.log").open('w')
                p=subprocess.Popen([sys.executable,'-B',__file__,'_worker',phase,j['id']],stdout=log,stderr=subprocess.STDOUT)
                active.append((p,log,time.monotonic(),j))
            for item in list(active):
                p,log,start,j=item
                if p.poll() is None:
                    if time.monotonic()-start>plan['job_fuse']: raise TimeoutError(j['id'])
                    continue
                log.close(); active.remove(item)
                require(p.returncode==0,'worker failed: '+j['id'])
                read_result(phase,j); done.append(j['id'])
                print(json.dumps(dict(phase=phase,completed=len(done),job=j['id'])),flush=True)
            status('running')
            if not active and (OUT/'STOP_AFTER_JOB').exists(): break
            time.sleep(.3)
        status('complete' if len(done)==len(plan['jobs']) else 'paused')
    except BaseException as exc:
        status('error',repr(exc)); raise
    finally:
        for p,log,_,_ in active:
            if p.poll() is None: p.kill()
            p.wait(); log.close()
        lock.unlink()


def analyze(phase):
    plan=load(); rows=[read_result(phase,j) for j in plan['jobs']]
    report=dict(schema='lns2.reservation_mediation_report.v1',phase=phase,errors=0,
                plan_sha256=sha256_file(OUT/'plan.json'),
                result_sha256={r['job']['id']:sha256_file(OUT/phase/f"{r['job']['id']}.json") for r in rows})
    if phase=='round1':
        report.update(parity_passed=24,first_divergent_agents=[r['data']['explanation'] for r in rows],
                      changed_queries=[sum(q['path_changed'] for q in r['data']['queries']) for r in rows],
                      query_count=sum(len(r['data']['queries']) for r in rows),
                      decision='controlled_sensitivity_only_not_full_rng_causal_proof')
    else:
        values=[r['data']['path_check']['conflicts'] for r in rows]
        base=[r['data']['baseline_conflicts']['compressed11'] for r in rows]
        better=sum(a<b for a,b in zip(values,base))
        passed=better>=4 and sum(values)<=sum(base)
        report.update(conflicts=values,baseline11=base,
                      baseline12=[r['data']['baseline_conflicts']['compressed12'] for r in rows],
                      better=better,worse=sum(a>b for a,b in zip(values,base)),
                      rollback=sum(r['data']['result']['rolled_back'] for r in rows),
                      mean=sum(values)/len(values),passed=passed,
                      decision='posthoc_opportunity_only' if passed else 'no_go')
    write_json(OUT/f'{phase}-report.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','verify','collect','analyze','_worker'))
    p.add_argument('phase',nargs='?',choices=('round1','round2'))
    p.add_argument('job',nargs='?')
    a=p.parse_args()
    if a.command=='prepare': prepare()
    elif a.command=='verify': load(); print('Verified registered inputs')
    elif a.command=='_worker': worker(a.phase,a.job)
    elif a.command=='collect': collect(a.phase)
    else: analyze(a.phase)


if __name__=='__main__':
    main()
