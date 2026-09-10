"""Finite single-external soft-CAT opportunity census; never an online policy."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from experiments._common import read_json, write_json
from experiments.full_neighborhood_recovery import SearchBudget, check_paths, run_sequence
from experiments.local_path_compatibility import digest, run_lock, sha256_file
from experiments.native_path_compatibility import check_seal, edge_set, modules, paths_of, seal
from scripts.diagnose_reservation_feedback import load as load_source
from scripts.diagnose_reservation_mediation import load as load_witness, released_pp, require
from scripts.diagnose_search_occupancy import signature, write_gzip

CONFIG = ROOT/'configs/single_release_opportunity_v1.json'
REG = ROOT/'artifacts/initlns-single-release-opportunity-v1/registration.json'


def select_cases(jobs, salt):
    groups = defaultdict(dict)
    for j in jobs:
        c = j['case']
        if c['role'] in ('failed_long', 'failed_long_unchanged'):
            groups[c['map_id']][c['case_id']] = c
    return [min(group.values(), key=lambda c: hashlib.sha256(
        (salt+c['case_id']).encode()).hexdigest()) for _, group in sorted(groups.items())]


def make_jobs(conditions, states):
    jobs = []
    for c in conditions:
        ids = [a['id'] for a in states[c['case']['case_id']]['agents']]
        require(ids == list(range(len(ids))), 'native requires scenario-indexed IDs')
        require(len(set(c['order'])) == len(c['order']) and set(c['order']) <= set(ids), 'order')
        for agent in sorted(set(ids)-set(c['order'])):
            value = dict(condition=c['id'], released=agent)
            jobs.append(dict(value, id=digest(value)[:24]))
    require(len({j['id'] for j in jobs}) == len(jobs), 'duplicate job')
    return jobs


def prepare():
    cfg = read_json(CONFIG); source = load_source(); witness = load_witness()
    cases = select_cases(source['jobs'], cfg['selection_salt'])
    require([c['case_id'] for c in cases] == cfg['expected_cases'], 'frozen state selection')
    selected = {c['case_id'] for c in cases}
    conditions = [{k: j[k] for k in ('id', 'case', 'order', 'seed', 'source_result')}
                  for j in source['jobs'] if j['case']['case_id'] in selected]
    require(len(conditions) == 6, 'six baseline conditions required')
    for c in cases:
        require(sorted(j['seed'] for j in conditions if j['case']==c) == cfg['seeds'], 'seed isolation')
    states = {c['case_id']: read_json(ROOT/c['state_file']) for c in cases}
    jobs = make_jobs(conditions, states)
    require(len(jobs) == cfg['expected_jobs'], 'job count')
    files = dict(source['files']); files.update(witness['files'])
    evpath = ROOT/'artifacts/initlns-reservation-mediation-v1/evidence.json'
    ev = read_json(evpath); reportpath = ROOT/ev['round2_report']
    require(sha256_file(reportpath) == ev['round2_sha256'], 'known witness report changed')
    report = read_json(reportpath); controls = []
    for trial in (0, 1):
        old = f'build/initlns-compressed-neighborhood-native-v1/{trial:02d}-compressed11.json'
        expected = f'build/initlns-reservation-mediation-v1/round2/trial-{trial:02d}.json'
        require(sha256_file(ROOT/expected) == report['result_sha256'][f'trial-{trial:02d}'], 'witness row SHA')
        action = read_json(ROOT/old)['job']['action']
        controls.append(dict(case=witness['case'], state_file='build/initlns-compressed-neighborhood-native-v1/before.json',
                             order=action['repair_order'], seed=action['pp_random_seed'],
                             released=witness['released_agent'], expected=expected))
        files[expected] = sha256_file(ROOT/expected)
    for path in (CONFIG, Path(__file__), ROOT/'tests/test_single_release_opportunity.py',
                 ROOT/'docs/SINGLE_RELEASE_OPPORTUNITY_PROTOCOL_ZH.md', evpath, reportpath,
                 ROOT/'scripts/diagnose_search_occupancy.py'):
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    plan = seal(dict(schema=cfg['schema'], config=cfg, native=source['config'], cases=cases,
                     conditions=conditions, jobs=jobs, controls=controls, files=files))
    out = ROOT/cfg['output']; out.mkdir(parents=True, exist_ok=True)
    require(not (out/'plan.json').exists(), 'use a new output version, not overwrite')
    write_json(out/'plan.json', plan)
    write_json(REG, dict(schema=cfg['schema'], plan_file=cfg['output']+'/plan.json',
                         plan_sha256=sha256_file(out/'plan.json'), jobs=len(jobs),
                         states=len(cases), scope=cfg['scope']))
    return dict(jobs=len(jobs), baselines=len(conditions), witness_controls=len(controls),
                workers=cfg['workers'], search_cpu_budget_seconds=len(jobs)*cfg['search_seconds'],
                session_seconds=cfg['session_seconds'], plan_sha256=sha256_file(out/'plan.json'))


def load():
    reg = read_json(REG); path = ROOT/reg['plan_file']
    require(sha256_file(path) == reg['plan_sha256'], 'registration mismatch')
    p = read_json(path); check_seal(p)
    for file, h in p['files'].items():
        require(sha256_file(ROOT/file) == h, 'bound file changed: '+file)
    return p, path.parent


def reconstruct(state, order, raw):
    """Rebuild final paths and audit real occupancy, including the omitted CAT path."""
    require(raw['status'] == 'ok', 'only completed PP has paths')
    original = {a['id']: a['path'] for a in state['agents']}
    visible = {i: path for i, path in original.items() if i not in order}
    bound = len(edge_set(original, order)); pairs = set(); overrides = {}
    records = raw['diagnostics']
    require([r['agent'] for r in records] == order[:len(records)], 'diagnostic order/prefix')
    require(0 < len(records) <= len(order), 'record count')
    for index, record in enumerate(records):
        require(record['status'] == 'path' and record['path'], 'invalid completed search')
        aid = record['agent']; visible[aid] = record['path']; overrides[aid] = record['path']
        pairs |= edge_set(visible, [aid])
        if len(pairs) > bound:
            require(raw['rolled_back'] and index == len(records)-1, 'rollback boundary')
        else:
            require(not raw['rolled_back'] or index != len(records)-1, 'spurious rollback')
    require(len(pairs) == raw['attempted_pairs'], 'actual conflict ledger mismatch')
    if not raw['rolled_back']:
        require(len(records) == len(order), 'accepted incomplete repair')
    return [original[i] if raw['rolled_back'] else overrides.get(i, original[i]) for i in original]


def compact(raw):
    result = {k: v for k, v in raw.items() if k != 'paths'}
    if 'paths' in raw:
        result['paths_sha256'] = digest(raw['paths'])
    return result


class Engine:
    def __init__(self, plan):
        self.plan = plan
        self.native, self.module = modules(plan['native'])
        self.cache = {}

    def context(self, case, state_file=None):
        key = state_file or case['state_file']
        if key not in self.cache:
            state = read_json(ROOT/key)
            probe = self.module.NativePathProbe(str(ROOT/case['map_file']), str(ROOT/case['scenario_file']), paths_of(state))
            env = self.native.LNS2RepairEnv(str(ROOT/case['map_file']), str(ROOT/case['scenario_file']),
                                          len(state['agents']), time_limit=60)
            self.cache[key] = state, probe, env
        return self.cache[key]

    def ordinary(self, c):
        state, probe, _ = self.context(c['case'])
        cfg = self.plan['config']
        return run_sequence(probe, state, c['order'], c['seed'],
                            SearchBudget(cfg['search_seconds'], cfg['call_seconds']))

    def release(self, c, agent, state_file=None):
        state, probe, env = self.context(c['case'], state_file)
        before = digest(state); cfg = self.plan['config']
        raw = released_pp(probe, state, c['order'], c['seed'], agent, cfg['search_seconds'], cfg['call_seconds'])
        require(digest(state) == before, 'state mutation')
        check = None
        if raw['status'] == 'ok':
            check = check_paths(state, raw['paths'], c['order'])
            # Every accepted result is independently loaded by the frozen native validator.
            observed = env.reset_paths(raw['paths'], seed=c['seed'])
            require(paths_of(observed) == raw['paths'] and observed['num_of_colliding_pairs'] == check['conflicts'],
                    'native validation mismatch')
        else:
            require(raw['status'] in ('unknown', 'not_found'), 'unexpected PP status')
        return raw, check


def preflight():
    plan, out = load(); engine = Engine(plan); rows = []
    for c in plan['conditions']:
        expected = read_json(ROOT/c['source_result'])['base']
        a = engine.ordinary(c)
        require(signature(a) == signature(expected), 'frozen baseline mismatch')
        state, _, _ = engine.context(c['case'])
        external = min(set(range(len(state['agents'])))-set(c['order']))
        b, _ = engine.release(c, external)
        again = engine.ordinary(c)
        require(signature(a) == signature(again), 'A-B-A process reuse mismatch')
        repeated, _ = engine.release(c, external)
        require(b['status'] == repeated['status'] == 'ok' and signature(b) == signature(repeated), 'release reuse mismatch')
        require(reconstruct(state, c['order'], compact(b)) == b['paths'], 'compact reconstruction')
        rows.append(dict(condition=c['id'], baseline_signature_sha=digest(signature(a)),
                         aba_equal=True, release_repeat_equal=True, conflicts=len(edge_set(dict(enumerate(a['paths']))))))
        print(json.dumps(dict(preflight_baseline=len(rows), total=6)), flush=True)
    controls = []
    for c in plan['controls']:
        raw, check = engine.release(c, c['released'], c['state_file'])
        expected = read_json(ROOT/c['expected'])['data']['result']
        require(signature(raw) == signature(expected), 'known opportunity witness mismatch')
        controls.append(dict(seed=c['seed'], conflicts=check['conflicts'], signature_sha=digest(signature(raw))))
    result = seal(dict(plan=plan['content_sha256'], passed=True, baselines=rows, controls=controls))
    write_json(out/'preflight.json', result)
    return result


def result_path(out, job):
    return out/'results'/(job['id']+'.json')


def read_result(plan, out, job):
    r = read_json(result_path(out, job)); check_seal(r)
    require(r['job'] == job and r['plan'] == plan['content_sha256'], 'result identity mismatch')
    require(r['status'] in ('ok', 'unknown', 'not_found'), 'invalid result status')
    if r.get('raw_sha256'):
        path = out/'raw'/(job['id']+'.json.gz')
        require(sha256_file(path) == r['raw_sha256'], 'raw result SHA')
    return r


def worker(plan, out_string, inbox, events):
    out = Path(out_string)
    try:
        engine = Engine(plan)
        conditions = {c['id']: c for c in plan['conditions']}
        events.put(('ready', os.getpid(), None))
        while True:
            job = inbox.get()
            if job is None:
                break
            events.put(('start', os.getpid(), dict(job=job, started=time.monotonic())))
            start = time.monotonic(); c = conditions[job['condition']]
            raw, check = engine.release(c, job['released'])
            packed = seal(dict(job=job, plan=plan['content_sha256'], raw=compact(raw)))
            path = out/'raw'/(job['id']+'.json.gz'); write_gzip(path, packed)
            row = seal(dict(job=job, plan=plan['content_sha256'], status=raw['status'],
                            raw_sha256=sha256_file(path), path_check=check,
                            rolled_back=raw.get('rolled_back'), attempted_pairs=raw.get('attempted_pairs'),
                            generated=sum(r.get('generated', 0) for r in raw['diagnostics']),
                            expanded=sum(r.get('expanded', 0) for r in raw['diagnostics']),
                            native_paths_verified=raw['status']=='ok', seconds=time.monotonic()-start))
            write_json(result_path(out, job), row)
            events.put(('done', os.getpid(), job))
    except BaseException:
        events.put(('error', os.getpid(), traceback.format_exc()))
        raise


def collect(limit=None, reviewed_error=False):
    plan, out = load(); cfg = plan['config']
    if (out/'run_status.json').exists():
        previous = read_json(out/'run_status.json')
        require(previous['status'] != 'error' or reviewed_error, 'inspect previous error before explicit resume')
    pre = read_json(out/'preflight.json'); check_seal(pre)
    require(pre['passed'] and pre['plan'] == plan['content_sha256'], 'preflight required')
    if sys.platform.startswith('linux'):
        mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        require(int(mem['MemAvailable'].split()[0])/1024**2 >= cfg['minimum_available_memory_gib'], 'memory headroom')
    for sub in ('raw', 'results'):
        (out/sub).mkdir(exist_ok=True)
    done = [j for j in plan['jobs'] if result_path(out, j).exists()]
    for j in done:
        read_result(plan, out, j)
    done_ids = {j['id'] for j in done}; pending = [j for j in plan['jobs'] if j['id'] not in done_ids]
    if limit is not None:
        require(limit > 0, 'limit must be positive')
        pending = pending[:limit]
    if not pending:
        return dict(status='complete', completed=len(done))
    context = mp.get_context('spawn'); events = context.Queue(); active = {}; processes = {}; inboxes = {}
    start = time.monotonic(); stopping = False; next_index = 0

    def status(kind, error=None):
        row = dict(status=kind, completed=len(done_ids), total=len(plan['jobs']), active=len(active),
                   seconds=time.monotonic()-start, error=error)
        write_json(out/'run_status.json', row)
        return row

    def launch():
        inbox = context.Queue(); proc = context.Process(target=worker, args=(plan, str(out), inbox, events))
        proc.start(); processes[proc.pid] = proc; inboxes[proc.pid] = inbox

    def assign(pid):
        nonlocal next_index
        if not stopping and next_index < len(pending):
            job = pending[next_index]; next_index += 1
            active[pid] = dict(job=job, started=time.monotonic(), phase='queued')
            inboxes[pid].put(job)
        else:
            inboxes[pid].put(None)

    with run_lock(out):
        try:
            for _ in range(min(cfg['workers'], len(pending))):
                launch()
            status('running')
            while processes:
                stopping = stopping or (out/'STOP_AFTER_JOB').exists() or time.monotonic()-start >= cfg['session_seconds']
                try:
                    kind, pid, payload = events.get(timeout=.3)
                except queue.Empty:
                    kind = None
                if kind == 'ready':
                    assign(pid)
                elif kind == 'start':
                    require(active[pid]['job'] == payload['job'], 'worker dispatch mismatch')
                    active[pid] = dict(payload, phase='running')
                elif kind == 'done':
                    require(active[pid]['job'] == payload, 'worker completion mismatch')
                    read_result(plan, out, payload); done_ids.add(payload['id']); active.pop(pid)
                    with (out/'progress.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(job=payload['id'], completed=len(done_ids), time=time.time()))+'\n')
                    if len(done_ids) % 100 == 0:
                        print(json.dumps(status('running')), flush=True)
                    assign(pid)
                elif kind == 'error':
                    raise RuntimeError(payload)
                for pid, item in list(active.items()):
                    if time.monotonic()-item['started'] > cfg['job_fuse_seconds']:
                        # Stop the cohort on an external fuse; do not silently rerun lost jobs.
                        p = processes[pid]; p.terminate(); p.join()
                        j = item['job']
                        if not result_path(out, j).exists():
                            write_json(result_path(out, j), seal(dict(job=j, plan=plan['content_sha256'],
                                status='unknown', reason='external_fuse', raw_sha256=None)))
                        raise TimeoutError('external fuse: '+j['id'])
                for pid, p in list(processes.items()):
                    if p.exitcode is not None:
                        require(p.exitcode == 0 and pid not in active, 'unexpected worker exit')
                        p.join(); processes.pop(pid); inboxes[pid].close()
                if time.monotonic()-start > cfg['session_seconds']+cfg['job_fuse_seconds']+10:
                    raise TimeoutError('session shutdown fuse')
                status('draining' if stopping else 'running')
            return status('complete' if len(done_ids)==len(plan['jobs']) else 'paused')
        except BaseException as exc:
            status('error', repr(exc)); raise
        finally:
            for p in processes.values():
                if p.is_alive():
                    p.terminate()
                p.join()
            for inbox in inboxes.values():
                inbox.close()
            events.close()


def summarize(plan, rows, baseline):
    by_condition = defaultdict(dict)
    for row in rows:
        j = row['job']; require(j['released'] not in by_condition[j['condition']], 'duplicate release')
        by_condition[j['condition']][j['released']] = row
    case_rows = []; known = sum(r['status']=='ok' for r in rows)
    for case in plan['cases']:
        conditions = [c for c in plan['conditions'] if c['case']==case]
        sets = []; seed_rows = []
        for c in conditions:
            results = by_condition[c['id']]; base = baseline[c['id']]
            good = {a for a, r in results.items() if r['status']=='ok' and r['path_check']['conflicts']<base}
            sets.append(good)
            counts = [r['path_check']['conflicts'] for r in results.values() if r['status']=='ok']
            seed_rows.append(dict(seed=c['seed'], baseline=base, improving_agents=sorted(good),
                                  evaluated=len(results), known=len(counts), best_conflicts=min(counts) if counts else None,
                                  rollback=sum(bool(r.get('rolled_back')) for r in results.values())))
        common = set.intersection(*sets)
        benefits = {a: sum((baseline[c['id']]-by_condition[c['id']][a]['path_check']['conflicts'])/
                           max(1, baseline[c['id']]) for c in conditions)/len(conditions) for a in common}
        best = min(common, key=lambda a: (-benefits[a], a)) if common else None
        case_rows.append(dict(case_id=case['case_id'], map_id=case['map_id'], seeds=seed_rows,
                              stable_agents=sorted(common), best_fixed_agent=best,
                              best_fixed_agent_mean_relative_reduction=benefits[best] if best is not None else 0))
    stable = [c for c in case_rows if c['stable_agents'] and
              c['best_fixed_agent_mean_relative_reduction'] >= plan['config']['minimum_mean_relative_reduction']]
    complete = len(rows)==len(plan['jobs']) and known==len(plan['jobs'])
    passed = complete and len(stable)>=plan['config']['minimum_stable_states']
    decision = ('opportunity_only_selector_not_validated' if passed else
                'incomplete_or_unknown' if not complete else
                'no_go_insufficient_stable_single_release_opportunity')
    return dict(cases=case_rows, total=len(rows), known=known, unknown_or_not_found=len(rows)-known,
                complete=complete, stable_states_at_five_percent=len(stable), passed=passed, decision=decision)


_AUDIT = None


def audit_init(plan, output):
    global _AUDIT
    _AUDIT = (plan, Path(output), {c['id']: c for c in plan['conditions']},
              {c['case_id']: read_json(ROOT/c['state_file']) for c in plan['cases']})


def audit_job(j):
    plan, out, conditions, states = _AUDIT
    r = read_result(plan, out, j); verified = False
    if r.get('raw_sha256'):
        with gzip.open(out/'raw'/(j['id']+'.json.gz'), 'rt', encoding='utf-8') as f:
            raw = json.load(f)
        check_seal(raw); require(raw['job']==j and raw['plan']==plan['content_sha256'], 'raw identity')
        raw = raw['raw']; require(raw['status']==r['status'], 'status mismatch')
        if raw['status']=='ok':
            c = conditions[j['condition']]; state = states[c['case']['case_id']]
            paths = reconstruct(state, c['order'], raw)
            require(digest(paths)==raw['paths_sha256'], 'paths digest mismatch')
            check = check_paths(state, paths, c['order'])
            require(check==r['path_check'] and r['native_paths_verified'], 'path summary mismatch')
            require(raw['rolled_back']==r['rolled_back'] and raw['attempted_pairs']==r['attempted_pairs'], 'ledger summary')
            for key in ('generated', 'expanded'):
                require(sum(d[key] for d in raw['diagnostics'])==r[key], 'node summary')
            verified = True
    return r, sha256_file(result_path(out, j)), verified


def analyze():
    plan, out = load(); pre = read_json(out/'preflight.json'); check_seal(pre)
    require(pre['passed'] and pre['plan']==plan['content_sha256'], 'preflight identity')
    baseline = {r['condition']: r['conflicts'] for r in pre['baselines']}
    require(len(baseline)==len(plan['conditions']), 'baseline completeness')
    for c in plan['conditions']:
        original = read_json(ROOT/c['source_result'])['base']
        require(baseline[c['id']]==len(edge_set(dict(enumerate(original['paths'])))), 'baseline counts')
        row = next(r for r in pre['baselines'] if r['condition']==c['id'])
        require(row['baseline_signature_sha']==digest(signature(original)), 'baseline signature')
    rows = []; hashes = {}; verified = 0
    jobs = [j for j in plan['jobs'] if result_path(out, j).exists()]
    with mp.get_context('spawn').Pool(min(plan['config']['workers'], max(1, len(jobs))),
                                      initializer=audit_init, initargs=(plan, str(out))) as pool:
        for r, h, valid in pool.imap(audit_job, jobs, chunksize=1):
            rows.append(r); hashes[r['job']['id']] = h; verified += valid
            if len(rows) % 200 == 0:
                print(json.dumps(dict(audited=len(rows), total=len(jobs))), flush=True)
    report = seal(dict(schema=plan['schema'], plan=plan['content_sha256'],
                       result_sha256=hashes, independently_verified=verified,
                       **summarize(plan, rows, baseline)))
    write_json(out/'report.json', report)
    return {k: v for k, v in report.items() if k not in ('result_sha256', 'cases')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'dry-run', 'preflight', 'collect', 'analyze'))
    parser.add_argument('--limit', type=int, help='collect only this many pending jobs, then safely pause')
    parser.add_argument('--resume-after-reviewed-error', action='store_true')
    args = parser.parse_args()
    if args.phase == 'prepare':
        result = prepare()
    elif args.phase == 'dry-run':
        plan, _ = load(); result = dict(jobs=len(plan['jobs']), conditions=len(plan['conditions']), config=plan['config'])
    elif args.phase == 'collect':
        result = collect(args.limit, args.resume_after_reviewed_error)
    else:
        result = {'preflight': preflight, 'analyze': analyze}[args.phase]()
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
