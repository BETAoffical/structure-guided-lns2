"""Registered full-neighborhood mechanism collection with a fail-closed timing gate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments._common import read_json, write_json, write_jsonl, _native_filesystem_path
from experiments.closed_loop_trace_storage import apply_state_delta
from experiments.full_neighborhood_recovery import (
    METHODS, SearchBudget, mechanism_gate, recover, run_sequence, scientific_signature,
)
from experiments.local_path_compatibility import ROOT, contained, digest, run_lock, sha256_file
from experiments.local_path_search import validate_state
from experiments.native_path_compatibility import (
    check_seal, load as load_previous, modules, paths_of, seal,
)
from experiments.repair_collection import state_fingerprint
from lns2_selector.runtime.fingerprints import repair_structure_fingerprint

SCHEMA = 'lns2.full_neighborhood_recovery.v1'
CODE = ('experiments/full_neighborhood_recovery.py', 'experiments/full_recovery_collection.py',
        'scripts/diagnose_full_neighborhood_recovery.py', 'experiments/closed_loop_trace_storage.py',
        'experiments/repair_collection.py', 'lns2_selector/runtime/fingerprints.py')


def choose_sources(episodes, anchors, limit, seed):
    maps = {c['map_id'] for c in anchors}
    used_jobs = {c['source_job_id'] for c in anchors}
    used_tasks = {c['task_id'] for c in anchors}
    groups = defaultdict(list)
    for episode in episodes:
        if episode['controller'] not in ('v2-full', 'dual16') or episode['map_id'] not in maps or episode['steps'] == 0:
            continue
        if episode['job_id'] in used_jobs or episode['task_id'] in used_tasks:
            continue
        role = ('failed_long' if not episode['success'] else
                'successful_stall' if episode['max_same_state_run'] >= 3 else 'fast_control')
        groups[(role, episode['map_id'], episode['controller'])].append(dict(episode, recovery_role=role))
    for rows in groups.values():
        rows.sort(key=lambda r: digest([seed, r['job_id']]))
    selected = []
    while len(selected) < limit and any(groups.values()):
        for key in sorted(groups):
            while groups[key] and groups[key][0]['task_id'] in used_tasks:
                groups[key].pop(0)
            if not groups[key]:
                continue
            row = groups[key].pop(0)
            selected.append(row)
            used_tasks.add(row['task_id'])
            if len(selected) == limit:
                break
    return selected


def extract_source(args):
    output_name, source_name, row, task = args
    output, source = contained(output_name), contained(source_name)
    trace = contained(row['source_trace'])
    if sha256_file(trace) != row['trace_sha256']:
        raise ValueError('source trace SHA mismatch')
    step = row['repeat_state_examples'][0]['first_step'] if row['recovery_role'] != 'fast_control' and row['repeat_state_examples'] else 1
    target, state, blob = None, None, None
    prefix = []
    with gzip.open(_native_filesystem_path(trace), 'rt', encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            if event['event'] == 'initial':
                blob = (source/'episodes'/row['job_id']/'first_phase'/event['state_blob']).resolve()
                if not blob.is_relative_to(source):
                    raise ValueError('state blob path escapes source')
                with gzip.open(_native_filesystem_path(blob), 'rt', encoding='utf-8') as reader:
                    state = json.load(reader)
                if state_fingerprint(state) != event['state_fingerprint']:
                    raise ValueError('initial blob fingerprint mismatch')
            elif event['event'] == 'transition':
                if state is None or state_fingerprint(state) != event['before_fingerprint']:
                    raise ValueError('trace prefix fingerprint mismatch')
                if event['decision_index'] == step-1:
                    target = event
                    break
                prefix.append(dict(decision_index=event['decision_index'], action=event['action'],
                    metrics=event['metrics'], before_fingerprint=event['before_fingerprint'], after_fingerprint=event['after_fingerprint']))
                state = apply_state_delta(state, event['state_delta'])
                if state_fingerprint(state) != event['after_fingerprint']:
                    raise ValueError('trace post-state fingerprint mismatch')
    if target is None or len(prefix) != step-1:
        raise ValueError('missing source decision')
    validate_state(state)
    action, metrics = target['action'], target['metrics']
    if action['mode'] != 'explicit_neighborhood' or sorted(action['agents']) != sorted(metrics['repair_order']):
        raise ValueError('source must provide unchanged full explicit neighborhood and order')
    case_id = 'full-' + digest([row['job_id'], step])[:16]
    folder = output/'states'/case_id
    write_json(folder/'state.json', state)
    write_json(folder/'transition.json', dict(action=action, metrics=metrics))
    write_jsonl(folder/'prefix.jsonl', prefix)
    case = dict(case_id=case_id, role=row['recovery_role'], controller=row['controller'], map_id=row['map_id'],
        task_id=row['task_id'], solver_seed=int(row['solver_seed']), source_job_id=row['job_id'], decision_index=step-1,
        state_file=(folder/'state.json').relative_to(ROOT).as_posix(),
        historical_transition_file=(folder/'transition.json').relative_to(ROOT).as_posix(),
        prefix_file=(folder/'prefix.jsonl').relative_to(ROOT).as_posix(),
        source_trace=row['source_trace'], source_trace_sha256=row['trace_sha256'],
        initial_blob=blob.relative_to(ROOT).as_posix(), full_state_fingerprint=state_fingerprint(state),
        repair_structure_fingerprint=repair_structure_fingerprint(state), post_hoc=True)
    for name in ('map_file', 'scenario_file', 'task_file'):
        case[name] = (source/'inputs'/task[name]).relative_to(ROOT).as_posix()
    return case


def make_jobs(cases, seeds):
    jobs = []
    for case in cases:
        for seed in seeds:
            for method in ('parity',) + METHODS:
                job = dict(case_id=case['case_id'], seed=seed, method=method)
                jobs.append(dict(job, job_id=digest(job)[:20]))
    return jobs


def prepare(config_path):
    config = read_json(config_path)
    if config['schema'] != SCHEMA or not 1 <= config['workers'] <= 20:
        raise ValueError('invalid configuration')
    output = contained(config['output'])
    if not output.is_relative_to(ROOT/'build'):
        raise ValueError('output must be inside build')
    if (output/'manifest.json').exists():
        manifest = load(output)
        if manifest['config'] != config:
            raise ValueError('configuration mismatch')
        return summary(manifest)
    previous = load_previous(contained(config['previous_output']))
    if not read_json(contained(config['previous_output'])/'verification.json')['passed']:
        raise ValueError('previous native gate is not verified')
    source = contained(config['source_output'])
    audit = read_json(contained(config['transition_audit']))
    if audit['status'] != 'complete' or audit['integrity_errors'] != 0:
        raise ValueError('invalid historical audit')
    if sha256_file(source/'pressure_analysis/episodes.csv') != audit['input_csv_sha256']:
        raise ValueError('historical CSV mismatch')
    if sha256_file(ROOT/'experiments/closed_loop_trace_storage.py') != audit['decoder_sha256']:
        raise ValueError('historical trace decoder changed')
    tasks = [json.loads(line) for line in (source/'inputs/manifest.jsonl').read_text().splitlines()]
    lookup = {t['task_id']: t for t in tasks}
    cases = list(previous['cases'])
    selected = choose_sources(audit['episodes'], cases, config['maximum_states']-len(cases), config['sampling_seed'])
    with run_lock(output):
        write_json(output/'preparation_status.json', dict(status='extracting', workers=config['workers'], sources=len(selected)))
        args = [(config['output'], config['source_output'], row, lookup[row['task_id']]) for row in selected]
        with ProcessPoolExecutor(max_workers=min(config['workers'], max(1, len(args)))) as pool:
            for case in pool.map(extract_source, args):
                cases.append(case)
                print('prepared', len(cases), case['case_id'], case['role'], flush=True)
        # Do not count identical path/conflict snapshots as independent states.
        seen, unique, duplicates = set(), [], []
        for case in cases:
            key = case['repair_structure_fingerprint']
            if key in seen:
                duplicates.append(case['case_id'])
            else:
                unique.append(case)
                seen.add(key)
        cases = unique
        training_maps = {c['map_id'] for c in cases}
        heldout = []
        for map_id in sorted({t['map_id'] for t in tasks}-training_maps):
            available = [t for t in tasks if t['map_id'] == map_id]
            available.sort(key=lambda t: digest([config['sampling_seed'], t['task_id']]))
            # The fixed source dataset has task metadata with OD identifiers.
            chosen, od_seen = [], set()
            for task in available:
                meta = read_json(source/'inputs'/task['task_file'])
                ratio = meta['metadata']['required_bottleneck_crossing_ratio']
                od = 'bottleneck_eligible' if ratio > 0 else 'balanced_service'
                if od in od_seen:
                    continue
                chosen.append(dict(task, od_mode=od))
                od_seen.add(od)
                if len(chosen) == 2:
                    break
            heldout.extend(chosen)
        files = dict(previous['files'])
        names = [*CODE, config_path.relative_to(ROOT).as_posix(), config['transition_audit'],
            config['source_output']+'/inputs/manifest.jsonl', config['source_output']+'/pressure_analysis/episodes.csv',
            config['previous_output']+'/manifest.json', config['previous_output']+'/report.json',
            config['previous_output']+'/verification.json']
        for case in cases:
            names.extend(case[k] for k in ('state_file', 'historical_transition_file', 'prefix_file', 'map_file',
                                          'scenario_file', 'task_file', 'source_trace'))
            if 'initial_blob' in case:
                names.append(case['initial_blob'])
        for task in heldout:
            names.extend((source/'inputs'/task[key]).relative_to(ROOT).as_posix()
                         for key in ('map_file','scenario_file','task_file'))
        for name in names:
            files[name] = sha256_file(contained(name))
        for key in ('native_file', 'probe_file', 'core_file'):
            if config[key] != previous['config'][key]:
                raise ValueError('unexpected binary change before native integration')
        manifest = seal(dict(schema=SCHEMA, config=config, cases=cases, jobs=make_jobs(cases, config['pp_seeds']),
            files=files, excluded_duplicate_states=duplicates, heldout_tasks=heldout,
            timing_authorized=True, native_transaction_ready=False, training_maps=sorted(training_maps),
            timing_gate='mechanism_pass AND transaction_tests_pass AND preregistered_serial_schedule'))
        write_json(output/'manifest.json', manifest)
        write_json(output/'preparation_status.json', dict(status='completed', **summary(manifest)))
        return summary(manifest)


def load(output, verify_files=True):
    manifest = read_json(output/'manifest.json')
    check_seal(manifest)
    if manifest['schema'] != SCHEMA:
        raise ValueError('schema mismatch')
    if verify_files:
        for name, expected in manifest['files'].items():
            if sha256_file(contained(name)) != expected:
                raise ValueError('registered input changed: ' + name)
    return manifest


def summary(m):
    return dict(states=len(m['cases']), jobs=len(m['jobs']),
        parity_jobs=sum(j['method']=='parity' for j in m['jobs']),
        mechanism_jobs=sum(j['method']!='parity' for j in m['jobs']),
        workers=m['config']['workers'], maps=m['training_maps'],
        heldout_tasks=len(m['heldout_tasks']), heldout_maps=len({t['map_id'] for t in m['heldout_tasks']}),
        fingerprint=m['content_sha256'])


def output_file(output, job):
    return output/'results'/(job['job_id']+'.json')


def check_output(m, job, r):
    check_seal(r)
    if r['job'] != job or r['fingerprint'] != m['content_sha256'] or r['schema'] != SCHEMA:
        raise ValueError('result identity mismatch')


def worker(output, job_id):
    m = load(output)
    config = m['config']
    job = next(j for j in m['jobs'] if j['job_id'] == job_id)
    case = next(c for c in m['cases'] if c['case_id'] == job['case_id'])
    result = dict(schema=SCHEMA, job=job, fingerprint=m['content_sha256'])
    try:
        native, extension = modules(config)
        state = read_json(contained(case['state_file']))
        if state_fingerprint(state) != case['full_state_fingerprint']:
            raise ValueError('case fingerprint changed')
        transition = read_json(contained(case['historical_transition_file']))
        order = transition['metrics']['repair_order']
        if sorted(order) != sorted(transition['action']['agents']):
            raise ValueError('historical full neighborhood was changed')
        env = native.LNS2RepairEnv(str(contained(case['map_file'])), str(contained(case['scenario_file'])),
                                  len(state['agents']), time_limit=config['job_seconds']+30)
        restored = env.reset_paths(paths_of(state), seed=job['seed'])
        if paths_of(restored) != paths_of(state) or restored['num_of_colliding_pairs'] != state['num_of_colliding_pairs']:
            raise ValueError('native reset mismatch')
        probe = extension.NativePathProbe(str(contained(case['map_file'])), str(contained(case['scenario_file'])), paths_of(state))
        if job['method'] == 'parity':
            ref = env.step(dict(mode='explicit_neighborhood', agents=transition['action']['agents'], repair_order=order,
                pp_random_seed=job['seed'], collect_pp_diagnostics=True, pp_time_limit_seconds=config['job_seconds']/2))
            test = run_sequence(probe, state, order, job['seed'], SearchBudget(config['job_seconds']/2, config['job_seconds']/2))
            metrics = ref['metrics']
            if metrics['pp_failure_reason'] == 'time_limit' or test['status']=='unknown':
                result.update(status='unknown', reason='parity_censored', candidate=test, reference=metrics)
            elif test['status'] not in ('accepted', 'rolled_back'):
                result.update(status='error', reason='unexpected_empty_soft_baseline', candidate=test)
            else:
                checks = dict(paths=test['paths']==paths_of(ref['observation']), action_valid=metrics['action_valid'],
                    order=metrics['repair_order']==order, rollback=(test['status']=='rolled_back')==metrics['pp_rolled_back'],
                    conflict_pairs=test['attempted_pairs']==metrics['pp_attempt_conflict_pair_count'],
                    attempted=len(test['records'])==metrics['pp_attempted_agent_count'],
                    costs=[r['search']['cost'] for r in test['records']]==[r['path_cost_after'] for r in metrics['pp_agent_diagnostics']],
                    collisions=[r['search']['low_level_collisions'] for r in test['records']]==[r['low_level_collision_count'] for r in metrics['pp_agent_diagnostics']])
                result.update(status='pass' if all(checks.values()) else 'mismatch', checks=checks, candidate=test,
                              reference=metrics, reference_paths=paths_of(ref['observation']))
        else:
            value = recover(probe, state, order, job['seed'], job['method'], config)
            result.update(value)
            if value['status'] == 'ok':
                paths = value['final']['paths']
                verified = env.reset_paths(paths, seed=job['seed'])
                if paths_of(verified) != paths or verified['num_of_colliding_pairs'] != value['path_check']['conflicts']:
                    raise ValueError('native final-path verification failed')
                result['native_paths_verified'] = True
        import resource
        result['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        result.update(status='error', traceback=traceback.format_exc())
    write_json(output_file(output, job), seal(result))


def parity_gate(m, output):
    for job in m['jobs']:
        if job['method'] == 'parity':
            row = read_json(output_file(output, job))
            check_output(m, job, row)
            if row['status'] != 'pass':
                raise ValueError('parity incomplete or failed')


def available_memory_gib():
    path = Path('/proc/meminfo')
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/(1024*1024)
    return None


def collect(output, phase, workers=None, resume=False, max_jobs=None):
    m = load(output)
    workers = workers or m['config']['workers']
    if not 1 <= workers <= 20 or (max_jobs is not None and max_jobs < 1):
        raise ValueError('invalid concurrency or job limit')
    if phase == 'mechanism':
        parity_gate(m, output)
    jobs = [j for j in m['jobs'] if (j['method']=='parity') == (phase=='parity')]
    with run_lock(output):
        active, results, pending = {}, [], []
        for job in jobs:
            if output_file(output, job).exists():
                if not resume:
                    raise ValueError('existing output requires resume')
                r = read_json(output_file(output, job))
                check_output(m, job, r)
                if r['status'] in ('error', 'mismatch'):
                    raise ValueError('inspect failed run before resuming')
                results.append(r)
            else:
                pending.append(job)
        started, launched, stopping, failed = time.monotonic(), 0, False, False
        def status(value):
            write_json(output/'run_status.json', dict(status=value, phase=phase, completed=len(results), total=len(jobs),
                workers=workers, available_memory_gib=available_memory_gib(), fingerprint=m['content_sha256'],
                active=[dict(job_id=j, pid=v[0].pid) for j,v in active.items()]))
            write_jsonl(output/(phase+'_manifest.jsonl'), [dict(job_id=r['job']['job_id'], status=r['status'],
                path=output_file(output,r['job']).relative_to(ROOT).as_posix(), sha256=sha256_file(output_file(output,r['job']))) for r in results])
        try:
            status('running')
            while pending or active:
                stopping |= (output/'STOP').exists() or (max_jobs is not None and launched >= max_jobs)
                stopping |= time.monotonic()-started > m['config']['collection_session_hours']*3600
                while pending and len(active)<workers and not stopping:
                    available = available_memory_gib()
                    if available is not None and available < m['config']['minimum_available_memory_gib']:
                        if not active:
                            stopping = True
                        break
                    job = pending.pop(0)
                    log = output/'logs'/(job['job_id']+'.log')
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('wb') as stream:
                        child = subprocess.Popen([sys.executable, str(ROOT/'scripts/diagnose_full_neighborhood_recovery.py'),
                            '_job', '--output', str(output), '--job-id', job['job_id']], cwd=ROOT,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    active[job['job_id']] = (child, time.monotonic(), job)
                    launched += 1
                    stopping |= max_jobs is not None and launched>=max_jobs
                    status('running')
                for jid, (child, began, job) in list(active.items()):
                    if child.poll() is None and time.monotonic()-began > m['config']['process_fuse_seconds']:
                        child.kill()
                        child.wait()
                        if not output_file(output,job).exists():
                            write_json(output_file(output,job), seal(dict(schema=SCHEMA, fingerprint=m['content_sha256'],
                                job=job, status='unknown', reason='external_fuse')))
                    if child.poll() is not None:
                        del active[jid]
                        if not output_file(output,job).exists():
                            write_json(output_file(output,job), seal(dict(schema=SCHEMA, fingerprint=m['content_sha256'],
                                job=job, status='error', reason='worker_exit', exit_code=child.returncode)))
                        r = read_json(output_file(output,job))
                        check_output(m,job,r)
                        results.append(r)
                        failed |= r['status'] in ('error','mismatch') or (phase=='parity' and r['status']!='pass')
                        failed |= child.returncode != 0 and r.get('reason') != 'external_fuse'
                        stopping |= failed
                        print(f"{phase} {len(results)}/{len(jobs)} {job['case_id']} {job['method']} {r['status']} recovered={r.get('recovered')}", flush=True)
                        with (output/'progress.jsonl').open('a',encoding='utf-8') as f:
                            f.write(json.dumps(dict(job_id=jid,status=r['status'],time=time.time()))+'\n')
                        status('stopping' if stopping else 'running')
                if stopping and not active:
                    break
                try:
                    time.sleep(.25)
                except KeyboardInterrupt:
                    stopping = True
            status('failed' if failed else 'completed' if not pending else 'paused')
        finally:
            for child, _, _ in active.values():
                if child.poll() is None:
                    child.terminate()
                child.wait()
            if active:
                active.clear()
                status('interrupted')
        if failed:
            raise RuntimeError('run stopped; inspect failure, never retry by increasing budget')


def failure_label(row):
    if row['status'] != 'ok':
        return row['status']
    if row['budget_exhausted']:
        return 'budget_unknown'
    if row['recovered']:
        return 'recovered'
    if not row['triggered']:
        return 'ordinary_improved' if row['base'].get('strict_decrease') else 'ordinary_changed_not_retried'
    feedback = row.get('feedback')
    if feedback and not feedback['candidates']:
        return 'external_feedback_only' if feedback['external_blockers'] else 'no_usable_internal_feedback'
    attempts = row['attempts']
    if not attempts:
        return 'ordinary_stalled'
    if all(a['result']['status']=='not_found' for a in attempts):
        return 'cost_bounded_path_search_failed'
    return 'alternatives_do_not_reduce_conflicts'


def analyze(output):
    m = load(output)
    parity_rows, rows = [], []
    for job in m['jobs']:
        if output_file(output,job).exists():
            row=read_json(output_file(output,job))
            check_output(m,job,row)
            (parity_rows if job['method']=='parity' else rows).append(row)
    groups=defaultdict(list)
    for row in rows:
        groups[(row['job']['case_id'],row['job']['seed'])].append(row)
    baseline_mismatches=[]
    for key, group in groups.items():
        signatures = [digest(scientific_signature(r['base'])) for r in group if r['status']=='ok']
        if len(set(signatures))>1:
            baseline_mismatches.append(list(key))
    expected=len(m['cases'])*len(m['config']['pp_seeds'])*len(METHODS)
    gate=mechanism_gate(rows,m['cases'],expected)
    if len(m['cases']) < m['config']['maximum_states']:
        gate=dict(passed=False,reason='insufficient_distinct_states',mechanism_details=gate)
    parity_ok=len(parity_rows)==len(m['cases'])*len(m['config']['pp_seeds']) and all(r['status']=='pass' for r in parity_rows)
    if not parity_ok or baseline_mismatches:
        gate=dict(passed=False,reason='parity_or_paired_baseline_mismatch', mechanism_details=gate)
    aggregate={}
    for method in METHODS:
        group=[r for r in rows if r['job']['method']==method]
        aggregate[method]=dict(n=len(group), recovered=sum(r.get('recovered',False) for r in group),
            strict_decrease=sum(r.get('final',{}).get('strict_decrease',False) for r in group),
            expanded=sum(r.get('expanded',0) for r in group), generated=sum(r.get('generated',0) for r in group),
            failures=dict(Counter(failure_label(r) for r in group)),
            diagnostic_seconds=sum(r.get('diagnostic_seconds',0) for r in group))
    value=seal(dict(schema=SCHEMA, fingerprint=m['content_sha256'], parity_passed=parity_ok,
        complete=len(rows)==expected, baseline_mismatches=baseline_mismatches, methods=aggregate,
        gate=gate, decision='mechanism_pass_requires_native_transaction_validation' if gate['passed'] else 'mechanism_no_go',
        timing_allowed=False, timing_blocker='native_transaction_not_integrated' if gate['passed'] else 'mechanism_gate_failed',
        recovered_conditions=[dict(job=r['job'], path_check=r['path_check']) for r in rows if r.get('recovered')],
        result_hashes={output_file(output,r['job']).relative_to(ROOT).as_posix():sha256_file(output_file(output,r['job'])) for r in parity_rows+rows}))
    write_json(output/'report.json',value)
    write_jsonl(output/'failure_analysis.jsonl',[dict(job=r['job'],category=failure_label(r),
        base_status=r.get('base',{}).get('status'), feedback=r.get('feedback'),
        attempt_statuses=[a['result']['status'] for a in r.get('attempts',[])]) for r in rows])
    return {k:v for k,v in value.items() if k not in ('result_hashes','recovered_conditions')}


def timing_readiness(output):
    m=load(output)
    report=read_json(output/'report.json')
    check_seal(report)
    if report['fingerprint']!=m['content_sha256']:
        raise ValueError('timing gate report identity mismatch')
    return dict(authorized=m['timing_authorized'], ready=False,
        mechanism_pass=report['gate']['passed'], native_transaction_ready=False,
        heldout_tasks=len(m['heldout_tasks']), reason=report['timing_blocker'],
        policy='No timing launcher exists until all native transaction and registration gates pass.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('prepare','dry-run','parity','diagnose','analyze','timing-readiness','_job'))
    parser.add_argument('--config',default='configs/full_neighborhood_recovery_v1.json')
    parser.add_argument('--output',default='build/initlns-full-neighborhood-recovery-v1-final')
    parser.add_argument('--workers',type=int)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-jobs',type=int)
    parser.add_argument('--job-id')
    args=parser.parse_args()
    p=Path(args.output)
    output=p.resolve() if p.is_absolute() else contained(args.output)
    if not output.is_relative_to(ROOT/'build'):
        raise ValueError('output outside build')
    if args.mode=='prepare':
        result=prepare(contained(args.config))
    elif args.mode=='dry-run':
        result=summary(load(output))
    elif args.mode=='_job':
        worker(output,args.job_id)
        return
    elif args.mode in ('parity','diagnose'):
        collect(output,'parity' if args.mode=='parity' else 'mechanism',args.workers,args.resume,args.max_jobs)
        return
    elif args.mode=='timing-readiness':
        result=timing_readiness(output)
    else:
        result=analyze(output)
    print(json.dumps(result,indent=2))
