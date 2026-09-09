"""Isolated native parity gate and hard-path mechanism probes, never a controller."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from experiments._common import read_json, write_json, write_jsonl
from experiments.local_path_compatibility import (
    ROOT, contained, digest, load_manifest as load_reference, run_lock, sha256_file,
)
from experiments.local_path_search import directed_constraints, select_pairs, validate_state, validate_witness
from experiments.state_analysis import reconstruct_conflicts

SCHEMA = 'lns2.native_path_compatibility.v1'
CODE = ('experiments/native_path_compatibility.py', 'scripts/diagnose_native_path_compatibility.py',
        'src/path_probe/native_path_probe.cpp', 'src/path_probe/CMakeLists.txt')


def seal(value):
    return dict(value, content_sha256=digest(value))


def check_seal(value):
    if value['content_sha256'] != digest({k: v for k, v in value.items() if k != 'content_sha256'}):
        raise ValueError('content hash mismatch')


def make_jobs(cases, states, config):
    jobs = []
    for case in cases:
        for seed in config['parity_seeds']:
            jobs.append(dict(case_id=case['case_id'], kind='parity', seed=seed))
        for pair in select_pairs(states[case['case_id']], 2):
            for order in (pair, pair[::-1]):
                for method in ('ordinary', 'random_paths', 'directed_paths'):
                    jobs.append(dict(case_id=case['case_id'], kind='hard_pair', order=order,
                                     method=method, seed=config['path_seed']))
    return [dict(j, job_id=digest(j)[:20]) for j in jobs]


def prepare(config_file):
    config = read_json(config_file)
    if config['schema'] != SCHEMA:
        raise ValueError('configuration schema')
    reference_dir = contained(config['reference_output'])
    reference = load_reference(reference_dir)
    files = dict(reference['files'])
    for name in (*CODE, config_file.relative_to(ROOT).as_posix(),
                 config['native_file'], config['probe_file'], config['core_file'],
                 config['reference_output'] + '/manifest.json',
                 config['reference_output'] + '/report.json',
                 config['reference_output'] + '/verification.json'):
        files[name] = sha256_file(contained(name))
    if files[config['native_file']] != config['native_sha256']:
        raise ValueError('frozen native changed')
    states = {c['case_id']: read_json(contained(c['state_file'])) for c in reference['cases']}
    for state in states.values():
        validate_state(state)
        if [a['id'] for a in state['agents']] != list(range(len(state['agents']))):
            raise ValueError('scenario-indexed native adapter requires contiguous ordered IDs')
    manifest = seal(dict(schema=SCHEMA, config=config, files=files, cases=reference['cases'],
                         jobs=make_jobs(reference['cases'], states, config)))
    output = contained(config['output'])
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.json').exists() and read_json(output/'manifest.json') != manifest:
        raise ValueError('output already registered to different inputs')
    write_json(output/'manifest.json', manifest)
    return dict(output=config['output'], jobs=len(manifest['jobs']),
                phases=dict(Counter(j['kind'] for j in manifest['jobs'])),
                worst_search_seconds=len(manifest['jobs'])*config['job_seconds'],
                fingerprint=manifest['content_sha256'])


def load(output):
    manifest = read_json(output/'manifest.json')
    check_seal(manifest)
    if manifest['schema'] != SCHEMA:
        raise ValueError('manifest schema')
    for name, expected in manifest['files'].items():
        if sha256_file(contained(name)) != expected:
            raise ValueError('registered file changed: ' + name)
    return manifest


def modules(config):
    loaded = []
    for name, key in (('lns2_env', 'native_file'), ('lns2_path_probe_native', 'probe_file')):
        path = contained(config[key])
        sys.path.insert(0, str(path.parent))
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != path:
            raise ValueError('wrong module: ' + name)
        loaded.append(module)
    if loaded[0].native_semantics_schema != 'lns2.native_semantics.official_step_timed_extension.v3':
        raise ValueError('native semantics mismatch')
    if loaded[1].schema != 'lns2.native_path_probe.v1':
        raise ValueError('probe schema mismatch')
    return loaded


def paths_of(state):
    return [a['path'] for a in state['agents']]


def edge_set(paths, selected=None):
    edges = {(e.left, e.right) for e in reconstruct_conflicts(
        [dict(id=i, path=p) for i, p in paths.items()])}
    return edges if selected is None else {p for p in edges if set(p) & set(selected)}


def soft_pp(probe, state, order, seed, seconds):
    """Emulate the documented PP acceptance/rollback boundary for parity only."""
    original = dict(enumerate(paths_of(state)))
    fixed = [i for i in original if i not in order]
    old_edges = edge_set(original, order)
    overrides, attempted_edges, diagnostics = {}, set(), []
    started = time.monotonic()
    probe.seed_rng(seed)
    for aid in order:
        remaining = max(0., seconds - (time.monotonic()-started))
        found = probe.plan(aid, fixed, overrides, False, seconds=remaining)
        diagnostics.append(dict(agent=aid, **found))
        if found['status'] != 'path':
            return dict(status='unknown' if found['status'] == 'unknown' else 'error',
                        reason='native_single_agent_' + found['status'], diagnostics=diagnostics)
        overrides[aid] = found['path']
        visible = {i: overrides.get(i, original[i]) for i in fixed + [aid]}
        attempted_edges |= edge_set(visible, [aid])
        if len(attempted_edges) > len(old_edges):
            return dict(status='ok', rolled_back=True, paths=list(original.values()),
                        attempted_pairs=len(attempted_edges), diagnostics=diagnostics)
        fixed.append(aid)
    return dict(status='ok', rolled_back=False, paths=[overrides.get(i, p) for i, p in original.items()],
                attempted_pairs=len(attempted_edges), diagnostics=diagnostics)


def parity(case, state, config, seed, native, probe):
    transition = read_json(contained(case['historical_transition_file']))
    order = transition['metrics']['repair_order']
    if sorted(order) != sorted(transition['action']['agents']):
        raise ValueError('historical order and selected neighborhood differ')
    env = native.LNS2RepairEnv(str(contained(case['map_file'])), str(contained(case['scenario_file'])),
                              len(state['agents']), time_limit=config['job_seconds'] + 10)
    restored = env.reset_paths(paths_of(state), seed=seed)
    if paths_of(restored) != paths_of(state) or restored['num_of_colliding_pairs'] != state['num_of_colliding_pairs']:
        raise ValueError('native source reset mismatch')
    reference = env.step(dict(mode='explicit_neighborhood', agents=transition['action']['agents'],
                             repair_order=order, pp_random_seed=seed,
                             pp_time_limit_seconds=config['job_seconds']/2, collect_pp_diagnostics=True))
    metrics = reference['metrics']
    if not metrics['action_valid'] or metrics['repair_order'] != order:
        raise ValueError('frozen native rejected or changed explicit order')
    if metrics['pp_failure_reason'] == 'time_limit':
        return dict(status='unknown', reason='reference_timeout', reference_metrics=metrics)
    candidate = soft_pp(probe, state, order, seed, config['job_seconds']/2)
    if candidate['status'] != 'ok':
        return candidate
    checks = dict(paths=candidate['paths'] == paths_of(reference['observation']),
                  rollback=candidate['rolled_back'] == metrics['pp_rolled_back'],
                  attempted_pairs=candidate['attempted_pairs'] == metrics['pp_attempt_conflict_pair_count'],
                  attempted_agents=len(candidate['diagnostics']) == metrics['pp_attempted_agent_count'],
                  per_agent_costs=[d['cost'] for d in candidate['diagnostics']] ==
                                 [d['path_cost_after'] for d in metrics['pp_agent_diagnostics']],
                  per_agent_collisions=[d['low_level_collisions'] for d in candidate['diagnostics']] ==
                                       [d['low_level_collision_count'] for d in metrics['pp_agent_diagnostics']])
    return dict(status='pass' if all(checks.values()) else 'mismatch', checks=checks,
                reference_paths=paths_of(reference['observation']), reference_metrics=metrics, probe=candidate)


def hard_pair(state, probe, order, method, seed, config):
    started = time.monotonic()
    external = [a['id'] for a in state['agents'] if a['id'] not in order]
    calls = []
    def plan(aid, fixed, overrides, constraints=(), max_cost=-1):
        budget = min(config['call_seconds'], max(0., config['job_seconds']-(time.monotonic()-started)))
        found = probe.plan(aid, fixed, overrides, True, list(constraints), max_cost, budget)
        calls.append(dict(agent=aid, constraints=list(constraints), **found))
        return found

    def pair(first_constraints=(), cap=-1):
        first = plan(order[0], external, {}, first_constraints, cap)
        if first['status'] != 'path':
            return dict(first=first, second=None)
        second = plan(order[1], external + [order[0]], {order[0]: first['path']})
        return dict(first=first, second=second)

    def solved(attempt):
        return attempt['second'] is not None and attempt['second']['status'] == 'path'

    probe.seed_rng(seed)
    base = pair()
    attempts = [dict(kind='ordinary', result=base)]
    winner = base if solved(base) else None
    constraints, relaxed = [], None
    if winner is None and base['first']['status'] == 'path' and method != 'ordinary':
        if method == 'directed_paths':
            relaxed = plan(order[1], external, {})
            if relaxed['status'] == 'path':
                constraints = directed_constraints(base['first']['path'], relaxed['path'], config['reselections'])
            branches = [(c, seed) for c in constraints]
        elif method == 'random_paths':
            branches = [(None, seed + 1 + i) for i in range(config['reselections'])]
        else:
            raise ValueError('unknown method')
        for constraint, trial_seed in branches:
            if time.monotonic()-started >= config['job_seconds']:
                break
            probe.seed_rng(trial_seed)
            attempt = pair([constraint] if constraint is not None else (), base['first']['cost'])
            attempts.append(dict(kind=method, constraint=constraint, seed=trial_seed, result=attempt))
            if solved(attempt):
                winner = attempt
                break
    result = dict(status='feasible' if winner else 'unknown' if any(c['status'] == 'unknown' for c in calls)
                  or time.monotonic()-started >= config['job_seconds'] else 'not_found',
                  attempts=attempts, constraints=constraints, relaxed_second=relaxed,
                  expanded=sum(c['expanded'] for c in calls), generated=sum(c['generated'] for c in calls),
                  calls=len(calls), elapsed_seconds=time.monotonic()-started)
    if winner:
        result['paths'] = {str(a): winner[k]['path'] for a, k in zip(order, ('first', 'second'))}
        result['witness'] = validate_witness(state, result['paths'])
        result['first_cost_delta'] = winner['first']['cost'] - base['first']['cost']
    return result


def result_file(output, job):
    return output/'results'/(job['job_id'] + '.json')


def check_result(manifest, job, result):
    check_seal(result)
    if result['fingerprint'] != manifest['content_sha256'] or result['job'] != job:
        raise ValueError('result identity mismatch')


def worker(output, job_id):
    m = load(output)
    job = next(j for j in m['jobs'] if j['job_id'] == job_id)
    case = next(c for c in m['cases'] if c['case_id'] == job['case_id'])
    result = dict(schema=SCHEMA, job=job, fingerprint=m['content_sha256'])
    try:
        native, extension = modules(m['config'])
        state = read_json(contained(case['state_file']))
        probe = extension.NativePathProbe(str(contained(case['map_file'])),
                                          str(contained(case['scenario_file'])), paths_of(state))
        if job['kind'] == 'parity':
            value = parity(case, state, m['config'], job['seed'], native, probe)
        else:
            value = hard_pair(state, probe, job['order'], job['method'], job['seed'], m['config'])
            if value['status'] == 'feasible':
                paths = [value['paths'].get(str(a['id']), a['path']) for a in state['agents']]
                env = native.LNS2RepairEnv(str(contained(case['map_file'])),
                      str(contained(case['scenario_file'])), len(paths), time_limit=30)
                actual = env.reset_paths(paths, seed=job['seed'])
                if paths_of(actual) != paths or actual['num_of_colliding_pairs'] != value['witness']['final_conflict_pairs']:
                    raise ValueError('native witness validation failed')
                value['native_witness_verified'] = True
        result.update(value)
    except Exception:
        result.update(status='error', traceback=traceback.format_exc())
    write_json(result_file(output, job), seal(result))


def collect(output, phase, workers=4, resume=False, max_jobs=None):
    m = load(output)
    if not 1 <= workers <= 4 or (max_jobs is not None and max_jobs < 1):
        raise ValueError('invalid worker or job limit')
    if phase == 'hard_pair':
        for job in m['jobs']:
            if job['kind'] == 'parity':
                r = read_json(result_file(output, job))
                check_result(m, job, r)
                if r['status'] != 'pass':
                    raise ValueError('hard-pair gate requires every parity job to pass')
    jobs = [j for j in m['jobs'] if j['kind'] == phase]
    with run_lock(output):
        pending, results, active = [], [], {}
        for job in jobs:
            if result_file(output, job).exists():
                if not resume:
                    raise ValueError('existing results require --resume')
                saved = read_json(result_file(output, job))
                check_result(m, job, saved)
                if saved['status'] in ('error', 'mismatch'):
                    raise ValueError('inspect failed job before resuming')
                results.append(saved)
            else:
                pending.append(job)
        launched, stopping, failed = 0, False, False
        def status(value):
            write_json(output/'run_status.json', dict(status=value, phase=phase,
                fingerprint=m['content_sha256'], completed=len(results), total=len(jobs),
                active=[dict(job_id=jid, pid=v[0].pid) for jid, v in active.items()]))
            write_jsonl(output/(phase + '_manifest.jsonl'), [dict(job_id=r['job']['job_id'],
                status=r['status'], sha256=sha256_file(result_file(output, r['job']))) for r in results])
        try:
            status('running')
            while pending or active:
                stopping |= (output/'STOP').exists() or (max_jobs is not None and launched >= max_jobs)
                while pending and len(active) < workers and not stopping:
                    job = pending.pop(0)
                    log = output/'logs'/(job['job_id']+'.log')
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open('wb') as stream:
                        child = subprocess.Popen([sys.executable, str(ROOT/'scripts/diagnose_native_path_compatibility.py'),
                            '_job', '--output', str(output), '--job-id', job['job_id']],
                            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    active[job['job_id']] = (child, time.monotonic(), job)
                    launched += 1
                    stopping |= max_jobs is not None and launched >= max_jobs
                    status('running')
                for jid, (child, start, job) in list(active.items()):
                    if child.poll() is None and time.monotonic()-start > m['config']['job_seconds']+m['config']['fuse_buffer_seconds']:
                        child.kill()
                        child.wait()
                        if not result_file(output, job).exists():
                            write_json(result_file(output, job), seal(dict(schema=SCHEMA, fingerprint=m['content_sha256'],
                                job=job, status='unknown', reason='external_fuse')))
                    if child.poll() is not None:
                        del active[jid]
                        if child.returncode != 0 or not result_file(output, job).exists():
                            if not result_file(output, job).exists():
                                write_json(result_file(output, job), seal(dict(schema=SCHEMA,
                                    fingerprint=m['content_sha256'], job=job, status='error',
                                    reason='worker_exit', exit_code=child.returncode)))
                            # A recorded external timeout is unknown, not an unexplained crash.
                            if read_json(result_file(output, job)).get('reason') != 'external_fuse':
                                failed = stopping = True
                        saved = read_json(result_file(output, job))
                        check_result(m, job, saved)
                        results.append(saved)
                        if saved['status'] in ('error', 'mismatch') or (phase == 'parity' and saved['status'] != 'pass'):
                            failed = stopping = True
                        print(f"{phase} {len(results)}/{len(jobs)} {job['case_id']} {saved['status']}", flush=True)
                        with (output/'progress.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(job_id=jid, status=saved['status'], time=time.time()))+'\n')
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
            raise RuntimeError('diagnostic gate failed; no automatic retries')


def report(output):
    m = load(output)
    results = []
    for job in m['jobs']:
        if result_file(output, job).exists():
            r = read_json(result_file(output, job))
            check_result(m, job, r)
            results.append(r)
    parity_rows = [r for r in results if r['job']['kind'] == 'parity']
    groups = {}
    for method in ('ordinary', 'random_paths', 'directed_paths'):
        rows = [r for r in results if r['job'].get('method') == method]
        groups[method] = dict(count=len(rows), statuses=dict(Counter(r['status'] for r in rows)),
            expanded=sum(r.get('expanded', 0) for r in rows),
            runtime_seconds=sum(r.get('elapsed_seconds', 0) for r in rows),
            recovered=[dict(case=r['job']['case_id'], order=r['job']['order'],
                first_cost_delta=r.get('first_cost_delta'), globally_feasible=r['witness']['globally_feasible'])
                for r in rows if r['status'] == 'feasible' and not (
                    r['attempts'][0]['result']['second'] and r['attempts'][0]['result']['second']['status'] == 'path')])
    complete = len(results) == len(m['jobs'])
    parity_pass = len(parity_rows) == len([j for j in m['jobs'] if j['kind'] == 'parity']) and all(r['status'] == 'pass' for r in parity_rows)
    value = seal(dict(schema=SCHEMA, fingerprint=m['content_sha256'], complete=complete,
        parity_pass=parity_pass, parity_statuses=dict(Counter(r['status'] for r in parity_rows)), methods=groups,
        result_files={result_file(output, r['job']).relative_to(ROOT).as_posix(): sha256_file(result_file(output, r['job'])) for r in results},
        decision='native_parity_gate_failed' if not parity_pass else 'hard_pair_mechanism_only' if complete else 'incomplete',
        boundary='Posthoc native hard-constraint pair probes; not controller or end-to-end performance evidence.'))
    write_json(output/'report.json', value)
    return {k: v for k, v in value.items() if k != 'result_files'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare', 'dry-run', 'parity', 'diagnose', 'report', '_job'))
    parser.add_argument('--config', default='configs/native_path_compatibility_v1.json')
    parser.add_argument('--output', default='build/initlns-native-path-compatibility-v1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--job-id')
    args = parser.parse_args()
    output = Path(args.output)
    output = output.resolve() if output.is_absolute() else contained(args.output)
    if not output.is_relative_to(ROOT/'build'):
        raise ValueError('diagnostic output must stay in build/')
    if args.mode == 'prepare':
        print(json.dumps(prepare(contained(args.config)), indent=2))
    elif args.mode == 'dry-run':
        m = load(output)
        print(json.dumps(dict(jobs=len(m['jobs']), phases=dict(Counter(j['kind'] for j in m['jobs'])),
            maximum_search_seconds=len(m['jobs'])*m['config']['job_seconds']), indent=2))
    elif args.mode == '_job':
        worker(output, args.job_id)
    elif args.mode in ('parity', 'diagnose'):
        collect(output, 'parity' if args.mode == 'parity' else 'hard_pair', args.workers, args.resume, args.max_jobs)
    else:
        print(json.dumps(report(output), indent=2))
