"""Auditable, resumable mechanism experiment; never calls native step()."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time
import traceback

from experiments._common import (
    read_json, write_json, write_jsonl, sha256_file as file_sha256,
    atomic_write_text, _native_filesystem_path,
)
from experiments.local_path_search import (
    Budget, LimitReached, at, cbs_pair, graph_of, select_pairs, sequential_method,
    validate_state, validate_witness, witness_paths,
)
from experiments.state_analysis import reconstruct_conflicts

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'lns2.local_path_compatibility.v1'
IMPLEMENTATION = ('experiments/local_path_search.py', 'experiments/local_path_compatibility.py',
                  'scripts/diagnose_local_path_compatibility.py', 'experiments/_common.py',
                  'experiments/state_analysis.py')


def sha256_file(path):
    return file_sha256(_native_filesystem_path(path))


def digest(value):
    value = json.loads(json.dumps(value, allow_nan=False))
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def contained(relative, base=ROOT):
    if (not isinstance(relative, str) or '\\' in relative or Path(relative).is_absolute()
            or PureWindowsPath(relative).drive):
        raise ValueError('expected relative POSIX path')
    target = (base / relative).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError('path escapes root')
    return target


def identity(manifest):
    return digest({k: v for k, v in manifest.items() if k != 'fingerprint'})


def load_manifest(output):
    m = read_json(output / 'manifest.json')
    if m['schema'] != SCHEMA or m['fingerprint'] != identity(m):
        raise ValueError('manifest fingerprint mismatch')
    for file, expected in m['files'].items():
        if sha256_file(contained(file)) != expected:
            raise ValueError('registered input or implementation changed: ' + file)
    return m


@contextmanager
def run_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'run.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), created=time.time()), stream)
    try:
        yield
    finally:
        lock.unlink()


def source_analysis(state, selected):
    graph, agents = validate_state(state)
    active = {i for pair in select_pairs(state) for i in pair}
    result = []
    for aid in sorted(active):
        a = agents[aid]
        frontier, early = {a['start']}, []
        for tick in range(1, 4):
            reachable, blocked = set(), []
            for cell in sorted(frontier):
                for dest in graph[cell]:
                    blockers = []
                    for other, b in agents.items():
                        if other == aid:
                            continue
                        current, previous = at(b['path'], tick), at(b['path'], tick-1)
                        kind = ('vertex' if current == dest else
                                'edge' if current == cell and previous == dest and cell != dest else None)
                        if kind:
                            blockers.append(dict(agent=other, kind=kind, selected=other in selected,
                                                 terminal=tick >= len(b['path'])-1))
                    if blockers:
                        blocked.append(dict(source=cell, destination=dest, blockers=blockers))
                    else:
                        reachable.add(dest)
            early.append(dict(time=tick, frontier_before=sorted(frontier),
                              frontier_after=sorted(reachable), blocked=blocked))
            frontier = reachable
            if not frontier:
                break
        occupied = {b['goal']: other for other, b in agents.items() if other != aid}
        component, stack = {a['goal']}, [a['goal']]
        while stack:
            cell = stack.pop()
            for dest in graph[cell]:
                if dest not in component and dest not in occupied:
                    component.add(dest)
                    stack.append(dest)
        boundary = sorted({v for u in component for v in graph[u] if v in occupied})
        result.append(dict(agent=aid, early_reachability=early,
            final_goal_component_size=len(component),
            goal_boundary=[dict(cell=v, agent=occupied[v], selected=occupied[v] in selected,
                                arrival=len(agents[occupied[v]]['path'])-1) for v in boundary]))
    return result


def previous_reference_check(source):
    sys.path.insert(0, str(source))
    old = importlib.import_module('audit_spatial_path_constraints')
    saved = read_json(source / 'spatial-path-constraints-v1/analysis.json')
    plan = read_json(source / 'conditioned-path-diagnostic-v1/plan.json')
    if len(plan['conditions']) != 16 or len(saved['pair_probes']) != 7:
        raise ValueError('prior counts')
    if saved['script_sha256'] != sha256_file(source / 'audit_spatial_path_constraints.py'):
        raise ValueError('prior script changed')
    for file, expected in saved['input_files'].items():
        if sha256_file(contained(file)) != expected:
            raise ValueError('prior input changed: ' + file)
    for condition, expected in zip(plan['conditions'], saved['conditions']):
        if digest(old.condition_job(condition)) != digest(expected):
            raise ValueError('prior fixed-path result changed')
    cases = {c['case_id']: c for c in plan['cases']}
    for expected in saved['pair_probes']:
        actual = old.constrained_pair(cases[expected['case_id']], expected['order'][0],
                    expected['forbidden'], expected['minimum_first_arrival'], expected['label'])
        if digest(actual) != digest(expected):
            raise ValueError('prior pair result changed')
    return dict(conditions=16, pair_probes=7, passed=True)


def prepare(config_path, output):
    config = read_json(config_path)
    if config['schema'] != SCHEMA or config['methods'] != ['ordinary', 'random_paths', 'directed_paths', 'cbs_pair']:
        raise ValueError('configuration schema or methods')
    if config['pairs_per_state'] != 2 or config['max_reselections'] != 4:
        raise ValueError('unsupported sampling changes')
    if config['seconds_per_pair_method'] <= 0 or config['expanded_per_pair_method'] <= 0:
        raise ValueError('invalid budget')
    source = contained(config['source_directory'])
    inputs = {config['formal_report']: config['formal_report_sha256'],
        (source/'spatial-path-constraints-v1/analysis.json').relative_to(ROOT).as_posix(): config['prior_analysis_sha256'],
        (source/'spatial-path-constraints-v1/native_validation.json').relative_to(ROOT).as_posix(): config['prior_native_validation_sha256']}
    for file, expected in inputs.items():
        if sha256_file(contained(file)) != expected:
            raise ValueError('frozen evidence changed: ' + file)
    prior = previous_reference_check(source)
    package = read_json(source / 'mechanism-preparation-v1/manifest.json')
    cases = package['cases']
    if len(cases) != 6 or sum(c['role'] == 'fast_control' for c in cases) != 2:
        raise ValueError('six-state source cohort changed')
    files = dict(inputs)
    def register(path, expected=None):
        path = path.resolve()
        name = path.relative_to(ROOT).as_posix()
        actual = sha256_file(path)
        if expected is not None and actual != expected:
            raise ValueError('source identity changed: ' + name)
        files[name] = actual
    register(config_path)
    register(source / 'mechanism-preparation-v1/manifest.json')
    for file, sha in package['files'].items():
        register(contained(file), sha)
    previous = read_json(source / 'spatial-path-constraints-v1/analysis.json')
    for file, sha in previous['input_files'].items():
        register(contained(file), sha)
    for path in source.glob('*.py'):
        register(path)
    for name in IMPLEMENTATION:
        register(ROOT / name)
    for folder in ('initlns-closed-loop-controller-v2', 'initlns-closed-loop-policy-v1'):
        for path in (ROOT / 'artifacts' / folder).rglob('*'):
            if path.is_file():
                register(path)
    native_config = read_json(contained(config['pressure_config']))
    register(contained(config['pressure_config']))
    register(contained(native_config['native_file']), native_config['native_sha256'])
    jobs, diagnostics = [], []
    for case in cases:
        for key in ('state_file', 'candidates_file', 'prefix_file', 'historical_transition_file',
                    'map_file', 'scenario_file', 'task_file'):
            register(contained(case[key]))
        register(contained(case['source_trace']), case['source_trace_sha256'])
        state = read_json(contained(case['state_file']))
        validate_state(state)
        candidates = read_json(contained(case['candidates_file']))
        selected = next(c['agents'] for c in candidates if c['candidate_id'] == case['selected_candidate_id'])
        diagnostics.append(dict(case_id=case['case_id'], selected_agents=selected,
                                agents=source_analysis(state, selected)))
        for pair in select_pairs(state):
            for method in config['methods']:
                job = dict(case_id=case['case_id'], pair=pair, method=method)
                job['job_id'] = digest(job)[:20]
                jobs.append(job)
    m = dict(schema=SCHEMA, config=config, cases=cases, jobs=jobs, files=files,
             prior_reference=prior, semantics='hard_fixed_external_paths_not_official_PP',
             source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    m['fingerprint'] = identity(m)
    with run_lock(output):
        if (output/'manifest.json').exists():
            if load_manifest(output)['fingerprint'] != m['fingerprint']:
                raise ValueError('refuse to overwrite a different run')
        else:
            write_json(output/'manifest.json', m)
            write_json(output/'source_diagnostics.json', diagnostics)
            write_json(output/'run_status.json', dict(status='prepared', fingerprint=m['fingerprint']))
    return dry_run(m)


def dry_run(m):
    primary = sum(j['method'] != 'cbs_pair' for j in m['jobs'])
    maximum = len(m['jobs'])
    return dict(states=len(m['cases']), selected_pairs=maximum//4, primary_jobs=primary,
                conditional_joint_jobs=maximum-primary, maximum_jobs=maximum,
                workers=m['config']['workers'], seconds_per_job=m['config']['seconds_per_pair_method'],
                cpu_lower_bound_seconds=0, cpu_lower_bound_note='No meaningful positive bound before running.',
                serial_budget_upper_seconds=maximum*m['config']['seconds_per_pair_method'],
                theoretical_four_worker_budget_seconds=maximum*m['config']['seconds_per_pair_method']/4,
                memory_note='Up to one million low-level expansions per active job; use four workers.',
                disk_note='Paths and diagnostics only; no repeated full-state traces.',
                scope=m['config']['scope'])


def result_file(output, job):
    return output/'results'/f"{job['job_id']}.json"


def check_result(m, job, result):
    if result['fingerprint'] != m['fingerprint'] or result['job'] != job:
        raise ValueError('result identity mismatch')
    if result['result_sha256'] != digest({k:v for k,v in result.items() if k != 'result_sha256'}):
        raise ValueError('result checksum mismatch')
    if result['status'] == 'error':
        raise ValueError('previous job error requires inspection: ' + job['job_id'])
    if result['status'] not in ('feasible', 'not_found', 'infeasible', 'unknown', 'skipped'):
        raise ValueError('invalid job status')
    if result['status'] == 'infeasible' and job['method'] != 'cbs_pair':
        raise ValueError('sequential failure cannot prove joint infeasibility')
    if result['status'] == 'skipped' and job['method'] != 'cbs_pair':
        raise ValueError('only conditional CBS may be skipped')
    witnesses = list(witness_paths(result))
    if (result['status'] == 'feasible') != bool(witnesses):
        raise ValueError('witness/status inconsistency')
    case = next(c for c in m['cases'] if c['case_id'] == job['case_id'])
    state = read_json(contained(case['state_file']))
    for paths in witnesses:
        if set(map(int, paths)) != set(job['pair']):
            raise ValueError('witness pair identity')
        validate_witness(state, paths)


def seal(result):
    result['result_sha256'] = digest(result)
    return result


def run_job(output, job_id):
    m = load_manifest(output)
    job = next(j for j in m['jobs'] if j['job_id'] == job_id)
    case = next(c for c in m['cases'] if c['case_id'] == job['case_id'])
    state = read_json(contained(case['state_file']))
    config = m['config']
    budget = Budget(config['seconds_per_pair_method'], config['expanded_per_pair_method'])
    result = dict(schema=SCHEMA, fingerprint=m['fingerprint'], job=job)
    try:
        if job['method'] == 'cbs_pair':
            prerequisite = [j for j in m['jobs'] if j['case_id'] == job['case_id'] and j['pair'] == job['pair']
                            and j['method'] != 'cbs_pair']
            prior = []
            for previous in prerequisite:
                saved = read_json(result_file(output, previous))
                check_result(m, previous, saved)
                prior.append(saved)
            if any(r['status'] == 'feasible' for r in prior):
                result.update(status='skipped', reason='pair_solved_by_prior_method')
            else:
                result.update(cbs_pair(state, job['pair'], budget))
        else:
            result.update(sequential_method(state, job['pair'], job['method'], budget, config['random_seed']))
        result['witness_checks'] = [validate_witness(state, p) for p in witness_paths(result)]
    except LimitReached as exc:
        result.update(status='unknown', reason=str(exc))
    except Exception:
        result.update(status='error', traceback=traceback.format_exc())
    result.update(expanded=budget.expanded, runtime_seconds=time.monotonic()-budget.started)
    write_json(result_file(output, job), seal(result))


def diagnose(output, workers=4, resume=False, max_jobs=None):
    m = load_manifest(output)
    if not 1 <= workers <= 4:
        raise ValueError('workers must be between one and four')
    fresh = read_json(output/'historical_native.json')
    if fresh['fingerprint'] != m['fingerprint'] or not fresh['passed'] or fresh['count'] != 3:
        raise ValueError('fresh historical native validation required')
    with run_lock(output):
        results, active, completed = [], {}, set()
        if not resume and any((output/'results').glob('*.json')):
            raise ValueError('existing results require --resume')
        for job in m['jobs']:
            if result_file(output, job).exists():
                saved = read_json(result_file(output, job))
                check_result(m, job, saved)
                results.append(saved)
                completed.add(job['job_id'])
        pending = [j for j in m['jobs'] if j['job_id'] not in completed]
        pending.sort(key=lambda j: (j['method'] == 'cbs_pair', j['job_id']))
        launched, stopping, failed = 0, False, False
        def status(value):
            write_json(output/'run_status.json', dict(status=value, fingerprint=m['fingerprint'],
                completed=len(completed), total=len(m['jobs']), active=[dict(job_id=j, pid=v[0].pid) for j,v in active.items()]))
            write_jsonl(output/'results_manifest.jsonl',
                [dict(job_id=r['job']['job_id'], file=result_file(output,r['job']).relative_to(output).as_posix(),
                      sha256=sha256_file(result_file(output,r['job'])), status=r['status']) for r in results])
        try:
            status('running')
            while pending or active:
                stopping = stopping or (output/'STOP').exists() or (max_jobs is not None and launched >= max_jobs)
                while pending and len(active) < workers and not stopping:
                    job = next((j for j in pending if j['method'] != 'cbs_pair' or all(
                        p['job_id'] in completed for p in m['jobs'] if p['case_id'] == j['case_id']
                        and p['pair'] == j['pair'] and p['method'] != 'cbs_pair')), None)
                    if job is None:
                        break
                    pending.remove(job)
                    log = output/'logs'/f"{job['job_id']}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    stream = log.open('wb')
                    command = [sys.executable, str(ROOT/'scripts/diagnose_local_path_compatibility.py'),
                               '_job', '--output', str(output), '--job-id', job['job_id']]
                    try:
                        child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                    finally:
                        stream.close()
                    active[job['job_id']] = (child, time.monotonic(), job)
                    launched += 1
                    stopping = max_jobs is not None and launched >= max_jobs
                    status('running')
                for jid, (child, started, job) in list(active.items()):
                    fuse = m['config']['seconds_per_pair_method'] + m['config']['external_fuse_buffer_seconds']
                    if child.poll() is None and time.monotonic()-started > fuse:
                        child.kill()
                        child.wait()
                        if not result_file(output,job).exists():
                            write_json(result_file(output,job), seal(dict(schema=SCHEMA, fingerprint=m['fingerprint'],
                                job=job,status='unknown', reason='external_fuse', expanded=None,
                                runtime_seconds=time.monotonic()-started)))
                    if child.poll() is not None:
                        del active[jid]
                        if not result_file(output,job).exists():
                            write_json(result_file(output,job), seal(dict(schema=SCHEMA,fingerprint=m['fingerprint'],
                                job=job,status='error',reason='child_exit_without_result', exit_code=child.returncode)))
                        saved = read_json(result_file(output,job))
                        try:
                            check_result(m,job,saved)
                        except Exception:
                            failed = stopping = True
                        results.append(saved)
                        completed.add(jid)
                        print(f"{len(completed)}/{len(m['jobs'])} {job['case_id']} {job['pair']} {job['method']}: {saved['status']}", flush=True)
                        with (output/'progress.jsonl').open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(job_id=jid,status=saved['status'],time=time.time()))+'\n')
                        status('stopping' if stopping else 'running')
                if stopping and not active:
                    break
                try:
                    time.sleep(0.25)
                except KeyboardInterrupt:
                    stopping = True
                    print('Safe stop requested; finishing active jobs.', flush=True)
            status('failed' if failed else 'completed' if len(completed)==len(m['jobs']) else 'paused')
        except BaseException:
            status('interrupted')
            raise
        finally:
            for child, _, _ in active.values():
                if child.poll() is None:
                    child.terminate()
                child.wait()
        if failed:
            raise RuntimeError('diagnostic error; run paused for inspection')


def native_check(m, witnesses):
    config = read_json(contained(m['config']['pressure_config']))
    native = contained(config['native_file'])
    sys.path.insert(0, str(native.parent))
    import lns2_env
    if Path(lns2_env.__file__).resolve() != native or sha256_file(native) != config['native_sha256']:
        raise ValueError('wrong native identity')
    if lns2_env.native_semantics_schema != 'lns2.native_semantics.official_step_timed_extension.v3':
        raise ValueError('native semantics')
    checks = []
    for case, paths, label in witnesses:
        state = read_json(contained(case['state_file']))
        expected = validate_witness(state, paths)
        ordered = [paths.get(str(a['id']), a['path']) for a in state['agents']]
        if [a['id'] for a in state['agents']] != list(range(len(ordered))):
            raise ValueError('native scenario mapping requires ordered contiguous IDs')
        env = lns2_env.LNS2RepairEnv(str(contained(case['map_file'])), str(contained(case['scenario_file'])),
            case['agent_count'], time_limit=120, neighborhood_size=8, destroy_strategy='Adaptive',
            replan_algorithm='PP', use_sipp=True, max_repair_iterations=0, screen=0)
        actual = env.reset_paths(ordered, seed=case['solver_seed'])
        if ([a['path'] for a in actual['agents']] != ordered
                or actual['sum_of_costs'] != expected['sum_of_costs']
                or {tuple(sorted(e)) for e in actual['conflict_edges']} != set(map(tuple,expected['external_conflict_pairs']))
                or bool(actual['feasible']) != expected['globally_feasible']):
            raise ValueError('native path validation mismatch')
        checks.append(dict(label=label,case_id=case['case_id'],paths_sha256=digest(ordered),**expected))
    return dict(passed=True,count=len(checks),checks=checks,native_sha256=config['native_sha256'],
                fingerprint=m['fingerprint'],scope='reset_paths only; no native repair or TTF')


def verify(output, native=False, historical_only=False, repeat=False):
    m = load_manifest(output)
    cases = {c['case_id']:c for c in m['cases']}
    if historical_only:
        if not native:
            raise ValueError('--historical-only requires --native')
        saved = read_json(contained(m['config']['source_directory'])/'spatial-path-constraints-v1/analysis.json')
        witnesses = [(cases[p['case_id']], {str(r['agent']):r['result']['path'] for r in p['plans']}, p['label'])
                     for p in saved['pair_probes'] if p['feasible']]
        result = native_check(m,witnesses)
        if result['count'] != 3:
            raise ValueError('historical witness count')
        write_json(output/'historical_native.json',result)
        return result
    results, witnesses, file_hashes = [], [], {}
    for job in m['jobs']:
        path = result_file(output,job)
        saved = read_json(path)
        check_result(m,job,saved)
        file_hashes[path.relative_to(output).as_posix()] = sha256_file(path)
        results.append(saved)
        for index, paths in enumerate(witness_paths(saved)):
            witnesses.append((cases[job['case_id']],paths,f"{job['job_id']}-{index}"))
        if repeat and job['method'] != 'cbs_pair' and saved['status'] not in ('unknown','skipped'):
            state = read_json(contained(cases[job['case_id']]['state_file']))
            b = Budget(m['config']['seconds_per_pair_method'],m['config']['expanded_per_pair_method'])
            actual = sequential_method(state,job['pair'],job['method'],b,m['config']['random_seed'])
            if any(digest(saved[k]) != digest(v) for k,v in actual.items()):
                raise ValueError('deterministic replay mismatch: '+job['job_id'])
    result = dict(passed=True,jobs=len(results),witnesses=len(witnesses),fingerprint=m['fingerprint'],
                  result_files=file_hashes,deterministic_replay=repeat)
    if native:
        result['native'] = native_check(m,witnesses)
    write_json(output/('native_verification.json' if native else 'verification.json'),result)
    return result


def report(output):
    m = load_manifest(output)
    verification = read_json(output/'native_verification.json')
    if verification['fingerprint'] != m['fingerprint'] or not verification['passed']:
        raise ValueError('native verification missing or stale')
    for file,sha in verification['result_files'].items():
        if sha256_file(contained(file, output)) != sha:
            raise ValueError('verified output changed')
    rows = [read_json(result_file(output,j)) for j in m['jobs']]
    by_method = {method:dict(Counter(r['status'] for r in rows if r['job']['method']==method))
                 for method in m['config']['methods']}
    details = []
    for job in m['jobs']:
        if job['method'] != 'ordinary':
            continue
        matching = {r['job']['method']:r for r in rows if r['job']['case_id']==job['case_id'] and r['job']['pair']==job['pair']}
        item = dict(case_id=job['case_id'],pair=job['pair'],methods={k:v['status'] for k,v in matching.items()})
        item['orders'] = {k:[dict(order=o['order'],status=o['status'], recovery=o.get('recovery'),reason=o.get('reason'),
                                attempts=len(o['attempts'])) for o in v.get('orders',[])] for k,v in matching.items()}
        item['runtime_seconds'] = {k:v['runtime_seconds'] for k,v in matching.items()}
        details.append(item)
    recovered = {}
    for method in ('random_paths','directed_paths'):
        improvements = []
        for item in details:
            original = {tuple(o['order']):o for o in item['orders']['ordinary']}
            for order in item['orders'][method]:
                if order['status']=='feasible' and original[tuple(order['order'])]['status']!='feasible':
                    improvements.append(dict(case_id=item['case_id'],order=order['order']))
        recovered[method] = improvements
    result = dict(schema=SCHEMA,fingerprint=m['fingerprint'],method_statuses=by_method,details=details,
                  same_order_recoveries=recovered,native_witnesses=verification['native']['count'],
                  decision='mechanism_diagnostic_only_no_controller_promotion',
                  conclusions_do_not_estimate_population_rates=True)
    write_json(output/'report.json',result)
    lines = ['# 局部路径兼容性机制诊断','',
        '本轮为六个事后选择状态的机制诊断，不是独立确认、模型训练或端到端 TTF 实验。',
        '参考搜索使用固定外部路径的硬约束，不等同于允许中间冲突的官方 InitLNS PP。','',
        '| 案例 | agent 对 | 普通顺序 | 随机路径 | 定向约束 | 两车 CBS |',
        '|---|---|---|---|---|---|']
    for item in details:
        lines.append('| '+ ' | '.join([item['case_id'],str(item['pair'])]+[item['methods'][k] for k in m['config']['methods']])+' |')
    lines += ['', f"随机路径新增同顺序恢复：{len(recovered['random_paths'])}；定向约束新增同顺序恢复：{len(recovered['directed_paths'])}。",
        f"冻结 native 校验见证 {verification['native']['count']} 组，只调用 reset_paths，不调用修复。",'',
        '## 判读边界','',
        '- feasible：选中两车与彼此及外部路径无冲突；外部之间原有冲突保留。',
        '- not_found：指定顺序或有限重选未找到解，不证明两车联合无解。',
        '- infeasible：仅联合搜索完整穷尽，或单车在忽略伙伴后仍不可行时使用。',
        '- unknown：资源预算耗尽，不自动扩展搜索或邻域。',
        '- skipped：至少一种前置方法已经解决该 agent 对，按计划不运行联合搜索。',
        '- 多进程 runtime 是资源诊断，不能用作串行 TTF 或修复器加速证据。','',
        '## 后续','',
        '只有自动定向反馈显示额外恢复且开销可控，才另立原生实验分支；不得晋级或改写正式结论。',
        '联合搜索单独有效时保留为协调机会。外部停车问题需另做定向补充，不能直接全量扩大邻域。']
    atomic_write_text(output/'REPORT_ZH.md','\n'.join(lines)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','diagnose','verify','report','_job'])
    parser.add_argument('--config',default='configs/local_path_compatibility_v1.json')
    parser.add_argument('--output')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--stop',action='store_true')
    parser.add_argument('--clear-stop',action='store_true')
    parser.add_argument('--max-jobs',type=int)
    parser.add_argument('--native',action='store_true')
    parser.add_argument('--historical-only',action='store_true')
    parser.add_argument('--repeat',action='store_true')
    parser.add_argument('--job-id')
    args = parser.parse_args()
    config_path = contained(args.config)
    output = Path(args.output).resolve() if args.output else contained(read_json(config_path)['output'])
    if not output.is_relative_to((ROOT/'build').resolve()):
        raise ValueError('output must be inside build')
    if args.stop:
        if not (output/'manifest.json').exists():
            raise ValueError('no prepared run')
        atomic_write_text(output/'STOP','safe stop requested\n')
        print('Stop requested; no new jobs will start. Active jobs finish within their budgets.')
        return
    if args.clear_stop:
        if (output/'run.lock').exists():
            raise ValueError('cannot clear stop while collector owns lock')
        (output/'STOP').unlink(missing_ok=True)
    if args.dry_run:
        value = dry_run(load_manifest(output))
    elif args.command == 'prepare':
        value = prepare(config_path, output)
    elif args.command == 'diagnose':
        value = diagnose(output,args.workers,args.resume,args.max_jobs)
    elif args.command == '_job':
        run_job(output,args.job_id)
        return
    elif args.command == 'verify':
        value = verify(output,args.native,args.historical_only,args.repeat)
    else:
        value = report(output)
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)
