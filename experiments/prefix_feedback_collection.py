"""Registered isolated prefix feedback comparison; frozen collectors stay unchanged."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import subprocess
import sys
import time
import traceback

from experiments._common import read_json, write_json, write_jsonl
from experiments.conflict_budget_attribution import reconstruct_ledger, prefix_difference
from experiments.full_recovery_collection import available_memory_gib, check_output as check_old, load as load_old, output_file
from experiments.local_path_compatibility import ROOT, contained, digest, run_lock, sha256_file
from experiments.native_path_compatibility import check_seal, modules, paths_of, seal
from experiments.prefix_budget_feedback import METHODS, evaluate_gate, recover_prefix, result_signature, scientific_signature
from experiments.repair_collection import state_fingerprint

SCHEMA = 'lns2.prefix_budget_feedback.v1'
CONFIG = 'configs/prefix_budget_feedback_v1.json'
CODE = ('experiments/prefix_budget_feedback.py','experiments/prefix_feedback_collection.py',
        'scripts/diagnose_prefix_budget_feedback.py','experiments/conflict_budget_attribution.py')


def checked_output(config):
    output,source = contained(config['output']),contained(config['source_collection'])
    if output==ROOT/'build' or not output.is_relative_to(ROOT/'build'):
        raise ValueError('output must be strictly inside build')
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('output overlaps frozen source')
    return output


def prepare(config_path):
    config = read_json(config_path)
    if config['schema']!=SCHEMA or not 1<=config['workers']<=20:
        raise ValueError('invalid prefix configuration')
    output = checked_output(config)
    old = load_old(contained(config['source_collection']))
    if len(old['cases'])!=24:
        raise ValueError('all original 24 states required')
    for key in ('pp_seeds','job_seconds','call_seconds','max_attempts','max_feedback_blockers','process_fuse_seconds'):
        if config[key]!=old['config'][key]:
            raise ValueError('same-budget comparison changed '+key)
    evidence = read_json(contained(config['source_evidence']))
    source = contained(config['source_collection'])
    for name,expected in evidence['files'].items():
        if sha256_file(source/name)!=expected:
            raise ValueError('frozen evidence changed: '+name)
    report = read_json(source/'report.json')
    check_seal(report)
    if not report['complete'] or not report['parity_passed'] or report['fingerprint']!=old['content_sha256']:
        raise ValueError('complete verified source required')
    source_rows,files = {},dict(old['files'])
    for job in old['jobs']:
        path = output_file(source,job)
        name = path.relative_to(ROOT).as_posix()
        if sha256_file(path)!=report['result_hashes'][name]:
            raise ValueError('source result SHA changed')
        row = read_json(path)
        check_old(old,job,row)
        if job['method']=='parity':
            if row['status']!='pass':
                raise ValueError('source native parity failed')
        elif row['status']!='ok' or row['budget_exhausted']:
            raise ValueError('source is censored or invalid')
        source_rows[(job['case_id'],job['seed'],job['method'])] = name
        files[name] = report['result_hashes'][name]
    jobs,references = [],{}
    for case in old['cases']:
        for seed in config['pp_seeds']:
            for method in METHODS:
                key = dict(case_id=case['case_id'],seed=seed,method=method)
                job = dict(key,job_id=digest(key)[:20])
                jobs.append(job)
                references[job['job_id']] = dict(baseline=source_rows[(case['case_id'],seed,'ordinary')],
                    control=source_rows.get((case['case_id'],seed,method)))
    for name in (*CODE,config_path.relative_to(ROOT).as_posix(),config['source_evidence'],
                 config['source_collection']+'/manifest.json',config['source_collection']+'/report.json'):
        files[name] = sha256_file(contained(name))
    runtime = {**config,**{k:old['config'][k] for k in ('native_file','native_sha256','probe_file','core_file')}}
    manifest = seal(dict(schema=SCHEMA,config=runtime,cases=old['cases'],jobs=jobs,references=references,files=files,
        source_manifest=old['content_sha256'],source_native_parity_passed=48,
        timing_allowed=False,production_controller_modified=False))
    with run_lock(output):
        if (output/'manifest.json').exists() and read_json(output/'manifest.json')!=manifest:
            raise ValueError('configuration or implementation changed; use a new registered output')
        write_json(output/'manifest.json',manifest)
    return dict(output=config['output'],jobs=len(jobs),states=len(old['cases']),workers=config['workers'],
                fingerprint=manifest['content_sha256'],solver_calls=0)


def load(output, job_id=None):
    m = read_json(output/'manifest.json')
    check_seal(m)
    if m['schema']!=SCHEMA or output.resolve()!=checked_output(m['config']):
        raise ValueError('manifest schema or output mismatch')
    names = set(m['files'])
    if job_id is not None:
        job = next(j for j in m['jobs'] if j['job_id']==job_id)
        case = next(c for c in m['cases'] if c['case_id']==job['case_id'])
        # Parent verifies all archival inputs; each child rechecks every code/binary and its actual inputs.
        names = {p for p in names if p.endswith(('.py','.so','.a'))}
        names.update(case[k] for k in ('state_file','historical_transition_file','map_file','scenario_file'))
        names.update(p for p in m['references'][job_id].values() if p)
    for name in sorted(names):
        if sha256_file(contained(name))!=m['files'][name]:
            raise ValueError('registered input changed: '+name)
    return m


def check_result(m, job, row):
    check_seal(row)
    if row['schema']!=SCHEMA or row['fingerprint']!=m['content_sha256'] or row['job']!=job:
        raise ValueError('result identity mismatch')


def worker(output, job_id):
    m = load(output,job_id)
    config = m['config']
    job = next(j for j in m['jobs'] if j['job_id']==job_id)
    case = next(c for c in m['cases'] if c['case_id']==job['case_id'])
    result = dict(schema=SCHEMA,fingerprint=m['content_sha256'],job=job)
    try:
        state = read_json(contained(case['state_file']))
        if state_fingerprint(state)!=case['full_state_fingerprint']:
            raise ValueError('state fingerprint mismatch')
        transition = read_json(contained(case['historical_transition_file']))
        order = transition['metrics']['repair_order']
        if sorted(order)!=sorted(transition['action']['agents']):
            raise ValueError('full original neighborhood changed')
        native,extension = modules(config)
        env = native.LNS2RepairEnv(str(contained(case['map_file'])),str(contained(case['scenario_file'])),
                                  len(state['agents']),time_limit=config['job_seconds']+30)
        restored = env.reset_paths(paths_of(state),seed=job['seed'])
        if paths_of(restored)!=paths_of(state) or restored['num_of_colliding_pairs']!=state['num_of_colliding_pairs']:
            raise ValueError('frozen native reset mismatch')
        probe = extension.NativePathProbe(str(contained(case['map_file'])),str(contained(case['scenario_file'])),paths_of(state))
        value = recover_prefix(probe,state,order,job['seed'],job['method'],config)
        result.update(value)
        if value['budget_exhausted']:
            result.update(status='unknown',reason='search_budget_censored')
        elif value['status']=='ok':
            reference = m['references'][job_id]
            baseline = read_json(contained(reference['baseline']))
            result['baseline_matches_frozen'] = scientific_signature(value['base'])==scientific_signature(baseline['base'])
            if not result['baseline_matches_frozen']:
                raise ValueError('ordinary baseline differs from frozen native-parity evidence')
            if reference['control']:
                control = read_json(contained(reference['control']))
                result['control_matches_frozen'] = result_signature(value)==result_signature(control)
                if not result['control_matches_frozen']:
                    raise ValueError('frozen control replay differs')
            paths = value['final']['paths']
            verified = env.reset_paths(paths,seed=job['seed'])
            if paths_of(verified)!=paths or verified['num_of_colliding_pairs']!=value['path_check']['conflicts']:
                raise ValueError('native final-path check failed')
            result['native_paths_verified'] = True
            base_ledger = reconstruct_ledger(state,order,value['base'])
            result['attempt_ledgers'] = []
            for attempt in value['attempts']:
                ledger = reconstruct_ledger(state,order,attempt['result'])
                result['attempt_ledgers'].append(dict(ledger=ledger,common_prefix_change=prefix_difference(base_ledger,ledger)))
        import resource
        result['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        result.update(status='error',traceback=traceback.format_exc())
    write_json(output_file(output,job),seal(result))


def collect(output, workers=None, resume=False, max_jobs=None):
    m = load(output)
    workers = m['config']['workers'] if workers is None else workers
    if not 1<=workers<=20 or (max_jobs is not None and max_jobs<1):
        raise ValueError('invalid workers or job limit')
    with run_lock(output):
        rows,pending,active,hashes = [],[],{},{}
        for job in m['jobs']:
            path = output_file(output,job)
            if path.exists():
                if not resume:
                    raise ValueError('existing results require resume')
                row = read_json(path)
                check_result(m,job,row)
                if row['status']=='error':
                    raise ValueError('inspect saved error before resume')
                rows.append(row)
                hashes[job['job_id']] = sha256_file(path)
            else:
                pending.append(job)
        started,launched,stopping,failed = time.monotonic(),0,False,False
        def status(label):
            write_json(output/'run_status.json',dict(status=label,completed=len(rows),total=len(m['jobs']),workers=workers,
                fingerprint=m['content_sha256'],available_memory_gib=available_memory_gib(),
                active=[dict(job_id=k,pid=v[0].pid) for k,v in active.items()]))
            write_jsonl(output/'collection_manifest.jsonl',[dict(job_id=r['job']['job_id'],status=r['status'],
                path=output_file(output,r['job']).relative_to(ROOT).as_posix(),sha256=hashes[r['job']['job_id']]) for r in rows])
        try:
            status('running')
            while pending or active:
                stopping |= (output/'STOP').exists() or (max_jobs is not None and launched>=max_jobs)
                stopping |= time.monotonic()-started>m['config']['collection_session_hours']*3600
                while pending and len(active)<workers and not stopping:
                    memory = available_memory_gib()
                    if memory is not None and memory<m['config']['minimum_available_memory_gib']:
                        if not active:
                            stopping=True
                        break
                    job = pending.pop(0)
                    log = output/'logs'/(job['job_id']+'.log')
                    log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('wb') as stream:
                        child = subprocess.Popen([sys.executable,str(ROOT/'scripts/diagnose_prefix_budget_feedback.py'),
                            '_job','--output',str(output),'--job-id',job['job_id']],cwd=ROOT,
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    active[job['job_id']] = (child,time.monotonic(),job)
                    launched+=1
                    stopping |= max_jobs is not None and launched>=max_jobs
                    status('running')
                for key,(child,began,job) in list(active.items()):
                    if child.poll() is None and time.monotonic()-began>m['config']['process_fuse_seconds']:
                        child.kill(); child.wait()
                        if not output_file(output,job).exists():
                            write_json(output_file(output,job),seal(dict(schema=SCHEMA,fingerprint=m['content_sha256'],
                                job=job,status='unknown',reason='external_fuse')))
                    if child.poll() is not None:
                        del active[key]
                        if not output_file(output,job).exists():
                            write_json(output_file(output,job),seal(dict(schema=SCHEMA,fingerprint=m['content_sha256'],
                                job=job,status='error',reason='worker_exit',exit_code=child.returncode)))
                        row = read_json(output_file(output,job))
                        check_result(m,job,row)
                        rows.append(row)
                        hashes[key] = sha256_file(output_file(output,job))
                        failed |= row['status']=='error' or (child.returncode!=0 and row.get('reason')!='external_fuse')
                        stopping |= failed
                        print(f"completed {len(rows)}/{len(m['jobs'])} {job['case_id']} {job['method']} {row['status']} recovered={row.get('recovered')}",flush=True)
                        with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(dict(job_id=key,status=row['status'],time=time.time()))+'\n')
                        status('stopping' if stopping else 'running')
                if stopping and not active:
                    break
                try:
                    time.sleep(.25)
                except KeyboardInterrupt:
                    stopping=True
            status('failed' if failed else 'paused' if pending else 'completed')
        finally:
            for child,_,_ in active.values():
                if child.poll() is None:
                    child.terminate()
                child.wait()
            if active:
                active.clear(); status('interrupted')
        if failed:
            raise RuntimeError('inspect failure; never silently overwrite or enlarge budgets')


def analyze(output):
    m = load(output)
    rows = []
    for job in m['jobs']:
        if output_file(output,job).exists():
            row = read_json(output_file(output,job))
            check_result(m,job,row)
            rows.append(row)
    cases = {c['case_id']:c for c in m['cases']}
    summary = {}
    for method in METHODS:
        group = [r for r in rows if r['job']['method']==method and r['status']=='ok']
        triggered = [r for r in group if r['triggered']]
        summary[method] = dict(conditions=len(group),triggered=len(triggered),added_recovery=sum(r['recovered'] for r in group),
            strict_decrease=sum(r['final']['strict_decrease'] for r in group),
            expanded=sum(r['expanded'] for r in group),generated=sum(r['generated'] for r in group),
            failed_tail_recovery=sum(r['recovered'] for r in group if cases[r['job']['case_id']]['role'] in ('failed_long','failed_long_unchanged')),
            no_feedback=sum(bool(r['feedback']) and not r['feedback']['candidates'] for r in triggered),
            attempts=dict(Counter(a['result']['status'] for r in group for a in r['attempts'])),
            no_harm_to_ordinary=all(r['final']['conflicts']<=r['base']['conflicts'] for r in group))
    by_key = defaultdict(dict)
    for row in rows:
        by_key[(row['job']['case_id'],row['job']['seed'])][row['job']['method']]=row
    comparisons = {}
    for method in ('directed_resources','random_retry'):
        paired = [(g['prefix_resources'],g[method]) for g in by_key.values() if 'prefix_resources' in g and method in g
                  and all(g[x]['status']=='ok' for x in ('prefix_resources',method))]
        comparisons[method] = dict(conditions=len(paired),
            better=sum(a['final']['conflicts']<b['final']['conflicts'] for a,b in paired),
            equal=sum(a['final']['conflicts']==b['final']['conflicts'] for a,b in paired),
            worse=sum(a['final']['conflicts']>b['final']['conflicts'] for a,b in paired))
    gate = evaluate_gate(rows,m['cases'],m['config']['pp_seeds'])
    result = seal(dict(schema=SCHEMA,fingerprint=m['content_sha256'],complete=len(rows)==len(m['jobs']),
        methods=summary,prefix_comparisons=comparisons,gate=gate,
        statuses=dict(Counter(r['status'] for r in rows)),
        decision='mechanism_pass_requires_further_validation' if gate['passed'] else 'mechanism_no_go',
        timing_allowed=False,production_controller_modified=False,
        result_hashes={output_file(output,r['job']).relative_to(ROOT).as_posix():sha256_file(output_file(output,r['job'])) for r in rows}))
    write_json(output/'report.json',result)
    return {k:v for k,v in result.items() if k!='result_hashes'}
