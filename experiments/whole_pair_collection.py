"""Hash-bound independent whole-pair diagnosis using frozen historical controls."""
from __future__ import annotations

from collections import Counter
import json
import subprocess
import sys
import time
import traceback

from experiments._common import read_json, write_json, write_jsonl
from experiments.conflict_budget_attribution import reconstruct_ledger, prefix_difference
from experiments.full_recovery_collection import available_memory_gib, output_file
from experiments.full_neighborhood_recovery import check_paths, scientific_signature
from experiments.local_path_compatibility import ROOT, contained, digest, run_lock, sha256_file
from experiments.native_path_compatibility import seal, check_seal, modules, paths_of
from experiments.prefix_feedback_collection import load as load_source, check_result as check_source, checked_output
from experiments.repair_collection import state_fingerprint
from experiments.whole_pair_feedback import METHODS, recover_pair, options_of, verify_pair_branch

SCHEMA='lns2.whole_pair_feedback.v1'
CODE=('experiments/whole_pair_feedback.py','experiments/whole_pair_collection.py',
      'scripts/diagnose_whole_pair_feedback.py')


def prepare(config_path):
    config=read_json(config_path)
    if config['schema']!=SCHEMA or not 1<=config['workers']<=20:
        raise ValueError('invalid configuration')
    output=checked_output(config)
    source=contained(config['source_collection'])
    old=load_source(source)
    if len(old['cases'])!=24:
        raise ValueError('all frozen 24 cases required')
    for key in ('pp_seeds','job_seconds','call_seconds','max_attempts','max_feedback_blockers','process_fuse_seconds'):
        if config[key]!=old['config'][key]:
            raise ValueError('changed registered budget or seeds: '+key)
    evidence=read_json(contained(config['source_evidence']))
    for name,value in evidence['files'].items():
        if sha256_file(source/name)!=value:
            raise ValueError('frozen source certificate changed: '+name)
    report=read_json(source/'report.json')
    check_seal(report)
    certificate=read_json(source/'verification.json')
    check_seal(certificate)
    if not report['complete'] or not certificate['passed'] or certificate['partial']:
        raise ValueError('complete verified source required')
    if certificate['report_sha256']!=sha256_file(source/'report.json'):
        raise ValueError('source report and verification disagree')
    controls,files=[],dict(old['files'])
    for job in old['jobs']:
        path=output_file(source,job)
        name=path.relative_to(ROOT).as_posix()
        row=read_json(path)
        check_source(old,job,row)
        if row['status']!='ok' or row['budget_exhausted'] or sha256_file(path)!=report['result_hashes'][name]:
            raise ValueError('invalid or altered control')
        controls.append(dict(job=job,path=name))
        files[name]=sha256_file(path)
    jobs=[]
    for case in old['cases']:
        for seed in config['pp_seeds']:
            for method in METHODS:
                key=dict(case_id=case['case_id'],seed=seed,method=method)
                jobs.append(dict(key,job_id=digest(key)[:20]))
    names=(*CODE,config_path.relative_to(ROOT).as_posix(),config['source_evidence'],
           config['source_collection']+'/manifest.json',config['source_collection']+'/report.json',
           config['source_collection']+'/verification.json')
    files.update({p:sha256_file(contained(p)) for p in names})
    runtime={**config,**{k:old['config'][k] for k in ('native_file','native_sha256','probe_file','core_file')}}
    m=seal(dict(schema=SCHEMA,config=runtime,cases=old['cases'],jobs=jobs,controls=controls,files=files,
                source_fingerprint=old['content_sha256'],timing_allowed=False))
    with run_lock(output):
        if (output/'manifest.json').exists() and read_json(output/'manifest.json')!=m:
            raise ValueError('existing registration differs')
        write_json(output/'manifest.json',m)
    return dict(jobs=len(jobs),states=len(m['cases']),controls_reused=len(controls),
                fingerprint=m['content_sha256'],solver_calls=0)


def load(output, job_id=None):
    m=read_json(output/'manifest.json')
    check_seal(m)
    if m['schema']!=SCHEMA or checked_output(m['config'])!=output.resolve():
        raise ValueError('manifest identity mismatch')
    names=set(m['files'])
    if job_id is not None:
        job=next(j for j in m['jobs'] if j['job_id']==job_id)
        case=next(c for c in m['cases'] if c['case_id']==job['case_id'])
        names={p for p in names if p.endswith(('.py','.so','.a'))}
        names.update(case[k] for k in ('state_file','historical_transition_file','map_file','scenario_file'))
        names.update(c['path'] for c in m['controls'] if c['job']['case_id']==job['case_id'] and c['job']['seed']==job['seed'])
    for name in sorted(names):
        if sha256_file(contained(name))!=m['files'][name]:
            raise ValueError('registered input changed: '+name)
    return m


def check_result(m,job,row):
    check_seal(row)
    if row['schema']!=SCHEMA or row['fingerprint']!=m['content_sha256'] or row['job']!=job:
        raise ValueError('result identity mismatch')
    if job not in m['jobs'] or row['status'] not in ('ok','error','unknown'):
        raise ValueError('unregistered job or invalid status')


def baseline(m,job):
    source=next(c for c in m['controls'] if c['job']['case_id']==job['case_id'] and
                c['job']['seed']==job['seed'] and c['job']['method']=='ordinary')
    return read_json(contained(source['path']))


def worker(output,job_id):
    m=load(output,job_id)
    job=next(j for j in m['jobs'] if j['job_id']==job_id)
    case=next(c for c in m['cases'] if c['case_id']==job['case_id'])
    result=dict(schema=SCHEMA,fingerprint=m['content_sha256'],job=job)
    try:
        state=read_json(contained(case['state_file']))
        transition=read_json(contained(case['historical_transition_file']))
        order=transition['metrics']['repair_order']
        if state_fingerprint(state)!=case['full_state_fingerprint'] or sorted(order)!=sorted(transition['action']['agents']):
            raise ValueError('state or original neighborhood changed')
        native,extension=modules(m['config'])
        env=native.LNS2RepairEnv(str(contained(case['map_file'])),str(contained(case['scenario_file'])),
                                len(state['agents']),time_limit=m['config']['job_seconds']+30)
        initial=env.reset_paths(paths_of(state),seed=job['seed'])
        if paths_of(initial)!=paths_of(state) or initial['num_of_colliding_pairs']!=state['num_of_colliding_pairs']:
            raise ValueError('native initial path mismatch')
        probe=extension.NativePathProbe(str(contained(case['map_file'])),str(contained(case['scenario_file'])),paths_of(state))
        result.update(recover_pair(probe,state,order,job['seed'],job['method'],m['config']))
        if result['budget_exhausted']:
            result.update(status='unknown',reason='search_budget_censored')
        elif result['status']=='ok':
            if scientific_signature(result['base'])!=scientific_signature(baseline(m,job)['base']):
                raise ValueError('frozen ordinary baseline mismatch')
            result['baseline_matches_frozen']=True
            final=env.reset_paths(result['final']['paths'],seed=job['seed'])
            if paths_of(final)!=result['final']['paths'] or final['num_of_colliding_pairs']!=result['final']['conflicts']:
                raise ValueError('native final path mismatch')
            result['native_paths_verified']=True
            ledger=reconstruct_ledger(state,order,result['base'])
            result['attempt_ledgers']=[]
            for attempt in result['attempts']:
                branch=attempt['result']
                if branch is None:
                    result['attempt_ledgers'].append(None)
                else:
                    other=reconstruct_ledger(state,order,branch)
                    result['attempt_ledgers'].append(dict(ledger=other,common_prefix_change=prefix_difference(ledger,other)))
        import resource
        result['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        result.update(status='error',traceback=traceback.format_exc())
    write_json(output_file(output,job),seal(result))


def worker_command(output,job):
    return [sys.executable,str(ROOT/'scripts/diagnose_whole_pair_feedback.py'),
            '_job','--output',output.resolve().relative_to(ROOT).as_posix(),'--job-id',job['job_id']]


def collect(output,workers=20,resume=False,max_jobs=None):
    m=load(output)
    if not 1<=workers<=20 or (max_jobs is not None and max_jobs<1):
        raise ValueError('invalid process or smoke limit')
    with run_lock(output):
        rows,hashes,pending,active=[],{},[],{}
        for job in m['jobs']:
            path=output_file(output,job)
            if not path.exists():
                pending.append(job)
                continue
            if not resume:
                raise ValueError('existing results require resume')
            row=read_json(path); check_result(m,job,row)
            if row['status']=='error':
                raise ValueError('inspect saved error; do not overwrite')
            rows.append(row); hashes[job['job_id']]=sha256_file(path)
        launched,stopping,failed=0,False,False
        started=time.monotonic()
        def status(label):
            write_json(output/'run_status.json',dict(status=label,completed=len(rows),total=len(m['jobs']),workers=workers,
                active=[dict(job_id=k,pid=c.pid) for k,(c,_,_) in active.items()],fingerprint=m['content_sha256']))
            write_jsonl(output/'collection_manifest.jsonl',[dict(job_id=r['job']['job_id'],status=r['status'],
                path=output_file(output,r['job']).relative_to(ROOT).as_posix(),sha256=hashes[r['job']['job_id']]) for r in rows])
        try:
            while pending or active:
                stopping |= (output/'STOP').exists() or time.monotonic()-started>m['config']['collection_session_hours']*3600
                stopping |= max_jobs is not None and launched>=max_jobs
                while pending and len(active)<workers and not stopping:
                    memory=available_memory_gib()
                    if memory is not None and memory<m['config']['minimum_available_memory_gib']:
                        if not active: stopping=True
                        break
                    job=pending.pop(0)
                    log=output/'logs'/(job['job_id']+'.log'); log.parent.mkdir(parents=True,exist_ok=True)
                    with log.open('wb') as stream:
                        child=subprocess.Popen(worker_command(output,job),cwd=ROOT,
                            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    active[job['job_id']]=(child,time.monotonic(),job)
                    launched+=1
                    stopping |= max_jobs is not None and launched>=max_jobs
                    status('running')
                for key,(child,began,job) in list(active.items()):
                    if child.poll() is None and time.monotonic()-began>m['config']['process_fuse_seconds']:
                        child.kill(); child.wait()
                        if not output_file(output,job).exists():
                            write_json(output_file(output,job),seal(dict(schema=SCHEMA,fingerprint=m['content_sha256'],
                                job=job,status='unknown',reason='external_fuse')))
                    if child.poll() is None: continue
                    del active[key]
                    if not output_file(output,job).exists():
                        write_json(output_file(output,job),seal(dict(schema=SCHEMA,fingerprint=m['content_sha256'],
                            job=job,status='error',reason='worker_exit',exit_code=child.returncode)))
                    row=read_json(output_file(output,job)); check_result(m,job,row)
                    rows.append(row); hashes[key]=sha256_file(output_file(output,job))
                    failed |= row['status']=='error' or (child.returncode!=0 and row.get('reason')!='external_fuse')
                    stopping |= failed
                    print(f"completed {len(rows)}/{len(m['jobs'])} {job['case_id']} {job['method']} {row['status']} recovered={row.get('recovered')}",flush=True)
                    with (output/'progress.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(dict(job_id=key,status=row['status'],time=time.time()))+'\n')
                    status('stopping' if stopping else 'running')
                if stopping and not active: break
                try: time.sleep(.25)
                except KeyboardInterrupt: stopping=True
            status('failed' if failed else 'paused' if pending else 'completed')
        finally:
            for child,_,_ in active.values():
                if child.poll() is None: child.terminate()
                child.wait()
            if active:
                active.clear(); status('interrupted')
        if failed: raise RuntimeError('saved experiment error; inspect before continuation')


def analyze(output,partial=False):
    m=load(output)
    controls=[read_json(contained(c['path'])) for c in m['controls']]
    rows=[]
    journal=[json.loads(line) for line in (output/'collection_manifest.jsonl').read_text().splitlines()]
    hashes={r['job_id']:r['sha256'] for r in journal}
    if len(hashes)!=len(journal): raise ValueError('duplicate manifest condition')
    for job in m['jobs']:
        path=output_file(output,job)
        if not path.exists():
            if partial: continue
            raise ValueError('incomplete collection')
        row=read_json(path); check_result(m,job,row)
        if hashes.get(job['job_id'])!=sha256_file(path): raise ValueError('result SHA mismatch')
        if row['status']=='ok':
            case=next(c for c in m['cases'] if c['case_id']==job['case_id'])
            state=read_json(contained(case['state_file']))
            order=read_json(contained(case['historical_transition_file']))['metrics']['repair_order']
            if scientific_signature(row['base'])!=scientific_signature(baseline(m,job)['base']):
                raise ValueError('baseline mismatch in saved result')
            if not row['native_paths_verified'] or not row['baseline_matches_frozen']:
                raise ValueError('missing native verification')
            if check_paths(state,row['final']['paths'],order)!=row['path_check']:
                raise ValueError('final path check mismatch')
            if not row['recovered'] and row['final']!=row['base']:
                raise ValueError('non-recovery mutated baseline')
            if row['recovered'] and not row['final']['conflicts']<row['base']['conflicts']:
                raise ValueError('recovery without strict decrease')
            if row['feedback']:
                options=options_of(row['feedback'],order,row['base'],m['config']['max_attempts'])
                if [a['option'] for a in row['attempts']]!=options[:len(row['attempts'])]:
                    raise ValueError('changed reference selection')
            for attempt in row['attempts']:
                if verify_pair_branch(state,order,row['base'],attempt,job['method']=='whole_pair_guarded')!=attempt['pair_check']:
                    raise ValueError('pair verification mismatch')
        rows.append(row)
    if len(rows)!=len(journal): raise ValueError('unexpected collection result')
    combined=controls+rows
    by_method={method:[r for r in combined if r['job']['method']==method and r['status']=='ok']
               for method in ('ordinary','random_retry','directed_resources','prefix_resources',*METHODS)}
    cases={c['case_id']:c for c in m['cases']}
    trigger={cid:{r['job']['seed'] for r in by_method['ordinary'] if r['job']['case_id']==cid and r['triggered']} for cid in cases}
    stable={method:{cid for cid in cases if trigger[cid]==set(m['config']['pp_seeds']) and
                   {r['job']['seed'] for r in group if r['job']['case_id']==cid and r['recovered']}==set(m['config']['pp_seeds'])}
            for method,group in by_method.items()}
    main=stable['whole_pair_guarded']
    old_union=set().union(*(stable[k] for k in ('random_retry','directed_resources','prefix_resources')))
    integrity=len(rows)==len(m['jobs']) and all(r['status']=='ok' and not r['budget_exhausted'] for r in rows)
    checks=dict(complete_valid_uncensored=integrity,three_stable_states=len(main)>=3,
        two_maps=len({cases[c]['map_id'] for c in main})>=2,
        failed_tail=any(cases[c]['role'] in ('failed_long','failed_long_unchanged') for c in main),
        two_exclusive_states=len(main-old_union)>=2,
        not_fewer_than_controls=len(main)>=max(len(stable[k]) for k in ('random_retry','directed_resources','prefix_resources')),
        no_conflict_harm=all(r['final']['conflicts']<=r['base']['conflicts'] for r in rows if r['status']=='ok'))
    summary={method:dict(conditions=len(group),triggered=sum(r['triggered'] for r in group),
        added_recovery=sum(r['recovered'] for r in group),strict_decrease=sum(r['final']['strict_decrease'] for r in group),
        failed_tail_recovery=sum(r['recovered'] for r in group if cases[r['job']['case_id']]['role'] in ('failed_long','failed_long_unchanged')),
        expanded=sum(r['expanded'] for r in group),generated=sum(r['generated'] for r in group),
        attempts=dict(Counter(a['result']['status'] if a['result'] else a['reason'] for r in group for a in r['attempts'])),
        pair_checks=dict(Counter(a['pair_check']['classification'] for r in group for a in r['attempts'] if 'pair_check' in a)))
        for method,group in by_method.items()}
    report=seal(dict(schema=SCHEMA,fingerprint=m['content_sha256'],complete=len(rows)==len(m['jobs']),
        statuses=dict(Counter(r['status'] for r in rows)),methods=summary,gate=dict(passed=all(checks.values()),checks=checks),
        stable_recovered_states={k:sorted(v) for k,v in stable.items()},
        decision='mechanism_pass_requires_independent_validation' if all(checks.values()) else 'mechanism_no_go',
        timing_allowed=False,production_controller_modified=False,controls_reused=len(controls),
        result_hashes={output_file(output,r['job']).relative_to(ROOT).as_posix():hashes[r['job']['job_id']] for r in rows}))
    write_json(output/('partial_report.json' if partial else 'report.json'),report)
    return {k:v for k,v in report.items() if k!='result_hashes'}
