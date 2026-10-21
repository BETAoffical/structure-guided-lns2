"""Prepare five frozen methods and fresh tasks. No solve or timing command exists."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import random
import shutil
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
from scripts import run_sa_frozen_confirmation_ttf as generator
from scripts.run_sa_completion_independent_ttf import check_isolation
from experiments import sa_direct_confirmation as rt
from experiments.sa_direct_selector import DirectActor
from experiments.sa_raw_residual_actor import validate_bundle
from lns2_selector.evaluation.path_quality_preflight import audit_task

CONFIG = 'configs/sa_direct_confirmation.json'
CODE = (CONFIG, 'scripts/prepare_sa_direct_confirmation.py',
        'experiments/sa_direct_confirmation.py',
        'tests/evaluation/test_sa_direct_confirmation.py',
        'docs/SA_DIRECT_CONFIRMATION_PROTOCOL_ZH.md', 'configs/retention_manifest.json')
require = run.require


def config():
    c = run.read_json(ROOT/CONFIG)
    require(c['schema']=='lns2.sa_direct_confirmation.config.v1' and
            c['output']=='build/sa-direct-five-arm-confirmation-v1' and
            c['maps']==12 and c['densities']==[.2,.25] and c['bottleneck_ratio']==.4 and
            c['solver_seeds']==[359,367,373] and c['replicas']==[0] and
            tuple(c['comparison_arms'])==rt.ARMS and c['timed_workers']==1 and
            c['budget_seconds']==120. and c['process_fuse_seconds']==240. and
            c['max_decisions'] is None and c['execution_node_budget'] is None and
            c['execution_authorized'] is False and c['no_training'] is True and
            c['automatic_promotion'] is False, 'preparation scope changed')
    return c


def history_for(c, source):
    # Reuse the retained full historical snapshot; do not rescan the whole build tree.
    history = deepcopy(run.read_json(ROOT/c['history_snapshot'])['history'])
    seeds, hashes = set(history['seeds']), set(history['map_hashes'])
    additions = {}
    for path in sorted((ROOT/'build').glob('*/cases.json')):
        if path.parent == ROOT/c['output']:
            continue
        doc = run.read_json(path)
        if isinstance(doc,dict) and 'map_seeds' in doc and 'task_seeds' in doc:
            seeds.update(doc['map_seeds'])
            seeds.update(doc['task_seeds'])
            hashes.update(doc.get('map_sha256',{}).values())
            additions[path.relative_to(ROOT).as_posix()] = run.sha256_file(path)
    for job in source['training_templates']+source['evaluation_templates']:
        case = job['case']
        hashes.add(run.sha256_file(ROOT/case['files']['map_file']))
        task = run.read_json(ROOT/case['files']['task_file'])
        for key in ('seed','task_seed','map_seed'):
            if type(task.get(key)) is int:
                seeds.add(task[key])
    history.update(seeds=sorted(seeds), map_hashes=sorted(hashes),
                   current_case_registries=additions,
                   scope='Retained historical snapshot + current top-level cases + direct source tasks')
    return history


def prepare():
    c = config()
    out = ROOT/c['output']
    require(not out.exists(), 'existing preparation: verify/resume, never overwrite')
    source = run.read_json(ROOT/c['source_design'])
    plan = deepcopy(source['training_templates'][0]['plan'])
    plan['config'].update(output=c['output'],stream_seed=c['stream_seed'])
    require(plan['proposal']['max_decisions'] is None and
            plan['template']['environment']['max_repair_iterations']==0, 'hidden iteration cap')
    history = history_for(c,source)
    rng = random.Random(c['master_seed'])
    masters = [rng.randrange(1,2**31) for _ in range(c['maps'])]
    require(len(set(masters))==12 and not set(masters)&set(history['seeds']), 'master overlap; no redraw')
    models = {}
    for arm,key in (('direct','direct'),('a','a')):
        path = run.contained_file(ROOT,c[key+'_model'],field='frozen model')
        require(run.sha256_file(path)==c[key+'_sha256'], 'frozen model changed: '+arm)
        bundle = run.read_json(path)
        policy_sha = DirectActor(bundle).sha if arm=='direct' else validate_bundle(bundle)
        if arm=='a':
            require(bundle['iteration']==2,'wrong A version')
        models[arm] = dict(path=(out/'models'/f'{arm}.json').relative_to(ROOT).as_posix(),
                           sha256=c[key+'_sha256'],policy_sha256=policy_sha,source=c[key+'_model'])
    paths = set(CODE)|{c[k] for k in ('source_design','history_snapshot','dataset_base','direct_model','a_model')}
    paths.add(plan['native_file'])
    paths.update(history['current_case_registries'])
    paths.update(str(p.relative_to(ROOT).as_posix()) for folder in
                 ('generators','lns2_selector/runtime','lns2_selector/training')
                 for p in (ROOT/folder).rglob('*.py'))
    paths.update(str(p.relative_to(ROOT).as_posix()) for p in (ROOT/'experiments').glob('sa_*.py'))
    paths.update(('experiments/_common.py','experiments/online_feature_engine.py',
                  'experiments/feature_schema_v2.py','experiments/repair_collection.py',
                  'lns2_selector/evaluation/path_quality_preflight.py',
                  'scripts/run_sa_onpolicy.py','scripts/run_sa_paired_closed_loop.py',
                  'scripts/run_sa_expanded_ttf.py','scripts/run_sa_completion_independent_ttf.py',
                  'scripts/run_sa_frozen_confirmation_ttf.py','scripts/audit_sa_history_information.py'))
    for folder in (plan['template']['frozen_models'],
                   plan['template']['model_registration']['portable_models'],
                   'artifacts/initlns-closed-loop-controller-v2'):
        paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT/folder).rglob('*') if p.is_file())
    inputs = {name:run.sha256_file(run.contained_file(ROOT,name,field='preparation input')) for name in sorted(paths)}
    require(inputs[plan['native_file']]==plan['config']['native_sha256'], 'frozen native changed')
    body = dict(schema='lns2.sa_direct_confirmation.design.v1',config=c,inputs=inputs,
                models=models,source_plan=plan,history=history,map_masters=masters,
                method_labels=rt.LABELS,execution_authorized=False,
                source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    out.joinpath('models').mkdir(parents=True)
    for spec in models.values():
        shutil.copyfile(ROOT/spec['source'],ROOT/spec['path'])
        require(run.sha256_file(ROOT/spec['path'])==spec['sha256'],'model copy mismatch')
    run.once(out/'design.json',run.sealed(body))
    return dict(models_frozen=True,maps=12,tasks=24,conditions=72,planned_jobs=360,timing_started=False)


def verify_design():
    c = config()
    out = ROOT/c['output']
    r = run.check_seal(run.read_json(out/'design.json'))
    require(r['config']==c and r['binding']==run.json_fingerprint(
        {k:v for k,v in r.items() if k not in ('binding','integrity')}),'design changed')
    for name,sha in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,name,field='preparation input'))==sha,'changed input: '+name)
    for spec in r['models'].values():
        require(run.sha256_file(ROOT/spec['path'])==spec['sha256'],'frozen copy changed')
    return r,out


def generate_worker(job):
    # Existing generator has static map/task validation; this adds a BFS/scenario audit.
    receipt = generator.generate_worker(job)
    audits = [audit_task(ROOT/row['map_file'],ROOT/row['scenario_file'],
        ROOT/row['task_file'],row['agent_count']) for row in receipt['rows']]
    return run.sealed({k:v for k,v in receipt.items() if k!='integrity'} | {'static_audits':audits})


def generate():
    r,out = verify_design()
    require(not (out/'cases.json').exists(),'tasks already prepared')
    shards,pending = [],[]
    for i in range(r['config']['maps']):
        path = out/f'dataset/shards/{i:02d}/receipt.json'
        if path.exists():
            shard = run.check_seal(run.read_json(path))
            require(shard['binding']==r['binding'],'stale shard')
            for name,sha in shard['files'].items():
                require(run.sha256_file(ROOT/name)==sha,'changed generated file: '+name)
            shard['static_audits'] = [audit_task(ROOT/row['map_file'],ROOT/row['scenario_file'],
                ROOT/row['task_file'],row['agent_count']) for row in shard['rows']]
            shards.append(shard)
        else:
            pending.append(dict(registration=r,index=i))
    if pending:
        with ProcessPoolExecutor(max_workers=min(r['config']['generation_workers'],len(pending))) as pool:
            futures = {pool.submit(generate_worker,j):j['index'] for j in pending}
            for future in as_completed(futures):
                shards.append(future.result())
                print(json.dumps(dict(map_completed=futures[future],completed=len(shards),
                                      total=12,no_solver_calls=True)),flush=True)
    rows = sorted((row for s in shards for row in s['rows']),key=lambda x:x['task_id'])
    maps = check_isolation(rows,r['history'])
    audits = {row['task_id']:audit for s in shards for row,audit in zip(s['rows'],s['static_audits'])}
    cases = [dict(task_id=row['task_id'],map_id=row['map_id'],family='warehouse',
        agent_count=row['agent_count'],solver_seeds=r['config']['solver_seeds'],
        static_audit=audits[row['task_id']],status='static_ready_runtime_unverified',
        task_variant=row['task_variant'],density=.2 if row['task_variant']=='bottleneck_d20' else .25,
        endpoint_amended=row['endpoint_amended'],sampling_failure=row['sampling_failure'],
        files={k:row[k] for k in ('map_file','scenario_file','task_file')}) for row in rows]
    run.once(out/'cases.json',run.sealed(dict(binding=r['binding'],cases=cases,map_sha256=maps,
        files={k:v for s in shards for k,v in s['files'].items()},
        map_seeds=sorted({row['map_seed'] for row in rows}),task_seeds=sorted({row['task_seed'] for row in rows}))))
    return dict(maps=12,tasks=24,amended_tasks=sum(c['endpoint_amended'] for c in cases),no_solver_calls=True)


def cases_for(r,out):
    proof = run.check_seal(run.read_json(out/'cases.json'))
    require(proof['binding']==r['binding'],'case binding changed')
    for name,sha in proof['files'].items():
        require(run.sha256_file(ROOT/name)==sha,'changed task file: '+name)
    return proof['cases']


def schedule():
    r,out = verify_design()
    cases = cases_for(r,out)
    jobs = rt.make_schedule(cases,r['config'],r['models'],r['source_plan'],r['binding'])
    counts = rt.check_schedule(jobs,r['config'])
    run.once(out/'schedule.json',run.sealed(dict(binding=r['binding'],jobs=jobs,counts=counts,
        execution_authorized=False,status='prepared_not_started')))
    path = out/'schedule.csv'
    require(not path.exists(),'CSV already exists; verify instead')
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(('index','pair_id','task_id','map_id','density','robots','solver_seed',
                         'method','within_pair_position','job_id','execution_authorized'))
        for j in jobs:
            case = j['case']
            writer.writerow((j['schedule_index'],j['pair_id'],case['task_id'],case['map_id'],
                case['density'],case['agent_count'],j['solver_seed'],j['comparison_arm'],
                j['within_pair_position'],j['job_id'],False))
    run.once(out/'status.json',run.sealed(dict(binding=r['binding'],status='prepared_not_started',
        maps=12,tasks=24,conditions=counts['conditions'],planned_jobs=len(jobs),
        no_solver_calls=True,timing_started=False,execution_authorized=False,
        runtime_initial_qualification=False,schedule_csv_sha256=run.sha256_file(path))))
    return dict(conditions=counts['conditions'],planned_jobs=len(jobs),timing_started=False)


def verify():
    r,out = verify_design()
    cases = cases_for(r,out)
    saved = run.check_seal(run.read_json(out/'schedule.json'))
    expected = rt.make_schedule(cases,r['config'],r['models'],r['source_plan'],r['binding'])
    require(saved['binding']==r['binding'] and saved['jobs']==expected and
            saved['execution_authorized'] is False,'schedule changed')
    counts = rt.check_schedule(saved['jobs'],r['config'])
    status = run.check_seal(run.read_json(out/'status.json'))
    require(status['binding']==r['binding'] and status['timing_started'] is False and
            status['execution_authorized'] is False and status['schedule_csv_sha256']==
            run.sha256_file(out/'schedule.csv'),'preparation status changed')
    forbidden = ('episodes','qualification','qualification.json','registration.json','preflight',
                 'results','report.json','ttf.json')
    require(not any((out/name).exists() for name in forbidden),'unexpected execution artifact')
    require(len(cases)==24 and all(c['static_audit']['static_valid'] and
            c['static_audit']['native_reset_performed'] is False for c in cases),'static readiness changed')
    return dict(status='prepared_not_started',maps=12,tasks=24,conditions=counts['conditions'],
        planned_jobs=len(expected),methods=rt.LABELS,amended_tasks=sum(c['endpoint_amended'] for c in cases),
        robot_ranges={str(d):[min(c['agent_count'] for c in cases if c['density']==d),
                             max(c['agent_count'] for c in cases if c['density']==d)] for d in (.2,.25)},
        no_solver_calls=True,timing_started=False,runtime_initial_qualification=False,
        historical_missing_maps=len(r['history']['missing_map_files']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','generate','schedule','verify','all'))
    args = parser.parse_args()
    if args.phase=='all':
        out = ROOT/config()['output']
        if not (out/'design.json').exists():
            print(json.dumps(prepare()),flush=True)
        if not (out/'cases.json').exists():
            print(json.dumps(generate()),flush=True)
        if not (out/'schedule.json').exists():
            print(json.dumps(schedule()),flush=True)
        result = verify()
    else:
        result = globals()[args.phase]()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
