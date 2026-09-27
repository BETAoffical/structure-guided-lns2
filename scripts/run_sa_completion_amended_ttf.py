"""Repair failed endpoint assignments, retaining all maps and completed tasks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_sa_completion_independent_ttf as old
from experiments import sa_completion_endpoint_repair as repair
from experiments.sa_raw_selection_fast import _bind

run, first, rt = old.run, old.first, old.rt
require = run.require
CONFIG = 'configs/sa_completion_generation_amendment.json'
CODE = (CONFIG,'scripts/run_sa_completion_amended_ttf.py',
    'experiments/sa_completion_endpoint_repair.py','tests/evaluation/test_sa_completion_endpoint_repair.py',
    'docs/SA_COMPLETION_ENDPOINT_AMENDMENT_ZH.md')


def amendment():
    c = run.read_json(ROOT/CONFIG)
    require(c == dict(schema='lns2.sa_completion_endpoint_amendment.v1',
        source='build/sa-completion-independent-ttf-v1',output='build/sa-completion-independent-ttf-v2',
        failed_shards=[7,10],method=repair.METHOD,matching_attempts=16,replace_maps=False,
        redraw_task_seed=False,change_completed_tasks=False,change_controller=False,
        analysis_excluding_amended_tasks=True,analysis_excluding_amended_maps=True), 'amendment changed')
    return c


def config():
    a = amendment()
    return old.config() | dict(schema='lns2.sa_completion_independent_ttf.config.v2',
        output=a['output'],generation_amendment=a)


def prepare():
    previous, source = old.verify_design()
    c = config()
    out = ROOT/c['output']
    require(not out.exists(), 'already registered; inspect existing amendment')
    require(not (source/'registration.json').exists() and not (source/'qualification').exists(),
        'amendment must precede solver observations')
    missing = [i for i in range(12) if not (source/f'dataset/shards/{i:02d}/receipt.json').exists()]
    require(missing==amendment()['failed_shards'], 'unexpected failed shards')
    body = deepcopy(previous)
    body.pop('binding')
    body.pop('integrity',None)
    body.update(schema='lns2.sa_completion_independent_ttf.design.v2',config=c,
        original_design_binding=previous['binding'],
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    body['source_plan']['config']['output'] = c['output']
    paths = [ROOT/p for p in CODE]+[source/'design.json', source/'run_status.json']
    paths += [p for p in (source/'dataset').rglob('*') if p.is_file()]
    for p in paths:
        body['inputs'][p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    body['binding'] = run.json_fingerprint(body)
    with first.recovery.strict_lock(out,body['binding'],'amendment-prepare'):
        run.once(out/'design.json',run.sealed(body))
    return dict(maps_preserved=12,completed_tasks_preserved=22,tasks_to_repair=2,timing_started=False)


verify_design = _bind(old.verify_design,config=config)


def repair_worker(job):
    from generators.models import MapData
    from generators.config import merge_dicts
    from generators.io import write_instance_bundle
    r,index = job['registration'],job['index']
    source = ROOT/r['config']['generation_amendment']['source']/f'dataset/shards/{index:02d}/confirmation'
    saved = run.read_json(next((source/'maps').glob('*.json')))
    md = MapData(**{k:saved[k] for k in ('map_id','seed','grid','metadata')})
    low_path = source/'instances'/(md.map_id+'__task_0000.json')
    low = run.read_json(low_path)
    rng = random.Random(r['map_masters'][index])
    require(rng.randrange(1,2**31)==md.seed and rng.randrange(1,2**31)==low['seed'], 'seed identity changed')
    seed = rng.randrange(1,2**31)
    base = run.read_json(ROOT/r['config']['dataset_base'])
    tc = merge_dicts(base['task'],base['task_variants'][0]['task'])
    tc.update(agent_density=.25,required_bottleneck_crossing_ratio=.4)
    task = repair.repair_task(md,tc,seed,md.map_id+'__task_0001',low['metadata'])
    target = ROOT/r['config']['output']/f'dataset/repaired/{index:02d}/instances'
    require(not target.exists(), 'existing/partial repaired task; inspect')
    write_instance_bundle(target,md,task)
    rows = []
    for task_index,document,folder in ((0,low,source/'instances'),
        (1,dict(task_id=task.task_id,seed=task.seed,metadata=task.metadata),target)):
        name = document['task_id']
        rows.append(dict(map_id=md.map_id,map_seed=md.seed,task_id=name,task_seed=document['seed'],
            map_file=(source/'maps'/(md.map_id+'.map')).relative_to(ROOT).as_posix(),
            scenario_file=(folder/(name+'.scen')).relative_to(ROOT).as_posix(),
            task_file=(folder/(name+'.json')).relative_to(ROOT).as_posix(),
            task_variant='bottleneck_d20' if task_index==0 else 'bottleneck_d25',
            agent_count=document['metadata']['agent_count'],endpoint_amended=task_index==1))
    proof = run.sealed(dict(binding=r['binding'],rows=rows,method=repair.METHOD,
        files={p.relative_to(ROOT).as_posix():run.sha256_file(p) for p in sorted(target.iterdir()) if p.is_file()},
        matching=task.metadata['endpoint_generation_amendment']))
    run.once(target.parent/'receipt.json',proof)
    return proof


def generate():
    from lns2_selector.evaluation.path_quality_preflight import audit_task
    r,out = verify_design()
    source = ROOT/amendment()['source']
    with first.recovery.strict_lock(out,r['binding'],'amendment-generate'):
        require(not (out/'cases.json').exists(),'already generated')
        shards = []
        pending = []
        for i in range(12):
            if i in amendment()['failed_shards']:
                proof = out/f'dataset/repaired/{i:02d}/receipt.json'
                if proof.exists():
                    shard = run.check_seal(run.read_json(proof))
                    require(shard['binding']==r['binding'],'repaired binding changed')
                    shards.append(shard)
                else:
                    pending.append(dict(registration=r,index=i))
            else:
                shard = run.check_seal(run.read_json(source/f'dataset/shards/{i:02d}/receipt.json'))
                require(shard['binding']==r['original_design_binding'],'old receipt identity')
                shards.append(shard)
        if pending:
            with ProcessPoolExecutor(max_workers=len(pending)) as pool:
                shards += list(pool.map(repair_worker,pending))
        for shard in shards:
            for name,sha in shard['files'].items():
                require(run.sha256_file(ROOT/name)==sha,'changed task data: '+name)
        rows = sorted([row for s in shards for row in s['rows']],key=lambda x:x['task_id'])
        maps = old.check_isolation(rows,r['history'])
        cases = []
        for row in rows:
            checked = audit_task(ROOT/row['map_file'],ROOT/row['scenario_file'],ROOT/row['task_file'],row['agent_count'])
            cases.append(dict(task_id=row['task_id'],map_id=row['map_id'],family='warehouse',
                solver_seeds=r['config']['solver_seeds'],static_audit=checked,
                status='static_ready_runtime_unverified',task_variant=row['task_variant'],
                density=.2 if row['task_variant']=='bottleneck_d20' else .25,
                endpoint_amended=row.get('endpoint_amended',False),
                files={k:row[k] for k in ('map_file','scenario_file','task_file')}))
        require(sum(x['endpoint_amended'] for x in cases)==2,'unexpected amendment scope')
        files = {k:v for shard in shards for k,v in shard['files'].items()}
        for case in cases:
            for path in case['files'].values():
                files[path] = run.sha256_file(ROOT/path)
        run.once(out/'cases.json',run.sealed(dict(binding=r['binding'],cases=cases,map_sha256=maps,files=files,
            map_seeds=sorted({x['map_seed'] for x in rows}),task_seeds=sorted({x['task_seed'] for x in rows}))))
    return dict(maps=12,tasks=24,unchanged_tasks=22,amended_tasks=2,no_solver_calls=True)


condition_jobs = old.condition_jobs
qualify = _bind(old.qualify,verify_design=verify_design)
register = _bind(old.register,verify_design=verify_design)
verify = _bind(old.verify,config=config)
phase = _bind(old.phase,verify=verify)
_collect = _bind(first.collect,verify=verify,rt=rt)
collect = _bind(old.collect,verify=verify,_collect=_collect)
_report = _bind(old.report,verify=verify)


def report():
    r,out = verify()
    if not (out/'report.json').exists():
        _report()
    else:
        for name in ('collect','audit'):
            first.source.check_complete(r,out,name,first.source.jobs_for(r))
    result = run.check_seal(run.read_json(out/'report.json'))
    require(result['binding']==r['binding'],'stale report')
    amended = {j['case']['task_id'] for j in r['jobs'] if j['case']['endpoint_amended']}
    maps = {j['case']['map_id'] for j in r['jobs'] if j['case']['endpoint_amended']}
    c = r['config']
    sensitivity = dict(schema='lns2.sa_completion_endpoint_sensitivity.v1',binding=r['binding'],
        report_sha256=run.sha256_file(out/'report.json'),amended_tasks=sorted(amended),
        excluding_amended_tasks=rt.summarize([x for x in result['episodes'] if x['task_id'] not in amended],c['bootstrap'],c['bootstrap_seed']),
        excluding_amended_maps=rt.summarize([x for x in result['episodes'] if x['map_id'] not in maps],c['bootstrap'],c['bootstrap_seed']))
    run.once(out/'generation_sensitivity.json',run.sealed(sensitivity))
    return {k:v for k,v in result.items() if k not in ('episodes','by_map','by_density','integrity')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','generate','qualify','register','verify','preflight','collect','resume','audit','report','stop','all'))
    args = p.parse_args()
    if args.phase=='verify':
        r,_ = verify()
        value = dict(inputs=len(r['inputs']),episodes=len(r['jobs']))
    elif args.phase=='stop':
        out = ROOT/config()['output']
        require(out.exists(),'unregistered')
        run.write_json(out/'STOP_AFTER_EPISODE',dict(requested=True))
        value = dict(stop_after_current_episode=True)
    elif args.phase in ('preflight','audit'):
        value = phase(args.phase)
    elif args.phase=='resume':
        value = collect(True)
    elif args.phase=='all':
        value = collect()
        if not value.get('paused'):
            phase('audit')
            value = report()
    else:
        value = globals()[args.phase]()
    print(json.dumps(value,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
