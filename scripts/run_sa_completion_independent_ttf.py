"""Freeze, generate, admit, serially time and audit twelve new warehouse maps."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import csv
import itertools
import json
import os
from pathlib import Path
import random
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_onpolicy as run
from scripts import run_sa_terminal_efficiency_ttf as previous
from scripts import run_sa_raw_fast_ttf as first
from experiments import sa_completion_independent_timing as rt
from experiments.sa_raw_selection_fast import _bind

CONFIG = 'configs/sa_completion_independent_ttf.json'
CODE = (CONFIG, 'scripts/run_sa_completion_independent_ttf.py',
        'experiments/sa_completion_independent_timing.py',
        'tests/evaluation/test_sa_completion_independent_timing.py',
        'docs/SA_COMPLETION_INDEPENDENT_TTF_PROTOCOL_ZH.md')
require = run.require


def config():
    c = run.read_json(ROOT/CONFIG)
    fixed = dict(schema='lns2.sa_completion_independent_ttf.config.v1',
        output='build/sa-completion-independent-ttf-v1', source='build/sa-terminal-efficiency-ttf-v2',
        maps=12, densities=[.2,.25], bottleneck_ratio=.4, solver_seeds=[271,277], replicas=[0],
        master_seed=2026092801, stream_seed=2026092802, phase='completion-independent-20260928',
        comparison_arms=list(rt.ARMS), workers=1, audit_workers=20, budget_seconds=120.,
        process_fuse_seconds=240., preflight_fuse_seconds=300., audit_fuse_seconds=900.,
        max_decisions=None, execution_node_budget=None, bootstrap=5000, bootstrap_seed=2026092803,
        timing_mode=rt.TIMING_MODE, no_training=True, automatic_promotion=False, replacement_permitted=False)
    require(all(c[k] == v for k,v in fixed.items()), 'frozen independent scope changed')
    return c


def prepare():
    from scripts.run_sa_independent_confirmation import historical_inventory
    c = config()
    out = ROOT/c['output']
    require(not out.exists(), 'already registered; inspect/resume')
    old, old_out = previous.verify()
    require(run.sha256_file(old_out/'report.json') == c['source_report_sha256'], 'source result changed')
    history = historical_inventory(exclude=out)
    rng = random.Random(c['master_seed'])
    masters = [rng.randrange(1, 2**31) for _ in range(c['maps'])]
    require(len(set(masters)) == c['maps'] and not set(masters)&set(history['seeds']), 'master overlap; no redraw')
    models = {}
    for arm in ('parent','completion'):
        j = next(j for j in old['jobs'] if j['comparison_arm'] == arm)
        require(j['model']['sha256'] == c[arm+'_sha256'], 'frozen model changed')
        models[arm] = dict(model=j['model'], iteration=j['iteration'])
    plan = deepcopy(old['jobs'][0]['plan'])
    plan['config'].update(output=c['output'], stream_seed=c['stream_seed'])
    # The 25M/256 values remain frozen feature normalization references, not stop rules.
    require(plan['proposal']['max_decisions'] is None, 'inherited decision cap')
    inputs = dict(old['inputs']) | history['manifests']
    paths = list(CODE) + [c['dataset_base'], 'scripts/run_sa_independent_confirmation.py',
        'scripts/confirm_sa_raw_independent.py']
    paths += [(old_out/n).relative_to(ROOT).as_posix() for n in ('registration.json','report.json')]
    paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'generators').rglob('*.py')]
    for path in paths:
        inputs[path] = run.sha256_file(run.contained_file(ROOT,path,field='confirmation input'))
    body = dict(schema='lns2.sa_completion_independent_ttf.design.v1', config=c, inputs=inputs,
        models=models, source_plan=plan, history=history, map_masters=masters,
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with first.recovery.strict_lock(out,body['binding'],'independent-design'):
        run.once(out/'design.json',run.sealed(body))
    return dict(maps=12,tasks=24,conditions=48,episodes=144,timing_started=False,
                planning_budget_upper_hours=4.8,process_fuse_upper_hours=9.6,
                historical_missing_maps=len(history['missing_map_files']))


def verify_design():
    c = config()
    out = ROOT/c['output']
    r = run.check_seal(run.read_json(out/'design.json'))
    require(r['config'] == c and r['binding'] == run.json_fingerprint(
        {k:v for k,v in r.items() if k not in ('binding','integrity')}), 'design changed')
    for path,sha in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,path,field='confirmation input')) == sha, 'changed input: '+path)
    return r,out


def generate_worker(job):
    from generators.dataset import generate_dataset
    from experiments._common import read_jsonl
    r,index = job['registration'],job['index']
    c = r['config']
    folder = ROOT/c['output']/f'dataset/shards/{index:02d}'
    require(not folder.exists(), 'partial/existing generation; no redraw')
    base = run.read_json(ROOT/c['dataset_base'])
    base.update(master_seed=r['map_masters'][index], map_id_prefix=f'sa_completion_confirm_v1_m{index:02d}',
                tasks_per_map=2, splits={'confirmation':{'layout_counts':{'station_centric':1}}})
    template = deepcopy(base['task_variants'][0])
    base['task_variants'] = []
    for density in c['densities']:
        variant = deepcopy(template)
        variant['name'] = f'bottleneck_d{int(density*100)}'
        variant['task'].update(agent_density=density,required_bottleneck_crossing_ratio=c['bottleneck_ratio'])
        base['task_variants'].append(variant)
    generate_dataset(base,output_override=folder.relative_to(ROOT))
    rows = read_jsonl(folder/'confirmation/manifest.jsonl')
    for row in rows:
        for key in ('map_file','map_metadata_file','task_file','scenario_file','instance_file','legacy_instance_file'):
            row[key] = (folder/'confirmation'/row[key]).relative_to(ROOT).as_posix()
    receipt = run.sealed(dict(binding=r['binding'],rows=rows,
        files={p.relative_to(ROOT).as_posix():run.sha256_file(p) for p in sorted(folder.rglob('*')) if p.is_file()}))
    run.once(folder/'receipt.json',receipt)
    return receipt


def check_isolation(rows, history):
    maps = {r['map_id']:run.sha256_file(ROOT/r['map_file']) for r in rows}
    map_seeds = {r['map_seed'] for r in rows}
    task_seeds = [r['task_seed'] for r in rows]
    require(len(rows)==24 and len(maps)==len(set(maps.values()))==len(map_seeds)==12 and
            len(set(task_seeds))==24 and not map_seeds&set(task_seeds), 'duplicate map/task identity')
    require(not (map_seeds|set(task_seeds))&set(history['seeds']) and
            not set(maps.values())&set(history['map_hashes']), 'historical overlap; no replacement')
    for m in maps:
        group = [r for r in rows if r['map_id']==m]
        require(len(group)==2 and {r['task_variant'] for r in group}=={'bottleneck_d20','bottleneck_d25'}, 'density pairing')
    return maps


def generate():
    from lns2_selector.evaluation.path_quality_preflight import audit_task
    r,out = verify_design()
    with first.recovery.strict_lock(out,r['binding'],'independent-generate'):
        with ProcessPoolExecutor(max_workers=min(20,r['config']['maps'])) as pool:
            shards = list(pool.map(generate_worker,[dict(registration=r,index=i) for i in range(r['config']['maps'])]))
        rows = sorted([row for shard in shards for row in shard['rows']],key=lambda x:x['task_id'])
        maps = check_isolation(rows,r['history'])
        cases = []
        for row in rows:
            checked = audit_task(ROOT/row['map_file'],ROOT/row['scenario_file'],ROOT/row['task_file'],row['agent_count'])
            cases.append(dict(task_id=row['task_id'],map_id=row['map_id'],family='warehouse',
                solver_seeds=r['config']['solver_seeds'],static_audit=checked,
                status='static_ready_runtime_unverified',task_variant=row['task_variant'],
                density=.2 if row['task_variant']=='bottleneck_d20' else .25,
                files={k:row[k] for k in ('map_file','scenario_file','task_file')}))
        run.once(out/'cases.json',run.sealed(dict(binding=r['binding'],cases=cases,map_sha256=maps,
            files={k:v for shard in shards for k,v in shard['files'].items()},
            map_seeds=sorted({row['map_seed'] for row in rows}),task_seeds=sorted({row['task_seed'] for row in rows}))))
    return dict(maps=len(maps),tasks=len(cases),historical_seed_overlap=0,historical_geometry_overlap=0)


def cases_for(r,out):
    proof = run.check_seal(run.read_json(out/'cases.json'))
    require(proof['binding']==r['binding'], 'case binding changed')
    for name,sha in proof['files'].items():
        require(run.sha256_file(ROOT/name)==sha, 'changed generated file: '+name)
    return proof['cases']


def condition_jobs(r,out):
    return [dict(case=case,solver_seed=seed,split='independent_confirmation',
        pair_id=f"{case['task_id']}-s{seed}", replica=0, plan=r['source_plan'],
        phase=r['config']['phase'],output=r['config']['output'],
        job_id=run.json_fingerprint(['completion-independent-reset',case['task_id'],seed])[:24])
        for case in cases_for(r,out) for seed in r['config']['solver_seeds']]


def qualify():
    r,out = verify_design()
    jobs = [dict(j,parent_pid=os.getpid(),timing_binding=r['binding']) for j in condition_jobs(r,out)]
    with first.recovery.strict_lock(out,r['binding'],'independent-reset'):
        require(not (out/'qualification').exists(),'qualification already attempted; inspect')
        rows = first.source.execute(r,out,jobs,rt.qualification_worker,'qualification',20,300.)
        run.once(out/'qualification.complete.json',run.sealed(dict(binding=r['binding'],jobs=len(jobs),
            files={j['job_id']:run.sha256_file(out/'qualification'/(j['job_id']+'.json')) for j in jobs})))
        run.once(out/'qualification.json',run.sealed(dict(binding=r['binding'],rows=rows,
            valid=len(rows),nonzero=sum(not x['feasible'] for x in rows),all_tasks_retained=True)))
    return dict(valid=len(rows),nonzero=sum(not x['feasible'] for x in rows),no_conflict_filter=True)


def schedule(jobs):
    groups = {}
    for j in jobs:
        group = groups.setdefault((j['pair_id'],j['replica']),{})
        require(j['comparison_arm'] not in group, 'duplicate arm')
        group[j['comparison_arm']] = j
    require(len(groups)==48, 'expected 48 conditions')
    orders = list(itertools.permutations(rt.ARMS))
    result = []
    for i,(_,group) in enumerate(sorted(groups.items())):
        require(set(group)==set(rt.ARMS), 'missing arm')
        for field in ('expected_initial','case','solver_seed','phase','replica'):
            require(all(group[a][field]==group['parent'][field] for a in rt.ARMS), 'unpaired '+field)
        require(len({group[a]['plan']['config']['stream_seed'] for a in rt.ARMS})==1, 'unpaired stream')
        for arm in orders[(i//4+i%4)%6]:
            result.append(dict(group[arm],schedule_index=len(result)))
    for arm in rt.ARMS:
        group = [j for j in result if j['comparison_arm']==arm]
        require(sorted(Counter(j['case']['map_id'] for j in group).values())==[4]*12,'map coverage')
        for m in {j['case']['map_id'] for j in group}:
            require({(j['case']['task_variant'],j['solver_seed'],j['replica']) for j in group if j['case']['map_id']==m}
                =={(v,s,0) for v in ('bottleneck_d20','bottleneck_d25') for s in (271,277)}, 'task/seed coverage')
    return result


def register():
    r,out = verify_design()
    conditions = condition_jobs(r,out)
    first.source.check_complete(r,out,'qualification',conditions)
    qs = {j['pair_id']:run.check_seal(run.read_json(out/'qualification'/(j['job_id']+'.json'))) for j in conditions}
    jobs = []
    for j in conditions:
        for arm in rt.ARMS:
            spec = r['models'].get(arm,dict(model=None,iteration=0))
            jobs.append(dict(j,**spec,comparison_arm=arm,arm='official_sa' if arm=='official_sa' else 'trained_actor',
                job_id=run.json_fingerprint(['completion-independent-ttf',j['pair_id'],arm])[:24],
                expected_initial=qs[j['pair_id']]['initial_fingerprint'],
                budget_seconds=120.,timing_mode=rt.TIMING_MODE,max_decisions=None,execution_node_budget=None))
    inputs = dict(r['inputs'])
    case_proof = run.check_seal(run.read_json(out/'cases.json'))
    inputs.update(case_proof['files'])
    for p in [out/'design.json',out/'cases.json',out/'qualification.json',out/'qualification.complete.json']+list((out/'qualification').glob('*.json')):
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    body = dict(schema='lns2.sa_completion_independent_ttf.registration.v1',config=r['config'],inputs=inputs,
        jobs=schedule(jobs),design_binding=r['binding'],runtime_labels=rt.LABELS,
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    body['binding'] = run.json_fingerprint(body)
    with first.recovery.strict_lock(out,body['binding'],'independent-register'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(episodes=len(jobs),workers=1,automatic_promotion=False)


def verify():
    c = config()
    out = ROOT/c['output']
    r = run.check_seal(run.read_json(out/'registration.json'))
    require(r['config']==c and r['binding']==run.json_fingerprint(
        {k:v for k,v in r.items() if k not in ('binding','integrity')}),'registration changed')
    for path,sha in r['inputs'].items():
        require(run.sha256_file(run.contained_file(ROOT,path,field='confirmation input'))==sha,'changed input: '+path)
    require(r['jobs']==schedule(r['jobs']), 'schedule changed')
    return r,out


def phase(name):
    require(name in ('preflight','audit'), 'unknown phase')
    r,out = verify()
    jobs = first.source.jobs_for(r)
    if name=='audit':
        first.source.check_complete(r,out,'collect',jobs)
    with first.recovery.strict_lock(out,r['binding'],'independent-'+name):
        require(not (out/name).exists(),'phase already attempted; inspect')
        rows = first.source.execute(r,out,jobs,rt.preflight_worker if name=='preflight' else rt.audit_worker,
            name,20,r['config']['preflight_fuse_seconds' if name=='preflight' else 'audit_fuse_seconds'])
        run.once(out/(name+'.complete.json'),run.sealed(dict(binding=r['binding'],jobs=len(jobs),
            files={j['job_id']:run.sha256_file(out/name/(j['job_id']+'.json')) for j in jobs})))
    return dict(phase=name,verified=len(rows))


_collect = _bind(first.collect,verify=verify,rt=rt)


def collect(resume=False):
    r,out = verify()
    require(not any((out/'failures').glob('*.json')), 'preserved error requires inspection; no automatic retry')
    for job in r['jobs']:
        path = out/'collect'/(job['job_id']+'.json')
        if path.exists():
            receipt = run.check_seal(run.read_json(path))
            require(receipt['binding']==r['binding'] and receipt['job_id']==job['job_id'] and
                    receipt['status']=='ok', 'invalid collection receipt')
    return _collect(resume)


def report():
    r,out = verify()
    jobs = first.source.jobs_for(r)
    for name in ('collect','audit'):
        first.source.check_complete(r,out,name,jobs)
    rows = [first.source.read_result(r,out,j) for j in jobs]
    for j in jobs:
        proof = run.check_seal(run.read_json(out/'audit'/(j['job_id']+'.json')))
        require(proof['result_sha256']==run.sha256_file(out/'episodes'/j['job_id']/'result.json'),'stale audit')
    variants = {j['case']['task_id']:j['case']['task_variant'] for j in jobs}
    c = r['config']
    result = rt.summarize(rows,c['bootstrap'],c['bootstrap_seed'])
    result.update(schema='lns2.sa_completion_independent_ttf.report.v1',binding=r['binding'],episodes=rows,
        by_density={v:rt.summarize([row for row in rows if variants[row['task_id']]==v],c['bootstrap'],c['bootstrap_seed'])
                    for v in ('bottleneck_d20','bottleneck_d25')})
    run.once(out/'report.json',run.sealed(result))
    fields = ('job_id','pair_id','map_id','task_variant','agents','solver_seed','model','initial_conflicts',
              'success','ttf_seconds','delivery_seconds','decisions','generated','soc','makespan',
              'modeled_completion_1s','final_conflicts','stop')
    by_id = {j['job_id']:j for j in jobs}
    with (out/'episodes.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            j = by_id[row['job_id']]
            record = {k:row[k] for k in fields if k in row}
            record.update(task_variant=variants[row['task_id']],agents=j['case']['static_audit']['agent_count'],
                model=row['comparison_arm'],success=row['success_within_budget'],
                ttf_seconds=row['ttf_seconds'] if row['success_within_budget'] else None,
                soc=row['soc'] if row['success_within_budget'] else None,
                makespan=row['makespan'] if row['success_within_budget'] else None,
                modeled_completion_1s=row['delivery_seconds']+row['makespan'] if row['delivered_within_budget'] else None)
            writer.writerow(record)
    run.write_json(out/'run_status.json',dict(status='complete',completed=len(jobs),total=len(jobs),binding=r['binding']))
    return {k:v for k,v in result.items() if k not in ('episodes','by_map','by_density')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','generate','qualify','register','verify','preflight',
                                       'collect','resume','audit','report','stop','all'))
    args = parser.parse_args()
    if args.phase=='verify':
        r,_ = verify()
        value = dict(inputs=len(r['inputs']),episodes=len(r['jobs']))
    elif args.phase=='stop':
        out = ROOT/config()['output']
        require(out.exists(),'not registered')
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
