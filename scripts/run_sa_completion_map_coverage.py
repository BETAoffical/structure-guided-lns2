"""One completion-only update from A on fresh maps; node-budget development, not TTF."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sa_completion_continuation as prior
from scripts import run_sa_frozen_confirmation_ttf as dataset
from scripts import run_sa_terminal_efficiency as old
from experiments.sa_raw_selection_fast import _bind

run, raw, require = old.run, old.raw, old.require
CONFIG = 'configs/sa_completion_map_coverage.json'
ARMS = ('parent', 'reference', 'completion')
objective = SimpleNamespace(**(vars(old.objective) | {'OBJECTIVES': ('completion',)}))
parity_objective = SimpleNamespace(OBJECTIVES=('reference', 'completion'))


def configuration():
    cfg = run.read_json(ROOT/CONFIG)
    fixed = dict(maps=18, train_maps=12, development_maps=6, workers=20, max_decisions=None,
        node_budget=25000000, pp_safety_seconds=20., episode_safety_seconds=900., process_fuse_seconds=960.,
        maximum_updates_per_arm=1, train_replicas=4, comparison_replicas=2, objective='completion',
        minimum_mixed_conditions=8, minimum_credited_maps=6, bootstrap=5000, bootstrap_seed=2026093019,
        master_seed=2026093017, stream_seed=2026093018, train_solver_seeds=[317,331],
        development_solver_seeds=[337,347], densities=[.2,.25], bottleneck_ratio=.4,
        phase='completion-map-coverage-train-20260930', comparison_phase='completion-map-coverage-dev-20260930',
        formal_ttf=False, automatic_promotion=False, replacement_permitted=False,
        parent_sha256='81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062',
        reference_sha256='9ba2192c53f02563c02b8f09637d69a5448ccfb7d847f1f88de4f642784d15a9')
    require(all(cfg[k] == v for k,v in fixed.items()), 'fixed map-coverage scope changed')
    return cfg


def prepare():
    from scripts.run_sa_independent_confirmation import historical_inventory
    cfg = configuration()
    out = ROOT/cfg['output']
    require(not out.exists(), 'existing output; inspect/resume')
    source, src = prior.verify()
    require(run.sha256_file(src/'registration.json') == cfg['source_registration_sha256'], 'source registration')
    parent, reference = (run.read_json(ROOT/cfg[k+'_model']) for k in ('parent','reference'))
    require(parent['iteration'] == 2 and reference['iteration'] == 3 and
            reference['parent_policy'] == raw.validate_bundle(parent), 'A/A2 lineage')
    inputs = {}
    def register(path, expected=None):
        path = Path(path)
        sha = run.sha256_file(path)
        require(expected is None or sha == expected, 'changed input: '+str(path))
        inputs[path.relative_to(ROOT).as_posix()] = sha
    for k in ('parent','reference'):
        register(ROOT/cfg[k+'_model'],cfg[k+'_sha256'])
    register(ROOT/cfg['dataset_base'],cfg['dataset_base_sha256'])
    history = historical_inventory(exclude=out)
    for p in sorted((ROOT/'build').glob('*/cases.json')):
        doc = run.read_json(p)
        if isinstance(doc,dict) and 'map_seeds' in doc and 'task_seeds' in doc:
            history['seeds'] = sorted(set(history['seeds']) | set(doc['map_seeds']) | set(doc['task_seeds']))
            history['map_hashes'] = sorted(set(history['map_hashes']) | set(doc.get('map_sha256',{}).values()))
            history['manifests'][p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    rng = random.Random(cfg['master_seed'])
    masters = [rng.randrange(1,2**31) for _ in range(cfg['maps'])]
    require(len(set(masters)) == cfg['maps'] and not set(masters)&set(history['seeds']), 'master overlap; no redraw')
    plan = deepcopy(source['plans']['train'])
    plan['config'].update(output=cfg['output'],stream_seed=cfg['stream_seed'])
    require(plan['template']['environment']['max_repair_iterations'] == 0, 'native iteration cap')
    register(ROOT/plan['native_file'],plan['config']['native_sha256'])
    for path in (ROOT/CONFIG,src/'registration.json',ROOT/'docs/SA_COMPLETION_MAP_COVERAGE_PROTOCOL_ZH.md',
                 ROOT/'tests/evaluation/test_sa_completion_map_coverage.py'):
        register(path)
    for directory in ('scripts','experiments','lns2_selector','generators'):
        for path in (ROOT/directory).rglob('*.py'):
            register(path)
    for path in (ROOT/'artifacts/initlns-closed-loop-controller-v2').glob('*.json'):
        register(path)
    inputs.update(history['manifests'])
    body = dict(schema=cfg['schema']+'.design',config=cfg,inputs=inputs,source_plan=plan,
        parent=parent,reference=reference,history=history,map_masters=masters,
        source_commit=run.subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        no_ttf=True,no_promotion=True,timing_results_used_for_training=False)
    body['binding'] = run.json_fingerprint(body)
    with old.recovery.strict_lock(out,body['binding'],'map-coverage-prepare'):
        run.once(out/'design.json',run.sealed(body))
    return dict(maps=18,train=192,comparison_if_updated=144,workers=20,no_ttf=True)


verify_design = _bind(dataset.old.old.verify_design,config=configuration)
generate_worker = dataset.generate_worker


def split_cases(rows, cfg):
    maps = sorted({row['map_id'] for row in rows})
    require(len(maps) == cfg['maps'], 'map coverage')
    return [dict(row,split='train' if maps.index(row['map_id']) < cfg['train_maps']
                 else 'development_holdout',solver_seeds=cfg['train_solver_seeds']
                 if maps.index(row['map_id']) < cfg['train_maps'] else cfg['development_solver_seeds']) for row in rows]


def check_isolation(rows, history, cfg):
    maps = {r['map_id']:run.sha256_file(ROOT/r['map_file']) for r in rows}
    seeds = {r['map_seed'] for r in rows}
    tasks = [r['task_seed'] for r in rows]
    require(len(rows)==2*cfg['maps'] and len(maps)==len(set(maps.values()))==len(seeds)==cfg['maps']
            and len(set(tasks))==2*cfg['maps'] and not seeds&set(tasks), 'duplicate map/task')
    require(not (seeds|set(tasks))&set(history['seeds']) and
            not set(maps.values())&set(history['map_hashes']), 'historical overlap; no replacement')
    for m in maps:
        group = [r for r in rows if r['map_id']==m]
        require(len(group)==2 and {r['task_variant'] for r in group}=={'bottleneck_d20','bottleneck_d25'}, 'density pairing')
    return maps


def generate():
    from lns2_selector.evaluation.path_quality_preflight import audit_task
    r,out = verify_design()
    with old.recovery.strict_lock(out,r['binding'],'map-coverage-generate'):
        require(not (out/'cases.json').exists(), 'already generated')
        shards,pending = [],[]
        for i in range(r['config']['maps']):
            p = out/f'dataset/shards/{i:02d}/receipt.json'
            if p.exists():
                shard = run.check_seal(run.read_json(p))
                require(shard['binding']==r['binding'], 'stale shard')
                require(all(run.sha256_file(ROOT/n)==s for n,s in shard['files'].items()), 'changed shard')
                shards.append(shard)
            else:
                pending.append(dict(registration=r,index=i))
        if pending:
            with ProcessPoolExecutor(max_workers=20) as pool:
                shards += list(pool.map(generate_worker,pending))
        rows = sorted([x for shard in shards for x in shard['rows']],key=lambda x:x['task_id'])
        maps = check_isolation(rows,r['history'],r['config'])
        cases = []
        for row in split_cases(rows,r['config']):
            checked = audit_task(ROOT/row['map_file'],ROOT/row['scenario_file'],ROOT/row['task_file'],row['agent_count'])
            cases.append(dict(task_id=row['task_id'],map_id=row['map_id'],family='warehouse',split=row['split'],
                solver_seeds=row['solver_seeds'],static_audit=checked,status='static_ready_runtime_unverified',
                task_variant=row['task_variant'],density=.2 if row['task_variant']=='bottleneck_d20' else .25,
                endpoint_amended=row['endpoint_amended'],files={k:row[k] for k in ('map_file','scenario_file','task_file')}))
        run.once(out/'cases.json',run.sealed(dict(binding=r['binding'],cases=cases,map_sha256=maps,
            files={k:v for shard in shards for k,v in shard['files'].items()},
            map_seeds=sorted({r['map_seed'] for r in rows}),task_seeds=sorted({r['task_seed'] for r in rows}))))
    return dict(maps=18,tasks=36,amended_tasks=sum(c['endpoint_amended'] for c in cases),no_solver_calls=True)


def condition_jobs(r,out):
    cases = _bind(dataset.old.old.cases_for)(r,out)
    return [dict(case=c,solver_seed=s,split=c['split'],pair_id=f"{c['task_id']}-s{s}",replica=0,
        plan=r['source_plan'],phase='map-coverage-reset-20260930',output=r['config']['output'],
        job_id=run.json_fingerprint(['map-coverage-reset',c['task_id'],s])[:24]) for c in cases for s in c['solver_seeds']]


qualify = _bind(dataset.old.old.qualify,verify_design=verify_design,condition_jobs=condition_jobs,rt=dataset.rt)


def schedule(conditions, initial, phase, replicas, arms):
    require(len({c['pair_id'] for c in conditions})==len(conditions) and
            set(initial)=={c['pair_id'] for c in conditions}, 'condition coverage')
    return [dict(c,phase=phase,replica=i,comparison_arm=a,arm='trained_actor',expected_initial=initial[c['pair_id']],
        job_id=run.json_fingerprint([phase,c['pair_id'],i,a])[:24]) for c in conditions for i in range(replicas) for a in arms]


def register():
    r,out = verify_design()
    conditions = condition_jobs(r,out)
    dataset.first.source.check_complete(r,out,'qualification',conditions)
    inputs = r['inputs'] | run.check_seal(run.read_json(out/'cases.json'))['files']
    initial = {j['pair_id']:run.check_seal(run.read_json(out/'qualification'/(j['job_id']+'.json')))['initial_fingerprint'] for j in conditions}
    rows = [{k:j[k] for k in ('case','solver_seed','split','pair_id')} for j in conditions]
    train,evaluate = ([j for j in rows if j['split']==s] for s in ('train','development_holdout'))
    require(len(train)==48 and len(evaluate)==24, 'train/development counts')
    prior.validate_separation(train,evaluate,inputs)
    for p in [out/n for n in ('design.json','cases.json','qualification.json','qualification.complete.json')]+list((out/'qualification').glob('*.json')):
        inputs[p.relative_to(ROOT).as_posix()] = run.sha256_file(p)
    body = dict(schema=r['config']['schema'],config=r['config'],inputs=inputs,parent=r['parent'],reference=r['reference'],
        plans=dict(train=r['source_plan'],comparison=r['source_plan']),design_binding=r['binding'],source_commit=r['source_commit'],
        train_jobs=schedule(train,{c['pair_id']:initial[c['pair_id']] for c in train},r['config']['phase'],4,['parent']),
        comparison_jobs=schedule(evaluate,{c['pair_id']:initial[c['pair_id']] for c in evaluate},r['config']['comparison_phase'],2,ARMS),
        no_ttf=True,no_promotion=True,evaluation_role='fresh_map_disjoint_development',timing_results_used_for_training=False)
    body['binding'] = run.json_fingerprint(body)
    with old.recovery.strict_lock(out,body['binding'],'map-coverage-register'):
        run.once(out/'registration.json',run.sealed(body))
    return dict(train=192,comparison=144,binding=body['binding'],no_ttf=True)


def _call(function,*args):
    return _bind(function,ROOT=ROOT,configuration=configuration,verify=verify,model_spec=model_spec,
        jobs_for=jobs_for,audited_rows=audited_rows,objective=objective,ARMS=ARMS)(*args)


def verify():
    return _call(old.verify)


def model_spec(reg,arm):
    if arm=='reference':
        cfg = reg['config']
        return dict(path=cfg['reference_model'],sha256=cfg['reference_sha256'],policy_sha256=raw.validate_bundle(reg['reference']))
    return _call(old.model_spec,reg,arm)


def jobs_for(reg,lane):
    require(lane in ('train','comparison'), 'unknown lane')
    jobs = _call(old.jobs_for,reg,lane)
    for job in jobs:
        job['iteration'] = reg['parent']['iteration']+int(job['comparison_arm']!='parent')
    return jobs


def audited_rows(reg,out,lane):
    return _call(old.audited_rows,reg,out,lane)


def collect(lane,resume=False):
    fn = _bind(old.collect,ROOT=ROOT,configuration=configuration,verify=verify,model_spec=model_spec,
        jobs_for=jobs_for,objective=parity_objective)
    return fn(lane,resume)


def audit(lane):
    return _call(old.audit,lane)


def credit_worker(job):
    row = old.previous.read_result(job)
    steps = [{k:e[k] for k in ('decision','policy_sha256','probabilities','selected_id','behavior_log_probability')}
             for e in run.trace_read(old.previous.folder(job))]
    return dict(row,steps=steps)


def credit_summary(rows,coefficients,cfg):
    groups = defaultdict(list)
    for row in rows:
        require(row['split']=='train', 'heldout credit')
        groups[row['pair_id']].append(row)
    mixed = sorted(p for p,g in groups.items() if any(r['success'] for r in g) and not all(r['success'] for r in g))
    maps = sorted({r['map_id'] for p in mixed for r in groups[p]})
    return dict(episodes=len(rows),conditions=len(groups),successes=sum(r['success'] for r in rows),
        decisions=sum(r['decisions'] for r in rows),mixed_conditions=mixed,credited_maps=maps,
        credited_episodes=sum(c['coefficient']!=0 for c in coefficients),
        all_success_conditions=sum(all(r['success'] for r in g) for g in groups.values()),
        all_failure_conditions=sum(not any(r['success'] for r in g) for g in groups.values()),
        update_allowed=len(mixed)>=cfg['minimum_mixed_conditions'] and len(maps)>=cfg['minimum_credited_maps'])


def credit():
    reg,out = verify()
    jobs = [j for j,_ in audited_rows(reg,out,'train')]
    with old.recovery.strict_lock(out,reg['binding'],'map-coverage-credit'):
        with ProcessPoolExecutor(max_workers=20) as pool:
            rows = list(pool.map(credit_worker,jobs))
        cs = objective.coefficients(rows,objective='completion',policy_sha256=raw.validate_bundle(reg['parent']),
            expected_groups={j['pair_id']:j['case']['map_id'] for j in jobs},replicas=4,node_budget=reg['config']['node_budget'])
        summary = credit_summary(rows,cs,reg['config'])
        run.once(out/'credit.json',run.sealed(dict(binding=reg['binding'],summary=summary,coefficients=cs,
            train_audit_sha256=run.sha256_file(out/'train.audit.json'))))
    return summary


def train():
    import torch
    from experiments.sa_raw_residual_update import guarded_update
    torch.set_num_threads(1)
    reg,out = verify()
    proof = run.check_seal(run.read_json(out/'credit.json'))
    require(proof['binding']==reg['binding'] and proof['train_audit_sha256']==run.sha256_file(out/'train.audit.json'), 'credit identity')
    require(torch.__version__==reg['parent']['base']['training_library'], 'Torch version')
    with old.recovery.strict_lock(out,reg['binding'],'map-coverage-one-update'):
        require(not (out/'update.completion.json').exists() and not (out/'models/completion.json').exists(), 'one update only')
        result = dict(binding=reg['binding'],objective='completion',credit_sha256=run.sha256_file(out/'credit.json'),
            credit_summary=proof['summary'],no_ttf=True,no_promotion=True)
        if not proof['summary']['update_allowed']:
            result.update(updated=False,diagnostic=dict(decision='insufficient_credit_coverage'))
        else:
            jobs = [j for j,_ in audited_rows(reg,out,'train')]
            require(len(jobs)==192 and all(j['split']=='train' for j in jobs), 'Train-only coverage')
            lookup = {c['episode_id']:c for c in proof['coefficients']}
            packs = []
            with ProcessPoolExecutor(max_workers=20) as pool:
                for row,pack in pool.map(old.previous.extract_worker,jobs):
                    if pack is not None:
                        c = lookup[row['episode_id']]
                        packs.append(dict(pack,weight=c['episode_weight'],coefficient=c['coefficient']))
                    print(json.dumps(dict(phase='extract',episode=row['episode_id'],total=192)),flush=True)
            proposed,diagnostic = guarded_update(reg['parent'],packs,reg['binding'])
            result.update(updated=proposed is not None,diagnostic=diagnostic)
            if proposed is not None:
                run.once(out/'models/completion.json',proposed)
                result.update(model_sha256=run.sha256_file(out/'models/completion.json'),policy_sha256=raw.validate_bundle(proposed))
        run.once(out/'update.completion.json',run.sealed(result))
    return result


def parity(platform):
    return _bind(old.parity,ROOT=ROOT,verify=verify,model_spec=model_spec,audited_rows=audited_rows,
                 objective=parity_objective)(platform)


def report():
    reg,out = verify()
    update = run.check_seal(run.read_json(out/'update.completion.json'))
    result = dict(binding=reg['binding'],no_ttf=True,no_promotion=True,updated=update['updated'],
        credit=update['credit_summary'],update_sha256=run.sha256_file(out/'update.completion.json'))
    if not update['updated']:
        result['decision'] = 'no_update_keep_A'
    else:
        rows = [dict(r,comparison_arm=j['comparison_arm']) for j,r in audited_rows(reg,out,'comparison')]
        cfg = reg['config']
        contrasts = {a:objective.contrast(rows,a,'completion',bootstrap=cfg['bootstrap'],seed=cfg['bootstrap_seed'])
                     for a in ('parent','reference')}
        signal = prior.signal(contrasts['parent'])
        result.update(episodes=rows,contrasts=contrasts,development_signal=signal,
            successes={a:sum(r['success'] for r in rows if r['comparison_arm']==a) for a in ARMS},
            by_density={v:{a:objective.contrast([r for r in rows if next(j['case']['task_variant'] for j in reg['comparison_jobs']
                if j['job_id']==r['job_id'])==v],a,'completion',bootstrap=cfg['bootstrap'],seed=cfg['bootstrap_seed'])
                for a in ('parent','reference')} for v in ('bottleneck_d20','bottleneck_d25')},
            decision='development_signal_requires_new_TTF' if signal else 'no_development_gain_keep_A')
    run.once(out/'report.json',run.sealed(result))
    run.write_json(out/'run_status.json',dict(status='complete',binding=reg['binding'],decision=result['decision']))
    return {k:v for k,v in result.items() if k not in ('episodes','by_density')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('prepare','generate','qualify','register','verify','collect','audit','credit','train','parity','compare','report','stop'))
    p.add_argument('--resume',action='store_true')
    p.add_argument('--lane',choices=('train','comparison'),default='train')
    p.add_argument('--platform',choices=('windows','wsl'))
    a = p.parse_args()
    if a.phase=='verify': result = dict(binding=verify()[0]['binding'],verified=True)
    elif a.phase in ('collect','compare'): result = collect('train' if a.phase=='collect' else 'comparison',a.resume)
    elif a.phase=='audit': result = audit(a.lane)
    elif a.phase=='parity':
        require(a.platform is not None, 'platform required')
        result = parity(a.platform)
    elif a.phase=='stop':
        run.write_json(ROOT/configuration()['output']/'STOP_AFTER_BATCH',dict(requested=True))
        result = dict(stop_after_current_batch=True)
    else: result = globals()[a.phase]()
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
